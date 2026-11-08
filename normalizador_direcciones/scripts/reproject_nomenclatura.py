"""Reproject the certified-nomenclature shapefile from EPSG:6249 to EPSG:4326.

Writes ``<shp>_4326.parquet`` (every source field plus ``lon``, ``lat``, ``x_6249``, ``y_6249``,
``crs_source``, ``crs_target``) and ``reprojection_report.json`` next to it.

Gates (exit code 2 when one fails; the parquet is not written in that case):
  * every finite point must fall inside the Cali bounding box;
  * the median haversine distance between the reprojected point and the cadastral centroid of the
    same NPN must be <= 100 m (a warning is printed above 30 m).
Exit code 1: unreadable input (missing shapefile or .prj).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import nomenclatura  # noqa: E402
from cali_address.geo import haversine_m  # noqa: E402

DEFAULT_SHP = os.path.join(
    ROOT, "train_data", "nomenclatura_certificada_20261030", "nomenclatura_certificada_20261030.shp"
)
DEFAULT_DOCS = os.path.join(ROOT, "artifacts", "catastro_docs.parquet")
WARN_SHIFT_M = 30.0
ABORT_SHIFT_M = 100.0


def _bbox_before(df: pd.DataFrame) -> dict:
    x, y = df["x_src"], df["y_src"]
    if not x.notna().any():
        return {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    return {"x_min": float(x.min()), "x_max": float(x.max()), "y_min": float(y.min()), "y_max": float(y.max())}


def _bbox_after(df: pd.DataFrame) -> dict:
    lon, lat = df["lon"], df["lat"]
    if not lon.notna().any():
        return {"lon_min": None, "lon_max": None, "lat_min": None, "lat_max": None}
    return {"lon_min": float(lon.min()), "lon_max": float(lon.max()),
            "lat_min": float(lat.min()), "lat_max": float(lat.max())}


def sanity_shift(df: pd.DataFrame, docs_path: str) -> dict:
    """Distance between each reprojected point and the cadastral centroid of the same NPN."""
    empty = {"shared_npn": 0, "median_shift_m": None, "p90_shift_m": None, "max_shift_m": None}
    if not os.path.isfile(docs_path):
        return {**empty, "skipped": f"catastro docs not found: {docs_path}"}
    docs = pd.read_parquet(docs_path, columns=["numero_predial_nacional", "centroid_lon", "centroid_lat"])
    docs = docs.dropna().drop_duplicates("numero_predial_nacional", keep="first")
    pts = df[["NdNpn", "lon", "lat"]].dropna()
    pts = pts.assign(NdNpn=pts["NdNpn"].astype(str).str.strip())
    pts = pts[pts["NdNpn"] != ""].drop_duplicates("NdNpn", keep="first")
    merged = pts.merge(docs, left_on="NdNpn", right_on="numero_predial_nacional", how="inner")
    if merged.empty:
        return empty
    d = haversine_m(merged["lat"], merged["lon"], merged["centroid_lat"], merged["centroid_lon"])
    return {
        "shared_npn": int(len(merged)),
        "median_shift_m": float(np.median(d)),
        "p90_shift_m": float(np.quantile(d, 0.9)),
        "max_shift_m": float(np.max(d)),
    }


def write_geojson(df: pd.DataFrame, path: str) -> int:
    pts = df.dropna(subset=["lon", "lat"])
    props_cols = [c for c in pts.columns if c not in ("lon", "lat", "x_6249", "y_6249")]
    feats = []
    for rec in pts[props_cols + ["lon", "lat"]].itertuples(index=False):
        d = rec._asdict() if hasattr(rec, "_asdict") else dict(zip(props_cols + ["lon", "lat"], rec))
        lon, lat = float(d.pop("lon")), float(d.pop("lat"))
        props = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [lon, lat]}})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, default=str)
    return len(feats)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shp", default=DEFAULT_SHP)
    ap.add_argument("--out", default=None, help="default: <shp>_4326.parquet")
    ap.add_argument("--catastro-docs", default=DEFAULT_DOCS)
    ap.add_argument("--geojson", default=None, help="optional GeoJSON export of the reprojected points")
    args = ap.parse_args(argv)

    out = args.out or os.path.splitext(args.shp)[0] + "_4326.parquet"
    report_path = os.path.join(os.path.dirname(os.path.abspath(out)), "reprojection_report.json")
    prj = os.path.splitext(args.shp)[0] + ".prj"

    t0 = time.time()
    try:
        transformer = nomenclatura.transformer_from_prj(prj)
        raw = nomenclatura.read_shapefile_points(args.shp)
    except (nomenclatura.CrsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"read {len(raw)} records in {time.time() - t0:.1f}s", flush=True)

    df = nomenclatura.reproject_frame(raw, transformer)
    df["x_6249"] = df["x_src"]
    df["y_6249"] = df["y_src"]
    df = df.drop(columns=["x_src", "y_src"])
    df["crs_source"] = "EPSG:6249"
    df["crs_target"] = "EPSG:4326"
    df_for_stats = df.assign(x_src=df["x_6249"], y_src=df["y_6249"])

    count = int(len(df))
    nan_geometry = int((~(np.isfinite(df["lon"]) & np.isfinite(df["lat"]))).sum())
    outside = int(nomenclatura.outside_bbox_mask(df["lon"], df["lat"]).sum())
    sanity = sanity_shift(df, args.catastro_docs)

    bbox_ok = count > 0 and outside == 0
    median = sanity["median_shift_m"]
    sanity_ok = None if median is None else bool(median <= ABORT_SHIFT_M)
    report = {
        "source": os.path.abspath(args.shp),
        "crs_source": "EPSG:6249",
        "crs_target": "EPSG:4326",
        "count": count,
        "nan_geometry": nan_geometry,
        "outside_bbox": outside,
        "cali_bbox": list(nomenclatura.CALI_BBOX),
        "bbox_before": _bbox_before(df_for_stats),
        "bbox_after": _bbox_after(df),
        "sanity": sanity,
        "thresholds": {"warn_median_shift_m": WARN_SHIFT_M, "abort_median_shift_m": ABORT_SHIFT_M},
        "gates": {"bbox_ok": bool(bbox_ok), "sanity_ok": sanity_ok},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if median is not None and WARN_SHIFT_M < median <= ABORT_SHIFT_M:
        print(f"WARNING: median shift to cadastre centroids is {median:.1f} m (> {WARN_SHIFT_M:.0f} m)", flush=True)
    if not bbox_ok or sanity_ok is False:
        reasons = []
        if count == 0:
            reasons.append("no records")
        if outside:
            reasons.append(f"{outside} point(s) outside the Cali bounding box")
        if sanity_ok is False:
            reasons.append(f"median shift {median:.1f} m > {ABORT_SHIFT_M:.0f} m")
        print("ABORT: gate failed: " + "; ".join(reasons) + f". Report: {report_path}", file=sys.stderr)
        return 2

    tmp = out + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    if args.geojson:
        n = write_geojson(df, args.geojson)
        print(f"WROTE {args.geojson} features={n}", flush=True)
    print(f"WROTE {out} rows={count} nan={nan_geometry} median_shift_m={median}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
