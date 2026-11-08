"""Download the Cali domiciliary nomenclature layer from the IDESC WFS (primary data source).

Pages (``sortBy=gid``, ``count``, ``startIndex``) are cached one per file under
``artifacts/nomenclatura_parts/`` so a crash resumes where it stopped; the final merge goes to
``artifacts/nomenclatura_wfs.parquet`` (all 40 properties plus ``lon`` / ``lat`` from the geometry)
and ``artifacts/nomenclatura_meta.json`` records what the server said and what was received.

Exit codes:
  0  downloaded (and, with --validate-shp, validated)
  1  the service failed after the allowed retries, or an input file is unreadable
  2  the downloaded row count does not equal numberMatched (or the layer is empty)
  3  the comparison against the shapefile snapshot is out of tolerance
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import nomenclatura  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
SERVICE = "https://ws-idesc.cali.gov.co/geoserver/wfs"
TYPE_NAMES = "dapm:pdt_nmc_nomenclatura_domiciliaria"
SRS_NAME = "EPSG:4326"
PAGE = 5000
RETRYABLE_STATUS = {408, 429}


class DownloadError(RuntimeError):
    """A page could not be obtained; the message is safe to show to the user."""


def _params(start: int, count: int) -> dict:
    return {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": TYPE_NAMES,
        "outputFormat": "application/json",
        "srsName": SRS_NAME,
        "sortBy": "gid",
        "count": str(count),
        "startIndex": str(start),
    }


def fetch_page(session, base_url: str, start: int, count: int, *, timeout: float, retries: int,
               backoff: float) -> tuple[dict, int]:
    """GET one page and validate it; returns ``(payload, retries_used)``.

    5xx, 408/429, timeouts, connection errors, non-JSON bodies (e.g. an HTML error page served with
    HTTP 200) and JSON without a ``features`` list are retried with exponential backoff. Any other
    4xx fails immediately. After ``retries`` attempts a :class:`DownloadError` is raised.
    """
    last = "no attempt made"
    for attempt in range(max(1, retries)):
        if attempt:
            time.sleep(min(60.0, backoff * 2 ** (attempt - 1)))
        try:
            r = session.get(base_url, params=_params(start, count), timeout=timeout)
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
        if r.status_code >= 500 or r.status_code in RETRYABLE_STATUS:
            last = f"HTTP {r.status_code}"
            continue
        if r.status_code >= 400:
            raise DownloadError(f"page startIndex={start}: HTTP {r.status_code} (not retried)")
        try:
            data = r.json()
        except ValueError as exc:
            last = f"response is not valid JSON ({exc})"
            continue
        if not isinstance(data, dict) or not isinstance(data.get("features"), list):
            last = "response has no 'features' list"
            continue
        return data, attempt
    raise DownloadError(f"page startIndex={start} failed after {max(1, retries)} attempt(s): {last}")


def features_to_frame(features: list[dict]) -> tuple[pd.DataFrame, set[str]]:
    """Features -> DataFrame with the 40 expected columns (+ unexpected extras) and ``lon`` / ``lat``.

    Also returns the set of property names seen. Null properties stay null; a missing or non-Point
    geometry gives NaN coordinates. Never raises for one malformed feature.
    """
    rows, seen = [], set()
    for f in features:
        props = f.get("properties") if isinstance(f, dict) else None
        props = props if isinstance(props, dict) else {}
        seen.update(props)
        lon = lat = np.nan
        geom = f.get("geometry") if isinstance(f, dict) else None
        if isinstance(geom, dict) and geom.get("type") == "Point":
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    lon, lat = float(coords[0]), float(coords[1])
                except (TypeError, ValueError):
                    lon = lat = np.nan
        rows.append({**props, "lon": lon, "lat": lat})
    extras = sorted(seen - set(nomenclatura.WFS_FIELDS))
    columns = list(nomenclatura.WFS_FIELDS) + extras + ["lon", "lat"]
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    for col in nomenclatura.WFS_INT_FIELDS:
        num = pd.to_numeric(df[col], errors="coerce")
        try:
            df[col] = num.astype("Int64")
        except (TypeError, ValueError):
            df[col] = num.astype("float64")
    for col in nomenclatura.WFS_FLOAT_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in df.columns:
        if col not in nomenclatura.WFS_INT_FIELDS and col not in nomenclatura.WFS_FLOAT_FIELDS \
                and col not in ("lon", "lat"):
            df[col] = df[col].astype("object")
    df["lon"] = df["lon"].astype("float64")
    df["lat"] = df["lat"].astype("float64")
    return df, seen


def _part_paths(parts_dir: str, start: int) -> tuple[str, str]:
    stem = os.path.join(parts_dir, f"part_{start:07d}")
    return stem + ".parquet", stem + ".json"


def load_cached_part(parts_dir: str, start: int, count: int):
    """A cached full page as ``(frame, sidecar)`` or ``None`` when absent, short or unreadable."""
    pq, side = _part_paths(parts_dir, start)
    if not (os.path.isfile(pq) and os.path.isfile(side)):
        return None
    try:
        df = pd.read_parquet(pq)
        with open(side, "r", encoding="utf-8") as fh:
            info = json.load(fh)
    except Exception:  # noqa: BLE001 - any corruption just means "fetch it again"
        return None
    if len(df) != count or not isinstance(info, dict):
        return None
    return df, info


def save_part(parts_dir: str, start: int, df: pd.DataFrame, info: dict) -> None:
    pq, side = _part_paths(parts_dir, start)
    with open(side + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    os.replace(side + ".tmp", side)
    df.to_parquet(pq + ".tmp", index=False)
    os.replace(pq + ".tmp", pq)  # the parquet is the completion marker


def download(base_url: str, parts_dir: str, *, count: int, timeout: float, retries: int, backoff: float,
             log=print) -> dict:
    """Walk the pages until a short one; returns frames, matched values and bookkeeping."""
    os.makedirs(parts_dir, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "cali-address-normalizer/1.0"
    frames, matched_seen, seen_props = [], [], set()
    retries_total, fetched_pages, start = 0, 0, 0
    t0 = time.time()
    while True:
        cached = load_cached_part(parts_dir, start, count)
        if cached is not None:
            df, info = cached
            matched = info.get("number_matched")
            seen_props.update(info.get("properties", []))
            returned = len(df)
            log(f"startIndex={start} cached rows={returned}")
        else:
            data, used = fetch_page(session, base_url, start, count, timeout=timeout, retries=retries,
                                    backoff=backoff)
            retries_total += used
            fetched_pages += 1
            matched = data.get("numberMatched")
            matched = int(matched) if isinstance(matched, (int, float)) and not isinstance(matched, bool) else None
            df, props = features_to_frame(data["features"])
            seen_props.update(props)
            returned = len(df)
            if returned:
                save_part(parts_dir, start, df, {"number_matched": matched, "properties": sorted(props)})
            log(f"startIndex={start} rows={returned} numberMatched={matched} [{time.time() - t0:.0f}s]")
        matched_seen.append(matched)
        if returned:
            frames.append(df)
        if returned < count:
            break
        start += count
    return {"frames": frames, "matched_seen": matched_seen, "properties": seen_props,
            "retries": retries_total, "pages_fetched": fetched_pages, "pages": len(matched_seen),
            "seconds": time.time() - t0}


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _optional_columns(path: str, wanted: list[str]) -> list[str]:
    import pyarrow.parquet as pq

    present = set(pq.read_schema(path).names)
    return [c for c in wanted if c in present]


def validate_against_shapefile(wfs_path: str, shp_path: str, vs_json: str, log=print) -> int:
    if not os.path.isfile(shp_path):
        print(f"ERROR: shapefile snapshot not found: {shp_path}", file=sys.stderr)
        return 1
    try:
        shp_cols = ["NdID", "NdEstado", "NdNpn", "NdDirecc", "lon", "lat"]
        shp = pd.read_parquet(shp_path, columns=shp_cols + _optional_columns(shp_path, ["x_6249", "y_6249"]))
        wfs_cols = ["ndid", "ndidestado", "ndnpn", "nddirecc", "lon", "lat"]
        wfs = pd.read_parquet(wfs_path, columns=wfs_cols + _optional_columns(wfs_path, ["ndcoordx", "ndcoordy"]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot read inputs for the validation: {exc}", file=sys.stderr)
        return 1
    report = nomenclatura.compare_wfs_shapefile(wfs, shp)
    report["wfs_file"] = os.path.abspath(wfs_path)
    report["shapefile_file"] = os.path.abspath(shp_path)
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(vs_json, report)
    log(f"WROTE {vs_json} passed={report['passed']}")
    if not report["passed"]:
        failed = [k for k in ("cert_count_ok", "npn_ok", "address_ok", "coord_ok") if not report[k]]
        print(f"VALIDATION FAILED: {', '.join(failed)}. See {vs_json}", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=SERVICE)
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "nomenclatura_wfs.parquet"))
    ap.add_argument("--parts-dir", default=os.path.join(ARTIFACTS, "nomenclatura_parts"))
    ap.add_argument("--meta", default=os.path.join(ARTIFACTS, "nomenclatura_meta.json"))
    ap.add_argument("--count", type=int, default=PAGE)
    ap.add_argument("--retries", type=int, default=6, help="maximum attempts per page")
    ap.add_argument("--backoff", type=float, default=2.0, help="base of the exponential backoff, seconds")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--validate-shp", default=None, help="reprojected shapefile parquet to compare against")
    ap.add_argument("--vs-json", default=os.path.join(ARTIFACTS, "nomenclatura_wfs_vs_shp.json"))
    ap.add_argument("--skip-download", action="store_true", help="only run the validation on --out")
    args = ap.parse_args(argv)

    if args.validate_shp and not os.path.isfile(args.validate_shp):
        print(f"ERROR: shapefile snapshot not found: {args.validate_shp}", file=sys.stderr)
        return 1
    if args.skip_download:
        if not args.validate_shp:
            print("ERROR: --skip-download needs --validate-shp", file=sys.stderr)
            return 1
        return validate_against_shapefile(args.out, args.validate_shp, args.vs_json)

    meta = {
        "source": args.base_url, "type_names": TYPE_NAMES, "srs_name": SRS_NAME, "page_size": args.count,
        "status": "started", "warnings": [],
    }
    try:
        res = download(args.base_url, args.parts_dir, count=args.count, timeout=args.timeout,
                       retries=args.retries, backoff=args.backoff, log=lambda m: print(m, flush=True))
    except DownloadError as exc:
        meta.update(status="failed", error=str(exc), downloaded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        _write_json(args.meta, meta)
        print(f"ERROR: {exc}. Cached pages are kept; re-run the command to resume.", file=sys.stderr)
        return 1

    frames = res["frames"]
    df = pd.concat(frames, ignore_index=True) if frames else features_to_frame([])[0]
    n_before = len(df)
    has_gid = df["gid"].notna()
    df = pd.concat([df[has_gid].drop_duplicates("gid", keep="first"), df[~has_gid]]).sort_index()
    df = df.reset_index(drop=True)
    duplicates = n_before - len(df)

    matched_seen = [m for m in res["matched_seen"] if m is not None]
    first, last = (matched_seen[0], matched_seen[-1]) if matched_seen else (None, None)
    warnings = meta["warnings"]
    if len(set(matched_seen)) > 1:
        warnings.append(f"numberMatched changed during the download: {sorted(set(matched_seen))} "
                        "(the layer was edited while paging; compare with the final count)")
    if not matched_seen:
        warnings.append("the server did not report numberMatched; the row-count check was skipped")
    if duplicates:
        warnings.append(f"{duplicates} duplicate gid row(s) removed (pages shifted during the download)")

    seen = res["properties"]
    estado = df["ndidestado"].value_counts(dropna=True)
    fechcort = df["ndfechcort"].dropna()
    meta.update(
        number_matched_first=first, number_matched_last=last, rows_downloaded=int(len(df)),
        pages=res["pages"], pages_fetched=res["pages_fetched"], retries=res["retries"],
        duplicates_removed=int(duplicates),
        estado_counts={str(k): int(v) for k, v in estado.items()},
        ndfechcort_max=str(fechcort.max()) if len(fechcort) else None,
        missing_properties=sorted(set(nomenclatura.WFS_FIELDS) - seen) if len(df) else [],
        extra_properties=sorted(seen - set(nomenclatura.WFS_FIELDS)),
        seconds=round(res["seconds"], 1), downloaded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if meta["missing_properties"]:
        warnings.append(f"properties absent from every feature: {meta['missing_properties']}")

    if len(df) == 0:
        meta["status"] = "empty"
        _write_json(args.meta, meta)
        print("ERROR: the service returned no features; nothing to write.", file=sys.stderr)
        return 2
    if last is not None and len(df) != last:
        meta["status"] = "count_mismatch"
        _write_json(args.meta, meta)
        print(f"ERROR: downloaded {len(df)} rows but numberMatched is {last}. "
              "Cached pages are kept; re-run to resume or inspect the meta file.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)
    meta["status"] = "ok"
    _write_json(args.meta, meta)
    print(f"WROTE {args.out} rows={len(df)} numberMatched={last}", flush=True)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)

    if args.validate_shp:
        return validate_against_shapefile(args.out, args.validate_shp, args.vs_json, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
