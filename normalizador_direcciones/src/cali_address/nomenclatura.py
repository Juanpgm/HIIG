"""Helpers to read and reproject the official certified-nomenclature point layer.

The layer ships as a shapefile in a projected CRS (MAGNA-SIRGAS / Cali urban grid, EPSG:6249).
Everything downstream of ingestion works in EPSG:4326, so the first step is a faithful
reprojection that keeps the source coordinates next to the new ones.

Nothing here raises for a single bad row: null or non-finite geometry stays NaN and is counted
by the caller.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# (lon_min, lat_min, lon_max, lat_max): generous box around the municipality of Cali.
CALI_BBOX = (-76.65, 3.30, -76.40, 3.55)


# The 40 lowercase properties exposed by the WFS layer dapm:pdt_nmc_nomenclatura_domiciliaria.
WFS_FIELDS = (
    "gid", "ndid", "ndconexion", "ndcomuna", "ndidbarrio", "ndbarrio", "ndidpredio", "ndnpn",
    "ndnumepred", "ndmanzana", "ndterreno", "ndpredio", "ndmejora", "ndcondic", "nddirant",
    "nddirecc", "ndnomvial", "ndidtipvia", "ndtipovia", "ndnumvia", "ndplaca", "ndinterior",
    "ndidtipdir", "ndtipodir", "ndcodpost", "ndidestado", "ndestado", "ndfecha", "ndresoluc",
    "ndobservac", "ndnombreur", "ndidtipzon", "ndtipozona", "nddircons1", "nddircons2",
    "ndcoordx", "ndcoordy", "ndfechcort",
)
WFS_INT_FIELDS = ("gid", "ndid", "ndidpredio")
WFS_FLOAT_FIELDS = ("ndcoordx", "ndcoordy")


class CrsError(ValueError):
    """The .prj file is missing, empty or cannot be interpreted."""


def transformer_from_prj(prj_path: str):
    """Build a pyproj ``Transformer`` from the shapefile CRS to EPSG:4326 (``always_xy=True``)."""
    import pyproj

    if not os.path.isfile(prj_path):
        raise CrsError(f"projection file (.prj) not found: {prj_path}")
    with open(prj_path, "r", encoding="utf-8", errors="replace") as fh:
        wkt = fh.read().strip()
    if not wkt:
        raise CrsError(f"projection file (.prj) is empty: {prj_path}")
    try:
        crs = pyproj.CRS.from_wkt(wkt)
    except Exception as exc:  # noqa: BLE001 - pyproj raises several exception types
        raise CrsError(f"could not parse the CRS in {prj_path}: {exc}") from exc
    return pyproj.Transformer.from_crs(crs, pyproj.CRS.from_epsg(4326), always_xy=True)


def read_shapefile_points(shp_path: str, encoding: str | None = None) -> pd.DataFrame:
    """Read a POINT shapefile into a DataFrame: every DBF field plus ``x_src`` / ``y_src``.

    Records and shapes are streamed (``iterRecords`` / ``iterShapes``) so the wide DBF is never
    materialised twice. A null or empty geometry gives NaN coordinates. ``encoding`` defaults to the
    ``.cpg`` sidecar when present, else UTF-8.
    """
    import shapefile

    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"shapefile not found: {shp_path}")
    if encoding is None:
        cpg = os.path.splitext(shp_path)[0] + ".cpg"
        encoding = "utf-8"
        if os.path.isfile(cpg):
            with open(cpg, "r", encoding="ascii", errors="ignore") as fh:
                encoding = fh.read().strip() or "utf-8"
    reader = shapefile.Reader(os.path.splitext(shp_path)[0], encoding=encoding)
    try:
        names = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
        rows, xs, ys = [], [], []
        for shape, record in zip(reader.iterShapes(), reader.iterRecords()):
            pts = getattr(shape, "points", None)
            if pts:
                xs.append(float(pts[0][0]))
                ys.append(float(pts[0][1]))
            else:
                xs.append(np.nan)
                ys.append(np.nan)
            rows.append(list(record))
    finally:
        reader.close()
    df = pd.DataFrame(rows, columns=names)
    df["x_src"] = np.asarray(xs, dtype="float64")
    df["y_src"] = np.asarray(ys, dtype="float64")
    return df


def reproject_frame(df: pd.DataFrame, transformer) -> pd.DataFrame:
    """Return a copy of ``df`` with ``lon`` / ``lat`` computed from ``x_src`` / ``y_src``.

    Rows whose source coordinates are NaN or infinite get NaN in both outputs.
    """
    out = df.copy()
    x = out["x_src"].to_numpy(dtype="float64")
    y = out["y_src"].to_numpy(dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    lon = np.full(len(out), np.nan)
    lat = np.full(len(out), np.nan)
    if ok.any():
        lon[ok], lat[ok] = transformer.transform(x[ok], y[ok])
    out["lon"] = lon
    out["lat"] = lat
    return out


def reproject_xy(x, y, transformer) -> tuple[np.ndarray, np.ndarray]:
    """Projected ``x`` / ``y`` (metres) -> ``(lon, lat)`` arrays through ``transformer``.

    ``transformer`` comes from :func:`transformer_from_prj` (``always_xy=True``: x/easting first,
    lon first on output). Pairs with a non-finite input, and any pair that projects to a non-finite
    value, give NaN in both outputs; nothing raises for a single bad point.
    """
    xs = np.asarray(x, dtype="float64")
    ys = np.asarray(y, dtype="float64")
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same shape, got {xs.shape} and {ys.shape}")
    lon = np.full(xs.shape, np.nan)
    lat = np.full(xs.shape, np.nan)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.any():
        lo, la = transformer.transform(xs[ok], ys[ok])
        lon[ok], lat[ok] = lo, la
    bad = ~(np.isfinite(lon) & np.isfinite(lat))
    lon[bad] = np.nan
    lat[bad] = np.nan
    return lon, lat


def reproject_wfs_coordinates(df: pd.DataFrame, transformer) -> tuple[np.ndarray, np.ndarray]:
    """``(lon, lat)`` of the WFS projected columns ``ndcoordx`` / ``ndcoordy`` (EPSG:6249 -> 4326).

    Null (including nullable-dtype ``NA``) and non-finite values give NaN. These are the coordinates
    the index uses; the WFS' own geometry is only kept for audit.
    """
    def column(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)

    return reproject_xy(column("ndcoordx"), column("ndcoordy"), transformer)


def outside_bbox_mask(lon, lat, bbox: tuple[float, float, float, float] = CALI_BBOX) -> np.ndarray:
    """True where a finite point lies outside the (closed) box; NaN points are not "outside"."""
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    lon_min, lat_min, lon_max, lat_max = bbox
    finite = np.isfinite(lon) & np.isfinite(lat)
    inside = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    return finite & ~inside


# ---------------------------------------------------------------------------
# WFS vs shapefile comparison (both sides already in EPSG:4326)
# ---------------------------------------------------------------------------

def _norm_address(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.upper().str.split().str.join(" ")


def _keyed(df: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, int]:
    """Rows with a usable key (non-null, non-zero) deduplicated by key; also the duplicates dropped."""
    df = df.reset_index(drop=True)  # callers may pass concatenated frames with repeated labels
    ids = pd.to_numeric(df[id_col], errors="coerce")
    usable = df.loc[ids.notna() & (ids != 0)].copy()
    usable["_key"] = ids[usable.index].astype("int64")
    n_before = len(usable)
    usable = usable.drop_duplicates("_key", keep="first")
    return usable.set_index("_key"), n_before - len(usable)


def coord_shift_by_ndid(wfs: pd.DataFrame, shp: pd.DataFrame) -> pd.Series:
    """Haversine distance (m) between WFS and shapefile points, indexed by the common ``ndid``.

    Pairs with a non-finite coordinate on either side are left out. Null and zero ids are never keys.
    """
    from .geo import haversine_m

    w, _ = _keyed(wfs, "ndid")
    s, _ = _keyed(shp, "NdID")
    common = w.index.intersection(s.index)
    if len(common) == 0:
        return pd.Series([], dtype="float64", index=pd.Index([], dtype="int64"), name="shift_m")
    wl, sl = w.loc[common], s.loc[common]
    d = haversine_m(wl["lat"].to_numpy(float), wl["lon"].to_numpy(float),
                    sl["lat"].to_numpy(float), sl["lon"].to_numpy(float))
    out = pd.Series(d, index=common, name="shift_m")
    return out[np.isfinite(out)]


def compare_wfs_shapefile(
    wfs: pd.DataFrame,
    shp: pd.DataFrame,
    *,
    max_count_diff: float = 0.005,
    min_agreement: float = 0.995,
    min_npn_lot_agreement: float = 0.98,
    max_median_shift_m: float = 1.0,
    max_p99_shift_m: float = 2.0,
) -> dict:
    """Agreement between the WFS download and the (reprojected) shapefile snapshot.

    ``wfs`` needs ``ndid, ndidestado, ndnpn, nddirecc, lon, lat``; ``shp`` needs
    ``NdID, NdEstado, NdNpn, NdDirecc, lon, lat``. Gates (all inclusive):

    * certified-count difference below ``max_count_diff``;
    * NPN agreement at LOT level (``NPN[:21]``) on common ids at least ``min_npn_lot_agreement``.
      The full-NPN agreement (``npn_agreement``) is reported for information only: the last two
      digits are a unit suffix that differs between the two snapshots for ~30 % of the ids;
    * address agreement on common ids at least ``min_agreement``;
    * coordinates: median shift at most ``max_median_shift_m`` AND p99 at most ``max_p99_shift_m``.
      The shift is a datum-transformation difference between GeoServer and pyproj (the projected
      coordinates are identical), so a hard per-point limit would fail on a systematic sub-metre
      offset. The share over 1 m / 2 m is reported for information.
    """
    w_estado = wfs["ndidestado"].astype("string").str.strip().str.lower()
    s_estado = shp["NdEstado"].astype("string").str.strip().str.lower()
    wfs_cert = int((w_estado == "cert").sum())
    wfs_pcer = int((w_estado == "pcer").sum())
    shp_cert = int((s_estado == "cert").sum())
    count_diff = abs(wfs_cert - shp_cert) / shp_cert if shp_cert else None
    cert_count_ok = bool(count_diff is not None and count_diff < max_count_diff)

    w, w_dups = _keyed(wfs, "ndid")
    s, s_dups = _keyed(shp, "NdID")
    common = w.index.intersection(s.index)
    npn_agreement = addr_agreement = npn_lot_agreement = source_identical = None
    if len(common):
        wl, sl = w.loc[common], s.loc[common]
        w_npn = wl["ndnpn"].astype("string").str.strip().fillna("")
        s_npn = sl["NdNpn"].astype("string").str.strip().fillna("")
        npn_agreement = float((w_npn == s_npn).mean())
        # The first 21 characters identify the lot (the rest is the unit suffix): this is the NPN gate.
        npn_lot_agreement = float((w_npn.str[:21] == s_npn.str[:21]).mean())
        addr_agreement = float((_norm_address(wl["nddirecc"]) == _norm_address(sl["NdDirecc"])).mean())
        if {"ndcoordx", "ndcoordy"} <= set(wl.columns) and {"x_6249", "y_6249"} <= set(sl.columns):
            dx = pd.to_numeric(wl["ndcoordx"], errors="coerce").to_numpy(float) - sl["x_6249"].to_numpy(float)
            dy = pd.to_numeric(wl["ndcoordy"], errors="coerce").to_numpy(float) - sl["y_6249"].to_numpy(float)
            both = np.isfinite(dx) & np.isfinite(dy)
            if both.any():
                source_identical = float(((np.abs(dx[both]) < 1e-3) & (np.abs(dy[both]) < 1e-3)).mean())
    shifts = coord_shift_by_ndid(wfs, shp)
    coord_compared = int(len(shifts))
    over_1m = int((shifts > 1.0).sum())
    over_2m = int((shifts > 2.0).sum())
    share_within_1m = float((shifts <= 1.0).mean()) if coord_compared else None
    share_within_2m = float((shifts <= 2.0).mean()) if coord_compared else None
    median_shift = float(shifts.median()) if coord_compared else None
    p99_shift = float(shifts.quantile(0.99)) if coord_compared else None

    npn_ok = bool(npn_lot_agreement is not None and npn_lot_agreement >= min_npn_lot_agreement)
    addr_ok = bool(addr_agreement is not None and addr_agreement >= min_agreement)
    coord_ok = bool(coord_compared and median_shift <= max_median_shift_m and p99_shift <= max_p99_shift_m)
    return {
        "wfs_cert_count": wfs_cert,
        "wfs_pcer_count": wfs_pcer,
        "wfs_rows": int(len(wfs)),
        "shp_cert_count": shp_cert,
        "shp_rows": int(len(shp)),
        "cert_count_diff_share": None if count_diff is None else float(count_diff),
        "cert_count_ok": cert_count_ok,
        "common_ndid": int(len(common)),
        "wfs_duplicate_ndid": int(w_dups),
        "shp_duplicate_ndid": int(s_dups),
        "npn_agreement": npn_agreement,
        "npn_ok": npn_ok,
        "npn_lot_agreement": npn_lot_agreement,
        "source_coords_identical_share": source_identical,
        "address_agreement": addr_agreement,
        "address_ok": addr_ok,
        "coord_compared": coord_compared,
        "coord_over_1m": over_1m,
        "coord_over_2m": over_2m,
        "coord_share_within_1m": share_within_1m,
        "coord_share_within_2m": share_within_2m,
        "coord_median_shift_m": median_shift,
        "coord_p99_shift_m": p99_shift,
        "coord_max_shift_m": float(shifts.max()) if coord_compared else None,
        "coord_ok": coord_ok,
        "thresholds": {
            "max_count_diff": max_count_diff, "min_agreement": min_agreement,
            "min_npn_lot_agreement": min_npn_lot_agreement,
            "max_median_shift_m": max_median_shift_m, "max_p99_shift_m": max_p99_shift_m,
        },
        "passed": bool(cert_count_ok and npn_ok and addr_ok and coord_ok),
    }
