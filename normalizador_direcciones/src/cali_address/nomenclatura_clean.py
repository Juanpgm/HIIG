"""Cleaning of the WFS nomenclature download, with a per-rule report (plan D1b).

Pure functions only; ``scripts/clean_nomenclatura.py`` does the I/O. The rules, in order:

1. schema   - the 40 WFS columns plus ``lon`` / ``lat`` must exist (missing -> :class:`SchemaError`);
              wrong dtypes are coerced; extra columns are recorded and dropped.
2. text     - address columns: control characters, stray quotes, ``(Direccion Catastral)``-style markers,
              number-sign variants -> ``#``, uppercase, collapsed whitespace.
3. consistency - ``nddirecc`` is rebuilt from ``ndnomvial`` (or type code + number), ``#``, ``ndplaca``
              and ``ndinterior`` when it disagrees (whitespace aside); the raw value is kept.
              Unit-suffix consistency rule: ``nddirecc`` may legitimately be LONGER than the rebuilt
              address, because the layer appends the unit to the base address (``... # 46 - 131 ED 47
              AP 302``). A longer address that starts with the rebuilt one (spaces ignored) is
              therefore consistent and is NOT rebuilt (the unit is kept) - unless the extra text just
              continues the last digit of the plate (``... - 131`` vs ``... - 1317``), which is a real
              disagreement. A shorter or otherwise different address IS rebuilt from the components.
              Rows without components (``pcer``) keep their cleaned ``nddirecc``.
4. NPN      - stripped; ``npn_valid`` = 30 digits starting with ``76001``. Invalid NPNs are KEPT.
5. status   - only ``cert`` / ``pcer`` pass.
6. coordinates - ``lon`` / ``lat`` are the LOCAL reprojection of ``ndcoordx`` / ``ndcoordy``
              (EPSG:6249 -> 4326; ``lon_wfs`` / ``lat_wfs`` keep the WFS point for audit). Rows with
              null / non-finite / non-positive projected coordinates fall back to the WFS point
              (``coord_source = "wfs"``). Then: no usable point, outside the Cali bounding box
              (checked on the coordinate that is used), or (optional hook) farther than
              ``max_shift_m`` from the locally reprojected shapefile point of the same ``ndid``.
7. duplicates - among the rows that passed the rules above: one row per ``gid``, then one row per
              non-null, non-zero ``ndid``; the latest ``ndfecha`` wins.

No rule raises for a single bad row: it goes to the quarantine frame with every reason that applies.
Rule counts are computed over ALL input rows, so a row hitting several rules is counted in each.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import nomenclatura
from .nomenclatura import CALI_BBOX, WFS_FIELDS, WFS_FLOAT_FIELDS, WFS_INT_FIELDS, outside_bbox_mask

QUARANTINE_SHARE_LIMIT = 0.005
VALID_STATUSES = ("cert", "pcer")
COORD_COLUMNS = ("lon", "lat")
#: ``coord_source`` of a row whose ``lon`` / ``lat`` come from reprojecting ``ndcoordx`` / ``ndcoordy``.
LOCAL_SOURCE = "local_6249_to_4326"
#: Columns this module adds for the coordinates; a cleaned frame fed back in is not "extra".
DERIVED_COORD_COLUMNS = ("lon_wfs", "lat_wfs", "coord_source")

# Columns holding (parts of) an address: full cleaning. Every other text column is only stripped.
ADDRESS_COLUMNS = ("nddirecc", "nddircons1", "nddircons2", "nddirant", "ndnomvial", "ndplaca",
                   "ndinterior", "ndnumvia", "ndidtipvia")

_MARKER_RE = re.compile(
    r"[\(\[]\s*(?:(?:direcci[oó]n|dir\.?)\s*)?(?:catastral|provisional|por\s+certificar)\s*[\)\]]?",
    re.IGNORECASE,
)
_QUOTES = set("\"'`“”„«»‘’")
# "N" + degree/ordinal sign, with an optional period ("N°", "Nº."), and the numero sign.
_NUMBER_SIGN_N_RE = re.compile(r"(?<![A-ZÑ])N\s*[º°˚]\.?")
_NUMBER_SIGN_NUMERO_RE = re.compile("№\\.?")
# "NO." only when it is immediately followed by a digit; bare "NO" is a quadrant or part of a word.
_NUMBER_SIGN_NO_RE = re.compile(r"(?<![A-ZÑ\d])NO\.\s*(?=\d)")


class SchemaError(ValueError):
    """The input does not have the expected columns; the message lists them."""


@dataclass
class CleanResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    report: dict


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

def _to_text(value) -> str | None:
    """Any cell -> ``str`` or ``None`` (null-like). Never raises."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - exotic objects with a failing __str__
        return None


def clean_address_text(value) -> tuple[str | None, dict]:
    """Clean one address string; returns ``(text or None, flags)``.

    Flags: ``marker_removed``, ``number_sign``, ``control_chars``, ``quotes``. Idempotent.
    """
    flags = {"marker_removed": False, "number_sign": False, "control_chars": False, "quotes": False}
    text = _to_text(value)
    if text is None:
        return None, flags
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in _QUOTES:
            flags["quotes"] = True
        elif cat == "Cc":
            if ch in "\t\n\r\x0b\x0c":
                out.append(" ")
            else:
                flags["control_chars"] = True
        elif cat == "Cf":
            flags["control_chars"] = True
        else:
            out.append(ch)
    text = "".join(out)
    text, n = _MARKER_RE.subn(" ", text)
    flags["marker_removed"] = n > 0
    text = text.upper()
    for pattern in (_NUMBER_SIGN_N_RE, _NUMBER_SIGN_NUMERO_RE, _NUMBER_SIGN_NO_RE):
        text, n = pattern.subn(" # ", text)
        if n:
            flags["number_sign"] = True
    text = " ".join(text.split())
    return (text or None), flags


def _clean_address_series(s: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Apply :func:`clean_address_text` once per distinct value."""
    raw = s.astype(object).map(_to_text)
    uniques = raw.dropna().unique()
    mapping = {u: clean_address_text(u) for u in uniques}
    cleaned = raw.map(lambda v: mapping[v][0] if v is not None and v in mapping else None)
    flag_names = ("marker_removed", "number_sign", "control_chars", "quotes")
    flags = pd.DataFrame({
        f: raw.map(lambda v, f=f: mapping[v][1][f] if v is not None and v in mapping else False).astype(bool)
        for f in flag_names
    })
    return cleaned.astype(object), flags


def _light_clean_series(s: pd.Series) -> pd.Series:
    """Strip; empty -> None; non-strings are stringified. Case and punctuation are left alone."""
    txt = s.astype(object).map(_to_text).astype("string").str.strip()
    txt = txt.where(txt.fillna("") != "", other=pd.NA)
    return txt.astype(object).where(txt.notna(), None)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def _validate_and_coerce(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = list(WFS_FIELDS) + list(COORD_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError("input is missing required column(s): " + ", ".join(missing))
    extra = sorted(c for c in df.columns if c not in required and c not in DERIVED_COORD_COLUMNS)
    work = df[required].copy().reset_index(drop=True)
    coerced = []
    for c in list(WFS_INT_FIELDS) + list(WFS_FLOAT_FIELDS) + list(COORD_COLUMNS):
        if not pd.api.types.is_numeric_dtype(work[c]) or pd.api.types.is_bool_dtype(work[c]):
            if work[c].notna().any():
                coerced.append(c)
            work[c] = pd.to_numeric(work[c].astype(object).map(_to_text), errors="coerce")
        if c in WFS_INT_FIELDS:
            try:
                work[c] = work[c].astype("Int64")
            except (TypeError, ValueError):
                work[c] = work[c].astype("float64")
        else:
            work[c] = work[c].astype("float64")
    numeric = set(WFS_INT_FIELDS) | set(WFS_FLOAT_FIELDS) | set(COORD_COLUMNS)
    for c in WFS_FIELDS:
        if c in numeric:
            continue
        if pd.api.types.is_numeric_dtype(work[c]) and work[c].notna().any():
            coerced.append(c)
    info = {"expected_columns": len(required), "missing_columns": [], "extra_columns": extra,
            "coerced_columns": sorted(set(coerced))}
    return work, info


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _numeric(s: pd.Series) -> np.ndarray:
    """Any column -> float64 array; unparseable / null cells (including ``pd.NA``) become NaN."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _shift_statistics(local_lon, local_lat, wfs_lon, wfs_lat) -> dict:
    """Distance (m) between the local reprojection and the WFS point, over rows that have both."""
    from .geo import haversine_m

    both = np.isfinite(local_lon) & np.isfinite(local_lat) & np.isfinite(wfs_lon) & np.isfinite(wfs_lat)
    if not both.any():
        return {"n": 0, "median_m": None, "p99_m": None, "max_m": None, "share_over_1m": None,
                "share_over_2m": None, "over_100m": 0}
    d = np.asarray(haversine_m(local_lat[both], local_lon[both], wfs_lat[both], wfs_lon[both]), dtype="float64")
    return {"n": int(both.sum()), "median_m": float(np.median(d)), "p99_m": float(np.quantile(d, 0.99)),
            "max_m": float(d.max()), "share_over_1m": float((d > 1.0).mean()),
            "share_over_2m": float((d > 2.0).mean()), "over_100m": int((d > 100.0).sum())}


def _squash(s: pd.Series) -> pd.Series:
    return s.str.replace(r"\s+", "", regex=True)


def _differs(current: str, expected: str) -> bool:
    """Whitespace-free comparison. A longer ``current`` that starts with ``expected`` is consistent
    (the extra text is a unit such as ``AP 302``), unless it just continues the last digit of the plate."""
    if current == expected:
        return False
    if not expected or not current.startswith(expected):
        return True
    return expected[-1].isdigit() and current[len(expected)].isdigit()


def _fecha_rank(ndfecha: pd.Series) -> pd.Series:
    """Rank of ``ndfecha`` (1 = most recent); unparseable / null dates rank last."""
    txt = ndfecha.astype(object).map(_to_text).astype("string").str.strip().str.rstrip("Zz")
    dates = pd.to_datetime(txt, errors="coerce", format="mixed")
    return dates.rank(method="min", ascending=False, na_option="bottom")


def _signature(df: pd.DataFrame) -> pd.Series:
    def col(name):
        return df[name].astype(object).map(_to_text).fillna("").astype(str)

    return (col("ndnpn") + "|" + col("nddirecc") + "|" + col("ndinterior") + "|" + col("ndidestado") + "|"
            + df["lon"].round(7).astype(str) + "|" + df["lat"].round(7).astype(str))


def _dedupe(candidates: np.ndarray, key: pd.Series, rank: pd.Series, sig: pd.Series):
    """Positions to drop so each ``key`` keeps its most recent row; split into identical / conflicting."""
    pos = np.flatnonzero(candidates)
    if len(pos) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    cand = pd.DataFrame({"pos": pos, "key": key.to_numpy()[pos], "rank": rank.to_numpy()[pos],
                         "sig": sig.to_numpy()[pos]})
    cand = cand.sort_values(["rank", "pos"], kind="stable")
    keeper_sig = cand.groupby("key")["sig"].transform("first")
    dup = cand.duplicated("key", keep="first")
    dropped = cand[dup]
    same = dropped["sig"].to_numpy() == keeper_sig[dup].to_numpy()
    return dropped["pos"].to_numpy()[same], dropped["pos"].to_numpy()[~same]


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def clean_nomenclatura(
    df: pd.DataFrame,
    *,
    transformer=None,
    shp_shift: pd.Series | None = None,
    max_shift_m: float = 1.0,
    bbox: tuple[float, float, float, float] = CALI_BBOX,
    share_limit: float = QUARANTINE_SHARE_LIMIT,
) -> CleanResult:
    """Clean the WFS frame; returns clean rows, quarantined rows (raw + reason) and the report.

    ``transformer`` (``nomenclatura.transformer_from_prj``) reprojects ``ndcoordx`` / ``ndcoordy``
    (EPSG:6249) to EPSG:4326: the output ``lon`` / ``lat`` are those local values
    (``coord_source == "local_6249_to_4326"``) and the WFS point is kept as ``lon_wfs`` / ``lat_wfs``.
    Rows whose projected coordinates are null, non-finite or non-positive (the layer uses 0 / 0 for
    "none") fall back to the WFS point (``coord_source == "wfs"``, counted in
    ``rules["coords_fallback_to_wfs"]``). The bounding-box gate sees the coordinate that is used.
    Without a ``transformer`` every row keeps its WFS point (``coord_source == "wfs"``).

    ``shp_shift`` optionally maps ``ndid`` -> metres between the WFS point and the locally
    reprojected shapefile point (see ``nomenclatura.coord_shift_by_ndid``); rows above
    ``max_shift_m`` are quarantined.
    """
    raw_input = df.reset_index(drop=True)
    work, schema_info = _validate_and_coerce(raw_input)
    n = len(work)
    rules: dict[str, int] = {}

    # --- text --------------------------------------------------------------------------------
    flag_any = {f: np.zeros(n, dtype=bool) for f in ("marker_removed", "number_sign", "control_chars", "quotes")}
    cleaned_cols = {}
    for c in ADDRESS_COLUMNS:
        cleaned_cols[c], fl = _clean_address_series(work[c])
        for f in flag_any:
            flag_any[f] |= fl[f].to_numpy()
    nddirecc_original = work["nddirecc"].astype(object).map(_to_text)
    for c in ADDRESS_COLUMNS:
        work[c] = cleaned_cols[c]
    for c in WFS_FIELDS:
        if c in ADDRESS_COLUMNS or c in WFS_INT_FIELDS or c in WFS_FLOAT_FIELDS:
            continue
        work[c] = _light_clean_series(work[c])
    rules["marker_removed"] = int(flag_any["marker_removed"].sum())
    rules["number_sign_normalized"] = int(flag_any["number_sign"].sum())
    rules["control_chars_removed"] = int(flag_any["control_chars"].sum())
    rules["quotes_removed"] = int(flag_any["quotes"].sum())

    # --- consistency -------------------------------------------------------------------------
    nomvial = work["ndnomvial"].astype("string")
    alt_via = work["ndidtipvia"].astype("string") + " " + work["ndnumvia"].astype("string")
    via = nomvial.fillna(alt_via)
    placa = work["ndplaca"].astype("string")
    interior = work["ndinterior"].astype("string")
    expected = via + " # " + placa + (" " + interior).fillna("")
    has_parts = (via.notna() & placa.notna()).to_numpy()
    current = work["nddirecc"].astype("string")
    differs = np.array([
        _differs(cur, exp) for cur, exp in zip(_squash(current.fillna("")).tolist(), _squash(expected.fillna("")).tolist())
    ], dtype=bool) if n else np.zeros(0, dtype=bool)
    differs |= current.isna().to_numpy()
    rebuilt = has_parts & differs
    work["nddirecc"] = pd.Series(
        np.where(rebuilt, expected.astype(object).to_numpy(), work["nddirecc"].to_numpy()), dtype=object
    ).where(lambda s: s.notna(), None)
    rules["rebuilt_from_parts"] = int(rebuilt.sum())
    empty_address = work["nddirecc"].isna().to_numpy()
    rules["empty_address"] = int(empty_address.sum())

    # --- NPN ---------------------------------------------------------------------------------
    npn = work["ndnpn"].astype("string")
    npn_len = npn.str.len().fillna(0).astype("int64")
    npn_valid = ((npn_len == 30) & npn.str.startswith("76001").fillna(False) & npn.str.fullmatch(r"\d+").fillna(False))
    rules["npn_invalid"] = int((~npn_valid).sum())
    rules["npn_len_28"] = int((npn_len == 28).sum())
    rules["npn_len_29"] = int((npn_len == 29).sum())

    # --- status ------------------------------------------------------------------------------
    estado_literal = work["ndidestado"].astype("string")
    estado_norm = estado_literal.str.lower()
    status_ok = estado_norm.isin(VALID_STATUSES).to_numpy()
    work["ndidestado"] = estado_norm.where(estado_norm.isin(VALID_STATUSES), estado_literal).astype(object)
    work["ndidestado"] = work["ndidestado"].where(work["ndidestado"].notna(), None)
    literals = estado_literal.fillna("<null>").where(estado_literal.fillna("") != "", "<null>").astype(object)
    rules["status_invalid"] = int((~status_ok).sum())
    literal_counts = Counter(literals[~status_ok].tolist())

    # --- coordinates -------------------------------------------------------------------------
    # The index coordinate is the LOCAL reprojection of ndcoordx / ndcoordy; the WFS point is the
    # fallback and the audit value. A cleaned frame fed back in keeps its own lon_wfs / lat_wfs.
    prior = {"lon_wfs", "lat_wfs"} <= set(raw_input.columns)
    wfs_lon = _numeric(raw_input["lon_wfs"] if prior else work["lon"])
    wfs_lat = _numeric(raw_input["lat_wfs"] if prior else work["lat"])
    proj_x, proj_y = _numeric(work["ndcoordx"]), _numeric(work["ndcoordy"])
    proj_missing = ~(np.isfinite(proj_x) & np.isfinite(proj_y))
    proj_placeholder = ~proj_missing & ((proj_x <= 0) | (proj_y <= 0))   # 0 / 0 is how the layer says "none"
    local_lon = np.full(n, np.nan)
    local_lat = np.full(n, np.nan)
    if transformer is not None:
        usable = ~proj_missing & ~proj_placeholder
        if usable.any():
            local_lon[usable], local_lat[usable] = nomenclatura.reproject_xy(proj_x[usable], proj_y[usable], transformer)
    elif prior and "coord_source" in raw_input.columns:    # no transformer: keep the earlier provenance
        earlier = (raw_input["coord_source"].astype(object) == LOCAL_SOURCE).to_numpy(dtype=bool)
        local_lon = np.where(earlier, _numeric(work["lon"]), np.nan)
        local_lat = np.where(earlier, _numeric(work["lat"]), np.nan)
    have_local = np.isfinite(local_lon) & np.isfinite(local_lat)
    have_wfs = np.isfinite(wfs_lon) & np.isfinite(wfs_lat)
    use_wfs = ~have_local & have_wfs
    lon = np.where(have_local, local_lon, np.where(use_wfs, wfs_lon, np.nan))
    lat = np.where(have_local, local_lat, np.where(use_wfs, wfs_lat, np.nan))
    coord_source = np.where(have_local, LOCAL_SOURCE, np.where(use_wfs, "wfs", None)).astype(object)
    work["lon"], work["lat"] = lon, lat
    work["lon_wfs"], work["lat_wfs"] = wfs_lon, wfs_lat
    work["coord_source"] = coord_source
    rules["coords_projected_missing"] = int(proj_missing.sum())
    rules["coords_projected_zero"] = int(proj_placeholder.sum())
    rules["coords_fallback_to_wfs"] = int(use_wfs.sum()) if transformer is not None else 0
    coords_nan = ~(np.isfinite(lon) & np.isfinite(lat))
    coords_out = outside_bbox_mask(lon, lat, bbox)       # the gate sees the coordinate that will be used
    shift_bad = np.zeros(n, dtype=bool)
    if shp_shift is not None and len(shp_shift):
        lookup = {int(k): float(v) for k, v in shp_shift.items() if pd.notna(k) and pd.notna(v)}
        ndid_num = pd.to_numeric(work["ndid"], errors="coerce").to_numpy(dtype="float64")
        shifts = np.array([lookup.get(int(v), np.nan) if np.isfinite(v) and v != 0 else np.nan for v in ndid_num])
        shift_bad = shifts > max_shift_m
    rules["coords_nan"] = int(coords_nan.sum())
    rules["coords_outside_bbox"] = int(coords_out.sum())
    rules["coords_shift_vs_shp"] = int(shift_bad.sum())

    # --- duplicates (only among rows that survived the rules above) ----------------------------
    bad = ~status_ok | coords_nan | coords_out | shift_bad | empty_address
    rank = _fecha_rank(work["ndfecha"])
    sig = _signature(work)
    gid_drop_same, gid_drop_conf = _dedupe(~bad & work["gid"].notna().to_numpy(), work["gid"].astype("float64"), rank, sig)
    dup_gid = np.zeros(n, dtype=bool)
    dup_gid[gid_drop_same] = True
    dup_gid[gid_drop_conf] = True
    ndid = pd.to_numeric(work["ndid"], errors="coerce").astype("float64")
    usable_ndid = (ndid.notna() & (ndid != 0)).to_numpy()
    nd_same, nd_conf = _dedupe(~bad & ~dup_gid & usable_ndid, ndid, rank, sig)
    dup_ndid = np.zeros(n, dtype=bool)
    dup_ndid[nd_same] = True
    dup_ndid[nd_conf] = True
    rules["duplicate_gid"] = int(dup_gid.sum())
    rules["duplicate_ndid_identical"] = int(len(nd_same))
    rules["duplicate_ndid_conflicting"] = int(len(nd_conf))

    # --- quarantine reasons --------------------------------------------------------------------
    reject = bad | dup_gid | dup_ndid
    reasons: list[str | None] = [None] * n
    for i in np.flatnonzero(reject):
        parts = []
        if not status_ok[i]:
            parts.append(f"status_invalid:{literals.iloc[i]}")
        if coords_nan[i]:
            parts.append("coords_nan")
        if coords_out[i]:
            parts.append("coords_outside_bbox")
        if shift_bad[i]:
            parts.append("coords_shift_vs_shp")
        if empty_address[i]:
            parts.append("empty_address")
        if dup_gid[i]:
            parts.append("duplicate_gid")
        if dup_ndid[i]:
            parts.append("duplicate_ndid")
        reasons[i] = ";".join(parts)

    # --- outputs -----------------------------------------------------------------------------
    work["nddirecc_original"] = nddirecc_original.astype(object)
    work["marker_removed"] = flag_any["marker_removed"]
    work["rebuilt_from_parts"] = rebuilt
    work["npn_valid"] = npn_valid.to_numpy(dtype=bool)
    work["npn_len"] = npn_len.to_numpy()
    clean = work.loc[~reject].reset_index(drop=True)
    quarantine = raw_input.loc[reject].copy().reset_index(drop=True)
    quarantine["quarantine_reason"] = [reasons[i] for i in np.flatnonzero(reject)]

    n_q = int(reject.sum())
    share = (n_q / n) if n else 0.0
    breakdown = Counter(r.split(":")[0] for rs in quarantine["quarantine_reason"] for r in rs.split(";"))
    report = {
        "input_rows": int(n),
        "clean_rows": int(len(clean)),
        "quarantined_rows": n_q,
        "quarantine_share": float(share),
        "quarantine_share_limit": share_limit,
        "gate_passed": bool(share <= share_limit),
        "rules": {k: int(v) for k, v in rules.items()},
        "quarantine_reason_rows": {k: int(v) for k, v in sorted(breakdown.items())},
        "status_invalid_literals": {k: int(v) for k, v in literal_counts.items()},
        "estado_counts_clean": {str(k): int(v) for k, v in clean["ndidestado"].value_counts().items()},
        "coord_source_counts": {str(k): int(v) for k, v in clean["coord_source"].value_counts().items()},
        "coord_shift_local_vs_wfs": _shift_statistics(local_lon, local_lat, wfs_lon, wfs_lat),
        "schema": schema_info,
    }
    return CleanResult(clean=clean, quarantine=quarantine, report=report)
