"""Loading and sampling of the five private evaluation datasets (``source_a`` .. ``source_e``).

The workbooks are not shipped. The context directory defaults to ``context/`` and each file
name can be overridden with ``CALI_EVAL_SOURCE_<A..E>_FILE``.

Column names in these workbooks carry accents and stray whitespace, so every
column is matched on an ASCII-folded, lowercased, stripped form (prefix match).
"""

from __future__ import annotations

import os
import re
import unicodedata

import pandas as pd

__all__ = [
    "DATASETS", "DatasetSpec", "default_context_dir", "norm_col", "pick_col", "load_dataset",
    "sample_dataset",
]

SEED = 42
SAMPLE_N = 500


def norm_col(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def pick_col(df: pd.DataFrame, *prefixes: str) -> str | None:
    """Return the first column whose normalized name starts with any prefix."""
    norm = {c: norm_col(c) for c in df.columns}
    for prefix in prefixes:
        target = norm_col(prefix)
        for col, n in norm.items():
            if n == target:
                return col
        for col, n in norm.items():
            if n.startswith(target):
                return col
    return None


class DatasetSpec:
    def __init__(
        self,
        key: str,
        filename: str,
        sheet: str,
        header: int,
        address_col: str,
        normalized_col: str | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
        coords_col: str | None = None,
        accuracy_col: str | None = None,
        sample_n: int = SAMPLE_N,
        note: str = "",
    ) -> None:
        self.key = key
        self.filename = filename
        self.sheet = sheet
        self.header = header
        self.address_col = address_col
        self.normalized_col = normalized_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.coords_col = coords_col
        self.accuracy_col = accuracy_col
        self.sample_n = sample_n
        self.note = note


def default_context_dir(root: str | None = None) -> str:
    """Directory holding the private workbooks: ``CALI_EVAL_CONTEXT_DIR`` or ``<root>/context``."""
    return os.environ.get("CALI_EVAL_CONTEXT_DIR") or os.path.join(root or os.getcwd(), "context")


def _env_file(var: str, default: str) -> str:
    """File name of one private source: environment variable override, else a generic default."""
    return os.environ.get(var, default)


def _env_sheet(var: str, default: str) -> str:
    """Sheet name of one private source: environment variable override, else a generic default."""
    return os.environ.get(var, default)


# File names and sheet layout of the private evaluation workbooks are deployment
# specific. Override each file with ``CALI_EVAL_SOURCE_<A..E>_FILE`` (a name relative to the
# context directory, or an absolute path) and ``CALI_EVAL_SOURCE_<A..E>_SHEET`` for the sheet.
# Defaults are generic placeholders.
DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        "source_a",
        _env_file("CALI_EVAL_SOURCE_A_FILE", "source_a.xlsx"),
        _env_sheet("CALI_EVAL_SOURCE_A_SHEET", "Sheet1"), 0,
        address_col="direccion",
        normalized_col="direccion_normalizada",
        lat_col="latitud (y) wgs84",
        lon_col="longitud (x) wgs 84",
        note="Insured-property register. Has its own normalized column and WGS84 coordinates.",
    ),
    DatasetSpec(
        "source_b",
        _env_file("CALI_EVAL_SOURCE_B_FILE", "source_b.xlsx"),
        _env_sheet("CALI_EVAL_SOURCE_B_SHEET", "Sheet1"), 0,
        address_col="direccion_bien",
        note="Registry of affected households. No coordinates; several addresses per cell.",
    ),
    DatasetSpec(
        "source_c",
        _env_file("CALI_EVAL_SOURCE_C_FILE", "source_c.xlsx"),
        _env_sheet("CALI_EVAL_SOURCE_C_SHEET", "Sheet1"), 2,
        address_col="direccion",
        normalized_col="direccion_norm",
        coords_col="coords",
        accuracy_col="gps_error_m",
        note=("Field structural inspections. `coords` is a 'lat, lon' string filled for most "
              "rows; `gps_error_m` carries the reported GPS error."),
    ),
    DatasetSpec(
        "source_d",
        _env_file("CALI_EVAL_SOURCE_D_FILE", "source_d.xlsx"),
        _env_sheet("CALI_EVAL_SOURCE_D_SHEET", "Sheet1"), 5,
        address_col="direccion",
        lat_col="lat",
        lon_col="lng",
        accuracy_col="accuracy",
        note=("Field forms captured on a mobile app; lat/lng from device GPS. The "
              "`accuracy` column exists but is empty in the delivered file."),
    ),
    DatasetSpec(
        "source_e",
        _env_file("CALI_EVAL_SOURCE_E_FILE", "source_e.xlsx"),
        _env_sheet("CALI_EVAL_SOURCE_E_SHEET", "Sheet1"), 2,
        address_col="direccion",
        sample_n=10_000,  # only 78 rows exist -> all of them are used
        note="Small list of candidate properties. Only 78 rows, so the whole dataset is evaluated.",
    ),
]


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )


def load_dataset(spec: DatasetSpec, context_dir: str) -> pd.DataFrame:
    """Read one workbook and return a tidy frame: raw / their_norm / lat / lon / accuracy."""
    raw = pd.read_excel(
        os.path.join(context_dir, spec.filename), sheet_name=spec.sheet, header=spec.header
    )
    addr = pick_col(raw, spec.address_col)
    if addr is None:
        raise KeyError(f"{spec.key}: address column {spec.address_col!r} not found")
    out = pd.DataFrame({"raw_address": raw[addr].astype("object")})
    out["dataset"] = spec.key
    out["source_row"] = raw.index

    if spec.normalized_col:
        col = pick_col(raw, spec.normalized_col)
        out["their_normalized"] = raw[col].astype("object") if col else None
    else:
        out["their_normalized"] = None

    lat = lon = None
    if spec.coords_col:
        col = pick_col(raw, spec.coords_col)
        if col is not None:
            pieces = raw[col].astype(str).str.split(",", n=1, expand=True)
            if pieces.shape[1] == 2:
                lat = _coerce_float(pieces[0])
                lon = _coerce_float(pieces[1])
    if lat is None and spec.lat_col:
        cl, co = pick_col(raw, spec.lat_col), pick_col(raw, spec.lon_col or "")
        if cl is not None and co is not None:
            lat, lon = _coerce_float(raw[cl]), _coerce_float(raw[co])
    out["gt_lat"] = lat if lat is not None else pd.NA
    out["gt_lon"] = lon if lon is not None else pd.NA

    if spec.accuracy_col:
        col = pick_col(raw, spec.accuracy_col)
        out["gps_accuracy_m"] = _coerce_float(raw[col]) if col else pd.NA
    else:
        out["gps_accuracy_m"] = pd.NA

    # Cali bounding box sanity check: anything outside is treated as missing.
    lat_n = pd.to_numeric(out["gt_lat"], errors="coerce")
    lon_n = pd.to_numeric(out["gt_lon"], errors="coerce")
    inside = lat_n.between(3.2, 3.65) & lon_n.between(-76.75, -76.4)
    out["gt_lat"] = lat_n.where(inside)
    out["gt_lon"] = lon_n.where(inside)
    return out


def sample_dataset(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Drop null/blank addresses, then take up to ``n`` rows with a fixed seed."""
    s = df["raw_address"]
    keep = s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.strip().str.lower() != "nan")
    clean = df[keep]
    if len(clean) <= n:
        return clean.reset_index(drop=True)
    return clean.sample(n, random_state=seed).reset_index(drop=True)
