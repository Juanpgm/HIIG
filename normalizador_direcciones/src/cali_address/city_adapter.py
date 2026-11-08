"""Generic ingestion adapter: any city's address table -> the cleaned nomenclature schema of ``build_index.py``.

A declarative mapping (JSON, TOML, or YAML when PyYAML is installed) says which input column plays which role::

    {
      "city": "Ciudad Sintetica",
      "input": "registros.csv",                 # relative to the mapping file
      "crs": "EPSG:3116",                       # CRS of x / y (default EPSG:4326: x = lon, y = lat)
      "columns": {
        "address": "dir_texto",                 # MANDATORY (or a list of columns joined with a space)
        "record_id": "id_registro",             # MANDATORY, unique
        "x": "coord_x", "y": "coord_y",         # MANDATORY unless the input carries geometry
        "parcel_id": "codigo_predial",          # optional (NPN or equivalent)
        "block": "manzana_cod", "neighborhood": "barrio_nombre", "district": "distrito_nombre",
        "status": "estado_registro", "unit": "unidad"
      },
      "status_values": {"cert": ["vigente"], "pcer": ["provisional"]},
      "bbox": [-74.2, 4.4, -73.9, 4.9],         # optional sanity box in lon/lat
      "max_km_from_median": 30,                 # optional outlier radius (null disables)
      "max_quarantine_share": 0.05              # gate: more quarantined rows than this fails the run
    }

:func:`adapt` is pure (no I/O): it validates the table, quarantines bad rows with ONE reason each and returns the
clean frame in the schema ``index_union.build_nomenclatura`` reads (``gid``, ``nddirecc``, ``ndnpn``,
``ndidestado``, ``lon``, ``lat`` plus the optional ``ndinterior``, ``ndcomuna``, ``ndidbarrio``, ``ndmanzana``).
File-level problems (missing column, undeclared or wrong CRS, empty table, unknown mapping key) raise
:class:`AdapterError` with a one-line, actionable message; they never surface as a traceback in
``scripts/prepare_city.py``.

Limits (documented in ``documentacion/11-entrenar-para-cualquier-ciudad.md``): the parser behind the index targets
Colombian urban nomenclature. District and neighbourhood names become synthetic codes (``city_codes.json``) because
the hierarchy labels of training are read from them; a parcel id is only interpreted as a national predial number
(NPN) by the index builder when it is a Cali code, otherwise its first 17 characters become the block when it has
at least 17.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "AdapterError", "InputReadError", "CityMapping", "AdapterResult", "QUARANTINE_REASONS", "MANDATORY_ROLES",
    "OPTIONAL_ROLES", "load_mapping", "mapping_from_dict", "read_source", "adapt", "repair_mojibake",
]

#: Roles a mapping may assign. ``x`` / ``y`` are mandatory unless the input has geometry (GeoJSON / shapefile).
MANDATORY_ROLES = ("address", "record_id", "x", "y")
OPTIONAL_ROLES = ("parcel_id", "block", "neighborhood", "district", "status", "unit")
_TOP_KEYS = {"city", "input", "crs", "columns", "status_values", "bbox", "max_km_from_median",
             "max_quarantine_share", "format", "encoding", "delimiter", "sheet", "header_row", "layer"}
#: Reasons, in the order they are tested (a row gets the first that applies).
QUARANTINE_REASONS = (
    "missing_record_id", "duplicate_record_id", "empty_address", "missing_coordinates",
    "coordinates_out_of_range", "outside_bbox", "far_from_median", "status_not_accepted", "duplicate_row",
)
_GEO_FORMATS = ("geojson", "shp", "gpkg")
#: Share of rows with coordinates that must be plausible after the CRS transform, or the CRS is wrong.
MIN_PLAUSIBLE_SHARE = 0.5
DEFAULT_MAX_QUARANTINE_SHARE = 0.05
_MOJIBAKE = re.compile("[ÃÂ][\u0080-¿‘-›Œ-Ÿˆ˜]")


class AdapterError(Exception):
    """A user-fixable problem with the mapping or the table (exit code 2)."""

    exit_code = 2


class InputReadError(AdapterError):
    """The input could not be read (exit code 1)."""

    exit_code = 1


@dataclass(frozen=True)
class CityMapping:
    city: str
    input: str | None
    crs: str
    columns: dict
    status_values: dict | None = None
    bbox: tuple | None = None
    max_km_from_median: float | None = None
    max_quarantine_share: float = DEFAULT_MAX_QUARANTINE_SHARE
    read_options: dict = field(default_factory=dict)
    crs_declared: bool = False


@dataclass
class AdapterResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    report: dict
    codes: dict

    @property
    def gates_failed(self) -> list:
        return list(self.report.get("gates_failed", []))


# ---------------------------------------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------------------------------------

def _load_text_mapping(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except FileNotFoundError:
        raise AdapterError(f"mapping file not found: {path}") from None
    except OSError as exc:
        raise AdapterError(f"cannot read mapping file {path}: {exc.strerror or exc}") from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AdapterError(f"mapping file {path} is not UTF-8 text") from None
    try:
        if ext == ".json":
            data = json.loads(text)
        elif ext == ".toml":
            import tomllib
            data = tomllib.loads(text)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise AdapterError("a YAML mapping needs PyYAML (pip install pyyaml); or write the mapping as "
                                   ".json or .toml, which need nothing extra") from None
            data = yaml.safe_load(text)
        else:
            raise AdapterError(f"unsupported mapping extension {ext or '(none)'}: use .json, .toml or .yaml")
    except AdapterError:
        raise
    except Exception as exc:  # noqa: BLE001 - json / toml / yaml raise different types
        raise AdapterError(f"mapping file {path} is malformed: {str(exc).splitlines()[0]}") from None
    if not isinstance(data, dict):
        raise AdapterError(f"mapping file {path} must hold an object/table at the top level")
    return data


def _number(value, name: str, low: float, high: float, allow_none: bool = False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AdapterError(f"mapping key '{name}' must be a number, got {value!r}")
    if not low <= float(value) <= high:
        raise AdapterError(f"mapping key '{name}' must be within [{low}, {high}], got {value!r}")
    return float(value)


def mapping_from_dict(data: dict, base_dir: str | None = None) -> CityMapping:
    """Validate a mapping dictionary (see the module docstring); relative ``input`` paths resolve against ``base_dir``."""
    if not isinstance(data, dict):
        raise AdapterError("the mapping must be an object/table")
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise AdapterError(f"unknown mapping key(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(_TOP_KEYS))}")
    columns = data.get("columns")
    if not isinstance(columns, dict) or not columns:
        raise AdapterError("the mapping needs a 'columns' object that assigns input columns to roles "
                           f"(mandatory: {', '.join(MANDATORY_ROLES)})")
    bad_roles = sorted(set(columns) - set(MANDATORY_ROLES) - set(OPTIONAL_ROLES))
    if bad_roles:
        raise AdapterError(f"unknown column role(s): {', '.join(bad_roles)}. "
                           f"Roles: {', '.join(MANDATORY_ROLES + OPTIONAL_ROLES)}")
    clean_cols: dict = {}
    for role, value in columns.items():
        if value is None or value == "" or value == []:
            continue
        if role == "address" and isinstance(value, list):
            if not all(isinstance(v, str) and v for v in value):
                raise AdapterError("columns.address as a list must hold non-empty column names")
            clean_cols[role] = list(value)
        elif isinstance(value, str):
            clean_cols[role] = value
        else:
            raise AdapterError(f"columns.{role} must be a column name (text), got {value!r}")
    for role in ("address", "record_id"):
        if role not in clean_cols:
            raise AdapterError(f"columns.{role} is mandatory")
    if ("x" in clean_cols) != ("y" in clean_cols):
        raise AdapterError("columns.x and columns.y go together: map both or neither (geometry input)")
    crs_declared = "crs" in data and data["crs"] not in (None, "")
    crs = str(data["crs"]) if crs_declared else "EPSG:4326"
    status_values = data.get("status_values")
    if status_values is not None:
        if not isinstance(status_values, dict) or not set(status_values) <= {"cert", "pcer"} or not status_values:
            raise AdapterError("status_values must be an object with 'cert' and/or 'pcer' lists of accepted values")
        if "status" not in clean_cols:
            raise AdapterError("status_values needs columns.status")
        norm: dict = {}
        for tier, values in status_values.items():
            if not isinstance(values, list) or not values:
                raise AdapterError(f"status_values.{tier} must be a non-empty list")
            norm[tier] = sorted({str(v).strip().lower() for v in values})
        overlap = set(norm.get("cert", [])) & set(norm.get("pcer", []))
        if overlap:
            raise AdapterError(f"status value(s) listed as both cert and pcer: {', '.join(sorted(overlap))}")
        status_values = norm
    elif "status" in clean_cols:
        raise AdapterError("columns.status needs status_values (which values build documents: cert / pcer)")
    bbox = data.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise AdapterError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
        bbox = tuple(_number(v, "bbox", -180.0, 180.0) for v in bbox)
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3] and -90 <= bbox[1] and bbox[3] <= 90):
            raise AdapterError("bbox must be [min_lon, min_lat, max_lon, max_lat] with min < max and valid latitudes")
    max_km = _number(data.get("max_km_from_median"), "max_km_from_median", 0.001, 20000.0, allow_none=True)
    max_q = _number(data.get("max_quarantine_share", DEFAULT_MAX_QUARANTINE_SHARE), "max_quarantine_share", 0.0, 1.0)
    source = data.get("input")
    if source is not None:
        if not isinstance(source, str) or not source:
            raise AdapterError("input must be a file path")
        if base_dir and not os.path.isabs(source) and "://" not in source:
            source = os.path.normpath(os.path.join(base_dir, source))
    read_options = {}
    for key, target in (("format", "fmt"), ("encoding", "encoding"), ("delimiter", "delimiter"),
                        ("sheet", "sheet"), ("header_row", "header_row"), ("layer", "table")):
        if data.get(key) is not None:
            read_options[target] = data[key]
    city = data.get("city") or "sin_nombre"
    if not isinstance(city, str):
        raise AdapterError("city must be text")
    return CityMapping(city=city, input=source, crs=crs, columns=clean_cols, status_values=status_values,
                       bbox=bbox, max_km_from_median=max_km, max_quarantine_share=max_q,
                       read_options=read_options, crs_declared=crs_declared)


def load_mapping(path: str) -> CityMapping:
    """Read and validate a mapping file (.json / .toml / .yaml)."""
    return mapping_from_dict(_load_text_mapping(path), base_dir=os.path.dirname(os.path.abspath(path)))


# ---------------------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------------------

def read_source(mapping: CityMapping, source: str | None = None) -> tuple[pd.DataFrame, str, list[str]]:
    """``(frame, format, warnings)`` of the mapped input, read whole with the any-dataset readers (text as text)."""
    from .io.errors import DatasetError
    from .io.readers import read_table

    path = source or mapping.input
    if not path:
        raise AdapterError("no input: set 'input' in the mapping or pass --input")
    try:
        with read_table(path, **mapping.read_options) as table:
            frames = list(table)
            columns, fmt, warnings = table.columns, table.format, list(table.warnings)
    except DatasetError as exc:
        raise InputReadError(str(exc).splitlines()[0]) from None
    except (OSError, ValueError, UnicodeError) as exc:
        raise InputReadError(f"cannot read {path}: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}") from None
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return frame, fmt, warnings


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------

def _text(value) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        if value.is_integer():
            return str(int(value))
    return re.sub(r"\s+", " ", str(value)).strip()


def repair_mojibake(text: str) -> tuple[str, bool]:
    """Undo UTF-8 text that was decoded as cp1252/latin-1 (``BogotÃ¡`` -> ``Bogotá``); ``(text, changed)``.

    A file that mixes UTF-8 and cp1252 rows is not valid UTF-8, so the reader decodes it as cp1252 and every UTF-8
    accent becomes two characters. Only strings that hold such a pair are touched, and only when the round trip
    decodes cleanly, so genuine cp1252 text is left alone.
    """
    if not text or not _MOJIBAKE.search(text):
        return text, False
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        return fixed, fixed != text
    return text, False


def _coord(series: pd.Series) -> np.ndarray:
    """Numbers of a coordinate column; text with a decimal comma (``3,45``) is accepted, junk becomes NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    text = series.map(_text)
    comma = text.str.fullmatch(r"[-+]?\d+,\d+")
    text = text.where(~comma, text.str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _plausible(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)


#: Margin (degrees) around a CRS's declared area of use before a transformed point counts as implausible.
AREA_MARGIN_DEG = 2.0


def _area(source) -> tuple | None:
    area = getattr(source, "area_of_use", None)
    if area is None or area.west is None:
        return None
    return (area.west - AREA_MARGIN_DEG, area.south - AREA_MARGIN_DEG,
            area.east + AREA_MARGIN_DEG, area.north + AREA_MARGIN_DEG)


def _to_wgs84(x: np.ndarray, y: np.ndarray, crs: str) -> tuple[np.ndarray, np.ndarray, bool, tuple | None]:
    """``(lon, lat, geographic, area)`` of coordinates in ``crs``; raises AdapterError for a bad CRS.

    ``area`` is the CRS's area of use in lon/lat widened by :data:`AREA_MARGIN_DEG` (``None`` when unknown or
    when it wraps the antimeridian): a projected system applied to the wrong numbers usually lands outside it.
    """
    from pyproj import CRS, Transformer
    from pyproj.exceptions import CRSError

    try:
        source = CRS.from_user_input(crs)
    except CRSError:
        raise AdapterError(f"unknown CRS {crs!r}: use an EPSG code such as 'EPSG:4326' (lon/lat) or the projected "
                           "system the coordinates are written in") from None
    area = _area(source)
    if area is not None and area[0] > area[2]:
        area = None
    if source.equals(CRS.from_epsg(4326)) or (source.is_geographic and source.to_epsg() == 4326):
        return x.copy(), y.copy(), True, None
    transformer = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
    ok = np.isfinite(x) & np.isfinite(y)
    lon = np.full(len(x), np.nan)
    lat = np.full(len(x), np.nan)
    if ok.any():
        with np.errstate(all="ignore"):
            lo, la = transformer.transform(x[ok], y[ok], errcheck=False)
        lon[ok], lat[ok] = np.asarray(lo, dtype="float64"), np.asarray(la, dtype="float64")
    return lon, lat, bool(source.is_geographic), area


def _haversine_km(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lam1, phi1, lam0, phi0 = map(np.radians, (lon, lat, lon0, lat0))
    a = np.sin((phi1 - phi0) / 2) ** 2 + np.cos(phi0) * np.cos(phi1) * np.sin((lam1 - lam0) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _crs_hint(x: np.ndarray, y: np.ndarray, crs: str, geographic: bool, bbox: tuple | None = None) -> str:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return ""
    mx, my = float(np.median(x[finite])), float(np.median(y[finite]))
    if geographic and bbox is not None and bbox[0] <= my <= bbox[2] and bbox[1] <= mx <= bbox[3]:
        return " x and y look swapped: map the longitude to columns.x and the latitude to columns.y."
    if geographic and (abs(mx) > 180 or abs(my) > 90):
        return (f" The median coordinate is ({mx:.0f}, {my:.0f}): that looks like a projected system in metres, "
                "declare it with 'crs' (for example 'EPSG:3116' or 'EPSG:9377' in Colombia).")
    if geographic and abs(mx) <= 90 and abs(my) <= 180 and abs(my) > 90:
        return " x and y look swapped: map the longitude to columns.x and the latitude to columns.y."
    if not geographic and abs(mx) <= 180 and abs(my) <= 90:
        return (f" The median coordinate is ({mx:.4f}, {my:.4f}): that looks like longitude/latitude, so the CRS "
                "is probably 'EPSG:4326'.")
    return f" Check that {crs} is really the system the coordinates are written in."


# ---------------------------------------------------------------------------------------------------------
# adapt
# ---------------------------------------------------------------------------------------------------------

def _codes(names: list[str], width_min: int = 2) -> dict:
    uniq = sorted({n for n in names if n})
    width = max(width_min, len(str(len(uniq))))
    return {name: f"{i + 1:0{width}d}" for i, name in enumerate(uniq)}


def adapt(frame: pd.DataFrame, mapping: CityMapping, fmt: str | None = None,
          reader_warnings: list[str] | None = None) -> AdapterResult:
    """Validate and convert ``frame`` (see the module docstring). Raises :class:`AdapterError` for file-level problems."""
    cols = dict(mapping.columns)
    warnings = list(reader_warnings or [])
    n = int(len(frame))
    if n == 0:
        raise AdapterError("the input has no data rows (an empty file or a header only): nothing to index")
    crs = mapping.crs
    geometry_input = "x" not in cols
    if geometry_input:
        if not {"_lon", "_lat"} <= set(frame.columns):
            raise AdapterError("columns.x and columns.y are mandatory for a table without geometry "
                               f"(format {fmt or 'unknown'}); map the coordinate columns and declare 'crs'")
        cols["x"], cols["y"] = "_lon", "_lat"
        if fmt == "shp" and any("no .prj" in w for w in warnings):
            if not mapping.crs_declared:
                raise AdapterError("the shapefile has no .prj: declare the CRS of its coordinates with 'crs'")
        elif fmt in ("shp", "gpkg"):
            if mapping.crs_declared and mapping.crs.upper() not in ("EPSG:4326", "4326"):
                warnings.append(f"crs {mapping.crs} ignored: the {fmt} reader already returns lon/lat (EPSG:4326)")
            crs = "EPSG:4326"
    elif not mapping.crs_declared:
        warnings.append("no 'crs' declared: x / y are read as EPSG:4326 longitude / latitude")
    wanted: list[str] = []
    for role, value in cols.items():
        wanted.extend(value if isinstance(value, list) else [value])
    missing = [c for c in dict.fromkeys(wanted) if c not in frame.columns]
    if missing:
        available = ", ".join(str(c) for c in frame.columns[:40])
        raise AdapterError(f"column(s) not found in the input: {', '.join(missing)}. Available columns: {available}")

    frame = frame.reset_index(drop=True)
    repaired = 0

    def text_col(name: str) -> list[str]:
        nonlocal repaired
        out = []
        for value in frame[name].tolist():
            t, changed = repair_mojibake(_text(value))
            repaired += changed
            out.append(t)
        return out

    addr_cols = cols["address"] if isinstance(cols["address"], list) else [cols["address"]]
    parts = [text_col(c) for c in addr_cols]
    address = [" ".join(p for p in row if p) for row in zip(*parts)]
    record_id = [_text(v) for v in frame[cols["record_id"]].tolist()]
    optional = {role: (text_col(cols[role]) if role in cols else [""] * n) for role in OPTIONAL_ROLES}
    optional["parcel_id"] = [re.sub(r"\s+", "", v) for v in optional["parcel_id"]]

    x, y = _coord(frame[cols["x"]]), _coord(frame[cols["y"]])
    lon, lat, geographic, area = _to_wgs84(x, y, crs)
    has_xy = np.isfinite(x) & np.isfinite(y)
    plausible = _plausible(lon, lat)
    if area is not None:
        plausible &= (lon >= area[0]) & (lon <= area[2]) & (lat >= area[1]) & (lat <= area[3])
    if has_xy.any() and plausible[has_xy].mean() < MIN_PLAUSIBLE_SHARE:
        raise AdapterError(
            f"{int((~plausible[has_xy]).sum())} of {int(has_xy.sum())} rows with coordinates fall outside valid "
            f"longitude/latitude (or outside the area of use of {crs}) after reading them as {crs}." + _crs_hint(x, y, crs, geographic))
    if mapping.bbox is not None and plausible.any():
        b = mapping.bbox
        inside = plausible & (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
        if inside[plausible].mean() < MIN_PLAUSIBLE_SHARE:
            raise AdapterError(
                f"only {int(inside.sum())} of {int(plausible.sum())} rows fall inside the declared bbox {list(b)} "
                f"after reading the coordinates as {crs}." + _crs_hint(x, y, crs, geographic, mapping.bbox))
    else:
        inside = plausible
        if plausible.any() and abs(float(np.median(lat[plausible]))) > 60:
            warnings.append(f"the median latitude is {float(np.median(lat[plausible])):.2f}: check that columns.x is the "
                            "longitude and columns.y the latitude (declare a 'bbox' to enforce the city extent)")

    far = np.zeros(n, dtype=bool)
    center = None
    if mapping.max_km_from_median is not None and inside.any():
        center = (float(np.median(lon[inside])), float(np.median(lat[inside])))
        dist = np.full(n, np.inf)
        dist[inside] = _haversine_km(lon[inside], lat[inside], *center)
        far = inside & (dist > mapping.max_km_from_median)

    tiers = ["cert"] * n
    status_ok = np.ones(n, dtype=bool)
    if mapping.status_values is not None:
        lookup = {v: tier for tier, values in mapping.status_values.items() for v in values}
        for i, raw in enumerate(optional["status"]):
            tier = lookup.get(raw.strip().lower())
            if tier is None:
                status_ok[i] = False
            else:
                tiers[i] = tier

    reasons: list[str | None] = [None] * n
    seen_ids: set = set()
    seen_rows: set = set()
    for i in range(n):
        if not record_id[i]:
            reasons[i] = "missing_record_id"
            continue
        if record_id[i] in seen_ids:
            reasons[i] = "duplicate_record_id"
            continue
        seen_ids.add(record_id[i])
        if not re.search(r"[0-9A-Za-z]", address[i]):
            reasons[i] = "empty_address"
        elif not has_xy[i]:
            reasons[i] = "missing_coordinates"
        elif not plausible[i]:
            reasons[i] = "coordinates_out_of_range"
        elif not inside[i]:
            reasons[i] = "outside_bbox"
        elif far[i]:
            reasons[i] = "far_from_median"
        elif not status_ok[i]:
            reasons[i] = "status_not_accepted"
        else:
            row_key = (address[i].upper(), round(float(lon[i]), 6), round(float(lat[i]), 6),
                       optional["parcel_id"][i], optional["unit"][i].upper())
            if row_key in seen_rows:
                reasons[i] = "duplicate_row"
            else:
                seen_rows.add(row_key)

    keep = np.array([r is None for r in reasons], dtype=bool)
    district_codes = _codes([optional["district"][i] for i in range(n) if keep[i]])
    hood_names = sorted({(optional["district"][i], optional["neighborhood"][i]) for i in range(n)
                         if keep[i] and optional["neighborhood"][i]})
    hood_codes: dict = {}
    per_district: dict = {}
    for district, hood in hood_names:
        per_district.setdefault(district, []).append(hood)
    for district, hoods in per_district.items():
        local = _codes(hoods)
        prefix = district_codes.get(district, "00")
        for hood, code in local.items():
            hood_codes[(district, hood)] = prefix + code

    idx = np.flatnonzero(keep)
    clean = pd.DataFrame({
        "gid": (idx + 1).astype("int64"),
        "nddirecc": pd.Series([address[i] for i in idx], dtype="str"),
        "ndnpn": pd.Series([optional["parcel_id"][i] for i in idx], dtype="str"),
        "ndidestado": pd.Series([tiers[i] for i in idx], dtype="str"),
        "lon": lon[idx].astype("float64"),
        "lat": lat[idx].astype("float64"),
        "ndinterior": pd.Series([optional["unit"][i] for i in idx], dtype="str"),
        "ndcomuna": pd.Series([district_codes.get(optional["district"][i], "") for i in idx], dtype="str"),
        "ndidbarrio": pd.Series([hood_codes.get((optional["district"][i], optional["neighborhood"][i]), "")
                                 for i in idx], dtype="str"),
        "ndmanzana": pd.Series([optional["block"][i] for i in idx], dtype="str"),
        "source_record_id": pd.Series([record_id[i] for i in idx], dtype="str"),
    })
    qidx = np.flatnonzero(~keep)
    quarantine = pd.DataFrame({
        "row_number": (qidx + 1).astype("int64"),
        "record_id": pd.Series([record_id[i] for i in qidx], dtype="str"),
        "reason": pd.Series([reasons[i] for i in qidx], dtype="str"),
        "address": pd.Series([address[i] for i in qidx], dtype="str"),
        "x": x[qidx], "y": y[qidx],
    })
    by_reason = {r: int(sum(1 for v in reasons if v == r)) for r in QUARANTINE_REASONS}
    share = float(len(qidx) / n)
    gates_failed = []
    if share > mapping.max_quarantine_share:
        gates_failed.append("quarantine_share")
    if not len(clean):
        gates_failed.append("no_rows_left")

    npn = [optional["parcel_id"][i] for i in idx]
    report = {
        "schema": 1,
        "city": mapping.city,
        "input_format": fmt,
        "input_rows": n,
        "kept_rows": int(len(clean)),
        "quarantined_rows": int(len(qidx)),
        "quarantine_share": share,
        "max_quarantine_share": mapping.max_quarantine_share,
        "by_reason": by_reason,
        "gates_failed": gates_failed,
        "crs": {"declared": mapping.crs if mapping.crs_declared else None, "used": crs,
                "output": "EPSG:4326 (lon/lat)"},
        "center_lon_lat": center,
        "bbox_out": ([float(clean["lon"].min()), float(clean["lat"].min()), float(clean["lon"].max()),
                      float(clean["lat"].max())] if len(clean) else None),
        "by_tier": {t: int((clean["ndidestado"] == t).sum()) for t in ("cert", "pcer")},
        "codes": {"districts": len(district_codes), "neighborhoods": len(hood_codes)},
        "parcel_id": {
            "mapped": "parcel_id" in cols,
            "blank": int(sum(1 for v in npn if not v)),
            "npn_30_digits": int(sum(1 for v in npn if len(v) == 30 and v.isdigit())),
            "other_17_or_more_chars": int(sum(1 for v in npn if len(v) >= 17 and not (len(v) == 30 and v.isdigit()))),
        },
        "mojibake_repaired_cells": int(repaired),
        "columns": {role: value for role, value in cols.items()},
        "warnings": warnings,
    }
    if report["parcel_id"]["other_17_or_more_chars"]:
        warnings.append(f"{report['parcel_id']['other_17_or_more_chars']} parcel ids have 17+ characters but are not "
                        "30-digit predial numbers: the index builder takes their first 17 characters as the block")
    codes = {
        "districts": {code: name for name, code in district_codes.items()},
        "neighborhoods": {code: {"district": d, "name": h} for (d, h), code in hood_codes.items()},
    }
    return AdapterResult(clean=clean, quarantine=quarantine, report=report, codes=codes)
