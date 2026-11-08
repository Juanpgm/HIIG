"""Output contract of ``service.normalize_strict`` as pydantic models, plus a conservative validate/correct layer.

The vocabularies below mirror the real engine (``service.PUBLIC_STATUSES``, ``_TIER_ORDER``, ``_PRECISION_ORDER``,
``_NIVEL_OF_TIER``, ``POSITION_RADIUS_M``); a test pins them so a drift fails loudly. This module imports neither
``service`` nor torch, so it stays cheap to import and ``service`` can import it without a cycle.

Modes of :func:`validate_and_correct`:

``off``      nothing is looked at; the input is returned as is.
``report``   violations are counted per rule; the output is NOT touched (same object returned).
``correct``  only CONSERVATIVE corrections are applied, never one that makes a row more optimistic: confidence
             clamped to [0, 1] (NaN / non-numeric -> null), whitespace stripped, an invalid predial nulled,
             invalid coordinates nulled with a note in ``motivo``, ``nivel_precision`` / ``position_tier`` lowered
             to what the available data supports, a source without text nulled, text of a non-OK ``high``
             fidelity-risk row removed. Violations that cannot be fixed conservatively (``estado`` /
             ``position_tier`` outside the vocabulary) are reported and left alone.

A bad row never raises. Only a missing REQUIRED column (or an unknown mode) raises.

``NaN`` handling: a float64 DataFrame column cannot tell ``None`` from ``NaN`` (a non-OK row's missing confidence
is stored as NaN), so there NaN means "null". In a dict row or an object column a float NaN in a confidence field
is a violation (``confidence_nan``) and becomes null.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

__all__ = [
    "CALI_BBOX",
    "Correction",
    "BANDAS",
    "ESTADOS",
    "FUENTES",
    "MODES",
    "NIVELES",
    "NIVEL_OF_TIER",
    "NormalizedRecord",
    "OutputSchemaError",
    "POSITION_RADIUS_M",
    "POSITION_TIERS",
    "REQUIRED_COLUMNS",
    "RULES",
    "ValidationReport",
    "validate_and_correct",
]

MODES = ("off", "report", "correct")

#: Bounding box of Cali (degrees, inclusive).
CALI_BBOX = {"lon_min": -76.65, "lon_max": -76.40, "lat_min": 3.30, "lat_max": 3.55}

#: ``estado`` values a caller can ever see.
ESTADOS = ("OK", "SIN_MATCH", "NO_PARSEABLE")
#: Position tiers from the most to the least precise.
POSITION_TIERS = ("predio", "lote", "tramo", "manzana", "via", "none")
#: ``nivel_precision`` values from the least to the most precise (``None`` on non-OK rows).
NIVELES = ("direccion", "via", "esquina", "manzana", "predio")
#: Most precise ``nivel_precision`` each tier vouches for (CHANGELOG table).
NIVEL_OF_TIER = {"predio": "predio", "lote": "manzana", "tramo": "manzana", "manzana": "manzana", "via": "via"}
#: Geometric error radius in metres of each tier (``service.POSITION_RADIUS_M``).
POSITION_RADIUS_M = {"predio": 15, "lote": 44, "manzana": 56, "tramo": 23, "via": 21}

#: Columns the rules read; their absence is an error (everything else is optional or ignored).
REQUIRED_COLUMNS = (
    "estado", "motivo", "direccion_normalizada", "fuente_normalizacion", "numero_predial_nacional", "manzana",
    "lat", "lon", "confianza", "nivel_precision", "position_tier", "fidelity_risk",
)
#: Probability-like fields in [0, 1]. The last three exist only in calibrated mode (``cali_address.decision``).
CONFIDENCE_FIELDS = (
    "confianza", "confiabilidad", "confiabilidad_manzana", "confiabilidad_calibrada", "p_predio_cal", "p_manzana_cal",
)
#: Text columns stripped of surrounding whitespace. ``direccion_entrada`` is the echoed input and is never touched.
STRIP_FIELDS = (
    "estado", "motivo", "direccion_normalizada", "fuente_normalizacion", "numero_predial_nacional", "manzana",
    "nivel_precision", "position_tier", "position_source", "fidelity_risk", "direccion_catastral",
    "complemento_normalizado", "barrio_vereda", "comuna_corregimiento", "catalog_status",
)

#: ``fuente`` values of a cadastral document (optional output column, see ``service.normalize_strict``).
FUENTES = ("nomenclatura_cert", "nomenclatura_pcer", "catastro")

#: ``banda_confiabilidad`` values of the calibrated decision layer (optional column, null on non-OK rows).
BANDAS = ("alta", "media", "baja")

NPN_LENGTH = 30
NOTE_COORDS_NULLED = "validacion: coordenadas invalidas anuladas"

#: Every rule id, in the order corrections are applied.
RULES = (
    "whitespace", "estado_vocab", "nivel_vocab", "position_tier_vocab", "confidence_nan", "confidence_range",
    "npn_format", "coords_partial", "coords_out_of_bbox", "high_risk_text", "text_source",
    "tier_nivel_incoherent", "ok_without_support", "fuente_vocab", "banda_vocab",
)


class OutputSchemaError(ValueError):
    """The input cannot be validated at all (missing required columns)."""


class Correction(BaseModel):
    """One field change made by ``mode='correct'``. ``row`` is the frame's index label (``None`` for a dict row)."""

    field: str
    before: Any = None
    after: Any = None
    rule: str
    row: Any = None


class ValidationReport(BaseModel):
    mode: str
    rows: int = 0
    rows_with_violations: int = 0
    rows_corrected: int = 0
    violations: dict[str, int] = Field(default_factory=dict)
    corrected: dict[str, int] = Field(default_factory=dict)
    corrections: list[Correction] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-friendly summary (the per-field correction list is reduced to its length)."""
        return {
            "mode": self.mode, "rows": self.rows, "rows_with_violations": self.rows_with_violations,
            "rows_corrected": self.rows_corrected, "violations": dict(self.violations),
            "corrected": dict(self.corrected), "n_corrections": len(self.corrections),
        }


# --------------------------------------------------------------------------- value helpers
_BAD = object()


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(isinstance(value, float) and math.isnan(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic scalar
        return False


def _blank(value: Any) -> bool:
    return _is_null(value) or (isinstance(value, str) and not value.strip())


def _as_float(value: Any) -> Any:
    """``float`` for a real number (NaN / inf included), ``_BAD`` for anything else (bools, strings...)."""
    if isinstance(value, bool):
        return _BAD
    try:
        import numbers

        if isinstance(value, numbers.Real):
            return float(value)
    except (TypeError, ValueError, OverflowError):  # pragma: no cover
        pass
    return _BAD


def _same(a: Any, b: Any) -> bool:
    if _is_null(a) and _is_null(b):
        return True
    try:
        return bool(a == b)
    except (TypeError, ValueError):  # pragma: no cover
        return False


def _npn_valid(value: Any) -> bool:
    return isinstance(value, str) and len(value) == NPN_LENGTH and value.isascii() and value.isdigit()


def _nivel_rank(nivel: str) -> int:
    return NIVELES.index(nivel)


def _tier_index(tier: str) -> int:
    return POSITION_TIERS.index(tier)


def _support(cur: dict) -> tuple[str, str]:
    """``(most precise nivel, most precise tier)`` the available cadastral codes vouch for."""
    if _npn_valid(cur.get("numero_predial_nacional")):
        return "predio", "predio"
    if not _blank(cur.get("manzana")):
        return "manzana", "lote"
    return "via", "via"


# --------------------------------------------------------------------------- detectors: rule -> offending fields
def _d_estado_vocab(cur: dict) -> list[str]:
    estado = cur.get("estado")
    return [] if isinstance(estado, str) and estado in ESTADOS else ["estado"]


def _d_nivel_vocab(cur: dict) -> list[str]:
    nivel = cur.get("nivel_precision")
    return [] if _is_null(nivel) or (isinstance(nivel, str) and nivel in NIVELES) else ["nivel_precision"]


def _d_tier_vocab(cur: dict) -> list[str]:
    tier = cur.get("position_tier")
    return [] if isinstance(tier, str) and tier in POSITION_TIERS else ["position_tier"]


def _d_fuente_vocab(cur: dict) -> list[str]:
    # Only present when the cadastral docs carry the column; a null is fine (no selected document). Report-only.
    fuente = cur.get("fuente")
    return [] if _is_null(fuente) or (isinstance(fuente, str) and fuente in FUENTES) else ["fuente"]


def _d_banda_vocab(cur: dict) -> list[str]:
    # Only present in calibrated mode; a null is fine (non-OK rows). Report-only.
    banda = cur.get("banda_confiabilidad")
    return [] if _is_null(banda) or (isinstance(banda, str) and banda in BANDAS) else ["banda_confiabilidad"]


def _d_conf_nan(cur: dict) -> list[str]:
    out = []
    for field in CONFIDENCE_FIELDS:
        value = cur.get(field)
        if value is None or value is pd.NA:
            continue
        number = _as_float(value)
        if number is _BAD or math.isnan(number):
            out.append(field)
    return out


def _d_conf_range(cur: dict) -> list[str]:
    out = []
    for field in CONFIDENCE_FIELDS:
        number = _as_float(cur.get(field)) if cur.get(field) is not None else _BAD
        if number is not _BAD and not math.isnan(number) and not 0.0 <= number <= 1.0:
            out.append(field)
    return out


def _d_npn(cur: dict) -> list[str]:
    value = cur.get("numero_predial_nacional")
    if _is_null(value) or (isinstance(value, str) and value == ""):
        return []
    return [] if _npn_valid(value) else ["numero_predial_nacional"]


def _d_coords_partial(cur: dict) -> list[str]:
    lat_null, lon_null = _is_null(cur.get("lat")), _is_null(cur.get("lon"))
    return ["lat"] if lat_null != lon_null else []


def _d_coords_oob(cur: dict) -> list[str]:
    if _is_null(cur.get("lat")) or _is_null(cur.get("lon")):
        return []
    lat, lon = _as_float(cur.get("lat")), _as_float(cur.get("lon"))
    if lat is _BAD or lon is _BAD:
        return ["lat"]
    inside = (
        CALI_BBOX["lat_min"] <= lat <= CALI_BBOX["lat_max"] and CALI_BBOX["lon_min"] <= lon <= CALI_BBOX["lon_max"]
    )
    return [] if inside else ["lat"]


def _d_high_risk(cur: dict) -> list[str]:
    if cur.get("estado") == "OK" or cur.get("fidelity_risk") != "high" or _blank(cur.get("direccion_normalizada")):
        return []
    return ["direccion_normalizada"]


def _d_text_source(cur: dict) -> list[str]:
    return ["fuente_normalizacion"] if _blank(cur.get("direccion_normalizada")) and not _is_null(
        cur.get("fuente_normalizacion")) else []


def _d_tier_nivel(cur: dict) -> list[str]:
    tier, nivel = cur.get("position_tier"), cur.get("nivel_precision")
    if not (isinstance(tier, str) and tier in NIVEL_OF_TIER and isinstance(nivel, str) and nivel in NIVELES):
        return []
    return ["nivel_precision"] if _nivel_rank(nivel) > _nivel_rank(NIVEL_OF_TIER[tier]) else []


def _d_ok_support(cur: dict) -> list[str]:
    if cur.get("estado") != "OK":
        return []
    nivel_cap, tier_cap = _support(cur)
    nivel, tier = cur.get("nivel_precision"), cur.get("position_tier")
    too_precise_nivel = isinstance(nivel, str) and nivel in NIVELES and _nivel_rank(nivel) > _nivel_rank(nivel_cap)
    too_precise_tier = isinstance(tier, str) and tier in POSITION_TIERS and _tier_index(tier) < _tier_index(tier_cap)
    return ["nivel_precision"] if too_precise_nivel or too_precise_tier else []


# --------------------------------------------------------------------------- fixers: mutate ``cur`` and emit corrections
def _set(cur: dict, field: str, after: Any, rule: str, emit) -> None:
    before = cur.get(field)
    # a NaN / <NA> replaced by None is a real change (the cell is rewritten), a None kept as None is not
    if not _same(before, after) or (before is not None and after is None):
        emit(field, before, after, rule)
    cur[field] = after


def _f_nivel_vocab(cur, emit, fields):
    _set(cur, "nivel_precision", "direccion" if cur.get("estado") == "OK" else None, "nivel_vocab", emit)


def _f_conf_nan(cur, emit, fields):
    for field in fields:
        _set(cur, field, None, "confidence_nan", emit)


def _f_conf_range(cur, emit, fields):
    for field in fields:
        number = _as_float(cur[field])
        _set(cur, field, 1.0 if number > 1.0 else 0.0, "confidence_range", emit)


def _f_npn(cur, emit, fields):
    _set(cur, "numero_predial_nacional", None, "npn_format", emit)


def _null_coordinates(rule: str):
    def fix(cur, emit, fields):
        _set(cur, "lat", None, rule, emit)
        _set(cur, "lon", None, rule, emit)
        if "position_source" in cur:
            _set(cur, "position_source", "none", rule, emit)
        motivo = cur.get("motivo")
        _set(cur, "motivo", NOTE_COORDS_NULLED if _blank(motivo) else f"{motivo}; {NOTE_COORDS_NULLED}", rule, emit)
    return fix


def _f_high_risk(cur, emit, fields):
    _set(cur, "direccion_normalizada", None, "high_risk_text", emit)
    _set(cur, "fuente_normalizacion", None, "high_risk_text", emit)


def _f_text_source(cur, emit, fields):
    _set(cur, "fuente_normalizacion", None, "text_source", emit)


def _f_tier_nivel(cur, emit, fields):
    _set(cur, "nivel_precision", NIVEL_OF_TIER[cur["position_tier"]], "tier_nivel_incoherent", emit)


def _f_ok_support(cur, emit, fields):
    nivel_cap, tier_cap = _support(cur)
    nivel, tier = cur.get("nivel_precision"), cur.get("position_tier")
    if isinstance(nivel, str) and nivel in NIVELES and _nivel_rank(nivel) > _nivel_rank(nivel_cap):
        _set(cur, "nivel_precision", nivel_cap, "ok_without_support", emit)
    if isinstance(tier, str) and tier in POSITION_TIERS and _tier_index(tier) < _tier_index(tier_cap):
        _set(cur, "position_tier", tier_cap, "ok_without_support", emit)
        if "position_radius_m" in cur:
            _set(cur, "position_radius_m", float(POSITION_RADIUS_M[tier_cap]), "ok_without_support", emit)


#: ``(rule, detector, fixer or None)`` in application order.
_PIPELINE = (
    ("estado_vocab", _d_estado_vocab, None),
    ("nivel_vocab", _d_nivel_vocab, _f_nivel_vocab),
    ("position_tier_vocab", _d_tier_vocab, None),
    ("fuente_vocab", _d_fuente_vocab, None),
    ("banda_vocab", _d_banda_vocab, None),
    ("confidence_nan", _d_conf_nan, _f_conf_nan),
    ("confidence_range", _d_conf_range, _f_conf_range),
    ("npn_format", _d_npn, _f_npn),
    ("coords_partial", _d_coords_partial, _null_coordinates("coords_partial")),
    ("coords_out_of_bbox", _d_coords_oob, _null_coordinates("coords_out_of_bbox")),
    ("high_risk_text", _d_high_risk, _f_high_risk),
    ("text_source", _d_text_source, _f_text_source),
    ("tier_nivel_incoherent", _d_tier_nivel, _f_tier_nivel),
    ("ok_without_support", _d_ok_support, _f_ok_support),
)


def _canonicalize(row: dict, nan_is_null: frozenset) -> tuple[dict, list[str]]:
    """Copy of ``row`` with stripped text (and NaN -> None in float columns); the stripped field names."""
    canon = dict(row)
    stripped: list[str] = []
    for field in STRIP_FIELDS:
        value = canon.get(field)
        if isinstance(value, str) and value != value.strip():
            canon[field] = value.strip()
            stripped.append(field)
    for field in nan_is_null:
        if field in canon and _is_null(canon[field]):
            canon[field] = None
    return canon, stripped


def _violations(canon: dict, stripped: list[str]) -> list[tuple[str, str]]:
    found = [("whitespace", field) for field in stripped]
    for rule, detect, _fix in _PIPELINE:
        found.extend((rule, field) for field in detect(canon))
    return found


# --------------------------------------------------------------------------- the pydantic schema
class NormalizedRecord(BaseModel):
    """One output row of ``normalize_strict``. Validation fails (all rules at once) when a rule is violated.

    Field names are the REAL column names (``confianza`` is the confidence). Extra columns are ignored.
    """

    model_config = ConfigDict(extra="ignore")

    estado: str
    motivo: str | None = None
    direccion_normalizada: str | None = None
    fuente_normalizacion: str | None = None
    numero_predial_nacional: str | None = None
    manzana: str | int | None = None
    lat: float | None = None
    lon: float | None = None
    confianza: float | None = None
    confiabilidad: float | None = None
    confiabilidad_manzana: float | None = None
    nivel_precision: str | None = None
    position_tier: str | None = None
    position_radius_m: float | None = None
    position_source: str | None = None
    fidelity_risk: str | None = None
    #: Optional: only when the cadastral docs carry them (nomenclature indexes).
    fuente: str | None = None
    estado_nomenclatura: str | None = None
    #: Optional: only in calibrated mode.
    confiabilidad_calibrada: float | None = None
    banda_confiabilidad: str | None = None
    motivos_confiabilidad: str | None = None
    p_predio_cal: float | None = None
    p_manzana_cal: float | None = None

    @model_validator(mode="wrap")
    @classmethod
    def _check_rules(cls, value: Any, handler) -> Any:
        if isinstance(value, dict):
            canon, stripped = _canonicalize(value, frozenset())
            found = _violations(canon, stripped)
            if found:
                raise ValidationError.from_exception_data(
                    cls.__name__,
                    [{"type": PydanticCustomError(rule, f"output rule '{rule}' violated"), "loc": (field,),
                      "input": value.get(field)} for rule, field in found],
                )
            value = {k: (None if _is_null(v) else v) for k, v in canon.items()}
        return handler(value)


# --------------------------------------------------------------------------- driver
def _missing(columns) -> list[str]:
    present = set(columns)
    return [c for c in REQUIRED_COLUMNS if c not in present]


def _process(row: dict, correct: bool, nan_is_null: frozenset, label: Any):
    """``(violations, corrected dict or None, corrections, fixed counter)`` for one row."""
    canon, stripped = _canonicalize(row, nan_is_null)
    found = _violations(canon, stripped)
    if not correct or not found:
        return found, None, [], Counter()
    corrections: list[Correction] = []
    fixed: Counter = Counter()

    def emit(field, before, after, rule):
        corrections.append(Correction(field=field, before=before, after=after, rule=rule, row=label))

    for field in stripped:
        emit(field, row[field], canon[field], "whitespace")
        fixed["whitespace"] += 1
    cur = dict(canon)
    for rule, detect, fix in _PIPELINE:
        if fix is None:
            continue
        fields = detect(cur)
        if fields:
            fix(cur, emit, fields)
            fixed[rule] += len(fields)
    return found, cur, corrections, fixed


def validate_and_correct(data, mode: str = "report"):
    """Validate (and optionally correct) a dict row or a DataFrame of ``normalize_strict`` output.

    Returns ``(output, ValidationReport)``. ``off`` and ``report`` return ``data`` itself; ``correct`` returns a
    corrected copy (the input is never mutated). Raises :class:`OutputSchemaError` for missing REQUIRED columns
    and ``ValueError`` for an unknown mode; a bad row never raises.
    """
    if mode not in MODES:
        raise ValueError(f"unknown validate_outputs mode {mode!r}; valid modes: {', '.join(MODES)}")
    if mode == "off":
        return data, ValidationReport(mode=mode)
    correct = mode == "correct"
    if isinstance(data, dict):
        absent = _missing(data)
        if absent:
            raise OutputSchemaError(f"missing required output keys: {', '.join(absent)}")
        found, cur, corrections, fixed = _process(data, correct, frozenset(), None)
        report = ValidationReport(
            mode=mode, rows=1, rows_with_violations=int(bool(found)), rows_corrected=int(bool(corrections)),
            violations=dict(Counter(rule for rule, _ in found)), corrected=dict(fixed), corrections=corrections,
        )
        if not correct or cur is None:
            return data, report
        out = dict(data)
        for c in corrections:
            out[c.field] = cur[c.field]
        return out, report
    if isinstance(data, pd.DataFrame):
        return _validate_frame(data, mode)
    raise TypeError(f"validate_and_correct expects a dict row or a DataFrame, got {type(data).__name__}")


def _validate_frame(frame: pd.DataFrame, mode: str):
    absent = _missing(frame.columns)
    if absent:
        raise OutputSchemaError(f"missing required output columns: {', '.join(absent)}")
    correct = mode == "correct"
    used = [c for c in frame.columns if c in set(REQUIRED_COLUMNS) | set(CONFIDENCE_FIELDS) | set(STRIP_FIELDS)
            | {"position_radius_m", "position_source", "fuente", "banda_confiabilidad"}]
    nan_is_null = frozenset(
        c for c in (*CONFIDENCE_FIELDS, "position_radius_m") if c in frame.columns
        and pd.api.types.is_float_dtype(frame[c].dtype)
    )
    violations: Counter = Counter()
    corrected: Counter = Counter()
    corrections: list[Correction] = []
    updates: list[tuple[int, Any, dict]] = []
    rows_with = rows_fixed = 0
    labels = frame.index
    for pos, record in enumerate(frame[used].to_dict("records")):
        found, cur, row_corrections, fixed = _process(record, correct, nan_is_null, labels[pos])
        if found:
            rows_with += 1
            violations.update(rule for rule, _ in found)
        if row_corrections:
            rows_fixed += 1
            corrections.extend(row_corrections)
            corrected.update(fixed)
            updates.append((pos, cur, {c.field for c in row_corrections}))
    report = ValidationReport(
        mode=mode, rows=len(frame), rows_with_violations=rows_with, rows_corrected=rows_fixed,
        violations=dict(violations), corrected=dict(corrected), corrections=corrections,
    )
    if not correct or not updates:
        return (frame if not correct else frame.copy()), report
    out = frame.copy()
    for pos, cur, fields in updates:
        for field in fields:
            out.iat[pos, out.columns.get_loc(field)] = cur[field]
    return out, report
