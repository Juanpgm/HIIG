"""Shared normalization engine: batching, scoring, gating, table IO.

This module holds everything that used to live inside ``scripts/normalizar.py``
so the CLI and the HTTP service in ``cali_address.api`` run the exact same code
path. The CLI is now only argument parsing plus stdout/stderr formatting.

Strictness contract
-------------------
Unlike ``AddressNormalizer.normalize_batch`` (which always returns the best
candidate, even for junk input), :func:`normalize_strict` is strict: a row only
receives a cadastral match when every rule passes. Everything else is reported
with an explicit status and NO candidate fields, so a low-quality guess can
never be mistaken for a real match.

Statuses
--------
OK                  parsed, matched above the tuned confidence cutoff, inside the
                    detected place (when one was detected), and the matched
                    cadastral address agrees structurally with the input
NO_PARSEABLE        the address grammar could not parse the text (the detected
                    place columns are still filled)
SIN_MATCH           no cadastral record can be assigned. ``motivo`` says why:
                    blank/placeholder input, confidence below the cutoff, a
                    structural rule violated by the best candidate (via type /
                    via number / cross number / plate / low agreement), or every
                    candidate outside the detected barrio / comuna / corregimiento

Table reading
-------------
:func:`read_table` accepts xlsx/xls, csv, GeoJSON and a zipped shapefile and
returns a plain ``pandas.DataFrame``. Format, warnings and any geometry columns
it derived are attached to ``df.attrs`` (see :data:`ATTRS_KEYS`).

Column ambiguity
----------------
:func:`detect_address_column` scores every column against a Spanish/English
synonym set; :func:`resolve_address_column` turns that into either a single
answer or an :class:`AddressColumnError` carrying the ranked candidates, which
is how both the CLI and the API ask the caller to pick a field.
"""

from __future__ import annotations

import io
import math
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from .gazetteer import (
    ZONE_BUFFER_M,
    Candidate,
    Detection,
    Gazetteer,
    gate_candidates,
)
from .inference import (  # noqa: F401  (SOFT_RULE_NAMES is re-exported: public API of this module)
    ONE_SIDED_FIELDS,
    PLATE_TRAMO_WINDOW,
    SOFT_RULE_NAMES,
    AddressNormalizer,
    base_mismatches,
    normalize_soft_rules,
    validate_max_soft,
)
from .tables import (  # noqa: F401  (re-exported: the public API of this module)
    AddressColumnError,
    KeepColumnsError,
    TableFormatError,
    ADDRESS_SYNONYMS,
    _MIN_SUBSTRING_LEN,
    CANDIDATE_SCORE_FLOOR,
    AUTO_SCORE_FLOOR,
    AUTO_MARGIN,
    _normalize_column_name,
    _score_column,
    detect_address_column,
    resolve_address_column,
    _MAX_LABEL_LEN,
    _looks_like_label,
    guess_header_row,
    SUPPORTED_EXTENSIONS,
    ATTRS_KEYS,
    _HEADER_SCAN_ROWS,
    _GEOMETRY_COLUMNS,
    _finalize,
    _read_excel_bytes,
    _sniff_delimiter,
    _decode_csv,
    _csv_rows,
    _pad,
    _width_stable_from,
    _first_stable_width_row,
    _read_csv_bytes,
    _centroid,
    _read_geojson_bytes,
    _SHAPEFILE_REQUIRED,
    _read_shapefile_zip,
    _prj_transformer,
    read_table,
)
from .parser import QUADRANT_SHORT, _QUADRANT_WORDS, canonical, merge_complement, parse_address, render_complement
from .parser import rule_canonical as _rule_canonical
from .catalog import decide as _catalog_decide
from .catalog import door_catalogued as _door_catalogued
from .catalog import keeps_rule_text as _keeps_rule_text
from .catalog import keys_for as _catalog_keys_for
from .interp import extrap_radius_m as _extrap_radius_m
from .interp import index_for as _interp_index_for
from .interp import street_key as _street_key
from .interp import tramo_radius_m as _tramo_radius_m
from .refine import refine_address
from .fidelity import assess_fidelity

__all__ = [
    "AddressColumnError",
    "KeepColumnsError",
    "TableFormatError",
    "ADDRESS_SYNONYMS",
    "ATTRS_KEYS",
    "CLI_BARRIO_BUFFER_M",
    "DEFAULT_GATE_ESCALATE",
    "DEFAULT_MIN_STRUCT",
    "EXTRAPOLATED_COORDS_ENABLED",
    "FIDELITY_CAPS_ENABLED",
    "FIDELITY_TIER_CAP",
    "OUTPUT_COLUMNS",
    "POSITION_RADIUS_M",
    "reliability_features",
    "PLACEHOLDERS",
    "STATUS_MAP",
    "SUPPORTED_EXTENSIONS",
    "ZONE_COLUMNS",
    "attach_source_columns",
    "detect_address_column",
    "guess_header_row",
    "is_unspecified",
    "load_gazetteer",
    "normalize_strict",
    "read_input_file",
    "read_table",
    "resolve_address_column",
    "results_to_geojson",
    "results_to_records",
    "summarize",
    "write_csv_bytes",
    "write_output",
    "write_xlsx_bytes",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_BASEMAPS_DIR = os.path.join(PROJECT_ROOT, "basemaps")

#: Default barrio-gate slack of :func:`normalize_strict`, the CLI, the API and the IO
#: pipeline (one source of truth). A 200 m gate withholds many rows whose GPS reference is
#: close, because the barrio names of some sources do not line up with the official polygons;
#: 1000 m + escalation keeps them without raising the share of large errors.
CLI_BARRIO_BUFFER_M = 1000.0

#: Default of the gate escalation (barrio -> comuna -> corregimiento instead of
#: withholding the row), shared like :data:`CLI_BARRIO_BUFFER_M`.
DEFAULT_GATE_ESCALATE = True

#: Minimum weighted structural agreement between the input parse and the matched
#: cadastral parse. Tuned default, shared by the CLI flag and the API form field.
DEFAULT_MIN_STRUCT = 0.6

PLACEHOLDERS = {
    "", "-", "--", ".", "0", "00", "N/A", "NA", "N.A", "N.A.", "NAN", "NONE", "NULL",
    "NO APLICA", "NO REGISTRA", "NO REPORTA", "NO INFORMA", "NINGUNA", "NINGUNO",
    "SIN ESPECIFICAR", "SIN DATO", "SIN DATOS", "SIN DIRECCION", "SIN INFORMACION",
    "S/D", "S.D", "SD", "DESCONOCIDA", "DESCONOCIDO", "NO TIENE", "NO HAY", "X", "XX", "XXX",
}

#: Normalized zone, one value each: the official barrio (or vereda) name and the
#: official comuna (or corregimiento) name. On OK rows they come from the polygon
#: the matched predio sits in; on every other row from the place detected in the
#: text, so a rural or unparseable address still gets a normalized zone.
ZONE_COLUMNS = ["barrio_vereda", "comuna_corregimiento"]

OUTPUT_COLUMNS = [
    "direccion_entrada",
    "estado",
    "motivo",
    "direccion_normalizada",
    "fuente_normalizacion",
    "numero_predial_nacional",
    "manzana",
    "lat",
    "lon",
    "confianza",
    "confiabilidad",
    "confiabilidad_manzana",
    "margen",
    "nivel_precision",
    *ZONE_COLUMNS,
    # The unit part of ``direccion_normalizada`` (``TO F AP 101``), taken from the input.
    "complemento_normalizado",
    # Position tiers (appended last: every earlier column keeps its place).
    "position_tier",
    "position_radius_m",
    "position_source",
    "selected_doc_id",
    "override_rejected",
    # The matched cadastral address as the cadastre writes it (OK rows only). ``direccion_normalizada`` is the
    # cadastral base only for predio / lote rows; elsewhere it is the query's own normalized text.
    "direccion_catastral",
    # Fidelity of ``direccion_normalizada`` with respect to the ORIGINAL raw input (cali_address.fidelity):
    # ``none`` / ``low`` / ``medium`` / ``high`` plus the ``;``-joined flags that explain it.
    "fidelity_risk",
    "fidelity_flags",
    # Catalogue status (cali_address.catalog): ``verified`` / ``ambiguous`` / ``door_not_catalogued`` /
    # ``rejected_mismatch`` / ``unparsed`` / ``below_cutoff``; ``address_verified`` is True exactly for ``verified``.
    "catalog_status",
    "address_verified",
]

#: Geometric error radius in metres (p90, derived from the cadastre itself: no GPS) of each position tier.
#: ``none`` (no position) has no radius. ``predio``: parcel centroid to its farthest vertex (14.8 m); ``lote``:
#: lot centroid to its farthest vertex for multi-unit lots / PH (44.1 m); ``manzana``: parcel centroid to its
#: block centroid (55.9 m); ``via``: leave-one-out linear extrapolation error (20.8 m). ``tramo`` is the
#: radius at the interpolation cap (the (25, 50] m gap bin, 23 m): a ``tramo`` ROW reports the radius of its
#: actual gap (:func:`cali_address.interp.tramo_radius_m`), and an extrapolated ``via`` row that of its
#: distance from the last parcel (:func:`cali_address.interp.extrap_radius_m`).
POSITION_RADIUS_M = {"predio": 15, "lote": 44, "manzana": 56, "tramo": 23, "via": 21}

#: Tiers from the most to the least precise (``none`` = no position at all).
_TIER_ORDER = ("predio", "lote", "tramo", "manzana", "via", "none")

#: ``nivel_precision`` each tier maps to (vocabulary unchanged). ``lote`` withholds the predial and
#: ``tramo`` is not a parcel, so both vouch for the manzana only; ``via`` is an extrapolation.
_NIVEL_OF_TIER = {"predio": "predio", "lote": "manzana", "tramo": "manzana", "manzana": "manzana", "via": "via"}

#: When True, a plate beyond the last known parcel of its street face (tier ``via``) is published
#: with the linearly extrapolated coordinates. DEFAULT FALSE: the tier and radius are reported but
#: the coordinates are withheld, because no parcel of the cadastre stands there.
EXTRAPOLATED_COORDS_ENABLED = False

#: Policy hook: when True, a row whose output drifted from its raw input is capped (never raised): ``high``
#: fidelity risk -> position tier ``via`` (predial cleared, coordinates withheld like any ``via`` row),
#: ``medium`` -> ``manzana`` (predial cleared). Set False to keep the columns but leave the tiers untouched.
FIDELITY_CAPS_ENABLED = True
FIDELITY_TIER_CAP = {"high": "via", "medium": "manzana"}

#: Spanish label of a guard cap that is not already explained by the ``aproximado:`` note.
_GUARD_NOTES = {
    "plate_letters": "letra de placa distinta al catastro",
    "plate_bis": "bis de placa distinto al catastro",
    "plate_quadrant": "cuadrante final en un solo lado",
    "truncated_parse": "parse truncado",
}

#: Every rejection except an unparseable address is reported as SIN_MATCH; the
#: specific reason stays in ``motivo``.
STATUS_MAP = {"SIN_ESPECIFICAR": "SIN_MATCH", "SIN_MATCH_REGLAS": "SIN_MATCH"}

#: The three statuses a caller can ever see in ``estado``.
PUBLIC_STATUSES = ("OK", "SIN_MATCH", "NO_PARSEABLE")


# ---------------------------------------------------------------------------
# Placeholder / rule helpers (moved verbatim from scripts/normalizar.py)
# ---------------------------------------------------------------------------
def _clean_key(value) -> str:
    """Uppercase, accent-free, whitespace-collapsed key used for placeholder detection."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def is_unspecified(value) -> bool:
    key = _clean_key(value)
    if key in PLACEHOLDERS:
        return True
    # only punctuation / digits with no letters and fewer than 3 digits
    if not re.search(r"[A-Z]", key) and len(re.sub(r"\D", "", key)) < 3:
        return True
    return False


#: Field name -> Spanish label used in the ``letra de ...`` violation messages.
_LETTER_FIELD_LABELS = {
    "via_letters": "letra de via",
    "cross_letters": "letra de cruce",
    "via_suffix_letters": "letra de sufijo de via",
    "cross_suffix_letters": "letra de sufijo de cruce",
}

#: Field name -> Spanish label used in the ``numero de sufijo ...`` violation messages.
#: ``KR 26 H 1`` ... ``KR 26 H 4`` are different streets with different lots, so the
#: numeric suffix is an identity field like the via number (audit 2026-10, finding 2).
_SUFFIX_FIELD_LABELS = {
    "via_suffix": "numero de sufijo de via",
    "cross_suffix": "numero de sufijo de cruce",
}

#: Stable, machine-readable token ``motivo`` carries when a candidate was rejected ONLY
#: because a numeric suffix is present on one side (``KR 26 H`` vs ``KR 26 H 3``). The
#: rule stays hard; part of those rejections are the right
#: lot (see CHANGELOG), so downstream can filter on this token to review them.
ONE_SIDED_SUFFIX_TOKEN = "regla: sufijo unilateral"

_ONE_SIDED_SUFFIX_RE = re.compile(
    "(?:" + "|".join(re.escape(label) for label in _SUFFIX_FIELD_LABELS.values()) + r") (?:- != \S+|\S+ != -)"
)


def _is_one_sided_suffix(message: str) -> bool:
    """Whether a violation message is a numeric suffix present on exactly one side."""
    return _ONE_SIDED_SUFFIX_RE.fullmatch(message) is not None

#: Field name -> Spanish label used in the ``bis de ...`` violation messages.
_BIS_FIELD_LABELS = {
    "via_bis": "bis de via",
    "cross_bis": "bis de cruce",
}

#: Field name -> Spanish label used in the ``cuadrante de ...`` violation messages.
_QUADRANT_FIELD_LABELS = {
    "via_quadrant": "cuadrante de via",
    "cross_quadrant": "cuadrante de cruce",
}

#: Standalone complement tokens the parser missed as a quadrant (see the module
#: docstring background on ``CL 5 # 38-10 NORTE``): the full quadrant words and
#: the multi-letter abbreviations from :data:`QUADRANT_SHORT`. Single cardinal
#: letters (``N``, ``S``, ``E``, ``O``/``W``) are deliberately excluded, because
#: in complement position they are unit letters (``BLQ E``), not quadrants.
_COMPLEMENT_QUADRANT_WORDS = frozenset(_QUADRANT_WORDS) | frozenset(QUADRANT_SHORT)


def _norm_suffix(value) -> str:
    """Comparison key for a numeric suffix: stripped, without leading zeros.

    ``None``/blank -> ``""`` (no suffix), which differs from ``"0"``.
    """
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def _norm_letters(value) -> str:
    """Comparison key for a letters field: upper/stripped, ``None``/``""`` -> ``""``."""
    if value is None:
        return ""
    return str(value).strip().upper()


#: Field name -> its quadrant field on the same side, for the abbreviated
#: quadrant-letter equivalence below (via_letters/via_quadrant, cross_letters/
#: cross_quadrant). Suffix-letter fields are not covered: users do not type
#: ``NORTE`` glued after a secondary number.
_LETTER_QUADRANT_SIDES = {
    "via_letters": "via_quadrant",
    "cross_letters": "cross_quadrant",
}

#: A trailing cardinal LETTER the parser reads as a street letter (``CALLE 6N``,
#: ``AV 12 O``, ``KR 46 W``) but that is equivalent to a spelled-out quadrant the
#: cadastre carries instead (``CL 6 NORTE``, ``AV 12 OESTE``). ``E``/``S`` are
#: deliberately excluded: the cadastre spells those out too, but a bare ``E``
#: or ``S`` letter is a real, distinct street (``KR 41 E``) - ESTE never occurs
#: in the cadastral layer and SUR occurs once, so treating them as equivalent
#: would erase real streets, not typos.
_QUADRANT_LETTER_EQUIVALENTS = {"N": "NORTE", "O": "OESTE", "W": "OESTE"}


def _relax_letters_for_quadrant(query_letters: str, cadastral_quadrant: str | None) -> str:
    """Strip a trailing cardinal-letter suffix from ``query_letters`` when it is
    equivalent to ``cadastral_quadrant`` (see :data:`_QUADRANT_LETTER_EQUIVALENTS`).

    ``CALLE 6N`` -> via_letters ``"N"`` compared against a cadastral via_quadrant
    ``"NORTE"`` becomes an empty letters value, same as the cadastre's own
    (letter-less) via_letters. A multi-letter run (``"D W"``) only loses its
    trailing cardinal letter (-> ``"D"``); everything else is left untouched.
    """
    if not query_letters or cadastral_quadrant not in ("NORTE", "OESTE"):
        return query_letters
    if _QUADRANT_LETTER_EQUIVALENTS.get(query_letters[-1]) != cadastral_quadrant:
        return query_letters
    return query_letters[:-1].strip()


def _complement_quadrant_violation(query_parse, cadastral_parse) -> str | None:
    """A quadrant word the query's parser dropped into ``complement`` that the
    matched cadastral address does not carry in either quadrant field, nor as
    the same leftover word in its own complement (some real cadastral strings -
    ``AV 4 2 2 OESTE # 13 B - 19``, ``KR 72 CL 13 ZONA VERDE OCCIDENTAL SUR`` -
    push a quadrant word into the complement on both the query and the matched
    record itself, so comparing an address against its own parse must stay
    clean)."""
    complement = query_parse.complement or ""
    cadastral_quadrants = {cadastral_parse.via_quadrant, cadastral_parse.cross_quadrant}
    cadastral_complement_tokens = set((cadastral_parse.complement or "").split())
    for token in complement.split():
        if token in _QUADRANT_WORDS:
            word = token
        elif token in QUADRANT_SHORT:
            word = QUADRANT_SHORT[token]
        else:
            continue
        if word in cadastral_quadrants or word in cadastral_complement_tokens:
            continue
        return f"cuadrante en complemento {word} no esta en catastro"
    return None


#: Plate differences up to this many units (both plates numeric) are a *soft*
#: violation: the row is kept but only guaranteed at manzana level. The
#: ``plate_tolerance`` tunable is the hard tolerance inside it (no violation).
PLATE_SOFT_WINDOW = PLATE_TRAMO_WINDOW


def _rule_violations(query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0) -> list[str]:
    """Structural rules a match must satisfy; returns every violated rule (hard and soft)."""
    return [msg for msg, _soft in _tagged_violations(
        query_parse, cadastral_parse, min_struct, struct_score, plate_tolerance)]


def _classify_violations(
    query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0,
    soft_rules: frozenset = frozenset(),
) -> tuple[list[str], list[str]]:
    """Split the violations into ``(hard, soft)`` message lists.

    Soft: a one-sided ``letra de via`` (exactly one side has a via letter), a
    numeric plate difference within :data:`PLATE_SOFT_WINDOW`, and every class
    named in ``soft_rules`` (see :data:`SOFT_RULE_NAMES`). Everything else is hard.
    """
    hard, soft, _caps = _classify_with_caps(
        query_parse, cadastral_parse, min_struct, struct_score, plate_tolerance, soft_rules)
    return hard, soft


def _classify_with_caps(
    query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0,
    soft_rules: frozenset = frozenset(),
) -> tuple[list[str], list[str], list[tuple[str | None, str, str]]]:
    """:func:`_classify_violations` plus the precision ceilings of the row.

    The third element lists ``(note, reason, level)``: an accepted row that is
    nevertheless unsafe to label ``predio`` / ``manzana``. ``note`` (``None`` when
    the soft violation already says it) is added to the ``aproximado:`` group of the
    ``motivo``; ``level`` is the most precise ``nivel_precision`` the row may carry
    (see :data:`ONE_SIDED_VIA_LETTER_LEVEL` and :data:`ONE_SIDED_QUADRANT_LEVEL`).
    Ceilings do not count against ``max_soft``.
    """
    violations, caps = _scan_rules(
        query_parse, cadastral_parse, min_struct, struct_score, plate_tolerance, soft_rules)
    return [m for m, soft in violations if not soft], [m for m, soft in violations if soft], caps


def _letter_is_soft(field: str, qv: str, cv: str, soft_rules: frozenset) -> bool:
    """Whether a letters mismatch on ``field`` is downgraded to soft."""
    one_sided = not qv or not cv
    if field == "via_letters":
        return one_sided or "letra_via" in soft_rules
    if field == "cross_letters":
        if "letra_cruce" in soft_rules:
            return True
        if one_sided and "letra_cruce_una_cara" in soft_rules:
            return True
        return (
            one_sided and "letra_cruce_cardinal" in soft_rules
            and (qv or cv) in _QUADRANT_LETTER_EQUIVALENTS
        )
    return False


def _tagged_violations(
    query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0,
    soft_rules: frozenset = frozenset(),
) -> list[tuple[str, bool]]:
    """Every violation as ``(message, is_soft)`` in a stable order."""
    return _scan_rules(
        query_parse, cadastral_parse, min_struct, struct_score, plate_tolerance, soft_rules)[0]


#: Most precise ``nivel_precision`` of a row accepted with a letter on only one side
#: of the via (``KR 987`` vs ``KR 987 H`` are different streets). Such rows hit the right manzana less often
#: than exact agreement and sit far from the reference more often. It used to be ``esquina``; the position tiers cap it at ``manzana`` (the tier says
#: so with its own 135 m radius), the same ceiling as every other one-sided finding.
ONE_SIDED_VIA_LETTER_LEVEL = "manzana"

#: Most precise level of a row accepted with a quadrant on only one side. Such
#: rows keep most of the manzanas but their large-error share is higher than that of exact
#: quadrant agreement, so they cannot be ``predio``.
ONE_SIDED_QUADRANT_LEVEL = "manzana"


def _quadrants_of(parse) -> set[str]:
    """Quadrants a parse carries anywhere: via, cross, or a quadrant word left in the complement."""
    found = {q for q in (parse.via_quadrant, parse.cross_quadrant) if q}
    for letters in (parse.via_letters, parse.cross_letters):
        # ``CALLE 6N`` reads the quadrant as a street letter (see _QUADRANT_LETTER_EQUIVALENTS).
        equivalent = _QUADRANT_LETTER_EQUIVALENTS.get(_norm_letters(letters)[-1:])
        if equivalent:
            found.add(equivalent)
    for token in (parse.complement or "").split():
        if token in _QUADRANT_WORDS:
            found.add(token)
        elif token in QUADRANT_SHORT:
            found.add(QUADRANT_SHORT[token])
    return found


def _scan_rules(
    query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0,
    soft_rules: frozenset = frozenset(),
) -> tuple[list[tuple[str, bool]], list[tuple[str | None, str, str]]]:
    """``(violations, caps)``: every violation as ``(message, is_soft)`` plus the level ceilings."""
    violations: list[tuple[str, bool]] = []
    caps: list[tuple[str | None, str, str]] = []
    if not cadastral_parse.parse_ok:
        return [("candidato catastral no parseable", False)], caps
    if query_parse.via_type and cadastral_parse.via_type and query_parse.via_type != cadastral_parse.via_type:
        violations.append((f"tipo de via {query_parse.via_type} != {cadastral_parse.via_type}", False))
    if query_parse.via_number and cadastral_parse.via_number:
        if query_parse.via_number.lstrip("0") != str(cadastral_parse.via_number).lstrip("0"):
            violations.append((f"numero de via {query_parse.via_number} != {cadastral_parse.via_number}", False))
    if query_parse.cross_number and cadastral_parse.cross_number:
        if query_parse.cross_number.lstrip("0") != str(cadastral_parse.cross_number).lstrip("0"):
            violations.append((f"numero de cruce {query_parse.cross_number} != {cadastral_parse.cross_number}", False))
    if query_parse.plate and cadastral_parse.plate:
        qp, cp = str(query_parse.plate).lstrip("0"), str(cadastral_parse.plate).lstrip("0")
        if qp != cp:
            try:
                delta = abs(int(qp or 0) - int(cp or 0))
            except ValueError:
                delta = None
            if delta is None or delta > plate_tolerance:
                soft = delta is not None and delta <= PLATE_SOFT_WINDOW
                violations.append((f"placa {query_parse.plate} != {cadastral_parse.plate}", soft))
    if (
        query_parse.cross_type and cadastral_parse.cross_type
        and query_parse.cross_type != cadastral_parse.cross_type
    ):
        violations.append((f"tipo de cruce {query_parse.cross_type} != {cadastral_parse.cross_type}", False))

    # Identity rule: a numeric suffix present on only one side, or different on
    # both, is another street. Hard whatever ``soft_rules`` says.
    for field, label in _SUFFIX_FIELD_LABELS.items():
        qv, cv = _norm_suffix(getattr(query_parse, field)), _norm_suffix(getattr(cadastral_parse, field))
        if qv != cv:
            violations.append((f"{label} {qv or '-'} != {cv or '-'}", False))

    for field, label in _LETTER_FIELD_LABELS.items():
        qv = _norm_letters(getattr(query_parse, field))
        cv = _norm_letters(getattr(cadastral_parse, field))
        quadrant_field = _LETTER_QUADRANT_SIDES.get(field)
        if quadrant_field is not None and getattr(query_parse, quadrant_field) is None:
            qv = _relax_letters_for_quadrant(qv, getattr(cadastral_parse, quadrant_field))
        if qv != cv:
            soft = _letter_is_soft(field, qv, cv, soft_rules)
            violations.append((f"{label} {qv or '-'} != {cv or '-'}", soft))
            if field == "via_letters" and (not qv or not cv):
                caps.append((None, "via con letra en un solo lado", ONE_SIDED_VIA_LETTER_LEVEL))

    for field, label in _BIS_FIELD_LABELS.items():
        qv, cv = bool(getattr(query_parse, field)), bool(getattr(cadastral_parse, field))
        if qv != cv:
            soft = ("bis_via" if field == "via_bis" else "bis_cruce") in soft_rules
            violations.append((f"{label} {'si' if qv else 'no'} != {'si' if cv else 'no'}", soft))

    # Symmetric-lenient: a one-sided quadrant (either side) is allowed - it is
    # the more common, correct case (17 correct vs 3 wrong on the strict-eval
    # ground truth). Only a genuine both-sides disagreement is a violation.
    for field, label in _QUADRANT_FIELD_LABELS.items():
        qv = getattr(query_parse, field)
        cv = getattr(cadastral_parse, field)
        if qv is not None and cv is not None and qv != cv:
            violations.append((f"{label} {qv} != {cv}", False))
    # Contradiction across fields (``CL 5 NORTE # 38-10`` vs ``CL 5 # 38 OESTE - 10``): each
    # side names a quadrant, none in common, and no same-field disagreement said so already.
    # This situation was not seen among the evaluated OK rows, so rejecting costs nothing.
    if not any(m.startswith("cuadrante de ") for m, _soft in violations):
        query_fields = {getattr(query_parse, f) for f in _QUADRANT_FIELD_LABELS} - {None}
        cadastral_fields = {getattr(cadastral_parse, f) for f in _QUADRANT_FIELD_LABELS} - {None}
        if query_fields and cadastral_fields and not (query_fields & cadastral_fields):
            violations.append((f"cuadrante {'/'.join(sorted(query_fields))} != {'/'.join(sorted(cadastral_fields))}", False))
    # One-sided quadrant (audit 2026-10): accepted, but it can never be labelled
    # ``predio``. Not a violation: it does not count against ``max_soft``. A quadrant
    # that merely sits in the other field (or in the complement) is not one-sided.
    query_quadrants, cadastral_quadrants = _quadrants_of(query_parse), _quadrants_of(cadastral_parse)
    for field, label in _QUADRANT_FIELD_LABELS.items():
        qv, cv = getattr(query_parse, field), getattr(cadastral_parse, field)
        if (qv is None) == (cv is None):
            continue
        other_side = cadastral_quadrants if qv is not None else query_quadrants
        if (qv or cv) not in other_side:
            caps.append((f"{label} {qv or '-'} != {cv or '-'}", "cuadrante en un solo lado", ONE_SIDED_QUADRANT_LEVEL))

    complement_violation = _complement_quadrant_violation(query_parse, cadastral_parse)
    if complement_violation is not None:
        violations.append((complement_violation, "cuadrante_compl" in soft_rules))

    if struct_score < min_struct:
        violations.append((f"coincidencia estructural {struct_score:.2f} < {min_struct:.2f}", False))
    return violations, caps


def _comuna_label(value) -> str | None:
    """`19`, `"19"`, `"019"` -> `"Comuna 19"`; other names pass through."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.isdigit():
        return f"Comuna {int(text)}"
    return text


def _zone_columns(detection: Detection | None, located: dict | None = None,
                  comuna_fallback=None) -> dict:
    """One normalized barrio/vereda and one comuna/corregimiento value.

    Priority: the polygon the matched predio falls in (OK rows) > the place
    detected in the text > the comuna encoded in the predial number.
    """
    located = located or {}
    barrio = located.get("barrio") or located.get("vereda")
    zone = located.get("comuna")
    zone = _comuna_label(zone) if zone is not None else located.get("corregimiento")
    if detection is not None and detection.detected:
        barrio = barrio or detection.barrio or detection.vereda
        if not zone:
            zone = _comuna_label(detection.comuna) if detection.comuna is not None else detection.corregimiento
    if not zone and comuna_fallback is not None:
        zone = _comuna_label(comuna_fallback)
    return {"barrio_vereda": barrio or None, "comuna_corregimiento": zone or None}


def _precision_level(parsed) -> str:
    """How much of the via/cross/plate triple the query itself confirmed.

    An OK row can still bind to a real predio when the query left a field
    unconfirmed (see the module docstring background on ``Calle 5 - 38`` and
    ``Calle 5 # 38``): ``_rule_violations`` skips a rule whenever the query's
    side of it is empty, so the match comes from the *other* fields plus the
    model, not from that field. This says which piece was actually pinned down
    by the input, single place to grow when a shared match needs a coarser
    level (``"manzana"``, ``"direccion"``).
    """
    if parsed.plate is None:
        return "esquina"
    if parsed.cross_number is None:
        return "via"
    return "predio"


#: Levels from least to most precise; the combined level of a row is the least
#: precise one that applies to it.
_PRECISION_ORDER = ("direccion", "via", "esquina", "manzana", "predio")


def _least_precise(*levels: str) -> str:
    """Combine precision levels: the least precise wins.

    Order (least to most precise): ``direccion`` < ``via`` < ``esquina`` <
    ``manzana`` < ``predio``. Used to merge the incomplete-input level of
    :func:`_precision_level` with the shared-address level.
    """
    return min(levels, key=_PRECISION_ORDER.index)


def reliability_features(
    confidence: float, struct: float, relaxed_letter: bool, relaxed_plate: bool,
    n_predios: int, nivel: str, gate_applied: bool,
    margen_manzana: float | None = None, margen_predio: float | None = None,
    n_competidores: int = 0,
) -> dict[str, float]:
    """Features of one OK row for the calibrated reliability model.

    Pure and shared by serving and the offline fit (via ``feature_sink``), so the
    two can never drift. ``nivel`` is one-hot encoded against the base level
    ``predio`` (an unknown level also maps to the base). ``n_predios <= 1`` gives
    ``log_n_predios == 0``. The margins are clipped to ``[0, MARGIN_CLIP]`` with
    ``None``/NaN meaning "no competition" (== ``MARGIN_CLIP``); ``n_competidores``
    is clipped to ``0..COMPETITOR_CLIP``.
    """
    def _margin(value) -> float:
        if value is None or math.isnan(value):
            return MARGIN_CLIP
        return min(max(float(value), 0.0), MARGIN_CLIP)


    return {
        "confidence": float(confidence),
        "struct": float(struct),
        "relaxed_letter": 1.0 if relaxed_letter else 0.0,
        "relaxed_plate": 1.0 if relaxed_plate else 0.0,
        "log_n_predios": math.log(max(int(n_predios), 1)),
        "nivel_esquina": 1.0 if nivel == "esquina" else 0.0,
        "nivel_via": 1.0 if nivel == "via" else 0.0,
        "nivel_manzana": 1.0 if nivel == "manzana" else 0.0,
        "nivel_direccion": 1.0 if nivel == "direccion" else 0.0,
        "gate_applied": 1.0 if gate_applied else 0.0,
        "margen_manzana": _margin(margen_manzana),
        "margen_predio": _margin(margen_predio),
        "n_competidores": float(min(max(int(n_competidores), 0), COMPETITOR_CLIP)),
    }


#: Production default of the ambiguity abstention (0 disables it).
DEFAULT_AMBIGUITY_DELTA = 0.02
#: Fused-score gap above which a runner-up is treated as "no competition" by the model.
MARGIN_CLIP = 0.3
#: ``n_competidores`` counts rule-passing rivals in other manzanas within this fused-score gap.
COMPETITOR_WINDOW = 0.05
#: Clip of ``n_competidores`` in the model features.
COMPETITOR_CLIP = 5
_EPS = 1e-9


def _competition(
    normalizer, kept: list, best, parsed, struct_by_doc: dict, min_struct: float, plate_tolerance: int,
    soft_rules: frozenset = frozenset(), max_soft: int = 1,
) -> tuple[float | None, float | None, int]:
    """``(margen_manzana, margen_predio, n_competidores)`` for the chosen candidate.

    Only rivals that would themselves be OK-able compete: fused score at or above
    the threshold, no hard rule violation and at most ``max_soft`` soft ones. NaN scores and
    the chosen doc itself are skipped. ``margen_manzana`` is the gap to the best
    rival in a DIFFERENT manzana, ``margen_predio`` the gap to the best rival with
    a different doc (any manzana); ``None`` when there is none. ``n_competidores``
    counts rivals in other manzanas within :data:`COMPETITOR_WINDOW`.
    """
    best_manzana = str(normalizer._manzana[best.doc])
    margen_manzana = margen_predio = None
    n_competidores = 0
    rivals = sorted(
        (c for c in kept if c.doc != best.doc and not math.isnan(c.score) and c.score >= normalizer.threshold),
        key=lambda c: -c.score,
    )
    for cand in rivals:
        gap = max(best.score - cand.score, 0.0)
        # Sorted by score: once both margins are known and we left the window, nothing changes.
        if margen_manzana is not None and margen_predio is not None and gap > COMPETITOR_WINDOW + _EPS:
            break
        hard, soft = _classify_violations(
            parsed, parse_address(str(normalizer._direccion[cand.doc])), min_struct,
            struct_by_doc[cand.doc], plate_tolerance, soft_rules,
        )
        # A candidate that fails ONLY the one-sided suffix rule is a different street on paper,
        # but the evidence cannot tell: it still competes (on the gold sets, the
        # chosen row was the wrong lot in most of those ambiguities). Suffixes that differ on
        # both sides stay a real rejection.
        hard = [m for m in hard if not _is_one_sided_suffix(m)]
        if hard or len(soft) > max_soft:
            continue
        if margen_predio is None:
            margen_predio = gap
        if str(normalizer._manzana[cand.doc]) != best_manzana:
            if margen_manzana is None:
                margen_manzana = gap
            if gap <= COMPETITOR_WINDOW + _EPS:
                n_competidores += 1
    return margen_manzana, margen_predio, n_competidores


#: Texts a null cadastral code takes once it went through ``str()`` (``None``, ``NaN``, ``pd.NA``).
_NULL_CODE_TEXTS = frozenset({"", "NONE", "NAN", "NAT", "NULL", "<NA>"})


def _clean_code(value) -> str | None:
    """A cadastral code as text, or ``None`` when it is null (never the string ``"None"``)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return None if text.upper() in _NULL_CODE_TEXTS else text


def _npn_anomaly(npn: str) -> str | None:
    """Why a digit-led NPN is not a well-formed 30-digit code (``None`` when it is, or when
    it is not NPN-shaped at all, e.g. a synthetic placeholder)."""
    if not npn[:1].isdigit() or re.fullmatch(r"\d{30}", npn):
        return None
    if re.search(r"\D", npn):
        return "caracteres no numericos"
    return f"{len(npn)} caracteres"


def _reported_point(normalizer, best) -> tuple[float, float, bool]:
    """``(lon, lat, is_parcel_point)`` reported for the chosen document.

    The centroid of the parcel whose NPN is returned when the artifacts carry it
    (:func:`cali_address.inference.load_own_centroids`: the sidecar or ``catastro.parquet``),
    else the address centroid the candidate came with. Candidate retrieval and the
    geographic gate keep using the address centroid: only the reported point changes.
    """
    own_lon, own_lat = normalizer._own_lon, normalizer._own_lat
    if own_lon is not None and own_lat is not None:
        lon, lat = float(own_lon[best.doc]), float(own_lat[best.doc])
        if math.isfinite(lon) and math.isfinite(lat):
            return lon, lat, True
    return best.lon, best.lat, False


def _empty_result(raw, estado: str, motivo: str, rule_canonical: str | None = None,
                  detection: Detection | None = None, complement: str | None = None,
                  override_rejected: bool = False) -> dict:
    """A non-OK row: place columns may be filled, cadastral columns never are.

    ``complement`` is the rule-canonical unit text (``TO F``) when the parser got one.
    """
    return {
        "direccion_entrada": raw,
        "estado": STATUS_MAP.get(estado, estado),
        "motivo": motivo,
        "direccion_normalizada": rule_canonical or None,
        "fuente_normalizacion": "reglas" if rule_canonical else None,
        "numero_predial_nacional": None,
        "manzana": None,
        "lat": None,
        "lon": None,
        "confianza": None,
        "confiabilidad": None,
        "confiabilidad_manzana": None,
        "margen": None,
        "nivel_precision": None,
        **_zone_columns(detection),
        "complemento_normalizado": complement,
        "position_tier": "none",
        "position_radius_m": float("nan"),
        "position_source": "none",
        "selected_doc_id": None,
        "override_rejected": bool(override_rejected),
        "direccion_catastral": None,
        "fidelity_risk": "none",
        "fidelity_flags": "" if rule_canonical else "no_address",
        "catalog_status": "unparsed",
        "address_verified": False,
    }


def _apply_fidelity(row: dict, raw, cleaned: str | None) -> dict:
    """Fill ``fidelity_risk`` / ``fidelity_flags`` and apply the tier caps of :data:`FIDELITY_TIER_CAP`.

    ``raw`` is the ORIGINAL input (before any opt-in refinement); ``cleaned`` the text the gazetteer left,
    so removed place names are not counted as lost text. Caps only ever worsen a tier; ``confiabilidad``
    keeps its calibrated value.
    """
    report = assess_fidelity(raw, row["direccion_normalizada"], cleaned=cleaned)
    row["fidelity_risk"], row["fidelity_flags"] = report.risk, ";".join(report.flags)
    cap = FIDELITY_TIER_CAP.get(report.risk)
    if cap is None or row["estado"] != "OK" or not FIDELITY_CAPS_ENABLED:
        return row
    row["motivo"] = "; ".join(part for part in (row["motivo"], f"fidelidad: {row['fidelity_flags']}") if part)
    if row["position_tier"] == "none":
        return row
    tier = _worse_tier(row["position_tier"], cap)
    row["position_tier"] = tier
    row["position_radius_m"] = float(POSITION_RADIUS_M[tier])
    row["numero_predial_nacional"] = None
    if tier == "via":
        row["manzana"] = None
        if not EXTRAPOLATED_COORDS_ENABLED:
            row["lat"], row["lon"], row["position_source"] = None, None, "none"
    row["nivel_precision"] = _least_precise(row["nivel_precision"], _NIVEL_OF_TIER[tier])
    return row


_HIGH_BLANK_NOTE = "fidelidad alta: direccion no emitida"


def _apply_emission_rule(row: dict, parsed) -> None:
    """Blank ``direccion_normalizada`` / ``fuente_normalizacion`` of a non-OK row whose rule text is not trustworthy.

    See :func:`cali_address.catalog.keeps_rule_text`. ``OK`` rows keep their text (a cadastral match is the
    ``modelo`` source downstream) EXCEPT when the fidelity risk is ``high`` (round 8: corrupted text; legacy
    behaviour when :data:`FIDELITY_CAPS_ENABLED` is off). ``estado``,
    ``catalog_status``, ``position_tier`` and ``direccion_catastral`` (audit only) are untouched; an ``OK`` row that
    gets blanked also gets a ``fidelidad alta: direccion no emitida`` note in ``motivo``. The pipeline then falls
    back to the raw text.
    """
    ok = row["estado"] == "OK"
    if (ok and (row["fidelity_risk"] != "high" or not FIDELITY_CAPS_ENABLED)) or not row.get("direccion_normalizada"):
        return
    if _keeps_rule_text(row["catalog_status"], parsed, row["fidelity_risk"]):
        return
    row["direccion_normalizada"], row["fuente_normalizacion"] = None, None
    if ok:
        row["motivo"] = "; ".join(part for part in (row["motivo"], _HIGH_BLANK_NOTE) if part)


def _apply_catalog_status(normalizer, rows: list, parses: list) -> None:
    """Fill ``catalog_status`` / ``address_verified`` of every row (see :func:`cali_address.catalog.decide`).

    The cadastre key set is only fetched (once) for a row that needs it: a batch of verified rows never builds it.
    The private ``_catalog_*`` hints left by :func:`_build_row` are removed from the rows.
    """
    box: list = []

    def keys():
        if not box:
            try:
                box.append(_catalog_keys_for(normalizer))
            except Exception:  # noqa: BLE001 - without keys the door is simply "unknown"
                box.append(None)
        return box[0]

    for row, parsed in zip(rows, parses):
        exact, shared = row.pop("_catalog_exact", False), row.pop("_catalog_shared", False)
        status = _catalog_decide(
            parsed, row["estado"], row["motivo"], row["fidelity_risk"], bool(exact), bool(shared),
            door_known=lambda parsed=parsed: None if keys() is None else _door_catalogued(parsed, keys()),
        )
        row["catalog_status"] = status
        row["address_verified"] = status == "verified"
        _apply_emission_rule(row, parsed)


def normalize_strict(
    normalizer: AddressNormalizer,
    raws: list,
    min_struct: float = DEFAULT_MIN_STRUCT,
    plate_tolerance: int = 0,
    k: int = 20,
    gazetteer: Gazetteer | None = None,
    stats: dict | None = None,
    barrio_buffer_m: float = CLI_BARRIO_BUFFER_M,
    zone_buffer_m: float = ZONE_BUFFER_M,
    gate_escalate: bool = DEFAULT_GATE_ESCALATE,
    feature_sink: list | None = None,
    ambiguity_delta: float = DEFAULT_AMBIGUITY_DELTA,
    soft_rules=None,
    max_soft: int | None = None,
    gate_fallback: bool | None = None,
    refine_input: bool = False,
) -> pd.DataFrame:
    """Normalize a list of raw strings; only rows passing every rule get cadastral fields.

    ``refine_input`` (default off) first runs :func:`cali_address.refine.refine_address` on every
    string; ``direccion_entrada`` keeps the original text and a ``refine_changes`` column (rule ids)
    is appended AFTER the usual columns. Off, the output is exactly the historical one.

    ``stats`` is filled in place with gate counters when a dict is passed, which is
    how the CLI reports what the geographic gate actually changed.

    ``feature_sink``, when a list, receives one entry per input row: the
    :func:`reliability_features` dict for OK rows and ``None`` otherwise. The
    reliability columns come from ``normalizer.reliability`` (``None`` when the
    artifact is missing, which leaves them ``None`` on OK rows).

    ``ambiguity_delta`` (default :data:`DEFAULT_AMBIGUITY_DELTA`, 0 = off) is a soft
    abstention: when positive, an OK row whose ``margen`` (fused-score gap to the
    best rule-passing candidate in another manzana) is strictly below it becomes
    ``SIN_MATCH``. A row with no competitor (``margen`` None) never abstains.

    Stage 3 options (``None`` = "use the artifact's ``decision`` block, else the historical
    behaviour"; an explicit value always wins):

    ``soft_rules``  names from :data:`SOFT_RULE_NAMES` whose violations become *soft* (row stays OK,
                    ``nivel_precision`` capped at ``manzana``) instead of rejecting the row.
    ``max_soft``    how many soft violations an OK row may carry (default 1).
    ``gate_fallback`` when the geographic gate would leave no candidate, keep the text-ranked
                    candidates instead of abstaining (note added, ``nivel_precision`` capped at
                    ``manzana``). The confidence threshold and every rule still apply.
    """
    original_raws, refined = raws, None
    if refine_input:
        refined = [refine_address(r) if isinstance(r, str) else None for r in raws]
        raws = [r if rr is None else rr.text for rr, r in zip(refined, raws)]
    decision = getattr(normalizer, "decision", None) or {}
    soft_rules = normalize_soft_rules(decision.get("soft_rules") if soft_rules is None else soft_rules)
    max_soft = validate_max_soft(decision.get("max_soft", 1) if max_soft is None else max_soft)
    gate_fallback = bool(decision.get("gate_fallback", False) if gate_fallback is None else gate_fallback)
    counters = {
        "filas_con_lugar": 0, "filas_con_reja": 0, "candidatos_descartados": 0,
        "mejor_candidato_cambiado": 0, "filas_sin_candidato_tras_reja": 0,
        "texto_limpiado": 0, "parseable_tras_limpiar": 0,
    }
    rows: list[dict | None] = [None] * len(raws)
    detections: list[Detection | None] = [None] * len(raws)
    queries: list[str | None] = [None] * len(raws)
    parses: list[object] = [None] * len(raws)
    to_model: list[int] = []
    feats: list[dict | None] = [None] * len(raws)
    reliability = getattr(normalizer, "reliability", None)

    for i, raw in enumerate(raws):
        if is_unspecified(raw):
            rows[i] = _empty_result(raw, "SIN_ESPECIFICAR", "dato vacio o marcador sin informacion")
            continue
        detection = gazetteer.detect(raw) if gazetteer is not None else None
        detections[i] = detection
        if detection is not None and detection.detected:
            counters["filas_con_lugar"] += 1
        text = str(raw) if detection is None else detection.cleaned_text
        if text != str(raw):
            counters["texto_limpiado"] += 1
        parsed = parse_address(text)
        if not parsed.parse_ok and text != str(raw):
            # The cleaned text must never lose a row the raw text could parse.
            fallback = parse_address(str(raw))
            if fallback.parse_ok:
                text, parsed = str(raw), fallback
        elif parsed.parse_ok and text != str(raw) and not parse_address(str(raw)).parse_ok:
            counters["parseable_tras_limpiar"] += 1
        if not parsed.parse_ok:
            rows[i] = _empty_result(
                raw, "NO_PARSEABLE",
                "; ".join(parsed.notes) or "gramatica de direccion no reconocida",
                detection=detection,
            )
            continue
        queries[i] = text
        parses[i] = parsed
        to_model.append(i)

    if to_model:
        comps = normalizer.score_components([queries[i] for i in to_model], k=k)
        weights = normalizer.weights
        fused = (
            weights["sim"] * comps["sim"] + weights["fuzz"] * comps["fuzz"]
            + weights["struct"] * comps["struct"] + weights["geo"] * comps["geo"]
        )
        for pos, i in enumerate(to_model):
            rows[i] = _build_row(
                normalizer, gazetteer, raws[i], parses[i], detections[i],
                comps["doc"][pos], comps["struct"][pos], fused[pos],
                min_struct, plate_tolerance, counters,
                barrio_buffer_m, zone_buffer_m, gate_escalate,
                reliability, feats, i, ambiguity_delta,
                soft_rules, max_soft, gate_fallback,
            )

    for i, row in enumerate(rows):
        cleaned = detections[i].cleaned_text if detections[i] is not None else None
        rows[i] = _apply_fidelity(row, original_raws[i], cleaned)
    _apply_catalog_status(normalizer, rows, parses)
    if feature_sink is not None:
        feature_sink.extend(feats)
    if stats is not None:
        stats.update(counters)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame["selected_doc_id"] = pd.array(frame["selected_doc_id"], dtype="Int64")
    frame["override_rejected"] = frame["override_rejected"].astype(bool)
    if refined is not None:
        frame["direccion_entrada"] = list(original_raws)
        frame["refine_changes"] = ["" if rr is None else ", ".join(rr.changes) for rr in refined]
    return frame


def _count_one_sided_matches(normalizer, kept, parsed, plate_tolerance, soft_rules) -> int:
    """Candidates of the scored top-k whose base equals the query's apart from one-sided quadrant / letters."""
    count = 0
    for candidate in kept:
        try:
            other = parse_address(str(normalizer._direccion[candidate.doc]))
            hard, caps = base_mismatches(
                parsed, other, mode="position", plate_tolerance=plate_tolerance, soft_rules=soft_rules)
        except Exception:  # noqa: BLE001 - an unreadable candidate simply does not count
            continue
        if not hard and set(caps) <= ONE_SIDED_FIELDS:
            count += 1
    return count


def _worse_tier(a: str, b: str) -> str:
    """The less precise of two position tiers."""
    return max(a, b, key=_TIER_ORDER.index)


def _resolve_tramo(normalizer, parsed, cadastral_parsed, candidate_point):
    """``(tier, coords, source, note, radius)`` of a row capped at ``tramo``.

    The query's plate is placed on the face of the matched street pair, between its two nearest
    known parcels of the same side: ``tramo`` with interpolated coordinates. Beyond the last known
    parcel it is an extrapolation: tier ``via``, coordinates only when
    :data:`EXTRAPOLATED_COORDS_ENABLED`. With no parcel on that side to interpolate from, the
    best candidate's own point (``candidate_point``) is reported as ``manzana`` (never as a tramo).
    """
    try:
        number = int(str(parsed.plate).lstrip("0") or 0)
        located = _interp_index_for(normalizer).locate(_street_key(cadastral_parsed), number)
    except Exception:  # noqa: BLE001 - an index problem must never lose the row: degrade to manzana
        located = None
    if located is None or located.kind == "none":
        return ("manzana", candidate_point, "parcel",
                "posicion maxima manzana: sin parcelas vecinas para interpolar", float(POSITION_RADIUS_M["manzana"]))
    if located.kind == "interp":
        return ("tramo", (located.lon, located.lat), "interp", "posicion interpolada entre parcelas vecinas",
                _tramo_radius_m(located.gap_m))
    why = ("posicion no interpolada: parcelas vecinas demasiado separadas" if located.kind == "gap"
           else "posicion extrapolada: sin parcela vecina mas alla")
    if located.kind == "gap":
        radius = _tramo_radius_m(located.gap_m)
    elif located.gap_m is not None:
        radius = _extrap_radius_m(located.gap_m)
    else:
        radius = float(POSITION_RADIUS_M["via"])
    if EXTRAPOLATED_COORDS_ENABLED and located.lat is not None:
        return "via", (located.lon, located.lat), "interp", why, radius
    return "via", None, "none", why + ": coordenadas retenidas", radius


def _build_row(
    normalizer, gazetteer, raw, parsed, detection,
    docs, struct_row, fused_row, min_struct, plate_tolerance, counters,
    barrio_buffer_m, zone_buffer_m, gate_escalate,
    reliability=None, feats: list | None = None, index: int = 0,
    ambiguity_delta: float = DEFAULT_AMBIGUITY_DELTA,
    soft_rules: frozenset = frozenset(), max_soft: int = 1, gate_fallback: bool = False,
) -> dict:
    """Score one already-parsed row: fuse, gate, threshold, then apply the rules."""
    # Same unit rendering as OK rows: only recognised units, never the free text.
    rule_base = canonical(parsed, style="spaced", with_complement=False)
    rule_complement = render_complement(parsed.complement_text) if rule_base else None
    rule_canonical = _rule_canonical(parsed)
    slots = np.flatnonzero(docs >= 0)
    candidates = [
        Candidate(
            score=float(fused_row[s]), doc=int(docs[s]),
            lon=float(normalizer._lon[docs[s]]), lat=float(normalizer._lat[docs[s]]),
        )
        for s in slots
    ]
    struct_by_doc = {int(docs[s]): float(struct_row[s]) for s in slots}
    if not candidates:
        return _empty_result(
            raw, "SIN_MATCH", f"confianza 0.00 < umbral {normalizer.threshold:.2f}",
            rule_canonical, detection, complement=rule_complement,
        )

    best_before = max(candidates, key=lambda c: c.score)
    kept, gate_label = (
        gate_candidates(
            gazetteer, detection, candidates,
            barrio_buffer_m=barrio_buffer_m, zone_buffer_m=zone_buffer_m,
            escalate=gate_escalate,
        )
        if (gazetteer is not None and detection is not None) else (candidates, None)
    )
    if gate_label is not None:
        counters["filas_con_reja"] += 1
        counters["candidatos_descartados"] += len(candidates) - len(kept)
    gate_overridden = None
    if not kept and gate_fallback and gate_label is not None:
        kept, gate_overridden = list(candidates), gate_label
    if not kept:
        counters["filas_sin_candidato_tras_reja"] += 1
        return _empty_result(
            raw, "SIN_MATCH_REGLAS", f"candidatos fuera de {gate_label}",
            rule_canonical, detection, complement=rule_complement,
        )

    best = max(kept, key=lambda c: c.score)
    if gate_label is not None and best.doc != best_before.doc:
        counters["mejor_candidato_cambiado"] += 1

    confidence = float(min(max(best.score, 0.0), 1.0))
    if confidence < normalizer.threshold:
        return _empty_result(
            raw, "SIN_MATCH",
            f"confianza {confidence:.2f} < umbral {normalizer.threshold:.2f}",
            rule_canonical, detection, complement=rule_complement,
        )

    cadastral_address = str(normalizer._direccion[best.doc])
    cadastral_parsed = parse_address(cadastral_address)
    hard, soft, caps = _classify_with_caps(
        parsed, cadastral_parsed, min_struct,
        struct_by_doc[best.doc], plate_tolerance, soft_rules,
    )
    if hard or len(soft) > max_soft:
        reasons = hard + soft
        if any(_is_one_sided_suffix(m) for m in hard):
            reasons.append(ONE_SIDED_SUFFIX_TOKEN)
        return _empty_result(
            raw, "SIN_MATCH_REGLAS", "; ".join(reasons), rule_canonical, detection, complement=rule_complement,
            override_rejected=True,
        )

    guard_hard, guard_caps = base_mismatches(
        parsed, cadastral_parsed, mode="position", plate_tolerance=plate_tolerance, soft_rules=soft_rules)
    if guard_hard:
        return _empty_result(
            raw, "SIN_MATCH_REGLAS", "base distinta al catastro: " + ", ".join(guard_hard),
            rule_canonical, detection, complement=rule_complement, override_rejected=True,
        )

    margen_manzana, margen_predio, n_competidores = _competition(
        normalizer, kept, best, parsed, struct_by_doc, min_struct, plate_tolerance, soft_rules, max_soft,
    )
    if ambiguity_delta > 0 and margen_manzana is not None and margen_manzana < ambiguity_delta:
        return _empty_result(
            raw, "SIN_MATCH",
            f"ambiguo: {max(n_competidores, 1)} candidatos cercanos en otras manzanas",
            rule_canonical, detection, complement=rule_complement,
        )

    point_lon, point_lat, point_is_parcel = _reported_point(normalizer, best)
    predial = _clean_code(normalizer._predial[best.doc])
    manzana = _clean_code(normalizer._manzana[best.doc])
    nivel = _precision_level(parsed)
    notes: list[str] = []
    approximations = soft + [note for note, _reason, _level in caps if note]
    if approximations:
        # Soft violation(s): keep the predio but only vouch for the manzana.
        notes.append("aproximado: " + "; ".join(approximations))
    if soft:
        nivel = _least_precise(nivel, "manzana")
    ceilings: list[tuple[str, str]] = []
    for _note, reason, level in caps:
        if (level, reason) not in ceilings:
            ceilings.append((level, reason))
            notes.append(f"nivel maximo {level}: {reason}")
    if gate_overridden is not None:
        # The detected place excluded every candidate: keep the text match, vouch for the manzana only.
        notes.append(f"fuera de {gate_overridden}")
        nivel = _least_precise(nivel, "manzana")
    n_predios = int(normalizer._n_predios[best.doc])
    if n_predios > 1:
        # The cadastre cannot tell the predios of a shared address apart: the
        # first predio (by OBJECTID) is reported, flagged and downgraded.
        notes.append(f"direccion compartida por {n_predios} predios")
        if not point_is_parcel:
            # Without the parcel's own centroid the point is the median of all of them.
            notes.append(f"coordenada: centroide de la direccion compartida por {n_predios} predios")
        if int(normalizer._manzana_nunique[best.doc]) > 1:
            nivel = _least_precise(nivel, "direccion")
        else:
            nivel = _least_precise(nivel, "manzana")
    if predial is None:
        # No NPN to report: the point is the address, nothing identifies a predio.
        notes.append("NPN nulo en catastro")
        ceilings.append(("direccion", "NPN nulo en catastro"))
    else:
        anomaly = _npn_anomaly(predial)
        if anomaly is not None:
            notes.append(f"NPN anomalo en catastro ({anomaly})")
    # The base address comes from the cadastre; the units (TORRE F, AP 101...) come
    # from the input. Purely textual: it never touches the gating above, and it reads
    # ``complement_text`` (never scored), not the frozen scoring ``complement``.
    # The calibrated reliability model keeps its legacy input (the level before the
    # audit ceilings), so ``confiabilidad`` does not move; the public level does.
    nivel_model = nivel
    nivel = _least_precise(nivel, *(level for level, _reason in ceilings))
    # Position tier: the most restrictive of the guard caps and of the approximations above.
    tier = "predio"
    for cap in guard_caps.values():
        tier = _worse_tier(tier, cap)
    if any(not m.startswith("placa") for m in soft) or caps or gate_overridden is not None or n_predios > 1:
        # a soft violation other than the plate (that one is a tramo cap), a one-sided finding, a
        # place outside the gate, a shared address: the parcel is a manzana-level hint at best
        tier = _worse_tier(tier, "manzana")
    if predial is None:
        tier = _worse_tier(tier, "lote")
    if parsed.plate is None:
        tier = _worse_tier(tier, "manzana")
    # A quadrant / street letter on one side only: the parcel may be on another street. Never keep its predial;
    # keep the manzana tier only when no other candidate differs from the query by that token alone.
    one_sided = sorted(f for f in guard_caps if f in ONE_SIDED_FIELDS)
    if one_sided:
        matching = _count_one_sided_matches(normalizer, kept, parsed, plate_tolerance, soft_rules)
        if matching > 1:
            tier = _worse_tier(tier, "via")
            notes.append(f"posicion maxima via: {matching} candidatos difieren solo por un cuadrante o letra")
        else:
            tier = _worse_tier(tier, "manzana")
    if "cross_number" in guard_caps:
        notes.append("posicion maxima via: sin calle de cruce en la consulta o en el catastro")
    for field, cap in guard_caps.items():
        label = _GUARD_NOTES.get(field)
        if label is None and field.startswith("unit_"):
            label = f"unidad {field[5:]} distinta al catastro"
        if label is not None:
            notes.append(f"posicion maxima {cap}: {label}")
    coords = (point_lon, point_lat)
    source = "parcel"
    radius = float(POSITION_RADIUS_M[tier])
    if tier == "tramo":
        tier, coords, source, note, radius = _resolve_tramo(normalizer, parsed, cadastral_parsed, coords)
        notes.append(note)
    elif tier == "via" and not EXTRAPOLATED_COORDS_ENABLED:
        coords, source = None, "none"
    nivel = _least_precise(nivel, _NIVEL_OF_TIER[tier])
    if tier in ("lote", "tramo", "via") or one_sided:
        predial = None
    if tier == "via":
        manzana = None
    located_lon, located_lat = coords if coords is not None else (point_lon, point_lat)
    located = gazetteer.locate(located_lon, located_lat) if gazetteer is not None else {}
    merged = merge_complement(cadastral_address, parsed.complement_text)
    if merged.conflicts:
        notes.append("complemento distinto al catastro: " + ", ".join(merged.conflicts))
    motivo = "; ".join(notes)
    # The cadastral base is only the row's address when it says what the query says: predio, lote, or a manzana
    # that merely reflects a shared address. Whenever the cadastre differs (tramo, via, one-sided or capped
    # findings) the row's address is the query's own normalized text, so nobody reads the neighbour's address.
    base_differs = (
        tier in ("tramo", "via") or bool(soft) or bool(caps) or bool(one_sided)
        or any(cap != "lote" for cap in guard_caps.values())
    )
    out_address, out_source = merged.address, "catastro"
    if base_differs and rule_canonical:
        out_address, out_source = rule_canonical, "reglas"
    features = reliability_features(
        confidence, struct_by_doc[best.doc],
        relaxed_letter=any(m.startswith(("letra de", "bis de", "cuadrante")) for m in soft),
        relaxed_plate=any(m.startswith("placa") for m in soft),
        n_predios=n_predios, nivel=nivel_model, gate_applied=gate_label is not None,
        margen_manzana=margen_manzana, margen_predio=margen_predio, n_competidores=n_competidores,
    )
    if feats is not None:
        feats[index] = features
    conf_predio = conf_manzana = None
    if reliability is not None:
        pp, pm = reliability.predict(features)
        conf_predio, conf_manzana = round(pp, 3), round(pm, 3)
    return {
        "direccion_entrada": raw,
        "estado": "OK",
        "motivo": motivo,
        "direccion_normalizada": out_address,
        "fuente_normalizacion": out_source,
        "numero_predial_nacional": predial,
        "manzana": manzana,
        "lat": None if coords is None else coords[1],
        "lon": None if coords is None else coords[0],
        "confianza": round(confidence, 4),
        "confiabilidad": conf_predio,
        "confiabilidad_manzana": conf_manzana,
        "margen": None if margen_manzana is None else round(margen_manzana, 4),
        "nivel_precision": nivel,
        **_zone_columns(detection, located, normalizer._comuna[best.doc]),
        "complemento_normalizado": merged.complement,
        "position_tier": tier,
        "position_radius_m": radius,
        "position_source": source,
        "selected_doc_id": int(best.doc),
        "override_rejected": False,
        "direccion_catastral": cadastral_address,
        "fidelity_risk": "none",
        "fidelity_flags": "",
        "catalog_status": "rejected_mismatch",
        "address_verified": False,
        # private hints for the catalogue status (popped by normalize_strict): the guard found nothing to cap and
        # the row's plate was stated (an exact base), and whether several parcels share the address
        "_catalog_exact": not guard_caps and not soft and not caps and gate_overridden is None and parsed.plate is not None,
        "_catalog_shared": n_predios > 1,
    }


def load_gazetteer(basemaps_dir: str, warn=None) -> Gazetteer | None:
    """Load the gazetteer, degrading to plain normalization if the basemaps are gone.

    ``warn`` receives one human-readable line when the gazetteer is unavailable;
    the CLI routes it to stderr, the API to the startup log.
    """
    try:
        return Gazetteer.load(basemaps_dir)
    except (FileNotFoundError, ValueError) as exc:
        if warn is not None:
            warn(f"warning: gazetteer disabled ({exc})")
        return None


# ---------------------------------------------------------------------------
# CLI-facing input reading
# ---------------------------------------------------------------------------
def read_input_file(path: str, column: str | None, sheet, header: int | None
                    ) -> tuple[list, pd.DataFrame | None, str | None]:
    """Return (raw addresses, source dataframe or None for plain text files, address column).

    ``header`` of ``None`` means "guess with :func:`guess_header_row`"; an explicit
    integer always wins. Raises :class:`AddressColumnError` when the address
    column cannot be resolved.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls", ".csv"):
        with open(path, "rb") as fh:
            data = fh.read()
        df = read_table(data, path, sheet=sheet, header=header)
    elif ext in (".geojson", ".json", ".zip"):
        with open(path, "rb") as fh:
            data = fh.read()
        df = read_table(data, path)
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\r\n") for line in fh], None, None
    resolved = resolve_address_column(df, column)
    return df[resolved].tolist(), df.reset_index(drop=True), resolved


def attach_source_columns(result: pd.DataFrame, source: pd.DataFrame | None, keep: str | None,
                          address_column: str | None = None) -> pd.DataFrame:
    """Prepend selected source columns.

    Output column names are the documented contract, so a source column that
    collides with one of them (`comuna`, `lat`, `barrio_vereda`...) is carried as
    `<name>_original`. Source coordinates are suffixed as a pair so `lat`/`lng`
    never end up half renamed.
    """
    if source is None or not keep:
        return result
    if keep.strip().lower() == "all":
        cols = list(source.columns)
    else:
        cols = [c.strip() for c in keep.split(",") if c.strip()]
        missing = [c for c in cols if c not in source.columns]
        if missing:
            raise KeepColumnsError(
                f"--keep-columns not found in input: {missing}. Columns: {list(source.columns)}"
            )
    if address_column in cols:
        # the input address is already carried by the source column: keep exactly
        # two address columns, the input and the normalized output
        result = result.drop(columns=["direccion_entrada"])
    coordinate_names = {"lat", "lon", "lng", "latitud", "longitud"}
    rename = {c: f"{c}_original" for c in cols if c in result.columns}
    if coordinate_names & {c.lower() for c in cols}:
        rename.update({c: f"{c}_original" for c in cols if c.lower() in coordinate_names})
    kept = source[cols].rename(columns=rename).reset_index(drop=True)
    return pd.concat([kept, result.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------
def _write_xlsx(result: pd.DataFrame, target) -> None:
    """Three sheets: every row, the ones to review, and the status counts."""
    summary = result["estado"].value_counts().rename_axis("estado").reset_index(name="filas")
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="normalizado", index=False)
        result[result["estado"] != "OK"].to_excel(writer, sheet_name="revisar", index=False)
        summary.to_excel(writer, sheet_name="resumen", index=False)


def write_output(result: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".xlsx"):
        _write_xlsx(result, path)
    else:
        result.to_csv(path, index=False, encoding="utf-8-sig")


def write_xlsx_bytes(result: pd.DataFrame) -> bytes:
    """Same workbook :func:`write_output` produces, in memory."""
    buffer = io.BytesIO()
    _write_xlsx(result, buffer)
    return buffer.getvalue()


def write_csv_bytes(result: pd.DataFrame) -> bytes:
    return result.to_csv(index=False).encode("utf-8-sig")


def results_to_records(result: pd.DataFrame) -> list[dict]:
    """JSON-safe records: every NaN/NaT becomes ``None``."""
    return result.astype(object).where(pd.notna(result), None).to_dict(orient="records")


def summarize(result: pd.DataFrame) -> dict:
    """Status counts, always with all three public statuses present."""
    counts = result["estado"].value_counts().to_dict() if len(result) else {}
    out = {status: int(counts.get(status, 0)) for status in PUBLIC_STATUSES}
    out["total"] = int(len(result))
    return out


def results_to_geojson(result: pd.DataFrame) -> dict:
    """FeatureCollection with a Point per resolved row.

    Rows without a coordinate keep ``"geometry": null`` rather than being dropped,
    so the feature count always matches the row count and a caller can see what
    could not be located.
    """
    features = []
    for record in results_to_records(result):
        lat, lon = record.get("lat"), record.get("lon")
        geometry = None
        try:
            if lat is not None and lon is not None:
                flon, flat = float(lon), float(lat)
                if np.isfinite(flon) and np.isfinite(flat):
                    geometry = {"type": "Point", "coordinates": [flon, flat]}
        except (TypeError, ValueError):
            geometry = None
        properties = {k: v for k, v in record.items() if k not in ("lat", "lon")}
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
