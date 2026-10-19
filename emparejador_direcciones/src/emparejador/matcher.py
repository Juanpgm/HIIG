"""Pairwise scoring and matching for two already-normalized cadastral addresses.

Design: weighted averages dilute as fields are added (one wrong street
letter would still score around 0.94 out of many fields), so this module
instead multiplies survival factors: ``score = round(prod(1 - p_i), 4)``
over every field that differs, and any hard veto forces ``score = 0.0``
regardless of the product. See the project plan for the full penalty
table and the proof that no reachable score lands in the (0.85, 0.9126)
dead band around the default 0.90 threshold.

Plate vetoes: a plate on only one side, a plate difference beyond
``plate_tolerance``, and a plate *letter* that differs (for example
``KR 1 # 4 - 80 A`` vs ``KR 1 # 4 - 80``: a lettered plate is a different
corner/unit, not the same parcel) are all hard vetoes.

Open-ended addresses: when both sides lack a plate, or both lack a cross
number, the verdict is unchanged (missing on both sides is agreement), but
``MatchResult.warnings`` carries ``sin_placa`` / ``sin_cruce`` so callers
can tell the comparison rested on less structure than usual.

Bad address input (``None``, empty, garbage, unparseable) never raises:
it is scored as a parse-failure veto. Bad settings (``threshold``,
``plate_tolerance``) do raise, since those are programming errors, not
user data.

Opaque complements: a complement chunk whose kind is not a registered
``COMPLEMENT_KINDS`` code (a raw cadastral tail such as ``8 C`` or
``GASS 5``) is compared for equality only. Equal on both sides is exact,
different on both sides is a veto, and present on ONE side only costs the
regular one-sided penalty (0.15) instead of the cheap 0.02, so an
unstructured tail never auto-links to the bare address. Any opaque chunk on
either side adds the warning ``complemento_no_estructurado``. A registered
kind with no usable value (``LT`` alone, ``AP NORTE``) is treated the same
way.

Levels: ``level="unit"`` (default) compares every field, complement
included. ``level="building"`` ignores, on BOTH sides, only the complement
kinds that denote a sub-unit inside one building/predio and that a
cadastre-wide check showed pose no measured risk of conflating different
predios (``BUILDING_IGNORED_KINDS``: BG, INT, PISO, PH; see the constant's
own comment for the numbers). It is stricter than "drop every complement":
kinds that identify the predio itself (MZ, LT, CGTO, VDA, BR, UR, CONJ, ED,
ET) AND kinds that sound unit-like but were measured to often name a
separately-titled parcel in this cadastre (AP, TO, BLQ, CA, LC, OF, GA, PQ)
are still compared exactly as at unit level; every opaque/unstructured chunk
(including a kind with no usable value, such as ``AP`` alone) is also always
compared. When EITHER address has no plate nothing is ignored (the level
then behaves exactly like ``"unit"``), because without a plate the
complement is what identifies the predio. Only scores the same penalty model
already produces are possible. Complements are handled AFTER parsing, so an
address whose complement makes the parse fail (for example a complement that
is too long) is still a parse-failure veto at both levels. Warnings are
computed from the original complements and ``MatchResult.parsed_a`` /
``parsed_b`` are always the original parsed sides, never the ones actually
compared.

``level="structure"`` is the third, explicit definition ("same building" as
the master integration index defines it): it ignores, on both sides and when
both carry a plate, every UNIT kind (``STRUCTURE_UNIT_KINDS``: AP, LC, OF,
PISO, PH, BG, GA, PQ) but still distinguishes ``STRUCTURE_DISTINGUISHING_KINDS``
(TO, BLQ, INT, ET, AGR, MZ, SEC, CA, LT). A distinguishing kind present on ONE
side only is a veto (score 0.0): an unspecified tower is not assumed to be any
tower. ``"unit"`` and ``"building"`` are unchanged.

Identical strings: ``match(x, x)`` is a MATCH even when ``x`` is rejected by the
grammar only because its complement is too long or it holds several addresses (both
sides then carry exactly those parse notes, contain a digit and are equal after
``clean_text``); the result carries the warning ``cadena_identica``. Any other
unparseable input, identical or not, stays a parse-failure veto.

Quadrants: a trailing ``N`` / ``O`` / ``W`` letter (``AV 876 N``) equals the spelled quadrant
of the SAME segment (``AV 876 NORTE``); ``E`` and ``S`` stay street letters. A quadrant on the
via and one on the cross street are different places and are never equated. The fold only
affects the comparison; the plate / via / cross hard vetoes are untouched.

Zero runtime dependencies (only the standard library and this project's
own ``parser`` module).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from emparejador.parser import (
    COMPLEMENT_KINDS,
    MAX_PLATE_DIGITS,
    OPAQUE_KIND,
    QUADRANTS,
    CanonicalAddress,
    clean_text,
    parse_canonical,
    render,
)

__all__ = [
    "BUILDING_IGNORED_KINDS",
    "DEFAULT_LEVEL",
    "DEFAULT_PLATE_TOLERANCE",
    "DEFAULT_THRESHOLD",
    "FieldComparison",
    "InvalidLevelError",
    "InvalidThresholdError",
    "InvalidToleranceError",
    "LEVELS",
    "MAX_PLATE_DIGITS",
    "MatchResult",
    "STRUCTURE_DISTINGUISHING_KINDS",
    "STRUCTURE_UNIT_KINDS",
    "building_view",
    "match",
    "match_parsed",
    "structure_view",
]

DEFAULT_THRESHOLD = 0.90
DEFAULT_PLATE_TOLERANCE = 0
LEVELS = ("unit", "building", "structure")
DEFAULT_LEVEL = "unit"

# Field names that carry a hard veto when both sides are present and differ.
_HARD_VETO_FIELDS = ("via_type", "via_number", "cross_number")

# Field names that use the "one-sided 0.15 / conflict 0.30" soft penalty.
_SOFT_STRUCTURAL_FIELDS = (
    "via_letters",
    "via_suffix",
    "via_suffix_letters",
    "via_quadrant",
    "cross_letters",
    "cross_suffix",
    "cross_suffix_letters",
    "cross_quadrant",
)

# Field names that use the "one-sided 0.15, no conflict case" boolean penalty.
_BIS_FIELDS = ("via_bis", "cross_bis")

# The ONE registry of survival penalties. ``_score`` charges every penalty by
# reading this mapping at the point of application, and the dead-band proof
# in ``tests/test_matcher_review.py`` enumerates the reachable scores from
# it. Do not add a module-level ``_*_PENALTY`` constant: a test fails for any
# penalty that is not registered here.
_PENALTIES: dict[str, float] = {
    "one_sided": 0.15,
    "conflict": 0.30,
    "cross_type_one_sided": 0.03,
    "cross_type_conflict": 0.30,
    "plate_tolerance": 0.04,
    "complement_one_sided": 0.02,
}


class InvalidThresholdError(ValueError):
    """Raised when ``threshold`` is outside [0, 1] or is NaN."""


class InvalidLevelError(ValueError):
    """Raised when ``level`` is not one of ``LEVELS``."""


class InvalidToleranceError(ValueError):
    """Raised when ``plate_tolerance`` is negative or not an int."""


@dataclass(frozen=True)
class FieldComparison:
    """Outcome of comparing one field between two parsed addresses."""

    field: str
    a: str | None
    b: str | None
    status: str
    penalty: float
    veto: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "a": self.a,
            "b": self.b,
            "status": self.status,
            "penalty": self.penalty,
            "veto": self.veto,
        }


@dataclass(frozen=True)
class MatchResult:
    """Outcome of comparing two addresses."""

    is_match: bool
    score: float
    threshold: float
    reason: str
    parsed_a: CanonicalAddress
    parsed_b: CanonicalAddress
    fields: tuple[FieldComparison, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_match": self.is_match,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
            "parsed_a": _address_to_dict(self.parsed_a),
            "parsed_b": _address_to_dict(self.parsed_b),
            "fields": [f.to_dict() for f in self.fields],
            "warnings": list(self.warnings),
        }


_ADDRESS_FIELD_NAMES = (
    "via_type",
    "via_number",
    "via_letters",
    "via_suffix",
    "via_suffix_letters",
    "via_bis",
    "via_quadrant",
    "cross_type",
    "cross_number",
    "cross_letters",
    "cross_suffix",
    "cross_suffix_letters",
    "cross_bis",
    "cross_quadrant",
    "plate",
    "complement",
)


def _address_to_dict(addr: CanonicalAddress) -> dict[str, object]:
    """Build the JSON-friendly shape for one parsed address: the original
    input, whether it parsed, its rendered canonical form (falling back to
    the raw input when parsing failed), every ``CanonicalAddress`` field,
    and the parser's notes."""
    fields: dict[str, object] = {}
    for name in _ADDRESS_FIELD_NAMES:
        value = getattr(addr, name)
        fields[name] = list(value) if name == "complement" else value
    return {
        "input": addr.raw,
        "parse_ok": addr.parse_ok,
        "canonical": render(addr),
        "fields": fields,
        "notes": list(addr.notes),
    }


def _passes(score: float, threshold: float) -> bool:
    """Pure helper: does ``score`` clear ``threshold``."""
    return score >= threshold


def _validate_threshold(threshold: object) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise InvalidThresholdError(f"umbral inválido (tipo no numérico): {threshold!r}")
    value = float(threshold)
    if math.isnan(value):
        raise InvalidThresholdError("umbral inválido: NaN")
    if value < 0.0 or value > 1.0:
        raise InvalidThresholdError(f"umbral fuera de rango [0,1]: {value}")
    return value


def _validate_level(level: object) -> str:
    # ``in`` would accept equal-but-different types, so check the type first.
    if not isinstance(level, str) or level not in LEVELS:
        raise InvalidLevelError(f"level must be one of {LEVELS}: {level!r}")
    return level


def _validate_plate_tolerance(tolerance: object) -> int:
    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise InvalidToleranceError(f"tolerancia de placa inválida (debe ser entero): {tolerance!r}")
    if tolerance < 0:
        raise InvalidToleranceError(f"tolerancia de placa negativa: {tolerance}")
    return tolerance


def _compare_soft_structural(
    name: str, a: CanonicalAddress, b: CanonicalAddress
) -> FieldComparison:
    val_a = getattr(a, name)
    val_b = getattr(b, name)
    if val_a is None and val_b is None:
        return FieldComparison(name, val_a, val_b, "ausente_en_ambos", 0.0, False)
    if val_a == val_b:
        return FieldComparison(name, val_a, val_b, "igual", 0.0, False)
    if val_a is None or val_b is None:
        return FieldComparison(name, val_a, val_b, "solo_en_uno", _PENALTIES["one_sided"], False)
    return FieldComparison(name, val_a, val_b, "difiere", _PENALTIES["conflict"], False)


def _compare_bis(name: str, a: CanonicalAddress, b: CanonicalAddress) -> FieldComparison:
    val_a = bool(getattr(a, name))
    val_b = bool(getattr(b, name))
    label_a = "BIS" if val_a else None
    label_b = "BIS" if val_b else None
    if val_a == val_b:
        status = "ausente_en_ambos" if not val_a else "igual"
        return FieldComparison(name, label_a, label_b, status, 0.0, False)
    return FieldComparison(name, label_a, label_b, "solo_en_uno", _PENALTIES["one_sided"], False)


def _compare_hard_veto(name: str, a: CanonicalAddress, b: CanonicalAddress) -> FieldComparison:
    val_a = getattr(a, name)
    val_b = getattr(b, name)
    if val_a is None and val_b is None:
        return FieldComparison(name, val_a, val_b, "ausente_en_ambos", 0.0, False)
    if val_a == val_b:
        return FieldComparison(name, val_a, val_b, "igual", 0.0, False)
    if val_a is None or val_b is None:
        # cross_number missing on one side means "no cross street given",
        # which is itself a structural mismatch for this grammar.
        return FieldComparison(name, val_a, val_b, "solo_en_uno", 0.0, True)
    return FieldComparison(name, val_a, val_b, "difiere", 0.0, True)


def _compare_cross_type(a: CanonicalAddress, b: CanonicalAddress) -> FieldComparison:
    val_a = a.cross_type
    val_b = b.cross_type
    if val_a is None and val_b is None:
        return FieldComparison("cross_type", val_a, val_b, "ausente_en_ambos", 0.0, False)
    if val_a == val_b:
        return FieldComparison("cross_type", val_a, val_b, "igual", 0.0, False)
    if val_a is None or val_b is None:
        return FieldComparison(
            "cross_type", val_a, val_b, "solo_en_uno", _PENALTIES["cross_type_one_sided"], False
        )
    return FieldComparison(
        "cross_type", val_a, val_b, "difiere", _PENALTIES["cross_type_conflict"], False
    )


# The parser already rejects plates with more than ``MAX_PLATE_DIGITS``
# significant digits (finding 1); the same shared constant is checked again
# in ``_compare_plate`` in case a hand-built ``CanonicalAddress`` reaches it
# unchecked, so this module never converts an unbounded digit string to int.


def _compare_plate(
    a: CanonicalAddress, b: CanonicalAddress, plate_tolerance: int
) -> FieldComparison:
    val_a = a.plate
    val_b = b.plate
    if val_a is None and val_b is None:
        return FieldComparison("plate", val_a, val_b, "ausente_en_ambos", 0.0, False)
    if val_a is None or val_b is None:
        # A plate on only one side means one address is a corner (no
        # parcel), not the same parcel: hard veto.
        return FieldComparison("plate", val_a, val_b, "solo_en_uno", 0.0, True)
    if val_a == val_b:
        return FieldComparison("plate", val_a, val_b, "igual", 0.0, False)
    digits_a, letter_a = _split_plate(val_a)
    digits_b, letter_b = _split_plate(val_b)
    if letter_a != letter_b:
        # A plate letter marks a different corner/unit, not the same
        # parcel (e.g. "80" vs "80 A"): hard veto (finding 4/10).
        return FieldComparison("plate", val_a, val_b, "difiere", 0.0, True)
    if digits_a == digits_b:
        return FieldComparison("plate", val_a, val_b, "equivalente", 0.0, False)
    if (
        len(digits_a) > MAX_PLATE_DIGITS
        or len(digits_b) > MAX_PLATE_DIGITS
    ):
        return FieldComparison("plate", val_a, val_b, "difiere", 0.0, True)
    diff = abs(int(digits_a) - int(digits_b))
    if diff <= plate_tolerance:
        return FieldComparison(
            "plate", val_a, val_b, "dentro_tolerancia", _PENALTIES["plate_tolerance"], False
        )
    return FieldComparison("plate", val_a, val_b, "difiere", 0.0, True)


def _split_plate(plate: str) -> tuple[str, str]:
    """Split a plate token into its zero-stripped digit part and its
    optional trailing letter, without ever calling ``int()`` on a
    potentially unbounded digit string (finding 1)."""
    if plate and plate[-1].isalpha():
        digits, letter = plate[:-1], plate[-1]
    else:
        digits, letter = plate, ""
    stripped = digits.lstrip("0") or "0"
    return stripped, letter


def _is_opaque_chunk(kind: str, value: str) -> bool:
    """True when a chunk carries no usable structure.

    That is an unregistered kind, or a registered kind whose value is empty
    (``LT`` alone) or made only of quadrant words (``AP NORTE``): the value
    says nothing about which unit the chunk names, so it must not link to the
    bare address at the cheap known-complement price.
    """
    # A chunk keyed by a registered ALIAS (``SECTOR``, ``AGRUPACION`` kept as plain
    # text by the head-only rule) is not a canonical kind: it stays opaque.
    if COMPLEMENT_KINDS.get(kind) != kind:
        return True
    return value == "" or all(part in QUADRANTS for part in value.split("/"))


def _has_opaque_chunk(chunks: tuple[tuple[str, str], ...]) -> bool:
    """True when any chunk is opaque (see ``_is_opaque_chunk``)."""
    return any(_is_opaque_chunk(kind, value) for kind, value in chunks)


def _render_complement_chunks(chunks: tuple[tuple[str, str], ...]) -> str | None:
    """Render complement chunks as ``"KIND VALUE"`` pairs, joined by ``, ``,
    for human-readable comparisons and veto reasons (finding 10) instead
    of a raw tuple repr."""
    if not chunks:
        return None
    return ", ".join(
        (value if kind == OPAQUE_KIND else f"{kind} {value}").strip() for kind, value in sorted(chunks)
    )


def _compare_complement(a: CanonicalAddress, b: CanonicalAddress) -> FieldComparison:
    """Compare complement chunk multisets.

    The parser already normalizes each side's complement (aliases resolved,
    leading zeros stripped, chunks sorted), so chunks are compared as sorted
    tuples: order does not matter but a repeated chunk (``AP 1 AP 1``)
    counts. Only four cases remain: both empty (agreement); exactly equal (agreement); one
    side empty (a soft one-sided penalty, since a parcel can still match
    even if only one record specifies a unit; 0.15 instead of 0.02 when
    the present side has an opaque chunk); or anything else (a hard
    veto — a differing value for a shared kind and an extra kind on only
    one side both mean the two sides disagree about what complement the
    parcel has, so both collapse to the same "difiere" outcome).
    """
    set_a = tuple(sorted(a.complement))
    set_b = tuple(sorted(b.complement))
    rendered_a = _render_complement_chunks(a.complement)
    rendered_b = _render_complement_chunks(b.complement)
    if not set_a and not set_b:
        return FieldComparison("complement", rendered_a, rendered_b, "ausente_en_ambos", 0.0, False)
    if set_a == set_b:
        return FieldComparison("complement", rendered_a, rendered_b, "igual", 0.0, False)
    if not set_a or not set_b:
        present = a.complement or b.complement
        penalty = _PENALTIES["one_sided"] if _has_opaque_chunk(present) else _PENALTIES["complement_one_sided"]
        return FieldComparison("complement", rendered_a, rendered_b, "solo_en_uno", penalty, False)
    return FieldComparison("complement", rendered_a, rendered_b, "difiere", 0.0, True)


def _parse_failure_result(
    a: CanonicalAddress, b: CanonicalAddress, threshold: float
) -> MatchResult:
    notes: list[str] = []
    if not a.parse_ok:
        notes.append(f"dirección A no canónica ({', '.join(a.notes) or 'inválida'})")
    if not b.parse_ok:
        notes.append(f"dirección B no canónica ({', '.join(b.notes) or 'inválida'})")
    reason = "veto: " + "; ".join(notes)
    return MatchResult(
        is_match=False,
        score=0.0,
        threshold=threshold,
        reason=reason,
        parsed_a=a,
        parsed_b=b,
        fields=(),
        warnings=(),
    )


def match(
    a: object,
    b: object,
    threshold: float = DEFAULT_THRESHOLD,
    plate_tolerance: int = DEFAULT_PLATE_TOLERANCE,
    *,
    level: str = DEFAULT_LEVEL,
) -> MatchResult:
    """Compare two (ideally already-normalized) address strings.

    Never raises on bad address input: unparseable input on either side
    is scored as a veto (score 0.0, no match). Raises
    ``InvalidThresholdError`` / ``InvalidToleranceError`` /
    ``InvalidLevelError`` for bad settings. ``level="building"`` ignores only BG/INT/PISO/PH,
    ``level="structure"`` ignores every unit kind but still distinguishes
    tower/block/interior/etc. (see the module docstring).
    """
    validated_threshold = _validate_threshold(threshold)
    validated_tolerance = _validate_plate_tolerance(plate_tolerance)
    validated_level = _validate_level(level)
    return _score(
        parse_canonical(a), parse_canonical(b), validated_threshold, validated_tolerance, validated_level
    )


def match_parsed(
    a: CanonicalAddress,
    b: CanonicalAddress,
    threshold: float = DEFAULT_THRESHOLD,
    plate_tolerance: int = DEFAULT_PLATE_TOLERANCE,
    *,
    level: str = DEFAULT_LEVEL,
) -> MatchResult:
    """Same as ``match`` for addresses that are already parsed.

    Lets callers that compare one address against many (see
    ``emparejador.index``) parse each address once. Settings are validated
    exactly like ``match``.
    """
    validated_threshold = _validate_threshold(threshold)
    validated_tolerance = _validate_plate_tolerance(plate_tolerance)
    validated_level = _validate_level(level)
    return _score(a, b, validated_threshold, validated_tolerance, validated_level)


# Complement kinds that denote a unit INSIDE one building/predio and are
# ignored at ``level="building"`` (only when both sides carry a plate); every
# other kind, and every opaque chunk, keeps identifying the predio and is
# still compared.
#
# This set is deliberately much smaller than "every kind that sounds like a
# unit". A cadastre-wide check against ``numero_predial_nacional`` (the first
# 21 characters identify the land parcel; see ``tools/eval_realworld.py``)
# measured, for every candidate kind, how often two records that share the
# exact same via/cross/plate but differ on that kind's value ALSO differ in
# their land parcel (a real different predio, not a unit of the same one):
#
#   AP, TO, LC, OF and BLQ often differ in land parcel; CA, GA and PQ do so
#   less often; INT and BG had no differing pairs (thin samples);
#   PISO/PH: no cadastre occurrences to test
#
# In this cadastre, apartments, towers, locals, offices, blocks, garages and
# parking spots routinely get their OWN predial number even at an identical
# street address (Colombia's horizontal-property regime assigns each unit an
# independent folio, not a suffix of the matriz's number here), so blanket
# treating them as "the same building" would produce real false MATCHes
# (worse than a miss, per this project's design). AP/TO/LC/OF/BLQ/CA/GA/PQ are
# therefore kept as identifying, same as MZ/LT/CGTO/VDA/BR/UR/CONJ/ED/ET.
# Only BG (bodega/storage) and INT (interior) showed zero measured risk (thin
# samples, but rare and structurally a sub-unit, not a titled parcel), and
# PISO (floor) / PH (horizontal-property flag) have no data to contradict
# their sub-unit semantics, so those four are ignored.
BUILDING_IGNORED_KINDS = frozenset({"BG", "INT", "PISO", "PH"})


def _drop_unit_complements(addr: CanonicalAddress) -> CanonicalAddress:
    """Return ``addr`` without its unit-inside-a-building complement chunks.

    Opaque chunks are never dropped: an unregistered kind, or a registered
    kind with no usable value (``AP`` alone, ``AP NORTE``), says nothing about
    which unit the chunk names, so it must keep costing what it costs at unit
    level.
    """
    kept = tuple(
        chunk
        for chunk in addr.complement
        if chunk[0] not in BUILDING_IGNORED_KINDS or _is_opaque_chunk(*chunk)
    )
    if len(kept) == len(addr.complement):
        return addr
    return replace(addr, complement=kept)


# ``level="structure"``: the definition the master integration index uses for
# "same building". UNIT kinds name a unit INSIDE a building and are ignored on
# both sides (when both addresses carry a plate); DISTINGUISHING kinds name a
# physical sub-division of the site (tower, block, interior, stage, grouping,
# block of lots, sector, house, lot): a different value is a different
# building, and a distinguishing kind present on ONE side only is NOT a match
# (veto, score 0.0), because "tower unspecified" cannot be assumed to be any of
# the towers. Every other registered kind (ED, CONJ, UR, BR, CGTO, VDA) and
# every opaque chunk keeps its ``unit``-level price. Unlike ``"building"`` this
# level does NOT rest on the parcel-number measurement: it states a structural
# identity, not "same predio". Kinds listed in neither set (HAB, deposito,
# consultorio) are not registered in the parser and stay opaque.
STRUCTURE_UNIT_KINDS = frozenset({"AP", "LC", "OF", "PISO", "PH", "BG", "GA", "PQ"})
STRUCTURE_DISTINGUISHING_KINDS = frozenset({"TO", "BLQ", "INT", "ET", "AGR", "MZ", "SEC", "CA", "LT"})


def structure_view(addr: CanonicalAddress) -> CanonicalAddress:
    """The address as ``level="structure"`` compares it: UNIT-kind complement
    chunks removed (opaque chunks never), but only when the address has a
    plate."""
    if not addr.plate:
        return addr
    kept = tuple(
        chunk
        for chunk in addr.complement
        if chunk[0] not in STRUCTURE_UNIT_KINDS or _is_opaque_chunk(*chunk)
    )
    if len(kept) == len(addr.complement):
        return addr
    return replace(addr, complement=kept)


def building_view(addr: CanonicalAddress) -> CanonicalAddress:
    """The address as ``level="building"`` compares it: unit-inside-a-building
    complement chunks removed, but only when the address has a plate (without
    one the complement is what identifies the predio)."""
    if not addr.plate:
        return addr
    return _drop_unit_complements(addr)


# Parse notes that only say "the grammar cannot take this string whole", not "this is
# not an address": a complement over the length limit, or several addresses in one string.
_IDENTICAL_OK_NOTES = frozenset({"complemento_demasiado_largo", "multiples_direcciones"})


_SEGMENT_SPLIT = re.compile(r"\s*;\s*|\s*//\s*|\s+Y\s+")


def _identity_key(raw: str) -> str:
    """``raw`` after ``clean_text`` with ``;`` spaced out and whitespace collapsed."""
    return " ".join(re.sub(r"\s*;\s*", " ; ", clean_text(raw)).split())


def _is_identical_unparseable(a: CanonicalAddress, b: CanonicalAddress) -> bool:
    """True for two equal strings the grammar rejects only for length or multiplicity.

    Requires a digit so that identical junk (``SIN ESPECIFICAR``) never matches. A
    multi-address string must really hold two or more addresses that each parse on their
    own (``KR 1 # 4 - 80 # 5`` is junk, not two addresses).
    """
    for side in (a, b):
        if side.parse_ok or not side.notes or not set(side.notes) <= _IDENTICAL_OK_NOTES:
            return False
        if not isinstance(side.raw, str):
            return False
    key_a, key_b = _identity_key(a.raw), _identity_key(b.raw)
    if key_a != key_b or not any(ch.isdigit() for ch in key_a):
        return False
    if "multiples_direcciones" in a.notes or "multiples_direcciones" in b.notes:
        segments = [seg for seg in _SEGMENT_SPLIT.split(key_a) if seg.strip()]
        return len(segments) >= 2 and all(parse_canonical(seg).parse_ok for seg in segments)
    return True


# A trailing cardinal letter the parser keeps as a street letter is the quadrant the
# cadastre spells out (``AV 6 N`` == ``AV 6 NORTE``). ``E``/``S`` are real street letters.
_CARDINAL_LETTER_QUADRANT = {"N": "NORTE", "O": "OESTE", "W": "OESTE"}


def _fold_segment(
    letters_a: str | None, quadrant_a: str | None, letters_b: str | None, quadrant_b: str | None
) -> tuple[str | None, str | None, str | None, str | None]:
    """Fold a cardinal letter into the spelled quadrant of the SAME segment, one-way per
    side and only when the other side spells exactly that quadrant."""

    def one_way(letters, quadrant, other_quadrant):
        if quadrant is None and letters and other_quadrant is not None:
            parts = letters.split()
            if _CARDINAL_LETTER_QUADRANT.get(parts[-1]) == other_quadrant:
                return (" ".join(parts[:-1]) or None), other_quadrant
        return letters, quadrant

    letters_a, quadrant_a = one_way(letters_a, quadrant_a, quadrant_b)
    letters_b, quadrant_b = one_way(letters_b, quadrant_b, quadrant_a)
    return letters_a, quadrant_a, letters_b, quadrant_b


def _segment_prefix(addr: CanonicalAddress, side: str) -> tuple[object, ...]:
    """Everything a segment says besides its letters and quadrant."""
    return (
        getattr(addr, f"{side}_type", None) if side == "cross" else addr.via_type,
        getattr(addr, f"{side}_number"),
        getattr(addr, f"{side}_suffix"),
        getattr(addr, f"{side}_suffix_letters"),
        bool(getattr(addr, f"{side}_bis")),
    )


def _fold_quadrant_pair(
    a: CanonicalAddress, b: CanonicalAddress
) -> tuple[CanonicalAddress, CanonicalAddress]:
    """Compare ``AV 6 N`` equal to ``AV 6 NORTE`` (same segment, same prefix) and nothing else.

    A quadrant on the via and one on the cross street are different places
    (``CL 5 N # 10 - 20`` is Calle 5 Norte; ``CL 5 # 10 N - 20`` crosses Carrera 10 Norte),
    so the quadrant is never moved between segments. The fold is all-or-nothing: it applies
    only when, in BOTH segments, everything except the cardinal letter is identical on both
    sides (type, number, suffixes, BIS, street letters) and the folded quadrants then agree.
    In every other case both sides are returned untouched, so scores and reasons are exactly
    the ones the grammar alone produces and a lone matching N/NORTE can never make
    different segments look alike.
    """
    changes_a: dict[str, str | None] = {}
    changes_b: dict[str, str | None] = {}
    for side in ("via", "cross"):
        if _segment_prefix(a, side) != _segment_prefix(b, side):
            return a, b
        la, qa, lb, qb = _fold_segment(
            getattr(a, f"{side}_letters"), getattr(a, f"{side}_quadrant"),
            getattr(b, f"{side}_letters"), getattr(b, f"{side}_quadrant"),
        )
        if (la, qa) != (lb, qb):
            return a, b
        if (la, qa) != (getattr(a, f"{side}_letters"), getattr(a, f"{side}_quadrant")):
            changes_a[f"{side}_letters"], changes_a[f"{side}_quadrant"] = la, qa
        if (lb, qb) != (getattr(b, f"{side}_letters"), getattr(b, f"{side}_quadrant")):
            changes_b[f"{side}_letters"], changes_b[f"{side}_quadrant"] = lb, qb
    return (replace(a, **changes_a) if changes_a else a, replace(b, **changes_b) if changes_b else b)


def _score(
    parsed_a: CanonicalAddress,
    parsed_b: CanonicalAddress,
    validated_threshold: float,
    validated_tolerance: int,
    level: str = DEFAULT_LEVEL,
) -> MatchResult:
    if not parsed_a.parse_ok or not parsed_b.parse_ok:
        if _is_identical_unparseable(parsed_a, parsed_b):
            return MatchResult(
                is_match=_passes(1.0, validated_threshold),
                score=1.0,
                threshold=validated_threshold,
                reason="coincidencia exacta (cadenas idénticas)",
                parsed_a=parsed_a,
                parsed_b=parsed_b,
                fields=(),
                warnings=("cadena_identica",),
            )
        return _parse_failure_result(parsed_a, parsed_b, validated_threshold)

    # ``parsed_a`` / ``parsed_b`` stay the ORIGINAL parsed sides (returned in
    # the result, and the source of the warnings); ``cmp_a`` / ``cmp_b`` are
    # what the fields are compared on.
    cmp_a, cmp_b = _fold_quadrant_pair(parsed_a, parsed_b)
    if level == "building" and cmp_a.plate and cmp_b.plate:
        cmp_a = building_view(cmp_a)
        cmp_b = building_view(cmp_b)
    elif level == "structure" and cmp_a.plate and cmp_b.plate:
        cmp_a = structure_view(cmp_a)
        cmp_b = structure_view(cmp_b)

    comparisons: list[FieldComparison] = []
    for name in _HARD_VETO_FIELDS:
        comparisons.append(_compare_hard_veto(name, cmp_a, cmp_b))
    comparisons.append(_compare_plate(cmp_a, cmp_b, validated_tolerance))
    complement = _compare_complement(cmp_a, cmp_b)
    if level == "structure" and complement.status == "solo_en_uno":
        present = cmp_a.complement or cmp_b.complement
        if any(kind in STRUCTURE_DISTINGUISHING_KINDS for kind, _ in present):
            complement = replace(complement, penalty=0.0, veto=True)
    comparisons.append(complement)
    for name in _SOFT_STRUCTURAL_FIELDS:
        comparisons.append(_compare_soft_structural(name, cmp_a, cmp_b))
    for name in _BIS_FIELDS:
        comparisons.append(_compare_bis(name, cmp_a, cmp_b))
    comparisons.append(_compare_cross_type(cmp_a, cmp_b))

    # Open-ended addresses (finding 8): missing on both sides is scored as
    # agreement, per the field table, but it is still worth flagging that
    # the comparison rests on less structure than usual.
    by_field = {c.field: c for c in comparisons}
    warnings: list[str] = []
    if by_field["plate"].status == "ausente_en_ambos":
        warnings.append("sin_placa")
    if by_field["cross_number"].status == "ausente_en_ambos":
        warnings.append("sin_cruce")
    if _has_opaque_chunk(parsed_a.complement) or _has_opaque_chunk(parsed_b.complement):
        warnings.append("complemento_no_estructurado")
    result_warnings = tuple(warnings)

    vetoed_fields = [c for c in comparisons if c.veto]
    if vetoed_fields:
        reason = "veto: " + "; ".join(_veto_reason(c) for c in vetoed_fields)
        return MatchResult(
            is_match=False,
            score=0.0,
            threshold=validated_threshold,
            reason=reason,
            parsed_a=parsed_a,
            parsed_b=parsed_b,
            fields=tuple(comparisons),
            warnings=result_warnings,
        )

    score = 1.0
    for comparison in comparisons:
        score *= 1.0 - comparison.penalty
    score = round(score, 4)

    is_match = _passes(score, validated_threshold)
    reason = _build_reason(score, validated_threshold, comparisons, is_match)

    return MatchResult(
        is_match=is_match,
        score=score,
        threshold=validated_threshold,
        reason=reason,
        parsed_a=parsed_a,
        parsed_b=parsed_b,
        fields=tuple(comparisons),
        warnings=result_warnings,
    )


def _veto_reason(comparison: FieldComparison) -> str:
    label = _FIELD_LABELS.get(comparison.field, comparison.field)
    if comparison.status == "solo_en_uno":
        return f"{label} presente solo en un lado"
    return f"{label} difiere ({comparison.a} vs {comparison.b})"


_FIELD_LABELS: dict[str, str] = {
    "via_type": "tipo de vía",
    "via_number": "número de vía",
    "cross_number": "número de vía generadora",
    "plate": "placa",
    "complement": "complemento",
}


def _build_reason(
    score: float, threshold: float, comparisons: list[FieldComparison], is_match: bool
) -> str:
    # Only fields with an actual, nonzero penalty should show up as
    # "differences" in the reason text: status "equivalente" (e.g. plate
    # "80" vs "080") carries zero penalty and must still read as an exact
    # match, since it does not affect the score at all.
    differing = [c for c in comparisons if c.penalty > 0.0]
    if not differing:
        return "coincidencia exacta"
    if is_match:
        names = ", ".join(c.field for c in differing)
        return f"coincide; diferencias menores: {names}"
    names = ", ".join(_describe_difference(c) for c in differing)
    return f"puntaje {score:.4f} < umbral {threshold:g}: {names}"


def _describe_difference(comparison: FieldComparison) -> str:
    if comparison.status == "solo_en_uno":
        return f"{comparison.field} solo en un lado"
    return f"{comparison.field} difiere"
