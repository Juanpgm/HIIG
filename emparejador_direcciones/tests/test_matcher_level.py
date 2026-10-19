"""Tests for the ``level`` option of ``emparejador.matcher`` ("unit" | "building")."""

from __future__ import annotations

import hashlib
import itertools

import pytest

from emparejador import AddressIndex, match, match_parsed, parse_canonical
from emparejador import matcher as matcher_module
from emparejador.parser import COMPLEMENT_KINDS

# ---------------------------------------------------------------------------
# Synthetic sample used to lock the default ("unit") behavior byte for byte.
# ---------------------------------------------------------------------------

_BASES = (
    "KR 26 # 73 - 10",
    "KR 26 A # 73 - 10",
    "KR 26 BIS # 73 - 10",
    "KR 26 # 73 - 12",
    "KR 26 # 73 - 10 A",
    "CL 5 # 10 - 20",
    "CL 5 N # 10 - 20",
    "CL 5 # 11 - 20",
    "AV 3 N # 50 - 05",
    "KR 1 # 4",
    "KR 1 # 4 - 80",
)
_TAILS = ("", " AP 101", " AP 102", " BLQ 2", " AP 101 BLQ 2", " CS 3", " CA 4", " LC 7", " AP", " 8 C")

_SAMPLE = tuple(base + tail for base in _BASES for tail in _TAILS) + (
    "",
    "   ",
    "garbage",
    "KR 26 73 10",
    "KR 26 # 73 - 10 // KR 26 # 74 - 10",
)

# sha256 over "score|is_match|reason|warnings" of every ordered pair of _SAMPLE,
# computed with the matcher BEFORE the ``level`` option existed. Re-baselined when
# the sample's "KR 1 # 9 ..." placeholder became "KR 1 # 4 ..." (privacy cleanup):
# every pair was verified to keep the same score/is_match/warnings and the same
# reason up to the cross-number digit, and the old sample still reproduced the
# previous digest (7b2380c6...) with the unchanged matcher.
#
# Re-baselined (2026-10) for ONE intentional change, verified pair by pair against the
# previous matcher over all of _SAMPLE: the identical two-address string
# ``KR 26 # 73 - 10 // KR 26 # 74 - 10`` went from veto 0.0 to MATCH 1.0 (``match(x, x)``
# must be a MATCH). Every other pair is byte-identical. Previous digest: 2fa97068...
_UNIT_GOLDEN_PAIRS = len(_SAMPLE) ** 2
_UNIT_GOLDEN_DIGEST = "135f47332b2137a8619713d5b59cc7fe78d0c842dc61b67ace23c939578673e0"
_IDENTICAL_TWO_ADDRESSES = "KR 26 # 73 - 10 // KR 26 # 74 - 10"


def _digest(**kwargs: object) -> str:
    h = hashlib.sha256()
    for a, b in itertools.product(_SAMPLE, repeat=2):
        r = match(a, b, **kwargs)
        h.update(f"{r.score!r}|{r.is_match}|{r.reason}|{','.join(r.warnings)}\n".encode())
    return h.hexdigest()


def test_default_level_is_byte_identical_to_the_pre_level_matcher() -> None:
    assert _digest() == _UNIT_GOLDEN_DIGEST


def test_explicit_unit_level_equals_the_default() -> None:
    assert _digest(level="unit") == _UNIT_GOLDEN_DIGEST


# ---------------------------------------------------------------------------
# Building level: complements are dropped on BOTH sides
# ---------------------------------------------------------------------------

_BUILDING = "KR 26 # 73 - 10"


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 26 # 73 - 10 BG 1", "KR 26 # 73 - 10 BG 2"),
        ("KR 26 # 73 - 10 INT 1", "KR 26 # 73 - 10 INT 2"),
        ("KR 26 # 73 - 10 PISO 2", "KR 26 # 73 - 10 PISO 3"),
        ("KR 26 # 73 - 10 PH 1", "KR 26 # 73 - 10 PH 2"),
        ("KR 26 # 73 - 10 BG 1", "KR 26 # 73 - 10 INT 7"),  # different kinds
        ("KR 26 # 73 - 10 BG 101 PISO 2", "KR 26 # 73 - 10 INT 3"),
        ("KR 26 # 73 - 10 BG 101 LT 5", "KR 26 # 73 - 10 BG 102 LT 5"),  # identifying kind equal
    ],
)
def test_building_level_ignores_unit_kind_complement_differences(a: str, b: str) -> None:
    assert match(a, b).score == 0.0  # the unit level vetoes
    result = match(a, b, level="building")
    assert result.score == 1.0
    assert result.is_match is True


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 1 # 4 - 80 LT 5", "KR 1 # 4 - 80 LT 6"),
        ("KR 1 # 4 - 80 MZ 3 CA 4", "KR 1 # 4 - 80 MZ 7 CA 2"),
        ("KR 1 # 4 - 80 MZ 3", "KR 1 # 4 - 80 MZ 7"),
        ("CL 1 # 2 - 7 CGTO X", "CL 1 # 2 - 7 CGTO Y"),
        ("CL 1 # 2 - 7 VDA X", "CL 1 # 2 - 7 VDA Y"),
        ("CL 1 # 2 - 7 BR X", "CL 1 # 2 - 7 BR Y"),
        ("CL 1 # 2 - 7 ED X", "CL 1 # 2 - 7 ED Y"),
        ("CL 1 # 2 - 7 UR X", "CL 1 # 2 - 7 UR Y"),
        ("CL 1 # 2 - 7 CONJ X", "CL 1 # 2 - 7 CONJ Y"),
        ("CL 1 # 2 - 7 ET 1", "CL 1 # 2 - 7 ET 2"),
        ("KR 1 # 4 - 80 8 C", "KR 1 # 4 - 80 9 D"),
        ("KR 1 # 4 - 80 CS 3", "KR 1 # 4 - 80 CS 4"),
        ("KR 1 # 4 - 80 BG 1 LT 5", "KR 1 # 4 - 80 BG 1 LT 6"),
        ("KR 1 # 4 - 80 BG 1 LT 5", "KR 1 # 4 - 80 BG 1 MZ 5"),
        # AP/TO/BLQ/CA/LC/OF/GA/PQ measured to often name a different land
        # parcel even at an identical street address (see BUILDING_IGNORED_KINDS'
        # comment): they stay identifying, unlike BG/INT/PISO/PH.
        ("KR 26 # 73 - 10 AP 101", "KR 26 # 73 - 10 AP 102"),
        ("KR 26 # 73 - 10 TO 1", "KR 26 # 73 - 10 TO 2"),
        ("KR 26 # 73 - 10 BLQ 2", "KR 26 # 73 - 10 BLQ 3"),
        ("KR 26 # 73 - 10 CA 4", "KR 26 # 73 - 10 CA 5"),
        ("KR 26 # 73 - 10 LC 7", "KR 26 # 73 - 10 LC 8"),
        ("KR 26 # 73 - 10 OF 7", "KR 26 # 73 - 10 OF 8"),
        ("KR 26 # 73 - 10 GA 1", "KR 26 # 73 - 10 GA 2"),
        ("KR 26 # 73 - 10 PQ 1", "KR 26 # 73 - 10 PQ 2"),
    ],
)
def test_building_level_keeps_identifying_complements(a: str, b: str) -> None:
    assert match(a, b, level="building").score == 0.0
    assert match(a, b, level="building").is_match is False


@pytest.mark.parametrize("tail", ["8 C", "AB", "82", "CS 3", "LT 5", "MZ 3", "ED X", "AP", "BG"])
def test_building_level_does_not_silently_link_identifying_or_opaque_tail_to_bare(tail: str) -> None:
    for a, b in ((f"KR 1 # 4 - 80 {tail}", "KR 1 # 4 - 80"), ("KR 1 # 4 - 80", f"KR 1 # 4 - 80 {tail}")):
        unit = match(a, b)
        building = match(a, b, level="building")
        assert building.score == unit.score
        assert building.reason == unit.reason
    opaque = tail in ("8 C", "AB", "82", "CS 3", "AP", "BG")
    result = match(f"KR 1 # 4 - 80 {tail}", "KR 1 # 4 - 80", level="building")
    assert result.is_match is (not opaque)
    assert ("complemento_no_estructurado" in result.warnings) is opaque


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 1 # 4 AP 1", "KR 1 # 4 AP 2"),
        ("KR 1 # 4 - AP 1", "KR 1 # 4 - 80 AP 2"),
        ("KR 1 # 4 - 80 AP 1", "KR 1 # 4 AP 2"),
        ("KR 1 # 4 - 80 AP 1", "KR 1 # 4"),
        ("KR 1 # 4", "KR 1 # 4 - 80 AP 1"),
        ("KR 1 # 4 8 C", "KR 1 # 4"),
        ("KR 1 # 4 LT 5", "KR 1 # 4 LT 6"),
        ("KR 1 # 4 AP 1", "KR 1 # 4"),
    ],
)
def test_building_level_without_a_plate_on_either_side_equals_unit_level(a: str, b: str) -> None:
    unit = match(a, b)
    building = match(a, b, level="building")
    assert (building.score, building.is_match, building.reason, building.warnings) == (
        unit.score,
        unit.is_match,
        unit.reason,
        unit.warnings,
    )


def test_building_level_without_plate_never_drops_ignored_kinds() -> None:
    assert match("KR 1 # 4 BG 1", "KR 1 # 4 BG 2", level="building").score == 0.0
    assert match("KR 1 # 4 - 80 BG 1", "KR 1 # 4 - 80 BG 2", level="building").score == 1.0


@pytest.mark.parametrize("kind", ["INT", "BG", "PISO", "PH"])
def test_no_plate_guard_prevents_a_false_match_for_every_ignored_kind(kind: str) -> None:
    # Without the "no plate on either side" guard, dropping an ignored kind
    # here would leave nothing to distinguish these two addresses, and they
    # would score 1.0 even though nothing proves they name the same predio.
    # The '-' before the kind matters: it is what makes the complement the
    # ONLY thing that differs (without it the two sides are not exercising
    # the same code path; see the guard's own dash-vs-no-dash behaviour).
    a, b = f"KR 1 # 4 - {kind} 1", f"KR 1 # 4 - {kind} 2"
    unit = match(a, b)
    building = match(a, b, level="building")
    assert unit.score == 0.0
    assert (building.score, building.is_match) == (0.0, False)


def test_building_level_veto_still_returns_the_original_parsed_sides() -> None:
    # A veto (different plate) must not leak the stripped "building view":
    # parsed_a/parsed_b are the real, original parsed addresses even on the
    # branch that never reaches the shared-scoring code.
    a, b = "KR 1 # 4 - 80 BG 1", "KR 1 # 4 - 81 BG 2"
    result = match(a, b, level="building")
    assert result.score == 0.0
    assert result.parsed_a == parse_canonical(a)
    assert result.parsed_b == parse_canonical(b)
    assert ("BG", "1") in result.parsed_a.complement
    assert ("BG", "2") in result.parsed_b.complement


def test_building_level_keeps_warnings_computed_from_the_original_complements() -> None:
    result = match("KR 1 # 4 - 80 CS 3 BG 1", "KR 1 # 4 - 80 CS 3 BG 2", level="building")
    assert result.is_match is True
    assert result.warnings == ("complemento_no_estructurado",)
    bare = match("KR 1 # 4 - 80 AB", "KR 1 # 4 - 80", level="building")
    assert "complemento_no_estructurado" in bare.warnings
    assert match("KR 1 # 4 - 80 BG 1", "KR 1 # 4 - 80 BG 2", level="building").warnings == ()


def test_building_level_parsed_sides_are_the_original_parsed_addresses() -> None:
    a, b = "KR 1 # 4 - 80 BG 1 LT 5", "KR 1 # 4 - 80 BG 2 LT 5"
    result = match(a, b, level="building")
    assert result.parsed_a == parse_canonical(a)
    assert result.parsed_b == parse_canonical(b)
    assert ("BG", "1") in result.parsed_a.complement
    assert ("BG", "2") in result.parsed_b.complement
    pa, pb = parse_canonical(a), parse_canonical(b)
    again = match_parsed(pa, pb, level="building")
    assert again.parsed_a is pa and again.parsed_b is pb


def test_building_level_unit_kinds_table_is_exactly_the_documented_set() -> None:
    # This set is deliberately small: a cadastre-wide check (see the constant's
    # own comment in matcher.py) showed most "sounds like a unit" kinds
    # (AP, TO, BLQ, CA, LC, OF, GA, PQ) routinely name a different land parcel
    # even at the same street address, so they are NOT ignored here.
    assert matcher_module.BUILDING_IGNORED_KINDS == frozenset({"BG", "INT", "PISO", "PH"})
    assert matcher_module.BUILDING_IGNORED_KINDS <= set(COMPLEMENT_KINDS.values())
    for kept in ("MZ", "LT", "CGTO", "VDA", "BR", "UR", "CONJ", "ED", "ET",
                 "AP", "TO", "BLQ", "CA", "LC", "OF", "GA", "PQ"):
        assert kept not in matcher_module.BUILDING_IGNORED_KINDS


def test_building_level_one_sided_ignored_kind_complement_is_a_full_match() -> None:
    for a, b in ((f"{_BUILDING} BG 101", _BUILDING), (_BUILDING, f"{_BUILDING} BG 101")):
        assert match(a, b).score == pytest.approx(0.98)
        assert match(a, b, level="building").score == 1.0


def test_building_level_one_sided_identifying_kind_stays_the_same_at_both_levels() -> None:
    for a, b in ((f"{_BUILDING} AP 101", _BUILDING), (_BUILDING, f"{_BUILDING} AP 101")):
        assert match(a, b).score == pytest.approx(0.98)
        assert match(a, b, level="building").score == pytest.approx(0.98)


def test_building_level_opaque_complement_is_still_penalised() -> None:
    assert match(f"{_BUILDING} 8 C", _BUILDING).score == pytest.approx(0.85)
    assert match(f"{_BUILDING} 8 C", _BUILDING, level="building").score == pytest.approx(0.85)


def test_building_level_both_bare_is_unchanged() -> None:
    assert match(_BUILDING, _BUILDING, level="building").score == 1.0
    assert match(_BUILDING, _BUILDING, level="building").reason == "coincidencia exacta"


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 26 # 73 - 10", "KR 26 # 73 - 12"),  # plate
        ("KR 26 # 73 - 10", "KR 26 # 73 - 10 A"),  # plate letter
        ("KR 26 # 73 - 10", "KR 27 # 73 - 10"),  # via number
        ("KR 26 # 73 - 10", "CL 26 # 73 - 10"),  # via type
        ("KR 26 # 73 - 10", "KR 26 # 74 - 10"),  # cross number
        ("KR 26 # 73 - 10 AP 1", "KR 26 # 73 - 12 AP 1"),  # equal complements do not rescue it
        ("KR 26 # 73 - 10 AP 1", "KR 26 # 73 - 12 AP 2"),
    ],
)
def test_building_level_does_not_ignore_plate_via_or_cross(a: str, b: str) -> None:
    assert match(a, b, level="building").score == 0.0
    assert match(a, b, level="building").is_match is False


def test_building_level_keeps_soft_structural_penalties() -> None:
    assert match("KR 26 # 73 - 10", "KR 26 A # 73 - 10", level="building").score == pytest.approx(0.85)
    assert match("KR 26 # 73 - 10 BG 1", "KR 26 A # 73 - 10 BG 2", level="building").score == pytest.approx(0.85)


@pytest.mark.parametrize("bad", [None, "", "   ", "garbage", "KR 26 73 10", 12345, b"KR 26 # 73 - 10"])
def test_parse_failure_vetoes_at_both_levels(bad: object) -> None:
    for level in ("unit", "building"):
        for a, b in ((bad, _BUILDING), (_BUILDING, bad), (bad, bad)):
            result = match(a, b, level=level)
            assert result.score == 0.0
            assert result.is_match is False
            assert result.reason.startswith("veto:")


def test_building_level_does_not_rescue_a_complement_that_fails_to_parse() -> None:
    # Documented limitation: complements are dropped AFTER parsing, so an
    # address whose complement makes the parse fail still vetoes.
    too_long = _BUILDING + " AP 1 BLQ 2 CS 3 CA 4 LC 5 AP 6 AP 7 AP 8 AP 9 AP 10 AP 11 AP 12"
    parsed = parse_canonical(too_long)
    assert not parsed.parse_ok
    assert match(too_long, _BUILDING, level="building").score == 0.0


def test_building_level_is_symmetric_over_the_sample() -> None:
    for a, b in itertools.product(_SAMPLE, repeat=2):
        assert match(a, b, level="building").score == match(b, a, level="building").score, (a, b)


def test_building_level_is_idempotent_over_the_sample() -> None:
    for a in _SAMPLE:
        first = match(a, a, level="building")
        # an identical string holding two valid addresses is a MATCH too (see the golden note)
        expected = 1.0 if parse_canonical(a).parse_ok or a == _IDENTICAL_TWO_ADDRESSES else 0.0
        assert first.score == expected
        # dropping complements twice is the same as once
        again = match_parsed(parse_canonical(a), parse_canonical(a), level="building")
        assert again.score == first.score


def test_building_score_is_never_below_the_unit_score_when_unit_is_not_vetoed() -> None:
    for a, b in itertools.product(_SAMPLE, repeat=2):
        unit = match(a, b).score
        building = match(a, b, level="building").score
        if unit > 0.0:
            assert building >= unit, (a, b)


def test_threshold_boundary_at_building_level() -> None:
    a, b = "KR 26 # 73 - 10", "KR 26 A # 73 - 10"  # score 0.85
    assert match(a, b, threshold=0.85, level="building").is_match is True
    assert match(a, b, threshold=0.9, level="building").is_match is False
    assert match(_BUILDING, _BUILDING, threshold=1.0, level="building").is_match is True


def test_building_level_reports_original_complements_in_parsed_side() -> None:
    result = match(f"{_BUILDING} AP 101", _BUILDING, level="building")
    assert result.parsed_a.complement == (("AP", "101"),)
    assert "complemento_no_estructurado" not in result.warnings


def test_plate_tolerance_still_applies_at_building_level() -> None:
    result = match("KR 26 # 73 - 10 BG 1", "KR 26 # 73 - 11 BG 2", plate_tolerance=1, level="building")
    assert result.score == pytest.approx(0.96)


# ---------------------------------------------------------------------------
# Invalid level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "Building", "BUILDING", "floor", None, 1, True, b"unit", ["unit"]])
def test_invalid_level_raises_a_clear_value_error(bad: object) -> None:
    with pytest.raises(ValueError, match="level"):
        match(_BUILDING, _BUILDING, level=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level"):
        match_parsed(parse_canonical(_BUILDING), parse_canonical(_BUILDING), level=bad)  # type: ignore[arg-type]


def test_invalid_level_raises_even_when_an_address_is_unparseable() -> None:
    with pytest.raises(ValueError, match="level"):
        match(None, None, level="nope")  # type: ignore[arg-type]


def test_level_is_keyword_only_position_is_stable() -> None:
    # A fifth positional argument must not silently select a level.
    with pytest.raises(TypeError):
        match(_BUILDING, _BUILDING, 0.9, 0, "building")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dead band: building-level scores come from the same penalty registry
# ---------------------------------------------------------------------------


def test_building_scores_stay_inside_the_registry_and_dead_band() -> None:
    reachable = {0.0}
    registry = matcher_module._PENALTIES
    soft = registry["one_sided"], registry["conflict"]
    for a, b in itertools.product(_SAMPLE, repeat=2):
        score = match(a, b, level="building").score
        reachable.add(score)
        assert not (0.85 < score < 0.9126), (a, b, score)
    assert soft  # registry access sanity
    assert 1.0 in reachable and 0.85 in reachable


_KIND_SAMPLE = tuple(
    base + tail
    for base in ("KR 1 # 4", "KR 1 # 4 - 80", "KR 1 A # 4 - 80", "KR 1 # 4 - 80 A")
    for tail in ("", " AP 1", " AP 2", " LT 5", " LT 6", " MZ 3 CA 4", " MZ 3", " ED X", " CS 3", " 8 C", " AP", " BLQ 2 LT 5")
)


def test_building_level_symmetry_and_dead_band_over_the_kind_sample() -> None:
    for a, b in itertools.product(_KIND_SAMPLE, repeat=2):
        forward = match(a, b, level="building")
        backward = match(b, a, level="building")
        assert forward.score == backward.score, (a, b)
        assert forward.is_match == backward.is_match, (a, b)
        assert not (0.85 < forward.score < 0.9126), (a, b, forward.score)
        assert forward.is_match == (forward.score >= 0.9)


# ---------------------------------------------------------------------------
# AddressIndex pass-through
# ---------------------------------------------------------------------------


def test_index_find_level_matches_brute_force() -> None:
    records = [(i, addr) for i, addr in enumerate(_SAMPLE)]
    index = AddressIndex.from_records(records)
    for query in ("KR 26 # 73 - 10 AP 999", "CL 5 # 10 - 20", "KR 1 # 4 - 80 LC 1", "nonsense"):
        for level in ("unit", "building"):
            found = index.find(query, level=level)
            expected = {}
            for i, addr in records:
                r = match(query, addr, level=level)
                if r.is_match:
                    expected[i] = r.score
            assert {c.record_id: c.score for c in found.candidates} == expected, (query, level)


def test_index_find_default_level_is_unit() -> None:
    index = AddressIndex.from_records([(1, "KR 26 # 73 - 10 BG 1")])
    assert index.find("KR 26 # 73 - 10 BG 2").candidates == []
    assert [c.record_id for c in index.find("KR 26 # 73 - 10 BG 2", level="building").candidates] == [1]


def test_index_find_invalid_level_raises() -> None:
    index = AddressIndex.from_records([(1, _BUILDING)])
    with pytest.raises(ValueError, match="level"):
        index.find(_BUILDING, level="floor")  # type: ignore[arg-type]
