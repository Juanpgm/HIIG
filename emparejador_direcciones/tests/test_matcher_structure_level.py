"""Tests for ``level="structure"``: same physical building = same structural
identity (tower/block/interior/stage/grouping/block-of-lots/sector/house/lot),
ignoring the UNIT kinds (apartment, local, office, floor, penthouse, storage,
garage, parking). Audit finding M2: ``level="building"`` did not group by
building; ``"structure"`` is the explicit new level, the other two are
unchanged."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from emparejador import AddressIndex, match, match_parsed, parse_canonical
from emparejador import matcher as matcher_module
from emparejador.matcher import (
    DEFAULT_LEVEL,
    LEVELS,
    STRUCTURE_DISTINGUISHING_KINDS,
    STRUCTURE_UNIT_KINDS,
    InvalidLevelError,
)

_BASE = "KR 1 # 4 - 80"


def _s(a: str, b: str):
    return match(a, b, level="structure")


def test_level_registry_keeps_existing_levels_and_adds_structure() -> None:
    assert LEVELS == ("unit", "building", "structure")
    assert DEFAULT_LEVEL == "unit"
    with pytest.raises(InvalidLevelError):
        match(_BASE, _BASE, level="edificio")
    with pytest.raises(InvalidLevelError):
        match(_BASE, _BASE, level=None)  # type: ignore[arg-type]


def test_kind_sets_are_disjoint_and_cover_the_documented_kinds() -> None:
    assert STRUCTURE_UNIT_KINDS == frozenset({"AP", "LC", "OF", "PISO", "PH", "BG", "GA", "PQ"})
    assert STRUCTURE_DISTINGUISHING_KINDS == frozenset(
        {"TO", "BLQ", "INT", "ET", "AGR", "MZ", "SEC", "CA", "LT"}
    )
    assert not STRUCTURE_UNIT_KINDS & STRUCTURE_DISTINGUISHING_KINDS


@pytest.mark.parametrize("kind", sorted(STRUCTURE_UNIT_KINDS))
def test_different_units_of_every_unit_kind_are_the_same_building(kind: str) -> None:
    a = f"{_BASE} {kind} 301"
    b = f"{_BASE} {kind} 402"
    assert _s(a, b).score == 1.0
    # The same pair is a veto at unit level for the identifying unit kinds
    # (AP, LC, OF, GA, PQ) and unchanged for the others: no level changes.
    assert match(a, b).score == match(a, b, level="unit").score


def test_apartment_301_vs_402_same_building_but_not_same_unit() -> None:
    a, b = f"{_BASE} AP 301", f"{_BASE} AP 402"
    assert _s(a, b).is_match is True and _s(a, b).score == 1.0
    assert match(a, b, level="unit").score == 0.0
    assert match(a, b, level="building").score == 0.0  # AP is not ignored there (M2)


def test_unit_on_one_side_only_costs_nothing_at_structure_level() -> None:
    assert _s(f"{_BASE} AP 301", _BASE).score == 1.0
    assert _s(_BASE, f"{_BASE} LC 5 PQ 2").score == 1.0


@pytest.mark.parametrize("kind", sorted(STRUCTURE_DISTINGUISHING_KINDS))
def test_distinguishing_kinds_with_different_values_are_different(kind: str) -> None:
    assert _s(f"{_BASE} {kind} A", f"{_BASE} {kind} B").is_match is False
    assert _s(f"{_BASE} {kind} 1", f"{_BASE} {kind} 2").score == 0.0


@pytest.mark.parametrize("kind", sorted(STRUCTURE_DISTINGUISHING_KINDS))
def test_distinguishing_kind_on_one_side_only_is_not_a_match(kind: str) -> None:
    result = _s(f"{_BASE} {kind} A", _BASE)
    assert result.is_match is False
    assert result.score == 0.0
    assert "complemento" in result.reason
    # symmetric
    assert _s(_BASE, f"{_BASE} {kind} A").score == 0.0
    # unit level keeps the cheap one-sided price (documented difference)
    assert match(f"{_BASE} {kind} A", _BASE).score == pytest.approx(0.98)


def test_tower_a_vs_tower_b_different_and_same_tower_with_other_units_same() -> None:
    assert _s(f"{_BASE} TO A", f"{_BASE} TO B").is_match is False
    assert _s(f"{_BASE} TO A AP 301", f"{_BASE} TO A AP 402").score == 1.0
    assert _s(f"{_BASE} TO A AP 301", f"{_BASE} TO B AP 301").score == 0.0
    assert _s(f"{_BASE} TO A AP 301", f"{_BASE} AP 301").score == 0.0  # one-sided tower
    assert _s(f"{_BASE} TO A2 AP 301", f"{_BASE} TO A2 AP 9").score == 1.0


def test_tower_and_block_both_must_agree() -> None:
    assert _s(f"{_BASE} TO A BLQ 1 AP 2", f"{_BASE} TO A BLQ 1 AP 3").score == 1.0
    assert _s(f"{_BASE} TO A BLQ 1 AP 2", f"{_BASE} TO A BLQ 2 AP 2").score == 0.0
    assert _s(f"{_BASE} TO A BLQ 1", f"{_BASE} TO A").score == 0.0


def test_other_identifying_kinds_keep_unit_level_cost() -> None:
    # ED / CONJ / UR / BR / CGTO / VDA are not part of the index vocabulary:
    # they are compared exactly as at unit level.
    for kind in ("ED", "CONJ", "UR", "BR", "CGTO", "VDA"):
        a, b = f"{_BASE} {kind} X1", f"{_BASE} {kind} X2"
        assert _s(a, b).score == match(a, b).score == 0.0
        assert _s(f"{_BASE} {kind} X1", _BASE).score == match(f"{_BASE} {kind} X1", _BASE).score


def test_opaque_chunks_are_always_compared() -> None:
    assert _s(f"{_BASE} 8 C", f"{_BASE} 9 C").score == 0.0
    assert _s(f"{_BASE} 8 C", _BASE).score == pytest.approx(0.85)
    assert _s(f"{_BASE} AP", _BASE).score == pytest.approx(0.85)  # kind without value
    assert _s(f"{_BASE} AP NORTE", _BASE).score == pytest.approx(0.85)


def test_without_a_plate_on_both_sides_nothing_is_ignored() -> None:
    assert _s("KR 1 # 4 - AP 1", "KR 1 # 4 - AP 2").score == match("KR 1 # 4 - AP 1", "KR 1 # 4 - AP 2").score == 0.0
    assert _s("KR 1 # 4 - AP 1", "KR 1 # 4 - 80 AP 1").score == 0.0  # plate on one side only


def test_street_fields_still_decide_at_structure_level() -> None:
    assert _s(f"{_BASE} AP 1", "KR 1 # 4 - 81 AP 1").is_match is False
    assert _s(f"{_BASE} AP 1", "KR 2 # 4 - 80 AP 1").score == 0.0
    assert _s(f"{_BASE} TO A", "KR 1 A # 4 - 80 TO A").score == pytest.approx(0.85)


def test_self_match_is_one_and_symmetric_for_a_grid_of_addresses() -> None:
    tails = ("", " AP 1", " TO A", " TO A AP 1", " TO B AP 1", " LC 2", " BLQ 3 PISO 4", " 8 C", " ED X")
    addresses = [_BASE + tail for tail in tails]
    for a in addresses:
        assert _s(a, a).score == 1.0
    for a, b in itertools.product(addresses, repeat=2):
        assert _s(a, b).score == _s(b, a).score
        assert _s(a, b).is_match == _s(b, a).is_match


def test_structure_is_at_least_as_permissive_as_unit_for_unit_kinds_only() -> None:
    # Any pair differing only by unit kinds goes from <1 to 1.0; a pair that
    # differs by a distinguishing kind is never promoted.
    pairs = [
        (f"{_BASE} AP 1", f"{_BASE} AP 2", True),
        (f"{_BASE} TO A AP 1", f"{_BASE} TO B AP 1", False),
    ]
    for a, b, promoted in pairs:
        assert (_s(a, b).score > match(a, b).score) is promoted


def test_unit_and_building_levels_are_untouched() -> None:
    # Golden values for the two older levels (computed with the matcher
    # before the structure level existed).
    assert match(f"{_BASE} AP 1", f"{_BASE} AP 2").score == 0.0
    assert match(f"{_BASE} BG 1", f"{_BASE} BG 2", level="building").score == 1.0
    assert match(f"{_BASE} AP 1", f"{_BASE} AP 2", level="building").score == 0.0
    assert match(f"{_BASE} TO A", _BASE, level="building").score == pytest.approx(0.98)


def test_parsed_sides_in_the_result_are_the_originals() -> None:
    a, b = f"{_BASE} AP 301", f"{_BASE} AP 402"
    result = _s(a, b)
    assert result.parsed_a == parse_canonical(a)
    assert result.parsed_b == parse_canonical(b)
    assert match_parsed(parse_canonical(a), parse_canonical(b), level="structure").score == 1.0


def test_parse_failures_stay_vetoes_at_structure_level() -> None:
    for bad in (None, "", "garbage", "KR 1 # 4 - 80 # 5"):
        result = _s(bad, _BASE)
        assert result.score == 0.0 and result.is_match is False
        assert _s(bad, bad).score == 0.0


def test_warnings_are_computed_from_the_original_complements() -> None:
    assert "complemento_no_estructurado" in _s(f"{_BASE} AP", f"{_BASE} AP 2").warnings


def test_plate_tolerance_and_threshold_still_apply() -> None:
    a, b = f"{_BASE} AP 1", "KR 1 # 4 - 81 AP 2"
    assert _s(a, b).is_match is False
    result = match(a, b, plate_tolerance=1, level="structure")
    assert result.score == pytest.approx(0.96)
    assert match(a, b, threshold=0.99, plate_tolerance=1, level="structure").is_match is False


def test_index_find_at_structure_level_groups_units_of_one_building() -> None:
    index = AddressIndex()
    index.add("a", f"{_BASE} TO A AP 301")
    index.add("b", f"{_BASE} TO A AP 402")
    index.add("c", f"{_BASE} TO B AP 301")
    index.add("d", f"{_BASE} AP 301")
    found = index.find(f"{_BASE} TO A AP 999", level="structure")
    assert {c.record_id for c in found.candidates} == {"a", "b"}
    unit = index.find(f"{_BASE} TO A AP 999")
    assert list(unit.candidates) == []


def test_index_equals_brute_force_at_structure_level() -> None:
    records = {
        i: addr
        for i, addr in enumerate(
            f"{base}{tail}"
            for base in (_BASE, "KR 1 # 4 - 81", "KR 1 A # 4 - 80", "CL 5 # 10 - 20")
            for tail in ("", " AP 1", " AP 2", " TO A", " TO A AP 1", " TO B", " LC 3", " 8 C", " ED X", " SEC 2")
        )
    }
    index = AddressIndex()
    for i, addr in records.items():
        index.add(i, addr)
    for query in records.values():
        got = {c.record_id: c.score for c in index.find(query, level="structure").candidates}
        expected = {
            i: r.score
            for i, other in records.items()
            if (r := match(query, other, level="structure")).is_match
        }
        assert got == expected


def test_drop_helper_never_removes_distinguishing_or_opaque_chunks() -> None:
    parsed = parse_canonical(f"{_BASE} 8 C AP 1 TO A LC 5 ED X")
    view = matcher_module.structure_view(parsed)
    kinds = {kind for kind, _ in view.complement}
    assert kinds == {"TO", "?", "ED"}
    no_plate = parse_canonical("KR 1 # 4 - AP 1")
    assert matcher_module.structure_view(no_plate) == no_plate
    assert dataclasses.is_dataclass(view)


def test_one_sided_distinguishing_kind_is_a_veto_even_without_plates() -> None:
    assert _s("KR 1 # 4 - TO A", "KR 1 # 4 - ").score == 0.0
    assert match("KR 1 # 4 - TO A", "KR 1 # 4 - ").score == pytest.approx(0.98)


def test_level_names_are_part_of_the_public_package_surface() -> None:
    import emparejador

    for name in (
        "LEVELS",
        "DEFAULT_LEVEL",
        "InvalidLevelError",
        "building_view",
        "structure_view",
        "STRUCTURE_UNIT_KINDS",
        "STRUCTURE_DISTINGUISHING_KINDS",
        "BUILDING_IGNORED_KINDS",
    ):
        assert name in emparejador.__all__, name
        assert hasattr(emparejador, name), name
    assert sorted(emparejador.__all__) == sorted(set(emparejador.__all__))


def test_readme_parse_failure_reporting_example_separates_the_outcomes() -> None:
    from collections import Counter

    from emparejador import AddressIndex

    base = [(1, "KR 1 # 4 - 80"), (2, "garbage"), (3, "CL 5 # 10 - 20 TO A2")]
    index = AddressIndex()
    for record_id, address in base:
        index.add(record_id, address)
    assert index.rejected == 1
    assert index.stats()["rejected"] == 1 and index.stats()["records"] == 2

    queries = ["KR 1 # 4 - 80", "garbage", "", None, "AV 9 # 9 - 99", "CL 5 # 10 - 20 TO A2"]
    outcome: Counter[str] = Counter()
    notes: Counter[str] = Counter()
    for query in queries:
        result = index.find(query)
        if not result.parse_ok:
            outcome["consulta_ilegible"] += 1
            notes[result.reason] += 1
        elif result.best is None:
            outcome["sin_candidato"] += 1
        elif result.ambiguous:
            outcome["ambiguo"] += 1
        else:
            outcome["enlazado"] += 1
    assert outcome == {"consulta_ilegible": 3, "sin_candidato": 1, "enlazado": 2}
    assert all(reason.startswith("consulta no canónica") for reason in notes)
