"""Position-aware guard: ``base_mismatches(rule, cadastral, mode="position")``.

Returns ``(hard, caps)``: ``hard`` are field names that reject the pair, ``caps`` maps a field to the
best position tier the pair may still carry (``lote`` / ``tramo`` / ``manzana``). All addresses here
are synthetic.
"""

from __future__ import annotations

import pytest

from conftest import make_stub  # noqa: F401  (also puts src/ on sys.path)
from cali_address.inference import PLATE_TRAMO_WINDOW, base_mismatches
from cali_address.parser import parse_address


def g(rule: str | None, cad: str | None, **kw):
    r = None if rule is None else parse_address(rule)
    c = None if cad is None else parse_address(cad)
    return base_mismatches(r, c, mode="position", **kw)


CLEAN = ([], {})


# ---------------------------------------------------------------------------
# identity and defaults
# ---------------------------------------------------------------------------
def test_default_mode_keeps_returning_a_plain_list():
    r, c = parse_address("CL 987 # 654 - 20"), parse_address("CL 987 # 654 - 21")
    assert base_mismatches(r, c) == ["plate"]
    assert base_mismatches(r, c, mode="legacy") == ["plate"]


def test_unknown_mode_is_rejected():
    r = parse_address("CL 987 # 654 - 20")
    with pytest.raises(ValueError):
        base_mismatches(r, r, mode="nope")


@pytest.mark.parametrize("text", [
    "CL 987 # 654 - 20", "KR 987 NORTE # 654 NORTE - 03", "KR 926G6 # 972S1 - 08CA", "CL 922NORTE # 653 - 5",
    "CL 915BNORTE # 653 - 5", "CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 66C", "CL 987 # 654 - 20 TO 2 AP 3",
    "CL 987 # 654 - 20 BIS", "CL 91A9 # 653 - 250", "CL 987 # 654",
])
def test_an_address_never_conflicts_with_itself(text):
    assert g(text, text) == CLEAN
    assert g(text, text) == g(text, text)  # deterministic


# ---------------------------------------------------------------------------
# unparsed / malformed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule, cad", [
    (None, "CL 987 # 654 - 20"), ("CL 987 # 654 - 20", None), (None, None),
    ("", "CL 987 # 654 - 20"), ("hola mundo", "CL 987 # 654 - 20"), ("CL 987 # 654 - 20", "sin direccion"),
])
def test_unparsed_is_a_hard_reject(rule, cad):
    assert g(rule, cad) == (["unparsed"], {})


def test_non_parse_objects_do_not_raise():
    assert base_mismatches(object(), object(), mode="position") == (["unparsed"], {})


def test_very_long_and_unicode_inputs_do_not_raise():
    long_tail = "CL 987 # 654 - 20 " + "ÑÁÉ" * 3000
    hard, caps = g(long_tail, "CL 987 # 654 - 20")
    assert isinstance(hard, list) and isinstance(caps, dict)
    hard, caps = g("CL 987 # 654 - 20 ‮\u0000", "CL 987 # 654 - 20")
    assert isinstance(hard, list)


# ---------------------------------------------------------------------------
# quadrant position
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule, cad", [
    ("CL 987 N # 654 - 20", "CL 987 NORTE # 654 - 20"),     # N == NORTE: same segment, same prefix
    ("CL 987 NTE # 654 - 20", "CL 987 NORTE # 654 - 20"),
    ("CL 987 O # 654 - 20", "CL 987 OESTE # 654 - 20"),
    ("CL 987 # 654 N - 20", "CL 987 # 654 NORTE - 20"),
    ("CL 987 # 654 - 20 N", "CL 987 # 654 - 20 NORTE"),
    ("CL 922NORTE # 653 - 5", "CL 922 NORTE # 653 - 5"),
])
def test_cardinal_letter_equals_spelled_quadrant_on_the_same_segment(rule, cad):
    assert g(rule, cad) == CLEAN
    assert g(cad, rule) == CLEAN


@pytest.mark.parametrize("rule, cad", [
    ("CL 987 N # 654 - 20", "CL 987 # 654 NORTE - 20"),      # via -> cross
    ("CL 987 # 654 N - 20", "CL 987 NORTE # 654 - 20"),
    ("AV 876 # 654 N - 06", "AV 876 NORTE # 654 - 06"),      # user-confirmed example
    ("CL 987 # 654 - 20 NORTE", "CL 987 NORTE # 654 - 20"),  # trailing -> via
    ("CL 987 # 654 - 20 NORTE", "CL 987 # 654 NORTE - 20"),  # trailing -> cross
    ("CL 987 NORTE # 654 - 20", "CL 987 # 654 - 20 NORTE"),
])
def test_quadrant_moved_to_another_position_is_a_hard_reject(rule, cad):
    hard, _caps = g(rule, cad)
    assert "quadrant_moved" in hard
    hard_rev, _ = g(cad, rule)
    assert "quadrant_moved" in hard_rev


@pytest.mark.parametrize("rule, cad, field", [
    ("CL 987 NORTE # 654 - 20", "CL 987 OESTE # 654 - 20", "via_quadrant"),
    ("CL 987 # 654 NORTE - 20", "CL 987 # 654 OESTE - 20", "cross_quadrant"),
    ("CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20 OESTE", "plate_quadrant"),
])
def test_different_quadrants_in_the_same_position_are_hard(rule, cad, field):
    hard, _ = g(rule, cad)
    assert field in hard


@pytest.mark.parametrize("rule, cad, field", [
    ("CL 987 NORTE # 654 - 20", "CL 987 # 654 - 20", "via_quadrant"),
    ("CL 987 # 654 - 20", "CL 987 NORTE # 654 - 20", "via_quadrant"),
    ("CL 987 # 654 NORTE - 20", "CL 987 # 654 - 20", "cross_quadrant"),
    ("CL 987 # 654 - 20", "CL 987 # 654 OESTE - 20", "cross_quadrant"),
    ("CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20", "plate_quadrant"),
    ("CL 987 # 654 - 20", "CL 987 # 654 - 20 NORTE", "plate_quadrant"),
])
def test_one_sided_quadrant_caps_at_manzana(rule, cad, field):
    assert g(rule, cad) == ([], {field: "manzana"})


def test_cross_quadrant_only_in_the_cadastre_caps_manzana_and_is_not_rejected():
    # 'KR 987NORTE 654 3' vs the cadastral form: the false rejection of the research guard
    assert g("KR 987NORTE 654 3", "KR 987 NORTE # 654 NORTE - 03") == ([], {"cross_quadrant": "manzana"})


def test_same_quadrant_word_different_prefix_is_hard():
    hard, _ = g("CL 987 A N # 654 - 20", "CL 987 B NORTE # 654 - 20")
    assert "via_letters" in hard
    assert g("CL 987 A NORTE # 654 - 20", "CL 987 NORTE # 654 - 20") == ([], {"via_letters": "manzana"})


# ---------------------------------------------------------------------------
# plate number boundaries
# ---------------------------------------------------------------------------
def test_window_constant():
    assert PLATE_TRAMO_WINDOW == 2


@pytest.mark.parametrize("plate, expected", [
    (20, CLEAN), (21, ([], {"plate": "tramo"})), (19, ([], {"plate": "tramo"})),
    (22, ([], {"plate": "tramo"})), (18, ([], {"plate": "tramo"})),
    (23, (["plate"], {})), (17, (["plate"], {})), (200, (["plate"], {})),
])
def test_plate_delta_boundaries(plate, expected):
    assert g(f"CL 987 # 654 - {plate}", "CL 987 # 654 - 20") == expected


def test_plate_leading_zeros_are_equal():
    assert g("CL 987 # 654 - 03", "CL 987 # 654 - 3") == CLEAN
    assert g("CL 987 # 654 - 3", "CL 987 # 654 - 03") == CLEAN


def test_plate_zero_vs_small_plates():
    assert g("CL 987 # 654 - 0", "CL 987 # 654 - 1") == ([], {"plate": "tramo"})
    assert g("CL 987 # 654 - 0", "CL 987 # 654 - 3") == (["plate"], {})


def test_plate_missing_on_one_side():
    assert g("CL 987 # 654", "CL 987 # 654 - 20") == CLEAN  # the query gave no door: nothing to contradict
    # the query names a door and the cadastral parcel has none: rejected (was skipped; the research guard agrees)
    assert g("CL 987 # 654 - 20", "CL 987 # 654") == (["plate"], {})


def test_plate_tolerance_swallows_the_delta():
    assert g("CL 987 # 654 - 22", "CL 987 # 654 - 20", plate_tolerance=2) == CLEAN
    assert g("CL 987 # 654 - 22", "CL 987 # 654 - 20", plate_tolerance=1) == ([], {"plate": "tramo"})


def test_odd_even_sides_are_both_inside_the_window():
    # 655 vs 12 are opposite sides of the street; the guard only sees the number
    assert g("CL 987 # 654 - 11", "CL 987 # 654 - 12") == ([], {"plate": "tramo"})


# ---------------------------------------------------------------------------
# plate letters and BIS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule, cad, field", [
    ("CL 987 # 654 - 66 C", "CL 987 # 654 - 66", "plate_letters"),
    ("CL 987 # 654 - 66", "CL 987 # 654 - 66 C", "plate_letters"),
    ("CL 987 # 654 - 66 C", "CL 987 # 654 - 66 D", "plate_letters"),
    ("CL 987 # 654 - 66C", "CL 987 # 654 - 66", "plate_letters"),
    ("CL 987 # 654 - 66 BIS", "CL 987 # 654 - 66", "plate_bis"),
])
def test_genuine_plate_letters_or_bis_cap_at_tramo(rule, cad, field):
    assert g(rule, cad) == ([], {field: "tramo"})


@pytest.mark.parametrize("cad", [
    "CL 987 # 654 - 69 1 C", "CL 987 # 654 - 79 7 G", "CL 987 # 654 - 69 P 1", "CL 987 # 654 - 69 12 C",
])
def test_cadastral_tail_tokens_are_not_plate_letters(cad):
    plain = cad.split(" - ")[0] + " - " + cad.split(" - ")[1].split()[0]
    assert g(plain, cad) == CLEAN
    # the fix lives on the cadastre side only: the same text typed by a user keeps the lenient
    # reading, so its trailing letter is still a (possible) plate letter
    assert g(cad, cad)[0] == []


def test_cadastral_tail_fix_also_applies_to_the_legacy_mode():
    r, c = parse_address("CL 987 # 654 - 79"), parse_address("CL 987 # 654 - 79 7 G")
    assert "plate_letters" not in base_mismatches(r, c)


def test_glued_letters_in_the_cadastre_still_count():
    assert g("CL 987 # 654 - 66", "CL 987 # 654 - 66C") == ([], {"plate_letters": "tramo"})
    assert g("CL 987 # 654 - 08", "CL 987 # 654 - 08CA") == ([], {"plate_letters": "tramo"})  # glued 08CA: letters, not a unit


# ---------------------------------------------------------------------------
# via / cross identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule, cad, field", [
    ("CL 986 # 654 - 20", "CL 987 # 654 - 20", "via_number"),
    ("CL 987 # 655 - 20", "CL 987 # 654 - 20", "cross_number"),
    ("KR 985 # 654 - 20", "CL 987 # 654 - 20", "via_type"),
    ("CL 987 # KR 654 - 20", "CL 987 # CL 654 - 20", "cross_type"),
    ("KR 926G6 # 972S1 - 8", "KR 926G6 # 972S2 - 8", "cross_suffix"),
    ("KR 926G6 # 972S1 - 8", "KR 926G7 # 972S1 - 8", "via_suffix"),
    ("CL 987 BIS # 654 - 20", "CL 987 # 654 - 20", "via_bis"),
])
def test_identity_fields_are_hard(rule, cad, field):
    hard, _ = g(rule, cad)
    assert field in hard


def test_cross_type_blank_on_one_side_is_skipped():
    assert g("KR 985 # CL 654 - 20", "KR 985 # 654 - 20") == CLEAN
    assert g("KR 985 # 654 - 20", "KR 985 # CL 654 - 20") == CLEAN


def test_cross_number_missing_on_one_side_is_not_hard():
    # the parser reads 'CL 987 - 20' as via 5 / cross 20: it is a different cross, never a crash
    hard, caps = g("CL 987 - 20", "CL 987 # 654 - 20")
    assert hard == ["cross_number"]


@pytest.mark.parametrize("rule, cad, field", [
    ("KR 765 E # 432 - 20", "KR 765 # 432 - 20", "via_letters"),
    ("KR 765 # 432 - 20", "KR 765 E # 432 - 20", "via_letters"),
    ("KR 765 # 432 E - 20", "KR 765 # 432 - 20", "cross_letters"),
    ("KR 765 # 432 - 20", "KR 765 # 432 E - 20", "cross_letters"),
])
def test_one_sided_street_letters_cap_at_manzana(rule, cad, field):
    assert g(rule, cad) == ([], {field: "manzana"})


@pytest.mark.parametrize("rule, cad, field", [
    ("KR 765 E # 432 - 20", "KR 765 F # 432 - 20", "via_letters"),
    ("KR 765 # 432 E - 20", "KR 765 # 432 F - 20", "cross_letters"),
])
def test_street_letters_differing_on_both_sides_are_hard(rule, cad, field):
    hard, caps = g(rule, cad)
    assert field in hard and field not in caps


def test_soft_rules_downgrade_letter_and_bis_rejections_to_a_manzana_cap():
    hard, caps = g("KR 765 E # 432 - 20", "KR 765 F # 432 - 20", soft_rules=frozenset({"letra_via"}))
    assert hard == [] and caps == {"via_letters": "manzana"}
    hard, caps = g("KR 765 # 432 E - 20", "KR 765 # 432 F - 20", soft_rules=frozenset({"letra_cruce"}))
    assert hard == [] and caps == {"cross_letters": "manzana"}
    hard, caps = g("CL 987 BIS # 654 - 20", "CL 987 # 654 - 20", soft_rules=frozenset({"bis_via"}))
    assert hard == [] and caps == {"via_bis": "manzana"}
    # a rule that is not enabled does not downgrade anything else
    hard, _ = g("KR 765 E # 432 - 20", "KR 765 F # 432 - 20", soft_rules=frozenset({"bis_via"}))
    assert hard == ["via_letters"]


# ---------------------------------------------------------------------------
# site units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["TO", "BLQ", "CA", "AGR"])
def test_unit_differing_on_both_sides_caps_at_lote(kind):
    assert g(f"CL 987 # 654 - 20 {kind} 1", f"CL 987 # 654 - 20 {kind} 2") == ([], {f"unit_{kind}": "lote"})


@pytest.mark.parametrize("kind", ["TO", "BLQ", "CA", "AGR"])
def test_unit_only_in_the_cadastre_does_not_cap(kind):
    # measured on the dev set: when the input names no unit, the cadastral predial of such rows was almost always right
    # on gold rows, so withholding it would throw away correct predios
    assert g("CL 987 # 654 - 20", f"CL 987 # 654 - 20 {kind} 2") == CLEAN


@pytest.mark.parametrize("kind", ["TO", "BLQ", "CA", "AGR"])
def test_unit_only_in_the_input_does_not_cap(kind):
    assert g(f"CL 987 # 654 - 20 {kind} 2", "CL 987 # 654 - 20") == CLEAN


def test_equal_units_and_building_units_do_not_cap():
    assert g("CL 987 # 654 - 20 TO 2 AP 301", "CL 987 # 654 - 20 TO 2 AP 101") == CLEAN
    assert g("CL 987 # 654 - 20 AP 301", "CL 987 # 654 - 20 AP 101") == CLEAN


# ---------------------------------------------------------------------------
# combination, determinism and agreement with the research path
# ---------------------------------------------------------------------------
def test_several_findings_are_all_reported():
    hard, caps = g("CL 987 # 654 N - 22 TO 1", "CL 987 # 654 - 20 TO 2")
    assert hard == []
    assert caps == {"plate": "tramo", "cross_quadrant": "manzana", "unit_TO": "lote"}


def test_hard_and_caps_are_independent():
    hard, caps = g("CL 986 # 654 - 22", "CL 987 # 654 - 20")
    assert hard == ["via_number"] and caps == {"plate": "tramo"}


PAIRS = [
    ("CL 987 # 654 - 20", "CL 987 # 654 - 20"),
    ("CL 987 # 654 - 21", "CL 987 # 654 - 20"),
    ("CL 987 # 654 - 29", "CL 987 # 654 - 20"),
    ("CL 986 # 654 - 20", "CL 987 # 654 - 20"),
    ("CL 987 # 655 - 20", "CL 987 # 654 - 20"),
    ("CL 987 N # 654 - 20", "CL 987 NORTE # 654 - 20"),
    ("CL 987 N # 654 - 20", "CL 987 # 654 NORTE - 20"),
    ("AV 876 # 654 N - 06", "AV 876 NORTE # 654 - 06"),
    ("KR 987NORTE 654 3", "KR 987 NORTE # 654 NORTE - 03"),
    ("CL 987 # 654 - 66 C", "CL 987 # 654 - 66"),
    ("CL 987 # 654 - 66", "CL 987 # 654 - 79 7 G"),
    ("CL 987 # 654 - 20 TO 1", "CL 987 # 654 - 20 TO 2"),
    ("KR 765 E # 432 - 20", "KR 765 F # 432 - 20"),
    ("KR 765 # 432 - 20", "KR 765 E # 432 - 20"),
    ("CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20"),
    ("CL 987 NORTE # 654 - 20", "CL 987 OESTE # 654 - 20"),
    (None, "CL 987 # 654 - 20"),
]


@pytest.mark.parametrize("rule, cad", PAIRS)
def test_position_guard_and_research_guard_agree_on_hard_vs_soft(rule, cad):
    r = None if rule is None else parse_address(rule)
    c = None if cad is None else parse_address(cad)
    legacy = base_mismatches(r, c)
    hard, caps = base_mismatches(r, c, mode="position")
    if not legacy:
        # the research path sees the same base: the production guard may not complain
        assert hard == [] and caps == {}
    if hard:
        # whatever the production guard rejects, the research guard flags too
        assert legacy
    # the research path is the stricter one: it never accepts what production caps
    if caps:
        assert legacy


def test_caps_only_use_known_tiers():
    for rule, cad in PAIRS:
        r = None if rule is None else parse_address(rule)
        c = parse_address(cad)
        _hard, caps = base_mismatches(r, c, mode="position")
        assert set(caps.values()) <= {"lote", "tramo", "manzana"}
