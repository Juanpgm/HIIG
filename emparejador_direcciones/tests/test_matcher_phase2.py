"""Phase 2 matcher tests: opaque complements, plate-less addresses,
quadrant abbreviations, legacy via type, ``match_parsed``."""

from __future__ import annotations

import json

import pytest

from emparejador import match, parse_canonical
from emparejador.matcher import InvalidThresholdError, InvalidToleranceError, match_parsed

OPAQUE = "K 49 E # 49 - 50 8 C"
OPAQUE_OTHER = "K 49 E # 49 - 50 9 C"
BARE = "K 49 E # 49 - 50"


# ---------------------------------------------------------------------------
# D4: opaque complement scoring
# ---------------------------------------------------------------------------


def test_opaque_equal_on_both_sides_is_exact_with_warning() -> None:
    result = match(OPAQUE, "KR 49 E # 49 - 050 08C")
    assert result.score == 1.0
    assert result.is_match is True
    assert "complemento_no_estructurado" in result.warnings
    complement = next(f for f in result.fields if f.field == "complement")
    assert complement.status == "igual"


def test_opaque_both_different_is_veto() -> None:
    result = match(OPAQUE, OPAQUE_OTHER)
    assert result.score == 0.0
    assert result.is_match is False
    assert any(f.field == "complement" and f.veto for f in result.fields)
    assert "complemento_no_estructurado" in result.warnings


@pytest.mark.parametrize("a,b", [(OPAQUE, BARE), (BARE, OPAQUE)])
def test_opaque_one_sided_costs_015_and_does_not_link(a: str, b: str) -> None:
    result = match(a, b)
    assert result.score == pytest.approx(0.85)
    assert result.is_match is False
    assert "complemento_no_estructurado" in result.warnings
    complement = next(f for f in result.fields if f.field == "complement")
    assert complement.status == "solo_en_uno" and complement.penalty == 0.15


def test_opaque_one_sided_boundary_with_threshold() -> None:
    assert match(OPAQUE, BARE, threshold=0.85).is_match is True
    assert match(OPAQUE, BARE, threshold=0.8501).is_match is False


def test_known_one_sided_unchanged_and_no_warning() -> None:
    result = match("KR 1 # 4 - 80 AP 5", "KR 1 # 4 - 80")
    assert result.score == pytest.approx(0.98)
    assert "complemento_no_estructurado" not in result.warnings


def test_any_opaque_chunk_on_the_side_triggers_the_heavy_penalty() -> None:
    result = match("K 49 E # 49 - 50 8 C LT 5", BARE)
    assert result.score == pytest.approx(0.85)


def test_known_only_side_stays_cheap_even_when_other_bare() -> None:
    assert match("K 49 E # 49 - 50 LT 5 AP 3", BARE).score == pytest.approx(0.98)


def test_opaque_vs_known_both_present_vetoes() -> None:
    assert match(OPAQUE, "K 49 E # 49 - 50 LT 5").score == 0.0
    assert match("K 49 E # 49 - 50 8 C LT 5", "K 49 E # 49 - 50 LT 5").score == 0.0


def test_opaque_with_named_kind_behaves_the_same() -> None:
    a, b = "CL 9 # 51 - 48 GASS 5", "CL 9 # 51 - 48"
    assert match(a, a).score == 1.0
    assert match(a, b).score == pytest.approx(0.85)
    assert match(a, "CL 9 # 51 - 48 GASS 6").score == 0.0


def test_opaque_symmetry() -> None:
    pairs = [
        (OPAQUE, OPAQUE_OTHER),
        (OPAQUE, BARE),
        (OPAQUE, OPAQUE),
        (OPAQUE, "K 49 E # 49 - 50 LT 5"),
        ("KR 69 # 33 - LT 20", "KR 69 # 33"),
        ("K 8 # 22 - 48 /50 /52", "K 8 # 22 - 48 50 52"),
    ]
    for a, b in pairs:
        ab, ba = match(a, b), match(b, a)
        assert ab.score == ba.score and ab.is_match == ba.is_match
        assert set(ab.warnings) == set(ba.warnings)


def test_opaque_reason_and_json_are_human_readable() -> None:
    result = match(OPAQUE, OPAQUE_OTHER)
    assert "8C" in result.reason and "9C" in result.reason
    assert "?" not in result.reason
    json.dumps(result.to_dict())


def test_opaque_reason_for_one_sided() -> None:
    result = match(OPAQUE, BARE)
    assert "complement solo en un lado" in result.reason


def test_opaque_warning_absent_when_no_opaque_chunk() -> None:
    assert match("KR 1 # 4 - 80", "KR 1 # 4 - 80").warnings == ()


# ---------------------------------------------------------------------------
# D1/D2: plate-less addresses
# ---------------------------------------------------------------------------


def test_trailing_dash_equals_no_dash() -> None:
    result = match("C 1 # 74 -", "C 1 # 74")
    assert result.score == 1.0
    assert "sin_placa" in result.warnings


def test_plate_none_vs_plate_present_vetoes() -> None:
    assert match("C 1 # 74 -", "C 1 # 74 - 20").score == 0.0
    assert match("C 1 # 74 - 20", "C 1 # 74 -").score == 0.0


def test_plate_slot_complement_equal_and_different() -> None:
    assert match("KR 69 # 33 - LT 20", "KR 69 # 33 - LT 20").score == 1.0
    assert match("KR 69 # 33 - LT 20", "KR 69 # 33 - LT 21").score == 0.0
    assert match("KR 69 # 33 - LT 20", "KR 69 # 33 - LOTE 020").score == 1.0


def test_plate_slot_complement_vs_bare_is_one_sided_known() -> None:
    assert match("KR 69 # 33 - LT 20", "KR 69 # 33").score == pytest.approx(0.98)


def test_plate_slot_complement_vs_plated_address_vetoes() -> None:
    assert match("KR 69 # 33 - LT 20", "KR 69 # 33 - 05 LT 20").score == 0.0


# ---------------------------------------------------------------------------
# D5/D6: quadrant abbreviations and legacy type A
# ---------------------------------------------------------------------------


def test_abbreviated_quadrant_equals_full_word() -> None:
    assert match("C 70 B N # 4 C - 104", "CL 70 B NORTE # 4 C - 104").score == 1.0
    assert match("C 1 A BIS O # 81 - 19", "CL 1 A BIS OESTE # 81 - 19").score == 1.0
    assert match("KR 5 B W # 4 - 80", "KR 5 B OESTE # 4 - 80").score == 1.0


def test_abbreviated_quadrant_conflict_scores_070() -> None:
    assert match("C 70 B N # 4 - 10", "CL 70 B S # 4 - 10").score == pytest.approx(0.70)


def test_direct_letter_is_not_a_quadrant_for_scoring() -> None:
    assert match("KR 41 E # 4 - 80", "KR 41 ESTE # 4 - 80").score == pytest.approx(0.7225)


def test_legacy_a_equals_av() -> None:
    assert match("A 9 # 5 - 3", "AV 9 # 5 - 3").score == 1.0
    assert match("A 9 # 5 - 3", "KR 9 # 5 - 3").score == 0.0


# ---------------------------------------------------------------------------
# match_parsed
# ---------------------------------------------------------------------------


def test_match_parsed_equals_match() -> None:
    a, b = "KR 1 # 4 - 80", "KR 1 A # 4 - 80"
    direct = match(a, b, threshold=0.85)
    via_parsed = match_parsed(parse_canonical(a), parse_canonical(b), threshold=0.85)
    assert via_parsed.score == direct.score
    assert via_parsed.is_match == direct.is_match
    assert via_parsed.reason == direct.reason
    assert via_parsed.warnings == direct.warnings


def test_match_parsed_handles_unparseable_and_validates_settings() -> None:
    bad = parse_canonical("junk")
    good = parse_canonical("KR 1 # 4 - 80")
    assert match_parsed(bad, good).score == 0.0
    with pytest.raises(InvalidThresholdError):
        match_parsed(good, good, threshold=2.0)
    with pytest.raises(InvalidToleranceError):
        match_parsed(good, good, plate_tolerance=-1)
