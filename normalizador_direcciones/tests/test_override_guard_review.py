"""Review round for the cadastral override guard.

Information after the plate (letter, BIS, quadrant, site-identifying units) is part of the
base; quadrants are compared PER POSITION (via, cross, trailing); candidates below the
top one are tried; rejected rows carry no parcel data. Also pins parser behaviour touched in
the same round (``NO,`` position, comma between via and cross) and checks that the guard and
the matcher agree on quadrant semantics.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from test_override_guard import NAN, batch, make_normalizer, row_for

from cali_address.inference import base_mismatches
from cali_address.parser import canonical, parse_address


def _mm(a: str, b: str) -> list[str]:
    return base_mismatches(parse_address(a), parse_address(b))


# ---------------------------------------------------------------------------
# C1 / C2: information after the plate is part of the base
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, cadastral, expected, field", [
    ("CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20", "CL 987 # 654 - 20 NORTE", "plate_quadrant"),
    ("CL 987 # 654 - 20 SUR", "CL 987 # 654 - 20", "CL 987 # 654 - 20 SUR", "plate_quadrant"),
    ("CL 987 # 654 - 20 OESTE", "CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20 OESTE", "plate_quadrant"),
    ("KR 987 # 654 - 66C", "KR 987 # 654 - 66", "KR 987 # 654 - 66 C", "plate_letters"),
    ("KR 987 # 654 - 66 A", "KR 987 # 654 - 66 B", "KR 987 # 654 - 66 A", "plate_letters"),
    ("KR 987 # 654 - 66 BIS", "KR 987 # 654 - 66", "KR 987 # 654 - 66 BIS", "plate_bis"),
    ("KR 987 # 654 - 3 A4", "KR 987 # 654 - 3", "KR 987 # 654 - 03 A4", "plate_letters"),
    ("KR 987 # 654 - 66", "KR 987 # 654 - 66 C", "KR 987 # 654 - 66", "plate_letters"),
    ("CL 987 # 654 - 20 CA 1", "CL 987 # 654 - 20 CA 2", "CL 987 # 654 - 20 CA 1", "unit_CA"),
    ("CL 987 # 654 - 20 TORRE B", "CL 987 # 654 - 20 TO A", "CL 987 # 654 - 20 TO B", "unit_TO"),
    ("KR 987 # 654 B 2 - 29", "KR 987 # 654 B 3 - 29", "KR 987 # 654 B 2 - 29", "cross_suffix"),
    # a trailing quadrant is NOT the quadrant of the via, and the other way round
    ("CL 987 # 654 - 20 NORTE", "CL 987 NORTE # 654 - 20", "CL 987 # 654 - 20 NORTE", "plate_quadrant"),
    ("CL 987 NORTE # 654 - 20", "CL 987 # 654 - 20 NORTE", "CL 987 NORTE # 654 - 20", "via_quadrant"),
])
def test_rejected_override_keeps_the_rule_output_exactly(raw, cadastral, expected, field):
    row = row_for(raw, cadastral)
    assert row["matched"] is False
    assert row["canonical_address"] == expected
    assert field in row["parse_notes"]


@pytest.mark.parametrize("raw, cadastral, expected", [
    ("KR 987 # 654 - 66C", "KR 987 # 654 - 66 C", "KR 987 # 654 - 66 C"),
    ("KR 987 # 654 - 66 BIS", "KR 987 # 654 - 66 BIS", "KR 987 # 654 - 66 BIS"),
    ("KR 987 # 654 - 3 A4", "KR 987 # 654 - 03 A4", "KR 987 # 654 - 03 A4"),
    ("CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20 NORTE"),
    ("CL 987 NORTE # 654 - 20", "CL 987 NORTE # 654 - 20", "CL 987 NORTE # 654 - 20"),
    ("KR 987 # 654 - 66C AP 302", "KR 987 # 654 - 66 C AP 101", "KR 987 # 654 - 66 C AP 302"),
    ("CL 987 # 654 - 20 CA 1", "CL 987 # 654 - 20 CA 1", "CL 987 # 654 - 20 CA 1"),
])
def test_accepted_override_keeps_plate_tail_and_quadrant_exactly(raw, cadastral, expected):
    row = row_for(raw, cadastral)
    assert row["matched"] is True, row["parse_notes"]
    assert row["canonical_address"] == expected
    assert row["override_rejected"] is False


@pytest.mark.parametrize("a, b, field", [
    ("KR 987 # 654 - 66C", "KR 987 # 654 - 66", "plate_letters"),
    ("KR 987 # 654 - 66 A", "KR 987 # 654 - 66 B", "plate_letters"),
    ("KR 987 # 654 - 66 BIS", "KR 987 # 654 - 66", "plate_bis"),
    ("CL 987 # 654 - 20 NORTE", "CL 987 # 654 - 20", "plate_quadrant"),
    ("CL 987 # 654 - 20 CA 1", "CL 987 # 654 - 20 CA 2", "unit_CA"),
    ("KR 987 # 654 B 2 - 29", "KR 987 # 654 B 3 - 29", "cross_suffix"),
    ("KR 987 # 654 B 2 - 29", "KR 987 # 654 B - 29", "cross_suffix"),
    ("KR 987 # 654 B 2 A - 29", "KR 987 # 654 B 2 C - 29", "cross_suffix_letters"),
    ("KR 987 H 1 A # 654 - 29", "KR 987 H 1 C # 654 - 29", "via_suffix_letters"),
    ("KR 987 H 1 # 654 - 29", "KR 987 H 2 # 654 - 29", "via_suffix"),
])
def test_tail_and_suffix_fields_are_compared(a, b, field):
    assert field in _mm(a, b)
    assert field in _mm(b, a)


@pytest.mark.parametrize("a, b", [
    ("KR 987 # 654 - 66C", "KR 987 # 654 - 66 C"),            # glued vs spaced
    ("KR 987 # 654 - 66C AP 3", "KR 987 # 654 - 66 C AP 9"),    # AP is not a site-identifying kind
    ("CL 987 # 654 - 20 CA 1", "CL 987 # 654 - 20"),            # unit on one side only
    ("CL 987 # 654 - 20 CASA 1", "CL 987 # 654 - 20 CA 1"),     # alias spelling
    ("CL 987 # 654 - 20 N", "CL 987 # 654 - 20 NORTE"),         # trailing letter == trailing quadrant
])
def test_equivalent_tails(a, b):
    assert _mm(a, b) == []


# ---------------------------------------------------------------------------
# Quadrants per position (via / cross / trailing), never as a set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a, b, fields", [
    ("CL 987 NORTE # 654 SUR - 21", "CL 987 SUR # 654 NORTE - 21", {"via_quadrant", "cross_quadrant"}),
    ("CL 987 N # 654 N - 21", "CL 987 NORTE # 654 - 21", {"cross_quadrant"}),
    ("CL 987 N # 654 - 21", "CL 987 # 654 N - 21", {"via_quadrant", "cross_quadrant"}),
    ("CL 987 NORTE # 654 - 21", "CL 987 # 654 NORTE - 21", {"via_quadrant", "cross_quadrant"}),
    ("CL 987 B N # 654 - 21", "CL 987 B # 654 NORTE - 21", {"via_quadrant", "cross_quadrant"}),
    ("CL 987 NORTE # 654 - 21", "CL 987 # 654 - 21 NORTE", {"via_quadrant", "plate_quadrant"}),
])
def test_quadrants_are_compared_per_position(a, b, fields):
    assert fields <= set(_mm(a, b))
    assert fields <= set(_mm(b, a))


@pytest.mark.parametrize("a, b", [
    ("CL 987 N # 654 - 21", "CL 987 NORTE # 654 - 21"),
    ("CL 987 # 654 N - 21", "CL 987 # 654 NORTE - 21"),
    ("CL 987 W # 654 - 21", "CL 987 OESTE # 654 - 21"),
    ("CL 987 O # 654 - 21", "CL 987 OESTE # 654 - 21"),
    ("CL 987 NORTE # 654 SUR - 21", "CL 987 NORTE # 654 SUR - 21"),
])
def test_same_position_quadrant_spellings_are_equivalent(a, b):
    assert _mm(a, b) == []


# ---------------------------------------------------------------------------
# W7: never raises
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a, b", [
    (None, None), (None, "CL 987 # 654 - 20"), ("CL 987 # 654 - 20", None),
    ("SIN ESPECIFICAR", "CL 987 # 654 - 20"), ("", ""),
])
def test_base_mismatches_never_raises(a, b):
    pa = None if a is None else parse_address(a)
    pb = None if b is None else parse_address(b)
    assert base_mismatches(pa, pb) == ["unparsed"]


# ---------------------------------------------------------------------------
# W3 / V4: candidate selection and the rejected-row schema
# ---------------------------------------------------------------------------
def _multi_row(raw, cadastrals, fused_scores, threshold=0.73):
    n = make_normalizer(cadastrals, threshold=threshold)
    parsed = parse_address(raw)
    canon = canonical(parsed, style="spaced", with_complement=True)
    scored = [(f, f, 0.9, 0.9, i, NAN) for i, f in enumerate(fused_scores)]
    return n._row(raw, canon, parsed, scored, 20, threshold)


def test_a_lower_ranked_exact_candidate_is_selected_when_the_top_one_mismatches():
    row = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48", "KR 987 # 654 - 84"], [0.99, 0.98])
    assert row["matched"] is True
    assert row["canonical_address"] == "KR 987 # 654 - 84"
    assert row["matched_cadastral_address"] == "KR 987 # 654 - 84"
    assert row["numero_predial_nacional"] == "P1"           # the selected candidate, not P0
    assert row["confidence"] == pytest.approx(0.98)
    assert row["override_rejected"] is False


def test_the_first_accepted_candidate_wins():
    row = _multi_row(
        "KR 987 # 654 - 84", ["KR 987 # 654 - 48", "KR 987 # 654 - 84", "KR 987 # 654 - 84"],
        [0.99, 0.95, 0.90])
    assert row["numero_predial_nacional"] == "P1"


def test_an_exact_candidate_below_the_cutoff_is_not_selected():
    row = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48", "KR 987 # 654 - 84"], [0.99, 0.50])
    assert row["matched"] is False
    assert row["canonical_address"] == "KR 987 # 654 - 84"
    assert row["numero_predial_nacional"] is None
    assert row["override_rejected"] is True


def test_no_accepted_candidate_clears_parcel_data_and_keeps_the_scores():
    row = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48", "KR 987 # 654 - 74"], [0.99, 0.98])
    assert row["matched"] is False
    assert row["numero_predial_nacional"] is None and row["manzana"] is None
    assert row["comuna"] is None and row["barrio_code"] is None
    assert np.isnan(row["lat"]) and np.isnan(row["lon"])
    assert row["n_predios_at_address"] == 0
    assert row["confidence"] == pytest.approx(0.99)          # confidence semantics unchanged
    assert row["top5_doc_ids"] == [0, 1]
    assert "cadastral_candidate_rejected" in row["parse_notes"]
    assert row["override_rejected"] is True


def test_low_score_row_keeps_the_candidate_and_is_not_flagged():
    row = row_for("KR 987 # 654 - 84", "KR 987 # 654 - 48", fused=0.2)
    assert row["override_rejected"] is False
    assert row["numero_predial_nacional"] == "P0"


def test_row_schema_is_identical_for_empty_matched_and_rejected_rows():
    n = make_normalizer(["KR 987 # 654 - 84"])
    out = n.normalize_batch(["KR 987 # 654 - 84", "KR 987 # 654 - 99", "SIN ESPECIFICAR"], use_model=False)
    assert out["override_rejected"].tolist() == [False, True, False]  # matched, rejected, no candidate


def test_rule_only_batch_marks_a_rejected_override():
    out = batch("KR 987 # 654 - 84 NORTE", ["KR 987 # 654 - 84"])
    assert out["matched"] is False
    assert out["override_rejected"] is True
    assert out["canonical_address"] == "KR 987 # 654 - 84 NORTE"


def test_unparsed_row_with_a_high_score_is_flagged_and_has_no_parcel():
    row = row_for("SIN ESPECIFICAR", "KR 987 # 654 - 84", fused=0.99)
    assert row["override_rejected"] is True
    assert row["numero_predial_nacional"] is None


# ---------------------------------------------------------------------------
# W6: NO, only as the number sign of the main address
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "CL 987 NO, 654 - 20", "CL 987 NO: 654 - 20", "CL 987 No,654-20", "CL 987 N°, 654 - 20",
])
def test_no_punctuation_before_the_plate_is_the_number_sign(raw):
    assert canonical(parse_address(raw)) == "CL 987 # 654 - 20"


@pytest.mark.parametrize("raw, expected", [
    ("CL 987 # 654 - 20 CASA NO,4", "CL 987 # 654 - 20 CA NO"),
    ("CL 987 # 654 - 20 CASA NO, 4", "CL 987 # 654 - 20 CA NO"),
    ("CL 987 # 654 - 20 APTO NO: 4", "CL 987 # 654 - 20 AP NO 4"),
    ("CL 987 654 - 20 NO, 4", "CL 987 # 654 - 20 NO"),
])
def test_no_punctuation_in_the_complement_is_left_alone(raw, expected):
    p = parse_address(raw)
    assert canonical(p) == expected
    assert "multiple_addresses_first_taken" not in p.notes


# ---------------------------------------------------------------------------
# V1: a comma between the via and the cross/plate group
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Carrera 987, 654B-20", "KR 987 # 654 B - 20"),
    ("Calle 987, 654 - 20", "CL 987 # 654 - 20"),
    ("Calle 987,654-20", "CL 987 # 654 - 20"),
])
def test_comma_between_via_and_cross_is_a_separator(raw, expected):
    assert canonical(parse_address(raw)) == expected


@pytest.mark.parametrize("raw, expected", [
    ("CL 987 # 654 - 20, Navarro", "CL 987 # 654 - 20"),
    ("CL 987 # 654 - 20, 3", "CL 987 # 654 - 20"),
    ("Calle 987, Navarro", "CL 987"),
    ("Calle 987,", "CL 987"),
])
def test_trailing_comma_segments_are_still_dropped(raw, expected):
    assert canonical(parse_address(raw)) == expected


# ---------------------------------------------------------------------------
# V2: glued tokens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("CL 12 # 94b1-51", "CL 12 # 94 B 1 - 51"),
    ("KR 94b1 # 51 - 7", "KR 94 B 1 # 51 - 07"),
    ("CL 12 # 94 B 1 - 51", "CL 12 # 94 B 1 - 51"),
    ("Calle 987 # 654-0", "CL 987 # 654 - 00"),             # plate '00' padding is accepted
])
def test_glued_token_parses_are_pinned(raw, expected):
    assert canonical(parse_address(raw)) == expected


def test_ambiguous_glued_run_is_unchanged_by_the_plate_relaxation():
    # round 7: via number + letter + digit glued to the letter + cross designator + plate is unambiguous
    # (the reading the explicit-hash form always had); the stricter reading below stays for the ambiguous shapes
    assert canonical(parse_address("KR 987G6 654S1 39")) == "KR 987 G 6 # 654 S 1 - 39"
    assert canonical(parse_address("KR 987G6 654S1 39")) == canonical(parse_address("KR 987G6 # 654S1 - 39"))


# ---------------------------------------------------------------------------
# Cross-package consistency: guard and matcher give the same quadrant verdict
# ---------------------------------------------------------------------------
_EMPAREJADOR_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "emparejador_direcciones", "src")

_PAIRS = [
    ("CL 987 N # 654 - 21", "CL 987 NORTE # 654 - 21", True),
    ("CL 987 # 654 N - 21", "CL 987 # 654 NORTE - 21", True),
    ("CL 987 W # 654 - 21", "CL 987 OESTE # 654 - 21", True),
    ("CL 987 O # 654 - 21", "CL 987 OESTE # 654 - 21", True),
    ("CL 987 D N # 654 - 21", "CL 987 D NORTE # 654 - 21", True),
    ("CL 987 NORTE # 654 - 21", "CL 987 NORTE # 654 - 21", True),
    ("CL 987 N # 654 - 21", "CL 987 # 654 N - 21", False),
    ("CL 987 NORTE # 654 - 21", "CL 987 # 654 NORTE - 21", False),
    ("CL 987 B N # 654 - 21", "CL 987 B # 654 NORTE - 21", False),
    ("CL 987 NORTE # 654 SUR - 21", "CL 987 SUR # 654 NORTE - 21", False),
    ("CL 987 N # 654 N - 21", "CL 987 NORTE # 654 - 21", False),
    ("AV 876 N # 543 - 06", "AV 876 NORTE # 543 - 06", True),
    ("AV 876 O # 543 - 06", "AV 876 OESTE # 543 - 06", True),
    ("AV 876 # 543 O - 06", "AV 876 # 543 OESTE - 06", True),
    ("CL 987 N # 654 - 21", "CL 986 NORTE # 654 - 21", False),
    ("CL 987 N # 654 - 21", "KR 987 NORTE # 654 - 21", False),
    ("CL 987 A N # 654 - 21", "CL 987 NORTE # 654 - 21", False),
    ("CL 987 # 654 N - 21", "CL 987 # 655 NORTE - 21", False),
    ("AV 876 N # 543 - 06", "CL 876 NORTE # 543 - 06", False),
    ("AV 876 # 543 N - 06", "AV 876 NORTE # 543 - 06", False),
    ("AV 876 # 543 NORTE - 06", "AV 876 N # 543 - 06", False),
    ("AV 876 # 543 O - 06", "AV 876 OESTE # 543 - 06", False),
    ("AV 876 O # 543 - 06", "AV 876 # 543 OESTE - 06", False),
    ("CL 987 # 654 - 21", "CL 987 NORTE # 654 - 21", False),
    ("CL 987 NORTE # 654 - 21", "CL 987 SUR # 654 - 21", False),
    ("KR 987 S # 654 - 21", "KR 987 SUR # 654 - 21", False),
    ("KR 987 D N # 654 - 21", "KR 987 NORTE # 654 - 21", False),
]


@pytest.mark.parametrize("a, b, same", _PAIRS)
def test_guard_and_matcher_agree_on_quadrants(a, b, same):
    if not os.path.isdir(_EMPAREJADOR_SRC):
        pytest.skip("emparejador_direcciones is not next to this package")
    sys.path.insert(0, _EMPAREJADOR_SRC)
    try:
        from emparejador.matcher import match
    finally:
        sys.path.remove(_EMPAREJADOR_SRC)
    guard = base_mismatches(parse_address(a), parse_address(b)) == []
    assert guard is same
    assert match(a, b).is_match is same


# ---------------------------------------------------------------------------
# Glued plate letters always reach the compared fields
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "KR 987 # 654 - 66C", "KR 987 # 654 - 66T", "KR 987 # 654 - 66AT", "KR 987 # 654 - 3A4",
    "KR 987 # 654 - 66BIS", "KR 987 # 654 - 20NORTE",
    "A 987N5 4 F # - 654T",            # cross-segment leftovers in front of the glued letter
])
def test_glued_plate_letters_are_compared(raw):
    p = parse_address(raw)
    assert p.parse_ok and p.plate is not None
    bare = parse_address(canonical(p, with_complement=False))
    assert any(f.startswith("plate_") for f in base_mismatches(p, bare))
    assert base_mismatches(p, p) == []
