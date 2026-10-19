"""Truncated parses (a comma that cost the parse its cross street or plate) can never be verified.

Round 5. On ``CL 987, # 654 - 12`` the parser used to keep only ``CL 987`` and note
``dropped_trailing_segment``; the guard then verified the TRUNCATED base against any parcel of that
street. Two layers are pinned here:

* parser: a comma that merely separates a bare via from ``cross - plate`` is read as one address; a
  dropped tail that may carry address data (it has a digit) from a head that lacks the cross or the
  plate is noted ``truncated_parse``;
* guard: research mode rejects such a parse for EVERY candidate rank, production caps the tier at
  ``via`` with the motive ``parse truncado``.

Synthetic addresses only.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_stub
from test_override_guard import NAN, make_normalizer

from cali_address.inference import base_mismatches
from cali_address.parser import TRUNCATION_NOTES, canonical, is_truncated, parse_address
from cali_address.service import normalize_strict


def fields(raw):
    p = parse_address(raw)
    return (p.via_type, p.via_number, p.cross_number, p.plate)


# ---------------------------------------------------------------------------
# parser: the comma between a bare via and cross + plate
# ---------------------------------------------------------------------------
RECOVERED = [
    ("CL 987, # 654 - 12", ("CL", "987", "654", "12")),
    ("CL 987, 654 - 12", ("CL", "987", "654", "12")),
    ("CL 987 , # 654-12", ("CL", "987", "654", "12")),
    ("CL 987,# 654 - 12", ("CL", "987", "654", "12")),
    ("KR 987 A, 654 B - 12", ("KR", "987", "654", "12")),
    ("KR 987, 8B-20", ("KR", "987", "8", "20")),
    ("CL 987 OESTE, # 654 OESTE - 12", ("CL", "987", "654", "12")),
    ("AV 987 B 6, 654 NORTE - 12", ("AV", "987", "654", "12")),
    ("KR 987 C 2, # 654 A - 40", ("KR", "987", "654", "40")),
    ("CL 987, # 654 - 12, CALI", ("CL", "987", "654", "12")),
    ("CL 987, # 654 - 12, CALI, VALLE DEL CAUCA", ("CL", "987", "654", "12")),
    ("CL 987, # 654 - 12 AP 301", ("CL", "987", "654", "12")),
    ("cl 987, # 654 - 12", ("CL", "987", "654", "12")),
    ("CL 987 OESTE, # AV 654 OESTE - 12", ("CL", "987", "654", "12")),   # the cross carries its own via type
    ("CL 987 B, #AV 654-12", ("CL", "987", "654", "12")),
]


@pytest.mark.parametrize(("raw", "expected"), RECOVERED)
def test_comma_between_bare_via_and_cross_plate_is_parsed_not_dropped(raw, expected):
    p = parse_address(raw)
    assert fields(raw) == expected
    assert "dropped_trailing_segment" not in p.notes and not is_truncated(p)


def test_recovered_comma_equals_the_comma_free_address():
    for raw, plain in [("CL 987, # 654 - 12", "CL 987 # 654 - 12"),
                       ("KR 987 A, 654 B - 12", "KR 987 A # 654 B - 12"),
                       ("CL 987 OESTE, # 654 OESTE - 12", "CL 987 OESTE # 654 OESTE - 12")]:
        assert canonical(parse_address(raw)) == canonical(parse_address(plain))


TRUNCATED = [
    "CL 987 # 654, 12",            # the plate sits after the comma
    "CL 987, # 654",               # cross without plate
    "CL 987, ESQUINA 654 - 12",    # words before the cross: not a clean cross + plate
    "CL 987, 20 DE JULIO - 12",    # a name with a number is a neighbourhood, not a cross
    "CL 987, 12",
]


@pytest.mark.parametrize("raw", TRUNCATED)
def test_a_dropped_tail_with_digits_from_an_incomplete_head_is_truncated(raw):
    p = parse_address(raw)
    assert "dropped_trailing_segment" in p.notes
    assert is_truncated(p) and "truncated_parse" in p.notes


NOT_TRUNCATED = [
    "CL 987 # 654 - 12, SAN FERNANDO",     # a neighbourhood after a complete address
    "CL 987 # 654 - 12, TORRE B",
    "CL 987 # 654 - 12, BARRIO 3 DE JULIO",
    "CL 987, SAN FERNANDO",                # nothing numeric was lost
    "CL 987 # 654, SAN FERNANDO",
    "CL 987 # 654 - 12",
    "CL 987 # 654 - 12, CALI",
]


@pytest.mark.parametrize("raw", NOT_TRUNCATED)
def test_complete_heads_and_tails_without_digits_are_not_truncated(raw):
    assert not is_truncated(parse_address(raw))


@pytest.mark.parametrize("raw", [None, "", "   ", ",", ",,,", "CL 987,", "CL 987, #", "CL 987,,, 12",
                                 "# , 12", "Ñú, 12", "CL 987, # 654 - 12" + ", X" * 400,
                                 "CL 987 # 654, " + "9" * 5000])
def test_degenerate_and_long_inputs_never_raise_and_are_idempotent(raw):
    first, second = parse_address(raw), parse_address(raw)
    assert first.notes == second.notes and first.parse_ok == second.parse_ok
    assert isinstance(is_truncated(first), bool)


def test_is_truncated_tolerates_none_and_foreign_objects():
    assert is_truncated(None) is False
    assert is_truncated(object()) is False
    assert TRUNCATION_NOTES == ("truncated_parse",)


# ---------------------------------------------------------------------------
# research guard (default mode of base_mismatches / AddressNormalizer._row)
# ---------------------------------------------------------------------------
def research_row(raw, cadastrals, fused):
    n = make_normalizer(cadastrals)
    parsed = parse_address(raw)
    canon = canonical(parsed, style="spaced", with_complement=True)
    scored = [(f, f, 0.9, 0.9, i, NAN) for i, f in enumerate(fused)]
    return n._row(raw, canon, parsed, scored, 20, 0.73)


def test_base_mismatches_rejects_a_truncated_parse_even_when_the_truncated_base_equals_the_parcel():
    truncated = parse_address("CL 987 # 654, 12")
    assert base_mismatches(truncated, parse_address("CL 987 # 654 -")) == ["truncated_parse"]
    # the same parcel is a perfect match for the untruncated query
    assert base_mismatches(parse_address("CL 987 # 654 -"), parse_address("CL 987 # 654 -")) == []


def test_base_mismatches_unparsed_still_wins_over_truncated():
    assert base_mismatches(parse_address("CL 987 # 654, 12"), parse_address("hola")) == ["unparsed"]
    assert base_mismatches(None, parse_address("CL 987 # 654 -")) == ["unparsed"]


def test_research_row_rejects_truncated_and_keeps_the_rule_text():
    row = research_row("CL 987 # 654, 12", ["CL 987 # 654 -"], [0.99])
    assert row["override_rejected"] is True and row["matched"] is False
    assert row["selected_doc_id"] == -1 and row["numero_predial_nacional"] is None
    assert row["canonical_address"] == "CL 987 # 654"           # the rule text, not the parcel's
    assert "truncated_parse" in row["parse_notes"]


def test_research_row_never_accepts_a_lower_ranked_candidate_of_a_truncated_parse():
    row = research_row("CL 987 # 654, 12", ["CL 987 # 655 -", "CL 987 # 654 -"], [0.99, 0.98])
    assert row["override_rejected"] is True and row["matched"] is False
    assert "cadastral_candidate_rank" not in row["parse_notes"]


def test_research_row_still_accepts_a_complete_parse_with_a_trailing_neighbourhood():
    row = research_row("CL 987 # 654 - 12, SAN FERNANDO", ["CL 987 # 654 - 12"], [0.95])
    assert row["matched"] is True and row["override_rejected"] is False


def test_research_row_accepts_the_recovered_comma_form():
    row = research_row("CL 987, # 654 - 12", ["CL 987 # 654 - 12"], [0.95])
    assert row["matched"] is True and row["canonical_address"] == "CL 987 # 654 - 12"


def test_below_cutoff_truncated_row_is_not_marked_rejected():
    row = research_row("CL 987 # 654, 12", ["CL 987 # 654 -"], [0.2])
    assert row["matched"] is False and row["override_rejected"] is False


# ---------------------------------------------------------------------------
# production (position mode + service)
# ---------------------------------------------------------------------------
def test_position_guard_caps_a_truncated_parse_at_via():
    hard, caps = base_mismatches(parse_address("CL 987 # 654, 12"), parse_address("CL 987 # 654 -"),
                                 mode="position")
    assert hard == [] and caps["truncated_parse"] == "via"
    _hard, caps = base_mismatches(parse_address("CL 987 # 654 - 12"), parse_address("CL 987 # 654 - 12"),
                                  mode="position")
    assert "truncated_parse" not in caps


def test_service_caps_truncated_row_at_via_and_withholds_position():
    stub = make_stub([{"direccion": "CL 987 # 654 -", "lat": 3.41, "lon": -76.5, "predial": "P1"}])
    row = normalize_strict(stub, ["CL 987 # 654, 12"], gazetteer=None, ambiguity_delta=0.0).iloc[0]
    assert row["estado"] == "OK" and row["position_tier"] == "via"
    assert row["numero_predial_nacional"] is None and row["manzana"] is None
    assert row["lat"] is None and row["lon"] is None and row["position_source"] == "none"
    assert "parse truncado" in row["motivo"]


def test_service_truncated_never_falls_back_to_a_lower_ranked_candidate():
    docs = [{"direccion": "CL 987 # 655 -", "predial": "P1"}, {"direccion": "CL 987 # 654 -", "predial": "P2"}]
    stub = make_stub(docs, scores=[[0.9, 0.8]])
    row = normalize_strict(stub, ["CL 987 # 654, 12"], gazetteer=None, ambiguity_delta=0.0).iloc[0]
    assert row["numero_predial_nacional"] in (None, "P1") and row["position_tier"] in ("via", "none")


def test_service_complete_head_with_barrio_keeps_its_tier():
    stub = make_stub([{"direccion": "CL 987 # 654 - 12", "lat": 3.41, "lon": -76.5, "predial": "P1"}])
    row = normalize_strict(stub, ["CL 987 # 654 - 12, SAN FERNANDO"], gazetteer=None, ambiguity_delta=0.0).iloc[0]
    assert row["estado"] == "OK" and "parse truncado" not in row["motivo"]
    assert row["position_tier"] in ("predio", "manzana")


def test_service_is_deterministic_for_truncated_rows():
    stub = make_stub([{"direccion": "CL 987 # 654 -", "predial": "P1"}])
    a = normalize_strict(stub, ["CL 987 # 654, 12"] * 3, gazetteer=None, ambiguity_delta=0.0)
    assert a.drop(columns=[]).nunique(dropna=False).max() == 1
    assert np.all(a["position_tier"] == "via")
