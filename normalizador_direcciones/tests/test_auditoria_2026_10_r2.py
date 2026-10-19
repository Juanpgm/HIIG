"""Second-round regression tests of the October 2026 audit of the normalizer.

Round 2 closes what the independent review of round 1 found: the ambiguity
competition must still see candidates that only fail the one-sided suffix rule,
the suffix rejection is auditable from ``motivo``, and a set of rule branches
that no test pinned. Synthetic addresses only (see ``test_auditoria_2026_10.py``
and the CHANGELOG for the audit itself).
"""

from __future__ import annotations

import pytest

from cali_address.parser import parse_address
from cali_address.service import normalize_strict


def _ambiguity_run(stub_normalizer, query, best, rival, scores=(0.90, 0.895), delta=0.02, **kw):
    stub = stub_normalizer(
        [{"direccion": best, "manzana": "M1", "predial": "P1"},
         {"direccion": rival, "manzana": "M2", "predial": "P2"}],
        scores=[list(scores)], threshold=0.5,
    )
    sink: list = []
    row = normalize_strict(stub, [query], gazetteer=None, ambiguity_delta=delta, feature_sink=sink, **kw).iloc[0]
    return row, sink[0]


# ---------------------------------------------------------------------------
# 1. A one-sided suffix mismatch rejects a candidate, but it still competes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, best, rival", [
    ("KR 987 H # 654 - 21", "KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21"),          # cadastre-only via suffix
    ("KR 987 H 3 # 654 - 21", "KR 987 H 3 # 654 - 21", "KR 987 H # 654 - 21"),        # query-only via suffix
    ("KR 876 B # 543 B - 29", "KR 876 B # 543 B - 29", "KR 876 B # 543 B 2 - 29"),    # cadastre-only cross suffix
    ("KR 876 B # 543 B 2 - 29", "KR 876 B # 543 B 2 - 29", "KR 876 B # 543 B - 29"),  # query-only cross suffix
])
def test_a_rival_that_only_fails_the_one_sided_suffix_still_makes_the_row_ambiguous(stub_normalizer, query, best, rival):
    row, _ = _ambiguity_run(stub_normalizer, query, best, rival)
    assert row["estado"] == "SIN_MATCH"
    assert row["motivo"] == "ambiguo: 1 candidatos cercanos en otras manzanas"


def test_that_rival_also_feeds_the_margin_features(stub_normalizer):
    row, feats = _ambiguity_run(
        stub_normalizer, "KR 987 H # 654 - 21", "KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21", delta=0.0)
    assert row["estado"] == "OK" and row["margen"] == pytest.approx(0.005)
    assert feats["margen_manzana"] == pytest.approx(0.005) and feats["n_competidores"] == 1.0


def test_a_rival_with_different_suffixes_on_both_sides_is_still_dropped(stub_normalizer):
    row, feats = _ambiguity_run(
        stub_normalizer, "KR 987 H 1 # 654 - 21", "KR 987 H 1 # 654 - 21", "KR 987 H 3 # 654 - 21")
    assert row["estado"] == "OK" and row["margen"] is None
    assert feats["n_competidores"] == 0.0


def test_a_rival_with_one_one_sided_and_one_two_sided_suffix_is_dropped(stub_normalizer):
    row, _ = _ambiguity_run(
        stub_normalizer, "KR 876 B 1 # 543 B - 29", "KR 876 B 1 # 543 B - 29", "KR 876 B 3 # 543 B 2 - 29")
    assert row["estado"] == "OK" and row["margen"] is None


def test_a_one_sided_suffix_rival_with_another_hard_violation_is_dropped(stub_normalizer):
    row, _ = _ambiguity_run(
        stub_normalizer, "KR 987 H # 654 - 21", "KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 40")
    assert row["estado"] == "OK" and row["margen"] is None


def test_a_one_sided_suffix_rival_still_pays_the_soft_budget(stub_normalizer):
    near_plate = ("KR 987 H # 654 - 21", "KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 22")
    row, _ = _ambiguity_run(stub_normalizer, *near_plate)                   # one soft violation, max_soft=1
    assert row["estado"] == "SIN_MATCH" and row["motivo"].startswith("ambiguo")
    row, _ = _ambiguity_run(stub_normalizer, *near_plate, max_soft=0)       # budget exhausted: not a rival
    assert row["estado"] == "OK"


def test_a_one_sided_suffix_rival_below_the_threshold_or_far_behind_does_not_compete(stub_normalizer):
    row, _ = _ambiguity_run(
        stub_normalizer, "KR 987 H # 654 - 21", "KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21", scores=(0.90, 0.40))
    assert row["estado"] == "OK" and row["margen"] is None
    row, _ = _ambiguity_run(
        stub_normalizer, "KR 987 H # 654 - 21", "KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21", scores=(0.90, 0.80))
    assert row["estado"] == "OK" and row["margen"] == pytest.approx(0.10)


def test_a_one_sided_suffix_rival_in_the_same_manzana_is_no_manzana_competitor(stub_normalizer):
    stub = stub_normalizer(
        [{"direccion": "KR 987 H # 654 - 21", "manzana": "M1", "predial": "P1"},
         {"direccion": "KR 987 H 3 # 654 - 21", "manzana": "M1", "predial": "P2"}],
        scores=[[0.90, 0.895]], threshold=0.5,
    )
    row = normalize_strict(stub, ["KR 987 H # 654 - 21"], gazetteer=None, ambiguity_delta=0.02).iloc[0]
    assert row["estado"] == "OK" and row["margen"] is None


# ---------------------------------------------------------------------------
# 2. The one-sided suffix rejection is auditable from ``motivo``
# ---------------------------------------------------------------------------
def _single(stub_normalizer, query, cadastral, **kw):
    stub = stub_normalizer([{"direccion": cadastral, "predial": "P1", "manzana": "M1"}], scores=[[0.95]], threshold=0.5)
    kw.setdefault("ambiguity_delta", 0.0)
    return normalize_strict(stub, [query], gazetteer=None, **kw).iloc[0]


@pytest.mark.parametrize("query, cadastral, message", [
    ("KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21", "numero de sufijo de via - != 3"),
    ("KR 987 H 3 # 654 - 21", "KR 987 H # 654 - 21", "numero de sufijo de via 3 != -"),
    ("KR 876 B # 543 B - 29", "KR 876 B # 543 B 2 - 29", "numero de sufijo de cruce - != 2"),
])
def test_one_sided_suffix_rejection_carries_the_stable_audit_token(stub_normalizer, query, cadastral, message):
    row = _single(stub_normalizer, query, cadastral)
    assert row["estado"] == "SIN_MATCH"
    assert row["motivo"] == f"{message}; regla: sufijo unilateral"


def test_the_audit_token_appears_once_even_with_both_suffix_fields_one_sided(stub_normalizer):
    row = _single(stub_normalizer, "KR 876 B 3 # 543 B - 29", "KR 876 B # 543 B 2 - 29")
    assert row["motivo"].count("regla: sufijo unilateral") == 1
    assert "sufijo de via 3 != -" in row["motivo"] and "sufijo de cruce - != 2" in row["motivo"]


def test_a_two_sided_suffix_difference_has_no_audit_token(stub_normalizer):
    row = _single(stub_normalizer, "KR 987 H 1 # 654 - 21", "KR 987 H 3 # 654 - 21")
    assert row["estado"] == "SIN_MATCH" and "regla:" not in row["motivo"]


def test_a_clean_row_and_other_rejections_have_no_audit_token(stub_normalizer):
    assert _single(stub_normalizer, "KR 987 H # 654 - 21", "KR 987 H # 654 - 21")["motivo"] == ""
    assert "regla:" not in _single(stub_normalizer, "KR 987 H # 654 - 21", "KR 123 H # 654 - 21")["motivo"]


# ---------------------------------------------------------------------------
# 3. Rule branches that round 1 left unpinned (found by mutation testing)
# ---------------------------------------------------------------------------
def test_the_via_letter_cap_needs_a_letter_on_exactly_one_side(stub_normalizer):
    # a letter on BOTH sides (different) is a soft violation only under ``letra_via``; it is NOT one-sided
    row = _single(stub_normalizer, "KR 987 F # 654 - 21", "KR 987 H # 654 - 21", soft_rules=["letra_via"])
    assert row["estado"] == "OK" and row["nivel_precision"] == "manzana"
    assert row["motivo"] == "aproximado: letra de via F != H"
    assert "nivel maximo esquina" not in row["motivo"]


def test_a_cardinal_letter_on_the_cross_is_the_same_quadrant_as_the_cadastral_word(stub_normalizer):
    row = _single(stub_normalizer, "CL 987 # 654 N - 21", "CL 987 # 654 NORTE - 21")   # query letter, cadastre word
    assert row["estado"] == "OK" and row["nivel_precision"] == "predio" and row["motivo"] == ""


def test_two_one_sided_quadrants_say_the_ceiling_once(stub_normalizer):
    row = _single(stub_normalizer, "CL 987 NORTE # 654 OESTE - 21", "CL 987 # 654 - 21")
    assert row["estado"] == "OK" and row["nivel_precision"] == "manzana"
    assert row["motivo"] == (
        "aproximado: cuadrante de via NORTE != -; cuadrante de cruce OESTE != -"
        "; nivel maximo manzana: cuadrante en un solo lado"
    )


def test_a_via_letter_and_a_quadrant_ceiling_both_show_and_the_lowest_wins(stub_normalizer):
    row = _single(stub_normalizer, "KR 987 NORTE # 654 - 21", "KR 987 H # 654 - 21")
    # one-sided via letter is capped at manzana now (was esquina); both reasons still show
    assert row["nivel_precision"] == "manzana"
    assert row["motivo"].count("nivel maximo") == 2
    assert "nivel maximo manzana: via con letra en un solo lado" in row["motivo"]
    assert "nivel maximo manzana: cuadrante en un solo lado" in row["motivo"]


# ---------------------------------------------------------------------------
# 4. A quadrant that conflicts across fields is a hard rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral, message", [
    ("CL 987 NORTE # 654 - 21", "CL 987 # 654 OESTE - 21", "cuadrante NORTE != OESTE"),
    ("CL 987 # 654 OESTE - 21", "CL 987 NORTE # 654 - 21", "cuadrante OESTE != NORTE"),
    ("CL 987 # 654 NORTE - 21", "CL 987 OESTE # 654 - 21", "cuadrante NORTE != OESTE"),
])
def test_conflicting_quadrants_on_different_fields_are_rejected(stub_normalizer, query, cadastral, message):
    row = _single(stub_normalizer, query, cadastral)
    assert row["estado"] == "SIN_MATCH" and row["numero_predial_nacional"] is None
    assert row["motivo"] == message


def test_a_same_field_quadrant_conflict_is_reported_once(stub_normalizer):
    row = _single(stub_normalizer, "CL 987 NORTE # 654 - 21", "CL 987 SUR # 654 - 21")
    assert row["motivo"] == "cuadrante de via NORTE != SUR"


def test_a_quadrant_shared_by_both_sides_is_not_a_conflict(stub_normalizer):
    # the query also types a cross quadrant the cadastre lacks: one-sided (capped), not contradictory
    row = _single(stub_normalizer, "CL 987 NORTE # 654 OESTE - 21", "CL 987 NORTE # 654 - 21")
    assert row["estado"] == "OK" and row["nivel_precision"] == "manzana"


# ---------------------------------------------------------------------------
# 5. Parser: bare tower alias and trailing-unit edge cases
# ---------------------------------------------------------------------------
from cali_address.parser import _bare_tower_alias, extract_complement_units  # noqa: E402


def _units_of(raw):
    p = parse_address(raw)
    assert p.parse_ok, raw
    return extract_complement_units(p.complement_text)


@pytest.mark.parametrize("raw, expected", [
    ("Calle 987 # 654-21 TORRE O 2", [("TO", "O2")]),          # the 2 is part of the value, as in TORRE B 2
    ("Calle 987 # 654-21 TORRE O", [("TO", "O")]),
    ("Calle 987 # 654-21 TORRE B 2", [("TO", "B2")]),
    ("Calle 987 # 654-21 TORRE O 2 PISOS", [("TO", "O")]),     # a count noun still ends the value
    ("Calle 987 # 654-21 T2 APTO 301", [("TO", "2"), ("AP", "301")]),
    ("Calle 987 # 654-21 T12", [("TO", "12")]),
    ("Calle 987 # 654-21 T 2 APTO 301", [("TO", "2"), ("AP", "301")]),
])
def test_tower_value_edge_cases(raw, expected):
    assert _units_of(raw) == expected


@pytest.mark.parametrize("raw", [
    "Calle 987 # 654-21 ESQUINA T 2",     # ESQUINA is a place word, T is not a tower
    "Calle 987 # 654-21 FRENTE T B",
    "Calle 987 # 654-21 AP 101 T2",        # glued after a number: a unit letter, not a tower
    "Calle 987 # 654-21 BLQ T2",
    "Calle 987 # 654-21 T1234",            # 4 digits is not a tower number
    "Calle 987 # 654-21 NO T 2",           # marker word before the T
    "Calle 987 # 654-21 N T B",
    "Calle 987 # 654-21 LETRA T 2",
])
def test_bare_t_guards_keep_the_text_unchanged(raw):
    assert ("TO", "2") not in _units_of(raw) and ("TO", "B") not in _units_of(raw)


@pytest.mark.parametrize("text, expected", [
    ("T 123", "TO 123"),         # 3 digits is the longest tower number
    ("T 1234", "T 1234"),
    ("T123", "TO 123"),          # glued: same 3-digit limit
    ("T1234", "T1234"),
    ("AP 101 T2", "AP 101 T2"),
    ("ESQUINA T2", "ESQUINA T2"),
    ("NO T2", "NO T2"),
    ("BLQ T2", "BLQ T2"),
    ("T2 APTO 301", "TO 2 APTO 301"),
    ("T B", "TO B"),
    ("NO T B", "NO T B"),        # a marker word right before T
    ("NRO T 2", "NRO T 2"),
    ("N T 2", "N T 2"),
    ("ESQUINA T 2", "ESQUINA T 2"),
    ("T", "T"),
    ("", ""),
])
def test_bare_tower_alias_directly(text, expected):
    assert _bare_tower_alias(text) == expected


@pytest.mark.parametrize("raw, complement_text", [
    ("Calle 987 # 654-21, Torre B, Carrera 11 # 22-33", "TORRE B"),     # a 2nd address after a unit: not text
    ("Calle 987 # 654-21, San Fernando, Torre B", "TORRE B"),            # a place before the unit
    ("Calle 987 # 654-21; Torre B, Apto 4, Cali", "TORRE B APTO 4"),
    ("Calle 987 # 654-21, Torre B; Apto 4", "TORRE B APTO 4"),
])
def test_each_comma_or_semicolon_segment_is_judged_on_its_own(raw, complement_text):
    assert parse_address(raw).complement_text == complement_text


# ---------------------------------------------------------------------------
# 6. Degenerate inputs: concrete, stable outputs (never an exception)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, notes", [
    ("", ["empty_input"]),
    ("  ", ["empty_input"]),
    (None, ["null_input"]),
    (",", ["empty_after_cleanup"]),
    (";", ["multiple_addresses_first_taken", "empty_after_cleanup"]),
    (",;", ["multiple_addresses_first_taken", "empty_after_cleanup"]),
    ("Torre B", ["implicit_hash", "empty_via_segment"]),
    (", Torre B", ["implicit_hash", "empty_via_segment"]),
    ("; Torre B", ["multiple_addresses_first_taken", "empty_after_cleanup"]),
])
def test_degenerate_inputs_have_a_concrete_unparseable_result(raw, notes):
    p = parse_address(raw)
    assert p.parse_ok is False
    assert p.notes == notes
    assert p.complement_text is None and p.complement is None and p.via_type is None and p.plate is None
