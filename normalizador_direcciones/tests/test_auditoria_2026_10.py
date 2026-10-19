"""Regression tests for the October 2026 audit of the normalizer.

Each section reproduces the input the audit demonstrated and pins the corrected
behaviour, plus the boundary / empty / malformed cases around it. Audit numbers
(measured on real data) are summarised in the CHANGELOG (the audit report itself is not part
of this repository); the tests here only use synthetic addresses.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cali_address.parser import parse_address
from cali_address.service import _classify_violations, normalize_strict


def _cv(query, cadastral, soft_rules=frozenset(), plate_tolerance=0, min_struct=0.0, struct=1.0):
    q, c = parse_address(query), parse_address(cadastral)
    assert q.parse_ok and c.parse_ok
    return _classify_violations(q, c, min_struct, struct, plate_tolerance, soft_rules)


def _row(stub_normalizer, query, cadastral, doc_extra=None, neighbours=(), **kw):
    doc = {"direccion": cadastral, "predial": "P1", "manzana": "M1"}
    doc.update(doc_extra or {})
    # ``neighbours``: plates of other parcels on the same street pair (position tiers interpolate
    # between them); lat grows linearly with the plate so every interpolation is predictable
    docs = [doc] + [{"direccion": cadastral.rsplit("-", 1)[0] + f"- {n}", "predial": f"N{n}", "manzana": "M1",
                     "lat": 3.4 + (n - 40) * 2.5e-5, "lon": -76.5} for n in neighbours]  # ~2.8 m per plate: inside the 50 m interpolation cap
    stub = stub_normalizer(docs, scores=[[0.95] + [0.0] * len(neighbours)], threshold=0.5)
    kw.setdefault("ambiguity_delta", 0.0)
    return normalize_strict(stub, [query], gazetteer=None, **kw).iloc[0]


# ---------------------------------------------------------------------------
# 1. Numeric via / cross suffix (KR 987 H 1..4 are different streets)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral", [
    ("KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21"),        # audit demo: cadastre-only suffix
    ("KR 987 H 3 # 654 - 21", "KR 987 H # 654 - 21"),        # query-only suffix
    ("KR 987 H 1 # 654 - 21", "KR 987 H 3 # 654 - 21"),      # both sides, different
    ("KR 876 B # 543 B 3 - 29", "KR 876 B # 543 B 2 - 29"),    # cross suffix, both sides
    ("KR 876 B # 543 B - 29", "KR 876 B # 543 B 2 - 29"),      # cross suffix, cadastre-only
    ("KR 876 B # 543 B 2 - 29", "KR 876 B # 543 B - 29"),      # cross suffix, query-only
])
def test_numeric_suffix_difference_is_a_hard_violation(query, cadastral):
    hard, soft = _cv(query, cadastral, soft_rules=frozenset({"letra_via", "letra_cruce"}))
    assert any("sufijo" in m for m in hard), (hard, soft)
    assert not any("sufijo" in m for m in soft)


def test_numeric_suffix_message_names_the_field_and_both_values():
    hard, _ = _cv("KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21")
    assert "numero de sufijo de via - != 3" in hard
    hard, _ = _cv("KR 876 B # 543 B 3 - 29", "KR 876 B # 543 B 2 - 29")
    assert "numero de sufijo de cruce 3 != 2" in hard


def test_equal_numeric_suffix_is_clean():
    assert _cv("KR 987 H 3 # 654 - 21", "KR 987 H 3 # 654 - 21") == ([], [])
    assert _cv("KR 876 B # 543 B 2 - 29", "KR 876 B # 543 B 2 - 29") == ([], [])


def test_numeric_suffix_compares_without_leading_zeros_and_blanks():
    q, c = parse_address("KR 987 H 3 # 654 - 21"), parse_address("KR 987 H 3 # 654 - 21")
    for qs, cs in [("03", "3"), (" 3 ", "3"), ("3", "003")]:
        hard, soft = _classify_violations(replace(q, via_suffix=qs), replace(c, via_suffix=cs), 0.0, 1.0)
        assert (hard, soft) == ([], [])
    for blank in (None, "", "  "):
        hard, _ = _classify_violations(replace(q, via_suffix=blank), replace(c, via_suffix=blank), 0.0, 1.0)
        assert hard == []


def test_a_zero_suffix_is_not_the_same_as_no_suffix():
    q = parse_address("KR 987 H 3 # 654 - 21")
    hard, _ = _classify_violations(replace(q, via_suffix="0"), replace(q, via_suffix=None), 0.0, 1.0)
    assert any("sufijo de via" in m for m in hard)


def test_suffix_mismatch_is_not_an_ok_row_end_to_end(stub_normalizer):
    row = _row(stub_normalizer, "KR 987 H # 654 - 21", "KR 987 H 3 # 654 - 21")
    assert row["estado"] == "SIN_MATCH"
    assert "sufijo de via" in row["motivo"]
    assert row["numero_predial_nacional"] is None and row["nivel_precision"] is None


def test_suffix_mismatch_cannot_be_rescued_by_soft_rules(stub_normalizer):
    row = _row(stub_normalizer, "KR 987 H 1 # 654 - 21", "KR 987 H 3 # 654 - 21",
               soft_rules=["letra_via", "letra_cruce", "bis_via", "bis_cruce", "cuadrante_compl"], max_soft=5)
    assert row["estado"] == "SIN_MATCH" and "sufijo" in row["motivo"]


def test_matching_suffix_keeps_predio_and_clean_motivo(stub_normalizer):
    row = _row(stub_normalizer, "KR 987 H 3 # 654 - 21", "KR 987 H 3 # 654 - 21")
    assert row["estado"] == "OK" and row["nivel_precision"] == "predio" and row["motivo"] == ""


# ---------------------------------------------------------------------------
# 2. One-sided via letter (KR 987 vs KR 987 H are different streets)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral", [
    ("KR 987 # 654 - 21", "KR 987 H # 654 - 21"),   # cadastre-only letter (audit demo shape)
    ("KR 987 H # 654 - 21", "KR 987 # 654 - 21"),   # query-only letter
    ("KR 987 B # 654 - 21", "KR 987 # 654 - 21"),
    ("KR 987 # 654 - 21", "KR 987 AB # 654 - 21"),  # two-letter run
])
def test_one_sided_via_letter_is_capped_at_manzana_tier(stub_normalizer, query, cadastral):
    sink: list = []
    row = _row(stub_normalizer, query, cadastral, feature_sink=sink)
    assert row["estado"] == "OK"
    # position tiers: a one-sided via letter caps at manzana (was esquina), radius 135 m
    assert row["nivel_precision"] == "manzana"
    assert row["position_tier"] == "manzana" and row["position_radius_m"] == 56
    assert row["motivo"].startswith("aproximado: letra de via")
    assert "nivel maximo manzana: via con letra en un solo lado" in row["motivo"]
    # the calibrated reliability model keeps its legacy input (nivel == manzana)
    assert sink[0]["relaxed_letter"] == 1.0 and sink[0]["nivel_manzana"] == 1.0 and sink[0]["nivel_esquina"] == 0.0


def test_one_sided_via_letter_still_counts_as_one_soft_violation(stub_normalizer):
    row = _row(stub_normalizer, "KR 987 # 654 - 19", "KR 987 H # 654 - 21")
    assert row["estado"] == "SIN_MATCH" and row["numero_predial_nacional"] is None


def test_one_sided_via_letter_with_max_soft_zero_is_rejected(stub_normalizer):
    row = _row(stub_normalizer, "KR 987 # 654 - 21", "KR 987 H # 654 - 21", max_soft=0)
    assert row["estado"] == "SIN_MATCH" and "letra de via" in row["motivo"]


def test_one_sided_via_letter_shared_across_manzanas_keeps_the_lowest_level(stub_normalizer):
    row = _row(stub_normalizer, "KR 987 # 654 - 21", "KR 987 H # 654 - 21",
               doc_extra={"n_predios": 3, "manzana_nunique": 2})
    assert row["nivel_precision"] == "direccion"


def test_two_sided_via_letter_difference_stays_rejected_by_default(stub_normalizer):
    row = _row(stub_normalizer, "KR 987 F # 654 - 21", "KR 987 H # 654 - 21")
    assert row["estado"] == "SIN_MATCH"


def test_cardinal_letter_equivalent_to_the_cadastral_quadrant_is_not_one_sided(stub_normalizer):
    row = _row(stub_normalizer, "CL 987 N # 654 - 21", "CL 987 NORTE # 654 - 21")
    assert row["estado"] == "OK" and row["nivel_precision"] == "predio" and row["motivo"] == ""


# ---------------------------------------------------------------------------
# 3. One-sided quadrant: allowed, but never labelled predio
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral, note", [
    ("CL 987 NORTE # 654 - 21", "CL 987 # 654 - 21", "cuadrante de via NORTE != -"),
    ("CL 987 # 654 - 21", "CL 987 NORTE # 654 - 21", "cuadrante de via - != NORTE"),
    ("CL 987 # 654 NORTE - 21", "CL 987 # 654 - 21", "cuadrante de cruce NORTE != -"),
    ("CL 987 # 654 - 21", "CL 987 # 654 OESTE - 21", "cuadrante de cruce - != OESTE"),
])
def test_one_sided_quadrant_is_ok_but_capped_at_manzana(stub_normalizer, query, cadastral, note):
    sink: list = []
    row = _row(stub_normalizer, query, cadastral, feature_sink=sink)
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "manzana"
    assert row["motivo"] == f"aproximado: {note}; nivel maximo manzana: cuadrante en un solo lado"
    assert row["numero_predial_nacional"] is None  # one-sided quadrant: predial never kept (was P1)
    assert row["position_tier"] == "manzana" and row["position_radius_m"] == 56  # unchanged level, radius added
    # reliability inputs are untouched: it is not a soft rule and not "relaxed"
    assert sink[0]["relaxed_letter"] == 0.0 and sink[0]["nivel_manzana"] == 0.0


def test_quadrant_moved_between_via_and_cross_is_rejected(stub_normalizer):
    # position is significant ('CL 5 N # 10 - 20' != 'CL 5 # 10 N - 20'): this used to be pinned as predio
    row = _row(stub_normalizer, "CL 987 NORTE # 654 - 21", "CL 987 # 654 NORTE - 21")
    assert row["estado"] == "SIN_MATCH" and row["motivo"] == "base distinta al catastro: quadrant_moved"
    assert bool(row["override_rejected"]) and row["position_tier"] == "none"
    assert row["numero_predial_nacional"] is None


def test_quadrant_typed_in_the_complement_is_a_moved_quadrant(stub_normalizer):
    # a trailing quadrant is its own position: against a via quadrant it is rejected (was pinned as predio)
    row = _row(stub_normalizer, "CL 987 # 654 - 21 NORTE", "CL 987 NORTE # 654 - 21")
    assert row["estado"] == "SIN_MATCH" and row["motivo"] == "base distinta al catastro: quadrant_moved"
    assert bool(row["override_rejected"])


def test_same_quadrant_on_both_sides_is_clean(stub_normalizer):
    row = _row(stub_normalizer, "CL 987 NORTE # 654 - 21", "CL 987 NORTE # 654 - 21")
    assert row["nivel_precision"] == "predio" and row["motivo"] == ""


def test_different_quadrants_are_still_a_hard_rejection(stub_normalizer):
    row = _row(stub_normalizer, "CL 987 NORTE # 654 - 21", "CL 987 SUR # 654 - 21")
    assert row["estado"] == "SIN_MATCH" and "cuadrante" in row["motivo"]


def test_one_sided_quadrant_does_not_use_the_soft_budget(stub_normalizer):
    # quadrant (advisory) + plate off by one (soft) with the default max_soft=1 stays OK
    row = _row(stub_normalizer, "CL 987 NORTE # 654 - 19", "CL 987 # 654 - 20")
    assert row["estado"] == "OK" and row["nivel_precision"] == "manzana"
    assert "placa 19 != 20" in row["motivo"] and "cuadrante de via NORTE != -" in row["motivo"]
    assert row["motivo"].count("aproximado:") == 1
    assert row["position_tier"] == "manzana" and row["position_radius_m"] == 56  # quadrant cap beats the tramo
    row0 = _row(stub_normalizer, "CL 987 NORTE # 654 - 21", "CL 987 # 654 - 21", max_soft=0)
    assert row0["estado"] == "OK" and row0["nivel_precision"] == "manzana"


# ---------------------------------------------------------------------------
# 4. Plate +-1/+-2: manzana level and an explicit motivo, never predio
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("delta", [-2, -1, 1, 2])
@pytest.mark.parametrize("extra", [{}, {"n_predios": 2}, {"n_predios": 4, "manzana_nunique": 2}])
def test_plate_soft_match_is_never_predio_and_says_so(stub_normalizer, delta, extra):
    # neighbours on both sides of the street: without them the plate would be an extrapolation (tier via)
    row = _row(stub_normalizer, f"CL 987 # 654 - {40 + delta}", "CL 987 # 654 - 40", doc_extra=extra,
               neighbours=(36, 44, 37, 43))
    assert row["estado"] == "OK"
    assert row["nivel_precision"] in ("manzana", "direccion")
    assert row["motivo"].startswith(f"aproximado: placa {40 + delta} != 40")


@pytest.mark.parametrize("delta", [-3, 3, 10])
def test_plate_beyond_the_soft_window_is_rejected(stub_normalizer, delta):
    row = _row(stub_normalizer, f"CL 987 # 654 - {40 + delta}", "CL 987 # 654 - 40")
    assert row["estado"] == "SIN_MATCH" and row["numero_predial_nacional"] is None


@pytest.mark.parametrize("tolerance", [0, 1])
def test_plate_delta_two_is_manzana_for_every_tolerance_below_it(stub_normalizer, tolerance):
    # neighbours added: the plate sits between 40 and 44 -> tramo (interpolated), still nivel manzana
    row = _row(stub_normalizer, "CL 987 # 654 - 42", "CL 987 # 654 - 40", plate_tolerance=tolerance,
               neighbours=(36, 44))
    assert row["position_tier"] == "tramo" and row["lat"] == pytest.approx(3.40005)
    assert row["nivel_precision"] == "manzana" and "placa 42 != 40" in row["motivo"]


# ---------------------------------------------------------------------------
# 5. Tower / block / stage after a comma or semicolon, `TORRE O`, `T B`
# ---------------------------------------------------------------------------
from cali_address.parser import extract_complement_units, merge_complement, render_complement  # noqa: E402


def _units(raw):
    p = parse_address(raw)
    assert p.parse_ok, raw
    return extract_complement_units(p.complement_text)


@pytest.mark.parametrize("raw, expected", [
    ("Calle 987 # 654-21, Torre B", [("TO", "B")]),
    ("Calle 987 # 654-21; Torre B", [("TO", "B")]),
    ("Calle 987 # 654-21, Bloque 3", [("BLQ", "3")]),
    ("Calle 987 # 654-21, Etapa 2", [("ET", "2")]),
    ("Calle 987 # 654-21, Interior 4", [("INT", "4")]),
    ("Calle 987 # 654-21, Torre B, Apto 402", [("TO", "B"), ("AP", "402")]),
    ("Calle 987 # 654-21, Barrio Las Flores, Torre B", [("TO", "B")]),
    ("Calle 987 # 654-21, Torre B, Cali, Valle del Cauca", [("TO", "B")]),
    ("Calle 987 # 654-21, Bloque 3; Etapa 2", [("BLQ", "3"), ("ET", "2")]),
    ("Calle 987 # 654-21 Torre A, Torre B", [("TO", "A"), ("TO", "B")]),
    ("CALLE 987 # 654-21, TORRE HUDSON", [("TO", "HUDSON")]),
])
def test_unit_after_comma_or_semicolon_is_kept_as_complement(raw, expected):
    assert _units(raw) == expected


def test_trailing_unit_never_leaks_into_the_plate_or_the_scoring_complement():
    plain = parse_address("Calle 987 # 654-21")
    tail = parse_address("Calle 987 # 654-21, Torre B")
    assert (tail.via_number, tail.cross_number, tail.plate) == ("987", "654", "21")
    # the encoder query text (frozen scoring complement) must not move
    assert tail.complement == plain.complement
    assert tail.canonical(with_complement=True) == plain.canonical(with_complement=True)
    assert "TORRE B" in tail.complement_text


@pytest.mark.parametrize("raw", [
    "Calle 987 # 654-21, Barrio Las Flores",
    "Calle 987 # 654-21, San Fernando",
    "Calle 987 # 654-21, Cali, Valle del Cauca",
    "Calle 987 # 654-21, Vereda El Hormiguero",
    "Calle 987 # 654-21, frente al parque",
    "Calle 987 # 654-21, Torre",                      # kind without a value
    "Calle 987 # 654-21;",
    "Calle 987 # 654-21,",
    "Calle 987 # 654-21, ,, ;",
    "Calle 987 # 654-21; Carrera 11 # 22-33 Torre B",  # a second address, not a unit of this one
])
def test_city_barrio_and_free_text_tails_stay_ignored(raw):
    assert _units(raw) == []


def test_ignored_tail_still_reports_the_dropped_segment():
    assert "dropped_trailing_segment" in parse_address("Calle 987 # 654-21, San Fernando").notes
    assert "complement_from_trailing_segment" in parse_address("Calle 987 # 654-21, Torre B").notes


def test_a_comma_head_that_is_not_an_address_keeps_the_old_behaviour():
    # a unit written BEFORE the address is still unparseable (out of scope of the audit fix)
    assert parse_address("Torre B, Calle 987 # 654-21").parse_ok is False


@pytest.mark.parametrize("raw, expected", [
    ("Calle 987 # 654-21 Torre O", [("TO", "O")]),
    ("Calle 987 # 654-21, Torre O", [("TO", "O")]),
    ("Calle 987 # 654-21 T B", [("TO", "B")]),
    ("Calle 987 # 654-21, T B", [("TO", "B")]),
    ("Calle 987 # 654-21 T 2", [("TO", "2")]),
    ("Calle 987 # 654-21 T-B", [("TO", "B")]),
])
def test_tower_aliases(raw, expected):
    assert _units(raw) == expected


@pytest.mark.parametrize("raw", [
    "Calle 987 # 654-21 Apto 101 T B",   # a letter after a number is a unit letter, not a tower
    "Calle 987 # 654-21 T",              # bare T without a value
    "Calle 987 # 654-21 T HUDSON",       # T is not a tower kind before a word
    "Calle 987 # 654-21 O",              # a stray O is not a tower
    "Calle 987 # 654-21 Apto O",         # ``O`` is a connector unless it follows a tower kind
])
def test_tower_alias_guards(raw):
    assert ("TO", "B") not in _units(raw) and ("TO", "O") not in _units(raw) and ("TO", "HUDSON") not in _units(raw)


def test_via_type_t_still_means_transversal():
    p = parse_address("T 987 # 654-21")
    assert p.via_type == "TV" and p.complement_text is None


def test_trailing_unit_reaches_the_rendered_complement_and_the_merge():
    p = parse_address("Calle 987 # 654-21, Torre B")
    assert render_complement(p.complement_text) == "TO B"
    merged = merge_complement("CL 987 # 654 - 21", p.complement_text)
    assert merged.address == "CL 987 # 654 - 21 TO B" and merged.conflicts == ()


def test_trailing_unit_survives_normalize_strict_end_to_end(stub_normalizer):
    row = _row(stub_normalizer, "Calle 987 # 654-21, Torre B", "CL 987 # 654 - 21")
    assert row["estado"] == "OK"
    assert row["direccion_normalizada"] == "CL 987 # 654 - 21 TO B"
    assert row["complemento_normalizado"] == "TO B"


def test_trailing_unit_is_kept_on_a_non_ok_row_too(stub_normalizer):
    row = _row(stub_normalizer, "Calle 987 # 654-21, Torre B", "CL 123 # 654 - 21")
    assert row["estado"] == "SIN_MATCH" and row["complemento_normalizado"] == "TO B"


@pytest.mark.parametrize("raw", ["", "  ", ",", ";", ",;", "Torre B", ", Torre B", "; Torre B"])
def test_degenerate_inputs_do_not_raise(raw):
    # concrete outputs (notes per input) live in test_auditoria_2026_10_r2.py
    p = parse_address(raw)
    assert p.parse_ok is False and p.complement_text is None and p.via_type is None


def test_trailing_unit_parsing_is_deterministic():
    raw = "Calle 987 # 654-21, Torre B; Apto 402, Etapa 2"
    assert parse_address(raw).to_dict() == parse_address(raw).to_dict()


# ---------------------------------------------------------------------------
# 7. Null and anomalous NPN in the cadastre
# ---------------------------------------------------------------------------
NPN_OK = "760010100000000010001000000000"  # synthetic, 30 digits
assert len(NPN_OK) == 30


@pytest.mark.parametrize("null_value", [None, float("nan"), "None", "none", "nan", "NaN", "<NA>", "", "  "])
def test_null_npn_is_returned_as_none_not_as_the_text_none(stub_normalizer, null_value):
    row = _row(stub_normalizer, "CL 987 # 654 - 21", "CL 987 # 654 - 21", doc_extra={"predial": null_value})
    assert row["estado"] == "OK"
    assert row["numero_predial_nacional"] is None
    assert "NPN nulo en catastro" in row["motivo"]
    assert row["nivel_precision"] == "direccion"
    assert row["lat"] == 3.4 and row["lon"] == -76.5  # the point is still returned


@pytest.mark.parametrize("null_value", [None, float("nan"), "None", ""])
def test_null_manzana_is_returned_as_none_not_as_the_text_none(stub_normalizer, null_value):
    row = _row(stub_normalizer, "CL 987 # 654 - 21", "CL 987 # 654 - 21",
               doc_extra={"predial": NPN_OK, "manzana": null_value})
    assert row["manzana"] is None
    assert row["numero_predial_nacional"] == NPN_OK


@pytest.mark.parametrize("npn, why", [
    (NPN_OK[:-1], "29 caracteres"),
    (NPN_OK[:-2], "28 caracteres"),
    (NPN_OK + "*", "caracteres no numericos"),
    (NPN_OK[:-1] + "*", "caracteres no numericos"),
    ("7600101000000000100010000000" + "0*", "caracteres no numericos"),
    (NPN_OK + "0", "31 caracteres"),
])
def test_anomalous_npn_is_flagged_but_not_dropped(stub_normalizer, npn, why):
    row = _row(stub_normalizer, "CL 987 # 654 - 21", "CL 987 # 654 - 21", doc_extra={"predial": npn})
    assert row["estado"] == "OK"
    assert row["numero_predial_nacional"] == npn
    assert f"NPN anomalo en catastro ({why})" in row["motivo"]


@pytest.mark.parametrize("npn", [NPN_OK, " " + NPN_OK + " ", "P1", "PXYZ", "ABC"])
def test_well_formed_or_non_numeric_placeholder_npn_is_not_flagged(stub_normalizer, npn):
    row = _row(stub_normalizer, "CL 987 # 654 - 21", "CL 987 # 654 - 21", doc_extra={"predial": npn})
    assert row["motivo"] == "" and row["nivel_precision"] == "predio"
    assert row["numero_predial_nacional"] == npn.strip()


def test_null_npn_is_none_in_json_and_geojson_helpers(stub_normalizer):
    from cali_address.service import results_to_geojson, results_to_records
    stub = stub_normalizer([{"direccion": "CL 987 # 654 - 21", "predial": None}], scores=[[0.95]], threshold=0.5)
    df = normalize_strict(stub, ["CL 987 # 654 - 21"], gazetteer=None, ambiguity_delta=0.0)
    assert results_to_records(df)[0]["numero_predial_nacional"] is None
    assert results_to_geojson(df)["features"][0]["properties"]["numero_predial_nacional"] is None


# ---------------------------------------------------------------------------
# 6. Shared addresses: the NPN is the first parcel's, so is the point
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _shared_row(stub_normalizer, own=None, n_predios=3, query="CL 987 # 654 - 21"):
    doc = {"direccion": "CL 987 # 654 - 21", "predial": NPN_OK, "manzana": "M1", "n_predios": n_predios,
           "lat": 3.40, "lon": -76.50}
    if own not in (None, "unavailable"):
        doc["own_lon"], doc["own_lat"] = own
    # "unavailable": a real normalizer without the sidecar / catastro.parquet (the attributes exist, empty)
    stub = stub_normalizer([doc], scores=[[0.95]], threshold=0.5, own_points=own != "unavailable")
    return normalize_strict(stub, [query], gazetteer=None, ambiguity_delta=0.0).iloc[0]


def test_shared_address_returns_the_point_of_the_returned_parcel(stub_normalizer):
    row = _shared_row(stub_normalizer, own=(-76.4912, 3.4123))
    assert (row["lon"], row["lat"]) == (-76.4912, 3.4123)
    assert row["numero_predial_nacional"] == NPN_OK
    assert "direccion compartida por 3 predios" in row["motivo"]
    assert "centroide de la direccion" not in row["motivo"]


def test_without_a_parcel_point_the_address_centroid_is_kept_and_labelled(stub_normalizer):
    for own in ("unavailable", (float("nan"), float("nan"))):
        row = _shared_row(stub_normalizer, own=own)
        assert (row["lon"], row["lat"]) == (-76.50, 3.40)
        assert "coordenada: centroide de la direccion compartida por 3 predios" in row["motivo"]


def test_single_parcel_address_has_no_centroid_note(stub_normalizer):
    for own in ("unavailable", (-76.5001, 3.4001)):
        row = _shared_row(stub_normalizer, own=own, n_predios=1)
        assert "centroide" not in row["motivo"]
    assert (row["lon"], row["lat"]) == (-76.5001, 3.4001)


def test_a_half_missing_parcel_point_falls_back_to_the_address_centroid(stub_normalizer):
    row = _shared_row(stub_normalizer, own=(-76.4912, float("nan")))
    assert (row["lon"], row["lat"]) == (-76.50, 3.40)


def _catastro_files(tmp_path, npns=("A1", "A2", "A3"), objectids=(10, 11, 12)):
    docs = pd.DataFrame({
        "direccion": ["CL 981 # 982 - 33", "CL 981 # 982 - 35", "CL 981 # 982 - 37"],
        "numero_predial_nacional": list(npns),
        "objectid": list(objectids),
        "centroid_lon": [-76.5, -76.51, -76.52], "centroid_lat": [3.4, 3.41, 3.42],
    })
    cat = pd.DataFrame({
        "OBJECTID": [10, 11, 12, 13], "numero_predial_nacional": ["A1", "A2", "A3", "A4"],
        "centroid_lon": [-76.501, -76.511, np.nan, -76.0], "centroid_lat": [3.401, 3.411, np.nan, 3.0],
    })
    cat.to_parquet(tmp_path / "catastro.parquet")
    return docs


def test_load_own_centroids_maps_objectid_to_the_parcel_point(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _catastro_files(tmp_path)
    lon, lat = load_own_centroids(docs, str(tmp_path))
    assert lon[:2].tolist() == [-76.501, -76.511] and lat[:2].tolist() == [3.401, 3.411]
    assert np.isnan(lon[2]) and np.isnan(lat[2])  # parcel without a geometry -> NaN, not a wrong point


def test_load_own_centroids_rejects_a_catastro_that_is_not_the_docs_source(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _catastro_files(tmp_path, npns=("X1", "X2", "X3"))
    assert load_own_centroids(docs, str(tmp_path)) is None  # same OBJECTIDs, other NPNs: never use it


def test_load_own_centroids_is_none_without_the_objectid_column(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _catastro_files(tmp_path).drop(columns=["objectid"])
    assert load_own_centroids(docs, str(tmp_path)) is None


def test_load_own_centroids_survives_a_malformed_file(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _catastro_files(tmp_path)
    (tmp_path / "catastro.parquet").write_bytes(b"not a parquet file")
    assert load_own_centroids(docs, str(tmp_path)) is None
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "catastro.parquet")
    assert load_own_centroids(docs, str(tmp_path)) is None


def test_load_own_centroids_on_empty_docs(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _catastro_files(tmp_path).iloc[0:0]
    assert load_own_centroids(docs, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# 8. Results must not depend on the batch size (bf16 autocast / fp16 products)
# ---------------------------------------------------------------------------

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tiny import tiny_config, tiny_data_dir  # noqa: E402

from cali_address.inference import AddressNormalizer  # noqa: E402
from cali_address.model import encode_batch  # noqa: E402
from cali_address.paths import DEFAULT_ARTIFACTS_DIR  # noqa: E402
from cali_address.train import embed_documents, save_checkpoint, train_model  # noqa: E402

_SAMPLE = [
    "CL 1 # 10 - 10", "calle 2 No 10-12", "CL 3 # 11 - 14", "Cl 4 #11 16", "CL 5 # 10 - 18",
    "CL 6 # 11 - 20", "calle 7 # 10 - 99", "KR 8 # 3 - 3", "CL 2 # 10 - 14", "xyz",
]


@pytest.fixture(scope="module")
def tiny_normalizer(tmp_path_factory):
    base = tmp_path_factory.mktemp("tiny_norm")
    docs, data, neg = tiny_data_dir(str(base))
    out = train_model(data, neg, tiny_config(), device="cpu")
    save_checkpoint(str(base / "model.pt"), out["model"], out["config"], data)
    torch.save(embed_documents(out["model"], data["d_tokens"][:, :40], device="cpu").cpu(), str(base / "catastro_emb.pt"))
    norm = AddressNormalizer(str(base), device="cpu")
    # An untrained tiny model scores every pair negative (clipped to 0, which would hide
    # any precision problem): negating the (fp16-exact) document matrix makes the sims positive.
    norm.doc_emb = -norm.doc_emb
    return norm


def _fp32_reference(norm, texts):
    """Exact (fp32, no autocast) similarity of every query to every document."""
    ids = torch.from_numpy(encode_batch(texts, norm.max_len).astype(np.int64))
    with torch.no_grad():
        emb = F.normalize(norm.model.embed(ids, role=0).float(), dim=-1)
        return (emb @ norm.doc_emb.float().T).numpy()


def test_candidate_similarity_is_the_exact_fp32_product(tiny_normalizer):
    from cali_address.parser import _pre_normalize, canonical, parse_address
    norm = tiny_normalizer
    comps = norm.score_components(_SAMPLE, k=5)
    parsed = [parse_address(t) for t in _SAMPLE]
    canon = [canonical(p, style="spaced", with_complement=True) if p.parse_ok else "" for p in parsed]
    prenorm = [_pre_normalize(t)[: norm.max_len] for t in _SAMPLE]
    ref = np.maximum(_fp32_reference(norm, [c or p for c, p in zip(canon, prenorm)]), 0.0)
    ref2 = np.maximum(_fp32_reference(norm, prenorm), 0.0)
    for r in range(len(_SAMPLE)):
        for slot, doc in enumerate(comps["doc"][r]):
            if doc < 0:
                continue
            got = float(comps["sim"][r, slot])
            assert min(abs(got - ref[r, doc]), abs(got - ref2[r, doc])) < 2e-6, (r, doc, got)


def test_scores_do_not_depend_on_the_batch_composition(tiny_normalizer):
    norm = tiny_normalizer
    whole = norm.score_components(_SAMPLE, k=5)
    for size in (1, 2, 3, 7):
        for start in range(0, len(_SAMPLE), size):
            part = norm.score_components(_SAMPLE[start:start + size], k=5)
            for i in range(part["doc"].shape[0]):
                a = dict(zip(whole["doc"][start + i].tolist(), zip(whole["sim"][start + i], whole["geo"][start + i])))
                b = dict(zip(part["doc"][i].tolist(), zip(part["sim"][i], part["geo"][i])))
                assert set(a) == set(b)
                for doc, (sim, geo) in b.items():
                    assert abs(sim - a[doc][0]) < 1e-6 and abs(geo - a[doc][1]) < 1e-6


def test_empty_and_degenerate_batches_do_not_raise(tiny_normalizer):
    comps = tiny_normalizer.score_components([], k=5)
    assert comps["doc"].shape[0] == 0
    comps = tiny_normalizer.score_components(["", "   ", None, "!!!"], k=5)
    assert comps["doc"].shape[0] == 4


def test_scoring_is_deterministic(tiny_normalizer):
    a = tiny_normalizer.score_components(_SAMPLE, k=5)
    b = tiny_normalizer.score_components(_SAMPLE, k=5)
    for key in ("doc", "sim", "fuzz", "struct", "geo"):
        assert np.array_equal(a[key], b[key])


@pytest.mark.requires_artifacts
def test_real_artifacts_status_and_scores_are_batch_size_independent():
    """The audit saw status flips between one batch and batches of 64 on GPU."""
    from cali_address.service import normalize_strict
    norm = AddressNormalizer(DEFAULT_ARTIFACTS_DIR)
    sample = [f"CL {1 + i % 60} # {2 + i % 50} - {10 + (i * 7) % 90}" for i in range(160)]
    sample += [f"carrera {3 + i % 40} No {1 + i % 30}-{5 + (i * 3) % 80}" for i in range(80)]
    one = normalize_strict(norm, sample, gazetteer=None)
    chunked = pd.concat(
        [normalize_strict(norm, sample[i:i + 64], gazetteer=None) for i in range(0, len(sample), 64)],
        ignore_index=True,
    )
    assert (one["estado"] == chunked["estado"]).all()
    assert one["numero_predial_nacional"].fillna("").tolist() == chunked["numero_predial_nacional"].fillna("").tolist()
    both = one["confianza"].notna() & chunked["confianza"].notna()
    assert float((one["confianza"][both] - chunked["confianza"][both]).abs().max()) < 2e-4


# ---------------------------------------------------------------------------
# 10. One set of gate defaults: library == CLI == API == pipeline
# ---------------------------------------------------------------------------
import inspect  # noqa: E402

from cali_address import service  # noqa: E402


class _Place:
    barrio, comuna, corregimiento, vereda = "Zzz", 7, None, None
    cleaned_text = "CL 987 # 654 - 21"
    notes: list = []
    detected = True


class _DistanceGazetteer:
    """The candidate sits 600 m outside the barrio and inside the comuna."""

    def detect(self, raw):
        return _Place()

    def lookup(self, kind, name):
        return object()

    def comuna_entry(self, value):
        return object()

    def contains(self, kind, name, lon, lat, buffer_m=0.0):
        return True if kind == "comuna" else buffer_m >= 600.0

    def locate(self, lon, lat):
        return {}


def _gate_row(stub_normalizer, **kw):
    stub = stub_normalizer([{"direccion": "CL 987 # 654 - 21"}], scores=[[0.95]], threshold=0.5)
    return normalize_strict(stub, ["CL 987 # 654 - 21"], gazetteer=_DistanceGazetteer(), ambiguity_delta=0.0, **kw).iloc[0]


def test_library_gate_defaults_are_the_shared_constants():
    sig = inspect.signature(normalize_strict).parameters
    assert service.CLI_BARRIO_BUFFER_M == 1000.0 and service.DEFAULT_GATE_ESCALATE is True
    assert sig["barrio_buffer_m"].default == service.CLI_BARRIO_BUFFER_M
    assert sig["gate_escalate"].default is service.DEFAULT_GATE_ESCALATE


def test_cli_api_and_pipeline_use_the_same_constants():
    from cali_address.api.schemas import Tunables as ApiTunables
    from cali_address.cli import _project_defaults
    from cali_address.io.pipeline import Tunables
    sig = inspect.signature(normalize_strict).parameters
    cli = _project_defaults()
    assert cli["barrio_buffer"] == sig["barrio_buffer_m"].default and cli["gate_escalate"] is sig["gate_escalate"].default
    tun = Tunables().as_kwargs()
    assert tun["barrio_buffer_m"] == sig["barrio_buffer_m"].default and tun["gate_escalate"] is sig["gate_escalate"].default
    fields = ApiTunables.model_fields
    assert fields["barrio_buffer"].default == sig["barrio_buffer_m"].default
    assert fields["gate_escalate"].default is sig["gate_escalate"].default


def test_default_call_keeps_a_candidate_just_outside_the_barrio(stub_normalizer):
    row = _gate_row(stub_normalizer)  # no gate arguments at all: library defaults
    assert row["estado"] == "OK"


def test_explicit_strict_gate_still_withholds_it(stub_normalizer):
    row = _gate_row(stub_normalizer, barrio_buffer_m=200.0, gate_escalate=False)
    assert row["estado"] == "SIN_MATCH" and row["motivo"].startswith("candidatos fuera de barrio Zzz")


def test_default_escalates_to_the_comuna_when_the_barrio_is_empty(stub_normalizer):
    row = _gate_row(stub_normalizer, barrio_buffer_m=0.0)
    assert row["estado"] == "OK"  # barrio tier empty -> comuna tier keeps it
    strict = _gate_row(stub_normalizer, barrio_buffer_m=0.0, gate_escalate=False)
    assert strict["estado"] == "SIN_MATCH"
