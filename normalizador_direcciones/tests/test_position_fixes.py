"""Fix round after the adversarial review: C1 interpolation caps, C2 blank plate/cross, C3 output text,
W1 one-sided tokens, W2 unit markers, W5 extras. Synthetic addresses only.

The face ``CL 987 # 654`` uses ``lat = 3.4 + plate * 1e-4`` (about 11.1 m per plate unit).
"""

from __future__ import annotations


import pandas as pd
import pytest

from conftest import make_stub
from cali_address import interp, service
from cali_address.inference import base_mismatches
from cali_address.interp import build_index, street_key
from cali_address.parser import parse_address
from cali_address.service import normalize_strict

LON = -76.5
M_PER_PLATE = 11.12


def lat_of(plate):
    return round(3.4 + plate * 1e-4, 6)


def doc(plate, street="CL 987 # 654", **extra):
    d = {"direccion": f"{street} - {plate}", "lat": lat_of(plate), "lon": LON, "predial": f"P{plate}",
         "manzana": "MZ7"}
    d.update(extra)
    return d


def run(query, docs, **kw):
    return normalize_strict(make_stub(docs), [query], gazetteer=None, **kw).iloc[0]


def g(rule, cad, **kw):
    return base_mismatches(parse_address(rule), parse_address(cad), mode="position", **kw)


KEY = street_key(parse_address("CL 987 # 654 - 10"))


def idx(plates, lat_step=1e-4):
    rows = [(f"CL 987 # 654 - {p}", 3.4 + p * lat_step, LON) for p in plates]
    return build_index([r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows])


# ---------------------------------------------------------------------------
# C1: interpolation caps
# ---------------------------------------------------------------------------
def test_cap_constants_exist_with_the_derived_values():
    # round 5: derived from the cadastre (largest bin edge whose p90 interpolation error stays <= 30 m)
    assert interp.INTERP_MAX_GAP_M == 50.0 and interp.INTERP_MAX_PLATE_GAP == 60


def test_plate_gap_boundaries(monkeypatch):
    monkeypatch.setattr(interp, "INTERP_MAX_GAP_M", 1e9)
    for gap, kind in [(58, "interp"), (60, "interp"), (62, "gap")]:  # same-parity plates: even gaps
        got = idx([10, 10 + gap]).locate(KEY, 12)
        assert got.kind == kind, gap
        assert got.lat == pytest.approx(3.4 + 12e-4)  # coordinates are still reported (flag decides)


def test_metre_gap_boundaries(monkeypatch):
    index = idx([10, 20])
    gap_m = interp._metres(3.4 + 10e-4, LON, 3.4 + 20e-4, LON)
    monkeypatch.setattr(interp, "INTERP_MAX_GAP_M", gap_m)
    assert index.locate(KEY, 14).kind == "interp"          # exactly at the cap
    monkeypatch.setattr(interp, "INTERP_MAX_GAP_M", gap_m + 1e-6)
    assert index.locate(KEY, 14).kind == "interp"          # just below
    monkeypatch.setattr(interp, "INTERP_MAX_GAP_M", gap_m - 1e-6)
    assert index.locate(KEY, 14).kind == "gap"             # just above


def test_consecutive_plates_twelve_kilometres_apart():
    far = idx([10, 14], lat_step=0.027)  # ~3 km per plate unit -> ~12 km between 10 and 14
    assert far.locate(KEY, 12).kind == "gap"
    assert far.locate(KEY, 10).kind == "interp"  # a known plate is always exact


def test_four_plate_gap_beyond_500_m_is_not_interpolated():
    index = idx([10, 14], lat_step=5e-3)  # ~555 m per plate unit
    assert index.locate(KEY, 12).kind == "gap"


def test_equal_plate_and_known_plates_never_gap():
    index = idx([10, 10, 20])
    assert index.locate(KEY, 10).kind == "interp"
    assert index.locate(KEY, 20).kind == "interp"


def test_extrapolation_displacement_is_capped_in_metres(monkeypatch):
    index = idx([10, 14, 20], lat_step=2.5e-5)  # ~2.8 m per plate unit
    near = index.locate(KEY, 22)  # 2 plates ~ 5.6 m
    assert near.kind == "extrap" and near.lat is not None
    far = index.locate(KEY, 20 + 30)  # 30 plates ~ 83 m > 25 m (INTERP_MAX_EXTRAP_M)
    assert far.kind == "extrap" and far.lat is None
    monkeypatch.setattr(interp, "INTERP_MAX_EXTRAP_M", 400.0)
    assert index.locate(KEY, 20 + 30).lat is not None


def test_extrapolation_with_a_far_end_pair_uses_the_end_point_not_the_slope():
    index = idx([10, 14], lat_step=5e-3)  # end pair is a gap
    got = index.locate(KEY, 16)
    assert got.kind == "extrap" and got.lat == pytest.approx(3.4 + 14 * 5e-3)  # the last parcel itself


def test_first_last_single_parcel_still_work():
    assert idx([20]).locate(KEY, 22).lat == pytest.approx(3.4 + 20e-4)
    assert idx([10, 20]).locate(KEY, 8).kind == "extrap"
    assert idx([10, 20]).locate(KEY, 22).kind == "extrap"


def test_service_gap_degrades_to_via_without_coordinates():
    docs = [doc(10), doc(60), doc(11), doc(15)]  # evens 10 and 60: 50 plates apart
    row = run("CL 987 # 654 - 12", docs)
    assert row["estado"] == "OK" and row["position_tier"] == "via" and row["position_radius_m"] == 1200  # gap > 150 m
    assert pd.isna(row["lat"]) and row["numero_predial_nacional"] is None and row["position_source"] == "none"
    assert row["nivel_precision"] == "via"


def test_service_gap_publishes_coordinates_only_with_the_flag(monkeypatch):
    monkeypatch.setattr(service, "EXTRAPOLATED_COORDS_ENABLED", True)
    row = run("CL 987 # 654 - 12", [doc(10), doc(60)])
    assert row["position_tier"] == "via" and row["position_source"] == "interp"
    assert row["numero_predial_nacional"] is None and not pd.isna(row["lat"])


def test_service_metre_gap_with_close_plates():
    far = [doc(10, lat=3.40), doc(14, lat=3.51)]  # 4 plates, 12 km
    row = run("CL 987 # 654 - 12", far)
    assert row["position_tier"] == "via" and pd.isna(row["lat"])


def test_service_close_neighbours_still_interpolate():
    row = run("CL 987 # 654 - 14", [doc(12), doc(16)])
    assert row["position_tier"] == "tramo" and row["lat"] == pytest.approx(lat_of(14))


# ---------------------------------------------------------------------------
# C2: blank plate / cross
# ---------------------------------------------------------------------------
def test_guard_rejects_a_plateless_parcel_when_the_query_has_a_plate():
    assert g("CL 987 # 654 - 20", "CL 987 # 654 -") == (["plate"], {})
    assert g("KR 987 B # 654 - 1", "K 987 B # 654 -")[0] == ["plate"]


def test_guard_plate_blank_on_the_query_only_or_both_is_not_hard():
    assert g("CL 987 # 654", "CL 987 # 654 - 20") == ([], {})
    assert g("CL 987 # 654", "CL 987 # 654 -") == ([], {})


@pytest.mark.parametrize("rule, cad", [
    ("CL 987 # - 20", "CL 987 # 654 - 20"), ("CL 987 # 654 - 20", "CL 987 # - 20"), ("CL 987 # - 20", "CL 987 # - 20"),
])
def test_guard_missing_cross_on_either_side_caps_at_via(rule, cad):
    hard, caps = g(rule, cad)
    assert hard == [] and caps == {"cross_number": "via"}


def test_service_plateless_parcel_is_rejected_with_the_rule_text():
    row = run("CL 987 # 654 - 1", [{"direccion": "CL 987 # 654 -", "lat": 3.4, "lon": LON, "predial": "PX"}])
    assert row["estado"] == "SIN_MATCH" and bool(row["override_rejected"]) and row["position_tier"] == "none"
    assert row["motivo"] == "base distinta al catastro: plate"
    assert row["direccion_normalizada"] == "CL 987 # 654 - 01" and row["numero_predial_nacional"] is None  # canonical pads the plate


@pytest.mark.parametrize("query, cad", [
    ("CL 987 # - 20", "CL 987 # 654 - 20"),   # cross missing on the query
    ("CL 987 # 654 - 20", "CL 987 # - 20"),   # cross missing on the cadastre
    ("CL 987 # - 20", "CL 987 # - 20"),       # both
])
def test_service_missing_cross_is_via_without_predial(query, cad):
    row = run(query, [{"direccion": cad, "lat": 3.4, "lon": LON, "predial": "PX", "manzana": "M1"}])
    assert row["estado"] == "OK"
    assert (row["position_tier"], row["position_radius_m"], row["nivel_precision"]) == ("via", 21, "via")
    assert row["numero_predial_nacional"] is None and pd.isna(row["lat"]) and pd.isna(row["manzana"])


# ---------------------------------------------------------------------------
# C3: direccion_normalizada
# ---------------------------------------------------------------------------
def test_new_matched_cadastral_column_is_appended_last():
    assert service.OUTPUT_COLUMNS[-5] == "direccion_catastral"  # the fidelity and catalogue columns follow
    assert service.OUTPUT_COLUMNS[-10:-5] == ["position_tier", "position_radius_m", "position_source",
                                             "selected_doc_id", "override_rejected"]


@pytest.mark.parametrize("plate", [13, 14])
def test_tramo_row_shows_the_query_text_and_exposes_the_matched_address(plate):
    docs = [doc(12), doc(16), doc(11), doc(15)]
    row = run(f"CL 987 # 654 - {plate}", docs)
    assert row["position_tier"] == "tramo"
    assert row["direccion_normalizada"] == f"CL 987 # 654 - {plate}"
    assert row["fuente_normalizacion"] == "reglas" and row["direccion_catastral"] == "CL 987 # 654 - 12"


def test_plate_plus_four_is_rejected_with_the_rule_text():
    row = run("CL 987 # 654 - 16", [doc(12), doc(20)])
    assert row["estado"] == "SIN_MATCH" and row["direccion_normalizada"] == "CL 987 # 654 - 16"
    assert row["fuente_normalizacion"] == "reglas" and pd.isna(row["direccion_catastral"])


def test_via_row_shows_the_query_text():
    row = run("CL 987 # 654 - 22", [doc(20), doc(16), doc(12)])
    assert row["position_tier"] == "via" and row["direccion_normalizada"] == "CL 987 # 654 - 22"
    assert row["fuente_normalizacion"] == "reglas" and row["direccion_catastral"] == "CL 987 # 654 - 20"


def test_predio_and_lote_keep_the_cadastral_base():
    p = run("CL 987 # 654 - 12 AP 301", [doc(12)])
    assert p["position_tier"] == "predio" and p["fuente_normalizacion"] == "catastro"
    assert p["direccion_normalizada"] == "CL 987 # 654 - 12 AP 301" and p["direccion_catastral"] == "CL 987 # 654 - 12"
    cad = {"direccion": "C 987 # 654 - 12 TO 2", "lat": 3.4, "lon": LON, "predial": "PX"}
    lote = run("CL 987 # 654 - 12 TO 3", [cad])
    assert lote["position_tier"] == "lote" and lote["fuente_normalizacion"] == "catastro"
    assert lote["direccion_normalizada"] == "CL 987 # 654 - 12 TO 3"


def test_output_text_is_idempotent_for_tramo_via_and_rejected_rows():
    for query, docs in [("CL 987 # 654 - 14", [doc(12), doc(16)]), ("CL 987 # 654 - 22", [doc(20), doc(16), doc(12)]),
                        ("CL 987 # 654 - 30", [doc(12)])]:
        first = run(query, docs)
        again = run(str(first["direccion_normalizada"]), docs)
        assert again["direccion_normalizada"] == first["direccion_normalizada"]
        assert again["position_tier"] == first["position_tier"]


# ---------------------------------------------------------------------------
# W1: one-sided tokens
# ---------------------------------------------------------------------------
def cad_doc(text, **kw):
    return {"direccion": text, "lat": 3.41, "lon": LON, "predial": "PX", "manzana": "M1", **kw}


@pytest.mark.parametrize("query, unique", [
    ("CL 987 # 654 - 12", "CL 987 NORTE # 654 - 12"),
    ("CL 987 # 654 - 12", "CL 987 # 654 NORTE - 12"),
    ("CL 987 # 654 - 12", "CL 987 # 654 - 12 NORTE"),
    ("CL 987 # 654 - 12", "CL 987 A # 654 - 12"),
    ("CL 987 A # 654 - 12", "CL 987 # 654 - 12"),
    ("CL 987 NORTE # 654 - 12", "CL 987 # 654 - 12"),
])
def test_unique_one_sided_token_is_manzana_without_predial(query, unique):
    row = run(query, [cad_doc(unique)])
    assert row["estado"] == "OK" and row["position_tier"] == "manzana" and row["position_radius_m"] == 56
    assert row["numero_predial_nacional"] is None and row["manzana"] == "M1" and row["lat"] == 3.41
    assert row["direccion_normalizada"] == service.canonical(parse_address(query), style="spaced")


@pytest.mark.parametrize("query, a, b", [
    ("CL 987 # 654 - 12", "CL 987 NORTE # 654 - 12", "CL 987 SUR # 654 - 12"),
    ("CL 987 # 654 - 12", "CL 987 A # 654 - 12", "CL 987 B # 654 - 12"),
    ("CL 987 # 654 - 12", "CL 987 # 654 NORTE - 12", "CL 987 # 654 OESTE - 12"),
])
def test_ambiguous_one_sided_token_is_via_without_coordinates(query, a, b):
    row = run(query, [cad_doc(a), cad_doc(b)])
    assert row["estado"] == "OK" and row["position_tier"] == "via" and row["position_radius_m"] == 21
    assert row["numero_predial_nacional"] is None and pd.isna(row["lat"]) and pd.isna(row["manzana"])
    assert row["nivel_precision"] == "via" and row["position_source"] == "none"


def test_a_candidate_that_differs_elsewhere_does_not_make_it_ambiguous():
    row = run("CL 987 # 654 - 12", [cad_doc("CL 987 NORTE # 654 - 12"), cad_doc("CL 987 SUR # 654 - 90")])
    assert row["position_tier"] == "manzana" and row["numero_predial_nacional"] is None


def test_n_against_a_prefixed_quadrant_is_a_one_sided_letter_not_a_predio():
    row = run("CL 987 N # 654 - 12", [cad_doc("CL 987 A NORTE # 654 - 12")])
    assert row["estado"] == "OK" and row["position_tier"] == "manzana" and row["numero_predial_nacional"] is None


def test_different_prefix_with_the_same_quadrant_is_a_hard_reject():
    row = run("CL 987 B N # 654 - 12", [cad_doc("CL 987 A NORTE # 654 - 12")])
    assert row["estado"] == "SIN_MATCH"


def test_guard_ntilde_is_not_a_north_quadrant():
    assert g("CLLE 987 N # 654 - 20", "CL 987 NORTE # 654 - 20") == ([], {})
    hard, caps = g("CLLE 987 Ñ # 654 - 20", "CL 987 NORTE # 654 - 20")
    assert hard == [] and caps  # not equivalent: capped, never clean
    row = run("CLLE 987 Ñ # 654 - 12", [cad_doc("CL 987 NORTE # 654 - 12")])
    assert row["numero_predial_nacional"] is None


# ---------------------------------------------------------------------------
# W2: unit markers are not plate letters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tail", ["P 1", "T 9", "G 2", "P 12", "PISO 2", "AP 301", "TORRE 9", "TO 9 AP 2"])
def test_unit_markers_are_not_plate_letters(tail):
    assert g(f"CL 987 # 654 - 20 {tail}", "CL 987 # 654 - 20") == ([], {})


@pytest.mark.parametrize("tail, field", [("P", "plate_letters"), ("T", "plate_letters"), ("G", "plate_letters"),
                                         ("A", "plate_letters"), ("BIS", "plate_bis")])
def test_a_separated_letter_without_a_digit_is_still_a_plate_letter(tail, field):
    assert g(f"CL 987 # 654 - 20 {tail}", "CL 987 # 654 - 20") == ([], {field: "tramo"})


@pytest.mark.parametrize("glued", ["20P", "20T", "20G", "20C"])
def test_glued_letters_stay_plate_letters(glued):
    assert g(f"CL 987 # 654 - {glued}", "CL 987 # 654 - 20") == ([], {"plate_letters": "tramo"})


def test_service_exact_address_with_a_unit_marker_stays_predio():
    for tail in ("P 1", "T 9", "G 2"):
        assert run(f"CL 987 # 654 - 12 {tail}", [doc(12)])["position_tier"] == "predio"


# ---------------------------------------------------------------------------
# W5: research vs production parity on hard vs soft
# ---------------------------------------------------------------------------
PARITY = [
    # rule, cadastre, production hard?, production caps?
    ("CL 987 # 654 - 20", "CL 987 # 654 - 20", False, False),
    ("CL 987 # 654 - 21", "CL 987 # 654 - 20", False, True),
    ("CL 987 # 654 - 25", "CL 987 # 654 - 20", True, False),
    ("CL 987 # 654 - 20", "CL 987 # 654 -", True, False),
    ("CL 987 N # 654 - 20", "CL 987 NORTE # 654 - 20", False, False),
    ("CL 987 N # 654 - 20", "CL 987 # 654 NORTE - 20", True, False),
    ("CL 987 # 654 - 20", "CL 987 NORTE # 654 - 20", False, True),
    ("CL 987 A # 654 - 20", "CL 987 B # 654 - 20", True, False),
    ("CL 987 A # 654 - 20", "CL 987 # 654 - 20", False, True),
    ("CL 987 # 654 - 20 TO 1", "CL 987 # 654 - 20 TO 2", False, True),
    ("CL 987 # 654 - 20 P 1", "CL 987 # 654 - 20", False, False),
    ("CL 987 # - 20", "CL 987 # 654 - 20", False, True),
]


@pytest.mark.parametrize("rule, cad, hard, caps", PARITY)
def test_research_and_production_guards_agree_on_hard_vs_soft(rule, cad, hard, caps):
    r, c = parse_address(rule), parse_address(cad)
    legacy = base_mismatches(r, c)
    p_hard, p_caps = base_mismatches(r, c, mode="position")
    assert bool(p_hard) is hard and bool(p_caps) is caps
    if hard:
        assert legacy  # research rejects whatever production rejects
    if not legacy:
        assert not p_hard and not p_caps  # and production adds nothing on a clean pair
    if rule.endswith("TO 1"):
        assert legacy  # units: the research guard names it too
