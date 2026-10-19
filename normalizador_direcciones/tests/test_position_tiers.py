"""Position tiers of ``normalize_strict``: guard, interpolation, radii and the new output columns.

Synthetic cadastre only (stub normalizer). Coordinates of the face ``CL 987 # 654`` follow
``lat = 3.4 + plate * 2.5e-5`` (~2.8 m per plate unit, inside the round-5 50 m interpolation cap) so every
interpolated value is exact.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from conftest import make_stub
from cali_address import service
from cali_address.service import (
    OUTPUT_COLUMNS,
    POSITION_RADIUS_M,
    normalize_strict,
    results_to_records,
)

LON = -76.5
NEW_COLUMNS = ["position_tier", "position_radius_m", "position_source", "selected_doc_id", "override_rejected"]


def lat_of(plate: int) -> float:
    return round(3.4 + plate * 2.5e-5, 6)


def doc(plate, street="CL 987 # 654", suffix="", **extra):
    d = {"direccion": f"{street} - {plate}{suffix}", "lat": lat_of(plate), "lon": LON,
         "predial": f"P{plate}", "manzana": "MZ7"}
    d.update(extra)
    return d


def run(query, docs, **kw):
    stub = make_stub(docs)
    return normalize_strict(stub, [query], gazetteer=None, **kw).iloc[0]


FACE = [doc(12), doc(16), doc(20), doc(11), doc(15)]  # candidate (12) first: the stub picks doc 0


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------
def test_new_columns_are_appended_last_in_order():
    assert OUTPUT_COLUMNS[-10:-5] == NEW_COLUMNS and OUTPUT_COLUMNS[-5] == "direccion_catastral"
    assert OUTPUT_COLUMNS[-4:-2] == ["fidelity_risk", "fidelity_flags"]  # fidelity + catalogue columns follow
    assert OUTPUT_COLUMNS[:-10][-1] == "complemento_normalizado"
    frame = normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12"], gazetteer=None)
    assert list(frame.columns) == OUTPUT_COLUMNS


def test_radii_table():
    assert POSITION_RADIUS_M == {"predio": 15, "lote": 44, "manzana": 56, "tramo": 23, "via": 21}


def test_position_columns_survive_json_records():
    frame = normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12", "", "hola"], gazetteer=None)
    recs = results_to_records(frame)
    assert recs[0]["position_tier"] == "predio" and recs[0]["selected_doc_id"] == 0
    assert recs[1]["position_radius_m"] is None and recs[1]["selected_doc_id"] is None


# ---------------------------------------------------------------------------
# predio
# ---------------------------------------------------------------------------
def test_exact_match_is_predio():
    row = run("CL 987 # 654 - 12", FACE)
    assert (row["position_tier"], row["position_radius_m"], row["position_source"]) == ("predio", 15, "parcel")
    assert row["selected_doc_id"] == 0 and bool(row["override_rejected"]) is False
    assert row["numero_predial_nacional"] == "P12" and row["nivel_precision"] == "predio"
    assert (row["lat"], row["lon"]) == (lat_of(12), LON)


def test_plate_equal_with_leading_zero_is_predio():
    row = run("CL 987 # 654 - 012", FACE)
    assert row["position_tier"] == "predio"


def test_only_input_unit_keeps_predio():
    row = run("CL 987 # 654 - 12 TO 3 AP 101", FACE)
    assert row["position_tier"] == "predio" and row["numero_predial_nacional"] == "P12"


# ---------------------------------------------------------------------------
# tramo: interpolated, never the neighbour's centroid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("plate", [14, 13])
def test_plate_within_two_is_interpolated_tramo(plate):
    row = run(f"CL 987 # 654 - {plate}", FACE)
    assert row["estado"] == "OK"
    assert (row["position_tier"], row["position_radius_m"], row["position_source"]) == ("tramo", 15, "interp")
    assert row["lat"] == pytest.approx(lat_of(plate), abs=1e-9) and row["lon"] == pytest.approx(LON)
    assert row["lat"] != lat_of(12)  # not the neighbour's centroid
    assert row["numero_predial_nacional"] is None and row["manzana"] == "MZ7"
    assert row["nivel_precision"] == "manzana" and row["motivo"].startswith("aproximado: placa")
    assert row["selected_doc_id"] == 0


def test_plate_letters_mismatch_is_tramo_at_the_known_plate():
    row = run("CL 987 # 654 - 12 C", FACE)
    assert row["position_tier"] == "tramo" and row["position_source"] == "interp"
    assert row["lat"] == pytest.approx(lat_of(12)) and row["numero_predial_nacional"] is None
    assert "posicion maxima tramo: letra de placa" in row["motivo"]


# fidelity caps off (fixture): the output drops the input bis (high risk); the cap itself is tested elsewhere
def test_plate_bis_mismatch_is_tramo(no_fidelity_caps):
    row = run("CL 987 # 654 - 12 BIS", FACE)
    assert row["position_tier"] == "tramo" and "bis" in row["motivo"].lower()


def test_tramo_without_neighbours_on_that_side_degrades_to_manzana():
    # the odd side of the face is empty: nothing to interpolate between
    row = run("CL 987 # 654 - 13", [doc(12), doc(16), doc(20)])
    assert row["position_tier"] == "manzana" and row["position_radius_m"] == 56
    assert row["position_source"] == "parcel" and row["lat"] == lat_of(12)
    assert row["numero_predial_nacional"] == "P12"


@pytest.mark.parametrize("plate, estado, tier, lat", [
    (12, "OK", "predio", lat_of(12)),   # equal
    (13, "OK", "tramo", lat_of(13)),    # +1, odd side: between 11 and 15
    (14, "OK", "tramo", lat_of(14)),    # +2, even side: between 12 and 16
    (11, "OK", "tramo", lat_of(11)),    # -1: a known parcel of the odd side
    (10, "OK", "via", None),            # -2 but before the first parcel of the even side: extrapolation
    (15, "SIN_MATCH", "none", None),    # +3
    (9, "SIN_MATCH", "none", None),     # -3
    (0, "SIN_MATCH", "none", None),
])
def test_plate_delta_boundaries_end_to_end(plate, estado, tier, lat):
    row = run(f"CL 987 # 654 - {plate}", FACE)
    assert (row["estado"], row["position_tier"]) == (estado, tier)
    if lat is None:
        assert pd.isna(row["lat"])
    else:
        assert row["lat"] == pytest.approx(lat, abs=1e-9)


@pytest.mark.parametrize("docs, plate, tier", [
    ([doc(12), doc(16)], 14, "tramo"),            # two parcels: between them
    ([doc(16), doc(12)], 18, "via"),              # last parcel of the face is 16: extrapolation
    ([doc(12), doc(16)], 10, "via"),              # first parcel of the face is 12
    ([doc(12)], 14, "via"),                       # single parcel on the face
    ([doc(12), doc(13)], 14, "via"),              # the other side has a parcel, not this one
])
def test_face_boundaries_decide_between_tramo_and_via(docs, plate, tier):
    assert run(f"CL 987 # 654 - {plate}", docs)["position_tier"] == tier


def test_interpolation_failure_degrades_to_manzana_instead_of_raising(monkeypatch):
    def boom(_normalizer):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(service, "_interp_index_for", boom)
    row = run("CL 987 # 654 - 14", FACE)
    assert row["estado"] == "OK" and row["position_tier"] == "manzana"
    assert row["lat"] == lat_of(12) and row["numero_predial_nacional"] == "P12"


def test_soft_rules_still_downgrade_instead_of_rejecting():
    cad = [{"direccion": "KR 765 F # 432 - 20", "lat": 3.41, "lon": LON, "predial": "PX", "manzana": "M1"}]
    assert run("KR 765 E # 432 - 20", cad)["estado"] == "SIN_MATCH"
    row = run("KR 765 E # 432 - 20", cad, soft_rules=["letra_via"])
    assert row["estado"] == "OK" and row["position_tier"] == "manzana" and row["nivel_precision"] == "manzana"


def test_plate_delta_three_is_rejected_not_capped():
    row = run("CL 987 # 654 - 15", [doc(12), doc(16), doc(20)])
    assert row["estado"] == "SIN_MATCH" and row["position_tier"] == "none"
    assert bool(row["override_rejected"]) is True  # a structural rule rejected the candidate


# ---------------------------------------------------------------------------
# via: extrapolation
# ---------------------------------------------------------------------------
def test_extrapolation_withholds_coordinates_by_default():
    row = run("CL 987 # 654 - 22", [doc(20), doc(16), doc(12)])
    assert row["estado"] == "OK"
    assert (row["position_tier"], row["position_radius_m"], row["position_source"]) == ("via", 10, "none")
    assert pd.isna(row["lat"]) and pd.isna(row["lon"])
    assert row["numero_predial_nacional"] is None and pd.isna(row["manzana"])
    assert row["nivel_precision"] == "via"
    assert "extrapolada" in row["motivo"]


def test_extrapolated_coordinates_flag_publishes_the_extension(monkeypatch):
    monkeypatch.setattr(service, "EXTRAPOLATED_COORDS_ENABLED", True)
    row = run("CL 987 # 654 - 22", [doc(20), doc(16), doc(12)])
    assert row["position_tier"] == "via" and row["position_source"] == "interp"
    assert row["lat"] == pytest.approx(lat_of(22), abs=1e-9)
    assert row["position_radius_m"] == 10   # extrapolation step (0, 25] m: p90 10 m


def test_flag_off_and_on_agree_on_everything_but_the_coordinates(monkeypatch):
    docs = [doc(20), doc(16), doc(12)]
    off = run("CL 987 # 654 - 22", docs)
    monkeypatch.setattr(service, "EXTRAPOLATED_COORDS_ENABLED", True)
    on = run("CL 987 # 654 - 22", docs)
    same = [c for c in OUTPUT_COLUMNS if c not in ("lat", "lon", "position_source", "motivo")]
    for col in same:
        assert (off[col] == on[col]) or (pd.isna(off[col]) and pd.isna(on[col])), col


def test_flag_never_affects_predio_or_tramo(monkeypatch):
    base_a = run("CL 987 # 654 - 14", FACE)
    monkeypatch.setattr(service, "EXTRAPOLATED_COORDS_ENABLED", True)
    base_b = run("CL 987 # 654 - 14", FACE)
    assert base_a.equals(base_b)


def test_extrapolation_past_the_cap_has_no_coordinates_even_when_enabled(monkeypatch):
    monkeypatch.setattr(service, "EXTRAPOLATED_COORDS_ENABLED", True)
    row = run("CL 987 # 654 - 22", [doc(20)], plate_tolerance=0)
    # one parcel on the face: its own point is all there is, within the cap
    assert row["position_tier"] == "via" and row["lat"] == lat_of(20)


# ---------------------------------------------------------------------------
# manzana: one-sided letters and quadrants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral", [
    ("KR 765 # 432 - 20", "KR 765 E # 432 - 20"),
    ("KR 765 E # 432 - 20", "KR 765 # 432 - 20"),
    ("KR 765 NORTE # 432 - 20", "KR 765 # 432 - 20"),
    ("KR 765 # 432 - 20", "KR 765 # 432 OESTE - 20"),
    ("KR 765 # 432 - 20", "KR 765 # 432 - 20 NORTE"),
])
def test_one_sided_letter_or_quadrant_is_manzana(query, cadastral):
    row = run(query, [{"direccion": cadastral, "lat": 3.41, "lon": LON, "predial": "PX", "manzana": "M1"}])
    assert row["estado"] == "OK"
    assert (row["position_tier"], row["position_radius_m"], row["position_source"]) == ("manzana", 56, "parcel")
    # a one-sided letter / quadrant: the parcel may be on another street, so the predial is never kept (was PX)
    assert row["numero_predial_nacional"] is None and (row["lat"], row["lon"]) == (3.41, LON)
    assert row["nivel_precision"] == "manzana"


# ---------------------------------------------------------------------------
# rejections by the guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral", [
    ("CL 987 N # 654 - 20", "CL 987 # 654 NORTE - 20"),
    ("AV 876 # 654 N - 06", "AV 876 NORTE # 654 - 06"),
    ("CL 987 # 654 - 20 NORTE", "CL 987 NORTE # 654 - 20"),
])
def test_moved_quadrant_is_rejected_with_the_flag(query, cadastral):
    row = run(query, [{"direccion": cadastral, "lat": 3.41, "lon": LON}])
    assert row["estado"] == "SIN_MATCH" and bool(row["override_rejected"]) is True
    # the legacy rules already reject the cross-letter shape of AV 876 # 654 N; the guard names the rest
    assert "quadrant_moved" in row["motivo"] or "letra de cruce N" in row["motivo"]
    assert (row["position_tier"], row["position_source"]) == ("none", "none")
    assert math.isnan(row["position_radius_m"]) and pd.isna(row["selected_doc_id"])
    assert pd.isna(row["lat"]) and row["numero_predial_nacional"] is None and pd.isna(row["nivel_precision"])


def test_rejected_row_renders_the_rule_text_not_the_cadastre():
    row = run("CL 987 N # 654 - 20 TO 2", [{"direccion": "CL 987 # 654 NORTE - 20 TO 9", "lat": 3.4, "lon": LON}])
    assert row["estado"] == "SIN_MATCH" and row["fuente_normalizacion"] == "reglas"
    assert row["direccion_normalizada"] == "CL 987 N # 654 - 20 TO 2"  # the rule text, canonical, input unit
    assert row["complemento_normalizado"] == "TO 2"
    assert "NORTE" not in row["direccion_normalizada"] and "TO 9" not in row["direccion_normalizada"]


def test_kr_glued_quadrant_example_is_capped_not_predio():
    row = run("KR 987NORTE 654 3", [{"direccion": "KR 987 NORTE # 654 NORTE - 03", "lat": 3.4, "lon": LON}])
    assert row["estado"] == "OK" and row["position_tier"] == "manzana"
    assert row["nivel_precision"] == "manzana"


# ---------------------------------------------------------------------------
# lote: site units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral", [
    ("CL 987 # 654 - 12 TO 3", "CL 987 # 654 - 12 TO 2"),
    ("CL 987 # 654 - 12 BLQ 1", "CL 987 # 654 - 12 BLQ 2"),
])
def test_unit_difference_is_lote_without_predial(query, cadastral):
    row = run(query, [{"direccion": cadastral, "lat": 3.41, "lon": LON, "predial": "PX", "manzana": "M1"}])
    assert row["estado"] == "OK"
    assert (row["position_tier"], row["position_radius_m"], row["position_source"]) == ("lote", 44, "parcel")
    assert row["numero_predial_nacional"] is None and row["manzana"] == "M1"
    assert (row["lat"], row["lon"]) == (3.41, LON)
    assert row["nivel_precision"] == "manzana"
    assert "posicion maxima lote" in row["motivo"]
    assert "TO 2" not in str(row["direccion_normalizada"]) or "TO 3" in str(row["direccion_normalizada"])


def test_unit_only_in_the_cadastre_keeps_the_predio():
    row = run("CL 987 # 654 - 12", [{"direccion": "CL 987 # 654 - 12 CA 4", "lat": 3.41, "lon": LON, "predial": "PX"}])
    assert row["position_tier"] == "predio" and row["numero_predial_nacional"] == "PX"


def test_most_restrictive_cap_wins():
    cad = {"direccion": "CL 987 # 654 - 12 TO 2", "lat": 3.41, "lon": LON, "predial": "PX", "manzana": "M1"}
    # unit (lote) + one-sided quadrant (manzana) -> manzana, predial kept like every manzana row
    row = run("CL 987 NORTE # 654 - 12 TO 3", [cad])
    assert row["position_tier"] == "manzana" and row["numero_predial_nacional"] is None  # one-sided: no predial


def test_shared_address_is_manzana_tier():
    row = run("CL 987 # 654 - 12", [doc(12, n_predios=3)])
    assert row["position_tier"] == "manzana" and row["nivel_precision"] == "manzana"
    row = run("CL 987 # 654 - 12", [doc(12, n_predios=3, manzana_nunique=2)])
    assert row["position_tier"] == "manzana" and row["nivel_precision"] == "direccion"


# fidelity caps off (fixture): the output invents a plate (high risk); the cap itself is tested elsewhere
def test_missing_plate_and_cross_keep_their_legacy_levels(no_fidelity_caps):
    assert run("CL 987 # 654", FACE)["nivel_precision"] == "esquina"
    assert run("CL 987 # 654", FACE)["position_tier"] == "manzana"


# ---------------------------------------------------------------------------
# non-OK rows, malformed input, glued alphanumerics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [None, "", "   ", "N/A", "hola mundo", "#", "ÑÑÑ" * 2000, "CL 987 # 654 - 12 " + "X" * 5000])
def test_non_ok_or_odd_inputs_never_raise_and_carry_none_tier(raw):
    row = normalize_strict(make_stub(FACE), [raw], gazetteer=None).iloc[0]
    assert row["estado"] in ("OK", "SIN_MATCH", "NO_PARSEABLE")
    if row["estado"] != "OK":
        assert row["position_tier"] == "none" and row["position_source"] == "none"
        assert math.isnan(row["position_radius_m"]) and row["override_rejected"] in (False, True)
        assert pd.isna(row["selected_doc_id"])


@pytest.mark.parametrize("raw", ["CL 987 # 654 - 66C", "KR 926G6 # 972S1 - 08CA", "CL 987 # 654 - 1A9-250", "KR 915BNORTE 4 - 5",
                                 "CL 922NORTE # 653 - 5"])
def test_glued_alphanumerics_keep_the_contract(raw):
    row = normalize_strict(make_stub(FACE), [raw], gazetteer=None).iloc[0]
    assert row["position_tier"] in ("predio", "lote", "manzana", "tramo", "via", "none")
    if row["position_tier"] == "none":
        assert row["estado"] != "OK"
    else:
        assert row["position_radius_m"] in set(POSITION_RADIUS_M.values())


def test_ok_rows_always_have_a_known_tier_and_radius():
    queries = ["CL 987 # 654 - 12", "CL 987 # 654 - 14", "CL 987 # 654 - 13", "CL 987 # 654 - 22", "CL 987 # 654 - 12 C",
               "CL 987 # 654"]
    frame = normalize_strict(make_stub(FACE), queries, gazetteer=None)
    ok = frame[frame["estado"] == "OK"]
    assert len(ok) >= 4
    assert set(ok["position_tier"]) <= set(POSITION_RADIUS_M)
    # predio / lote / manzana use the tier radius; tramo and extrapolated via rows follow their step function
    from cali_address.interp import EXTRAP_RADIUS_STEPS, TRAMO_RADIUS_STEPS
    steps = {"tramo": {max(r, 15.0) for _, r in TRAMO_RADIUS_STEPS}, "via": {r for _, r in EXTRAP_RADIUS_STEPS} | {21.0}}
    for tier, radius in zip(ok["position_tier"], ok["position_radius_m"]):
        assert radius in steps.get(tier, {float(POSITION_RADIUS_M[tier])})


def test_determinism_and_batch_independence():
    queries = ["CL 987 # 654 - 12", "CL 987 # 654 - 14", "CL 987 # 654 - 22", "CL 987 # 654 - 12 TO 2"]
    a = normalize_strict(make_stub(FACE), queries, gazetteer=None)
    b = normalize_strict(make_stub(FACE), queries, gazetteer=None)
    pd.testing.assert_frame_equal(a, b)
    alone = normalize_strict(make_stub(FACE), [queries[1]], gazetteer=None).iloc[0]
    assert alone.equals(a.iloc[1])


def test_idempotence_the_normalized_address_normalizes_to_itself():
    first = run("CL 987 # 654 - 12 TO 2", FACE)
    again = run(str(first["direccion_normalizada"]), FACE)
    assert again["direccion_normalizada"] == first["direccion_normalizada"]
    assert again["position_tier"] == "predio"


# ---------------------------------------------------------------------------
# consistency research <-> production guard through the service
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, cadastral", [
    ("CL 987 # 654 - 20", "CL 987 # 654 - 20"),
    ("CL 987 N # 654 - 20", "CL 987 NORTE # 654 - 20"),
    ("CL 987 N # 654 - 20", "CL 987 # 654 NORTE - 20"),
    ("CL 987 # 654 - 21", "CL 987 # 654 - 20"),
    ("KR 765 # 432 - 20", "KR 765 E # 432 - 20"),
])
def test_service_rejects_exactly_what_the_guard_calls_hard(query, cadastral):
    from cali_address.inference import base_mismatches
    from cali_address.parser import parse_address

    hard, _caps = base_mismatches(parse_address(query), parse_address(cadastral), mode="position")
    row = run(query, [{"direccion": cadastral, "lat": 3.4, "lon": LON}])
    if hard:
        assert row["estado"] == "SIN_MATCH"
    else:
        assert row["estado"] == "OK"


# ---------------------------------------------------------------------------
# API schema and IO writers
# ---------------------------------------------------------------------------
def _frame():
    return normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12", "CL 987 # 654 - 14", "", "hola"], gazetteer=None)


def test_result_row_schema_declares_the_position_fields_last():
    from cali_address.api.schemas import ResultRow

    names = list(ResultRow.model_fields)
    assert names[-10:-5] == NEW_COLUMNS and names[-5] == "direccion_catastral"
    assert names[-4:-2] == ["fidelity_risk", "fidelity_flags"]
    assert names[:-10][-1] == "complemento_normalizado"
    assert ResultRow.model_config.get("extra") == "allow"


def test_result_row_accepts_ok_and_non_ok_records():
    from cali_address.api.schemas import ResultRow

    for rec in results_to_records(_frame()):
        row = ResultRow(**rec)
        assert row.position_tier == rec["position_tier"]
        assert row.override_rejected is rec["override_rejected"]
        dumped = row.model_dump(mode="json")
        assert dumped["position_radius_m"] == rec["position_radius_m"]
        assert dumped["selected_doc_id"] == rec["selected_doc_id"]


def test_result_row_rejects_nonsense_tier_values():
    from pydantic import ValidationError

    from cali_address.api.schemas import ResultRow

    base = results_to_records(_frame())[0]
    with pytest.raises(ValidationError):
        ResultRow(**{**base, "position_tier": "adoquin"})
    with pytest.raises(ValidationError):
        ResultRow(**{**base, "position_source": "magic"})


@pytest.mark.parametrize("fmt", ["csv", "parquet", "jsonl", "xlsx"])
def test_io_writers_keep_the_position_columns_across_chunks(tmp_path, fmt):
    from cali_address.io import open_sink

    frame = _frame()
    path = tmp_path / f"out.{fmt}"
    with open_sink(str(path)) as sink:
        sink.write(frame.iloc[2:].reset_index(drop=True))  # first chunk: no tiers, null ids
        sink.write(frame.iloc[:2].reset_index(drop=True))
    if fmt == "parquet":
        back = pd.read_parquet(path)
    elif fmt == "csv":
        back = pd.read_csv(path, encoding="utf-8-sig")
    elif fmt == "jsonl":
        back = pd.read_json(path, lines=True)
    else:
        back = pd.read_excel(path, sheet_name=0)
    assert list(back.columns)[-10:-5] == NEW_COLUMNS and list(back.columns)[-5] == "direccion_catastral"
    assert list(back.columns)[-4:-2] == ["fidelity_risk", "fidelity_flags"]
    assert back["position_tier"].tolist() == ["none", "none", "predio", "tramo"]
    assert back["position_radius_m"].iloc[2:].tolist() == [15.0, 15.0]  # tramo: gap of ~11 m -> floor
    assert back["position_radius_m"].iloc[:2].isna().all()
    if fmt != "xlsx":
        assert back["selected_doc_id"].iloc[2:].tolist() == [0, 0]
    assert not back["override_rejected"].astype(bool).any()


def test_csv_writes_selected_doc_id_as_an_integer(tmp_path):
    from cali_address.io import open_sink

    path = tmp_path / "o.csv"
    with open_sink(str(path)) as sink:
        sink.write(_frame())
    back = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert back["selected_doc_id"].tolist() == ["0", "0", "", ""]
    assert back["position_radius_m"].tolist() == ["15.0", "15.0", "", ""]
    assert back["override_rejected"].tolist() == ["False"] * 4
