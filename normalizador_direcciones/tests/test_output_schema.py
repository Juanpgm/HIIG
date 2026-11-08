"""Output validation (``cali_address.output_schema``): pydantic schema + conservative corrections.

Only invented data is used (improbable numbers, 900+). The vocabularies pinned here are derived from the real
``service`` module, so a drift between the two fails a test instead of silently rotting.
"""

from __future__ import annotations

import copy
import math
import random

import pandas as pd
import pytest
from pydantic import ValidationError

from cali_address import output_schema as osch
from cali_address import service
from cali_address.output_schema import (
    CALI_BBOX,
    Correction,
    NormalizedRecord,
    OutputSchemaError,
    validate_and_correct,
)

NPN = "76001" + "0" * 22 + "905"  # 30 digits, invented
MZ = "76001" + "0" * 12 + "905"


def ok_row(**over) -> dict:
    row = {
        "direccion_entrada": "CRA 905 # 950-10",
        "estado": "OK",
        "motivo": "",
        "direccion_normalizada": "CARRERA 905 950 10",
        "fuente_normalizacion": "catastro",
        "numero_predial_nacional": NPN,
        "manzana": MZ,
        "lat": 3.40,
        "lon": -76.50,
        "confianza": 0.9,
        "confiabilidad": 0.8,
        "confiabilidad_manzana": 0.85,
        "nivel_precision": "predio",
        "position_tier": "predio",
        "position_radius_m": 15.0,
        "position_source": "parcel",
        "fidelity_risk": "none",
    }
    row.update(over)
    return row


def sin_match_row(**over) -> dict:
    row = ok_row(
        estado="SIN_MATCH", motivo="confianza 0.10 < umbral 0.50", numero_predial_nacional=None, manzana=None,
        lat=None, lon=None, confianza=None, confiabilidad=None, confiabilidad_manzana=None,
        nivel_precision=None, position_tier="none", position_radius_m=float("nan"), position_source="none",
        fuente_normalizacion="reglas",
    )
    row.update(over)
    return row


def rules_of(row: dict) -> set[str]:
    _, report = validate_and_correct(row, "report")
    return set(report.violations)


# --------------------------------------------------------------------------- vocabularies mirror the engine
def test_vocabularies_match_the_real_service_module():
    assert osch.ESTADOS == service.PUBLIC_STATUSES
    assert osch.POSITION_TIERS == service._TIER_ORDER
    assert osch.NIVELES == service._PRECISION_ORDER
    assert osch.NIVEL_OF_TIER == service._NIVEL_OF_TIER
    assert osch.POSITION_RADIUS_M == service.POSITION_RADIUS_M


def test_output_columns_required_by_the_schema_exist_in_the_service_output():
    assert set(osch.REQUIRED_COLUMNS) <= set(service.OUTPUT_COLUMNS)


# --------------------------------------------------------------------------- already valid rows
@pytest.mark.parametrize("row", [ok_row(), sin_match_row(), sin_match_row(direccion_normalizada=None, fuente_normalizacion=None)])
def test_valid_rows_have_no_violations_and_stay_identical_in_every_mode(row):
    for mode in ("off", "report", "correct"):
        out, report = validate_and_correct(row, mode)
        assert report.violations == {}
        assert out == row or (
            # NaN != NaN: compare through repr for the NaN radius
            repr(out) == repr(row)
        )


def test_normalized_record_accepts_a_valid_row_and_ignores_extra_columns():
    rec = NormalizedRecord.model_validate({**ok_row(), "some_future_column": 1})
    assert rec.estado == "OK" and rec.confianza == 0.9


def test_normalized_record_lists_every_violated_rule_at_once():
    with pytest.raises(ValidationError) as info:
        NormalizedRecord.model_validate(ok_row(confianza=1.5, numero_predial_nacional="123", lat=10.0, lon=-76.5))
    assert {e["type"] for e in info.value.errors()} >= {"confidence_range", "npn_format", "coords_out_of_bbox"}


# --------------------------------------------------------------------------- modes
def test_unknown_mode_raises_a_clear_error():
    with pytest.raises(ValueError, match="mode"):
        validate_and_correct(ok_row(), "fix-everything")


def test_mode_off_is_a_noop_even_for_garbage():
    bad = ok_row(confianza=7, lat=99.0)
    out, report = validate_and_correct(bad, "off")
    assert out is bad and report.mode == "off" and report.violations == {} and report.corrections == []


def test_mode_report_counts_without_changing_the_output():
    bad = ok_row(confianza=7.0, lat=99.0)
    snapshot = copy.deepcopy(bad)
    out, report = validate_and_correct(bad, "report")
    assert out is bad and bad == snapshot
    assert report.violations == {"confidence_range": 1, "coords_out_of_bbox": 1}
    assert report.corrections == []


def test_mode_report_on_a_frame_returns_the_same_frame_untouched():
    frame = pd.DataFrame([ok_row(confianza=2.0), ok_row()])
    before = frame.copy()
    out, report = validate_and_correct(frame, "report")
    assert out is frame
    pd.testing.assert_frame_equal(frame, before)
    assert report.violations == {"confidence_range": 1}
    assert report.rows == 2 and report.rows_with_violations == 1


# --------------------------------------------------------------------------- confidence
@pytest.mark.parametrize("value", [0.0, 1.0, 0.5, 1e-9, 0.9999999])
def test_confidence_inside_the_unit_interval_is_valid(value):
    assert rules_of(ok_row(confianza=value, confiabilidad=value, confiabilidad_manzana=value)) == set()


@pytest.mark.parametrize("value,expected", [
    (1.0000001, 1.0), (1.5, 1.0), (-0.0001, 0.0), (-3.0, 0.0), (float("inf"), 1.0), (float("-inf"), 0.0),
])
def test_confidence_out_of_range_is_clamped(value, expected):
    row = ok_row(confianza=value)
    assert rules_of(row) == {"confidence_range"}
    out, report = validate_and_correct(row, "correct")
    assert out["confianza"] == expected
    assert report.corrections == [Correction(field="confianza", before=value, after=expected, rule="confidence_range")]
    assert rules_of(out) == set()


@pytest.mark.parametrize("field", ["confianza", "confiabilidad", "confiabilidad_manzana"])
def test_every_confidence_column_is_checked(field):
    out, report = validate_and_correct(ok_row(**{field: 1.2}), "correct")
    assert out[field] == 1.0 and report.violations == {"confidence_range": 1}


def test_confidence_nan_in_a_dict_row_becomes_null_with_a_violation():
    out, report = validate_and_correct(ok_row(confianza=float("nan")), "correct")
    assert out["confianza"] is None
    assert report.violations == {"confidence_nan": 1}


def test_confidence_not_a_number_becomes_null_with_a_violation():
    out, report = validate_and_correct(ok_row(confianza="high"), "correct")
    assert out["confianza"] is None and report.violations == {"confidence_nan": 1}


def test_confidence_boolean_is_not_a_number():
    assert rules_of(ok_row(confianza=True)) == {"confidence_nan"}


def test_null_confidence_is_valid():
    assert rules_of(ok_row(confianza=None, confiabilidad=None)) == set()


def test_float_column_nan_of_a_frame_is_null_not_a_violation():
    """A float64 column cannot tell None from NaN: a non-OK row's missing confidence is not an error."""
    frame = pd.DataFrame([ok_row(), sin_match_row()])
    assert frame["confianza"].dtype == "float64"
    _, report = validate_and_correct(frame, "report")
    assert report.violations == {}


def test_object_column_nan_of_a_frame_is_a_violation():
    frame = pd.DataFrame([ok_row(), ok_row()])
    frame["confianza"] = pd.Series([0.9, float("nan")], dtype=object)
    out, report = validate_and_correct(frame, "correct")
    assert report.violations == {"confidence_nan": 1}
    assert out["confianza"].iloc[1] is None or pd.isna(out["confianza"].iloc[1])


# --------------------------------------------------------------------------- numero predial nacional
@pytest.mark.parametrize("npn", [None, NPN, float("nan"), ""])
def test_npn_null_or_30_digits_is_valid(npn):
    row = ok_row(numero_predial_nacional=npn, nivel_precision="manzana", position_tier="manzana")
    assert "npn_format" not in rules_of(row)


@pytest.mark.parametrize("npn", [NPN[:-1], NPN + "1", "7" * 29, "7" * 31, NPN[:-1] + "X", "٣" * 30, 7600100000000000000000000000905])
def test_npn_with_wrong_length_or_non_digits_is_a_violation_and_is_nulled(npn):
    row = ok_row(numero_predial_nacional=npn)
    assert "npn_format" in rules_of(row)
    out, report = validate_and_correct(row, "correct")
    assert out["numero_predial_nacional"] is None
    assert any(c.rule == "npn_format" and c.field == "numero_predial_nacional" for c in report.corrections)


def test_npn_that_only_needs_stripping_is_kept():
    out, report = validate_and_correct(ok_row(numero_predial_nacional=f" {NPN}\n"), "correct")
    assert out["numero_predial_nacional"] == NPN
    assert report.violations == {"whitespace": 1}


def test_npn_nulled_lowers_the_predio_claim_to_the_manzana():
    out, report = validate_and_correct(ok_row(numero_predial_nacional="123"), "correct")
    assert out["nivel_precision"] == "manzana"
    assert out["position_tier"] == "lote"
    assert out["position_radius_m"] == float(service.POSITION_RADIUS_M["lote"])
    assert {"npn_format", "ok_without_support"} <= set(report.violations)


# --------------------------------------------------------------------------- coordinates
@pytest.mark.parametrize("lon,lat", [
    (CALI_BBOX["lon_min"], CALI_BBOX["lat_min"]), (CALI_BBOX["lon_max"], CALI_BBOX["lat_max"]),
    (CALI_BBOX["lon_min"], CALI_BBOX["lat_max"]), (-76.5, 3.4),
])
def test_coordinates_exactly_on_the_bbox_edge_are_valid(lon, lat):
    assert rules_of(ok_row(lon=lon, lat=lat)) == set()


@pytest.mark.parametrize("lon,lat", [
    (-76.65 - 1e-9, 3.4), (-76.40 + 1e-9, 3.4), (-76.5, 3.30 - 1e-9), (-76.5, 3.55 + 1e-9),
    (0.0, 0.0), (-76.5, float("inf")), (float("-inf"), 3.4),
])
def test_coordinates_just_outside_the_bbox_are_nulled_with_a_note(lon, lat):
    row = ok_row(lon=lon, lat=lat, motivo="aproximado: algo")
    assert rules_of(row) == {"coords_out_of_bbox"}
    out, report = validate_and_correct(row, "correct")
    assert out["lat"] is None and out["lon"] is None
    assert out["motivo"].startswith("aproximado: algo; ") and out["motivo"].endswith(osch.NOTE_COORDS_NULLED)
    assert out["position_source"] == "none"
    assert {c.field for c in report.corrections} == {"lat", "lon", "motivo", "position_source"}
    assert rules_of(out) == set()


def test_coordinate_note_on_an_empty_motivo_has_no_leading_separator():
    out, _ = validate_and_correct(ok_row(lat=50.0, motivo=""), "correct")
    assert out["motivo"] == osch.NOTE_COORDS_NULLED


def test_only_one_coordinate_present_is_a_violation_and_both_are_nulled():
    for row in (ok_row(lat=None), ok_row(lon=None)):
        assert rules_of(row) == {"coords_partial"}
        out, _ = validate_and_correct(row, "correct")
        assert out["lat"] is None and out["lon"] is None


def test_both_coordinates_null_is_valid_for_a_via_row():
    row = ok_row(lat=None, lon=None, nivel_precision="via", position_tier="via", numero_predial_nacional=None,
                 manzana=None, position_radius_m=21.0, position_source="none")
    assert rules_of(row) == set()


def test_unparseable_coordinate_is_out_of_bbox():
    assert rules_of(ok_row(lat="north")) == {"coords_out_of_bbox"}


# --------------------------------------------------------------------------- tier / nivel coherence
@pytest.mark.parametrize("tier,nivel", [
    ("predio", "predio"), ("lote", "manzana"), ("tramo", "manzana"), ("manzana", "manzana"), ("via", "via"),
    ("predio", "manzana"), ("manzana", "esquina"), ("manzana", "direccion"), ("via", "direccion"),
])
def test_coherent_tier_and_nivel_pairs_are_valid(tier, nivel):
    row = ok_row(position_tier=tier, nivel_precision=nivel)
    if tier in ("lote", "tramo", "via"):
        row["numero_predial_nacional"] = None
    if tier == "via":
        row.update(manzana=None, lat=None, lon=None, position_source="none")
    assert rules_of(row) == set()


@pytest.mark.parametrize("tier,nivel,lowered", [
    ("manzana", "predio", "manzana"), ("lote", "predio", "manzana"), ("tramo", "predio", "manzana"),
    ("via", "manzana", "via"), ("via", "predio", "via"), ("via", "esquina", "via"),
])
def test_nivel_more_precise_than_the_tier_allows_is_lowered(tier, nivel, lowered):
    row = ok_row(position_tier=tier, nivel_precision=nivel)
    assert "tier_nivel_incoherent" in rules_of(row)
    out, report = validate_and_correct(row, "correct")
    assert out["nivel_precision"] == lowered
    assert out["position_tier"] == tier
    assert any(c.rule == "tier_nivel_incoherent" for c in report.corrections)


def test_tier_vocabulary_violation_is_reported_and_left_alone():
    row = ok_row(position_tier="exacto")
    out, report = validate_and_correct(row, "correct")
    assert report.violations == {"position_tier_vocab": 1}
    assert out["position_tier"] == "exacto"


# --------------------------------------------------------------------------- OK rows need support for their level
def test_predio_level_without_npn_is_lowered_to_manzana_and_lote():
    row = ok_row(numero_predial_nacional=None)
    assert rules_of(row) == {"ok_without_support"}
    out, report = validate_and_correct(row, "correct")
    assert (out["nivel_precision"], out["position_tier"]) == ("manzana", "lote")
    assert {c.field for c in report.corrections} == {"nivel_precision", "position_tier", "position_radius_m"}


def test_manzana_level_without_any_cadastral_code_is_lowered_to_via():
    row = ok_row(numero_predial_nacional=None, manzana=None, nivel_precision="manzana", position_tier="manzana")
    assert rules_of(row) == {"ok_without_support"}
    out, _ = validate_and_correct(row, "correct")
    assert (out["nivel_precision"], out["position_tier"]) == ("via", "via")


def test_manzana_blank_string_counts_as_missing():
    row = ok_row(numero_predial_nacional=None, manzana="   ", nivel_precision="manzana", position_tier="manzana")
    assert "ok_without_support" in rules_of(row)


def test_via_level_needs_no_cadastral_code():
    row = ok_row(numero_predial_nacional=None, manzana=None, nivel_precision="via", position_tier="via",
                 lat=None, lon=None, position_source="none")
    assert rules_of(row) == set()


def test_non_ok_rows_are_not_asked_for_cadastral_support():
    assert rules_of(sin_match_row()) == set()


# --------------------------------------------------------------------------- text / source
def test_source_without_text_is_nulled():
    row = sin_match_row(direccion_normalizada=None, fuente_normalizacion="reglas")
    assert rules_of(row) == {"text_source"}
    out, report = validate_and_correct(row, "correct")
    assert out["fuente_normalizacion"] is None
    assert report.corrections == [Correction(field="fuente_normalizacion", before="reglas", after=None, rule="text_source")]


@pytest.mark.parametrize("text", [None, "", "   ", float("nan")])
def test_blank_text_counts_as_no_text(text):
    assert "text_source" in rules_of(sin_match_row(direccion_normalizada=text, fuente_normalizacion="reglas"))


def test_text_without_source_is_not_invented():
    assert rules_of(sin_match_row(fuente_normalizacion=None)) == set()


def test_non_ok_row_with_high_fidelity_risk_must_not_emit_text():
    row = sin_match_row(fidelity_risk="high")
    assert rules_of(row) == {"high_risk_text"}
    out, report = validate_and_correct(row, "correct")
    assert out["direccion_normalizada"] is None and out["fuente_normalizacion"] is None
    assert {c.field for c in report.corrections} == {"direccion_normalizada", "fuente_normalizacion"}


def test_ok_row_with_high_fidelity_risk_is_not_this_rules_business():
    assert "high_risk_text" not in rules_of(ok_row(fidelity_risk="high"))


@pytest.mark.parametrize("risk", ["none", "low", "medium", None])
def test_non_ok_row_with_lower_risk_keeps_its_text(risk):
    assert rules_of(sin_match_row(fidelity_risk=risk)) == set()


# --------------------------------------------------------------------------- vocabularies / whitespace
@pytest.mark.parametrize("estado", ["ok", "MATCH", "", None, 3])
def test_estado_outside_the_vocabulary_is_reported_but_never_rewritten(estado):
    row = sin_match_row(estado=estado)
    out, report = validate_and_correct(row, "correct")
    assert "estado_vocab" in report.violations
    assert out["estado"] == estado


def test_nivel_outside_the_vocabulary_is_lowered_to_the_least_precise_level_for_ok_rows():
    out, report = validate_and_correct(ok_row(nivel_precision="exacto"), "correct")
    assert "nivel_vocab" in report.violations
    assert out["nivel_precision"] == "direccion"


def test_nivel_outside_the_vocabulary_is_nulled_for_non_ok_rows():
    out, _ = validate_and_correct(sin_match_row(nivel_precision="exacto"), "correct")
    assert out["nivel_precision"] is None


def test_whitespace_is_stripped_but_the_echoed_input_is_left_alone():
    row = ok_row(direccion_entrada="  CRA 905 # 950-10 ", direccion_normalizada=" CARRERA 905 950 10 ", estado=" OK ")
    out, report = validate_and_correct(row, "correct")
    assert out["direccion_entrada"] == "  CRA 905 # 950-10 "
    assert out["direccion_normalizada"] == "CARRERA 905 950 10" and out["estado"] == "OK"
    assert report.violations == {"whitespace": 2}


def test_stripped_estado_is_judged_after_stripping():
    assert "estado_vocab" not in rules_of(ok_row(estado=" OK "))


def test_empty_motivo_of_a_valid_ok_row_is_preserved():
    out, _ = validate_and_correct(ok_row(motivo=""), "correct")
    assert out["motivo"] == ""


# --------------------------------------------------------------------------- several violations / idempotency / optimism
def test_a_row_with_several_violations_reports_each_rule_once():
    row = ok_row(confianza=1.7, lat=50.0, numero_predial_nacional="1" * 29, estado=" OK ",
                 position_tier="manzana", nivel_precision="predio")
    _, report = validate_and_correct(row, "report")
    assert set(report.violations) >= {"confidence_range", "coords_out_of_bbox", "npn_format", "whitespace", "tier_nivel_incoherent"}
    assert all(v == 1 for v in report.violations.values())


def test_correct_leaves_no_fixable_violation_behind():
    row = ok_row(confianza=1.7, lat=50.0, numero_predial_nacional="1" * 29, estado=" OK ",
                 position_tier="manzana", nivel_precision="predio", direccion_normalizada=None)
    out, _ = validate_and_correct(row, "correct")
    assert rules_of(out) == set()


def _same(a, b) -> bool:
    return repr(a) == repr(b)


def test_correct_twice_equals_correct_once():
    row = ok_row(confianza=-4, lat=50.0, numero_predial_nacional="x", estado=" OK ", position_tier="via")
    once, _ = validate_and_correct(row, "correct")
    twice, report2 = validate_and_correct(once, "correct")
    assert _same(once, twice)
    assert report2.violations == {} and report2.corrections == []


def _random_row(rng: random.Random) -> dict:
    tiers = list(osch.POSITION_TIERS) + ["bogus"]
    niveles = list(osch.NIVELES) + ["bogus", None]
    estados = ["OK", "SIN_MATCH", "NO_PARSEABLE", " OK", "ok"]
    conf = [None, 0.0, 1.0, 0.5, 1.0000001, -0.0001, float("nan"), float("inf"), float("-inf"), 7.0, "x"]
    return ok_row(
        estado=rng.choice(estados),
        confianza=rng.choice(conf), confiabilidad=rng.choice(conf), confiabilidad_manzana=rng.choice(conf),
        numero_predial_nacional=rng.choice([None, NPN, NPN[:-1], NPN + "0", "abc", " " + NPN]),
        manzana=rng.choice([None, MZ, "", "  "]),
        lat=rng.choice([None, 3.4, 3.3, 3.55, 3.56, 10.0, float("nan")]),
        lon=rng.choice([None, -76.5, -76.65, -76.4, -76.7, 0.0]),
        nivel_precision=rng.choice(niveles), position_tier=rng.choice(tiers),
        direccion_normalizada=rng.choice([None, "CARRERA 905 950 10", "  ", " X "]),
        fuente_normalizacion=rng.choice([None, "catastro", "reglas"]),
        fidelity_risk=rng.choice(["none", "low", "medium", "high", None]),
        motivo=rng.choice(["", "x", None]),
    )


def _rank_nivel(v):
    return osch.NIVELES.index(v) if v in osch.NIVELES else -1


def _rank_tier(v):
    return -osch.POSITION_TIERS.index(v) if v in osch.POSITION_TIERS else 99


def test_property_correct_is_idempotent_and_never_raises_optimism_on_random_rows():
    rng = random.Random(42)
    for _ in range(1500):
        row = _random_row(rng)
        before = copy.deepcopy(row)
        out, _ = validate_and_correct(row, "correct")
        assert _same(row, before), "the input row must not be mutated"
        # idempotent
        again, report = validate_and_correct(out, "correct")
        assert _same(out, again), (before, out, again)
        assert report.corrections == []
        # never more optimistic
        for key in ("confianza", "confiabilidad", "confiabilidad_manzana"):
            b, a = before[key], out[key]
            if isinstance(b, float) and math.isfinite(b) and 0.0 <= b <= 1.0:
                assert a == b
            if a is not None:
                assert 0.0 <= a <= 1.0
        if before["nivel_precision"] in osch.NIVELES and out["nivel_precision"] in osch.NIVELES:
            assert _rank_nivel(out["nivel_precision"]) <= _rank_nivel(before["nivel_precision"])
        if before["position_tier"] in osch.POSITION_TIERS:
            assert _rank_tier(out["position_tier"]) <= _rank_tier(before["position_tier"])
        if before["fuente_normalizacion"] is None:
            assert out["fuente_normalizacion"] is None
        if before["numero_predial_nacional"] is None:
            assert out["numero_predial_nacional"] is None
        if pd.isna(before["lat"]) or pd.isna(before["lon"]):
            assert pd.isna(out["lat"]) and pd.isna(out["lon"])


# --------------------------------------------------------------------------- frames
def test_empty_dataframe_is_fine_in_every_mode():
    frame = pd.DataFrame(columns=service.OUTPUT_COLUMNS)
    for mode in ("off", "report", "correct"):
        out, report = validate_and_correct(frame, mode)
        assert len(out) == 0 and report.rows == 0 and report.violations == {}


def test_missing_required_columns_raise_a_clear_error_listing_them():
    frame = pd.DataFrame([ok_row()]).drop(columns=["estado", "confianza"])
    with pytest.raises(OutputSchemaError, match="confianza") as info:
        validate_and_correct(frame, "report")
    assert "estado" in str(info.value)


def test_missing_required_key_in_a_dict_row_raises_too():
    row = ok_row()
    del row["position_tier"]
    with pytest.raises(OutputSchemaError, match="position_tier"):
        validate_and_correct(row, "correct")


def test_mode_off_does_not_even_look_at_the_columns():
    frame = pd.DataFrame({"unrelated": [1]})
    out, _ = validate_and_correct(frame, "off")
    assert out is frame


def test_extra_and_optional_columns_are_ignored():
    frame = pd.DataFrame([ok_row()]).drop(columns=["confiabilidad", "confiabilidad_manzana", "position_source"])
    frame["whatever"] = "zzz"
    out, report = validate_and_correct(frame, "correct")
    assert report.violations == {} and list(out.columns) == list(frame.columns)


def test_wrong_input_type_is_rejected():
    with pytest.raises(TypeError):
        validate_and_correct([1, 2, 3], "report")


def test_correct_on_a_frame_keeps_the_index_dtypes_and_untouched_rows():
    frame = pd.DataFrame([ok_row(), ok_row(confianza=3.0, lat=40.0), sin_match_row()], index=[10, 20, 30])
    before = frame.copy()
    out, report = validate_and_correct(frame, "correct")
    pd.testing.assert_frame_equal(frame, before)  # input not mutated
    assert list(out.index) == [10, 20, 30]
    assert out["confianza"].dtype == "float64" and out["lat"].dtype == "float64"
    assert out.loc[20, "confianza"] == 1.0 and pd.isna(out.loc[20, "lat"]) and pd.isna(out.loc[20, "lon"])
    pd.testing.assert_series_equal(out.loc[10], frame.loc[10])
    pd.testing.assert_series_equal(out.loc[30], frame.loc[30])
    assert {c.row for c in report.corrections} == {20}
    assert report.rows_corrected == 1 and report.rows_with_violations == 1


def test_correct_twice_on_a_frame_is_stable():
    frame = pd.DataFrame([ok_row(confianza=3.0), ok_row(lat=40.0), ok_row(numero_predial_nacional="12")])
    once, _ = validate_and_correct(frame, "correct")
    twice, report = validate_and_correct(once, "correct")
    pd.testing.assert_frame_equal(once, twice)
    assert report.violations == {}


def test_report_to_dict_is_json_serialisable():
    import json

    _, report = validate_and_correct(pd.DataFrame([ok_row(confianza=3.0)]), "correct")
    payload = report.to_dict()
    assert json.loads(json.dumps(payload))["violations"] == {"confidence_range": 1}
    assert payload["corrected"] == {"confidence_range": 1}
