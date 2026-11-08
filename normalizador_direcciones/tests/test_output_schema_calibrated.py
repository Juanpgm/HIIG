"""Output schema: vocabulary and ranges of the calibrated-decision columns (optional, null on non-OK rows)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cali_address import output_schema as osch
from cali_address.output_schema import NormalizedRecord, validate_and_correct

NPN = "760010100000000900001000000000"


def frame(**over) -> pd.DataFrame:
    base = {
        "estado": ["OK", "SIN_MATCH"], "motivo": ["", "x"], "direccion_normalizada": ["A", None],
        "fuente_normalizacion": ["catastro", None], "numero_predial_nacional": [NPN, None],
        "manzana": ["760010100000000900001", None], "lat": [3.4, None], "lon": [-76.5, None],
        "confianza": [0.9, 0.5], "nivel_precision": ["predio", None], "position_tier": ["predio", "none"],
        "fidelity_risk": ["none", "none"],
        "confiabilidad_calibrada": [0.98, np.nan], "banda_confiabilidad": ["alta", None],
        "motivos_confiabilidad": ["+a", None], "p_predio_cal": [0.98, np.nan], "p_manzana_cal": [0.99, np.nan],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_band_vocabulary_is_pinned():
    assert osch.BANDAS == ("alta", "media", "baja")
    assert "banda_vocab" in osch.RULES


def test_valid_calibrated_frame_has_no_violations():
    assert validate_and_correct(frame(), "report")[1].violations == {}


def test_nan_in_the_float_columns_of_a_non_ok_row_is_null_not_a_violation():
    report = validate_and_correct(frame(), "report")[1]
    assert "confidence_nan" not in report.violations


@pytest.mark.parametrize("column", ["confiabilidad_calibrada", "p_predio_cal", "p_manzana_cal"])
@pytest.mark.parametrize("value", [1.2, -0.1, np.inf])
def test_probability_outside_unit_interval_is_a_violation_and_is_clamped(column, value):
    data = frame(**{column: [value, np.nan]})
    report = validate_and_correct(data, "report")[1]
    assert report.violations.get("confidence_range") == 1
    fixed, fixed_report = validate_and_correct(data, "correct")
    assert fixed.loc[0, column] in (0.0, 1.0) and fixed_report.corrected["confidence_range"] == 1


def test_unknown_band_is_reported_and_left_alone():
    data = frame(banda_confiabilidad=["excelente", None])
    out, report = validate_and_correct(data, "correct")
    assert report.violations.get("banda_vocab") == 1
    assert out.loc[0, "banda_confiabilidad"] == "excelente"


def test_null_band_is_fine_and_a_frame_without_the_columns_is_unaffected():
    plain = frame().drop(columns=["confiabilidad_calibrada", "banda_confiabilidad", "motivos_confiabilidad",
                                  "p_predio_cal", "p_manzana_cal"])
    assert validate_and_correct(plain, "report")[1].violations == {}


def test_record_model_accepts_the_new_optional_fields_and_rejects_a_bad_band():
    row = {"estado": "OK", "motivo": "", "direccion_normalizada": "A", "fuente_normalizacion": "catastro",
           "numero_predial_nacional": NPN, "manzana": "M", "lat": 3.4, "lon": -76.5, "confianza": 0.9,
           "nivel_precision": "predio", "position_tier": "predio", "fidelity_risk": "none",
           "confiabilidad_calibrada": 0.98, "banda_confiabilidad": "alta", "motivos_confiabilidad": "+a",
           "p_predio_cal": 0.98, "p_manzana_cal": 0.99}
    record = NormalizedRecord.model_validate(row)
    assert record.banda_confiabilidad == "alta" and record.p_manzana_cal == 0.99
    with pytest.raises(Exception):
        NormalizedRecord.model_validate({**row, "banda_confiabilidad": "excelente"})
    with pytest.raises(Exception):
        NormalizedRecord.model_validate({**row, "p_predio_cal": 1.5})


def test_dict_row_with_nan_probability_is_a_violation_like_the_other_confidence_fields():
    row = {"estado": "SIN_MATCH", "motivo": "", "direccion_normalizada": None, "fuente_normalizacion": None,
           "numero_predial_nacional": None, "manzana": None, "lat": None, "lon": None, "confianza": None,
           "nivel_precision": None, "position_tier": "none", "fidelity_risk": "none",
           "p_predio_cal": float("nan")}
    _, report = validate_and_correct(row, "report")
    assert report.violations.get("confidence_nan") == 1
