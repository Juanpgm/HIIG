"""``fuente`` / ``estado_nomenclatura`` of the selected document exposed by ``normalize_strict``.

The columns exist only when the cadastral docs carry them (new nomenclature indexes); with a production-style docs
table the output must stay exactly the historical one (no new columns at all).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cali_address import output_schema as osch
from cali_address.api.schemas import ResultRow
from cali_address.service import OUTPUT_COLUMNS, normalize_strict

from conftest import make_stub

NPN = "760010100000000900001000000000"  # 30 digits, invented
RAWS = ["CARRERA 905 # 950-10", "CL 906 # 951-20", "TEXTO SIN DIRECCION", ""]


def _stub(fuente=("nomenclatura_cert", "nomenclatura_pcer"), estado=("vigente", "pendiente")):
    stub = make_stub(
        [{"direccion": "CARRERA 905 # 950-10", "predial": NPN, "manzana": "760010100000000900001"},
         {"direccion": "CALLE 906 # 951-20", "predial": NPN, "manzana": "760010100000000900002"}],
        scores=[[0.9, 0.1], [0.1, 0.9], [0.1, 0.1], [0.1, 0.1]], threshold=0.5,
    )
    if fuente is not None:
        stub._fuente = np.array(list(fuente), dtype=object)
    if estado is not None:
        stub._estado_nomenclatura = np.array(list(estado), dtype=object)
    return stub


def test_columns_are_appended_last_with_the_selected_docs_values():
    frame = normalize_strict(_stub(), RAWS)
    assert list(frame.columns) == OUTPUT_COLUMNS + ["fuente", "estado_nomenclatura"]
    assert frame["fuente"].tolist()[:2] == ["nomenclatura_cert", "nomenclatura_pcer"]
    assert frame["estado_nomenclatura"].tolist()[:2] == ["vigente", "pendiente"]


def test_unmatched_rows_get_null():
    frame = normalize_strict(_stub(), RAWS)
    assert frame["estado"].tolist()[2:] != ["OK", "OK"]
    assert frame["fuente"].isna().tolist()[2:] == [True, True]
    assert frame["estado_nomenclatura"].isna().tolist()[2:] == [True, True]


def test_no_fuente_in_the_docs_means_no_new_columns():
    frame = normalize_strict(_stub(fuente=None, estado=None), RAWS)
    assert list(frame.columns) == OUTPUT_COLUMNS


def test_only_the_available_column_is_added():
    frame = normalize_strict(_stub(estado=None), RAWS)
    assert list(frame.columns) == OUTPUT_COLUMNS + ["fuente"]


def test_null_fuente_values_in_the_docs_stay_null():
    frame = normalize_strict(_stub(fuente=("catastro", None)), RAWS)
    assert frame["fuente"].tolist()[0] == "catastro" and pd.isna(frame["fuente"].iloc[1])


def test_empty_input_keeps_the_columns():
    frame = normalize_strict(_stub(), [])
    assert len(frame) == 0 and list(frame.columns) == OUTPUT_COLUMNS + ["fuente", "estado_nomenclatura"]


def test_existing_columns_are_unchanged_by_the_new_ones():
    with_src = normalize_strict(_stub(), RAWS)
    without = normalize_strict(_stub(fuente=None, estado=None), RAWS)
    pd.testing.assert_frame_equal(with_src[OUTPUT_COLUMNS], without)


def test_unknown_fuente_value_is_flagged_in_the_report_and_never_corrected():
    stats: dict = {}
    frame = normalize_strict(_stub(fuente=("nomenclatura_cert", "mystery")), RAWS, validate_outputs="correct", stats=stats)
    assert stats["output_validation"]["violations"] == {"fuente_vocab": 1}
    assert frame["fuente"].tolist()[1] == "mystery"  # report-only: left alone


def test_valid_fuente_values_produce_no_violation():
    stats: dict = {}
    normalize_strict(_stub(fuente=("catastro", "nomenclatura_pcer")), RAWS, validate_outputs="report", stats=stats)
    assert "fuente_vocab" not in stats["output_validation"]["violations"]


def test_fuente_rule_is_registered_and_vocab_is_exposed():
    assert "fuente_vocab" in osch.RULES
    assert osch.FUENTES == ("nomenclatura_cert", "nomenclatura_pcer", "catastro")


@pytest.mark.parametrize("bad", ["", " ", "CATASTRO", 5, True])
def test_record_rejects_a_value_outside_the_vocabulary(bad):
    row = {"estado": "SIN_MATCH", "motivo": None, "direccion_normalizada": None, "fuente_normalizacion": None,
           "numero_predial_nacional": None, "manzana": None, "lat": None, "lon": None, "confianza": None,
           "nivel_precision": None, "position_tier": "none", "fidelity_risk": "none", "fuente": bad}
    with pytest.raises(Exception) as info:
        osch.NormalizedRecord(**row)
    assert "fuente_vocab" in str(info.value)


def test_record_accepts_fuente_null_absent_and_valid():
    base = {"estado": "SIN_MATCH", "motivo": None, "direccion_normalizada": None, "fuente_normalizacion": None,
            "numero_predial_nacional": None, "manzana": None, "lat": None, "lon": None, "confianza": None,
            "nivel_precision": None, "position_tier": "none", "fidelity_risk": "none"}
    assert osch.NormalizedRecord(**base).fuente is None
    assert osch.NormalizedRecord(**base, fuente=None, estado_nomenclatura=None).fuente is None
    rec = osch.NormalizedRecord(**base, fuente="nomenclatura_pcer", estado_nomenclatura="vigente")
    assert rec.fuente == "nomenclatura_pcer" and rec.estado_nomenclatura == "vigente"


def _api_record(frame):
    from cali_address.service import results_to_records

    return results_to_records(frame)[0]


def test_api_row_model_passes_the_optional_fields_through_when_present():
    rec = _api_record(normalize_strict(_stub(), RAWS))
    assert rec["fuente"] == "nomenclatura_cert" and rec["estado_nomenclatura"] == "vigente"
    dumped = ResultRow(**rec).model_dump()
    assert dumped["fuente"] == "nomenclatura_cert" and dumped["estado_nomenclatura"] == "vigente"


def test_api_row_model_adds_no_key_for_production_style_docs():
    rec = _api_record(normalize_strict(_stub(fuente=None, estado=None), RAWS))
    dumped = ResultRow(**rec).model_dump()
    assert "fuente" not in dumped and "estado_nomenclatura" not in dumped
    assert set(dumped) == set(ResultRow.model_fields)
