"""Optional ``src_catastro`` feature of the calibrated decision layer (invented fixtures).

It exists only in parameter files fitted with ``fit_decision.py --catastro-feature`` and only when at least 30 fit rows
carry it. Every existing artifact (24 features) loads, predicts and fits exactly as before.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_stub
from test_decision_model import NAMES, SIGNS, decision_doc, head, params
from test_fit_decision import synthetic

from cali_address import decision as D
from cali_address.service import normalize_strict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPN_A = "760010100000000900001000000000"
NPN_B = "760010100000000900002000000000"


def _fit():
    spec = importlib.util.spec_from_file_location("script_fit_decision_cat", os.path.join(ROOT, "scripts", "fit_decision.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def with_catastro(**over) -> dict:
    """Parameter document that also lists src_catastro (25 features)."""
    data = params(**over)
    data["feature_names"] = [*NAMES, "src_catastro"]
    data["signs"] = {**SIGNS, "src_catastro": 0}
    data["motivo_codes"] = {**data["motivo_codes"], "src_catastro": "fuente_catastro"}
    data["motivo_reference"] = {**data["motivo_reference"], "src_catastro": 0.0}
    for name in ("predio", "manzana"):
        data[name] = {**data[name], "coef": [*data[name]["coef"], 0.0], "means": [*data[name]["means"], 0.1],
                      "stds": [*data[name]["stds"], 0.2]}
    return data


# ------------------------------------------------------------------------------------------------------- the loader

def test_the_serving_contract_of_existing_artifacts_is_unchanged():
    assert len(D.FEATURE_NAMES) == 24 and "src_catastro" not in D.FEATURE_NAMES
    assert D.OPTIONAL_FEATURE_NAMES == ("src_catastro",)


def test_a_25_feature_file_loads_and_a_24_feature_file_still_does(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(with_catastro()), encoding="utf-8")
    loaded = D.load_calibrated(str(path))
    assert loaded is not None and loaded.feature_names[-1] == "src_catastro" and len(loaded.feature_names) == 25
    path.write_text(json.dumps(params()), encoding="utf-8")
    assert len(D.load_calibrated(str(path)).feature_names) == 24


def test_unknown_names_are_still_rejected(tmp_path):
    data = with_catastro()
    data["feature_names"][-1] = "src_other"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert D.load_calibrated(str(path)) is None


def test_the_feature_moves_the_probability_and_gets_a_motivo():
    data = with_catastro()
    data["predio"]["coef"][-1] = -0.8
    loaded = D.parse_calibrated(json.loads(json.dumps(data)))
    features = {n: 0.1 for n in NAMES}
    plain = loaded.predict({**features, "src_catastro": 0.0})[0]
    cat = loaded.predict({**features, "src_catastro": 1.0})[0]
    assert cat < plain
    assert loaded.motivos("predio", {**features, "src_catastro": 1.0}).startswith("-fuente_catastro")
    assert loaded.predict(features) == loaded.predict({**features, "src_catastro": 0.1})       # missing = the head mean


# ----------------------------------------------------------------------------------------------------- the service

DOCS = [{"direccion": "CL 987 # 654 - 12", "predial": NPN_A, "manzana": "M1"},
        {"direccion": "CL 987 # 654 - 14", "predial": NPN_B, "manzana": "M2"}]


class Recorder:
    """Model double that records the features it is asked about; optionally declares a feature list."""

    version = "rel-test-1"

    def __init__(self, feature_names=None):
        self.seen = []
        if feature_names is not None:
            self.feature_names = tuple(feature_names)

    def predict(self, features):
        self.seen.append(dict(features))
        return 0.99, 0.995

    def motivos(self, head_name, features):
        return ""


def run_with(model, fuente, query_doc=1, sink=None):
    stub = make_stub(DOCS, scores=[[0.1, 0.1]])
    stub._scores = [[0.95 if i == query_doc else 0.1 for i in range(2)]]
    stub._fuente = np.array(fuente, dtype=object)
    stub.calibrated = (model, D.validate_decision(decision_doc()))
    normalize_strict(stub, [DOCS[query_doc]["direccion"]], gazetteer=None, calibration_sink=sink)
    return model.seen


def test_a_model_without_the_feature_sees_exactly_the_24_features():
    seen = run_with(Recorder(), ["nomenclatura_cert", "catastro"])
    assert set(seen[0]) == set(D.FEATURE_NAMES)
    assert run_with(Recorder(NAMES), ["nomenclatura_cert", "catastro"])[0].keys() == seen[0].keys()


def test_a_model_with_the_feature_gets_it_from_the_selected_doc():
    on = run_with(Recorder([*NAMES, "src_catastro"]), ["nomenclatura_cert", "catastro"], query_doc=1)
    off = run_with(Recorder([*NAMES, "src_catastro"]), ["nomenclatura_cert", "catastro"], query_doc=0)
    assert on[0]["src_catastro"] == 1.0 and off[0]["src_catastro"] == 0.0 and set(on[0]) == {*NAMES, "src_catastro"}


def test_the_feature_is_zero_for_other_sources_missing_fuente_and_unseen_values():
    for fuente in (["nomenclatura_cert", "nomenclatura_pcer"], ["x", None], ["", "CATASTRO"]):
        seen = run_with(Recorder([*NAMES, "src_catastro"]), fuente, query_doc=1)
        assert seen[0]["src_catastro"] == 0.0, fuente
    stub = make_stub(DOCS, scores=[[0.1, 0.95]])                                  # production-style docs: no `fuente`
    model = Recorder([*NAMES, "src_catastro"])
    stub.calibrated = (model, D.validate_decision(decision_doc()))
    normalize_strict(stub, [DOCS[1]["direccion"]], gazetteer=None)
    assert model.seen[0]["src_catastro"] == 0.0


def test_the_calibration_sink_always_carries_the_feature():
    sink: list = []
    stub = make_stub(DOCS, scores=[[0.1, 0.95]])
    stub._fuente = np.array(["nomenclatura_cert", "catastro"], dtype=object)
    stub.calibrated = None
    normalize_strict(stub, [DOCS[1]["direccion"]], gazetteer=None, calibration_sink=sink, ambiguity_delta=0.0)
    assert sink[0]["src_catastro"] == 1.0 and set(D.FEATURE_NAMES) <= set(sink[0])


# ------------------------------------------------------------------------------------------------------- the fit

def table_with_catastro(n_catastro: int, n=2000, seed=3) -> pd.DataFrame:
    table = synthetic(n=n, seed=seed)
    rng = np.random.default_rng(11)
    flag = np.zeros(n)
    flag[rng.choice(n, n_catastro, replace=False)] = 1.0
    table["src_catastro"] = flag
    table["emits_manzana"] = True
    return table


FIT_ARGS = dict(tag="t", artifacts_dir="x", fit_date="2026-10-31T00:00:00")


def test_the_feature_enters_only_with_the_flag_and_enough_rows():
    fd = _fit()
    table = table_with_catastro(40)
    default = fd.fit_tables(table, **FIT_ARGS)
    assert default["reliability"]["feature_names"] == NAMES                                # flag off: nothing changes
    included = fd.fit_tables(table, catastro_feature=True, **FIT_ARGS)
    names = included["reliability"]["feature_names"]
    assert names == [*NAMES, "src_catastro"] and included["reliability"]["signs"]["src_catastro"] == 0
    assert included["reliability"]["motivo_codes"]["src_catastro"] == "fuente_catastro"
    assert included["report"]["catastro_feature"] == {"requested": True, "rows_with_feature": 40, "min_rows": 30, "included": True}
    assert len(included["reliability"]["predio"]["coef"]) == 25
    assert D.parse_calibrated(json.loads(json.dumps(included["reliability"]))).feature_names[-1] == "src_catastro"


def test_fewer_than_30_rows_keep_the_24_feature_artifact_exactly():
    fd = _fit()
    plain = fd.fit_tables(synthetic(), **FIT_ARGS)
    for n_rows in (0, 1, 29):
        out = fd.fit_tables(table_with_catastro(n_rows), catastro_feature=True, **FIT_ARGS)
        assert out["reliability"] == plain["reliability"], n_rows
        assert out["decision"] == plain["decision"], n_rows
        assert out["report"]["catastro_feature"] == {"requested": True, "rows_with_feature": n_rows, "min_rows": 30, "included": False}
    assert fd.fit_tables(table_with_catastro(30), catastro_feature=True, **FIT_ARGS)["report"]["catastro_feature"]["included"]


def test_a_table_without_the_column_is_not_an_error_when_the_flag_is_on():
    fd = _fit()
    out = fd.fit_tables(synthetic(), catastro_feature=True, **FIT_ARGS)
    assert out["reliability"]["feature_names"] == NAMES
    assert out["report"]["catastro_feature"]["rows_with_feature"] == 0


def test_the_extra_column_never_changes_a_default_fit():
    fd = _fit()
    assert fd.fit_tables(table_with_catastro(500), **FIT_ARGS)["reliability"] == fd.fit_tables(synthetic(), **FIT_ARGS)["reliability"]


def test_a_feature_present_in_every_row_is_a_constant_and_gets_no_weight():
    """All rows from the cadastre: the column is constant (std 1, frozen at 0): harmless, never a crash."""
    fd = _fit()
    table = synthetic()
    table["src_catastro"] = 1.0
    out = fd.fit_tables(table, catastro_feature=True, **FIT_ARGS)
    assert out["reliability"]["feature_names"] == NAMES              # the other class has no 30 rows either: not included


def test_the_fitted_25_feature_file_serves_through_the_pipeline(tmp_path):
    fd = _fit()
    out = fd.fit_tables(table_with_catastro(60), catastro_feature=True, **FIT_ARGS)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(out["reliability"]), encoding="utf-8")
    loaded = D.load_calibrated(str(path))
    assert loaded is not None
    stub = make_stub(DOCS, scores=[[0.1, 0.95]])
    stub._fuente = np.array(["nomenclatura_cert", "catastro"], dtype=object)
    decision = copy.deepcopy(out["decision"])
    stub.calibrated = (loaded, D.validate_decision(decision))
    frame = normalize_strict(stub, [DOCS[1]["direccion"]], gazetteer=None)
    assert frame.loc[0, "fuente"] == "catastro" and "p_predio_cal" in frame.columns
