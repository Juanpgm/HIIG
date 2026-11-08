"""Opt-in ``src_pcer`` reliability feature: serving mechanism, artifact schema and ``fit_reliability --source-feature``.

Default OFF: every artifact without the flag (production's included) must keep producing exactly the same confidences.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from cali_address import reliability as rel
from cali_address.reliability import FEATURE_NAMES, Reliability, load_reliability
from cali_address.service import normalize_strict

from conftest import make_stub

NPN = "760010100000000900001000000000"  # invented


def _fit_script():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_fit_reliability_src", os.path.join(root, "scripts", "fit_reliability.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload(names, source_flag=None, coef_src=0.0):
    n = len(names)
    coef = [0.0] * n
    coef[names.index("confidence")] = 2.0
    if "src_pcer" in names:
        coef[names.index("src_pcer")] = coef_src
    data = {
        "kind": "logistic", "version": "t", "feature_names": list(names), "means": [0.0] * n, "stds": [1.0] * n,
        "predio": {"coef": coef, "intercept": -1.0}, "manzana": {"coef": coef, "intercept": 0.0},
    }
    if source_flag is not None:
        data["source_feature"] = source_flag
    return data


def _write(tmp_path, data, name="r.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _feats(**over):
    base = {name: 0.0 for name in FEATURE_NAMES}
    base["confidence"] = 0.8
    base.update(over)
    return base


# --------------------------------------------------------------------------- constants and the pure helper
def test_default_feature_contract_is_unchanged():
    assert len(FEATURE_NAMES) == 13 and "src_pcer" not in FEATURE_NAMES
    assert rel.SOURCE_FEATURE == "src_pcer" and rel.SOURCE_FEATURE in rel.OPTIONAL_FEATURE_NAMES


@pytest.mark.parametrize("value,expected", [
    ("nomenclatura_pcer", 1.0), ("nomenclatura_cert", 0.0), ("catastro", 0.0),
    (None, 0.0), (float("nan"), 0.0), ("", 0.0), (" nomenclatura_pcer ", 0.0), ("NOMENCLATURA_PCER", 0.0),
    (5, 0.0), (True, 0.0), (pd.NA, 0.0),
])
def test_source_feature_value(value, expected):
    assert rel.source_feature_value(value) == expected


# --------------------------------------------------------------------------- loading / predicting
def test_production_style_json_without_the_flag_loads_and_ignores_the_feature(tmp_path):
    model = load_reliability(_write(tmp_path, _payload(list(FEATURE_NAMES))))
    assert isinstance(model, Reliability) and "src_pcer" not in model.feature_names
    assert model.predict(_feats()) == model.predict(_feats(src_pcer=1.0))


def test_json_with_the_source_feature_loads_and_uses_it(tmp_path):
    names = list(FEATURE_NAMES) + ["src_pcer"]
    model = load_reliability(_write(tmp_path, _payload(names, source_flag=True, coef_src=1.5)))
    assert model is not None and model.feature_names[-1] == "src_pcer"
    assert model.predict(_feats(src_pcer=1.0))[0] > model.predict(_feats(src_pcer=0.0))[0]


def test_missing_source_feature_value_is_neutral_at_serving(tmp_path):
    names = list(FEATURE_NAMES) + ["src_pcer"]
    model = load_reliability(_write(tmp_path, _payload(names, source_flag=True, coef_src=1.5)))
    assert model.predict(_feats()) == model.predict(_feats(src_pcer=0.0))        # mean is 0.0 here
    assert model.predict(_feats(src_pcer=float("nan"))) == model.predict(_feats())


def test_flag_true_without_the_feature_name_is_rejected(tmp_path):
    assert load_reliability(_write(tmp_path, _payload(list(FEATURE_NAMES), source_flag=True))) is None


def test_duplicated_source_feature_name_is_rejected(tmp_path):
    names = list(FEATURE_NAMES) + ["src_pcer", "src_pcer"]
    assert load_reliability(_write(tmp_path, _payload(names, source_flag=True))) is None


def test_other_unknown_names_are_still_rejected(tmp_path):
    names = list(FEATURE_NAMES) + ["src_other"]
    assert load_reliability(_write(tmp_path, _payload(names))) is None


# --------------------------------------------------------------------------- serving through normalize_strict
def _stub(fuente=None, reliability=None):
    stub = make_stub(
        [{"direccion": "CARRERA 905 # 950-10", "predial": NPN, "manzana": "760010100000000900001"},
         {"direccion": "CALLE 906 # 951-20", "predial": NPN, "manzana": "760010100000000900002"}],
        scores=[[0.9, 0.1], [0.1, 0.9]], threshold=0.5,
    )
    if fuente is not None:
        stub._fuente = np.array(list(fuente), dtype=object)
    stub.reliability = reliability
    return stub


RAWS = ["CARRERA 905 # 950-10", "CL 906 # 951-20"]


def _model(tmp_path, with_src):
    names = list(FEATURE_NAMES) + (["src_pcer"] if with_src else [])
    return load_reliability(_write(tmp_path, _payload(names, source_flag=True if with_src else None, coef_src=1.5)))


def test_sink_and_confidences_are_untouched_for_a_model_without_the_flag(tmp_path):
    sink_a, sink_b = [], []
    a = normalize_strict(_stub(reliability=_model(tmp_path, False)), RAWS, feature_sink=sink_a)
    b = normalize_strict(_stub(fuente=("nomenclatura_pcer", "nomenclatura_cert"), reliability=_model(tmp_path, False)),
                         RAWS, feature_sink=sink_b)
    assert [c for c in a["confiabilidad"]] == [c for c in b["confiabilidad"]]
    assert [c for c in a["confiabilidad_manzana"]] == [c for c in b["confiabilidad_manzana"]]
    assert all(set(f) == set(FEATURE_NAMES) for f in sink_a + sink_b)


def test_pcer_documents_get_the_feature_at_serving(tmp_path):
    sink: list = []
    frame = normalize_strict(_stub(fuente=("nomenclatura_pcer", "nomenclatura_cert"), reliability=_model(tmp_path, True)),
                             RAWS, feature_sink=sink)
    assert [f["src_pcer"] for f in sink] == [1.0, 0.0]
    assert frame["confiabilidad"].iloc[0] > frame["confiabilidad"].iloc[1]


def test_model_with_the_flag_on_docs_without_fuente_gets_zero(tmp_path):
    sink: list = []
    normalize_strict(_stub(reliability=_model(tmp_path, True)), RAWS, feature_sink=sink)
    assert [f["src_pcer"] for f in sink] == [0.0, 0.0]


def test_null_fuente_in_the_selected_doc_gets_zero(tmp_path):
    sink: list = []
    normalize_strict(_stub(fuente=(None, "catastro"), reliability=_model(tmp_path, True)), RAWS, feature_sink=sink)
    assert [f["src_pcer"] for f in sink] == [0.0, 0.0]


# --------------------------------------------------------------------------- fit_reliability --source-feature
def test_source_feature_column_helper():
    fit = _fit_script()
    col = fit.source_feature_column(pd.DataFrame({"fuente": ["nomenclatura_pcer", None, "catastro", float("nan")]}))
    assert col.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_source_feature_column_requires_the_fuente_column():
    fit = _fit_script()
    with pytest.raises(SystemExit):
        fit.source_feature_column(pd.DataFrame({"estado": ["OK"]}))


def _rows_file(tmp_path, with_fuente=True, n=150):
    rng = np.random.default_rng(7)
    frames = []
    for d in ("ds_a", "ds_b", "ds_c"):
        conf = rng.uniform(0.6, 1.0, n)
        hit = rng.uniform(size=n) < 0.05 + 1.2 * (conf - 0.6)     # the raw score over-states the real hit rate
        df = pd.DataFrame({"dataset": d, "estado": "OK", "has_gt": True, "confianza": conf,
                           "correct_predial": hit, "correct_manzana": hit | (rng.uniform(size=n) < 0.5),
                           "motivo": "", "gt_quality": "gold"})
        for name in FEATURE_NAMES:
            df[f"feat_{name}"] = 0.0
        df["feat_confidence"] = conf
        df["feat_struct"] = rng.uniform(0.6, 1.0, n)
        if with_fuente:
            df["fuente"] = np.where(rng.uniform(size=n) < 0.3, "nomenclatura_pcer", "nomenclatura_cert")
        frames.append(df)
    path = tmp_path / "rows.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path)
    return str(path)


def test_cli_flag_records_the_feature_in_the_artifact(tmp_path):
    fit = _fit_script()
    out = tmp_path / "rel.json"
    assert fit.main(["--kind", "logistic", "--rows", _rows_file(tmp_path), "--out", str(out), "--source-feature"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source_feature"] is True
    assert data["feature_names"] == list(FEATURE_NAMES) + ["src_pcer"]
    assert len(data["means"]) == len(data["stds"]) == len(data["predio"]["coef"]) == len(data["manzana"]["coef"]) == 14
    assert load_reliability(str(out)) is not None and math.isfinite(data["predio"]["intercept"])


def test_cli_without_the_flag_writes_the_historical_schema(tmp_path):
    fit = _fit_script()
    out = tmp_path / "rel.json"
    assert fit.main(["--kind", "logistic", "--rows", _rows_file(tmp_path), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "source_feature" not in data and data["feature_names"] == list(FEATURE_NAMES)
    assert len(data["predio"]["coef"]) == 13


def test_cli_flag_without_a_fuente_column_stops_with_a_message(tmp_path):
    fit = _fit_script()
    with pytest.raises(SystemExit):
        fit.main(["--kind", "logistic", "--rows", _rows_file(tmp_path, with_fuente=False),
                  "--out", str(tmp_path / "x.json"), "--source-feature"])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("kind", ["hybrid", "cells"])
def test_cli_flag_is_only_for_the_logistic_kind(tmp_path, kind):
    fit = _fit_script()
    with pytest.raises(SystemExit) as info:
        fit.main(["--kind", kind, "--rows", _rows_file(tmp_path), "--out", str(tmp_path / "x.json"), "--source-feature"])
    assert info.value.code == 2
