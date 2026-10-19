"""Reliability v5: logistic calibrator fitted on text-consistent ("gold") labels only.

Covers the fit-side plumbing (``--gold-only``, several ``--rows`` files, artifact metadata),
the eval-side plumbing (``gt_quality`` carried into the rows, label columns for files that
lack them) and the serving contract of the resulting artifact (it is a plain ``logistic``
artifact, so the current loader serves it unchanged).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fr = _load_script("fit_reliability")

from cali_address.reliability import FEATURE_NAMES, P_MAX, P_MIN, Reliability, load_reliability  # noqa: E402


def _rows(n=240, seed=3, quality=None, datasets=("a", "b", "c")):
    """Synthetic eval rows where a small ``margen_predio`` means a wrong predio."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"feat_{name}": np.zeros(n) for name in FEATURE_NAMES})
    df["feat_confidence"] = rng.uniform(0.7, 1.0, n)
    df["feat_struct"] = rng.uniform(0.6, 1.0, n)
    df["feat_margen_manzana"] = rng.uniform(0.0, 0.3, n)
    df["feat_margen_predio"] = rng.uniform(0.0, 0.3, n)
    df["feat_log_n_predios"] = (rng.uniform(size=n) < 0.2) * math.log(2)
    df["dataset"] = np.resize(np.array(datasets, dtype=object), n)
    df["estado"] = "OK"
    df["has_gt"] = True
    df["confianza"] = df["feat_confidence"]
    p = 1 / (1 + np.exp(-(df["feat_margen_predio"].to_numpy() - 0.05) * 40))
    df["correct_predial"] = (rng.uniform(size=n) < p).astype(float)
    df["correct_manzana"] = np.maximum(df["correct_predial"], (rng.uniform(size=n) < 0.9).astype(float))
    df["motivo"] = ""
    if quality is not None:
        df["gt_quality"] = np.resize(np.array(quality, dtype=object), n)
    return df


# ---------------------------------------------------------------------------
# load_training_rows
# ---------------------------------------------------------------------------
def test_gold_only_keeps_only_gold_rows(tmp_path):
    path = tmp_path / "rows.parquet"
    _rows(quality=["gold", "noisy", "no_gt", None]).to_parquet(path, index=False)
    out = fr.load_training_rows(str(path), gold_only=True)
    assert len(out) == 60 and set(out["gt_quality"]) == {"gold"}


def test_without_gold_only_all_labelled_rows_are_kept(tmp_path):
    path = tmp_path / "rows.parquet"
    _rows(quality=["gold", "noisy"]).to_parquet(path, index=False)
    assert len(fr.load_training_rows(str(path))) == 240


def test_gold_only_without_quality_column_refuses(tmp_path):
    path = tmp_path / "rows.parquet"
    _rows().to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="gt_quality"):
        fr.load_training_rows(str(path), gold_only=True)


def test_gold_only_with_no_gold_row_refuses(tmp_path):
    path = tmp_path / "rows.parquet"
    _rows(quality=["noisy"]).to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="no labelled"):
        fr.load_training_rows(str(path), gold_only=True)


def test_several_row_files_are_concatenated(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _rows(n=90, quality=["gold"]).to_parquet(a, index=False)
    _rows(n=60, seed=9, quality=["gold", "noisy"]).to_parquet(b, index=False)
    out = fr.load_training_rows([str(a), str(b)], gold_only=True)
    assert len(out) == 90 + 30
    assert out.index.equals(pd.RangeIndex(len(out)))


def test_rows_label_names_every_file():
    assert fr.rows_label("x/a.parquet") == "a.parquet"
    assert fr.rows_label(["x/a.parquet", "y/b.parquet"]) == "a.parquet, b.parquet"


# ---------------------------------------------------------------------------
# logistic gold fit end to end (main)
# ---------------------------------------------------------------------------
def _fit(tmp_path, *extra, name="rel.json", quality=("gold", "gold", "noisy")):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _rows(quality=list(quality)).to_parquet(a, index=False)
    _rows(n=120, seed=11, quality=["gold"]).to_parquet(b, index=False)
    out = tmp_path / name
    code = fr.main(["--kind", "logistic", "--rows", str(a), str(b), "--out", str(out), *extra])
    return code, out


def test_gold_logistic_writes_a_v5_artifact_the_current_loader_serves(tmp_path):
    code, out = _fit(tmp_path, "--gold-only")
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "logistic"
    assert payload["version"].startswith("reliability-v5-gold-")
    assert payload["label_quality"] == "gold"
    assert payload["n_train"] == 160 + 120
    assert payload["source_rows"] == "a.parquet, b.parquet"
    assert len(payload["code_fingerprint"]) == 12 and all(c in "0123456789abcdef" for c in payload["code_fingerprint"])
    assert "leave-one-dataset-out" in payload["cv_protocol"]
    model = load_reliability(str(out))
    assert isinstance(model, Reliability) and model.version == payload["version"]


def test_default_fit_keeps_the_v2_version_and_all_labels(tmp_path):
    code, out = _fit(tmp_path)
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"].startswith("reliability-v2-") and payload["label_quality"] == "all"
    assert payload["n_train"] == 240 + 120


def test_gold_fit_is_deterministic(tmp_path):
    _, out1 = _fit(tmp_path, "--gold-only", name="r1.json")
    _, out2 = _fit(tmp_path, "--gold-only", name="r2.json")
    p1, p2 = (json.loads(p.read_text(encoding="utf-8")) for p in (out1, out2))
    for key in ("predio", "manzana", "means", "stds", "cv"):
        assert p1[key] == p2[key]


def test_gold_fit_survives_a_single_class_dataset(tmp_path):
    a = tmp_path / "a.parquet"
    df = _rows(quality=["gold"])
    df.loc[df["dataset"] == "c", ["correct_predial", "correct_manzana"]] = 1.0
    df.to_parquet(a, index=False)
    out = tmp_path / "rel.json"
    assert fr.main(["--kind", "logistic", "--rows", str(a), "--out", str(out), "--gold-only"]) in (0, 2)
    if out.exists():
        assert isinstance(load_reliability(str(out)), Reliability)


def test_gold_fit_learns_the_predio_margin_signal(tmp_path):
    _, out = _fit(tmp_path, "--gold-only")
    model = load_reliability(str(out))
    base = {n: 0.0 for n in FEATURE_NAMES} | {"confidence": 0.9, "struct": 0.8, "margen_manzana": 0.1}
    tight = model.predict(base | {"margen_predio": 0.0})[0]
    loose = model.predict(base | {"margen_predio": 0.3})[0]
    assert loose > tight


# ---------------------------------------------------------------------------
# serving contract of the v5 artifact
# ---------------------------------------------------------------------------
@pytest.fixture()
def v5_model(tmp_path):
    _, out = _fit(tmp_path, "--gold-only")
    return load_reliability(str(out))


@pytest.mark.parametrize("features", [
    {},
    {n: None for n in FEATURE_NAMES},
    {n: float("nan") for n in FEATURE_NAMES},
    {n: "not-a-number" for n in FEATURE_NAMES},
    {n: float("inf") for n in FEATURE_NAMES},
    {n: float("-inf") for n in FEATURE_NAMES},
    {"confidence": 0.9, "unseen_feature": 7.0, "nivel": "categoria_nueva"},
])
def test_v5_predictions_are_bounded_and_ordered_for_degenerate_inputs(v5_model, features):
    pp, pm = v5_model.predict(features)
    assert all(math.isfinite(v) for v in (pp, pm))
    assert P_MIN <= pp <= pm <= P_MAX


def test_v5_is_non_decreasing_in_the_raw_score(v5_model):
    base = {n: 0.0 for n in FEATURE_NAMES} | {"struct": 0.8, "margen_manzana": 0.1, "margen_predio": 0.1}
    values = [v5_model.predict(base | {"confidence": c})[0] for c in np.linspace(0.0, 1.5, 31)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_v5_artifact_with_a_foreign_feature_list_is_rejected(tmp_path):
    _, out = _fit(tmp_path, "--gold-only")
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["feature_names"] = [*payload["feature_names"][:-1], "feature_from_the_future"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(bad)) is None


def test_v5_artifact_with_an_unknown_kind_is_rejected(tmp_path):
    _, out = _fit(tmp_path, "--gold-only")
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["kind"] = "logistic-v9"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(bad)) is None


def test_v5_artifact_with_truncated_coefficients_is_rejected(tmp_path):
    _, out = _fit(tmp_path, "--gold-only")
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["predio"]["coef"] = payload["predio"]["coef"][:-1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(bad)) is None


# ---------------------------------------------------------------------------
# eval_strict plumbing
# ---------------------------------------------------------------------------
es = _load_script("eval_strict")


def test_label_columns_are_carried_when_present():
    df = pd.DataFrame({"gt_quality": ["gold", "noisy"], "x": [1, 2]})
    assert es.passthrough_label_columns(df) == {"gt_quality": ["gold", "noisy"]}


def test_label_columns_are_absent_when_the_input_has_none():
    assert es.passthrough_label_columns(pd.DataFrame({"x": [1]})) == {}


def test_ensure_label_columns_fills_a_pairs_file():
    pairs = pd.DataFrame({"raw_address": ["a", "b"], "dataset": ["d", "d"],
                          "gt_manzana": ["1", None], "gt_predial": ["11", None]})
    out = es.ensure_label_columns(pairs, label_quality="gold")
    assert out["has_gt"].tolist() == [True, False]
    assert out["gt_lat"].isna().all() and out["gt_lon"].isna().all()
    assert out["gt_quality"].tolist() == ["gold", "no_gt"]
    assert "has_gt" not in pairs.columns  # input untouched


def test_ensure_label_columns_leaves_a_complete_file_alone():
    full = pd.DataFrame({"raw_address": ["a"], "dataset": ["d"], "gt_manzana": ["1"], "gt_predial": ["11"],
                         "has_gt": [False], "gt_lat": [3.4], "gt_lon": [-76.5], "gt_quality": ["noisy"]})
    out = es.ensure_label_columns(full, label_quality=None)
    pd.testing.assert_frame_equal(out, full)


def test_ensure_label_columns_refuses_to_override_an_existing_quality():
    full = pd.DataFrame({"raw_address": ["a"], "dataset": ["d"], "gt_manzana": ["1"], "gt_predial": ["11"],
                         "gt_quality": ["noisy"]})
    with pytest.raises(ValueError, match="gt_quality"):
        es.ensure_label_columns(full, label_quality="gold")


def test_ensure_label_columns_without_any_ground_truth_column_refuses():
    with pytest.raises(KeyError):
        es.ensure_label_columns(pd.DataFrame({"raw_address": ["a"], "dataset": ["d"]}), label_quality=None)
