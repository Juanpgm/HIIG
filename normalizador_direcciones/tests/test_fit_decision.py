"""``scripts/fit_decision.py``: fits the two sign-constrained heads and the thresholds of the calibrated decision layer.

Tiny synthetic data only (no real addresses). The produced files must load with the serving loaders.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_stub

from cali_address import decision as D

NAMES = list(D.FEATURE_NAMES)
NPN_A = "760010100000000900001000000000"
NPN_B = "760010100000000900002000000000"


def _fit():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_fit_decision", os.path.join(root, "scripts", "fit_decision.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(n=2000, seed=3, groups=500) -> pd.DataFrame:
    """Rows whose lot / manzana correctness depends on a few signed features (invented)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(0.0, index=range(n), columns=NAMES)
    X["confidence"] = rng.uniform(0.7, 1.0, n)
    X["margen_manzana"] = rng.uniform(0, 0.3, n)
    X["margen_predio"] = rng.uniform(0, 0.3, n)
    X["raw_margin_lot"] = rng.uniform(0, 0.3, n)
    X["soft_n"] = rng.integers(0, 3, n).astype(float)
    X["src_pcer"] = (rng.uniform(size=n) < 0.3).astype(float)
    X["nivel_manzana"] = (rng.uniform(size=n) < 0.2).astype(float)
    X["cat_verified"] = (rng.uniform(size=n) < 0.8).astype(float)
    logit_lot = 7.0 + 8 * X["raw_margin_lot"] - 3.5 * X["soft_n"] - 1.0 * X["src_pcer"]
    logit_manzana = 7.0 + 6 * X["margen_manzana"] - 3.0 * X["soft_n"]
    X["y_lot"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_lot))).astype(float)
    X["y_manzana"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_manzana))).astype(float)
    X["emits_predial"] = rng.uniform(size=n) < 0.8
    X["group"] = [f"g{i}" for i in rng.integers(0, groups, n)]
    return X


# --------------------------------------------------------------------------- signs
def test_signs_cover_exactly_the_serving_features_in_order():
    fd = _fit()
    assert list(fd.SIGNS) == NAMES
    assert set(fd.MOTIVO_CODES) == set(NAMES)


# --------------------------------------------------------------------------- tau
def test_choose_tau_returns_the_smallest_value_reaching_the_target():
    fd = _fit()
    p = np.array([0.72, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99])
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    tau, reached, curve = fd.choose_tau(p, y, target=0.99)
    assert reached and tau == pytest.approx(0.805) and curve[0]["tau"] == 0.70      # 0.8 is wrong: the grid steps by 0.005


def test_choose_tau_is_the_floor_when_everything_is_right():
    fd = _fit()
    tau, reached, _ = fd.choose_tau(np.array([0.75, 0.9]), np.array([1, 1]), target=0.99)
    assert reached and tau == 0.70


def test_choose_tau_flags_an_unreachable_target_and_returns_the_top_of_the_grid():
    fd = _fit()
    tau, reached, _ = fd.choose_tau(np.array([0.8, 0.9, 0.99]), np.array([0, 1, 0]), target=0.99)
    assert not reached and tau == pytest.approx(0.995)


def test_choose_tau_ignores_nan_probabilities_and_rejects_empty_input():
    fd = _fit()
    tau, reached, _ = fd.choose_tau(np.array([np.nan, 0.9, 0.95]), np.array([0, 1, 1]), target=0.99)
    assert reached and tau == pytest.approx(0.70)
    with pytest.raises(ValueError):
        fd.choose_tau(np.array([]), np.array([]), target=0.99)
    with pytest.raises(ValueError):
        fd.choose_tau(np.array([np.nan]), np.array([1]), target=0.99)


def test_choose_tau_boundary_is_inclusive():
    fd = _fit()
    tau, reached, _ = fd.choose_tau(np.array([0.85, 0.5]), np.array([1, 0]), target=1.0, floor=0.85)
    assert reached and tau == pytest.approx(0.85)                   # p == tau counts the row
    tau, _, _ = fd.choose_tau(np.array([0.85, 0.85, 0.5]), np.array([1, 1, 0]), target=1.0, floor=0.80)
    assert tau == pytest.approx(0.80)


# --------------------------------------------------------------------------- the signed logistic
def test_signed_fit_respects_the_signs_and_recovers_the_direction():
    fd = _fit()
    t = synthetic()
    sel = t["y_lot"].notna()
    head = fd.fit_head(t.loc[sel, NAMES].to_numpy(float), t.loc[sel, "y_lot"].to_numpy(float), list(fd.SIGNS.values()), C=1.0)
    coef = dict(zip(NAMES, head["coef"]))
    for name, sign in fd.SIGNS.items():
        assert sign == 0 or coef[name] * sign >= 0, name
    assert coef["raw_margin_lot"] > 0.1 and coef["soft_n"] < -0.1


def test_a_binary_feature_with_little_support_is_frozen_at_zero():
    fd = _fit()
    t = synthetic()
    t["relaxed_plate"] = 0.0
    t.loc[t.index[:5], "relaxed_plate"] = 1.0                       # 5 rows only
    t.loc[t.index[:5], "y_lot"] = 0.0
    head = fd.fit_head(t[NAMES].to_numpy(float), t["y_lot"].to_numpy(float), list(fd.SIGNS.values()), C=1.0)
    assert head["coef"][NAMES.index("relaxed_plate")] == 0.0 and NAMES.index("relaxed_plate") in head["frozen"]


def test_constant_columns_get_a_unit_std_and_a_zero_coefficient():
    fd = _fit()
    t = synthetic()
    head = fd.fit_head(t[NAMES].to_numpy(float), t["y_lot"].to_numpy(float), list(fd.SIGNS.values()), C=1.0)
    j = NAMES.index("cross_missing")
    assert head["stds"][j] == 1.0 and head["coef"][j] == 0.0


def test_fit_head_rejects_one_class_and_non_finite_input():
    fd = _fit()
    t = synthetic()
    X = t[NAMES].to_numpy(float)
    with pytest.raises(ValueError, match="both classes"):
        fd.fit_head(X, np.ones(len(X)), list(fd.SIGNS.values()), C=1.0)
    bad = X.copy()
    bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fd.fit_head(bad, t["y_lot"].to_numpy(float), list(fd.SIGNS.values()), C=1.0)
    bad[3, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fd.fit_head(bad, t["y_lot"].to_numpy(float), list(fd.SIGNS.values()), C=1.0)


def test_fit_head_is_deterministic():
    fd = _fit()
    t = synthetic()
    args = (t[NAMES].to_numpy(float), t["y_lot"].to_numpy(float), list(fd.SIGNS.values()))
    a, b = fd.fit_head(*args, C=0.3), fd.fit_head(*args, C=0.3)
    assert a["coef"] == b["coef"] and a["intercept"] == b["intercept"]


# --------------------------------------------------------------------------- group CV
def test_folds_never_split_a_group():
    fd = _fit()
    groups = np.array([f"g{i % 40}" for i in range(400)])
    for train, test in fd.group_folds(groups, 5):
        assert not set(groups[train]) & set(groups[test])


def test_folds_cover_every_row_once_and_shrink_to_the_number_of_groups():
    fd = _fit()
    groups = np.array(["a", "a", "b", "b", "c"])
    folds = list(fd.group_folds(groups, 5))
    assert len(folds) == 3
    assert sorted(np.concatenate([te for _tr, te in folds]).tolist()) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="groups"):
        list(fd.group_folds(np.array(["a", "a"]), 5))


def test_choose_c_picks_a_value_of_the_grid():
    fd = _fit()
    t = synthetic()
    c = fd.choose_c(t[NAMES].to_numpy(float), t["y_lot"].to_numpy(float), t["group"].to_numpy(), list(fd.SIGNS.values()))
    assert c in fd.C_GRID


# --------------------------------------------------------------------------- end to end on a table
_CACHE: dict = {}


def fit_all(fd, table=None, **kw):
    cacheable = table is None and not kw
    if cacheable and "default" in _CACHE:
        return _CACHE["default"]
    table = synthetic() if table is None else table
    kw.setdefault("fit_date", "2026-01-01T00:00:00")
    result = fd.fit_tables(table, tag="t", artifacts_dir="artifacts/experiments/x", **kw)
    if cacheable:
        _CACHE["default"] = result
    return result


def test_outputs_load_with_the_serving_loaders(tmp_path):
    fd = _fit()
    result = fit_all(fd)
    (tmp_path / "decision.json").write_text(json.dumps(result["decision"]), encoding="utf-8")
    (tmp_path / "reliability_calibrated.json").write_text(json.dumps(result["reliability"]), encoding="utf-8")
    model, decision = D.load_calibrated_pair(str(tmp_path))
    assert model.feature_names == tuple(NAMES) and decision["tau_p"] == result["decision"]["tau_p"]
    pp, pm = model.predict({n: 0.5 for n in NAMES})
    assert 0 < pp <= pm < 1


def test_thresholds_respect_the_floors_and_the_band_structure():
    fd = _fit()
    d = fit_all(fd)["decision"]
    assert d["floor"] == 0.70 and d["confianza_floor"] == 0.70
    assert d["floor"] <= d["tau_m"] <= 0.995 and d["floor"] <= d["tau_p"] <= d["bands"]["alta"] <= 1
    assert d["bands"]["media"] == d["tau_p"]


def test_fit_is_deterministic_byte_for_byte():
    fd = _fit()
    a = fit_all(fd)
    b = fd.fit_tables(synthetic(), tag="t", artifacts_dir="artifacts/experiments/x", fit_date="2026-01-01T00:00:00")
    assert json.dumps(a["decision"], sort_keys=True) == json.dumps(b["decision"], sort_keys=True)
    assert json.dumps(a["reliability"], sort_keys=True) == json.dumps(b["reliability"], sort_keys=True)


def test_every_stored_coefficient_respects_its_sign():
    fd = _fit()
    r = fit_all(fd)["reliability"]
    for head in ("predio", "manzana"):
        for name, coef in zip(r["feature_names"], r[head]["coef"]):
            sign = r["signs"][name]
            assert sign == 0 or coef * sign >= 0


def test_provenance_holds_counts_and_no_row_level_data():
    fd = _fit()
    result = fit_all(fd)
    prov = result["decision"]["fit_provenance"]
    assert prov["n_predio"] > 100 and prov["n_manzana"] > 100 and prov["positives_predio"] > 0
    text = json.dumps(result)
    assert "g12" not in text                                       # no group key / row identifiers leak
    assert result["report"]["n_rows"] == 2000


def test_motivo_reference_is_the_training_mean_for_confidence_and_margins_and_zero_otherwise():
    fd = _fit()
    r = fit_all(fd)["reliability"]
    ref = r["motivo_reference"]
    assert ref["confidence"] == pytest.approx(r["predio"]["means"][NAMES.index("confidence")])
    assert ref["margen_manzana"] == pytest.approx(r["predio"]["means"][NAMES.index("margen_manzana")])
    assert ref["soft_n"] == 0.0 and ref["src_pcer"] == 0.0


def test_rows_below_the_confianza_floor_do_not_set_tau_but_do_feed_the_heads():
    fd = _fit()
    t = synthetic()
    low = t.copy()
    low["confidence"] = 0.5                                        # nothing reaches the floor
    with pytest.raises(ValueError, match="floor"):
        fit_all(fd, low)


def test_unpredicted_target_columns_are_validated():
    fd = _fit()
    t = synthetic().drop(columns=["y_lot"])
    with pytest.raises(ValueError, match="y_lot"):
        fit_all(fd, t)


def test_too_few_labelled_rows_is_a_clear_error():
    fd = _fit()
    with pytest.raises(ValueError, match="rows"):
        fit_all(fd, synthetic(n=20, groups=10))


def test_bands_alta_cutoff_is_configurable_and_validated():
    fd = _fit()
    assert fit_all(fd, band_alta=0.98)["decision"]["bands"]["alta"] == 0.98
    with pytest.raises(ValueError):
        fit_all(fd, band_alta=1.5)


def test_neutral_rule_configuration_is_recorded_in_the_decision():
    fd = _fit()
    n = fit_all(fd, soft_rules=("letra_via", "bis_via"))["decision"]["neutral"]
    assert n == {"soft_rules": ["bis_via", "letra_via"], "max_soft": 99, "gate_fallback": True,
                 "ambiguity_delta": 0.0, "threshold": "confianza_floor"}


# --------------------------------------------------------------------------- table from a pipeline run (stub normalizer)
DECISION = {
    "version": "d", "reliability_version": "r", "tau_p": 0.94, "tau_m": 0.7, "floor": 0.7, "confianza_floor": 0.7,
    "bands": {"alta": 0.97, "media": 0.94}, "neutral": {"soft_rules": ["letra_via"], "max_soft": 99,
                                                        "gate_fallback": True, "ambiguity_delta": 0.0},
}


class FakeCalibrated:
    def predict(self, features):
        raise AssertionError("the calibrated layer must not run while fitting")

    def motivos(self, head, features):
        raise AssertionError("the calibrated layer must not run while fitting")


def pairs_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "raw_address": ["CL 987 # 654 - 12", "CL 987 # 654 - 14", "CL 987 # 654 - 16", "TEXTO SIN DIRECCION", "CL 987 # 654 - 12"],
        "gt_predial": [NPN_A, NPN_B, NPN_A, NPN_A, None],
        "gt_manzana": ["760010100000000900001", "760010100000000900002", "760010100000000900009", "x", "760010100000000900001"],
        "address_key": ["k1", "k2", "k3", "k4", "k5"],
    })


def stub():
    docs = [{"direccion": "CL 987 # 654 - 12", "predial": NPN_A, "manzana": "760010100000000900001"},
            {"direccion": "CL 987 # 654 - 14", "predial": NPN_B, "manzana": "760010100000000900002"},
            {"direccion": "CL 987 # 654 - 16", "predial": NPN_B, "manzana": "760010100000000900002"}]
    scores = [[0.95, 0.1, 0.1], [0.1, 0.95, 0.1], [0.1, 0.1, 0.95], [0.95, 0.1, 0.1]]
    return make_stub(docs, scores=scores)


def test_build_table_labels_and_columns():
    fd = _fit()
    table = fd.build_table(stub(), pairs_frame(), gazetteer=None, soft_rules=("letra_via",))
    assert set(NAMES) <= set(table.columns) and {"y_lot", "y_manzana", "group", "emits_predial"} <= set(table.columns)
    assert len(table) == 4                                          # the unparseable row is not an OK row
    by_group = table.set_index("group")
    key = fd.group_key
    assert by_group.loc[key("k1"), "y_lot"] == 1.0 and by_group.loc[key("k1"), "y_manzana"] == 1.0
    assert by_group.loc[key("k2"), "y_lot"] == 1.0
    assert by_group.loc[key("k3"), "y_lot"] == 0.0 and by_group.loc[key("k3"), "y_manzana"] == 0.0     # right lot is another one
    assert np.isnan(by_group.loc[key("k5"), "y_lot"])               # no GT predial: not scored for the lot head


def test_build_table_switches_a_present_calibrated_mode_off_and_restores_it():
    fd = _fit()
    plain = fd.build_table(stub(), pairs_frame(), gazetteer=None, soft_rules=("letra_via",))
    armed = stub()
    sentinel = (FakeCalibrated(), D.validate_decision(DECISION))
    armed.calibrated = sentinel
    table = fd.build_table(armed, pairs_frame(), gazetteer=None, soft_rules=("letra_via",))
    pd.testing.assert_frame_equal(table, plain)                     # the layer did not run (it would have raised)
    assert armed.calibrated is sentinel
    bare = stub()
    fd.build_table(bare, pairs_frame(), gazetteer=None, soft_rules=("letra_via",))
    assert not hasattr(bare, "calibrated")


def test_build_table_never_stores_address_text():
    fd = _fit()
    table = fd.build_table(stub(), pairs_frame(), gazetteer=None, soft_rules=("letra_via",))
    assert "raw_address" not in table.columns
    assert not any(isinstance(v, str) and "CL 987" in v for col in table.columns for v in table[col])


def test_group_key_is_stable_salted_and_hides_the_value():
    fd = _fit()
    assert fd.group_key("k1") == fd.group_key("k1") and fd.group_key("k1") != fd.group_key("k2")
    assert "k1" not in fd.group_key("k1")


def test_build_table_of_empty_pairs_is_an_empty_table_with_columns():
    fd = _fit()
    table = fd.build_table(stub(), pairs_frame().iloc[0:0], gazetteer=None, soft_rules=("letra_via",))
    assert len(table) == 0 and "y_lot" in table.columns


def test_build_table_requires_the_label_columns():
    fd = _fit()
    with pytest.raises(KeyError, match="gt_predial"):
        fd.build_table(stub(), pairs_frame().drop(columns=["gt_predial", "gt_manzana"]), gazetteer=None)


# --------------------------------------------------------------------------- CLI guards
def test_cli_refuses_the_frozen_split(tmp_path):
    fd = _fit()
    assert fd.main(["--artifacts-dir", str(tmp_path), "--pairs", str(tmp_path / "frozen_test.parquet"),
                    "--out-dir", str(tmp_path / "out"), "--tag", "x"]) == 2
    assert not (tmp_path / "out").exists()


def test_cli_refuses_to_overwrite_existing_parameter_files(tmp_path, monkeypatch):
    fd = _fit()
    out = tmp_path / "out"
    out.mkdir()
    (out / "decision_x.json").write_text("{}", encoding="utf-8")
    pairs = tmp_path / "pairs.parquet"
    pairs_frame().to_parquet(pairs)
    assert fd.main(["--artifacts-dir", str(tmp_path), "--pairs", str(pairs), "--out-dir", str(out), "--tag", "x"]) == 2
    assert (out / "decision_x.json").read_text(encoding="utf-8") == "{}"


# ---------------------------------------------------------------- unconstrained refit (A6 ship rule)
def test_report_carries_the_unconstrained_refit_oof_metrics_next_to_the_constrained_ones():
    fd = _fit()
    heads = fit_all(fd)["report"]["heads"]
    for name in ("predio", "manzana"):
        h = heads[name]
        assert set(h["refit_oof"]) == set(h["oof"])                      # same metric keys, same rows
        assert h["refit_oof"]["n"] == h["oof"]["n"]
        for key in ("ece10", "brier", "auc"):
            assert 0.0 <= h["refit_oof"][key] <= 1.0
        # sign constraints can only cost (or tie) fit quality: the refit is never far better on this synthetic set
        assert h["oof"]["brier"] <= h["refit_oof"]["brier"] + 0.01


def test_the_refit_is_unconstrained_so_it_can_pick_a_wrong_signed_coefficient_the_served_head_cannot():
    fd = _fit()
    table = synthetic().copy()
    table["soft_n"] = table["y_lot"].fillna(0).to_numpy() * 2.0 + table["soft_n"]     # a "-" feature made to correlate "+"
    result = fit_all(fd, table=table)
    soft = NAMES.index("soft_n")
    assert result["reliability"]["predio"]["coef"][soft] <= 0.0                       # the served head obeys its sign
    assert result["report"]["heads"]["predio"]["refit_oof"]["brier"] <= result["report"]["heads"]["predio"]["oof"]["brier"]
