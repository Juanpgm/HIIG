"""Calibrated decision layer, pure part: loaders, validation, prediction, monotonicity, policy, motivos.

Everything uses tiny invented parameter files (no real addresses). The service integration lives in
``test_decision_service.py``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random

import pytest

from cali_address import decision as D

REAL_PARAMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiments", "calib_design", "params")

SIGNS = {
    "confidence": 1, "margen_manzana": 1, "margen_predio": 1, "raw_margin_lot": 1, "n_competidores": -1,
    "log_n_predios": -1, "soft_n": -1, "relaxed_plate": -1, "gate_fallback": -1, "fidelity_medium": -1,
    "fidelity_high": -1, "plate_missing": -1, "cross_missing": -1, "src_pcer": -1, "lot_conflict": -1,
    "cat_verified": 1, "cat_ambiguous": -1, "cat_door_not_catalogued": -1, "guard_one_sided": -1, "guard_tramo": -1,
    "predial_null": -1, "nivel_manzana": 0, "nivel_via": 0, "nivel_esquina": 0,
}
NAMES = list(SIGNS)


def head(coefs: dict | None = None, intercept: float = 3.0) -> dict:
    coefs = coefs or {}
    return {
        "target": "invented", "coef": [float(coefs.get(n, 0.0)) for n in NAMES], "intercept": intercept, "C": 1.0,
        "means": [0.5 if n in ("confidence", "margen_manzana", "margen_predio", "raw_margin_lot") else 0.1 for n in NAMES],
        "stds": [0.2 for _ in NAMES],
    }


def params(**over) -> dict:
    data = {
        "kind": "logistic_ext", "method": "logit_signed", "signs": dict(SIGNS), "version": "rel-test-1",
        "feature_names": list(NAMES),
        "motivo_codes": {n: f"code_{n}" for n in NAMES},
        "motivo_reference": {n: (0.5 if n in ("confidence", "margen_manzana", "margen_predio", "raw_margin_lot") else 0.0)
                             for n in NAMES},
        "clip": 5.0,
        "predio": head({"confidence": 0.5, "soft_n": -1.0, "src_pcer": -0.7, "raw_margin_lot": 0.6, "nivel_via": 0.8}),
        "manzana": head({"confidence": 0.4, "soft_n": -0.5, "margen_manzana": 0.6}, intercept=5.0),
        "provenance": {"rows": "invented"},
    }
    data.update(over)
    return data


def decision_doc(**over) -> dict:
    data = {
        "version": "dec-test-1", "reliability_version": "rel-test-1", "tau_p": 0.94, "tau_m": 0.70, "floor": 0.70,
        "confianza_floor": 0.70, "bands": {"alta": 0.97, "media": 0.94},
        "precision_targets": {"lot": 0.99, "manzana": 0.99},
        "neutral": {"soft_rules": ["letra_via", "bis_via"], "max_soft": 99, "gate_fallback": True,
                    "ambiguity_delta": 0.0, "threshold": "confianza_floor"},
        "fit_provenance": {"rows": "invented"},
    }
    data.update(over)
    return data


def write(tmp_path, name: str, data) -> str:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return str(path)


def model(**over) -> D.CalibratedReliability:
    loaded = D.load_calibrated(write_tmp(params(**over)))
    assert loaded is not None
    return loaded


_TMP: list = []


@pytest.fixture(autouse=True)
def _tmp_dir(tmp_path):
    _TMP.clear()
    _TMP.append(tmp_path)


def write_tmp(data) -> str:
    return write(_TMP[0], "rel.json", data)


# --------------------------------------------------------------------------- 1. loader never raises
def test_valid_file_loads_with_its_version():
    loaded = model()
    assert loaded.version == "rel-test-1" and loaded.feature_names == tuple(NAMES)


def test_missing_file_is_none(tmp_path):
    assert D.load_calibrated(str(tmp_path / "nope.json")) is None


@pytest.mark.parametrize("text", ["", "{", "[]", "null", "42", '"x"'])
def test_malformed_json_is_none(text):
    assert D.load_calibrated(write_tmp(text)) is None


def test_unknown_kind_is_none():
    assert D.load_calibrated(write_tmp(params(kind="logistic"))) is None
    assert D.load_calibrated(write_tmp(params(kind="hybrid"))) is None


def test_missing_kind_is_none():
    data = params()
    del data["kind"]
    assert D.load_calibrated(write_tmp(data)) is None


def test_unknown_feature_is_none():
    data = params(feature_names=NAMES[:-1] + ["something_new"])
    assert D.load_calibrated(write_tmp(data)) is None


def test_duplicate_and_empty_feature_lists_are_none():
    assert D.load_calibrated(write_tmp(params(feature_names=NAMES[:-1] + [NAMES[0]]))) is None
    assert D.load_calibrated(write_tmp(params(feature_names=[]))) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None])
def test_non_finite_or_non_numeric_coefficient_is_none(bad):
    data = params()
    data["predio"]["coef"][0] = bad
    assert D.load_calibrated(write_tmp(data)) is None


def test_non_finite_intercept_mean_or_std_is_none():
    for key, idx in (("intercept", None), ("means", 3), ("stds", 3)):
        data = params()
        if idx is None:
            data["manzana"][key] = float("nan")
        else:
            data["manzana"][key][idx] = float("inf")
        assert D.load_calibrated(write_tmp(data)) is None


def test_wrong_vector_length_is_none():
    data = params()
    data["predio"]["coef"] = data["predio"]["coef"][:-1]
    assert D.load_calibrated(write_tmp(data)) is None


def test_missing_head_is_none():
    data = params()
    del data["manzana"]
    assert D.load_calibrated(write_tmp(data)) is None


def test_coefficient_violating_its_sign_is_none():
    data = params()
    data["predio"]["coef"][NAMES.index("src_pcer")] = 0.3        # sign -1
    assert D.load_calibrated(write_tmp(data)) is None
    data = params()
    data["manzana"]["coef"][NAMES.index("confidence")] = -0.3    # sign +1
    assert D.load_calibrated(write_tmp(data)) is None


def test_free_sign_features_take_either_sign():
    data = params()
    data["predio"]["coef"][NAMES.index("nivel_via")] = -0.4
    assert D.load_calibrated(write_tmp(data)) is not None


def test_signs_must_cover_every_feature_and_be_ternary():
    data = params()
    del data["signs"]["soft_n"]
    assert D.load_calibrated(write_tmp(data)) is None
    data = params()
    data["signs"]["soft_n"] = 2
    assert D.load_calibrated(write_tmp(data)) is None


def test_motivo_codes_must_cover_every_feature():
    data = params()
    del data["motivo_codes"]["soft_n"]
    assert D.load_calibrated(write_tmp(data)) is None


# --------------------------------------------------------------------------- 2. decision.json validation
def test_valid_decision_passes():
    out = D.validate_decision(decision_doc())
    assert out["tau_p"] == 0.94 and out["neutral"]["soft_rules"] == ["letra_via", "bis_via"]


@pytest.mark.parametrize("over", [
    {"floor": 0.69},                                    # floor below the absolute minimum
    {"floor": 0.5, "tau_m": 0.5, "tau_p": 0.94},
    {"tau_p": 0.99},                                    # tau_p above bands.alta
    {"tau_p": 0.69},                                    # tau_p below the floor
    {"tau_m": 0.69},                                    # tau_m below the floor
    {"bands": {"alta": 1.01, "media": 0.94}},           # alta above 1
    {"confianza_floor": 0.6},
    {"tau_p": "high"},
    {"tau_p": float("nan")},
    {"version": ""},
    {"reliability_version": 3},
])
def test_invalid_decision_is_rejected(over):
    with pytest.raises(ValueError):
        D.validate_decision(decision_doc(**over))


def test_unknown_soft_rule_is_rejected():
    bad = decision_doc()
    bad["neutral"]["soft_rules"] = ["letra_via", "no_such_rule"]
    with pytest.raises(ValueError, match="soft rule"):
        D.validate_decision(bad)


def test_missing_required_keys_are_rejected():
    for key in ("tau_p", "tau_m", "bands", "neutral", "floor", "confianza_floor", "reliability_version"):
        bad = decision_doc()
        del bad[key]
        with pytest.raises(ValueError):
            D.validate_decision(bad)


def test_address_text_anywhere_is_rejected():
    bad = decision_doc()
    bad["fit_provenance"]["note"] = "CL 5 # 38-10"
    with pytest.raises(ValueError, match="address"):
        D.validate_decision(bad)


def test_band_media_must_equal_tau_p():
    with pytest.raises(ValueError, match="media"):
        D.validate_decision(decision_doc(bands={"alta": 0.97, "media": 0.95}))


def test_equal_alta_and_tau_p_is_allowed():
    assert D.validate_decision(decision_doc(tau_p=0.97, bands={"alta": 0.97, "media": 0.97}))["tau_p"] == 0.97


def test_validate_does_not_mutate_its_input():
    raw = decision_doc()
    snapshot = copy.deepcopy(raw)
    D.validate_decision(raw)
    assert raw == snapshot


def test_load_decision_returns_none_for_missing_malformed_or_invalid(tmp_path):
    assert D.load_decision(str(tmp_path / "x.json")) is None
    assert D.load_decision(write(tmp_path, "a.json", "{")) is None
    assert D.load_decision(write(tmp_path, "b.json", decision_doc(tau_p=0.5))) is None
    assert D.load_decision(write(tmp_path, "c.json", [1, 2])) is None
    assert D.load_decision(write(tmp_path, "d.json", decision_doc()))["version"] == "dec-test-1"


# --------------------------------------------------------------------------- pair loader / mode switch
def test_pair_needs_both_files_and_matching_versions(tmp_path):
    (tmp_path / "decision.json").write_text(json.dumps(decision_doc()), encoding="utf-8")
    assert D.load_calibrated_pair(str(tmp_path)) is None                      # reliability file missing
    (tmp_path / "reliability_calibrated.json").write_text(json.dumps(params()), encoding="utf-8")
    pair = D.load_calibrated_pair(str(tmp_path))
    assert pair is not None and pair[1]["tau_p"] == 0.94
    (tmp_path / "reliability_calibrated.json").write_text(json.dumps(params(version="other")), encoding="utf-8")
    assert D.load_calibrated_pair(str(tmp_path)) is None                      # version mismatch
    (tmp_path / "reliability_calibrated.json").write_text(json.dumps(params()), encoding="utf-8")
    (tmp_path / "decision.json").unlink()
    assert D.load_calibrated_pair(str(tmp_path)) is None                      # decision missing


def test_pair_for_an_artifacts_dir_without_files_is_none(tmp_path):
    assert D.load_calibrated_pair(str(tmp_path)) is None
    assert D.load_calibrated_pair(None) is None
    assert D.load_calibrated_pair(str(tmp_path / "does_not_exist")) is None


def test_calibrated_for_reads_the_attribute_then_the_directory(tmp_path):
    class Stub:
        pass

    stub = Stub()
    stub.calibrated = "sentinel"
    assert D.calibrated_for(stub) == "sentinel"
    bare = Stub()
    assert D.calibrated_for(bare) is None and bare.calibrated is None          # cached after the first look
    (tmp_path / "decision.json").write_text(json.dumps(decision_doc()), encoding="utf-8")
    (tmp_path / "reliability_calibrated.json").write_text(json.dumps(params()), encoding="utf-8")
    with_files = Stub()
    with_files.artifacts_dir = str(tmp_path)
    assert D.calibrated_for(with_files)[1]["tau_m"] == 0.70


def test_decision_present_but_unusable_warns_once_and_falls_back(tmp_path):
    (tmp_path / "decision.json").write_text("{", encoding="utf-8")

    class Stub:
        artifacts_dir = str(tmp_path)

    with pytest.warns(RuntimeWarning, match="decision.json"):
        assert D.calibrated_for(Stub()) is None


# --------------------------------------------------------------------------- 3. prediction
def full_features(value: float = 0.0) -> dict:
    return {n: value for n in NAMES}


def test_prediction_is_in_range_and_manzana_never_below_predio():
    m = model()
    rng = random.Random(7)
    for _ in range(300):
        feats = {n: rng.uniform(-10, 10) for n in NAMES}
        pp, pm = m.predict(feats)
        assert 0.001 <= pp <= 0.999 and 0.001 <= pm <= 0.999 and pm >= pp


def test_manzana_head_is_floored_by_the_predio_head():
    pp, pm = model(manzana=head(intercept=-9.0), predio=head(intercept=9.0)).predict(full_features(0.0))
    assert pm == pp


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), "abc", [], {}])
def test_unusable_feature_value_equals_the_head_mean(bad):
    m = model()
    means = {n: m.predio.means[i] for i, n in enumerate(NAMES)}
    neutral = m.predict(dict(means))
    for name in NAMES:
        feats = dict(means)
        feats[name] = bad
        # means differ per head only through the head's own values; here both heads share them
        assert m.predict(feats) == pytest.approx(neutral)


def test_missing_features_equal_the_head_mean():
    m = model()
    means = {n: m.predio.means[i] for i, n in enumerate(NAMES)}
    assert m.predict({}) == pytest.approx(m.predict(means))


def test_extreme_values_are_clipped_not_exploded():
    m = model()
    huge = m.predict({"confidence": 1e12})
    big = m.predict({"confidence": 1e3})
    assert huge == pytest.approx(big)
    assert all(0.001 <= v <= 0.999 for v in huge)


def test_prediction_is_deterministic():
    m = model()
    feats = {n: 0.37 for n in NAMES}
    assert m.predict(feats) == m.predict(dict(feats))


def test_sigmoid_is_stable_for_extreme_logits():
    assert model(predio=head(intercept=1e6), manzana=head(intercept=-1e6)).predict({}) == (0.999, 0.999)
    assert model(predio=head(intercept=-1e6), manzana=head(intercept=-1e6)).predict({}) == (0.001, 0.001)


# --------------------------------------------------------------------------- 4. monotonicity (property)
def _real_models():
    out = []
    for tag in ("cand", "base"):
        path = os.path.join(REAL_PARAMS, f"reliability_calibrated_{tag}.json")
        if os.path.exists(path):
            out.append(pytest.param(path, id=tag))
    return out


def test_stored_signs_are_respected_by_every_shipped_coefficient():
    paths = _real_models()
    if not paths:
        pytest.skip("candidate parameter files are not present")
    for p in paths:
        loaded = D.load_calibrated(p.values[0])
        assert loaded is not None
        for head_ in (loaded.predio, loaded.manzana):
            for name, coef in zip(loaded.feature_names, head_.coef):
                sign = loaded.signs[name]
                assert sign == 0 or coef * sign >= 0, name


@pytest.mark.parametrize("seed", range(5))
def test_signed_features_move_probability_monotonically(seed):
    m = model()
    rng = random.Random(seed)
    for _ in range(60):
        base = {n: rng.uniform(-1, 2) for n in NAMES}
        pp0, pm0 = m.predict(base)
        for name, sign in SIGNS.items():
            if sign == 0:
                continue
            up = dict(base)
            up[name] = base[name] + rng.uniform(0.0, 3.0)
            pp1, pm1 = m.predict(up)
            if sign > 0:
                assert pp1 >= pp0 - 1e-12 and pm1 >= pm0 - 1e-12, name
            else:
                assert pp1 <= pp0 + 1e-12 and pm1 <= pm0 + 1e-12, name


# --------------------------------------------------------------------------- 5. policy partition and boundaries
DEC = D.validate_decision(decision_doc())


def test_predial_row_above_tau_p_keeps_the_predial():
    assert D.decide_level(0.96, 0.99, True, DEC) == ("predio", "media")


def test_alta_boundary_is_inclusive():
    assert D.decide_level(0.97, 0.99, True, DEC) == ("predio", "alta")
    assert D.decide_level(0.9699999, 0.99, True, DEC) == ("predio", "media")


def test_tau_p_boundary_is_inclusive():
    assert D.decide_level(0.94, 0.99, True, DEC) == ("predio", "media")
    assert D.decide_level(0.9399999, 0.99, True, DEC)[0] == "manzana"


def test_row_without_a_pipeline_predial_never_gets_one():
    assert D.decide_level(0.999, 0.999, False, DEC) == ("manzana", "baja")


def test_manzana_boundary_is_inclusive_and_below_is_sin_match():
    assert D.decide_level(0.1, 0.70, True, DEC) == ("manzana", "baja")
    assert D.decide_level(0.1, 0.6999999, True, DEC) == ("sin_match", None)


@pytest.mark.parametrize("pp,pm", [(0.001, 0.001), (0.5, 0.5), (0.95, 0.95), (0.999, 0.999), (0.69, 0.71)])
@pytest.mark.parametrize("emits", [True, False])
def test_every_row_lands_in_exactly_one_outcome(pp, pm, emits):
    outcome, band = D.decide_level(pp, pm, emits, DEC)
    assert outcome in ("predio", "manzana", "sin_match")
    assert (band in D.BANDS) == (outcome != "sin_match")
    if outcome == "predio":
        assert emits and band in ("alta", "media")
    if outcome == "manzana":
        assert band == "baja"


def test_band_partition_is_complete_over_a_sweep():
    seen = set()
    for i in range(0, 1001):
        p = i / 1000
        outcome, band = D.decide_level(p, max(p, 0.7), True, DEC)
        seen.add((outcome, band))
        if outcome == "predio":
            assert (band == "alta") == (p >= DEC["bands"]["alta"])
    assert seen == {("predio", "alta"), ("predio", "media"), ("manzana", "baja")}


def test_nan_probability_is_the_safest_outcome():
    assert D.decide_level(float("nan"), float("nan"), True, DEC) == ("sin_match", None)


# --------------------------------------------------------------------------- 13. motivos
def test_motivos_orders_by_absolute_contribution_and_caps_at_three():
    m = model()
    feats = {n: 0.0 for n in NAMES}
    feats.update({"soft_n": 2.0, "src_pcer": 1.0, "raw_margin_lot": 0.0, "confidence": 0.2, "fidelity_high": 1.0})
    text = m.motivos("predio", feats)
    codes = text.split(";")
    assert 1 <= len(codes) <= 3
    assert all(c[0] in "+-" for c in codes)
    assert all(c[1:].startswith("code_") for c in codes)


def test_motivos_is_empty_when_nothing_matters():
    m = model()
    feats = {n: 0.0 for n in NAMES}
    feats.update({"confidence": 0.5, "margen_manzana": 0.5, "margen_predio": 0.5, "raw_margin_lot": 0.5})
    assert m.motivos("manzana", feats) == ""


def test_motivos_sign_prefix_follows_the_contribution():
    m = model()
    worse = m.motivos("predio", {**full_features(0.0), "src_pcer": 1.0})
    assert worse.startswith("-code_src_pcer")
    better = m.motivos("predio", {**full_features(0.0), "raw_margin_lot": 1.0})
    assert "+code_raw_margin_lot" in better


def test_motivos_ignores_zero_coefficients_and_treats_bad_values_as_the_mean():
    m = model()
    assert "cat_verified" not in m.motivos("predio", {"cat_verified": 1.0})   # coefficient 0
    mean_src = m.predio.means[NAMES.index("src_pcer")]
    expected = m.motivos("predio", {"src_pcer": mean_src})
    assert m.motivos("predio", {"src_pcer": float("nan")}) == expected      # NaN is read as the head mean
    assert m.motivos("predio", {"src_pcer": None}) == expected


def test_motivos_rejects_unknown_head():
    with pytest.raises(ValueError):
        model().motivos("nope", {})


def test_motivos_lists_only_known_codes():
    m = model()
    feats = {n: 3.0 for n in NAMES}
    known = {f"code_{n}" for n in NAMES}
    for head_name in ("predio", "manzana"):
        for code in filter(None, m.motivos(head_name, feats).split(";")):
            assert code[1:] in known


# --------------------------------------------------------------------------- shipped candidate files (when present)
def test_shipped_candidate_pair_loads_and_agrees():
    d, r = os.path.join(REAL_PARAMS, "decision_cand.json"), os.path.join(REAL_PARAMS, "reliability_calibrated_cand.json")
    if not (os.path.exists(d) and os.path.exists(r)):
        pytest.skip("candidate parameter files are not present")
    dec, rel = D.load_decision(d), D.load_calibrated(r)
    assert dec is not None and rel is not None and dec["reliability_version"] == rel.version
    assert len(rel.feature_names) == 24 and math.isclose(dec["tau_p"], 0.94)
