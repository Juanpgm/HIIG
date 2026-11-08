"""Calibrated decision layer wired into ``service.normalize_strict`` (invented fixtures, no real addresses).

The layer is ON only when the normalizer carries a calibrated ``(model, decision)`` pair (``decision.json`` +
``reliability_calibrated.json`` in the artifact directory). Without it the output is byte-identical to the historical one
(the pinned sha256 lives in ``test_output_validation_wiring.py``; here the column list and a second fixture are pinned too).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_stub
from test_decision_model import NAMES, decision_doc, head, params

from cali_address import decision as D
from cali_address import service
from cali_address.output_schema import validate_and_correct
from cali_address.service import OUTPUT_COLUMNS, normalize_strict

LOT_A = "760010100000000900001"          # 21 characters, invented
LOT_B = "760010100000000900002"
NPN_A = LOT_A + "000000000"
NPN_B = LOT_B + "000000000"
MANZ_A, MANZ_B = "760010100000000900001", "760010100000000900002"

NEW_COLUMNS = ["confiabilidad_calibrada", "banda_confiabilidad", "motivos_confiabilidad", "p_predio_cal", "p_manzana_cal"]


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def const_model(p_predio: float, p_manzana: float) -> D.CalibratedReliability:
    """Heads with zero coefficients: the probability is the intercept, whatever the features are."""
    data = params(predio=head(intercept=logit(p_predio)), manzana=head(intercept=logit(p_manzana)))
    data["predio"]["coef"] = [0.0] * len(NAMES)
    data["manzana"]["coef"] = [0.0] * len(NAMES)
    path = None
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        loaded = D.load_calibrated(path)
    assert loaded is not None
    return loaded


def calibrated(p_predio=0.99, p_manzana=0.995, **decision_over):
    return const_model(p_predio, p_manzana), D.validate_decision(decision_doc(**decision_over))


class FakeModel:
    """Records the feature dict it is asked about; constant probabilities."""

    version = "rel-test-1"

    def __init__(self, pp=0.99, pm=0.995):
        self.pp, self.pm, self.seen, self.motivo_heads = pp, pm, [], []

    def predict(self, features):
        self.seen.append(dict(features))
        return self.pp, self.pm

    def motivos(self, head_name, features):
        self.motivo_heads.append(head_name)
        return "+fake"


QUERY = "CL 987 # 654 - 12"
DOCS = [{"direccion": "CL 987 # 654 - 12", "predial": NPN_A, "manzana": MANZ_A}]


def run(raws=None, docs=None, scores=None, cal="default", stub_kw=None, **kw):
    raws = [QUERY] if raws is None else raws
    stub = make_stub(docs or DOCS, scores=scores or [[0.95] + [0.1] * (len(docs or DOCS) - 1)] * len(raws), **(stub_kw or {}))
    if cal == "default":
        cal = calibrated()
    if cal is not None:
        stub.calibrated = cal
    kw.setdefault("gazetteer", None)
    return normalize_strict(stub, raws, **kw)


def digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_json(orient="split", default_handler=str).encode()).hexdigest()


# --------------------------------------------------------------------------- 11. byte identity without the pair
def test_without_the_pair_the_columns_are_exactly_the_historical_ones():
    frame = run(cal=None)
    assert list(frame.columns) == OUTPUT_COLUMNS


def test_without_the_pair_the_output_is_independent_of_the_calibration_sink():
    base = run(["CL 987 # 654 - 12", "TEXTO SIN DIRECCION", ""], cal=None, ambiguity_delta=0.02)
    sink: list = []
    with_sink = run(["CL 987 # 654 - 12", "TEXTO SIN DIRECCION", ""], cal=None, ambiguity_delta=0.02, calibration_sink=sink)
    assert digest(base) == digest(with_sink)
    assert len(sink) == 3 and sink[1] is None and sink[2] is None and set(NAMES) <= set(sink[0])


def test_a_stub_with_calibrated_none_behaves_like_the_historical_pipeline():
    stub = make_stub(DOCS, scores=[[0.95]])
    stub.calibrated = None
    assert list(normalize_strict(stub, [QUERY], gazetteer=None).columns) == OUTPUT_COLUMNS


def test_default_ambiguity_delta_is_unchanged_without_the_pair():
    docs = [{"direccion": QUERY, "predial": NPN_A, "manzana": MANZ_A}, {"direccion": QUERY, "predial": NPN_B, "manzana": MANZ_B}]
    row = run(docs=docs, scores=[[0.90, 0.89]], cal=None).iloc[0]
    assert row["estado"] == "SIN_MATCH" and row["motivo"].startswith("ambiguo")


# --------------------------------------------------------------------------- columns
def test_new_columns_are_appended_last_in_calibrated_mode():
    frame = run()
    assert list(frame.columns) == OUTPUT_COLUMNS + NEW_COLUMNS


def test_new_columns_follow_the_source_and_refine_columns():
    stub = make_stub(DOCS, scores=[[0.95]])
    stub._fuente = np.array(["nomenclatura_cert"], dtype=object)
    stub._estado_nomenclatura = np.array(["vigente"], dtype=object)
    stub.calibrated = calibrated()
    frame = normalize_strict(stub, [QUERY], gazetteer=None, refine_input=True)
    assert list(frame.columns) == OUTPUT_COLUMNS + ["fuente", "estado_nomenclatura", "refine_changes"] + NEW_COLUMNS


def test_ok_row_keeps_the_predial_in_the_alta_band():
    row = run(cal=calibrated(0.99, 0.995)).iloc[0]
    assert row["estado"] == "OK" and row["numero_predial_nacional"] == NPN_A
    assert row["banda_confiabilidad"] == "alta" and row["confiabilidad_calibrada"] == pytest.approx(0.99, abs=1e-3)
    assert row["p_predio_cal"] == pytest.approx(0.99, abs=1e-3) and row["p_manzana_cal"] == pytest.approx(0.995, abs=1e-3)
    assert isinstance(row["motivos_confiabilidad"], str)


def test_media_band_between_tau_p_and_alta():
    row = run(cal=calibrated(0.95, 0.99)).iloc[0]
    assert row["estado"] == "OK" and row["numero_predial_nacional"] == NPN_A and row["banda_confiabilidad"] == "media"


def test_values_are_rounded_to_three_decimals():
    row = run(cal=calibrated(0.987654, 0.99876)).iloc[0]
    assert row["p_predio_cal"] == 0.988 and row["p_manzana_cal"] == 0.999 and row["confiabilidad_calibrada"] == 0.988


def test_low_predio_probability_downgrades_to_manzana_consistently():
    frame = run(cal=calibrated(0.80, 0.95))
    row = frame.iloc[0]
    assert row["estado"] == "OK" and pd.isna(row["numero_predial_nacional"])
    assert row["manzana"] == MANZ_A and row["banda_confiabilidad"] == "baja"
    assert row["confiabilidad_calibrada"] == pytest.approx(0.95, abs=1e-3)
    assert row["position_tier"] == "manzana" and row["position_radius_m"] == service.POSITION_RADIUS_M["manzana"]
    assert row["nivel_precision"] in ("manzana", "esquina", "via", "direccion")
    assert "confiabilidad del predio baja (p=0.80)" in row["motivo"]
    assert validate_and_correct(frame, "report")[1].violations == {}


def test_both_probabilities_low_gives_sin_match_with_the_confianza_kept():
    frame = run(cal=calibrated(0.30, 0.60))
    row = frame.iloc[0]
    assert row["estado"] == "SIN_MATCH" and row["motivo"] == "confiabilidad baja: p_manzana 0.60 < 0.70"
    assert pd.isna(row["numero_predial_nacional"]) and pd.isna(row["manzana"]) and row["position_tier"] == "none"
    assert row["confianza"] == pytest.approx(0.95)
    assert all(pd.isna(row[c]) for c in NEW_COLUMNS)
    assert validate_and_correct(frame, "report")[1].violations == {}


def test_sin_match_by_the_layer_is_not_reported_as_below_cutoff():
    row = run(cal=calibrated(0.30, 0.60)).iloc[0]
    assert row["catalog_status"] != "below_cutoff" and row["address_verified"] is False or row["address_verified"] == False  # noqa: E712


def test_boundaries_are_inclusive_through_the_service():
    assert run(cal=calibrated(0.94, 0.99)).iloc[0]["numero_predial_nacional"] == NPN_A      # P == tau_p emits
    # exact float equality: the thresholds are set to the very value the model predicts
    p_exact = const_model(0.97, 0.99).predict({})[0]
    edge = calibrated(0.97, 0.99, tau_p=p_exact, bands={"alta": p_exact, "media": p_exact})
    assert run(cal=edge).iloc[0]["banda_confiabilidad"] == "alta"                         # P == alta
    m_exact = const_model(0.30, 0.70).predict({})[1]
    edge_m = calibrated(0.30, 0.70, tau_m=m_exact)
    assert run(cal=edge_m).iloc[0]["estado"] == "OK"                                      # P_manzana == tau_m


# --------------------------------------------------------------------------- 6. floors
def test_confianza_below_the_floor_is_sin_match_whatever_the_probability():
    row = run(scores=[[0.65]], cal=calibrated(0.999, 0.999)).iloc[0]
    assert row["estado"] == "SIN_MATCH" and row["motivo"] == "confianza 0.65 < umbral 0.70"


def test_confianza_equal_to_the_floor_passes():
    assert run(scores=[[0.70]], cal=calibrated(0.999, 0.999)).iloc[0]["estado"] == "OK"


def test_the_floor_replaces_the_artifact_threshold_only_in_calibrated_mode():
    legacy = run(scores=[[0.65]], cal=None).iloc[0]                     # stub threshold 0.5
    assert legacy["estado"] == "OK"


def test_no_ok_row_has_confianza_or_probability_below_the_floor():
    raws = [QUERY] * 6
    for cal in (calibrated(0.99, 0.995), calibrated(0.80, 0.95), calibrated(0.30, 0.60), calibrated(0.5, 0.71)):
        frame = run(raws=raws, scores=[[s] for s in (0.5, 0.65, 0.699, 0.70, 0.85, 0.99)], cal=cal)
        ok = frame[frame["estado"] == "OK"]
        assert (ok["confianza"] >= 0.70).all() and (ok["confiabilidad_calibrada"] >= 0.70).all()


# --------------------------------------------------------------------------- neutral configuration
SOFT_DOC = [{"direccion": "CL 987 A # 654 - 13", "predial": NPN_A, "manzana": MANZ_A}]


def test_neutral_soft_rules_accept_what_the_historical_config_rejects():
    assert run(docs=SOFT_DOC, cal=None).iloc[0]["estado"] == "SIN_MATCH"       # 2 soft findings, max_soft 1
    row = run(docs=SOFT_DOC).iloc[0]                                           # letra_via soft, plate window, max_soft 99
    assert row["estado"] == "OK"


def test_soft_n_reaches_the_features():
    model = FakeModel()
    run(docs=SOFT_DOC, cal=(model, D.validate_decision(decision_doc())))
    assert model.seen[0]["soft_n"] == 2.0


def test_explicit_caller_arguments_win_over_the_decision_file():
    # soft_rules=() makes the bis mismatch hard again; the file (bis_via soft) would have accepted it
    bis_doc = [{"direccion": "CL 987 BIS # 654 - 12", "predial": NPN_A, "manzana": MANZ_A}]
    assert run(docs=bis_doc).iloc[0]["estado"] == "OK"
    assert run(docs=bis_doc, soft_rules=()).iloc[0]["estado"] == "SIN_MATCH"
    # explicit max_soft
    assert run(docs=SOFT_DOC, max_soft=1).iloc[0]["estado"] == "SIN_MATCH"
    # explicit ambiguity delta re-enables the abstention
    docs = [{"direccion": QUERY, "predial": NPN_A, "manzana": MANZ_A}, {"direccion": QUERY, "predial": NPN_B, "manzana": MANZ_B}]
    assert run(docs=docs, scores=[[0.90, 0.89]], ambiguity_delta=0.02).iloc[0]["estado"] == "SIN_MATCH"
    assert run(docs=docs, scores=[[0.90, 0.89]]).iloc[0]["estado"] == "OK"     # file: ambiguity_delta 0
    # explicit threshold
    assert run(scores=[[0.75]], threshold=0.80).iloc[0]["motivo"] == "confianza 0.75 < umbral 0.80"


def test_the_normalizer_threshold_is_never_mutated():
    stub = make_stub(DOCS, scores=[[0.95]], threshold=0.5)
    stub.calibrated = calibrated()
    normalize_strict(stub, [QUERY], gazetteer=None)
    normalize_strict(stub, [QUERY], gazetteer=None, calibration_sink=[])
    assert stub.threshold == 0.5


class _Gazetteer:
    """A gazetteer whose gate (patched below) always leaves no candidate."""

    def detect(self, raw):
        from types import SimpleNamespace
        return SimpleNamespace(cleaned_text=str(raw), detected=True, barrio=None, vereda=None, comuna=None,
                               corregimiento=None)

    def locate(self, lon, lat):
        return {}


def test_gate_fallback_is_on_in_calibrated_mode_and_sets_the_feature(monkeypatch):
    monkeypatch.setattr(service, "gate_candidates", lambda g, d, c, **k: ([], "barrio INVENTADO"))
    legacy = run(cal=None, gazetteer=_Gazetteer()).iloc[0]
    assert legacy["estado"] == "SIN_MATCH" and legacy["motivo"] == "candidatos fuera de barrio INVENTADO"
    model = FakeModel()
    row = run(cal=(model, D.validate_decision(decision_doc())), gazetteer=_Gazetteer()).iloc[0]
    assert row["estado"] == "OK" and "fuera de barrio INVENTADO" in row["motivo"]
    assert model.seen[0]["gate_fallback"] == 1.0
    explicit = run(cal=calibrated(), gazetteer=_Gazetteer(), gate_fallback=False).iloc[0]
    assert explicit["estado"] == "SIN_MATCH"


# --------------------------------------------------------------------------- features
def features_of(docs=None, scores=None, raws=None, stub_attrs=None, **kw):
    model = FakeModel()
    stub = make_stub(docs or DOCS, scores=scores or [[0.95] + [0.1] * (len(docs or DOCS) - 1)] * len(raws or [QUERY]))
    for key, value in (stub_attrs or {}).items():
        setattr(stub, key, value)
    stub.calibrated = (model, D.validate_decision(decision_doc()))
    normalize_strict(stub, raws or [QUERY], gazetteer=None, **kw)
    return model.seen


def test_features_cover_exactly_the_serving_contract():
    seen = features_of()
    assert set(seen[0]) == set(NAMES) and set(NAMES) == set(D.FEATURE_NAMES)
    assert all(isinstance(v, float) and np.isfinite(v) for v in seen[0].values())


def test_plain_exact_row_features():
    f = features_of()[0]
    assert f["confidence"] == pytest.approx(0.95)
    assert f["plate_missing"] == 0.0 and f["cross_missing"] == 0.0 and f["soft_n"] == 0.0
    assert f["gate_fallback"] == 0.0 and f["src_pcer"] == 0.0 and f["lot_conflict"] == 0.0
    assert f["predial_null"] == 0.0 and f["guard_one_sided"] == 0.0 and f["guard_tramo"] == 0.0
    assert f["margen_manzana"] == 0.3 and f["margen_predio"] == 0.3 and f["raw_margin_lot"] == 0.3   # no rival
    assert f["n_competidores"] == 0.0 and f["log_n_predios"] == 0.0
    assert f["fidelity_medium"] == 0.0 and f["fidelity_high"] == 0.0
    assert f["cat_verified"] == 1.0 and f["cat_ambiguous"] == 0.0 and f["cat_door_not_catalogued"] == 0.0


def test_missing_plate_and_cross_flags():
    f = features_of(raws=["CL 987"], min_struct=0.0)[0]
    assert f["plate_missing"] == 1.0 and f["cross_missing"] == 1.0


def test_rival_margins_and_raw_margin_lot():
    docs = [{"direccion": QUERY, "predial": NPN_A, "manzana": MANZ_A},
            {"direccion": QUERY, "predial": NPN_B, "manzana": MANZ_B}]
    f = features_of(docs=docs, scores=[[0.90, 0.86]], ambiguity_delta=0.0)[0]
    assert f["margen_manzana"] == pytest.approx(0.04) and f["margen_predio"] == pytest.approx(0.04)
    assert f["raw_margin_lot"] == pytest.approx(0.04) and f["n_competidores"] == 1.0


def test_raw_margin_lot_ignores_a_rival_in_the_same_lot_and_does_not_need_rule_compliance():
    docs = [{"direccion": QUERY, "predial": NPN_A, "manzana": MANZ_A},
            {"direccion": QUERY, "predial": LOT_A + "000000001", "manzana": MANZ_A},      # same lot
            {"direccion": "KR 5 # 6 - 7", "predial": NPN_B, "manzana": MANZ_B}]          # other lot, rule-violating
    f = features_of(docs=docs, scores=[[0.90, 0.89, 0.70]], ambiguity_delta=0.0)[0]
    assert f["raw_margin_lot"] == pytest.approx(0.20)      # the same-lot twin is skipped, the unfit rival counts
    assert f["margen_predio"] == pytest.approx(0.01)       # the twin competes at document level


def test_raw_margin_lot_is_clipped():
    docs = [{"direccion": QUERY, "predial": NPN_A, "manzana": MANZ_A}, {"direccion": "KR 5 # 6 - 7", "predial": NPN_B, "manzana": MANZ_B}]
    f = features_of(docs=docs, scores=[[0.95, 0.10]])[0]
    assert f["raw_margin_lot"] == 0.3


def test_src_pcer_lot_conflict_and_unseen_categories():
    fuente = np.array(["nomenclatura_pcer"], dtype=object)
    f = features_of(stub_attrs={"_fuente": fuente, "docs": pd.DataFrame({"lot_conflict": [1.0]})})[0]
    assert f["src_pcer"] == 1.0 and f["lot_conflict"] == 1.0
    odd = features_of(stub_attrs={"_fuente": np.array(["fuente_nueva"], dtype=object),
                                  "docs": pd.DataFrame({"lot_conflict": [np.nan]})})[0]
    assert odd["src_pcer"] == 0.0 and odd["lot_conflict"] == 0.0
    none_fuente = features_of(stub_attrs={"_fuente": np.array([None], dtype=object)})[0]
    assert none_fuente["src_pcer"] == 0.0
    no_cols = features_of(stub_attrs={"docs": pd.DataFrame({"other": [1]})})[0]
    assert no_cols["lot_conflict"] == 0.0 and no_cols["src_pcer"] == 0.0


def test_lot_conflict_is_binarised():
    f = features_of(stub_attrs={"docs": pd.DataFrame({"lot_conflict": [7.0]})})[0]
    assert f["lot_conflict"] == 1.0


def test_predial_null_and_shared_address_features():
    docs = [{"direccion": QUERY, "predial": None, "manzana": MANZ_A, "n_predios": 3}]
    f = features_of(docs=docs)[0]
    assert f["predial_null"] == 1.0 and f["log_n_predios"] == pytest.approx(np.log(3))
    assert f["cat_ambiguous"] == 1.0 and f["cat_verified"] == 0.0


def test_door_not_catalogued_feature_when_the_row_is_not_exact():
    f = features_of(docs=SOFT_DOC)[0]
    assert f["cat_verified"] == 0.0 and f["soft_n"] == 2.0


def test_fidelity_features_follow_the_row_after_the_fidelity_step(monkeypatch):
    from cali_address.fidelity import FidelityReport
    monkeypatch.setattr(service, "assess_fidelity",
                        lambda raw, output, cleaned=None: FidelityReport("high", ("plate_changed",), 0.3, {}))
    f = features_of()[0]
    assert f["fidelity_high"] == 1.0 and f["fidelity_medium"] == 0.0 and f["cat_verified"] == 0.0


def test_nivel_one_hot_uses_the_public_level_of_the_row():
    via = features_of(raws=["CL 987 # 654"])[0]     # no plate: the fidelity step lowers the row to via
    assert via["nivel_via"] == 1.0 and via["nivel_manzana"] == 0.0 and via["nivel_esquina"] == 0.0
    manzana = features_of(docs=SOFT_DOC)[0]         # soft findings cap the row at manzana
    assert manzana["nivel_manzana"] == 1.0 and manzana["nivel_via"] == 0.0


def test_gate_fallback_feature_is_off_without_a_gazetteer():
    assert features_of()[0]["gate_fallback"] == 0.0


def test_the_head_used_for_motivos_is_the_delivered_level():
    model = FakeModel(0.99, 0.995)
    run(cal=(model, D.validate_decision(decision_doc())))
    assert model.motivo_heads == ["predio"]
    model = FakeModel(0.5, 0.9)
    run(cal=(model, D.validate_decision(decision_doc())))
    assert model.motivo_heads == ["manzana"]


def test_motivos_text_is_exposed():
    assert run(cal=(FakeModel(), D.validate_decision(decision_doc()))).iloc[0]["motivos_confiabilidad"] == "+fake"


def test_row_without_predial_from_the_pipeline_never_gets_one_and_is_banded_baja():
    docs = [{"direccion": QUERY, "predial": None, "manzana": MANZ_A}]
    row = run(docs=docs, cal=calibrated(0.999, 0.999)).iloc[0]
    assert row["estado"] == "OK" and pd.isna(row["numero_predial_nacional"]) and row["banda_confiabilidad"] == "baja"
    assert "confiabilidad del predio baja" not in row["motivo"]


# --------------------------------------------------------------------------- 8/9 empty and degenerate input
def test_empty_input_has_the_same_columns_and_does_not_raise():
    frame = run(raws=[])
    assert list(frame.columns) == OUTPUT_COLUMNS + NEW_COLUMNS and len(frame) == 0


def test_all_unparseable_input_has_null_new_columns():
    frame = run(raws=["TEXTO SIN DIRECCION", "", "---"])
    assert list(frame.columns) == OUTPUT_COLUMNS + NEW_COLUMNS
    assert set(frame["estado"]) <= {"NO_PARSEABLE", "SIN_MATCH"}
    assert frame[NEW_COLUMNS].isna().all().all()


def test_single_row():
    frame = run(raws=[QUERY])
    assert len(frame) == 1 and frame.iloc[0]["banda_confiabilidad"] == "alta"


def test_band_is_in_the_vocabulary_for_ok_rows_and_null_otherwise():
    frame = run(raws=[QUERY, "TEXTO SIN DIRECCION", QUERY], scores=[[0.95], [0.5], [0.65]])
    ok = frame["estado"] == "OK"
    assert set(frame.loc[ok, "banda_confiabilidad"]) <= set(D.BANDS)
    assert frame.loc[~ok, NEW_COLUMNS].isna().all().all()


# --------------------------------------------------------------------------- 10 determinism
def test_same_input_twice_and_any_batch_size_give_identical_new_columns():
    # the stub indexes ``scores`` by the position among PARSEABLE queries, so the unparseable one goes last
    raws = [QUERY, "CL 987 A # 654 - 13", "CL 987 # 654", QUERY, "TEXTO"]
    docs = DOCS + SOFT_DOC
    scores = [[0.95, 0.2], [0.3, 0.9], [0.9, 0.1], [0.8, 0.1], [0.1, 0.1]]
    cal = calibrated(0.96, 0.99)
    first = run(raws=raws, docs=docs, scores=scores, cal=cal)
    second = run(raws=raws, docs=docs, scores=scores, cal=cal)
    pd.testing.assert_frame_equal(first, second)
    singles = pd.concat(
        [run(raws=[r], docs=docs, scores=[s], cal=cal) for r, s in zip(raws, scores)], ignore_index=True)
    pd.testing.assert_frame_equal(first[NEW_COLUMNS], singles[NEW_COLUMNS])


# --------------------------------------------------------------------------- 12 validate_outputs / stats
def test_validate_outputs_correct_does_not_change_the_layer_rows():
    raws = [QUERY, QUERY, QUERY]
    kw = dict(raws=raws, scores=[[0.95], [0.95], [0.65]])
    for cal in (calibrated(0.99, 0.995), calibrated(0.80, 0.95), calibrated(0.3, 0.6)):
        off = run(cal=cal, **kw)
        corrected = run(cal=cal, validate_outputs="correct", **kw)
        pd.testing.assert_frame_equal(off, corrected)


def test_stats_report_the_decision_counts():
    stats: dict = {}
    run(raws=[QUERY, QUERY, QUERY, "TEXTO"], scores=[[0.95]] * 4, cal=calibrated(0.80, 0.95), stats=stats)
    cal = stats["calibrated"]
    assert cal["reached"] == 3 and cal["abstained"] == 0 and cal["predio"] == 0 and cal["manzana"] == 3
    stats = {}
    run(raws=[QUERY, QUERY], scores=[[0.95]] * 2, cal=calibrated(0.30, 0.60), stats=stats)
    assert stats["calibrated"]["abstained"] == 2 and len(stats["calibrated"]["p_predio"]) == 2


def test_stats_have_no_calibrated_block_without_the_pair():
    stats: dict = {}
    run(cal=None, stats=stats)
    assert "calibrated" not in stats


def test_feature_sink_of_the_legacy_reliability_is_untouched_in_calibrated_mode():
    sink: list = []
    run(raws=[QUERY, "TEXTO"], scores=[[0.95], [0.1]], feature_sink=sink)
    assert sink[0] is not None and "margen_manzana" in sink[0] and sink[1] is None
