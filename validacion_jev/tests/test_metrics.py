import math

import pytest

from validacion_jev.metrics import (
    brier_score,
    cohen_kappa,
    confusion_matrix,
    disagreements,
    jev_label,
    precision_recall_f1,
    reliability_table,
    roc_auc,
    summarize,
    wilson_interval,
)
from validacion_jev.validators import Thresholds

T = Thresholds()


# ---- jev_label --------------------------------------------------------------


@pytest.mark.parametrize(
    "p,label",
    [
        (0, "no"), (0.15, "no"), (0.1500001, "indeterminate"), (0.5, "indeterminate"),
        (0.6, "indeterminate"), (0.8499999, "indeterminate"), (0.85, "yes"), (1, "yes"),
    ],
)
def test_jev_label_boundaries(p, label):
    assert jev_label(p, T) == label


def test_jev_label_none_is_none():
    assert jev_label(None, T) is None


def test_jev_label_respects_stricter_thresholds():
    assert jev_label(0.9, Thresholds(auto=0.95, review=0.6)) == "indeterminate"


# ---- Wilson ---------------------------------------------------------------------


def test_wilson_zero_n_is_none():
    assert wilson_interval(0, 0) is None


def test_wilson_zero_successes():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35  # known ~0.2775


def test_wilson_all_successes():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75


def test_wilson_midpoint_known_value():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_single_observation():
    lo, hi = wilson_interval(1, 1)
    assert 0 < lo < 1 and hi == 1.0


@pytest.mark.parametrize("k,n", [(-1, 5), (6, 5)])
def test_wilson_invalid(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


# ---- confusion / kappa / prf --------------------------------------------------------


def test_confusion_counts():
    pairs = [(True, True), (True, False), (False, True), (False, False), (True, True)]
    assert confusion_matrix(pairs) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_confusion_empty():
    assert confusion_matrix([]) == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def test_kappa_perfect_agreement():
    assert cohen_kappa({"tp": 5, "fp": 0, "fn": 0, "tn": 5}) == 1.0


def test_kappa_known_value():
    # po=0.7, pe=0.5 -> 0.4
    assert cohen_kappa({"tp": 4, "fp": 1, "fn": 2, "tn": 3}) == pytest.approx(0.4, abs=0.2)
    cm = {"tp": 35, "fp": 15, "fn": 15, "tn": 35}
    assert cohen_kappa(cm) == pytest.approx(0.4)


def test_kappa_chance_level_is_zero():
    assert cohen_kappa({"tp": 25, "fp": 25, "fn": 25, "tn": 25}) == pytest.approx(0.0)


def test_kappa_undefined_when_degenerate():
    assert cohen_kappa({"tp": 10, "fp": 0, "fn": 0, "tn": 0}) is None  # all same class
    assert cohen_kappa({"tp": 0, "fp": 0, "fn": 0, "tn": 0}) is None  # empty
    assert cohen_kappa({"tp": 0, "fp": 0, "fn": 0, "tn": 7}) is None


def test_kappa_negative_when_systematic_disagreement():
    assert cohen_kappa({"tp": 0, "fp": 5, "fn": 5, "tn": 0}) == pytest.approx(-1.0)


def test_prf_values():
    p, r, f = precision_recall_f1({"tp": 6, "fp": 2, "fn": 3, "tn": 9}).values()
    assert p == pytest.approx(0.75) and r == pytest.approx(6 / 9)
    assert f == pytest.approx(2 * p * r / (p + r))


def test_prf_degenerate_none():
    out = precision_recall_f1({"tp": 0, "fp": 0, "fn": 0, "tn": 4})
    assert out == {"precision": None, "recall": None, "f1": None}


def test_prf_zero_tp_gives_zero_f1():
    out = precision_recall_f1({"tp": 0, "fp": 2, "fn": 2, "tn": 0})
    assert out["precision"] == 0.0 and out["recall"] == 0.0 and out["f1"] == 0.0


# ---- ROC AUC ----------------------------------------------------------------------------


def test_auc_perfect_and_inverted():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0


def test_auc_ties_counted_half():
    assert roc_auc([0.5, 0.5], [False, True]) == 0.5


def test_auc_all_equal_scores_is_half():
    assert roc_auc([0.3] * 6, [True, False, True, False, True, False]) == 0.5


def test_auc_known_mixed():
    # positives: .8 .4 ; negatives: .6 .2 ; wins: .8>.6,.8>.2,.4>.2 =3 of 4
    assert roc_auc([0.8, 0.4, 0.6, 0.2], [True, True, False, False]) == pytest.approx(0.75)


@pytest.mark.parametrize("labels", [[True, True], [False, False], []])
def test_auc_one_class_or_empty_is_none(labels):
    assert roc_auc([0.1] * len(labels), labels) is None


def test_auc_length_mismatch():
    with pytest.raises(ValueError):
        roc_auc([0.1], [True, False])


# ---- Brier --------------------------------------------------------------------------------


def test_brier_known():
    assert brier_score([1.0, 0.0], [True, False]) == 0.0
    assert brier_score([0.0, 1.0], [True, False]) == 1.0
    assert brier_score([0.5, 0.5], [True, False]) == 0.25


def test_brier_empty_none():
    assert brier_score([], []) is None


def test_brier_ignores_out_of_range_scores():
    assert brier_score([1.0, 7.0, math.nan], [True, False, True]) == 0.0


# ---- reliability table ----------------------------------------------------------------------


def test_reliability_five_equal_count_bins():
    scores = [i / 10 for i in range(10)]
    p_yes = [0.0] * 5 + [1.0] * 5
    bins = reliability_table(scores, p_yes, bins=5)
    assert [b["n"] for b in bins] == [2, 2, 2, 2, 2]
    assert [b["mean_p_yes"] for b in bins] == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert bins[0]["score_min"] == 0.0 and bins[-1]["score_max"] == 0.9


def test_reliability_uneven_counts_sum_to_n():
    bins = reliability_table([i for i in range(13)], [0.5] * 13)
    assert sum(b["n"] for b in bins) == 13 and len(bins) == 5
    assert max(b["n"] for b in bins) - min(b["n"] for b in bins) <= 1


def test_reliability_fewer_items_than_bins():
    bins = reliability_table([0.1, 0.9], [0.0, 1.0])
    assert [b["n"] for b in bins] == [1, 1]


def test_reliability_empty():
    assert reliability_table([], []) == []


# ---- disagreements ----------------------------------------------------------------------------


def result(id, decision, p, score=None, status="ok", **kw):
    label = jev_label(p, T)
    return {
        "id": id, "status": status, "decision": decision, "jev_p": p, "label": label,
        "score": score, "score_name": "score" if score is not None else None,
        "a": f"a{id}", "b": f"b{id}", "gates": {"p": "auto"},
        "usage": {"input_tokens": 10, "output_tokens": 2}, "flags": [], **kw,
    }


def test_disagreements_sorted_by_certainty_desc():
    rs = [
        result("lo", True, 0.1),    # certainty .9
        result("hi", True, 0.0),    # certainty 1.0
        result("agree", True, 0.99),
        result("ind", True, 0.5),   # indeterminate -> not a disagreement
    ]
    out = disagreements(rs)
    assert [d["id"] for d in out] == ["hi", "lo"]
    assert out[0]["a"] == "ahi" and out[0]["b"] == "bhi"


def test_disagreements_exclude_errors():
    assert disagreements([result("e", True, None, status="error")]) == []


# ---- summarize -----------------------------------------------------------------------------------


def test_summarize_basic():
    rs = [
        result("1", True, 0.99, score=0.9),
        result("2", True, 0.01, score=0.7),   # disagreement
        result("3", False, 0.02, score=0.2),
        result("4", False, 0.97, score=0.1),  # disagreement
        result("5", True, 0.5, score=0.5),    # indeterminate
    ]
    s = summarize(rs, "matcher", T)
    assert s["has_valid_evidence"] is True
    assert s["counts"]["total"] == 5 and s["counts"]["ok"] == 5
    assert s["counts"]["indeterminate"] == 1 and s["counts"]["labeled"] == 4
    assert s["agreement"]["n"] == 4 and s["agreement"]["agree"] == 2
    assert s["agreement"]["rate"] == 0.5
    assert len(s["agreement"]["ci95"]) == 2
    assert s["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert s["kappa"] == pytest.approx(0.0)
    assert len(s["disagreements"]) == 2
    assert s["usage"] == {"input_tokens": 50, "output_tokens": 10}
    assert s["auc"] == pytest.approx(0.5)
    assert "juez basado en un modelo" in s["caveat"].lower()
    assert s["api_failure_rate"] == 0.0


def test_summarize_failures_excluded_and_rate_reported():
    rs = [
        result("1", True, 0.99),
        result("2", True, None, status="error"),
        result("3", True, None, status="error"),
        {"id": "4", "status": "skipped", "skip_reason": "duplicate_pair"},
    ]
    s = summarize(rs, "matcher", T)
    assert s["counts"]["error"] == 2 and s["counts"]["skipped"] == 1
    assert s["api_failure_rate"] == pytest.approx(2 / 3)
    assert s["agreement"]["n"] == 1 and s["agreement"]["rate"] == 1.0
    assert s["counts"]["skipped_by_reason"] == {"duplicate_pair": 1}


def test_summarize_zero_valid_items():
    s = summarize([result("1", True, None, status="error")], "matcher", T)
    assert s["has_valid_evidence"] is False
    assert s["agreement"]["rate"] is None and s["kappa"] is None and s["auc"] is None
    assert s["api_failure_rate"] == 1.0


def test_summarize_empty():
    s = summarize([], "matcher", T)
    assert s["has_valid_evidence"] is False
    assert s["api_failure_rate"] is None
    assert s["counts"]["total"] == 0


def test_summarize_all_indeterminate_is_not_valid_evidence_for_agreement():
    rs = [result(str(i), True, 0.5) for i in range(3)]
    s = summarize(rs, "matcher", T)
    assert s["counts"]["indeterminate"] == 3
    assert s["agreement"]["n"] == 0 and s["agreement"]["rate"] is None
    assert s["kappa"] is None and s["has_valid_evidence"] is False


def test_summarize_all_same_class_kappa_undefined_auc_none():
    rs = [result(str(i), True, 0.99, score=0.9) for i in range(4)]
    s = summarize(rs, "matcher", T)
    assert s["agreement"]["rate"] == 1.0
    assert s["kappa"] is None and s["auc"] is None


def test_summarize_normalizer_scores_names_reported():
    rs = [result("1", True, 0.99, score=0.9, score_name="confiabilidad")]
    rs[0]["score_name"] = "confiabilidad"
    s = summarize(rs, "normalizador", T)
    assert s["reliability"]["score_names"] == ["confiabilidad"]


def test_summarize_gate_distribution():
    rs = [result("1", True, 0.99), result("2", True, 0.7)]
    rs[1]["gates"] = {"p": "review"}
    s = summarize(rs, "matcher", T)
    assert s["gates"]["p"] == {"auto": 1, "review": 1, "escalate": 0}


def test_summarize_includes_load_info():
    s = summarize([], "matcher", T, load_info={"total_rows": 500, "capped": True, "max_rows": 200})
    assert s["input"]["capped"] is True
