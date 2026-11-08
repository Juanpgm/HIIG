"""cali_address.final_evidence: selection rule, Wilson, reliability bins, paired bootstrap, outcome shares (tiny fixtures)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cali_address import final_evidence as fe


def rows(spec):
    """spec: list of (estado, predial, lot_ok, man_ok, raw_ok, band, fuente, p_pred, p_man)."""
    out = []
    for i, (estado, predial, lot_ok, man_ok, raw_ok, band, fuente, pp, pm) in enumerate(spec):
        out.append({"raw_address": f"a{i}", "estado": estado, "numero_predial_nacional": predial,
                    "correct_predial_lot": lot_ok, "correct_manzana": man_ok, "raw_top1_correct_manzana": raw_ok,
                    "banda_confiabilidad": band, "fuente": fuente, "p_predio_cal": pp, "p_manzana_cal": pm,
                    "confianza": 0.9})
    return pd.DataFrame(out)


BASE = [
    ("OK", "P1", True, True, True, "alta", "cert", 0.99, 0.99),
    ("OK", "P2", False, True, True, "media", "pcer", 0.95, 0.99),
    ("OK", None, None, True, True, "baja", "cert", 0.5, 0.8),
    ("OK", "P4", True, False, False, "alta", "cert", 0.99, 0.99),
    ("SIN_MATCH", None, None, None, True, None, None, None, None),
    ("NO_PARSEABLE", None, None, None, False, None, None, None, None),
]


# ------------------------------------------------------------------------------------------------ wilson
def test_wilson_matches_known_values_and_is_inside_unit_interval():
    lo, hi = fe.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert fe.wilson(0, 10)[0] == 0.0 and fe.wilson(10, 10)[1] == 1.0
    assert 0.0 <= fe.wilson(1, 3)[0] <= fe.wilson(1, 3)[1] <= 1.0


def test_wilson_of_an_empty_sample_is_nan_and_bad_counts_are_rejected():
    lo, hi = fe.wilson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    with pytest.raises(ValueError):
        fe.wilson(5, 3)
    with pytest.raises(ValueError):
        fe.wilson(-1, 3)


# ------------------------------------------------------------------------------------------------ run_metrics
def test_run_metrics_counts_follow_the_evaluation_definitions():
    m = fe.run_metrics(rows(BASE))
    assert m["n"] == 6 and m["ok"] == 4
    assert m["rows_emitting_predial"] == 3
    assert m["lot_emitted_scored"] == 3 and m["lot_correct"] == 2 and m["lot_precision"] == pytest.approx(2 / 3)
    assert m["manzana_scored"] == 4 and m["manzana_correct"] == 3 and m["manzana_precision"] == pytest.approx(0.75)
    assert m["lost_correct"] == 1                              # SIN_MATCH with the raw top-1 right; NO_PARSEABLE wrong


def test_run_metrics_on_empty_and_all_unknown_rows_is_nan_not_an_error():
    empty = fe.run_metrics(rows([]).reindex(columns=rows(BASE).columns))
    assert empty["n"] == 0 and empty["ok"] == 0 and math.isnan(empty["lot_precision"])
    unknown = fe.run_metrics(rows([("OK", "P1", None, None, None, "alta", "cert", 0.9, 0.9)]))
    assert unknown["lot_emitted_scored"] == 0 and math.isnan(unknown["manzana_precision"]) and unknown["lost_correct"] == 0


def test_run_metrics_requires_the_columns():
    with pytest.raises(KeyError):
        fe.run_metrics(pd.DataFrame({"estado": ["OK"]}))


# ------------------------------------------------------------------------------------------------ selection
def metrics(tag_ok, lot_prec, n_emit, man_prec, n=289):
    return {"n": n, "ok": tag_ok, "lot_precision": lot_prec, "lot_emitted_scored": n_emit, "gold_manzana": man_prec}


def test_selection_picks_the_highest_lot_precision_among_runs_that_keep_the_baseline_coverage():
    base = {"gold_ok_coverage": 278 / 289}
    runs = {
        "a": {"gold_ok_coverage": 281 / 289, "gold_lot_precision": 0.980, "gold_lot_n": 250, "all_manzana_precision": 0.86},
        "b": {"gold_ok_coverage": 270 / 289, "gold_lot_precision": 0.999, "gold_lot_n": 250, "all_manzana_precision": 0.9},
        "c": {"gold_ok_coverage": 280 / 289, "gold_lot_precision": 0.990, "gold_lot_n": 250, "all_manzana_precision": 0.85},
    }
    sel = fe.select_candidate(runs, base)
    assert sel["selected"] == "c" and sel["eligible"] == ["a", "c"] and "b" not in sel["eligible"]


def test_selection_coverage_boundary_is_inclusive_and_nobody_eligible_selects_nothing():
    base = {"gold_ok_coverage": 0.96}
    equal = {"x": {"gold_ok_coverage": 0.96, "gold_lot_precision": 0.9, "gold_lot_n": 100, "all_manzana_precision": 0.8}}
    assert fe.select_candidate(equal, base)["selected"] == "x"
    below = {"x": {"gold_ok_coverage": 0.9599, "gold_lot_precision": 0.9, "gold_lot_n": 100, "all_manzana_precision": 0.8}}
    sel = fe.select_candidate(below, base)
    assert sel["selected"] is None and sel["eligible"] == [] and "no run" in sel["reason"]


def test_selection_tie_within_one_gold_row_is_broken_by_all_rows_manzana_precision():
    base = {"gold_ok_coverage": 0.9}
    runs = {   # 0.98805 vs 0.98824 on ~250 emitting rows: 0.0002 < one row (0.004) -> tie
        "s42": {"gold_ok_coverage": 0.97, "gold_lot_precision": 248 / 251, "gold_lot_n": 251, "all_manzana_precision": 0.8614},
        "s43": {"gold_ok_coverage": 0.97, "gold_lot_precision": 252 / 255, "gold_lot_n": 255, "all_manzana_precision": 0.8606},
    }
    sel = fe.select_candidate(runs, base)
    assert sel["selected"] == "s42" and sel["tie"] == ["s42", "s43"] and "manzana" in sel["reason"]


def test_selection_a_gap_of_more_than_one_row_is_not_a_tie_and_equal_everything_keeps_input_order():
    base = {"gold_ok_coverage": 0.9}
    runs = {"a": {"gold_ok_coverage": 0.97, "gold_lot_precision": 0.99, "gold_lot_n": 100, "all_manzana_precision": 0.5},
            "b": {"gold_ok_coverage": 0.97, "gold_lot_precision": 0.97, "gold_lot_n": 100, "all_manzana_precision": 0.9}}
    sel = fe.select_candidate(runs, base)
    assert sel["selected"] == "a" and sel["tie"] == ["a"]
    same = {"p": dict(runs["a"]), "q": dict(runs["a"])}
    assert fe.select_candidate(same, base)["selected"] == "p"


def test_selection_ignores_nan_metrics_and_validates_input():
    base = {"gold_ok_coverage": 0.9}
    runs = {"nan": {"gold_ok_coverage": 0.97, "gold_lot_precision": float("nan"), "gold_lot_n": 0, "all_manzana_precision": 0.9},
            "ok": {"gold_ok_coverage": 0.97, "gold_lot_precision": 0.9, "gold_lot_n": 10, "all_manzana_precision": 0.1}}
    assert fe.select_candidate(runs, base)["selected"] == "ok"
    with pytest.raises(ValueError):
        fe.select_candidate({}, base)
    with pytest.raises(ValueError):
        fe.select_candidate(runs, {})


# ------------------------------------------------------------------------------------------------ reliability bins
def test_reliability_bins_are_equal_count_and_cover_every_row_once():
    p = np.linspace(0.05, 0.95, 100)
    y = (p > 0.5).astype(float)
    bins = fe.reliability_bins(p, y, bins=10)
    assert [b["n"] for b in bins] == [10] * 10
    assert bins[0]["predicted"] < bins[-1]["predicted"] and bins[0]["observed"] == 0.0 and bins[-1]["observed"] == 1.0
    assert sum(b["n"] for b in bins) == 100


def test_reliability_bins_drop_nan_pairs_and_shrink_to_the_sample_size():
    p = np.array([0.9, np.nan, 0.5, 0.1])
    y = np.array([1.0, 1.0, np.nan, 0.0])
    bins = fe.reliability_bins(p, y, bins=10)
    assert sum(b["n"] for b in bins) == 2 and len(bins) == 2
    assert fe.reliability_bins(np.array([]), np.array([])) == []
    with pytest.raises(ValueError):
        fe.reliability_bins(np.array([0.1, 0.2]), np.array([1.0]))


# ------------------------------------------------------------------------------------------------ bootstrap
def pair():
    old = rows(BASE * 20)
    new = old.copy()
    new.loc[new.index % 6 == 3, "correct_manzana"] = True            # the one wrong manzana of every block is fixed
    return old, new


def test_paired_bootstrap_is_deterministic_for_a_seed_and_differs_between_seeds():
    old, new = pair()
    a = fe.paired_bootstrap(old, new, "manzana_precision", resamples=500, seed=7)
    b = fe.paired_bootstrap(old, new, "manzana_precision", resamples=500, seed=7)
    c = fe.paired_bootstrap(old, new, "manzana_precision", resamples=500, seed=8)
    assert a == b and a != c


def test_paired_bootstrap_point_estimate_and_interval_cover_the_difference():
    old, new = pair()
    r = fe.paired_bootstrap(old, new, "manzana_precision", resamples=1000, seed=1)
    assert r["old"] == pytest.approx(0.75) and r["new"] == pytest.approx(1.0) and r["diff"] == pytest.approx(0.25)
    assert r["lo"] <= r["diff"] <= r["hi"] and r["p_diff_lt0"] == 0.0 and r["resamples"] == 1000


def test_paired_bootstrap_of_identical_runs_has_a_zero_difference_everywhere():
    old, _ = pair()
    for metric in ("ok", "lot_precision", "manzana_precision", "lost_correct"):
        r = fe.paired_bootstrap(old, old.copy(), metric, resamples=200, seed=3)
        assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_paired_bootstrap_counts_metrics_are_sums_and_unknown_metric_or_misaligned_rows_are_rejected():
    old, new = pair()
    r = fe.paired_bootstrap(old, new, "ok", resamples=100, seed=1)
    assert r["old"] == 80 and r["diff"] == 0
    with pytest.raises(ValueError):
        fe.paired_bootstrap(old, new, "nope", resamples=10, seed=1)
    shifted = new.copy()
    shifted["raw_address"] = shifted["raw_address"] + "x"
    with pytest.raises(ValueError):
        fe.paired_bootstrap(old, shifted, "ok", resamples=10, seed=1)
    with pytest.raises(ValueError):
        fe.paired_bootstrap(old, new.iloc[:-1], "ok", resamples=10, seed=1)
    with pytest.raises(ValueError):
        fe.paired_bootstrap(old, new, "ok", resamples=0, seed=1)


# ------------------------------------------------------------------------------------------------ shares and tables
def test_outcome_shares_partition_the_rows():
    s = fe.outcome_shares(rows(BASE))
    assert s["n"] == 6
    assert s["counts"] == {"ok_predial": 3, "ok_manzana_only": 1, "sin_match": 1, "no_parseable": 1, "other": 0}
    assert sum(s["counts"].values()) == s["n"] and sum(s["shares"].values()) == pytest.approx(1.0)


def test_outcome_shares_unknown_estado_goes_to_other_and_empty_is_zero():
    r = rows([("RARO", None, None, None, None, None, None, None, None)])
    assert fe.outcome_shares(r)["counts"]["other"] == 1
    assert fe.outcome_shares(rows(BASE).iloc[0:0])["n"] == 0


def test_band_table_reports_wilson_intervals_and_ignores_rows_without_band():
    t = fe.band_table(rows(BASE))
    assert [r["band"] for r in t] == ["alta", "media", "baja"]
    alta = t[0]
    assert alta["n"] == 2 and alta["lot_scored"] == 2 and alta["lot_correct"] == 2 and t[1]["lot_correct"] == 0
    assert alta["lot_lo"] <= alta["lot_precision"] <= alta["lot_hi"]
    baja = t[2]
    assert baja["lot_scored"] == 0 and math.isnan(baja["lot_precision"]) and baja["manzana_scored"] == 1


def test_fuente_table_groups_ok_rows_by_source_and_handles_a_missing_column():
    t = fe.fuente_table(rows(BASE))
    assert {r["fuente"] for r in t} == {"cert", "pcer"}
    cert = next(r for r in t if r["fuente"] == "cert")
    assert cert["n"] == 3 and cert["lot_scored"] == 2
    assert fe.fuente_table(rows(BASE).drop(columns=["fuente"])) == []


# ---------------------------------------------------------------------------------------- JEV reliability cross-check
def jev_rows(spec):
    """spec: list of (status, label, jev_p, band, fuente, p_predio, noise)."""
    return pd.DataFrame([{"status": st, "label": lab, "jev_p": jp, "tag_banda_confiabilidad": b, "tag_fuente": f,
                          "extra_p_predio_cal": pp, "extra_noise": nz} for st, lab, jp, b, f, pp, nz in spec])


def test_jev_crosscheck_counts_and_group_means_ignore_errors_and_skipped_rows():
    frame = jev_rows([
        ("ok", "yes", 0.95, "alta", "nomenclatura_cert", 0.99, "light"),
        ("ok", "yes", 0.90, "alta", "nomenclatura_pcer", 0.97, "heavy"),
        ("ok", "indeterminate", 0.50, "media", "nomenclatura_cert", 0.90, "light"),
        ("ok", "no", 0.05, "baja", "catastro", 0.50, "heavy"),
        ("error", None, None, "alta", "nomenclatura_cert", 0.98, "light"),
        ("skipped", None, None, None, None, None, "heavy"),
    ])
    out = fe.jev_crosscheck(frame, resamples=50, seed=1)
    assert out["counts"] == {"total": 6, "judged": 4, "error": 1, "skipped": 1, "yes": 2, "no": 1, "indeterminate": 1}
    assert out["decisive_no_rate"] == pytest.approx(0.25) and out["mean_jev_p"] == pytest.approx((0.95 + 0.9 + 0.5 + 0.05) / 4)
    alta = out["by_band"]["alta"]
    assert alta["judged"] == 2 and alta["mean_jev_p"] == pytest.approx(0.925) and alta["decisive_no"] == 0
    assert out["by_band"]["baja"]["decisive_no_rate"] == 1.0
    assert set(out["by_fuente"]) == {"nomenclatura_cert", "nomenclatura_pcer", "catastro"}
    assert out["by_noise"]["heavy"]["judged"] == 2


def test_jev_crosscheck_spearman_is_one_for_a_monotone_relation_with_a_degenerate_interval():
    frame = jev_rows([("ok", "yes", 0.1 * i, "alta", "c", 0.05 * i, "light") for i in range(1, 9)])
    sp = fe.jev_crosscheck(frame, resamples=200, seed=3)["spearman_p_predio"]
    assert sp["n"] == 8 and sp["rho"] == pytest.approx(1.0) and sp["lo"] == pytest.approx(1.0) and sp["hi"] == pytest.approx(1.0)


def test_jev_crosscheck_spearman_is_negative_for_an_inverse_relation_and_is_reproducible():
    frame = jev_rows([("ok", "yes", 1.0 - 0.1 * i, "alta", "c", 0.05 * i + (i % 2) * 0.01, "light") for i in range(1, 10)])
    a = fe.jev_crosscheck(frame, resamples=300, seed=9)["spearman_p_predio"]
    b = fe.jev_crosscheck(frame, resamples=300, seed=9)["spearman_p_predio"]
    assert a["rho"] < -0.9 and a == b


def test_jev_crosscheck_spearman_is_undefined_for_constant_or_tiny_samples():
    const = jev_rows([("ok", "yes", 0.9, "alta", "c", 0.1 * i, "light") for i in range(1, 6)])
    assert math.isnan(fe.jev_crosscheck(const, resamples=20, seed=1)["spearman_p_predio"]["rho"])
    tiny = jev_rows([("ok", "yes", 0.9, "alta", "c", 0.5, "light"), ("ok", "no", 0.1, "alta", "c", 0.6, "light")])
    sp = fe.jev_crosscheck(tiny, resamples=20, seed=1)["spearman_p_predio"]
    assert sp["n"] == 2 and math.isnan(sp["rho"])


def test_jev_crosscheck_skips_rows_without_our_probability_and_handles_a_missing_column():
    frame = jev_rows([("ok", "yes", 0.2 * i, "alta", "c", None if i == 2 else 0.1 * i, "light") for i in range(1, 7)])
    assert fe.jev_crosscheck(frame, resamples=20, seed=1)["spearman_p_predio"]["n"] == 5
    assert fe.jev_crosscheck(frame.drop(columns=["extra_p_predio_cal"]), resamples=20, seed=1)["spearman_p_predio"] is None


def test_jev_crosscheck_empty_frame_and_no_judged_rows_do_not_crash():
    empty = fe.jev_crosscheck(jev_rows([]), resamples=10, seed=1)
    assert empty["counts"]["total"] == 0 and empty["decisive_no_rate"] is None and empty["by_band"] == {}
    none_judged = fe.jev_crosscheck(jev_rows([("error", None, None, "alta", "c", 0.9, "light")]), resamples=10, seed=1)
    assert none_judged["counts"]["judged"] == 0 and none_judged["mean_jev_p"] is None


def test_jev_crosscheck_requires_the_basic_columns_and_never_returns_text():
    with pytest.raises(KeyError):
        fe.jev_crosscheck(pd.DataFrame({"status": ["ok"]}), resamples=5, seed=1)
    frame = jev_rows([("ok", "yes", 0.9, "alta", "c", 0.9, "light")] * 3)
    frame["input_text"] = "CL 900 # 1"
    assert "CL 900" not in str(fe.jev_crosscheck(frame, resamples=5, seed=1))
