"""scripts/compare_indexes.py, scripts/compare_runs.py and retrain_compare.collect_run_metrics on tiny fixtures."""

from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from tiny import tiny_docs

from cali_address import retrain_compare as rc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _docs(n=8, shift=0.0):
    d = tiny_docs(n)
    d["direccion"] = [f"CL {900 + i // 4} # {910 + i % 4} - {920 + i}" for i in range(n)]
    d["centroid_lat"] = d["centroid_lat"] + shift
    return d


def _make_old(tmp_path, n=8):
    old = tmp_path / "old"
    (old / "splits").mkdir(parents=True)
    d = _docs(n)
    d.to_parquet(old / "catastro_docs.parquet")
    gold = pd.DataFrame({"gt_predial": list(d["numero_predial_nacional"][:3]) + ["ZZZ"], "gt_quality": "gold"})
    gold.to_parquet(old / "splits" / "dev.parquet")
    gold.iloc[:2].to_parquet(old / "splits" / "frozen_test.parquet")
    return old


def _make_new(tmp_path, n=8, shift=0.0009):
    new = tmp_path / "index_new"
    new.mkdir()
    d = _docs(n, shift)
    d["fuente"] = "nomenclatura_cert"
    d["npn_catastro"] = d["numero_predial_nacional"]
    d["npn_choice"] = "single"
    d.to_parquet(new / "catastro_docs.parquet")
    (new / "index_meta.json").write_text(json.dumps({
        "n_docs": n, "tiers": ["cert", "catastro"], "counts_by_fuente": {"nomenclatura_cert": n},
        "accounting": {"input": 10, "total": 10, "buckets": {"dropped": 0}, "ok": True},
        "parse_failure_rate": 0.0, "crs": {"source": "EPSG:6249"}, "files": {}}), encoding="utf-8")
    return new


# ---------------------------------------------------------------- compare_indexes
def test_compare_indexes_writes_aggregates_next_to_the_index_by_default(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _make_new(tmp_path)
    assert script.main(["--old", str(old), "--new", str(new)]) == 0
    out = json.load(open(new / "compare_indexes.json"))
    assert out["profile"]["n_docs"] == {"old": 8, "new": 8}
    assert out["gold_npn_coverage"]["old"]["all"]["n_gold_rows"] == 6 and out["gold_npn_coverage"]["old"]["by_split"]["dev"]["n_gold_rows"] == 4 and out["gold_npn_coverage"]["new_ge_old"] is True
    assert out["coordinate_shift"]["n_shared_npn"] == 8 and 99 < out["coordinate_shift"]["quantiles_m"]["p50"] < 100.5
    assert out["index_meta"]["n_docs"] == 8 and out["index_meta"]["tiers"] == ["cert", "catastro"]


def _lot_new(tmp_path, keep=6, unit="000000007"):
    """A new index whose documents carry a unit suffix and cover only the first ``keep`` lots of the old one."""
    new = _make_new(tmp_path)
    d = pd.read_parquet(new / "catastro_docs.parquet").iloc[:keep].copy()
    d["numero_predial_nacional"] = [("7600101" + f"{i:014d}") + unit for i in range(keep)]    # tiny NPNs are the 21-char lots
    d.to_parquet(new / "catastro_docs.parquet")
    return new


def test_compare_indexes_coverage_is_lot_level_and_ignores_the_unit_suffix(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _lot_new(tmp_path, keep=2)
    assert script.main(["--old", str(old), "--new", str(new)]) == 0
    out = json.load(open(new / "compare_indexes.json"))
    cov = out["gold_npn_coverage"]
    assert cov["old"]["all"]["share_present"] == pytest.approx(5 / 6)       # 5 of 6 gold rows exist in the old index ("ZZZ" does not)
    assert cov["new"]["all"]["n_present"] == 4 and cov["new"]["all"]["n_missing"] == 2     # lots 0,1 in dev and in frozen
    assert cov["new_ge_old"] is False and cov["delta_pp"] < 0
    assert cov["margin_pp"] == 2.0 and cov["new_ge_old_minus_margin"] is False


def test_compare_indexes_reports_production_doc_coverage_by_lot_and_by_doc_key(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _lot_new(tmp_path, keep=6)
    script.main(["--old", str(old), "--new", str(new)])
    prod = json.load(open(new / "compare_indexes.json"))["production_docs_coverage"]
    assert prod["n_old_docs"] == 8
    assert prod["by_lot"] == {"n_considered": 8, "n_covered": 6, "n_lost": 2, "share_covered": 0.75}
    assert prod["by_doc_key"]["n_considered"] == 8 and prod["by_doc_key"]["n_covered"] == 6


def test_compare_indexes_reports_all_with_gt_coverage_by_lot_per_split(tmp_path):
    script = _load_script("compare_indexes")
    old = _make_old(tmp_path)
    extra = pd.DataFrame({"gt_predial": ["7600101" + f"{i:014d}" for i in (4, 5)], "gt_quality": "noisy"})
    dev = pd.read_parquet(old / "splits" / "dev.parquet")
    pd.concat([dev, extra], ignore_index=True).to_parquet(old / "splits" / "dev.parquet")
    new = _lot_new(tmp_path, keep=5)
    script.main(["--old", str(old), "--new", str(new)])
    out = json.load(open(new / "compare_indexes.json"))
    gold, allgt = out["gold_npn_coverage"]["new"]["by_split"]["dev"], out["all_with_gt_lot_coverage"]["new"]["by_split"]["dev"]
    assert gold["n_gold_rows"] == 4 and allgt["n_gold_rows"] == 6        # the noisy rows only count in the all-with-GT view
    assert allgt["n_present"] == 4 and allgt["n_missing"] == 2           # lots 0,1,2 (gold) + 4 (noisy); "ZZZ" and lot 5 missing
    assert out["all_with_gt_lot_coverage"]["old"]["all"]["n_with_gt_predial"] >= 6


def test_compare_indexes_coverage_output_stays_aggregate_only(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _lot_new(tmp_path)
    script.main(["--old", str(old), "--new", str(new)])
    text = open(new / "compare_indexes.json").read()
    for npn in _docs()["numero_predial_nacional"]:
        assert npn not in text


def test_compare_indexes_output_has_no_addresses(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _make_new(tmp_path)
    script.main(["--old", str(old), "--new", str(new)])
    text = open(new / "compare_indexes.json").read()
    for addr in _docs()["direccion"]:
        assert addr not in text


def test_compare_indexes_out_option_leaves_the_index_directory_untouched(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _make_new(tmp_path)
    before = sorted(os.listdir(new))
    target = tmp_path / "elsewhere" / "cmp.json"
    assert script.main(["--old", str(old), "--new", str(new), "--out", str(target)]) == 0
    assert target.exists() and sorted(os.listdir(new)) == before


def test_compare_indexes_missing_inputs_exit_2_with_a_message(tmp_path, capsys):
    script = _load_script("compare_indexes")
    assert script.main(["--old", str(tmp_path / "nope"), "--new", str(tmp_path / "nope2")]) == 2
    assert "catastro_docs.parquet" in capsys.readouterr().err
    old = _make_old(tmp_path)
    assert script.main(["--old", str(old), "--new", str(tmp_path / "nope2")]) == 2


def test_compare_indexes_without_splits_still_reports_the_rest(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _make_new(tmp_path)
    for f in os.listdir(old / "splits"):
        os.remove(old / "splits" / f)
    assert script.main(["--old", str(old), "--new", str(new)]) == 0
    out = json.load(open(new / "compare_indexes.json"))
    assert out["gold_npn_coverage"] is None and "splits" in out["warnings"][0]


def test_compare_indexes_without_index_meta_still_runs(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _make_new(tmp_path)
    os.remove(new / "index_meta.json")
    assert script.main(["--old", str(old), "--new", str(new)]) == 0
    assert json.load(open(new / "compare_indexes.json"))["index_meta"] is None


def test_compare_indexes_does_not_modify_the_input_docs(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _make_new(tmp_path)
    snap = {p: (p).read_bytes() for p in [old / "catastro_docs.parquet", new / "catastro_docs.parquet", new / "index_meta.json"]}
    script.main(["--old", str(old), "--new", str(new)])
    assert all(p.read_bytes() == b for p, b in snap.items())


# ---------------------------------------------------------------- collect_run_metrics
def _rows(n=4, conf=0.9):
    return pd.DataFrame({
        "raw_address": [f"addr {i}" for i in range(n)], "estado": "OK", "confianza": conf,
        "correct_predial": [True, True, False, True][:n], "correct_manzana": True,
        "gt_predial": "P1", "numero_predial_nacional": "P1"})


def _reliability(ship=True, hybrid=None, logistic=None):
    hy = {"ece": 0.06, "brier": 0.10, "auc": 0.80, **(hybrid or {})}
    lg = {"ece": 0.07, "brier": 0.10, "auc": 0.75, **(logistic or {})}
    return {"kind": "hybrid", "ship_rule_passed": ship,
            "cv": {"pooled": {"predio": {"hybrid": hy, "logistic": lg},
                              "manzana": {"hybrid": {**hy, "ece": 0.04}, "logistic": lg}}}}


def _exp(path, tag="e", gold=True, frozen=True, reliability=True, untouched=True, rows=None):
    path.mkdir(parents=True, exist_ok=True)
    payload = {"leaderboard_row": {"dev_gold_ok": 277, "dev_gold_predial_precision": 0.98,
                                   "dev_gold_predial_lot_precision": 0.985, "dev_predial_lot_precision": 0.7,
                                   "dev_gold_lot_missing_from_index": 4,
                                   "dev_manzana_precision": 0.84, "dev_lost_correct": 96},
               "production_untouched": untouched}
    if frozen:
        payload["frozen_test"] = {"gold": {"test_gold_ok": 400, "test_gold_predial_precision": 0.99,
                                           "test_gold_predial_lot_precision": 0.995}}
    json.dump(payload, open(path / "experiment.json", "w"))
    if gold:
        (rows if rows is not None else _rows()).to_parquet(path / "strict_eval_dev_gold_rows.parquet")
    if reliability:
        json.dump(_reliability(), open(path / "reliability.json", "w"))
    return path


def test_collect_run_metrics_reads_everything_available(tmp_path):
    m = rc.collect_run_metrics(str(_exp(tmp_path / "e")))
    assert m["dev_gold_ok"] == 277 and m["dev_gold_predial_precision"] == 0.98 and m["dev_lost_correct"] == 96
    assert m["frozen_gold_ok"] == 400 and m["frozen_gold_predial_precision"] == 0.99
    assert m["dev_gold_predial_lot_precision"] == 0.985 and m["frozen_gold_predial_lot_precision"] == 0.995
    assert m["dev_predial_lot_precision"] == 0.7 and m["dev_gold_lot_missing_from_index"] == 4
    assert m["production_untouched"] is True and m["ship_rule_passed"] is True
    assert m["calib_error_cv_predio"] == 0.06 and m["calib_error_cv_manzana"] == 0.04
    point = next(c for c in m["confidence_curve"] if c["threshold"] == 0.7)
    assert point["n_ok_at_t"] == 4 and point["predial_precision"] == 0.75


def test_collect_run_metrics_omits_lot_metrics_of_an_older_run(tmp_path):
    e = _exp(tmp_path / "e")
    payload = json.load(open(e / "experiment.json"))
    for key in ("dev_gold_predial_lot_precision", "dev_predial_lot_precision", "dev_gold_lot_missing_from_index"):
        del payload["leaderboard_row"][key]
    del payload["frozen_test"]["gold"]["test_gold_predial_lot_precision"]
    json.dump(payload, open(e / "experiment.json", "w"))
    m = rc.collect_run_metrics(str(e))
    assert not [k for k in m if "lot" in k and k != "confidence_curve"] and m["dev_gold_predial_precision"] == 0.98


def test_collect_run_metrics_keeps_a_nan_lot_precision_so_the_check_can_fail_it(tmp_path):
    e = _exp(tmp_path / "e")
    payload = json.load(open(e / "experiment.json"))
    payload["leaderboard_row"]["dev_gold_predial_lot_precision"] = float("nan")
    json.dump(payload, open(e / "experiment.json", "w"))
    assert math.isnan(rc.collect_run_metrics(str(e))["dev_gold_predial_lot_precision"])


def test_collect_run_metrics_omits_what_is_missing_instead_of_inventing_it(tmp_path):
    m = rc.collect_run_metrics(str(_exp(tmp_path / "e", gold=False, frozen=False, reliability=False)))
    assert "frozen_gold_ok" not in m and "confidence_curve" not in m and "ship_rule_passed" not in m
    assert m["dev_gold_ok"] == 277


def test_collect_run_metrics_on_an_empty_or_missing_directory_is_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    assert rc.collect_run_metrics(str(tmp_path / "empty")) == {}
    assert rc.collect_run_metrics(str(tmp_path / "nope")) == {}


def test_collect_run_metrics_survives_corrupt_json_files(tmp_path):
    e = _exp(tmp_path / "e")
    (e / "experiment.json").write_text("{corrupt")
    (e / "reliability.json").write_text("[]")
    m = rc.collect_run_metrics(str(e))
    assert "dev_gold_ok" not in m and "ship_rule_passed" not in m and "confidence_curve" in m


def test_collect_run_metrics_keeps_nan_so_the_check_can_fail_it(tmp_path):
    e = _exp(tmp_path / "e")
    payload = json.load(open(e / "experiment.json"))
    payload["leaderboard_row"]["dev_gold_predial_precision"] = float("nan")
    json.dump(payload, open(e / "experiment.json", "w"))
    assert math.isnan(rc.collect_run_metrics(str(e))["dev_gold_predial_precision"])


def test_collect_run_metrics_unknown_reliability_kind_gives_no_calibration(tmp_path):
    e = _exp(tmp_path / "e")
    r = json.load(open(e / "reliability.json"))
    r["kind"] = "mystery"
    json.dump(r, open(e / "reliability.json", "w"))
    m = rc.collect_run_metrics(str(e))
    assert "calib_error_cv_predio" not in m and m["ship_rule_passed"] is True


# ---------------------------------------------------------------- compare_runs
def _two_runs(tmp_path, with_final=True):
    old = _exp(tmp_path / "old_run", rows=_rows())
    new_rows = _rows()
    new_rows["correct_predial"] = [True, False, True, True]
    new = _exp(tmp_path / "new_run", rows=new_rows)
    if with_final:
        _rows(2).to_parquet(old / "strict_eval_final_gold_rows.parquet")
        _rows(2).to_parquet(new / "strict_eval_final_gold_rows.parquet")
    return old, new


def _thresholds_file(tmp_path):
    p = tmp_path / "thr.json"
    ref = {"dev_gold_n": 289, "dev_gold_ok": 277, "dev_gold_predial_precision": 0.98,
           "dev_gold_manzana_precision": 0.99, "dev_manzana_precision": 0.84, "dev_lost_correct": 96}
    p.write_text(json.dumps(rc.build_acceptance_thresholds(ref, created="t")))
    return p


def test_compare_runs_writes_counts_rows_and_acceptance(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    assert script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))]) == 0
    agree = json.load(open(new / "probe_agreement.json"))
    assert agree["n"] == 6 and agree["sets"] == ["dev_gold", "final_gold"]
    assert agree["counts"]["only_old"] == 1 and agree["counts"]["only_new"] == 1
    rows = pd.read_parquet(new / "probe_rows.parquet")
    assert len(rows) == 6 and "raw_address" in rows.columns
    acc = json.load(open(new / "acceptance.json"))
    assert {r["id"] for r in acc["results"]} == {f"A{i}" for i in range(1, 13)}
    assert "all_passed" in acc["summary"]


def test_compare_runs_json_has_counts_only(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))])
    for name in ("probe_agreement.json", "acceptance.json"):
        assert "addr " not in open(new / name).read()


def test_compare_runs_misaligned_rows_exit_2_and_write_nothing(tmp_path, capsys):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path, with_final=False)
    bad = _rows()
    bad.loc[1, "raw_address"] = "shuffled"
    bad.to_parquet(new / "strict_eval_dev_gold_rows.parquet")
    assert script.main(["--old", str(old), "--new", str(new)]) == 2
    err = capsys.readouterr().err
    assert "misaligned" in err.lower() and "shuffled" not in err
    assert not (new / "probe_agreement.json").exists() and not (new / "probe_rows.parquet").exists()


def test_compare_runs_missing_rows_exit_2(tmp_path, capsys):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path, with_final=False)
    os.remove(new / "strict_eval_dev_gold_rows.parquet")
    assert script.main(["--old", str(old), "--new", str(new)]) == 2
    assert "strict_eval" in capsys.readouterr().err


def test_compare_runs_uses_only_the_sets_both_runs_have(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path, with_final=False)
    _rows(2).to_parquet(old / "strict_eval_final_gold_rows.parquet")             # only the old run was scored on frozen
    assert script.main(["--old", str(old), "--new", str(new)]) == 0
    assert json.load(open(new / "probe_agreement.json"))["sets"] == ["dev_gold"]


def test_compare_runs_skips_acceptance_when_the_thresholds_are_missing(tmp_path, capsys):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    assert script.main(["--old", str(old), "--new", str(new), "--thresholds", str(tmp_path / "none.json")]) == 0
    assert not (new / "acceptance.json").exists() and "acceptance" in capsys.readouterr().out.lower()


def test_compare_runs_skips_acceptance_when_a_run_has_no_experiment_json(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    os.remove(old / "experiment.json")
    assert script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))]) == 0
    assert (new / "probe_agreement.json").exists() and not (new / "acceptance.json").exists()


def test_compare_runs_extra_metrics_fill_the_gates_that_runs_cannot_know(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"candidate": {"pytest_green": True, "no_real_addresses_exit": 0}}))
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path)),
                 "--extra-metrics", str(extra)])
    acc = {r["id"]: r for r in json.load(open(new / "acceptance.json"))["results"]}
    assert acc["A9"]["passed"] is True and acc["A7"]["passed"] is None


def test_compare_runs_out_dir_option_and_idempotent_rerun(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    out = tmp_path / "out"
    args = ["--old", str(old), "--new", str(new), "--out-dir", str(out), "--thresholds", str(_thresholds_file(tmp_path))]
    assert script.main(args) == 0 and script.main(args) == 0
    assert sorted(os.listdir(out)) == ["acceptance.json", "probe_agreement.json", "probe_rows.parquet"]
    assert not (new / "probe_agreement.json").exists()


def test_compare_runs_compare_indexes_json_feeds_the_a7_inputs(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({
        "gold_npn_coverage": {"new": {"all": {"share_present": 0.97, "n_missing": 12}},
                              "old": {"all": {"share_present": 0.95}}},
        "index_meta": {"parse_failure_rate": 0.0, "accounting": {"buckets": {"dropped": 0}}}}))
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path)),
                 "--compare-indexes", str(ci)])
    a7 = {r["id"]: r for r in json.load(open(new / "acceptance.json"))["results"]}["A7"]
    status = {p["name"]: p["status"] for p in a7["parts"]}
    assert status["gold_lot_coverage_new"] == "pass" and status["parser_failure_rate_cert"] == "pass"
    assert status["gold_lot_missing_new"] == "pass"
    assert status["accounting_dropped"] == "pass" and status["crs_gates_passed"] == "missing"
    assert a7["passed"] is None


def test_from_compare_indexes_maps_lot_coverage_and_the_missing_count(tmp_path):
    script = _load_script("compare_runs")
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({"gold_npn_coverage": {"new": {"all": {"share_present": 0.75, "n_missing": 30}},
                                                    "old": {"all": {"share_present": 0.99}}}}))
    out = script._from_compare_indexes(str(ci))
    assert out == {"gold_lot_coverage_new": 0.75, "gold_lot_coverage_old": 0.99, "gold_lot_missing_new": 30}


@pytest.mark.parametrize("content", ["{corrupt", "[]", "{}", '{"gold_npn_coverage": null}'])
def test_from_compare_indexes_tolerates_missing_or_malformed_files(tmp_path, content):
    script = _load_script("compare_runs")
    ci = tmp_path / "ci.json"
    ci.write_text(content)
    assert script._from_compare_indexes(str(ci)) == {}
    assert script._from_compare_indexes(str(tmp_path / "nope.json")) == {}


def test_compare_runs_reports_lot_level_agreement_when_both_runs_have_the_lot_column(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path, with_final=False)
    o, n = _rows(), _rows()
    o["correct_predial_lot"] = [True, True, True, True]
    n["correct_predial_lot"] = [True, False, True, True]
    o.to_parquet(old / "strict_eval_dev_gold_rows.parquet")
    n.to_parquet(new / "strict_eval_dev_gold_rows.parquet")
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))])
    agree = json.load(open(new / "probe_agreement.json"))
    assert agree["lot_level"]["counts"] == {"both_ok_predial_match": 3, "only_old": 1, "only_new": 0, "both_wrong": 0}
    assert "counts" in agree and agree["counts"]["only_old"] + agree["counts"]["only_new"] >= 0     # exact buckets still there


def test_compare_runs_without_lot_columns_keeps_the_probe_agreement_as_before(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))])
    assert "lot_level" not in json.load(open(new / "probe_agreement.json"))


def test_compare_runs_reports_by_fuente_when_the_new_rows_carry_the_column(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path, with_final=False)
    n = _rows()
    n["correct_predial"] = [True, False, True, True]
    n["fuente"] = ["nomenclatura_cert", "nomenclatura_cert", "nomenclatura_pcer", None]
    n.to_parquet(new / "strict_eval_dev_gold_rows.parquet")
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))])
    by = json.load(open(new / "probe_agreement.json"))["by_fuente"]
    assert by["nomenclatura_cert"] == {"both_ok_predial_match": 1, "only_old": 1, "only_new": 0, "both_wrong": 0}
    assert by["nomenclatura_pcer"]["only_new"] == 1 and by["unknown"]["both_ok_predial_match"] == 1
    assert sum(sum(v.values()) for v in by.values()) == 4


def test_compare_runs_without_fuente_anywhere_has_no_by_fuente_section(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))])
    assert "by_fuente" not in json.load(open(new / "probe_agreement.json"))


def test_compare_runs_correct_run_enables_a12(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    fixed = _exp(tmp_path / "fixed_run")
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path)),
                 "--correct-run", str(fixed)])
    a12 = {r["id"]: r for r in json.load(open(new / "acceptance.json"))["results"]}["A12"]
    assert a12["status"] in ("pass", "fail", "not_evaluated") and a12["parts"]
    assert any(p["name"] == "dev_gold_ok_not_worse" and p["status"] == "pass" for p in a12["parts"])


# ---- A7 (c1)/(c2) from index_meta.json "oracle_ceiling" ----------------------------------------------------
CEIL = {"rows": 400, "prod_lot_matched": 396, "new_lot_matched": 380, "lot_absent_from_source": 20,
        "lot_lost_incomplete": 3, "lot_not_chosen": 1}


def test_from_compare_indexes_derives_the_oracle_ceiling_metrics(tmp_path):
    script = _load_script("compare_runs")
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({"index_meta": {"oracle_ceiling": CEIL}}))
    out = script._from_compare_indexes(str(ci))
    assert out["gold_lot_cond_new"] == pytest.approx(380 / 380) and out["gold_lot_cond_prod"] == pytest.approx(396 / 380)
    assert out["gold_coverage_loss"] == pytest.approx(16 / 400) and out["gold_inherent_loss"] == pytest.approx(20 / 400)


def test_compare_indexes_copies_the_oracle_ceiling_block(tmp_path):
    script = _load_script("compare_indexes")
    old, new = _make_old(tmp_path), _make_new(tmp_path)
    meta = json.load(open(new / "index_meta.json"))
    meta["oracle_ceiling"] = CEIL
    json.dump(meta, open(new / "index_meta.json", "w"))
    script.main(["--old", str(old), "--new", str(new)])
    assert json.load(open(new / "compare_indexes.json"))["index_meta"]["oracle_ceiling"] == CEIL


def test_compare_runs_a7_is_not_evaluated_when_the_builder_wrote_no_oracle_ceiling(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({"index_meta": {"parse_failure_rate": 0.0, "accounting": {"buckets": {"dropped": 0}}}}))
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path)),
                 "--compare-indexes", str(ci)])
    a7 = {r["id"]: r for r in json.load(open(new / "acceptance.json"))["results"]}["A7"]
    assert a7["passed"] is None and "gold_lot_cond_new" in a7["reason"]


def test_collect_run_metrics_reads_the_oracle_ceiling_of_a_linked_index_meta(tmp_path):
    e = _exp(tmp_path / "e")
    (e / "index_meta.json").write_text(json.dumps({"oracle_ceiling": CEIL}))
    m = rc.collect_run_metrics(str(e))
    assert m["gold_inherent_loss"] == pytest.approx(0.05) and m["gold_lot_not_chosen"] == 1
    (e / "index_meta.json").write_text("{corrupt")
    assert "gold_inherent_loss" not in rc.collect_run_metrics(str(e))


# ---- reliability: candidate-file fallback and the ship rule against the refit ------------------------------
def test_collect_run_metrics_falls_back_to_the_candidate_reliability_file(tmp_path):
    e = _exp(tmp_path / "e", reliability=False)
    json.dump(_reliability(ship=False), open(e / "reliability_hybrid_candidate.json", "w"))
    m = rc.collect_run_metrics(str(e))
    assert m["ship_rule_passed"] is False and m["ship_rule_vs_refit"] is True
    assert m["calib_error_cv_predio"] == 0.06 and m["calib_error_cv_manzana"] == 0.04


def test_reliability_json_wins_over_a_candidate_file(tmp_path):
    e = _exp(tmp_path / "e")
    json.dump(_reliability(hybrid={"ece": 0.5}), open(e / "reliability_hybrid_candidate.json", "w"))
    assert rc.collect_run_metrics(str(e))["calib_error_cv_predio"] == 0.06


def test_a_corrupt_reliability_json_falls_through_to_the_candidate(tmp_path):
    e = _exp(tmp_path / "e")
    (e / "reliability.json").write_text("{corrupt")
    json.dump(_reliability(ship=False), open(e / "reliability_hybrid_candidate.json", "w"))
    assert rc.collect_run_metrics(str(e))["ship_rule_passed"] is False


def test_no_reliability_file_at_all_gives_no_reliability_metrics(tmp_path):
    m = rc.collect_run_metrics(str(_exp(tmp_path / "e", reliability=False)))
    assert not [k for k in m if "ship" in k or "calib" in k]


@pytest.mark.parametrize("hybrid,logistic,expected", [
    ({"brier": 0.102}, {}, True),                   # exactly +0.002 passes
    ({"brier": 0.1021}, {}, False),                 # just above fails
    ({"auc": 0.75}, {}, True),                      # equal AUC passes
    ({"auc": 0.7499}, {}, False),
    ({"auc": float("nan")}, {}, False),
    ({"brier": float("nan")}, {}, False),
])
def test_ship_rule_vs_refit_boundaries(hybrid, logistic, expected):
    assert rc.ship_rule_vs_refit(_reliability(hybrid=hybrid, logistic=logistic)) is expected


def test_ship_rule_vs_refit_is_none_when_the_numbers_are_missing():
    rel = _reliability()
    del rel["cv"]["pooled"]["manzana"]["logistic"]
    assert rc.ship_rule_vs_refit(rel) is None
    assert rc.ship_rule_vs_refit({"kind": "hybrid"}) is None
    assert rc.ship_rule_vs_refit({"cv": {"pooled": {}}}) is None
    assert rc.ship_rule_vs_refit(_reliability(hybrid={"auc": "x"})) is None


def test_ship_rule_fails_in_either_head():
    rel = _reliability()
    rel["cv"]["pooled"]["manzana"]["hybrid"]["auc"] = 0.5
    assert rc.ship_rule_vs_refit(rel) is False


def test_collect_run_metrics_curve_carries_the_emitted_precision_key(tmp_path):
    rows = _rows()
    rows["correct_predial_lot"] = [True, True, False, True]
    m = rc.collect_run_metrics(str(_exp(tmp_path / "e", rows=rows)))
    point = next(c for c in m["confidence_curve"] if c["threshold"] == 0.7)
    assert point["n_emitting_predial"] == 4 and point["predial_lot_precision_emitted"] == 0.75


# ---------------------------------------------------------------- calibrated reliability (A6 of a calibrated candidate)
def _cal_report(ece_p=0.004, ece_m=0.002, brier_p=0.027, brier_m=0.005, auc_p=0.90, auc_m=0.90,
                refit_brier_p=0.027, refit_brier_m=0.005, refit_auc_p=0.90, refit_auc_m=0.90):
    def head(ece, brier, auc, rbrier, rauc):
        return {"oof": {"n": 100, "ece10": ece, "brier": brier, "auc": auc},
                "refit_oof": {"n": 100, "ece10": ece, "brier": rbrier, "auc": rauc}}
    return {"heads": {"predio": head(ece_p, brier_p, auc_p, refit_brier_p, refit_auc_p),
                      "manzana": head(ece_m, brier_m, auc_m, refit_brier_m, refit_auc_m)}}


def _cal_dir(path, report=None, write_report=True, kind="logistic_ext"):
    e = _exp(path, reliability=False)
    json.dump({"kind": kind, "version": "reliability-calib-v1-x", "predio": {}, "manzana": {}},
              open(e / "reliability_calibrated.json", "w"))
    if write_report:
        json.dump(report or _cal_report(), open(e / "fit_report.json", "w"))
    return e


def test_calibrated_reliability_feeds_a6_with_the_oof_ece_of_both_heads(tmp_path):
    m = rc.collect_run_metrics(str(_cal_dir(tmp_path / "e")))
    assert m["calib_error_cv_predio"] == 0.004 and m["calib_error_cv_manzana"] == 0.002
    assert m["ship_rule_vs_refit"] is True and m["calibrated_layer"] is True


def test_calibrated_reliability_takes_precedence_over_a_legacy_file_in_the_same_directory(tmp_path):
    e = _cal_dir(tmp_path / "e")
    json.dump(_reliability(), open(e / "reliability.json", "w"))
    m = rc.collect_run_metrics(str(e))
    assert m["calib_error_cv_predio"] == 0.004 and m["calibrated_layer"] is True


def test_calibrated_ship_rule_boundaries_are_inclusive_for_brier_and_exact_for_auc(tmp_path):
    ok = rc.collect_run_metrics(str(_cal_dir(tmp_path / "a", _cal_report(brier_p=0.029, refit_brier_p=0.027))))
    assert ok["ship_rule_vs_refit"] is True                                   # exactly refit + 0.002
    bad_brier = rc.collect_run_metrics(str(_cal_dir(tmp_path / "b", _cal_report(brier_m=0.0071, refit_brier_m=0.005))))
    assert bad_brier["ship_rule_vs_refit"] is False
    bad_auc = rc.collect_run_metrics(str(_cal_dir(tmp_path / "c", _cal_report(auc_p=0.8999, refit_auc_p=0.90))))
    assert bad_auc["ship_rule_vs_refit"] is False
    equal_auc = rc.collect_run_metrics(str(_cal_dir(tmp_path / "d", _cal_report(auc_m=0.90, refit_auc_m=0.90))))
    assert equal_auc["ship_rule_vs_refit"] is True


def test_calibrated_ship_rule_nan_fails_and_missing_refit_is_omitted(tmp_path):
    nan = rc.collect_run_metrics(str(_cal_dir(tmp_path / "a", _cal_report(auc_p=float("nan")))))
    assert nan["ship_rule_vs_refit"] is False
    report = _cal_report()
    del report["heads"]["predio"]["refit_oof"]
    m = rc.collect_run_metrics(str(_cal_dir(tmp_path / "b", report)))
    assert "ship_rule_vs_refit" not in m and m["calib_error_cv_predio"] == 0.004


def test_calibrated_ece_is_omitted_when_the_report_is_missing_or_malformed(tmp_path):
    m = rc.collect_run_metrics(str(_cal_dir(tmp_path / "a", write_report=False)))
    assert "calib_error_cv_predio" not in m and "ship_rule_vs_refit" not in m and m["calibrated_layer"] is True
    e = _cal_dir(tmp_path / "b")
    (e / "fit_report.json").write_text("{corrupt")
    assert "calib_error_cv_manzana" not in rc.collect_run_metrics(str(e))
    e2 = _cal_dir(tmp_path / "c", {"heads": {"predio": {"oof": "x"}, "manzana": []}})
    m2 = rc.collect_run_metrics(str(e2))
    assert "calib_error_cv_predio" not in m2 and "calib_error_cv_manzana" not in m2


def test_calibrated_nan_ece_is_kept_so_the_check_can_fail_it(tmp_path):
    m = rc.collect_run_metrics(str(_cal_dir(tmp_path / "e", _cal_report(ece_p=float("nan")))))
    assert math.isnan(m["calib_error_cv_predio"])


def test_calibrated_file_of_another_kind_or_unreadable_is_ignored(tmp_path):
    e = _cal_dir(tmp_path / "a", kind="hybrid")
    m = rc.collect_run_metrics(str(e))
    assert "calibrated_layer" not in m and "calib_error_cv_predio" not in m
    e2 = _cal_dir(tmp_path / "b")
    (e2 / "reliability_calibrated.json").write_text("[]")
    assert "calibrated_layer" not in rc.collect_run_metrics(str(e2))


def test_calibrated_candidate_passes_a6_against_a_baseline_refit_through_check_acceptance(tmp_path):
    cand = rc.collect_run_metrics(str(_cal_dir(tmp_path / "e")))
    base = {"calib_error_cv_predio": 0.0975, "calib_error_cv_manzana": 0.0338}
    thresholds = {"baseline_reference": {}, "constants": {}}
    res = {r["id"]: r for r in rc.check_acceptance({"candidate": cand, "baseline": base}, thresholds)}
    assert res["A6"]["passed"] is True


def test_compare_runs_marks_probe_agreement_and_acceptance_as_second_look_when_asked(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    argv = ["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))]
    assert script.main([*argv, "--frozen-second-look"]) == 0
    for name in ("probe_agreement.json", "acceptance.json"):
        payload = json.load(open(new / name))
        assert payload["frozen_second_look"] is True and "NOT BLIND" in payload["frozen_second_look_note"], name


def test_compare_runs_without_the_option_writes_no_second_look_marker(tmp_path):
    script = _load_script("compare_runs")
    old, new = _two_runs(tmp_path)
    script.main(["--old", str(old), "--new", str(new), "--thresholds", str(_thresholds_file(tmp_path))])
    for name in ("probe_agreement.json", "acceptance.json"):
        assert "frozen_second_look" not in json.load(open(new / name))
