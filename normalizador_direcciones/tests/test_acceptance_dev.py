"""``scripts/acceptance_dev.py``: A1..A12 on dev-only evidence (no frozen_test) written to ``acceptance_dev.json``.

Tiny invented experiment directories (summary csv + rows parquet); nothing here reads real data.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from cali_address import retrain_compare as rc

nan = float("nan")


def _script():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_acceptance_dev", os.path.join(root, "scripts", "acceptance_dev.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SUMMARY_COLS = ["dataset", "estado", "n", "n_with_gt", "manzana_precision", "predial_precision", "idesc_text_agreement",
                "lost_correct", "lost_correct_share", "n_ok_sin_predial", "n_ok_sin_manzana", "predial_lot_precision",
                "lot_present_in_index_share", "n_lot_missing_from_index"]


def summary(n_ok, n_total, manz, lot, lost, missing=0) -> pd.DataFrame:
    rows = [
        ["ALL", "OK", n_ok, n_ok, manz, 0.5, 0.6, 0, 0.0, 0, 0, lot, 1.0, missing],
        ["ALL", "SIN_MATCH", n_total - n_ok, n_total - n_ok, nan, nan, nan, lost, 0.1, 0, 0, nan, 1.0, 0],
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def gold_rows(n_ok=8, n=10, conf=0.9) -> pd.DataFrame:
    rows = []
    for i in range(n):
        ok = i < n_ok
        rows.append({"estado": "OK" if ok else "SIN_MATCH", "confianza": conf if ok else nan,
                     "correct_predial": True if ok else nan, "correct_manzana": True if ok else nan,
                     "correct_predial_lot": True if ok else nan, "numero_predial_nacional": f"N{i}" if ok else None,
                     "has_gt": True})
    return pd.DataFrame(rows)


def make_run(root, name, *, gold_ok=8, gold_n=10, all_ok=60, all_n=100, manz=0.90, lot=0.95, lost=20, suffix="",
             validation=None, write_exp=True):
    d = root / name
    d.mkdir(exist_ok=True)
    summary(all_ok, all_n, manz, lot, lost).to_csv(d / f"strict_eval_dev_all{suffix}_summary.csv", index=False)
    summary(gold_ok, gold_n, 1.0, lot, 1).to_csv(d / f"strict_eval_dev_gold{suffix}_summary.csv", index=False)
    gold_rows(gold_ok, gold_n).to_parquet(d / f"strict_eval_dev_gold{suffix}_rows.parquet")
    if validation is not None:
        pd.DataFrame(validation, columns=["rule", "violations", "corrected", "mode", "rows"]).to_csv(
            d / f"strict_eval_dev_all{suffix}_validation.csv", index=False)
    if write_exp and not suffix:
        (d / "experiment.json").write_text(json.dumps({"tag": name}), encoding="utf-8")
    return d


def thresholds():
    return {"baseline_reference": {"dev_gold_ok": 8, "dev_gold_predial_lot_precision": 0.95, "dev_manzana_precision": 0.9,
                                   "dev_lost_correct": 20},
            "constants": dict(rc._DEFAULT_CONSTANTS), "noise_band": {}}


# --------------------------------------------------------------------------- metrics
def test_metrics_come_from_the_summaries_and_the_gold_rows(tmp_path):
    ad = _script()
    d = make_run(tmp_path, "run", gold_ok=8, gold_n=10, all_ok=60, all_n=100, manz=0.9, lot=0.95, lost=20)
    m = ad.run_metrics(str(d))
    assert m["dev_gold_ok"] == 8 and m["dev_ok"] == 60 and m["dev_n"] == 100
    assert m["dev_gold_predial_lot_precision"] == 0.95 and m["dev_manzana_precision"] == 0.9
    assert m["dev_lost_correct"] == 20 and m["dev_gold_lost_correct"] == 1
    assert any(abs(p["threshold"] - 0.7) < 1e-9 for p in m["confidence_curve"])


def test_missing_files_leave_the_metrics_out_instead_of_inventing_them(tmp_path):
    ad = _script()
    d = tmp_path / "empty"
    d.mkdir()
    assert ad.run_metrics(str(d)) == {}
    only_all = make_run(tmp_path, "partial")
    os.remove(only_all / "strict_eval_dev_gold_summary.csv")
    m = ad.run_metrics(str(only_all))
    assert "dev_ok" in m and "dev_gold_ok" not in m


def test_a_malformed_summary_is_skipped_not_fatal(tmp_path):
    ad = _script()
    d = make_run(tmp_path, "bad")
    (d / "strict_eval_dev_all_summary.csv").write_text("not,a,summary\n1,2", encoding="utf-8")
    m = ad.run_metrics(str(d))
    assert "dev_ok" not in m and "dev_manzana_precision" not in m


def test_validation_remaining_counts_what_was_not_corrected(tmp_path):
    ad = _script()
    clean = make_run(tmp_path, "c", suffix="_validate", validation=[])
    assert ad.run_metrics(str(clean), "_validate")["violations_remaining_dev"] == 0
    fixed = make_run(tmp_path, "f", suffix="_validate",
                     validation=[["npn_format", 3, 3, "correct", 100], ["estado_vocab", 2, 0, "correct", 100]])
    assert ad.run_metrics(str(fixed), "_validate")["violations_remaining_dev"] == 2
    absent = make_run(tmp_path, "a", suffix="_validate")
    assert "violations_remaining_dev" not in ad.run_metrics(str(absent), "_validate")


# --------------------------------------------------------------------------- production untouched
def test_production_untouched_compares_recorded_hashes(tmp_path):
    ad = _script()
    f = tmp_path / "model.pt"
    f.write_bytes(b"abc")
    import hashlib
    recorded = {"model.pt": hashlib.sha256(b"abc").hexdigest(), "gone.pt": "00"}
    assert ad.production_untouched(recorded, str(tmp_path), ignore_missing=True) is True
    f.write_bytes(b"abd")
    assert ad.production_untouched(recorded, str(tmp_path), ignore_missing=True) is False
    assert ad.production_untouched({}, str(tmp_path)) is None                      # nothing to compare: unknown


def test_missing_production_file_is_not_silently_untouched(tmp_path):
    ad = _script()
    assert ad.production_untouched({"model.pt": "00"}, str(tmp_path)) is False


# --------------------------------------------------------------------------- the checklist
def evaluate(tmp_path, cand_kw=None, base_kw=None, correct=True, **flags):
    ad = _script()
    base = make_run(tmp_path, "base", **(base_kw or {}))
    cand = make_run(tmp_path, "cand", **(cand_kw or {}))
    if correct:
        make_run(tmp_path, "cand", suffix="_validate", validation=[], **(cand_kw or {}))
    return ad.evaluate(str(cand), str(base), thresholds(), **flags)


def by_id(results):
    return {r["id"]: r for r in results}


def test_a_candidate_equal_to_the_baseline_passes_the_dev_criteria(tmp_path):
    results = by_id(evaluate(tmp_path)["results"])
    for cid in ("A1", "A2", "A4", "A5", "A10"):
        assert results[cid]["status"] == "pass", (cid, results[cid]["reason"])
    # A12 embeds A3 (frozen gold): the full criterion cannot be judged, but its dev parts are
    assert results["A12"]["status"] == "not_evaluated" and results["A12"]["dev_parts_status"] == "pass"


def test_frozen_dependent_and_external_criteria_are_not_evaluated_not_passed(tmp_path):
    results = by_id(evaluate(tmp_path)["results"])
    for cid in ("A3", "A6", "A7", "A11"):
        assert results[cid]["status"] == "not_evaluated", cid


def test_a9_follows_the_flags_and_is_not_evaluated_without_them(tmp_path):
    assert by_id(evaluate(tmp_path)["results"])["A9"]["status"] == "not_evaluated"
    done = by_id(evaluate(tmp_path, pytest_green=True, no_real_addresses_exit=0)["results"])
    assert done["A9"]["status"] == "pass"
    failed = by_id(evaluate(tmp_path, pytest_green=True, no_real_addresses_exit=1)["results"])
    assert failed["A9"]["status"] == "fail"


def test_a_worse_candidate_fails_the_right_criteria(tmp_path):
    out = evaluate(tmp_path, cand_kw=dict(gold_ok=6, lot=0.90, lost=130, manz=0.80))
    results = by_id(out["results"])
    assert results["A1"]["status"] == "fail" and results["A2"]["status"] == "fail"
    assert results["A4"]["status"] == "fail" and results["A5"]["status"] == "fail"
    assert "A1" in out["summary"]["failed"]


def test_a1_boundary_is_inclusive(tmp_path):
    ok = by_id(evaluate(tmp_path, cand_kw=dict(gold_ok=7), base_kw=dict(gold_ok=8))["results"])
    assert ok["A1"]["status"] == "pass"                                             # baseline - 1
    bad = by_id(evaluate(tmp_path, cand_kw=dict(gold_ok=6), base_kw=dict(gold_ok=8))["results"])
    assert bad["A1"]["status"] == "fail"


def test_a12_fails_when_violations_remain(tmp_path):
    ad = _script()
    base = make_run(tmp_path, "b2")
    cand = make_run(tmp_path, "c2")
    make_run(tmp_path, "c2", suffix="_validate", validation=[["estado_vocab", 1, 0, "correct", 10]])
    a12 = by_id(ad.evaluate(str(cand), str(base), thresholds())["results"])["A12"]
    assert a12["status"] == "fail" and a12["dev_parts_status"] == "fail"


def test_a12_is_not_evaluated_without_the_correct_run(tmp_path):
    ad = _script()
    base = make_run(tmp_path, "b3")
    cand = make_run(tmp_path, "c3")
    a12 = by_id(ad.evaluate(str(cand), str(base), thresholds())["results"])["A12"]
    assert a12["status"] == "not_evaluated" and a12["dev_parts_status"] == "not_evaluated"


def test_a8_uses_the_production_check_when_given(tmp_path):
    ad = _script()
    base = make_run(tmp_path, "b4")
    cand = make_run(tmp_path, "c4")
    out = by_id(ad.evaluate(str(cand), str(base), thresholds(), production_untouched=True)["results"])
    assert out["A8"]["status"] == "pass"
    out = by_id(ad.evaluate(str(cand), str(base), thresholds(), production_untouched=False)["results"])
    assert out["A8"]["status"] == "fail"
    out = by_id(ad.evaluate(str(cand), str(base), thresholds())["results"])
    assert out["A8"]["status"] == "not_evaluated"


def test_baseline_without_the_needed_metrics_is_a_clear_error(tmp_path):
    ad = _script()
    empty = tmp_path / "nothing"
    empty.mkdir()
    cand = make_run(tmp_path, "c5")
    with pytest.raises(ValueError, match="baseline"):
        ad.evaluate(str(cand), str(empty), thresholds())
    with pytest.raises(ValueError, match="candidate"):
        ad.evaluate(str(empty), str(cand), thresholds())


# --------------------------------------------------------------------------- output
def test_table_lists_all_twelve_criteria_with_value_threshold_and_status(tmp_path):
    ad = _script()
    text = ad.format_table(evaluate(tmp_path)["results"])
    for i in range(1, 13):
        assert f"A{i} " in text or f"A{i}\t" in text or f"| A{i} " in text
    assert "pass" in text and "not_evaluated" in text


def test_main_writes_acceptance_dev_json_and_never_overwrites_silently(tmp_path, capsys):
    ad = _script()
    make_run(tmp_path, "base")
    cand = make_run(tmp_path, "cand")
    make_run(tmp_path, "cand", suffix="_validate", validation=[])
    th = tmp_path / "th.json"
    th.write_text(json.dumps(thresholds()), encoding="utf-8")
    code = ad.main(["--candidate", str(cand), "--baseline", str(tmp_path / "base"), "--thresholds", str(th)])
    assert code == 0
    payload = json.loads((cand / "acceptance_dev.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "dev-only" and len(payload["results"]) == 12 and "summary" in payload
    assert "A1" in capsys.readouterr().out
    assert ad.main(["--candidate", str(cand), "--baseline", str(tmp_path / "base"), "--thresholds", str(th)]) == 2


def test_main_with_an_unreadable_thresholds_file_fails_cleanly(tmp_path):
    ad = _script()
    make_run(tmp_path, "base")
    cand = make_run(tmp_path, "cand")
    assert ad.main(["--candidate", str(cand), "--baseline", str(tmp_path / "base"),
                    "--thresholds", str(tmp_path / "missing.json")]) == 2
    assert not (cand / "acceptance_dev.json").exists()


def test_baseline_metrics_match_the_recorded_leaderboard_row_when_the_real_run_is_present():
    ad = _script()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(root, "artifacts", "experiments", "baseline_prod_2026_10")
    if not os.path.isfile(os.path.join(base, "strict_eval_dev_gold_summary.csv")):
        pytest.skip("baseline run not present")
    row = json.load(open(os.path.join(base, "experiment.json"), encoding="utf-8"))["leaderboard_row"]
    m = ad.run_metrics(base)
    for key in ("dev_gold_ok", "dev_ok", "dev_n", "dev_lost_correct"):
        assert m[key] == row[key], key
    for key in ("dev_manzana_precision", "dev_gold_predial_lot_precision"):
        assert m[key] == pytest.approx(row[key])
