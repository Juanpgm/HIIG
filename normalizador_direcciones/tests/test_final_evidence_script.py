"""scripts/final_evidence.py on tiny fixtures: seed comparison + selection, and the evidence bundle."""

from __future__ import annotations

import importlib.util
import json
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script():
    spec = importlib.util.spec_from_file_location("final_evidence_script", os.path.join(ROOT, "scripts", "final_evidence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table(n_ok, n_wrong_lot=0, n=10, n_sin=0, fuente="cert", band="alta"):
    rows = []
    for i in range(n):
        ok = i < n_ok
        rows.append({
            "raw_address": f"a{i}", "estado": "OK" if ok else "SIN_MATCH", "confianza": 0.9,
            "numero_predial_nacional": "P" if ok else None,
            "correct_predial_lot": (i >= n_wrong_lot) if ok else None, "correct_manzana": True if ok else None,
            "raw_top1_correct_manzana": True, "correct_predial": (i >= n_wrong_lot) if ok else None, "fuente": fuente if ok else None,
            "banda_confiabilidad": band if ok else None, "p_predio_cal": 0.95 if ok else None,
            "p_manzana_cal": 0.99 if ok else None})
    return pd.DataFrame(rows)


def run_dir(path, gold, alln, tau=(0.94, 0.7), ece=(0.004, 0.002)):
    path.mkdir(parents=True)
    gold.to_parquet(path / "strict_eval_dev_gold_rows.parquet")
    alln.to_parquet(path / "strict_eval_dev_all_rows.parquet")
    json.dump({"tau_p": tau[0], "tau_m": tau[1]}, open(path / "decision.json", "w"))
    report = {"heads": {h: {"oof": {"ece10": e, "brier": 0.02, "auc": 0.9}, "refit_oof": {"brier": 0.02, "auc": 0.9}}
                        for h, e in zip(("predio", "manzana"), ece)}}
    json.dump(report, open(path / "fit_report.json", "w"))
    return path


@pytest.fixture()
def world(tmp_path):
    base = run_dir(tmp_path / "base", table(8), table(6))
    a = run_dir(tmp_path / "a", table(9, 1), table(7), ece=(0.004, 0.002))
    b = run_dir(tmp_path / "b", table(9, 1), table(7, n_sin=1), tau=(0.945, 0.7), ece=(0.0043, 0.0017))
    return tmp_path, base, a, b


def test_seed_comparison_has_both_runs_the_absolute_differences_and_calibration_quality(world):
    tmp, base, a, b = world
    mod = _script()
    out = mod.seed_comparison({"a": str(a), "b": str(b)})
    assert out["runs"]["a"]["gold_ok"] == 9 and out["runs"]["a"]["gold_lot_n"] == 9
    assert out["runs"]["a"]["calibration"]["predio"]["ece10"] == 0.004
    assert out["runs"]["b"]["tau_p"] == 0.945
    assert out["abs_diff"]["gold_ok"] == 0 and out["abs_diff"]["tau_p"] == pytest.approx(0.005)
    assert out["gold_n"] == 10 and out["one_gold_row_pp"] == pytest.approx(10.0)


def test_selection_file_holds_aggregates_only_and_the_rule_outcome(world):
    tmp, base, a, b = world
    mod = _script()
    sel = mod.selection_record({"a": str(a), "b": str(b)}, str(base))
    assert sel["selected"] in ("a", "b") and sel["baseline"]["gold_ok_coverage"] == 0.8
    text = json.dumps(sel)
    assert "raw_address" not in text and "a0" not in text


def test_main_writes_both_files_and_never_overwrites(world, tmp_path):
    tmp, base, a, b = world
    mod = _script()
    seed, selection = tmp / "seed.json", tmp / "selection.json"
    argv = ["seed-compare", "--run", f"a={a}", "--run", f"b={b}", "--baseline", str(base),
            "--out-seed", str(seed), "--out-selection", str(selection)]
    assert mod.main(argv) == 0 and seed.is_file() and selection.is_file()
    assert mod.main(argv) == 2                                   # refuses to overwrite an existing file


def test_main_rejects_a_missing_run_directory(world, tmp_path):
    tmp, base, a, b = world
    mod = _script()
    argv = ["seed-compare", "--run", f"a={tmp / 'nope'}", "--baseline", str(base),
            "--out-seed", str(tmp / "s.json"), "--out-selection", str(tmp / "x.json")]
    assert mod.main(argv) == 2


def test_evidence_bundle_writes_the_expected_aggregate_files(world, tmp_path):
    tmp, base, a, b = world
    mod = _script()
    for d in (base, a):
        table(6, n=8).to_parquet(d / "strict_eval_final_gold_rows.parquet")
        table(7, n=12).to_parquet(d / "strict_eval_final_all_rows.parquet")
    argv = ["evidence", "--candidate", str(a), "--baseline", str(base), "--resamples", "50"]
    assert mod.main(argv) == 0
    for name in ("confidence_curve_compare.json", "reliability_diagrams.json", "bands.json", "by_fuente.json",
                 "bootstrap.json", "outcome_shares.json"):
        assert (a / name).is_file(), name
    boot = json.load(open(a / "bootstrap.json"))
    assert {"dev", "frozen", "seed", "resamples"} <= set(boot) and boot["resamples"] == 50 and set(boot["dev"]) == {"gold_ok", "gold_lot_precision", "all_manzana_precision",
                                                                   "lost_correct"}
    assert "raw_address" not in open(a / "bands.json").read()
    assert mod.main(argv) == 2                                   # outputs exist now: refuse to overwrite


def test_evidence_without_the_frozen_rows_still_writes_the_dev_part(world):
    tmp, base, a, b = world
    mod = _script()
    assert mod.main(["evidence", "--candidate", str(a), "--baseline", str(base), "--resamples", "20"]) == 0
    assert json.load(open(a / "bootstrap.json"))["frozen"] is None


# ---------------------------------------------------------------------------------------- rows infix (validate runs)
def _rename_to_validate(directory):
    for name in ("gold", "all"):
        (directory / f"strict_eval_dev_{name}_rows.parquet").rename(directory / f"strict_eval_dev_{name}_validate_rows.parquet")


def test_selection_can_read_the_validate_rows_when_the_plain_rows_do_not_exist(world):
    tmp, base, a, b = world
    mod = _script()
    _rename_to_validate(a)
    with pytest.raises(FileNotFoundError):
        mod.selection_record({"a": str(a)}, str(base))
    sel = mod.selection_record({"a": str(a)}, str(base), rows_infix="_validate")
    assert sel["selected"] == "a" and sel["candidates"]["a"]["gold_ok"] == 9


def test_seed_comparison_accepts_the_rows_infix(world):
    tmp, base, a, b = world
    mod = _script()
    _rename_to_validate(a)
    out = mod.seed_comparison({"a": str(a)}, rows_infix="_validate")
    assert out["runs"]["a"]["gold_ok"] == 9


def test_main_seed_compare_passes_the_rows_infix_option(world, tmp_path):
    tmp, base, a, b = world
    mod = _script()
    _rename_to_validate(a)
    argv = ["seed-compare", "--run", f"a={a}", "--baseline", str(base), "--rows-infix", "_validate",
            "--out-seed", str(tmp / "s.json"), "--out-selection", str(tmp / "x.json")]
    assert mod.main(argv) == 0 and (tmp / "x.json").is_file()


# ---------------------------------------------------------------------------------------- second look + round 1 vs round 2
def _add_frozen(directory, gold, alln):
    gold.to_parquet(directory / "strict_eval_final_gold_rows.parquet")
    alln.to_parquet(directory / "strict_eval_final_all_rows.parquet")


def _with_frozen(world):
    tmp, base, a, b = world
    _add_frozen(base, table(6, n=8), table(7, n=12))
    _add_frozen(a, table(7, 1, n=8), table(8, n=12))
    _add_frozen(b, table(5, 2, n=8), table(6, n=12))
    return tmp, base, a, b


def test_every_derived_file_carries_the_frozen_second_look_flag_when_asked(world):
    tmp, base, a, b = _with_frozen(world)
    mod = _script()
    argv = ["evidence", "--candidate", str(a), "--baseline", str(base), "--resamples", "20", "--frozen-second-look",
            "--round1", str(b)]
    assert mod.main(argv) == 0
    names = ["confidence_curve_compare.json", "reliability_diagrams.json", "bands.json", "by_fuente.json", "bootstrap.json",
             "outcome_shares.json", "round_comparison.json"]
    for name in names:
        payload = json.load(open(a / name))
        assert payload["frozen_second_look"] is True, name
        assert "SECOND LOOK" in payload["frozen_second_look_note"] and "NOT BLIND" in payload["frozen_second_look_note"], name


def test_without_the_option_no_flag_is_written(world):
    tmp, base, a, b = _with_frozen(world)
    mod = _script()
    assert mod.main(["evidence", "--candidate", str(a), "--baseline", str(base), "--resamples", "20"]) == 0
    assert "frozen_second_look" not in json.load(open(a / "bands.json"))
    assert not (a / "round_comparison.json").exists()


def test_reliability_diagrams_include_the_frozen_gold_rows(world):
    tmp, base, a, b = _with_frozen(world)
    mod = _script()
    assert mod.main(["evidence", "--candidate", str(a), "--baseline", str(base), "--resamples", "20"]) == 0
    rel = json.load(open(a / "reliability_diagrams.json"))
    assert {"dev_gold", "dev_all", "frozen_gold"} <= set(rel)
    assert rel["frozen_gold"]["predio"]["n"] == 7 and rel["frozen_gold"]["manzana"]["n"] == 7


def test_confidence_curve_compare_adds_the_frozen_curve_when_both_have_it(world):
    tmp, base, a, b = _with_frozen(world)
    mod = _script()
    assert mod.main(["evidence", "--candidate", str(a), "--baseline", str(base), "--resamples", "20"]) == 0
    curve = json.load(open(a / "confidence_curve_compare.json"))
    assert [p["threshold"] for p in curve["frozen"]["candidate"]][0] == 0.5 and len(curve["frozen"]["baseline"]) == 10


def test_round_comparison_counts_transitions_between_the_two_frozen_runs(tmp_path):
    mod = _script()
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    for d, g_ok, wrong in ((r1, 5, 0), (r2, 7, 1)):
        d.mkdir()
        _add_frozen(d, table(g_ok, wrong, n=8), table(g_ok, wrong, n=12))
    json.dump({"frozen_test": {"gold": {"test_gold_lot_missing_from_index": 30, "test_gold_ok": 5}}}, open(r1 / "experiment.json", "w"))
    json.dump({"frozen_test": {"gold": {"test_gold_lot_missing_from_index": 15, "test_gold_ok": 7}}}, open(r2 / "experiment.json", "w"))
    out = mod.round_comparison(str(r1), str(r2))
    assert out["frozen_gold"]["round1"]["ok"] == 5 and out["frozen_gold"]["round2"]["ok"] == 7
    assert out["frozen_gold"]["delta"]["ok"] == 2
    assert out["frozen_gold"]["round1"]["lot_missing_from_index"] == 30 and out["frozen_gold"]["round2"]["lot_missing_from_index"] == 15
    assert out["frozen_gold"]["transitions"] == {"ok_gained": 2, "ok_lost": 0, "both_ok": 5, "neither_ok": 1}
    assert "raw_address" not in json.dumps(out)


def test_round_comparison_rejects_misaligned_rows(tmp_path):
    mod = _script()
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    for d in (r1, r2):
        d.mkdir()
    _add_frozen(r1, table(5, n=8), table(5, n=12))
    other = table(5, n=8)
    other["raw_address"] = [f"z{i}" for i in range(8)]
    _add_frozen(r2, other, table(5, n=12))
    with pytest.raises(ValueError):
        mod.round_comparison(str(r1), str(r2))


def test_round_comparison_without_frozen_rows_is_a_clear_error(tmp_path):
    mod = _script()
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    with pytest.raises(FileNotFoundError):
        mod.round_comparison(str(tmp_path / "r1"), str(tmp_path / "r2"))


# ---------------------------------------------------------------------------------------- jev-crosscheck sub-command
def _jev_parquet(path, n=8):
    pd.DataFrame({"status": ["ok"] * n, "label": ["yes"] * n, "jev_p": [0.1 * (i + 1) for i in range(n)],
                  "tag_banda_confiabilidad": ["alta"] * n, "tag_fuente": ["nomenclatura_cert"] * n,
                  "extra_p_predio_cal": [0.1 * (i + 1) for i in range(n)], "extra_noise": ["light"] * n,
                  "input_text": ["CL 900 # 1"] * n}).to_parquet(path)


def test_jev_crosscheck_command_writes_one_aggregate_block_per_run_without_text(tmp_path):
    mod = _script()
    a, b, out = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "x.json"
    _jev_parquet(a)
    _jev_parquet(b, n=5)
    argv = ["jev-crosscheck", "--rows", f"candidate={a}", "--rows", f"baseline={b}", "--out", str(out), "--resamples", "50"]
    assert mod.main(argv) == 0
    payload = json.load(open(out))
    assert set(payload["runs"]) == {"candidate", "baseline"} and payload["runs"]["candidate"]["counts"]["judged"] == 8
    assert "CL 900" not in out.read_text()
    assert mod.main(argv) == 2                                   # refuses to overwrite


def test_jev_crosscheck_command_rejects_a_missing_rows_file(tmp_path):
    mod = _script()
    assert mod.main(["jev-crosscheck", "--rows", f"x={tmp_path / 'nope.parquet'}", "--out", str(tmp_path / "o.json")]) == 2
