"""Wiring of ``output_schema`` into ``service.normalize_strict``, ``tuning.json`` and ``scripts/eval_strict.py``.

Default (``validate_outputs`` unset / ``off``) must leave the historical output byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os

import pandas as pd
import pytest

from cali_address.inference import DECISION_KEYS, load_tuning, validate_decision
from cali_address.service import normalize_strict

from conftest import make_stub

NPN_OK = "760010100000000900001000000000"  # 30 digits, invented

#: sha256 of ``frame.to_json(orient="split", default_handler=str)`` of the fixture below, captured from the code
#: BEFORE the validate_outputs option existed. A change here means the default output moved.
GOLDEN_DEFAULT_SHA256 = "b521b4d446111cf1e7b2b34e308279e96d6c3b1773277daac9d9a71ac2bca0f0"

RAWS = ["CARRERA 905 # 950-10", "CL 906 # 951-20", "TEXTO SIN DIRECCION", "", "CR 99 # 1"]


def _stub():
    return make_stub(
        [{"direccion": "CARRERA 905 # 950-10", "predial": NPN_OK, "manzana": "760010100000000900001"},
         {"direccion": "CALLE 906 # 951-20", "predial": "P1", "manzana": "M2"}],
        scores=[[0.9, 0.1], [0.1, 0.9], [0.1, 0.1]], threshold=0.5,
    )


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_json(orient="split", default_handler=str).encode()).hexdigest()


# --------------------------------------------------------------------------- default is byte-identical
def test_default_output_is_byte_identical_to_the_pre_option_golden():
    assert _digest(normalize_strict(_stub(), RAWS)) == GOLDEN_DEFAULT_SHA256


def test_explicit_off_equals_the_default():
    assert _digest(normalize_strict(_stub(), RAWS, validate_outputs="off")) == GOLDEN_DEFAULT_SHA256


def test_off_in_the_decision_block_equals_the_default():
    stub = _stub()
    stub.decision = {"validate_outputs": "off"}
    assert _digest(normalize_strict(stub, RAWS)) == GOLDEN_DEFAULT_SHA256


def test_off_does_not_touch_the_stats_dict():
    stats: dict = {}
    normalize_strict(_stub(), RAWS, stats=stats)
    assert "output_validation" not in stats


def test_report_mode_leaves_the_output_identical_and_fills_stats():
    stats: dict = {}
    frame = normalize_strict(_stub(), RAWS, validate_outputs="report", stats=stats)
    assert _digest(frame) == GOLDEN_DEFAULT_SHA256
    summary = stats["output_validation"]
    assert summary["mode"] == "report" and summary["rows"] == len(RAWS)
    # only the second cadastral doc (invented predial "P1") violates the schema; nothing is corrected in report mode
    assert summary["violations"] == {"npn_format": 1, "ok_without_support": 1} and summary["n_corrections"] == 0


def test_report_mode_without_a_stats_dict_does_not_fail():
    assert len(normalize_strict(_stub(), RAWS, validate_outputs="report")) == len(RAWS)


# --------------------------------------------------------------------------- a fixture that really violates the schema
def _invalid_predial_stub():
    # The second cadastral doc carries a 2-char predial: an OK row built from it violates npn_format.
    return make_stub(
        [{"direccion": "CARRERA 905 # 950-10", "predial": "P1", "manzana": "760010100000000900001"}],
        scores=[[0.9]], threshold=0.5,
    )


def test_report_counts_the_violation_without_changing_anything():
    stats: dict = {}
    base = normalize_strict(_invalid_predial_stub(), ["CARRERA 905 # 950-10"])
    frame = normalize_strict(_invalid_predial_stub(), ["CARRERA 905 # 950-10"], validate_outputs="report", stats=stats)
    pd.testing.assert_frame_equal(frame, base)
    assert stats["output_validation"]["violations"].get("npn_format") == 1


def test_correct_mode_nulls_the_invalid_predial_and_lowers_the_claim():
    stats: dict = {}
    frame = normalize_strict(_invalid_predial_stub(), ["CARRERA 905 # 950-10"], validate_outputs="correct", stats=stats)
    row = frame.iloc[0]
    assert row["estado"] == "OK" and pd.isna(row["numero_predial_nacional"])
    assert row["nivel_precision"] != "predio" and row["position_tier"] != "predio"
    assert stats["output_validation"]["corrected"]["npn_format"] == 1
    # corrected output validates clean
    from cali_address.output_schema import validate_and_correct

    assert validate_and_correct(frame, "report")[1].violations == {}


def test_explicit_argument_wins_over_the_decision_block():
    stub = _invalid_predial_stub()
    stub.decision = {"validate_outputs": "correct"}
    corrected = normalize_strict(stub, ["CARRERA 905 # 950-10"])
    assert pd.isna(corrected.iloc[0]["numero_predial_nacional"])
    untouched = normalize_strict(stub, ["CARRERA 905 # 950-10"], validate_outputs="off")
    assert untouched.iloc[0]["numero_predial_nacional"] == "P1"


def test_unknown_mode_is_rejected_before_any_work():
    with pytest.raises(ValueError, match="validate_outputs"):
        normalize_strict(_stub(), RAWS, validate_outputs="fix")


def test_empty_input_with_validation_on_returns_an_empty_frame():
    for mode in ("report", "correct"):
        assert len(normalize_strict(_stub(), [], validate_outputs=mode)) == 0


def test_extra_refine_columns_do_not_break_validation():
    frame = normalize_strict(_stub(), RAWS, validate_outputs="correct", refine_input=True)
    assert "refine_changes" in frame.columns


# --------------------------------------------------------------------------- tuning.json decision block
def _write_tuning(tmp_path, decision):
    chosen = {"weights": {"sim": 0.7, "fuzz": 0.1, "struct": 0.1, "geo": 0.1}, "threshold": 0.7, "decision": decision}
    (tmp_path / "tuning.json").write_text(json.dumps({"chosen": chosen}), encoding="utf-8")
    return str(tmp_path)


def test_validate_outputs_is_a_decision_key():
    assert "validate_outputs" in DECISION_KEYS


@pytest.mark.parametrize("mode", ["off", "report", "correct"])
def test_decision_block_round_trips_validate_outputs(tmp_path, mode):
    assert load_tuning(_write_tuning(tmp_path, {"validate_outputs": mode}))["decision"] == {"validate_outputs": mode}


@pytest.mark.parametrize("bad", ["fix", "", None, 1, True, ["report"]])
def test_decision_block_rejects_a_bad_validate_outputs(bad):
    with pytest.raises(ValueError, match="validate_outputs"):
        validate_decision({"validate_outputs": bad})


def test_decision_block_still_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown decision keys"):
        validate_decision({"validate_output": "off"})


# --------------------------------------------------------------------------- scripts/eval_strict.py
def _eval_strict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_eval_strict_d12", os.path.join(root, "scripts", "eval_strict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_strict_decision_kwargs_are_unchanged_by_the_new_flag():
    ev = _eval_strict()
    ns = argparse.Namespace(soft_rules=None, max_soft=None, gate_fallback=None, validate_outputs="correct")
    assert ev.decision_kwargs(ns) == {"soft_rules": None, "max_soft": None, "gate_fallback": None}


def test_eval_strict_validate_outputs_kwarg_defaults_to_the_artifact():
    ev = _eval_strict()
    assert ev.validate_outputs_kwarg(argparse.Namespace(validate_outputs=None)) is None
    assert ev.validate_outputs_kwarg(argparse.Namespace(validate_outputs="report")) == "report"


def test_eval_strict_validate_outputs_kwarg_rejects_garbage():
    ev = _eval_strict()
    with pytest.raises(ValueError):
        ev.validate_outputs_kwarg(argparse.Namespace(validate_outputs="fix"))


def test_eval_strict_cli_rejects_an_unknown_mode():
    ev = _eval_strict()
    with pytest.raises(SystemExit) as info:
        ev.main(["--validate-outputs", "fix"])
    assert info.value.code == 2


def test_eval_strict_writes_the_validation_counts_per_rule(tmp_path):
    ev = _eval_strict()
    summary = {"mode": "correct", "rows": 10, "rows_with_violations": 3, "rows_corrected": 3,
               "violations": {"npn_format": 2, "confidence_range": 1}, "corrected": {"npn_format": 2, "confidence_range": 1},
               "n_corrections": 4}
    path = ev.write_validation_summary(summary, str(tmp_path), "t1")
    assert os.path.basename(path) == "strict_eval_t1_validation.csv"
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["rule", "violations", "corrected", "mode", "rows"]
    assert frame.set_index("rule")["violations"].to_dict() == {"confidence_range": 1, "npn_format": 2}
    assert set(frame["mode"]) == {"correct"} and set(frame["rows"]) == {10}


def test_eval_strict_validation_file_for_a_clean_run_still_has_a_header(tmp_path):
    ev = _eval_strict()
    path = ev.write_validation_summary({"mode": "report", "rows": 5, "violations": {}, "corrected": {}}, str(tmp_path), "t2")
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["rule", "violations", "corrected", "mode", "rows"] and len(frame) == 0


def test_eval_strict_writes_no_validation_file_when_the_mode_is_off(tmp_path):
    ev = _eval_strict()
    assert ev.write_validation_summary(None, str(tmp_path), "t3") is None
    assert os.listdir(tmp_path) == []
