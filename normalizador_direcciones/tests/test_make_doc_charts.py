"""Data-loading and extraction functions of scripts/make_doc_charts.py (synthetic JSON only, no real artifacts)."""

import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "make_doc_charts.py")

spec = importlib.util.spec_from_file_location("make_doc_charts", SCRIPT)
mdc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdc)


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


# ----------------------------------------------------------------------------------------------- load_json
def test_load_json_ok(tmp_path):
    assert mdc.load_json(write(tmp_path, "a.json", '{"x": 1}')) == {"x": 1}


def test_load_json_missing_file(tmp_path):
    with pytest.raises(mdc.ChartDataError, match="missing input file"):
        mdc.load_json(str(tmp_path / "nope.json"))


def test_load_json_directory_is_not_a_file(tmp_path):
    with pytest.raises(mdc.ChartDataError):
        mdc.load_json(str(tmp_path))


@pytest.mark.parametrize("content", ["", "   \n\t "])
def test_load_json_empty_file(tmp_path, content):
    with pytest.raises(mdc.ChartDataError, match="empty file"):
        mdc.load_json(write(tmp_path, "e.json", content))


@pytest.mark.parametrize("content", ["{}", "[]", "null"])
def test_load_json_empty_content(tmp_path, content):
    with pytest.raises(mdc.ChartDataError, match="empty JSON"):
        mdc.load_json(write(tmp_path, "e.json", content))


@pytest.mark.parametrize("content", ["{not json", '{"a": 1,}', "[1, 2", "NaN junk"])
def test_load_json_malformed(tmp_path, content):
    with pytest.raises(mdc.ChartDataError, match="malformed JSON"):
        mdc.load_json(write(tmp_path, "m.json", content))


def test_load_json_invalid_utf8(tmp_path):
    path = tmp_path / "b.json"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(mdc.ChartDataError):
        mdc.load_json(str(path))


# ----------------------------------------------------------------------------------------------- extractors
def diagrams(bins=None, n=10):
    bins = [{"bin": 0, "n": 5, "predicted": 0.9, "observed": 0.8}] if bins is None else bins
    return {"dev_gold": {"predio": {"n": n, "mean_predicted": 0.9, "observed": 0.8, "bins": bins}}}


def test_reliability_bins_ok():
    out = mdc.extract_reliability_bins(diagrams(), "dev_gold")
    assert out["bins"] == [(5, 0.9, 0.8)] and out["n"] == 10


def test_reliability_bins_missing_scope_head_and_empty_bins():
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_reliability_bins(diagrams(), "frozen_gold")
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_reliability_bins(diagrams(), "dev_gold", head="manzana")
    with pytest.raises(mdc.ChartDataError, match="empty"):
        mdc.extract_reliability_bins(diagrams(bins=[]), "dev_gold")


@pytest.mark.parametrize("bad", [None, "0.9", float("nan"), float("inf"), True])
def test_reliability_bins_rejects_non_finite_numbers(bad):
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_reliability_bins(
            diagrams(bins=[{"bin": 0, "n": 5, "predicted": bad, "observed": 0.8}]), "dev_gold")


def curve(precision=0.99, share=0.9):
    return [{"predial_lot_precision_emitted": precision, "share_of_rows": share}] * 2


def test_confidence_curves_ok_and_none_precision():
    data = {"thresholds": [0.5, 0.6], "candidate": curve(), "baseline": curve(None)}
    out = mdc.extract_confidence_curves(data)
    assert out["candidate"]["precision"] == [0.99, 0.99] and out["baseline"]["precision"] == [None, None]


def test_confidence_curves_length_mismatch_and_missing_model():
    with pytest.raises(mdc.ChartDataError, match="does not match"):
        mdc.extract_confidence_curves({"thresholds": [0.5], "candidate": curve(), "baseline": curve()})
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_confidence_curves({"thresholds": [0.5, 0.6], "candidate": curve()})


def test_bands_unscored_band_gives_none():
    rows = [{"band": "alta", "n": 10, "lot_scored": 10, "lot_precision": 0.9, "lot_lo": 0.6, "lot_hi": 0.98},
            {"band": "baja", "n": 3, "lot_scored": 0, "lot_precision": None, "lot_lo": None, "lot_hi": None}]
    out = mdc.extract_bands({"dev_gold": rows}, "dev_gold")
    assert out[0]["lot_precision"] == 0.9 and out[1]["lot_precision"] is None


def test_bands_missing_or_empty_scope():
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_bands({}, "dev_gold")
    with pytest.raises(mdc.ChartDataError, match="no bands"):
        mdc.extract_bands({"dev_gold": []}, "dev_gold")


def test_bootstrap_ok_and_bad_blocks():
    block = {"m": {"diff": 1.0, "lo": -1.0, "hi": 2.0, "p_diff_lt0": 0.2}}
    out = mdc.extract_bootstrap({"dev": block, "frozen": block})
    assert len(out) == 2 and out[0]["split"] == "dev"
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_bootstrap({"dev": block})
    with pytest.raises(mdc.ChartDataError, match="empty"):
        mdc.extract_bootstrap({"dev": block, "frozen": {}})
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_bootstrap({"dev": {"m": {"diff": 1.0}}, "frozen": block})


def test_bench_ok_and_empty():
    r = {"latency_ms": {"p50": 10.0, "p95": 12.0},
         "throughput": {"100": {"rows_per_s_median": 5.0}, "10000": {"rows_per_s_median": 7.0}}}
    bench = {"threads16": {"prod": {"onnx": r}}, "threads4": {"prod": {"onnx": r}}}
    out = mdc.extract_bench(bench)
    assert out["latency"][(16, "prod", "onnx")] == (10.0, 12.0)
    assert out["throughput"][("prod", "onnx")] == 7.0  # the largest batch wins
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_bench({"threads16": {"prod": {"onnx": {}}}, "threads4": {}})
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_bench({})


def test_acceptance_ok_unknown_status_and_empty():
    out = mdc.extract_acceptance({"results": [{"id": "A1", "status": "pass", "description": "d"}]})
    assert out == [{"id": "A1", "status": "pass", "description": "d"}]
    with pytest.raises(mdc.ChartDataError, match="unknown status"):
        mdc.extract_acceptance({"results": [{"id": "A1", "status": "maybe"}]})
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_acceptance({"results": []})
    with pytest.raises(mdc.ChartDataError):
        mdc.extract_acceptance({})


# ----------------------------------------------------------------------------------------------- CLI
def test_main_reports_missing_inputs_with_exit_code_1(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    code = mdc.main(["--experiments-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    assert code == 1 and "missing input file" in capsys.readouterr().err
