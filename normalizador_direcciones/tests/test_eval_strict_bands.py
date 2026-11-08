"""Calibrated-run support of ``scripts/eval_strict.py``: pass-through columns, ``..._by_band.csv``, default delta."""

from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

nan = float("nan")
NEW = ["confiabilidad_calibrada", "banda_confiabilidad", "motivos_confiabilidad", "p_predio_cal", "p_manzana_cal"]


def _eval_strict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_eval_strict_bands", os.path.join(root, "scripts", "eval_strict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result(n=3, with_fuente=True, with_new=True) -> pd.DataFrame:
    frame = pd.DataFrame({"estado": ["OK", "OK", "SIN_MATCH"][:n]})
    if with_fuente:
        frame["fuente"] = ["nomenclatura_cert", "nomenclatura_pcer", None][:n]
    if with_new:
        frame["confiabilidad_calibrada"] = [0.98, 0.95, nan][:n]
        frame["banda_confiabilidad"] = ["alta", "baja", None][:n]
        frame["motivos_confiabilidad"] = ["+a", "-b", None][:n]
        frame["p_predio_cal"] = [0.98, 0.5, nan][:n]
        frame["p_manzana_cal"] = [0.99, 0.95, nan][:n]
    return frame


def _rows(n=3) -> pd.DataFrame:
    return pd.DataFrame({"estado": ["OK", "OK", "SIN_MATCH"][:n], "has_gt": [True] * n})


def test_calibrated_columns_are_appended_last_after_fuente():
    ev = _eval_strict()
    rows = ev.with_fuente_column(_rows(), _result())
    out = ev.with_calibrated_columns(rows, _result())
    assert list(out.columns) == ["estado", "has_gt", "fuente"] + NEW
    assert out["banda_confiabilidad"].tolist()[:2] == ["alta", "baja"] and pd.isna(out["banda_confiabilidad"].iloc[2])
    assert out["p_predio_cal"].isna().tolist() == [False, False, True]


def test_rows_are_unchanged_without_calibrated_columns():
    ev = _eval_strict()
    rows = _rows()
    out = ev.with_calibrated_columns(rows, _result(with_new=False))
    pd.testing.assert_frame_equal(out, rows)


def test_length_mismatch_is_an_error():
    ev = _eval_strict()
    with pytest.raises(ValueError, match="rows"):
        ev.with_calibrated_columns(_rows(2), _result())


def test_with_calibrated_columns_does_not_mutate_its_input():
    ev = _eval_strict()
    rows = _rows()
    ev.with_calibrated_columns(rows, _result())
    assert list(rows.columns) == ["estado", "has_gt"]


def _banded_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "estado": ["OK", "OK", "OK"], "banda_confiabilidad": ["alta", "alta", "baja"],
        "fuente": ["nomenclatura_cert", "nomenclatura_pcer", "nomenclatura_cert"], "has_gt": [True, True, True],
        "gt_quality": ["gold", "gold", "noisy"], "correct_predial_lot": [True, False, nan],
        "correct_manzana": [True, True, False],
    })


def test_band_csv_is_written_for_a_calibrated_run(tmp_path):
    ev = _eval_strict()
    path = ev.write_band_breakdown(_banded_rows(), str(tmp_path), "t")
    assert path is not None and os.path.basename(path) == "strict_eval_t_by_band.csv"
    table = pd.read_csv(path)
    assert {"scope", "banda", "fuente", "n_rows", "lot_precision", "lot_ci_low", "lot_ci_high"} <= set(table.columns)
    row = table[(table.scope == "all") & (table.banda == "alta") & (table.fuente == "(all)")].iloc[0]
    assert row["n_rows"] == 2 and row["lot_precision"] == 0.5


def test_band_csv_is_not_written_without_bands(tmp_path):
    ev = _eval_strict()
    assert ev.write_band_breakdown(pd.DataFrame({"estado": ["OK"]}), str(tmp_path), "t") is None
    assert ev.write_band_breakdown(pd.DataFrame(columns=["estado", "banda_confiabilidad"]), str(tmp_path), "t") is None
    assert not list(tmp_path.iterdir())


def test_band_csv_for_all_null_bands_is_not_written(tmp_path):
    ev = _eval_strict()
    rows = _banded_rows()
    rows["banda_confiabilidad"] = None
    assert ev.write_band_breakdown(rows, str(tmp_path), "t") is None


def test_ambiguity_delta_flag_is_only_forwarded_when_given():
    ev = _eval_strict()
    assert ev.ambiguity_kwarg(argparse.Namespace(ambiguity_delta=None)) == {}
    assert ev.ambiguity_kwarg(argparse.Namespace(ambiguity_delta=0.0)) == {"ambiguity_delta": 0.0}
    assert ev.ambiguity_kwarg(argparse.Namespace(ambiguity_delta=0.02)) == {"ambiguity_delta": 0.02}


def test_default_cli_flag_leaves_the_delta_to_the_service(monkeypatch):
    ev = _eval_strict()
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, argv=None, namespace=None):
        args = real(self, argv, namespace)
        captured["args"] = args
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    with pytest.raises(SystemExit):
        ev.main([])
    assert captured["args"].ambiguity_delta is None


def test_calibrated_signals_json_has_the_reached_quantiles_and_counters(tmp_path):
    import json
    ev = _eval_strict()
    rows = pd.DataFrame({
        "estado": ["OK", "OK", "SIN_MATCH"], "motivo": ["", "", "confiabilidad baja: p_manzana 0.40 < 0.70"],
        "confianza": [0.9, 0.9, nan], "margen": [0.2, nan, nan], "banda_confiabilidad": ["alta", "baja", None],
        "p_predio_cal": [0.99, 0.5, nan], "fuente": ["nomenclatura_cert", "nomenclatura_pcer", None],
    })
    stats = {"calibrated": {"reached": 3, "abstained": 1, "predio": 1, "manzana": 1, "alta": 1, "media": 0, "baja": 1,
                            "p_predio": [0.99, 0.5, 0.2]}}
    path = ev.write_calibrated_signals(rows, stats, str(tmp_path), "t")
    assert os.path.basename(path) == "strict_eval_t_calibrated.json"
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["p_predio_scope"] == "reached" and data["decision_counts"]["abstained"] == 1
    assert data["p_predio_quantiles"]["p50"] == 0.5 and "p_predio" not in data["decision_counts"]
    assert data["abstention_rate"] == pytest.approx(1 / 3)


def test_calibrated_signals_are_not_written_without_the_stats_block(tmp_path):
    ev = _eval_strict()
    assert ev.write_calibrated_signals(pd.DataFrame({"estado": ["OK"]}), {}, str(tmp_path), "t") is None
    assert not list(tmp_path.iterdir())
