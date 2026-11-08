"""Per-source (``fuente``) outputs of ``scripts/eval_strict.py``: the rows column and ``..._by_fuente.csv``."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest


def _eval_strict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_eval_strict_fuente", os.path.join(root, "scripts", "eval_strict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows() -> pd.DataFrame:
    nan = float("nan")
    return pd.DataFrame({
        "dataset": ["d"] * 7,
        "estado": ["OK", "OK", "OK", "OK", "SIN_MATCH", "SIN_MATCH", "OK"],
        "fuente": ["nomenclatura_cert", "nomenclatura_cert", "nomenclatura_pcer", "nomenclatura_pcer", None, None, "nomenclatura_cert"],
        "has_gt": [True, True, True, True, True, True, False],
        "gt_quality": ["gold", "noisy", "gold", "gold", "gold", "noisy", "no_gt"],
        "correct_manzana": [True, True, True, False, nan, nan, nan],
        "correct_predial": [True, False, True, True, nan, nan, nan],
        "correct_predial_lot": [True, True, False, True, nan, nan, nan],
        "raw_top1_correct_manzana": [nan, nan, nan, nan, True, False, nan],
    })


def test_breakdown_has_one_row_per_fuente_and_scope():
    ev = _eval_strict()
    out = ev.fuente_breakdown(_rows())
    assert list(out.columns) == ev.FUENTE_BREAKDOWN_COLUMNS
    keys = set(zip(out["scope"], out["fuente"]))
    assert keys == {("all", "nomenclatura_cert"), ("all", "nomenclatura_pcer"), ("all", "(none)"),
                    ("gold", "nomenclatura_cert"), ("gold", "nomenclatura_pcer"), ("gold", "(none)")}


def test_breakdown_all_rows_numbers():
    ev = _eval_strict()
    out = ev.fuente_breakdown(_rows()).set_index(["scope", "fuente"])
    cert = out.loc[("all", "nomenclatura_cert")]
    assert cert["n_rows"] == 3 and cert["n_ok"] == 3
    assert cert["manzana_precision"] == 1.0          # 2 rows with GT, both right; the no-GT row is excluded
    assert cert["predial_precision"] == 0.5
    assert cert["predial_lot_precision"] == 1.0
    pcer = out.loc[("all", "nomenclatura_pcer")]
    assert pcer["manzana_precision"] == 0.5 and pcer["predial_precision"] == 1.0 and pcer["predial_lot_precision"] == 0.5


def test_lost_correct_lives_on_the_rows_without_a_selected_document():
    ev = _eval_strict()
    out = ev.fuente_breakdown(_rows()).set_index(["scope", "fuente"])
    assert out.loc[("all", "(none)")]["lost_correct"] == 1 and out.loc[("all", "(none)")]["n_ok"] == 0
    assert out.loc[("all", "nomenclatura_cert")]["lost_correct"] == 0
    assert np.isnan(out.loc[("all", "(none)")]["manzana_precision"])      # no OK rows -> no precision


def test_gold_scope_keeps_only_gold_rows():
    ev = _eval_strict()
    out = ev.fuente_breakdown(_rows()).set_index(["scope", "fuente"])
    assert out.loc[("gold", "nomenclatura_cert")]["n_rows"] == 1
    assert out.loc[("gold", "nomenclatura_cert")]["predial_precision"] == 1.0
    assert out.loc[("gold", "(none)")]["lost_correct"] == 1


def test_without_gt_quality_only_the_all_scope_is_written():
    ev = _eval_strict()
    out = ev.fuente_breakdown(_rows().drop(columns=["gt_quality"]))
    assert set(out["scope"]) == {"all"}


def test_without_the_lot_column_the_lot_precision_is_nan():
    ev = _eval_strict()
    out = ev.fuente_breakdown(_rows().drop(columns=["correct_predial_lot"]))
    assert out["predial_lot_precision"].isna().all()


def test_blank_and_nan_fuente_values_group_as_none():
    ev = _eval_strict()
    rows = _rows()
    rows["fuente"] = ["", " ", np.nan, "nomenclatura_pcer", None, None, "nomenclatura_cert"]
    out = ev.fuente_breakdown(rows).set_index(["scope", "fuente"])
    assert out.loc[("all", "(none)")]["n_rows"] == 5


def test_empty_frame_gives_an_empty_table_with_the_header():
    ev = _eval_strict()
    out = ev.fuente_breakdown(_rows().iloc[0:0])
    assert len(out) == 0 and list(out.columns) == ev.FUENTE_BREAKDOWN_COLUMNS


def test_all_rows_one_fuente_with_no_ground_truth_gives_nan_precision():
    ev = _eval_strict()
    rows = _rows()
    rows["has_gt"] = False
    out = ev.fuente_breakdown(rows).set_index(["scope", "fuente"])
    assert np.isnan(out.loc[("all", "nomenclatura_cert")]["predial_precision"])


# --------------------------------------------------------------------------- files / rows column
def test_writer_creates_the_csv_when_the_rows_have_fuente(tmp_path):
    ev = _eval_strict()
    path = ev.write_fuente_breakdown(_rows(), str(tmp_path), "t1")
    assert os.path.basename(path) == "strict_eval_t1_by_fuente.csv"
    assert list(pd.read_csv(path).columns) == ev.FUENTE_BREAKDOWN_COLUMNS


def test_writer_writes_nothing_without_the_fuente_column(tmp_path):
    ev = _eval_strict()
    assert ev.write_fuente_breakdown(_rows().drop(columns=["fuente"]), str(tmp_path), "t2") is None
    assert os.listdir(tmp_path) == []


def test_writer_writes_nothing_when_every_fuente_is_null(tmp_path):
    ev = _eval_strict()
    rows = _rows()
    rows["fuente"] = None
    assert ev.write_fuente_breakdown(rows, str(tmp_path), "t3") is None
    assert os.listdir(tmp_path) == []


def test_fuente_column_is_appended_last_when_the_output_has_it():
    ev = _eval_strict()
    rows = _rows().drop(columns=["fuente"])
    result = pd.DataFrame({"fuente": ["a", None, "b", "c", None, None, "a"]})
    out = ev.with_fuente_column(rows, result)
    assert list(out.columns) == list(rows.columns) + ["fuente"]
    assert out["fuente"].tolist()[:3] == ["a", None, "b"] or pd.isna(out["fuente"].iloc[1])


def test_fuente_column_is_not_added_when_the_output_lacks_it():
    ev = _eval_strict()
    rows = _rows().drop(columns=["fuente"])
    out = ev.with_fuente_column(rows, pd.DataFrame({"estado": ["OK"] * len(rows)}))
    assert list(out.columns) == list(rows.columns)


def test_fuente_column_rejects_a_length_mismatch():
    ev = _eval_strict()
    rows = _rows().drop(columns=["fuente"])
    with pytest.raises(ValueError):
        ev.with_fuente_column(rows, pd.DataFrame({"fuente": ["a"]}))
