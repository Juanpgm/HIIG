"""Lot-level predial scoring (``NPN[:21]``) added next to the exact-code metrics.

The ground truth ``gt_predial`` is cadastral and lot-level, while nomenclature NPNs may carry a unit suffix, so
the evaluation reports the predial at LOT level in addition to the exact code. Nothing here changes an existing
column or its meaning. All data is invented.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address import evaluate as ev  # noqa: E402
from cali_address.evaluate import STRICT_SUMMARY_COLUMNS, strict_summary  # noqa: E402

LOT_A = "760010100000000010001"          # 21 digits, invented
LOT_B = "760010100000000010002"
UNIT_0 = "000000000"
UNIT_7 = "000000007"


# ---------------------------------------------------------------------------
# predial_lot / lots_match
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", [21, 28, 29, 30, 31])
def test_lot_of_a_valid_code_is_its_first_21_characters(length):
    code = (LOT_A + "0123456789012345")[:length]
    assert ev.predial_lot(code) == LOT_A


@pytest.mark.parametrize("code", ["", "   ", None, float("nan"), pd.NA, "nan", "NaN", "None"])
def test_lot_of_a_missing_value_is_none(code):
    assert ev.predial_lot(code) is None


@pytest.mark.parametrize("code", ["7600101000000000100", LOT_A[:-1], "123", "0"])
def test_lot_of_a_short_code_is_none(code):
    assert len(code) < 21
    assert ev.predial_lot(code) is None


@pytest.mark.parametrize("code", [LOT_A[:-1] + "X", "76001010000000001000A" + UNIT_0, "7600101000000000100 1" + UNIT_0,
                                  LOT_A + "-00000001", 1.5, True])
def test_lot_of_a_non_digit_code_is_none(code):
    assert ev.predial_lot(code) is None


def test_lot_strips_surrounding_whitespace():
    assert ev.predial_lot(f"  {LOT_A}{UNIT_0}\n") == LOT_A


def test_same_lot_with_a_different_unit_matches_at_lot_level_only():
    a, b = LOT_A + UNIT_0, LOT_A + UNIT_7
    assert ev.lots_match(a, b) is True
    assert a != b                                   # the exact code differs: that is the whole point


def test_different_lot_does_not_match_even_if_units_are_equal():
    assert ev.lots_match(LOT_A + UNIT_0, LOT_B + UNIT_0) is False


@pytest.mark.parametrize("a,b", [
    (None, LOT_A + UNIT_0), (LOT_A + UNIT_0, None), (float("nan"), float("nan")), ("", ""),
    (LOT_A[:-1], LOT_A[:-1]),                       # equal but too short to be a lot
    (LOT_A + UNIT_0, LOT_A[:-1]),                   # one side short
    ("X" * 30, "X" * 30),                           # equal but not digits
])
def test_lots_match_is_false_when_either_code_is_not_a_valid_lot(a, b):
    assert ev.lots_match(a, b) is False


def test_lots_match_accepts_28_29_30_character_codes_mixed():
    assert ev.lots_match((LOT_A + UNIT_0)[:28], (LOT_A + UNIT_0)[:30]) is True
    assert ev.lots_match((LOT_A + UNIT_0)[:29], LOT_A) is True


# ---------------------------------------------------------------------------
# index_lots
# ---------------------------------------------------------------------------
def test_index_lots_collects_valid_lots_and_skips_the_rest():
    npns = pd.Series([LOT_A + UNIT_0, LOT_A + UNIT_7, LOT_B + UNIT_0, None, np.nan, "123", "", "X" * 30])
    assert ev.index_lots(npns) == {LOT_A, LOT_B}


def test_index_lots_of_an_empty_series_is_empty():
    assert ev.index_lots(pd.Series([], dtype=object)) == set()


# ---------------------------------------------------------------------------
# add_lot_columns (the rows table written by eval_strict.py)
# ---------------------------------------------------------------------------
def _rows(preds, gts, has_gt=None):
    n = len(preds)
    return pd.DataFrame({
        "numero_predial_nacional": preds, "gt_predial": gts,
        "has_gt": [True] * n if has_gt is None else has_gt,
        "correct_predial": [False] * n,             # an existing column that must stay as it was
    })


def test_add_lot_columns_appends_after_the_existing_columns_and_changes_nothing_else():
    rows = _rows([LOT_A + UNIT_0], [LOT_A + UNIT_7])
    out = ev.add_lot_columns(rows, {LOT_A})
    assert list(out.columns) == list(rows.columns) + ["correct_predial_lot", "lot_present_in_index"]
    pd.testing.assert_frame_equal(out[list(rows.columns)], rows)
    assert list(rows.columns) == ["numero_predial_nacional", "gt_predial", "has_gt", "correct_predial"]   # input untouched


def test_correct_predial_lot_true_when_the_lot_is_equal_and_the_unit_differs():
    out = ev.add_lot_columns(_rows([LOT_A + UNIT_0], [LOT_A + UNIT_7]), {LOT_A})
    assert out["correct_predial_lot"].tolist() == [True]


def test_correct_predial_lot_false_when_the_lot_differs():
    out = ev.add_lot_columns(_rows([LOT_B + UNIT_0], [LOT_A + UNIT_0]), {LOT_A})
    assert out["correct_predial_lot"].tolist() == [False]


def test_correct_predial_lot_is_nan_without_ground_truth():
    out = ev.add_lot_columns(_rows([LOT_A + UNIT_0], [LOT_A + UNIT_0], has_gt=[False]), {LOT_A})
    assert pd.isna(out["correct_predial_lot"].iloc[0]) and pd.isna(out["lot_present_in_index"].iloc[0])


@pytest.mark.parametrize("pred", [None, np.nan, "", "nan"])
def test_correct_predial_lot_is_nan_when_the_output_has_no_predial_like_the_exact_column(pred):
    # Shared addresses answer OK without a predial: excluded from the denominator, as for correct_predial.
    out = ev.add_lot_columns(_rows([pred], [LOT_A + UNIT_0]), {LOT_A})
    assert pd.isna(out["correct_predial_lot"].iloc[0])


@pytest.mark.parametrize("gt", [None, np.nan, ""])
def test_gt_without_a_code_gives_nan_in_both_columns(gt):
    out = ev.add_lot_columns(_rows([LOT_A + UNIT_0], [gt]), {LOT_A})
    assert pd.isna(out["correct_predial_lot"].iloc[0]) and pd.isna(out["lot_present_in_index"].iloc[0])


def test_short_codes_are_false_not_nan_when_both_sides_have_text():
    out = ev.add_lot_columns(_rows(["123"], ["123"]), {LOT_A})
    assert out["correct_predial_lot"].tolist() == [False]
    assert pd.isna(out["lot_present_in_index"].iloc[0])      # no valid lot to look up


def test_lot_present_in_index_is_true_false_by_membership_of_the_gt_lot():
    out = ev.add_lot_columns(_rows([None, None], [LOT_A + UNIT_7, LOT_B + UNIT_0]), {LOT_A})
    assert out["lot_present_in_index"].tolist() == [True, False]


def test_lot_present_in_index_is_nan_when_no_index_lots_are_given():
    out = ev.add_lot_columns(_rows([None], [LOT_A + UNIT_0]), None)
    assert pd.isna(out["lot_present_in_index"].iloc[0])


def test_add_lot_columns_on_an_empty_frame_returns_the_two_columns():
    out = ev.add_lot_columns(_rows([], []), {LOT_A})
    assert len(out) == 0 and list(out.columns)[-2:] == ["correct_predial_lot", "lot_present_in_index"]


def test_add_lot_columns_without_the_predial_columns_raises_a_clear_error():
    with pytest.raises(KeyError, match="numero_predial_nacional"):
        ev.add_lot_columns(pd.DataFrame({"has_gt": [True]}), {LOT_A})


# ---------------------------------------------------------------------------
# strict_summary: new columns LAST, nothing existing changes
# ---------------------------------------------------------------------------
OLD_COLUMNS = ["dataset", "estado", "n", "n_with_gt", "manzana_precision", "predial_precision", "idesc_text_agreement",
               "lost_correct", "lost_correct_share", "n_ok_sin_predial", "n_ok_sin_manzana"]

# Output of the pre-change strict_summary on the fixture below, captured before the lot columns existed.
OLD_SUMMARY_CSV = """dataset,estado,n,n_with_gt,manzana_precision,predial_precision,idesc_text_agreement,lost_correct,lost_correct_share,n_ok_sin_predial,n_ok_sin_manzana
dsA,OK,4,3,0.6666666666666666,0.5,0.5,0,0.0,0,0
dsA,REVISAR,1,1,,,,1,1.0,0,0
dsB,OK,1,1,1.0,1.0,1.0,0,0.0,0,0
dsB,RECHAZADO,1,1,,,,0,0.0,0,0
ALL,OK,5,4,0.75,0.6666666666666666,0.6666666666666666,0,0.0,0,0
ALL,RECHAZADO,1,1,,,,0,0.0,0,0
ALL,REVISAR,1,1,,,,1,1.0,0,0
"""


def _srow(ds, estado, has_gt, cm, cp, raw, txt, **extra):
    return {"dataset": ds, "estado": estado, "has_gt": has_gt, "correct_manzana": cm, "correct_predial": cp,
            "raw_top1_correct_manzana": raw, "text_agree_idesc": txt,
            "numero_predial_nacional": "X" if estado == "OK" else None, "manzana": "M" if estado == "OK" else None,
            **extra}


def _legacy_fixture():
    nan = np.nan
    return pd.DataFrame([
        _srow("dsA", "OK", True, True, True, nan, True), _srow("dsA", "OK", True, True, False, nan, False),
        _srow("dsA", "OK", True, False, nan, nan, nan), _srow("dsA", "OK", False, nan, nan, nan, nan),
        _srow("dsA", "REVISAR", True, nan, nan, True, nan), _srow("dsB", "OK", True, True, True, nan, True),
        _srow("dsB", "RECHAZADO", True, nan, nan, False, nan),
    ])


def test_summary_columns_keep_the_old_ones_first_and_append_the_lot_ones_last():
    assert STRICT_SUMMARY_COLUMNS[:len(OLD_COLUMNS)] == OLD_COLUMNS
    assert STRICT_SUMMARY_COLUMNS[len(OLD_COLUMNS):] == ["predial_lot_precision", "lot_present_in_index_share",
                                                         "n_lot_missing_from_index"]


def test_old_summary_columns_are_byte_identical_on_a_frame_without_the_lot_columns():
    summary = strict_summary(_legacy_fixture())
    assert summary[OLD_COLUMNS].to_csv(index=False, lineterminator="\n") == OLD_SUMMARY_CSV
    assert summary["predial_lot_precision"].isna().all()           # no correct_predial_lot -> unknown, not 0
    assert summary["lot_present_in_index_share"].isna().all()


def test_old_summary_columns_are_unchanged_when_the_lot_columns_are_present():
    frame = _legacy_fixture()
    frame["correct_predial_lot"] = [True, True, np.nan, np.nan, np.nan, True, np.nan]
    frame["lot_present_in_index"] = [True, True, False, np.nan, True, True, False]
    summary = strict_summary(frame)
    assert summary[OLD_COLUMNS].to_csv(index=False, lineterminator="\n") == OLD_SUMMARY_CSV


def test_predial_lot_precision_counts_lot_hits_among_ok_rows_with_gt_and_a_predial():
    frame = pd.DataFrame([
        _srow("dsA", "OK", True, True, False, np.nan, np.nan, correct_predial_lot=True),    # exact miss, lot hit
        _srow("dsA", "OK", True, True, False, np.nan, np.nan, correct_predial_lot=False),
        _srow("dsA", "OK", True, True, True, np.nan, np.nan, correct_predial_lot=True),
        _srow("dsA", "OK", True, True, np.nan, np.nan, np.nan, correct_predial_lot=np.nan),  # no predial: excluded
        _srow("dsA", "OK", False, np.nan, np.nan, np.nan, np.nan, correct_predial_lot=True),  # no GT: excluded
    ])
    ok = strict_summary(frame).query("dataset == 'ALL' and estado == 'OK'").iloc[0]
    assert ok["predial_precision"] == pytest.approx(1 / 3)
    assert ok["predial_lot_precision"] == pytest.approx(2 / 3)


def test_predial_lot_precision_is_nan_for_non_ok_rows_and_for_groups_without_gt():
    frame = pd.DataFrame([
        _srow("dsA", "REVISAR", True, np.nan, np.nan, False, np.nan, correct_predial_lot=True),
        _srow("dsB", "OK", False, np.nan, np.nan, np.nan, np.nan, correct_predial_lot=np.nan),
    ])
    summary = strict_summary(frame)
    assert summary["predial_lot_precision"].isna().all()


def test_lot_presence_share_and_missing_count_cover_every_estado_with_gt():
    frame = pd.DataFrame([
        _srow("dsA", "OK", True, True, True, np.nan, np.nan, lot_present_in_index=True),
        _srow("dsA", "OK", True, True, True, np.nan, np.nan, lot_present_in_index=False),
        _srow("dsA", "OK", True, True, True, np.nan, np.nan, lot_present_in_index=np.nan),   # no valid lot: not counted
        _srow("dsA", "OK", False, np.nan, np.nan, np.nan, np.nan, lot_present_in_index=False),   # no GT: not counted
        _srow("dsA", "REVISAR", True, np.nan, np.nan, False, np.nan, lot_present_in_index=False),
    ])
    summary = strict_summary(frame).set_index(["dataset", "estado"])
    ok = summary.loc[("dsA", "OK")]
    assert ok["lot_present_in_index_share"] == pytest.approx(1 / 2) and ok["n_lot_missing_from_index"] == 1
    rev = summary.loc[("dsA", "REVISAR")]
    assert rev["lot_present_in_index_share"] == 0.0 and rev["n_lot_missing_from_index"] == 1


def test_summary_of_an_empty_frame_has_all_the_columns():
    assert list(strict_summary(pd.DataFrame()).columns) == STRICT_SUMMARY_COLUMNS


def test_summary_csv_round_trip_keeps_the_new_columns_last():
    frame = _legacy_fixture()
    frame["correct_predial_lot"] = np.nan
    text = strict_summary(frame).to_csv(index=False)
    header = pd.read_csv(io.StringIO(text), nrows=0).columns.tolist()
    assert header == STRICT_SUMMARY_COLUMNS


# ---------------------------------------------------------------------------
# scripts/eval_strict.py wiring
# ---------------------------------------------------------------------------
def _eval_strict():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_eval_strict_lot", os.path.join(root, "scripts", "eval_strict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeNormalizer:
    def __init__(self, npns):
        self._predial = np.array(npns, dtype=object)


def test_eval_strict_adds_the_lot_columns_from_the_normalizer_index():
    es = _eval_strict()
    rows = _rows([LOT_A + UNIT_0, LOT_B + UNIT_0], [LOT_A + UNIT_7, LOT_A + UNIT_7])
    out = es.with_lot_columns(rows, _FakeNormalizer([LOT_A + UNIT_0, None, "123"]))
    assert out["correct_predial_lot"].tolist() == [True, False]
    assert out["lot_present_in_index"].tolist() == [True, True]
    assert out["correct_predial"].tolist() == rows["correct_predial"].tolist()      # exact column untouched


def test_eval_strict_reports_a_gt_lot_missing_from_the_index():
    es = _eval_strict()
    out = es.with_lot_columns(_rows([None], [LOT_B + UNIT_0]), _FakeNormalizer([LOT_A + UNIT_0]))
    assert out["lot_present_in_index"].tolist() == [False]


def test_eval_strict_keeps_the_exact_comparison_as_it_was():
    es = _eval_strict()
    assert es._codes_equal(LOT_A + UNIT_0, LOT_A + UNIT_7) is False       # unit differs: exact stays False
    assert es._codes_equal(LOT_A + UNIT_0, LOT_A + UNIT_0) is True
    assert np.isnan(es._codes_equal(None, LOT_A))
