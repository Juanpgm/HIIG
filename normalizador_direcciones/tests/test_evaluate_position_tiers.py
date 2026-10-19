"""Per-tier calibration table of the strict evaluation (synthetic rows)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_stub  # noqa: F401  (puts src/ on sys.path)
from cali_address.evaluate import POSITION_CALIBRATION_COLUMNS, position_tier_calibration


def rows(spec):
    """spec: (quality, estado, tier, radius, lat_present, dist_m)."""
    return pd.DataFrame([{
        "gt_quality": q, "estado": e, "position_tier": t, "position_radius_m": r,
        "lat": 3.4 if has_lat else np.nan, "dist_m": d, "has_gt": q != "no_gt",
    } for q, e, t, r, has_lat, d in spec])


SPEC = [
    ("gold", "OK", "predio", 65, True, 10.0), ("gold", "OK", "predio", 65, True, 30.0),
    ("gold", "OK", "predio", 65, True, 90.0), ("gold", "OK", "tramo", 110, True, 40.0),
    ("gold", "OK", "via", 260, False, np.nan), ("gold", "SIN_MATCH", "none", np.nan, False, np.nan),
    ("noisy", "OK", "predio", 65, True, 500.0), ("noisy", "OK", "manzana", 135, True, 100.0),
    ("no_gt", "OK", "predio", 65, True, np.nan),
]


def cell(table, quality, tier, col):
    return table[(table["quality"] == quality) & (table["tier"] == tier)][col].iloc[0]


def test_columns_and_segments():
    t = position_tier_calibration(rows(SPEC))
    assert list(t.columns) == POSITION_CALIBRATION_COLUMNS
    assert set(t["quality"]) == {"gold", "noisy", "ALL"}


def test_share_median_p90_coverage_and_radius_hit_rate():
    t = position_tier_calibration(rows(SPEC))
    assert cell(t, "gold", "predio", "n") == 3
    assert cell(t, "gold", "predio", "share") == pytest.approx(3 / 6)
    assert cell(t, "gold", "predio", "median_m") == 30.0
    assert cell(t, "gold", "predio", "p90_m") == pytest.approx(78.0)
    assert cell(t, "gold", "predio", "coverage") == 1.0
    assert cell(t, "gold", "predio", "within_radius") == pytest.approx(2 / 3)  # 90 m > 65 m
    assert cell(t, "noisy", "predio", "within_radius") == 0.0
    assert cell(t, "gold", "via", "coverage") == 0.0 and np.isnan(cell(t, "gold", "via", "median_m"))
    assert cell(t, "gold", "none", "n") == 1 and np.isnan(cell(t, "gold", "none", "median_m"))


def test_all_segment_excludes_rows_without_ground_truth():
    t = position_tier_calibration(rows(SPEC))
    assert cell(t, "ALL", "predio", "n") == 4  # 3 gold + 1 noisy, the no_gt row is out


def test_single_value_and_two_values():
    t = position_tier_calibration(rows([("gold", "OK", "predio", 65, True, 7.0)]))
    assert cell(t, "gold", "predio", "median_m") == 7.0 and cell(t, "gold", "predio", "p90_m") == 7.0


@pytest.mark.parametrize("frame", [None, pd.DataFrame(), pd.DataFrame({"estado": []}),
                                   pd.DataFrame({"estado": ["OK"], "x": [1]})])
def test_empty_or_incomplete_frames_return_an_empty_table(frame):
    t = position_tier_calibration(frame)
    assert list(t.columns) == POSITION_CALIBRATION_COLUMNS and len(t) == 0


def test_missing_gt_quality_column_reports_only_all():
    f = rows(SPEC).drop(columns=["gt_quality"])
    t = position_tier_calibration(f)
    assert set(t["quality"]) == {"ALL"}


def test_nan_and_infinite_distances_are_ignored():
    t = position_tier_calibration(rows([("gold", "OK", "predio", 65, True, np.inf),
                                        ("gold", "OK", "predio", 65, True, 5.0)]))
    assert cell(t, "gold", "predio", "median_m") == 5.0
    assert cell(t, "gold", "predio", "n_dist") == 1
