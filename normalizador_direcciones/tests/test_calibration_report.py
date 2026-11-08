"""Reporting helpers of the calibrated decision layer: Wilson intervals, per-band precision, drift monitoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cali_address import calibration_report as R

nan = float("nan")


# --------------------------------------------------------------------------- Wilson
def test_wilson_matches_the_textbook_value():
    low, high = R.wilson_interval(90, 100)
    assert low == pytest.approx(0.8256, abs=2e-4) and high == pytest.approx(0.9448, abs=2e-4)


def test_wilson_extremes_stay_inside_the_unit_interval():
    low, high = R.wilson_interval(10, 10)
    assert high == 1.0 and 0.0 < low < 1.0
    low, high = R.wilson_interval(0, 10)
    assert low == 0.0 and 0.0 < high < 1.0


def test_wilson_with_no_observations_is_nan():
    low, high = R.wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


@pytest.mark.parametrize("k,n", [(-1, 5), (6, 5), (1, -2)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        R.wilson_interval(k, n)


def test_wilson_interval_narrows_with_more_data():
    a = R.wilson_interval(9, 10)
    b = R.wilson_interval(900, 1000)
    assert (b[1] - b[0]) < (a[1] - a[0])


# --------------------------------------------------------------------------- band breakdown
def rows() -> pd.DataFrame:
    return pd.DataFrame({
        "estado": ["OK", "OK", "OK", "OK", "OK", "SIN_MATCH", "OK"],
        "banda_confiabilidad": ["alta", "alta", "media", "baja", "alta", None, "alta"],
        "fuente": ["nomenclatura_cert", "nomenclatura_pcer", "nomenclatura_cert", "nomenclatura_pcer",
                   "nomenclatura_cert", None, "nomenclatura_cert"],
        "has_gt": [True, True, True, True, True, True, False],
        "gt_quality": ["gold", "gold", "noisy", "gold", "noisy", "gold", "no_gt"],
        "correct_predial_lot": [True, False, True, nan, False, nan, nan],
        "correct_manzana": [True, True, True, False, True, nan, nan],
    })


def test_breakdown_columns_and_scopes():
    out = R.band_breakdown(rows())
    assert list(out.columns) == R.BAND_BREAKDOWN_COLUMNS
    assert set(out["scope"]) == {"all", "gold"}
    assert set(out["banda"]) == {"alta", "media", "baja", "(all)"}
    assert set(out["fuente"]) == {"(all)", "nomenclatura_cert", "nomenclatura_pcer"}


def test_breakdown_counts_and_precision_all_scope():
    out = R.band_breakdown(rows()).set_index(["scope", "banda", "fuente"])
    alta = out.loc[("all", "alta", "(all)")]
    assert alta["n_rows"] == 4                      # the no-GT OK row counts as a row but is not scored
    assert alta["n_lot_scored"] == 3 and alta["lot_precision"] == pytest.approx(1 / 3)
    assert alta["n_manzana_scored"] == 3 and alta["manzana_precision"] == 1.0
    assert alta["lot_ci_low"] <= alta["lot_precision"] <= alta["lot_ci_high"]
    media = out.loc[("all", "media", "(all)")]
    assert media["n_lot_scored"] == 1 and media["lot_precision"] == 1.0 and media["lot_ci_high"] == 1.0


def test_breakdown_baja_band_has_no_lot_observations():
    out = R.band_breakdown(rows()).set_index(["scope", "banda", "fuente"])
    baja = out.loc[("all", "baja", "(all)")]
    assert baja["n_lot_scored"] == 0 and math.isnan(baja["lot_precision"]) and math.isnan(baja["lot_ci_low"])
    assert baja["manzana_precision"] == 0.0


def test_breakdown_gold_scope_and_fuente_split():
    out = R.band_breakdown(rows()).set_index(["scope", "banda", "fuente"])
    assert out.loc[("gold", "alta", "(all)")]["n_rows"] == 2
    assert out.loc[("all", "alta", "nomenclatura_pcer")]["lot_precision"] == 0.0
    assert out.loc[("all", "(all)", "nomenclatura_cert")]["n_rows"] == 4


def test_breakdown_all_band_row_equals_the_ok_rows_with_a_band():
    out = R.band_breakdown(rows()).set_index(["scope", "banda", "fuente"])
    assert out.loc[("all", "(all)", "(all)")]["n_rows"] == 6      # the SIN_MATCH row has no band


@pytest.mark.parametrize("frame", [
    pd.DataFrame(columns=["estado", "banda_confiabilidad"]),
    pd.DataFrame({"estado": ["OK"]}),                                    # not a calibrated run
    pd.DataFrame(),
])
def test_breakdown_of_degenerate_frames_is_empty_with_the_columns(frame):
    out = R.band_breakdown(frame)
    assert list(out.columns) == R.BAND_BREAKDOWN_COLUMNS and len(out) == 0


def test_breakdown_without_fuente_or_gold_columns_still_works():
    slim = rows().drop(columns=["fuente", "gt_quality"])
    out = R.band_breakdown(slim)
    assert set(out["scope"]) == {"all"} and set(out["fuente"]) == {"(all)"}


def test_breakdown_is_deterministic():
    pd.testing.assert_frame_equal(R.band_breakdown(rows()), R.band_breakdown(rows().copy()))


# --------------------------------------------------------------------------- monitoring
def monitored() -> pd.DataFrame:
    return pd.DataFrame({
        "estado": ["OK"] * 8 + ["SIN_MATCH", "SIN_MATCH", "SIN_MATCH"],
        "motivo": [""] * 8 + ["confiabilidad baja: p_manzana 0.40 < 0.70", "confianza 0.50 < umbral 0.70", "x"],
        "confianza": [0.9, 0.95, 0.85, 0.99, 0.8, 0.9, 0.92, 0.88, 0.9, 0.5, None],
        "margen": [0.3, 0.2, nan, 0.1, 0.25, 0.3, 0.2, 0.15, nan, nan, nan],
        "banda_confiabilidad": ["alta"] * 4 + ["media"] * 2 + ["baja"] * 2 + [None] * 3,
        "p_predio_cal": [0.99, 0.98, 0.97, 0.99, 0.95, 0.94, 0.5, 0.4, nan, nan, nan],
        "fuente": ["nomenclatura_pcer"] * 2 + ["nomenclatura_cert"] * 6 + [None] * 3,
    })


def test_signals_values():
    s = R.monitoring_signals(monitored())
    assert s["ok_rows"] == 8
    assert s["band_share"] == {"alta": 0.5, "media": 0.25, "baja": 0.25}
    assert s["share_pcer_ok"] == pytest.approx(0.25)
    assert s["abstention_rate"] == pytest.approx(1 / 9)           # 1 layer abstention among 8 OK + 1 reaching the layer
    # no competitor (null margen) reads as the clip 0.3, exactly like the margen_manzana feature
    assert s["mean_margen_ok"] == pytest.approx(np.mean([0.3, 0.2, 0.3, 0.1, 0.25, 0.3, 0.2, 0.15]))
    assert s["mean_confianza_ok"] == pytest.approx(np.mean([0.9, 0.95, 0.85, 0.99, 0.8, 0.9, 0.92, 0.88]))
    assert s["p_predio_scope"] == "ok_rows"
    assert s["p_predio_quantiles"]["p50"] == pytest.approx(np.quantile([0.99, 0.98, 0.97, 0.99, 0.95, 0.94, 0.5, 0.4], 0.5))


def test_signals_use_the_reached_probabilities_when_given():
    s = R.monitoring_signals(monitored(), p_predio_reached=[0.1, 0.2, 0.3, 0.4])
    assert s["p_predio_scope"] == "reached" and s["p_predio_quantiles"]["p50"] == pytest.approx(0.25)


def test_signals_of_an_empty_or_uncalibrated_frame_are_none_not_errors():
    for frame in (pd.DataFrame(columns=["estado", "motivo"]), pd.DataFrame({"estado": ["SIN_MATCH"], "motivo": ["x"]})):
        s = R.monitoring_signals(frame)
        assert s["ok_rows"] == 0 and s["band_share"] is None and s["abstention_rate"] is None
        assert s["mean_margen_ok"] is None and s["p_predio_quantiles"] is None and s["share_pcer_ok"] is None


def test_signals_without_a_fuente_column_have_no_pcer_share():
    assert R.monitoring_signals(monitored().drop(columns=["fuente"]))["share_pcer_ok"] is None


def test_margin_mean_clips_to_the_feature_range():
    frame = monitored()
    frame.loc[0, "margen"] = 0.9                                    # above the clip
    frame.loc[1, "margen"] = -0.2                                   # below zero
    s = R.monitoring_signals(frame)
    assert s["mean_margen_ok"] == pytest.approx(np.mean([0.3, 0.0, 0.3, 0.1, 0.25, 0.3, 0.2, 0.15]))


def test_signals_are_json_serialisable():
    import json
    json.dumps(R.monitoring_signals(monitored()))


def baseline():
    return {"p_predio_quantiles": {"p10": 0.65, "p25": 0.95, "p50": 0.99}, "band_share": {"alta": 0.74, "media": 0.13, "baja": 0.13},
            "share_pcer_ok": 0.345, "abstention_rate": 0.097, "mean_margen_ok": 0.258, "mean_confianza_ok": 0.916}


def test_no_alarm_on_the_baseline_itself():
    alarms = R.compare_monitoring(baseline(), baseline())
    assert not any(a["alarm"] for a in alarms.values())


@pytest.mark.parametrize("edit,signal", [
    (lambda c: c["p_predio_quantiles"].update(p25=0.89), "p_predio_p25"),
    (lambda c: c["band_share"].update(alta=0.63), "alta_share"),
    (lambda c: c.update(share_pcer_ok=0.46), "share_pcer_ok"),
    (lambda c: c.update(share_pcer_ok=0.23), "share_pcer_ok"),
    (lambda c: c.update(abstention_rate=0.15), "abstention_rate"),
    (lambda c: c.update(mean_margen_ok=0.22), "mean_margen_ok"),
    (lambda c: c.update(mean_confianza_ok=0.89), "mean_confianza_ok"),
])
def test_each_drift_alarm_fires(edit, signal):
    current = baseline()
    edit(current)
    assert R.compare_monitoring(baseline(), current)[signal]["alarm"] is True


def test_alarm_boundaries_are_strict():
    current = baseline()
    current["p_predio_quantiles"]["p25"] = 0.90               # a drop of exactly 0.05 does not alarm
    assert R.compare_monitoring(baseline(), current)["p_predio_p25"]["alarm"] is False


def test_missing_signals_never_alarm():
    out = R.compare_monitoring(baseline(), {"band_share": None, "p_predio_quantiles": None})
    assert not any(a["alarm"] for a in out.values())
