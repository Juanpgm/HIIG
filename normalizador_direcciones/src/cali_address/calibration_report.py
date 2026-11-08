"""Reporting helpers of the calibrated decision layer: Wilson intervals, precision per reliability band, drift signals.

Aggregates only: nothing here reads or stores address text. ``band_breakdown`` works on the rows table of
``scripts/eval_strict.py`` (OK rows carrying ``banda_confiabilidad``); ``monitoring_signals`` works on the output frame
of ``service.normalize_strict`` in calibrated mode.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "ALL_LABEL", "ALARMS", "BAND_BREAKDOWN_COLUMNS", "band_breakdown", "compare_monitoring", "monitoring_signals",
    "wilson_interval",
]

#: Label of the aggregate row of a dimension (all bands / all sources).
ALL_LABEL = "(all)"
BAND_BREAKDOWN_COLUMNS = [
    "scope", "banda", "fuente", "n_rows", "n_lot_scored", "lot_precision", "lot_ci_low", "lot_ci_high",
    "n_manzana_scored", "manzana_precision", "manzana_ci_low", "manzana_ci_high",
]
_BANDS = ("alta", "media", "baja")
#: Clip of the margins (``service.MARGIN_CLIP``); a row without competitor counts as the clip.
MARGIN_CLIP = 0.3


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of ``k`` successes in ``n`` trials (95 % by default); ``(nan, nan)`` when ``n == 0``.

    Raises ``ValueError`` for impossible counts (``k < 0``, ``n < 0`` or ``k > n``).
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"impossible counts k={k} n={n}")
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _scored(values: pd.Series, has_gt: np.ndarray) -> tuple[int, int]:
    """``(successes, scored)`` of a bool/NaN series restricted to the rows that have ground truth."""
    present = values[has_gt].dropna()
    return int(sum(bool(v) for v in present)), int(len(present))


def _cell(group: pd.DataFrame, scope: str, banda: str, fuente: str) -> dict:
    has_gt = group["has_gt"].fillna(False).astype(bool).to_numpy() if "has_gt" in group else np.ones(len(group), bool)
    out = {"scope": scope, "banda": banda, "fuente": fuente, "n_rows": int(len(group))}
    for prefix, column in (("lot", "correct_predial_lot"), ("manzana", "correct_manzana")):
        if column in group.columns:
            k, n = _scored(group[column], has_gt)
        else:
            k, n = 0, 0
        low, high = wilson_interval(k, n)
        out[f"n_{prefix}_scored"] = n
        out[f"{prefix}_precision"] = k / n if n else float("nan")
        out[f"{prefix}_ci_low"], out[f"{prefix}_ci_high"] = low, high
    return out


def band_breakdown(rows: pd.DataFrame) -> pd.DataFrame:
    """Precision of the OK rows per reliability band and source, with Wilson 95 % intervals.

    One line per (scope, banda, fuente) where ``scope`` is ``all`` and ``gold`` (``gt_quality == "gold"``; omitted without
    that column), ``banda`` is ``alta`` / ``media`` / ``baja`` plus the ``(all)`` aggregate, and ``fuente`` is each
    source plus ``(all)`` (only ``(all)`` when the rows carry no ``fuente``). ``n_rows`` counts the OK rows of the cell;
    precisions use the rows with ground truth whose lot / manzana code was scored (``correct_predial_lot`` /
    ``correct_manzana`` not null), as ``strict_summary`` does. An empty frame, or one without ``banda_confiabilidad``
    (a run that is not calibrated), or with no banded OK row, gives an empty table with the same columns. Never raises on such input.
    """
    if rows is None or len(rows) == 0 or "banda_confiabilidad" not in rows.columns or "estado" not in rows.columns:
        return pd.DataFrame(columns=BAND_BREAKDOWN_COLUMNS)
    banded = rows[(rows["estado"] == "OK") & rows["banda_confiabilidad"].isin(_BANDS)]
    if banded.empty:
        return pd.DataFrame(columns=BAND_BREAKDOWN_COLUMNS)
    scopes = [("all", banded)]
    if "gt_quality" in banded.columns:
        scopes.append(("gold", banded[banded["gt_quality"] == "gold"]))
    out = []
    for scope, frame in scopes:
        fuentes = [ALL_LABEL]
        if "fuente" in frame.columns:
            fuentes += sorted({str(v) for v in frame["fuente"].dropna() if str(v).strip()})
        for fuente in fuentes:
            by_fuente = frame if fuente == ALL_LABEL else frame[frame["fuente"].astype(str) == fuente]
            for banda in (*_BANDS, ALL_LABEL):
                cell = by_fuente if banda == ALL_LABEL else by_fuente[by_fuente["banda_confiabilidad"] == banda]
                if len(cell) == 0 and banda != ALL_LABEL and fuente != ALL_LABEL:
                    continue
                out.append(_cell(cell, scope, banda, fuente))
    return pd.DataFrame(out, columns=BAND_BREAKDOWN_COLUMNS)


# ---------------------------------------------------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------------------------------------------------
def _clean_floats(values) -> np.ndarray:
    array = np.asarray(list(values), dtype="float64")
    return array[np.isfinite(array)]


def monitoring_signals(frame: pd.DataFrame, p_predio_reached=None) -> dict:
    """Drift signals of one run in calibrated mode (every value ``None`` when it cannot be computed).

    * ``p_predio_quantiles`` (p10 / p25 / p50 / p75 / p90): of ``p_predio_reached`` (every row that reached the decision,
      from ``stats["calibrated"]["p_predio"]``) when given, else of the OK rows' ``p_predio_cal`` (``p_predio_scope``);
    * ``band_share`` among OK rows; ``share_pcer_ok``: OK rows resolved to a ``nomenclatura_pcer`` document;
    * ``abstention_rate``: layer abstentions (``SIN_MATCH`` with a ``confiabilidad baja`` motivo) over OK rows plus those;
    * ``mean_margen_ok``: mean ``margen`` of the OK rows, null (no competitor) = 0.3, clipped to [0, 0.3];
      ``mean_confianza_ok``: mean ``confianza`` of the OK rows (nulls ignored).
    """
    empty = {
        "rows": 0, "ok_rows": 0, "p_predio_scope": None, "p_predio_quantiles": None, "band_share": None,
        "share_pcer_ok": None, "abstention_rate": None, "mean_margen_ok": None, "mean_confianza_ok": None,
    }
    if frame is None or len(frame) == 0 or "estado" not in frame.columns:
        return empty
    ok = frame[frame["estado"] == "OK"]
    out = dict(empty, rows=int(len(frame)), ok_rows=int(len(ok)))
    if "motivo" in frame.columns:
        abstained = int(((frame["estado"] == "SIN_MATCH") & frame["motivo"].fillna("").astype(str).str.startswith(
            "confiabilidad baja")).sum())
        reached = len(ok) + abstained
        out["abstention_rate"] = abstained / reached if reached else None
    if p_predio_reached is not None:
        probs, out["p_predio_scope"] = _clean_floats(p_predio_reached), "reached"
    elif "p_predio_cal" in ok.columns:
        probs, out["p_predio_scope"] = _clean_floats(ok["p_predio_cal"]), "ok_rows"
    else:
        probs = np.array([])
    if len(probs):
        out["p_predio_quantiles"] = {f"p{int(q * 100)}": float(np.quantile(probs, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    else:
        out["p_predio_scope"] = None
    if len(ok) and "banda_confiabilidad" in ok.columns and ok["banda_confiabilidad"].notna().any():
        out["band_share"] = {b: float((ok["banda_confiabilidad"] == b).mean()) for b in _BANDS}
    if len(ok) and "fuente" in ok.columns:
        out["share_pcer_ok"] = float((ok["fuente"] == "nomenclatura_pcer").mean())
    if len(ok) and "margen" in ok.columns:
        # "no competitor" (null margen) is the clip, exactly as the margen_manzana feature reads it
        margins = pd.to_numeric(ok["margen"], errors="coerce").fillna(MARGIN_CLIP).clip(0.0, MARGIN_CLIP)
        out["mean_margen_ok"] = float(margins.mean())
    if "confianza" in ok.columns:
        values = _clean_floats(pd.to_numeric(ok["confianza"], errors="coerce"))
        out["mean_confianza_ok"] = float(values.mean()) if len(values) else None
    return out


#: Alarm rules of SPEC section 10: ``signal -> (description, extractor, comparator)``; each comparator receives
#: ``(baseline_value, current_value)`` and says whether the drift is an alarm. Boundaries are strict.
def _get(d, *path):
    for key in path:
        if not isinstance(d, dict) or d.get(key) is None:
            return None
        d = d[key]
    return d


ALARMS = {
    "p_predio_p25": (("p_predio_quantiles", "p25"), lambda b, c: b - c > 0.05, "P_predio p25 drops by more than 0.05"),
    "alta_share": (("band_share", "alta"), lambda b, c: b - c > 0.10, "alta share falls by more than 10 pp"),
    "share_pcer_ok": (("share_pcer_ok",), lambda b, c: abs(c - b) > 0.10, "share of pcer documents moves by more than 10 pp"),
    "abstention_rate": (("abstention_rate",), lambda b, c: c > 1.5 * b, "abstention rate grows by more than x1.5"),
    "mean_margen_ok": (("mean_margen_ok",), lambda b, c: b - c > 0.03, "mean margen_manzana of OK rows drops by more than 0.03"),
    "mean_confianza_ok": (("mean_confianza_ok",), lambda b, c: b - c > 0.02, "mean confianza of OK rows drops by more than 0.02"),
}


def compare_monitoring(baseline: dict, current: dict) -> dict:
    """``{signal: {baseline, current, alarm, rule}}`` for the signals of :data:`ALARMS`.

    A signal missing on either side never alarms (``alarm`` is ``False``); thresholds are strict (a drop of exactly the
    limit is not an alarm). Floating-point noise is absorbed with a tolerance of 1e-9.
    """
    out = {}
    for name, (path, rule, text) in ALARMS.items():
        b, c = _get(baseline, *path), _get(current, *path)
        alarm = False
        if b is not None and c is not None:
            alarm = bool(rule(float(b), float(c))) and not _on_boundary(name, float(b), float(c))
        out[name] = {"baseline": b, "current": c, "alarm": alarm, "rule": text}
    return out


_LIMITS = {"p_predio_p25": 0.05, "alta_share": 0.10, "share_pcer_ok": 0.10, "mean_margen_ok": 0.03, "mean_confianza_ok": 0.02}


def _on_boundary(name: str, b: float, c: float) -> bool:
    """True when the drift equals the limit up to floating-point noise (so it must not alarm)."""
    limit = _LIMITS.get(name)
    if limit is None:
        return name == "abstention_rate" and abs(c - 1.5 * b) <= 1e-9
    drift = abs(c - b) if name == "share_pcer_ok" else b - c
    return abs(drift - limit) <= 1e-9
