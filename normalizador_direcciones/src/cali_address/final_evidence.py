"""Aggregate evidence for the final evaluation of a candidate against the baseline (counts and rates only).

Everything here works on the ``strict_eval_*_rows.parquet`` tables written by ``scripts/eval_strict.py`` and returns
plain numbers: no address text ever leaves a function. Used by ``scripts/final_evidence.py``.

* :func:`run_metrics`      headline counts of one rows table (same definitions as the leaderboard).
* :func:`select_candidate` the documented selection rule (``docs/experiments.md``) applied to a set of runs.
* :func:`wilson`, :func:`reliability_bins`, :func:`band_table`, :func:`fuente_table`, :func:`outcome_shares`.
* :func:`paired_bootstrap` candidate minus baseline on the SAME rows, resampling rows jointly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

OUTCOME_CLASSES = ("ok_predial", "ok_manzana_only", "sin_match", "no_parseable", "other")
BANDS = ("alta", "media", "baja")
_REQUIRED = ("estado", "numero_predial_nacional", "correct_predial_lot", "correct_manzana", "raw_top1_correct_manzana")


# ------------------------------------------------------------------------------------------------ helpers
def _flags(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """``(known, value)`` boolean arrays of an object column holding True / False / None / NaN."""
    values = series.tolist()
    known = np.array([isinstance(v, (bool, np.bool_)) for v in values], dtype=bool)
    value = np.array([bool(v) if k else False for v, k in zip(values, known)], dtype=bool)
    return known, value


def _require(rows: pd.DataFrame, columns=_REQUIRED) -> None:
    missing = [c for c in columns if c not in rows.columns]
    if missing:
        raise KeyError(f"rows table lacks column(s) {missing}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of ``k`` successes in ``n`` trials; ``(nan, nan)`` when ``n == 0``."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


# ------------------------------------------------------------------------------------------------ headline metrics
def _row_arrays(rows: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Per-row ``(numerator, denominator)`` indicator arrays of each metric (denominator ``None`` for counts)."""
    _require(rows)
    ok = (rows["estado"] == "OK").to_numpy()
    lot_known, lot_value = _flags(rows["correct_predial_lot"])
    man_known, man_value = _flags(rows["correct_manzana"])
    raw_known, raw_value = _flags(rows["raw_top1_correct_manzana"])
    emitting = ok & rows["numero_predial_nacional"].notna().to_numpy()
    lot_den = ok & lot_known
    man_den = ok & man_known
    return {
        "ok": (ok.astype(float), None),
        "rows_emitting_predial": (emitting.astype(float), None),
        "lot_precision": ((lot_den & lot_value).astype(float), lot_den.astype(float)),
        "manzana_precision": ((man_den & man_value).astype(float), man_den.astype(float)),
        "lost_correct": (((~ok) & raw_known & raw_value).astype(float), None),
    }


def run_metrics(rows: pd.DataFrame) -> dict:
    """Counts and rates of one rows table: OK rows, lot / manzana precision over the OK rows with a known label
    (the lot precision is over rows that EMIT a predial), rows emitting a predial and ``lost_correct`` (non-OK rows whose
    raw top-1 manzana was right)."""
    arrays = _row_arrays(rows)
    lot_num, lot_den = arrays["lot_precision"]
    man_num, man_den = arrays["manzana_precision"]
    out = {
        "n": int(len(rows)), "ok": int(arrays["ok"][0].sum()),
        "rows_emitting_predial": int(arrays["rows_emitting_predial"][0].sum()),
        "lot_emitted_scored": int(lot_den.sum()), "lot_correct": int(lot_num.sum()),
        "manzana_scored": int(man_den.sum()), "manzana_correct": int(man_num.sum()),
        "lost_correct": int(arrays["lost_correct"][0].sum()),
    }
    out["lot_precision"] = _ratio(out["lot_correct"], out["lot_emitted_scored"])
    out["manzana_precision"] = _ratio(out["manzana_correct"], out["manzana_scored"])
    out["ok_share"] = _ratio(out["ok"], out["n"])
    return out


# ------------------------------------------------------------------------------------------------ selection
def _finite(v) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(v)


def select_candidate(runs: dict[str, dict], baseline: dict, tie_rows: int = 1) -> dict:
    """The documented selection rule on DEV only.

    Eligible: ``gold_ok_coverage`` >= the measured baseline's. Winner: the highest ``gold_lot_precision``; runs within
    ``tie_rows`` gold rows of the best (``tie_rows / gold_lot_n`` of the best) are tied and the highest
    ``all_manzana_precision`` wins; a remaining tie keeps the input order. Runs with a NaN lot precision are not
    ranked. ``selected`` is ``None`` when no run is eligible.
    """
    if not runs:
        raise ValueError("no runs to select from")
    if not _finite(baseline.get("gold_ok_coverage")):
        raise ValueError("the baseline needs a finite gold_ok_coverage")
    floor = baseline["gold_ok_coverage"]
    eligible = [t for t, r in runs.items()
                if _finite(r.get("gold_ok_coverage")) and r["gold_ok_coverage"] >= floor - 1e-12]
    ranked = [t for t in eligible if _finite(runs[t].get("gold_lot_precision"))]
    if not ranked:
        return {"selected": None, "eligible": eligible, "tie": [], "baseline_gold_ok_coverage": floor,
                "reason": "no run keeps the baseline gold OK coverage with a finite lot precision"}
    best = max(ranked, key=lambda t: runs[t]["gold_lot_precision"])
    n_best = runs[best].get("gold_lot_n") or 0
    margin = (tie_rows / n_best) if n_best else 0.0
    tie = [t for t in ranked if runs[best]["gold_lot_precision"] - runs[t]["gold_lot_precision"] <= margin + 1e-12]

    def manzana(t):
        v = runs[t].get("all_manzana_precision")
        return v if _finite(v) else -1.0

    selected = max(tie, key=manzana) if len(tie) > 1 else best
    if len(tie) > 1:
        reason = (f"lot precision within {tie_rows} gold row(s) of the best among {tie}; tie broken by manzana precision "
                  f"on all dev rows")
    else:
        reason = "highest gold lot precision among the runs that keep the baseline gold OK coverage"
    return {"selected": selected, "eligible": eligible, "tie": tie, "baseline_gold_ok_coverage": floor, "reason": reason}


# ------------------------------------------------------------------------------------------------ diagrams and tables
def reliability_bins(p, y, bins: int = 10) -> list[dict]:
    """Equal-count reliability bins: ``{bin, n, predicted, observed}`` (rows with a NaN ``p`` or ``y`` are dropped)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.shape != y.shape:
        raise ValueError("p and y need the same length")
    keep = np.isfinite(p) & np.isfinite(y)
    p, y = p[keep], y[keep]
    if len(p) == 0:
        return []
    order = np.argsort(p, kind="stable")
    out = []
    for i, idx in enumerate(c for c in np.array_split(order, min(bins, len(p))) if len(c)):
        out.append({"bin": i, "n": int(len(idx)), "predicted": float(p[idx].mean()), "observed": float(y[idx].mean())})
    return out


def _share_row(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"correct": int(k), "scored": int(n), "precision": _ratio(k, n), "lo": lo, "hi": hi}


def band_table(rows: pd.DataFrame, column: str = "banda_confiabilidad") -> list[dict]:
    """Per reliability band (alta / media / baja): OK rows, lot precision (rows emitting a predial) and manzana precision,
    both with Wilson 95 % intervals. Rows without a band are ignored."""
    _require(rows, _REQUIRED + (column,))
    arrays = _row_arrays(rows)
    out = []
    for band in BANDS:
        sel = (rows[column] == band).to_numpy()
        lot = _share_row(int((arrays["lot_precision"][0] * sel).sum()), int((arrays["lot_precision"][1] * sel).sum()))
        man = _share_row(int((arrays["manzana_precision"][0] * sel).sum()),
                         int((arrays["manzana_precision"][1] * sel).sum()))
        out.append({"band": band, "n": int(sel.sum()),
                    "lot_correct": lot["correct"], "lot_scored": lot["scored"], "lot_precision": lot["precision"],
                    "lot_lo": lot["lo"], "lot_hi": lot["hi"],
                    "manzana_correct": man["correct"], "manzana_scored": man["scored"],
                    "manzana_precision": man["precision"], "manzana_lo": man["lo"], "manzana_hi": man["hi"]})
    return out


def fuente_table(rows: pd.DataFrame, column: str = "fuente") -> list[dict]:
    """Per document source (cert / pcer) among the OK rows: lot and manzana precision with Wilson intervals; ``[]`` when
    the table has no source column."""
    if column not in rows.columns:
        return []
    _require(rows)
    arrays = _row_arrays(rows)
    out = []
    for fuente in sorted(str(f) for f in rows.loc[rows["estado"] == "OK", column].dropna().unique()):
        sel = ((rows[column] == fuente) & (rows["estado"] == "OK")).to_numpy()
        lot = _share_row(int((arrays["lot_precision"][0] * sel).sum()), int((arrays["lot_precision"][1] * sel).sum()))
        man = _share_row(int((arrays["manzana_precision"][0] * sel).sum()),
                         int((arrays["manzana_precision"][1] * sel).sum()))
        out.append({"fuente": fuente, "n": int(sel.sum()),
                    "lot_correct": lot["correct"], "lot_scored": lot["scored"], "lot_precision": lot["precision"],
                    "lot_lo": lot["lo"], "lot_hi": lot["hi"],
                    "manzana_correct": man["correct"], "manzana_scored": man["scored"],
                    "manzana_precision": man["precision"], "manzana_lo": man["lo"], "manzana_hi": man["hi"]})
    return out


def outcome_shares(rows: pd.DataFrame) -> dict:
    """Partition of the rows into outcome classes: OK with a predial, OK at manzana only, SIN_MATCH, NO_PARSEABLE, other."""
    _require(rows, ("estado", "numero_predial_nacional"))
    estado = rows["estado"]
    has_predial = rows["numero_predial_nacional"].notna()
    counts = {
        "ok_predial": int(((estado == "OK") & has_predial).sum()),
        "ok_manzana_only": int(((estado == "OK") & ~has_predial).sum()),
        "sin_match": int((estado == "SIN_MATCH").sum()),
        "no_parseable": int((estado == "NO_PARSEABLE").sum()),
    }
    counts["other"] = int(len(rows) - sum(counts.values()))
    n = int(len(rows))
    return {"n": n, "counts": counts, "shares": {k: _ratio(v, n) if n else 0.0 for k, v in counts.items()}}


# ------------------------------------------------------------------------------------------------ paired bootstrap
def _aligned(old: pd.DataFrame, new: pd.DataFrame) -> None:
    if len(old) != len(new):
        raise ValueError(f"the runs have different row counts ({len(old)} vs {len(new)})")
    if "raw_address" in old.columns and "raw_address" in new.columns \
            and not (old["raw_address"].to_numpy() == new["raw_address"].to_numpy()).all():
        raise ValueError("rows are not aligned: raw_address differs at some position")


def _stat(num: np.ndarray, den: np.ndarray | None, idx: np.ndarray) -> np.ndarray:
    top = num[idx].sum(axis=1)
    if den is None:
        return top
    bottom = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(bottom > 0, top / bottom, np.nan)


def paired_bootstrap(old: pd.DataFrame, new: pd.DataFrame, metric: str, resamples: int = 10000, seed: int = 20261031,
                     level: float = 0.95, chunk: int = 500) -> dict:
    """Candidate (``new``) minus baseline (``old``) on rows resampled jointly, for one of ``ok``, ``lot_precision``,
    ``manzana_precision``, ``lost_correct``. Percentile interval; ``p_diff_lt0`` is the share of resamples with a
    negative difference (resamples where either ratio is undefined are skipped)."""
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    _aligned(old, new)
    a, b = _row_arrays(old), _row_arrays(new)
    if metric not in a:
        raise ValueError(f"unknown metric {metric!r}; choose one of {sorted(a)}")
    n = len(old)
    full = np.arange(n)[None, :]
    point_old = float(_stat(*a[metric], full)[0]) if n else float("nan")
    point_new = float(_stat(*b[metric], full)[0]) if n else float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    done = 0
    while done < resamples and n:
        size = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        diffs.append(_stat(*b[metric], idx) - _stat(*a[metric], idx))
        done += size
    d = np.concatenate(diffs) if diffs else np.array([])
    d = d[np.isfinite(d)]
    tail = (1 - level) / 2 * 100
    return {
        "metric": metric, "old": point_old, "new": point_new, "diff": point_new - point_old,
        "lo": float(np.percentile(d, tail)) if len(d) else float("nan"),
        "hi": float(np.percentile(d, 100 - tail)) if len(d) else float("nan"),
        "p_diff_lt0": float((d < 0).mean()) if len(d) else float("nan"),
        "resamples": int(resamples), "valid_resamples": int(len(d)), "seed": int(seed), "n_rows": int(n),
    }


# ------------------------------------------------------------------------------------------------ JEV cross-check
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); NaN when fewer than 3 pairs or either side is constant."""
    if len(x) < 3:
        return float("nan")
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _jev_group(frame: pd.DataFrame, column: str) -> dict:
    out = {}
    keys = frame[column].where(frame[column].notna(), "sin_dato")
    for key in sorted(keys.unique(), key=str):
        part = frame[keys == key]
        no = int((part["label"] == "no").sum())
        n = int(len(part))
        out[str(key)] = {"judged": n, "yes": int((part["label"] == "yes").sum()), "no": no,
                         "decisive_no": no, "indeterminate": int((part["label"] == "indeterminate").sum()),
                         "mean_jev_p": float(part["jev_p"].mean()) if n else None,
                         "decisive_no_rate": no / n if n else None}
    return out


def jev_crosscheck(rows: pd.DataFrame, resamples: int = 10000, seed: int = 20261031, level: float = 0.95) -> dict:
    """Reliability cross-check of our bands / sources / calibrated P_predio against the JEV judge (counts and rates only).

    ``rows`` is ``jev_audit_rows.parquet``: needs ``status``, ``label`` and ``jev_p``; uses ``tag_banda_confiabilidad``,
    ``tag_fuente``, ``extra_noise`` and ``extra_p_predio_cal`` when present. Only judged rows (``status == "ok"``) enter the
    means and the correlation. ``spearman_p_predio`` is the rank correlation between our calibrated P_predio and JEV's
    P(yes) on rows where both exist, with a percentile bootstrap interval (NaN when undefined; ``None`` when the table has
    no P_predio column). JEV is an LLM judge, not ground truth: this is a consistency check, not an accuracy measure.
    """
    if len(rows) == 0:                                    # an empty table has nothing to be missing
        rows = rows.reindex(columns=list(dict.fromkeys([*rows.columns, "status", "label", "jev_p"])))
    _require(rows, ("status", "label", "jev_p"))
    status = rows["status"]
    judged = rows[status == "ok"].copy()
    judged["jev_p"] = pd.to_numeric(judged["jev_p"], errors="coerce")
    n_j = int(len(judged))
    no = int((judged["label"] == "no").sum())
    out: dict = {
        "counts": {"total": int(len(rows)), "judged": n_j, "error": int((status == "error").sum()),
                   "skipped": int((status == "skipped").sum()), "yes": int((judged["label"] == "yes").sum()),
                   "no": no, "indeterminate": int((judged["label"] == "indeterminate").sum())},
        "mean_jev_p": float(judged["jev_p"].mean()) if n_j and judged["jev_p"].notna().any() else None,
        "decisive_no_rate": no / n_j if n_j else None,
    }
    for name, column in (("by_band", "tag_banda_confiabilidad"), ("by_fuente", "tag_fuente"), ("by_noise", "extra_noise")):
        out[name] = _jev_group(judged, column) if column in judged.columns and n_j else {}
    if "extra_p_predio_cal" not in rows.columns:
        out["spearman_p_predio"] = None
        return out
    ours = pd.to_numeric(judged["extra_p_predio_cal"], errors="coerce")
    keep = (ours.notna() & judged["jev_p"].notna()).to_numpy()
    x, y = ours.to_numpy(float)[keep], judged["jev_p"].to_numpy(float)[keep]
    rho = _spearman(x, y)
    lo = hi = float("nan")
    if len(x) >= 3 and resamples >= 1 and not math.isnan(rho):
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(resamples):
            idx = rng.integers(0, len(x), size=len(x))
            r = _spearman(x[idx], y[idx])
            if not math.isnan(r):
                draws.append(r)
        if draws:
            tail = (1 - level) / 2 * 100
            lo, hi = float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))
    out["spearman_p_predio"] = {"n": int(len(x)), "rho": rho, "lo": lo, "hi": hi, "resamples": int(resamples), "seed": int(seed)}
    return out
