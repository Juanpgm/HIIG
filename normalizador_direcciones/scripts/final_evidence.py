"""Aggregate evidence for the final evaluation (counts and rates only; no address text is written).

Two sub-commands, both CPU-only and read-only on the experiment directories they read:

  seed-compare   side-by-side of calibrated runs on DEV (gold + all), calibration quality, tau, absolute differences, and the
                 documented selection rule (``selection_final.json``).
  evidence       for the candidate directory (default after ``run_experiment.py --final``): confidence curve vs baseline,
                 reliability-diagram data, bands, by-fuente tables, paired bootstrap and outcome shares, dev and frozen.

Usage (project root):
  .venv/Scripts/python.exe scripts/final_evidence.py seed-compare --run s42=artifacts/experiments/<a> --run s43=<b> \
      --fit-report s42=<fit_report.json> --baseline artifacts/experiments/baseline_prod_2026_10 \
      --out-seed artifacts/experiments/seed_comparison.json --out-selection artifacts/experiments/selection_final.json
  .venv/Scripts/python.exe scripts/final_evidence.py jev-crosscheck --rows candidate=<jev_audit_rows.parquet> --out <json>
  .venv/Scripts/python.exe scripts/final_evidence.py evidence --candidate artifacts/experiments/v2_final \
      --baseline artifacts/experiments/baseline_prod_2026_10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import final_evidence as fe  # noqa: E402
from cali_address import retrain_compare as rc  # noqa: E402

BOOTSTRAP_SEED = 20261031
BOOTSTRAP_METRICS = {   # name -> (rows table, metric)
    "gold_ok": ("gold", "ok"), "gold_lot_precision": ("gold", "lot_precision"),
    "all_manzana_precision": ("all", "manzana_precision"), "lost_correct": ("all", "lost_correct"),
}


def _rows(directory: str, tag: str) -> pd.DataFrame | None:
    path = os.path.join(directory, f"strict_eval_{tag}_rows.parquet")
    return pd.read_parquet(path) if os.path.isfile(path) else None


def _json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_new(path: str, payload) -> None:
    """Write JSON, refusing to overwrite (``FileExistsError``)."""
    if os.path.exists(path):
        raise FileExistsError(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: None if isinstance(o, float) and o != o else float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))


def _clean(obj):
    """NaN -> None recursively, so the files are strict JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if obj != obj else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ------------------------------------------------------------------------------------------------ seed comparison
def _flat(directory: str, rows_infix: str = "") -> dict:
    gold, alln = _rows(directory, f"dev_gold{rows_infix}"), _rows(directory, f"dev_all{rows_infix}")
    if gold is None or alln is None:
        raise FileNotFoundError(f"{directory} lacks strict_eval_dev_gold{rows_infix}_rows.parquet / "
                                f"strict_eval_dev_all{rows_infix}_rows.parquet")
    g, a = fe.run_metrics(gold), fe.run_metrics(alln)
    return {
        "gold_n": g["n"], "gold_ok": g["ok"], "gold_ok_coverage": g["ok_share"],
        "gold_lot_precision": g["lot_precision"], "gold_lot_n": g["lot_emitted_scored"], "gold_lot_correct": g["lot_correct"],
        "gold_manzana_precision": g["manzana_precision"], "gold_rows_emitting_predial": g["rows_emitting_predial"],
        "all_ok": a["ok"], "all_lot_precision": a["lot_precision"], "all_lot_n": a["lot_emitted_scored"],
        "all_manzana_precision": a["manzana_precision"], "lost_correct": a["lost_correct"],
        "all_rows_emitting_predial": a["rows_emitting_predial"],
    }


def _calibration(report: dict | None) -> dict | None:
    heads = (report or {}).get("heads") if isinstance(report, dict) else None
    if not isinstance(heads, dict):
        return None
    out = {}
    for head in ("predio", "manzana"):
        h = heads.get(head) or {}
        oof, refit = h.get("oof") or {}, h.get("refit_oof") or {}
        out[head] = {"n": oof.get("n"), "ece10": oof.get("ece10"), "brier": oof.get("brier"), "auc": oof.get("auc"),
                     "refit_brier": refit.get("brier"), "refit_auc": refit.get("auc")}
    return out


def seed_comparison(runs: dict[str, str], fit_reports: dict[str, str] | None = None, rows_infix: str = "") -> dict:
    """``{"runs": {name: flat metrics + tau + calibration}, "abs_diff": {...}}`` (differences only for two runs)."""
    fit_reports = fit_reports or {}
    out_runs = {}
    for name, directory in runs.items():
        flat = _flat(directory, rows_infix)
        decision = _json(os.path.join(directory, "decision.json")) or {}
        flat["tau_p"], flat["tau_m"] = decision.get("tau_p"), decision.get("tau_m")
        report = _json(fit_reports.get(name) or os.path.join(directory, "fit_report.json"))
        flat["calibration"] = _calibration(report)
        out_runs[name] = flat
    result = {"runs": out_runs, "gold_n": next(iter(out_runs.values()))["gold_n"] if out_runs else 0}
    result["one_gold_row_pp"] = 100.0 / result["gold_n"] if result["gold_n"] else None
    names = list(out_runs)
    if len(names) == 2:
        a, b = out_runs[names[0]], out_runs[names[1]]
        diff = {}
        for key, va in a.items():
            vb = b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                diff[key] = abs(va - vb)
        for head in ("predio", "manzana"):
            ca, cb = (a.get("calibration") or {}).get(head), (b.get("calibration") or {}).get(head)
            if ca and cb:
                for k in ("ece10", "brier", "auc"):
                    if ca.get(k) is not None and cb.get(k) is not None:
                        diff[f"oof_{head}_{k}"] = abs(ca[k] - cb[k])
        result["abs_diff"] = diff
        result["pair"] = names
    return result


def selection_record(runs: dict[str, str], baseline: str, rows_infix: str = "") -> dict:
    """The selection rule applied to the dev scores of ``runs`` against the measured ``baseline`` directory."""
    flat = {name: _flat(directory, rows_infix) for name, directory in runs.items()}
    base = _flat(baseline)
    sel = fe.select_candidate(flat, base)
    return _clean({
        "rule": "docs/experiments.md: highest dev_gold_predial_lot_precision among runs with dev_gold_ok_coverage >= the "
                "measured baseline's; ties within one gold row broken by dev_manzana_precision on all dev rows",
        "baseline": {"dir": os.path.basename(os.path.normpath(baseline)), **{k: base[k] for k in (
            "gold_ok", "gold_ok_coverage", "gold_lot_precision", "gold_lot_n", "all_manzana_precision", "lost_correct")}},
        "candidates": {name: {k: v[k] for k in (
            "gold_ok", "gold_ok_coverage", "gold_lot_precision", "gold_lot_n", "gold_lot_correct", "all_manzana_precision",
            "lost_correct")} for name, v in flat.items()},
        **sel,
    })


# ------------------------------------------------------------------------------------------------ evidence bundle
def _y(series: pd.Series, mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    known, value = fe._flags(series)
    keep = known & mask.to_numpy()
    return np.where(keep, value.astype(float), np.nan), keep


def reliability_data(rows: pd.DataFrame) -> dict:
    """Reliability-diagram bins for P_predio (OK rows that emit a predial, lot correctness) and P_manzana (OK rows with a
    manzana label); also the mean predicted vs observed, so the bias is visible without the bins."""
    ok = rows["estado"] == "OK"
    out = {}
    for head, p_col, y_col, mask in (
            ("predio", "p_predio_cal", "correct_predial_lot", ok & rows["numero_predial_nacional"].notna()),
            ("manzana", "p_manzana_cal", "correct_manzana", ok)):
        if p_col not in rows.columns:
            out[head] = None
            continue
        y, keep = _y(rows[y_col], mask)
        p = pd.to_numeric(rows[p_col], errors="coerce").to_numpy(float)
        sel = keep & np.isfinite(p)
        out[head] = {"n": int(sel.sum()), "mean_predicted": float(p[sel].mean()) if sel.any() else None,
                     "observed": float(y[sel].mean()) if sel.any() else None, "bins": fe.reliability_bins(p[sel], y[sel])}
    return out


SECOND_LOOK_NOTE = ("SECOND LOOK, NOT BLIND: the catastro fallback was added after seeing the round-1 frozen result "
                    "(coverage gap); nothing was tuned on frozen, but this frozen scoring is not an independent test.")


def round_comparison(round1: str, round2: str) -> dict:
    """Frozen round 1 vs round 2 (counts and rates only): headline metrics of each, the difference, the gold lots missing
    from the index (from each ``experiment.json``) and the row transitions of OK. Rows are joined by position and
    ``raw_address`` must agree (``ValueError`` otherwise); ``FileNotFoundError`` when a frozen rows table is missing."""
    out: dict[str, object] = {"definition": "round2 minus round1 on the SAME frozen rows"}
    for scope in ("gold", "all"):
        r1, r2 = _rows(round1, f"final_{scope}"), _rows(round2, f"final_{scope}")
        if r1 is None or r2 is None:
            raise FileNotFoundError(f"strict_eval_final_{scope}_rows.parquet is needed in both round directories")
        fe._aligned(r1, r2)
        m1, m2 = fe.run_metrics(r1), fe.run_metrics(r2)
        for m, directory in ((m1, round1), (m2, round2)):
            exp = _json(os.path.join(directory, "experiment.json")) or {}
            section = ((exp.get("frozen_test") or {}).get(scope) or {})
            key = "test_gold_lot_missing_from_index" if scope == "gold" else "test_lot_missing_from_index"
            m["lot_missing_from_index"] = section.get(key)
        ok1, ok2 = (r1["estado"] == "OK").to_numpy(), (r2["estado"] == "OK").to_numpy()
        lot1 = fe._flags(r1["correct_predial_lot"])
        lot2 = fe._flags(r2["correct_predial_lot"])
        c1, c2 = lot1[0] & lot1[1] & ok1, lot2[0] & lot2[1] & ok2
        out[f"frozen_{scope}"] = {
            "round1": m1, "round2": m2,
            "delta": {k: (m2[k] - m1[k]) for k in ("ok", "lot_precision", "manzana_precision", "lost_correct",
                                                   "rows_emitting_predial")},
            "transitions": {"ok_gained": int((~ok1 & ok2).sum()), "ok_lost": int((ok1 & ~ok2).sum()),
                            "both_ok": int((ok1 & ok2).sum()), "neither_ok": int((~ok1 & ~ok2).sum())},
            "lot_correct_transitions": {"gained": int((~c1 & c2).sum()), "lost": int((c1 & ~c2).sum()),
                                        "both": int((c1 & c2).sum())}}
    return out


def _tables(directory: str, kind: str) -> dict:
    """``{set: rows}`` for dev/frozen x gold/all that exist in the directory."""
    names = {"dev": "dev", "frozen": "final"}
    out = {}
    for split, prefix in names.items():
        for scope in ("gold", "all"):
            frame = _rows(directory, f"{prefix}_{scope}")
            if frame is not None:
                out[f"{split}_{scope}"] = frame
    return out


def evidence(candidate: str, baseline: str, resamples: int, round1: str | None = None,
             second_look: bool = False) -> dict[str, object]:
    cand, base = _tables(candidate, "candidate"), _tables(baseline, "baseline")
    if "dev_gold" not in cand or "dev_gold" not in base:
        raise FileNotFoundError("both runs need strict_eval_dev_gold_rows.parquet")
    files: dict[str, object] = {}
    files["confidence_curve_compare.json"] = {
        "definition": "relative A10: dev gold rows, OK rows with confianza >= t; share over all gold rows",
        "thresholds": list(rc.DEFAULT_CONFIDENCE_THRESHOLDS),
        "candidate": rc.confidence_curve(cand["dev_gold"]), "baseline": rc.confidence_curve(base["dev_gold"])}
    if "frozen_gold" in cand and "frozen_gold" in base:
        files["confidence_curve_compare.json"]["frozen"] = {
            "definition": "same curve on the frozen gold rows (reported, not a criterion)",
            "candidate": rc.confidence_curve(cand["frozen_gold"]), "baseline": rc.confidence_curve(base["frozen_gold"])}
    files["reliability_diagrams.json"] = {
        "note": "P_predio / P_manzana of the calibrated layer vs observed; dev all mixes gold with noisy GPS labels "
                "(observed is lower there by construction, see calib_design/SPEC.md section 2)",
        **{k: reliability_data(cand[k]) for k in ("dev_gold", "dev_all", "frozen_gold") if k in cand}}
    files["bands.json"] = {k: fe.band_table(v) for k, v in cand.items() if "banda_confiabilidad" in v.columns}
    files["by_fuente.json"] = {"candidate": {k: fe.fuente_table(v) for k, v in cand.items()},
                               "baseline": {k: fe.fuente_table(v) for k, v in base.items()}}
    files["outcome_shares.json"] = {"candidate": {k: fe.outcome_shares(v) for k, v in cand.items()},
                                    "baseline": {k: fe.outcome_shares(v) for k, v in base.items()}}
    boot: dict[str, object] = {"dev": {}, "frozen": None, "seed": BOOTSTRAP_SEED, "resamples": resamples}
    for split in ("dev", "frozen"):
        if f"{split}_gold" in cand and f"{split}_all" in cand and f"{split}_gold" in base and f"{split}_all" in base:
            boot[split] = {name: fe.paired_bootstrap(base[f"{split}_{scope}"], cand[f"{split}_{scope}"], metric,
                                                     resamples=resamples, seed=BOOTSTRAP_SEED)
                           for name, (scope, metric) in BOOTSTRAP_METRICS.items()}
    boot["note"] = "diff = candidate minus baseline; percentile 95 % interval; rows resampled jointly"
    files["bootstrap.json"] = boot
    if round1:
        files["round_comparison.json"] = round_comparison(round1, candidate)
    if second_look:
        files = {name: {**payload, "frozen_second_look": True, "frozen_second_look_note": SECOND_LOOK_NOTE}
                 for name, payload in files.items()}
    return files


# ------------------------------------------------------------------------------------------------ CLI
def _parse_pairs(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items or []:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"expected NAME=PATH, got {item!r}")
        out[name] = path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("seed-compare")
    sc.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    sc.add_argument("--fit-report", action="append", default=[], metavar="NAME=PATH")
    sc.add_argument("--baseline", required=True)
    sc.add_argument("--rows-infix", default="", help="e.g. _validate: read strict_eval_dev_*_validate_rows.parquet")
    sc.add_argument("--out-seed", required=True)
    sc.add_argument("--out-selection", required=True)
    ev = sub.add_parser("evidence")
    ev.add_argument("--candidate", required=True)
    ev.add_argument("--baseline", required=True)
    ev.add_argument("--resamples", type=int, default=10000)
    ev.add_argument("--round1", default=None, help="round-1 experiment dir: also writes round_comparison.json")
    ev.add_argument("--frozen-second-look", action="store_true",
                    help="mark every output as derived from a non-blind (second look) frozen scoring")
    jc = sub.add_parser("jev-crosscheck", help="reliability cross-check of JEV audit rows (counts and rates only)")
    jc.add_argument("--rows", action="append", required=True, metavar="NAME=PARQUET", help="jev_audit_rows.parquet per run")
    jc.add_argument("--out", required=True)
    jc.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "jev-crosscheck":
            if os.path.exists(args.out):
                raise FileExistsError(args.out)
            runs = {name: fe.jev_crosscheck(pd.read_parquet(path), resamples=args.resamples, seed=BOOTSTRAP_SEED)
                    for name, path in _parse_pairs(args.rows).items()}
            _write_new(args.out, _clean({"note": "JEV is an LLM judge, not ground truth; counts and rates only",
                                         "resamples": args.resamples, "seed": BOOTSTRAP_SEED, "runs": runs}))
            print(f"wrote {args.out}")
            return 0
        if args.cmd == "seed-compare":
            runs = _parse_pairs(args.run)
            for path in (args.out_seed, args.out_selection):
                if os.path.exists(path):
                    raise FileExistsError(path)
            seed = _clean(seed_comparison(runs, _parse_pairs(args.fit_report), args.rows_infix))
            selection = selection_record(runs, args.baseline, args.rows_infix)
            _write_new(args.out_seed, seed)
            _write_new(args.out_selection, selection)
            print(f"selected: {selection['selected']} ({selection['reason']})")
            return 0
        files = evidence(args.candidate, args.baseline, args.resamples, args.round1, args.frozen_second_look)
        for name in files:
            if os.path.exists(os.path.join(args.candidate, name)):
                raise FileExistsError(os.path.join(args.candidate, name))
        for name, payload in files.items():
            _write_new(os.path.join(args.candidate, name), _clean(payload))
        print(f"wrote {len(files)} files to {args.candidate}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
