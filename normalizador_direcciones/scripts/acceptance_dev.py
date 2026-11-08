"""Acceptance checklist A1..A12 on DEV-ONLY evidence, written to ``acceptance_dev.json`` in the candidate directory.

No ``frozen_test`` is read: A3 (frozen gold) stays ``not_evaluated``, and so do the criteria whose evidence does not come
from a strict dev evaluation (A6 reliability refit, A7 index build, A11 build scripts) unless extra metrics exist. A
criterion that cannot be judged is NEVER reported as passed.

Evidence per experiment directory (written by ``scripts/eval_strict.py --tag dev_all|dev_gold``):
  ``strict_eval_dev_all_summary.csv`` / ``strict_eval_dev_gold_summary.csv``  headline numbers (same extraction as the
  leaderboard row), ``strict_eval_dev_gold_rows.parquet`` the confidence curve (A10), and, for A12, the same files with the
  ``_validate`` tag suffix from a ``--validate-outputs correct`` run plus its ``..._validation.csv`` (violations left).

Usage:
  .venv/Scripts/python.exe scripts/acceptance_dev.py --candidate artifacts/experiments/<cand> \
      --baseline artifacts/experiments/baseline_prod_2026_10 [--pytest-green --no-real-addresses-exit 0]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import experiment as ex  # noqa: E402
from cali_address import retrain_compare as rc  # noqa: E402

DEFAULT_THRESHOLDS = os.path.join(ROOT, "artifacts", "experiments", "retrain_2026_10_acceptance.json")
CORRECT_SUFFIX = "_validate"


def _read_summary(path: str) -> list[dict] | None:
    """Rows of a ``strict_summary`` csv as dicts, or ``None`` when missing / unreadable / without the pooled OK row."""
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if not {"dataset", "estado", "n"} <= set(frame.columns):
        return None
    rows = frame.to_dict("records")
    if not any(r.get("dataset") == "ALL" and r.get("estado") == "OK" for r in rows):
        return None
    return rows


def run_metrics(exp_dir: str, suffix: str = "") -> dict:
    """Flat metrics of one experiment directory (``suffix`` selects the tag variant, e.g. ``_validate``).

    Whatever cannot be read is OMITTED, never invented, so the checklist reports it as not evaluated.
    """
    out: dict = {}
    dev_all = _read_summary(os.path.join(exp_dir, f"strict_eval_dev_all{suffix}_summary.csv"))
    dev_gold = _read_summary(os.path.join(exp_dir, f"strict_eval_dev_gold{suffix}_summary.csv"))
    if dev_all is not None:
        out.update(ex.eval_metrics(dev_all, "dev"))
        out.update(ex.lot_metrics(dev_all, "dev"))
    if dev_gold is not None:
        out.update(ex.eval_metrics(dev_gold, "dev_gold"))
        out.update(ex.lot_metrics(dev_gold, "dev_gold"))
        n = out.get("dev_gold_n")
        out["dev_gold_ok_coverage"] = (out["dev_gold_ok"] / n) if n else None
    rows_path = os.path.join(exp_dir, f"strict_eval_dev_gold{suffix}_rows.parquet")
    if os.path.isfile(rows_path):
        try:
            out["confidence_curve"] = rc.confidence_curve(pd.read_parquet(rows_path))
        except (ValueError, OSError, KeyError):
            pass
    validation = os.path.join(exp_dir, f"strict_eval_dev_all{suffix}_validation.csv")
    if os.path.isfile(validation):
        try:
            table = pd.read_csv(validation)
            left = (pd.to_numeric(table["violations"], errors="coerce").fillna(0)
                    - pd.to_numeric(table["corrected"], errors="coerce").fillna(0)).clip(lower=0)
            out["violations_remaining_dev"] = int(left.sum())
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            pass
    return {k: v for k, v in out.items() if v is not None}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_untouched(recorded: dict, root: str, ignore_missing: bool = False) -> bool | None:
    """Whether every file of ``recorded`` (``{name: sha256}``) under ``root`` still has that hash.

    ``None`` when there is nothing to compare; a missing file is a ``False`` unless ``ignore_missing``.
    """
    if not recorded:
        return None
    for name, expected in recorded.items():
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            if ignore_missing:
                continue
            return False
        if _sha256(path) != expected:
            return False
    return True


def dev_parts_status(result: dict) -> str:
    """Status of a criterion over its DEV parts only (parts that need frozen_test are ignored).

    A12 embeds the A3 comparison, which needs frozen_test; without it the criterion as a whole is not evaluated even
    when everything measurable on dev holds. This view says what the dev evidence alone shows.
    """
    parts = [p for p in result["parts"] if "A3" not in p["name"] and "frozen_" not in p["name"]]
    if any(p["status"] == "fail" for p in parts):
        return "fail"
    if not parts or any(p["status"] == "missing" for p in parts):
        return "not_evaluated"
    return "pass"


def evaluate(candidate_dir: str, baseline_dir: str, thresholds: dict, *, pytest_green: bool | None = None,
             no_real_addresses_exit: int | None = None, production_untouched: bool | None = None,
             correct_suffix: str = CORRECT_SUFFIX) -> dict:
    """``{"results", "summary", "metrics"}`` of A1..A12 for the candidate against the measured baseline.

    ``ValueError`` when either run lacks the dev-gold headline numbers (nothing to compare).
    """
    cand, base = run_metrics(candidate_dir), run_metrics(baseline_dir)
    for name, metrics in (("candidate", cand), ("baseline", base)):
        if "dev_gold_ok" not in metrics:
            raise ValueError(f"the {name} run has no usable strict_eval_dev_gold_summary.csv (dev gold headline numbers)")
    if pytest_green is not None:
        cand["pytest_green"] = bool(pytest_green)
    if no_real_addresses_exit is not None:
        cand["no_real_addresses_exit"] = int(no_real_addresses_exit)
    if production_untouched is not None:
        cand["production_untouched"] = bool(production_untouched)
    metrics = {"candidate": cand, "baseline": base}
    corrected = run_metrics(candidate_dir, correct_suffix)
    if "dev_gold_ok" in corrected:
        metrics["candidate_correct"] = corrected
    results = rc.check_acceptance(metrics, thresholds)
    for result in results:
        if result["id"] == "A12":
            result["dev_parts_status"] = dev_parts_status(result)
    return {"results": results, "summary": rc.summarize_acceptance(results), "metrics": metrics}


def _fmt(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_table(results: list[dict]) -> str:
    """One line per criterion: id, status, value, threshold (and why, when it did not pass)."""
    lines = [f"{'id':<4} {'status':<14} {'value':<48} threshold"]
    for r in results:
        lines.append(f"{r['id'] + ' ':<4} {r['status']:<14} {_fmt(r['value'])[:48]:<48} {_fmt(r['threshold'])}")
        if "dev_parts_status" in r:
            lines.append(f"     dev parts only (no frozen_test): {r['dev_parts_status']}")
        if r["reason"] and r["status"] != "pass":
            lines.append(f"     -> {r['reason'][:200]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="candidate experiment directory (acceptance_dev.json goes here)")
    ap.add_argument("--baseline", required=True, help="measured baseline experiment directory")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--pytest-green", action="store_true", help="the full pytest run was green (A9)")
    ap.add_argument("--no-real-addresses-exit", type=int, default=None, help="exit code of check_no_real_addresses.py (A9)")
    ap.add_argument("--check-production", action="store_true",
                    help="compare artifacts/ against the sha256 recorded in the baseline's experiment.json (A8)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out_path = os.path.join(args.candidate, "acceptance_dev.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    try:
        with open(args.thresholds, "r", encoding="utf-8") as fh:
            thresholds = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"cannot read thresholds {args.thresholds}: {exc}", file=sys.stderr)
        return 2
    untouched = None
    if args.check_production:
        try:
            with open(os.path.join(args.baseline, "experiment.json"), "r", encoding="utf-8") as fh:
                recorded = json.load(fh).get("production_sha256_before", {})
        except (OSError, ValueError):
            recorded = {}
        untouched = production_untouched(recorded, os.path.join(ROOT, "artifacts"))
    try:
        outcome = evaluate(args.candidate, args.baseline, thresholds, pytest_green=True if args.pytest_green else None,
                           no_real_addresses_exit=args.no_real_addresses_exit, production_untouched=untouched)
        payload = {
            "created": dt.datetime.now().isoformat(timespec="seconds"), "mode": "dev-only",
            "candidate": os.path.basename(os.path.abspath(args.candidate)),
            "baseline": os.path.basename(os.path.abspath(args.baseline)),
            "note": "no frozen_test was read; A3 and every criterion without dev evidence are not_evaluated",
            **outcome,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except (ValueError, OSError) as exc:
        print(f"acceptance failed: {exc}", file=sys.stderr)
        return 2
    print(format_table(outcome["results"]))
    s = outcome["summary"]
    print(f"passed {s['passed']} | failed {s['failed']} | not evaluated {s['not_evaluated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
