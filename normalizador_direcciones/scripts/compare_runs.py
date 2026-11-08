"""Compare two experiment runs row by row and evaluate the acceptance checklist.

    .venv/Scripts/python scripts/compare_runs.py --old artifacts/experiments/baseline_prod_2026_10 \
        --new artifacts/experiments/v2_final

Inputs per run: ``strict_eval_dev_gold_rows.parquet`` (required) and ``strict_eval_final_gold_rows.parquet``
(used when BOTH runs have it), written by ``scripts/eval_strict.py``. The rows are joined BY POSITION and
``raw_address`` must be identical at every position; otherwise the script exits 2 and writes nothing.

Outputs (default: the ``--new`` directory, which lives under the untracked ``artifacts/``):
  probe_agreement.json   counts only: both_ok_predial_match / only_old / only_new / both_wrong (+ by fuente, by set)
  probe_rows.parquet     row-level table WITH addresses: never commit it
  acceptance.json        ``check_acceptance`` results (needs experiment.json in both runs and the thresholds file)

Gates a run cannot know (A7, A9, A11, A12's violations...) come from ``--extra-metrics`` (a JSON object with
``candidate`` / ``baseline`` / ``candidate_correct`` sections) and, for A7, from ``--compare-indexes``. A12 needs
``--correct-run``: the experiment directory of the same candidate scored with ``validate_outputs=correct``.
Anything not provided is reported as "not evaluated".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import retrain_compare as rc  # noqa: E402
from cali_address.paths import DEFAULT_ARTIFACTS_DIR  # noqa: E402

SETS = ("dev_gold", "final_gold")
DEFAULT_THRESHOLDS = os.path.join(DEFAULT_ARTIFACTS_DIR, "experiments", "retrain_2026_10_acceptance.json")
SECOND_LOOK = {"frozen_second_look": True,
               "frozen_second_look_note": ("SECOND LOOK, NOT BLIND: the catastro fallback was added after seeing the round-1 "
                                           "frozen result; nothing was tuned on frozen, but this scoring is not independent.")}


def _rows_path(exp_dir: str, name: str) -> str:
    return os.path.join(exp_dir, f"strict_eval_{name}_rows.parquet")


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _from_compare_indexes(path: str) -> dict:
    """Candidate gate-A7 inputs derivable from ``compare_indexes.json`` (aggregates already computed there)."""
    data = _read_json(path)
    out: dict = {}
    if not isinstance(data, dict):
        return out
    cov = data.get("gold_npn_coverage") or {}            # lot-level (NPN[:21]) since the 2026-10-31 amendment
    new, old = ((cov.get("new") or {}).get("all") or {}), ((cov.get("old") or {}).get("all") or {})
    if new.get("share_present") is not None:
        out["gold_lot_coverage_new"] = new["share_present"]
    if old.get("share_present") is not None:
        out["gold_lot_coverage_old"] = old["share_present"]
    if new.get("n_missing") is not None:
        out["gold_lot_missing_new"] = new["n_missing"]
    # A7(c1)/(c2): the builder's oracle ceiling (index_meta.json), copied into compare_indexes.json
    out.update(rc.oracle_ceiling_metrics((data.get("index_meta") or {}).get("oracle_ceiling")
                                         if isinstance(data.get("index_meta"), dict) else None))
    meta = data.get("index_meta") or {}
    if meta.get("parse_failure_rate") is not None:
        out["parser_failure_rate_cert"] = meta["parse_failure_rate"]
    dropped = ((meta.get("accounting") or {}).get("buckets") or {}).get("dropped")
    if dropped is not None:
        out["accounting_dropped"] = dropped
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="baseline experiment directory")
    ap.add_argument("--new", required=True, help="candidate experiment directory")
    ap.add_argument("--out-dir", default=None, help="where to write the outputs (default: --new)")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--extra-metrics", default=None)
    ap.add_argument("--compare-indexes", default=None, help="compare_indexes.json feeding gate A7")
    ap.add_argument("--correct-run", default=None, help="experiment dir of the candidate with validate_outputs=correct (A12)")
    ap.add_argument("--frozen-second-look", action="store_true",
                    help="mark every output as derived from a non-blind (second look) frozen scoring")
    args = ap.parse_args(argv)

    sets = [s for s in SETS if os.path.isfile(_rows_path(args.old, s)) and os.path.isfile(_rows_path(args.new, s))]
    if "dev_gold" not in sets:
        for d in (args.old, args.new):
            if not os.path.isfile(_rows_path(d, "dev_gold")):
                print(f"ERROR: {_rows_path(d, 'dev_gold')} not found (strict_eval rows of the dev gold set)",
                      file=sys.stderr)
                return 2
    old_rows = pd.concat([pd.read_parquet(_rows_path(args.old, s)).assign(_set=s) for s in sets], ignore_index=True)
    new_rows = pd.concat([pd.read_parquet(_rows_path(args.new, s)).assign(_set=s) for s in sets], ignore_index=True)

    docs_new = None
    docs_path = os.path.join(args.new, "catastro_docs.parquet")
    if os.path.isfile(docs_path):
        try:                                   # the by-fuente breakdown needs these two columns; skip it without them
            docs_new = pd.read_parquet(docs_path, columns=["numero_predial_nacional", "fuente"])
        except (KeyError, ValueError, OSError):
            docs_new = None
    try:
        table = rc.probe_table(old_rows, new_rows, docs_new)
    except (rc.ProbeMisalignmentError, ValueError) as exc:
        print(f"ERROR: cannot compare the runs: {exc}", file=sys.stderr)
        return 2
    table["set"] = old_rows["_set"].to_numpy()

    summary = rc.probe_agreement(old_rows, new_rows, docs_new)
    summary["sets"] = sets
    if args.frozen_second_look:
        summary.update(SECOND_LOOK)
        table["frozen_second_look"] = True
    if "correct_predial_lot" in old_rows.columns and "correct_predial_lot" in new_rows.columns:
        lot = rc.probe_agreement(old_rows, new_rows, correct_col="correct_predial_lot")      # same rows, lot-level hit
        summary["lot_level"] = {"n": lot["n"], "counts": lot["counts"]}
    summary["by_set"] = {s: {b: int(((table["set"] == s) & (table["bucket"] == b)).sum()) for b in rc.PROBE_BUCKETS}
                         for s in sets}

    out_dir = args.out_dir or args.new
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_agreement.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    table.to_parquet(os.path.join(out_dir, "probe_rows.parquet"), index=False)
    print(f"agreement over {summary['n']} rows ({', '.join(sets)}): {summary['counts']}")

    _acceptance(args, out_dir)
    return 0


def _acceptance(args, out_dir: str) -> None:
    thresholds = _read_json(args.thresholds)
    if not isinstance(thresholds, dict):
        print(f"acceptance skipped: thresholds file {args.thresholds} not found or unreadable")
        return
    cand, base = rc.collect_run_metrics(args.new), rc.collect_run_metrics(args.old)
    if "dev_gold_ok" not in cand or "dev_gold_ok" not in base:
        print("acceptance skipped: both runs need an experiment.json with a leaderboard_row")
        return
    metrics = {"candidate": {**cand, **_from_compare_indexes(args.compare_indexes)} if args.compare_indexes else cand,
               "baseline": base}
    if args.correct_run:
        metrics["candidate_correct"] = rc.collect_run_metrics(args.correct_run)
    extra = _read_json(args.extra_metrics) if args.extra_metrics else None
    if isinstance(extra, dict):
        for section in ("candidate", "baseline", "candidate_correct"):
            if isinstance(extra.get(section), dict):
                metrics.setdefault(section, {}).update(extra[section])
    results = rc.check_acceptance(metrics, thresholds)
    payload = {"results": results, "summary": rc.summarize_acceptance(results)}
    if args.frozen_second_look:
        payload.update(SECOND_LOOK)
    with open(os.path.join(out_dir, "acceptance.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    s = payload["summary"]
    print(f"acceptance: passed {s['passed']} | failed {s['failed']} | not evaluated {s['not_evaluated']}")


if __name__ == "__main__":
    sys.exit(main())
