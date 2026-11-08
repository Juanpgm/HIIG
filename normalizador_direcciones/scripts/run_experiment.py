"""Run one training experiment end to end and log it on the leaderboard.

    PYTHONPATH=src python scripts/run_experiment.py --tag mixed_r30 \
        --set real_frac=0.3 --set epochs=6 --set lr=2e-3
    PYTHONPATH=src python scripts/run_experiment.py --tag ft_a --mode finetune \
        --set lr=2e-4 --set epochs=3 --set freeze_aux=true
    PYTHONPATH=src python scripts/run_experiment.py --tag baseline_prod --eval-only --from-dir artifacts

Pipeline (everything lands in ``artifacts/experiments/<tag>/``; production
artifacts are never written):

    train.py  ->  (finish_training_artifacts.py if the embeddings are missing)
              ->  tune_threshold.py  ->  eval_strict.py on dev (all rows, gold-only)
              ->  leaderboard row in artifacts/experiments/leaderboard.csv

``frozen_test`` is scored ONLY with ``--final`` and the runner refuses to score it
otherwise. ``--final`` is for the single, already-selected candidate.

Selection metric (documented rule, decided before looking at any result)
-----------------------------------------------------------------------
  1. Among runs whose ``dev_gold_ok_coverage`` (OK gold rows / gold rows) is at
     least the MEASURED baseline's (baseline_prod_2026_10), pick the highest ``dev_gold_predial_lot_precision``.
  2. Ties (within one gold row) are broken by ``dev_manzana_precision`` on all
     dev rows.
  ``real_val_recall@k`` and ``syn_recall@k`` are diagnostics, not selectors.
  NEVER select by ``frozen_test`` and never tune on it.

Get the baseline row with ``--eval-only --from-dir artifacts`` (evaluates the
production artifacts read-only, writing only into the experiment directory).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.experiment import (  # noqa: E402
    IndexValidationError, append_leaderboard, check_training_complete, effective_sets, experiment_dir,
    guard_frozen, index_meta_record, leaderboard_row, link_index_files, parse_overrides, prepare_out_dir,
    safe_tag, validate_index_dir, write_experiment_json,
)
from cali_address.paths import (  # noqa: E402
    DEFAULT_ARTIFACTS_DIR, code_fingerprint, sha256_serving_files,
)

SPLITS = os.path.join(DEFAULT_ARTIFACTS_DIR, "splits")
DEV_PATH = os.path.join(SPLITS, "dev.parquet")
FROZEN_PATH = os.path.join(SPLITS, "frozen_test.parquet")
LEADERBOARD = os.path.join(DEFAULT_ARTIFACTS_DIR, "experiments", "leaderboard.csv")


def plan_evals(final: bool, eval_set: str = "dev") -> list[tuple[str, str, bool]]:
    """``[(tag, scored_path, gold_only)]`` to run; raises if frozen_test is requested without ``final``."""
    plan: list[tuple[str, str, bool]] = []
    if eval_set in ("dev", "both"):
        plan += [("dev_all", DEV_PATH, False), ("dev_gold", DEV_PATH, True)]
    if eval_set in ("frozen_test", "both") or final:
        guard_frozen(FROZEN_PATH, final)
        plan += [("final_all", FROZEN_PATH, False), ("final_gold", FROZEN_PATH, True)]
    return plan


def _run(cmd: list[str], log, env: dict | None = None) -> None:
    """Run a child process, teeing its output to the console and the run log."""
    e = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src"), "PYTHONIOENCODING": "utf-8", **(env or {})}
    log.write(f"$ {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    if proc.wait() != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _read_summary(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _training_check(out: str) -> dict:
    """Early-stop verdict of the run in ``out`` (``train_report.json`` + the configured epochs)."""
    exp = _read_json(os.path.join(out, "experiment.json"))
    configured = (exp or {}).get("hyperparams", {}).get("epochs") if isinstance(exp, dict) else None
    return check_training_complete(_read_json(os.path.join(out, "train_report.json")), configured)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--mode", choices=("mixed", "finetune"), default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny GPU/CPU run to prove the pipeline")
    ap.add_argument("--realistic-noise", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--final", action="store_true", help="also score frozen_test (final candidate only)")
    ap.add_argument("--eval-set", choices=("dev", "frozen_test", "both"), default="dev")
    ap.add_argument("--eval-only", action="store_true", help="skip training/tuning; evaluate --from-dir")
    ap.add_argument("--from-dir", default=None, help="with --eval-only: artifacts directory to evaluate (read-only)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--root", default=None, help="experiments root (default artifacts/experiments)")
    ap.add_argument("--leaderboard", default=None)
    ap.add_argument("--no-hash", action="store_true", help="skip the production sha256 before/after check")
    ap.add_argument("--index-dir", default=None,
                    help="a built index directory (index_meta.json written last): validated, linked into the "
                         "experiment dir so it is self-contained; max_minutes defaults to 120 for the run")
    args = ap.parse_args(argv)

    tag = safe_tag(args.tag)
    parse_overrides(args.sets)                       # fail on unknown keys before any work
    plan = plan_evals(args.final, args.eval_set)     # refuse frozen_test without --final, before any work
    if args.eval_only and not args.from_dir:
        ap.error("--eval-only needs --from-dir")
    if args.eval_only and args.index_dir:
        ap.error("--index-dir is for training runs; --eval-only evaluates --from-dir (a linked experiment dir)")
    leaderboard = args.leaderboard or (
        os.path.join(args.root, "leaderboard.csv") if args.root else LEADERBOARD)

    index_dir = os.path.abspath(args.index_dir) if args.index_dir else None
    index_rec = None
    if index_dir:
        try:
            index_meta = validate_index_dir(index_dir)      # before anything is created
        except IndexValidationError as exc:
            print(f"ERROR: index {index_dir} is not usable: {exc}", file=sys.stderr)
            return 2
        index_rec = index_meta_record(index_dir, index_meta)

    out = prepare_out_dir(experiment_dir(tag, args.root), overwrite=args.overwrite)
    if index_dir:
        link_index_files(index_dir, out, index_meta, overwrite=args.overwrite)
    art_dir = os.path.abspath(args.from_dir) if args.eval_only else out
    t0 = time.time()
    sha_before = None if args.no_hash else sha256_serving_files()
    log_path = os.path.join(out, "run.log")
    py = sys.executable
    tune_env = {"N_TUNE": "1500"} if args.smoke else {}
    training_check = None

    with open(log_path, "a", encoding="utf-8") as log:
        if not args.eval_only:
            cmd = [py, os.path.join(ROOT, "scripts", "train.py"), "--out-dir", out, "--tag", tag]
            for s in effective_sets(args.sets, index_dir):
                cmd += ["--set", s]
            if args.mode:
                cmd += ["--mode", args.mode]
            if args.smoke:
                cmd.append("--smoke")
            if args.realistic_noise:
                cmd.append("--realistic-noise")
            if args.device:
                cmd += ["--device", args.device]
            _run(cmd, log)
            training_check = _training_check(out)
            if not training_check["ok"]:
                # A run cut short is not a candidate: say so loudly and do not spend time scoring it.
                print("ERROR: " + "; ".join(training_check["problems"]), file=sys.stderr)
                plan = []
            else:
                if not os.path.exists(os.path.join(out, "catastro_emb.pt")):
                    _run([py, os.path.join(ROOT, "scripts", "finish_training_artifacts.py"),
                          "--artifacts-dir", out], log)
                _run([py, os.path.join(ROOT, "scripts", "tune_threshold.py"), "--artifacts-dir", out],
                     log, env=tune_env)

        for eval_tag, scored, gold in plan:
            cmd = [py, os.path.join(ROOT, "scripts", "eval_strict.py"), "--scored", scored,
                   "--artifacts-dir", art_dir, "--out-dir", out, "--tag", eval_tag]
            if gold:
                cmd.append("--gold-only")
            if args.final:
                cmd.append("--final")
            if args.device:
                cmd += ["--device", args.device]
            _run(cmd, log)

    seconds = time.time() - t0
    sha_after = None if args.no_hash else sha256_serving_files()
    untouched = (sha_before == sha_after) if sha_before is not None else None

    history, mode = [], ("eval-only" if args.eval_only else (args.mode or "mixed"))
    exp_path = os.path.join(out, "experiment.json")
    if os.path.exists(exp_path) and not args.eval_only:
        with open(exp_path, encoding="utf-8") as fh:
            exp = json.load(fh)
        m = exp.get("metrics", {})
        history = [{k.replace("syn_", ""): v for k, v in m.items()}]
        mode = exp.get("hyperparams", {}).get("mode", mode)

    def summary(name):
        p = os.path.join(out, f"strict_eval_{name}_summary.csv")
        return _read_summary(p) if os.path.exists(p) else []

    incomplete = training_check is not None and not training_check["ok"]
    row = leaderboard_row(tag, mode, history, summary("dev_all"), summary("dev_gold"), seconds,
                          smoke=args.smoke, final=args.final, fingerprint=code_fingerprint(),
                          index_id=(index_rec or {}).get("index_id"),
                          stopped_early=(incomplete if training_check is not None else None),
                          final_gold=(summary("final_gold") if args.final and not incomplete else None))
    append_leaderboard(leaderboard, row)
    extra = {"leaderboard_row": row, "runner_seconds": round(seconds, 1),
             "production_sha256_before": sha_before, "production_sha256_after": sha_after,
             "production_untouched": untouched}
    if index_rec is not None:
        extra["index_meta"] = index_rec
    if training_check is not None:
        extra["training_check"] = training_check
    if args.final and not incomplete:
        from cali_address.experiment import frozen_section
        extra["frozen_test"] = frozen_section(summary("final_all"), summary("final_gold"))
    write_experiment_json(out, extra)
    print(f"leaderboard row appended to {leaderboard}")
    print(row)
    if untouched is False:
        print("ERROR: production serving artifacts changed during the run", file=sys.stderr)
        return 3
    if incomplete:
        print("ERROR: training did not finish; this run must not be used as a candidate", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
