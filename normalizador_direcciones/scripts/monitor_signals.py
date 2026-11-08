"""Drift signals of a calibrated run (aggregates only), optionally compared with a baseline signals file.

Input: the ``strict_eval_<tag>_rows.parquet`` of ``scripts/eval_strict.py`` (or any frame with the output columns
``estado``, ``motivo``, ``confianza``, ``margen``, ``banda_confiabilidad``, ``p_predio_cal``, ``fuente``). Output: a JSON
with the signals of SPEC section 10 (P_predio quantiles, band shares, share of pcer documents, abstention rate, mean
margen / confianza of the OK rows) and, with ``--baseline``, an ``alarms`` block with the verdict of each rule.

  .venv/Scripts/python.exe scripts/monitor_signals.py --rows <dir>/strict_eval_dev_all_rows.parquet --out signals_base.json
  .venv/Scripts/python.exe scripts/monitor_signals.py --rows <new rows> --baseline signals_base.json --out signals_new.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.calibration_report import compare_monitoring, monitoring_signals  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="rows parquet of a calibrated run")
    ap.add_argument("--baseline", default=None, help="signals json of the reference run (adds the alarms block)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and not args.overwrite:
        print(f"{args.out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    try:
        frame = pd.read_parquet(args.rows)
        baseline = None
        if args.baseline:
            with open(args.baseline, "r", encoding="utf-8") as fh:
                baseline = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"cannot read input: {exc}", file=sys.stderr)
        return 2
    signals = monitoring_signals(frame)
    if baseline is not None:
        signals["alarms"] = compare_monitoring(baseline, signals)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(signals, fh, indent=1)
    fired = [name for name, a in signals.get("alarms", {}).items() if a["alarm"]]
    print(f"signals -> {args.out}" + (f" | ALARMS: {', '.join(fired)}" if fired else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
