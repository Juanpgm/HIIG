"""Clean the WFS nomenclature download before any ingestion, with a per-rule report.

Reads ``artifacts/nomenclatura_wfs.parquet`` and writes the clean rows, the quarantined rows (raw
values plus ``quarantine_reason``) and ``nomenclatura_cleaning_report.json``.

The output ``lon`` / ``lat`` are the local reprojection of ``ndcoordx`` / ``ndcoordy`` (EPSG:6249 -> 4326,
through ``--prj``); the WFS geometry is kept as ``lon_wfs`` / ``lat_wfs`` for audit and ``coord_source``
says which one each row uses.

Exit codes:
  0  done, quarantine share within the limit (0.5 %)
  1  unreadable input or a schema problem (missing columns are listed)
  3  quarantine share above the limit: outputs are still written, but a human must decide
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import nomenclatura, nomenclatura_clean  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
DEFAULT_PRJ = os.path.join(ROOT, "train_data", "nomenclatura_certificada_20261030",
                           "nomenclatura_certificada_20261030.prj")


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", default=os.path.join(ARTIFACTS, "nomenclatura_wfs.parquet"))
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "nomenclatura_clean.parquet"))
    ap.add_argument("--quarantine", default=os.path.join(ARTIFACTS, "nomenclatura_quarantine.parquet"))
    ap.add_argument("--report", default=os.path.join(ARTIFACTS, "nomenclatura_cleaning_report.json"))
    ap.add_argument("--shp-4326", default=None,
                    help="reprojected shapefile parquet: quarantine WFS points >1 m from the local reprojection")
    ap.add_argument("--max-shift-m", type=float, default=1.0)
    ap.add_argument("--prj", default=DEFAULT_PRJ,
                    help="shapefile .prj (EPSG:6249): ndcoordx/ndcoordy are reprojected with it to lon/lat")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.inp):
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1
    try:
        df = pd.read_parquet(args.inp)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot read {args.inp}: {exc}", file=sys.stderr)
        return 1
    try:
        transformer = nomenclatura.transformer_from_prj(args.prj)
    except nomenclatura.CrsError as exc:
        print(f"ERROR: {exc}. The index coordinates are the local reprojection of ndcoordx/ndcoordy; "
              "pass --prj with the shapefile's .prj.", file=sys.stderr)
        return 1

    shp_shift = None
    if args.shp_4326:
        try:
            shp = pd.read_parquet(args.shp_4326, columns=["NdID", "lon", "lat"])
            shp_shift = nomenclatura.coord_shift_by_ndid(df[["ndid", "lon", "lat"]], shp)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: cannot compare against {args.shp_4326}: {exc}", file=sys.stderr)
            return 1

    t0 = time.time()
    try:
        res = nomenclatura_clean.clean_nomenclatura(df, transformer=transformer, shp_shift=shp_shift,
                                                    max_shift_m=args.max_shift_m)
    except nomenclatura_clean.SchemaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _write_parquet(res.clean, args.out)
    _write_parquet(res.quarantine, args.quarantine)
    report = {**res.report, "input_file": os.path.abspath(args.inp), "prj_file": os.path.abspath(args.prj),
              "shp_compared": bool(args.shp_4326),
              "seconds": round(time.time() - t0, 1), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    r = res.report
    print(f"input={r['input_rows']} clean={r['clean_rows']} quarantined={r['quarantined_rows']} "
          f"share={r['quarantine_share']:.4%}", flush=True)
    for k, v in r["rules"].items():
        print(f"  {k}: {v}", flush=True)
    s = r["coord_shift_local_vs_wfs"]
    if s["n"]:
        print(f"  coordinates: {r['coord_source_counts']}; local vs WFS shift median={s['median_m']:.2f} m "
              f"p99={s['p99_m']:.2f} m max={s['max_m']:.2f} m share>2m={s['share_over_2m']:.4%}", flush=True)
    if r["schema"]["extra_columns"]:
        print(f"  ignored extra columns: {r['schema']['extra_columns']}", flush=True)
    if not r["gate_passed"]:
        print(f"WARNING: quarantine share {r['quarantine_share']:.4%} exceeds {r['quarantine_share_limit']:.2%}. "
              "Stop and review the quarantine file before ingesting.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
