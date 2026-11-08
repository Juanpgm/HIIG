"""Compare a new index with the production one and write ``compare_indexes.json`` (aggregates only).

    .venv/Scripts/python scripts/compare_indexes.py --old artifacts --new artifacts/index_v2a_cert

``--old`` is a directory with ``catastro_docs.parquet`` (and ``splits/dev.parquet`` +
``splits/frozen_test.parquet`` for the gold coverage); ``--new`` is a built index directory. The report holds
counts, shares and quantiles: no addresses, no row-level data. By default it is written next to the index
(``<index_dir>/compare_indexes.json``, a file ``index_meta.json`` does not list, so the index stays valid);
pass ``--out`` to write it elsewhere and leave the index directory untouched.

Sections: ``profile`` (docs by fuente, key overlap, npn_choice, n_predios shift, comunas, duplicates),
``gold_npn_coverage`` (share of dev + frozen gold ``gt_predial`` whose LOT, ``NPN[:21]``, is among each index's
document lots: gate A7, which allows the new index to trail the production one by ``margin_pp`` = 2.0 points),
``all_with_gt_lot_coverage`` (the same over every row with a ground-truth predial, any quality),
``production_docs_coverage`` (share of the old index's documents still reached by the new one, by lot and by
``doc_key``), ``coordinate_shift`` (metres, shared NPNs) and ``index_meta`` (counts copied from the new index's
meta). ``npn_catastro`` is never read.
Exit code 2 when a required input is missing.
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

from cali_address import retrain_compare as rc  # noqa: E402
from cali_address.paths import DEFAULT_ARTIFACTS_DIR, IndexMetaError, read_index_meta  # noqa: E402

DOCS = "catastro_docs.parquet"
SPLIT_FILES = {"dev": "dev.parquet", "frozen_test": "frozen_test.parquet"}
#: Keys copied from ``index_meta.json``: counts and flags only (its ``sources`` hold local paths).
META_KEYS = ("n_docs", "tiers", "counts_by_fuente", "counts_by_estado", "accounting", "interior_strip_method",
             "npn_choice", "disagreements", "address_key_collisions", "docs_from_units_only", "parse_failure_rate",
             "fallback_doc_share_of_docs", "queries", "units_rows", "train_pairs", "pcer", "catastro", "crs",
             "pre_normalize_separates_letters", "oracle_ceiling")


def _read_gold_splits(splits_dir: str) -> dict[str, pd.DataFrame]:
    found = {}
    for name, fname in SPLIT_FILES.items():
        path = os.path.join(splits_dir, fname)
        if os.path.isfile(path):
            cols = ["gt_predial", "gt_quality"]
            frame = pd.read_parquet(path)
            found[name] = frame[[c for c in cols if c in frame.columns]]
    return found


#: A7 lets the new index's gold lot coverage trail the production index's by this many percentage points.
MARGIN_PP = 2.0


def _coverage(splits: dict[str, pd.DataFrame], docs: pd.DataFrame, gold_only: bool = True) -> dict:
    both = pd.concat(list(splits.values()), ignore_index=True)
    return {"all": rc.gold_npn_coverage(both, docs, gold_only=gold_only),
            "by_split": {name: rc.gold_npn_coverage(frame, docs, gold_only=gold_only)
                         for name, frame in splits.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=DEFAULT_ARTIFACTS_DIR, help="directory with the old catastro_docs.parquet")
    ap.add_argument("--new", required=True, help="built index directory")
    ap.add_argument("--splits-dir", default=None, help="directory with dev.parquet / frozen_test.parquet "
                                                        "(default <old>/splits)")
    ap.add_argument("--out", default=None, help="output path (default <new>/compare_indexes.json)")
    args = ap.parse_args(argv)

    old_docs_path, new_docs_path = os.path.join(args.old, DOCS), os.path.join(args.new, DOCS)
    for path in (old_docs_path, new_docs_path):
        if not os.path.isfile(path):
            print(f"ERROR: {DOCS} not found at {path}", file=sys.stderr)
            return 2

    warnings: list[str] = []
    old_docs, new_docs = pd.read_parquet(old_docs_path), pd.read_parquet(new_docs_path)
    try:
        profile = rc.index_profile(old_docs, new_docs)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    splits = _read_gold_splits(args.splits_dir or os.path.join(args.old, "splits"))
    coverage = all_gt_coverage = None
    if splits:
        old_cov, new_cov = _coverage(splits, old_docs), _coverage(splits, new_docs)
        o, n = old_cov["all"]["share_present"], new_cov["all"]["share_present"]
        known = n is not None and o is not None
        coverage = {"old": old_cov, "new": new_cov, "margin_pp": MARGIN_PP,
                    "delta_pp": (n - o) * 100 if known else None,
                    "new_ge_old": (n >= o) if known else None,
                    "new_ge_old_minus_margin": ((n - o) * 100 >= -MARGIN_PP - 1e-9) if known else None}
        all_gt_coverage = {"old": _coverage(splits, old_docs, gold_only=False),
                           "new": _coverage(splits, new_docs, gold_only=False)}
    else:
        warnings.append("gold splits not found (dev.parquet / frozen_test.parquet): gold NPN coverage skipped")

    try:
        meta = read_index_meta(args.new)
    except IndexMetaError as exc:
        meta = None
        warnings.append(f"index_meta.json unusable: {exc}")
    if meta is None and not any("index_meta" in w for w in warnings):
        warnings.append("index_meta.json not found in the new index directory")

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "new_index": os.path.basename(os.path.abspath(args.new)),
        "profile": profile,
        "gold_npn_coverage": coverage,
        "all_with_gt_lot_coverage": all_gt_coverage,
        "production_docs_coverage": rc.docs_lot_coverage(old_docs, new_docs),
        "coordinate_shift": rc.coordinate_shift(old_docs, new_docs),
        "index_meta": ({k: meta[k] for k in META_KEYS if k in meta} if meta else None),
        "warnings": warnings,
    }
    out = args.out or os.path.join(args.new, "compare_indexes.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
