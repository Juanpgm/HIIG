"""Cadastre fallback tier of an address index: add it, remove it, or build its experiment directory.

The tier appends the cadastre documents of the lots that NO nomenclature record has (rule and details:
``cali_address.index_fallback`` and docs/respaldo-catastro.md). It is reversible by construction: the base index is
never modified and ``remove`` gives back the base files byte for byte.

    add         base index  -> NEW index directory with the appended docs (index_meta.json written last)
    experiment  fallback index + base experiment (model, embeddings) -> NEW experiment directory; the appended docs are
                embedded with the same model on CPU, no training
    remove      fallback index (or experiment) -> NEW directory identical to the base files

Exit codes: 0 done, 1 unreadable / inconsistent input or a refused operation (message on stderr), 2 usage error.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import index_fallback, index_union, paths  # noqa: E402
from cali_address.experiment import IndexValidationError, validate_index_dir  # noqa: E402
from cali_address.parser import pre_normalize_separation  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
ORACLE_SPLITS = ("dev", "frozen_test")


def _log(message: str) -> None:
    print(message, flush=True)


def _refuse_default_dir(ap: argparse.ArgumentParser, out_dir: str) -> None:
    if paths.is_default_dir(os.path.abspath(out_dir)):
        ap.error("refusing to write into the default artifacts directory; use a dedicated directory")


def _lot_sets(wfs_path: str, base_meta: dict) -> dict:
    """``{"source", "kept", "excluded"}`` lots of the nomenclature records the base index was built from."""
    recorded = (base_meta.get("sources", {}).get("wfs_clean") or {}).get("sha256")
    if recorded and paths.sha256_file(wfs_path) != recorded:
        raise ValueError("the --wfs-clean file is not the one the base index was built from (sources.wfs_clean)")
    wfs = pd.read_parquet(wfs_path)
    tiers = tuple(base_meta.get("tiers") or index_union.TIERS)
    _log(f"classifying {len(wfs)} nomenclature records for the kept-lot set (tiers {','.join(tiers)})")
    with pre_normalize_separation(bool(base_meta.get(paths.PRE_NORMALIZE_KEY, False))):
        build = index_union.build_nomenclatura(wfs, tiers=tiers, log=_log)
    return build.lot_sets


def _gold(splits_dir: str | None) -> dict | None:
    if not splits_dir:
        return None
    gold = {}
    for name in ORACLE_SPLITS:
        path = os.path.join(splits_dir, f"{name}.parquet")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input not found: {path}")
        gold[name] = pd.read_parquet(path, columns=["gt_doc_id", "gt_predial", "gt_quality"])
    return gold


def _cmd_add(args, ap: argparse.ArgumentParser) -> int:
    _refuse_default_dir(ap, args.out_dir)
    base_meta = validate_index_dir(args.base_index)
    lot_sets = None
    if not args.no_kept_lots:
        if not os.path.isfile(args.wfs_clean):
            raise FileNotFoundError(
                f"wfs file not found: {args.wfs_clean} (pass --wfs-clean, or --no-kept-lots to cover only the docs and units)")
        lot_sets = _lot_sets(args.wfs_clean, base_meta)
    meta = index_fallback.add_fallback(
        args.base_index, args.out_dir, catastro_docs_path=args.catastro_docs, own_centroids_path=args.own_centroids,
        pairs_path=args.pairs or None, lot_sets=lot_sets, gold=_gold(args.splits_dir), log=_log)
    block = meta["fallback"]
    _log(f"fallback index built: {block['n_docs_added']} docs added after {block['first_doc_id']} base docs, "
         f"{block['n_collisions_skipped']} doc_key collisions skipped -> {args.out_dir}")
    return 0


def _cmd_remove(args, ap: argparse.ArgumentParser) -> int:
    _refuse_default_dir(ap, args.out_dir)
    index_fallback.remove_fallback(args.index_dir, args.out_dir, log=_log)
    _log(f"fallback removed: base files restored byte for byte -> {args.out_dir}")
    return 0


def _cmd_experiment(args, ap: argparse.ArgumentParser) -> int:
    _refuse_default_dir(ap, args.out_dir)
    payload = index_fallback.build_fallback_experiment(
        args.index_dir, args.model_dir, args.out_dir, tag=args.tag, device=args.device, log=_log)
    _log(f"created {args.out_dir} ({payload['tag']}): {payload['fallback']['n_docs_added']} docs embedded, "
         f"{payload['inherited_prefix_rows']} base rows inherited")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="build the fallback index from a base index")
    add.add_argument("--base-index", required=True, help="complete base index directory (never modified)")
    add.add_argument("--catastro-docs", default=os.path.join(ARTIFACTS, "catastro_docs.parquet"))
    add.add_argument("--own-centroids", default=os.path.join(ARTIFACTS, paths.OWN_CENTROIDS_FILENAME))
    add.add_argument("--pairs", default=os.path.join(ARTIFACTS, "splits", "train_pairs.parquet"),
                     help="real training pairs the base was built from (remapped onto the appended docs); '' keeps the base file")
    add.add_argument("--wfs-clean", default=os.path.join(ARTIFACTS, "nomenclatura_clean.parquet"),
                     help="cleaned nomenclature the base was built from: gives the lots of every kept record")
    add.add_argument("--no-kept-lots", action="store_true",
                     help="skip the nomenclature file: only the lots of the base docs and units count as covered")
    add.add_argument("--splits-dir", default=None, help="dev.parquet / frozen_test.parquet: recompute the lot-level oracle "
                                                          "ceiling (counts only; no model is scored)")
    add.add_argument("--out-dir", required=True, help="new index directory (must be empty or absent)")

    exp = sub.add_parser("experiment", help="build the experiment directory (embeddings) of a fallback index")
    exp.add_argument("--index-dir", required=True, help="fallback index built by `add` (never modified)")
    exp.add_argument("--model-dir", required=True, help="base experiment holding model.pt, tuning.json, catastro_emb.pt")
    exp.add_argument("--out-dir", required=True, help="new experiment directory (must not exist)")
    exp.add_argument("--tag", default=None)
    exp.add_argument("--device", default="cpu")

    rem = sub.add_parser("remove", help="slice the appended docs off an index or experiment directory")
    rem.add_argument("--index-dir", required=True, help="fallback index or experiment directory (never modified)")
    rem.add_argument("--out-dir", required=True, help="new directory with the base files (must be empty or absent)")

    args = ap.parse_args(argv)
    handler = {"add": _cmd_add, "experiment": _cmd_experiment, "remove": _cmd_remove}[args.command]
    try:
        return handler(args, ap)
    except (OSError, ValueError, IndexValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
