"""Build a retrieval index from the official WFS nomenclature ONLY (index v2c / v2d).

Inputs: the cleaned WFS nomenclature (``clean_nomenclatura.py``), the old cadastral documents (READ-ONLY: audit
columns, remapping of the real training pairs, oracle gate, coverage figures) and the real training pairs.
The cadastre is never a source of documents, NPN, n_predios, manzana or centroids. Output directory (never
``artifacts/`` itself, so production is not touched):

    catastro_docs.parquet            the documents (production columns first, new columns after)
    catastro_own_centroids.parquet   point of the row each document reports, same row order
    spatial_negatives.npy            hard negatives (dataset.build_spatial_negatives)
    train_data.npz                   augmented queries + official strings (q_source 0/1) + document tokens
    train_pairs.parquet              the real pairs remapped to the new doc ids
    nomenclatura_units.parquet       rows that carry an interior (AP 302...), keyed to their base document
    index_dropped.parquet            quarantined records with the reason
    index_meta.json                  written LAST: the completion marker (counts, accounting, sha256...)

``--tiers cert`` builds index v2c, ``cert,pcer`` index v2d (priority cert > pcer on a shared address). The
``catastro`` tier no longer exists and is rejected. Everything runs with the letter separation of
``parser._pre_normalize`` switched on: the artifact declares ``pre_normalize_separates_letters: true``.

Exit codes:
  0  index built
  1  unreadable input / missing column
  2  usage error (also: refusing ``artifacts/`` or a non-empty directory) or a gate failed (parse-failure
     rate >= 1 %, accounting mismatch, dropped share > 0.5 %, lot-level oracle ceiling more than 2 points
     below production); a gate failure writes only ``index_dropped.parquet`` and ``index_build_failed.json``,
     never the marker
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import dataset, index_union, paths  # noqa: E402
from cali_address.parser import pre_normalize_separation  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
ALL_TIERS = ("cert", "pcer")
MAX_PARSE_FAILURE_RATE = 0.01     # >= this share of nomenclature records ending as fallback docs fails the gate
MAX_DROPPED_SHARE = 0.005         # > this share of dropped records fails the gate
ORACLE_SPLITS = ("dev", "frozen_test")
FAILED_NAME = "index_build_failed.json"
DROPPED_NAME = "index_dropped.parquet"
#: Files written by this script, in writing order (the marker comes after all of them).
OUTPUT_FILES = (
    "catastro_docs.parquet", "catastro_own_centroids.parquet", "spatial_negatives.npy", "train_data.npz",
    "train_pairs.parquet", "nomenclatura_units.parquet", DROPPED_NAME,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_npy(array: np.ndarray, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def _write_npz(arrays: dict, path: str) -> None:
    """Compressed ``.npz`` with fixed member timestamps and order, so equal arrays give equal bytes."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arrays[name]), allow_pickle=True)
    os.replace(tmp, path)


def _write_json(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _parse_tiers(raw: str, ap: argparse.ArgumentParser) -> tuple[str, ...]:
    chosen = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if "catastro" in chosen:
        ap.error("the 'catastro' tier was removed: the index is built from the WFS nomenclature only "
                 f"(use --tiers {','.join(ALL_TIERS)} or a subset)")
    unknown = sorted(set(chosen) - set(ALL_TIERS))
    if not chosen or unknown:
        ap.error(f"--tiers must be a comma-separated subset of {','.join(ALL_TIERS)} "
                 f"(got {raw!r}{'; unknown: ' + ','.join(unknown) if unknown else ''})")
    return tuple(t for t in ALL_TIERS if t in chosen)


def _clear_previous_index(out_dir: str) -> None:
    """Remove the marker FIRST, then every index file of a previous build (and nothing else)."""
    for name in (paths.INDEX_MARKER, FAILED_NAME):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    for name in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, name)
        known = name in OUTPUT_FILES or any(fnmatch.fnmatchcase(name, p) for p in paths.INDEX_FILES) or name.endswith(".tmp")
        if known and os.path.isfile(full):
            os.remove(full)


def _read_inputs(args) -> dict | str:
    """The input frames, or an error message."""
    frames = {}
    for key, path in (("wfs", args.wfs_clean), ("catastro_docs", args.catastro_docs), ("pairs", args.pairs)):
        if not os.path.isfile(path):
            return f"input not found: {path}"
        try:
            frames[key] = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - pyarrow raises several types
            return f"cannot read {path}: {exc}"
    missing = [c for c in index_union.REQUIRED_COLUMNS if c not in frames["wfs"].columns]
    if missing:
        return f"{args.wfs_clean} lacks required column(s): {', '.join(missing)}"
    for key, needed in (("catastro_docs", ("direccion", "numero_predial_nacional")),
                        ("pairs", ("gt_doc_id", "gt_predial"))):
        lacking = [c for c in needed if c not in frames[key].columns]
        if lacking:
            return f"the {key} table lacks column(s): {', '.join(lacking)}"
    gold = {}
    if args.splits_dir:
        for name in ORACLE_SPLITS:
            path = os.path.join(args.splits_dir, f"{name}.parquet")
            if not os.path.isfile(path):
                return f"input not found: {path}"
            try:
                gold[name] = pd.read_parquet(path, columns=["gt_doc_id", "gt_predial", "gt_quality"])
            except Exception as exc:  # noqa: BLE001
                return f"cannot read {path}: {exc}"
    frames["gold"] = gold
    frames["cleaning"] = {}
    if args.cleaning_report and os.path.isfile(args.cleaning_report):
        try:
            with open(args.cleaning_report, encoding="utf-8") as fh:
                frames["cleaning"] = json.load(fh)
        except (OSError, ValueError) as exc:
            return f"cannot read {args.cleaning_report}: {exc}"
    return frames


def _cleaning_counts(report: dict) -> tuple[int, int]:
    """``(quarantined rows, excluded lote/prov rows)`` of the cleaning report (zeros when absent)."""
    literals = report.get("status_invalid_literals") or {}
    lote_prov = sum(int(v) for k, v in literals.items() if str(k).lower() in ("lote", "prov"))
    return int(report.get("quarantined_rows") or 0), lote_prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wfs-clean", default=os.path.join(ARTIFACTS, "nomenclatura_clean.parquet"))
    ap.add_argument("--catastro-docs", default=os.path.join(ARTIFACTS, "catastro_docs.parquet"),
                    help="old cadastral documents, READ-ONLY (audit, pair remap, oracle gate, coverage)")
    ap.add_argument("--pairs", default=os.path.join(ARTIFACTS, "splits", "train_pairs.parquet"))
    ap.add_argument("--splits-dir", default=None,
                    help="directory with dev.parquet / frozen_test.parquet: enables the lot-level oracle-ceiling gate")
    ap.add_argument("--cleaning-report", default=os.path.join(ARTIFACTS, "nomenclatura_cleaning_report.json"),
                    help="cleaning report of clean_nomenclatura.py (quarantine counts for the accounting)")
    ap.add_argument("--tiers", default=",".join(ALL_TIERS), help="cert (v2c) or cert,pcer (v2d)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true", help="rebuild into a non-empty directory (its old index files are removed)")
    ap.add_argument("--n-variants", type=int, default=dataset.N_VARIANTS)
    ap.add_argument("--seed", type=int, default=dataset.SEED)
    args = ap.parse_args(argv)

    tiers = _parse_tiers(args.tiers, ap)
    out_dir = os.path.abspath(args.out_dir)
    if paths.is_default_dir(out_dir):
        ap.error("refusing to write into the default artifacts directory; use a dedicated directory such as "
                 "artifacts/index_v2c_cert_r2 (production artifacts are never overwritten by an index build)")
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.overwrite:
        ap.error(f"{out_dir} is not empty; pass --overwrite to rebuild it (index files only are replaced)")

    frames = _read_inputs(args)
    if isinstance(frames, str):
        print(f"ERROR: {frames}", file=sys.stderr)
        return 1

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    _clear_previous_index(out_dir)          # from here a crash can never leave a stale completion marker

    with pre_normalize_separation(True):
        _log(f"[1/7] keying {len(frames['catastro_docs'])} old cadastral documents (audit only)")
        cat = index_union.catastro_key_table(frames["catastro_docs"])
        _log(f"[2/7] ingesting {len(frames['wfs'])} nomenclature records, tiers={','.join(tiers)}")
        build = index_union.build_nomenclatura(frames["wfs"], tiers=tiers, log=_log)
        _log("[3/7] documents and gates")
        docs = index_union.union_docs(build.docs, cat)

        n_input = len(frames["wfs"])
        quarantined, lote_prov = _cleaning_counts(frames["cleaning"])
        gates_failed: list[str] = []
        acc = None
        try:
            acc = index_union.accounting(n_input, build.buckets, quarantined=quarantined, excluded_lote_prov=lote_prov)
        except index_union.AccountingError as exc:
            gates_failed.append("accounting")
            print(f"GATE accounting: {exc}", file=sys.stderr)
        selected = build.stats["selected_rows"]
        parse_rate = build.stats["parse_failure_rate"]
        dropped_share = (build.buckets["dropped"] / n_input) if n_input else 0.0
        if selected and parse_rate >= MAX_PARSE_FAILURE_RATE:
            gates_failed.append("parse_failure_rate")
            print(f"GATE parse_failure_rate {parse_rate:.4%} >= {MAX_PARSE_FAILURE_RATE:.0%}", file=sys.stderr)
        if dropped_share > MAX_DROPPED_SHARE:
            gates_failed.append("dropped_share")
            print(f"GATE dropped_share {dropped_share:.4%} > {MAX_DROPPED_SHARE:.1%}", file=sys.stderr)
        if not len(docs):
            gates_failed.append("no_documents")
            print("GATE no_documents: the selected tiers build no document", file=sys.stderr)
        oracle = None
        if frames["gold"] and len(docs):
            oracle = index_union.oracle_ceiling(frames["gold"], frames["catastro_docs"], docs, build.lot_sets)
            gate = index_union.oracle_ceiling_gate(oracle)
            oracle = {"gate": gate, **oracle}
            print("oracle ceiling (counts): " + json.dumps({k: v for k, v in oracle.items() if k != "gate"}), flush=True)
            if not gate["ok"]:
                gates_failed.append("oracle_ceiling")
                print(f"GATE oracle_ceiling: construction loss above {gate['max_drop_points']} points "
                      "(conditional on lots present in the source)", file=sys.stderr)
        _write_parquet(build.dropped, os.path.join(out_dir, DROPPED_NAME))
        if gates_failed:
            _write_json({"gates_failed": gates_failed, "buckets": build.buckets, "stats": build.stats,
                         "parse_failure_rate": parse_rate, "dropped_share": dropped_share, "tiers": list(tiers),
                         "oracle_ceiling": oracle}, os.path.join(out_dir, FAILED_NAME))
            return 2

        _log(f"[4/7] own centroids, units, official queries ({len(docs)} documents)")
        own = index_union.union_own_centroids(build.docs)
        units = index_union.assign_unit_doc_ids(build.units, docs)
        queries = index_union.official_queries(build.rows, docs)
        pairs, pair_report = index_union.remap_train_pairs(frames["pairs"], frames["catastro_docs"], docs)
        pair_report["agreement_before"] = index_union.pair_npn_agreement(frames["pairs"], frames["catastro_docs"])
        pair_report["agreement_after"] = index_union.pair_npn_agreement(pairs, docs)

        _log("[5/7] spatial negatives")
        negatives = dataset.build_spatial_negatives(docs[index_union.COMPAT_COLUMNS])
        _log(f"[6/7] training arrays ({len(queries)} official queries)")
        arrays = dataset.build_training_arrays(
            docs[index_union.COMPAT_COLUMNS], n_variants=args.n_variants, seed=args.seed, extra_queries=queries)

    _log("[7/7] writing files")
    _write_parquet(docs, os.path.join(out_dir, "catastro_docs.parquet"))
    _write_parquet(own, os.path.join(out_dir, "catastro_own_centroids.parquet"))
    _write_npy(negatives, os.path.join(out_dir, "spatial_negatives.npy"))
    _write_npz(arrays, os.path.join(out_dir, "train_data.npz"))
    _write_parquet(pairs, os.path.join(out_dir, "train_pairs.parquet"))
    _write_parquet(units, os.path.join(out_dir, "nomenclatura_units.parquet"))

    n_official = int(arrays["q_source"].sum())
    per_doc = pd.Series([d for _, d in queries], dtype="int64").value_counts() if queries else pd.Series(dtype="int64")
    n_queries = int(len(arrays["q_source"]))
    pcer_block = None
    if "pcer" in tiers:
        winner = dict(zip(build.docs["doc_key"], build.docs["estado_nomenclatura"]))
        pcer_rows = build.rows[build.rows["estado"] == "pcer"]
        pcer_block = {
            "records": int(len(pcer_rows)),
            "rows_with_key_in_cert": int(sum(winner.get(k) == "cert" for k in pcer_rows["doc_key"])),
            "docs_added": int((build.docs["estado_nomenclatura"] == "pcer").sum()),
        }
    meta = {
        "schema": 2,
        "n_docs": int(len(docs)),
        "tiers": list(tiers),
        "counts_by_fuente": {k: int(v) for k, v in docs["fuente"].value_counts().sort_index().items()},
        "counts_by_estado": {k: int(v) for k, v in docs["estado_nomenclatura"].dropna().value_counts().sort_index().items()},
        "accounting": acc,
        "interior_strip_method": build.stats["interior_strip_method"],
        "npn_choice": build.stats["npn_choice"],
        "disagreements": build.stats["disagreements"],
        "address_key_collisions": build.stats["address_key_collisions"],
        "docs_from_units_only": build.stats["docs_from_units_only"],
        "spread": build.stats["spread"],
        "lots": {"docs_n_predios_gt1": int((docs["n_predios"] > 1).sum()),
                 "docs_lot_conflict": int(docs["lot_conflict"].sum())},
        "parse_failure_rate": parse_rate,
        "fallback_doc_share_of_docs": (float((~docs["parse_ok"]).sum() / len(docs)) if len(docs) else 0.0),
        "queries": {"synthetic": n_queries - n_official, "official": n_official,
                    "official_requested": int(len(queries)),
                    "official_share": (n_official / n_queries) if n_queries else 0.0,
                    "official_max_per_doc": int(per_doc.max()) if len(per_doc) else 0,
                    "official_cap_per_doc": index_union.OFFICIAL_CAP_PER_DOC},
        "units_rows": int(len(units)),
        "train_pairs": pair_report,
        "pcer": pcer_block,
        "catastro": {"input_docs": int(len(cat)), "role": "read-only audit (never a source of documents)",
                     "coverage": index_union.catastro_coverage(docs, cat)},
        "oracle_ceiling": oracle if oracle is not None else {"skipped": "no --splits-dir given"},
        "objectid_unique": bool(docs["objectid"].is_unique),
        "objectid_offset": index_union.NOMENCLATURE_OBJECTID_OFFSET,
        "own_centroids_complete": bool(len(own) == len(docs) and own[["own_lon", "own_lat"]].notna().all().all()),
        "crs": {
            "source": "EPSG:6249 (MAGNA-SIRGAS Cali urban grid): WFS ndcoordx / ndcoordy",
            "index_coordinates": "EPSG:4326, local pyproj reprojection of ndcoordx / ndcoordy (the WFS lon/lat are audit only)",
            "metric": "EPSG:3116 through geo.to_metric (x_m / y_m)",
        },
        "pre_normalize_separates_letters": True,
        "code_fingerprint": paths.code_fingerprint(),
        "files": {name: {"sha256": paths.sha256_file(os.path.join(out_dir, name)),
                         "bytes": os.path.getsize(os.path.join(out_dir, name))}
                  for name in OUTPUT_FILES},
        "sources": {key: {"path": os.path.abspath(path), "sha256": paths.sha256_file(path)}
                    for key, path in (("wfs_clean", args.wfs_clean), ("catastro_docs", args.catastro_docs),
                                      ("pairs", args.pairs))},
        "seed": args.seed, "n_variants": args.n_variants,
        "seconds": round(time.time() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(meta, os.path.join(out_dir, paths.INDEX_MARKER))     # the marker: written last
    _log(f"index built: {meta['n_docs']} docs {meta['counts_by_fuente']} in {meta['seconds']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
