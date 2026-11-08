"""Cadastre fallback tier: a REVERSIBLE last-resort addition to a nomenclature index.

The nomenclature index (``index_union``) is built from the official WFS nomenclature only. A small share of real
lots exists in the cadastre but in none of the nomenclature records, so no query can ever reach them. This module
appends the cadastre documents of exactly those lots AFTER all base documents:

* **Rule** (deterministic, no tunable parameter): a cadastre doc is added iff its valid lot (``NPN[:21]``) is absent
  from the base index (its docs, its units sidecar and the lots of every kept nomenclature record, so a lot that
  exists but was not chosen is NOT re-added) and its ``doc_key`` collides with no existing doc.
* **Append only**: the base documents keep their ``doc_id`` and order, so the base embeddings stay valid as the first
  ``N`` rows of the new matrix and the operation can be undone by slicing the tail (see :func:`remove_fallback`).
* ``fuente = "catastro"``, ``estado_nomenclatura`` null; ``objectid = source objectid + CATASTRO_OBJECTID_OFFSET`` (the
  original id is kept in ``source_objectid``); the nomenclature-only columns take neutral values (see
  :func:`fallback_docs`).

This file holds the pure parts (selection, frames); the file-level operations are at the bottom.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from . import index_union
from .decision import CATASTRO_FUENTE
from .experiment import validate_index_dir
from .index_union import COMPAT_COLUMNS, EXTRA_COLUMNS, OWN_CENTROID_COLUMNS, _ensure_keyed, _s
from .paths import INDEX_MARKER, code_fingerprint, sha256_file

__all__ = [
    "CATASTRO_FUENTE", "CATASTRO_OBJECTID_OFFSET", "EMBEDDINGS_NAME", "RULE", "add_fallback", "build_fallback_experiment", "fallback_docs",
    "extra_pairs", "fallback_own_centroids", "oracle_not_worse", "present_lots", "remove_fallback", "select_fallback", "valid_lot",
]

EMBEDDINGS_NAME = "catastro_emb.pt"

#: Added to the cadastre ``objectid`` of an appended doc. Nomenclature docs use ``gid + 10_000_000``; the production
#: cadastre ids are below 1,000,000, so both ranges were already disjoint; the offset keeps the three origins apart
#: for good (an appended doc is recognisable by ``objectid >= 20_000_000``).
CATASTRO_OBJECTID_OFFSET = 20_000_000
#: Human-readable rule, stored verbatim in ``index_meta.json`` (``fallback.rule``).
RULE = ("add the cadastre doc iff its valid lot (NPN[:21], 21 digits, prefix 76001) is absent from the base index "
        "(docs, units sidecar, kept nomenclature records) and its doc_key collides with no existing doc")

_LOT_LEN = 21
_NPN_PREFIX = "76001"
_NEUTRAL = {"n_units": 0, "spread_m": 0.0, "n_lots_cert": 0, "n_lots_pcer": 0, "lot_conflict": False,
            "npn_choice": "catastro"}


def valid_lot(npn: object) -> str:
    """``NPN[:21]`` of a well-formed cadastre code (21+ digits starting with ``76001``), else ``""``."""
    text = _s(npn)
    lot = text[:_LOT_LEN]
    return lot if len(lot) == _LOT_LEN and lot.isdigit() and lot.startswith(_NPN_PREFIX) else ""


def present_lots(docs: pd.DataFrame, units: pd.DataFrame | None = None, extra=None) -> set:
    """Lots the nomenclature already covers: the docs' NPN lots, the units' NPN lots and ``extra`` (verbatim)."""
    lots: set = set()
    for frame in (docs, units):
        if frame is not None and len(frame):
            lots.update(valid_lot(v) for v in frame["numero_predial_nacional"].tolist())
    lots.discard("")
    if extra:
        lots.update(extra)
    return lots


def select_fallback(cat: pd.DataFrame, lots_present: set, existing_keys: set) -> tuple[np.ndarray, dict]:
    """``(mask, stats)``: which rows of the keyed cadastre docs ``cat`` are appended (see the module docstring).

    The lot test is against ``lots_present`` only (it never grows while selecting: several cadastre addresses of one
    absent lot are all added). ``doc_key`` collisions are checked against ``existing_keys`` and against the rows
    already selected (the first one wins). Counts only: ``candidates``, ``selected``, ``skipped_invalid_lot``,
    ``skipped_lot_present``, ``collisions_skipped``.
    """
    cat = _ensure_keyed(cat)
    mask = np.zeros(len(cat), dtype=bool)
    stats = {"candidates": int(len(cat)), "selected": 0, "skipped_invalid_lot": 0, "skipped_lot_present": 0,
             "collisions_skipped": 0}
    taken = set(existing_keys)
    for pos, (npn, key) in enumerate(zip(cat["numero_predial_nacional"].tolist(), cat["doc_key"].tolist())):
        lot = valid_lot(npn)
        if not lot:
            stats["skipped_invalid_lot"] += 1
        elif lot in lots_present:
            stats["skipped_lot_present"] += 1
        elif not key or key in taken:
            stats["collisions_skipped"] += 1
        else:
            taken.add(key)
            mask[pos] = True
    stats["selected"] = int(mask.sum())
    return mask, stats


def fallback_docs(cat_selected: pd.DataFrame, base_docs: pd.DataFrame, first_doc_id: int) -> pd.DataFrame:
    """The appended documents in the column order and dtypes of ``base_docs``.

    The cadastre text and codes are kept verbatim (``direccion``, NPN, manzana, comuna, barrio, centroid,
    ``n_predios``, ``manzana_nunique``, ``x_m``, ``y_m``, ``doc_key``, ``address_key``, ``parse_ok``). Neutral values
    for what only a nomenclature doc has: ``estado_nomenclatura`` / ``npn_catastro`` / ``direccion_catastro`` / ``zona``
    null, ``n_units`` 0 (no unit rows), ``spread_m`` 0.0, ``n_lots_cert`` / ``n_lots_pcer`` 0, ``lot_conflict`` False,
    ``npn_choice`` ``"catastro"``. ``doc_id`` continues from ``first_doc_id``.
    """
    missing = [c for c in COMPAT_COLUMNS + EXTRA_COLUMNS if c not in base_docs.columns]
    if missing:
        raise ValueError(f"the base docs lack the nomenclature index columns {missing}: not a nomenclature index")
    cat = _ensure_keyed(cat_selected)
    n = len(cat)
    values: dict = {}
    for col in COMPAT_COLUMNS + EXTRA_COLUMNS:
        if col in ("objectid",):
            values[col] = cat["objectid"].to_numpy(dtype="int64") + CATASTRO_OBJECTID_OFFSET
        elif col == "doc_id":
            values[col] = np.arange(first_doc_id, first_doc_id + n, dtype="int64")
        elif col == "fuente":
            values[col] = [CATASTRO_FUENTE] * n
        elif col == "source_objectid":
            values[col] = cat["objectid"].to_numpy(dtype="int64")
        elif col in _NEUTRAL:
            values[col] = [_NEUTRAL[col]] * n
        elif col in cat.columns:
            values[col] = cat[col].tolist()
        else:                                           # estado_nomenclatura, npn_catastro, direccion_catastro, zona
            values[col] = [None] * n
    out = pd.DataFrame(values, columns=COMPAT_COLUMNS + EXTRA_COLUMNS)
    for col in out.columns:
        out[col] = out[col].astype(base_docs[col].dtype)
    return out


def fallback_own_centroids(cat_selected: pd.DataFrame, own: pd.DataFrame) -> pd.DataFrame:
    """Own-centroid sidecar rows of the appended docs, in their order.

    The point comes from the cadastre sidecar ``own`` by (source ``objectid``, NPN); a row whose NPN disagrees or that
    is absent gets NaN (never a wrong point). The ``objectid`` is the offset one, as in the docs.
    """
    cat = cat_selected.reset_index(drop=True)
    lookup: dict = {}
    for oid, npn, lon, lat in zip(own["objectid"].tolist(), own["numero_predial_nacional"].tolist(),
                                  own["own_lon"].tolist(), own["own_lat"].tolist()):
        lookup.setdefault((int(oid), _s(npn)), (lon, lat))
    lons, lats = [], []
    for oid, npn in zip(cat["objectid"].tolist(), cat["numero_predial_nacional"].tolist()):
        lon, lat = lookup.get((int(oid), _s(npn)), (np.nan, np.nan))
        lons.append(lon)
        lats.append(lat)
    return pd.DataFrame({
        "objectid": cat["objectid"].to_numpy(dtype="int64") + CATASTRO_OBJECTID_OFFSET,
        "numero_predial_nacional": cat["numero_predial_nacional"].astype("str").tolist(),
        "own_lon": np.asarray(lons, dtype="float64"), "own_lat": np.asarray(lats, dtype="float64"),
    }, columns=OWN_CENTROID_COLUMNS).astype({"numero_predial_nacional": "str"})


# ---------------------------------------------------------------------------------------------------------
# training pairs
# ---------------------------------------------------------------------------------------------------------

def extra_pairs(pairs_in: pd.DataFrame, cat_old: pd.DataFrame, base_docs: pd.DataFrame,
                added_docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pairs the base index dropped that an appended doc can now take, keyed by the ground-truth LOT.

    The base pairs are never touched (a pair the base kept stays on its base doc, even when an appended doc has the key
    of its old cadastre doc: its ``gt_predial`` is the truth, not the old doc). A pair the base dropped goes to the
    appended doc with the same lot as its ``gt_predial`` (``valid_lot``): the only one; else the one with the exact NPN;
    else the one whose ``doc_key`` is the old cadastre doc's; else it is dropped as ambiguous. Returns
    ``(rows with the new gt_doc_id, report)``; counts only.
    """
    report = {"candidates": 0, "added": 0, "dropped_no_lot": 0, "dropped_ambiguous": 0}
    tagged = pairs_in.reset_index(drop=True).copy()
    tagged["__pos"] = np.arange(len(tagged), dtype="int64")
    kept, _ = index_union.remap_train_pairs(tagged, cat_old, base_docs)
    dropped = tagged[~tagged["__pos"].isin(set(kept["__pos"].tolist()))]
    report["candidates"] = int(len(dropped))
    npns = [_s(v) for v in added_docs["numero_predial_nacional"].tolist()]
    ids = [int(v) for v in added_docs["doc_id"].tolist()]
    by_lot: dict = {}
    by_key: dict = {}
    for pos, (npn, key) in enumerate(zip(npns, added_docs["doc_key"].tolist())):
        by_lot.setdefault(valid_lot(npn), []).append(pos)
        by_key.setdefault(key, pos)
    gid = pd.to_numeric(dropped["gt_doc_id"], errors="coerce").tolist() if "gt_doc_id" in dropped.columns else [np.nan] * len(dropped)
    old_keys = index_union._old_keys(cat_old, index_union._wanted_ids(gid))
    picked_rows, picked_ids = [], []
    for pos, (old, predial) in enumerate(zip(gid, dropped["gt_predial"].tolist() if "gt_predial" in dropped.columns else [None] * len(dropped))):
        predial = _s(predial)
        lot = valid_lot(predial)
        cands = by_lot.get(lot) if lot else None
        if not cands:
            report["dropped_no_lot"] += 1
            continue
        target = None
        if len(cands) == 1:
            target = cands[0]
        else:
            exact = [c for c in cands if npns[c] == predial]
            if len(exact) == 1:
                target = exact[0]
            else:
                by_old_key = by_key.get(old_keys.get(int(old))) if old == old and int(old) in old_keys else None
                if by_old_key in cands:
                    target = by_old_key
        if target is None:
            report["dropped_ambiguous"] += 1
            continue
        picked_rows.append(pos)
        picked_ids.append(ids[target])
    out = dropped.iloc[picked_rows].drop(columns=["__pos"]).reset_index(drop=True).copy()
    if len(out):
        out["gt_doc_id"] = np.asarray(picked_ids, dtype="int64")
    report["added"] = int(len(out))
    return out, report


# ---------------------------------------------------------------------------------------------------------
# oracle comparison
# ---------------------------------------------------------------------------------------------------------

def oracle_not_worse(base: dict | None, new: dict) -> list[str]:
    """Names (``total``, ``dev``, ...) whose ``new_lot_matched`` fell below the base's; ``[]`` when none or nothing to compare."""
    failed = []
    for name, base_block in (base or {}).items():
        if not (isinstance(base_block, dict) and "new_lot_matched" in base_block):
            continue
        block = new.get(name)
        if isinstance(block, dict) and block.get("new_lot_matched", 0) < base_block["new_lot_matched"]:
            failed.append(name)
    return failed


# ---------------------------------------------------------------------------------------------------------
# file level: add / remove
# ---------------------------------------------------------------------------------------------------------

#: Files the fallback rewrites (docs and the two row-aligned tables); every other listed file is linked unchanged.
_REWRITTEN = ("catastro_docs.parquet", "catastro_own_centroids.parquet", "train_pairs.parquet")


def _say(log):
    return log if log is not None else (lambda _message: None)


def _write_parquet(frame: pd.DataFrame, path: str) -> None:
    frame.to_parquet(path, index=False)


def _write_json(payload: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)


def _place(src: str, dst: str) -> None:
    """Hardlink ``src`` at ``dst`` (copy where links are unavailable); the source is never opened for writing."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def save_embeddings(tensor, out_dir: str) -> str:
    """``torch.save`` the matrix as ``out_dir/catastro_emb.pt``.

    The archive root inside a ``.pt`` file is the file NAME, so the matrix is saved under its final name (in a scratch
    sub-directory, then moved): saving ``x.pt`` and renaming gives different bytes than saving ``catastro_emb.pt``.
    With the same name and the same tensor the bytes are identical, which is what makes the removal byte-exact.
    """
    import torch

    scratch = tempfile.mkdtemp(prefix=".emb_", dir=out_dir)
    try:
        tmp = os.path.join(scratch, EMBEDDINGS_NAME)
        torch.save(tensor.clone().contiguous(), tmp)
        final = os.path.join(out_dir, EMBEDDINGS_NAME)
        os.replace(tmp, final)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return final


def _check_columns(frame: pd.DataFrame, needed, label: str) -> None:
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"the {label} lack(s) column(s) {', '.join(missing)}")


def _prepare_out_dir(out_dir: str) -> bool:
    """Create ``out_dir``; ``FileExistsError`` for a non-empty one. Returns whether this call created it."""
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"{out_dir} is not empty")
    created = not os.path.isdir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return created


def _cleanup(out_dir: str, created: bool) -> None:
    if created:
        shutil.rmtree(out_dir, ignore_errors=True)
    else:
        for name in os.listdir(out_dir):
            full = os.path.join(out_dir, name)
            shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) else os.remove(full)


def _manifest(directory: str, names) -> dict:
    return {name: {"sha256": sha256_file(os.path.join(directory, name)), "bytes": os.path.getsize(os.path.join(directory, name))}
            for name in names}


def add_fallback(base_dir: str, out_dir: str, *, catastro_docs_path: str, own_centroids_path: str,
                 pairs_path: str | None = None, lot_sets: dict | None = None, gold: dict | None = None,
                 log=None) -> dict:
    """Write ``out_dir`` = the base index plus the cadastre fallback docs; return the new ``index_meta`` dict.

    ``lot_sets`` (``{"source", "kept", "excluded"}`` of ``NomenclatureBuild``) adds the lots of every kept nomenclature
    record to the "already covered" set; without it only the base docs and units count (``fallback.lot_basis`` says
    which). ``pairs_path`` (the real pairs the base was built from, sha256 checked) remaps ``train_pairs`` onto the
    appended docs; without it the base file is kept. ``gold`` (``{"dev": frame, ...}``) recomputes the lot-level oracle
    ceiling; it may not fall below the base's. The base index is only read; ``out_dir`` is removed again on any error.
    Errors: ``FileExistsError`` (non-empty ``out_dir``), ``IndexValidationError`` (broken base), ``ValueError``
    (anything else: base already has a fallback, same directory, wrong inputs, remap or oracle regression).
    """
    say = _say(log)
    base_dir, out_dir = os.path.abspath(base_dir), os.path.abspath(out_dir)
    if os.path.normcase(base_dir) == os.path.normcase(out_dir):
        raise ValueError("out_dir is the base index: the fallback is written to a NEW directory")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"{out_dir} is not empty")
    base_meta = validate_index_dir(base_dir)
    if "fallback" in base_meta:
        raise ValueError(f"{base_dir} already has a fallback tier: add it to the original base index instead")
    sources = base_meta.get("sources", {})

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(base_dir, name))

    base_docs, base_own, base_pairs, units = (read(n) for n in (
        "catastro_docs.parquet", "catastro_own_centroids.parquet", "train_pairs.parquet", "nomenclatura_units.parquet"))
    _check_columns(base_docs, COMPAT_COLUMNS + EXTRA_COLUMNS, "base docs")
    if base_docs["doc_id"].tolist() != list(range(len(base_docs))):
        raise ValueError("the base doc_id is not 0..N-1 in order: cannot append")
    cat = pd.read_parquet(catastro_docs_path)
    _check_columns(cat, ("direccion", "numero_predial_nacional", "objectid"), "cadastre docs")
    cat_sha = sha256_file(catastro_docs_path)
    recorded = (sources.get("catastro_docs") or {}).get("sha256")
    if recorded and recorded != cat_sha:
        raise ValueError("the cadastre docs are not the file the base index was built against (sources.catastro_docs)")
    own_cat = pd.read_parquet(own_centroids_path)
    _check_columns(own_cat, OWN_CENTROID_COLUMNS, "cadastre own centroids")
    pairs_in = None
    if pairs_path is not None:
        pairs_sha = sha256_file(pairs_path)
        recorded_pairs = (sources.get("pairs") or {}).get("sha256")
        if recorded_pairs and recorded_pairs != pairs_sha:
            raise ValueError("the pairs file is not the one the base index was built from (sources.pairs)")
        pairs_in = pd.read_parquet(pairs_path)

    say(f"keying {len(cat)} cadastre docs")
    keyed = index_union.catastro_key_table(cat)
    covered = present_lots(base_docs, units, extra=(lot_sets or {}).get("kept"))
    mask, stats = select_fallback(keyed, covered, set(base_docs["doc_key"].tolist()))
    selected = keyed[mask].reset_index(drop=True)
    first = int(len(base_docs))
    added = fallback_docs(selected, base_docs, first)
    docs = pd.concat([base_docs, added], ignore_index=True)
    own = pd.concat([base_own, fallback_own_centroids(selected, own_cat)], ignore_index=True)
    for frame, base_frame in ((docs, base_docs), (own, base_own)):
        if frame.dtypes.astype(str).to_dict() != base_frame.dtypes.astype(str).to_dict():
            raise ValueError("appending changed a column dtype: refusing to write an inconsistent index")
    # a nomenclature index may repeat a doc_key on its own docs (groups split by lot); the appended ones must not
    if not docs["objectid"].is_unique or not added["doc_key"].is_unique or set(added["doc_key"]) & set(base_docs["doc_key"]):
        raise ValueError("the appended docs would break objectid / doc_key uniqueness")

    pair_report = copy.deepcopy(base_meta.get("train_pairs"))
    if pairs_in is not None:
        reproduced, _ = index_union.remap_train_pairs(pairs_in, cat, base_docs)
        try:
            pd.testing.assert_frame_equal(reproduced, base_pairs)
        except AssertionError as exc:
            raise ValueError(f"the pairs file does not reproduce the base index's training pairs: {exc}") from exc
        extra, extra_report = extra_pairs(pairs_in, cat, base_docs, added)
        pairs = pd.concat([base_pairs, extra], ignore_index=True)       # base pairs first: removal keeps gt_doc_id < first
        pair_report = {**(pair_report or {}), "kept": int(len(pairs)), "fallback": extra_report,
                       "agreement_after": index_union.pair_npn_agreement(pairs, docs)}
        pairs_block = {"remapped": True, **extra_report}
    else:
        pairs, pairs_block = base_pairs, {"remapped": False, "added": 0}

    oracle, oracle_vs_base = None, None
    if gold:
        result = index_union.oracle_ceiling(gold, cat, docs, lot_sets)
        gate = index_union.oracle_ceiling_gate(result)
        reference = index_union.oracle_ceiling(gold, cat, base_docs, lot_sets)
        worse = oracle_not_worse(reference, result)
        if worse:
            raise ValueError(f"the lot-level oracle ceiling got worse than the base for {worse}")
        oracle = {"gate": gate, **result}
        oracle_vs_base = {"base_new_lot_matched": int(reference["total"]["new_lot_matched"]),
                          "new_lot_matched": int(result["total"]["new_lot_matched"]),
                          "new_lot_matched_gain": int(result["total"]["new_lot_matched"] - reference["total"]["new_lot_matched"])}

    created = _prepare_out_dir(out_dir)
    try:
        say(f"writing {len(docs)} docs ({len(added)} appended)")
        _write_parquet(docs, os.path.join(out_dir, "catastro_docs.parquet"))
        _write_parquet(own, os.path.join(out_dir, "catastro_own_centroids.parquet"))
        _write_parquet(pairs, os.path.join(out_dir, "train_pairs.parquet"))
        for name in base_meta["files"]:
            if name not in _REWRITTEN:
                _place(os.path.join(base_dir, name), os.path.join(out_dir, name))
        meta = copy.deepcopy(base_meta)
        meta.update({
            "n_docs": int(len(docs)),
            "counts_by_fuente": {k: int(v) for k, v in docs["fuente"].value_counts().sort_index().items()},
            "lots": {"docs_n_predios_gt1": int((docs["n_predios"] > 1).sum()), "docs_lot_conflict": int(docs["lot_conflict"].sum())},
            "train_pairs": pair_report,
            "objectid_unique": bool(docs["objectid"].is_unique),
            "own_centroids_complete": bool(len(own) == len(docs) and own[["own_lon", "own_lat"]].notna().all().all()),
            "oracle_ceiling": oracle if oracle is not None else base_meta.get("oracle_ceiling"),
            "files": _manifest(out_dir, base_meta["files"]),
        })
        meta["sources"] = {**sources, "catastro_fallback_docs": {"path": os.path.abspath(catastro_docs_path), "sha256": cat_sha},
                           "catastro_fallback_own_centroids": {"path": os.path.abspath(own_centroids_path),
                                                               "sha256": sha256_file(own_centroids_path)}}
        meta["fallback"] = {
            "fuente": CATASTRO_FUENTE, "first_doc_id": first, "n_docs_added": int(len(added)),
            "n_collisions_skipped": int(stats["collisions_skipped"]), "rule": RULE,
            "source_sha256": cat_sha, "base_index_sha256": sha256_file(os.path.join(base_dir, INDEX_MARKER)),
            "base_index_id": os.path.basename(base_dir), "objectid_offset": CATASTRO_OBJECTID_OFFSET,
            "lot_basis": "docs+units+kept_nomenclature_records" if lot_sets else "docs+units",
            "stats": stats, "train_pairs": pairs_block,
            "oracle_ceiling_vs_base": oracle_vs_base,
            "training_data": "unchanged (train_data.npz, spatial_negatives.npy, units and dropped are the base files; "
                             "the model is not retrained)",
            "code_fingerprint": code_fingerprint(),
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "base_meta": base_meta,
        }
        _write_json(meta, os.path.join(out_dir, INDEX_MARKER))                       # the marker: written last
    except BaseException:
        _cleanup(out_dir, created)
        raise
    return meta


def remove_fallback(index_dir: str, out_dir: str, log=None) -> dict:
    """Slice the appended rows off ``index_dir`` into ``out_dir``: the result is byte-identical to the base index.

    Docs, own-centroid sidecar and (when present) the embeddings lose their tail; training pairs of appended docs are
    dropped; the files that were never rewritten are linked as they are; ``index_meta.json`` is rewritten from the copy
    of the base meta stored in ``fallback.base_meta``. Every output file is checked against the base manifest and the
    base ``index_meta`` hash; any difference removes ``out_dir`` and raises ``ValueError``. Returns the base meta.
    """
    say = _say(log)
    index_dir, out_dir = os.path.abspath(index_dir), os.path.abspath(out_dir)
    if os.path.normcase(index_dir) == os.path.normcase(out_dir):
        raise ValueError("out_dir is the same directory as the index: the removal is written to a NEW directory")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"{out_dir} is not empty")
    meta = validate_index_dir(index_dir)
    block = meta.get("fallback")
    if not isinstance(block, dict) or not isinstance(block.get("base_meta"), dict):
        raise ValueError(f"{index_dir} has no fallback block in {INDEX_MARKER}: nothing to remove")
    base_meta, first = block["base_meta"], int(block["first_doc_id"])
    docs = pd.read_parquet(os.path.join(index_dir, "catastro_docs.parquet"))
    if len(docs) != first + int(block["n_docs_added"]):
        raise ValueError("catastro_docs.parquet does not have first_doc_id + n_docs_added rows")
    created = _prepare_out_dir(out_dir)
    try:
        say(f"slicing {len(docs) - first} appended docs off")
        _write_parquet(docs.iloc[:first].reset_index(drop=True), os.path.join(out_dir, "catastro_docs.parquet"))
        own = pd.read_parquet(os.path.join(index_dir, "catastro_own_centroids.parquet"))
        _write_parquet(own.iloc[:first].reset_index(drop=True), os.path.join(out_dir, "catastro_own_centroids.parquet"))
        pairs = pd.read_parquet(os.path.join(index_dir, "train_pairs.parquet"))
        _write_parquet(pairs[pairs["gt_doc_id"] < first].reset_index(drop=True), os.path.join(out_dir, "train_pairs.parquet"))
        for name in base_meta["files"]:
            if name not in _REWRITTEN:
                _place(os.path.join(index_dir, name), os.path.join(out_dir, name))
        emb_path = os.path.join(index_dir, EMBEDDINGS_NAME)
        if os.path.isfile(emb_path):
            import torch

            matrix = torch.load(emb_path, map_location="cpu")
            if matrix.shape[0] != len(docs):
                raise ValueError(f"{EMBEDDINGS_NAME} has {matrix.shape[0]} rows but the index has {len(docs)} docs")
            save_embeddings(matrix[:first], out_dir)
        for name, info in base_meta["files"].items():
            if sha256_file(os.path.join(out_dir, name)) != info["sha256"]:
                raise ValueError(f"{name} is not byte-identical to the base index file after removing the fallback")
        _write_json(base_meta, os.path.join(out_dir, INDEX_MARKER))
        if sha256_file(os.path.join(out_dir, INDEX_MARKER)) != block["base_index_sha256"]:
            raise ValueError(f"{INDEX_MARKER} is not byte-identical to the base index marker after removing the fallback")
        _check_embeddings_hash(index_dir, out_dir)
    except BaseException:
        _cleanup(out_dir, created)
        raise
    return base_meta


#: Files taken from the model experiment unchanged.
_MODEL_FILES = ("model.pt", "tuning.json")

REVERSE_SWITCH = ("serving-time switch, no rebuild: set \"exclude_fuentes\": [\"catastro\"] in decision.json (calibrated "
                  "experiments) and/or the environment variable NORMALIZER_EXCLUDE_FUENTES=catastro; the appended docs are "
                  "then dropped from retrieval (the index is the same as without them)")


def build_fallback_experiment(index_dir: str, model_dir: str, out_dir: str, tag: str | None = None,
                              device: str = "cpu", log=None) -> dict:
    """Experiment directory = fallback index + the model of ``model_dir`` + embeddings of ALL docs. No training.

    ``catastro_emb.pt`` is the base matrix (``model_dir``, proved to be the first ``first_doc_id`` docs) concatenated with
    the embeddings of the appended docs, computed here with the same model and ``dropout=0``. The base rows are
    copied, never recomputed, and asserted bit-identical. ``model.pt`` / ``tuning.json`` are hardlinked. Nothing is
    created when a check fails (``FileExistsError``, ``FileNotFoundError``, ``ValueError``).
    """
    import torch

    from .experiment import index_meta_record, link_index_files
    from .tokenizer import encode_batch
    from .train import embed_documents, load_checkpoint

    say = _say(log)
    index_dir, model_dir, out_dir = (os.path.abspath(p) for p in (index_dir, model_dir, out_dir))
    if os.path.lexists(out_dir):
        raise FileExistsError(f"{out_dir} already exists")
    meta = validate_index_dir(index_dir)
    block = meta.get("fallback")
    if not isinstance(block, dict):
        raise ValueError(f"{index_dir} has no fallback block in {INDEX_MARKER}: build it with `index_fallback.py add` first")
    for name in (*_MODEL_FILES, EMBEDDINGS_NAME, "catastro_docs.parquet"):
        if not os.path.isfile(os.path.join(model_dir, name)):
            raise FileNotFoundError(f"{name} not found in {model_dir}")
    first = int(block["first_doc_id"])
    docs = pd.read_parquet(os.path.join(index_dir, "catastro_docs.parquet"))
    model_docs = pd.read_parquet(os.path.join(model_dir, "catastro_docs.parquet"))
    if len(model_docs) != first or not (
            model_docs["direccion"].astype(str).to_numpy() == docs["direccion"].astype(str).iloc[:first].to_numpy()).all():
        raise ValueError("the model directory's docs are not the base prefix of the fallback index (direccion text or order differs)")
    if "doc_key" in model_docs.columns and not (
            model_docs["doc_key"].astype(str).to_numpy() == docs["doc_key"].astype(str).iloc[:first].to_numpy()).all():
        raise ValueError("the model directory's docs are not the base prefix of the fallback index (doc_key differs)")
    base_emb_path = os.path.join(model_dir, EMBEDDINGS_NAME)
    base_emb = torch.load(base_emb_path, map_location="cpu")
    if base_emb.shape[0] != first:
        raise ValueError(f"{EMBEDDINGS_NAME} has {base_emb.shape[0]} rows but the base index has {first} docs")
    model, _ckpt = load_checkpoint(os.path.join(model_dir, "model.pt"), device, dropout=0.0)
    model.eval()
    added = docs.iloc[first:]
    say(f"embedding {len(added)} appended docs on {device}")
    if len(added):
        new_emb = embed_documents(model, encode_batch([str(t).upper() for t in added["direccion"]]), device=device).cpu()
        if new_emb.dtype != base_emb.dtype or new_emb.shape[1] != base_emb.shape[1]:
            raise ValueError("the new embeddings do not match the base matrix (dtype or width)")
        full = torch.cat([base_emb, new_emb])
    else:
        full = base_emb
    if not torch.equal(full[:first], base_emb):
        raise ValueError("internal error: the base embedding rows changed")

    model_meta: dict = {}
    model_json = os.path.join(model_dir, "experiment.json")
    if os.path.isfile(model_json):
        with open(model_json, "r", encoding="utf-8") as fh:
            model_meta = json.load(fh)
    model_tag = model_meta.get("tag", os.path.basename(model_dir))
    link_index_files(index_dir, out_dir, meta)
    try:
        for name in _MODEL_FILES:
            _place(os.path.join(model_dir, name), os.path.join(out_dir, name))
        save_embeddings(full, out_dir)
        payload = {
            "tag": tag or os.path.basename(out_dir),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "inherits_model_from": model_tag,
            "inherits_model_reason": (
                f"the first {first} documents of {os.path.basename(index_dir)} are the documents {model_tag} was trained and "
                f"embedded on (same text and order); the {len(added)} appended cadastre docs are embedded with the same model "
                "and the base rows are copied: model.pt and tuning.json are reused, nothing is retrained"),
            "inherited_prefix_rows": first,
            "index_meta": index_meta_record(index_dir, meta),
            "model_sha256": {name: sha256_file(os.path.join(out_dir, name)) for name in (*_MODEL_FILES[:1], EMBEDDINGS_NAME, "tuning.json")},
            "fallback": {
                "fuente": block["fuente"], "first_doc_id": first, "n_docs_added": int(block["n_docs_added"]),
                "n_collisions_skipped": int(block["n_collisions_skipped"]),
                "base_catastro_emb_sha256": sha256_file(base_emb_path), "base_rows_bit_identical": True,
                "embedding": {"device": device, "dtype": str(full.dtype), "added_rows": int(len(added)),
                              "note": "base rows copied from the base experiment; added rows computed here on CPU in fp32 "
                                      "(the base rows came from the training device), so they can differ from a GPU "
                                      "recomputation in the last fp16 digits"},
                "reverse": {
                    "base": f"the base index ({block.get('base_index_id')}) and the base experiment ({model_tag}) are never "
                            "modified: serve them directly",
                    "remove": "python scripts/index_fallback.py remove --index-dir <this directory> --out-dir <new directory>: "
                              "slices the appended rows off docs, sidecar, pairs and embeddings; the result is byte-identical to "
                              "the base files",
                    "switch": REVERSE_SWITCH,
                },
            },
        }
        with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return payload


def _check_embeddings_hash(index_dir: str, out_dir: str) -> None:
    """When the experiment recorded the base embeddings hash, the sliced matrix must reproduce it."""
    emb = os.path.join(out_dir, EMBEDDINGS_NAME)
    record = os.path.join(index_dir, "experiment.json")
    if not (os.path.isfile(emb) and os.path.isfile(record)):
        return
    with open(record, "r", encoding="utf-8") as fh:
        expected = ((json.load(fh).get("fallback") or {}).get("base_catastro_emb_sha256"))
    if expected and sha256_file(emb) != expected:
        raise ValueError(f"{EMBEDDINGS_NAME} is not byte-identical to the base embeddings after removing the fallback")
