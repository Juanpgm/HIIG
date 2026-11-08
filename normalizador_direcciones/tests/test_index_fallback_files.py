"""File-level add / remove of the cadastre fallback tier on a tiny built index (invented data, numbers 900+).

The base index is produced by the real ``scripts/build_index.py`` so the layout, dtypes and manifest are the real ones.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from test_build_index import _argv, _catastro_entries, _nom_rows, _write_inputs, _write_splits

from cali_address import index_fallback as F
from cali_address import paths
from cali_address.experiment import IndexValidationError, validate_index_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_FILES = {
    "catastro_docs.parquet", "catastro_own_centroids.parquet", "spatial_negatives.npy", "train_data.npz",
    "train_pairs.parquet", "nomenclatura_units.parquet", "index_dropped.parquet",
}
N_ADDED, N_COLLISIONS = 7, 5          # 7 cadastre docs of lot 0920 are added, the 5 of lot 0919 collide on doc_key


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _build_index():
    spec = importlib.util.spec_from_file_location("script_build_index_fb", os.path.join(ROOT, "scripts", "build_index.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """``{"base": index dir, "inputs": paths, "src": dir, "own": own-centroid parquet}`` built once for the module."""
    src = tmp_path_factory.mktemp("fb_src")
    inputs = _write_inputs(src, _nom_rows())
    base = tmp_path_factory.mktemp("fb_base") / "index_base"
    assert _build_index().main(_argv(inputs, base)) == 0
    cat = pd.read_parquet(inputs["docs"])
    own = pd.DataFrame({"objectid": cat["objectid"], "numero_predial_nacional": cat["numero_predial_nacional"],
                        "own_lon": cat["centroid_lon"] + 1e-5, "own_lat": cat["centroid_lat"] + 1e-5})
    own_path = str(src / "catastro_own_centroids.parquet")
    own.to_parquet(own_path)
    return {"base": base, "inputs": inputs, "src": src, "own": own_path}


def add(world, out, **kw):
    kw.setdefault("pairs_path", world["inputs"]["pairs"])
    return F.add_fallback(str(world["base"]), str(out), catastro_docs_path=world["inputs"]["docs"],
                          own_centroids_path=world["own"], **kw)


def meta_of(directory) -> dict:
    return json.loads((directory / paths.INDEX_MARKER).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------- add

def test_add_writes_a_complete_valid_index_with_the_marker_last(world, tmp_path):
    out = tmp_path / "with_fb"
    meta = add(world, out)
    assert BASE_FILES <= {p.name for p in out.iterdir()}
    validate_index_dir(str(out))                                  # every listed sha256 verifies
    assert set(meta["files"]) == BASE_FILES == set(meta_of(world["base"])["files"])
    latest = max(p.stat().st_mtime_ns for p in out.iterdir() if p.name != paths.INDEX_MARKER)
    assert (out / paths.INDEX_MARKER).stat().st_mtime_ns >= latest
    assert meta_of(out) == json.loads(json.dumps(meta))


def test_add_appends_after_the_base_docs_and_keeps_them_untouched(world, tmp_path):
    out = tmp_path / "with_fb"
    meta = add(world, out)
    base = pd.read_parquet(world["base"] / "catastro_docs.parquet")
    new = pd.read_parquet(out / "catastro_docs.parquet")
    n = len(base)
    assert len(new) == n + N_ADDED == meta["n_docs"]
    pd.testing.assert_frame_equal(new.iloc[:n].reset_index(drop=True), base)
    assert new["doc_id"].tolist() == list(range(n + N_ADDED))
    tail = new.iloc[n:]
    assert (tail["fuente"] == "catastro").all() and tail["estado_nomenclatura"].isna().all()
    assert (tail["objectid"] >= F.CATASTRO_OBJECTID_OFFSET).all() and new["objectid"].is_unique
    assert new["doc_key"].is_unique
    cat = pd.read_parquet(world["inputs"]["docs"])
    expected = cat[cat["numero_predial_nacional"].str[:21].str.endswith("0920")]
    assert tail["direccion"].tolist() == expected["direccion"].tolist()                  # verbatim, cadastre order
    assert tail["numero_predial_nacional"].tolist() == expected["numero_predial_nacional"].tolist()
    assert meta["counts_by_fuente"] == {"catastro": N_ADDED, **meta_of(world["base"])["counts_by_fuente"]}
    assert meta["objectid_unique"] is True


def test_add_fallback_block_and_manifest(world, tmp_path):
    out = tmp_path / "with_fb"
    meta = add(world, out)
    base_meta = meta_of(world["base"])
    block = meta["fallback"]
    assert block["fuente"] == "catastro" and block["first_doc_id"] == base_meta["n_docs"]
    assert block["n_docs_added"] == N_ADDED and block["n_collisions_skipped"] == N_COLLISIONS
    assert block["rule"] == F.RULE and block["objectid_offset"] == F.CATASTRO_OBJECTID_OFFSET
    assert block["source_sha256"] == sha(world["inputs"]["docs"])
    assert block["base_index_sha256"] == sha(world["base"] / paths.INDEX_MARKER)
    assert block["base_meta"] == base_meta
    assert block["stats"]["selected"] == N_ADDED
    for name, info in meta["files"].items():
        assert sha(out / name) == info["sha256"], name
    # the training data are NOT regenerated: byte-for-byte the base files
    for name in ("spatial_negatives.npy", "train_data.npz", "nomenclatura_units.parquet", "index_dropped.parquet"):
        assert sha(out / name) == sha(world["base"] / name), name


def test_add_never_touches_the_base_index(world, tmp_path):
    before = {p.name: sha(p) for p in world["base"].iterdir()}
    add(world, tmp_path / "with_fb")
    assert {p.name: sha(p) for p in world["base"].iterdir()} == before


def test_add_own_centroids_follow_the_docs(world, tmp_path):
    out = tmp_path / "with_fb"
    add(world, out)
    docs = pd.read_parquet(out / "catastro_docs.parquet")
    own = pd.read_parquet(out / "catastro_own_centroids.parquet")
    base_own = pd.read_parquet(world["base"] / "catastro_own_centroids.parquet")
    assert own["objectid"].tolist() == docs["objectid"].tolist()
    assert own["numero_predial_nacional"].tolist() == docs["numero_predial_nacional"].tolist()
    pd.testing.assert_frame_equal(own.iloc[:len(base_own)].reset_index(drop=True), base_own)
    assert own.iloc[len(base_own):][["own_lon", "own_lat"]].notna().all().all()


def test_add_remaps_train_pairs_to_the_appended_docs(world, tmp_path):
    out = tmp_path / "with_fb"
    meta = add(world, out)
    base_pairs = pd.read_parquet(world["base"] / "train_pairs.parquet")
    pairs = pd.read_parquet(out / "train_pairs.parquet")
    first = meta["fallback"]["first_doc_id"]
    old = pairs[pairs["gt_doc_id"] < first].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, base_pairs)                      # the base pairs are unchanged
    new = pairs[pairs["gt_doc_id"] >= first]
    assert len(new) == N_ADDED and meta["fallback"]["train_pairs"]["added"] == N_ADDED
    docs = pd.read_parquet(out / "catastro_docs.parquet")
    for gid, predial in zip(new["gt_doc_id"], new["gt_predial"]):
        assert docs["numero_predial_nacional"].iloc[int(gid)][:21] == predial[:21]


def test_add_without_pairs_keeps_the_base_pairs_file(world, tmp_path):
    out = tmp_path / "with_fb"
    meta = add(world, out, pairs_path=None)
    assert sha(out / "train_pairs.parquet") == sha(world["base"] / "train_pairs.parquet")
    assert meta["fallback"]["train_pairs"]["added"] == 0


def test_add_oracle_ceiling_improves_and_is_reported(world, tmp_path):
    split_dir = _write_splits(world["src"], _catastro_entries(), n_gold=12)
    gold = {name: pd.read_parquet(os.path.join(split_dir, f"{name}.parquet")) for name in ("dev", "frozen_test")}
    old_docs = pd.read_parquet(world["inputs"]["docs"])
    from cali_address import index_union

    base_docs = pd.read_parquet(world["base"] / "catastro_docs.parquet")
    base_oracle = index_union.oracle_ceiling(gold, old_docs, base_docs)
    out = tmp_path / "with_fb"
    meta = add(world, out, gold=gold)
    new_total = meta["oracle_ceiling"]["total"]
    assert new_total["new_lot_matched"] == base_oracle["total"]["new_lot_matched"] + 2 * N_ADDED
    gate = meta["oracle_ceiling"]["gate"]
    assert set(gate) >= {"ok", "failed", "shares"}              # reported; the refusal rule is "not worse than the base"
    assert meta["fallback"]["oracle_ceiling_vs_base"]["new_lot_matched_gain"] == 2 * N_ADDED


def test_oracle_not_worse_flags_a_regression():
    base = {"total": {"new_lot_matched": 10}, "dev": {"new_lot_matched": 6}, "frozen_test": {"new_lot_matched": 4}}
    same = {"total": {"new_lot_matched": 10}, "dev": {"new_lot_matched": 6}, "frozen_test": {"new_lot_matched": 4}}
    worse = {"total": {"new_lot_matched": 9}, "dev": {"new_lot_matched": 6}, "frozen_test": {"new_lot_matched": 3}}
    assert F.oracle_not_worse(base, same) == []
    assert F.oracle_not_worse(base, worse) == ["total", "frozen_test"]
    assert F.oracle_not_worse({"skipped": "x"}, worse) == []            # nothing to compare with


def test_add_lot_sets_keep_lots_that_exist_but_were_not_chosen_out(world, tmp_path):
    """A lot listed as kept by the nomenclature (but absent from docs / units) must NOT be re-added."""
    lot_0920 = pd.read_parquet(world["inputs"]["docs"])["numero_predial_nacional"].iloc[-1][:21]
    out = tmp_path / "with_fb"
    meta = add(world, out, lot_sets={"source": {lot_0920}, "kept": {lot_0920}, "excluded": set()})
    assert meta["n_docs"] == meta_of(world["base"])["n_docs"] and meta["fallback"]["n_docs_added"] == 0
    assert meta["fallback"]["stats"]["skipped_lot_present"] == N_ADDED
    assert meta["fallback"]["lot_basis"] == "docs+units+kept_nomenclature_records"


def test_add_with_nothing_to_add_still_writes_a_valid_identical_index(world, tmp_path):
    cat = pd.read_parquet(world["inputs"]["docs"])
    every_lot = set(cat["numero_predial_nacional"].str[:21])
    out = tmp_path / "with_fb"
    meta = add(world, out, lot_sets={"source": every_lot, "kept": every_lot, "excluded": set()})
    validate_index_dir(str(out))
    assert meta["fallback"]["n_docs_added"] == 0 and meta["fallback"]["first_doc_id"] == meta["n_docs"]
    assert sha(out / "catastro_docs.parquet") == sha(world["base"] / "catastro_docs.parquet")


def test_add_accepts_a_base_that_already_repeats_a_doc_key(world, tmp_path):
    """Real nomenclature indexes split a group by lot: the same doc_key appears on several base docs. Only the appended
    docs must be unique among themselves and against the base keys."""
    twin = tmp_path / "twin_base"
    shutil.copytree(world["base"], twin)
    docs = pd.read_parquet(twin / "catastro_docs.parquet")
    kr = docs.index[docs["direccion"].str.startswith("KR 9")][:2]          # pcer docs no cadastre doc shares a key with
    docs.loc[kr[1], "doc_key"] = docs.loc[kr[0], "doc_key"]
    os.remove(twin / "catastro_docs.parquet")
    docs.to_parquet(twin / "catastro_docs.parquet", index=False)
    meta = meta_of(twin)
    meta["files"]["catastro_docs.parquet"] = {"sha256": sha(twin / "catastro_docs.parquet"),
                                              "bytes": (twin / "catastro_docs.parquet").stat().st_size}
    (twin / paths.INDEX_MARKER).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    result = F.add_fallback(str(twin), str(out), catastro_docs_path=world["inputs"]["docs"], own_centroids_path=world["own"])
    assert result["fallback"]["n_docs_added"] == N_ADDED
    F.remove_fallback(str(out), str(tmp_path / "back"))
    assert sha(tmp_path / "back" / "catastro_docs.parquet") == sha(twin / "catastro_docs.parquet")


# --------------------------------------------------------------------------------------------------------- refusals

def test_add_refuses_an_existing_non_empty_out_dir_and_leaves_it_alone(world, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        add(world, out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_add_refuses_to_write_into_the_base_index(world):
    with pytest.raises(ValueError, match="base"):
        add(world, world["base"])


def test_add_refuses_a_base_that_already_has_a_fallback(world, tmp_path):
    first = tmp_path / "first"
    add(world, first)
    with pytest.raises(ValueError, match="already"):
        F.add_fallback(str(first), str(tmp_path / "second"), catastro_docs_path=world["inputs"]["docs"],
                       own_centroids_path=world["own"])
    assert not (tmp_path / "second").exists()


def test_add_refuses_a_tampered_base_index(world, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(world["base"], broken)
    with open(broken / "catastro_docs.parquet", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(IndexValidationError):
        F.add_fallback(str(broken), str(tmp_path / "out"), catastro_docs_path=world["inputs"]["docs"],
                       own_centroids_path=world["own"])
    assert not (tmp_path / "out").exists()


def test_add_refuses_cadastre_docs_without_the_needed_columns(world, tmp_path):
    bad = tmp_path / "bad.parquet"
    pd.read_parquet(world["inputs"]["docs"]).drop(columns=["numero_predial_nacional"]).to_parquet(bad)
    with pytest.raises(ValueError, match="numero_predial_nacional"):
        F.add_fallback(str(world["base"]), str(tmp_path / "out"), catastro_docs_path=str(bad),
                       own_centroids_path=world["own"])
    assert not (tmp_path / "out").exists()


def test_add_refuses_a_pairs_file_that_is_not_the_one_the_base_was_built_from(world, tmp_path):
    other = tmp_path / "other_pairs.parquet"
    pd.read_parquet(world["inputs"]["pairs"]).iloc[:3].to_parquet(other)
    with pytest.raises(ValueError, match="pairs"):
        add(world, tmp_path / "out", pairs_path=str(other))
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------------------ remove / undo

def test_round_trip_add_then_remove_is_byte_identical(world, tmp_path):
    with_fb, without = tmp_path / "with_fb", tmp_path / "without"
    add(world, with_fb)
    F.remove_fallback(str(with_fb), str(without))
    assert {p.name for p in without.iterdir()} == {p.name for p in world["base"].iterdir()}
    for p in world["base"].iterdir():
        assert sha(without / p.name) == sha(p), p.name
    validate_index_dir(str(without))


def test_round_trip_without_pairs_and_with_gold_and_lot_sets(world, tmp_path):
    split_dir = _write_splits(world["src"], _catastro_entries(), n_gold=12)
    gold = {name: pd.read_parquet(os.path.join(split_dir, f"{name}.parquet")) for name in ("dev", "frozen_test")}
    for kw in ({"pairs_path": None}, {"gold": gold}, {"lot_sets": {"source": set(), "kept": set(), "excluded": set()}}):
        with_fb, without = tmp_path / f"w{len(kw)}{next(iter(kw))}", tmp_path / f"o{len(kw)}{next(iter(kw))}"
        add(world, with_fb, **kw)
        F.remove_fallback(str(with_fb), str(without))
        for p in world["base"].iterdir():
            assert sha(without / p.name) == sha(p), (kw.keys(), p.name)


def test_remove_refuses_an_index_without_a_fallback_and_an_existing_out_dir(world, tmp_path):
    with pytest.raises(ValueError, match="fallback"):
        F.remove_fallback(str(world["base"]), str(tmp_path / "out"))
    with_fb = tmp_path / "with_fb"
    add(world, with_fb)
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "x").write_text("x")
    with pytest.raises(FileExistsError):
        F.remove_fallback(str(with_fb), str(taken))
    with pytest.raises(ValueError, match="same"):
        F.remove_fallback(str(with_fb), str(with_fb))


def test_remove_refuses_a_tampered_fallback_index_and_leaves_no_output(world, tmp_path):
    with_fb = tmp_path / "with_fb"
    add(world, with_fb)
    with open(with_fb / "catastro_docs.parquet", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(IndexValidationError):
        F.remove_fallback(str(with_fb), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_remove_detects_edited_base_rows_and_cleans_up(world, tmp_path):
    """If the leading rows are no longer the base's, the slice would not be the base: refuse, leave nothing behind."""
    with_fb = tmp_path / "with_fb"
    add(world, with_fb)
    docs = pd.read_parquet(with_fb / "catastro_docs.parquet")
    docs.loc[0, "direccion"] = "CL 999 # 999 - 999"
    docs.to_parquet(with_fb / "catastro_docs.parquet", index=False)
    meta = meta_of(with_fb)
    meta["files"]["catastro_docs.parquet"] = {"sha256": sha(with_fb / "catastro_docs.parquet"),
                                              "bytes": (with_fb / "catastro_docs.parquet").stat().st_size}
    (with_fb / paths.INDEX_MARKER).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="byte"):
        F.remove_fallback(str(with_fb), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()
