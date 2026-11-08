"""Experiment directory of the fallback index: embeddings of the appended docs and the undo of the whole experiment.

Tiny CPU model (random weights), invented data. The base rows of the new embedding matrix must be BIT-identical to the
base file and removing the fallback must give back the base ``catastro_emb.pt`` byte for byte.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from test_index_fallback_files import add, meta_of, sha, world  # noqa: F401  (world is a fixture)

from cali_address import index_fallback as F
from cali_address import paths
from cali_address.experiment import link_index_files, validate_index_dir
from cali_address.model import AddressEncoder
from cali_address.tokenizer import encode_batch
from cali_address.train import embed_documents, load_checkpoint, save_checkpoint


def _tiny_model_files(directory) -> None:
    torch.manual_seed(7)
    model = AddressEncoder(d_model=16, n_layers=1, n_heads=2, d_out=16, dropout=0.0, n_comuna=3, n_barrio=3, max_len=40)
    data = {"doc_comuna": np.array([0, 1, 2]), "doc_barrio": np.array([0, 1, 2]), "xy_mean": np.zeros(2),
            "xy_std": np.ones(2), "comuna_codes": np.array(["01", "02", "03"], dtype=object),
            "barrio_codes": np.array(["01", "02", "03"], dtype=object)}
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_out": 16, "dropout": 0.0, "max_len": 40}
    save_checkpoint(str(directory / "model.pt"), model, cfg, data)
    (directory / "tuning.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")


@pytest.fixture()
def base_experiment(world, tmp_path):  # noqa: F811
    """Experiment dir of the BASE index: its files, a tiny model, and the base embeddings."""
    exp = tmp_path / "exp_base"
    exp.mkdir()
    link_index_files(str(world["base"]), str(exp), meta_of(world["base"]))
    _tiny_model_files(exp)
    model, _ = load_checkpoint(str(exp / "model.pt"), "cpu", dropout=0.0)
    docs = pd.read_parquet(exp / "catastro_docs.parquet")
    emb = embed_documents(model, encode_batch([t.upper() for t in docs["direccion"]]), device="cpu").cpu()
    F.save_embeddings(emb, str(exp))
    (exp / "experiment.json").write_text(json.dumps({"tag": "exp_base"}), encoding="utf-8")
    return exp


def build(world, base_experiment, tmp_path, **kw):  # noqa: F811
    with_fb = tmp_path / "idx_fb"
    add(world, with_fb)
    out = tmp_path / "exp_fb"
    payload = F.build_fallback_experiment(str(with_fb), str(base_experiment), str(out), **kw)
    return with_fb, out, payload


def test_experiment_embeddings_keep_the_base_rows_bit_identical(world, base_experiment, tmp_path):  # noqa: F811
    _, out, payload = build(world, base_experiment, tmp_path, tag="exp_fb")
    base_emb = torch.load(base_experiment / "catastro_emb.pt", map_location="cpu")
    emb = torch.load(out / "catastro_emb.pt", map_location="cpu")
    n = base_emb.shape[0]
    assert emb.shape == (n + 7, base_emb.shape[1]) and emb.dtype == base_emb.dtype == torch.float16
    assert torch.equal(emb[:n], base_emb)
    assert emb[:n].numpy().tobytes() == base_emb.numpy().tobytes()
    docs = pd.read_parquet(out / "catastro_docs.parquet")
    model, _ = load_checkpoint(str(out / "model.pt"), "cpu", dropout=0.0)
    fresh = embed_documents(model, encode_batch([t.upper() for t in docs["direccion"].iloc[n:]]), device="cpu").cpu()
    assert torch.equal(emb[n:], fresh)
    assert payload["tag"] == "exp_fb"


def test_experiment_dir_is_self_contained_and_leaves_the_sources_alone(world, base_experiment, tmp_path):  # noqa: F811
    before = {p.name: sha(p) for p in base_experiment.iterdir()}
    base_before = {p.name: sha(p) for p in world["base"].iterdir()}
    with_fb, out, payload = build(world, base_experiment, tmp_path)
    validate_index_dir(str(out))
    for name in ("model.pt", "tuning.json"):
        assert sha(out / name) == sha(base_experiment / name)
    assert sha(out / paths.INDEX_MARKER) == sha(with_fb / paths.INDEX_MARKER)
    assert {p.name: sha(p) for p in base_experiment.iterdir()} == before
    assert {p.name: sha(p) for p in world["base"].iterdir()} == base_before


def test_experiment_json_records_lineage_and_the_three_ways_back(world, base_experiment, tmp_path):  # noqa: F811
    _, out, payload = build(world, base_experiment, tmp_path)
    on_disk = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(payload))
    assert on_disk["inherits_model_from"] == "exp_base"
    block = on_disk["fallback"]
    assert block["first_doc_id"] == meta_of(world["base"])["n_docs"] and block["n_docs_added"] == 7
    assert block["base_catastro_emb_sha256"] == sha(base_experiment / "catastro_emb.pt")
    assert block["base_rows_bit_identical"] is True
    assert "remove" in block["reverse"] and "exclude_fuentes" in block["reverse"]["switch"]
    assert "NORMALIZER_EXCLUDE_FUENTES" in block["reverse"]["switch"]
    assert on_disk["model_sha256"]["catastro_emb.pt"] == sha(out / "catastro_emb.pt")
    assert on_disk["index_meta"]["n_docs"] == meta_of(world["base"])["n_docs"] + 7


def test_removing_the_fallback_from_the_experiment_restores_the_base_files_byte_for_byte(world, base_experiment, tmp_path):  # noqa: F811
    _, out, _ = build(world, base_experiment, tmp_path)
    restored = tmp_path / "restored"
    F.remove_fallback(str(out), str(restored))
    for name in ("catastro_docs.parquet", "catastro_own_centroids.parquet", "train_pairs.parquet",
                 "index_meta.json", "train_data.npz", "spatial_negatives.npy", "nomenclatura_units.parquet"):
        assert sha(restored / name) == sha(base_experiment / name), name
    assert sha(restored / "catastro_emb.pt") == sha(base_experiment / "catastro_emb.pt")


def test_experiment_with_nothing_added_is_a_faithful_copy(world, base_experiment, tmp_path):  # noqa: F811
    cat = pd.read_parquet(world["inputs"]["docs"])
    every = set(cat["numero_predial_nacional"].str[:21])
    with_fb = tmp_path / "idx_fb"
    add(world, with_fb, lot_sets={"source": every, "kept": every, "excluded": set()})
    out = tmp_path / "exp_fb"
    F.build_fallback_experiment(str(with_fb), str(base_experiment), str(out))
    assert sha(out / "catastro_emb.pt") == sha(base_experiment / "catastro_emb.pt")


# --------------------------------------------------------------------------------------------------------- refusals

def test_experiment_refuses_an_index_without_fallback(world, base_experiment, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="fallback"):
        F.build_fallback_experiment(str(world["base"]), str(base_experiment), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_experiment_refuses_embeddings_that_do_not_match_the_base_docs(world, base_experiment, tmp_path):  # noqa: F811
    emb = torch.load(base_experiment / "catastro_emb.pt", map_location="cpu")
    os.remove(base_experiment / "catastro_emb.pt")
    F.save_embeddings(emb[:-3], str(base_experiment))
    with pytest.raises(ValueError, match="rows"):
        build(world, base_experiment, tmp_path)
    assert not (tmp_path / "exp_fb").exists()


def test_experiment_refuses_a_model_dir_whose_docs_are_not_the_base_prefix(world, base_experiment, tmp_path):  # noqa: F811
    docs = pd.read_parquet(base_experiment / "catastro_docs.parquet")
    os.remove(base_experiment / "catastro_docs.parquet")                     # unlink the hardlink, never write through it
    docs.loc[0, "direccion"] = "CL 999 # 999 - 999"
    docs.to_parquet(base_experiment / "catastro_docs.parquet", index=False)
    with pytest.raises(ValueError, match="prefix"):
        build(world, base_experiment, tmp_path)
    assert not (tmp_path / "exp_fb").exists()


def test_experiment_refuses_missing_model_files_and_an_existing_out_dir(world, base_experiment, tmp_path):  # noqa: F811
    with_fb = tmp_path / "idx_fb"
    add(world, with_fb)
    os.remove(base_experiment / "model.pt")
    with pytest.raises(FileNotFoundError):
        F.build_fallback_experiment(str(with_fb), str(base_experiment), str(tmp_path / "out"))
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        F.build_fallback_experiment(str(with_fb), str(base_experiment), str(tmp_path / "taken"))
    assert not (tmp_path / "out").exists()
