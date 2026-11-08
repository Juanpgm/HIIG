"""``scripts/make_inherit_experiment.py``: serve a smaller index with a model/embeddings trained on a larger one.

The embeddings are reused ONLY when the new index's document text array (and the tokenised training input) is provably
the prefix of the old one; anything else is refused and nothing is created.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from test_decision_model import decision_doc, params


def _script():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_make_inherit_experiment",
                                                  os.path.join(root, "scripts", "make_inherit_experiment.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


OLD_TEXT = ["CL 901 # 902 - 903", "CL 901 # 902 - 904", "KR 905 # 906 - 907", "KR 905 # 906 - 908", "AV 909 # 911 - 912"]


def write_index(root, name: str, texts: list[str], tokens: np.ndarray) -> str:
    d = root / name
    d.mkdir()
    pd.DataFrame({"direccion": texts, "doc_key": [f"k{i}" for i in range(len(texts))]}).to_parquet(d / "catastro_docs.parquet")
    np.savez(d / "train_data.npz", d_tokens=tokens)
    np.save(d / "spatial_negatives.npy", np.zeros((len(texts), 2)))
    files = {}
    for fname in ("catastro_docs.parquet", "train_data.npz", "spatial_negatives.npy"):
        files[fname] = {"sha256": sha(d / fname), "bytes": os.path.getsize(d / fname)}
    meta = {"schema": 1, "n_docs": len(texts), "tiers": ["cert"], "pre_normalize_separates_letters": True, "files": files}
    (d / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(d)


@pytest.fixture
def world(tmp_path):
    tokens = np.arange(5 * 4).reshape(5, 4)
    new_index = write_index(tmp_path, "index_new", OLD_TEXT[:3], tokens[:3])
    model_dir = tmp_path / "exp_old"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"weights")
    (model_dir / "tuning.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"direccion": OLD_TEXT}).to_parquet(model_dir / "catastro_docs.parquet")
    np.savez(model_dir / "train_data.npz", d_tokens=tokens)
    torch.save(torch.arange(5 * 3, dtype=torch.float16).reshape(5, 3), model_dir / "catastro_emb.pt")
    (model_dir / "experiment.json").write_text(json.dumps({"tag": "exp_old"}), encoding="utf-8")
    dec, rel = tmp_path / "d.json", tmp_path / "r.json"
    dec.write_text(json.dumps(decision_doc()), encoding="utf-8")
    rel.write_text(json.dumps(params()), encoding="utf-8")
    return {"index": new_index, "model": str(model_dir), "out": str(tmp_path / "exp_new"), "dec": str(dec),
            "rel": str(rel), "tmp": tmp_path}


def build(world, **kw):
    return _script().build_inherit_experiment(world["index"], world["model"], world["out"], tag="exp_new", **kw)


def test_builds_a_self_contained_directory_with_the_prefix_of_the_embeddings(world):
    payload = build(world)
    out = world["out"]
    emb = torch.load(os.path.join(out, "catastro_emb.pt"), map_location="cpu")
    old = torch.load(os.path.join(world["model"], "catastro_emb.pt"), map_location="cpu")
    assert emb.shape == (3, 3) and emb.dtype == torch.float16 and torch.equal(emb, old[:3])
    for name in ("model.pt", "tuning.json", "catastro_docs.parquet", "index_meta.json", "train_data.npz",
                 "spatial_negatives.npy", "experiment.json"):
        assert os.path.isfile(os.path.join(out, name)), name
    assert sha(os.path.join(out, "model.pt")) == sha(os.path.join(world["model"], "model.pt"))
    assert payload["inherits_model_from"] == "exp_old" and payload["inherited_prefix_rows"] == 3
    assert payload["prefix_proof"]["direccion_equal"] is True and payload["prefix_proof"]["d_tokens_equal"] is True
    assert payload["model_sha256"]["catastro_emb.pt"] == sha(os.path.join(out, "catastro_emb.pt"))
    assert "decision" not in payload


def test_the_decision_pair_is_installed_when_given(world):
    payload = build(world, decision_path=world["dec"], reliability_path=world["rel"])
    assert payload["decision"] == "calibrated" and payload["decision_provenance"]["tau_p"] == 0.94
    assert sha(os.path.join(world["out"], "decision.json")) == sha(world["dec"])
    assert sha(os.path.join(world["out"], "reliability_calibrated.json")) == sha(world["rel"])


def test_a_changed_text_is_refused_and_nothing_is_created(world):
    docs = pd.read_parquet(os.path.join(world["model"], "catastro_docs.parquet"))
    docs.loc[1, "direccion"] = "CL 901 # 902 - 999"
    docs.to_parquet(os.path.join(world["model"], "catastro_docs.parquet"))
    with pytest.raises(ValueError, match="prefix"):
        build(world)
    assert not os.path.exists(world["out"])


def test_a_different_order_is_refused(world):
    docs = pd.read_parquet(os.path.join(world["model"], "catastro_docs.parquet"))
    docs.loc[[0, 1], "direccion"] = docs.loc[[1, 0], "direccion"].to_numpy()
    docs.to_parquet(os.path.join(world["model"], "catastro_docs.parquet"))
    with pytest.raises(ValueError, match="prefix"):
        build(world)
    assert not os.path.exists(world["out"])


def test_different_tokens_are_refused_even_when_the_text_matches(world):
    np.savez(os.path.join(world["model"], "train_data.npz"), d_tokens=np.zeros((5, 4), dtype=int))
    with pytest.raises(ValueError, match="d_tokens"):
        build(world)
    assert not os.path.exists(world["out"])


def test_a_larger_new_index_is_refused(world):
    big = write_index(world["tmp"], "index_big", OLD_TEXT + ["KR 913 # 914 - 915"], np.zeros((6, 4)))
    world["index"] = big
    with pytest.raises(ValueError, match="larger"):
        build(world)


def test_embeddings_that_do_not_match_the_old_docs_are_refused(world):
    torch.save(torch.zeros((4, 3), dtype=torch.float16), os.path.join(world["model"], "catastro_emb.pt"))
    with pytest.raises(ValueError, match="rows"):
        build(world)
    assert not os.path.exists(world["out"])


def test_existing_output_is_refused(world):
    os.makedirs(world["out"])
    with pytest.raises(FileExistsError):
        build(world)


def test_missing_model_files_are_refused(world):
    os.remove(os.path.join(world["model"], "model.pt"))
    with pytest.raises(FileNotFoundError):
        build(world)
    assert not os.path.exists(world["out"])


def test_an_invalid_decision_pair_is_refused_before_anything_is_created(world):
    open(world["dec"], "w", encoding="utf-8").write(json.dumps(decision_doc(tau_p=0.1)))
    with pytest.raises(ValueError, match="decision"):
        build(world, decision_path=world["dec"], reliability_path=world["rel"])
    assert not os.path.exists(world["out"])


def test_the_model_directory_is_not_modified(world):
    before = {f: sha(os.path.join(world["model"], f)) for f in os.listdir(world["model"])}
    build(world)
    assert before == {f: sha(os.path.join(world["model"], f)) for f in os.listdir(world["model"])}
