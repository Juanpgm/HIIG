"""Serve a smaller index with the model and embeddings of an experiment trained on a larger one (no training).

Typical case: ``index_v2f_cert_r3`` (cert documents only) holds, in the same order, the first N documents of the
cert + pcer index the model ``v2d_cert_pcer_s42`` was trained and embedded on. The document embeddings of the new index
are then the first N rows of the old ones. This script REFUSES anything it cannot prove:

* the new ``direccion`` array equals the first N entries of the old one, element by element (text AND order),
* ``doc_key`` agrees on that prefix when both files have it,
* ``d_tokens`` of the two ``train_data.npz`` agree on that prefix (the exact input the embeddings were computed from),
* the old embedding matrix has exactly one row per old document.

If the proof fails nothing is created: re-embed with ``scripts/finish_training_artifacts.py --device cpu`` instead.
The model directory is never modified; ``model.pt`` / ``tuning.json`` are hardlinked (copied where links are unavailable),
``catastro_emb.pt`` is a fresh file holding the prefix. Optionally installs a ``decision.json`` /
``reliability_calibrated.json`` pair.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cali_address.experiment import index_meta_record, link_index_files, validate_index_dir  # noqa: E402
from make_decision_experiment import _place, install_pair, validate_pair  # noqa: E402

#: Files taken from the model experiment.
MODEL_FILES = ("model.pt", "tuning.json")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prove_prefix(new_index: str, model_dir: str) -> dict:
    """Evidence that the new index is the leading part of the model's index; ``ValueError`` when it is not.

    Returns ``{"n": N, "direccion_equal": True, "doc_key_equal": bool|None, "d_tokens_equal": bool|None}`` (``None`` =
    not comparable because a file lacks the column / array).
    """
    new_docs = pd.read_parquet(os.path.join(new_index, "catastro_docs.parquet"))
    old_docs = pd.read_parquet(os.path.join(model_dir, "catastro_docs.parquet"))
    n = len(new_docs)
    if n > len(old_docs):
        raise ValueError(f"the new index is larger than the model's ({n} > {len(old_docs)} documents): not a prefix")
    if not (new_docs["direccion"].astype(str).to_numpy() == old_docs["direccion"].astype(str).iloc[:n].to_numpy()).all():
        raise ValueError("the new index is not a prefix of the model's index: the direccion text or order differs")
    proof = {"n": int(n), "direccion_equal": True, "doc_key_equal": None, "d_tokens_equal": None}
    if "doc_key" in new_docs.columns and "doc_key" in old_docs.columns:
        proof["doc_key_equal"] = bool((new_docs["doc_key"].astype(str).to_numpy()
                                       == old_docs["doc_key"].astype(str).iloc[:n].to_numpy()).all())
        if not proof["doc_key_equal"]:
            raise ValueError("the new index is not a prefix of the model's index: doc_key differs")
    new_npz, old_npz = os.path.join(new_index, "train_data.npz"), os.path.join(model_dir, "train_data.npz")
    if os.path.isfile(new_npz) and os.path.isfile(old_npz):
        with np.load(new_npz, allow_pickle=False) as a, np.load(old_npz, allow_pickle=False) as b:
            if "d_tokens" in a.files and "d_tokens" in b.files:
                same = a["d_tokens"].shape[0] == n and bool(np.array_equal(a["d_tokens"], b["d_tokens"][:n]))
                proof["d_tokens_equal"] = same
                if not same:
                    raise ValueError("the new index is not a prefix of the model's index: d_tokens differ")
    return proof


def build_inherit_experiment(index_dir: str, model_dir: str, out_dir: str, tag: str | None = None,
                             decision_path: str | None = None, reliability_path: str | None = None) -> dict:
    """Create ``out_dir`` (see the module docstring) and return the ``experiment.json`` payload.

    ``FileExistsError`` for an existing ``out_dir``; ``FileNotFoundError`` for a missing model file;
    ``ValueError`` for a failed prefix proof, embeddings that do not match the old documents, or an unusable decision
    pair. Nothing is created in any of those cases.
    """
    import torch

    if os.path.lexists(out_dir):
        raise FileExistsError(f"{out_dir} already exists")
    for name in (*MODEL_FILES, "catastro_emb.pt", "catastro_docs.parquet"):
        if not os.path.isfile(os.path.join(model_dir, name)):
            raise FileNotFoundError(f"{name} not found in {model_dir}")
    if (decision_path is None) != (reliability_path is None):
        raise ValueError("pass both decision and reliability files, or neither")
    meta = validate_index_dir(index_dir)
    proof = prove_prefix(index_dir, model_dir)
    n = proof["n"]
    old_rows = len(pd.read_parquet(os.path.join(model_dir, "catastro_docs.parquet"), columns=["direccion"]))
    embeddings = torch.load(os.path.join(model_dir, "catastro_emb.pt"), map_location="cpu")
    if embeddings.shape[0] != old_rows:
        raise ValueError(f"the embeddings have {embeddings.shape[0]} rows but the model's index has {old_rows} documents")
    if meta.get("n_docs") not in (None, n):
        raise ValueError(f"index_meta declares {meta.get('n_docs')} documents but catastro_docs.parquet has {n}")
    pair = validate_pair(decision_path, reliability_path) if decision_path else None
    model_meta: dict = {}
    model_json = os.path.join(model_dir, "experiment.json")
    if os.path.isfile(model_json):
        with open(model_json, "r", encoding="utf-8") as fh:
            model_meta = json.load(fh)
    model_tag = model_meta.get("tag", os.path.basename(os.path.abspath(model_dir)))

    link_index_files(index_dir, out_dir, meta)
    for name in MODEL_FILES:
        _place(os.path.join(model_dir, name), os.path.join(out_dir, name))
    emb_path = os.path.join(out_dir, "catastro_emb.pt")
    torch.save(embeddings[:n].clone().contiguous(), emb_path)
    payload = {
        "tag": tag or os.path.basename(os.path.abspath(out_dir)),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "inherits_model_from": model_tag,
        "inherits_model_reason": (
            f"the document text array, doc_key and tokenised input of {os.path.basename(os.path.abspath(index_dir))} are "
            f"identical to the first {n} documents of the index {model_tag} was trained and embedded on, so model.pt and "
            "tuning.json are reused and catastro_emb.pt is the first rows of the old embeddings (no retraining)"),
        "inherited_prefix_rows": n,
        "prefix_proof": proof,
        "index_meta": index_meta_record(index_dir, meta),
        "model_sha256": {name: _sha256(os.path.join(out_dir, name)) for name in ("model.pt", "catastro_emb.pt", "tuning.json")},
    }
    if pair is not None:
        payload["decision"] = "calibrated"
        payload["decision_provenance"] = install_pair(out_dir, decision_path, reliability_path, *pair)
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", required=True, help="complete index directory to serve (never modified)")
    ap.add_argument("--model-dir", required=True, help="experiment directory holding model.pt, tuning.json, catastro_emb.pt")
    ap.add_argument("--out", required=True, help="new experiment directory (must not exist)")
    ap.add_argument("--decision", default=None)
    ap.add_argument("--reliability", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    try:
        payload = build_inherit_experiment(args.index_dir, args.model_dir, args.out, args.tag, args.decision, args.reliability)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"created {args.out} ({payload['tag']}): {payload['inherited_prefix_rows']} inherited embedding rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
