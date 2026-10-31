from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile

import numpy as np
import pytest

from conftest import MODEL_DIR, ROOT, requires_model

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import preparar_indice  # noqa: E402

pytestmark = requires_model


def _source(synthetic_bundle, tmp_path, keep_emb: bool) -> str:
    src = tmp_path / "origen"
    src.mkdir()
    for name in ("catastro_docs.parquet", "index_meta.json") + (("catastro_emb.npy",) if keep_emb else ()):
        shutil.copy2(os.path.join(synthetic_bundle, name), src / name)
    return str(src)


def test_builds_embeddings_with_doc_graph(synthetic_bundle, tmp_path, model_meta):
    src = _source(synthetic_bundle, tmp_path, keep_emb=False)
    dest = str(tmp_path / "indice")
    assert preparar_indice.main(["--origen", src, "--destino", dest]) == 0
    manifest = json.load(open(os.path.join(dest, "indice_manifest.json"), encoding="utf-8"))
    assert manifest["emb_source"] == "onnx-doc-graph" and manifest["model_verified"] is True
    assert manifest["model_sha256"] == model_meta["model_sha256"] and manifest["n_docs"] == 120
    emb = np.load(os.path.join(dest, "catastro_emb.npy"))
    assert emb.dtype == np.float16 and emb.shape == (120, 256)
    assert np.allclose(np.linalg.norm(emb.astype("float32"), axis=1), 1.0, atol=1e-2)


def test_copies_existing_npy_and_warns_without_checkpoint(synthetic_bundle, tmp_path, capsys):
    src = _source(synthetic_bundle, tmp_path, keep_emb=True)
    dest = str(tmp_path / "indice")
    assert preparar_indice.main(["--origen", src, "--destino", dest, "--enlazar"]) == 0
    assert "no trae model.pt" in capsys.readouterr().err
    manifest = json.load(open(os.path.join(dest, "indice_manifest.json"), encoding="utf-8"))
    assert manifest["emb_source"] == "copied-npy" and manifest["model_verified"] is False


def test_rejects_embeddings_from_another_model(synthetic_bundle, tmp_path):
    src = _source(synthetic_bundle, tmp_path, keep_emb=True)
    with open(os.path.join(src, "model.pt"), "wb") as fh:
        fh.write(b"a different checkpoint")
    with pytest.raises(SystemExit, match="otro modelo"):
        preparar_indice.main(["--origen", src, "--destino", str(tmp_path / "indice")])


def test_missing_docs(tmp_path):
    (tmp_path / "origen").mkdir()
    with pytest.raises(SystemExit, match="catastro_docs.parquet"):
        preparar_indice.main(["--origen", str(tmp_path / "origen"), "--destino", str(tmp_path / "d")])


def test_same_source_and_dest_rejected(tmp_path):
    with pytest.raises(SystemExit, match="distintos"):
        preparar_indice.main(["--origen", str(tmp_path), "--destino", str(tmp_path)])


def test_package_sha_matches_and_archive_is_flat(synthetic_bundle, tmp_path, capsys):
    src = _source(synthetic_bundle, tmp_path, keep_emb=True)
    archive = str(tmp_path / "indice.tar.gz")
    assert preparar_indice.main(["--origen", src, "--destino", str(tmp_path / "indice"),
                                 "--empaquetar", archive]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sha256"] == hashlib.sha256(open(archive, "rb").read()).hexdigest()
    with tarfile.open(archive) as tf:
        names = set(tf.getnames())
    assert {"catastro_docs.parquet", "catastro_emb.npy", "index_meta.json", "indice_manifest.json"} <= names
    assert all("/" not in n for n in names)


def test_prewarm_builds_caches(synthetic_bundle, tmp_path):
    src = _source(synthetic_bundle, tmp_path, keep_emb=True)
    dest = tmp_path / "indice"
    assert preparar_indice.main(["--origen", src, "--destino", str(dest), "--precalentar"]) == 0
    manifest = json.load(open(dest / "indice_manifest.json", encoding="utf-8"))
    assert "interp_index_v1" in manifest["prewarmed"] or "catalog_keys_v1" in manifest["prewarmed"]


def test_doc_graph_must_match_serving_model(synthetic_bundle, tmp_path):
    model = tmp_path / "modelo"
    shutil.copytree(MODEL_DIR, model)
    meta = json.load(open(model / "model.doc.meta.json", encoding="utf-8"))
    meta["model_sha256"] = "f" * 64
    json.dump(meta, open(model / "model.doc.meta.json", "w", encoding="utf-8"))
    src = _source(synthetic_bundle, tmp_path, keep_emb=False)
    with pytest.raises(SystemExit, match="no corresponde"):
        preparar_indice.main(["--origen", src, "--destino", str(tmp_path / "d"), "--modelo", str(model)])
