from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import zipfile

import numpy as np
import pytest

from conftest import MODEL_DIR, build_synthetic_bundle, requires_model
from servicio.bundle import (
    INDEX_REQUIRED,
    BundleError,
    IndexIncompatibleError,
    IndexMissingError,
    ModelMissingError,
    assemble_runtime,
    check_index,
    check_model_dir,
    download_index,
    ensure_index,
    resolve_backend,
)


def _copy(bundle: str, tmp_path) -> str:
    dest = str(tmp_path / "copia")
    shutil.copytree(bundle, dest)
    return dest


# -- model directory / backend fallback -------------------------------------------------------
@requires_model
def test_model_dir_ok():
    meta = check_model_dir(MODEL_DIR, "onnx")
    assert meta["d_out"] == 256 and meta["max_len"] == 40
    check_model_dir(MODEL_DIR, "onnx-int8")


def test_model_dir_missing(tmp_path):
    with pytest.raises(ModelMissingError, match="no existe"):
        check_model_dir(str(tmp_path / "nada"), "onnx")


def test_model_dir_without_meta(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    with pytest.raises(ModelMissingError, match="onnx_meta.json"):
        check_model_dir(str(tmp_path), "onnx")


def test_model_meta_missing_key(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "onnx_meta.json").write_text(json.dumps({"max_len": 40}), encoding="utf-8")
    with pytest.raises(ModelMissingError, match="d_out"):
        check_model_dir(str(tmp_path), "onnx")


def test_model_meta_malformed(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "onnx_meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BundleError):
        check_model_dir(str(tmp_path), "onnx")


def test_int8_falls_back_to_fp32_when_missing(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    backend, notes = resolve_backend("onnx-int8", str(tmp_path), allow_fallback=True)
    assert backend == "onnx" and "model.int8.onnx" in notes[0]


def test_int8_fallback_disabled_is_an_error(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    with pytest.raises(ModelMissingError, match="FALLBACK=false"):
        resolve_backend("onnx-int8", str(tmp_path), allow_fallback=False)


def test_int8_without_any_graph_is_an_error(tmp_path):
    with pytest.raises(ModelMissingError):
        resolve_backend("onnx-int8", str(tmp_path), allow_fallback=True)


def test_int8_present_is_used(tmp_path):
    (tmp_path / "model.int8.onnx").write_bytes(b"x")
    assert resolve_backend("onnx-int8", str(tmp_path), True) == ("onnx-int8", [])


def test_fp32_never_falls_back(tmp_path):
    assert resolve_backend("onnx", str(tmp_path), True) == ("onnx", [])


# -- index checks ---------------------------------------------------------------------------
def test_missing_index_dir_has_actionable_message(tmp_path, model_meta):
    with pytest.raises(IndexMissingError) as exc:
        check_index(str(tmp_path / "sin_indice"), model_meta)
    text = str(exc.value)
    assert "INDEX_DIR" in text and "INDEX_URL" in text and "preparar_indice.py" in text


def test_empty_index_dir_lists_every_file(tmp_path, model_meta):
    with pytest.raises(IndexMissingError) as exc:
        check_index(str(tmp_path), model_meta)
    for name in INDEX_REQUIRED:
        assert name in str(exc.value)


@pytest.mark.parametrize("name", INDEX_REQUIRED)
def test_one_missing_file(synthetic_bundle, tmp_path, model_meta, name):
    d = _copy(synthetic_bundle, tmp_path)
    os.remove(os.path.join(d, name))
    with pytest.raises(IndexMissingError, match=name.replace(".", r"\.")):
        check_index(d, model_meta)


def test_valid_bundle_passes(synthetic_bundle, model_meta):
    assert check_index(synthetic_bundle, model_meta) == []


def test_no_manifest_only_warns(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    os.remove(os.path.join(d, "indice_manifest.json"))
    warnings = check_index(d, model_meta)
    assert len(warnings) == 1 and "indice_manifest.json" in warnings[0]


def test_row_mismatch_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    emb = np.load(os.path.join(d, "catastro_emb.npy"))
    np.save(os.path.join(d, "catastro_emb.npy"), emb[:-1])
    with pytest.raises(IndexIncompatibleError, match="filas"):
        check_index(d, model_meta)


def test_dimension_mismatch_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    emb = np.load(os.path.join(d, "catastro_emb.npy"))
    np.save(os.path.join(d, "catastro_emb.npy"), emb[:, :128])
    with pytest.raises(IndexIncompatibleError, match="dimensiones"):
        check_index(d, model_meta)


def test_corrupt_embeddings_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    with open(os.path.join(d, "catastro_emb.npy"), "wb") as fh:
        fh.write(b"not a numpy file")
    with pytest.raises(IndexIncompatibleError, match="no se puede leer"):
        check_index(d, model_meta)


def test_malformed_index_meta_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    with open(os.path.join(d, "index_meta.json"), "w", encoding="utf-8") as fh:
        fh.write("[1, 2")
    with pytest.raises(BundleError):
        check_index(d, model_meta)


def test_manifest_from_other_model_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    path = os.path.join(d, "indice_manifest.json")
    manifest = json.load(open(path, encoding="utf-8"))
    manifest["model_sha256"] = "0" * 64
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    with pytest.raises(IndexIncompatibleError, match="otro modelo"):
        check_index(d, model_meta)


def test_truncated_file_detected_by_manifest(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    with open(os.path.join(d, "index_meta.json"), "a", encoding="utf-8") as fh:
        fh.write(" ")
    with pytest.raises(IndexIncompatibleError, match="manifiesto"):
        check_index(d, model_meta)


def test_empty_index_rejected(tmp_path, model_meta):
    import pandas as pd

    d = tmp_path / "vacio"
    d.mkdir()
    pd.DataFrame({"direccion": pd.Series([], dtype="object")}).to_parquet(d / "catastro_docs.parquet")
    np.save(d / "catastro_emb.npy", np.zeros((0, 256), dtype="float16"))
    (d / "index_meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IndexIncompatibleError, match="no tiene documentos"):
        check_index(str(d), model_meta)


# -- runtime layout ---------------------------------------------------------------------------
def test_assemble_links_only_known_files(synthetic_bundle, tmp_path):
    index = _copy(synthetic_bundle, tmp_path)
    (tmp_path / "copia" / "datos_privados.csv").write_text("x", encoding="utf-8")
    basemaps = tmp_path / "bm"
    basemaps.mkdir()
    (basemaps / "barrios_veredas.geojson").write_text("{}", encoding="utf-8")
    (basemaps / "otra.geojson").write_text("{}", encoding="utf-8")
    artifacts, bm = assemble_runtime(str(tmp_path / "rt"), MODEL_DIR, index, str(basemaps))
    names = set(os.listdir(artifacts))
    assert {"catastro_docs.parquet", "catastro_emb.npy", "index_meta.json", "onnx_meta.json"} <= names
    assert "datos_privados.csv" not in names and "indice_manifest.json" not in names
    assert os.listdir(bm) == ["barrios_veredas.geojson"]


def test_assemble_is_idempotent_and_tolerates_missing_basemaps(synthetic_bundle, tmp_path):
    rt = str(tmp_path / "rt")
    for _ in range(2):
        artifacts, bm = assemble_runtime(rt, MODEL_DIR, synthetic_bundle, str(tmp_path / "no_basemaps"))
    assert os.listdir(bm) == []
    assert os.path.isfile(os.path.join(artifacts, "catastro_emb.npy"))


# -- download (INDEX_URL) ---------------------------------------------------------------------
def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _tar(bundle: str, out, prefix: str = "") -> str:
    with tarfile.open(out, "w:gz") as tf:
        for name in os.listdir(bundle):
            tf.add(os.path.join(bundle, name), arcname=prefix + name)
    return str(out)


def test_download_tar_with_good_sha(synthetic_bundle, tmp_path, model_meta):
    archive = _tar(synthetic_bundle, tmp_path / "b.tar.gz")
    dest = str(tmp_path / "data" / "indice")
    lines = ensure_index(dest, "file:///" + archive.replace("\\", "/").lstrip("/"), _sha(archive), allow_file_url=True)
    assert lines and check_index(dest, model_meta) == []


def test_download_nested_folder_and_zip(synthetic_bundle, tmp_path, model_meta):
    archive = tmp_path / "b.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in os.listdir(synthetic_bundle):
            zf.write(os.path.join(synthetic_bundle, name), arcname="paquete/" + name)
    dest = str(tmp_path / "indice")
    download_index("file:///" + str(archive).replace("\\", "/").lstrip("/"), _sha(archive), dest, allow_file_url=True)
    assert check_index(dest, model_meta) == []


def test_download_bad_sha_leaves_nothing(synthetic_bundle, tmp_path):
    archive = _tar(synthetic_bundle, tmp_path / "b.tar.gz")
    dest = tmp_path / "data" / "indice"
    with pytest.raises(BundleError, match="INDEX_SHA256"):
        download_index("file:///" + archive.replace("\\", "/").lstrip("/"), "0" * 64, str(dest), allow_file_url=True)
    assert not dest.exists() or os.listdir(dest) == []
    assert [p for p in os.listdir(tmp_path / "data") if p.startswith(".indice-")] == []


def test_download_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"{}"
        info = tarfile.TarInfo("../escape.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(BundleError, match="insegura"):
        download_index("file:///" + str(archive).replace("\\", "/").lstrip("/"), _sha(archive),
                       str(tmp_path / "indice"), allow_file_url=True)
    assert not (tmp_path / "escape.json").exists()


def test_download_not_an_archive(tmp_path):
    blob = tmp_path / "x.bin"
    blob.write_bytes(b"plain bytes")
    with pytest.raises(BundleError, match="no es un"):
        download_index("file:///" + str(blob).replace("\\", "/").lstrip("/"), _sha(blob), str(tmp_path / "i"), allow_file_url=True)


def test_download_archive_without_bundle(tmp_path):
    archive = tmp_path / "otro.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("leeme.txt")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"hi"))
    with pytest.raises(BundleError, match="index_meta.json"):
        download_index("file:///" + str(archive).replace("\\", "/").lstrip("/"), _sha(archive),
                       str(tmp_path / "i"), allow_file_url=True)


def test_download_unreachable_url(tmp_path):
    with pytest.raises(BundleError, match="no se pudo descargar"):
        download_index("file:///" + str(tmp_path / "no_existe.tar.gz").replace("\\", "/").lstrip("/"),
                       "0" * 64, str(tmp_path / "i"), timeout_s=2, allow_file_url=True)


def test_ensure_index_skips_download_when_complete(synthetic_bundle):
    assert ensure_index(synthetic_bundle, "https://example.invalid/never-called", "0" * 64) == []


def test_ensure_index_without_url_does_nothing(tmp_path):
    assert ensure_index(str(tmp_path / "vacio"), "", "") == []


# -- manifest integrity (sha256 of every file) ------------------------------------------------
def _manifest(d):
    path = os.path.join(d, "indice_manifest.json")
    return path, json.load(open(path, encoding="utf-8"))


def test_tampered_file_same_size_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    path = os.path.join(d, "index_meta.json")
    raw = open(path, "rb").read()
    tampered = raw.replace(b'"schema": 2', b'"schema": 3')  # still valid JSON, same length
    assert tampered != raw and len(tampered) == len(raw)
    open(path, "wb").write(tampered)
    with pytest.raises(IndexIncompatibleError, match="sha256"):
        check_index(d, model_meta)


def test_tampered_optional_pickle_same_size_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    pkl = os.path.join(d, "gazetteer.pkl")
    open(pkl, "wb").write(b"A" * 64)
    path, manifest = _manifest(d)
    manifest["files"]["gazetteer.pkl"] = {"bytes": 64, "sha256": hashlib.sha256(b"B" * 64).hexdigest()}
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    with pytest.raises(IndexIncompatibleError, match="gazetteer.pkl"):
        check_index(d, model_meta)


def test_file_missing_from_manifest_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    open(os.path.join(d, "interp_index_v1.pkl"), "wb").write(b"pickle-bytes")  # present, never hashed
    with pytest.raises(IndexIncompatibleError, match="interp_index_v1.pkl"):
        check_index(d, model_meta)


def test_required_file_dropped_from_manifest_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    path, manifest = _manifest(d)
    del manifest["files"]["catastro_emb.npy"]
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    with pytest.raises(IndexIncompatibleError, match="catastro_emb.npy"):
        check_index(d, model_meta)


def test_manifest_lists_absent_optional_file_rejected(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    path, manifest = _manifest(d)
    manifest["files"]["gazetteer.pkl"] = {"bytes": 3, "sha256": "0" * 64}
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    with pytest.raises(IndexIncompatibleError, match="no esta"):
        check_index(d, model_meta)


@pytest.mark.parametrize("digest", [None, "", "abc", 123, "z" * 64])
def test_manifest_entry_without_valid_sha256_rejected(synthetic_bundle, tmp_path, model_meta, digest):
    d = _copy(synthetic_bundle, tmp_path)
    path, manifest = _manifest(d)
    manifest["files"]["index_meta.json"]["sha256"] = digest
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    with pytest.raises(IndexIncompatibleError, match="sha256"):
        check_index(d, model_meta)


@pytest.mark.parametrize("files", [{}, None, [], "x"])
def test_manifest_without_files_rejected(synthetic_bundle, tmp_path, model_meta, files):
    d = _copy(synthetic_bundle, tmp_path)
    path, manifest = _manifest(d)
    manifest["files"] = files
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    with pytest.raises(IndexIncompatibleError, match="files"):
        check_index(d, model_meta)


def test_uppercase_manifest_digest_accepted(synthetic_bundle, tmp_path, model_meta):
    d = _copy(synthetic_bundle, tmp_path)
    path, manifest = _manifest(d)
    info = manifest["files"]["index_meta.json"]
    info["sha256"] = info["sha256"].upper()
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    assert check_index(d, model_meta) == []


def test_valid_bundle_with_optional_files_passes(synthetic_bundle, tmp_path, model_meta):
    from servicio.bundle import build_manifest

    d = _copy(synthetic_bundle, tmp_path)
    open(os.path.join(d, "gazetteer.pkl"), "wb").write(b"cache")
    json.dump(build_manifest(d, model_meta), open(os.path.join(d, "indice_manifest.json"), "w", encoding="utf-8"))
    assert check_index(d, model_meta) == []


# -- download hardening -----------------------------------------------------------------------
def _file_url(p) -> str:
    return "file:///" + str(p).replace("\\", "/").lstrip("/")


def test_file_url_refused_by_default(synthetic_bundle, tmp_path):
    archive = _tar(synthetic_bundle, tmp_path / "b.tar.gz")
    with pytest.raises(BundleError, match="file://"):
        download_index(_file_url(archive), _sha(archive), str(tmp_path / "i"))
    assert not (tmp_path / "i").exists()


@pytest.mark.parametrize("url", ["http://example.invalid/i.tar.gz", "ftp://example.invalid/i", "", "/etc/passwd"])
def test_non_https_schemes_refused(tmp_path, url):
    with pytest.raises(BundleError, match="https"):
        download_index(url, "0" * 64, str(tmp_path / "i"))


def test_download_size_cap_aborts_and_cleans_up(synthetic_bundle, tmp_path):
    archive = _tar(synthetic_bundle, tmp_path / "b.tar.gz")
    size = os.path.getsize(archive)
    dest = tmp_path / "data" / "indice"
    with pytest.raises(BundleError, match="limite"):
        download_index(_file_url(archive), _sha(archive), str(dest), max_bytes=size - 1, allow_file_url=True)
    assert [p for p in os.listdir(tmp_path / "data") if p.startswith(".indice-")] == []


def test_download_size_cap_boundary_accepted(synthetic_bundle, tmp_path, model_meta):
    archive = _tar(synthetic_bundle, tmp_path / "b.tar.gz")
    dest = str(tmp_path / "indice")
    download_index(_file_url(archive), _sha(archive), dest, max_bytes=os.path.getsize(archive),
                   allow_file_url=True)
    assert check_index(dest, model_meta) == []


def test_readonly_index_dir_becomes_bundle_error(synthetic_bundle, tmp_path, monkeypatch):
    archive = _tar(synthetic_bundle, tmp_path / "b.tar.gz")

    def boom(*_a, **_k):
        raise PermissionError(13, "Read-only file system")

    monkeypatch.setattr("servicio.bundle.shutil.move", boom)
    with pytest.raises(BundleError, match="INDEX_DIR"):
        download_index(_file_url(archive), _sha(archive), str(tmp_path / "indice"), allow_file_url=True)
    assert [p for p in os.listdir(tmp_path) if p.startswith(".indice-")] == []


def test_unwritable_parent_becomes_bundle_error(synthetic_bundle, tmp_path, monkeypatch):
    archive = _tar(synthetic_bundle, tmp_path / "b.tar.gz")

    def boom(*_a, **_k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("servicio.bundle.tempfile.mkdtemp", boom)
    with pytest.raises(BundleError, match="INDEX_DIR"):
        download_index(_file_url(archive), _sha(archive), str(tmp_path / "indice"), allow_file_url=True)
