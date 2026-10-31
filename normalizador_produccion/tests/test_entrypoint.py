"""``python -m servicio check`` as a subprocess: exit codes and messages operators will see."""

from __future__ import annotations

import os
import subprocess
import sys

from conftest import MODEL_DIR, SRC, requires_model


def _run(tmp_path, **env):
    full = {k: v for k, v in os.environ.items() if not k.startswith(("NORMALIZER_", "INDEX_", "MAX_"))}
    full.update({"PYTHONPATH": SRC, "MODEL_DIR": MODEL_DIR, "RUNTIME_DIR": str(tmp_path / "rt"),
                 "INDEX_DIR": str(tmp_path / "sin_indice"), "BASEMAPS_DIR": str(tmp_path / "bm")})
    full.update(env)
    return subprocess.run([sys.executable, "-m", "servicio", "check"], env=full, capture_output=True,
                          text=True, timeout=120)


def test_missing_index_exits_3_with_clear_message(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 3, r.stderr
    assert "no hay paquete de indice" in r.stderr and "GUIA_DESPLIEGUE.md" in r.stderr
    assert "Traceback" not in r.stderr


def test_bad_env_exits_2(tmp_path):
    r = _run(tmp_path, PORT="http")
    assert r.returncode == 2 and "PORT" in r.stderr and "Traceback" not in r.stderr


def test_torch_backend_exits_2(tmp_path):
    r = _run(tmp_path, NORMALIZER_BACKEND="torch")
    assert r.returncode == 2 and "torch" in r.stderr


def test_missing_model_dir_exits_3(tmp_path):
    r = _run(tmp_path, MODEL_DIR=str(tmp_path / "sin_modelo"))
    assert r.returncode == 3 and "MODEL_DIR" in r.stderr


@requires_model
def test_valid_bundle_check_ok(tmp_path, synthetic_bundle):
    r = _run(tmp_path, INDEX_DIR=synthetic_bundle, NORMALIZER_BACKEND="onnx-int8", NORMALIZER_ONNX_THREADS="2")
    assert r.returncode == 0, r.stderr
    assert '"backend": "onnx-int8"' in r.stderr and '"threads": 2' in r.stderr
    assert "aviso: no hay mapas base" in r.stderr


@requires_model
def test_int8_fallback_visible_in_log(tmp_path, synthetic_bundle):
    model = tmp_path / "modelo"
    model.mkdir()
    for name in ("model.onnx", "onnx_meta.json"):
        (model / name).write_bytes(open(os.path.join(MODEL_DIR, name), "rb").read())
    r = _run(tmp_path, INDEX_DIR=synthetic_bundle, MODEL_DIR=str(model), NORMALIZER_BACKEND="onnx-int8")
    assert r.returncode == 0, r.stderr
    assert "se usa el backend onnx" in r.stderr and '"backend": "onnx"' in r.stderr
    r = _run(tmp_path, INDEX_DIR=synthetic_bundle, MODEL_DIR=str(model), NORMALIZER_BACKEND="onnx-int8",
             NORMALIZER_BACKEND_FALLBACK="false")
    assert r.returncode == 3


@requires_model
def test_incompatible_index_exits_3(tmp_path, synthetic_bundle):
    import shutil

    import numpy as np

    d = tmp_path / "roto"
    shutil.copytree(synthetic_bundle, d)
    np.save(d / "catastro_emb.npy", np.zeros((3, 256), dtype="float16"))
    r = _run(tmp_path, INDEX_DIR=str(d))
    assert r.returncode == 3 and "filas" in r.stderr


@requires_model
def test_blas_threads_exported_before_numpy_import(tmp_path, synthetic_bundle):
    """Regression: OpenBLAS reads its thread count when numpy is first imported (inside check_index)."""
    code = (
        "import os, sys, servicio.bundle as b, servicio.__main__ as m\n"
        "seen = {}\n"
        "orig = b.check_index\n"
        "def spy(*a, **k):\n"
        "    seen['numpy_loaded'] = 'numpy' in sys.modules\n"
        "    seen['blas'] = os.environ.get('OPENBLAS_NUM_THREADS')\n"
        "    return orig(*a, **k)\n"
        "m.check_index = spy\n"
        "m.prepare()\n"
        "print(seen)\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NORMALIZER_", "OMP_", "OPENBLAS_", "MKL_"))}
    env.update({"PYTHONPATH": SRC, "MODEL_DIR": MODEL_DIR, "INDEX_DIR": synthetic_bundle,
                "RUNTIME_DIR": str(tmp_path / "rt"), "BASEMAPS_DIR": str(tmp_path / "bm"),
                "NORMALIZER_ONNX_THREADS": "3"})
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "'numpy_loaded': False" in r.stdout and "'blas': '3'" in r.stdout


def test_user_blas_setting_is_respected(tmp_path):
    code = ("import os, servicio.__main__ as m\n"
            "try:\n    m.prepare()\nexcept Exception:\n    pass\n"
            "print(os.environ['OPENBLAS_NUM_THREADS'])\n")
    env = dict(os.environ, PYTHONPATH=SRC, MODEL_DIR=MODEL_DIR, INDEX_DIR=str(tmp_path / "x"),
               RUNTIME_DIR=str(tmp_path / "rt"), NORMALIZER_ONNX_THREADS="3", OPENBLAS_NUM_THREADS="7")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == "7", r.stderr


@requires_model
def test_tampered_index_file_exits_3(tmp_path, synthetic_bundle):
    import shutil

    d = str(tmp_path / "copia")
    shutil.copytree(synthetic_bundle, d)
    meta = os.path.join(d, "index_meta.json")
    raw = open(meta, "rb").read()
    open(meta, "wb").write(raw.replace(b'"schema": 2', b'"schema": 3'))  # same size, valid JSON
    r = _run(tmp_path, INDEX_DIR=d, NORMALIZER_ONNX_THREADS="2")
    assert r.returncode == 3 and "sha256" in r.stderr and "Traceback" not in r.stderr


def test_file_url_exits_2_without_flag(tmp_path):
    r = _run(tmp_path, INDEX_URL="file:///tmp/x.tar.gz", INDEX_SHA256="a" * 64)
    assert r.returncode == 2 and "file://" in r.stderr


def test_readonly_index_dir_download_exits_3(tmp_path, synthetic_bundle):
    import tarfile

    archive = tmp_path / "b.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name in os.listdir(synthetic_bundle):
            tf.add(os.path.join(synthetic_bundle, name), arcname=name)
    import hashlib

    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    blocker = tmp_path / "bloqueo"
    blocker.write_text("un archivo, no un directorio", encoding="utf-8")  # makedirs under it fails (OSError)
    r = _run(tmp_path, INDEX_DIR=str(blocker / "indice"), INDEX_URL="file:///" + str(archive).replace("\\", "/"),
             INDEX_SHA256=sha, INDEX_ALLOW_FILE_URL="true")
    assert r.returncode == 3 and "INDEX_DIR" in r.stderr and "Traceback" not in r.stderr
