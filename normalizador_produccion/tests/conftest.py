"""Shared fixtures. Every address here is an invented street-grid string (no real data)."""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
MODEL_DIR = os.path.join(ROOT, "modelo")

import cali_address  # noqa: E402

# The tests must exercise THIS folder's copy, not another installed cali_address.
assert os.path.abspath(cali_address.__file__).startswith(SRC), cali_address.__file__

HAS_MODEL = os.path.isfile(os.path.join(MODEL_DIR, "model.onnx")) and os.path.isfile(
    os.path.join(MODEL_DIR, "model.doc.onnx"))
requires_model = pytest.mark.skipif(not HAS_MODEL, reason="modelo/model.onnx or model.doc.onnx absent")

VIAS = ("CALLE", "CARRERA", "AVENIDA", "DIAGONAL", "TRANSVERSAL")


def synthetic_addresses(n: int) -> list[str]:
    out = []
    i = 0
    while len(out) < n:
        via = VIAS[i % len(VIAS)]
        out.append(f"{via} {10 + i % 70} # {5 + (i * 7) % 90} - {10 + (i * 13) % 80}")
        i += 1
    return out


def build_synthetic_bundle(dest: str, n: int = 120, with_manifest: bool = True) -> str:
    """A tiny but structurally complete index bundle (invented addresses and coordinates)."""
    import numpy as np
    import pandas as pd

    from cali_address.geo import to_metric
    from servicio.bundle import build_manifest

    os.makedirs(dest, exist_ok=True)
    addrs = synthetic_addresses(n)
    rng = np.random.default_rng(0)
    lon = -76.53 + rng.uniform(-0.03, 0.03, n)
    lat = 3.42 + rng.uniform(-0.03, 0.03, n)
    x, y = to_metric(lon, lat)
    npn = [f"76001{(100 + i % 20):02d}{i % 99:04d}{i:019d}"[:30] for i in range(n)]
    docs = pd.DataFrame({
        "direccion": addrs,
        "numero_predial_nacional": npn,
        "manzana": [p[:17] for p in npn],
        "comuna": [p[9:11] for p in npn],
        "barrio": [p[11:13] for p in npn],
        "barrio_code": [p[9:13] for p in npn],
        "centroid_lon": lon,
        "centroid_lat": lat,
        "n_predios": np.ones(n, dtype="int64"),
        "manzana_nunique": np.ones(n, dtype="int64"),
        "objectid": np.arange(1, n + 1, dtype="int64"),
        "x_m": x,
        "y_m": y,
        "doc_id": np.arange(n, dtype="int64"),
    })
    docs.to_parquet(os.path.join(dest, "catastro_docs.parquet"), index=False)
    with open(os.path.join(dest, "index_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": 2, "n_docs": n, "synthetic": True}, fh)
    if HAS_MODEL:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from preparar_indice import embed_with_doc_graph

        embed_with_doc_graph(os.path.join(dest, "catastro_docs.parquet"), MODEL_DIR,
                             os.path.join(dest, "catastro_emb.npy"))
    else:
        emb = rng.normal(size=(n, 256)).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        np.save(os.path.join(dest, "catastro_emb.npy"), emb.astype("float16"))
    if with_manifest:
        with open(os.path.join(MODEL_DIR, "onnx_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(os.path.join(dest, "indice_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(build_manifest(dest, meta), fh)
    return dest


@pytest.fixture(scope="session")
def synthetic_bundle(tmp_path_factory) -> str:
    return build_synthetic_bundle(str(tmp_path_factory.mktemp("indice")))


@pytest.fixture
def model_meta() -> dict:
    with open(os.path.join(MODEL_DIR, "onnx_meta.json"), encoding="utf-8") as fh:
        return json.load(fh)
