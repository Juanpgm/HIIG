#!/usr/bin/env python3
"""Build the index bundle the service mounts at INDEX_DIR, from data the user supplies.

The bundle is derived from the reference data (cadastre / nomenclature) and is NEVER committed or baked
into the public image. Input: a directory with at least ``catastro_docs.parquet`` and ``index_meta.json``
(the output of ``normalizador_direcciones/scripts/build_index.py``). Embeddings are taken, in order, from:

1. ``catastro_emb.npy`` in the source directory (exact copy), else
2. ``catastro_emb.pt`` in the source directory (converted; needs torch, only for this step), else
3. computed here with ``modelo/model.doc.onnx`` (the document side of the same model; torch-free).

Usage (from this folder)::

    python scripts/preparar_indice.py --origen <dir> --destino ./indice [--basemaps <dir>] [--precalentar]
                                      [--empaquetar indice.tar.gz] [--enlazar]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from servicio.bundle import (  # noqa: E402
    INDEX_OPTIONAL,
    INDEX_REQUIRED,
    MANIFEST_FILENAME,
    BundleError,
    build_manifest,
    check_index,
    check_model_dir,
    sha256_file,
)

DOC_GRAPH = "model.doc.onnx"
DOC_META = "model.doc.meta.json"
#: Files copied from the source when present (embeddings are handled separately).
COPY_FILES = tuple(n for n in INDEX_REQUIRED + INDEX_OPTIONAL if n != "catastro_emb.npy")


def _place(src: str, dst: str, link: bool) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _convert_pt(pt_path: str, out_npy: str) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "catastro_emb.pt necesita torch para convertirse. Instale torch (CPU) solo para este paso, "
            "o borre el .pt para que el script calcule los embeddings con modelo/model.doc.onnx."
        ) from None
    from cali_address.onnx_backend import convert_embeddings

    convert_embeddings(pt_path, out_npy)


def embed_with_doc_graph(docs_path: str, model_dir: str, out_npy: str, threads: int | None = None,
                         batch: int = 256, progress=None) -> int:
    """Embed every ``direccion.upper()`` of the documents with the document-side ONNX graph (fp16 output)."""
    import numpy as np
    import onnxruntime as ort
    import pandas as pd

    from cali_address.tokenizer import encode_batch

    graph = os.path.join(model_dir, DOC_GRAPH)
    meta_path = os.path.join(model_dir, DOC_META)
    if not (os.path.isfile(graph) and os.path.isfile(meta_path)):
        raise SystemExit(f"faltan {DOC_GRAPH} / {DOC_META} en {model_dir}: no se pueden calcular los embeddings")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(model_dir, "onnx_meta.json"), "r", encoding="utf-8") as fh:
        serving_meta = json.load(fh)
    if meta.get("model_sha256") != serving_meta.get("model_sha256") or int(meta.get("role", -1)) != 1:
        raise SystemExit(f"{DOC_GRAPH} no corresponde al modelo de servicio (model_sha256/role distintos)")
    max_len, d_out = int(meta["max_len"]), int(meta["d_out"])
    texts = pd.read_parquet(docs_path, columns=["direccion"])["direccion"].fillna("").astype(str).str.upper().tolist()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if threads:
        opts.intra_op_num_threads = int(threads)
    sess = ort.InferenceSession(graph, opts, providers=["CPUExecutionProvider"])
    out = np.empty((len(texts), d_out), dtype=np.float16)
    for i in range(0, len(texts), batch):
        ids = encode_batch(texts[i:i + batch], max_len).astype(np.int64)
        emb, _xy = sess.run(["emb", "xy"], {"ids": ids})
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        out[i:i + batch] = emb.astype(np.float16)
        if progress is not None:
            progress(min(i + batch, len(texts)), len(texts))
    partial = out_npy + ".partial.npy"
    np.save(partial, out)
    os.replace(partial, out_npy)
    return len(texts)


def prewarm(dest: str, basemaps: str | None) -> list[str]:
    """Build the startup caches inside the bundle (interpolation index, catalogue keys, gazetteer)."""
    from cali_address import catalog, interp

    done = []
    if interp.warm_cache(dest):
        done.append("interp_index_v1")
    if catalog.warm_cache(dest):
        done.append("catalog_keys_v1")
    if basemaps:
        from cali_address.gazetteer import Gazetteer

        Gazetteer.load(basemaps, cache_path=os.path.join(dest, "gazetteer.pkl"))
        done.append("gazetteer")
    return done


def package(dest: str, archive: str) -> str:
    with tarfile.open(archive, "w:gz") as tf:
        for name in sorted(os.listdir(dest)):
            if name in INDEX_REQUIRED + INDEX_OPTIONAL + (MANIFEST_FILENAME,):
                tf.add(os.path.join(dest, name), arcname=name)
    return sha256_file(archive)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--origen", required=True, help="directory with catastro_docs.parquet + index_meta.json")
    ap.add_argument("--destino", required=True, help="output bundle directory (INDEX_DIR)")
    ap.add_argument("--modelo", default=os.path.join(ROOT, "modelo"), help="model directory (default: ./modelo)")
    ap.add_argument("--basemaps", help="IDESC basemaps directory, to pre-build gazetteer.pkl")
    ap.add_argument("--precalentar", action="store_true", help="pre-build the startup caches in the bundle")
    ap.add_argument("--recalcular-embeddings", action="store_true",
                    help="ignore embeddings in --origen and compute them with modelo/model.doc.onnx")
    ap.add_argument("--hilos", type=int, default=0, help="ONNX threads for the embedding step (0 = auto)")
    ap.add_argument("--enlazar", action="store_true", help="hard-link instead of copying (same disk only)")
    ap.add_argument("--empaquetar", help="also write a .tar.gz of the bundle and print its sha256")
    args = ap.parse_args(argv)

    src, dest = os.path.abspath(args.origen), os.path.abspath(args.destino)
    if src == dest:
        raise SystemExit("--origen y --destino deben ser directorios distintos")
    for name in ("catastro_docs.parquet", "index_meta.json"):
        if not os.path.isfile(os.path.join(src, name)):
            raise SystemExit(f"falta {name} en --origen {src}")
    try:
        meta = check_model_dir(args.modelo, "onnx")
    except BundleError as exc:
        raise SystemExit(str(exc)) from None
    os.makedirs(dest, exist_ok=True)
    stale = os.path.join(dest, MANIFEST_FILENAME)
    if os.path.exists(stale):
        os.remove(stale)  # the manifest is written last: its presence means a complete bundle

    for name in COPY_FILES:
        path = os.path.join(src, name)
        if os.path.isfile(path):
            _place(path, os.path.join(dest, name), args.enlazar)

    emb_out = os.path.join(dest, "catastro_emb.npy")
    model_verified = False
    started = time.time()
    if not args.recalcular_embeddings and os.path.isfile(os.path.join(src, "catastro_emb.npy")):
        _place(os.path.join(src, "catastro_emb.npy"), emb_out, args.enlazar)
        source = "copied-npy"
    elif not args.recalcular_embeddings and os.path.isfile(os.path.join(src, "catastro_emb.pt")):
        _convert_pt(os.path.join(src, "catastro_emb.pt"), emb_out)
        source = "converted-pt"
    else:
        def progress(done, total):
            if done == total or done % (256 * 200) == 0:
                print(f"  embeddings {done}/{total}", file=sys.stderr, flush=True)

        embed_with_doc_graph(os.path.join(src, "catastro_docs.parquet"), args.modelo, emb_out,
                             threads=args.hilos or None, progress=progress)
        source = "onnx-doc-graph"
        model_verified = True
    checkpoint = os.path.join(src, "model.pt")
    if source != "onnx-doc-graph":
        if os.path.isfile(checkpoint):
            if sha256_file(checkpoint) != meta.get("model_sha256"):
                raise SystemExit(
                    "los embeddings de --origen son de otro modelo (model.pt distinto al de modelo/onnx_meta.json). "
                    "Use --recalcular-embeddings para calcularlos con este modelo.")
            model_verified = True
        else:
            print("aviso: --origen no trae model.pt; no se puede comprobar que sus embeddings sean de este modelo. "
                  "Si hay dudas use --recalcular-embeddings.", file=sys.stderr)

    try:
        for line in check_index(dest, meta):
            if MANIFEST_FILENAME not in line:
                print(f"aviso: {line}", file=sys.stderr)
    except BundleError as exc:
        raise SystemExit(f"el paquete no es valido: {exc}") from None

    warmed = prewarm(dest, args.basemaps) if args.precalentar else []
    manifest = build_manifest(dest, meta)
    manifest.update({"emb_source": source, "model_verified": model_verified, "prewarmed": warmed,
                     "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with open(stale + ".partial", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(stale + ".partial", stale)
    summary = {"destino": dest, "n_docs": manifest["n_docs"], "emb_source": source,
               "model_verified": model_verified, "prewarmed": warmed, "seconds": round(time.time() - started, 1)}
    if args.empaquetar:
        summary["archivo"] = os.path.abspath(args.empaquetar)
        summary["sha256"] = package(dest, args.empaquetar)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
