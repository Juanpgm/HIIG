"""Optional ONNX Runtime serving backend (fp32 and int8) that does not need torch at run time.

What serving needs from the neural model, per query string (always ``role=0``, the query side):

* the L2-normalised retrieval embedding (``AddressEncoder.embed``), and
* the auxiliary coordinate head on the pooled representation (``coord_head(pool)``).

Both come out of ONE exported graph (``ids -> emb, xy``) with a dynamic batch axis and a fixed
``max_len`` token axis. The document embeddings (``catastro_emb.pt``, a torch pickle) are converted
once to ``catastro_emb.npy`` (fp16) so a torch-free environment can read them, and the cosine
top-k runs in numpy (:class:`NumpyDocIndex`).

Only the export and the one-off embedding conversion import torch (lazily); everything the serving
path touches (``OnnxEncoder``, ``NumpyDocIndex``, ``load_onnx_serving``) needs numpy + onnxruntime.
See ``docs/onnx.md``.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .paths import model_path, sha256_file
from .tokenizer import encode_batch

__all__ = [
    "BACKENDS",
    "ENV_BACKEND",
    "ENV_THREADS",
    "ENV_INDEX",
    "ONNX_FILENAME",
    "INT8_FILENAME",
    "META_FILENAME",
    "EMB_NPY_FILENAME",
    "OnnxUnavailableError",
    "resolve_backend",
    "export_onnx",
    "quantize_int8",
    "convert_embeddings",
    "NumpyDocIndex",
    "OnnxEncoder",
    "load_onnx_serving",
]

#: Serving backends. ``torch`` is the default and the reference.
BACKENDS = ("torch", "onnx", "onnx-int8")
ENV_BACKEND = "NORMALIZER_BACKEND"
#: Optional intra-op thread count of the ONNX session (default: let onnxruntime decide).
ENV_THREADS = "NORMALIZER_ONNX_THREADS"
#: Precision of the resident document matrix: ``fp32`` (default, fast single queries) or ``fp16`` (half the memory).
ENV_INDEX = "NORMALIZER_ONNX_INDEX"

ONNX_FILENAME = "model.onnx"
INT8_FILENAME = "model.int8.onnx"
META_FILENAME = "onnx_meta.json"
EMB_NPY_FILENAME = "catastro_emb.npy"

#: Rows per ONNX run. onnxruntime keeps its activation arena at the largest batch seen: 1,024 rows peak at
#: 1.1-1.4 GB of working set, 256 rows at 0.3-0.4 GB (and run faster); smaller than 256 gains no memory.
DEFAULT_ENCODE_BATCH = 256

#: Smallest opset that exports ``nn.TransformerEncoder`` (pre-norm, GELU) with fused LayerNorm.
MIN_OPSET = 17

_EXPORT_HINT = "create it with scripts/export_onnx.py (cali_address.onnx_backend.export_onnx)"


class OnnxUnavailableError(RuntimeError):
    """``onnxruntime`` is not installed but an ONNX backend was requested."""


def _import_onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise OnnxUnavailableError(
            "the 'onnx' serving backend needs onnxruntime, which is not installed: "
            "pip install 'cali-address[onnx]' (or: pip install onnxruntime). "
            "The default torch backend (NORMALIZER_BACKEND=torch) is unaffected."
        ) from exc
    return onnxruntime


def resolve_backend(value: str | None = None) -> str:
    """Parameter > ``NORMALIZER_BACKEND`` > ``"torch"``. Unknown values raise ``ValueError``."""
    chosen = value
    if chosen is None:
        chosen = os.environ.get(ENV_BACKEND)
    chosen = "torch" if chosen is None or not str(chosen).strip() else str(chosen).strip().lower()
    if chosen not in BACKENDS:
        raise ValueError(f"unknown serving backend {chosen!r}; valid values: {', '.join(BACKENDS)}")
    return chosen


# --------------------------------------------------------------------------------------------
# Export (torch needed here only)
# --------------------------------------------------------------------------------------------
def export_onnx(model_pt: str, out_path: str, opset: int = MIN_OPSET, role: int = 0) -> dict:
    """Export the serving graph of ``model_pt`` to ``out_path`` and write ``onnx_meta.json`` next to it.

    Graph: ``ids`` int64 ``[batch, max_len]`` -> ``emb`` fp32 ``[batch, d_out]`` (L2-normalised) and
    ``xy`` fp32 ``[batch, 2]`` (coordinate head, still standardised). ``batch`` is dynamic; ``max_len``
    is fixed by the checkpoint. Returns the metadata dict that was written.
    """
    if opset < MIN_OPSET:
        raise ValueError(f"opset must be >= {MIN_OPSET}, got {opset}")
    if not os.path.isfile(model_pt):
        raise FileNotFoundError(f"checkpoint not found: {model_pt} (expected the model.pt of the artifacts dir)")
    import torch
    import torch.nn.functional as F

    from .model import AddressEncoder

    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    max_len = int(cfg.get("max_len", 40))
    model = AddressEncoder(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], d_out=cfg["d_out"],
        dropout=0.0, n_comuna=ckpt["n_comuna"], n_barrio=ckpt["n_barrio"], max_len=max_len,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class _ServingGraph(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, ids):
            pooled = self.encoder.pool(ids, role)
            emb = F.normalize(self.encoder.proj(pooled), dim=-1)
            return emb, self.encoder.coord_head(pooled)

    graph = _ServingGraph(model).eval()
    sample = torch.from_numpy(encode_batch(["CL 5 # 38 - 10", "KR 26 H 3 # 73 - 10"], max_len).astype(np.int64))
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    # In eval mode nn.TransformerEncoderLayer takes a fused "fast path" kernel that has no ONNX
    # symbolic; disabling it exports the plain (and numerically equivalent) op-by-op graph.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(
                graph, (sample,), out_path, input_names=["ids"], output_names=["emb", "xy"],
                dynamic_axes={"ids": {0: "batch"}, "emb": {0: "batch"}, "xy": {0: "batch"}},
                opset_version=opset, dynamo=False,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    meta = {
        "format": 1,
        "graph": {"inputs": ["ids"], "outputs": ["emb", "xy"]},
        "max_len": max_len,
        "d_out": int(cfg["d_out"]),
        "role": int(role),
        "opset": int(opset),
        "xy_mean": [float(v) for v in np.asarray(ckpt["xy_mean"], dtype="float64").ravel()],
        "xy_std": [float(v) for v in np.asarray(ckpt["xy_std"], dtype="float64").ravel()],
        "model_sha256": sha256_file(model_pt),
    }
    with open(os.path.join(out_dir, META_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return meta


def quantize_int8(onnx_path: str, out_path: str, weight_type: str = "qint8", per_channel: bool = False) -> str:
    """Dynamic int8 quantization of the MatMul / Gemm weights (activations are quantized at run time).

    ``weight_type``: ``"qint8"`` (default, onnxruntime's default for dynamic quantization) or ``"quint8"``.
    Embedding tables, LayerNorm and the softmax stay fp32. Returns ``out_path``.
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"{onnx_path} not found; {_EXPORT_HINT} first")
    _import_onnxruntime()
    from onnxruntime.quantization import QuantType, quantize_dynamic

    kinds = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}
    if weight_type not in kinds:
        raise ValueError(f"weight_type must be one of {sorted(kinds)}, got {weight_type!r}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    quantize_dynamic(
        onnx_path, out_path, op_types_to_quantize=["MatMul", "Gemm"], weight_type=kinds[weight_type],
        per_channel=per_channel,
    )
    return out_path


def convert_embeddings(emb_pt: str, out_npy: str) -> str:
    """One-off conversion of ``catastro_emb.pt`` (torch pickle) to an fp16 ``.npy`` readable without torch."""
    if not os.path.isfile(emb_pt):
        raise FileNotFoundError(f"embeddings not found: {emb_pt}")
    import torch

    emb = torch.load(emb_pt, map_location="cpu")
    arr = emb.detach().cpu().numpy()
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16)
    np.save(out_npy, arr)
    return out_npy


# --------------------------------------------------------------------------------------------
# Search (numpy only)
# --------------------------------------------------------------------------------------------
class NumpyDocIndex:
    """Exact cosine top-k over unit-norm document embeddings, without torch.

    Candidates come from a chunked fp32 matmul against the fp16 documents (cast block by block, so
    the resident index stays fp16) plus ``argpartition``; the returned similarities are then
    recomputed as exact fp32 dot products of the unrounded query with the fp16 document, like
    ``AddressNormalizer.search_exact`` does, so they do not depend on the batch or the block size.
    Rows come back sorted by descending similarity (ties: lower document id first).

    ``fp32_cache=True`` keeps the documents as fp32 INSTEAD of fp16 (2x the resident memory, no second copy;
    the upcast is lossless so results are identical) and skips the per-call cast, which otherwise costs
    ~0.15 s per search over a 330k-document index and dominates single-address latency.
    """

    def __init__(self, doc_emb, block: int = 32768, fp32_cache: bool = False) -> None:
        docs = np.asarray(doc_emb)
        if docs.ndim != 2:
            raise ValueError(f"doc_emb must be a 2-D matrix, got shape {docs.shape}")
        if block < 1:
            raise ValueError(f"block must be >= 1, got {block}")
        self._block = int(block)
        self._docs32 = None
        if fp32_cache:
            self._docs32 = docs.astype(np.float32)
            self._docs = self._docs32
        else:
            self._docs = docs

    @property
    def shape(self) -> tuple[int, int]:
        return self._docs.shape

    @property
    def dtype(self):
        return self._docs.dtype

    @property
    def resident_bytes(self) -> int:
        """Bytes held by the document matrix (the only large allocation of the index)."""
        return int(self._docs.nbytes)

    def _block32(self, start: int, stop: int) -> np.ndarray:
        if self._docs32 is not None:
            return self._docs32[start:stop]
        return self._docs[start:stop].astype(np.float32)

    def _search_chunk(self, q: np.ndarray, k: int):
        n_docs = self._docs.shape[0]
        scores, ids = [], []
        for start in range(0, n_docs, self._block):
            stop = min(start + self._block, n_docs)
            s = q @ self._block32(start, stop).T
            width = stop - start
            if k < width:
                part = np.argpartition(s, width - k, axis=1)[:, width - k:]
                scores.append(np.take_along_axis(s, part, axis=1))
                ids.append(part + start)
            else:
                scores.append(s)
                ids.append(np.broadcast_to(np.arange(start, stop), s.shape))
        cand_s = np.concatenate(scores, axis=1)
        cand_i = np.concatenate(ids, axis=1)
        if cand_s.shape[1] > k:
            keep = np.argpartition(cand_s, cand_s.shape[1] - k, axis=1)[:, cand_s.shape[1] - k:]
            cand_i = np.take_along_axis(cand_i, keep, axis=1)
        exact = (q[:, None, :] * self._docs[cand_i].astype(np.float32)).sum(-1)
        order = np.lexsort((cand_i, -exact), axis=1)
        return np.take_along_axis(exact, order, axis=1), np.take_along_axis(cand_i, order, axis=1).astype(np.int64)

    def search_exact(self, q_emb, k: int = 20, chunk: int = 512):
        """``(scores fp32 [n, k'], ids int64 [n, k'])`` with ``k' = min(k, n_docs)``."""
        q = np.asarray(q_emb, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self._docs.shape[1]:
            raise ValueError(f"query dimension mismatch: expected (n, {self._docs.shape[1]}), got {q.shape}")
        k_eff = max(0, min(int(k), self._docs.shape[0]))
        if q.shape[0] == 0 or k_eff == 0:
            return np.zeros((q.shape[0], k_eff), dtype=np.float32), np.zeros((q.shape[0], k_eff), dtype=np.int64)
        out_s, out_i = [], []
        for i in range(0, q.shape[0], chunk):
            s, ix = self._search_chunk(q[i:i + chunk], k_eff)
            out_s.append(s)
            out_i.append(ix)
        return np.concatenate(out_s), np.concatenate(out_i)


# --------------------------------------------------------------------------------------------
# Encoder (onnxruntime)
# --------------------------------------------------------------------------------------------
class OnnxEncoder:
    """Query encoder backed by an ONNX Runtime CPU session.

    ``embed`` mirrors ``AddressNormalizer.encode_queries_exact`` (fp32, unit norm); ``embed_with_xy``
    also returns the auxiliary coordinate in metres (``AddressNormalizer.predict_xy``) from the same pass.
    """

    def __init__(self, onnx_path: str, meta_path: str | None = None, threads: int | None = None) -> None:
        ort = _import_onnxruntime()
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"{onnx_path} not found; {_EXPORT_HINT}")
        meta_path = meta_path or os.path.join(os.path.dirname(os.path.abspath(onnx_path)), META_FILENAME)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"{meta_path} not found; {_EXPORT_HINT}")
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.max_len = int(self.meta["max_len"])
        self.d_out = int(self.meta["d_out"])
        self.xy_mean = np.asarray(self.meta["xy_mean"], dtype="float64")
        self.xy_std = np.asarray(self.meta["xy_std"], dtype="float64")
        if threads is None:
            raw = os.environ.get(ENV_THREADS, "").strip()
            threads = int(raw) if raw else 0
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        if threads:
            options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])

    def embed_with_xy(self, texts, batch: int = DEFAULT_ENCODE_BATCH):
        n = len(texts)
        emb = np.empty((n, self.d_out), dtype=np.float32)
        xy = np.empty((n, 2), dtype=np.float64)
        for i in range(0, n, batch):
            ids = encode_batch(texts[i:i + batch], self.max_len).astype(np.int64)
            e, c = self.session.run(["emb", "xy"], {"ids": ids})
            e = e.astype(np.float32, copy=False)
            emb[i:i + batch] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            xy[i:i + batch] = c.astype(np.float64) * self.xy_std + self.xy_mean
        return emb, xy

    def embed(self, texts, batch: int = DEFAULT_ENCODE_BATCH) -> np.ndarray:
        return self.embed_with_xy(texts, batch=batch)[0]

    def predict_xy(self, texts, batch: int = DEFAULT_ENCODE_BATCH) -> np.ndarray:
        return self.embed_with_xy(texts, batch=batch)[1]


def load_onnx_serving(artifacts_dir: str | None, backend: str, threads: int | None = None):
    """``(OnnxEncoder, NumpyDocIndex, meta)`` for ``backend`` in {``onnx``, ``onnx-int8``}.

    The ONNX files and ``catastro_emb.npy`` sit next to the ``model.pt`` the artifacts directory resolves
    to (an experiment that inherits the production model reads them from production). Raises a clear
    error for a missing file, for an export older than ``model.pt`` ("stale") and for an embedding
    matrix whose width differs from the model's.
    """
    if backend not in ("onnx", "onnx-int8"):
        raise ValueError(f"not an ONNX backend: {backend!r}")
    base = os.path.dirname(model_path("model.pt", artifacts_dir))
    onnx_file = os.path.join(base, ONNX_FILENAME if backend == "onnx" else INT8_FILENAME)
    npy_file = os.path.join(base, EMB_NPY_FILENAME)
    encoder = OnnxEncoder(onnx_file, threads=threads)
    if not os.path.isfile(npy_file):
        raise FileNotFoundError(
            f"{npy_file} not found; {_EXPORT_HINT} (it converts catastro_emb.pt for torch-free serving)")
    checkpoint = os.path.join(base, "model.pt")
    if os.path.isfile(checkpoint) and sha256_file(checkpoint) != encoder.meta.get("model_sha256"):
        raise ValueError(
            f"stale ONNX export: {onnx_file} was made from a different model.pt; re-run scripts/export_onnx.py")
    doc_emb = np.load(npy_file)
    if doc_emb.ndim != 2 or doc_emb.shape[1] != encoder.d_out:
        raise ValueError(
            f"{npy_file} has shape {doc_emb.shape}; the model embeds into {encoder.d_out} dimensions")
    precision = os.environ.get(ENV_INDEX, "").strip().lower() or "fp32"
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"{ENV_INDEX} must be 'fp32' or 'fp16', got {precision!r}")
    return encoder, NumpyDocIndex(doc_emb, fp32_cache=(precision == "fp32")), encoder.meta
