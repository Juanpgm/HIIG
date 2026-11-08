"""Evaluation of an index trained for a new city (aggregates only, no address text leaves a function).

Used by ``scripts/eval_city.py`` and the synthetic-city end-to-end test. Ground truth comes from the index itself:
held-out noisy variants of the documents (``augment.corrupt_many`` with a seed that training never uses), so the
figures measure how well the encoder + reranker recover a known document from a corrupted spelling. That proves the
pipeline learns; it is NOT a measure of accuracy on real user input (see documentacion/12).

* :func:`eval_queries`      ``(texts, truth doc ids, is_val_doc)`` of held-out noisy queries.
* :func:`encoder_recall`    recall@k of the bare encoder (cosine top-k), trained or untrained.
* :func:`pipeline_metrics`  the full ``AddressNormalizer.normalize_batch`` path: recall@k after reranking, coverage
  at the cutoff, parcel and block precision of emitted rows with Wilson intervals, ECE of ``confidence``.
* :func:`paired_bootstrap`  percentile interval of the difference of two per-row indicator means.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from .augment import corrupt_many
from .dataset import query_text_for
from .model import encode_batch

__all__ = ["EVAL_SEED", "eval_queries", "encoder_recall", "pipeline_metrics", "ece", "paired_bootstrap",
           "chance_recall", "wilson_interval"]

#: Seed of the held-out noise; training uses ``dataset.SEED`` (42) for its variants.
EVAL_SEED = 20261031
KS = (1, 5, 20)


def eval_queries(docs: pd.DataFrame, n_docs: int = 1000, variants: int = 2, seed: int = EVAL_SEED,
                 val_mask: np.ndarray | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Noisy variants of ``n_docs`` documents sampled without replacement (all of them when fewer).

    Returns ``(texts, doc_ids, is_val)``; ``is_val`` marks documents of the training validation split (their
    queries were never trained on) when ``val_mask`` is given, else all False.
    """
    if variants < 1:
        raise ValueError("variants must be >= 1")
    total = len(docs)
    if total == 0:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    gen = np.random.default_rng(seed)
    chosen = np.sort(gen.choice(total, size=min(n_docs, total), replace=False))
    rng = random.Random(seed)
    texts, ids = [], []
    canon = docs["direccion"].astype(str).tolist()
    for d in chosen:
        for variant in corrupt_many(canon[d], variants, rng):
            texts.append(variant)
            ids.append(int(d))
    ids_arr = np.asarray(ids, dtype=np.int64)
    is_val = val_mask[ids_arr] if val_mask is not None and len(ids_arr) else np.zeros(len(ids_arr), dtype=bool)
    return texts, ids_arr, np.asarray(is_val, dtype=bool)


def chance_recall(n_docs: int, ks=KS) -> dict:
    """Recall@k of a uniformly random ranking: ``k / n_docs``."""
    return {f"recall@{k}": (min(k, n_docs) / n_docs if n_docs else float("nan")) for k in ks}


def encoder_recall(model, texts: list[str], truth: np.ndarray, d_tokens: np.ndarray, device: str = "cpu",
                   ks=KS) -> tuple[dict, np.ndarray]:
    """``(recall@k dict, per-row top-1 hit array)`` of the bare encoder over the whole index."""
    import torch

    from .train import embed_documents

    if not len(texts):
        return {f"recall@{k}": float("nan") for k in ks}, np.zeros(0, dtype=bool)
    q_tokens = encode_batch([query_text_for(t)[0] for t in texts])[:, : model.max_len].astype(np.int64)
    doc_emb = embed_documents(model, d_tokens, device=device).float()
    kmax = min(max(ks), doc_emb.shape[0])
    hits = {k: np.zeros(len(texts), dtype=bool) for k in ks}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            ids = torch.from_numpy(q_tokens[i:i + 512]).to(device)
            emb = torch.nn.functional.normalize(model.embed(ids, role=0).float(), dim=-1)
            top = (emb @ doc_emb.T).topk(kmax, dim=-1).indices.cpu().numpy()
            t = truth[i:i + 512, None]
            for k in ks:
                hits[k][i:i + 512] = (top[:, :k] == t).any(axis=1)
    return {f"recall@{k}": float(hits[k].mean()) for k in ks}, hits[1]


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins (same definition as ``scripts/fit_reliability.py``)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    return float(sum(len(ix) / len(p) * abs(p[ix].mean() - y[ix].mean())
                     for ix in np.array_split(order, bins) if len(ix)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (same formula as ``calibration_report.wilson_interval``; kept local so this module only
    needs the core package); ``(nan, nan)`` when ``n == 0``."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"impossible counts k={k} n={n}")
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _share_ci(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n if n else float("nan")), "ci95": [lo, hi]}


def pipeline_metrics(normalizer, texts: list[str], truth: np.ndarray, docs: pd.DataFrame, k: int = 20,
                     use_model: bool = True, bands=(0.9,)) -> tuple[dict, pd.DataFrame]:
    """Metrics of ``normalizer.normalize_batch`` (rerank on) against the true documents.

    Returns ``(metrics, per-row indicators)``; the per-row frame holds only booleans and scores (no text).
    """
    out = normalizer.normalize_batch(texts, k=k, use_model=use_model, rerank=True)
    npn = docs["numero_predial_nacional"].astype(str).to_numpy()
    mz = docs["manzana"].astype(str).to_numpy()
    topk = out["topk_doc_ids"].tolist()
    top1 = np.array([t[0] if len(t) else -1 for t in topk], dtype=np.int64)
    rows = pd.DataFrame({
        "matched": out["matched"].astype(bool).to_numpy(),
        "confidence": out["confidence"].astype(float).to_numpy(),
        "top1_parcel": np.array([t >= 0 and npn[t] == npn[g] for t, g in zip(top1, truth)], dtype=bool),
        "top1_block": np.array([t >= 0 and mz[t] == mz[g] for t, g in zip(top1, truth)], dtype=bool),
    })
    sel = out["selected_doc_id"].astype(int).to_numpy()
    rows["emitted_parcel"] = [bool(m) and s >= 0 and npn[s] == npn[g] for m, s, g in zip(rows["matched"], sel, truth)]
    rows["emitted_block"] = [bool(m) and s >= 0 and mz[s] == mz[g] for m, s, g in zip(rows["matched"], sel, truth)]
    for kk in KS:
        rows[f"hit@{kk}"] = [int(g) in t[:kk] for t, g in zip(topk, truth)]
    n = len(rows)
    m = int(rows["matched"].sum())
    metrics = {
        "rows": n,
        "threshold": float(normalizer.threshold),
        **{f"recall@{kk}": float(rows[f"hit@{kk}"].mean()) if n else float("nan") for kk in KS},
        "coverage": _share_ci(m, n),
        "parcel_precision": _share_ci(int(rows["emitted_parcel"].sum()), m),
        "block_precision": _share_ci(int(rows["emitted_block"].sum()), m),
        "lost_correct": int((~rows["matched"] & rows["top1_parcel"]).sum()),
        "ece_confidence_vs_top1_parcel": ece(rows["confidence"], rows["top1_parcel"]),
        "bands": {},
    }
    edges = [float(normalizer.threshold), *sorted(b for b in bands if b > normalizer.threshold)]
    labels = [f"[{lo:.2f}, {hi:.2f})" for lo, hi in zip(edges, edges[1:] + [1.01])]
    emitted = rows[rows["matched"]]
    for (lo, hi), label in zip(zip(edges, edges[1:] + [1.01]), labels):
        part = emitted[(emitted["confidence"] >= lo) & (emitted["confidence"] < hi)]
        metrics["bands"][label] = _share_ci(int(part["emitted_parcel"].sum()), len(part))
    return metrics, rows


def paired_bootstrap(a, b, resamples: int = 2000, seed: int = EVAL_SEED, level: float = 0.95) -> dict:
    """Mean of ``b`` minus mean of ``a`` (per-row indicators of the SAME rows) with a percentile interval."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired_bootstrap needs arrays of equal length")
    if len(a) == 0:
        return {"diff": float("nan"), "ci": [float("nan"), float("nan")], "resamples": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return {"diff": float(b.mean() - a.mean()),
            "ci": [float(np.percentile(diffs, tail)), float(np.percentile(diffs, 100 - tail))],
            "resamples": int(resamples)}
