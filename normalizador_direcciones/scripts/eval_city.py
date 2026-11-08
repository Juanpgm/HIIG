"""Evaluate a model trained on a new city's index with held-out noisy queries (aggregates only).

    python scripts/eval_city.py --index-dir runs/micity/index [--n-docs 1000 --variants 2] [--out eval_city.json]

The index directory must hold ``model.pt`` and ``catastro_emb.pt`` (``scripts/train.py --out-dir`` writes them).
Reported, all on the same held-out queries (noise seed never used in training):

* ``chance``: recall@k of a random ranking (k / documents);
* ``encoder_untrained`` / ``encoder_trained``: recall@k of the bare encoder before and after training;
* ``pipeline``: the full ``normalize_batch`` path (encoder + rerank + cutoff): recall@k, coverage, parcel / block
  precision with Wilson intervals, ECE of ``confidence``, precision per confidence band;
* ``rules_only``: the same path without the model (exact key + fuzzy candidates), the no-learning baseline;
* ``trained_minus_untrained_recall@1``: paired bootstrap interval;
* ``val_docs``: pipeline recall@1 on documents of the training validation split (never trained as queries).

Exit codes: 0 ok, 1 missing artifacts, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def evaluate(index_dir: str, n_docs: int = 1000, variants: int = 2, seed: int | None = None,
             device: str = "cpu") -> dict:
    import torch

    from cali_address.city_eval import (
        EVAL_SEED, chance_recall, encoder_recall, eval_queries, paired_bootstrap, pipeline_metrics,
    )
    from cali_address.inference import AddressNormalizer
    from cali_address.model import AddressEncoder
    from cali_address.train import load_checkpoint

    seed = EVAL_SEED if seed is None else seed
    started = time.time()
    docs = pd.read_parquet(os.path.join(index_dir, "catastro_docs.parquet"))
    data = np.load(os.path.join(index_dir, "train_data.npz"), allow_pickle=True)
    d_tokens = data["d_tokens"]
    val_mask = np.zeros(len(docs), dtype=bool)
    val_mask[np.unique(data["q_doc"][data["q_is_val"]])] = True
    texts, truth, is_val = eval_queries(docs, n_docs=n_docs, variants=variants, seed=seed, val_mask=val_mask)

    model, ckpt = load_checkpoint(os.path.join(index_dir, "model.pt"), device=device, dropout=0.0)
    trained, trained_hit1 = encoder_recall(model, texts, truth, d_tokens, device=device)
    cfg = ckpt["config"]
    torch.manual_seed(seed)
    untrained_model = AddressEncoder(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], d_out=cfg["d_out"], dropout=0.0,
        n_comuna=ckpt["n_comuna"], n_barrio=ckpt["n_barrio"], max_len=int(cfg.get("max_len", 40))).to(device)
    untrained, untrained_hit1 = encoder_recall(untrained_model, texts, truth, d_tokens, device=device)

    normalizer = AddressNormalizer(index_dir, device=device, docs=docs)
    pipe, rows = pipeline_metrics(normalizer, texts, truth, docs)
    rules, _ = pipeline_metrics(normalizer, texts, truth, docs, use_model=False)
    val_rows = rows[is_val]
    return {
        "index_dir_name": os.path.basename(os.path.normpath(index_dir)),
        "n_docs_index": int(len(docs)),
        "queries": int(len(texts)), "query_docs": int(len(np.unique(truth))), "variants": variants, "seed": seed,
        "model": {k: cfg[k] for k in ("d_model", "n_layers", "n_heads", "d_out", "epochs", "batch_size")},
        "chance": chance_recall(len(docs)),
        "encoder_untrained": untrained,
        "encoder_trained": trained,
        "trained_minus_untrained_recall@1": paired_bootstrap(untrained_hit1, trained_hit1, seed=seed),
        "pipeline": pipe,
        "rules_only": rules,
        "val_docs": {"queries": int(len(val_rows)),
                     "encoder_trained_recall@1": float(trained_hit1[is_val].mean()) if len(val_rows) else None,
                     "pipeline_recall@1": float(val_rows["hit@1"].mean()) if len(val_rows) else None},
        "seconds": round(time.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=1000, help="documents sampled for queries")
    ap.add_argument("--variants", type=int, default=2, help="noisy variants per sampled document")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the metrics JSON here")
    args = ap.parse_args(argv)
    if args.n_docs < 1 or args.variants < 1:
        print("error: --n-docs and --variants must be >= 1", file=sys.stderr)
        return 2
    missing = [n for n in ("model.pt", "catastro_emb.pt", "catastro_docs.parquet", "train_data.npz")
               if not os.path.isfile(os.path.join(args.index_dir, n))]
    if missing:
        print(f"error: {args.index_dir} lacks {', '.join(missing)} (train first: scripts/train.py --out-dir)",
              file=sys.stderr)
        return 1
    result = evaluate(args.index_dir, args.n_docs, args.variants, args.seed, args.device)
    text = json.dumps(result, indent=2, ensure_ascii=False, default=float)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
