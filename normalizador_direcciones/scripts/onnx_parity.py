"""Parity of the ONNX backends against the torch reference (CPU).

For one serving directory (model.pt, catastro_emb.pt, model.onnx, model.int8.onnx, catastro_emb.npy) compares,
on invented strings and on N public document strings (``direccion`` of ``catastro_docs.parquet``):

* cosine between the torch query embedding and the ONNX one (min / mean),
* identity of the top-1 document (exact fp32 search over the same fp16 index),
* the largest difference of the predicted coordinate, in metres.

Public cadastral strings only; no split or frozen data is read. Writes ``--out`` (JSON) when given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cali_address.inference import AddressNormalizer  # noqa: E402
from cali_address.paths import resolve_artifacts_dir, shared_path  # noqa: E402

INVENTED = [
    "CL 5 # 38 - 10", "KR 26 H 3 # 73 - 10", "calle 2 No 10-12", "CRA 7 Nº 12 – 45 ÁÉÍÓÚ Ñ", "", "   ",
    "AV 3 N # 45 - 67 APTO 301", "TV 25 # 4 - 40", "xyz", "CL 1 # 10 - 10 " + "X" * 60,
]


def _stats(name: str, torch_norm: AddressNormalizer, onnx_norm: AddressNormalizer, texts: list, strings) -> dict:
    q_t = torch_norm.encode_queries_exact(texts).cpu().numpy()
    q_o = onnx_norm.encode_queries_exact(texts)
    cos = (q_t * q_o).sum(1) / (np.linalg.norm(q_t, axis=1) * np.linalg.norm(q_o, axis=1))
    top_t = torch_norm.search_exact(torch_norm.encode_queries_exact(texts), k=1)[1][:, 0]
    top_o = onnx_norm.search_exact(q_o, k=1)[1][:, 0]
    dxy = np.abs(torch_norm.predict_xy(texts) - onnx_norm.predict_xy(texts)).max()
    return {
        "set": name, "n": len(texts), "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
        "top1_identical": float((top_t == top_o).mean()), "top1_differ": int((top_t != top_o).sum()),
        # an index can hold several documents with the same address string (identical embeddings): a
        # different id among them is a tie, not a disagreement
        "top1_same_address_string": float((strings[top_t] == strings[top_o]).mean()),
        "xy_max_abs_diff_m": float(dxy),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", required=True, help="serving directory with the ONNX files next to model.pt")
    ap.add_argument("--n", type=int, default=2000, help="public document strings to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    art = resolve_artifacts_dir(args.artifacts_dir)

    docs = pd.read_parquet(shared_path("catastro_docs.parquet", art))
    sample = docs["direccion"].astype(str).sample(n=min(args.n, len(docs)), random_state=args.seed).tolist()
    torch_norm = AddressNormalizer(art, device="cpu", backend="torch")
    strings = docs["direccion"].astype(str).to_numpy()
    result = {"artifacts_dir": art, "n_docs": int(len(docs)), "backends": {}}
    for backend in ("onnx", "onnx-int8"):
        onnx_norm = AddressNormalizer(art, device="cpu", backend=backend)
        result["backends"][backend] = [_stats("invented", torch_norm, onnx_norm, INVENTED, strings),
                                       _stats("public_docs", torch_norm, onnx_norm, sample, strings)]
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
