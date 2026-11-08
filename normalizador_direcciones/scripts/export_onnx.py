"""Export the serving files of the ONNX backends from an artifacts directory.

Writes, next to ``--out-dir`` (default: the directory that holds ``model.pt``):

* ``model.onnx``        fp32 graph ``ids -> emb, xy`` with a dynamic batch axis,
* ``model.int8.onnx``   dynamic int8 quantization of the same graph,
* ``onnx_meta.json``    max_len, d_out, coordinate scaling and the sha256 of ``model.pt``,
* ``catastro_emb.npy``  the document embeddings as fp16, readable without torch.

Needs torch (export only) and ``pip install 'cali-address[onnx]'``. It never modifies its inputs.
See docs/onnx.md.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cali_address.onnx_backend import (  # noqa: E402
    EMB_NPY_FILENAME, INT8_FILENAME, MIN_OPSET, ONNX_FILENAME, convert_embeddings, export_onnx, quantize_int8,
)
from cali_address.paths import model_path, resolve_artifacts_dir  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default=None,
                    help="directory with model.pt and catastro_emb.pt (env CALI_ARTIFACTS_DIR; default artifacts/)")
    ap.add_argument("--out-dir", default=None, help="where to write the files (default: next to model.pt)")
    ap.add_argument("--opset", type=int, default=MIN_OPSET, help=f"ONNX opset (>= {MIN_OPSET})")
    ap.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8", help="int8 weight type")
    ap.add_argument("--per-channel", action="store_true", help="per-channel int8 weight scales")
    args = ap.parse_args(argv)

    art = resolve_artifacts_dir(args.artifacts_dir)
    checkpoint = model_path("model.pt", art)
    embeddings = model_path("catastro_emb.pt", art)
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(checkpoint))
    try:
        meta = export_onnx(checkpoint, os.path.join(out_dir, ONNX_FILENAME), opset=args.opset)
        quantize_int8(os.path.join(out_dir, ONNX_FILENAME), os.path.join(out_dir, INT8_FILENAME),
                      weight_type=args.weight_type, per_channel=args.per_channel)
        convert_embeddings(embeddings, os.path.join(out_dir, EMB_NPY_FILENAME))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name in (ONNX_FILENAME, INT8_FILENAME, EMB_NPY_FILENAME, "onnx_meta.json"):
        path = os.path.join(out_dir, name)
        print(f"{path}  {os.path.getsize(path) / 1e6:.2f} MB")
    print(f"max_len={meta['max_len']} d_out={meta['d_out']} opset={meta['opset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
