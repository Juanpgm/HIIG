"""Build ``artifacts/catastro_own_centroids.parquet``: the centroid of the parcel each document reports.

``catastro_docs.parquet`` has one row per ADDRESS. Its NPN is the first parcel's (lowest OBJECTID)
but its ``centroid_*`` is the median over every parcel that shares the address, hundreds of metres
away for some addresses. The service reports the parcel's own point (see ``docs/API.md``); this
script computes it once, at build time, so the deployed image does not need the 300 MB
``catastro.parquet``.

The output is a NEW file next to the documents (``catastro_docs.parquet`` is never modified, so
``catastro_emb.pt`` stays valid and nothing is re-embedded): one row per document in document order
with ``objectid``, ``numero_predial_nacional``, ``own_lon``, ``own_lat``. The runtime
(``cali_address.inference.load_own_centroids``) only accepts it when those rows match the documents.

    python scripts/build_own_centroids.py [--artifacts-dir DIR] [--catastro PATH] [--out PATH]

Exit codes: 0 written, 2 usage / missing or unusable input (nothing is written).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.dataset import own_centroid_table  # noqa: E402
from cali_address.paths import DEFAULT_ARTIFACTS_DIR, OWN_CENTROIDS_FILENAME  # noqa: E402

DOCS_FILENAME = "catastro_docs.parquet"


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR, help="directory with catastro_docs.parquet")
    ap.add_argument("--catastro", default=None, help="catastro.parquet (default: <artifacts-dir>/catastro.parquet)")
    ap.add_argument("--out", default=None, help=f"output file (default: <artifacts-dir>/{OWN_CENTROIDS_FILENAME})")
    args = ap.parse_args(argv)

    docs_path = os.path.join(args.artifacts_dir, DOCS_FILENAME)
    cat_path = args.catastro or os.path.join(args.artifacts_dir, "catastro.parquet")
    out_path = args.out or os.path.join(args.artifacts_dir, OWN_CENTROIDS_FILENAME)
    if os.path.normcase(os.path.abspath(out_path)) == os.path.normcase(os.path.abspath(docs_path)):
        return _fail(f"--out must not be {DOCS_FILENAME}: the documents are never overwritten")
    for path in (docs_path, cat_path):
        if not os.path.isfile(path):
            return _fail(f"{os.path.basename(path)} not found at {path}")

    docs = pd.read_parquet(docs_path)
    if "objectid" not in docs.columns or "numero_predial_nacional" not in docs.columns:
        return _fail(f"{DOCS_FILENAME} has no objectid / numero_predial_nacional column: rebuild it with scripts/prepare_training.py")
    catastro = pd.read_parquet(cat_path, columns=["OBJECTID", "numero_predial_nacional", "centroid_lon", "centroid_lat"])
    table = own_centroid_table(docs, catastro)
    found = int(np.isfinite(table["own_lon"].to_numpy()).sum())
    if found == 0:
        return _fail(f"no document matches {os.path.basename(cat_path)} (OBJECTID + NPN): it is not the source of {DOCS_FILENAME}")
    tmp = out_path + ".tmp"
    table.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    print(f"{out_path}: {len(table)} documents, {found} with the parcel's own centroid, {len(table) - found} without")
    return 0


if __name__ == "__main__":
    sys.exit(main())
