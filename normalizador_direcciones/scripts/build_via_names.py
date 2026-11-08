"""Regenerate ``src/cali_address/via_names.py`` from the reviewed CSV (only ``status == verified`` rows are shipped).

The engine fingerprint hashes ``*.py`` only, so the parser reads the generated module, never the CSV: editing a status in
``src/cali_address/data/via_names_review.csv`` and re-running this script is what changes the fingerprint.

    python scripts/build_via_names.py [--check]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "src" / "cali_address" / "data" / "via_names_review.csv"
OUT_PATH = ROOT / "src" / "cali_address" / "via_names.py"
HEADER = (
    '"""VERIFIED street-name table (generated from ``data/via_names_review.csv`` by ``scripts/build_via_names.py``; do not edit)."""\n'
    "\n"
    "#: ``(folded variant, canonical street)``, longest variants first. Only rows reviewed as ``verified`` are shipped.\n"
    "VIA_NAMES: tuple[tuple[str, str], ...] = ("
)


def entries(path: Path = CSV_PATH) -> list[tuple[str, str]]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"].strip() != "verified":
                continue
            via = row["canonical_via"].strip()
            if not via:
                raise SystemExit(f"verified name {row['name']!r} has no canonical_via")
            for v in row["variants"].split("|"):
                v = v.strip().upper()
                if v:
                    if out.get(v, via) != via:
                        raise SystemExit(f"variant {v!r} maps to two streets")
                    out[v] = via
    return sorted(out.items(), key=lambda kv: (-len(kv[0]), kv[0]))


def render(items: list[tuple[str, str]]) -> str:
    body = "".join(f"\n    ({v!r}, {c!r})," for v, c in items)
    return HEADER + (body + "\n" if items else "") + ")\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail when the generated module is stale")
    a = ap.parse_args()
    text = render(entries())
    if a.check:
        return 0 if OUT_PATH.read_text(encoding="utf-8") == text else 1
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"{len(entries())} verified variants -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
