"""Generate ``src/cali_address/complement_lexicon.py`` (tower-name test lexicons).

The complement stage reads a word after ``TORRE``/``BLOQUE``/``ETAPA`` as a tower
name (``TO HUDSON``) unless it is a descriptive word (``TORRE ESQUINERA``). The
decision uses two disjoint word sets, built here from public reference data only:

* ``TOWER_NAME_WORDS``: words of the place names in the IDESC basemaps
  (``basemaps/barrios_veredas.geojson``: barrio, vereda and corregimiento names).
  Complexes and towers in Cali are very often named like places (CHIMINANGOS,
  GUAYACANES), and these words must never be classed as descriptive.
* ``DESCRIPTIVE_WORDS``: ordinary words of the cadastral address text
  (``artifacts/catastro_docs.parquet`` ``direccion``) that are not address grammar,
  seen at least ``min_cadastre_count`` times (ZONA, VERDE, PEATONAL, DEPOSITO...),
  plus :data:`DESCRIPTIVE_SEED`.

Why a seed at all: the cadastre writes codes after TO/BLQ (no descriptive value in
330k addresses), so the words people actually write after a tower kind
(ESQUINERA, PRINCIPAL, FRENTE, LATERAL) are not in any reference corpus. The seed is
closed-class address vocabulary (position, role, use, ordinals, colours, landmarks,
free-text abbreviations), not names; it wins over a place-name word (``LA ESQUINA``
does not make ESQUINA a name). It lives in ``cali_address.parser.DESCRIPTIVE_SEED``
because the parser also reads it directly (building names); it is re-exported here as
``DESCRIPTIVE_SEED``. No survey, evaluation or other citizen data is read.

Every word passes the parser's lexicon-free shape test (``_is_name_shaped``), so
kind words, connectors, grammar words and kind misspellings never appear.

Usage (CPU, a few minutes: parses nothing, tokenizes 330k strings):
    PYTHONPATH=src python scripts/build_complement_lexicons.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.parser import DESCRIPTIVE_SEED, _is_name_shaped, strip_accents  # noqa: E402,F401

DEFAULT_CADASTRE = os.path.join(ROOT, "artifacts", "catastro_docs.parquet")
DEFAULT_PLACES = os.path.join(ROOT, "basemaps", "barrios_veredas.geojson")
DEFAULT_OUT = os.path.join(ROOT, "src", "cali_address", "complement_lexicon.py")
#: Place-name properties of ``barrios_veredas.geojson``.
PLACE_FIELDS = ("BARRIO", "VEREDA", "CORREGIMIE")
#: A cadastral word must appear in this many addresses to count as ordinary vocabulary
#: (rarer words are mostly building names: SKY, TOWER, TRIBECA).
MIN_CADASTRE_COUNT = 3


def _words(text: str) -> list[str]:
    """Upper-case, accent-free alphabetic words of ``text``."""
    return re.findall(r"[A-Z]+", strip_accents(str(text)).upper())


def build_lexicons(
    cadastre_addresses: Iterable[str | None],
    place_names: Iterable[str | None],
    min_cadastre_count: int = MIN_CADASTRE_COUNT,
) -> tuple[frozenset[str], frozenset[str]]:
    """``(tower_name_words, descriptive_words)``, disjoint, every word name-shaped
    except that the descriptive set always keeps the whole seed."""
    seed = frozenset(w for w in DESCRIPTIVE_SEED if len(w) >= 3 and w.isalpha())
    names = {
        w for name in place_names if name
        for w in _words(name) if _is_name_shaped(w)
    } - seed
    counts: Counter[str] = Counter()
    for address in cadastre_addresses:
        if address:
            counts.update({w for w in _words(address) if _is_name_shaped(w)})
    cadastral = {w for w, n in counts.items() if n >= min_cadastre_count}
    descriptive = (cadastral - names) | seed
    return frozenset(names), frozenset(descriptive)


def read_cadastre_addresses(path: str) -> list[str]:
    import pandas as pd

    return pd.read_parquet(path, columns=["direccion"])["direccion"].dropna().astype(str).tolist()


def read_place_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        features = json.load(fh).get("features", [])
    return [
        str(props[field]) for props in (f.get("properties") or {} for f in features)
        for field in PLACE_FIELDS if props.get(field)
    ]


def _format_set(name: str, words: frozenset[str], doc: str) -> str:
    body = "".join(f'    "{w}",\n' for w in sorted(words))
    return f"#: {doc}\n{name}: frozenset[str] = frozenset({{\n{body}}})\n"


def render_module(names: frozenset[str], descriptive: frozenset[str], sources: dict[str, str]) -> str:
    """Python source of ``complement_lexicon.py``."""
    src = "\n".join(f"* {k}: ``{v}``" for k, v in sources.items())
    return (
        '"""Tower-name test lexicons for the complement stage (generated, do not edit).\n\n'
        "Regenerate with ``PYTHONPATH=src python scripts/build_complement_lexicons.py``;\n"
        "see that script for how each set is built. Sources:\n\n"
        f"{src}\n"
        '"""\n\n'
        + _format_set(
            "TOWER_NAME_WORDS", names, f"Place-name words (IDESC basemaps), {len(names)} words."
        )
        + "\n"
        + _format_set(
            "DESCRIPTIVE_WORDS", descriptive,
            f"Ordinary cadastral address words plus the descriptive seed, {len(descriptive)} words.",
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cadastre", default=DEFAULT_CADASTRE)
    ap.add_argument("--places", default=DEFAULT_PLACES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--min-cadastre-count", type=int, default=MIN_CADASTRE_COUNT)
    args = ap.parse_args(argv)
    names, descriptive = build_lexicons(
        read_cadastre_addresses(args.cadastre), read_place_names(args.places), args.min_cadastre_count,
    )
    sources = {
        "place names": "basemaps/barrios_veredas.geojson " + "/".join(PLACE_FIELDS),
        "cadastral words": f"artifacts/catastro_docs.parquet direccion, >= {args.min_cadastre_count} addresses",
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_module(names, descriptive, sources))
    print(f"wrote {args.out}: {len(names)} name words, {len(descriptive)} descriptive words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
