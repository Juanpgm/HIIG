#!/usr/bin/env python3
"""Build a ground-truth pair set to measure REAL precision/recall of the matcher.

Why this exists: ``tools/eval_catastro.py`` builds its positives with the
matcher's own grammar, so it can only confirm what the matcher already
believes. The pairs built here do not depend on the matcher's rules:

* ``synthetic_positives.jsonl``: a real cadastre address plus a mutated
  spelling of the SAME place. The mutations are implemented in this file
  (own regex, own alias tables); neither the parser nor the matcher is
  called to create them. Their only label is ``same_door`` (true).
* ``candidates_to_label.jsonl``: real pairs from the cadastre for a person
  to label. The primary unit is the BUILDING / predio (``same_door``: street
  + plate, ignoring any unit); the unit (apartment, block, house, local) is
  a secondary label (``same_unit``). The parser/index-style blocking and
  matcher scores are used ONLY to find plausible pairs and to oversample the
  score region around the threshold. Scores are never written to the
  labeling files.
* ``random_negatives.jsonl``: random pairs from different comunas (sanity;
  ``same_door`` false).
* ``label_sheet.csv``: shuffled sheet for the candidates only, WITHOUT the
  stratum (anti-anchoring); join back to the strata through ``pair_id``.
* ``MANIFEST.json``: seed, counts, shortfalls, score-bin counts and sha256 of
  every output.

``<private-dir>/matcher_scores_sidecar.jsonl`` (default: sibling directory
``<out-dir>_private``) holds ``{pair_id, score, decision}`` for every pair,
kept outside the labeling folder so the labeler is not anchored.

Everything is deterministic for a given ``--seed``. Only the parquet reading
needs ``pyarrow`` (imported inside ``load_catastro``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from emparejador import match, parse_canonical  # noqa: E402
from emparejador.parser import COMPLEMENT_KINDS  # noqa: E402

__all__ = [
    "Record",
    "TRANSFORMS",
    "TRANSFORM_RATIONALE",
    "apply_transform",
    "build_groundtruth",
    "default_private_dir",
    "generate_candidates",
    "generate_random_negatives",
    "generate_synthetic",
    "load_catastro",
    "main",
    "pair_id",
    "score_bin",
    "split_address",
]

THRESHOLD = 0.90
MAX_ADDRESS_LEN = 160  # longer strings are never real addresses; also bounds regex work
TOOL_VERSION = "build_groundtruth/2"
SIDECAR_NAME = "matcher_scores_sidecar.jsonl"
NOVEL_PREFIX = "novel_"
QUERY_SUFFIX = ", Cali, Valle del Cauca, Colombia"

STRATA = (
    "complement_one_sided",
    "cross_type_presence",
    "plate_letter",
    "adjacent_or_swap",
    "same_base_diff_complement",
    "opaque_or_degenerate",
)
BIN_EDGES = (0.70, 0.85, 0.90, 0.95)
BIN_LABELS = ("<0.70", "0.70-0.85", "0.85-0.90", "0.90-0.95", ">=0.95")
# Oversample around the 0.90 decision boundary.
BIN_WEIGHTS = (1.0, 1.0, 1.5, 1.5, 1.0)
DEFAULT_SIZES = {"synthetic": 250, "per_stratum": 60, "sanity": 30}
POOL_CAP = 4000
MAX_RECORD_USES = 2


class Record(NamedTuple):
    """One cadastre row (only the columns this tool needs)."""

    direccion: Any
    npn: Any
    manzana: Any
    comuna: Any
    barrio: Any = None


# ---------------------------------------------------------------------------
# Own address splitting (independent of the matcher's parser)
# ---------------------------------------------------------------------------

_ADDR_RE = re.compile(
    r"^(?P<via_type>[A-Z]{1,9}) (?P<via>[0-9A-Z ]+?) # (?P<cross>[0-9A-Z ]+?) - "
    r"(?P<plate>[0-9]{1,6}[A-Z]?)(?: (?P<complement>[0-9A-Z ]+))?$"
)


def split_address(addr: object) -> dict[str, str] | None:
    """Split a canonical cadastre address, or ``None`` if it is not one.

    Rejects non-strings, empty strings and anything longer than
    ``MAX_ADDRESS_LEN`` before running the regex.
    """
    if not isinstance(addr, str) or not addr or len(addr) > MAX_ADDRESS_LEN:
        return None
    m = _ADDR_RE.match(addr)
    if m is None:
        return None
    parts = m.groupdict()
    parts["complement"] = parts["complement"] or ""
    return parts


def _render(parts: dict[str, str]) -> str:
    text = f"{parts['via_type']} {parts['via']} # {parts['cross']} - {parts['plate']}"
    return f"{text} {parts['complement']}" if parts["complement"] else text


def _digits(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"\d+", text)]


# ---------------------------------------------------------------------------
# Transformations. Each one keeps every digit run and every letter of the
# address, and only changes how the same place is written.
# ---------------------------------------------------------------------------

# Via-type equivalence classes as a person would write them.
_TYPE_LONG = {
    "KR": ("CARRERA",), "K": ("CARRERA",), "CARRERA": ("CARRERA",),
    "CL": ("CALLE",), "C": ("CALLE",), "CALLE": ("CALLE",),
    "AV": ("AVENIDA",), "A": ("AVENIDA",), "AVENIDA": ("AVENIDA",),
    "DG": ("DIAGONAL",), "D": ("DIAGONAL",), "DIAGONAL": ("DIAGONAL",),
    "TV": ("TRANSVERSAL",), "T": ("TRANSVERSAL",), "TRANSVERSAL": ("TRANSVERSAL",),
    "CIRCULAR": ("CIRCULAR",),
}
# Spellings the parser is EXPECTED to know ("known vocabulary"). Recall on
# these is not independent evidence: it mostly confirms the parser's tables.
_TYPE_SHORT = {
    "CARRERA": ("KR", "CRA", "KRA", "K", "CR"),
    "CALLE": ("CL", "CLL", "C"),
    "AVENIDA": ("AV",),
    "DIAGONAL": ("DG", "D"),
    "TRANSVERSAL": ("TV", "T"),
}
# Plausible spellings the parser did NOT know when this set was designed
# (novel vocabulary). The parser has since learned most of them (CIRCULAR/CIR/
# CIRC stay unknown on purpose), so these tags now act as regression coverage.
_NOVEL_TYPE = {
    "CARRERA": ("CARR", "CRRA"),
    "AVENIDA": ("AVDA", "AVEN", "AVE"),
    "DIAGONAL": ("DIAG", "DIAGO"),
    "TRANSVERSAL": ("TRANSV", "TRV"),
    "CIRCULAR": ("CIR", "CIRC"),
}
# Only unambiguous complement abbreviations are expanded.
_COMPLEMENT_LONG = {
    "AP": ("APARTAMENTO", "APTO"),
    "BLQ": ("BLOQUE", "BL"),
    "ED": ("EDIFICIO",),
    "CA": ("CASA",),
    "LC": ("LOCAL",),
    "OF": ("OFICINA",),
    "INT": ("INTERIOR",),
    "TO": ("TORRE",),
    "PQ": ("PARQUEADERO",),
}
# Same meaning, spellings outside the parser's tables (each one is a common
# way to write the same word: APT/APART = apartamento, LOC = local,
# OFIC = oficina, EDIF/EDF = edificio, BLOQ = bloque).
_NOVEL_COMPLEMENT = {
    "AP": ("APT", "APT.", "APART"),
    "LC": ("LOC", "LOC."),
    "OF": ("OFIC",),
    "ED": ("EDIF", "EDF"),
    "BLQ": ("BLOQ",),
}
_NOVEL_NUMBER_WORDS = ("NUMERO",)


def _swap_type(token: str, options_for: Callable[[str], Sequence[str]], rng: random.Random) -> str | None:
    options = [o for o in options_for(token) if o != token]
    return rng.choice(options) if options else None


def _type_options_long(token: str) -> Sequence[str]:
    return _TYPE_LONG.get(token, ())


def _type_options_short(token: str) -> Sequence[str]:
    classes = _TYPE_LONG.get(token)
    if not classes:
        return ()
    return _TYPE_SHORT.get(classes[0], ())


def _type_options_novel(token: str) -> Sequence[str]:
    classes = _TYPE_LONG.get(token)
    if not classes:
        return ()
    return _NOVEL_TYPE.get(classes[0], ())


def _retype(parts: dict[str, str], options_for: Callable[[str], Sequence[str]], rng: random.Random) -> str | None:
    new_type = _swap_type(parts["via_type"], options_for, rng)
    cross_tokens = parts["cross"].split(" ")
    new_cross = None
    if cross_tokens[0] in _TYPE_LONG:
        new_cross = _swap_type(cross_tokens[0], options_for, rng)
    if new_type is None and new_cross is None:
        return None
    change_via = new_type is not None and (new_cross is None or rng.random() < 0.85)
    change_cross = new_cross is not None and (not change_via or rng.random() < 0.5)
    if change_via:
        parts["via_type"] = new_type  # type: ignore[assignment]
    if change_cross:
        cross_tokens[0] = new_cross  # type: ignore[assignment]
        parts["cross"] = " ".join(cross_tokens)
    return _render(parts)


def _t_type_alias_long(parts: dict[str, str], rng: random.Random) -> str | None:
    # Cadastre abbreviations (K, C, KR, CL...) written as the full street-type word.
    return _retype(parts, lambda t: _type_options_long(t), rng)


def _t_type_alias_short(parts: dict[str, str], rng: random.Random) -> str | None:
    # Another abbreviation of the same street type (KR -> CRA, CL -> CLL, AV -> AVDA...).
    return _retype(parts, lambda t: _type_options_short(t), rng)


def _t_type_alias_novel(parts: dict[str, str], rng: random.Random) -> str | None:
    # Street-type spellings outside the parser's tables (KR -> CARR, AV -> AVDA...).
    return _retype(parts, lambda t: _type_options_novel(t), rng)


def _alias_complement(parts: dict[str, str], table: dict[str, tuple[str, ...]], rng: random.Random) -> str | None:
    tokens = parts["complement"].split(" ") if parts["complement"] else []
    changed = False
    for i, token in enumerate(tokens):
        if token in table:
            tokens[i] = rng.choice(table[token])
            changed = True
    if not changed:
        return None
    parts["complement"] = " ".join(tokens)
    return _render(parts)


def _t_complement_alias(parts: dict[str, str], rng: random.Random) -> str | None:
    return _alias_complement(parts, _COMPLEMENT_LONG, rng)


def _t_novel_complement_alias(parts: dict[str, str], rng: random.Random) -> str | None:
    return _alias_complement(parts, _NOVEL_COMPLEMENT, rng)


def _t_novel_number_word(parts: dict[str, str], rng: random.Random) -> str | None:
    # '#' written as the word NUMERO ('CL 5 NUMERO 10 - 20'); the word is not in the parser's tables.
    text = _render(parts)
    return text.replace(" # ", f" {rng.choice(_NOVEL_NUMBER_WORDS)} ", 1) if " # " in text else None


def _pad_first_number(text: str, rng: random.Random) -> str:
    m = re.search(r"\d+", text)
    if m is None:
        return text
    digits = m.group(0)
    width = len(digits) + rng.choice((1, 2))
    return text[: m.start()] + digits.zfill(width) + text[m.end():]


def _t_zero_padded(parts: dict[str, str], rng: random.Random) -> str | None:
    parts["plate"] = _pad_first_number(parts["plate"], rng)
    if rng.random() < 0.5:
        parts["via"] = _pad_first_number(parts["via"], rng)
    if rng.random() < 0.5:
        parts["cross"] = _pad_first_number(parts["cross"], rng)
    return _render(parts)


def _t_glued_tokens(parts: dict[str, str], rng: random.Random) -> str | None:
    ops = [op for op in ("type", "hash", "dash") if rng.random() < 0.7] or ["hash"]
    via_type, via, cross, plate = parts["via_type"], parts["via"], parts["cross"], parts["plate"]
    glue_type = "type" in ops and via[:1].isdigit()
    head = f"{via_type}{via}" if glue_type else f"{via_type} {via}"
    hash_sep = "#" if "hash" in ops else " # "
    dash = "-" if "dash" in ops else " - "
    out = f"{head}{hash_sep}{cross}{dash}{plate}"
    return f"{out} {parts['complement']}" if parts["complement"] else out


def _t_lowercase(text: str, rng: random.Random) -> str | None:
    return rng.choice((text.lower(), text.title()))


def _t_extra_whitespace(text: str, rng: random.Random) -> str | None:
    out = re.sub(" ", lambda _m: " " * rng.choice((1, 1, 2, 3)), text)
    if out == text:
        out = text.replace(" ", "  ", 1)
    return out


def _t_hash_to_no(text: str, rng: random.Random) -> str | None:
    if "#" not in text:
        return None
    return text.replace("#", rng.choice(("No", "No.", "N°", "Nº", "NO")), 1)


def _t_hash_absent(text: str, rng: random.Random) -> str | None:
    # Only when the split between the two street numbers stays unambiguous:
    # the via and the cross each carry exactly one digit run.
    parts = split_address(text)
    if parts is None or len(_digits(parts["via"])) != 1 or len(_digits(parts["cross"])) != 1:
        return None
    return text.replace(" # ", " ", 1)


def _t_punctuation(text: str, rng: random.Random) -> str | None:
    kind = rng.choice(("type_dot", "comma_before_hash", "trailing_dot"))
    if kind == "type_dot":
        head, sep, rest = text.partition(" ")
        return f"{head}.{sep}{rest}" if sep else None
    if kind == "comma_before_hash":
        return text.replace(" # ", ", # ", 1) if " # " in text else None
    return text + "."


def _t_unicode_spacing(text: str, rng: random.Random) -> str | None:
    kind = rng.choice(("nbsp", "en_dash", "em_dash"))
    if kind == "nbsp":
        return text.replace(" ", " ", 1 + rng.randrange(3))
    dash = "–" if kind == "en_dash" else "—"
    return text.replace(" - ", f" {dash} ", 1) if " - " in text else None


def _t_unicode_symbols(text: str, rng: random.Random) -> str | None:
    if "#" not in text:
        return None
    return text.replace("#", rng.choice(("＃", "№")), 1)


def _t_trailing_whitespace(text: str, rng: random.Random) -> str | None:
    suffix = rng.choice((" ", "  ", "\t", " "))
    return (suffix + text) if rng.random() < 0.25 else (text + suffix)


STRUCTURAL: dict[str, Callable[[dict[str, str], random.Random], str | None]] = {
    "type_alias_long": _t_type_alias_long,
    "type_alias_short": _t_type_alias_short,
    "complement_alias": _t_complement_alias,
    "zero_padded": _t_zero_padded,
    "glued_tokens": _t_glued_tokens,
}
SURFACE: dict[str, Callable[[str, random.Random], str | None]] = {
    "lowercase": _t_lowercase,
    "extra_whitespace": _t_extra_whitespace,
    "hash_to_no": _t_hash_to_no,
    "hash_absent": _t_hash_absent,
    "punctuation": _t_punctuation,
    "unicode_spacing": _t_unicode_spacing,
    "unicode_symbols": _t_unicode_symbols,
    "trailing_whitespace": _t_trailing_whitespace,
}
NOVEL: dict[str, Callable[[dict[str, str], random.Random], str | None]] = {
    NOVEL_PREFIX + "type_alias": _t_type_alias_novel,
    NOVEL_PREFIX + "complement_alias": _t_novel_complement_alias,
    NOVEL_PREFIX + "number_word": _t_novel_number_word,
}
TRANSFORMS: dict[str, Callable[..., str | None]] = {**STRUCTURAL, **SURFACE, **NOVEL}
COMBO_TAG = "combo"
# Surface mutations that commute with each other and with structural ones.
_COMBO_SURFACE = ("lowercase", "extra_whitespace", "hash_to_no", "trailing_whitespace", "punctuation")

TRANSFORM_RATIONALE = {
    "type_alias_long": "Street-type abbreviation written as the full word (KR -> CARRERA); same street, digits untouched.",
    "type_alias_short": "Another abbreviation of the same street type that the parser is expected to know (KR -> CRA, CL -> CLL); same street.",
    "complement_alias": "Unit word spelled out (AP -> APARTAMENTO, BLQ -> BLOQUE); same unit, numbers untouched.",
    "zero_padded": "Leading zeros on numbers (26 -> 026); numeric value is identical.",
    "glued_tokens": "Spaces removed around type/number, '#' and '-' (CL5#10-20); same tokens, no digit changed.",
    "lowercase": "Letter case only (lower or Title Case); case never carries meaning in an address.",
    "extra_whitespace": "Repeated blanks between tokens; whitespace is not information.",
    "hash_to_no": "'#' written as No / No. / N° / Nº; all mean 'number'.",
    "hash_absent": "'#' omitted ('CL 5 10 - 20'), only for single-number streets so the split stays unambiguous; tokens and order unchanged.",
    "punctuation": "Stray '.' or ',' after the street type or at the end; punctuation only.",
    "unicode_spacing": "Non-breaking spaces or en/em dash instead of '-' (typical of copy/paste from documents).",
    "unicode_symbols": "Full-width '#' or the numero sign; compatibility characters for the same '#'/'No'.",
    "trailing_whitespace": "Leading or trailing blanks/tabs; whitespace is not information.",
    "novel_type_alias": "Street-type spellings that were NOT in the parser tables when the set was designed (KR -> CARR/CRRA, AV -> AVDA/AVEN, DG -> DIAG); same street, digits untouched.",
    "novel_complement_alias": "Unit-word spellings that were NOT in the parser tables when the set was designed (AP -> APT/APART, LC -> LOC, OF -> OFIC, ED -> EDIF/EDF, BLQ -> BLOQ); same unit, numbers untouched.",
    "novel_number_word": "'#' written as the word NUMERO; same meaning, not in the parser tables when the set was designed.",
}
COMBO_RATIONALE = "One structural mutation followed by one or two surface mutations; each preserves identity, so the composition does too."


def apply_transform(tag: str, addr: object, rng: random.Random) -> str | None:
    """Mutate ``addr`` with the named transform.

    Returns ``None`` when the input is not a canonical address, when the
    transform does not apply, when the result equals the input, or when it
    would alter any digit run (defensive check; the rules never do).
    """
    parts = split_address(addr)
    if parts is None:
        return None
    assert isinstance(addr, str)
    if tag in STRUCTURAL:
        out = STRUCTURAL[tag](dict(parts), rng)
    elif tag in NOVEL:
        out = NOVEL[tag](dict(parts), rng)
    elif tag in SURFACE:
        out = SURFACE[tag](addr, rng)
    elif tag == COMBO_TAG:
        out = _combo(dict(parts), rng)
    else:
        raise KeyError(f"unknown transform {tag!r}")
    if out is None or out == addr or _digits(out) != _digits(addr):
        return None
    return out


def _combo(parts: dict[str, str], rng: random.Random) -> str | None:
    structural = list(STRUCTURAL)
    rng.shuffle(structural)
    text: str | None = None
    for tag in structural:
        text = STRUCTURAL[tag](dict(parts), rng)
        if text is not None:
            break
    if text is None:
        return None
    surface = list(_COMBO_SURFACE)
    rng.shuffle(surface)
    applied = 0
    for tag in surface:
        mutated = SURFACE[tag](text, rng)
        if mutated is not None and mutated != text:
            text = mutated
            applied += 1
            if applied == 2 or (applied == 1 and rng.random() < 0.5):
                break
    return text if applied else None


# ---------------------------------------------------------------------------
# Ids and bins
# ---------------------------------------------------------------------------


def pair_id(prefix: str, x: tuple[str, str], y: tuple[str, str]) -> str:
    """Stable id, invariant to the (a, b) / (b, a) order."""
    first, second = sorted(("\x1f".join(map(str, x)), "\x1f".join(map(str, y))))
    digest = hashlib.sha1(f"{prefix}\x1e{first}\x1e{second}".encode("utf-8", "surrogatepass")).hexdigest()
    return f"{prefix}-{digest[:12]}"


def score_bin(score: float) -> int:
    """0: <0.70, 1: 0.70-0.85, 2: 0.85-0.90, 3: 0.90-0.95, 4: >=0.95."""
    return bisect_right(BIN_EDGES, score)


def default_scorer(a: str, b: str) -> float:
    return match(a, b, THRESHOLD).score


# ---------------------------------------------------------------------------
# Synthetic positives
# ---------------------------------------------------------------------------


def generate_synthetic(records: Sequence[Record], n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(f"synthetic:{seed}")
    sources = sorted({r.direccion for r in records if split_address(r.direccion) is not None})
    tags = sorted(TRANSFORMS) + [COMBO_TAG]
    quotas = {t: n // len(tags) for t in tags}
    order = tags[:]
    rng.shuffle(order)
    for t in order[: n % len(tags)]:
        quotas[t] += 1

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_tag: dict[str, int] = {}
    for tag in tags:
        quota = quotas[tag]
        if quota == 0 or not sources:
            by_tag[tag] = 0
            continue
        picks = rng.sample(range(len(sources)), min(len(sources), quota * 30 + 100))
        made = 0
        for idx in picks:
            if made == quota:
                break
            source = sources[idx]
            mutated = apply_transform(tag, source, rng)
            if mutated is None or mutated == source:
                continue
            pid = pair_id("syn", (source, ""), (mutated, ""))
            if pid in seen:
                continue
            seen.add(pid)
            rows.append({
                "pair_id": pid, "addr_a": source, "addr_b": mutated, "same_door": True,
                "tier": "synthetic", "transform": tag, "source_direccion": source,
            })
            made += 1
        by_tag[tag] = made
    rng.shuffle(rows)
    report = {"requested": n, "produced": len(rows), "shortfall": n - len(rows), "by_transform": by_tag}
    return rows, report


# ---------------------------------------------------------------------------
# Real-data candidates
# ---------------------------------------------------------------------------

_FIELDS = (
    "via_type", "via_number", "via_letters", "via_suffix", "via_suffix_letters", "via_bis", "via_quadrant",
    "cross_type", "cross_number", "cross_letters", "cross_suffix", "cross_suffix_letters", "cross_bis",
    "cross_quadrant", "plate", "complement",
)


def _key(p: Any, drop: Iterable[str] = ()) -> tuple:
    dropped = set(drop)
    return tuple(getattr(p, f) for f in _FIELDS if f not in dropped)


def _plate_letter(p: Any) -> str:
    """Trailing plate letter (``""`` when the plate has none or is missing)."""
    if p.plate is None or not p.plate[-1].isalpha():
        return ""
    return p.plate[-1]


def _plate_int(p: Any) -> int | None:
    if p.plate is None:
        return None
    digits = p.plate[:-1] if p.plate[-1].isalpha() else p.plate
    return int(digits)


def _group_pairs(members: list[int], rng: random.Random, ok: Callable[[int, int], bool], cap: int) -> list[tuple[int, int]]:
    """Up to ``cap`` valid pairs of one group: all of them for small groups, random draws for big ones."""
    if len(members) < 2:
        return []
    found: list[tuple[int, int]] = []
    if len(members) <= 12:
        found = [(i, j) for i, j in combinations(members, 2) if ok(i, j)]
        if len(found) > cap:
            found = rng.sample(found, cap)
        return found
    tried: set[tuple[int, int]] = set()
    for _ in range(cap * 6):
        i, j = rng.sample(members, 2)
        pair = (i, j) if i < j else (j, i)
        if pair in tried:
            continue
        tried.add(pair)
        if ok(*pair):
            found.append(pair)
            if len(found) == cap:
                break
    return found


class _Corpus:
    """Parsed view of the cadastre used to find candidate pairs."""

    def __init__(self, records: Sequence[Record]):
        self.records = records
        self.valid: list[int] = []
        self.parsed: dict[int, Any] = {}
        self.unparsed: list[int] = []
        self.prefixes: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            addr = rec.direccion
            if not isinstance(addr, str) or not addr.strip() or len(addr) > MAX_ADDRESS_LEN:
                continue
            self.valid.append(i)
            p = parse_canonical(addr)
            if p.parse_ok:
                self.parsed[i] = p
            else:
                self.unparsed.append(i)
            tokens = addr.split()
            if len(tokens) >= 3:
                self.prefixes[" ".join(tokens[:3])].append(i)

    def groups(self, key_fn: Callable[[Any], Any | None]) -> dict[Any, list[int]]:
        out: dict[Any, list[int]] = defaultdict(list)
        for i, p in self.parsed.items():
            k = key_fn(p)
            if k is not None:
                out[k].append(i)
        return out


def _pools(corpus: _Corpus, rng: random.Random, cap: int = 6) -> dict[str, dict[str, list[tuple[int, int]]]]:
    P = corpus.parsed
    pools: dict[str, dict[str, list[tuple[int, int]]]] = {s: {} for s in STRATA}

    # 1. complement one-sided: same base, one has a complement and the other none
    g = corpus.groups(lambda p: _key(p, ("complement",)))
    out: list[tuple[int, int]] = []
    for members in g.values():
        out += _group_pairs(members, rng, lambda i, j: bool(P[i].complement) != bool(P[j].complement), cap)
    pools["complement_one_sided"]["all"] = out

    # 2. cross type present vs absent
    g = corpus.groups(lambda p: _key(p, ("cross_type",)) if p.cross_number is not None else None)
    out = []
    for members in g.values():
        out += _group_pairs(members, rng, lambda i, j: (P[i].cross_type is None) != (P[j].cross_type is None), cap)
    pools["cross_type_presence"]["all"] = out

    # 3. plate letter: same numeric plate (zero padding ignored), different letter
    #    (present vs absent, or two different letters)
    g = corpus.groups(lambda p: (_key(p, ("plate",)), _plate_int(p)) if p.plate is not None else None)
    out = []
    for members in g.values():
        out += _group_pairs(members, rng, lambda i, j: _plate_letter(P[i]) != _plate_letter(P[j]), cap)
    pools["plate_letter"]["all"] = out

    # 4a. adjacent plates (+/-1, +/-2), everything else equal
    g = corpus.groups(lambda p: _key(p, ("plate",)) if p.plate is not None else None)
    adjacent: list[tuple[int, int]] = []
    for members in g.values():
        ordered = sorted(members, key=lambda i: (_plate_int(P[i]), i))
        found: list[tuple[int, int]] = []
        for x, i in enumerate(ordered):
            for j in ordered[x + 1: x + 12]:
                delta = _plate_int(P[j]) - _plate_int(P[i])
                if delta > 2:
                    break
                if delta > 0:
                    found.append((i, j))
        adjacent += rng.sample(found, cap) if len(found) > cap else found
    # 4b. via number swapped with the cross number, same plate
    swap_index: dict[tuple, list[int]] = defaultdict(list)
    for i, p in P.items():
        if p.via_number is not None and p.cross_number is not None and p.via_number != p.cross_number:
            swap_index[(p.via_number, p.cross_number, _plate_int(p))].append(i)
    swaps: list[tuple[int, int]] = []
    for (vn, cn, plate), members in swap_index.items():
        if vn < cn:  # visit each unordered swap once
            for i in members[:3]:
                for j in swap_index.get((cn, vn, plate), [])[:3]:
                    swaps.append((i, j))
    pools["adjacent_or_swap"]["adjacent_plate"] = adjacent
    pools["adjacent_or_swap"]["via_number_swap"] = swaps

    # 5. same base, different complements (both present)
    g = corpus.groups(lambda p: _key(p, ("complement",)))
    out = []
    for members in g.values():
        out += _group_pairs(members, rng, lambda i, j: bool(P[i].complement) and bool(P[j].complement) and P[i].complement != P[j].complement, cap)
    pools["same_base_diff_complement"]["all"] = out

    # 6. opaque / degenerate
    face = corpus.groups(lambda p: (p.via_type, p.via_number, p.cross_number))
    no_plate: list[tuple[int, int]] = []
    for members in face.values():
        bare = [i for i in members if P[i].plate is None]
        for i in bare:
            others = [j for j in members if j != i]
            for j in others[:cap]:
                no_plate.append((i, j))
    opaque: list[tuple[int, int]] = []
    base_groups = corpus.groups(lambda p: _key(p, ("complement",)))
    for members in base_groups.values():
        odd = [i for i in members if any(kind not in COMPLEMENT_KINDS for kind, _v in P[i].complement)]
        for i in odd:
            for j in [m for m in members if m != i][:cap]:
                opaque.append((i, j))
    truncated: list[tuple[int, int]] = []
    for i in corpus.unparsed:
        tokens = corpus.records[i].direccion.split()
        if len(tokens) < 3:
            continue
        peers = [j for j in corpus.prefixes.get(" ".join(tokens[:3]), []) if j != i]
        for j in (rng.sample(peers, cap) if len(peers) > cap else peers):
            truncated.append((i, j))
    pools["opaque_or_degenerate"]["missing_plate"] = no_plate
    pools["opaque_or_degenerate"]["opaque_complement"] = opaque
    pools["opaque_or_degenerate"]["truncated_unparsed"] = truncated
    return pools


def _select(
    scored: list[tuple[tuple[int, int], float]],
    quota: int,
    rng: random.Random,
    claim: Callable[[tuple[int, int]], bool],
) -> list[tuple[int, int]]:
    """Take up to ``quota`` pairs spread over the score bins (weighted toward the threshold)."""
    if quota <= 0:
        return []
    bins: list[list[tuple[int, int]]] = [[] for _ in BIN_LABELS]
    for pair, score in scored:
        bins[score_bin(score)].append(pair)
    for b in bins:
        rng.shuffle(b)
    total_w = sum(BIN_WEIGHTS)
    alloc = [int(quota * w / total_w) for w in BIN_WEIGHTS]
    by_weight = sorted(range(len(BIN_WEIGHTS)), key=lambda k: (-BIN_WEIGHTS[k], k))
    for k in by_weight[: quota - sum(alloc)]:
        alloc[k] += 1
    chosen: list[tuple[int, int]] = []
    cursor = [0] * len(bins)

    def take(k: int, want: int) -> int:
        got = 0
        while got < want and cursor[k] < len(bins[k]):
            pair = bins[k][cursor[k]]
            cursor[k] += 1
            if claim(pair):
                chosen.append(pair)
                got += 1
        return got

    for k in by_weight:
        take(k, alloc[k])
    while len(chosen) < quota:  # redistribute unused allocation
        progress = 0
        for k in by_weight:
            if len(chosen) >= quota:
                break
            progress += take(k, 1)
        if not progress:
            break
    return chosen


def generate_candidates(
    records: Sequence[Record],
    per_stratum: int,
    seed: int,
    scorer: Callable[[str, str], float] | None = None,
    pool_cap: int = POOL_CAP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample real-data candidate pairs for human labeling, six strata."""
    score_fn = scorer or default_scorer
    rng = random.Random(f"candidates:{seed}")
    corpus = _Corpus(records)
    pools = _pools(corpus, rng)
    seen: set[frozenset[int]] = set()
    uses: dict[int, int] = defaultdict(int)

    def claim(pair: tuple[int, int]) -> bool:
        i, j = pair
        key = frozenset(pair)
        if i == j or len(key) < 2 or key in seen:
            return False
        if uses[i] >= MAX_RECORD_USES or uses[j] >= MAX_RECORD_USES:
            return False
        a, b = records[i].direccion, records[j].direccion
        if a == b:
            return False
        seen.add(key)
        uses[i] += 1
        uses[j] += 1
        return True

    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    bin_counts: dict[str, dict[str, int]] = {}
    for stratum in STRATA:
        subtypes = sorted(pools[stratum])
        scored_by_sub: dict[str, list[tuple[tuple[int, int], float]]] = {}
        for sub in subtypes:
            pool = pools[stratum][sub]
            uniq = list(dict.fromkeys((min(i, j), max(i, j)) for i, j in pool if i != j))
            if len(uniq) > pool_cap // max(1, len(subtypes)):
                uniq = rng.sample(uniq, pool_cap // max(1, len(subtypes)))
            scored_by_sub[sub] = [
                (p, score_fn(records[p[0]].direccion, records[p[1]].direccion)) for p in uniq
            ]
        chosen: list[tuple[int, int]] = []
        share = per_stratum // max(1, len(subtypes))
        for sub in subtypes:
            chosen += _select(scored_by_sub[sub], share, rng, claim)
        if len(chosen) < per_stratum:
            leftovers = [
                (p, s) for sub in subtypes for p, s in scored_by_sub[sub]
                if frozenset(p) not in seen
            ]
            chosen += _select(leftovers, per_stratum - len(chosen), rng, claim)
        counts = [0] * len(BIN_LABELS)
        score_lookup = {p: s for sub in subtypes for p, s in scored_by_sub[sub]}
        for i, j in chosen:
            counts[score_bin(score_lookup[(i, j)])] += 1
            if rng.random() < 0.5:
                i, j = j, i
            ra, rb = records[i], records[j]
            rows.append({
                "pair_id": pair_id("cand", (ra.direccion, str(ra.npn)), (rb.direccion, str(rb.npn))),
                "addr_a": ra.direccion, "addr_b": rb.direccion, "stratum": stratum,
                "npn_a": ra.npn, "npn_b": rb.npn,
                "comuna_a": ra.comuna, "comuna_b": rb.comuna,
                "barrio_a": ra.barrio, "barrio_b": rb.barrio,
                "same_manzana": bool(ra.manzana is not None and ra.manzana == rb.manzana),
            })
        bin_counts[stratum] = dict(zip(BIN_LABELS, counts))
        report[stratum] = {
            "requested": per_stratum,
            "produced": len(chosen),
            "shortfall": per_stratum - len(chosen),
            "pool_size": sum(len(v) for v in scored_by_sub.values()),
            "score_bins": bin_counts[stratum],
        }
    return rows, report


# ---------------------------------------------------------------------------
# Random negatives
# ---------------------------------------------------------------------------


def generate_random_negatives(records: Sequence[Record], n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(f"sanity:{seed}")
    by_comuna: dict[Any, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if isinstance(rec.direccion, str) and rec.direccion.strip() and len(rec.direccion) <= MAX_ADDRESS_LEN:
            by_comuna[rec.comuna].append(i)
    comunas = sorted(by_comuna, key=str)
    rows: list[dict[str, Any]] = []
    seen: set[frozenset[int]] = set()
    if len(comunas) >= 2:
        for _ in range(n * 50 + 100):
            if len(rows) == n:
                break
            ca, cb = rng.sample(comunas, 2)
            i, j = rng.choice(by_comuna[ca]), rng.choice(by_comuna[cb])
            ra, rb = records[i], records[j]
            key = frozenset((i, j))
            if key in seen or ra.direccion == rb.direccion:
                continue
            seen.add(key)
            rows.append({
                "pair_id": pair_id("neg", (ra.direccion, str(ra.npn)), (rb.direccion, str(rb.npn))),
                "addr_a": ra.direccion, "addr_b": rb.direccion, "same_door": False, "tier": "sanity",
                "npn_a": ra.npn, "npn_b": rb.npn,
            })
    return rows, {"requested": n, "produced": len(rows), "shortfall": n - len(rows)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# No stratum and no score: the labeler must not be anchored by either.
_SHEET_COLUMNS = [
    "pair_id", "addr_a", "addr_b", "query_a", "query_b", "comuna_a", "comuna_b", "barrio_a", "barrio_b",
    "npn_a", "npn_b", "same_door", "same_unit", "evidence_source", "notes",
]


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def _sheet(rows: Iterable[dict[str, Any]], seed: int) -> str:
    """Label sheet: seeded shuffle, no stratum, map-lookup queries, blank label columns."""
    ordered = list(rows)
    random.Random(f"sheet:{seed}").shuffle(ordered)
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_SHEET_COLUMNS)
    for r in ordered:
        writer.writerow([
            r["pair_id"], r["addr_a"], r["addr_b"],
            f"{r['addr_a']}{QUERY_SUFFIX}", f"{r['addr_b']}{QUERY_SUFFIX}",
            _cell(r.get("comuna_a")), _cell(r.get("comuna_b")),
            _cell(r.get("barrio_a")), _cell(r.get("barrio_b")),
            _cell(r.get("npn_a")), _cell(r.get("npn_b")),
            "", "", "", "",
        ])
    return buf.getvalue()


def _bin_counts(scores: Iterable[float]) -> dict[str, int]:
    counts = [0] * len(BIN_LABELS)
    for score in scores:
        counts[score_bin(score)] += 1
    return dict(zip(BIN_LABELS, counts))


def build_groundtruth(
    records: Sequence[Record],
    out_dir: str | Path,
    seed: int,
    sizes: dict[str, int] | None = None,
    scorer: Callable[[str, str], float] | None = None,
    private_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate every output file and return the manifest.

    Public files go to ``out_dir``; the score sidecar goes to ``private_dir``
    (default: the sibling directory ``<out_dir name>_private``).
    """
    cfg = {**DEFAULT_SIZES, **(sizes or {})}
    score_fn = scorer or default_scorer
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    private = Path(private_dir) if private_dir is not None else default_private_dir(out)
    private.mkdir(parents=True, exist_ok=True)

    synthetic, syn_report = generate_synthetic(records, cfg["synthetic"], seed)
    candidates, cand_report = generate_candidates(records, cfg["per_stratum"], seed, scorer=score_fn)
    negatives, neg_report = generate_random_negatives(records, cfg["sanity"], seed)

    sidecar = []
    score_of: dict[str, float] = {}
    for row in synthetic + candidates + negatives:
        score = round(float(score_fn(row["addr_a"], row["addr_b"])), 6)
        score_of[row["pair_id"]] = score
        sidecar.append({"pair_id": row["pair_id"], "score": score, "decision": "MATCH" if score >= THRESHOLD else "NO_MATCH"})

    files = {
        "synthetic_positives.jsonl": _jsonl(synthetic),
        "candidates_to_label.jsonl": _jsonl(candidates),
        "random_negatives.jsonl": _jsonl(negatives),
        "label_sheet.csv": _sheet(candidates, seed),
    }
    digests = {}
    for name, text in files.items():
        data = text.encode("utf-8")
        (out / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    sidecar_bytes = _jsonl(sidecar).encode("utf-8")
    (private / SIDECAR_NAME).write_bytes(sidecar_bytes)

    strata_counts = {s: sum(1 for r in candidates if r["stratum"] == s) for s in STRATA}
    manifest = {
        "tool": TOOL_VERSION,
        "seed": seed,
        "parquet_rows": len(records),
        "sizes": cfg,
        "threshold_for_sidecar_decision": THRESHOLD,
        "counts": {
            "synthetic": len(synthetic),
            "synthetic_by_transform": syn_report["by_transform"],
            "candidates": len(candidates),
            "candidates_by_stratum": strata_counts,
            "sanity_negatives": len(negatives),
        },
        "shortfalls": {
            "synthetic": {k: syn_report[k] for k in ("requested", "produced", "shortfall")},
            "candidates": {s: {k: cand_report[s][k] for k in ("requested", "produced", "shortfall")} for s in STRATA},
            "sanity": neg_report,
        },
        "candidate_score_bins_by_stratum": {s: cand_report[s]["score_bins"] for s in STRATA},
        "synthetic_score_bins": _bin_counts(score_of[r["pair_id"]] for r in synthetic),
        "sanity_score_bins": _bin_counts(score_of[r["pair_id"]] for r in negatives),
        "sidecar": {"file": SIDECAR_NAME, "sha256": hashlib.sha256(sidecar_bytes).hexdigest()},
        "transform_rationale": {**TRANSFORM_RATIONALE, COMBO_TAG: COMBO_RATIONALE},
        "sha256": digests,
    }
    (out / "MANIFEST.json").write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return manifest


def default_private_dir(out_dir: str | Path) -> Path:
    """Sibling directory ``<out_dir name>_private`` (kept away from the labeler)."""
    out = Path(out_dir)
    return out.parent / f"{out.name}_private"


def load_catastro(path: str | Path) -> list[Record]:
    """Read the cadastre parquet (needs pyarrow; isolated here so tests can stub it)."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(path), columns=["direccion", "numero_predial_nacional", "manzana", "comuna", "barrio"]
    ).to_pydict()
    return [
        Record(d, n, m, c, b)
        for d, n, m, c, b in zip(
            table["direccion"], table["numero_predial_nacional"], table["manzana"], table["comuna"], table["barrio"]
        )
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the ground-truth pair set from the cadastre parquet.")
    ap.add_argument("--catastro", required=True, help="Path to the cadastre parquet (data only).")
    ap.add_argument("--out-dir", default="data/groundtruth")
    ap.add_argument("--private-dir", default=None,
                    help="Where the score sidecar goes (default: sibling '<out-dir>_private').")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--n-synthetic", type=int, default=DEFAULT_SIZES["synthetic"])
    ap.add_argument("--per-stratum", type=int, default=DEFAULT_SIZES["per_stratum"])
    ap.add_argument("--n-sanity", type=int, default=DEFAULT_SIZES["sanity"])
    args = ap.parse_args(argv)
    records = load_catastro(args.catastro)
    manifest = build_groundtruth(
        records, args.out_dir, args.seed,
        {"synthetic": args.n_synthetic, "per_stratum": args.per_stratum, "sanity": args.n_sanity},
        private_dir=args.private_dir,
    )
    print(json.dumps({"counts": manifest["counts"], "shortfalls": manifest["shortfalls"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
