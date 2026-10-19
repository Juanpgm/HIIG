#!/usr/bin/env python3
"""Evaluate the matcher against a large corpus of already-normalized addresses.

The corpus is the Cali cadastral base (a parquet file consumed purely as
data, referenced by path, never copied into this project). The harness
measures, for a deterministic sample of addresses:

A. parse coverage (and the shape of what the parser rejects);
B. self-match and symmetry invariants;
C. recall on format variants (positives by construction);
D. rejection of true differences (negatives by construction);
E. false-link risk inside the real cadastre;
F. precision/recall/F1 at several thresholds and plate tolerances;
G. throughput;
H. linking against the full cadastre with ``AddressIndex`` (own-parcel
   recovery, ambiguity, false best, throughput, build time and memory).

Only the parquet reading needs ``pyarrow`` and it is imported inside
``main``; every helper at module level is pure Python so it can be unit
tested without pyarrow or the data (see ``tests/test_eval_helpers.py``).

Privacy: reports contain aggregates plus at most ``MAX_EXAMPLES`` example
addresses per category. Parcel ids are used only to tell records apart and
are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections.abc import Hashable
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

# This project's package lives under <project>/src, not at the project
# root, so it must be added to sys.path before it can be imported.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from emparejador import AddressIndex, CanonicalAddress, clean_text, match, parse_canonical, render  # noqa: E402
from emparejador.matcher import _is_opaque_chunk  # noqa: E402 - single source for 'opaque' (S4)
from emparejador.parser import COMPLEMENT_KINDS, QUADRANTS, VIA_TYPES  # noqa: E402

__all__ = [
    "AddressIndex",
    "EXPECTED_MUTATION_OUTCOME",
    "base_key",
    "build_link_queries",
    "classify_pair",
    "cross_signature",
    "evaluate_link_queries",
    "failure_cause",
    "format_table",
    "main",
    "measure_linking",
    "negative_mutations",
    "note_family",
    "pct",
    "positive_variants",
    "process_memory_bytes",
    "precision_recall_f1",
    "run_evaluation",
    "sample_indices",
    "shape_signature",
    "summarize_scores",
    "via_signature",
]

ENV_CATASTRO = "CATASTRO_PARQUET"
LINK_SAMPLE = 5_000
HIT_THRESHOLD = 0.90
MAX_EXAMPLES = 5
F_THRESHOLDS = (0.70, 0.85, 0.90, 0.95, 1.0)
# (threshold, plate_tolerance) configurations reported in table F.
F_CONFIGS: tuple[tuple[float, int], ...] = (
    (0.70, 0),
    (0.85, 0),
    (0.90, 0),
    (0.95, 0),
    (1.0, 0),
    (0.90, 1),
    (0.90, 2),
)

# ---------------------------------------------------------------------------
# Pure helpers: formatting, sampling, numbers
# ---------------------------------------------------------------------------


def pct(numerator: int, denominator: int) -> str:
    """Format ``numerator / denominator`` as a percentage (``n/a`` if empty)."""
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def format_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a plain-text, left-aligned table."""
    body = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in body:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)``; zero denominators give 0.0."""
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError("counts must be non-negative")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def sample_indices(total: int, sample: int, seed: int) -> list[int]:
    """Deterministic, sorted, unique sample of ``range(total)``."""
    if total < 0 or sample < 0:
        raise ValueError("total and sample must be non-negative")
    if sample >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), sample))


def _summarize_counts(counts: Mapping[float, int]) -> dict[str, Any]:
    total = sum(counts.values())
    bins = {"0.00": 0, "0.01-0.89": 0, "0.90-0.99": 0, "1.00": 0}
    weighted = 0.0
    for score, n in counts.items():
        weighted += score * n
        if score <= 0.0:
            bins["0.00"] += n
        elif score < HIT_THRESHOLD:
            bins["0.01-0.89"] += n
        elif score < 1.0:
            bins["0.90-0.99"] += n
        else:
            bins["1.00"] += n
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    return {
        "count": total,
        "min": min(counts) if counts else None,
        "max": max(counts) if counts else None,
        "mean": (weighted / total) if total else None,
        "bins": bins,
        "top_values": [[s, n] for s, n in top],
    }


def summarize_scores(scores: Iterable[float]) -> dict[str, Any]:
    """Summary (count/min/max/mean and coarse bins) of a score collection."""
    return _summarize_counts(Counter(scores))


def note_family(note: object) -> str:
    """``token_inesperado:XYZ`` -> ``token_inesperado`` (empty for None)."""
    if not isinstance(note, str):
        return ""
    return note.split(":", 1)[0]


# ---------------------------------------------------------------------------
# Pure helpers: shape signatures (parse-failure clustering)
# ---------------------------------------------------------------------------

_KNOWN_TOKENS: frozenset[str] = (
    frozenset(k for k in VIA_TYPES if len(k) >= 2)
    | QUADRANTS
    | frozenset({"BIS"})
    | frozenset(COMPLEMENT_KINDS)
)
_RUN_RE = re.compile(r"\d+|[^\W\d_]+")


def _signature_repl(match_obj: re.Match[str]) -> str:
    run = match_obj.group()
    if run[0].isdigit():
        return "9"
    return run.upper() if run.upper() in _KNOWN_TOKENS else "A"


def shape_signature(text: object) -> str:
    """Coarse shape: digit runs -> ``9``, unknown letter runs -> ``A``.

    Known grammar tokens (via types, quadrants, ``BIS``, complement kinds)
    and punctuation are kept, so addresses that fail for the same grammar
    reason cluster together.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    return " ".join(_RUN_RE.sub(_signature_repl, text).split())


def _adjacent_single_letters(body: list[str]) -> bool:
    singles = [len(t) == 1 and t.isalpha() for t in body]
    return any(a and b for a, b in zip(singles, singles[1:]))


def failure_cause(raw: object) -> str:
    """Heuristic, human-oriented cause of a parse failure for ``raw``.

    Independent from the parser's own ``notes`` (which only name the first
    token it choked on): the via and cross segments are re-parsed in
    isolation to locate *where* the grammar stops fitting, and the parser
    note is used only to label failures that sit in the plate/tail part.
    Returns ``parses_ok`` for addresses that parse.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "empty_or_non_text"
    if ";" in raw:
        return "multiple_addresses"
    parsed = parse_canonical(raw)
    if parsed.parse_ok:
        return "parses_ok"
    tokens = clean_text(raw).split()
    if tokens.count("#") > 1:
        return "multiple_addresses"
    if "#" not in tokens:
        return "no_hash_separator"
    hash_at = tokens.index("#")
    via, rest = tokens[:hash_at], tokens[hash_at + 1 :]
    if not parse_canonical(" ".join(via) + " #").parse_ok:
        body = via[1:] if via and via[0] in VIA_TYPES else via
        if not body or not body[0].isdigit():
            return "via_number_missing"
        if _adjacent_single_letters(body):
            return "via_two_letters"
        return "via_unexpected_token"
    cross = rest[: rest.index("-")] if "-" in rest else rest
    if cross and not parse_canonical("KR 1 # " + " ".join(cross)).parse_ok:
        body = cross[1:] if cross[0] in VIA_TYPES else cross
        if _adjacent_single_letters(body):
            return "cross_two_letters"
        if "-" not in rest:
            return "no_plate_separator"
        return "cross_unexpected_token"
    note = parsed.notes[0] if parsed.notes else ""
    if note == "complemento_demasiado_largo":
        return "tail_too_long"
    if note == "placa_invalida":
        return "plate_too_long"
    if note == "multiples_direcciones":
        return "tail_via_type_then_number"
    if note.startswith("token_inesperado"):
        return "tail_unexpected_token"
    return "other"


# ---------------------------------------------------------------------------
# Pure helpers: positive variants (same address, different formatting)
# ---------------------------------------------------------------------------

_VIA_LONG: dict[str, str] = {
    "KR": "CARRERA",
    "CL": "CALLE",
    "AV": "AVENIDA",
    "AC": "AVENIDA CALLE",
    "AK": "AVENIDA CARRERA",
    "DG": "DIAGONAL",
    "TV": "TRANSVERSAL",
    "PJ": "PASAJE",
    "PS": "PASEO",
    "CV": "CIRCUNVALAR",
    "AU": "AUTOPISTA",
    "CT": "CALLEJON",
}
_VIA_DOTTED: dict[str, str] = {
    "KR": "CRA.",
    "CL": "CLL.",
    "AV": "AV.",
    "AC": "AC.",
    "AK": "AK.",
    "DG": "DG.",
    "TV": "TV.",
    "PJ": "PJ.",
    "PS": "PS.",
    "CV": "CV.",
    "AU": "AU.",
    "CT": "CT.",
}
_COMPLEMENT_LONG: dict[str, str] = {
    "AP": "APTO",
    "TO": "TORRE",
    "LC": "LOCAL",
    "CA": "CASA",
    "ET": "ETAPA",
    "BLQ": "BLOQUE",
    "MZ": "MANZANA",
    "LT": "LOTE",
    "OF": "OFICINA",
    "ED": "EDIFICIO",
    "CONJ": "CONJUNTO",
    "INT": "INTERIOR",
    "PQ": "PARQUEADERO",
    "GA": "GARAJE",
    "BG": "BODEGA",
    "BR": "BARRIO",
    "UR": "URBANIZACION",
    "VDA": "VEREDA",
    "CGTO": "CORREGIMIENTO",
}
_MAX_INT_DIGITS = 18


def _tail_tokens(tokens: list[str], parsed: CanonicalAddress) -> tuple[int, list[str]] | None:
    """Locate the complement tail in cleaned ``tokens`` (needs a plate)."""
    if "#" not in tokens or (parsed.plate is None and not parsed.complement):
        return None
    try:
        dash = tokens.index("-", tokens.index("#"))
    except ValueError:
        return None
    if parsed.plate is None:
        plate_tokens = 0  # the complement sits in the plate slot
    else:
        plate_tokens = 2 if parsed.plate[-1].isalpha() else 1
    start = dash + 1 + plate_tokens
    return start, tokens[start:]


def _alias_token(tokens: list[str], idx: int, mapping: Mapping[str, str]) -> str | None:
    """Alias for the via-type token at ``idx`` (legacy prefixes included)."""
    if idx >= len(tokens):
        return None
    token = tokens[idx]
    if token == "AVENIDA" and idx + 1 < len(tokens) and tokens[idx + 1] in {"CALLE", "CARRERA"}:
        return None  # two-word alias: leave as is
    canonical = VIA_TYPES.get(token)
    return mapping.get(canonical) if canonical is not None else None


def _alias_variant(tokens: list[str], parsed: CanonicalAddress, mapping: Mapping[str, str]) -> str:
    out = list(tokens)
    if parsed.via_type is not None:
        alias = _alias_token(out, 0, mapping)
        if alias is not None:
            out[0] = alias
    if parsed.cross_type is not None and "#" in out:
        idx = out.index("#") + 1
        alias = _alias_token(out, idx, mapping)
        if alias is not None:
            out[idx] = alias
    return " ".join(out)


def positive_variants(addr: object) -> dict[str, str]:
    """Formatting variants of ``addr`` that should still match it.

    Returns ``{}`` when ``addr`` does not parse. Variants are *not*
    validated: a variant the parser rejects is precisely a recall miss the
    harness wants to measure. Only variants that apply to the address
    (and differ from it) are returned.
    """
    parsed = parse_canonical(addr)
    if not parsed.parse_ok or not isinstance(addr, str):
        return {}
    base = clean_text(addr)
    tokens = base.split()
    out: dict[str, str] = {}

    def add(kind: str, text: str) -> None:
        if text and text != addr:
            out[kind] = text

    add("lowercase", addr.lower())
    add("extra_whitespace", "  " + " \t ".join(tokens) + "\t")
    add("accent_marks", " ".join("".join(ch + "́" if i == 0 and ch.isalpha() else ch for i, ch in enumerate(t)) for t in tokens))
    add("glued", re.sub(r"(?<=[0-9]) (?=[A-Z](?: |$))", "", base))
    if tokens and tokens[0] in VIA_TYPES:
        add("glued_type", re.sub(r"^([A-Z]+) (?=[0-9])", r"\1", base))
    add("hash_tight", base.replace("# ", "#").replace(" - ", "-"))
    add("hash_no_dot", base.replace(" # ", " No. "))
    add("hash_ordinal", base.replace(" # ", " Nº "))

    if parsed.plate is not None:
        digits = parsed.plate.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        padded = digits.zfill(len(digits) + 1)
        add("plate_zero_pad", re.sub(r"( - )" + re.escape(digits) + r"(?=( [A-Z])?( |$))", r"\g<1>" + padded, base, count=1))

    if parsed.via_type in _VIA_LONG or parsed.cross_type in _VIA_LONG:
        add("alias_long", _alias_variant(tokens, parsed, _VIA_LONG))
        add("alias_dotted", _alias_variant(tokens, parsed, _VIA_DOTTED))

    located = _tail_tokens(tokens, parsed)
    if located is not None and located[1]:
        start, tail = located
        head = tokens[:start]
        add("complement_alias", " ".join(head + [_COMPLEMENT_LONG.get(t, t) for t in tail]))
        padded_tail = ["0" + t if t.isdigit() else t for t in tail]
        add("complement_zero_pad", " ".join(head + padded_tail))
        if tail[0] in COMPLEMENT_KINDS:
            chunks: list[list[str]] = []
            for token in tail:
                if token in COMPLEMENT_KINDS or not chunks:
                    chunks.append([token])
                else:
                    chunks[-1].append(token)
            if len(chunks) >= 2:
                reordered = [tok for chunk in reversed(chunks) for tok in chunk]
                if reordered != tail:
                    add("complement_reorder", " ".join(head + reordered))
    return out


# ---------------------------------------------------------------------------
# Pure helpers: negative mutations (a genuinely different address)
# ---------------------------------------------------------------------------

# Expected verdict of the current matcher design for each mutation kind.
# Everything is a true difference (score < 0.90) except adding/removing a
# whole KNOWN complement, which is designed to match (one-sided penalty
# 0.98); removing an opaque complement is a true difference (0.85).
EXPECTED_MUTATION_OUTCOME: dict[str, str] = {
    "via_number_plus1": "no_match",
    "via_number_minus1": "no_match",
    "via_type_change": "no_match",
    "cross_number_plus1": "no_match",
    "cross_number_minus1": "no_match",
    "plate_plus1": "no_match",
    "plate_minus1": "no_match",
    "plate_plus2": "no_match",
    "plate_minus2": "no_match",
    "via_letter_add": "no_match",
    "via_letter_drop": "no_match",
    "via_letter_change": "no_match",
    "cross_letter_add": "no_match",
    "cross_letter_drop": "no_match",
    "cross_letter_change": "no_match",
    "via_bis_add": "no_match",
    "via_bis_remove": "no_match",
    "cross_bis_add": "no_match",
    "cross_bis_remove": "no_match",
    "via_quadrant_add": "no_match",
    "via_quadrant_remove": "no_match",
    "cross_quadrant_add": "no_match",
    "cross_quadrant_remove": "no_match",
    "complement_value_change": "no_match",
    "complement_added": "match",
    "complement_removed": "match",
    # An opaque (unstructured) complement dropped on one side costs 0.15, so
    # it is a true difference and must NOT link.
    "complement_removed_opaque": "no_match",
}


def _bump(value: str | None, delta: int) -> str | None:
    """``str(int(value) + delta)``; ``None`` if unusable or result < 1."""
    if value is None or not value.isdigit() or len(value.lstrip("0")) > _MAX_INT_DIGITS:
        return None
    result = int(value) + delta
    return str(result) if result >= 1 else None


def _next_letter(letter: str) -> str:
    return chr((ord(letter) - 65 + 1) % 26 + 65)


def _change_complement_value(value: str) -> str | None:
    if not value:
        return None
    if value[-1].isdigit():
        m = re.search(r"([0-9]+)$", value)
        assert m is not None
        if len(m.group(1)) > _MAX_INT_DIGITS:
            return None
        return value[: m.start()] + str(int(m.group(1)) + 1)
    if "A" <= value[-1] <= "Z":
        return value[:-1] + _next_letter(value[-1])
    return None


_CONTENT_FIELDS = (
    "via_type", "via_number", "via_letters", "via_suffix", "via_suffix_letters", "via_bis",
    "via_quadrant", "cross_type", "cross_number", "cross_letters", "cross_suffix",
    "cross_suffix_letters", "cross_bis", "cross_quadrant", "plate", "complement",
)


def _content(parsed: CanonicalAddress) -> tuple[object, ...]:
    return tuple(getattr(parsed, f) for f in _CONTENT_FIELDS)


def negative_mutations(addr: object) -> dict[str, str]:
    """One-field mutations of ``addr`` that produce a different address.

    Returns ``{}`` when ``addr`` does not parse. Mutations that cannot be
    applied, that render back to the same address, or whose result the
    parser rejects are skipped; this function never raises.
    """
    parsed = parse_canonical(addr)
    if not parsed.parse_ok:
        return {}
    base_text = render(parsed)
    base_content = _content(parsed)
    out: dict[str, str] = {}

    def add(kind: str, **changes: object) -> None:
        text = render(replace(parsed, **changes))  # type: ignore[arg-type]
        if text == base_text:
            return
        candidate = parse_canonical(text)
        if not candidate.parse_ok or _content(candidate) == base_content:
            return
        out[kind] = text

    for kind, delta in (("plus1", 1), ("minus1", -1)):
        new = _bump(parsed.via_number, delta)
        if new is not None:
            add(f"via_number_{kind}", via_number=new)
        if parsed.cross_number is not None:
            new = _bump(parsed.cross_number, delta)
            if new is not None:
                add(f"cross_number_{kind}", cross_number=new)

    add("via_type_change", via_type="CL" if parsed.via_type != "CL" else "KR")

    if parsed.plate is not None:
        letter = parsed.plate[-1] if parsed.plate[-1].isalpha() else ""
        digits = parsed.plate[: len(parsed.plate) - len(letter)]
        for kind, delta in (("plus1", 1), ("minus1", -1), ("plus2", 2), ("minus2", -2)):
            new = _bump(digits, delta)
            if new is not None:
                add(f"plate_{kind}", plate=new + letter)

    for side, present in (("via", True), ("cross", parsed.cross_number is not None)):
        if not present:
            continue
        letters = getattr(parsed, f"{side}_letters")
        if letters:
            add(f"{side}_letter_drop", **{f"{side}_letters": None})
            add(f"{side}_letter_change", **{f"{side}_letters": _next_letter(letters)})
        else:
            add(f"{side}_letter_add", **{f"{side}_letters": "A"})
        if getattr(parsed, f"{side}_bis"):
            add(f"{side}_bis_remove", **{f"{side}_bis": False})
        else:
            add(f"{side}_bis_add", **{f"{side}_bis": True})
        if getattr(parsed, f"{side}_quadrant"):
            add(f"{side}_quadrant_remove", **{f"{side}_quadrant": None})
        else:
            add(f"{side}_quadrant_add", **{f"{side}_quadrant": "NORTE"})

    if parsed.complement:
        chunks = list(parsed.complement)
        for idx, (kind_code, value) in enumerate(chunks):
            new_value = _change_complement_value(value)
            if new_value is not None:
                changed = chunks[:idx] + [(kind_code, new_value)] + chunks[idx + 1 :]
                add("complement_value_change", complement=tuple(sorted(changed)))
                break
        opaque = any(_is_opaque_chunk(kind, value) for kind, value in parsed.complement)
        add("complement_removed_opaque" if opaque else "complement_removed", complement=())
    else:
        add("complement_added", complement=(("AP", "101"),))
    return out


# ---------------------------------------------------------------------------
# Pure helpers: signatures and pair classification
# ---------------------------------------------------------------------------


def via_signature(parsed: CanonicalAddress) -> tuple[object, ...] | None:
    if not parsed.parse_ok:
        return None
    return (
        parsed.via_type, parsed.via_number, parsed.via_letters, parsed.via_suffix,
        parsed.via_suffix_letters, parsed.via_bis, parsed.via_quadrant,
    )


def cross_signature(parsed: CanonicalAddress) -> tuple[object, ...] | None:
    if not parsed.parse_ok:
        return None
    return (
        parsed.cross_type, parsed.cross_number, parsed.cross_letters, parsed.cross_suffix,
        parsed.cross_suffix_letters, parsed.cross_bis, parsed.cross_quadrant,
    )


def _plate_key(parsed: CanonicalAddress) -> str | None:
    if parsed.plate is None:
        return None
    letter = parsed.plate[-1] if parsed.plate[-1].isalpha() else ""
    digits = parsed.plate[: len(parsed.plate) - len(letter)]
    return (digits.lstrip("0") or "0") + letter


def base_key(parsed: CanonicalAddress) -> tuple[object, ...] | None:
    """Identity of an address ignoring its complement."""
    if not parsed.parse_ok:
        return None
    return (via_signature(parsed), cross_signature(parsed), _plate_key(parsed))


def classify_pair(a: CanonicalAddress, b: CanonicalAddress) -> str:
    """Categorize a pair of parsed addresses for false-link analysis."""
    if not a.parse_ok or not b.parse_ok:
        return "unparseable"
    if base_key(a) == base_key(b):
        set_a, set_b = tuple(sorted(a.complement)), tuple(sorted(b.complement))  # multiset, like the matcher
        if set_a == set_b:
            return "same_base_same_complement"
        if not set_a or not set_b:
            return "same_base_one_side_bare"
        return "same_base_complements_differ"
    if via_signature(a) == via_signature(b) and cross_signature(a) == cross_signature(b):
        return "different_plate"
    return "different_base_other"


# ---------------------------------------------------------------------------
# Measurements (pure Python over lists; no pyarrow)
# ---------------------------------------------------------------------------


def _example(record: dict[str, Any], bucket: list[dict[str, Any]]) -> None:
    if len(bucket) < MAX_EXAMPLES:
        bucket.append(record)


def measure_parse_coverage(addresses: Sequence[object], parsed: Sequence[CanonicalAddress]) -> dict[str, Any]:
    """Table A: parse coverage, failure notes and shape clusters."""
    total = len(addresses)
    ok = sum(1 for p in parsed if p.parse_ok)
    notes: Counter[str] = Counter()
    families: Counter[str] = Counter()
    signatures: dict[str, dict[str, Any]] = {}
    causes: dict[str, dict[str, Any]] = {}
    plate_letters: Counter[str] = Counter()
    unknown_kinds: Counter[str] = Counter()
    for addr, p in zip(addresses, parsed):
        if p.parse_ok:
            if p.plate and p.plate[-1].isalpha():
                plate_letters[p.plate[-1]] += 1
            for kind, _value in p.complement:
                if kind not in COMPLEMENT_KINDS:
                    unknown_kinds[kind] += 1
            continue
        cause = causes.setdefault(failure_cause(addr), {"count": 0, "examples": []})
        cause["count"] += 1
        if len(cause["examples"]) < MAX_EXAMPLES:
            cause["examples"].append(addr)
        for note in p.notes or ("sin_nota",):
            notes[note] += 1
            families[note_family(note)] += 1
        sig = shape_signature(addr)
        entry = signatures.setdefault(sig, {"count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 2:
            entry["examples"].append(addr)
    top = sorted(signatures.items(), key=lambda kv: (-kv[1]["count"], kv[0]))[:25]
    return {
        "total": total,
        "parse_ok": ok,
        "parse_fail": total - ok,
        "note_families": families.most_common(),
        "notes_top": notes.most_common(20),
        "distinct_failure_signatures": len(signatures),
        "failure_causes": [
            {"cause": c, "count": e["count"], "examples": e["examples"]}
            for c, e in sorted(causes.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
        ],
        "parsed_plate_letters": plate_letters.most_common(12),
        "parsed_unknown_complement_kinds": unknown_kinds.most_common(15),
        "top_signatures": [
            {"signature": sig, "count": e["count"], "examples": e["examples"]} for sig, e in top
        ],
    }


class _PairEvaluator:
    """Streams B, C, D, F and collects a throughput pool over sampled addresses."""

    def __init__(self, pool_limit: int) -> None:
        self.self_total = 0
        self.self_violations = 0
        self.self_examples: list[dict[str, Any]] = []
        self.sym_total = 0
        self.sym_violations = 0
        self.sym_examples: list[dict[str, Any]] = []
        self.pos: dict[str, dict[str, Any]] = {}
        self.neg: dict[str, dict[str, Any]] = {}
        self.neg_scores: Counter[float] = Counter()
        self.conf = {cfg: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cfg in F_CONFIGS}
        self.pool: list[tuple[str, str]] = []
        self.pool_limit = pool_limit
        self.addresses_used = 0

    def _sym(self, a: str, b: str, r_ab: Any, kind: str) -> None:
        r_ba = match(b, a)
        self.sym_total += 1
        if r_ab.score != r_ba.score or r_ab.is_match != r_ba.is_match:
            self.sym_violations += 1
            _example({"kind": kind, "a": a, "b": b, "score_ab": r_ab.score, "score_ba": r_ba.score}, self.sym_examples)

    def _confusion(self, label: bool, a: str, b: str, r0: Any) -> None:
        need = any(f.field == "plate" and f.status == "difiere" for f in r0.fields)
        scores = {0: r0.score}
        for tol in (1, 2):
            scores[tol] = match(a, b, plate_tolerance=tol).score if need else r0.score
        for (thr, tol), cell in self.conf.items():
            predicted = scores[tol] >= thr
            key = ("tp" if predicted else "fn") if label else ("fp" if predicted else "tn")
            cell[key] += 1

    def process(self, addr: str) -> None:
        self.addresses_used += 1
        self.self_total += 1
        r_self = match(addr, addr)
        if not r_self.is_match or r_self.score != 1.0:
            self.self_violations += 1
            _example({"a": addr, "score": r_self.score, "reason": r_self.reason}, self.self_examples)
        if len(self.pool) < self.pool_limit:
            self.pool.append((addr, addr))

        for kind, text in positive_variants(addr).items():
            r = match(addr, text)
            stat = self.pos.setdefault(kind, {"n": 0, "hit": 0, "unparseable": 0, "examples": [], "scores": Counter()})
            stat["n"] += 1
            stat["scores"][r.score] += 1
            if r.score >= HIT_THRESHOLD:
                stat["hit"] += 1
            else:
                if not r.parsed_b.parse_ok:
                    stat["unparseable"] += 1
                _example({"address": addr, "variant": text, "score": r.score, "reason": r.reason[:140]}, stat["examples"])
            self._confusion(True, addr, text, r)
            if len(self.pool) < self.pool_limit:
                self.pool.append((addr, text))

        for kind, text in negative_mutations(addr).items():
            r = match(addr, text)
            expected = EXPECTED_MUTATION_OUTCOME.get(kind, "no_match")
            stat = self.neg.setdefault(
                kind,
                {"n": 0, "as_designed": 0, "veto": 0, "penalized": 0, "passed": 0,
                 "expected": expected, "examples": [], "scores": Counter()},
            )
            stat["n"] += 1
            stat["scores"][r.score] += 1
            passed = r.score >= HIT_THRESHOLD
            if r.score == 0.0:
                stat["veto"] += 1
            elif passed:
                stat["passed"] += 1
            else:
                stat["penalized"] += 1
            if passed == (expected == "match"):
                stat["as_designed"] += 1
            else:
                _example({"address": addr, "mutant": text, "score": r.score, "reason": r.reason[:140]}, stat["examples"])
            if expected == "no_match":
                self.neg_scores[r.score] += 1
                self._confusion(False, addr, text, r)
            self._sym(addr, text, r, kind)
            if len(self.pool) < self.pool_limit:
                self.pool.append((addr, text))


def _finalize_kind_stats(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in sorted(stats):
        stat = dict(stats[kind])
        stat["scores"] = _summarize_counts(stat["scores"])
        out[kind] = stat
    return out


def measure_pairs(
    sample: Sequence[str], seed: int, pool_limit: int = 20_000
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Tables B, C, D, F over the sampled parseable addresses."""
    rng = random.Random(seed)
    order = list(sample)
    rng.shuffle(order)
    ev = _PairEvaluator(pool_limit)
    for addr in order:
        ev.process(addr)

    # Symmetry on random cross pairs (mostly vetoed, still must be symmetric).
    cross_pairs = 0
    for i in range(0, len(order) - 1, 2):
        a, b = order[i], order[i + 1]
        r_ab = match(a, b)
        ev.sym_total += 1
        cross_pairs += 1
        r_ba = match(b, a)
        if r_ab.score != r_ba.score or r_ab.is_match != r_ba.is_match:
            ev.sym_violations += 1
            _example({"kind": "cross_random", "a": a, "b": b, "score_ab": r_ab.score, "score_ba": r_ba.score}, ev.sym_examples)

    f_rows = []
    for (thr, tol), cell in ev.conf.items():
        p, r, f1 = precision_recall_f1(cell["tp"], cell["fp"], cell["fn"])
        f_rows.append({"threshold": thr, "plate_tolerance": tol, **cell, "precision": p, "recall": r, "f1": f1})

    result = {
        "sample_parseable": len(order),
        "B": {
            "self_match_total": ev.self_total,
            "self_match_violations": ev.self_violations,
            "self_match_examples": ev.self_examples,
            "symmetry_total": ev.sym_total,
            "symmetry_cross_random_pairs": cross_pairs,
            "symmetry_violations": ev.sym_violations,
            "symmetry_examples": ev.sym_examples,
        },
        "C": _finalize_kind_stats(ev.pos),
        "D": _finalize_kind_stats(ev.neg),
        "D_overall_no_match_expected": _summarize_counts(ev.neg_scores),
        "F": f_rows,
    }
    return result, ev.pool


def _sample_group_pairs(
    groups: Iterable[list[int]], max_pairs: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Sample about ``max_pairs`` index pairs, proportionally to group size."""
    glist = [g for g in groups if len(g) >= 2]
    total = sum(len(g) * (len(g) - 1) // 2 for g in glist)
    chosen: list[tuple[int, int]] = []
    for g in glist:
        m = len(g)
        pairs_g = m * (m - 1) // 2
        quota = pairs_g if total <= max_pairs else max(1, round(max_pairs * pairs_g / total))
        if quota >= pairs_g:
            picks: Iterable[tuple[int, int]] = combinations(range(m), 2)
        elif quota * 2 >= pairs_g:
            picks = rng.sample(list(combinations(range(m), 2)), quota)
        else:
            seen: set[tuple[int, int]] = set()
            while len(seen) < quota:
                i, j = rng.randrange(m), rng.randrange(m)
                if i != j:
                    seen.add((min(i, j), max(i, j)))
            picks = sorted(seen)
        chosen.extend((g[i], g[j]) for i, j in picks)
    return chosen


def _differing_fields(result: Any) -> tuple[str, ...]:
    return tuple(
        c.field for c in result.fields if c.status not in {"igual", "ausente_en_ambos", "equivalente"}
    )


def measure_false_links(
    addresses: Sequence[object],
    parcels: Sequence[object] | None,
    parsed: Sequence[CanonicalAddress],
    seed: int,
    max_pairs: int = 200_000,
) -> dict[str, Any]:
    """Table E: false-link risk inside the cadastre itself."""
    rng = random.Random(seed)
    ok_idx = [i for i, p in enumerate(parsed) if p.parse_ok]
    have_parcels = parcels is not None

    def same_parcel(i: int, j: int) -> bool:
        return have_parcels and parcels[i] is not None and parcels[i] == parcels[j]  # type: ignore[index]

    # E1: pairs on the same block face (same via + cross signature).
    faces: dict[tuple[object, object], list[int]] = defaultdict(list)
    for i in ok_idx:
        faces[(via_signature(parsed[i]), cross_signature(parsed[i]))].append(i)
    face_pairs = _sample_group_pairs(faces.values(), max_pairs, rng)
    cats: dict[str, dict[str, Any]] = {}
    skipped_same_parcel = 0
    for i, j in face_pairs:
        if same_parcel(i, j):
            skipped_same_parcel += 1
            continue
        cat = classify_pair(parsed[i], parsed[j])
        r = match(addresses[i], addresses[j])
        stat = cats.setdefault(cat, {"n": 0, "at_or_above_threshold": 0, "scores": Counter(), "examples": []})
        stat["n"] += 1
        stat["scores"][r.score] += 1
        if r.score >= HIT_THRESHOLD:
            stat["at_or_above_threshold"] += 1
            _example({"a": addresses[i], "b": addresses[j], "score": r.score}, stat["examples"])
    for stat in cats.values():
        stat["scores"] = _summarize_counts(stat["scores"])

    # E2: ambiguity inherent to the base.
    canon: dict[str, list[int]] = defaultdict(list)
    raw_counts: Counter[str] = Counter()
    for i in ok_idx:
        canon[render(parsed[i])].append(i)
        raw_counts[str(addresses[i])] += 1
    dup_groups = 0
    dup_rows = 0
    dup_pairs = 0
    max_group = 0
    size_hist: Counter[str] = Counter()
    for members in canon.values():
        if len(members) < 2:
            continue
        distinct = len({parcels[i] for i in members}) if have_parcels else len(members)  # type: ignore[index]
        if distinct < 2:
            continue
        dup_groups += 1
        dup_rows += len(members)
        dup_pairs += distinct * (distinct - 1) // 2
        max_group = max(max_group, len(members))
        size_hist["2" if len(members) == 2 else "3" if len(members) == 3 else "4" if len(members) == 4 else "5+"] += 1
    bases: dict[object, list[int]] = defaultdict(list)
    for i in ok_idx:
        bases[base_key(parsed[i])].append(i)
    identical_pairs = 0
    bare_vs_complement_pairs = 0
    multi_complement_bases = 0
    for members in bases.values():
        m = len(members)
        if m < 2:
            continue
        by_comp: Counter[tuple[tuple[str, str], ...]] = Counter(parsed[i].complement for i in members)
        identical_pairs += sum(k * (k - 1) // 2 for k in by_comp.values())
        bare = by_comp.get((), 0)
        bare_vs_complement_pairs += bare * (m - bare)
        if len(by_comp) > 1:
            multi_complement_bases += 1
    e2 = {
        "have_parcel_ids": have_parcels,
        "raw_string_duplicates": sum(1 for c in raw_counts.values() if c > 1),
        "rendered_groups_with_multiple_parcels": dup_groups,
        "rows_in_those_groups": dup_rows,
        "largest_group": max_group,
        "group_size_histogram": dict(size_hist),
        "pairs_scoring_1_0_but_different_parcels": dup_pairs,
        "base_groups_with_more_than_one_row": sum(1 for m in bases.values() if len(m) > 1),
        "base_groups_with_multiple_complement_variants": multi_complement_bases,
        "base_pairs_identical_complement_or_both_bare": identical_pairs,
        "base_pairs_one_side_bare_designed_match_0_98": bare_vs_complement_pairs,
    }

    # E3: hard negatives across faces (different base, same plate).
    by_via: dict[tuple[object, object], list[int]] = defaultdict(list)
    by_cross: dict[tuple[object, object], list[int]] = defaultdict(list)
    for i in ok_idx:
        p = parsed[i]
        by_via[(via_signature(p), _plate_key(p))].append(i)
        by_cross[(cross_signature(p), _plate_key(p))].append(i)
    hard_cap = max(1, max_pairs // 2)
    hard: dict[str, dict[str, Any]] = {}
    for label, groups in (("same_via_same_plate_other_cross", by_via), ("same_cross_same_plate_other_via", by_cross)):
        stat = {"n": 0, "at_or_above_threshold": 0, "scores": Counter(), "examples": [], "differing_fields": Counter()}
        for i, j in _sample_group_pairs(groups.values(), hard_cap, rng):
            if same_parcel(i, j) or classify_pair(parsed[i], parsed[j]) != "different_base_other":
                continue
            r = match(addresses[i], addresses[j])
            stat["n"] += 1
            stat["scores"][r.score] += 1
            if r.score >= HIT_THRESHOLD:
                stat["at_or_above_threshold"] += 1
                stat["differing_fields"][",".join(_differing_fields(r))] += 1
                _example({"a": addresses[i], "b": addresses[j], "score": r.score}, stat["examples"])
        stat["scores"] = _summarize_counts(stat["scores"])
        stat["differing_fields"] = stat["differing_fields"].most_common(10)
        hard[label] = stat

    return {
        "faces_with_2plus_records": sum(1 for g in faces.values() if len(g) >= 2),
        "face_pairs_available": sum(len(g) * (len(g) - 1) // 2 for g in faces.values()),
        "face_pairs_sampled": len(face_pairs),
        "skipped_same_parcel_id": skipped_same_parcel,
        "E1_same_face": cats,
        "E2_ambiguity": e2,
        "E3_cross_face_hard_negatives": hard,
    }


def measure_throughput(pool: Sequence[tuple[str, str]], parse_seconds: float, parse_count: int) -> dict[str, Any]:
    """Table G: single-thread match pairs/second and mean parse time."""
    n = len(pool)
    start = time.perf_counter()
    for a, b in pool:
        match(a, b)
    elapsed = time.perf_counter() - start
    return {
        "pairs_timed": n,
        "match_seconds": elapsed,
        "pairs_per_second": (n / elapsed) if elapsed > 0 else None,
        "mean_match_microseconds": (1e6 * elapsed / n) if n else None,
        "parse_addresses": parse_count,
        "parse_seconds": parse_seconds,
        "mean_parse_microseconds": (1e6 * parse_seconds / parse_count) if parse_count else None,
    }


# ---------------------------------------------------------------------------
# Section H: linking against the whole cadastre with AddressIndex
# ---------------------------------------------------------------------------


def process_memory_bytes() -> tuple[int, int] | None:
    """``(current, peak)`` resident memory of this process, or ``None``.

    Stdlib only: ``ctypes`` + ``GetProcessMemoryInfo`` on Windows,
    ``/proc/self/statm`` plus ``resource`` on POSIX. Returns ``None`` when
    the platform offers neither.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        current = peak
        statm = Path("/proc/self/statm")
        if statm.is_file():
            current = int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return current, max(peak, current)
    except (ImportError, OSError, AttributeError, ValueError):
        return None


def evaluate_link_queries(
    index: AddressIndex, queries: Sequence[tuple[Hashable, object]], threshold: float = HIT_THRESHOLD
) -> dict[str, Any]:
    """Run ``(own_id, query_text)`` queries against ``index`` and score them.

    ``false_best`` is a best candidate that is not the own parcel and not
    part of an ambiguity tie that contains the own parcel; a tie where the
    own parcel merely loses the deterministic id tie-break is counted as
    ``tie_wrong_first`` instead. ``own_missing`` counts queries whose own
    parcel is not among the candidates at all.
    """
    n = len(queries)
    stats: dict[str, Any] = {
        "n": n, "top1_correct": 0, "own_missing": 0, "ambiguous": 0, "false_best": 0,
        "tie_wrong_first": 0, "no_best": 0, "unparseable_queries": 0,
    }
    total_candidates = 0
    total_compared = 0
    max_candidates = 0
    max_compared = 0
    false_examples: list[Any] = []
    missing_examples: list[Any] = []
    started = time.perf_counter()
    for own_id, text in queries:
        result = index.find(text, threshold)
        candidates = result.candidates
        total_candidates += len(candidates)
        total_compared += result.n_compared
        max_candidates = max(max_candidates, len(candidates))
        max_compared = max(max_compared, result.n_compared)
        if not result.parse_ok:
            stats["unparseable_queries"] += 1
        best = result.best
        ambiguous = result.ambiguous
        stats["ambiguous"] += int(ambiguous)
        top_ids = [c.record_id for c in candidates if best is not None and c.score == best.score]
        if not any(c.record_id == own_id for c in candidates):
            stats["own_missing"] += 1
            _example({"query": text}, missing_examples)
        if best is None:
            stats["no_best"] += 1
            continue
        if best.record_id == own_id:
            stats["top1_correct"] += 1
        elif ambiguous and own_id in top_ids:
            stats["tie_wrong_first"] += 1
        else:
            stats["false_best"] += 1
            _example({"query": text, "best": render(best.result.parsed_b), "score": best.score}, false_examples)
    elapsed = time.perf_counter() - started
    stats.update(
        avg_candidates=(total_candidates / n) if n else None,
        avg_compared=(total_compared / n) if n else None,
        max_candidates=max_candidates,
        max_compared=max_compared,
        seconds=elapsed,
        queries_per_second=(n / elapsed) if n and elapsed > 0 else None,
        false_best_examples=false_examples,
        own_missing_examples=missing_examples,
    )
    return stats


def build_link_queries(
    addresses: Sequence[object],
    ids: Sequence[Hashable],
    parsed: Sequence[CanonicalAddress],
    seed: int,
    sample: int,
) -> dict[str, list[tuple[Hashable, str]]]:
    """Deterministic ``exact`` and ``variant`` query lists for section H.

    Rows are sampled among the parseable ones; ``exact`` queries use the
    row's own address, ``variant`` queries one seeded formatting variant of
    it (rows without any variant are skipped there).
    """
    ok_idx = [i for i, p in enumerate(parsed) if p.parse_ok]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(ok_idx, min(max(sample, 0), len(ok_idx))))
    exact: list[tuple[Hashable, str]] = []
    variant: list[tuple[Hashable, str]] = []
    for i in chosen:
        addr = addresses[i]
        if not isinstance(addr, str):
            continue
        exact.append((ids[i], addr))
        variants = sorted(positive_variants(addr).items())
        if variants:
            variant.append((ids[i], rng.choice(variants)[1]))
    return {"exact": exact, "variant": variant}


def measure_linking(
    addresses: Sequence[object],
    parcels: Sequence[object] | None,
    parsed: Sequence[CanonicalAddress],
    seed: int,
    sample: int = LINK_SAMPLE,
    plate_tolerance: int = 0,
) -> dict[str, Any]:
    """Table H: build an ``AddressIndex`` over ALL parseable rows and link a
    sample of them back (exact address, and one formatting variant)."""
    ids: list[Hashable] = [
        parcels[i] if parcels is not None and parcels[i] is not None else i  # type: ignore[misc]
        for i in range(len(addresses))
    ]
    ok_idx = [i for i, p in enumerate(parsed) if p.parse_ok]
    memory_before = process_memory_bytes()
    started = time.perf_counter()
    index = AddressIndex.from_records(((ids[i], addresses[i]) for i in ok_idx), plate_tolerance=plate_tolerance)
    build_seconds = time.perf_counter() - started
    memory_after = process_memory_bytes()
    queries = build_link_queries(addresses, ids, parsed, seed, sample)
    memory = None
    if memory_before is not None and memory_after is not None:
        memory = {
            "before_bytes": memory_before[0],
            "after_bytes": memory_after[0],
            "index_delta_bytes": memory_after[0] - memory_before[0],
            "process_peak_bytes": memory_after[1],
        }
    return {
        "index": index.stats(),
        "distinct_ids": len({ids[i] for i in ok_idx}),
        "build_seconds": build_seconds,
        "memory": memory,
        "sample": sample,
        "exact": evaluate_link_queries(index, queries["exact"]),
        "variant": evaluate_link_queries(index, queries["variant"]),
    }


# ---------------------------------------------------------------------------
# Orchestration and report rendering
# ---------------------------------------------------------------------------


def run_evaluation(
    addresses: Sequence[object],
    parcels: Sequence[object] | None,
    *,
    sample: int | None,
    seed: int,
    max_pairs: int = 200_000,
    link_sample: int = LINK_SAMPLE,
    log: Callable[[str], None] = lambda _msg: None,
) -> dict[str, Any]:
    """Run every measurement. ``sample=None`` means all rows."""
    started = time.perf_counter()
    log(f"parsing {len(addresses)} addresses")
    t0 = time.perf_counter()
    parsed = [parse_canonical(a) for a in addresses]
    parse_seconds = time.perf_counter() - t0

    result: dict[str, Any] = {"seed": seed, "rows": len(addresses)}
    result["A"] = measure_parse_coverage(addresses, parsed)

    indices = list(range(len(addresses))) if sample is None else sample_indices(len(addresses), sample, seed)
    sample_addrs = [addresses[i] for i in indices if parsed[i].parse_ok]  # type: ignore[misc]
    result["sample_requested"] = "all" if sample is None else sample
    result["sample_rows"] = len(indices)

    log(f"pair measurements on {len(sample_addrs)} parseable sampled addresses")
    pair_result, pool = measure_pairs(sample_addrs, seed)  # type: ignore[arg-type]
    result.update(pair_result)

    log("false-link analysis")
    result["E"] = measure_false_links(addresses, parcels, parsed, seed, max_pairs)

    log("throughput")
    result["G"] = measure_throughput(pool, parse_seconds, len(addresses))

    log("linking against the full cadastre (index)")
    result["H"] = measure_linking(addresses, parcels, parsed, seed, link_sample)
    result["wall_seconds"] = time.perf_counter() - started
    return result


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(res: dict[str, Any]) -> str:
    """Render the plain-text report for the tables A-G."""
    out: list[str] = []
    a = res["A"]
    out.append("EVALUATION AGAINST THE CADASTRE")
    out.append(f"rows={res['rows']} sample_rows={res['sample_rows']} (requested={res['sample_requested']}) "
               f"parseable_in_sample={res['sample_parseable']} seed={res['seed']}")
    out.append("")
    out.append("== A. Parse coverage (all rows) ==")
    out.append(f"parse_ok: {a['parse_ok']} / {a['total']} = {pct(a['parse_ok'], a['total'])}")
    out.append(f"parse_fail: {a['parse_fail']} = {pct(a['parse_fail'], a['total'])}")
    out.append("")
    out.append("Failure notes (family):")
    out.append(format_table(["family", "count", "share_of_failures"],
                            [[f, n, pct(n, a["parse_fail"])] for f, n in a["note_families"]]))
    out.append("")
    out.append("Failure notes (exact, top 20):")
    out.append(format_table(["note", "count"], a["notes_top"]))
    out.append("")
    out.append("Failure causes (heuristic location of the grammar mismatch):")
    out.append(format_table(
        ["cause", "count", "share", "examples"],
        [[c["cause"], c["count"], pct(c["count"], a["parse_fail"]), " | ".join(c["examples"][:3])]
         for c in a["failure_causes"]]))
    out.append("")
    out.append(f"Failing shape signatures (top 25 of {a['distinct_failure_signatures']}):")
    out.append(format_table(
        ["#", "count", "share", "signature", "examples"],
        [[i + 1, s["count"], pct(s["count"], a["parse_fail"]), s["signature"], " | ".join(s["examples"])]
         for i, s in enumerate(a["top_signatures"])]))
    out.append("")
    out.append(f"Among PARSED addresses: plate letters {a['parsed_plate_letters']}; "
               f"unregistered complement kinds {a['parsed_unknown_complement_kinds']}")
    b = res["B"]
    out.append("")
    out.append("== B. Self-match and symmetry ==")
    out.append(format_table(["check", "total", "violations"], [
        ["match(a, a) is MATCH with score 1.0", b["self_match_total"], b["self_match_violations"]],
        ["match(a, b) == match(b, a) (mutations + random cross pairs)", b["symmetry_total"], b["symmetry_violations"]],
    ]))
    for label, key in (("self-match violations", "self_match_examples"), ("symmetry violations", "symmetry_examples")):
        for ex in b[key]:
            out.append(f"  {label}: {ex}")
    out.append("")
    out.append(f"== C. Recall on format variants (score >= {HIT_THRESHOLD}) ==")
    out.append(format_table(
        ["variant", "n", "hits", "recall", "miss_unparseable", "mean_score"],
        [[k, v["n"], v["hit"], pct(v["hit"], v["n"]), v["unparseable"], _fmt(v["scores"]["mean"])]
         for k, v in res["C"].items()]))
    total_n = sum(v["n"] for v in res["C"].values())
    total_hit = sum(v["hit"] for v in res["C"].values())
    out.append(f"overall: {total_hit} / {total_n} = {pct(total_hit, total_n)}")
    for kind, v in res["C"].items():
        for ex in v["examples"]:
            out.append(f"  miss[{kind}]: {ex['address']!r} -> {ex['variant']!r} score={ex['score']} ({ex['reason']})")
    out.append("")
    out.append(f"== D. Rejection of true differences (correct = score < {HIT_THRESHOLD}; "
               "designed-match kinds correct = score >= threshold) ==")
    out.append(format_table(
        ["mutation", "expected", "n", "as_designed", "rate", "veto(0.0)", "penalized", "passed>=0.90", "mean_score"],
        [[k, v["expected"], v["n"], v["as_designed"], pct(v["as_designed"], v["n"]), v["veto"], v["penalized"],
          v["passed"], _fmt(v["scores"]["mean"])] for k, v in res["D"].items()]))
    d_all = res["D_overall_no_match_expected"]
    out.append(f"score distribution over true-difference mutations (n={d_all['count']}): bins={d_all['bins']} "
               f"top_values={d_all['top_values']}")
    for kind in ("complement_added", "complement_removed"):
        if kind in res["D"]:
            out.append(f"designed-match {kind}: top scores={res['D'][kind]['scores']['top_values']}")
    for kind, v in res["D"].items():
        for ex in v["examples"]:
            out.append(f"  off-design[{kind}]: {ex['address']!r} -> {ex['mutant']!r} score={ex['score']}")
    e = res["E"]
    out.append("")
    out.append("== E. False-link risk inside the cadastre ==")
    out.append(f"block faces with >=2 records: {e['faces_with_2plus_records']}; pairs available: "
               f"{e['face_pairs_available']}; sampled: {e['face_pairs_sampled']}; skipped (same parcel id): "
               f"{e['skipped_same_parcel_id']}")
    out.append("E1. Pairs on the same block face, different parcel ids:")
    expected_note = {
        "same_base_complements_differ": "should be veto",
        "different_plate": "should be veto",
        "same_base_one_side_bare": "by design (0.98)",
        "same_base_same_complement": "identical address",
    }
    out.append(format_table(
        ["category", "note", "pairs", ">=0.90", "share", "score_bins"],
        [[c, expected_note.get(c, ""), s["n"], s["at_or_above_threshold"], pct(s["at_or_above_threshold"], s["n"]),
          s["scores"]["bins"]] for c, s in sorted(e["E1_same_face"].items())]))
    for c, s in e["E1_same_face"].items():
        if c in ("same_base_complements_differ", "different_plate"):
            for ex in s["examples"]:
                out.append(f"  unexpected >=0.90 [{c}]: {ex['a']!r} vs {ex['b']!r} score={ex['score']}")
    e2 = e["E2_ambiguity"]
    out.append("E2. Ambiguity inherent to the base:")
    out.append(format_table(["metric", "value"], [[k, v] for k, v in e2.items()]))
    out.append("E3. Different base address (same plate, different cross or via), pairs scoring >= 0.90 are false positives:")
    out.append(format_table(
        ["group", "pairs", ">=0.90", "share", "score_bins"],
        [[lbl, s["n"], s["at_or_above_threshold"], pct(s["at_or_above_threshold"], s["n"]), s["scores"]["bins"]]
         for lbl, s in e["E3_cross_face_hard_negatives"].items()]))
    for lbl, s in e["E3_cross_face_hard_negatives"].items():
        if s["differing_fields"]:
            out.append(f"  differing fields among false positives [{lbl}]: {s['differing_fields']}")
        for ex in s["examples"]:
            out.append(f"  false positive [{lbl}]: {ex['a']!r} vs {ex['b']!r} score={ex['score']}")
    out.append("")
    out.append("== F. Threshold sensitivity (positives = C variants, negatives = D true differences) ==")
    out.append(format_table(
        ["threshold", "plate_tol", "tp", "fp", "fn", "tn", "precision", "recall", "f1"],
        [[f"{r['threshold']:.2f}", r["plate_tolerance"], r["tp"], r["fp"], r["fn"], r["tn"],
          _fmt(r["precision"]), _fmt(r["recall"]), _fmt(r["f1"])] for r in res["F"]]))
    g = res["G"]
    out.append("")
    out.append("== G. Throughput (single thread) ==")
    out.append(format_table(["metric", "value"], [
        ["match pairs timed", g["pairs_timed"]],
        ["match pairs/second", _fmt(g["pairs_per_second"], 0)],
        ["mean match time (us)", _fmt(g["mean_match_microseconds"], 1)],
        ["addresses parsed (all rows)", g["parse_addresses"]],
        ["mean parse time (us/address)", _fmt(g["mean_parse_microseconds"], 1)],
    ]))
    if "H" in res:
        out.extend(_format_linking(res["H"]))
    out.append(f"\nwall time: {res['wall_seconds']:.1f}s")
    return "\n".join(out)


def _format_linking(h: dict[str, Any]) -> list[str]:
    """Render section H (linking against the full cadastre)."""
    idx = h["index"]
    out = ["", f"== H. Linking against the full cadastre (AddressIndex, sample={h['sample']}) =="]
    mem = h["memory"]
    mem_rows: list[list[object]] = (
        [
            ["index build memory delta (MiB, working set)", _fmt(mem["index_delta_bytes"] / 2**20, 1)],
            ["process peak working set (MiB)", _fmt(mem["process_peak_bytes"] / 2**20, 1)],
        ]
        if mem
        else [["memory", "not measurable on this platform"]]
    )
    out.append(format_table(["index metric", "value"], [
        ["records indexed", idx["records"]],
        ["distinct parcel ids among them", h["distinct_ids"]],
        ["blocks", idx["blocks"]],
        ["largest block", idx["max_block_size"]],
        ["build time incl. parsing (s)", _fmt(h["build_seconds"], 2)],
        *mem_rows,
    ]))
    rows = []
    for label, key in (("exact address", "exact"), ("format variant", "variant")):
        q = h[key]
        n = q["n"]
        rows.append([
            label, n, pct(q["top1_correct"], n), pct(q["own_missing"], n), q["own_missing"],
            pct(q["ambiguous"], n), q["tie_wrong_first"], q["false_best"], q["no_best"], q["unparseable_queries"],
            _fmt(q["avg_candidates"], 3), _fmt(q["avg_compared"], 2), q["max_compared"],
            _fmt(q["queries_per_second"], 0),
        ])
    out.append("")
    out.append(format_table(
        ["queries", "n", "top1_own", "own_missing%", "own_missing", "ambiguous", "tie_wrong_first", "false_best",
         "no_best", "unparseable", "avg_cand", "avg_compared", "max_compared", "queries/s"],
        rows))
    for label, key in (("false best", "false_best_examples"), ("own parcel missing", "own_missing_examples")):
        for name in ("exact", "variant"):
            for ex in h[name][key]:
                out.append(f"  {label} [{name}]: {ex}")
    return out


def _json_default(obj: object) -> object:
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_catastro",
        description="Evaluate the address matcher against the Cali cadastral base (parquet).",
    )
    parser.add_argument("--catastro", default=os.environ.get(ENV_CATASTRO),
                        help=f"Path to the cadastral parquet (or env var {ENV_CATASTRO}).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for every random choice (default 42).")
    parser.add_argument("--sample", type=int, default=20_000, help="Sample size for B-D,F,G (default 20000).")
    parser.add_argument("--full", action="store_true", help="Use every row instead of a sample (slow).")
    parser.add_argument("--max-pairs", type=int, default=200_000, help="Cap for sampled pairs in E (default 200000).")
    parser.add_argument("--link-sample", type=int, default=LINK_SAMPLE, help="Rows linked back through the index in H (default 5000).")
    parser.add_argument("--address-col", default="direccion", help="Address column (default direccion).")
    parser.add_argument("--parcel-col", default="numero_predial_nacional", help="Parcel id column.")
    parser.add_argument("--json-out", default=None, help="Also write the results as JSON to this path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    args = _build_arg_parser().parse_args(argv)
    if not args.catastro:
        print(f"error: pass --catastro or set {ENV_CATASTRO}", file=sys.stderr)
        return 2
    if args.sample < 0 or args.max_pairs < 1 or args.link_sample < 0:
        print("error: --sample must be >= 0 and --max-pairs >= 1", file=sys.stderr)
        return 2

    try:
        import pyarrow.parquet as pq  # imported here so the module works without pyarrow
    except ImportError:
        print("error: pyarrow is required (pip install -r requirements-dev.txt)", file=sys.stderr)
        return 2

    path = Path(args.catastro)
    if not path.is_file():
        print(f"error: parquet not found: {path}", file=sys.stderr)
        return 2
    names = pq.ParquetFile(path).schema_arrow.names
    if args.address_col not in names:
        print(f"error: column {args.address_col!r} not in parquet columns {names}", file=sys.stderr)
        return 2
    have_parcel = args.parcel_col in names
    columns = [args.address_col] + ([args.parcel_col] if have_parcel else [])
    table = pq.read_table(path, columns=columns).to_pydict()
    addresses = table[args.address_col]
    parcels = table[args.parcel_col] if have_parcel else None
    if not have_parcel:
        print(f"note: no {args.parcel_col!r} column; E treats every row as a distinct record", file=sys.stderr)

    results = run_evaluation(
        addresses,
        parcels,
        sample=None if args.full else args.sample,
        seed=args.seed,
        max_pairs=args.max_pairs,
        link_sample=args.link_sample,
        log=lambda msg: print(f"[eval] {msg}", file=sys.stderr),
    )
    results["source"] = {"parquet": str(path), "parcel_column_present": have_parcel}
    print(format_report(results))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
