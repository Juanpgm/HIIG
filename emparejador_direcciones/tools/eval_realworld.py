#!/usr/bin/env python3
"""Evaluate the matcher on REAL dirty addresses with GPS-derived labels.

Inputs (all passed on the command line, none hard-coded):

* ``--train-pairs``: labelled split whose rows are all treated as ``gold``;
* ``--dev``: labelled split with a ``gt_quality`` column
  (``gold`` / ``noisy`` / ``no_gt``);
* ``--catastro``: the cadastre (``doc_id`` -> ``direccion``).

Ground truth caveat: the label of a row comes from a GPS point-in-polygon
lookup, which is noisy (the hit is often the neighbouring parcel). ``gold``
rows are those whose text agrees with the labelled parcel's own address, so
they are the only rows used for the headline numbers, and even those are
"gold rows, GPS-derived labels, not population precision". ``noisy`` rows
may be a badly written address of the right parcel or the neighbour, so a
"miss" on them is not a certain error: only their distributions are shown.

Input mode and level
    ``--input raw`` (default) feeds the raw dirty text to the matcher: a STRESS
    test, since the real matcher input is an already-normalized address.
    ``--input normalized`` feeds the chosen normalized column of the dev rows
    (``--normalized-column``); ``train_pairs`` has no normalized column, so only
    dev is evaluated in that mode. Rows whose normalized value is empty or not
    text are excluded and counted.
    Every run scores BOTH matcher levels side by side: ``unit`` (complement
    compared) and ``building`` (complement ignored on both sides); ``--level``
    only picks which one is printed as the headline.

Measurements
    A. positives: raw address vs the cadastre address of its labelled parcel
       (parse rate, recall at the threshold, score histogram, miss reasons);
    B. hard negatives: raw address vs adjacent but DIFFERENT parcels
       (false-match rate by kind of neighbour);
    C. retrieval over the whole cadastre with ``AddressIndex``;
    D. noisy rows: distributions only.

Safety rules enforced in code
    * any path containing ``frozen`` is refused unless ``--final`` is passed,
      and even then a loud warning says the set is burned after use;
    * parquet files are read with an explicit, allow-listed column set;
    * real addresses are written ONLY under ``--out-dir`` (private,
      gitignored); stdout and ``--json-out`` hold aggregates only.

Only the parquet reading needs ``pyarrow`` (imported lazily); everything else
is pure Python and unit tested with stubbed loaders (``tests/test_eval_realworld.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from emparejador import AddressIndex, CanonicalAddress, match_parsed, parse_canonical, render  # noqa: E402
from emparejador.matcher import building_view  # noqa: E402

__all__ = [
    "ALLOWED_COLUMNS",
    "LEVELS",
    "MAX_EXAMPLES",
    "NORMALIZED_COLUMNS",
    "NeighbourIndex",
    "Row",
    "evaluate",
    "fmt_rate",
    "format_comparison",
    "guard_columns",
    "check_out_dir",
    "load_cadastre",
    "load_cadastre_with_parcels",
    "load_rows",
    "main",
    "passes",
    "wilson",
]

HIT_THRESHOLD = 0.90
MAX_EXAMPLES = 5
TOP_REASONS = 10
DEFAULT_OUT_DIR = "data/realworld_private"
HEADLINE_LABEL = "gold rows, GPS-derived labels, not population precision"

LEVELS = ("unit", "building")
DEFAULT_LEVEL = "building"
PARCEL_KEY_CHARS = 21

# Already-normalized text columns of the dev split that may be used as the
# matcher INPUT. ``rule_canonical`` is the deterministic grammar form of the
# input (independent of any cadastre lookup) and is the default.
# ``canonical_address`` is the normalizer's final output, which for matched rows
# is the cadastre record itself, so scoring it against the truth is partly
# circular. ``their_normalized`` is the data source's own normalization. No
# prediction, score or match column is ever allowed.
NORMALIZED_COLUMNS = ("rule_canonical", "canonical_address", "their_normalized")
DEFAULT_NORMALIZED_COLUMN = "rule_canonical"

# The only columns this tool may ever request from any parquet file.
# ``numero_predial_nacional`` is requested ONLY from the cadastre file, only its
# first PARCEL_KEY_CHARS characters (the land parcel) are kept in memory, and it
# is never written to any output.
ALLOWED_COLUMNS = frozenset(
    {"raw_address", "dataset", "gt_doc_id", "gt_predial", "gt_quality", "doc_id", "direccion",
     "numero_predial_nacional", *NORMALIZED_COLUMNS}
)

Loader = Callable[[str, Sequence[str]], Mapping[str, Sequence[Any]]]

HISTOGRAM_BINS = ("0.00", "0.01-0.49", "0.50-0.89", "0.90-0.94", "0.95-0.99", "1.00")

# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def passes(score: float, threshold: float) -> bool:
    """Threshold boundary is inclusive: a score equal to it passes."""
    return score >= threshold


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion; ``None`` when ``total`` is 0."""
    if successes < 0 or total < 0 or successes > total:
        raise ValueError(f"invalid counts: {successes}/{total}")
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(successes: int, total: int) -> str:
    """``12.34% [10.00-15.00] (k/n)`` or ``n/a`` for an empty denominator."""
    ci = wilson(successes, total)
    if ci is None:
        return "n/a"
    return f"{100 * successes / total:.2f}% [{100 * ci[0]:.2f}-{100 * ci[1]:.2f}] ({successes}/{total})"


def _rate(successes: int, total: int) -> float | None:
    return None if total == 0 else successes / total


def _ci(successes: int, total: int) -> list[float] | None:
    ci = wilson(successes, total)
    return None if ci is None else [ci[0], ci[1]]


def _bin(score: float) -> str:
    if score <= 0.0:
        return "0.00"
    if score < 0.50:
        return "0.01-0.49"
    if score < 0.90:
        return "0.50-0.89"
    if score < 0.95:
        return "0.90-0.94"
    if score < 1.0:
        return "0.95-0.99"
    return "1.00"


def _norm_id(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value == int(value) else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _note_family(note: str) -> str:
    return note.split(":", 1)[0]


# ---------------------------------------------------------------------------
# Loading (the single gateway to parquet)
# ---------------------------------------------------------------------------


def guard_columns(columns: Sequence[str]) -> list[str]:
    """Refuse an empty column list or any column outside the allow-list."""
    cols = list(columns)
    if not cols:
        raise ValueError("explicit columns are mandatory")
    extra = sorted(set(cols) - ALLOWED_COLUMNS)
    if extra:
        raise ValueError(f"columns not allowed: {extra}")
    return cols


def pyarrow_loader(path: str, columns: Sequence[str]) -> dict[str, list[Any]]:
    """Read ONLY ``columns`` from a parquet file (imports pyarrow lazily)."""
    cols = guard_columns(columns)
    import pyarrow.parquet as pq  # noqa: PLC0415 - optional dependency

    if not Path(path).is_file():
        raise FileNotFoundError(f"parquet not found: {path}")
    return pq.read_table(path, columns=cols).to_pydict()


@dataclass(frozen=True, slots=True)
class Row:
    """One labelled real-world row. ``raw`` may be any object (None, bytes...)."""

    raw: object
    dataset: str
    doc_id: int | None
    quality: str  # gold | noisy | no_gt | other
    split: str


def load_rows(
    loader: Loader, path: str, split: str, text_column: str = "raw_address"
) -> list[Row]:
    """Load one labelled split. ``train_pairs`` has no quality column: all gold.

    ``text_column`` is the column whose value becomes ``Row.raw`` (the matcher
    input): ``raw_address`` or one of ``NORMALIZED_COLUMNS``. Only that text
    column is requested, so a normalized run never reads the raw text.
    """
    if text_column != "raw_address" and text_column not in NORMALIZED_COLUMNS:
        raise ValueError(f"text column not allowed as matcher input: {text_column!r}")
    has_quality = split != "train_pairs"
    columns = [text_column, "dataset", "gt_doc_id"] + (["gt_quality"] if has_quality else [])
    table = loader(path, guard_columns(columns))
    raws = table[text_column]
    datasets = table["dataset"]
    ids = table["gt_doc_id"]
    qualities = table["gt_quality"] if has_quality else ["gold"] * len(raws)
    rows = []
    for raw, dataset, doc_id, quality in zip(raws, datasets, ids, qualities):
        q = quality if quality in ("gold", "noisy", "no_gt") else "other"
        rows.append(Row(raw, str(dataset), _norm_id(doc_id), q, split))
    return rows


def load_cadastre(loader: Loader, path: str) -> dict[int, str]:
    table = loader(path, guard_columns(["doc_id", "direccion"]))
    cadastre: dict[int, str] = {}
    for doc_id, address in zip(table["doc_id"], table["direccion"]):
        key = _norm_id(doc_id)
        if key is not None and isinstance(address, str):
            cadastre.setdefault(key, address)
    return cadastre


def load_cadastre_with_parcels(loader: Loader, path: str) -> tuple[dict[int, str], dict[int, str]]:
    """Cadastre addresses plus, per record, the land-parcel key (the first
    ``PARCEL_KEY_CHARS`` characters of the national predial number).

    Records whose number is missing or malformed simply have no key (they are
    never filtered). Only the truncated key is kept; the full number is dropped.
    """
    table = loader(path, guard_columns(["doc_id", "direccion", "numero_predial_nacional"]))
    cadastre: dict[int, str] = {}
    parcels: dict[int, str] = {}
    for doc_id, address, npn in zip(table["doc_id"], table["direccion"], table["numero_predial_nacional"]):
        key = _norm_id(doc_id)
        if key is None or not isinstance(address, str) or key in cadastre:
            continue
        cadastre[key] = address
        if isinstance(npn, str) and len(npn) >= PARCEL_KEY_CHARS and npn[:PARCEL_KEY_CHARS].isdigit():
            parcels[key] = npn[:PARCEL_KEY_CHARS]
    return cadastre, parcels


# ---------------------------------------------------------------------------
# Hard-negative neighbour selection
# ---------------------------------------------------------------------------

KIND_ADJACENT = "adjacent_plate"
KIND_LETTER = "plate_letter"
KIND_CROSS = "same_plate_other_cross"
KINDS = (KIND_ADJACENT, KIND_LETTER, KIND_CROSS)


def _via_sig(a: CanonicalAddress) -> tuple:
    return (a.via_type, a.via_number, a.via_letters, a.via_suffix, a.via_suffix_letters, a.via_bis, a.via_quadrant)


def _cross_sig(a: CanonicalAddress) -> tuple:
    return (
        a.cross_type, a.cross_number, a.cross_letters, a.cross_suffix,
        a.cross_suffix_letters, a.cross_bis, a.cross_quadrant,
    )


def _plate_parts(plate: str | None) -> tuple[int | None, str]:
    if not plate:
        return None, ""
    letter = plate[-1] if plate[-1].isalpha() else ""
    digits = plate[:-1] if letter else plate
    return (int(digits) if digits.isdigit() else None), letter


@dataclass(slots=True)
class Selection:
    picks: list[tuple[str, int]] = field(default_factory=list)
    homonyms: int = 0
    same_parcel: int = 0


def _render_at(parsed: CanonicalAddress, level: str) -> str:
    """Rendered address as the given matcher level compares it."""
    return render(building_view(parsed) if level == "building" else parsed)


class NeighbourIndex:
    """Blocking structure to find adjacent-but-different cadastre addresses.

    A neighbour must differ from the truth in via, cross or plate (a gold
    text equals its labelled address, so those are different doors; the pools
    below guarantee it, so a homonym can never be picked and is only counted).
    When ``parcels`` (doc id -> land-parcel key) is given, candidates in the
    same land parcel as the truth are dropped and counted: they are the same
    building or plot, not a different predio.
    """

    def __init__(self, parcels: Mapping[int, str] | None = None) -> None:
        self._entries: dict[int, tuple[tuple, tuple, int | None, str]] = {}
        self._by_face: dict[tuple, list[int]] = defaultdict(list)
        self._by_via_plate: dict[tuple, list[int]] = defaultdict(list)
        self._render_count: dict[str, Counter[str]] = {level: Counter() for level in LEVELS}
        self._render: dict[int, dict[str, str]] = {}
        self._parcels: Mapping[int, str] = parcels or {}

    @classmethod
    def from_cadastre(cls, cadastre: Mapping[int, str], parcels: Mapping[int, str] | None = None) -> NeighbourIndex:
        index = cls(parcels)
        for doc_id in sorted(cadastre):
            parsed = parse_canonical(cadastre[doc_id])
            if not parsed.parse_ok:
                continue
            via, cross = _via_sig(parsed), _cross_sig(parsed)
            number, letter = _plate_parts(parsed.plate)
            index._entries[doc_id] = (via, cross, number, letter)
            index._render[doc_id] = {level: _render_at(parsed, level) for level in LEVELS}
            for level in LEVELS:
                index._render_count[level][index._render[doc_id][level]] += 1
            index._by_face[(via, cross)].append(doc_id)
            if number is not None:
                index._by_via_plate[(via, number)].append(doc_id)
        return index

    def select(self, truth_id: int, rng: random.Random, k: int = 2, level: str = "unit") -> Selection:
        truth = self._entries.get(truth_id)
        if truth is None:
            return Selection()
        via, cross, number, letter = truth
        homonyms = self._render_count[level][self._render[truth_id][level]] - 1
        if number is None:
            return Selection([], homonyms)
        pools: dict[str, list[int]] = {kind: [] for kind in KINDS}
        for doc_id in self._by_face[(via, cross)]:
            other_number, other_letter = self._entries[doc_id][2], self._entries[doc_id][3]
            if other_number is None:
                continue
            if 1 <= abs(other_number - number) <= 2:
                pools[KIND_ADJACENT].append(doc_id)
            elif other_number == number and other_letter != letter:
                pools[KIND_LETTER].append(doc_id)
        for doc_id in self._by_via_plate[(via, number)]:
            if self._entries[doc_id][1] != cross:
                pools[KIND_CROSS].append(doc_id)
        truth_parcel = self._parcels.get(truth_id)
        same_parcel = 0
        for kind in KINDS:
            kept = []
            for d in sorted(pools[kind]):
                if d == truth_id:
                    continue
                if truth_parcel is not None and self._parcels.get(d) == truth_parcel:
                    same_parcel += 1
                    continue
                kept.append(d)
            pools[kind] = kept
        available = [kind for kind in KINDS if pools[kind]]
        chosen = sorted(rng.sample(available, min(k, len(available))), key=KINDS.index)
        return Selection([(kind, rng.choice(pools[kind])) for kind in chosen], homonyms, same_parcel)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class _Group:
    """Accumulator for positives (parse rate, recall, histogram, reasons)."""

    def __init__(self) -> None:
        self.n = 0
        self.parsed = 0
        self.hits = 0
        self.hist: Counter[str] = Counter({b: 0 for b in HISTOGRAM_BINS})
        self.reasons: Counter[str] = Counter()

    def add(self, parsed: bool, hit: bool, score: float, reason: str | None) -> None:
        self.n += 1
        self.parsed += parsed
        self.hits += hit
        self.hist[_bin(score)] += 1
        if reason is not None:
            self.reasons[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "parsed": self.parsed,
            "parse_rate": _rate(self.parsed, self.n),
            "hits": self.hits,
            "recall": _rate(self.hits, self.n),
            "recall_ci": _ci(self.hits, self.n),
            "histogram": dict(self.hist),
            "reasons": self.reasons.most_common(TOP_REASONS),
        }


def _miss_reason(raw: CanonicalAddress, truth: CanonicalAddress, result: Any) -> str:
    if not raw.parse_ok:
        return "parse:" + (_note_family(raw.notes[0]) if raw.notes else "unknown")
    if not truth.parse_ok:
        return "cadastre_unparseable"
    for comparison in result.fields:
        if comparison.veto:
            return "veto:" + comparison.field
    return "below_threshold"


def _latency_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(samples)
    return {
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def _add_example(bucket: dict[str, list[dict[str, Any]]], key: str, record: dict[str, Any]) -> None:
    examples = bucket.setdefault(key, [])
    if len(examples) < MAX_EXAMPLES:
        examples.append(record)


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _same_place(a: CanonicalAddress, b: CanonicalAddress, level: str) -> bool:
    """Same rendered address as the matcher level compares it (at building
    level only unit-inside-a-building complements are ignored)."""
    if not (a.parse_ok and b.parse_ok):
        return False
    return _render_at(a, level) == _render_at(b, level)


def evaluate(
    rows: Sequence[Row],
    cadastre: Mapping[int, str],
    seed: int,
    threshold: float = HIT_THRESHOLD,
    level: str = "unit",
    exclude_empty: bool = False,
    input_mode: str = "raw",
    parcels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Run measurements A-D at one matcher ``level``. Returns aggregates plus a
    ``private`` section (real addresses; must only ever be written under the
    private out-dir).

    ``exclude_empty`` drops rows whose matcher input is empty, blank or not text
    (counted in ``empty_input_excluded``, and ``..._gold`` for gold rows) instead
    of scoring them as misses. At ``level="building"`` retrieval counts a
    candidate as correct when it is the same building (unit-level complements
    ignored). ``parcels`` (doc id -> land-parcel key, local use only) drops hard
    negatives that sit in the truth's own land parcel.

    A hard negative whose cadastre address equals the raw input as the level
    renders it, while the truth's does not, is a ``label_conflict`` (the text
    agrees with the neighbour, not with its GPS label): it is counted apart and
    excluded from the false-match rate.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}: {level!r}")
    counts: Counter[str] = Counter()
    counts["rows_total"] = len(rows)
    seen: set[tuple] = set()
    gold: list[Row] = []
    noisy: list[Row] = []
    for r in rows:
        if exclude_empty and not _has_text(r.raw):
            counts["empty_input_excluded"] += 1
            counts["empty_input_excluded_gold"] += r.quality == "gold"
            continue
        key = (r.split, repr(r.raw), r.doc_id, r.quality)
        if key in seen:
            counts["duplicates_dropped"] += 1
            continue
        seen.add(key)
        if r.quality not in ("gold", "noisy"):
            counts[r.quality] += 1
            continue
        if r.doc_id is None:
            counts["skipped_no_doc_id"] += 1
            continue
        if r.doc_id not in cadastre:
            counts["skipped_missing_cadastre"] += 1
            continue
        counts[r.quality] += 1
        (gold if r.quality == "gold" else noisy).append(r)

    private: dict[str, Any] = {"miss_examples": {}, "false_match_examples": {}, "false_best_examples": [],
                               "parser_first_notes": []}
    groups: dict[str, dict[str, _Group]] = {"by_split": {}, "by_dataset": {}, "by_split_dataset": {}}
    overall = _Group()
    first_notes: Counter[str] = Counter()

    parsed_cache: dict[int, CanonicalAddress] = {}

    def truth_of(doc_id: int) -> CanonicalAddress:
        if doc_id not in parsed_cache:
            parsed_cache[doc_id] = parse_canonical(cadastre[doc_id])
        return parsed_cache[doc_id]

    hn = {"n": 0, "fm": 0, "p_n": 0, "p_fm": 0, "homonyms": 0, "no_neighbours": 0, "label_conflict": 0,
          "same_parcel": 0,
          "by_kind": defaultdict(lambda: {"n": 0, "fm": 0, "p_n": 0, "p_fm": 0, "lc": 0})}
    ret = {"n": 0, "top1": 0, "top5": 0, "no_best": 0, "false_best": 0, "ambiguous": 0}
    latencies: list[float] = []
    noisy_acc = {"n": 0, "parsed": 0, "ge": 0, "none": 0, "same": 0, "other": 0,
                 "hist": Counter({b: 0 for b in HISTOGRAM_BINS})}

    index: AddressIndex | None = None
    neighbours: NeighbourIndex | None = None
    build_s = 0.0
    if gold or noisy:
        t0 = time.perf_counter()
        index = AddressIndex.from_records(cadastre.items())
        neighbours = NeighbourIndex.from_cadastre(cadastre, parcels)
        build_s = time.perf_counter() - t0

    def correct(candidate_id: Hashable, parsed_b: CanonicalAddress, doc_id: int, truth: CanonicalAddress) -> bool:
        if candidate_id == doc_id:
            return True
        return _same_place(parsed_b, truth, level)

    rng = random.Random(seed)
    for r in gold:
        assert index is not None and neighbours is not None and r.doc_id is not None
        raw = parse_canonical(r.raw)
        truth = truth_of(r.doc_id)
        result = match_parsed(raw, truth, threshold, level=level)
        hit = bool(result.is_match)
        reason = None if hit else _miss_reason(raw, truth, result)
        if reason is not None:
            _add_example(private["miss_examples"], reason, {
                "raw": str(r.raw), "truth": cadastre[r.doc_id], "score": result.score,
                "split": r.split, "dataset": r.dataset,
            })
            if reason.startswith("parse:") and raw.notes:
                first_notes[raw.notes[0]] += 1
        for bucket, key in ((groups["by_split"], r.split), (groups["by_dataset"], r.dataset),
                            (groups["by_split_dataset"], f"{r.split}/{r.dataset}")):
            bucket.setdefault(key, _Group()).add(raw.parse_ok, hit, result.score, reason)
        overall.add(raw.parse_ok, hit, result.score, reason)

        # B. hard negatives
        selection = neighbours.select(r.doc_id, rng, level=level)
        hn["homonyms"] += selection.homonyms
        hn["same_parcel"] += selection.same_parcel
        if not selection.picks:
            hn["no_neighbours"] += 1
        for kind, neighbour_id in selection.picks:
            slot = hn["by_kind"][kind]
            neighbour = truth_of(neighbour_id)
            if _same_place(raw, neighbour, level) and not _same_place(raw, truth, level):
                hn["label_conflict"] += 1
                slot["lc"] += 1
                continue
            neg = match_parsed(raw, neighbour, threshold, level=level)
            false_match = bool(neg.is_match)
            for acc in (hn, slot):
                acc["n"] += 1
                acc["fm"] += false_match
                if raw.parse_ok:
                    acc["p_n"] += 1
                    acc["p_fm"] += false_match
            if false_match:
                _add_example(private["false_match_examples"], kind, {
                    "raw": str(r.raw), "neighbour": cadastre[neighbour_id], "truth": cadastre[r.doc_id],
                    "score": neg.score,
                })

        # C. retrieval
        t0 = time.perf_counter()
        found = index.find(r.raw, threshold, level=level)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ret["n"] += 1
        best = found.best
        if best is None:
            ret["no_best"] += 1
        else:
            ret["ambiguous"] += found.ambiguous
            if correct(best.record_id, best.result.parsed_b, r.doc_id, truth):
                ret["top1"] += 1
            else:
                ret["false_best"] += 1
                if len(private["false_best_examples"]) < MAX_EXAMPLES:
                    private["false_best_examples"].append({
                        "raw": str(r.raw), "truth": cadastre[r.doc_id],
                        "best": render(best.result.parsed_b), "score": best.score,
                    })
            if any(correct(c.record_id, c.result.parsed_b, r.doc_id, truth) for c in found.candidates[:5]):
                ret["top5"] += 1

    # D. noisy rows: distributions only
    for r in noisy:
        assert index is not None and r.doc_id is not None
        raw = parse_canonical(r.raw)
        truth = truth_of(r.doc_id)
        result = match_parsed(raw, truth, threshold, level=level)
        noisy_acc["n"] += 1
        noisy_acc["parsed"] += raw.parse_ok
        noisy_acc["ge"] += passes(result.score, threshold)
        noisy_acc["hist"][_bin(result.score)] += 1
        best = index.find(r.raw, threshold, level=level).best
        if best is None:
            noisy_acc["none"] += 1
        elif correct(best.record_id, best.result.parsed_b, r.doc_id, truth):
            noisy_acc["same"] += 1
        else:
            noisy_acc["other"] += 1

    private["parser_first_notes"] = first_notes.most_common(60)
    n_noisy = noisy_acc["n"]
    summary: dict[str, Any] = {
        "meta": {"threshold": threshold, "seed": seed, "headline": HEADLINE_LABEL, "level": level,
                 "input": input_mode},
        "counts": dict(counts) | {k: counts.get(k, 0) for k in (
            "gold", "noisy", "no_gt", "other", "duplicates_dropped",
            "skipped_no_doc_id", "skipped_missing_cadastre", "empty_input_excluded",
            "empty_input_excluded_gold")},
        "positives": {
            "overall": overall.to_dict(),
            **{name: {k: g.to_dict() for k, g in sorted(bucket.items())} for name, bucket in groups.items()},
        },
        "hard_negatives": {
            "n": hn["n"],
            "false_matches": hn["fm"],
            "false_match_rate": _rate(hn["fm"], hn["n"]),
            "false_match_ci": _ci(hn["fm"], hn["n"]),
            "parseable_n": hn["p_n"],
            "parseable_false_matches": hn["p_fm"],
            "parseable_rate": _rate(hn["p_fm"], hn["p_n"]),
            "parseable_ci": _ci(hn["p_fm"], hn["p_n"]),
            "homonyms_excluded": hn["homonyms"],
            "same_parcel_excluded": hn["same_parcel"],
            "label_conflict": hn["label_conflict"],
            "rows_without_neighbours": hn["no_neighbours"],
            "by_kind": {
                kind: {"n": s["n"], "false_matches": s["fm"], "label_conflict": s["lc"], "rate": _rate(s["fm"], s["n"]),
                       "ci": _ci(s["fm"], s["n"]), "parseable_n": s["p_n"],
                       "parseable_false_matches": s["p_fm"], "parseable_rate": _rate(s["p_fm"], s["p_n"])}
                for kind, s in sorted(hn["by_kind"].items())
            },
        },
        "retrieval": {
            "n": ret["n"],
            "top1_correct": ret["top1"],
            "top1_rate": _rate(ret["top1"], ret["n"]),
            "top1_ci": _ci(ret["top1"], ret["n"]),
            "top5_correct": ret["top5"],
            "top5_rate": _rate(ret["top5"], ret["n"]),
            "no_best": ret["no_best"],
            "false_best": ret["false_best"],
            "false_best_rate": _rate(ret["false_best"], ret["n"]),
            "false_best_ci": _ci(ret["false_best"], ret["n"]),
            "ambiguous_best": ret["ambiguous"],
            "latency_ms": _latency_stats(latencies),
            "index_build_s": build_s,
        },
        "noisy": {
            "n": n_noisy,
            "parsed": noisy_acc["parsed"],
            "parse_rate": _rate(noisy_acc["parsed"], n_noisy),
            "histogram": dict(noisy_acc["hist"]),
            "share_ge_threshold": _rate(noisy_acc["ge"], n_noisy),
            "best_none": noisy_acc["none"],
            "best_same_predio": noisy_acc["same"],
            "best_other_predio": noisy_acc["other"],
        },
        "private": private,
    }
    return summary


# ---------------------------------------------------------------------------
# Reporting and CLI
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


WIDE_INTERVAL_BELOW = 500


def format_report(s: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    c = s["counts"]
    meta = s["meta"]
    add(f"threshold={meta['threshold']} seed={meta['seed']} input={meta.get('input', 'raw')} "
        f"level={meta.get('level', 'unit')}")
    add(f"rows={c['rows_total']} gold={c['gold']} noisy={c['noisy']} no_gt={c['no_gt']} "
        f"duplicates_dropped={c['duplicates_dropped']} skipped_no_doc_id={c['skipped_no_doc_id']} "
        f"skipped_missing_cadastre={c['skipped_missing_cadastre']} "
        f"empty_input_excluded={c['empty_input_excluded']} (gold {c['empty_input_excluded_gold']})")
    add("")
    add(f"HEADLINE level={meta.get('level', 'unit')} ({HEADLINE_LABEL})")
    if 0 < s["positives"]["overall"]["n"] < WIDE_INTERVAL_BELOW:
        add(f"  note: only {s['positives']['overall']['n']} gold rows, so the intervals are wide")
    pos = s["positives"]

    def line(name: str, g: Mapping[str, Any]) -> str:
        return (f"  {name:<32} n={g['n']:<5} parse={_pct(g['parse_rate']):<8} "
                f"recall={fmt_rate(g['hits'], g['n'])}")

    add(line("overall", pos["overall"]))
    for section in ("by_split", "by_dataset", "by_split_dataset"):
        for key, g in pos[section].items():
            add(line(key, g))
    add("  score histogram (overall): " + ", ".join(f"{k}={v}" for k, v in pos["overall"]["histogram"].items()))
    add("  top miss reasons (overall, parser-note families only):")
    for reason, count in pos["overall"]["reasons"]:
        add(f"    {count:>5}  {reason}")
    hn = s["hard_negatives"]
    add("")
    add("HARD NEGATIVES (adjacent, different predio)")
    add(f"  all pairs: false-match {fmt_rate(hn['false_matches'], hn['n'])}")
    add(f"  parseable raw only: false-match {fmt_rate(hn['parseable_false_matches'], hn['parseable_n'])}")
    add(f"  homonyms excluded={hn['homonyms_excluded']} rows_without_neighbours={hn['rows_without_neighbours']}"
        f" same_parcel_excluded={hn['same_parcel_excluded']} label_conflict={hn['label_conflict']}"
        " (label_conflict: the text equals the neighbour, not its label; not a false match)")
    for kind, k in hn["by_kind"].items():
        add(f"  {kind:<24} {fmt_rate(k['false_matches'], k['n'])}")
    r = s["retrieval"]
    add("")
    add("RETRIEVAL over the full cadastre")
    add(f"  n={r['n']} top1={fmt_rate(r['top1_correct'], r['n'])} top5={_pct(r['top5_rate'])} "
        f"no_best={r['no_best']} false_best={fmt_rate(r['false_best'], r['n'])} ambiguous_best={r['ambiguous_best']}")
    lat = r["latency_ms"]
    add(f"  latency ms: mean={lat['mean']:.2f} p50={lat['p50']:.2f} p95={lat['p95']:.2f} max={lat['max']:.2f} "
        f"index_build_s={r['index_build_s']:.1f}")
    nz = s["noisy"]
    add("")
    add("NOISY rows (distribution only; NOT part of the headline: a low score may be a badly")
    add("written address of the right parcel or the neighbouring parcel, so it is not a certain miss)")
    add(f"  n={nz['n']} parse={_pct(nz['parse_rate'])} share>=threshold={_pct(nz['share_ge_threshold'])} "
        f"best_none={nz['best_none']} best_same_predio={nz['best_same_predio']} "
        f"best_other_predio={nz['best_other_predio']}")
    add("  histogram: " + ", ".join(f"{k}={v}" for k, v in nz["histogram"].items()))
    return "\n".join(lines)


def format_comparison(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """One line per level with the same headline metrics, side by side."""
    lines = ["LEVEL COMPARISON (same rows, same threshold; aggregates only)"]
    for level, s in summaries.items():
        pos, hn, ret = s["positives"]["overall"], s["hard_negatives"], s["retrieval"]
        lines.append(
            f"  {level:<9} parse={_pct(pos['parse_rate'])} recall={fmt_rate(pos['hits'], pos['n'])} "
            f"hard_neg_false_match={fmt_rate(hn['false_matches'], hn['n'])} "
            f"label_conflict={hn['label_conflict']} "
            f"top1={fmt_rate(ret['top1_correct'], ret['n'])} top5={_pct(ret['top5_rate'])} "
            f"false_best={fmt_rate(ret['false_best'], ret['n'])}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-world evaluation (gold rows, GPS-derived labels).")
    p.add_argument("--train-pairs", default=None,
                   help="labelled train split parquet (all rows gold); required with --input raw, "
                        "ignored with --input normalized (it has no normalized column)")
    p.add_argument("--dev", required=True, help="labelled dev split parquet with gt_quality")
    p.add_argument("--catastro", required=True, help="cadastre parquet (doc_id, direccion)")
    p.add_argument("--input", choices=("raw", "normalized"), default="raw",
                   help="matcher input: raw dirty text (stress test) or a normalized dev column")
    p.add_argument("--normalized-column", choices=NORMALIZED_COLUMNS, default=DEFAULT_NORMALIZED_COLUMN,
                   help="dev column used as input with --input normalized. WARNING: canonical_address "
                        "is circular (for matched rows it IS the cadastre record, so scoring it "
                        "against the truth inflates every metric); prefer rule_canonical")
    p.add_argument("--level", choices=LEVELS, default=DEFAULT_LEVEL,
                   help="matcher level of the headline; both levels are always reported")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="private output directory (gitignored)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=HIT_THRESHOLD)
    p.add_argument("--json-out", default=None, help="optional path for the aggregate-only summary")
    p.add_argument("--final", action="store_true",
                   help="allow a path containing 'frozen'; the frozen set is BURNED after one use")
    return p


_REPO_ROOT = Path(__file__).resolve().parent.parent


def check_out_dir(out_dir: str | Path) -> str | None:
    """Refuse an output directory inside the repository unless git ignores it.

    Real addresses are written under ``--out-dir``; inside the repo that is only
    safe when the path is gitignored. Returns an error message or ``None``.
    """
    target = Path(out_dir)
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    try:
        rel = target.relative_to(_REPO_ROOT)
    except ValueError:
        return None
    try:
        # A child name is appended so that a directory pattern such as
        # ``data/realworld_private/`` matches for a directory that does not exist yet.
        probe = (rel / "_probe").as_posix()
        completed = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "check-ignore", "-q", "--", probe],
            capture_output=True, check=False, timeout=30,
        )
        ignored = completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ignored = False
    if ignored:
        return None
    return (f"refusing --out-dir {target}: it is inside the repository and not gitignored, so real "
            "addresses could be committed. Use a gitignored directory (default data/realworld_private/) "
            "or a path outside the repo.")


def main(argv: Sequence[str] | None = None, loader: Loader | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    args = _build_parser().parse_args(argv)
    normalized = args.input == "normalized"
    if not normalized and not args.train_pairs:
        print("error: --train-pairs is required with --input raw", file=sys.stderr)
        return 2
    paths = [p for p in (None if normalized else args.train_pairs, args.dev, args.catastro) if p]
    frozen = [p for p in paths if "frozen" in str(p).lower()]
    if frozen and not args.final:
        print("error: refusing to touch a 'frozen' file. That set is reserved for ONE final "
              "measurement; pass --final only when that measurement is intended.", file=sys.stderr)
        return 2
    if frozen:
        print("!" * 72 + "\nWARNING: --final with a frozen set. This set is BURNED after this run: "
              "never tune, retry or iterate on it again.\n" + "!" * 72, file=sys.stderr)
    problem = check_out_dir(args.out_dir)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    load = loader or pyarrow_loader
    try:
        if normalized:
            rows = load_rows(load, args.dev, "dev", text_column=args.normalized_column)
        else:
            rows = load_rows(load, args.train_pairs, "train_pairs") + load_rows(load, args.dev, "dev")
        cadastre, parcels = load_cadastre_with_parcels(load, args.catastro)
    except (ImportError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summaries: dict[str, dict[str, Any]] = {}
    privates: dict[str, Any] = {}
    for level in LEVELS:
        summary = evaluate(rows, cadastre, args.seed, args.threshold, level=level,
                           exclude_empty=normalized, input_mode=args.input, parcels=parcels)
        privates[level] = summary.pop("private")
        summaries[level] = summary
    combined = {
        "meta": {"input": args.input, "normalized_column": args.normalized_column if normalized else None,
                 "headline_level": args.level, "threshold": args.threshold, "seed": args.seed},
        "levels": summaries,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(combined, indent=2, ensure_ascii=False)
    (out_dir / "realworld_summary.json").write_text(text, encoding="utf-8")
    (out_dir / "realworld_private_examples.json").write_text(
        json.dumps(privates, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    if normalized:
        print(f"input=normalized column={args.normalized_column}: train_pairs has no normalized "
              "column, so only dev is evaluated")
    print(format_report(summaries[args.level]))
    print("\n" + format_comparison(summaries))
    print(f"\nprivate examples written under {out_dir} (never commit them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
