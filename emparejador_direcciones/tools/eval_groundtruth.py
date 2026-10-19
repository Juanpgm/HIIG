#!/usr/bin/env python3
"""Measure the matcher against a ground-truth pair set, at the door level.

The PRIMARY unit of analysis is the building / predio (street + plate,
ignoring any unit). Labels use two columns:

* ``same_door`` (primary): ``yes`` / ``no`` / ``unsure``. Drives the headline
  precision/recall/F1.
* ``same_unit`` (secondary): ``yes`` / ``no`` / ``unsure`` / ``n/a``. Must be
  ``n/a`` (or blank) when ``same_door`` is ``no`` or when either side has no
  unit/complement. Reported in its own section, never mixed into the headline.

Inputs (produced by ``tools/build_groundtruth.py``, in ``--pairs-dir``):

* ``candidates_to_label.jsonl``  real-data pairs; labels come from ``--labels``
* ``synthetic_positives.jsonl``  pairs that carry ``same_door`` (true)
* ``random_negatives.jsonl``     pairs that carry ``same_door`` (false)

``--labels`` is a CSV (``label_sheet.csv`` shape) or JSONL with ``pair_id``,
``same_door`` and optionally ``same_unit``. Values are ``yes``/``no``/
``unsure`` in any case (blank allowed); JSON booleans ``true``/``false`` count
as ``yes``/``no``. The matcher is run at ``--threshold`` (default 0.90) and a
pair is a MATCH when ``score >= threshold``, exactly like the matcher itself.
``--scores-dir`` reads frozen scores from ``matcher_scores_sidecar.jsonl``
instead of running the matcher (default: live scoring, so a changed matcher is
always measured as it is).

Metrics ignore ``unsure`` and blank labels (both are reported). Precision,
recall and F1 that are undefined for the data at hand are reported as ``n/a``.

Only the standard library and this project's own package are used.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from emparejador import match  # noqa: E402

__all__ = [
    "GroundTruthError",
    "compute_metrics",
    "evaluate",
    "format_ratio",
    "is_novel_tag",
    "load_pairs",
    "load_scores",
    "main",
    "read_labels",
    "render_markdown",
]

DEFAULT_THRESHOLD = 0.90
NOVEL_PREFIX = "novel_"
SIDECAR_NAME = "matcher_scores_sidecar.jsonl"
DOOR_VALUES = ("yes", "no", "unsure")
UNIT_VALUES = ("yes", "no", "unsure", "n/a")
_NA_SPELLINGS = ("n/a", "na", "n.a.", "n/a.")
PAIR_FILES = (
    ("candidates_to_label.jsonl", "candidate"),
    ("synthetic_positives.jsonl", "synthetic"),
    ("random_negatives.jsonl", "sanity"),
)
HEADLINE_LABEL = "Human-labeled candidates: per-stratum rates (stratified sample, not population precision)"


class GroundTruthError(ValueError):
    """Bad input files (duplicate ids, missing columns, malformed rows)."""


def is_novel_tag(tag: object) -> bool:
    """Synthetic tags for spellings the parser is not expected to know."""
    return isinstance(tag, str) and tag.startswith(NOVEL_PREFIX)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(tp: int, fp: int, fn: int, tn: int, excluded_unsure: int = 0) -> dict[str, Any]:
    """Confusion counts plus precision/recall/F1 (``None`` when undefined)."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "n": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "excluded_unsure": excluded_unsure,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


# ---------------------------------------------------------------------------
# Reading labels and pairs
# ---------------------------------------------------------------------------


def _normalize(value: object, allowed: tuple[str, ...], allow_na: bool) -> tuple[str | None, str | None]:
    """Return ``(label, invalid_raw)``; blank gives ``(None, None)``."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return ("yes" if value else "no"), None
    text = str(value).strip().lower()
    if not text:
        return None, None
    if text in ("true", "false"):
        return ("yes" if text == "true" else "no"), None
    if allow_na and text in _NA_SPELLINGS:
        return "n/a", None
    if text in allowed:
        return text, None
    return None, str(value).strip()


def _normalize_door(value: object) -> tuple[str | None, str | None]:
    return _normalize(value, DOOR_VALUES, allow_na=False)


def _normalize_unit(value: object) -> tuple[str | None, str | None]:
    return _normalize(value, UNIT_VALUES, allow_na=True)


def _decode(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GroundTruthError(f"cannot read {path}: {exc}") from exc
    return raw.decode("utf-8-sig")


_LEGACY_MSG = (
    "found the legacy single 'label' column; labels now use two columns: 'same_door' (primary) and "
    "'same_unit' (secondary). See docs/groundtruth.md"
)


def read_labels(path: str | Path) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Read a CSV or JSONL labels file.

    Returns ``(labels, info)``. ``labels`` maps pair_id to the ``same_door``
    value (``yes``/``no``/``unsure`` or ``None`` for blank or invalid).
    ``info["invalid"]`` maps pair_id to an unrecognised ``same_door`` value;
    ``info["units"]`` maps pair_id to ``yes``/``no``/``unsure``/``n/a`` or
    ``None`` (blank, treated as n/a); ``info["invalid_units"]`` maps pair_id
    to an unrecognised ``same_unit`` value. A duplicated pair_id raises
    ``GroundTruthError``; so does a file that only has the legacy ``label``
    column.
    """
    path = Path(path)
    text = _decode(path)
    labels: dict[str, str | None] = {}
    units: dict[str, str | None] = {}
    invalid: dict[str, str] = {}
    invalid_units: dict[str, str] = {}

    def add(pair_id: object, raw_door: object, raw_unit: object) -> None:
        pid = "" if pair_id is None else str(pair_id).strip()
        if not pid:
            return
        if pid in labels:
            raise GroundTruthError(f"duplicate pair_id in labels file: {pid!r}")
        door, bad = _normalize_door(raw_door)
        labels[pid] = door
        if bad is not None:
            invalid[pid] = bad
        unit, bad_unit = _normalize_unit(raw_unit)
        units[pid] = unit
        if bad_unit is not None:
            invalid_units[pid] = bad_unit

    if path.suffix.lower() in (".jsonl", ".json", ".ndjson"):
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GroundTruthError(f"{path}: malformed JSON at line {number}: {exc}") from exc
            if not isinstance(row, dict):
                raise GroundTruthError(f"{path}: line {number} is not a JSON object")
            if "label" in row and "same_door" not in row:
                raise GroundTruthError(f"{path}: line {number}: {_LEGACY_MSG}")
            if "same_door" not in row:
                raise GroundTruthError(
                    f"{path}: line {number}: missing required key 'same_door' (found {sorted(row)}); "
                    "use null or an empty value for a blank label"
                )
            add(row.get("pair_id"), row.get("same_door"), row.get("same_unit"))
    else:
        if text.strip():
            reader = csv.reader(io.StringIO(text, newline=""))
            header = next(reader, None) or []
            columns = [h.strip().lower() for h in header]
            if "pair_id" not in columns:
                raise GroundTruthError(f"{path}: missing required column 'pair_id' (found {columns})")
            if "label" in columns and "same_door" not in columns:
                raise GroundTruthError(f"{path}: {_LEGACY_MSG}")
            if "same_door" not in columns:
                raise GroundTruthError(f"{path}: missing required column 'same_door' (found {columns})")
            id_col = columns.index("pair_id")
            door_col = columns.index("same_door")
            unit_col = columns.index("same_unit") if "same_unit" in columns else None

            def cell(row: list[str], col: int | None) -> str:
                return row[col] if col is not None and col < len(row) else ""

            for row in reader:
                if not row or not any(c.strip() for c in row):
                    continue
                add(cell(row, id_col), cell(row, door_col), cell(row, unit_col))
    return labels, {"invalid": invalid, "units": units, "invalid_units": invalid_units}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(_decode(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GroundTruthError(f"{path}: malformed JSON at line {number}: {exc}") from exc
        if not isinstance(row, dict) or "pair_id" not in row:
            raise GroundTruthError(f"{path}: line {number} has no pair_id")
        rows.append(row)
    return rows


def load_scores(path: str | Path) -> dict[str, float]:
    """Read frozen matcher scores (``{pair_id, score}`` rows) from a sidecar file or its directory."""
    target = Path(path)
    if target.is_dir():
        target = target / SIDECAR_NAME
    if not target.is_file():
        raise GroundTruthError(f"scores file not found: {target}")
    scores: dict[str, float] = {}
    for row in _read_jsonl(target):
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise GroundTruthError(f"{target}: pair {row['pair_id']!r} has no numeric score")
        scores[str(row["pair_id"])] = float(score)
    return scores


def load_pairs(pairs_dir: str | Path) -> list[dict[str, Any]]:
    """Load every pair file present in ``pairs_dir``.

    Each pair gets ``group`` (stratum for candidates, tier otherwise),
    ``kind`` (candidate / synthetic / sanity) and, for the self-labelled
    tiers, ``builtin_label`` (``yes``/``no``, from ``same_door``).
    """
    directory = Path(pairs_dir)
    if not directory.is_dir():
        raise GroundTruthError(f"pairs dir not found: {directory}")
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    found = False
    for name, kind in PAIR_FILES:
        path = directory / name
        if not path.is_file():
            continue
        found = True
        for row in _read_jsonl(path):
            pid = str(row["pair_id"])
            if pid in seen:
                raise GroundTruthError(f"duplicate pair_id across pair files: {pid!r}")
            seen.add(pid)
            builtin = None
            if kind != "candidate":
                builtin = "yes" if row.get("same_door") is True else "no"
            group = str(row.get("stratum", "unknown")) if kind == "candidate" else kind
            pairs.append(
                {
                    "pair_id": pid,
                    "addr_a": row.get("addr_a"),
                    "addr_b": row.get("addr_b"),
                    "kind": kind,
                    "group": group,
                    "builtin_label": builtin,
                    "transform": str(row.get("transform", "unknown")) if kind == "synthetic" else None,
                }
            )
    if not found:
        raise GroundTruthError(f"no pair files found in {directory}")
    return pairs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _default_scorer(plate_tolerance: int) -> Callable[[object, object], float]:
    def scorer(a: object, b: object) -> float:
        return match(a, b, DEFAULT_THRESHOLD, plate_tolerance).score

    return scorer


def _empty() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def evaluate(
    pairs: list[dict[str, Any]],
    labels: Mapping[str, str | None],
    threshold: float = DEFAULT_THRESHOLD,
    plate_tolerance: int = 0,
    scorer: Callable[[Any, Any], float] | None = None,
    top: int = 10,
    invalid: Mapping[str, str] | None = None,
    units: Mapping[str, str | None] | None = None,
    invalid_units: Mapping[str, str] | None = None,
    scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Merge labels, run the matcher and compute the confusion report.

    ``labels`` holds the ``same_door`` values (headline); ``units`` the
    optional ``same_unit`` values (secondary section only).
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1]: {threshold!r}")
    score_fn = scorer if scorer is not None else _default_scorer(plate_tolerance)
    invalid = dict(invalid or {})
    units = dict(units or {})
    invalid_units = dict(invalid_units or {})
    frozen = dict(scores or {})
    warnings: list[str] = []

    known_ids = {p["pair_id"] for p in pairs}
    unknown = sorted(pid for pid in labels if pid not in known_ids)
    if unknown:
        warnings.append(f"{len(unknown)} label pair_id(s) unknown to the pairs dir (ignored): {', '.join(unknown[:5])}")
    overridden = sorted(
        p["pair_id"] for p in pairs if p["builtin_label"] is not None and labels.get(p["pair_id"]) is not None
    )
    if overridden:
        warnings.append(f"{len(overridden)} label(s) for synthetic/sanity pairs ignored (they carry their own label)")

    counts: dict[str, dict[str, int]] = {}
    by_transform: dict[str, dict[str, int]] = {}
    recall_split = {"known_vocab": _empty(), "novel": _empty()}
    unsure_by_group: dict[str, int] = {}
    unit_counts: dict[str, dict[str, int]] = {}
    unit_unsure_by_group: dict[str, int] = {}

    def bucket(name: str) -> dict[str, int]:
        return counts.setdefault(name, _empty())

    invalid_unit_ids: list[str] = []
    human = {"pairs": 0, "labeled": 0, "unsure": 0, "blank": 0, "invalid": 0}
    unit_cov = {"yes": 0, "no": 0, "unsure": 0, "n/a": 0, "invalid": 0, "ignored_door_no": 0}
    inconsistent: list[str] = []
    false_pos: list[dict[str, Any]] = []
    false_neg: list[dict[str, Any]] = []
    frozen_used = 0

    def score_of(pair: Mapping[str, Any]) -> float:
        nonlocal frozen_used
        if pair["pair_id"] in frozen:
            frozen_used += 1
            return frozen[pair["pair_id"]]
        return float(score_fn(pair["addr_a"], pair["addr_b"]))

    for pair in pairs:
        pid = pair["pair_id"]
        unit_label: str | None = None
        if pair["builtin_label"] is not None:
            label: str | None = pair["builtin_label"]
        else:
            human["pairs"] += 1
            label = labels.get(pid)
            if label is None:
                human["invalid" if pid in invalid else "blank"] += 1
                continue
            if label == "unsure":
                human["unsure"] += 1
                unsure_by_group[pair["group"]] = unsure_by_group.get(pair["group"], 0) + 1
                continue
            human["labeled"] += 1
            raw_unit = units.get(pid)
            if pid in invalid_units:
                # Counted and reported for both door labels: never dropped silently.
                unit_cov["invalid"] += 1
                invalid_unit_ids.append(pid)
            elif label == "no":
                if raw_unit in ("yes", "no", "unsure"):
                    inconsistent.append(pid)
                    unit_cov["ignored_door_no"] += 1
            elif raw_unit in ("yes", "no", "unsure"):
                unit_cov[raw_unit] += 1
                unit_label = raw_unit
            else:
                unit_cov["n/a"] += 1
        if label not in ("yes", "no"):
            continue
        score = score_of(pair)
        predicted = score >= threshold
        outcome = ("tp" if predicted else "fn") if label == "yes" else ("fp" if predicted else "tn")
        for name in ("overall", pair["group"]) + (("human",) if pair["kind"] == "candidate" else ()):
            bucket(name)[outcome] += 1
        if pair["transform"] is not None:
            t = by_transform.setdefault(pair["transform"], _empty())
            t[outcome] += 1
            recall_split["novel" if is_novel_tag(pair["transform"]) else "known_vocab"][outcome] += 1
        if unit_label == "unsure":
            unit_unsure_by_group[pair["group"]] = unit_unsure_by_group.get(pair["group"], 0) + 1
        elif unit_label in ("yes", "no"):
            unit_outcome = ("tp" if predicted else "fn") if unit_label == "yes" else ("fp" if predicted else "tn")
            for name in ("unit_overall", pair["group"]):
                unit_counts.setdefault(name, _empty())[unit_outcome] += 1
        record = {"pair_id": pid, "group": pair["group"], "addr_a": pair["addr_a"], "addr_b": pair["addr_b"], "score": score}
        if outcome == "fp":
            false_pos.append(record)
        elif outcome == "fn":
            false_neg.append(record)

    if inconsistent:
        warnings.append(
            f"{len(inconsistent)} pair(s) have same_door=no but a same_unit value; same_unit must be n/a or "
            f"blank there, so it was ignored (door label kept): {', '.join(sorted(inconsistent)[:5])}"
        )
    if invalid_unit_ids:
        warnings.append(
            f"{len(invalid_unit_ids)} pair(s) have an invalid same_unit value (counted as invalid, not scored): "
            f"{', '.join(sorted(invalid_unit_ids)[:5])}"
        )
    if frozen_used:
        warnings.append(f"{frozen_used} pair(s) scored from the frozen sidecar (not the live matcher)")

    def metrics_for(name: str) -> dict[str, Any]:
        c = counts.get(name, _empty())
        return compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"], unsure_by_group.get(name, 0))

    group_names = sorted(n for n in counts if n not in ("overall", "human"))
    for name in unsure_by_group:
        if name not in group_names:
            group_names.append(name)
    overall = metrics_for("overall")
    overall["excluded_unsure"] = human["unsure"]
    human_metrics = metrics_for("human")
    human_metrics["excluded_unsure"] = human["unsure"]

    def unit_metrics(name: str, unsure: int) -> dict[str, Any]:
        c = unit_counts.get(name, _empty())
        return compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"], unsure)

    unit_groups = sorted(n for n in unit_counts if n != "unit_overall")
    unit_overall = unit_metrics("unit_overall", unit_cov["unsure"])

    false_pos.sort(key=lambda r: (-r["score"], r["pair_id"]))
    false_neg.sort(key=lambda r: (r["score"], r["pair_id"]))
    return {
        "threshold": threshold,
        "plate_tolerance": plate_tolerance,
        "unit_of_analysis": "same_door (building / predio)",
        "coverage": {
            "human_pairs": human["pairs"],
            "labeled": human["labeled"],
            "unsure": human["unsure"],
            "blank": human["blank"],
            "invalid": human["invalid"],
            "coverage_pct": (100.0 * human["labeled"] / human["pairs"]) if human["pairs"] else None,
        },
        "unknown_label_ids": unknown,
        "warnings": warnings,
        "overall": overall,
        "human": human_metrics,
        "groups": {name: metrics_for(name) for name in group_names},
        "synthetic_by_transform": {
            tag: compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"]) for tag, c in sorted(by_transform.items())
        },
        "synthetic_recall_split": {
            key: compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"]) for key, c in recall_split.items()
        },
        "same_unit": {
            "coverage": unit_cov,
            "overall": unit_overall,
            "groups": {name: unit_metrics(name, unit_unsure_by_group.get(name, 0)) for name in unit_groups},
        },
        "false_positives": false_pos[:top],
        "false_negatives": false_neg[:top],
    }


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def _metric_row(name: str, m: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
        f"{format_ratio(m['precision'])} | {format_ratio(m['recall'])} | {format_ratio(m['f1'])} | {m['excluded_unsure']} |"
    )


_TABLE_HEAD = [
    "| group | n | TP | FP | FN | TN | precision | recall | F1 | unsure excluded |",
    "|---|---|---|---|---|---|---|---|---|---|",
]


def render_markdown(report: Mapping[str, Any]) -> str:
    cov = report["coverage"]
    cov_pct = "n/a" if cov["coverage_pct"] is None else f"{cov['coverage_pct']:.1f}%"
    lines = [
        "# Ground-truth evaluation",
        "",
        f"Threshold: {report['threshold']:.2f} (score >= threshold is a MATCH), plate tolerance: {report['plate_tolerance']}",
        f"Unit of analysis: {report['unit_of_analysis']}. Headline metrics use same_door only.",
        "",
        "## Coverage of human labels",
        "",
        f"- candidate pairs: {cov['human_pairs']}",
        f"- same_door labeled yes/no: {cov['labeled']} ({cov_pct})",
        f"- unsure (excluded): {cov['unsure']}",
        f"- blank (excluded): {cov['blank']}",
        f"- invalid value (excluded): {cov['invalid']}",
    ]
    for warning in report["warnings"]:
        lines.append(f"- WARNING: {warning}")
    lines += [
        "",
        "## Overall (same_door)",
        "",
        *_TABLE_HEAD,
        _metric_row("Overall (all tiers; mixes synthetic/sanity, not a precision estimate)", report["overall"]),
        _metric_row(HEADLINE_LABEL, report["human"]),
        "",
        "The candidate sample is stratified and oversampled around the threshold, with no weights: the rates above "
        "are per-stratum conditional rates, NOT the matcher's precision/recall on the real population.",
        "",
        "## Per stratum / tier (same_door)",
        "",
        *_TABLE_HEAD,
    ]
    for name, metrics in report["groups"].items():
        lines.append(_metric_row(name, metrics))
    if report["synthetic_by_transform"]:
        split = report["synthetic_recall_split"]
        lines += [
            "", "## Synthetic recall: known vocabulary vs novel", "",
            "| family | n | TP | FN | recall |", "|---|---|---|---|---|",
        ]
        for key, label in (("known_vocab", "known-vocab tags"), ("novel", "novel_* tags")):
            m = split[key]
            lines.append(f"| {label} | {m['n']} | {m['tp']} | {m['fn']} | {format_ratio(m['recall'])} |")
        lines += [
            "",
            "Known-vocab recall is NOT independent evidence: those spellings are the ones the parser was built to "
            "know. Only the novel_* tags (spellings that were outside the parser tables when the set was designed) say something about unseen input; once the parser learns them they become regression coverage.",
            "",
            "## Synthetic recall by transform", "",
            "| transform | n | TP | FN | recall |", "|---|---|---|---|---|",
        ]
        for tag, m in report["synthetic_by_transform"].items():
            lines.append(f"| {tag} | {m['n']} | {m['tp']} | {m['fn']} | {format_ratio(m['recall'])} |")
    unit = report["same_unit"]
    uc = unit["coverage"]
    lines += [
        "",
        "## Same-unit (secondary, never part of the headline)",
        "",
        "Only pairs with same_door=yes and same_unit yes/no. A MATCH on a same_unit=no pair counts as FP here.",
        "",
        f"- same_unit yes: {uc['yes']}, no: {uc['no']}, unsure (excluded): {uc['unsure']}, "
        f"n/a or blank: {uc['n/a']}, invalid: {uc['invalid']}, ignored (same_door=no): {uc['ignored_door_no']}",
        "",
        *_TABLE_HEAD,
        _metric_row("same_unit (all strata)", unit["overall"]),
    ]
    for name, metrics in unit["groups"].items():
        lines.append(_metric_row(name, metrics))
    for title, key in (("False positives (top, highest score first)", "false_positives"),
                       ("False negatives (top, lowest score first)", "false_negatives")):
        lines += ["", f"## {title}", ""]
        rows = report[key]
        if not rows:
            lines.append("None.")
        for r in rows:
            lines.append(f"- [{r['group']}] {r['pair_id']} score={r['score']:.3f}: `{r['addr_a']}` vs `{r['addr_b']}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the matcher against the ground-truth pair set.")
    parser.add_argument("--labels", help="CSV or JSONL with pair_id, same_door, same_unit. Optional.")
    parser.add_argument("--pairs-dir", default="data/groundtruth", help="Directory produced by build_groundtruth.py.")
    parser.add_argument("--scores-dir", default=None,
                        help="Read frozen scores from <dir>/matcher_scores_sidecar.jsonl "
                             "(e.g. data/groundtruth_private). Default: score live with the matcher.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--plate-tolerance", type=int, default=0)
    parser.add_argument("--top", type=int, default=10, help="Max false positives/negatives to list.")
    parser.add_argument("--json-out", help="Also write the report as JSON to this path.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # addresses may hold non-ASCII text; never crash on a narrow console
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        labels: dict[str, str | None] = {}
        info: dict[str, Any] = {"invalid": {}, "units": {}, "invalid_units": {}}
        if args.labels:
            labels, info = read_labels(args.labels)
        pairs = load_pairs(args.pairs_dir)
        scores = load_scores(args.scores_dir) if args.scores_dir else None
        report = evaluate(pairs, labels, threshold=args.threshold, plate_tolerance=args.plate_tolerance,
                          top=args.top, invalid=info["invalid"], units=info["units"],
                          invalid_units=info["invalid_units"], scores=scores)
    except GroundTruthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(render_markdown(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
