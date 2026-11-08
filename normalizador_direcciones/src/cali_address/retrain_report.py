"""Retrain report: loaders (read-only) and pure Markdown renderers for ``docs/retrain_*_nomenclatura.md``.

The report is a tracked document, so it carries AGGREGATES ONLY: counts, rates, metric values, rule names and
pattern names. Two defences keep row-level data out of it:

* deny by default: renderers read an explicit list of fields from every artifact; anything else (sample rows,
  ``disagreements`` lists, file paths, free text) is never copied;
* free labels that come from data (``fuente`` values, rule names, tags) must look like identifiers
  (:func:`safe_label`); anything else is replaced by ``(otra etiqueta)``.

``load_inputs`` is the only function that touches the file system; it tolerates every missing or malformed input
(the matching report cells read ``n/d (pendiente)``). ``render_report`` and everything below it are pure and
deterministic: no clock, no environment; the only date in the text is the optional ``generated_at`` argument.

The verdict is computed from the acceptance results (``acceptance.json`` of the final run, written by
``scripts/compare_runs.py``): any failed criterion -> ``FAIL``; otherwise any criterion that could not be
evaluated -> ``INCONCLUSIVE``; otherwise ``PASS``. Promotion is never decided here. The report shows the
Spanish labels in ``VERDICT_LABEL``.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from typing import Iterable

PENDING = "n/d (pendiente)"
OTHER_LABEL = "(otra etiqueta)"
VERDICT_LABEL = {"PASS": "APRUEBA", "FAIL": "NO APRUEBA", "INCONCLUSIVE": "INCONCLUSO"}
CRITERIA_IDS = tuple(f"A{i}" for i in range(1, 13))
DEFAULT_CONFIDENCE = 0.7
BASELINE_DEFAULT = "baseline_prod_2026_10"
INDEX_MARKER = "index_meta.json"
_LABEL_RE = re.compile(r"^[A-Za-z_(][A-Za-z0-9_().:\-]{0,59}$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_MAX_EVIDENCE = 240
_MAX_REASON = 160

# (key in the flat metrics dict, label, kind); labels start the Markdown rows, so tests and readers can find them.
METRIC_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    ("dev (todas las filas)", (
        ("dev_n", "dev filas", "count"),
        ("dev_ok", "dev OK", "count"),
        ("dev_manzana_precision", "dev precisión manzana", "rate"),
        ("dev_predial_precision", "dev precisión predial exacta", "rate"),
        ("dev_predial_lot_precision", "dev precisión predial de lote", "rate"),
        ("dev_lost_correct", "dev lost_correct", "count"),
    )),
    ("dev (filas gold)", (
        ("dev_gold_n", "dev gold filas", "count"),
        ("dev_gold_ok", "dev gold OK", "count"),
        ("dev_gold_ok_coverage", "dev gold proporción de OK respondidas", "rate"),
        ("dev_gold_manzana_precision", "dev gold precisión manzana", "rate"),
        ("dev_gold_predial_precision", "dev gold precisión predial exacta", "rate"),
        ("dev_gold_predial_lot_precision", "dev gold precisión predial de lote", "rate"),
        ("dev_gold_lost_correct", "dev gold lost_correct", "count"),
    )),
    ("congelada (prueba evaluada una sola vez, --final)", (
        ("frozen_all_ok", "congelada todas OK", "count"),
        ("frozen_all_manzana_precision", "congelada todas precisión manzana", "rate"),
        ("frozen_all_predial_precision", "congelada todas precisión predial exacta", "rate"),
        ("frozen_all_predial_lot_precision", "congelada todas precisión predial de lote", "rate"),
        ("frozen_all_lost_correct", "congelada todas lost_correct", "count"),
        ("frozen_gold_ok", "congelada gold OK", "count"),
        ("frozen_gold_manzana_precision", "congelada gold precisión manzana", "rate"),
        ("frozen_gold_predial_precision", "congelada gold precisión predial exacta", "rate"),
        ("frozen_gold_predial_lot_precision", "congelada gold precisión predial de lote", "rate"),
        ("frozen_gold_lost_correct", "congelada gold lost_correct", "count"),
    )),
)
_LEADERBOARD_KEYS = tuple(k for _, rows in METRIC_GROUPS[:2] for k, _, _ in rows) + ("seconds",)
_LEADERBOARD_FROZEN = {"final_gold_ok": "frozen_gold_ok", "final_gold_predial_precision": "frozen_gold_predial_precision",
                       "final_gold_predial_lot_precision": "frozen_gold_predial_lot_precision"}
_FROZEN_SECTIONS = {"all": "frozen_all", "gold": "frozen_gold"}
_FROZEN_FIELDS = ("ok", "manzana_precision", "predial_precision", "predial_lot_precision", "lost_correct")
_BUCKET_ORDER = ("base_doc_new", "base_doc_merged", "unit", "fallback_doc", "dropped", "excluded_tier",
                 "incomplete_address", "spread_outlier")
_PROBE_BUCKETS = ("both_ok_predial_match", "only_old", "only_new", "both_wrong")


# =====================================================================================================
# scalar helpers (pure)
# =====================================================================================================
def _num(value):
    """``int`` / ``float`` for finite numbers and numeric strings; ``None`` for everything else (bool, NaN, inf, text)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _INT_RE.match(text):
            return int(text)
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    try:                                                   # numpy scalars
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt_rate(value, digits: int = 4) -> str:
    """A rate / metric with ``digits`` decimals; integers stay integers; unusable values read ``n/d (pendiente)``."""
    number = _num(value)
    if number is None:
        return PENDING
    if isinstance(number, int):
        return str(number)
    return f"{number:.{digits}f}"


def fmt_count(value) -> str:
    number = _num(value)
    if number is None:
        return PENDING
    if isinstance(number, float):
        if number != int(number):
            return f"{number:.1f}"
        number = int(number)
    return str(number)


def fmt_float(value, digits: int = 2) -> str:
    number = _num(value)
    return PENDING if number is None else f"{float(number):.{digits}f}"


def fmt_delta(new, base, kind: str = "rate") -> str:
    """Signed difference ``new - base`` (``+1`` / ``-1`` / ``0`` for counts, ``+0.0049`` for rates)."""
    a, b = _num(new), _num(base)
    if a is None or b is None:
        return PENDING
    diff = a - b
    if kind == "count":
        diff = int(round(diff))
        return "0" if diff == 0 else f"{diff:+d}"
    if round(diff, 4) == 0:
        return "0.0000"
    return f"{diff:+.4f}"


def fmt_bool(value, yes: str = "aprueba", no: str = "no aprueba") -> str:
    if isinstance(value, bool):
        return yes if value else no
    return PENDING


def cell(value, limit: int = 160) -> str:
    """One Markdown table cell: single line, no pipes, bounded."""
    text = re.sub(r"\s+", " ", str(value)).strip().replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def safe_label(value) -> str:
    """Identifier-looking labels pass; anything else (addresses, ids, names, paths) becomes ``(otra etiqueta)``."""
    if not isinstance(value, str):
        return OTHER_LABEL
    text = value.strip()
    return text if _LABEL_RE.match(text) else OTHER_LABEL


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _get(d, *path):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def table(headers: list[str], rows: Iterable[Iterable]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(cell(c, 700) for c in row) + " |")
    return out


# =====================================================================================================
# verdict (pure)
# =====================================================================================================
def _result_status(entry: dict) -> str:
    """``pass`` / ``fail`` / ``not_evaluated`` for one acceptance result; a failure signal always wins."""
    status = entry.get("status")
    passed = entry.get("passed")
    if passed is False or status == "fail":
        return "fail"
    if passed is True and status in (None, "pass"):
        return "pass"
    if status == "pass" and passed in (None, True):
        return "pass"
    return "not_evaluated"


def normalize_results(results) -> dict[str, dict]:
    """``{id: entry}`` for the usable entries of an acceptance results list; duplicate ids keep the worst status."""
    if isinstance(results, dict):
        results = results.get("results")
    rank = {"pass": 0, "not_evaluated": 1, "fail": 2}
    out: dict[str, dict] = {}
    for entry in _list(results):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        status = _result_status(entry)
        prev = out.get(entry["id"])
        if prev is None or rank[status] > rank[prev["_status"]]:
            out[entry["id"]] = {**entry, "_status": status}
    return out


def compute_verdict(results) -> dict:
    """Verdict from acceptance results: FAIL if any criterion failed, else INCONCLUSIVE if any A1..A12 was not
    evaluated (or is absent), else PASS. No usable results -> INCONCLUSIVE."""
    by_id = normalize_results(results)
    if not by_id:
        return {"verdict": "INCONCLUSIVE", "passed": [], "failed": [], "not_evaluated": list(CRITERIA_IDS)}
    ids = list(CRITERIA_IDS) + sorted(i for i in by_id if i not in CRITERIA_IDS)
    passed = [i for i in ids if i in by_id and by_id[i]["_status"] == "pass"]
    failed = [i for i in ids if i in by_id and by_id[i]["_status"] == "fail"]
    unknown = [i for i in ids if i not in by_id or by_id[i]["_status"] == "not_evaluated"]
    unknown = [i for i in unknown if i in CRITERIA_IDS or i in by_id]
    verdict = "FAIL" if failed else ("INCONCLUSIVE" if unknown else "PASS")
    return {"verdict": verdict, "passed": passed, "failed": failed, "not_evaluated": unknown}


# =====================================================================================================
# loaders (the only I/O)
# =====================================================================================================
def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _read_csv(path: str) -> list[dict] | None:
    """Rows as dicts; ``None`` when the file is unreadable or empty (no header)."""
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return None
            rows = [dict(r) for r in reader]
            return rows if rows else [{"__header_only__": ",".join(reader.fieldnames)}]
    except (OSError, ValueError, UnicodeDecodeError, csv.Error):
        return None


def _leaderboard(path: str) -> dict[str, dict]:
    rows = _read_csv(path) or []
    out: dict[str, dict] = {}
    for row in rows:
        tag = row.get("tag")
        if tag:
            out[tag] = row                                   # the last row of a tag wins
    return out


def _run_metrics(experiment: dict, leaderboard_row: dict | None) -> dict:
    metrics: dict = {}
    row = _dict(experiment.get("leaderboard_row")) or _dict(leaderboard_row)
    for key in _LEADERBOARD_KEYS:
        if key in row and _num(row[key]) is not None:
            metrics[key] = _num(row[key])
    frozen = _dict(experiment.get("frozen_test"))
    for scope, prefix in _FROZEN_SECTIONS.items():
        section = _dict(frozen.get(scope))
        for field in _FROZEN_FIELDS:
            key = f"test_{field}" if scope == "all" else f"test_gold_{field}"
            if _num(section.get(key)) is not None:
                metrics[f"{prefix}_{field}"] = _num(section[key])
    if not any(k.startswith("frozen_") for k in metrics):
        for src, dst in _LEADERBOARD_FROZEN.items():
            if _num(row.get(src)) is not None:
                metrics[dst] = _num(row[src])
    return metrics


def _curve_of(data) -> list[dict]:
    curve = data.get("curve") if isinstance(data, dict) else data
    return [p for p in _list(curve) if isinstance(p, dict)]


def _csv_groups(run_dir: str, suffix: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, f"strict_eval_*_{suffix}.csv"))):
        name = os.path.basename(path)[len("strict_eval_"):-len(f"_{suffix}.csv")]
        rows = _read_csv(path)
        if rows is not None:
            out[name] = rows
    return out


_INHERIT_KEYS = ("inherits_from_experiment", "inherits_model_from", "source_experiment")
_MAX_INHERIT_HOPS = 6


def _inherited_train_report(exp_root: str, experiment: dict, own_tag: str) -> tuple[dict, str | None]:
    """The ``train_report.json`` of the nearest ancestor (calibrated / fallback variants reuse a trained model).

    Follows ``inherits_from_experiment`` / ``inherits_model_from`` breadth first, bounded and cycle-safe.
    Returns ``({}, None)`` when no ancestor has one."""
    seen = {own_tag}
    queue = [experiment]
    hops = 0
    while queue and hops < _MAX_INHERIT_HOPS:
        hops += 1
        nxt: list[dict] = []
        for exp in queue:
            for key in _INHERIT_KEYS:
                parent = exp.get(key)
                if not isinstance(parent, str) or not parent or os.path.basename(parent) != parent or parent in seen:
                    continue
                seen.add(parent)
                parent_dir = os.path.join(exp_root, parent)
                report = _dict(_load_json(os.path.join(parent_dir, "train_report.json")))
                if report:
                    return report, safe_label(parent)
                nxt.append(_dict(_load_json(os.path.join(parent_dir, "experiment.json"))))
        queue = nxt
    return {}, None


def _curve_from_compare(data) -> list[dict]:
    return [p for p in _list(_dict(data).get("candidate")) if isinstance(p, dict)]


def _load_run(exp_root: str, tag: str, leaderboard: dict[str, dict]) -> dict:
    run_dir = os.path.join(exp_root, tag)
    present = os.path.isdir(run_dir)
    experiment = _dict(_load_json(os.path.join(run_dir, "experiment.json"))) if present else {}
    train = _dict(_load_json(os.path.join(run_dir, "train_report.json"))) if present else {}
    inherited_from = None
    if present and not train:
        train, inherited_from = _inherited_train_report(exp_root, experiment, tag)
    curve = _curve_of(_load_json(os.path.join(run_dir, "confidence_curve.json"))) if present else []
    if present and not curve:
        curve = _curve_from_compare(_load_json(os.path.join(run_dir, "confidence_curve_compare.json")))
    run = {
        "tag": tag, "present": present, "experiment": experiment,
        "train_report": train, "train_inherited_from": inherited_from,
        "confidence_curve": curve,
        "acceptance": _load_json(os.path.join(run_dir, "acceptance.json")) if present else None,
        "probe_agreement": _dict(_load_json(os.path.join(run_dir, "probe_agreement.json"))) if present else {},
        "by_fuente": _csv_groups(run_dir, "by_fuente") if present else {},
        "validation": _csv_groups(run_dir, "validation") if present else {},
        "index_meta": _dict(_load_json(os.path.join(run_dir, INDEX_MARKER))) if present else {},
    }
    run["metrics"] = _run_metrics(experiment, leaderboard.get(tag))
    return run


def _discover_indexes(artifacts_dir: str) -> list[str]:
    names = []
    for path in sorted(glob.glob(os.path.join(artifacts_dir, "index_v2*"))):
        if os.path.isfile(os.path.join(path, INDEX_MARKER)):
            names.append(os.path.basename(path))
    return names


def _dedupe(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if isinstance(item, str) and item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


_CALIBRATED_KEYS = {
    "gold_n": "dev_gold_n", "gold_ok": "dev_gold_ok", "gold_ok_coverage": "dev_gold_ok_coverage",
    "gold_lot_precision": "dev_gold_predial_lot_precision", "gold_manzana_precision": "dev_gold_manzana_precision",
    "all_ok": "dev_ok", "all_lot_precision": "dev_predial_lot_precision", "all_manzana_precision": "dev_manzana_precision",
    "lost_correct": "dev_lost_correct",
}
ONNX_DIRS_DEFAULT = ("onnx_bench_final", "onnx_bench")


def _exp_path(exp_root: str, name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(exp_root, name)


def apply_calibrated_metrics(runs: dict, exp_root: str, specs: Iterable[str]) -> None:
    """Replace the DEV metrics of calibrated variants with the calibrated ones.

    A calibrated experiment directory inherits the ``leaderboard_row`` of the run it was trained in, which was scored
    BEFORE calibration; showing it would mix two things. ``specs`` are ``FILE:KEY=TAG`` strings (``FILE`` under the
    experiments directory or absolute; ``KEY`` a run inside its ``runs`` / ``candidates`` mapping). Anything malformed,
    unreadable or pointing at an unknown tag is ignored. Stale ``dev_*`` / ``seconds`` keys are dropped."""
    for spec in specs or ():
        if not isinstance(spec, str) or "=" not in spec or ":" not in spec.split("=", 1)[0]:
            continue
        left, tag = spec.split("=", 1)
        name, key = left.rsplit(":", 1)
        if not name or not key or tag not in runs:
            continue
        data = _dict(_load_json(_exp_path(exp_root, name)))
        source = _dict(_dict(data.get("runs")).get(key)) or _dict(_dict(data.get("candidates")).get(key))
        if not source:
            continue
        metrics = {k: v for k, v in _dict(runs[tag].get("metrics")).items() if not k.startswith("dev_") and k != "seconds"}
        for src, dst in _CALIBRATED_KEYS.items():
            if _num(source.get(src)) is not None:
                metrics[dst] = _num(source[src])
        runs[tag]["metrics"] = metrics


def _load_onnx(exp_root: str, onnx_dirs: Iterable[str] | None) -> dict:
    """The first benchmark directory holding a usable bench file; later ones are fallbacks (preliminary).

    The main 16-thread file is ``bench.json`` or ``bench_threads16.json``. ``bench_cand_index_fp16.json`` (fp16 index
    variant) is optional. ``dev_metrics.json`` is taken from the chosen directory or, if absent there, from the first
    later directory that has it (it measures accuracy, not speed); its origin is recorded in ``dev_metrics_dir``."""
    names = [n for n in (onnx_dirs if onnx_dirs is not None else ONNX_DIRS_DEFAULT) if isinstance(n, str) and n]
    for index, name in enumerate(names):
        path = _exp_path(exp_root, name)
        bench = {}
        for fname in ("bench.json", "bench_threads16.json"):
            bench = _dict(_load_json(os.path.join(path, fname)))
            if _dict(bench.get("runs")):
                break
        if not _dict(bench.get("runs")):
            continue
        label = safe_label(os.path.basename(os.path.normpath(path)))
        dev, dev_dir = {}, label
        for other in names[index:]:
            opath = _exp_path(exp_root, other)
            dev = _dict(_load_json(os.path.join(opath, "dev_metrics.json")))
            if dev:
                dev_dir = safe_label(os.path.basename(os.path.normpath(opath)))
                break
        return {"dir": label, "preliminary": label != ONNX_DIRS_DEFAULT[0], "bench": bench,
                "bench_threads4": _dict(_load_json(os.path.join(path, "bench_threads4.json"))),
                "bench_fp16": _dict(_load_json(os.path.join(path, "bench_cand_index_fp16.json"))),
                "dev_metrics": dev, "dev_metrics_dir": dev_dir,
                "parity": {k: _dict(_load_json(os.path.join(path, f"parity_{k}_serving.json"))) for k in ("prod", "cand")}}
    return {}


def load_inputs(artifacts_dir: str, baseline_tag: str = BASELINE_DEFAULT, candidates: Iterable[str] = (),
                controls: Iterable[str] = (), final_tag: str | None = None, jev_dirs: Iterable[str] = (),
                indexes: Iterable[str] | None = None, reprojection: str | None = None,
                thresholds: str | None = None, calibrated_metrics: Iterable[str] = (),
                seed_comparison: str | None = None, onnx_dirs: Iterable[str] | None = None) -> dict:
    """Read every input the report can use. Missing or malformed files become ``None`` / empty dicts."""
    art = os.path.abspath(artifacts_dir)
    exp_root = os.path.join(art, "experiments")
    leaderboard = _leaderboard(os.path.join(exp_root, "leaderboard.csv"))
    candidates, controls = _dedupe(candidates), _dedupe(controls)
    run_tags = _dedupe([baseline_tag, *candidates, *controls, final_tag or ""])
    runs = {tag: _load_run(exp_root, tag, leaderboard) for tag in run_tags}
    apply_calibrated_metrics(runs, exp_root, calibrated_metrics)
    final_dir = os.path.join(exp_root, final_tag) if final_tag else None

    if reprojection is None:
        found = sorted(glob.glob(os.path.join(os.path.dirname(art), "train_data", "*", "reprojection_report.json")))
        reprojection = found[0] if found else os.path.join(art, "reprojection_report.json")
    index_names = _dedupe(indexes) if indexes is not None else _discover_indexes(art)
    index_records = {}
    for name in index_names:
        meta = _load_json(os.path.join(art, name, INDEX_MARKER))
        compare = _load_json(os.path.join(art, name, "compare_indexes.json"))
        if not isinstance(compare, dict):
            compare = _load_json(os.path.join(art, f"compare_indexes_{name}.json"))
        index_records[name] = {"meta": _dict(meta), "compare": _dict(compare)}

    jev = []
    for directory in jev_dirs or ():
        data = _load_json(os.path.join(directory, "jev_audit.json"))
        jev.append({"label": safe_label(os.path.basename(os.path.normpath(directory))), "data": _dict(data)})

    final = final_tag if final_tag in runs else None
    return {
        "baseline_tag": baseline_tag, "candidates": candidates, "controls": controls, "final_tag": final,
        "run_order": run_tags, "runs": runs,
        "meta": _dict(_load_json(os.path.join(art, "nomenclatura_meta.json"))),
        "wfs_vs_shp": _dict(_load_json(os.path.join(art, "nomenclatura_wfs_vs_shp.json"))),
        "cleaning": _dict(_load_json(os.path.join(art, "nomenclatura_cleaning_report.json"))),
        "reprojection": _dict(_load_json(reprojection)),
        "indexes": index_records,
        "thresholds": _dict(_load_json(thresholds or os.path.join(exp_root, "retrain_2026_10_acceptance.json"))),
        "jev": jev,
        "summary": _dict(_load_json(os.path.join(final_dir, "summary_for_report.json"))) if final_dir else {},
        "decision": _dict(_load_json(os.path.join(final_dir, "decision.json"))) if final_dir else {},
        "seed_comparison": _dict(_load_json(_exp_path(exp_root, seed_comparison or "seed_comparison.json"))),
        "onnx": _load_onnx(exp_root, onnx_dirs),
    }


# =====================================================================================================
# renderers (pure)
# =====================================================================================================
def _focus_tag(inputs: dict) -> str | None:
    """The run the deltas and the verdict refer to: the final tag, else the first candidate."""
    if inputs.get("final_tag"):
        return inputs["final_tag"]
    cands = inputs.get("candidates") or []
    return cands[0] if cands else None


def _compare_tags(inputs: dict) -> list[str]:
    tags = list(inputs.get("candidates") or []) + list(inputs.get("controls") or [])
    if inputs.get("final_tag") and inputs["final_tag"] not in tags:
        tags.append(inputs["final_tag"])
    return tags


def _metrics(inputs: dict, tag: str | None) -> dict:
    return _dict(_dict(_dict(inputs.get("runs")).get(tag)).get("metrics"))


def _run(inputs: dict, tag: str | None) -> dict:
    return _dict(_dict(inputs.get("runs")).get(tag))


def _acceptance_results(inputs: dict):
    focus = _focus_tag(inputs)
    return _run(inputs, focus).get("acceptance") if focus else None


def _curve_point(curve: list[dict], threshold: float):
    for point in curve:
        t = _num(point.get("threshold"))
        if t is not None and abs(t - threshold) < 1e-9:
            return point
    return None


def _section(n: int, title: str, body: list[str]) -> list[str]:
    return [f"## {n}. {title}", "", *body, ""]


def _first_index_meta(inputs: dict) -> dict:
    focus_meta = _dict(_run(inputs, _focus_tag(inputs)).get("index_meta"))
    if focus_meta:
        return focus_meta
    for record in _dict(inputs.get("indexes")).values():
        if _dict(record.get("meta")):
            return record["meta"]
    return {}


# ---- 1 ---------------------------------------------------------------------------------------------
def _sec_summary(inputs: dict) -> list[str]:
    verdict = compute_verdict(_acceptance_results(inputs))
    focus = _focus_tag(inputs)
    base_tag = inputs.get("baseline_tag")
    body = [f"Veredicto: {VERDICT_LABEL[verdict['verdict']]}", "",
            "Estado de promoción: promoción pendiente de aprobación explícita del usuario", ""]
    if _acceptance_results(inputs) is None:
        body.append(f"- Resultados de aceptación: {PENDING} (no hay `acceptance.json` para la corrida final; el veredicto "
                    "no puede decidirse hasta que `scripts/compare_runs.py` lo haya generado).")
    else:
        body.append(f"- Criterios aprobados: {', '.join(verdict['passed']) or 'ninguno'}.")
        body.append(f"- Criterios no aprobados: {', '.join(verdict['failed']) or 'ninguno'}.")
        body.append(f"- Criterios no evaluados: {', '.join(verdict['not_evaluated']) or 'ninguno'}.")
    if focus:
        m, b = _metrics(inputs, focus), _metrics(inputs, base_tag)
        body.append(f"- Corrida evaluada: `{safe_label(focus)}` frente a la línea base medida `{safe_label(base_tag)}`.")
        for key, label, kind in (("dev_gold_ok", "dev gold OK", "count"),
                                 ("dev_gold_predial_lot_precision", "dev gold precisión predial de lote", "rate"),
                                 ("dev_lost_correct", "dev lost_correct", "count"),
                                 ("frozen_gold_ok", "congelada gold OK", "count")):
            body.append(f"  - {label}: {fmt_rate(m.get(key))} (línea base {fmt_rate(b.get(key))}, "
                        f"diferencia {fmt_delta(m.get(key), b.get(key), kind)}).")
    else:
        body.append(f"- Corrida evaluada: {PENDING} (no se indicó ninguna etiqueta candidata ni final).")
    docs = {name: _get(rec, "meta", "n_docs") for name, rec in _dict(inputs.get("indexes")).items()}
    if docs:
        body.append("- Documentos del índice: " + "; ".join(f"`{safe_label(n)}` = {fmt_count(v)}" for n, v in docs.items()) + ".")
    for item in _list(inputs.get("jev")):
        agreement = _get(item, "data", "agreement")
        if isinstance(agreement, dict):
            body.append(f"- Auditoría JEV `{item['label']}`: concordancia {fmt_rate(agreement.get('rate'))} sobre "
                        f"{fmt_count(agreement.get('n'))} filas etiquetadas (la concordancia no es exactitud).")
    body.append("- Las secciones 13 a 21 resumen la historia en tres etapas, la banda de ruido, la calibración, JEV, ONNX, la "
                "lectura de A1-A12 y las decisiones abiertas. La prueba congelada de la ronda final es una SEGUNDA MIRADA, "
                "no una prueba ciega (sección 13).")
    body.append("- La promoción de cualquier artefacto a `artifacts/` es un paso aparte que requiere la aprobación "
                "explícita de los mantenedores; este informe no la autoriza.")
    return _section(1, "Resumen ejecutivo", body)


# ---- 2 ---------------------------------------------------------------------------------------------
def _sec_scope(inputs: dict) -> list[str]:
    body = [
        "Decisiones que determinan el significado de las cifras (las fechas son las de cada decisión):",
        "",
        "- 2026-10-30: la fuente principal es la capa WFS de IDESC (registros certificados `cert` más los aún no "
        "certificados `pcer`); la copia del shapefile se conserva solo como referencia de validación.",
        "- 2026-10-30: todas las coordenadas del índice provienen de reproyectar `ndcoordx` / `ndcoordy` (EPSG:6249 a "
        "EPSG:4326) localmente con el `.prj` del shapefile; las lon/lat del WFS se conservan solo para auditoría. Las "
        "compuertas de comparación se ajustaron a nivel de lote para el NPN (>= 98 %) y a mediana <= 1 m / p99 <= 2 m "
        "para coordenadas.",
        "- 2026-10-31: solo fuentes nuevas. El índice se construye únicamente con los registros de nomenclatura; el "
        "catastro es de solo lectura (auditoría y compuerta de techo del oráculo) y nunca es fuente de documentos, NPN, "
        "`n_predios`, manzana o centroides. La cobertura de direcciones catastrales sin registro de nomenclatura se "
        "pierde por construcción y se cuantifica en las secciones 4 y 6.",
        "- 2026-10-31: la semántica predial es a nivel de lote. `gt_predial` es catastral y a nivel de lote, mientras que "
        "un NPN de nomenclatura puede llevar un sufijo de unidad; por eso A2/A3 y la compuerta del oráculo comparan "
        "`NPN[:21]`, y la precisión exacta por NPN se reporta junto a ella. La línea base medida se recalcula con la "
        "misma métrica de lote.",
        "- 2026-10-31: la pre-normalización que separa letras (`pre_normalize_separates_letters`) está desactivada por "
        "defecto para que el comportamiento de producción y la línea base no cambien; se activa por artefacto mediante "
        "`index_meta.json`. Los índices nuevos la llevan activada.",
        "- La validación de salidas (`validate_outputs`: off / report / correct) está desactivada por defecto; este "
        "informe muestra el efecto de `report` y `correct` por separado (sección 8).",
        "",
    ]
    rows = []
    for name, rec in _dict(inputs.get("indexes")).items():
        meta = _dict(rec.get("meta"))
        flag = meta.get("pre_normalize_separates_letters")
        rows.append([f"`{safe_label(name)}`", fmt_bool(flag, "true", "false"), fmt_count(meta.get("objectid_offset")),
                     fmt_count(meta.get("schema"))])
    if rows:
        body += ["Indicadores registrados por artefacto:", ""]
        body += table(["Índice", "pre_normalize_separates_letters", "desplazamiento objectid", "esquema de index_meta"], rows)
    else:
        body.append(f"Indicadores registrados por artefacto: {PENDING} (no se encontró `index_v2*/index_meta.json`).")
    return _section(2, "Alcance y decisiones", body)


# ---- 3 ---------------------------------------------------------------------------------------------
def _sec_data(inputs: dict) -> list[str]:
    meta, wvs, clean, repro = (_dict(inputs.get(k)) for k in ("meta", "wfs_vs_shp", "cleaning", "reprojection"))
    cadastre = None
    for rec in _dict(inputs.get("indexes")).values():
        cadastre = _get(rec, "meta", "catastro", "input_docs")
        if cadastre is not None:
            break
    est = _dict(meta.get("estado_counts"))
    body = ["Tres fuentes, una primaria:", ""]
    body += table(["Fuente", "Filas", "Notas"], [
        ["Copia del shapefile (solo certificados)", fmt_count(wfs_value(wvs, repro)), "EPSG:6249, referencia de validación"],
        ["Capa WFS (primaria)", fmt_count(meta.get("rows_downloaded", wvs.get("wfs_rows"))),
         "; ".join(f"{safe_label(k)}={fmt_count(v)}" for k, v in sorted(est.items())) or PENDING],
        ["Catastro de producción (solo lectura)", fmt_count(cadastre), "referencia de auditoría, nunca fuente de documentos"],
    ])
    body += ["", f"Descarga WFS: páginas {fmt_count(meta.get('pages'))}, coincidencias {fmt_count(meta.get('number_matched_last'))}, "
                 f"descargadas {fmt_count(meta.get('rows_downloaded'))}, estado {cell(safe_label(meta.get('status')))}, "
                 f"corte de la capa {cell(meta.get('ndfechcort_max') if isinstance(meta.get('ndfechcort_max'), str) and len(meta.get('ndfechcort_max')) <= 12 else PENDING)}. "
                 "El shapefile (exportado el 2026-10-30) puede ser más reciente que el corte del WFS; la compuerta a nivel "
                 "de lote absorbe las diferencias por sufijo de unidad.", ""]

    body += ["Limpieza (antes de cualquier ingesta):", ""]
    if clean:
        body += table(["Magnitud", "Valor"], [
            ["filas de entrada", fmt_count(clean.get("input_rows"))], ["filas limpias", fmt_count(clean.get("clean_rows"))],
            ["filas en cuarentena", fmt_count(clean.get("quarantined_rows"))],
            ["proporción en cuarentena", fmt_rate(clean.get("quarantine_share"))],
            ["límite de cuarentena", fmt_rate(clean.get("quarantine_share_limit"))],
            ["compuerta", fmt_bool(clean.get("gate_passed"))],
        ])
        rules = _dict(clean.get("rules"))
        if rules:
            body += ["", *table(["Regla de limpieza", "Filas"], [[safe_label(k), fmt_count(v)] for k, v in sorted(rules.items())])]
        literal = _dict(clean.get("status_invalid_literals"))
        if literal:
            body += ["", "Estados excluidos del índice: " + "; ".join(
                f"{safe_label(k)}={fmt_count(v)}" for k, v in sorted(literal.items())) + "."]
        src = _dict(clean.get("coord_source_counts"))
        if src:
            body += ["Fuente de coordenadas: " + "; ".join(f"{safe_label(k)}={fmt_count(v)}" for k, v in sorted(src.items())) + "."]
    else:
        body.append(f"{PENDING} (`nomenclatura_cleaning_report.json` no disponible).")

    body += ["", "Reproyección del shapefile (EPSG:6249 a EPSG:4326):", ""]
    if repro:
        sanity = _dict(repro.get("sanity"))
        body += table(["Magnitud", "Valor"], [
            ["SRC", f"{cell(safe_label_crs(repro.get('crs_source')))} a {cell(safe_label_crs(repro.get('crs_target')))}"],
            ["puntos", fmt_count(repro.get("count"))], ["geometrías NaN", fmt_count(repro.get("nan_geometry"))],
            ["fuera del bbox de Cali", fmt_count(repro.get("outside_bbox"))],
            ["NPN compartidos verificados contra el catastro", fmt_count(sanity.get("shared_npn"))],
            ["desplazamiento mediano (m)", fmt_float(sanity.get("median_shift_m"), 3)],
            ["desplazamiento p90 (m)", fmt_float(sanity.get("p90_shift_m"), 3)],
            ["desplazamiento máximo (m)", fmt_float(sanity.get("max_shift_m"), 1)],
            ["compuerta bbox", fmt_bool(_get(repro, "gates", "bbox_ok"))],
            ["compuerta de cordura (mediana < 30 m advertencia, > 100 m aborta)", fmt_bool(_get(repro, "gates", "sanity_ok"))],
        ])
    else:
        body.append(f"{PENDING} (`reprojection_report.json` no disponible).")

    body += ["", "WFS frente a shapefile (compuertas ajustadas; cada resultado se recalcula a partir del valor y el umbral):", ""]
    if wvs:
        th = {"max_count_diff": 0.005, "min_agreement": 0.995, "min_npn_lot_agreement": 0.98, "max_median_shift_m": 1.0,
              "max_p99_shift_m": 2.0}
        th.update({k: _num(v) for k, v in _dict(wvs.get("thresholds")).items() if _num(v) is not None})

        def verdict(value, op, limit):
            v = _num(value)
            if v is None:
                return PENDING
            return "aprueba" if (v <= limit if op == "<=" else v >= limit) else "no aprueba"

        body += table(["Control", "Valor", "Umbral", "Resultado"], [
            ["proporción de diferencia de conteo certificado", fmt_rate(wvs.get("cert_count_diff_share"), 5), f"<= {th['max_count_diff']}",
             verdict(wvs.get("cert_count_diff_share"), "<=", th["max_count_diff"])],
            ["concordancia de direcciones (ids comunes)", fmt_rate(wvs.get("address_agreement"), 5), f">= {th['min_agreement']}",
             verdict(wvs.get("address_agreement"), ">=", th["min_agreement"])],
            ["concordancia NPN a nivel de lote (NPN[:21])", fmt_rate(wvs.get("npn_lot_agreement")),
             f">= {th['min_npn_lot_agreement']} (98 %)", verdict(wvs.get("npn_lot_agreement"), ">=", th["min_npn_lot_agreement"])],
            ["concordancia NPN, 30 dígitos completos", fmt_rate(wvs.get("npn_agreement")), "informativo", "informativo"],
            ["desplazamiento de coordenadas, mediana (m)", fmt_float(wvs.get("coord_median_shift_m"), 3), f"<= {th['max_median_shift_m']}",
             verdict(wvs.get("coord_median_shift_m"), "<=", th["max_median_shift_m"])],
            ["desplazamiento de coordenadas, p99 (m)", fmt_float(wvs.get("coord_p99_shift_m"), 3), f"<= {th['max_p99_shift_m']}",
             verdict(wvs.get("coord_p99_shift_m"), "<=", th["max_p99_shift_m"])],
            ["ids comunes comparados", fmt_count(wvs.get("common_ndid")), "-", "informativo"],
        ])
        body += ["", "La discrepancia del NPN completo es el sufijo de unidad (últimos dos dígitos); el lote coincide. El "
                     "desplazamiento de coordenadas es una diferencia de transformación de datum entre GeoServer y pyproj, "
                     "no un error de datos."]
    else:
        body.append(f"{PENDING} (`nomenclatura_wfs_vs_shp.json` no disponible).")
    return _section(3, "Perfil de datos", body)


def wfs_value(wvs: dict, repro: dict):
    return wvs.get("shp_rows", repro.get("count"))


def safe_label_crs(value) -> str:
    return value if isinstance(value, str) and re.match(r"^EPSG:\d{4,6}$", value) else PENDING


# ---- 4 ---------------------------------------------------------------------------------------------
def _fuente_text(counts) -> str:
    items = [f"{safe_label(k)}={fmt_count(v)}" for k, v in sorted(_dict(counts).items())]
    return "; ".join(items) if items else PENDING


def _sec_indexes(inputs: dict) -> list[str]:
    records = _dict(inputs.get("indexes"))
    if not records:
        return _section(4, "Composición del índice y contabilidad de registros",
                        [f"{PENDING} (no se encontró `index_v2*/index_meta.json`)."])
    names = list(records)
    metas = {n: _dict(records[n].get("meta")) for n in names}
    body = ["Composición por índice (los documentos son direcciones normalizadas únicas; las unidades viven en un archivo auxiliar):", ""]
    rows = []
    for n in names:
        m = metas[n]
        tiers = m.get("tiers")
        rows.append([f"`{safe_label(n)}`", ",".join(safe_label(t) for t in tiers) if isinstance(tiers, list) and tiers else PENDING,
                     fmt_count(m.get("n_docs")), _fuente_text(m.get("counts_by_fuente")),
                     fmt_rate(m.get("parse_failure_rate"), 4), fmt_rate(m.get("fallback_doc_share_of_docs"), 4),
                     fmt_rate(_get(m, "queries", "official_share"), 4), fmt_count(_get(m, "queries", "synthetic")),
                     fmt_count(_get(m, "queries", "official")), fmt_count(m.get("units_rows"))])
    body += table(["Índice", "Niveles", "Documentos", "Documentos por fuente", "Tasa de fallo del parser",
                   "Proporción de documentos de respaldo", "Proporción de consultas oficiales", "Consultas sintéticas",
                   "Consultas oficiales", "Filas de unidades"], rows)

    bucket_names = list(_BUCKET_ORDER)
    for m in metas.values():
        for b in _dict(_get(m, "accounting", "buckets")):
            if b not in bucket_names:
                bucket_names.append(b)
    bucket_names = [b for b in bucket_names if any(b in _dict(_get(m, "accounting", "buckets")) for m in metas.values())]
    body += ["", "Contabilidad de registros: cada registro de entrada limpio cae en exactamente un grupo.", ""]
    acc_rows = [[safe_label(b), *[fmt_count(_dict(_get(metas[n], "accounting", "buckets")).get(b)) for n in names]]
                for b in bucket_names]
    for label, key in (("total (grupos)", "total"), ("entrada", "input"), ("en cuarentena antes de indexar", "quarantined")):
        acc_rows.append([label, *[fmt_count(_get(metas[n], "accounting", key)) for n in names]])
    acc_rows.append(["contabilidad correcta", *[fmt_bool(_get(metas[n], "accounting", "ok")) for n in names]])
    body += table(["Grupo", *[f"`{safe_label(n)}`" for n in names]], acc_rows)

    body += ["", "Techo del oráculo (mejor coincidencia de lote alcanzable en filas gold) y pérdida de cobertura:", ""]
    orc = []
    for n in names:
        gate = _dict(_get(metas[n], "oracle_ceiling", "gate"))
        total = _dict(_get(gate, "shares", "total"))
        failed = gate.get("failed")
        orc.append([f"`{safe_label(n)}`", fmt_bool(gate.get("ok")), fmt_float(gate.get("max_drop_points"), 1),
                    fmt_float(total.get("production_cond_pct")), fmt_float(total.get("new_cond_pct")),
                    fmt_float(gate.get("coverage_loss_headline_pp")), fmt_float(gate.get("coverage_loss_inherent_pp")),
                    ",".join(safe_label(x) for x in failed) if isinstance(failed, list) and failed else
                    ("ninguna" if isinstance(failed, list) else PENDING)])
    body += table(["Índice", "Compuerta", "Caída máxima (pp)", "Producción coincidente, lotes en la fuente (%)",
                   "Nuevo coincidente, lotes en la fuente (%)", "Pérdida de cobertura, titular (pp)",
                   "Pérdida inherente a la fuente (pp)", "Particiones fallidas"], orc)
    body += ["", "La pérdida titular cuenta cada lote gold que cubre el índice de producción; la parte inherente es el lote "
                 "ausente de la propia fuente de nomenclatura, que ningún modelo puede recuperar. La compuerta de "
                 "construcción compara solo los lotes presentes en la fuente."]

    comp = [(n, _dict(records[n].get("compare"))) for n in names if _dict(records[n].get("compare"))]
    if comp:
        body += ["", "Comparación de índices contra el índice de producción (auditoría de solo lectura):", ""]
        crow = []
        for n, c in comp:
            q = _dict(_get(c, "coordinate_shift", "quantiles_m"))
            cov = _dict(c.get("gold_npn_coverage"))
            crow.append([f"`{safe_label(n)}`", fmt_count(_get(c, "coordinate_shift", "n_shared_npn")), fmt_float(q.get("p50"), 3),
                         fmt_float(q.get("p90"), 3), fmt_float(q.get("p99"), 2), fmt_float(q.get("max"), 1),
                         fmt_rate(_get(cov, "old", "all", "share_present")), fmt_rate(_get(cov, "new", "all", "share_present"))])
        body += table(["Índice", "NPN compartidos", "Desplazamiento p50 (m)", "p90 (m)", "p99 (m)", "máx (m)",
                       "Cobertura de lotes gold, producción", "Cobertura de lotes gold, nuevo"], crow)
    return _section(4, "Composición del índice y contabilidad de registros", body)


# ---- 5 ---------------------------------------------------------------------------------------------
def _sec_training(inputs: dict) -> list[str]:
    tags = list(inputs.get("candidates") or []) + list(inputs.get("controls") or [])
    if inputs.get("final_tag") and inputs["final_tag"] not in tags:
        tags.append(inputs["final_tag"])
    if not tags:
        return _section(5, "Entrenamiento", [f"{PENDING} (no se indicó ninguna corrida candidata ni de control)."])
    def _is_smoke(t):
        return _get(_run(inputs, t), "experiment", "hyperparams", "smoke") is True

    real = [t for t in tags if _run(inputs, t).get("train_report") and not _is_smoke(t)]
    pool = real or [t for t in tags if _run(inputs, t).get("train_report")]
    first = _dict(_run(inputs, pool[0]).get("train_report")) if pool else {}
    cfg = _dict(first.get("config"))
    env = _dict(_dict(_run(inputs, tags[0]).get("experiment")).get("environment"))
    gpu = _dict(first.get("gpu"))
    body = [
        "Receta: receta de producción desde cero (`mode=mixed`, `AddressEncoder` nuevo, con semilla primero), "
        f"épocas configuradas {fmt_count(cfg.get('epochs'))}, batch {fmt_count(cfg.get('batch_size'))}, lr {fmt_rate(cfg.get('lr'), 4)}, "
        f"minutos máximos {fmt_float(cfg.get('max_minutes'), 0)}; las cadenas oficiales son consultas de entrenamiento "
        "(`q_source=official`).",
        "",
        f"Entorno: GPU {cell(gpu.get('name')) if isinstance(gpu.get('name'), str) and len(gpu.get('name')) < 80 else PENDING} "
        f"(capacidad de cómputo {cell(gpu.get('capability')) if isinstance(gpu.get('capability'), (str, int, float)) else PENDING}), "
        f"torch {cell(env.get('torch')) if isinstance(env.get('torch'), str) else PENDING}, "
        f"CUDA {cell(env.get('cuda')) if isinstance(env.get('cuda'), str) else PENDING}.",
        "",
    ]
    rows = []
    for t in tags:
        run = _run(inputs, t)
        tr = _dict(run.get("train_report"))
        done, conf = _num(tr.get("epochs_completed")), _num(tr.get("epochs_configured"))
        epochs = f"{int(done)}/{int(conf)}" if done is not None and conf is not None and conf > 0 else PENDING
        flags = []
        if done is not None and conf is not None and conf > 0 and done < conf:
            flags.append("INCOMPLETO")
        stopped = tr.get("stopped_early")
        if stopped is True:
            flags.append("parada temprana")
        if _is_smoke(t):
            flags.append("corrida de humo")
        if run.get("train_inherited_from"):
            flags.append(f"heredado de {run['train_inherited_from']}")
        recall = _dict(tr.get("final_val_recall"))
        off, syn = _dict(recall.get("official")), _dict(recall.get("synthetic"))
        seconds = _num(tr.get("seconds"))
        rows.append([safe_label(t), fmt_count(_dict(tr.get("config")).get("seed")), epochs,
                     "sí" if stopped is True else ("no" if stopped is False else PENDING),
                     fmt_rate(off.get("recall@1")), fmt_rate(syn.get("recall@1")),
                     fmt_float(seconds, 1) if seconds is not None else PENDING, ", ".join(flags) or ("correcto" if epochs != PENDING else PENDING)])
    body += table(["Corrida", "Semilla", "Épocas completadas/configuradas", "Detenida antes", "Recall@1 de validación, consultas oficiales",
                   "Recall@1 de validación, consultas sintéticas", "Segundos de entrenamiento", "Estado"], rows)
    body += ["", "El recall de validación sintético no es comparable entre índices (documentos distintos, mezcla de corrupción "
                 "distinta); el recall de consultas oficiales es más cercano pero sigue ligado al índice. Ninguno sirve como "
                 "selector: la selección usa solo las métricas dev gold."]
    return _section(5, "Entrenamiento", body)


# ---- 6 ---------------------------------------------------------------------------------------------
def _sec_results(inputs: dict) -> list[str]:
    base = inputs.get("baseline_tag")
    cols = _compare_tags(inputs)
    focus = _focus_tag(inputs)
    if not cols:
        return _section(6, "Resultados antes y después", [f"{PENDING} (no se indicó ninguna etiqueta candidata, de control ni final)."])
    headers = ["Métrica", f"{safe_label(base)} (línea base)", *[safe_label(t) for t in cols],
               f"Diferencia ({safe_label(focus)} - línea base)" if focus else "Diferencia"]
    body = ["Las tasas son fracciones. Las diferencias comparan la corrida evaluada con la línea base MEDIDA (mismo código, "
            "mismas filas). Dev gold es pequeño (289 filas): una variación de 1-2 filas está dentro de la banda de ruido.", ""]
    rows = []
    for title, group in METRIC_GROUPS:
        rows.append([f"**{title}**", *[""] * (len(cols) + 2)])
        for key, label, kind in group:
            bm = _metrics(inputs, base).get(key)
            rows.append([label, fmt_rate(bm), *[fmt_rate(_metrics(inputs, t).get(key)) for t in cols],
                         fmt_delta(_metrics(inputs, focus).get(key), bm, kind) if focus else PENDING])
    body += table(headers, rows)

    threshold = float(_dict(_dict(inputs.get("thresholds")).get("constants")).get("A10_confidence", DEFAULT_CONFIDENCE))
    curves = {t: _run(inputs, t).get("confidence_curve") or [] for t in [base, *cols]}
    body += ["", f"Confianza en {threshold:.2f} (dev gold, definición relativa de A10: la línea base dentro de los márgenes; "
                 "la precisión se calcula sobre filas OK que emiten predial):", ""]
    points = {t: _curve_point(curves[t], threshold) for t in curves}
    crow = []
    for key, label, kind in (("share_of_rows", f"proporción de filas gold con confianza >= {threshold:.2f}", "rate"),
                             ("predial_lot_precision_emitted", "precisión predial de lote (filas que emiten predial)", "rate"),
                             ("manzana_error_rate", "tasa de error manzana", "rate"),
                             ("n_emitting_predial", "filas OK que emiten predial", "count")):
        bval = _dict(points.get(base)).get(key)
        crow.append([label, fmt_rate(bval), *[fmt_rate(_dict(points.get(t)).get(key)) for t in cols],
                     fmt_delta(_dict(points.get(focus)).get(key), bval, kind) if focus else PENDING])
    body += table(["Magnitud", "Línea base", *[safe_label(t) for t in cols], "Diferencia"], crow)

    body += ["", "Precisión frente al umbral de confianza (línea base y corrida evaluada):", ""]
    brow = []
    ts = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    for t in ts:
        b, f = _dict(_curve_point(curves.get(base) or [], t)), _dict(_curve_point(curves.get(focus) or [], t)) if focus else {}
        brow.append([f"{t:.2f}", fmt_rate(b.get("share_of_rows")), fmt_rate(b.get("predial_lot_precision_emitted")),
                     fmt_rate(f.get("share_of_rows")), fmt_rate(f.get("predial_lot_precision_emitted"))])
    body += table(["Umbral", "Proporción de filas (línea base)", "Precisión de lote (línea base)",
                   "Proporción de filas (corrida)", "Precisión de lote (corrida)"], brow)
    if all(not curves[t] for t in curves):
        body += ["", f"Curvas de confianza: {PENDING} (`confidence_curve.json` no disponible en ninguna corrida)."]
    return _section(6, "Resultados antes y después (dev y congelada)", body)


# ---- 7 ---------------------------------------------------------------------------------------------
def _sec_by_source(inputs: dict) -> list[str]:
    tags = _compare_tags(inputs)
    rows = []
    for t in tags:
        for set_name, items in sorted(_dict(_run(inputs, t).get("by_fuente")).items()):
            for r in items:
                if "__header_only__" in r or "fuente" not in r:
                    continue
                rows.append([safe_label(t), safe_label(set_name), safe_label(r.get("scope")), safe_label(r.get("fuente")),
                             fmt_count(r.get("n_rows")), fmt_count(r.get("n_ok")), fmt_rate(r.get("manzana_precision")),
                             fmt_rate(r.get("predial_precision")), fmt_rate(r.get("predial_lot_precision")),
                             fmt_count(r.get("lost_correct"))])
    body = ["Aciertos separados por nivel de fuente del documento seleccionado (`nomenclatura_cert` certificado, "
            "`nomenclatura_pcer` pendiente de certificación; `(none)` = filas sin documento seleccionado):", ""]
    if rows:
        body += table(["Corrida", "Conjunto", "Alcance", "Fuente", "Filas", "OK", "Precisión manzana", "Precisión predial exacta",
                       "Precisión predial de lote", "lost_correct"], rows)
    else:
        body.append(f"{PENDING} (no hay `strict_eval_*_by_fuente.csv` para las corridas indicadas).")

    prow = []
    for t in tags:
        pa = _dict(_run(inputs, t).get("probe_agreement"))
        counts = _dict(pa.get("counts"))
        if counts:
            prow.append([safe_label(t), "todas", fmt_count(pa.get("n")), *[fmt_count(counts.get(b)) for b in _PROBE_BUCKETS]])
            for f, c in sorted(_dict(pa.get("by_fuente")).items()):
                prow.append([safe_label(t), safe_label(f), PENDING, *[fmt_count(_dict(c).get(b)) for b in _PROBE_BUCKETS]])
            lot = _dict(_dict(pa.get("lot_level")).get("counts"))
            if lot:
                prow.append([safe_label(t), "nivel de lote", fmt_count(_get(pa, "lot_level", "n")),
                             *[fmt_count(lot.get(b)) for b in _PROBE_BUCKETS]])
    body += ["", "Concordancia de filas entre la línea base (antigua) y la corrida (nueva) sobre filas gold de dev + congelada, "
                 "unidas por posición (solo conteos):", ""]
    if prow:
        body += table(["Corrida", "Corte", "Filas", *_PROBE_BUCKETS], prow)
    else:
        body.append(f"{PENDING} (no hay `probe_agreement.json`).")
    return _section(7, "Desglose por fuente (cert vs pcer)", body)


# ---- 8 ---------------------------------------------------------------------------------------------
def _sec_validation(inputs: dict) -> list[str]:
    tags = _dedupe([inputs.get("baseline_tag"), *_compare_tags(inputs)])
    rows, found = [], False
    for t in tags:
        for set_name, items in sorted(_dict(_run(inputs, t).get("validation")).items()):
            if not any("rule" in r or "__header_only__" in r for r in items):
                continue
            found = True
            if all("__header_only__" in r for r in items):
                rows.append([safe_label(t), safe_label(set_name), PENDING, "(sin violaciones)", "0", "0", PENDING])
                continue
            for r in items:
                if "rule" not in r:
                    continue
                rows.append([safe_label(t), safe_label(set_name), safe_label(r.get("mode")), safe_label(r.get("rule")),
                             fmt_count(r.get("violations")), fmt_count(r.get("corrected")), fmt_count(r.get("rows"))])
    body = ["Validación de salidas con pydantic (`validate_outputs`). En modo `report` las violaciones se cuentan y la salida "
            "no se modifica; en modo `correct` solo se aplican correcciones conservadoras (nunca más optimistas) y el conteo de "
            "violaciones restantes debería llegar a 0:", ""]
    if found:
        body += table(["Corrida", "Conjunto de evaluación", "Modo", "Regla", "Violaciones", "Corregidas", "Filas revisadas"], rows)
    else:
        body.append(f"{PENDING} (no hay `strict_eval_*_validation.csv` para las corridas indicadas; la evaluación con "
                    "`--validate-outputs report|correct` sigue pendiente).")
    return _section(8, "Validación de salidas (pydantic)", body)


# ---- 9 ---------------------------------------------------------------------------------------------
def _undefined(value) -> str:
    """A statistic the audit could not compute (no class variation, no scores) is undefined, not pending."""
    return "indefinido" if _num(value) is None else fmt_rate(value)


_JEV_CAVEAT = ("JEV es un juez basado en LLM, no una verdad de referencia: la concordancia no es exactitud, y cada "
               "discrepancia requiere revisión humana. Solo se auditan filas públicas de nomenclatura y consultas sintéticas "
               "derivadas del índice público; las direcciones ciudadanas (dev, congelada, pares de entrenamiento) nunca se envían.")


def _sec_jev(inputs: dict) -> list[str]:
    audits = [a for a in _list(inputs.get("jev")) if _dict(a.get("data"))]
    if not audits:
        return _section(9, "Auditoría JEV", [f"{PENDING}: no se encontró `jev_audit.json` en los directorios indicados (la "
                                             "auditoría requiere una clave JEV proporcionada por el usuario).", "", _JEV_CAVEAT])
    body = [_JEV_CAVEAT, ""]
    for a in audits:
        d = _dict(a["data"])
        counts, agree, conf = _dict(d.get("counts")), _dict(d.get("agreement")), _dict(d.get("confusion"))
        origins = _dict(_get(d, "privacy", "origins"))
        ci = agree.get("ci95")
        ci_text = f"{fmt_rate(ci[0])} a {fmt_rate(ci[1])}" if isinstance(ci, list) and len(ci) == 2 else PENDING
        body += [f"Auditoría `{a['label']}` (fuente {cell(safe_label(d.get('source')))}; orígenes "
                 f"{'; '.join(f'{safe_label(k)}={fmt_count(v)}' for k, v in sorted(origins.items())) or PENDING}; "
                 f"filas privadas permitidas: {fmt_bool(_get(d, 'privacy', 'allow_private'), 'sí', 'no')}):", ""]
        body += table(["Magnitud", "Valor"], [
            ["filas auditadas", fmt_count(counts.get("total"))], ["respondidas", fmt_count(counts.get("ok"))],
            ["errores", fmt_count(counts.get("error"))], ["omitidas", fmt_count(counts.get("skipped"))],
            ["veredictos indeterminados", fmt_count(counts.get("indeterminate"))],
            ["veredictos decisivos (etiquetados)", fmt_count(counts.get("labeled"))],
            ["tasa de concordancia", fmt_rate(agree.get("rate"))], ["IC 95 % de concordancia (Wilson)", ci_text],
            ["concordancia / n", f"{fmt_count(agree.get('agree'))} / {fmt_count(agree.get('n'))}"],
            ["kappa", _undefined(d.get("kappa"))], ["AUC", _undefined(d.get("auc"))], ["Brier", _undefined(d.get("brier"))],
        ])
        negatives = [_num(conf.get(k)) for k in ("tn", "fn")]
        positives = [_num(conf.get(k)) for k in ("tp", "fp")]
        one_class = (all(v == 0 for v in negatives if v is not None) and any(v is not None for v in negatives)
                     or all(v == 0 for v in positives if v is not None) and any(v is not None for v in positives))
        if _num(d.get("kappa")) is None or _num(d.get("auc")) is None:
            if one_class:
                body += ["", "No se devolvió ningún veredicto decisivo \"no\" (o ningún \"sí\" decisivo), así que los veredictos "
                             "etiquetados contienen una sola clase: kappa y AUC son indefinidos (n/d), no cero, y la tasa de "
                             "concordancia no puede separar a un buen juez de uno permisivo."]
            else:
                body += ["", "No se pudieron calcular kappa o AUC a partir de los veredictos devueltos; aquí son indefinidos (n/d)."]
        gates = _dict(_dict(d.get("gates")).get("misma_ubicacion"))
        if gates:
            body += ["", "Conteos de compuerta: " + "; ".join(f"{safe_label(k)}={fmt_count(v)}" for k, v in sorted(gates.items())) + "."]
        grows = []
        for gname, labels in sorted(_dict(d.get("groups")).items()):
            for lname, g in sorted(_dict(labels).items()):
                grows.append([safe_label(gname), safe_label(lname), fmt_count(_get(g, "counts", "total")),
                              fmt_count(_get(g, "agreement", "n")), fmt_rate(_get(g, "agreement", "rate"))])
        if grows:
            body += ["", *table(["Grupo", "Etiqueta", "Filas", "Etiquetadas", "Tasa de concordancia"], grows)]
        body.append("")
    return _section(9, "Auditoría JEV", body[:-1] if body and body[-1] == "" else body)


# ---- 10 --------------------------------------------------------------------------------------------
def _evidence(entry: dict) -> str:
    parts = [p for p in _list(entry.get("parts")) if isinstance(p, dict) and p.get("op") != "report" and "name" in p]
    if parts:
        pieces = []
        for p in parts[:6]:
            v, t = p.get("value"), p.get("threshold")
            vtxt = fmt_rate(v) if _num(v) is not None else (str(v) if isinstance(v, (bool, str)) else PENDING)
            ttxt = fmt_rate(t) if _num(t) is not None else (str(t) if isinstance(t, (bool, str)) else PENDING)
            pieces.append(f"{safe_label(p.get('name'))}={cell(vtxt, 40)} {cell(p.get('op', ''), 4)} {cell(ttxt, 40)}")
        return cell("; ".join(pieces), _MAX_EVIDENCE)
    v, t = entry.get("value"), entry.get("threshold")
    if isinstance(v, dict):
        return cell("; ".join(f"{safe_label(k)}={fmt_rate(x) if _num(x) is not None else cell(x, 30)}" for k, x in v.items()),
                    _MAX_EVIDENCE)
    if v is None:
        return PENDING
    return cell(f"{fmt_rate(v) if _num(v) is not None else cell(v, 40)} frente a "
                f"{fmt_rate(t) if _num(t) is not None else cell(t, 40)}", _MAX_EVIDENCE)


_STATUS_LABEL = {"pass": "APRUEBA", "fail": "NO APRUEBA", "not_evaluated": "NO EVALUADO"}


def _sec_checklist(inputs: dict) -> list[str]:
    results = normalize_results(_acceptance_results(inputs))
    criteria = _dict(_dict(inputs.get("thresholds")).get("criteria"))
    rows = []
    for cid in CRITERIA_IDS:
        entry = results.get(cid)
        text = criteria.get(cid) if isinstance(criteria.get(cid), str) else (entry or {}).get("description")
        text = text if isinstance(text, str) else PENDING
        if entry is None:
            rows.append([cid, "NO EVALUADO", text, "ausente de los resultados de aceptación"])
            continue
        label = _STATUS_LABEL[entry["_status"]]
        reason = cell(entry.get("reason") or "", _MAX_REASON)
        evidence = _evidence(entry)
        rows.append([cid, label, text, evidence + (f" -- {reason}" if reason and entry["_status"] != "pass" else "")])
    body = ["Los umbrales están escritos en `docs/retrain_2026-10_acceptance.md` antes de cualquier corrida. Un criterio que "
            "no pudo evaluarse bloquea un veredicto APRUEBA; nunca cuenta como aprobado.", ""]
    if _acceptance_results(inputs) is None:
        body += [f"Resultados de aceptación: {PENDING}.", ""]
    body += table(["ID", "Estado", "Criterio", "Evidencia"], rows)
    return _section(10, "Lista de criterios de aceptación (A1-A12)", body)


# ---- 11 --------------------------------------------------------------------------------------------
def _sec_risks(inputs: dict) -> list[str]:
    gate = _dict(_get(_first_index_meta(inputs), "oracle_ceiling", "gate"))
    noise = _dict(_dict(inputs.get("thresholds")).get("noise_band"))
    band = ", ".join(f"{safe_label(k)}={fmt_count(_dict(v).get('dev_lost_correct'))}" for k, v in sorted(noise.items())) or PENDING
    return _section(11, "Riesgos y limitaciones", [
        f"- Costo de cobertura: el índice de solo fuentes nuevas pierde direcciones catastrales sin registro de nomenclatura. "
        f"Medido en filas gold: pérdida titular {fmt_float(gate.get('coverage_loss_headline_pp'))} pp, de la cual "
        f"{fmt_float(gate.get('coverage_loss_inherent_pp'))} pp es inherente a la fuente (sección 4).",
        "- Sesgo de la verdad de referencia hacia el catastro: `gt_predial` es catastral. Un NPN certificado legítimamente "
        "distinto cuenta como error, así que la precisión predial es una cota inferior de la calidad real; la evaluación a "
        "nivel de lote y el diagnóstico \"correcto frente a cualquier fuente\" reducen, pero no eliminan, ese sesgo.",
        f"- Banda de ruido: dev gold tiene 289 filas, así que 1-2 filas (aproximadamente 0.3-0.7 puntos) son ruido. Los "
        f"reentrenamientos de referencia de la receta de producción dieron dev lost_correct {band}.",
        "- El recall de validación sintético y el oficial no son comparables entre índices; son diagnósticos, nunca selectores.",
        "- La prueba congelada se evalúa una sola vez, solo sobre la corrida seleccionada (`--final`); no se debe ajustar contra ella.",
        "- Los registros `pcer` están pendientes de certificación: se etiquetan `nomenclatura_pcer` y pueden eliminarse del "
        "índice sin tocar el resto si la sección 7 muestra que restan.",
        "- La concordancia JEV no es exactitud y cubre solo filas públicas o sintéticas (sección 9).",
        "- La promoción no forma parte de este informe: requiere la aprobación explícita de los mantenedores.",
    ])


# ---- 12 --------------------------------------------------------------------------------------------
def _sec_repro(inputs: dict) -> list[str]:
    base = safe_label(inputs.get("baseline_tag"))
    cands = [safe_label(t) for t in inputs.get("candidates") or []]
    ctrls = [safe_label(t) for t in inputs.get("controls") or []]
    final = safe_label(inputs["final_tag"]) if inputs.get("final_tag") else None
    cmd = ["python scripts/write_retrain_report.py --out docs/retrain_2026-10_nomenclatura.md \\",
           f"  --baseline-tag {base} \\"]
    if cands:
        cmd.append("  --candidates " + " ".join(cands) + " \\")
    if ctrls:
        cmd.append("  --control " + " ".join(ctrls) + " \\")
    if final:
        cmd.append(f"  --final-tag {final} \\")
    cmd.append("  --jev-dir artifacts/jev_audit artifacts/jev_audit_synthetic")
    body = ["Ejecutar desde `models/normalizador_direcciones` con el `.venv` (CPU); el entrenamiento usa `.venv-train` (CUDA). "
            "Pasos en orden:",
            "", "```bash",
            ".venv/Scripts/python scripts/reproject_nomenclatura.py --shp train_data/nomenclatura_certificada_20261030/nomenclatura_certificada_20261030.shp",
            ".venv/Scripts/python scripts/download_nomenclatura.py --out artifacts/nomenclatura_wfs.parquet",
            ".venv/Scripts/python scripts/clean_nomenclatura.py --in artifacts/nomenclatura_wfs.parquet --out artifacts/nomenclatura_clean.parquet",
            "# Índices (r3): base cert+pcer, solo cert y con respaldo de catastro (el respaldo se agrega sobre el base, que no se modifica)",
            ".venv/Scripts/python scripts/build_index.py --tiers cert,pcer --out-dir artifacts/index_v2e_cert_pcer_r3   # ver docs/model-artifacts.md",
            ".venv/Scripts/python scripts/build_index.py --tiers cert --out-dir artifacts/index_v2f_cert_r3",
            ".venv/Scripts/python scripts/index_fallback.py add --base-index artifacts/index_v2e_cert_pcer_r3 --splits-dir artifacts/splits --out-dir artifacts/index_v2g_cert_pcer_catfb_r3",
            ".venv/Scripts/python scripts/compare_indexes.py --old artifacts --new artifacts/index_v2g_cert_pcer_catfb_r3",
            "# Modelo semilla 42: ya entrenado sobre el índice r2 anterior (mismo texto y orden de documentos que el r3); se reutiliza sin reentrenar:",
            "# .venv-train/Scripts/python scripts/run_experiment.py --tag v2d_cert_pcer_s42 --index-dir artifacts/index_v2d_cert_pcer_r2 --set max_minutes=120",
            ".venv/Scripts/python scripts/make_inherit_experiment.py --index-dir artifacts/index_v2e_cert_pcer_r3 --model-dir artifacts/experiments/v2d_cert_pcer_s42 --out artifacts/experiments/v2e_cert_pcer_s42_r3",
            "# Semilla 43 (banda de ruido): entrenamiento completo sobre el índice r3",
            ".venv-train/Scripts/python scripts/run_experiment.py --tag v2e_cert_pcer_s43 --index-dir artifacts/index_v2e_cert_pcer_r3 --set seed=43 --set max_minutes=120",
            "# Calibración de la capa de decisión (CPU): se ajusta con los pares de train_pairs y se instala en un directorio nuevo",
            ".venv/Scripts/python scripts/fit_decision.py --artifacts-dir artifacts/experiments/v2e_cert_pcer_s42_r3 --pairs artifacts/splits/train_pairs.parquet --out-dir artifacts/experiments/calib_design/params --tag cand --device cpu",
            ".venv/Scripts/python scripts/make_decision_experiment.py --source artifacts/experiments/v2e_cert_pcer_s42_r3 --out artifacts/experiments/v2e_cert_pcer_s42_r3_cal --decision artifacts/experiments/calib_design/params/decision_cand.json --reliability artifacts/experiments/calib_design/params/reliability_calibrated_cand.json",
            ".venv/Scripts/python scripts/fit_decision.py --artifacts-dir artifacts/experiments/v2e_cert_pcer_s43 --pairs artifacts/splits/train_pairs.parquet --out-dir artifacts/experiments/calib_s43/params --tag cand43 --device cpu",
            ".venv/Scripts/python scripts/make_decision_experiment.py --source artifacts/experiments/v2e_cert_pcer_s43 --out artifacts/experiments/v2e_cert_pcer_s43_cal --decision artifacts/experiments/calib_s43/params/decision_cand43.json --reliability artifacts/experiments/calib_s43/params/reliability_calibrated_cand43.json",
            "# Candidato final: modelo del r3 con los documentos de respaldo embebidos y la calibración del candidato (sin reentrenar)",
            ".venv/Scripts/python scripts/index_fallback.py experiment --index-dir artifacts/index_v2g_cert_pcer_catfb_r3 --model-dir artifacts/experiments/v2e_cert_pcer_s42_r3 --out-dir artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 --device cpu",
            ".venv/Scripts/python scripts/make_decision_experiment.py --source artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 --out artifacts/experiments/v2g_cert_pcer_catfb_s42_r3_basecal --decision artifacts/experiments/calib_design/params/decision_cand.json --reliability artifacts/experiments/calib_design/params/reliability_calibrated_cand.json",
            f".venv/Scripts/python scripts/run_experiment.py --tag {final or '<tag>'} --eval-only --from-dir artifacts/experiments/v2g_cert_pcer_catfb_s42_r3_basecal --eval-set both --final",
            f".venv/Scripts/python scripts/run_experiment.py --tag {base} --eval-only --from-dir artifacts --eval-set both --final",
            f".venv/Scripts/python scripts/compare_runs.py --old artifacts/experiments/{base} --new artifacts/experiments/{final or '<tag>'}",
            ".venv/Scripts/python scripts/audit_with_jev.py --source nomenclatura --sample 300 --env-file <ruta al .env>   # requiere una clave JEV; agregar --dry-run primero",
            *cmd, "```"]
    return _section(12, "Comandos de reproducción", body)


# =====================================================================================================
# final-report sections 13-21 (read the experiment summary; aggregates only, text always written here)
# =====================================================================================================
SECOND_LOOK_WARNING = (
    "ADVERTENCIA: la evaluación sobre la prueba congelada de la ronda final es una SEGUNDA MIRADA y no es ciega. La decisión "
    "de agregar el respaldo de catastro se tomó después de ver el resultado congelado de la ronda 1 (la brecha de cobertura). "
    "No se ajustó ningún parámetro con la congelada y se puntuó una sola vez, pero esa puntuación no es una prueba independiente.")
_SET_ORDER = ("dev_gold", "dev_all", "frozen_gold", "frozen_all")
_SET_LABEL = {"dev_gold": "dev_gold", "dev_all": "dev_all", "frozen_gold": "frozen_gold (segunda mirada)",
              "frozen_all": "frozen_all (segunda mirada)"}
_BAND_ORDER = ("alta", "media", "baja", "sin_dato")
_COUNT_METRICS = frozenset({"gold_ok", "lost_correct", "ok"})
_BOOT_LABEL = {"gold_ok": "filas gold OK", "gold_lot_precision": "precisión de lote gold",
               "all_manzana_precision": "precisión manzana (todas las filas)", "lost_correct": "lost_correct"}
_METRIC_RE = re.compile(r"^[A-Za-z0-9_@.\-]{1,60}$")


def _metric_label(value) -> str:
    return value if isinstance(value, str) and _METRIC_RE.match(value) else OTHER_LABEL


def _ci_text(p, lo, hi) -> str:
    """``0.9955 [0.9751 a 0.9992]``; a rate that could not be computed (no scored rows) says so."""
    pn, ln, hn = _num(p), _num(lo), _num(hi)
    if pn is None:
        return "n/d (sin filas evaluadas)"
    if ln is None or hn is None:
        return fmt_rate(pn)
    return f"{pn:.4f} [{ln:.4f} a {hn:.4f}]"


def _ratio(a, b) -> str:
    return f"{fmt_count(a)} / {fmt_count(b)}"


def _fmt_any(value, nested: bool = False) -> str:
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, str):
        return cell(value, 40)
    if isinstance(value, dict):
        return cell("; ".join(f"{_metric_label(k)}={_fmt_any(v, True)}" for k, v in value.items()), _MAX_EVIDENCE)
    number = _num(value)
    if number is None:
        return "no evaluado" if nested and value is None else PENDING
    return fmt_rate(number)


def _ordered(mapping: dict, order: tuple[str, ...]) -> list[str]:
    keys = [k for k in mapping if isinstance(k, str)]
    return [k for k in order if k in keys] + sorted(k for k in keys if k not in order)


def _pending_line(what: str, source: str) -> str:
    return f"{what}: {PENDING} (no se encontró o no se pudo leer `{source}`)."


def _selected_candidate(summary: dict) -> tuple[str | None, dict]:
    sel = _dict(summary.get("selection"))
    name = sel.get("selected") if isinstance(sel.get("selected"), str) else None
    return name, _dict(_dict(sel.get("candidates")).get(name)) if name else {}


# ---- 13 --------------------------------------------------------------------------------------------
_ROUND_METRICS = (("n", "filas", "count"), ("ok", "filas OK", "count"),
                  ("rows_emitting_predial", "filas que emiten predial", "count"), ("lot_correct", "lotes correctos", "count"),
                  ("lot_precision", "precisión de lote (filas que emiten predial)", "rate"),
                  ("manzana_precision", "precisión manzana", "rate"), ("lost_correct", "lost_correct", "count"),
                  ("lot_missing_from_index", "lote ausente del índice", "count"))


def _sec_story(inputs: dict) -> list[str]:
    summary = _dict(inputs.get("summary"))
    rc = _dict(summary.get("round_comparison"))
    fb = _dict(_get(summary, "index", "fallback"))
    stats = _dict(fb.get("stats"))
    selected, _ = _selected_candidate(summary)
    sel = _dict(summary.get("selection"))
    round1 = safe_label(summary.get("round1")) if isinstance(summary.get("round1"), str) else PENDING
    body = [SECOND_LOOK_WARNING, ""]
    if not summary:
        body += [_pending_line("Historia de la ronda final", "summary_for_report.json"), ""]
    g1, g2 = _dict(_get(rc, "frozen_gold", "round1")), _dict(_get(rc, "frozen_gold", "round2"))
    body += [
        f"**Etapa 1: ronda 1, sin respaldo de catastro (corrida `{round1}`).** El índice se construyó solo con registros de "
        f"nomenclatura (cert y pcer). En la prueba congelada gold el resultado fue {fmt_count(g1.get('ok'))} filas OK de "
        f"{fmt_count(g1.get('n'))}, con {fmt_count(g1.get('lot_missing_from_index'))} filas cuyo lote no está en el índice: esa "
        "es la brecha de cobertura que motivó la etapa siguiente.", "",
        "**Etapa 2: decisión de agregar el respaldo de catastro.** Se agrega, como último recurso y de forma reversible "
        f"(sección 21), {fmt_count(fb.get('n_docs_added'))} documentos de catastro a partir de "
        f"{fmt_count(stats.get('candidates'))} candidatos (colisiones omitidas: {fmt_count(fb.get('n_collisions_skipped'))}); el "
        f"índice resultante tiene {fmt_count(_get(summary, 'index', 'n_docs'))} documentos. El modelo no se reentrena: se reutiliza "
        "el de la ronda 1 y los documentos agregados se embeben con él.", "",
        "**Etapa 3: ronda 2, con respaldo.** Se evaluaron dos variantes de calibración sobre el índice con respaldo (`basecal` "
        "conserva la calibración de la ronda 1; `cal` la reajusta) y se eligió con las métricas dev gold, sin usar la prueba "
        "congelada.", ""]
    cands = _dict(sel.get("candidates"))
    floor = _dict(sel.get("a5_floor_check"))
    rows = []
    for name in _ordered(cands, ("basecal", "cal")):
        c = _dict(cands[name])
        rows.append([safe_label(name) + (" (SELECCIONADA)" if name == selected else ""), fmt_count(c.get("gold_ok")),
                     _ratio(c.get("gold_lot_correct"), c.get("gold_lot_n")), fmt_rate(c.get("gold_lot_precision")),
                     fmt_rate(c.get("all_manzana_precision")), fmt_count(c.get("lost_correct")),
                     fmt_bool(floor.get(name), "cumple", "no cumple")])
    if rows:
        body += table(["Variante", "dev gold OK", "lotes correctos / emitidos", "precisión de lote dev gold",
                       "precisión manzana dev (todas)", "dev lost_correct", f"piso de A5 ({fmt_rate(sel.get('a5_floor_all_dev_manzana_precision'))})"], rows)
        tie = [safe_label(t) for t in _list(sel.get("tie")) if isinstance(t, str)]
        below = [safe_label(t) for t in tie if t != selected and floor.get(t) is False]
        body += ["", f"Selección: `{safe_label(selected)}`. " + (
            "Empate en precisión de lote dentro de una fila gold entre " + " y ".join(f"`{t}`" for t in tie) +
            "; se desempató por la precisión manzana sobre todas las filas dev" +
            (f", y {', '.join(f'`{t}`' for t in below)} además queda bajo el piso de A5." if below else ".")
            if len(tie) > 1 else "")]
    else:
        body.append(_pending_line("Variantes de la ronda 2", "summary_for_report.json (selection)"))

    for key, title in (("frozen_gold", "filas gold"), ("frozen_all", "todas las filas")):
        part = _dict(rc.get(key))
        body += ["", f"Prueba congelada, {title}, ronda 2 menos ronda 1 sobre las MISMAS filas (SEGUNDA MIRADA):", ""]
        if not part:
            body.append(_pending_line("Comparación de rondas", "summary_for_report.json (round_comparison)"))
            continue
        r1, r2 = _dict(part.get("round1")), _dict(part.get("round2"))
        body += table(["Métrica", "Ronda 1", "Ronda 2", "Diferencia"],
                      [[label, fmt_rate(r1.get(k)), fmt_rate(r2.get(k)), fmt_delta(r2.get(k), r1.get(k), kind)]
                       for k, label, kind in _ROUND_METRICS])
        tr, lt = _dict(part.get("transitions")), _dict(part.get("lot_correct_transitions"))
        body += ["", f"Transiciones: filas OK ganadas {fmt_count(tr.get('ok_gained'))}, perdidas {fmt_count(tr.get('ok_lost'))}; "
                     f"lotes correctos ganados {fmt_count(lt.get('gained'))}, perdidos {fmt_count(lt.get('lost'))}, "
                     f"conservados {fmt_count(lt.get('both'))}."]
    return _section(13, "Historia en tres etapas y segunda mirada a la prueba congelada", body)


# ---- 14 --------------------------------------------------------------------------------------------
_SEED_ROWS = (("gold_ok", "dev gold OK", "count"), ("gold_ok_coverage", "dev gold proporción de OK respondidas", "rate"),
              ("gold_lot_n", "dev gold filas que emiten predial", "count"),
              ("gold_lot_precision", "dev gold precisión de lote", "rate"),
              ("gold_manzana_precision", "dev gold precisión manzana", "rate"), ("all_ok", "dev OK (todas)", "count"),
              ("all_lot_precision", "dev precisión de lote (todas)", "rate"),
              ("all_manzana_precision", "dev precisión manzana (todas)", "rate"),
              ("lost_correct", "dev lost_correct", "count"), ("tau_p", "umbral tau_p", "rate"))


def _abs_diff(a, b, kind: str) -> str:
    x, y = _num(a), _num(b)
    if x is None or y is None:
        return PENDING
    return fmt_count(abs(x - y)) if kind == "count" else f"{abs(x - y):.4f}"


def _sec_noise(inputs: dict) -> list[str]:
    summary = _dict(inputs.get("summary"))
    sc = _dict(inputs.get("seed_comparison"))
    runs = _dict(sc.get("runs"))
    pair = [k for k in _list(sc.get("pair")) if isinstance(k, str) and k in runs] or sorted(runs)[:2]
    one_row = _num(sc.get("one_gold_row_pp")) if _num(sc.get("one_gold_row_pp")) is not None else \
        _num(_get(summary, "seed_spread", "one_gold_row_pp_dev"))
    body = ["La banda de ruido se mide con dos semillas del mismo procedimiento sobre el índice de la ronda 1: la diferencia "
            "entre ellas es lo que se mueve por azar de entrenamiento y no por un cambio real. "
            f"Una fila gold de dev equivale a {fmt_float(one_row, 3)} puntos porcentuales ({fmt_count(sc.get('gold_n'))} filas).", ""]
    if len(pair) == 2:
        a, b = _dict(runs[pair[0]]), _dict(runs[pair[1]])
        rows = [[label, fmt_rate(a.get(k)), fmt_rate(b.get(k)), _abs_diff(a.get(k), b.get(k), kind)] for k, label, kind in _SEED_ROWS]
        for head in ("predio", "manzana"):
            ca, cb = _dict(_dict(a.get("calibration")).get(head)), _dict(_dict(b.get("calibration")).get(head))
            for k, label in (("ece10", "ECE fuera de muestra"), ("brier", "Brier fuera de muestra"), ("auc", "AUC fuera de muestra")):
                rows.append([f"{label} ({head})", fmt_rate(ca.get(k)), fmt_rate(cb.get(k)), _abs_diff(ca.get(k), cb.get(k), "rate")])
        body += table(["Métrica", f"Semilla {safe_label(pair[0])}", f"Semilla {safe_label(pair[1])}", "Diferencia absoluta"], rows)

        name, cand = _selected_candidate(summary)
        body += ["", "Lectura contra la corrida seleccionada (las semillas se midieron sobre el índice sin respaldo, así que la "
                     "comparación sirve como orden de magnitud y no como igualdad):", ""]
        for key, label, kind in (("lost_correct", "dev lost_correct", "count"), ("gold_lot_precision", "dev gold precisión de lote", "rate"),
                                 ("gold_ok", "dev gold OK", "count")):
            c, va, vb = _num(cand.get(key)), _num(a.get(key)), _num(b.get(key))
            if c is None or va is None or vb is None:
                body.append(f"- {label}: {PENDING}.")
                continue
            inside = round(min(va, vb), 4) <= round(c, 4) <= round(max(va, vb), 4)
            body.append(f"- {label}: seleccionada {fmt_rate(c)}; semillas {fmt_rate(va)} y {fmt_rate(vb)}: "
                        f"{'dentro' if inside else 'fuera'} del rango entre semillas.")
    else:
        body.append(_pending_line("Comparación entre semillas", "seed_comparison.json"))

    boot = _dict(summary.get("bootstrap"))
    body += ["", "Ruido de muestreo: diferencias pareadas contra la línea base con bootstrap sobre las filas "
                 f"({fmt_count(boot.get('resamples'))} remuestreos, semilla {fmt_count(boot.get('seed'))}; diferencia = candidato - "
                 "línea base; intervalo percentil de 95 %). Si el intervalo contiene el cero, la diferencia no se distingue del "
                 "ruido de muestreo:", ""]
    brow = []
    for scope, label in (("dev", "dev"), ("frozen", "congelada (segunda mirada)")):
        for metric in sorted(_dict(boot.get(scope))):
            m = _dict(_dict(boot[scope])[metric])
            kind = "count" if metric in _COUNT_METRICS else "rate"
            fmt = fmt_count if kind == "count" else fmt_rate
            brow.append([label, _BOOT_LABEL.get(metric, _metric_label(metric)), fmt(m.get("old")), fmt(m.get("new")),
                         fmt_delta(m.get("new"), m.get("old"), kind), f"[{fmt(m.get('lo'))} a {fmt(m.get('hi'))}]",
                         fmt_rate(m.get("p_diff_lt0"))])
    if brow:
        body += table(["Conjunto", "Métrica", "Línea base", "Candidato", "Diferencia", "IC 95 %", "P(diferencia < 0)"], brow)
    else:
        body.append(_pending_line("Bootstrap", "summary_for_report.json (bootstrap)"))
    return _section(14, "Banda de ruido entre semillas y ruido de muestreo", body)


# ---- 15 --------------------------------------------------------------------------------------------
def _wilson_table(groups: dict, key: str, header: str) -> list[str]:
    rows = []
    for set_name in _ordered(groups, _SET_ORDER):
        for item in _list(groups[set_name]):
            if not isinstance(item, dict) or key not in item:
                continue
            rows.append([_SET_LABEL.get(set_name, safe_label(set_name)), safe_label(item.get(key)), fmt_count(item.get("n")),
                         _ratio(item.get("lot_correct"), item.get("lot_scored")),
                         _ci_text(item.get("lot_precision"), item.get("lot_lo"), item.get("lot_hi")),
                         _ratio(item.get("manzana_correct"), item.get("manzana_scored")),
                         _ci_text(item.get("manzana_precision"), item.get("manzana_lo"), item.get("manzana_hi"))])
    if not rows:
        return [f"{PENDING} (sin datos para esta tabla)."]
    return table(["Conjunto", header, "Filas", "Lotes correctos / evaluados", "Precisión de lote [IC 95 % Wilson]",
                  "Manzanas correctas / evaluadas", "Precisión manzana [IC 95 % Wilson]"], rows)


def _sec_bands(inputs: dict) -> list[str]:
    summary = _dict(inputs.get("summary"))
    body = ["Los intervalos son de Wilson al 95 % (un intervalo ancho indica pocas filas; una banda sin filas evaluadas no tiene "
            "precisión definida). Las filas congeladas son una SEGUNDA MIRADA, no una prueba ciega (ver la advertencia de la sección 13).", "",
            "Por banda de confianza (`alta` >= 0.97, `media` >= 0.94, `baja` por debajo; `sin_dato` sin banda):", ""]
    body += _wilson_table(_dict(summary.get("bands")), "band", "Banda")
    body += ["", "Por nivel de fuente del documento seleccionado (candidato; la línea base no tiene niveles de fuente):", ""]
    body += _wilson_table(_dict(_dict(summary.get("by_fuente")).get("candidate")), "fuente", "Fuente")
    return _section(15, "Precisión por banda de confianza y por fuente (intervalos de Wilson)", body)


# ---- 16 --------------------------------------------------------------------------------------------
def _sec_calibration(inputs: dict) -> list[str]:
    summary = _dict(inputs.get("summary"))
    sc = _dict(inputs.get("seed_comparison"))
    runs = _dict(sc.get("runs"))
    first = _dict(runs.get(next((k for k in _list(sc.get("pair")) if isinstance(k, str) and k in runs), None))) or \
        (_dict(runs[sorted(runs)[0]]) if runs else {})
    decision = _dict(inputs.get("decision"))
    a6 = _dict(_get(summary, "acceptance", "a6_oof_ece"))
    bands = _dict(decision.get("bands"))
    body = ["La calibración se ajusta sobre pares reales de entrenamiento nunca usados para entrenar el modelo y se mide fuera de "
            "muestra (validación cruzada de 5 grupos por dirección). El candidato seleccionado conserva la calibración de la ronda 1.",
            "", f"Parámetros de decisión: tau_p {fmt_float(decision.get('tau_p'))}, tau_m {fmt_float(decision.get('tau_m'))}, piso de "
                f"confianza {fmt_float(decision.get('confianza_floor', decision.get('floor')))}, bandas: alta >= {fmt_float(bands.get('alta'))}, "
                f"media >= {fmt_float(bands.get('media'))}.", ""]
    crow = []
    for head in ("predio", "manzana"):
        c = _dict(_dict(first.get("calibration")).get(head))
        crow.append([head, fmt_count(c.get("n")), fmt_rate(c.get("ece10")), fmt_rate(c.get("brier")), fmt_rate(c.get("auc")),
                     fmt_rate(c.get("refit_brier")), fmt_rate(c.get("refit_auc")), fmt_rate(a6.get(f"baseline_refit_limit_{head}"))])
    body += ["Calidad fuera de muestra (OOF) de las dos cabezas:", ""]
    body += table(["Cabeza", "Pares", "ECE (10 bins)", "Brier", "AUC", "Brier del refit", "AUC del refit",
                   "Límite de ECE de A6 (refit de la línea base)"], crow) if runs or a6 else \
        [_pending_line("Calibración OOF", "seed_comparison.json")]

    diagrams = _dict(summary.get("reliability_diagrams"))
    body += ["", "Diagramas de confiabilidad como tabla (probabilidad predicha frente a frecuencia observada por bin de igual "
                 "tamaño; sin imágenes):"]
    if not diagrams:
        body += ["", _pending_line("Diagramas de confiabilidad", "summary_for_report.json (reliability_diagrams)")]
    for set_name in _ordered(diagrams, _SET_ORDER):
        heads = _dict(diagrams[set_name])
        pred, man = _dict(heads.get("predio")), _dict(heads.get("manzana"))
        by_bin = {}
        for head, data in (("predio", pred), ("manzana", man)):
            for b in _list(data.get("bins")):
                if isinstance(b, dict) and _num(b.get("bin")) is not None:
                    by_bin.setdefault(int(_num(b["bin"])), {})[head] = b
        body += ["", f"{_SET_LABEL.get(set_name, safe_label(set_name))}: predio n {fmt_count(pred.get('n'))}, predicho medio "
                     f"{fmt_rate(pred.get('mean_predicted'))}, observado {fmt_rate(pred.get('observed'))}; manzana n "
                     f"{fmt_count(man.get('n'))}, predicho medio {fmt_rate(man.get('mean_predicted'))}, observado "
                     f"{fmt_rate(man.get('observed'))}.", ""]
        rows = [[str(i), fmt_count(_dict(by_bin[i].get("predio")).get("n")), fmt_rate(_dict(by_bin[i].get("predio")).get("predicted")),
                 fmt_rate(_dict(by_bin[i].get("predio")).get("observed")), fmt_count(_dict(by_bin[i].get("manzana")).get("n")),
                 fmt_rate(_dict(by_bin[i].get("manzana")).get("predicted")), fmt_rate(_dict(by_bin[i].get("manzana")).get("observed"))]
                for i in sorted(by_bin)]
        if rows:
            body += table(["Bin", "Predio n", "Predio predicho", "Predio observado", "Manzana n", "Manzana predicha",
                           "Manzana observada"], rows)
        else:
            body.append(f"{PENDING} (sin bins).")
    return _section(16, "Calidad de calibración", body)


# ---- 17 --------------------------------------------------------------------------------------------
_JEV_ORDER = ("synthetic_candidate", "synthetic_baseline", "catfbdocs_candidate", "catfbdocs_baseline")


def _sec_jev_final(inputs: dict) -> list[str]:
    jev = _dict(_dict(inputs.get("summary")).get("jev"))
    body = ["Límites: JEV es un juez basado en LLM de equivalencia textual entre la consulta y la dirección normalizada; no es una "
            "verdad de referencia ni un oráculo de corrección de lote. Casi no devuelve veredictos decisivos \"no\", así que la "
            "concordancia es trivial (una sola clase: kappa 0 o indefinido, AUC sin significado con muy pocos negativos) y no "
            "distingue a un buen juez de uno permisivo; los veredictos indeterminados son una fracción grande. Solo se auditan "
            "filas públicas de nomenclatura/catastro y consultas sintéticas; las direcciones ciudadanas nunca se envían. "
            "Cada discrepancia requiere revisión humana.", ""]
    audits = [k for k in _ordered(jev, _JEV_ORDER) if k not in ("limits", "crosscheck") and isinstance(jev.get(k), dict)]
    rows = []
    for k in audits:
        d = _dict(jev[k])
        c, ag = _dict(d.get("counts")), _dict(d.get("agreement"))
        ci = ag.get("ci95")
        ci_text = f"{fmt_rate(ci[0])} a {fmt_rate(ci[1])}" if isinstance(ci, list) and len(ci) == 2 else PENDING
        rows.append([safe_label(k), fmt_count(c.get("total")), fmt_count(c.get("ok")), fmt_count(c.get("skipped")),
                     fmt_count(c.get("indeterminate")), fmt_count(c.get("labeled")), fmt_rate(ag.get("rate")), ci_text,
                     _undefined(d.get("kappa")), _undefined(d.get("auc")), _undefined(d.get("brier"))])
    if rows:
        body += table(["Auditoría", "Filas", "Respondidas", "Omitidas", "Indeterminadas", "Etiquetadas", "Concordancia",
                       "IC 95 % (Wilson)", "kappa", "AUC", "Brier"], rows)
    else:
        body.append(_pending_line("Auditorías JEV", "summary_for_report.json (jev)"))

    cross = _dict(jev.get("crosscheck"))
    body += ["", "Contraste por banda de confianza y por fuente (qué tan seguido JEV dice \"no\" donde el modelo es menos seguro):", ""]
    crows, concentrated = [], []
    for k in _ordered(cross, _JEV_ORDER):
        for dim, order in (("by_band", _BAND_ORDER), ("by_fuente", ())):
            groups = _dict(_dict(cross[k]).get(dim))
            for g in _ordered(groups, order):
                d = _dict(groups[g])
                no = d.get("no", d.get("decisive_no"))
                crows.append([safe_label(k), "banda" if dim == "by_band" else "fuente", safe_label(g), fmt_count(d.get("judged")),
                              fmt_count(d.get("yes")), fmt_count(no), fmt_count(d.get("indeterminate")),
                              fmt_rate(d.get("mean_jev_p")), fmt_rate(d.get("decisive_no_rate"))])
                if dim == "by_band" and (_num(no) or 0) > 0:
                    concentrated.append(f"`{safe_label(k)}`/{safe_label(g)} ({fmt_count(no)})")
    if crows:
        body += table(["Auditoría", "Corte", "Grupo", "Juzgadas", "Sí", "No decisivos", "Indeterminadas", "p medio de JEV",
                       "Tasa de no decisivo"], crows)
        body += ["", "Los \"no\" decisivos por banda aparecen en: " + (", ".join(concentrated) or "ninguna banda") + "."]
    else:
        body.append(_pending_line("Contraste por banda", "summary_for_report.json (jev.crosscheck)"))
    body += ["", "La sección 9 detalla cada auditoría por directorio cuando se indicaron con `--jev-dir`."]
    return _section(17, "Auditoría JEV: límites y contraste por banda", body)


# ---- 18 --------------------------------------------------------------------------------------------
_SET_ES = {"prod": "producción", "cand": "candidato"}


def _bench_table(bench: dict) -> list[str]:
    rows = []
    for s in _ordered(_dict(bench.get("runs")), ("prod", "cand")):
        for backend in _ordered(_dict(bench["runs"][s]), ("torch", "onnx", "onnx-int8")):
            r = _dict(_dict(bench["runs"][s])[backend])
            lat, thr = _dict(r.get("latency_ms")), _dict(r.get("throughput"))
            rows.append([_SET_ES.get(s, safe_label(s)), safe_label(backend), fmt_count(r.get("n_docs")),
                         fmt_float(r.get("cold_start_s")), fmt_float(lat.get("p50"), 1), fmt_float(lat.get("p95"), 1),
                         *[fmt_float(_dict(thr.get(b)).get("rows_per_s_median"), 0) for b in ("100", "2000", "10000")],
                         fmt_float(r.get("peak_rss_mb"), 0)])
    return table(["Conjunto", "Backend", "Documentos", "Arranque en frío (s)", "Latencia p50 (ms)", "Latencia p95 (ms)",
                  "Filas/s (lote 100)", "Filas/s (lote 2000)", "Filas/s (lote 10000)", "RSS pico (MB)"], rows)


def _sec_perf(inputs: dict) -> list[str]:
    onnx = _dict(inputs.get("onnx"))
    if not onnx:
        return _section(18, "Rendimiento: ONNX y benchmark", [
            f"{PENDING} (no hay `bench.json` en `onnx_bench_final` ni en `onnx_bench`)."])
    bench = _dict(onnx.get("bench"))
    body = [(f"Resultados PRELIMINARES: se leyeron de `{onnx.get('dir')}` porque `onnx_bench_final` todavía no tiene resultados; "
             "deben repetirse con la configuración final.") if onnx.get("preliminary") else
            f"Fuente: `{onnx.get('dir')}` (corrida final del benchmark).", "",
            f"Benchmark con {fmt_count(_get(bench, 'config', 'threads'))} hilos (consultas públicas corrompidas; mediana de "
            f"{fmt_count(_get(bench, 'config', 'repeats'))} repeticiones):", ""]
    body += _bench_table(bench)
    b4 = _dict(onnx.get("bench_threads4"))
    if _dict(b4.get("runs")):
        body += ["", f"Benchmark con {fmt_count(_get(b4, 'config', 'threads'))} hilos:", ""]
        body += _bench_table(b4)
    b16 = _dict(onnx.get("bench_fp16"))
    if _dict(b16.get("runs")):
        body += ["", f"Benchmark del candidato con índice fp16 ({fmt_count(_get(b16, 'config', 'threads'))} hilos):", ""]
        body += _bench_table(b16)
    prow = []
    for s in ("prod", "cand"):
        par = _dict(_dict(onnx.get("parity")).get(s))
        for backend, entries in sorted(_dict(par.get("backends")).items()):
            for e in _list(entries):
                if isinstance(e, dict) and e.get("set") == "public_docs":
                    prow.append([_SET_ES[s], safe_label(backend), fmt_count(e.get("n")), fmt_rate(e.get("cosine_min"), 6),
                                 fmt_rate(e.get("top1_identical")), fmt_count(e.get("top1_differ")),
                                 fmt_float(e.get("xy_max_abs_diff_m"), 1)])
    body += ["", "Paridad con torch (documentos públicos): el top-1 que difiere y la deriva máxima de coordenadas en metros:", ""]
    body += table(["Conjunto", "Backend", "Consultas", "Coseno mínimo", "Top-1 idéntico", "Top-1 distinto", "Deriva máxima de coordenadas (m)"],
                  prow) if prow else [f"{PENDING} (sin `parity_*_serving.json`)."]
    drows = []
    for name, d in sorted(_dict(onnx.get("dev_metrics")).items()):
        d = _dict(d)
        if "/" in str(name) and str(name).endswith("dev_gold"):
            backend = safe_label(str(name).split("/")[0])
            ref = "(referencia)" if backend == "torch" else PENDING
            drows.append([backend, fmt_count(d.get("n")), fmt_count(d.get("ok")),
                          fmt_count(d["estado_equal_to_torch"]) if "estado_equal_to_torch" in d else ref,
                          fmt_count(d["npn_equal_to_torch"]) if "npn_equal_to_torch" in d else ref])
    note = (f" (fuente: `{onnx.get('dev_metrics_dir')}`)" if onnx.get("dev_metrics_dir") not in (None, onnx.get("dir")) else "")
    body += ["", f"Efecto sobre las métricas dev gold{note} (filas con el mismo estado y el mismo NPN que torch):", ""]
    body += table(["Backend", "Filas", "OK", "Mismo estado que torch", "Mismo NPN que torch"], drows) if drows else \
        [f"{PENDING} (sin `dev_metrics.json`)."]
    return _section(18, "Rendimiento: ONNX y benchmark", body)


# ---- 19 --------------------------------------------------------------------------------------------
def _reason_es(entry: dict) -> str:
    if entry.get("_status") == "pass":
        return ""
    texts = []
    for p in _list(entry.get("parts")):
        if not isinstance(p, dict) or p.get("status") != "fail":
            continue
        name, op = _metric_label(p.get("name")), p.get("op")
        if op == "is_true":
            texts.append(f"{name} debe ser sí")
        elif op in (">=", "<=", "<", ">", "=="):
            texts.append(f"{name}={_fmt_any(p.get('value'))} no cumple {op} {_fmt_any(p.get('threshold'))}")
        else:
            texts.append(f"{name} no cumple su umbral")
    return cell("; ".join(texts) or entry.get("reason") or "", _MAX_REASON)


def _boot(summary: dict, scope: str, metric: str) -> dict:
    return _dict(_dict(_dict(summary.get("bootstrap")).get(scope)).get(metric))


def _contains_zero(m: dict) -> bool | None:
    lo, hi = _num(m.get("lo")), _num(m.get("hi"))
    return None if lo is None or hi is None else lo <= 0 <= hi


def _reading(cid: str, entry: dict, inputs: dict) -> str:
    summary = _dict(inputs.get("summary"))
    sc = _dict(_dict(inputs.get("seed_comparison")).get("runs"))
    seeds = [_dict(sc[k]) for k in sorted(sc)][:2]
    _, cand = _selected_candidate(summary)
    if cid in ("A2", "A10"):
        m = _boot(summary, "dev", "gold_lot_precision")
        zero = _contains_zero(m)
        wrong, emitted = None, _num(cand.get("gold_lot_n"))
        if emitted is not None and _num(cand.get("gold_lot_correct")) is not None:
            wrong = int(emitted - _num(cand["gold_lot_correct"]))
        head = ("Es la misma magnitud que A2 evaluada con confianza >= 0.70. " if cid == "A10" else "")
        text = (f"{head}El candidato tiene {fmt_count(wrong)} lotes erróneos de {fmt_count(emitted)} filas gold que emiten predial "
                f"(precisión {fmt_rate(cand.get('gold_lot_precision'))}) frente a la línea base ({fmt_rate(m.get('old'))}). "
                f"Bootstrap pareado: diferencia {fmt_delta(m.get('new'), m.get('old'))}, IC 95 % "
                f"[{fmt_rate(m.get('lo'))} a {fmt_rate(m.get('hi'))}], P(diferencia < 0) = {fmt_rate(m.get('p_diff_lt0'))}: ")
        text += ("dentro del ruido de muestreo (el intervalo contiene el cero). " if zero else
                 "fuera del ruido de muestreo (el intervalo no contiene el cero). " if zero is False else f"{PENDING}. ")
        vals = [_num(s.get("gold_lot_precision")) for s in seeds]
        base = _num(m.get("old"))
        if len(vals) == 2 and None not in vals and base is not None and max(vals) < base:
            text += (f"No está dentro de la dispersión entre semillas: ambas quedan por debajo de la línea base "
                     f"({fmt_rate(vals[0])} y {fmt_rate(vals[1])}), una brecha estable de pocas filas, no un efecto de semilla.")
        return text
    if cid == "A4":
        m = _boot(summary, "dev", "lost_correct")
        zero = _contains_zero(m)
        vals = [_num(s.get("lost_correct")) for s in seeds]
        text = (f"dev lost_correct {fmt_rate(entry.get('value'))} frente al límite {fmt_rate(entry.get('threshold'))}. Bootstrap pareado: "
                f"diferencia {fmt_delta(m.get('new'), m.get('old'), 'count')}, IC 95 % [{fmt_count(m.get('lo'))} a "
                f"{fmt_count(m.get('hi'))}]: ")
        text += ("dentro del ruido de muestreo. " if zero else "fuera del ruido de muestreo. " if zero is False else f"{PENDING}. ")
        if len(vals) == 2 and None not in vals:
            text += (f"Las dos semillas del candidato dan {fmt_count(vals[0])} y {fmt_count(vals[1])} (difieren en "
                     f"{fmt_count(abs(vals[0] - vals[1]))}), así que el valor queda dentro de la dispersión entre semillas."
                     if min(vals) <= (_num(cand.get("lost_correct")) or -1) <= max(vals) else
                     f"Las dos semillas dan {fmt_count(vals[0])} y {fmt_count(vals[1])}.")
        return text
    if cid == "A6":
        a6 = _dict(_get(summary, "acceptance", "a6_oof_ece"))
        return (f"Esto no es una cuestión de ruido: la regla de envío frente al refit sin restricción da falso, mientras que el ECE fuera "
                f"de muestra de ambas cabezas (predio {fmt_rate(a6.get('candidate_predio'))}, manzana {fmt_rate(a6.get('candidate_manzana'))}) "
                f"está muy por debajo de los límites ({fmt_rate(a6.get('baseline_refit_limit_predio'))} y "
                f"{fmt_rate(a6.get('baseline_refit_limit_manzana'))}). Es una decisión de criterio (sección 20).")
    if cid == "A9":
        return ("Esto no es una cuestión de ruido: el único test que falla preexiste y pertenece a otro proyecto de la misma carpeta; "
                f"{_pytest_counts(summary)}. `check_no_real_addresses.py` sale con código 0. Es una decisión de criterio (sección 20).")
    if cid == "A12":
        return ("La corrección de salidas en dev no cambia nada (0 violaciones restantes, métricas idénticas); el fallo se hereda de "
                "A2, A4 y A10. La parte congelada no se reevalúa por diseño (la congelada no se vuelve a puntuar con "
                "`validate_outputs=correct`).")
    return "sin lectura automática para este criterio; revisar el motivo de la tabla."


def _pytest_counts(summary: dict) -> str:
    text = _get(summary, "verification", "pytest")
    m = re.search(r"(\d+) passed, (\d+) failed", text) if isinstance(text, str) else None
    return f"pytest: {m.group(1)} pruebas pasan y {m.group(2)} falla" if m else f"pytest: {PENDING}"


def _sec_reading(inputs: dict) -> list[str]:
    results = normalize_results(_acceptance_results(inputs))
    rows = []
    for cid in CRITERIA_IDS:
        entry = results.get(cid)
        if entry is None:
            rows.append([cid, "NO EVALUADO", PENDING, PENDING, "ausente de los resultados de aceptación"])
            continue
        rows.append([cid, _STATUS_LABEL[entry["_status"]], _fmt_any(entry.get("value")), _fmt_any(entry.get("threshold")),
                     _reason_es(entry)])
    body = table(["ID", "Estado", "Valor", "Umbral", "Motivo"], rows)
    failed = [cid for cid in CRITERIA_IDS if cid in results and results[cid]["_status"] == "fail"]
    body += ["", "Lectura en lenguaje llano de lo que no aprueba (qué cae dentro del ruido y qué no):", ""]
    if not failed:
        body.append("No hay ningún criterio fallido que explicar.")
    for cid in failed:
        body.append(f"- **{cid}**: {_reading(cid, results[cid], inputs)}")
    return _section(19, "Lectura de A1-A12 frente a la banda de ruido", body)


# ---- 20 --------------------------------------------------------------------------------------------
def _sec_decisions(inputs: dict) -> list[str]:
    summary = _dict(inputs.get("summary"))
    m = _boot(summary, "dev", "gold_lot_precision")
    mf = _boot(summary, "frozen", "all_manzana_precision")
    one_row = _num(_get(summary, "seed_spread", "one_gold_row_pp_dev"))
    a6 = _dict(_get(summary, "acceptance", "a6_oof_ece"))
    body = ["Estas decisiones son del usuario; el informe solo expone los hechos. Ninguna se tomó automáticamente.", "",
            f"1. **A2, tolerancia cero.** A2 exige que la precisión de lote en dev gold sea >= la de la línea base ({fmt_rate(m.get('old'))}) sin "
            f"ninguna tolerancia. El candidato da {fmt_rate(m.get('new'))} (diferencia {fmt_delta(m.get('new'), m.get('old'))}; una fila gold "
            f"son {fmt_float(one_row, 3)} puntos). El IC del bootstrap es [{fmt_rate(m.get('lo'))} a {fmt_rate(m.get('hi'))}]. ¿Se mantiene la "
            "tolerancia cero (no aprueba) o se acepta una tolerancia de una o dos filas gold, coherente con la banda de ruido?",
            f"2. **A6, regla de envío frente al refit sin restricción.** El ECE fuera de muestra del candidato es muy bajo "
            f"({fmt_rate(a6.get('candidate_predio'))} y {fmt_rate(a6.get('candidate_manzana'))}), pero la regla de envío contra el refit "
            "sin restricción de la línea base da falso. ¿Se sigue exigiendo esa regla o basta el límite de ECE?",
            f"3. **A9, causada por un proyecto ajeno.** A9 falla solo por una prueba de otro proyecto de la misma carpeta; {_pytest_counts(summary)}. "
            "¿Se cuenta como aprobado el criterio con esa excepción documentada, o se corrige primero el proyecto ajeno?",
            f"4. **Pérdida de precisión manzana en filas congeladas ruidosas.** En la prueba congelada con todas las filas (SEGUNDA MIRADA) la "
            f"precisión manzana baja de {fmt_rate(mf.get('old'))} a {fmt_rate(mf.get('new'))} (diferencia {fmt_delta(mf.get('new'), mf.get('old'))}, "
            f"IC 95 % [{fmt_rate(mf.get('lo'))} a {fmt_rate(mf.get('hi'))}]), una caída que se distingue del ruido de muestreo, mientras que "
            "en dev A5 sí se cumple y la cobertura de filas OK sube. A5 se evalúa solo en dev y no cubre esas filas congeladas. ¿Es aceptable "
            "esa pérdida a cambio de la cobertura, o debe fijarse un criterio para las filas congeladas ruidosas?"]
    return _section(20, "Decisiones abiertas para el usuario", body)


# ---- 21 --------------------------------------------------------------------------------------------
def _sec_promotion(inputs: dict) -> list[str]:
    verdict = compute_verdict(_acceptance_results(inputs))
    body = [f"Veredicto calculado: {VERDICT_LABEL[verdict['verdict']]}.", "",
            "Estado de promoción: promoción pendiente de aprobación explícita del usuario. Este informe no promueve nada a `artifacts/`.",
            "", "Reversibilidad del respaldo de catastro (tres mecanismos independientes, con pruebas automáticas; ver "
                "`docs/respaldo-catastro.md`):", "",
            "1. **El índice base y los experimentos base nunca se modifican.** El respaldo vive en directorios nuevos y el modelo se "
            "reutiliza sin cambios; servir el índice base es seguir usando el índice base (`index_v2e_cert_pcer_r3`).",
            "2. **Retirar las filas agregadas** (índice y directorio de experimento): "
            "`.venv/Scripts/python.exe scripts/index_fallback.py remove --index-dir <directorio con respaldo> --out-dir <directorio nuevo>`. "
            "Verifica cada archivo contra el manifiesto del índice base; si algo no es idéntico byte a byte, falla y borra la salida.",
            "3. **Interruptor en tiempo de servicio, sin reconstruir.** `\"exclude_fuentes\": [\"catastro\"]` en `decision.json` y/o la variable "
            "de entorno `NORMALIZER_EXCLUDE_FUENTES=catastro` (se combinan). El servicio atiende exactamente el índice sin esos "
            "documentos; se aplica una vez por objeto normalizador y, para cambiarlo, se crea uno nuevo."]
    return _section(21, "Promoción y reversibilidad del respaldo de catastro", body)


# ---- document --------------------------------------------------------------------------------------
def render_report(inputs: dict, generated_at: str | None = None) -> str:
    """The whole Markdown document. Pure: same ``inputs`` and ``generated_at`` give identical text."""
    inputs = _dict(inputs)
    head = ["# Informe de reentrenamiento: normalizador basado en nomenclatura (2026-10)", "",
            "Público: los mantenedores y el equipo de datos que deciden si el normalizador reentrenado es al menos tan "
            "confiable como el de producción. Solo agregados (conteos, tasas, valores de métricas, nombres de reglas); sin "
            "direcciones, identificadores ni nombres."]
    if generated_at:
        head.append(f"Generado el: {cell(generated_at, 40)}")
    head.append("")
    lines = list(head)
    for build in (_sec_summary, _sec_scope, _sec_data, _sec_indexes, _sec_training, _sec_results, _sec_by_source,
                  _sec_validation, _sec_jev, _sec_checklist, _sec_risks, _sec_repro, _sec_story, _sec_noise,
                  _sec_bands, _sec_calibration, _sec_jev_final, _sec_perf, _sec_reading, _sec_decisions,
                  _sec_promotion):
        lines += build(inputs)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_if_changed(path: str, text: str) -> bool:
    """Write ``text`` (UTF-8, LF) unless the file already holds exactly that. Returns whether it wrote."""
    data = text.encode("utf-8")
    try:
        with open(path, "rb") as fh:
            if fh.read() == data:
                return False
    except OSError:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return True
