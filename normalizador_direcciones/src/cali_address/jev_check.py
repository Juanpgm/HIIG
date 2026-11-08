"""Offline semantic audit of the normalizer with the JEV judge (TypeSafe "System One").

Thin adapter over the sibling project ``models/validacion_jev`` (standard library only, tested on its own): it reuses
its client, question set, payload builder, confidence gate and metrics instead of re-implementing them. JEV is an
AUDITOR, never part of the serving path: one external call per row would add latency, cost and data exposure.

PRIVACY GUARD (non-negotiable). Every row carries an ``origin``. Only the origins in :data:`PUBLIC_ORIGINS` may be
sent to the external service:

``nomenclatura_public``  official public nomenclature rows: the raw ``direccion_original`` as input vs the
                         cleaned canonical address as output;
``synthetic``            queries produced by :func:`cali_address.augment.corrupt_many` from PUBLIC cadastre
                         addresses, scored through the production normalizer path.

Any other origin (``dev``, ``frozen_test``, ``train_pairs``, ``unlabeled``, missing, unknown) is refused with
:class:`PrivacyError` before a payload is built, unless the explicit ``allow_private=True`` is passed. The pipeline
never passes it. Only address strings leave the machine (the payload builder rejects any other field); the tags of a
row (estado, zone, cleaning flags) are used locally to split the metrics and are never sent.

The API key is handled only by ``validacion_jev`` (``JevConfig.from_env``); this module never reads it.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pandas as pd

__all__ = [
    "AuditResult",
    "AuditRow",
    "DEFAULT_MAX_ROWS",
    "JevUnavailableError",
    "PUBLIC_ORIGINS",
    "PrivacyError",
    "audit_rows",
    "disagreement_type",
    "dry_run_payloads",
    "load_validacion_jev",
    "rows_from_nomenclatura",
    "rows_from_synthetic",
    "sample_positions",
    "write_audit",
]

DEFAULT_MAX_ROWS = 300
KIND = "normalizador"
PUBLIC_ORIGINS = frozenset({"nomenclatura_public", "synthetic"})
ENV_SRC = "VALIDACION_JEV_SRC"
#: ``models/validacion_jev/src`` relative to this file (``models/normalizador_direcciones/src/cali_address``).
SIBLING_SRC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "validacion_jev", "src")
)


class PrivacyError(ValueError):
    """Rows whose origin is not public were about to be sent to the external service."""


class JevUnavailableError(RuntimeError):
    """The ``validacion_jev`` package cannot be imported or found."""


# --------------------------------------------------------------------------- lazy loader
def load_validacion_jev() -> SimpleNamespace:
    """Import ``validacion_jev`` lazily; fall back to ``$VALIDACION_JEV_SRC`` then to the sibling ``../validacion_jev/src``."""
    names = ("client", "io", "metrics", "questions", "validators")

    def _import():
        importlib.import_module("validacion_jev")
        return SimpleNamespace(**{n: importlib.import_module(f"validacion_jev.{n}") for n in names})

    try:
        return _import()
    except ImportError:
        pass
    tried = []
    for candidate in (os.environ.get(ENV_SRC), SIBLING_SRC):
        if not candidate:
            continue
        tried.append(candidate)
        if os.path.isdir(os.path.join(candidate, "validacion_jev")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            try:
                return _import()
            except ImportError:
                continue
    raise JevUnavailableError(
        "The 'validacion_jev' package was not found. Install it (pip install -e ../validacion_jev) or set "
        f"{ENV_SRC} to its 'src' directory. Looked in: {', '.join(tried) or '(no candidate paths)'}."
    )


# --------------------------------------------------------------------------- data model
@dataclass
class AuditRow:
    """One (input, output) pair to judge. ``tags`` are local grouping labels and never leave the machine."""

    id: str
    origin: str | None
    input_text: Any
    output_text: Any
    score: float | None = None
    tags: dict = field(default_factory=dict)
    #: Local numeric / label extras (never grouped, never sent): written as ``extra_<name>`` columns of the rows file.
    extra: dict = field(default_factory=dict)


@dataclass
class AuditResult:
    results: list[dict]
    summary: dict
    groups: dict
    disagreement_types: dict
    rows: list[AuditRow]
    capped: bool = False
    total_rows: int = 0
    source: str | None = None
    allow_private: bool = False

    def summary_for_json(self) -> dict:
        """The summary without any address text (disagreements keep ids, labels, certainty and a generic type)."""
        summary = {k: v for k, v in self.summary.items() if k != "disagreements"}
        by_id = {r["id"]: r for r in self.results}
        summary["disagreements"] = [
            {"id": d["id"], "jev_p": d["jev_p"], "label": d["label"], "certainty": d["certainty"], "score": d.get("score"),
             "type": by_id[d["id"]].get("disagreement_type"), "tags": by_id[d["id"]].get("tags", {})}
            for d in self.summary.get("disagreements", [])
        ]
        return _clean_json({
            "source": self.source,
            "privacy": {"origins": dict(Counter(str(r.origin) for r in self.rows)), "allow_private": self.allow_private},
            "capped": self.capped,
            "total_rows_offered": self.total_rows,
            **summary,
            "groups": self.groups,
            "disagreement_types": dict(self.disagreement_types),
        })


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean_json(value.item())
        except (ValueError, AttributeError):  # pragma: no cover
            return str(value)
    return value


# --------------------------------------------------------------------------- guard
def check_origins(rows: Sequence[AuditRow], allow_private: bool = False) -> None:
    """Raise :class:`PrivacyError` unless every row has a public origin (or ``allow_private``). Never echoes content."""
    if allow_private:
        return
    bad = Counter(str(r.origin) for r in rows if r.origin not in PUBLIC_ORIGINS)
    if bad:
        listing = ", ".join(f"{origin!r}: {count}" for origin, count in sorted(bad.items()))
        raise PrivacyError(
            f"refusing to send rows with non-public origin to the external JEV service ({listing}); "
            f"allowed origins: {', '.join(sorted(PUBLIC_ORIGINS))}. Nothing was sent."
        )


# --------------------------------------------------------------------------- items
def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _to_items(jev: SimpleNamespace, rows: Sequence[AuditRow]) -> list:
    items = []
    seen_ids: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        a, b = _text(row.input_text), _text(row.output_text)
        score = row.score if isinstance(row.score, (int, float)) and not isinstance(row.score, bool) and math.isfinite(row.score) else None
        item = jev.io.Item(str(row.id), a, b, True, score, "score" if score is not None else None, None, [])
        count = seen_ids.get(item.id, 0) + 1
        seen_ids[item.id] = count
        if count > 1:
            item.id = f"{item.id}#{count}"
            item.flags.append("duplicate_id")
        if a is None or b is None:
            item.skip_reason, item.detail = "invalid_input", "empty or non-textual address"
        else:
            pair = (a.strip(), b.strip())
            if pair in seen_pairs:
                item.skip_reason, item.detail = "duplicate_pair", "pair already sent in another row"
            seen_pairs.add(pair)
        items.append(item)
    return items


def _cap(rows: Sequence[AuditRow], max_rows: int) -> tuple[list[AuditRow], bool]:
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    rows = list(rows)
    return rows[:max_rows], len(rows) > max_rows


def dry_run_payloads(rows: Sequence[AuditRow], model: str = "jev-latest", max_rows: int = DEFAULT_MAX_ROWS,
                     allow_private: bool = False) -> list[dict]:
    """The exact request bodies that WOULD be sent (no client, no key, no network)."""
    check_origins(rows, allow_private)
    jev = load_validacion_jev()
    capped, _ = _cap(rows, max_rows)
    return jev.validators.dry_run_payloads(KIND, _to_items(jev, capped), model)


# --------------------------------------------------------------------------- disagreement types
_ABBREVIATIONS = {
    "CL": "CALLE", "CLL": "CALLE", "KR": "CARRERA", "CR": "CARRERA", "CRA": "CARRERA", "AV": "AVENIDA",
    "DG": "DIAGONAL", "TV": "TRANSVERSAL", "AP": "APARTAMENTO", "APTO": "APARTAMENTO", "CA": "CASA",
    "LC": "LOCAL", "OF": "OFICINA", "BG": "BODEGA", "PI": "PISO", "TO": "TORRE",
}


def _tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    return [_ABBREVIATIONS.get(t, t) for t in re.findall(r"[A-Z0-9]+", text)]


def disagreement_type(a: Any, b: Any) -> str:
    """Generic, non-identifying pattern of a mismatch between an input and its normalized output."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return "missing_text"
    if ta == tb:
        return "same_tokens_after_abbreviation"
    sa, sb = set(ta), set(tb)
    if sb < sa:
        return "tokens_dropped"
    if sa < sb:
        return "tokens_added"
    if re.findall(r"\d+", " ".join(ta)) != re.findall(r"\d+", " ".join(tb)):
        return "numbers_differ"
    if sorted(ta) == sorted(tb):
        return "reordered"
    return "letters_differ"


# --------------------------------------------------------------------------- audit
def _plain(value: Any) -> Any:
    """JSON-friendly scalar: numpy scalars unwrapped, NaN / NA -> None."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):  # pragma: no cover
            return str(value)
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _group_key(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "unknown" if value is None else str(value)


def audit_rows(
    rows: Sequence[AuditRow],
    client,
    thresholds=None,
    max_rows: int = DEFAULT_MAX_ROWS,
    allow_private: bool = False,
    source: str | None = None,
) -> AuditResult:
    """Judge the rows with JEV through ``client`` (a ``validacion_jev`` ``JevClient`` or any object with ``query``).

    Raises :class:`PrivacyError` BEFORE any payload is built when a row has a non-public origin, and the sibling's
    ``AuthenticationError`` (a ``ConfigurationError``) when the key is rejected. API failures, timeouts and
    malformed answers become ``status == "error"`` rows that are excluded from the agreement and counted.
    """
    rows = list(rows)
    check_origins(rows, allow_private)
    jev = load_validacion_jev()
    thresholds = thresholds or jev.validators.Thresholds()
    capped_rows, was_capped = _cap(rows, max_rows)
    items = _to_items(jev, capped_rows)
    results = jev.validators.run_validation(KIND, items, client, thresholds) if items else []
    duplicates = sum(1 for i in items if i.skip_reason == "duplicate_pair")

    primary = jev.questions.PRIMARY_QUESTION[KIND]
    for row, result in zip(capped_rows, results):
        result["origin"] = row.origin
        result["tags"] = {k: _plain(v) for k, v in row.tags.items()}
        result["extra"] = {k: _plain(v) for k, v in row.extra.items()}
        result["input_text"], result["output_text"] = row.input_text, row.output_text
        result["disagreement_type"] = (
            disagreement_type(row.input_text, row.output_text) if result.get("label") == "no" else None
        )
        result["primary_question"] = primary
    load_info = {"total_rows": len(rows), "max_rows": max_rows, "capped": was_capped, "duplicates_skipped": duplicates}
    summary = jev.metrics.summarize(results, KIND, thresholds, load_info)

    groups: dict[str, dict] = {}
    tag_names = sorted({k for r in capped_rows for k in r.tags})
    for name in tag_names:
        buckets: dict[str, list[dict]] = {}
        for row, result in zip(capped_rows, results):
            buckets.setdefault(_group_key(row.tags.get(name)), []).append(result)
        groups[name] = {}
        for key, members in sorted(buckets.items()):
            sub = jev.metrics.summarize(members, KIND, thresholds)
            groups[name][key] = {k: sub[k] for k in ("counts", "agreement", "api_failure_rate")}
    types = Counter(r["disagreement_type"] for r in results if r.get("disagreement_type"))
    return AuditResult(
        results=results, summary=summary, groups=groups, disagreement_types=dict(types), rows=capped_rows,
        capped=was_capped, total_rows=len(rows), source=source, allow_private=allow_private,
    )


def write_audit(result: AuditResult, out_dir: str) -> dict:
    """Write ``jev_audit.json`` (no address text) and ``jev_audit_rows.parquet`` (rows; keep it untracked)."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "jev_audit.json")
    rows_path = os.path.join(out_dir, "jev_audit_rows.parquet")
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result.summary_for_json(), fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    records = []
    for r in result.results:
        primary = r.get("primary_question")
        record = {
            "id": r["id"], "origin": r.get("origin"), "input_text": r.get("input_text"), "output_text": r.get("output_text"),
            "status": r.get("status"), "label": r.get("label"), "jev_p": r.get("jev_p"),
            "gate": (r.get("gates") or {}).get(primary), "score": r.get("score"),
            "flags": ";".join(r.get("flags") or []), "skip_reason": r.get("skip_reason"),
            "disagreement_type": r.get("disagreement_type"),
        }
        for name, value in (r.get("tags") or {}).items():
            record[f"tag_{name}"] = value
        for name, value in (r.get("extra") or {}).items():
            record[f"extra_{name}"] = value
        records.append(record)
    columns = ["id", "origin", "input_text", "output_text", "status", "label", "jev_p", "gate", "score", "flags",
               "skip_reason", "disagreement_type"]
    frame = pd.DataFrame(records, columns=None if records else columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    for column in ("input_text", "output_text"):
        frame[column] = frame[column].map(lambda v: v if isinstance(v, str) else None)
    frame.to_parquet(rows_path, index=False)
    return {"json": json_path, "rows": rows_path}


# --------------------------------------------------------------------------- row sources
def sample_positions(total: int, n: int, seed: int) -> list[int]:
    """Sorted positions of a deterministic sample without replacement (``n`` > ``total`` returns everything)."""
    if n < 0 or total < 0:
        raise ValueError("sizes must be non-negative")
    import numpy as np

    size = min(n, total)
    if size == 0:
        return []
    return sorted(int(i) for i in np.random.default_rng(seed).choice(total, size=size, replace=False))


_NOMENCLATURA_TAGS = {
    "estado_nomenclatura": "ndidestado", "zona": "ndidtipzon", "marker_removed": "marker_removed",
    "rebuilt_from_parts": "rebuilt_from_parts",
}


def rows_from_nomenclatura(frame: pd.DataFrame, n: int, seed: int) -> list[AuditRow]:
    """Public official rows: raw ``nddirecc_original`` (input) vs cleaned ``nddirecc`` (output). Origin ``nomenclatura_public``."""
    missing = [c for c in ("nddirecc_original", "nddirecc") if c not in frame.columns]
    if missing:
        raise KeyError(f"missing columns in the nomenclatura file: {', '.join(missing)}")
    usable = frame[[c for c in ("nddirecc_original", "nddirecc", *_NOMENCLATURA_TAGS.values()) if c in frame.columns]]
    keep = usable["nddirecc_original"].map(lambda v: _text(v) is not None) & usable["nddirecc"].map(lambda v: _text(v) is not None)
    usable = usable[keep]
    rows = []
    for pos in sample_positions(len(usable), n, seed):
        record = usable.iloc[pos]
        tags = {tag: record[col] for tag, col in _NOMENCLATURA_TAGS.items() if col in usable.columns}
        tags = {k: (bool(v) if k in ("marker_removed", "rebuilt_from_parts") else v) for k, v in tags.items()}
        rows.append(AuditRow(id=f"n{usable.index[pos]}", origin="nomenclatura_public", input_text=record["nddirecc_original"],
                             output_text=record["nddirecc"], tags=tags))
    return rows


def rows_from_synthetic(addresses: Sequence[str], n: int, seed: int, normalize: Callable[[list[str]], pd.DataFrame]) -> list[AuditRow]:
    """Synthetic queries (``corrupt_many`` of PUBLIC cadastre addresses) scored by ``normalize`` (the production path).

    ``normalize`` maps a list of query strings to a frame with ``direccion_normalizada`` and, optionally, ``estado``,
    ``confiabilidad`` / ``confianza`` and ``fuente_normalizacion``. Origin ``synthetic``.
    """
    from .augment import corrupt_many

    usable = [a for a in addresses if _text(a) is not None]
    chosen = [usable[i] for i in sample_positions(len(usable), n, seed)]
    if not chosen:
        return []
    rng = random.Random(seed)
    queries = [corrupt_many(address, 2, rng)[i % 2] for i, address in enumerate(chosen)]
    frame = normalize(queries)
    if len(frame) != len(queries):
        raise ValueError(f"the normalizer returned {len(frame)} rows for {len(queries)} queries")
    rows = []
    for i, query in enumerate(queries):
        record = frame.iloc[i]
        score = None
        for column in ("confiabilidad", "confianza"):
            value = record.get(column) if column in frame.columns else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                score = float(value)
                break
        tags = {k: record[k] for k in ("estado", "fuente_normalizacion", "banda_confiabilidad", "fuente") if k in frame.columns}
        tags = {k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v) for k, v in tags.items()}
        extra = {"noise": "light" if i % 2 == 0 else "heavy"}       # corrupt_many: variant 0 mild, variant 1 heavy
        for k in ("p_predio_cal", "p_manzana_cal"):
            if k in frame.columns:
                v = record[k]
                extra[k] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
        output = record.get("direccion_normalizada") if "direccion_normalizada" in frame.columns else None
        rows.append(AuditRow(id=f"s{i}", origin="synthetic", input_text=query,
                             output_text=output if isinstance(output, str) else None, score=score, tags=tags, extra=extra))
    return rows
