"""Builds JEV payloads from items, applies confidence gating and runs the judge.

Privacy boundary: ``build_state`` is the ONLY place a payload is assembled, and
``check_state`` re-verifies it. Only address strings (plus the categorical
``estado``) are ever sent; any other column of the input is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import AuthenticationError, JevClient
from .questions import MATCHER, NORMALIZER, PRIMARY_QUESTION, questions_for

MAX_FIELD_CHARS = 300
EPSILON = 1e-9  # absorbs float noise such as 1 - 0.15 != 0.85

REQUIRED_FIELDS = {
    NORMALIZER: ("direccion_original", "direccion_normalizada"),
    MATCHER: ("direccion_a", "direccion_b"),
}
OPTIONAL_FIELDS = {NORMALIZER: ("estado",), MATCHER: ()}


class StateError(ValueError):
    """The state violates the privacy / shape contract and must not be sent."""


@dataclass(frozen=True)
class Thresholds:
    auto: float = 0.85
    review: float = 0.60

    def __post_init__(self) -> None:
        if not (0.0 <= self.review <= self.auto <= 1.0):
            raise ValueError("Los umbrales deben cumplir 0 <= revision <= automatico <= 1.")


def certainty(answer: dict) -> float:
    """noul: max(p, 1-p) computed here; choice/score: the model's confidence."""
    if answer["type"] == "noul":
        p = answer["noul"]
        return max(p, 1.0 - p)
    return answer["confidence"]


def gate(value: float, thresholds: Thresholds) -> str:
    if value >= thresholds.auto - EPSILON:
        return "auto"
    if value >= thresholds.review - EPSILON:
        return "review"
    return "escalate"


def jev_label(p: float | None, thresholds: Thresholds) -> str | None:
    """yes / no only when the noul answer clears the 'auto' gate; else indeterminate."""
    if p is None:
        return None
    if gate(max(p, 1.0 - p), thresholds) != "auto":
        return "indeterminate"
    return "yes" if p >= 0.5 else "no"


def truncate(value: str, limit: int = MAX_FIELD_CHARS) -> tuple[str, bool]:
    return value[:limit], len(value) > limit


def _clean(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateError(f"El campo '{name}' debe ser una cadena no vacia.")
    return value.strip()


def build_state(kind: str, item: Any) -> tuple[dict[str, str], list[str]]:
    fields = REQUIRED_FIELDS[kind]
    state: dict[str, str] = {}
    flags: list[str] = []
    for name, value in zip(fields, (item.a, item.b)):
        state[name], cut = truncate(_clean(value, name))
        if cut:
            flags.append("truncated")
    estado = getattr(item, "estado", None) if kind == NORMALIZER else None
    if isinstance(estado, str) and estado.strip():
        state["estado"], cut = truncate(estado.strip())
        if cut:
            flags.append("truncated")
    check_state(kind, state)
    return state, sorted(set(flags))


def check_state(kind: str, state: dict) -> None:
    required, optional = REQUIRED_FIELDS[kind], OPTIONAL_FIELDS[kind]
    extra = set(state) - set(required) - set(optional)
    if extra:
        raise StateError(f"Campos no permitidos en el estado: {sorted(extra)}")
    for name in required:
        if name not in state:
            raise StateError(f"Falta el campo '{name}'.")
    for name, value in state.items():
        if not isinstance(value, str):
            raise StateError(f"El campo '{name}' debe ser una cadena.")
        if len(value) > MAX_FIELD_CHARS:
            raise StateError(f"El campo '{name}' supera {MAX_FIELD_CHARS} caracteres.")


def _prepare(kind: str, item: Any):
    """Return (state, flags, questions) or None when the item cannot be sent."""
    if item.skip_reason:
        return None
    try:
        state, flags = build_state(kind, item)
    except StateError:
        item.skip_reason = "invalid_input"
        return None
    return state, flags, questions_for(kind, include_estado="estado" in state)


def dry_run_payloads(kind: str, items: list, model: str) -> list[dict]:
    """Exact request bodies that would be sent (the key is never part of the body)."""
    payloads = []
    for item in items:
        prepared = _prepare(kind, item)
        if prepared is None:
            continue
        state, _, qs = prepared
        payloads.append({"id": item.id, "body": {"state": state, "model": model, "questions": qs}})
    return payloads


def _skipped(item: Any) -> dict:
    return {
        "id": item.id,
        "status": "skipped",
        "skip_reason": item.skip_reason,
        "detail": getattr(item, "detail", None),
        "flags": list(item.flags),
    }


def run_validation(kind: str, items: list, client: JevClient, thresholds: Thresholds) -> list[dict]:
    """Judge every sendable item. Raises AuthenticationError (aborts the run)."""
    primary = PRIMARY_QUESTION[kind]
    results: list[dict] = []
    for item in items:
        prepared = _prepare(kind, item)
        if prepared is None:
            results.append(_skipped(item))
            continue
        state, state_flags, qs = prepared
        res = client.query(state, qs)  # AuthenticationError propagates on purpose
        flags = sorted(set(item.flags) | set(state_flags))
        answers = res.answers
        p = answers[primary]["noul"] if primary in answers else None
        label = jev_label(p, thresholds)
        certainties = {qid: certainty(a) for qid, a in answers.items()}
        if p is None:
            status = "error"
            flags.append("error")
        else:
            status = "ok"
            if res.errors:
                flags.append("partial_error")
            if label == "indeterminate":
                flags.append("indeterminate")
        results.append(
            {
                "id": item.id,
                "kind": kind,
                "status": status,
                "decision": item.decision,
                "score": item.score,
                "score_name": item.score_name,
                "a": state[REQUIRED_FIELDS[kind][0]],
                "b": state[REQUIRED_FIELDS[kind][1]],
                "estado": state.get("estado"),
                "answers": answers,
                "certainty": certainties,
                "gates": {qid: gate(c, thresholds) for qid, c in certainties.items()},
                "jev_p": p,
                "label": label,
                "errors": res.errors,
                "warnings": res.warnings,
                "usage": res.usage,
                "models": list(res.models),
                "flags": sorted(set(flags)),
            }
        )
    return results
