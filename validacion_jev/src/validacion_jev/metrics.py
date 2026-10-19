"""Pure metric functions. JEV is a model-based judge, not ground truth."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Sequence

from .validators import Thresholds, jev_label  # noqa: F401  (jev_label re-exported)

CAVEAT = (
    "JEV es un juez basado en un modelo de lenguaje, no una verdad de referencia: la concordancia "
    "no equivale a exactitud, y toda discrepancia requiere revision humana antes de concluir "
    "quien tiene la razon."
)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError("successes debe estar en [0, n].")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == n else min(1.0, center + half)
    return lo, hi


def confusion_matrix(pairs: Sequence[tuple[bool, bool]]) -> dict[str, int]:
    """pairs of (model says match, JEV says yes)."""
    c = Counter(pairs)
    return {
        "tp": c[(True, True)],
        "fp": c[(True, False)],
        "fn": c[(False, True)],
        "tn": c[(False, False)],
    }


def cohen_kappa(cm: dict[str, int]) -> float | None:
    n = sum(cm.values())
    if n == 0:
        return None
    po = (cm["tp"] + cm["tn"]) / n
    pe = ((cm["tp"] + cm["fp"]) * (cm["tp"] + cm["fn"]) + (cm["fn"] + cm["tn"]) * (cm["fp"] + cm["tn"])) / (n * n)
    if math.isclose(pe, 1.0):
        return None
    return (po - pe) / (1 - pe)


def precision_recall_f1(cm: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Mann-Whitney U with average ranks for ties; None if a class is missing."""
    if len(scores) != len(labels):
        raise ValueError("scores y labels deben tener la misma longitud.")
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier_score(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("scores y labels deben tener la misma longitud.")
    sq = [(s - float(y)) ** 2 for s, y in zip(scores, labels) if math.isfinite(s) and 0.0 <= s <= 1.0]
    return statistics.fmean(sq) if sq else None


def reliability_table(scores: Sequence[float], p_yes: Sequence[float], bins: int = 5) -> list[dict[str, Any]]:
    """Equal-count bins of the model score vs the mean JEV P(yes)."""
    pairs = sorted(zip(scores, p_yes), key=lambda t: t[0])
    n = len(pairs)
    if n == 0:
        return []
    k = min(bins, n)
    table = []
    for b in range(k):
        chunk = pairs[b * n // k : (b + 1) * n // k]
        table.append(
            {
                "n": len(chunk),
                "score_min": chunk[0][0],
                "score_max": chunk[-1][0],
                "mean_score": statistics.fmean(s for s, _ in chunk),
                "mean_p_yes": statistics.fmean(p for _, p in chunk),
            }
        )
    return table


def _certainty(p: float) -> float:
    return max(p, 1.0 - p)


def disagreements(results: Sequence[dict]) -> list[dict]:
    out = []
    for r in results:
        if r.get("status") == "ok" and r.get("label") in ("yes", "no") and r["decision"] != (r["label"] == "yes"):
            out.append(
                {
                    "id": r["id"],
                    "a": r.get("a"),
                    "b": r.get("b"),
                    "decision": r["decision"],
                    "jev_p": r["jev_p"],
                    "label": r["label"],
                    "certainty": _certainty(r["jev_p"]),
                    "score": r.get("score"),
                }
            )
    return sorted(out, key=lambda d: (-d["certainty"], str(d["id"])))


def summarize(
    results: Sequence[dict],
    kind: str,
    thresholds: Thresholds,
    load_info: dict | None = None,
) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    labeled = [r for r in ok if r.get("label") in ("yes", "no")]
    indeterminate = [r for r in ok if r.get("label") == "indeterminate"]

    agree = sum(1 for r in labeled if r["decision"] == (r["label"] == "yes"))
    n = len(labeled)
    cm = confusion_matrix([(bool(r["decision"]), r["label"] == "yes") for r in labeled])

    scored = [r for r in ok if _finite(r.get("score")) and r.get("jev_p") is not None]
    scored_labeled = [r for r in labeled if _finite(r.get("score"))]
    s_scores = [r["score"] for r in scored_labeled]
    s_labels = [r["label"] == "yes" for r in scored_labeled]

    gates: dict[str, dict[str, int]] = {}
    for r in ok:
        for qid, g in (r.get("gates") or {}).items():
            gates.setdefault(qid, {"auto": 0, "review": 0, "escalate": 0})[g] += 1

    usage = {"input_tokens": 0, "output_tokens": 0}
    for r in results:
        for k in usage:
            usage[k] += (r.get("usage") or {}).get(k, 0)

    attempted = len(ok) + len(errors)
    return {
        "kind": kind,
        "has_valid_evidence": n > 0,
        "counts": {
            "total": len(results),
            "ok": len(ok),
            "error": len(errors),
            "skipped": len(skipped),
            "skipped_by_reason": dict(Counter(r.get("skip_reason") for r in skipped)),
            "indeterminate": len(indeterminate),
            "labeled": n,
        },
        "api_failure_rate": len(errors) / attempted if attempted else None,
        "agreement": {
            "n": n,
            "agree": agree,
            "rate": agree / n if n else None,
            "ci95": list(wilson_interval(agree, n)) if n else None,
        },
        "confusion": cm,
        "kappa": cohen_kappa(cm),
        "precision_recall": precision_recall_f1(cm),
        "reliability": {
            "score_names": sorted({r["score_name"] for r in scored if r.get("score_name")}),
            "bins": reliability_table([r["score"] for r in scored], [r["jev_p"] for r in scored]),
        },
        "auc": roc_auc(s_scores, s_labels),
        "brier": brier_score(s_scores, s_labels),
        "disagreements": disagreements(results),
        "gates": gates,
        "usage": usage,
        "models_seen": sorted({m for r in results for m in (r.get("models") or []) if isinstance(m, str)}),
        "thresholds": {"auto": thresholds.auto, "review": thresholds.review},
        "input": dict(load_info or {}),
        "caveat": CAVEAT,
    }


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
