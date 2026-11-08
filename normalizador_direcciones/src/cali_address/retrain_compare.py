"""Comparison tooling for the nomenclature retrain: index profile, gold coverage, coordinate shift,
row-level agreement, the confidence curve and the acceptance checklist (A1..A12).

Everything here is a pure function over DataFrames / dicts (no model, no network). The outputs are
AGGREGATES: counts, quantiles and shares. The only function that keeps row-level detail is
:func:`probe_table`, whose result belongs in ``artifacts/`` (untracked), never in a report.

Acceptance (see ``docs/retrain_2026-10_acceptance.md``): :func:`check_acceptance` implements the criteria
exactly as written there, including the 2026-10-31 amendment: A2, A3 and A7 are LOT-level (``NPN[:21]``) and
A7 no longer depends on ``npn_catastro``. Thresholds that are relative to a baseline (A3 frozen numbers, A6, A10, A12) are
computed from measured baseline metrics at check time; nothing is hard-coded for them.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd

from .evaluate import index_lots, predial_lot
from .geo import haversine_m
from .index_union import catastro_key_table

__all__ = [
    "PROBE_BUCKETS", "DEFAULT_CONFIDENCE_THRESHOLDS", "ProbeMisalignmentError", "index_profile",
    "gold_npn_coverage", "docs_lot_coverage", "oracle_ceiling_metrics", "ship_rule_vs_refit", "coordinate_shift", "probe_table", "probe_agreement", "predial_match_either",
    "predial_either_summary", "confidence_curve", "check_acceptance", "summarize_acceptance",
    "build_acceptance_thresholds", "collect_run_metrics", "ACCEPTANCE_CRITERIA", "BASELINE_RUN",
]

PROBE_BUCKETS = ("both_ok_predial_match", "only_old", "only_new", "both_wrong")
DEFAULT_CONFIDENCE_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))     # 0.50 .. 0.95
BASELINE_RUN = "baseline_prod_2026_10"
_EPS = 1e-12


class ProbeMisalignmentError(ValueError):
    """The two evaluation tables do not describe the same rows in the same order."""


# =====================================================================================================
# small helpers
# =====================================================================================================
def _clean(series: pd.Series) -> pd.Series:
    """Object series of stripped strings; null / NaN become ``""``."""
    return series.astype(object).map(lambda v: "" if v is None or v is pd.NA or (isinstance(v, float) and v != v)
                                     else str(v).strip())


def _need(df: pd.DataFrame, cols: Iterable[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing and len(df):
        raise ValueError(f"{what} lacks required column(s): {', '.join(missing)}")


def _truthy(value) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value == value and value == 1)
    return str(value).strip().lower() in ("true", "1")


def _explicit_false(value) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value == value and value == 0)
    return str(value).strip().lower() in ("false", "0")


def _py(obj):
    """JSON-friendly copy: numpy scalars to Python numbers."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _quantiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {}
    return {"p50": float(np.quantile(values, 0.5)), "p90": float(np.quantile(values, 0.9)),
            "p99": float(np.quantile(values, 0.99)), "max": float(np.max(values))}


# =====================================================================================================
# index profile
# =====================================================================================================
def _keyed(docs: pd.DataFrame, side: str) -> pd.DataFrame:
    """Docs with ``address_key`` and ``doc_key`` columns (computed from ``direccion`` when absent)."""
    _need(docs, ("direccion", "comuna"), f"{side} docs")
    if not len(docs):
        out = docs.copy()
        for col in ("address_key", "doc_key", "direccion", "comuna"):
            if col not in out.columns:
                out[col] = pd.Series([], dtype="object")
        return out
    out = catastro_key_table(docs.reset_index(drop=True))
    return out


def _key_overlap(old: pd.Series, new: pd.Series) -> dict:
    o, n = set(old[old != ""]), set(new[new != ""])
    return {"old_unique": len(o), "new_unique": len(n), "shared": len(o & n), "old_only": len(o - n),
            "new_only": len(n - o), "old_blank": int((old == "").sum()), "new_blank": int((new == "").sum())}


def _fuente_counts(docs: pd.DataFrame) -> dict:
    if "fuente" not in docs.columns:
        return {"catastro": int(len(docs))} if len(docs) else {}
    values = _clean(docs["fuente"]).replace("", "unknown")
    return {str(k): int(v) for k, v in values.value_counts().items()}


def _duplicates(docs: pd.DataFrame) -> dict:
    out = {}
    for key in ("address_key", "doc_key"):
        vals = docs[key]
        vals = vals[vals != ""]
        counts = vals.value_counts()
        dup = counts[counts > 1]
        out[f"{key}_groups"] = int(len(dup))
        out["docs_in_groups" if key == "address_key" else "docs_in_doc_key_groups"] = int(dup.sum())
    out["duplicate_doc_id"] = int(docs["doc_id"].duplicated().sum()) if "doc_id" in docs.columns else 0
    return out


_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), (">10", 11, 10 ** 9))


def _histogram(values: np.ndarray) -> dict:
    return {name: int(((values >= lo) & (values <= hi)).sum()) for name, lo, hi in _BUCKETS}


def _n_predios_shift(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    empty = {"shared_keys": 0, "equal": 0, "new_greater": 0, "new_smaller": 0,
             "quantiles": {"old": {}, "new": {}}, "histogram": {"old": _histogram(np.array([])),
                                                                "new": _histogram(np.array([]))}}
    if "n_predios" not in old.columns or "n_predios" not in new.columns or not len(old) or not len(new):
        return empty
    o = old.loc[old["address_key"] != "", ["address_key", "n_predios"]].drop_duplicates("address_key")
    n = new.loc[new["address_key"] != "", ["address_key", "n_predios"]].drop_duplicates("address_key")
    both = o.merge(n, on="address_key", suffixes=("_old", "_new"))
    if both.empty:
        return empty
    a = both["n_predios_old"].to_numpy(dtype="float64")
    b = both["n_predios_new"].to_numpy(dtype="float64")
    return {"shared_keys": int(len(both)), "equal": int((a == b).sum()), "new_greater": int((b > a).sum()),
            "new_smaller": int((b < a).sum()),
            "quantiles": {"old": _quantiles(a), "new": _quantiles(b)},
            "histogram": {"old": _histogram(a), "new": _histogram(b)}}


def index_profile(old_docs: pd.DataFrame, new_docs: pd.DataFrame) -> dict:
    """Aggregate comparison of two ``catastro_docs`` tables (counts, overlaps, distributions; no addresses).

    Keys are ``address_key`` (letters and digits only, collisions possible) and ``doc_key`` (keeps the
    ``#`` / ``-`` separators); both come from the columns when present, else from ``direccion``. A side
    without a ``fuente`` column is the v1 cadastre and counts as ``catastro``.
    """
    old, new = _keyed(old_docs, "old"), _keyed(new_docs, "new")
    oc = _clean(old["comuna"]) if len(old) else pd.Series([], dtype=object)
    nc = _clean(new["comuna"]) if len(new) else pd.Series([], dtype=object)
    o_counts, n_counts = oc[oc != ""].value_counts(), nc[nc != ""].value_counts()
    codes = sorted(set(o_counts.index) | set(n_counts.index))
    choice = {}
    for side, docs in (("old", old), ("new", new)):
        if "npn_choice" in docs.columns and len(docs):
            vals = _clean(docs["npn_choice"]).replace("", "none")
            choice[side] = {str(k): int(v) for k, v in vals.value_counts().items()}
        else:
            choice[side] = {}
    return _py({
        "n_docs": {"old": len(old), "new": len(new)},
        "counts_by_fuente": {"old": _fuente_counts(old), "new": _fuente_counts(new)},
        "address_key": _key_overlap(_clean(old["address_key"]), _clean(new["address_key"])),
        "doc_key": _key_overlap(_clean(old["doc_key"]), _clean(new["doc_key"])),
        "npn_choice": choice,
        "n_predios_shift": _n_predios_shift(
            old.assign(address_key=_clean(old["address_key"])) if len(old) else old,
            new.assign(address_key=_clean(new["address_key"])) if len(new) else new),
        "comunas": {
            "counts": {c: {"old": int(o_counts.get(c, 0)), "new": int(n_counts.get(c, 0))} for c in codes},
            "only_old": [c for c in codes if o_counts.get(c, 0) and not n_counts.get(c, 0)],
            "only_new": [c for c in codes if n_counts.get(c, 0) and not o_counts.get(c, 0)],
        },
        "duplicates": {
            "old": _duplicates(old.assign(address_key=_clean(old["address_key"]), doc_key=_clean(old["doc_key"])))
            if len(old) else {"address_key_groups": 0, "docs_in_groups": 0, "doc_key_groups": 0,
                              "docs_in_doc_key_groups": 0, "duplicate_doc_id": 0},
            "new": _duplicates(new.assign(address_key=_clean(new["address_key"]), doc_key=_clean(new["doc_key"])))
            if len(new) else {"address_key_groups": 0, "docs_in_groups": 0, "doc_key_groups": 0,
                              "docs_in_doc_key_groups": 0, "duplicate_doc_id": 0},
        },
    })


# =====================================================================================================
# gold coverage and coordinate shift
# =====================================================================================================
def gold_npn_coverage(splits_rows: pd.DataFrame, docs: pd.DataFrame, gold_only: bool = True) -> dict:
    """Share of ``gt_predial`` values whose LOT (``NPN[:21]``) is present among the index documents' NPN lots.

    ``splits_rows`` is dev + frozen concatenated. With a ``gt_quality`` column and ``gold_only`` only the
    ``gold`` rows count (``gold_only=False`` gives the all-with-ground-truth view). The ground truth is cadastral
    and lot-level while an index NPN may carry a unit suffix, so only the first 21 characters are compared;
    ``n_present_exact`` additionally reports the rows whose full code is an index NPN. Rows without a
    ``gt_predial`` are reported (``n_gold_rows``) but not in the denominator. A ground-truth code that is not a
    valid lot (short, non-digit) stays in the denominator as NOT present and is counted in ``n_invalid_gt``;
    invalid index NPNs are ignored. ``npn_catastro`` is never read: the audit-only catastral code is not an
    index key. ``n_missing`` is the absolute number of rows whose lot the index lacks.
    """
    if "gt_predial" not in splits_rows.columns:
        raise ValueError("splits_rows lacks required column(s): gt_predial")
    if "numero_predial_nacional" not in docs.columns:
        raise ValueError("docs lack required column(s): numero_predial_nacional")
    rows = splits_rows
    if gold_only and "gt_quality" in rows.columns:
        rows = rows[rows["gt_quality"] == "gold"]
    gt = _clean(rows["gt_predial"])
    has_gt = gt != ""
    npns = _clean(docs["numero_predial_nacional"])
    lots, exact = index_lots(npns), set(npns) - {""}
    gt_lot = gt.map(predial_lot)
    in_lots = has_gt & gt_lot.map(lambda v: v is not None and v in lots)
    n_with, n_present = int(has_gt.sum()), int(in_lots.sum())
    return {"n_gold_rows": int(len(rows)), "n_with_gt_predial": n_with, "n_present": n_present,
            "n_missing": n_with - n_present, "share_present": (n_present / n_with) if n_with else None,
            "n_present_exact": int((has_gt & gt.isin(exact)).sum()),
            "n_invalid_gt": int((has_gt & gt_lot.isna()).sum())}


def docs_lot_coverage(old_docs: pd.DataFrame, new_docs: pd.DataFrame) -> dict:
    """How much of the old (production) index the new one still reaches, by lot and by ``doc_key``.

    ``by_lot``: old documents with a valid lot whose lot (first 21 characters of the NPN, suffix ignored) is
    among the new documents' lots. ``by_doc_key``: old documents with a non-blank ``doc_key`` (computed from
    ``direccion`` when absent) whose key is a new document's key. ``n_lost`` documents have no counterpart. An
    empty denominator gives ``share_covered = None``. Counts only: no addresses.
    """
    for name, d in (("old", old_docs), ("new", new_docs)):
        _need(d, ("numero_predial_nacional",), f"{name} docs")
    old_lots = _clean(old_docs["numero_predial_nacional"]).map(predial_lot) if len(old_docs) else pd.Series([], dtype=object)
    new_lots = index_lots(_clean(new_docs["numero_predial_nacional"])) if len(new_docs) else set()
    has_lot = old_lots.notna().to_numpy(dtype=bool)
    lot_hit = np.array([v is not None and v in new_lots for v in old_lots], dtype=bool)

    def block(considered: int, covered: int) -> dict:
        return {"n_considered": considered, "n_covered": covered, "n_lost": considered - covered,
                "share_covered": (covered / considered) if considered else None}

    old_keys = _clean(_keyed(old_docs, "old")["doc_key"]) if len(old_docs) else pd.Series([], dtype=object)
    new_keys = set(_clean(_keyed(new_docs, "new")["doc_key"])) if len(new_docs) else set()
    new_keys.discard("")
    has_key = (old_keys != "").to_numpy(dtype=bool)
    key_hit = np.array([k in new_keys for k in old_keys], dtype=bool)
    return {"n_old_docs": int(len(old_docs)),
            "by_lot": block(int(has_lot.sum()), int((has_lot & lot_hit).sum())),
            "by_doc_key": block(int(has_key.sum()), int((has_key & key_hit).sum()))}


def _count(v) -> int | None:
    return int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v >= 0 else None


def oracle_ceiling_metrics(ceiling) -> dict:
    """A7 inputs derived from the ``oracle_ceiling`` block of ``index_meta.json`` (read defensively).

    The block is either flat or, as the index builder writes it, per split (``dev``, ``frozen_test``) plus a
    ``total`` entry; ``total`` (dev + frozen gold) is the one used. Needs non-negative integer ``rows``,
    ``prod_lot_matched``, ``new_lot_matched`` and ``lot_absent_from_source`` with ``rows > lot_absent_from_source``;
    anything else (not a dict, field missing, NaN, bool, string, no gold row with its lot in the source, an
    ``rows_with_lot_in_source`` that is not ``rows - lot_absent_from_source``) returns ``{}`` so the criterion is
    reported as not evaluated, never invented. ``rows_with_lot_in_source = rows - lot_absent_from_source`` is the
    construction population. The production share on that population uses ``prod_lot_matched_in_source`` when the
    builder writes it (production also matches lots that are not in the source, so ``prod_lot_matched`` over the
    in-source population could exceed 1), else ``prod_lot_matched``. Conditional shares: ``*_lot_matched`` (in
    source) / ``rows_with_lot_in_source``; coverage loss ``(prod_lot_matched - new_lot_matched) / rows``; inherent
    loss ``lot_absent_from_source / rows``. The optional ``lot_lost_incomplete`` / ``lot_not_chosen`` counts are
    passed through for the report.
    """
    if not isinstance(ceiling, dict):
        return {}
    src = ceiling
    if "total" in ceiling:
        src = ceiling["total"]
        if not isinstance(src, dict):
            return {}
    rows, prod, new, absent = (_count(src.get(k)) for k in
                               ("rows", "prod_lot_matched", "new_lot_matched", "lot_absent_from_source"))
    if None in (rows, prod, new, absent) or rows <= 0 or absent >= rows:
        return {}
    in_source = rows - absent
    if "rows_with_lot_in_source" in src and _count(src["rows_with_lot_in_source"]) != in_source:
        return {}
    prod_in = prod
    if "prod_lot_matched_in_source" in src:
        prod_in = _count(src["prod_lot_matched_in_source"])
        if prod_in is None or prod_in > in_source:
            return {}
    out = {"gold_rows_with_lot_in_source": in_source, "gold_lot_cond_prod": prod_in / in_source,
           "gold_lot_cond_new": new / in_source, "gold_coverage_loss": (prod - new) / rows,
           "gold_inherent_loss": absent / rows, "gold_lot_coverage_old": prod / rows,
           "gold_lot_coverage_new": new / rows, "gold_lot_missing_new": rows - new,
           "gold_lot_absent_from_source": absent}
    for key, dst in (("lot_lost_incomplete", "gold_lot_lost_incomplete"), ("lot_not_chosen", "gold_lot_not_chosen")):
        if _count(src.get(key)) is not None:
            out[dst] = _count(src[key])
    return out


def _npn_centroids(docs: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Median centroid per non-blank NPN, the number of distinct NPNs and the rows dropped for NaN."""
    if not len(docs):
        return pd.DataFrame(columns=["lon", "lat"]), 0, 0
    npn = _clean(docs["numero_predial_nacional"])
    frame = pd.DataFrame({"npn": npn.to_numpy(),
                          "lon": pd.to_numeric(docs["centroid_lon"], errors="coerce").to_numpy(),
                          "lat": pd.to_numeric(docs["centroid_lat"], errors="coerce").to_numpy()})
    frame = frame[frame["npn"] != ""]
    n_unique = int(frame["npn"].nunique())
    bad = frame[["lon", "lat"]].isna().any(axis=1)
    return frame[~bad].groupby("npn")[["lon", "lat"]].median(), n_unique, int(bad.sum())


def coordinate_shift(old_docs: pd.DataFrame, new_docs: pd.DataFrame) -> dict:
    """Distance in metres between the old and the new centroid of each NPN present on both sides.

    A NPN with several documents on one side uses the median of their centroids. Rows with a NaN
    coordinate are dropped and counted; a blank NPN never joins.
    """
    for name, d in (("old", old_docs), ("new", new_docs)):
        _need(d, ("numero_predial_nacional", "centroid_lon", "centroid_lat"), f"{name} docs")
    o, n_old, bad_o = _npn_centroids(old_docs)
    n, n_new, bad_n = _npn_centroids(new_docs)
    both = o.join(n, how="inner", lsuffix="_old", rsuffix="_new")
    out = {"n_old_npn": n_old, "n_new_npn": n_new, "n_shared_npn": int(len(both)), "n_nan_dropped": bad_o + bad_n,
           "quantiles_m": None, "share_over_m": None}
    if len(both):
        d = haversine_m(both["lat_old"].to_numpy(), both["lon_old"].to_numpy(),
                        both["lat_new"].to_numpy(), both["lon_new"].to_numpy())
        out["quantiles_m"] = {f"p{int(q * 100)}": float(np.quantile(d, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
        out["quantiles_m"].update(max=float(d.max()), mean=float(d.mean()))
        out["share_over_m"] = {str(m): float((d > m).mean()) for m in (50, 100, 500)}
    return out


# =====================================================================================================
# row agreement
# =====================================================================================================
def _fuente_lookup(docs: pd.DataFrame) -> dict:
    """``numero_predial_nacional -> fuente`` (``mixed`` when documents sharing the NPN disagree)."""
    if "numero_predial_nacional" not in docs.columns or "fuente" not in docs.columns:
        return {}
    frame = pd.DataFrame({"npn": _clean(docs["numero_predial_nacional"]).to_numpy(),
                          "fuente": _clean(docs["fuente"]).to_numpy()})
    frame = frame[(frame["npn"] != "") & (frame["fuente"] != "")]
    return {npn: (g.iloc[0] if g.nunique() == 1 else "mixed") for npn, g in frame.groupby("npn")["fuente"]}


def probe_table(rows_old: pd.DataFrame, rows_new: pd.DataFrame, docs_new: pd.DataFrame | None = None,
                correct_col: str = "correct_predial") -> pd.DataFrame:
    """Row-by-row agreement of two evaluation tables, joined BY POSITION.

    ``raw_address`` must be identical at every position; otherwise :class:`ProbeMisalignmentError`
    (naming the position, never the text). A row is "OK" when ``estado == 'OK'`` and ``correct_col`` (default
    the exact ``correct_predial``; ``correct_predial_lot`` for the lot-level view) is true; unknown correctness
    counts as wrong. The matched document's ``fuente`` comes from a ``fuente``
    column of ``rows_new`` or, with ``docs_new``, from the output NPN.
    """
    for name, rows in (("old", rows_old), ("new", rows_new)):
        _need(rows, ("raw_address", "estado", correct_col), f"{name} rows")
    if len(rows_old) != len(rows_new):
        raise ProbeMisalignmentError(
            f"the tables have different length ({len(rows_old)} vs {len(rows_new)}): rows are misaligned")
    if not len(rows_old):
        return pd.DataFrame({"raw_address": [], "estado_old": [], "estado_new": [], "ok_old": [], "ok_new": [],
                             "bucket": []})
    a = _clean(rows_old["raw_address"]).to_numpy() if len(rows_old) else np.array([], dtype=object)
    b = _clean(rows_new["raw_address"]).to_numpy() if len(rows_new) else np.array([], dtype=object)
    bad = np.flatnonzero(a != b)
    if len(bad):
        raise ProbeMisalignmentError(
            f"raw_address differs at {len(bad)} position(s), first at row {int(bad[0])}: rows are misaligned")
    ok_old = np.array([e == "OK" and _truthy(c) for e, c in zip(rows_old["estado"], rows_old[correct_col])], dtype=bool)
    ok_new = np.array([e == "OK" and _truthy(c) for e, c in zip(rows_new["estado"], rows_new[correct_col])], dtype=bool)
    bucket = np.where(ok_old & ok_new, "both_ok_predial_match",
                      np.where(ok_old, "only_old", np.where(ok_new, "only_new", "both_wrong")))
    table = pd.DataFrame({"raw_address": rows_old["raw_address"].to_numpy(), "estado_old": rows_old["estado"].to_numpy(),
                          "estado_new": rows_new["estado"].to_numpy(), "ok_old": ok_old, "ok_new": ok_new,
                          "bucket": bucket})
    if "fuente" in rows_new.columns:
        table["fuente"] = _clean(rows_new["fuente"]).replace("", "unknown").to_numpy()
    elif docs_new is not None and "numero_predial_nacional" in rows_new.columns:
        lookup = _fuente_lookup(docs_new)
        table["fuente"] = [lookup.get(npn, "unknown") for npn in _clean(rows_new["numero_predial_nacional"])]
    return table


def probe_agreement(rows_old: pd.DataFrame, rows_new: pd.DataFrame, docs_new: pd.DataFrame | None = None,
                    correct_col: str = "correct_predial") -> dict:
    """Counts of the four agreement buckets (and per ``fuente`` when it is known). Counts only."""
    table = probe_table(rows_old, rows_new, docs_new, correct_col=correct_col)
    counts = {b: int((table["bucket"] == b).sum()) for b in PROBE_BUCKETS}
    out: dict = {"n": int(len(table)), "counts": counts}
    if "fuente" in table.columns:
        out["by_fuente"] = {
            str(f): {b: int(((table["fuente"] == f) & (table["bucket"] == b)).sum()) for b in PROBE_BUCKETS}
            for f in sorted(table["fuente"].unique())
        }
    return out


def predial_match_either(rows: pd.DataFrame, docs: pd.DataFrame) -> pd.Series:
    """Per row: output NPN == ``gt_predial`` OR the matched document's ``npn_catastro`` == ``gt_predial``.

    Measures how much of the "error" is the ground truth being cadastral. The matched document comes from a
    ``doc_id`` column of ``rows`` when present, else every document with the output NPN is a candidate.
    Rows without ``gt_predial`` are False.
    """
    _need(rows, ("numero_predial_nacional", "gt_predial"), "rows")
    if not len(rows):
        return pd.Series([], dtype=bool, index=rows.index)
    out_npn, gt = _clean(rows["numero_predial_nacional"]), _clean(rows["gt_predial"])
    cat = _clean(docs["npn_catastro"]) if "npn_catastro" in docs.columns else pd.Series([""] * len(docs), dtype=object)
    result = (gt != "") & (out_npn == gt)
    if "doc_id" in rows.columns:
        keyed = docs["doc_id"] if "doc_id" in docs.columns else pd.Series(docs.index, index=docs.index)
        by_id = dict(zip(keyed.tolist(), cat.tolist()))
        doc_cat = rows["doc_id"].map(lambda i: by_id.get(i, "") if i == i and i is not None else "")
        return pd.Series(((gt != "") & ((out_npn == gt) | (doc_cat.to_numpy() == gt.to_numpy()))).to_numpy(),
                         index=rows.index)
    npn_docs = _clean(docs["numero_predial_nacional"]) if "numero_predial_nacional" in docs.columns else cat.iloc[:0]
    sets: dict[str, set] = {}
    for npn, c in zip(npn_docs.tolist(), cat.tolist()):
        if npn and c:
            sets.setdefault(npn, set()).add(c)
    via_cat = pd.Series([bool(g) and g in sets.get(o, ()) for o, g in zip(out_npn.tolist(), gt.tolist())],
                        index=rows.index)
    return pd.Series((result | via_cat).to_numpy(), index=rows.index)


def predial_either_summary(rows: pd.DataFrame, docs: pd.DataFrame) -> dict:
    """Among ``estado == 'OK'`` rows: strict predial hits, hits against either source and the gain."""
    _need(rows, ("estado",), "rows")
    ok = (rows["estado"] == "OK").to_numpy() if len(rows) else np.array([], dtype=bool)
    either = predial_match_either(rows, docs).to_numpy(dtype=bool) if len(rows) else np.array([], dtype=bool)
    if len(rows):
        strict = ((_clean(rows["numero_predial_nacional"]) == _clean(rows["gt_predial"]))
                  & (_clean(rows["gt_predial"]) != "")).to_numpy()
    else:
        strict = np.array([], dtype=bool)
    n_strict, n_either = int((ok & strict).sum()), int((ok & either).sum())
    return {"n_ok_rows": int(ok.sum()), "strict": n_strict, "either": n_either, "gain": n_either - n_strict}


# =====================================================================================================
# confidence curve (A10)
# =====================================================================================================
def confidence_curve(rows: pd.DataFrame, thresholds: Iterable[float] | None = None) -> list[dict]:
    """Precision versus confidence threshold over ``estado == 'OK'`` rows (the real column is ``confianza``).

    For every threshold ``t``: ``n_ok_at_t`` OK rows with ``confianza >= t``; ``share_of_ok`` (over OK
    rows) and ``share_of_rows`` (over all rows given, i.e. the gold set); ``predial_precision`` and
    ``manzana_error_rate`` among those rows, plus ``predial_lot_precision`` (same, at lot level, when the rows
    carry ``correct_predial_lot``) and ``predial_lot_precision_emitted`` / ``n_emitting_predial`` (lot precision
    among the rows that emit a predial; the denominator A10 uses). A ratio with an empty denominator is ``None``, never a division
    by zero. Unknown correctness counts as not-correct for the precision and as not-wrong for the error rate;
    NaN confidence never reaches a threshold.
    """
    ts = list(DEFAULT_CONFIDENCE_THRESHOLDS if thresholds is None else thresholds)
    _need(rows, ("estado", "confianza", "correct_predial", "correct_manzana"), "rows")
    n_rows = int(len(rows))
    has_lot = "correct_predial_lot" in rows.columns
    if n_rows:
        ok = (rows["estado"] == "OK").to_numpy()
        conf = pd.to_numeric(rows["confianza"], errors="coerce").to_numpy(dtype="float64")
        pred = np.array([_truthy(v) for v in rows["correct_predial"]], dtype=bool)
        pred_lot = np.array([_truthy(v) for v in rows["correct_predial_lot"]], dtype=bool) if has_lot else None
        emits = (np.array([v != "" for v in _clean(rows["numero_predial_nacional"])], dtype=bool)
                 if "numero_predial_nacional" in rows.columns else None)
        manz_bad = np.array([_explicit_false(v) for v in rows["correct_manzana"]], dtype=bool)
    else:
        ok = conf = pred = manz_bad = np.array([])
        pred_lot = emits = None
    n_ok = int(ok.sum()) if n_rows else 0
    out = []
    for t in ts:
        m = (ok & (conf >= t)) if n_rows else np.array([], dtype=bool)
        n_at = int(m.sum()) if n_rows else 0
        out.append({
            "threshold": float(t), "n_rows": n_rows, "n_ok": n_ok, "n_ok_at_t": n_at,
            "share_of_ok": (n_at / n_ok) if n_ok else None,
            "share_of_rows": (n_at / n_rows) if n_rows else None,
            "predial_precision": float(pred[m].mean()) if n_at else None,
            "manzana_error_rate": float(manz_bad[m].mean()) if n_at else None,
            # Lot-level counterpart (``NPN[:21]``); None when the rows predate the lot columns.
            "predial_lot_precision": float(pred_lot[m].mean()) if (n_at and pred_lot is not None) else None,
            # The same lot-level precision over the OK rows that EMIT a predial only (shared addresses answer OK
            # without one and would count as wrong above); ``n_emitting_predial`` is that denominator.
            "n_emitting_predial": int((m & emits).sum()) if (n_rows and emits is not None) else None,
            "predial_lot_precision_emitted": (float(pred_lot[m & emits].mean())
                                              if (n_rows and emits is not None and pred_lot is not None
                                                  and (m & emits).any()) else None),
        })
    return out


# =====================================================================================================
# acceptance
# =====================================================================================================
ACCEPTANCE_CRITERIA = {
    "A1": "dev_gold_ok >= baseline - 1",
    "A2": "dev_gold_predial_lot_precision >= baseline (lot level, NPN[:21]; the exact precision is reported)",
    "A3": "frozen gold OK >= baseline - 2 rows; frozen gold lot-level predial precision >= baseline - 2 / baseline "
          "frozen gold OK",
    "A4": "dev_lost_correct <= max(measured baseline, s2b_retrain_s42, s2b_retrain_s43 dev_lost_correct) (112 when "
          "the noise band is not recorded)",
    "A5": "dev_manzana_precision >= baseline - 0.005",
    "A6": "refitted reliability (same kind as production), BOTH heads: pooled CV calibration error (ECE) <= the "
          "refitted baseline's + 0.01, and the ship rule against the refit on the same rows passes (the rule against "
          "the v2 reference is not required)",
    "A7": "cert parser failure rate < 1 %; CRS gates pass; (c1) construction: over gold rows whose lot exists in the "
          "cleaned source records, the new index's matched share >= the production index's - 2.0 pp; (c2) coverage "
          "loss: (production lot share - new lot share) on gold <= the inherent loss (lot absent from the source / "
          "rows) + 1.0 pp; the headline coverage numbers and the missing-lot counts are reported; accounting.dropped "
          "== 0 or every drop explained",
    "A8": "production_untouched is true in every run",
    "A9": "full pytest green in .venv; check_no_real_addresses.py exits 0",
    "A10": "at confianza >= 0.7 on dev gold rows, relative to the measured baseline: (a) share of gold rows with "
           "estado OK >= baseline - 1.0 pp; (b) lot-level predial precision among OK rows that emit a predial >= "
           "baseline - 0.5 pp; (c) manzana error rate among OK rows <= baseline + 0.5 pp",
    "A11": "cleaning quarantine < 0.5 %; build_index.py and download_nomenclatura.py exit 0 on real data and on "
           "the anomaly fixtures",
    "A12": "with validate_outputs=correct: A1-A5 and A10 do not get worse and 0 violations remain on dev",
}

_DEFAULT_CONSTANTS = {
    "A1_margin_rows": 1, "A3_margin_rows": 2, "A4_max_lost_correct": 112, "A5_margin": 0.005, "A6_margin": 0.01,
    "A7_max_parser_failure_rate": 0.01, "A7_construction_margin": 0.02, "A7_loss_margin": 0.01, "A10_confidence": 0.7, "A10_share_margin": 0.01,
    "A10_precision_margin": 0.005, "A10_manzana_margin": 0.005,
    "A11_max_quarantine_share": 0.005,
}
_REQUIRED_BASELINE = ("dev_gold_ok", "dev_gold_predial_precision", "dev_manzana_precision", "dev_lost_correct")
_BASELINE_INT_KEYS = {"dev_ok", "dev_n", "dev_gold_n", "dev_gold_ok", "dev_lost_correct", "dev_gold_lost_correct"}
_BASELINE_FLOAT_KEYS = {"dev_gold_ok_coverage", "dev_gold_manzana_precision", "dev_gold_predial_precision",
                        "dev_manzana_precision", "dev_predial_precision",
                        # Lot level: present only when the leaderboard row was written after the lot columns existed.
                        "dev_gold_predial_lot_precision", "dev_predial_lot_precision"}
_SELECTION_RULE = (
    "Among runs whose dev_gold_ok_coverage (gold rows answered OK / gold rows) is at least the MEASURED baseline's "
    "(baseline_prod_2026_10), pick the highest dev_gold_predial_lot_precision; ties (within one gold row) go to "
    "dev_manzana_precision on all dev rows. "
    "Recall metrics are diagnostics, never selectors. frozen_test is scored once, with --final, on the winner."
)


def _noise_band(noise_rows: dict | None) -> dict:
    """``{tag: {"dev_lost_correct": int}}`` from leaderboard rows; unusable entries (blank, NaN, text) are skipped."""
    out: dict = {}
    for tag, row in (noise_rows or {}).items():
        try:
            value = float((row or {}).get("dev_lost_correct"))
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            out[str(tag)] = {"dev_lost_correct": int(round(value))}
    return out


def build_acceptance_thresholds(baseline_row: dict, created: str, noise_rows: dict | None = None) -> dict:
    """The machine-readable thresholds file from a ``leaderboard.csv`` baseline row (strings are coerced).

    ``noise_rows`` maps a tag to its leaderboard row (the two production-recipe retrains ``s2b_retrain_s42`` /
    ``s2b_retrain_s43``): their ``dev_lost_correct`` form the noise band behind A4 (see :func:`check_acceptance`).
    """
    ref: dict = {}
    for key in set(_BASELINE_INT_KEYS) | set(_BASELINE_FLOAT_KEYS):
        raw = baseline_row.get(key)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"baseline value {key}={raw!r} is not a number") from None
        if math.isnan(value):
            raise ValueError(f"baseline value {key} is NaN")
        ref[key] = int(round(value)) if key in _BASELINE_INT_KEYS else value
    for key in _REQUIRED_BASELINE:
        if key not in ref:
            raise ValueError(f"baseline row lacks a usable {key}")
    return {
        "schema": 1, "created": created, "change": "retrain_2026-10_nomenclatura",
        "baseline_run": BASELINE_RUN, "baseline_leaderboard_tag": "baseline_prod",
        "baseline_reference": dict(sorted(ref.items())),
        "noise_band": _noise_band(noise_rows),
        "constants": dict(_DEFAULT_CONSTANTS),
        "criteria": dict(ACCEPTANCE_CRITERIA),
        "relative_to_measured_baseline": ["A1", "A2", "A3", "A4", "A5", "A6", "A10", "A12"],
        "relative_note": ("The MEASURED baseline run (baseline_prod_2026_10, current code) replaces the stored "
                          "baseline_prod numbers wherever it is provided: A1, A4 (with the noise band), A5, A2/A3 "
                          "(lot level), A6 (refitted baseline), A10 and A12. The stored reference is only the "
                          "fallback and historical information."),
        "selection_rule": _SELECTION_RULE,
    }


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _part(name: str, value, op: str, threshold, threshold_name: str | None = None, tol: float = _EPS) -> dict:
    """One comparison: ``status`` is ``pass``, ``fail`` or ``missing`` and ``reason`` says why."""
    tname = threshold_name or "threshold"
    base = {"name": name, "value": value, "threshold": threshold, "op": op}

    def done(status: str, reason: str = "") -> dict:
        return {**base, "status": status, "reason": reason}

    if value is None:
        return done("missing", f"missing metric {name}")
    if threshold is None:
        return done("missing", f"missing {tname} (needed to judge {name})")
    if op == "is_true":
        return done("pass") if value is True or isinstance(value, np.bool_) and bool(value) else done(
            "fail", f"{name} is {value!r}, expected true" if isinstance(value, (bool, np.bool_)) else
            f"{name}={value!r} is not a boolean")
    if not _is_number(value):
        return done("fail", f"{name}={value!r} is not a number")
    if not _is_number(threshold):
        return done("fail", f"{tname}={threshold!r} is not a number")
    if math.isnan(value):
        return done("fail", f"NaN metric {name}")
    if math.isnan(threshold):
        return done("fail", f"NaN {tname} (cannot judge {name})")
    ok = {">=": value >= threshold - tol, "<=": value <= threshold + tol, "<": value < threshold - tol,
          "==": value == threshold}[op]
    return done("pass") if ok else done("fail", f"{name}={value} violates {op} {threshold}")


def _criterion(cid: str, parts: list[dict]) -> dict:
    fails = [p for p in parts if p["status"] == "fail"]
    missing = [p for p in parts if p["status"] == "missing"]
    status = "fail" if fails else ("not_evaluated" if missing else "pass")
    if len(parts) == 1:
        value, threshold = parts[0]["value"], parts[0]["threshold"]
    else:
        value = {p["name"]: p["value"] for p in parts}
        threshold = {p["name"]: p["threshold"] for p in parts}
    return {"id": cid, "description": ACCEPTANCE_CRITERIA[cid], "value": value, "threshold": threshold,
            "passed": {"pass": True, "fail": False, "not_evaluated": None}[status], "status": status,
            "reason": "; ".join(p["reason"] for p in fails + missing), "parts": parts}


def _sub(x: float | None, d: float) -> float | None:
    return None if x is None else (x - d if _is_number(x) else x)


def _add(x, d):
    return None if x is None else (x + d if _is_number(x) else x)


def _curve_point(curve, t: float):
    if not isinstance(curve, list):
        return None
    for p in curve:
        if isinstance(p, dict) and abs(float(p.get("threshold", -1)) - t) < 1e-9:
            return p
    return None


class _Ctx:
    """Candidate / baseline metric lookup with the stored baseline reference as the fallback."""

    def __init__(self, cand: dict, base: dict, ref: dict, k: dict, noise: list | None = None) -> None:
        self.c, self.b, self.ref, self.k = cand, base, ref, k
        self.noise = noise or []

    def base(self, key: str):
        v = self.b.get(key)
        return v if v is not None else self.ref.get(key)


def _a4_limit(ctx: _Ctx):
    """max(measured baseline, noise band) of ``dev_lost_correct``; the constant when no noise band is recorded."""
    if not ctx.noise:
        return ctx.k["A4_max_lost_correct"]
    base = ctx.base("dev_lost_correct")
    values = list(ctx.noise) + ([base] if _is_number(base) and not math.isnan(base) else [])
    return max(values)


def _a_simple(ctx: _Ctx) -> dict:
    k, c = ctx.k, ctx.c
    return {
        "A1": [_part("dev_gold_ok", c.get("dev_gold_ok"), ">=", _sub(ctx.base("dev_gold_ok"), k["A1_margin_rows"]),
                     "baseline dev_gold_ok")],
        "A2": [_part("dev_gold_predial_lot_precision", c.get("dev_gold_predial_lot_precision"), ">=",
                     ctx.base("dev_gold_predial_lot_precision"), "baseline dev_gold_predial_lot_precision")],
        "A3": _a3(ctx),
        "A4": [_part("dev_lost_correct", c.get("dev_lost_correct"), "<=", _a4_limit(ctx))],
        "A5": [_part("dev_manzana_precision", c.get("dev_manzana_precision"), ">=",
                     _sub(ctx.base("dev_manzana_precision"), k["A5_margin"]), "baseline dev_manzana_precision")],
    }


def _a3(ctx: _Ctx) -> list[dict]:
    c, k = ctx.c, ctx.k
    base_ok, base_prec = ctx.base("frozen_gold_ok"), ctx.base("frozen_gold_predial_lot_precision")
    prec_floor = None
    if base_prec is not None and _is_number(base_ok) and base_ok and not math.isnan(base_ok):
        prec_floor = _sub(base_prec, k["A3_margin_rows"] / base_ok)
    elif _is_number(base_ok) and not base_ok:
        prec_floor = None
    return [
        _part("frozen_gold_ok", c.get("frozen_gold_ok"), ">=", _sub(base_ok, k["A3_margin_rows"]),
              "baseline frozen_gold_ok"),
        _part("frozen_gold_predial_lot_precision", c.get("frozen_gold_predial_lot_precision"), ">=", prec_floor,
              "baseline frozen_gold_predial_lot_precision"),
    ]


def _a6(ctx: _Ctx) -> list[dict]:
    c, k = ctx.c, ctx.k
    parts = [_part("ship_rule_vs_refit", c.get("ship_rule_vs_refit"), "is_true", True)]
    for head in ("predio", "manzana"):                     # both heads, always
        key = f"calib_error_cv_{head}"
        parts.append(_part(key, c.get(key), "<=", _add(ctx.b.get(key), k["A6_margin"]), f"refitted baseline {key}"))
    return parts


def _a7(ctx: _Ctx) -> list[dict]:
    c, k = ctx.c, ctx.k
    dropped, explained = c.get("accounting_dropped"), c.get("accounting_dropped_explained")
    if dropped is None:
        drop = _part("accounting_dropped", None, "==", 0)
    elif _is_number(dropped) and not math.isnan(dropped) and dropped != 0 and explained is True:
        drop = {"name": "accounting_dropped", "value": dropped, "threshold": "0 or every drop explained", "op": "==",
                "status": "pass", "reason": ""}
    else:
        drop = _part("accounting_dropped", dropped, "==", 0)
    parts = [
        _part("parser_failure_rate_cert", c.get("parser_failure_rate_cert"), "<", k["A7_max_parser_failure_rate"]),
        _part("crs_gates_passed", c.get("crs_gates_passed"), "is_true", True),
        _part("gold_lot_cond_new", c.get("gold_lot_cond_new"), ">=",
              _sub(c.get("gold_lot_cond_prod"), k["A7_construction_margin"]),
              "production gold_lot_cond_prod (minus the margin)"),
        _part("gold_coverage_loss", c.get("gold_coverage_loss"), "<=",
              _add(c.get("gold_inherent_loss"), k["A7_loss_margin"]),
              "gold_inherent_loss (plus the margin)"),
        drop,
    ]
    for key in ("gold_lot_coverage_new", "gold_lot_coverage_old", "gold_lot_missing_new", "gold_lot_absent_from_source",
                "gold_lot_lost_incomplete", "gold_lot_not_chosen"):
        if c.get(key) is not None:                         # information only, never a gate
            parts.append({"name": key, "value": c[key], "threshold": "reported only", "op": "report",
                          "status": "pass", "reason": ""})
    return parts


def _a8(ctx: _Ctx) -> list[dict]:
    parts = []
    for name, src in (("production_untouched", ctx.c), ("baseline_production_untouched", ctx.b)):
        v = src.get("production_untouched")
        if name.startswith("baseline") and v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = all(bool(x) is True for x in v) if len(v) else None
        parts.append(_part(name, v, "is_true", True))
    return parts


def _a10(ctx: _Ctx, cand: dict | None = None) -> list[dict]:
    """Relative, lot level, with the right denominator (see ``ACCEPTANCE_CRITERIA['A10']``)."""
    t, k = float(ctx.k["A10_confidence"]), ctx.k
    cand = ctx.c if cand is None else cand
    cp, bp = _curve_point(cand.get("confidence_curve"), t), _curve_point(ctx.b.get("confidence_curve"), t)
    if cp is None or bp is None:
        which = "candidate" if cp is None else "baseline"
        return [_part("confidence_curve", None, ">=", 0, f"{which} confidence curve point at {t}")
                | {"reason": f"missing {which} confidence_curve point at threshold {t}"}]
    key = "predial_lot_precision_emitted"
    parts = [_part(f"share_of_rows@{t}", cp.get("share_of_rows"), ">=", _sub(bp.get("share_of_rows"), k["A10_share_margin"]),
                   "baseline share_of_rows (minus the margin)")]
    if not cp.get("n_ok_at_t"):
        parts.append({"name": f"{key}@{t}", "value": None, "threshold": None, "op": ">=", "status": "fail",
                      "reason": f"no rows with confianza >= {t} in the candidate"})
    elif cp.get("n_emitting_predial") == 0:
        parts.append({"name": f"{key}@{t}", "value": None, "threshold": None, "op": ">=", "status": "fail",
                      "reason": f"no OK row with confianza >= {t} emits a predial in the candidate"})
    else:
        parts.append(_part(f"{key}@{t}", cp.get(key), ">=", _sub(bp.get(key), k["A10_precision_margin"]),
                           f"baseline {key} (minus the margin; the baseline needs rows that emit a predial at the threshold)"))
    if cp.get("n_emitting_predial") is not None:
        parts.append({"name": f"n_emitting_predial@{t}", "value": cp["n_emitting_predial"], "threshold": "reported only",
                      "op": "report", "status": "pass", "reason": ""})
    cand_err = cp.get("manzana_error_rate") if cp.get("n_ok_at_t") else 0.0
    parts.append(_part(f"manzana_error_rate@{t}", cand_err, "<=", _add(bp.get("manzana_error_rate"), k["A10_manzana_margin"]),
                       "baseline manzana error rate (plus the margin; the baseline has no rows at or above the threshold)"))
    return parts


def _a12(metrics: dict, ctx: _Ctx) -> list[dict]:
    cc = metrics.get("candidate_correct")
    if not isinstance(cc, dict) or not cc:
        return [_part("candidate_correct", None, "is_true", True)]
    k = ctx.k
    cctx = _Ctx(cc, ctx.b, ctx.ref, k, ctx.noise)
    parts: list[dict] = []
    for cid, group in {**_a_simple(cctx), "A10": _a10(cctx)}.items():
        crit = _criterion(cid, group)
        parts.append({"name": f"{cid}_with_correct", "value": crit["value"], "threshold": crit["threshold"], "op": "criterion",
                      "status": {"pass": "pass", "fail": "fail", "not_evaluated": "missing"}[crit["status"]],
                      "reason": f"{cid} with validate_outputs=correct: {crit['reason']}" if crit["reason"] else ""})
    for key, op in (("dev_gold_ok", ">="), ("dev_gold_predial_lot_precision", ">="), ("dev_lost_correct", "<="),
                    ("dev_manzana_precision", ">="), ("frozen_gold_ok", ">="),
                    ("frozen_gold_predial_lot_precision", ">=")):
        parts.append(_part(f"{key}_not_worse", cc.get(key), op, ctx.c.get(key), f"uncorrected candidate {key}"))
    t = float(k["A10_confidence"])
    cpc, cpu = _curve_point(cc.get("confidence_curve"), t), _curve_point(ctx.c.get("confidence_curve"), t)
    for key, op in (("share_of_rows", ">="), ("predial_lot_precision_emitted", ">="), ("manzana_error_rate", "<=")):
        parts.append(_part(f"{key}@{t}_not_worse", None if cpc is None else cpc.get(key), op,
                           None if cpu is None else cpu.get(key), f"uncorrected candidate {key}@{t}"))
    parts.append(_part("violations_remaining_dev", cc.get("violations_remaining_dev"), "==", 0))
    return parts


def check_acceptance(metrics: dict, thresholds: dict) -> list[dict]:
    """Evaluate A1..A12. ``metrics = {'candidate': {...}, 'baseline': {...}, 'candidate_correct': {...}}``.

    Returns one dict per criterion: ``id``, ``value``, ``threshold``, ``passed`` (True / False / None when it
    could not be evaluated), ``status`` (``pass`` / ``fail`` / ``not_evaluated``), ``reason`` and the
    ``parts`` compared. Boundaries are inclusive (equal passes; ``<`` criteria are strict). A NaN metric
    FAILS with a reason; a missing one is ``not_evaluated`` and never silently passes. Baseline values come
    from ``metrics['baseline']`` first, then from ``thresholds['baseline_reference']``.
    """
    if not isinstance(thresholds, dict) or not isinstance(thresholds.get("baseline_reference"), dict) \
            or not isinstance(thresholds.get("constants"), dict):
        raise ValueError("thresholds must hold 'baseline_reference' and 'constants' objects")
    metrics = metrics or {}
    for section in ("candidate", "baseline", "candidate_correct"):
        if section in metrics and not isinstance(metrics[section], dict):
            raise ValueError(f"metrics['{section}'] must be an object, got {type(metrics[section]).__name__}")
    constants = {**_DEFAULT_CONSTANTS, **thresholds["constants"]}
    noise = []
    for row in (thresholds.get("noise_band") or {}).values():
        v = row.get("dev_lost_correct") if isinstance(row, dict) else None
        if _is_number(v) and not math.isnan(v):
            noise.append(v)
    ctx = _Ctx(metrics.get("candidate", {}), metrics.get("baseline", {}), thresholds["baseline_reference"], constants,
               noise)
    groups = _a_simple(ctx)
    groups.update({
        "A6": _a6(ctx), "A7": _a7(ctx), "A8": _a8(ctx),
        "A9": [_part("pytest_green", ctx.c.get("pytest_green"), "is_true", True),
               _part("no_real_addresses_exit", ctx.c.get("no_real_addresses_exit"), "==", 0)],
        "A10": _a10(ctx),
        "A11": [_part("quarantine_share", ctx.c.get("quarantine_share"), "<", constants["A11_max_quarantine_share"]),
                _part("build_index_exit", ctx.c.get("build_index_exit"), "==", 0),
                _part("download_exit", ctx.c.get("download_exit"), "==", 0),
                _part("anomaly_fixtures_ok", ctx.c.get("anomaly_fixtures_ok"), "is_true", True)],
        "A12": _a12(metrics, ctx),
    })
    return [_criterion(f"A{i}", groups[f"A{i}"]) for i in range(1, 13)]


def summarize_acceptance(results: Iterable[dict]) -> dict:
    """``all_passed`` is true only when every criterion passed: a criterion that was not evaluated blocks it."""
    results = list(results)
    failed = [r["id"] for r in results if r["passed"] is False]
    unknown = [r["id"] for r in results if r["passed"] is None]
    return {"all_passed": bool(results) and not failed and not unknown, "failed": failed, "not_evaluated": unknown,
            "passed": [r["id"] for r in results if r["passed"] is True]}


# =====================================================================================================
# run metrics (for scripts/compare_runs.py)
# =====================================================================================================
def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


_SHIP_BRIER_TOLERANCE = 0.002        # mirrors scripts/fit_reliability.py (SHIP_BRIER_TOLERANCE)


def _load_reliability(exp_dir: str):
    """``reliability.json`` of the directory, else the ``reliability_<kind>_candidate.json`` a failed ship rule leaves."""
    rel = _load_json(os.path.join(exp_dir, "reliability.json"))
    if isinstance(rel, dict):
        return rel
    try:
        names = sorted(n for n in os.listdir(exp_dir) if n.startswith("reliability_") and n.endswith("_candidate.json"))
    except OSError:
        return None
    for name in names:
        cand = _load_json(os.path.join(exp_dir, name))
        if isinstance(cand, dict):
            return cand
    return None


def ship_rule_vs_refit(rel: dict) -> bool | None:
    """The ship rule against the logistic REFIT on the same rows, recomputed from ``cv.pooled`` (the file stores
    only the combined ``ship_rule_passed``): per head (predio, manzana) the file's ``kind`` has Brier <= the
    refit's + 0.002 and AUC >= the refit's. ``None`` when the numbers are missing; a NaN makes it ``False``."""
    kind = rel.get("kind")
    pooled = (rel.get("cv") or {}).get("pooled") if isinstance(rel.get("cv"), dict) else None
    if not isinstance(pooled, dict) or not isinstance(kind, str):
        return None
    result = True
    for head in ("predio", "manzana"):
        h = pooled.get(head)
        if not isinstance(h, dict) or not isinstance(h.get(kind), dict) or not isinstance(h.get("logistic"), dict):
            return None
        c, b = h[kind], h["logistic"]
        vals = [c.get("brier"), b.get("brier"), c.get("auc"), b.get("auc")]
        if not all(_is_number(v) for v in vals):
            return None
        if any(math.isnan(v) for v in vals):
            result = False
            continue
        if not (c["brier"] <= b["brier"] + _SHIP_BRIER_TOLERANCE + 1e-9 and c["auc"] >= b["auc"]):
            result = False
    return result


def _calibrated_metrics(exp_dir: str) -> dict:
    """A6 inputs of a directory served with the calibrated layer (``reliability_calibrated.json``, kind ``logistic_ext``).

    The out-of-fold numbers live in ``fit_report.json`` next to it (``heads.<head>.oof`` for the served constrained head
    and ``heads.<head>.refit_oof`` for the unconstrained refit on the same rows and folds, both written by
    ``scripts/fit_decision.py``). ``calib_error_cv_<head>`` is the OOF ``ece10``; ``ship_rule_vs_refit`` is the legacy rule
    applied to those pairs (Brier <= refit + 0.002 and AUC >= refit, both heads). Anything unreadable is omitted; a NaN is
    kept (or makes the rule False) so the check can fail it.
    """
    cal = _load_json(os.path.join(exp_dir, "reliability_calibrated.json"))
    if not isinstance(cal, dict) or cal.get("kind") != "logistic_ext":
        return {}
    out: dict = {"calibrated_layer": True}
    report = _load_json(os.path.join(exp_dir, "fit_report.json"))
    heads = report.get("heads") if isinstance(report, dict) else None
    if not isinstance(heads, dict):
        return out
    rule: bool | None = True
    for head in ("predio", "manzana"):
        h = heads.get(head)
        oof = h.get("oof") if isinstance(h, dict) else None
        if not isinstance(oof, dict):
            rule = None
            continue
        if oof.get("ece10") is not None:
            out[f"calib_error_cv_{head}"] = oof["ece10"]
        refit = h.get("refit_oof")
        vals = [oof.get("brier"), oof.get("auc"), refit.get("brier") if isinstance(refit, dict) else None,
                refit.get("auc") if isinstance(refit, dict) else None]
        if not all(_is_number(v) for v in vals):
            rule = None
            continue
        if any(math.isnan(v) for v in vals):
            if rule is not None:
                rule = False
            continue
        if not (vals[0] <= vals[2] + _SHIP_BRIER_TOLERANCE + 1e-9 and vals[1] >= vals[3] - 1e-12):
            if rule is not None:
                rule = False
    if rule is not None:
        out["ship_rule_vs_refit"] = rule
    return out


def collect_run_metrics(exp_dir: str) -> dict:
    """Flat metrics of one experiment directory; whatever cannot be read is OMITTED (never invented).

    Reads ``experiment.json`` (leaderboard row, frozen gold numbers, ``production_untouched``),
    ``reliability.json`` (``ship_rule_passed`` and the CV calibration error of its ``kind``) and
    ``strict_eval_dev_gold_rows.parquet`` (the confidence curve). The reliability file must be the one fitted
    INSIDE this directory; nothing is taken from the production ``artifacts/``.
    """
    m: dict = {}
    exp = _load_json(os.path.join(exp_dir, "experiment.json"))
    if isinstance(exp, dict):
        row = exp.get("leaderboard_row")
        if isinstance(row, dict):
            for key in ("dev_gold_ok", "dev_gold_predial_precision", "dev_manzana_precision", "dev_lost_correct",
                        "dev_gold_n", "dev_gold_manzana_precision", "dev_gold_predial_lot_precision",
                        "dev_predial_lot_precision", "dev_gold_lot_missing_from_index"):
                if row.get(key) is not None:
                    m[key] = row[key]
        gold = (exp.get("frozen_test") or {}).get("gold") if isinstance(exp.get("frozen_test"), dict) else None
        if isinstance(gold, dict):
            for src, dst in (("test_gold_ok", "frozen_gold_ok"),
                             ("test_gold_predial_precision", "frozen_gold_predial_precision"),
                             ("test_gold_predial_lot_precision", "frozen_gold_predial_lot_precision")):
                if gold.get(src) is not None:
                    m[dst] = gold[src]
        if exp.get("production_untouched") is not None:
            m["production_untouched"] = exp["production_untouched"]
    rel = _load_reliability(exp_dir)
    if isinstance(rel, dict):
        vs_refit = ship_rule_vs_refit(rel)
        if vs_refit is not None:
            m["ship_rule_vs_refit"] = vs_refit
        if isinstance(rel.get("ship_rule_passed"), bool):
            m["ship_rule_passed"] = rel["ship_rule_passed"]
        kind = rel.get("kind")
        pooled = (rel.get("cv") or {}).get("pooled") if isinstance(rel.get("cv"), dict) else None
        if isinstance(pooled, dict) and isinstance(kind, str):
            for head in ("predio", "manzana"):
                ece = ((pooled.get(head) or {}).get(kind) or {}).get("ece") if isinstance(pooled.get(head), dict) else None
                if ece is not None:
                    m[f"calib_error_cv_{head}"] = ece
    calibrated = _calibrated_metrics(exp_dir)
    if calibrated:                                                    # the served (calibrated) heads win over a legacy file
        for key in ("calib_error_cv_predio", "calib_error_cv_manzana", "ship_rule_vs_refit"):
            m.pop(key, None)
        m.update(calibrated)
    meta = _load_json(os.path.join(exp_dir, "index_meta.json"))       # linked into index-trained experiment dirs
    if isinstance(meta, dict):
        m.update(oracle_ceiling_metrics(meta.get("oracle_ceiling")))
    rows_path = os.path.join(exp_dir, "strict_eval_dev_gold_rows.parquet")
    if os.path.isfile(rows_path):
        try:
            m["confidence_curve"] = confidence_curve(pd.read_parquet(rows_path))
        except (ValueError, OSError):
            pass
    return m
