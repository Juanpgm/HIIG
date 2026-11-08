"""Document index built ONLY from the official nomenclature (index v2c / v2d).

Pure functions (no I/O) behind ``scripts/build_index.py``:

* :func:`build_nomenclatura` / :func:`build_nomenclatura_docs` turn cleaned WFS rows
  (``artifacts/nomenclatura_clean.parquet``) into BASE documents (one per ``doc_key`` and, for groups that
  span several distant lots, one per lot), a sidecar of units (rows that carry an interior such as
  ``AP 302``), a per-record accounting and the table of quarantined records.
* :func:`union_docs` / :func:`union_own_centroids` add the ``doc_id`` / metric coordinates and the sidecar.
  The cadastre is READ-ONLY audit input (``npn_catastro`` / ``direccion_catastro``); it is never a source of
  documents, NPN, ``n_predios``, manzana or centroids.
* :func:`official_queries` turns the nomenclature strings into ``(text, doc_id)`` training pairs
  (token-level dedup, at most :data:`OFFICIAL_CAP_PER_DOC` per document).
* :func:`accounting` checks that every input record landed in exactly one bucket.
* :func:`remap_train_pairs`, :func:`oracle_ceiling`, :func:`catastro_coverage` compare the new index with the
  old cadastral one.

Resilience: a bad row never raises. It lands in exactly one accounting bucket (:data:`BUCKETS`). Records
that cannot become a faithful document are quarantined (counted, kept with a reason): no coordinates / empty
address (``dropped``), an address without via number, cross number or plate (``incomplete_address``) and
points far from the rest of their address group (``spread_outlier``). Everything is deterministic: records
are processed in ``gid`` order and documents are ordered ``cert`` (by ``gid``), then ``pcer`` (by ``gid``).

Id space: nomenclature documents carry ``objectid = gid + NOMENCLATURE_OBJECTID_OFFSET`` (raw gid in
``source_objectid``) so the id can never collide with a cadastral OBJECTID. NOTE: the ``catastro.parquet``
OBJECTID fallback of ``inference.load_own_centroids`` is NOT valid for these indexes (their objectids are not
cadastral OBJECTIDs); the sidecar ``catastro_own_centroids.parquet`` written with the index is mandatory.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .dataset import OWN_CENTROID_COLUMNS, query_text_for
from .geo import to_metric
from .labels import address_key
from .model import encode_text
from .parser import (
    COMPLEMENT_CANON, _pre_normalize, canonical, is_truncated, normalize_spacing, parse_address,
)

__all__ = [
    "BUCKETS", "TIERS", "FUENTE", "COMPAT_COLUMNS", "AccountingError", "NomenclatureBuild",
    "build_nomenclatura", "build_nomenclatura_docs", "doc_key", "catastro_key_table",
    "union_docs", "union_own_centroids", "assign_unit_doc_ids", "official_queries", "accounting",
    "remap_train_pairs", "pair_npn_agreement", "oracle_ceiling", "oracle_ceiling_gate", "catastro_coverage",
]

#: Accounting buckets of a nomenclature record. ``excluded_tier`` covers records whose ``ndidestado`` is
#: not among the tiers being built (e.g. ``pcer`` rows in an index of ``cert`` only).
BUCKETS = ("base_doc_new", "base_doc_merged", "unit", "fallback_doc", "dropped", "excluded_tier",
           "incomplete_address", "spread_outlier")

#: Nomenclature tiers by priority (lower rank wins a shared address key).
TIERS = ("cert", "pcer")
_RANK = {tier: i for i, tier in enumerate(TIERS)}
FUENTE = {"cert": "nomenclatura_cert", "pcer": "nomenclatura_pcer"}

#: Added to the WFS ``gid`` to build the document ``objectid``.
NOMENCLATURE_OBJECTID_OFFSET = 10_000_000
#: A group whose points are farther than this (m) from their median centroid is split by lot / trimmed.
SPREAD_SPLIT_M = 1000.0
#: Maximum official training queries per document.
OFFICIAL_CAP_PER_DOC = 12
OFFICIAL_SEED = 42
#: Lot-level oracle gate: the new share may be at most this many percentage points below production.
ORACLE_MAX_DROP_POINTS = 2.0

#: Columns of ``catastro_docs.parquet`` in their production order: they keep name and dtype.
COMPAT_COLUMNS = [
    "direccion", "numero_predial_nacional", "manzana", "comuna", "barrio", "barrio_code", "centroid_lon",
    "centroid_lat", "n_predios", "manzana_nunique", "objectid", "x_m", "y_m", "doc_id",
]
_STR_COMPAT = ("direccion", "numero_predial_nacional", "manzana", "comuna", "barrio", "barrio_code")
#: Columns appended AFTER the compatible ones (``n_lots_cert``, ``n_lots_pcer``, ``lot_conflict`` last).
EXTRA_COLUMNS = [
    "fuente", "estado_nomenclatura", "npn_catastro", "direccion_catastro", "npn_choice", "n_units",
    "source_objectid", "parse_ok", "zona", "address_key", "doc_key", "spread_m",
    "n_lots_cert", "n_lots_pcer", "lot_conflict",
]
_STR_EXTRA = ("fuente", "estado_nomenclatura", "npn_catastro", "direccion_catastro", "npn_choice", "zona",
              "address_key", "doc_key")

REQUIRED_COLUMNS = ("gid", "nddirecc", "ndnpn", "ndidestado", "lon", "lat")
UNIT_COLUMNS = ["address_key_base", "interior", "numero_predial_nacional", "own_lon", "own_lat", "objectid",
                "direccion_original", "estado", "doc_ord"]
FALLBACK_MAX_CHARS = 40
_LOT_LEN, _MZ_LEN = 21, 17


class AccountingError(ValueError):
    """The per-record buckets do not add up to the input count (or are malformed)."""


# ---------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------

def _s(value) -> str:
    """Text of a possibly-null cell: ``""`` for None / NaN / NA, whitespace-stripped otherwise."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _npn_ok(npn: str) -> bool:
    return len(npn) == 30 and npn.startswith("76001") and npn.isdigit()


def _str_series(values, index=None) -> pd.Series:
    return pd.Series(list(values), dtype="str", index=index)


def _lot(npn: object) -> str:
    text = _s(npn)
    return text[:_LOT_LEN] if len(text) >= _LOT_LEN else ""


# ---------------------------------------------------------------------------------------------------------
# row -> base text
# ---------------------------------------------------------------------------------------------------------

def _registered_kind(tail: str) -> bool:
    """True when ``tail`` starts with a complement kind registered in the parser (AP, BLQ, TO, GA, ...)."""
    parts = tail.split()
    return bool(parts) and parts[0].upper().strip(".") in COMPLEMENT_CANON


def _split_interior(full: str, interior: str) -> tuple[str, str, str]:
    """``(base_text, interior_text, method)`` of one address.

    Methods: ``suffix`` (``interior`` closes the address after whitespace), ``parser`` (the parser's
    complement, only when it starts with a REGISTERED complement kind), ``none_unregistered_tail`` (the parser
    found a tail such as a bare ``9 9`` that is not a registered kind: it is NOT stripped, the full text is the
    base) and ``none``.
    """
    if interior and full.endswith(interior) and len(full) > len(interior) and full[-len(interior) - 1].isspace():
        return full[: -len(interior)].rstrip(), interior, "suffix"
    parsed = parse_address(full)
    if parsed.parse_ok and not is_truncated(parsed):
        base = canonical(parsed, style="spaced", with_complement=False)
        whole = canonical(parsed, style="spaced", with_complement=True)
        if base and whole.startswith(base) and whole[len(base):].strip():
            tail = whole[len(base):].strip()
            if _registered_kind(tail):
                return base, tail, "parser"
            return full, "", "none_unregistered_tail"
    return full, "", "none"


def _is_complete(parsed) -> bool:
    return all(v not in (None, "") for v in (parsed.via_number, parsed.cross_number, parsed.plate))


def _doc_text(base: str) -> tuple[str, bool, str, bool]:
    """``(direccion, parse_ok, doc_key, complete)`` of a base text.

    A base that does not parse, or whose parse was truncated, can never be rebuilt faithfully, so it is
    indexed as ``_pre_normalize(base)[:40]`` with ``parse_ok = False`` (and counts as complete: it is a
    fallback document, not a degenerate one). A parsed base without via number, cross number or plate is
    ``complete = False``.
    """
    parsed = parse_address(normalize_spacing(base))
    if parsed.parse_ok and not is_truncated(parsed):
        text = canonical(parsed, style="spaced")
        key = doc_key(text) if text else ""
        if text and key:
            return text, True, key, _is_complete(parsed)
    fallback = _pre_normalize(base)[:FALLBACK_MAX_CHARS].strip()
    return fallback, False, (doc_key(fallback) if fallback else ""), True


def doc_key(text: object) -> str:
    """Join key of a document: the whitespace-free canonical text (``""`` for a blank / symbol-only text).

    Same spelling folding as :func:`labels.address_key` (``KR 32B # 46 - 131`` and ``KR 32 B # 46 - 131``
    share it) but the ``#`` and ``-`` separators are kept. ``address_key`` keeps only letters and digits,
    so ``CL 1 # 4 - 49`` and ``CL 14 # 49 -`` collapse to one key although they are different addresses
    (hundreds of real documents collide that way); joining on it would silently merge them. A text that
    does not parse, or whose parse was truncated, keys on its pre-normalised text.
    """
    raw = _s(text)
    if not raw:
        return ""
    parsed = parse_address(raw)
    canon = canonical(parsed, style="spaced") if parsed.parse_ok and not is_truncated(parsed) else ""
    basis = canon or _pre_normalize(raw)
    key = re.sub(r"\s+", "", basis)
    return key if re.search(r"[0-9A-Za-z]", key) else ""


def _punctuation_free(text: str) -> str:
    return _squash(re.sub(r"[#\-]", " ", text))


# ---------------------------------------------------------------------------------------------------------
# group spread
# ---------------------------------------------------------------------------------------------------------

def _spread(idx: list[int], x: np.ndarray, y: np.ndarray) -> float:
    """Max distance (m) of the points ``idx`` to their median centroid."""
    if len(idx) <= 1:
        return 0.0
    xs, ys = x[idx], y[idx]
    return float(np.max(np.hypot(xs - np.median(xs), ys - np.median(ys))))


def _split_group(members: list[int], wrows: list[int], x: np.ndarray, y: np.ndarray,
                 lot_of: Callable[[int], str]) -> tuple[list[list[int]], list[int]]:
    """``(parts, outliers)`` of one key group (positions ascending).

    Within :data:`SPREAD_SPLIT_M` the group stays whole. Otherwise, with two or more distinct valid lots among
    the winning-tier rows the group is split by lot (rows without a valid lot become outliers; lower-tier rows
    follow their lot, or the nearest part); with fewer, the rows farther than the limit from the densest point
    are outliers.
    """
    if _spread(wrows, x, y) <= SPREAD_SPLIT_M:
        return [members], []
    wset = set(wrows)
    lots = sorted({lot_of(i) for i in wrows} - {""})
    if len(lots) >= 2:
        parts: dict[str, list[int]] = {lot: [] for lot in lots}
        outliers: list[int] = []
        for i in wrows:
            lot = lot_of(i)
            (parts[lot] if lot else outliers).append(i)
        centers = {lot: (float(np.median(x[p])), float(np.median(y[p]))) for lot, p in parts.items()}
        for i in members:
            if i in wset:
                continue
            lot = lot_of(i)
            if lot not in parts:
                lot = min(lots, key=lambda k: (np.hypot(x[i] - centers[k][0], y[i] - centers[k][1]), k))
            parts[lot].append(i)
        return [sorted(p) for p in sorted(parts.values(), key=min)], outliers
    w = np.asarray(wrows)
    xs, ys = x[w], y[w]
    counts = np.zeros(len(w), dtype=np.int64)
    for start in range(0, len(w), 512):
        block = np.hypot(xs[start:start + 512, None] - xs[None, :], ys[start:start + 512, None] - ys[None, :])
        counts[start:start + 512] = (block <= SPREAD_SPLIT_M).sum(axis=1)
    anchor = int(np.argmax(counts))
    near = np.hypot(xs - xs[anchor], ys - ys[anchor]) <= SPREAD_SPLIT_M
    outliers = [int(w[j]) for j in range(len(w)) if not near[j]]
    gone = set(outliers)
    return [[i for i in members if i not in gone]], outliers


# ---------------------------------------------------------------------------------------------------------
# nomenclature docs
# ---------------------------------------------------------------------------------------------------------

@dataclass
class NomenclatureBuild:
    """Result of :func:`build_nomenclatura`."""

    docs: pd.DataFrame
    units: pd.DataFrame
    rows: pd.DataFrame          # one line per kept record, for :func:`official_queries`
    dropped: pd.DataFrame       # quarantined records with the reason
    buckets: dict
    stats: dict = field(default_factory=dict)
    #: ``source`` / ``kept`` / ``excluded`` sets of NPN[:21] lots: every selected-tier record, the records that
    #: became documents or units, and the incomplete_address / spread_outlier records (for the oracle ceiling).
    lot_sets: dict = field(default_factory=dict)


def _check_columns(nom: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in nom.columns]
    if missing:
        raise ValueError(f"nomenclature frame lacks required column(s): {', '.join(missing)}")


def _column(nom: pd.DataFrame, name: str, order: np.ndarray) -> list:
    if name in nom.columns:
        return nom[name].iloc[order].tolist()
    return [None] * len(order)


def build_nomenclatura(
    nom: pd.DataFrame,
    tiers: tuple[str, ...] = TIERS,
    log: Callable[[str], None] | None = None,
) -> NomenclatureBuild:
    """Base documents, units and accounting from cleaned WFS rows (see the module docstring).

    ``tiers`` are the ``ndidestado`` values that build documents (priority ``cert`` > ``pcer``). The document
    NPN is the nomenclature one: ``single`` when the winning rows agree, else the lowest NPN among rows
    without an interior (``base_row``), else the lowest overall (``lowest``).
    """
    _check_columns(nom)
    say = log or (lambda _msg: None)
    tiers = tuple(t for t in tiers if t in _RANK)

    gid_num = pd.to_numeric(nom["gid"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    order = np.argsort(gid_num, kind="stable")                   # NaN gids go last, ties keep input order
    n = len(order)
    gids = [(-1 if not np.isfinite(g) else int(g)) for g in gid_num[order]]
    estados = [_s(v).lower() for v in nom["ndidestado"].iloc[order].tolist()]
    fulls = [_squash(_s(v)) for v in nom["nddirecc"].iloc[order].tolist()]
    npns = [_s(v) for v in nom["ndnpn"].iloc[order].tolist()]
    lons = nom["lon"].iloc[order].tolist()
    lats = nom["lat"].iloc[order].tolist()
    interiors = [_s(v) for v in _column(nom, "ndinterior", order)]
    cons2s = [_s(v) for v in _column(nom, "nddircons2", order)]
    wfs_comuna = [_s(v) for v in _column(nom, "ndcomuna", order)]
    wfs_barrio = [_s(v) for v in _column(nom, "ndidbarrio", order)]
    wfs_manzana = [_s(v) for v in _column(nom, "ndmanzana", order)]
    zonas = [_s(v) for v in _column(nom, "ndidtipzon", order)]
    if "npn_valid" in nom.columns:
        flags = nom["npn_valid"].iloc[order].tolist()
        npn_valid = [bool(f) and _npn_ok(p) if f is not pd.NA else _npn_ok(p) for f, p in zip(flags, npns)]
    else:
        npn_valid = [_npn_ok(p) for p in npns]

    buckets = {name: 0 for name in BUCKETS}
    strip_methods = {"suffix": 0, "parser": 0, "none": 0, "none_unregistered_tail": 0}
    split_cache: dict = {}
    text_cache: dict = {}

    # ---- pass 1: classify every record, derive base text and key ----
    kept: list[int] = []                      # positions (in `order`) of records that build documents
    rec_key: dict[int, str] = {}
    rec_text: dict[int, tuple[str, bool]] = {}   # direccion, parse_ok
    rec_base: dict[int, str] = {}
    rec_interior: dict[int, str] = {}
    drop_rows: list[tuple[int, str, str, str]] = []
    source_lots: set = set()
    excluded_lots: set = set()
    for i in range(n):
        if estados[i] not in tiers:
            buckets["excluded_tier"] += 1
            continue
        if len(npns[i]) >= _LOT_LEN:
            source_lots.add(npns[i][:_LOT_LEN])
        if not fulls[i]:
            drop_rows.append((gids[i], npns[i], fulls[i], "empty_address"))
            buckets["dropped"] += 1
            continue
        if not (_finite(lons[i]) and _finite(lats[i])):
            drop_rows.append((gids[i], npns[i], fulls[i], "no_coordinates"))
            buckets["dropped"] += 1
            continue
        sk = (fulls[i], interiors[i])
        split = split_cache.get(sk)
        if split is None:
            split = split_cache[sk] = _split_interior(*sk)
        base, interior, method = split
        text = text_cache.get(base)
        if text is None:
            text = text_cache[base] = _doc_text(base)
        direccion, ok, key, complete = text
        if not key:
            drop_rows.append((gids[i], npns[i], fulls[i], "empty_address"))
            buckets["dropped"] += 1
            continue
        if not complete:
            drop_rows.append((gids[i], npns[i], fulls[i], "incomplete_address"))
            buckets["incomplete_address"] += 1
            if len(npns[i]) >= _LOT_LEN:
                excluded_lots.add(npns[i][:_LOT_LEN])
            continue
        strip_methods[method] += 1
        kept.append(i)
        rec_key[i], rec_text[i], rec_base[i], rec_interior[i] = key, (direccion, ok), base, interior
        if len(kept) % 100_000 == 0:
            say(f"  classified {len(kept)} records ({i + 1}/{n})")

    # ---- metric coordinates of the kept records (spread) ----
    px = np.full(n, np.nan)
    py = np.full(n, np.nan)
    if kept:
        mx, my = to_metric([float(lons[i]) for i in kept], [float(lats[i]) for i in kept])
        px[kept], py[kept] = mx, my

    def lot_of(i: int) -> str:
        return npns[i][:_LOT_LEN] if npn_valid[i] else ""

    # ---- group by key, split far groups ----
    groups: dict[str, list[int]] = {}
    for i in kept:
        groups.setdefault(rec_key[i], []).append(i)

    bucket_of: dict[int, str] = {}
    sub_groups: list[tuple[str, list[int]]] = []
    split_groups = 0
    for key, members in groups.items():
        wrank = min(_RANK[estados[i]] for i in members)
        wrows = [i for i in members if _RANK[estados[i]] == wrank]
        parts, outliers = _split_group(members, wrows, px, py, lot_of)
        if len(parts) > 1:
            split_groups += 1
        for i in outliers:
            bucket_of[i] = "spread_outlier"
            if len(npns[i]) >= _LOT_LEN:
                excluded_lots.add(npns[i][:_LOT_LEN])
            drop_rows.append((gids[i], npns[i], fulls[i], "spread_outlier"))
        for part in parts:
            sub_groups.append((key, part))

    disagreements = {"comuna": 0, "barrio": 0, "manzana": 0, "invalid_npn_fallback": 0}
    choice_hist: dict[str, int] = {}
    units_only = 0
    npn_valid_docs = 0
    doc_rows: list[dict] = []

    for gi, (key, members) in enumerate(sub_groups):
        win = min(_RANK[estados[i]] for i in members)
        wrows = [i for i in members if _RANK[estados[i]] == win]
        proper = [i for i in wrows if rec_text[i][1]]
        head = proper[0] if proper else wrows[0]
        direccion, parse_ok = rec_text[head]

        # record buckets
        first_base = True
        for i in members:
            if not rec_text[i][1]:
                bucket_of[i] = "fallback_doc"
            elif rec_interior[i]:
                bucket_of[i] = "unit"
            elif _RANK[estados[i]] == win and first_base:
                bucket_of[i] = "base_doc_new"
                first_base = False
            else:
                bucket_of[i] = "base_doc_merged"
        if all(rec_interior[i] for i in wrows):
            units_only += 1

        # NPN choice among the winning tier's rows (valid codes preferred)
        pool = [i for i in wrows if npn_valid[i]] or [i for i in wrows if npns[i]] or wrows
        distinct = sorted({npns[i] for i in pool if npns[i]})
        if len(distinct) <= 1:
            choice, chosen = "single", (distinct[0] if distinct else "")
        else:
            base_npns = sorted({npns[i] for i in pool if not rec_interior[i] and npns[i]})
            if base_npns:
                choice, chosen = "base_row", base_npns[0]
            else:
                choice, chosen = "lowest", distinct[0]
        choice_hist[choice] = choice_hist.get(choice, 0) + 1
        candidates = [i for i in pool if npns[i] == chosen] or pool
        row_i = next((i for i in candidates if not rec_interior[i]), candidates[0])

        # lots of ALL kept records resolving to this doc (winning tier, merged lower tiers and unit rows)
        lots_by_tier = [{npns[i][:_LOT_LEN] for i in members if npn_valid[i] and _RANK[estados[i]] == r}
                        for r in range(len(TIERS))]
        all_lots = set().union(*lots_by_tier)
        lots_cert, lots_pcer = lots_by_tier[_RANK["cert"]], lots_by_tier[_RANK["pcer"]]

        # territory from the chosen row: the NPN wins, the WFS fields are the fallback / audit
        if npn_valid[row_i]:
            comuna, barrio, barrio_code, manzana = chosen[9:11], chosen[11:13], chosen[9:13], chosen[:_MZ_LEN]
            if wfs_comuna[row_i] and wfs_comuna[row_i] != comuna:
                disagreements["comuna"] += 1
            if wfs_barrio[row_i] and wfs_barrio[row_i] != barrio_code:
                disagreements["barrio"] += 1
            if wfs_manzana[row_i] and wfs_manzana[row_i] != chosen[13:_MZ_LEN]:
                disagreements["manzana"] += 1
            npn_valid_docs += 1
        else:
            comuna, barrio_code = wfs_comuna[row_i], wfs_barrio[row_i]
            barrio = barrio_code[2:4] if len(barrio_code) == 4 else ""
            manzana = chosen[:_MZ_LEN] if len(chosen) >= _MZ_LEN else wfs_manzana[row_i]
            disagreements["invalid_npn_fallback"] += 1

        if len(wrows) == 1:
            lon_c, lat_c = float(lons[wrows[0]]), float(lats[wrows[0]])
        else:
            lon_c = float(np.median([float(lons[i]) for i in wrows]))
            lat_c = float(np.median([float(lats[i]) for i in wrows]))
        gid_row = gids[row_i]
        doc_rows.append({
            "_rank": win, "_pos": wrows[0], "_members": members, "direccion": direccion,
            "numero_predial_nacional": chosen or None, "manzana": manzana, "comuna": comuna, "barrio": barrio,
            "barrio_code": barrio_code, "centroid_lon": lon_c, "centroid_lat": lat_c,
            "n_predios": max(len(all_lots), 1), "manzana_nunique": max(len({p[:_MZ_LEN] for p in distinct}), 1),
            "objectid": (NOMENCLATURE_OBJECTID_OFFSET + gid_row) if gid_row >= 0 else 2 * NOMENCLATURE_OBJECTID_OFFSET + row_i,
            "fuente": FUENTE[TIERS[win]], "estado_nomenclatura": TIERS[win],
            "npn_catastro": None, "direccion_catastro": None, "npn_choice": choice, "n_units": max(len(distinct), 1),
            "source_objectid": gid_row, "parse_ok": bool(parse_ok), "zona": zonas[row_i] or None,
            "address_key": address_key(direccion), "doc_key": key, "spread_m": _spread(wrows, px, py),
            "n_lots_cert": len(lots_cert), "n_lots_pcer": len(lots_pcer),
            "lot_conflict": bool(lots_cert and lots_pcer and lots_cert != lots_pcer),
            "own_lon": float(lons[row_i]), "own_lat": float(lats[row_i]),
        })
        if (gi + 1) % 100_000 == 0:
            say(f"  built {gi + 1}/{len(sub_groups)} documents")

    # ---- documents in deterministic order: cert by gid, then pcer by gid ----
    doc_rows.sort(key=lambda d: (d["_rank"], d["_pos"]))
    doc_ord: dict[int, int] = {}
    for ordinal, d in enumerate(doc_rows):
        for i in d["_members"]:
            doc_ord[i] = ordinal
    docs = _nomenclature_frame(doc_rows)

    # ---- per-record outputs ----
    rows_out: list[tuple] = []
    unit_rows: list[tuple] = []
    for i in kept:
        bucket = bucket_of[i]
        buckets[bucket] += 1
        if bucket == "spread_outlier":
            continue
        cons2 = cons2s[i] or _punctuation_free(fulls[i])
        rows_out.append((gids[i], rec_key[i], doc_ord[i], bucket, fulls[i], cons2, rec_base[i], rec_interior[i], estados[i]))
        if rec_interior[i]:
            unit_rows.append((rec_key[i], rec_interior[i], npns[i] or None, float(lons[i]), float(lats[i]), gids[i],
                              fulls[i], estados[i], doc_ord[i]))
    rows = pd.DataFrame(rows_out, columns=["gid", "doc_key", "doc_ord", "bucket", "direccion_original", "nddircons2",
                                           "base_text", "interior", "estado"])
    rows["doc_ord"] = rows["doc_ord"].astype("int64")
    units = pd.DataFrame(unit_rows, columns=UNIT_COLUMNS)
    units["numero_predial_nacional"] = units["numero_predial_nacional"].astype("str")
    units["objectid"] = units["objectid"].astype("int64")
    units["doc_ord"] = units["doc_ord"].astype("int64")
    drop_rows.sort(key=lambda r: r[0])
    dropped = pd.DataFrame(drop_rows, columns=["gid", "ndnpn", "nddirecc", "reason"])
    dropped["gid"] = dropped["gid"].astype("int64")

    selected = n - buckets["excluded_tier"]
    spread = docs["spread_m"] if len(docs) else pd.Series([], dtype="float64")
    stats = {
        "input_rows": n, "selected_rows": selected, "docs": len(docs),
        "interior_strip_method": strip_methods, "npn_choice": dict(sorted(choice_hist.items())),
        "disagreements": disagreements, "npn_valid_docs": npn_valid_docs,
        "parse_failure_rows": buckets["fallback_doc"],
        "parse_failure_rate": (buckets["fallback_doc"] / selected) if selected else 0.0,
        "address_key_collisions": int(docs.loc[docs["address_key"].duplicated(keep=False), "address_key"].nunique()),
        "docs_from_units_only": units_only,
        "spread": {"split_groups": split_groups, "outlier_records": buckets["spread_outlier"],
                   "docs_over_200m": int((spread > 200).sum()), "docs_over_1000m": int((spread > SPREAD_SPLIT_M).sum())},
        "incomplete_address_records": buckets["incomplete_address"],
    }
    say(f"  nomenclature: {len(docs)} base documents from {selected} records ({buckets['dropped']} dropped)")
    kept_lots = {npns[i][:_LOT_LEN] for i in kept if bucket_of[i] != "spread_outlier" and len(npns[i]) >= _LOT_LEN}
    lot_sets = {"source": source_lots, "kept": kept_lots, "excluded": excluded_lots}
    return NomenclatureBuild(docs=docs, units=units, rows=rows, dropped=dropped, buckets=buckets, stats=stats,
                             lot_sets=lot_sets)


def build_nomenclatura_docs(
    nom: pd.DataFrame,
    tiers: tuple[str, ...] = TIERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(docs, units)`` of :func:`build_nomenclatura` (the rest of the build is in the full result)."""
    build = build_nomenclatura(nom, tiers=tiers)
    return build.docs, build.units


_NOM_FLOATS = ("centroid_lon", "centroid_lat", "own_lon", "own_lat", "spread_m")
_NOM_INTS = ("n_predios", "manzana_nunique", "objectid", "n_units", "source_objectid", "n_lots_cert", "n_lots_pcer")


def _nomenclature_frame(doc_rows: list[dict]) -> pd.DataFrame:
    """Typed frame of the nomenclature documents (still without ``doc_id`` / ``x_m`` / ``y_m``)."""
    cols = [c for c in COMPAT_COLUMNS if c not in ("x_m", "y_m", "doc_id")] + [
        c for c in EXTRA_COLUMNS] + ["own_lon", "own_lat"]
    data = {c: [d[c] for d in doc_rows] for c in cols}
    df = pd.DataFrame(data, columns=cols)
    for c in _STR_COMPAT + _STR_EXTRA:
        df[c] = _str_series(df[c].tolist()) if len(df) else pd.Series([], dtype="str")
    for c in _NOM_FLOATS:
        df[c] = df[c].astype("float64")
    for c in _NOM_INTS:
        df[c] = df[c].astype("int64")
    df["parse_ok"] = df["parse_ok"].astype(bool)
    df["lot_conflict"] = df["lot_conflict"].astype(bool)
    return df


# ---------------------------------------------------------------------------------------------------------
# cadastre side (read-only audit)
# ---------------------------------------------------------------------------------------------------------

def catastro_key_table(catastro_docs: pd.DataFrame) -> pd.DataFrame:
    """Copy of the cadastral docs with ``doc_key``, ``address_key`` and ``parse_ok`` columns (kept when present)."""
    out = catastro_docs.reset_index(drop=True).copy()
    texts = out["direccion"].tolist()
    cache: dict = {}
    keys, akeys, oks = [], [], []
    for text in texts:
        hit = cache.get(text)
        if hit is None:
            raw = _s(text)
            parsed = parse_address(raw) if raw else None
            hit = cache[text] = (doc_key(raw), address_key(raw), bool(parsed is not None and parsed.parse_ok))
        keys.append(hit[0])
        akeys.append(hit[1])
        oks.append(hit[2])
    if "doc_key" not in out.columns:
        out["doc_key"] = _str_series(keys)
    if "address_key" not in out.columns:
        out["address_key"] = _str_series(akeys)
    if "parse_ok" not in out.columns:
        out["parse_ok"] = np.asarray(oks, dtype=bool)
    return out


def _ensure_keyed(catastro_docs: pd.DataFrame) -> pd.DataFrame:
    if {"doc_key", "address_key", "parse_ok"} <= set(catastro_docs.columns):
        return catastro_docs.reset_index(drop=True)
    return catastro_key_table(catastro_docs)


def union_docs(nom_docs: pd.DataFrame, catastro_docs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The nomenclature docs with ``doc_id`` (their order) and metric coordinates.

    ``catastro_docs`` is optional READ-ONLY audit input: a cadastral doc at the key of a nomenclature doc
    contributes only its NPN (``npn_catastro``) and, when the text differs, its spelling
    (``direccion_catastro``). The cadastre never adds documents or changes any other field.
    """
    nom = nom_docs.reset_index(drop=True).copy()
    if catastro_docs is not None and len(catastro_docs) and len(nom):
        cat = _ensure_keyed(catastro_docs)
        first: dict = {}
        for pos, key in enumerate(cat["doc_key"].tolist()):
            first.setdefault(key, pos)
        cat_npn = cat["numero_predial_nacional"].tolist()
        cat_dir = cat["direccion"].tolist()
        npn_cat, dir_cat = [], []
        for key, text in zip(nom["doc_key"].tolist(), nom["direccion"].tolist()):
            pos = first.get(key)
            if pos is None:
                npn_cat.append(None)
                dir_cat.append(None)
            else:
                npn_cat.append(_s(cat_npn[pos]) or None)
                dir_cat.append(None if cat_dir[pos] == text else cat_dir[pos])
        nom["npn_catastro"] = _str_series(npn_cat, nom.index)
        nom["direccion_catastro"] = _str_series(dir_cat, nom.index)
    docs = nom.reset_index(drop=True)
    x, y = to_metric(docs["centroid_lon"].to_numpy(), docs["centroid_lat"].to_numpy()) if len(docs) else (
        np.zeros(0), np.zeros(0))
    docs["x_m"], docs["y_m"] = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    docs["doc_id"] = np.arange(len(docs), dtype=np.int64)
    for c in _STR_COMPAT + _STR_EXTRA:
        docs[c] = docs[c].astype("str")
    return docs[COMPAT_COLUMNS + EXTRA_COLUMNS]


def union_own_centroids(nom_docs: pd.DataFrame) -> pd.DataFrame:
    """Own-centroid sidecar, same row order as :func:`union_docs`: the point of each doc's chosen row."""
    nom = nom_docs.reset_index(drop=True)
    return pd.DataFrame({
        "objectid": nom["objectid"].astype("int64"),
        "numero_predial_nacional": nom["numero_predial_nacional"].astype("str"),
        "own_lon": nom["own_lon"].astype("float64"), "own_lat": nom["own_lat"].astype("float64"),
    }, columns=OWN_CENTROID_COLUMNS)


def assign_unit_doc_ids(units: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """Units with ``doc_id`` inserted after ``address_key_base``, which becomes the doc's ``address_key``.

    Each unit follows its record's document (``doc_ord``, the position in the nomenclature docs); without
    ``doc_ord`` the unit joins the first doc of its ``doc_key``.
    """
    doc_ids = docs["doc_id"].tolist()
    akeys = docs["address_key"].tolist()
    out = units.copy()
    if "doc_ord" in out.columns:
        ords = out["doc_ord"].astype("int64").tolist()
        present = [0 <= o < len(doc_ids) for o in ords]
        out = out[np.asarray(present, dtype=bool)].reset_index(drop=True)
        ords = [o for o, p in zip(ords, present) if p]
        out["address_key_base"] = _str_series([akeys[o] for o in ords])
        out.insert(1, "doc_id", np.asarray([int(doc_ids[o]) for o in ords], dtype="int64"))
        return out.drop(columns=["doc_ord"])
    ids: dict = {}
    by_key_akey: dict = {}
    for key, akey, doc_id in zip(docs["doc_key"].tolist(), akeys, doc_ids):
        if key not in ids:
            ids[key], by_key_akey[key] = int(doc_id), akey
    mapped = out["address_key_base"].map(ids)
    present = mapped.notna().to_numpy()
    out = out[present].reset_index(drop=True)
    out["address_key_base"] = _str_series([by_key_akey[k] for k in out["address_key_base"].tolist()])
    out.insert(1, "doc_id", mapped[present].astype("int64").to_numpy())
    return out


# ---------------------------------------------------------------------------------------------------------
# training queries, accounting
# ---------------------------------------------------------------------------------------------------------

def _canon_form(text: str) -> str:
    """Spelling-free form of a query text (letters glued/separated, punctuation and case ignored)."""
    return re.sub(r"[^0-9A-Z]+", " ", normalize_spacing(text).upper()).strip()


def _token_key(text: str) -> int:
    """64-bit digest of the encoder input of ``text``: same path as ``dataset.build_training_arrays``."""
    a, b = query_text_for(text)
    h = hashlib.blake2b(digest_size=8)
    h.update(encode_text(a).tobytes())
    h.update(b"|")
    h.update(encode_text(b).tobytes())
    return int.from_bytes(h.digest(), "little")


def _select(cands: list[tuple[str, int, str]], cap: int, rng: random.Random) -> list[int]:
    """Indices (ascending) of at most ``cap`` candidates: one as written and one spaced, then one per distinct form."""
    n = len(cands)
    if n <= cap:
        return list(range(n))
    chosen: list[int] = []
    for origin in (0, 1):
        pool = [i for i in range(n) if cands[i][1] == origin]
        if pool and len(chosen) < cap:
            chosen.append(rng.choice(pool))
    forms: dict[str, list[int]] = {}
    for i, (_, _, form) in enumerate(cands):
        forms.setdefault(form, []).append(i)
    taken = {cands[i][2] for i in chosen}
    order = list(forms)
    rng.shuffle(order)
    for form in order:
        if len(chosen) >= cap:
            break
        if form not in taken:
            chosen.append(rng.choice(forms[form]))
            taken.add(form)
    if len(chosen) < cap:
        have = set(chosen)
        rest = [i for i in range(n) if i not in have]
        rng.shuffle(rest)
        chosen.extend(rest[: cap - len(chosen)])
    return sorted(set(chosen))


def official_queries(rows: pd.DataFrame, docs: pd.DataFrame, cap: int = OFFICIAL_CAP_PER_DOC,
                     seed: int = OFFICIAL_SEED) -> list[tuple[str, int]]:
    """``(text, doc_id)`` training pairs from the kept nomenclature records, at most ``cap`` per document.

    Per record: the cleaned ``nddirecc`` (with interior), the punctuation-free ``nddircons2`` (derived when
    blank) and the interior-stripped text, each as written (letters glued) AND with
    :func:`parser.normalize_spacing` (letters separated). Deduplicated AFTER tokenisation (what the encoder
    sees, same path as ``dataset.build_training_arrays``) per document; a document over the cap keeps a
    seeded, deterministic sample that holds one as-written and one spaced form and prefers distinct
    spelling-free forms. Output is ordered by ``doc_id``.
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")
    doc_ids = [int(v) for v in (docs["doc_id"].tolist() if "doc_id" in docs.columns else range(len(docs)))]
    if "doc_ord" in rows.columns:
        refs = [int(o) for o in rows["doc_ord"].tolist()]
        resolved = [doc_ids[o] if 0 <= o < len(doc_ids) else None for o in refs]
    else:
        by_key: dict = {}
        for key, doc_id in zip(docs["doc_key"].tolist(), doc_ids):
            by_key.setdefault(key, doc_id)
        resolved = [by_key.get(k) for k in rows["doc_key"].tolist()]
    token_cache: dict[str, int] = {}
    per_doc: dict[int, list[tuple[str, int, str]]] = {}
    seen: dict[int, set] = {}
    for doc_id, orig, cons2, base in zip(resolved, rows["direccion_original"].tolist(), rows["nddircons2"].tolist(),
                                         rows["base_text"].tolist()):
        if doc_id is None:
            continue
        bucket, keys = per_doc.setdefault(doc_id, []), seen.setdefault(doc_id, set())
        for text in (_s(orig), _s(cons2) or _punctuation_free(_s(orig)), _s(base)):
            if not text:
                continue
            for origin, form in enumerate((text, normalize_spacing(text))):
                tk = token_cache.get(form)
                if tk is None:
                    if len(token_cache) > 500_000:
                        token_cache.clear()
                    tk = token_cache[form] = _token_key(form)
                if tk in keys:
                    continue
                keys.add(tk)
                bucket.append((form, origin, _canon_form(form)))
    out: list[tuple[str, int]] = []
    for doc_id in sorted(per_doc):
        cands = per_doc[doc_id]
        for i in _select(cands, cap, random.Random(f"{seed}:{doc_id}")):
            out.append((cands[i][0], doc_id))
    return out


def _count(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise AccountingError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def accounting(n_input: int, buckets: dict, quarantined: int = 0, excluded_lote_prov: int = 0) -> dict:
    """Validate that ``buckets`` partition the ``n_input`` records; raise :class:`AccountingError` if not.

    ``quarantined`` (rows the cleaning step removed before this build) and ``excluded_lote_prov`` (the part of
    them whose status was ``lote`` / ``prov``) are reported next to the buckets:
    ``accounting_total_with_quarantine = total + quarantined``.
    """
    unknown = sorted(set(buckets) - set(BUCKETS))
    if unknown:
        raise AccountingError(f"unknown accounting bucket(s): {', '.join(unknown)}")
    full = {name: buckets.get(name, 0) for name in BUCKETS}
    for name, count in full.items():
        _count(f"bucket {name}", count)
    quarantined = _count("quarantined", quarantined)
    excluded_lote_prov = _count("excluded_lote_prov", excluded_lote_prov)
    total = int(sum(full.values()))
    if total != int(n_input):
        raise AccountingError(f"buckets add up to {total} but the input has {int(n_input)} records: {full}")
    return {
        "input": int(n_input), "total": total, "buckets": {k: int(v) for k, v in full.items()},
        "quarantined": quarantined, "excluded_lote_prov": excluded_lote_prov,
        "accounting_total_with_quarantine": total + quarantined,
        "dropped_share": (full["dropped"] / n_input) if n_input else 0.0, "ok": True,
    }


# ---------------------------------------------------------------------------------------------------------
# comparison with the old cadastral index
# ---------------------------------------------------------------------------------------------------------

def _old_keys(old_docs: pd.DataFrame, wanted: set) -> dict:
    """``old doc_id -> doc_key`` for the wanted ids only (parsing every old doc is wasteful)."""
    old_ids = old_docs["doc_id"].tolist() if "doc_id" in old_docs.columns else list(range(len(old_docs)))
    texts = old_docs["direccion"].tolist()
    out: dict = {}
    for doc_id, text in zip(old_ids, texts):
        if int(doc_id) in wanted:
            out[int(doc_id)] = doc_key(text)
    return out


def _wanted_ids(values) -> set:
    out = set()
    for v in pd.to_numeric(pd.Series(list(values)), errors="coerce").tolist():
        if v == v:
            out.add(int(v))
    return out


def remap_train_pairs(pairs: pd.DataFrame, old_docs: pd.DataFrame, new_docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Move ``gt_doc_id`` from the old index to the new one; every other column is kept.

    Old id -> old ``direccion`` -> ``doc_key`` -> new ``doc_id``. When several new docs share the key (groups
    split by lot) the one holding the pair's ``gt_predial`` wins (exact NPN, then same lot), else the first.
    When the key is gone (or the old id is missing / out of range) the ``gt_predial`` is looked up among the
    new documents' NPN and ``npn_catastro``: a unique match remaps, none counts ``dropped_no_npn`` (also for a
    blank predial), several count ``dropped_ambiguous``. Returns ``(pairs, report)``.
    """
    new = _ensure_keyed(new_docs)
    new_ids = [int(v) for v in (new["doc_id"].tolist() if "doc_id" in new.columns else range(len(new)))]
    new_npn = [_s(v) for v in new["numero_predial_nacional"].tolist()]
    by_key: dict = {}
    for pos, key in enumerate(new["doc_key"].tolist()):
        by_key.setdefault(key, []).append(pos)
    by_npn: dict = {}
    for col in ("numero_predial_nacional", "npn_catastro"):
        if col in new.columns:
            for npn, doc_id in zip(new[col].tolist(), new_ids):
                npn = _s(npn)
                if npn:
                    by_npn.setdefault(npn, set()).add(int(doc_id))

    gt_old = pd.to_numeric(pairs["gt_doc_id"], errors="coerce").tolist() if "gt_doc_id" in pairs.columns else [np.nan] * len(pairs)
    predials = pairs["gt_predial"].tolist() if "gt_predial" in pairs.columns else [None] * len(pairs)
    old_key = _old_keys(old_docs, _wanted_ids(gt_old))

    report = {"input": int(len(pairs)), "kept": 0, "remapped_by_key": 0, "remapped_by_predial": 0,
              "dropped_no_npn": 0, "dropped_ambiguous": 0}
    keep_pos: list[int] = []
    new_gid: list[int] = []
    for pos, (old, predial) in enumerate(zip(gt_old, predials)):
        target = None
        predial = _s(predial)
        if old == old and int(old) in old_key:
            cands = by_key.get(old_key[int(old)])
            if cands:
                pick = None
                if predial:
                    pick = next((p for p in cands if new_npn[p] == predial), None)
                    if pick is None and len(predial) >= _LOT_LEN:
                        pick = next((p for p in cands if new_npn[p][:_LOT_LEN] == predial[:_LOT_LEN]), None)
                target = new_ids[cands[0] if pick is None else pick]
                report["remapped_by_key"] += 1
        if target is None:
            candidates = by_npn.get(predial) if predial else None
            if not candidates:
                report["dropped_no_npn"] += 1
                continue
            if len(candidates) > 1:
                report["dropped_ambiguous"] += 1
                continue
            target = next(iter(candidates))
            report["remapped_by_predial"] += 1
        keep_pos.append(pos)
        new_gid.append(int(target))
    out = pairs.iloc[keep_pos].reset_index(drop=True).copy()
    out["gt_doc_id"] = np.asarray(new_gid, dtype="int64")
    report["kept"] = int(len(out))
    return out, report


def pair_npn_agreement(pairs: pd.DataFrame, docs: pd.DataFrame) -> dict:
    """Pairs whose ``gt_doc_id`` doc has the same NPN (``npn_equal``) / same lot (``lot_equal``) as ``gt_predial``."""
    npns = [_s(v) for v in docs["numero_predial_nacional"].tolist()]
    exact = lot = 0
    ids = pd.to_numeric(pairs["gt_doc_id"], errors="coerce").tolist()
    for doc_id, predial in zip(ids, pairs["gt_predial"].tolist()):
        predial = _s(predial)
        if doc_id != doc_id or not predial or not 0 <= int(doc_id) < len(npns):
            continue
        got = npns[int(doc_id)]
        exact += got == predial
        lot += bool(_lot(got)) and _lot(got) == _lot(predial)
    return {"pairs": int(len(pairs)), "npn_equal": int(exact), "lot_equal": int(lot)}


_ORACLE_KEYS = ("rows", "prod_lot_matched", "new_lot_matched", "production_exact", "new_exact",
                "lot_absent_from_source", "rows_with_lot_in_source", "prod_lot_matched_in_source",
                "lot_lost_incomplete", "lot_not_chosen", "lot_lost_other")


def oracle_ceiling(gold_by_split: dict, old_docs: pd.DataFrame, new_docs: pd.DataFrame,
                   lot_sets: dict | None = None) -> dict:
    """Best achievable lot-level agreement of the new index on the gold evaluation rows, split by cause.

    For every ``gt_quality == 'gold'`` row: old ``gt_doc_id`` -> old ``direccion`` -> ``doc_key`` -> new docs
    (when the key is gone, the new docs holding the ``gt_predial`` lot). ``new_lot_matched`` counts the rows with
    at least one such doc whose NPN[:21] equals ``gt_predial[:21]``; ``prod_lot_matched`` the same for the old
    doc itself; ``production_exact`` / ``new_exact`` the full-NPN counterparts. A row whose lot is in no new doc
    is attributed with ``lot_sets`` (see :class:`NomenclatureBuild`): ``lot_absent_from_source`` (in no
    selected-tier record: the accepted coverage cost), ``lot_lost_incomplete`` (only in incomplete_address /
    spread_outlier records), ``lot_not_chosen`` (in kept records but not the lot of the doc NPN) or
    ``lot_lost_other``. Without ``lot_sets`` nothing counts as absent. ``rows_with_lot_in_source`` is
    ``rows - lot_absent_from_source`` and ``prod_lot_matched_in_source`` the production matches among those rows.
    Counts only.
    """
    new = _ensure_keyed(new_docs)
    new_npn = [_s(v) for v in new["numero_predial_nacional"].tolist()]
    by_key: dict = {}
    lots: dict = {}
    for pos, (key, npn) in enumerate(zip(new["doc_key"].tolist(), new_npn)):
        by_key.setdefault(key, []).append(pos)
        if _lot(npn):
            lots.setdefault(_lot(npn), []).append(pos)
    old_npn = [_s(v) for v in old_docs["numero_predial_nacional"].tolist()]
    old_ids = [int(v) for v in (old_docs["doc_id"].tolist() if "doc_id" in old_docs.columns else range(len(old_docs)))]
    old_pos = {doc_id: p for p, doc_id in enumerate(old_ids)}
    source = None if lot_sets is None else lot_sets.get("source", set())
    kept = set() if lot_sets is None else lot_sets.get("kept", set())
    excluded = set() if lot_sets is None else lot_sets.get("excluded", set())

    frames = {}
    for name, frame in gold_by_split.items():
        if "gt_quality" in frame.columns:
            frame = frame[frame["gt_quality"] == "gold"]
        frames[name] = frame
    wanted = set()
    for frame in frames.values():
        wanted |= _wanted_ids(frame["gt_doc_id"].tolist())
    keys = _old_keys(old_docs, wanted)

    result: dict = {}
    total = dict.fromkeys(_ORACLE_KEYS, 0)
    for name, frame in frames.items():
        c = dict.fromkeys(_ORACLE_KEYS, 0)
        for gid, predial in zip(pd.to_numeric(frame["gt_doc_id"], errors="coerce").tolist(), frame["gt_predial"].tolist()):
            predial = _s(predial)
            lot = _lot(predial)
            c["rows"] += 1
            absent = source is not None and lot not in source
            c["lot_absent_from_source"] += absent
            gid_ok = gid == gid and int(gid) in old_pos
            if gid_ok and predial:
                old_value = old_npn[old_pos[int(gid)]]
                c["production_exact"] += old_value == predial
                prod_hit = bool(_lot(old_value)) and _lot(old_value) == lot
                c["prod_lot_matched"] += prod_hit
                c["prod_lot_matched_in_source"] += prod_hit and not absent
            cands = by_key.get(keys.get(int(gid))) if gid_ok else None
            if not cands and lot:
                cands = lots.get(lot)
            if cands and predial:
                c["new_exact"] += any(new_npn[p] == predial for p in cands)
                c["new_lot_matched"] += bool(lot) and any(_lot(new_npn[p]) == lot for p in cands)
            if lot and lot not in lots and not absent:
                if lot in kept:
                    c["lot_not_chosen"] += 1
                elif lot in excluded:
                    c["lot_lost_incomplete"] += 1
                else:
                    c["lot_lost_other"] += 1
        c["rows_with_lot_in_source"] = c["rows"] - c["lot_absent_from_source"]
        result[name] = {k: int(v) for k, v in c.items()}
        for k in total:
            total[k] += c[k]
    result["total"] = {k: int(v) for k, v in total.items()}
    return result


def oracle_ceiling_gate(result: dict, max_drop_points: float = ORACLE_MAX_DROP_POINTS) -> dict:
    """Construction gate on :func:`oracle_ceiling`, evaluated on the ``total`` only.

    Over the rows whose lot is present in the source records, ``new_lot_matched`` may be at most
    ``max_drop_points`` percentage points below the production matches on the same rows. Lots absent from the
    source are the accepted coverage cost and never fail it. Also reports the headline and inherent losses.
    """
    total = result["total"]
    rows, in_source = total["rows"], total["rows_with_lot_in_source"]
    shares = {}
    for name, c in result.items():
        n = c["rows_with_lot_in_source"]
        if n:
            shares[name] = {"production_cond_pct": 100.0 * c["prod_lot_matched_in_source"] / n,
                            "new_cond_pct": 100.0 * c["new_lot_matched"] / n}
    limit = total["prod_lot_matched_in_source"] - in_source * max_drop_points / 100.0 - 1e-9
    ok = not in_source or total["new_lot_matched"] >= limit
    return {
        "ok": bool(ok), "max_drop_points": max_drop_points, "failed": [] if ok else ["total"], "shares": shares,
        "coverage_loss_headline_pp": (100.0 * (total["prod_lot_matched"] - total["new_lot_matched"]) / rows) if rows else 0.0,
        "coverage_loss_inherent_pp": (100.0 * total["lot_absent_from_source"] / rows) if rows else 0.0,
    }


def catastro_coverage(docs: pd.DataFrame, catastro_docs: pd.DataFrame) -> dict:
    """Cadastral docs with no nomenclature doc at their ``doc_key`` / with a lot present in no nomenclature doc."""
    cat = _ensure_keyed(catastro_docs)
    keys = set(docs["doc_key"].tolist())
    lots = {_lot(v) for v in docs["numero_predial_nacional"].tolist()} - {""}
    cat_keys = cat["doc_key"].tolist()
    cat_lots = [_lot(v) for v in cat["numero_predial_nacional"].tolist()]
    return {
        "cadastre_docs": int(len(cat)),
        "no_nomenclature_doc_by_key": int(sum(k not in keys for k in cat_keys)),
        "no_nomenclature_doc_by_lot": int(sum(l not in lots for l in cat_lots)),
    }
