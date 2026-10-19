"""Inference pipeline: free-form text -> canonical address + predio + confidence.

    rule-based canonicalization
      -> char-Transformer encoding on the GPU (two views: canonical + raw)
      -> top-k cosine search over the 330k cadastral embeddings (fp16 matmul)
      -> rerank with a fused score (model similarity + rapidfuzz + structure)
      -> best predio, canonical address, confidence, manzana, centroid, comuna
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rapidfuzz import fuzz

from .dataset import OWN_CENTROID_COLUMNS, own_centroid_table
from .geo import to_metric
from .model import AddressEncoder, encode_batch
from .parser import (
    _pre_normalize,
    canonical,
    extract_complement_units,
    is_truncated,
    match_key,
    parse_address,
    render_complement,
)
from .paths import DEFAULT_ARTIFACTS_DIR, OWN_CENTROIDS_FILENAME, model_path, shared_path
from .refine import refine_address
from .reliability import RELIABILITY_FILENAME, load_reliability

__all__ = [
    "AddressNormalizer",
    "FUSE_WEIGHTS",
    "CONFIDENCE_THRESHOLD",
    "GEO_TAU_M",
    "load_tuning",
    "load_own_centroids",
    "SOFT_RULE_NAMES",
    "DECISION_KEYS",
    "normalize_soft_rules",
    "validate_decision",
]

#: Fallback weights of the reranking score, used when ``artifacts/tuning.json`` is
#: absent. The shipped values are the ones ``scripts/tune_threshold.py`` selected
#: on the **held-out cadastral validation split** (documents never seen as
#: positives during training); they were never fitted on the Excel evaluation
#: rows. ``AddressNormalizer`` reloads whatever that file contains, so the code
#: and the artifact cannot drift apart.
FUSE_WEIGHTS = {"sim": 0.70, "fuzz": 0.10, "struct": 0.10, "geo": 0.10}

#: Length scale (metres) of the geographic consistency term. A candidate sitting
#: `GEO_TAU_M` metres away from the coordinate predicted by the auxiliary head
#: keeps 1/e of its geographic credit. Fixed a priori, not tuned.
GEO_TAU_M = 600.0

#: Fallback high-confidence cutoff on the fused score; see ``load_tuning``.
#: Selected by maximizing F1 of "matched => correct document" on the held-out
#: cadastral split mixed with synthetic non-address negatives; stable across
#: assumed junk rates from 5 % to 40 %.
CONFIDENCE_THRESHOLD = 0.73

#: Name of the tuning artifact produced by ``scripts/tune_threshold.py``.
TUNING_FILENAME = "tuning.json"


#: Rule classes the decision layer may downgrade from hard to soft (see
#: ``service._tagged_violations``). A soft violation keeps the row OK but only vouches for the
#: manzana. The empty selection is today's behaviour.
SOFT_RULE_NAMES = (
    "letra_via",             # via letter differs on both sides (one-sided is already soft)
    "letra_cruce_cardinal",  # cross letter is a lone cardinal (N/O/W) on one side only
    "letra_cruce_una_cara",  # cross letter present on exactly one side
    "letra_cruce",           # any cross-letter mismatch
    "bis_via",
    "bis_cruce",
    "cuadrante_compl",       # quadrant word in the complement missing from the cadastre
)

#: Keys the optional ``chosen.decision`` block of ``tuning.json`` may carry.
DECISION_KEYS = ("soft_rules", "max_soft", "gate_fallback")


def normalize_soft_rules(value) -> frozenset:
    """Validate a ``soft_rules`` selection: ``None``/empty -> empty set; unknown name -> ``ValueError``.

    Accepts a comma-separated string or any iterable of names.
    """
    if value is None:
        return frozenset()
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        try:
            names = list(value)
        except TypeError as exc:
            raise ValueError(f"soft_rules must be a string or an iterable of names, got {value!r}") from exc
    for name in names:
        if not isinstance(name, str) or name not in SOFT_RULE_NAMES:
            raise ValueError(f"unknown soft rule {name!r}; valid names: {', '.join(SOFT_RULE_NAMES)}")
    return frozenset(names)


def validate_max_soft(value) -> int:
    """``max_soft`` must be a non-negative ``int`` (``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"max_soft must be an int >= 0, got {value!r}")
    return value


def validate_decision(decision) -> dict:
    """Validate the ``chosen.decision`` block of ``tuning.json`` and return a plain dict copy."""
    if not isinstance(decision, dict):
        raise ValueError(f"tuning decision block must be an object, got {type(decision).__name__}")
    unknown = sorted(set(decision) - set(DECISION_KEYS))
    if unknown:
        raise ValueError(f"unknown decision keys {unknown}; valid keys: {', '.join(DECISION_KEYS)}")
    out: dict = {}
    if "soft_rules" in decision:
        normalize_soft_rules(decision["soft_rules"])
        out["soft_rules"] = list(decision["soft_rules"]) if not isinstance(decision["soft_rules"], str) else [
            part.strip() for part in decision["soft_rules"].split(",") if part.strip()]
    if "max_soft" in decision:
        out["max_soft"] = validate_max_soft(decision["max_soft"])
    if "gate_fallback" in decision:
        if not isinstance(decision["gate_fallback"], bool):
            raise ValueError(f"gate_fallback must be a bool, got {decision['gate_fallback']!r}")
        out["gate_fallback"] = decision["gate_fallback"]
    return out


def load_tuning(artifacts_dir: str) -> dict:
    """Read the tuning artifact, falling back to the module defaults.

    Returns ``{"weights", "threshold", "thresholds", "decision", "source"}``. ``decision`` is the
    optional ``chosen.decision`` block (defaults of ``soft_rules`` / ``max_soft`` / ``gate_fallback``
    for the strict decision layer; ``{}`` reproduces the historical behaviour). ``thresholds``
    maps an ablation variant name to its own cutoff, because the variants score
    candidates on different scales (raw cosine vs fused score) and a single
    cutoff would make their coverage numbers incomparable.
    """
    path = os.path.join(artifacts_dir, TUNING_FILENAME)
    out = {
        "weights": dict(FUSE_WEIGHTS),
        "threshold": CONFIDENCE_THRESHOLD,
        "thresholds": {},
        "decision": {},
        "source": "module defaults (artifacts/tuning.json not found)",
    }
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    chosen = data.get("chosen", {})
    if chosen.get("weights"):
        out["weights"] = {k: float(v) for k, v in chosen["weights"].items()}
    if chosen.get("threshold") is not None:
        out["threshold"] = float(chosen["threshold"])
    out["thresholds"] = {k: float(v) for k, v in (chosen.get("thresholds") or {}).items()}
    if chosen.get("decision") is not None:
        out["decision"] = validate_decision(chosen["decision"])
    out["source"] = path
    return out

_STRUCT_WEIGHTS = {
    "via_type": 1.0,
    "via_number": 2.0,
    "via_letters": 1.0,
    "via_quadrant": 0.75,
    "cross_number": 2.0,
    "cross_letters": 1.0,
    "cross_quadrant": 0.75,
    "cross_type": 0.5,
    "plate": 1.5,
    "via_bis": 0.75,
    "cross_bis": 0.75,
}

#: Fields whose "no value" sentinel is ``False`` rather than ``None``. Two plain
#: addresses (``via_bis=cross_bis=False`` on both sides) must not get a bigger
#: denominator than a pair that never mentions these fields at all, or a
#: structural mismatch elsewhere would be diluted by a free, meaningless match.
_BOOLEAN_STRUCT_FIELDS = frozenset({"via_bis", "cross_bis"})


def load_manzana_nunique(docs: pd.DataFrame) -> np.ndarray:
    """Distinct manzanas behind each document's address.

    Older ``catastro_docs.parquet`` files predate the column; they fall back to
    1 for every doc (the address is assumed to sit in a single manzana).
    """
    if "manzana_nunique" in docs.columns:
        return docs["manzana_nunique"].fillna(1).to_numpy(dtype="int64")
    return np.ones(len(docs), dtype="int64")


def _read_own_sidecar(docs: pd.DataFrame, artifacts_dir: str | None):
    """The own-centroid sidecar as ``(lon, lat)`` arrays, or ``None`` unless it describes exactly these docs.

    Row count, OBJECTID and NPN must equal the documents' (same order): a sidecar built for another
    ``catastro_docs.parquet`` is never trusted. Never raises.
    """
    try:
        path = shared_path(OWN_CENTROIDS_FILENAME, artifacts_dir)
        if not os.path.exists(path):
            return None
        table = pd.read_parquet(path)
        if not set(OWN_CENTROID_COLUMNS).issubset(table.columns) or len(table) != len(docs):
            return None
        if not table["objectid"].reset_index(drop=True).equals(docs["objectid"].reset_index(drop=True)):
            return None
        theirs = table["numero_predial_nacional"].astype(object).reset_index(drop=True)
        if not theirs.equals(docs["numero_predial_nacional"].astype(object).reset_index(drop=True)):
            return None
        lon, lat = table["own_lon"].to_numpy(dtype="float64"), table["own_lat"].to_numpy(dtype="float64")
    except Exception:  # noqa: BLE001 - a broken optional artifact must not stop the service
        return None
    return (lon, lat) if np.isfinite(lon).any() else None


def load_own_centroids(docs: pd.DataFrame, artifacts_dir: str | None = None):
    """Centroid of the *returned* parcel of every document, as ``(lon, lat)`` arrays.

    ``catastro_docs.parquet`` keeps one row per address: its NPN is the first parcel's
    (lowest OBJECTID) but its ``centroid_*`` is the median over all the parcels that
    share the address, up to hundreds of metres away from the parcel reported. The real
    centroid of that first parcel comes, in order of preference, from

    1. the sidecar ``catastro_own_centroids.parquet`` (built by ``scripts/build_own_centroids.py``,
       shipped in the deployed image, validated against the documents), else
    2. ``catastro.parquet`` (three columns only; the geometries are not loaded), validated
       against the NPN of every document (:func:`cali_address.dataset.own_centroid_table`).

    A document whose parcel has no centroid, or whose NPN differs from the cadastre's, gets NaN.
    ``None`` when nothing usable exists (no ``objectid`` column, no / malformed / foreign file,
    empty docs): callers then keep the address centroid. Never raises.
    """
    if "objectid" not in docs.columns or "numero_predial_nacional" not in docs.columns or len(docs) == 0:
        return None
    from_sidecar = _read_own_sidecar(docs, artifacts_dir)
    if from_sidecar is not None:
        return from_sidecar
    try:
        path = shared_path("catastro.parquet", artifacts_dir)
        if not os.path.exists(path):
            return None
        cat = pd.read_parquet(path, columns=["OBJECTID", "numero_predial_nacional", "centroid_lon", "centroid_lat"])
        table = own_centroid_table(docs, cat)
    except Exception:  # noqa: BLE001 - a broken optional artifact must not stop the service
        return None
    lon, lat = table["own_lon"].to_numpy(), table["own_lat"].to_numpy()
    return (lon, lat) if np.isfinite(lon).any() else None


def _structural_agreement(a, b) -> float:
    """Weighted share of structural fields that agree between two parses."""
    total = 0.0
    got = 0.0
    for field, w in _STRUCT_WEIGHTS.items():
        va, vb = getattr(a, field), getattr(b, field)
        if field in _BOOLEAN_STRUCT_FIELDS:
            if va is False and vb is False:
                continue
        elif va is None and vb is None:
            continue
        total += w
        if va == vb:
            got += w
        elif field == "plate" and va is not None and vb is not None:
            # plates are often mistyped by one digit; give partial credit
            if str(va).lstrip("0") == str(vb).lstrip("0"):
                got += w
            elif abs(int(va) - int(vb)) <= 2:
                got += 0.4 * w
    if total == 0:
        return 0.0
    return got / total


#: A trailing cardinal LETTER the parser reads as a street letter (``CL 6N``) is the
#: same thing as the spelled-out quadrant the cadastre carries (``CL 6 NORTE``), but only
#: within the SAME position. Same table as ``service._QUADRANT_LETTER_EQUIVALENTS``
#: (``E``/``S`` are real street letters).
_CARDINAL_LETTER_QUADRANT = {"N": "NORTE", "O": "OESTE", "W": "OESTE"}

#: Complement kinds that name a different site, not a unit inside one building. The same
#: set the matcher treats as distinguishing at ``level="structure"``: two addresses that
#: state different values for one of them are different places. Units such as AP are not
#: listed: the input's own units always win, and the cadastral ones are never emitted.
_SITE_UNIT_KINDS = frozenset({"TO", "BLQ", "INT", "ET", "AGR", "MZ", "SEC", "CA", "LT"})

_QUADRANT_WORDS = ("NORTE", "SUR", "OESTE", "ESTE")
#: A plate letter is one letter (optionally followed by digits, ``A4``). Separated two-letter
#: tokens (``LA``, ``PI``, ``CL``, ``PH`` ...) are words or unit codes and belong to the
#: complement; two letters only count when glued to the plate digits (``66AT``). ``Y`` is the
#: conjunction, never a plate letter.
_TAIL_LETTER_RE = re.compile(r"^[A-Z]\d{0,2}$")

#: Single letters that, written apart and followed by a number, mark a unit (``P 1`` piso, ``T 9``, ``G 2``
#: garaje) and never a plate letter. ``66 A`` / ``66C`` / ``66 BIS`` remain plate letters.
_UNIT_MARKER_LETTERS = frozenset({"P", "T", "G"})


def _num_key(value) -> str:
    """Comparison key of a numeric field: no leading zeros, ``None``/blank -> ``""``."""
    if value is None:
        return ""
    text = str(value).strip().upper()
    return (text.lstrip("0") or "0") if text.isdigit() else text


def _letters_and_quadrant(letters, quadrant, cardinal: bool = True) -> tuple[str, str | None]:
    """Street letters without a trailing cardinal letter, plus the quadrant it stands for.

    ``cardinal=False`` keeps the trailing letter as a plain street letter (an ``N`` that is really an ``Ñ``).
    """
    text = "" if letters is None else str(letters).strip().upper()
    if cardinal and quadrant is None and text:
        parts = text.split()
        if parts[-1] in _CARDINAL_LETTER_QUADRANT:
            return " ".join(parts[:-1]), _CARDINAL_LETTER_QUADRANT[parts[-1]]
    return text, quadrant


def _is_tail_token(tok: str, glued: bool = False) -> bool:
    if tok == "BIS" or tok in _QUADRANT_WORDS or tok in _CARDINAL_LETTER_QUADRANT:
        return True
    if tok == "Y":
        return False
    return bool(_TAIL_LETTER_RE.match(tok)) or (glued and tok.isalpha() and len(tok) <= 3)


def _glued_to_plate(parsed, tok: str) -> bool:
    """True when ``tok`` is written right after digits in the input (``66AT``, ``08CA``)."""
    return re.search(r"\d" + re.escape(tok) + r"(?![A-Z0-9])", str(parsed.raw).upper()) is not None


def _plate_tail(parsed, cadastral: bool = False) -> tuple[tuple[str, ...], str, bool, str | None]:
    """What the address carries right after the plate (``66 C``, ``66 BIS``, ``20 NORTE``).

    The parser keeps it at the head of the complement text. Returns ``(tokens, letters,
    bis, quadrant)``: the normalized tail tokens in order (a lone N / O / W is the
    spelled quadrant), the letter run, the BIS flag and the trailing quadrant. The tail ends
    at the first token that is not plate-like or where the building units start.

    ``cadastral=True`` reads a cadastre string, where what follows the plate is a unit code
    (``69 1 C``, ``79 7 G``, ``32 P 1``), not plate letters: there is no cross-segment leftover to
    skip, and a letter followed by a number is a unit. Only letters glued to the plate digits
    (``66C``) or a lone letter right after the plate count. User input keeps the lenient reading.
    """
    tokens = (parsed.complement_text or "").upper().split()
    units = extract_complement_units(" ".join(tokens))
    start = len(tokens)
    if units:
        start = max(i for i in range(len(tokens)) if extract_complement_units(" ".join(tokens[i:])) == units)
    head = tokens[:start]
    offset = 0
    if head and not _is_tail_token(head[0], _glued_to_plate(parsed, head[0])):
        # leftovers of the cross segment sit in front of the plate tail (``4 F T``): read the
        # run of plate-like tokens at the end instead. In a cadastre string the run only counts
        # when it holds a letter glued to the plate digits (``4 F T`` with a glued T does;
        # ``69 1 C`` is a unit code).
        k = len(head)
        while k and _is_tail_token(head[k - 1], _glued_to_plate(parsed, head[k - 1])):
            k -= 1
        if cadastral and not any(_glued_to_plate(parsed, t) for t in head[k:]):
            k = len(head)
        offset = k
        head = head[k:]
    out: list[str] = []
    letters: list[str] = []
    bis, quadrant = False, None
    for idx, tok in enumerate(head):
        following = tokens[offset + idx + 1] if offset + idx + 1 < len(tokens) else ""
        if (
            (cadastral or tok in _UNIT_MARKER_LETTERS) and not _glued_to_plate(parsed, tok)
            and following.isdigit() and tok not in _QUADRANT_WORDS
        ):
            break  # a letter followed by a number is a unit marker (``P 1``, ``7 G``), not a plate letter
        if tok == "BIS":
            bis = True
            out.append(tok)
        elif tok in _QUADRANT_WORDS:
            quadrant = tok
            out.append(tok)
        elif tok in _CARDINAL_LETTER_QUADRANT and quadrant is None and not letters:
            quadrant = _CARDINAL_LETTER_QUADRANT[tok]
            out.append(quadrant)
        elif _is_tail_token(tok, _glued_to_plate(parsed, tok)):
            letters.append(tok)
            out.append(tok)
        else:
            break
    return tuple(out), " ".join(letters), bis, quadrant


def base_mismatches(rule, cadastral, mode: str = "legacy", *, plate_tolerance: int = 0,
                    soft_rules=frozenset()):
    """Fields of the address base where two parses disagree (empty list: same base).

    The base is ``<via> # <cross> - <plate>`` plus everything the address says about its
    own identity: street letters / numeric suffixes / BIS, what follows the plate (letters,
    BIS, quadrant) and site-identifying units (tower, block, house, lot ...). Quadrants are
    compared PER POSITION (via, cross, trailing): a quadrant on the via and one on the cross
    street are different places. A cardinal letter equals the spelled quadrant of the same
    position. A plate missing on one side is a mismatch; building units such as AP are not
    compared. Returns ``["unparsed"]`` when either side is ``None`` or did not parse.

    ``mode="position"`` is the production guard of the position tiers: it returns
    ``(hard, caps)`` instead of the list (see :func:`_position_guard`). The default ``"legacy"``
    mode is the research-path behaviour, unchanged.
    """
    if mode not in ("legacy", "position"):
        raise ValueError(f"unknown base_mismatches mode: {mode!r}")
    if mode == "position":
        try:
            return _position_guard(rule, cadastral, plate_tolerance, frozenset(soft_rules))
        except (AttributeError, TypeError, ValueError):
            return ["unparsed"], {}
    try:
        return _base_mismatches(rule, cadastral)
    except (AttributeError, TypeError, ValueError):
        return ["unparsed"]


#: Plate differences up to this many units (same street pair, both numeric) are a ``tramo``: the
#: position is interpolated between the neighbouring parcels instead of copying the neighbour's.
PLATE_TRAMO_WINDOW = 2

#: Guard cap fields that mean "a quadrant / street letter present on ONE side only": the matched parcel may
#: sit on a different street, so the service never keeps its predial and checks the rest of the candidates.
ONE_SIDED_FIELDS = frozenset({"via_quadrant", "cross_quadrant", "plate_quadrant", "via_letters", "cross_letters"})

#: Position tiers from most to least precise.
POSITION_TIER_ORDER = ("predio", "lote", "tramo", "manzana", "via", "none")

#: Soft rules (``SOFT_RULE_NAMES``) that downgrade a hard guard finding to a ``manzana`` cap.
_GUARD_SOFT_RULES = {
    "via_letters": ("letra_via",),
    "cross_letters": ("letra_cruce",),
    "via_bis": ("bis_via",),
    "cross_bis": ("bis_cruce",),
}


def _blank(value) -> bool:
    return value is None or str(value).strip() == ""


def _position_guard(rule, cadastral, plate_tolerance: int, soft_rules: frozenset):
    """``(hard, caps)`` of a rule parse against the cadastral parse of the chosen candidate.

    ``hard`` lists the fields that reject the pair; ``caps`` maps a field to the best position
    tier the pair may still carry: ``lote`` (a site unit both sides state differs: the lot is right,
    the unit is not), ``tramo`` (plate letters / BIS / a plate within :data:`PLATE_TRAMO_WINDOW`: the street
    segment is right, the door is not) and ``manzana`` (a street letter or quadrant on one side
    only). Quadrants are compared per POSITION (via, cross, trailing): a quadrant that moved to
    another position is never equivalent. A cardinal letter equals the spelled quadrant only on
    the same segment. A blank on one side of ``cross_type`` / ``cross_number`` / ``plate`` is
    skipped (the precision level of the query already says it was not confirmed).
    """
    if rule is None or cadastral is None or not rule.parse_ok or not cadastral.parse_ok:
        return ["unparsed"], {}
    hard: list[str] = []
    caps: dict[str, str] = {}
    if is_truncated(rule):
        # the parse lost its cross / plate to a comma: the truncated base is never a verified address
        caps["truncated_parse"] = "via"

    def downgrade(field: str) -> bool:
        return any(name in soft_rules for name in _GUARD_SOFT_RULES.get(field, ()))

    for field in ("via_type", "cross_type", "cross_number"):
        rv, cv = getattr(rule, field), getattr(cadastral, field)
        if not _blank(rv) and not _blank(cv) and _num_key(rv) != _num_key(cv):
            hard.append(field)
    if _blank(rule.cross_number) or _blank(cadastral.cross_number):
        # no cross street on either side: the match rests on the via and the plate alone, so there is no
        # parcel to vouch for, only the street (tier via)
        caps["cross_number"] = "via"
    for field in ("via_number", "via_suffix", "via_suffix_letters", "cross_suffix", "cross_suffix_letters"):
        if _num_key(getattr(rule, field)) != _num_key(getattr(cadastral, field)):
            hard.append(field)
    for field in ("via_bis", "cross_bis"):
        if bool(getattr(rule, field)) != bool(getattr(cadastral, field)):
            if downgrade(field):
                caps[field] = "manzana"
            else:
                hard.append(field)

    # plate number: the tramo window, never a verified door
    rp, cp = rule.plate, cadastral.plate
    if not _blank(rp) and not _blank(cp) and _num_key(rp) != _num_key(cp):
        try:
            delta = abs(int(_num_key(rp)) - int(_num_key(cp)))
        except ValueError:
            delta = None
        if delta is None or delta > max(PLATE_TRAMO_WINDOW, plate_tolerance):
            hard.append("plate")
        elif delta > plate_tolerance:
            caps["plate"] = "tramo"
    elif not _blank(rp) and _blank(cp):
        hard.append("plate")  # the query names a door and the cadastral parcel has none: nothing to verify

    # street letters and per-position quadrants
    positions: dict[str, tuple[str | None, str | None]] = {}
    # an ``Ñ`` the parser folded into ``N`` is not the cardinal letter (it would equal NORTE)
    cardinal = "\u00d1" not in str(getattr(rule, "raw", "") or "").upper()
    for side in ("via", "cross"):
        r_letters, r_quadrant = _letters_and_quadrant(
            getattr(rule, f"{side}_letters"), getattr(rule, f"{side}_quadrant"), cardinal)
        c_letters, c_quadrant = _letters_and_quadrant(
            getattr(cadastral, f"{side}_letters"), getattr(cadastral, f"{side}_quadrant"))
        if r_letters != c_letters:
            field = f"{side}_letters"
            if r_letters and c_letters and not downgrade(field):
                hard.append(field)
            else:
                caps[field] = "manzana"
        positions[side] = (r_quadrant, c_quadrant)
    _, r_pl, r_bis, r_trail = _plate_tail(rule)
    _, c_pl, c_bis, c_trail = _plate_tail(cadastral, cadastral=True)
    positions["plate"] = (r_trail, c_trail)
    if r_pl != c_pl:
        caps["plate_letters"] = "tramo"
    if r_bis != c_bis:
        caps["plate_bis"] = "tramo"
    only_rule: dict[str, str] = {}
    only_cad: dict[str, str] = {}
    for pos, (rq, cq) in positions.items():
        if rq == cq:
            continue
        if rq is not None and cq is not None:
            hard.append(f"{pos}_quadrant")
        elif rq is not None:
            only_rule[pos] = rq
        else:
            only_cad[pos] = cq
    moved = set(only_rule.values()) & set(only_cad.values())
    for pos, quadrant in [*only_rule.items(), *only_cad.items()]:
        if quadrant in moved:
            if "quadrant_moved" not in hard:
                hard.append("quadrant_moved")
        else:
            caps[f"{pos}_quadrant"] = "manzana"

    # site units both sides state and that differ: the lot is right, the unit is not. A unit only the
    # cadastre carries is not a conflict (dev set: its predial was right for 72 of 73 gold rows), and
    # one only the input carries is a complement the cadastral base never had.
    r_units: dict[str, set[str]] = {}
    c_units: dict[str, set[str]] = {}
    for target, parsed in ((r_units, rule), (c_units, cadastral)):
        for kind, value in extract_complement_units(parsed.complement_text):
            if kind in _SITE_UNIT_KINDS:
                target.setdefault(kind, set()).add(value)
    for kind in sorted(set(c_units) & set(r_units)):
        if r_units[kind] != c_units[kind]:
            caps[f"unit_{kind}"] = "lote"
    return hard, caps


def _base_mismatches(rule, cadastral) -> list[str]:
    if rule is None or cadastral is None or not rule.parse_ok or not cadastral.parse_ok:
        return ["unparsed"]
    if is_truncated(rule):
        # never verifiable, whatever the candidate (also blocks every lower-ranked candidate in ``_row``)
        return ["truncated_parse"]
    out: list[str] = []
    for field in (
        "via_type", "via_number", "via_suffix", "via_suffix_letters",
        "cross_type", "cross_number", "cross_suffix", "cross_suffix_letters", "plate",
    ):
        if _num_key(getattr(rule, field)) != _num_key(getattr(cadastral, field)):
            out.append(field)
    for side in ("via", "cross"):
        r_letters, r_quadrant = _letters_and_quadrant(
            getattr(rule, f"{side}_letters"), getattr(rule, f"{side}_quadrant"))
        c_letters, c_quadrant = _letters_and_quadrant(
            getattr(cadastral, f"{side}_letters"), getattr(cadastral, f"{side}_quadrant"))
        if r_letters != c_letters:
            out.append(f"{side}_letters")
        if r_quadrant != c_quadrant:
            out.append(f"{side}_quadrant")
    for field in ("via_bis", "cross_bis"):
        if bool(getattr(rule, field)) != bool(getattr(cadastral, field)):
            out.append(field)
    _, r_letters, r_bis, r_quadrant = _plate_tail(rule)
    _, c_letters, c_bis, c_quadrant = _plate_tail(cadastral, cadastral=True)
    if r_letters != c_letters:
        out.append("plate_letters")
    if r_bis != c_bis:
        out.append("plate_bis")
    if r_quadrant != c_quadrant:
        out.append("plate_quadrant")
    r_units: dict[str, set[str]] = {}
    c_units: dict[str, set[str]] = {}
    for target, parsed in ((r_units, rule), (c_units, cadastral)):
        for kind, value in extract_complement_units(parsed.complement_text):
            if kind in _SITE_UNIT_KINDS:
                target.setdefault(kind, set()).add(value)
    for kind in sorted(set(r_units) & set(c_units)):
        if r_units[kind] != c_units[kind]:
            out.append(f"unit_{kind}")
    return out


@dataclass
class _Doc:
    direccion: str
    predial: str
    manzana: str
    comuna: str
    barrio: str
    lat: float
    lon: float
    n_predios: int


class AddressNormalizer:
    """Loads the cached model + embeddings and normalizes addresses."""

    def __init__(
        self,
        artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
        device: str | None = None,
        docs: pd.DataFrame | None = None,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tuning = load_tuning(artifacts_dir)
        self.weights = dict(tuning["weights"])
        self.threshold = float(tuning["threshold"])
        self.variant_thresholds = dict(tuning["thresholds"])
        #: Decision-layer defaults carried by the artifact (see ``load_tuning``).
        self.decision = dict(tuning["decision"])
        self.tuning_source = tuning["source"]
        #: Calibrated reliability model, or None (columns stay None) if the artifact is absent.
        self.reliability = load_reliability(os.path.join(artifacts_dir, RELIABILITY_FILENAME))
        ckpt = torch.load(model_path("model.pt", artifacts_dir), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.max_len = int(cfg.get("max_len", 40))
        self.model = AddressEncoder(
            d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
            d_out=cfg["d_out"], dropout=0.0, n_comuna=ckpt["n_comuna"],
            n_barrio=ckpt["n_barrio"], max_len=self.max_len,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        self.xy_mean = np.asarray(ckpt["xy_mean"], dtype="float64")
        self.xy_std = np.asarray(ckpt["xy_std"], dtype="float64")

        emb = torch.load(model_path("catastro_emb.pt", artifacts_dir), map_location="cpu")
        self.doc_emb = emb.to(self.device)
        if docs is None:
            # An experiment dir may omit the model-independent cadastral index.
            docs = pd.read_parquet(shared_path("catastro_docs.parquet", artifacts_dir))
        self.docs = docs.reset_index(drop=True)
        assert len(self.docs) == self.doc_emb.shape[0], "docs / embedding mismatch"
        self._direccion = self.docs["direccion"].to_numpy()
        self._predial = self.docs["numero_predial_nacional"].to_numpy()
        self._manzana = self.docs["manzana"].to_numpy()
        self._comuna = self.docs["comuna"].to_numpy()
        self._barrio = self.docs["barrio_code"].to_numpy()
        self._lat = self.docs["centroid_lat"].to_numpy()
        self._lon = self.docs["centroid_lon"].to_numpy()
        #: Centroid of the returned parcel (the address centroid above is a median over
        #: every parcel sharing the address); ``None`` when the cadastre is unavailable.
        own = load_own_centroids(self.docs, artifacts_dir)
        self._own_lon, self._own_lat = own if own is not None else (None, None)
        self._n_predios = self.docs["n_predios"].to_numpy()
        self._manzana_nunique = load_manzana_nunique(self.docs)
        if {"x_m", "y_m"}.issubset(self.docs.columns):
            self._x_m = self.docs["x_m"].to_numpy()
            self._y_m = self.docs["y_m"].to_numpy()
        else:
            self._x_m, self._y_m = to_metric(self._lon, self._lat)
        # exact-lookup table used by the rule-only baseline and as a fast path
        self._exact = {}
        for i, d in enumerate(self._direccion):
            self._exact.setdefault(match_key(str(d)), i)
        self._parsed_docs: dict[int, object] = {}

    # -- encoding ---------------------------------------------------------
    @torch.no_grad()
    def encode_queries(self, texts: list[str], role: int = 0, batch: int = 1024) -> torch.Tensor:
        out = torch.empty((len(texts), self.doc_emb.shape[1]), dtype=torch.float16, device=self.device)
        for i in range(0, len(texts), batch):
            ids = torch.from_numpy(
                encode_batch(texts[i : i + batch], self.max_len).astype(np.int64)
            ).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                emb = self.model.embed(ids, role=role)
            out[i : i + batch] = F.normalize(emb.float(), dim=-1).half()
        return out

    @torch.no_grad()
    def encode_queries_exact(self, texts: list[str], role: int = 0, batch: int = 1024) -> torch.Tensor:
        """Unit-norm query embeddings in fp32, without autocast.

        :meth:`encode_queries` runs the encoder in bf16 on GPU and stores fp16: a row's
        embedding then depends on the batch it travels in (kernel choice, padding), enough
        to flip a status near the confidence cutoff (audit 2026-10: 4 flips in 5,725 rows
        between one batch and batches of 64). Scoring uses this one instead.
        """
        out = torch.empty((len(texts), self.doc_emb.shape[1]), dtype=torch.float32, device=self.device)
        for i in range(0, len(texts), batch):
            ids = torch.from_numpy(
                encode_batch(texts[i : i + batch], self.max_len).astype(np.int64)
            ).to(self.device)
            out[i : i + batch] = F.normalize(self.model.embed(ids, role=role).float(), dim=-1)
        return out

    @torch.no_grad()
    def search_exact(self, q_emb: torch.Tensor, k: int = 20, chunk: int = 512):
        """Top-k search whose scores are exact fp32 products.

        The candidate set comes from the same fp16 matmul as :meth:`search` (it only
        decides who is in the top-k, so a rounding tie at rank k is the worst case); the
        returned similarity of each candidate is recomputed in fp32 from the unrounded query
        embedding, so it no longer carries fp16 quantisation or batch-dependent rounding.
        """
        sims_out, idx_out = [], []
        for i in range(0, q_emb.shape[0], chunk):
            q = q_emb[i : i + chunk]
            top = (q.half() @ self.doc_emb.T).topk(min(k, self.doc_emb.shape[0]), dim=-1)
            exact = (q.unsqueeze(1) * self.doc_emb[top.indices].float()).sum(-1)
            sims_out.append(exact.cpu().numpy())
            idx_out.append(top.indices.cpu().numpy())
        return np.concatenate(sims_out), np.concatenate(idx_out)

    @torch.no_grad()
    def predict_xy(self, texts: list[str], batch: int = 1024) -> np.ndarray:
        """Coordinate predicted by the auxiliary regression head, in metres (EPSG:3116).

        This is the spatial prior the model learned during training; the reranker
        uses it to damp candidates that match the text but sit in the wrong part
        of the city.
        """
        out = np.zeros((len(texts), 2), dtype=np.float64)
        for i in range(0, len(texts), batch):
            ids = torch.from_numpy(
                encode_batch(texts[i : i + batch], self.max_len).astype(np.int64)
            ).to(self.device)
            # fp32 on purpose: a bf16 coordinate is off by tens of metres and moves with the batch
            pooled = self.model.pool(ids, 0)
            xy = self.model.coord_head(pooled)
            out[i : i + batch] = xy.float().cpu().numpy()
        return out * self.xy_std + self.xy_mean

    @torch.no_grad()
    def search(self, q_emb: torch.Tensor, k: int = 20, chunk: int = 512):
        """Top-k cosine search against the full index (chunked fp16 matmul)."""
        sims_out, idx_out = [], []
        for i in range(0, q_emb.shape[0], chunk):
            sims = q_emb[i : i + chunk] @ self.doc_emb.T
            top = sims.topk(min(k, sims.shape[1]), dim=-1)
            sims_out.append(top.values.float().cpu().numpy())
            idx_out.append(top.indices.cpu().numpy())
        return np.concatenate(sims_out), np.concatenate(idx_out)

    def _doc_parse(self, i: int):
        p = self._parsed_docs.get(i)
        if p is None:
            p = parse_address(str(self._direccion[i]))
            self._parsed_docs[i] = p
        return p

    # -- score components -------------------------------------------------
    def score_components(self, raws, k: int = 20) -> dict:
        """Per-candidate reranking components, without applying any weights.

        Returns ``{doc, sim, fuzz, struct, geo}`` arrays of shape ``(n, 2k)``
        (``doc == -1`` marks an empty slot) plus the parse results. Fusion weights
        can then be swept offline without recomputing the encoder or the search,
        which is what ``scripts/tune_threshold.py`` does.
        """
        raws = [None if r is None else str(r) for r in raws]
        parsed = [parse_address(r) for r in raws]
        canon_nocomp = [
            canonical(p, style="spaced", with_complement=False) if p.parse_ok else "" for p in parsed
        ]
        canon = [canonical(p, style="spaced", with_complement=True) if p.parse_ok else "" for p in parsed]
        prenorm = [_pre_normalize(r or "")[: self.max_len] for r in raws]
        q_text = [c or pn for c, pn in zip(canon, prenorm)]

        n = len(raws)
        if n == 0:
            width = 2 * min(k, self.doc_emb.shape[0])
            empty = {key: np.zeros((0, width), dtype=dt) for key, dt in (
                ("doc", np.int64), ("sim", np.float32), ("fuzz", np.float32), ("struct", np.float32), ("geo", np.float32))}
            empty["parse_ok"] = np.zeros(0, dtype=bool)
            return empty
        e1 = self.encode_queries_exact(q_text, role=0)
        e2 = self.encode_queries_exact(prenorm, role=0)
        s1, i1 = self.search_exact(e1, k=k)
        s2, i2 = self.search_exact(e2, k=k)
        cand_idx = np.concatenate([i1, i2], axis=1)
        cand_sims = np.concatenate([s1, s2], axis=1)
        pred_xy = self.predict_xy(q_text)

        width = cand_idx.shape[1]
        out = {
            "doc": np.full((n, width), -1, dtype=np.int64),
            "sim": np.zeros((n, width), dtype=np.float32),
            "fuzz": np.zeros((n, width), dtype=np.float32),
            "struct": np.zeros((n, width), dtype=np.float32),
            "geo": np.zeros((n, width), dtype=np.float32),
        }
        for r in range(n):
            keep: dict[int, float] = {}
            for j, sim in zip(cand_idx[r], cand_sims[r]):
                keep[int(j)] = max(keep.get(int(j), -1.0), float(sim))
            px, py = pred_xy[r]
            text = canon_nocomp[r] or prenorm[r]
            for slot, (j, sim) in enumerate(keep.items()):
                target = str(self._direccion[j])
                out["doc"][r, slot] = j
                out["sim"][r, slot] = max(sim, 0.0)
                out["fuzz"][r, slot] = fuzz.token_set_ratio(text, target) / 100.0
                out["struct"][r, slot] = (
                    _structural_agreement(parsed[r], self._doc_parse(j)) if parsed[r].parse_ok else 0.0
                )
                dm = float(np.hypot(self._x_m[j] - px, self._y_m[j] - py))
                out["geo"][r, slot] = float(np.exp(-dm / GEO_TAU_M))
        out["parse_ok"] = np.array([p.parse_ok for p in parsed], dtype=bool)
        return out

    # -- main API ---------------------------------------------------------
    def normalize_batch(
        self, raws, k: int = 20, use_model: bool = True, rerank: bool = True,
        use_geo: bool = True, threshold: float | None = None,
        weights: dict | None = None, refine_input: bool = False,
    ) -> pd.DataFrame:
        cutoff = self.threshold if threshold is None else float(threshold)
        fuse_w = dict(self.weights if weights is None else weights)
        raws = [None if r is None else str(r) for r in raws]
        original_raws, refined = raws, None
        if refine_input:  # opt-in: ``raw_address`` keeps the original text, ``refine_changes`` is appended
            refined = [None if r is None else refine_address(r) for r in raws]
            raws = [r if rr is None else rr.text for rr, r in zip(refined, raws)]
        parsed = [parse_address(r) for r in raws]
        canon = [canonical(p, style="spaced", with_complement=True) if p.parse_ok else "" for p in parsed]
        canon_nocomp = [
            canonical(p, style="spaced", with_complement=False) if p.parse_ok else "" for p in parsed
        ]
        prenorm = [_pre_normalize(r or "")[: self.max_len] for r in raws]
        q_text = [c or pn for c, pn in zip(canon, prenorm)]

        n = len(raws)
        if use_model:
            e1 = self.encode_queries(q_text, role=0)
            e2 = self.encode_queries(prenorm, role=0)
            s1, i1 = self.search(e1, k=k)
            s2, i2 = self.search(e2, k=k)
            cand_sims = np.concatenate([s1, s2], axis=1)
            cand_idx = np.concatenate([i1, i2], axis=1)
        else:
            # rule-only baseline: exact key lookup, then rapidfuzz over the
            # addresses that share the same via/cross numbers.
            cand_idx, cand_sims = self._rule_candidates(parsed, canon_nocomp, k)

        pred_xy = (
            self.predict_xy(q_text) if (use_model and rerank and use_geo)
            else np.full((n, 2), np.nan)
        )

        rows = []
        for r in range(n):
            idxs, sims = cand_idx[r], cand_sims[r]
            keep: dict[int, float] = {}
            for j, s in zip(idxs, sims):
                if j < 0:
                    continue
                keep[int(j)] = max(keep.get(int(j), -1.0), float(s))
            if not keep:
                rows.append(self._empty_row(raws[r], canon[r], parsed[r]))
                continue
            cand = list(keep.items())
            if rerank:
                scored = []
                px, py = pred_xy[r]
                have_geo = np.isfinite(px) and np.isfinite(py)
                w = dict(fuse_w)
                if not have_geo:
                    # Redistribute the geographic weight over the other terms so
                    # the score stays on the same 0-1 scale.
                    share = w.pop("geo") / 3.0
                    for key in ("sim", "fuzz", "struct"):
                        w[key] += share
                for j, sim in cand:
                    target = str(self._direccion[j])
                    fz = fuzz.token_set_ratio(canon_nocomp[r] or prenorm[r], target) / 100.0
                    st = _structural_agreement(parsed[r], self._doc_parse(j)) if parsed[r].parse_ok else 0.0
                    fused = (
                        w["sim"] * max(sim, 0.0) + w["fuzz"] * fz + w["struct"] * st
                    )
                    geo = float("nan")
                    if have_geo:
                        dm = float(np.hypot(self._x_m[j] - px, self._y_m[j] - py))
                        geo = float(np.exp(-dm / GEO_TAU_M))
                        fused += w["geo"] * geo
                    scored.append((fused, sim, fz, st, j, geo))
                scored.sort(key=lambda t: t[0], reverse=True)
            else:
                scored = [
                    (max(sim, 0.0), sim, float("nan"), float("nan"), j, float("nan"))
                    for j, sim in cand
                ]
                scored.sort(key=lambda t: t[0], reverse=True)
            rows.append(self._row(raws[r], canon[r], parsed[r], scored, k, cutoff))
        frame = pd.DataFrame(rows)
        if refined is not None:
            frame["raw_address"] = original_raws
            frame["refine_changes"] = ["" if rr is None else ", ".join(rr.changes) for rr in refined]
        return frame

    def normalize_address(self, raw: str, k: int = 20) -> dict:
        """Normalize a single free-form address.

        Always check ``matched`` / ``confidence`` before using the result: a
        low-confidence row still carries the best available candidate, including
        its coordinates, and for non-address input that candidate is meaningless.
        """
        return self.normalize_batch([raw], k=k).iloc[0].to_dict()

    # -- helpers ----------------------------------------------------------
    def _rule_candidates(self, parsed, canon_nocomp, k: int):
        """Baseline candidate generation without the neural model."""
        from rapidfuzz import process

        idx = np.full((len(parsed), k), -1, dtype=np.int64)
        sims = np.zeros((len(parsed), k), dtype=np.float32)
        if not hasattr(self, "_num_buckets"):
            # Light regex bucketing (`KR 26 ...` -> ('KR', '26')): a full parse of
            # all 330k cadastral addresses would cost ~20 s and 150 MB for nothing.
            import re as _re

            from .parser import VIA_TYPE_CANON

            pat = _re.compile(r"^([A-Z]+)\s+(\d+)")
            buckets: dict[tuple, list[int]] = {}
            for i, d in enumerate(self._direccion):
                m = pat.match(str(d))
                if m is None:
                    continue
                vt = VIA_TYPE_CANON.get(m.group(1))
                if vt is None:
                    continue
                buckets.setdefault((vt, str(int(m.group(2)))), []).append(i)
            self._num_buckets = buckets
        for r, p in enumerate(parsed):
            key = match_key(canon_nocomp[r])
            hit = self._exact.get(key)
            cands: list[int] = []
            if hit is not None:
                cands.append(hit)
            if p.parse_ok:
                cands.extend(self._num_buckets.get((p.via_type, p.via_number), [])[:4000])
            if not cands:
                continue
            choices = {i: str(self._direccion[i]) for i in dict.fromkeys(cands)}
            best = process.extract(
                canon_nocomp[r] or "", choices, scorer=fuzz.token_set_ratio, limit=k
            )
            for slot, (_, score, i) in enumerate(best):
                idx[r, slot] = i
                sims[r, slot] = score / 100.0
        return idx, sims

    def _empty_row(self, raw, canon, parsed) -> dict:
        return {
            "raw_address": raw,
            "parse_ok": parsed.parse_ok,
            "rule_canonical": canon,
            "canonical_address": canon,
            "numero_predial_nacional": None,
            "manzana": None,
            "comuna": None,
            "barrio_code": None,
            "lat": np.nan,
            "lon": np.nan,
            "confidence": 0.0,
            "model_similarity": np.nan,
            "fuzz_score": np.nan,
            "struct_score": np.nan,
            "geo_score": np.nan,
            "n_predios_at_address": 0,
            "matched": False,
            "top5_doc_ids": [],
            "topk_doc_ids": [],
            "parse_notes": ";".join(parsed.notes),
            "override_rejected": False,
            "selected_doc_id": -1,
        }

    def _row(self, raw, canon, parsed, scored, k, cutoff: float | None = None) -> dict:
        cutoff = self.threshold if cutoff is None else cutoff
        top_fused = scored[0][0]
        # A cadastral address replaces the rule output only when it names the very same
        # base (see ``base_mismatches``): the fused score barely penalizes a one-digit plate
        # difference, and a neighbouring parcel is a wrong address, not a correction. The
        # candidates already scored are tried in order; the first one at or above the
        # cutoff whose base is equal wins, and every field of the row comes from it.
        canonical_address, notes = canon, list(parsed.notes)
        selected = None
        rejected = False
        if not parsed.parse_ok:
            # Nothing was understood: do not hand out a random parcel as the address.
            canonical_address = ""
            if top_fused >= cutoff:
                notes.append("no_rule_parse")
                rejected = True
        elif top_fused >= cutoff:
            first_diff = None
            for rank, cand in enumerate(scored):
                if cand[0] < cutoff:
                    break
                diff = base_mismatches(parsed, self._doc_parse(cand[4]))
                if diff:
                    first_diff = first_diff or diff
                    continue
                selected = cand
                if rank:
                    notes.append(f"cadastral_candidate_rank:{rank + 1}")
                break
            if selected is None:
                rejected = True
                notes.append("cadastral_base_mismatch:" + ",".join(first_diff))
                notes.append("cadastral_candidate_rejected:rank=1")
            else:
                # Re-render the cadastral base (legacy via types -> canonical), then the
                # plate tail the input states, then the input's units: raw cadastral units
                # are never emitted and nothing the rule output says is dropped.
                cad = self._doc_parse(selected[4])
                tail, _, _, _ = _plate_tail(parsed)
                units = render_complement(parsed.complement_text)
                canonical_address = " ".join(
                    part for part in (
                        canonical(cad, style="spaced", with_complement=False), " ".join(tail), units,
                    ) if part
                )
        chosen = selected if selected is not None else scored[0]
        fused, sim, fz, st, j, geo = chosen
        row = {
            "raw_address": raw,
            "parse_ok": parsed.parse_ok,
            "rule_canonical": canon,
            "canonical_address": canonical_address,
            "matched_cadastral_address": str(self._direccion[j]),
            "numero_predial_nacional": str(self._predial[j]),
            "manzana": str(self._manzana[j]),
            "comuna": str(self._comuna[j]),
            "barrio_code": str(self._barrio[j]),
            "lat": float(self._lat[j]),
            "lon": float(self._lon[j]),
            "confidence": float(min(max(fused, 0.0), 1.0)),
            "model_similarity": float(sim),
            "fuzz_score": float(fz),
            "struct_score": float(st),
            "geo_score": float(geo),
            "n_predios_at_address": int(self._n_predios[j]),
            "matched": selected is not None,
            "top5_doc_ids": [int(s[4]) for s in scored[:5]],
            "topk_doc_ids": [int(s[4]) for s in scored[:k]],
            "parse_notes": ";".join(notes),
            "override_rejected": rejected,
            # The document the row's parcel data comes from (-1: none). Evaluation reads this,
            # not ``top5_doc_ids[0]``: the selected candidate can be ranked below the top one.
            "selected_doc_id": -1 if rejected else int(j),
        }
        if rejected:
            # A rejected neighbour must not leave its parcel data on an unmatched row; the
            # scores stay (confidence semantics are unchanged).
            row.update(
                matched_cadastral_address=None, numero_predial_nacional=None, manzana=None,
                comuna=None, barrio_code=None, lat=np.nan, lon=np.nan, n_predios_at_address=0,
            )
        return row
