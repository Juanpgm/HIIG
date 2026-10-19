"""Catalogue status of an address: what the pipeline can honestly say about a row.

``catalog_status`` (one of :data:`STATUS_VALUES`) answers "is this address in the cadastre, and did we
match it?". ``address_verified`` is True exactly for ``verified``. The status is decided by
:func:`decide` with this PRECEDENCE (the first rule that holds wins):

1. ``unparsed``: no address, a grammar that failed, or a TRUNCATED parse (the parsed base is not the address);
2. ``verified`` / ``ambiguous``: an OK row whose base equals the cadastre's (exact guard, no caps, no soft
   findings) with fidelity risk none / low; ``ambiguous`` when several parcels share the address;
3. ``door_not_catalogued``: the base parses fully and its exact key (via, cross, plate, per-position
   quadrants, letters, BIS) is absent from the cadastre address set;
4. ``below_cutoff``: a SIN_MATCH row under the confidence cut-off (or without candidates);
5. ``ambiguous``: a SIN_MATCH row that abstained because parcels of different blocks compete;
6. ``rejected_mismatch``: every other non-verified outcome (guard / rule rejection, tier caps, unit
   conflicts, fidelity risk medium or high).

The calibrated-LR probability is deliberately NOT exposed here: it is only valid under the synthetic
mixture it was fitted on.

The set of base keys is built lazily, once, from the normalizer's documents and cached next to the
interpolation index (``catalog_keys_v1.pkl`` + sidecar ``catalog_keys_v1.json``) with the same
discipline as :mod:`cali_address.interp`: the sidecar (source hash + SHA-256 of the pickle) is checked
BEFORE anything is unpickled, and a missing, corrupt or mismatching cache is ignored and rebuilt. A cache
that cannot be written is a warning, never an error. Keys are 16-byte BLAKE2b digests of the base key.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import warnings

from .inference import _letters_and_quadrant, _num_key, _plate_tail
from .interp import _atomic_write
from .parser import is_truncated, parse_address

__all__ = [
    "CACHE_FILENAME",
    "INDEX_VERSION",
    "SIDECAR_FILENAME",
    "STATUS_VALUES",
    "base_key",
    "build_keys",
    "decide",
    "door_catalogued",
    "keys_for",
    "load_or_build",
    "source_hash",
    "warm_cache",
]

INDEX_VERSION = 1
CACHE_FILENAME = "catalog_keys_v1.pkl"
SIDECAR_FILENAME = "catalog_keys_v1.json"

STATUS_VALUES = ("verified", "ambiguous", "door_not_catalogued", "rejected_mismatch", "unparsed", "below_cutoff")

_LOCK = threading.RLock()


def base_key(parsed, cadastral: bool = False) -> bytes | None:
    """Identity of the address base of a parse (``None`` when it did not parse).

    The same identity as :func:`cali_address.inference.base_mismatches` minus the building units: via type /
    number / suffix, cross type / number / suffix, plate (no leading zeros), street letters and quadrants PER
    position, BIS flags, and what follows the plate (letters, BIS, quadrant). ``cadastral=True`` reads a
    cadastre string (a unit code after the plate is not plate letters), as the guard does for documents.
    """
    try:
        if parsed is None or not parsed.parse_ok:
            return None
        parts = [_num_key(getattr(parsed, f)) for f in (
            "via_type", "via_number", "via_suffix", "via_suffix_letters",
            "cross_type", "cross_number", "cross_suffix", "cross_suffix_letters", "plate")]
        for side in ("via", "cross"):
            letters, quadrant = _letters_and_quadrant(
                getattr(parsed, f"{side}_letters"), getattr(parsed, f"{side}_quadrant"))
            parts += [letters, quadrant or ""]
        parts += [bool(parsed.via_bis), bool(parsed.cross_bis)]
        _, letters, bis, quadrant = _plate_tail(parsed, cadastral=cadastral)
        parts += [letters, bool(bis), quadrant or ""]
        return hashlib.blake2b("\x1f".join(map(str, parts)).encode("utf-8", "replace"), digest_size=16).digest()
    except (AttributeError, TypeError, ValueError):
        return None


def build_keys(direcciones) -> frozenset:
    """Keys of the parseable cadastre addresses; everything else is skipped."""
    keys = set()
    for text in direcciones:
        if not isinstance(text, str) or not text.strip():
            continue
        key = base_key(parse_address(text), cadastral=True)
        if key is not None:
            keys.add(key)
    return frozenset(keys)


def door_catalogued(parsed, keys) -> bool:
    """Whether the exact base of ``parsed`` is in the cadastre key set ``keys``."""
    if not keys:
        return False
    key = base_key(parsed)
    return key is not None and key in keys


def source_hash(direcciones) -> str:
    """Deterministic fingerprint of the addresses the key set is built from."""
    digest = hashlib.sha256()
    digest.update(f"v{INDEX_VERSION}:{len(direcciones)}\n".encode())
    digest.update("\n".join("" if d is None else str(d) for d in direcciones).encode("utf-8", "replace"))
    return digest.hexdigest()


def _read_cache(path: str, sidecar: str, wanted_hash: str) -> frozenset | None:
    """The cached key set, only after the sidecar vouches for the source AND the pickle's own digest."""
    try:
        with open(sidecar, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if (not isinstance(meta, dict) or meta.get("version") != INDEX_VERSION
                or meta.get("source_hash") != wanted_hash):
            return None
        with open(path, "rb") as handle:
            raw = handle.read()
        if hashlib.sha256(raw).hexdigest() != meta.get("pickle_sha256"):
            return None
        payload = pickle.loads(raw)  # noqa: S301 - verified against the sidecar digest just above
        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list) or payload.get("version") != INDEX_VERSION:
            return None
        return frozenset(keys)
    except Exception:  # noqa: BLE001 - a corrupt or foreign cache is simply rebuilt
        return None


def _write_cache(path: str, sidecar: str, source: str, keys: frozenset) -> None:
    blob = pickle.dumps({"version": INDEX_VERSION, "keys": sorted(keys)}, protocol=4)
    _atomic_write(path, blob)
    meta = {"version": INDEX_VERSION, "source_hash": source, "pickle_sha256": hashlib.sha256(blob).hexdigest()}
    _atomic_write(sidecar, json.dumps(meta, sort_keys=True).encode("utf-8"))


def load_or_build(direcciones, cache_dir: str | None = None) -> frozenset:
    """The key set of this source: from the cache when sidecar and digest match, else built.

    A cache that cannot be written keeps the set in memory and warns. ``cache_dir=None`` never touches the disk.
    """
    if cache_dir is None:
        return build_keys(direcciones)
    source = source_hash(direcciones)
    path = os.path.join(cache_dir, CACHE_FILENAME)
    sidecar = os.path.join(cache_dir, SIDECAR_FILENAME)
    cached = _read_cache(path, sidecar, source)
    if cached is not None:
        return cached
    keys = build_keys(direcciones)
    try:
        _write_cache(path, sidecar, source, keys)
    except Exception as exc:  # noqa: BLE001 - never fail a normalization for a cache
        warnings.warn(f"catalogue key cache not written to {path} ({exc}); kept in memory", UserWarning,
                      stacklevel=2)
    return keys


def keys_for(normalizer) -> frozenset:
    """The (lazily built, memoized, thread-safe) key set of a normalizer's cadastre."""
    cached = getattr(normalizer, "_catalog_keys", None)
    if cached is not None:
        return cached
    with _LOCK:
        cached = getattr(normalizer, "_catalog_keys", None)
        if cached is not None:
            return cached
        keys = load_or_build(list(normalizer._direccion), getattr(normalizer, "artifacts_dir", None))
        normalizer._catalog_keys = keys
        return keys


def warm_cache(artifacts_dir: str) -> bool:
    """Build (or verify) the on-disk key set of ``artifacts_dir`` without loading the model. Never raises."""
    try:
        import pandas as pd

        docs_path = os.path.join(artifacts_dir, "catastro_docs.parquet")
        if not os.path.exists(docs_path):
            return False
        docs = pd.read_parquet(docs_path, columns=["direccion"])
        load_or_build(list(docs["direccion"].to_numpy()), artifacts_dir)
        return True
    except Exception as exc:  # noqa: BLE001 - warming is an optimisation
        warnings.warn(f"catalogue key warm-up failed ({exc})", UserWarning, stacklevel=2)
        return False


def decide(parsed, estado, motivo, fidelity_risk, exact: bool, shared: bool, door_known=None) -> str:
    """The catalogue status of one row (see the module docstring for the precedence).

    ``exact``: the OK row's guard found no cap, soft finding or unit conflict; ``shared``: several parcels
    share the matched address; ``door_known``: a zero-argument callable returning True / False (the exact base
    is / is not in the cadastre) or None (unknown), called only when the answer is needed.
    """
    if parsed is None or not getattr(parsed, "parse_ok", False) or is_truncated(parsed) or estado == "NO_PARSEABLE":
        return "unparsed"
    if estado == "OK" and exact and fidelity_risk in (None, "none", "low"):
        return "ambiguous" if shared else "verified"
    try:
        known = door_known() if door_known is not None else None
    except Exception:  # noqa: BLE001 - an unavailable key set must never lose the row
        known = None
    if known is False:
        return "door_not_catalogued"
    if estado == "SIN_MATCH":
        text = str(motivo or "")
        if text.startswith("confianza"):
            return "below_cutoff"
        if text.startswith("ambiguo"):
            return "ambiguous"
    return "rejected_mismatch"


#: Statuses whose rule text is emitted only for a complete, low-risk reading (see :func:`keeps_rule_text`).
_EMIT_IF_TRUSTED = ("below_cutoff", "rejected_mismatch")
_EMIT_RISKS = (None, "none", "low")


def complete_parse(parsed) -> bool:
    """True when ``parsed`` holds via type + via number + cross number + plate and is not truncated (never raises)."""
    return bool(
        getattr(parsed, "parse_ok", False)
        and getattr(parsed, "via_type", None) and getattr(parsed, "via_number", None)
        and getattr(parsed, "cross_number", None) and getattr(parsed, "plate", None) is not None
        and not is_truncated(parsed)
    )


def keeps_rule_text(status: str, parsed, fidelity_risk) -> bool:
    """Whether a non-OK row may still emit its rule-canonical text as ``direccion_normalizada``.

    The pipeline treats an empty normalized address as "not normalized" and falls back to the raw text, which
    is better than a half-read address: ``unparsed`` never emits; a ``high`` fidelity risk never emits (the rule
    text lost or invented door digits / letters / quadrant: it is corrupted text); ``below_cutoff`` /
    ``rejected_mismatch`` emit only when the fidelity risk is none / low AND the parse is complete; every other
    status keeps its text. ``verified`` rows are OK rows (never high) and are not blanked either way.
    """
    if status == "unparsed":
        return False
    if fidelity_risk == "high" and status != "verified":
        return False
    if status in _EMIT_IF_TRUSTED:
        return fidelity_risk in _EMIT_RISKS and complete_parse(parsed)
    return True
