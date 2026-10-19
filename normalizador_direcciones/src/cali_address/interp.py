"""Interpolation of a plate along a street face of the cadastre.

The cadastre knows where parcels ARE, not where a house number that does not exist would be. For a
plate that differs from the matched parcel's by one or two units (or by its letters), the honest
position is *between* the two nearest known parcels of the same street pair and side, never the
neighbour's own centroid.

The index is keyed by street pair (via type / number / letters / quadrant / suffix / BIS and the
cross number / letters / quadrant / suffix / BIS) and holds, per side of the street (plate parity),
the plate numbers in ascending order with their coordinates. It is built lazily, once, from the
normalizer's documents, and cached to ``interp_index_v1.pkl`` next to the other artifacts. The cache
is verified through a tiny sidecar JSON (source hash + SHA-256 of the pickle) BEFORE anything is
unpickled; a missing, corrupt or mismatching sidecar means the pickle is ignored and the index rebuilt. A
cache that cannot be written is a warning, never an error.

Interpolation is only trusted over short gaps: between two known parcels farther apart than
:data:`INTERP_MAX_PLATE_GAP` plate numbers or :data:`INTERP_MAX_GAP_M` metres the plate is NOT interpolated
(``kind="gap"``), and an extrapolation never moves more than :data:`INTERP_MAX_EXTRAP_M` metres from the last
known parcel. The caps and the radii of the position tiers come from a leave-one-out study of the cadastre itself
(numbers next to the constants below); all are module-level, read at call time so they can be retuned.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import tempfile
import threading
import warnings
from bisect import bisect_left
from dataclasses import dataclass
from statistics import median

import numpy as np

from .inference import _letters_and_quadrant, _num_key
from .parser import parse_address

__all__ = [
    "CACHE_FILENAME",
    "INDEX_VERSION",
    "EXTRAP_RADIUS_STEPS",
    "INTERP_MAX_EXTRAP_M",
    "INTERP_MAX_GAP_M",
    "INTERP_MAX_PLATE_GAP",
    "TRAMO_RADIUS_FLOOR_M",
    "TRAMO_RADIUS_STEPS",
    "extrap_radius_m",
    "step_radius",
    "tramo_radius_m",
    "SIDECAR_FILENAME",
    "warm_cache",
    "MAX_EXTRAPOLATION_PLATES",
    "InterpIndex",
    "Located",
    "build_index",
    "index_for",
    "load_or_build",
    "source_hash",
    "street_key",
]

INDEX_VERSION = 1

#: New, versioned file next to the serving artifacts; never an existing artifact.
CACHE_FILENAME = "interp_index_v1.pkl"

#: Sidecar with the source hash and the pickle's SHA-256; checked before the pickle is read.
SIDECAR_FILENAME = "interp_index_v1.json"

# ---------------------------------------------------------------------------------------------------
# Caps and radii, DERIVED FROM THE CADASTRE (no GPS): leave-one-out (LOO) on the 31,229 street faces with at
# least three plates of the same side (233,791 interpolations, 63,852 extrapolations). Each known parcel is
# removed in turn and re-estimated from its flanking parcels; the error is the distance to its true centroid.
#
# p90 of the interpolation error by gap between the two flanking parcels (metres) and by plate-number gap:
#   metres:  (0,25] 3.7 | (25,50] 23.1 | (50,100] 61.0 | (100,150] 101.6 | (150,inf) 1200
#   plates:  (8,16] 3.2 | (16,24] 6.7 | (24,32] 12.0 | (32,40] 16.2 | (40,60] 27.4 | (60,100] 58.4 | >100 92.7
# p90 of the extrapolation error by distance of the estimate from the last known parcel (metres):
#   (0,25] 10.3 | (25,50] 55.6 | (50,100] 130.9 | (100,150] 199.4 | (150,inf) 3096
#
# Cap rule: the largest bin edge whose bin p90 stays <= 30 m. Interpolation: 50 m (the next bin is 61 m);
# plate gap: 60 (the next bin is 58 m); extrapolation: 25 m (the next bin is 56 m).
# ---------------------------------------------------------------------------------------------------

#: Two known parcels of the same face farther apart than this many metres (or plate numbers, below) are not
#: interpolated between: the position degrades to tier ``via``.
INTERP_MAX_GAP_M = 50.0
INTERP_MAX_PLATE_GAP = 60

#: An extrapolated point farther than this many metres from the last known parcel is not published.
INTERP_MAX_EXTRAP_M = 25.0

#: Beyond the last (or before the first) known plate of a face by more than this many plate units
#: no coordinates are extrapolated (the tier is still ``via``). Not re-derived: the metre cap binds first.
MAX_EXTRAPOLATION_PLATES = 50

#: ``(upper edge in metres, radius in metres)`` per bin, right-closed, the last bin open: the p90 above, rounded.
#: Interpolation radius by the gap between the flanking parcels (``tramo`` tier, see :func:`tramo_radius_m`).
TRAMO_RADIUS_STEPS = ((25.0, 4.0), (50.0, 23.0), (100.0, 61.0), (150.0, 102.0), (math.inf, 1200.0))

#: Extrapolation radius by the distance of the estimate from the last known parcel.
EXTRAP_RADIUS_STEPS = ((25.0, 10.0), (50.0, 56.0), (100.0, 131.0), (150.0, 199.0), (math.inf, 3096.0))

#: A ``tramo`` row is never more precise than the parcel footprint itself (the ``predio`` p90 radius).
TRAMO_RADIUS_FLOOR_M = 15.0


def step_radius(steps, metres) -> float:
    """Radius of the first bin of ``steps`` whose upper edge is >= ``metres``; an unknown, negative or NaN
    value (or a non-number) gets the last, widest bin."""
    try:
        value = float(metres)
    except (TypeError, ValueError):
        return float(steps[-1][1])
    if math.isnan(value) or value < 0:
        return float(steps[-1][1])
    for edge, radius in steps:
        if value <= edge:
            return float(radius)
    return float(steps[-1][1])


def tramo_radius_m(gap_m) -> float:
    """Radius of an interpolated position from the gap (m) between its flanking parcels (0 for a known plate)."""
    return max(step_radius(TRAMO_RADIUS_STEPS, gap_m), TRAMO_RADIUS_FLOOR_M)


def extrap_radius_m(distance_m) -> float:
    """Radius of an extrapolated position from its distance (m) to the last known parcel."""
    return step_radius(EXTRAP_RADIUS_STEPS, distance_m)

_LOCK = threading.RLock()


@dataclass(frozen=True)
class Located:
    """Result of :meth:`InterpIndex.locate`.

    ``kind`` is ``interp`` (the plate is a known one or lies between two NEARBY known ones of its side),
    ``gap`` (between two known ones that are too far apart to interpolate: ``lat`` / ``lon`` are the
    straight-line value, only to be published when the caller accepts via-tier coordinates), ``extrap``
    (beyond the first / last known plate: the linear extension, or ``None`` past the plate or metre caps)
    or ``none`` (no parcel of that side is known).
    """

    kind: str
    lat: float | None = None
    lon: float | None = None
    #: Metres between the two flanking parcels (``interp`` / ``gap``; 0 for a known plate) or between the
    #: estimate and the last known parcel (``extrap``); ``None`` when there is no such distance.
    gap_m: float | None = None


_NONE = Located("none")


def _metres(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(a)))


def street_key(parsed) -> tuple | None:
    """Street-pair identity of a parse, or ``None`` when it has no via, cross or plate."""
    try:
        if parsed is None or not parsed.parse_ok or parsed.plate is None:
            return None
        parts: list = [str(parsed.via_type or ""), _num_key(parsed.via_number)]
        for side in ("via", "cross"):
            letters, quadrant = _letters_and_quadrant(
                getattr(parsed, f"{side}_letters"), getattr(parsed, f"{side}_quadrant"))
            if side == "cross":
                parts.append(_num_key(parsed.cross_number))
            parts += [letters, quadrant or "", _num_key(getattr(parsed, f"{side}_suffix")),
                      bool(getattr(parsed, f"{side}_bis"))]
        return tuple(parts)
    except (AttributeError, TypeError, ValueError):
        return None


def _plate_number(parsed) -> int | None:
    try:
        text = _num_key(parsed.plate)
        return int(text) if text.isdigit() else None
    except (AttributeError, TypeError, ValueError):
        return None


class InterpIndex:
    """``{street key: {parity: (plates, lats, lons)}}`` with ascending plates."""

    def __init__(self, faces: dict | None = None) -> None:
        self._faces = faces or {}

    def __len__(self) -> int:
        return len(self._faces)

    def to_payload(self) -> dict:
        return {k: {p: (tuple(v[0]), tuple(v[1]), tuple(v[2])) for p, v in sorted(f.items())}
                for k, f in sorted(self._faces.items(), key=lambda kv: repr(kv[0]))}

    def locate(self, key, plate) -> Located:
        """Coordinates of ``plate`` on the face ``key`` (see :class:`Located`)."""
        try:
            number = int(plate)
        except (TypeError, ValueError):
            return _NONE
        face = self._faces.get(key) if key is not None else None
        side = face.get(number % 2) if face else None
        if not side:
            return _NONE
        plates, lats, lons = side
        i = bisect_left(plates, number)
        if i < len(plates) and plates[i] == number:
            return Located("interp", lats[i], lons[i], 0.0)
        if 0 < i < len(plates):
            t = (number - plates[i - 1]) / (plates[i] - plates[i - 1])
            lat = lats[i - 1] + t * (lats[i] - lats[i - 1])
            lon = lons[i - 1] + t * (lons[i] - lons[i - 1])
            gap_m = _metres(lats[i - 1], lons[i - 1], lats[i], lons[i])
            if self._far(plates[i - 1], plates[i], lats[i - 1], lons[i - 1], lats[i], lons[i]):
                return Located("gap", lat, lon, gap_m)
            return Located("interp", lat, lon, gap_m)
        # beyond one end of the face
        end, prev = (0, 1) if i == 0 else (len(plates) - 1, len(plates) - 2)
        if abs(number - plates[end]) > MAX_EXTRAPOLATION_PLATES:
            return Located("extrap")
        if len(plates) == 1 or self._far(plates[prev], plates[end], lats[prev], lons[prev], lats[end], lons[end]):
            # nothing reliable to extend a line from: the last parcel's own point
            return Located("extrap", lats[end], lons[end])
        slope = (number - plates[end]) / (plates[end] - plates[prev])
        lat = lats[end] + slope * (lats[end] - lats[prev])
        lon = lons[end] + slope * (lons[end] - lons[prev])
        distance = _metres(lats[end], lons[end], lat, lon)
        if distance > INTERP_MAX_EXTRAP_M:
            return Located("extrap", None, None, distance)
        return Located("extrap", lat, lon, distance)

    @staticmethod
    def _far(p1, p2, lat1, lon1, lat2, lon2) -> bool:
        """Whether two known parcels are too far apart (plate numbers or metres) to trust a line between them."""
        return abs(p2 - p1) > INTERP_MAX_PLATE_GAP or _metres(lat1, lon1, lat2, lon2) > INTERP_MAX_GAP_M


def build_index(direcciones, lons, lats) -> InterpIndex:
    """Index of the parseable, located addresses; everything else is skipped."""
    grouped: dict[tuple, dict[tuple[int, int], tuple[list, list]]] = {}
    for text, lon, lat in zip(direcciones, lons, lats):
        try:
            lon, lat = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lon) and math.isfinite(lat)) or not isinstance(text, str) or not text.strip():
            continue
        parsed = parse_address(text)
        key, number = street_key(parsed), _plate_number(parsed)
        if key is None or number is None:
            continue
        cell = grouped.setdefault(key, {}).setdefault((number % 2, number), ([], []))
        cell[0].append(lat)
        cell[1].append(lon)
    faces: dict = {}
    for key, cells in grouped.items():
        sides: dict[int, tuple[list, list, list]] = {}
        for (parity, number), (cell_lats, cell_lons) in sorted(cells.items()):
            plates, la, lo = sides.setdefault(parity, ([], [], []))
            plates.append(number)
            la.append(float(median(cell_lats)))
            lo.append(float(median(cell_lons)))
        faces[key] = sides
    return InterpIndex(faces)


def source_hash(direcciones, lons, lats) -> str:
    """Deterministic fingerprint of the addresses and coordinates the index is built from."""
    digest = hashlib.sha256()
    digest.update(f"v{INDEX_VERSION}:{len(direcciones)}\n".encode())
    digest.update("\n".join("" if d is None else str(d) for d in direcciones).encode("utf-8", "replace"))
    digest.update(np.ascontiguousarray(np.asarray(lons, dtype="float64")).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(lats, dtype="float64")).tobytes())
    return digest.hexdigest()


def _from_payload(payload) -> InterpIndex | None:
    faces = payload.get("faces") if isinstance(payload, dict) else None
    if not isinstance(faces, dict):
        return None
    return InterpIndex({k: {int(p): (list(v[0]), list(v[1]), list(v[2])) for p, v in f.items()}
                        for k, f in faces.items()})


def _read_cache(path: str, sidecar: str, wanted_hash: str) -> InterpIndex | None:
    """The cached index, only after the sidecar vouches for the source AND the pickle's own digest."""
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
        if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
            return None
        return _from_payload(payload)
    except Exception:  # noqa: BLE001 - a corrupt or foreign cache is simply rebuilt
        return None


def _atomic_write(path: str, data: bytes) -> None:
    folder = os.path.dirname(path) or "."
    handle, tmp = tempfile.mkstemp(prefix=".interp_index_", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_cache(path: str, sidecar: str, source: str, index: InterpIndex) -> None:
    blob = pickle.dumps({"version": INDEX_VERSION, "faces": index.to_payload()}, protocol=4)
    _atomic_write(path, blob)
    meta = {"version": INDEX_VERSION, "source_hash": source, "pickle_sha256": hashlib.sha256(blob).hexdigest()}
    _atomic_write(sidecar, json.dumps(meta, sort_keys=True).encode("utf-8"))


def load_or_build(direcciones, lons, lats, cache_dir: str | None = None) -> InterpIndex:
    """The index of this source: from the cache when sidecar and digest match, else built.

    A cache that cannot be written (read-only folder, full disk) keeps the index in memory and warns once.
    ``cache_dir=None`` never touches the disk.
    """
    if cache_dir is None:
        return build_index(direcciones, lons, lats)
    source = source_hash(direcciones, lons, lats)
    path = os.path.join(cache_dir, CACHE_FILENAME)
    sidecar = os.path.join(cache_dir, SIDECAR_FILENAME)
    cached = _read_cache(path, sidecar, source)
    if cached is not None:
        return cached
    index = build_index(direcciones, lons, lats)
    try:
        _write_cache(path, sidecar, source, index)
    except Exception as exc:  # noqa: BLE001 - never fail a normalization for a cache
        warnings.warn(f"interpolation index cache not written to {path} ({exc}); kept in memory",
                      UserWarning, stacklevel=2)
    return index


def _effective_coords(lon, lat, own_lon, own_lat):
    """Parcel's own centroid where known, the address centroid elsewhere (same rule as the service)."""
    lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    if own_lon is not None and own_lat is not None:
        own_lon, own_lat = np.asarray(own_lon, dtype="float64"), np.asarray(own_lat, dtype="float64")
        ok = np.isfinite(own_lon) & np.isfinite(own_lat)
        lon, lat = np.where(ok, own_lon, lon), np.where(ok, own_lat, lat)
    return lon, lat


def warm_cache(artifacts_dir: str) -> bool:
    """Build (or verify) the on-disk index of ``artifacts_dir`` without loading the model.

    Used by the image preparation script; the service warms it at startup and keeps the lazy path as a
    fallback. Returns False when the cadastre file is not there or unreadable. Never raises.
    """
    try:
        import pandas as pd

        from .inference import load_own_centroids

        docs_path = os.path.join(artifacts_dir, "catastro_docs.parquet")
        if not os.path.exists(docs_path):
            return False
        docs = pd.read_parquet(docs_path).reset_index(drop=True)
        own = load_own_centroids(docs, artifacts_dir)
        own_lon, own_lat = own if own is not None else (None, None)
        lon, lat = _effective_coords(docs["centroid_lon"].to_numpy(), docs["centroid_lat"].to_numpy(),
                                     own_lon, own_lat)
        load_or_build(list(docs["direccion"].to_numpy()), lon, lat, artifacts_dir)
        return True
    except Exception as exc:  # noqa: BLE001 - warming is an optimisation
        warnings.warn(f"interpolation index warm-up failed ({exc})", UserWarning, stacklevel=2)
        return False


def index_for(normalizer) -> InterpIndex:
    """The (lazily built, memoized, thread-safe) index of a normalizer's cadastre."""
    cached = getattr(normalizer, "_interp_index", None)
    if cached is not None:
        return cached
    with _LOCK:
        cached = getattr(normalizer, "_interp_index", None)
        if cached is not None:
            return cached
        lon, lat = _effective_coords(
            normalizer._lon, normalizer._lat,
            getattr(normalizer, "_own_lon", None), getattr(normalizer, "_own_lat", None))
        index = load_or_build(
            list(normalizer._direccion), lon, lat, getattr(normalizer, "artifacts_dir", None))
        normalizer._interp_index = index
        return index
