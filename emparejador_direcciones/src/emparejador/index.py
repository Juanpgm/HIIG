"""In-memory index to link one address against a whole base of addresses.

A pairwise ``match`` over a 330k-row cadastre would need one comparison per
row. The index avoids that without losing any candidate by blocking on the
fields that are HARD VETOES in ``emparejador.matcher``: via type, via number,
cross number and the plate as an integer. Two addresses that differ in any
of those can never link, so they never need to be compared. With
``plate_tolerance > 0`` the lookup covers every plate within the tolerance
(the plate letter is deliberately not part of the key; the matcher vetoes
letter differences itself). Addresses without a plate or cross number have
``None`` in that key slot and therefore find equal-``None`` peers, exactly
as ``match`` treats "missing on both sides" as agreement.

Every candidate is scored by the real matcher (``match_parsed``), so the
index returns the same scores as a brute-force ``match`` scan over the same
records; the property test in ``tests/test_index.py`` enforces that.

Record ids follow Python equality and hashing: ids are compared as given,
so ``1``, ``1.0`` and ``True`` are ONE id while ``1`` and ``"1"`` are two.
``len(index)`` and ``stats()["records"]`` count DISTINCT ids, not ``add``
calls (the same id added twice is one record, several addresses).

Only the standard library and this project's own modules are used.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

from emparejador.matcher import (
    DEFAULT_LEVEL,
    DEFAULT_PLATE_TOLERANCE,
    DEFAULT_THRESHOLD,
    MatchResult,
    _validate_level,
    _validate_plate_tolerance,
    _validate_threshold,
    match_parsed,
)
from emparejador.parser import MAX_PLATE_DIGITS, CanonicalAddress, parse_canonical

__all__ = ["AddressIndex", "Candidate", "LinkResult"]

_FaceKey = tuple[str | None, str | None, str | None]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One indexed record that scored at or above the query threshold."""

    record_id: Hashable
    score: float
    result: MatchResult


@dataclass(slots=True)
class LinkResult:
    """Outcome of ``AddressIndex.find``.

    ``candidates`` holds every record scoring at or above the threshold,
    best first (score descending, then record id). ``n_compared`` counts
    the records that were scored, including those below the threshold.

    ``truncated`` is True when ``find`` was given ``max_block_scan`` and at
    least one block held more records than that cap, so the scan of that
    block stopped early and a better candidate may have been skipped.
    """

    query: CanonicalAddress
    threshold: float
    candidates: list[Candidate] = field(default_factory=list)
    n_compared: int = 0
    reason: str = ""
    truncated: bool = False

    @property
    def parse_ok(self) -> bool:
        return self.query.parse_ok

    @property
    def best(self) -> Candidate | None:
        """Top candidate, or ``None`` when nothing reached the threshold."""
        return self.candidates[0] if self.candidates else None

    @property
    def ambiguous(self) -> bool:
        """True when more than one DISTINCT record ties at the top score.

        Records that share a canonical string always score identically
        against any query, so a canonical string shared by several ids also
        shows up here as a tie.
        """
        if len(self.candidates) < 2:
            return False
        return self.candidates[1].score == self.candidates[0].score

    @property
    def alternatives(self) -> int:
        """Number of DISTINCT record ids besides ``best`` at or above the
        threshold (``candidates`` already holds one entry per id)."""
        return max(len(self.candidates) - 1, 0)

    @property
    def near_ambiguous(self) -> bool:
        """True when another distinct record besides ``best`` also reached the
        threshold, even without an exact tie (a known-complement unit of the
        same building scores 0.98, a cross-type variant 0.97). ``ambiguous``
        is the stricter exact-tie signal and is unchanged."""
        return self.alternatives > 0


def _plate_key(plate: str | None) -> int | None:
    """Plate as an integer (zero-stripped digits, letter ignored) or ``None``.

    The parser caps plates at ``MAX_PLATE_DIGITS`` significant digits, so the
    conversion is always safe for a parsed address.
    """
    if plate is None:
        return None
    digits = plate[:-1] if plate[-1].isalpha() else plate
    stripped = digits.lstrip("0") or "0"
    assert len(stripped) <= MAX_PLATE_DIGITS  # guaranteed by parse_canonical
    return int(stripped)


def _validate_max_block_scan(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_block_scan debe ser un entero >= 1 o None: {value!r}")
    return value


def _id_sort_key(record_id: Hashable) -> tuple[str, object]:
    """Deterministic ordering that also works for ids of mixed types."""
    if isinstance(record_id, (int, float, str, bytes)):
        return (type(record_id).__name__, record_id)
    return (type(record_id).__name__, repr(record_id))


class AddressIndex:
    """Blocked in-memory index of parsed addresses.

    ``add`` parses the address once and files it under its block; ``find``
    scores the query only against the blocks that can still link.
    """

    __slots__ = ("_faces", "_ids", "_blocks", "_max_block", "plate_tolerance", "rejected")

    def __init__(self, plate_tolerance: int = DEFAULT_PLATE_TOLERANCE) -> None:
        self.plate_tolerance = _validate_plate_tolerance(plate_tolerance)
        self._faces: dict[_FaceKey, dict[int | None, list[tuple[Hashable, CanonicalAddress]]]] = {}
        self._ids: set[Hashable] = set()
        self._blocks = 0
        self._max_block = 0
        self.rejected = 0

    @classmethod
    def from_records(
        cls,
        records: Iterable[tuple[Hashable, object]],
        plate_tolerance: int = DEFAULT_PLATE_TOLERANCE,
    ) -> AddressIndex:
        """Build an index from ``(record_id, address)`` pairs."""
        index = cls(plate_tolerance=plate_tolerance)
        for record_id, address in records:
            index.add(record_id, address)
        return index

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, record_id: Hashable, address: object) -> bool:
        """Index one record. Returns ``False`` (and counts it in ``rejected``)
        when the address does not parse. ``record_id`` must be hashable."""
        hash(record_id)  # unhashable ids are a programming error: TypeError
        parsed = parse_canonical(address)
        if not parsed.parse_ok:
            self.rejected += 1
            return False
        face: _FaceKey = (parsed.via_type, parsed.via_number, parsed.cross_number)
        plates = self._faces.setdefault(face, {})
        bucket = plates.get(_plate_key(parsed.plate))
        if bucket is None:
            bucket = plates[_plate_key(parsed.plate)] = []
            self._blocks += 1
        bucket.append((record_id, parsed))
        self._ids.add(record_id)
        if len(bucket) > self._max_block:
            self._max_block = len(bucket)
        return True

    def stats(self) -> dict[str, int]:
        """Counters: distinct record ids indexed, blocks, rejected records,
        largest block (entries, so an id added twice counts twice there)."""
        return {
            "records": len(self._ids),
            "blocks": self._blocks,
            "rejected": self.rejected,
            "max_block_size": self._max_block,
        }

    def _blocks_for(self, parsed: CanonicalAddress) -> list[list[tuple[Hashable, CanonicalAddress]]]:
        plates = self._faces.get((parsed.via_type, parsed.via_number, parsed.cross_number))
        if not plates:
            return []
        key = _plate_key(parsed.plate)
        tolerance = self.plate_tolerance
        if key is None or tolerance == 0:
            bucket = plates.get(key)
            return [bucket] if bucket else []
        # Walk whichever is smaller: the tolerance window or the face's plates.
        if 2 * tolerance + 1 <= len(plates):
            window = (plates.get(p) for p in range(key - tolerance, key + tolerance + 1))
            return [bucket for bucket in window if bucket]
        return [
            bucket
            for plate, bucket in plates.items()
            if plate is not None and abs(plate - key) <= tolerance
        ]

    def find(
        self,
        query: object,
        threshold: float = DEFAULT_THRESHOLD,
        max_block_scan: int | None = None,
        *,
        level: str = DEFAULT_LEVEL,
    ) -> LinkResult:
        """Link ``query`` against the indexed records.

        Never raises on bad query input: an unparseable query yields an
        empty result carrying the parse notes in ``reason``. A bad
        ``threshold`` or ``max_block_scan`` is a settings error and raises.

        ``max_block_scan`` (default ``None``: no cap) bounds how many records
        of ONE block are scored. When a block is larger, its scan stops and
        the result is flagged ``truncated``.

        ``level`` is passed to the matcher unchanged (``"unit"`` or
        ``"building"`` or ``"structure"``); blocking only uses via, cross and
        plate, so it is valid at every level. At ``"building"`` and
        ``"structure"`` every unit of one building ties, so expect
        ``ambiguous`` to be common.
        """
        validated = _validate_threshold(threshold)
        validated_level = _validate_level(level)
        cap = _validate_max_block_scan(max_block_scan)
        parsed = parse_canonical(query)
        if not parsed.parse_ok:
            reason = f"consulta no canónica ({', '.join(parsed.notes) or 'inválida'})"
            return LinkResult(query=parsed, threshold=validated, reason=reason)

        best_by_id: dict[Hashable, Candidate] = {}
        n_compared = 0
        truncated = False
        for bucket in self._blocks_for(parsed):
            if cap is not None and len(bucket) > cap:
                truncated = True
                bucket = bucket[:cap]
            for record_id, other in bucket:
                n_compared += 1
                result = match_parsed(parsed, other, validated, self.plate_tolerance, level=validated_level)
                if not result.is_match:
                    continue
                known = best_by_id.get(record_id)
                if known is None or result.score > known.score:
                    best_by_id[record_id] = Candidate(record_id, result.score, result)
        candidates = sorted(
            best_by_id.values(), key=lambda c: (-c.score, _id_sort_key(c.record_id))
        )
        reason = "" if candidates else "sin candidatos sobre el umbral"
        return LinkResult(
            query=parsed,
            threshold=validated,
            candidates=candidates,
            n_compared=n_compared,
            reason=reason,
            truncated=truncated,
        )
