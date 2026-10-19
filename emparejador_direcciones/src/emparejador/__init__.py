"""Public package surface for the emparejador (address matcher).

This package is independent of any other project: it owns its own parsing
grammar and its own scoring logic. It re-exports the small public API that
callers (the CLI, tests, or another program) are expected to use.
"""

from __future__ import annotations

__version__ = "0.2.0"

from emparejador.index import AddressIndex, Candidate, LinkResult
from emparejador.matcher import (
    BUILDING_IGNORED_KINDS,
    DEFAULT_LEVEL,
    DEFAULT_PLATE_TOLERANCE,
    DEFAULT_THRESHOLD,
    LEVELS,
    STRUCTURE_DISTINGUISHING_KINDS,
    STRUCTURE_UNIT_KINDS,
    FieldComparison,
    InvalidLevelError,
    InvalidThresholdError,
    InvalidToleranceError,
    MatchResult,
    building_view,
    match,
    match_parsed,
    structure_view,
)
from emparejador.parser import CanonicalAddress, clean_text, parse_canonical, render

__all__ = [
    "__version__",
    "BUILDING_IGNORED_KINDS",
    "DEFAULT_LEVEL",
    "DEFAULT_PLATE_TOLERANCE",
    "DEFAULT_THRESHOLD",
    "LEVELS",
    "STRUCTURE_DISTINGUISHING_KINDS",
    "STRUCTURE_UNIT_KINDS",
    "AddressIndex",
    "Candidate",
    "CanonicalAddress",
    "FieldComparison",
    "InvalidLevelError",
    "InvalidThresholdError",
    "InvalidToleranceError",
    "LinkResult",
    "MatchResult",
    "building_view",
    "clean_text",
    "match",
    "match_parsed",
    "parse_canonical",
    "render",
    "structure_view",
]
