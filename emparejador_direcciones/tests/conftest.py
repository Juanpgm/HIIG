"""Pytest configuration: make the ``emparejador`` package importable.

The package lives under ``src/`` rather than at the project root, so the
test suite must add ``<project>/src`` to ``sys.path`` before any test
imports ``emparejador``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
