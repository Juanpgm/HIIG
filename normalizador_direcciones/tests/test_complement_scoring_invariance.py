"""The complement stage must not change what the retriever scores or gates on.

``canonical(p, with_complement=True)`` is the encoder query text (and the
coordinate-regressor input) in ``inference.py``, and ``ParsedAddress.complement``
feeds the quadrant-in-complement rule in ``service.py``. Both must read exactly
as they did before the complement stage existed; otherwise new unit kinds
(PISO, AGR, SEC), hyphen gluing (``T-F``) or name handling silently shift
``confianza`` and can flip accept/abstain. The complement stage reads its own
field, ``ParsedAddress.complement_text``.

Expected values below are the outputs of the parser before the complement stage.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import (  # noqa: E402
    COMPLEMENT_CANON,
    canonical,
    merge_complement,
    parse_address,
    render_complement,
)

BASE = "CL 5 # 38 - 10"

#: raw input -> scoring complement produced by the parser before the complement stage.
LEGACY_SCORING_COMPLEMENT = [
    ("CL 5 # 38-10 SECTOR 2 AGRUPACION 4 TORRE B", "SECTOR 2 AGRUPACION 4 TO B"),
    ("CL 5 # 38-10 PISO 3", "PISO 3"),
    ("CL 5 # 38-10 PIS 3", "PIS 3"),
    ("CL 5 # 38-10 AGRUP 4 SECT 2", "AGRUP 4 SECT 2"),
    ("CL 5 # 38-10 TORRE -A", "TO - A"),
    ("CL 5 # 38-10 T-F", "T - F"),
    ("CL 5 # 38-10 t-f ap-101", "T - F AP - 101"),
    ("CL 5 # 38-10 CONJ DIVENTO BARRIO LA FLORA", "CONJ DIVENTO BR LA FLORA"),
    ("CL 5 # 38-10 sector los balsos", "SECTOR LOS BALSOS"),
    ("CL 5 # 38-10 TORRE F", "TO F"),
    ("CL 5 # 38-10 BL 3", "BLQ 3"),
]


@pytest.mark.parametrize("raw, expected", LEGACY_SCORING_COMPLEMENT)
def test_scoring_complement_matches_the_pre_stage_parser(raw, expected):
    assert parse_address(raw).complement == expected


@pytest.mark.parametrize("raw, expected", LEGACY_SCORING_COMPLEMENT)
def test_encoder_query_text_is_unchanged_by_the_complement_stage(raw, expected):
    p = parse_address(raw)
    assert canonical(p, style="spaced", with_complement=True) == f"{BASE} {expected}"
    assert canonical(p, style="spaced", with_complement=False) == BASE


@pytest.mark.parametrize("alias", ["PISO", "PIS", "AGR", "AGRUPACION", "AGRUP", "AGP", "SEC", "SECTOR", "SECT"])
def test_complement_stage_kinds_stay_verbatim_in_the_scoring_complement(alias):
    # Registered for the complement stage...
    assert alias in COMPLEMENT_CANON
    # ...but invisible to the retriever.
    assert parse_address(f"CL 5 # 38-10 {alias} 7").complement == f"{alias} 7"


@pytest.mark.parametrize(
    "raw, units",
    [
        ("CL 5 # 38-10 SECTOR 2 AGRUPACION 4 TORRE B", "AGR 4 SEC 2 TO B"),
        ("CL 5 # 38-10 T-F", "TO F"),
        ("CL 5 # 38-10 t-f ap-101", "TO F AP 101"),
        ("CL 5 # 38-10 Torre F Piso 3", "TO F PISO 3"),
        ("CL 5 # 38-10 TORRE -A", "TO A"),
        ("CL 5 # 38-10 ED. Torre de Cristal", "ED TORRE DE CRISTAL"),
    ],
)
def test_complement_stage_reads_complement_text_not_the_scoring_complement(raw, units):
    p = parse_address(raw)
    assert render_complement(p.complement_text) == units
    assert merge_complement(BASE, p.complement_text).address == f"{BASE} {units}"


@pytest.mark.parametrize("raw", [None, "", "   ", "asdf qwer", "CL 5 # 38-10"])
def test_complement_text_is_none_without_a_complement(raw):
    assert parse_address(raw).complement_text is None


def test_complement_text_is_upper_and_whitespace_collapsed():
    # (Text after a comma is a locality - barrio, city - and is cut by the parser.)
    p = parse_address("CL 5 # 38-10   torre   f    apto 101 ")
    assert p.complement_text == "TORRE F APTO 101"
