"""Multi-part unit values (``APTO 3-02``, ``TORRE F-2``, ``TORRE UNO A``) stay whole.

Different units of one building must never render to the same text, or the
downstream building key merges them. A value is one or more code tokens joined by
a hyphen or a space; its canonical spelling is:

* digit and letter runs are glued where the class changes (``101 A`` -> ``101A``,
  ``F-2`` -> ``F2``), the spelling the cadastre uses (``AP 101A``);
* two runs of the same class are joined with a hyphen (``3 02`` -> ``3-02``), so
  ``3-02`` and ``302`` stay different;
* the first digit run loses its leading zeros (``AP 0101`` -> ``AP 101``); later
  runs keep them (``3-02`` is not ``3-2``).

Marker words (``N``, ``NO``, ``NRO``, ``NUM``, ``NUMERO``, ``LETRA``) before a value
are skipped, and a number word followed by a letter keeps both (``UNO A`` -> ``1A``).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import extract_complement_units, merge_complement, render_complement  # noqa: E402

BASE = "CL 99 # 9 - 99"


# ---------------------------------------------------------------------------
# Distinct inputs never collapse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a, b",
    [
        ("APTO 3-02", "AP 3-03"),
        ("TORRE B APTO 3 02", "TORRE B APTO 3 03"),
        ("TORRE F-2", "TORRE F-3"),
        ("TORRE N 3", "TORRE N 4"),
        ("APTO N 101", "APTO N 102"),
        ("CASA N 5", "CASA N 6"),
        ("LOTE N 5", "LOTE N 6"),
        ("PISO N 2", "PISO N 3"),
        ("TORRE LETRA A", "TORRE LETRA B"),
        ("TORRE UNO A", "TORRE UNO B"),
        ("APTO 3-02", "APTO 302"),
        ("APTO 3-02", "APTO 3-2"),
        ("TORRE A-B", "TORRE A-C"),
        ("APTO 101 A", "APTO 101 B"),
    ],
)
def test_different_units_render_differently(a, b):
    assert render_complement(a) is not None and render_complement(b) is not None
    assert render_complement(a) != render_complement(b)
    assert merge_complement(BASE, a).address != merge_complement(BASE, b).address


@pytest.mark.parametrize(
    "text, expected",
    [
        ("APTO 3-02", [("AP", "3-02")]),
        ("APTO 3 - 02", [("AP", "3-02")]),
        ("APTO 3 02", [("AP", "3-02")]),
        ("TORRE B APTO 3 02", [("TO", "B"), ("AP", "3-02")]),
        ("TORRE F-2", [("TO", "F2")]),
        ("TORRE F 2", [("TO", "F2")]),
        ("TORRE F2", [("TO", "F2")]),
        ("TORRE N 3", [("TO", "3")]),
        ("TORRE NO. 3", [("TO", "3")]),
        ("TORRE NRO 3", [("TO", "3")]),
        ("TORRE NUM 3", [("TO", "3")]),
        ("TORRE NUMERO 3", [("TO", "3")]),
        ("APTO N 101", [("AP", "101")]),
        ("APTO NO 101", [("AP", "101")]),
        ("CASA N 5", [("CA", "5")]),
        ("LOTE N 5", [("LT", "5")]),
        ("PISO N 2", [("PISO", "2")]),
        ("TORRE LETRA A", [("TO", "A")]),
        ("TORRE UNO A", [("TO", "1A")]),
        ("TORRE UNO-A", [("TO", "1A")]),
        ("TORRE A-B", [("TO", "A-B")]),
        ("TORRE II-A", [("TO", "II-A")]),
        ("APTO 0302", [("AP", "302")]),
        ("APTO 02-03", [("AP", "2-03")]),
        ("APTO 3-02 A", [("AP", "3-02A")]),
    ],
)
def test_canonical_value_spelling(text, expected):
    assert extract_complement_units(text) == expected


@pytest.mark.parametrize("text", ["APTO 101A", "APTO 101 A", "APTO 101-A", "APTO 101 - A", "AP 0101A", "apto 101a"])
def test_letter_suffix_spellings_agree(text):
    assert extract_complement_units(text) == [("AP", "101A")]
    assert render_complement(text) == "AP 101A"


@pytest.mark.parametrize("text", ["TORRE 1 A", "TORRE UNO A", "TORRE 1A", "TORRE 01-A"])
def test_number_word_and_digit_towers_agree(text):
    assert render_complement(text) == "TO 1A"


# ---------------------------------------------------------------------------
# Markers alone, and markers that are the value
# ---------------------------------------------------------------------------
def test_single_letter_n_is_still_a_tower_code():
    assert extract_complement_units("TORRE N") == [("TO", "N")]
    assert extract_complement_units("CASA N") == [("CA", "N")]


@pytest.mark.parametrize("text", ["TORRE LETRA", "TORRE NUMERO", "TORRE NO", "APTO NO", "TORRE NRO"])
def test_marker_without_a_value_is_not_a_unit(text):
    assert extract_complement_units(text) == []


# ---------------------------------------------------------------------------
# What must still stop a value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("AP 101 Y 102", [("AP", "101")]),
        ("CASA 5 A LA ESQUINA", [("CA", "5")]),
        ("CASA 17 2 PISOS", [("CA", "17")]),
        ("CASA B 2 PISOS", [("CA", "B")]),
        ("TO B SO 1 GA 1", [("TO", "B"), ("GA", "1")]),
        ("TORRE 2 APTO 101", [("TO", "2"), ("AP", "101")]),
        ("TORRE B 2DO PISO", [("TO", "B")]),
        ("APTO 101 SEGUNDO PISO", [("AP", "101")]),
    ],
)
def test_values_stop_at_connectors_counts_and_free_text(text, expected):
    assert extract_complement_units(text) == expected


def test_hyphen_before_a_kind_word_does_not_glue_the_kind_into_the_value():
    assert extract_complement_units("BLQ 3-AP 201") == [("BLQ", "3"), ("AP", "201")]


def test_value_token_cap_keeps_long_runs_bounded():
    out = extract_complement_units("APTO " + " ".join(["1"] * 50))
    assert len(out) == 1 and out[0][0] == "AP"
    assert len(out[0][1]) <= 16


# ---------------------------------------------------------------------------
# Determinism and idempotence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    ["APTO 3-02", "TORRE B APTO 3 02", "TORRE F-2", "TORRE N 3", "TORRE LETRA A", "TORRE UNO A",
     "APTO 101 A", "TORRE II-A", "APTO 3-02 A", "TORRE A-B"],
)
def test_render_is_idempotent_and_merge_is_stable(text):
    rendered = render_complement(text)
    assert render_complement(rendered) == rendered
    assert render_complement(text) == rendered  # deterministic
    first = merge_complement(BASE, text)
    assert merge_complement(first.address, text).address == first.address
    assert merge_complement(first.address, rendered).address == first.address
