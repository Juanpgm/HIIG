"""Unit values glued to words (``AP 501TORRED``) must not become unit codes.

A legit code is digits plus at most a short letter suffix/prefix (``501A``, ``12B``,
``2AB``, ``1C``). Digits glued to a registered kind word are split at the kind
(``501TORRED`` -> ``501`` + ``TORRE D``); digits glued to any other word keep the
digits and drop the word (``501UMBRAL`` -> ``501``). The scoring complement is not
touched by any of this (see test_complement_scoring_invariance.py).
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import (  # noqa: E402
    extract_complement_units,
    merge_complement,
    parse_address,
    render_complement,
)

BASE = "KR 29 A # 36 - 87"


# ---------------------------------------------------------------------------
# Legit codes keep passing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("AP 501A", [("AP", "501A")]),
        ("AP 501 A", [("AP", "501A")]),
        ("CASA 12B", [("CA", "12B")]),
        ("LC 2AB", [("LC", "2AB")]),
        ("TO A1", [("TO", "A1")]),
        ("AP 1C 23", [("AP", "1C")]),
        ("AP 0501A", [("AP", "501A")]),
        ("OF 301", [("OF", "301")]),
    ],
)
def test_digit_plus_short_letter_codes_are_accepted(text, expected):
    assert extract_complement_units(text) == expected


# ---------------------------------------------------------------------------
# Digits glued to a kind word: split at the kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("AP 501TORRED", [("AP", "501"), ("TO", "D")]),
        ("AP 501TorreD", [("AP", "501"), ("TO", "D")]),
        ("ap 501torred", [("AP", "501"), ("TO", "D")]),
        ("AP 501TORRE D", [("AP", "501"), ("TO", "D")]),
        ("AP 501TO D", [("AP", "501"), ("TO", "D")]),
        ("AP 501TORRE", [("AP", "501")]),
        ("PISO 5TO", [("PISO", "5")]),
        ("CASA 12BLOQUEC", [("CA", "12"), ("BLQ", "C")]),
    ],
)
def test_digits_glued_to_a_kind_word_are_split(text, expected):
    assert extract_complement_units(text) == expected


# ---------------------------------------------------------------------------
# Digits glued to any other word: the word is never part of the value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("AP 501UMBRAL", [("AP", "501")]),
        ("AP 501TorreD PortalDeLasPalmas", [("AP", "501"), ("TO", "D")]),
        ("CASA 2PISOS", [("CA", "2")]),
        ("CASA DE 2PISOS", []),
        ("TO 3TORRES", [("TO", "3")]),
        ("AP 501TORREDELSOL", [("AP", "501")]),
    ],
)
def test_digits_glued_to_other_words_keep_only_the_digits(text, expected):
    assert extract_complement_units(text) == expected


@pytest.mark.parametrize("token", ["5B1TORRE", "A1TORRE", "1TORRE2", "12ABC3"])
def test_mixed_tokens_with_a_long_letter_run_are_not_codes(token):
    assert extract_complement_units(f"AP {token}") == []


# ---------------------------------------------------------------------------
# Unicode, case, long and degenerate inputs
# ---------------------------------------------------------------------------
def test_accented_glued_word_is_split_after_accent_stripping():
    assert extract_complement_units("AP 501TÓRRED") == [("AP", "501"), ("TO", "D")]


def test_non_latin_letters_never_join_a_code():
    # Non A-Z letters are separators after accent stripping.
    assert extract_complement_units("AP 501Ж") == [("AP", "501")]


def test_very_long_glued_token_is_bounded_and_rejected():
    start = time.perf_counter()
    assert extract_complement_units("AP 501" + "X" * 5000) == [("AP", "501")]
    assert extract_complement_units("AP " + "1" * 5000 + "TORRED") == [("TO", "D")]
    assert time.perf_counter() - start < 2.0


def test_many_glued_tokens_stay_fast():
    text = " ".join(["AP 501TORRED"] * 5000)
    start = time.perf_counter()
    assert extract_complement_units(text) == [("AP", "501"), ("TO", "D")]
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("text", ["501TORRED", "TORRED", "501", "  ", ""])
def test_glued_text_without_a_leading_kind_yields_no_bogus_unit(text):
    out = extract_complement_units(text)
    assert all(v != "501TORRED" for _, v in out)
    assert ("AP", "501TORRED") not in out


# ---------------------------------------------------------------------------
# merge_complement / render_complement / parsed rows
# ---------------------------------------------------------------------------
def test_merge_splits_glued_value():
    m = merge_complement(BASE, "AP 501TORRED")
    assert m.address == f"{BASE} TO D AP 501"
    assert m.conflicts == ()


def test_merge_glued_value_matches_cadastral_chunk_without_conflict():
    m = merge_complement(f"{BASE} AP 501", "AP 501TORRED")
    assert m.address == f"{BASE} TO D AP 501"
    assert m.conflicts == ()


def test_render_complement_of_real_glued_input():
    p = parse_address("CL 13 A # 74 - 22 AP 501TorreD PortalDeLasPalmas")
    assert render_complement(p.complement_text) == "TO D AP 501"


def test_glued_split_does_not_touch_the_scoring_complement():
    p = parse_address("CL 13 A # 74 - 22 AP 501TorreD PortalDeLasPalmas")
    assert p.complement == "AP 501TORRED PORTALDELASPALMAS"
