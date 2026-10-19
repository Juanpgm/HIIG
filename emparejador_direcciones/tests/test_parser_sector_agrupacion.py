"""I2: ``SECTOR`` / ``AGRUPACION`` (and ``SECT`` / ``AGRUP``) are "head-only" kinds.

Registering AGR/SEC made every raw word ``SECTOR`` / ``AGRUPACION`` a complement
kind, which split names such as ``ED SECTOR NORTE`` into ``ED ''`` + ``SEC NORTE``.
The long aliases are now a kind only when they head a SHORT value (one or two
tokens of at most 3 characters, or digits, then the end or another kind) and do
not directly follow a name-bearing kind (ED, CONJ, UR, BR, CGTO, VDA). The
canonical codes ``AGR`` / ``SEC`` / ``AGP`` stay kinds as the normalizer emits them.
"""

from __future__ import annotations

import pytest

from emparejador import match, parse_canonical

BASE = "CL 987 # 654 - 21"


def _complement(tail: str):
    parsed = parse_canonical(f"{BASE} {tail}")
    assert parsed.parse_ok is True, parsed.notes
    return parsed.complement


@pytest.mark.parametrize(
    "tail,expected_complement,score,is_match",
    [
        # a real sector designation vs the bare address: one-sided known kind, soft 0.02
        ("SECTOR B", (("SEC", "B"),), 0.98, True),
        # name words after a name-bearing kind belong to the name
        ("CONJ AGRUPACION LOS PINOS", (("CONJ", "AGRUPACION/LOS/PINOS"),), 0.98, True),
        ("ED SECTOR NORTE", (("ED", "SECTOR/NORTE"),), 0.98, True),
        ("BR EL SECTOR", (("BR", "EL/SECTOR"),), 0.98, True),
    ],
)
def test_the_four_measured_examples_are_pinned(tail, expected_complement, score, is_match):
    assert _complement(tail) == expected_complement
    result = match(f"{BASE} {tail}", BASE)
    assert result.score == pytest.approx(score)
    assert result.is_match is is_match
    assert match(BASE, f"{BASE} {tail}").score == pytest.approx(score)


@pytest.mark.parametrize("tail", ["ED SECTOR 5", "UR AGRUPACION 2", "VDA SECTOR B", "CGTO AGRUPACION 3"])
def test_long_alias_right_after_a_name_kind_is_part_of_the_name(tail):
    kinds = {kind for kind, _ in _complement(tail)}
    assert kinds == {tail.split()[0]}


@pytest.mark.parametrize("tail", ["TO AGRUPACION", "AP 5 SECTOR", "SECTOR", "AGRUPACION"])
def test_long_alias_without_a_value_is_not_a_kind(tail):
    assert {kind for kind, _ in _complement(tail)}.isdisjoint({"SEC", "AGR"})


@pytest.mark.parametrize("tail", ["SECTOR NORTE", "SECTOR LAGUNAS", "AGRUPACION LOS PINOS"])
def test_long_alias_followed_by_a_name_stays_an_unstructured_chunk(tail):
    kinds = {kind for kind, _ in _complement(tail)}
    assert kinds.isdisjoint({"SEC", "AGR"})
    # scored exactly like any other unstructured alphabetic tail (here a made-up first word)
    control = "ZZZ " + tail.split(" ", 1)[1]
    assert match(f"{BASE} {tail}", BASE).score == match(f"{BASE} {control}", BASE).score


@pytest.mark.parametrize(
    "tail,expected",
    [
        ("SECTOR 5", (("SEC", "5"),)),
        ("SECTOR 1234", (("SEC", "1234"),)),
        ("AGRUPACION 3 A", (("AGR", "3A"),)),
        ("SECTOR B TO C", (("SEC", "B"), ("TO", "C"))),
        ("TO C SECTOR B", (("SEC", "B"), ("TO", "C"))),
        ("AGRUP 7", (("AGR", "7"),)),
        ("SECT 2", (("SEC", "2"),)),
    ],
)
def test_long_alias_with_a_short_value_is_still_a_kind(tail, expected):
    assert _complement(tail) == expected


def test_canonical_codes_stay_kinds_in_every_position():
    assert _complement("AGR 3 SEC 5 TO C") == (("AGR", "3"), ("SEC", "5"), ("TO", "C"))
    assert _complement("CONJ LOS PINOS AGR 3") == (("AGR", "3"), ("CONJ", "LOS/PINOS"))
    assert _complement("ED SEC 5") == (("ED", ""), ("SEC", "5"))
    assert _complement("AGP 4") == (("AGR", "4"),)


def test_words_inside_another_kinds_value_do_not_change_the_comparison():
    with_word = f"{BASE} TO A AGRUPACION"
    assert parse_canonical(with_word).parse_ok is True
    assert match(with_word, with_word).score == 1.0
    assert match(f"{BASE} CONJ AGRUPACION LOS PINOS", f"{BASE} CONJ AGRUPACION LOS PINOS").score == 1.0
    assert match(f"{BASE} CONJ AGRUPACION LOS PINOS", f"{BASE} CONJ AGRUPACION LAS PALMAS").score == 0.0   # two names: veto


def test_same_sector_in_two_spellings_is_the_same_complement_but_different_sectors_veto():
    assert match(f"{BASE} SECTOR B", f"{BASE} SEC B").score == 1.0
    assert match(f"{BASE} SECTOR B", f"{BASE} SECTOR C").score == 0.0
