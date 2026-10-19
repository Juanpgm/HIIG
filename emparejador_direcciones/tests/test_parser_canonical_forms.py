"""Canonical forms emitted by the normalizer that the parser must read.

Audit findings M1/L9: ``TO A2`` (letter + digits as a complement value),
``4 ABIS`` (letter glued to BIS / quadrant), ``AGR``/``SEC`` complement kinds
and invisible characters. The invariant for every canonical address the
normalizer emits is: it parses, it renders back to something that parses to
the same fields, and it matches itself with score 1.0.
"""

from __future__ import annotations

import dataclasses

import pytest

from emparejador import match, parse_canonical, render
from emparejador.parser import COMPLEMENT_KINDS, clean_text


def _fields(address: str) -> dict[str, object]:
    data = dataclasses.asdict(parse_canonical(address))
    data.pop("raw")
    return data


# ---------------------------------------------------------------------------
# A: complement value made of a single via-type letter followed by digits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address,expected",
    [
        ("KR 987 # 654 C - 21 TO A2", (("TO", "A2"),)),
        ("KR 987 # 654 C - 21 TO C2", (("TO", "C2"),)),
        ("KR 987 # 654 C - 21 TO D12", (("TO", "D12"),)),
        ("KR 987 # 654 C - 21 BLQ K3", (("BLQ", "K3"),)),
        ("KR 987 # 654 C - 21 INT T1", (("INT", "T1"),)),
        ("KR 987 # 654 C - 21 TO A 2", (("TO", "A2"),)),
        ("KR 987 # 654 C - 21 TO A2 AP 301", (("AP", "301"), ("TO", "A2"))),
        ("KR 987 # 654 C - 21 AP 301 TO A2", (("AP", "301"), ("TO", "A2"))),
    ],
)
def test_letter_plus_digits_complement_value_is_read(address: str, expected: tuple) -> None:
    parsed = parse_canonical(address)
    assert parsed.parse_ok is True, parsed.notes
    assert parsed.complement == expected


@pytest.mark.parametrize(
    "address",
    [
        "KR 987 # 654 C - 21 TO A2",
        "KR 987 # 654 C - 21 BLQ K3 AP 301",
        "KR 987 # 654 C - 21 INT T1",
    ],
)
def test_letter_plus_digits_value_matches_itself_and_roundtrips(address: str) -> None:
    result = match(address, address)
    assert result.score == 1.0 and result.is_match is True
    rendered = render(parse_canonical(address))
    assert _fields(rendered) == _fields(address)
    assert match(rendered, address).score == 1.0


def test_tower_a2_differs_from_tower_a3_and_from_no_tower() -> None:
    a2 = "KR 987 # 654 C - 21 TO A2"
    assert match(a2, "KR 987 # 654 C - 21 TO A3").score == 0.0
    assert match(a2, "KR 987 # 654 C - 21").score == pytest.approx(0.98)


@pytest.mark.parametrize(
    "address",
    [
        "CL 5 # 10 - 20 CL 5",  # a real second address
        "CL 5 # 10 - 20 AP 3 CL 5",
        "CL 5 # 10 - 20 TO CL 5",  # multi-letter via type stays a second address
        "CL 5 # 10 - 20 80 A 2",
        "CL 5 # 10 - 20 3 A 2",
        "CL 5 # 10 - 20 TO A2 # 5 - 6",
    ],
)
def test_second_address_in_the_tail_is_still_rejected(address: str) -> None:
    parsed = parse_canonical(address)
    # Either rejected, or (only for the digit-led opaque shapes) never read
    # as a structured tower value.
    if parsed.parse_ok:
        assert ("TO", "A2") not in parsed.complement
        assert all(kind != "TO" or value != "CL/5" for kind, value in parsed.complement)
    else:
        assert parsed.notes


def test_bare_via_letter_with_number_without_a_kind_is_still_multiple_addresses() -> None:
    parsed = parse_canonical("KR 1 # 4 - 80 3 A 2")
    assert parsed.parse_ok is False
    assert parsed.notes == ("multiples_direcciones",)


@pytest.mark.parametrize(
    "address,expected",
    [
        ("CL 962 B # 641 A 9 - 80 AP 5A31", (("AP", "5A31"),)),
        ("CL 962 B # 641 A 9 - 80 AP 1C43", (("AP", "1C43"),)),
        ("AV 986 NORTE # 652 N - 24 AP 1204T4", (("AP", "1204T4"),)),
        ("KR 970 # 651 C - 70 AP 107A108A", (("AP", "107A108A"),)),
        ("CL 962 B # 641 A 9 - 80 TO C AP 1C54", (("AP", "1C54"), ("TO", "C"))),
        ("CL 962 B # 641 A 9 - 80 AP 3 A 2", (("AP", "3A2"),)),
        ("CL 910 # 653 - 59 ED SOL NC T1", (("ED", "SOL/NC/T1"),)),
    ],
)
def test_apartment_values_with_letters_between_digit_runs_are_read(address: str, expected: tuple) -> None:
    """Real normalizer output (`AP 2A41`, `AP 1204T4`): a lone via letter
    inside a known kind's value is part of the value, not a second address."""
    parsed = parse_canonical(address)
    assert parsed.parse_ok is True, parsed.notes
    assert parsed.complement == expected
    assert match(address, address).score == 1.0
    assert _fields(render(parsed)) == _fields(address)


def test_multi_letter_via_type_or_hash_inside_a_value_is_still_a_second_address() -> None:
    assert parse_canonical("CL 5 # 10 - 20 AP 3 CL 5").notes == ("multiples_direcciones",)
    assert parse_canonical("CL 5 # 10 - 20 AP 3 A 2 # 4 - 5").notes == ("multiples_direcciones",)


# ---------------------------------------------------------------------------
# B: letter glued to BIS / quadrant
# ---------------------------------------------------------------------------


def test_letter_glued_to_bis_in_the_cross_segment() -> None:
    parsed = parse_canonical("KR 937 # 654 ABIS - 49")
    assert parsed.parse_ok is True, parsed.notes
    assert parsed.cross_number == "654"
    assert parsed.cross_letters == "A"
    assert parsed.cross_bis is True
    assert _fields("KR 937 # 654 ABIS - 49") == _fields("KR 937 # 654 A BIS - 49")


def test_letter_glued_to_bis_in_the_via_segment_and_glued_digits() -> None:
    assert _fields("KR 937 ABIS # 654 - 49") == _fields("KR 937 A BIS # 654 - 49")
    assert _fields("KR 937 # 654ABIS - 49") == _fields("KR 937 # 654 A BIS - 49")
    assert parse_canonical("KR 937 BBIS # 654 - 49").via_letters == "B"


@pytest.mark.parametrize(
    "glued,spaced",
    [
        ("CL 5 ASUR # 4 - 49", "CL 5 A SUR # 4 - 49"),
        ("CL 5 BNORTE # 4 - 49", "CL 5 B NORTE # 4 - 49"),
        ("CL 5 AOESTE # 4 - 49", "CL 5 A OESTE # 4 - 49"),
        ("CL 5 AESTE # 4 - 49", "CL 5 A ESTE # 4 - 49"),
        ("CL 5 # 4 ASUR - 49", "CL 5 # 4 A SUR - 49"),
    ],
)
def test_letter_glued_to_quadrant(glued: str, spaced: str) -> None:
    assert parse_canonical(glued).parse_ok is True
    assert _fields(glued) == _fields(spaced)


def test_plain_quadrant_words_are_not_split() -> None:
    for word in ("OESTE", "ESTE", "SUR", "NORTE"):
        assert clean_text(f"CL 5 {word} # 4 - 49") == f"CL 5 {word} # 4 - 49"
    parsed = parse_canonical("CL 905 OESTE # 654 - 49")
    assert parsed.via_quadrant == "OESTE" and parsed.via_letters is None


def test_glued_bis_roundtrips_and_matches_itself() -> None:
    address = "KR 937 # 654 ABIS - 49"
    assert match(address, address).score == 1.0
    assert match(address, "KR 937 # 654 A BIS - 49").score == 1.0
    assert _fields(render(parse_canonical(address))) == _fields(address)


def test_other_glued_word_is_still_a_failure() -> None:
    parsed = parse_canonical("KR 937 # 654 AXYZ - 49")
    assert parsed.parse_ok is False


# ---------------------------------------------------------------------------
# C: AGR / SEC complement kinds
# ---------------------------------------------------------------------------


def test_agr_and_sec_are_registered_with_the_normalizer_canon() -> None:
    for alias in ("AGR", "AGRUPACION", "AGRUP", "AGP"):
        assert COMPLEMENT_KINDS[alias] == "AGR"
    for alias in ("SEC", "SECTOR", "SECT"):
        assert COMPLEMENT_KINDS[alias] == "SEC"


def test_agr_sec_tower_address_parses_and_matches_itself() -> None:
    address = "CL 962 B # 641 A 9 - 75 AGR 3 SEC 5 TO C"
    parsed = parse_canonical(address)
    assert parsed.parse_ok is True
    assert dict(parsed.complement) == {"AGR": "3", "SEC": "5", "TO": "C"}
    assert match(address, address).score == 1.0
    assert _fields(render(parsed)) == _fields(address)


def test_agr_sec_one_sided_costs_the_known_complement_price_not_the_opaque_one() -> None:
    bare = "CL 962 B # 641 A 9 - 75"
    # Before the fix AGR/SEC were opaque chunks: 0.85. Registered kinds cost 0.02.
    assert match(bare, bare + " AGR 3").score == pytest.approx(0.98)
    assert match(bare, bare + " SEC 5").score == pytest.approx(0.98)
    assert match(bare, bare + " AGR 3 SEC 5 TO C").score == pytest.approx(0.98)
    # An extra kind on one side while the other side has a complement is the
    # existing "difiere" veto, unchanged by registering the kinds.
    assert match(bare + " AGR 3 SEC 5 TO C", bare + " TO C").score == 0.0


def test_agr_spellings_are_the_same_complement() -> None:
    base = "CL 962 B # 641 A 9 - 75"
    assert match(base + " AGRUPACION 3", base + " AGR 3").score == 1.0
    assert match(base + " SECTOR 5", base + " SEC 5").score == 1.0
    assert match(base + " AGR 3", base + " AGR 4").score == 0.0


def test_kind_after_a_registered_kind_does_not_pollute_the_tower_value() -> None:
    parsed = parse_canonical("CL 962 B # 641 A 9 - 75 TO C AGR 3")
    assert dict(parsed.complement) == {"TO": "C", "AGR": "3"}
    # Before the fix the tower was read as "C/AGR3" and the pair was a veto for
    # the wrong reason; now the tower value is "C" on both sides.
    assert match("CL 962 B # 641 A 9 - 75 TO C AGR 3", "CL 962 B # 641 A 9 - 75 TO C AGR 3").score == 1.0
    assert match("CL 962 B # 641 A 9 - 75 TO C AGR 3", "CL 962 B # 641 A 9 - 75 TO C AGR 4").score == 0.0


def test_agr_without_value_is_still_opaque() -> None:
    from emparejador.matcher import _is_opaque_chunk

    assert _is_opaque_chunk("AGR", "") is True
    assert _is_opaque_chunk("SEC", "") is True


# ---------------------------------------------------------------------------
# D: zero-width / invisible characters
# ---------------------------------------------------------------------------

_INVISIBLES = ("​", "‌", "‍", "⁠", "﻿", "­", "‎", "‏")


@pytest.mark.parametrize("char", _INVISIBLES)
def test_invisible_character_at_the_end_does_not_break_parsing(char: str) -> None:
    assert parse_canonical("KR 1 # 4 - 80" + char).parse_ok is True
    assert match("KR 1 # 4 - 80" + char, "KR 1 # 4 - 80").score == 1.0


@pytest.mark.parametrize("char", _INVISIBLES)
def test_invisible_character_inside_and_between_tokens(char: str) -> None:
    assert _fields(f"KR{char} 1 #{char}4 - 8{char}0") == _fields("KR 1 # 4 - 80")
    assert _fields(f"{char}KR 1 # 4 - 80 TO{char} A{char}2") == _fields("KR 1 # 4 - 80 TO A2")


def test_only_invisible_characters_is_an_empty_input() -> None:
    parsed = parse_canonical("​​﻿")
    assert parsed.parse_ok is False
    assert parsed.notes == ("entrada_vacia",)


def test_clean_text_strips_format_characters_but_keeps_real_content() -> None:
    assert clean_text("kr​ 1 # 4 - 80‍") == "KR 1 # 4 - 80"
    assert clean_text("CL 5 # 10 - 20") == "CL 5 # 10 - 20"
