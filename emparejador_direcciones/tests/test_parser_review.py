"""Tests for the review findings on emparejador.parser (S1, S6)."""

from __future__ import annotations

import pytest

from emparejador import match, parse_canonical
from emparejador.parser import COMPLEMENT_KINDS, MAX_PLATE_DIGITS, QUADRANTS, clean_text

# ---------------------------------------------------------------------------
# S1: quadrant abbreviations after the removed dead guard
# ---------------------------------------------------------------------------


def test_single_letter_after_the_street_number_is_a_street_letter() -> None:
    parsed = parse_canonical("KR 41 E # 5 - 10")
    assert parsed.parse_ok is True
    assert parsed.via_letters == "E"
    assert parsed.via_quadrant is None


def test_single_letter_after_a_street_letter_is_a_quadrant() -> None:
    parsed = parse_canonical("C 70 B N # 5 - 10")
    assert parsed.parse_ok is True
    assert parsed.via_letters == "B"
    assert parsed.via_quadrant == "NORTE"


@pytest.mark.parametrize(
    "address,letters,quadrant",
    [
        ("KR 41 E S # 5 - 10", "E", "SUR"),
        ("KR 41 E O # 5 - 10", "E", "OESTE"),
        ("KR 41 E W # 5 - 10", "E", "OESTE"),
        ("KR 41 N # 5 - 10", "N", None),
        ("KR 41 S # 5 - 10", "S", None),
    ],
)
def test_quadrant_abbreviation_only_follows_another_segment_element(
    address: str, letters: str, quadrant: str | None
) -> None:
    parsed = parse_canonical(address)
    assert parsed.via_letters == letters
    assert parsed.via_quadrant == quadrant


def test_quadrant_abbreviation_after_bis_and_suffix_is_a_quadrant() -> None:
    assert parse_canonical("KR 41 BIS N # 5 - 10").via_quadrant == "NORTE"
    assert parse_canonical("K 3 A 3 N # 5 - 10").via_quadrant == "NORTE"


def test_quadrant_abbreviation_in_the_cross_segment() -> None:
    parsed = parse_canonical("KR 1 # 70 B N - 10")
    assert parsed.cross_letters == "B" and parsed.cross_quadrant == "NORTE"
    parsed = parse_canonical("KR 1 # 41 E - 10")
    assert parsed.cross_letters == "E" and parsed.cross_quadrant is None


def test_removed_plate_checks_rest_on_words_longer_than_one_letter() -> None:
    """The plate-letter code no longer re-checks quadrants/complement kinds:
    that is sound only while none of them is a single letter."""
    assert all(len(word) > 1 for word in QUADRANTS)
    assert all(len(kind) > 1 for kind in COMPLEMENT_KINDS)


def test_single_letter_after_plate_is_always_the_plate_letter() -> None:
    parsed = parse_canonical("KR 1 # 4 - 80 A")
    assert parsed.plate == "80A" and parsed.complement == ()
    parsed = parse_canonical("KR 1 # 4 - 80 N")
    assert parsed.plate == "80N" and parsed.complement == ()


def test_plate_cap_constant_is_six_and_enforced_at_the_boundary() -> None:
    assert MAX_PLATE_DIGITS == 6
    assert parse_canonical("KR 1 # 4 - " + "9" * MAX_PLATE_DIGITS).parse_ok is True
    rejected = parse_canonical("KR 1 # 4 - 1" + "0" * MAX_PLATE_DIGITS)
    assert rejected.parse_ok is False and rejected.notes == ("placa_invalida",)


# ---------------------------------------------------------------------------
# S6: what non-ASCII digits really do (NFKD folds compatibility digits;
# only digits without a decomposition are rejected)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compat_digit",
    ["²", "②", "２", "₂"],  # superscript, circled, fullwidth, subscript
)
def test_compatibility_digits_fold_to_ascii(compat_digit: str) -> None:
    folded = parse_canonical(f"KR {compat_digit} # 2 - 3")
    plain = parse_canonical("KR 2 # 2 - 3")
    assert folded.parse_ok is True
    assert folded.via_number == "2"
    assert match(f"KR {compat_digit} # 2 - 3", "KR 2 # 2 - 3").score == 1.0
    assert (folded.via_type, folded.cross_number, folded.plate) == (
        plain.via_type,
        plain.cross_number,
        plain.plate,
    )


def test_superscript_two_equals_ascii_two_in_the_plate_too() -> None:
    assert match("KR 1 # 4 - 8²", "KR 1 # 4 - 82").score == 1.0


@pytest.mark.parametrize(
    "digit",
    ["٣", "۳", "३"],  # Arabic-Indic, Extended Arabic-Indic, Devanagari
)
def test_digits_without_a_decomposition_are_rejected(digit: str) -> None:
    parsed = parse_canonical(f"KR {digit} # 4 - 80")
    assert parsed.parse_ok is False
    assert match(f"KR {digit} # 4 - 80", "KR 3 # 4 - 80").score == 0.0


def test_clean_text_leaves_undecomposable_digits_untouched() -> None:
    assert "٣" in clean_text("KR ٣ # 4 - 80")
