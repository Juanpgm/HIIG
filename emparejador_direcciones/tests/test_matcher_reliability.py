"""Matcher reliability: identical strings and quadrant placement.

``match(x, x)`` must be a MATCH even when ``x`` cannot be parsed because of a long
complement or several addresses, and a quadrant written as a letter, spelled out, on the
via or on the cross is the same quadrant. Plate / via / cross hard vetoes are untouched.
"""

from __future__ import annotations

import pytest

from emparejador.matcher import DEFAULT_PLATE_TOLERANCE, match, match_parsed
from emparejador.parser import parse_canonical

LONG_COMPLEMENT = "AP 5 TO 3 BLQ 2 CA 4 LT 5 MZ 7 BG 2 ET 3 INT 4 PISO 2 PH 1 GA 2 PQ 3 OF 4 LC 5"
MULTI = "KR 987 # 654 - 10; KR 987 # 654 - 12"


# ---------------------------------------------------------------------------
# 1. Identical strings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [
    f"KR 987 # 654 - 10 {LONG_COMPLEMENT}",
    MULTI,
    "CL 987 # 654 - 10 " + "TORRE ALAMEDA DEL PARQUE " * 12,        # > 300 chars
    "KR 987 # 654 - 10 // KR 988 # 655 - 11",                        # '//' separated
    "KR 987 # 654 - 10 Y KR 988 # 655 - 11",                         # 'Y' separated
])
def test_identical_unparseable_address_strings_match(x):
    assert parse_canonical(x).parse_ok is False  # precondition: the parser alone rejects it
    result = match(x, x)
    assert result.is_match is True
    assert result.score == 1.0


def test_identical_strings_match_ignoring_case_and_whitespace():
    result = match(MULTI, "  " + MULTI.lower().replace("; ", " ;   ") + " ")
    assert result.is_match is True


def test_identical_match_respects_a_threshold_above_one_is_invalid_not_forced():
    # the threshold validation still runs first
    from emparejador.matcher import InvalidThresholdError
    with pytest.raises(InvalidThresholdError):
        match(MULTI, MULTI, threshold=1.5)


def test_match_parsed_applies_the_same_short_circuit():
    a, b = parse_canonical(MULTI), parse_canonical(MULTI)
    assert match_parsed(a, b).is_match is True


@pytest.mark.parametrize("a, b", [
    (MULTI, "KR 987 # 654 - 10; KR 987 # 654 - 13"),                  # one digit differs
    (f"KR 987 # 654 - 10 {LONG_COMPLEMENT}", f"KR 987 # 654 - 11 {LONG_COMPLEMENT}"),
])
def test_non_identical_unparseable_strings_do_not_match(a, b):
    assert match(a, b).is_match is False


@pytest.mark.parametrize("x", [
    "SIN ESPECIFICAR", "", "   ", "Cali", "NO APLICA; NO REGISTRA", "ABC", "CL ABC # DEF - GHI",
    "KR 987 # 654 - 10 # 5",             # a dangling second '#': junk, not two addresses
    "KR 987 # 654 - 10; ABC; 12",        # a segment that is not an address
    "KR 987 # 654 - 10 AP 5 # KR 988 # 655 - 11",
])
def test_identical_garbage_never_matches(x):
    result = match(x, x)
    assert result.is_match is False
    assert result.score == 0.0


@pytest.mark.parametrize("x", [None, 123, 4.5, [], {}])
def test_identical_non_strings_never_match(x):
    assert match(x, x).is_match is False


def test_identical_parseable_strings_still_match_exactly():
    result = match("KR 987 # 654 - 10", "KR 987 # 654 - 10")
    assert result.is_match is True and result.reason == "coincidencia exacta"


# ---------------------------------------------------------------------------
# 2. Quadrant spelling inside ONE segment (letter == spelled quadrant of that segment)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a, b", [
    ("AV 987 N # 543 - 06", "AV 987 NORTE # 543 - 06"),        # letter vs spelled, via
    ("AV 987 NORTE # 543 - 06", "AV 987 N # 543 - 06"),        # symmetric
    ("AV 987 # 543 N - 06", "AV 987 # 543 NORTE - 06"),        # letter vs spelled, cross
    ("CL 987 O # 654 - 21", "CL 987 OESTE # 654 - 21"),        # O == OESTE
    ("CL 987 W # 654 - 21", "CL 987 OESTE # 654 - 21"),        # W == OESTE (existing convention)
    ("CL 987N # 654 - 21", "CL 987 NORTE # 654 - 21"),         # glued 987N
    ("CL 987 D N # 654 - 21", "CL 987 D NORTE # 654 - 21"),    # street letter kept, quadrant folded
    ("AV 987 NORTE # 543 - 06", "AV 987 NORTE # 543 - 06"),
])
def test_same_segment_quadrant_spelling_is_equivalent(a, b):
    result = match(a, b)
    assert result.is_match is True, result.reason
    assert result.score == 1.0


# A quadrant on the via and a quadrant on the cross street name DIFFERENT places
# (``CL 5 N # 10 - 20`` is Calle 5 Norte, ``CL 5 # 10 N - 20`` crosses Carrera 10 Norte).
@pytest.mark.parametrize("a, b", [
    ("CL 987 N # 654 - 21", "CL 987 # 654 N - 21"),
    ("CL 987 NORTE # 654 - 21", "CL 987 # 654 NORTE - 21"),
    ("CL 987 # 654 NORTE - 21", "CL 987 NORTE # 654 - 21"),
    ("AV 987 # 543 N - 06", "AV 987 NORTE # 543 - 06"),
    ("CL 987 B N # 654 - 21", "CL 987 B # 654 NORTE - 21"),
    ("CL 987 NORTE # 654 SUR - 21", "CL 987 SUR # 654 NORTE - 21"),  # swapped
    ("CL 987 N # 654 N - 21", "CL 987 NORTE # 654 - 21"),            # extra cross quadrant
])
def test_via_and_cross_quadrants_never_match_each_other(a, b):
    for x, y in ((a, b), (b, a)):
        result = match(x, y)
        assert result.is_match is False, result.reason
        assert 0.0 < result.score < 0.9
        assert "cuadrante_plegado" not in result.warnings


def test_via_vs_cross_quadrant_keeps_the_original_score():
    assert match("CL 987 N # 654 - 21", "CL 987 # 654 N - 21").score == 0.7225


@pytest.mark.parametrize("a, b", [
    ("AV 987 N # 543 - 06", "AV 987 OESTE # 543 - 06"),        # different quadrants
    ("AV 987 NORTE # 543 - 06", "AV 987 SUR # 543 - 06"),
    ("AV 987 # 543 - 06", "AV 987 NORTE # 543 - 06"),           # quadrant on one side only
    ("AV 987 # 543 N - 06", "AV 987 # 543 - 06"),
    ("KR 987 S # 543 - 06", "KR 987 SUR # 543 - 06"),           # S stays a street letter
    ("KR 987 E # 543 - 06", "KR 987 # 543 - 06"),
    ("KR 987 D N # 543 - 06", "KR 987 NORTE # 543 - 06"),       # street letter D differs
    ("CL 987 W # 654 - 21", "CL 987 O # 654 - 21"),             # two bare letters: as before
])
def test_quadrant_differences_still_penalized(a, b):
    assert match(a, b).is_match is False


@pytest.mark.parametrize("a, b", [
    ("AV 987 N # 543 - 06", "AV 987 NORTE # 543 - 07"),         # plate differs by 1
    ("AV 987 N # 543 - 06", "AV 987 NORTE # 543 - 08"),         # by 2
    ("AV 987 N # 543 - 06", "AV 988 NORTE # 543 - 06"),         # via number
    ("AV 987 N # 543 - 06", "AV 987 NORTE # 544 - 06"),         # cross number
    ("AV 987 N # 543 - 06", "KR 987 NORTE # 543 - 06"),         # via type
    ("AV 987 N # 543 - 06", "AV 987 NORTE # 543"),              # plate on one side only
    ("AV 987 N # 543 - 06", "AV 987 NORTE # 543 - 06 A"),       # plate letter
])
def test_hard_vetoes_are_not_loosened_by_the_letter_fold(a, b):
    result = match(a, b)
    assert result.is_match is False
    assert result.score == 0.0
    assert DEFAULT_PLATE_TOLERANCE == 0


def test_letter_fold_does_not_change_the_returned_parsed_sides():
    result = match("AV 987 N # 543 - 06", "AV 987 NORTE # 543 - 06")
    assert result.parsed_a.via_letters == "N" and result.parsed_a.via_quadrant is None
    assert result.parsed_b.via_quadrant == "NORTE"


def test_quadrant_equivalence_is_symmetric_and_deterministic():
    a, b = "AV 987 # 543 N - 06", "AV 987 # 543 NORTE - 06"
    assert match(a, b).score == match(b, a).score == match(a, b).score


def test_one_sided_letter_keeps_the_original_reason():
    # no spelled quadrant on the other side: nothing to fold, same text as before
    assert "via_letters" in match("CL 987 # 654 - 21", "CL 987 N # 654 - 21").reason


# ---------------------------------------------------------------------------
# 3. The N/NORTE equivalence needs the same segment, the same position and an identical prefix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a, b", [
    ("CL 987 N # 654 - 21", "CL 987 NORTE # 654 - 21"),
    ("CL 987 # 654 N - 21", "CL 987 # 654 NORTE - 21"),
    ("AV 876 N # 543 - 06", "AV 876 NORTE # 543 - 06"),
    ("AV 876 O # 543 - 06", "AV 876 OESTE # 543 - 06"),
    ("AV 876 # 543 O - 06", "AV 876 # 543 OESTE - 06"),
])
def test_n_norte_with_identical_prefix_is_equivalent(a, b):
    assert match(a, b).is_match is True


@pytest.mark.parametrize("a, b", [
    ("CL 987 N # 654 - 21", "CL 986 NORTE # 654 - 21"),        # via number differs
    ("CL 987 N # 654 - 21", "KR 987 NORTE # 654 - 21"),        # via type differs
    ("CL 987 A N # 654 - 21", "CL 987 NORTE # 654 - 21"),      # via letters differ
    ("CL 987 # 654 N - 21", "CL 987 # 655 NORTE - 21"),        # cross number differs
    ("AV 876 N # 543 - 06", "CL 876 NORTE # 543 - 06"),        # via type differs
    ("AV 876 # 543 N - 06", "AV 876 NORTE # 543 - 06"),        # position: cross vs via
    ("AV 876 # 543 NORTE - 06", "AV 876 N # 543 - 06"),
    ("AV 876 # 543 O - 06", "AV 876 OESTE # 543 - 06"),
    ("AV 876 O # 543 - 06", "AV 876 # 543 OESTE - 06"),
])
def test_n_norte_does_not_hide_a_different_prefix_or_position(a, b):
    for x, y in ((a, b), (b, a)):
        assert match(x, y).is_match is False


# ---------------------------------------------------------------------------
# 4. The fold needs an identical segment prefix, at every threshold
# ---------------------------------------------------------------------------
_PREFIX_DIFFERS = [
    ("CL 987 N # 654 - 21", "CL 987 A NORTE # 654 - 21"),          # via letters differ
    ("CL 987 N # 654 - 21", "CL 987 BIS NORTE # 654 - 21"),        # BIS on one side
    ("CL 987 N # 654 - 21", "CL 987 2 NORTE # 654 - 21"),          # numeric suffix on one side
    ("CL 987 N # 654 N - 21", "CL 987 NORTE # 654 - 21"),          # the cross segment cannot fold
    ("CL 987 N # 654 - 21", "CL 987 NORTE # 654 B - 21"),          # cross letters on one side
    ("CL 987 B N # 654 - 21", "CL 987 A NORTE # 654 - 21"),        # different letters
    ("CL 987 # 654 N - 21", "CL 987 # 654 A NORTE - 21"),
    ("CL 987 # 654 N - 21", "CL 987 # 654 BIS NORTE - 21"),
]


@pytest.mark.parametrize("threshold", [0.85, 0.90, 0.95])
@pytest.mark.parametrize("a, b", _PREFIX_DIFFERS)
def test_fold_never_applies_across_a_different_prefix(a, b, threshold):
    for x, y in ((a, b), (b, a)):
        result = match(x, y, threshold=threshold)
        assert result.is_match is False, (result.score, result.reason)
        assert result.score < 0.85, result.score        # the pre-change score, not 0.85


@pytest.mark.parametrize("threshold", [0.85, 0.90, 0.95, 1.0])
@pytest.mark.parametrize("a, b", [
    ("CL 987 N # 654 - 21", "CL 987 NORTE # 654 - 21"),
    ("CL 987 # 654 N - 21", "CL 987 # 654 NORTE - 21"),
    ("CL 987 B N # 654 - 21", "CL 987 B NORTE # 654 - 21"),
    ("AV 876 O # 543 - 06", "AV 876 OESTE # 543 - 06"),
    ("CL 987 N # 654 N - 21", "CL 987 NORTE # 654 NORTE - 21"),    # both segments fold
])
def test_fold_with_identical_prefixes_matches_at_every_threshold(a, b, threshold):
    assert match(a, b, threshold=threshold).is_match is True


@pytest.mark.parametrize("threshold", [0.85, 0.90, 0.95])
def test_via_vs_cross_quadrant_stays_below_every_threshold(threshold):
    assert match("CL 987 N # 654 - 21", "CL 987 # 654 N - 21", threshold=threshold).is_match is False
