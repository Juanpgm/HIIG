"""Parser vocabulary growth: extra aliases and the no-'#' address form."""

from __future__ import annotations

import time

import pytest

from emparejador import match, parse_canonical
from emparejador.parser import COMPLEMENT_KINDS, VIA_TYPES, clean_text, render

CANON = "CL 5 # 10 - 20"

# ---------------------------------------------------------------------------
# Street-type aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,code",
    [
        ("CARR", "KR"),
        ("CRRA", "KR"),
        ("AVDA", "AV"),
        ("AVEN", "AV"),
        ("AVE", "AV"),
        ("DIAG", "DG"),
        ("DIAGO", "DG"),
        ("TRANSV", "TV"),
        ("TRV", "TV"),
    ],
)
def test_new_via_type_alias_parses_to_canonical_type(alias: str, code: str) -> None:
    parsed = parse_canonical(f"{alias} 5 # 10 - 20")
    assert parsed.parse_ok is True
    assert parsed.via_type == code
    assert match(f"{alias} 5 # 10 - 20", f"{code} 5 # 10 - 20").score == 1.0


def test_new_via_alias_works_as_cross_type() -> None:
    parsed = parse_canonical("CL 5 # AVDA 10 - 20")
    assert parsed.parse_ok is True
    assert parsed.cross_type == "AV"


@pytest.mark.parametrize("alias", ["CIR", "CIRC", "CIRCULAR"])
def test_ambiguous_circular_alias_is_not_added(alias: str) -> None:
    # circular vs circunvalar: no unambiguous canonical form, so not guessed.
    assert alias not in VIA_TYPES
    assert parse_canonical(f"{alias} 5 # 10 - 20").parse_ok is False


# ---------------------------------------------------------------------------
# Complement aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,code",
    [
        ("APT", "AP"),
        ("APART", "AP"),
        ("LOC", "LC"),
        ("OFIC", "OF"),
        ("EDIF", "ED"),
        ("EDF", "ED"),
        ("BLOQ", "BLQ"),
    ],
)
def test_new_complement_alias_maps_to_canonical_kind(alias: str, code: str) -> None:
    assert COMPLEMENT_KINDS[alias] == code
    parsed = parse_canonical(f"{CANON} {alias} 301")
    assert parsed.parse_ok is True
    assert parsed.complement == ((code, "301"),)
    assert match(f"{CANON} {alias} 301", f"{CANON} {code} 301").score == 1.0


@pytest.mark.parametrize("text", ["CL 5 # 10 - 20 APT. 301", "CL 5 # 10 - 20 LOC. 2", "CL 5 # 10 - 20 apt 301"])
def test_complement_alias_with_dot_and_case(text: str) -> None:
    assert parse_canonical(text).parse_ok is True
    assert parse_canonical(text).complement[0][0] in {"AP", "LC"}


def test_complement_alias_in_plate_slot() -> None:
    parsed = parse_canonical("CL 5 # 10 - LOC 3")
    assert parsed.parse_ok is True
    assert parsed.plate is None
    assert parsed.complement == (("LC", "3"),)


# ---------------------------------------------------------------------------
# Aliases embedded in longer words are never rewritten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["LOCALIDAD", "CAPARTIDA", "APTITUD", "EDIFICACION", "BLOQUEO", "OFICIAL", "NUMEROSO"])
def test_alias_inside_longer_word_is_not_touched(word: str) -> None:
    assert word not in COMPLEMENT_KINDS
    assert word not in VIA_TYPES
    assert clean_text(f"CL 5 # 10 - 20 {word}") == f"CL 5 # 10 - 20 {word}"
    parsed = parse_canonical(f"{CANON} {word} 5")
    assert parsed.parse_ok is True
    assert parsed.complement == ((word, "5"),)


def test_alias_glued_to_a_longer_word_is_an_opaque_chunk() -> None:
    parsed = parse_canonical("CL 5 # 10 - 20 LOCAL 4")
    assert parsed.complement == (("LC", "4"),)  # existing alias, unchanged
    assert parse_canonical("CL 5 # 10 - 20 LOCALES 4").complement == (("LOCALES", "4"),)


# ---------------------------------------------------------------------------
# NUMERO as a '#' equivalent
# ---------------------------------------------------------------------------


def test_numero_word_is_a_hash_marker() -> None:
    assert clean_text("CL 5 NUMERO 10 - 20") == CANON
    assert render(parse_canonical("CL 5 numero 10 - 20")) == CANON
    assert parse_canonical("CL 5 NUMERO 10 - 20").parse_ok is True
    assert match("CL 5 NUMERO 10 - 20", CANON).score == 1.0


def test_numero_in_complement_does_not_collide_with_a_real_hash() -> None:
    # A real '#' already exists: NUMERO inside the tail is plain text, not a
    # second address separator.
    parsed = parse_canonical("CL 5 # 10 - 20 LOTE NUMERO 3")
    assert parsed.parse_ok is True
    assert parsed.plate == "20"


def test_two_numero_words_are_not_two_hashes() -> None:
    parsed = parse_canonical("CL 5 NUMERO 10 - 20 LOTE NUMERO 3")
    assert parsed.parse_ok is True
    assert parsed.cross_number == "10"


# ---------------------------------------------------------------------------
# No-'#' form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "no_hash,with_hash",
    [
        ("CL 5 10 - 20", "CL 5 # 10 - 20"),
        ("CL 5 10 - 20 APT 301", "CL 5 # 10 - 20 AP 301"),
        ("KR 100 3 - 07", "KR 100 # 3 - 07"),
        ("cl 5 10 - 20", "CL 5 # 10 - 20"),
        ("CL5 10 - 20", "CL 5 # 10 - 20"),
        ("CL 5 10 – 20", "CL 5 # 10 - 20"),  # en dash
        ("CL 5 10 - 20", "CL 5 # 10 - 20"),  # NBSP
        ("CL 5 10 - 20 A", "CL 5 # 10 - 20 A"),
        ("K 5 10 - 20", "KR 5 # 10 - 20"),
    ],
)
def test_no_hash_form_equals_hash_form(no_hash: str, with_hash: str) -> None:
    a, b = parse_canonical(no_hash), parse_canonical(with_hash)
    assert a.parse_ok is True
    assert render(a) == render(b)
    assert match(no_hash, with_hash).score == 1.0
    assert match(with_hash, no_hash).score == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "KR 26 G 5 73 - 13",
        "CL 72 L 3 B NORTE - 14",
        "CL 5 A 10 - 20",
        "CL 5 10 A - 20",
        "KR 26 5 G 73 - 13",  # letter after the cross: dash is not at tokens[3]
        "CL 5 10 A 3 - 4",  # several tokens before the dash
        "CL 5 10 20",  # no dash at all
        "CL 5 10 -",  # empty plate
        "CL 5 - 20",  # a single number: nothing to split
        "5 10 - 20",  # no street type
        "CL 5 10 - ",
        "CL 5 10 - APT 3",  # no plate
        "CL 5 6 10 - 20",  # three numbers: ambiguous split
        "CL 5 BIS 10 - 20",
        "CL 5 10 - 20 - 30 - 40",
    ],
)
def test_ambiguous_no_hash_strings_stay_unparseable(text: str) -> None:
    parsed = parse_canonical(text)
    assert parsed.parse_ok is False
    assert match(text, CANON).score == 0.0
    assert match(CANON, text).score == 0.0


def test_no_hash_rule_never_overrides_an_explicit_hash() -> None:
    parsed = parse_canonical("CL 5 # 10 - 20")
    assert parsed.parse_ok is True and parsed.cross_number == "10"


@pytest.mark.parametrize("plate", ["0", "00", "000"])
def test_no_hash_plate_zero(plate: str) -> None:
    a = parse_canonical(f"CL 5 10 - {plate}")
    b = parse_canonical(f"CL 5 # 10 - {plate}")
    assert a.parse_ok is True
    assert render(a) == render(b)
    assert match(f"CL 5 10 - {plate}", "CL 5 # 10 - 00").score == 1.0


def test_no_hash_different_plate_or_cross_does_not_match() -> None:
    assert match("CL 5 10 - 20", "CL 5 # 10 - 21").is_match is False
    assert match("CL 5 10 - 20", "CL 5 # 11 - 20").is_match is False


# ---------------------------------------------------------------------------
# Edge inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, 5, 3.2, b"CL 5 10 - 20", [], "", "   ", " 　"])
def test_empty_and_non_string_inputs_never_raise(value: object) -> None:
    assert parse_canonical(value).parse_ok is False


def test_fullwidth_and_unicode_variants() -> None:
    # NFKD folds full-width forms into ASCII; the result must equal the ASCII parse.
    full = "ＣＬ ５ ＃ １０ － ２０"
    assert parse_canonical(full).parse_ok is True
    assert match(full, CANON).score == 1.0
    assert parse_canonical("ＣＬ ５ １０ － ２０").parse_ok is True


def test_huge_inputs_stay_bounded_and_fail_cleanly() -> None:
    start = time.perf_counter()
    for text in ("CL 5 10 - 20 " + "APT 1 " * 50_000, "CL " + "5 " * 100_000 + "- 20", "CL 5 10 - " + "9" * 100_000):
        assert parse_canonical(text).parse_ok is False
    assert time.perf_counter() - start < 5.0


def test_alias_collisions_between_tables_are_absent() -> None:
    # A token cannot be a via type AND a complement kind with a different meaning.
    assert set(VIA_TYPES) & set(COMPLEMENT_KINDS) == set()


def test_all_aliases_point_to_canonical_codes() -> None:
    assert set(VIA_TYPES.values()) <= set(VIA_TYPES)
    assert set(COMPLEMENT_KINDS.values()) <= set(COMPLEMENT_KINDS)


# ---------------------------------------------------------------------------
# Idempotence, symmetry, threshold
# ---------------------------------------------------------------------------

_SAMPLE = [
    "CL 5 # 10 - 20",
    "KR 26 G # 5 - 73 LC 2",
    "CL 72 L # 3 B NORTE - 14",
    "AV 6 N # 28 N - 10 OF 301",
    "DG 15 # 71 A - 08 AP 101 TO 2",
    "CL 5 10 - 20 APT 301",
    "AVDA 5 NUMERO 10 - 20 EDIF SOL",
    "C 5 # K 10 - 20",
    "CL 5 # 10 -",
    "CL 5 # 10 - LT 4",
]


@pytest.mark.parametrize("text", _SAMPLE)
def test_render_is_idempotent(text: str) -> None:
    once = render(parse_canonical(text))
    twice = render(parse_canonical(once))
    assert once == twice
    assert parse_canonical(once).parse_ok is True


@pytest.mark.parametrize("a", _SAMPLE)
def test_match_is_symmetric(a: str) -> None:
    for b in _SAMPLE + ["KR 26 G 5 73 - 13", "garbage", ""]:
        assert match(a, b).score == match(b, a).score
        assert match(a, b).is_match == match(b, a).is_match


def test_threshold_boundary_with_new_forms() -> None:
    # Identical after alias mapping: score 1.0, matches at threshold 1.0.
    assert match("CL 5 10 - 20 APT 3", "CL 5 # 10 - 20 AP 3", threshold=1.0).is_match is True
    # One-sided complement stays a soft difference, still >= 0.90 by design.
    one_sided = match("CL 5 10 - 20 LOC 3", CANON)
    assert one_sided.is_match is True and 0.90 <= one_sided.score < 1.0
    # Conflicting complement (mapped alias vs different value) is a veto.
    assert match("CL 5 10 - 20 LOC 3", "CL 5 # 10 - 20 LC 4").score == 0.0
    # Nothing lands in the dead band.
    for a in _SAMPLE:
        for b in _SAMPLE:
            assert not (0.85 < match(a, b).score < 0.9126)


def test_previously_parseable_inputs_are_unchanged() -> None:
    # Regression lock: forms already accepted before the vocabulary growth.
    p = parse_canonical("KR 26 G # 5 - 73 LC 2")
    assert (p.via_type, p.via_number, p.via_letters, p.cross_number, p.plate) == ("KR", "26", "G", "5", "73")
    assert p.complement == (("LC", "2"),)
    p = parse_canonical("CL 5 # 10 - SO 1")
    assert p.plate is None and p.complement == (("SO", "1"),)
    p = parse_canonical("CL 5 # 10 - 20 NO 3")  # 'NO' after a real '#' is a second address marker, so the parse fails
    assert p.parse_ok is False
