"""Tests for emparejador.parser: cleanup, grammar, failures, fuzz, idempotency."""

from __future__ import annotations

import random
import string

import pytest

from emparejador.parser import (
    COMPLEMENT_KINDS,
    CanonicalAddress,
    clean_text,
    parse_canonical,
    render,
)


# ---------------------------------------------------------------------------
# 1. Cleanup layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("avenida 15 óeste", "AVENIDA 15 OESTE"),
        ("KR 1 – 9", "KR 1 - 9"),  # en dash
        ("KR 1 — 9", "KR 1 - 9"),  # em dash
        ("KR 1 No. 4 - 80", "KR 1 # 4 - 80"),
        ("KR 1 Nº 4 - 80", "KR 1 # 4 - 80"),
        ("KR 1 NRO 4 - 80", "KR 1 # 4 - 80"),
        ("KR 1 NUM 4 - 80", "KR 1 # 4 - 80"),
        ("CRA. 1 # 4 - 80", "CRA 1 # 4 - 80"),
        ("KR   1   #   4   -   80", "KR 1 # 4 - 80"),
        ("KR\t1\n#\t4 - 80", "KR 1 # 4 - 80"),
    ],
)
def test_clean_text_basic_hygiene(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KR 96F # 96 - 66", "KR 96 F # 96 - 66"),
        ("CL12 # 5 - 6", "CL 12 # 5 - 6"),
        ("KR 1A # 4 - 80", "KR 1 A # 4 - 80"),
    ],
)
def test_clean_text_splits_glued_tokens(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


# ---------------------------------------------------------------------------
# 2. Happy-path field extraction
# ---------------------------------------------------------------------------


def test_parse_simple_address() -> None:
    parsed = parse_canonical("KR 1 # 4 - 80")
    assert parsed.parse_ok is True
    assert parsed.via_type == "KR"
    assert parsed.via_number == "1"
    assert parsed.cross_number == "4"
    assert parsed.plate == "80"


def test_parse_with_via_letters_and_suffix() -> None:
    parsed = parse_canonical("KR 26 H 3 # 73 - 10")
    assert parsed.parse_ok is True
    assert parsed.via_type == "KR"
    assert parsed.via_number == "26"
    assert parsed.via_letters == "H"
    assert parsed.via_suffix == "3"
    assert parsed.cross_number == "73"
    assert parsed.plate == "10"


def test_parse_cross_letters_and_suffix() -> None:
    parsed = parse_canonical("CL 72 T 1 # 26 G 11 - 70")
    assert parsed.parse_ok is True
    assert parsed.via_type == "CL"
    assert parsed.via_number == "72"
    assert parsed.via_letters == "T"
    assert parsed.via_suffix == "1"
    assert parsed.cross_number == "26"
    assert parsed.cross_letters == "G"
    assert parsed.cross_suffix == "11"
    assert parsed.plate == "70"


def test_parse_via_bis() -> None:
    parsed = parse_canonical("CL 12 # 48 BIS - 31")
    assert parsed.parse_ok is True
    assert parsed.via_number == "12"
    assert parsed.cross_number == "48"
    assert parsed.cross_bis is True
    assert parsed.plate == "31"


def test_parse_via_quadrant() -> None:
    parsed = parse_canonical("AV 15 OESTE # 9 OESTE - 137")
    assert parsed.parse_ok is True
    assert parsed.via_quadrant == "OESTE"
    assert parsed.cross_quadrant == "OESTE"
    assert parsed.via_number == "15"
    assert parsed.cross_number == "9"
    assert parsed.plate == "137"


def test_parse_cross_type() -> None:
    parsed = parse_canonical("CL 25 NORTE # AV 6 - 30")
    assert parsed.parse_ok is True
    assert parsed.via_quadrant == "NORTE"
    assert parsed.cross_type == "AV"
    assert parsed.cross_number == "6"
    assert parsed.plate == "30"


def test_parse_complement_after_plate() -> None:
    parsed = parse_canonical("CL 12 A # 52 - 64 SO 1 PQ 19")
    assert parsed.parse_ok is True
    assert parsed.via_letters == "A"
    assert parsed.plate == "64"
    assert ("SO", "1") in parsed.complement
    assert ("PQ", "19") in parsed.complement


def test_parse_extra_worked_examples() -> None:
    parsed = parse_canonical("KR 28 D 3 # 72 L - 04")
    assert parsed.parse_ok is True
    assert parsed.via_letters == "D"
    assert parsed.via_suffix == "3"
    assert parsed.cross_letters == "L"
    assert parsed.plate == "04"


def test_parse_glued_style_equals_spaced_style() -> None:
    glued = parse_canonical("KR 96F # 96 - 66")
    spaced = parse_canonical("KR 96 F # 96 - 66")
    assert glued.parse_ok is True and spaced.parse_ok is True
    fields = (
        "via_type",
        "via_number",
        "via_letters",
        "via_suffix",
        "cross_number",
        "plate",
    )
    for field in fields:
        assert getattr(glued, field) == getattr(spaced, field)


# ---------------------------------------------------------------------------
# 3. Number normalization
# ---------------------------------------------------------------------------


def test_via_number_leading_zeros_normalized() -> None:
    parsed = parse_canonical("KR 01 # 04 - 80")
    assert parsed.via_number == "1"
    assert parsed.cross_number == "4"


def test_plate_kept_as_written() -> None:
    parsed = parse_canonical("KR 1 # 4 - 080")
    assert parsed.plate == "080"


# ---------------------------------------------------------------------------
# 4. Complement chunks
# ---------------------------------------------------------------------------


def test_complement_equivalent_reordered_alias_zeros() -> None:
    a = parse_canonical("CL 10 # 42 - 02 AP 501 TO 3")
    b = parse_canonical("CL 10 # 42 - 2 TO 3 APTO 0501")
    assert set(a.complement) == set(b.complement)


def test_complement_digit_letter_run_variants() -> None:
    a = parse_canonical("CL 10 # 42 - 02 AP 101-A")
    b = parse_canonical("CL 10 # 42 - 02 AP 101A")
    assert set(a.complement) == set(b.complement)


def test_complement_leading_unknown_chunk_keyed_by_first_token() -> None:
    # "BO" is not in COMPLEMENT_KINDS (finding 13), so this genuinely
    # exercises the leading-unknown-chunk rule (finding 4).
    parsed = parse_canonical("CL 10 # 42 - 02 BO 002003 LT 0031")
    assert parsed.parse_ok is True
    assert set(parsed.complement) == {("BO", "2003"), ("LT", "31")}


def test_complement_kind_without_value() -> None:
    parsed = parse_canonical("CL 10 # 42 - 02 PH")
    assert ("PH", "") in parsed.complement


def test_complement_digit_digit_run_kept_distinct_with_dash() -> None:
    """Finding 3: a dash-joined digit-digit run must not fuse into one
    number; it must stay distinguishable from the literally joined value."""
    fused = parse_canonical("CL 10 # 42 - 02 LOTE 1-2")
    spaced = parse_canonical("CL 10 # 42 - 02 LOTE 1 2")
    joined = parse_canonical("CL 10 # 42 - 02 LOTE 12")
    assert set(fused.complement) == set(spaced.complement)
    assert set(fused.complement) != set(joined.complement)


def test_complement_digit_digit_run_kept_distinct_no_dash() -> None:
    a = parse_canonical("CL 10 # 42 - 02 AP 1 2")
    b = parse_canonical("CL 10 # 42 - 02 AP 12")
    assert set(a.complement) != set(b.complement)


def test_complement_digit_letter_run_still_fuses() -> None:
    # Regression: the finding-3 fix must not disturb the digit+letter fusion
    # rule (only digit-digit runs need a separator).
    a = parse_canonical("CL 10 # 42 - 02 AP 101-A")
    b = parse_canonical("CL 10 # 42 - 02 AP 101 A")
    c = parse_canonical("CL 10 # 42 - 02 AP 101A")
    assert set(a.complement) == set(b.complement) == set(c.complement)


def test_so_and_bo_are_not_known_complement_kinds() -> None:
    """Finding 13: SO/BO must not be pre-registered kinds; they only work
    through the leading-unknown-chunk rule."""
    assert "SO" not in COMPLEMENT_KINDS
    assert "BO" not in COMPLEMENT_KINDS


def test_leading_unknown_chunk_so_then_known_kind() -> None:
    parsed = parse_canonical("CL 12 A # 52 - 64 SO 1 PQ 19")
    assert parsed.parse_ok is True
    assert set(parsed.complement) == {("SO", "1"), ("PQ", "19")}


# ---------------------------------------------------------------------------
# 5. Failures: never raise, parse_ok=False with a note
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "\t\n",
        123456,
        3.14,
        ["KR", 1],
        {"a": 1},
        "10293847",
        "SIN DIRECCION",
        "###---",
        "KR 1 4 80",
        "KR # 4 - 80",
        "KR 1 # 4 - 80; CL 5 # 3 - 2",
        "KR 1 X Y Z Q # 4 - 80",
        "X" * 100,
        "★☆♦♣ ¿¡€ 漢字",
    ],
)
def test_parse_failures_never_raise(raw: object) -> None:
    parsed = parse_canonical(raw)  # type: ignore[arg-type]
    assert parsed.parse_ok is False
    assert isinstance(parsed.notes, tuple)
    assert len(parsed.notes) >= 1


def test_parse_failure_no_hash_note() -> None:
    parsed = parse_canonical("KR 1 4 80")
    assert "falta_separador_hash" in parsed.notes


def test_parse_failure_no_via_number_note() -> None:
    parsed = parse_canonical("KR # 4 - 80")
    assert "numero_via_faltante" in parsed.notes


def test_parse_failure_multiple_addresses_note() -> None:
    parsed = parse_canonical("KR 1 # 4 - 80; CL 5 # 3 - 2")
    assert "multiples_direcciones" in parsed.notes


def test_parse_failure_empty_note() -> None:
    parsed = parse_canonical("")
    assert "entrada_vacia" in parsed.notes


def test_parse_failure_unexpected_token_note() -> None:
    parsed = parse_canonical("KR 1 X Y Z Q # 4 - 80")
    assert any(note.startswith("token_inesperado:") for note in parsed.notes)


# ---------------------------------------------------------------------------
# 5b. Plate digit-run cap (finding 1): never raises, only real overflows fail
# ---------------------------------------------------------------------------


def test_plate_extremely_long_digit_run_is_invalid() -> None:
    parsed = parse_canonical("KR 1 # 4 - " + "9" * 4301)
    assert parsed.parse_ok is False
    assert "placa_invalida" in parsed.notes


def test_plate_leading_zero_padding_does_not_trigger_cap() -> None:
    parsed = parse_canonical("KR 1 # 4 - 0000080")
    assert parsed.parse_ok is True
    assert parsed.plate == "0000080"


# ---------------------------------------------------------------------------
# 5c. Junk after the plate is rejected, not silently kept as a complement
# (finding 2 / finding 12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_note"),
    [
        ("KR 1 # 4 - 80 CL 5 # 3 - 2", "multiples_direcciones"),
        ("KR 1 # 4 - 80 # 5", "multiples_direcciones"),
        ("KR 1 # 4 - 80 - 82", "token_inesperado:-"),
        ("KR 1 # 4 - 80 CL 5", "multiples_direcciones"),
        ("KR 1 # 4 - 80 NORTE", "token_inesperado:NORTE"),
        ("KR 1 # 4 - 80 SUR", "token_inesperado:SUR"),
        # Phase 2: "KR 1 # 4 - 80 XXXX" is no longer junk, it is an opaque
        # complement (see tests/test_parser_phase2.py).
    ],
)
def test_junk_after_plate_is_rejected(raw: str, expected_note: str) -> None:
    parsed = parse_canonical(raw)
    assert parsed.parse_ok is False
    assert expected_note in parsed.notes


# ---------------------------------------------------------------------------
# 5d. Non-ASCII digits are rejected consistently (finding 9)
# ---------------------------------------------------------------------------


def test_non_ascii_digits_rejected_on_via_number() -> None:
    parsed = parse_canonical("KR ٣ # 4 - 80")  # Arabic-Indic 3
    assert parsed.parse_ok is False


def test_non_ascii_digits_rejected_on_plate() -> None:
    parsed = parse_canonical("KR 1 # 4 - ٣")  # Arabic-Indic 3
    assert parsed.parse_ok is False


# ---------------------------------------------------------------------------
# 6. Fuzz: 3000 random strings never raise
# ---------------------------------------------------------------------------


def test_fuzz_random_strings_never_raise() -> None:
    rng = random.Random(20260101)
    alphabet = string.printable
    for _ in range(3000):
        length = rng.randint(0, 40)
        raw = "".join(rng.choice(alphabet) for _ in range(length))
        parsed = parse_canonical(raw)
        assert isinstance(parsed.parse_ok, bool)


def test_fuzz_unicode_strings_never_raise() -> None:
    """Finding 11: fuzz also over accented Latin, ñ, fullwidth #, NBSP,
    Arabic-Indic digits and emoji, up to length 200."""
    rng = random.Random(20260202)
    alphabet = (
        string.printable
        + "áéíóúÁÉÍÓÚñÑ"
        + "＃"  # fullwidth "＃"
        + " "  # NBSP
        + "٠١٢٣٤٥٦٧٨٩"  # Arabic-Indic digits
        + "😀🏠📍"  # emoji
    )
    for _ in range(1000):
        length = rng.randint(0, 200)
        raw = "".join(rng.choice(alphabet) for _ in range(length))
        parsed = parse_canonical(raw)
        assert isinstance(parsed.parse_ok, bool)


# ---------------------------------------------------------------------------
# 7. Idempotency: parse_canonical(render(parse_canonical(x))) == parse_canonical(x)
# ---------------------------------------------------------------------------


def _field_tuple(parsed: CanonicalAddress) -> tuple[object, ...]:
    """Extract the comparable field tuple from a parsed address."""
    return (
        parsed.via_type,
        parsed.via_number,
        parsed.via_letters,
        parsed.via_suffix,
        parsed.via_suffix_letters,
        parsed.via_bis,
        parsed.via_quadrant,
        parsed.cross_type,
        parsed.cross_number,
        parsed.cross_letters,
        parsed.cross_suffix,
        parsed.cross_suffix_letters,
        parsed.cross_bis,
        parsed.cross_quadrant,
        parsed.plate,
        frozenset(parsed.complement),
    )


@pytest.mark.parametrize(
    "raw",
    [
        "KR 1 # 4 - 80",
        "KR 26 H 3 # 73 - 10",
        "CL 72 T 1 # 26 G 11 - 70",
        "CL 12 # 48 BIS - 31",
        "AV 15 OESTE # 9 OESTE - 137",
        "CL 25 NORTE # AV 6 - 30",
        "CL 12 A # 52 - 64 SO 1 PQ 19",
        "KR 96F # 96 - 66",
        "KR 28 D 3 # 72 L - 04",
        "CL 10 # 42 - 02 AP 501 TO 3",
        "CL 10 # 42 - 02 TO 3 APTO 0501",
        "CL 10 # 42 - 02 AP 101-A",
        "CL 10 # 42 - 02 LOTE 1-2",
        "CL 10 # 42 - 02 BO 002003 LT 0031",
        "CL 10 # 42 - 02 PH",
        "KR 1 # 4 - 80 A",
        "KR 1 # 4",
        "KR 1 # - 80",
        "KR 1 # 4 11 G - 80",
        "KR 26 1 A # 73 - 10",
    ],
)
def test_idempotency_render_reparse_same_fields(raw: str) -> None:
    first = parse_canonical(raw)
    second = parse_canonical(render(first))
    assert _field_tuple(first) == _field_tuple(second)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KR 26 H 3 # 73 - 10 AP 501 TO 3", "KR 26 H 3 # 73 - 10 AP 501 TO 3"),
        ("kr 1 #4-080", "KR 1 # 4 - 80"),  # plate zero-stripped
        ("KR 1 # 4 - 2", "KR 1 # 4 - 02"),  # plate zero-padded to 2 digits
        ("KR 1 # 4 - 0000080", "KR 1 # 4 - 80"),
        ("CL 10 # 42 - 2 TO 3 APTO 0501", "CL 10 # 42 - 02 AP 501 TO 3"),
        ("KR 96F # 96 - 66", "KR 96 F # 96 - 66"),
        # The leading-unknown chunk (SO) is emitted first even though it
        # sorts after PQ in the stored complement, so re-parsing recovers it.
        ("CL 12 A # 52 - 64 SO 1 PQ 19", "CL 12 A # 52 - 64 SO 1 PQ 19"),
    ],
)
def test_render_emits_canonical_spaced_form(raw: str, expected: str) -> None:
    parsed = parse_canonical(raw)
    assert parsed.parse_ok is True
    assert render(parsed) == expected


def test_render_falls_back_to_raw_when_parse_failed() -> None:
    assert render(parse_canonical("SIN DIRECCION")) == "SIN DIRECCION"
    assert render(parse_canonical(None)) == "None"  # type: ignore[arg-type]
