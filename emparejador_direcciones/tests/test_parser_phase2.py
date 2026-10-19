"""Phase 2 parser tests: raw cadastral shapes (legacy prefixes, empty plate,
complement in the plate slot, opaque tails, single-letter quadrants)."""

from __future__ import annotations

import random

import pytest

from emparejador.parser import (
    COMPLEMENT_KINDS,
    VIA_TYPES,
    CanonicalAddress,
    clean_text,
    parse_canonical,
    render,
)


def _plate_identity(plate: str | None) -> str | None:
    """Plates are kept as written; identity ignores zero padding."""
    if plate is None:
        return None
    letter = plate[-1] if plate[-1].isalpha() else ""
    digits = plate[: len(plate) - len(letter)]
    return (digits.lstrip("0") or "0") + letter


def _fields(parsed: CanonicalAddress) -> tuple[object, ...]:
    return (
        parsed.parse_ok,
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
        _plate_identity(parsed.plate),
        frozenset(parsed.complement),
    )


# ---------------------------------------------------------------------------
# D7: doubled separators
# ---------------------------------------------------------------------------


def test_clean_text_collapses_adjacent_hashes_and_dashes() -> None:
    assert clean_text("KR 1 # # 4 - - 80") == "KR 1 # 4 - 80"
    assert clean_text("K 125 ## 18 - 55") == "K 125 # 18 - 55"
    assert clean_text("KR 1 # 4 - - - 80") == "KR 1 # 4 - 80"


def test_clean_text_does_not_collapse_non_adjacent_separators() -> None:
    assert clean_text("KR 1 # 4 - 80 # 5") == "KR 1 # 4 - 80 # 5"


def test_doubled_hash_address_parses() -> None:
    assert _fields(parse_canonical("K 125 # # 18 - 55")) == _fields(parse_canonical("KR 125 # 18 - 55"))


def test_doubled_hash_and_dash_address_parses() -> None:
    parsed = parse_canonical("K 111 # # 15 - - 120")
    assert parsed.parse_ok is True
    assert parsed.cross_number == "15"
    assert parsed.plate == "120"


def test_two_separated_hashes_still_rejected() -> None:
    parsed = parse_canonical("KR 1 # 4 - 80 # 5")
    assert parsed.parse_ok is False
    assert "multiples_direcciones" in parsed.notes


# ---------------------------------------------------------------------------
# D1: trailing empty plate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["C 1 # 74 -", "KR 85 B # 16 -", "K 41 # 31 B -", "KR 1 # -", "KR 1 #-"],
)
def test_trailing_empty_plate_parses_with_no_plate(raw: str) -> None:
    parsed = parse_canonical(raw)
    assert parsed.parse_ok is True
    assert parsed.plate is None
    assert parsed.complement == ()


def test_trailing_dash_is_pure_noise() -> None:
    with_dash = parse_canonical("C 1 # 74 -")
    without = parse_canonical("C 1 # 74")
    assert _fields(with_dash) == _fields(without)
    assert with_dash.via_type == "CL" and with_dash.cross_number == "74"


def test_trailing_dash_with_cross_letters() -> None:
    parsed = parse_canonical("K 41 # 31 B -")
    assert parsed.cross_number == "31" and parsed.cross_letters == "B"


def test_no_cross_no_plate_dash_only() -> None:
    parsed = parse_canonical("KR 1 # -")
    assert parsed.cross_number is None and parsed.plate is None


# ---------------------------------------------------------------------------
# D2: complement in the plate slot
# ---------------------------------------------------------------------------


def test_complement_in_plate_slot_known_kind() -> None:
    parsed = parse_canonical("KR 69 # 33 - LT 20")
    assert parsed.parse_ok is True
    assert parsed.plate is None
    assert set(parsed.complement) == {("LT", "20")}


def test_complement_in_plate_slot_with_unknown_leading_chunk() -> None:
    parsed = parse_canonical("CL 6 OESTE # KR 4 - BO 000101 LT 0372")
    assert parsed.parse_ok is True
    assert parsed.via_quadrant == "OESTE"
    assert parsed.cross_type == "KR" and parsed.cross_number == "4"
    assert parsed.plate is None
    assert set(parsed.complement) == {("BO", "101"), ("LT", "372")}


def test_complement_in_plate_slot_glued_unknown_kind() -> None:
    parsed = parse_canonical("K 47B # 55 B - Q2 T")
    assert parsed.parse_ok is True
    assert parsed.via_number == "47" and parsed.via_letters == "B"
    assert parsed.cross_number == "55" and parsed.cross_letters == "B"
    assert parsed.plate is None
    assert set(parsed.complement) == {("Q", "2T")}


def test_single_letter_alone_in_plate_slot_is_a_complement() -> None:
    parsed = parse_canonical("C 13 A 1 # 70 - T")
    assert parsed.parse_ok is True
    assert parsed.plate is None
    assert set(parsed.complement) == {("T", "")}


def test_plate_slot_quadrant_still_rejected() -> None:
    parsed = parse_canonical("KR 1 # 4 - NORTE")
    assert parsed.parse_ok is False
    assert "token_inesperado:NORTE" in parsed.notes


def test_plate_slot_second_address_rejected() -> None:
    parsed = parse_canonical("KR 1 # 4 - CL 5")
    assert parsed.parse_ok is False
    assert "multiples_direcciones" in parsed.notes


def test_plate_letter_stays_part_of_plate_identity() -> None:
    assert parse_canonical("KR 1 # 4 - 80 A").plate == "80A"
    assert parse_canonical("K 41 # 31 B - 12 T").plate == "12T"
    assert parse_canonical("KR 1 # 4 - 80 N").plate == "80N"


def test_overlong_plate_still_invalid() -> None:
    parsed = parse_canonical("KR 1 # 4 - " + "9" * 4301)
    assert parsed.parse_ok is False
    assert "placa_invalida" in parsed.notes


# ---------------------------------------------------------------------------
# D3: opaque complements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "plate", "expected"),
    [
        ("K 49 E # 49 - 50 8 C", "50", {("?", "8C")}),
        ("C 65 B # 2 D - 32 12 C", "32", {("?", "12C")}),
        ("K 1 D # 46 A - 44 8BC", "44", {("?", "8BC")}),
        ("K 47 B # 54 C - 89 2AC", "89", {("?", "2AC")}),
        ("K 103 # 12 C - 50 J 8 C", "50J", {("?", "8C")}),
        ("CL 9 # 51 - 48 GASS 5", "48", {("GASS", "5")}),
        ("K 125 # 19 - 58 SS 3 G", "58", {("SS", "3G")}),
        ("CL 34 A NORTE # 2 B - 120 DP 4", "120", {("DP", "4")}),
        ("K 8 # 22 - 48 /50 /52", "48", {("?", "50/52")}),
        ("KR 1 # 4 - 80 XXXX", "80", {("XXXX", "")}),
    ],
)
def test_opaque_tails_are_accepted(raw: str, plate: str, expected: set[tuple[str, str]]) -> None:
    parsed = parse_canonical(raw)
    assert parsed.parse_ok is True, parsed.notes
    assert parsed.plate == plate
    assert set(parsed.complement) == expected


def test_opaque_digit_tail_is_format_insensitive() -> None:
    a = parse_canonical("K 49 E # 49 - 50 8 C")
    b = parse_canonical("K 49 E # 49 - 50 8C")
    c = parse_canonical("K 49 E # 49 - 50 08 C")
    d = parse_canonical("K 49 E # 49 - 050 0008c")
    assert set(a.complement) == set(b.complement) == set(c.complement) == set(d.complement)


def test_opaque_two_digit_runs_stay_separated() -> None:
    spaced = parse_canonical("K 8 # 22 - 48 50 52")
    slashed = parse_canonical("K 8 # 22 - 48 /50 /52")
    fused = parse_canonical("K 8 # 22 - 48 5052")
    assert set(spaced.complement) == set(slashed.complement)
    assert set(spaced.complement) != set(fused.complement)


def test_opaque_chunk_followed_by_known_kind() -> None:
    parsed = parse_canonical("K 49 E # 49 - 50 8 C LT 5")
    assert set(parsed.complement) == {("?", "8C"), ("LT", "5")}


def test_piso_is_a_known_kind() -> None:
    assert "PISO" in COMPLEMENT_KINDS
    parsed = parse_canonical("CL 10 # 42 - 02 PISO 3")
    assert set(parsed.complement) == {("PISO", "3")}


def test_slash_only_tail_tokens_are_ignored() -> None:
    parsed = parse_canonical("K 8 # 22 - 48 / LT 1")
    assert parsed.parse_ok is True
    assert set(parsed.complement) == {("LT", "1")}


# ---------------------------------------------------------------------------
# D3: tail hygiene
# ---------------------------------------------------------------------------


def test_tail_of_exactly_40_chars_is_accepted() -> None:
    tail = " ".join(["A" * 10, "B" * 9, "C" * 9, "D" * 9])
    assert len(tail) == 40
    assert parse_canonical(f"KR 1 # 4 - 80 {tail}").parse_ok is True


def test_tail_of_41_chars_is_rejected_with_note() -> None:
    tail = " ".join(["A" * 10, "B" * 10, "C" * 9, "D" * 9])
    assert len(tail) == 41
    parsed = parse_canonical(f"KR 1 # 4 - 80 {tail}")
    assert parsed.parse_ok is False
    assert "complemento_demasiado_largo" in parsed.notes


def test_single_token_tail_boundary_40_41() -> None:
    assert parse_canonical("KR 1 # 4 - 80 " + "X" * 40).parse_ok is True
    over = parse_canonical("KR 1 # 4 - 80 " + "X" * 41)
    assert over.parse_ok is False and "complemento_demasiado_largo" in over.notes


def test_tail_length_boundary_applies_in_plate_slot_too() -> None:
    assert parse_canonical("KR 1 # 4 - " + "X" * 40).parse_ok is True
    assert parse_canonical("KR 1 # 4 - " + "X" * 41).parse_ok is False


def test_ten_thousand_char_junk_tail_rejected() -> None:
    for junk in ("X" * 10_000, "X 1 " * 3000, "LT 5 " * 2000):
        parsed = parse_canonical("KR 1 # 4 - 80 " + junk)
        assert parsed.parse_ok is False
        assert "complemento_demasiado_largo" in parsed.notes


def test_huge_digit_tail_rejected_not_raised() -> None:
    parsed = parse_canonical("KR 1 # 4 - 80 " + "9" * 4301)
    assert parsed.parse_ok is False


def test_ten_thousand_hashes_do_not_crash() -> None:
    assert parse_canonical("#" * 10_000).parse_ok is False
    assert parse_canonical("KR 1 # 4 - 80 " + "# " * 5000).parse_ok is False


@pytest.mark.parametrize(
    ("raw", "note"),
    [
        ("KR 1 # 4 - 80 XX # 5", "multiples_direcciones"),
        ("KR 1 # 4 - 80 CL 5", "multiples_direcciones"),
        ("KR 1 # 4 - 80 XX A 5", "multiples_direcciones"),
        ("KR 1 # 4 - 80 XX T 5", "multiples_direcciones"),
        ("KR 1 # 4 - 80 NORTE", "token_inesperado:NORTE"),
        ("KR 1 # 4 - 80 - 82", "token_inesperado:-"),
        ("KR 1 # 4 - 80 XX ( 5", "token_inesperado:("),
        ("KR 1 # 4 - 80 X$Y", "token_inesperado:X$Y"),
        ("KR 1 # 4 - 80 ★", "token_inesperado:★"),
        ("KR 1 # 4 - /50 LT 1", "token_inesperado:/50"),
    ],
)
def test_tail_hygiene_rejections(raw: str, note: str) -> None:
    parsed = parse_canonical(raw)
    assert parsed.parse_ok is False
    assert note in parsed.notes


def test_cascaded_addresses_stay_rejected() -> None:
    assert parse_canonical("KR 1 # 4 - 80 CL 5 # 3 - 2").parse_ok is False


# ---------------------------------------------------------------------------
# D5: single-letter quadrant abbreviations
# ---------------------------------------------------------------------------


def test_quadrant_after_via_letter() -> None:
    parsed = parse_canonical("C 70 B N # 4 C - 104 38 C")
    assert parsed.parse_ok is True
    assert parsed.via_letters == "B" and parsed.via_quadrant == "NORTE"
    assert parsed.cross_letters == "C" and parsed.cross_quadrant is None
    assert parsed.plate == "104"
    assert set(parsed.complement) == {("?", "38C")}


def test_quadrant_after_via_letter_with_cross_type() -> None:
    parsed = parse_canonical("C 3 A O # K 90 -")
    assert parsed.parse_ok is True
    assert parsed.via_quadrant == "OESTE"
    assert parsed.cross_type == "KR" and parsed.cross_number == "90"


def test_quadrant_after_bis() -> None:
    parsed = parse_canonical("C 1 A BIS O # 81 - 19")
    assert parsed.parse_ok is True
    assert parsed.via_letters == "A" and parsed.via_bis is True and parsed.via_quadrant == "OESTE"


def test_quadrant_after_cross_letter() -> None:
    parsed = parse_canonical("C 71 I # 3 C N - 13 66 C")
    assert parsed.parse_ok is True
    assert parsed.cross_letters == "C" and parsed.cross_quadrant == "NORTE"


def test_quadrant_after_suffix_number() -> None:
    parsed = parse_canonical("K 3 A 3 N # 71 H - 13 19 C")
    assert parsed.parse_ok is True
    assert parsed.via_letters == "A" and parsed.via_suffix == "3"
    assert parsed.via_suffix_letters is None and parsed.via_quadrant == "NORTE"
    assert parsed.cross_letters == "H"


@pytest.mark.parametrize(
    ("abbr", "full"),
    [("N", "NORTE"), ("S", "SUR"), ("E", "ESTE"), ("O", "OESTE"), ("W", "OESTE")],
)
def test_every_abbreviation_maps_to_full_quadrant(abbr: str, full: str) -> None:
    via = parse_canonical(f"KR 5 B {abbr} # 4 - 80")
    cross = parse_canonical(f"KR 5 # 9 B {abbr} - 80")
    assert via.via_quadrant == full and via.via_letters == "B"
    assert cross.cross_quadrant == full and cross.cross_letters == "B"


def test_abbreviation_equals_full_word() -> None:
    a = parse_canonical("C 70 B N # 4 C - 104")
    b = parse_canonical("CL 70 B NORTE # 4 C - 104")
    assert _fields(a) == _fields(b)


@pytest.mark.parametrize("letter", ["N", "S", "E", "O", "W"])
def test_abbreviation_directly_after_number_stays_a_letter(letter: str) -> None:
    via = parse_canonical(f"KR 41 {letter} # 4 - 80")
    cross = parse_canonical(f"KR 1 # 4 {letter} - 80")
    assert via.via_letters == letter and via.via_quadrant is None
    assert cross.cross_letters == letter and cross.cross_quadrant is None


def test_legacy_w_after_number_is_a_letter() -> None:
    parsed = parse_canonical("A 9 W # -")
    assert parsed.parse_ok is True
    assert parsed.via_type == "AV" and parsed.via_letters == "W" and parsed.via_quadrant is None


def test_letter_then_abbreviation_then_another_is_rejected() -> None:
    assert parse_canonical("KR 24 B N O # 9 - 47").parse_ok is False


def test_two_plain_letters_still_rejected() -> None:
    assert parse_canonical("KR 24 # 9 C B - 47").parse_ok is False
    assert parse_canonical("KR 24 B C # 9 - 47").parse_ok is False


def test_letter_then_quadrant_abbreviation_in_direct_letter_slot() -> None:
    parsed = parse_canonical("KR 41 E N # 4 - 80")
    assert parsed.via_letters == "E" and parsed.via_quadrant == "NORTE"


def test_suffix_letter_then_abbreviation() -> None:
    parsed = parse_canonical("KR 26 1 E N # 73 - 10")
    assert parsed.via_suffix == "1" and parsed.via_suffix_letters == "E"
    assert parsed.via_quadrant == "NORTE"


def test_regular_suffix_letters_are_unchanged() -> None:
    parsed = parse_canonical("KR 26 1 A # 73 - 10")
    assert parsed.via_suffix == "1" and parsed.via_suffix_letters == "A" and parsed.via_quadrant is None


def test_abbreviation_never_a_quadrant_after_plate_or_before_number() -> None:
    assert parse_canonical("KR N 5 # 4 - 80").parse_ok is False
    assert parse_canonical("KR 1 # N - 80").parse_ok is False
    tail = parse_canonical("KR 1 # 4 - 80 12 N")
    assert tail.cross_quadrant is None and set(tail.complement) == {("?", "12N")}
    plate_slot = parse_canonical("KR 1 # 4 - N")
    assert plate_slot.plate is None and set(plate_slot.complement) == {("N", "")}


def test_abbreviation_after_bis_is_a_quadrant() -> None:
    parsed = parse_canonical("KR 5 BIS N # 4 - 80")
    assert parsed.parse_ok is True and parsed.via_bis is True and parsed.via_quadrant == "NORTE"


# ---------------------------------------------------------------------------
# D6: legacy via type "A" -> AV
# ---------------------------------------------------------------------------


def test_a_is_a_via_type_at_segment_start() -> None:
    assert VIA_TYPES["A"] == "AV"
    parsed = parse_canonical("A 9 # 5 - 3")
    assert parsed.via_type == "AV" and parsed.via_number == "9"


def test_a_as_via_type_with_letter_and_quadrant() -> None:
    parsed = parse_canonical("A 9 A N # 54 N - 65 T")
    assert parsed.via_type == "AV" and parsed.via_letters == "A" and parsed.via_quadrant == "NORTE"
    assert parsed.cross_letters == "N" and parsed.cross_quadrant is None
    assert parsed.plate == "65T"


def test_a_as_street_letter_stays_a_letter() -> None:
    parsed = parse_canonical("KR 9 A # 5 A - 3")
    assert parsed.via_letters == "A" and parsed.cross_letters == "A" and parsed.via_type == "KR"


def test_a_as_cross_type() -> None:
    parsed = parse_canonical("KR 1 # A 5 - 3")
    assert parsed.cross_type == "AV" and parsed.cross_number == "5"


def test_a_without_number_is_not_a_segment() -> None:
    assert parse_canonical("A # 4 - 80").parse_ok is False


@pytest.mark.parametrize("raw", ["M # -", "M 5 # 4 - 80", "X 5 # 4 - 80", "Z 1 # 2 - 3"])
def test_unknown_single_letter_types_stay_rejected(raw: str) -> None:
    assert parse_canonical(raw).parse_ok is False


# ---------------------------------------------------------------------------
# Render: idempotency for plate-less and opaque addresses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "C 1 # 74 -",
        "KR 85 B # 16 -",
        "A 9 W # -",
        "KR 69 # 33 - LT 20",
        "CL 6 OESTE # KR 4 - BO 000101 LT 0372",
        "K 47B # 55 B - Q2 T",
        "C 13 A 1 # 70 - T",
        "K 49 E # 49 - 50 8 C",
        "K 1 D # 46 A - 44 8BC",
        "K 103 # 12 C - 50 J 8 C",
        "K 125 # 19 - 58 SS 3 G",
        "K 8 # 22 - 48 /50 /52",
        "K 49 E # 49 - 50 8 C LT 5",
        "C 70 B N # 4 C - 104 38 C",
        "K 3 A 3 N # 71 H - 13 19 C",
        "C 1 A BIS O # 81 - 19",
        "KR 1 # -  LT 3",
        "KR 1 # 4 - 80 XXXX",
        "KR 1 # 4 - 80 A XXXX",
        # A dash-fused value whose letter run is also a via type ("A-1"):
        # glued as "A1" it would re-read as a second address, so render
        # keeps the fusion dash.
        "CL 45 # 93 - 38 1103A-1",
        "CL 10 # 42 - 02 SS T-5",
        "CL 10 # 42 - 02 AP 1 T-5 LT 4",
    ],
)
def test_render_reparse_idempotent(raw: str) -> None:
    first = parse_canonical(raw)
    assert first.parse_ok is True, first.notes
    rendered = render(first)
    second = parse_canonical(rendered)
    assert _fields(first) == _fields(second), rendered
    assert render(second) == rendered


def test_render_forms() -> None:
    assert render(parse_canonical("C 1 # 74 -")) == "CL 1 # 74"
    assert render(parse_canonical("KR 69 # 33 - LT 20")) == "KR 69 # 33 - LT 20"
    assert render(parse_canonical("K 49 E # 49 - 50 8 C")) == "KR 49 E # 49 - 50 8C"
    assert render(parse_canonical("C 70 B N # 4 C - 104 38 C")) == "CL 70 B NORTE # 4 C - 104 38C"
    assert render(parse_canonical("K 8 # 22 - 48 /50 /52")) == "KR 8 # 22 - 48 50 52"


def test_render_keeps_fusion_dash_when_glue_would_read_as_address() -> None:
    assert render(parse_canonical("CL 45 # 93 - 38 1103A-1")) == "CL 45 # 93 - 38 1103 A - 1"


def test_render_plate_less_complement_keeps_the_dash() -> None:
    assert "-" in render(parse_canonical("KR 1 # 4 - LT 3")).split()


def _random_address(rng: random.Random) -> str:
    def segment(cross: bool) -> str:
        toks: list[str] = []
        if rng.random() < 0.7:
            toks.append(rng.choice(["KR", "CL", "K", "C", "A", "AV", "DG", "TV"]))
        toks.append(str(rng.randint(0, 200)))
        if rng.random() < 0.5:
            toks.append(rng.choice("ABCDEFGHNSOW"))
        if rng.random() < 0.3:
            toks.append(str(rng.randint(1, 9)))
            if rng.random() < 0.3:
                toks.append(rng.choice("ABCEN"))
        if rng.random() < 0.2:
            toks.append("BIS")
        if rng.random() < 0.3:
            toks.append(rng.choice(["N", "S", "E", "O", "W", "NORTE", "SUR", "OESTE", "ESTE"]))
        return " ".join(toks)

    parts = [segment(False), "#"]
    if rng.random() < 0.85:
        parts.append(segment(True))
    if rng.random() < 0.9:
        parts.append("-")
        if rng.random() < 0.8:
            parts.append(str(rng.randint(0, 999)))
            if rng.random() < 0.2:
                parts.append(rng.choice("ABCTPW"))
    if rng.random() < 0.6:
        tail: list[str] = []
        for _ in range(rng.randint(1, 3)):
            choice = rng.random()
            if choice < 0.4:
                tail.extend([rng.choice(["AP", "LT", "TO", "PISO", "BO", "SS", "DP", "GASS"]), str(rng.randint(0, 500))])
            elif choice < 0.7:
                tail.extend([str(rng.randint(0, 50)), rng.choice(["C", "BC", "AC", "G"])])
            else:
                tail.append(rng.choice(["T", "Q", "XX", "/50"]))
        parts.extend(tail)
    return " ".join(parts)


def test_random_render_reparse_idempotent() -> None:
    rng = random.Random(7)
    parsed_ok = 0
    for _ in range(4000):
        raw = _random_address(rng)
        first = parse_canonical(raw)
        if not first.parse_ok:
            continue
        parsed_ok += 1
        rendered = render(first)
        second = parse_canonical(rendered)
        assert second.parse_ok, (raw, rendered, second.notes)
        assert _fields(first) == _fields(second), (raw, rendered)
        assert render(second) == rendered
    assert parsed_ok > 500
