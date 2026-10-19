"""Round 7: parser / fidelity families behind TRUE high fidelity-risk corruptions. Synthetic addresses only.

Each family pins the intended reading where it is unambiguous and keeps the stricter (historical) outcome where
it is not:

* F1 letter-digit designators without ``#`` (``KR 987 5B1 321``): the plate is no longer lost;
* F2 ``NO18`` is the number sign, never the quadrant N + OESTE (a genuine ``NO #`` quadrant stays);
* F3 a street letter glued to a spelled quadrant (``3BNORTE``) survives;
* F4 a spelled quadrant right after the plate is kept in the rule text (``ESTE`` is not: no cadastral occurrence);
* F5 ``entre`` / ``con`` intersections never invent a plate from a street number;
* F6 ``ET2`` / ``ETAPA 2`` is a unit, never the plate;
* F7 ordinal indicators / double-encoded degree signs are not letters;
* F8 coordinates and long digit runs are not door digits in the fidelity check;
* F9 a non-OK row with ``fidelity_risk == 'high'`` carries no rule text.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import make_stub

from cali_address import catalog, service
from cali_address.fidelity import FidelityReport, assess_fidelity
from cali_address.parser import is_truncated, parse_address, rule_canonical
from cali_address.service import normalize_strict


def rule(text):
    return rule_canonical(parse_address(text))


# ---------------------------------------------------------------------------
# F1: letter-digit designators without '#'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CRA 987 5B1 321 APT 201", "KR 987 # 5 B 1 - 321 AP 201"),     # cross designator, door kept
        ("CRA 987 5B1 321", "KR 987 # 5 B 1 - 321"),
        ("KR 987 D2 N 654C 321", "KR 987 D 2 N # 654 C - 321"),         # via designator + cross + plate
        ("Cra 987C2 654-321", "KR 987 C 2 # 654 - 321"),
        ("Cra 987a9 654-321", "KR 987 A 9 # 654 - 321"),
        ("Cra987d7 654-321", "KR 987 D 7 # 654 - 321"),
        ("KR 987G6 654S1 321", "KR 987 G 6 # 654 S 1 - 321"),            # both designators glued
        ("KR 987G6 654S1 321 APTO 5", "KR 987 G 6 # 654 S 1 - 321 AP 5"),
    ],
)
def test_f1_glued_designators_without_hash(raw, expected):
    assert rule(raw) == expected


@pytest.mark.parametrize(
    ("raw", "explicit"),
    [
        ("CRA 987 5B1 321 APT 201", "CRA 987 # 5B1 - 321 APT 201"),
        ("KR 987 D2 N 654C 321", "KR 987 D2 N # 654C - 321"),
        ("Cra 987C2 654-321", "Cra 987C2 # 654-321"),
    ],
)
def test_f1_reading_matches_the_explicit_hash_reading(raw, explicit):
    assert rule(raw) == rule(explicit)


@pytest.mark.parametrize(
    ("raw", "kept"),
    [
        ("Cra 987A9-250", "KR 987 A # 9 - 250"),        # one number left of the dash: ambiguous, stricter reading
        ("Cra 987A9 654", "KR 987 A # 9 - 654"),        # two numbers only: ambiguous
        ("KR 987 5 B 1 321", "KR 987 # 5 B 1 - 321"),   # separated designator: historical reading
        ("CL 987 # 654 - 321", "CL 987 # 654 - 321"),
    ],
)
def test_f1_ambiguous_readings_keep_the_historical_outcome(raw, kept):
    assert rule(raw) == kept


@pytest.mark.parametrize("raw", ["CRA 987 5B1 321 APT 201", "KR 987 D2 N 654C 321", "Cra 987C2 654-321"])
def test_f1_no_door_digit_is_lost(raw):
    out = rule(raw)
    report = assess_fidelity(raw, out)
    assert "door_digits_dropped" not in report.flags and report.risk in ("none", "low")


# ---------------------------------------------------------------------------
# F2: NO + digits is the number sign
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CARRERA 987 NO18 61 CASA 36", "KR 987 # 18 - 61 CA 36"),
        ("CARRERA 987 NO18-61", "KR 987 # 18 - 61"),
        ("CARRERA 987 No.18-61", "KR 987 # 18 - 61"),
        ("CARRERA 987 N°18-61", "KR 987 # 18 - 61"),
        ("CARRERA 987 NO 18-61", "KR 987 # 18 - 61"),
        ("carrera 987 no18-61", "KR 987 # 18 - 61"),
    ],
)
def test_f2_no_digits_is_the_number_sign(raw, expected):
    assert rule(raw) == expected


def test_f2_a_genuine_quadrant_position_is_untouched():
    assert rule("CL 987 NO # 654") == "CL 987 N OESTE # 654"          # NO before '#' keeps its historical reading
    assert rule("CL 987 # 654 - 321 CASA NO18") == "CL 987 # 654 - 321 CA NO18"   # inside the complement: no number sign
    assert rule("CL 987 NORTE # 654 - 321") == "CL 987 NORTE # 654 - 321"
    assert rule("CL 987 # 654 NORTE - 321") == "CL 987 # 654 NORTE - 321"


def test_f2_quadrant_position_differences_stay_distinct():
    assert rule("CL 987 N # 654 - 321") != rule("CL 987 # 654 N - 321")
    assert rule("AV 987 # 654 N - 321") != rule("AV 987 NORTE # 654 - 321")


# ---------------------------------------------------------------------------
# F3: a letter glued to a spelled quadrant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CL 987 3BNORTE 321", "CL 987 # 3 B NORTE - 321"),
        ("CL 987 15BNORTE 321", "CL 987 # 15 B NORTE - 321"),
        ("CL 987 3AOESTE 321", "CL 987 # 3 A OESTE - 321"),
        ("CL 987 3BN 321", "CL 987 # 3 B NORTE - 321"),
        ("CL 987 22NORTE 321", "CL 987 # 22 NORTE - 321"),
        ("CL 987 3B NORTE 321", "CL 987 # 3 B NORTE - 321"),
        ("CL 987 3 BSUR 321", "CL 987 # 3 B SUR - 321"),
    ],
)
def test_f3_letter_glued_to_quadrant_survives(raw, expected):
    assert rule(raw) == expected


def test_f3_the_quadrant_stays_in_its_own_segment():
    p = parse_address("CL 987 3BNORTE 321")
    assert (p.cross_letters, p.cross_quadrant, p.via_quadrant) == ("B", "NORTE", None)


def test_f3_typo_of_a_quadrant_is_still_repaired():
    assert rule("CL 987 # 654 NORTR - 321") == "CL 987 # 654 NORTE - 321"
    assert rule("DG 987 OOESTE # TV 654 - 321") == "DG 987 OESTE # TV 654 - 321"      # doubled letter: a typo, not a letter
    assert rule("DG 987 # 654 NNORTE - 321") == "DG 987 # 654 NORTE - 321"


# ---------------------------------------------------------------------------
# F4: a trailing quadrant after the plate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KR 987 # 654-321 OESTE SAN FERNANDO", "KR 987 # 654 - 321 OESTE"),
        ("KR 987 # 654-321 NORTE", "KR 987 # 654 - 321 NORTE"),
        ("KR 987 # 654-321 Nte.", "KR 987 # 654 - 321 NORTE"),
        ("KR 987 # 654-321 SUR", "KR 987 # 654 - 321 SUR"),
        ("KR 987 # 654-321 OE", "KR 987 # 654 - 321 OESTE"),
        ("KR 987 # 654-321 OESTE APTO 5", "KR 987 # 654 - 321 OESTE AP 5"),
    ],
)
def test_f4_trailing_quadrant_is_kept(raw, expected):
    assert rule(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "KR 987 # 654-321 ESTE DE CALI",       # ESTE has no cadastral occurrence: it starts a place name
        "KR 987 # 654-321 SAN OESTE",          # not the first word after the plate
    ],
)
def test_f4_does_not_invent_a_quadrant(raw):
    assert "OESTE" not in rule(raw) and not rule(raw).endswith(" ESTE")


def test_f4_via_quadrant_is_not_duplicated():
    assert rule("KR 987 OESTE # 654 - 321") == "KR 987 OESTE # 654 - 321"
    assert rule("KR 987 OESTE # 654 - 321 NORTE") == "KR 987 OESTE # 654 - 321 NORTE"


# ---------------------------------------------------------------------------
# F5: entre / con intersections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "Carrera 987 entre calle 654 y calle 321",
        "CR 987 A CON CL 654/321",
        "CL 987 CON 654 321",
        "Carrera 987 ENTRE calles 654 y 321",
    ],
)
def test_f5_intersections_never_invent_a_plate(raw):
    p = parse_address(raw)
    assert p.parse_ok and p.plate is None
    assert " - " not in rule(raw)


def test_f5_reads_via_plus_first_cross_and_flags_the_lost_text():
    assert rule("Carrera 987 entre calle 654 y calle 321") == "KR 987 # CL 654"
    assert rule("CR 987 A CON CL 654/321") == "KR 987 A # CL 654"
    assert is_truncated(parse_address("Carrera 987 entre calle 654 y calle 321"))
    assert not is_truncated(parse_address("CL 987 con KR 654"))
    assert parse_address("CL 987 con KR 654").plate is None        # a lone cross keeps its historical reading
    raw = "CR 987 A CON CL 654/321"
    assert assess_fidelity(raw, rule(raw)).risk == "high"       # a lost street number is flagged


def test_f5_does_not_touch_words_that_merely_contain_con_or_a_plate():
    assert rule("CL 987 # 654 - 321 CONJUNTO ALTOS") == "CL 987 # 654 - 321"
    assert rule("CL 987 # 654 - 321 con vista") == "CL 987 # 654 - 321"
    assert parse_address("CL 987 # 654 - 321 ENTRE CL 5 Y CL 6").plate == "321"


# ---------------------------------------------------------------------------
# F6: stage designators are units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "CLL 987 A OESTE # AV 654 OESTE C J PQ CENTRAL ET2 TR",
        "CL 987 # 654 ET 2",
        "CL 987 # 654 ETAPA 2",
        "CL 987 654 ET2",
        "CL 987 654 ETAPA 2 TORRE 3",
    ],
)
def test_f6_stage_number_is_not_the_plate(raw):
    assert parse_address(raw).plate is None


def test_f6_a_real_plate_before_the_stage_is_kept():
    assert rule("CL 987 # 654 - 321 ET 2") == "CL 987 # 654 - 321 ET 2"
    assert parse_address("CL 987 # 654 - 321 ET2").plate == "321"
    assert rule("CL 987 654 321 ETAPA 2") == "CL 987 # 654 - 321 ET 2"
    assert parse_address("CL 987 654 321 ET2").plate == "321"


# ---------------------------------------------------------------------------
# F7: ordinal indicators and double-encoded degree signs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Av. 987ª Oe. #654-321", "AV 987 OESTE # 654 - 321"),
        ("Cl. 987º A #654-321", "CL 987 A # 654 - 321"),
        ("CALLE987ANº654-321", "CL 987 A # 654 - 321"),
        ("Cra 987 Nº 654 - 321", "KR 987 # 654 - 321"),
        ("Cra 987 Nª 654 - 321", "KR 987 # 654 - 321"),
        ("CARRERA 987 Nª 654 321 APT 403", "KR 987 # 654 - 321 AP 403"),
        ("Cra 987 N° 654 - 321", "KR 987 # 654 - 321"),
        ("Cra 987 NÂ° 654 - 321", "KR 987 # 654 - 321"),
        ("Cra 987 NÂº 654 - 321", "KR 987 # 654 - 321"),
        ("Cra 987 #·654-321", "KR 987 # 654 - 321"),
        ("CRA31·654-321", "KR 31 # 654 - 321"),
    ],
)
def test_f7_symbols_are_not_letters(raw, expected):
    assert rule(raw) == expected


def test_f7_accents_on_letters_are_untouched():
    assert rule("CALLE 987 # 654 - 321 AP 5 NIÑO") == rule("CALLE 987 # 654 - 321 AP 5 NINO")
    assert rule("CL 987 # 654 - 321 BR ÁLAMOS") == rule("CL 987 # 654 - 321 BR ALAMOS")


def test_f7_fidelity_reads_the_same_symbols():
    for raw in ("Av. 987ª Oe. #654-321", "CALLE987ANº654-321", "Cra 987 Nº 654 - 321"):
        assert assess_fidelity(raw, rule(raw)).risk in ("none", "low")
    # the double-encoded sign is repaired in the fidelity fold too
    raw = "Cra 987 NÂº 654 - 321"
    assert assess_fidelity(raw, rule(raw)).risk in ("none", "low")


# ---------------------------------------------------------------------------
# F8: coordinates / long digit runs are not door digits (fidelity)
# ---------------------------------------------------------------------------
COORD = "CALLE 987 N, 654-321 // 1,2345678901234, -70,98765"


def test_f8_coordinate_tail_is_not_door_digits():
    r = assess_fidelity(COORD, "CL 987 N # 654 - 321")
    assert "door_digits_dropped" not in r.flags and r.risk in ("none", "low")


@pytest.mark.parametrize(
    "raw",
    [
        "CALLE 987 N, 654-321 1.2345678 -70.98765",
        "CALLE 987 N 654-321 3101234567",
        "CALLE 987 N 654-321 // 1,234 -70,987",
        "CALLE 987 N 654-321 // -70,98765",
        "CALLE 987 N 654-321 NPN 760010100000000123456",
    ],
)
def test_f8_decimals_and_long_runs_are_ignored(raw):
    r = assess_fidelity(raw, "CL 987 N # 654 - 321")
    assert "door_digits_dropped" not in r.flags and r.risk in ("none", "low", "medium")


@pytest.mark.parametrize(
    "out",
    ["CL 987 N # 654 - 322", "CL 987 N # 655 - 321", "CL 988 N # 654 - 321", "CL 987 N # 654", "CL 987 N"],
)
def test_f8_real_door_digits_are_still_compared(out):
    assert assess_fidelity(COORD, out).risk == "high"


# ---------------------------------------------------------------------------
# F9: high-risk non-OK rows carry no rule text
# ---------------------------------------------------------------------------
STATUSES = ["unparsed", "below_cutoff", "rejected_mismatch", "ambiguous", "door_not_catalogued", "something_new"]


@pytest.mark.parametrize("status", STATUSES)
def test_f9_high_risk_never_keeps_the_rule_text(status):
    assert catalog.keeps_rule_text(status, parse_address("CL 987 # 654 - 321"), "high") is False


@pytest.mark.parametrize("status", ["ambiguous", "door_not_catalogued", "something_new"])
@pytest.mark.parametrize("risk", [None, "none", "low", "medium"])
def test_f9_other_risks_keep_the_historical_rule(status, risk):
    assert catalog.keeps_rule_text(status, parse_address("CL 987"), risk) is True


FACE = [{"direccion": f"CL 987 # 654 - {p}", "lat": 3.4 + p * 2.5e-5, "lon": -76.5, "predial": f"P{p}", "manzana": "MZ7"}
        for p in (12, 16, 20, 11, 15)]
LOW = [[0.1, 0, 0, 0, 0]]


def _force(monkeypatch, risk):
    monkeypatch.setattr(service, "assess_fidelity",
                        lambda raw, output, cleaned=None: FidelityReport(risk, ("plate_changed",), 0.3, {}))


def _row(query, scores=None, **kw):
    kw.setdefault("ambiguity_delta", 0.0)
    return normalize_strict(make_stub(FACE, scores=scores), [query], gazetteer=None, **kw).iloc[0]


def _blank(row):
    return row["direccion_normalizada"] is None or pd.isna(row["direccion_normalizada"])


@pytest.mark.parametrize(("risk", "blank"), [("none", False), ("low", False), ("medium", False), ("high", True)])
def test_f9_door_not_catalogued_emission_by_risk(monkeypatch, risk, blank):
    _force(monkeypatch, risk)
    row = _row("CL 987 # 654 - 77", scores=LOW)
    assert row["estado"] != "OK" and row["catalog_status"] == "door_not_catalogued"
    assert _blank(row) is blank
    assert (row["fuente_normalizacion"] is None or pd.isna(row["fuente_normalizacion"])) is blank
    assert row["fidelity_risk"] == risk


def test_f9_keeps_estado_motivo_status_and_tier(monkeypatch):
    _force(monkeypatch, "none")
    keep = _row("CL 987 # 654 - 77", scores=LOW)
    _force(monkeypatch, "high")
    blanked = _row("CL 987 # 654 - 77", scores=LOW)
    for col in ("estado", "motivo", "catalog_status", "position_tier", "address_verified"):
        assert keep[col] == blanked[col], col
    assert list(keep.index) == list(blanked.index)


def test_f9_ok_rows_are_blanked_only_when_high(monkeypatch):
    # round 8: OK + high is now blanked (OK + medium stays emitted)
    _force(monkeypatch, "high")
    row = _row("CL 987 # 654 - 12")
    assert row["estado"] == "OK" and _blank(row)


# ---------------------------------------------------------------------------
# edge cases / determinism / idempotence / property grid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [None, "", "   ", "\n", 12345, b"CL 987", pytest.param("A" * 50000, id="very_long"), pytest.param("#" * 500, id="hashes"), "NO18", "º", "··"])
def test_boundaries_never_raise(raw):
    p = parse_address(raw)
    out = rule_canonical(p)
    assert out is None or isinstance(out, str)
    assert assess_fidelity(raw, out).risk in ("none", "low", "medium", "high")


def test_rule_canonical_tolerates_foreign_objects():
    assert rule_canonical(None) is None
    assert rule_canonical(object()) is None


@pytest.mark.parametrize(
    "raw",
    ["CRA 987 5B1 321 APT 201", "KR 987 D2 N 654C 321", "CL 987 3BNORTE 321", "KR 987 # 654-321 OESTE SAN FERNANDO",
     "Carrera 987 entre calle 654 y calle 321", "Av. 987ª Oe. #654-321", "CARRERA 987 NO18 61 CASA 36"],
)
def test_deterministic_and_idempotent(raw):
    first = rule(raw)
    assert first == rule(raw)
    assert rule(first) == first                     # a rendered rule text re-parses to itself


@pytest.mark.parametrize(
    "base",
    ["CL 987 # 654 - 321", "KR 987 A # 654 B - 321", "AV 987 NORTE # 654 - 321", "KR 987 # 654 - 321 OESTE",
     "CL 987 BIS # 654 - 321", "KR 987 G 6 # 654 S 1 - 321"],
)
def test_property_dropping_or_changing_a_door_digit_ends_high(base):
    import re

    assert assess_fidelity(base, base).risk in ("none", "low")
    runs = list(re.finditer(r"\d+", base))
    for m in runs:
        digits = m.group()
        changed = base[:m.start()] + str(int(digits) + 1) + base[m.end():]
        dropped = (base[:m.start()] + base[m.end():]).replace("  ", " ")
        assert assess_fidelity(base, changed).risk == "high", changed
        assert assess_fidelity(base, dropped).risk == "high", dropped
        assert assess_fidelity(base + " // 1,2345678901234, -70,98765", changed).risk == "high"


# ---------------------------------------------------------------------------
# regressions found while measuring the fresh sample (kept as characterization)
# ---------------------------------------------------------------------------
def test_via_designator_needs_a_single_street_letter():
    # ``NO`` / ``NTE`` / ``KRA`` are words, not street letters: the glued digit after them is a cross, not a suffix
    assert parse_address("CL987NO654D321BLOQUE3APTO101").via_suffix is None
    assert parse_address("CL987NTE5B41APTO104").via_suffix is None
    assert parse_address("CLLE 987 KRA654 A APT 602").via_suffix is None


def test_two_bare_numbers_after_a_designator_keep_the_historical_reading():
    p = parse_address("CALLE987C654 321EDIFICIOEJEMPLO APARTAMENTO 211")
    assert (p.via_letters, p.via_suffix, p.cross_number, p.plate) == ("C", None, "654", "321")


def test_a_lone_cross_after_con_keeps_the_historical_reading():
    # only numbers that would otherwise become a plate (two or more) trigger the intersection reading
    assert parse_address("CLLE 987 B2 con carrera 654").via_suffix is None
    assert not is_truncated(parse_address("CLLE 987 B2 con carrera 654"))


@pytest.mark.parametrize(
    "raw",
    ["CL 987 # 654 - 321 OF 601/1,2345678/-70,98765", "CALLE 987 N 654-321 LOCAL 501 1,2345678/-70,987654"],
)
def test_coordinates_glued_to_an_address_do_not_leak_into_the_units(raw):
    out = rule(raw)
    assert "2345678" not in out and "98765" not in out
    assert assess_fidelity(raw, out).risk in ("none", "low")


def test_chain_of_glued_designators_keeps_the_historical_reading():
    # ``B5B2 47`` reads as via letter + cross ``5B2`` + plate (all digits kept); not guessed as via ``B5B`` + cross 2
    assert rule("Cr 987 B5B2 47") == "KR 987 B # 5 B 2 - 47"
    assert rule("CALLE 987 B1 A9-80") == "CL 987 B # 1 A 9 - 80"


def test_f4_only_a_quadrant_right_after_the_plate_is_kept():
    # ``OE`` sits BEFORE the plate: it is the via quadrant (F10), never repeated after the plate
    out = rule("CRA 987B OE 654 T 321 APTO 101")
    assert out.count("OESTE") == 1 and out.startswith("KR 987 B OESTE #")
    assert "OESTE" not in rule("KR 987 # 654 - 321 T OE 5")
    assert rule("KR 987 # 654-321 OESTE") == "KR 987 # 654 - 321 OESTE"
    assert rule("KR 987 # 654 - 0321 NORTE") == "KR 987 # 654 - 321 NORTE"


# ---------------------------------------------------------------------------
# F10: the same `#`-less via reading as the explicit-hash one (second letter, short quadrant, glued quadrant)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Cra. 987B Oe., Normandia", "KR 987 B OESTE"),
        ("Av. 987 Bis Oe., COMUNA 1", "AV 987 BIS OESTE"),
        ("CRA 987A N 654 B 321", "KR 987 A N # 654 B - 321"),
        ("Cra 987 D N 654 - 321", "KR 987 D N # 654 - 321"),
        ("CL 987BNORTE 2F 321", "CL 987 B NORTE # 2 F - 321"),
        ("CR 987 654-321SUR BL 14 AP 103", "KR 987 # 654 - 321 SUR BLQ 14 AP 103"),
    ],
)
def test_f10_hashless_via_keeps_letters_and_quadrant(raw, expected):
    assert rule(raw) == expected


@pytest.mark.parametrize(
    ("raw", "explicit"),
    [
        ("CRA 987A N 654 B 321", "CRA 987A N # 654 B - 321"),
        ("Cra 987 D N 654 - 321", "Cra 987 D N # 654 - 321"),
    ],
)
def test_f10_matches_the_explicit_hash_reading(raw, explicit):
    assert rule(raw) == rule(explicit)


def test_f10_does_not_swallow_a_cross_letter_or_a_unit():
    assert rule("CL 987 B 654 321") == "CL 987 B # 654 - 321"
    assert parse_address("CL 987 B N").plate is None


@pytest.mark.parametrize("raw", ["C 987 A K 654", "CL 987 A T 654", "CL 987 A D 654 321", "CL 987 A C 654"])
def test_f10_a_via_type_letter_after_the_street_letter_is_the_cross_type(raw):
    # ``K`` / ``T`` / ``D`` / ``C`` are via-type aliases: they open the cross street, they are not a second letter
    assert parse_address(raw).via_letters == "A"


# ---------------------------------------------------------------------------
# glued tokens and quadrant positions (characterization with shifted numbers)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CL 987 # 654 - 66C", "CL 987 # 654 - 66"),                 # plate letters stay out of the rule text
        ("CL 987 # 654 - 08CA", "CL 987 # 654 - 08"),
        ("Cra 987C1 # 654 - 321", "KR 987 C 1 # 654 - 321"),
        ("Av987an # 654 - 321", "AV 987 A NORTE # 654 - 321"),
        ("Cra 987A9-250", "KR 987 A # 9 - 250"),
        ("Cra 987 5B1 321", "KR 987 # 5 B 1 - 321"),
        ("CL 987 # 654NORTE - 321", "CL 987 # 654 NORTE - 321"),
        ("CL 987 # 654 N - 321", "CL 987 # 654 N - 321"),
        ("CL 987 # 654 - 321 ETAPA 2 TORRE 3", "CL 987 # 654 - 321 ET 2 TO 3"),
    ],
)
def test_glued_token_readings_are_pinned(raw, expected):
    assert rule(raw) == expected


def test_two_bare_numbers_after_a_glued_via_designator_are_not_guessed():
    # ``987C1 654 321`` could be via 987C1 + cross 654 + plate 321 or via 987C + cross 1 + plate 654: stay strict and flagged
    raw = "Cra 987C1 654 321"
    assert rule(raw) == "KR 987 C # 1 - 654"
    assert assess_fidelity(raw, rule(raw)).risk == "high"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("CL 987 N # 654 - 321", "CL 987 # 654 N - 321"),
        ("AV 987 # 654 N - 321", "AV 987 NORTE # 654 - 321"),
        ("CL 987 # 654 - 321 NORTE", "CL 987 NORTE # 654 - 321"),
    ],
)
def test_quadrant_position_is_part_of_the_reading(a, b):
    assert rule(a) != rule(b)
    assert assess_fidelity(a, rule(b)).risk == "high"          # moving a quadrant is corruption, not a rewording


# ---------------------------------------------------------------------------
# emission matrix: every (estado, catalog_status, fidelity risk) combination
# ---------------------------------------------------------------------------
def _expected_keep(status, risk):
    if status == "unparsed":
        return False
    if status in ("below_cutoff", "rejected_mismatch"):
        return risk in ("none", "low")
    return risk != "high"


@pytest.mark.parametrize("status", ["unparsed", "below_cutoff", "rejected_mismatch", "ambiguous", "door_not_catalogued"])
@pytest.mark.parametrize("risk", ["none", "low", "medium", "high"])
def test_emission_matrix_for_non_ok_rows(monkeypatch, status, risk):
    _force(monkeypatch, risk)
    monkeypatch.setattr(service, "_catalog_decide", lambda *a, **k: status)
    row = _row("CL 987 # 654 - 77", scores=LOW)
    assert row["estado"] != "OK"
    assert (not _blank(row)) is _expected_keep(status, risk)
    assert row["catalog_status"] == status and row["fidelity_risk"] == risk
    assert _blank(row) == (row["fuente_normalizacion"] is None or pd.isna(row["fuente_normalizacion"]))


@pytest.mark.parametrize("risk", ["none", "low", "medium", "high"])
def test_emission_matrix_ok_rows_are_never_blanked(monkeypatch, risk):
    _force(monkeypatch, risk)
    row = _row("CL 987 # 654 - 12")
    # round 8: only OK + high is blanked
    assert row["estado"] == "OK" and _blank(row) is (risk == "high")


@pytest.mark.parametrize("raw", ["", "sin informacion de direccion", "hola", None])
def test_emission_unparseable_and_empty_inputs_never_emit(raw):
    row = normalize_strict(make_stub(FACE), [raw], gazetteer=None).iloc[0]
    assert row["estado"] != "OK" and _blank(row)


def test_emission_is_deterministic_across_batches(monkeypatch):
    _force(monkeypatch, "high")
    rows = ["CL 987 # 654 - 77", "CL 987 # 654 - 12", "hola"]
    a = normalize_strict(make_stub(FACE, scores=[[0.1, 0, 0, 0, 0]] * 3), rows, gazetteer=None, ambiguity_delta=0.0)
    b = normalize_strict(make_stub(FACE, scores=[[0.1, 0, 0, 0, 0]] * 3), rows, gazetteer=None, ambiguity_delta=0.0)
    assert a.equals(b)


def test_concurrent_parsing_is_deterministic():
    from concurrent.futures import ThreadPoolExecutor

    texts = ["CRA 987 5B1 321 APT 201", "KR 987 D2 N 654C 321", "CL 987 3BNORTE 321", "Carrera 987 entre calle 654 y calle 321",
             "CARRERA 987 NO18 61 CASA 36", "KR 987 # 654-321 OESTE SAN FERNANDO"] * 40
    expected = [rule(t) for t in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(rule, texts)) == expected


def test_long_digit_runs_are_garbage_not_door_digits():
    # the parser does not turn id-like runs into an address; the fidelity check does not count them as door digits
    raw = "CL 987 4030650 654 4741020"
    assert rule(raw) == "CL 987"
    assert assess_fidelity(raw, "CL 987 # 654 -").risk in ("none", "low")
    assert assess_fidelity(raw, "CL 987").risk == "high"          # the lost street number is still flagged
