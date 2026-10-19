"""Parser: number sign followed by punctuation, glued plates and plate-guard consequences."""

from __future__ import annotations

import pytest

from cali_address.parser import canonical, parse_address
from cali_address.service import normalize_strict


@pytest.mark.parametrize("raw", [
    "CALLE 987 NO, 654 - 100",
    "Calle 987 No, 654-100",
    "calle 987 no,654-100",
    "CALLE 987 NO : 654 - 100",
    "CALLE 987 NO. 654 - 100",
    "Calle 987 No.654-100",
    "Calle 987 N° 654-100",
    "Calle 987 Nº, 654-100",
    "Calle 987 N°. 654-100",
    "Calle 987 NO 654-100",
    "Calle 987 Nro. 654-100",
    "Calle 987 # 654-100",
])
def test_number_sign_variants_are_not_read_as_a_quadrant(raw):
    p = parse_address(raw)
    assert p.parse_ok
    assert (p.via_type, p.via_number, p.via_quadrant) == ("CL", "987", None)
    assert (p.cross_number, p.plate) == ("654", "100")
    assert canonical(p) == "CL 987 # 654 - 100"


def test_full_complex_input_keeps_via_plate_and_units():
    p = parse_address("CALLE 987 NO, 654 - 100 APTO 302 TORRE/BLOQUE 18")
    assert (p.via_number, p.cross_number, p.plate) == ("987", "654", "100")
    assert p.via_quadrant is None
    assert "AP 302" in canonical(p)


@pytest.mark.parametrize("raw, quadrant", [
    ("CALLE 987 OESTE # 654 - 100", "OESTE"),
    ("CALLE 987 NORTE # 654 - 100", "NORTE"),
    ("CL 987 N # 654 - 100", None),   # lone N stays a letter-like token, not a number sign
])
def test_real_quadrants_are_still_read(raw, quadrant):
    p = parse_address(raw)
    assert p.via_number == "987" and p.plate == "100"
    assert p.via_quadrant == quadrant


@pytest.mark.parametrize("raw", ["NO, ", "NO,", "CALLE NO, 8", "NO 8", "N°", ",", "NO,,,5"])
def test_malformed_number_sign_inputs_do_not_raise(raw):
    parse_address(raw)


def test_no_without_digit_after_is_untouched():
    p = parse_address("CALLE 987 NO, SAN FERNANDO")
    assert p.via_number == "987"


# --- glued plates ---------------------------------------------------------
@pytest.mark.parametrize("raw, plate", [
    ("CL 987 # 654 - 08CA", "08"),
    ("CL 987 # 654 - 8CA", "8"),
    ("CL 987 # 654 - 08 CA", "08"),
    ("CL 987 # 654 - 12A", "12"),
])
def test_glued_plate_letters_do_not_hide_the_plate(raw, plate):
    p = parse_address(raw)
    assert p.parse_ok
    assert p.plate == plate


def test_glued_plate_is_checked_by_the_production_rule(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 987 # 654 - 12"}], scores=[[0.95]], threshold=0.5)
    row = normalize_strict(stub, ["CL 987 # 654 - 08CA"], gazetteer=None, ambiguity_delta=0.0).iloc[0]
    assert "placa" in str(row.to_dict())  # the plate mismatch is reported, not silently skipped
