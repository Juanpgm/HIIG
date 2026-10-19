"""Guard on the cadastral override of ``AddressNormalizer`` (research path).

``normalize_batch`` used to replace a correct rule output with the nearest cadastral
address whenever the fused score cleared the cutoff, without checking that via, cross and
plate agree (JEV: 100 % agreement when the rule output is kept, 7 % when overridden). The
override is now accepted only when the structured base is identical; otherwise the rule
output is kept and ``matched`` is False. The tests use a model-free normalizer: only the
attributes ``_row`` / the rule-only ``normalize_batch`` read are populated.
"""

from __future__ import annotations

import numpy as np
import pytest

from cali_address.inference import AddressNormalizer, base_mismatches
from cali_address.parser import canonical, match_key, parse_address

NAN = float("nan")


def make_normalizer(addresses: list[str], threshold: float = 0.73) -> AddressNormalizer:
    """Model-free normalizer over ``addresses`` (rule-only candidates)."""
    n = AddressNormalizer.__new__(AddressNormalizer)
    n._direccion = np.array(addresses, dtype=object)
    size = len(addresses)
    n._predial = np.array([f"P{i}" for i in range(size)], dtype=object)
    n._manzana = np.array(["M1"] * size, dtype=object)
    n._comuna = np.array(["1"] * size, dtype=object)
    n._barrio = np.array(["B1"] * size, dtype=object)
    n._lat = np.full(size, 3.4)
    n._lon = np.full(size, -76.5)
    n._n_predios = np.ones(size, dtype=int)
    n._exact = {}
    for i, d in enumerate(addresses):
        n._exact.setdefault(match_key(d), i)
    n._parsed_docs = {}
    n.threshold = threshold
    n.weights = {"sim": 0.70, "fuzz": 0.10, "struct": 0.10, "geo": 0.10}
    n.max_len = 40
    return n


def row_for(raw, cadastral: str, fused: float = 0.95, threshold: float = 0.73) -> dict:
    """Run ``_row`` as if ``cadastral`` were the top candidate with score ``fused``."""
    n = make_normalizer([cadastral], threshold=threshold)
    parsed = parse_address(raw)
    canon = canonical(parsed, style="spaced", with_complement=True) if parsed.parse_ok else ""
    return n._row(raw, canon, parsed, [(fused, fused, 0.9, 0.9, 0, NAN)], 20, threshold)


def batch(raw: str, gazetteer: list[str]) -> dict:
    n = make_normalizer(gazetteer)
    return n.normalize_batch([raw], use_model=False).iloc[0].to_dict()


# ---------------------------------------------------------------------------
# 1. Regression cases: a neighbouring plate / via must never overwrite the rule output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, neighbour, expected", [
    ("Carrera 987A #654-84", "KR 987 A # 654 - 48", "KR 987 A # 654 - 84"),
    ("Calle 876 #543-87", "CL 876 # 543 - 78", "CL 876 # 543 - 87"),
    ("CALLE 765 #432-09", "CL 765 # 432 - 39", "CL 765 # 432 - 09"),
    ("Calle 654#321-25", "CL 655 # 321 - 25", "CL 654 # 321 - 25"),
    ("CR 987 654-12 AP 501", "KR 987 # 653 - 12", "KR 987 # 654 - 12 AP 501"),
])
def test_neighbour_with_high_score_does_not_overwrite_rule_output(raw, neighbour, expected):
    row = row_for(raw, neighbour, fused=0.99)
    assert row["canonical_address"] == expected
    assert row["matched"] is False
    assert "cadastral_base_mismatch" in row["parse_notes"]
    # the rejected neighbour must not leave its parcel data on an unmatched row
    assert row["matched_cadastral_address"] is None
    assert row["numero_predial_nacional"] is None
    assert np.isnan(row["lat"]) and np.isnan(row["lon"])
    assert row["override_rejected"] is True
    assert "cadastral_candidate_rejected" in row["parse_notes"]


@pytest.mark.parametrize("raw, neighbour, expected", [
    ("Carrera 987A #654-84", "KR 987 A # 654 - 48", "KR 987 A # 654 - 84"),
    ("Calle 876 #543-87", "CL 876 # 543 - 78", "CL 876 # 543 - 87"),
    ("CALLE 765 #432-09", "CL 765 # 432 - 39", "CL 765 # 432 - 09"),
])
def test_rule_only_batch_keeps_rule_output_when_exact_base_is_absent(raw, neighbour, expected):
    out = batch(raw, [neighbour])
    assert out["canonical_address"] == expected
    assert out["matched"] is False


def test_rule_only_batch_accepts_the_exact_base():
    out = batch("Carrera 987A #654-84", ["KR 987 A # 654 - 48", "KR 987 A # 654 - 84"])
    assert out["canonical_address"] == "KR 987 A # 654 - 84"
    assert out["matched"] is True
    assert out["matched_cadastral_address"] == "KR 987 A # 654 - 84"


def test_complex_address_keeps_via_and_plate():
    raw = "CALLE 987 NO, 654 - 100 APTO 302 TORRE/BLOQUE 18"
    row = row_for(raw, "CL 987 # 654 - 100", fused=0.9)
    assert row["canonical_address"] == "CL 987 # 654 - 100 BLQ 18 AP 302"
    assert row["matched"] is True


# ---------------------------------------------------------------------------
# 2. Boundary values on the plate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cadastral, accepted", [
    ("CL 876 # 543 - 87", True),    # equal
    ("CL 876 # 543 - 088", False),  # differs by one
    ("CL 876 # 543 - 89", False),   # differs by two
    ("CL 876 # 543 - 0087", True),  # leading zeros are not a difference
    ("CL 876 # 543 -", False),      # no plate on the cadastral side
    ("CL 876 # 543 - 0", False),    # plate 0 vs 87
])
def test_plate_boundaries(cadastral, accepted):
    row = row_for("Calle 876 #543-87", cadastral)
    assert row["matched"] is accepted
    assert row["canonical_address"] == "CL 876 # 543 - 87"


def test_missing_plate_on_the_rule_side_is_not_overwritten_with_a_plated_address():
    row = row_for("Calle 876 # 543", "CL 876 # 543 - 87")
    assert row["matched"] is False
    assert "plate" in row["parse_notes"]


def test_plate_zero_on_both_sides_is_equal():
    row = row_for("CL 876 # 543 - 0", "CL 876 # 543 - 0")
    assert row["matched"] is True


# ---------------------------------------------------------------------------
# 3. Unparsed / empty / malformed inputs never receive a cadastral address
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "SIN ESPECIFICAR", "", "   ", None, "Cali", "Cali, Valle del Cauca", "123456", "12 34 56",
    "x" * 400, "SIN ESPECIFICAR " * 30,
])
def test_unparsed_rows_get_no_cadastral_address(raw):
    row = row_for(raw, "COR PICHINDE VER PICHINDE U", fused=0.99)
    assert row["canonical_address"] in ("", None)
    assert row["matched"] is False
    assert "no_rule_parse" in row["parse_notes"]


def test_unparsed_row_in_batch_has_no_cadastral_address():
    n = make_normalizer(["CL 987 # 654 - 25"])
    out = n.normalize_batch(["SIN ESPECIFICAR", "   ", "CL 987 # 654 - 25"], use_model=False)
    assert list(out["matched"]) == [False, False, True]
    assert out["canonical_address"].tolist()[:2] == ["", ""]


def test_low_score_with_parsed_rule_output_keeps_rule_output_without_marker():
    row = row_for("CL 876 # 543 - 87", "CL 876 # 543 - 87", fused=0.2)
    assert row["canonical_address"] == "CL 876 # 543 - 87"
    assert row["matched"] is False
    assert "cadastral_base_mismatch" not in row["parse_notes"]


# ---------------------------------------------------------------------------
# 4. Units and legacy via types are never copied raw from the cadastre
# ---------------------------------------------------------------------------
def test_cadastral_only_units_are_not_emitted():
    row = row_for("Carrera 987A #654-84", "KR 987 A # 654 - 84 AP 101 TO 2")
    assert row["matched"] is True
    assert row["canonical_address"] == "KR 987 A # 654 - 84"


def test_input_units_are_kept_instead_of_cadastral_units():
    row = row_for("Carrera 987A #654-84 apto 302", "KR 987 A # 654 - 84 AP 101 TO 2")
    assert row["canonical_address"] == "KR 987 A # 654 - 84 AP 302"


def test_input_units_with_different_values_win_over_cadastre():
    row = row_for("CL 987 # 654 - 25 APTO 12", "CL 987 # 654 - 25 AP 99")
    assert row["matched"] is True
    assert row["canonical_address"] == "CL 987 # 654 - 25 AP 12"


@pytest.mark.parametrize("legacy, expected_prefix", [
    ("C 5 # 654 - 25", "CL 5"),
    ("K 5 # 654 - 25", "KR 5"),
    ("T 5 # 654 - 25", "TV 5"),
    ("D 5 # 654 - 25", "DG 5"),
])
def test_legacy_via_types_are_re_rendered(legacy, expected_prefix):
    raw = f"{expected_prefix} # 654 - 25"
    cad = parse_address(legacy)
    assert cad.parse_ok
    row = row_for(raw, legacy)
    assert row["matched"] is True
    assert row["canonical_address"] == f"{expected_prefix} # 654 - 25"


# ---------------------------------------------------------------------------
# 5. base_mismatches: field-level behaviour
# ---------------------------------------------------------------------------
def _mm(a: str, b: str) -> list[str]:
    return base_mismatches(parse_address(a), parse_address(b))


@pytest.mark.parametrize("a, b", [
    ("CL 987 # 654 - 25", "CL 987 # 654 - 25"),
    ("Calle 987 #654-25", "CL 987 # 654 - 25 AP 3"),            # complement is not part of the base
    ("KR 987A # 654-84", "KR 987 A # 654 - 84"),               # glued letter
    ("CL 876N # 654 - 25", "CL 876 NORTE # 654 - 25"),            # cardinal letter == spelled quadrant
    ("AV 876 N # 543 - 06", "AV 876 NORTE # 543 - 06"),
    ("AV 876 # 543 N - 06", "AV 876 # 543 NORTE - 06"),           # same segment (cross)
    ("CL 1C4 # 5 - 3", "CL 1 C 4 # 5 - 03"),                 # glued alphanumerics
    ("KR 5B1 # 94C1 - 7", "KR 5 B 1 # 94 C 1 - 7"),
])
def test_equivalent_bases(a, b):
    assert _mm(a, b) == []


@pytest.mark.parametrize("a, b, field", [
    ("CL 987 # 654 - 25", "KR 5 # 654 - 25", "via_type"),
    ("CL 987 # 654 - 25", "CL 655 # 321 - 25", "via_number"),
    ("CL 987 # 654 - 25", "CL 987 # 39 - 25", "cross_number"),
    ("CL 987 # 654 - 25", "CL 987 # 654 - 26", "plate"),
    ("CL 987 A # 654 - 25", "CL 987 B # 654 - 25", "via_letters"),
    ("CL 987 # 654 A - 25", "CL 987 # 654 - 25", "cross_letters"),
    ("KR 987 H 1 # 654 - 21", "KR 987 H 3 # 654 - 21", "via_suffix"),
    ("CL 987 BIS # 654 - 25", "CL 987 # 654 - 25", "via_bis"),
    ("CL 987 NORTE # 654 - 25", "CL 987 SUR # 654 - 25", "via_quadrant"),
    ("CL 987 # 654 - 25", "CL 987 NORTE # 654 - 25", "via_quadrant"),
    ("CL 987 # 654 NORTE - 25", "CL 987 # 654 SUR - 25", "cross_quadrant"),
])
def test_each_base_field_is_detected(a, b, field):
    assert field in _mm(a, b)


def test_base_mismatches_is_symmetric_and_deterministic():
    a, b = "CL 987 # 654 - 25", "CL 6 # 654 - 26"
    assert sorted(_mm(a, b)) == sorted(_mm(b, a))
    assert _mm(a, b) == _mm(a, b)


# ---------------------------------------------------------------------------
# 6. Idempotence, determinism, multi-address and unicode inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "Carrera 987A #654-84", "CALLE 987 NO, 654 - 100 APTO 302", "CR 987 654-12 AP 501",
])
def test_normalizing_a_canonical_output_is_idempotent(raw):
    gaz = ["KR 987 A # 654 - 84", "CL 987 # 654 - 100", "KR 987 # 654 - 12", "CL 987 # 654 - 101"]
    n = make_normalizer(gaz)
    first = n.normalize_batch([raw], use_model=False).iloc[0]["canonical_address"]
    second = n.normalize_batch([first], use_model=False).iloc[0]["canonical_address"]
    assert first == second


def test_is_deterministic():
    n = make_normalizer(["KR 987 A # 654 - 48", "KR 987 A # 654 - 84"])
    a = n.normalize_batch(["Carrera 987A #654-84"] * 3, use_model=False)
    assert a["canonical_address"].nunique() == 1


def test_multi_address_string_uses_the_first_address_only():
    row = row_for("Carrera 987A #654-84; Calle 987 # 654 - 25", "KR 987 A # 654 - 84")
    assert row["matched"] is True
    assert row["canonical_address"] == "KR 987 A # 654 - 84"


def test_unicode_accents_and_lowercase():
    row = row_for("carrerá 987a #654-84", "KR 987 A # 654 - 84")
    assert row["canonical_address"] == "KR 987 A # 654 - 84"
    assert row["matched"] is True


def test_very_long_parseable_string_does_not_crash():
    raw = "Calle 987 # 654 - 25 " + "apartamento torre bloque " * 20
    row = row_for(raw, "CL 987 # 654 - 25")
    assert row["canonical_address"].startswith("CL 987 # 654 - 25")
    assert row["matched"] is True
