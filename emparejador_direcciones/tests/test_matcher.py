"""Tests for emparejador.matcher: penalties, vetoes, thresholds, symmetry."""

from __future__ import annotations

import json

import pytest

from emparejador.matcher import (
    DEFAULT_PLATE_TOLERANCE,
    DEFAULT_THRESHOLD,
    FieldComparison,
    InvalidThresholdError,
    InvalidToleranceError,
    MatchResult,
    match,
)


# ---------------------------------------------------------------------------
# 1. Unparseable input never matches and never raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        (None, "KR 1 # 4 - 80"),
        ("", "KR 1 # 4 - 80"),
        ("KR 1 4 80", "KR 1 # 4 - 80"),
        ("SIN DIRECCION", "SIN DIRECCION"),
        ("KR 1 # 4 - 80; CL 5 # 3 - 2", "KR 1 # 4 - 80"),
        (123, "KR 1 # 4 - 80"),
    ],
)
def test_unparseable_never_matches_and_never_raises(a: object, b: object) -> None:
    result = match(a, b)  # type: ignore[arg-type]
    assert result.is_match is False
    assert result.score == 0.0


def test_unparseable_reason_mentions_no_canonical() -> None:
    result = match("KR 1 4 80", "KR 1 # 4 - 80")
    assert "no canónica" in result.reason


# ---------------------------------------------------------------------------
# 1b. Finding 1: an unbounded plate digit run must never raise
# ---------------------------------------------------------------------------


def test_extremely_long_plate_digit_run_never_raises() -> None:
    result = match("KR 1 # 4 - " + "9" * 4301, "KR 1 # 4 - 8")
    assert result.is_match is False
    assert result.score == 0.0


def test_plate_leading_zero_padding_scores_equivalent() -> None:
    result = match("KR 1 # 4 - 0000080", "KR 1 # 4 - 80")
    assert result.score == 1.0
    assert result.is_match is True


# ---------------------------------------------------------------------------
# 2. Exact / equivalent matches -> 1.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 1 # 4 - 80", "KR 1 # 4 - 80"),
        ("KR 1 # 4 - 80", "KR 1 # 4 - 080"),
        ("KR 1 # 4 - 80", "kr 1 #4-80"),
        ("KR 1 # 4 - 80", "KR 01 # 04 - 80"),
        ("KR 1 # 4 - 80", "Cra. 1 # 4 - 80"),
        ("KR 96 F # 96 - 66", "KR 96F # 96 - 66"),
    ],
)
def test_exact_and_equivalent_score_one(a: str, b: str) -> None:
    result = match(a, b)
    assert result.score == 1.0
    assert result.is_match is True


# ---------------------------------------------------------------------------
# 3. Hard vetoes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 1 # 4 - 80", "CL 1 # 4 - 80"),  # via_type differs
        ("KR 1 # 4 - 80", "KR 2 # 4 - 80"),  # via_number differs
        ("KR 1 # 4 - 80", "KR 1 # 10 - 80"),  # cross_number differs
        ("KR 1 # 4 - 80", "KR 1 # - 80"),  # cross present on one side only
        ("KR 1 # 4 - 80", "KR 1 # 4 - 82"),  # plate differs (tolerance 0)
        ("KR 1 # 4 - 80", "KR 1 # 4"),  # plate present on one side only
        ("CL 10 # 42 - 02 AP 501", "CL 10 # 42 - 02 AP 502"),  # complement conflict
        ("CL 10 # 42 - 02 AP 501", "CL 10 # 42 - 02 AP 501 TO 3"),  # complement conflict (extra kind)
    ],
)
def test_hard_vetoes_force_no_match(a: str, b: str) -> None:
    result = match(a, b)
    assert result.is_match is False
    assert result.score == 0.0
    assert any(field.veto for field in result.fields)


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 1 # 4 - 80", "CL 1 # 4 - 80"),
        ("KR 1 # 4 - 80", "KR 2 # 4 - 80"),
    ],
)
def test_veto_ignores_threshold_zero(a: str, b: str) -> None:
    result = match(a, b, threshold=0.0)
    assert result.is_match is False


def test_plate_trailing_letter_vetoes() -> None:
    """A plate letter (e.g. corner unit "80 A") is a different parcel from
    the bare plate "80": this is documented as a hard veto (finding 4/10)."""
    result = match("KR 1 # 4 - 80 A", "KR 1 # 4 - 80")
    assert result.is_match is False
    assert result.score == 0.0
    assert any(f.field == "plate" and f.veto for f in result.fields)


# ---------------------------------------------------------------------------
# 4. Soft penalties per field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 1 # 4 - 80", "KR 1 A # 4 - 80"),  # via_letters one-sided
        ("KR 26 H 3 # 73 - 10", "KR 26 H # 73 - 10"),  # via_suffix one-sided
        ("CL 12 # 48 BIS - 31", "CL 12 # 48 - 31"),  # cross_bis one-sided
    ],
)
def test_one_sided_soft_penalty_gives_085(a: str, b: str) -> None:
    result = match(a, b)
    assert result.score == pytest.approx(0.85)
    assert result.is_match is False  # below default threshold 0.90


def test_conflict_soft_penalty_via_letters() -> None:
    result = match("KR 1 A # 4 - 80", "KR 1 B # 4 - 80")
    assert result.score == pytest.approx(0.70)


def test_conflict_soft_penalty_quadrant() -> None:
    result = match("CL 5 NORTE # 4 - 80", "CL 5 SUR # 4 - 80")
    assert result.score == pytest.approx(0.70)


def test_two_missing_fields_multiply() -> None:
    result = match("AV 15 OESTE # 9 OESTE - 137", "AV 15 # 9 - 137")
    assert result.score == pytest.approx(0.7225)
    assert result.is_match is False


# ---------------------------------------------------------------------------
# 4b. Finding 4: one-sided AND conflict coverage for every soft field
# (mutation-testing gap closed here; via_letters, cross_bis, via_bis and
# cross_type/via_quadrant-conflict are already covered above)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 26 1 # 73 - 10", "KR 26 # 73 - 10"),  # via_suffix
        ("KR 26 1 A # 73 - 10", "KR 26 1 # 73 - 10"),  # via_suffix_letters
        ("CL 12 BIS # 48 - 31", "CL 12 # 48 - 31"),  # via_bis
        ("CL 5 NORTE # 4 - 80", "CL 5 # 4 - 80"),  # via_quadrant (alone)
        ("KR 1 # 4 G - 80", "KR 1 # 4 - 80"),  # cross_letters
        ("KR 1 # 4 11 - 80", "KR 1 # 4 - 80"),  # cross_suffix
        ("KR 1 # 4 11 G - 80", "KR 1 # 4 11 - 80"),  # cross_suffix_letters
        ("KR 1 # 4 OESTE - 80", "KR 1 # 4 - 80"),  # cross_quadrant
    ],
)
def test_every_soft_field_one_sided_gives_085(a: str, b: str) -> None:
    result = match(a, b)
    assert result.score == pytest.approx(0.85)
    assert result.is_match is False


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 26 1 # 73 - 10", "KR 26 2 # 73 - 10"),  # via_suffix
        ("KR 26 1 A # 73 - 10", "KR 26 1 B # 73 - 10"),  # via_suffix_letters
        ("CL 5 NORTE # 4 - 80", "CL 5 SUR # 4 - 80"),  # via_quadrant (alone)
        ("KR 1 # 4 G - 80", "KR 1 # 4 H - 80"),  # cross_letters
        ("KR 1 # 4 11 - 80", "KR 1 # 4 12 - 80"),  # cross_suffix
        ("KR 1 # 4 11 G - 80", "KR 1 # 4 11 H - 80"),  # cross_suffix_letters
        ("KR 1 # 4 OESTE - 80", "KR 1 # 4 NORTE - 80"),  # cross_quadrant
    ],
)
def test_every_soft_field_conflict_gives_070(a: str, b: str) -> None:
    result = match(a, b)
    assert result.score == pytest.approx(0.70)
    assert result.is_match is False


# ---------------------------------------------------------------------------
# 5. cross_type
# ---------------------------------------------------------------------------


def test_cross_type_one_sided_matches() -> None:
    result = match("CL 25 NORTE # AV 6 - 30", "CL 25 NORTE # 6 - 30")
    assert result.score == pytest.approx(0.97)
    assert result.is_match is True


def test_cross_type_conflict() -> None:
    result = match("CL 25 NORTE # AV 6 - 30", "CL 25 NORTE # KR 6 - 30")
    assert result.score == pytest.approx(0.70)
    assert result.is_match is False


# ---------------------------------------------------------------------------
# 6. Complement rules
# ---------------------------------------------------------------------------


def test_complement_equal_scores_one() -> None:
    result = match("CL 10 # 42 - 02 AP 501 TO 3", "CL 10 # 42 - 02 AP 501 TO 3")
    assert result.score == 1.0


def test_complement_reordered_alias_zeros_scores_one() -> None:
    result = match("CL 10 # 42 - 02 AP 501 TO 3", "CL 10 # 42 - 2 TO 3 APTO 0501")
    assert result.score == 1.0
    assert result.is_match is True


@pytest.mark.parametrize(
    "a,b",
    [
        ("CL 10 # 42 - 02 AP 501 TO 3", "CL 10 # 42 - 02"),
        ("CL 10 # 42 - 02", "CL 10 # 42 - 02 AP 501 TO 3"),
    ],
)
def test_complement_one_sided_scores_098_match(a: str, b: str) -> None:
    result = match(a, b)
    assert result.score == pytest.approx(0.98)
    assert result.is_match is True


def test_complement_conflict_vetoes() -> None:
    result = match("CL 10 # 42 - 02 AP 501", "CL 10 # 42 - 02 AP 502")
    assert result.is_match is False
    assert result.score == 0.0


def test_complement_both_absent_scores_one() -> None:
    result = match("KR 1 # 4 - 80", "KR 1 # 4 - 80")
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# 7. Plate tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tolerance,other_plate,expected",
    [
        (0, "82", None),  # veto
        (2, "82", 0.96),
        (2, "83", None),  # veto (outside tolerance)
        (2, "080", 1.0),
    ],
)
def test_plate_tolerance_table(tolerance: int, other_plate: str, expected: float | None) -> None:
    result = match("KR 1 # 4 - 80", f"KR 1 # 4 - {other_plate}", plate_tolerance=tolerance)
    if expected is None:
        assert result.is_match is False
        assert result.score == 0.0
    else:
        assert result.score == pytest.approx(expected)


def test_negative_plate_tolerance_raises() -> None:
    with pytest.raises(InvalidToleranceError):
        match("KR 1 # 4 - 80", "KR 1 # 4 - 82", plate_tolerance=-1)


# ---------------------------------------------------------------------------
# 8. Symmetry
# ---------------------------------------------------------------------------


_SYMMETRY_PAIRS = [
    ("KR 1 # 4 - 80", "KR 1 # 4 - 80"),
    ("KR 1 # 4 - 80", "KR 1 A # 4 - 80"),
    ("KR 1 # 4 - 80", "CL 1 # 4 - 80"),
    ("KR 1 # 4 - 80", "KR 2 # 4 - 80"),
    ("KR 1 # 4 - 80", "KR 1 # 10 - 80"),
    ("KR 1 # 4 - 80", "KR 1 # - 80"),
    ("KR 1 # 4 - 80", "KR 1 # 4 - 82"),
    ("KR 1 # 4 - 80", "KR 1 # 4"),
    ("CL 10 # 42 - 02 AP 501", "CL 10 # 42 - 02 AP 502"),
    ("CL 10 # 42 - 02 AP 501 TO 3", "CL 10 # 42 - 02"),
    ("CL 25 NORTE # AV 6 - 30", "CL 25 NORTE # 6 - 30"),
    ("AV 15 OESTE # 9 OESTE - 137", "AV 15 # 9 - 137"),
    ("KR 1 A # 4 - 80", "KR 1 B # 4 - 80"),
    ("KR 1 4 80", "KR 1 # 4 - 80"),
]


@pytest.mark.parametrize("a,b", _SYMMETRY_PAIRS)
def test_match_is_symmetric(a: str, b: str) -> None:
    forward = match(a, b)
    backward = match(b, a)
    assert forward.is_match == backward.is_match
    assert forward.score == backward.score
    forward_fields = {(f.field, f.status, f.penalty, f.veto) for f in forward.fields}
    backward_fields = {(f.field, f.status, f.penalty, f.veto) for f in backward.fields}
    assert forward_fields == backward_fields


# ---------------------------------------------------------------------------
# 9. Threshold behavior
# ---------------------------------------------------------------------------


def test_passes_boundary_090() -> None:
    from emparejador.matcher import _passes

    assert _passes(0.90, 0.90) is True
    assert _passes(0.899, 0.90) is False


def test_threshold_085_boundary() -> None:
    at = match("KR 1 # 4 - 80", "KR 1 A # 4 - 80", threshold=0.85)
    above = match("KR 1 # 4 - 80", "KR 1 A # 4 - 80", threshold=0.8501)
    assert at.is_match is True
    assert above.is_match is False


def test_threshold_097_boundary() -> None:
    at = match("CL 25 NORTE # AV 6 - 30", "CL 25 NORTE # 6 - 30", threshold=0.97)
    above = match("CL 25 NORTE # AV 6 - 30", "CL 25 NORTE # 6 - 30", threshold=0.9701)
    assert at.is_match is True
    assert above.is_match is False


def test_threshold_one() -> None:
    identical = match("KR 1 # 4 - 80", "KR 1 # 4 - 80", threshold=1.0)
    near = match("CL 10 # 42 - 02 AP 501 TO 3", "CL 10 # 42 - 02", threshold=1.0)
    equivalent = match("KR 1 # 4 - 80", "KR 1 # 4 - 080", threshold=1.0)
    assert identical.is_match is True
    assert near.is_match is False
    assert equivalent.is_match is True


def test_threshold_zero() -> None:
    two_missing = match("AV 15 OESTE # 9 OESTE - 137", "AV 15 # 9 - 137", threshold=0.0)
    vetoed = match("KR 1 # 4 - 80", "KR 2 # 4 - 80", threshold=0.0)
    assert two_missing.is_match is True
    assert vetoed.is_match is False


@pytest.mark.parametrize("bad_threshold", [1.01, -0.1, float("nan"), "0.9"])
def test_invalid_threshold_raises(bad_threshold: object) -> None:
    with pytest.raises(InvalidThresholdError):
        match("KR 1 # 4 - 80", "KR 1 # 4 - 80", threshold=bad_threshold)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. Dead band: no reachable score lies in (0.85, 0.9126) (finding 5)
#
# The proof lives in ``tests/test_matcher_review.py`` (S3): it enumerates the
# penalties from the single registry ``matcher._PENALTIES`` that the matcher
# itself applies, and fails when a ``_*_PENALTY`` constant is not registered.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. Result shape
# ---------------------------------------------------------------------------


def test_to_dict_keys_and_json_serializable() -> None:
    result = match("KR 1 # 4 - 80", "KR 1 A # 4 - 80")
    payload = result.to_dict()
    for key in ("is_match", "score", "threshold", "reason", "fields", "warnings"):
        assert key in payload
    json.dumps(payload)  # must not raise
    assert 0.0 <= payload["score"] <= 1.0


def test_field_comparison_dataclass_fields() -> None:
    result = match("KR 1 # 4 - 80", "KR 1 A # 4 - 80")
    for field in result.fields:
        assert isinstance(field, FieldComparison)
        assert field.status in {
            "igual",
            "equivalente",
            "ausente_en_ambos",
            "solo_en_uno",
            "difiere",
            "dentro_tolerancia",
        }


def test_default_constants() -> None:
    assert DEFAULT_THRESHOLD == 0.90
    assert DEFAULT_PLATE_TOLERANCE == 0


def test_match_result_is_frozen_dataclass_like() -> None:
    result = match("KR 1 # 4 - 80", "KR 1 # 4 - 80")
    assert isinstance(result, MatchResult)
    assert hasattr(result, "parsed_a")
    assert hasattr(result, "parsed_b")


# ---------------------------------------------------------------------------
# 12. Finding 7: threshold is formatted with {:g}, not a fixed 2-decimal form
# ---------------------------------------------------------------------------


def test_reason_formats_threshold_with_g_not_fixed_decimals() -> None:
    result = match("KR 1 # 4 - 80", "KR 1 A # 4 - 80", threshold=0.8501)
    assert "0.8501" in result.reason
    assert "0.85 " not in result.reason and not result.reason.endswith("0.85")


# ---------------------------------------------------------------------------
# 13. Finding 8: sin_placa / sin_cruce warnings for open-ended addresses
# ---------------------------------------------------------------------------


def test_warnings_empty_for_full_addresses() -> None:
    result = match("KR 1 # 4 - 80", "KR 1 # 4 - 80")
    assert result.warnings == ()


def test_warning_sin_placa_when_both_sides_lack_plate() -> None:
    result = match("KR 1 # 4", "KR 1 # 4")
    assert result.is_match is True
    assert "sin_placa" in result.warnings


def test_warning_sin_cruce_when_both_sides_lack_cross() -> None:
    result = match("KR 1 # - 80", "KR 1 # - 80")
    assert result.is_match is True
    assert "sin_cruce" in result.warnings


# ---------------------------------------------------------------------------
# 14. Finding 10: FieldComparison.a/b are always str | None, and the
# complement veto reason is human-readable, not a tuple repr.
# ---------------------------------------------------------------------------


def test_field_comparison_values_are_strings_or_none() -> None:
    result = match("CL 10 # 42 - 02 AP 501", "CL 10 # 42 - 02 AP 502")
    for f in result.fields:
        assert f.a is None or isinstance(f.a, str)
        assert f.b is None or isinstance(f.b, str)


def test_complement_veto_reason_is_human_readable() -> None:
    result = match("CL 10 # 42 - 02 AP 501", "CL 10 # 42 - 02 AP 502")
    assert "AP 501" in result.reason
    assert "AP 502" in result.reason
    assert "(('AP'" not in result.reason


def test_bis_field_comparison_uses_bis_label_not_bool() -> None:
    result = match("CL 12 # 48 BIS - 31", "CL 12 # 48 - 31")
    bis_field = next(f for f in result.fields if f.field == "cross_bis")
    assert bis_field.a == "BIS"
    assert bis_field.b is None
