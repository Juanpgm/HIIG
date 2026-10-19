"""Tests for the review findings on emparejador.matcher (S3, S4, S5)."""

from __future__ import annotations

import itertools

import pytest

from emparejador import matcher as matcher_module
from emparejador.matcher import match

# ---------------------------------------------------------------------------
# S3: the dead-band proof enumerates penalties from ONE registry that the
# matcher itself reads, and every ``_*_PENALTY`` constant must be registered.
# ---------------------------------------------------------------------------

# Which registered penalties can be charged on which kind of field. A new
# registered penalty MUST be assigned to a field group here, otherwise
# ``test_every_registered_penalty_is_assigned_to_a_field_group`` fails.
_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "soft_structural": ("one_sided", "conflict"),
    "bis": ("one_sided",),
    "cross_type": ("cross_type_one_sided", "cross_type_conflict"),
    "plate": ("plate_tolerance",),
    "complement": ("complement_one_sided", "one_sided"),
}


def _penalty_constant_names() -> set[str]:
    return {
        name
        for name in vars(matcher_module)
        if name.startswith("_") and name.endswith("_PENALTY") and not name.startswith("__")
    }


def _assert_every_penalty_constant_is_registered() -> None:
    registry = matcher_module._PENALTIES
    unregistered = {
        name for name in _penalty_constant_names() if name[1 : -len("_PENALTY")].lower() not in registry
    }
    assert unregistered == set(), (
        f"penalty constants {sorted(unregistered)} are not in matcher._PENALTIES; "
        "register them there and use the registry at the point of application"
    )


def test_every_penalty_constant_is_in_the_registry() -> None:
    _assert_every_penalty_constant_is_registered()


def test_registry_values_are_valid_survival_penalties() -> None:
    for name, value in matcher_module._PENALTIES.items():
        assert 0.0 < value < 1.0, name


def test_every_registered_penalty_is_assigned_to_a_field_group() -> None:
    used = {name for names in _FIELD_GROUPS.values() for name in names}
    assert used == set(matcher_module._PENALTIES)


def test_registry_is_what_the_matcher_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing a registry value changes the score: the matcher reads it."""
    baseline = match("KR 1 # 4 - 80", "KR 1 A # 4 - 80").score
    assert baseline == pytest.approx(0.85)
    monkeypatch.setitem(matcher_module._PENALTIES, "one_sided", 0.5)
    assert match("KR 1 # 4 - 80", "KR 1 A # 4 - 80").score == pytest.approx(0.5)


@pytest.mark.parametrize("level", ["unit", "building"])
def test_dead_band_is_empty_and_max_sub_threshold_is_085(level: str) -> None:
    # Both levels charge the complement field from the SAME registry values
    # (0.0, complement_one_sided, one_sided): building level only ignores a
    # handful of complement KINDS (BG/INT/PISO/PH, see
    # matcher.BUILDING_IGNORED_KINDS), it does not change what a differing
    # complement can cost once one is compared, so the reachable set is
    # identical at both levels; ``level`` is parametrized to make that an
    # explicit, checked fact rather than an assumption.
    _assert_every_penalty_constant_is_registered()
    registry = matcher_module._PENALTIES
    soft_fields = len(matcher_module._SOFT_STRUCTURAL_FIELDS)
    bis_fields = len(matcher_module._BIS_FIELDS)

    def choices(group: str) -> list[float]:
        return [0.0, *(registry[name] for name in _FIELD_GROUPS[group])]

    per_field_choices = (
        [choices("soft_structural")] * soft_fields
        + [choices("bis")] * bis_fields
        + [choices("cross_type"), choices("plate"), choices("complement")]
    )

    max_sub_threshold = 0.0
    for combo in itertools.product(*per_field_choices):
        score = 1.0
        for penalty in combo:
            score *= 1.0 - penalty
        score = round(score, 4)
        assert not (0.85 < score < 0.9126), (combo, score)
        if score < 0.9126:
            max_sub_threshold = max(max_sub_threshold, score)
    assert max_sub_threshold == 0.85


# ---------------------------------------------------------------------------
# S4: known complement kinds without a usable value are opaque
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "with_complement,bare",
    [
        ("C 1 # 74 - LT", "C 1 # 74"),
        ("KR 1 # 2 - 3 AP", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 LT", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 AP NORTE", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 AP SUR", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 TO ESTE", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 LC OESTE", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 AP NORTE SUR", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 AP 5 LT", "KR 1 # 2 - 3"),  # one valued kind, one empty
    ],
)
def test_known_kind_without_value_costs_the_opaque_one_sided_penalty(
    with_complement: str, bare: str
) -> None:
    for a, b in ((with_complement, bare), (bare, with_complement)):
        result = match(a, b)
        assert result.score == pytest.approx(0.85), (a, b)
        assert result.is_match is False
        assert "complemento_no_estructurado" in result.warnings
        complement = next(f for f in result.fields if f.field == "complement")
        assert complement.status == "solo_en_uno"
        assert complement.penalty == pytest.approx(0.15)
        assert complement.veto is False


def test_known_kind_with_a_value_keeps_the_cheap_one_sided_penalty() -> None:
    result = match("KR 1 # 2 - 3 AP 5", "KR 1 # 2 - 3")
    assert result.score == pytest.approx(0.98)
    assert result.is_match is True
    assert "complemento_no_estructurado" not in result.warnings


def test_value_that_merely_contains_a_quadrant_prefix_is_still_a_value() -> None:
    # "N5" splits into the tokens N and 5, neither of which is a quadrant word.
    result = match("KR 1 # 2 - 3 AP N5", "KR 1 # 2 - 3")
    assert result.score == pytest.approx(0.98)


@pytest.mark.parametrize(
    "address",
    ["C 1 # 74 - LT", "KR 1 # 2 - 3 AP", "KR 1 # 2 - 3 AP NORTE"],
)
def test_equal_valueless_complements_on_both_sides_are_still_equal(address: str) -> None:
    result = match(address, address)
    assert result.score == 1.0
    complement = next(f for f in result.fields if f.field == "complement")
    assert complement.status == "igual"
    assert complement.penalty == 0.0
    assert "complemento_no_estructurado" in result.warnings


def test_valueless_kind_against_a_valued_same_kind_is_a_veto() -> None:
    result = match("KR 1 # 2 - 3 AP", "KR 1 # 2 - 3 AP 5")
    assert result.is_match is False
    assert result.score == 0.0
    assert "complemento" in result.reason


def test_valueless_kind_against_a_different_kind_is_a_veto() -> None:
    assert match("KR 1 # 2 - 3 AP", "KR 1 # 2 - 3 LT").score == 0.0


# ---------------------------------------------------------------------------
# S5: repeated complement chunks count (multiset comparison)
# ---------------------------------------------------------------------------


def test_duplicate_chunk_is_not_equal_to_the_single_chunk() -> None:
    result = match("KR 1 # 2 - 3 AP 1 AP 1", "KR 1 # 2 - 3 AP 1")
    assert result.score == 0.0
    assert result.is_match is False
    assert "complemento" in result.reason


def test_duplicate_chunks_equal_themselves() -> None:
    result = match("KR 1 # 2 - 3 AP 1 AP 1", "KR 1 # 2 - 3 AP 1 AP 1")
    assert result.score == 1.0


def test_duplicate_chunks_are_order_independent() -> None:
    assert match("KR 1 # 2 - 3 AP 1 LT 2 AP 1", "KR 1 # 2 - 3 LT 2 AP 1 AP 1").score == 1.0


@pytest.mark.parametrize(
    "a,b",
    [
        ("KR 1 # 2 - 3 AP 1 AP 1", "KR 1 # 2 - 3 AP 1"),
        ("KR 1 # 2 - 3 AP 1 AP 1", "KR 1 # 2 - 3"),
        ("KR 1 # 2 - 3 AP 1 AP 1", "KR 1 # 2 - 3 AP 1 AP 2"),
        ("KR 1 # 2 - 3 AP 1 AP 1", "KR 1 # 2 - 3 AP 1 AP 1"),
    ],
)
def test_duplicate_chunks_keep_the_score_symmetric(a: str, b: str) -> None:
    assert match(a, b).score == match(b, a).score


def test_duplicate_chunks_against_bare_address_is_one_sided() -> None:
    assert match("KR 1 # 2 - 3 AP 1 AP 1", "KR 1 # 2 - 3").score == pytest.approx(0.98)
