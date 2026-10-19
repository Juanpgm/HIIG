"""I1: known divergences of ``level="structure"`` from the master integration
index, pinned so that any change is deliberate (see the README section
"Known divergences of level=structure")."""

from __future__ import annotations

import pytest

from emparejador import match

BASE = "CL 987 # 654 - 21"


def _structure(a: str, b: str):
    return match(f"{BASE}{a}", f"{BASE}{b}", level="structure")


@pytest.mark.parametrize(
    "a,b",
    [
        (" AP 1203T4", " AP 1204T5"),     # tower digit + letter embedded in the unit value
        (" AP K101", " AP J101"),         # tower letter embedded in the unit value
        (" TO 1 AP 1203T4", " TO 1 AP 1204T5"),
    ],
)
def test_a_tower_embedded_in_a_unit_value_is_not_compared_at_structure_level(a, b):
    result = _structure(a, b)
    assert result.score == 1.0 and result.is_match is True
    # ... while the unit level does tell them apart
    assert match(f"{BASE}{a}", f"{BASE}{b}").score == 0.0


def test_a_tower_written_as_its_own_chunk_is_compared():
    assert _structure(" TO 1 AP 101", " TO 2 AP 101").score == 0.0
    assert _structure(" TO 1 AP 1203T4", " TO 2 AP 1203T4").score == 0.0


@pytest.mark.parametrize("a,b", [(" CS 5", " CASA 5"), (" CS 5", " CA 5"), (" CS 5", " CS 6")])
def test_cs_is_opaque_and_never_equals_casa(a, b):
    assert _structure(a, b).score == 0.0
    assert _structure(a, b).is_match is False


def test_cs_against_itself_still_matches_exactly():
    assert _structure(" CS 5", " CS 5").score == 1.0


@pytest.mark.parametrize("name", ["ED SOL", "CONJ LOS PINOS"])
def test_ed_and_conj_names_are_one_sided_soft_not_identifying(name):
    one_sided = _structure(f" {name}", "")
    assert one_sided.score == pytest.approx(0.98) and one_sided.is_match is True
    assert match(f"{BASE} {name}", BASE).score == pytest.approx(0.98)          # same price at unit level


def test_two_different_ed_names_are_a_veto():
    assert _structure(" ED SOL", " ED LUNA").score == 0.0


def test_distinguishing_kind_on_one_side_is_still_a_veto_for_the_other_kinds():
    for kind in ("TO A", "BLQ 2", "SEC 5", "AGR 3", "CA 7"):
        assert _structure(f" {kind}", "").score == 0.0, kind
