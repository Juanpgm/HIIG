"""Phase 0.2c: letters glued to via / cross numbers are separated (``32B`` -> ``32 B``).

All addresses are invented. The regression block compares hashes of values computed from the real
splits BEFORE the helper existed: the split keys and the parser output must not move.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import parser  # noqa: E402
from cali_address.labels import address_key  # noqa: E402
from cali_address.parser import _pre_normalize, normalize_spacing, parse_address  # noqa: E402

SPLITS = os.path.join(ROOT, "artifacts", "splits")


@pytest.fixture(autouse=True)
def _letters_separated():
    """The separation is opt-in per artifact (default off); this module tests it switched on."""
    with parser.pre_normalize_separation(True):
        yield


# ------------------------------------------------------------------ the helper

@pytest.mark.parametrize("raw,expected", [
    ("KR 32B # 46 - 131", "KR 32 B # 46 - 131"),                  # letter on the via
    ("KR 32 # 46B - 131", "KR 32 # 46 B - 131"),                  # letter on the cross
    ("CL 8A # 32B - 10", "CL 8 A # 32 B - 10"),                   # both
    ("kr 32b # 46 - 131", "kr 32 b # 46 - 131"),                  # case is preserved
    ("KR32B # 4 - 5", "KR32 B # 4 - 5"),                          # type glued to the number
    ("KR 32B#46-131", "KR 32 B#46-131"),                          # glued around the hash
    ("KR 32B", "KR 32 B"),                                        # no hash at all
    ("KR 32B 46 - 131", "KR 32 B 46 - 131"),                      # hash-less cross
    ("KR 32 B # 46 - 131", "KR 32 B # 46 - 131"),                 # already separated
    ("CL 9 # 4 - 1", "CL 9 # 4 - 1"),                             # no letters
])
def test_letter_glued_to_via_or_cross_number_is_separated(raw, expected):
    assert normalize_spacing(raw) == expected


@pytest.mark.parametrize("raw", [
    "CL 12BIS # 4 - 5",            # BIS is a word, not a street letter
    "KR 32BIS A # 4 - 5",
    "KR 10OESTE # 5 - 3",          # glued quadrant keeps its meaning
    "KR 3BNORTE # 4 - 5",          # street letter + spelled quadrant (CHANGELOG round 7)
    "CL 25NORTE # AV 6 - 30",
    "AV 15 OESTE # 9 OESTE - 137",
    "KR 5 NO654",                  # NO654 is the number sign, not a quadrant
    "KR 5 NO654-321",
    "KR 3RA # 4 - 5",              # ordinal noise
    "CRA 987 5B1 321",             # letter-digit designators keep their historical reading
    "KR 987G6 654S1 321",
])
def test_bis_quadrants_number_sign_and_designators_are_untouched(raw):
    assert normalize_spacing(raw) == raw


@pytest.mark.parametrize("raw", [
    "CL 5 # 12 - 3 AP 302",
    "CL 5 # 12 - 3 TO 2 AP 302B",
    "CL 5 # 12 - 3 ED 47B",
    "CL 5 # 12 - 3 AP302B",
    "CL 5 # 12 APT 201B",          # complement without a plate
    "CL 5 # 12 CASA 4B",
    "CL 5 # 12 - 3 LOTE 5B BLQ 3C",
    "CL 5 # 12 - 3 CONJ RESIDENCIAL 14C",
    "CL 5 # 12 - 3 T-F",
])
def test_complements_are_untouched(raw):
    assert normalize_spacing(raw) == raw


@pytest.mark.parametrize("raw", ["CL 5 # 12 - 3B", "CL 5 # 12 - 34A", "KR 32 B # 46 - 131C"])
def test_plate_letters_are_untouched(raw):
    assert normalize_spacing(raw) == raw


def test_only_the_main_address_is_split_when_a_complement_follows():
    assert normalize_spacing("KR 32B # 46 - 131 AP 302B") == "KR 32 B # 46 - 131 AP 302B"
    assert normalize_spacing("KR 32B # 46B APT 201B") == "KR 32 B # 46 B APT 201B"


@pytest.mark.parametrize("raw", ["KR 32B 46 - 131", "KR 987 D2 N 654C 321", "KR 32B", "CRA 987 5B1 321"])
def test_require_hash_leaves_hashless_addresses_alone(raw):
    # Without a number sign the reading of a letter next to a number is ambiguous (round 7 keeps the
    # historical parse), so the wired variant only acts on the explicit-hash form.
    assert normalize_spacing(raw, require_hash=True) == raw


def test_require_hash_still_separates_the_explicit_hash_form():
    assert normalize_spacing("KR 32B # 46B - 131", require_hash=True) == "KR 32 B # 46 B - 131"


@pytest.mark.parametrize("raw", [
    "KR 32B # 46 - 131", "CL 8A # 32B - 10", "KR32B#46-131", "CL 5 # 12 APT 201B", "KR 10OESTE # 5 - 3",
    "kr 3b # 4c - 5 ap 302b", "", "   ", "AVENIDA SEXTA", "KR 32Ñ # 4 - 5",
])
def test_normalize_spacing_is_idempotent(raw):
    once = normalize_spacing(raw)
    assert normalize_spacing(once) == once


@pytest.mark.parametrize("raw,expected", [(None, ""), ("", ""), ("   ", "   "), (123, "123")])
def test_empty_null_and_non_string_inputs_never_raise(raw, expected):
    assert normalize_spacing(raw) == expected


def test_unicode_letter_is_separated_like_any_other_letter():
    assert normalize_spacing("KR 32Ñ # 4 - 5") == "KR 32 Ñ # 4 - 5"


def test_text_without_digit_letter_boundaries_is_returned_unchanged():
    for raw in ("AVENIDA SEXTA", "CL B # 4 - 5", "KR 32 # 46 - 131", "#"):
        assert normalize_spacing(raw) == raw


# ------------------------------------------------------------------ wired into _pre_normalize

PAIRS = [
    ("KR 32B # 46 - 131", "KR 32 B # 46 - 131"),
    ("CL 8A # 32B - 10", "CL 8 A # 32 B - 10"),
    ("Carrera 32b No 46-131", "Carrera 32 b No 46-131"),
    ("KR 926H 1 # 973 - 910", "KR 926 H 1 # 973 - 910"),
]


@pytest.mark.parametrize("glued,spaced", PAIRS)
def test_pre_normalize_gives_the_same_view_for_glued_and_spaced(glued, spaced):
    assert _pre_normalize(glued) == _pre_normalize(spaced)


def test_pre_normalize_separates_the_street_letter():
    assert _pre_normalize("KR 32B # 46 - 131") == "KR 32 B # 46 - 131"


@pytest.mark.parametrize("glued,spaced", PAIRS)
def test_address_key_is_identical_for_glued_and_spaced_forms(glued, spaced):
    assert address_key(glued) == address_key(spaced) != ""


@pytest.mark.parametrize("glued,spaced", PAIRS)
def test_parse_result_is_identical_for_glued_and_spaced_forms(glued, spaced):
    a, b = parse_address(glued).to_dict(), parse_address(spaced).to_dict()
    a.pop("raw"), b.pop("raw")
    assert a == b


def test_pre_normalize_spacing_can_be_switched_off_to_reproduce_the_old_view(monkeypatch):
    # Reproducibility of the production baseline: the encoder view of the current model was built
    # without this step, so it must be possible to evaluate it exactly as before.
    monkeypatch.setattr(parser, "PRE_NORMALIZE_SEPARATES_LETTERS", False)
    assert _pre_normalize("KR 32B # 46 - 131") == "KR 32B # 46 - 131"
    monkeypatch.setattr(parser, "PRE_NORMALIZE_SEPARATES_LETTERS", True)
    assert _pre_normalize("KR 32B # 46 - 131") == "KR 32 B # 46 - 131"


def test_pre_normalize_keeps_complement_values_verbatim():
    assert _pre_normalize("CL 5 # 12 - 3 AP 302B") == "CL 5 # 12 - 3 AP 302B"


def test_pre_normalize_keeps_no654_as_number_sign():
    assert _pre_normalize("KR 5 NO654") == "KR 5 # 654"


def test_normalize_spacing_is_exported_by_the_module():
    assert "normalize_spacing" in parser.__all__


# ------------------------------------------------------------------ regression on the real splits
# Hashes computed with the code as it was BEFORE normalize_spacing existed (privacy: only hashes).

BASELINE_KEY_SHA = {
    "dev": "cafac2f49f75fd486900c47f46fb7892ca1c6f7456de1233e711b5d08f5cf91f",
    "frozen_test": "a656fbae1aae21358aa91064681a435d06cfe6cfe5a1a66320a3fe6ac82fbf9a",
    "train_pairs": "2d60ff9e7a5af685d030c031787d5cfeff3b65784dadc4b9b7f51732906b2614",
}
BASELINE_PARSE_SHA = {
    "dev": "cd769bce0d5ab46f75f8d28e0249ebefad93e7ef1d47b464924ef9cd5b74c591",
    "frozen_test": "969bf025ceca04162e89cfa008c9939de177cdf990a307cd0b24df82f436986f",
    "train_pairs": "2f4ee08b69c743f175d349083135ac9cd89e1effea5be4c043ce4c00cb2158fd",
}


def _split(name):
    path = os.path.join(SPLITS, f"{name}.parquet")
    if not os.path.isfile(path):
        pytest.skip(f"{path} is not available")
    return pd.read_parquet(path, columns=["raw_address"])["raw_address"]


@pytest.mark.parametrize("name", sorted(BASELINE_KEY_SHA))
def test_address_key_of_every_split_row_is_unchanged(name):
    keys = [address_key(a) for a in _split(name)]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == BASELINE_KEY_SHA[name]


@pytest.mark.parametrize("name", sorted(BASELINE_PARSE_SHA))
def test_parser_output_of_every_split_row_is_unchanged(name):
    rows = []
    for a in _split(name):
        p = parse_address(str(a)).to_dict()
        p.pop("raw", None)
        rows.append(json.dumps(p, sort_keys=True, default=str))
    assert hashlib.sha256("\n".join(rows).encode()).hexdigest() == BASELINE_PARSE_SHA[name]
