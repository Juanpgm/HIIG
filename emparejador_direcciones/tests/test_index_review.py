"""Tests for the review findings on emparejador.index (W1, S1 plate cap, S2a, S7, S9)."""

from __future__ import annotations

import pytest

from emparejador import AddressIndex, LinkResult
from emparejador import index as index_module
from emparejador import matcher as matcher_module
from emparejador import parser as parser_module

# ---------------------------------------------------------------------------
# W1: near_ambiguous / alternatives
# ---------------------------------------------------------------------------


def test_exact_tie_is_ambiguous_and_near_ambiguous() -> None:
    index = AddressIndex()
    index.add("a", "KR 1 # 4 - 80")
    index.add("b", "KR 01 # 04 - 080")
    result = index.find("KR 1 # 4 - 80")
    assert result.ambiguous is True
    assert result.near_ambiguous is True
    assert result.alternatives == 1


def test_perfect_score_plus_known_complement_runner_up_is_near_ambiguous() -> None:
    index = AddressIndex()
    index.add("bare", "KR 1 # 4 - 80")
    index.add("unit", "KR 1 # 4 - 80 AP 5")
    result = index.find("KR 1 # 4 - 80")
    assert result.best is not None and result.best.record_id == "bare"
    assert result.candidates[1].score == pytest.approx(0.98)
    assert result.ambiguous is False  # unchanged semantics: no exact tie
    assert result.near_ambiguous is True
    assert result.alternatives == 1


def test_perfect_score_plus_cross_type_runner_up_is_near_ambiguous() -> None:
    index = AddressIndex()
    index.add("typed", "CL 25 NORTE # AV 6 - 30")
    index.add("untyped", "CL 25 NORTE # 6 - 30")
    result = index.find("CL 25 NORTE # 6 - 30")
    assert result.best is not None and result.best.record_id == "untyped"
    assert result.candidates[1].score == pytest.approx(0.97)
    assert result.ambiguous is False
    assert result.near_ambiguous is True
    assert result.alternatives == 1


def test_several_alternatives_are_counted() -> None:
    index = AddressIndex()
    index.add(0, "KR 1 # 4 - 80")
    for i in range(1, 4):
        index.add(i, f"KR 1 # 4 - 80 AP {i}")
    result = index.find("KR 1 # 4 - 80")
    assert result.alternatives == 3
    assert result.near_ambiguous is True
    assert result.ambiguous is False


def test_single_candidate_has_no_alternatives() -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    result = index.find("KR 1 # 4 - 80")
    assert result.near_ambiguous is False
    assert result.alternatives == 0


def test_no_candidate_has_no_alternatives() -> None:
    result = AddressIndex().find("KR 1 # 4 - 80")
    assert result.near_ambiguous is False
    assert result.alternatives == 0
    unparseable = AddressIndex().find("junk")
    assert unparseable.near_ambiguous is False and unparseable.alternatives == 0


def test_candidates_below_threshold_are_not_alternatives() -> None:
    index = AddressIndex()
    index.add("plain", "KR 1 # 4 - 80")
    index.add("letter", "KR 1 A # 4 - 80")  # scores 0.85 < 0.90
    result = index.find("KR 1 # 4 - 80")
    assert result.alternatives == 0
    assert result.near_ambiguous is False


def test_duplicate_ids_are_not_alternatives() -> None:
    index = AddressIndex()
    index.add("x", "KR 1 # 4 - 80")
    index.add("x", "KR 1 # 4 - 80 AP 5")  # same id, other address
    index.add("x", "KR 01 # 04 - 080")
    result = index.find("KR 1 # 4 - 80")
    assert [c.record_id for c in result.candidates] == ["x"]
    assert result.alternatives == 0
    assert result.near_ambiguous is False


def test_mixed_type_ids_count_as_distinct_alternatives() -> None:
    index = AddressIndex()
    index.add(10, "KR 1 # 4 - 80")
    index.add("10", "KR 1 # 4 - 80")
    index.add(b"10", "KR 1 # 4 - 80")
    result = index.find("KR 1 # 4 - 80")
    assert result.alternatives == 2
    assert result.near_ambiguous is True
    assert result.ambiguous is True


def test_link_result_defaults() -> None:
    result = LinkResult(query=AddressIndex().find("KR 1 # 4 - 80").query, threshold=0.9)
    assert result.near_ambiguous is False
    assert result.alternatives == 0
    assert result.truncated is False


# ---------------------------------------------------------------------------
# S1: one plate digit cap shared by parser, matcher and index
# ---------------------------------------------------------------------------


def test_plate_digit_cap_is_a_single_shared_constant() -> None:
    assert parser_module.MAX_PLATE_DIGITS == 6
    assert matcher_module.MAX_PLATE_DIGITS is parser_module.MAX_PLATE_DIGITS
    assert index_module.MAX_PLATE_DIGITS is parser_module.MAX_PLATE_DIGITS
    for module in (parser_module, matcher_module, index_module):
        assert not hasattr(module, "_MAX_PLATE_SIGNIFICANT_DIGITS")


def test_plate_at_the_cap_is_indexed_and_beyond_it_is_rejected() -> None:
    index = AddressIndex()
    assert index.add(1, "KR 1 # 4 - 999999") is True
    assert index.add(2, "KR 1 # 4 - 0000999999") is True  # leading zeros are not significant
    assert index.add(3, "KR 1 # 4 - 1000000") is False
    assert [c.record_id for c in index.find("KR 1 # 4 - 999999").candidates] == [1, 2]


def test_plate_key_is_always_an_int_or_none() -> None:
    assert index_module._plate_key(None) is None
    assert index_module._plate_key("080") == 80
    assert index_module._plate_key("80A") == 80
    assert index_module._plate_key("000") == 0
    assert isinstance(index_module._plate_key("999999"), int)


# ---------------------------------------------------------------------------
# S2a: blocking strength (n_compared stays at the records sharing the key)
# ---------------------------------------------------------------------------


def _many_faces_index() -> AddressIndex:
    index = AddressIndex()
    n = 0
    for via in range(1, 13):
        for cross in range(1, 13):
            for plate in range(1, 4):
                index.add(n, f"KR {via} # {cross} - {plate}")
                n += 1
    return index


def test_blocking_only_compares_records_sharing_via_cross_and_plate() -> None:
    index = _many_faces_index()
    index.add("twin", "KR 5 # 7 - 03")  # a second record on the exact same key
    result = index.find("KR 5 # 7 - 3")
    assert len(index) == 12 * 12 * 3 + 1
    assert result.n_compared == 2
    assert len(result.candidates) == 2


def test_blocking_does_not_merge_faces_that_differ_only_in_cross_number() -> None:
    index = AddressIndex()
    for cross in range(1, 40):
        index.add(cross, f"KR 3 # {cross} - 10")
    result = index.find("KR 3 # 20 - 10")
    assert result.n_compared == 1


def test_blocking_does_not_merge_faces_that_differ_only_in_via_number() -> None:
    index = AddressIndex()
    for via in range(1, 40):
        index.add(via, f"KR {via} # 3 - 10")
    assert index.find("KR 20 # 3 - 10").n_compared == 1


def test_missing_plate_is_not_filed_with_plate_zero() -> None:
    index = AddressIndex()
    index.add("none", "KR 1 # 4")
    index.add("zero_a", "KR 1 # 4 - 0")
    index.add("zero_b", "KR 1 # 4 - 00")
    index.add("zero_c", "KR 1 # 4 - 0 A")
    without_plate = index.find("KR 1 # 4")
    assert without_plate.n_compared == 1
    assert [c.record_id for c in without_plate.candidates] == ["none"]
    with_plate_zero = index.find("KR 1 # 4 - 0")
    assert with_plate_zero.n_compared == 3  # the plate letter is not part of the key
    assert [c.record_id for c in with_plate_zero.candidates] == ["zero_a", "zero_b"]


def test_missing_plate_is_not_filed_with_plate_zero_under_tolerance() -> None:
    index = AddressIndex(plate_tolerance=2)
    index.add("none", "KR 1 # 4")
    index.add("zero", "KR 1 # 4 - 0")
    index.add("one", "KR 1 # 4 - 1")
    assert index.find("KR 1 # 4").n_compared == 1
    assert index.find("KR 1 # 4 - 0").n_compared == 2


# ---------------------------------------------------------------------------
# S7: len(index) / stats()["records"] count DISTINCT ids
# ---------------------------------------------------------------------------


def test_len_counts_distinct_ids_not_add_calls() -> None:
    index = AddressIndex()
    index.add("x", "KR 1 # 4 - 80")
    index.add("x", "KR 1 # 4 - 80")
    index.add("x", "KR 2 # 4 - 80")
    index.add("y", "KR 3 # 4 - 80")
    assert len(index) == 2
    assert index.stats()["records"] == 2


def test_ids_are_compared_as_given_and_true_one_collide() -> None:
    """Documented rule: ids follow Python equality/hash, so 1, 1.0 and True
    are ONE id, while 1 and "1" are two."""
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    index.add(1.0, "KR 1 # 4 - 80")
    index.add(True, "KR 1 # 4 - 80")
    assert len(index) == 1
    assert index.stats()["records"] == 1
    index.add("1", "KR 1 # 4 - 80")
    assert len(index) == 2
    assert len(index.find("KR 1 # 4 - 80").candidates) == 2


def test_rejected_records_do_not_count_as_ids() -> None:
    index = AddressIndex()
    index.add("x", "junk")
    assert len(index) == 0
    index.add("x", "KR 1 # 4 - 80")
    assert len(index) == 1
    assert index.rejected == 1


def test_stats_blocks_and_block_size_still_count_entries() -> None:
    index = AddressIndex()
    index.add("x", "KR 1 # 4 - 80")
    index.add("x", "KR 1 # 4 - 80")
    stats = index.stats()
    assert stats == {"records": 1, "blocks": 1, "rejected": 0, "max_block_size": 2}


# ---------------------------------------------------------------------------
# S9: max_block_scan guard
# ---------------------------------------------------------------------------


def _big_block(n: int) -> AddressIndex:
    index = AddressIndex()
    for i in range(n):
        index.add(i, f"KR 1 # 4 - 80 AP {i}")
    return index


def test_default_has_no_cap_and_is_not_truncated() -> None:
    index = _big_block(5000)
    result = index.find("KR 1 # 4 - 80")
    assert result.n_compared == 5000
    assert result.truncated is False
    assert len(result.candidates) == 5000


def test_cap_stops_scanning_a_5000_record_block() -> None:
    index = _big_block(5000)
    result = index.find("KR 1 # 4 - 80", max_block_scan=100)
    assert result.n_compared == 100
    assert result.truncated is True
    assert len(result.candidates) == 100


def test_cap_equal_to_block_size_is_not_truncated() -> None:
    index = _big_block(50)
    result = index.find("KR 1 # 4 - 80", max_block_scan=50)
    assert result.n_compared == 50
    assert result.truncated is False


def test_cap_one_below_block_size_is_truncated() -> None:
    index = _big_block(50)
    result = index.find("KR 1 # 4 - 80", max_block_scan=49)
    assert result.n_compared == 49
    assert result.truncated is True


def test_cap_larger_than_block_is_not_truncated() -> None:
    index = _big_block(10)
    result = index.find("KR 1 # 4 - 80", max_block_scan=10_000)
    assert result.n_compared == 10 and result.truncated is False


def test_cap_applies_per_block_under_tolerance() -> None:
    index = AddressIndex(plate_tolerance=1)
    for i in range(20):
        index.add(f"a{i}", f"KR 1 # 4 - 80 AP {i}")
        index.add(f"b{i}", f"KR 1 # 4 - 81 AP {i}")
    result = index.find("KR 1 # 4 - 80", max_block_scan=5)
    assert result.n_compared == 10  # 5 from each of the two plate blocks
    assert result.truncated is True


def test_cap_on_small_blocks_never_truncates() -> None:
    index = _big_block(3)
    assert index.find("KR 1 # 4 - 80", max_block_scan=3).truncated is False


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True])
def test_invalid_cap_raises(bad: object) -> None:
    with pytest.raises(ValueError):
        AddressIndex().find("KR 1 # 4 - 80", max_block_scan=bad)  # type: ignore[arg-type]


def test_unparseable_query_is_not_truncated() -> None:
    result = _big_block(5).find("junk", max_block_scan=1)
    assert result.truncated is False
