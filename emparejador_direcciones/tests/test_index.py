"""Tests for emparejador.index.AddressIndex (blocking must lose nothing)."""

from __future__ import annotations

import random
import time

import pytest

from emparejador import (
    AddressIndex,
    Candidate,
    InvalidThresholdError,
    InvalidToleranceError,
    LinkResult,
    match,
)

# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_empty_index() -> None:
    index = AddressIndex()
    result = index.find("KR 1 # 4 - 80")
    assert isinstance(result, LinkResult)
    assert result.candidates == []
    assert result.best is None
    assert result.ambiguous is False
    assert result.n_compared == 0
    assert result.parse_ok is True
    assert len(index) == 0
    assert index.stats() == {"records": 0, "blocks": 0, "rejected": 0, "max_block_size": 0}


def test_add_returns_true_and_counts() -> None:
    index = AddressIndex()
    assert index.add(1, "KR 1 # 4 - 80") is True
    assert len(index) == 1
    assert index.rejected == 0


@pytest.mark.parametrize("bad", [None, "", "   ", "junk", 123, "KR 1 # 4 - 80; KR 2 # 4 - 80", "X" * 10_000])
def test_unparseable_records_are_rejected_and_counted(bad: object) -> None:
    index = AddressIndex()
    assert index.add("id", bad) is False
    assert len(index) == 0
    assert index.rejected == 1
    assert index.find("KR 1 # 4 - 80").candidates == []


def test_find_exact_and_format_variant() -> None:
    index = AddressIndex()
    index.add("p1", "KR 1 # 4 - 80")
    index.add("p2", "KR 1 # 4 - 82")
    for query in ("KR 1 # 4 - 80", "kr 1 #4-080", "Cra. 1 No. 4 - 80"):
        result = index.find(query)
        assert [c.record_id for c in result.candidates] == ["p1"]
        assert result.best is not None and result.best.record_id == "p1"
        assert result.best.score == 1.0
        assert result.best.result.is_match is True
        assert result.ambiguous is False


def test_from_records_accepts_generators_and_counts_rejections() -> None:
    records = ((i, addr) for i, addr in enumerate(["KR 1 # 4 - 80", "junk", "KR 2 # 4 - 80"]))
    index = AddressIndex.from_records(records)
    assert len(index) == 2 and index.rejected == 1


def test_from_records_forwards_tolerance() -> None:
    index = AddressIndex.from_records([(1, "KR 1 # 4 - 80")], plate_tolerance=2)
    assert [c.record_id for c in index.find("KR 1 # 4 - 82").candidates] == [1]


def test_candidate_and_result_types() -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    result = index.find("KR 1 # 4 - 80")
    candidate = result.candidates[0]
    assert isinstance(candidate, Candidate)
    assert not hasattr(candidate, "__dict__")
    assert result.query.parse_ok is True


# ---------------------------------------------------------------------------
# Thresholds and scoring
# ---------------------------------------------------------------------------


def test_threshold_filters_candidates_but_counts_comparisons() -> None:
    index = AddressIndex()
    index.add("plain", "KR 1 # 4 - 80")
    index.add("letter", "KR 1 A # 4 - 80")
    strict = index.find("KR 1 # 4 - 80")
    loose = index.find("KR 1 # 4 - 80", threshold=0.85)
    assert [c.record_id for c in strict.candidates] == ["plain"]
    assert [c.record_id for c in loose.candidates] == ["plain", "letter"]
    assert strict.n_compared == 2 and loose.n_compared == 2
    assert loose.candidates[1].score == pytest.approx(0.85)


def test_threshold_boundary_is_inclusive() -> None:
    index = AddressIndex()
    index.add(1, "CL 25 NORTE # AV 6 - 30")
    assert index.find("CL 25 NORTE # 6 - 30", threshold=0.97).best is not None
    assert index.find("CL 25 NORTE # 6 - 30", threshold=0.9701).best is None


def test_threshold_extremes() -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    index.add(2, "KR 1 # 4 - 80 AP 5")
    assert [c.record_id for c in index.find("KR 1 # 4 - 80", threshold=1.0).candidates] == [1]
    assert {c.record_id for c in index.find("KR 1 # 4 - 80", threshold=0.0).candidates} == {1, 2}


@pytest.mark.parametrize("bad", [1.01, -0.1, float("nan"), "0.9", None])
def test_invalid_threshold_raises_even_on_empty_index(bad: object) -> None:
    with pytest.raises(InvalidThresholdError):
        AddressIndex().find("KR 1 # 4 - 80", threshold=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1, 1.5, "2", True, None])
def test_invalid_tolerance_raises(bad: object) -> None:
    with pytest.raises(InvalidToleranceError):
        AddressIndex(plate_tolerance=bad)  # type: ignore[arg-type]


def test_complement_one_sided_candidates() -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80 AP 1")
    index.add(2, "KR 1 # 4 - 80 AP 2")
    index.add(3, "KR 1 # 4 - 80 CA 7 8 C")
    result = index.find("KR 1 # 4 - 80")
    assert {c.record_id for c in result.candidates} == {1, 2, 3}
    assert all(c.score == pytest.approx(0.98) for c in result.candidates)
    assert result.ambiguous is True


def test_opaque_record_does_not_link_to_bare_query() -> None:
    index = AddressIndex()
    index.add(1, "K 49 E # 49 - 50 8 C")
    assert index.find("K 49 E # 49 - 50").candidates == []
    assert index.find("K 49 E # 49 - 50 8C").best.record_id == 1  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Ambiguity and duplicates
# ---------------------------------------------------------------------------


def test_duplicate_canonical_strings_with_different_ids_are_ambiguous() -> None:
    index = AddressIndex()
    index.add("b", "KR 1 # 4 - 80")
    index.add("a", "KR 01 # 04 - 080")
    result = index.find("KR 1 # 4 - 80")
    assert [c.record_id for c in result.candidates] == ["a", "b"]
    assert result.ambiguous is True
    assert result.best is not None and result.best.record_id == "a"


def test_single_exact_hit_is_not_ambiguous() -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    index.add(2, "KR 1 # 4 - 80 AP 5")
    result = index.find("KR 1 # 4 - 80")
    assert result.best is not None and result.best.record_id == 1
    assert result.ambiguous is False
    assert [c.record_id for c in result.candidates] == [1, 2]


def test_tie_below_top_score_is_not_ambiguous() -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    index.add(2, "KR 1 # 4 - 80 AP 1")
    index.add(3, "KR 1 # 4 - 80 AP 2")
    assert index.find("KR 1 # 4 - 80").ambiguous is False


def test_same_id_added_twice_is_a_single_record_not_a_tie() -> None:
    index = AddressIndex()
    index.add("x", "KR 1 # 4 - 80")
    index.add("x", "KR 1 # 4 - 80")
    result = index.find("KR 1 # 4 - 80")
    assert [c.record_id for c in result.candidates] == ["x"]
    assert result.ambiguous is False
    # S7: ``len`` counts distinct ids, not ``add`` calls.
    assert len(index) == 1


def test_same_id_with_two_addresses_keeps_best_score() -> None:
    index = AddressIndex()
    index.add("x", "KR 1 # 4 - 80 AP 5")
    index.add("x", "KR 1 # 4 - 80")
    result = index.find("KR 1 # 4 - 80")
    assert len(result.candidates) == 1 and result.candidates[0].score == 1.0


# ---------------------------------------------------------------------------
# Blocking: plate tolerance, None plates, None cross, None type
# ---------------------------------------------------------------------------


def _plates_index(tolerance: int) -> AddressIndex:
    index = AddressIndex(plate_tolerance=tolerance)
    for plate in (77, 78, 79, 80, 81, 82, 83, 84):
        index.add(f"p{plate}", f"KR 1 # 4 - {plate}")
    index.add("p80A", "KR 1 # 4 - 80 A")
    index.add("other_cross", "KR 1 # 10 - 80")
    return index


def test_tolerance_zero_finds_only_the_exact_plate() -> None:
    result = _plates_index(0).find("KR 1 # 4 - 80")
    assert [c.record_id for c in result.candidates] == ["p80"]


def test_tolerance_two_finds_neighbors_sorted_by_score_then_id() -> None:
    result = _plates_index(2).find("KR 1 # 4 - 80")
    ids = [c.record_id for c in result.candidates]
    assert ids == ["p80", "p78", "p79", "p81", "p82"]
    assert [c.score for c in result.candidates] == [1.0, 0.96, 0.96, 0.96, 0.96]


def test_tolerance_ignores_plate_letters_variants() -> None:
    result = _plates_index(2).find("KR 1 # 4 - 80 A")
    assert [c.record_id for c in result.candidates] == ["p80A"]


def test_tolerance_one_boundary_excludes_distance_two() -> None:
    ids = {c.record_id for c in _plates_index(1).find("KR 1 # 4 - 80").candidates}
    assert ids == {"p79", "p80", "p81"}


def test_huge_tolerance_does_not_explode() -> None:
    index = _plates_index(10**9)
    result = index.find("KR 1 # 4 - 80")
    assert {c.record_id for c in result.candidates} >= {"p80"}


def test_tolerance_neighbors_near_zero() -> None:
    index = AddressIndex(plate_tolerance=2)
    for plate in (0, 1, 2, 3):
        index.add(plate, f"KR 1 # 4 - {plate}")
    assert {c.record_id for c in index.find("KR 1 # 4 - 1").candidates} == {0, 1, 2, 3}
    assert {c.record_id for c in index.find("KR 1 # 4 - 0").candidates} == {0, 1, 2}


def test_none_plate_blocks() -> None:
    index = AddressIndex()
    index.add("bare", "KR 1 # 4")
    index.add("dash", "KR 1 # 4 -")
    index.add("lot", "KR 1 # 4 - LT 5")
    index.add("plated", "KR 1 # 4 - 80")
    result = index.find("KR 1 # 4")
    assert [c.record_id for c in result.candidates] == ["bare", "dash", "lot"]
    assert [c.score for c in result.candidates] == [1.0, 1.0, 0.98]
    assert result.ambiguous is True
    assert [c.record_id for c in index.find("KR 1 # 4 - 80").candidates] == ["plated"]


def test_none_plate_with_tolerance_still_finds_none_plates_only() -> None:
    index = AddressIndex(plate_tolerance=3)
    index.add("bare", "KR 1 # 4")
    index.add("plated", "KR 1 # 4 - 1")
    assert [c.record_id for c in index.find("KR 1 # 4").candidates] == ["bare"]
    assert [c.record_id for c in index.find("KR 1 # 4 - 2").candidates] == ["plated"]


def test_none_cross_blocks() -> None:
    index = AddressIndex()
    index.add("nocross", "KR 1 # - 80")
    index.add("cross", "KR 1 # 4 - 80")
    assert [c.record_id for c in index.find("KR 1 # - 80").candidates] == ["nocross"]
    assert [c.record_id for c in index.find("KR 1 # 4 - 80").candidates] == ["cross"]


def test_via_type_none_block_only_meets_typeless_peers() -> None:
    index = AddressIndex()
    index.add("typed", "KR 5 # 4 - 80")
    index.add("typeless", "5 # 4 - 80")
    assert [c.record_id for c in index.find("5 # 4 - 80").candidates] == ["typeless"]
    assert [c.record_id for c in index.find("KR 5 # 4 - 80").candidates] == ["typed"]


def test_legacy_prefix_and_alias_share_a_block() -> None:
    index = AddressIndex()
    index.add(1, "K 41 # 31 B -")
    assert index.find("CARRERA 41 No. 31 B").best.record_id == 1  # type: ignore[union-attr]
    index.add(2, "A 9 # 5 - 3")
    assert index.find("AV 9 # 5 - 3").best.record_id == 2  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Ordering, ids, huge blocks, stats
# ---------------------------------------------------------------------------


def test_deterministic_order_independent_of_insertion_order() -> None:
    records = [(i, f"KR 1 # 4 - 80 AP {i}") for i in range(30)] + [(99, "KR 1 # 4 - 80")]
    baseline = AddressIndex.from_records(records).find("KR 1 # 4 - 80")
    shuffled = list(records)
    random.Random(3).shuffle(shuffled)
    other = AddressIndex.from_records(shuffled).find("KR 1 # 4 - 80")
    assert [(c.record_id, c.score) for c in baseline.candidates] == [(c.record_id, c.score) for c in other.candidates]
    assert baseline.candidates[0].record_id == 99
    tail_ids = [c.record_id for c in baseline.candidates[1:]]
    assert tail_ids == sorted(tail_ids)


def test_int_and_str_ids_are_supported_together() -> None:
    index = AddressIndex()
    index.add(10, "KR 1 # 4 - 80")
    index.add("10", "KR 1 # 4 - 80")
    index.add(9, "KR 1 # 4 - 80")
    result = index.find("KR 1 # 4 - 80")
    assert len(result.candidates) == 3
    again = index.find("KR 1 # 4 - 80")
    assert [c.record_id for c in again.candidates] == [c.record_id for c in result.candidates]
    assert result.ambiguous is True


def test_unhashable_id_is_a_programming_error() -> None:
    with pytest.raises(TypeError):
        AddressIndex().add(["list"], "KR 1 # 4 - 80")


def test_huge_block_still_correct_and_fast_enough() -> None:
    index = AddressIndex()
    n = 3000
    for i in range(n):
        index.add(i, f"KR 1 # 4 - 80 AP {i}")
    start = time.perf_counter()
    result = index.find("KR 1 # 4 - 80 AP 1500")
    elapsed = time.perf_counter() - start
    assert result.n_compared == n
    assert result.best is not None and result.best.record_id == 1500
    assert [c.record_id for c in result.candidates] == [1500]
    assert elapsed < 10.0
    assert index.stats()["max_block_size"] == n


def test_stats() -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    index.add(2, "KR 1 # 4 - 80 AP 1")
    index.add(3, "KR 2 # 4 - 80")
    index.add(4, "junk")
    assert index.stats() == {"records": 3, "blocks": 2, "rejected": 1, "max_block_size": 2}


# ---------------------------------------------------------------------------
# Bad queries never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        None,
        "",
        "   ",
        "junk",
        123,
        b"KR 1 # 4 - 80",
        ["KR 1 # 4 - 80"],
        "KR 1 # 4 - 80; KR 2 # 4 - 80",
        "X" * 10_000,
        "KR 1 # 4 - " + "9" * 4301,
        "KR " + "9" * 4301 + " # 4 - 80",
        "\U0001F3E0 中文 # ★",
        "#" * 10_000,
    ],
)
def test_bad_query_returns_empty_result_with_reason(query: object) -> None:
    index = AddressIndex()
    index.add(1, "KR 1 # 4 - 80")
    result = index.find(query)  # type: ignore[arg-type]
    assert result.candidates == [] and result.best is None
    assert result.ambiguous is False and result.n_compared == 0
    if not result.parse_ok:
        assert result.reason
        assert result.query.notes


def test_unparseable_query_reason_mentions_parse_note() -> None:
    result = AddressIndex().find("KR 1 4 80")
    assert result.parse_ok is False
    assert "falta_separador_hash" in result.reason


def test_unicode_records_and_queries() -> None:
    index = AddressIndex()
    index.add("ñ", "Cra. 1 Nº 4 – 80")
    assert index.find("KR 1 # 4 - 80").best.record_id == "ñ"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Property: blocking loses nothing (index == brute force over match())
# ---------------------------------------------------------------------------


def _random_addr(rng: random.Random) -> str:
    parts = [rng.choice(["KR", "CL", "K", "A", "AV"]), str(rng.randint(1, 3))]
    if rng.random() < 0.4:
        parts.append(rng.choice("ABN"))
    if rng.random() < 0.15:
        parts.append(rng.choice(["NORTE", "OESTE"]))
    parts.append("#")
    if rng.random() < 0.9:
        if rng.random() < 0.2:
            parts.append(rng.choice(["KR", "CL"]))
        parts.append(str(rng.randint(1, 3)))
        if rng.random() < 0.3:
            parts.append(rng.choice("AB"))
    if rng.random() < 0.9:
        parts.append("-")
        if rng.random() < 0.85:
            parts.append(str(rng.randint(0, 9)))
            if rng.random() < 0.15:
                parts.append("A")
    r = rng.random()
    if r < 0.25:
        parts.extend(["AP", str(rng.randint(1, 3))])
    elif r < 0.35:
        parts.extend([str(rng.randint(1, 3)), rng.choice("CG")])
    elif r < 0.4:
        parts.extend(["LT", str(rng.randint(1, 2))])
    return " ".join(parts)


def _brute_force(records: list[tuple[int, str]], query: str, threshold: float, tolerance: int) -> list[tuple[int, float]]:
    best: dict[int, float] = {}
    for rid, addr in records:
        result = match(query, addr, threshold=threshold, plate_tolerance=tolerance)
        if result.score >= threshold:
            best[rid] = max(best.get(rid, 0.0), result.score)
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


@pytest.mark.parametrize("tolerance", [0, 1, 2])
@pytest.mark.parametrize("threshold", [0.9, 0.85])
def test_index_equals_brute_force_on_random_addresses(tolerance: int, threshold: float) -> None:
    rng = random.Random(1234 + tolerance)
    records: list[tuple[int, str]] = []
    while len(records) < 300:
        addr = _random_addr(rng)
        records.append((len(records), addr))
    index = AddressIndex.from_records(records, plate_tolerance=tolerance)
    parseable = [(rid, addr) for rid, addr in records if match(addr, addr).score == 1.0]
    assert len(parseable) > 150
    assert len(index) == len(parseable)

    queries = [addr for _, addr in parseable[:100]] + [_random_addr(rng) for _ in range(100)]
    for query in queries:
        result = index.find(query, threshold=threshold)
        expected = _brute_force(parseable, query, threshold, tolerance)
        got = [(c.record_id, c.score) for c in result.candidates]
        assert got == expected, query
        if expected:
            top = expected[0][1]
            ties = [rid for rid, score in expected if score == top]
            assert result.best is not None and result.best.record_id == expected[0][0]
            assert result.ambiguous == (len(ties) > 1)
        else:
            assert result.best is None and result.ambiguous is False


def test_every_record_finds_itself() -> None:
    rng = random.Random(99)
    records = [(i, _random_addr(rng)) for i in range(400)]
    index = AddressIndex.from_records(records)
    parseable_ids = {rid for rid, addr in records if match(addr, addr).score == 1.0}
    for rid, addr in records:
        if rid not in parseable_ids:
            continue
        result = index.find(addr)
        assert rid in {c.record_id for c in result.candidates}
        assert result.best is not None and result.best.score == 1.0
