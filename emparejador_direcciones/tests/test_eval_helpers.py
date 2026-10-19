"""Unit tests for the pure helpers of ``tools/eval_catastro.py``.

These tests never touch the cadastral parquet nor pyarrow: they only cover
the perturbation generators and categorizers that live at module level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emparejador import AddressIndex, match, parse_canonical  # noqa: E402
from tools import eval_catastro as ev  # noqa: E402

BASE = "KR 26 H 3 # 73 - 10"
WITH_COMPLEMENT = "CL 12 A # 52 - 64 AP 501 TO 3"
BARE = "CL 12 # 48 BIS - 31"
NO_PLATE = "CL 5 # 10"


# --------------------------------------------------------------------------
# shape_signature
# --------------------------------------------------------------------------


class TestShapeSignature:
    def test_digits_and_letters_are_collapsed(self) -> None:
        assert ev.shape_signature("KR 26 H 3 # 73 - 10") == "KR 9 A 9 # 9 - 9"

    def test_known_tokens_are_kept(self) -> None:
        sig = ev.shape_signature("CL 12 BIS NORTE # AP 5")
        assert "BIS" in sig and "NORTE" in sig and "AP" in sig

    def test_unknown_word_becomes_A(self) -> None:
        assert ev.shape_signature("ZZZZ 5") == "A 9"

    def test_glued_letters_and_digits(self) -> None:
        assert ev.shape_signature("KR98F") == "KR9A"

    @pytest.mark.parametrize("value", [None, "", "   ", 42])
    def test_empty_or_non_string(self, value: object) -> None:
        assert ev.shape_signature(value) == ""

    def test_unicode_is_letter_run(self) -> None:
        assert ev.shape_signature("ÑANDÚ 5") == "A 9"

    def test_very_long_string_is_fast_and_collapses(self) -> None:
        assert ev.shape_signature("9" * 100_000) == "9"

    def test_same_shape_same_signature(self) -> None:
        assert ev.shape_signature("KR 1 # 2 - 3") == ev.shape_signature("KR 88 # 77 - 66")


# --------------------------------------------------------------------------
# positive variants
# --------------------------------------------------------------------------


class TestPositiveVariants:
    def test_all_variants_parse_and_differ_from_input(self) -> None:
        for addr in (BASE, WITH_COMPLEMENT, BARE, NO_PLATE):
            for kind, text in ev.positive_variants(addr).items():
                assert text != addr, kind
                assert parse_canonical(text).parse_ok, (kind, text)

    def test_expected_kinds_for_full_address(self) -> None:
        kinds = set(ev.positive_variants(WITH_COMPLEMENT))
        for expected in (
            "lowercase",
            "extra_whitespace",
            "glued",
            "hash_tight",
            "hash_no_dot",
            "hash_ordinal",
            "plate_zero_pad",
            "alias_long",
            "alias_dotted",
            "complement_alias",
            "complement_reorder",
        ):
            assert expected in kinds

    def test_glued_style(self) -> None:
        assert ev.positive_variants("KR 96 F # 96 - 66")["glued"] == "KR 96F # 96 - 66"

    def test_hash_variants(self) -> None:
        variants = ev.positive_variants("KR 1 # 4 - 80")
        assert variants["hash_tight"] == "KR 1 #4-80"
        assert variants["hash_no_dot"] == "KR 1 No. 4 - 80"

    def test_plate_zero_padding(self) -> None:
        assert ev.positive_variants("KR 1 # 4 - 80")["plate_zero_pad"].endswith("- 080")
        assert ev.positive_variants("KR 1 # 4 - 2")["plate_zero_pad"].endswith("- 02")

    def test_long_alias(self) -> None:
        variants = ev.positive_variants("KR 1 # CL 9 - 80")
        assert variants["alias_long"].startswith("CARRERA 1")
        assert "CALLE 9" in variants["alias_long"]
        assert variants["alias_dotted"].startswith("KR. 1") or variants["alias_dotted"].startswith("CRA. 1")

    def test_complement_alias_and_reorder(self) -> None:
        variants = ev.positive_variants(WITH_COMPLEMENT)
        assert "APTO" in variants["complement_alias"]
        assert variants["complement_reorder"].endswith("AP 501")

    def test_no_complement_no_complement_variants(self) -> None:
        kinds = set(ev.positive_variants(BASE))
        assert "complement_alias" not in kinds
        assert "complement_reorder" not in kinds

    def test_no_plate_no_plate_variants(self) -> None:
        assert "plate_zero_pad" not in ev.positive_variants(NO_PLATE)

    @pytest.mark.parametrize("value", [None, "", "   ", "GARBAGE", 7, "KR 1 # 4 - 80 CL 5 # 3"])
    def test_unparseable_input_yields_nothing(self, value: object) -> None:
        assert ev.positive_variants(value) == {}

    def test_very_short_string(self) -> None:
        assert ev.positive_variants("K") == {}

    def test_very_long_string_does_not_raise(self) -> None:
        assert ev.positive_variants("KR 1 # 4 - " + "8" * 5000) == {}

    def test_deterministic(self) -> None:
        assert ev.positive_variants(WITH_COMPLEMENT) == ev.positive_variants(WITH_COMPLEMENT)

    def test_every_variant_matches_original_with_current_matcher_on_core_kinds(self) -> None:
        # The matcher is expected to accept these formatting variants.
        for kind in ("lowercase", "extra_whitespace", "glued", "hash_tight", "plate_zero_pad"):
            text = ev.positive_variants(BASE)[kind]
            assert match(BASE, text).score == 1.0, kind


# --------------------------------------------------------------------------
# negative mutations
# --------------------------------------------------------------------------


class TestNegativeMutations:
    def test_all_mutations_parse_and_differ(self) -> None:
        base = parse_canonical(WITH_COMPLEMENT)
        for kind, text in ev.negative_mutations(WITH_COMPLEMENT).items():
            parsed = parse_canonical(text)
            assert parsed.parse_ok, (kind, text)
            assert parsed != base
            assert text != WITH_COMPLEMENT

    def test_expected_kinds_for_rich_address(self) -> None:
        kinds = set(ev.negative_mutations("KR 26 H 3 # 73 - 10 AP 5"))
        for expected in (
            "via_number_plus1",
            "via_number_minus1",
            "via_type_change",
            "cross_number_plus1",
            "cross_number_minus1",
            "plate_plus1",
            "plate_minus1",
            "plate_plus2",
            "plate_minus2",
            "via_letter_drop",
            "via_letter_change",
            "cross_letter_add",
            "via_bis_add",
            "via_quadrant_add",
            "complement_value_change",
            "complement_removed",
        ):
            assert expected in kinds, expected

    def test_complement_added_when_none(self) -> None:
        assert "complement_added" in ev.negative_mutations(BASE)
        assert "complement_removed" not in ev.negative_mutations(BASE)

    def test_bis_remove_and_quadrant_remove(self) -> None:
        kinds = set(ev.negative_mutations("AV 15 BIS OESTE # 9 - 137"))
        assert "via_bis_remove" in kinds
        assert "via_quadrant_remove" in kinds

    def test_plate_minus_below_zero_is_skipped(self) -> None:
        kinds = set(ev.negative_mutations("KR 1 # 4 - 1"))
        assert "plate_minus1" not in kinds
        assert "plate_minus2" not in kinds
        assert "plate_plus1" in kinds

    def test_via_number_minus1_skipped_at_zero(self) -> None:
        assert "via_number_minus1" not in ev.negative_mutations("KR 0 # 9 - 10")

    def test_plate_letter_is_preserved(self) -> None:
        text = ev.negative_mutations("KR 1 # 4 - 80 A")["plate_plus1"]
        assert parse_canonical(text).plate == "81A"

    def test_no_cross_no_cross_mutations(self) -> None:
        kinds = set(ev.negative_mutations("KR 1 # - 10"))
        assert not any(k.startswith("cross_number") for k in kinds)

    def test_no_plate_no_plate_mutations(self) -> None:
        assert not any(k.startswith("plate_") for k in ev.negative_mutations(NO_PLATE))

    @pytest.mark.parametrize("value", [None, "", "  ", "junk", 3.5, "KR 1 # 4 - 80 CL 5 # 3"])
    def test_unparseable_input_yields_nothing(self, value: object) -> None:
        assert ev.negative_mutations(value) == {}

    def test_very_short_string(self) -> None:
        assert ev.negative_mutations("#") == {}

    def test_expectation_table_covers_every_generated_kind(self) -> None:
        rich = ["KR 26 H 3 # 73 - 10 AP 5", "AV 15 BIS OESTE # 9 - 137", BASE, NO_PLATE, BARE]
        for addr in rich:
            for kind in ev.negative_mutations(addr):
                assert kind in ev.EXPECTED_MUTATION_OUTCOME, kind

    def test_expectation_values_are_valid(self) -> None:
        assert set(ev.EXPECTED_MUTATION_OUTCOME.values()) <= {"no_match", "match"}
        assert ev.EXPECTED_MUTATION_OUTCOME["complement_added"] == "match"
        assert ev.EXPECTED_MUTATION_OUTCOME["complement_removed"] == "match"
        assert ev.EXPECTED_MUTATION_OUTCOME["plate_plus1"] == "no_match"

    def test_deterministic(self) -> None:
        assert ev.negative_mutations(BASE) == ev.negative_mutations(BASE)

    def test_current_matcher_rejects_core_mutations(self) -> None:
        for kind, text in ev.negative_mutations(BASE).items():
            if ev.EXPECTED_MUTATION_OUTCOME[kind] == "no_match":
                assert match(BASE, text).score < 0.90, kind


# --------------------------------------------------------------------------
# pair categorization / signatures
# --------------------------------------------------------------------------


class TestPairClassification:
    def test_signatures(self) -> None:
        parsed = parse_canonical("KR 26 H 3 # 73 - 10")
        assert ev.via_signature(parsed) == ("KR", "26", "H", "3", None, False, None)
        assert ev.cross_signature(parsed) == (None, "73", None, None, None, False, None)

    def test_unparsed_has_no_signature(self) -> None:
        bad = parse_canonical("junk")
        assert ev.via_signature(bad) is None
        assert ev.cross_signature(bad) is None
        assert ev.base_key(bad) is None

    def test_same_base_both_complements_differ(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80 AP 1")
        b = parse_canonical("KR 1 # 4 - 80 AP 2")
        assert ev.classify_pair(a, b) == "same_base_complements_differ"

    def test_same_base_one_side_lacks_complement(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80 AP 1")
        b = parse_canonical("KR 1 # 4 - 80")
        assert ev.classify_pair(a, b) == "same_base_one_side_bare"
        assert ev.classify_pair(b, a) == "same_base_one_side_bare"

    def test_same_base_same_complement(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80 AP 1")
        b = parse_canonical("KR 1 # 4 - 080 APTO 1")
        assert ev.classify_pair(a, b) == "same_base_same_complement"

    def test_same_base_both_bare(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80")
        b = parse_canonical("KR 1 # 4 - 080")
        assert ev.classify_pair(a, b) == "same_base_same_complement"

    def test_different_plate(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80")
        b = parse_canonical("KR 1 # 4 - 82")
        assert ev.classify_pair(a, b) == "different_plate"

    def test_different_cross(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80")
        b = parse_canonical("KR 1 # 10 - 80")
        assert ev.classify_pair(a, b) == "different_base_other"

    def test_unparseable(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80")
        assert ev.classify_pair(a, parse_canonical("")) == "unparseable"
        assert ev.classify_pair(parse_canonical(None), a) == "unparseable"

    def test_plate_letter_difference_is_a_different_plate(self) -> None:
        a = parse_canonical("KR 1 # 4 - 80")
        b = parse_canonical("KR 1 # 4 - 80 A")
        assert ev.classify_pair(a, b) == "different_plate"


# --------------------------------------------------------------------------
# small numeric / sampling helpers
# --------------------------------------------------------------------------


class TestPrecisionRecall:
    def test_basic(self) -> None:
        p, r, f1 = ev.precision_recall_f1(tp=8, fp=2, fn=2)
        assert (p, r) == (0.8, 0.8)
        assert f1 == pytest.approx(0.8)

    def test_all_zero_is_zero_not_error(self) -> None:
        assert ev.precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)

    def test_no_predicted_positive(self) -> None:
        p, r, f1 = ev.precision_recall_f1(0, 0, 5)
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_perfect(self) -> None:
        assert ev.precision_recall_f1(5, 0, 0) == (1.0, 1.0, 1.0)

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValueError):
            ev.precision_recall_f1(-1, 0, 0)


class TestSampleIndices:
    def test_deterministic_for_seed(self) -> None:
        assert ev.sample_indices(1000, 50, seed=7) == ev.sample_indices(1000, 50, seed=7)

    def test_different_seed_differs(self) -> None:
        assert ev.sample_indices(1000, 50, seed=7) != ev.sample_indices(1000, 50, seed=8)

    def test_sorted_unique_in_range(self) -> None:
        idx = ev.sample_indices(1000, 50, seed=1)
        assert idx == sorted(set(idx))
        assert len(idx) == 50 and 0 <= idx[0] and idx[-1] < 1000

    def test_sample_larger_than_total_returns_all(self) -> None:
        assert ev.sample_indices(5, 100, seed=1) == [0, 1, 2, 3, 4]

    def test_empty_and_zero(self) -> None:
        assert ev.sample_indices(0, 10, seed=1) == []
        assert ev.sample_indices(10, 0, seed=1) == []

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            ev.sample_indices(-1, 3, seed=1)


class TestScoreSummary:
    def test_empty(self) -> None:
        s = ev.summarize_scores([])
        assert s["count"] == 0 and s["mean"] is None

    def test_basic(self) -> None:
        s = ev.summarize_scores([0.0, 0.0, 0.85, 1.0])
        assert s["count"] == 4
        assert s["min"] == 0.0 and s["max"] == 1.0
        assert s["mean"] == pytest.approx(0.4625)
        assert sum(s["bins"].values()) == 4
        assert s["bins"]["0.00"] == 2 and s["bins"]["1.00"] == 1

    def test_bin_edges_are_stable(self) -> None:
        s = ev.summarize_scores([0.9, 0.8999, 0.9999])
        assert s["bins"]["0.90-0.99"] == 2
        assert s["bins"]["0.01-0.89"] == 1


class TestNoteFamily:
    def test_strips_token_suffix(self) -> None:
        assert ev.note_family("token_inesperado:XYZ") == "token_inesperado"

    def test_plain_note_unchanged(self) -> None:
        assert ev.note_family("placa_faltante") == "placa_faltante"

    def test_empty(self) -> None:
        assert ev.note_family("") == ""
        assert ev.note_family(None) == ""


class TestRateFormatting:
    def test_pct(self) -> None:
        assert ev.pct(1, 4) == "25.00%"

    def test_pct_zero_denominator(self) -> None:
        assert ev.pct(0, 0) == "n/a"


class TestFailureCause:
    @pytest.mark.parametrize("value", [None, "", "   ", 5])
    def test_empty_or_non_text(self, value: object) -> None:
        assert ev.failure_cause(value) == "empty_or_non_text"

    def test_no_hash(self) -> None:
        # Ambiguous split (more than one number per street): never guessed.
        assert ev.failure_cause("KR 26 G 5 73 - 13") == "no_hash_separator"
        assert ev.failure_cause("KR 1 9") == "no_hash_separator"

    def test_unambiguous_no_hash_is_now_parsed(self) -> None:
        assert ev.failure_cause("KR 1 4 - 80") == "parses_ok"

    def test_multiple_addresses(self) -> None:
        assert ev.failure_cause("KR 1 # 4 - 80 CL 5 # 3 - 2") == "multiple_addresses"
        assert ev.failure_cause("KR 1 # 4 - 80; CL 5 # 3 - 2") == "multiple_addresses"

    def test_via_number_missing(self) -> None:
        assert ev.failure_cause("M # -") == "via_number_missing"
        assert ev.failure_cause("KR # 4 - 80") == "via_number_missing"

    def test_two_letters_in_via(self) -> None:
        assert ev.failure_cause("KR 24 B C # 9 - 47") == "via_two_letters"

    def test_two_letters_in_cross(self) -> None:
        assert ev.failure_cause("KR 24 # 9 C B - 47") == "cross_two_letters"

    def test_via_other_unexpected_token(self) -> None:
        assert ev.failure_cause("KR 1 X 5 5 # 4 - 80") == "via_unexpected_token"

    def test_cross_other_unexpected_token(self) -> None:
        assert ev.failure_cause("KR 1 # 4 X 5 5 - 80") == "cross_unexpected_token"

    def test_tail_too_long(self) -> None:
        assert ev.failure_cause("KR 1 # 4 - 80 " + "X" * 41) == "tail_too_long"

    def test_tail_looks_like_second_address(self) -> None:
        assert ev.failure_cause("KR 1 # 4 - 80 CL 5") == "tail_via_type_then_number"

    def test_tail_unexpected_token(self) -> None:
        assert ev.failure_cause("KR 1 # 4 - 80 NORTE") == "tail_unexpected_token"
        assert ev.failure_cause("KR 1 # 4 - 80 X$Y") == "tail_unexpected_token"

    def test_plate_too_long(self) -> None:
        assert ev.failure_cause("KR 1 # 4 - " + "9" * 30) == "plate_too_long"

    @pytest.mark.parametrize(
        "text",
        [
            "C 1 # 74 -",
            "CL 35 # - LT 39",
            "K 49 E # 49 - 50 8 C",
            "C 71 I # 3 C N - 13",
            "C 70 B N # 4 C - 104",
            "KR 1 # 4 - 80 XXXX",
            "A 3 A N # 23 D N - 59",
            "KR 1 # 4 - 80",
        ],
    )
    def test_addresses_now_parseable_are_unclassified_ok(self, text: str) -> None:
        assert ev.failure_cause(text) == "parses_ok"

    def test_never_raises_on_garbage(self) -> None:
        for text in ("#", "-", "# -", "###", "٣ # 4", "K" * 10_000):
            assert isinstance(ev.failure_cause(text), str)


# --------------------------------------------------------------------------
# Phase 2: variants/mutations on the new raw cadastral shapes
# --------------------------------------------------------------------------

NEW_SHAPES = [
    "K 41 # 31 B -",
    "C 1 # 74 -",
    "A 9 W # -",
    "KR 69 # 33 - LT 20",
    "CL 6 OESTE # KR 4 - BO 000101 LT 0372",
    "K 47B # 55 B - Q2 T",
    "K 49 E # 49 - 50 8 C",
    "C 70 B N # 4 C - 104 38 C",
    "C 1 A BIS O # 81 - 19",
    "K 3 A 3 N # 71 H - 13 19 C",
    "CL 9 # 51 - 48 GASS 5",
    "K 8 # 22 - 48 /50 /52",
    "K 125 # # 18 - 55",
    "KR 1 # 4 - 80 XXXX LT 3",
]


class TestPhase2Shapes:
    @pytest.mark.parametrize("addr", NEW_SHAPES)
    def test_every_variant_parses_and_matches_the_original(self, addr: str) -> None:
        variants = ev.positive_variants(addr)
        assert variants, addr
        for kind, text in variants.items():
            assert match(addr, text).score == 1.0, (addr, kind, text)

    def test_legacy_prefix_gets_long_alias_variants(self) -> None:
        variants = ev.positive_variants("K 41 # 31 B -")
        assert variants["alias_long"].startswith("CARRERA 41")

    def test_legacy_a_prefix_gets_long_alias_variants(self) -> None:
        assert ev.positive_variants("A 9 # 5 - 3")["alias_long"].startswith("AVENIDA 9")

    def test_plate_less_complement_variants_exist(self) -> None:
        kinds = set(ev.positive_variants("KR 69 # 33 - LT 20"))
        assert {"complement_alias", "complement_zero_pad"} <= kinds

    @pytest.mark.parametrize("addr", NEW_SHAPES)
    def test_mutations_behave_as_designed(self, addr: str) -> None:
        for kind, text in ev.negative_mutations(addr).items():
            assert kind in ev.EXPECTED_MUTATION_OUTCOME, kind
            score = match(addr, text).score
            if ev.EXPECTED_MUTATION_OUTCOME[kind] == "match":
                assert score >= 0.90, (addr, kind, text, score)
            else:
                assert score < 0.90, (addr, kind, text, score)

    def test_removing_an_opaque_complement_is_a_true_difference(self) -> None:
        mutations = ev.negative_mutations("K 49 E # 49 - 50 8 C")
        assert "complement_removed" not in mutations
        assert ev.EXPECTED_MUTATION_OUTCOME["complement_removed_opaque"] == "no_match"
        assert match("K 49 E # 49 - 50 8 C", mutations["complement_removed_opaque"]).score == pytest.approx(0.85)

    def test_removing_a_known_complement_is_still_designed_match(self) -> None:
        mutations = ev.negative_mutations("KR 69 # 33 - LT 20")
        assert "complement_removed" in mutations and "complement_removed_opaque" not in mutations

    @pytest.mark.parametrize("address", ["KR 42 # 2 A - 09 LC", "CL 44 # 26 O - 25 GA", "KR 1 # 2 - 3 AP NORTE"])
    def test_removing_a_valueless_known_complement_is_a_true_difference(self, address: str) -> None:
        """S4: a known kind with no usable value is opaque for the matcher."""
        mutations = ev.negative_mutations(address)
        assert "complement_removed" not in mutations
        assert match(address, mutations["complement_removed_opaque"]).score == pytest.approx(0.85)

    def test_classify_pair_counts_a_repeated_chunk(self) -> None:
        """S5: complements compare as multisets, so AP 1 AP 1 is not AP 1."""
        a = parse_canonical("KR 1 # 2 - 3 AP 1 AP 1")
        b = parse_canonical("KR 1 # 2 - 3 AP 1")
        assert ev.classify_pair(a, b) == "same_base_complements_differ"

    def test_plate_less_address_has_no_plate_mutations(self) -> None:
        assert not any(k.startswith("plate_") for k in ev.negative_mutations("C 1 # 74 -"))

    def test_changing_an_opaque_value_vetoes(self) -> None:
        mutations = ev.negative_mutations("K 49 E # 49 - 50 8 C")
        assert match("K 49 E # 49 - 50 8 C", mutations["complement_value_change"]).score == 0.0

    def test_classify_pair_with_plate_less_addresses(self) -> None:
        a = parse_canonical("KR 69 # 33 - LT 20")
        b = parse_canonical("KR 69 # 33 - LT 21")
        assert ev.classify_pair(a, b) == "same_base_complements_differ"
        assert ev.classify_pair(parse_canonical("C 1 # 74 -"), parse_canonical("C 1 # 74")) == "same_base_same_complement"


# --------------------------------------------------------------------------
# Section H helpers (no pyarrow, tiny in-memory base)
# --------------------------------------------------------------------------

H_ROWS = [
    ("p1", "KR 1 # 4 - 80"),
    ("p2", "KR 1 # 4 - 82"),
    ("p3", "K 49 E # 49 - 50 8 C"),
    ("p4", "KR 1 # 4 - 80 AP 5"),
    ("p5", "CL 7 # 8 - 10"),
    ("p6", "CL 07 # 08 - 010"),  # same canonical as p5, different parcel
    ("p7", "unparseable text"),
]


def _h_index() -> AddressIndex:
    return AddressIndex.from_records(H_ROWS)


class TestProcessMemory:
    def test_returns_none_or_consistent_numbers(self) -> None:
        mem = ev.process_memory_bytes()
        assert mem is None or (mem[0] > 0 and mem[1] >= mem[0])


class TestEvaluateLinkQueries:
    def test_exact_queries_recover_the_own_parcel(self) -> None:
        queries = [("p1", "KR 1 # 4 - 80"), ("p2", "KR 1 # 4 - 82"), ("p3", "K 49 E # 49 - 50 8 C")]
        stats = ev.evaluate_link_queries(_h_index(), queries)
        assert stats["n"] == 3
        assert stats["top1_correct"] == 3
        assert stats["own_missing"] == 0
        assert stats["false_best"] == 0
        assert stats["ambiguous"] == 0

    def test_ambiguity_is_counted_and_not_a_false_best(self) -> None:
        stats = ev.evaluate_link_queries(_h_index(), [("p5", "CL 7 # 8 - 10"), ("p6", "CL 7 # 8 - 10")])
        assert stats["ambiguous"] == 2
        assert stats["own_missing"] == 0
        assert stats["false_best"] == 0
        assert stats["top1_correct"] + stats["tie_wrong_first"] == 2
        assert stats["tie_wrong_first"] == 1  # p6 loses the id tie-break to p5

    def test_false_best_is_reported(self) -> None:
        stats = ev.evaluate_link_queries(_h_index(), [("p2", "KR 1 # 4 - 80")])
        assert stats["false_best"] == 1
        assert stats["own_missing"] == 1
        assert stats["top1_correct"] == 0

    def test_unparseable_query_counts_as_missing_not_crash(self) -> None:
        stats = ev.evaluate_link_queries(_h_index(), [("p1", "garbage")])
        assert stats["n"] == 1 and stats["unparseable_queries"] == 1 and stats["own_missing"] == 1
        assert stats["no_best"] == 1

    def test_empty_query_list(self) -> None:
        stats = ev.evaluate_link_queries(_h_index(), [])
        assert stats["n"] == 0 and stats["avg_candidates"] is None and stats["queries_per_second"] is None

    def test_averages_and_examples(self) -> None:
        stats = ev.evaluate_link_queries(_h_index(), [("p1", "KR 1 # 4 - 80"), ("p2", "KR 1 # 4 - 80")])
        assert stats["avg_candidates"] == pytest.approx(2.0)
        assert stats["avg_compared"] == pytest.approx(2.0)
        assert len(stats["false_best_examples"]) == 1

    def test_query_with_complement_record_stays_unambiguous(self) -> None:
        stats = ev.evaluate_link_queries(_h_index(), [("p4", "KR 1 # 4 - 80 AP 5")])
        assert stats["top1_correct"] == 1 and stats["ambiguous"] == 0


class TestBuildLinkQueries:
    @staticmethod
    def _inputs() -> tuple[list[str], list[str], list]:
        addresses = [a for _, a in H_ROWS]
        ids = [i for i, _ in H_ROWS]
        return addresses, ids, [parse_canonical(a) for a in addresses]

    def test_deterministic_and_bounded(self) -> None:
        addresses, ids, parsed = self._inputs()
        first = ev.build_link_queries(addresses, ids, parsed, seed=42, sample=3)
        again = ev.build_link_queries(addresses, ids, parsed, seed=42, sample=3)
        assert first == again
        assert len(first["exact"]) == 3
        assert len(first["variant"]) <= 3
        assert all(qid in ids for qid, _ in first["exact"])

    def test_only_parseable_rows_are_sampled(self) -> None:
        addresses, ids, parsed = self._inputs()
        queries = ev.build_link_queries(addresses, ids, parsed, seed=1, sample=100)
        assert len(queries["exact"]) == 6
        assert "p7" not in {qid for qid, _ in queries["exact"]}

    def test_variants_differ_from_exact_and_match_the_original(self) -> None:
        addresses, ids, parsed = self._inputs()
        queries = ev.build_link_queries(addresses, ids, parsed, seed=5, sample=6)
        exact = dict(queries["exact"])
        assert queries["variant"]
        for qid, text in queries["variant"]:
            assert text != exact[qid]
            assert match(exact[qid], text).score >= 0.90

    def test_zero_sample(self) -> None:
        addresses, ids, parsed = self._inputs()
        assert ev.build_link_queries(addresses, ids, parsed, seed=1, sample=0) == {"exact": [], "variant": []}


class TestMeasureLinking:
    def test_end_to_end_small(self) -> None:
        addresses = [a for _, a in H_ROWS]
        parcels = [i for i, _ in H_ROWS]
        parsed = [parse_canonical(a) for a in addresses]
        out = ev.measure_linking(addresses, parcels, parsed, seed=42, sample=6)
        assert out["index"]["records"] == 6 and out["index"]["rejected"] == 0
        assert out["exact"]["own_missing"] == 0
        assert out["variant"]["own_missing"] == 0
        assert out["build_seconds"] >= 0.0

    def test_without_parcel_ids_uses_row_numbers(self) -> None:
        addresses = [a for _, a in H_ROWS]
        parsed = [parse_canonical(a) for a in addresses]
        out = ev.measure_linking(addresses, None, parsed, seed=42, sample=6)
        assert out["exact"]["own_missing"] == 0

    def test_empty_corpus(self) -> None:
        out = ev.measure_linking([], None, [], seed=1, sample=10)
        assert out["index"]["records"] == 0 and out["exact"]["n"] == 0


class TestRunEvaluationSmoke:
    def test_full_pipeline_on_tiny_corpus_renders_every_section(self) -> None:
        addresses = [a for _, a in H_ROWS] + NEW_SHAPES
        parcels = [i for i, _ in H_ROWS] + [f"n{k}" for k in range(len(NEW_SHAPES))]
        res = ev.run_evaluation(addresses, parcels, sample=None, seed=42, max_pairs=1000, link_sample=50)
        report = ev.format_report(res)
        for marker in ("== A.", "== B.", "== C.", "== D.", "== E.", "== F.", "== G.", "== H."):
            assert marker in report
        assert res["B"]["self_match_violations"] == 0
        assert res["B"]["symmetry_violations"] == 0
        assert res["H"]["exact"]["own_missing"] == 0
        assert res["H"]["variant"]["own_missing"] == 0
        for kind, stat in res["D"].items():
            assert stat["as_designed"] == stat["n"], kind
