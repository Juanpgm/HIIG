"""Tests for tools/build_groundtruth.py (ground-truth pair set builder).

All fixtures are small in-memory cadastre-like records; the real parquet is
never needed (reading it is isolated in ``load_catastro``).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import build_groundtruth as gt  # noqa: E402

STRATA = (
    "complement_one_sided",
    "cross_type_presence",
    "plate_letter",
    "adjacent_or_swap",
    "same_base_diff_complement",
    "opaque_or_degenerate",
)


def _digits(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"\d+", text)]


def make_records(n_streets: int = 14) -> list[gt.Record]:
    """Deterministic fake cadastre with every structure the strata need."""
    records: list[gt.Record] = []
    k = 0

    def add(addr: str, comuna: str = "01", manzana: str = "M1") -> None:
        nonlocal k
        k += 1
        records.append(gt.Record(addr, f"NPN{k:06d}", manzana, comuna, f"BARRIO {comuna}"))

    for s in range(1, n_streets + 1):
        comuna = f"{(s % 4) + 1:02d}"
        for c in range(3, 9):
            for plate in (10, 11, 12, 14, 20):
                base = f"KR {s} # {c} - {plate}"
                add(base, comuna, f"MZ{s}")
                add(f"KR {s} # {c} - {plate} AP 101", comuna, f"MZ{s}")
                add(f"KR {s} # {c} - {plate} AP 202", comuna, f"MZ{s}")
                add(f"KR {s} # {c} - {plate} BLQ 3", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 30 A", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 30", comuna, f"MZ{s}")
            # Zero-padded twins: the same numeric plate written with and without leading zeros.
            add(f"KR {s} # {c} - 030", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 05", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 5", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 005", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 5 B", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 31 B", comuna, f"MZ{s}")
            add(f"KR {s} # {c} -", comuna, f"MZ{s}")
            add(f"KR {s} # {c} - 40 8 C", comuna, f"MZ{s}")
            add(f"CL {c} # {s} - 10", comuna, f"MZ{s}")
            add(f"CL {s} # AV {c} - 50", comuna, f"MZ{s}")
            add(f"CL {s} # KR {c} - 50", comuna, f"MZ{s}")  # two different cross types on the same base
            add(f"CL {s} # {c} - 50", comuna, f"MZ{s}")
    return records


SIZES_SMALL = {
    "synthetic": 40,
    "per_stratum": 6,
    "sanity": 8,
}


# ---------------------------------------------------------------------------
# split_address / transforms
# ---------------------------------------------------------------------------


class TestSplitAddress:
    def test_splits_standard_address(self):
        parts = gt.split_address("KR 26 H 3 # 73 - 10 AP 101")
        assert parts is not None
        assert parts["via_type"] == "KR"
        assert parts["plate"] == "10"
        assert parts["complement"] == "AP 101"

    @pytest.mark.parametrize("bad", [None, "", "   ", 5, 3.2, b"CL 5 # 1 - 2", [], {}, "no hash here"])
    def test_empty_none_and_non_string_addresses_are_rejected(self, bad):
        assert gt.split_address(bad) is None

    def test_input_over_the_length_cap_is_rejected_without_running_the_regex(self):
        # Rejection is a constant-time length check, not regex work.
        assert len("CL " + "A " * 50_000 + "# 1 - 2") > gt.MAX_ADDRESS_LEN
        assert gt.split_address("CL " + "A " * 50_000 + "# 1 - 2") is None
        assert gt.split_address("CL 5 # " + "9" * 10_000 + " - 1") is None

    def test_pathological_input_just_under_the_cap_stays_fast(self):
        import time

        near_cap = "CL 5 # 10 - 20 " + "A" * (gt.MAX_ADDRESS_LEN - 15)
        assert len(near_cap) <= gt.MAX_ADDRESS_LEN
        for text in (near_cap, "CL " + "1 " * (gt.MAX_ADDRESS_LEN // 2 - 5), "CL 5 # " + "9" * (gt.MAX_ADDRESS_LEN - 20)):
            assert len(text) <= gt.MAX_ADDRESS_LEN
            start = time.perf_counter()
            gt.split_address(text)  # result irrelevant here: it must simply return promptly
            assert time.perf_counter() - start < 0.5

    def test_unicode_dash_is_not_a_canonical_separator(self):
        # The strict splitter only knows ASCII '-'; the en dash must be what makes it fail.
        assert gt.split_address("CL 5 # 10 - 20") is not None
        assert gt.split_address("CL 5 # 10 – 20") is None
        assert gt.split_address("CL 5 # 10 — 20") is None

    def test_fullwidth_input_is_not_canonical(self):
        assert gt.split_address("ＣＬ ５ ＃ １０ － ２０") is None

class TestTransforms:
    ADDRS = [
        "KR 26 H 3 # 73 - 10",
        "CL 5 # 10 - 20",
        "K 41 B # 31 - 55",
        "AV 5 OESTE # 26 OESTE - 56",
        "CL 25 NORTE # AV 6 - 30",
        "KR 100 A 1 # 2 A OESTE - 05 ED H AP 101",
        "DG 23 # 5 - 4 BLQ 7 LC 2",
        "TV 4 # 20 A - 110",
    ]

    def test_every_transform_has_a_documented_rationale(self):
        assert set(gt.TRANSFORMS) == set(gt.TRANSFORM_RATIONALE)
        assert all(gt.TRANSFORM_RATIONALE[t].strip() for t in gt.TRANSFORMS)

    def test_transform_set_covers_required_families(self):
        required = {"lowercase", "extra_whitespace", "hash_to_no", "hash_absent", "type_alias_long",
                    "type_alias_short", "glued_tokens", "zero_padded", "unicode_spacing",
                    "trailing_whitespace", "complement_alias"}
        assert required <= set(gt.TRANSFORMS)

    @pytest.mark.parametrize("tag", sorted(gt.TRANSFORMS))
    def test_mutations_never_alter_digits(self, tag):
        import random

        produced = 0
        for addr in self.ADDRS:
            for seed in range(5):
                out = gt.apply_transform(tag, addr, random.Random(seed))
                if out is None:
                    continue
                produced += 1
                assert _digits(out) == _digits(addr), (tag, addr, out)
                assert out != addr
        assert produced > 0, f"{tag} never applied to any sample"

    @pytest.mark.parametrize("tag", sorted(gt.TRANSFORMS))
    def test_mutation_leaving_address_unchanged_is_discarded(self, tag):
        import random

        # Already lower-case / already-padded / no complement etc. => None or changed, never equal.
        for addr in ["cl 5 # 10 - 20", "CL 5 # 010 - 020", "", "x"]:
            out = gt.apply_transform(tag, addr, random.Random(0))
            assert out is None or out != addr

    @pytest.mark.parametrize("tag", sorted(gt.TRANSFORMS))
    @pytest.mark.parametrize("bad", [None, "", 12, b"x", ["CL 5 # 1 - 2"]])
    def test_transforms_tolerate_non_string_and_empty(self, tag, bad):
        import random

        assert gt.apply_transform(tag, bad, random.Random(0)) is None

    def test_lowercase_transform_changes_case_only(self):
        import random

        out = gt.apply_transform("lowercase", "KR 26 H 3 # 73 - 10", random.Random(1))
        assert out is not None and out.upper() == "KR 26 H 3 # 73 - 10"

    def test_unicode_and_huge_inputs_do_not_crash_transforms(self):
        import random

        for tag in gt.TRANSFORMS:
            assert gt.apply_transform(tag, "CL 5 # 10 - 20 " + "ñ" * 5000, random.Random(0)) is None
            assert gt.apply_transform(tag, "Ñ" * 100_000, random.Random(0)) is None


class TestHashAbsent:
    @pytest.mark.parametrize("addr", ["KR 26 H 3 # 73 - 10", "KR 100 A 1 # 2 A OESTE - 05 ED H AP 101", "CL 5 # 10 1 - 20"])
    def test_multi_number_streets_are_skipped(self, addr):
        import random

        for seed in range(5):
            assert gt.apply_transform("hash_absent", addr, random.Random(seed)) is None

    @pytest.mark.parametrize("addr, expected", [
        ("CL 5 # 10 - 20", "CL 5 10 - 20"),
        ("CL 25 NORTE # AV 6 - 30", "CL 25 NORTE AV 6 - 30"),
        ("K 41 B # 31 - 55", "K 41 B 31 - 55"),
    ])
    def test_single_number_streets_drop_only_the_hash(self, addr, expected):
        import random

        assert gt.apply_transform("hash_absent", addr, random.Random(0)) == expected

    def test_generated_rows_never_come_from_ambiguous_sources(self):
        recs = [gt.Record("KR 26 H 3 # 73 - 10", "N1", "m", "01"), gt.Record("KR 1 A 2 # 3 B 4 - 5", "N2", "m", "01")]
        rows, report = gt.generate_synthetic(recs, n=60, seed=1)
        assert report["by_transform"]["hash_absent"] == 0
        assert not [r for r in rows if r["transform"] == "hash_absent"]

    def test_generated_rows_use_single_number_sources_when_available(self):
        rows, report = gt.generate_synthetic(make_records() + [gt.Record("KR 26 H 3 # 73 - 10", "X", "m", "01")],
                                             n=170, seed=2)
        made = [r for r in rows if r["transform"] == "hash_absent"]
        assert made and report["by_transform"]["hash_absent"] == len(made)
        for r in made:
            parts = gt.split_address(r["source_direccion"])
            assert len(_digits(parts["via"])) == 1 and len(_digits(parts["cross"])) == 1
            assert " # " not in r["addr_b"] and "#" not in r["addr_b"]


class TestNovelAliases:
    NOVEL = ("novel_type_alias", "novel_complement_alias", "novel_number_word")
    ADDRS = ["KR 26 # 73 - 10", "AV 5 # 10 - 20 AP 101", "DG 23 # 5 - 4 BLQ 7 LC 2", "TV 4 # 20 - 110 OF 5 ED 3",
             "KR 5 # 10 - 20 AP 1"]

    def test_novel_tags_are_registered_with_rationale(self):
        for tag in self.NOVEL:
            assert tag in gt.TRANSFORMS and gt.TRANSFORM_RATIONALE[tag].strip()

    @pytest.mark.parametrize("tag", NOVEL)
    def test_each_novel_family_applies_and_keeps_digits(self, tag):
        import random

        produced = 0
        for addr in self.ADDRS:
            for seed in range(6):
                out = gt.apply_transform(tag, addr, random.Random(seed))
                if out is not None:
                    produced += 1
                    assert _digits(out) == _digits(addr) and out != addr
        assert produced > 0

    def test_novel_complement_spellings_keep_meaning_and_map_to_the_source_kind(self):
        import random

        from emparejador.parser import COMPLEMENT_KINDS

        reverse = {alias: canonical for canonical, aliases in gt._NOVEL_COMPLEMENT.items() for alias in aliases}
        # These spellings were unknown when the set was designed; the parser
        # has since learned them, so they must now all map to the SAME kind
        # the builder derived them from.
        for canonical, aliases in gt._NOVEL_COMPLEMENT.items():
            for alias in aliases:
                assert COMPLEMENT_KINDS[alias.rstrip(".")] == canonical, alias
        seen = set()
        for addr in self.ADDRS:
            for seed in range(20):
                out = gt.apply_transform("novel_complement_alias", addr, random.Random(seed))
                if out is None:
                    continue
                back = " ".join(reverse.get(tok, tok) for tok in out.split(" "))
                assert back == addr, (addr, out)  # meaning preserved: mapping back gives the original
                seen.update(tok for tok in out.split(" ") if tok in reverse)
        assert {"APT", "LOC"} <= {t.rstrip(".") for t in seen}

    def test_novel_type_spellings_map_to_the_source_type(self):
        from emparejador.parser import VIA_TYPES

        # Learned since the set was designed: each spelling must map to the
        # same canonical type as the long word it was derived from.
        # CIRCULAR/CIR/CIRC are the exception: deliberately not aliased
        # (circular vs circunvalar is ambiguous), so they stay unparseable.
        for long_word, aliases in gt._NOVEL_TYPE.items():
            for alias in aliases:
                if long_word == "CIRCULAR":
                    assert alias not in VIA_TYPES, alias
                else:
                    assert VIA_TYPES[alias] == VIA_TYPES[long_word], alias

    def test_novel_number_word_is_not_a_known_marker(self):
        import random

        assert not set(gt._NOVEL_NUMBER_WORDS) & {"NO", "NRO", "NUM"}  # distinct from the older markers
        out = gt.apply_transform("novel_number_word", "CL 5 # 10 - 20", random.Random(0))
        assert out == "CL 5 NUMERO 10 - 20"

    def test_known_vocab_tables_only_use_spellings_the_parser_knows(self):
        from emparejador.parser import COMPLEMENT_KINDS, VIA_TYPES

        for aliases in gt._COMPLEMENT_LONG.values():
            assert set(aliases) <= set(COMPLEMENT_KINDS)
        for aliases in gt._TYPE_SHORT.values():
            assert set(aliases) <= set(VIA_TYPES)

    def test_quotas_include_novel_tags_and_total_stays_near_request(self):
        rows, report = gt.generate_synthetic(make_records(), n=250, seed=1)
        assert all(report["by_transform"][t] > 0 for t in self.NOVEL)
        assert 240 <= len(rows) <= 250


# ---------------------------------------------------------------------------
# synthetic positives
# ---------------------------------------------------------------------------


class TestSyntheticPositives:
    def test_rows_have_required_fields_and_label(self):
        rows, report = gt.generate_synthetic(make_records(), n=40, seed=3)
        assert rows
        for r in rows:
            assert set(r) == {"pair_id", "addr_a", "addr_b", "same_door", "tier", "transform", "source_direccion"}
            assert r["same_door"] is True and r["tier"] == "synthetic"
            assert r["addr_a"] == r["source_direccion"]
            assert r["addr_a"] != r["addr_b"]
            assert _digits(r["addr_a"]) == _digits(r["addr_b"])
            assert r["transform"] in gt.TRANSFORMS or r["transform"] == "combo"

    def test_stratified_roughly_evenly_across_tags(self):
        rows, _ = gt.generate_synthetic(make_records(), n=len(gt.TRANSFORMS) * 4, seed=5)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["transform"]] = counts.get(r["transform"], 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 3
        assert len(counts) >= len(gt.TRANSFORMS) - 1
        assert any(t.startswith("novel_") for t in counts)

    def test_pair_ids_unique_and_no_duplicate_pairs(self):
        rows, _ = gt.generate_synthetic(make_records(), n=120, seed=1)
        assert len({r["pair_id"] for r in rows}) == len(rows)
        assert len({(r["addr_a"], r["addr_b"]) for r in rows}) == len(rows)

    def test_no_self_pairs(self):
        rows, _ = gt.generate_synthetic(make_records(), n=80, seed=2)
        assert all(r["addr_a"] != r["addr_b"] for r in rows)

    def test_determinism_and_seed_sensitivity(self):
        a, _ = gt.generate_synthetic(make_records(), n=40, seed=7)
        b, _ = gt.generate_synthetic(make_records(), n=40, seed=7)
        c, _ = gt.generate_synthetic(make_records(), n=40, seed=8)
        assert a == b
        assert a != c

    def test_empty_and_dirty_records_do_not_crash(self):
        dirty = [gt.Record(None, "1", "m", "01"), gt.Record("", "2", "m", "01"), gt.Record(42, "3", "m", "01"),  # type: ignore[arg-type]
                 gt.Record("ÑÑÑ", "4", "m", "01"), gt.Record("X" * 100_000, "5", "m", "01")]
        rows, report = gt.generate_synthetic(dirty, n=10, seed=1)
        assert rows == []
        assert report["produced"] == 0

    def test_empty_input_reports_shortfall_without_crash(self):
        rows, report = gt.generate_synthetic([], n=10, seed=1)
        assert rows == [] and report["requested"] == 10 and report["produced"] == 0

    def test_zero_requested(self):
        rows, _ = gt.generate_synthetic(make_records(), n=0, seed=1)
        assert rows == []


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def _fake_scorer(a: str, b: str) -> float:
    """Stable pseudo score spread over [0, 1] so all bins fill."""
    return (sum(map(ord, a + b)) % 100) / 100.0


class TestCandidates:
    def test_every_stratum_is_populated_on_rich_data(self):
        rows, report = gt.generate_candidates(
            make_records(), per_stratum=6, seed=1, scorer=_fake_scorer
        )
        by = {s: [r for r in rows if r["stratum"] == s] for s in STRATA}
        for s in STRATA:
            assert by[s], f"stratum {s} empty"
            assert len(by[s]) <= 6
            assert report[s]["produced"] == len(by[s])
            assert report[s]["requested"] == 6

    def test_rows_never_contain_scores(self):
        rows, _ = gt.generate_candidates(make_records(), per_stratum=5, seed=1, scorer=_fake_scorer)
        for r in rows:
            assert set(r) == {"pair_id", "addr_a", "addr_b", "stratum", "npn_a", "npn_b", "comuna_a", "comuna_b",
                              "barrio_a", "barrio_b", "same_manzana"}
            assert isinstance(r["same_manzana"], bool)
            assert "score" not in json.dumps(r).lower()

    def test_no_self_pairs_and_no_symmetric_duplicates(self):
        rows, _ = gt.generate_candidates(make_records(), per_stratum=10, seed=4, scorer=_fake_scorer)
        seen = set()
        for r in rows:
            assert r["addr_a"] != r["addr_b"]
            assert r["npn_a"] != r["npn_b"]
            key = frozenset((r["npn_a"], r["npn_b"]))
            assert key not in seen, "symmetric duplicate (a,b)/(b,a)"
            seen.add(key)
        assert len({r["pair_id"] for r in rows}) == len(rows)

    def test_pair_id_is_symmetric_invariant(self):
        assert gt.pair_id("cand", ("A", "1"), ("B", "2")) == gt.pair_id("cand", ("B", "2"), ("A", "1"))
        assert gt.pair_id("cand", ("A", "1"), ("B", "2")) != gt.pair_id("cand", ("A", "1"), ("C", "2"))

    def test_stratum_semantics(self):
        rows, _ = gt.generate_candidates(make_records(), per_stratum=8, seed=2, scorer=_fake_scorer)
        for r in rows:
            a, b = r["addr_a"], r["addr_b"]
            if r["stratum"] == "complement_one_sided":
                assert (" AP " in a) != (" AP " in b) or (" BLQ " in a) != (" BLQ " in b)
            if r["stratum"] == "cross_type_presence":
                assert (_cross_type(a) == "") != (_cross_type(b) == "")
            if r["stratum"] == "plate_letter":
                assert a != b
                assert _digits(a) == _digits(b)
            if r["stratum"] == "same_base_diff_complement":
                assert " - " in a and a.split(" - ")[0] == b.split(" - ")[0]

    def test_plate_letter_pairs_differ_by_letter_never_by_zero_padding(self):
        rows, report = gt.generate_candidates(make_records(), per_stratum=60, seed=3, scorer=_fake_scorer)
        chosen = [r for r in rows if r["stratum"] == "plate_letter"]
        assert len(chosen) >= 20, report["plate_letter"]
        for r in chosen:
            (na, la), (nb, lb) = _plate_parts(r["addr_a"]), _plate_parts(r["addr_b"])
            assert na == nb, (r["addr_a"], r["addr_b"])  # same numeric plate
            assert la != lb, (r["addr_a"], r["addr_b"])  # and the letter is what differs
            assert _base(r["addr_a"]) == _base(r["addr_b"])

    def test_cross_type_presence_pairs_have_exactly_one_typed_cross(self):
        rows, report = gt.generate_candidates(make_records(), per_stratum=60, seed=3, scorer=_fake_scorer)
        chosen = [r for r in rows if r["stratum"] == "cross_type_presence"]
        assert len(chosen) >= 20, report["cross_type_presence"]
        for r in chosen:
            a, b = r["addr_a"], r["addr_b"]
            assert _cross_type(a) != "" or _cross_type(b) != ""
            assert (_cross_type(a) == "") != (_cross_type(b) == ""), (a, b)
            assert _strip_cross_type(a) == _strip_cross_type(b), (a, b)

    def test_adjacent_or_swap_semantics_with_zero_padded_plates(self):
        rows, report = gt.generate_candidates(make_records(), per_stratum=60, seed=3, scorer=_fake_scorer)
        chosen = [r for r in rows if r["stratum"] == "adjacent_or_swap"]
        assert len(chosen) >= 20, report["adjacent_or_swap"]
        kinds = set()
        for r in chosen:
            pa, pb = gt.split_address(r["addr_a"]), gt.split_address(r["addr_b"])
            assert pa is not None and pb is not None, r
            (na, _), (nb, _) = _plate_parts(r["addr_a"]), _plate_parts(r["addr_b"])
            same_street = (pa["via_type"], pa["via"], pa["cross"]) == (pb["via_type"], pb["via"], pb["cross"])
            swapped = (_digits(pa["via"]) == _digits(pb["cross"]) and _digits(pa["cross"]) == _digits(pb["via"])
                       and _digits(pa["via"]) != _digits(pa["cross"]))
            if same_street:
                # adjacent plate: a real +/-1 or +/-2 step; zero padding (delta 0) does not count
                assert abs(na - nb) in (1, 2), (r["addr_a"], r["addr_b"])
                kinds.add("adjacent")
            else:
                assert swapped and na == nb, (r["addr_a"], r["addr_b"])
                kinds.add("swap")
        assert kinds == {"adjacent", "swap"}

    def test_opaque_or_degenerate_semantics_with_zero_padded_plates(self):
        from emparejador.parser import COMPLEMENT_KINDS

        rows, report = gt.generate_candidates(make_records(), per_stratum=60, seed=3, scorer=_fake_scorer)
        chosen = [r for r in rows if r["stratum"] == "opaque_or_degenerate"]
        assert len(chosen) >= 10, report["opaque_or_degenerate"]
        for r in chosen:
            a, b = r["addr_a"], r["addr_b"]
            odd = 0
            for x, y in ((a, b), (b, a)):
                parts = gt.split_address(x)
                no_plate = parts is None and x.rstrip().endswith("-")
                opaque = parts is not None and bool(parts["complement"]) and (
                    parts["complement"].split(" ")[0] not in COMPLEMENT_KINDS and len(parts["complement"]) > 1
                )
                unparsed = parts is None
                odd += bool(no_plate or opaque or unparsed)
            assert odd >= 1, (a, b)

    def test_determinism_and_seed_sensitivity(self):
        recs = make_records()
        a, _ = gt.generate_candidates(recs, per_stratum=6, seed=11, scorer=_fake_scorer)
        b, _ = gt.generate_candidates(recs, per_stratum=6, seed=11, scorer=_fake_scorer)
        c, _ = gt.generate_candidates(recs, per_stratum=6, seed=12, scorer=_fake_scorer)
        assert a == b
        assert a != c

    def test_quota_shortfall_reported_not_crashing(self):
        tiny = [gt.Record("KR 1 # 2 - 3", "N1", "m", "01"), gt.Record("KR 1 # 2 - 3 AP 1", "N2", "m", "01")]
        rows, report = gt.generate_candidates(tiny, per_stratum=50, seed=1, scorer=_fake_scorer)
        assert len([r for r in rows if r["stratum"] == "complement_one_sided"]) == 1
        assert report["complement_one_sided"]["produced"] == 1
        assert report["complement_one_sided"]["shortfall"] == 49
        assert report["plate_letter"]["produced"] == 0

    def test_empty_records_and_dirty_records(self):
        rows, report = gt.generate_candidates([], per_stratum=5, seed=1, scorer=_fake_scorer)
        assert rows == [] and all(report[s]["shortfall"] == 5 for s in STRATA)
        dirty = [gt.Record(None, "1", "m", "01"), gt.Record(7, "2", "m", "01"), gt.Record("", "3", None, "01")]  # type: ignore[arg-type]
        rows, _ = gt.generate_candidates(dirty, per_stratum=5, seed=1, scorer=_fake_scorer)
        assert rows == []

    def test_huge_and_unicode_direccion_do_not_crash(self):
        recs = make_records(3) + [gt.Record("Ñ" * 50_000, "H1", "m", "01"), gt.Record("CL 5 # 10 – 20 ñ", "H2", "m", "01")]
        rows, _ = gt.generate_candidates(recs, per_stratum=3, seed=1, scorer=_fake_scorer)
        assert isinstance(rows, list)

    def test_bins_are_oversampled_near_threshold(self):
        # Plenty of pairs in every bin: the 0.85-0.95 bins must get their heavier share, exactly.
        scored = [((i, i + 1), score) for i, score in enumerate(
            s for b in (0.5, 0.75, 0.87, 0.92, 0.99) for s in [b] * 400)]
        chosen = gt._select(scored, 60, __import__("random").Random(1), lambda _pair: True)
        per_bin = [0] * len(gt.BIN_LABELS)
        lookup = dict(scored)
        for pair in chosen:
            per_bin[gt.score_bin(lookup[pair])] += 1
        assert len(chosen) == 60
        assert per_bin == [10, 10, 15, 15, 10]
        assert per_bin[2] + per_bin[3] > 0.45 * 60  # the threshold neighbourhood is over-represented

    def test_bin_weights_favour_the_threshold_neighbourhood(self):
        assert len(gt.BIN_WEIGHTS) == len(gt.BIN_LABELS)
        assert gt.BIN_WEIGHTS[2] > gt.BIN_WEIGHTS[0] and gt.BIN_WEIGHTS[3] > gt.BIN_WEIGHTS[4]
        assert gt.BIN_WEIGHTS[2] == gt.BIN_WEIGHTS[3]

    def test_select_redistributes_when_a_bin_is_empty(self):
        scored = [((i, i + 1), 0.99) for i in range(100)]
        chosen = gt._select(scored, 30, __import__("random").Random(1), lambda _pair: True)
        assert len(chosen) == 30

    def test_plate_letter_shortfall_is_reported_when_only_zero_padding_differs(self):
        recs = [gt.Record(f"KR 1 # 2 - {p}", f"N{k}", "m", "01") for k, p in enumerate(("5", "05", "005", "0005"))]
        rows, report = gt.generate_candidates(recs, per_stratum=10, seed=1, scorer=_fake_scorer)
        assert [r for r in rows if r["stratum"] == "plate_letter"] == []
        assert report["plate_letter"]["produced"] == 0 and report["plate_letter"]["shortfall"] == 10

    def test_plate_letter_present_vs_absent_and_different_letters_qualify(self):
        recs = [gt.Record(f"KR 1 # 2 - {p}", f"N{k}", "m", "01") for k, p in enumerate(("5", "05 A", "5 B"))]
        rows, _ = gt.generate_candidates(recs, per_stratum=10, seed=1, scorer=_fake_scorer)
        chosen = [r for r in rows if r["stratum"] == "plate_letter"]
        assert len(chosen) == 3

    def test_score_bin_boundaries(self):
        assert gt.score_bin(0.6999) == 0
        assert gt.score_bin(0.70) == 1
        assert gt.score_bin(0.85) == 2
        assert gt.score_bin(0.90) == 3  # exactly the threshold is on the MATCH side
        assert gt.score_bin(0.95) == 4
        assert gt.score_bin(1.0) == 4
        assert gt.score_bin(0.0) == 0


def _plate_parts(addr: str) -> tuple[int, str]:
    m = re.search(r" - (\d+)(?: ?([A-Z])(?![A-Z0-9]))?(?: |$)", addr)
    assert m is not None, addr
    return int(m.group(1)), m.group(2) or ""


def _base(addr: str) -> str:
    return addr.split(" - ")[0]


def _cross_type(addr: str) -> str:
    m = re.search(r" # ([A-Z]+) \d", addr)
    return m.group(1) if m else ""


def _strip_cross_type(addr: str) -> str:
    return re.sub(r" # [A-Z]+ (\d)", r" # \1", addr)


class TestRandomNegatives:
    def test_negatives_are_cross_comuna_and_labeled_false(self):
        rows, report = gt.generate_random_negatives(make_records(), n=12, seed=1)
        assert len(rows) == 12 and report["produced"] == 12
        recs = {r.npn: r for r in make_records()}
        for r in rows:
            assert r["same_door"] is False and r["tier"] == "sanity" and "label" not in r
            assert r["addr_a"] != r["addr_b"]
            assert recs[r["npn_a"]].comuna != recs[r["npn_b"]].comuna

    def test_single_comuna_yields_shortfall_not_crash(self):
        recs = [gt.Record(f"KR {i} # 1 - 1", f"N{i}", "m", "01") for i in range(1, 10)]
        rows, report = gt.generate_random_negatives(recs, n=5, seed=1)
        assert rows == [] and report["shortfall"] == 5

    def test_deterministic(self):
        a, _ = gt.generate_random_negatives(make_records(), n=10, seed=9)
        b, _ = gt.generate_random_negatives(make_records(), n=10, seed=9)
        c, _ = gt.generate_random_negatives(make_records(), n=10, seed=10)
        assert a == b and a != c

    def test_no_symmetric_duplicates(self):
        rows, _ = gt.generate_random_negatives(make_records(), n=40, seed=2)
        keys = {frozenset((r["npn_a"], r["npn_b"])) for r in rows}
        assert len(keys) == len(rows)


# ---------------------------------------------------------------------------
# end-to-end build
# ---------------------------------------------------------------------------


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _build(tmp_path, seed=1, sizes=SIZES_SMALL, records=None, name="groundtruth", **kw):
    """Build into ``tmp_path/<name>``; the private sidecar lands in the sibling ``<name>_private``."""
    out = tmp_path / name
    manifest = gt.build_groundtruth(make_records() if records is None else records, out, seed=seed, sizes=sizes, **kw)
    return out, manifest


class TestBuild:
    def test_outputs_and_manifest(self, tmp_path):
        tmp_path, manifest = _build(tmp_path)
        for name in ("synthetic_positives.jsonl", "candidates_to_label.jsonl", "random_negatives.jsonl",
                     "label_sheet.csv", "MANIFEST.json"):
            assert (tmp_path / name).is_file(), name
        on_disk = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["seed"] == 1
        assert manifest["parquet_rows"] == len(make_records())
        assert set(manifest["sha256"]) >= {"synthetic_positives.jsonl", "label_sheet.csv"}
        assert "counts" in manifest and "synthetic_by_transform" in manifest["counts"]

    def test_sidecar_lives_outside_the_labeling_folder(self, tmp_path):
        out, manifest = _build(tmp_path)
        private = tmp_path / "groundtruth_private"
        assert (private / "matcher_scores_sidecar.jsonl").is_file()
        assert not (out / "matcher_scores_sidecar.jsonl").exists()
        assert [p.name for p in out.iterdir() if "score" in p.name.lower()] == []
        assert manifest["sidecar"]["file"] == "matcher_scores_sidecar.jsonl" and "dir_name" not in manifest["sidecar"]
        assert "matcher_scores_sidecar.jsonl" not in manifest["sha256"]

    def test_default_private_dir_is_a_sibling(self, tmp_path):
        assert gt.default_private_dir(tmp_path / "data" / "groundtruth") == tmp_path / "data" / "groundtruth_private"

    def test_explicit_private_dir(self, tmp_path):
        custom = tmp_path / "elsewhere"
        out, _ = _build(tmp_path, private_dir=custom)
        assert (custom / "matcher_scores_sidecar.jsonl").is_file()
        assert not (tmp_path / "groundtruth_private").exists()

    def test_sidecar_covers_every_pair_and_only_sidecar_has_scores(self, tmp_path):
        out, _ = _build(tmp_path)
        sidecar = [json.loads(l) for l in (tmp_path / "groundtruth_private" / "matcher_scores_sidecar.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
        tmp_path = out
        ids = set()
        for name in ("synthetic_positives.jsonl", "candidates_to_label.jsonl", "random_negatives.jsonl"):
            text = (tmp_path / name).read_text(encoding="utf-8")
            for line in text.splitlines():
                row = json.loads(line)
                ids.add(row["pair_id"])
                assert "score" not in row and "decision" not in row
        assert {s["pair_id"] for s in sidecar} == ids
        assert len(sidecar) == len(ids)
        assert all(s["decision"] in {"MATCH", "NO_MATCH"} and 0.0 <= s["score"] <= 1.0 for s in sidecar)
        sheet = (tmp_path / "label_sheet.csv").read_text(encoding="utf-8-sig")
        assert "score" not in sheet.lower().splitlines()[0]

    def test_label_sheet_format(self, tmp_path):
        tmp_path, _ = _build(tmp_path)
        with open(tmp_path / "label_sheet.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert list(rows[0]) == ["pair_id", "addr_a", "addr_b", "query_a", "query_b", "comuna_a", "comuna_b",
                                 "barrio_a", "barrio_b", "npn_a", "npn_b", "same_door", "same_unit",
                                 "evidence_source", "notes"]
        assert all(r["same_door"] == "" and r["same_unit"] == "" and r["evidence_source"] == "" and r["notes"] == ""
                   for r in rows)
        assert "stratum" not in rows[0] and "label" not in rows[0]
        cand = {c["pair_id"]: c for c in (json.loads(l) for l in
                (tmp_path / "candidates_to_label.jsonl").read_text(encoding="utf-8").splitlines())}
        assert {r["pair_id"] for r in rows} == set(cand) and len(rows) == len(cand)
        for r in rows:
            c = cand[r["pair_id"]]
            assert r["addr_a"] == c["addr_a"] and r["addr_b"] == c["addr_b"]
            assert r["query_a"] == c["addr_a"] + ", Cali, Valle del Cauca, Colombia"
            assert r["query_b"] == c["addr_b"] + ", Cali, Valle del Cauca, Colombia"
            assert (r["comuna_a"], r["comuna_b"]) == (c["comuna_a"], c["comuna_b"])
            assert (r["barrio_a"], r["barrio_b"]) == (c["barrio_a"], c["barrio_b"])
            assert (r["npn_a"], r["npn_b"]) == (c["npn_a"], c["npn_b"])

    def test_label_sheet_is_shuffled_deterministically_and_hides_strata(self, tmp_path):
        out1, _ = _build(tmp_path, seed=5, name="one")
        out2, _ = _build(tmp_path, seed=5, name="two")
        sheet1 = (out1 / "label_sheet.csv").read_bytes()
        assert sheet1 == (out2 / "label_sheet.csv").read_bytes()
        with open(out1 / "label_sheet.csv", newline="", encoding="utf-8") as fh:
            sheet_ids = [r["pair_id"] for r in csv.DictReader(fh)]
        cand = [json.loads(l) for l in (out1 / "candidates_to_label.jsonl").read_text(encoding="utf-8").splitlines()]
        assert sheet_ids != [c["pair_id"] for c in cand]  # not in generation (stratum) order
        # strata are not recoverable from the row order: no long run of one stratum
        stratum = {c["pair_id"]: c["stratum"] for c in cand}
        longest = run = 1
        for prev, cur in zip(sheet_ids, sheet_ids[1:]):
            run = run + 1 if stratum[prev] == stratum[cur] else 1
            longest = max(longest, run)
        assert longest < SIZES_SMALL["per_stratum"]
        assert "stratum" not in sheet1.decode("utf-8").splitlines()[0]

    def test_label_sheet_blank_barrio_and_comuna_are_empty_cells(self, tmp_path):
        recs = [gt.Record("KR 1 # 2 - 3", "N1", "m", None), gt.Record("KR 1 # 2 - 3 AP 1", "N2", "m", "02")]
        out, _ = _build(tmp_path, records=recs)
        with open(out / "label_sheet.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert {rows[0]["barrio_a"], rows[0]["barrio_b"]} == {""}
        assert {rows[0]["comuna_a"], rows[0]["comuna_b"]} == {"", "02"}

    def test_manifest_records_score_bins_per_stratum_and_tier(self, tmp_path):
        _, m = _build(tmp_path)
        by_stratum = m["candidate_score_bins_by_stratum"]
        assert set(by_stratum) == set(STRATA)
        for s in STRATA:
            assert list(by_stratum[s]) == list(gt.BIN_LABELS)
            assert sum(by_stratum[s].values()) == m["counts"]["candidates_by_stratum"][s]
        assert sum(m["synthetic_score_bins"].values()) == m["counts"]["synthetic"]
        assert sum(m["sanity_score_bins"].values()) == m["counts"]["sanity_negatives"]

    def test_pair_ids_unique_across_all_files(self, tmp_path):
        tmp_path, _ = _build(tmp_path)
        ids = []
        for name in ("synthetic_positives.jsonl", "candidates_to_label.jsonl", "random_negatives.jsonl"):
            ids += [json.loads(l)["pair_id"] for l in (tmp_path / name).read_text(encoding="utf-8").splitlines()]
        assert len(ids) == len(set(ids))

    def test_same_seed_is_byte_identical_and_other_seed_differs(self, tmp_path):
        d1, _ = _build(tmp_path, seed=5, name="a")
        d2, _ = _build(tmp_path, seed=5, name="b")
        d3, _ = _build(tmp_path, seed=6, name="c")
        names = sorted(p.name for p in d1.iterdir())
        assert names == sorted(p.name for p in d2.iterdir())
        for n in names:
            assert _read(d1 / n) == _read(d2 / n), n
        side = "matcher_scores_sidecar.jsonl"
        assert _read(tmp_path / "a_private" / side) == _read(tmp_path / "b_private" / side)
        assert _read(tmp_path / "a_private" / side) != _read(tmp_path / "c_private" / side)
        assert _read(d1 / "candidates_to_label.jsonl") != _read(d3 / "candidates_to_label.jsonl")
        assert _read(d1 / "MANIFEST.json") != _read(d3 / "MANIFEST.json")

    def test_manifest_hashes_match_files(self, tmp_path):
        import hashlib

        tmp_path, m = _build(tmp_path)
        for name, digest in m["sha256"].items():
            assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest

    def test_small_dataset_reports_shortfalls_and_does_not_crash(self, tmp_path):
        recs = [gt.Record("KR 1 # 2 - 3", "N1", "m", "01"), gt.Record("KR 1 # 2 - 3 AP 1", "N2", "m", "02")]
        tmp_path, m = _build(tmp_path, records=recs)
        assert m["shortfalls"]["candidates"]["plate_letter"]["shortfall"] == 6
        assert m["counts"]["candidates_by_stratum"]["plate_letter"] == 0
        assert (tmp_path / "label_sheet.csv").is_file()

    def test_empty_records(self, tmp_path):
        tmp_path, m = _build(tmp_path, records=[])
        assert m["parquet_rows"] == 0
        assert (tmp_path / "candidates_to_label.jsonl").read_text(encoding="utf-8") == ""

    def test_non_ascii_written_as_utf8(self, tmp_path):
        recs = make_records(3)
        tmp_path, _ = _build(tmp_path, records=recs)
        (tmp_path / "synthetic_positives.jsonl").read_bytes().decode("utf-8")


class TestCli:
    def test_main_uses_stubbed_loader_and_requires_catastro(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(gt, "load_catastro", lambda path: make_records())
        rc = gt.main(["--catastro", "whatever.parquet", "--out-dir", str(tmp_path / "gt"), "--seed", "2",
                      "--n-synthetic", "20", "--per-stratum", "4", "--n-sanity", "5"])
        assert rc == 0
        assert (tmp_path / "gt" / "MANIFEST.json").is_file()
        assert (tmp_path / "gt_private" / "matcher_scores_sidecar.jsonl").is_file()
        rc = gt.main(["--catastro", "whatever.parquet", "--out-dir", str(tmp_path / "g2"), "--private-dir",
                      str(tmp_path / "secret"), "--n-synthetic", "5", "--per-stratum", "2", "--n-sanity", "2"])
        assert rc == 0 and (tmp_path / "secret" / "matcher_scores_sidecar.jsonl").is_file()
        with pytest.raises(SystemExit):
            gt.main(["--out-dir", str(tmp_path)])

    def test_source_has_no_reference_to_other_project(self):
        text = (_ROOT / "tools" / "build_groundtruth.py").read_text(encoding="utf-8").lower()
        forbidden = ("normali" + "zador", "cali_" + "address")
        assert not any(word in text for word in forbidden)
