"""Tests for ``tools/eval_realworld.py``.

Everything here uses tiny synthetic FAKE addresses and stubbed loaders: no
parquet file, no pyarrow and no real data is ever touched.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import eval_realworld as ev  # noqa: E402

# FAKE cadastre. Doc 6 is a legacy-alias homonym of doc 1 (same canonical).
CADASTRE = {
    1: "KR 26 # 73 - 10",
    2: "KR 26 # 73 - 12",
    3: "KR 26 # 73 - 11",
    4: "KR 26 # 73 - 10A",
    5: "KR 26 # 75 - 10",
    6: "K 26 # 73 - 10",
    7: "CL 5 # 10 - 20",
    8: "ZZZ",
    9: "CL 5 # 10 - 22",
}


def row(raw, doc_id=1, dataset="fake_a", quality="gold", split="train_pairs"):
    return ev.Row(raw=raw, dataset=dataset, doc_id=doc_id, quality=quality, split=split)


class FakeLoader:
    """Serves fixture tables and records every column request.

    Fixtures carry extra sensitive-looking columns; the loader only ever
    returns the requested ones, and the test asserts what was requested.
    """

    def __init__(self, tables):
        self.tables = tables
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, path, columns):
        self.calls.append((str(path), tuple(columns)))
        table = self.tables[str(path)]
        missing = [c for c in columns if c not in table]
        if missing:
            raise KeyError(missing)
        return {c: list(table[c]) for c in columns}


def make_tables():
    return {
        "train.parquet": {
            "raw_address": ["KR 26 # 73 - 10", "KR 26 # 73 - 12"],
            "dataset": ["fake_a", "fake_b"],
            "gt_doc_id": [1, 2],
            "gt_manzana": ["SECRET", "SECRET"],
            "gt_predial": ["SECRET", "SECRET"],
            "source_row_id": [0, 1],
        },
        "dev.parquet": {
            "raw_address": ["KR 26 # 73 - 11", "CL 5 # 10 - 20", "xx"],
            "dataset": ["fake_a", "fake_b", "fake_b"],
            "gt_doc_id": [3, 7, None],
            "gt_quality": ["gold", "noisy", "no_gt"],
            "their_normalized": ["SECRET"] * 3,
            "rule_canonical": ["KR 26 # 73 - 11", "CL 5 # 10 - 20", None],
            "canonical_address": ["KR 26 # 73 - 11", "CL 5 # 10 - 20", "ZZZ"],
            "gt_lat": [1.0, 2.0, 3.0],
            "idesc_estado": ["SECRET"] * 3,
        },
        "cad.parquet": {
            "doc_id": list(CADASTRE),
            "direccion": list(CADASTRE.values()),
            "numero_predial_nacional": ["SECRET"] * len(CADASTRE),
            "centroid_lat": [0.0] * len(CADASTRE),
        },
    }


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_zero_successes():
    lo, hi = ev.wilson(0, 50)
    assert lo == 0.0
    assert 0.0 < hi < 0.1


def test_wilson_all_successes():
    lo, hi = ev.wilson(50, 50)
    assert hi == 1.0
    assert 0.9 < lo < 1.0


def test_wilson_empty_is_none_and_formats_na():
    assert ev.wilson(0, 0) is None
    assert ev.fmt_rate(0, 0) == "n/a"


def test_wilson_midpoint_is_symmetric():
    lo, hi = ev.wilson(50, 100)
    assert lo == pytest.approx(1 - hi, abs=1e-9)
    assert lo < 0.5 < hi


def test_wilson_pinned_reference_values():
    # Reference values from the closed-form Wilson score interval (z = 1.96).
    # They pin the centre adjustment: a centre of plain p changes every one.
    lo, hi = ev.wilson(0, 234)
    assert lo == 0.0
    assert hi == pytest.approx(0.01615, abs=1e-5)
    lo, hi = ev.wilson(1, 10)
    assert lo == pytest.approx(0.0179, abs=1e-4)
    assert hi == pytest.approx(0.4042, abs=1e-4)
    lo, hi = ev.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)


def test_wilson_rejects_invalid_counts():
    with pytest.raises(ValueError):
        ev.wilson(5, 3)
    with pytest.raises(ValueError):
        ev.wilson(-1, 3)


# ---------------------------------------------------------------------------
# threshold boundary
# ---------------------------------------------------------------------------


def test_threshold_boundary_inclusive():
    assert ev.passes(0.90, 0.90)
    assert not ev.passes(0.8999, 0.90)
    assert ev.passes(1.0, 0.90)


def test_exact_match_hits_even_at_threshold_one():
    summary = ev.evaluate([row("KR 26 # 73 - 10", 1)], CADASTRE, seed=1, threshold=1.0)
    assert summary["positives"]["overall"]["hits"] == 1


# ---------------------------------------------------------------------------
# frozen guard
# ---------------------------------------------------------------------------


def test_frozen_path_refused_without_final(capsys):
    loader = FakeLoader(make_tables())
    code = ev.main(
        ["--train-pairs", "train.parquet", "--dev", "some/frozen_test.parquet", "--catastro", "cad.parquet"],
        loader=loader,
    )
    assert code == 2
    assert loader.calls == []  # nothing was read, not even the other files
    assert "frozen" in capsys.readouterr().err.lower()


def test_frozen_refusal_is_case_insensitive_and_covers_every_path(capsys):
    loader = FakeLoader(make_tables())
    for flag in ("--train-pairs", "--catastro"):
        argv = ["--train-pairs", "train.parquet", "--dev", "dev.parquet", "--catastro", "cad.parquet"]
        argv[argv.index(flag) + 1] = "X/FROZEN_Test.parquet"
        assert ev.main(argv, loader=loader) == 2
    assert loader.calls == []


def test_frozen_with_final_prints_burn_warning(capsys, tmp_path):
    tables = make_tables()
    tables["frozen_test.parquet"] = tables["dev.parquet"]
    loader = FakeLoader(tables)
    code = ev.main(
        [
            "--train-pairs", "train.parquet",
            "--dev", "frozen_test.parquet",
            "--catastro", "cad.parquet",
            "--final",
            "--out-dir", str(tmp_path / "out"),
        ],
        loader=loader,
    )
    err = capsys.readouterr().err
    assert code == 0
    assert "BURNED" in err
    assert any("frozen_test" in c[0] for c in loader.calls)


def test_final_flag_without_frozen_path_stays_quiet(capsys, tmp_path):
    loader = FakeLoader(make_tables())
    code = ev.main(
        ["--train-pairs", "train.parquet", "--dev", "dev.parquet", "--catastro", "cad.parquet",
         "--final", "--out-dir", str(tmp_path / "out")],
        loader=loader,
    )
    assert code == 0
    assert "BURNED" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# column restriction
# ---------------------------------------------------------------------------


def test_only_allowed_columns_are_requested(tmp_path):
    loader = FakeLoader(make_tables())
    code = ev.main(
        ["--train-pairs", "train.parquet", "--dev", "dev.parquet", "--catastro", "cad.parquet",
         "--out-dir", str(tmp_path / "out")],
        loader=loader,
    )
    assert code == 0
    assert loader.calls, "loader was never used"
    for _path, columns in loader.calls:
        assert columns, "explicit columns are mandatory"
        assert set(columns) <= ev.ALLOWED_COLUMNS
    requested = {c for _p, cols in loader.calls for c in cols}
    for forbidden in ("their_normalized", "gt_lat", "idesc_estado", "gt_manzana"):
        assert forbidden not in requested
    # The parcel key is read from the cadastre file ONLY.
    for path, columns in loader.calls:
        assert ("numero_predial_nacional" in columns) == (path == "cad.parquet"), path


def test_default_loader_refuses_disallowed_columns():
    with pytest.raises(ValueError):
        ev.guard_columns(["raw_address", "gt_lat"])
    for prediction_column in ("pred_doc_id", "matched", "confidence", "correct_top1", "idesc_estado"):
        with pytest.raises(ValueError):
            ev.guard_columns(["rule_canonical", prediction_column])
    with pytest.raises(ValueError):
        ev.guard_columns([])


# ---------------------------------------------------------------------------
# row loading
# ---------------------------------------------------------------------------


def test_load_rows_train_is_all_gold_and_dev_uses_quality():
    loader = FakeLoader(make_tables())
    train = ev.load_rows(loader, "train.parquet", "train_pairs")
    dev = ev.load_rows(loader, "dev.parquet", "dev")
    assert [r.quality for r in train] == ["gold", "gold"]
    assert [r.quality for r in dev] == ["gold", "noisy", "no_gt"]
    assert dev[2].doc_id is None


def test_load_rows_empty_split():
    loader = FakeLoader({"e.parquet": {"raw_address": [], "dataset": [], "gt_doc_id": []}})
    assert ev.load_rows(loader, "e.parquet", "train_pairs") == []


def test_load_cadastre_maps_ids():
    loader = FakeLoader(make_tables())
    assert ev.load_cadastre(loader, "cad.parquet") == CADASTRE


# ---------------------------------------------------------------------------
# evaluate: structure and edge cases
# ---------------------------------------------------------------------------


def test_empty_rows_do_not_crash_and_report_na():
    s = ev.evaluate([], CADASTRE, seed=1)
    assert s["counts"]["rows_total"] == 0
    assert s["positives"]["overall"]["n"] == 0
    assert s["positives"]["overall"]["recall"] is None
    assert s["hard_negatives"]["n"] == 0
    assert s["retrieval"]["n"] == 0
    assert s["noisy"]["n"] == 0


def test_gold_only_rows_have_no_noisy_section():
    s = ev.evaluate([row("KR 26 # 73 - 10", 1), row("CL 5 # 10 - 20", 7)], CADASTRE, seed=1)
    assert s["positives"]["overall"]["n"] == 2
    assert s["positives"]["overall"]["hits"] == 2
    assert s["noisy"]["n"] == 0
    assert s["counts"]["gold"] == 2


def test_all_noisy_rows_excluded_from_headline():
    rows = [row("KR 26 # 73 - 10", 1, quality="noisy"), row("KR 26 # 73 - 12", 1, quality="noisy")]
    s = ev.evaluate(rows, CADASTRE, seed=1)
    assert s["positives"]["overall"]["n"] == 0
    assert s["hard_negatives"]["n"] == 0
    assert s["retrieval"]["n"] == 0
    assert s["noisy"]["n"] == 2
    assert s["noisy"]["share_ge_threshold"] == pytest.approx(0.5)
    # second row's best match is a different door than the (neighbour) GT id 1
    assert s["noisy"]["best_same_predio"] == 1
    assert s["noisy"]["best_other_predio"] == 1


def test_no_gt_rows_are_counted_only():
    s = ev.evaluate([row("whatever", None, quality="no_gt")], CADASTRE, seed=1)
    assert s["counts"]["no_gt"] == 1
    assert s["positives"]["overall"]["n"] == 0


def test_missing_gt_doc_id_in_cadastre_is_skipped_and_counted():
    s = ev.evaluate([row("KR 26 # 73 - 10", 999), row("KR 26 # 73 - 10", None), row("KR 26 # 73 - 10", 1)],
                    CADASTRE, seed=1)
    assert s["counts"]["skipped_missing_cadastre"] == 1
    assert s["counts"]["skipped_no_doc_id"] == 1
    assert s["positives"]["overall"]["n"] == 1


def test_duplicate_rows_are_dropped_once_per_split():
    rows = [row("KR 26 # 73 - 10", 1), row("KR 26 # 73 - 10", 1), row("KR 26 # 73 - 10", 1, split="dev")]
    s = ev.evaluate(rows, CADASTRE, seed=1)
    assert s["counts"]["duplicates_dropped"] == 1
    assert s["positives"]["overall"]["n"] == 2
    assert s["positives"]["by_split"]["train_pairs"]["n"] == 1
    assert s["positives"]["by_split"]["dev"]["n"] == 1


@pytest.mark.parametrize("raw", [None, 12345, b"KR 26 # 73 - 10", "", "   ", "CALLE ÑÁÉ ### ☃", "ZZZ"])
def test_junk_raw_counts_as_miss_not_crash(raw):
    s = ev.evaluate([row(raw, 1)], CADASTRE, seed=1)
    pos = s["positives"]["overall"]
    assert pos["n"] == 1 and pos["hits"] == 0 and pos["parsed"] == 0
    assert pos["recall"] == 0.0
    assert sum(c for _r, c in pos["reasons"]) == 1
    assert s["retrieval"]["no_best"] == 1


def test_unicode_raw_that_parses_is_a_hit():
    s = ev.evaluate([row("  kr 26 # 73 – 10 ", 1)], CADASTRE, seed=1)
    assert s["positives"]["overall"]["hits"] == 1


def test_unparseable_cadastre_truth_is_a_labelled_miss():
    s = ev.evaluate([row("KR 26 # 73 - 10", 8)], CADASTRE, seed=1)
    assert s["positives"]["overall"]["hits"] == 0
    reasons = dict(s["positives"]["overall"]["reasons"])
    assert reasons == {"cadastre_unparseable": 1}


def test_score_histogram_sums_to_n():
    rows = [row("KR 26 # 73 - 10", 1), row("KR 26 # 73 - 12", 1), row("nonsense", 1)]
    hist = ev.evaluate(rows, CADASTRE, seed=1)["positives"]["overall"]["histogram"]
    assert sum(hist.values()) == 3
    assert hist["1.00"] == 1
    assert hist["0.00"] >= 1


def test_breakdowns_by_split_and_dataset():
    rows = [
        row("KR 26 # 73 - 10", 1, dataset="fake_a", split="train_pairs"),
        row("CL 5 # 10 - 20", 7, dataset="fake_b", split="dev"),
        row("junk", 7, dataset="fake_b", split="dev"),
    ]
    pos = ev.evaluate(rows, CADASTRE, seed=1)["positives"]
    assert pos["by_split"]["train_pairs"]["recall"] == 1.0
    assert pos["by_split"]["dev"]["hits"] == 1
    assert pos["by_dataset"]["fake_b"]["n"] == 2
    assert pos["by_split_dataset"]["dev/fake_b"]["n"] == 2


def test_reasons_are_families_only():
    s = ev.evaluate([row("ZZZ 1 2 3", 1)], CADASTRE, seed=1)
    for reason, _count in s["positives"]["overall"]["reasons"]:
        assert ":" not in reason or reason.startswith(("parse:", "veto:"))
        assert "ZZZ" not in reason


# ---------------------------------------------------------------------------
# hard negatives
# ---------------------------------------------------------------------------


def test_neighbours_never_include_truth_or_homonym():
    nb = ev.NeighbourIndex.from_cadastre(CADASTRE)
    for seed in range(30):
        sel = nb.select(1, random.Random(seed), k=2)
        ids = [doc for _kind, doc in sel.picks]
        assert 1 not in ids
        assert 6 not in ids  # legacy-alias homonym of doc 1
        assert 8 not in ids  # unparseable
        assert set(ids) <= {2, 3, 4, 5}
        assert len(ids) == len(set(ids))
        assert len({kind for kind, _doc in sel.picks}) == len(sel.picks)


def test_neighbour_kinds_are_correct():
    nb = ev.NeighbourIndex.from_cadastre(CADASTRE)
    kinds = {}
    for seed in range(60):
        for kind, doc in nb.select(1, random.Random(seed), k=3).picks:
            kinds[doc] = kind
    assert kinds[2] == kinds[3] == "adjacent_plate"
    assert kinds[4] == "plate_letter"
    assert kinds[5] == "same_plate_other_cross"


def test_homonyms_are_counted_not_picked():
    nb = ev.NeighbourIndex.from_cadastre(CADASTRE)
    assert nb.select(1, random.Random(0)).homonyms == 1
    assert nb.select(6, random.Random(0)).homonyms == 1
    assert nb.select(7, random.Random(0)).homonyms == 0


def test_neighbour_select_unknown_or_unparseable_truth_is_empty():
    nb = ev.NeighbourIndex.from_cadastre(CADASTRE)
    assert nb.select(999, random.Random(0)).picks == []
    assert nb.select(8, random.Random(0)).picks == []


def test_truth_without_plate_gets_no_plate_neighbours():
    cad = {1: "KR 26 # 73", 2: "KR 26 # 73 - 10", 3: "KR 26 # 75"}
    nb = ev.NeighbourIndex.from_cadastre(cad)
    assert nb.select(1, random.Random(0)).picks == []


def test_hard_negatives_report_by_kind_and_are_rejected():
    s = ev.evaluate([row("KR 26 # 73 - 10", 1)], CADASTRE, seed=3)
    hn = s["hard_negatives"]
    assert hn["n"] >= 1
    assert hn["false_matches"] == 0
    assert hn["homonyms_excluded"] == 1
    assert set(hn["by_kind"]) <= {"adjacent_plate", "plate_letter", "same_plate_other_cross"}
    assert sum(k["n"] for k in hn["by_kind"].values()) == hn["n"]


def test_hard_negative_false_match_is_detected(monkeypatch):
    from types import SimpleNamespace

    def always_match(a, b, threshold=0.9, tolerance=0, **_kwargs):
        return SimpleNamespace(is_match=True, score=1.0, fields=(), parsed_b=b, reason="")

    monkeypatch.setattr(ev, "match_parsed", always_match)
    s = ev.evaluate([row("KR 26 # 73 - 10", 1)], CADASTRE, seed=3)
    assert s["hard_negatives"]["false_matches"] == s["hard_negatives"]["n"] >= 1
    assert s["hard_negatives"]["false_match_rate"] == 1.0


def test_unparseable_raw_is_excluded_from_parseable_negative_denominator():
    s = ev.evaluate([row("junk", 1)], CADASTRE, seed=3)
    hn = s["hard_negatives"]
    assert hn["n"] >= 1
    assert hn["parseable_n"] == 0
    assert hn["parseable_rate"] is None


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_retrieval_homonym_counts_as_correct():
    r = ev.evaluate([row("KR 26 # 73 - 10", 1)], CADASTRE, seed=1)["retrieval"]
    assert r["n"] == 1
    assert r["top1_correct"] == 1
    assert r["top5_correct"] == 1
    assert r["false_best"] == 0
    assert r["no_best"] == 0


def test_retrieval_false_best_when_gt_points_elsewhere():
    r = ev.evaluate([row("CL 5 # 10 - 20", 5)], CADASTRE, seed=1)["retrieval"]
    assert r["false_best"] == 1
    assert r["top1_correct"] == 0


def test_retrieval_latency_reported():
    r = ev.evaluate([row("KR 26 # 73 - 10", 1)], CADASTRE, seed=1)["retrieval"]
    assert r["latency_ms"]["mean"] >= 0.0
    assert r["latency_ms"]["max"] >= r["latency_ms"]["p50"] >= 0.0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def _strip_latency(summary):
    s = json.loads(json.dumps(summary))
    s["retrieval"].pop("latency_ms", None)
    s["retrieval"].pop("index_build_s", None)
    return s


def test_same_seed_same_result():
    rows = [row("KR 26 # 73 - 10", 1), row("CL 5 # 10 - 20", 7), row("KR 26 # 75 - 10", 5)]
    a = _strip_latency(ev.evaluate(rows, CADASTRE, seed=7))
    b = _strip_latency(ev.evaluate(rows, CADASTRE, seed=7))
    assert a == b


# ---------------------------------------------------------------------------
# privacy: outputs and stdout
# ---------------------------------------------------------------------------


def test_outputs_only_under_out_dir_and_stdout_has_no_addresses(tmp_path, capsys):
    loader = FakeLoader(make_tables())
    work = tmp_path / "work"
    work.mkdir()
    out = work / "private"
    code = ev.main(
        ["--train-pairs", "train.parquet", "--dev", "dev.parquet", "--catastro", "cad.parquet",
         "--out-dir", str(out)],
        loader=loader,
    )
    assert code == 0
    written = {p.relative_to(work) for p in work.rglob("*") if p.is_file()}
    assert written
    assert all(p.parts[0] == "private" for p in written)
    stdout = capsys.readouterr().out
    for address in list(CADASTRE.values())[:7]:
        assert address not in stdout
    assert "gold rows" in stdout.lower()
    assert "not population precision" in stdout.lower()


def test_json_out_is_aggregate_only(tmp_path):
    loader = FakeLoader(make_tables())
    target = tmp_path / "summary.json"
    ev.main(
        ["--train-pairs", "train.parquet", "--dev", "dev.parquet", "--catastro", "cad.parquet",
         "--out-dir", str(tmp_path / "out"), "--json-out", str(target)],
        loader=loader,
    )
    text = target.read_text(encoding="utf-8")
    assert "private" not in json.loads(text)
    for address in CADASTRE.values():
        assert address not in text


def test_private_examples_capped_per_reason(tmp_path):
    rows = [row(f"junk {i}", 1) for i in range(12)]
    result = ev.evaluate(rows, CADASTRE, seed=1)
    for examples in result["private"]["miss_examples"].values():
        assert len(examples) <= ev.MAX_EXAMPLES


# ---------------------------------------------------------------------------
# committed sources hold no real data and no sibling-project references
# ---------------------------------------------------------------------------


def _committed_sources():
    return [_ROOT / "tools" / "eval_realworld.py", Path(__file__)]


def test_no_npn_like_numbers_in_committed_files():
    pattern = re.compile(r"(?<!\d)\d{30}(?!\d)")
    for path in _committed_sources():
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_no_sibling_project_references():
    forbidden = re.compile("cali_" + "address|normali" + "zador", re.IGNORECASE)
    for path in _committed_sources():
        assert not forbidden.search(path.read_text(encoding="utf-8")), path
