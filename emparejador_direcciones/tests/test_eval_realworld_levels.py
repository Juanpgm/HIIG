"""Tests for the ``--input`` / ``--level`` options of ``tools/eval_realworld.py``.

FAKE fixtures and stubbed loaders only: no parquet, no pyarrow, no real data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_eval_realworld import CADASTRE, FakeLoader, make_tables, row  # noqa: E402
from tools import eval_realworld as ev  # noqa: E402

# FAKE cadastre with several units in one building.
UNITS = {
    1: "KR 26 # 73 - 10 BG 101",
    2: "KR 26 # 73 - 10 BG 102",
    3: "KR 26 # 73 - 12",
    4: "CL 5 # 10 - 20",
}


def _argv(tmp_path, *extra, train=True):
    argv = ["--dev", "dev.parquet", "--catastro", "cad.parquet", "--out-dir", str(tmp_path / "out")]
    if train:
        argv += ["--train-pairs", "train.parquet"]
    return argv + list(extra)


# ---------------------------------------------------------------------------
# evaluate(level=...)
# ---------------------------------------------------------------------------


def test_evaluate_rejects_invalid_level():
    with pytest.raises(ValueError, match="level"):
        ev.evaluate([row("KR 26 # 73 - 10", 1)], CADASTRE, seed=1, level="floor")


def test_default_evaluate_level_is_unit_and_recorded_in_meta():
    assert ev.evaluate([], CADASTRE, seed=1)["meta"]["level"] == "unit"
    assert ev.evaluate([], CADASTRE, seed=1, level="building")["meta"]["level"] == "building"


def test_unit_level_misses_a_different_apartment_of_the_same_building():
    s = ev.evaluate([row("KR 26 # 73 - 10 BG 999", 1)], UNITS, seed=1, level="unit")
    assert s["positives"]["overall"]["hits"] == 0
    assert dict(s["positives"]["overall"]["reasons"]) == {"veto:complement": 1}


def test_building_level_hits_a_different_apartment_of_the_same_building():
    s = ev.evaluate([row("KR 26 # 73 - 10 BG 999", 1)], UNITS, seed=1, level="building")
    assert s["positives"]["overall"]["hits"] == 1
    assert s["positives"]["overall"]["reasons"] == []


def test_building_level_still_misses_another_plate():
    s = ev.evaluate([row("KR 26 # 73 - 12", 1)], UNITS, seed=1, level="building")
    assert s["positives"]["overall"]["hits"] == 0
    assert dict(s["positives"]["overall"]["reasons"]) == {"veto:plate": 1}


def test_retrieval_correctness_is_per_building_at_building_level_only():
    rows = [row("KR 26 # 73 - 10 BG 101", 2)]  # GT is the sibling apartment
    unit = ev.evaluate(rows, UNITS, seed=1, level="unit")["retrieval"]
    building = ev.evaluate(rows, UNITS, seed=1, level="building")["retrieval"]
    assert (unit["top1_correct"], unit["false_best"]) == (0, 1)
    assert (building["top1_correct"], building["false_best"]) == (1, 0)
    assert building["top5_correct"] == 1


def test_hard_negatives_are_still_rejected_at_building_level():
    s = ev.evaluate([row("KR 26 # 73 - 10 BG 101", 1)], UNITS, seed=3, level="building")
    hn = s["hard_negatives"]
    assert hn["n"] >= 1
    assert hn["false_matches"] == 0


def test_noisy_rows_follow_the_level():
    rows = [row("KR 26 # 73 - 10 BG 999", 1, quality="noisy")]
    assert ev.evaluate(rows, UNITS, seed=1, level="unit")["noisy"]["share_ge_threshold"] == 0.0
    assert ev.evaluate(rows, UNITS, seed=1, level="building")["noisy"]["share_ge_threshold"] == 1.0


# ---------------------------------------------------------------------------
# empty normalized values
# ---------------------------------------------------------------------------


def test_exclude_empty_counts_and_drops_empty_inputs():
    rows = [row(None, 1), row("", 1), row("   ", 1), row(b"KR 26 # 73 - 10", 1), row(7, 1),
            row("KR 26 # 73 - 10", 1)]
    s = ev.evaluate(rows, CADASTRE, seed=1, exclude_empty=True)
    assert s["counts"]["empty_input_excluded"] == 5
    assert s["counts"]["empty_input_excluded_gold"] == 5
    assert s["positives"]["overall"]["n"] == 1
    assert s["positives"]["overall"]["hits"] == 1


def test_empty_inputs_are_kept_as_misses_without_exclude_empty():
    s = ev.evaluate([row(None, 1), row("", 1)], CADASTRE, seed=1)
    assert s["counts"]["empty_input_excluded"] == 0
    assert s["positives"]["overall"]["n"] == 2
    assert s["positives"]["overall"]["hits"] == 0


def test_all_inputs_empty_does_not_crash():
    s = ev.evaluate([row(None, 1), row("", 1)], CADASTRE, seed=1, exclude_empty=True)
    assert s["positives"]["overall"]["n"] == 0
    assert s["positives"]["overall"]["recall"] is None
    assert s["retrieval"]["n"] == 0


# ---------------------------------------------------------------------------
# column selection
# ---------------------------------------------------------------------------


def test_load_rows_selects_the_text_column_and_requests_only_it():
    loader = FakeLoader(make_tables())
    dev = ev.load_rows(loader, "dev.parquet", "dev", text_column="rule_canonical")
    assert [r.raw for r in dev] == ["KR 26 # 73 - 11", "CL 5 # 10 - 20", None]
    assert loader.calls == [("dev.parquet", ("rule_canonical", "dataset", "gt_doc_id", "gt_quality"))]


@pytest.mark.parametrize("column", ["gt_lat", "pred_doc_id", "matched_cadastral_address", "", "idesc_estado"])
def test_load_rows_refuses_a_non_input_text_column(column):
    loader = FakeLoader(make_tables())
    with pytest.raises(ValueError):
        ev.load_rows(loader, "dev.parquet", "dev", text_column=column)
    assert loader.calls == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_normalized_mode_reads_only_dev_and_only_the_chosen_column(tmp_path):
    loader = FakeLoader(make_tables())
    assert ev.main(_argv(tmp_path, "--input", "normalized"), loader=loader) == 0
    assert "train.parquet" not in [p for p, _c in loader.calls]
    dev_columns = next(c for p, c in loader.calls if p == "dev.parquet")
    assert dev_columns == ("rule_canonical", "dataset", "gt_doc_id", "gt_quality")
    requested = {c for _p, cols in loader.calls for c in cols}
    assert not requested & {"raw_address", "their_normalized", "canonical_address"}


def test_normalized_mode_does_not_need_train_pairs(tmp_path):
    loader = FakeLoader(make_tables())
    assert ev.main(_argv(tmp_path, "--input", "normalized", train=False), loader=loader) == 0


def test_raw_mode_requires_train_pairs(tmp_path, capsys):
    loader = FakeLoader(make_tables())
    assert ev.main(_argv(tmp_path, train=False), loader=loader) == 2
    assert "train" in capsys.readouterr().err.lower()
    assert loader.calls == []


def test_normalized_column_choice_is_honoured(tmp_path):
    loader = FakeLoader(make_tables())
    args = _argv(tmp_path, "--input", "normalized", "--normalized-column", "canonical_address")
    assert ev.main(args, loader=loader) == 0
    assert next(c for p, c in loader.calls if p == "dev.parquet")[0] == "canonical_address"


def test_unknown_normalized_column_is_rejected_by_the_cli(tmp_path):
    with pytest.raises(SystemExit) as exc:
        ev.main(_argv(tmp_path, "--input", "normalized", "--normalized-column", "pred_doc_id"),
                loader=FakeLoader(make_tables()))
    assert exc.value.code == 2


def test_missing_normalized_column_is_a_clear_error(tmp_path, capsys):
    tables = make_tables()
    del tables["dev.parquet"]["rule_canonical"]
    code = ev.main(_argv(tmp_path, "--input", "normalized"), loader=FakeLoader(tables))
    assert code == 2
    assert "rule_canonical" in capsys.readouterr().err


def test_normalized_mode_reports_empty_inputs_and_dev_only_note(tmp_path, capsys):
    assert ev.main(_argv(tmp_path, "--input", "normalized"), loader=FakeLoader(make_tables())) == 0
    out = capsys.readouterr().out
    assert "input=normalized" in out
    assert "train_pairs has no normalized column" in out
    assert "empty_input_excluded=1" in out  # the third dev row has a None value
    assert "intervals are wide" in out  # only a handful of gold rows


def test_both_levels_are_reported_side_by_side(tmp_path, capsys):
    target = tmp_path / "summary.json"
    args = _argv(tmp_path, "--input", "normalized", "--json-out", str(target))
    assert ev.main(args, loader=FakeLoader(make_tables())) == 0
    out = capsys.readouterr().out
    assert "HEADLINE level=building" in out
    comparison = out.split("LEVEL COMPARISON", 1)[1]
    for level in ev.LEVELS:
        assert level in comparison
    data = json.loads(target.read_text(encoding="utf-8"))
    assert set(data["levels"]) == {"unit", "building"}
    assert data["levels"]["unit"]["meta"]["level"] == "unit"
    assert data["levels"]["building"]["meta"]["level"] == "building"
    assert data["meta"]["input"] == "normalized"
    assert "private" not in json.dumps(data)


def test_level_flag_only_changes_the_headline(tmp_path, capsys):
    assert ev.main(_argv(tmp_path, "--level", "unit"), loader=FakeLoader(make_tables())) == 0
    out = capsys.readouterr().out
    assert "HEADLINE level=unit" in out
    assert "LEVEL COMPARISON" in out


def test_invalid_level_flag_is_rejected_by_the_cli(tmp_path):
    with pytest.raises(SystemExit) as exc:
        ev.main(_argv(tmp_path, "--level", "floor"), loader=FakeLoader(make_tables()))
    assert exc.value.code == 2


def test_format_comparison_lists_both_levels_with_the_same_metrics():
    rows = [row("KR 26 # 73 - 10 BG 999", 1)]
    summaries = {lvl: ev.evaluate(rows, UNITS, seed=1, level=lvl) for lvl in ev.LEVELS}
    text = ev.format_comparison(summaries)
    lines = {lvl: [ln for ln in text.splitlines() if ln.strip().startswith(lvl)] for lvl in ev.LEVELS}
    assert all(len(v) >= 1 for v in lines.values())
    assert "recall=0.00%" in lines["unit"][0]  # recall 0/1
    assert "recall=100.00%" in lines["building"][0]  # recall 1/1
    for address in UNITS.values():
        assert address not in text


def test_normalized_run_writes_only_under_out_dir(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    assert ev.main(_argv(work, "--input", "normalized"), loader=FakeLoader(make_tables())) == 0
    written = {p.relative_to(work) for p in work.rglob("*") if p.is_file()}
    assert written and all(p.parts[0] == "out" for p in written)


# ---------------------------------------------------------------------------
# parcel key (numero_predial_nacional[:21]) and hard-negative hygiene
# ---------------------------------------------------------------------------

import random  # noqa: E402


def _npn(parcel_digit: str, unit: str) -> str:
    """FAKE 30-character parcel number: 21-character land parcel + 9-character unit."""
    return parcel_digit * 21 + unit * 9


def test_npn_column_is_allowed_only_for_the_cadastre_loader():
    assert "numero_predial_nacional" in ev.ALLOWED_COLUMNS
    loader = FakeLoader(make_tables())
    ev.load_rows(loader, "train.parquet", "train_pairs")
    ev.load_rows(loader, "dev.parquet", "dev")
    assert all("numero_predial_nacional" not in cols for _p, cols in loader.calls)
    with pytest.raises(ValueError):
        ev.load_rows(loader, "dev.parquet", "dev", text_column="numero_predial_nacional")


def test_load_cadastre_with_parcels_keeps_only_the_21_character_parcel_key():
    tables = make_tables()
    tables["cad.parquet"]["numero_predial_nacional"] = [_npn(str(i), "0") for i in range(len(CADASTRE))]
    loader = FakeLoader(tables)
    cadastre, parcels = ev.load_cadastre_with_parcels(loader, "cad.parquet")
    assert cadastre == CADASTRE
    assert loader.calls == [("cad.parquet", ("doc_id", "direccion", "numero_predial_nacional"))]
    assert parcels[1] == "0" * 21
    assert all(len(v) == 21 for v in parcels.values())


@pytest.mark.parametrize("bad", [None, 12345, "", "123", b"1" * 30, float("nan")])
def test_load_cadastre_with_parcels_ignores_malformed_npn(bad):
    tables = make_tables()
    tables["cad.parquet"]["numero_predial_nacional"] = [bad] * len(CADASTRE)
    _cadastre, parcels = ev.load_cadastre_with_parcels(FakeLoader(tables), "cad.parquet")
    assert parcels == {}


def test_main_reads_parcel_keys_and_never_writes_them(tmp_path):
    tables = make_tables()
    tables["cad.parquet"]["numero_predial_nacional"] = [_npn("7", "3")] * len(CADASTRE)
    assert ev.main(_argv(tmp_path), loader=FakeLoader(tables)) == 0
    files = [p for p in (tmp_path / "out").rglob("*") if p.is_file()]
    assert files
    for path in files:
        assert "7" * 21 not in path.read_text(encoding="utf-8")


SAME_BUILDING = {
    1: "KR 26 # 73 - 10",
    2: "KR 26 # 73 - 12",
    3: "KR 26 # 73 - 11",
}


def test_neighbours_in_the_same_land_parcel_as_the_truth_are_dropped_and_counted():
    parcels = {1: "A" * 21, 2: "A" * 21, 3: "B" * 21}
    with_parcels = ev.NeighbourIndex.from_cadastre(SAME_BUILDING, parcels)
    without = ev.NeighbourIndex.from_cadastre(SAME_BUILDING)
    picked = {d for seed in range(20) for _k, d in without.select(1, random.Random(seed), k=3).picks}
    assert picked == {2, 3}  # sanity: without the key, doc 2 is a candidate
    for seed in range(20):
        sel = with_parcels.select(1, random.Random(seed), k=3)
        assert [d for _k, d in sel.picks] == [3]
        assert sel.same_parcel == 1


def test_unknown_parcel_keys_do_not_filter_anything():
    sel = ev.NeighbourIndex.from_cadastre(SAME_BUILDING, {1: "A" * 21}).select(1, random.Random(0), k=3)
    assert len(sel.picks) >= 1
    assert sel.same_parcel == 0


def test_same_parcel_exclusions_are_reported_in_the_summary():
    parcels = {1: "A" * 21, 2: "A" * 21, 3: "B" * 21}
    s = ev.evaluate([row("KR 26 # 73 - 10", 1)], SAME_BUILDING, seed=1, parcels=parcels)
    assert s["hard_negatives"]["same_parcel_excluded"] == 1
    assert s["hard_negatives"]["n"] == 1


def test_label_conflict_is_not_counted_as_a_false_match():
    cad = {1: "KR 26 # 73 - 10", 2: "KR 26 # 73 - 12"}
    for level in ev.LEVELS:
        hn = ev.evaluate([row("KR 26 # 73 - 12", 1)], cad, seed=1, level=level)["hard_negatives"]
        assert hn["label_conflict"] == 1, level
        assert hn["false_matches"] == 0 and hn["n"] == 0, level


def test_label_conflict_at_building_level_uses_the_building_render():
    cad = {1: "KR 26 # 73 - 10 BG 101", 2: "KR 26 # 73 - 12 BG 5"}
    raw = "KR 26 # 73 - 12 BG 9"  # same building as the neighbour, other apartment
    unit = ev.evaluate([row(raw, 1)], cad, seed=1, level="unit")["hard_negatives"]
    building = ev.evaluate([row(raw, 1)], cad, seed=1, level="building")["hard_negatives"]
    assert (unit["label_conflict"], unit["n"], unit["false_matches"]) == (0, 1, 0)
    assert (building["label_conflict"], building["n"], building["false_matches"]) == (1, 0, 0)


def test_a_real_false_match_is_still_counted_when_the_label_is_not_in_conflict(monkeypatch):
    from types import SimpleNamespace

    def always_match(a, b, threshold=0.9, tolerance=0, **_kwargs):
        return SimpleNamespace(is_match=True, score=1.0, fields=(), parsed_b=b, reason="")

    monkeypatch.setattr(ev, "match_parsed", always_match)
    cad = {1: "KR 26 # 73 - 10", 2: "KR 26 # 73 - 12"}
    hn = ev.evaluate([row("KR 26 # 73 - 10", 1)], cad, seed=1)["hard_negatives"]
    assert (hn["n"], hn["false_matches"], hn["label_conflict"]) == (1, 1, 0)


def test_homonyms_are_counted_by_the_building_render_at_building_level():
    cad = {1: "KR 26 # 73 - 10 BG 1", 2: "KR 26 # 73 - 10 BG 2", 3: "KR 26 # 73 - 12"}
    nb = ev.NeighbourIndex.from_cadastre(cad)
    assert nb.select(1, random.Random(0), level="unit").homonyms == 0
    assert nb.select(1, random.Random(0), level="building").homonyms == 1
    assert nb.select(3, random.Random(0), level="building").homonyms == 0


def test_building_level_keeps_identifying_complements_when_counting_homonyms():
    cad = {1: "KR 26 # 73 - 10 LT 1", 2: "KR 26 # 73 - 10 LT 2"}
    nb = ev.NeighbourIndex.from_cadastre(cad)
    assert nb.select(1, random.Random(0), level="building").homonyms == 0


def test_neighbour_picks_never_share_the_truth_render_at_either_level():
    cad = {
        1: "KR 26 # 73 - 10 BG 1", 2: "KR 26 # 73 - 10 BG 2", 3: "K 26 # 73 - 10 AP 1",
        4: "KR 26 # 73 - 12", 5: "KR 26 # 73 - 10A", 6: "KR 26 # 75 - 10", 7: "KR 26 # 73 - 11 AP 1",
    }
    nb = ev.NeighbourIndex.from_cadastre(cad)
    for level in ev.LEVELS:
        for seed in range(30):
            for _kind, doc in nb.select(1, random.Random(seed), k=3, level=level).picks:
                assert doc not in (1, 2, 3), (level, doc)


# ---------------------------------------------------------------------------
# suggestions: circular column warning, private out-dir guard
# ---------------------------------------------------------------------------


def test_help_warns_that_canonical_address_is_circular(capsys):
    with pytest.raises(SystemExit):
        ev.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "canonical_address" in out and "circular" in out


def _inside_git_work_tree() -> bool:
    """True when the project sits in a git work tree (``check_out_dir`` needs ``git check-ignore``)."""
    try:
        done = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


@pytest.mark.skipif(not _inside_git_work_tree(), reason="needs a git work tree to evaluate .gitignore")
def test_check_out_dir_accepts_outside_the_repo_and_the_default_private_dir(tmp_path):
    assert ev.check_out_dir(tmp_path / "anything") is None
    assert ev.check_out_dir(ev.DEFAULT_OUT_DIR) is None
    assert ev.check_out_dir(_ROOT / "data" / "realworld_private" / "run1") is None


@pytest.mark.parametrize("inside", ["docs/out", "out_not_ignored", "tests", "src/emparejador/x", "."])
def test_check_out_dir_refuses_a_non_gitignored_dir_inside_the_repo(inside):
    message = ev.check_out_dir(_ROOT / inside)
    assert message is not None and "gitignored" in message


def test_check_out_dir_resolves_dot_dot_escapes():
    assert ev.check_out_dir(_ROOT / "data" / "realworld_private" / ".." / ".." / "docs") is not None


def test_main_refuses_an_unignored_repo_out_dir_before_reading_or_writing(tmp_path, capsys):
    loader = FakeLoader(make_tables())
    target = _ROOT / "zz_unignored_out_dir_for_test"
    assert not target.exists()
    code = ev.main(_argv(tmp_path, "--out-dir", str(target)), loader=loader)
    assert code == 2
    assert "gitignored" in capsys.readouterr().err
    assert loader.calls == []
    assert not target.exists()
