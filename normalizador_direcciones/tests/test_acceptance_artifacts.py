"""The written acceptance artifacts must agree with each other and with the leaderboard baseline row."""

from __future__ import annotations

import csv
import json
import os
import re

import pytest

from cali_address import retrain_compare as rc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "retrain_2026-10_acceptance.md")
THRESHOLDS = os.path.join(ROOT, "artifacts", "experiments", "retrain_2026_10_acceptance.json")
LEADERBOARD = os.path.join(ROOT, "artifacts", "experiments", "leaderboard.csv")

needs_artifacts = pytest.mark.skipif(
    not (os.path.isfile(THRESHOLDS) and os.path.isfile(LEADERBOARD)), reason="artifacts/ not present")


def test_the_document_has_a_timestamp_line_every_criterion_and_the_selection_rule():
    text = open(DOC, encoding="utf-8").read()
    assert re.search(r"^Written: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", text, re.M)
    for i in range(1, 13):
        assert re.search(rf"^\| A{i} \|", text, re.M), f"A{i} missing"
    assert "Selection rule" in text and "dev_gold_predial_precision" in text


def test_the_document_states_the_baseline_numbers_of_the_thresholds_defaults():
    text = open(DOC, encoding="utf-8").read()
    for token in ("289", "277", "0.9819494584837545", "0.8431001890359168", "| 96 |", "112"):
        assert token in text, token


def test_every_criterion_of_the_code_is_described_in_the_document():
    text = open(DOC, encoding="utf-8").read()
    for cid in rc.ACCEPTANCE_CRITERIA:
        assert f"| {cid} |" in text


@needs_artifacts
def test_the_json_matches_the_leaderboard_baseline_row_and_the_code_defaults():
    with open(LEADERBOARD, newline="", encoding="utf-8") as fh:
        rows = {r["tag"]: r for r in csv.DictReader(fh)}
    written = json.load(open(THRESHOLDS, encoding="utf-8"))
    noise = {t: rows[t] for t in ("s2b_retrain_s42", "s2b_retrain_s43")}
    fresh = rc.build_acceptance_thresholds(rows["baseline_prod"], created=written["created"], noise_rows=noise)
    assert written == fresh


@needs_artifacts
def test_the_json_has_no_relative_numbers_hard_coded():
    written = json.load(open(THRESHOLDS, encoding="utf-8"))
    text = json.dumps(written)
    for forbidden in ("frozen_gold_ok", "calib_error_cv", "confidence_curve", "violations_remaining"):
        assert forbidden not in written["baseline_reference"], forbidden
    assert set(written["relative_to_measured_baseline"]) == {"A1", "A2", "A3", "A4", "A5", "A6", "A10", "A12"} and text
    assert "dev_gold_predial_lot_precision" not in written["baseline_reference"]       # measured later, never stored
    assert written["constants"]["A7_construction_margin"] == 0.02 and written["constants"]["A7_loss_margin"] == 0.01


def test_the_document_records_the_lot_level_amendment_with_its_reasons():
    text = open(DOC, encoding="utf-8").read()
    m = re.search(r"^## Amendment 2026-10-31 \(before any training run\)$", text, re.M)
    assert m, "amendment heading missing"
    section = text[m.start():]
    for token in ("NPN[:21]", "A2", "A3", "A7", "npn_catastro", "2.0", "gt_predial", "unit suffix", "dev_gold_predial_lot_precision",
                  "gold_lot_missing_new", "oracle_ceiling", "rows_with_lot_in_source", "lot_absent_from_source",
                  "(c1)", "(c2)", "1.0"):
        assert token in section, token
    assert re.search(r"before any model was trained", section, re.I)


def test_the_criteria_table_states_the_lot_level_wording_and_no_catastro_dependency():
    text = open(DOC, encoding="utf-8").read()
    row = {i: re.search(rf"^\| A{i} \|.*$", text, re.M).group(0) for i in (2, 3, 7)}
    assert "dev_gold_predial_lot_precision" in row[2] and "lot" in row[3].lower()
    assert "npn_catastro" not in row[7] and "lot" in row[7].lower() and "(c1)" in row[7] and "(c2)" in row[7]
    for cid in ("A2", "A3", "A7"):
        assert "lot" in rc.ACCEPTANCE_CRITERIA[cid].lower()


def test_the_json_records_the_noise_band_from_the_leaderboard():
    written = json.load(open(THRESHOLDS, encoding="utf-8"))
    with open(LEADERBOARD, newline="", encoding="utf-8") as fh:
        rows = {r["tag"]: r for r in csv.DictReader(fh)}
    for tag in ("s2b_retrain_s42", "s2b_retrain_s43"):
        assert written["noise_band"][tag]["dev_lost_correct"] == int(float(rows[tag]["dev_lost_correct"]))


def test_the_document_records_the_second_amendment_decisions():
    text = open(DOC, encoding="utf-8").read()
    for token in ("measured baseline", "baseline_prod_2026_10", "reliability_<kind>_candidate.json", "ship rule against the refit",
                  "- 1.0 pp", "- 0.5 pp", "+ 0.5 pp", "dev_gold_predial_lot_precision", "noise band"):
        assert token in text, token
    assert "unreachable" in text and "emit a predial" in text.lower()
