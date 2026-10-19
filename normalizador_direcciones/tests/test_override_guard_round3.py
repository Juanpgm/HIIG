"""Round 3 of the override guard: evaluation uses the selected candidate, the comma rule is
narrow, plate letters are not words, parse notes carry no cadastral text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_override_guard import NAN, make_normalizer

from cali_address.evaluate import score_predictions
from cali_address.inference import base_mismatches
from cali_address.parser import canonical, parse_address


def _mm(a, b):
    return base_mismatches(parse_address(a), parse_address(b))


def _multi_row(raw, cadastrals, fused_scores, threshold=0.73):
    n = make_normalizer(cadastrals, threshold=threshold)
    parsed = parse_address(raw)
    canon = canonical(parsed, style="spaced", with_complement=True)
    scored = [(f, f, 0.9, 0.9, i, NAN) for i, f in enumerate(fused_scores)]
    return n._row(raw, canon, parsed, scored, 20, threshold)


# ---------------------------------------------------------------------------
# R1: the evaluation judges the SELECTED candidate
# ---------------------------------------------------------------------------
def _score(rows, gt_doc_id, manzanas=("M0", "M1", "M2")):
    preds = pd.DataFrame(rows)
    evaluated = pd.DataFrame({
        "gt_doc_id": [gt_doc_id] * len(rows), "gt_manzana": [manzanas[gt_doc_id]] * len(rows),
        "gt_lat": [3.4] * len(rows), "gt_lon": [-76.5] * len(rows),
    })
    docs = pd.DataFrame({"manzana": list(manzanas)})
    return score_predictions(evaluated, preds, docs)


def test_evaluation_uses_the_selected_candidate_not_the_top_one():
    row = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48", "KR 987 # 654 - 84"], [0.99, 0.98])
    assert row["selected_doc_id"] == 1 and row["top5_doc_ids"][0] == 0
    out = _score([row], gt_doc_id=1)
    assert out["pred_doc_id"].iloc[0] == 1
    assert bool(out["correct_top1"].iloc[0]) is True
    assert out["pred_manzana"].iloc[0] == "M1" and bool(out["correct_manzana"].iloc[0]) is True


def test_a_rejected_row_is_not_judged_on_the_neighbour():
    row = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48", "KR 987 # 654 - 74"], [0.99, 0.98])
    assert row["selected_doc_id"] == -1
    out = _score([row], gt_doc_id=0)           # the rejected top-1 neighbour IS the ground truth
    assert out["pred_doc_id"].iloc[0] == -1
    assert bool(out["correct_top1"].iloc[0]) is False
    assert out["pred_manzana"].iloc[0] is None and bool(out["correct_manzana"].iloc[0]) is False
    assert np.isnan(out["dist_m"].iloc[0])
    assert bool(out["correct_top5"].iloc[0]) is True     # retrieval recall is not a prediction


def test_selected_doc_id_is_consistent_with_the_row_for_every_kind_of_row():
    accepted = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 84"], [0.95])
    low = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48"], [0.2])
    rejected = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48"], [0.99])
    assert accepted["selected_doc_id"] == 0 and accepted["numero_predial_nacional"] == "P0"
    assert low["selected_doc_id"] == low["top5_doc_ids"][0] == 0 and low["numero_predial_nacional"] == "P0"
    assert rejected["selected_doc_id"] == -1 and rejected["numero_predial_nacional"] is None
    assert accepted["selected_doc_id"] in accepted["top5_doc_ids"]


def test_old_prediction_frames_without_selected_doc_id_still_score_from_top5():
    preds = pd.DataFrame([{"top5_doc_ids": [2, 1], "topk_doc_ids": [2, 1], "lat": 3.4, "lon": -76.5}])
    out = _score(preds.to_dict("records"), gt_doc_id=2)
    assert bool(out["correct_top1"].iloc[0]) is True


def test_empty_row_carries_the_same_selection_column():
    n = make_normalizer(["KR 987 # 654 - 84"])
    out = n.normalize_batch(["SIN ESPECIFICAR", "KR 987 # 654 - 84"], use_model=False)
    assert out["selected_doc_id"].tolist() == [-1, 0]


# ---------------------------------------------------------------------------
# R2: the comma separates via and cross only when the head is a bare via
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Carrera 987, 654B-20", "KR 987 # 654 B - 20"),
    ("CL 987 B, 654 - 20", "CL 987 B # 654 - 20"),
    ("KR 987 BIS NORTE, 654 - 20", "KR 987 BIS NORTE # 654 - 20"),
    ("CL 987 NORTE, 8 B - 20 AP 3", "CL 987 NORTE # 8 B - 20 AP 3"),
    ("Carrera 987, 654 - 20, Navarro", "KR 987 # 654 - 20"),
])
def test_comma_after_a_bare_via_separates_cross_and_plate(raw, expected):
    assert canonical(parse_address(raw)) == expected


@pytest.mark.parametrize("raw, expected", [
    ("CL 987 10 20, 3 - 4", "CL 987 # 10 - 20"),                       # numbers after the via
    ("KR 987 CL 10 20 LOCAL 3, 4 - 5", "KR 987 CL # 10 - 20 LC 3"),    # more than a via in the head
    ("CL 987 SECTOR 3, 10 - 20", "CL 987 SECTOR 3"),                   # SECTOR word
    ("KR 987 LOCAL 3, 654 - 20", "KR 987 LC 3"),                       # LOCAL unit before the comma
    ("KR 987 AP 3, 654 - 20", "KR 987 AP # 3 -"),                      # unit before the comma
    ("CL 987 - 3, 8 B - 20", "CL 987 # 3"),                            # a plate separator in the head
    ("CL 987 - B, 654 - 20", "CL 987 B"),
    ("CL 987 -, 654 - 20", "CL 987"),                                  # a dangling dash in the head
    ("CL 987 # 654 - 20, Navarro", "CL 987 # 654 - 20"),               # '#' in the head
    ("CL 987 # 654 - 20, 3", "CL 987 # 654 - 20"),
    ("Calle 987, Navarro", "CL 987"),
    ("Calle 987, 3 piso", "CL 987"),                                   # tail has no plate separator
    ("Calle 987, 3 - ", "CL 987"),                                     # tail is not cross + plate
])
def test_comma_keeps_the_previous_behaviour_everywhere_else(raw, expected):
    assert canonical(parse_address(raw)) == expected


def test_comma_rule_keeps_the_plate_in_place():
    p = parse_address("CL 987 10 20, 3 - 4")
    assert p.plate == "20"


# ---------------------------------------------------------------------------
# R3: plate letters are single letters, BIS, glued letters or a quadrant; words are not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tail", [
    "LA ESPERANZA", "PI 2", "Y ", "CL", "CR", "CJ", "PH", "CA", "II", "LA", "EL", "DE LA PAZ", "SAN FERNANDO",
])
def test_words_after_the_plate_do_not_reject_against_a_bare_cadastre(tail):
    raw = f"KR 987 # 654 - 66 {tail}".strip()
    assert _mm(raw, "KR 987 # 654 - 66") == []
    assert _mm("KR 987 # 654 - 66", raw) == []


@pytest.mark.parametrize("raw", [
    "KR 987 # 654 - 66C", "KR 987 # 654 - 66 C", "KR 987 # 654 - 66 A", "KR 987 # 654 - 66 BIS",
    "KR 987 # 654 - 66AT", "KR 987 # 654 - 66 A4", "KR 987 # 654 - 66 NORTE",
])
def test_true_plate_letters_still_reject(raw):
    assert _mm(raw, "KR 987 # 654 - 66") != []


def test_a_glued_two_letter_run_is_a_plate_letter_but_a_separated_one_is_a_word():
    assert "plate_letters" in _mm("KR 987 # 654 - 66AT", "KR 987 # 654 - 66")
    assert _mm("KR 987 # 654 - 66 AT", "KR 987 # 654 - 66") == []


# ---------------------------------------------------------------------------
# R4: parse notes carry no cadastral text
# ---------------------------------------------------------------------------
def test_rejected_notes_hold_only_rank_and_field_names():
    neighbour = "KR 987 # 654 - 48"
    row = _multi_row("KR 987 # 654 - 84", [neighbour, "KR 987 # 654 - 74"], [0.99, 0.98])
    notes = row["parse_notes"].split(";")
    assert "cadastral_candidate_rejected:rank=1" in notes
    assert any(n.startswith("cadastral_base_mismatch:") and "plate" in n for n in notes)
    text = row["parse_notes"]
    for fragment in (neighbour, "654 - 48", "KR 987 #", "P0", "P1"):
        assert fragment not in text
    assert all(";" not in n for n in notes)


def test_selected_rank_note_is_a_bare_number():
    row = _multi_row("KR 987 # 654 - 84", ["KR 987 # 654 - 48", "KR 987 # 654 - 84"], [0.99, 0.98])
    assert "cadastral_candidate_rank:2" in row["parse_notes"].split(";")


# ---------------------------------------------------------------------------
# base_mismatches never raises
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("obj", [object(), 42, "CL 987 # 654 - 20", {}, [], type("X", (), {})()])
def test_base_mismatches_on_objects_without_parse_fields(obj):
    good = parse_address("CL 987 # 654 - 20")
    assert base_mismatches(obj, good) == ["unparsed"]
    assert base_mismatches(good, obj) == ["unparsed"]
    assert base_mismatches(obj, obj) == ["unparsed"]


def test_base_mismatches_on_a_partial_parse_object():
    class Partial:
        parse_ok = True
    assert base_mismatches(Partial(), parse_address("CL 987 # 654 - 20")) == ["unparsed"]
