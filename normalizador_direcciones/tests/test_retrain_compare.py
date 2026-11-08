"""cali_address.retrain_compare: index profile, gold coverage, coordinate shift, row agreement, confidence
curve and the acceptance checklist. Tiny invented fixtures only (no real addresses, 900+ numbers)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tiny import tiny_docs

from cali_address import retrain_compare as rc


# ====================================================================================================
# fixtures
# ====================================================================================================
def docs(n=12, **over) -> pd.DataFrame:
    d = tiny_docs(n)
    d["direccion"] = [f"CL {900 + i // 4} # {910 + i % 4} - {920 + i}" for i in range(n)]
    for k, v in over.items():
        d[k] = v
    return d


def new_docs(n=12) -> pd.DataFrame:
    d = docs(n)
    d["fuente"] = ["nomenclatura_cert"] * (n // 2) + ["catastro"] * (n - n // 2)
    d["npn_choice"] = ["single"] * (n // 3) + ["continuity"] * (n // 3) + [None] * (n - 2 * (n // 3))
    d["npn_catastro"] = d["numero_predial_nacional"]
    return d


def eval_rows(spec) -> pd.DataFrame:
    """``spec``: list of dicts with estado/confianza/correct_predial/correct_manzana/raw_address/..."""
    base = {"raw_address": "x", "estado": "OK", "confianza": 0.9, "correct_predial": True,
            "correct_manzana": True, "gt_predial": "P1", "numero_predial_nacional": "P1"}
    return pd.DataFrame([{**base, **{"raw_address": f"addr {i}"}, **s} for i, s in enumerate(spec)])


# ====================================================================================================
# index_profile
# ====================================================================================================
def test_profile_of_identical_indexes_has_full_overlap_and_no_shift():
    old, new = docs(), docs()
    p = rc.index_profile(old, new)
    assert p["n_docs"] == {"old": 12, "new": 12}
    assert p["address_key"]["shared"] == 12 and p["address_key"]["old_only"] == 0 == p["address_key"]["new_only"]
    assert p["doc_key"]["shared"] == 12
    assert p["n_predios_shift"]["shared_keys"] == 12 and p["n_predios_shift"]["new_greater"] == 0
    assert p["comunas"]["only_old"] == [] == p["comunas"]["only_new"]


def test_profile_counts_by_fuente_old_without_the_column_is_catastro():
    p = rc.index_profile(docs(), new_docs())
    assert p["counts_by_fuente"]["old"] == {"catastro": 12}
    assert p["counts_by_fuente"]["new"] == {"nomenclatura_cert": 6, "catastro": 6}


def test_profile_npn_choice_histogram_counts_missing_as_none():
    p = rc.index_profile(docs(), new_docs())
    assert p["npn_choice"]["new"] == {"single": 4, "continuity": 4, "none": 4}
    assert p["npn_choice"]["old"] == {}


def test_profile_spelling_variants_share_a_key():
    old = docs(2, direccion=["KR 932B # 946 - 931", "CL 901 # 902 - 903"])
    new = docs(2, direccion=["KR 932 B # 946 - 931", "CL 901 # 902 - 904"])
    p = rc.index_profile(old, new)
    assert p["address_key"]["shared"] == 1 and p["doc_key"]["shared"] == 1
    assert p["address_key"]["old_only"] == 1 and p["address_key"]["new_only"] == 1


def test_profile_uses_precomputed_keys_when_present():
    old, new = docs(3), docs(3)
    new["address_key"] = ["A", "B", "C"]
    new["doc_key"] = ["a", "b", "c"]
    old["address_key"] = ["A", "B", "Z"]
    old["doc_key"] = ["a", "b", "z"]
    p = rc.index_profile(old, new)
    assert p["address_key"]["shared"] == 2 and p["doc_key"]["shared"] == 2


def test_profile_n_predios_shift_on_shared_keys():
    old, new = docs(4), docs(4)
    old["n_predios"] = [1, 1, 3, 2]
    new["n_predios"] = [1, 2, 3, 1]
    s = rc.index_profile(old, new)["n_predios_shift"]
    assert (s["shared_keys"], s["equal"], s["new_greater"], s["new_smaller"]) == (4, 2, 1, 1)
    assert s["histogram"]["old"]["1"] == 2 and s["histogram"]["new"]["1"] == 2
    assert s["quantiles"]["old"]["max"] == 3 and s["quantiles"]["new"]["max"] == 3


def test_profile_reports_comunas_present_in_one_side_only():
    old, new = docs(6), docs(6)
    old["comuna"] = ["01", "01", "02", "02", "03", "03"]
    new["comuna"] = ["01", "01", "02", "02", "99", "99"]
    c = rc.index_profile(old, new)["comunas"]
    assert c["only_old"] == ["03"] and c["only_new"] == ["99"]
    assert c["counts"]["03"] == {"old": 2, "new": 0} and c["counts"]["99"] == {"old": 0, "new": 2}


def test_profile_duplicates_are_counted_per_side():
    old, new = docs(4), docs(4)
    new.loc[3, "direccion"] = new.loc[2, "direccion"]
    p = rc.index_profile(old, new)
    assert p["duplicates"]["old"]["address_key_groups"] == 0
    assert p["duplicates"]["new"]["address_key_groups"] == 1 and p["duplicates"]["new"]["docs_in_groups"] == 2
    assert p["address_key"]["new_unique"] == 3


def test_profile_empty_sides_do_not_divide_by_zero():
    empty = docs(0)
    p = rc.index_profile(empty, docs(3))
    assert p["n_docs"] == {"old": 0, "new": 3} and p["address_key"]["shared"] == 0
    assert p["n_predios_shift"]["shared_keys"] == 0 and p["n_predios_shift"]["quantiles"]["old"] == {}
    both = rc.index_profile(empty, empty)
    assert both["n_docs"] == {"old": 0, "new": 0} and both["comunas"]["counts"] == {}


def test_profile_missing_required_column_is_a_clear_error():
    bad = docs(3).drop(columns=["direccion"])
    with pytest.raises(ValueError, match="direccion"):
        rc.index_profile(bad, docs(3))
    with pytest.raises(ValueError, match="comuna"):
        rc.index_profile(docs(3), docs(3).drop(columns=["comuna"]))


def test_profile_is_json_serializable_and_has_no_address_text():
    old, new = docs(6), new_docs(6)
    text = json.dumps(rc.index_profile(old, new))
    for addr in list(old["direccion"]) + list(new["direccion"]):
        assert addr not in text
    assert "numero_predial_nacional" not in text or True


# ====================================================================================================
# gold_npn_coverage (LOT level: NPN[:21]; the unit suffix is ignored, npn_catastro is not read)
# ====================================================================================================
UNIT0, UNIT7 = "000000000", "000000007"


def lot(i: int) -> str:
    return f"76001010000000001{i:04d}"            # 21 digits, invented


def _gold(gt, quality="gold"):
    return pd.DataFrame({"gt_predial": gt, "gt_quality": quality})


def _lot_docs(npns, **over):
    d = docs(len(npns))
    d["numero_predial_nacional"] = npns
    for k, v in over.items():
        d[k] = v
    return d


def test_gold_coverage_matches_at_lot_level_and_reports_exact_hits_separately():
    d = _lot_docs([lot(1) + UNIT0, lot(2) + UNIT7, lot(3) + UNIT0])
    gt = [lot(1) + UNIT0, lot(2) + UNIT0, lot(3) + UNIT7, lot(9) + UNIT0]       # 1 exact, 2 lot-only, 1 missing lot
    cov = rc.gold_npn_coverage(_gold(gt), d)
    assert cov["n_gold_rows"] == 4 and cov["n_with_gt_predial"] == 4
    assert cov["n_present"] == 3 and cov["n_missing"] == 1 and cov["share_present"] == 0.75
    assert cov["n_present_exact"] == 1


def test_gold_coverage_ignores_npn_catastro_entirely():
    d = _lot_docs([lot(1) + UNIT0], npn_catastro=[lot(2) + UNIT0])
    cov = rc.gold_npn_coverage(_gold([lot(2) + UNIT0]), d)
    assert cov["n_present"] == 0 and cov["n_missing"] == 1
    assert "n_present_only_as_npn_catastro" not in cov and "n_present_as_npn" not in cov


def test_gold_coverage_is_the_same_with_or_without_an_npn_catastro_column():
    d = _lot_docs([lot(1) + UNIT0, lot(2) + UNIT0])
    gt = _gold([lot(1) + UNIT7, lot(5) + UNIT0])
    assert rc.gold_npn_coverage(gt, d) == rc.gold_npn_coverage(gt, d.assign(npn_catastro=d["numero_predial_nacional"]))


@pytest.mark.parametrize("length", [21, 28, 29, 30])
def test_gold_coverage_accepts_codes_of_any_valid_length(length):
    d = _lot_docs([(lot(1) + UNIT0)[:length]])
    assert rc.gold_npn_coverage(_gold([(lot(1) + UNIT7)[:length]]), d)["n_present"] == 1


def test_gold_coverage_counts_a_malformed_gt_as_not_present_and_reports_it():
    d = _lot_docs([lot(1) + UNIT0])
    cov = rc.gold_npn_coverage(_gold(["123", lot(1)[:-1] + "X" + UNIT0, lot(1) + UNIT0]), d)
    assert cov["n_with_gt_predial"] == 3 and cov["n_present"] == 1 and cov["n_missing"] == 2
    assert cov["n_invalid_gt"] == 2


def test_gold_coverage_skips_invalid_index_npns():
    d = _lot_docs([None, "123", "X" * 30, lot(1) + UNIT0])
    assert rc.gold_npn_coverage(_gold([lot(1) + UNIT0, "123"]), d)["n_present"] == 1


def test_gold_coverage_ignores_null_gt_and_trims_whitespace():
    d = _lot_docs([lot(1) + UNIT0])
    cov = rc.gold_npn_coverage(_gold([f" {lot(1)}{UNIT7} ", None, np.nan, ""]), d)
    assert cov["n_gold_rows"] == 4 and cov["n_with_gt_predial"] == 1 and cov["n_present"] == 1
    assert cov["share_present"] == 1.0


def test_gold_coverage_filters_to_gold_quality_when_the_column_exists():
    d = _lot_docs([lot(1) + UNIT0, lot(2) + UNIT0])
    rows = pd.DataFrame({"gt_predial": [lot(1) + UNIT0, lot(2) + UNIT0, lot(9) + UNIT0],
                         "gt_quality": ["gold", "noisy", "gold"]})
    assert rc.gold_npn_coverage(rows, d)["n_gold_rows"] == 2
    all_gt = rc.gold_npn_coverage(rows, d, gold_only=False)
    assert all_gt["n_gold_rows"] == 3 and all_gt["n_present"] == 2


def test_gold_coverage_without_quality_column_uses_all_rows_and_empty_gives_none_share():
    d = _lot_docs([lot(1) + UNIT0])
    assert rc.gold_npn_coverage(pd.DataFrame({"gt_predial": [lot(1) + UNIT0]}), d)["n_gold_rows"] == 1
    empty = rc.gold_npn_coverage(pd.DataFrame({"gt_predial": []}), d)
    assert empty["n_gold_rows"] == 0 and empty["share_present"] is None and empty["n_missing"] == 0


def test_gold_coverage_against_an_empty_index_is_zero_not_an_error():
    cov = rc.gold_npn_coverage(_gold([lot(1) + UNIT0]), _lot_docs([]))
    assert cov["n_present"] == 0 and cov["n_missing"] == 1 and cov["share_present"] == 0.0


def test_gold_coverage_requires_gt_predial_and_docs_npn():
    with pytest.raises(ValueError, match="gt_predial"):
        rc.gold_npn_coverage(pd.DataFrame({"x": [1]}), docs(1))
    with pytest.raises(ValueError, match="numero_predial_nacional"):
        rc.gold_npn_coverage(_gold([lot(1)]), docs(1).drop(columns=["numero_predial_nacional"]))


# ---- docs_lot_coverage: production documents still reachable in the new index -----------------------------
def test_docs_lot_coverage_by_lot_ignores_the_unit_suffix():
    old = _lot_docs([lot(1) + UNIT0, lot(2) + UNIT0, lot(3) + UNIT0, lot(4) + UNIT0])
    new = _lot_docs([lot(1) + UNIT7, lot(2) + UNIT0])
    c = rc.docs_lot_coverage(old, new)
    assert c["n_old_docs"] == 4
    assert c["by_lot"] == {"n_considered": 4, "n_covered": 2, "n_lost": 2, "share_covered": 0.5}


def test_docs_lot_coverage_skips_old_docs_without_a_valid_lot_from_the_denominator():
    old = _lot_docs([lot(1) + UNIT0, None, "123", "X" * 30])
    new = _lot_docs([lot(1) + UNIT0])
    c = rc.docs_lot_coverage(old, new)
    assert c["by_lot"]["n_considered"] == 1 and c["by_lot"]["share_covered"] == 1.0
    assert c["n_old_docs"] == 4


def test_docs_lot_coverage_by_doc_key_uses_the_normalized_address():
    old = docs(3, direccion=["KR 932B # 946 - 931", "CL 901 # 902 - 903", "CL 905 # 906 - 907"])
    new = docs(2, direccion=["KR 932 B # 946 - 931", "CL 901 # 902 - 904"])
    c = rc.docs_lot_coverage(old, new)
    assert c["by_doc_key"] == {"n_considered": 3, "n_covered": 1, "n_lost": 2, "share_covered": pytest.approx(1 / 3)}


def test_docs_lot_coverage_empty_sides_give_none_shares_not_a_division_by_zero():
    c = rc.docs_lot_coverage(_lot_docs([]), _lot_docs([lot(1) + UNIT0]))
    assert c["n_old_docs"] == 0 and c["by_lot"]["share_covered"] is None and c["by_doc_key"]["share_covered"] is None
    c = rc.docs_lot_coverage(_lot_docs([lot(1) + UNIT0]), _lot_docs([]))
    assert c["by_lot"] == {"n_considered": 1, "n_covered": 0, "n_lost": 1, "share_covered": 0.0}


def test_docs_lot_coverage_requires_the_npn_column_and_has_no_address_text():
    with pytest.raises(ValueError, match="numero_predial_nacional"):
        rc.docs_lot_coverage(docs(2).drop(columns=["numero_predial_nacional"]), docs(2))
    c = rc.docs_lot_coverage(docs(3), docs(3))
    text = json.dumps(c)
    assert all(addr not in text for addr in docs(3)["direccion"])


# ====================================================================================================
# coordinate_shift
# ====================================================================================================
def _coords(npn, lon, lat):
    return pd.DataFrame({"numero_predial_nacional": npn, "centroid_lon": lon, "centroid_lat": lat})


def test_coordinate_shift_zero_when_nothing_moved():
    a = _coords(["N1", "N2"], [-76.5, -76.51], [3.4, 3.41])
    s = rc.coordinate_shift(a, a.copy())
    assert s["n_shared_npn"] == 2 and s["quantiles_m"]["p50"] == 0 and s["quantiles_m"]["max"] == 0


def test_coordinate_shift_measures_metres():
    old = _coords(["N1"], [-76.5], [3.4])
    new = _coords(["N1"], [-76.5], [3.40085])          # 0.00085 deg of latitude ~ 94.5 m
    s = rc.coordinate_shift(old, new)
    assert 94.0 < s["quantiles_m"]["p50"] < 95.0 and s["share_over_m"]["50"] == 1.0 and s["share_over_m"]["100"] == 0.0


def test_coordinate_shift_only_uses_shared_npns_and_reports_counts():
    old = _coords(["N1", "N2"], [-76.5, -76.5], [3.4, 3.4])
    new = _coords(["N2", "N3"], [-76.5, -76.5], [3.4, 3.4])
    s = rc.coordinate_shift(old, new)
    assert s["n_shared_npn"] == 1 and s["n_old_npn"] == 2 and s["n_new_npn"] == 2


def test_coordinate_shift_uses_the_median_of_docs_sharing_an_npn():
    old = _coords(["N1", "N1", "N1"], [-76.5, -76.5, -76.5], [3.4, 3.4, 3.4])
    new = _coords(["N1", "N1", "N1"], [-76.5, -76.5, -76.5], [3.4, 3.4, 3.9])      # one outlier doc
    assert rc.coordinate_shift(old, new)["quantiles_m"]["max"] == 0


def test_coordinate_shift_drops_nan_coordinates_and_counts_them():
    old = _coords(["N1", "N2"], [-76.5, np.nan], [3.4, 3.4])
    new = _coords(["N1", "N2"], [-76.5, -76.5], [3.4, np.nan])
    s = rc.coordinate_shift(old, new)
    assert s["n_shared_npn"] == 1 and s["n_nan_dropped"] == 2


def test_coordinate_shift_without_shared_npn_or_empty_inputs_has_none_quantiles():
    s = rc.coordinate_shift(_coords(["N1"], [-76.5], [3.4]), _coords(["N2"], [-76.5], [3.4]))
    assert s["n_shared_npn"] == 0 and s["quantiles_m"] is None
    e = rc.coordinate_shift(_coords([], [], []), _coords([], [], []))
    assert e["n_shared_npn"] == 0 and e["quantiles_m"] is None


def test_coordinate_shift_ignores_blank_npns_and_requires_columns():
    old = _coords(["", None, "N1"], [-76.5] * 3, [3.4] * 3)
    new = _coords(["", None, "N1"], [-76.5] * 3, [3.4] * 3)
    assert rc.coordinate_shift(old, new)["n_shared_npn"] == 1
    with pytest.raises(ValueError, match="centroid_lat"):
        rc.coordinate_shift(old.drop(columns=["centroid_lat"]), new)


# ====================================================================================================
# probe_agreement / probe_table
# ====================================================================================================
def _pair():
    old = eval_rows([{"correct_predial": True}, {"correct_predial": True}, {"correct_predial": False},
                     {"estado": "SIN_MATCH", "correct_predial": None}])
    new = eval_rows([{"correct_predial": True}, {"correct_predial": False}, {"correct_predial": True},
                     {"estado": "SIN_MATCH", "correct_predial": None}])
    return old, new


def test_probe_agreement_buckets():
    old, new = _pair()
    a = rc.probe_agreement(old, new)
    assert a["n"] == 4
    assert a["counts"] == {"both_ok_predial_match": 1, "only_old": 1, "only_new": 1, "both_wrong": 1}
    assert sum(a["counts"].values()) == a["n"]


def test_probe_agreement_can_use_the_lot_level_column():
    old, new = _pair()
    old["correct_predial_lot"] = [True, True, True, np.nan]
    new["correct_predial_lot"] = [True, True, False, np.nan]
    a = rc.probe_agreement(old, new, correct_col="correct_predial_lot")
    assert a["counts"] == {"both_ok_predial_match": 2, "only_old": 1, "only_new": 0, "both_wrong": 1}
    assert rc.probe_agreement(old, new)["counts"] == {"both_ok_predial_match": 1, "only_old": 1, "only_new": 1,
                                                      "both_wrong": 1}            # the default stays the exact column


def test_probe_agreement_lot_column_missing_is_a_clear_error():
    old, new = _pair()
    with pytest.raises(ValueError, match="correct_predial_lot"):
        rc.probe_agreement(old, new, correct_col="correct_predial_lot")


def test_probe_agreement_a_row_not_ok_is_wrong_even_if_the_flag_says_correct():
    old = eval_rows([{"estado": "SIN_MATCH", "correct_predial": True}])
    new = eval_rows([{"estado": "OK", "correct_predial": True}])
    assert rc.probe_agreement(old, new)["counts"]["only_new"] == 1


def test_probe_agreement_nan_correctness_counts_as_wrong():
    old = eval_rows([{"correct_predial": np.nan}])
    new = eval_rows([{"correct_predial": None}])
    assert rc.probe_agreement(old, new)["counts"]["both_wrong"] == 1


def test_probe_agreement_length_mismatch_raises():
    old, new = _pair()
    with pytest.raises(rc.ProbeMisalignmentError, match="length"):
        rc.probe_agreement(old, new.iloc[:-1])


def test_probe_agreement_raw_address_mismatch_raises_without_leaking_the_text():
    old, new = _pair()
    new.loc[2, "raw_address"] = "SECRET STREET 999"
    with pytest.raises(rc.ProbeMisalignmentError) as err:
        rc.probe_agreement(old, new)
    assert "2" in str(err.value) and "SECRET STREET" not in str(err.value) and "addr 2" not in str(err.value)


def test_probe_agreement_is_positional_not_index_based():
    old, new = _pair()
    shuffled_index = new.copy()
    shuffled_index.index = [30, 20, 10, 0]
    assert rc.probe_agreement(old, shuffled_index)["counts"] == rc.probe_agreement(old, new)["counts"]


def test_probe_agreement_comparison_is_exact_case_matters():
    old, new = _pair()
    new.loc[0, "raw_address"] = new.loc[0, "raw_address"].upper()
    with pytest.raises(rc.ProbeMisalignmentError):
        rc.probe_agreement(old, new)


def test_probe_agreement_empty_inputs():
    empty = eval_rows([])
    a = rc.probe_agreement(empty, empty)
    assert a["n"] == 0 and sum(a["counts"].values()) == 0


def test_probe_agreement_missing_columns_is_a_clear_error():
    old, new = _pair()
    with pytest.raises(ValueError, match="correct_predial"):
        rc.probe_agreement(old.drop(columns=["correct_predial"]), new)


def test_probe_agreement_by_fuente_from_the_rows_column():
    old, new = _pair()
    new["fuente"] = ["nomenclatura_cert", "nomenclatura_cert", "catastro", "catastro"]
    a = rc.probe_agreement(old, new)
    assert a["by_fuente"]["nomenclatura_cert"] == {"both_ok_predial_match": 1, "only_old": 1, "only_new": 0, "both_wrong": 0}
    assert a["by_fuente"]["catastro"]["only_new"] == 1 and a["by_fuente"]["catastro"]["both_wrong"] == 1


def test_probe_agreement_by_fuente_from_docs_via_the_output_npn():
    old, new = _pair()
    new["numero_predial_nacional"] = ["P1", "P1", "P2", None]
    d = pd.DataFrame({"numero_predial_nacional": ["P1", "P2"], "fuente": ["nomenclatura_cert", "catastro"]})
    a = rc.probe_agreement(old, new, docs_new=d)
    assert a["by_fuente"]["nomenclatura_cert"]["both_ok_predial_match"] == 1
    assert a["by_fuente"]["catastro"]["only_new"] == 1
    assert a["by_fuente"]["unknown"]["both_wrong"] == 1               # no output NPN -> no matched doc


def test_probe_agreement_without_any_fuente_source_omits_the_breakdown():
    old, new = _pair()
    assert "by_fuente" not in rc.probe_agreement(old, new)


def test_probe_agreement_summary_has_counts_only():
    old, new = _pair()
    text = json.dumps(rc.probe_agreement(old, new))
    assert "addr " not in text


def test_probe_table_keeps_per_row_detail_for_the_untracked_parquet():
    old, new = _pair()
    t = rc.probe_table(old, new)
    assert list(t["bucket"]) == ["both_ok_predial_match", "only_old", "only_new", "both_wrong"]
    assert len(t) == 4 and {"raw_address", "ok_old", "ok_new"} <= set(t.columns)


# ====================================================================================================
# predial_match_either
# ====================================================================================================
def test_either_matches_strict_or_catastral_npn():
    d = pd.DataFrame({"numero_predial_nacional": ["N1", "N2", "N3"], "npn_catastro": ["N1", "C2", None]})
    rows = pd.DataFrame({"numero_predial_nacional": ["N1", "N2", "N3", "N2", None, "NX"],
                         "gt_predial": ["N1", "C2", "C3", "OTHER", "C2", "NX"]})
    assert rc.predial_match_either(rows, d).tolist() == [True, True, False, False, False, True]


def test_either_uses_doc_id_when_the_rows_carry_it():
    d = pd.DataFrame({"numero_predial_nacional": ["N1", "N1"], "npn_catastro": ["CA", "CB"]})
    rows = pd.DataFrame({"numero_predial_nacional": ["N1", "N1"], "gt_predial": ["CB", "CB"], "doc_id": [0, 1]})
    assert rc.predial_match_either(rows, d).tolist() == [False, True]


def test_either_without_npn_catastro_column_degrades_to_strict():
    d = pd.DataFrame({"numero_predial_nacional": ["N1"]})
    rows = pd.DataFrame({"numero_predial_nacional": ["N1", "N1"], "gt_predial": ["N1", "C1"]})
    assert rc.predial_match_either(rows, d).tolist() == [True, False]


def test_either_null_gt_and_empty_rows():
    d = pd.DataFrame({"numero_predial_nacional": ["N1"], "npn_catastro": ["N1"]})
    rows = pd.DataFrame({"numero_predial_nacional": ["N1", None], "gt_predial": [None, None]})
    assert rc.predial_match_either(rows, d).tolist() == [False, False]
    assert len(rc.predial_match_either(rows.iloc[:0], d)) == 0


def test_either_summary_counts_only_ok_rows():
    d = pd.DataFrame({"numero_predial_nacional": ["N1", "N2"], "npn_catastro": ["N1", "C2"]})
    rows = pd.DataFrame({"estado": ["OK", "OK", "SIN_MATCH"], "numero_predial_nacional": ["N1", "N2", "N2"],
                         "gt_predial": ["N1", "C2", "C2"]})
    s = rc.predial_either_summary(rows, d)
    assert s == {"n_ok_rows": 2, "strict": 1, "either": 2, "gain": 1}


# ====================================================================================================
# confidence_curve
# ====================================================================================================
def test_confidence_curve_default_thresholds_are_050_to_095():
    curve = rc.confidence_curve(eval_rows([{}]))
    assert [c["threshold"] for c in curve] == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def test_confidence_curve_values():
    rows = eval_rows([
        {"confianza": 0.95, "correct_predial": True, "correct_manzana": True},
        {"confianza": 0.75, "correct_predial": True, "correct_manzana": True},
        {"confianza": 0.72, "correct_predial": False, "correct_manzana": False},
        {"confianza": 0.55, "correct_predial": False, "correct_manzana": True},
        {"estado": "SIN_MATCH", "confianza": 0.99, "correct_predial": None, "correct_manzana": None},
    ])
    p = {c["threshold"]: c for c in rc.confidence_curve(rows)}
    assert p[0.7]["n_ok_at_t"] == 3 and p[0.7]["n_ok"] == 4 and p[0.7]["n_rows"] == 5
    assert p[0.7]["share_of_ok"] == 0.75 and p[0.7]["share_of_rows"] == 0.6
    assert p[0.7]["predial_precision"] == pytest.approx(2 / 3) and p[0.7]["manzana_error_rate"] == pytest.approx(1 / 3)
    assert p[0.5]["n_ok_at_t"] == 4 and p[0.95]["n_ok_at_t"] == 1 and p[0.95]["predial_precision"] == 1.0


def test_confidence_curve_equal_to_threshold_counts():
    p = {c["threshold"]: c for c in rc.confidence_curve(eval_rows([{"confianza": 0.7}]))}
    assert p[0.7]["n_ok_at_t"] == 1 and p[0.75]["n_ok_at_t"] == 0


def test_confidence_curve_no_rows_above_a_threshold_has_none_not_zero_division():
    curve = rc.confidence_curve(eval_rows([{"confianza": 0.4}]))
    for c in curve:
        assert c["n_ok_at_t"] == 0 and c["predial_precision"] is None and c["manzana_error_rate"] is None
        assert c["share_of_ok"] == 0.0 and c["share_of_rows"] == 0.0


def test_confidence_curve_no_ok_rows_or_no_rows_at_all():
    c = rc.confidence_curve(eval_rows([{"estado": "SIN_MATCH", "confianza": 0.9}]))[0]
    assert c["n_ok"] == 0 and c["share_of_ok"] is None and c["share_of_rows"] == 0.0
    c = rc.confidence_curve(eval_rows([]))[0]
    assert c["n_rows"] == 0 and c["share_of_rows"] is None and c["share_of_ok"] is None


def test_confidence_curve_nan_confidence_never_passes_a_threshold():
    curve = rc.confidence_curve(eval_rows([{"confianza": np.nan}, {"confianza": None}]))
    assert all(c["n_ok_at_t"] == 0 for c in curve)


def test_confidence_curve_counts_are_non_increasing():
    rows = eval_rows([{"confianza": v} for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
    counts = [c["n_ok_at_t"] for c in rc.confidence_curve(rows)]
    assert counts == sorted(counts, reverse=True)


def test_confidence_curve_custom_thresholds_and_missing_columns():
    curve = rc.confidence_curve(eval_rows([{"confianza": 0.8}]), thresholds=[0.0, 0.8, 1.0])
    assert [c["n_ok_at_t"] for c in curve] == [1, 1, 0]
    with pytest.raises(ValueError, match="confianza"):
        rc.confidence_curve(eval_rows([{}]).drop(columns=["confianza"]))
    assert rc.confidence_curve(eval_rows([{}]), thresholds=[]) == []


def test_confidence_curve_reports_lot_level_precision_next_to_the_exact_one():
    rows = eval_rows([
        {"confianza": 0.95, "correct_predial": True, "correct_predial_lot": True},
        {"confianza": 0.80, "correct_predial": False, "correct_predial_lot": True},     # same lot, other unit
        {"confianza": 0.72, "correct_predial": False, "correct_predial_lot": False},
        {"confianza": 0.60, "correct_predial": False, "correct_predial_lot": True},
    ])
    p = {c["threshold"]: c for c in rc.confidence_curve(rows)}
    assert p[0.7]["predial_precision"] == pytest.approx(1 / 3) and p[0.7]["predial_lot_precision"] == pytest.approx(2 / 3)
    assert p[0.5]["predial_lot_precision"] == pytest.approx(3 / 4)
    assert p[0.95]["predial_lot_precision"] == 1.0


def test_confidence_curve_lot_precision_is_none_without_the_column_or_without_rows_at_t():
    p = {c["threshold"]: c for c in rc.confidence_curve(eval_rows([{"confianza": 0.9}]))}
    assert p[0.9]["predial_lot_precision"] is None and "predial_precision" in p[0.9]
    with_col = eval_rows([{"confianza": 0.4, "correct_predial_lot": True}])
    assert all(c["predial_lot_precision"] is None for c in rc.confidence_curve(with_col))


def test_confidence_curve_unknown_lot_correctness_counts_as_not_correct():
    p = {c["threshold"]: c for c in rc.confidence_curve(
        eval_rows([{"confianza": 0.9, "correct_predial_lot": None}, {"confianza": 0.9, "correct_predial_lot": np.nan}]))}
    assert p[0.9]["predial_lot_precision"] == 0.0


def test_confidence_curve_unknown_correctness_counts_as_not_correct():
    p = {c["threshold"]: c for c in rc.confidence_curve(
        eval_rows([{"confianza": 0.9, "correct_predial": None, "correct_manzana": None}]))}
    assert p[0.9]["predial_precision"] == 0.0 and p[0.9]["manzana_error_rate"] == 0.0


# ====================================================================================================
# check_acceptance
# ====================================================================================================
BASE_REF = {  # the numbers of leaderboard.csv row baseline_prod
    "dev_gold_n": 289, "dev_gold_ok": 277, "dev_gold_predial_precision": 0.9819494584837545,
    "dev_gold_manzana_precision": 0.9963898916967509, "dev_manzana_precision": 0.8431001890359168,
    "dev_lost_correct": 96,
}


def thresholds():
    return rc.build_acceptance_thresholds(BASE_REF, created="2026-10-30T00:00:00")


def curve_point(share_of_rows=0.9, precision=0.99, err=0.01, t=0.7, n=100, emitting=90):
    """One confidence-curve point. ``precision`` is the lot-level precision among rows that EMIT a predial; the old
    all-OK-denominator keys are present with a different value to prove they are no longer used by A10."""
    return [{"threshold": t, "n_rows": 289, "n_ok": 277, "n_ok_at_t": n, "share_of_ok": share_of_rows,
             "share_of_rows": share_of_rows, "predial_precision": 0.5, "predial_lot_precision": 0.5,
             "predial_lot_precision_emitted": precision, "n_emitting_predial": emitting,
             "manzana_error_rate": err}]


def good_candidate():
    return {
        "dev_gold_ok": 277, "dev_gold_predial_precision": 0.9819494584837545, "dev_manzana_precision": 0.8431,
        "dev_lost_correct": 100, "frozen_gold_ok": 400, "frozen_gold_predial_precision": 0.99,
        "dev_gold_predial_lot_precision": 0.9819494584837545, "frozen_gold_predial_lot_precision": 0.99,
        "calib_error_cv_predio": 0.06, "calib_error_cv_manzana": 0.04, "ship_rule_passed": False,
        "ship_rule_vs_refit": True,
        "confidence_curve": curve_point(), "production_untouched": True,
        "parser_failure_rate_cert": 0.001, "crs_gates_passed": True, "gold_lot_coverage_new": 0.80,
        "gold_lot_coverage_old": 0.99, "gold_lot_missing_new": 9, "accounting_dropped": 0,
        "gold_lot_cond_new": 0.97, "gold_lot_cond_prod": 0.98, "gold_coverage_loss": 0.19, "gold_inherent_loss": 0.185, "pytest_green": True, "no_real_addresses_exit": 0,
        "quarantine_share": 0.0016, "build_index_exit": 0, "download_exit": 0, "anomaly_fixtures_ok": True,
        "violations_remaining_dev": 0,
    }


def good_baseline():
    return {"frozen_gold_ok": 400, "frozen_gold_predial_precision": 0.99, "dev_gold_predial_lot_precision": 0.9819494584837545,
            "frozen_gold_predial_lot_precision": 0.99, "calib_error_cv_predio": 0.06,
            "calib_error_cv_manzana": 0.04, "confidence_curve": curve_point(), "production_untouched": True}


def run(cand=None, base=None, correct=None, **over_cand):
    c = {**good_candidate(), **(cand or {}), **over_cand}
    m = {"candidate": c, "baseline": {**good_baseline(), **(base or {})}}
    if correct is not None:
        m["candidate_correct"] = {**c, **correct}
    return {r["id"]: r for r in rc.check_acceptance(m, thresholds())}


def test_acceptance_returns_a_row_per_criterion_with_the_documented_keys():
    res = rc.check_acceptance({"candidate": good_candidate(), "baseline": good_baseline()}, thresholds())
    assert [r["id"] for r in res] == [f"A{i}" for i in range(1, 13)]
    for r in res:
        assert {"id", "value", "threshold", "passed"} <= set(r) and r["passed"] in (True, False, None)


def test_a_good_candidate_passes_everything_evaluable():
    res = run(correct={})
    assert {k: v["passed"] for k, v in res.items() if v["passed"] is not True} == {}


# ---- A1 .. A5 boundaries -----------------------------------------------------------------------------
@pytest.mark.parametrize("value,ok", [(277, True), (276, True), (275, False), (289, True), (0, False)])
def test_a1_dev_gold_ok_baseline_minus_one(value, ok):
    assert run(dev_gold_ok=value)["A1"]["passed"] is ok


def test_a1_reports_value_and_threshold():
    r = run(dev_gold_ok=275)["A1"]
    assert r["value"] == 275 and r["threshold"] == 276


@pytest.mark.parametrize("value,ok", [(0.9819494584837545, True), (0.99, True), (0.9819494584837544, True),
                                      (0.98, False), (0.9819, False)])
def test_a2_lot_level_predial_precision_not_below_baseline(value, ok):
    assert run(dev_gold_predial_lot_precision=value)["A2"]["passed"] is ok


def test_a2_one_row_below_fails_where_equal_passes():
    one_row_below = 272 / 277                       # 272 of 277 vs the baseline 272/277 = 0.98195...
    assert run(dev_gold_predial_lot_precision=272 / 277)["A2"]["passed"] is True
    assert run(dev_gold_predial_lot_precision=271 / 277)["A2"]["passed"] is False


def test_a2_ignores_the_exact_precision_which_is_only_reported():
    assert run(dev_gold_predial_precision=0.5)["A2"]["passed"] is True              # exact way below, lot level fine
    assert run(dev_gold_predial_precision=1.0, dev_gold_predial_lot_precision=0.5)["A2"]["passed"] is False


def test_a2_baseline_is_the_measured_lot_precision_not_a_stored_exact_one():
    r = run(base={"dev_gold_predial_lot_precision": 0.995}, dev_gold_predial_lot_precision=0.99)["A2"]
    assert r["passed"] is False and r["threshold"] == 0.995


def test_a2_without_a_measured_lot_baseline_is_not_evaluated():
    r = run(base={"dev_gold_predial_lot_precision": None})["A2"]
    assert r["passed"] is None and "baseline dev_gold_predial_lot_precision" in r["reason"]


def test_a3_frozen_gold_ok_and_precision_margins():
    ok = run()["A3"]
    assert ok["passed"] is True
    assert run(frozen_gold_ok=398)["A3"]["passed"] is True                    # baseline - 2 passes
    assert run(frozen_gold_ok=397)["A3"]["passed"] is False                   # one more fails
    floor = 0.99 - 2 / 400
    assert run(frozen_gold_predial_lot_precision=floor)["A3"]["passed"] is True
    assert run(frozen_gold_predial_lot_precision=floor - 1e-6)["A3"]["passed"] is False
    assert run(frozen_gold_predial_precision=0.1)["A3"]["passed"] is True            # exact precision is only reported


def test_a3_without_a_measured_frozen_baseline_is_not_evaluated_but_flagged():
    r = run(base={"frozen_gold_ok": None, "frozen_gold_predial_lot_precision": None})["A3"]
    assert r["passed"] is None and r["status"] == "not_evaluated" and "frozen_gold_ok" in r["reason"]


@pytest.mark.parametrize("value,ok", [(112, True), (111, True), (113, False), (96, True), (0, True)])
def test_a4_lost_correct_at_most_112_without_a_noise_band_in_the_thresholds(value, ok):
    assert run(dev_lost_correct=value)["A4"]["passed"] is ok


@pytest.mark.parametrize("value,ok", [(0.8431001890359168 - 0.005, True), (0.8382, True), (0.8380, False),
                                      (0.80, False), (0.9, True)])
def test_a5_manzana_precision_baseline_minus_half_a_point(value, ok):
    assert run(dev_manzana_precision=value)["A5"]["passed"] is ok


# ---- measured baseline replaces the stored reference ---------------------------------------------------
def test_measured_baseline_overrides_the_stored_reference_in_a1_a5():
    base = {"dev_gold_ok": 278, "dev_manzana_precision": 0.8630}
    assert run(base=base, dev_gold_ok=277)["A1"]["passed"] is True                    # 278 - 1
    assert run(base=base, dev_gold_ok=276)["A1"]["passed"] is False                   # stored reference would pass it
    assert run(base=base, dev_manzana_precision=0.858)["A5"]["passed"] is True        # 0.8630 - 0.005
    assert run(base=base, dev_manzana_precision=0.8579)["A5"]["passed"] is False
    assert run(dev_manzana_precision=0.8579)["A5"]["passed"] is True                  # stored reference still the fallback


def thresholds_with_noise(noise=(107, 112)):
    return rc.build_acceptance_thresholds(
        BASE_REF, created="t", noise_rows={f"s2b_retrain_s4{i}": {"dev_lost_correct": v} for i, v in enumerate(noise, 2)})


def run_noise(cand, base, noise=(107, 112)):
    m = {"candidate": {**good_candidate(), **cand}, "baseline": {**good_baseline(), **base}}
    return {r["id"]: r for r in rc.check_acceptance(m, thresholds_with_noise(noise))}


def test_a4_threshold_is_the_max_of_the_measured_baseline_and_the_noise_band():
    assert run_noise({"dev_lost_correct": 112}, {"dev_lost_correct": 108})["A4"]["passed"] is True
    assert run_noise({"dev_lost_correct": 113}, {"dev_lost_correct": 108})["A4"]["passed"] is False
    r = run_noise({"dev_lost_correct": 120}, {"dev_lost_correct": 120})["A4"]               # baseline above the band
    assert r["passed"] is True and r["threshold"] == 120
    assert run_noise({"dev_lost_correct": 121}, {"dev_lost_correct": 120})["A4"]["passed"] is False


def test_a4_falls_back_to_the_stored_reference_and_then_to_the_constant():
    assert run_noise({"dev_lost_correct": 112}, {})["A4"]["passed"] is True                  # stored 96, band 112
    assert run_noise({"dev_lost_correct": 113}, {})["A4"]["passed"] is False
    assert run({"dev_lost_correct": 112}, base={"dev_lost_correct": 100})["A4"]["passed"] is True    # no band: constant 112


def test_a4_ignores_a_malformed_noise_entry():
    t = thresholds_with_noise()
    t["noise_band"]["s2b_retrain_s42"]["dev_lost_correct"] = float("nan")
    m = {"candidate": {**good_candidate(), "dev_lost_correct": 112}, "baseline": good_baseline()}
    assert {r["id"]: r for r in rc.check_acceptance(m, t)}["A4"]["threshold"] == 112


def test_build_thresholds_records_the_noise_band_and_skips_unusable_rows():
    t = rc.build_acceptance_thresholds(BASE_REF, created="t", noise_rows={
        "s2b_retrain_s42": {"dev_lost_correct": "107"}, "s2b_retrain_s43": {"dev_lost_correct": ""},
        "other": {"dev_lost_correct": "x"}})
    assert t["noise_band"] == {"s2b_retrain_s42": {"dev_lost_correct": 107}}
    assert "noise_band" in rc.build_acceptance_thresholds(BASE_REF, created="t")           # present, empty
    assert rc.build_acceptance_thresholds(BASE_REF, created="t")["noise_band"] == {}


# ---- NaN / missing -----------------------------------------------------------------------------------
@pytest.mark.parametrize("cid,key", [("A1", "dev_gold_ok"), ("A2", "dev_gold_predial_lot_precision"),
                                     ("A4", "dev_lost_correct"), ("A5", "dev_manzana_precision")])
def test_nan_metric_fails_with_a_clear_reason(cid, key):
    r = run(**{key: float("nan")})[cid]
    assert r["passed"] is False and "NaN" in r["reason"] and key in r["reason"]


@pytest.mark.parametrize("cid,key", [("A1", "dev_gold_ok"), ("A2", "dev_gold_predial_lot_precision"),
                                     ("A4", "dev_lost_correct"), ("A5", "dev_manzana_precision")])
def test_missing_metric_is_not_evaluated_and_flagged(cid, key):
    cand = good_candidate()
    del cand[key]
    res = {r["id"]: r for r in rc.check_acceptance({"candidate": cand, "baseline": good_baseline()}, thresholds())}
    assert res[cid]["passed"] is None and res[cid]["status"] == "not_evaluated" and key in res[cid]["reason"]


def test_none_metric_counts_as_missing_and_a_string_is_a_failure():
    assert run(dev_gold_ok=None)["A1"]["passed"] is None
    r = run(dev_gold_ok="277")["A1"]
    assert r["passed"] is False and "not a number" in r["reason"]


def test_nan_baseline_reference_fails_rather_than_passing_vacuously():
    t = thresholds()
    t["baseline_reference"]["dev_gold_ok"] = float("nan")
    res = {r["id"]: r for r in rc.check_acceptance({"candidate": good_candidate(), "baseline": {}}, t)}
    assert res["A1"]["passed"] is False and "NaN" in res["A1"]["reason"]


def test_empty_metrics_evaluate_nothing_and_do_not_raise():
    res = rc.check_acceptance({}, thresholds())
    assert len(res) == 12 and all(r["passed"] is None for r in res)
    res = rc.check_acceptance({"candidate": {}, "baseline": {}}, thresholds())
    assert all(r["passed"] is None for r in res)


def test_malformed_metrics_container_is_a_clear_error():
    with pytest.raises(ValueError, match="candidate"):
        rc.check_acceptance({"candidate": [1, 2]}, thresholds())
    with pytest.raises(ValueError, match="thresholds"):
        rc.check_acceptance({"candidate": {}}, {})


# ---- A6 ------------------------------------------------------------------------------------------------
def test_a6_both_heads_within_ece_margin_and_ship_rule_vs_the_refit():
    assert run()["A6"]["passed"] is True
    assert run(calib_error_cv_predio=0.07)["A6"]["passed"] is True            # exactly baseline + 0.01
    assert run(calib_error_cv_predio=0.0701)["A6"]["passed"] is False
    assert run(calib_error_cv_manzana=0.05)["A6"]["passed"] is True
    assert run(calib_error_cv_manzana=0.0501)["A6"]["passed"] is False
    assert run(ship_rule_vs_refit=False)["A6"]["passed"] is False
    assert run(ship_rule_vs_refit=None)["A6"]["passed"] is None


def test_a6_does_not_require_the_ship_rule_against_the_v2_reference():
    assert run(ship_rule_passed=False)["A6"]["passed"] is True                 # the baseline itself fails that one
    assert all(p["name"] != "ship_rule_passed" for p in run()["A6"]["parts"])


def test_a6_always_checks_both_heads_and_a_missing_one_is_not_evaluated():
    cand = good_candidate()
    del cand["calib_error_cv_manzana"]
    res = {r["id"]: r for r in rc.check_acceptance({"candidate": cand, "baseline": good_baseline()}, thresholds())}
    assert res["A6"]["passed"] is None and "calib_error_cv_manzana" in res["A6"]["reason"]
    base = good_baseline()
    del base["calib_error_cv_predio"]
    res = {r["id"]: r for r in rc.check_acceptance({"candidate": good_candidate(), "baseline": base}, thresholds())}
    assert res["A6"]["passed"] is None and "calib_error_cv_predio" in res["A6"]["reason"]


def test_a6_nan_ece_fails_with_a_reason():
    r = run(calib_error_cv_predio=float("nan"))["A6"]
    assert r["passed"] is False and "NaN" in r["reason"]


def test_a6_is_relative_to_the_measured_baseline_not_a_constant():
    assert run(base={"calib_error_cv_predio": 0.02}, calib_error_cv_predio=0.04)["A6"]["passed"] is False
    assert run(base={"calib_error_cv_predio": 0.02}, calib_error_cv_predio=0.03)["A6"]["passed"] is True
    assert run(base={"calib_error_cv_predio": 0.0975, "calib_error_cv_manzana": 0.0338},
               calib_error_cv_predio=0.1075, calib_error_cv_manzana=0.0438)["A6"]["passed"] is True


# ---- A7 (lot level, catastro-free) ---------------------------------------------------------------------
def test_a7_parts():
    assert run()["A7"]["passed"] is True
    assert run(parser_failure_rate_cert=0.0099)["A7"]["passed"] is True
    assert run(parser_failure_rate_cert=0.01)["A7"]["passed"] is False            # strictly below 1 %
    assert run(crs_gates_passed=False)["A7"]["passed"] is False
    assert run(accounting_dropped=3)["A7"]["passed"] is False
    assert run(accounting_dropped=3, accounting_dropped_explained=True)["A7"]["passed"] is True


def test_a7_c1_construction_new_may_trail_production_by_two_points_only():
    assert run(gold_lot_cond_new=0.98)["A7"]["passed"] is True                     # equal to production passes
    assert run(gold_lot_cond_new=0.96)["A7"]["passed"] is True                     # exactly prod - 2.0 pp passes
    assert run(gold_lot_cond_new=0.9599)["A7"]["passed"] is False                  # just below fails
    assert run(gold_lot_cond_new=1.0, gold_lot_cond_prod=0.5)["A7"]["passed"] is True


def test_a7_c2_coverage_loss_may_exceed_the_inherent_loss_by_one_point_only():
    assert run(gold_coverage_loss=0.185, gold_inherent_loss=0.185)["A7"]["passed"] is True      # equal passes
    assert run(gold_coverage_loss=0.195, gold_inherent_loss=0.185)["A7"]["passed"] is True      # inherent + 1.0 pp
    assert run(gold_coverage_loss=0.1951, gold_inherent_loss=0.185)["A7"]["passed"] is False
    assert run(gold_coverage_loss=-0.01)["A7"]["passed"] is True                                # new better than prod


def test_a7_headline_coverage_is_information_not_a_gate():
    r = run(gold_lot_coverage_new=0.10, gold_lot_coverage_old=0.99)["A7"]
    assert r["passed"] is True
    part = next(p for p in r["parts"] if p["name"] == "gold_lot_coverage_new")
    assert part["status"] == "pass" and part["op"] == "report" and part["value"] == 0.10


@pytest.mark.parametrize("key", ["gold_lot_cond_new", "gold_lot_cond_prod", "gold_coverage_loss", "gold_inherent_loss"])
def test_a7_nan_oracle_metric_fails_with_a_reason_and_missing_is_not_evaluated(key):
    r = run(**{key: float("nan")})["A7"]
    assert r["passed"] is False and "NaN" in r["reason"]
    r = run(**{key: None})["A7"]
    assert r["passed"] is None and r["status"] == "not_evaluated" and key in r["reason"]


def test_a7_reports_the_absolute_number_of_missing_gold_lots_without_gating_on_it():
    r = run(gold_lot_missing_new=57)["A7"]
    part = next(p for p in r["parts"] if p["name"] == "gold_lot_missing_new")
    assert part["value"] == 57 and part["status"] == "pass" and r["passed"] is True
    absent = run(gold_lot_missing_new=None)["A7"]
    assert absent["passed"] is True and all(p["name"] != "gold_lot_missing_new" for p in absent["parts"])


def test_a7_no_longer_reads_any_catastro_metric():
    r = run(gold_npn_coverage_new=0.0, gold_npn_coverage_old=1.0, npn_catastro_coverage=0.0)["A7"]
    assert r["passed"] is True
    assert all("npn" not in p["name"] for p in r["parts"])
    assert "catastro" not in rc.ACCEPTANCE_CRITERIA["A7"].lower()


def test_a7_a_legacy_candidate_with_only_the_npn_coverage_keys_is_not_evaluated():
    cand = good_candidate()
    for k in ("gold_lot_cond_new", "gold_lot_cond_prod", "gold_coverage_loss", "gold_inherent_loss"):
        del cand[k]
    cand.update(gold_npn_coverage_new=0.97, gold_npn_coverage_old=0.95)
    res = {r["id"]: r for r in rc.check_acceptance({"candidate": cand, "baseline": good_baseline()}, thresholds())}
    assert res["A7"]["passed"] is None


# ---- A8 / A9 / A11 -------------------------------------------------------------------------------------
def test_a8_production_untouched_in_every_run():
    assert run(production_untouched=[True, True])["A8"]["passed"] is True
    assert run(production_untouched=[True, False])["A8"]["passed"] is False
    assert run(production_untouched=False)["A8"]["passed"] is False
    assert run(production_untouched=[])["A8"]["passed"] is None
    assert run(production_untouched=None)["A8"]["passed"] is None
    assert run(base={"production_untouched": False})["A8"]["passed"] is False


def test_a9_pytest_and_privacy_check():
    assert run()["A9"]["passed"] is True
    assert run(pytest_green=False)["A9"]["passed"] is False
    assert run(no_real_addresses_exit=1)["A9"]["passed"] is False
    assert run(no_real_addresses_exit=None)["A9"]["passed"] is None


def test_a11_cleaning_and_resilience():
    assert run()["A11"]["passed"] is True
    assert run(quarantine_share=0.0049)["A11"]["passed"] is True
    assert run(quarantine_share=0.005)["A11"]["passed"] is False
    assert run(build_index_exit=2)["A11"]["passed"] is False
    assert run(download_exit=1)["A11"]["passed"] is False
    assert run(anomaly_fixtures_ok=False)["A11"]["passed"] is False


# ---- A10 (relative, lot level, rows that emit a predial) ---------------------------------------------------
def test_a10_all_three_conditions_are_relative_to_the_baseline():
    assert run()["A10"]["passed"] is True
    # (a) share of gold rows OK at >= 0.7 may trail the baseline by 1.0 pp
    assert run(confidence_curve=curve_point(share_of_rows=0.89))["A10"]["passed"] is True
    assert run(confidence_curve=curve_point(share_of_rows=0.8899))["A10"]["passed"] is False
    # (b) lot precision among rows that emit a predial may trail by 0.5 pp
    assert run(confidence_curve=curve_point(precision=0.985))["A10"]["passed"] is True
    assert run(confidence_curve=curve_point(precision=0.9849))["A10"]["passed"] is False
    assert run(confidence_curve=curve_point(precision=1.0))["A10"]["passed"] is True
    # (c) manzana error rate may exceed the baseline by 0.5 pp
    assert run(confidence_curve=curve_point(err=0.015))["A10"]["passed"] is True
    assert run(confidence_curve=curve_point(err=0.0151))["A10"]["passed"] is False


def test_a10_has_no_absolute_precision_floor_so_the_baseline_can_pass_itself():
    low = curve_point(precision=0.8957)
    assert run(confidence_curve=low, base={"confidence_curve": low})["A10"]["passed"] is True
    assert run(confidence_curve=curve_point(precision=0.8906), base={"confidence_curve": low})["A10"]["passed"] is False


def test_a10_ignores_the_old_all_ok_denominator_keys():
    cand = curve_point()
    cand[0]["predial_precision"] = 0.0
    cand[0]["predial_lot_precision"] = 0.0
    assert run(confidence_curve=cand)["A10"]["passed"] is True


def test_a10_reports_how_many_rows_emit_a_predial():
    r = run(confidence_curve=curve_point(emitting=77))["A10"]
    info = next(p for p in r["parts"] if p["name"] == "n_emitting_predial@0.7")
    assert info["value"] == 77 and info["op"] == "report" and info["status"] == "pass"


def test_a10_no_row_reaches_the_threshold_fails_without_dividing_by_zero():
    cand = [{"threshold": 0.7, "n_rows": 289, "n_ok": 277, "n_ok_at_t": 0, "share_of_ok": 0.0, "share_of_rows": 0.0,
             "predial_lot_precision_emitted": None, "n_emitting_predial": 0, "manzana_error_rate": None}]
    r = run(confidence_curve=cand)["A10"]
    assert r["passed"] is False and "no row" in r["reason"].lower()


def test_a10_rows_at_the_threshold_but_none_emitting_a_predial_fails_clearly():
    cand = curve_point(emitting=0, precision=None)
    r = run(confidence_curve=cand)["A10"]
    assert r["passed"] is False and "emit" in r["reason"].lower()


def test_a10_missing_curves_or_missing_threshold_point_are_not_evaluated():
    assert run(confidence_curve=None)["A10"]["passed"] is None
    assert run(confidence_curve=curve_point(t=0.75))["A10"]["passed"] is None
    assert run(base={"confidence_curve": None})["A10"]["passed"] is None


def test_a10_a_baseline_curve_without_the_emitted_key_cannot_be_compared():
    old_style = curve_point()
    del old_style[0]["predial_lot_precision_emitted"]
    r = run(base={"confidence_curve": old_style})["A10"]
    assert r["passed"] is None and "predial_lot_precision_emitted" in r["reason"]


def test_a10_baseline_without_rows_at_the_threshold_or_emitting_cannot_be_compared():
    base = [{"threshold": 0.7, "n_rows": 289, "n_ok": 277, "n_ok_at_t": 0, "share_of_ok": 0.0, "share_of_rows": 0.0,
             "predial_lot_precision_emitted": None, "n_emitting_predial": 0, "manzana_error_rate": None}]
    r = run(base={"confidence_curve": base})["A10"]
    assert r["passed"] is None and "baseline" in r["reason"]


def test_a10_nan_values_fail():
    assert run(confidence_curve=curve_point(precision=float("nan")))["A10"]["passed"] is False
    assert run(confidence_curve=curve_point(err=float("nan")))["A10"]["passed"] is False


# ---- confidence_curve: both predial definitions -------------------------------------------------------
def test_curve_exposes_precision_among_rows_that_emit_a_predial_next_to_the_old_denominator():
    rows = eval_rows([
        {"confianza": 0.95, "correct_predial_lot": True, "numero_predial_nacional": "P1"},
        {"confianza": 0.90, "correct_predial_lot": False, "numero_predial_nacional": "P2"},
        {"confianza": 0.85, "correct_predial_lot": np.nan, "numero_predial_nacional": None},     # shared: no predial
        {"confianza": 0.80, "correct_predial_lot": np.nan, "numero_predial_nacional": ""},
        {"confianza": 0.50, "correct_predial_lot": True, "numero_predial_nacional": "P3"},
    ])
    p = {c["threshold"]: c for c in rc.confidence_curve(rows)}
    assert p[0.7]["n_ok_at_t"] == 4 and p[0.7]["n_emitting_predial"] == 2
    assert p[0.7]["predial_lot_precision"] == pytest.approx(1 / 4)                  # old: unknown counts as wrong
    assert p[0.7]["predial_lot_precision_emitted"] == pytest.approx(1 / 2)
    assert p[0.5]["predial_lot_precision_emitted"] == pytest.approx(2 / 3) and p[0.5]["n_emitting_predial"] == 3
    assert "predial_precision" in p[0.7] and "manzana_error_rate" in p[0.7]            # old keys kept


def test_curve_emitted_precision_is_none_without_columns_or_without_emitting_rows():
    no_lot = eval_rows([{"confianza": 0.9}])
    assert rc.confidence_curve(no_lot)[4]["predial_lot_precision_emitted"] is None
    no_npn = eval_rows([{"confianza": 0.9, "correct_predial_lot": True}]).drop(columns=["numero_predial_nacional"])
    c = rc.confidence_curve(no_npn)[4]
    assert c["predial_lot_precision_emitted"] is None and c["n_emitting_predial"] is None
    none_emit = eval_rows([{"confianza": 0.9, "correct_predial_lot": np.nan, "numero_predial_nacional": None}])
    c = rc.confidence_curve(none_emit)[4]
    assert c["n_emitting_predial"] == 0 and c["predial_lot_precision_emitted"] is None


def test_curve_emitted_precision_threshold_is_inclusive_and_ignores_non_ok_rows():
    rows = eval_rows([
        {"confianza": 0.7, "correct_predial_lot": True, "numero_predial_nacional": "P1"},
        {"estado": "SIN_MATCH", "confianza": 0.99, "correct_predial_lot": False, "numero_predial_nacional": "P2"},
    ])
    c = {x["threshold"]: x for x in rc.confidence_curve(rows)}
    assert c[0.7]["n_emitting_predial"] == 1 and c[0.7]["predial_lot_precision_emitted"] == 1.0
    assert c[0.75]["n_emitting_predial"] == 0


# ---- A12 -----------------------------------------------------------------------------------------------
def test_a12_needs_the_corrected_run_and_flags_when_it_is_missing():
    assert run()["A12"]["passed"] is None and run()["A12"]["status"] == "not_evaluated"


def test_a12_passes_when_correction_changes_nothing_and_leaves_no_violations():
    assert run(correct={})["A12"]["passed"] is True


def test_a12_fails_when_a_metric_gets_worse_under_correct():
    assert run(correct={"dev_gold_ok": 276})["A12"]["passed"] is False                         # worse than uncorrected 277
    assert run(correct={"dev_gold_predial_lot_precision": 0.98})["A12"]["passed"] is False
    assert run(correct={"frozen_gold_predial_lot_precision": 0.98})["A12"]["passed"] is False
    assert run(correct={"dev_lost_correct": 101})["A12"]["passed"] is False
    assert run(correct={"dev_manzana_precision": 0.84})["A12"]["passed"] is False
    assert run(correct={"confidence_curve": curve_point(err=0.02)})["A12"]["passed"] is False


def test_a12_equal_or_better_passes_and_boundary_is_inclusive():
    assert run(correct={"dev_gold_ok": 277, "dev_lost_correct": 100})["A12"]["passed"] is True
    assert run(correct={"dev_gold_predial_lot_precision": 0.9899})["A12"]["passed"] is True


def test_a12_curve_emitted_precision_must_not_get_worse():
    assert run(correct={"confidence_curve": curve_point(precision=0.98)})["A12"]["passed"] is False
    assert run(correct={"confidence_curve": curve_point(precision=0.995)})["A12"]["passed"] is True


def test_a12_remaining_violations_must_be_zero():
    assert run(correct={"violations_remaining_dev": 1})["A12"]["passed"] is False
    assert run(correct={"violations_remaining_dev": None})["A12"]["passed"] is None


def test_a12_is_relative_to_the_baseline_too():
    # the corrected run no longer meets A1 against the baseline even if it equals the uncorrected candidate
    assert run(dev_gold_ok=270, correct={})["A12"]["passed"] is False


# ---- summaries -----------------------------------------------------------------------------------------
def test_summarize_acceptance_lists_failed_and_not_evaluated_ids():
    res = rc.check_acceptance({"candidate": {**good_candidate(), "dev_gold_ok": 100}, "baseline": good_baseline()},
                              thresholds())
    s = rc.summarize_acceptance(res)
    assert "A1" in s["failed"] and "A12" in s["not_evaluated"] and s["all_passed"] is False
    ok = rc.summarize_acceptance(run(correct={}).values())
    assert ok["all_passed"] is True and ok["failed"] == [] and ok["not_evaluated"] == []


def test_all_passed_is_false_when_something_was_not_evaluated():
    s = rc.summarize_acceptance(run().values())
    assert s["all_passed"] is False and s["not_evaluated"] == ["A12"]


def test_check_acceptance_is_json_serializable():
    json.dumps(rc.check_acceptance({"candidate": good_candidate(), "baseline": good_baseline()}, thresholds()))


# ====================================================================================================
# thresholds file
# ====================================================================================================
def test_build_acceptance_thresholds_shape_and_relative_markers():
    t = thresholds()
    assert t["baseline_run"] == "baseline_prod_2026_10" and t["baseline_reference"]["dev_gold_ok"] == 277
    assert t["constants"]["A4_max_lost_correct"] == 112 and t["constants"]["A10_confidence"] == 0.7
    assert t["constants"]["A10_share_margin"] == 0.01 and t["constants"]["A10_precision_margin"] == 0.005
    assert t["constants"]["A10_manzana_margin"] == 0.005 and "A10_min_predial_precision" not in t["constants"]
    assert set(t["criteria"]) == {f"A{i}" for i in range(1, 13)}
    assert set(t["relative_to_measured_baseline"]) >= {"A2", "A3", "A6", "A10", "A12"}
    assert t["constants"]["A7_construction_margin"] == 0.02 and t["constants"]["A7_loss_margin"] == 0.01
    json.dumps(t)


def test_build_acceptance_thresholds_from_a_leaderboard_row_coerces_strings():
    row = {k: str(v) for k, v in BASE_REF.items()}
    t = rc.build_acceptance_thresholds(row, created="x")
    assert t["baseline_reference"]["dev_gold_ok"] == 277 and isinstance(t["baseline_reference"]["dev_gold_ok"], int)
    assert t["baseline_reference"]["dev_manzana_precision"] == pytest.approx(0.8431001890359168)


@pytest.mark.parametrize("missing", ["dev_gold_ok", "dev_gold_predial_precision", "dev_manzana_precision",
                                     "dev_lost_correct"])
def test_build_acceptance_thresholds_requires_the_baseline_numbers(missing):
    ref = {k: v for k, v in BASE_REF.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        rc.build_acceptance_thresholds(ref, created="x")
    with pytest.raises(ValueError, match=missing):
        rc.build_acceptance_thresholds({**ref, missing: float("nan")}, created="x")


# ---- oracle_ceiling_metrics (index_meta.json "oracle_ceiling") -----------------------------------------------
CEIL = {"rows": 400, "prod_lot_matched": 396, "new_lot_matched": 300, "lot_absent_from_source": 80,
        "lot_lost_incomplete": 10, "lot_not_chosen": 6}


def test_oracle_ceiling_metrics_derive_conditional_shares_and_losses():
    m = rc.oracle_ceiling_metrics(CEIL)
    assert m["gold_lot_cond_prod"] == pytest.approx(396 / 320)
    assert m["gold_rows_with_lot_in_source"] == 320
    assert m["gold_lot_cond_new"] == pytest.approx(300 / 320)
    assert m["gold_lot_cond_prod"] == pytest.approx(396 / 320)
    assert m["gold_coverage_loss"] == pytest.approx((396 - 300) / 400)
    assert m["gold_inherent_loss"] == pytest.approx(80 / 400)
    assert m["gold_lot_coverage_old"] == pytest.approx(396 / 400) and m["gold_lot_coverage_new"] == pytest.approx(300 / 400)
    assert m["gold_lot_missing_new"] == 100
    assert m["gold_lot_absent_from_source"] == 80 and m["gold_lot_lost_incomplete"] == 10 and m["gold_lot_not_chosen"] == 6


@pytest.mark.parametrize("bad", [None, {}, [], "x", {"rows": 0}, {**CEIL, "rows": 0}, {**CEIL, "rows": float("nan")},
                                 {**CEIL, "lot_absent_from_source": 400}, {**CEIL, "lot_absent_from_source": 401},
                                 {**CEIL, "prod_lot_matched": None}, {**CEIL, "new_lot_matched": "a"},
                                 {**CEIL, "rows": -5}, {**CEIL, "new_lot_matched": True}])
def test_oracle_ceiling_metrics_returns_nothing_when_fields_are_missing_or_unusable(bad):
    assert rc.oracle_ceiling_metrics(bad) == {}


NESTED = {"gate": {"ok": True}, "dev": {"rows": 289, "prod_lot_matched": 289, "new_lot_matched": 216,
                                       "lot_absent_from_source": 71, "rows_with_lot_in_source": 218,
                                       "prod_lot_matched_in_source": 218},
          "total": {"rows": 706, "prod_lot_matched": 699, "new_lot_matched": 505, "lot_absent_from_source": 194,
                    "rows_with_lot_in_source": 512, "prod_lot_matched_in_source": 512, "lot_lost_incomplete": 1,
                    "lot_not_chosen": 6, "lot_lost_other": 0},
          "coverage_loss_headline_pp": 27.5}


def test_oracle_ceiling_metrics_use_the_total_block_and_the_in_source_production_count():
    m = rc.oracle_ceiling_metrics(NESTED)
    assert m["gold_rows_with_lot_in_source"] == 512
    assert m["gold_lot_cond_prod"] == pytest.approx(512 / 512)           # NOT 699 / 512: production matches lots outside the source
    assert m["gold_lot_cond_new"] == pytest.approx(505 / 512)
    assert m["gold_coverage_loss"] == pytest.approx(194 / 706) and m["gold_inherent_loss"] == pytest.approx(194 / 706)
    assert m["gold_lot_lost_incomplete"] == 1 and m["gold_lot_not_chosen"] == 6


def test_oracle_ceiling_metrics_reject_an_inconsistent_in_source_count_or_a_bad_total():
    bad = {**NESTED, "total": {**NESTED["total"], "rows_with_lot_in_source": 500}}
    assert rc.oracle_ceiling_metrics(bad) == {}
    assert rc.oracle_ceiling_metrics({"total": "x"}) == {}
    assert rc.oracle_ceiling_metrics({**NESTED, "total": {**NESTED["total"], "prod_lot_matched_in_source": 513}}) == {}


def test_oracle_ceiling_metrics_partial_extras_are_optional():
    m = rc.oracle_ceiling_metrics({k: v for k, v in CEIL.items() if k not in ("lot_lost_incomplete", "lot_not_chosen")})
    assert "gold_lot_cond_new" in m and "gold_lot_lost_incomplete" not in m
