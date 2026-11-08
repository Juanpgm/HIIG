"""Round-2 defects of the index build (F1-F10): predial semantics, degenerate records, group spread,
official-query cap, interior tails, n_predios, id space, remap, accounting.

All addresses are invented (numbers 900+).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nom_fixtures import catastro_docs_frame, catastro_parcels_frame, nom_frame, npn, row

from cali_address import index_union
from cali_address.dataset import query_text_for
from cali_address.index_union import (
    AccountingError, accounting, assign_unit_doc_ids, build_nomenclatura, catastro_key_table,
    doc_key, official_queries, remap_train_pairs, union_docs, union_own_centroids,
)
from cali_address.model import encode_text

ADDRESS = "KR 932B # 946 - 931"
KEY = doc_key("KR 932 B # 946 - 931")


def _build(rows, cat_entries=(), cat_frame=None, tiers=("cert", "pcer")):
    cat = cat_frame if cat_frame is not None else (catastro_key_table(catastro_docs_frame(list(cat_entries))) if cat_entries else None)
    build = build_nomenclatura(nom_frame(*rows), tiers=tiers)
    docs = union_docs(build.docs, cat)
    return build, cat, docs


# -- F1 (scope change): the document NPN is always the nomenclature one -----------------------------------------

def test_the_cadastre_never_changes_the_document_npn_even_in_the_same_lot():
    nom_npn, cat_npn = npn(unit="000000001"), npn(unit="000000777")
    _, _, docs = _build([row(1, ADDRESS, npn_value=nom_npn)], [("KR 932 B # 946 - 931", cat_npn)])
    d = docs.iloc[0]
    assert d["numero_predial_nacional"] == nom_npn and d["npn_choice"] == "single"
    assert d["npn_catastro"] == cat_npn                      # audit only


def test_only_nomenclature_choice_labels_exist():
    rows = [row(1, ADDRESS + " AP 1", interior="AP 1", npn_value=npn(unit="000000001")),
            row(2, ADDRESS, npn_value=npn(unit="000000007")),
            row(3, "CL 947 # 931 - 912 AP 1", interior="AP 1", npn_value=npn(unit="000000009")),
            row(4, "CL 947 # 931 - 912 AP 2", interior="AP 2", npn_value=npn(unit="000000004")),
            row(5, "CL 948 # 931 - 912")]
    build = build_nomenclatura(nom_frame(*rows))
    assert set(build.stats["npn_choice"]) == {"base_row", "lowest", "single"}


def test_catastro_only_addresses_are_not_documents():
    _, _, docs = _build([row(1, ADDRESS)], [("KR 26 H 1 # 973 - 910", npn(lot="0918"))])
    assert len(docs) == 1 and set(docs["fuente"]) == {"nomenclatura_cert"}


def test_union_without_a_cadastre_leaves_the_audit_columns_empty():
    _, _, docs = _build([row(1, ADDRESS)])
    assert docs["npn_catastro"].iloc[0] in (None, "None") or pd.isna(docs["npn_catastro"].iloc[0])


def test_the_own_centroid_sidecar_follows_the_docs():
    build, _, docs = _build([row(1, ADDRESS), row(2, "CL 947 # 931 - 912")])
    own = union_own_centroids(build.docs)
    assert own["objectid"].tolist() == docs["objectid"].tolist()
    assert own["numero_predial_nacional"].tolist() == docs["numero_predial_nacional"].tolist()
    assert own[["own_lon", "own_lat"]].notna().all().all()


# -- F2: degenerate records ------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["CL 946 # -", "KR 932B # 946 -", "KR 932 # - 12", "CL 946"])
def test_a_record_without_via_number_cross_number_or_plate_is_quarantined_not_indexed(text):
    nom = nom_frame(row(1, text), row(2, "CL 947 # 931 - 912"))
    build = build_nomenclatura(nom)
    assert build.buckets["incomplete_address"] == 1 and build.buckets["dropped"] == 0
    assert build.docs["direccion"].tolist() == ["CL 947 # 931 - 912"]
    assert build.dropped["reason"].tolist() == ["incomplete_address"]
    assert build.dropped["gid"].tolist() == [1]
    assert sum(build.buckets.values()) == 2


def test_an_incomplete_record_with_an_interior_is_quarantined_too_and_makes_no_unit():
    nom = nom_frame(row(1, "CL 946 # - AP 302", interior="AP 302"), row(2, "CL 947 # 931 - 912"))
    build = build_nomenclatura(nom)
    assert build.buckets["incomplete_address"] == 1 and build.units.empty


def test_an_incomplete_record_cannot_absorb_complete_ones():
    # three incomplete records with different lots used to share one doc
    rows = [row(i, "CL 946 # -", npn_value=npn(lot=f"09{i:02d}", unit="000000001")) for i in range(1, 4)]
    build = build_nomenclatura(nom_frame(*rows, row(9, "CL 946 # 931 - 912")))
    assert len(build.docs) == 1 and build.buckets["incomplete_address"] == 3


def test_unparseable_text_stays_a_fallback_doc_and_is_not_incomplete():
    build = build_nomenclatura(nom_frame(row(1, "XYZ 912 QQ")))
    assert build.buckets["fallback_doc"] == 1 and build.buckets["incomplete_address"] == 0


def test_incomplete_records_are_excluded_from_official_queries():
    build, _, docs = _build([row(1, "CL 946 # -"), row(2, "CL 947 # 931 - 912")])
    queries = official_queries(build.rows, docs)
    assert queries and all("946" not in text for text, _ in queries)


# -- F2: group spread ------------------------------------------------------------------------------------------

def _pt(gid, lat_off, lon_off=0.0, lot="0917", text=ADDRESS, **kw):
    return row(gid, text, lon=-76.5 + lon_off, lat=3.4 + lat_off, npn_value=npn(lot=lot, unit=f"{gid:09d}"), **kw)


def test_spread_threshold_constants():
    assert index_union.SPREAD_SPLIT_M == 1000.0


def test_single_record_has_zero_spread():
    assert build_nomenclatura(nom_frame(_pt(1, 0.0))).docs.iloc[0]["spread_m"] == 0.0


def test_spread_is_the_max_distance_to_the_median_centroid_in_metres():
    build = build_nomenclatura(nom_frame(_pt(1, 0.0), _pt(2, 0.0), _pt(3, 0.006)))     # ~667 m
    assert len(build.docs) == 1
    assert build.docs.iloc[0]["spread_m"] == pytest.approx(667, abs=15)
    assert build.buckets["spread_outlier"] == 0


def test_groups_spanning_distinct_lots_far_apart_split_into_one_doc_per_lot():
    rows = [_pt(1, 0.0, lot="0917"), _pt(2, 0.0001, lot="0917"), _pt(3, 0.03, lot="0918"), _pt(4, 0.0301, lot="0918")]
    build = build_nomenclatura(nom_frame(*rows))
    assert len(build.docs) == 2
    assert set(build.docs["doc_key"]) == {KEY}
    assert sorted(d[13:21] for d in build.docs["numero_predial_nacional"]) == ["0917", "0918"] or True
    assert sorted({p[17:21] for p in build.docs["numero_predial_nacional"]}) == ["0917", "0918"]
    assert (build.docs["spread_m"] < 50).all()
    assert build.buckets["base_doc_new"] == 2 and build.buckets["base_doc_merged"] == 2
    assert build.rows.groupby("doc_ord")["gid"].apply(sorted).tolist() == [[1, 2], [3, 4]]


def test_far_points_of_a_single_lot_are_outliers_not_a_split():
    rows = [_pt(1, 0.0), _pt(2, 0.0001), _pt(3, 0.0002), _pt(4, 0.05)]
    build = build_nomenclatura(nom_frame(*rows))
    assert len(build.docs) == 1 and build.docs.iloc[0]["spread_m"] < 100
    assert build.buckets["spread_outlier"] == 1
    assert build.dropped["reason"].tolist() == ["spread_outlier"] and build.dropped["gid"].tolist() == [4]
    assert sum(build.buckets.values()) == 4


def test_without_two_valid_lots_the_far_group_is_not_split_by_an_invalid_npn():
    rows = [_pt(1, 0.0), _pt(2, 0.0001), row(3, ADDRESS, lon=-76.5, lat=3.45, npn_value=npn()[:29])]
    build = build_nomenclatura(nom_frame(*rows))
    assert len(build.docs) == 1 and build.buckets["spread_outlier"] == 1


def test_two_far_points_keep_exactly_one_and_quarantine_the_other():
    build = build_nomenclatura(nom_frame(_pt(1, 0.0, lot="0917"), row(2, ADDRESS, lon=-76.5, lat=3.45, npn_value=npn()[:29])))
    assert len(build.docs) == 1 and build.buckets["spread_outlier"] == 1
    assert build.docs.iloc[0]["spread_m"] == 0.0


def test_a_lower_tier_member_of_a_split_group_follows_its_lot():
    rows = [_pt(1, 0.0, lot="0917"), _pt(2, 0.03, lot="0918"), _pt(3, 0.0301, lot="0918", estado="pcer")]
    build = build_nomenclatura(nom_frame(*rows))
    by_gid = build.rows.set_index("gid")["doc_ord"]
    assert by_gid[3] == by_gid[2] != by_gid[1]
    assert sum(build.buckets.values()) == 3


def test_spread_does_not_depend_on_input_order():
    rows = [_pt(1, 0.0, lot="0917"), _pt(2, 0.0001, lot="0917"), _pt(3, 0.03, lot="0918"), _pt(4, 0.05, lot="0917")]
    a = build_nomenclatura(nom_frame(*rows))
    b = build_nomenclatura(nom_frame(*reversed(rows)))
    pd.testing.assert_frame_equal(a.docs, b.docs)
    assert a.buckets == b.buckets


# -- F4: interior tails ----------------------------------------------------------------------------------------

def test_a_registered_complement_tail_is_still_stripped_without_ndinterior():
    build = build_nomenclatura(nom_frame(row(1, "CL 946 # 931 - 912 TORRE 3 AP 302", estado="pcer", interior=None)))
    assert build.docs.iloc[0]["direccion"] == "CL 946 # 931 - 912"
    assert build.stats["interior_strip_method"]["parser"] == 1


@pytest.mark.parametrize("tail", ["9 9 X", "9 9", "12 B", "5"])
def test_a_bare_number_tail_is_not_stripped_and_is_marked(tail):
    build = build_nomenclatura(nom_frame(row(1, f"CL 946 # 931 - 912 {tail}", estado="pcer", interior=None)))
    methods = build.stats["interior_strip_method"]
    assert methods["none_unregistered_tail"] == 1 and methods["parser"] == 0
    assert build.units.empty and build.buckets["unit"] == 0
    assert build.docs.iloc[0]["direccion"].startswith("CL 946 # 931 - 912")
    assert tail.split()[0] in build.rows.iloc[0]["base_text"]


def test_every_strip_method_is_always_reported():
    build = build_nomenclatura(nom_frame(row(1, "CL 946 # 931 - 912")))
    assert set(build.stats["interior_strip_method"]) == {"suffix", "parser", "none", "none_unregistered_tail"}


# -- F5 (scope change): n_predios counts nomenclature lots only -----------------------------------------------

def test_n_predios_counts_distinct_nomenclature_lots_and_ignores_the_cadastre():
    cat = catastro_docs_frame([("KR 932 B # 946 - 931", npn(unit="000000771"))])
    cat["n_predios"] = 9
    rows = [row(1, ADDRESS, npn_value=npn(lot="0917")), row(2, ADDRESS, npn_value=npn(lot="0918"))]
    _, _, docs = _build(rows, cat_frame=catastro_key_table(cat))
    assert docs.iloc[0]["n_predios"] == 2


# -- F6: id space ----------------------------------------------------------------------------------------------

def test_nomenclature_objectid_is_offset_and_the_raw_gid_is_kept():
    build, _, docs = _build([row(5, "CL 947 # 931 - 912")])
    d = docs.iloc[0]
    assert index_union.NOMENCLATURE_OBJECTID_OFFSET == 10_000_000
    assert d["objectid"] == 5 + 10_000_000 and d["source_objectid"] == 5


def test_objectid_is_unique_across_the_docs_table():
    rows = [row(i, f"CL {900 + i} # 931 - 912") for i in range(1, 8)] + [row(9, "KR 950 # 931 - 912", estado="pcer")]
    _, _, docs = _build(rows)
    assert docs["objectid"].is_unique and len(docs) == 8


def test_a_missing_gid_still_gets_a_unique_objectid():
    nom = nom_frame(row(1, "CL 947 # 931 - 912"), row(2, "CL 948 # 931 - 912"))
    nom["gid"] = pd.array([None, None], dtype="Int64")
    docs = union_docs(build_nomenclatura(nom).docs)
    assert docs["objectid"].is_unique


def test_the_docstring_warns_that_the_catalog_objectid_fallback_is_invalid():
    assert "load_own_centroids" in index_union.__doc__ and "fallback" in index_union.__doc__


# -- F8: expansion-zone comunas ------------------------------------------------------------------------------

@pytest.mark.parametrize("comuna", ["51", "52", "53"])
def test_expansion_zone_comuna_codes_are_kept_as_they_are(comuna):
    value = npn(comuna=comuna, barrio="05")
    _, _, docs = _build([row(1, "CL 947 # 931 - 912", npn_value=value, zona="EXP")])
    d = docs.iloc[0]
    assert (d["comuna"], d["barrio_code"], d["barrio"]) == (comuna, comuna + "05", "05")


# -- F3: official queries --------------------------------------------------------------------------------------

def _tokens(text):
    a, b = query_text_for(text)
    return encode_text(a).tobytes(), encode_text(b).tobytes()


def _units_rows(n, address=ADDRESS):
    return [row(i, f"{address} AP {i}", interior=f"AP {i}", cons2=f"KR 932B 946 931 AP {i}",
                npn_value=npn(unit="000000001")) for i in range(1, n + 1)]


def test_official_cap_is_twelve_per_doc():
    assert index_union.OFFICIAL_CAP_PER_DOC == 12
    build, _, docs = _build(_units_rows(40))
    queries = official_queries(build.rows, docs)
    assert len(queries) == 12 and {d for _, d in queries} == {0}


def test_a_cap_can_be_overridden_and_is_deterministic():
    build, _, docs = _build(_units_rows(40))
    a = official_queries(build.rows, docs, cap=5)
    assert len(a) == 5 and a == official_queries(build.rows, docs, cap=5)


def test_the_token_level_duplicates_collapse_into_one_query():
    # "KR 932B # 946 - 931" and "KR 932B 946 931"-style spellings that encode alike must count once per doc
    rows = [row(1, "CL 947 # 931 - 912", cons2="CL 947 # 931 - 912"), row(2, "CL 947 # 931 - 912", cons2="CL 947 # 931 - 912")]
    build, _, docs = _build(rows)
    queries = official_queries(build.rows, docs)
    keys = [(_tokens(t), d) for t, d in queries]
    assert len(keys) == len(set(keys))


def test_texts_that_differ_only_in_case_or_spacing_but_encode_alike_are_deduplicated():
    rows = [row(1, "CL 947 # 931 - 912", cons2="cl 947 931 912"), row(2, "CL 947 # 931 - 912", cons2="CL 947 931 912")]
    build, _, docs = _build(rows)
    keys = [(_tokens(t), d) for t, d in official_queries(build.rows, docs)]
    assert len(keys) == len(set(keys))


def test_the_cap_keeps_one_glued_and_one_spaced_form():
    build, _, docs = _build(_units_rows(40))
    texts = [t for t, _ in official_queries(build.rows, docs)]
    assert any("932B" in t for t in texts) and any("932 B" in t for t in texts)


def test_the_cap_prefers_distinct_canonical_forms():
    build, _, docs = _build(_units_rows(40))
    texts = [t for t, _ in official_queries(build.rows, docs)]
    import re

    forms = {re.sub(r"[^0-9A-Z]+", " ", t.upper().replace("932B", "932 B")).strip() for t in texts}
    assert len(forms) >= 8                         # 12 picks over ~40 distinct forms: not 12 spellings of one form


def test_docs_under_the_cap_keep_every_distinct_encoding():
    build, _, docs = _build(_units_rows(2))
    queries = official_queries(build.rows, docs)
    assert 2 <= len(queries) <= 12
    assert len({(_tokens(t), d) for t, d in queries}) == len(queries)


def test_official_queries_of_a_doc_without_records_do_not_appear_and_empty_input_is_empty():
    build = build_nomenclatura(nom_frame())
    docs = union_docs(build.docs, catastro_key_table(catastro_docs_frame([])))
    assert official_queries(build.rows, docs) == []


def test_split_docs_get_their_own_records_queries():
    rows = [_pt(1, 0.0, lot="0917"), _pt(2, 0.03, lot="0918", text="KR 932 B # 946 - 931")]
    build, _, docs = _build(rows)
    assert len(docs) == 2 and docs["doc_key"].nunique() == 1
    ids = {d for _, d in official_queries(build.rows, docs)}
    assert ids == {0, 1}


def test_units_follow_their_record_doc_in_a_split_group():
    rows = [_pt(1, 0.0, lot="0917"), _pt(2, 0.03, lot="0918"),
            _pt(3, 0.0301, lot="0918", text=ADDRESS + " AP 5", interior="AP 5")]
    build, _, docs = _build(rows)
    units = assign_unit_doc_ids(build.units, docs)
    assert units["doc_id"].tolist() == [int(build.rows.set_index("gid").loc[2, "doc_ord"])]
    assert "doc_ord" not in units.columns


# -- F7: remap -------------------------------------------------------------------------------------------------

def _pairs(*entries):
    return pd.DataFrame([{
        "raw_address": f"raw {i}", "address_key": f"K{i}", "dataset": "fasecolda", "gt_doc_id": gid,
        "gt_manzana": "m", "gt_predial": pred, "gt_comuna": "08", "source_row_id": f"r{i}",
    } for i, (gid, pred) in enumerate(entries)])


def test_remap_with_a_duplicated_doc_key_picks_the_doc_that_holds_the_gt_predial():
    old = catastro_docs_frame([("KR 932 B # 946 - 931", npn(lot="0918", unit="000000001"))])
    new = catastro_key_table(catastro_docs_frame([
        ("KR 932 B # 946 - 931", npn(lot="0917", unit="000000001")),
        ("KR 932 B # 946 - 931", npn(lot="0918", unit="000000002")),
    ], start_oid=5001))
    new["fuente"] = "nomenclatura_cert"
    out, report = remap_train_pairs(_pairs((0, npn(lot="0918", unit="000000001"))), old, new)
    assert out["gt_doc_id"].tolist() == [1] and report["remapped_by_key"] == 1


def test_remap_with_a_duplicated_doc_key_and_no_predial_match_takes_the_first():
    old = catastro_docs_frame([("KR 932 B # 946 - 931", npn(lot="0918", unit="000000001"))])
    new = catastro_key_table(catastro_docs_frame([
        ("KR 932 B # 946 - 931", npn(lot="0917", unit="000000001")),
        ("KR 932 B # 946 - 931", npn(lot="0919", unit="000000002")),
    ], start_oid=5001))
    out, _ = remap_train_pairs(_pairs((0, npn(lot="0918", unit="000000001"))), old, new)
    assert out["gt_doc_id"].tolist() == [0]


def test_remap_report_buckets_still_sum_to_the_input():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = catastro_key_table(catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))]))
    _, rep = remap_train_pairs(_pairs((0, npn(unit="000000001")), (5, None), (7, npn(unit="000000044"))), old, new)
    assert rep["kept"] + rep["dropped_no_npn"] + rep["dropped_ambiguous"] == rep["input"] == 3


def test_pair_npn_agreement_counts_pairs_whose_doc_npn_equals_the_gt_predial():
    docs = catastro_key_table(catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001")),
                                                   ("CL 947 # 931 - 912", npn(unit="000000002"))]))
    pairs = _pairs((0, npn(unit="000000001")), (1, npn(unit="000000099")), (0, None), (7, npn(unit="000000001")))
    assert index_union.pair_npn_agreement(pairs, docs) == {"pairs": 4, "npn_equal": 1, "lot_equal": 2}


# -- F1 (scope change): construction-only oracle ceiling ------------------------------------------------------

def _gold(*entries):
    df = _pairs(*entries)
    df["gt_quality"] = "gold"
    return df


def _lot(value):
    return value[:21]


def test_the_build_exposes_the_lot_sets_of_the_source_kept_and_excluded_records():
    rows = [
        row(1, "CL 947 # 931 - 912", npn_value=npn(lot="0917")),                   # kept
        row(2, "CL 948 # -", npn_value=npn(lot="0918")),                             # incomplete
        row(3, "CL 949 # 931 - 912", lon=None, npn_value=npn(lot="0920")),          # dropped: no coordinates
        row(4, "CL 950 # 931 - 912", estado="pcer", npn_value=npn(lot="0921")),     # excluded tier
    ]
    sets = build_nomenclatura(nom_frame(*rows), tiers=("cert",)).lot_sets
    assert sets["source"] == {_lot(npn(lot="0917")), _lot(npn(lot="0918")), _lot(npn(lot="0920"))}
    assert sets["kept"] == {_lot(npn(lot="0917"))}
    assert sets["excluded"] == {_lot(npn(lot="0918"))}


def _sets(source, kept, excluded=()):
    return {"source": {_lot(npn(lot=x)) for x in source}, "kept": {_lot(npn(lot=x)) for x in kept},
            "excluded": {_lot(npn(lot=x)) for x in excluded}}


def test_oracle_ceiling_counts_lot_and_exact_agreement_separately():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001")), ("CL 947 # 931 - 912", npn(unit="000000002"))])
    new = catastro_key_table(catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001")),
                                                  ("CL 947 # 931 - 912", npn(unit="000000777"))], start_oid=5001))
    gold = _gold((0, npn(unit="000000001")), (1, npn(unit="000000002")))
    got = index_union.oracle_ceiling({"dev": gold}, old, new, _sets(["0917"], ["0917"]))["dev"]
    assert (got["rows"], got["prod_lot_matched"], got["new_lot_matched"]) == (2, 2, 2)
    assert (got["production_exact"], got["new_exact"]) == (2, 1)
    assert got["lot_absent_from_source"] == 0 and got["rows_with_lot_in_source"] == 2


def test_oracle_ceiling_classifies_every_lost_lot():
    old = catastro_docs_frame([(f"CL {946 + i} # 931 - 912", npn(lot=f"09{17 + i}", unit="000000001")) for i in range(5)])
    new = catastro_key_table(catastro_docs_frame([("CL 946 # 931 - 912", npn(lot="0917", unit="000000001"))]))
    gold = _gold(*[(i, old["numero_predial_nacional"][i]) for i in range(5)])
    sets = _sets(source=["0917", "0919", "0920", "0921"], kept=["0917", "0919"], excluded=["0920"])
    got = index_union.oracle_ceiling({"dev": gold}, old, new, sets)["dev"]
    # 0917 matches; 0918 is in no cleaned record; 0919 kept but not the doc's lot; 0920 only in an excluded record;
    # 0921 is in the source but in no kept / excluded record (e.g. no coordinates)
    assert got["new_lot_matched"] == 1
    assert (got["lot_absent_from_source"], got["lot_not_chosen"], got["lot_lost_incomplete"], got["lot_lost_other"]) == (1, 1, 1, 1)
    assert got["rows"] == got["lot_absent_from_source"] + got["rows_with_lot_in_source"] == 5


def test_oracle_ceiling_only_counts_gold_rows_and_survives_unmapped_ids():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = catastro_key_table(catastro_docs_frame([("CL 950 # 931 - 912", npn(lot="0918"))]))
    gold = _gold((0, npn(unit="000000001")), (np.nan, npn(unit="000000001")), (99, npn(unit="000000001")))
    gold = pd.concat([gold, _gold((0, npn(unit="000000001"))).assign(gt_quality="noisy")], ignore_index=True)
    got = index_union.oracle_ceiling({"dev": gold}, old, new, _sets(["0917", "0918"], ["0918"]))["dev"]
    assert got["rows"] == 3 and got["new_lot_matched"] == 0 and got["lot_lost_other"] == 3


def test_oracle_ceiling_falls_back_to_the_predial_lot_when_the_key_is_gone():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = catastro_key_table(catastro_docs_frame([("CL 950 # 931 - 912", npn(unit="000000099"))]))
    got = index_union.oracle_ceiling({"dev": _gold((0, npn(unit="000000001")))}, old, new, _sets(["0917"], ["0917"]))["dev"]
    assert got["new_lot_matched"] == 1 and got["new_exact"] == 0


def test_oracle_ceiling_uses_any_doc_of_a_duplicated_key():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = catastro_key_table(catastro_docs_frame([("CL 946 # 931 - 912", npn(lot="0918")),
                                                  ("CL 946 # 931 - 912", npn(unit="000000001"))], start_oid=5001))
    got = index_union.oracle_ceiling({"dev": _gold((0, npn(unit="000000001")))}, old, new, _sets(["0917", "0918"], ["0917", "0918"]))
    assert got["dev"]["new_lot_matched"] == 1 and got["dev"]["new_exact"] == 1


def test_oracle_ceiling_with_an_empty_split_has_zero_counts():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    got = index_union.oracle_ceiling({"dev": _gold((0, "x")).iloc[0:0]}, old, catastro_key_table(old.copy()), _sets([], []))
    assert got["total"]["rows"] == 0


def _entry(rows, prod, new, absent=0, prod_in_source=None):
    return {"rows": rows, "prod_lot_matched": prod, "new_lot_matched": new, "production_exact": 0, "new_exact": 0,
            "lot_absent_from_source": absent, "rows_with_lot_in_source": rows - absent,
            "prod_lot_matched_in_source": prod if prod_in_source is None else prod_in_source,
            "lot_not_chosen": 0, "lot_lost_incomplete": 0, "lot_lost_other": 0}


def test_gate_passes_when_the_only_loss_is_lots_absent_from_the_source():
    total = _entry(706, 699, 671, absent=34, prod_in_source=671)
    gate = index_union.oracle_ceiling_gate({"dev": total, "total": total})
    assert index_union.ORACLE_MAX_DROP_POINTS == 2.0
    assert gate["ok"] is True and gate["failed"] == []
    assert gate["coverage_loss_inherent_pp"] == pytest.approx(100 * 34 / 706)
    assert gate["coverage_loss_headline_pp"] == pytest.approx(100 * (699 - 671) / 706)


def test_gate_fails_when_a_construction_defect_loses_lots_that_are_in_the_source():
    total = _entry(706, 699, 600, absent=34, prod_in_source=671)        # 71 lots lost beyond the absent ones
    gate = index_union.oracle_ceiling_gate({"total": total})
    assert gate["ok"] is False and gate["failed"] == ["total"]


def test_gate_margin_is_two_points_of_the_in_source_rows():
    ok = _entry(1000, 900, 880, prod_in_source=900)                      # exactly -2.0 pp
    bad = _entry(1000, 900, 879, prod_in_source=900)
    assert index_union.oracle_ceiling_gate({"total": ok})["ok"] is True
    assert index_union.oracle_ceiling_gate({"total": bad})["ok"] is False


def test_gate_is_evaluated_on_the_total_only_not_per_split():
    small = _entry(10, 10, 5)                                            # a tiny split that is bad
    big = _entry(990, 900, 900)
    total = _entry(1000, 910, 905)
    assert index_union.oracle_ceiling_gate({"dev": small, "frozen_test": big, "total": total})["ok"] is True


def test_gate_with_no_gold_rows_or_no_lot_in_source_passes_vacuously():
    assert index_union.oracle_ceiling_gate({"total": _entry(0, 0, 0)})["ok"] is True
    assert index_union.oracle_ceiling_gate({"total": _entry(5, 0, 0, absent=5)})["ok"] is True


# -- coverage loss --------------------------------------------------------------------------------------------

def test_catastro_coverage_counts_cadastral_docs_without_a_nomenclature_doc():
    _, _, docs = _build([row(1, "CL 946 # 931 - 912")])
    cat = catastro_key_table(catastro_docs_frame([
        ("CL 946 # 931 - 912", npn(unit="000000771")),            # same key and lot
        ("CL 950 # 931 - 912", npn(unit="000000772")),            # other key, lot present
        ("CL 951 # 931 - 912", npn(lot="0999")),                  # other key, lot absent
    ]))
    assert index_union.catastro_coverage(docs, cat) == {
        "cadastre_docs": 3, "no_nomenclature_doc_by_key": 2, "no_nomenclature_doc_by_lot": 1}


# -- F9/F10: accounting ----------------------------------------------------------------------------------------

def test_new_buckets_exist_and_count_in_the_sum():
    assert {"incomplete_address", "spread_outlier"} <= set(index_union.BUCKETS)
    acc = accounting(10, {"base_doc_new": 5, "incomplete_address": 3, "spread_outlier": 2})
    assert acc["total"] == 10


def test_accounting_reports_the_quarantine_next_to_the_buckets():
    acc = accounting(10, {"base_doc_new": 10}, quarantined=1126, excluded_lote_prov=817)
    assert acc["quarantined"] == 1126 and acc["excluded_lote_prov"] == 817
    assert acc["accounting_total_with_quarantine"] == 10 + 1126


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_accounting_rejects_malformed_quarantine_counts(bad):
    with pytest.raises(AccountingError):
        accounting(1, {"base_doc_new": 1}, quarantined=bad)


def test_accounting_over_a_degenerate_fixture_adds_up_and_can_fail():
    rows = [
        row(1, ADDRESS), row(2, "KR 932 B # 946 - 931"),                       # base + merged (duplicate key)
        row(3, ADDRESS + " AP 1", interior="AP 1"),                           # unit
        row(4, "XYZ 912 QQ"),                                                 # fallback
        row(5, "CL 947 # 931 - 912", lon=None),                               # dropped
        row(6, "CL 946 # -"), row(7, "KR 932B # 946 -"),                      # incomplete
        _pt(8, 0.0, text="CL 948 # 931 - 912"), _pt(9, 0.0001, text="CL 948 # 931 - 912"),
        _pt(10, 0.06, text="CL 948 # 931 - 912"),                             # spread outlier
        row(11, "CL 949 # 931 - 912", estado="pcer"),                         # excluded tier (cert only)
    ]
    build = build_nomenclatura(nom_frame(*rows), tiers=("cert",))
    assert build.buckets == {
        "base_doc_new": 2, "base_doc_merged": 2, "unit": 1, "fallback_doc": 1, "dropped": 1,
        "excluded_tier": 1, "incomplete_address": 2, "spread_outlier": 1,
    }
    assert accounting(len(rows), build.buckets)["ok"] is True
    broken = dict(build.buckets, base_doc_new=build.buckets["base_doc_new"] - 1)
    with pytest.raises(AccountingError):
        accounting(len(rows), broken)
    lost = dict(build.buckets)
    lost.pop("spread_outlier")
    with pytest.raises(AccountingError):
        accounting(len(rows), lost)
