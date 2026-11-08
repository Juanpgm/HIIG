"""Nomenclature-only index: tier priority, keys, NPN choice, centroids, units, accounting, queries, pair remap.

All addresses are invented (numbers 900+). The cadastre is read-only audit input here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nom_fixtures import catastro_docs_frame, nom_frame, npn, row

from cali_address.index_union import (
    AccountingError, accounting, doc_key, assign_unit_doc_ids, build_nomenclatura, build_nomenclatura_docs,
    catastro_key_table, official_queries, remap_train_pairs, union_docs, union_own_centroids,
)
from cali_address.labels import address_key

#: Columns of the production ``catastro_docs.parquet``: they keep name, order and dtype.
COMPAT = ["direccion", "numero_predial_nacional", "manzana", "comuna", "barrio", "barrio_code", "centroid_lon",
          "centroid_lat", "n_predios", "manzana_nunique", "objectid", "x_m", "y_m", "doc_id"]


def _build(nom, cat_entries=(), tiers=("cert", "pcer")):
    cat = catastro_key_table(catastro_docs_frame(list(cat_entries))) if cat_entries else None
    build = build_nomenclatura(nom, tiers=tiers)
    return build, cat, union_docs(build.docs, cat)


# -- priority: cert > pcer ------------------------------------------------------------------------------------

def test_cert_wins_over_pcer_and_the_cadastre_is_audit_only():
    cat_npn = npn(unit="000000777")
    nom = nom_frame(
        row(1, "KR 932B # 946 - 931", estado="cert", npn_value=npn(unit="000000001")),
        row(2, "KR 932 B # 946 - 931", estado="pcer", npn_value=npn(unit="000000002")),
    )
    build, cat, docs = _build(nom, [("KR 932 B # 946 - 931", cat_npn)])
    assert len(docs) == 1
    d = docs.iloc[0]
    assert d["fuente"] == "nomenclatura_cert"
    assert d["estado_nomenclatura"] == "cert"
    assert d["numero_predial_nacional"] == npn(unit="000000001")      # neither the pcer row's NPN nor the cadastre's
    assert d["npn_catastro"] == cat_npn
    assert build.buckets["base_doc_new"] == 1 and build.buckets["base_doc_merged"] == 1


def test_pcer_builds_a_doc_when_no_cert_row_shares_the_key():
    _, _, docs = _build(nom_frame(row(1, "KR 932B # 946 - 931", estado="pcer")))
    assert docs.iloc[0]["fuente"] == "nomenclatura_pcer" and docs.iloc[0]["estado_nomenclatura"] == "pcer"


def test_cadastral_docs_never_become_documents():
    nom = nom_frame(row(1, "CL 946 # 931 - 912"))
    _, _, docs = _build(nom, [("CL 946 # 931 - 912", npn(unit="000000777")), ("KR 26 H 1 # 973 - 910", npn(lot="0918"))])
    assert docs["fuente"].tolist() == ["nomenclatura_cert"]


def test_direccion_catastro_is_recorded_only_when_the_text_differs():
    nom = nom_frame(row(1, "KR 932B # 946 - 931"), row(2, "CL 947 # 931 - 912"))
    _, _, docs = _build(nom, [("KR 932 B # 946 - 931", npn(unit="000000771")), ("CL 947 # 931 - 912", npn(unit="000000772"))])
    by_dir = docs.set_index("direccion")
    assert by_dir.loc["KR 932 B # 946 - 931", "direccion_catastro"] in (None, "None") or pd.isna(by_dir.loc["KR 932 B # 946 - 931", "direccion_catastro"])
    assert by_dir.loc["KR 932 B # 946 - 931", "npn_catastro"] == npn(unit="000000771")
    nom2 = nom_frame(row(1, "KR 932B # 946 - 931"))
    _, _, docs2 = _build(nom2, [("KR 932B # 946 - 931", npn(unit="000000771"))])
    assert docs2.iloc[0]["direccion_catastro"] == "KR 932B # 946 - 931"


def test_variants_of_the_same_address_collide_on_one_key():
    nom = nom_frame(row(1, "KR 932B # 946 - 931"), row(2, "KR 932 B # 946 - 931"))
    docs, _ = build_nomenclatura_docs(nom)
    assert len(docs) == 1
    assert address_key("KR 932B # 946 - 931") == address_key("KR 932 B # 946 - 931")
    assert doc_key("KR 932B # 946 - 931") == doc_key("KR 932 B # 946 - 931")
    assert docs.iloc[0]["address_key"] == address_key("KR 932 B # 946 - 931")


def test_different_addresses_with_the_same_lossy_address_key_stay_separate_docs():
    a, b = "CL 91 # 94 - 192", "CL 919 # 4 - 192"
    assert address_key(a) == address_key(b) and doc_key(a) != doc_key(b)
    build = build_nomenclatura(nom_frame(row(1, a), row(2, b)))
    assert sorted(build.docs["direccion"]) == sorted([a, b])
    assert build.docs["address_key"].nunique() == 1 and build.docs["doc_key"].nunique() == 2
    assert build.stats["address_key_collisions"] == 1
    assert build.buckets["base_doc_new"] == 2


def test_empty_cert_tier_leaves_no_documents():
    build = build_nomenclatura(nom_frame(row(1, "CL 946 # 931 - 912", estado="pcer")), tiers=("cert",))
    assert build.docs.empty and build.buckets["excluded_tier"] == 1


def test_all_rows_pcer_with_tiers_cert_only_build_no_nomenclature_docs():
    nom = nom_frame(*[row(i, f"CL {946 + i} # 931 - 912", estado="pcer") for i in range(1, 5)])
    build = build_nomenclatura(nom, tiers=("cert",))
    assert build.docs.empty and build.units.empty
    assert build.buckets["excluded_tier"] == 4
    assert accounting(len(nom), build.buckets)["total"] == 4


def test_everything_empty_gives_an_empty_well_typed_frame():
    build = build_nomenclatura(nom_frame())
    docs = union_docs(build.docs, catastro_key_table(catastro_docs_frame([])))
    assert docs.empty
    assert list(docs.columns)[: len(COMPAT)] == COMPAT


# -- compatibility with catastro_docs.parquet ------------------------------------------------------------------

def test_compat_columns_keep_name_order_and_dtype_and_new_columns_come_after():
    nom = nom_frame(row(1, "CL 946 # 931 - 912"), row(2, "KR 932B # 946 - 931", estado="pcer"))
    cat_docs = catastro_docs_frame([("KR 26 H 1 # 973 - 910", npn(lot="0918"))])
    _, _, docs = _build(nom)
    assert list(docs.columns[: len(COMPAT)]) == COMPAT
    for col in COMPAT:
        assert docs[col].dtype == cat_docs[col].dtype, col
    extra = list(docs.columns[len(COMPAT):])
    for name in ("fuente", "estado_nomenclatura", "npn_catastro", "direccion_catastro", "npn_choice", "n_units",
                 "source_objectid", "parse_ok", "address_key"):
        assert name in extra
    assert extra[-4:] == ["spread_m", "n_lots_cert", "n_lots_pcer", "lot_conflict"]


def test_doc_id_is_contiguous_and_metric_coordinates_are_filled():
    nom = nom_frame(*[row(i, f"CL {946 + i} # 931 - 912", lon=-76.5 + i * 1e-4, lat=3.4 + i * 1e-4) for i in range(1, 6)])
    _, _, docs = _build(nom)
    assert docs["doc_id"].tolist() == list(range(len(docs)))
    assert docs["x_m"].notna().all() and docs["y_m"].notna().all()
    assert (docs["x_m"] > 1e5).all()          # metres, not degrees


def test_doc_order_is_cert_by_gid_then_pcer_by_gid():
    nom = nom_frame(
        row(10, "CL 950 # 931 - 912", estado="pcer"),
        row(5, "CL 951 # 931 - 912", estado="cert"),
        row(3, "CL 952 # 931 - 912", estado="pcer"),
        row(7, "CL 953 # 931 - 912", estado="cert"),
    )
    _, _, docs = _build(nom)
    assert docs["direccion"].tolist() == ["CL 951 # 931 - 912", "CL 953 # 931 - 912", "CL 952 # 931 - 912", "CL 950 # 931 - 912"]


def test_union_is_deterministic_across_two_runs():
    nom = nom_frame(*[row(i, f"CL {946 + i % 4} # 931 - 91{i % 5}", estado="cert" if i % 3 else "pcer") for i in range(1, 40)])
    a = _build(nom)[2]
    b = _build(nom.sample(frac=1.0, random_state=3).reset_index(drop=True))[2]
    pd.testing.assert_frame_equal(a, b)


# -- NPN choice -------------------------------------------------------------------------------------------------

ADDRESS = "KR 932B # 946 - 931"


def _choice(rows):
    docs, _ = build_nomenclatura_docs(nom_frame(*rows))
    assert len(docs) == 1
    d = docs.iloc[0]
    return d["npn_choice"], d["numero_predial_nacional"], d


def test_choice_single_when_there_is_one_distinct_npn():
    rows = [row(1, ADDRESS, npn_value=npn(unit="000000005")), row(2, ADDRESS, npn_value=npn(unit="000000005"))]
    assert _choice(rows)[:2] == ("single", npn(unit="000000005"))


def test_choice_base_row_takes_the_lowest_npn_among_rows_without_interior():
    rows = [
        row(1, ADDRESS + " AP 101", interior="AP 101", npn_value=npn(unit="000000001")),
        row(2, ADDRESS, npn_value=npn(unit="000000007")),
        row(3, ADDRESS + " AP 201", interior="AP 201", npn_value=npn(unit="000000003")),
    ]
    assert _choice(rows)[:2] == ("base_row", npn(unit="000000007"))


def test_choice_lowest_when_every_row_has_an_interior():
    rows = [
        row(1, ADDRESS + " AP 301", interior="AP 301", npn_value=npn(unit="000000009")),
        row(2, ADDRESS + " AP 101", interior="AP 101", npn_value=npn(unit="000000004")),
    ]
    assert _choice(rows)[:2] == ("lowest", npn(unit="000000004"))


def test_invalid_npn_is_never_chosen_when_a_valid_one_exists():
    invalid = npn(unit="000000001")[:29]            # sorts below the valid one
    rows = [row(1, ADDRESS, npn_value=invalid), row(2, ADDRESS, npn_value=npn(unit="000000005"))]
    choice, chosen, d = _choice(rows)
    assert chosen == npn(unit="000000005")
    assert d["n_units"] == 1                       # the invalid code is not counted as a unit either


def test_only_invalid_npns_still_give_a_doc():
    rows = [row(1, ADDRESS, npn_value=npn()[:28]), row(2, ADDRESS, npn_value=npn(unit="000000002")[:29])]
    _, chosen, _ = _choice(rows)
    assert chosen in (npn()[:28], npn(unit="000000002")[:29])


def test_blank_npn_does_not_break_the_choice():
    rows = [row(1, ADDRESS, npn_value=""), row(2, ADDRESS, npn_value=npn(unit="000000005"))]
    assert _choice(rows)[1] == npn(unit="000000005")


# -- counts -------------------------------------------------------------------------------------------------------

def test_n_predios_counts_lots_not_units():
    rows = [
        row(1, ADDRESS + " AP 101", interior="AP 101", npn_value=npn(lot="0917", unit="000000001")),
        row(2, ADDRESS + " AP 201", interior="AP 201", npn_value=npn(lot="0917", unit="000000002")),
        row(3, ADDRESS + " AP 301", interior="AP 301", npn_value=npn(lot="0917", unit="000000003")),
        row(4, ADDRESS, npn_value=npn(lot="0918", manzana="0957", unit="000000001")),
    ]
    d = _choice(rows)[2]
    assert d["n_units"] == 4
    assert d["n_predios"] == 2
    assert d["manzana_nunique"] == 2


def test_duplicate_npn_rows_count_once_as_units():
    value = npn(unit="000000005")
    rows = [row(1, ADDRESS + " AP 101", interior="AP 101", npn_value=value),
            row(2, ADDRESS + " AP 201", interior="AP 201", npn_value=value)]
    d = _choice(rows)[2]
    assert d["n_units"] == 1 and d["n_predios"] == 1


# -- centroids ----------------------------------------------------------------------------------------------------

def test_doc_centroid_is_the_median_and_own_centroid_is_the_chosen_rows_point():
    rows = [
        row(1, ADDRESS, npn_value=npn(unit="000000001"), lon=-76.5000, lat=3.4000),
        row(2, ADDRESS + " AP 1", interior="AP 1", npn_value=npn(unit="000000002"), lon=-76.5010, lat=3.4010),
        row(3, ADDRESS + " AP 2", interior="AP 2", npn_value=npn(unit="000000003"), lon=-76.5030, lat=3.4030),
    ]
    docs, _ = build_nomenclatura_docs(nom_frame(*rows))
    d = docs.iloc[0]
    assert d["centroid_lon"] == pytest.approx(-76.5010)
    assert d["centroid_lat"] == pytest.approx(3.4010)
    assert d["numero_predial_nacional"] == npn(unit="000000001")
    assert (d["own_lon"], d["own_lat"]) == (-76.5000, 3.4000)


def test_union_own_centroids_follow_the_doc_order_and_columns():
    nom = nom_frame(row(1, "CL 946 # 931 - 912", lon=-76.51, lat=3.41), row(2, "CL 947 # 931 - 912", lon=-76.52, lat=3.42))
    build, _, docs = _build(nom)
    own = union_own_centroids(build.docs)
    assert list(own.columns) == ["objectid", "numero_predial_nacional", "own_lon", "own_lat"]
    assert len(own) == len(docs) == 2
    assert own["objectid"].tolist() == docs["objectid"].tolist()
    assert own["numero_predial_nacional"].tolist() == docs["numero_predial_nacional"].tolist()
    assert (own.loc[0, "own_lon"], own.loc[0, "own_lat"]) == (-76.51, 3.41)


# -- units sidecar -------------------------------------------------------------------------------------------------

def test_units_get_the_doc_id_of_their_base_document():
    nom = nom_frame(
        row(1, "CL 946 # 931 - 912"),
        row(2, "KR 932B # 946 - 931 AP 302", interior="AP 302"),
        row(3, "KR 932B # 946 - 931 AP 303", interior="AP 303", estado="pcer"),
    )
    build, _, docs = _build(nom)
    units = assign_unit_doc_ids(build.units, docs)
    expected = docs.set_index("doc_key")["doc_id"]
    keys = [doc_key(d) for d in build.rows.loc[build.rows["interior"] != "", "base_text"]]
    assert units["doc_id"].tolist() == [int(expected[k]) for k in keys]
    assert set(units["address_key_base"]) <= set(docs["address_key"])
    assert list(units.columns) == ["address_key_base", "doc_id", "interior", "numero_predial_nacional", "own_lon",
                                   "own_lat", "objectid", "direccion_original", "estado"]
    assert units["doc_id"].dtype == np.int64


def test_a_pcer_unit_under_a_cert_base_points_to_the_cert_doc():
    nom = nom_frame(
        row(1, "KR 932B # 946 - 931", estado="cert"),
        row(2, "KR 932B # 946 - 931 AP 303", interior="AP 303", estado="pcer"),
    )
    build, _, docs = _build(nom)
    units = assign_unit_doc_ids(build.units, docs)
    assert len(docs) == 1 and units["doc_id"].tolist() == [0]
    assert units["estado"].tolist() == ["pcer"]


# -- accounting ---------------------------------------------------------------------------------------------------

def test_accounting_sums_to_the_input():
    acc = accounting(10, {"base_doc_new": 4, "base_doc_merged": 2, "unit": 2, "fallback_doc": 1, "dropped": 1})
    assert acc["total"] == 10
    assert acc["buckets"]["excluded_tier"] == 0
    assert acc["dropped_share"] == pytest.approx(0.1)


@pytest.mark.parametrize("n_input", [9, 11, 0])
def test_accounting_mismatch_raises(n_input):
    with pytest.raises(AccountingError):
        accounting(n_input, {"base_doc_new": 10})


def test_accounting_rejects_an_unknown_bucket_and_negative_counts():
    with pytest.raises(AccountingError):
        accounting(1, {"base_doc_new": 1, "mystery": 0})
    with pytest.raises(AccountingError):
        accounting(1, {"base_doc_new": 2, "dropped": -1})


# -- official queries -----------------------------------------------------------------------------------------------

def _queries(rows):
    build, _, docs = _build(nom_frame(*rows))
    return official_queries(build.rows, docs), docs


def test_official_queries_map_to_the_right_doc_in_glued_and_spaced_forms():
    rows = [
        row(1, "KR 932B # 946 - 931 AP 302", interior="AP 302", cons2="KR 932B 946 931 AP 302"),
        row(2, "CL 947 # 931 - 912"),
    ]
    queries, docs = _queries(rows)
    id_a = int(docs.loc[docs["direccion"] == "KR 932 B # 946 - 931", "doc_id"].iloc[0])
    id_b = int(docs.loc[docs["direccion"] == "CL 947 # 931 - 912", "doc_id"].iloc[0])
    got = set(queries)
    for text in ("KR 932B # 946 - 931 AP 302", "KR 932 B # 946 - 931 AP 302",
                 "KR 932B 946 931 AP 302", "KR 932 B 946 931 AP 302",
                 "KR 932B # 946 - 931", "KR 932 B # 946 - 931"):
        assert (text, id_a) in got, text
    assert ("CL 947 # 931 - 912", id_b) in got
    assert all(text != "KR 932B # 946 - 931 AP 302" or d == id_a for text, d in queries)


def test_official_queries_are_deduplicated_and_deterministic():
    rows = [row(i, "CL 947 # 931 - 912", cons2="CL 947 931 912") for i in (1, 2, 3)]
    queries, _ = _queries(rows)
    assert len(queries) == len(set(queries))
    assert queries == _queries(rows)[0]
    assert queries == _queries(list(reversed(rows)))[0]


def test_official_queries_without_nddircons2_derive_a_punctuation_free_form():
    queries, _ = _queries([row(1, "CL 947 # 931 - 912", cons2="")])
    assert "CL 947 931 912" in {text for text, _ in queries}


def test_official_queries_include_merged_pcer_rows_under_the_winning_doc():
    rows = [row(1, "CL 947 # 931 - 912", estado="cert"), row(2, "CL 947 # 931 - 912 AP 5", interior="AP 5", estado="pcer")]
    queries, docs = _queries(rows)
    assert len(docs) == 1
    assert ("CL 947 # 931 - 912 AP 5", 0) in set(queries)


def test_official_queries_skip_dropped_rows_and_empty_input():
    queries, _ = _queries([row(1, "CL 947 # 931 - 912", lon=None)])
    assert queries == []
    empty = build_nomenclatura(nom_frame())
    assert official_queries(empty.rows, union_docs(empty.docs)) == []


def test_official_queries_feed_the_training_arrays_with_a_clean_validation_split():
    from cali_address.dataset import build_training_arrays

    rows = [row(i, f"CL {946 + i} # 931 - 912") for i in range(1, 61)]
    queries, docs = _queries(rows)
    out = build_training_arrays(docs[COMPAT], n_variants=1, seed=7, extra_queries=queries)
    assert int(out["q_source"].sum()) == len(queries)
    val_ids = set(out["q_doc"][out["q_is_val"]].tolist())
    train_ids = set(out["q_doc"][~out["q_is_val"]].tolist())
    assert val_ids and not (val_ids & train_ids)


# -- train pair remap -----------------------------------------------------------------------------------------------

def _pairs(*entries):
    return pd.DataFrame([{
        "raw_address": f"raw {i}", "address_key": f"K{i}", "dataset": "fasecolda", "gt_doc_id": gid,
        "gt_manzana": "m", "gt_predial": pred, "gt_comuna": "08", "source_row_id": f"r{i}",
    } for i, (gid, pred) in enumerate(entries)])


def _new_docs(entries):
    return catastro_key_table(catastro_docs_frame(entries))


def test_remap_follows_the_address_key_to_the_new_doc_id():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001")), ("KR 932 B # 946 - 931", npn(unit="000000002"))])
    new = _new_docs([("KR 932B # 946 - 931", npn(unit="000000002")), ("CL 946 # 931 - 912", npn(unit="000000001"))])
    pairs = _pairs((0, npn(unit="000000001")), (1, npn(unit="000000002")))
    out, report = remap_train_pairs(pairs, old, new)
    assert out["gt_doc_id"].tolist() == [1, 0]
    assert report["kept"] == 2 and report["remapped_by_key"] == 2
    assert list(out.columns) == list(pairs.columns)
    assert out["source_row_id"].tolist() == ["r0", "r1"]


def test_remap_falls_back_to_a_unique_predial_when_the_key_is_gone():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = _new_docs([("CL 950 # 931 - 912", npn(unit="000000001"))])
    out, report = remap_train_pairs(_pairs((0, npn(unit="000000001"))), old, new)
    assert out["gt_doc_id"].tolist() == [0]
    assert report["remapped_by_predial"] == 1


def test_remap_fallback_also_matches_npn_catastro():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = _new_docs([("CL 950 # 931 - 912", npn(unit="000000009"))])
    new["npn_catastro"] = npn(unit="000000001")
    out, _ = remap_train_pairs(_pairs((0, npn(unit="000000001"))), old, new)
    assert out["gt_doc_id"].tolist() == [0]


def test_remap_drops_pairs_whose_npn_is_missing_and_counts_them():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = _new_docs([("CL 950 # 931 - 912", npn(unit="000000009"))])
    out, report = remap_train_pairs(_pairs((0, npn(unit="000000001")), (0, None), (0, "")), old, new)
    assert out.empty
    assert report["dropped_no_npn"] == 3 and report["kept"] == 0


def test_remap_drops_ambiguous_predials_and_counts_them():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = _new_docs([("CL 950 # 931 - 912", npn(unit="000000001")), ("CL 951 # 931 - 912", npn(unit="000000001"))])
    out, report = remap_train_pairs(_pairs((0, npn(unit="000000001"))), old, new)
    assert out.empty and report["dropped_ambiguous"] == 1


def test_remap_handles_nan_and_out_of_range_doc_ids_and_empty_input():
    old = catastro_docs_frame([("CL 946 # 931 - 912", npn(unit="000000001"))])
    new = _new_docs([("CL 946 # 931 - 912", npn(unit="000000001"))])
    pairs = _pairs((np.nan, npn(unit="000000001")), (99, npn(unit="000000001")), (-1, npn(unit="000000001")))
    out, report = remap_train_pairs(pairs, old, new)
    assert out["gt_doc_id"].tolist() == [0, 0, 0]
    assert report["remapped_by_predial"] == 3
    empty, rep = remap_train_pairs(pairs.iloc[0:0], old, new)
    assert empty.empty and rep["kept"] == 0 and rep["input"] == 0
