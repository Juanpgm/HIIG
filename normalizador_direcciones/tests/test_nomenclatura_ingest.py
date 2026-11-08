"""Ingestion of cleaned nomenclature rows into base documents, units and accounting buckets.

All addresses are invented (numbers 900+).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nom_fixtures import nom_frame, npn, row

from cali_address.index_union import (
    BUCKETS, build_nomenclatura, build_nomenclatura_docs,
)
from cali_address.parser import _pre_normalize


def _one(docs: pd.DataFrame) -> pd.Series:
    assert len(docs) == 1, docs
    return docs.iloc[0]


# -- text of the document ------------------------------------------------------------------------

def test_glued_letters_become_the_spaced_canonical_text():
    docs, _ = build_nomenclatura_docs(nom_frame(row(1, "KR 932B # 946 - 931")))
    d = _one(docs)
    assert d["direccion"] == "KR 932 B # 946 - 931"
    assert d["parse_ok"]
    assert d["fuente"] == "nomenclatura_cert"
    assert d["estado_nomenclatura"] == "cert"


def test_interior_suffix_is_stripped_and_goes_to_the_units_table():
    nom = nom_frame(row(1, "KR 932B # 946 - 931 AP 302", interior="AP 302"))
    build = build_nomenclatura(nom)
    d = _one(build.docs)
    assert d["direccion"] == "KR 932 B # 946 - 931"
    assert build.stats["interior_strip_method"] == {"suffix": 1, "parser": 0, "none": 0, "none_unregistered_tail": 0}
    unit = _one(build.units)
    assert unit["interior"] == "AP 302"
    assert unit["direccion_original"] == "KR 932B # 946 - 931 AP 302"
    assert unit["estado"] == "cert"
    assert unit["address_key_base"] == d["doc_key"]            # the join key until doc ids are assigned
    assert build.buckets["unit"] == 1


def test_interior_that_is_not_a_suffix_falls_back_to_the_parser():
    nom = nom_frame(row(1, "CL 946 # 931 - 912 AP 302 TO 2", interior="AP 302"))
    build = build_nomenclatura(nom)
    assert _one(build.docs)["direccion"] == "CL 946 # 931 - 912"
    assert build.stats["interior_strip_method"]["parser"] == 1
    assert _one(build.units)["interior"] == "AP 302 TO 2"


def test_pcer_row_without_ndinterior_still_loses_its_complement_through_the_parser():
    nom = nom_frame(row(1, "CL 946 # 931 - 912 AP 302", estado="pcer", interior=None))
    build = build_nomenclatura(nom)
    assert _one(build.docs)["direccion"] == "CL 946 # 931 - 912"
    assert build.stats["interior_strip_method"] == {"suffix": 0, "parser": 1, "none": 0, "none_unregistered_tail": 0}
    assert len(build.units) == 1
    assert _one(build.units)["interior"] == "AP 302"


def test_row_without_interior_uses_method_none():
    build = build_nomenclatura(nom_frame(row(1, "CL 946 # 931 - 912")))
    assert build.stats["interior_strip_method"] == {"suffix": 0, "parser": 0, "none": 1, "none_unregistered_tail": 0}
    assert build.units.empty


def test_interior_equal_to_the_whole_address_does_not_empty_the_base():
    # a malformed row where ndinterior repeats the full address: the base must not become empty
    nom = nom_frame(row(1, "CL 946 # 931 - 912", interior="CL 946 # 931 - 912"))
    build = build_nomenclatura(nom)
    assert _one(build.docs)["direccion"] == "CL 946 # 931 - 912"
    assert sum(build.buckets.values()) == 1
    assert build.buckets["dropped"] == 0


def test_catastral_text_is_kept_verbatim_in_original_and_queries_table():
    nom = nom_frame(row(1, "KR 932B # 946 - 931"))
    build = build_nomenclatura(nom)
    assert _one(build.rows)["direccion_original"] == "KR 932B # 946 - 931"


# -- fallback documents (never dropped) ---------------------------------------------------------------

def test_unparseable_base_becomes_a_fallback_doc_not_a_drop():
    nom = nom_frame(row(1, "XYZ 912 QQ"))
    build = build_nomenclatura(nom)
    d = _one(build.docs)
    assert d["direccion"] == _pre_normalize("XYZ 912 QQ")[:40]
    assert not d["parse_ok"]
    assert build.buckets["fallback_doc"] == 1
    assert build.buckets["dropped"] == 0


def test_truncated_parse_becomes_a_fallback_doc():
    nom = nom_frame(row(1, "KR 933 # 940, 31"))
    build = build_nomenclatura(nom)
    d = _one(build.docs)
    assert not d["parse_ok"]
    assert d["direccion"] == _pre_normalize("KR 933 # 940, 31")[:40]
    assert build.buckets["fallback_doc"] == 1


def test_fallback_text_is_capped_at_forty_characters():
    long_text = "ZZZ " + "9" * 80
    build = build_nomenclatura(nom_frame(row(1, long_text)))
    assert len(_one(build.docs)["direccion"]) <= 40


def test_fallback_row_with_an_interior_still_lands_in_the_units_table():
    nom = nom_frame(row(1, "XYZ 912 QQ AP 302", interior="AP 302"))
    build = build_nomenclatura(nom)
    assert build.buckets["fallback_doc"] == 1
    assert len(build.units) == 1


# -- drops --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lon,lat", [(None, 3.4), (-76.5, None), (None, None), (np.inf, 3.4)])
def test_row_without_coordinates_is_dropped_and_counted(lon, lat):
    nom = nom_frame(row(1, "CL 946 # 931 - 912", lon=lon, lat=lat), row(2, "CL 947 # 931 - 912"))
    build = build_nomenclatura(nom)
    assert build.buckets["dropped"] == 1
    assert len(build.docs) == 1
    assert build.dropped["reason"].tolist() == ["no_coordinates"]
    assert build.dropped["gid"].tolist() == [1]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_address_is_dropped_and_counted(text):
    nom = nom_frame(row(1, "CL 946 # 931 - 912"))
    nom.loc[0, "nddirecc"] = text
    build = build_nomenclatura(nom)
    assert build.docs.empty
    assert build.dropped["reason"].tolist() == ["empty_address"]
    assert build.buckets["dropped"] == 1


def test_symbols_only_address_is_dropped_as_empty():
    build = build_nomenclatura(nom_frame(row(1, "### ---")))
    assert build.buckets["dropped"] == 1
    assert build.dropped["reason"].tolist() == ["empty_address"]


# -- accounting: every record in exactly one bucket ------------------------------------------------------

def test_every_record_lands_in_exactly_one_bucket():
    nom = nom_frame(
        row(1, "KR 932B # 946 - 931"),                                  # base_doc_new
        row(2, "KR 932 B # 946 - 931"),                                 # base_doc_merged (same key)
        row(3, "KR 932B # 946 - 931 AP 302", interior="AP 302"),        # unit
        row(4, "XYZ 912 QQ"),                                           # fallback_doc
        row(5, "CL 946 # 931 - 912", lon=None),                         # dropped
        row(6, "CL 948 # 931 - 912", estado="pcer"),                    # excluded tier (cert only)
    )
    build = build_nomenclatura(nom, tiers=("cert",))
    assert build.buckets == {
        "base_doc_new": 1, "base_doc_merged": 1, "unit": 1, "fallback_doc": 1, "dropped": 1, "excluded_tier": 1,
        "incomplete_address": 0, "spread_outlier": 0,
    }
    assert set(build.buckets) == set(BUCKETS)
    assert sum(build.buckets.values()) == len(nom)


def test_unknown_estado_is_excluded_not_lost():
    build = build_nomenclatura(nom_frame(row(1, "CL 946 # 931 - 912", estado="lote")))
    assert build.buckets["excluded_tier"] == 1
    assert build.docs.empty


def test_empty_frame_gives_empty_docs_and_zero_buckets():
    build = build_nomenclatura(nom_frame())
    assert build.docs.empty and build.units.empty
    assert sum(build.buckets.values()) == 0


def test_missing_required_columns_fail_with_a_clear_message():
    nom = nom_frame(row(1, "CL 946 # 931 - 912")).drop(columns=["ndnpn"])
    with pytest.raises(ValueError, match="ndnpn"):
        build_nomenclatura(nom)


# -- NPN validity ------------------------------------------------------------------------------------

@pytest.mark.parametrize("length", [28, 29])
def test_short_npn_is_kept_with_npn_valid_false(length):
    short = npn()[:length]
    build = build_nomenclatura(nom_frame(row(1, "CL 946 # 931 - 912", npn_value=short)))
    d = _one(build.docs)
    assert d["numero_predial_nacional"] == short
    assert build.stats["npn_valid_docs"] == 0
    assert build.buckets["base_doc_new"] == 1


def test_invalid_npn_row_falls_back_to_wfs_territory_fields():
    short = npn()[:29]
    nom = nom_frame(row(1, "CL 946 # 931 - 912", npn_value=short, comuna="14", barrio="1405", manzana="0777"))
    d = _one(build_nomenclatura_docs(nom)[0])
    assert (d["comuna"], d["barrio_code"], d["barrio"]) == ("14", "1405", "05")


def test_npn_derived_territory_wins_over_wfs_fields_and_disagreements_are_counted():
    value = npn(comuna="08", barrio="10", manzana="0956")
    nom = nom_frame(row(1, "CL 946 # 931 - 912", npn_value=value, comuna="09", barrio="0915", manzana="0001"))
    build = build_nomenclatura(nom)
    d = _one(build.docs)
    assert (d["comuna"], d["barrio"], d["barrio_code"], d["manzana"]) == ("08", "10", "0810", value[:17])
    assert build.stats["disagreements"] == {"comuna": 1, "barrio": 1, "manzana": 1, "invalid_npn_fallback": 0}


def test_agreeing_territory_fields_report_zero_disagreements():
    build = build_nomenclatura(nom_frame(row(1, "CL 946 # 931 - 912")))
    assert build.stats["disagreements"] == {"comuna": 0, "barrio": 0, "manzana": 0, "invalid_npn_fallback": 0}


def test_blank_wfs_territory_fields_are_not_counted_as_disagreements():
    nom = nom_frame(row(1, "CL 946 # 931 - 912"))
    for col in ("ndcomuna", "ndidbarrio", "ndmanzana"):
        nom[col] = None
    build = build_nomenclatura(nom)
    assert build.stats["disagreements"]["comuna"] == 0
    assert _one(build.docs)["comuna"] == "08"


# -- expansion zone, determinism ----------------------------------------------------------------------

def test_zone_flag_is_carried_to_the_doc():
    nom = nom_frame(row(1, "CL 946 # 931 - 912", zona="EXP"))
    assert _one(build_nomenclatura_docs(nom)[0])["zona"] == "EXP"


def test_input_order_does_not_change_the_result():
    rows = [row(i, f"CL {946 + i} # 931 - 912") for i in range(1, 6)] + [row(9, "KR 932B # 946 - 931")]
    a = build_nomenclatura_docs(nom_frame(*rows))[0]
    b = build_nomenclatura_docs(nom_frame(*reversed(rows)))[0]
    pd.testing.assert_frame_equal(a, b)


def test_two_runs_are_identical():
    nom = nom_frame(*[row(i, f"CL {946 + i % 3} # 931 - 91{i % 7}") for i in range(1, 30)])
    pd.testing.assert_frame_equal(build_nomenclatura_docs(nom)[0], build_nomenclatura_docs(nom)[0])


def test_a_bad_row_never_raises():
    nom = nom_frame(row(1, "CL 946 # 931 - 912"), row(2, "\x00\x01 ??"), row(3, "9" * 400), row(4, "KR , , ,"))
    nom.loc[1, "ndnpn"] = None
    build = build_nomenclatura(nom)
    assert sum(build.buckets.values()) == 4
