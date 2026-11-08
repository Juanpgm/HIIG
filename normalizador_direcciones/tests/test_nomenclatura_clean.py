"""Phase 0.2b: cleaning of the WFS nomenclature download, rule by rule.

Every row below is invented. Each rule has its own test plus its edge cases; none of them may raise
for a single bad row.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import nomenclatura  # noqa: E402
from cali_address import nomenclatura_clean as nc  # noqa: E402

WFS_FIELDS = nomenclatura.WFS_FIELDS


def row(i: int = 0, **over) -> dict:
    r = {k: None for k in WFS_FIELDS}
    r.update({
        "gid": i + 1,
        "ndid": 900000 + i,
        "ndcomuna": "03",
        "ndidbarrio": "0309",
        "ndbarrio": "Barrio Inventado",
        "ndidpredio": 700000 + i,
        "ndnpn": f"76001010003090015002100000{i:04d}",
        "ndmanzana": "0015",
        "nddirecc": "KR 15 # 12 - 83",
        "ndnomvial": "KR 15",
        "ndidtipvia": "KR",
        "ndtipovia": "Carrera",
        "ndnumvia": "15",
        "ndplaca": "12 - 83",
        "ndidestado": "cert",
        "ndestado": "Certificada",
        "ndfecha": "2021-08-21Z",
        "ndidtipzon": "URB",
        "ndtipozona": "Urbano",
        "nddircons2": "KR 15 12 83",
        "ndcoordx": 1060967.2254,
        "ndcoordy": 872476.141,
        "ndfechcort": "2026-09-03Z",
        "lon": -76.5 + i * 1e-5,
        "lat": 3.44 + i * 1e-5,
    })
    r.update(over)
    return r


def pcer_row(i: int = 0, **over) -> dict:
    base = dict(ndidestado="pcer", ndnomvial=None, ndidtipvia=None, ndtipovia=None, ndnumvia=None,
                ndplaca=None, nddirecc="CL 977 E # 931 F 6 - 953 (Dirección Catastral)")
    base.update(over)
    return row(i, **base)


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(WFS_FIELDS) + ["lon", "lat"])
    for c in nomenclatura.WFS_INT_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in nomenclatura.WFS_FLOAT_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def clean(rows, **kw):
    return nc.clean_nomenclatura(frame(rows), **kw)


def one(rows_or_row, **kw):
    """Clean a single row and return (clean_row or None, quarantine_reason or None, result)."""
    res = clean([rows_or_row] if isinstance(rows_or_row, dict) else rows_or_row, **kw)
    c = res.clean.iloc[0] if len(res.clean) else None
    q = res.quarantine["quarantine_reason"].iloc[0] if len(res.quarantine) else None
    return c, q, res


# ------------------------------------------------------------------ text rule

@pytest.mark.parametrize("raw,expected", [
    ("  kr 15   #  12 -  83 ", "KR 15 # 12 - 83"),
    ("KR\t15\n#\r12 - 83", "KR 15 # 12 - 83"),
    ("KR 15 # 12 - 83", "KR 15 # 12 - 83"),
    ("cl 9 # 4 - 1", "CL 9 # 4 - 1"),
])
def test_text_strip_collapse_uppercase(raw, expected):
    text, _ = nc.clean_address_text(raw)
    assert text == expected


@pytest.mark.parametrize("raw", [
    "KR 15 N° 12 - 83",       # N + degree sign
    "KR 15 Nº 12 - 83",       # N + ordinal indicator
    "KR 15 Nº. 12 - 83",
    "KR 15 NO. 12 - 83",
    "KR 15 no. 12 - 83",
    "KR 15 № 12 - 83",        # numero sign
])
def test_number_sign_variants_become_hash(raw):
    text, flags = nc.clean_address_text(raw)
    assert text == "KR 15 # 12 - 83"
    assert flags["number_sign"] is True


@pytest.mark.parametrize("raw", [
    "CL 10 NO 5 - 3",              # NO without a period is the north-west quadrant / ambiguous
    "KR 5 NORTE # 3 - 4",
    "CL 5 NO654",                  # documented glued-number case (CHANGELOG round 7)
    "AV 3 NO. SUR # 4 - 5",        # "NO." not followed by a digit is not a number sign
    "KR 15 # 12 - 83",
])
def test_number_sign_is_not_touched_where_it_is_not_one(raw):
    text, flags = nc.clean_address_text(raw)
    assert text == raw
    assert flags["number_sign"] is False


@pytest.mark.parametrize("marker", [
    "(Dirección Catastral)", "(DIRECCION CATASTRAL)", "(direccion catastral)",
    "[Dirección Catastral]", "( Dirección   Catastral )", "(Dir. Catastral)", "(Catastral)",
    "(Dirección Catastral)",   # decomposed accent
])
def test_catastral_marker_is_removed_case_and_accent_insensitive(marker):
    text, flags = nc.clean_address_text(f"CL 977 E # 931 F 6 - 953 {marker}")
    assert text == "CL 977 E # 931 F 6 - 953"
    assert flags["marker_removed"] is True


def test_marker_flag_is_false_without_a_marker():
    _, flags = nc.clean_address_text("CL 977 # 931 - 953")
    assert flags["marker_removed"] is False


def test_marker_without_closing_parenthesis_is_still_removed():
    text, flags = nc.clean_address_text("CL 977 # 931 - 953 (Dirección Catastral")
    assert text == "CL 977 # 931 - 953" and flags["marker_removed"] is True


def test_control_characters_and_zero_width_are_dropped():
    text, flags = nc.clean_address_text("KR\x00 15 #​ 12﻿ - 83\x07")
    assert text == "KR 15 # 12 - 83"
    assert flags["control_chars"] is True


@pytest.mark.parametrize("raw", ['KR 15 # "12" - 83', "KR 15 # “12” - 83", "KR 15 # 12' - 83", "`KR 15 # 12 - 83`"])
def test_stray_quotes_are_removed(raw):
    text, flags = nc.clean_address_text(raw)
    assert text == "KR 15 # 12 - 83"
    assert flags["quotes"] is True


def test_unicode_accents_survive_and_uppercase():
    text, _ = nc.clean_address_text("cl 5 # 12 - 3 peña blanca áé")
    assert text == "CL 5 # 12 - 3 PEÑA BLANCA ÁÉ"


@pytest.mark.parametrize("raw", [None, np.nan, pd.NA, "", "   ", "\t\n", "\x00\x01"])
def test_null_like_values_become_none(raw):
    text, flags = nc.clean_address_text(raw)
    assert text is None
    assert flags["marker_removed"] is False


def test_non_string_values_never_raise():
    assert nc.clean_address_text(12345)[0] == "12345"
    assert nc.clean_address_text(b"KR 1")[0] is not None
    assert nc.clean_address_text({"a": 1})[0] is not None
    assert nc.clean_address_text([1, 2])[0] is not None


@pytest.mark.parametrize("raw", ["  kr 15 N° 12 - 83 (Dirección Catastral)", "KR 15 # 12 - 83", 'CL "5" # 1'])
def test_text_cleaning_is_idempotent(raw):
    once, _ = nc.clean_address_text(raw)
    twice, flags = nc.clean_address_text(once)
    assert twice == once
    assert not any(flags.values())


def test_marker_in_a_pcer_row_is_removed_and_counted():
    c, q, res = one(pcer_row())
    assert q is None
    assert c["nddirecc"] == "CL 977 E # 931 F 6 - 953"
    assert bool(c["marker_removed"]) is True
    assert res.report["rules"]["marker_removed"] == 1


# ------------------------------------------------------------------ consistency rule

def test_consistent_address_is_not_rebuilt_and_keeps_original():
    c, q, res = one(row(nddirecc="kr 15  #  12 - 83"))
    assert q is None
    assert c["nddirecc"] == "KR 15 # 12 - 83"
    assert bool(c["rebuilt_from_parts"]) is False
    assert c["nddirecc_original"] == "kr 15  #  12 - 83"
    assert res.report["rules"]["rebuilt_from_parts"] == 0


def test_mismatch_is_rebuilt_from_components_and_original_is_kept():
    c, q, res = one(row(nddirecc="KR 99 # 1 - 1"))
    assert q is None
    assert c["nddirecc"] == "KR 15 # 12 - 83"
    assert bool(c["rebuilt_from_parts"]) is True
    assert c["nddirecc_original"] == "KR 99 # 1 - 1"
    assert res.report["rules"]["rebuilt_from_parts"] == 1


def test_interior_is_part_of_the_expected_address():
    r = row(ndinterior="ED 47 AP 302", nddirecc="KR 15 # 12 - 83 ED 47 AP 302")
    c, _, _ = one(r)
    assert bool(c["rebuilt_from_parts"]) is False
    r2 = row(ndinterior="ED 47 AP 302", nddirecc="KR 15 # 12 - 83")
    c2, _, _ = one(r2)
    assert c2["nddirecc"] == "KR 15 # 12 - 83 ED 47 AP 302"
    assert bool(c2["rebuilt_from_parts"]) is True


def test_unit_suffix_in_nddirecc_without_interior_component_is_not_a_mismatch():
    # Real pcer rows carry the unit in nddirecc while ndinterior is null: rebuilding would drop it.
    c, _, res = one(row(nddirecc="KR 15 # 12 - 83 AP 302 (Dirección Catastral)"))
    assert c["nddirecc"] == "KR 15 # 12 - 83 AP 302"
    assert bool(c["rebuilt_from_parts"]) is False
    assert res.report["rules"]["rebuilt_from_parts"] == 0


def test_plate_that_merely_starts_like_the_component_is_a_mismatch():
    # "12 - 831" is not "12 - 83" plus a suffix: the extra digit belongs to the plate.
    c, _, _ = one(row(nddirecc="KR 15 # 12 - 831"))
    assert c["nddirecc"] == "KR 15 # 12 - 83" and bool(c["rebuilt_from_parts"]) is True


def test_address_shorter_than_the_components_is_rebuilt():
    c, _, _ = one(row(nddirecc="KR 15 # 12"))
    assert c["nddirecc"] == "KR 15 # 12 - 83" and bool(c["rebuilt_from_parts"]) is True


def test_difference_only_in_spacing_is_not_a_mismatch():
    c, _, _ = one(row(nddirecc="KR 15#12-83"))
    assert bool(c["rebuilt_from_parts"]) is False
    assert c["nddirecc"] == "KR 15#12-83"


def test_missing_components_use_cleaned_nddirecc():
    c, q, _ = one(pcer_row(nddirecc="  cl 5 # 4 - 3 "))
    assert q is None and c["nddirecc"] == "CL 5 # 4 - 3"
    assert bool(c["rebuilt_from_parts"]) is False


@pytest.mark.parametrize("field", ["ndplaca", "ndnomvial"])
def test_a_single_missing_component_counts_as_no_components(field):
    over = {field: None}
    if field == "ndnomvial":
        over["ndnumvia"] = None
    c, q, _ = one(row(nddirecc="KR 99 # 1 - 1", **over))
    assert q is None and c["nddirecc"] == "KR 99 # 1 - 1"
    assert bool(c["rebuilt_from_parts"]) is False


def test_via_is_built_from_type_code_and_number_when_ndnomvial_is_null():
    c, _, _ = one(row(ndnomvial=None, nddirecc="KR 99 # 1 - 1"))
    assert c["nddirecc"] == "KR 15 # 12 - 83"


def test_null_nddirecc_with_components_is_rebuilt():
    c, q, _ = one(row(nddirecc=None))
    assert q is None and c["nddirecc"] == "KR 15 # 12 - 83" and bool(c["rebuilt_from_parts"]) is True


def test_no_address_at_all_is_quarantined():
    c, q, res = one(pcer_row(nddirecc=None))
    assert c is None and q == "empty_address"
    assert res.report["rules"]["empty_address"] == 1


# ------------------------------------------------------------------ NPN rule

def test_valid_npn_30_digits_with_prefix():
    c, _, _ = one(row())
    assert bool(c["npn_valid"]) is True and int(c["npn_len"]) == 30


@pytest.mark.parametrize("npn,length", [
    ("7600101000309001500210000001", 28),
    ("76001010003090015002100000001", 29),
])
def test_short_npns_are_kept_with_npn_valid_false(npn, length):
    c, q, res = one(row(ndnpn=npn))
    assert q is None
    assert c["ndnpn"] == npn and bool(c["npn_valid"]) is False and int(c["npn_len"]) == length
    assert res.report["rules"]["npn_invalid"] == 1
    assert res.report["rules"][f"npn_len_{length}"] == 1


def test_npn_with_wrong_prefix_or_non_digits_is_invalid_but_kept():
    wrong_prefix = "85001010003090015002100000001"[:30].ljust(30, "0")
    c, q, _ = one(row(ndnpn=wrong_prefix))
    assert q is None and bool(c["npn_valid"]) is False
    c2, q2, _ = one(row(ndnpn="76001" + "A" * 25))
    assert q2 is None and bool(c2["npn_valid"]) is False


def test_npn_is_stripped_before_validation():
    c, _, _ = one(row(ndnpn="  760010100030900150021000000001\t"))
    assert c["ndnpn"] == "760010100030900150021000000001" and bool(c["npn_valid"]) is True


def test_null_or_empty_npn_is_kept_as_invalid():
    for v in (None, "", "   "):
        c, q, _ = one(row(ndnpn=v))
        assert q is None and bool(c["npn_valid"]) is False and int(c["npn_len"]) == 0
        assert pd.isna(c["ndnpn"]) or c["ndnpn"] is None


# ------------------------------------------------------------------ duplicates rule

def test_duplicate_gid_keeps_the_latest_ndfecha_and_quarantines_the_other():
    rows = [row(0, gid=7, ndfecha="2020-01-01Z", nddirecc="KR 15 # 12 - 83"),
            row(1, gid=7, ndfecha="2022-01-01Z")]
    res = clean(rows)
    assert len(res.clean) == 1 and res.clean["ndid"].iloc[0] == 900001
    assert res.quarantine["quarantine_reason"].iloc[0] == "duplicate_gid"
    assert res.report["rules"]["duplicate_gid"] == 1


def test_duplicate_ndid_with_different_content_keeps_latest_and_counts():
    rows = [row(0, ndid=555, ndfecha="2019-05-05Z", gid=1),
            row(1, ndid=555, ndfecha="2023-05-05Z", gid=2, nddirecc="KR 16 # 1 - 1", ndnumvia="16",
                 ndnomvial="KR 16", ndplaca="1 - 1")]
    res = clean(rows)
    assert list(res.clean["gid"]) == [2]
    assert res.quarantine["quarantine_reason"].iloc[0] == "duplicate_ndid"
    assert res.report["rules"]["duplicate_ndid_conflicting"] == 1
    assert res.report["rules"]["duplicate_ndid_identical"] == 0


def test_duplicate_ndid_with_identical_content_is_counted_separately():
    a = row(0, ndid=555, gid=1)
    b = dict(a, gid=2)
    res = clean([a, b])
    assert len(res.clean) == 1
    assert res.report["rules"]["duplicate_ndid_identical"] == 1
    assert res.report["rules"]["duplicate_ndid_conflicting"] == 0


def test_ndid_zero_and_null_are_never_keys():
    rows = [row(0, ndid=0, gid=1), row(1, ndid=0, gid=2), row(2, ndid=None, gid=3), row(3, ndid=None, gid=4)]
    res = clean(rows)
    assert len(res.clean) == 4 and len(res.quarantine) == 0


def test_latest_ndfecha_tie_keeps_the_first_row_and_unparseable_dates_lose():
    tie = clean([row(0, ndid=9, gid=1), row(1, ndid=9, gid=2)])
    assert list(tie.clean["gid"]) == [1]
    bad = clean([row(0, ndid=9, gid=1, ndfecha="garbage"), row(1, ndid=9, gid=2, ndfecha="2001-01-01Z")])
    assert list(bad.clean["gid"]) == [2]
    none = clean([row(0, ndid=9, gid=1, ndfecha=None), row(1, ndid=9, gid=2, ndfecha="2001-01-01Z")])
    assert list(none.clean["gid"]) == [2]


def test_an_invalid_newer_duplicate_does_not_evict_the_valid_older_one():
    rows = [row(0, ndid=9, gid=1, ndfecha="2001-01-01Z"),
            row(1, ndid=9, gid=2, ndfecha="2025-01-01Z", lon=np.nan, lat=np.nan)]
    res = clean(rows)
    assert list(res.clean["gid"]) == [1]
    assert res.quarantine["quarantine_reason"].iloc[0] == "coords_nan"


def test_null_gid_rows_are_not_deduplicated():
    res = clean([row(0, gid=None), row(1, gid=None, ndid=9001)])
    assert len(res.clean) == 2


# ------------------------------------------------------------------ coordinates rule

@pytest.mark.parametrize("lon,lat", [(np.nan, 3.44), (-76.5, np.nan), (np.nan, np.nan), (np.inf, 3.44), (-76.5, -np.inf)])
def test_non_finite_coordinates_are_quarantined(lon, lat):
    c, q, res = one(row(lon=lon, lat=lat))
    assert c is None and q == "coords_nan"
    assert res.report["rules"]["coords_nan"] == 1


@pytest.mark.parametrize("lon,lat", [(-75.0, 3.44), (-76.5, 4.9), (3.44, -76.5), (0.0, 0.0), (-76.6501, 3.44), (-76.5, 3.5501)])
def test_points_outside_the_cali_bbox_are_quarantined(lon, lat):
    c, q, res = one(row(lon=lon, lat=lat))
    assert c is None and q == "coords_outside_bbox"
    assert res.report["rules"]["coords_outside_bbox"] == 1


@pytest.mark.parametrize("lon,lat", [(-76.65, 3.30), (-76.40, 3.55), (-76.52, 3.44)])
def test_points_on_the_bbox_edge_are_kept(lon, lat):
    c, q, _ = one(row(lon=lon, lat=lat))
    assert q is None and c is not None


def test_shapefile_shift_hook_quarantines_only_above_one_metre():
    shift = pd.Series({900000: 1.0, 900001: 1.0001, 900002: 25.0}, name="shift_m")
    res = clean([row(0), row(1), row(2), row(3)], shp_shift=shift)
    assert sorted(res.clean["ndid"]) == [900000, 900003]  # 900003 has no shapefile twin: kept
    reasons = dict(zip(res.quarantine["ndid"], res.quarantine["quarantine_reason"]))
    assert reasons == {900001: "coords_shift_vs_shp", 900002: "coords_shift_vs_shp"}
    assert res.report["rules"]["coords_shift_vs_shp"] == 2


def test_shapefile_shift_hook_off_by_default():
    res = clean([row(0)])
    assert res.report["rules"]["coords_shift_vs_shp"] == 0


def test_shapefile_shift_with_nan_values_does_not_quarantine():
    res = clean([row(0)], shp_shift=pd.Series({900000: np.nan}))
    assert len(res.clean) == 1


# ------------------------------------------------------------------ status rule

@pytest.mark.parametrize("status", ["cert", "pcer", " CERT ", "Pcer"])
def test_cert_and_pcer_pass(status):
    c, q, _ = one(row(ndidestado=status))
    assert q is None and c["ndidestado"] in ("cert", "pcer")


@pytest.mark.parametrize("status", ["dver", "anul", "cert2", "", None])
def test_any_other_status_goes_to_quarantine_with_the_literal(status):
    c, q, res = one(row(ndidestado=status))
    assert c is None
    literal = "<null>" if status in (None, "") else status
    assert q == f"status_invalid:{literal}"
    assert res.report["rules"]["status_invalid"] == 1
    assert res.report["status_invalid_literals"] == {literal: 1}


# ------------------------------------------------------------------ schema rule

def test_missing_column_aborts_listing_every_missing_name():
    df = frame([row(0)]).drop(columns=["ndnpn", "ndplaca"])
    with pytest.raises(nc.SchemaError) as exc:
        nc.clean_nomenclatura(df)
    assert "ndnpn" in str(exc.value) and "ndplaca" in str(exc.value)


def test_missing_lon_lat_is_a_schema_error():
    df = frame([row(0)]).drop(columns=["lon"])
    with pytest.raises(nc.SchemaError, match="lon"):
        nc.clean_nomenclatura(df)


def test_extra_columns_are_recorded_and_ignored():
    df = frame([row(0)])
    df["surprise"] = "x"
    res = nc.clean_nomenclatura(df)
    assert res.report["schema"]["extra_columns"] == ["surprise"]
    assert "surprise" not in res.clean.columns
    assert len(res.clean) == 1


def test_string_typed_numeric_columns_are_coerced_not_rejected():
    df = frame([row(0), row(1)])
    df["gid"] = df["gid"].astype(str)
    df["ndcoordx"] = df["ndcoordx"].astype(str)
    res = nc.clean_nomenclatura(df)
    assert len(res.clean) == 2
    assert set(res.report["schema"]["coerced_columns"]) >= {"gid", "ndcoordx"}


def test_unparseable_numeric_values_become_null_without_raising():
    df = frame([row(0), row(1)])
    df["ndid"] = pd.Series(["12", "not-a-number"], dtype="object")
    res = nc.clean_nomenclatura(df)
    assert len(res.clean) + len(res.quarantine) == 2


def test_input_dataframe_is_not_mutated():
    df = frame([row(0, nddirecc="  kr 15 # 12 - 83 ")])
    before = df.copy(deep=True)
    nc.clean_nomenclatura(df)
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------------------------ whole-frame behaviour

def test_empty_dataframe_with_the_schema_is_not_an_error():
    res = nc.clean_nomenclatura(frame([]))
    assert len(res.clean) == 0 and len(res.quarantine) == 0
    assert res.report["input_rows"] == 0 and res.report["quarantine_share"] == 0.0
    assert res.report["gate_passed"] is True
    assert "nddirecc_original" in res.clean.columns and "quarantine_reason" in res.quarantine.columns


def test_dataframe_without_columns_is_a_schema_error():
    with pytest.raises(nc.SchemaError):
        nc.clean_nomenclatura(pd.DataFrame())


def test_everything_quarantined():
    res = clean([row(i, ndidestado="dver") for i in range(5)])
    assert len(res.clean) == 0 and len(res.quarantine) == 5
    assert res.report["quarantine_share"] == 1.0 and res.report["gate_passed"] is False


def test_row_of_nulls_everywhere_is_quarantined_with_every_applicable_reason():
    nulls = {k: None for k in WFS_FIELDS}
    nulls.update(lon=np.nan, lat=np.nan)
    res = nc.clean_nomenclatura(frame([nulls]))
    reason = res.quarantine["quarantine_reason"].iloc[0]
    assert set(reason.split(";")) == {"status_invalid:<null>", "coords_nan", "empty_address"}


def test_row_hitting_several_rules_records_all_reasons_and_counts_each():
    r = pcer_row(0, ndidestado="dver", lon=np.nan, lat=np.nan, ndnpn="7600101")
    res = clean([r])
    reasons = set(res.quarantine["quarantine_reason"].iloc[0].split(";"))
    assert reasons == {"status_invalid:dver", "coords_nan"}
    rules = res.report["rules"]
    assert rules["status_invalid"] == 1 and rules["coords_nan"] == 1 and rules["npn_invalid"] == 1
    assert rules["marker_removed"] == 1
    assert res.report["quarantined_rows"] == 1  # one row, however many rules it hit


def test_quarantined_rows_keep_their_raw_values():
    res = clean([pcer_row(0, ndidestado="dver")])
    q = res.quarantine.iloc[0]
    assert "Dirección Catastral" in q["nddirecc"]  # raw, uncleaned


def test_row_counts_always_add_up():
    rows = [row(0), row(1, ndidestado="dver"), row(2, lon=np.nan), pcer_row(3), row(4, gid=1)]
    res = clean(rows)
    assert len(res.clean) + len(res.quarantine) == len(rows) == res.report["input_rows"]
    assert res.report["clean_rows"] == len(res.clean)


def test_cleaning_is_idempotent_on_its_own_output():
    rows = [row(0, nddirecc="  kr 15  N° 12 - 83 "), pcer_row(1), row(2, nddirecc="KR 99 # 1 - 1")]
    first = clean(rows).clean
    again = nc.clean_nomenclatura(first.drop(columns=["nddirecc_original", "marker_removed", "rebuilt_from_parts",
                                                      "npn_valid", "npn_len"]))
    assert list(again.clean["nddirecc"]) == list(first["nddirecc"])
    assert not again.clean["rebuilt_from_parts"].any() and not again.clean["marker_removed"].any()


def test_input_order_is_preserved_among_clean_rows():
    res = clean([row(5), row(1), row(3)])
    assert list(res.clean["ndid"]) == [900005, 900001, 900003]


def test_quarantine_share_gate_boundary():
    ok = nc.clean_nomenclatura(frame([row(i) for i in range(199)] + [row(900, ndidestado="dver")]))  # 1/200 = 0.5 %
    assert ok.report["gate_passed"] is True
    bad = nc.clean_nomenclatura(frame([row(i) for i in range(198)] + [row(900, ndidestado="dver"), row(901, ndidestado="dver")]))
    assert bad.report["gate_passed"] is False


def test_report_is_json_serialisable():
    res = clean([row(0), row(1, ndidestado="dver"), pcer_row(2)])
    json.dumps(res.report)


# ------------------------------------------------------------------ script

def _load_script():
    spec = importlib.util.spec_from_file_location("script_clean_nomenclatura", os.path.join(ROOT, "scripts", "clean_nomenclatura.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PRJ_WKT = (
    'PROJCS["MAGNA_Cali_Valle_del_Cauca_2009",GEOGCS["GCS_MAGNA",DATUM["D_MAGNA",'
    'SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["IGAC_Plano_Cartesiano"],'
    'PARAMETER["False_Easting",1061900.18],PARAMETER["False_Northing",872364.63],'
    'PARAMETER["Longitude_Of_Center",-76.5205625],PARAMETER["Latitude_Of_Center",3.441883333333334],'
    'PARAMETER["Height",1000.0],UNIT["Meter",1.0]]'
)


def _prj(tmp_path) -> str:
    path = tmp_path / "grid.prj"
    path.write_text(_PRJ_WKT, encoding="utf-8")
    return str(path)


def _run_script(tmp_path, rows, *extra):
    src = tmp_path / "wfs.parquet"
    frame(rows).to_parquet(src, index=False)
    mod = _load_script()
    out = tmp_path / "clean.parquet"
    rc = mod.main(["--in", str(src), "--out", str(out), "--quarantine", str(tmp_path / "q.parquet"),
                   "--report", str(tmp_path / "report.json"), "--prj", _prj(tmp_path), *extra])
    return rc, out


def test_script_writes_clean_quarantine_and_report(tmp_path):
    rows = [row(i) for i in range(300)] + [row(900, ndidestado="dver")]
    rc, out = _run_script(tmp_path, rows)
    assert rc == 0
    assert len(pd.read_parquet(out)) == 300
    q = pd.read_parquet(tmp_path / "q.parquet")
    assert len(q) == 1 and q["quarantine_reason"].iloc[0] == "status_invalid:dver"
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert rep["input_rows"] == 301 and rep["quarantined_rows"] == 1 and rep["rules"]["status_invalid"] == 1


def test_script_returns_three_and_warns_loudly_over_the_gate(tmp_path, capsys):
    rows = [row(i) for i in range(10)] + [row(900, ndidestado="dver")]
    rc, out = _run_script(tmp_path, rows)
    assert rc == 3
    assert "WARNING" in capsys.readouterr().err
    assert out.exists() and (tmp_path / "report.json").exists()  # outputs are still written for inspection


def test_script_missing_input_is_a_clear_error(tmp_path, capsys):
    mod = _load_script()
    rc = mod.main(["--in", str(tmp_path / "nope.parquet"), "--out", str(tmp_path / "o.parquet"),
                   "--quarantine", str(tmp_path / "q.parquet"), "--report", str(tmp_path / "r.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "nope.parquet" in err and "Traceback" not in err


def test_script_schema_error_lists_columns_without_traceback(tmp_path, capsys):
    df = frame([row(0)]).drop(columns=["ndnpn"])
    src = tmp_path / "wfs.parquet"
    df.to_parquet(src, index=False)
    mod = _load_script()
    rc = mod.main(["--in", str(src), "--out", str(tmp_path / "o.parquet"),
                   "--quarantine", str(tmp_path / "q.parquet"), "--report", str(tmp_path / "r.json"),
                   "--prj", _prj(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ndnpn" in err and "Traceback" not in err
    assert not (tmp_path / "o.parquet").exists()


def test_script_empty_input_succeeds(tmp_path):
    rc, out = _run_script(tmp_path, [])
    assert rc == 0 and len(pd.read_parquet(out)) == 0
    assert len(pd.read_parquet(tmp_path / "q.parquet")) == 0


def test_script_shp_shift_option_quarantines_far_points(tmp_path):
    rows = [row(i) for i in range(300)]
    shp = pd.DataFrame({
        "NdID": [900000 + i for i in range(300)], "NdEstado": ["cert"] * 300,
        "NdNpn": [r["ndnpn"] for r in rows], "NdDirecc": [r["nddirecc"] for r in rows],
        "lon": [r["lon"] for r in rows], "lat": [r["lat"] + (0.001 if i == 7 else 0.0) for i, r in enumerate(rows)],
    })
    shp_path = tmp_path / "shp.parquet"
    shp.to_parquet(shp_path)
    rc, out = _run_script(tmp_path, rows, "--shp-4326", str(shp_path))
    assert rc == 0
    q = pd.read_parquet(tmp_path / "q.parquet")
    assert q["quarantine_reason"].tolist() == ["coords_shift_vs_shp"] and q["ndid"].tolist() == [900007]
