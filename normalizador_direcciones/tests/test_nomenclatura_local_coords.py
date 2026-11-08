"""Plan 8.1: every index coordinate is the LOCAL reprojection of ``ndcoordx`` / ``ndcoordy``.

The WFS ``lon`` / ``lat`` are kept as ``lon_wfs`` / ``lat_wfs`` for audit only. Rows without usable
projected coordinates fall back to the WFS point and say so in ``coord_source``. Invented data only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pyproj
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import nomenclatura  # noqa: E402
from cali_address import nomenclatura_clean as nc  # noqa: E402
from cali_address.geo import haversine_m  # noqa: E402

from test_nomenclatura_clean import frame, pcer_row, row  # noqa: E402

LOCAL = "local_6249_to_4326"

# Same .prj as the real shapefile (MAGNA-SIRGAS / Cali urban grid, EPSG:6249).
PRJ_WKT = (
    'PROJCS["MAGNA_Cali_Valle_del_Cauca_2009",GEOGCS["GCS_MAGNA",DATUM["D_MAGNA",'
    'SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["IGAC_Plano_Cartesiano"],'
    'PARAMETER["False_Easting",1061900.18],PARAMETER["False_Northing",872364.63],'
    'PARAMETER["Longitude_Of_Center",-76.5205625],PARAMETER["Latitude_Of_Center",3.441883333333334],'
    'PARAMETER["Height",1000.0],UNIT["Meter",1.0]]'
)
CENTER_X, CENTER_Y = 1061900.18, 872364.63
CENTER_LON, CENTER_LAT = -76.5205625, 3.441883333333334


@pytest.fixture(scope="module")
def tr(tmp_path_factory):
    prj = tmp_path_factory.mktemp("prj") / "a.prj"
    prj.write_text(PRJ_WKT, encoding="utf-8")
    return nomenclatura.transformer_from_prj(str(prj))


def at_center(i=0, d_lat=1e-5, **over):
    """A row whose projected point is the grid centre and whose WFS point is ``d_lat`` degrees north."""
    base = dict(ndcoordx=CENTER_X, ndcoordy=CENTER_Y, lon=CENTER_LON, lat=CENTER_LAT + d_lat)
    base.update(over)
    return row(i, **base)


# ------------------------------------------------------------------ nomenclatura.reproject_xy

def test_reproject_xy_known_point_is_within_a_metre(tr):
    lon, lat = nomenclatura.reproject_xy([CENTER_X], [CENTER_Y], tr)
    assert float(haversine_m(lat[0], lon[0], CENTER_LAT, CENTER_LON)) < 1.0


def test_reproject_xy_round_trip_below_one_millimetre(tr):
    xs = np.array([CENTER_X, CENTER_X + 4000.0, CENTER_X - 7000.0])
    ys = np.array([CENTER_Y, CENTER_Y - 3500.0, CENTER_Y + 9000.0])
    lon, lat = nomenclatura.reproject_xy(xs, ys, tr)
    back = pyproj.Transformer.from_crs(tr.target_crs, tr.source_crs, always_xy=True)
    bx, by = back.transform(lon, lat)
    assert np.max(np.hypot(bx - xs, by - ys)) < 1e-3


def test_reproject_xy_uses_lon_lat_order(tr):
    lon, lat = nomenclatura.reproject_xy([CENTER_X], [CENTER_Y], tr)
    assert -77 < lon[0] < -76 and 3 < lat[0] < 4


def test_reproject_xy_non_finite_inputs_give_nan_and_keep_alignment(tr):
    xs = [CENTER_X, np.nan, np.inf, CENTER_X, -np.inf]
    ys = [CENTER_Y, CENTER_Y, CENTER_Y, np.nan, CENTER_Y]
    lon, lat = nomenclatura.reproject_xy(xs, ys, tr)
    assert np.isfinite(lon[0]) and np.isfinite(lat[0])
    assert np.isnan(lon[1:]).all() and np.isnan(lat[1:]).all()
    assert len(lon) == len(lat) == 5


def test_reproject_xy_empty_and_all_nan(tr):
    lon, lat = nomenclatura.reproject_xy([], [], tr)
    assert len(lon) == 0 and len(lat) == 0
    lon, lat = nomenclatura.reproject_xy([np.nan], [np.nan], tr)
    assert np.isnan(lon).all()


def test_reproject_xy_absurd_values_never_leak_infinities(tr):
    lon, lat = nomenclatura.reproject_xy([1e30, CENTER_X], [1e30, CENTER_Y], tr)
    assert not np.isinf(lon).any() and not np.isinf(lat).any()
    assert np.isfinite(lon[1])


def test_reproject_xy_mismatched_lengths_are_an_error(tr):
    with pytest.raises(ValueError):
        nomenclatura.reproject_xy([1.0, 2.0], [1.0], tr)


def test_reproject_wfs_coordinates_reads_the_nd_columns_and_tolerates_nullable_dtypes(tr):
    df = pd.DataFrame({"ndcoordx": pd.array([CENTER_X, None], dtype="Float64"),
                       "ndcoordy": pd.array([CENTER_Y, 5.0], dtype="Float64")})
    lon, lat = nomenclatura.reproject_wfs_coordinates(df, tr)
    assert np.isfinite(lon[0]) and np.isnan(lon[1]) and np.isnan(lat[1])


# ------------------------------------------------------------------ cleaning: local coordinates win

def one(rows, transformer, **kw):
    res = nc.clean_nomenclatura(frame(rows if isinstance(rows, list) else [rows]), transformer=transformer, **kw)
    return res


def test_clean_output_uses_the_local_reprojection_and_keeps_the_wfs_point(tr):
    res = one(at_center(0), tr)
    c = res.clean.iloc[0]
    assert float(haversine_m(c["lat"], c["lon"], CENTER_LAT, CENTER_LON)) < 1.0
    assert c["lon_wfs"] == CENTER_LON and c["lat_wfs"] == pytest.approx(CENTER_LAT + 1e-5)
    assert c["lat"] != c["lat_wfs"]
    assert c["coord_source"] == LOCAL


def test_null_projected_coordinates_fall_back_to_wfs_and_are_counted(tr):
    res = one([at_center(0), at_center(1, ndcoordx=None), at_center(2, ndcoordy=np.nan)], tr)
    assert len(res.clean) == 3 and len(res.quarantine) == 0
    assert res.clean["coord_source"].tolist() == [LOCAL, "wfs", "wfs"]
    for i in (1, 2):
        assert res.clean["lon"].iloc[i] == res.clean["lon_wfs"].iloc[i]
        assert res.clean["lat"].iloc[i] == res.clean["lat_wfs"].iloc[i]
    assert res.report["rules"]["coords_fallback_to_wfs"] == 2
    assert res.report["coord_source_counts"] == {LOCAL: 1, "wfs": 2}


def test_placeholder_zero_projected_coordinates_fall_back_to_wfs_not_to_quarantine(tr):
    # The real layer has thousands of rows with ndcoordx = ndcoordy = 0 and a valid WFS point.
    res = one([at_center(0, ndcoordx=0.0, ndcoordy=0.0), at_center(1, ndcoordx=0.0), at_center(2, ndcoordy=0.0)], tr)
    assert len(res.clean) == 3 and len(res.quarantine) == 0
    assert set(res.clean["coord_source"]) == {"wfs"}
    assert res.report["rules"]["coords_projected_zero"] == 3
    assert res.report["rules"]["coords_fallback_to_wfs"] == 3


def test_infinite_projected_coordinates_fall_back_to_wfs(tr):
    res = one([at_center(0, ndcoordx=np.inf), at_center(1, ndcoordy=-np.inf)], tr)
    assert res.clean["coord_source"].tolist() == ["wfs", "wfs"]


def test_negative_projected_coordinates_are_placeholders_too(tr):
    res = one(at_center(0, ndcoordx=-5.0), tr)
    assert res.clean["coord_source"].tolist() == ["wfs"]


def test_row_with_neither_projected_nor_wfs_coordinates_is_quarantined(tr):
    res = one([at_center(0, ndcoordx=None, ndcoordy=None, lon=np.nan, lat=np.nan), at_center(1)], tr)
    assert res.quarantine["quarantine_reason"].tolist() == ["coords_nan"]
    assert len(res.clean) == 1
    assert res.report["rules"]["coords_nan"] == 1


def test_only_one_wfs_coordinate_and_no_projected_is_quarantined(tr):
    res = one(at_center(0, ndcoordx=None, ndcoordy=None, lat=np.nan), tr)
    assert res.quarantine["quarantine_reason"].tolist() == ["coords_nan"]


def test_wfs_missing_but_projected_present_still_uses_local(tr):
    res = one(at_center(0, lon=np.nan, lat=np.nan), tr)
    c = res.clean.iloc[0]
    assert c["coord_source"] == LOCAL and np.isfinite(c["lon"]) and np.isnan(c["lon_wfs"])


def test_bbox_gate_applies_to_the_local_coordinates(tr):
    far_local = at_center(0, ndcoordx=CENTER_X + 300_000.0)              # WFS point inside, local point outside
    res = one([far_local, at_center(1)], tr)
    assert res.quarantine["quarantine_reason"].tolist() == ["coords_outside_bbox"]
    assert res.clean["ndid"].tolist() == [900001]


def test_wfs_point_outside_the_bbox_does_not_matter_when_the_local_point_is_inside(tr):
    res = one(at_center(0, lon=-75.0, lat=4.9), tr)
    assert len(res.clean) == 1 and res.clean["coord_source"].iloc[0] == LOCAL


def test_bbox_gate_applies_to_the_wfs_fallback_too(tr):
    res = one(at_center(0, ndcoordx=None, lon=-75.0), tr)
    assert res.quarantine["quarantine_reason"].tolist() == ["coords_outside_bbox"]


def test_custom_bbox_is_checked_against_the_local_point(tr):
    tight = (CENTER_LON - 1e-5, CENTER_LAT - 1e-5, CENTER_LON + 1e-5, CENTER_LAT + 1e-5)
    assert len(one(at_center(0), tr, bbox=tight).clean) == 1
    assert len(one(at_center(5, ndcoordx=CENTER_X + 5.0), tr, bbox=tight).clean) == 0   # 5 m east of the centre


def test_without_a_transformer_coordinates_are_the_wfs_ones(tr):
    res = nc.clean_nomenclatura(frame([at_center(0)]))
    c = res.clean.iloc[0]
    assert c["lon"] == c["lon_wfs"] and c["lat"] == c["lat_wfs"] and c["coord_source"] == "wfs"
    assert res.report["rules"]["coords_fallback_to_wfs"] == 0


# ------------------------------------------------------------------ report

def test_report_has_local_vs_wfs_shift_statistics(tr):
    rows = [at_center(i, d_lat=1e-5) for i in range(40)]                  # WFS ~1.1 m north of local
    rows.append(at_center(40, d_lat=5e-5))                                 # ~5.6 m
    s = one(rows, tr).report["coord_shift_local_vs_wfs"]
    assert s["n"] == 41
    assert 0.9 < s["median_m"] < 1.4 and s["max_m"] > 5 and s["p99_m"] >= s["median_m"]
    assert s["share_over_2m"] == pytest.approx(1 / 41)
    assert s["over_100m"] == 0


def test_shift_statistics_exclude_fallback_rows_and_handle_no_pairs(tr):
    s = one([at_center(0, ndcoordx=None)], tr).report["coord_shift_local_vs_wfs"]
    assert s["n"] == 0 and s["median_m"] is None and s["share_over_2m"] is None


def test_shift_statistics_flag_gross_disagreements(tr):
    s = one([at_center(0, d_lat=0.01)], tr).report["coord_shift_local_vs_wfs"]   # ~1.1 km
    assert s["over_100m"] == 1


def test_report_with_a_transformer_is_json_serialisable(tr):
    res = one([at_center(0), at_center(1, ndcoordx=None), pcer_row(2, ndcoordx=0.0, ndcoordy=0.0)], tr)
    json.loads(json.dumps(res.report))


def test_empty_frame_with_a_transformer(tr):
    res = nc.clean_nomenclatura(frame([]), transformer=tr)
    assert len(res.clean) == 0 and res.report["coord_source_counts"] == {}
    assert {"lon_wfs", "lat_wfs", "coord_source"} <= set(res.clean.columns)


def test_input_frame_is_not_mutated(tr):
    df = frame([at_center(0)])
    snapshot = df.copy()
    nc.clean_nomenclatura(df, transformer=tr)
    pd.testing.assert_frame_equal(df, snapshot)


def test_quarantine_share_stays_zero_when_only_fallbacks_happen(tr):
    res = one([at_center(i, ndcoordx=None) for i in range(200)], tr)
    assert res.report["quarantined_rows"] == 0 and res.report["gate_passed"] is True


def test_module_documents_the_unit_suffix_consistency_rule():
    doc = nc.__doc__
    assert "Unit-suffix consistency rule" in doc and "LONGER" in doc and "continues the last digit" in doc
    assert "local_reprojection" not in doc and "LOCAL reprojection" in doc


# ------------------------------------------------------------------ idempotency

def test_rerunning_the_cleaning_on_its_own_output_changes_nothing(tr):
    rows = [at_center(0), at_center(1, ndcoordx=None), at_center(2, ndcoordx=0.0, ndcoordy=0.0),
            at_center(3, lon=np.nan, lat=np.nan)]
    first = one(rows, tr).clean
    again = nc.clean_nomenclatura(first, transformer=tr).clean
    cols = ["lon", "lat", "lon_wfs", "lat_wfs", "coord_source", "ndid", "nddirecc"]
    pd.testing.assert_frame_equal(again[cols].reset_index(drop=True), first[cols].reset_index(drop=True))


def test_rerun_without_a_transformer_keeps_the_original_provenance(tr):
    first = one([at_center(0), at_center(1, ndcoordx=None)], tr).clean
    again = nc.clean_nomenclatura(first).clean
    assert again["coord_source"].tolist() == first["coord_source"].tolist()
    assert again["lon"].tolist() == first["lon"].tolist()


# ------------------------------------------------------------------ script

def _load_script():
    spec = importlib.util.spec_from_file_location("script_clean_nomenclatura_local", os.path.join(ROOT, "scripts", "clean_nomenclatura.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path, *extra):
    return ["--in", str(tmp_path / "wfs.parquet"), "--out", str(tmp_path / "clean.parquet"),
            "--quarantine", str(tmp_path / "q.parquet"), "--report", str(tmp_path / "report.json"), *extra]


@pytest.fixture
def prj_file(tmp_path):
    p = tmp_path / "grid.prj"
    p.write_text(PRJ_WKT, encoding="utf-8")
    return str(p)


def test_script_writes_local_and_wfs_columns_and_reports_the_fallbacks(tmp_path, prj_file):
    rows = [at_center(i) for i in range(300)] + [at_center(400, ndcoordx=None), at_center(401, ndcoordx=0.0, ndcoordy=0.0)]
    frame(rows).to_parquet(tmp_path / "wfs.parquet", index=False)
    assert _load_script().main(_args(tmp_path, "--prj", prj_file)) == 0
    out = pd.read_parquet(tmp_path / "clean.parquet")
    assert len(out) == 302
    assert {"lon", "lat", "lon_wfs", "lat_wfs", "coord_source"} <= set(out.columns)
    assert out["coord_source"].value_counts().to_dict() == {LOCAL: 300, "wfs": 2}
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert rep["rules"]["coords_fallback_to_wfs"] == 2 and rep["rules"]["coords_projected_zero"] == 1
    assert rep["coord_shift_local_vs_wfs"]["n"] == 300 and rep["prj_file"].endswith("grid.prj")


def test_script_rerun_on_the_same_input_is_deterministic(tmp_path, prj_file):
    frame([at_center(i) for i in range(300)]).to_parquet(tmp_path / "wfs.parquet", index=False)
    mod = _load_script()
    assert mod.main(_args(tmp_path, "--prj", prj_file)) == 0
    first = pd.read_parquet(tmp_path / "clean.parquet")
    assert mod.main(_args(tmp_path, "--prj", prj_file)) == 0
    pd.testing.assert_frame_equal(first, pd.read_parquet(tmp_path / "clean.parquet"))


def test_script_missing_prj_is_a_clear_error_and_writes_nothing(tmp_path, capsys):
    frame([at_center(0)]).to_parquet(tmp_path / "wfs.parquet", index=False)
    rc = _load_script().main(_args(tmp_path, "--prj", str(tmp_path / "nope.prj")))
    err = capsys.readouterr().err
    assert rc == 1 and "nope.prj" in err and "Traceback" not in err
    assert not (tmp_path / "clean.parquet").exists()


@pytest.mark.parametrize("content", ["", "   ", "not a wkt"])
def test_script_unusable_prj_is_a_clear_error(tmp_path, capsys, content):
    frame([at_center(0)]).to_parquet(tmp_path / "wfs.parquet", index=False)
    bad = tmp_path / "bad.prj"
    bad.write_text(content, encoding="utf-8")
    rc = _load_script().main(_args(tmp_path, "--prj", str(bad)))
    assert rc == 1 and "bad.prj" in capsys.readouterr().err
    assert not (tmp_path / "clean.parquet").exists()
