"""Phase 0.1: reproject the certified-nomenclature shapefile from EPSG:6249 to EPSG:4326.

All fixtures are tiny shapefiles written with ``shapefile.Writer`` in ``tmp_path``; the real
12 MB / 554 MB files are never touched. Addresses are invented.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pyproj
import pytest
import shapefile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import nomenclatura  # noqa: E402
from cali_address.geo import haversine_m  # noqa: E402

# Copy of the .prj shipped with the real shapefile (MAGNA-SIRGAS / Cali urban grid, EPSG:6249).
PRJ_WKT = (
    'PROJCS["MAGNA_Cali_Valle_del_Cauca_2009",GEOGCS["GCS_MAGNA",DATUM["D_MAGNA",'
    'SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["IGAC_Plano_Cartesiano"],'
    'PARAMETER["False_Easting",1061900.18],PARAMETER["False_Northing",872364.63],'
    'PARAMETER["Longitude_Of_Center",-76.5205625],PARAMETER["Latitude_Of_Center",3.441883333333334],'
    'PARAMETER["Height",1000.0],UNIT["Meter",1.0]]'
)

# A point in the grid and its expected geographic position (computed once from the .prj above and
# cross-checked with the official false origin: the projection centre maps to -76.5205625 / 3.44188333).
KNOWN_X, KNOWN_Y = 1061900.18, 872364.63
KNOWN_LON, KNOWN_LAT = -76.5205625, 3.441883333333334


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "script_reproject_nomenclatura", os.path.join(ROOT, "scripts", "reproject_nomenclatura.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_shp(tmp_path, points, name="nom", prj=PRJ_WKT, null_rows=()):
    """Write a POINT shapefile with the real 15-field schema; ``points`` is a list of (x, y)."""
    base = str(tmp_path / name)
    w = shapefile.Writer(base, shapeType=shapefile.POINT, encoding="utf-8")
    w.field("OBJECTID", "N", 10, 0)
    w.field("NdID", "N", 10, 0)
    w.field("NdIdPredio", "N", 10, 0)
    w.field("NdNpn", "C", 30)
    w.field("NdDirecc", "C", 80)
    w.field("NdNomVial", "C", 40)
    w.field("NdTipoVia", "C", 10)
    w.field("NdNumVia", "C", 10)
    w.field("NdPlaca", "C", 20)
    w.field("NdInterior", "C", 40)
    w.field("NdEstado", "C", 10)
    w.field("NdCoordX", "N", 20, 4)
    w.field("NdCoordY", "N", 20, 4)
    w.field("NdFuente", "C", 20)
    for i, (x, y) in enumerate(points):
        if i in null_rows:
            w.null()
        else:
            w.point(x, y)
        w.record(i + 1, 1000 + i, 5000 + i, f"76001010007070001000300000{i:04d}",
                 f"KR {10 + i} # 5 - {i}", f"KR {10 + i}", "KR", str(10 + i), f"5 - {i}", "",
                 "cert", x if x is not None else 0, y if y is not None else 0, "Nomenclatura")
    w.close()
    if prj is not None:
        with open(base + ".prj", "w", encoding="utf-8") as fh:
            fh.write(prj)
    with open(base + ".cpg", "w", encoding="utf-8") as fh:
        fh.write("UTF-8")
    return base + ".shp"


# ---------------------------------------------------------------- transformer_from_prj

def test_known_cali_point_lands_within_one_metre(tmp_path):
    prj = tmp_path / "a.prj"
    prj.write_text(PRJ_WKT, encoding="utf-8")
    tr = nomenclatura.transformer_from_prj(str(prj))
    lon, lat = tr.transform(KNOWN_X, KNOWN_Y)
    assert float(haversine_m(lat, lon, KNOWN_LAT, KNOWN_LON)) < 1.0


def test_transformer_uses_lon_lat_axis_order(tmp_path):
    prj = tmp_path / "a.prj"
    prj.write_text(PRJ_WKT, encoding="utf-8")
    lon, lat = nomenclatura.transformer_from_prj(str(prj)).transform(KNOWN_X, KNOWN_Y)
    assert -77 < lon < -76 and 3 < lat < 4


def test_missing_prj_raises_clear_error(tmp_path):
    with pytest.raises(nomenclatura.CrsError, match=r"\.prj"):
        nomenclatura.transformer_from_prj(str(tmp_path / "does_not_exist.prj"))


def test_empty_prj_raises_clear_error(tmp_path):
    prj = tmp_path / "empty.prj"
    prj.write_text("   \n", encoding="utf-8")
    with pytest.raises(nomenclatura.CrsError, match="empty"):
        nomenclatura.transformer_from_prj(str(prj))


def test_garbage_prj_raises_clear_error(tmp_path):
    prj = tmp_path / "bad.prj"
    prj.write_text("this is not a CRS", encoding="utf-8")
    with pytest.raises(nomenclatura.CrsError, match="parse"):
        nomenclatura.transformer_from_prj(str(prj))


# ---------------------------------------------------------------- read_shapefile_points

def test_reader_returns_attributes_and_source_coordinates(tmp_path):
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y), (KNOWN_X + 100, KNOWN_Y - 50)])
    df = nomenclatura.read_shapefile_points(shp)
    assert len(df) == 2
    assert list(df["x_src"]) == pytest.approx([KNOWN_X, KNOWN_X + 100])
    assert list(df["y_src"]) == pytest.approx([KNOWN_Y, KNOWN_Y - 50])
    for col in ("OBJECTID", "NdID", "NdNpn", "NdDirecc", "NdEstado", "NdCoordX", "NdFuente"):
        assert col in df.columns
    assert df["NdDirecc"].iloc[0] == "KR 10 # 5 - 0"


def test_reader_keeps_null_geometry_as_nan(tmp_path):
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y), (0, 0), (KNOWN_X, KNOWN_Y)], null_rows=(1,))
    df = nomenclatura.read_shapefile_points(shp)
    assert len(df) == 3
    assert np.isnan(df["x_src"].iloc[1]) and np.isnan(df["y_src"].iloc[1])
    assert df["x_src"].notna().sum() == 2


def test_reader_empty_shapefile_returns_empty_frame_with_columns(tmp_path):
    shp = write_shp(tmp_path, [])
    df = nomenclatura.read_shapefile_points(shp)
    assert len(df) == 0
    assert "x_src" in df.columns and "NdNpn" in df.columns


def test_reader_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        nomenclatura.read_shapefile_points(str(tmp_path / "nope.shp"))


def test_reader_decodes_accents_from_utf8_dbf(tmp_path):
    base = str(tmp_path / "acc")
    w = shapefile.Writer(base, shapeType=shapefile.POINT, encoding="utf-8")
    w.field("NdDirecc", "C", 40)
    w.point(KNOWN_X, KNOWN_Y)
    w.record("CALLE NIÑO # 1 - 2")
    w.close()
    with open(base + ".cpg", "w", encoding="utf-8") as fh:
        fh.write("UTF-8")
    df = nomenclatura.read_shapefile_points(base + ".shp")
    assert df["NdDirecc"].iloc[0] == "CALLE NIÑO # 1 - 2"


# ---------------------------------------------------------------- reprojection

def test_round_trip_is_under_one_millimetre(tmp_path):
    pts = [(KNOWN_X + dx, KNOWN_Y + dy) for dx in (-9000, 0, 12000) for dy in (-8000, 0, 15000)]
    shp = write_shp(tmp_path, pts)
    df = nomenclatura.read_shapefile_points(shp)
    tr = nomenclatura.transformer_from_prj(shp[:-4] + ".prj")
    out = nomenclatura.reproject_frame(df, tr)
    back = pyproj.Transformer.from_crs(4326, pyproj.CRS.from_wkt(PRJ_WKT), always_xy=True)
    bx, by = back.transform(out["lon"].to_numpy(), out["lat"].to_numpy())
    assert np.max(np.hypot(bx - out["x_src"].to_numpy(), by - out["y_src"].to_numpy())) < 1e-3


def test_reproject_keeps_nan_rows_as_nan_and_does_not_raise(tmp_path):
    df = pd.DataFrame({"x_src": [KNOWN_X, np.nan, KNOWN_X], "y_src": [KNOWN_Y, np.nan, np.nan]})
    prj = tmp_path / "a.prj"
    prj.write_text(PRJ_WKT, encoding="utf-8")
    out = nomenclatura.reproject_frame(df, nomenclatura.transformer_from_prj(str(prj)))
    assert out["lon"].notna().tolist() == [True, False, False]
    assert out["lat"].notna().tolist() == [True, False, False]


def test_reproject_infinite_coordinates_become_nan(tmp_path):
    df = pd.DataFrame({"x_src": [np.inf, KNOWN_X], "y_src": [KNOWN_Y, -np.inf]})
    prj = tmp_path / "a.prj"
    prj.write_text(PRJ_WKT, encoding="utf-8")
    out = nomenclatura.reproject_frame(df, nomenclatura.transformer_from_prj(str(prj)))
    assert out["lon"].isna().all() and out["lat"].isna().all()


def test_reproject_empty_frame(tmp_path):
    prj = tmp_path / "a.prj"
    prj.write_text(PRJ_WKT, encoding="utf-8")
    df = pd.DataFrame({"x_src": pd.Series([], dtype="float64"), "y_src": pd.Series([], dtype="float64")})
    out = nomenclatura.reproject_frame(df, nomenclatura.transformer_from_prj(str(prj)))
    assert len(out) == 0 and {"lon", "lat"} <= set(out.columns)


# ---------------------------------------------------------------- bbox gate

def test_outside_bbox_mask_flags_only_points_outside_cali():
    lon = np.array([-76.52, -75.0, -76.52, np.nan, -76.65, -76.40])
    lat = np.array([3.44, 3.44, 10.0, 3.44, 3.30, 3.55])
    mask = nomenclatura.outside_bbox_mask(lon, lat)
    # NaN is not "outside" (it is counted separately); edge values are inside (closed interval).
    assert mask.tolist() == [False, True, True, False, False, False]


def test_outside_bbox_mask_just_beyond_the_edge_is_outside():
    mask = nomenclatura.outside_bbox_mask(np.array([-76.6501, -76.3999]), np.array([3.30, 3.55]))
    assert mask.tolist() == [True, True]


# ---------------------------------------------------------------- script end-to-end

def _docs_fixture(tmp_path, npn, lon, lat):
    p = tmp_path / "catastro_docs.parquet"
    pd.DataFrame({"numero_predial_nacional": [npn], "centroid_lon": [lon], "centroid_lat": [lat]}).to_parquet(p)
    return str(p)


def test_script_writes_parquet_and_report(tmp_path):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y), (KNOWN_X + 500, KNOWN_Y + 500)])
    npn0 = "760010100070700010003000000000"
    docs = _docs_fixture(tmp_path, npn0, KNOWN_LON, KNOWN_LAT)
    out = tmp_path / "out.parquet"
    rc = mod.main(["--shp", shp, "--out", str(out), "--catastro-docs", docs])
    assert rc == 0
    df = pd.read_parquet(out)
    assert len(df) == 2
    for col in ("lon", "lat", "x_6249", "y_6249", "crs_source", "crs_target", "NdNpn", "NdCoordX"):
        assert col in df.columns
    assert (df["crs_source"] == "EPSG:6249").all() and (df["crs_target"] == "EPSG:4326").all()
    report = json.loads((tmp_path / "reprojection_report.json").read_text(encoding="utf-8"))
    assert report["count"] == 2
    assert report["nan_geometry"] == 0
    assert report["bbox_before"]["x_min"] == pytest.approx(KNOWN_X)
    assert report["bbox_after"]["lon_min"] == pytest.approx(KNOWN_LON, abs=1e-6)
    assert report["sanity"]["shared_npn"] == 1
    assert report["sanity"]["median_shift_m"] < 1.0
    assert report["gates"]["bbox_ok"] is True and report["gates"]["sanity_ok"] is True


def test_script_counts_nan_geometry_and_still_succeeds(tmp_path):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y), (0, 0)], null_rows=(1,))
    out = tmp_path / "out.parquet"
    rc = mod.main(["--shp", shp, "--out", str(out), "--catastro-docs", str(tmp_path / "missing.parquet")])
    assert rc == 0
    report = json.loads((tmp_path / "reprojection_report.json").read_text(encoding="utf-8"))
    assert report["nan_geometry"] == 1
    assert report["sanity"]["shared_npn"] == 0  # no catastro file: sanity skipped, not failed
    assert report["gates"]["sanity_ok"] is None
    df = pd.read_parquet(out)
    assert df["lon"].isna().sum() == 1


def test_script_fails_bbox_gate_for_point_outside_cali(tmp_path):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y), (KNOWN_X + 400000, KNOWN_Y)])
    out = tmp_path / "out.parquet"
    rc = mod.main(["--shp", shp, "--out", str(out), "--catastro-docs", str(tmp_path / "missing.parquet")])
    assert rc == 2
    report = json.loads((tmp_path / "reprojection_report.json").read_text(encoding="utf-8"))
    assert report["gates"]["bbox_ok"] is False
    assert report["outside_bbox"] == 1


def test_script_aborts_when_median_shift_exceeds_100_m(tmp_path):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y)])
    docs = _docs_fixture(tmp_path, "760010100070700010003000000000", KNOWN_LON + 0.01, KNOWN_LAT)  # ~1.1 km away
    rc = mod.main(["--shp", shp, "--out", str(tmp_path / "o.parquet"), "--catastro-docs", docs])
    assert rc == 2
    report = json.loads((tmp_path / "reprojection_report.json").read_text(encoding="utf-8"))
    assert report["gates"]["sanity_ok"] is False


def test_script_warns_but_passes_between_30_and_100_m(tmp_path, capsys):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y)])
    docs = _docs_fixture(tmp_path, "760010100070700010003000000000", KNOWN_LON + 0.0006, KNOWN_LAT)  # ~66 m
    rc = mod.main(["--shp", shp, "--out", str(tmp_path / "o.parquet"), "--catastro-docs", docs])
    assert rc == 0
    assert "WARNING" in capsys.readouterr().out


def test_script_empty_shapefile_does_not_crash(tmp_path):
    mod = _load_script()
    shp = write_shp(tmp_path, [])
    rc = mod.main(["--shp", shp, "--out", str(tmp_path / "o.parquet"), "--catastro-docs", str(tmp_path / "m.parquet")])
    assert rc == 2  # nothing to reproject is a failed gate, not a traceback
    report = json.loads((tmp_path / "reprojection_report.json").read_text(encoding="utf-8"))
    assert report["count"] == 0


def test_script_missing_prj_returns_clear_error(tmp_path, capsys):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y)], prj=None)
    rc = mod.main(["--shp", shp, "--out", str(tmp_path / "o.parquet"), "--catastro-docs", str(tmp_path / "m.parquet")])
    assert rc == 1
    assert ".prj" in capsys.readouterr().err
    assert not (tmp_path / "o.parquet").exists()


def test_script_is_idempotent(tmp_path):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y)])
    args = ["--shp", shp, "--out", str(tmp_path / "o.parquet"), "--catastro-docs", str(tmp_path / "m.parquet")]
    assert mod.main(args) == 0
    first = pd.read_parquet(tmp_path / "o.parquet")
    assert mod.main(args) == 0
    pd.testing.assert_frame_equal(first, pd.read_parquet(tmp_path / "o.parquet"))


def test_script_geojson_option_writes_feature_collection(tmp_path):
    mod = _load_script()
    shp = write_shp(tmp_path, [(KNOWN_X, KNOWN_Y), (0, 0)], null_rows=(1,))
    gj = tmp_path / "o.geojson"
    rc = mod.main(["--shp", shp, "--out", str(tmp_path / "o.parquet"), "--catastro-docs",
                   str(tmp_path / "m.parquet"), "--geojson", str(gj)])
    assert rc == 0
    data = json.loads(gj.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1  # the NaN-geometry row has no point to export
    assert math.isclose(data["features"][0]["geometry"]["coordinates"][0], KNOWN_LON, abs_tol=1e-6)
