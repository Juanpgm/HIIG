"""Any-city flow: synthetic city generator, generic ingestion adapter, prepare_city script and a short end-to-end
training on an invented city (CPU, small encoder). Everything here is invented: no real address, parcel or person."""

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

from cali_address.city_adapter import (  # noqa: E402
    QUARANTINE_REASONS, AdapterError, InputReadError, adapt, load_mapping, mapping_from_dict, read_source,
    repair_mojibake,
)
from cali_address.city_eval import chance_recall, ece, eval_queries, paired_bootstrap, wilson_interval  # noqa: E402
from cali_address.parser import canonical, parse_address  # noqa: E402
from cali_address.synthetic_city import (  # noqa: E402
    CALLE_BASE, CARRERA_BASE, default_mapping, grid_for_size, make_city,
)


#: Columns ``index_union.build_nomenclatura`` requires (checked against the module when it is present).
REQUIRED_COLUMNS = ("gid", "nddirecc", "ndnpn", "ndidestado", "lon", "lat")
HAS_INDEX_FLOW = (os.path.isfile(os.path.join(ROOT, "scripts", "build_index.py"))
                  and importlib.util.find_spec("cali_address.index_union") is not None)
needs_index_flow = pytest.mark.skipif(
    not HAS_INDEX_FLOW, reason="the nomenclature index flow (scripts/build_index.py, index_union) is not in this checkout")


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mapping(**overrides) -> dict:
    data = default_mapping("unused.csv")
    data.pop("input")
    data.update(overrides)
    return data


def _city(n: int = 300, **kw) -> pd.DataFrame:
    return make_city(n, seed=11, **kw)


def _lonlat_frame(n: int = 20) -> pd.DataFrame:
    """A tiny invented table already in lon/lat (EPSG:4326)."""
    return pd.DataFrame({
        "id": [f"R{i}" for i in range(n)],
        "dir": [f"KR {CARRERA_BASE + i} # {CALLE_BASE + i} - {10 + i:02d}" for i in range(n)],
        "lon": [-72.7 + i * 1e-4 for i in range(n)],
        "lat": [4.41 + i * 1e-4 for i in range(n)],
    })


def _lonlat_mapping(**overrides) -> dict:
    data = {"city": "Prueba", "crs": "EPSG:4326",
            "columns": {"address": "dir", "record_id": "id", "x": "lon", "y": "lat"}}
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------------------------------------
# synthetic city generator
# ---------------------------------------------------------------------------------------------------------

class TestSyntheticCity:
    def test_every_address_round_trips_through_the_parser(self):
        city = _city(400)
        base = city[city["unidad"] == ""]["dir_texto"]
        bad = [t for t in base if canonical(parse_address(t), style="spaced", with_complement=True) != t]
        assert bad == []

    def test_numbers_are_above_the_reference_city_range(self):
        city = _city(300)
        for text in city["dir_texto"]:
            p = parse_address(text)
            assert int(p.via_number) >= CALLE_BASE and int(p.cross_number) >= CALLE_BASE

    def test_parcel_codes_have_30_digits_and_fake_municipality(self):
        city = _city(300)
        assert city["codigo_predial"].str.fullmatch(r"99999\d{25}").all()
        assert city["codigo_predial"].is_unique and city["id_registro"].is_unique

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        pd.testing.assert_frame_equal(make_city(200, seed=3), make_city(200, seed=3))
        assert not make_city(200, seed=3)["coord_x"].equals(make_city(200, seed=4)["coord_x"])

    def test_has_the_nomenclature_variety_the_parser_must_cover(self):
        text = " ".join(_city(2000)["dir_texto"])
        for token in (" SUR ", " BIS ", "DG ", "TV ", " A # ", " AP "):
            assert token in text, token

    @pytest.mark.parametrize("bad", [0, -5])
    def test_grid_rejects_non_positive_sizes(self, bad):
        with pytest.raises(ValueError):
            grid_for_size(bad)

    def test_grid_minimum_is_two_by_two(self):
        assert grid_for_size(1) == (2, 2)

    def test_shares_out_of_range_are_rejected(self):
        with pytest.raises(ValueError):
            make_city(100, unit_share=1.5)


# ---------------------------------------------------------------------------------------------------------
# mapping validation
# ---------------------------------------------------------------------------------------------------------

class TestMapping:
    def test_valid_mapping(self):
        m = mapping_from_dict(_mapping())
        assert m.crs == "EPSG:3116" and m.crs_declared and m.columns["address"] == "dir_texto"

    @pytest.mark.parametrize("data, needle", [
        ({"columns": {"address": "a", "record_id": "b"}, "colour": 1}, "unknown mapping key"),
        ({"columns": {}}, "needs a 'columns'"),
        ({"columns": {"address": "a"}}, "columns.record_id is mandatory"),
        ({"columns": {"record_id": "a"}}, "columns.address is mandatory"),
        ({"columns": {"address": "a", "record_id": "b", "street": "c"}}, "unknown column role"),
        ({"columns": {"address": "a", "record_id": "b", "x": "c"}}, "go together"),
        ({"columns": {"address": "a", "record_id": "b", "status": "s"}}, "needs status_values"),
        ({"columns": {"address": "a", "record_id": "b"}, "status_values": {"cert": ["x"]}}, "needs columns.status"),
        ({"columns": {"address": "a", "record_id": "b", "status": "s"},
          "status_values": {"cert": ["x"], "pcer": ["X"]}}, "both cert and pcer"),
        ({"columns": {"address": "a", "record_id": "b", "status": "s"}, "status_values": {"ok": ["x"]}},
         "'cert' and/or 'pcer'"),
        ({"columns": {"address": "a", "record_id": "b"}, "bbox": [1, 2, 3]}, "bbox"),
        ({"columns": {"address": "a", "record_id": "b"}, "bbox": [5, 2, 3, 4]}, "min < max"),
        ({"columns": {"address": "a", "record_id": "b"}, "max_quarantine_share": 1.5}, "within"),
        ({"columns": {"address": "a", "record_id": "b"}, "max_quarantine_share": True}, "must be a number"),
        ({"columns": {"address": "a", "record_id": "b"}, "max_km_from_median": 0}, "within"),
        ({"columns": {"address": 3, "record_id": "b"}}, "column name"),
        ({"columns": {"address": ["a", ""], "record_id": "b"}}, "non-empty column names"),
    ])
    def test_invalid_mappings_give_one_clear_message(self, data, needle):
        with pytest.raises(AdapterError) as exc:
            mapping_from_dict(data)
        assert needle in str(exc.value)

    def test_not_an_object(self):
        with pytest.raises(AdapterError):
            mapping_from_dict(["columns"])

    def test_input_resolves_against_the_mapping_directory(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({**_lonlat_mapping(), "input": "data/x.csv"}), encoding="utf-8")
        assert load_mapping(str(path)).input == os.path.normpath(str(tmp_path / "data" / "x.csv"))

    def test_toml_and_yaml_mappings(self, tmp_path):
        toml = tmp_path / "m.toml"
        toml.write_text('city = "T"\ncrs = "EPSG:4326"\n[columns]\naddress = "dir"\nrecord_id = "id"\n'
                        'x = "lon"\ny = "lat"\n', encoding="utf-8")
        assert load_mapping(str(toml)).columns["x"] == "lon"
        pytest.importorskip("yaml")
        yml = tmp_path / "m.yaml"
        yml.write_text("city: T\ncolumns:\n  address: dir\n  record_id: id\n  x: lon\n  y: lat\n", encoding="utf-8")
        m = load_mapping(str(yml))
        assert m.columns["y"] == "lat" and not m.crs_declared

    @pytest.mark.parametrize("name, content, needle", [
        ("m.json", "{not json", "malformed"),
        ("m.json", "[1, 2]", "top level"),
        ("m.ini", "x=1", "unsupported mapping extension"),
    ])
    def test_bad_mapping_files(self, tmp_path, name, content, needle):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        with pytest.raises(AdapterError) as exc:
            load_mapping(str(path))
        assert needle in str(exc.value)

    def test_missing_mapping_file(self, tmp_path):
        with pytest.raises(AdapterError, match="not found"):
            load_mapping(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------------------------------------

class TestAdapt:
    @needs_index_flow
    def test_required_columns_match_the_index_builder(self):
        from cali_address.index_union import REQUIRED_COLUMNS as BUILDER_COLUMNS

        assert tuple(BUILDER_COLUMNS) == REQUIRED_COLUMNS

    def test_happy_path_produces_the_build_index_schema(self):
        city = _city(300)
        res = adapt(city, mapping_from_dict(_mapping()), fmt="csv")
        clean = res.clean
        assert set(REQUIRED_COLUMNS) <= set(clean.columns)
        assert clean["gid"].is_unique and clean["gid"].dtype == "int64"
        assert clean["lon"].between(-180, 180).all() and clean["lat"].between(-90, 90).all()
        assert set(clean["ndidestado"]) <= {"cert", "pcer"}
        assert res.report["by_reason"]["status_not_accepted"] == int((city["estado_registro"] == "anulado").sum())
        assert res.report["kept_rows"] + res.report["quarantined_rows"] == len(city)
        assert res.gates_failed == []
        # every district / neighbourhood becomes a code; neighbourhood codes start with their district code
        for code in clean["ndidbarrio"].unique():
            assert code[:2] in set(clean["ndcomuna"])
        assert len(res.codes["districts"]) == city["distrito_nombre"].nunique()

    def test_report_holds_no_address_text(self):
        city = _city(200)
        res = adapt(city, mapping_from_dict(_mapping()))
        dumped = json.dumps(res.report, default=str)
        assert not any(t in dumped for t in city["dir_texto"].head(50))

    def test_empty_frame_is_a_clear_error(self):
        empty = _city(100).iloc[0:0]
        with pytest.raises(AdapterError, match="no data rows"):
            adapt(empty, mapping_from_dict(_mapping()))

    def test_missing_mandatory_column_lists_available_columns(self):
        city = _city(100).drop(columns=["coord_y"])
        with pytest.raises(AdapterError) as exc:
            adapt(city, mapping_from_dict(_mapping()))
        assert "coord_y" in str(exc.value) and "Available columns" in str(exc.value)

    def test_projected_coordinates_declared_as_lonlat_are_rejected_with_a_hint(self):
        with pytest.raises(AdapterError) as exc:
            adapt(_city(100), mapping_from_dict(_mapping(crs="EPSG:4326")))
        assert "projected" in str(exc.value)

    def test_lonlat_declared_as_projected_is_rejected_with_a_hint(self):
        with pytest.raises(AdapterError) as exc:
            adapt(_lonlat_frame(), mapping_from_dict(_lonlat_mapping(crs="EPSG:3116")))
        assert "EPSG:4326" in str(exc.value)

    def test_swapped_lonlat_gets_a_swap_hint_with_a_bbox(self):
        frame = _lonlat_frame().rename(columns={"lon": "lat", "lat": "lon"})
        with pytest.raises(AdapterError, match="swapped"):
            adapt(frame, mapping_from_dict(_lonlat_mapping(bbox=[-73.5, 3.5, -72.0, 5.0])))

    def test_swapped_lonlat_without_bbox_is_only_a_warning(self):
        # (4.41, -72.7) is a valid lon/lat (Antarctica): without a bbox it cannot be rejected, only flagged
        frame = _lonlat_frame().rename(columns={"lon": "lat", "lat": "lon"})
        res = adapt(frame, mapping_from_dict(_lonlat_mapping()))
        assert any("median latitude" in w for w in res.report["warnings"])

    def test_unknown_crs(self):
        with pytest.raises(AdapterError, match="unknown CRS"):
            adapt(_lonlat_frame(), mapping_from_dict(_lonlat_mapping(crs="EPSG:999999")))

    def test_undeclared_crs_defaults_to_lonlat_with_a_warning(self):
        data = _lonlat_mapping()
        data.pop("crs")
        res = adapt(_lonlat_frame(), mapping_from_dict(data))
        assert res.report["kept_rows"] == 20 and any("no 'crs' declared" in w for w in res.report["warnings"])

    def test_nan_junk_and_decimal_comma_coordinates(self):
        frame = _lonlat_frame(40).astype({"lon": object, "lat": object})
        frame.loc[0, "lon"] = np.nan
        frame.loc[1, "lat"] = None
        frame.loc[2, "lon"] = "abc"
        frame.loc[3, "lon"] = "-72,7003"            # decimal comma: accepted
        frame.loc[4, "lat"] = float("inf")
        res = adapt(frame, mapping_from_dict(_lonlat_mapping()))
        assert res.report["by_reason"]["missing_coordinates"] == 4
        assert "R3" in set(res.clean["source_record_id"])
        assert res.report["quarantine_share"] == pytest.approx(4 / 40)
        assert res.gates_failed == ["quarantine_share"]          # 10 % > default 5 %

    def test_out_of_range_coordinates_after_transform(self):
        frame = _lonlat_frame(40)
        frame.loc[5, "lat"] = 95.0
        res = adapt(frame, mapping_from_dict(_lonlat_mapping(max_quarantine_share=0.2)))
        assert res.report["by_reason"]["coordinates_out_of_range"] == 1

    def test_duplicates_ids_rows_and_missing_ids(self):
        frame = _lonlat_frame(40)
        frame.loc[1, "id"] = "R0"                      # same id as row 0
        frame.loc[2, "id"] = ""                        # missing id
        frame.loc[3, ["dir", "lon", "lat"]] = frame.loc[4, ["dir", "lon", "lat"]].values   # same record, new id
        res = adapt(frame, mapping_from_dict(_lonlat_mapping(max_quarantine_share=0.2)))
        r = res.report["by_reason"]
        assert (r["duplicate_record_id"], r["missing_record_id"], r["duplicate_row"]) == (1, 1, 1)
        assert res.clean["source_record_id"].is_unique
        assert list(res.quarantine.columns) == ["row_number", "record_id", "reason", "address", "x", "y"]
        assert set(res.quarantine["reason"]) <= set(QUARANTINE_REASONS)

    def test_empty_and_symbol_only_addresses(self):
        frame = _lonlat_frame(40)
        frame.loc[0, "dir"] = "   "
        frame.loc[1, "dir"] = "#-"
        frame.loc[2, "dir"] = None
        res = adapt(frame, mapping_from_dict(_lonlat_mapping(max_quarantine_share=0.2)))
        assert res.report["by_reason"]["empty_address"] == 3

    def test_far_from_median_and_bbox(self):
        frame = _lonlat_frame(40)
        frame.loc[0, "lon"] = -70.9                    # about 200 km east
        res = adapt(frame, mapping_from_dict(_lonlat_mapping(max_km_from_median=20, max_quarantine_share=0.2)))
        assert res.report["by_reason"]["far_from_median"] == 1
        res = adapt(frame, mapping_from_dict(_lonlat_mapping(bbox=[-73, 4, -72, 5], max_quarantine_share=0.2)))
        assert res.report["by_reason"]["outside_bbox"] == 1

    def test_bbox_that_misses_most_rows_is_a_crs_error(self):
        with pytest.raises(AdapterError, match="bbox"):
            adapt(_lonlat_frame(), mapping_from_dict(_lonlat_mapping(bbox=[-80, 0, -79, 1])))

    def test_status_mapping_case_insensitive(self):
        frame = _lonlat_frame(10).assign(st=["Vigente"] * 8 + ["PROVISIONAL", "anulado"])
        data = _lonlat_mapping(status_values={"cert": ["vigente"], "pcer": ["provisional"]},
                               max_quarantine_share=0.2)
        data["columns"]["status"] = "st"
        res = adapt(frame, mapping_from_dict(data))
        assert res.report["by_tier"] == {"cert": 8, "pcer": 1}
        assert res.report["by_reason"]["status_not_accepted"] == 1

    def test_address_from_several_columns(self):
        frame = _lonlat_frame(5)
        frame[["via", "num"]] = frame["dir"].str.split(" # ", n=1, expand=True)
        data = _lonlat_mapping()
        data["columns"]["address"] = ["via", "num"]
        res = adapt(frame.drop(columns=["dir"]), mapping_from_dict(data))
        assert res.clean["nddirecc"].iloc[0] == frame["via"].iloc[0] + " " + frame["num"].iloc[0]

    def test_geometry_input_without_xy(self):
        frame = _lonlat_frame(10).rename(columns={"lon": "_lon", "lat": "_lat"})
        data = _lonlat_mapping()
        del data["columns"]["x"], data["columns"]["y"]
        res = adapt(frame, mapping_from_dict(data), fmt="geojson")
        assert res.report["kept_rows"] == 10
        with pytest.raises(AdapterError, match="mandatory"):
            adapt(frame.drop(columns=["_lon"]), mapping_from_dict(data), fmt="csv")

    def test_shapefile_without_prj_needs_a_declared_crs(self):
        frame = _lonlat_frame(5).rename(columns={"lon": "_lon", "lat": "_lat"})
        data = _lonlat_mapping()
        del data["columns"]["x"], data["columns"]["y"], data["crs"]
        with pytest.raises(AdapterError, match=".prj"):
            adapt(frame, mapping_from_dict(data), fmt="shp",
                  reader_warnings=["the shapefile has no .prj; EPSG:4326 is assumed for the centroids"])

    def test_parcel_ids_that_are_not_npn_are_reported(self):
        frame = _lonlat_frame(5).assign(pid=["ABCDEFGHIJKLMNOPQRS"] * 5)
        data = _lonlat_mapping()
        data["columns"]["parcel_id"] = "pid"
        res = adapt(frame, mapping_from_dict(data))
        assert res.report["parcel_id"]["other_17_or_more_chars"] == 5
        assert any("first 17 characters" in w for w in res.report["warnings"])


class TestMojibake:
    @pytest.mark.parametrize("text, expected, changed", [
        ("PeÃ±Ã³n Azul", "Peñón Azul", True),
        ("Peñón Azul", "Peñón Azul", False),          # genuine text untouched
        ("", "", False),
        ("CL 205 # 312 - 40", "CL 205 # 312 - 40", False),
    ])
    def test_repair(self, text, expected, changed):
        assert repair_mojibake(text) == (expected, changed)

    def test_csv_mixing_utf8_and_cp1252_rows(self, tmp_path):
        header = "id,dir,lon,lat,barrio\n".encode("ascii")
        name = "Peñón Azul"
        rows = [f"R{i},KR {CARRERA_BASE + i} # {CALLE_BASE} - 1{i},-72.70{i},4.41{i},".encode("ascii")
                + name.encode("utf-8" if i % 2 else "cp1252") + b"\n" for i in range(6)]
        path = tmp_path / "mixed.csv"
        path.write_bytes(header + b"".join(rows))
        data = _lonlat_mapping(input=str(path))
        data["columns"]["neighborhood"] = "barrio"
        mapping = mapping_from_dict(data)
        frame, fmt, warnings = read_source(mapping)
        res = adapt(frame, mapping, fmt=fmt, reader_warnings=warnings)
        assert res.report["mojibake_repaired_cells"] == 3
        assert [v["name"] for v in res.codes["neighborhoods"].values()] == [name]


class TestReadSource:
    def test_missing_file_is_an_input_error(self, tmp_path):
        with pytest.raises(InputReadError):
            read_source(mapping_from_dict(_lonlat_mapping(input=str(tmp_path / "nope.csv"))))

    def test_no_input_anywhere(self):
        with pytest.raises(AdapterError, match="no input"):
            read_source(mapping_from_dict(_lonlat_mapping()))

    def test_geojson_points(self, tmp_path):
        frame = _lonlat_frame(4)
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
             "properties": {"id": r.id, "dir": r.dir}} for r in frame.itertuples()]}
        path = tmp_path / "p.geojson"
        path.write_text(json.dumps(fc), encoding="utf-8")
        data = {"columns": {"address": "dir", "record_id": "id"}, "input": str(path)}
        mapping = mapping_from_dict(data)
        got, fmt, warnings = read_source(mapping)
        res = adapt(got, mapping, fmt=fmt, reader_warnings=warnings)
        assert fmt == "geojson" and res.report["kept_rows"] == 4
        assert res.clean["lon"].iloc[0] == pytest.approx(frame["lon"].iloc[0])


# ---------------------------------------------------------------------------------------------------------
# prepare_city script (exit codes, files, build_index integration)
# ---------------------------------------------------------------------------------------------------------

class TestPrepareCityScript:
    def _write(self, tmp_path, frame: pd.DataFrame, mapping: dict) -> str:
        frame.to_csv(tmp_path / "registros.csv", index=False)
        mapping = {**mapping, "input": "registros.csv"}
        (tmp_path / "city.json").write_text(json.dumps(mapping), encoding="utf-8")
        return str(tmp_path / "city.json")

    @needs_index_flow
    def test_staging_and_index(self, tmp_path, capsys):
        mp = self._write(tmp_path, _city(300), _mapping())
        code = _script("prepare_city").main(["--mapping", mp, "--out-dir", str(tmp_path / "st"),
                                             "--index-dir", str(tmp_path / "ix")])
        assert code == 0
        for name in ("nomenclatura_clean.parquet", "quarantine.parquet", "adapter_report.json", "city_codes.json",
                     "nomenclatura_cleaning_report.json", "audit_catastro_docs_empty.parquet",
                     "train_pairs_empty.parquet"):
            assert (tmp_path / "st" / name).is_file(), name
        meta = json.loads((tmp_path / "ix" / "index_meta.json").read_text(encoding="utf-8"))
        report = json.loads((tmp_path / "st" / "adapter_report.json").read_text(encoding="utf-8"))
        assert meta["accounting"]["ok"] and meta["accounting"]["quarantined"] == report["quarantined_rows"]
        assert meta["accounting"]["accounting_total_with_quarantine"] == report["input_rows"]
        assert meta["parse_failure_rate"] == 0.0 and meta["n_docs"] > 0

    def test_next_command_is_printed_without_index_dir(self, tmp_path, capsys):
        mp = self._write(tmp_path, _city(100), _mapping())
        assert _script("prepare_city").main(["--mapping", mp, "--out-dir", str(tmp_path / "st")]) == 0
        assert "next: python scripts/build_index.py" in capsys.readouterr().out

    def test_non_empty_out_dir_needs_overwrite(self, tmp_path, capsys):
        mp = self._write(tmp_path, _city(100), _mapping())
        (tmp_path / "st").mkdir()
        (tmp_path / "st" / "old.txt").write_text("x", encoding="utf-8")
        assert _script("prepare_city").main(["--mapping", mp, "--out-dir", str(tmp_path / "st")]) == 2
        assert "--overwrite" in capsys.readouterr().err

    def test_gate_failure_writes_quarantine_but_not_the_clean_table(self, tmp_path, capsys):
        frame = _city(100)
        frame.loc[frame.index[:20], "coord_x"] = np.nan
        mp = self._write(tmp_path, frame, _mapping())
        assert _script("prepare_city").main(["--mapping", mp, "--out-dir", str(tmp_path / "st")]) == 2
        err = capsys.readouterr().err
        assert "quarantine_share" in err and "Traceback" not in err
        assert (tmp_path / "st" / "quarantine.parquet").is_file()
        assert not (tmp_path / "st" / "nomenclatura_clean.parquet").exists()

    @pytest.mark.parametrize("content, expected_code, needle", [
        (b"id_registro,dir_texto,coord_x,coord_y\n", 2, "no data rows"),
        (b"", None, "error:"),
    ])
    def test_empty_files(self, tmp_path, capsys, content, expected_code, needle):
        (tmp_path / "registros.csv").write_bytes(content)
        data = _mapping()
        data["columns"] = {"address": "dir_texto", "record_id": "id_registro", "x": "coord_x", "y": "coord_y"}
        data.pop("status_values")
        (tmp_path / "city.json").write_text(json.dumps({**data, "input": "registros.csv"}), encoding="utf-8")
        code = _script("prepare_city").main(["--mapping", str(tmp_path / "city.json"), "--out-dir", str(tmp_path / "st")])
        err = capsys.readouterr().err
        assert code in ((1, 2) if expected_code is None else (expected_code,))
        assert needle in err and "Traceback" not in err

    def test_wrong_crs_exits_2_with_one_line(self, tmp_path, capsys):
        mp = self._write(tmp_path, _city(100), _mapping(crs="EPSG:4326"))
        assert _script("prepare_city").main(["--mapping", mp, "--out-dir", str(tmp_path / "st")]) == 2
        err = capsys.readouterr().err.strip().splitlines()
        assert len(err) == 1 and err[0].startswith("error:")

    def test_missing_input_exits_1(self, tmp_path, capsys):
        (tmp_path / "city.json").write_text(json.dumps({**_mapping(), "input": "absent.csv"}), encoding="utf-8")
        assert _script("prepare_city").main(["--mapping", str(tmp_path / "city.json"),
                                             "--out-dir", str(tmp_path / "st")]) == 1


# ---------------------------------------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------------------------------------

class TestEvalHelpers:
    def test_chance(self):
        assert chance_recall(200)["recall@5"] == pytest.approx(0.025)
        assert chance_recall(3)["recall@20"] == 1.0
        assert np.isnan(chance_recall(0)["recall@1"])

    def test_ece_bounds(self):
        assert ece([1.0, 1.0], [1, 1]) == 0.0
        assert ece([1.0, 1.0], [0, 0]) == 1.0
        assert np.isnan(ece([], []))

    def test_paired_bootstrap(self):
        out = paired_bootstrap([0, 0, 0, 0], [1, 1, 1, 1], resamples=50)
        assert out["diff"] == 1.0 and out["ci"] == [1.0, 1.0]
        with pytest.raises(ValueError):
            paired_bootstrap([0], [0, 1])
        assert np.isnan(paired_bootstrap([], [])["diff"])

    def test_wilson_matches_the_reporting_helper(self):
        assert np.isnan(wilson_interval(0, 0)[0])
        with pytest.raises(ValueError):
            wilson_interval(3, 2)
        lo, hi = wilson_interval(50, 100)
        assert lo < 0.5 < hi
        try:
            from cali_address.calibration_report import wilson_interval as reference
        except ImportError:
            pytest.skip("calibration_report not in this checkout")
        for k, n in ((0, 10), (7, 10), (10, 10), (289, 300)):
            assert wilson_interval(k, n) == pytest.approx(reference(k, n))

    def test_eval_queries_use_held_out_noise(self):
        docs = pd.DataFrame({"direccion": [f"CL {CALLE_BASE + i} # {CARRERA_BASE} - 10" for i in range(30)]})
        texts, ids, is_val = eval_queries(docs, n_docs=10, variants=2)
        assert len(texts) == 20 and len(set(ids)) == 10 and not is_val.any()
        texts_all, ids_all, _ = eval_queries(docs, n_docs=1000, variants=1)
        assert len(set(ids_all)) == 30
        assert eval_queries(docs.iloc[0:0])[0] == []
        with pytest.raises(ValueError):
            eval_queries(docs, variants=0)


# ---------------------------------------------------------------------------------------------------------
# end to end: invented city -> adapter -> index -> short CPU training -> evaluation
# ---------------------------------------------------------------------------------------------------------

@needs_index_flow
def test_end_to_end_training_learns_beyond_chance(tmp_path):
    e2e = _script("synthetic_city_e2e")
    # 6 epochs: with a few hundred documents the limit is optimizer steps, not data (documentacion/11, section 6)
    row = e2e.run_size(str(tmp_path), size=600, epochs=6, max_minutes=3.0, seed=7, eval_docs=300, quiet=True)
    assert row["n_docs"] > 400 and row["quarantined_rows"] > 0
    # the bare encoder: untrained is at chance level, trained is far above it
    assert row["encoder_untrained_recall@5"] < 0.1
    assert row["encoder_trained_recall@5"] >= 0.5
    assert row["encoder_trained_recall@5"] > 10 * row["chance_recall@5"]
    assert row["trained_minus_untrained_recall@1_ci"][0] > 0.2
    # the full pipeline emits rows and those rows point at the right parcel
    assert row["coverage"] > 0.3 and row["parcel_precision"] >= 0.9
    assert (tmp_path / "n600" / "index" / "eval_city.json").is_file()
