"""Phase 0.2: WFS download of the nomenclature layer, tested against a fake local HTTP server.

No test touches the real service. Every address below is invented.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import nomenclatura  # noqa: E402

WFS_FIELDS = nomenclatura.WFS_FIELDS


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "script_download_nomenclatura", os.path.join(ROOT, "scripts", "download_nomenclatura.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_feature(i: int, estado: str = "cert", **overrides) -> dict:
    props = {k: None for k in WFS_FIELDS}
    props.update({
        "gid": i + 1,
        "ndid": 900000 + i,
        "ndcomuna": "03",
        "ndidbarrio": "0309",
        "ndbarrio": "Barrio Inventado",
        "ndidpredio": 700000 + i,
        "ndnpn": f"76001010003090015002100000{i:04d}",
        "ndmanzana": "0015",
        "nddirecc": f"KR {10 + i} # 12 - 83",
        "ndnomvial": f"KR {10 + i}",
        "ndtipovia": "Carrera",
        "ndnumvia": str(10 + i),
        "ndplaca": "12 - 83",
        "ndidestado": estado,
        "ndestado": "Certificada",
        "ndfecha": "2021-08-21Z",
        "ndidtipzon": "URB",
        "nddircons2": f"KR {10 + i} 12 83",
        "ndcoordx": 1060967.2254 + i,
        "ndcoordy": 872476.141,
        "ndfechcort": "2026-09-03Z",
    })
    props.update(overrides)
    return {
        "type": "Feature",
        "id": f"layer.{i}",
        "geometry": {"type": "Point", "coordinates": [-76.5 + i * 1e-5, 3.44 + i * 1e-5]},
        "geometry_name": "the_geom",
        "properties": props,
    }


class FakeWfs:
    """A tiny in-process WFS. ``faults[start]`` is a list of faults consumed one per request."""

    def __init__(self, features, matched=None, matched_by_start=None):
        self.features = features
        self.matched = len(features) if matched is None else matched
        self.matched_by_start = matched_by_start or {}
        self.faults: dict[int, list[str]] = {}
        self.requests: list[tuple[int, int]] = []  # (startIndex, count)
        self.params: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, status, body: bytes, ctype="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                start, count = int(q.get("startIndex", 0)), int(q.get("count", 5000))
                with outer._lock:
                    outer.requests.append((start, count))
                    outer.params.append(q)
                    fault = outer.faults.get(start, []).pop(0) if outer.faults.get(start) else None
                if fault == "500":
                    return self._send(500, b"internal error", "text/plain")
                if fault == "503":
                    return self._send(503, b"unavailable", "text/plain")
                if fault == "404":
                    return self._send(404, b"not found", "text/plain")
                if fault == "html":
                    return self._send(200, b"<html><body>Service temporarily unavailable</body></html>", "text/html")
                if fault == "truncated":
                    return self._send(200, b'{"type":"FeatureCollection","numberMatched":12,"features":[{"type":"Feat')
                if fault == "no_features_key":
                    return self._send(200, json.dumps({"exceptions": [{"code": "NoApplicableCode"}]}).encode())
                if fault == "hang":
                    time.sleep(1.5)
                    return self._send(200, b"{}")
                page = outer.features[start:start + count]
                body = {
                    "type": "FeatureCollection",
                    "numberMatched": outer.matched_by_start.get(start, outer.matched),
                    "numberReturned": len(page),
                    "features": page,
                }
                self._send(200, json.dumps(body).encode("utf-8"))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/geoserver/wfs"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def make_server():
    servers = []

    def factory(*a, **kw):
        s = FakeWfs(*a, **kw)
        servers.append(s)
        return s

    yield factory
    for s in servers:
        s.close()


@pytest.fixture
def mod():
    return _load_script()


def run(mod, tmp_path, server, *extra, count=5, retries=3):
    args = [
        "--base-url", server.url,
        "--out", str(tmp_path / "wfs.parquet"),
        "--parts-dir", str(tmp_path / "parts"),
        "--meta", str(tmp_path / "meta.json"),
        "--count", str(count),
        "--retries", str(retries),
        "--backoff", "0",
        "--timeout", "5",
        *extra,
    ]
    return mod.main(args)


def meta(tmp_path) -> dict:
    return json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ paging

def test_pages_until_number_returned_is_below_count(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(12)])
    assert run(mod, tmp_path, srv) == 0
    assert [r[0] for r in srv.requests] == [0, 5, 10]
    df = pd.read_parquet(tmp_path / "wfs.parquet")
    assert len(df) == 12
    assert list(df["gid"]) == list(range(1, 13))
    assert set(WFS_FIELDS) <= set(df.columns) and {"lon", "lat"} <= set(df.columns)
    assert df["lon"].iloc[0] == pytest.approx(-76.5)
    assert df["lat"].iloc[0] == pytest.approx(3.44)


def test_request_parameters_match_the_documented_query(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(3)])
    assert run(mod, tmp_path, srv) == 0
    q = srv.params[0]
    assert q["service"] == "WFS" and q["version"] == "2.0.0" and q["request"] == "GetFeature"
    assert q["typeNames"] == "dapm:pdt_nmc_nomenclatura_domiciliaria"
    assert q["outputFormat"] == "application/json" and q["srsName"] == "EPSG:4326"
    assert q["sortBy"] == "gid" and q["count"] == "5" and q["startIndex"] == "0"


def test_exact_multiple_of_page_size_ends_on_an_empty_page(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(10)])
    assert run(mod, tmp_path, srv) == 0
    assert [r[0] for r in srv.requests] == [0, 5, 10]
    assert len(pd.read_parquet(tmp_path / "wfs.parquet")) == 10


def test_single_short_page(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(2)])
    assert run(mod, tmp_path, srv) == 0
    assert len(srv.requests) == 1


def test_empty_layer_is_a_clear_error_not_an_empty_success(mod, make_server, tmp_path, capsys):
    srv = make_server([])
    assert run(mod, tmp_path, srv) == 2
    assert "no features" in capsys.readouterr().err.lower()
    assert not (tmp_path / "wfs.parquet").exists()


# ------------------------------------------------------------------ resume

def test_resume_does_not_refetch_complete_cached_pages(mod, make_server, tmp_path):
    feats = [make_feature(i) for i in range(12)]
    srv = make_server(feats)
    srv.faults[5] = ["500"] * 10  # page 2 never succeeds in the first run
    assert run(mod, tmp_path, srv, retries=2) == 1
    assert (tmp_path / "parts" / "part_0000000.parquet").exists()
    srv.faults.clear()
    srv.requests.clear()
    assert run(mod, tmp_path, srv) == 0
    assert [r[0] for r in srv.requests] == [5, 10]
    assert len(pd.read_parquet(tmp_path / "wfs.parquet")) == 12


def test_stale_short_cached_page_is_refetched(mod, make_server, tmp_path):
    feats = [make_feature(i) for i in range(7)]
    srv = make_server(feats)
    parts = tmp_path / "parts"
    parts.mkdir()
    pd.DataFrame({"gid": [1, 2]}).to_parquet(parts / "part_0000000.parquet")  # shorter than count=5
    assert run(mod, tmp_path, srv) == 0
    assert [r[0] for r in srv.requests] == [0, 5]
    assert len(pd.read_parquet(tmp_path / "wfs.parquet")) == 7


def test_corrupted_cached_page_is_refetched(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(7)])
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "part_0000000.parquet").write_bytes(b"not a parquet file")
    assert run(mod, tmp_path, srv) == 0
    assert len(pd.read_parquet(tmp_path / "wfs.parquet")) == 7


def test_rerun_after_success_is_idempotent(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(12)])
    assert run(mod, tmp_path, srv) == 0
    first = pd.read_parquet(tmp_path / "wfs.parquet")
    srv.requests.clear()
    assert run(mod, tmp_path, srv) == 0
    assert [r[0] for r in srv.requests] == [10]  # only the short final page is re-checked
    pd.testing.assert_frame_equal(first, pd.read_parquet(tmp_path / "wfs.parquet"))


# ------------------------------------------------------------------ retries and failures

@pytest.mark.parametrize("fault", ["500", "503", "html", "truncated", "no_features_key"])
def test_transient_fault_is_retried_then_succeeds(mod, make_server, tmp_path, fault):
    srv = make_server([make_feature(i) for i in range(7)])
    srv.faults[0] = [fault, fault]
    assert run(mod, tmp_path, srv, retries=4) == 0
    assert [r[0] for r in srv.requests].count(0) == 3
    assert len(pd.read_parquet(tmp_path / "wfs.parquet")) == 7
    assert meta(tmp_path)["retries"] == 2


def test_timeout_is_retried(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(3)])
    srv.faults[0] = ["hang"]
    assert run(mod, tmp_path, srv, "--timeout", "0.3", retries=3) == 0
    assert [r[0] for r in srv.requests].count(0) == 2


@pytest.mark.parametrize("fault", ["500", "html", "truncated", "no_features_key"])
def test_persistent_fault_gives_clear_error_without_traceback(mod, make_server, tmp_path, capsys, fault):
    srv = make_server([make_feature(i) for i in range(3)])
    srv.faults[0] = [fault] * 20
    assert run(mod, tmp_path, srv, retries=2) == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and "startIndex=0" in err
    assert "Traceback" not in err
    assert not (tmp_path / "wfs.parquet").exists()


def test_client_error_is_not_retried_forever(mod, make_server, tmp_path, capsys):
    srv = make_server([make_feature(i) for i in range(3)])
    srv.faults[0] = ["404"] * 20
    assert run(mod, tmp_path, srv, retries=3) == 1
    assert "404" in capsys.readouterr().err
    assert len(srv.requests) <= 3  # bounded by --retries


def test_unreachable_server_is_a_clear_error(mod, tmp_path, capsys):
    rc = mod.main(["--base-url", "http://127.0.0.1:9/geoserver/wfs", "--out", str(tmp_path / "w.parquet"),
                   "--parts-dir", str(tmp_path / "p"), "--meta", str(tmp_path / "m.json"),
                   "--retries", "2", "--backoff", "0", "--timeout", "1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and "Traceback" not in err


# ------------------------------------------------------------------ counts and meta

def test_row_count_must_equal_number_matched(mod, make_server, tmp_path, capsys):
    srv = make_server([make_feature(i) for i in range(12)], matched=20)
    assert run(mod, tmp_path, srv) == 2
    err = capsys.readouterr().err
    assert "12" in err and "20" in err
    assert not (tmp_path / "wfs.parquet").exists()
    assert (tmp_path / "parts" / "part_0000000.parquet").exists()  # kept for a later resume


def test_number_matched_change_between_pages_is_recorded_as_warning(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(12)], matched_by_start={0: 11, 5: 12, 10: 12})
    assert run(mod, tmp_path, srv) == 0
    m = meta(tmp_path)
    assert m["number_matched_first"] == 11 and m["number_matched_last"] == 12
    assert any("numberMatched changed" in w for w in m["warnings"])


def test_duplicates_from_shifting_pages_are_removed_and_counted(mod, make_server, tmp_path):
    feats = [make_feature(i) for i in range(10)]
    feats.insert(5, dict(feats[4]))  # the layer shifted: gid 5 shows up twice
    srv = make_server(feats, matched=10)
    assert run(mod, tmp_path, srv) == 0
    df = pd.read_parquet(tmp_path / "wfs.parquet")
    assert len(df) == 10 and df["gid"].is_unique
    assert meta(tmp_path)["duplicates_removed"] == 1


def test_meta_has_the_documented_fields(mod, make_server, tmp_path):
    feats = [make_feature(i) for i in range(6)] + [make_feature(6, estado="pcer", ndfechcort="2026-09-04Z")]
    srv = make_server(feats)
    assert run(mod, tmp_path, srv) == 0
    m = meta(tmp_path)
    assert m["number_matched_last"] == 7 and m["rows_downloaded"] == 7 and m["pages"] == 2
    assert m["estado_counts"] == {"cert": 6, "pcer": 1}
    assert m["ndfechcort_max"] == "2026-09-04Z"
    assert m["type_names"] == "dapm:pdt_nmc_nomenclatura_domiciliaria" and m["srs_name"] == "EPSG:4326"
    assert m["warnings"] == [] and m["missing_properties"] == [] and m["extra_properties"] == []
    assert "downloaded_at" in m and "source" in m


# ------------------------------------------------------------------ content fidelity

def test_null_fields_are_preserved_for_pcer_rows(mod, make_server, tmp_path):
    feats = [make_feature(0), make_feature(1, estado="pcer", ndnomvial=None, ndtipovia=None, ndnumvia=None,
                                           ndplaca=None, nddirecc="CL 977 E # 931 F 6 - 953 (Dirección Catastral)")]
    srv = make_server(feats)
    assert run(mod, tmp_path, srv) == 0
    df = pd.read_parquet(tmp_path / "wfs.parquet")
    pc = df[df["ndidestado"] == "pcer"].iloc[0]
    assert pd.isna(pc["ndnomvial"]) and pd.isna(pc["ndplaca"]) and pd.isna(pc["ndinterior"])
    assert "Dirección Catastral" in pc["nddirecc"]  # accents survive the round trip
    assert df["ndinterior"].isna().all()  # a column that is null in every row still exists


def test_missing_geometry_and_non_point_geometry_give_nan_coordinates(mod, make_server, tmp_path):
    f_null = make_feature(1)
    f_null["geometry"] = None
    f_line = make_feature(2)
    f_line["geometry"] = {"type": "LineString", "coordinates": [[-76.5, 3.4], [-76.51, 3.41]]}
    f_short = make_feature(3)
    f_short["geometry"] = {"type": "Point", "coordinates": []}
    srv = make_server([make_feature(0), f_null, f_line, f_short])
    assert run(mod, tmp_path, srv) == 0
    df = pd.read_parquet(tmp_path / "wfs.parquet")
    assert df["lon"].notna().tolist() == [True, False, False, False]
    assert df["lat"].notna().tolist() == [True, False, False, False]


def test_extra_and_missing_properties_are_recorded(mod, make_server, tmp_path):
    f = make_feature(0)
    f["properties"]["brand_new_field"] = "x"
    del f["properties"]["ndresoluc"]
    srv = make_server([f, make_feature(1)])
    assert run(mod, tmp_path, srv) == 0
    m = meta(tmp_path)
    assert m["extra_properties"] == ["brand_new_field"]
    # present in another feature, so only absent from the first one: not "missing" overall
    assert m["missing_properties"] == []
    df = pd.read_parquet(tmp_path / "wfs.parquet")
    assert "brand_new_field" in df.columns and "ndresoluc" in df.columns


def test_property_absent_from_every_feature_is_missing_but_column_exists(mod, make_server, tmp_path):
    feats = []
    for i in range(3):
        f = make_feature(i)
        del f["properties"]["ndcodpost"]
        feats.append(f)
    srv = make_server(feats)
    assert run(mod, tmp_path, srv) == 0
    assert meta(tmp_path)["missing_properties"] == ["ndcodpost"]
    assert "ndcodpost" in pd.read_parquet(tmp_path / "wfs.parquet").columns


def test_numeric_columns_keep_numeric_dtypes_even_with_nulls(mod, make_server, tmp_path):
    feats = [make_feature(0), make_feature(1, ndidpredio=None, ndcoordx=None)]
    srv = make_server(feats)
    assert run(mod, tmp_path, srv) == 0
    df = pd.read_parquet(tmp_path / "wfs.parquet")
    assert pd.api.types.is_numeric_dtype(df["ndidpredio"]) and pd.api.types.is_numeric_dtype(df["ndcoordx"])
    assert df["ndcoordx"].isna().sum() == 1


# ------------------------------------------------------------------ comparison with the shapefile

def _wfs_frame(n=10):
    return pd.DataFrame({
        "ndid": np.arange(1, n + 1),
        "ndidestado": ["cert"] * n,
        "ndnpn": [f"76001{i:025d}" for i in range(n)],
        "nddirecc": [f"KR {10 + i} # 5 - {i}" for i in range(n)],
        "lon": -76.5 + np.arange(n) * 1e-4,
        "lat": 3.4 + np.arange(n) * 1e-4,
    })


def _shp_frame(n=10):
    w = _wfs_frame(n)
    return pd.DataFrame({
        "NdID": w["ndid"], "NdEstado": ["cert"] * n, "NdNpn": w["ndnpn"], "NdDirecc": w["nddirecc"],
        "lon": w["lon"], "lat": w["lat"],
    })


def test_compare_identical_inputs_passes():
    r = nomenclatura.compare_wfs_shapefile(_wfs_frame(), _shp_frame())
    assert r["passed"] is True
    assert r["common_ndid"] == 10 and r["npn_agreement"] == 1.0 and r["address_agreement"] == 1.0
    assert r["coord_max_shift_m"] < 0.01


def test_compare_cert_count_difference_boundary():
    shp = _shp_frame(1000)
    ok = _wfs_frame(1000).iloc[:996]  # 0.4 % short: inside the < 0.5 % tolerance
    assert nomenclatura.compare_wfs_shapefile(ok, shp)["cert_count_ok"] is True
    bad = _wfs_frame(1000).iloc[:994]  # 0.6 % short
    r = nomenclatura.compare_wfs_shapefile(bad, shp)
    assert r["cert_count_ok"] is False and r["passed"] is False


def test_compare_counts_only_cert_rows_of_the_wfs():
    wfs = pd.concat([_wfs_frame(10), _wfs_frame(10).assign(ndid=np.arange(101, 111), ndidestado="pcer")])
    r = nomenclatura.compare_wfs_shapefile(wfs, _shp_frame(10))
    assert r["wfs_cert_count"] == 10 and r["wfs_pcer_count"] == 10 and r["cert_count_ok"] is True


def test_compare_npn_disagreement_above_threshold_fails():
    wfs = _wfs_frame(200)
    shp = _shp_frame(200)
    shp.loc[:5, "NdNpn"] = "760019999999999999999999999999"  # 6 / 200 = 3 %
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["npn_agreement"] == pytest.approx(194 / 200)
    assert r["npn_ok"] is False and r["passed"] is False


def test_compare_address_is_case_and_whitespace_insensitive():
    shp = _shp_frame(10)
    shp["NdDirecc"] = shp["NdDirecc"].str.lower().str.replace(" ", "   ")
    assert nomenclatura.compare_wfs_shapefile(_wfs_frame(10), shp)["address_agreement"] == 1.0


def test_compare_ignores_null_and_zero_ndid_as_keys():
    wfs = _wfs_frame(5)
    shp = _shp_frame(5)
    wfs.loc[0, "ndid"] = 0
    shp.loc[0, "NdID"] = 0  # both zero: must NOT be matched as a common id
    wfs.loc[1, "ndid"] = np.nan
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["common_ndid"] == 3


def test_compare_duplicate_ndid_does_not_inflate_the_common_set():
    wfs = pd.concat([_wfs_frame(5), _wfs_frame(5).iloc[:2]])
    r = nomenclatura.compare_wfs_shapefile(wfs, _shp_frame(5))
    assert r["common_ndid"] == 5 and r["wfs_duplicate_ndid"] == 2


def test_compare_coordinate_shift_above_one_metre_is_counted():
    wfs = _wfs_frame(100)
    shp = _shp_frame(100)
    shp.loc[:2, "lon"] += 0.0001  # ~11 m
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["coord_over_1m"] == 3
    assert r["coord_max_shift_m"] > 5
    assert r["coord_share_within_1m"] == pytest.approx(0.97)
    assert r["coord_ok"] is False  # 3 % of the points are 11 m away: p99 is far above 2 m
    assert r["passed"] is False


def test_compare_coord_gate_passes_when_a_tiny_share_exceeds_one_metre():
    wfs = _wfs_frame(1000)
    shp = _shp_frame(1000)
    shp.loc[:2, "lon"] += 0.0001  # 3 / 1000 = 0.3 % over 1 m: median and p99 are still tiny
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["coord_over_1m"] == 3 and r["coord_ok"] is True


def test_compare_nan_coordinates_do_not_raise():
    wfs = _wfs_frame(10)
    wfs.loc[0, "lon"] = np.nan
    r = nomenclatura.compare_wfs_shapefile(wfs, _shp_frame(10))
    assert r["coord_compared"] == 9


def test_compare_empty_frames_fail_cleanly():
    r = nomenclatura.compare_wfs_shapefile(_wfs_frame(0), _shp_frame(0))
    assert r["passed"] is False and r["common_ndid"] == 0
    assert r["npn_agreement"] is None


def test_compare_no_common_ids_fails_cleanly():
    shp = _shp_frame(10).assign(NdID=np.arange(500, 510))
    r = nomenclatura.compare_wfs_shapefile(_wfs_frame(10), shp)
    assert r["passed"] is False and r["common_ndid"] == 0


def test_coord_shift_by_ndid_returns_metres_per_common_id():
    wfs = _wfs_frame(4)
    shp = _shp_frame(4)
    shp.loc[2, "lat"] += 0.0001
    s = nomenclatura.coord_shift_by_ndid(wfs, shp)
    assert set(s.index) == {1, 2, 3, 4}
    assert s.loc[3] == pytest.approx(11.1, abs=0.5)
    assert s.loc[1] < 0.01


def test_validate_shp_writes_json_and_returns_zero_when_in_tolerance(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(6)])
    shp_df = pd.DataFrame({
        "NdID": [900000 + i for i in range(6)],
        "NdEstado": ["cert"] * 6,
        "NdNpn": [f"76001010003090015002100000{i:04d}" for i in range(6)],
        "NdDirecc": [f"KR {10 + i} # 12 - 83" for i in range(6)],
        "lon": [-76.5 + i * 1e-5 for i in range(6)],
        "lat": [3.44 + i * 1e-5 for i in range(6)],
    })
    shp_path = tmp_path / "shp_4326.parquet"
    shp_df.to_parquet(shp_path)
    vs = tmp_path / "vs.json"
    rc = run(mod, tmp_path, srv, "--validate-shp", str(shp_path), "--vs-json", str(vs))
    assert rc == 0
    rep = json.loads(vs.read_text(encoding="utf-8"))
    assert rep["passed"] is True and rep["common_ndid"] == 6


def test_validate_shp_returns_three_when_out_of_tolerance(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(6)])
    shp_df = pd.DataFrame({
        "NdID": [900000 + i for i in range(6)], "NdEstado": ["cert"] * 6,
        "NdNpn": ["760019999999999999999999999999"] * 6,
        "NdDirecc": [f"KR {10 + i} # 12 - 83" for i in range(6)],
        "lon": [-76.5 + i * 1e-5 for i in range(6)], "lat": [3.44 + i * 1e-5 for i in range(6)],
    })
    shp_path = tmp_path / "shp_4326.parquet"
    shp_df.to_parquet(shp_path)
    rc = run(mod, tmp_path, srv, "--validate-shp", str(shp_path), "--vs-json", str(tmp_path / "vs.json"))
    assert rc == 3


def test_validate_shp_missing_file_is_a_clear_error(mod, make_server, tmp_path, capsys):
    srv = make_server([make_feature(i) for i in range(3)])
    rc = run(mod, tmp_path, srv, "--validate-shp", str(tmp_path / "nope.parquet"),
             "--vs-json", str(tmp_path / "vs.json"))
    assert rc == 1
    assert "nope.parquet" in capsys.readouterr().err


# ------------------------------------------------------------------ diagnostics added after the first real run

def test_compare_reports_lot_level_npn_agreement_separately():
    wfs = _wfs_frame(10)
    shp = _shp_frame(10)
    shp["NdNpn"] = shp["NdNpn"].str[:-2] + "99"  # same lot (first 21 chars), different unit suffix
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["npn_agreement"] == 0.0
    assert r["npn_lot_agreement"] == 1.0
    assert r["npn_ok"] is True  # the NPN gate is lot-level; full-NPN agreement is informational
    assert r["passed"] is True


def test_compare_lot_agreement_is_none_without_common_ids():
    r = nomenclatura.compare_wfs_shapefile(_wfs_frame(0), _shp_frame(0))
    assert r["npn_lot_agreement"] is None


def test_compare_reports_share_of_identical_projected_coordinates_when_available():
    wfs = _wfs_frame(10).assign(ndcoordx=1000.0, ndcoordy=2000.0)
    shp = _shp_frame(10).assign(x_6249=1000.0, y_6249=2000.0)
    shp.loc[:1, "x_6249"] = 1500.0
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["source_coords_identical_share"] == pytest.approx(0.8)


def test_compare_skips_projected_coordinate_diagnostic_when_columns_are_missing():
    r = nomenclatura.compare_wfs_shapefile(_wfs_frame(5), _shp_frame(5))
    assert r["source_coords_identical_share"] is None


def test_compare_reports_shift_quantiles():
    r = nomenclatura.compare_wfs_shapefile(_wfs_frame(100), _shp_frame(100))
    assert r["coord_p99_shift_m"] is not None and r["coord_p99_shift_m"] < 0.01


# ------------------------------------------------------------------ gate corrections (plan 8.1)

def test_default_thresholds_are_lot_level_npn_and_median_p99_coordinates():
    th = nomenclatura.compare_wfs_shapefile(_wfs_frame(10), _shp_frame(10))["thresholds"]
    assert th["min_npn_lot_agreement"] == 0.98
    assert th["max_median_shift_m"] == 1.0 and th["max_p99_shift_m"] == 2.0
    assert th["max_count_diff"] == 0.005 and th["min_agreement"] == 0.995


def _lot_frames(n, n_different_lot):
    """n common ids; the first ``n_different_lot`` have a different lot (first 21 NPN characters)."""
    wfs, shp = _wfs_frame(n), _shp_frame(n)
    shp.loc[: n_different_lot - 1, "NdNpn"] = "760019999999999999999" + "999999999"
    return wfs, shp


def test_npn_gate_passes_exactly_at_98_percent_lot_agreement():
    r = nomenclatura.compare_wfs_shapefile(*_lot_frames(200, 4))          # 196 / 200 = 0.98
    assert r["npn_lot_agreement"] == 0.98
    assert r["npn_ok"] is True and r["passed"] is True


def test_npn_gate_fails_just_below_98_percent_lot_agreement():
    r = nomenclatura.compare_wfs_shapefile(*_lot_frames(200, 5))          # 195 / 200 = 0.975
    assert r["npn_lot_agreement"] == pytest.approx(0.975)
    assert r["npn_ok"] is False and r["passed"] is False


def test_full_npn_agreement_is_informational_only():
    wfs, shp = _wfs_frame(100), _shp_frame(100)
    shp["NdNpn"] = shp["NdNpn"].str[:-2] + "ZZ"                            # every unit suffix differs
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["npn_agreement"] == 0.0 and r["npn_lot_agreement"] == 1.0
    assert r["npn_ok"] is True and r["passed"] is True


def test_npn_gate_threshold_is_configurable_and_inclusive():
    wfs, shp = _lot_frames(10, 1)                                          # 0.9
    assert nomenclatura.compare_wfs_shapefile(wfs, shp, min_npn_lot_agreement=0.9)["npn_ok"] is True
    assert nomenclatura.compare_wfs_shapefile(wfs, shp, min_npn_lot_agreement=0.91)["npn_ok"] is False


def test_npn_lot_gate_ignores_short_npns_without_raising():
    wfs, shp = _wfs_frame(10), _shp_frame(10)
    wfs.loc[0, "ndnpn"] = None
    shp.loc[1, "NdNpn"] = "7600"
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["npn_lot_agreement"] == pytest.approx(0.8)


def _shifted(n, shift_lon, which):
    wfs, shp = _wfs_frame(n), _shp_frame(n)
    shp.loc[which, "lon"] += shift_lon
    return wfs, shp


def test_coordinate_gate_is_inclusive_at_the_median_and_p99_thresholds():
    wfs, shp = _shifted(100, 1.5 / 111_320.0 / 0.9984, slice(None))        # every point ~1.5 m away
    r = nomenclatura.compare_wfs_shapefile(wfs, shp, max_median_shift_m=10.0, max_p99_shift_m=10.0)
    med, p99 = r["coord_median_shift_m"], r["coord_p99_shift_m"]
    assert 1.0 < med < 2.0 and r["coord_ok"] is True
    at = nomenclatura.compare_wfs_shapefile(wfs, shp, max_median_shift_m=med, max_p99_shift_m=p99)
    assert at["coord_ok"] is True                                          # exactly at both thresholds
    below_med = nomenclatura.compare_wfs_shapefile(wfs, shp, max_median_shift_m=np.nextafter(med, 0),
                                                   max_p99_shift_m=p99)
    assert below_med["coord_ok"] is False                                  # median just beyond
    below_p99 = nomenclatura.compare_wfs_shapefile(wfs, shp, max_median_shift_m=med,
                                                   max_p99_shift_m=np.nextafter(p99, 0))
    assert below_p99["coord_ok"] is False                                  # p99 just beyond


def test_a_median_shift_above_one_metre_fails_even_when_p99_is_within_two():
    wfs, shp = _shifted(100, 1.5 / 111_320.0, slice(None))                 # ~1.5 m for everyone
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert 1.0 < r["coord_median_shift_m"] < 2.0 and r["coord_p99_shift_m"] <= 2.0
    assert r["coord_ok"] is False and r["passed"] is False


def test_a_p99_above_two_metres_fails_even_when_the_median_is_tiny():
    wfs, shp = _shifted(100, 3.0 / 111_320.0, slice(0, 3))                 # 3 % of the points ~3 m away
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["coord_median_shift_m"] < 0.01 and r["coord_p99_shift_m"] > 2.0
    assert r["coord_ok"] is False


def test_the_real_run_pattern_passes_the_corrected_gate():
    # median ~0.72 m and p99 ~1.4 m (the datum difference found on the real data) must pass
    wfs, shp = _shifted(1000, 0.72 / 111_320.0, slice(None))
    shp.loc[:9, "lon"] += 0.7 / 111_320.0                                  # 1 % at ~1.4 m
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["coord_median_shift_m"] < 1.0 and r["coord_p99_shift_m"] < 2.0
    assert r["coord_ok"] is True and r["passed"] is True


def test_report_keeps_the_share_over_one_and_two_metres_as_information():
    wfs, shp = _shifted(100, 1.5 / 111_320.0, list(range(10)))
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["coord_over_1m"] == 10 and r["coord_over_2m"] == 0
    assert r["coord_share_within_2m"] == 1.0


def test_gate_with_no_coordinate_pairs_fails_cleanly():
    wfs, shp = _wfs_frame(5), _shp_frame(5)
    wfs[["lon", "lat"]] = np.nan
    r = nomenclatura.compare_wfs_shapefile(wfs, shp)
    assert r["coord_compared"] == 0 and r["coord_ok"] is False and r["coord_median_shift_m"] is None


def test_validate_shp_returns_zero_when_only_the_unit_suffix_differs(mod, make_server, tmp_path):
    srv = make_server([make_feature(i) for i in range(6)])
    shp_df = pd.DataFrame({
        "NdID": [900000 + i for i in range(6)], "NdEstado": ["cert"] * 6,
        "NdNpn": [f"76001010003090015002100000{i:04d}"[:28] + "99" for i in range(6)],
        "NdDirecc": [f"KR {10 + i} # 12 - 83" for i in range(6)],
        "lon": [-76.5 + i * 1e-5 for i in range(6)], "lat": [3.44 + i * 1e-5 for i in range(6)],
    })
    shp_path = tmp_path / "shp_4326.parquet"
    shp_df.to_parquet(shp_path)
    vs = tmp_path / "vs.json"
    assert run(mod, tmp_path, srv, "--validate-shp", str(shp_path), "--vs-json", str(vs)) == 0
    rep = json.loads(vs.read_text(encoding="utf-8"))
    assert rep["npn_ok"] is True and rep["npn_agreement"] < 1.0 and rep["npn_lot_agreement"] == 1.0
