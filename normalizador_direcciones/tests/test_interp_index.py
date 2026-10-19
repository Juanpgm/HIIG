"""Interpolation index: street pair -> sorted plates per side -> interpolated coordinates.

Synthetic cadastre only. Coordinates are chosen so that every expected value is exact.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import warnings

import numpy as np
import pytest

from conftest import make_stub  # noqa: F401  (puts src/ on sys.path)
from cali_address import interp
from cali_address.interp import (
    CACHE_FILENAME,
    SIDECAR_FILENAME,
    MAX_EXTRAPOLATION_PLATES,
    build_index,
    index_for,
    load_or_build,
    source_hash,
    street_key,
)
from cali_address.parser import parse_address

BASE = 3.4000


def lat_of(plate: int) -> float:
    return round(BASE + plate * 2.5e-5, 6)  # ~2.8 m per plate unit: gaps stay inside the 50 m cap


def cadastre(plates, street="CL 987 # 654", lon=-76.5, extra=()):
    rows = [(f"{street} - {p}", lat_of(p), lon) for p in plates]
    rows.extend(extra)
    return rows


def build(rows):
    return build_index([r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows])


def key(street="CL 987 # 654"):
    return street_key(parse_address(f"{street} - 10"))


EVENS = [10, 14, 20]
ODDS = [11, 15]


@pytest.fixture
def idx():
    return build(cadastre(EVENS + ODDS))


def test_exact_known_plate_returns_that_parcel(idx):
    got = idx.locate(key(), 14)
    assert got.kind == "interp" and got.lat == lat_of(14) and got.lon == -76.5


@pytest.mark.parametrize("plate, expected", [(12, 12), (16, 16), (18, 18)])
def test_even_side_interpolates_linearly_between_neighbours(idx, plate, expected):
    got = idx.locate(key(), plate)
    assert got.kind == "interp"
    assert got.lat == pytest.approx(lat_of(expected), abs=1e-9)


def test_odd_side_uses_only_odd_neighbours(idx):
    got = idx.locate(key(), 13)
    assert got.kind == "interp" and got.lat == pytest.approx(lat_of(13), abs=1e-9)
    # make the even side lie: odd interpolation must not use it
    liar = build(cadastre([10, 14, 20]) + [("CL 987 # 654 - 11", 9.0, -70.0), ("CL 987 # 654 - 15", 9.2, -70.0)])
    got = liar.locate(key(), 13)
    assert got.lat == pytest.approx(9.1)


def test_beyond_the_last_parcel_is_extrapolation(idx):
    got = idx.locate(key(), 22)
    assert got.kind == "extrap" and got.lat == pytest.approx(lat_of(22), abs=1e-9)


def test_before_the_first_parcel_is_extrapolation(idx):
    got = idx.locate(key(), 8)
    assert got.kind == "extrap" and got.lat == pytest.approx(lat_of(8), abs=1e-9)
    assert idx.locate(key(), 0).kind == "extrap"


def test_extrapolation_distance_is_capped(monkeypatch):
    monkeypatch.setattr(interp, "INTERP_MAX_GAP_M", 1e9)  # isolate the plate-unit cap from the metre caps
    monkeypatch.setattr(interp, "INTERP_MAX_EXTRAP_M", 1e9)
    idx = build(cadastre(EVENS))
    near = idx.locate(key(), 20 + MAX_EXTRAPOLATION_PLATES)
    far = idx.locate(key(), 20 + MAX_EXTRAPOLATION_PLATES + 2)
    assert near.kind == "extrap" and near.lat is not None
    assert far.kind == "extrap" and far.lat is None and far.lon is None
    assert idx.locate(key(), 10 - MAX_EXTRAPOLATION_PLATES - 2).lat is None


def test_single_parcel_on_a_face(idx):
    single = build(cadastre([20]))
    assert single.locate(key(), 20).kind == "interp"
    got = single.locate(key(), 22)
    assert got.kind == "extrap" and got.lat == lat_of(20)  # nothing to extend from: its own point
    assert single.locate(key(), 21).kind == "none"  # the odd side has no parcel at all


def test_two_parcels_on_a_face_extend_their_line():
    two = build(cadastre([10, 20]))
    assert two.locate(key(), 24).lat == pytest.approx(lat_of(24), abs=1e-9)   # 24 not 30: 30 is past the 25 m extrapolation cap


@pytest.mark.parametrize("street", ["CL 986 # 654", "CL 987 # 655", "KR 985 # 654", "CL 987 NORTE # 654", "CL 987 A # 654",
                                    "CL 987 # 654 NORTE", "CL 987 BIS # 654"])
def test_other_street_pairs_have_no_face(idx, street):
    assert idx.locate(key(street), 12).kind == "none"


def test_quadrant_position_is_part_of_the_key():
    via_n = street_key(parse_address("CL 987 NORTE # 654 - 10"))
    cross_n = street_key(parse_address("CL 987 # 654 NORTE - 10"))
    plain = street_key(parse_address("CL 987 # 654 - 10"))
    assert len({via_n, cross_n, plain}) == 3
    # a cardinal letter is the spelled quadrant of the same position
    assert street_key(parse_address("CL 987 N # 654 - 10")) == via_n


def test_key_of_a_parse_without_plate_or_street_is_none():
    assert street_key(None) is None
    assert street_key(parse_address("hola mundo")) is None


def test_duplicates_on_one_plate_use_the_median():
    rows = cadastre([10, 20]) + [("CL 987 # 654 - 14", 3.5, -76.5), ("CL 987 # 654 - 14", 3.6, -76.5),
                                 ("CL 987 # 654 - 14", 9.9, -76.5)]
    got = build(rows).locate(key(), 14)
    assert got.lat == pytest.approx(3.6)


def test_leading_zero_plates_and_letters_share_the_numeric_plate():
    rows = [("CL 987 # 654 - 04", 3.40004, -76.5), ("CL 987 # 654 - 10 C", 3.4001, -76.5)]   # ~6.7 m apart (50 m cap)
    got = build(rows).locate(key(), 8)
    assert got.kind == "interp" and got.lat == pytest.approx(3.40008, abs=1e-9)
    assert build(rows).locate(key(), 4).kind == "interp"
    assert build(rows).locate(key(), 10).kind == "interp"


def test_bad_rows_are_skipped_not_fatal():
    rows = [
        ("CL 987 # 654 - 10", float("nan"), -76.5), ("CL 987 # 654 - 12", lat_of(12), float("inf")),
        ("hola mundo", 3.4, -76.5), ("", 3.4, -76.5), (None, 3.4, -76.5), ("CL 987 # 654", 3.4, -76.5),
        ("CL 987 # 654 - 14", lat_of(14), -76.5), ("CL 987 # 654 - 20", lat_of(20), -76.5),
    ]
    idx = build(rows)
    assert idx.locate(key(), 16).lat == pytest.approx(lat_of(16), abs=1e-9)
    assert idx.locate(key(), 12).kind == "extrap"  # only 14 and 20 survive: 12 lies before them


def test_empty_cadastre():
    idx = build_index([], [], [])
    assert idx.locate(key(), 12).kind == "none"
    assert len(idx) == 0


def test_locate_rejects_non_numeric_plates_without_raising(idx):
    assert idx.locate(key(), None).kind == "none"
    assert idx.locate(None, 12).kind == "none"
    assert idx.locate(key(), -4).kind in ("none", "extrap")


def test_build_is_deterministic_and_order_independent():
    rows = cadastre(EVENS + ODDS)
    a, b = build(rows), build(list(reversed(rows)))
    assert a.to_payload() == b.to_payload()
    assert a.to_payload() == build(rows).to_payload()


def test_source_hash_changes_with_any_input():
    d, lo, la = ["CL 987 # 654 - 10"], [-76.5], [3.4]
    h = source_hash(d, lo, la)
    assert h == source_hash(list(d), list(lo), list(la))
    assert h != source_hash(["CL 987 # 654 - 11"], lo, la)
    assert h != source_hash(d, [-76.51], la)
    assert h != source_hash(d, lo, [3.41])
    assert h != source_hash(d + d, lo + lo, la + la)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _arrays():
    rows = cadastre(EVENS + ODDS)
    return [r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows]


def _sidecar(tmp_path):
    return tmp_path / SIDECAR_FILENAME


def test_cache_is_written_with_a_sidecar_and_reused(tmp_path, monkeypatch):
    d, lo, la = _arrays()
    first = load_or_build(d, lo, la, str(tmp_path))
    path = tmp_path / CACHE_FILENAME
    assert CACHE_FILENAME == "interp_index_v1.pkl" and SIDECAR_FILENAME == "interp_index_v1.json" and path.exists()
    meta = json.loads(_sidecar(tmp_path).read_text())
    assert meta["source_hash"] == source_hash(d, lo, la) and meta["version"] == 1
    assert meta["pickle_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    def boom(*a, **k):
        raise AssertionError("must load from the cache")

    monkeypatch.setattr(interp, "build_index", boom)
    again = load_or_build(d, lo, la, str(tmp_path))
    assert again.to_payload() == first.to_payload()


def test_cache_is_rebuilt_when_the_source_hash_differs(tmp_path):
    d, lo, la = _arrays()
    load_or_build(d, lo, la, str(tmp_path))
    d2 = d + ["CL 987 # 654 - 30"]
    idx = load_or_build(d2, lo + [-76.5], la + [lat_of(30)], str(tmp_path))
    assert idx.locate(key(), 26).kind == "interp"
    assert json.loads(_sidecar(tmp_path).read_text())["source_hash"] == source_hash(
        d2, lo + [-76.5], la + [lat_of(30)])


class _NoUnpickle:
    """Fails the test if anything unpickles: the sidecar must be verified first."""

    def __init__(self, monkeypatch):
        class Unverified(BaseException):  # not an Exception: the loader's own except clause must not hide it
            pass

        def boom(*a, **k):
            raise Unverified("pickle.loads called on an unverified cache")

        monkeypatch.setattr(interp.pickle, "loads", boom)


def _evil_pickle():
    class Evil:
        def __reduce__(self):
            return (os.getcwd, ())

    return pickle.dumps(Evil())


@pytest.mark.parametrize("sidecar", [
    None,                                           # missing
    b"",                                            # empty
    b"not json",                                    # corrupt
    b"[1, 2]",                                      # wrong shape
    json.dumps({"version": 99, "source_hash": "x", "pickle_sha256": "y"}).encode(),
    json.dumps({"version": 1, "source_hash": "other", "pickle_sha256": "y"}).encode(),   # hash mismatch
])
def test_a_bad_sidecar_means_the_pickle_is_never_read(tmp_path, monkeypatch, sidecar):
    d, lo, la = _arrays()
    (tmp_path / CACHE_FILENAME).write_bytes(_evil_pickle())
    if sidecar is not None:
        _sidecar(tmp_path).write_bytes(sidecar)
    real_loads = pickle.loads
    _NoUnpickle(monkeypatch)
    idx = load_or_build(d, lo, la, str(tmp_path))  # rebuilt, not unpickled
    assert idx.locate(key(), 12).kind == "interp"
    monkeypatch.undo()
    # and the rebuilt cache is now valid and loadable
    assert real_loads(( tmp_path / CACHE_FILENAME).read_bytes())["version"] == 1
    assert json.loads(_sidecar(tmp_path).read_text())["source_hash"] == source_hash(d, lo, la)


def test_a_tampered_pickle_with_a_matching_source_hash_is_rejected_by_its_digest(tmp_path, monkeypatch):
    d, lo, la = _arrays()
    load_or_build(d, lo, la, str(tmp_path))
    (tmp_path / CACHE_FILENAME).write_bytes(_evil_pickle())  # sidecar still claims the old digest
    _NoUnpickle(monkeypatch)
    idx = load_or_build(d, lo, la, str(tmp_path))
    assert idx.locate(key(), 12).kind == "interp"


@pytest.mark.parametrize("junk", [b"", b"not a pickle", pickle.dumps([1, 2, 3]), pickle.dumps({"version": 99}),
                                  pickle.dumps({"version": 1, "faces": 5})])
def test_corrupt_pickle_with_a_valid_sidecar_is_rebuilt(tmp_path, junk):
    d, lo, la = _arrays()
    (tmp_path / CACHE_FILENAME).write_bytes(junk)
    _sidecar(tmp_path).write_text(json.dumps({
        "version": 1, "source_hash": source_hash(d, lo, la), "pickle_sha256": hashlib.sha256(junk).hexdigest()}))
    idx = load_or_build(d, lo, la, str(tmp_path))
    assert idx.locate(key(), 12).kind == "interp"
    assert pickle.loads((tmp_path / CACHE_FILENAME).read_bytes())["version"] == 1


def test_unwritable_cache_keeps_the_index_in_memory_and_warns_once(tmp_path):
    d, lo, la = _arrays()
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        idx = load_or_build(d, lo, la, str(blocker / "sub"))
    assert idx.locate(key(), 12).kind == "interp"
    assert [str(w.message) for w in caught if "interpolation index" in str(w.message)].__len__() == 1


def test_replace_failure_warns_once_and_leaves_no_partial_file(tmp_path, monkeypatch):
    d, lo, la = _arrays()

    def fail(*a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "replace", fail)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_or_build(d, lo, la, str(tmp_path))
    assert len([w for w in caught if "interpolation index" in str(w.message)]) == 1
    assert not (tmp_path / CACHE_FILENAME).exists() and not _sidecar(tmp_path).exists()
    assert [p.name for p in tmp_path.iterdir()] == []


def test_warm_cache_builds_the_same_cache_the_service_will_load(tmp_path):
    import pandas as pd

    rows = cadastre(EVENS + ODDS)
    pd.DataFrame({"direccion": [r[0] for r in rows], "centroid_lat": [r[1] for r in rows],
                  "centroid_lon": [r[2] for r in rows]}).to_parquet(tmp_path / "catastro_docs.parquet")
    assert interp.warm_cache(str(tmp_path)) is True
    assert (tmp_path / CACHE_FILENAME).exists() and _sidecar(tmp_path).exists()
    stub = make_stub([{"direccion": r[0], "lat": r[1], "lon": r[2]} for r in rows], own_points=False)
    stub.artifacts_dir = str(tmp_path)
    before = (tmp_path / CACHE_FILENAME).read_bytes()
    index_for(stub)  # same source -> loaded, not rebuilt
    assert (tmp_path / CACHE_FILENAME).read_bytes() == before


def test_warm_cache_without_docs_is_a_quiet_false(tmp_path):
    assert interp.warm_cache(str(tmp_path)) is False


def test_no_cache_dir_is_memory_only_and_silent():
    d, lo, la = _arrays()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx = load_or_build(d, lo, la, None)
    assert idx.locate(key(), 12).kind == "interp"


def test_existing_artifacts_are_never_touched(tmp_path):
    d, lo, la = _arrays()
    other = tmp_path / "catastro_docs.parquet"
    other.write_bytes(b"precious")
    load_or_build(d, lo, la, str(tmp_path))
    assert other.read_bytes() == b"precious"


# ---------------------------------------------------------------------------
# lazy, thread-safe attachment to a normalizer
# ---------------------------------------------------------------------------
def _stub(own_points=True):
    docs = [{"direccion": f"CL 987 # 654 - {p}", "lat": lat_of(p), "lon": -76.5} for p in EVENS]
    return make_stub(docs, own_points=own_points)


def test_index_for_is_lazy_memoized_and_uses_own_centroids():
    stub = _stub()
    assert not hasattr(stub, "_interp_index")
    stub._own_lat = np.asarray(stub._own_lat) + 1.0
    a = index_for(stub)
    assert index_for(stub) is a
    assert a.locate(key(), 12).lat == pytest.approx(lat_of(12) + 1.0, abs=1e-9)


def test_index_for_falls_back_to_address_centroids_without_own_points():
    stub = _stub(own_points=False)
    assert index_for(stub).locate(key(), 12).lat == pytest.approx(lat_of(12), abs=1e-9)


def test_index_for_falls_back_per_doc_when_an_own_point_is_nan():
    stub = _stub()
    stub._own_lat = np.asarray(stub._own_lat, dtype=float).copy()
    stub._own_lat[1] = np.nan
    assert index_for(stub).locate(key(), 14).lat == pytest.approx(lat_of(14), abs=1e-9)


def test_index_for_builds_once_under_concurrency(monkeypatch):
    stub = _stub()
    calls = []
    real = interp.load_or_build

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(interp, "load_or_build", counting)
    out = []
    threads = [threading.Thread(target=lambda: out.append(index_for(stub))) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and all(o is out[0] for o in out)
