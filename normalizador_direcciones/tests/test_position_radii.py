"""Round 5: geometric position radii and interpolation caps (no GPS anywhere).

Derived from the cadastre itself (see ``interp.py`` for the rule and the numbers):

* ``predio`` 15 m, ``lote`` 44 m, ``manzana`` 56 m, ``via`` 21 m: p90 of the geometric extent / the
  leave-one-out extrapolation error;
* ``tramo``: p90 of the leave-one-out interpolation error as a STEP function of the gap between the two
  flanking parcels (bins (0, 25], (25, 50], (50, 100], (100, 150], beyond), never below the predio radius;
* ``INTERP_MAX_GAP_M`` = 50 (largest bin edge whose p90 stays <= 30 m), ``INTERP_MAX_PLATE_GAP`` = 60 (the same
  rule over plate-number gaps), ``INTERP_MAX_EXTRAP_M`` = 25.

Synthetic addresses only. The face ``CL 987 # 654`` uses ``lat = 3.4 + k * step`` (1e-4 deg is about 11.1 m).
"""

from __future__ import annotations

import math

import pytest
from conftest import make_stub

from cali_address import interp, service
from cali_address.interp import Located, build_index, extrap_radius_m, step_radius, street_key, tramo_radius_m
from cali_address.parser import parse_address
from cali_address.service import POSITION_RADIUS_M, normalize_strict

LON = -76.5
M_PER_DEG = 111195.0  # metres per degree of latitude (the index uses a 6371008.8 m sphere)
KEY = street_key(parse_address("CL 987 # 654 - 10"))


def deg(metres):
    return metres / M_PER_DEG


def idx_m(points):
    """Index with ``(plate, metres north of the origin)`` pairs on the face CL 987 # 654."""
    rows = [(f"CL 987 # 654 - {p}", 3.4 + deg(m), LON) for p, m in points]
    return build_index([r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows])


def doc(plate, metres, **extra):
    d = {"direccion": f"CL 987 # 654 - {plate}", "lat": round(3.4 + deg(metres), 7), "lon": LON,
         "predial": f"P{plate}", "manzana": "MZ7"}
    d.update(extra)
    return d


def run(query, docs, **kw):
    return normalize_strict(make_stub(docs), [query], gazetteer=None, **kw).iloc[0]


# ---------------------------------------------------------------------------
# constants and step functions
# ---------------------------------------------------------------------------
def test_tier_radii_are_the_geometric_p90s():
    assert POSITION_RADIUS_M == {"predio": 15, "lote": 44, "manzana": 56, "tramo": 23, "via": 21}


def test_caps_are_the_derived_values():
    assert interp.INTERP_MAX_GAP_M == 50.0
    assert interp.INTERP_MAX_PLATE_GAP == 60
    assert interp.INTERP_MAX_EXTRAP_M == 25.0
    assert interp.MAX_EXTRAPOLATION_PLATES == 50   # not re-derived: the metre cap binds first


def test_step_tables_are_monotone_and_end_open():
    for steps in (interp.TRAMO_RADIUS_STEPS, interp.EXTRAP_RADIUS_STEPS):
        edges = [e for e, _ in steps]
        radii = [r for _, r in steps]
        assert edges == sorted(edges) and edges[:-1] == [25.0, 50.0, 100.0, 150.0] and math.isinf(edges[-1])
        assert radii == sorted(radii) and len(set(radii)) == len(radii)


@pytest.mark.parametrize(("gap", "expected"), [
    (0.0, 15.0), (0.01, 15.0), (25.0, 15.0),              # bin (0, 25]: p90 4 m, floored at the predio radius
    (25.0001, 23.0), (50.0, 23.0),                          # bin (25, 50]
    (50.0001, 61.0), (100.0, 61.0),                         # bin (50, 100]
    (100.0001, 102.0), (150.0, 102.0),                      # bin (100, 150]
])
def test_tramo_radius_is_a_right_closed_step_function_of_the_gap(gap, expected):
    assert tramo_radius_m(gap) == expected


@pytest.mark.parametrize("gap", [150.0001, 1e6, math.inf, None, math.nan, -1.0, "x"])
def test_tramo_radius_beyond_or_unknown_gap_is_the_worst_bin(gap):
    assert tramo_radius_m(gap) == interp.TRAMO_RADIUS_STEPS[-1][1] > 102


@pytest.mark.parametrize(("dist", "expected"), [
    (0.0, 10.0), (25.0, 10.0), (25.0001, 56.0), (50.0, 56.0), (50.0001, 131.0), (100.0, 131.0),
    (100.0001, 199.0), (150.0, 199.0),
])
def test_extrapolation_radius_steps(dist, expected):
    assert extrap_radius_m(dist) == expected


def test_extrapolation_radius_unknown_distance_is_the_worst_bin():
    assert extrap_radius_m(None) == extrap_radius_m(math.nan) == interp.EXTRAP_RADIUS_STEPS[-1][1]


def test_step_radius_generic_edges():
    steps = ((1.0, 5.0), (2.0, 9.0), (math.inf, 20.0))
    assert [step_radius(steps, v) for v in (0, 1, 1.5, 2, 3)] == [5.0, 5.0, 9.0, 9.0, 20.0]


# ---------------------------------------------------------------------------
# Located.gap_m and the caps at their bin edges
# ---------------------------------------------------------------------------
def test_located_carries_the_flanking_gap_in_metres():
    index = idx_m([(10, 0.0), (14, 40.0)])
    got = index.locate(KEY, 12)
    assert got.kind == "interp" and got.gap_m == pytest.approx(40.0, abs=0.5)
    assert index.locate(KEY, 10).gap_m == 0.0                  # a known plate is exact


def test_located_default_gap_is_none():
    assert Located("none").gap_m is None


def test_interpolation_cap_boundary_in_metres():
    inside = idx_m([(10, 0.0), (14, 49.5)]).locate(KEY, 12)
    edge = idx_m([(10, 0.0), (14, 50.0)]).locate(KEY, 12)
    outside = idx_m([(10, 0.0), (14, 50.6)]).locate(KEY, 12)
    assert inside.kind == "interp" and edge.kind in ("interp", "gap")
    assert outside.kind == "gap" and outside.gap_m == pytest.approx(50.6, abs=0.3)


def test_interpolation_cap_is_inclusive_on_the_constant(monkeypatch):
    index = idx_m([(10, 0.0), (14, 30.0)])
    gap = index.locate(KEY, 12).gap_m
    monkeypatch.setattr(interp, "INTERP_MAX_GAP_M", gap)
    assert index.locate(KEY, 12).kind == "interp"
    monkeypatch.setattr(interp, "INTERP_MAX_GAP_M", gap - 1e-6)
    assert index.locate(KEY, 12).kind == "gap"


@pytest.mark.parametrize(("gap_plates", "kind"), [(58, "interp"), (60, "interp"), (62, "gap")])
def test_plate_gap_cap_boundaries(gap_plates, kind):
    index = idx_m([(10, 0.0), (10 + gap_plates, 1.0)])           # tiny metre gap: only the plate cap binds
    assert index.locate(KEY, 12).kind == kind


def test_extrapolation_cap_boundary_in_metres():
    # plates 10 and 12 are 20 m apart: plate 14 extends 20 m past 12 (inside), plate 16 would be 40 m (outside)
    index = idx_m([(10, 0.0), (12, 20.0)])
    near, far = index.locate(KEY, 14), index.locate(KEY, 16)
    assert near.kind == "extrap" and near.lat is not None and near.gap_m == pytest.approx(20.0, abs=0.3)
    assert far.kind == "extrap" and far.lat is None and far.gap_m == pytest.approx(40.0, abs=0.5)
    edge = idx_m([(10, 0.0), (12, 24.9)]).locate(KEY, 14)       # just inside 25 m past the last parcel
    assert edge.lat is not None


def test_extrapolation_beyond_the_plate_cap_has_no_distance():
    index = idx_m([(10, 0.0), (12, 2.0)])
    got = index.locate(KEY, 12 + interp.MAX_EXTRAPOLATION_PLATES + 2)
    assert got.kind == "extrap" and got.lat is None and got.gap_m is None


def test_single_parcel_face_extrapolation_reports_no_distance():
    got = idx_m([(20, 0.0)]).locate(KEY, 22)
    assert got.kind == "extrap" and got.lat is not None and got.gap_m is None


def test_empty_none_and_malformed_plates():
    index = idx_m([(10, 0.0), (14, 10.0)])
    for bad in (None, "", "x", math.nan):
        assert index.locate(KEY, bad).kind == "none" and index.locate(KEY, bad).gap_m is None
    assert index.locate(None, 12).kind == "none"
    assert interp.InterpIndex().locate(KEY, 12).kind == "none"


# ---------------------------------------------------------------------------
# service: the radius follows the actual gap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("gap_m", "radius"), [(12.0, 15.0), (24.0, 15.0), (30.0, 23.0), (48.0, 23.0)])
def test_service_tramo_radius_follows_the_flanking_gap(gap_m, radius):
    docs = [doc(10, 0.0), doc(14, gap_m), doc(11, 500.0), doc(15, 510.0)]
    row = run("CL 987 # 654 - 12", docs)
    assert row["position_tier"] == "tramo" and row["position_source"] == "interp"
    assert row["position_radius_m"] == radius


def test_service_plate_found_with_a_letter_difference_is_a_tramo_with_the_smallest_radius():
    docs = [doc(12, 20.0), doc(10, 0.0), doc(14, 40.0)]
    row = run("CL 987 # 654 - 12 A", docs)
    assert row["position_tier"] == "tramo" and row["position_radius_m"] == 15.0


def test_service_gap_beyond_the_cap_is_via_without_coordinates_and_reports_its_step_radius():
    docs = [doc(10, 0.0), doc(14, 80.0), doc(11, 500.0), doc(15, 510.0)]
    row = run("CL 987 # 654 - 12", docs)
    assert row["position_tier"] == "via" and row["lat"] is None and row["position_source"] == "none"
    assert row["position_radius_m"] == 61.0                     # the (50, 100] step of the flanking gap


def test_service_extrapolation_radius_follows_the_distance_from_the_last_parcel():
    near = run("CL 987 # 654 - 14", [doc(12, 20.0), doc(10, 0.0)])     # 20 m per two plates
    far = run("CL 987 # 654 - 14", [doc(12, 40.0), doc(10, 0.0)])      # 40 m per two plates
    assert near["position_tier"] == far["position_tier"] == "via"
    assert near["lat"] is None and far["lat"] is None            # withheld by default
    assert near["position_radius_m"] == 10.0                      # 20 m past the last parcel
    assert far["position_radius_m"] == 56.0                       # 40 m past the last parcel


def test_service_extrapolated_coordinates_use_the_flag_and_the_new_cap(monkeypatch):
    monkeypatch.setattr(service, "EXTRAPOLATED_COORDS_ENABLED", True)
    near = run("CL 987 # 654 - 14", [doc(12, 20.0), doc(10, 0.0)])
    far = run("CL 987 # 654 - 14", [doc(12, 40.0), doc(10, 0.0)])
    assert near["lat"] is not None and near["position_source"] == "interp"
    assert far["lat"] is None                                     # 40 m > INTERP_MAX_EXTRAP_M


def test_service_every_ok_row_radius_is_a_table_value():
    docs = [doc(10, 0.0), doc(14, 30.0), doc(11, 500.0)]
    table = {float(v) for v in POSITION_RADIUS_M.values()} | {r for _, r in interp.TRAMO_RADIUS_STEPS} | {
        r for _, r in interp.EXTRAP_RADIUS_STEPS}
    for query in ("CL 987 # 654 - 10", "CL 987 # 654 - 12", "CL 987 # 654 - 16", "CL 987 # 654"):
        row = run(query, docs)
        if row["estado"] == "OK":
            assert float(row["position_radius_m"]) in table | {15.0}


def test_service_is_deterministic_for_a_tramo_row():
    docs = [doc(10, 0.0), doc(14, 30.0)]
    a, b = run("CL 987 # 654 - 12", docs), run("CL 987 # 654 - 12", docs)
    assert a.equals(b)
