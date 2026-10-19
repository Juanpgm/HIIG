"""Fidelity wiring in ``normalize_strict``: columns, original-raw comparison and the tier-cap policy.

Synthetic cadastre only (stub normalizer). The policy tests force the report through a monkeypatched
``assess_fidelity`` so each risk level is exercised deterministically, plus end-to-end rows that are
faithful (never capped) and rows with nothing to compare.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from conftest import make_stub
from cali_address import service
from cali_address.fidelity import FidelityReport
from cali_address.service import OUTPUT_COLUMNS, normalize_strict, results_to_records

LON = -76.5


def lat_of(plate: int) -> float:
    return round(3.4 + plate * 1e-4, 6)


def doc(plate, street="CL 987 # 654", **extra):
    d = {"direccion": f"{street} - {plate}", "lat": lat_of(plate), "lon": LON, "predial": f"P{plate}",
         "manzana": "MZ7"}
    d.update(extra)
    return d


FACE = [doc(12), doc(16), doc(20), doc(11), doc(15)]


def run(query, docs=FACE, **kw):
    return normalize_strict(make_stub(docs), [query], gazetteer=None, **kw).iloc[0]


def force(monkeypatch, risk, flags=("plate_changed",), seen=None):
    def fake(raw, output, cleaned=None):
        if seen is not None:
            seen.append((raw, output))
        return FidelityReport(risk, tuple(flags), 0.3, {})
    monkeypatch.setattr(service, "assess_fidelity", fake)


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------
def test_fidelity_columns_are_appended_last():
    assert OUTPUT_COLUMNS[-4:-2] == ["fidelity_risk", "fidelity_flags"]  # the catalogue columns follow
    assert OUTPUT_COLUMNS[-5] == "direccion_catastral"
    frame = normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12", "", "hola"], gazetteer=None)
    assert list(frame.columns) == OUTPUT_COLUMNS


def test_faithful_ok_row_is_none_and_keeps_predio():
    row = run("Calle 987 No. 654-12")
    assert row["estado"] == "OK" and row["position_tier"] == "predio"
    assert row["fidelity_risk"] == "none" and row["fidelity_flags"] == ""
    assert row["numero_predial_nacional"] == "P12" and "fidelidad" not in row["motivo"]


@pytest.mark.parametrize("raw", ["", "SIN ESPECIFICAR", "hola", None])
def test_rows_without_output_are_none_no_address(raw):
    row = run(raw)
    assert row["fidelity_risk"] == "none" and row["fidelity_flags"] == "no_address"
    assert row["position_tier"] == "none"


def test_records_carry_the_fidelity_columns():
    frame = normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12"], gazetteer=None)
    rec = results_to_records(frame)[0]
    assert rec["fidelity_risk"] == "none" and rec["fidelity_flags"] == ""


def test_empty_result_defaults_for_pipeline_skip_rows():
    row = service._empty_result("x", "FUERA_DE_AREA", "otro municipio")
    assert row["fidelity_risk"] == "none" and row["fidelity_flags"] == "no_address"


# ---------------------------------------------------------------------------
# the comparison is raw (original) vs final direccion_normalizada
# ---------------------------------------------------------------------------
def test_compares_original_raw_with_final_output(monkeypatch):
    seen: list = []
    force(monkeypatch, "none", flags=(), seen=seen)
    row = run("calle 987 # 654 - 12 TO 3")
    assert seen == [("calle 987 # 654 - 12 TO 3", row["direccion_normalizada"])]


def test_refine_input_still_compares_the_original_raw(monkeypatch):
    seen: list = []
    force(monkeypatch, "none", flags=(), seen=seen)
    stub = make_stub(FACE)
    frame = normalize_strict(stub, ["Cl 987 No 654-12"], gazetteer=None, refine_input=True)
    assert seen[0][0] == "Cl 987 No 654-12" and seen[0][1] == frame.iloc[0]["direccion_normalizada"]


# ---------------------------------------------------------------------------
# policy: high -> via, medium -> manzana, never raised
# ---------------------------------------------------------------------------
def test_high_caps_at_via_clears_predial_and_withholds_coordinates(monkeypatch):
    force(monkeypatch, "high", flags=("plate_changed", "letters_dropped"))
    row = run("CL 987 # 654 - 12")
    assert row["estado"] == "OK" and row["fidelity_risk"] == "high"
    assert row["fidelity_flags"] == "plate_changed;letters_dropped"
    assert (row["position_tier"], row["position_radius_m"], row["position_source"]) == ("via", 21, "none")
    assert row["numero_predial_nacional"] is None and pd.isna(row["lat"]) and pd.isna(row["lon"])
    assert pd.isna(row["manzana"]) and row["nivel_precision"] == "via"
    assert "fidelidad: plate_changed;letters_dropped" in row["motivo"]


def test_medium_caps_at_manzana_and_clears_predial_but_keeps_the_point(monkeypatch):
    force(monkeypatch, "medium", flags=("unit_digits_dropped",))
    row = run("CL 987 # 654 - 12")
    assert row["position_tier"] == "manzana" and row["position_radius_m"] == 56
    assert row["numero_predial_nacional"] is None and row["manzana"] == "MZ7"
    assert (row["lat"], row["lon"]) == (lat_of(12), LON)
    assert row["nivel_precision"] == "manzana" and "fidelidad: unit_digits_dropped" in row["motivo"]


@pytest.mark.parametrize("risk", ["none", "low"])
def test_none_and_low_never_change_anything(monkeypatch, risk):
    force(monkeypatch, risk, flags=() if risk == "none" else ("zero_padding",))
    row = run("CL 987 # 654 - 12")
    assert row["position_tier"] == "predio" and row["numero_predial_nacional"] == "P12"
    assert "fidelidad" not in row["motivo"]


def test_caps_never_raise_a_tier(monkeypatch):
    force(monkeypatch, "medium", flags=("unit_digits_dropped",))
    row = run("CL 987 # 654 - 14")            # interpolated tramo -> manzana, never back to predio
    assert row["position_tier"] == "manzana"
    force(monkeypatch, "high")
    row = run("CL 987 # 654 - 14")
    assert row["position_tier"] == "via" and pd.isna(row["lat"])
    force(monkeypatch, "medium", flags=("unit_digits_dropped",))
    row = run("CL 987 # 654 - 40", [doc(12)])  # a tier already worse than the cap stays as it is
    assert row["position_tier"] in ("via", "manzana", "none")


def test_high_on_a_row_without_position_stays_none(monkeypatch):
    force(monkeypatch, "high")
    row = run("hola")
    assert row["position_tier"] == "none" and row["estado"] != "OK"


def test_switch_disables_the_caps_but_keeps_the_columns(monkeypatch):
    force(monkeypatch, "high")
    monkeypatch.setattr(service, "FIDELITY_CAPS_ENABLED", False)
    row = run("CL 987 # 654 - 12")
    assert row["position_tier"] == "predio" and row["numero_predial_nacional"] == "P12"
    assert row["fidelity_risk"] == "high" and row["fidelity_flags"] == "plate_changed"


def test_cap_constants_are_the_documented_ones():
    assert service.FIDELITY_TIER_CAP == {"high": "via", "medium": "manzana"}
    assert service.FIDELITY_CAPS_ENABLED is True


def test_high_cap_with_extrapolated_coordinates_enabled_keeps_the_point(monkeypatch):
    force(monkeypatch, "high")
    monkeypatch.setattr(service, "EXTRAPOLATED_COORDS_ENABLED", True)
    row = run("CL 987 # 654 - 12")
    assert row["position_tier"] == "via" and row["numero_predial_nacional"] is None
    assert not math.isnan(row["lat"])


def test_mixed_batch_caps_only_the_flagged_rows(monkeypatch):
    def fake(raw, output, cleaned=None):
        bad = isinstance(raw, str) and raw.endswith("AP 5")
        return FidelityReport("high" if bad else "none", ("plate_changed",) if bad else (), 0.3, {})
    monkeypatch.setattr(service, "assess_fidelity", fake)
    frame = normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12", "CL 987 # 654 - 12 AP 5"], gazetteer=None)
    assert list(frame["position_tier"]) == ["predio", "via"]
    assert list(frame["fidelity_risk"]) == ["none", "high"]


def test_switch_off_leaves_motivo_untouched(monkeypatch):
    force(monkeypatch, "high")
    monkeypatch.setattr(service, "FIDELITY_CAPS_ENABLED", False)
    assert "fidelidad" not in run("CL 987 # 654 - 12")["motivo"]


# ---------------------------------------------------------------------------
# end to end with the real measure (no monkeypatching)
# ---------------------------------------------------------------------------
def test_missing_plate_output_invents_a_plate_and_is_capped_at_via():
    row = run("CL 987 # 654")
    assert row["fidelity_risk"] == "high" and "door_digits_added" in row["fidelity_flags"]
    assert row["position_tier"] == "via" and row["numero_predial_nacional"] is None and pd.isna(row["lat"])


def test_neighbour_snap_by_plate_tolerance_is_high_and_capped():
    # a +-1 plate tolerance accepts the neighbouring parcel and its address replaces the input's
    row = run("CL 987 # 654 - 13", [doc(12)], plate_tolerance=1)
    assert row["direccion_normalizada"] != "CL 987 # 654 - 13"      # the cadastral base was published
    assert row["fidelity_risk"] == "high" and "plate_changed" in row["fidelity_flags"]
    assert row["position_tier"] == "via" and row["numero_predial_nacional"] is None


def test_dropped_unit_digits_cap_at_manzana():
    row = service._apply_fidelity(
        {"estado": "OK", "motivo": "", "direccion_normalizada": "CL 987 # 654 - 12", "position_tier": "predio",
         "position_radius_m": 15.0, "numero_predial_nacional": "P12", "manzana": "MZ7", "lat": 3.4, "lon": LON,
         "position_source": "parcel", "nivel_precision": "predio"},
        "CL 987 # 654 - 12 APTO 301", None,
    )
    assert row["fidelity_risk"] == "medium" and row["position_tier"] == "manzana"
    assert row["numero_predial_nacional"] is None and row["nivel_precision"] == "manzana"


def _ok_row(tier, **extra):
    row = {"estado": "OK", "motivo": "", "direccion_normalizada": "CL 987 # 654 - 12", "position_tier": tier,
           "position_radius_m": float(service.POSITION_RADIUS_M[tier]), "numero_predial_nacional": "P12",
           "manzana": "MZ7", "lat": 3.4, "lon": LON, "position_source": "parcel", "nivel_precision": "predio"}
    row.update(extra)
    return row


def test_a_tier_already_worse_than_the_cap_is_never_raised(monkeypatch):
    force(monkeypatch, "medium", flags=("unit_digits_dropped",))
    row = service._apply_fidelity(_ok_row("via", nivel_precision="via", position_source="none", lat=None, lon=None),
                                  "x", None)
    assert row["position_tier"] == "via" and row["position_radius_m"] == 21.0 and row["nivel_precision"] == "via"
    assert row["numero_predial_nacional"] is None
    force(monkeypatch, "high")
    for tier in ("lote", "tramo", "manzana"):
        assert service._apply_fidelity(_ok_row(tier), "x", None)["position_tier"] == "via"
    assert service._apply_fidelity(_ok_row("manzana"), "x", None)["position_radius_m"] == 21.0
