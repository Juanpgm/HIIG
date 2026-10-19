"""Emission rule of ``direccion_normalizada`` by ``catalog_status`` (round 6). Synthetic addresses only.

The pipeline treats an empty normalized address as "not normalized" and falls back to the raw text
(``fuente_direccion == 'origen'``), so a row whose rule text cannot be trusted must not carry it:

* ``unparsed``: never emits (the parsed base is not the address);
* ``below_cutoff`` / ``rejected_mismatch``: emits only when fidelity risk is none/low AND the parse is complete
  (via type + via number + cross number + plate, not truncated);
* every other status keeps its text. ``OK`` rows (a real cadastral match, ``fuente_direccion == 'modelo'``) are
  never blanked: an empty ``modelo`` row would be worse than a flagged one (their tier is already capped).

``estado``, ``motivo`` (prefix), ``catalog_status`` and ``position_tier`` keep their meaning.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import make_stub

from cali_address import catalog, service
from cali_address.fidelity import FidelityReport
from cali_address.parser import parse_address
from cali_address.service import OUTPUT_COLUMNS, normalize_strict, results_to_records

FACE = [{"direccion": f"CL 987 # 654 - {p}", "lat": 3.4 + p * 2.5e-5, "lon": -76.5, "predial": f"P{p}", "manzana": "MZ7"}
        for p in (12, 16, 20, 11, 15)]
LOW = [[0.1, 0, 0, 0, 0]]


def one(query, docs=FACE, scores=None, **kw):
    kw.setdefault("ambiguity_delta", 0.0)
    return normalize_strict(make_stub(docs, scores=scores), [query], gazetteer=None, **kw).iloc[0]


def force_fidelity(monkeypatch, risk):
    monkeypatch.setattr(service, "assess_fidelity",
                        lambda raw, output, cleaned=None: FidelityReport(risk, ("plate_changed",), 0.3, {}))


def empty(row):
    return row["direccion_normalizada"] is None or pd.isna(row["direccion_normalizada"])


def keeps(status, text, risk="none"):
    return catalog.keeps_rule_text(status, parse_address(text) if isinstance(text, str) else text, risk)


# ---------------------------------------------------------------------------
# pure rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("risk", [None, "none", "low", "medium", "high"])
@pytest.mark.parametrize("text", ["CL 987 # 654 - 12", "CL 987", None])
def test_unparsed_never_emits(risk, text):
    assert keeps("unparsed", text, risk) is False


@pytest.mark.parametrize("status", ["below_cutoff", "rejected_mismatch"])
@pytest.mark.parametrize("risk", [None, "none", "low"])
def test_complete_parse_with_low_risk_keeps_the_text(status, risk):
    assert keeps(status, "CL 987 # 654 - 12", risk) is True
    assert keeps(status, "KR 987 A BIS # 654 B - 12", risk) is True


@pytest.mark.parametrize("status", ["below_cutoff", "rejected_mismatch"])
@pytest.mark.parametrize("risk", ["medium", "high", "bogus"])
def test_medium_or_high_risk_empties_the_text(status, risk):
    assert keeps(status, "CL 987 # 654 - 12", risk) is False


@pytest.mark.parametrize("status", ["below_cutoff", "rejected_mismatch"])
@pytest.mark.parametrize("text", ["CL 987", "CL 987 # 654", "# 654 - 12", "AV 905 OESTE", "", None, 42, object()])
def test_incomplete_parse_empties_the_text(status, text):
    assert keeps(status, text, "none") is False


def test_truncated_parse_is_not_complete():
    parsed = parse_address("CL 987 # 654, 12")
    assert not catalog.complete_parse(parsed)


@pytest.mark.parametrize("status", ["verified", "ambiguous", "door_not_catalogued"])
@pytest.mark.parametrize("risk", [None, "none", "medium"])
def test_other_statuses_keep_below_high_risk(status, risk):
    assert keeps(status, "CL 987", risk) is True


@pytest.mark.parametrize("status", ["ambiguous", "door_not_catalogued", "something_new"])
def test_other_statuses_never_keep_a_high_risk_text(status):
    # round 7: a high fidelity risk is corrupted text whatever the status (only ``verified`` OK rows are exempt)
    assert keeps(status, "CL 987", "high") is False


def test_unknown_status_keeps_the_text():
    assert keeps("something_new", "CL 987", "medium") is True


def test_complete_parse_is_total_and_deterministic():
    assert catalog.complete_parse(None) is False
    assert catalog.complete_parse(object()) is False
    p = parse_address("CL 987 # 654 - 12")
    assert catalog.complete_parse(p) is catalog.complete_parse(p) is True


# ---------------------------------------------------------------------------
# normalize_strict
# ---------------------------------------------------------------------------
def test_below_cutoff_complete_low_risk_keeps_rule_text():
    row = one("CL 987 # 654 - 12", scores=LOW)
    assert row["catalog_status"] == "below_cutoff"
    assert row["direccion_normalizada"] == "CL 987 # 654 - 12" and row["fuente_normalizacion"] == "reglas"


def test_below_cutoff_with_a_dropped_trailing_group_is_emptied():
    row = one("CL 987 # 654 - 12 - 906", scores=LOW)         # the trailing '- 906' is lost: fidelity medium
    assert row["catalog_status"] == "below_cutoff" and row["fidelity_risk"] == "medium"
    assert empty(row) and row["fuente_normalizacion"] is None


def test_emptied_row_keeps_estado_status_tier_and_motivo_prefix():
    row = one("CL 987 # 654 - 12 - 906", scores=LOW)
    assert row["estado"] == "SIN_MATCH" and row["position_tier"] == "none"
    assert row["motivo"].startswith("confianza") and "umbral" in row["motivo"]
    assert bool(row["address_verified"]) is False
    assert row["fidelity_risk"] == "medium" and "trailing_number_dropped" in row["fidelity_flags"]


def test_truncated_unparsed_sin_match_is_emptied():
    docs = [{"direccion": "CL 987 # 654 -", "predial": "P1"}]
    row = one("CL 987 # 654, 12", docs, [[0.1]])
    assert row["estado"] == "SIN_MATCH" and row["catalog_status"] == "unparsed"
    assert empty(row) and row["fuente_normalizacion"] is None


def test_truncated_unparsed_ok_row_is_blanked_when_fidelity_is_high():
    # round 8: this row is fidelity high (door_digits_dropped), so the OK exemption no longer applies
    docs = [{"direccion": "CL 987 # 654 -", "predial": "P1"}]
    row = one("CL 987 # 654, 12", docs)
    assert row["estado"] == "OK" and row["catalog_status"] == "unparsed" and row["position_tier"] == "via"
    assert row["fidelity_risk"] == "high" and empty(row) and row["fuente_normalizacion"] is None


@pytest.mark.parametrize("raw", ["", "   ", "hola", "Sector Norte hola", None, "12345", "Ñúéç中文", ";", "#;"])
def test_no_parseable_rows_have_no_text_and_no_source(raw):
    row = one(raw)
    assert row["catalog_status"] == "unparsed" and empty(row) and row["fuente_normalizacion"] is None


REJECTED_DOCS = [dict(FACE[0], direccion="CL 987 # 654 - 40"), FACE[0]]    # the top-ranked parcel is another door


@pytest.mark.parametrize("risk,kept", [("none", True), ("low", True), ("medium", False), ("high", False)])
def test_rejected_mismatch_follows_fidelity_risk(monkeypatch, risk, kept):
    force_fidelity(monkeypatch, risk)
    row = one("CL 987 # 654 - 12", REJECTED_DOCS, [[0.9, 0.1]])
    assert row["estado"] == "SIN_MATCH" and row["catalog_status"] == "rejected_mismatch"
    assert (not empty(row)) is kept
    assert row["fuente_normalizacion"] == ("reglas" if kept else None)


@pytest.mark.parametrize("risk", ["medium"])  # round 8: 'high' OK rows are blanked (test_round8_ok_high_blank.py)
def test_ok_rows_below_high_are_never_blanked(monkeypatch, risk):
    force_fidelity(monkeypatch, risk)
    row = one("CL 987 # 654 - 12")
    assert row["estado"] == "OK" and row["catalog_status"] == "rejected_mismatch"
    assert not empty(row) and row["fuente_normalizacion"] in ("catastro", "reglas")


@pytest.mark.parametrize("risk", ["none", "low", "medium", "high"])
def test_incomplete_parse_below_cutoff_is_emptied_whatever_the_risk(monkeypatch, risk):
    force_fidelity(monkeypatch, risk)
    monkeypatch.setattr(service, "_catalog_decide", lambda *a, **k: "below_cutoff")
    row = one("CL 987", scores=LOW)
    assert row["direccion_normalizada"] is None or pd.isna(row["direccion_normalizada"])


def test_verified_row_still_emits():
    row = one("CL 987 # 654 - 12")
    assert row["catalog_status"] == "verified" and row["direccion_normalizada"] == "CL 987 # 654 - 12"


def test_door_not_catalogued_keeps_its_rule_text():
    row = one("CL 987 # 654 - 77", scores=LOW)
    assert row["catalog_status"] == "door_not_catalogued" and row["direccion_normalizada"] == "CL 987 # 654 - 77"


def test_mixed_batch_keeps_order_and_is_deterministic():
    raws = ["CL 987 # 654 - 12", "hola", "CL 987 # 654 - 12 - 906", "CL 987 # 654 - 77"]
    frames = [normalize_strict(make_stub(FACE), raws, gazetteer=None, ambiguity_delta=0.0) for _ in range(2)]
    assert frames[0].equals(frames[1])
    assert list(frames[0]["direccion_entrada"]) == raws
    assert empty(frames[0].iloc[1])


def test_schema_is_unchanged_and_records_serialize_null_text():
    row = normalize_strict(make_stub(FACE), ["hola"], gazetteer=None)
    assert list(row.columns) == list(OUTPUT_COLUMNS)
    rec = results_to_records(row)[0]
    assert rec["direccion_normalizada"] is None and rec["fuente_normalizacion"] is None


def test_idempotent_on_emitted_text():
    first = one("CL 987 # 654 - 12", scores=LOW)["direccion_normalizada"]
    second = one(first, scores=LOW)["direccion_normalizada"]
    assert first == second
