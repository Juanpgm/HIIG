"""Round 8: an ``OK`` row with fidelity risk ``high`` no longer emits its text. Synthetic addresses only.

The pipeline treats an empty ``direccion_normalizada`` as "not normalized" and falls back to the raw text, so an OK
row whose cadastral/rule text is corrupted (``high``) is blanked; ``medium`` / ``low`` / ``none`` OK rows keep it.
``estado``, ``catalog_status``, ``position_tier`` (capped at ``via``) and ``fidelity_*`` keep their meaning.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from conftest import make_stub
from test_emission_rule import FACE, LOW, REJECTED_DOCS, empty, force_fidelity, one

from cali_address.service import OUTPUT_COLUMNS, normalize_strict, results_to_records

NOTE = "fidelidad alta: direccion no emitida"


def ok_row(monkeypatch, risk):
    force_fidelity(monkeypatch, risk)
    return one("CL 987 # 654 - 12")


def test_ok_high_is_blanked_with_note_and_everything_else_kept(monkeypatch):
    row = ok_row(monkeypatch, "high")
    assert row["estado"] == "OK" and row["catalog_status"] == "rejected_mismatch"
    assert empty(row) and row["fuente_normalizacion"] is None
    assert NOTE in row["motivo"] and "fidelidad: plate_changed" in row["motivo"]
    assert row["position_tier"] == "via" and row["fidelity_risk"] == "high"
    assert row["fidelity_flags"] == "plate_changed"
    assert row["selected_doc_id"] is not None and not pd.isna(row["selected_doc_id"])


@pytest.mark.parametrize("risk", ["none", "low", "medium"])
def test_ok_below_high_keeps_text_and_has_no_note(monkeypatch, risk):
    row = ok_row(monkeypatch, risk)
    assert row["estado"] == "OK" and not empty(row)
    assert row["fuente_normalizacion"] in ("catastro", "reglas")
    assert NOTE not in str(row["motivo"])


def test_non_ok_high_is_still_blank(monkeypatch):
    force_fidelity(monkeypatch, "high")
    row = one("CL 987 # 654 - 12", REJECTED_DOCS, [[0.9, 0.1]])
    assert row["estado"] == "SIN_MATCH" and empty(row) and row["fuente_normalizacion"] is None


def test_sin_match_without_risk_is_unchanged():
    row = one("CL 987 # 654 - 12", scores=LOW)
    assert row["estado"] == "SIN_MATCH" and row["fidelity_risk"] == "none"
    assert row["direccion_normalizada"] == "CL 987 # 654 - 12" and NOTE not in str(row["motivo"])


def test_ok_none_real_row_is_verified_and_emitted():
    row = one("CL 987 # 654 - 12")
    assert row["catalog_status"] == "verified" and row["direccion_normalizada"] == "CL 987 # 654 - 12"
    assert NOTE not in str(row["motivo"])


def test_note_is_appended_once_and_motivo_prefix_kept(monkeypatch):
    row = ok_row(monkeypatch, "high")
    assert row["motivo"].count(NOTE) == 1


def test_mixed_batch_is_deterministic_and_only_the_high_row_is_blank(monkeypatch):
    force_fidelity(monkeypatch, "high")
    raws = ["CL 987 # 654 - 12", "hola", "CL 987 # 654 - 77"]
    frames = [normalize_strict(make_stub(FACE), raws, gazetteer=None, ambiguity_delta=0.0) for _ in range(2)]
    assert frames[0].equals(frames[1])
    assert list(frames[0]["direccion_entrada"]) == raws
    assert [bool(empty(frames[0].iloc[i])) for i in range(3)] == [True, True, True]  # all forced high here


def test_mixed_fidelity_batch(monkeypatch):
    from cali_address import service
    from cali_address.fidelity import FidelityReport

    def fake(raw, output, cleaned=None):
        return FidelityReport("high" if raw == "CL 987 # 654 - 12" else "none", (), 0.3, {})

    monkeypatch.setattr(service, "assess_fidelity", fake)
    out = normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12", "CL 987 # 654 - 16", ""], gazetteer=None, ambiguity_delta=0.0)
    assert empty(out.iloc[0]) and not empty(out.iloc[1]) and empty(out.iloc[2])
    assert out["estado"].iloc[0] == "OK"


def test_schema_unchanged_and_null_round_trips(monkeypatch):
    force_fidelity(monkeypatch, "high")
    frame = normalize_strict(make_stub(FACE), ["CL 987 # 654 - 12"], gazetteer=None, ambiguity_delta=0.0)
    assert list(frame.columns) == list(OUTPUT_COLUMNS)
    rec = results_to_records(frame)[0]
    assert rec["direccion_normalizada"] is None and rec["fuente_normalizacion"] is None
    back = pd.read_csv(io.StringIO(frame.to_csv(index=False)), dtype=str)
    assert pd.isna(back.loc[0, "direccion_normalizada"]) and back.loc[0, "estado"] == "OK"
    pytest.importorskip("pyarrow")
    buf = io.BytesIO()
    frame.to_parquet(buf)
    buf.seek(0)
    assert pd.isna(pd.read_parquet(buf).loc[0, "direccion_normalizada"])


def test_idempotent_rerun_is_blank_again(monkeypatch):
    force_fidelity(monkeypatch, "high")
    a, b = ok_row(monkeypatch, "high"), ok_row(monkeypatch, "high")
    assert empty(a) and empty(b) and a["motivo"] == b["motivo"]


def test_legacy_switch_off_keeps_ok_high_text(monkeypatch):
    from cali_address import service

    monkeypatch.setattr(service, "FIDELITY_CAPS_ENABLED", False)
    row = ok_row(monkeypatch, "high")
    assert row["estado"] == "OK" and not empty(row) and NOTE not in str(row["motivo"])
