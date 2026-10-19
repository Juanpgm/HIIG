"""``catalog_status`` / ``address_verified``: what the pipeline can say about an address, per row.

Statuses, in the precedence the implementation applies (first rule that holds wins):

1. ``unparsed``            no address, a grammar that failed, or a TRUNCATED parse (the base is not the address)
2. ``verified``/``ambiguous`` the OK row whose base equals the cadastre's (exact guard, no caps), fidelity none/low;
                           ``ambiguous`` when several parcels share the address
3. ``door_not_catalogued`` the base parses fully and the exact base (via, cross, plate, per-position quadrants,
                           letters, BIS) is absent from the cadastre address set
4. ``below_cutoff``        a SIN_MATCH row below the confidence cut-off (or without candidates)
5. ``ambiguous``           a SIN_MATCH row abstained for ambiguity (base catalogued, parcels compete)
6. ``rejected_mismatch``   everything else that is not verified: guard / rule rejection, caps, fidelity risk

``address_verified`` is True exactly for ``verified``. Synthetic addresses only.
"""

from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import make_stub

from cali_address import catalog, service
from cali_address.catalog import (
    CACHE_FILENAME,
    SIDECAR_FILENAME,
    STATUS_VALUES,
    base_key,
    build_keys,
    door_catalogued,
    keys_for,
    load_or_build,
    source_hash,
)
from cali_address.fidelity import FidelityReport
from cali_address.parser import parse_address
from cali_address.service import OUTPUT_COLUMNS, normalize_strict, results_to_records


def key(raw):
    return base_key(parse_address(raw))


def doc(plate, street="CL 987 # 654", **extra):
    d = {"direccion": f"{street} - {plate}", "lat": 3.4 + plate * 2.5e-5, "lon": -76.5, "predial": f"P{plate}",
         "manzana": "MZ7"}
    d.update(extra)
    return d


FACE = [doc(12), doc(16), doc(20), doc(11), doc(15)]


def run(query, docs=FACE, scores=None, **kw):
    kw.setdefault("ambiguity_delta", 0.0)
    stub = make_stub(docs, scores=scores)
    return normalize_strict(stub, [query] if not isinstance(query, list) else query, gazetteer=None, **kw)


def one(query, docs=FACE, scores=None, **kw):
    return run(query, docs, scores, **kw).iloc[0]


def force_fidelity(monkeypatch, risk):
    monkeypatch.setattr(service, "assess_fidelity",
                        lambda raw, output, cleaned=None: FidelityReport(risk, ("plate_changed",), 0.3, {}))


# ---------------------------------------------------------------------------
# base key
# ---------------------------------------------------------------------------
def test_status_values_are_the_six_agreed_ones():
    assert STATUS_VALUES == ("verified", "ambiguous", "door_not_catalogued", "rejected_mismatch", "unparsed",
                             "below_cutoff")


@pytest.mark.parametrize(("a", "b"), [
    ("CL 987 # 654 - 12", "Calle 987 No 654-12"),
    ("CL 987 # 654 - 12", "cl987#654-12"),                       # glued tokens
    ("CL 987 # 654 - 012", "CL 987 # 654 - 12"),                 # leading zeros
    ("CL 987 # 654 - 12 AP 301", "CL 987 # 654 - 12 TO 2 AP 4"),  # units are not part of the base
    ("CL 987 N # 654 - 12", "CL 987 NORTE # 654 - 12"),          # a cardinal letter is the spelled quadrant
])
def test_equivalent_spellings_share_the_base_key(a, b):
    assert key(a) is not None and key(a) == key(b)


@pytest.mark.parametrize(("a", "b"), [
    ("CL 987 # 654 - 12", "CL 987 # 654 - 14"),                  # plate
    ("CL 987 # 654 - 12", "CL 987 # 655 - 12"),                  # cross
    ("CL 987 # 654 - 12", "KR 987 # 654 - 12"),                  # via type
    ("CL 987 A # 654 - 12", "CL 987 # 654 - 12"),                # via letters
    ("CL 987 # 654 A - 12", "CL 987 # 654 - 12"),                # cross letters
    ("CL 987 BIS # 654 - 12", "CL 987 # 654 - 12"),              # BIS
    ("CL 987 NORTE # 654 - 12", "CL 987 # 654 NORTE - 12"),      # the quadrant POSITION matters
    ("CL 987 # 654 - 12 NORTE", "CL 987 # 654 - 12"),            # trailing quadrant
    ("CL 987 # 654 - 12 C", "CL 987 # 654 - 12"),                # plate letters
])
def test_different_bases_have_different_keys(a, b):
    assert key(a) != key(b)


@pytest.mark.parametrize("raw", [None, "", "   ", "hola", "12345", ",", "Ñúé", "CL"])
def test_unparsed_input_has_no_key(raw):
    assert base_key(parse_address(raw)) is None


def test_base_key_tolerates_none_and_foreign_objects():
    assert base_key(None) is None and base_key(object()) is None


def test_base_key_is_deterministic_and_hashable():
    first, second = key("CL 987 # 654 - 12"), key("CL 987 # 654 - 12")
    assert first == second and isinstance(first, bytes) and hash(first) == hash(second)


def test_cadastral_reading_of_the_plate_tail_is_used_for_documents():
    # ``69 1 C`` in a cadastre string is a unit code, not plate letters: the guard reads it that way
    assert base_key(parse_address("CL 987 # 654 - 12 1 C"), cadastral=True) == key("CL 987 # 654 - 12")
    assert base_key(parse_address("CL 987 # 654 - 12 1 C")) is not None


# ---------------------------------------------------------------------------
# key set
# ---------------------------------------------------------------------------
def test_build_keys_skips_unparseable_and_non_strings():
    keys = build_keys(["CL 987 # 654 - 12", None, "", "hola", 42, float("nan"), "CL 987 # 654 - 12"])
    assert len(keys) == 1 and door_catalogued(parse_address("Calle 987 # 654-12"), keys)


def test_door_catalogued_edge_cases():
    keys = build_keys(["CL 987 # 654 - 12"])
    assert door_catalogued(parse_address("CL 987 # 654 - 12 AP 3"), keys)
    assert not door_catalogued(parse_address("CL 987 # 654 - 13"), keys)
    assert not door_catalogued(parse_address("hola"), keys) and not door_catalogued(None, keys)
    assert not door_catalogued(parse_address("CL 987 # 654 - 12"), frozenset())
    assert not door_catalogued(parse_address("CL 987 # 654 - 12"), None)


def test_build_keys_is_deterministic_and_order_independent():
    rows = ["CL 987 # 654 - 12", "KR 5 # 3 - 8", "CL 987 NORTE # 654 - 9"]
    assert build_keys(rows) == build_keys(rows[::-1]) == build_keys(rows)


# ---------------------------------------------------------------------------
# cache: sidecar discipline (same as the interpolation index)
# ---------------------------------------------------------------------------
ROWS = ["CL 987 # 654 - 12", "CL 987 # 654 - 16", "KR 5 # 3 - 8"]


def test_cache_roundtrip_and_files(tmp_path):
    built = load_or_build(ROWS, str(tmp_path))
    assert (tmp_path / CACHE_FILENAME).exists() and (tmp_path / SIDECAR_FILENAME).exists()
    meta = json.loads((tmp_path / SIDECAR_FILENAME).read_text())
    assert meta["source_hash"] == source_hash(ROWS) and meta["version"] == catalog.INDEX_VERSION
    assert load_or_build(ROWS, str(tmp_path)) == built


def test_cache_hit_does_not_rebuild(tmp_path, monkeypatch):
    load_or_build(ROWS, str(tmp_path))
    monkeypatch.setattr(catalog, "build_keys", lambda *_a, **_k: pytest.fail("rebuilt despite a valid cache"))
    assert len(load_or_build(ROWS, str(tmp_path))) == 3


def test_cache_rebuilt_when_the_source_changes(tmp_path):
    load_or_build(ROWS, str(tmp_path))
    changed = [*ROWS, "CL 987 # 654 - 18"]
    assert len(load_or_build(changed, str(tmp_path))) == 4
    assert json.loads((tmp_path / SIDECAR_FILENAME).read_text())["source_hash"] == source_hash(changed)


@pytest.mark.parametrize("damage", ["missing_sidecar", "garbage_sidecar", "tampered_pickle", "wrong_version",
                                    "malicious_pickle"])
def test_damaged_cache_is_ignored_and_rebuilt(tmp_path, damage):
    load_or_build(ROWS, str(tmp_path))
    pkl, side = tmp_path / CACHE_FILENAME, tmp_path / SIDECAR_FILENAME
    if damage == "missing_sidecar":
        side.unlink()
    elif damage == "garbage_sidecar":
        side.write_text("{not json")
    elif damage == "tampered_pickle":
        pkl.write_bytes(pkl.read_bytes()[:-5] + b"xxxxx")
    elif damage == "wrong_version":
        meta = json.loads(side.read_text()); meta["version"] = 99; side.write_text(json.dumps(meta))
    else:
        class Evil:
            def __reduce__(self):
                return (pytest.fail, ("unpickled an unverified cache",))
        pkl.write_bytes(pickle.dumps(Evil()))                      # sidecar digest no longer matches
    assert len(load_or_build(ROWS, str(tmp_path))) == 3


def test_unwritable_cache_dir_warns_and_keeps_the_keys(tmp_path):
    target = tmp_path / "file_not_dir"
    target.write_text("x")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = load_or_build(ROWS, str(target))
    assert len(got) == 3 and any("not written" in str(w.message) for w in caught)


def test_no_cache_dir_never_touches_the_disk():
    assert len(load_or_build(ROWS, None)) == 3


def test_keys_for_is_memoized_on_the_normalizer():
    stub = make_stub(FACE)
    first = keys_for(stub)
    assert keys_for(stub) is first and getattr(stub, "_catalog_keys") is first


# ---------------------------------------------------------------------------
# columns and plumbing
# ---------------------------------------------------------------------------
def test_columns_are_appended_last_after_the_fidelity_columns():
    assert OUTPUT_COLUMNS[-4:] == ["fidelity_risk", "fidelity_flags", "catalog_status", "address_verified"]
    frame = run(["CL 987 # 654 - 12", "", "hola", None])
    assert list(frame.columns) == OUTPUT_COLUMNS
    assert frame["address_verified"].dtype == bool
    assert set(frame["catalog_status"]) <= set(STATUS_VALUES) and frame["catalog_status"].notna().all()


def test_records_and_schema_carry_the_columns():
    from cali_address.api.schemas import ResultRow

    rec = results_to_records(run("CL 987 # 654 - 12"))[0]
    assert rec["catalog_status"] == "verified" and rec["address_verified"] is True
    assert ResultRow.model_fields["catalog_status"].default == "unparsed"
    assert ResultRow.model_fields["address_verified"].default is False
    ResultRow(**rec)


def test_empty_result_defaults_and_pipeline_skip_rows():
    row = service._empty_result("x", "FUERA_DE_AREA", "otro municipio")
    assert row["catalog_status"] == "unparsed" and bool(row["address_verified"]) is False
    row = service._empty_result("x", "ERROR", "boom")
    assert row["catalog_status"] == "unparsed" and bool(row["address_verified"]) is False


def test_confiabilidad_is_untouched_by_the_new_columns():
    assert "confiabilidad" in OUTPUT_COLUMNS and OUTPUT_COLUMNS.index("confiabilidad") < OUTPUT_COLUMNS.index("catalog_status")
    assert "confiabilidad_lr" not in OUTPUT_COLUMNS and not [c for c in OUTPUT_COLUMNS if "prob" in c]


# ---------------------------------------------------------------------------
# one test per status
# ---------------------------------------------------------------------------
def test_verified_exact_row():
    row = one("CL 987 # 654 - 12")
    assert row["estado"] == "OK" and row["position_tier"] == "predio"
    assert row["catalog_status"] == "verified" and bool(row["address_verified"]) is True


@pytest.mark.parametrize("raw", ["Calle 987 No. 654-12", "cl987#654-12", "  CL 987 # 654 - 12  ", "CL 987 # 654 - 12 AP 301",
                                 "CL 987, # 654 - 12", "CL 987 # 654 - 012"])
def test_verified_survives_spelling_noise(raw):
    row = one(raw)
    assert row["catalog_status"] == "verified", raw


def test_ambiguous_when_several_parcels_share_the_address():
    docs = [doc(12, n_predios=3), doc(16)]
    row = one("CL 987 # 654 - 12", docs)
    assert row["estado"] == "OK" and "compartida" in row["motivo"]
    assert row["catalog_status"] == "ambiguous" and bool(row["address_verified"]) is False


def test_ambiguous_for_a_sin_match_row_that_abstained_for_ambiguity():
    docs = [doc(12, manzana="MZ1", predial="PA"), doc(12, manzana="MZ2", predial="PB")]
    row = one("CL 987 # 654 - 12", docs, scores=[[0.95, 0.95]], ambiguity_delta=0.02)
    assert row["estado"] == "SIN_MATCH" and "ambiguo" in row["motivo"]
    assert row["catalog_status"] == "ambiguous" and bool(row["address_verified"]) is False


@pytest.mark.parametrize("query", ["CL 987 # 654 - 14", "CL 987 # 654 - 13", "CL 987 # 654 - 99", "CL 987 # 655 - 12",
                                   "CL 987 NORTE # 654 - 12", "CL 987 # 654 NORTE - 12", "CL 987 A # 654 - 12",
                                   "CL 987 # 654 - 12 C", "KR 55 # 66 - 77"])
def test_door_not_catalogued_when_the_exact_base_is_absent(query):
    row = one(query)
    assert row["catalog_status"] == "door_not_catalogued" and bool(row["address_verified"]) is False


def test_door_not_catalogued_wins_over_a_tier_capped_ok_row():
    row = one("CL 987 # 654 - 14")                 # tramo: OK, interpolated, never verified
    assert row["estado"] == "OK" and row["position_tier"] == "tramo"
    assert row["catalog_status"] == "door_not_catalogued"


def test_rejected_mismatch_when_the_base_exists_but_the_retrieved_candidate_differs():
    docs = [doc(40), doc(12)]                      # the top-ranked parcel is a different door
    row = one("CL 987 # 654 - 12", docs, scores=[[0.9, 0.1]])
    assert row["estado"] == "SIN_MATCH" and bool(row["override_rejected"]) is True
    assert row["catalog_status"] == "rejected_mismatch" and bool(row["address_verified"]) is False


def test_below_cutoff_when_the_base_exists_but_the_score_is_too_low():
    row = one("CL 987 # 654 - 12", scores=[[0.1, 0, 0, 0, 0]])
    assert row["estado"] == "SIN_MATCH" and "umbral" in row["motivo"]
    assert row["catalog_status"] == "below_cutoff" and bool(row["address_verified"]) is False


@pytest.mark.parametrize("raw", ["", "   ", "hola", "SIN ESPECIFICAR", None, "12345", "Ñúéç中文"])
def test_unparsed_statuses(raw):
    row = one(raw)
    assert row["catalog_status"] == "unparsed" and bool(row["address_verified"]) is False


def test_truncated_parse_is_unparsed_never_verified():
    docs = [{"direccion": "CL 987 # 654 -", "predial": "P1"}]
    row = one("CL 987 # 654, 12", docs)
    assert row["estado"] == "OK" and row["position_tier"] == "via"       # capped, not rejected, in production
    assert row["catalog_status"] == "unparsed" and bool(row["address_verified"]) is False


def test_fidelity_high_makes_an_otherwise_exact_row_rejected_mismatch(monkeypatch):
    force_fidelity(monkeypatch, "high")
    row = one("CL 987 # 654 - 12")
    assert row["position_tier"] == "via"
    assert row["catalog_status"] == "rejected_mismatch" and bool(row["address_verified"]) is False


@pytest.mark.parametrize(("risk", "status"), [("none", "verified"), ("low", "verified"), ("medium", "rejected_mismatch"),
                                              ("high", "rejected_mismatch")])
def test_fidelity_risk_gates_verification(monkeypatch, risk, status):
    force_fidelity(monkeypatch, risk)
    assert one("CL 987 # 654 - 12")["catalog_status"] == status


def test_site_unit_conflict_is_not_verified():
    docs = [doc(12, direccion="CL 987 # 654 - 12 TO 2")]
    row = one("CL 987 # 654 - 12 TO 3", docs)
    assert row["position_tier"] == "lote" and row["catalog_status"] == "rejected_mismatch"


def test_quadrant_moved_to_another_position_is_door_not_catalogued():
    docs = [doc(12, direccion="CL 987 # 654 NORTE - 12")]
    row = one("CL 987 NORTE # 654 - 12", docs)
    assert row["estado"] != "OK" and row["catalog_status"] == "door_not_catalogued"


def test_quadrant_in_the_same_position_is_verified():
    docs = [doc(12, direccion="CL 987 NORTE # 654 - 12")]
    assert one("CL 987 N # 654 - 12", docs)["catalog_status"] == "verified"


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------
def test_precedence_unparsed_beats_door_not_catalogued():
    docs = [doc(12)]
    row = one("CL 55 # 66, 77", docs)               # truncated AND the base is absent
    assert row["catalog_status"] == "unparsed"


def test_precedence_door_not_catalogued_beats_below_cutoff():
    row = one("CL 987 # 654 - 99", scores=[[0.1, 0, 0, 0, 0]])
    assert row["estado"] == "SIN_MATCH" and row["catalog_status"] == "door_not_catalogued"


def test_precedence_door_not_catalogued_beats_a_rule_rejection():
    row = one("CL 987 # 654 - 99")
    assert row["estado"] == "SIN_MATCH" and bool(row["override_rejected"]) and row["catalog_status"] == "door_not_catalogued"


def test_precedence_door_not_catalogued_beats_high_fidelity(monkeypatch):
    force_fidelity(monkeypatch, "high")
    assert one("CL 987 # 654 - 14")["catalog_status"] == "door_not_catalogued"


def test_precedence_fidelity_beats_shared_address(monkeypatch):
    force_fidelity(monkeypatch, "high")
    docs = [doc(12, n_predios=3)]
    assert one("CL 987 # 654 - 12", docs)["catalog_status"] == "rejected_mismatch"


def test_precedence_below_cutoff_beats_rejected_mismatch():
    docs = [doc(40), doc(12)]
    row = one("CL 987 # 654 - 12", docs, scores=[[0.1, 0.05]])
    assert row["catalog_status"] == "below_cutoff"


def test_unavailable_keys_degrade_to_the_next_rule(monkeypatch):
    def boom(_normalizer):
        raise RuntimeError("no keys")
    monkeypatch.setattr(service, "_catalog_keys_for", boom)
    assert one("CL 987 # 654 - 12")["catalog_status"] == "verified"           # no key needed to verify
    assert one("CL 987 # 654 - 99")["catalog_status"] == "rejected_mismatch"  # cannot say the door is absent


# ---------------------------------------------------------------------------
# robustness: batches, determinism, idempotence, long input
# ---------------------------------------------------------------------------
BATCH = ["CL 987 # 654 - 12", "CL 987 # 654 - 14", "CL 987 # 654 - 99", "hola", "", None, "CL 987 # 654, 12",
         "CL 987 # 654 - 12 AP 301", "KR 55 # 66 - 77"]


def test_status_of_a_row_does_not_depend_on_the_batch():
    together = run(BATCH)["catalog_status"].tolist()
    alone = [one(q)["catalog_status"] for q in BATCH]
    assert together == alone
    assert run(BATCH[::-1])["catalog_status"].tolist() == together[::-1]


def test_status_is_deterministic_and_idempotent():
    a, b = run(BATCH), run(BATCH)
    assert a["catalog_status"].tolist() == b["catalog_status"].tolist()
    assert a["address_verified"].tolist() == [s == "verified" for s in a["catalog_status"]]
    # feeding the normalized output back must give the same verdict for verified rows
    again = run([x for x in a.loc[a["catalog_status"] == "verified", "direccion_normalizada"]])
    assert set(again["catalog_status"]) == {"verified"}


@pytest.mark.parametrize("raw", ["CL 987 # 654 - 12" + " X" * 3000, "9" * 10000, "CL 987 # 654 - 12," * 500, "#" * 5000,
                                 "CL 987 # 654 - 12 ‮\u0000​"])
def test_long_and_hostile_strings_never_raise(raw):
    row = one(raw)
    assert row["catalog_status"] in STATUS_VALUES and isinstance(row["address_verified"], (bool, np.bool_))


def test_empty_cadastre_keys_mean_every_parsed_door_is_absent():
    row = one("CL 987 # 654 - 12", [{"direccion": ""}])
    assert row["catalog_status"] in ("door_not_catalogued", "rejected_mismatch", "below_cutoff")


def test_address_verified_is_never_null_or_object():
    frame = run(BATCH)
    assert frame["address_verified"].isna().sum() == 0 and frame["address_verified"].dtype == bool
    assert frame["catalog_status"].isna().sum() == 0 and pd.api.types.is_string_dtype(frame["catalog_status"])
