"""Serving-time switch of the cadastre fallback: ``exclude_fuentes`` (decision.json) / ``NORMALIZER_EXCLUDE_FUENTES``.

With the switch ON the normalizer serves exactly the index WITHOUT the excluded tail docs: the output is identical to
the output of an index that never had them, not merely "similar" (invented fixtures, numbers 900+).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import make_stub
from test_decision_model import decision_doc
from test_decision_service import NPN_A, NPN_B, calibrated, const_model, digest

from cali_address import decision as D
from cali_address import service
from cali_address.service import normalize_strict

LOT_C = "760010100000000900003"
NPN_C = LOT_C + "000000000"

CERT, CAT = "nomenclatura_cert", "catastro"
NOM_DOCS = [
    {"direccion": "CL 987 # 654 - 12", "predial": NPN_A, "manzana": "M1", "lon": -76.50, "lat": 3.40},
    {"direccion": "CL 987 # 654 - 14", "predial": NPN_B, "manzana": "M1", "lon": -76.51, "lat": 3.41},
]
FALLBACK_DOC = {"direccion": "CL 987 # 654 - 16", "predial": NPN_C, "manzana": "M2", "lon": -76.52, "lat": 3.42}
QUERIES = ["CL 987 # 654 - 16", "CL 987 # 654 - 12", "CL 987 # 654 - 14", "texto sin direccion"]
#: similarity of each query to [d0, d1, d2]
SCORES = [[0.30, 0.20, 0.95], [0.95, 0.20, 0.40], [0.20, 0.95, 0.40], [0.1, 0.1, 0.1]]


def stub_with_fallback(**kw):
    stub = make_stub(NOM_DOCS + [FALLBACK_DOC], scores=SCORES, **kw)
    stub._fuente = np.array([CERT, CERT, CAT], dtype=object)
    return stub


def stub_without_fallback(**kw):
    return make_stub(NOM_DOCS, scores=[row[:2] for row in SCORES], **kw)


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv(D.ENV_EXCLUDE_FUENTES, raising=False)


# ----------------------------------------------------------------------------------------------- decision helpers

def test_validate_decision_defaults_to_no_exclusion_and_adds_no_key():
    assert "exclude_fuentes" not in D.validate_decision(decision_doc())            # byte-compatible with old files
    assert D.validate_decision(decision_doc(exclude_fuentes=[]))["exclude_fuentes"] == []
    assert D.validate_decision(decision_doc(exclude_fuentes=["catastro"]))["exclude_fuentes"] == ["catastro"]


@pytest.mark.parametrize("bad", ["catastro", {"a": 1}, [1], [""], ["  "], [None], ["a", ["b"]], ["CL 987 # 654 - 12"], 5])
def test_validate_decision_rejects_a_malformed_exclusion_list(bad):
    with pytest.raises(ValueError):
        D.validate_decision(decision_doc(exclude_fuentes=bad))


def test_excluded_fuentes_merges_decision_and_environment():
    assert D.excluded_fuentes(None, {}) == ()
    assert D.excluded_fuentes({}, {}) == ()
    assert D.excluded_fuentes({"exclude_fuentes": ["b", "a"]}, {}) == ("a", "b")
    assert D.excluded_fuentes(None, {D.ENV_EXCLUDE_FUENTES: " catastro , , other "}) == ("catastro", "other")
    assert D.excluded_fuentes({"exclude_fuentes": ["catastro"]}, {D.ENV_EXCLUDE_FUENTES: "catastro,x"}) == ("catastro", "x")
    assert D.excluded_fuentes(None, {D.ENV_EXCLUDE_FUENTES: ""}) == ()
    assert D.excluded_fuentes(None, {D.ENV_EXCLUDE_FUENTES: " , ,"}) == ()


def test_excluded_fuentes_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv(D.ENV_EXCLUDE_FUENTES, "catastro")
    assert D.excluded_fuentes(None) == ("catastro",)


# ------------------------------------------------------------------------------------------------- service switch

def test_switch_off_changes_nothing():
    a = normalize_strict(stub_with_fallback(), QUERIES, gazetteer=None)
    plain = make_stub(NOM_DOCS + [FALLBACK_DOC], scores=SCORES)                # a stub without `_fuente` at all
    b = normalize_strict(plain, QUERIES, gazetteer=None)
    assert list(a.columns) == list(b.columns) + ["fuente"]                       # only the source column differs
    assert digest(a.drop(columns=["fuente"])) == digest(b)
    assert a.loc[0, "selected_doc_id"] == 2 and a.loc[0, "fuente"] == CAT        # the fallback doc is reachable when ON


def test_environment_switch_equals_the_index_without_the_fallback_docs(monkeypatch):
    monkeypatch.setenv(D.ENV_EXCLUDE_FUENTES, CAT)
    with_switch = normalize_strict(stub_with_fallback(), QUERIES, gazetteer=None)
    without = normalize_strict(stub_without_fallback(), QUERIES, gazetteer=None)
    assert digest(with_switch.drop(columns=["fuente"], errors="ignore")) == digest(without.drop(columns=["fuente"], errors="ignore"))
    assert 2 not in with_switch["selected_doc_id"].dropna().tolist()


def test_decision_json_switch_equals_the_index_without_the_fallback_docs_in_calibrated_mode():
    on = stub_with_fallback()
    on.calibrated = calibrated(exclude_fuentes=[CAT])
    off = stub_without_fallback()
    off.calibrated = calibrated()
    a = normalize_strict(on, QUERIES, gazetteer=None)
    b = normalize_strict(off, QUERIES, gazetteer=None)
    assert digest(a.drop(columns=["fuente"], errors="ignore")) == digest(b.drop(columns=["fuente"], errors="ignore"))
    keep_all = stub_with_fallback()
    keep_all.calibrated = calibrated()                                          # key absent: the fallback doc is served
    c = normalize_strict(keep_all, QUERIES, gazetteer=None)
    assert c.loc[0, "selected_doc_id"] == 2 and 2 not in a["selected_doc_id"].dropna().tolist()


def test_the_switch_slices_every_per_document_array_and_keeps_doc_ids():
    stub = stub_with_fallback()
    dropped = service.apply_exclusions(stub, [CAT])
    assert dropped == 1
    for name in ("_direccion", "_predial", "_manzana", "_comuna", "_lon", "_lat", "_n_predios", "_manzana_nunique",
                 "_own_lon", "_own_lat", "_fuente"):
        assert len(getattr(stub, name)) == 2, name
    assert list(stub._direccion) == [d["direccion"] for d in NOM_DOCS]            # a prefix: ids 0..N-1 are unchanged


def test_applying_the_switch_twice_is_idempotent(monkeypatch):
    monkeypatch.setenv(D.ENV_EXCLUDE_FUENTES, CAT)
    stub = stub_with_fallback()
    first = normalize_strict(stub, QUERIES, gazetteer=None)
    second = normalize_strict(stub, QUERIES, gazetteer=None)
    assert digest(first) == digest(second) and len(stub._direccion) == 2


def test_changing_the_set_after_it_was_applied_is_refused(monkeypatch):
    stub = stub_with_fallback()
    monkeypatch.setenv(D.ENV_EXCLUDE_FUENTES, CAT)
    normalize_strict(stub, QUERIES, gazetteer=None)
    monkeypatch.delenv(D.ENV_EXCLUDE_FUENTES)
    with pytest.raises(RuntimeError, match="new normalizer"):
        normalize_strict(stub, QUERIES, gazetteer=None)


def test_unknown_fuente_or_missing_column_is_a_no_op(monkeypatch):
    monkeypatch.setenv(D.ENV_EXCLUDE_FUENTES, "does_not_exist")
    stub = stub_with_fallback()
    assert service.apply_exclusions(stub, ["does_not_exist"]) == 0 and len(stub._direccion) == 3
    plain = make_stub(NOM_DOCS + [FALLBACK_DOC], scores=SCORES)                   # production-style docs: no `fuente`
    assert service.apply_exclusions(plain, [CAT]) == 0 and len(plain._direccion) == 3
    assert service.apply_exclusions(stub, []) == 0


def test_a_non_tail_exclusion_is_refused_so_doc_ids_never_shift():
    stub = make_stub([FALLBACK_DOC] + NOM_DOCS, scores=[[0.9, 0.1, 0.1]])
    stub._fuente = np.array([CAT, CERT, CERT], dtype=object)
    with pytest.raises(ValueError, match="tail"):
        service.apply_exclusions(stub, [CAT])
    assert len(stub._direccion) == 3                                              # nothing was changed


def test_excluding_every_document_is_refused():
    stub = make_stub([FALLBACK_DOC], scores=[[0.9]])
    stub._fuente = np.array([CAT], dtype=object)
    with pytest.raises(ValueError, match="every"):
        service.apply_exclusions(stub, [CAT])


def test_malformed_values_are_refused_without_touching_the_normalizer():
    stub = stub_with_fallback()
    for bad in ("catastro", [1], [""], None):
        with pytest.raises((TypeError, ValueError)):
            service.apply_exclusions(stub, bad)
    assert len(stub._direccion) == 3


def test_dataframe_and_tensor_attributes_are_sliced_too():
    stub = stub_with_fallback()
    stub.docs = pd.DataFrame({"fuente": stub._fuente, "x": [1, 2, 3]})
    stub.doc_emb = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    stub._exact = {"a": 0, "b": 1, "c": 2}
    stub._parsed_docs = {0: "p0", 2: "p2"}
    service.apply_exclusions(stub, [CAT])
    assert len(stub.docs) == 2 and stub.docs.index.tolist() == [0, 1]
    assert stub.doc_emb.shape == (2, 4) and stub._exact == {"a": 0, "b": 1} and stub._parsed_docs == {0: "p0"}


def test_a_tiny_two_doc_normalizer_does_not_slice_unrelated_arrays():
    """`xy_mean` has two entries as well: only the named per-document attributes may be sliced."""
    stub = make_stub([FALLBACK_DOC, FALLBACK_DOC], scores=[[0.9, 0.9]])
    stub._fuente = np.array([CERT, CAT], dtype=object)
    stub.xy_mean = np.array([1.0, 2.0])
    service.apply_exclusions(stub, [CAT])
    assert stub.xy_mean.tolist() == [1.0, 2.0] and len(stub._direccion) == 1


def test_caches_built_from_the_documents_are_reset():
    stub = stub_with_fallback()
    stub._interp_index, stub._catalog_keys = object(), object()
    service.apply_exclusions(stub, [CAT])
    assert stub._interp_index is None and stub._catalog_keys is None
    plain = stub_with_fallback()                       # a normalizer that never built them gains no attribute
    service.apply_exclusions(plain, [CAT])
    assert not hasattr(plain, "_interp_index")
