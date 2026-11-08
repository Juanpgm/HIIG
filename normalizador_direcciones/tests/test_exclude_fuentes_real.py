"""The serving switch on the REAL ``AddressNormalizer`` (tiny random model, CPU): switch ON == the base index, exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from test_index_fallback_experiment import base_experiment, build  # noqa: F401  (fixture + helper)
from test_index_fallback_files import world  # noqa: F401  (fixture)

from cali_address import decision as D
from cali_address import service
from cali_address.inference import AddressNormalizer
from cali_address.service import normalize_strict


@pytest.fixture()
def pair(world, base_experiment, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.delenv(D.ENV_EXCLUDE_FUENTES, raising=False)
    _, out, _ = build(world, base_experiment, tmp_path)
    return AddressNormalizer(str(out), device="cpu"), AddressNormalizer(str(base_experiment), device="cpu"), out


def queries(normalizer) -> list[str]:
    """Texts of the base docs, of the appended docs and noise."""
    texts = [str(t) for t in normalizer._direccion[:3]] + [str(t) for t in normalizer._direccion[-3:]]
    return texts + ["KR 26 H 1 # 973 - 912", "texto sin direccion"]


def test_switch_on_serves_exactly_the_index_without_the_appended_docs(pair):
    full, base, _ = pair
    n = len(base.docs)
    assert len(full.docs) == n + 7 and full._fuente[-1] == "catastro"
    dropped = service.apply_exclusions(full, ["catastro"])
    assert dropped == 7 and len(full.docs) == n and full.doc_emb.shape == base.doc_emb.shape
    assert torch.equal(full.doc_emb, base.doc_emb)
    for name in ("_direccion", "_predial", "_manzana", "_comuna", "_barrio", "_lat", "_lon", "_own_lon", "_own_lat",
                 "_n_predios", "_fuente", "_estado_nomenclatura", "_manzana_nunique", "_x_m", "_y_m"):
        pd.testing.assert_series_equal(pd.Series(getattr(full, name)), pd.Series(getattr(base, name)), obj=name)
    assert full._exact == base._exact
    pd.testing.assert_frame_equal(full.docs, base.docs)
    texts = queries(base)
    a, b = full.score_components(texts, k=20), base.score_components(texts, k=20)
    for key in ("doc", "sim", "fuzz", "struct", "geo"):
        assert np.array_equal(a[key], b[key]), key


def test_switch_on_gives_the_same_normalize_strict_rows_and_off_reaches_the_appended_docs(pair, monkeypatch):
    full, base, _ = pair
    texts = queries(full)
    unrestricted = normalize_strict(full, texts, gazetteer=None, threshold=0.0)
    assert "catastro" in unrestricted["fuente"].dropna().tolist()               # switch OFF: the appended docs are served
    monkeypatch.setenv(D.ENV_EXCLUDE_FUENTES, "catastro")
    restricted = normalize_strict(full, texts, gazetteer=None, threshold=0.0)
    expected = normalize_strict(base, texts, gazetteer=None, threshold=0.0)
    pd.testing.assert_frame_equal(restricted, expected)


def test_a_catastro_row_reports_the_source_and_its_own_centroid_when_the_switch_is_off(pair):
    full, _, _ = pair
    text = str(full._direccion[-1])
    row = normalize_strict(full, [text], gazetteer=None, threshold=0.0).iloc[0]
    assert row["fuente"] == "catastro" and row["selected_doc_id"] == len(full.docs) - 1
    assert row["estado_nomenclatura"] is None or pd.isna(row["estado_nomenclatura"])
    assert full._own_lon is not None and np.isfinite(full._own_lon[-1])
