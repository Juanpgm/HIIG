"""The letter-separation step of ``_pre_normalize`` is a per-artifact switch (plan 8.2).

Production artifacts were trained on the view WITHOUT the separation, so the default is off and an
artifact opts in with ``"pre_normalize_separates_letters": true`` in its ``index_meta.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import pandas as pd
import pytest
import torch

from tiny import tiny_config, tiny_data_dir

from cali_address import parser
from cali_address.inference import AddressNormalizer
from cali_address.parser import _pre_normalize
from cali_address.train import embed_documents, save_checkpoint, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "artifacts", "splits")

GLUED = "KR 32B # 946 - 131"
SPACED = "KR 32 B # 946 - 131"

# sha256 of "\n".join(_pre_normalize(raw) for raw in raw_address) computed with the parser as it was
# BEFORE normalize_spacing was wired into _pre_normalize (privacy: only hashes are stored).
PRE_NORMALIZE_VIEW_SHA = {
    "dev": "31ec977ccc083ad76af64e5e4e5ff1caeb6d1afd969aa4e7d6229e1953eadb70",
    "frozen_test": "55c1dcb7f37eddcdf6ea602bffaf4969b7e0922b496e0658af4dae46cfe5d067",
    "train_pairs": "48dacaedd11cdee322a1ea5bc08650199609b72e51a031a1141dc86531094a3d",
}


@pytest.fixture(autouse=True)
def _restore_flag():
    saved = parser.PRE_NORMALIZE_SEPARATES_LETTERS
    yield
    parser.PRE_NORMALIZE_SEPARATES_LETTERS = saved


# ------------------------------------------------------------------ default and API

def test_default_is_off_so_production_behaviour_is_unchanged():
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False
    assert _pre_normalize(GLUED) == GLUED


def test_context_manager_enables_and_restores():
    with parser.pre_normalize_separation(True):
        assert _pre_normalize(GLUED) == SPACED
    assert _pre_normalize(GLUED) == GLUED
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_context_manager_can_disable_inside_an_enabled_scope():
    parser.set_pre_normalize_separation(True)
    with parser.pre_normalize_separation(False):
        assert _pre_normalize(GLUED) == GLUED
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is True


def test_context_manager_restores_on_exception():
    with pytest.raises(RuntimeError):
        with parser.pre_normalize_separation(True):
            assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is True
            raise RuntimeError("boom")
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_context_managers_nest_and_unwind_in_order():
    with parser.pre_normalize_separation(True):
        with parser.pre_normalize_separation(False):
            assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False
        assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is True
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_setter_returns_the_previous_value():
    assert parser.set_pre_normalize_separation(True) is False
    assert parser.set_pre_normalize_separation(False) is True
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


@pytest.mark.parametrize("bad", ["false", "true", 1, 0, None, "", [] ])
def test_setter_and_context_manager_reject_non_booleans(bad):
    # "false" is truthy: silently accepting a string would turn the switch ON by mistake.
    with pytest.raises(TypeError):
        parser.set_pre_normalize_separation(bad)
    with pytest.raises(TypeError):
        with parser.pre_normalize_separation(bad):
            pass
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_helpers_are_exported():
    assert {"pre_normalize_separation", "set_pre_normalize_separation"} <= set(parser.__all__)


def test_parse_address_follows_the_switch_for_the_raw_view_only():
    # canonical output is the same either way (the parser already separates letters itself)
    a = parser.parse_address(GLUED).to_dict()
    with parser.pre_normalize_separation(True):
        b = parser.parse_address(GLUED).to_dict()
    a.pop("raw"), b.pop("raw")
    assert a == b


# ------------------------------------------------------------------ identical raw view when off

def _split(name):
    path = os.path.join(SPLITS, f"{name}.parquet")
    if not os.path.isfile(path):
        pytest.skip(f"{path} is not available")
    return pd.read_parquet(path, columns=["raw_address"])["raw_address"]


@pytest.mark.parametrize("name", sorted(PRE_NORMALIZE_VIEW_SHA))
def test_raw_view_of_every_split_row_is_identical_to_the_pre_spacing_code(name):
    raw = _split(name)
    with parser.pre_normalize_separation(False):
        digest = hashlib.sha256("\n".join(_pre_normalize(str(a)) for a in raw).encode()).hexdigest()
    assert digest == PRE_NORMALIZE_VIEW_SHA[name]


@pytest.mark.parametrize("name", sorted(PRE_NORMALIZE_VIEW_SHA))
def test_the_hash_is_sensitive_to_the_switch(name):
    # guards the guard: with the switch ON the view really differs, so the test above can fail
    raw = _split(name)
    with parser.pre_normalize_separation(True):
        digest = hashlib.sha256("\n".join(_pre_normalize(str(a)) for a in raw).encode()).hexdigest()
    assert digest != PRE_NORMALIZE_VIEW_SHA[name]


# ------------------------------------------------------------------ AddressNormalizer reads index_meta

@pytest.fixture(scope="module")
def tiny_artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("flag_artifacts")
    docs, data, neg = tiny_data_dir(str(base))
    out = train_model(data, neg, tiny_config(), device="cpu")
    save_checkpoint(str(base / "model.pt"), out["model"], out["config"], data)
    torch.save(embed_documents(out["model"], data["d_tokens"][:, :40], device="cpu").cpu(), str(base / "catastro_emb.pt"))
    return base


def _artifact_dir(tiny_artifacts, tmp_path, meta=None, raw_meta=None):
    dest = tmp_path / "art"
    shutil.copytree(tiny_artifacts, dest)
    if raw_meta is not None:
        (dest / "index_meta.json").write_text(raw_meta, encoding="utf-8")
    elif meta is not None:
        (dest / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(dest)


def test_normalizer_without_index_meta_keeps_the_switch_off(tiny_artifacts, tmp_path):
    parser.set_pre_normalize_separation(True)         # leaked state from "another" normalizer
    AddressNormalizer(_artifact_dir(tiny_artifacts, tmp_path), device="cpu")
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_normalizer_turns_the_switch_on_from_index_meta(tiny_artifacts, tmp_path):
    d = _artifact_dir(tiny_artifacts, tmp_path, {"pre_normalize_separates_letters": True, "n_docs": 48})
    AddressNormalizer(d, device="cpu")
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is True


def test_normalizer_explicit_false_in_index_meta_is_off(tiny_artifacts, tmp_path):
    parser.set_pre_normalize_separation(True)
    d = _artifact_dir(tiny_artifacts, tmp_path, {"pre_normalize_separates_letters": False})
    AddressNormalizer(d, device="cpu")
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_normalizer_index_meta_without_the_key_is_off(tiny_artifacts, tmp_path):
    parser.set_pre_normalize_separation(True)
    d = _artifact_dir(tiny_artifacts, tmp_path, {})
    AddressNormalizer(d, device="cpu")
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


@pytest.mark.parametrize("bad", ['"true"', "1", "null", "[]"])
def test_normalizer_rejects_a_non_boolean_flag(tiny_artifacts, tmp_path, bad):
    d = _artifact_dir(tiny_artifacts, tmp_path, raw_meta='{"pre_normalize_separates_letters": %s}' % bad)
    with pytest.raises(ValueError, match="pre_normalize_separates_letters"):
        AddressNormalizer(d, device="cpu")


def test_normalizer_with_a_malformed_index_meta_fails_loudly(tiny_artifacts, tmp_path):
    d = _artifact_dir(tiny_artifacts, tmp_path, raw_meta="{not json")
    with pytest.raises(ValueError, match="index_meta"):
        AddressNormalizer(d, device="cpu")


def test_second_normalizer_resets_the_process_global_switch(tiny_artifacts, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    on = _artifact_dir(tiny_artifacts, tmp_path / "a", {"pre_normalize_separates_letters": True})
    AddressNormalizer(on, device="cpu")
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is True
    off = _artifact_dir(tiny_artifacts, tmp_path / "b")
    AddressNormalizer(off, device="cpu")
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_normalizer_docstring_warns_about_the_process_global_switch():
    assert "process" in (AddressNormalizer.__doc__ or "").lower()
    assert "pre_normalize_separates_letters" in (AddressNormalizer.__doc__ or "")
