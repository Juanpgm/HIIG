"""scripts/tune_threshold.py: which validation queries feed the tuning sample (tiny fixtures, invented text)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from tiny import tiny_docs

from cali_address.dataset import build_training_arrays
from cali_address.model import encode_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tune():
    return _load_script("tune_threshold")


def _data(n_docs=48, official_per_doc=0, n_variants=3, seed=42):
    docs = tiny_docs(n_docs)
    extra = [(docs["direccion"][d].replace("#", "NO."), d) for d in range(n_docs) for _ in range(official_per_doc)]
    return docs, build_training_arrays(docs, n_variants=n_variants, seed=seed, extra_queries=extra or None)


# ------------------------------------------------------------------ old data: nothing changes
def test_old_data_replay_is_unchanged_and_validation_only(tune):
    docs, data = _data()
    texts, idx = tune.replay_validation_queries(docs, 3, 42, data["q_is_val"], data["q_doc"])
    assert idx.tolist() == np.flatnonzero(data["q_is_val"]).tolist() and len(texts) == len(idx) > 0
    # same call with an all-zero q_source (old arrays that carry the column) gives the same answer
    texts2, idx2 = tune.replay_validation_queries(docs, 3, 42, data["q_is_val"], data["q_doc"], data["q_source"])
    assert texts2 == texts and idx2.tolist() == idx.tolist()


def test_old_data_sample_matches_the_original_inline_selection(tune):
    docs, data = _data()
    texts, idx = tune.replay_validation_queries(docs, 3, 42, data["q_is_val"], data["q_doc"])
    ref_rng = np.random.default_rng(tune.TUNE_SEED)
    ref_pick = np.sort(ref_rng.choice(len(texts), size=min(7, len(texts)), replace=False))
    sample = tune.select_tuning_positives(docs, data, 7, np.random.default_rng(tune.TUNE_SEED))
    assert sample["texts"] == [texts[i] for i in ref_pick]
    assert sample["truth"].tolist() == data["q_doc"][idx][ref_pick].tolist()
    assert sample["n_official"] == 0 and sample["n_synthetic"] == len(ref_pick)


def test_old_data_without_q_source_key_still_works(tune):
    docs, data = _data(n_docs=500)
    data.pop("q_source")
    sample = tune.select_tuning_positives(docs, data, 5, np.random.default_rng(1))
    assert sample["n_official"] == 0 and len(sample["texts"]) == 5


# ------------------------------------------------------------------ new data: official queries are appended
def test_replay_with_appended_official_queries_returns_the_synthetic_validation_block(tune):
    """The doc-major assertion cannot hold once official queries follow the synthetic block."""
    docs, data = _data(official_per_doc=2)
    texts, idx = tune.replay_validation_queries(docs, 3, 42, data["q_is_val"], data["q_doc"], data["q_source"])
    syn_val = np.flatnonzero(data["q_is_val"] & (data["q_source"] == 0))
    assert idx.tolist() == syn_val.tolist() and len(texts) == len(syn_val)


def test_replayed_synthetic_texts_are_those_of_the_docs_only_build(tune):
    docs, plain = _data()
    _, mixed = _data(official_per_doc=2)
    t_plain, i_plain = tune.replay_validation_queries(docs, 3, 42, plain["q_is_val"], plain["q_doc"])
    t_mixed, i_mixed = tune.replay_validation_queries(docs, 3, 42, mixed["q_is_val"], mixed["q_doc"], mixed["q_source"])
    assert t_mixed == t_plain and i_mixed.tolist() == i_plain.tolist()


def test_n_variants_is_derived_from_the_synthetic_block_only(tune):
    docs, data = _data(official_per_doc=5)
    assert tune.n_variants_of(data["q_doc"], data["q_source"], len(docs)) == 3
    assert tune.n_variants_of(data["q_doc"][:144], None, len(docs)) == 3
    assert tune.n_variants_of(np.zeros(0, dtype=np.int64), None, len(docs)) == 0
    with pytest.raises(ValueError):
        tune.n_variants_of(data["q_doc"], data["q_source"], 0)


def test_official_validation_texts_are_validation_and_official_only(tune):
    docs, data = _data(official_per_doc=2)
    texts, idx = tune.official_validation_texts(data["q_tokens_raw"], data["q_is_val"], data["q_source"])
    expected = np.flatnonzero(data["q_is_val"] & (data["q_source"] == 1))
    assert idx.tolist() == expected.tolist() and len(texts) == len(idx) > 0
    assert data["q_is_val"][idx].all() and (data["q_source"][idx] == 1).all()
    # an official validation text decodes back to the pre-normalised view that was encoded
    doc = int(data["q_doc"][idx[0]])
    assert texts[0].replace(" ", "").startswith(docs["direccion"][doc].split()[0])


def test_decode_tokens_round_trip_padding_and_unknown_characters(tune):
    assert tune.decode_tokens(encode_text("CL 5 # 10 - 02")) == "CL 5 # 10 - 02"
    assert tune.decode_tokens(np.zeros(8, dtype=np.uint8)) == ""                  # all padding
    assert tune.decode_tokens(encode_text("CL 5 Ñ")) == "CL 5 ?"             # UNK keeps its slot, visibly
    assert tune.decode_tokens(encode_text("")) == ""


def test_mixed_sample_takes_half_synthetic_half_official_validation_queries(tune):
    docs, data = _data(n_docs=500, official_per_doc=4)
    sample = tune.select_tuning_positives(docs, data, 20, np.random.default_rng(tune.TUNE_SEED))
    assert sample["n_official"] == 10 and sample["n_synthetic"] == 10 and len(sample["texts"]) == 20
    val_docs = set(data["q_doc"][data["q_is_val"]].tolist())
    assert set(sample["truth"].tolist()) <= val_docs                              # validation documents only
    assert sample["source"].tolist() == [0] * 10 + [1] * 10


def test_mixed_sample_is_deterministic(tune):
    docs, data = _data(n_docs=500, official_per_doc=4)
    a = tune.select_tuning_positives(docs, data, 12, np.random.default_rng(tune.TUNE_SEED))
    b = tune.select_tuning_positives(docs, data, 12, np.random.default_rng(tune.TUNE_SEED))
    assert a["texts"] == b["texts"] and a["truth"].tolist() == b["truth"].tolist()


@pytest.mark.parametrize("n_syn,n_off,n_tune,exp_syn,exp_off", [
    (100, 0, 10, 10, 0),         # old data
    (100, 100, 10, 5, 5),        # even split
    (100, 100, 11, 6, 5),        # odd: the extra slot goes to synthetic
    (100, 3, 10, 7, 3),          # few official: the rest is synthetic
    (3, 100, 10, 3, 7),          # few synthetic: the rest is official
    (3, 2, 10, 3, 2),            # fewer than requested overall
    (0, 5, 10, 0, 5),            # no synthetic validation queries
    (0, 0, 10, 0, 0),            # nothing at all
    (100, 100, 0, 0, 0),         # zero requested
    (1, 1, 1, 1, 0),             # single slot
])
def test_stratified_counts(tune, n_syn, n_off, n_tune, exp_syn, exp_off):
    syn, off = tune.stratified_pick(np.random.default_rng(1), n_syn, n_off, n_tune)
    assert (len(syn), len(off)) == (exp_syn, exp_off)
    assert len(set(syn.tolist())) == len(syn) and len(set(off.tolist())) == len(off)      # no repeats
    assert (syn < max(n_syn, 1)).all() and (off < max(n_off, 1)).all()
    assert syn.tolist() == sorted(syn.tolist()) and off.tolist() == sorted(off.tolist())


def test_stratified_pick_without_official_reproduces_the_old_draw(tune):
    syn, off = tune.stratified_pick(np.random.default_rng(1234), 50, 0, 10)
    ref = np.sort(np.random.default_rng(1234).choice(50, size=10, replace=False))
    assert syn.tolist() == ref.tolist() and len(off) == 0


def test_empty_validation_split_yields_an_empty_sample_not_a_crash(tune):
    docs, data = _data()
    data["q_is_val"] = np.zeros(len(data["q_doc"]), dtype=bool)
    sample = tune.select_tuning_positives(docs, data, 5, np.random.default_rng(1))
    assert sample["texts"] == [] and len(sample["truth"]) == 0
