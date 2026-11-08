"""``build_training_arrays(extra_queries=...)``: official (text, doc_id) pairs join the training set.

All data is synthetic (tests/tiny.py).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tiny import tiny_docs

from cali_address.dataset import build_training_arrays
from cali_address.model import encode_batch

#: sha256 prefixes of the arrays built by the code BEFORE ``extra_queries`` existed
#: (``tiny_docs(60)``, ``n_variants=3``, ``seed=7``): the default path must not move.
LEGACY_HASHES = {
    "q_tokens": "fc17406fbf9588cb62e4",
    "q_tokens_raw": "f847d162afbe687d137c",
    "q_doc": "0c446f13089b06bee939",
    "q_is_val": "12e9f829b0d85eda5668",
    "d_tokens": "be08dada70bef087d803",
    "doc_comuna": "061564f52e87b4972f87",
    "doc_barrio": "e7bd726d98d384fecb6e",
}


def _h(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:20]


@pytest.fixture(scope="module")
def docs():
    return tiny_docs(60)


@pytest.fixture(scope="module")
def base(docs):
    return build_training_arrays(docs, n_variants=3, seed=7)


def test_default_path_is_byte_identical_to_the_legacy_arrays(base):
    for key, expected in LEGACY_HASHES.items():
        assert _h(base[key]) == expected, key


def test_default_path_adds_only_an_all_zero_q_source(base):
    assert base["q_source"].dtype == np.uint8
    assert len(base["q_source"]) == len(base["q_doc"])
    assert not base["q_source"].any()


def test_empty_extra_queries_equal_none(docs, base):
    again = build_training_arrays(docs, n_variants=3, seed=7, extra_queries=[])
    for key in LEGACY_HASHES:
        assert _h(again[key]) == _h(base[key]), key


def test_extra_queries_are_appended_after_the_synthetic_ones(docs, base):
    extra = [("CL 901 # 902 - 903", 5), ("KR 904 # 905 - 906", 7)]
    out = build_training_arrays(docs, n_variants=3, seed=7, extra_queries=extra)
    n = len(base["q_doc"])
    assert len(out["q_doc"]) == n + 2
    # the synthetic block is untouched (the extra pairs consume no randomness)
    assert np.array_equal(out["q_doc"][:n], base["q_doc"])
    assert np.array_equal(out["q_tokens"][:n], base["q_tokens"])
    assert out["q_doc"][n:].tolist() == [5, 7]
    assert out["q_source"][:n].tolist() == [0] * n
    assert out["q_source"][n:].tolist() == [1, 1]
    # the official text is encoded like any other query
    assert np.array_equal(out["q_tokens"][n], encode_batch(["CL 901 # 902 - 903"])[0])


def test_official_queries_of_a_validation_doc_are_validation_queries(docs, base):
    val_docs = sorted({int(d) for d in base["q_doc"][base["q_is_val"]]})
    train_docs = sorted(set(range(len(docs))) - set(val_docs))
    extra = [(f"CL 91{i} # 92 - 93", d) for i, d in enumerate(val_docs + train_docs[:3])]
    out = build_training_arrays(docs, n_variants=3, seed=7, extra_queries=extra)
    n = len(base["q_doc"])
    official_val = out["q_is_val"][n:]
    assert official_val[: len(val_docs)].all()
    assert not official_val[len(val_docs):].any()
    # no document sits on both sides of the split
    val_ids = set(out["q_doc"][out["q_is_val"]].tolist())
    train_ids = set(out["q_doc"][~out["q_is_val"]].tolist())
    assert not (val_ids & train_ids)


def test_out_of_range_doc_id_fails_loudly(docs):
    for bad in (-1, len(docs)):
        with pytest.raises(ValueError, match="doc_id"):
            build_training_arrays(docs, n_variants=1, seed=7, extra_queries=[("CL 9011 # 9022 - 9033", bad)])


def test_blank_texts_are_skipped_not_encoded(docs, base):
    extra = [("", 3), ("   ", 3), (None, 3), ("CL 9071 # 9082 - 9093", 3)]
    out = build_training_arrays(docs, n_variants=3, seed=7, extra_queries=extra)
    assert len(out["q_doc"]) == len(base["q_doc"]) + 1
    assert out["q_source"][-1] == 1
