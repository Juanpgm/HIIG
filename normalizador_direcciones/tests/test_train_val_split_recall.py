"""Validation recall split by query source (official vs synthetic), early-stop flags and --smoke alignment.

All fixtures are tiny CPU ones from ``tiny.py``; every address is invented.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

from tiny import tiny_config, tiny_docs, tiny_data_dir

from cali_address.dataset import build_spatial_negatives, build_training_arrays
from cali_address.train import train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OLD_HISTORY_KEYS = {
    "epoch", "loss", "loss_nce", "loss_coord", "loss_comuna", "loss_barrio", "inbatch_acc", "train_recall@1",
    "seconds", "epoch_seconds", "seqs_per_s", "gpu_mem_gb", "recall@1", "recall@5", "recall@20",
    "real_val_recall@1", "real_val_recall@5", "real_val_recall@20",
}
SPLIT_KEYS = {f"val_{src}_{k}" for src in ("official", "synthetic") for k in ("recall@1", "recall@5", "recall@20", "n")}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mixed_data(n_docs: int = 48, per_doc: int = 2):
    """Tiny training arrays with official queries appended and a deterministic validation mask."""
    docs = tiny_docs(n_docs)
    extra = [(f"{docs['direccion'][d]}".replace("#", "NO."), d) for d in range(n_docs) for _ in range(per_doc)]
    data = build_training_arrays(docs, n_variants=3, seed=7, extra_queries=extra)
    data["q_is_val"] = (data["q_doc"] % 4 == 0)
    return docs, data, build_spatial_negatives(docs, k=4)


def _run(data, neg, **cfg):
    return train_model(data, neg, tiny_config(**cfg), device="cpu")


# ------------------------------------------------------------------ split recall
def test_old_data_without_q_source_keeps_history_and_adds_no_new_keys():
    docs, data, neg = _mixed_data()
    data.pop("q_source")
    out = _run(data, neg)
    assert set(out["history"][-1]) == OLD_HISTORY_KEYS
    assert "final_val_recall" not in out


def test_all_zero_q_source_behaves_like_old_data():
    docs, data, neg = _mixed_data()
    data["q_source"] = np.zeros(len(data["q_doc"]), dtype=np.uint8)
    out = _run(data, neg)
    assert set(out["history"][-1]) == OLD_HISTORY_KEYS and "final_val_recall" not in out


def test_existing_overall_metrics_are_unchanged_by_the_split_report():
    docs, data, neg = _mixed_data()
    with_split = _run(data, neg)
    no_source = {k: v for k, v in data.items() if k != "q_source"}
    without = _run(no_source, neg)
    a, b = with_split["history"][-1], without["history"][-1]
    for key in ("recall@1", "recall@5", "recall@20", "train_recall@1", "loss"):
        assert a[key] == b[key], key        # the split must not consume training randomness


def test_mixed_sources_report_both_groups_separately():
    docs, data, neg = _mixed_data()
    out = _run(data, neg)
    row = out["history"][-1]
    assert SPLIT_KEYS <= set(row)
    val = data["q_is_val"]
    assert row["val_official_n"] == int((val & (data["q_source"] == 1)).sum()) > 0
    assert row["val_synthetic_n"] == int((val & (data["q_source"] == 0)).sum()) > 0
    for src in ("official", "synthetic"):
        for k in (1, 5, 20):
            assert 0.0 <= row[f"val_{src}_recall@{k}"] <= 1.0
    final = out["final_val_recall"]
    assert final["official"]["recall@1"] == row["val_official_recall@1"]
    assert final["synthetic"]["n"] == row["val_synthetic_n"]
    json.dumps(final)       # the report is written as JSON


def test_official_group_is_scored_on_official_queries_only():
    """With the official val queries made unmatchable the official recall collapses, the synthetic one does not."""
    docs, data, neg = _mixed_data()
    off_val = data["q_is_val"] & (data["q_source"] == 1)
    data["q_tokens"] = data["q_tokens"].copy()
    data["q_tokens"][off_val] = 0                      # all-padding queries
    out = _run(data, neg, epochs=2)
    row = out["history"][-1]
    assert row["val_official_recall@20"] <= row["val_synthetic_recall@20"]
    assert row["val_official_recall@1"] <= 1.0 / 1 and row["val_official_n"] > 0


def test_no_official_validation_queries_gives_none_recall_and_zero_n():
    docs, data, neg = _mixed_data()
    data["q_is_val"] = data["q_is_val"] & (data["q_source"] == 0)       # official only in train
    row = _run(data, neg)["history"][-1]
    assert row["val_official_n"] == 0 and row["val_official_recall@1"] is None
    assert row["val_synthetic_n"] > 0 and row["val_synthetic_recall@1"] is not None


def test_no_synthetic_validation_queries_gives_none_recall_and_zero_n():
    docs, data, neg = _mixed_data()
    data["q_is_val"] = data["q_is_val"] & (data["q_source"] == 1)
    row = _run(data, neg)["history"][-1]
    assert row["val_synthetic_n"] == 0 and row["val_synthetic_recall@5"] is None
    assert row["val_official_n"] > 0


def test_misaligned_q_source_is_rejected_before_training():
    docs, data, neg = _mixed_data()
    data["q_source"] = data["q_source"][:-1]
    with pytest.raises(ValueError, match="q_source"):
        _run(data, neg)


# ------------------------------------------------------------------ early stop flags
def test_report_flags_a_run_cut_by_the_time_budget():
    docs, data, neg = _mixed_data()
    out = _run(data, neg, epochs=3, max_minutes=0.0)
    assert out["stopped_early"] is True
    assert out["epochs_configured"] == 3 and out["epochs_completed"] < 3


def test_report_flags_a_complete_run():
    docs, data, neg = _mixed_data()
    out = _run(data, neg, epochs=2, max_minutes=5.0)
    assert out["stopped_early"] is False
    assert out["epochs_configured"] == 2 and out["epochs_completed"] == 2


# ------------------------------------------------------------------ scripts/train.py --smoke
def test_smoke_subset_slices_q_source_with_the_other_query_arrays():
    train = _load_script("train")
    n = 40_000
    q_doc = np.arange(n, dtype=np.int64)
    data = {
        "q_tokens": np.zeros((n, 3), np.uint8), "q_tokens_raw": np.zeros((n, 3), np.uint8),
        "q_doc": q_doc, "q_is_val": np.zeros(n, bool), "q_source": (q_doc % 2).astype(np.uint8),
        "d_tokens": np.zeros((5, 3), np.uint8),
    }
    train.subset_for_smoke(data)
    keep = 20_000
    assert len(data["q_doc"]) == keep == len(data["q_source"]) == len(data["q_tokens"]) == len(data["q_is_val"])
    assert data["q_source"].tolist() == (np.arange(keep) % 2).tolist()          # same rows, not a prefix by luck
    assert len(data["d_tokens"]) == 5                                            # documents are untouched


def test_smoke_subset_without_q_source_is_unchanged_and_does_not_invent_one():
    train = _load_script("train")
    data = {"q_tokens": np.zeros((4, 2), np.uint8), "q_tokens_raw": np.zeros((4, 2), np.uint8),
            "q_doc": np.array([0, 25_000, 1, 30_000]), "q_is_val": np.zeros(4, bool)}
    train.subset_for_smoke(data)
    assert data["q_doc"].tolist() == [0, 1] and "q_source" not in data


def test_smoke_subset_with_no_surviving_rows_is_empty_not_an_error():
    train = _load_script("train")
    data = {"q_tokens": np.zeros((2, 2), np.uint8), "q_tokens_raw": np.zeros((2, 2), np.uint8),
            "q_doc": np.array([30_000, 40_000]), "q_is_val": np.zeros(2, bool), "q_source": np.ones(2, np.uint8)}
    train.subset_for_smoke(data)
    assert len(data["q_source"]) == 0 == len(data["q_doc"])


def test_train_script_writes_stop_flags_and_split_into_train_report(tmp_path):
    train = _load_script("train")
    out = tmp_path / "exp"
    docs, data, neg = tiny_data_dir(str(out))
    mixed_docs, mixed, mneg = _mixed_data(48)
    np.savez_compressed(out / "train_data.npz", **mixed)
    rc = train.main(["--out-dir", str(out), "--device", "cpu", "--real-pairs", str(tmp_path / "none.parquet"),
                     "--set", "batch_size=16", "--set", "d_model=16", "--set", "n_layers=1",
                     "--set", "n_heads=2", "--set", "d_out=16", "--set", "epochs=1"])
    assert rc == 0
    report = json.load(open(out / "train_report.json"))
    assert report["stopped_early"] is False and report["epochs_completed"] == 1
    assert "final_val_recall" in report and report["final_val_recall"]["official"]["n"] > 0
