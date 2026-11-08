"""ONNX Runtime serving backend: export, int8 quantization, numpy search, backend selection.

Everything runs on CPU with tiny invented models; none of it touches ``artifacts/``. The torch backend
stays the default and is the reference every ONNX result is compared against.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from tiny import tiny_config, tiny_data_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")

from cali_address import model as model_mod  # noqa: E402
from cali_address.inference import AddressNormalizer  # noqa: E402
from cali_address.model import encode_batch  # noqa: E402
from cali_address.train import embed_documents, save_checkpoint, train_model  # noqa: E402

TORCH_BLOCKER = """
import importlib.abc, sys

class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked in this test")

sys.meta_path.insert(0, _BlockTorch())
"""


def run_python(code: str, **env) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = os.pathsep.join([SRC, TESTS])
    full_env.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=full_env, timeout=300)


# ------------------------------------------------------------------ shared tiny artifacts

STRINGS = [
    "CL 1 # 10 - 10", "calle 2 No 10-12", "KR 8 # 3 - 3", "xyz", "", "   ",
    "CL 5 # 10 - 18 " + "X" * 80,                      # over-length: truncated to max_len
    "A" * 40, "A" * 41,                                 # exactly max_len and one over
    "CRA 7 Nº 12 – 45 ÁÉÍÓÚ Ñ",                         # non-ASCII
    "CL 1", "KR 26 H 3 # 73 - 10",
]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("onnx_tiny")
    docs, data, neg = tiny_data_dir(str(base))
    out = train_model(data, neg, tiny_config(n_layers=2, d_model=32, n_heads=4, d_out=16), device="cpu")
    save_checkpoint(str(base / "model.pt"), out["model"], out["config"], data)
    torch.save(embed_documents(out["model"], data["d_tokens"][:, :40], device="cpu").cpu(), str(base / "catastro_emb.pt"))
    return str(base)


@pytest.fixture(scope="module")
def torch_model(tiny_dir):
    from cali_address.train import load_checkpoint
    model, ckpt = load_checkpoint(os.path.join(tiny_dir, "model.pt"), dropout=0.0)
    return model.eval()


def torch_reference(model, texts, max_len=40):
    ids = torch.from_numpy(encode_batch(texts, max_len).astype(np.int64))
    with torch.no_grad():
        pooled = model.pool(ids, 0)
        emb = F.normalize(model.proj(pooled), dim=-1)
        xy = model.coord_head(pooled)
    return emb.numpy(), xy.numpy()


def cosine_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


# ------------------------------------------------------------------ tokenizer split


def test_tokenizer_module_is_the_single_source_of_encode_batch():
    from cali_address import tokenizer
    assert model_mod.encode_batch is tokenizer.encode_batch
    assert model_mod.encode_text is tokenizer.encode_text
    assert model_mod.VOCAB is tokenizer.VOCAB


def test_tokenizer_imports_without_torch():
    proc = run_python(TORCH_BLOCKER + "from cali_address.tokenizer import encode_batch\n"
                      "print(encode_batch(['CL 1 # 2'], 40).shape)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "(1, 40)"


# ------------------------------------------------------------------ numpy search


def _brute(q, docs16, k):
    sims = q.astype(np.float32) @ docs16.astype(np.float32).T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(sims, order, axis=1), order


@pytest.mark.parametrize("block", [1, 7, 64, 1000, 10_000])
def test_numpy_index_matches_brute_force(block):
    from cali_address.onnx_backend import NumpyDocIndex
    rng = np.random.default_rng(0)
    docs = rng.standard_normal((1000, 16)).astype(np.float16)
    q = rng.standard_normal((9, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scores, idx = NumpyDocIndex(docs, block=block).search_exact(q, k=5)
    ref_scores, ref_idx = _brute(q, docs, 5)
    assert idx.shape == (9, 5) and scores.shape == (9, 5)
    assert np.array_equal(idx, ref_idx)
    assert np.allclose(scores, ref_scores, atol=1e-6)
    assert (np.diff(scores, axis=1) <= 0).all()          # descending


def test_numpy_index_chunks_do_not_change_the_result():
    from cali_address.onnx_backend import NumpyDocIndex
    rng = np.random.default_rng(1)
    docs = rng.standard_normal((300, 8)).astype(np.float16)
    q = rng.standard_normal((50, 8)).astype(np.float32)
    index = NumpyDocIndex(docs, block=64)
    whole = index.search_exact(q, k=4, chunk=512)
    parts = index.search_exact(q, k=4, chunk=7)
    assert np.array_equal(whole[1], parts[1]) and np.array_equal(whole[0], parts[0])


def test_numpy_index_edge_shapes():
    from cali_address.onnx_backend import NumpyDocIndex
    docs = np.eye(4, 8, dtype=np.float16)
    index = NumpyDocIndex(docs, block=2)
    empty_scores, empty_idx = index.search_exact(np.zeros((0, 8), dtype=np.float32), k=3)
    assert empty_scores.shape == (0, 3) and empty_idx.shape == (0, 3)
    scores, idx = index.search_exact(np.eye(1, 8, 2, dtype=np.float32), k=50)   # k > n_docs
    assert idx.shape == (1, 4) and idx[0, 0] == 2 and scores[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="dimension"):
        index.search_exact(np.zeros((1, 5), dtype=np.float32), k=3)
    assert (index.shape, str(index.dtype)) == ((4, 8), "float16")


def test_numpy_index_rejects_non_matrix():
    from cali_address.onnx_backend import NumpyDocIndex
    with pytest.raises(ValueError, match="2-D"):
        NumpyDocIndex(np.zeros(5, dtype=np.float16))


# ------------------------------------------------------------------ export + OnnxEncoder


@pytest.fixture(scope="module")
def onnx_files(tiny_dir, tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from cali_address.onnx_backend import export_onnx, quantize_int8
    out = tmp_path_factory.mktemp("onnx_out")
    fp32 = str(out / "model.onnx")
    meta = export_onnx(os.path.join(tiny_dir, "model.pt"), fp32, opset=17)
    int8 = quantize_int8(fp32, str(out / "model.int8.onnx"))
    return {"dir": str(out), "fp32": fp32, "int8": int8, "meta": meta}


def test_export_writes_model_and_meta(onnx_files, tiny_dir):
    meta = onnx_files["meta"]
    assert os.path.getsize(onnx_files["fp32"]) > 0
    on_disk = json.load(open(os.path.join(onnx_files["dir"], "onnx_meta.json"), encoding="utf-8"))
    assert on_disk == meta
    assert meta["max_len"] == 40 and meta["d_out"] == 16 and meta["opset"] == 17 and meta["role"] == 0
    assert len(meta["xy_mean"]) == 2 and len(meta["xy_std"]) == 2
    assert len(meta["model_sha256"]) == 64


def test_fp32_onnx_matches_torch_embeddings_and_coordinates(onnx_files, torch_model):
    from cali_address.onnx_backend import OnnxEncoder
    enc = OnnxEncoder(onnx_files["fp32"])
    emb, xy = enc.embed_with_xy(STRINGS)
    ref_emb, ref_xy = torch_reference(torch_model, STRINGS)
    assert emb.dtype == np.float32 and emb.shape == ref_emb.shape
    assert cosine_rows(emb, ref_emb).min() > 0.99999
    assert np.abs(emb - ref_emb).max() < 1e-5
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    mean, std = np.asarray(onnx_files["meta"]["xy_mean"]), np.asarray(onnx_files["meta"]["xy_std"])
    assert np.abs(xy - (ref_xy * std + mean)).max() < 1e-2   # metres; coordinates are ~1e6 in EPSG:3116


def test_dynamic_batch_gives_the_same_row_for_any_batch_size(onnx_files):
    from cali_address.onnx_backend import OnnxEncoder
    enc = OnnxEncoder(onnx_files["fp32"])
    whole = enc.embed(STRINGS)
    for size in (1, 2, 5, len(STRINGS)):
        parts = np.concatenate([enc.embed(STRINGS[i:i + size], batch=size) for i in range(0, len(STRINGS), size)])
        assert np.abs(parts - whole).max() < 1e-5


def test_empty_batch_returns_empty_arrays(onnx_files):
    from cali_address.onnx_backend import OnnxEncoder
    enc = OnnxEncoder(onnx_files["fp32"])
    emb, xy = enc.embed_with_xy([])
    assert emb.shape == (0, 16) and xy.shape == (0, 2)
    assert enc.embed([]).shape == (0, 16)


def test_single_text_and_over_length_text_equal_their_truncation(onnx_files):
    from cali_address.onnx_backend import OnnxEncoder
    enc = OnnxEncoder(onnx_files["fp32"])
    one = enc.embed(["CL 1 # 10 - 10"])
    assert one.shape == (1, 16)
    long_text = "A" * 41 + "ZZZ"
    assert np.array_equal(enc.embed([long_text]), enc.embed(["A" * 40]))


def test_int8_is_smaller_and_close_to_fp32(onnx_files, torch_model):
    from cali_address.onnx_backend import OnnxEncoder
    assert os.path.getsize(onnx_files["int8"]) < os.path.getsize(onnx_files["fp32"])
    emb = OnnxEncoder(onnx_files["int8"], meta_path=os.path.join(onnx_files["dir"], "onnx_meta.json")).embed(STRINGS)
    ref_emb, _ = torch_reference(torch_model, STRINGS)
    assert np.isfinite(emb).all()
    assert cosine_rows(emb, ref_emb).min() > 0.9      # tiny random model: only a sanity bound; real numbers are in docs/onnx.md


def test_export_rejects_old_opset_and_missing_checkpoint(tiny_dir, tmp_path):
    from cali_address.onnx_backend import export_onnx
    with pytest.raises(ValueError, match="opset"):
        export_onnx(os.path.join(tiny_dir, "model.pt"), str(tmp_path / "m.onnx"), opset=13)
    with pytest.raises(FileNotFoundError, match="model.pt"):
        export_onnx(str(tmp_path / "nope" / "model.pt"), str(tmp_path / "m.onnx"))


def test_quantize_missing_input_is_a_clear_error(tmp_path):
    pytest.importorskip("onnxruntime")
    from cali_address.onnx_backend import quantize_int8
    with pytest.raises(FileNotFoundError, match="export"):
        quantize_int8(str(tmp_path / "missing.onnx"), str(tmp_path / "out.onnx"))


def test_encoder_with_missing_file_explains_how_to_export(tmp_path):
    pytest.importorskip("onnxruntime")
    from cali_address.onnx_backend import OnnxEncoder
    with pytest.raises(FileNotFoundError, match="export_onnx.py"):
        OnnxEncoder(str(tmp_path / "model.onnx"))


def test_convert_embeddings_roundtrip(tiny_dir, tmp_path):
    from cali_address.onnx_backend import convert_embeddings
    out = convert_embeddings(os.path.join(tiny_dir, "catastro_emb.pt"), str(tmp_path / "catastro_emb.npy"))
    arr = np.load(out)
    ref = torch.load(os.path.join(tiny_dir, "catastro_emb.pt"), map_location="cpu").numpy()
    assert arr.dtype == np.float16 and np.array_equal(arr, ref)


# ------------------------------------------------------------------ backend selection


def test_resolve_backend_default_param_env_and_invalid(monkeypatch):
    from cali_address.onnx_backend import resolve_backend
    monkeypatch.delenv("NORMALIZER_BACKEND", raising=False)
    assert resolve_backend(None) == "torch"
    monkeypatch.setenv("NORMALIZER_BACKEND", "onnx-int8")
    assert resolve_backend(None) == "onnx-int8"
    assert resolve_backend("torch") == "torch"            # explicit parameter beats the environment
    monkeypatch.setenv("NORMALIZER_BACKEND", "  ONNX ")
    assert resolve_backend(None) == "onnx"
    monkeypatch.setenv("NORMALIZER_BACKEND", "")
    assert resolve_backend(None) == "torch"                # blank env = unset
    with pytest.raises(ValueError, match="torch, onnx, onnx-int8"):
        resolve_backend("tensorrt")


@pytest.fixture(scope="module")
def onnx_artifacts(tiny_dir, tmp_path_factory):
    """A copy of the tiny artifact dir with the ONNX files + catastro_emb.npy next to model.pt."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import shutil

    from cali_address.onnx_backend import convert_embeddings, export_onnx, quantize_int8
    dest = str(tmp_path_factory.mktemp("onnx_art") / "art")
    shutil.copytree(tiny_dir, dest)
    export_onnx(os.path.join(dest, "model.pt"), os.path.join(dest, "model.onnx"))
    quantize_int8(os.path.join(dest, "model.onnx"), os.path.join(dest, "model.int8.onnx"))
    convert_embeddings(os.path.join(dest, "catastro_emb.pt"), os.path.join(dest, "catastro_emb.npy"))
    return dest


SAMPLE = [
    "CL 1 # 10 - 10", "calle 2 No 10-12", "CL 3 # 11 - 14", "Cl 4 #11 16", "CL 5 # 10 - 18",
    "CL 6 # 11 - 20", "calle 7 # 10 - 99", "KR 8 # 3 - 3", "CL 2 # 10 - 14", "xyz",
]


def test_default_backend_is_torch(onnx_artifacts, monkeypatch):
    monkeypatch.delenv("NORMALIZER_BACKEND", raising=False)
    norm = AddressNormalizer(onnx_artifacts, device="cpu")
    assert norm.backend == "torch"
    assert isinstance(norm.doc_emb, torch.Tensor)


@pytest.mark.parametrize("backend, sim_tol", [("onnx", 2e-5), ("onnx-int8", 0.3)])
def test_onnx_backends_score_like_torch(onnx_artifacts, backend, sim_tol, monkeypatch):
    monkeypatch.delenv("NORMALIZER_BACKEND", raising=False)
    ref = AddressNormalizer(onnx_artifacts, device="cpu").score_components(SAMPLE, k=5)
    norm = AddressNormalizer(onnx_artifacts, device="cpu", backend=backend)
    assert norm.backend == backend and norm.device == "cpu"
    got = norm.score_components(SAMPLE, k=5)
    assert got["doc"].shape == ref["doc"].shape
    for r in range(len(SAMPLE)):
        a = {int(d): (s, g) for d, s, g in zip(ref["doc"][r], ref["sim"][r], ref["geo"][r]) if d >= 0}
        b = {int(d): (s, g) for d, s, g in zip(got["doc"][r], got["sim"][r], got["geo"][r]) if d >= 0}
        common = set(a) & set(b)
        assert len(common) >= len(a) - 1                      # a near-tie may swap the last slot
        if backend == "onnx":
            assert all(abs(a[d][0] - b[d][0]) < sim_tol for d in common)
            assert all(abs(a[d][1] - b[d][1]) < 1e-3 for d in common)


def test_env_variable_selects_the_backend(onnx_artifacts, monkeypatch):
    monkeypatch.setenv("NORMALIZER_BACKEND", "onnx")
    assert AddressNormalizer(onnx_artifacts, device="cpu").backend == "onnx"


def test_onnx_backend_ignores_a_cuda_request(onnx_artifacts):
    assert AddressNormalizer(onnx_artifacts, device="cuda", backend="onnx").device == "cpu"


def test_onnx_backend_degenerate_batches(onnx_artifacts):
    norm = AddressNormalizer(onnx_artifacts, device="cpu", backend="onnx")
    assert norm.score_components([], k=5)["doc"].shape[0] == 0
    assert norm.score_components(["", "   ", None, "!!!"], k=5)["doc"].shape[0] == 4
    assert len(norm.normalize_batch(["CL 1 # 10 - 10"])) == 1
    assert len(norm.normalize_batch([])) == 0


def test_onnx_backend_normalize_strict_matches_torch_decisions(onnx_artifacts):
    from cali_address.service import normalize_strict
    ref = normalize_strict(AddressNormalizer(onnx_artifacts, device="cpu"), SAMPLE, gazetteer=None)
    got = normalize_strict(AddressNormalizer(onnx_artifacts, device="cpu", backend="onnx"), SAMPLE, gazetteer=None)
    assert list(got.columns) == list(ref.columns)
    assert got["estado"].tolist() == ref["estado"].tolist()
    assert got["numero_predial_nacional"].fillna("").tolist() == ref["numero_predial_nacional"].fillna("").tolist()


def test_missing_onnx_file_is_a_clear_error_and_torch_still_works(onnx_artifacts, tmp_path):
    import shutil
    dest = str(tmp_path / "art")
    shutil.copytree(onnx_artifacts, dest)
    os.remove(os.path.join(dest, "model.int8.onnx"))
    with pytest.raises(FileNotFoundError, match=r"model\.int8\.onnx.*export_onnx\.py"):
        AddressNormalizer(dest, device="cpu", backend="onnx-int8")
    assert AddressNormalizer(dest, device="cpu", backend="onnx").backend == "onnx"
    assert AddressNormalizer(dest, device="cpu", backend="torch").backend == "torch"


def test_missing_embedding_npy_is_a_clear_error(onnx_artifacts, tmp_path):
    import shutil
    dest = str(tmp_path / "art")
    shutil.copytree(onnx_artifacts, dest)
    os.remove(os.path.join(dest, "catastro_emb.npy"))
    with pytest.raises(FileNotFoundError, match=r"catastro_emb\.npy.*export_onnx\.py"):
        AddressNormalizer(dest, device="cpu", backend="onnx")


def test_stale_export_is_refused(onnx_artifacts, tmp_path):
    import shutil
    dest = str(tmp_path / "art")
    shutil.copytree(onnx_artifacts, dest)
    with open(os.path.join(dest, "model.pt"), "ab") as fh:
        fh.write(b"\0")                                   # the checkpoint changed after the export
    with pytest.raises(ValueError, match="stale"):
        AddressNormalizer(dest, device="cpu", backend="onnx")


def test_embedding_count_mismatch_is_refused(onnx_artifacts, tmp_path):
    import shutil
    dest = str(tmp_path / "art")
    shutil.copytree(onnx_artifacts, dest)
    emb = np.load(os.path.join(dest, "catastro_emb.npy"))
    np.save(os.path.join(dest, "catastro_emb.npy"), emb[:-1])
    with pytest.raises((AssertionError, ValueError), match="mismatch"):
        AddressNormalizer(dest, device="cpu", backend="onnx")


def test_unknown_backend_value_raises_before_loading_anything(onnx_artifacts):
    with pytest.raises(ValueError, match="torch, onnx, onnx-int8"):
        AddressNormalizer(onnx_artifacts, device="cpu", backend="coreml")


def test_onnxruntime_not_installed_gives_a_clear_message_and_torch_is_unchanged(onnx_artifacts, monkeypatch):
    from cali_address.onnx_backend import OnnxUnavailableError
    monkeypatch.setitem(sys.modules, "onnxruntime", None)       # `import onnxruntime` now raises ImportError
    with pytest.raises(OnnxUnavailableError, match=r"cali-address\[onnx\]"):
        AddressNormalizer(onnx_artifacts, device="cpu", backend="onnx")
    norm = AddressNormalizer(onnx_artifacts, device="cpu")
    assert norm.backend == "torch"
    assert norm.score_components(SAMPLE[:2], k=3)["doc"].shape[0] == 2


def test_onnx_backend_works_when_torch_cannot_be_imported(onnx_artifacts):
    code = TORCH_BLOCKER + f"""
import json
from cali_address.inference import AddressNormalizer
from cali_address.service import normalize_strict
norm = AddressNormalizer({onnx_artifacts!r}, device="cpu", backend="onnx-int8")
frame = normalize_strict(norm, {SAMPLE!r}, gazetteer=None)
assert "torch" not in sys.modules, "torch leaked into the onnx serving path"
print(json.dumps(frame["estado"].tolist()))
"""
    proc = run_python(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert len(json.loads(proc.stdout.strip().splitlines()[-1])) == len(SAMPLE)


def test_torch_backend_without_torch_fails_with_the_import_error(onnx_artifacts):
    code = TORCH_BLOCKER + f"""
from cali_address.inference import AddressNormalizer
try:
    AddressNormalizer({onnx_artifacts!r}, device="cpu")
except ImportError as exc:
    print("IMPORT_ERROR", exc)
"""
    proc = run_python(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "IMPORT_ERROR" in proc.stdout


# ------------------------------------------------------------------ export_onnx.py CLI


def _run_script(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_onnx.py"), *args],
                          capture_output=True, text=True, timeout=300)


def test_export_script_writes_the_four_serving_files(tiny_dir, tmp_path):
    pytest.importorskip("onnxruntime")
    out = tmp_path / "serving"
    proc = _run_script("--artifacts-dir", tiny_dir, "--out-dir", str(out))
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("model.onnx", "model.int8.onnx", "onnx_meta.json", "catastro_emb.npy"):
        assert (out / name).stat().st_size > 0, name


def test_export_script_missing_checkpoint_is_a_clear_error(tmp_path):
    proc = _run_script("--artifacts-dir", str(tmp_path / "empty"), "--out-dir", str(tmp_path / "o"))
    assert proc.returncode != 0
    assert "model.pt" in proc.stderr


def test_onnx_backend_never_imports_torch_even_when_installed(onnx_artifacts):
    code = f"""
import sys
from cali_address.inference import AddressNormalizer
from cali_address.service import normalize_strict
norm = AddressNormalizer({onnx_artifacts!r}, device="cpu", backend="onnx")
normalize_strict(norm, {SAMPLE!r}, gazetteer=None)
print("TORCH_LOADED" if "torch" in sys.modules else "TORCH_ABSENT")
"""
    proc = run_python(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "TORCH_ABSENT"


# ------------------------------------------------------------------ bench_backends.py


def test_bench_script_smoke_on_tiny_artifacts(onnx_artifacts, tmp_path):
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "bench_backends.py"), "--set", f"tiny={onnx_artifacts}",
         "--backends", "torch,onnx,onnx-int8", "--repeats", "2", "--latency-n", "5", "--batch-sizes", "3,7",
         "--warmup-n", "4", "--pool-n", "10", "--threads", "2", "--no-gazetteer", "--no-sizes", "--out", str(out)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.load(open(out, encoding="utf-8"))
    runs = data["runs"]["tiny"]
    assert set(runs) == {"torch", "onnx", "onnx-int8"}
    for backend, r in runs.items():
        assert r["cold_start_s"] > 0 and r["peak_rss_mb"] > 0
        assert r["latency_ms"]["p50"] > 0 and r["latency_ms"]["p95"] >= r["latency_ms"]["p50"]
        assert set(r["throughput"]) == {"3", "7"} and r["throughput"]["7"]["rows_per_s_median"] > 0
        assert r["torch_imported"] is (backend == "torch")
    assert "BACKEND" in proc.stdout.upper()          # the table is printed


def test_bench_script_unknown_backend_fails_clearly(onnx_artifacts, tmp_path):
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "bench_backends.py"), "--set", f"tiny={onnx_artifacts}",
         "--backends", "tensorrt", "--out", str(tmp_path / "b.json")], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "tensorrt" in proc.stderr


# ------------------------------------------------------------------ resident index precision


def test_fp32_index_gives_the_same_results_and_holds_only_one_copy():
    from cali_address.onnx_backend import NumpyDocIndex
    rng = np.random.default_rng(3)
    docs = rng.standard_normal((500, 16)).astype(np.float16)
    q = rng.standard_normal((20, 16)).astype(np.float32)
    lean, wide = NumpyDocIndex(docs, block=64), NumpyDocIndex(docs, block=64, fp32_cache=True)
    assert lean.resident_bytes == 500 * 16 * 2 and wide.resident_bytes == 500 * 16 * 4   # no fp16 copy kept
    s1, i1 = lean.search_exact(q, k=7)
    s2, i2 = wide.search_exact(q, k=7)
    assert np.array_equal(i1, i2) and np.array_equal(s1, s2)       # fp16 -> fp32 is a lossless upcast
    assert wide.shape == (500, 16)


def test_index_precision_env_selects_fp32_or_fp16(onnx_artifacts, monkeypatch):
    monkeypatch.delenv("NORMALIZER_ONNX_INDEX", raising=False)
    assert AddressNormalizer(onnx_artifacts, device="cpu", backend="onnx").doc_emb.dtype == np.float32
    monkeypatch.setenv("NORMALIZER_ONNX_INDEX", "fp16")
    assert AddressNormalizer(onnx_artifacts, device="cpu", backend="onnx").doc_emb.dtype == np.float16
    monkeypatch.setenv("NORMALIZER_ONNX_INDEX", "int4")
    with pytest.raises(ValueError, match="NORMALIZER_ONNX_INDEX"):
        AddressNormalizer(onnx_artifacts, device="cpu", backend="onnx")


def test_onnx_encoder_default_batch_keeps_the_activation_arena_small():
    """Measured: 1,024-row runs peak at 1.1-1.4 GB of working set, 256-row runs at 0.3-0.4 GB (and are faster)."""
    import inspect

    from cali_address.onnx_backend import DEFAULT_ENCODE_BATCH, OnnxEncoder
    assert DEFAULT_ENCODE_BATCH == 256
    for method in (OnnxEncoder.embed, OnnxEncoder.embed_with_xy, OnnxEncoder.predict_xy):
        assert inspect.signature(method).parameters["batch"].default == DEFAULT_ENCODE_BATCH


def test_bench_merge_keeps_runs_measured_earlier(onnx_artifacts, tmp_path):
    out = tmp_path / "bench.json"
    base = [sys.executable, os.path.join(ROOT, "scripts", "bench_backends.py"), "--set", f"tiny={onnx_artifacts}",
            "--repeats", "1", "--latency-n", "3", "--batch-sizes", "3", "--warmup-n", "3", "--pool-n", "6",
            "--threads", "2", "--no-gazetteer", "--no-sizes", "--out", str(out)]
    assert subprocess.run(base + ["--backends", "torch"], capture_output=True, text=True, timeout=600).returncode == 0
    proc = subprocess.run(base + ["--backends", "onnx", "--merge"], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert set(json.load(open(out, encoding="utf-8"))["runs"]["tiny"]) == {"torch", "onnx"}
