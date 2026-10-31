from __future__ import annotations

import os

import pytest

from servicio.config import ConfigError, detect_cpu_limit, load_settings, resolve_threads


def test_defaults_are_production_ready():
    s = load_settings({})
    assert s.backend == "onnx"
    assert s.threads is None  # auto
    assert s.index_precision == "fp32"
    assert s.port == 8000 and s.workers == 1
    assert s.backend_fallback is True
    assert s.index_url == "" and s.warnings == ()


def test_blank_values_fall_back_to_defaults():
    s = load_settings({"PORT": "  ", "NORMALIZER_BACKEND": "", "MAX_ROWS": ""})
    assert s.port == 8000 and s.backend == "onnx" and s.max_rows == 20000


@pytest.mark.parametrize("value", ["abc", "0", "-1", "65536", "80.5", "8 000"])
def test_bad_port_rejected(value):
    with pytest.raises(ConfigError, match="PORT"):
        load_settings({"PORT": value})


@pytest.mark.parametrize("value", ["1", "65535"])
def test_port_boundaries_accepted(value):
    assert load_settings({"PORT": value}).port == int(value)


def test_torch_backend_rejected_with_explanation():
    with pytest.raises(ConfigError, match="no incluye torch"):
        load_settings({"NORMALIZER_BACKEND": "torch"})


@pytest.mark.parametrize("value", ["onnx-fp16", "cuda", "ONNX_INT8", "1"])
def test_unknown_backend_rejected(value):
    with pytest.raises(ConfigError, match="NORMALIZER_BACKEND"):
        load_settings({"NORMALIZER_BACKEND": value})


def test_backend_is_case_insensitive():
    assert load_settings({"NORMALIZER_BACKEND": "ONNX-INT8"}).backend == "onnx-int8"


@pytest.mark.parametrize("value,expected", [("auto", None), ("AUTO", None), ("0", None), ("1", 1), ("256", 256)])
def test_threads_values(value, expected):
    assert load_settings({"NORMALIZER_ONNX_THREADS": value}).threads == expected


@pytest.mark.parametrize("value", ["-1", "257", "x", "2.5"])
def test_bad_threads_rejected(value):
    with pytest.raises(ConfigError, match="NORMALIZER_ONNX_THREADS"):
        load_settings({"NORMALIZER_ONNX_THREADS": value})


@pytest.mark.parametrize("value", ["fp8", "int8", "half"])
def test_bad_index_precision_rejected(value):
    with pytest.raises(ConfigError, match="NORMALIZER_ONNX_INDEX"):
        load_settings({"NORMALIZER_ONNX_INDEX": value})


def test_index_precision_fp16_case_insensitive():
    assert load_settings({"NORMALIZER_ONNX_INDEX": "FP16"}).index_precision == "fp16"


@pytest.mark.parametrize("name,value", [
    ("MAX_ROWS", "0"), ("MAX_ROWS", "-5"), ("MAX_ROWS", "1000001"), ("MAX_ROWS", "many"),
    ("MAX_UPLOAD_MB", "0"), ("MAX_UPLOAD_MB", "-1"), ("MAX_UPLOAD_MB", "nan"), ("MAX_UPLOAD_MB", "inf"),
    ("WARMUP_TIMEOUT_S", "0"), ("UVICORN_WORKERS", "0"), ("UVICORN_WORKERS", "17"),
    ("LOG_LEVEL", "verbose"), ("NORMALIZER_BACKEND_FALLBACK", "maybe"), ("HOST", "0.0.0.0 x"),
])
def test_bad_values_rejected(name, value):
    with pytest.raises(ConfigError, match=name):
        load_settings({name: value})


def test_multiple_workers_warns_about_memory():
    s = load_settings({"UVICORN_WORKERS": "2"})
    assert s.workers == 2 and any("2 GB" in w for w in s.warnings)


def test_fallback_flag_parsing():
    assert load_settings({"NORMALIZER_BACKEND_FALLBACK": "false"}).backend_fallback is False
    assert load_settings({"NORMALIZER_BACKEND_FALLBACK": "SI"}).backend_fallback is True


def test_index_url_requires_sha256():
    with pytest.raises(ConfigError, match="INDEX_SHA256"):
        load_settings({"INDEX_URL": "https://example.invalid/indice.tar.gz"})
    with pytest.raises(ConfigError, match="INDEX_SHA256"):
        load_settings({"INDEX_URL": "https://example.invalid/indice.tar.gz", "INDEX_SHA256": "abc"})


def test_index_url_rejects_plain_http():
    with pytest.raises(ConfigError, match="https"):
        load_settings({"INDEX_URL": "http://example.invalid/i.tar.gz", "INDEX_SHA256": "a" * 64})


def test_index_url_accepted_and_sha_normalised():
    s = load_settings({"INDEX_URL": "https://example.invalid/i.tar.gz", "INDEX_SHA256": "AB" * 32})
    assert s.index_sha256 == "ab" * 32


def test_sha_without_url_warns():
    s = load_settings({"INDEX_SHA256": "a" * 64})
    assert any("INDEX_URL" in w for w in s.warnings)


def test_paths_are_absolute():
    s = load_settings({"INDEX_DIR": "rel/indice"})
    assert os.path.isabs(s.index_dir)


# -- CPU limit --------------------------------------------------------------------------------
def _cg(tmp_path, files: dict) -> str:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="ascii")
    return str(tmp_path)


def _host():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def test_cgroup_v2_quota(tmp_path):
    root = _cg(tmp_path, {"cpu.max": "200000 100000"})
    assert detect_cpu_limit(root) == min(_host(), 2)


def test_cgroup_v2_fractional_quota_rounds_up(tmp_path):
    root = _cg(tmp_path, {"cpu.max": "150000 100000"})
    assert detect_cpu_limit(root) == min(_host(), 2)


def test_cgroup_v2_unlimited_uses_host(tmp_path):
    assert detect_cpu_limit(_cg(tmp_path, {"cpu.max": "max 100000"})) == _host()


def test_cgroup_v1_quota(tmp_path):
    root = _cg(tmp_path, {"cpu/cpu.cfs_quota_us": "100000", "cpu/cpu.cfs_period_us": "100000"})
    assert detect_cpu_limit(root) == 1


def test_cgroup_v1_unlimited(tmp_path):
    root = _cg(tmp_path, {"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"})
    assert detect_cpu_limit(root) == _host()


@pytest.mark.parametrize("content", ["garbage", "", "0 100000", "100000 0", "a b"])
def test_cgroup_malformed_falls_back_to_host(tmp_path, content):
    assert detect_cpu_limit(_cg(tmp_path, {"cpu.max": content})) == _host()


def test_no_cgroup_files(tmp_path):
    assert detect_cpu_limit(str(tmp_path / "missing")) == _host()


def test_resolve_threads_explicit_wins(tmp_path):
    s = load_settings({"NORMALIZER_ONNX_THREADS": "3"})
    assert resolve_threads(s, _cg(tmp_path, {"cpu.max": "100000 100000"})) == 3
    s = load_settings({})
    assert resolve_threads(s, str(tmp_path)) == 1


# -- INDEX_URL hardening ----------------------------------------------------------------------
def test_file_url_rejected_by_default():
    with pytest.raises(ConfigError, match="file://"):
        load_settings({"INDEX_URL": "file:///tmp/i.tar.gz", "INDEX_SHA256": "a" * 64})


def test_file_url_allowed_with_explicit_flag():
    s = load_settings({"INDEX_URL": "file:///tmp/i.tar.gz", "INDEX_SHA256": "a" * 64,
                       "INDEX_ALLOW_FILE_URL": "true"})
    assert s.allow_file_url is True and s.index_url.startswith("file://")


def test_download_cap_default_and_override():
    assert load_settings({}).download_max_mb == 4096
    assert load_settings({"INDEX_MAX_DOWNLOAD_MB": "10"}).download_max_mb == 10


@pytest.mark.parametrize("value", ["0", "-1", "abc", "nan", "inf"])
def test_download_cap_invalid_values(value):
    with pytest.raises(ConfigError, match="INDEX_MAX_DOWNLOAD_MB"):
        load_settings({"INDEX_MAX_DOWNLOAD_MB": value})


def test_allow_file_url_must_be_boolean():
    with pytest.raises(ConfigError, match="INDEX_ALLOW_FILE_URL"):
        load_settings({"INDEX_ALLOW_FILE_URL": "maybe"})


def test_versions_are_aligned_with_the_release():
    import cali_address
    import servicio
    from cali_address.api.main import app

    assert servicio.__version__ == cali_address.__version__ == app.version == "0.2.0"
