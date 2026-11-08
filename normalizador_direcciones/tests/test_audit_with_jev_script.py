"""``scripts/audit_with_jev.py``: privacy guard at the CLI, degradation paths, outputs. No network, sentinel key only."""

from __future__ import annotations

import importlib.util
import json
import os

import pandas as pd
import pytest

SENTINEL_KEY = "SENTINEL-NOT-A-REAL-KEY-0000"
QUESTION = "misma_ubicacion"


def _script():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("audit_with_jev_cli", os.path.join(root, "scripts", "audit_with_jev.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nomenclatura_frame(n=12):
    return pd.DataFrame({
        "nddirecc_original": [f"CL {900 + i} # 950 - {i} (Direccion Catastral)" for i in range(n)],
        "nddirecc": [f"CL {900 + i} # 950 - {i}" for i in range(n)],
        "ndidestado": ["cert" if i % 2 else "pcer" for i in range(n)],
        "ndidtipzon": ["URB" if i % 3 else "EXP" for i in range(n)],
        "marker_removed": [True] * n,
        "rebuilt_from_parts": [False] * n,
    })


class Transport:
    """Fake HTTP transport: answers every question with ``p``; counts calls."""

    def __init__(self, p=0.95, status=200, body=None, exc=None):
        self.p, self.status, self.body, self.exc = p, status, body, exc
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        parsed = json.loads(payload.decode("utf-8"))
        self.calls.append({"url": url, "state": parsed["state"], "headers": dict(headers)})
        if self.exc is not None:
            raise self.exc
        if self.body is not None:
            return self.status, {}, self.body
        answers = {q: {"type": "noul", "noul": self.p} for q in parsed["questions"] if q == QUESTION}
        return self.status, {}, json.dumps({"answers": answers}).encode()


@pytest.fixture
def nomenclatura_file(tmp_path):
    path = tmp_path / "nomenclatura_clean.parquet"
    nomenclatura_frame().to_parquet(path)
    return str(path)


def run(argv, tmp_path, transport=None, environ=None, **kw):
    mod = _script()
    out = str(tmp_path / "out")
    code = mod.main([*argv, "--out", out], transport=transport, environ={} if environ is None else environ,
                    sleep=lambda s: None, **kw)
    return code, out


KEY_ENV = {"JEV_API_KEY": SENTINEL_KEY}


# --------------------------------------------------------------------------- dry run / key handling
def test_dry_run_prints_payloads_needs_no_key_and_sends_nothing(nomenclatura_file, tmp_path, capsys):
    transport = Transport()
    code, out = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "5", "--dry-run"], tmp_path, transport)
    captured = capsys.readouterr()
    assert code == 0 and transport.calls == [] and not os.path.exists(out)
    lines = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 5
    assert set(lines[0]["body"]["state"]) == {"direccion_original", "direccion_normalizada"}
    assert "Direccion Catastral" in lines[0]["body"]["state"]["direccion_original"]
    assert "Authorization" not in captured.out


def test_missing_key_exits_2_before_any_network(nomenclatura_file, tmp_path, capsys):
    transport = Transport()
    code, _ = run(["--source", "nomenclatura", "--input", nomenclatura_file], tmp_path, transport, environ={})
    assert code == 2 and transport.calls == []
    assert "JEV_API_KEY" in capsys.readouterr().err


def test_unreadable_env_file_exits_2(nomenclatura_file, tmp_path):
    transport = Transport()
    code, _ = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--env-file", str(tmp_path / "nope.env")], tmp_path, transport)
    assert code == 2 and transport.calls == []


def test_env_file_is_the_way_to_provide_the_key(nomenclatura_file, tmp_path, capsys):
    env_file = tmp_path / "fake.env"
    env_file.write_text(f"OTHER=1\nJEV_API_KEY={SENTINEL_KEY}\n", encoding="utf-8")
    transport = Transport()
    code, out = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "4", "--env-file", str(env_file)], tmp_path, transport)
    assert code == 0 and len(transport.calls) == 4
    assert transport.calls[0]["headers"]["Authorization"] == f"Bearer {SENTINEL_KEY}"
    captured = capsys.readouterr()
    assert SENTINEL_KEY not in captured.out + captured.err
    for name in os.listdir(out):
        path = os.path.join(out, name)
        assert SENTINEL_KEY.encode() not in open(path, "rb").read()


def test_rejected_key_exits_2_without_writing_results(nomenclatura_file, tmp_path):
    transport = Transport(status=401, body=b'{"error": "nope"}')
    code, out = run(["--source", "nomenclatura", "--input", nomenclatura_file], tmp_path, transport, environ=KEY_ENV)
    assert code == 2 and len(transport.calls) == 1 and not os.path.exists(out)


def test_missing_package_exits_2_with_a_clear_message(nomenclatura_file, tmp_path, monkeypatch, capsys):
    from cali_address import jev_check

    def boom():
        raise jev_check.JevUnavailableError("The 'validacion_jev' package was not found.")

    monkeypatch.setattr(jev_check, "load_validacion_jev", boom)
    code, _ = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--dry-run"], tmp_path)
    assert code == 2 and "validacion_jev" in capsys.readouterr().err


# --------------------------------------------------------------------------- live path with a fake transport
def test_live_run_writes_json_and_parquet_and_splits_by_group(nomenclatura_file, tmp_path, capsys):
    transport = Transport(p=0.95)
    code, out = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "8", "--seed", "42"], tmp_path, transport, environ=KEY_ENV)
    assert code == 0 and len(transport.calls) == 8
    payload = json.loads(open(os.path.join(out, "jev_audit.json"), encoding="utf-8").read())
    assert payload["source"] == "nomenclatura" and payload["agreement"]["rate"] == 1.0
    assert set(payload["groups"]) == {"estado_nomenclatura", "zona", "marker_removed", "rebuilt_from_parts"}
    assert set(payload["groups"]["estado_nomenclatura"]) == {"cert", "pcer"}
    assert payload["privacy"] == {"origins": {"nomenclatura_public": 8}, "allow_private": False}
    assert len(pd.read_parquet(os.path.join(out, "jev_audit_rows.parquet"))) == 8
    assert "CL 9" not in capsys.readouterr().out  # the console summary carries counts, not addresses


def test_the_pipeline_default_never_allows_private_rows(nomenclatura_file, tmp_path):
    transport = Transport()
    code, out = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "2"], tmp_path, transport, environ=KEY_ENV)
    assert json.loads(open(os.path.join(out, "jev_audit.json"), encoding="utf-8").read())["privacy"]["allow_private"] is False


def test_sampling_is_reproducible_across_runs(nomenclatura_file, tmp_path):
    first, second = Transport(), Transport()
    run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "5", "--seed", "7"], tmp_path / "a", first, environ=KEY_ENV)
    run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "5", "--seed", "7"], tmp_path / "b", second, environ=KEY_ENV)
    assert [c["state"] for c in first.calls] == [c["state"] for c in second.calls]


def test_sample_larger_than_the_file_uses_every_row(nomenclatura_file, tmp_path):
    transport = Transport()
    code, _ = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "5000", "--max-rows", "5000"], tmp_path, transport, environ=KEY_ENV)
    assert code == 0 and len(transport.calls) == 12


def test_max_rows_caps_what_is_sent_and_defaults_to_300(nomenclatura_file, tmp_path):
    assert _script().build_parser().parse_args(["--source", "nomenclatura"]).max_rows == 300
    transport = Transport()
    run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "12", "--max-rows", "5"], tmp_path, transport, environ=KEY_ENV)
    assert len(transport.calls) == 5


def test_all_rows_failing_exits_3_and_still_writes_the_failure_counts(nomenclatura_file, tmp_path):
    transport = Transport(status=500, body=b'{"error": "down"}')
    code, out = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "3"], tmp_path, transport, environ=KEY_ENV)
    assert code == 3
    payload = json.loads(open(os.path.join(out, "jev_audit.json"), encoding="utf-8").read())
    assert payload["counts"]["error"] == 3 and payload["has_valid_evidence"] is False


def test_timeouts_are_error_rows_not_crashes(nomenclatura_file, tmp_path):
    transport = Transport(exc=TimeoutError("slow"))
    code, _ = run(["--source", "nomenclatura", "--input", nomenclatura_file, "--sample", "2"], tmp_path, transport, environ=KEY_ENV)
    assert code == 3


def test_empty_file_exits_3_without_calls(tmp_path):
    path = tmp_path / "nomenclatura_clean.parquet"
    nomenclatura_frame(0).to_parquet(path)
    transport = Transport()
    code, _ = run(["--source", "nomenclatura", "--input", str(path)], tmp_path, transport, environ=KEY_ENV)
    assert code == 3 and transport.calls == []


def test_missing_columns_exit_2_naming_them(tmp_path, capsys):
    path = tmp_path / "nomenclatura_clean.parquet"
    pd.DataFrame({"other": [1]}).to_parquet(path)
    code, _ = run(["--source", "nomenclatura", "--input", str(path), "--dry-run"], tmp_path)
    assert code == 2 and "nddirecc_original" in capsys.readouterr().err


def test_missing_input_file_exits_2(tmp_path):
    code, _ = run(["--source", "nomenclatura", "--input", str(tmp_path / "ghost.parquet"), "--dry-run"], tmp_path)
    assert code == 2


@pytest.mark.parametrize("bad", [["--sample", "-1"], ["--sample", "x"], ["--max-rows", "0"], ["--seed", "1.5"], ["--source", "dev"]])
def test_bad_arguments_exit_2(bad, nomenclatura_file, tmp_path):
    code, _ = run(["--input", nomenclatura_file, *(["--source", "nomenclatura"] if "--source" not in bad else []), *bad], tmp_path, Transport(), environ=KEY_ENV)
    assert code == 2


# --------------------------------------------------------------------------- privacy guard at the CLI
@pytest.mark.parametrize("name", ["dev.parquet", "frozen_test.parquet", "train_pairs.parquet", "unlabeled_addresses.parquet", "mystery.parquet"])
def test_private_or_unknown_inputs_are_refused_and_nothing_leaves(name, tmp_path, capsys):
    path = tmp_path / "splits" / name
    path.parent.mkdir()
    nomenclatura_frame().to_parquet(path)
    transport = Transport()
    for extra in ([], ["--dry-run"]):
        code, out = run(["--source", "nomenclatura", "--input", str(path), *extra], tmp_path, transport, environ=KEY_ENV)
        assert code == 2 and not os.path.exists(out)
    assert transport.calls == []
    assert "CL 9" not in capsys.readouterr().out


def test_allow_private_is_the_explicit_override(tmp_path):
    path = tmp_path / "dev.parquet"
    nomenclatura_frame().to_parquet(path)
    transport = Transport()
    code, out = run(["--source", "nomenclatura", "--input", str(path), "--sample", "2", "--allow-private"], tmp_path, transport, environ=KEY_ENV)
    assert code == 0 and len(transport.calls) == 2
    assert json.loads(open(os.path.join(out, "jev_audit.json"), encoding="utf-8").read())["privacy"]["allow_private"] is True


def test_a_public_name_cannot_launder_a_frozen_directory(tmp_path):
    folder = tmp_path / "frozen_test"
    folder.mkdir()
    path = folder / "nomenclatura_clean.parquet"
    nomenclatura_frame().to_parquet(path)
    transport = Transport()
    code, _ = run(["--source", "nomenclatura", "--input", str(path)], tmp_path, transport, environ=KEY_ENV)
    assert code == 2 and transport.calls == []


# --------------------------------------------------------------------------- synthetic source
def _docs_file(tmp_path, name="catastro_docs.parquet", n=15):
    path = tmp_path / name
    pd.DataFrame({"direccion": [f"CL {900 + i} # 950 - {i}" for i in range(n)], "numero_predial_nacional": ["x"] * n}).to_parquet(path)
    return str(path)


def fake_normalize(queries):
    return pd.DataFrame({
        "direccion_normalizada": [q.upper() for q in queries], "estado": ["OK"] * len(queries),
        "confiabilidad": [0.8] * len(queries), "fuente_normalizacion": ["catastro"] * len(queries),
    })


def test_synthetic_run_scores_queries_through_the_injected_normalizer(tmp_path):
    transport = Transport(p=0.95)
    docs = _docs_file(tmp_path)
    code, out = run(["--source", "synthetic", "--input", docs, "--sample", "6", "--seed", "42"], tmp_path, transport,
                    environ=KEY_ENV, normalize_fn=fake_normalize)
    assert code == 0 and len(transport.calls) == 6
    payload = json.loads(open(os.path.join(out, "jev_audit.json"), encoding="utf-8").read())
    assert payload["source"] == "synthetic" and payload["privacy"]["origins"] == {"synthetic": 6}
    assert set(payload["groups"]) == {"estado", "fuente_normalizacion"}


def test_synthetic_dry_run_never_calls_the_network(tmp_path, capsys):
    transport = Transport()
    code, _ = run(["--source", "synthetic", "--input", _docs_file(tmp_path), "--sample", "3", "--dry-run"], tmp_path, transport,
                  normalize_fn=fake_normalize)
    assert code == 0 and transport.calls == [] and len(capsys.readouterr().out.splitlines()) == 3


def test_synthetic_from_a_non_cadastre_file_is_refused(tmp_path):
    transport = Transport()
    code, _ = run(["--source", "synthetic", "--input", _docs_file(tmp_path, "dev.parquet"), "--sample", "3"], tmp_path, transport,
                  environ=KEY_ENV, normalize_fn=fake_normalize)
    assert code == 2 and transport.calls == []


def test_synthetic_normalizer_failure_exits_2_with_a_message(tmp_path, capsys):
    def broken(queries):
        raise RuntimeError("artifacts missing")

    code, _ = run(["--source", "synthetic", "--input", _docs_file(tmp_path), "--sample", "3", "--dry-run"], tmp_path, normalize_fn=broken)
    assert code == 2 and "artifacts missing" in capsys.readouterr().err
