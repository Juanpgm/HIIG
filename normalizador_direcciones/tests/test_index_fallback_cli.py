"""``scripts/index_fallback.py``: add / remove / experiment (tiny invented data, CPU)."""

from __future__ import annotations

import importlib.util
import json
import os

import pandas as pd
import pytest
from test_index_fallback_experiment import _tiny_model_files  # noqa: F401
from test_index_fallback_experiment import base_experiment  # noqa: F401  (fixture)
from test_index_fallback_files import meta_of, sha, world  # noqa: F401  (fixture)

from cali_address import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("script_index_fallback", os.path.join(ROOT, "scripts", "index_fallback.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_argv(world, out, *extra):  # noqa: F811
    return ["add", "--base-index", str(world["base"]), "--catastro-docs", world["inputs"]["docs"],
            "--own-centroids", world["own"], "--pairs", world["inputs"]["pairs"], "--wfs-clean", world["inputs"]["wfs"],
            "--out-dir", str(out), *extra]


def test_add_cli_builds_the_index_and_uses_the_kept_nomenclature_lots(cli, world, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "idx_fb"
    assert cli.main(add_argv(world, out)) == 0
    meta = meta_of(out)
    assert meta["fallback"]["n_docs_added"] == 7 and meta["fallback"]["lot_basis"] == "docs+units+kept_nomenclature_records"
    assert "7" in capsys.readouterr().out


def test_add_cli_without_the_nomenclature_file_needs_an_explicit_opt_out(cli, world, tmp_path, capsys):  # noqa: F811
    argv = add_argv(world, tmp_path / "a")
    argv[argv.index("--wfs-clean") + 1] = str(tmp_path / "missing.parquet")
    assert cli.main(argv) == 1
    assert "wfs" in capsys.readouterr().err.lower() and not (tmp_path / "a").exists()
    argv = add_argv(world, tmp_path / "b", "--no-kept-lots")
    assert cli.main(argv) == 0
    assert meta_of(tmp_path / "b")["fallback"]["lot_basis"] == "docs+units"


def test_add_cli_with_splits_records_the_oracle_ceiling(cli, world, tmp_path):  # noqa: F811
    from test_build_index import _catastro_entries, _write_splits

    split_dir = _write_splits(world["src"], _catastro_entries(), n_gold=12)
    out = tmp_path / "idx_fb"
    assert cli.main(add_argv(world, out, "--splits-dir", split_dir)) == 0
    assert meta_of(out)["fallback"]["oracle_ceiling_vs_base"]["new_lot_matched_gain"] > 0


def test_remove_cli_restores_the_base_and_reports_errors_with_exit_1(cli, world, tmp_path, capsys):  # noqa: F811
    idx = tmp_path / "idx_fb"
    assert cli.main(add_argv(world, idx)) == 0
    out = tmp_path / "restored"
    assert cli.main(["remove", "--index-dir", str(idx), "--out-dir", str(out)]) == 0
    for p in world["base"].iterdir():
        assert sha(out / p.name) == sha(p), p.name
    assert cli.main(["remove", "--index-dir", str(world["base"]), "--out-dir", str(tmp_path / "x")]) == 1
    assert "fallback" in capsys.readouterr().err
    assert cli.main(["remove", "--index-dir", str(idx), "--out-dir", str(out)]) == 1           # out dir is not empty


def test_experiment_cli_builds_the_experiment(cli, world, base_experiment, tmp_path):  # noqa: F811
    idx = tmp_path / "idx_fb"
    assert cli.main(add_argv(world, idx)) == 0
    exp = tmp_path / "exp_fb"
    assert cli.main(["experiment", "--index-dir", str(idx), "--model-dir", str(base_experiment), "--out-dir", str(exp)]) == 0
    payload = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))
    assert payload["fallback"]["n_docs_added"] == 7 and payload["tag"] == "exp_fb"
    assert cli.main(["experiment", "--index-dir", str(idx), "--model-dir", str(base_experiment), "--out-dir", str(exp)]) == 1


def test_cli_usage_errors(cli, world, tmp_path):  # noqa: F811
    with pytest.raises(SystemExit) as exc:
        cli.main(["add"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:                                   # never write into the default artifacts dir
        cli.main(add_argv(world, paths.DEFAULT_ARTIFACTS_DIR))
    assert exc.value.code == 2
