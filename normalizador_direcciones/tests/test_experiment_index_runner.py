"""Phase 3: --index-dir support, loud early-stop failure and the leaderboard column append.

Everything runs on tiny fixtures; the heavy children of ``scripts/run_experiment.py`` are replaced by a fake
``_run`` so the orchestration itself is what is under test. No real addresses anywhere.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

import cali_address.paths as paths
from cali_address import experiment as ex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Header of the real artifacts/experiments/leaderboard.csv as it was before ``index_id`` existed.
REAL_HEADER_V1 = (
    "tag,timestamp,mode,smoke,final,syn_recall@1,syn_recall@5,syn_recall@20,real_val_recall@1,"
    "real_val_recall@5,real_val_recall@20,dev_ok,dev_n,dev_manzana_precision,dev_predial_precision,"
    "dev_lost_correct,dev_gold_n,dev_gold_ok,dev_gold_ok_coverage,dev_gold_manzana_precision,"
    "dev_gold_predial_precision,dev_gold_lost_correct,seconds,code_fingerprint"
)
REAL_ROW_V1 = (
    "baseline_prod,2026-09-26T13:24:44,eval-only,False,False,,,,,,,1141,2078,0.8431001890359168,"
    "0.6219281663516069,96,289,277,0.9584775086505191,0.9963898916967509,0.9819494584837545,9,10.7,2c69b977f0f4"
)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary(n_ok, n, man, pred, lost):
    return [
        {"dataset": "ALL", "estado": "OK", "n": str(n_ok), "manzana_precision": str(man),
         "predial_precision": str(pred), "lost_correct": "0"},
        {"dataset": "ALL", "estado": "SIN_MATCH", "n": str(n - n_ok), "manzana_precision": "",
         "predial_precision": "", "lost_correct": str(lost)},
    ]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_index(root, name="index_x", n_docs=48, extra_files=None, tiers=("cert", "catastro")):
    """A tiny directory that looks like a built index: files + a matching index_meta.json written last."""
    d = root / name
    d.mkdir(parents=True)
    contents = {"catastro_docs.parquet": b"docs", "train_data.npz": b"tensors", "spatial_negatives.npy": b"neg",
                "train_pairs.parquet": b"pairs", **(extra_files or {})}
    for fname, blob in contents.items():
        (d / fname).write_bytes(blob)
    meta = {"schema": 1, "n_docs": n_docs, "tiers": list(tiers), "pre_normalize_separates_letters": True,
            "files": {f: {"sha256": _sha(b), "bytes": len(b)} for f, b in contents.items()}}
    (d / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


# ---------------------------------------------------------------- leaderboard columns
def test_leaderboard_columns_start_with_the_old_ones_and_keep_index_id_in_place():
    cols = ex.LEADERBOARD_COLUMNS
    assert cols[:24] == REAL_HEADER_V1.split(",")           # existing readers keep their positions
    assert cols[25] == "index_id" and cols.count("index_id") == 1      # later columns are appended after it


def test_leaderboard_row_carries_index_id_and_stopped_early():
    row = ex.leaderboard_row("t", "mixed", [], [], [], 1.0, index_id="index_v2a_cert", stopped_early=True)
    assert row["index_id"] == "index_v2a_cert" and row["stopped_early"] is True
    plain = ex.leaderboard_row("t", "mixed", [], [], [], 1.0)
    assert plain["index_id"] is None and plain["stopped_early"] is None


def _old_leaderboard(tmp_path, rows=2, terminator=b"\r\n"):
    lb = tmp_path / "leaderboard.csv"
    lb.write_bytes(terminator.join([REAL_HEADER_V1.encode()] + [REAL_ROW_V1.encode()] * rows) + terminator)
    return lb


def test_appending_a_row_without_index_id_leaves_an_old_file_header_untouched(tmp_path):
    lb = _old_leaderboard(tmp_path)
    before = lb.read_bytes()
    row = ex.leaderboard_row("plain", "mixed", [], _summary(1, 2, 1.0, 1.0, 0), _summary(1, 1, 1.0, 1.0, 0), 1.0)
    ex.append_leaderboard(str(lb), row)
    after = lb.read_bytes()
    assert after.startswith(before)                                       # every old byte preserved
    assert after.split(b"\r\n")[0].decode() == REAL_HEADER_V1
    new_line = after[len(before):].decode().strip().split(",")
    assert len(new_line) == 24 and new_line[0] == "plain"                  # same width as the file it joins


def test_appending_an_indexed_row_upgrades_only_the_header_line(tmp_path):
    lb = _old_leaderboard(tmp_path)
    old_body = lb.read_bytes().split(b"\r\n", 1)[1]
    row = ex.leaderboard_row("v2", "mixed", [], _summary(1, 2, 1.0, 1.0, 0), _summary(1, 1, 1.0, 1.0, 0), 1.0,
                             index_id="index_v2a_cert", stopped_early=False)
    ex.append_leaderboard(str(lb), row)
    header, body = lb.read_bytes().split(b"\r\n", 1)
    assert header.decode() == ",".join(ex.LEADERBOARD_COLUMNS)
    assert body.startswith(old_body)                                      # old rows byte-for-byte
    rows = list(csv.DictReader(open(lb, newline="", encoding="utf-8")))
    assert [r["tag"] for r in rows] == ["baseline_prod", "baseline_prod", "v2"]
    assert rows[0]["index_id"] is None and rows[2]["index_id"] == "index_v2a_cert"
    assert rows[0]["dev_gold_ok"] == "277"                                # old columns still addressable by name
    for r in csv.reader(open(lb, newline="", encoding="utf-8")):
        assert len(r) <= len(ex.LEADERBOARD_COLUMNS)


def test_old_rows_still_parse_with_pandas_after_the_upgrade(tmp_path):
    lb = _old_leaderboard(tmp_path, rows=1)
    ex.append_leaderboard(str(lb), ex.leaderboard_row("v2", "mixed", [], [], [], 1.0, index_id="ix"))
    df = pd.read_csv(lb)
    assert list(df.columns) == ex.LEADERBOARD_COLUMNS and df.loc[0, "dev_gold_ok"] == 277
    assert pd.isna(df.loc[0, "index_id"]) and df.loc[1, "index_id"] == "ix"


def test_append_handles_lf_files_and_a_missing_final_newline(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    lb.write_bytes(f"{REAL_HEADER_V1}\n{REAL_ROW_V1}".encode())           # LF endings, no trailing newline
    ex.append_leaderboard(str(lb), ex.leaderboard_row("v2", "mixed", [], [], [], 1.0, index_id="ix"))
    rows = list(csv.DictReader(open(lb, newline="", encoding="utf-8")))
    assert [r["tag"] for r in rows] == ["baseline_prod", "v2"]            # the old last row was not glued to the new one
    assert lb.read_bytes().split(b"\n")[1].decode() == REAL_ROW_V1


def test_append_never_rewrites_an_unknown_header(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    lb.write_bytes(b"tag,something_else\r\nold,1\r\n")
    ex.append_leaderboard(str(lb), ex.leaderboard_row("v2", "mixed", [], [], [], 1.0, index_id="ix"))
    assert lb.read_bytes().startswith(b"tag,something_else\r\nold,1\r\n")
    assert [r["tag"] for r in csv.DictReader(open(lb, newline="", encoding="utf-8"))] == ["old", "v2"]


def test_second_indexed_append_does_not_touch_the_new_header(tmp_path):
    lb = _old_leaderboard(tmp_path, rows=1)
    for tag in ("a", "b"):
        ex.append_leaderboard(str(lb), ex.leaderboard_row(tag, "mixed", [], [], [], 1.0, index_id="ix"))
    lines = lb.read_bytes().split(b"\r\n")
    assert lines[0].decode() == ",".join(ex.LEADERBOARD_COLUMNS) and sum(x.startswith(b"tag,") for x in lines) == 1


# ---------------------------------------------------------------- index validation
def test_validate_index_dir_accepts_a_complete_index(tmp_path):
    meta = ex.validate_index_dir(str(_make_index(tmp_path)))
    assert meta["n_docs"] == 48 and set(meta["files"]) >= {"train_data.npz", "catastro_docs.parquet"}


def test_validate_index_dir_rejects_missing_directory_and_missing_marker(tmp_path):
    with pytest.raises(ex.IndexValidationError, match="not a directory"):
        ex.validate_index_dir(str(tmp_path / "nope"))
    (tmp_path / "bare").mkdir()
    with pytest.raises(ex.IndexValidationError, match="index_meta.json"):
        ex.validate_index_dir(str(tmp_path / "bare"))


def test_validate_index_dir_rejects_a_missing_listed_file(tmp_path):
    d = _make_index(tmp_path)
    (d / "train_data.npz").unlink()
    with pytest.raises(ex.IndexValidationError, match="train_data.npz.*missing"):
        ex.validate_index_dir(str(d))


def test_validate_index_dir_rejects_a_sha256_mismatch(tmp_path):
    d = _make_index(tmp_path)
    (d / "catastro_docs.parquet").write_bytes(b"tampered")
    with pytest.raises(ex.IndexValidationError, match="catastro_docs.parquet.*sha256"):
        ex.validate_index_dir(str(d))


@pytest.mark.parametrize("meta_text,match", [
    ("{not json", "unreadable|valid JSON"),
    ("[]", "JSON object"),
    ("{}", "files"),
    ('{"n_docs": 3, "files": {}}', "files"),
    ('{"n_docs": 3, "files": []}', "files"),
    ('{"n_docs": 3, "files": {"catastro_docs.parquet": "abc"}}', "sha256"),
    ('{"n_docs": 3, "files": {"catastro_docs.parquet": {"bytes": 1}}}', "sha256"),
    ('{"n_docs": 3, "files": {"../evil.bin": {"sha256": "00"}}}', "plain file name"),
    ('{"n_docs": 3, "files": {"sub/x.bin": {"sha256": "00"}}}', "plain file name"),
])
def test_validate_index_dir_rejects_malformed_meta(tmp_path, meta_text, match):
    d = _make_index(tmp_path)
    (d / "index_meta.json").write_text(meta_text, encoding="utf-8")
    with pytest.raises(ex.IndexValidationError, match=match):
        ex.validate_index_dir(str(d))


@pytest.mark.parametrize("n_docs", [None, -1, "48", 4.5, True])
def test_validate_index_dir_requires_an_integer_n_docs(tmp_path, n_docs):
    d = _make_index(tmp_path)
    meta = json.loads((d / "index_meta.json").read_text())
    meta["n_docs"] = n_docs
    (d / "index_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ex.IndexValidationError, match="n_docs"):
        ex.validate_index_dir(str(d))


@pytest.mark.parametrize("missing", ["catastro_docs.parquet", "train_data.npz", "spatial_negatives.npy"])
def test_validate_index_dir_requires_the_training_inputs_to_be_listed(tmp_path, missing):
    d = _make_index(tmp_path)
    meta = json.loads((d / "index_meta.json").read_text())
    del meta["files"][missing]
    (d / "index_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ex.IndexValidationError, match=missing):
        ex.validate_index_dir(str(d))


def test_index_meta_record_has_counts_tiers_and_the_meta_sha(tmp_path):
    d = _make_index(tmp_path, n_docs=77, tiers=("cert", "pcer", "catastro"))
    rec = ex.index_meta_record(str(d), ex.validate_index_dir(str(d)))
    assert rec["n_docs"] == 77 and rec["tiers"] == ["cert", "pcer", "catastro"] and rec["index_id"] == "index_x"
    assert rec["sha256"] == _sha((d / "index_meta.json").read_bytes())
    assert rec["pre_normalize_separates_letters"] is True and rec["n_files"] == 4


# ---------------------------------------------------------------- linking
def test_link_index_files_hardlinks_every_listed_file_and_copies_the_meta_last(tmp_path):
    d = _make_index(tmp_path)
    out = tmp_path / "exp"
    out.mkdir()
    meta = ex.validate_index_dir(str(d))
    done = ex.link_index_files(str(d), str(out), meta)
    assert done[-1] == "index_meta.json" and set(done[:-1]) == set(meta["files"])
    for name in meta["files"]:
        assert os.path.samefile(d / name, out / name), name              # one inode, no second copy on disk
    assert (out / "index_meta.json").read_bytes() == (d / "index_meta.json").read_bytes()
    assert not os.path.samefile(d / "index_meta.json", out / "index_meta.json")
    assert paths.is_index_dir(str(out)) and paths.shared_path("train_data.npz", str(out)) == str(out / "train_data.npz")


def test_link_falls_back_to_copy_when_hardlinks_are_not_possible(tmp_path, monkeypatch):
    d = _make_index(tmp_path)
    out = tmp_path / "exp"
    out.mkdir()

    def no_link(*a, **k):
        raise OSError("cross-device link")
    monkeypatch.setattr(os, "link", no_link)
    meta = ex.validate_index_dir(str(d))
    ex.link_index_files(str(d), str(out), meta)
    for name in meta["files"]:
        assert (out / name).read_bytes() == (d / name).read_bytes() and not os.path.samefile(d / name, out / name)


def test_meta_is_written_after_every_other_file(tmp_path, monkeypatch):
    d = _make_index(tmp_path)
    out = tmp_path / "exp"
    out.mkdir()
    seen = []
    real_link, real_copy = os.link, ex.shutil.copy2

    def spy_link(src, dst, *a, **k):
        seen.append(os.path.basename(dst))
        return real_link(src, dst, *a, **k)

    def spy_copy(src, dst, *a, **k):
        seen.append(os.path.basename(dst))
        return real_copy(src, dst, *a, **k)
    monkeypatch.setattr(os, "link", spy_link)
    monkeypatch.setattr(ex.shutil, "copy2", spy_copy)
    ex.link_index_files(str(d), str(out), ex.validate_index_dir(str(d)))
    assert seen[-1] == "index_meta.json" and "index_meta.json" not in seen[:-1]


def test_a_failure_midway_leaves_no_marker_so_the_dir_is_not_mistaken_for_an_index(tmp_path, monkeypatch):
    d = _make_index(tmp_path)
    out = tmp_path / "exp"
    out.mkdir()
    real_link, real_copy = os.link, ex.shutil.copy2
    calls = []

    def flaky(src, dst, *a, **k):
        calls.append(dst)
        if len(calls) == 3:
            raise PermissionError("denied")
        return real_link(src, dst, *a, **k)

    def flaky_copy(src, dst, *a, **k):
        if len(calls) == 3 and os.path.basename(dst) != "index_meta.json":
            raise PermissionError("denied")          # the copy fallback fails as well
        return real_copy(src, dst, *a, **k)
    monkeypatch.setattr(os, "link", flaky)
    monkeypatch.setattr(ex.shutil, "copy2", flaky_copy)
    with pytest.raises(PermissionError):
        ex.link_index_files(str(d), str(out), ex.validate_index_dir(str(d)))
    assert not (out / "index_meta.json").exists()


def test_relinking_into_an_existing_dir_needs_overwrite_and_is_then_idempotent(tmp_path):
    d = _make_index(tmp_path)
    out = tmp_path / "exp"
    out.mkdir()
    meta = ex.validate_index_dir(str(d))
    ex.link_index_files(str(d), str(out), meta)
    with pytest.raises(FileExistsError):
        ex.link_index_files(str(d), str(out), meta)                      # refuses by default
    first = {n: (out / n).read_bytes() for n in list(meta["files"]) + ["index_meta.json"]}
    ex.link_index_files(str(d), str(out), meta, overwrite=True)
    ex.link_index_files(str(d), str(out), meta, overwrite=True)
    assert {n: (out / n).read_bytes() for n in first} == first
    for name in meta["files"]:
        assert os.path.samefile(d / name, out / name)
    assert sorted(os.listdir(out)) == sorted(list(meta["files"]) + ["index_meta.json"])   # nothing extra left behind


def test_overwrite_replaces_a_stale_file_without_writing_through_the_old_link(tmp_path):
    d = _make_index(tmp_path)
    out = tmp_path / "exp"
    out.mkdir()
    (out / "train_data.npz").write_bytes(b"stale content from another index")
    ex.link_index_files(str(d), str(out), ex.validate_index_dir(str(d)), overwrite=True)
    assert (out / "train_data.npz").read_bytes() == b"tensors" and (d / "train_data.npz").read_bytes() == b"tensors"


def test_source_index_is_never_modified_by_linking(tmp_path):
    d = _make_index(tmp_path)
    snapshot = {n: (d / n).read_bytes() for n in os.listdir(d)}
    out = tmp_path / "exp"
    out.mkdir()
    ex.link_index_files(str(d), str(out), ex.validate_index_dir(str(d)))
    assert {n: (d / n).read_bytes() for n in os.listdir(d)} == snapshot and sorted(os.listdir(d)) == sorted(snapshot)


# ---------------------------------------------------------------- max_minutes default for index runs
def test_effective_sets_defaults_max_minutes_to_120_only_for_index_runs():
    assert ex.effective_sets(["lr=1e-3"], None) == ["lr=1e-3"]                         # non-index runs unchanged
    assert ex.effective_sets([], None) == []
    assert ex.effective_sets(["lr=1e-3"], "idx")[-1] == "max_minutes=120"
    assert ex.effective_sets([], "idx") == ["max_minutes=120"]


@pytest.mark.parametrize("given", ["max_minutes=45", "max-minutes=45", " max_minutes = 45 "])
def test_effective_sets_lets_an_explicit_max_minutes_win(given):
    assert ex.effective_sets([given], "idx") == [given]


# ---------------------------------------------------------------- early stop detection
def test_check_training_complete_passes_a_full_run():
    r = ex.check_training_complete({"stopped_early": False, "epochs_configured": 8, "epochs_completed": 8})
    assert r["ok"] is True and r["problems"] == [] and r["epochs_completed"] == 8


def test_check_training_complete_flags_stopped_early():
    r = ex.check_training_complete({"stopped_early": True, "epochs_configured": 8, "epochs_completed": 5})
    assert r["ok"] is False and r["stopped_early"] is True and "5 of 8" in " ".join(r["problems"])


def test_check_training_complete_flags_fewer_epochs_even_if_the_flag_is_missing():
    r = ex.check_training_complete({"history": [{}] * 3}, configured_epochs=8)         # a report from before the flag
    assert r["ok"] is False and r["epochs_completed"] == 3


def test_check_training_complete_accepts_an_old_report_with_all_epochs():
    assert ex.check_training_complete({"history": [{}] * 8}, configured_epochs=8)["ok"] is True


@pytest.mark.parametrize("report", [None, {}, {"history": []}])
def test_check_training_complete_cannot_verify_an_empty_report(report):
    r = ex.check_training_complete(report, configured_epochs=8)
    assert r["ok"] is False and r["problems"]


def test_check_training_complete_flags_a_stop_flag_even_when_epochs_look_complete():
    r = ex.check_training_complete({"stopped_early": True, "epochs_configured": 2, "epochs_completed": 2})
    assert r["ok"] is False


# ---------------------------------------------------------------- the runner, with the heavy children faked
@pytest.fixture
def fake_runner(monkeypatch):
    run = _load_script("run_experiment")
    calls: list[list[str]] = []
    behaviour = {"report": {"stopped_early": False, "epochs_configured": 8, "epochs_completed": 8}, "epochs": 8}

    def fake_run(cmd, log, env=None):
        calls.append(list(cmd))
        out = cmd[cmd.index("--out-dir") + 1] if "--out-dir" in cmd else None
        if os.path.basename(cmd[1]) == "train.py":
            if behaviour["report"] is not None:
                with open(os.path.join(out, "train_report.json"), "w") as fh:
                    json.dump(behaviour["report"], fh)
            with open(os.path.join(out, "experiment.json"), "w") as fh:
                json.dump({"hyperparams": {"epochs": behaviour["epochs"], "mode": "mixed"}, "metrics": {}}, fh)
            with open(os.path.join(out, "catastro_emb.pt"), "wb") as fh:
                fh.write(b"x")
    monkeypatch.setattr(run, "_run", fake_run)
    return run, calls, behaviour


def _argv(tmp_path, tag, index_dir=None, *extra):
    base = ["--tag", tag, "--root", str(tmp_path / "exps"), "--leaderboard", str(tmp_path / "lb.csv"), "--no-hash"]
    return base + (["--index-dir", str(index_dir)] if index_dir else []) + list(extra)


def _train_cmd(calls):
    return next(c for c in calls if os.path.basename(c[1]) == "train.py")


def test_runner_rejects_an_incomplete_index_with_exit_2_and_creates_nothing(fake_runner, tmp_path, capsys):
    run, calls, _ = fake_runner
    d = _make_index(tmp_path)
    (d / "train_data.npz").write_bytes(b"changed after the build")
    rc = run.main(_argv(tmp_path, "t_bad", d))
    err = capsys.readouterr().err
    assert rc == 2 and "sha256" in err and "train_data.npz" in err
    assert not (tmp_path / "exps" / "t_bad").exists() and calls == []
    assert not (tmp_path / "lb.csv").exists()


def test_runner_rejects_a_directory_without_marker_with_exit_2(fake_runner, tmp_path, capsys):
    run, calls, _ = fake_runner
    (tmp_path / "plain").mkdir()
    assert run.main(_argv(tmp_path, "t_plain", tmp_path / "plain")) == 2
    assert "index_meta.json" in capsys.readouterr().err and calls == []


def test_runner_index_run_is_self_contained_and_records_the_index(fake_runner, tmp_path):
    run, calls, _ = fake_runner
    d = _make_index(tmp_path, n_docs=48)
    assert run.main(_argv(tmp_path, "t_ok", d)) == 0
    exp_dir = tmp_path / "exps" / "t_ok"
    for name in ("catastro_docs.parquet", "train_data.npz", "spatial_negatives.npy", "train_pairs.parquet",
                 "index_meta.json"):
        assert (exp_dir / name).exists(), name
    exp = json.load(open(exp_dir / "experiment.json"))
    rec = exp["index_meta"]
    assert rec["n_docs"] == 48 and rec["tiers"] == ["cert", "catastro"] and rec["index_id"] == "index_x"
    assert rec["sha256"] == _sha((d / "index_meta.json").read_bytes())
    assert exp["training_check"]["ok"] is True
    rows = list(csv.DictReader(open(tmp_path / "lb.csv", newline="", encoding="utf-8")))
    assert rows[0]["index_id"] == "index_x" and rows[0]["stopped_early"] == "False"
    assert list(rows[0])[24:26] == ["stopped_early", "index_id"]
    assert "max_minutes=120" in _train_cmd(calls)                                    # index runs get 120 minutes


def test_runner_explicit_max_minutes_wins_and_non_index_runs_keep_the_old_default(fake_runner, tmp_path):
    run, calls, _ = fake_runner
    d = _make_index(tmp_path)
    assert run.main(_argv(tmp_path, "t_exp", d, "--set", "max_minutes=30")) == 0
    cmd = _train_cmd(calls)
    assert "max_minutes=30" in cmd and "max_minutes=120" not in cmd
    calls.clear()
    assert run.main(_argv(tmp_path, "t_plain_run")) == 0
    assert not any(a.startswith("max_minutes") for a in _train_cmd(calls))          # train.py's own default applies


def test_runner_fails_loudly_when_training_stopped_early(fake_runner, tmp_path, capsys):
    run, calls, behaviour = fake_runner
    behaviour["report"] = {"stopped_early": True, "epochs_configured": 8, "epochs_completed": 5}
    d = _make_index(tmp_path)
    rc = run.main(_argv(tmp_path, "t_early", d))
    assert rc == 4 and "stopped early" in capsys.readouterr().err
    exp = json.load(open(tmp_path / "exps" / "t_early" / "experiment.json"))
    assert exp["training_check"]["ok"] is False and exp["training_check"]["epochs_completed"] == 5
    rows = list(csv.DictReader(open(tmp_path / "lb.csv", newline="", encoding="utf-8")))
    assert rows[0]["stopped_early"] == "True" and rows[0]["index_id"] == "index_x"
    assert not any(os.path.basename(c[1]) == "eval_strict.py" for c in calls)         # nothing is scored from a cut run


def test_runner_fails_when_the_train_report_is_missing(fake_runner, tmp_path, capsys):
    run, calls, behaviour = fake_runner
    behaviour["report"] = None
    rc = run.main(_argv(tmp_path, "t_noreport", _make_index(tmp_path)))
    assert rc == 4 and "train_report.json" in capsys.readouterr().err


def test_runner_checks_early_stop_for_runs_without_an_index_too(fake_runner, tmp_path):
    run, calls, behaviour = fake_runner
    behaviour["report"] = {"stopped_early": True, "epochs_configured": 8, "epochs_completed": 1}
    assert run.main(_argv(tmp_path, "t_noindex_early")) == 4


def test_runner_eval_only_never_runs_the_early_stop_check(fake_runner, tmp_path):
    run, calls, behaviour = fake_runner
    src = tmp_path / "src_art"
    src.mkdir()
    assert run.main(_argv(tmp_path, "t_eval", None, "--eval-only", "--from-dir", str(src))) == 0
    rows = list(csv.DictReader(open(tmp_path / "lb.csv", newline="", encoding="utf-8")))
    assert rows[0]["stopped_early"] == "" and rows[0]["index_id"] == ""


def test_runner_rejects_index_dir_together_with_eval_only(fake_runner, tmp_path):
    run, calls, _ = fake_runner
    with pytest.raises(SystemExit):
        run.main(_argv(tmp_path, "t_both", _make_index(tmp_path), "--eval-only", "--from-dir", str(tmp_path)))


def test_runner_rerun_on_the_same_tag_needs_overwrite_and_then_relinks(fake_runner, tmp_path):
    run, calls, _ = fake_runner
    d = _make_index(tmp_path)
    assert run.main(_argv(tmp_path, "t_re", d)) == 0
    with pytest.raises(FileExistsError):
        run.main(_argv(tmp_path, "t_re", d))
    assert run.main(_argv(tmp_path, "t_re", d, "--overwrite")) == 0
    assert os.path.samefile(d / "train_data.npz", tmp_path / "exps" / "t_re" / "train_data.npz")


def test_the_experiment_dir_is_self_contained_for_a_later_eval_only_run(fake_runner, tmp_path):
    """Every index file resolves locally, nothing falls back to production."""
    run, calls, _ = fake_runner
    d = _make_index(tmp_path)
    assert run.main(_argv(tmp_path, "t_selfc", d)) == 0
    exp_dir = str(tmp_path / "exps" / "t_selfc")
    for name in ("catastro_docs.parquet", "train_data.npz", "spatial_negatives.npy"):
        assert paths.shared_path(name, exp_dir) == os.path.join(exp_dir, name)
    assert paths.read_pre_normalize_flag(exp_dir) is True


# ---------------------------------------------------------------- train.py must not write through a hardlink
def test_realistic_noise_rewrite_does_not_modify_the_linked_source_index(tmp_path):
    train = _load_script("train")
    src = tmp_path / "src.npz"
    np.savez_compressed(src, a=np.arange(3))
    linked = tmp_path / "exp" / "train_data.npz"
    linked.parent.mkdir()
    os.link(src, linked)
    before = src.read_bytes()
    train.write_npz_replacing(str(linked), a=np.arange(10))
    assert src.read_bytes() == before                                                  # the index file is intact
    assert np.load(linked)["a"].tolist() == list(range(10)) and not os.path.samefile(src, linked)
