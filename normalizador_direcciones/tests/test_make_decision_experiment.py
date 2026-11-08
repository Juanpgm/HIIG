"""``scripts/make_decision_experiment.py``: a candidate experiment dir that serves a source dir with the calibrated layer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import pytest
from test_decision_model import decision_doc, params


def _script():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_make_decision_experiment",
                                                  os.path.join(root, "scripts", "make_decision_experiment.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture
def world(tmp_path):
    src = tmp_path / "src_exp"
    src.mkdir()
    for name, text in (("model.pt", "weights"), ("tuning.json", "{}"), ("catastro_docs.parquet", "docs"),
                       ("catalog_keys_v1.pkl", "keys")):
        (src / name).write_text(text, encoding="utf-8")
    (src / "experiment.json").write_text(json.dumps({"tag": "src_exp", "inherits_model_from": "v2d"}), encoding="utf-8")
    (src / "strict_eval_dev_all_rows.parquet").write_text("old results", encoding="utf-8")
    dec, rel = tmp_path / "decision_x.json", tmp_path / "rel_x.json"
    dec.write_text(json.dumps(decision_doc()), encoding="utf-8")
    rel.write_text(json.dumps(params()), encoding="utf-8")
    return {"src": str(src), "out": str(tmp_path / "out_exp"), "dec": str(dec), "rel": str(rel), "tmp": tmp_path}


def build(world, **kw):
    return _script().build_calibrated_experiment(
        world["src"], world["out"], world["dec"], world["rel"], tag="out_exp", **kw)


def test_links_the_source_files_and_copies_the_decision_pair(world):
    build(world)
    out = world["out"]
    assert sorted(os.listdir(out)) == sorted(
        ["model.pt", "tuning.json", "catastro_docs.parquet", "catalog_keys_v1.pkl", "decision.json",
         "reliability_calibrated.json", "experiment.json"])
    assert sha(os.path.join(out, "decision.json")) == sha(world["dec"])
    assert sha(os.path.join(out, "reliability_calibrated.json")) == sha(world["rel"])
    assert sha(os.path.join(out, "model.pt")) == sha(os.path.join(world["src"], "model.pt"))


def test_big_files_are_hardlinks_when_the_filesystem_allows(world):
    build(world)
    src, dst = os.path.join(world["src"], "model.pt"), os.path.join(world["out"], "model.pt")
    assert os.path.samefile(src, dst) or sha(src) == sha(dst)       # hardlink, or a copy where links are unavailable


def test_results_of_the_source_are_not_carried_over(world):
    build(world)
    assert not [f for f in os.listdir(world["out"]) if f.startswith("strict_eval_")]


def test_experiment_json_records_provenance(world):
    build(world)
    meta = json.loads(open(os.path.join(world["out"], "experiment.json"), encoding="utf-8").read())
    assert meta["tag"] == "out_exp" and meta["decision"] == "calibrated"
    assert meta["inherits_model_from"] == "v2d"                      # the source's own lineage is kept
    assert meta["inherits_from_experiment"] == "src_exp"
    assert meta["decision_provenance"]["decision_sha256"] == sha(world["dec"])
    assert meta["decision_provenance"]["reliability_sha256"] == sha(world["rel"])
    assert meta["decision_provenance"]["decision_version"] == "dec-test-1"
    assert meta["decision_provenance"]["tau_p"] == 0.94


def test_the_source_directory_is_never_modified(world):
    before = {f: sha(os.path.join(world["src"], f)) for f in os.listdir(world["src"])}
    build(world)
    after = {f: sha(os.path.join(world["src"], f)) for f in os.listdir(world["src"])}
    assert before == after


def test_refuses_an_existing_output_directory(world):
    os.makedirs(world["out"])
    with pytest.raises(FileExistsError):
        build(world)


def test_refuses_an_invalid_decision_without_creating_anything(world):
    bad = decision_doc(tau_p=0.5)
    open(world["dec"], "w", encoding="utf-8").write(json.dumps(bad))
    with pytest.raises(ValueError, match="decision"):
        build(world)
    assert not os.path.exists(world["out"])


def test_refuses_an_unusable_reliability_file(world):
    open(world["rel"], "w", encoding="utf-8").write("{")
    with pytest.raises(ValueError, match="reliability"):
        build(world)
    assert not os.path.exists(world["out"])


def test_refuses_a_version_mismatch(world):
    open(world["rel"], "w", encoding="utf-8").write(json.dumps(params(version="another")))
    with pytest.raises(ValueError, match="version"):
        build(world)
    assert not os.path.exists(world["out"])


def test_refuses_a_missing_source(world):
    world["src"] = str(world["tmp"] / "nope")
    with pytest.raises(FileNotFoundError):
        build(world)
