"""Phase 1: a directory that declares an index (``index_meta.json``) never falls back to ``artifacts/``.

Without the marker nothing changes (regression). All data is synthetic (tests/tiny.py).
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from tiny import tiny_data_dir, write_pairs

import cali_address.paths as paths
from cali_address import parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Every name the guard protects, with a concrete example for the glob entry.
INDEX_NAMES = [
    "catastro_docs.parquet",
    "catastro_own_centroids.parquet",
    "train_data.npz",
    "spatial_negatives.npy",
    "nomenclatura_units.parquet",
    "plate_negatives_c8_d20.npy",
]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}_guard", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(directory, *names, payload=b"x"):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(payload)


def _declare(directory, meta=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / paths.INDEX_MARKER).write_text(json.dumps({} if meta is None else meta), encoding="utf-8")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake production directory full of decoys plus an empty experiment directory."""
    default, exp = tmp_path / "default", tmp_path / "exp"
    exp.mkdir()
    _touch(default, *INDEX_NAMES, "catastro.parquet", "gazetteer.pkl", "plate_negatives.npy", "other.bin")
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(default))
    monkeypatch.delenv(paths.ENV_ARTIFACTS_DIR, raising=False)
    return default, exp


# ------------------------------------------------------------------ constants

def test_constants_and_error_type():
    assert paths.INDEX_MARKER == "index_meta.json"
    expected = {"catastro_docs.parquet", paths.OWN_CENTROIDS_FILENAME, "train_data.npz", "spatial_negatives.npy",
                "catastro.parquet", "nomenclatura_units.parquet", "plate_negatives*.npy"}
    assert expected <= set(paths.INDEX_FILES)
    assert "gazetteer.pkl" not in paths.INDEX_FILES
    assert issubclass(paths.IndexFileMissingError, FileNotFoundError)
    assert issubclass(paths.IndexMetaError, ValueError)


@pytest.mark.parametrize("name", INDEX_NAMES + ["catastro.parquet", "plate_negatives.npy"])
def test_is_index_file_recognises_every_protected_name(name):
    assert paths.is_index_file(name)


@pytest.mark.parametrize("name", ["gazetteer.pkl", "model.pt", "plate_negatives.txt", "xplate_negatives.npy",
                                  "plate_negative.npy", "", "Plate_Negatives.NPY"])
def test_is_index_file_rejects_everything_else(name):
    assert not paths.is_index_file(name)


# ------------------------------------------------------------------ is_index_dir / read_index_meta

def test_is_index_dir_needs_the_marker_file(world):
    default, exp = world
    assert not paths.is_index_dir(str(exp))
    _declare(exp)
    assert paths.is_index_dir(str(exp))


def test_is_index_dir_false_for_missing_directory_and_marker_that_is_a_directory(world, tmp_path):
    assert not paths.is_index_dir(str(tmp_path / "does_not_exist"))
    fake = tmp_path / "fake"
    (fake / paths.INDEX_MARKER).mkdir(parents=True)
    assert not paths.is_index_dir(str(fake))


def test_is_index_dir_without_argument_looks_at_the_default_directory(world):
    default, _ = world
    assert not paths.is_index_dir()
    _declare(default)
    assert paths.is_index_dir()


def test_read_index_meta_is_none_when_absent(world):
    _, exp = world
    assert paths.read_index_meta(str(exp)) is None
    assert paths.read_index_meta(str(exp / "nope")) is None


def test_read_index_meta_returns_the_dict(world):
    _, exp = world
    _declare(exp, {"n_docs": 7, "note": "ñandú", "pre_normalize_separates_letters": True})
    assert paths.read_index_meta(str(exp)) == {"n_docs": 7, "note": "ñandú", "pre_normalize_separates_letters": True}


def test_marker_present_but_empty_object_is_a_declared_index(world):
    _, exp = world
    _declare(exp, {})
    assert paths.read_index_meta(str(exp)) == {}
    assert paths.is_index_dir(str(exp))
    with pytest.raises(paths.IndexFileMissingError):
        paths.shared_path("train_data.npz", str(exp))


@pytest.mark.parametrize("content", ["{not json", "", "   ", "[]", '"text"', "null", "42", '{"a": '])
def test_read_index_meta_malformed_content_is_a_clear_error(world, content):
    _, exp = world
    (exp / paths.INDEX_MARKER).write_text(content, encoding="utf-8")
    with pytest.raises(paths.IndexMetaError) as err:
        paths.read_index_meta(str(exp))
    assert paths.INDEX_MARKER in str(err.value) and str(exp) in str(err.value)


def test_read_index_meta_rejects_non_utf8_bytes(world):
    _, exp = world
    (exp / paths.INDEX_MARKER).write_bytes(b"\xff\xfe\x00{")
    with pytest.raises(paths.IndexMetaError):
        paths.read_index_meta(str(exp))


def test_malformed_marker_still_counts_as_declared_for_shared_path(world):
    # a broken marker must not silently turn the guard off
    _, exp = world
    (exp / paths.INDEX_MARKER).write_text("{broken", encoding="utf-8")
    assert paths.is_index_dir(str(exp))
    with pytest.raises(paths.IndexFileMissingError):
        paths.shared_path("train_data.npz", str(exp))


# ------------------------------------------------------------------ read_pre_normalize_flag

def test_flag_is_none_without_marker_or_key(world):
    _, exp = world
    assert paths.read_pre_normalize_flag(str(exp)) is None
    _declare(exp, {})
    assert paths.read_pre_normalize_flag(str(exp)) is None


@pytest.mark.parametrize("value", [True, False])
def test_flag_returns_the_declared_boolean(world, value):
    _, exp = world
    _declare(exp, {"pre_normalize_separates_letters": value})
    assert paths.read_pre_normalize_flag(str(exp)) is value


@pytest.mark.parametrize("raw", ['"true"', "1", "0", "null", "[]", '"false"'])
def test_flag_rejects_non_booleans(world, raw):
    _, exp = world
    (exp / paths.INDEX_MARKER).write_text('{"pre_normalize_separates_letters": %s}' % raw, encoding="utf-8")
    with pytest.raises(ValueError, match="pre_normalize_separates_letters"):
        paths.read_pre_normalize_flag(str(exp))


# ------------------------------------------------------------------ shared_path: declared index

@pytest.mark.parametrize("name", INDEX_NAMES)
def test_declared_index_never_returns_the_default_path(world, name):
    default, exp = world
    _declare(exp)
    assert (default / name).exists()                       # the decoy is there, and must be ignored
    with pytest.raises(paths.IndexFileMissingError) as err:
        paths.shared_path(name, str(exp))
    assert name in str(err.value) and str(exp) in str(err.value)
    assert str(default) not in str(err.value).replace(str(exp), "")


@pytest.mark.parametrize("name", INDEX_NAMES)
def test_declared_index_returns_the_local_copy_when_present(world, name):
    _, exp = world
    _declare(exp)
    _touch(exp, name)
    assert paths.shared_path(name, str(exp)) == str(exp / name)


def test_declared_index_raises_even_when_the_default_directory_has_nothing(world, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(tmp_path / "no_default"))
    _, exp = world
    _declare(exp)
    with pytest.raises(FileNotFoundError):
        paths.shared_path("train_data.npz", str(exp))


def test_a_declared_index_missing_only_train_data_raises_for_that_file_only(world):
    default, exp = world
    _declare(exp)
    _touch(exp, "catastro_docs.parquet", "spatial_negatives.npy")
    assert paths.shared_path("catastro_docs.parquet", str(exp)) == str(exp / "catastro_docs.parquet")
    with pytest.raises(paths.IndexFileMissingError):
        paths.shared_path("train_data.npz", str(exp))


def test_catastro_parquet_missing_in_a_declared_index_is_a_nonexistent_local_path(world):
    default, exp = world
    _declare(exp)
    got = paths.shared_path("catastro.parquet", str(exp))
    assert got == str(exp / "catastro.parquet") and not os.path.exists(got)
    assert (default / "catastro.parquet").exists()          # decoy ignored


def test_catastro_parquet_present_in_a_declared_index_is_used(world):
    _, exp = world
    _declare(exp)
    _touch(exp, "catastro.parquet")
    assert paths.shared_path("catastro.parquet", str(exp)) == str(exp / "catastro.parquet")


@pytest.mark.parametrize("name,local", [
    ("plate_negatives_c8_d20.npy", True), ("plate_negatives_c4_d10.npy", False),
    ("plate_negatives.npy", False), ("plate_negatives_.npy", False),
])
def test_plate_negatives_glob_is_guarded(world, name, local):
    default, exp = world
    _declare(exp)
    if local:
        _touch(exp, name)
        assert paths.shared_path(name, str(exp)) == str(exp / name)
    else:
        with pytest.raises(paths.IndexFileMissingError):
            paths.shared_path(name, str(exp))


def test_plate_negatives_with_another_extension_is_not_an_index_file(world):
    default, exp = world
    _declare(exp)
    _touch(default, "plate_negatives_c8_d20.txt")
    assert paths.shared_path("plate_negatives_c8_d20.txt", str(exp)) == str(default / "plate_negatives_c8_d20.txt")


def test_gazetteer_and_other_names_keep_falling_back_with_a_marker(world):
    default, exp = world
    _declare(exp)
    assert paths.shared_path("gazetteer.pkl", str(exp)) == str(default / "gazetteer.pkl")
    assert paths.shared_path("other.bin", str(exp)) == str(default / "other.bin")
    assert paths.shared_path("never_heard_of_it.bin", str(exp)) == str(default / "never_heard_of_it.bin")


def test_a_gazetteer_in_the_experiment_dir_still_wins(world):
    _, exp = world
    _declare(exp)
    _touch(exp, "gazetteer.pkl")
    assert paths.shared_path("gazetteer.pkl", str(exp)) == str(exp / "gazetteer.pkl")


def test_declared_default_directory_raises_instead_of_resolving_to_itself(world):
    default, _ = world
    _declare(default)
    (default / "train_data.npz").unlink()
    with pytest.raises(paths.IndexFileMissingError):
        paths.shared_path("train_data.npz")
    assert paths.shared_path("catastro_docs.parquet") == str(default / "catastro_docs.parquet")


# ------------------------------------------------------------------ shared_path: no marker (regression)

@pytest.mark.parametrize("name", INDEX_NAMES + ["catastro.parquet", "gazetteer.pkl", "unknown.bin"])
def test_without_marker_missing_files_fall_back_exactly_as_before(world, name):
    default, exp = world
    assert paths.shared_path(name, str(exp)) == str(default / name)


def test_without_marker_present_files_win_and_default_is_used_for_none(world):
    default, exp = world
    _touch(exp, "train_data.npz")
    assert paths.shared_path("train_data.npz", str(exp)) == str(exp / "train_data.npz")
    assert paths.shared_path("train_data.npz") == str(default / "train_data.npz")
    assert paths.shared_path("missing_everywhere.npz", str(exp)) == str(default / "missing_everywhere.npz")


def test_without_marker_never_raises_even_if_nothing_exists(world, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(tmp_path / "void"))
    _, exp = world
    assert paths.shared_path("train_data.npz", str(exp)) == str(tmp_path / "void" / "train_data.npz")


# ------------------------------------------------------------------ check_artifacts keeps listing, not raising

def test_check_artifacts_on_a_declared_index_lists_missing_docs_instead_of_raising(world):
    _, exp = world
    _declare(exp)
    missing, _ = paths.check_artifacts(str(exp))
    assert str(exp / "catastro_docs.parquet") in missing
    with pytest.raises(paths.ArtifactsMissingError):
        paths.require_artifacts(str(exp))


# ------------------------------------------------------------------ env var interplay

def test_env_var_pointing_at_a_declared_index_makes_resolved_lookups_strict(world, monkeypatch):
    _, exp = world
    _declare(exp)
    monkeypatch.setenv(paths.ENV_ARTIFACTS_DIR, str(exp))
    resolved = paths.resolve_artifacts_dir()
    assert resolved == str(exp)
    with pytest.raises(paths.IndexFileMissingError):
        paths.shared_path("train_data.npz", resolved)


def test_cli_directory_wins_over_a_declared_env_directory(world, tmp_path, monkeypatch):
    default, exp = world
    _declare(exp)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv(paths.ENV_ARTIFACTS_DIR, str(exp))
    resolved = paths.resolve_artifacts_dir(str(plain))
    assert resolved == str(plain)
    assert paths.shared_path("train_data.npz", resolved) == str(default / "train_data.npz")   # no marker: as before


def test_env_var_does_not_change_the_no_argument_shared_path(world, monkeypatch):
    default, exp = world
    _declare(exp)
    monkeypatch.setenv(paths.ENV_ARTIFACTS_DIR, str(exp))
    assert paths.shared_path("train_data.npz") == str(default / "train_data.npz")


# ------------------------------------------------------------------ inference.load_own_centroids

def _docs3():
    return pd.DataFrame({
        "direccion": ["CL 981 # 982 - 33", "CL 981 # 982 - 35", "CL 981 # 982 - 37"],
        "numero_predial_nacional": ["A1", "A2", "A3"],
        "objectid": [10, 11, 12],
        "centroid_lon": [-76.5, -76.51, -76.52],
        "centroid_lat": [3.4, 3.41, 3.42],
    })


def _decoy_catastro(path):
    pd.DataFrame({
        "OBJECTID": [10, 11, 12], "numero_predial_nacional": ["A1", "A2", "A3"],
        "centroid_lon": [-76.501, -76.511, -76.521], "centroid_lat": [3.401, 3.411, 3.421],
    }).to_parquet(path)


def test_without_marker_the_decoy_catastro_in_default_is_used(world):
    from cali_address.inference import load_own_centroids
    default, exp = world
    _decoy_catastro(default / "catastro.parquet")
    (default / "catastro_own_centroids.parquet").unlink()
    got = load_own_centroids(_docs3(), str(exp))
    assert got is not None and got[0].tolist() == [-76.501, -76.511, -76.521]


def test_declared_index_without_sidecar_returns_none_and_ignores_the_decoy(world):
    from cali_address.inference import load_own_centroids
    default, exp = world
    _decoy_catastro(default / "catastro.parquet")
    (default / "catastro_own_centroids.parquet").unlink()
    _declare(exp)
    assert load_own_centroids(_docs3(), str(exp)) is None


def test_declared_index_ignores_a_decoy_sidecar_in_default_too(world):
    from cali_address.inference import load_own_centroids
    default, exp = world
    docs = _docs3()
    pd.DataFrame({"objectid": docs["objectid"], "numero_predial_nacional": docs["numero_predial_nacional"],
                  "own_lon": [-1.0, -2.0, -3.0], "own_lat": [1.0, 2.0, 3.0]}).to_parquet(
        default / "catastro_own_centroids.parquet")
    _declare(exp)
    assert load_own_centroids(docs, str(exp)) is None


def test_declared_index_uses_its_own_sidecar(world):
    from cali_address.inference import load_own_centroids
    _, exp = world
    docs = _docs3()
    _declare(exp)
    pd.DataFrame({"objectid": docs["objectid"], "numero_predial_nacional": docs["numero_predial_nacional"],
                  "own_lon": [-1.0, -2.0, -3.0], "own_lat": [1.0, 2.0, 3.0]}).to_parquet(
        exp / "catastro_own_centroids.parquet")
    assert load_own_centroids(docs, str(exp))[0].tolist() == [-1.0, -2.0, -3.0]


def test_declared_index_uses_its_own_catastro_parquet(world):
    from cali_address.inference import load_own_centroids
    _, exp = world
    _declare(exp)
    _decoy_catastro(exp / "catastro.parquet")
    assert load_own_centroids(_docs3(), str(exp)) is not None


# ------------------------------------------------------------------ scripts/train.py helpers

def test_resolve_real_pairs_path_without_marker_is_unchanged(world):
    train = _load_script("train")
    _, exp = world
    assert train.resolve_real_pairs_path({"real_pairs": ""}, str(exp)) is None
    assert train.resolve_real_pairs_path({"real_pairs": None}, str(exp)) is None
    assert train.resolve_real_pairs_path({"real_pairs": "some/pairs.parquet"}, str(exp)) == "some/pairs.parquet"


def test_resolve_real_pairs_path_in_a_declared_index_uses_its_train_pairs(world):
    train = _load_script("train")
    _, exp = world
    _declare(exp)
    docs, _, _ = tiny_data_dir(str(exp))
    write_pairs(str(exp / "train_pairs.parquet"), docs)
    assert train.resolve_real_pairs_path({"real_pairs": ""}, str(exp)) == str(exp / "train_pairs.parquet")
    assert train.resolve_real_pairs_path({"real_pairs": None}, str(exp)) == str(exp / "train_pairs.parquet")


def test_resolve_real_pairs_path_explicit_option_wins_in_a_declared_index(world):
    train = _load_script("train")
    _, exp = world
    _declare(exp)
    assert train.resolve_real_pairs_path({"real_pairs": "mine.parquet"}, str(exp)) == "mine.parquet"


def test_resolve_real_pairs_path_missing_file_warns_and_gives_an_empty_set(world, capsys):
    from cali_address.realpairs import load_real_pairs
    train = _load_script("train")
    default, exp = world
    _declare(exp)
    # the production pairs must NOT be picked up as a silent default
    pd.DataFrame({"raw_address": ["CL 981 # 982 - 33"], "gt_doc_id": [0]}).to_parquet(default / "train_pairs.parquet")
    path = train.resolve_real_pairs_path({"real_pairs": ""}, str(exp))
    assert path == str(exp / "train_pairs.parquet") and not os.path.exists(path)
    err = capsys.readouterr()
    assert "train_pairs.parquet" in (err.out + err.err) and "warning" in (err.out + err.err).lower()
    real = load_real_pairs(48, path=path)
    assert real.n_train == 0 and real.n_val == 0


def test_check_index_n_docs_accepts_a_match_and_rejects_everything_else():
    train = _load_script("train")
    train.check_index_n_docs(48, {"n_docs": 48})
    with pytest.raises(ValueError, match="48.*50|50.*48"):
        train.check_index_n_docs(48, {"n_docs": 50})
    for bad in ({}, {"n_docs": None}, {"n_docs": "48"}, {"n_docs": True}, {"n_docs": 48.5}, {"n_docs": -1}):
        with pytest.raises(ValueError, match="n_docs"):
            train.check_index_n_docs(48, bad)
    train.check_index_n_docs(0, {"n_docs": 0})                       # boundary: an empty index is consistent


_SMALL = ["--device", "cpu", "--set", "batch_size=16", "--set", "d_model=16", "--set", "n_layers=1",
          "--set", "n_heads=2", "--set", "d_out=16", "--set", "epochs=1"]


def test_train_in_a_declared_index_without_train_data_raises_and_leaves_default_untouched(world):
    train = _load_script("train")
    default, exp = world
    _declare(exp, {"n_docs": 48})
    before = {p.name: p.read_bytes() for p in default.iterdir()}
    with pytest.raises(paths.IndexFileMissingError):
        train.main(["--out-dir", str(exp)] + _SMALL)
    assert {p.name: p.read_bytes() for p in default.iterdir()} == before
    assert not (exp / "model.pt").exists()


def test_train_in_a_declared_index_uses_local_pairs_and_checks_n_docs(world):
    train = _load_script("train")
    _, exp = world
    docs, _, _ = tiny_data_dir(str(exp))
    write_pairs(str(exp / "train_pairs.parquet"), docs, n=20)
    _declare(exp, {"n_docs": 48})
    assert train.main(["--out-dir", str(exp), "--set", "real_frac=0.5"] + _SMALL) == 0
    stats = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))["dataset_sizes"]["real"]
    assert stats["input"] == 20 and stats["n_train"] + stats["n_val"] > 0


def test_train_in_a_declared_index_without_pairs_trains_with_an_empty_set(world, capsys):
    train = _load_script("train")
    default, exp = world
    tiny_data_dir(str(exp))
    _declare(exp, {"n_docs": 48})
    assert train.main(["--out-dir", str(exp)] + _SMALL) == 0
    stats = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))["dataset_sizes"]["real"]
    assert stats["n_train"] == 0 and stats["input"] == 0
    assert "warning" in (capsys.readouterr().out).lower()


def test_train_refuses_when_n_docs_disagrees_with_the_tensors(world):
    train = _load_script("train")
    _, exp = world
    tiny_data_dir(str(exp))
    _declare(exp, {"n_docs": 49})
    with pytest.raises(ValueError, match="n_docs"):
        train.main(["--out-dir", str(exp)] + _SMALL)
    assert not (exp / "model.pt").exists()


def test_train_without_marker_has_no_n_docs_requirement(world):
    train = _load_script("train")
    default, exp = world
    tiny_data_dir(str(exp))
    # fall back to default pairs file location would be the production one; pass an explicit one
    pairs = write_pairs(str(exp.parent / "pairs.parquet"), pd.read_parquet(exp / "catastro_docs.parquet"), n=10)
    assert train.main(["--out-dir", str(exp), "--real-pairs", pairs] + _SMALL) == 0


def test_train_plate_negatives_are_cached_locally_in_a_declared_index(world):
    train = _load_script("train")
    default, exp = world
    tiny_data_dir(str(exp))
    _declare(exp, {"n_docs": 48})
    np.save(default / "plate_negatives_c8_d20.npy", np.zeros((48, 8), dtype=np.int64))   # decoy cache
    assert train.main(["--out-dir", str(exp), "--set", "plate_neg_weight=0.5"] + _SMALL) == 0
    built = np.load(exp / "plate_negatives_c8_d20.npy")
    assert built.shape[0] == 48 and not (built == 0).all()                  # built here, not the decoy


def _spy_flag(monkeypatch):
    import cali_address.dataset as dataset
    import cali_address.realpairs as realpairs

    seen = {"arrays": [], "pairs": []}
    real_arrays, real_pairs = dataset.build_training_arrays, realpairs.load_real_pairs

    def arrays(*a, **k):
        seen["arrays"].append(parser.PRE_NORMALIZE_SEPARATES_LETTERS)
        return real_arrays(*a, **k)

    def pairs(*a, **k):
        seen["pairs"].append(parser.PRE_NORMALIZE_SEPARATES_LETTERS)
        return real_pairs(*a, **k)

    monkeypatch.setattr(dataset, "build_training_arrays", arrays)
    monkeypatch.setattr(realpairs, "load_real_pairs", pairs)
    return seen


@pytest.mark.parametrize("meta,outer,inside", [
    ({"n_docs": 48, "pre_normalize_separates_letters": True}, False, True),
    ({"n_docs": 48, "pre_normalize_separates_letters": False}, True, False),
    ({"n_docs": 48}, False, False),                       # key absent: the current setting is left alone
    ({"n_docs": 48}, True, True),
])
def test_train_applies_the_artifact_flag_around_data_rebuilds_and_restores_it(world, monkeypatch, meta, outer, inside):
    train = _load_script("train")
    _, exp = world
    docs, _, _ = tiny_data_dir(str(exp))
    write_pairs(str(exp / "train_pairs.parquet"), docs, n=12)
    _declare(exp, meta)
    seen = _spy_flag(monkeypatch)
    with parser.pre_normalize_separation(outer):
        rc = train.main(["--out-dir", str(exp), "--realistic-noise", "--set", "n_variants=2"] + _SMALL)
        assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is outer           # restored after the run
    assert rc == 0
    assert seen["arrays"] == [inside] and seen["pairs"] == [inside]


def test_train_restores_the_flag_when_training_fails(world, monkeypatch):
    train = _load_script("train")
    _, exp = world
    tiny_data_dir(str(exp))
    _declare(exp, {"n_docs": 48, "pre_normalize_separates_letters": True})

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(train, "train_model", boom)
    with pytest.raises(RuntimeError):
        train.main(["--out-dir", str(exp)] + _SMALL)
    assert parser.PRE_NORMALIZE_SEPARATES_LETTERS is False


def test_train_rejects_a_non_boolean_flag_before_doing_any_work(world):
    train = _load_script("train")
    _, exp = world
    tiny_data_dir(str(exp))
    (exp / paths.INDEX_MARKER).write_text('{"n_docs": 48, "pre_normalize_separates_letters": "yes"}', encoding="utf-8")
    with pytest.raises(ValueError, match="pre_normalize_separates_letters"):
        train.main(["--out-dir", str(exp)] + _SMALL)
    assert not (exp / "model.pt").exists()
