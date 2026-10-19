"""The centroid of the returned parcel: build-time sidecar, runtime loader and deploy wiring.

``catastro_docs.parquet`` keeps one row per ADDRESS (median centroid over every parcel that
shares it) while the NPN is the first parcel's. The sidecar ``catastro_own_centroids.parquet``
(one row per document, in document order) carries the first parcel's own centroid so the
deployed image does not need the 300 MB ``catastro.parquet``. Synthetic data only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR = "catastro_own_centroids.parquet"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _docs(npns=("A1", "A2", "A3"), objectids=(10, 11, 12)):
    return pd.DataFrame({
        "direccion": ["CL 981 # 982 - 33", "CL 981 # 982 - 35", "CL 981 # 982 - 37"][: len(npns)],
        "numero_predial_nacional": list(npns),
        "objectid": list(objectids),
        "centroid_lon": [-76.5, -76.51, -76.52][: len(npns)],
        "centroid_lat": [3.4, 3.41, 3.42][: len(npns)],
    })


def _catastro(extra_rows=None):
    cat = pd.DataFrame({
        "OBJECTID": [10, 11, 12, 13], "numero_predial_nacional": ["A1", "A2", "A3", "A4"],
        "centroid_lon": [-76.501, -76.511, np.nan, -76.0], "centroid_lat": [3.401, 3.411, np.nan, 3.0],
    })
    if extra_rows is not None:
        cat = pd.concat([cat, pd.DataFrame(extra_rows)], ignore_index=True)
    return cat


def _sidecar(docs, lon=(-76.501, -76.511, -76.521), lat=(3.401, 3.411, 3.421), **overrides):
    table = pd.DataFrame({
        "objectid": docs["objectid"].to_numpy(),
        "numero_predial_nacional": docs["numero_predial_nacional"].to_numpy(),
        "own_lon": list(lon)[: len(docs)], "own_lat": list(lat)[: len(docs)],
    })
    for column, value in overrides.items():
        table[column] = value
    return table


# ---------------------------------------------------------------------------
# build: dataset.own_centroid_table
# ---------------------------------------------------------------------------
def test_table_is_aligned_with_the_documents_and_carries_the_parcel_point():
    from cali_address.dataset import own_centroid_table
    docs = _docs()
    table = own_centroid_table(docs, _catastro())
    assert list(table.columns) == ["objectid", "numero_predial_nacional", "own_lon", "own_lat"]
    assert table["objectid"].tolist() == [10, 11, 12]
    assert table["numero_predial_nacional"].tolist() == ["A1", "A2", "A3"]
    assert table["own_lon"].iloc[:2].tolist() == [-76.501, -76.511]
    assert table["own_lat"].iloc[:2].tolist() == [3.401, 3.411]
    assert np.isnan(table["own_lon"].iloc[2]) and np.isnan(table["own_lat"].iloc[2])  # parcel without a centroid


def test_table_gives_nan_for_an_objectid_missing_in_the_cadastre_and_for_other_npns():
    from cali_address.dataset import own_centroid_table
    table = own_centroid_table(_docs(objectids=(10, 99, 12), npns=("A1", "A2", "X3")), _catastro())
    assert table["own_lon"].iloc[0] == -76.501
    assert np.isnan(table["own_lon"].iloc[1])                 # OBJECTID 99 is not in the cadastre
    assert np.isnan(table["own_lon"].iloc[2])                 # same OBJECTID, other NPN: never use it


def test_table_survives_duplicate_objectids_in_the_cadastre_keeping_the_first():
    from cali_address.dataset import own_centroid_table
    cat = _catastro([{"OBJECTID": 10, "numero_predial_nacional": "A1", "centroid_lon": -1.0, "centroid_lat": 1.0}])
    table = own_centroid_table(_docs(), cat)
    assert table["own_lon"].iloc[0] == -76.501 and len(table) == 3


def test_table_on_empty_docs_is_empty_with_the_same_columns():
    from cali_address.dataset import own_centroid_table
    table = own_centroid_table(_docs().iloc[0:0], _catastro())
    assert len(table) == 0 and list(table.columns) == ["objectid", "numero_predial_nacional", "own_lon", "own_lat"]


def test_table_with_a_null_npn_document_is_nan():
    from cali_address.dataset import own_centroid_table
    table = own_centroid_table(_docs(npns=(None, "A2", "A3")), _catastro())
    assert np.isnan(table["own_lon"].iloc[0]) and table["own_lon"].iloc[1] == -76.511


# ---------------------------------------------------------------------------
# runtime: inference.load_own_centroids prefers the sidecar, falls back to catastro.parquet
# ---------------------------------------------------------------------------
def test_loader_uses_the_sidecar_when_catastro_parquet_is_absent(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _docs()
    _sidecar(docs).to_parquet(tmp_path / SIDECAR)
    lon, lat = load_own_centroids(docs, str(tmp_path))
    assert lon.tolist() == [-76.501, -76.511, -76.521] and lat.tolist() == [3.401, 3.411, 3.421]


def test_the_sidecar_wins_over_catastro_parquet(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _docs()
    _sidecar(docs, lon=(-1.0, -2.0, -3.0), lat=(1.0, 2.0, 3.0)).to_parquet(tmp_path / SIDECAR)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    lon, _ = load_own_centroids(docs, str(tmp_path))
    assert lon.tolist() == [-1.0, -2.0, -3.0]


@pytest.mark.parametrize("breaker", [
    lambda t, docs: t.iloc[:2],                                         # fewer rows than documents
    lambda t, docs: pd.concat([t, t.iloc[:1]], ignore_index=True),      # more rows than documents
    lambda t, docs: t.assign(objectid=[10, 11, 99]),                    # another cadastre build
    lambda t, docs: t.assign(numero_predial_nacional=["A1", "A2", "ZZ"]),
    lambda t, docs: t.iloc[::-1].reset_index(drop=True),                # rows out of document order
    lambda t, docs: t.drop(columns=["own_lat"]),
])
def test_a_sidecar_that_does_not_match_the_documents_is_ignored(tmp_path, breaker):
    from cali_address.inference import load_own_centroids
    docs = _docs()
    breaker(_sidecar(docs), docs).to_parquet(tmp_path / SIDECAR)
    assert load_own_centroids(docs, str(tmp_path)) is None


def test_a_mismatching_sidecar_falls_back_to_catastro_parquet(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _docs()
    _sidecar(docs).iloc[:2].to_parquet(tmp_path / SIDECAR)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    lon, _ = load_own_centroids(docs, str(tmp_path))
    assert lon[:2].tolist() == [-76.501, -76.511]


def test_a_malformed_sidecar_never_raises_and_falls_back(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _docs()
    (tmp_path / SIDECAR).write_bytes(b"not a parquet file")
    assert load_own_centroids(docs, str(tmp_path)) is None
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    assert load_own_centroids(docs, str(tmp_path)) is not None


def test_an_all_nan_sidecar_is_not_usable(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _docs()
    _sidecar(docs, lon=[np.nan] * 3, lat=[np.nan] * 3).to_parquet(tmp_path / SIDECAR)
    assert load_own_centroids(docs, str(tmp_path)) is None


def test_a_sidecar_keeps_nan_rows_so_the_service_falls_back_per_document(tmp_path):
    from cali_address.inference import load_own_centroids
    docs = _docs()
    _sidecar(docs, lon=(-76.501, np.nan, -76.521), lat=(3.401, np.nan, 3.421)).to_parquet(tmp_path / SIDECAR)
    lon, lat = load_own_centroids(docs, str(tmp_path))
    assert np.isnan(lon[1]) and np.isnan(lat[1]) and lon[2] == -76.521


def test_loader_survives_duplicate_objectids_in_catastro_parquet(tmp_path):
    from cali_address.inference import load_own_centroids
    cat = _catastro([{"OBJECTID": 10, "numero_predial_nacional": "A1", "centroid_lon": -1.0, "centroid_lat": 1.0}])
    cat.to_parquet(tmp_path / "catastro.parquet")
    lon, _ = load_own_centroids(_docs(), str(tmp_path))
    assert lon[0] == -76.501


# ---------------------------------------------------------------------------
# build script: scripts/build_own_centroids.py
# ---------------------------------------------------------------------------
def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_script_writes_a_new_file_next_to_the_docs_and_leaves_them_untouched(tmp_path):
    script = _load_script("build_own_centroids")
    docs = _docs()
    docs.to_parquet(tmp_path / "catastro_docs.parquet", index=False)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    before = _sha(tmp_path / "catastro_docs.parquet")
    assert script.main(["--artifacts-dir", str(tmp_path)]) == 0
    assert _sha(tmp_path / "catastro_docs.parquet") == before
    out = pd.read_parquet(tmp_path / SIDECAR)
    assert out["own_lon"].iloc[:2].tolist() == [-76.501, -76.511] and len(out) == 3
    assert sorted(os.listdir(tmp_path)) == sorted([SIDECAR, "catastro_docs.parquet", "catastro.parquet"])  # no temp file left


def test_script_output_is_what_the_loader_accepts(tmp_path):
    from cali_address.inference import load_own_centroids
    script = _load_script("build_own_centroids")
    docs = _docs()
    docs.to_parquet(tmp_path / "catastro_docs.parquet", index=False)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    script.main(["--artifacts-dir", str(tmp_path)])
    os.remove(tmp_path / "catastro.parquet")
    lon, _ = load_own_centroids(pd.read_parquet(tmp_path / "catastro_docs.parquet"), str(tmp_path))
    assert lon[:2].tolist() == [-76.501, -76.511]


def test_script_refuses_to_overwrite_the_docs_file(tmp_path, capsys):
    script = _load_script("build_own_centroids")
    _docs().to_parquet(tmp_path / "catastro_docs.parquet", index=False)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    before = _sha(tmp_path / "catastro_docs.parquet")
    assert script.main(["--artifacts-dir", str(tmp_path), "--out", str(tmp_path / "catastro_docs.parquet")]) == 2
    assert _sha(tmp_path / "catastro_docs.parquet") == before


@pytest.mark.parametrize("missing", ["catastro.parquet", "catastro_docs.parquet"])
def test_script_reports_a_missing_input_without_writing(tmp_path, capsys, missing):
    script = _load_script("build_own_centroids")
    _docs().to_parquet(tmp_path / "catastro_docs.parquet", index=False)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    os.remove(tmp_path / missing)
    assert script.main(["--artifacts-dir", str(tmp_path)]) == 2
    assert missing in capsys.readouterr().err
    assert not (tmp_path / SIDECAR).exists()


def test_script_refuses_a_catastro_that_is_not_the_docs_source(tmp_path, capsys):
    script = _load_script("build_own_centroids")
    _docs(npns=("X1", "X2", "X3")).to_parquet(tmp_path / "catastro_docs.parquet", index=False)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    assert script.main(["--artifacts-dir", str(tmp_path)]) == 2
    assert not (tmp_path / SIDECAR).exists()


def test_script_docs_without_an_objectid_column_is_an_error(tmp_path):
    script = _load_script("build_own_centroids")
    _docs().drop(columns=["objectid"]).to_parquet(tmp_path / "catastro_docs.parquet", index=False)
    _catastro().to_parquet(tmp_path / "catastro.parquet")
    assert script.main(["--artifacts-dir", str(tmp_path)]) == 2


# ---------------------------------------------------------------------------
# deploy wiring: the image must ship the sidecar and fail the build without it
# ---------------------------------------------------------------------------
def _prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_artifacts_r2", os.path.join(ROOT, "deploy", "prepare_artifacts.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sidecar_is_a_serving_artifact_everywhere():
    from cali_address import paths
    assert SIDECAR in _prepare_module().SERVING_ARTIFACTS
    assert SIDECAR in paths.SERVING_FILES and SIDECAR in paths.SHARED_INPUTS
    assert SIDECAR not in paths.REQUIRED_ARTIFACTS          # optional at runtime: older installs keep working
    assert SIDECAR not in paths.OPTIONAL_ARTIFACTS          # and no warning for an install without it


def test_prepare_artifacts_copies_the_sidecar_and_complains_when_it_is_missing(tmp_path):
    prepare = _prepare_module()
    source, target = tmp_path / "src", tmp_path / "dst"
    source.mkdir()
    for name in prepare.SERVING_ARTIFACTS:
        (source / name).write_bytes(b"x")
    prepare.prepare(str(source), str(target))
    assert (target / SIDECAR).exists()
    os.remove(source / SIDECAR)
    with pytest.raises(SystemExit) as info:
        prepare.prepare(str(source), str(tmp_path / "dst2"))
    assert SIDECAR in str(info.value) and "build_own_centroids" in str(info.value)


def test_the_dockerfile_fails_the_build_without_the_sidecar():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as fh:
        text = fh.read()
    assert f"test -f /app/artifacts/{SIDECAR}" in text


def test_the_training_prep_script_also_writes_the_sidecar():
    with open(os.path.join(ROOT, "scripts", "prepare_training.py"), encoding="utf-8") as fh:
        text = fh.read()
    assert "own_centroid_table" in text and "catastro_own_centroids.parquet" in text


# ---------------------------------------------------------------------------
# the real artifacts (skipped when absent): the sidecar agrees with the cadastre
# ---------------------------------------------------------------------------
@pytest.mark.requires_artifacts
def test_real_sidecar_matches_the_real_documents():
    from cali_address.inference import load_own_centroids
    from cali_address.paths import DEFAULT_ARTIFACTS_DIR
    if not os.path.exists(os.path.join(DEFAULT_ARTIFACTS_DIR, SIDECAR)):
        pytest.skip("sidecar not built: python scripts/build_own_centroids.py")
    docs = pd.read_parquet(os.path.join(DEFAULT_ARTIFACTS_DIR, "catastro_docs.parquet"))
    lon, lat = load_own_centroids(docs, DEFAULT_ARTIFACTS_DIR)
    assert len(lon) == len(docs) == len(lat)
    assert np.isfinite(lon).mean() > 0.9
