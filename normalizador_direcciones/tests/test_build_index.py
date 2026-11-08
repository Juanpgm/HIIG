"""End-to-end tests of ``scripts/build_index.py`` on tiny invented inputs (numbers 900+, no real data)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from nom_fixtures import catastro_docs_frame, nom_frame, npn, row

import cali_address.paths as paths
from cali_address import inference
from cali_address.index_union import COMPAT_COLUMNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_FILES = {
    "catastro_docs.parquet", "catastro_own_centroids.parquet", "spatial_negatives.npy", "train_data.npz",
    "train_pairs.parquet", "nomenclatura_units.parquet", "index_dropped.parquet",
}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}_idx", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build_index():
    return _load_script("build_index")


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _nom_rows(extra_bad: int = 0, no_coord: int = 0, pcer_only: bool = False):
    rows = []
    for i in range(1, 101):                                       # 100 cert bases
        rows.append(row(i, f"CL {900 + i} # 931 - 912", estado="pcer" if pcer_only else "cert"))
        if i % 5 == 0:                                            # + 20 cert units
            rows.append(row(1000 + i, f"CL {900 + i} # 931 - 912 AP 101", interior="AP 101",
                            estado="pcer" if pcer_only else "cert", npn_value=npn(unit=f"{900000 + i:09d}")))
    for i in range(1, 41):                                        # 40 pcer: 10 repeat a cert address, 30 are new
        text = f"CL {900 + i} # 931 - 912" if i <= 10 else f"KR {900 + i} # 941 - 922"
        rows.append(row(2000 + i, text, estado="pcer", npn_value=npn(lot="0918", unit=f"{i:09d}")))
    rows.append(row(3000, "XYZ 912 QQ"))                          # one fallback record (< 1 %)
    for k in range(extra_bad):
        rows.append(row(4000 + k, f"ZZZ {k} QQ"))
    for k in range(no_coord):
        rows.append(row(5000 + k, f"CL {990 + k} # 931 - 912", lon=None))
    return rows


def _catastro_entries():
    entries = [(f"CL {900 + i} # 931 - 912", npn(lot="0919", unit=f"{i:09d}")) for i in range(1, 6)]
    entries += [(f"KR 26 H 1 # 973 - 9{10 + i}", npn(lot="0920", unit=f"{i:09d}")) for i in range(7)]
    return entries


@pytest.fixture()
def inputs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    return _write_inputs(src, _nom_rows())


def _write_inputs(src, rows, entries=None):
    nom = nom_frame(*rows)
    entries = entries or _catastro_entries()
    cat = catastro_docs_frame(entries)
    pairs = pd.DataFrame([{
        "raw_address": f"raw {i}", "address_key": f"K{i}", "dataset": "fasecolda", "gt_doc_id": i,
        "gt_manzana": "m", "gt_predial": cat["numero_predial_nacional"][i], "gt_comuna": "08", "source_row_id": f"r{i}",
    } for i in range(len(cat))] + [{
        "raw_address": "orphan", "address_key": "KX", "dataset": "fasecolda", "gt_doc_id": 999,
        "gt_manzana": "m", "gt_predial": None, "gt_comuna": "08", "source_row_id": "rx",
    }])
    paths_ = {
        "wfs": str(src / "nomenclatura_clean.parquet"), "docs": str(src / "catastro_docs.parquet"),
        "pairs": str(src / "train_pairs.parquet"), "report": str(src / "cleaning_report.json"),
    }
    nom.to_parquet(paths_["wfs"])
    cat.to_parquet(paths_["docs"])
    pairs.to_parquet(paths_["pairs"])
    (src / "cleaning_report.json").write_text(json.dumps({
        "quarantined_rows": 7, "status_invalid_literals": {"lote": 3, "prov": 1}}), encoding="utf-8")
    return paths_


def _write_splits(src, entries, n_gold=5):
    """dev + frozen_test with ``n_gold`` gold rows pointing at the first old docs, plus a noisy row."""
    cat = catastro_docs_frame(entries)
    split_dir = src / "splits"
    split_dir.mkdir(exist_ok=True)
    for name in ("dev", "frozen_test"):
        frame = pd.DataFrame({
            "gt_doc_id": list(range(n_gold)) + [0], "gt_predial": list(cat["numero_predial_nacional"][:n_gold]) + ["x"],
            "gt_quality": ["gold"] * n_gold + ["noisy"],
        })
        frame.to_parquet(split_dir / f"{name}.parquet")
    return str(split_dir)


def _same_lot_entries():
    """Cadastral docs whose first five lots equal the nomenclature lot (0917)."""
    entries = [(f"CL {900 + i} # 931 - 912", npn(lot="0917", unit=f"{i:09d}")) for i in range(1, 6)]
    entries += [(f"KR 26 H 1 # 973 - 9{10 + i}", npn(lot="0920", unit=f"{i:09d}")) for i in range(7)]
    return entries


def _argv(inp, out_dir, tiers="cert,pcer", *extra):
    return ["--wfs-clean", inp["wfs"], "--catastro-docs", inp["docs"], "--cleaning-report", inp["report"],
            "--pairs", inp["pairs"], "--tiers", tiers, "--out-dir", str(out_dir), "--n-variants", "1", *extra]


def _meta(out_dir):
    return json.loads((out_dir / paths.INDEX_MARKER).read_text(encoding="utf-8"))


# -- happy path ----------------------------------------------------------------------------------------------------

def test_build_writes_a_complete_index_and_the_marker_last(build_index, inputs, tmp_path, capsys):
    out = tmp_path / "index_b"
    assert build_index.main(_argv(inputs, out)) == 0
    assert EXPECTED_FILES <= {p.name for p in out.iterdir()}
    assert paths.is_index_dir(str(out))
    meta = _meta(out)
    latest = max(p.stat().st_mtime_ns for p in out.iterdir() if p.name != paths.INDEX_MARKER)
    assert (out / paths.INDEX_MARKER).stat().st_mtime_ns >= latest
    assert "docs" in capsys.readouterr().out                    # progress is reported
    # every recorded sha256 verifies
    assert set(meta["files"]) == EXPECTED_FILES
    for name, info in meta["files"].items():
        assert _sha(out / name) == info["sha256"], name
    for src in meta["sources"].values():
        assert src["sha256"] == _sha(src["path"])


def test_meta_describes_the_index(build_index, inputs, tmp_path):
    out = tmp_path / "index_meta"
    build_index.main(_argv(inputs, out))
    meta = _meta(out)
    docs = pd.read_parquet(out / "catastro_docs.parquet")
    assert meta["n_docs"] == len(docs)
    assert meta["tiers"] == ["cert", "pcer"]
    assert meta["pre_normalize_separates_letters"] is True
    assert paths.read_pre_normalize_flag(str(out)) is True
    assert meta["code_fingerprint"] == paths.code_fingerprint()
    assert sum(meta["counts_by_fuente"].values()) == len(docs)
    assert set(meta["counts_by_fuente"]) == {"nomenclatura_cert", "nomenclatura_pcer"}
    acc = meta["accounting"]
    assert sum(acc["buckets"].values()) == acc["input"] == 100 + 20 + 40 + 1
    assert acc["buckets"]["dropped"] == 0 and acc["buckets"]["unit"] == 20
    assert acc["buckets"]["fallback_doc"] == 1
    assert meta["parse_failure_rate"] == pytest.approx(1 / 161)
    assert set(meta["npn_choice"]) <= {"base_row", "lowest", "single"}
    assert set(meta["disagreements"]) == {"comuna", "barrio", "manzana", "invalid_npn_fallback"}
    assert meta["crs"]["index_coordinates"].startswith("EPSG:4326")
    assert meta["seconds"] >= 0
    assert meta["pcer"]["rows_with_key_in_cert"] == 10
    assert meta["pcer"]["docs_added"] == 30
    assert meta["catastro"]["coverage"] == {"cadastre_docs": 12, "no_nomenclature_doc_by_key": 7, "no_nomenclature_doc_by_lot": 12}


def test_meta_reports_shared_address_and_lot_conflict_counts(build_index, inputs, tmp_path):
    out = tmp_path / "index_lots"
    build_index.main(_argv(inputs, out))
    docs = pd.read_parquet(out / "catastro_docs.parquet")
    lots = _meta(out)["lots"]
    assert lots == {"docs_n_predios_gt1": int((docs["n_predios"] > 1).sum()),
                    "docs_lot_conflict": int(docs["lot_conflict"].sum())}
    assert lots["docs_lot_conflict"] == 10                      # the 10 pcer rows that repeat a cert address in another lot
    assert docs["lot_conflict"].dtype == bool


def test_cert_only_build_reports_no_lot_conflict(build_index, inputs, tmp_path):
    out = tmp_path / "index_lots_cert"
    build_index.main(_argv(inputs, out, "cert"))
    assert _meta(out)["lots"]["docs_lot_conflict"] == 0


def test_docs_keep_the_production_columns_and_load_in_the_service(build_index, inputs, tmp_path):
    out = tmp_path / "index_docs"
    build_index.main(_argv(inputs, out))
    docs = pd.read_parquet(out / "catastro_docs.parquet")
    prod = pd.read_parquet(inputs["docs"])
    assert list(docs.columns[: len(COMPAT_COLUMNS)]) == COMPAT_COLUMNS
    for col in COMPAT_COLUMNS:
        assert docs[col].dtype == prod[col].dtype, col
    assert docs["doc_id"].tolist() == list(range(len(docs)))
    own = inference.load_own_centroids(docs, str(out))
    assert own is not None and len(own[0]) == len(docs)
    assert docs["objectid"].is_unique and (docs["objectid"] >= 10_000_000).all()


def test_the_index_passes_the_training_n_docs_check(build_index, inputs, tmp_path):
    out = tmp_path / "index_train"
    build_index.main(_argv(inputs, out))
    train = _load_script("train")
    data = np.load(out / "train_data.npz", allow_pickle=True)
    train.check_index_n_docs(len(data["d_tokens"]), paths.read_index_meta(str(out)))
    assert paths.shared_path("train_data.npz", str(out)) == str(out / "train_data.npz")
    assert len(np.load(out / "spatial_negatives.npy")) == len(data["d_tokens"])


def test_official_queries_are_marked_in_q_source_and_reported(build_index, inputs, tmp_path):
    out = tmp_path / "index_q"
    build_index.main(_argv(inputs, out))
    data = np.load(out / "train_data.npz", allow_pickle=True)
    meta = _meta(out)
    n_off = int(data["q_source"].sum())
    assert n_off == meta["queries"]["official"] > 0
    assert meta["queries"]["synthetic"] == len(data["q_source"]) - n_off
    assert (data["q_source"][: meta["queries"]["synthetic"]] == 0).all()
    val, train_ids = set(data["q_doc"][data["q_is_val"]].tolist()), set(data["q_doc"][~data["q_is_val"]].tolist())
    assert not (val & train_ids)


def test_units_and_dropped_tables(build_index, inputs, tmp_path):
    out = tmp_path / "index_units"
    build_index.main(_argv(inputs, out))
    units = pd.read_parquet(out / "nomenclatura_units.parquet")
    docs = pd.read_parquet(out / "catastro_docs.parquet")
    assert list(units.columns) == ["address_key_base", "doc_id", "interior", "numero_predial_nacional", "own_lon",
                                   "own_lat", "objectid", "direccion_original", "estado"]
    assert len(units) == 20 and units["doc_id"].isin(docs["doc_id"]).all()
    dropped = pd.read_parquet(out / "index_dropped.parquet")
    assert dropped.empty and "reason" in dropped.columns


def test_pairs_are_remapped_to_the_new_ids_and_orphans_are_counted(build_index, inputs, tmp_path):
    out = tmp_path / "index_pairs"
    build_index.main(_argv(inputs, out))
    meta, docs = _meta(out), pd.read_parquet(out / "catastro_docs.parquet")
    pairs = pd.read_parquet(out / "train_pairs.parquet")
    assert meta["train_pairs"]["input"] == 13 and meta["train_pairs"]["kept"] == len(pairs) == 5
    assert meta["train_pairs"]["remapped_by_key"] == 5 and meta["train_pairs"]["dropped_no_npn"] == 8     # 7 cadastre-only + orphan
    assert meta["train_pairs"]["agreement_before"]["npn_equal"] == 12 and meta["train_pairs"]["agreement_after"]["npn_equal"] == 0
    assert pairs["gt_doc_id"].between(0, len(docs) - 1).all()
    assert list(pairs.columns) == list(pd.read_parquet(inputs["pairs"]).columns)


def test_tiers_without_pcer_build_a_smaller_index(build_index, inputs, tmp_path):
    a, b = tmp_path / "v2a", tmp_path / "v2b"
    assert build_index.main(_argv(inputs, a, "cert")) == 0
    assert build_index.main(_argv(inputs, b, "cert,pcer")) == 0
    da, db = pd.read_parquet(a / "catastro_docs.parquet"), pd.read_parquet(b / "catastro_docs.parquet")
    assert "nomenclatura_pcer" not in set(da["fuente"])
    assert len(db) == len(da) + 30
    assert _meta(a)["accounting"]["buckets"]["excluded_tier"] == 40
    assert _meta(a)["tiers"] == ["cert"]


def test_no_documents_at_all_is_a_gate_failure_not_a_crash(build_index, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    inp = _write_inputs(src, _nom_rows(pcer_only=True))
    nom = pd.read_parquet(inp["wfs"])
    nom["ndidestado"] = "pcer"
    nom.to_parquet(inp["wfs"])
    out = tmp_path / "idx"
    assert build_index.main(_argv(inp, out, "cert")) == 2          # nothing is selected for tier cert
    assert "no_documents" in json.loads((out / "index_build_failed.json").read_text(encoding="utf-8"))["gates_failed"]
    assert not (out / paths.INDEX_MARKER).exists()


def test_the_catastro_tier_was_removed_and_is_rejected(build_index, inputs, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        build_index.main(_argv(inputs, tmp_path / "t", "cert,pcer,catastro"))
    assert exc.value.code == 2
    assert "catastro" in capsys.readouterr().err
    assert not (tmp_path / "t" / paths.INDEX_MARKER).exists()


def test_the_default_tiers_are_the_nomenclature_ones(build_index):
    assert build_index.ALL_TIERS == ("cert", "pcer")


def test_meta_reports_accounting_with_quarantine_queries_and_flags(build_index, inputs, tmp_path):
    out = tmp_path / "meta2"
    assert build_index.main(_argv(inputs, out)) == 0
    meta = _meta(out)
    acc = meta["accounting"]
    assert acc["quarantined"] == 7 and acc["excluded_lote_prov"] == 4
    assert acc["accounting_total_with_quarantine"] == acc["total"] + 7
    assert {"incomplete_address", "spread_outlier"} <= set(acc["buckets"])
    q = meta["queries"]
    assert q["official_cap_per_doc"] == 12 and 0 < q["official_max_per_doc"] <= 12
    assert q["official_share"] == pytest.approx(q["official"] / (q["official"] + q["synthetic"]))
    assert meta["objectid_unique"] is True and meta["objectid_offset"] == 10_000_000
    assert meta["own_centroids_complete"] is True
    assert set(meta["spread"]) == {"split_groups", "outlier_records", "docs_over_200m", "docs_over_1000m"}
    assert meta["oracle_ceiling"] == {"skipped": "no --splits-dir given"}


def test_incomplete_records_are_quarantined_and_accounted(build_index, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = _nom_rows() + [row(6000, "CL 946 # -"), row(6001, "KR 932B # 946 -")]
    inp = _write_inputs(src, rows)
    out = tmp_path / "inc"
    assert build_index.main(_argv(inp, out)) == 0
    meta = _meta(out)
    assert meta["accounting"]["buckets"]["incomplete_address"] == 2
    dropped = pd.read_parquet(out / "index_dropped.parquet")
    assert dropped["reason"].tolist() == ["incomplete_address"] * 2
    assert meta["accounting"]["total"] == len(rows)


def _check_oracle_consistency(oracle):
    for name, c in oracle.items():
        if name == "gate":
            continue
        assert c["rows"] == c["lot_absent_from_source"] + c["rows_with_lot_in_source"], name
        assert c["new_lot_matched"] <= c["rows_with_lot_in_source"], name


def test_oracle_gate_passes_when_the_lots_survive(build_index, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    entries = _same_lot_entries()
    inp = _write_inputs(src, _nom_rows(), entries)
    out = tmp_path / "oracle_ok"
    assert build_index.main(_argv(inp, out, "cert,pcer", "--splits-dir", _write_splits(src, entries))) == 0
    oracle = _meta(out)["oracle_ceiling"]
    _check_oracle_consistency(oracle)
    assert oracle["gate"]["ok"] is True
    t = oracle["total"]
    assert t["rows"] == 10 and t["prod_lot_matched"] == 10 and t["new_lot_matched"] == 10
    assert t["new_exact"] == 10 and t["lot_absent_from_source"] == 0


def test_oracle_gate_passes_when_the_only_loss_is_lots_absent_from_the_source(build_index, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    inp = _write_inputs(src, _nom_rows())                          # cadastral lot 0919 is in no nomenclature record
    out = tmp_path / "oracle_absent"
    assert build_index.main(_argv(inp, out, "cert,pcer", "--splits-dir", _write_splits(src, _catastro_entries()))) == 0
    oracle = _meta(out)["oracle_ceiling"]
    _check_oracle_consistency(oracle)
    assert oracle["total"]["lot_absent_from_source"] == 10 and oracle["total"]["new_lot_matched"] == 0
    assert oracle["gate"]["ok"] is True
    assert oracle["gate"]["coverage_loss_inherent_pp"] == pytest.approx(100.0)
    assert oracle["gate"]["coverage_loss_headline_pp"] == pytest.approx(100.0)


def test_oracle_gate_fails_with_exit_2_when_a_construction_defect_loses_a_present_lot(build_index, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # the cadastral lot 0919 exists in the source, but only in records quarantined as incomplete_address
    lost = [row(7000 + i, "CL 946 # -", npn_value=npn(lot="0919", unit=f"{i:09d}")) for i in range(1, 4)]
    inp = _write_inputs(src, _nom_rows() + lost)
    out = tmp_path / "oracle_bad"
    assert build_index.main(_argv(inp, out, "cert,pcer", "--splits-dir", _write_splits(src, _catastro_entries()))) == 2
    failed = json.loads((out / "index_build_failed.json").read_text(encoding="utf-8"))
    assert "oracle_ceiling" in failed["gates_failed"]
    t = failed["oracle_ceiling"]["total"]
    assert t["new_lot_matched"] == 0 and t["lot_lost_incomplete"] == 10 and t["lot_absent_from_source"] == 0
    assert not (out / paths.INDEX_MARKER).exists()


def test_oracle_gate_with_missing_split_files_is_an_input_error(build_index, inputs, tmp_path, capsys):
    empty = tmp_path / "nosplits"
    empty.mkdir()
    assert build_index.main(_argv(inputs, tmp_path / "o", "cert,pcer", "--splits-dir", str(empty))) == 1
    assert "dev.parquet" in capsys.readouterr().err


def test_two_runs_give_identical_files(build_index, inputs, tmp_path):
    a, b = tmp_path / "run_a", tmp_path / "run_b"
    build_index.main(_argv(inputs, a))
    build_index.main(_argv(inputs, b))
    for name in sorted(EXPECTED_FILES):
        assert _sha(a / name) == _sha(b / name), name
    ma, mb = _meta(a), _meta(b)
    assert {k: v["sha256"] for k, v in ma["files"].items()} == {k: v["sha256"] for k, v in mb["files"].items()}


# -- guards ----------------------------------------------------------------------------------------------------------

def test_refuses_the_default_artifacts_directory_before_touching_anything(build_index, tmp_path, capsys):
    ghost = {"wfs": "nope.parquet", "docs": "nope.parquet", "report": "nope.json", "pairs": "nope.parquet"}
    for target in (paths.DEFAULT_ARTIFACTS_DIR, os.path.join(paths.DEFAULT_ARTIFACTS_DIR, "."),
                   os.path.join(paths.DEFAULT_ARTIFACTS_DIR, "..", "artifacts")):
        with pytest.raises(SystemExit) as exc:
            build_index.main(_argv(ghost, target, "cert,pcer", "--overwrite"))   # --overwrite must not bypass it
        assert exc.value.code == 2
    assert "artifacts" in capsys.readouterr().err


def test_refuses_a_non_empty_out_dir_without_overwrite(build_index, inputs, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("precious", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        build_index.main(_argv(inputs, out))
    assert exc.value.code == 2
    assert (out / "keep.txt").read_text(encoding="utf-8") == "precious"
    assert not (out / paths.INDEX_MARKER).exists()


def test_an_empty_existing_out_dir_is_fine(build_index, inputs, tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    assert build_index.main(_argv(inputs, out)) == 0


def test_overwrite_rebuilds_and_clears_stale_index_files(build_index, inputs, tmp_path):
    out = tmp_path / "again"
    build_index.main(_argv(inputs, out))
    (out / "plate_negatives_c8_d20.npy").write_bytes(b"stale")
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    assert build_index.main(_argv(inputs, out, "cert,pcer", "--overwrite")) == 0
    assert not (out / "plate_negatives_c8_d20.npy").exists()           # an index file of the old build
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"   # anything else is left alone


def test_crash_midway_leaves_no_completion_marker(build_index, inputs, tmp_path, monkeypatch):
    out = tmp_path / "crash"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(build_index.dataset, "build_training_arrays", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        build_index.main(_argv(inputs, out))
    assert not (out / paths.INDEX_MARKER).exists()
    assert not paths.is_index_dir(str(out))


def test_crash_while_overwriting_a_finished_index_removes_its_marker(build_index, inputs, tmp_path, monkeypatch):
    out = tmp_path / "crash2"
    build_index.main(_argv(inputs, out))
    assert paths.is_index_dir(str(out))
    monkeypatch.setattr(build_index.dataset, "build_spatial_negatives", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        build_index.main(_argv(inputs, out, "cert,pcer", "--overwrite"))
    assert not paths.is_index_dir(str(out))


# -- gates -----------------------------------------------------------------------------------------------------------

def test_gate_parse_failure_rate_at_or_above_one_percent_exits_2(build_index, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    inp = _write_inputs(src, _nom_rows(extra_bad=2))              # 3 unparseable of 163 records = 1.8 %
    out = tmp_path / "gate1"
    assert build_index.main(_argv(inp, out)) == 2
    assert not (out / paths.INDEX_MARKER).exists()
    assert (out / "index_build_failed.json").exists()
    assert "parse_failure_rate" in json.loads((out / "index_build_failed.json").read_text(encoding="utf-8"))["gates_failed"]


def test_gate_dropped_share_above_half_a_percent_exits_2(build_index, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    inp = _write_inputs(src, _nom_rows(no_coord=1))               # 1 of 162 = 0.62 %
    out = tmp_path / "gate2"
    assert build_index.main(_argv(inp, out)) == 2
    failed = json.loads((out / "index_build_failed.json").read_text(encoding="utf-8"))
    assert "dropped_share" in failed["gates_failed"]
    assert not (out / paths.INDEX_MARKER).exists()
    dropped = pd.read_parquet(out / "index_dropped.parquet")
    assert dropped["reason"].tolist() == ["no_coordinates"]


def test_gate_accounting_mismatch_exits_2(build_index, inputs, tmp_path, monkeypatch):
    def broken(n_input, buckets, **kwargs):
        raise build_index.index_union.AccountingError("forced mismatch")

    monkeypatch.setattr(build_index.index_union, "accounting", broken)
    out = tmp_path / "gate3"
    assert build_index.main(_argv(inputs, out)) == 2
    assert not (out / paths.INDEX_MARKER).exists()


def test_missing_input_file_is_a_clear_error_not_a_traceback(build_index, inputs, tmp_path, capsys):
    broken = dict(inputs, wfs=str(tmp_path / "missing.parquet"))
    assert build_index.main(_argv(broken, tmp_path / "x")) == 1
    assert "missing.parquet" in capsys.readouterr().err
    assert not (tmp_path / "x" / paths.INDEX_MARKER).exists()


def test_bad_tiers_are_rejected(build_index, inputs, tmp_path):
    for bad in ("", "foo", "cert,foo", ",,"):
        with pytest.raises(SystemExit) as exc:
            build_index.main(_argv(inputs, tmp_path / "t", bad))
        assert exc.value.code == 2


def test_wfs_frame_missing_a_required_column_is_a_clear_error(build_index, inputs, tmp_path, capsys):
    nom = pd.read_parquet(inputs["wfs"]).drop(columns=["ndnpn"])
    nom.to_parquet(inputs["wfs"])
    assert build_index.main(_argv(inputs, tmp_path / "bad")) == 1
    assert "ndnpn" in capsys.readouterr().err
