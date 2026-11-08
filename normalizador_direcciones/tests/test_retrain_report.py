"""retrain_report (pure renderers + loaders) and scripts/write_retrain_report.py on tiny fixtures.

Every fixture is invented (900+ numbers); nothing here comes from a real address list. The privacy tests plant
row-level strings in the places a careless renderer could copy them from and assert they never reach the output.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from cali_address import retrain_report as rr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = "baseline_prod_2026_10"
SECTION_COUNT = 12

SECRET_ADDRESS = "CL 947 # 912 - 931 EDIFICIO SECRETO"
SECRET_ID = "760010999999999999999999999999"
SECRET_NAME = "MARIA PEREZ 9876"
SECRET_PATH = "C:\\Users\\Someone\\private\\data.parquet"


def _load_script():
    spec = importlib.util.spec_from_file_location("script_write_retrain_report",
                                                  os.path.join(ROOT, "scripts", "write_retrain_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------------------------------------
def _dump(path, obj) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _write(path, text: str) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _row(gold_ok=277, lost=107, manz=0.8457, lot=0.9881, index_id=None, final=False):
    row = {
        "tag": "x", "dev_n": 2078, "dev_ok": 1144, "dev_manzana_precision": manz, "dev_predial_precision": 0.6228,
        "dev_predial_lot_precision": 0.6747, "dev_lost_correct": lost, "dev_gold_n": 289, "dev_gold_ok": gold_ok,
        "dev_gold_ok_coverage": gold_ok / 289, "dev_gold_manzana_precision": 0.9964,
        "dev_gold_predial_precision": 0.9819, "dev_gold_predial_lot_precision": lot, "dev_gold_lost_correct": 9,
        "seconds": 1200.5, "stopped_early": False, "index_id": index_id,
        "raw_address": SECRET_ADDRESS,                       # a row-level field that must never be rendered
    }
    return row


def _frozen(ok=400, lot=0.9615):
    return {
        "all": {"test_n": 2000, "test_ok": 1002, "test_manzana_precision": 0.7313, "test_predial_precision": 0.4518,
                "test_predial_lot_precision": 0.4911, "test_lost_correct": 223},
        "gold": {"test_gold_n": 417, "test_gold_ok": ok, "test_gold_manzana_precision": 0.9948,
                 "test_gold_predial_precision": 0.923, "test_gold_predial_lot_precision": lot,
                 "test_gold_lost_correct": 12},
    }


def _curve(share=0.96, prec=0.996, err=0.0036, n_emit=250):
    pts = []
    for i in range(10):
        t = round(0.5 + 0.05 * i, 2)
        pts.append({"threshold": t, "n_rows": 289, "n_ok": 278, "n_ok_at_t": 278, "share_of_rows": share,
                    "predial_lot_precision_emitted": prec, "manzana_error_rate": err, "n_emitting_predial": n_emit})
    return {"set": "dev_gold", "n_rows": 289, "column": "confianza", "curve": pts}


def _train_report(done=8, configured=8, stopped=False, seed=42, off=0.93, syn=0.91):
    return {
        "history": [{"epoch": i, "seconds": 100.0} for i in range(done)], "seconds": 1234.5,
        "gpu": {"name": "NVIDIA Test GPU 9000", "capability": "12.0"},
        "config": {"seed": seed, "epochs": configured, "max_minutes": 120.0, "mode": "mixed", "lr": 0.003,
                   "batch_size": 2048},
        "stopped_early": stopped, "epochs_configured": configured, "epochs_completed": done,
        "final_val_recall": {"official": {"n": 2297, "recall@1": off, "recall@5": 0.97, "recall@20": 0.99},
                             "synthetic": {"n": 2310, "recall@1": syn, "recall@5": 0.95, "recall@20": 0.98}},
    }


def _results(spec: dict | None = None) -> list[dict]:
    """12 acceptance results; ``spec`` maps an id to ``pass`` / ``fail`` / ``not_evaluated``."""
    spec = spec or {}
    out = []
    for i in range(1, 13):
        cid = f"A{i}"
        status = spec.get(cid, "pass")
        out.append({"id": cid, "description": f"criterion {cid}", "value": 1.0 if i % 2 else {"a": 1, "b": 2},
                    "threshold": 0.5, "passed": {"pass": True, "fail": False, "not_evaluated": None}[status],
                    "status": status, "reason": f"{cid} reason text" if status != "pass" else "", "parts": []})
    return out


def _make_run(root, tag, *, gold_ok=277, lost=107, lot=0.9881, index_id=None, acceptance=None, frozen=None,
              train=None, validation=None, by_fuente=None, curve=None, probe=None):
    d = root / "artifacts" / "experiments" / tag
    exp = {"leaderboard_row": {**_row(gold_ok, lost, 0.8457, lot, index_id), "tag": tag}, "production_untouched": True,
           "sample_rows": [{"raw_address": SECRET_ADDRESS, "npn": SECRET_ID}]}
    if frozen is not None:
        exp["frozen_test"] = frozen
    if index_id:
        exp["index_meta"] = {"index_id": index_id, "n_docs": 1000, "tiers": ["cert", "pcer"]}
    _dump(d / "experiment.json", exp)
    if train is not None:
        _dump(d / "train_report.json", train)
    if curve is not None:
        _dump(d / "confidence_curve.json", curve)
    if acceptance is not None:
        _dump(d / "acceptance.json", {"results": acceptance, "summary": {"all_passed": False}})
    if probe is not None:
        _dump(d / "probe_agreement.json", probe)
    for name, text in (validation or {}).items():
        _write(d / f"strict_eval_{name}_validation.csv", text)
    for name, text in (by_fuente or {}).items():
        _write(d / f"strict_eval_{name}_by_fuente.csv", text)
    return d


def _index_meta(**over):
    meta = {
        "schema": 2, "n_docs": 394919, "tiers": ["cert", "pcer"],
        "counts_by_fuente": {"nomenclatura_cert": 243468, "nomenclatura_pcer": 151451},
        "counts_by_estado": {"cert": 243468, "pcer": 151451},
        "accounting": {"input": 691263, "total": 691263,
                       "buckets": {"base_doc_new": 365567, "base_doc_merged": 31695, "unit": 282831,
                                   "fallback_doc": 1107, "dropped": 0, "excluded_tier": 0,
                                   "incomplete_address": 10024, "spread_outlier": 39},
                       "quarantined": 1126, "excluded_lote_prov": 817, "dropped_share": 0.0, "ok": True},
        "npn_choice": {"base_row": 24985, "lowest": 7794, "single": 362140},
        "parse_failure_rate": 0.0016, "fallback_doc_share_of_docs": 0.0022,
        "queries": {"synthetic": 1184757, "official": 1120682, "official_share": 0.4861},
        "units_rows": 282831, "objectid_offset": 10000000, "pre_normalize_separates_letters": True,
        "oracle_ceiling": {"gate": {"ok": True, "max_drop_points": 2.0, "failed": [],
                                    "shares": {"total": {"production_cond_pct": 99.71, "new_cond_pct": 97.81}},
                                    "coverage_loss_headline_pp": 3.97, "coverage_loss_inherent_pp": 2.83}},
        "sources": {"wfs_clean": {"path": SECRET_PATH, "sha256": "ab" * 32}},
        "seed": 42,
    }
    meta.update(over)
    return meta


def _make_world(tmp_path, *, acceptance_spec=None, with_acceptance=True):
    """A complete (tiny) artifacts tree. Returns the artifacts dir."""
    root = tmp_path
    art = root / "artifacts"
    _dump(art / "nomenclatura_meta.json", {"rows_downloaded": 692389, "number_matched_last": 692389, "pages": 139,
                                           "estado_counts": {"cert": 437945, "pcer": 253627, "lote": 808, "prov": 9},
                                           "ndfechcort_max": "2026-09-03Z", "status": "ok", "warnings": [],
                                           "downloaded_at": "2026-10-30 20:45:36"})
    _dump(art / "nomenclatura_wfs_vs_shp.json", {
        "wfs_cert_count": 437945, "wfs_pcer_count": 253627, "wfs_rows": 692389, "shp_cert_count": 437825,
        "shp_rows": 437825, "cert_count_diff_share": 0.00027, "cert_count_ok": True, "common_ndid": 319690,
        "npn_agreement": 0.7003, "npn_lot_agreement": 0.9896, "address_agreement": 0.99998,
        "coord_median_shift_m": 0.72, "coord_p99_shift_m": 1.38, "coord_max_shift_m": 19.6,
        "thresholds": {"max_count_diff": 0.005, "min_agreement": 0.995, "min_npn_lot_agreement": 0.98,
                       "max_median_shift_m": 1.0, "max_p99_shift_m": 2.0},
        "passed": True, "wfs_file": SECRET_PATH, "sample_rows": [{"direccion": SECRET_ADDRESS}]})
    _dump(art / "nomenclatura_cleaning_report.json", {
        "input_rows": 692389, "clean_rows": 691263, "quarantined_rows": 1126, "quarantine_share": 0.0016,
        "quarantine_share_limit": 0.005, "gate_passed": True,
        "rules": {"marker_removed": 254528, "rebuilt_from_parts": 101, "empty_address": 305, "status_invalid": 817,
                  "coords_fallback_to_wfs": 7541},
        "status_invalid_literals": {"lote": 808, "prov": 9}, "estado_counts_clean": {"cert": 437941, "pcer": 253322},
        "coord_source_counts": {"local_6249_to_4326": 683722, "wfs": 7541}})
    _dump(root / "train_data" / "nom" / "reprojection_report.json", {
        "count": 437825, "nan_geometry": 0, "outside_bbox": 0, "crs_source": "EPSG:6249", "crs_target": "EPSG:4326",
        "sanity": {"shared_npn": 192180, "median_shift_m": 0.05, "p90_shift_m": 2.8, "max_shift_m": 4872.5},
        "gates": {"bbox_ok": True, "sanity_ok": True}, "source": SECRET_PATH})
    _dump(art / "index_v2c_cert_r2" / "index_meta.json",
          _index_meta(n_docs=300000, tiers=["cert"], counts_by_fuente={"nomenclatura_cert": 300000}))
    _dump(art / "index_v2d_cert_pcer_r2" / "index_meta.json", _index_meta())
    _dump(art / "compare_indexes_index_v2d_cert_pcer_r2.json", {
        "new_index": "index_v2d_cert_pcer_r2",
        "coordinate_shift": {"n_shared_npn": 254769, "quantiles_m": {"p50": 0.046, "p90": 1.5, "p99": 13.0, "max": 9328.2}},
        "gold_npn_coverage": {"old": {"all": {"share_present": 0.995}}, "new": {"all": {"share_present": 0.97}}}})
    # experiments
    _make_run(root, BASE, gold_ok=277, lost=96, lot=0.9819, frozen=_frozen(400, 0.9615), curve=_curve(0.96, 0.996, 0.0036),
              validation={"dev_all_validate": "rule,violations,corrected,mode,rows\nok_without_support,1,0,report,2078\n"})
    acc = _results(acceptance_spec) if with_acceptance else None
    _make_run(root, "v2d_cert_pcer_s42", gold_ok=278, lost=110, lot=0.9930, index_id="index_v2d_cert_pcer_r2",
              frozen=_frozen(401, 0.9650), train=_train_report(), curve=_curve(0.97, 0.997, 0.0030),
              acceptance=acc,
              probe={"n": 706, "counts": {"both_ok_predial_match": 650, "only_old": 5, "only_new": 9, "both_wrong": 42},
                     "by_fuente": {"nomenclatura_cert": {"both_ok_predial_match": 400, "only_old": 2, "only_new": 4,
                                                          "both_wrong": 20}}},
              validation={"dev_all_report": "rule,violations,corrected,mode,rows\nok_without_support,3,0,report,2078\n"
                                            "confidence_out_of_range,1,0,report,2078\n",
                          "dev_all_correct": "rule,violations,corrected,mode,rows\nok_without_support,3,3,correct,2078\n",
                          "dev_gold_clean": "rule,violations,corrected,mode,rows\n"},
              by_fuente={"dev_all": "scope,fuente,n_rows,n_ok,manzana_precision,predial_precision,predial_lot_precision,"
                                    "lost_correct,raw_address\n"
                                    f"all,nomenclatura_cert,1000,600,0.87,0.65,0.70,0,{SECRET_ADDRESS}\n"
                                    "all,nomenclatura_pcer,500,300,0.85,0.60,0.66,0,x\n"
                                    f"all,{SECRET_ADDRESS},5,5,1.0,1.0,1.0,0,y\n"
                                    "all,(none),578,0,,,,90,z\n"})
    _make_run(root, "v2c_cert_s42", gold_ok=276, lost=111, lot=0.9800, index_id="index_v2c_cert_r2",
              train=_train_report(done=6, configured=8, stopped=True, seed=42), curve=_curve())
    _write(art / "experiments" / "leaderboard.csv",
           "tag,dev_gold_ok,dev_lost_correct,seconds\nonly_in_leaderboard,270,120,10.0\n")
    _dump(art / "experiments" / "retrain_2026_10_acceptance.json", {
        "baseline_run": BASE, "baseline_reference": {"dev_gold_ok": 277},
        "noise_band": {"s2b_retrain_s42": {"dev_lost_correct": 107}, "s2b_retrain_s43": {"dev_lost_correct": 112}},
        "criteria": {f"A{i}": f"threshold text {i}" for i in range(1, 13)}, "constants": {"A10_confidence": 0.7}})
    # Jev
    _dump(art / "jev_audit" / "jev_audit.json", {
        "source": "nomenclatura", "privacy": {"origins": {"nomenclatura_public": 300}, "allow_private": False},
        "counts": {"total": 300, "ok": 300, "error": 0, "skipped": 0, "indeterminate": 4, "labeled": 296},
        "agreement": {"n": 296, "agree": 296, "rate": 1.0, "ci95": [0.987, 1.0]}, "kappa": None, "auc": None,
        "brier": None, "confusion": {"tp": 296, "fp": 0, "fn": 0, "tn": 0},
        "gates": {"misma_ubicacion": {"auto": 296, "review": 3, "escalate": 1}},
        "groups": {"estado_nomenclatura": {"cert": {"counts": {"total": 202, "labeled": 202},
                                                    "agreement": {"n": 202, "agree": 202, "rate": 1.0}}}},
        "disagreements": [{"input": SECRET_ADDRESS, "normalized": SECRET_NAME, "npn": SECRET_ID}]})
    return art


def _inputs(art, **over):
    kwargs = dict(artifacts_dir=str(art), baseline_tag=BASE, candidates=["v2d_cert_pcer_s42"], controls=["v2c_cert_s42"],
                  final_tag="v2d_cert_pcer_s42", jev_dirs=[str(art / "jev_audit")])
    kwargs.update(over)
    return rr.load_inputs(**kwargs)


def _render(art, **over) -> str:
    return rr.render_report(_inputs(art, **over), generated_at=over.get("generated_at"))


# ---------------------------------------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------------------------------------
def test_all_twelve_sections_present_in_order(tmp_path):
    text = _render(_make_world(tmp_path))
    positions = []
    for n in range(1, SECTION_COUNT + 1):
        needle = f"\n## {n}. "
        assert needle in text, f"section {n} missing"
        positions.append(text.index(needle))
    assert positions == sorted(positions)
    assert text.startswith("# ")                                # a single document title first


def test_section_titles_follow_the_plan(tmp_path):
    text = _render(_make_world(tmp_path))
    for title in ("Resumen ejecutivo", "Alcance y decisiones", "Perfil de datos", "Composición del índice",
                  "Entrenamiento", "Resultados", "Desglose por fuente", "Validación de salidas", "Auditoría JEV",
                  "Lista de criterios de aceptación", "Riesgos y limitaciones", "Comandos de reproducción"):
        assert title in text, title


def test_full_world_has_no_pending_in_core_sections(tmp_path):
    text = _render(_make_world(tmp_path))
    assert "Estado de promoción: promoción pendiente de aprobación explícita del usuario" in text
    assert "Veredicto: " in text
    assert "NVIDIA Test GPU 9000" in text
    assert "ok_without_support" in text and "confidence_out_of_range" in text


# ---------------------------------------------------------------------------------------------------------
# verdict logic (computed, never hard-coded)
# ---------------------------------------------------------------------------------------------------------
def test_verdict_all_pass():
    v = rr.compute_verdict(_results())
    assert v["verdict"] == "PASS" and v["failed"] == [] and v["not_evaluated"] == []
    assert v["passed"] == [f"A{i}" for i in range(1, 13)]


def test_verdict_one_fail_is_fail():
    v = rr.compute_verdict(_results({"A4": "fail"}))
    assert v["verdict"] == "FAIL" and v["failed"] == ["A4"]


def test_verdict_not_evaluated_is_inconclusive():
    v = rr.compute_verdict(_results({"A9": "not_evaluated", "A12": "not_evaluated"}))
    assert v["verdict"] == "INCONCLUSIVE" and v["not_evaluated"] == ["A9", "A12"]


def test_verdict_fail_outranks_not_evaluated():
    v = rr.compute_verdict(_results({"A2": "fail", "A9": "not_evaluated"}))
    assert v["verdict"] == "FAIL"


@pytest.mark.parametrize("results", [None, [], "garbage", {"results": []}, 7])
def test_verdict_without_usable_results_is_inconclusive(results):
    assert rr.compute_verdict(results)["verdict"] == "INCONCLUSIVE"


def test_verdict_missing_criteria_count_as_not_evaluated():
    v = rr.compute_verdict(_results()[:10])                      # only A1..A10 reported
    assert v["verdict"] == "INCONCLUSIVE" and v["not_evaluated"] == ["A11", "A12"]


def test_verdict_derives_from_status_when_passed_flag_is_missing_or_inconsistent():
    res = _results()
    res[0].pop("passed")                                          # status says pass
    res[1] = {"id": "A2", "status": "fail"}                       # no passed flag at all
    res[2] = {"id": "A3", "passed": True, "status": "fail"}      # contradictory: the failure wins
    v = rr.compute_verdict(res)
    assert v["verdict"] == "FAIL" and v["failed"] == ["A2", "A3"]


def test_verdict_ignores_malformed_entries_and_duplicate_ids_keep_the_worst():
    res = _results() + ["junk", None, {"no": "id"}, {"id": "A5", "status": "fail"}]
    v = rr.compute_verdict(res)
    assert v["verdict"] == "FAIL" and v["failed"] == ["A5"]


@pytest.mark.parametrize("spec,label", [({}, "APRUEBA"), ({"A3": "fail"}, "NO APRUEBA"),
                                        ({"A9": "not_evaluated"}, "INCONCLUSO")])
def test_report_renders_the_computed_verdict(tmp_path, spec, label):
    text = _render(_make_world(tmp_path, acceptance_spec=spec))
    assert f"Veredicto: {label}" in text
    for other in {"APRUEBA", "NO APRUEBA", "INCONCLUSO"} - {label}:
        assert f"Veredicto: {other}" not in text


def test_report_without_acceptance_file_is_inconclusive_and_says_why(tmp_path):
    text = _render(_make_world(tmp_path, with_acceptance=False))
    assert "Veredicto: INCONCLUSO" in text
    assert "acceptance.json" in text and "n/d (pendiente)" in text


def test_checklist_lists_a1_to_a12_with_statuses(tmp_path):
    text = _render(_make_world(tmp_path, acceptance_spec={"A4": "fail", "A9": "not_evaluated"}))
    sec = text.split("\n## 10. ")[1].split("\n## 11. ")[0]
    for i in range(1, 13):
        assert f"| A{i} |" in sec
    assert "| A4 | NO APRUEBA |" in sec
    assert "| A9 | NO EVALUADO |" in sec
    assert "| A1 | APRUEBA |" in sec
    assert "A4 reason text" in sec


# ---------------------------------------------------------------------------------------------------------
# formatting helpers and deltas
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("new,base,kind,expected", [
    (278, 277, "count", "+1"), (276, 277, "count", "-1"), (277, 277, "count", "0"),
    (0.9930, 0.9881, "rate", "+0.0049"), (0.9800, 0.9881, "rate", "-0.0081"), (0.5, 0.5, "rate", "0.0000"),
])
def test_fmt_delta_signs(new, base, kind, expected):
    assert rr.fmt_delta(new, base, kind) == expected


@pytest.mark.parametrize("new,base", [(None, 1), (1, None), (float("nan"), 1), (1, float("nan")), ("x", 1), (True, 1)])
def test_fmt_delta_unknown_is_pending(new, base):
    assert rr.fmt_delta(new, base, "count") == rr.PENDING


@pytest.mark.parametrize("value,expected", [(None, rr.PENDING), (float("nan"), rr.PENDING), ("", rr.PENDING),
                                            (float("inf"), rr.PENDING), (0.98194, "0.9819"), (12, "12"),
                                            ("0.5", "0.5000")])
def test_fmt_rate_handles_unusable_values(value, expected):
    assert rr.fmt_rate(value) == expected


def test_results_table_has_signed_deltas_against_the_baseline(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 6. ")[1].split("\n## 7. ")[0]
    gold_ok_line = next(l for l in sec.splitlines() if l.startswith("| dev gold OK "))
    assert gold_ok_line.rstrip().endswith("| +1 |")                 # 278 - 277
    lost_line = next(l for l in sec.splitlines() if l.startswith("| dev lost_correct "))
    assert lost_line.rstrip().endswith("| +14 |")                   # 110 - 96
    lot_line = next(l for l in sec.splitlines() if l.startswith("| dev gold precisión predial de lote "))
    assert lot_line.rstrip().endswith("| +0.0111 |")                # 0.9930 - 0.9819


def test_results_table_marks_frozen_as_baseline_only_for_unscored_runs(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 6. ")[1].split("\n## 7. ")[0]
    line = next(l for l in sec.splitlines() if l.startswith("| congelada gold OK "))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1] == "400" and cells[-1] == "+1"
    assert rr.PENDING in cells                                      # the control run was never scored on frozen


def test_confidence_curve_at_070_is_reported(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 6. ")[1].split("\n## 7. ")[0]
    assert "0.70" in sec and "0.9970" in sec and "0.9960" in sec
    # the full curve table covers 0.50 .. 0.95
    assert "| 0.50 |" in sec and "| 0.95 |" in sec


def test_missing_confidence_curve_is_pending_not_an_error(tmp_path):
    art = _make_world(tmp_path)
    os.remove(art / "experiments" / "v2d_cert_pcer_s42" / "confidence_curve.json")
    text = _render(art)
    assert "A10" in text and rr.PENDING in text


# ---------------------------------------------------------------------------------------------------------
# per-section content
# ---------------------------------------------------------------------------------------------------------
def test_training_section_flags_incomplete_and_early_stop(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 5. ")[1].split("\n## 6. ")[0]
    ok_line = next(l for l in sec.splitlines() if l.startswith("| v2d_cert_pcer_s42 "))
    assert "8/8" in ok_line and "0.9300" in ok_line and "0.9100" in ok_line
    bad_line = next(l for l in sec.splitlines() if l.startswith("| v2c_cert_s42 "))
    assert "6/8" in bad_line and "INCOMPLETO" in bad_line and "parada temprana" in bad_line.lower()
    assert "no es comparable" in sec.lower()                         # synthetic recall caveat


def test_index_section_has_accounting_oracle_and_loss_headline(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 4. ")[1].split("\n## 5. ")[0]
    for needle in ("index_v2d_cert_pcer_r2", "index_v2c_cert_r2", "394919", "base_doc_new", "365567", "incomplete_address",
                   "3.97", "2.83", "0.4861", "151451"):
        assert needle in sec, needle


def test_data_profile_has_cleaning_rules_gates_and_reprojection(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 3. ")[1].split("\n## 4. ")[0]
    for needle in ("692389", "437825", "marker_removed", "254528", "status_invalid", "0.9896", "EPSG:6249", "mediano"):
        assert needle in sec, needle
    assert "98" in sec and "p99" in sec                              # the amended gates are stated


def test_by_source_section_lists_cert_and_pcer_and_sanitizes_labels(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 7. ")[1].split("\n## 8. ")[0]
    assert "nomenclatura_cert" in sec and "nomenclatura_pcer" in sec and "(none)" in sec
    assert "both_ok_predial_match" in sec


def test_validation_section_shows_modes_and_clean_runs(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 8. ")[1].split("\n## 9. ")[0]
    assert "report" in sec and "correct" in sec
    assert "ok_without_support" in sec
    assert "sin violaciones" in sec.lower()                          # the header-only (clean) file


def test_jev_section_states_the_caveat_when_kappa_is_undefined(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 9. ")[1].split("\n## 10. ")[0]
    assert "296" in sec and "kappa" in sec.lower()
    assert "indefinid" in sec.lower()
    assert "decisivo" in sec.lower()


def test_jev_section_pending_without_audit(tmp_path):
    text = _render(_make_world(tmp_path), jev_dirs=[str(tmp_path / "nope")])
    sec = text.split("\n## 9. ")[1].split("\n## 10. ")[0]
    assert "pendiente" in sec.lower()


def test_scope_section_states_dated_decisions_and_per_artifact_flag(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 2. ")[1].split("\n## 3. ")[0]
    assert "2026-10-31" in sec and "fuentes nuevas" in sec.lower()
    assert "lote" in sec.lower() and "pre_normalize_separates_letters" in sec


def test_risks_section_mentions_the_standing_limitations(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 11. ")[1].split("\n## 12. ")[0]
    for needle in ("cobertura", "catastro", "ruido", "sintétic", "congelad"):
        assert needle in sec.lower(), needle


def test_reproduction_commands_use_the_tags_given(tmp_path):
    text = _render(_make_world(tmp_path))
    sec = text.split("\n## 12. ")[1]
    assert "write_retrain_report.py" in sec and "v2d_cert_pcer_s42" in sec and BASE in sec


# ---------------------------------------------------------------------------------------------------------
# graceful degradation
# ---------------------------------------------------------------------------------------------------------
def test_empty_artifacts_dir_renders_everything_as_pending(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    text = rr.render_report(rr.load_inputs(artifacts_dir=str(art), baseline_tag=BASE, candidates=["c1"], controls=[],
                                           final_tag=None, jev_dirs=[]))
    for n in range(1, SECTION_COUNT + 1):
        assert f"\n## {n}. " in text
    assert "Veredicto: INCONCLUSO" in text
    assert "pendiente de aprobación explícita del usuario" in text
    assert text.count(rr.PENDING) >= 10


def test_nonexistent_artifacts_dir_does_not_raise(tmp_path):
    inputs = rr.load_inputs(artifacts_dir=str(tmp_path / "missing"), baseline_tag=BASE, candidates=[], controls=[],
                            final_tag=None, jev_dirs=[])
    assert "## 1. " in rr.render_report(inputs)


@pytest.mark.parametrize("victim", ["nomenclatura_meta.json", "nomenclatura_cleaning_report.json",
                                    "nomenclatura_wfs_vs_shp.json", "index_v2d_cert_pcer_r2/index_meta.json",
                                    "experiments/v2d_cert_pcer_s42/experiment.json",
                                    "experiments/v2d_cert_pcer_s42/train_report.json",
                                    "experiments/v2d_cert_pcer_s42/acceptance.json",
                                    "experiments/retrain_2026_10_acceptance.json", "jev_audit/jev_audit.json"])
def test_truncated_json_degrades_to_pending(tmp_path, victim):
    art = _make_world(tmp_path)
    _write(art / victim, '{"truncated": [1, 2,')
    text = _render(art)
    for n in range(1, SECTION_COUNT + 1):
        assert f"\n## {n}. " in text


@pytest.mark.parametrize("payload", ["[]", "null", "42", '"text"', "{}"])
def test_wrong_json_shapes_do_not_raise(tmp_path, payload):
    art = _make_world(tmp_path)
    for rel in ("nomenclatura_meta.json", "nomenclatura_cleaning_report.json", "index_v2d_cert_pcer_r2/index_meta.json",
                "experiments/v2d_cert_pcer_s42/experiment.json", "experiments/v2d_cert_pcer_s42/train_report.json",
                "experiments/v2d_cert_pcer_s42/confidence_curve.json", "jev_audit/jev_audit.json"):
        _write(art / rel, payload)
    text = _render(art)
    assert "\n## 12. " in text


def test_garbage_values_inside_valid_json_do_not_raise(tmp_path):
    art = _make_world(tmp_path)
    _dump(art / "experiments" / "v2d_cert_pcer_s42" / "train_report.json",
          {"epochs_completed": "many", "epochs_configured": None, "seconds": "slow", "gpu": "none",
           "final_val_recall": {"official": "x", "synthetic": {"recall@1": float("nan")}}, "config": []})
    _dump(art / "index_v2d_cert_pcer_r2" / "index_meta.json",
          {"n_docs": "lots", "accounting": {"buckets": {"a": "x", "b": None}}, "oracle_ceiling": {"gate": "bad"},
           "tiers": "cert"})
    text = _render(art)
    assert "\n## 5. " in text and "\n## 4. " in text


def test_empty_and_header_only_csv_files_are_tolerated(tmp_path):
    art = _make_world(tmp_path)
    run = art / "experiments" / "v2d_cert_pcer_s42"
    _write(run / "strict_eval_dev_all_by_fuente.csv", "")
    _write(run / "strict_eval_dev_all_correct_validation.csv", "rule,violations,corrected,mode,rows\n")
    _write(run / "strict_eval_dev_all_report_validation.csv", "not,a,validation\n1,2,3\n")
    assert "\n## 8. " in _render(art)


def test_leaderboard_row_is_the_fallback_when_experiment_json_is_absent(tmp_path):
    art = _make_world(tmp_path)
    os.remove(art / "experiments" / "v2c_cert_s42" / "experiment.json")
    _write(art / "experiments" / "leaderboard.csv",
           "tag,dev_gold_ok,dev_lost_correct,seconds\nv2c_cert_s42,270,120,10.0\nv2c_cert_s42,271,119,11.0\n")
    inputs = _inputs(art)
    assert inputs["runs"]["v2c_cert_s42"]["metrics"]["dev_gold_ok"] == 271          # last row of the tag wins
    assert inputs["runs"]["v2c_cert_s42"]["metrics"]["dev_lost_correct"] == 119


def test_unknown_candidate_tag_is_pending_not_an_error(tmp_path):
    art = _make_world(tmp_path)
    text = _render(art, candidates=["v2d_cert_pcer_s42", "tag_that_does_not_exist"])
    assert "tag_that_does_not_exist" in text and rr.PENDING in text


def test_division_free_rendering_with_zero_denominators(tmp_path):
    art = _make_world(tmp_path)
    _dump(art / "experiments" / "v2d_cert_pcer_s42" / "train_report.json",
          {"epochs_configured": 0, "epochs_completed": 0, "seconds": 0, "history": []})
    assert "\n## 5. " in _render(art)


# ---------------------------------------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("secret", [SECRET_ADDRESS, SECRET_ID, SECRET_NAME, SECRET_PATH, "Someone", "EDIFICIO SECRETO"])
def test_no_row_level_strings_reach_the_output(tmp_path, secret):
    text = _render(_make_world(tmp_path))
    assert secret not in text


def test_label_that_looks_like_an_address_is_replaced(tmp_path):
    text = _render(_make_world(tmp_path))
    assert "(otra etiqueta)" in text                                # the address-like fuente label was masked


def test_unlisted_fields_are_never_copied(tmp_path):
    art = _make_world(tmp_path)
    _dump(art / "nomenclatura_meta.json", {"rows_downloaded": 5, "free_text": SECRET_ADDRESS, "sample": [SECRET_NAME]})
    text = _render(art)
    assert SECRET_ADDRESS not in text and SECRET_NAME not in text


def test_acceptance_reasons_are_sanitized_and_bounded(tmp_path):
    art = _make_world(tmp_path)
    res = _results({"A1": "fail"})
    res[0]["reason"] = "x" * 5000 + "\n| injected | row |"
    _dump(art / "experiments" / "v2d_cert_pcer_s42" / "acceptance.json", {"results": res})
    text = _render(art)
    sec = text.split("\n## 10. ")[1].split("\n## 11. ")[0]
    assert "x" * 400 not in sec
    assert "\n| injected |" not in sec


# ---------------------------------------------------------------------------------------------------------
# determinism and writing
# ---------------------------------------------------------------------------------------------------------
def test_same_inputs_give_identical_text(tmp_path):
    art = _make_world(tmp_path)
    assert _render(art) == _render(art)


def test_no_timestamp_unless_generated_at_is_given(tmp_path):
    art = _make_world(tmp_path)
    plain = _render(art)
    assert "Generado el" not in plain
    stamped = rr.render_report(_inputs(art), generated_at="2026-11-01")
    assert "Generado el: 2026-11-01" in stamped
    assert stamped.count("2026-11-01") == 1
    assert stamped.replace("Generado el: 2026-11-01\n", "") == plain


def test_output_does_not_depend_on_tag_order_of_dict_internals(tmp_path):
    art = _make_world(tmp_path)
    a = _render(art, controls=["v2c_cert_s42"], candidates=["v2d_cert_pcer_s42"])
    b = _render(art, controls=["v2c_cert_s42"], candidates=["v2d_cert_pcer_s42"])
    assert a == b


def test_write_if_changed_is_idempotent(tmp_path):
    out = tmp_path / "docs" / "report.md"
    assert rr.write_if_changed(str(out), "hello\n") is True
    mtime = os.stat(out).st_mtime_ns
    assert rr.write_if_changed(str(out), "hello\n") is False
    assert os.stat(out).st_mtime_ns == mtime
    assert rr.write_if_changed(str(out), "hello again\n") is True
    with open(out, encoding="utf-8", newline="") as fh:
        assert fh.read() == "hello again\n"


def test_output_is_lf_only_and_ends_with_one_newline(tmp_path):
    text = _render(_make_world(tmp_path))
    assert "\r" not in text and text.endswith("\n") and not text.endswith("\n\n")


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------
def _cli(art, out, *extra):
    mod = _load_script()
    argv = ["--artifacts-dir", str(art), "--out", str(out), "--baseline-tag", BASE, "--candidates", "v2d_cert_pcer_s42",
            "--control", "v2c_cert_s42", "--final-tag", "v2d_cert_pcer_s42", "--jev-dir", str(art / "jev_audit"), *extra]
    return mod.main(argv)


def test_cli_writes_the_report_and_exits_zero(tmp_path):
    art = _make_world(tmp_path)
    out = tmp_path / "docs" / "r.md"
    assert _cli(art, out) == 0
    text = out.read_text(encoding="utf-8")
    assert "\n## 12. " in text and "Veredicto: " in text


def test_cli_is_idempotent_on_disk(tmp_path):
    art = _make_world(tmp_path)
    out = tmp_path / "r.md"
    assert _cli(art, out, "--generated-at", "2026-11-01") == 0
    first = (out.read_bytes(), os.stat(out).st_mtime_ns)
    assert _cli(art, out, "--generated-at", "2026-11-01") == 0
    assert (out.read_bytes(), os.stat(out).st_mtime_ns) == first


def test_cli_exit_zero_on_empty_artifacts(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    assert _cli(art, tmp_path / "r.md") == 0                          # everything pending, still a valid report


def test_cli_missing_out_is_usage_error(tmp_path):
    assert _load_script().main(["--baseline-tag", BASE]) == 2


def test_cli_unknown_flag_is_usage_error(tmp_path):
    assert _load_script().main(["--out", str(tmp_path / "r.md"), "--bogus"]) == 2


def test_cli_out_that_is_a_directory_is_usage_error(tmp_path):
    art = _make_world(tmp_path)
    d = tmp_path / "somedir"
    d.mkdir()
    assert _cli(art, d) == 2


def test_cli_final_tag_outside_the_runs_is_usage_error(tmp_path):
    art = _make_world(tmp_path)
    mod = _load_script()
    argv = ["--artifacts-dir", str(art), "--out", str(tmp_path / "r.md"), "--baseline-tag", BASE,
            "--candidates", "v2d_cert_pcer_s42", "--final-tag", "somebody_else"]
    assert mod.main(argv) == 2
    assert not (tmp_path / "r.md").exists()


def test_cli_refuses_to_write_inside_an_experiment_or_index_directory(tmp_path):
    art = _make_world(tmp_path)
    assert _cli(art, art / "experiments" / "v2d_cert_pcer_s42" / "r.md") == 2
    assert _cli(art, art / "index_v2d_cert_pcer_r2" / "r.md") == 2


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        _load_script().main(["--help"])
    assert exc.value.code == 0


def test_script_never_imports_training_stacks():
    src = open(os.path.join(ROOT, "scripts", "write_retrain_report.py"), encoding="utf-8").read()
    mod_src = open(os.path.join(ROOT, "src", "cali_address", "retrain_report.py"), encoding="utf-8").read()
    for text in (src, mod_src):
        assert "import torch" not in text and "from cali_address import train" not in text


# =========================================================================================================
# final-report sections 13-21 (story, noise band, bands, calibration, JEV, ONNX, reading, decisions, promotion)
# =========================================================================================================
FINAL = "v2d_cert_pcer_s42"
NEW_SECTIONS = {13: "Historia en tres etapas", 14: "Banda de ruido", 15: "Precisión por banda", 16: "Calidad de calibración",
                17: "Auditoría JEV: límites", 18: "Rendimiento", 19: "Lectura de A1-A12", 20: "Decisiones abiertas",
                21: "Promoción y reversibilidad"}


def _wilson_row(label_key, label, n=224, ok=223, scored=224, lo=0.9751, hi=0.9992):
    return {label_key: label, "n": n, "lot_correct": ok, "lot_scored": scored,
            "lot_precision": (ok / scored if scored else None), "lot_lo": lo if scored else None,
            "lot_hi": hi if scored else None, "manzana_correct": 222, "manzana_scored": 224, "manzana_precision": 0.9911,
            "manzana_lo": 0.968, "manzana_hi": 0.9975, "raw_address": SECRET_ADDRESS}


def _side(n=312, ok=311, lost=14, ok_rows=381, lot_missing=30):
    return {"n": 417, "ok": ok_rows, "rows_emitting_predial": n, "lot_emitted_scored": n, "lot_correct": ok,
            "manzana_scored": 375, "manzana_correct": 369, "lost_correct": lost, "lot_precision": ok / n,
            "manzana_precision": 0.984, "ok_share": ok_rows / 417, "lot_missing_from_index": lot_missing}


def _bins(pred=0.97, obs=0.98):
    return {"n": 250, "mean_predicted": pred, "observed": obs,
            "bins": [{"bin": i, "n": 25, "predicted": pred, "observed": obs} for i in range(10)]}


def _summary() -> dict:
    sets = ("dev_gold", "dev_all", "frozen_gold", "frozen_all")
    return {
        "frozen_second_look": True, "frozen_second_look_note": "free text " + SECRET_ADDRESS,
        "candidate": "cand_basecal", "baseline": BASE, "round1": "round1_tag",
        "selection": {"selected": "basecal", "tie": ["basecal", "cal"],
                      "reason": "lot precision within 1 gold row(s)",
                      "candidates": {"basecal": {"gold_ok": 283, "gold_lot_precision": 0.9881, "gold_lot_n": 252,
                                                 "gold_lot_correct": 249, "all_manzana_precision": 0.8609,
                                                 "lost_correct": 121},
                                     "cal": {"gold_ok": 283, "gold_lot_precision": 0.9885, "gold_lot_n": 261,
                                             "gold_lot_correct": 258, "all_manzana_precision": 0.8485, "lost_correct": 90}},
                      "a5_floor_check": {"basecal": True, "cal": False}, "a5_floor_all_dev_manzana_precision": 0.858},
        "baseline_dev": {"dev_gold_ok": 278, "dev_gold_predial_lot_precision": 0.996, "dev_lost_correct": 108,
                         "dev_gold_n": 289},
        "round_comparison": {"frozen_gold": {"round1": _side(), "round2": _side(323, 321, 9, 399, 15),
                                             "delta": {"ok": 18}, "transitions": {"ok_gained": 18, "ok_lost": 0},
                                             "lot_correct_transitions": {"gained": 12, "lost": 2, "both": 309}},
                             "frozen_all": {"round1": _side(782, 393, 225, 1062, 305), "round2": _side(801, 404, 221, 1090, 87),
                                            "delta": {"ok": 28}, "transitions": {"ok_gained": 30, "ok_lost": 2},
                                            "lot_correct_transitions": {"gained": 15, "lost": 4, "both": 389}}},
        "bootstrap": {"seed": 1, "resamples": 10000,
                      "dev": {"gold_ok": {"old": 278.0, "new": 283.0, "diff": 5.0, "lo": 1.0, "hi": 10.0, "p_diff_lt0": 0.0},
                              "gold_lot_precision": {"old": 0.996, "new": 0.9881, "diff": -0.0079, "lo": -0.0202,
                                                     "hi": 0.0001, "p_diff_lt0": 0.901},
                              "lost_correct": {"old": 108.0, "new": 121.0, "diff": 13.0, "lo": -5.0, "hi": 31.0,
                                               "p_diff_lt0": 0.07}},
                      "frozen": {"all_manzana_precision": {"old": 0.7314, "new": 0.7121, "diff": -0.0192, "lo": -0.0366,
                                                           "hi": -0.0027, "p_diff_lt0": 0.9885}}},
        "bands": {s: [_wilson_row("band", "alta"), _wilson_row("band", "media", 28, 26, 28, 0.77, 0.98),
                      _wilson_row("band", "baja", 31, 0, 0)] for s in sets},
        "by_fuente": {"candidate": {s: [_wilson_row("fuente", "nomenclatura_cert"),
                                        _wilson_row("fuente", "catastro", 2, 2, 2, 0.34, 1.0),
                                        _wilson_row("fuente", SECRET_ADDRESS)] for s in sets},
                      "baseline": {s: [] for s in sets}},
        "reliability_diagrams": {"dev_gold": {"predio": _bins(), "manzana": _bins(0.99, 0.99)},
                                 "dev_all": {"predio": _bins(), "manzana": _bins()},
                                 "frozen_gold": {"predio": _bins(), "manzana": _bins()}},
        "acceptance": {
            "results": [], "summary": {},
            "a6_oof_ece": {"candidate_predio": 0.0044, "candidate_manzana": 0.0022, "baseline_refit_limit_predio": 0.1075,
                           "baseline_refit_limit_manzana": 0.0438},
            "noise_band_assessment": {"A2": SECRET_ADDRESS, "A4": "english text"}},
        "seed_spread": {"one_gold_row_pp_dev": 0.346, "gold_n_dev": 289},
        "index": {"n_docs": 401860, "fallback": {"n_docs_added": 6941, "n_collisions_skipped": 2245,
                                                 "stats": {"candidates": 330387, "selected": 6941}}},
        "jev": {
            "limits": "english text " + SECRET_ADDRESS,
            "synthetic_candidate": {"counts": {"total": 300, "ok": 274, "skipped": 26, "indeterminate": 82, "labeled": 192},
                                    "agreement": {"n": 192, "agree": 191, "rate": 0.9948, "ci95": [0.971, 0.999]},
                                    "kappa": 0.0, "auc": 0.78, "brier": 0.008},
            "catfbdocs_candidate": {"counts": {"total": 300, "ok": 210, "skipped": 90, "indeterminate": 87, "labeled": 123},
                                    "agreement": {"n": 123, "agree": 121, "rate": 0.9837, "ci95": [0.9426, 0.9955]},
                                    "kappa": None, "auc": None, "brier": 0.0249},
            "crosscheck": {"catfbdocs_candidate": {
                "counts": {"judged": 210, "yes": 121, "no": 2, "indeterminate": 87}, "mean_jev_p": 0.776,
                "decisive_no_rate": 0.0095,
                "by_band": {"alta": {"judged": 143, "yes": 107, "no": 0, "indeterminate": 36, "mean_jev_p": 0.87,
                                     "decisive_no_rate": 0.0},
                            "baja": {"judged": 35, "yes": 9, "no": 2, "indeterminate": 24, "mean_jev_p": 0.55,
                                     "decisive_no_rate": 0.057}},
                "by_fuente": {"catastro": {"judged": 185, "yes": 120, "no": 2, "indeterminate": 63}},
                "sample_rows": [SECRET_ADDRESS]}},
        },
        "verification": {"pytest": "6533 passed, 1 failed (pre-existing other project)", "check_no_real_addresses_exit": 0,
                         "frozen_scored_times": 1},
    }


def _seed_comparison() -> dict:
    run = {"gold_n": 289, "gold_ok": 281, "gold_ok_coverage": 0.9723, "gold_lot_precision": 0.988, "gold_lot_n": 251,
           "gold_manzana_precision": 0.9928, "all_ok": 1217, "all_lot_precision": 0.6667, "all_manzana_precision": 0.8614,
           "lost_correct": 115, "tau_p": 0.94, "tau_m": 0.7,
           "calibration": {"predio": {"n": 5087, "ece10": 0.0044, "brier": 0.0271, "auc": 0.895, "refit_brier": 0.027,
                                      "refit_auc": 0.8985},
                           "manzana": {"n": 5014, "ece10": 0.0022, "brier": 0.004, "auc": 0.9146, "refit_brier": 0.0039,
                                       "refit_auc": 0.9176}}}
    other = dict(run, gold_ok=280, lost_correct=126, tau_p=0.945, gold_lot_n=255)
    return {"runs": {"s42": run, "s43": other}, "gold_n": 289, "one_gold_row_pp": 0.346,
            "abs_diff": {"gold_ok": 1, "lost_correct": 11, "gold_lot_precision": 0.0002}, "pair": ["s42", "s43"]}


def _bench_run(rps=110.0, p50=25.0):
    return {"n_docs": 330387, "cold_start_s": 2.5, "peak_rss_mb": 1500.0,
            "latency_ms": {"p50": p50, "p95": p50 + 5},
            "throughput": {"100": {"rows_per_s_median": rps}, "2000": {"rows_per_s_median": rps},
                           "10000": {"rows_per_s_median": rps}}}


def _bench(threads=16):
    return {"config": {"threads": threads},
            "runs": {"prod": {"torch": _bench_run(), "onnx": _bench_run(180, 17), "onnx-int8": _bench_run(220, 17)},
                     "cand": {"torch": _bench_run(120, 23), "onnx": _bench_run(190, 20), "onnx-int8": _bench_run(200, 19)}}}


def _final_world(tmp_path, *, onnx_dir="onnx_bench"):
    art = _make_world(tmp_path)
    exp = art / "experiments"
    _dump(exp / FINAL / "summary_for_report.json", _summary())
    _dump(exp / "seed_comparison.json", _seed_comparison())
    _dump(exp / FINAL / "decision.json", {"tau_p": 0.94, "tau_m": 0.7, "floor": 0.7, "bands": {"alta": 0.97, "media": 0.94}})
    if onnx_dir:
        _dump(exp / onnx_dir / "bench.json", _bench(16))
        _dump(exp / onnx_dir / "bench_threads4.json", _bench(4))
        _dump(exp / onnx_dir / "dev_metrics.json", {"torch/dev_gold": {"n": 289, "ok": 278},
                                                    "onnx/dev_gold": {"n": 289, "ok": 278, "estado_equal_to_torch": 289,
                                                                      "npn_equal_to_torch": 289},
                                                    "onnx-int8/dev_gold": {"n": 289, "ok": 278, "estado_equal_to_torch": 289,
                                                                           "npn_equal_to_torch": 288}})
        _dump(exp / onnx_dir / "parity_cand_serving.json", {"n_docs": 394919, "backends": {"onnx": [
            {"set": "public_docs", "n": 2000, "cosine_min": 0.99999, "top1_identical": 0.9975, "top1_differ": 5,
             "xy_max_abs_diff_m": 0.008}], "onnx-int8": [{"set": "public_docs", "n": 2000, "cosine_min": 0.9995,
                                                           "top1_identical": 0.9975, "top1_differ": 5,
                                                           "xy_max_abs_diff_m": 149.0}]}})
    return art


def _final_inputs(art, **over):
    kwargs = dict(candidates=[FINAL], controls=["v2c_cert_s42"], final_tag=FINAL)
    kwargs.update(over)
    return _inputs(art, **kwargs)


def _final_text(art, **over) -> str:
    return rr.render_report(_final_inputs(art, **over))


def _sec(text: str, n: int) -> str:
    nxt = f"\n## {n + 1}. "
    body = text.split(f"\n## {n}. ")[1]
    return body.split(nxt)[0] if nxt in body else body


def test_new_sections_13_to_21_present_in_order(tmp_path):
    text = _final_text(_final_world(tmp_path))
    positions = []
    for n, title in NEW_SECTIONS.items():
        assert f"\n## {n}. " in text, n
        assert title in _sec(text, n).split("\n")[0], title
        positions.append(text.index(f"\n## {n}. "))
    assert positions == sorted(positions)
    assert text.index("\n## 12. ") < positions[0]                      # appended after the reproduction commands


def test_story_has_three_stages_and_the_second_look_warning(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 13)
    for needle in ("Etapa 1", "Etapa 2", "Etapa 3", "SEGUNDA MIRADA", "no es ciega", "una sola vez"):
        assert needle in sec, needle
    assert "6941" in sec and "2245" in sec                              # fallback docs added / collisions skipped
    assert "round1_tag" in sec and "basecal" in sec                    # selected variant named
    assert "+18" in sec                                                 # OK gained on the same frozen rows
    assert SECRET_ADDRESS not in sec                                    # free text of the artifact never copied


def test_story_without_summary_is_pending_but_keeps_the_warning(tmp_path):
    art = _final_world(tmp_path)
    os.remove(art / "experiments" / FINAL / "summary_for_report.json")
    sec = _sec(_final_text(art), 13)
    assert rr.PENDING in sec and "SEGUNDA MIRADA" in sec


def test_noise_band_lists_both_seeds_the_spread_and_the_bootstrap(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 14)
    assert "s42" in sec and "s43" in sec
    assert "115" in sec and "126" in sec and "11" in sec               # lost_correct per seed and its spread
    assert "0.346" in sec                                              # one gold row in points
    assert "bootstrap" in sec.lower() and "[-0.0202" in sec and "0.9010" in sec
    assert "dentro" in sec                                              # candidate vs seed range is spelled out


def test_noise_band_without_seed_comparison_is_pending(tmp_path):
    art = _final_world(tmp_path)
    os.remove(art / "experiments" / "seed_comparison.json")
    sec = _sec(_final_text(art), 14)
    assert rr.PENDING in sec


def test_bands_and_fuente_tables_show_wilson_intervals(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 15)
    assert "Wilson" in sec
    assert "[0.9751 a 0.9992]" in sec
    for band in ("alta", "media", "baja"):
        assert band in sec
    for s in ("dev_gold", "dev_all", "frozen_gold", "frozen_all"):
        assert s in sec
    assert "nomenclatura_cert" in sec and "catastro" in sec
    assert "SEGUNDA MIRADA" in sec                                      # frozen tables carry the caveat


def test_bands_with_no_scored_rows_do_not_divide_or_crash(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 15)
    assert "0 / 0" in sec                                               # the 'baja' band has no lot-scored rows
    assert "nan" not in sec.lower()


def test_unsafe_fuente_labels_and_row_fields_never_reach_section_15(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 15)
    assert SECRET_ADDRESS not in sec and rr.OTHER_LABEL in sec


def test_calibration_reports_oof_ece_brier_and_reliability_bins(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 16)
    assert "0.0044" in sec and "0.0022" in sec                         # OOF ECE of both heads
    assert "0.0271" in sec                                              # Brier
    assert "0.1075" in sec and "0.0438" in sec                         # baseline refit limits
    assert "0.94" in sec and "0.70" in sec                             # tau_p / tau_m
    assert sec.count("| 9 |") >= 3                                      # reliability table: 10 bins (0..9) per set
    assert "![" not in sec                                              # text only, no images


def test_calibration_with_missing_inputs_is_pending(tmp_path):
    art = _final_world(tmp_path)
    for rel in ("seed_comparison.json", f"{FINAL}/decision.json", f"{FINAL}/summary_for_report.json"):
        os.remove(art / "experiments" / rel)
    assert rr.PENDING in _sec(_final_text(art), 16)


def test_jev_section_states_limits_and_crosscheck_by_band(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 17)
    assert "no es una verdad de referencia" in sec
    assert "alta" in sec and "baja" in sec and "0.0570" in sec         # decisive-no rate by band
    assert "indefinido" in sec                                          # kappa/AUC undefined on one class
    assert "catfbdocs_candidate" in sec and "synthetic_candidate" in sec
    assert SECRET_ADDRESS not in sec


def test_onnx_preliminary_when_only_onnx_bench_exists(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 18)
    assert "preliminar" in sec.lower()
    assert "onnx-int8" in sec and "1500" in sec and "220" in sec
    assert "149" in sec                                                 # int8 coordinate drift reported
    assert "289" in sec


def test_onnx_final_directory_wins_and_is_not_preliminary(tmp_path):
    art = _final_world(tmp_path, onnx_dir="onnx_bench")
    _dump(art / "experiments" / "onnx_bench_final" / "bench.json", _bench(16))
    sec = _sec(_final_text(art), 18)
    assert "preliminar" not in sec.lower()
    assert "onnx_bench_final" in sec


def test_onnx_empty_final_directory_falls_back_to_preliminary(tmp_path):
    art = _final_world(tmp_path, onnx_dir="onnx_bench")
    (art / "experiments" / "onnx_bench_final" / "cand_serving").mkdir(parents=True)
    assert "preliminar" in _sec(_final_text(art), 18).lower()


def test_onnx_missing_everywhere_is_pending(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path, onnx_dir=None)), 18)
    assert rr.PENDING in sec


def test_reading_explains_each_failed_criterion_in_plain_language(tmp_path):
    art = _make_world(tmp_path, acceptance_spec={"A2": "fail", "A4": "fail", "A6": "fail", "A9": "fail", "A10": "fail",
                                                 "A12": "fail"})
    exp = art / "experiments"
    _dump(exp / FINAL / "summary_for_report.json", _summary())
    _dump(exp / "seed_comparison.json", _seed_comparison())
    sec = _sec(_final_text(art), 19)
    for cid in ("A1", "A2", "A12"):
        assert f"| {cid} |" in sec
    assert "Estado" in sec and "Valor" in sec and "Umbral" in sec and "Motivo" in sec
    for cid in ("A2", "A4", "A6", "A9", "A10", "A12"):
        assert f"**{cid}**" in sec, cid
    assert "dentro del ruido de muestreo" in sec
    assert "no es una cuestión de ruido" in sec                         # A6 / A9
    assert "**A1**" not in sec                                          # passing criteria get no reading


def test_reading_with_all_passed_says_nothing_to_explain(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 19)
    assert "ningún criterio fallido" in sec


def test_reading_unknown_failed_criterion_gets_a_generic_line(tmp_path):
    art = _make_world(tmp_path, acceptance_spec={"A8": "fail"})
    _dump(art / "experiments" / FINAL / "summary_for_report.json", _summary())
    sec = _sec(_final_text(art), 19)
    assert "**A8**" in sec and "sin lectura automática" in sec


def test_open_decisions_cover_the_four_questions(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 20)
    for needle in ("A2", "tolerancia cero", "A6", "regla de envío", "A9", "proyecto ajeno", "precisión manzana",
                   "filas congeladas"):
        assert needle in sec, needle
    assert "0.7314" in sec and "0.7121" in sec                          # numbers come from the bootstrap block


def test_promotion_line_and_three_reversibility_switches(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 21)
    assert "pendiente de aprobación explícita del usuario" in sec
    assert "index_fallback.py remove" in sec
    assert "exclude_fuentes" in sec and "NORMALIZER_EXCLUDE_FUENTES" in sec
    assert "nunca se modifican" in sec


@pytest.mark.parametrize("payload", ["[]", "null", "42", '"text"', "{}"])
def test_wrong_summary_shapes_do_not_raise(tmp_path, payload):
    art = _final_world(tmp_path)
    for rel in (f"{FINAL}/summary_for_report.json", "seed_comparison.json", f"{FINAL}/decision.json",
                "onnx_bench/bench.json", "onnx_bench/dev_metrics.json"):
        _write(art / "experiments" / rel, payload)
    text = _final_text(art)
    for n in NEW_SECTIONS:
        assert f"\n## {n}. " in text


def test_garbage_values_in_summary_do_not_raise(tmp_path):
    art = _final_world(tmp_path)
    bad = _summary()
    bad["bands"] = {"dev_gold": [None, "x", {"band": 3, "n": "many", "lot_precision": float("nan")}]}
    bad["bootstrap"] = {"dev": {"gold_ok": "bad"}, "frozen": []}
    bad["round_comparison"] = {"frozen_gold": {"round1": "x", "round2": None}}
    bad["reliability_diagrams"] = {"dev_gold": {"predio": {"bins": [None, {"bin": "a"}]}}}
    bad["jev"] = {"crosscheck": {"a": {"by_band": {"alta": "oops"}}}}
    _write(art / "experiments" / FINAL / "summary_for_report.json", json.dumps(bad))
    text = _final_text(art)
    for n in NEW_SECTIONS:
        assert f"\n## {n}. " in text


def test_empty_artifacts_render_the_new_sections_as_pending(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    text = rr.render_report(rr.load_inputs(artifacts_dir=str(art), baseline_tag=BASE, candidates=["c1"], controls=[],
                                           final_tag=None, jev_dirs=[]))
    for n in NEW_SECTIONS:
        assert f"\n## {n}. " in text
    assert "SEGUNDA MIRADA" in text


@pytest.mark.parametrize("secret", [SECRET_ADDRESS, SECRET_ID, SECRET_NAME, SECRET_PATH, "EDIFICIO SECRETO"])
def test_new_sections_never_copy_row_level_strings(tmp_path, secret):
    art = _final_world(tmp_path)
    s = _summary()
    s["probe_rows"] = [{"direccion": secret}]
    s["selection"]["reason"] = secret
    s["verification"]["pytest"] = secret
    _dump(art / "experiments" / FINAL / "summary_for_report.json", s)
    assert secret not in _final_text(art)


def test_new_sections_are_deterministic(tmp_path):
    art = _final_world(tmp_path)
    assert _final_text(art) == _final_text(art)


def test_calibrated_metrics_override_replaces_stale_dev_numbers(tmp_path):
    art = _final_world(tmp_path)
    assert _final_inputs(art)["runs"]["v2c_cert_s42"]["metrics"]["dev_gold_ok"] == 276
    inputs = _final_inputs(art, calibrated_metrics=["seed_comparison.json:s43=v2c_cert_s42"])
    metrics = inputs["runs"]["v2c_cert_s42"]["metrics"]
    assert metrics["dev_gold_ok"] == 280 and metrics["dev_lost_correct"] == 126
    assert metrics["dev_gold_predial_lot_precision"] == 0.988
    assert "dev_gold_predial_precision" not in metrics                  # stale key without a calibrated source is dropped


@pytest.mark.parametrize("spec", ["", "garbage", "seed_comparison.json:nokey=v2c_cert_s42", "missing.json:s43=v2c_cert_s42",
                                  "seed_comparison.json:s43=unknown_tag", ":=", "a=b=c"])
def test_calibrated_metrics_bad_specs_are_ignored(tmp_path, spec):
    art = _final_world(tmp_path)
    inputs = _final_inputs(art, calibrated_metrics=[spec])
    assert inputs["runs"]["v2c_cert_s42"]["metrics"]["dev_gold_ok"] == 276


def test_training_report_is_inherited_along_inherits_from_experiment(tmp_path):
    art = _final_world(tmp_path)
    exp = art / "experiments"
    _make_run(tmp_path, "parent_run", train=_train_report(done=8, configured=8, seed=43))
    _dump(exp / "child_cal" / "experiment.json", {"tag": "child_cal", "inherits_from_experiment": "parent_run"})
    inputs = _final_inputs(art, candidates=[FINAL, "child_cal"])
    assert inputs["runs"]["child_cal"]["train_report"]["config"]["seed"] == 43
    assert "heredado de parent_run" in _sec(rr.render_report(inputs), 5)


def test_inheritance_cycles_do_not_loop(tmp_path):
    art = _final_world(tmp_path)
    exp = art / "experiments"
    _dump(exp / "a_cal" / "experiment.json", {"inherits_from_experiment": "b_cal"})
    _dump(exp / "b_cal" / "experiment.json", {"inherits_from_experiment": "a_cal"})
    inputs = _final_inputs(art, candidates=[FINAL, "a_cal"])
    assert inputs["runs"]["a_cal"]["train_report"] == {}


def test_confidence_curve_compare_is_the_fallback_for_the_final_run(tmp_path):
    art = _final_world(tmp_path)
    run = art / "experiments" / FINAL
    os.remove(run / "confidence_curve.json")
    _dump(run / "confidence_curve_compare.json", {"candidate": _curve(0.97, 0.9881, 0.007)["curve"], "baseline": []})
    assert len(_final_inputs(art)["runs"][FINAL]["confidence_curve"]) == 10


def test_cli_accepts_the_new_options(tmp_path):
    art = _final_world(tmp_path)
    out = tmp_path / "docs" / "final.md"
    assert _cli(art, out, "--calibrated-metrics", "seed_comparison.json:s43=v2c_cert_s42", "--seed-comparison",
                str(art / "experiments" / "seed_comparison.json"), "--onnx-dirs", "onnx_bench_final", "onnx_bench") == 0
    text = out.read_text(encoding="utf-8")
    assert "\n## 21. " in text and "preliminar" in text.lower()


def test_onnx_final_directory_accepts_bench_threads16_name_and_fp16_index(tmp_path):
    art = _final_world(tmp_path, onnx_dir="onnx_bench")
    fin = art / "experiments" / "onnx_bench_final"
    _dump(fin / "bench_threads16.json", _bench(16))
    _dump(fin / "bench_threads4.json", _bench(4))
    fp16 = _bench(16)
    fp16["runs"] = {"cand_fp16": {"onnx": _bench_run(777, 11)}}
    _dump(fin / "bench_cand_index_fp16.json", fp16)
    sec = _sec(_final_text(art), 18)
    assert "preliminar" not in sec.lower()
    assert "índice fp16" in sec and "777" in sec
    assert "289" in sec                         # dev metrics fall back to the earlier directory, labelled


def test_onnx_malformed_fp16_file_is_ignored(tmp_path):
    art = _final_world(tmp_path, onnx_dir="onnx_bench")
    fin = art / "experiments" / "onnx_bench_final"
    _dump(fin / "bench_threads16.json", _bench(16))
    (fin / "bench_cand_index_fp16.json").write_text("{not json", encoding="utf-8")
    sec = _sec(_final_text(art), 18)
    assert "índice fp16" not in sec and "preliminar" not in sec.lower()


def _repro_commands(art):
    sec = _sec(_final_text(art), 12)
    return [ln for ln in sec.splitlines() if ln.strip() and not ln.startswith(("```", "Ejecutar"))]


def test_repro_section_uses_final_names_and_scripts(tmp_path):
    sec = _sec(_final_text(_final_world(tmp_path)), 12)
    for needle in ("index_v2e_cert_pcer_r3", "index_v2f_cert_r3", "index_v2g_cert_pcer_catfb_r3",
                   "v2e_cert_pcer_s42_r3_cal", "v2e_cert_pcer_s43_cal", "v2g_cert_pcer_catfb_s42_r3_basecal",
                   "index_fallback.py add", "index_fallback.py experiment", "fit_decision.py",
                   "make_decision_experiment.py", "make_inherit_experiment.py", "--tiers cert,pcer", "--tiers cert "):
        assert needle in sec, needle


def test_repro_section_has_no_stale_index_in_commands(tmp_path):
    cmds = _repro_commands(_final_world(tmp_path))
    for ln in cmds:
        code = ln.split("#", 1)[0]
        assert "index_v2c" not in ln
        assert "index_v2d" not in code, ln         # the old r2 index may only be named in an explanatory comment


def test_repro_section_without_final_tag_keeps_placeholder(tmp_path):
    text = rr.render_report(_inputs(_make_world(tmp_path), final_tag=None))
    sec = _sec(text, 12)
    assert "<tag>" in sec and "index_v2g_cert_pcer_catfb_r3" in sec
