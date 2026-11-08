"""``scripts/monitor_signals.py``: drift signals of a calibrated run from its rows parquet (aggregates only)."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd

nan = float("nan")


def _script():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_monitor_signals", os.path.join(root, "scripts", "monitor_signals.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(alta=6, media=2, baja=2, abstained=1, pcer=3) -> pd.DataFrame:
    n_ok = alta + media + baja
    return pd.DataFrame({
        "estado": ["OK"] * n_ok + ["SIN_MATCH"] * (abstained + 1),
        "motivo": [""] * n_ok + ["confiabilidad baja: p_manzana 0.40 < 0.70"] * abstained + ["otro"],
        "confianza": [0.9] * n_ok + [nan] * (abstained + 1),
        "margen": [0.25] * n_ok + [nan] * (abstained + 1),
        "banda_confiabilidad": ["alta"] * alta + ["media"] * media + ["baja"] * baja + [None] * (abstained + 1),
        "p_predio_cal": [0.99] * alta + [0.95] * media + [0.5] * baja + [nan] * (abstained + 1),
        "fuente": ["nomenclatura_pcer"] * pcer + ["nomenclatura_cert"] * (n_ok - pcer) + [None] * (abstained + 1),
    })


def test_writes_the_signals_json(tmp_path):
    ms = _script()
    path = tmp_path / "r.parquet"
    rows().to_parquet(path)
    out = tmp_path / "signals.json"
    assert ms.main(["--rows", str(path), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok_rows"] == 10 and data["band_share"]["alta"] == 0.6 and "alarms" not in data


def test_baseline_comparison_adds_the_alarm_block(tmp_path):
    ms = _script()
    base_path, cur_path = tmp_path / "b.parquet", tmp_path / "c.parquet"
    rows().to_parquet(base_path)
    rows(alta=2, media=4, baja=4).to_parquet(cur_path)
    base_json = tmp_path / "base.json"
    assert ms.main(["--rows", str(base_path), "--out", str(base_json)]) == 0
    out = tmp_path / "cur.json"
    assert ms.main(["--rows", str(cur_path), "--baseline", str(base_json), "--out", str(out)]) == 0
    alarms = json.loads(out.read_text(encoding="utf-8"))["alarms"]
    assert alarms["alta_share"]["alarm"] is True and alarms["p_predio_p25"]["alarm"] is True


def test_a_missing_or_uncalibrated_rows_file_fails_cleanly(tmp_path):
    ms = _script()
    assert ms.main(["--rows", str(tmp_path / "nope.parquet"), "--out", str(tmp_path / "o.json")]) == 2
    plain = tmp_path / "p.parquet"
    pd.DataFrame({"estado": ["OK"], "motivo": [""]}).to_parquet(plain)
    out = tmp_path / "o2.json"
    assert ms.main(["--rows", str(plain), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["band_share"] is None


def test_refuses_to_overwrite_without_the_flag(tmp_path):
    ms = _script()
    path = tmp_path / "r.parquet"
    rows().to_parquet(path)
    out = tmp_path / "s.json"
    out.write_text("{}", encoding="utf-8")
    assert ms.main(["--rows", str(path), "--out", str(out)]) == 2
    assert ms.main(["--rows", str(path), "--out", str(out), "--overwrite"]) == 0


def test_a_bad_baseline_file_fails_cleanly(tmp_path):
    ms = _script()
    path = tmp_path / "r.parquet"
    rows().to_parquet(path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert ms.main(["--rows", str(path), "--baseline", str(bad), "--out", str(tmp_path / "x.json")]) == 2
