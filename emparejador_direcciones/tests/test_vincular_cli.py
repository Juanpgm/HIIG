"""Subprocess tests for scripts/vincular.py (link one address against a CSV base)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "vincular.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(CLI_SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


def _write_base(tmp_path: Path, rows: list[tuple[str, str]], encoding: str = "utf-8", header: str = "ID,ADDR") -> Path:
    path = tmp_path / "base.csv"
    lines = [header] + [f'{rid},"{addr}"' for rid, addr in rows]
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


BASE_ROWS = [
    ("P1", "KR 1 # 4 - 80"),
    ("P2", "KR 1 # 4 - 82"),
    ("P3", "K 49 E # 49 - 50 8 C"),
    ("P4", "KR 1 # 4 - 80 AP 5"),
    ("P5", "not an address"),
]


def _args(path: Path, *extra: str, query: str = "KR 1 # 4 - 80") -> list[str]:
    return ["--base", str(path), "--id-column", "ID", "--address-column", "ADDR", *extra, query]


def test_match_exits_zero_and_lists_best(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS)))
    assert result.returncode == 0
    assert result.stdout.startswith("MATCH")
    assert "P1" in result.stdout


def test_no_match_exits_one(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), query="KR 7 # 4 - 80"))
    assert result.returncode == 1
    assert result.stdout.startswith("SIN_COINCIDENCIA")


def test_unparseable_query_exits_one_with_reason(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), query="direccion sin formato"))
    assert result.returncode == 1
    assert "falta_separador_hash" in result.stdout


def test_json_output_shape(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), "--json"))
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    for key in ("consulta", "canonica", "parse_ok", "umbral", "tolerancia_placa", "n_comparados",
                "ambiguo", "mejor", "candidatos", "registros", "rechazados"):
        assert key in payload
    assert payload["mejor"]["id"] == "P1"
    assert payload["candidatos"][0]["id"] == "P1" and payload["candidatos"][0]["puntaje"] == 1.0
    assert payload["rechazados"] == 1 and payload["registros"] == 4
    assert payload["ambiguo"] is False


def test_json_when_no_candidate(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), "--json", query="KR 7 # 4 - 80"))
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["mejor"] is None and payload["candidatos"] == []


def test_top_limits_candidates(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), "--json", "--top", "1", "--umbral", "0.9"))
    payload = json.loads(result.stdout)
    assert len(payload["candidatos"]) == 1


def test_top_shows_more_candidates(tmp_path: Path) -> None:
    payload = json.loads(_run(*_args(_write_base(tmp_path, BASE_ROWS), "--json", "--top", "5")).stdout)
    assert [c["id"] for c in payload["candidatos"]] == ["P1", "P4"]


def test_ambiguous_flag_reported(tmp_path: Path) -> None:
    rows = [("A", "KR 1 # 4 - 80"), ("B", "KR 01 # 04 - 080")]
    result = _run(*_args(_write_base(tmp_path, rows), "--json"))
    payload = json.loads(result.stdout)
    assert payload["ambiguo"] is True
    plain = _run(*_args(_write_base(tmp_path, rows)))
    assert "AMBIGUO" in plain.stdout


def test_umbral_and_tolerancia(tmp_path: Path) -> None:
    path = _write_base(tmp_path, BASE_ROWS)
    assert _run(*_args(path, query="KR 1 # 4 - 81")).returncode == 1
    result = _run(*_args(path, "--tolerancia-placa", "1", query="KR 1 # 4 - 81"))
    assert result.returncode == 0


def test_invalid_threshold_exits_two(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), "--umbral", "1.5"))
    assert result.returncode == 2
    assert "umbral" in result.stderr.lower()


def test_negative_tolerance_exits_two(tmp_path: Path) -> None:
    assert _run(*_args(_write_base(tmp_path, BASE_ROWS), "--tolerancia-placa", "-1")).returncode == 2


def test_invalid_top_exits_two(tmp_path: Path) -> None:
    path = _write_base(tmp_path, BASE_ROWS)
    assert _run(*_args(path, "--top", "0")).returncode == 2
    assert _run(*_args(path, "--top", "-3")).returncode == 2


def test_missing_file_exits_two(tmp_path: Path) -> None:
    result = _run("--base", str(tmp_path / "nope.csv"), "--id-column", "ID", "--address-column", "ADDR", "KR 1 # 4 - 80")
    assert result.returncode == 2
    assert result.stderr


def test_missing_column_exits_two(tmp_path: Path) -> None:
    path = _write_base(tmp_path, BASE_ROWS)
    result = _run("--base", str(path), "--id-column", "ID", "--address-column", "NOPE", "KR 1 # 4 - 80")
    assert result.returncode == 2
    assert "NOPE" in result.stderr


def test_missing_required_args_exit_two() -> None:
    assert _run("KR 1 # 4 - 80").returncode == 2
    assert _run().returncode == 2


def test_empty_file_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert _run(*_args(path)).returncode == 2


def test_header_only_base_exits_one(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, [])))
    assert result.returncode == 1


def test_bad_encoding_name_exits_two(tmp_path: Path) -> None:
    assert _run(*_args(_write_base(tmp_path, BASE_ROWS), "--encoding", "no-such-codec")).returncode == 2


def test_undecodable_bytes_exit_two(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(b"ID,ADDR\nP1,\"KR 1 # 4 - \xff\xfe80\"\n")
    result = _run(*_args(path))
    assert result.returncode == 2
    assert result.stderr


def test_latin1_encoding_option(tmp_path: Path) -> None:
    rows = [("P1", "CRA 1 Nº 4 - 80")]
    path = _write_base(tmp_path, rows, encoding="latin-1")
    result = _run(*_args(path, "--encoding", "latin-1"))
    assert result.returncode == 0


def test_short_rows_and_empty_ids_do_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text('ID,ADDR\nP1\n,"KR 1 # 4 - 80"\nP3,"KR 1 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path, "--json"))
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["rechazados"] == 1


def test_unicode_query_does_not_crash(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), query="KR ñ # 4 - 80 \U0001F3E0"))
    assert result.returncode == 1
