"""Subprocess tests for scripts/emparejar.py: exit codes, stdout, --json shape."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "emparejar.py"


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


def test_match_exits_zero_and_stdout_starts_with_match() -> None:
    result = _run("KR 1 # 4 - 80", "kr 1 #4-080")
    assert result.returncode == 0
    assert result.stdout.startswith("MATCH")


def test_no_match_exits_one() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 A # 4 - 80")
    assert result.returncode == 1
    assert result.stdout.startswith("NO_MATCH")


def test_unparseable_exits_one() -> None:
    result = _run("SIN DIRECCION", "SIN DIRECCION")
    assert result.returncode == 1


def test_missing_argument_exits_two() -> None:
    result = _run("KR 1 # 4 - 80")
    assert result.returncode == 2


def test_invalid_threshold_exits_two_with_umbral_in_stderr() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 A # 4 - 80", "--umbral", "1.5")
    assert result.returncode == 2
    assert "umbral" in result.stderr.lower()


def test_umbral_085_matches_085_pair() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 A # 4 - 80", "--umbral", "0.85")
    assert result.returncode == 0


def test_threshold_alias_accepted() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 A # 4 - 80", "--threshold", "0.85")
    assert result.returncode == 0


def test_tolerancia_placa_allows_plate_diff() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 # 4 - 82", "--tolerancia-placa", "2")
    assert result.returncode == 0


def test_json_output_has_expected_keys_and_types() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 A # 4 - 80", "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    for key in ("match", "resultado", "score", "threshold", "plate_tolerance", "reason", "a", "b", "fields", "warnings"):
        assert key in payload
    assert isinstance(payload["match"], bool)
    assert payload["match"] is False


def test_json_complement_conflict_reports_veto() -> None:
    result = _run(
        "CL 10 # 42 - 02 AP 501",
        "CL 10 # 42 - 02 AP 502",
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["match"] is False
    assert any(f.get("veto") for f in payload["fields"])


# ---------------------------------------------------------------------------
# Finding 6: plain output shows the rendered canonical form, JSON carries
# canonical + fields + notes per address
# ---------------------------------------------------------------------------


def test_plain_output_shows_rendered_canonical_form() -> None:
    result = _run("kr 1 #4-080", "KR 1 # 4 - 80")
    lines = result.stdout.splitlines()
    assert lines[1] == "A: KR 1 # 4 - 80"
    assert lines[2] == "B: KR 1 # 4 - 80"


def test_plain_output_falls_back_to_raw_input_when_unparseable() -> None:
    result = _run("SIN DIRECCION", "KR 1 # 4 - 80")
    lines = result.stdout.splitlines()
    assert lines[1] == "A: SIN DIRECCION"


def test_json_includes_canonical_and_fields_per_address() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 A # 4 - 80", "--json")
    payload = json.loads(result.stdout)
    assert payload["a"]["canonical"] == "KR 1 # 4 - 80"
    assert payload["a"]["fields"]["via_number"] == "1"
    assert payload["b"]["fields"]["via_letters"] == "A"
    assert "notes" in payload["a"]
    assert payload["a"]["input"] == "KR 1 # 4 - 80"


# ---------------------------------------------------------------------------
# Finding 7: threshold formatted with {:g} in the plain header too
# ---------------------------------------------------------------------------


def test_plain_header_formats_threshold_with_g() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 A # 4 - 80", "--umbral", "0.8501")
    assert "umbral=0.8501" in result.stdout


# ---------------------------------------------------------------------------
# Finding 8: "avisos" line for open-ended addresses
# ---------------------------------------------------------------------------


def test_avisos_line_present_when_plate_missing_both_sides() -> None:
    result = _run("KR 1 # 4", "KR 1 # 4")
    assert "avisos: sin_placa" in result.stdout


def test_no_avisos_line_for_full_addresses() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 # 4 - 80")
    assert "avisos" not in result.stdout


# ---------------------------------------------------------------------------
# Finding 11: unicode input and threshold/tolerance boundaries
# ---------------------------------------------------------------------------


def test_unicode_input_via_json_does_not_crash() -> None:
    result = _run("KR ñ # 4 - 80", "\U0001F3E0", "--json")
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert "match" in payload


def test_umbral_zero_boundary_accepted() -> None:
    result = _run("KR 1 # 4 - 80", "KR 2 # 4 - 80", "--umbral", "0")
    assert result.returncode in (0, 1)


def test_umbral_one_boundary_accepted() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 # 4 - 80", "--umbral", "1")
    assert result.returncode == 0


def test_negative_plate_tolerance_exits_two() -> None:
    result = _run("KR 1 # 4 - 80", "KR 1 # 4 - 82", "--tolerancia-placa", "-1")
    assert result.returncode == 2


def test_umbral_and_tolerancia_placa_together() -> None:
    result = _run(
        "KR 1 # 4 - 80",
        "KR 1 # 4 - 82",
        "--umbral",
        "0.5",
        "--tolerancia-placa",
        "2",
    )
    assert result.returncode == 0
