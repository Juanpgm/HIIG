"""Subprocess tests for the review findings on scripts/vincular.py (W1, W2, S2b, S7, S8, S9)."""

from __future__ import annotations

import json
from pathlib import Path

from test_vincular_cli import BASE_ROWS, _args, _run, _write_base

# ---------------------------------------------------------------------------
# W1: near-ambiguity signal (alternativas / casi_ambiguo)
# ---------------------------------------------------------------------------


def test_json_reports_alternatives_without_an_exact_tie(tmp_path: Path) -> None:
    rows = [("BARE", "KR 1 # 4 - 80"), ("UNIT", "KR 1 # 4 - 80 AP 5"), ("UNIT2", "KR 1 # 4 - 80 AP 6")]
    payload = json.loads(_run(*_args(_write_base(tmp_path, rows), "--json")).stdout)
    assert payload["ambiguo"] is False
    assert payload["casi_ambiguo"] is True
    assert payload["alternativas"] == 2
    assert payload["mejor"]["id"] == "BARE"


def test_json_single_candidate_has_no_alternatives(tmp_path: Path) -> None:
    payload = json.loads(_run(*_args(_write_base(tmp_path, BASE_ROWS[:1]), "--json")).stdout)
    assert payload["casi_ambiguo"] is False and payload["alternativas"] == 0


def test_json_no_candidate_has_no_alternatives(tmp_path: Path) -> None:
    payload = json.loads(_run(*_args(_write_base(tmp_path, BASE_ROWS), "--json", query="KR 7 # 4 - 80")).stdout)
    assert payload["casi_ambiguo"] is False and payload["alternativas"] == 0


def test_plain_output_shows_alternatives_and_a_spanish_note(tmp_path: Path) -> None:
    rows = [("BARE", "KR 1 # 4 - 80"), ("UNIT", "KR 1 # 4 - 80 AP 5")]
    result = _run(*_args(_write_base(tmp_path, rows)))
    assert "alternativas: 1" in result.stdout
    assert "nota:" in result.stdout
    assert "AMBIGUO" not in result.stdout


def test_plain_output_single_candidate_has_zero_alternatives_and_no_note(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS[:1])))
    assert "alternativas: 0" in result.stdout
    assert "nota:" not in result.stdout


# ---------------------------------------------------------------------------
# W2: candidate warnings are printed in plain output
# ---------------------------------------------------------------------------


def _line_starting_with(text: str, prefix: str) -> str | None:
    return next((line for line in text.splitlines() if line.startswith(prefix)), None)


def test_plain_prints_sin_placa_for_plateless_query(tmp_path: Path) -> None:
    rows = [("P1", "KR 1 # 4")]
    result = _run(*_args(_write_base(tmp_path, rows), query="KR 1 # 4"))
    assert result.returncode == 0
    line = _line_starting_with(result.stdout, "avisos:")
    assert line is not None and "sin_placa" in line


def test_plain_prints_opaque_complement_warning(tmp_path: Path) -> None:
    rows = [("P3", "K 49 E # 49 - 50 8 C")]
    result = _run(*_args(_write_base(tmp_path, rows), query="K 49 E # 49 - 50 8 C"))
    assert result.returncode == 0
    line = _line_starting_with(result.stdout, "avisos:")
    assert line is not None and "complemento_no_estructurado" in line


def test_plain_full_address_prints_no_avisos_line(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS[:1])))
    assert result.returncode == 0
    assert _line_starting_with(result.stdout, "avisos:") is None
    assert "avisos" not in result.stdout


def test_plain_top_list_shows_per_candidate_warnings(tmp_path: Path) -> None:
    rows = [("A", "KR 1 # 4 - 80"), ("B", "KR 1 # 4 - 80 8 C")]
    result = _run(*_args(_write_base(tmp_path, rows), "--umbral", "0.8", "--top", "5"))
    lines = result.stdout.splitlines()
    first = next(line for line in lines if line.startswith("1. "))
    second = next(line for line in lines if line.startswith("2. "))
    assert "avisos" not in first
    assert "complemento_no_estructurado" in second


def test_plain_no_candidate_prints_no_avisos_line(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), query="KR 7 # 4 - 80"))
    assert _line_starting_with(result.stdout, "avisos:") is None


# ---------------------------------------------------------------------------
# S2b: UTF-8 BOM in the header
# ---------------------------------------------------------------------------


def test_utf8_bom_header_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + b'ID,ADDR\nP1,"KR 1 # 4 - 80"\n')
    result = _run(*_args(path, "--json"))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mejor"]["id"] == "P1"


def test_utf8_bom_header_with_quoted_first_column(tmp_path: Path) -> None:
    path = tmp_path / "bom_quoted.csv"
    path.write_bytes(b"\xef\xbb\xbf" + b'"ID","ADDR"\nP1,"KR 1 # 4 - 80"\n')
    result = _run(*_args(path, "--json"))
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# S7: rows with an empty/whitespace id are skipped and counted
# ---------------------------------------------------------------------------


def test_empty_and_blank_ids_are_skipped_and_counted(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text(
        'ID,ADDR\n,"KR 1 # 4 - 80"\n   ,"KR 1 # 4 - 80"\n\t,"KR 1 # 4 - 80"\nP1,"KR 1 # 4 - 80"\n',
        encoding="utf-8",
    )
    result = _run(*_args(path, "--json"))
    payload = json.loads(result.stdout)
    assert payload["ids_vacios"] == 3
    assert payload["registros"] == 1
    assert [c["id"] for c in payload["candidatos"]] == ["P1"]
    assert payload["casi_ambiguo"] is False
    assert "3" in result.stderr and "sin id" in result.stderr


def test_all_ids_empty_exits_one_with_stderr_warning(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text('ID,ADDR\n,"KR 1 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path))
    assert result.returncode == 1
    assert "sin id" in result.stderr


def test_no_empty_ids_means_zero_and_no_stderr_warning(tmp_path: Path) -> None:
    result = _run(*_args(_write_base(tmp_path, BASE_ROWS), "--json"))
    assert json.loads(result.stdout)["ids_vacios"] == 0
    assert "sin id" not in result.stderr


def test_plain_output_reports_skipped_empty_ids(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text('ID,ADDR\n,"KR 1 # 4 - 80"\nP1,"KR 1 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path))
    base_line = _line_starting_with(result.stdout, "base:")
    assert base_line is not None and "1 sin id" in base_line


def test_id_with_surrounding_spaces_is_kept_as_given(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text('ID,ADDR\n P1 ,"KR 1 # 4 - 80"\n', encoding="utf-8")
    payload = json.loads(_run(*_args(path, "--json")).stdout)
    assert payload["mejor"]["id"] == " P1 "
    assert payload["ids_vacios"] == 0


# ---------------------------------------------------------------------------
# S8: malformed CSV is rejected (exit 2), valid oddities are accepted
# ---------------------------------------------------------------------------


def test_unterminated_quote_exits_two_with_spanish_error(tmp_path: Path) -> None:
    path = tmp_path / "quote.csv"
    path.write_text('ID,ADDR\nP1,"KR 1 # 4 - 80\nP2,"KR 2 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path))
    assert result.returncode == 2
    assert "CSV mal formado" in result.stderr
    assert result.stdout == ""


def test_unterminated_quote_at_end_of_file_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "quote.csv"
    path.write_text('ID,ADDR\nP1,"KR 1 # 4 - 80', encoding="utf-8")
    result = _run(*_args(path))
    assert result.returncode == 2
    assert "CSV mal formado" in result.stderr


def test_garbage_after_closing_quote_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "quote.csv"
    path.write_text('ID,ADDR\nP1,"KR 1 # 4 - 80"junk\n', encoding="utf-8")
    assert _run(*_args(path)).returncode == 2


def test_embedded_newline_in_a_quoted_field_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "newline.csv"
    path.write_text('ID,ADDR\nP1,"KR 1 # 4 -\n80"\nP2,"KR 2 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path, "--json"))
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert payload["registros"] + payload["rechazados"] == 2


def test_embedded_comma_in_a_quoted_field_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "comma.csv"
    path.write_text('ID,ADDR\nP1,"KR 1 # 4 - 80 AP 5, TORRE 1"\n', encoding="utf-8")
    assert _run(*_args(path, "--json")).returncode in (0, 1)


def test_wrong_delimiter_exits_two_with_a_hint(tmp_path: Path) -> None:
    path = tmp_path / "semi.csv"
    path.write_text('ID;ADDR\nP1;"KR 1 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path))
    assert result.returncode == 2
    assert "delimitador" in result.stderr


def test_tab_delimiter_exits_two_with_a_hint(tmp_path: Path) -> None:
    path = tmp_path / "tab.csv"
    path.write_text('ID\tADDR\nP1\t"KR 1 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path))
    assert result.returncode == 2
    assert "delimitador" in result.stderr


def test_row_with_extra_fields_exits_two(tmp_path: Path) -> None:
    """An unquoted comma inside the address would silently truncate it."""
    path = tmp_path / "long.csv"
    path.write_text("ID,ADDR\nP1,KR 1 # 4 - 80, TORRE 1\n", encoding="utf-8")
    result = _run(*_args(path))
    assert result.returncode == 2
    assert "CSV mal formado" in result.stderr
    assert "línea 2" in result.stderr


def test_row_with_only_empty_extra_fields_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "trailing.csv"
    path.write_text('ID,ADDR\nP1,"KR 1 # 4 - 80",\n', encoding="utf-8")
    result = _run(*_args(path, "--json"))
    assert result.returncode == 0
    assert json.loads(result.stdout)["mejor"]["id"] == "P1"


def test_short_row_is_rejected_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    path.write_text('ID,ADDR\nP1\nP2,"KR 1 # 4 - 80"\n', encoding="utf-8")
    result = _run(*_args(path, "--json"))
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["rechazados"] == 1 and payload["registros"] == 1


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    path.write_text('ID,ADDR\n\nP1,"KR 1 # 4 - 80"\n\n', encoding="utf-8")
    payload = json.loads(_run(*_args(path, "--json")).stdout)
    assert payload["registros"] == 1 and payload["rechazados"] == 0


# ---------------------------------------------------------------------------
# S9: --max-block-scan
# ---------------------------------------------------------------------------


def _big_block_csv(tmp_path: Path, n: int) -> Path:
    path = tmp_path / "big.csv"
    lines = ["ID,ADDR"] + [f'R{i},"KR 1 # 4 - 80 AP {i}"' for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_json_reports_truncated_false_by_default(tmp_path: Path) -> None:
    payload = json.loads(_run(*_args(_write_base(tmp_path, BASE_ROWS), "--json")).stdout)
    assert payload["truncado"] is False


def test_max_block_scan_truncates_a_5000_record_block(tmp_path: Path) -> None:
    path = _big_block_csv(tmp_path, 5000)
    result = _run(*_args(path, "--json", "--max-block-scan", "100", query="KR 1 # 4 - 80"))
    payload = json.loads(result.stdout)
    assert payload["truncado"] is True
    assert payload["n_comparados"] == 100
    plain = _run(*_args(path, "--max-block-scan", "100", query="KR 1 # 4 - 80"))
    assert "TRUNCADO" in plain.stdout


def test_default_scans_the_whole_5000_record_block(tmp_path: Path) -> None:
    path = _big_block_csv(tmp_path, 5000)
    payload = json.loads(_run(*_args(path, "--json", query="KR 1 # 4 - 80")).stdout)
    assert payload["truncado"] is False and payload["n_comparados"] == 5000


def test_invalid_max_block_scan_exits_two(tmp_path: Path) -> None:
    path = _write_base(tmp_path, BASE_ROWS)
    assert _run(*_args(path, "--max-block-scan", "0")).returncode == 2
    assert _run(*_args(path, "--max-block-scan", "-5")).returncode == 2
    assert _run(*_args(path, "--max-block-scan", "abc")).returncode == 2
