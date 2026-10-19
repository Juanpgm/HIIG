import json

import pytest

from conftest import SENTINEL_KEY
from validacion_jev.io import (
    InputError,
    load_items,
    parse_bool,
    render_summary_md,
    write_outputs,
)

NORM, MATCH = "normalizador", "matcher"


def write(tmp_path, text, name="in.csv", encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding) if isinstance(text, str) else text)
    return str(p)


# ---- parse_bool ---------------------------------------------------------------


@pytest.mark.parametrize(
    "tok,exp",
    [
        (True, True), (False, False), (1, True), (0, False),
        ("1", True), ("0", False), ("true", True), ("FALSE", False),
        ("si", True), ("Sí", True), ("NO", False),
    ],
)
def test_parse_bool_ok(tok, exp):
    assert parse_bool(tok) is exp


@pytest.mark.parametrize("tok", ["", " ", "maybe", "2", 2, None, "talvez", 0.5, [], "truee"])
def test_parse_bool_bad(tok):
    with pytest.raises(ValueError):
        parse_bool(tok)


# ---- matcher CSV ---------------------------------------------------------------


def test_matcher_csv_comma(tmp_path):
    p = write(tmp_path, "id,a,b,is_match,score\n1,CL 5,CALLE 5,true,0.9\n2,CL 6,CL 7,0,0.2\n")
    r = load_items(p, MATCH)
    assert [i.id for i in r.items] == ["1", "2"]
    assert r.items[0].decision is True and r.items[1].decision is False
    assert r.items[0].score == 0.9 and r.items[0].score_name == "score"
    assert r.total_rows == 2 and not r.capped


def test_matcher_csv_semicolon_and_bom_and_unicode(tmp_path):
    p = write(tmp_path, "﻿a;b;is_match\nCALLE 5 Ñ;CL 5 Ñ;si\n")
    r = load_items(p, MATCH)
    assert r.items[0].a == "CALLE 5 Ñ" and r.items[0].decision is True
    assert r.items[0].id == "1"  # generated from row number


def test_header_case_and_whitespace_insensitive(tmp_path):
    p = write(tmp_path, " A , B ,Is_Match\nx,y,no\n")
    assert load_items(p, MATCH).items[0].decision is False


def test_missing_required_columns(tmp_path):
    p = write(tmp_path, "a,b\nx,y\n")
    with pytest.raises(InputError) as ei:
        load_items(p, MATCH)
    assert "is_match" in str(ei.value)


def test_empty_file(tmp_path):
    with pytest.raises(InputError):
        load_items(write(tmp_path, ""), MATCH)


def test_header_only_gives_zero_items(tmp_path):
    r = load_items(write(tmp_path, "a,b,is_match\n"), MATCH)
    assert r.items == [] and r.total_rows == 0


def test_exactly_one_row(tmp_path):
    r = load_items(write(tmp_path, "a,b,is_match\nx,y,1\n"), MATCH)
    assert len(r.items) == 1


def test_bad_bool_token_marks_item_invalid_not_crash(tmp_path):
    r = load_items(write(tmp_path, "a,b,is_match\nx,y,quizas\nx2,y2,1\n"), MATCH)
    assert r.items[0].skip_reason == "invalid_input"
    assert r.items[1].skip_reason is None


@pytest.mark.parametrize("row", [",y,1", "x,,1", "  ,y,1"])
def test_blank_strings_invalid(tmp_path, row):
    r = load_items(write(tmp_path, f"a,b,is_match\n{row}\n"), MATCH)
    assert r.items[0].skip_reason == "invalid_input"


def test_short_row_missing_cells_invalid(tmp_path):
    r = load_items(write(tmp_path, "a,b,is_match\nx\n"), MATCH)
    assert r.items[0].skip_reason == "invalid_input"


def test_duplicate_ids_are_disambiguated_and_flagged(tmp_path):
    r = load_items(write(tmp_path, "id,a,b,is_match\n7,x,y,1\n7,p,q,0\n"), MATCH)
    ids = [i.id for i in r.items]
    assert len(set(ids)) == 2 and ids[0] == "7"
    assert "duplicate_id" in r.items[1].flags


def test_duplicate_pairs_skipped(tmp_path):
    r = load_items(write(tmp_path, "a,b,is_match\nx,y,1\nx,y,1\n"), MATCH)
    assert r.items[1].skip_reason == "duplicate_pair"
    assert r.duplicates_skipped == 1


def test_bad_score_is_flag_not_invalid(tmp_path):
    r = load_items(write(tmp_path, "a,b,is_match,score\nx,y,1,abc\n"), MATCH)
    assert r.items[0].score is None and "bad_score" in r.items[0].flags
    assert r.items[0].skip_reason is None


def test_invalid_utf8(tmp_path):
    with pytest.raises(InputError):
        load_items(write(tmp_path, b"a,b,is_match\n\xff\xfe,y,1\n"), MATCH)


def test_missing_file():
    with pytest.raises(InputError):
        load_items("/no/such/file.csv", MATCH)


# ---- max rows --------------------------------------------------------------------


@pytest.mark.parametrize("n,cap,expected_len,capped", [(4, 5, 4, False), (5, 5, 5, False), (6, 5, 5, True)])
def test_max_rows_cap(tmp_path, n, cap, expected_len, capped):
    rows = "".join(f"a{i},b{i},1\n" for i in range(n))
    r = load_items(write(tmp_path, "a,b,is_match\n" + rows), MATCH, max_rows=cap)
    assert len(r.items) == expected_len
    assert r.capped is capped
    assert r.total_rows == n


def test_max_rows_invalid(tmp_path):
    p = write(tmp_path, "a,b,is_match\nx,y,1\n")
    with pytest.raises(InputError):
        load_items(p, MATCH, max_rows=0)


# ---- normalizer ----------------------------------------------------------------------


def test_normalizer_columns_and_score_preference(tmp_path):
    p = write(
        tmp_path,
        "id,entrada,normalizada,estado,confianza,confiabilidad\n"
        "a,CRA 7 45 10,CARRERA 7 45 10,OK,0.4,0.9\n"
        "b,CL 1,CALLE 1,,0.3,\n",
    )
    r = load_items(p, NORM)
    first, second = r.items
    assert first.estado == "OK" and first.score == 0.9 and first.score_name == "confiabilidad"
    assert second.estado is None and second.score == 0.3 and second.score_name == "confianza"
    assert first.decision is True


def test_normalizer_missing_columns(tmp_path):
    with pytest.raises(InputError) as ei:
        load_items(write(tmp_path, "entrada\nx\n"), NORM)
    assert "normalizada" in str(ei.value)


def test_normalizer_blank_normalizada_invalid(tmp_path):
    r = load_items(write(tmp_path, "entrada,normalizada\nCL 1,\n"), NORM)
    assert r.items[0].skip_reason == "invalid_input"


# ---- JSONL ---------------------------------------------------------------------------------


def test_jsonl_matcher_with_native_types(tmp_path):
    lines = [
        {"id": "x", "a": "CL 5", "b": "CALLE 5", "is_match": True, "score": 0.8},
        {"a": "CL 6", "b": None, "is_match": 1},
        {"a": 5, "b": "x", "is_match": "si"},
        {"a": "p", "b": "q", "is_match": "perhaps"},
        "not-an-object",
    ]
    text = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n\n{broken json\n"
    r = load_items(write(tmp_path, text, "in.jsonl"), MATCH)
    reasons = [i.skip_reason for i in r.items]
    assert reasons == [None, "invalid_input", "invalid_input", "invalid_input", "invalid_input", "invalid_input"]
    assert r.items[0].decision is True and r.items[0].score == 0.8


def test_jsonl_extra_keys_kept_only_in_raw(tmp_path):
    line = json.dumps({"a": "x", "b": "y", "is_match": 1, "telefono": "300"})
    r = load_items(write(tmp_path, line + "\n", "in.jsonl"), MATCH)
    assert r.items[0].raw.get("telefono") == "300"


# ---- outputs -----------------------------------------------------------------------------------


def test_write_outputs_creates_files_without_key(tmp_path):
    out = tmp_path / "nested" / "salida_jev"
    results = [{"id": "1", "status": "ok", "a": "Ñ", "b": "ñ"}]
    summary = {
        "kind": MATCH,
        "has_valid_evidence": True,
        "counts": {"total": 1},
    }
    write_outputs(str(out), results, summary)
    assert (out / "results.jsonl").exists()
    assert (out / "summary.json").exists()
    assert (out / "summary.md").exists()
    line = (out / "results.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["a"] == "Ñ"
    for f in out.iterdir():
        assert SENTINEL_KEY not in f.read_text(encoding="utf-8")


def test_render_summary_md_spanish_no_evidence():
    md = render_summary_md({"kind": MATCH, "has_valid_evidence": False, "counts": {"total": 3, "ok": 0}})
    assert "sin evidencia" in md.lower()
    assert "juez basado en un modelo" in md.lower()
