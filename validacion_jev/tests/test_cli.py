import json

import pytest

from conftest import SENTINEL_KEY, FakeTransport, Sleeper, echo_ok, http, noul_answer, ok
from validacion_jev.cli import main

ENV = {"JEV_API_KEY": SENTINEL_KEY}


def csv_file(tmp_path, text, name="in.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


MATCH_CSV = "id,a,b,is_match,score\n1,CL 5 # 10-20,CALLE 5 10 20,1,0.9\n2,CL 6 # 1-1,CL 7 # 2-2,0,0.1\n"


def run(argv, transport=None, environ=ENV):
    return main(argv, transport=transport, environ=environ, sleep=Sleeper())


def all_text(tmp_path, capsys):
    cap = capsys.readouterr()
    out = cap.out + cap.err
    for f in tmp_path.rglob("*"):
        if f.is_file():
            out += f.read_text(encoding="utf-8", errors="replace")
    return out


def test_no_args_is_usage_error(capsys):
    assert run([]) == 2


def test_help_exits_zero(capsys):
    assert run(["--help"]) == 0
    assert "matcher" in capsys.readouterr().out


def test_unknown_option_is_usage_error():
    assert run(["matcher", "x.csv", "--nope"]) == 2


def test_happy_path_matcher(tmp_path, capsys):
    t = FakeTransport(echo_ok(0.98))
    out = tmp_path / "out"
    code = run(["matcher", csv_file(tmp_path, MATCH_CSV), "--out", str(out)], transport=t)
    assert code == 0
    assert len(t.calls) == 2
    lines = (out / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["id"] == "1"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["has_valid_evidence"] is True
    assert (out / "summary.md").exists()
    assert SENTINEL_KEY not in all_text(tmp_path, capsys)


def test_happy_path_normalizer(tmp_path):
    t = FakeTransport(echo_ok(0.99))
    p = csv_file(tmp_path, "entrada;normalizada;estado\nCRA 7 # 45-10;CARRERA 7 45 10;OK\n")
    out = tmp_path / "o"
    assert run(["normalizador", p, "--out", str(out)], transport=t) == 0
    assert "estado_adecuado" in t.bodies[0]["questions"]


def test_default_out_dir_is_salida_jev(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = FakeTransport(echo_ok())
    p = csv_file(tmp_path, MATCH_CSV)
    assert run(["matcher", p], transport=t) == 0
    assert (tmp_path / "salida_jev" / "results.jsonl").exists()


def test_missing_key_exit_2_before_any_io(tmp_path):
    t = FakeTransport(echo_ok())
    code = run(["matcher", csv_file(tmp_path, MATCH_CSV), "--out", str(tmp_path / "o")], transport=t, environ={})
    assert code == 2 and t.calls == []
    assert not (tmp_path / "o").exists()


def test_dry_run_prints_payloads_without_network_or_key(tmp_path, capsys):
    t = FakeTransport(echo_ok())
    code = run(
        ["matcher", csv_file(tmp_path, MATCH_CSV), "--dry-run", "--out", str(tmp_path / "o")],
        transport=t,
        environ={"JEV_API_KEY": SENTINEL_KEY},
    )
    cap = capsys.readouterr()
    assert code == 0 and t.calls == []
    assert "CALLE 5 10 20" in cap.out
    assert SENTINEL_KEY not in cap.out + cap.err
    assert "Authorization" not in cap.out
    assert not (tmp_path / "o").exists()


def test_dry_run_works_without_key(tmp_path):
    assert run(["matcher", csv_file(tmp_path, MATCH_CSV), "--dry-run"], environ={}) == 0


def test_all_items_failed_exit_3_and_no_evidence(tmp_path):
    t = FakeTransport(http(500, {"error": "boom"}))
    out = tmp_path / "o"
    code = run(["matcher", csv_file(tmp_path, MATCH_CSV), "--out", str(out)], transport=t)
    assert code == 3
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["has_valid_evidence"] is False and summary["counts"]["error"] == 2
    assert "sin evidencia" in (out / "summary.md").read_text(encoding="utf-8").lower()


def test_empty_input_exit_3(tmp_path):
    t = FakeTransport(echo_ok())
    code = run(["matcher", csv_file(tmp_path, "a,b,is_match\n"), "--out", str(tmp_path / "o")], transport=t)
    assert code == 3 and t.calls == []


def test_partial_failure_still_exit_0(tmp_path):
    def respond(url, headers, body):
        req = json.loads(body)
        if "CL 6" in json.dumps(req["state"]):
            return http(422, {"field": "state"})
        return echo_ok(0.98)(url, headers, body)

    t = FakeTransport(respond)
    out = tmp_path / "o"
    assert run(["matcher", csv_file(tmp_path, MATCH_CSV), "--out", str(out)], transport=t) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["error"] == 1 and summary["counts"]["ok"] == 1


def test_auth_failure_exit_2(tmp_path, capsys):
    t = FakeTransport(http(401, {"error": "nope"}))
    code = run(["matcher", csv_file(tmp_path, MATCH_CSV), "--out", str(tmp_path / "o")], transport=t)
    cap = capsys.readouterr()
    assert code == 2 and len(t.calls) == 1
    assert SENTINEL_KEY not in cap.out + cap.err


def test_missing_input_file_exit_2(tmp_path):
    assert run(["matcher", str(tmp_path / "nope.csv")]) == 2


def test_missing_columns_exit_2(tmp_path):
    assert run(["matcher", csv_file(tmp_path, "a,b\nx,y\n")]) == 2


@pytest.mark.parametrize("arg", ["0", "-3", "abc"])
def test_bad_max_rows_exit_2(tmp_path, arg):
    assert run(["matcher", csv_file(tmp_path, MATCH_CSV), "--max-rows", arg]) == 2


@pytest.mark.parametrize(
    "extra", [["--auto", "0.5", "--review", "0.9"], ["--auto", "1.5"], ["--review", "-1"]]
)
def test_bad_thresholds_exit_2(tmp_path, extra):
    assert run(["matcher", csv_file(tmp_path, MATCH_CSV), *extra]) == 2


@pytest.mark.parametrize("n,cap,calls", [(3, 4, 3), (3, 3, 3), (3, 2, 2)])
def test_max_rows_cap_limits_calls(tmp_path, n, cap, calls):
    rows = "".join(f"a{i},b{i},1\n" for i in range(n))
    t = FakeTransport(echo_ok())
    out = tmp_path / "o"
    code = run(
        ["matcher", csv_file(tmp_path, "a,b,is_match\n" + rows), "--max-rows", str(cap), "--out", str(out)],
        transport=t,
    )
    assert code == 0 and len(t.calls) == calls
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["input"]["capped"] is (n > cap)


def test_env_file_flag(tmp_path):
    env = tmp_path / "creds"
    env.write_text(f"JEV_API_KEY={SENTINEL_KEY}\nOTHER=1\n", encoding="utf-8")
    t = FakeTransport(echo_ok())
    code = run(
        ["matcher", csv_file(tmp_path, MATCH_CSV), "--env-file", str(env), "--out", str(tmp_path / "o")],
        transport=t,
        environ={},
    )
    assert code == 0 and t.calls[0]["headers"]["Authorization"] == f"Bearer {SENTINEL_KEY}"


def test_missing_env_file_exit_2(tmp_path):
    assert run(["matcher", csv_file(tmp_path, MATCH_CSV), "--env-file", str(tmp_path / "x")], environ={}) == 2


def test_summary_md_is_spanish_and_has_caveat(tmp_path):
    t = FakeTransport(echo_ok(0.98))
    out = tmp_path / "o"
    run(["matcher", csv_file(tmp_path, MATCH_CSV), "--out", str(out)], transport=t)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "Resumen" in md and "juez basado en un modelo" in md.lower()
    assert "Concordancia" in md


def test_dry_run_survives_non_utf8_console(tmp_path, monkeypatch):
    import io
    import sys

    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii", write_through=True))
    p = csv_file(tmp_path, "a,b,is_match\nCALLE 5 Ñandú,CL 5 Ñandú,1\n")
    assert run(["matcher", p, "--dry-run"], environ={}) == 0
    assert b"and" in raw.getvalue()
