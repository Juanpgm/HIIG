import json

import pytest

from conftest import SENTINEL_KEY, FakeTransport, Sleeper, choice_answer, echo_ok, http, noul_answer, ok
from validacion_jev.client import AuthenticationError, JevClient, JevConfig
from validacion_jev.io import Item
from validacion_jev.questions import PRIMARY_QUESTION, questions_for
from validacion_jev.validators import (
    MAX_FIELD_CHARS,
    StateError,
    Thresholds,
    build_state,
    certainty,
    check_state,
    dry_run_payloads,
    gate,
    run_validation,
    truncate,
)

NORM, MATCH = "normalizador", "matcher"


def norm_item(a="CRA 7 # 45-10", b="CARRERA 7 45 10", estado=None, id="1", **kw):
    return Item(id=id, a=a, b=b, decision=True, estado=estado, **kw)


def match_item(a="CL 5 # 10-20", b="CALLE 5 10 20", id="1", decision=True, **kw):
    return Item(id=id, a=a, b=b, decision=decision, **kw)


# ---- questions ---------------------------------------------------------------


def test_normalizer_questions_without_estado():
    qs = questions_for(NORM, include_estado=False)
    assert list(qs) == ["misma_ubicacion"]
    assert qs["misma_ubicacion"]["type"] == "noul"
    assert set(qs["misma_ubicacion"]["criteria"]) == {"true", "false"}
    assert "direccion_normalizada" in qs["misma_ubicacion"]["instructions"]


def test_normalizer_questions_with_estado():
    qs = questions_for(NORM, include_estado=True)
    assert list(qs) == ["misma_ubicacion", "estado_adecuado"]
    q = qs["estado_adecuado"]
    assert q["type"] == "choice"
    assert list(q["criteria"]) == ["adecuado", "debio_resolver", "debio_rechazar"]


def test_matcher_questions():
    qs = questions_for(MATCH)
    assert list(qs) == ["misma_puerta", "relacion"]
    assert list(qs["relacion"]["criteria"]) == [
        "misma_puerta",
        "mismo_edificio_distinta_unidad",
        "misma_via_distinta_puerta",
        "distinta_via",
    ]
    assert len(qs["relacion"]["criteria"]) <= 255


def test_questions_are_copies_not_shared_state():
    a = questions_for(MATCH)
    a["misma_puerta"]["instructions"] = "mutated"
    assert questions_for(MATCH)["misma_puerta"]["instructions"] != "mutated"


def test_primary_question_ids():
    assert PRIMARY_QUESTION == {NORM: "misma_ubicacion", MATCH: "misma_puerta"}


def test_unknown_kind():
    with pytest.raises(ValueError):
        questions_for("otro")


# ---- gating -------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,expected",
    [(0, 1.0), (1, 1.0), (0.5, 0.5), (0.15, 0.85), (0.85, 0.85), (0.6, 0.6), (0.4, 0.6)],
)
def test_noul_certainty(p, expected):
    assert certainty(noul_answer(p)) == pytest.approx(expected)


def test_choice_certainty_uses_confidence():
    assert certainty(choice_answer("a", {"a": 1.0}, 0.42)) == 0.42


@pytest.mark.parametrize(
    "c,expected",
    [
        (1.0, "auto"),
        (0.85, "auto"),
        (0.8499999, "review"),
        (0.60, "review"),
        (0.5999999, "escalate"),
        (0.5, "escalate"),
        (0.0, "escalate"),
    ],
)
def test_gate_boundaries(c, expected):
    assert gate(c, Thresholds()) == expected


@pytest.mark.parametrize("p,expected", [(0.15, "auto"), (0.1500001, "review"), (0.85, "auto")])
def test_gate_on_noul_boundaries_float_safe(p, expected):
    assert gate(certainty(noul_answer(p)), Thresholds()) == expected


def test_custom_thresholds():
    t = Thresholds(auto=0.95, review=0.7)
    assert gate(0.9, t) == "review" and gate(0.69, t) == "escalate"


@pytest.mark.parametrize("auto,review", [(0.5, 0.6), (1.1, 0.5), (0.9, -0.1)])
def test_invalid_thresholds(auto, review):
    with pytest.raises(ValueError):
        Thresholds(auto=auto, review=review)


# ---- truncation ---------------------------------------------------------------


@pytest.mark.parametrize("n,trunc", [(299, False), (300, False), (301, True)])
def test_truncate_boundaries(n, trunc):
    out, flag = truncate("x" * n)
    assert flag is trunc
    assert len(out) == min(n, MAX_FIELD_CHARS)


def test_truncate_unicode_counts_characters():
    out, flag = truncate("Ñ" * 301)
    assert flag and out == "Ñ" * 300


# ---- build_state / privacy filter ------------------------------------------------


def test_normalizer_state_only_allowed_fields():
    item = norm_item(estado="OK", raw={"telefono": "secret-phone-3001234567"})
    state, flags = build_state(NORM, item)
    assert state == {
        "direccion_original": "CRA 7 # 45-10",
        "direccion_normalizada": "CARRERA 7 45 10",
        "estado": "OK",
    }
    assert flags == []
    assert "secret" not in json.dumps(state)


def test_normalizer_state_without_estado_omits_key():
    state, _ = build_state(NORM, norm_item(estado=None))
    assert "estado" not in state


def test_matcher_state():
    state, _ = build_state(MATCH, match_item())
    assert state == {"direccion_a": "CL 5 # 10-20", "direccion_b": "CALLE 5 10 20"}


def test_truncation_is_flagged():
    state, flags = build_state(MATCH, match_item(a="A" * 301))
    assert len(state["direccion_a"]) == 300
    assert "truncated" in flags


def test_unicode_preserved():
    state, _ = build_state(MATCH, match_item(a="CALLE 5 Ñandú ÁÉÍ"))
    assert state["direccion_a"] == "CALLE 5 Ñandú ÁÉÍ"


def test_long_estado_truncated_too():
    state, flags = build_state(NORM, norm_item(estado="E" * 400))
    assert len(state["estado"]) == 300 and "truncated" in flags


@pytest.mark.parametrize("bad", [None, "", "   ", 123, 4.5, ["a"], {"a": 1}])
def test_non_string_or_blank_rejected(bad):
    with pytest.raises(StateError):
        build_state(MATCH, match_item(a=bad))


def test_check_state_rejects_extra_fields():
    with pytest.raises(StateError):
        check_state(MATCH, {"direccion_a": "x", "direccion_b": "y", "telefono": "1"})


def test_check_state_rejects_non_strings_and_missing():
    with pytest.raises(StateError):
        check_state(MATCH, {"direccion_a": "x", "direccion_b": 5})
    with pytest.raises(StateError):
        check_state(MATCH, {"direccion_a": "x"})
    with pytest.raises(StateError):
        check_state(NORM, {"direccion_original": "x", "direccion_normalizada": "y", "foo": "z"})


def test_check_state_enforces_length():
    with pytest.raises(StateError):
        check_state(MATCH, {"direccion_a": "x" * 301, "direccion_b": "y"})


# ---- dry run --------------------------------------------------------------------


def test_dry_run_payloads_contain_no_key_and_no_extras():
    items = [norm_item(estado="OK", id="a"), norm_item(id="b")]
    payloads = dry_run_payloads(NORM, items, "jev-latest")
    assert len(payloads) == 2
    assert payloads[0]["body"]["model"] == "jev-latest"
    assert "estado_adecuado" in payloads[0]["body"]["questions"]
    assert "estado_adecuado" not in payloads[1]["body"]["questions"]
    assert SENTINEL_KEY not in json.dumps(payloads)
    assert "Authorization" not in json.dumps(payloads)


def test_dry_run_skips_items_with_skip_reason():
    bad = norm_item(id="z")
    bad.skip_reason = "invalid_input"
    assert dry_run_payloads(NORM, [bad], "m") == []


def test_dry_run_empty():
    assert dry_run_payloads(MATCH, [], "m") == []


# ---- run_validation ----------------------------------------------------------------


def make_client(*script):
    t = FakeTransport(*script)
    return JevClient(JevConfig(api_key=SENTINEL_KEY), transport=t, sleep=Sleeper()), t


def test_run_validation_happy_path_matcher():
    client, t = make_client(echo_ok(0.97))
    res = run_validation(MATCH, [match_item(id="1")], client, Thresholds())
    r = res[0]
    assert r["status"] == "ok"
    assert r["jev_p"] == 0.97
    assert r["gates"]["misma_puerta"] == "auto"
    assert r["label"] == "yes"
    assert r["decision"] is True
    assert r["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert len(t.calls) == 1  # both questions in one request


def test_run_validation_indeterminate_flagged():
    client, _ = make_client(echo_ok(0.7))
    r = run_validation(MATCH, [match_item()], client, Thresholds())[0]
    assert r["label"] == "indeterminate"
    assert "indeterminate" in r["flags"]
    assert r["gates"]["misma_puerta"] == "review"


@pytest.mark.parametrize(
    "p,label",
    [(0, "no"), (0.15, "no"), (0.1500001, "indeterminate"), (0.85, "yes"), (0.8499999, "indeterminate"), (1, "yes"), (0.5, "indeterminate")],
)
def test_label_boundaries(p, label):
    client, _ = make_client(echo_ok(p))
    assert run_validation(MATCH, [match_item()], client, Thresholds())[0]["label"] == label


def test_skipped_items_never_sent_and_reported():
    dup = match_item(id="2")
    dup.skip_reason = "duplicate_pair"
    client, t = make_client(echo_ok())
    res = run_validation(MATCH, [match_item(id="1"), dup], client, Thresholds())
    assert len(t.calls) == 1
    assert res[1]["status"] == "skipped" and res[1]["skip_reason"] == "duplicate_pair"


def test_failed_item_is_error_status_not_agreement():
    client, _ = make_client(http(500, {}))
    r = run_validation(MATCH, [match_item()], client, Thresholds())[0]
    assert r["status"] == "error" and r["label"] is None and r["jev_p"] is None
    assert "error" in r["flags"] and r["errors"]


def test_secondary_failure_keeps_item_ok_with_flag():
    def respond(url, headers, body):
        return ok({"misma_puerta": noul_answer(0.99)})  # 'relacion' missing

    client, _ = make_client(respond)
    r = run_validation(MATCH, [match_item()], client, Thresholds())[0]
    assert r["status"] == "ok" and "partial_error" in r["flags"]


def test_auth_error_aborts_run():
    client, _ = make_client(http(401, {}))
    with pytest.raises(AuthenticationError):
        run_validation(MATCH, [match_item()], client, Thresholds())


def test_empty_items():
    client, t = make_client(echo_ok())
    assert run_validation(MATCH, [], client, Thresholds()) == []
    assert t.calls == []


def test_normalizer_with_estado_gets_choice_answer_and_gate():
    client, t = make_client(echo_ok(0.99))
    r = run_validation(NORM, [norm_item(estado="OK")], client, Thresholds())[0]
    assert "estado_adecuado" in r["answers"]
    assert r["gates"]["estado_adecuado"] == "auto"
    assert "estado_adecuado" in t.bodies[0]["questions"]


def test_truncated_flag_propagates_to_result():
    client, _ = make_client(echo_ok())
    r = run_validation(MATCH, [match_item(a="A" * 400)], client, Thresholds())[0]
    assert "truncated" in r["flags"]


def test_results_never_contain_key():
    client, _ = make_client(echo_ok())
    res = run_validation(MATCH, [match_item()], client, Thresholds())
    assert SENTINEL_KEY not in json.dumps(res)


def test_non_string_item_becomes_invalid_input_without_call():
    client, t = make_client(echo_ok())
    bad = match_item(a=None)
    res = run_validation(MATCH, [bad], client, Thresholds())
    assert t.calls == [] and res[0]["status"] == "skipped"
    assert res[0]["skip_reason"] == "invalid_input"
