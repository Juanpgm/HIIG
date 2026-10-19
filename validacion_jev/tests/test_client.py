import json
import logging
import math

import pytest

from conftest import (
    SENTINEL_KEY,
    FakeTransport,
    Sleeper,
    choice_answer,
    http,
    make_client,
    noul_answer,
    ok,
)
from validacion_jev.client import (
    MAX_QUESTIONS_PER_REQUEST,
    AuthenticationError,
    JevApiError,
    JevClient,
    JevResponseError,
    chunked,
    validate_answer,
)

STATE = {"direccion_a": "CL 5 # 10-20", "direccion_b": "CALLE 5 10 20"}
NOUL_Q = {"type": "noul", "instructions": "x"}
CHOICE_Q = {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}


def noul_questions(n):
    return {f"q{i}": dict(NOUL_Q) for i in range(n)}


def answers_for(questions, p=0.9):
    return {qid: noul_answer(p) for qid in questions}


# ---- chunking --------------------------------------------------------------


def test_max_questions_constant():
    assert MAX_QUESTIONS_PER_REQUEST == 12


@pytest.mark.parametrize(
    "n,expected_chunks", [(0, 0), (1, 1), (11, 1), (12, 1), (13, 2), (24, 2), (25, 3)]
)
def test_chunk_boundaries(config, n, expected_chunks):
    qs = noul_questions(n)

    def respond(url, headers, body):
        return ok(answers_for(json.loads(body)["questions"]))

    client, transport = make_client(config, respond)
    result = client.query(STATE, qs)
    assert len(transport.calls) == expected_chunks
    assert len(result.answers) == n
    assert result.errors == {}
    assert all(len(b["questions"]) <= 12 for b in transport.bodies)


def test_chunked_helper():
    assert list(chunked([], 12)) == []
    assert [len(c) for c in chunked(list(range(25)), 12)] == [12, 12, 1]


# ---- request shape ---------------------------------------------------------


def test_request_shape_and_headers(config):
    qs = {"only": dict(NOUL_Q)}
    client, transport = make_client(config, ok({"only": noul_answer(0.5)}))
    client.query(STATE, qs)
    call = transport.calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"]["Authorization"] == f"Bearer {SENTINEL_KEY}"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["timeout"] == 30.0
    assert not call["body"].startswith(b"\xef\xbb\xbf")
    body = json.loads(call["body"].decode("utf-8"))
    assert body == {"state": STATE, "model": "jev-latest", "questions": qs}


def test_unicode_is_sent_as_utf8(config):
    state = {"direccion_a": "CALLE 5 Ñ ÁÉ", "direccion_b": "x"}
    client, transport = make_client(config, ok({"q": noul_answer(1)}))
    client.query(state, {"q": dict(NOUL_Q)})
    assert "CALLE 5 Ñ ÁÉ" in transport.calls[0]["body"].decode("utf-8")


def test_usage_is_summed_across_chunks(config):
    def respond(url, headers, body):
        return ok(answers_for(json.loads(body)["questions"]), usage=(100, 7))

    client, _ = make_client(config, respond)
    result = client.query(STATE, noul_questions(13))
    assert result.usage == {"input_tokens": 200, "output_tokens": 14}


# ---- retries ---------------------------------------------------------------


def test_401_no_retry_and_clear_error(config, sleeper):
    client, transport = make_client(config, http(401, {"error": "bad key"}), sleeper=sleeper)
    with pytest.raises(AuthenticationError) as ei:
        client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 1
    assert sleeper.calls == []
    assert "401" in str(ei.value)
    assert SENTINEL_KEY not in str(ei.value)


@pytest.mark.parametrize("status", [500, 502, 529, 408])
def test_retryable_status_exhausts_retries(config, sleeper, status):
    client, transport = make_client(config, http(status, {"error": "x"}), sleeper=sleeper)
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 3  # first attempt + 2 retries
    assert len(sleeper.calls) == 2
    assert "q" in result.errors and str(status) in result.errors["q"]
    assert result.answers == {}


def test_retry_then_success(config, sleeper):
    client, transport = make_client(
        config, http(500, {}), ok({"q": noul_answer(0.9)}), sleeper=sleeper
    )
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 2
    assert result.answers["q"]["noul"] == 0.9


def test_429_retry_after_seconds_then_success(config, sleeper):
    client, transport = make_client(
        config,
        http(429, {}, {"Retry-After": "2"}),
        ok({"q": noul_answer(0.9)}),
        sleeper=sleeper,
    )
    client.query(STATE, {"q": dict(NOUL_Q)})
    assert sleeper.calls == [2.0]


def test_429_retry_after_ms_wins(config, sleeper):
    client, _ = make_client(
        config,
        http(429, {}, {"retry-after-ms": "1500", "retry-after": "9"}),
        ok({"q": noul_answer(0.9)}),
        sleeper=sleeper,
    )
    client.query(STATE, {"q": dict(NOUL_Q)})
    assert sleeper.calls == [1.5]


def test_retry_after_is_capped(config, sleeper):
    client, _ = make_client(
        config,
        http(429, {}, {"Retry-After": "86400"}),
        ok({"q": noul_answer(0.9)}),
        sleeper=sleeper,
    )
    client.query(STATE, {"q": dict(NOUL_Q)})
    assert sleeper.calls == [60.0]


def test_garbage_retry_after_falls_back_to_backoff(config, sleeper):
    client, _ = make_client(
        config,
        http(429, {}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        ok({"q": noul_answer(0.9)}),
        sleeper=sleeper,
    )
    client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(sleeper.calls) == 1 and 0 < sleeper.calls[0] <= 8


def test_exponential_backoff_grows(config, sleeper):
    client, _ = make_client(config, http(500, {}), sleeper=sleeper)
    client.query(STATE, {"q": dict(NOUL_Q)})
    assert sleeper.calls[1] > sleeper.calls[0]


@pytest.mark.parametrize("exc", [TimeoutError("timed out"), ConnectionResetError("reset")])
def test_transport_errors_are_retried_then_reported(config, sleeper, exc):
    client, transport = make_client(config, exc, sleeper=sleeper)
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 3
    assert "q" in result.errors


def test_timeout_then_success(config, sleeper):
    client, transport = make_client(
        config, TimeoutError("t"), ok({"q": noul_answer(0.2)}), sleeper=sleeper
    )
    assert client.query(STATE, {"q": dict(NOUL_Q)}).answers["q"]["noul"] == 0.2


def test_transport_error_message_is_redacted(config, sleeper):
    exc = ConnectionError(f"proxy said {SENTINEL_KEY}")
    client, _ = make_client(config, exc, sleeper=sleeper)
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert SENTINEL_KEY not in json.dumps(result.errors)


# ---- non-retryable errors (per-chunk, reported) -----------------------------


def test_422_includes_field_name(config, sleeper):
    body = {"error": {"message": "Invalid value", "field": "questions.q.type"}}
    client, transport = make_client(config, http(422, body), sleeper=sleeper)
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 1
    assert "questions.q.type" in result.errors["q"]
    assert "422" in result.errors["q"]


@pytest.mark.parametrize("status", [413, 415])
def test_413_415_not_retried(config, sleeper, status):
    client, transport = make_client(config, http(status, {"error": "no"}), sleeper=sleeper)
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 1
    assert str(status) in result.errors["q"]


def test_400_max_tokens_exceeded(config):
    client, transport = make_client(config, http(400, {"error": "max_tokens_exceeded"}))
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 1
    assert "max_tokens_exceeded" in result.errors["q"]


def test_error_body_truncated_and_non_json_tolerated(config):
    client, _ = make_client(config, http(400, b"<html>" + b"x" * 5000))
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(result.errors["q"]) < 600


# ---- malformed 200 responses ------------------------------------------------


def test_non_json_200(config):
    client, _ = make_client(config, http(200, b"not json at all"))
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert result.answers == {} and "q" in result.errors


@pytest.mark.parametrize(
    "payload",
    [[], "str", {"answers": []}, {"answers": None}, {"model": "x"}, 5],
)
def test_wrong_shape(config, payload):
    client, _ = make_client(config, http(200, payload))
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert result.answers == {} and "q" in result.errors


def test_missing_answer_id_reported_others_kept(config):
    qs = {"a": dict(NOUL_Q), "b": dict(NOUL_Q)}
    client, _ = make_client(config, ok({"a": noul_answer(0.9)}))
    result = client.query(STATE, qs)
    assert "a" in result.answers and "b" in result.errors


def test_extra_answer_id_ignored_with_warning(config):
    client, _ = make_client(config, ok({"a": noul_answer(0.9), "zzz": noul_answer(0.1)}))
    result = client.query(STATE, {"a": dict(NOUL_Q)})
    assert set(result.answers) == {"a"}
    assert any("zzz" in w for w in result.warnings)


def test_missing_usage_defaults_to_zero(config):
    client, _ = make_client(config, http(200, {"answers": {"a": noul_answer(0.5)}}))
    result = client.query(STATE, {"a": dict(NOUL_Q)})
    assert result.usage == {"input_tokens": 0, "output_tokens": 0}
    assert result.warnings


def test_invalid_answer_marks_only_that_question(config):
    qs = {"a": dict(NOUL_Q), "b": dict(NOUL_Q)}
    client, _ = make_client(config, ok({"a": noul_answer(7), "b": noul_answer(0.5)}))
    result = client.query(STATE, qs)
    assert "a" in result.errors and result.answers["b"]["noul"] == 0.5


# ---- partial batch failure ---------------------------------------------------


def test_partial_batch_failure_reported_and_excluded(config, sleeper):
    qs = noul_questions(25)
    state = {"n": 0}

    def respond(url, headers, body):
        req = json.loads(body)
        if "q12" in req["questions"]:  # second chunk
            return http(422, {"field": "questions"})
        return ok(answers_for(req["questions"]))

    client, transport = make_client(config, respond, sleeper=sleeper)
    result = client.query(STATE, qs)
    assert len(result.answers) == 13
    assert set(result.errors) == {f"q{i}" for i in range(12, 24)}
    assert "q24" in result.answers


def test_auth_error_in_later_chunk_propagates(config):
    def respond(url, headers, body):
        if "q12" in json.loads(body)["questions"]:
            return http(401, {})
        return ok(answers_for(json.loads(body)["questions"]))

    client, _ = make_client(config, respond)
    with pytest.raises(AuthenticationError):
        client.query(STATE, noul_questions(13))


# ---- validate_answer ---------------------------------------------------------


@pytest.mark.parametrize("p", [0, 0.15, 0.1500001, 0.6, 0.85, 1, 1.0, 0.0])
def test_noul_valid_boundaries(p):
    assert validate_answer(NOUL_Q, noul_answer(p))["noul"] == p


@pytest.mark.parametrize(
    "p", [-0.0001, 1.0001, math.nan, math.inf, "0.5", True, False, None, [0.5]]
)
def test_noul_invalid(p):
    with pytest.raises(JevResponseError):
        validate_answer(NOUL_Q, {"type": "noul", "noul": p})


def test_noul_wrong_type_tag_and_non_dict():
    with pytest.raises(JevResponseError):
        validate_answer(NOUL_Q, {"type": "choice", "noul": 0.5})
    with pytest.raises(JevResponseError):
        validate_answer(NOUL_Q, "0.5")


def test_choice_valid():
    a = choice_answer("a", {"a": 0.7, "b": 0.3}, 0.8)
    assert validate_answer(CHOICE_Q, a)["choice"] == "a"


def test_choice_probabilities_tolerate_rounding():
    validate_answer(CHOICE_Q, choice_answer("a", {"a": 0.7004, "b": 0.3}, 0.8))


@pytest.mark.parametrize(
    "answer",
    [
        choice_answer("a", {"a": 0.7, "b": 0.7}),  # sums 1.4
        choice_answer("a", {"a": 0.2, "b": 0.2}),  # sums 0.4
        choice_answer("zzz", {"a": 0.5, "b": 0.5}),  # not in options
        choice_answer("a", {"a": 0.5, "q": 0.5}),  # unknown option in probs
        choice_answer("a", {"a": 0.5, "b": 0.5}, 1.2),  # confidence high
        choice_answer("a", {"a": 0.5, "b": 0.5}, -0.1),  # confidence low
        choice_answer("a", {"a": 0.5, "b": 0.5}, math.nan),
        choice_answer("a", {"a": 0.5, "b": 0.5}, True),
        choice_answer("a", "oops"),
        choice_answer(3, {"a": 0.5, "b": 0.5}),
        {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}},  # no confidence
    ],
)
def test_choice_invalid(answer):
    with pytest.raises(JevResponseError):
        validate_answer(CHOICE_Q, answer)


def test_score_validation():
    q = {"type": "score", "instructions": "x", "criteria": ["bad", "ok", "good"]}
    ok_a = {"type": "score", "score": 1.5, "legend": {}, "probabilities": {}, "confidence": 0.7}
    assert validate_answer(q, ok_a)["score"] == 1.5
    for bad in (
        {**ok_a, "confidence": 2},
        {**ok_a, "score": "x"},
        {**ok_a, "score": math.nan},
    ):
        with pytest.raises(JevResponseError):
            validate_answer(q, bad)


def test_unknown_question_type_rejected():
    with pytest.raises(JevResponseError):
        validate_answer({"type": "wat"}, {"type": "wat"})


# ---- key hygiene -------------------------------------------------------------


def test_key_absent_from_client_repr_logs_and_errors(config, caplog, sleeper):
    caplog.set_level(logging.DEBUG)
    client, _ = make_client(config, http(500, {"error": f"echo {SENTINEL_KEY}"}), sleeper=sleeper)
    result = client.query(STATE, {"q": dict(NOUL_Q)})
    assert SENTINEL_KEY not in repr(client)
    assert SENTINEL_KEY not in str(client)
    assert SENTINEL_KEY not in caplog.text
    assert SENTINEL_KEY not in json.dumps(result.errors)
    assert SENTINEL_KEY not in repr(result)


def test_auth_error_redacts_echoed_key(config):
    client, _ = make_client(config, http(401, {"error": f"invalid {SENTINEL_KEY}"}))
    with pytest.raises(AuthenticationError) as ei:
        client.query(STATE, {"q": dict(NOUL_Q)})
    assert SENTINEL_KEY not in str(ei.value)
    assert SENTINEL_KEY not in repr(ei.value)


def test_default_transport_not_used_when_injected(config):
    client, transport = make_client(config, ok({"q": noul_answer(1)}))
    client.query(STATE, {"q": dict(NOUL_Q)})
    assert len(transport.calls) == 1


def test_empty_questions_makes_no_request(config):
    client, transport = make_client(config, ok({}))
    result = client.query(STATE, {})
    assert transport.calls == [] and result.answers == {}
