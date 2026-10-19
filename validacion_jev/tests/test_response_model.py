"""The model identifier reported by the API is recorded per row and in the summary."""
import json

from conftest import SENTINEL_KEY, FakeTransport, Sleeper, echo_ok, http, make_client, noul_answer, ok
from validacion_jev.client import JevClient, JevConfig
from validacion_jev.io import render_summary_md
from validacion_jev.metrics import summarize
from validacion_jev.validators import Thresholds, run_validation

STATE = {"direccion_a": "CL 5 # 10-20", "direccion_b": "CALLE 5 10 20"}
NOUL_Q = {"type": "noul", "instructions": "x"}


def _resp(model, p=0.9):
    body = {"answers": {"a": noul_answer(p)}, "usage": {"input_tokens": 1, "output_tokens": 1}}
    if model is not ...:
        body["model"] = model
    return http(200, body)


def test_model_string_is_recorded(config):
    client, _ = make_client(config, _resp("jev-2026-09-01"))
    assert client.query(STATE, {"a": dict(NOUL_Q)}).models == ["jev-2026-09-01"]


def test_model_absent_gives_empty_list(config):
    client, _ = make_client(config, _resp(...))
    assert client.query(STATE, {"a": dict(NOUL_Q)}).models == []


def test_model_non_string_or_blank_is_ignored(config):
    for bad in (None, 7, True, "", "   ", ["x"], {"v": 1}):
        client, _ = make_client(config, _resp(bad))
        assert client.query(STATE, {"a": dict(NOUL_Q)}).models == [], bad


def test_model_string_is_truncated_and_stripped(config):
    client, _ = make_client(config, _resp("  " + "m" * 500 + "  "))
    models = client.query(STATE, {"a": dict(NOUL_Q)}).models
    assert models == ["m" * 100]


def test_model_never_contains_the_api_key(config):
    client, _ = make_client(config, _resp(f"echo-{SENTINEL_KEY}"))
    models = client.query(STATE, {"a": dict(NOUL_Q)}).models
    assert all(SENTINEL_KEY not in m for m in models)


def test_models_deduplicated_across_chunks(config):
    qs = {f"q{i}": dict(NOUL_Q) for i in range(30)}

    def respond(url, headers, body):
        req = json.loads(body)
        return http(200, {"model": "jev-x", "answers": {q: noul_answer(0.9) for q in req["questions"]}})

    client, transport = make_client(config, respond)
    result = client.query(STATE, qs)
    assert len(transport.calls) > 1
    assert result.models == ["jev-x"]


def test_failed_chunk_contributes_no_model(config):
    client, _ = make_client(config, http(400, {"error": "bad"}))
    assert client.query(STATE, {"a": dict(NOUL_Q)}).models == []


def _item():
    from validacion_jev.io import Item

    return Item(id="1", a="CL 5 # 10-20", b="CALLE 5 10 20", decision=True)


def test_run_validation_row_carries_models():
    t = FakeTransport(echo_ok(0.97))
    client = JevClient(JevConfig(api_key=SENTINEL_KEY), transport=t, sleep=Sleeper())
    r = run_validation("matcher", [_item()], client, Thresholds())[0]
    assert r["models"] == ["jev-latest"]


def test_summary_lists_models_seen_and_md_mentions_them():
    rows = [
        {"id": "1", "status": "ok", "decision": True, "jev_p": 0.99, "label": "yes", "score": None,
         "a": "a", "b": "b", "gates": {}, "usage": {}, "flags": [], "models": ["jev-b"]},
        {"id": "2", "status": "ok", "decision": True, "jev_p": 0.99, "label": "yes", "score": None,
         "a": "a", "b": "b", "gates": {}, "usage": {}, "flags": [], "models": ["jev-a", "jev-b"]},
        {"id": "3", "status": "error", "decision": True, "jev_p": None, "label": None, "score": None,
         "a": "a", "b": "b", "gates": {}, "usage": {}, "flags": ["error"]},
    ]
    s = summarize(rows, "matcher", Thresholds())
    assert s["models_seen"] == ["jev-a", "jev-b"]
    assert "jev-a" in render_summary_md(s)


def test_summary_models_seen_empty_when_never_reported():
    s = summarize([], "matcher", Thresholds())
    assert s["models_seen"] == []
