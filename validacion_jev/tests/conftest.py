"""Shared fixtures: a scripted fake transport (no network, ever) and helpers."""
from __future__ import annotations

import json
import socket

import pytest

SENTINEL_KEY = "sk-SENTINEL-do-not-leak-9f3a7c"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Hard guarantee: any real socket use in tests fails loudly."""

    def _boom(*a, **k):  # pragma: no cover - only runs on a test bug
        raise AssertionError("network access attempted in tests")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)


class FakeTransport:
    """Scripted transport. Each script entry is a response tuple, an Exception
    instance (raised), or a callable(url, headers, body) returning one of those.
    When the script is exhausted the last entry repeats."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(
            {"url": url, "headers": dict(headers), "body": body, "timeout": timeout}
        )
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        item = self.script[idx]
        if callable(item) and not isinstance(item, tuple):
            item = item(url, headers, body)
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def bodies(self):
        return [json.loads(c["body"]) for c in self.calls]


def http(status, body, headers=None):
    if not isinstance(body, (bytes, bytearray)):
        body = json.dumps(body).encode("utf-8")
    return (status, dict(headers or {}), bytes(body))


def noul_answer(p):
    return {"type": "noul", "noul": p}


def choice_answer(choice, probs, confidence=0.9):
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": confidence}


def ok(answers, usage=(10, 5)):
    return http(
        200,
        {
            "model": "jev-latest",
            "answers": answers,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        },
    )


def echo_ok(p=0.97):
    """Transport callable answering every question sensibly based on its type."""

    def _respond(url, headers, body):
        req = json.loads(body)
        answers = {}
        for qid, q in req["questions"].items():
            if q["type"] == "noul":
                answers[qid] = noul_answer(p)
            elif q["type"] == "choice":
                opts = list(q["criteria"])
                probs = {o: 0.0 for o in opts}
                probs[opts[0]] = 1.0
                answers[qid] = choice_answer(opts[0], probs, 0.95)
        return ok(answers)

    return _respond


class Sleeper:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds):
        self.calls.append(seconds)


@pytest.fixture
def sleeper():
    return Sleeper()


@pytest.fixture
def config():
    from validacion_jev.client import JevConfig

    return JevConfig(api_key=SENTINEL_KEY)


def make_client(config, *script, sleeper=None):
    from validacion_jev.client import JevClient

    transport = FakeTransport(*script)
    client = JevClient(config, transport=transport, sleep=sleeper or Sleeper())
    return client, transport
