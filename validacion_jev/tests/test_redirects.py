"""Redirects must never be followed: urllib would resend the Authorization header.

These tests drive the REAL urllib opener chain (redirect handler, error
processor) over fake HTTP/HTTPS handlers, so no socket is ever opened.
"""
import email.message
import io
import urllib.request
import urllib.response

import pytest

from conftest import SENTINEL_KEY, http, make_client
from validacion_jev import client as c

_REAL_HTTPS = urllib.request.HTTPSHandler
_REAL_HTTP = urllib.request.HTTPHandler

QUESTIONS = {"q": {"type": "noul", "instructions": "i"}}
REDIRECT_CODES = [301, 302, 303, 307, 308]
LOCATIONS = {
    "other_host": "https://other.example.org/v1/systemone",
    "downgrade": "http://api.typesafe.ai/v1/systemone",
    "same_host": "https://api.typesafe.ai/elsewhere",
    "relative": "/v1/other",
}


def _response(url, code, location=None, body=b"redirect body"):
    headers = email.message.Message()
    if location is not None:
        headers["Location"] = location
    resp = urllib.response.addinfourl(io.BytesIO(body), headers, url, code)
    resp.msg = "scripted"
    return resp


@pytest.fixture
def wire(monkeypatch):
    """Replace both URL handlers; record every request that would hit the wire."""
    state = {"seen": [], "code": 200, "location": None}

    def record(req):
        state["seen"].append((req.full_url, dict(req.header_items())))
        return _response(req.full_url, state["code"], state["location"])

    class FakeHTTPS(_REAL_HTTPS):
        def https_open(self, req):
            return record(req)

    class FakeHTTP(_REAL_HTTP):
        def http_open(self, req):
            return record(req)

    monkeypatch.setattr(urllib.request, "HTTPSHandler", FakeHTTPS)
    monkeypatch.setattr(urllib.request, "HTTPHandler", FakeHTTP)
    return state


HEADERS = {"Authorization": f"Bearer {SENTINEL_KEY}", "Content-Type": "application/json"}
URL = "https://api.typesafe.ai/v1/systemone"


def test_baseline_200_still_works(wire):
    status, _, body = c.urllib_transport(URL, HEADERS, b"{}", 5.0)
    assert status == 200 and body == b"redirect body"
    assert len(wire["seen"]) == 1


@pytest.mark.parametrize("code", REDIRECT_CODES)
@pytest.mark.parametrize("target", sorted(LOCATIONS))
def test_transport_does_not_follow_redirect(wire, code, target):
    wire["code"], wire["location"] = code, LOCATIONS[target]
    status, headers, _ = c.urllib_transport(URL, HEADERS, b"{}", 5.0)
    assert status == code
    assert headers["location"] == LOCATIONS[target]
    # exactly ONE request, to the original URL: the key was never resent anywhere else
    assert [u for u, _ in wire["seen"]] == [URL]


@pytest.mark.parametrize("code", REDIRECT_CODES)
@pytest.mark.parametrize("target", sorted(LOCATIONS))
def test_client_reports_redirect_without_retry_or_key(wire, code, target, config, sleeper):
    wire["code"], wire["location"] = code, LOCATIONS[target]
    client = c.JevClient(config, sleep=sleeper)  # default transport on purpose
    result = client.query({"s": 1}, QUESTIONS)
    assert result.answers == {}
    message = result.errors["q"]
    assert f"HTTP {code}" in message and "redirecc" in message.lower()
    assert SENTINEL_KEY not in message
    assert len(wire["seen"]) == 1 and sleeper.calls == []  # not retried


def test_error_names_only_the_host_never_the_full_location(wire, config):
    wire["code"] = 302
    wire["location"] = f"https://evil.example/collect?k={SENTINEL_KEY}&x=1"
    result = c.JevClient(config).query({"s": 1}, QUESTIONS)
    message = result.errors["q"]
    assert SENTINEL_KEY not in message and "collect" not in message
    assert "evil.example" in message


def test_redirect_without_location_header_is_still_a_clear_error(wire, config):
    wire["code"], wire["location"] = 302, None
    result = c.JevClient(config).query({"s": 1}, QUESTIONS)
    assert "HTTP 302" in result.errors["q"] and "redirecc" in result.errors["q"].lower()


@pytest.mark.parametrize("code", [300, 304])
def test_other_3xx_are_refused_too(wire, code, config):
    wire["code"], wire["location"] = code, LOCATIONS["other_host"]
    result = c.JevClient(config).query({"s": 1}, QUESTIONS)
    assert f"HTTP {code}" in result.errors["q"]
    assert len(wire["seen"]) == 1


@pytest.mark.parametrize("code", REDIRECT_CODES)
def test_injected_transport_3xx_is_refused_without_retry(config, sleeper, code):
    client, transport = make_client(
        config, http(code, b"", {"location": "https://other.example.org/"}), sleeper=sleeper
    )
    result = client.query({"s": 1}, QUESTIONS)
    assert len(transport.calls) == 1 and sleeper.calls == []
    assert "redirecc" in result.errors["q"].lower()
    assert SENTINEL_KEY not in result.errors["q"]
    assert "other.example.org" in result.errors["q"]


def test_malformed_location_does_not_crash(config):
    client, _ = make_client(config, http(302, b"", {"location": "http://[bad"}))
    result = client.query({"s": 1}, QUESTIONS)
    assert "HTTP 302" in result.errors["q"]


def test_refuse_handler_returns_none():
    handler = c._RefuseRedirects()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://x/") is None
