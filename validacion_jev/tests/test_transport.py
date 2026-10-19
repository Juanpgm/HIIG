import io
import urllib.error
import urllib.request

import pytest

from validacion_jev import client as c


class FakeOpener:
    """Stands in for the opener built by ``client._build_opener``."""

    def __init__(self, fn):
        self._fn = fn

    def open(self, req, timeout=None):
        return self._fn(req, timeout=timeout)


def patch_opener(monkeypatch, fn):
    monkeypatch.setattr(c, "_build_opener", lambda context: FakeOpener(fn))


class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_success(monkeypatch):
    seen = {}

    def fake_open(req, timeout=None):
        seen["req"], seen["timeout"] = req, timeout
        return FakeResp(200, b"{}", {"X-A": "1"})

    patch_opener(monkeypatch, fake_open)
    status, headers, body = c.urllib_transport("https://h/x", {"A": "b"}, b"{}", 5.0)
    assert (status, body) == (200, b"{}")
    assert headers["x-a"] == "1"
    assert seen["timeout"] == 5.0 and seen["req"].get_method() == "POST"


def test_http_error_becomes_response(monkeypatch):
    def fake_open(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://h", 429, "slow", {"Retry-After": "3"}, io.BytesIO(b'{"e":1}')
        )

    patch_opener(monkeypatch, fake_open)
    status, headers, body = c.urllib_transport("https://h", {}, b"", 1.0)
    assert status == 429 and headers["retry-after"] == "3" and body == b'{"e":1}'


def test_url_error_propagates_as_oserror(monkeypatch):
    def fake_open(req, timeout=None):
        raise urllib.error.URLError("dns")

    patch_opener(monkeypatch, fake_open)
    with pytest.raises(OSError):
        c.urllib_transport("https://h", {}, b"", 1.0)


def test_body_over_32_mib_refused():
    from validacion_jev.client import JevApiError, JevClient, JevConfig

    cl = JevClient(JevConfig(api_key="k"), transport=lambda *a: (200, {}, b"{}"))
    huge = {"direccion_a": "x" * (33 * 1024 * 1024)}
    result = cl.query(huge, {"q": {"type": "noul", "instructions": "i"}})
    assert "q" in result.errors and "32" in result.errors["q"]
