"""End-to-end HTTP tests on the real ONNX model with a tiny synthetic index (no Docker, no real data)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import threading
import time

import pytest

from conftest import MODEL_DIR, requires_model

pytestmark = requires_model

SYNTH = ["CALLE 10 # 5 - 10", "carrera 11 #12-23", "AVENIDA 12 # 19 - 36"]


@pytest.fixture(scope="module")
def client(synthetic_bundle, tmp_path_factory):
    from fastapi.testclient import TestClient

    mp = pytest.MonkeyPatch()
    rt = tmp_path_factory.mktemp("runtime")
    for k, v in {
        "MODEL_DIR": MODEL_DIR, "INDEX_DIR": synthetic_bundle, "RUNTIME_DIR": str(rt),
        "BASEMAPS_DIR": str(rt / "sin_basemaps"), "NORMALIZER_ONNX_THREADS": "2",
        "MAX_ROWS": "50", "MAX_UPLOAD_MB": "0.05",
    }.items():
        mp.setenv(k, v)
    from servicio.__main__ import prepare

    prepare()
    from servicio.app import app

    with TestClient(app) as c:
        deadline = time.time() + 120
        while time.time() < deadline:
            r = c.get("/health")
            if r.status_code == 200 or r.json().get("status") == "error":
                break
            time.sleep(0.2)
        yield c
    mp.undo()


def _norm(client, addresses):
    return client.post("/api/v1/normalize-address", json={"addresses": addresses})


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["catastro_size"] == 120 and body["device"] == "cpu"
    assert body["gazetteer_loaded"] is False  # no basemaps mounted


def test_torch_never_imported(client):
    assert "torch" not in sys.modules


def test_normalize_synthetic(client):
    r = _norm(client, SYNTH)
    assert r.status_code == 200, r.text
    body = r.json()
    rows = body.get("results") or body.get("rows") or body.get("records")
    assert rows is not None and len(rows) == len(SYNTH)
    assert all("estado" in row for row in rows)


def test_exact_document_matches(client):
    body = _norm(client, ["CALLE 10 # 5 - 10"]).json()
    rows = body.get("results") or body.get("rows") or body.get("records")
    assert rows[0]["estado"] in ("OK", "SIN_MATCH", "NO_PARSEABLE")


def test_empty_list(client):
    r = _norm(client, [])
    assert r.status_code in (200, 422)


@pytest.mark.parametrize("value", ["", "   ", None, "###", "😀😀", "x" * 5000])
def test_degenerate_inputs_never_500(client, value):
    r = _norm(client, [value])
    assert r.status_code in (200, 422), r.text


def test_non_ascii_input(client):
    r = _norm(client, ["Cálle 10 Nº 5-10 Ñuñoa café", "CARRERA 11 # 12 - 23 ÁÉÍÓÚ"])
    assert r.status_code == 200, r.text


def test_row_cap(client):
    r = _norm(client, ["CALLE 10 # 5 - 10"] * 51)
    assert r.status_code == 422 and "MAX_ROWS" in r.text


def test_row_cap_boundary_accepted(client):
    assert _norm(client, ["CALLE 10 # 5 - 10"] * 50).status_code == 200


def test_oversize_body_rejected(client):
    big = {"addresses": ["CALLE 10 # 5 - 10 " + "A" * 200] * 400}
    r = client.post("/api/v1/normalize-address", content=json.dumps(big),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"] == "payload_too_large"


@pytest.mark.parametrize("payload", ["{not json", "[]", '{"addresses": "CALLE 1"}', '{"addresses": [1, 2]}', "{}"])
def test_malformed_json_is_422(client, payload):
    r = client.post("/api/v1/normalize-address", content=payload, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_normalize_json_needs_exactly_one_source(client):
    r = client.post("/api/v1/normalize-json", json={"addresses": SYNTH, "records": [{"d": "x"}]})
    assert r.status_code == 422


def test_concurrent_requests_match_sequential(client):
    expected = _norm(client, SYNTH).json()
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _norm(client, SYNTH), range(16)))
    assert all(r.status_code == 200 for r in results)
    assert all(r.json() == expected for r in results)


def test_unknown_route_404(client):
    assert client.get("/api/v1/no-existe").status_code == 404


# -- readiness gate -----------------------------------------------------------------------------
def test_gate_answers_503_while_loading():
    from servicio.app import ReadinessGate

    class _State:
        engine = type("E", (), {"ready": threading.Event()})()

    class _App:
        state = _State()

    sent = []

    async def inner(scope, receive, send):  # pragma: no cover - must not be reached
        raise AssertionError("inner app called while loading")

    async def send(msg):
        sent.append(msg)

    import asyncio

    gate = ReadinessGate(inner)
    asyncio.run(gate({"type": "http", "path": "/api/v1/normalize-address", "app": _App()}, None, send))
    assert sent[0]["status"] == 503 and (b"retry-after", b"5") in sent[0]["headers"]


def test_gate_lets_health_through_while_loading():
    from servicio.app import ReadinessGate

    class _App:
        class state:  # noqa: N801
            engine = type("E", (), {"ready": threading.Event()})()

    called = []

    async def inner(scope, receive, send):
        called.append(scope["path"])

    import asyncio

    gate = ReadinessGate(inner)
    for path in ("/health", "/", "/static/style.css", "/openapi.json"):
        asyncio.run(gate({"type": "http", "path": path, "app": _App()}, None, None))
    assert called == ["/health", "/", "/static/style.css", "/openapi.json"]


def test_gate_fails_closed_when_engine_is_absent():
    """No engine at all (before lifespan / after shutdown / app without state) must be a 503, not a pass-through."""
    import asyncio

    from servicio.app import ReadinessGate

    class _NoEngine:
        class state:  # noqa: N801
            engine = None

    class _NoState:
        pass

    async def inner(scope, receive, send):  # pragma: no cover - must not be reached
        raise AssertionError("inner app called without an engine")

    for app_obj in (_NoEngine(), _NoState(), None):
        sent = []

        async def send(msg, sent=sent):
            sent.append(msg)

        asyncio.run(ReadinessGate(inner)({"type": "http", "path": "/api/v1/normalize-address", "app": app_obj},
                                         None, send))
        assert sent[0]["status"] == 503 and (b"retry-after", b"5") in sent[0]["headers"]
    # a missing "app" key in the scope behaves the same
    sent = []

    async def send2(msg):
        sent.append(msg)

    asyncio.run(ReadinessGate(inner)({"type": "http", "path": "/api/v1/normalize-json"}, None, send2))
    assert sent[0]["status"] == 503


def test_gate_still_lets_health_through_without_engine():
    import asyncio

    from servicio.app import ReadinessGate

    called = []

    async def inner(scope, receive, send):
        called.append(scope["path"])

    asyncio.run(ReadinessGate(inner)({"type": "http", "path": "/health"}, None, None))
    assert called == ["/health"]


def test_gate_passes_when_engine_is_ready():
    import asyncio

    from servicio.app import ReadinessGate

    ready = threading.Event()
    ready.set()

    class _App:
        class state:  # noqa: N801
            engine = type("E", (), {"ready": ready})()

    called = []

    async def inner(scope, receive, send):
        called.append(scope["path"])

    asyncio.run(ReadinessGate(inner)({"type": "http", "path": "/api/v1/normalize-address", "app": _App()},
                                     None, None))
    assert called == ["/api/v1/normalize-address"]


def test_docs_integracion_is_served_from_shipped_api_md(client):
    r = client.get("/docs-integracion")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_docs_integracion_without_api_md_is_404_not_500(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "sin_docs"))
    r = client.get("/docs-integracion")
    assert r.status_code == 404 and "API.md" in r.text
