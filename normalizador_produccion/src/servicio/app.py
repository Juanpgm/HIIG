"""ASGI application: ``cali_address.api.main:app`` plus a readiness gate.

The wrapped endpoints are ``async def`` functions that block on the model load
(``threading.Event.wait``) inside the event loop, so one early request would freeze ``/health`` for the
whole warm-up. The gate answers 503 immediately for every non-health route until the model is loaded. It fails
closed: when the engine is absent altogether (before the lifespan started, after shutdown, or an app
without state) the answer is 503 as well, never a pass-through to an uninitialised handler.
"""

from __future__ import annotations

import json

from cali_address.api.main import app

__all__ = ["app", "ReadinessGate"]

#: Routes served while loading (cheap, never touch the model).
OPEN_PATHS = ("/health", "/", "/favicon.ico", "/docs", "/redoc", "/openapi.json", "/docs-integracion")
OPEN_PREFIXES = ("/static/", "/docs/")

_BODY = json.dumps(
    {"error": "loading", "message": "el modelo aun se esta cargando; reintente en unos segundos"}
).encode("utf-8")


class ReadinessGate:
    def __init__(self, inner) -> None:
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path not in OPEN_PATHS and not path.startswith(OPEN_PREFIXES):
                engine = getattr(getattr(scope.get("app"), "state", None), "engine", None)
                if engine is None or not engine.ready.is_set():
                    await send({
                        "type": "http.response.start",
                        "status": 503,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"retry-after", b"5"),
                            (b"content-length", str(len(_BODY)).encode("ascii")),
                        ],
                    })
                    await send({"type": "http.response.body", "body": _BODY})
                    return
        await self.inner(scope, receive, send)


app.add_middleware(ReadinessGate)
