#!/usr/bin/env python3
"""Smoke test + latency measurement of a running service (stdlib only; invented addresses only).

    python scripts/medir_servicio.py --url http://127.0.0.1:8000 [--esperar 300] [--n 200] [--lote 1000]

Prints one JSON object: seconds until /health answered 200 (from this script's start), health body,
p50/p95 latency of single-address requests and rows/second of one batch request.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

VIAS = ("CALLE", "CARRERA", "AVENIDA", "DIAGONAL", "TRANSVERSAL")
EXAMPLES = ["CALLE 10 # 5 - 10", "Cra 11 #12-23", "AV 3N # 45-10", "Calle 5 con carrera 100", "Ñ ##"]


def synthetic(n: int) -> list[str]:
    return [f"{VIAS[i % 5]} {10 + i % 70} # {5 + (i * 7) % 90} - {10 + (i * 13) % 80}" for i in range(n)]


def _request(url: str, payload: dict | None = None, timeout: float = 120.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, body


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--esperar", type=float, default=300.0, help="max seconds to wait for /health = 200")
    ap.add_argument("--n", type=int, default=200, help="single-address requests for p50/p95")
    ap.add_argument("--lote", type=int, default=1000, help="rows of the batch request (<= MAX_ROWS)")
    args = ap.parse_args(argv)
    base = args.url.rstrip("/")
    started = time.monotonic()
    health = None
    while time.monotonic() - started < args.esperar:
        try:
            status, body = _request(base + "/health", timeout=5)
            if status == 200:
                health = body
                break
            if isinstance(body, dict) and body.get("status") == "error":
                print(json.dumps({"ok": False, "health": body}, ensure_ascii=False))
                return 1
        except OSError:
            pass
        time.sleep(0.5)
    ready_s = time.monotonic() - started
    if health is None:
        print(json.dumps({"ok": False, "error": f"/health no respondio 200 en {args.esperar} s"}))
        return 1

    status, body = _request(base + "/api/v1/normalize-address", {"addresses": EXAMPLES})
    if status != 200:
        print(json.dumps({"ok": False, "status": status, "body": body}, ensure_ascii=False))
        return 1
    examples = [{"entrada": a, "estado": r.get("estado"), "salida": r.get("direccion_normalizada")}
                for a, r in zip(EXAMPLES, body["results"])]

    pool = synthetic(max(args.n, 1))
    lat = []
    for addr in pool[: args.n]:
        t0 = time.perf_counter()
        status, _ = _request(base + "/api/v1/normalize-address", {"addresses": [addr]})
        lat.append((time.perf_counter() - t0) * 1000)
        if status != 200:
            print(json.dumps({"ok": False, "status": status}))
            return 1
    batch = synthetic(args.lote)
    t0 = time.perf_counter()
    status, body = _request(base + "/api/v1/normalize-address", {"addresses": batch}, timeout=600)
    batch_s = time.perf_counter() - t0
    estados: dict[str, int] = {}
    for r in body.get("results", []) if isinstance(body, dict) else []:
        estados[r.get("estado")] = estados.get(r.get("estado"), 0) + 1
    print(json.dumps({
        "ok": status == 200,
        "seconds_until_healthy": round(ready_s, 2),
        "health": health,
        "examples": examples,
        "single_requests": args.n,
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(percentile(lat, 0.95), 1),
        "batch_rows": args.lote,
        "batch_seconds": round(batch_s, 2),
        "batch_rows_per_s": round(args.lote / batch_s, 1),
        "batch_estados": estados,
    }, ensure_ascii=False, indent=2))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
