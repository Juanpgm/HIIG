"""``python -m servicio [serve|check]``.

Exit codes: 0 ok, 2 bad configuration, 3 missing or incompatible model/index files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .bundle import (
    BundleError,
    assemble_runtime,
    check_index,
    check_model_dir,
    ensure_index,
    resolve_backend,
)
from .config import ConfigError, load_settings, resolve_threads

EXIT_CONFIG = 2
EXIT_FILES = 3
#: ``<root>/docs`` next to ``src/`` (``/app/docs`` in the image): ships API.md for /docs-integracion.
DEFAULT_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")


def _log(message: str) -> None:
    print(f"[servicio] {message}", file=sys.stderr, flush=True)


def prepare(environ: dict | None = None) -> dict:
    """Validate everything and build the runtime directory. Return a summary; raise on failure.

    Side effect: exports the variables the wrapped API reads (``ARTIFACTS_DIR``, ``BASEMAPS_DIR``,
    ``NORMALIZER_BACKEND``, thread counts...) into ``os.environ``.
    """
    env = os.environ if environ is None else environ
    settings = load_settings(env)
    for line in settings.warnings:
        _log(f"aviso: {line}")
    threads = resolve_threads(settings)
    # numpy's BLAS (document search) sizes its pool from the host core count the FIRST time numpy is
    # imported, ignoring the container CPU quota; dozens of spinning threads under a 4-CPU quota made
    # each query ~20x slower (measured). Export the limit BEFORE anything below imports numpy.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if not str(env.get(name, "")).strip():
            os.environ[name] = str(threads)
    backend, notes = resolve_backend(settings.backend, settings.model_dir, settings.backend_fallback)
    for line in notes:
        _log(f"aviso: {line}")
    meta = check_model_dir(settings.model_dir, backend)
    for line in ensure_index(settings.index_dir, settings.index_url, settings.index_sha256,
                             settings.download_timeout_s, int(settings.download_max_mb * 1024 * 1024),
                             settings.allow_file_url):
        _log(line)
    for line in check_index(settings.index_dir, meta):
        _log(f"aviso: {line}")
    artifacts, basemaps = assemble_runtime(
        settings.runtime_dir, settings.model_dir, settings.index_dir, settings.basemaps_dir)
    if not all(os.path.isfile(os.path.join(basemaps, n)) for n in ("barrios_veredas.geojson",
                                                                    "comunas_corregimientos.geojson")):
        _log(f"aviso: no hay mapas base IDESC en BASEMAPS_DIR={settings.basemaps_dir}; "
             "el servicio funciona sin gazetteer (sin barrio/comuna por poligono)")
    exports = {
        "ARTIFACTS_DIR": artifacts,
        "BASEMAPS_DIR": basemaps,
        "DOCS_DIR": os.environ.get("DOCS_DIR") or DEFAULT_DOCS_DIR,
        "NORMALIZER_BACKEND": backend,
        "NORMALIZER_DEVICE": "cpu",
        "NORMALIZER_ONNX_THREADS": str(threads),
        "NORMALIZER_ONNX_INDEX": settings.index_precision,
        "MAX_UPLOAD_MB": str(settings.max_upload_mb),
        "MAX_ROWS": str(settings.max_rows),
        "WARMUP_TIMEOUT_S": str(settings.warmup_timeout_s),
    }
    os.environ.update(exports)
    return {
        "backend": backend,
        "threads": threads,
        "index_precision": settings.index_precision,
        "model_dir": settings.model_dir,
        "index_dir": settings.index_dir,
        "runtime_artifacts": artifacts,
        "host": settings.host,
        "port": settings.port,
        "workers": settings.workers,
        "log_level": settings.log_level,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m servicio", description="Normalizador de direcciones (ONNX)")
    parser.add_argument("command", nargs="?", default="serve", choices=("serve", "check"))
    args = parser.parse_args(argv)
    try:
        summary = prepare()
    except ConfigError as exc:
        _log(f"ERROR de configuracion: {exc}")
        return EXIT_CONFIG
    except BundleError as exc:
        _log(f"ERROR: {exc}")
        return EXIT_FILES
    _log("configuracion: " + json.dumps(summary, ensure_ascii=False))
    if args.command == "check":
        return 0
    import uvicorn

    uvicorn.run(
        "servicio.app:app",
        host=summary["host"],
        port=summary["port"],
        workers=summary["workers"],
        log_level=summary["log_level"],
        proxy_headers=True,  # trusted proxies: uvicorn's FORWARDED_ALLOW_IPS (default 127.0.0.1)
        access_log=summary["log_level"] in ("debug", "trace"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
