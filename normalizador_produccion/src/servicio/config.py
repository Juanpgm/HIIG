"""Environment configuration of the production service, validated up front.

The wrapped API (``cali_address.api.main``) silently falls back to defaults on malformed values;
this module rejects them before anything is loaded so a typo fails the deploy with a clear message.
Every variable is documented in GUIA_DESPLIEGUE.md.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Mapping

__all__ = ["ConfigError", "Settings", "load_settings", "detect_cpu_limit", "resolve_threads"]

BACKENDS = ("onnx", "onnx-int8")
INDEX_PRECISIONS = ("fp32", "fp16")
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")
_TRUE = {"1", "true", "yes", "on", "si"}
_FALSE = {"0", "false", "no", "off"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

DEFAULTS = {
    "HOST": "0.0.0.0",
    "PORT": "8000",
    "NORMALIZER_BACKEND": "onnx",
    "NORMALIZER_BACKEND_FALLBACK": "true",
    "NORMALIZER_ONNX_THREADS": "auto",
    "NORMALIZER_ONNX_INDEX": "fp32",
    "UVICORN_WORKERS": "1",
    "LOG_LEVEL": "info",
    "MODEL_DIR": "/app/modelo",
    "INDEX_DIR": "/data/indice",
    "BASEMAPS_DIR": "/data/basemaps",
    "RUNTIME_DIR": "/tmp/normalizador",
    "INDEX_URL": "",
    "INDEX_SHA256": "",
    "INDEX_DOWNLOAD_TIMEOUT_S": "600",
    "INDEX_MAX_DOWNLOAD_MB": "4096",
    "INDEX_ALLOW_FILE_URL": "false",
    "MAX_UPLOAD_MB": "25",
    "MAX_ROWS": "20000",
    "WARMUP_TIMEOUT_S": "300",
}


class ConfigError(ValueError):
    """An environment variable has a value the service cannot use."""


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    backend: str
    backend_fallback: bool
    threads: int | None  # None = auto (resolved from the CPU limit at start)
    index_precision: str
    workers: int
    log_level: str
    model_dir: str
    index_dir: str
    basemaps_dir: str
    runtime_dir: str
    index_url: str
    index_sha256: str
    download_timeout_s: float
    download_max_mb: float
    allow_file_url: bool
    max_upload_mb: float
    max_rows: int
    warmup_timeout_s: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _get(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value is None or str(value).strip() == "":
        return DEFAULTS[name]
    return str(value).strip()


def _int(env, name: str, lo: int, hi: int) -> int:
    raw = _get(env, name)
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} no es un entero (rango {lo}..{hi})") from None
    if not lo <= value <= hi:
        raise ConfigError(f"{name}={value} esta fuera de rango ({lo}..{hi})")
    return value


def _float(env, name: str, lo: float, hi: float) -> float:
    raw = _get(env, name)
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} no es un numero (rango {lo}..{hi})") from None
    if not math.isfinite(value) or not lo < value <= hi:
        raise ConfigError(f"{name}={raw} esta fuera de rango (mayor que {lo} y hasta {hi})")
    return value


def _bool(env, name: str) -> bool:
    raw = _get(env, name).lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"{name}={raw!r} debe ser true o false")


def _choice(env, name: str, options: tuple[str, ...]) -> str:
    raw = _get(env, name).lower()
    if raw not in options:
        raise ConfigError(f"{name}={raw!r} no es valido; opciones: {', '.join(options)}")
    return raw


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Parse and validate the environment; raise :class:`ConfigError` on the first bad value."""
    env = os.environ if env is None else env
    warnings: list[str] = []

    raw_backend = _get(env, "NORMALIZER_BACKEND").lower()
    if raw_backend == "torch":
        raise ConfigError(
            "NORMALIZER_BACKEND=torch no esta disponible: esta imagen no incluye torch. "
            "Use onnx (recomendado) u onnx-int8."
        )
    backend = _choice(env, "NORMALIZER_BACKEND", BACKENDS)

    raw_threads = _get(env, "NORMALIZER_ONNX_THREADS").lower()
    if raw_threads in ("auto", "0"):
        threads = None
    else:
        threads = _int(env, "NORMALIZER_ONNX_THREADS", 1, 256)

    index_url = _get(env, "INDEX_URL")
    index_sha = _get(env, "INDEX_SHA256").lower()
    allow_file_url = _bool(env, "INDEX_ALLOW_FILE_URL")
    if index_url:
        if index_url.startswith("file://"):
            if not allow_file_url:
                raise ConfigError(
                    "INDEX_URL con file:// esta deshabilitado por seguridad; use https:// "
                    "(o INDEX_ALLOW_FILE_URL=true solo en pruebas locales)"
                )
        elif not index_url.startswith("https://"):
            raise ConfigError("INDEX_URL debe empezar por https://")
        if not _SHA256.match(index_sha):
            raise ConfigError(
                "INDEX_URL requiere INDEX_SHA256 (64 caracteres hexadecimales) para verificar la descarga"
            )
    elif index_sha:
        if not _SHA256.match(index_sha):
            raise ConfigError("INDEX_SHA256 debe tener 64 caracteres hexadecimales")
        warnings.append("INDEX_SHA256 esta definido pero INDEX_URL no: se ignora")

    host = _get(env, "HOST")
    if any(c.isspace() for c in host):
        raise ConfigError(f"HOST={host!r} no puede contener espacios")

    workers = _int(env, "UVICORN_WORKERS", 1, 16)
    if workers > 1:
        warnings.append(
            f"UVICORN_WORKERS={workers}: cada proceso carga su propia copia del indice (unos 2 GB cada uno)"
        )

    return Settings(
        host=host,
        port=_int(env, "PORT", 1, 65535),
        backend=backend,
        backend_fallback=_bool(env, "NORMALIZER_BACKEND_FALLBACK"),
        threads=threads,
        index_precision=_choice(env, "NORMALIZER_ONNX_INDEX", INDEX_PRECISIONS),
        workers=workers,
        log_level=_choice(env, "LOG_LEVEL", LOG_LEVELS),
        model_dir=os.path.abspath(_get(env, "MODEL_DIR")),
        index_dir=os.path.abspath(_get(env, "INDEX_DIR")),
        basemaps_dir=os.path.abspath(_get(env, "BASEMAPS_DIR")),
        runtime_dir=os.path.abspath(_get(env, "RUNTIME_DIR")),
        index_url=index_url,
        index_sha256=index_sha,
        download_timeout_s=_float(env, "INDEX_DOWNLOAD_TIMEOUT_S", 0, 86400),
        download_max_mb=_float(env, "INDEX_MAX_DOWNLOAD_MB", 0, 1_048_576),
        allow_file_url=allow_file_url,
        max_upload_mb=_float(env, "MAX_UPLOAD_MB", 0, 1024),
        max_rows=_int(env, "MAX_ROWS", 1, 1_000_000),
        warmup_timeout_s=_float(env, "WARMUP_TIMEOUT_S", 0, 3600),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------------------------
# CPU limit (containers): onnxruntime and OpenBLAS size their pools from the host core count, which
# oversubscribes a container limited by a cgroup quota. "auto" uses the quota instead.
# ---------------------------------------------------------------------------------------------
def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="ascii") as fh:
            return fh.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def detect_cpu_limit(cgroup_root: str = "/sys/fs/cgroup") -> int:
    """CPUs this process may use: cgroup v2 / v1 quota, else the affinity mask, else ``os.cpu_count()``."""
    host = None
    if hasattr(os, "sched_getaffinity"):
        try:
            host = len(os.sched_getaffinity(0))
        except OSError:
            host = None
    host = host or os.cpu_count() or 1

    quota = None
    v2 = _read(os.path.join(cgroup_root, "cpu.max"))
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                q, p = int(parts[0]), int(parts[1])
                if q > 0 and p > 0:
                    quota = q / p
            except ValueError:
                quota = None
    else:
        q = _read(os.path.join(cgroup_root, "cpu", "cpu.cfs_quota_us")) or _read(
            os.path.join(cgroup_root, "cpu.cfs_quota_us"))
        p = _read(os.path.join(cgroup_root, "cpu", "cpu.cfs_period_us")) or _read(
            os.path.join(cgroup_root, "cpu.cfs_period_us"))
        try:
            if q is not None and p is not None and int(q) > 0 and int(p) > 0:
                quota = int(q) / int(p)
        except ValueError:
            quota = None
    if quota is None:
        return max(1, host)
    return max(1, min(host, math.ceil(quota)))


def resolve_threads(settings: Settings, cgroup_root: str = "/sys/fs/cgroup") -> int:
    """Explicit ``NORMALIZER_ONNX_THREADS`` or the detected CPU limit."""
    return settings.threads if settings.threads is not None else detect_cpu_limit(cgroup_root)
