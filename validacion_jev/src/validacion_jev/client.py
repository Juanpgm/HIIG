"""Minimal JEV (TypeSafe System One) client. Standard library only.

Design notes:
- The transport is injectable so tests never touch the network.
- The API key lives only in ``JevConfig.api_key`` (hidden from repr/str) and in
  the Authorization header handed to the transport. Every error text that can
  reach the caller is passed through ``_redact``.
- Redirects are never followed (the default opener refuses them), so the key is
  never resent to another host or downgraded to http; a 3xx is an error.
"""
from __future__ import annotations

import json
import math
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
ENDPOINT_PATH = "/v1/systemone"
KEY_VARS = ("JEV_API_KEY", "TYPESAFE_API_KEY")
MAX_QUESTIONS_PER_REQUEST = 12
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_RETRY_DELAY = 60.0
PROB_TOLERANCE = 1e-3
RETRYABLE_STATUSES = frozenset({408, 429})
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ERROR_DETAIL_CHARS = 300
MODEL_ID_CHARS = 100  # the reported model identifier is truncated to this length


class JevError(Exception):
    """Base class for all errors raised by this package's client."""


class ConfigurationError(JevError):
    """Missing or invalid configuration (key, base URL, env file)."""


class AuthenticationError(ConfigurationError):
    """The API rejected the key (HTTP 401). Never retried."""


class JevApiError(JevError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class TransportError(JevError):
    """Connection error or timeout after exhausting retries."""


class JevResponseError(JevError):
    """The response (or one answer) does not match the documented contract."""


# --------------------------------------------------------------------------- config


def parse_env_file(path: str) -> dict[str, str]:
    """Tiny dotenv parser that returns ONLY the names in ``KEY_VARS``."""
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8-sig")
    except OSError as exc:
        raise ConfigurationError(f"No se pudo leer el archivo de entorno '{path}' ({type(exc).__name__}).") from None
    except UnicodeDecodeError:
        raise ConfigurationError(f"El archivo de entorno '{path}' no es UTF-8 valido.") from None
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or name not in KEY_VARS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        found[name] = value
    return found


def resolve_key(environ: Mapping[str, str], file_values: Mapping[str, str] | None = None) -> str:
    """Process environment first (JEV_API_KEY, then TYPESAFE_API_KEY), then env file."""
    for source in (environ, file_values or {}):
        for name in KEY_VARS:
            value = (source.get(name) or "").strip()
            if value:
                return value
    raise ConfigurationError(
        "No se encontro la clave de API. Defina la variable de entorno JEV_API_KEY "
        "(o TYPESAFE_API_KEY), o use --env-file."
    )


@dataclass(frozen=True, repr=False)
class JevConfig:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ConfigurationError("La clave de API esta vacia.")
        parsed = urllib.parse.urlparse(self.base_url or "")
        host = (parsed.hostname or "").lower()
        if not (parsed.scheme == "https" or (parsed.scheme == "http" and host in LOCAL_HOSTS)) or not host:
            raise ConfigurationError("JEV_BASE_URL debe ser una URL https (http solo para localhost).")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def endpoint(self) -> str:
        return self.base_url + ENDPOINT_PATH

    @classmethod
    def from_env(cls, environ: Mapping[str, str], env_file: str | None = None) -> "JevConfig":
        file_values = parse_env_file(env_file) if env_file else None
        key = resolve_key(environ, file_values)
        base_url = environ.get("JEV_BASE_URL")
        model = (environ.get("JEV_MODEL") or "").strip() or DEFAULT_MODEL
        return cls(api_key=key, base_url=DEFAULT_BASE_URL if base_url is None else base_url, model=model)

    def __repr__(self) -> str:
        return f"JevConfig(base_url={self.base_url!r}, model={self.model!r}, api_key=<redacted>)"

    __str__ = __repr__


# ------------------------------------------------------------------------- transport

Response = tuple[int, Mapping[str, str], bytes]
Transport = Callable[[str, Mapping[str, str], bytes, float], Response]


def _lower(headers: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in headers.items()}


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: urllib would resend the Authorization header.

    Returning ``None`` makes urllib raise ``HTTPError`` for the 3xx response
    instead, which ``urllib_transport`` hands back as a plain response.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _build_opener(context: ssl.SSLContext) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RefuseRedirects, urllib.request.HTTPSHandler(context=context))


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> Response:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    opener = _build_opener(ssl.create_default_context())
    try:
        with opener.open(request, timeout=timeout) as resp:
            return resp.status, _lower(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, _lower(exc.headers or {}), exc.read()


def _redirect_host(headers: Mapping[str, str]) -> str:
    """Host of the Location header only (never the path or query, which could carry secrets)."""
    location = {str(k).lower(): str(v) for k, v in headers.items()}.get("location", "")
    try:
        host = urllib.parse.urlparse(location).hostname
    except ValueError:
        host = None
    return (host or "desconocido")[:80]


# --------------------------------------------------------------------- validation


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _unit(x: Any, what: str) -> float:
    if not _is_number(x) or not 0.0 <= x <= 1.0:
        raise JevResponseError(f"{what} fuera de [0, 1] o no numerico: {x!r}")
    return x


def validate_answer(question: Mapping[str, Any], answer: Any) -> dict[str, Any]:
    """Check one answer against the documented contract for its question type."""
    qtype = question.get("type")
    if qtype not in ("noul", "choice", "score"):
        raise JevResponseError(f"Tipo de pregunta desconocido: {qtype!r}")
    if not isinstance(answer, dict) or answer.get("type") != qtype:
        raise JevResponseError(f"La respuesta no corresponde al tipo '{qtype}'.")
    if qtype == "noul":
        _unit(answer.get("noul"), "noul")
    elif qtype == "choice":
        options = question.get("criteria") or {}
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in options:
            raise JevResponseError(f"Opcion fuera de las permitidas: {choice!r}")
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            raise JevResponseError("Probabilidades ausentes o con forma invalida.")
        for opt, p in probs.items():
            if opt not in options:
                raise JevResponseError(f"Probabilidad para opcion desconocida: {opt!r}")
            _unit(p, "probabilidad")
        if abs(sum(probs.values()) - 1.0) > PROB_TOLERANCE:
            raise JevResponseError("Las probabilidades no suman 1.")
        _unit(answer.get("confidence"), "confidence")
    else:
        if not _is_number(answer.get("score")):
            raise JevResponseError("score no numerico.")
        _unit(answer.get("confidence"), "confidence")
    return answer


def _usage(payload: Mapping[str, Any], warnings: list[str]) -> dict[str, int]:
    raw = payload.get("usage")
    out = {"input_tokens": 0, "output_tokens": 0}
    if isinstance(raw, dict) and all(
        isinstance(raw.get(k), int) and not isinstance(raw.get(k), bool) and raw[k] >= 0 for k in out
    ):
        return {k: raw[k] for k in out}
    warnings.append("La respuesta no incluye 'usage' valido; se contabiliza en cero.")
    return out


def response_model(payload: Mapping[str, Any]) -> str | None:
    """The model identifier the API reports (``model``), stripped and truncated; None if absent/invalid."""
    value = payload.get("model")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:MODEL_ID_CHARS]


def parse_response(raw: bytes, questions: Mapping[str, Mapping[str, Any]]):
    """Return (answers, per-question errors, usage, warnings, reported model or None)."""
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        raise JevResponseError("La respuesta 200 no es JSON valido.") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise JevResponseError("La respuesta no tiene la forma esperada ('answers').")
    raw_answers = payload["answers"]
    answers: dict[str, dict] = {}
    errors: dict[str, str] = {}
    warnings: list[str] = []
    for qid, question in questions.items():
        if qid not in raw_answers:
            errors[qid] = "Respuesta ausente para la pregunta."
            continue
        try:
            answers[qid] = validate_answer(question, raw_answers[qid])
        except JevResponseError as exc:
            errors[qid] = f"Respuesta invalida: {exc}"
    for extra in raw_answers.keys() - questions.keys():
        warnings.append(f"Respuesta inesperada ignorada: {str(extra)[:60]!r}")
    return answers, errors, _usage(payload, warnings), warnings, response_model(payload)


def chunked(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ----------------------------------------------------------------------------- client


@dataclass
class QueryResult:
    answers: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    warnings: list[str] = field(default_factory=list)
    requests: int = 0
    models: list[str] = field(default_factory=list)  # distinct model ids reported by the API


def _parse_delay(value: str | None, scale: float) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value) * scale
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class JevClient:
    def __init__(self, config: JevConfig, transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep):
        self._config = config
        self._transport = transport or urllib_transport
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"JevClient(endpoint={self._config.endpoint!r}, model={self._config.model!r})"

    __str__ = __repr__

    # -- helpers
    def _redact(self, text: str) -> str:
        return text.replace(self._config.api_key, "<redacted>")

    def _detail(self, body: bytes) -> str:
        text = body.decode("utf-8", errors="replace")
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False)
        except ValueError:
            pass
        return self._redact(text)[:ERROR_DETAIL_CHARS]

    def _delay(self, headers: Mapping[str, str], attempt: int) -> float:
        lowered = {str(k).lower(): v for k, v in headers.items()}
        delay = _parse_delay(lowered.get("retry-after-ms"), 0.001)
        if delay is None:
            delay = _parse_delay(lowered.get("retry-after"), 1.0)
        if delay is None:
            delay = min(0.5 * 2**attempt, 8.0)
        return min(delay, MAX_RETRY_DELAY)

    def _post(self, payload: dict) -> bytes:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > MAX_BODY_BYTES:
            raise JevApiError(413, "La solicitud supera el maximo de 32 MiB; no se envio.")
        cfg = self._config
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        for attempt in range(cfg.max_retries + 1):
            last = attempt >= cfg.max_retries
            try:
                status, resp_headers, resp_body = self._transport(cfg.endpoint, headers, body, cfg.timeout)
            except OSError as exc:  # TimeoutError and ConnectionError are OSErrors
                if last:
                    raise TransportError(
                        self._redact(f"Error de conexion tras {attempt + 1} intentos: {type(exc).__name__}: {exc}")
                    ) from None
                self._sleep(min(0.5 * 2**attempt, 8.0))
                continue
            if 200 <= status < 300:
                return resp_body
            if 300 <= status < 400:
                raise JevApiError(
                    status,
                    self._redact(
                        f"HTTP {status}: redireccion rechazada (destino: {_redirect_host(resp_headers)}). "
                        "La clave no se reenvia; revise JEV_BASE_URL."
                    ),
                )
            detail = self._detail(resp_body)
            if status == 401:
                raise AuthenticationError(f"HTTP 401: clave de API rechazada o ausente. {detail}")
            if (status in RETRYABLE_STATUSES or status >= 500) and not last:
                self._sleep(self._delay(resp_headers, attempt))
                continue
            raise JevApiError(status, f"HTTP {status}: {detail}")
        raise AssertionError("unreachable")  # pragma: no cover

    # -- public
    def query(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> QueryResult:
        """Ask all questions about one state, chunked to the per-request limit.

        Chunk-level failures are recorded per question; only authentication
        failures abort the whole call.
        """
        result = QueryResult()
        for chunk in chunked(list(questions.items()), MAX_QUESTIONS_PER_REQUEST):
            qs = dict(chunk)
            result.requests += 1
            try:
                raw = self._post({"state": state, "model": self._config.model, "questions": qs})
                answers, errors, usage, warnings, model = parse_response(raw, qs)
            except AuthenticationError:
                raise
            except JevError as exc:
                message = self._redact(str(exc))
                result.errors.update({qid: message for qid in qs})
                continue
            result.answers.update(answers)
            result.errors.update(errors)
            result.warnings.extend(warnings)
            if model is not None:
                model = self._redact(model)
                if model not in result.models:
                    result.models.append(model)
            for k in result.usage:
                result.usage[k] += usage[k]
        return result
