import logging

import pytest

from conftest import SENTINEL_KEY
from validacion_jev.client import (
    ConfigurationError,
    JevConfig,
    parse_env_file,
    resolve_key,
)


def test_missing_key_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        JevConfig.from_env({})


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_key_is_missing(blank):
    with pytest.raises(ConfigurationError):
        JevConfig.from_env({"JEV_API_KEY": blank})


def test_jev_key_takes_precedence_over_typesafe():
    assert resolve_key({"JEV_API_KEY": "a", "TYPESAFE_API_KEY": "b"}) == "a"


def test_typesafe_key_is_fallback():
    assert resolve_key({"TYPESAFE_API_KEY": "b"}) == "b"


def test_blank_primary_falls_back_to_secondary():
    assert resolve_key({"JEV_API_KEY": " ", "TYPESAFE_API_KEY": "b"}) == "b"


def test_defaults():
    cfg = JevConfig.from_env({"JEV_API_KEY": "k"})
    assert cfg.base_url == "https://api.typesafe.ai"
    assert cfg.model == "jev-latest"
    assert cfg.timeout == 30.0
    assert cfg.max_retries == 2


def test_env_overrides_and_trailing_slash():
    cfg = JevConfig.from_env(
        {"JEV_API_KEY": "k", "JEV_BASE_URL": "https://x.example/", "JEV_MODEL": "jev-9"}
    )
    assert cfg.base_url == "https://x.example"
    assert cfg.model == "jev-9"
    assert cfg.endpoint == "https://x.example/v1/systemone"


@pytest.mark.parametrize("url", ["http://evil.example", "ftp://x", "api.typesafe.ai", ""])
def test_insecure_or_malformed_base_url_rejected(url):
    with pytest.raises(ConfigurationError):
        JevConfig.from_env({"JEV_API_KEY": "k", "JEV_BASE_URL": url})


def test_http_localhost_allowed():
    cfg = JevConfig.from_env({"JEV_API_KEY": "k", "JEV_BASE_URL": "http://localhost:8080"})
    assert cfg.base_url == "http://localhost:8080"


def test_key_never_in_repr_or_str(caplog):
    cfg = JevConfig(api_key=SENTINEL_KEY)
    assert SENTINEL_KEY not in repr(cfg)
    assert SENTINEL_KEY not in str(cfg)
    assert SENTINEL_KEY not in f"{cfg!r} {cfg}"
    assert SENTINEL_KEY not in caplog.text


def test_env_file_from_file_used_when_env_missing(tmp_path):
    p = tmp_path / "creds"
    p.write_text(f"JEV_API_KEY={SENTINEL_KEY}\n", encoding="utf-8")
    cfg = JevConfig.from_env({}, env_file=str(p))
    assert cfg.api_key == SENTINEL_KEY


def test_process_env_beats_env_file(tmp_path):
    p = tmp_path / "creds"
    p.write_text("JEV_API_KEY=fromfile\n", encoding="utf-8")
    cfg = JevConfig.from_env({"JEV_API_KEY": "fromenv"}, env_file=str(p))
    assert cfg.api_key == "fromenv"


# ---- env-file parser -------------------------------------------------------


def _write(tmp_path, text):
    p = tmp_path / "x.env"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_parser_handles_comments_quotes_export_spaces(tmp_path):
    path = _write(
        tmp_path,
        "# comment\n\n"
        "export JEV_API_KEY = 'abc def'\n"
        'TYPESAFE_API_KEY="xyz"  \n'
        "   # indented comment\n",
    )
    assert parse_env_file(path) == {"JEV_API_KEY": "abc def", "TYPESAFE_API_KEY": "xyz"}


def test_parser_ignores_unrelated_names(tmp_path):
    path = _write(tmp_path, "DATABASE_URL=postgres://secret\nOPENAI_API_KEY=zzz\nJEV_API_KEY=k\n")
    result = parse_env_file(path)
    assert result == {"JEV_API_KEY": "k"}
    assert "postgres" not in repr(result)


def test_parser_missing_file():
    with pytest.raises(ConfigurationError) as ei:
        parse_env_file("/no/such/file.env")
    assert "no/such" in str(ei.value) or "such" in str(ei.value)


def test_parser_handles_bom_crlf_and_garbage_lines(tmp_path):
    p = tmp_path / "b.env"
    p.write_bytes(b"\xef\xbb\xbfJEV_API_KEY=k1\r\nnot a pair\r\n=novalue\r\nTYPESAFE_API_KEY=\r\n")
    assert parse_env_file(str(p)) == {"JEV_API_KEY": "k1", "TYPESAFE_API_KEY": ""}


def test_parser_inline_comment_after_unquoted_value(tmp_path):
    path = _write(tmp_path, "JEV_API_KEY=abc # note\n")
    assert parse_env_file(path) == {"JEV_API_KEY": "abc"}


def test_parser_errors_do_not_leak_content(tmp_path):
    p = tmp_path / "bad.env"
    p.write_bytes(b"JEV_API_KEY=" + SENTINEL_KEY.encode() + b"\xff\xfe\n")
    with pytest.raises(ConfigurationError) as ei:
        parse_env_file(str(p))
    assert SENTINEL_KEY not in str(ei.value)
