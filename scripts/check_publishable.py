#!/usr/bin/env python3
"""Publish guard: fail when a tracked or staged file must not be published.

Stdlib only. Checks every file that git tracks (``git ls-files``) plus every
file staged for the next commit, and reports violations as
``path[:line]: RULE`` lines. It never prints file contents or secret values.

Rules
-----
BLOCKED_DIR      the path crosses a directory that holds private inputs, run
                 results or caches (context/, outputs/, runs/, notebooks/,
                 experiments/, splits/, candidatos/, salida_jev*/, .venv/, ...)
BLOCKED_NAME     the file name is a known run result, derived dataset or cache
                 (strict_eval_*, *leaderboard*, *.npz, *.log, *.ipynb, ...)
ENV_FILE         a ``.env*`` file (``.env.example`` is the only exception)
KEY_FILE         a private key / certificate / credentials file
DATA_NOT_ALLOWED a parquet/csv/json/xlsx/pt/pkl/... file that is not in the
                 allowlist ``DATA_ALLOWLIST`` below
RESULT_REPORT    a markdown/text file whose NAME looks like a results report
                 (eval_*, *_results*, resultados*, *metricas*, *leaderboard*,
                 *benchmark*) and that is not listed in ``RESULT_REPORT_ALLOWLIST``.
                 Only the name is checked, never the content: run results must
                 not be published, so a report named like one is blocked.
SIZE_OVER_50MB   file larger than 50 MB
SIZE_OVER_90MB   file larger than 90 MB (hard limit, no exemption)
SECRET:<name>    a secret pattern with a non-placeholder value

A line that carries the marker ``publish-guard: allow`` is skipped by the
secret scan (use it only for synthetic fixtures).

The real-address scan needs private data that CI does not have, so it is NOT
part of this script: run it locally before pushing (see scripts/pre-push).

Exit codes: 0 clean, 1 violations found, 2 usage or git error.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MB = 1024 * 1024
SIZE_LIMIT = 50 * MB
SIZE_HARD_LIMIT = 90 * MB
TEXT_SCAN_LIMIT = 12 * MB
ALLOW_MARKER = "publish-guard: allow"

# Directory names (any depth) that must never be tracked.
BLOCKED_DIRS = {
    "context", "outputs", "runs", "notebooks", "experiments", "splits",
    "repo_audit", "candidatos", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "build", "dist", ".atl", "scratch", "backup",
    "backups", ".ipynb_checkpoints",
}
BLOCKED_DIR_GLOBS = ("salida_jev*", "*.egg-info")
# Multi-component paths: the curated serving copy that deploy/prepare_artifacts.py assembles (199 MB).
BLOCKED_PATH_FRAGMENTS = ("deploy/artifacts",)

# Data-like extensions: tracked only when allowlisted.
DATA_EXTS = {
    ".parquet", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls",
    ".pt", ".pth", ".pkl", ".pickle", ".npz", ".npy", ".feather", ".h5",
    ".sqlite", ".db", ".geojson", ".gpkg", ".shp", ".zip",
}

# Exact (posix) path patterns, matched with fnmatch against the repo-relative
# path. Only the model, its small configuration and public cadastre/basemaps.
DATA_ALLOWLIST = (
    "normalizador_direcciones/artifacts/model.pt",
    "normalizador_direcciones/artifacts/tuning.json",
    "normalizador_direcciones/artifacts/reliability.json",
    "normalizador_direcciones/artifacts/reliability_v2.json",
    "normalizador_direcciones/artifacts/catastro_meta.json",
    "normalizador_direcciones/artifacts/catastro_docs.parquet",
    "normalizador_direcciones/artifacts/catastro_own_centroids.parquet",
    "normalizador_direcciones/basemaps/*.geojson",
    "emparejador_direcciones/data/MANIFEST.json",
)

# Text/markup documents are checked by NAME for results-report look-alikes.
TEXT_DOC_EXTS = {".md", ".markdown", ".txt", ".rst"}
RESULT_REPORT_NAME = re.compile(
    r"^eval_|_results|^results?[._-]|^resultados|metricas|leaderboard|benchmark"
)
# Exact (posix) paths of documents that carry such a name on purpose. Every entry must be a
# protocol/definition document with NO measured figures (reviewable in git). Keep it minimal.
RESULT_REPORT_ALLOWLIST = (
    "emparejador_direcciones/documentacion/08-metricas-y-confiabilidad.md",
    "normalizador_direcciones/documentacion/08-metricas-y-confiabilidad.md",
)

_DATA = r"\.(csv|tsv|json|jsonl|ndjson|parquet|xlsx|xls|feather|pkl|pickle|pt|npz|npy)$"
# (rule label, regex over the lower-cased base name)
BLOCKED_NAME_PATTERNS = (
    ("run-output", re.compile(r"^(strict_)?eval_.*" + _DATA)),
    ("run-output", re.compile(r"^strict_eval_")),
    ("run-output", re.compile(r"leaderboard")),
    ("run-output", re.compile(r"^idesc_.*" + _DATA)),
    ("run-output", re.compile(r"predictions?.*" + _DATA)),
    ("run-output", re.compile(r"_(motivos|niveles|summary)[^/]*\.csv$")),
    ("run-output", re.compile(r"^(summary|results)\.(csv|json|jsonl|parquet|xlsx)$")),
    ("derived-data", re.compile(r"^train_data")),
    ("derived-data", re.compile(r"^catastro_emb\.pt$")),
    ("derived-data", re.compile(r"\.(npz|npy)$")),
    ("derived-data", re.compile(r"sample.*" + _DATA)),
    ("derived-data", re.compile(r"^registros_")),
    ("derived-data", re.compile(r"^maestra")),
    ("derived-data", re.compile(r"^tabla_indices")),
    ("cache", re.compile(r"^interp_index_v1\.")),
    ("cache", re.compile(r"^catalog_keys_v1\.")),
    ("cache", re.compile(r"\.pkl$")),
    ("artifact", re.compile(r"\.ipynb$")),
    ("artifact", re.compile(r"\.log$")),
    ("artifact", re.compile(r"^\.coverage(\..*)?$")),
    ("artifact", re.compile(r"\.pyc$")),
)

KEY_FILE_GLOBS = ("*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "id_rsa*", "id_ed25519*",
                  "credentials*.json", "service-account*.json", "service_account*.json")

PLACEHOLDER_WORDS = (
    "example", "your", "changeme", "change-me", "placeholder", "dummy", "fake",
    "redacted", "xxxx", "****", "sample", "todo", "<", "${", "{{", "%(",
    "secret_value", "token_value", "password_here",
)
PLACEHOLDER_USERINFO = {
    "u", "p", "user", "usr", "pw", "pass", "password", "passwd", "username", "me",
    "name", "login", "admin", "root", "postgres", "bob", "alice",
}

SECRET_ASSIGN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|client[_-]?secret|auth[_-]?token|"
    r"token|passw(?:or)?d|secret|private[_-]?key)\b[\"']?\s*[:=]\s*[\"']([^\"'\s]{16,})[\"']"
)
PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
BEARER = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-_.=]{20,})")
CONN_STRING = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql)(?:\+\w+)?://"
    r"([^\s:@/'\"]+):([^\s@'\"/]+)@([^\s/'\"?]+)"
)
AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
GH_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")


def _is_placeholder(value: str) -> bool:
    v = value.lower()
    return any(w in v for w in PLACEHOLDER_WORDS) or len(set(v)) <= 3


def scan_line(line: str) -> list[str]:
    """Names of secret patterns found in one line (placeholders ignored)."""
    if ALLOW_MARKER in line:
        return []
    found: list[str] = []
    for m in SECRET_ASSIGN.finditer(line):
        if "://" not in m.group(2) and not _is_placeholder(m.group(2)):
            found.append("secret-assignment")
            break
    if PRIVATE_KEY.search(line):
        found.append("private-key-header")
    for m in BEARER.finditer(line):
        if not _is_placeholder(m.group(1)):
            found.append("bearer-token")
            break
    for m in CONN_STRING.finditer(line):
        user, pw, host = m.groups()
        host_l = host.lower()
        placeholder = (
            pw.lower() in PLACEHOLDER_USERINFO
            or len(pw) < 10  # real service passwords are longer than test fixtures
            or any(c in pw for c in "{}$")  # interpolated, not a literal value
            or _is_placeholder(pw)
            or host_l.startswith(("127.", "0.0.0.0", "[::1]"))
            or any(h in host_l for h in ("example", ".invalid", "localhost", "host"))
        )
        if not placeholder:
            found.append("connection-string-credentials")
            break
    if AWS_KEY.search(line):
        found.append("aws-access-key")
    if GH_TOKEN.search(line):
        found.append("github-token")
    return found


def _git(root: Path, *args: str) -> bytes:
    res = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", "replace").strip() or "git failed")
    return res.stdout


def candidate_files(root: Path) -> list[str]:
    """Tracked plus staged (added/copied/modified/renamed) paths, deduplicated."""
    tracked = _git(root, "ls-files", "-z").split(b"\0")
    staged = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split(b"\0")
    names = {p.decode("utf-8", "surrogateescape") for p in tracked + staged if p}
    return sorted(names)


def check_path(rel: str) -> list[str]:
    """Path-level rules (no file access)."""
    rules: list[str] = []
    p = PurePosixPath(rel)
    dirs = p.parts[:-1]
    base = p.name.lower()
    if any(d.lower() in BLOCKED_DIRS or any(fnmatch.fnmatch(d.lower(), g) for g in BLOCKED_DIR_GLOBS)
           for d in dirs) or any(f"/{frag}/" in f"/{rel.lower()}" for frag in BLOCKED_PATH_FRAGMENTS):
        rules.append("BLOCKED_DIR")
    for label, rx in BLOCKED_NAME_PATTERNS:
        if rx.search(base):
            rules.append(f"BLOCKED_NAME[{label}]")
            break
    if base.startswith(".env") and base not in {".env.example", ".env.sample", ".env.template"}:
        rules.append("ENV_FILE")
    if base.endswith(".env"):
        rules.append("ENV_FILE")
    if any(fnmatch.fnmatch(base, g) for g in KEY_FILE_GLOBS):
        rules.append("KEY_FILE")
    ext = p.suffix.lower()
    if ext in DATA_EXTS and not any(fnmatch.fnmatch(rel, g) for g in DATA_ALLOWLIST):
        rules.append("DATA_NOT_ALLOWED")
    if ext in TEXT_DOC_EXTS and RESULT_REPORT_NAME.search(base) and rel not in RESULT_REPORT_ALLOWLIST:
        rules.append("RESULT_REPORT")
    return rules


def check_file(root: Path, rel: str) -> list[tuple[int, str]]:
    """All violations of one file as (line, rule); line 0 means file-level."""
    out = [(0, r) for r in check_path(rel)]
    path = root / rel
    if not path.is_file():
        return out
    size = path.stat().st_size
    if size > SIZE_HARD_LIMIT:
        out.append((0, "SIZE_OVER_90MB"))
    elif size > SIZE_LIMIT:
        out.append((0, "SIZE_OVER_50MB"))
    if size > TEXT_SCAN_LIMIT:
        return out
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return out  # binary: path/size rules only
    for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        if len(line) > 20000:
            line = line[:20000]
        for name in scan_line(line):
            out.append((n, f"SECRET:{name}"))
    return out


def run(root: Path) -> list[str]:
    report: list[str] = []
    for rel in candidate_files(root):
        for line, rule in check_file(root, rel):
            report.append(f"{rel}:{line}: {rule}" if line else f"{rel}: {rule}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None, help="repository root (default: git toplevel of cwd)")
    ap.add_argument("--max-lines", type=int, default=200, help="cap on printed violations")
    args = ap.parse_args(argv)
    try:
        root = args.root or Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
        report = run(root)
    except (RuntimeError, OSError) as exc:
        print(f"check_publishable: error: {exc}", file=sys.stderr)
        return 2
    if not report:
        print("check_publishable: OK (no blocked, oversized, non-allowlisted or secret-bearing files)")
        return 0
    for line in report[: args.max_lines]:
        print(line)
    if len(report) > args.max_lines:
        print(f"... {len(report) - args.max_lines} more")
    print(f"check_publishable: FAIL ({len(report)} violation(s))", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
