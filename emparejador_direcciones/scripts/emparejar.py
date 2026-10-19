#!/usr/bin/env python3
"""CLI for the emparejador: compare two normalized addresses and report a match.

Exit codes:
    0 - MATCH
    1 - NO_MATCH (including unparseable input)
    2 - usage error (bad arguments, invalid threshold/tolerance)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This project's package lives under <project>/src, not at the project
# root, so it must be added to sys.path before it can be imported.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from emparejador import (  # noqa: E402 - path setup must run first
    DEFAULT_PLATE_TOLERANCE,
    DEFAULT_THRESHOLD,
    InvalidThresholdError,
    InvalidToleranceError,
    MatchResult,
    match,
    render,
)

EXIT_MATCH = 0
EXIT_NO_MATCH = 1
EXIT_USAGE_ERROR = 2


def _ensure_utf8_stdout() -> None:
    """Make stdout/stderr UTF-8 safe on Windows consoles that default to a
    legacy code page, so Spanish accented output does not raise or mangle.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emparejar",
        description="Compara dos direcciones ya normalizadas y reporta si coinciden.",
    )
    parser.add_argument("direccion_a", help="Primera dirección (canónica).")
    parser.add_argument("direccion_b", help="Segunda dirección (canónica).")
    parser.add_argument(
        "--umbral",
        "--threshold",
        dest="umbral",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Umbral de coincidencia en [0,1] (por defecto {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--tolerancia-placa",
        "--plate-tolerance",
        dest="tolerancia_placa",
        type=int,
        default=DEFAULT_PLATE_TOLERANCE,
        help=f"Tolerancia entera para la diferencia de placa (por defecto {DEFAULT_PLATE_TOLERANCE}).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Imprime el resultado como un único objeto JSON.",
    )
    return parser


def _print_plain(result: MatchResult) -> None:
    estado = "MATCH" if result.is_match else "NO_MATCH"
    print(f"{estado}  puntaje={result.score:.4f}  umbral={result.threshold:g}")
    print(f"A: {render(result.parsed_a)}")
    print(f"B: {render(result.parsed_b)}")
    print(f"motivo: {result.reason}")
    if result.warnings:
        print(f"avisos: {', '.join(result.warnings)}")


def _print_json(result: MatchResult, plate_tolerance: int) -> None:
    payload = result.to_dict()
    payload["match"] = payload.pop("is_match")
    payload["resultado"] = "MATCH" if result.is_match else "NO_MATCH"
    payload["plate_tolerance"] = plate_tolerance
    payload["a"] = payload.pop("parsed_a")
    payload["b"] = payload.pop("parsed_b")
    print(json.dumps(payload, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = match(
            args.direccion_a,
            args.direccion_b,
            threshold=args.umbral,
            plate_tolerance=args.tolerancia_placa,
        )
    except (InvalidThresholdError, InvalidToleranceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    if args.as_json:
        _print_json(result, args.tolerancia_placa)
    else:
        _print_plain(result)

    return EXIT_MATCH if result.is_match else EXIT_NO_MATCH


if __name__ == "__main__":
    raise SystemExit(main())
