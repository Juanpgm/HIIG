#!/usr/bin/env python3
"""CLI for the emparejador: link one address against a CSV base of addresses.

Exit codes:
    0 - at least one candidate at or above the threshold
    1 - no candidate (including an unparseable query)
    2 - usage error (bad arguments, unreadable or malformed CSV, missing column)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import TextIO

# This project's package lives under <project>/src, not at the project
# root, so it must be added to sys.path before it can be imported.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from emparejador import (  # noqa: E402 - path setup must run first
    DEFAULT_PLATE_TOLERANCE,
    DEFAULT_THRESHOLD,
    AddressIndex,
    InvalidToleranceError,
    LinkResult,
    render,
)

EXIT_FOUND = 0
EXIT_NOT_FOUND = 1
EXIT_USAGE_ERROR = 2

_DEFAULT_TOP = 5
_BOM = "\ufeff"
_DELIMITER_HINTS = (";", "\t", "|")


class _UsageError(Exception):
    """A problem with the arguments or the base file (exit code 2)."""


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
        prog="vincular",
        description="Vincula una dirección contra una base CSV de direcciones ya normalizadas.",
    )
    parser.add_argument("direccion", help="Dirección a vincular (canónica).")
    parser.add_argument("--base", required=True, help="Archivo CSV con la base de direcciones.")
    parser.add_argument("--id-column", required=True, help="Columna con el identificador del registro.")
    parser.add_argument("--address-column", required=True, help="Columna con la dirección.")
    parser.add_argument("--encoding", default="utf-8", help="Codificación del CSV (por defecto utf-8).")
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
        "--top",
        type=int,
        default=_DEFAULT_TOP,
        help=f"Cantidad máxima de candidatos a mostrar (por defecto {_DEFAULT_TOP}).",
    )
    parser.add_argument(
        "--max-block-scan",
        "--max-bloque",
        dest="max_block_scan",
        type=int,
        default=None,
        help="Tope de registros a comparar por bloque; si se supera, el resultado se marca TRUNCADO "
        "(por defecto sin tope).",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="Imprime un único objeto JSON.")
    return parser


def _skip_bom(handle: TextIO) -> None:
    """Leave ``handle`` positioned after a leading UTF-8 BOM, if any.

    Done on the stream (not on the header names) so a BOM in front of a
    quoted header (BOM followed by ``"ID"``) does not derail the CSV parser.
    """
    if handle.read(1) != _BOM:
        handle.seek(0)


def _delimiter_hint(names: list[str]) -> str:
    if len(names) == 1 and any(mark in names[0] for mark in _DELIMITER_HINTS):
        return "; el encabezado parece usar otro delimitador (';', tabulador o '|'): la base debe estar separada por comas"
    return ""


def _is_blank_id(value: object) -> bool:
    return value is None or str(value).strip() == ""


def _load_index(args: argparse.Namespace) -> tuple[AddressIndex, int]:
    """Index the CSV base. Returns ``(index, rows_skipped_for_empty_id)``.

    The CSV is parsed strictly: an unterminated quote, text after a closing
    quote, or a row with more non-empty fields than the header is a usage
    error (exit 2) instead of being silently indexed wrong.
    """
    try:
        index = AddressIndex(plate_tolerance=args.tolerancia_placa)
    except InvalidToleranceError as exc:
        raise _UsageError(str(exc)) from exc
    skipped_empty_id = 0
    reader: csv.DictReader[str] | None = None
    try:
        with open(args.base, newline="", encoding=args.encoding) as handle:
            _skip_bom(handle)
            reader = csv.DictReader(handle, strict=True)
            names = reader.fieldnames
            if not names:
                raise _UsageError(f"el archivo está vacío o no tiene encabezado: {args.base}")
            for column in (args.id_column, args.address_column):
                if column not in names:
                    raise _UsageError(
                        f"columna {column!r} no encontrada; columnas disponibles: {names}"
                        f"{_delimiter_hint(names)}"
                    )
            for row in reader:
                extra = row.get(None)
                if extra and any(field.strip() for field in extra):
                    raise _UsageError(
                        f"CSV mal formado en {args.base} (línea {reader.line_num}): "
                        "más campos que el encabezado (¿coma sin comillas dentro de la dirección?)"
                    )
                record_id = row.get(args.id_column)
                if _is_blank_id(record_id):
                    skipped_empty_id += 1
                    continue
                index.add(record_id, row.get(args.address_column))
    except LookupError as exc:
        raise _UsageError(f"codificación desconocida: {args.encoding!r}") from exc
    except UnicodeDecodeError as exc:
        raise _UsageError(f"no se pudo decodificar {args.base} como {args.encoding}: {exc}") from exc
    except csv.Error as exc:
        line = f" (línea {reader.line_num})" if reader is not None else ""
        raise _UsageError(f"CSV mal formado en {args.base}{line}: {exc}") from exc
    except OSError as exc:
        raise _UsageError(f"no se pudo leer la base {args.base}: {exc}") from exc
    return index, skipped_empty_id


def _candidate_payload(candidate: object) -> dict[str, object]:
    return {
        "id": candidate.record_id,  # type: ignore[attr-defined]
        "puntaje": candidate.score,  # type: ignore[attr-defined]
        "canonica": render(candidate.result.parsed_b),  # type: ignore[attr-defined]
        "avisos": list(candidate.result.warnings),  # type: ignore[attr-defined]
    }


def _print_plain(result: LinkResult, index: AddressIndex, top: int, skipped_empty_id: int) -> None:
    found = result.best is not None
    state = "MATCH" if found else "SIN_COINCIDENCIA"
    ambiguous = "  AMBIGUO" if result.ambiguous else ""
    truncated = "  TRUNCADO" if result.truncated else ""
    print(
        f"{state}  candidatos={len(result.candidates)}  comparados={result.n_compared}"
        f"  umbral={result.threshold:g}{ambiguous}{truncated}"
    )
    print(f"consulta: {render(result.query)}")
    stats = index.stats()
    skipped = f", {skipped_empty_id} sin id" if skipped_empty_id else ""
    print(f"base: {stats['records']} registros indexados, {stats['rejected']} rechazados{skipped}")
    if not result.parse_ok:
        print(f"motivo: {result.reason}")
    if result.best is not None:
        print(f"alternativas: {result.alternatives}")
        if result.near_ambiguous:
            print(
                "nota: otras parcelas también superan el umbral (mismo predio con complemento "
                "conocido o variante de tipo de cruce); revise las alternativas antes de vincular"
            )
        best_warnings = result.best.result.warnings
        if best_warnings:
            print(f"avisos: {', '.join(best_warnings)}")
    if result.truncated:
        print("nota: se alcanzó --max-block-scan; el bloque no se comparó completo y puede faltar un candidato mejor")
    for position, candidate in enumerate(result.candidates[:top], start=1):
        warnings = candidate.result.warnings
        suffix = f"  [avisos: {', '.join(warnings)}]" if warnings else ""
        print(
            f"{position}. {candidate.record_id}  puntaje={candidate.score:.4f}  "
            f"{render(candidate.result.parsed_b)}{suffix}"
        )
    hidden = len(result.candidates) - top
    if hidden > 0:
        print(f"... y {hidden} candidatos más")


def _print_json(
    result: LinkResult, index: AddressIndex, top: int, args: argparse.Namespace, skipped_empty_id: int
) -> None:
    stats = index.stats()
    payload = {
        "consulta": args.direccion,
        "canonica": render(result.query),
        "parse_ok": result.parse_ok,
        "motivo": result.reason,
        "umbral": result.threshold,
        "tolerancia_placa": args.tolerancia_placa,
        "n_comparados": result.n_compared,
        "n_candidatos": len(result.candidates),
        "ambiguo": result.ambiguous,
        "casi_ambiguo": result.near_ambiguous,
        "alternativas": result.alternatives,
        "truncado": result.truncated,
        "mejor": _candidate_payload(result.best) if result.best is not None else None,
        "candidatos": [_candidate_payload(c) for c in result.candidates[:top]],
        "registros": stats["records"],
        "rechazados": stats["rejected"],
        "ids_vacios": skipped_empty_id,
    }
    print(json.dumps(payload, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = _build_parser().parse_args(argv)
    try:
        if not 0.0 <= args.umbral <= 1.0:
            raise _UsageError(f"umbral fuera de rango [0,1]: {args.umbral}")
        if args.top < 1:
            raise _UsageError(f"--top debe ser >= 1: {args.top}")
        if args.max_block_scan is not None and args.max_block_scan < 1:
            raise _UsageError(f"--max-block-scan debe ser >= 1: {args.max_block_scan}")
        index, skipped_empty_id = _load_index(args)
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    if skipped_empty_id:
        print(f"aviso: {skipped_empty_id} filas sin id omitidas", file=sys.stderr)
    result = index.find(args.direccion, threshold=args.umbral, max_block_scan=args.max_block_scan)
    if args.as_json:
        _print_json(result, index, args.top, args, skipped_empty_id)
    else:
        _print_plain(result, index, args.top, skipped_empty_id)
    return EXIT_FOUND if result.best is not None else EXIT_NOT_FOUND


if __name__ == "__main__":
    raise SystemExit(main())
