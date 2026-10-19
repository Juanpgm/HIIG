"""Command line interface. Exit codes: 0 ok, 2 usage/config error, 3 no valid results."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence

from .client import (
    DEFAULT_MODEL,
    ConfigurationError,
    JevClient,
    JevConfig,
    Transport,
)
from .io import DEFAULT_MAX_ROWS, InputError, load_items, write_outputs
from .metrics import summarize
from .questions import MATCHER, NORMALIZER
from .validators import Thresholds, dry_run_payloads, run_validation

EXIT_OK, EXIT_USAGE, EXIT_NO_RESULTS = 0, 2, 3
DEFAULT_OUT = "salida_jev"


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"no es un entero: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("debe ser un entero positivo")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validacion-jev",
        description=(
            "Valida con el modelo JEV (juez semantico independiente) la salida del normalizador "
            "o las decisiones del matcher de direcciones. ATENCION: las direcciones se envian a un "
            "servicio externo (TypeSafe); no use datos personales."
        ),
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="mostrar esta ayuda y salir")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{normalizador,matcher}")
    for name, text in (
        (NORMALIZER, "valida la salida del normalizador (columnas: entrada, normalizada, ...)"),
        (MATCHER, "valida las decisiones del matcher (columnas: a, b, is_match, ...)"),
    ):
        p = sub.add_parser(name, help=text, description=text, add_help=False)
        p.add_argument("-h", "--help", action="help", help="mostrar esta ayuda y salir")
        p.add_argument("input", help="archivo CSV o JSONL (UTF-8) de entrada")
        p.add_argument("--out", default=DEFAULT_OUT, help=f"directorio de salida (por defecto ./{DEFAULT_OUT}/; agreguelo a .gitignore)")
        p.add_argument("--env-file", help="archivo dotenv; solo se leen JEV_API_KEY y TYPESAFE_API_KEY")
        p.add_argument("--max-rows", type=_positive_int, default=DEFAULT_MAX_ROWS, help="maximo de filas a enviar (por defecto %(default)s)")
        p.add_argument("--dry-run", action="store_true", help="muestra los payloads exactos sin llamar a la red ni requerir clave")
        p.add_argument("--auto", type=float, default=0.85, help="umbral de certeza para decision automatica (por defecto %(default)s)")
        p.add_argument("--review", type=float, default=0.60, help="umbral de certeza para revision humana (por defecto %(default)s)")
        p.add_argument("--timeout", type=float, default=30.0, help="timeout por solicitud en segundos (por defecto %(default)s)")
    return parser


def _err(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: Transport | None = None,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> int:
    environ = os.environ if environ is None else environ
    for stream in (sys.stdout, sys.stderr):  # never crash on a legacy console codec
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    kind = args.command

    try:
        thresholds = Thresholds(auto=args.auto, review=args.review)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_USAGE

    config = None
    if not args.dry_run:
        try:  # fail before touching the input or the network
            config = dataclasses.replace(JevConfig.from_env(environ, args.env_file), timeout=args.timeout)
        except ConfigurationError as exc:
            _err(str(exc))
            return EXIT_USAGE

    try:
        loaded = load_items(args.input, kind, max_rows=args.max_rows)
    except InputError as exc:
        _err(str(exc))
        return EXIT_USAGE

    if loaded.capped:
        print(
            f"Aviso: el archivo tiene {loaded.total_rows} filas; solo se procesan las primeras {loaded.max_rows}.",
            file=sys.stderr,
        )

    if args.dry_run:
        model = (environ.get("JEV_MODEL") or "").strip() or DEFAULT_MODEL
        payloads = dry_run_payloads(kind, loaded.items, model)
        for payload in payloads:
            print(json.dumps(payload, ensure_ascii=False))
        print(f"Simulacion: {len(payloads)} solicitudes; no se envio nada.", file=sys.stderr)
        return EXIT_OK

    print(
        f"Advertencia: las direcciones se enviaran a un servicio externo ({config.base_url}).",
        file=sys.stderr,
    )
    client = JevClient(config, transport=transport, sleep=sleep or time.sleep)
    try:
        results = run_validation(kind, loaded.items, client, thresholds)
    except ConfigurationError as exc:  # includes AuthenticationError
        _err(str(exc))
        return EXIT_USAGE

    load_info = {
        "total_rows": loaded.total_rows,
        "max_rows": loaded.max_rows,
        "capped": loaded.capped,
        "duplicates_skipped": loaded.duplicates_skipped,
    }
    summary = summarize(results, kind, thresholds, load_info)
    write_outputs(args.out, results, summary)
    counts = summary["counts"]
    print(
        f"Listo: {counts['ok']} evaluadas, {counts['error']} fallidas, {counts['skipped']} omitidas. "
        f"Resultados en {args.out}/"
    )
    if counts["ok"] == 0:
        _err("sin evidencia valida: ninguna fila pudo evaluarse (revise summary.md).")
        return EXIT_NO_RESULTS
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
