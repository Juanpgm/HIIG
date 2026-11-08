"""Offline audit of the normalizer with the JEV judge (TypeSafe) on PUBLIC data only.

Sources (the only two that may leave the machine, see ``cali_address.jev_check``):

``nomenclatura``  official public nomenclature rows: raw ``nddirecc_original`` vs the cleaned ``nddirecc``
                  (``artifacts/nomenclatura_clean.parquet``);
``synthetic``     ``corrupt_many`` queries of PUBLIC cadastre addresses (``artifacts/catastro_docs.parquet``) scored
                  through the production ``normalize_strict`` path.

dev, frozen_test, train_pairs and unlabeled_addresses hold real citizens' addresses: they are refused (exit 2)
unless the explicit ``--allow-private`` is given. No pipeline passes that flag.

Usage
-----
    PYTHONPATH=src python scripts/audit_with_jev.py --source nomenclatura --sample 20 --dry-run
    PYTHONPATH=src python scripts/audit_with_jev.py --source nomenclatura --sample 300 --env-file <path to .env>

The key is read ONLY by ``validacion_jev`` (``--env-file`` loads just JEV_API_KEY / TYPESAFE_API_KEY, or use the
process environment); it is never printed or written. Exit codes: 0 ok / dry-run, 2 usage, configuration, key or
privacy error, 3 no valid result (every row failed or there were no rows).

Outputs (``--out``, default ``artifacts/jev_audit/``, untracked): ``jev_audit.json`` (no address text) and
``jev_audit_rows.parquet``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import jev_check  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "artifacts", "jev_audit")
DEFAULT_NOMENCLATURA = os.path.join(ROOT, "artifacts", "nomenclatura_clean.parquet")
DEFAULT_DOCS = os.path.join(ROOT, "artifacts", "catastro_docs.parquet")
EXIT_OK, EXIT_USAGE, EXIT_NO_RESULTS = 0, 2, 3

#: File or directory names (lower case, without extension) that hold private citizen data -> origin label.
PRIVATE_NAMES = {
    "dev": "dev", "frozen_test": "frozen_test", "frozen": "frozen_test", "train_pairs": "train_pairs",
    "unlabeled_addresses": "unlabeled", "unlabeled": "unlabeled",
}
PUBLIC_PREFIX = {"nomenclatura": ("nomenclatura_public", "nomenclatura"), "synthetic": ("synthetic", "catastro")}


def origin_for(path: str, source: str) -> str:
    """Origin label derived from the input PATH (never from user free text).

    A private name anywhere in the path wins (``splits/frozen_test/nomenclatura_clean.parquet`` is frozen data); a
    public label needs the expected file-name prefix; everything else is ``unknown`` (refused).
    """
    parts = [p.lower() for p in os.path.normpath(os.path.abspath(path)).split(os.sep) if p]
    stem = os.path.splitext(parts[-1])[0] if parts else ""
    for part in [stem, *reversed(parts[:-1])]:
        if part in PRIVATE_NAMES:
            return PRIVATE_NAMES[part]
    if "splits" in parts:
        return "unknown"
    label, prefix = PUBLIC_PREFIX[source]
    return label if stem.startswith(prefix) else "unknown"


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("nomenclatura", "synthetic"), required=True)
    ap.add_argument("--input", default=None, help="parquet file (default: artifacts/nomenclatura_clean.parquet or "
                                                   "artifacts/catastro_docs.parquet according to --source)")
    ap.add_argument("--sample", type=_non_negative_int, default=300, help="rows to sample (default %(default)s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=_positive_int, default=jev_check.DEFAULT_MAX_ROWS,
                    help="hard cap on rows sent (default %(default)s)")
    ap.add_argument("--env-file", default=None, help="dotenv file; ONLY JEV_API_KEY / TYPESAFE_API_KEY are read from it")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="print the exact payloads; no key, no network")
    ap.add_argument("--allow-private", action="store_true",
                    help="send rows of private origin too (explicit consent only; NO pipeline passes this)")
    ap.add_argument("--auto", type=float, default=0.85)
    ap.add_argument("--review", type=float, default=0.60)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--artifacts", "--artifacts-dir", dest="artifacts", default=None,
                    help="artifacts directory of the normalizer (synthetic source only)")
    ap.add_argument("--basemaps", default=os.path.join(ROOT, "basemaps"))
    ap.add_argument("--device", default="cpu")
    return ap


def _err(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)


def default_normalize(args):
    """Production path (``normalize_strict``) as a ``list[str] -> DataFrame`` callable; heavy imports stay here."""
    from cali_address.inference import AddressNormalizer
    from cali_address.paths import resolve_artifacts_dir
    from cali_address.service import load_gazetteer, normalize_strict

    artifacts = resolve_artifacts_dir(args.artifacts)
    gazetteer = load_gazetteer(args.basemaps, warn=lambda m: print(m, file=sys.stderr))
    normalizer = AddressNormalizer(artifacts, device=args.device)

    def normalize(queries):
        return normalize_strict(normalizer, list(queries), gazetteer=gazetteer, gate_escalate=True,
                                barrio_buffer_m=1000.0, zone_buffer_m=300.0, min_struct=0.6)

    return normalize


def _summary_line(summary: dict) -> str:
    counts, agreement = summary["counts"], summary["agreement"]
    rate = "n/a" if agreement["rate"] is None else f"{agreement['rate']:.3f}"
    return (f"{counts['ok']} judged, {counts['error']} API errors, {counts['skipped']} skipped, "
            f"{counts['indeterminate']} indeterminate; agreement {rate} over {agreement['n']} labelled rows")


def main(argv=None, *, transport=None, environ=None, normalize_fn=None, sleep=None) -> int:
    environ = os.environ if environ is None else environ
    for stream in (sys.stdout, sys.stderr):  # payloads are UTF-8; never crash or garble on a legacy console codec
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    path = args.input or (DEFAULT_NOMENCLATURA if args.source == "nomenclatura" else DEFAULT_DOCS)

    origin = origin_for(path, args.source)
    if origin not in jev_check.PUBLIC_ORIGINS and not args.allow_private:
        _err(f"refusing input with origin {origin!r}: only public data may be sent to the external service "
             f"({', '.join(sorted(jev_check.PUBLIC_ORIGINS))}). Nothing was sent.")
        return EXIT_USAGE

    try:
        jev = jev_check.load_validacion_jev()
        thresholds = jev.validators.Thresholds(auto=args.auto, review=args.review)
    except jev_check.JevUnavailableError as exc:
        _err(str(exc))
        return EXIT_USAGE
    except ValueError as exc:
        _err(str(exc))
        return EXIT_USAGE

    config = None
    if not args.dry_run:
        import dataclasses

        try:  # fail before touching the input or the network
            config = dataclasses.replace(jev.client.JevConfig.from_env(environ, args.env_file), timeout=args.timeout)
        except jev.client.ConfigurationError as exc:
            _err(str(exc))
            return EXIT_USAGE

    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        _err(f"cannot read {path!r} ({type(exc).__name__})")
        return EXIT_USAGE
    try:
        if args.source == "nomenclatura":
            rows = jev_check.rows_from_nomenclatura(frame, args.sample, args.seed)
        else:
            if "direccion" not in frame.columns:
                raise KeyError("missing column in the cadastre docs file: direccion")
            normalize = normalize_fn or default_normalize(args)
            rows = jev_check.rows_from_synthetic(frame["direccion"].tolist(), args.sample, args.seed, normalize)
    except KeyError as exc:
        _err(str(exc.args[0]) if exc.args else "missing columns")
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - a broken normalizer / artifacts must be a clear exit, not a traceback
        _err(f"could not build the rows ({type(exc).__name__}: {exc})")
        return EXIT_USAGE
    for row in rows:  # the path decides the origin; nothing else can relabel a row as public
        row.origin = origin

    try:
        if args.dry_run:
            model = (environ.get("JEV_MODEL") or "").strip() or jev.client.DEFAULT_MODEL
            payloads = jev_check.dry_run_payloads(rows, model=model, max_rows=args.max_rows, allow_private=args.allow_private)
            for payload in payloads:
                print(json.dumps(payload, ensure_ascii=False))
            print(f"Dry run: {len(payloads)} requests; nothing was sent.", file=sys.stderr)
            return EXIT_OK
        print(f"Warning: addresses will be sent to an external service ({config.base_url}).", file=sys.stderr)
        client = jev.client.JevClient(config, transport=transport, **({"sleep": sleep} if sleep else {}))
        result = jev_check.audit_rows(rows, client, thresholds, max_rows=args.max_rows, allow_private=args.allow_private,
                                      source=args.source)
    except jev_check.PrivacyError as exc:
        _err(str(exc))
        return EXIT_USAGE
    except jev.client.ConfigurationError as exc:  # includes AuthenticationError
        _err(str(exc))
        return EXIT_USAGE

    paths = jev_check.write_audit(result, args.out)
    print(_summary_line(result.summary))
    print(f"wrote {paths['json']}", file=sys.stderr)
    print(f"wrote {paths['rows']}", file=sys.stderr)
    if result.summary["counts"]["ok"] == 0:
        _err("no valid evidence: no row could be judged (see jev_audit.json).")
        return EXIT_NO_RESULTS
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
