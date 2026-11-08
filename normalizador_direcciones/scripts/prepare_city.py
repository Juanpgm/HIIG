"""Prepare any city's address table for ``build_index.py`` (and optionally build the index right away).

    python scripts/prepare_city.py --mapping city.json --out-dir runs/micity/staging [--index-dir runs/micity/index]

Reads the input named in the mapping (CSV/TSV, Parquet, GeoJSON, shapefile, Excel, JSON ...), validates it with
``cali_address.city_adapter`` and writes into ``--out-dir``:

    nomenclatura_clean.parquet          the kept rows in the schema build_index.py reads (--wfs-clean)
    nomenclatura_cleaning_report.json   quarantine counts for build_index.py's accounting (--cleaning-report)
    audit_catastro_docs_empty.parquet   empty audit cadastre (--catastro-docs): a new city has no previous index
    train_pairs_empty.parquet           empty real-pair table (--pairs): add labelled pairs later if you have them
    quarantine.parquet                  every rejected row with its reason (PRIVATE: holds input rows, never publish)
    city_codes.json                     district / neighbourhood code tables (names -> codes used as labels)
    adapter_report.json                 counts, reasons, CRS, gates (aggregates only)

With ``--index-dir`` it then runs ``build_index.py`` with those files (tiers ``cert,pcer``).

Exit codes: 0 ok; 1 unreadable input; 2 mapping / validation error or a gate failed (the quarantine and the report
are still written, the clean table is not); 3 unexpected failure (``--debug`` shows the traceback); 130 interrupted.
Every expected error is ONE ``error:`` line, never a traceback.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.city_adapter import AdapterError, adapt, load_mapping, read_source  # noqa: E402

CLEAN_NAME = "nomenclatura_clean.parquet"
CLEANING_REPORT_NAME = "nomenclatura_cleaning_report.json"
AUDIT_DOCS_NAME = "audit_catastro_docs_empty.parquet"
PAIRS_NAME = "train_pairs_empty.parquet"
QUARANTINE_NAME = "quarantine.parquet"
CODES_NAME = "city_codes.json"
REPORT_NAME = "adapter_report.json"
OUTPUTS = (CLEAN_NAME, CLEANING_REPORT_NAME, AUDIT_DOCS_NAME, PAIRS_NAME, QUARANTINE_NAME, CODES_NAME, REPORT_NAME)


def _write_json(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _write_parquet(frame: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def empty_audit_docs() -> pd.DataFrame:
    return pd.DataFrame({"direccion": pd.Series([], dtype="str"),
                         "numero_predial_nacional": pd.Series([], dtype="str"),
                         "doc_id": pd.Series([], dtype="int64")})


def empty_pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "raw_address": pd.Series([], dtype="str"), "address_key": pd.Series([], dtype="str"),
        "dataset": pd.Series([], dtype="str"), "gt_doc_id": pd.Series([], dtype="int64"),
        "gt_manzana": pd.Series([], dtype="str"), "gt_predial": pd.Series([], dtype="str"),
        "gt_comuna": pd.Series([], dtype="str"), "source_row_id": pd.Series([], dtype="str"),
    })


def prepare(mapping_path: str, out_dir: str, input_override: str | None = None, overwrite: bool = False) -> dict:
    """Run the adapter and write the staging files; returns the report. Raises AdapterError."""
    mapping = load_mapping(mapping_path)
    out_dir = os.path.abspath(out_dir)
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise AdapterError(f"{out_dir} is not empty; pass --overwrite to replace its staging files")
    frame, fmt, warnings = read_source(mapping, input_override)
    result = adapt(frame, mapping, fmt=fmt, reader_warnings=warnings)
    os.makedirs(out_dir, exist_ok=True)
    for name in OUTPUTS:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    _write_parquet(result.quarantine, os.path.join(out_dir, QUARANTINE_NAME))
    _write_json(result.report, os.path.join(out_dir, REPORT_NAME))
    if result.gates_failed:
        return result.report
    _write_parquet(result.clean, os.path.join(out_dir, CLEAN_NAME))
    _write_parquet(empty_audit_docs(), os.path.join(out_dir, AUDIT_DOCS_NAME))
    _write_parquet(empty_pairs(), os.path.join(out_dir, PAIRS_NAME))
    _write_json(result.codes, os.path.join(out_dir, CODES_NAME))
    _write_json({"quarantined_rows": result.report["quarantined_rows"], "by_reason": result.report["by_reason"],
                 "status_invalid_literals": {}, "source": "scripts/prepare_city.py"},
                os.path.join(out_dir, CLEANING_REPORT_NAME))
    return result.report


def build_index_args(staging: str, index_dir: str, overwrite: bool = False, n_variants: int | None = None,
                     seed: int | None = None) -> list[str]:
    args = ["--wfs-clean", os.path.join(staging, CLEAN_NAME),
            "--catastro-docs", os.path.join(staging, AUDIT_DOCS_NAME),
            "--pairs", os.path.join(staging, PAIRS_NAME),
            "--cleaning-report", os.path.join(staging, CLEANING_REPORT_NAME),
            "--tiers", "cert,pcer", "--out-dir", index_dir]
    if overwrite:
        args.append("--overwrite")
    if n_variants is not None:
        args += ["--n-variants", str(n_variants)]
    if seed is not None:
        args += ["--seed", str(seed)]
    return args


def run_build_index(args: list[str]) -> int:
    spec = importlib.util.spec_from_file_location("_build_index", os.path.join(ROOT, "scripts", "build_index.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return int(module.main(args))
    except SystemExit as exc:          # argparse usage errors
        return int(exc.code or 0) if isinstance(exc.code, int) else 2


def _summary(report: dict) -> str:
    reasons = ", ".join(f"{k}={v}" for k, v in report["by_reason"].items() if v) or "none"
    return (f"{report['city']}: {report['input_rows']} rows read, {report['kept_rows']} kept, "
            f"{report['quarantined_rows']} quarantined ({report['quarantine_share']:.2%}; {reasons})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mapping", required=True, help="mapping file (.json / .toml / .yaml)")
    ap.add_argument("--out-dir", required=True, help="staging directory for the adapter outputs")
    ap.add_argument("--input", default=None, help="override the mapping's input path")
    ap.add_argument("--overwrite", action="store_true", help="replace previous outputs (staging and index)")
    ap.add_argument("--index-dir", default=None, help="also run build_index.py into this directory")
    ap.add_argument("--n-variants", type=int, default=None, help="build_index.py --n-variants")
    ap.add_argument("--seed", type=int, default=None, help="build_index.py --seed")
    ap.add_argument("--debug", action="store_true", help="show the traceback of unexpected failures")
    args = ap.parse_args(argv)
    try:
        report = prepare(args.mapping, args.out_dir, args.input, args.overwrite)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - last-resort guard: one line unless --debug
        if args.debug:
            raise
        print(f"error: unexpected failure ({type(exc).__name__}: {exc}); rerun with --debug", file=sys.stderr)
        return 3
    print(_summary(report))
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    if report["gates_failed"]:
        print(f"error: gate(s) failed: {', '.join(report['gates_failed'])} (max_quarantine_share="
              f"{report['max_quarantine_share']}); see {os.path.join(args.out_dir, REPORT_NAME)} and "
              f"{QUARANTINE_NAME}", file=sys.stderr)
        return 2
    staging = os.path.abspath(args.out_dir)
    bi_args = build_index_args(staging, args.index_dir or "<index-dir>", args.overwrite, args.n_variants, args.seed)
    if not args.index_dir:
        print("next: python scripts/build_index.py " + " ".join(f'"{a}"' if " " in a else a for a in bi_args))
        return 0
    return run_build_index(bi_args)


if __name__ == "__main__":
    sys.exit(main())
