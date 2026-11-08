"""Write the retrain report (aggregates only) from the artifacts of a retrain.

    .venv/Scripts/python scripts/write_retrain_report.py --out docs/retrain_2026-10_nomenclatura.md \
        --baseline-tag baseline_prod_2026_10 --candidates v2c_cert_s42 v2d_cert_pcer_s42 \
        --control s2b_retrain_s42 --final-tag v2d_cert_pcer_s42 \
        --jev-dir artifacts/jev_audit artifacts/jev_audit_synthetic

Every input is optional: whatever is missing or unreadable is printed in the report as ``n/d (pendiente)``. The verdict
(APRUEBA / NO APRUEBA / INCONCLUSO) is computed from ``artifacts/experiments/<final-tag>/acceptance.json``. The script only
READS the artifacts and writes ``--out``; it refuses an output inside ``artifacts/`` itself, an experiment directory
or an index directory. Output is deterministic and is not rewritten when unchanged; a date appears only through
``--generated-at``.

Exit codes: 0 report written (or already up to date), 2 usage error.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import retrain_report as rr  # noqa: E402

DEFAULT_ARTIFACTS = os.path.join(ROOT, "artifacts")


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):                                  # noqa: D102 - argparse hook
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    ap = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="archivo Markdown a escribir (fuera de artifacts/)")
    ap.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS, help="entradas de solo lectura (por defecto: ./artifacts)")
    ap.add_argument("--baseline-tag", default=rr.BASELINE_DEFAULT)
    ap.add_argument("--candidates", nargs="+", default=[], help="etiquetas de experimento candidatas")
    ap.add_argument("--control", "--controls", dest="control", nargs="+", default=[], help="etiquetas de experimento de control")
    ap.add_argument("--final-tag", default=None, help="la corrida cuyo acceptance.json decide el veredicto")
    ap.add_argument("--jev-dir", nargs="+", default=[], help="directorios que contienen jev_audit.json")
    ap.add_argument("--indexes", nargs="+", default=None,
                    help="nombres de directorios de índice bajo el directorio de artefactos (por defecto: todos los index_v2* con index_meta.json)")
    ap.add_argument("--reprojection", default=None, help="reprojection_report.json (por defecto: train_data/*/)")
    ap.add_argument("--thresholds", default=None,
                    help="JSON de umbrales de aceptación (por defecto: experiments/retrain_2026_10_acceptance.json)")
    ap.add_argument("--calibrated-metrics", nargs="+", default=[], metavar="FILE:KEY=TAG",
                    help="reemplaza las métricas dev de TAG por las calibradas de FILE (bajo experiments/) clave KEY "
                         "(p. ej. seed_comparison.json:s43=v2e_cert_pcer_s43_cal)")
    ap.add_argument("--seed-comparison", default=None,
                    help="seed_comparison.json de la banda de ruido (por defecto: experiments/seed_comparison.json)")
    ap.add_argument("--onnx-dirs", nargs="+", default=None,
                    help="directorios de benchmark ONNX bajo experiments/, en orden de preferencia; si se usa otro que no sea "
                         "onnx_bench_final, el resultado se marca preliminar (por defecto: onnx_bench_final onnx_bench)")
    ap.add_argument("--generated-at", default=None, help="texto de fecha a imprimir en el encabezado (omitido por defecto)")
    return ap


def _forbidden_output(out: str, artifacts_dir: str) -> str | None:
    """Why ``out`` must not be written, or ``None``."""
    out = os.path.abspath(out)
    if os.path.isdir(out):
        return f"--out {out} es un directorio"
    parent = os.path.dirname(out)
    if os.path.abspath(parent) == os.path.abspath(artifacts_dir):
        return "se niega a escribir en el propio directorio de artefactos (allí viven los archivos de producción)"
    for marker in ("experiment.json", rr.INDEX_MARKER):
        if os.path.isfile(os.path.join(parent, marker)):
            return f"se niega a escribir en un directorio de experimento o de índice (se encontró {marker} en {parent})"
    return None


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        problem = _forbidden_output(args.out, args.artifacts_dir)
        if problem:
            raise _UsageError(problem)
        runs = list(args.candidates) + list(args.control)
        if args.final_tag and args.final_tag not in runs:
            raise _UsageError(f"--final-tag {args.final_tag!r} debe ser uno de --candidates / --control")
    except _UsageError as exc:
        print(f"error de uso: {exc}", file=sys.stderr)
        return 2

    inputs = rr.load_inputs(args.artifacts_dir, baseline_tag=args.baseline_tag, candidates=args.candidates,
                            controls=args.control, final_tag=args.final_tag, jev_dirs=args.jev_dir,
                            indexes=args.indexes, reprojection=args.reprojection, thresholds=args.thresholds,
                            calibrated_metrics=args.calibrated_metrics, seed_comparison=args.seed_comparison,
                            onnx_dirs=args.onnx_dirs)
    text = rr.render_report(inputs, generated_at=args.generated_at)
    changed = rr.write_if_changed(args.out, text)
    verdict = rr.compute_verdict(rr._acceptance_results(inputs))["verdict"]
    pending = text.count(rr.PENDING)
    print(f"{'escrito' if changed else 'sin cambios'} {args.out} | veredicto {rr.VERDICT_LABEL[verdict]} | "
          f"{pending} celdas pendientes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
