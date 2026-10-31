"""Build the PNG charts of documentacion/13-mecanismos-de-validacion.md from AGGREGATE experiment JSONs.

Reads only aggregate files (no per-row tables, no address text):

* ``<experiments>/v2_final_catfb/``: reliability_diagrams.json, confidence_curve_compare.json, bands.json,
  bootstrap.json, acceptance.json
* ``<experiments>/onnx_bench_final/bench_final.json``

Usage (matplotlib is a docs-only dependency, it is NOT a runtime dependency of the package)::

    .venv/Scripts/python.exe -m pip install matplotlib      # or: uv pip install matplotlib
    .venv/Scripts/python.exe scripts/make_doc_charts.py --out-dir documentacion/img

The loaders (``load_json`` and the ``extract_*`` functions) do not import matplotlib, so they can be tested without it.
Chart titles and labels are in Spanish because the documentation is in Spanish.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPERIMENTS = os.path.join(ROOT, "artifacts", "experiments")
CANDIDATE_DIR = "v2_final_catfb"
BENCH_DIR = "onnx_bench_final"

# Palette: categorical slots 1 and 2 plus status colours of the dataviz reference palette (light surface).
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#0ca30c", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"


class ChartDataError(Exception):
    """An input JSON is missing, empty, malformed or lacks a field the chart needs."""


# --------------------------------------------------------------------------------------------- loaders
def load_json(path: str) -> Any:
    """Parsed JSON of ``path``. ``ChartDataError`` (never a bare OSError / JSONDecodeError) when the file is missing,
    unreadable, empty (zero bytes or only whitespace), malformed, or holds an empty object / list."""
    if not os.path.isfile(path):
        raise ChartDataError(f"missing input file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ChartDataError(f"cannot read {path}: {exc}") from exc
    if not text.strip():
        raise ChartDataError(f"empty file: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChartDataError(f"malformed JSON in {path}: {exc}") from exc
    if data in ({}, [], None):
        raise ChartDataError(f"empty JSON content: {path}")
    return data


def _need(data: Any, *path: str) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            raise ChartDataError(f"missing field {'.'.join(path)}")
        cur = cur[key]
    return cur


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ChartDataError(f"{name} is not a finite number")
    return float(value)


# --------------------------------------------------------------------------------------------- extractors
def extract_reliability_bins(diagrams: dict, scope: str, head: str = "predio") -> dict:
    """``{"n", "mean_predicted", "observed", "bins": [(n, predicted, observed), ...]}`` of one scope and head."""
    block = _need(diagrams, scope, head)
    bins = _need(block, "bins")
    if not isinstance(bins, list) or not bins:
        raise ChartDataError(f"{scope}.{head}.bins is empty")
    rows = [(int(_num(b.get("n"), "bin.n")), _num(b.get("predicted"), "bin.predicted"),
             _num(b.get("observed"), "bin.observed")) for b in bins]
    return {"n": int(_num(block.get("n"), "n")), "mean_predicted": _num(block.get("mean_predicted"), "mean_predicted"),
            "observed": _num(block.get("observed"), "observed"), "bins": rows}


def extract_confidence_curves(compare: dict, key: str = "predial_lot_precision_emitted") -> dict:
    """Thresholds and, per model (``candidate`` / ``baseline``), the precision series ``key`` and ``share_of_rows``."""
    thresholds = _need(compare, "thresholds")
    out: dict = {"thresholds": [_num(t, "threshold") for t in thresholds]}
    for model in ("candidate", "baseline"):
        series = _need(compare, model)
        if not isinstance(series, list) or len(series) != len(thresholds):
            raise ChartDataError(f"{model} curve does not match the thresholds")
        out[model] = {
            "precision": [None if s.get(key) is None else _num(s[key], key) for s in series],
            "share": [_num(s.get("share_of_rows"), "share_of_rows") for s in series],
        }
    return out


def extract_bands(bands: dict, scope: str) -> list[dict]:
    """Per band (alta / media / baja): n, lot precision with its Wilson interval (``None`` when not scored)."""
    rows = _need(bands, scope)
    if not isinstance(rows, list) or not rows:
        raise ChartDataError(f"{scope} has no bands")
    out = []
    for r in rows:
        scored = r.get("lot_scored") or 0
        out.append({"band": str(_need(r, "band")), "n": int(_num(r.get("n"), "n")),
                    "lot_scored": int(scored),
                    "lot_precision": _num(r["lot_precision"], "lot_precision") if scored else None,
                    "lot_lo": _num(r["lot_lo"], "lot_lo") if scored else None,
                    "lot_hi": _num(r["lot_hi"], "lot_hi") if scored else None})
    return out


def extract_bootstrap(boot: dict) -> list[dict]:
    """One entry per (split, metric): point difference and percentile interval of candidate minus baseline."""
    out = []
    for split in ("dev", "frozen"):
        block = _need(boot, split)
        if not isinstance(block, dict) or not block:
            raise ChartDataError(f"{split} bootstrap block is empty")
        for metric, r in block.items():
            out.append({"split": split, "metric": metric, "diff": _num(r.get("diff"), "diff"),
                        "lo": _num(r.get("lo"), "lo"), "hi": _num(r.get("hi"), "hi"),
                        "p_diff_lt0": _num(r.get("p_diff_lt0"), "p_diff_lt0")})
    return out


def extract_bench(bench: dict) -> dict:
    """Latency (p50 / p95, ms) per thread count, index set and backend, and throughput (rows/s, largest batch)."""
    out: dict = {"latency": {}, "throughput": {}}
    for section, threads in (("threads16", 16), ("threads4", 4)):
        sets = _need(bench, section)
        for index_set, backends in sets.items():
            for backend, r in backends.items():
                lat = _need(r, "latency_ms")
                out["latency"][(threads, index_set, backend)] = (_num(lat.get("p50"), "p50"), _num(lat.get("p95"), "p95"))
                if threads == 16:
                    thr = _need(r, "throughput")
                    biggest = max(thr, key=lambda k: int(k))
                    out["throughput"][(index_set, backend)] = _num(thr[biggest].get("rows_per_s_median"), "rows_per_s")
    if not out["latency"]:
        raise ChartDataError("benchmark file has no latency entries")
    return out


def extract_acceptance(acc: dict) -> list[dict]:
    """``[{"id", "status", "description"}]`` for A1..A12 in order."""
    results = _need(acc, "results")
    if not isinstance(results, list) or not results:
        raise ChartDataError("acceptance has no results")
    out = []
    for r in results:
        status = r.get("status")
        if status not in ("pass", "fail", "not_evaluated"):
            raise ChartDataError(f"unknown status {status!r} for {r.get('id')}")
        out.append({"id": str(_need(r, "id")), "status": status, "description": str(r.get("description", ""))})
    return out


# --------------------------------------------------------------------------------------------- plotting
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.axisbelow": True,
    })
    return plt


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    fig.clf()


def chart_reliability(diagrams: dict, path: str) -> None:
    plt = _plt()
    scopes = [("dev_gold", "Desarrollo (gold)"), ("frozen_gold", "Congelada (gold, segunda mirada)"),
              ("dev_all", "Desarrollo (todas, etiqueta GPS ruidosa)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (scope, title) in zip(axes, scopes):
        d = extract_reliability_bins(diagrams, scope)
        low = min(min(b[1] for b in d["bins"]), min(b[2] for b in d["bins"])) - 0.02
        ax.plot([low, 1], [low, 1], color=INK2, lw=1, ls="--", label="Calibración perfecta")
        ax.plot([b[1] for b in d["bins"]], [b[2] for b in d["bins"]], color=BLUE, lw=2, marker="o", ms=6,
                label="P_predio (10 grupos de igual tamaño)")
        ax.set_title(f"{title}\nn = {d['n']}")
        ax.set_xlabel("Probabilidad predicha (media del grupo)")
        ax.set_xlim(low, 1.005)
        ax.set_ylim(low, 1.005)
        ax.set_ylabel("Frecuencia observada de acierto de lote")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Diagrama de confiabilidad de P_predio (candidato v2_final_catfb)", fontweight="bold", y=1.03)
    _save(fig, path)


def chart_precision_vs_threshold(compare: dict, path: str) -> None:
    plt = _plt()
    d = extract_confidence_curves(compare)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, key, ylabel, title in ((a, "precision", "Precisión de lote entre filas que emiten predio",
                                    "Precisión frente al umbral de confianza"),
                                   (b, "share", "Fracción de filas gold con OK y confianza >= umbral",
                                    "Cobertura frente al umbral de confianza")):
        for model, color, label, marker in (("candidate", BLUE, "Candidato", "o"), ("baseline", ORANGE, "Línea base", "s")):
            ys = [float("nan") if v is None else v for v in d[model][key]]
            ax.plot(d["thresholds"], ys, color=color, lw=2, marker=marker, ms=5, label=label)
        ax.set_xlabel("Umbral de confianza")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axvline(0.7, color=INK2, lw=1, ls=":")
        ax.text(0.705, ax.get_ylim()[0], " piso 0,70", color=INK2, fontsize=8, va="bottom")
    a.set_ylim(0.985, 1.002)
    a.legend(frameon=False, loc="center left")
    fig.suptitle("Desarrollo gold (289 filas): curva de confianza, candidato y línea base", fontweight="bold", y=1.02)
    _save(fig, path)


def chart_bands(bands: dict, path: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=False)
    for ax, scope, title in ((axes[0], "dev_gold", "Desarrollo (gold)"), (axes[1], "frozen_gold", "Congelada (gold)")):
        rows = extract_bands(bands, scope)
        total = sum(r["n"] for r in rows)
        xs = range(len(rows))
        for x, r in zip(xs, rows):
            if r["lot_precision"] is None:
                ax.text(x, 0.85, "n/d\n(sin predio\nemitido)", ha="center", va="center", color=INK2, fontsize=8)
                continue
            ax.bar(x, r["lot_precision"], width=0.5, color=BLUE)
            ax.errorbar(x, r["lot_precision"], yerr=[[r["lot_precision"] - r["lot_lo"]], [r["lot_hi"] - r["lot_precision"]]],
                        color=INK, capsize=4, lw=1.5)
            ax.text(x, 0.705, f"{r['lot_precision']:.3f}", ha="center", va="bottom", color="white", fontsize=9)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{r['band']}\nn = {r['n']} ({r['n'] / total:.0%})" for r in rows])
        ax.set_ylim(0.7, 1.0)
        ax.set_ylabel("Precisión de lote (intervalo de Wilson 95 %)")
        ax.set_title(title)
    fig.suptitle("Precisión y cobertura por banda de confiabilidad (n y porcentaje de las filas gold)",
                 fontweight="bold", y=1.02)
    _save(fig, path)


_METRIC_LABELS = {"gold_ok": "Filas gold OK", "gold_lot_precision": "Precisión de lote (gold)",
                  "all_manzana_precision": "Precisión de manzana (todas)", "lost_correct": "lost_correct (todas)"}


def chart_bootstrap(boot: dict, path: str) -> None:
    plt = _plt()
    entries = extract_bootstrap(boot)
    metrics = list(dict.fromkeys(e["metric"] for e in entries))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.8))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for y, (split, label) in enumerate((("dev", "Desarrollo"), ("frozen", "Congelada"))):
            e = next(x for x in entries if x["metric"] == metric and x["split"] == split)
            color = GREEN if e["lo"] > 0 else RED if e["hi"] < 0 else INK2
            ax.errorbar(e["diff"], y, xerr=[[e["diff"] - e["lo"]], [e["hi"] - e["diff"]]], fmt="o", color=color,
                        capsize=4, lw=2, ms=6)
            ax.text(e["diff"], y + 0.2, f"{e['diff']:+.3g}", ha="center", fontsize=8, color=INK)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Desarrollo", "Congelada"] if ax is axes[0] else ["", ""])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_xlabel("Candidato menos línea base")
    fig.suptitle("Diferencia con intervalo bootstrap pareado 95 % (10 000 remuestreos)", fontweight="bold", y=1.03)
    _save(fig, path)


def chart_onnx(bench: dict, path: str) -> None:
    plt = _plt()
    d = extract_bench(bench)
    backends = ["torch", "onnx", "onnx-int8"]
    colors = {"torch": INK2, "onnx": BLUE, "onnx-int8": ORANGE}
    sets = [("prod", "Índice de producción (330 387 docs)"), ("cand", "Índice candidato (401 860 docs)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, threads in zip(axes[:2], (16, 4)):
        width = 0.25
        for j, backend in enumerate(backends):
            vals = [d["latency"][(threads, s, backend)] for s, _ in sets]
            xs = [i + (j - 1) * width for i in range(len(sets))]
            ax.bar(xs, [v[0] for v in vals], width=width * 0.9, color=colors[backend], label=backend)
            for x, v in zip(xs, vals):
                ax.text(x, v[0] + 0.4, f"{v[0]:.1f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels(["Producción", "Candidato"])
        ax.set_ylabel("Latencia p50 por consulta (ms)")
        ax.set_title(f"Latencia con {threads} hilos")
    ax = axes[2]
    width = 0.25
    for j, backend in enumerate(["onnx", "onnx-int8", "torch"]):
        vals = [d["throughput"].get((s, backend)) for s, _ in sets]
        if any(v is None for v in vals):
            continue
        xs = [i + (j - 1) * width for i in range(len(sets))]
        ax.bar(xs, vals, width=width * 0.9, color=colors[backend])
        for x, v in zip(xs, vals):
            ax.text(x, v + 3, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(["Producción", "Candidato"])
    ax.set_ylabel("Filas por segundo (lote de 10 000, 16 hilos)")
    ax.set_title("Rendimiento por lotes")
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=colors[b], label=b) for b in backends], loc="lower center", ncol=3, frameon=False,
               title="Backend", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Benchmark de backends (torch, ONNX fp32, ONNX int8), un solo trabajador en CPU", fontweight="bold", y=1.03)
    _save(fig, path)


ACCEPTANCE_LABELS = {
    "A1": "Filas gold OK no baja de la línea base menos 1",
    "A2": "Precisión de lote (gold, desarrollo) no baja de la línea base",
    "A3": "Prueba congelada: OK y precisión de lote con margen de 2 filas",
    "A4": "lost_correct dentro de la banda de ruido (máx. 112)",
    "A5": "Precisión de manzana (todas) no baja más de 0,005",
    "A6": "Confiabilidad: ECE de ambas cabezas y regla de despliegue",
    "A7": "Parser, CRS, construcción del índice y pérdida de cobertura",
    "A8": "Producción intacta",
    "A9": "pytest completo en verde y sin direcciones reales",
    "A10": "Curva de confianza a 0,70 relativa a la línea base",
    "A11": "Cuarentena de limpieza < 0,5 % y scripts de índice",
    "A12": "Con validate_outputs=correct nada empeora y 0 violaciones",
}


def chart_acceptance(acc: dict, path: str) -> None:
    plt = _plt()
    rows = extract_acceptance(acc)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rows) + 1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")
    names = {"pass": ("Aprobado", GREEN), "fail": ("No aprobado", RED), "not_evaluated": ("No evaluado", INK2)}
    marks = {"pass": "✔", "fail": "✘", "not_evaluated": "?"}
    for i, r in enumerate(rows):
        label, color = names[r["status"]]
        ax.add_patch(plt.Rectangle((0.0, i - 0.42), 1.2, 0.84, color=color))
        ax.text(0.6, i, r["id"], ha="center", va="center", color="white", fontweight="bold")
        ax.text(1.5, i, f"{marks[r['status']]} {label}", va="center", color=INK, fontweight="bold")
        short = ACCEPTANCE_LABELS.get(r["id"], r["id"])
        ax.text(3.6, i, short, va="center", color=INK2, fontsize=8)
    passed = sum(r["status"] == "pass" for r in rows)
    ax.set_title(f"Criterios de aceptación A1-A12: {passed} de {len(rows)} aprobados (v2_final_catfb)", loc="left")
    _save(fig, path)


# --------------------------------------------------------------------------------------------- main
def build_all(experiments_dir: str, out_dir: str) -> list[str]:
    cand = os.path.join(experiments_dir, CANDIDATE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ("reliability_predio.png", chart_reliability, os.path.join(cand, "reliability_diagrams.json")),
        ("precision_vs_umbral.png", chart_precision_vs_threshold, os.path.join(cand, "confidence_curve_compare.json")),
        ("bandas_confiabilidad.png", chart_bands, os.path.join(cand, "bands.json")),
        ("bootstrap_ic.png", chart_bootstrap, os.path.join(cand, "bootstrap.json")),
        ("onnx_benchmark.png", chart_onnx, os.path.join(experiments_dir, BENCH_DIR, "bench_final.json")),
        ("matriz_aceptacion.png", chart_acceptance, os.path.join(cand, "acceptance.json")),
    ]
    written = []
    for name, fn, src in jobs:
        target = os.path.join(out_dir, name)
        fn(load_json(src), target)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "documentacion", "img"))
    args = ap.parse_args(argv)
    try:
        for path in build_all(args.experiments_dir, args.out_dir):
            print(f"wrote {path}")
    except ChartDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
