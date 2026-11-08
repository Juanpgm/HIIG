"""Evaluate the PRODUCTION path (``normalize_strict``) against ground truth.

Every other eval script in this repo measures
``AddressNormalizer.normalize_batch``'s raw ``matched`` flag. That is NOT what
the CLI or the HTTP service return: both call
``cali_address.service.normalize_strict``, which adds the structural rules,
the geographic gate and the confidence threshold on top of the raw model
match, and is what actually produces ``estado == "OK"``. This script measures
that exact path.

Coordinates are secondary here and are never used as a correctness
criterion: a row is judged right or wrong by cadastral identity (does the
matched ``manzana`` / ``numero_predial_nacional`` equal the ground truth?)
and by text agreement with IDESC's own normalization. ``dist_m`` is kept in
the rows output purely as an informational column.

Usage
-----
    PYTHONPATH=src python scripts/eval_strict.py --tag baseline
    PYTHONPATH=src python scripts/eval_strict.py --limit 50 --tag smoke
    PYTHONPATH=src python scripts/eval_strict.py --datasets fasecolda,stickers
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import geo  # noqa: E402
from cali_address.calibration_report import band_breakdown, monitoring_signals  # noqa: E402
from cali_address.evaluate import (  # noqa: E402
    add_lot_columns,
    index_lots,
    motivo_breakdown,
    nivel_precision_breakdown,
    position_tier_calibration,
    strict_summary,
)
from cali_address.evaluate import _bool_or_false, _share  # noqa: E402
from cali_address.inference import AddressNormalizer  # noqa: E402
from cali_address.labels import select_quality  # noqa: E402
from cali_address.experiment import is_frozen_path  # noqa: E402
from cali_address.parser import match_key  # noqa: E402
from cali_address.paths import resolve_artifacts_dir  # noqa: E402
from cali_address.reliability import FEATURE_NAMES  # noqa: E402
from cali_address.inference import check_validate_outputs, normalize_soft_rules, validate_max_soft  # noqa: E402
from cali_address.service import DEFAULT_AMBIGUITY_DELTA, load_gazetteer, normalize_strict  # noqa: E402

DEFAULT_ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
DEFAULT_BASEMAPS_DIR = os.path.join(ROOT, "basemaps")


# ---------------------------------------------------------------------------
# Cadastral-code / text comparison (coordinate-free correctness)
# ---------------------------------------------------------------------------
def _norm_code(value) -> str | None:
    """String key for a manzana / predial code; ``None`` for blank/NaN/"nan"."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _codes_equal(a, b) -> bool | float:
    """``True``/``False`` when both codes are present, ``NaN`` when either is missing.

    Both codes are fixed-width digit strings in this project's tables, but a
    parquet/Excel round trip can silently turn a leading-zero code into an
    int and drop the padding, so purely numeric codes are also compared
    zero-padded to the longer width before giving up.
    """
    a, b = _norm_code(a), _norm_code(b)
    if a is None or b is None:
        return float("nan")
    if a == b:
        return True
    if a.isdigit() and b.isdigit():
        width = max(len(a), len(b))
        return a.zfill(width) == b.zfill(width)
    return False


def with_lot_columns(rows: pd.DataFrame, normalizer) -> pd.DataFrame:
    """``rows`` plus the lot-level columns (``NPN[:21]``), appended after every existing column.

    ``correct_predial_lot`` is the lot-level counterpart of ``correct_predial`` (the ground truth is cadastral
    and lot-level, a nomenclature NPN may carry a unit suffix); ``lot_present_in_index`` says whether the
    ground-truth lot exists among the lots of the index documents the normalizer serves. Edge cases are
    documented in :func:`cali_address.evaluate.add_lot_columns`.
    """
    return add_lot_columns(rows, index_lots(normalizer._predial))


def _text_agrees_with_idesc(direccion_normalizada, idesc_dir_ajusta) -> bool | float:
    """``match_key`` agreement between our output and IDESC's; NaN with no IDESC answer."""
    if not isinstance(idesc_dir_ajusta, str) or not idesc_dir_ajusta.strip():
        return float("nan")
    if direccion_normalizada is None or (
        isinstance(direccion_normalizada, float) and np.isnan(direccion_normalizada)
    ):
        return False
    return match_key(str(direccion_normalizada)) == match_key(idesc_dir_ajusta)


# ---------------------------------------------------------------------------
# Label columns
# ---------------------------------------------------------------------------
#: Label-quality columns copied from the scored input into the rows output (read by fit_reliability.py).
LABEL_PASSTHROUGH = ("gt_quality",)


def passthrough_label_columns(df: pd.DataFrame) -> dict:
    """``{column: values}`` of the label-quality columns present in ``df`` (none -> ``{}``)."""
    return {c: df[c].tolist() for c in LABEL_PASSTHROUGH if c in df.columns}


def ensure_label_columns(df: pd.DataFrame, label_quality: str | None) -> pd.DataFrame:
    """Copy of ``df`` with the label columns this script needs, for files that lack some.

    ``has_gt`` defaults to "a GT predial or manzana is present"; ``gt_lat``/``gt_lon`` default
    to NaN (coordinates are informational only). ``label_quality`` declares the quality of the
    rows WITH ground truth for a file without ``gt_quality`` (``train_pairs.parquet`` is gold by
    construction, see docs/splits.md); rows without ground truth get ``no_gt``. It refuses to
    override an existing ``gt_quality`` and needs at least one GT code column.
    """
    if "gt_predial" not in df.columns and "gt_manzana" not in df.columns:
        raise KeyError("no ground-truth column (gt_predial / gt_manzana) in the scored file")
    out = df.copy()
    present = pd.Series(False, index=out.index)
    for col in ("gt_predial", "gt_manzana"):
        if col not in out.columns:
            out[col] = None
        present |= out[col].map(lambda v: _norm_code(v) is not None)
    if "has_gt" not in out.columns:
        out["has_gt"] = present.to_numpy()
    for col in ("gt_lat", "gt_lon"):
        if col not in out.columns:
            out[col] = np.nan
    if label_quality is not None:
        if "gt_quality" in out.columns:
            raise ValueError("the scored file already has gt_quality; --label-quality would override it")
        out["gt_quality"] = np.where(out["has_gt"].astype(bool), label_quality, "no_gt")
    return out


# ---------------------------------------------------------------------------
# Raw (unfiltered) top-1: what the model would have picked before any rule,
# geo gate or threshold - this is what `lost_correct` is measured against.
# ---------------------------------------------------------------------------
def _raw_top1_manzana(normalizer: AddressNormalizer, raws: list, k: int) -> list:
    """Manzana of the model's raw top-1 candidate for every row.

    Fuses the same components with the same weights ``normalize_strict``
    fuses internally (service.py, ~line 350), but is computed directly from
    ``score_components`` on every raw string - parseable or not - which is
    what lets a rejected/unparseable row still be checked for a "lost" match.
    """
    comps = normalizer.score_components(raws, k=k)
    weights = normalizer.weights
    fused = (
        weights["sim"] * comps["sim"] + weights["fuzz"] * comps["fuzz"]
        + weights["struct"] * comps["struct"] + weights["geo"] * comps["geo"]
    )
    docs = comps["doc"]
    out: list[str | None] = []
    for i in range(len(raws)):
        slots = np.flatnonzero(docs[i] >= 0)
        if len(slots) == 0:
            out.append(None)
            continue
        best = slots[np.argmax(fused[i, slots])]
        out.append(str(normalizer._manzana[docs[i, best]]))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_headline(summary: pd.DataFrame) -> None:
    all_rows = summary[summary["dataset"] == "ALL"]
    pooled_ok = all_rows[all_rows["estado"] == "OK"]
    total = int(all_rows["n"].sum())
    lost = int(all_rows["lost_correct"].sum())
    print("--- headline (pooled ALL) ---")
    if pooled_ok.empty:
        print(f"OK rows:              0 (of {total})")
        print(f"lost_correct (total): {lost}")
        return
    row = pooled_ok.iloc[0]
    print(f"OK rows:              {int(row['n'])} (of {total})")
    print(f"manzana_precision:    {row['manzana_precision']:.4f}")
    print(f"predial_precision:    {row['predial_precision']:.4f}")
    print(f"idesc_text_agreement: {row['idesc_text_agreement']:.4f}")
    print(f"n_ok_sin_predial:     {int(row['n_ok_sin_predial'])}")
    print(f"n_ok_sin_manzana:     {int(row['n_ok_sin_manzana'])}")
    print(f"lost_correct (total): {lost}")


def ambiguity_kwarg(args) -> dict:
    """``{"ambiguity_delta": v}`` only when the flag was given, else ``{}``.

    Left out, ``normalize_strict`` picks its own default (0.02, or 0 for an artifact in calibrated mode); an explicit
    value always wins, so ``--ambiguity-delta 0.02`` forces the historical abstention even in calibrated mode.
    """
    return {} if args.ambiguity_delta is None else {"ambiguity_delta": float(args.ambiguity_delta)}


def decision_kwargs(args) -> dict:
    """``normalize_strict`` kwargs for the Stage 3 decision options.

    ``None`` (flag not given) lets the artifact's ``tuning.json`` ``decision`` block decide, so
    ``--artifacts-dir <experiment>`` scores that experiment's own configuration.
    """
    soft_rules = None if args.soft_rules is None else normalize_soft_rules(args.soft_rules)
    max_soft = None if args.max_soft is None else validate_max_soft(args.max_soft)
    gate_fallback = None if args.gate_fallback is None else bool(args.gate_fallback)
    return {"soft_rules": soft_rules, "max_soft": max_soft, "gate_fallback": gate_fallback}


def validate_outputs_kwarg(args) -> str | None:
    """``normalize_strict(validate_outputs=...)``: ``None`` (flag not given) lets ``tuning.json`` decide.

    Kept apart from :func:`decision_kwargs` so that function keeps returning exactly the Stage 3 keys.
    """
    return None if args.validate_outputs is None else check_validate_outputs(args.validate_outputs)


VALIDATION_COLUMNS = ["rule", "violations", "corrected", "mode", "rows"]


def write_validation_summary(summary: dict | None, out_dir: str, tag: str) -> str | None:
    """Write ``strict_eval_<tag>_validation.csv`` (one line per rule) and return its path.

    ``summary`` is ``ValidationReport.to_dict()``; ``None`` (mode ``off``) writes nothing. A clean run writes the
    header only, so "no violations" is distinguishable from "validation was not run".
    """
    if summary is None:
        return None
    rules = sorted(set(summary.get("violations", {})) | set(summary.get("corrected", {})))
    frame = pd.DataFrame(
        [{"rule": r, "violations": summary.get("violations", {}).get(r, 0),
          "corrected": summary.get("corrected", {}).get(r, 0), "mode": summary["mode"], "rows": summary["rows"]}
         for r in rules],
        columns=VALIDATION_COLUMNS,
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"strict_eval_{tag}_validation.csv")
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Per-source (fuente) breakdown: only when the index carries a ``fuente`` column
# ---------------------------------------------------------------------------
FUENTE_BREAKDOWN_COLUMNS = [
    "scope", "fuente", "n_rows", "n_ok", "manzana_precision", "predial_precision", "predial_lot_precision", "lost_correct",
]
#: Label of the rows without a selected document (so without ``fuente``): every non-OK row.
NO_FUENTE = "(none)"


def with_fuente_column(rows: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with the output's ``fuente`` appended as the LAST column; unchanged when ``result`` has none."""
    if "fuente" not in result.columns:
        return rows
    if len(result) != len(rows):
        raise ValueError(f"fuente has {len(result)} values for {len(rows)} rows")
    out = rows.copy()
    out["fuente"] = result["fuente"].to_numpy(dtype=object)
    return out


#: Output columns of the calibrated decision layer, copied into the rows file (after ``fuente``) when present.
CALIBRATED_ROW_COLUMNS = (
    "confiabilidad_calibrada", "banda_confiabilidad", "motivos_confiabilidad", "p_predio_cal", "p_manzana_cal",
)


def with_calibrated_columns(rows: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with the calibrated-layer columns of ``result`` appended LAST; unchanged when ``result`` has none."""
    present = [c for c in CALIBRATED_ROW_COLUMNS if c in result.columns]
    if not present:
        return rows
    if len(result) != len(rows):
        raise ValueError(f"calibrated columns have {len(result)} values for {len(rows)} rows")
    out = rows.copy()
    for column in present:
        out[column] = result[column].to_numpy(dtype=object if result[column].dtype == object else None)
    return out


def write_band_breakdown(rows: pd.DataFrame, out_dir: str, tag: str) -> str | None:
    """Write ``strict_eval_<tag>_by_band.csv`` (precision per reliability band x source, Wilson 95 %).

    ``None`` (nothing written) for a run that is not calibrated: no ``banda_confiabilidad`` column, or no banded row.
    """
    table = band_breakdown(rows)
    if table.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"strict_eval_{tag}_by_band.csv")
    table.to_csv(path, index=False)
    return path


def write_calibrated_signals(rows: pd.DataFrame, stats: dict, out_dir: str, tag: str) -> str | None:
    """Write ``strict_eval_<tag>_calibrated.json``: the drift signals of the run plus the decision counters.

    ``stats`` is the dict ``normalize_strict`` filled; without its ``calibrated`` block (the artifact is not in calibrated
    mode) nothing is written and ``None`` is returned. Aggregates only: the per-row probabilities of ``stats`` are
    reduced to quantiles (``p_predio_quantiles`` over every row that reached the decision).
    """
    block = (stats or {}).get("calibrated")
    if not block:
        return None
    signals = monitoring_signals(rows, p_predio_reached=block.get("p_predio"))
    signals["decision_counts"] = {k: v for k, v in block.items() if k != "p_predio"}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"strict_eval_{tag}_calibrated.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(signals, fh, indent=1)
    return path


def _fuente_label(value) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return NO_FUENTE
    text = str(value).strip()
    return text or NO_FUENTE


def fuente_breakdown(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (scope, fuente): sizes and the precisions of the OK rows with ground truth.

    ``scope`` is ``all`` (every row) and ``gold`` (``gt_quality == "gold"``; omitted when the column is absent).
    Precisions use the same denominators as ``strict_summary`` (OK rows with GT whose code is present); they are
    NaN without such rows. ``lost_correct`` counts non-OK rows whose raw top-1 manzana was right: those rows have
    no selected document, so they all sit under ``(none)``. Never raises on an empty or fuente-less frame.
    """
    if rows is None or len(rows) == 0 or "fuente" not in rows.columns:
        return pd.DataFrame(columns=FUENTE_BREAKDOWN_COLUMNS)
    labels = rows["fuente"].map(_fuente_label)
    scopes = [("all", np.ones(len(rows), dtype=bool))]
    if "gt_quality" in rows.columns:
        scopes.append(("gold", (rows["gt_quality"] == "gold").to_numpy()))
    out = []
    for scope, mask in scopes:
        for fuente in sorted(set(labels[mask])):
            group = rows[mask & (labels == fuente).to_numpy()]
            is_ok = (group["estado"] == "OK").to_numpy()
            scored = is_ok & _bool_or_false(group["has_gt"])
            lost = 0
            if "raw_top1_correct_manzana" in group.columns:
                lost = int(_bool_or_false(group.loc[~is_ok, "raw_top1_correct_manzana"]).sum())
            out.append({
                "scope": scope, "fuente": fuente, "n_rows": len(group), "n_ok": int(is_ok.sum()),
                "manzana_precision": _share(group.loc[scored, "correct_manzana"]),
                "predial_precision": _share(group.loc[scored, "correct_predial"]),
                "predial_lot_precision": (_share(group.loc[scored, "correct_predial_lot"])
                                          if "correct_predial_lot" in group.columns else float("nan")),
                "lost_correct": lost,
            })
    return pd.DataFrame(out, columns=FUENTE_BREAKDOWN_COLUMNS)


def write_fuente_breakdown(rows: pd.DataFrame, out_dir: str, tag: str) -> str | None:
    """Write ``strict_eval_<tag>_by_fuente.csv``; ``None`` (nothing written) without a ``fuente`` column or any value."""
    if "fuente" not in rows.columns or not (rows["fuente"].map(_fuente_label) != NO_FUENTE).any():
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"strict_eval_{tag}_by_fuente.csv")
    fuente_breakdown(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default=os.path.join(DEFAULT_ARTIFACTS_DIR, "eval_scored_full.parquet"),
                    help="scored parquet with raw_address/dataset/gt_*/idesc_* columns")
    ap.add_argument("--datasets", default=None, help="comma-separated subset (default: all present)")
    ap.add_argument("--gate-escalate", type=int, choices=(0, 1), default=1,
                    help="matches the API/CLI default of on; normalize_strict's own default is False")
    ap.add_argument("--barrio-buffer", type=float, default=1000.0)
    ap.add_argument("--zone-buffer", type=float, default=300.0)
    ap.add_argument("--min-struct", type=float, default=0.6)
    ap.add_argument("--plate-tolerance", type=int, default=0)
    ap.add_argument("--ambiguity-delta", type=float, default=None,
                    help=f"ambiguity abstention (0 disables). Default: normalize_strict's own ({DEFAULT_AMBIGUITY_DELTA}, "
                         "mirrors production; 0 for an artifact with decision.json). An explicit value always wins")
    ap.add_argument("--soft-rules", default=None,
                    help="comma-separated rule classes downgraded to soft (default: the artifact's decision block)")
    ap.add_argument("--max-soft", type=int, default=None,
                    help="soft violations an OK row may carry (default: the artifact's decision block, else 1)")
    ap.add_argument("--gate-fallback", type=int, choices=(0, 1), default=None,
                    help="keep text-ranked candidates when the geographic gate empties the list "
                         "(default: the artifact's decision block, else off)")
    ap.add_argument("--validate-outputs", choices=("off", "report", "correct"), default=None,
                    help="last step of normalize_strict: count (report) or conservatively fix (correct) schema violations "
                         "of the output; per-rule counts go to strict_eval_<tag>_validation.csv "
                         "(default: the artifact's decision block, else off)")
    ap.add_argument("--gold-only", action="store_true",
                    help="keep only rows with gt_quality == 'gold' (needs that column, see build_splits.py)")
    ap.add_argument("--label-quality", choices=("gold",), default=None,
                    help="declare gt_quality for a scored file without that column (train_pairs.parquet is gold "
                         "by construction); refused when the file already has gt_quality")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--artifacts", "--artifacts-dir", dest="artifacts", default=None,
                    help="artifacts directory holding model.pt etc. (env CALI_ARTIFACTS_DIR; default artifacts/)")
    ap.add_argument("--out-dir", default=None,
                    help="where the strict_eval_<tag>_* files are written (default: the artifacts directory)")
    ap.add_argument("--final", action="store_true",
                    help="required to score artifacts/splits/frozen_test.parquet (final measurement only)")
    ap.add_argument("--basemaps", default=DEFAULT_BASEMAPS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="subsample rows per dataset for a quick run")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args(argv)
    args.artifacts = resolve_artifacts_dir(args.artifacts)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else args.artifacts
    if is_frozen_path(args.scored) and not args.final:
        print("refusing to score frozen_test without --final (it is never used for selection)", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.scored)
    try:
        df = ensure_label_columns(df, args.label_quality)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        df = df[df["dataset"].isin(wanted)].reset_index(drop=True)
    if args.gold_only:
        try:
            df = select_quality(df, gold_only=True)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.limit:
        df = df.groupby("dataset", group_keys=False).head(args.limit).reset_index(drop=True)
    if not len(df):
        print("no rows selected", file=sys.stderr)
        return 1
    print(f"rows: {len(df)} across datasets {sorted(df['dataset'].unique())}", file=sys.stderr)

    gazetteer = load_gazetteer(args.basemaps, warn=lambda m: print(m, file=sys.stderr))
    normalizer = AddressNormalizer(args.artifacts, device=args.device)

    raws = df["raw_address"].astype(str).tolist()
    feature_sink: list = []
    strict_stats: dict = {}
    result = normalize_strict(
        normalizer, raws, feature_sink=feature_sink, min_struct=args.min_struct, plate_tolerance=args.plate_tolerance,
        gazetteer=gazetteer, barrio_buffer_m=args.barrio_buffer, zone_buffer_m=args.zone_buffer,
        gate_escalate=bool(args.gate_escalate), **ambiguity_kwarg(args),
        stats=strict_stats, validate_outputs=validate_outputs_kwarg(args),
        **decision_kwargs(args),
    ).reset_index(drop=True)

    raw_top1_manzana = _raw_top1_manzana(normalizer, raws, k=20)
    has_idesc_col = "idesc_dir_ajusta" in df.columns
    has_gt = df["has_gt"].astype(bool).to_numpy()

    rows = pd.DataFrame({
        "dataset": df["dataset"].to_numpy(),
        "raw_address": df["raw_address"].to_numpy(),
        "estado": result["estado"].to_numpy(),
        "motivo": result["motivo"].fillna("").to_numpy(),
        "nivel_precision": result["nivel_precision"].to_numpy(),
        "confianza": result["confianza"].to_numpy(),
        "confiabilidad": result["confiabilidad"].to_numpy(),
        "confiabilidad_manzana": result["confiabilidad_manzana"].to_numpy(),
        "margen": pd.to_numeric(result["margen"], errors="coerce").to_numpy(),
        "has_gt": has_gt,
        "gt_manzana": df["gt_manzana"].to_numpy(),
        "gt_predial": df["gt_predial"].to_numpy(),
        "manzana": result["manzana"].to_numpy(),
        "numero_predial_nacional": result["numero_predial_nacional"].to_numpy(),
        "direccion_normalizada": result["direccion_normalizada"].to_numpy(),
        "idesc_dir_ajusta": df["idesc_dir_ajusta"].to_numpy() if has_idesc_col else None,
        "lat": result["lat"].to_numpy(),
        "lon": result["lon"].to_numpy(),
        "gt_lat": df["gt_lat"].to_numpy(),
        "gt_lon": df["gt_lon"].to_numpy(),
        "raw_top1_manzana": raw_top1_manzana,
        "position_tier": result["position_tier"].to_numpy(),
        "position_radius_m": result["position_radius_m"].to_numpy(),
        "position_source": result["position_source"].to_numpy(),
        "selected_doc_id": result["selected_doc_id"].to_numpy(),
        "override_rejected": result["override_rejected"].to_numpy(),
        "catalog_status": result["catalog_status"].to_numpy(),
        "address_verified": result["address_verified"].to_numpy(),
        **passthrough_label_columns(df),
    })

    # Reliability-model inputs (None -> NaN for non-OK rows), read by fit_reliability.py.
    for name in FEATURE_NAMES:
        rows[f"feat_{name}"] = [np.nan if f is None else f[name] for f in feature_sink]

    # Informational only: never fed into strict_summary's correctness metrics.
    rows["dist_m"] = geo.haversine_m(
        pd.to_numeric(rows["gt_lat"], errors="coerce"), pd.to_numeric(rows["gt_lon"], errors="coerce"),
        pd.to_numeric(rows["lat"], errors="coerce"), pd.to_numeric(rows["lon"], errors="coerce"),
    )
    rows.loc[~rows["has_gt"], "dist_m"] = np.nan

    rows["correct_manzana"] = [
        _codes_equal(m, gt) if hg else float("nan")
        for m, gt, hg in zip(rows["manzana"], rows["gt_manzana"], rows["has_gt"])
    ]
    rows["correct_predial"] = [
        _codes_equal(p, gt) if hg else float("nan")
        for p, gt, hg in zip(rows["numero_predial_nacional"], rows["gt_predial"], rows["has_gt"])
    ]
    rows["raw_top1_correct_manzana"] = [
        _codes_equal(m, gt) if hg else float("nan")
        for m, gt, hg in zip(rows["raw_top1_manzana"], rows["gt_manzana"], rows["has_gt"])
    ]
    rows["text_agree_idesc"] = [
        _text_agrees_with_idesc(d, i)
        for d, i in zip(rows["direccion_normalizada"], rows["idesc_dir_ajusta"])
    ]
    rows = with_lot_columns(rows, normalizer)     # new columns last: correct_predial_lot, lot_present_in_index
    rows = with_fuente_column(rows, result)       # only when the index carries ``fuente``
    rows = with_calibrated_columns(rows, result)  # last of all, only for an artifact in calibrated mode

    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, f"strict_eval_{args.tag}_rows.parquet")
    summary_path = os.path.join(out_dir, f"strict_eval_{args.tag}_summary.csv")
    motivos_path = os.path.join(out_dir, f"strict_eval_{args.tag}_motivos.csv")

    rows.to_parquet(rows_path, index=False)
    summary = strict_summary(rows)
    summary.to_csv(summary_path, index=False)
    motivos = motivo_breakdown(rows)
    motivos.to_csv(motivos_path, index=False)
    niveles = nivel_precision_breakdown(rows)
    niveles.to_csv(os.path.join(out_dir, f"strict_eval_{args.tag}_niveles.csv"), index=False)
    tiers = position_tier_calibration(rows)
    tiers.to_csv(os.path.join(out_dir, f"strict_eval_{args.tag}_tiers.csv"), index=False)
    validation_path = write_validation_summary(strict_stats.get("output_validation"), out_dir, args.tag)
    by_fuente_path = write_fuente_breakdown(rows, out_dir, args.tag)
    by_band_path = write_band_breakdown(rows, out_dir, args.tag)
    calibrated_path = write_calibrated_signals(rows, strict_stats, out_dir, args.tag)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    _print_headline(summary)
    print("--- nivel_precision (OK rows) ---")
    print(niveles.to_string(index=False))
    print("--- position tiers: share, coverage and error vs ground truth (gold / noisy / ALL) ---")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(tiers.to_string(index=False))
    print(f"wrote {rows_path}", file=sys.stderr)
    print(f"wrote {summary_path}", file=sys.stderr)
    print(f"wrote {motivos_path}", file=sys.stderr)
    if validation_path:
        print(f"wrote {validation_path}", file=sys.stderr)
    if by_fuente_path:
        print(f"wrote {by_fuente_path}", file=sys.stderr)
    if by_band_path:
        print(f"wrote {by_band_path}", file=sys.stderr)
    if calibrated_path:
        print(f"wrote {calibrated_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
