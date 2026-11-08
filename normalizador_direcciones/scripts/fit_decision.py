"""Fit the calibrated decision layer (two sign-constrained logistic heads + thresholds) from a labelled pairs file.

What it does (CPU only, deterministic):

1. Runs ``service.normalize_strict`` in a DECISION-NEUTRAL configuration on the labelled pairs (confidence cut-off 0,
   no ambiguity abstention, the neutral soft rules with ``max_soft`` 99, gate fallback on, no structural floor) and
   collects, for every row that comes out OK, the 24 serve-time features plus the cadastral codes of the chosen document
   (``calibration_sink``). No address text is kept: groups are salted hashes.
2. Labels: ``y_lot`` = the lot (``NPN[:21]``) of the chosen document equals the ground-truth lot; ``y_manzana`` = the
   manzana equals the ground-truth manzana.
3. Per head: ``C`` from an inner 3-fold group CV (log-loss) over {0.01 .. 3}; final fit by L-BFGS-B with the sign
   constraints of ``SIGNS``; a binary feature whose rarer value has < 30 rows is frozen at 0.
4. Out-of-fold probabilities (5-fold GroupKFold) give ``tau_p`` / ``tau_m``: the smallest value >= 0.70 whose cumulative
   precision on the OOF rows with ``confianza >= 0.70`` reaches the target (0.99; lot precision over the rows that emit a
   predial).
5. Writes ``decision_<tag>.json``, ``reliability_calibrated_<tag>.json`` and ``fit_report_<tag>.json`` (aggregates only) to
   ``--out-dir``; never overwrites an existing file. Install them as ``decision.json`` / ``reliability_calibrated.json``
   in an artifact directory with ``scripts/make_decision_experiment.py``.

``frozen_test`` is refused: the fit uses ``train_pairs`` (never used for training the encoder), nothing else.

Usage (project root):
  .venv/Scripts/python.exe scripts/fit_decision.py --artifacts-dir artifacts/experiments/v2e_cert_pcer_s42_r3 \
      --pairs artifacts/splits/train_pairs.parquet --out-dir <params dir> --tag cand --device cpu
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import decision as D  # noqa: E402
from cali_address.evaluate import lots_match, predial_lot  # noqa: E402
from cali_address.experiment import is_frozen_path  # noqa: E402
from cali_address.inference import normalize_soft_rules  # noqa: E402
from cali_address.service import normalize_strict  # noqa: E402

#: Feature -> sign constraint of its coefficient (+1 raises P, -1 lowers it, 0 free). Order = ``decision.FEATURE_NAMES``.
SIGNS = {
    "confidence": 1, "margen_manzana": 1, "margen_predio": 1, "raw_margin_lot": 1, "n_competidores": -1,
    "log_n_predios": -1, "soft_n": -1, "relaxed_plate": -1, "gate_fallback": -1, "fidelity_medium": -1,
    "fidelity_high": -1, "plate_missing": -1, "cross_missing": -1, "src_pcer": -1, "lot_conflict": -1,
    "cat_verified": 1, "cat_ambiguous": -1, "cat_door_not_catalogued": -1, "guard_one_sided": -1, "guard_tramo": -1,
    "predial_null": -1, "nivel_manzana": 0, "nivel_via": 0, "nivel_esquina": 0,
}
NAMES = list(SIGNS)
assert tuple(NAMES) == D.FEATURE_NAMES, "SIGNS must follow decision.FEATURE_NAMES"

#: Opt-in feature (``--catastro-feature``, default OFF): 1.0 when the chosen document comes from the cadastre fallback tier.
#: Its sign is free (a cadastral document may be more or less reliable than a nomenclature one) and it enters the model
#: ONLY when at least ``MIN_SUPPORT`` fit rows carry each of its two values; otherwise the artifact is the 24-feature one.
OPTIONAL_SIGNS = {"src_catastro": 0}
OPTIONAL_MOTIVO_CODES = {"src_catastro": "fuente_catastro"}
assert tuple(OPTIONAL_SIGNS) == D.OPTIONAL_FEATURE_NAMES

#: Short Spanish code of each feature in ``motivos_confiabilidad`` (the output is Spanish-facing).
MOTIVO_CODES = {
    "confidence": "confianza", "margen_manzana": "margen_manzana", "margen_predio": "margen_predio",
    "raw_margin_lot": "margen_lote", "n_competidores": "competidores", "log_n_predios": "direccion_compartida",
    "soft_n": "reglas_suaves", "relaxed_plate": "placa_cercana", "gate_fallback": "fuera_de_zona",
    "fidelity_medium": "fidelidad_media", "fidelity_high": "fidelidad_alta", "plate_missing": "consulta_sin_placa",
    "cross_missing": "consulta_sin_cruce", "src_pcer": "fuente_pcer", "lot_conflict": "conflicto_lote",
    "cat_verified": "puerta_verificada", "cat_ambiguous": "puerta_ambigua",
    "cat_door_not_catalogued": "puerta_no_catalogada", "guard_one_sided": "letra_o_cuadrante_unilateral",
    "guard_tramo": "placa_aproximada", "predial_null": "npn_nulo", "nivel_manzana": "nivel_manzana",
    "nivel_via": "nivel_via", "nivel_esquina": "nivel_esquina",
}
#: Features whose motivo reference is the training mean (a margin reads "+" when wider than usual); 0 for the rest.
MEAN_REFERENCE = ("confidence", "margen_manzana", "margen_predio", "raw_margin_lot")

NEUTRAL_SOFT_RULES = ("letra_via", "letra_cruce_cardinal", "bis_via", "bis_cruce", "cuadrante_compl")
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CLIP = 5.0
P_MIN, P_MAX = D.P_MIN, D.P_MAX
#: A binary feature whose rarer value has fewer rows than this is frozen at 0 (it cannot be estimated).
MIN_SUPPORT = 30
#: Fewer labelled rows than this per head is refused.
MIN_ROWS = 100
FLOOR = 0.70
TARGET_LOT = 0.99
TARGET_MANZANA = 0.99
BAND_ALTA = 0.97
OUTER_FOLDS = 5
INNER_FOLDS = 3
SALT = "calib-salt|"


def group_key(value) -> str:
    """Salted hash of a grouping value (an address key): stable, and the table never holds the value itself."""
    return hashlib.sha256((SALT + str(value)).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------
# Signed logistic head
# ---------------------------------------------------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _check_xy(X: np.ndarray, y: np.ndarray, signs) -> None:
    if X.ndim != 2 or X.shape[1] != len(signs) or len(y) != len(X):
        raise ValueError("X must be (n, n_features) and y must have n entries")
    if not np.isfinite(X).all():
        raise ValueError("features must be finite (no NaN / inf)")
    if not np.isfinite(y).all():
        raise ValueError("labels must be finite")


def fit_head(X: np.ndarray, y: np.ndarray, signs, C: float, allow_one_class: bool = False) -> dict:
    """L2 logistic regression on standardized, clipped features with sign-constrained coefficients (L-BFGS-B).

    ``means`` / ``stds`` are the stored standardization (a constant column gets std 1); features whose rarer binary value
    has < :data:`MIN_SUPPORT` rows are frozen at 0. ``ValueError`` for NaN / inf input or (unless ``allow_one_class``) a
    single class. Returns ``coef``, ``intercept``, ``means``, ``stds``, ``C``, ``frozen`` (indices).
    """
    from scipy.optimize import minimize

    X, y = np.asarray(X, float), np.asarray(y, float)
    _check_xy(X, y, signs)
    mean = X.mean(0)
    std = np.where(X.std(0) < 1e-9, 1.0, X.std(0))
    k = X.shape[1]
    if len(np.unique(y)) < 2:
        if not allow_one_class:
            raise ValueError("labels must contain both classes")
        p = float(np.clip(y.mean() if len(y) else 0.5, 0.01, 0.99))
        return {"coef": [0.0] * k, "intercept": float(np.log(p / (1 - p))), "means": mean.tolist(), "stds": std.tolist(),
                "C": float(C), "frozen": list(range(k))}
    Z = np.clip((X - mean) / std, -CLIP, CLIP)
    n = len(Z)
    lam = 1.0 / (C * n)

    def fun(w):
        logits = Z @ w[:k] + w[k]
        p = _sigmoid(logits)
        loss = np.mean(np.logaddexp(0, logits) - y * logits) + 0.5 * lam * np.sum(w[:k] ** 2)
        grad = np.r_[Z.T @ (p - y) / n + lam * w[:k], np.mean(p - y)]
        return loss, grad

    bounds = [(0, None) if s > 0 else (None, 0) if s < 0 else (None, None) for s in signs] + [(None, None)]
    for j in range(k):
        column = X[:, j]
        if np.isin(column, (0.0, 1.0)).all() and min((column == 1).sum(), (column == 0).sum()) < MIN_SUPPORT:
            bounds[j] = (0, 0)
    frozen = [j for j in range(k) if bounds[j] == (0, 0)]
    result = minimize(fun, np.zeros(k + 1), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    return {"coef": [float(v) for v in result.x[:k]], "intercept": float(result.x[k]), "means": mean.tolist(),
            "stds": std.tolist(), "C": float(C), "frozen": frozen}


def predict_head(head: dict, X: np.ndarray) -> np.ndarray:
    Z = np.clip((np.asarray(X, float) - np.array(head["means"])) / np.array(head["stds"]), -CLIP, CLIP)
    return np.clip(_sigmoid(Z @ np.array(head["coef"]) + head["intercept"]), P_MIN, P_MAX)


def group_folds(groups, n_splits: int):
    """``(train_idx, test_idx)`` pairs of a GroupKFold (no group on both sides); shrinks to the number of groups."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("need at least 2 groups to cross-validate")
    k = min(n_splits, n_groups)
    yield from GroupKFold(k).split(np.zeros(len(groups)), groups=groups)


def _logloss(p, y) -> float:
    p = np.clip(p, P_MIN, P_MAX)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def choose_c(X, y, groups, signs, grid=C_GRID) -> float:
    """``C`` of the grid with the lowest inner 3-fold group-CV log-loss (ties: the smaller ``C``)."""
    best, best_loss = grid[0], math.inf
    for c in grid:
        oof = np.zeros(len(y))
        for train, test in group_folds(groups, INNER_FOLDS):
            head = fit_head(X[train], y[train], signs, c, allow_one_class=True)
            oof[test] = predict_head(head, X[test])
        loss = _logloss(oof, y)
        if loss < best_loss:
            best, best_loss = c, loss
    return float(best)


def oof_predict(X, y, groups, signs) -> np.ndarray:
    """Out-of-fold probabilities (5-fold GroupKFold, ``C`` re-chosen inside every training part)."""
    oof = np.zeros(len(y))
    for train, test in group_folds(groups, OUTER_FOLDS):
        c = choose_c(X[train], y[train], groups[train], signs)
        oof[test] = predict_head(fit_head(X[train], y[train], signs, c, allow_one_class=True), X[test])
    return oof


# ---------------------------------------------------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------------------------------------------------
def choose_tau(p, y, target: float, floor: float = FLOOR) -> tuple[float, bool, list]:
    """Smallest ``tau`` on the 0.005 grid from ``floor`` whose rows with ``p >= tau`` reach precision ``target``.

    Returns ``(tau, reached, curve)``; when no value reaches the target ``tau`` is the top of the grid (0.995) and
    ``reached`` is False. NaN probabilities are ignored; no usable row is a ``ValueError``.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    keep = np.isfinite(p) & np.isfinite(y)
    p, y = p[keep], y[keep]
    if len(p) == 0:
        raise ValueError("no usable rows to choose the threshold from")
    grid = np.round(np.arange(floor, 0.996, 0.005), 3)
    curve, chosen = [], None
    for t in grid:
        mask = p >= t
        precision = float(y[mask].mean()) if mask.any() else float("nan")
        curve.append({"tau": float(t), "n": int(mask.sum()), "precision": precision, "coverage": float(mask.mean())})
        if chosen is None and mask.any() and precision >= target:
            chosen = float(t)
    if chosen is None:
        return float(grid[-1]), False, curve
    return chosen, True, curve


# ---------------------------------------------------------------------------------------------------------------------
# Metrics (aggregates only)
# ---------------------------------------------------------------------------------------------------------------------
def _ece(p, y, bins: int = 10) -> float:
    order = np.argsort(p, kind="stable")
    return float(sum(len(i) / len(p) * abs(p[i].mean() - y[i].mean()) for i in np.array_split(order, bins) if len(i)))


def _head_metrics(p, y) -> dict:
    from sklearn.metrics import roc_auc_score

    return {"n": int(len(y)), "observed": float(y.mean()), "mean_pred": float(p.mean()), "ece10": _ece(p, y),
            "brier": float(np.mean((p - y) ** 2)), "logloss": _logloss(p, y),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")}


# ---------------------------------------------------------------------------------------------------------------------
# Fit from a feature table
# ---------------------------------------------------------------------------------------------------------------------
def fit_tables(table: pd.DataFrame, tag: str, artifacts_dir: str, *, soft_rules=NEUTRAL_SOFT_RULES,
               band_alta: float = BAND_ALTA, floor: float = FLOOR, target_lot: float = TARGET_LOT,
               target_manzana: float = TARGET_MANZANA, fit_date: str | None = None,
               rows_description: str = "labelled pairs (never used for training)",
               mode: str = "decision-neutral normalize_strict", catastro_feature: bool = False) -> dict:
    """``{"decision", "reliability", "report"}`` from a table with the 24 features, ``y_lot``, ``y_manzana``, ``group``
    and ``emits_predial`` (one row per OK row of the neutral run). Both documents are validated with the serving
    loaders before they are returned; ``ValueError`` explains anything unusable.

    ``catastro_feature`` (default off, nothing changes) adds the ``src_catastro`` column of the table as a 25th feature when
    each of its two values has at least :data:`MIN_SUPPORT` rows among the rows the ``predio`` head is fitted on; the report
    then carries a ``catastro_feature`` block either way.
    """
    names, signs_map, codes = list(NAMES), dict(SIGNS), dict(MOTIVO_CODES)
    catastro_report = None
    if catastro_feature:
        labelled_lot = pd.to_numeric(table["y_lot"], errors="coerce").notna() if "y_lot" in table.columns else None
        flag = pd.to_numeric(table["src_catastro"], errors="coerce") if "src_catastro" in table.columns else None
        n_cat = n_other = 0
        if flag is not None and labelled_lot is not None:
            n_cat, n_other = int((flag[labelled_lot] == 1).sum()), int((flag[labelled_lot] == 0).sum())
        included = min(n_cat, n_other) >= MIN_SUPPORT
        catastro_report = {"requested": True, "rows_with_feature": n_cat, "min_rows": MIN_SUPPORT, "included": included}
        if included:
            names += list(OPTIONAL_SIGNS)
            signs_map.update(OPTIONAL_SIGNS)
            codes.update(OPTIONAL_MOTIVO_CODES)
    required = names + ["y_lot", "y_manzana", "group", "emits_predial"]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise ValueError(f"missing column(s) {missing}")
    fit_date = fit_date or dt.datetime.now().isoformat(timespec="seconds")
    stamp = fit_date[:10].replace("-", "")
    rules = sorted(normalize_soft_rules(soft_rules))
    signs = list(signs_map.values())
    conf = table["confidence"].to_numpy(float)
    groups_all = table["group"].astype(str).to_numpy()
    heads, probs, labelled, report = {}, {}, {}, {"n_rows": int(len(table)), "heads": {}}
    for head_name, column in (("predio", "y_lot"), ("manzana", "y_manzana")):
        y_all = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        sel = np.isfinite(y_all)
        if int(sel.sum()) < MIN_ROWS:
            raise ValueError(f"need at least {MIN_ROWS} labelled rows for the {head_name} head, got {int(sel.sum())}")
        X, y, groups = table.loc[sel, names].to_numpy(float), y_all[sel], groups_all[sel]
        c = choose_c(X, y, groups, signs)
        heads[head_name] = fit_head(X, y, signs, c)
        in_sample = predict_head(heads[head_name], table[names].to_numpy(float))
        oof = oof_predict(X, y, groups, signs)
        in_sample[sel] = oof                                   # out-of-fold wherever the head has a label
        probs[head_name], labelled[head_name] = in_sample, (sel, y_all)
        report["heads"][head_name] = {
            "n": int(sel.sum()), "positives": int(y.sum()), "C": c,
            "frozen_features": [names[j] for j in heads[head_name]["frozen"]], "oof": _head_metrics(oof, y),
            # same rows and folds, no sign constraints: the reference of the A6 ship rule (Brier <= refit + 0.002)
            "refit_oof": _head_metrics(oof_predict(X, y, groups, [0] * len(signs)), y),
        }
    p_predio, p_manzana = probs["predio"], np.maximum(probs["manzana"], probs["predio"])
    emits = table["emits_predial"].fillna(False).astype(bool).to_numpy()
    sel_lot, y_lot = labelled["predio"]
    sel_man, y_man = labelled["manzana"]
    at_floor = conf >= floor - 1e-12
    lot_rows = emits & sel_lot & at_floor
    man_rows = sel_man & at_floor
    if not lot_rows.any() or not man_rows.any():
        raise ValueError("no labelled row reaches the confianza floor, cannot choose the thresholds")
    tau_p, reached_p, curve_p = choose_tau(p_predio[lot_rows], y_lot[lot_rows], target_lot, floor)
    tau_m, reached_m, curve_m = choose_tau(p_manzana[man_rows], y_man[man_rows], target_manzana, floor)
    if catastro_report is not None:
        report["catastro_feature"] = catastro_report
    report.update({"tau_p_reached_target": reached_p, "tau_m_reached_target": reached_m,
                   "curve_lot": curve_p, "curve_manzana": curve_m})
    n_lot, n_man = report["heads"]["predio"], report["heads"]["manzana"]
    provenance = {
        "rows": rows_description, "n_predio": n_lot["n"], "positives_predio": n_lot["positives"],
        "n_manzana": n_man["n"], "positives_manzana": n_man["positives"], "artifacts_dir": artifacts_dir,
        "mode": f"{mode} (calibrated mode)", "cv": f"{OUTER_FOLDS}-fold GroupKFold by address key; C by {INNER_FOLDS}-fold inner CV",
    }
    reliability_version = f"reliability-calib-v1-{tag}-{stamp}"

    def head_block(name: str, target: str) -> dict:
        h = heads[name]
        return {"target": target, "coef": h["coef"], "intercept": h["intercept"], "C": h["C"], "means": h["means"],
                "stds": h["stds"]}

    reliability = {
        "kind": D.KIND, "method": "logit_signed", "signs": dict(signs_map), "version": reliability_version,
        "fit_date": fit_date, "feature_names": list(names), "motivo_codes": dict(codes),
        "motivo_reference": {n: (heads["predio"]["means"][j] if n in MEAN_REFERENCE else 0.0)
                             for j, n in enumerate(names)},
        "clip": CLIP,
        "predio": head_block("predio", "lot (NPN[:21]) of the chosen document equals the cadastral GT lot"),
        "manzana": head_block("manzana", "manzana of the chosen document equals the GT manzana"),
        "provenance": provenance,
    }
    decision = {
        "version": f"decision-calib-v1-{tag}-{stamp}", "reliability_version": reliability_version,
        "tau_p": tau_p, "tau_m": tau_m, "floor": floor, "confianza_floor": floor,
        "bands": {"alta": band_alta, "media": tau_p},
        "precision_targets": {"lot": target_lot, "manzana": target_manzana},
        "neutral": {"soft_rules": rules, "max_soft": 99, "gate_fallback": True, "ambiguity_delta": 0.0,
                    "threshold": "confianza_floor"},
        "fit_provenance": provenance,
    }
    decision = D.validate_decision(decision)
    try:
        model = D.parse_calibrated(json.loads(json.dumps(reliability)))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"fitted reliability parameters are not loadable: {exc}") from exc
    if model.version != decision["reliability_version"]:
        raise ValueError("internal error: decision and reliability versions differ")
    return {"decision": decision, "reliability": reliability, "report": report}


# ---------------------------------------------------------------------------------------------------------------------
# Feature table from a pipeline run
# ---------------------------------------------------------------------------------------------------------------------
def _code(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return None if text.upper() in ("", "NONE", "NAN", "NAT", "NULL", "<NA>") else text


def _codes_equal(a, b) -> float:
    """1.0 / 0.0 when both codes are present (zero-padded numeric codes compare equal), NaN when either is missing."""
    a, b = _code(a), _code(b)
    if a is None or b is None:
        return float("nan")
    if a == b:
        return 1.0
    if a.isdigit() and b.isdigit():
        width = max(len(a), len(b))
        return float(a.zfill(width) == b.zfill(width))
    return 0.0


def build_table(normalizer, pairs: pd.DataFrame, gazetteer=None, *, soft_rules=NEUTRAL_SOFT_RULES, min_struct: float = 0.0,
                plate_tolerance: int = 0, barrio_buffer_m: float = 1000.0, zone_buffer_m: float = 300.0,
                gate_escalate: bool = True) -> pd.DataFrame:
    """Run the decision-neutral pipeline on ``pairs`` and return one table row per OK row (features + labels).

    ``pairs`` needs ``raw_address``, ``gt_predial`` and ``gt_manzana`` (``address_key`` groups the rows for the CV,
    ``raw_address`` is the fallback key). The table keeps no address text: ``group`` is a salted hash. A calibrated mode
    already present on ``normalizer`` is switched off for the run and restored afterwards.
    """
    for column in ("raw_address", "gt_predial", "gt_manzana"):
        if column not in pairs.columns:
            raise KeyError(f"pairs file has no {column!r} column")
    raws = pairs["raw_address"].astype(str).tolist()
    keys = pairs["address_key"] if "address_key" in pairs.columns else pairs["raw_address"]
    sink: list = []
    had = hasattr(normalizer, "calibrated")
    previous = getattr(normalizer, "calibrated", None)
    normalizer.calibrated = None
    try:
        normalize_strict(
            normalizer, raws, min_struct=min_struct, plate_tolerance=plate_tolerance, gazetteer=gazetteer,
            barrio_buffer_m=barrio_buffer_m, zone_buffer_m=zone_buffer_m, gate_escalate=gate_escalate,
            ambiguity_delta=0.0, soft_rules=normalize_soft_rules(soft_rules), max_soft=99, gate_fallback=True,
            threshold=0.0, validate_outputs="off", calibration_sink=sink,
        )
    finally:
        if had:
            normalizer.calibrated = previous
        else:
            del normalizer.calibrated
    records = []
    gt_p, gt_m = pairs["gt_predial"].tolist(), pairs["gt_manzana"].tolist()
    for i, entry in enumerate(sink):
        if entry is None:
            continue
        lot = float("nan")
        if _code(gt_p[i]) is not None and entry["doc_predial"] is not None and predial_lot(gt_p[i]) is not None:
            lot = float(lots_match(entry["doc_predial"], gt_p[i]))
        record = {name: float(entry[name]) for name in NAMES}
        record.update({
            "src_catastro": float(entry.get("src_catastro", 0.0)),
            "y_lot": lot, "y_manzana": _codes_equal(entry["doc_manzana"], gt_m[i]), "group": group_key(keys.iloc[i]),
            "emits_predial": bool(entry["emits_predial"]), "emits_manzana": bool(entry["emits_manzana"]),
        })
        records.append(record)
    columns = NAMES + ["src_catastro", "y_lot", "y_manzana", "group", "emits_predial", "emits_manzana"]
    return pd.DataFrame(records, columns=columns)


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------
def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", required=True, help="artifact directory the layer will serve (model, index)")
    ap.add_argument("--pairs", required=True, help="labelled pairs parquet (raw_address, gt_predial, gt_manzana, address_key)")
    ap.add_argument("--out-dir", required=True, help="where decision_<tag>.json / reliability_calibrated_<tag>.json go")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--soft-rules", default=",".join(NEUTRAL_SOFT_RULES), help="soft rules of the neutral run and of serving")
    ap.add_argument("--min-struct", type=float, default=0.0, help="structural floor of the neutral run (the study used 0)")
    ap.add_argument("--band-alta", type=float, default=BAND_ALTA)
    ap.add_argument("--target-lot", type=float, default=TARGET_LOT)
    ap.add_argument("--target-manzana", type=float, default=TARGET_MANZANA)
    ap.add_argument("--basemaps", default=os.path.join(ROOT, "basemaps"))
    ap.add_argument("--limit", type=int, default=None, help="use the first N pairs only (smoke runs)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--catastro-feature", action="store_true",
                    help="opt-in: add the src_catastro feature (cadastre fallback docs) when >= 30 fit rows carry each value")
    ap.add_argument("--fit-date", default=None, help="ISO timestamp recorded in the files (default: now)")
    args = ap.parse_args(argv)
    if is_frozen_path(args.pairs) or "frozen" in os.path.basename(args.pairs).lower():
        print("refusing frozen_test: the decision layer is fitted on train_pairs only", file=sys.stderr)
        return 2
    paths = {name: os.path.join(args.out_dir, f"{name}_{args.tag}.json")
             for name in ("decision", "reliability_calibrated", "fit_report")}
    clash = [p for p in paths.values() if os.path.exists(p)]
    if clash:
        print(f"refusing to overwrite existing file(s): {', '.join(clash)}", file=sys.stderr)
        return 2
    from cali_address.inference import AddressNormalizer
    from cali_address.service import load_gazetteer

    pairs = pd.read_parquet(args.pairs)
    if args.limit:
        pairs = pairs.head(args.limit)
    gazetteer = load_gazetteer(args.basemaps, warn=lambda m: print(m, file=sys.stderr))
    normalizer = AddressNormalizer(args.artifacts_dir, device=args.device)
    rules = tuple(r.strip() for r in args.soft_rules.split(",") if r.strip())
    table = build_table(normalizer, pairs, gazetteer, soft_rules=rules, min_struct=args.min_struct)
    print(f"neutral run: {len(pairs)} pairs -> {len(table)} OK rows", file=sys.stderr)
    try:
        result = fit_tables(
            table, args.tag, os.path.relpath(args.artifacts_dir, ROOT) if os.path.isabs(args.artifacts_dir) else args.artifacts_dir,
            soft_rules=rules, band_alta=args.band_alta, target_lot=args.target_lot, target_manzana=args.target_manzana,
            fit_date=args.fit_date, rows_description=f"{os.path.basename(args.pairs)} (labelled pairs, never used for training)",
            catastro_feature=args.catastro_feature,
        )
    except ValueError as exc:
        print(f"fit failed: {exc}", file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    _write_json(paths["decision"], result["decision"])
    _write_json(paths["reliability_calibrated"], result["reliability"])
    _write_json(paths["fit_report"], result["report"])
    d = result["decision"]
    print(f"tau_p={d['tau_p']} tau_m={d['tau_m']} alta={d['bands']['alta']} -> {args.out_dir}", file=sys.stderr)
    if "catastro_feature" in result["report"]:
        print(f"src_catastro: {result['report']['catastro_feature']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
