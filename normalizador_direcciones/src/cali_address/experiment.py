"""Experiment bookkeeping: config overrides, guards, experiment.json and the leaderboard.

Pure functions only (no GPU, no training) so they are unit-testable; the heavy
lifting lives in ``scripts/train.py`` and is orchestrated by
``scripts/run_experiment.py``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import time

from .paths import (
    DEFAULT_ARTIFACTS_DIR, INDEX_MARKER, IndexMetaError, code_fingerprint, is_default_dir, read_index_meta,
    sha256_file,
)
from .train import TrainConfig

__all__ = [
    "EXTRA_DEFAULTS",
    "LEADERBOARD_COLUMNS",
    "all_defaults",
    "parse_overrides",
    "safe_tag",
    "experiment_dir",
    "prepare_out_dir",
    "is_frozen_path",
    "guard_frozen",
    "final_epoch_metrics",
    "eval_metrics",
    "lot_metrics",
    "frozen_section",
    "leaderboard_row",
    "append_leaderboard",
    "write_experiment_json",
    "IndexValidationError",
    "DEFAULT_INDEX_MAX_MINUTES",
    "validate_index_dir",
    "index_meta_record",
    "link_index_files",
    "effective_sets",
    "check_training_complete",
]

#: Experiment-level knobs that are not fields of ``TrainConfig``.
EXTRA_DEFAULTS = {
    "realistic_noise": False,
    "real_pairs": "",        # path override for train_pairs.parquet ("" = default)
    "init_from": "",         # finetune: model.pt to start from ("" = production model.pt)
    "n_variants": 3,
    "real_split_seed": 42,   # fixed so every run reports on the same real validation fold
    "smoke": False,
}

LEADERBOARD_COLUMNS = [
    "tag", "timestamp", "mode", "smoke", "final",
    "syn_recall@1", "syn_recall@5", "syn_recall@20",
    "real_val_recall@1", "real_val_recall@5", "real_val_recall@20",
    "dev_ok", "dev_n", "dev_manzana_precision", "dev_predial_precision", "dev_lost_correct",
    "dev_gold_n", "dev_gold_ok", "dev_gold_ok_coverage",
    "dev_gold_manzana_precision", "dev_gold_predial_precision", "dev_gold_lost_correct",
    "seconds", "code_fingerprint",
    # Appended later. Only ever APPEND at the end: ``append_leaderboard`` upgrades an older file's header in place,
    # which only works while the old header is a prefix of this list.
    "stopped_early",         # True when training did not finish all epochs (or could not be verified)
    "index_id",              # name of the index directory a ``--index-dir`` run trained on
    # Lot-level predial (NPN[:21]) next to the exact-code columns above; ground truth is cadastral, lot-level.
    "dev_predial_lot_precision",
    "dev_gold_predial_lot_precision",
    "dev_gold_lot_missing_from_index",   # dev gold rows whose ground-truth lot is not among the index documents
    "final_gold_ok",                     # frozen_test gold, only for ``--final`` runs
    "final_gold_predial_precision",
    "final_gold_predial_lot_precision",
]

_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def all_defaults() -> dict:
    d = TrainConfig().to_dict()
    d.update(EXTRA_DEFAULTS)
    return d


def _coerce(raw: str, default):
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if default is None:
        return int(raw) if raw.lstrip("-").isdigit() else raw
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def parse_overrides(pairs, defaults: dict | None = None) -> dict:
    """``["lr=1e-3", "epochs=2"]`` -> ``{"lr": 0.001, "epochs": 2}``; unknown keys raise."""
    defaults = defaults if defaults is not None else all_defaults()
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip().replace("-", "_")
        if key not in defaults:
            raise ValueError(f"unknown hyper-parameter {key!r}; known: {sorted(defaults)}")
        out[key] = _coerce(raw.strip(), defaults[key])
    return out


def safe_tag(tag: str) -> str:
    if not tag or not _TAG_RE.match(tag):
        raise ValueError(f"invalid tag {tag!r}: use letters, digits, '_', '.', '-' (max 64)")
    return tag


def experiment_dir(tag: str, root: str | None = None) -> str:
    return os.path.join(root or os.path.join(DEFAULT_ARTIFACTS_DIR, "experiments"), safe_tag(tag))


def prepare_out_dir(out_dir: str, overwrite: bool = False) -> str:
    """Create ``out_dir``; refuse the production dir and (without ``overwrite``) a non-empty one."""
    if is_default_dir(out_dir):
        raise ValueError("refusing to use the production artifacts directory as an experiment directory")
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pick another --tag or pass --overwrite")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def is_frozen_path(path: str) -> bool:
    return "frozen_test" in os.path.basename(str(path)).lower()


def guard_frozen(scored_path: str, final: bool) -> None:
    """Raise unless ``--final`` was passed when ``scored_path`` is the frozen test set."""
    if is_frozen_path(scored_path) and not final:
        raise PermissionError(
            "frozen_test is the final measurement only; pass --final to score it "
            "(never use it for tuning or model selection)"
        )


def final_epoch_metrics(history: list[dict]) -> dict:
    """Recalls of the last epoch (synthetic held-out and real validation fold)."""
    last = history[-1] if history else {}
    out = {}
    for k in (1, 5, 20):
        out[f"syn_recall@{k}"] = last.get(f"recall@{k}")
        out[f"real_val_recall@{k}"] = last.get(f"real_val_recall@{k}")
    return out


def _all_ok_row(summary_rows: list[dict]) -> dict | None:
    for r in summary_rows:
        if r.get("dataset") == "ALL" and r.get("estado") == "OK":
            return r
    return None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def eval_metrics(summary_rows: list[dict], prefix: str) -> dict:
    """Headline numbers from ``strict_summary`` rows (dicts). ``n`` counts every ALL row."""
    total = sum(int(_num(r.get("n")) or 0) for r in summary_rows if r.get("dataset") == "ALL")
    lost = sum(int(_num(r.get("lost_correct")) or 0) for r in summary_rows if r.get("dataset") == "ALL")
    ok = _all_ok_row(summary_rows)
    return {
        f"{prefix}_n": total,
        f"{prefix}_ok": int(_num(ok.get("n")) or 0) if ok else 0,
        f"{prefix}_manzana_precision": _num(ok.get("manzana_precision")) if ok else None,
        f"{prefix}_predial_precision": _num(ok.get("predial_precision")) if ok else None,
        f"{prefix}_lost_correct": lost,
    }


def lot_metrics(summary_rows: list[dict], prefix: str) -> dict:
    """Lot-level numbers from ``strict_summary`` rows; kept apart from :func:`eval_metrics` so that one is unchanged.

    ``{prefix}_predial_lot_precision`` is the pooled OK row's ``predial_lot_precision`` and
    ``{prefix}_lot_missing_from_index`` the sum of ``n_lot_missing_from_index`` over the pooled ``ALL`` rows
    (every estado). A summary without those columns, or with blank / NaN / malformed cells, gives ``None``
    (unknown), never 0.
    """
    ok = _all_ok_row(summary_rows)
    missing = [_num(r.get("n_lot_missing_from_index")) for r in summary_rows if r.get("dataset") == "ALL"]
    known = [m for m in missing if m is not None]
    return {
        f"{prefix}_predial_lot_precision": _num(ok.get("predial_lot_precision")) if ok else None,
        f"{prefix}_lot_missing_from_index": int(sum(known)) if known else None,
    }


def frozen_section(summary_all: list[dict], summary_gold: list[dict]) -> dict:
    """The ``frozen_test`` block of ``experiment.json``: old ``test_*`` keys plus the lot-level ones."""
    return {"all": {**eval_metrics(summary_all, "test"), **lot_metrics(summary_all, "test")},
            "gold": {**eval_metrics(summary_gold, "test_gold"), **lot_metrics(summary_gold, "test_gold")}}


def leaderboard_row(tag: str, mode: str, history: list[dict], dev_all: list[dict],
                    dev_gold: list[dict], seconds: float, smoke: bool = False,
                    final: bool = False, fingerprint: str | None = None, index_id: str | None = None,
                    stopped_early: bool | None = None, final_gold: list[dict] | None = None) -> dict:
    row = {"tag": tag, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode,
           "smoke": bool(smoke), "final": bool(final)}
    row.update(final_epoch_metrics(history))
    a = eval_metrics(dev_all, "dev")
    g = eval_metrics(dev_gold, "dev_gold")
    row.update({k: a[k] for k in ("dev_ok", "dev_n", "dev_manzana_precision", "dev_predial_precision",
                                  "dev_lost_correct")})
    row.update({
        "dev_gold_n": g["dev_gold_n"],
        "dev_gold_ok": g["dev_gold_ok"],
        "dev_gold_ok_coverage": (g["dev_gold_ok"] / g["dev_gold_n"]) if g["dev_gold_n"] else None,
        "dev_gold_manzana_precision": g["dev_gold_manzana_precision"],
        "dev_gold_predial_precision": g["dev_gold_predial_precision"],
        "dev_gold_lost_correct": g["dev_gold_lost_correct"],
    })
    row["seconds"] = round(float(seconds), 1)
    row["code_fingerprint"] = (fingerprint or code_fingerprint())[:12]
    row["stopped_early"] = stopped_early
    row["index_id"] = index_id
    la, lg = lot_metrics(dev_all, "dev"), lot_metrics(dev_gold, "dev_gold")
    row["dev_predial_lot_precision"] = la["dev_predial_lot_precision"]
    row["dev_gold_predial_lot_precision"] = lg["dev_gold_predial_lot_precision"]
    row["dev_gold_lot_missing_from_index"] = lg["dev_gold_lot_missing_from_index"]
    if final_gold:
        f, fl = eval_metrics(final_gold, "final_gold"), lot_metrics(final_gold, "final_gold")
        row["final_gold_ok"] = f["final_gold_ok"]
        row["final_gold_predial_precision"] = f["final_gold_predial_precision"]
        row["final_gold_predial_lot_precision"] = fl["final_gold_predial_lot_precision"]
    else:
        row["final_gold_ok"] = row["final_gold_predial_precision"] = row["final_gold_predial_lot_precision"] = None
    return row


def append_leaderboard(path: str, row: dict) -> None:
    """Append one row; unknown keys are ignored. A new/empty file gets the full header.

    An existing file keeps every byte it has. If its header is an older, shorter version of
    :data:`LEADERBOARD_COLUMNS` (a prefix of it) the header line alone is upgraded, and only when the new row
    actually fills one of the added columns; its old rows stay as they are (readers see the new columns as
    missing). A header that is not a prefix is never rewritten: the row is written in that file's own columns.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _write_rows(path, "w", LEADERBOARD_COLUMNS, row, "\r\n", header=True)
        return
    with open(path, "rb") as fh:
        raw = fh.read()
    nl = raw.find(b"\n")
    header_bytes = raw if nl < 0 else raw[:nl]
    terminator = "\n" if (nl >= 0 and not header_bytes.endswith(b"\r")) else "\r\n"
    header = next(csv.reader([header_bytes.decode("utf-8").rstrip("\r")]), [])
    columns = header
    if header != LEADERBOARD_COLUMNS and LEADERBOARD_COLUMNS[:len(header)] == header:
        added = LEADERBOARD_COLUMNS[len(header):]
        if any(row.get(k) not in (None, "") for k in added):
            rest = b"" if nl < 0 else raw[nl + 1:]
            raw = (",".join(LEADERBOARD_COLUMNS) + terminator).encode("utf-8") + rest
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, path)
            columns = LEADERBOARD_COLUMNS
    glue = "" if raw.endswith(b"\n") else terminator          # never glue the new row to an unterminated last row
    _write_rows(path, "a", columns, row, terminator, header=False, prefix=glue)


def _write_rows(path: str, mode: str, columns: list[str], row: dict, terminator: str, header: bool,
                prefix: str = "") -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator=terminator)
    if header:
        w.writeheader()
    w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    with open(path, mode, newline="", encoding="utf-8") as fh:
        fh.write(prefix + buf.getvalue())


def write_experiment_json(out_dir: str, payload: dict) -> str:
    """Merge ``payload`` into ``<out_dir>/experiment.json`` (later calls add sections)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "experiment.json")
    current: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                current = json.load(fh)
        except (OSError, ValueError):
            current = {}
    current.update(payload)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------------------------------------
# --index-dir support (scripts/run_experiment.py)
# ---------------------------------------------------------------------------------------------------------

class IndexValidationError(ValueError):
    """An index directory is not complete or not trustworthy; the message says which check failed."""


#: Wall-clock budget (minutes) a ``--index-dir`` run gets unless the user passes ``--set max_minutes=...``.
#: A 330k-document index took ~16 minutes for 8 epochs; the union indexes are about twice as big.
DEFAULT_INDEX_MAX_MINUTES = 120

#: Files without which a training run cannot start; an index that does not list them is not usable.
_REQUIRED_INDEX_FILES = ("catastro_docs.parquet", "train_data.npz", "spatial_negatives.npy")


def _plain_name(name: object) -> bool:
    return (isinstance(name, str) and name not in ("", ".", "..") and name == os.path.basename(name)
            and "/" not in name and "\\" not in name and name != INDEX_MARKER)


def validate_index_dir(index_dir: str) -> dict:
    """Return the parsed ``index_meta.json`` of a COMPLETE index or raise :class:`IndexValidationError`.

    Complete means: the directory exists, the marker parses, ``files`` maps plain file names to
    ``{sha256, ...}``, the training inputs are listed, ``n_docs`` is a non-negative integer and every listed
    file exists with exactly the recorded sha256. Nothing is read from outside the directory.
    """
    if not os.path.isdir(index_dir):
        raise IndexValidationError(f"{index_dir} is not a directory")
    try:
        meta = read_index_meta(index_dir)
    except IndexMetaError as exc:
        raise IndexValidationError(str(exc)) from exc
    if meta is None:
        raise IndexValidationError(f"{INDEX_MARKER} not found in {index_dir}: the index build did not finish")
    files = meta.get("files")
    if not isinstance(files, dict) or not files:
        raise IndexValidationError(f"{INDEX_MARKER} in {index_dir} has no non-empty 'files' mapping")
    for name, info in files.items():
        if not _plain_name(name):
            raise IndexValidationError(f"{name!r} in {INDEX_MARKER} 'files' is not a plain file name")
        if not isinstance(info, dict) or not isinstance(info.get("sha256"), str) or not info["sha256"]:
            raise IndexValidationError(f"files[{name!r}] in {INDEX_MARKER} has no sha256")
    n_docs = meta.get("n_docs")
    if isinstance(n_docs, bool) or not isinstance(n_docs, int) or n_docs < 0:
        raise IndexValidationError(f"{INDEX_MARKER} must declare a non-negative integer n_docs, got {n_docs!r}")
    for name in _REQUIRED_INDEX_FILES:
        if name not in files:
            raise IndexValidationError(f"{name} is not listed in the 'files' of {INDEX_MARKER}")
    for name, info in files.items():
        path = os.path.join(index_dir, name)
        if not os.path.isfile(path):
            raise IndexValidationError(f"listed file {name} is missing from {index_dir}")
        found = sha256_file(path)
        if found != info["sha256"]:
            raise IndexValidationError(
                f"{name}: sha256 mismatch ({INDEX_MARKER} says {info['sha256'][:12]}..., the file is {found[:12]}...)"
            )
    return meta


def index_meta_record(index_dir: str, meta: dict) -> dict:
    """What ``experiment.json`` stores about the index a run used (counts and hashes, nothing row-level)."""
    return {
        "index_id": os.path.basename(os.path.abspath(index_dir)),
        "n_docs": meta.get("n_docs"),
        "tiers": meta.get("tiers"),
        "sha256": sha256_file(os.path.join(index_dir, INDEX_MARKER)),
        "n_files": len(meta.get("files", {})),
        "pre_normalize_separates_letters": meta.get("pre_normalize_separates_letters"),
    }


def link_index_files(index_dir: str, out_dir: str, meta: dict, overwrite: bool = False) -> list[str]:
    """Make ``out_dir`` self-contained: hardlink (copy as fallback) every listed file, the marker LAST.

    Without ``overwrite`` an existing destination file is an error and nothing is touched. With it the
    destination is replaced by a fresh link/copy (never written through, so the source index cannot change);
    a file that already is the same inode as its source is left alone, so re-running is idempotent. The old
    marker is removed first, so a failure midway leaves a directory that does not look like a built index.
    Returns the names in the order they were placed (the marker is the last).
    """
    names = list(meta["files"])
    marker_dst = os.path.join(out_dir, INDEX_MARKER)
    if not overwrite:
        clash = [n for n in names + [INDEX_MARKER] if os.path.lexists(os.path.join(out_dir, n))]
        if clash:
            raise FileExistsError(f"{out_dir} already has {', '.join(clash)}; pass --overwrite to replace them")
    elif os.path.lexists(marker_dst):
        os.remove(marker_dst)
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        src, dst = os.path.join(index_dir, name), os.path.join(out_dir, name)
        if os.path.lexists(dst):
            if os.path.exists(dst) and os.path.samefile(src, dst):
                continue
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    shutil.copy2(os.path.join(index_dir, INDEX_MARKER), marker_dst)
    return names + [INDEX_MARKER]


def effective_sets(sets: list[str] | None, index_dir: str | None) -> list[str]:
    """The ``--set`` list handed to ``train.py``: an index run defaults ``max_minutes`` to 120 (explicit wins)."""
    out = list(sets or [])
    if index_dir and not any(item.partition("=")[0].strip().replace("-", "_") == "max_minutes" for item in out):
        out.append(f"max_minutes={DEFAULT_INDEX_MAX_MINUTES}")
    return out


def check_training_complete(report: dict | None, configured_epochs: int | None = None) -> dict:
    """Did training run every configured epoch? ``{ok, stopped_early, epochs_configured, epochs_completed, problems}``.

    Reads ``train_report.json`` (``stopped_early``/``epochs_configured``/``epochs_completed``; older reports
    only have ``history``, whose length is then the completed count). A missing or empty report cannot be
    verified and counts as a failure: a silently shortened run is worse than a loud one.
    """
    result = {"ok": False, "stopped_early": False, "epochs_configured": configured_epochs,
              "epochs_completed": None, "problems": []}
    if not isinstance(report, dict) or not report:
        result["problems"].append("train_report.json is missing or empty; cannot verify that training finished")
        return result
    history = report.get("history")
    configured = report.get("epochs_configured", configured_epochs)
    completed = report.get("epochs_completed", len(history) if isinstance(history, list) else None)
    result.update(epochs_configured=configured, epochs_completed=completed,
                  stopped_early=bool(report.get("stopped_early", False)))
    if not completed and "stopped_early" not in report:
        result["problems"].append("train_report.json records no completed epoch")
    elif result["stopped_early"]:
        result["problems"].append(
            f"training stopped early (time budget): {completed} of {configured} epochs completed")
    elif configured is not None and completed is not None and completed < configured:
        result["problems"].append(f"only {completed} of {configured} epochs completed")
    result["ok"] = not result["problems"]
    return result
