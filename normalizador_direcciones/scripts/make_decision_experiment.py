"""Build a candidate experiment directory that serves an existing one with the calibrated decision layer.

Every file of the source experiment directory is hardlinked (copied where links are unavailable) EXCEPT its
``experiment.json`` and any ``strict_eval_*`` / ``acceptance_*`` result files; ``decision.json`` and
``reliability_calibrated.json`` are copied from the parameter files; a fresh ``experiment.json`` records the lineage and
the sha256 of the parameters. The source directory and the parameter files are never modified. The parameters are
validated with the same loaders the service uses, so a directory that cannot switch the layer on is never created.

Usage:
  python scripts/make_decision_experiment.py --source artifacts/experiments/v2e_cert_pcer_s42_r3 \
      --out artifacts/experiments/v2e_cert_pcer_s42_r3_cal \
      --decision artifacts/experiments/calib_design/params/decision_cand.json \
      --reliability artifacts/experiments/calib_design/params/reliability_calibrated_cand.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.decision import (  # noqa: E402
    CALIBRATED_FILENAME, DECISION_FILENAME, load_calibrated, load_decision,
)

#: Files of a source experiment that are not part of the served artifacts.
_SKIP_NAMES = {"experiment.json", DECISION_FILENAME, CALIBRATED_FILENAME, "run.log"}
_SKIP_PREFIXES = ("strict_eval_", "acceptance")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _place(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def validate_pair(decision_path: str, reliability_path: str):
    """``(decision, reliability)`` loaded with the serving loaders, or an exception that says why they cannot be used.

    ``FileNotFoundError`` for a missing file; ``ValueError`` for an invalid decision, an unusable reliability file or a
    version mismatch (so a pair that would not switch the layer on is never installed).
    """
    for path in (decision_path, reliability_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"parameter file not found: {path}")
    decision = load_decision(decision_path)
    if decision is None:
        raise ValueError(f"decision file {decision_path} is not usable (invalid or malformed)")
    reliability = load_calibrated(reliability_path)
    if reliability is None:
        raise ValueError(f"reliability file {reliability_path} is not usable (kind logistic_ext expected)")
    if decision["reliability_version"] != reliability.version:
        raise ValueError(
            f"version mismatch: decision wants {decision['reliability_version']!r}, reliability file is {reliability.version!r}")
    return decision, reliability


def install_pair(out_dir: str, decision_path: str, reliability_path: str, decision: dict, reliability) -> dict:
    """Copy the pair into ``out_dir`` and return the ``decision_provenance`` block for ``experiment.json``."""
    shutil.copyfile(decision_path, os.path.join(out_dir, DECISION_FILENAME))
    shutil.copyfile(reliability_path, os.path.join(out_dir, CALIBRATED_FILENAME))
    return {
        "decision_version": decision["version"], "reliability_version": reliability.version,
        "decision_sha256": _sha256(decision_path), "reliability_sha256": _sha256(reliability_path),
        "tau_p": decision["tau_p"], "tau_m": decision["tau_m"], "confianza_floor": decision["confianza_floor"],
        "bands": decision["bands"], "fit_provenance": decision.get("fit_provenance", {}),
    }


def build_calibrated_experiment(source_dir: str, out_dir: str, decision_path: str, reliability_path: str,
                                tag: str | None = None) -> dict:
    """Create ``out_dir`` and return the ``experiment.json`` payload written into it.

    ``FileNotFoundError`` for a missing source or parameter file, ``FileExistsError`` for an existing ``out_dir``,
    ``ValueError`` when the pair would not switch the layer on (invalid decision, unusable reliability file, version
    mismatch). Nothing is created in any of those cases.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"source experiment directory not found: {source_dir}")
    if os.path.lexists(out_dir):
        raise FileExistsError(f"{out_dir} already exists")
    decision, reliability = validate_pair(decision_path, reliability_path)
    source_meta: dict = {}
    source_json = os.path.join(source_dir, "experiment.json")
    if os.path.isfile(source_json):
        with open(source_json, "r", encoding="utf-8") as fh:
            source_meta = json.load(fh)
    os.makedirs(out_dir)
    for name in sorted(os.listdir(source_dir)):
        src = os.path.join(source_dir, name)
        if name in _SKIP_NAMES or name.startswith(_SKIP_PREFIXES) or not os.path.isfile(src):
            continue
        _place(src, os.path.join(out_dir, name))
    provenance = install_pair(out_dir, decision_path, reliability_path, decision, reliability)
    payload = {key: value for key, value in source_meta.items() if key not in ("tag", "created")}
    payload.update({
        "tag": tag or os.path.basename(os.path.abspath(out_dir)),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "inherits_from_experiment": source_meta.get("tag", os.path.basename(os.path.abspath(source_dir))),
        "decision": "calibrated",
        "decision_provenance": provenance,
    })
    if "inherits_model_from" not in payload:
        payload["inherits_model_from"] = None
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="existing experiment directory to serve (never modified)")
    ap.add_argument("--out", required=True, help="new experiment directory (must not exist)")
    ap.add_argument("--decision", required=True, help="decision.json to install")
    ap.add_argument("--reliability", required=True, help="reliability_calibrated.json to install")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    try:
        payload = build_calibrated_experiment(args.source, args.out, args.decision, args.reliability, args.tag)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"created {args.out} ({payload['tag']}) from {payload['inherits_from_experiment']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
