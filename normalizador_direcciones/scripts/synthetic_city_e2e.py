"""End-to-end proof on an invented city: generate -> adapter -> index -> short CPU training -> evaluation.

    python scripts/synthetic_city_e2e.py --work-dir runs/synthetic --sizes 500,2000,8000 [--epochs 3]

For every size it writes ``<work-dir>/n<size>/`` (records CSV, mapping, staging, index with the trained model and
``eval_city.json``) and finally ``<work-dir>/synthetic_city_summary.json`` with one aggregate row per size
(documents, adapter / index / training / evaluation seconds, recall and precision figures). Everything is invented
(``cali_address.synthetic_city``); the work directory holds no real data, but it is a run output: do not commit it.

The training is deliberately small (a 0.1 M-parameter encoder, a few epochs on CPU): it proves the flow runs and
learns beyond chance, NOT production quality. Exit codes: 0 ok, 1 a stage failed (its exit code is reported).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

#: Encoder of the smoke-sized training (production uses d_model 256, 4 layers, 8 epochs on GPU).
SMOKE_MODEL = {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_out": 64, "batch_size": 256}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_size(work_dir: str, size: int, epochs: int, max_minutes: float, seed: int, eval_docs: int,
             quiet: bool = False) -> dict:
    """Run the whole flow for one city size; returns the summary row (raises RuntimeError when a stage fails)."""
    import contextlib
    import io

    from cali_address.synthetic_city import default_mapping, make_city

    base = os.path.join(work_dir, f"n{size}")
    os.makedirs(base, exist_ok=True)
    row: dict = {"size_requested": size}
    sink = io.StringIO()
    redirect = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()

    t = time.time()
    records = make_city(size, seed=seed)
    records.to_csv(os.path.join(base, "registros.csv"), index=False)
    with open(os.path.join(base, "city.json"), "w", encoding="utf-8") as fh:
        json.dump(default_mapping("registros.csv"), fh, indent=2)
    row["records"] = int(len(records))
    row["seconds_generate"] = round(time.time() - t, 1)

    prepare = _script("prepare_city")
    t = time.time()
    with redirect:
        code = prepare.main(["--mapping", os.path.join(base, "city.json"), "--out-dir", os.path.join(base, "staging"),
                             "--index-dir", os.path.join(base, "index"), "--overwrite"])
    if code != 0:
        raise RuntimeError(f"prepare_city/build_index exited {code} for size {size}")
    row["seconds_adapter_and_index"] = round(time.time() - t, 1)
    with open(os.path.join(base, "staging", "adapter_report.json"), encoding="utf-8") as fh:
        adapter = json.load(fh)
    with open(os.path.join(base, "index", "index_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    row.update({"kept_rows": adapter["kept_rows"], "quarantined_rows": adapter["quarantined_rows"],
                "n_docs": meta["n_docs"], "train_queries_total": meta["queries"]["synthetic"] + meta["queries"]["official"]})

    train = _script("train")
    t = time.time()
    args = ["--out-dir", os.path.join(base, "index"), "--device", "cpu", "--epochs", str(epochs),
            "--max-minutes", str(max_minutes), "--seed", str(seed)]
    for key, value in SMOKE_MODEL.items():
        args += ["--" + key.replace("_", "-"), str(value)]
    with redirect:
        code = train.main(args)
    if code != 0:
        raise RuntimeError(f"train exited {code} for size {size}")
    row["seconds_train"] = round(time.time() - t, 1)
    with open(os.path.join(base, "index", "experiment.json"), encoding="utf-8") as fh:
        exp = json.load(fh)
    with open(os.path.join(base, "index", "train_report.json"), encoding="utf-8") as fh:
        row["epochs_completed"] = len(json.load(fh).get("history", []))
    row["train_val_recall@1"] = exp.get("metrics", {}).get("syn_recall@1")

    evaluate = _script("eval_city")
    t = time.time()
    result = evaluate.evaluate(os.path.join(base, "index"), n_docs=eval_docs, variants=2, device="cpu")
    with open(os.path.join(base, "index", "eval_city.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)
    row["seconds_eval"] = round(time.time() - t, 1)
    pipe = result["pipeline"]
    row.update({
        "eval_queries": result["queries"],
        "chance_recall@1": result["chance"]["recall@1"],
        "chance_recall@5": result["chance"]["recall@5"],
        "encoder_untrained_recall@1": result["encoder_untrained"]["recall@1"],
        "encoder_untrained_recall@5": result["encoder_untrained"]["recall@5"],
        "encoder_trained_recall@1": result["encoder_trained"]["recall@1"],
        "encoder_trained_recall@5": result["encoder_trained"]["recall@5"],
        "trained_minus_untrained_recall@1_ci": result["trained_minus_untrained_recall@1"]["ci"],
        "pipeline_recall@1": pipe["recall@1"], "pipeline_recall@5": pipe["recall@5"],
        "rules_only_recall@1": result["rules_only"]["recall@1"],
        "coverage": pipe["coverage"]["rate"], "parcel_precision": pipe["parcel_precision"]["rate"],
        "parcel_precision_ci": pipe["parcel_precision"]["ci95"], "block_precision": pipe["block_precision"]["rate"],
        "ece_confidence": pipe["ece_confidence_vs_top1_parcel"],
        "val_docs_queries": result["val_docs"]["queries"],
        "val_docs_encoder_recall@1": result["val_docs"]["encoder_trained_recall@1"],
    })
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--sizes", default="500,2000,8000", help="comma-separated target address counts")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=10.0, help="training time cap per size")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-docs", type=int, default=1000)
    ap.add_argument("--quiet", action="store_true", help="hide the stage logs")
    args = ap.parse_args(argv)
    try:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        print(f"error: --sizes must be integers, got {args.sizes!r}", file=sys.stderr)
        return 2
    if not sizes or min(sizes) < 50:
        print("error: --sizes needs at least one value >= 50", file=sys.stderr)
        return 2
    os.makedirs(args.work_dir, exist_ok=True)
    rows = []
    for size in sizes:
        try:
            row = run_size(args.work_dir, size, args.epochs, args.max_minutes, args.seed, args.eval_docs, args.quiet)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    import torch

    summary = {"model": SMOKE_MODEL, "epochs": args.epochs, "seed": args.seed,
               "hardware": {"device": "cpu", "torch_threads": torch.get_num_threads(), "machine": platform.machine(),
                            "processor": platform.processor(), "python": platform.python_version()},
               "rows": rows}
    with open(os.path.join(args.work_dir, "synthetic_city_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
