"""CPU benchmark of the serving backends (torch / onnx / onnx-int8) through ``normalize_strict``.

For every ``--set NAME=DIR`` (a serving directory that holds model.pt, catastro_emb.pt and the ONNX files) and
every backend it launches ONE FRESH PYTHON PROCESS (so cold start and peak RSS are not polluted by the previous
backend) with the thread count pinned (``--threads``, default: physical cores) and measures:

* cold start: constructing ``AddressNormalizer`` + the first answer (imports are reported apart),
* single-address latency over ``--latency-n`` queries (p50 / p95 of the median repeat),
* throughput for batches of ``--batch-sizes`` (median and p95 over ``--repeats``, after a warm-up),
* peak resident memory of the process (Windows peak working set, else ``ru_maxrss``).

Queries are synthetic: public document strings (``direccion`` of the set's ``catastro_docs.parquet``) corrupted
with ``augment.corrupt`` under a fixed seed; no split file is ever read. Results go to ``--out`` (bench.json)
and a table is printed. ``--no-sizes`` skips the installed-size report of torch vs onnxruntime. See docs/onnx.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

BACKENDS = ("torch", "onnx", "onnx-int8")


# --------------------------------------------------------------------------------------------
# helpers shared by the orchestrator and the worker
# --------------------------------------------------------------------------------------------
def physical_cores() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=60).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return int(out)
        except (OSError, subprocess.SubprocessError):
            pass
    return max(1, (os.cpu_count() or 2) // 2)


def peak_rss_mb() -> float:
    """Peak resident set of this process in MB."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e6
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e3 if sys.platform != "darwin" else peak / 1e6


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


# --------------------------------------------------------------------------------------------
# worker: one backend of one set, in its own process
# --------------------------------------------------------------------------------------------
def worker(args) -> dict:
    t_start = time.perf_counter()
    os.environ["NORMALIZER_BACKEND"] = args.backend
    os.environ["NORMALIZER_ONNX_THREADS"] = str(args.threads)
    import numpy as np
    import pandas as pd

    from cali_address.augment import corrupt_many
    from cali_address.inference import AddressNormalizer
    from cali_address.paths import shared_path
    from cali_address.service import DEFAULT_BASEMAPS_DIR, load_gazetteer, normalize_strict

    if args.backend == "torch":
        import torch

        torch.set_num_threads(args.threads)
    import_s = time.perf_counter() - t_start

    docs = pd.read_parquet(shared_path("catastro_docs.parquet", args.artifacts_dir), columns=["direccion"])
    pool_n = min(args.pool_n, len(docs))
    base = docs["direccion"].astype(str).sample(n=pool_n, random_state=args.seed).tolist()
    rng = random.Random(args.seed)
    queries = [corrupt_many(d, 2, rng)[i % 2] for i, d in enumerate(base)]

    gazetteer = None if args.no_gazetteer else load_gazetteer(DEFAULT_BASEMAPS_DIR)

    t0 = time.perf_counter()
    norm = AddressNormalizer(args.artifacts_dir, device="cpu", backend=args.backend)
    construct_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    normalize_strict(norm, [queries[0]], gazetteer=gazetteer)
    first_answer_s = time.perf_counter() - t0
    rss_after_load_mb = None  # peak so far is the honest figure here; reported below as peak_rss_after_cold_mb
    peak_after_cold = peak_rss_mb()

    # warm-up (parse caches, allocator, thread pools): not timed
    normalize_strict(norm, queries[: max(1, min(args.warmup_n, len(queries)))], gazetteer=gazetteer)

    lat_p50, lat_p95 = [], []
    for _ in range(args.repeats):
        samples = []
        for q in queries[: args.latency_n]:
            t0 = time.perf_counter()
            normalize_strict(norm, [q], gazetteer=gazetteer)
            samples.append((time.perf_counter() - t0) * 1000)
        lat_p50.append(percentile(samples, 0.50))
        lat_p95.append(percentile(samples, 0.95))

    throughput = {}
    for size in args.batch_sizes:
        batch = [queries[i % len(queries)] for i in range(size)]
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            normalize_strict(norm, batch, gazetteer=gazetteer)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        throughput[str(size)] = {
            "seconds_median": med, "seconds_p95": percentile(times, 0.95), "rows_per_s_median": size / med,
            "seconds_all": times,
        }
    return {
        "backend": args.backend, "threads": args.threads, "n_docs": int(len(docs)),
        "import_s": import_s, "construct_s": construct_s, "first_answer_s": first_answer_s,
        "cold_start_s": construct_s + first_answer_s,
        "latency_ms": {"p50": statistics.median(lat_p50), "p95": statistics.median(lat_p95), "n": args.latency_n,
                       "p50_repeats": lat_p50, "p95_repeats": lat_p95},
        "throughput": throughput,
        "peak_rss_after_cold_mb": peak_after_cold, "peak_rss_mb": peak_rss_mb(),
        "torch_imported": "torch" in sys.modules,
        "gazetteer": gazetteer is not None, "repeats": args.repeats,
        "rss_note": rss_after_load_mb,
    }


# --------------------------------------------------------------------------------------------
# installed sizes (torch vs onnxruntime) and the torch-free image estimate
# --------------------------------------------------------------------------------------------
def _canon(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _requires(dist) -> list[str]:
    from packaging.requirements import Requirement

    out = []
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        out.append(_canon(req.name))
    return out


def _closure(names) -> set[str]:
    from importlib import metadata

    seen: set[str] = set()
    todo = [_canon(n) for n in names]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen.add(name)
        todo.extend(_requires(dist))
    return seen


def _dist_size(name: str) -> int:
    from importlib import metadata

    dist = metadata.distribution(name)
    total = 0
    for f in dist.files or []:
        path = dist.locate_file(f)
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def installed_sizes() -> dict:
    """Bytes in site-packages of the torch stack and the onnxruntime stack, net of the shared base."""
    base = _closure(["numpy", "pandas", "pyarrow", "rapidfuzz", "shapely", "pyproj", "pyshp", "pydantic"])
    torch_set = _closure(["torch"]) - base
    ort_set = _closure(["onnxruntime"]) - base
    export_set = _closure(["onnx"]) - base - ort_set
    sizes = {n: _dist_size(n) for n in torch_set | ort_set | export_set}
    return {
        "method": "sum of the files each distribution installs (importlib.metadata), for the transitive "
                  "requirement closure of the package minus the closure of the shared base requirements",
        "torch": {"bytes": sum(sizes[n] for n in torch_set), "packages": sorted(torch_set)},
        "onnxruntime": {"bytes": sum(sizes[n] for n in ort_set), "packages": sorted(ort_set)},
        "onnx_export_only": {"bytes": sum(sizes[n] for n in export_set), "packages": sorted(export_set)},
    }


def image_estimate(sizes: dict, image_gb: float = 2.33, torch_layer_gb: float = 0.93) -> dict:
    """Serving image without torch = current image - torch layer + onnxruntime stack (serving needs no ``onnx``)."""
    ort_gb = sizes["onnxruntime"]["bytes"] / 1e9
    return {
        "current_image_gb": image_gb, "torch_layer_gb": torch_layer_gb, "onnxruntime_stack_gb": round(ort_gb, 3),
        "estimated_torch_free_image_gb": round(image_gb - torch_layer_gb + ort_gb, 2),
        "method": "measured current image on disk (2.33 GB) minus its torch CPU layer (0.93 GB) plus the "
                  "onnxruntime stack measured here; the model artifacts are unchanged in size "
                  "(model.onnx 13 MB / int8 3.6 MB replace model.pt 14 MB). Real value must be confirmed with docker build.",
    }


# --------------------------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------------------------
def _print_table(runs: dict, batch_sizes) -> None:
    head = ["set", "BACKEND", "cold s", "p50 ms", "p95 ms"] + [f"{b} rows/s" for b in batch_sizes] + ["peak RSS MB"]
    lines = [head]
    for name, per_backend in runs.items():
        for backend, r in per_backend.items():
            lines.append([name, backend, f"{r['cold_start_s']:.2f}", f"{r['latency_ms']['p50']:.1f}",
                          f"{r['latency_ms']['p95']:.1f}"]
                         + [f"{r['throughput'][str(b)]['rows_per_s_median']:.0f}" for b in batch_sizes]
                         + [f"{r['peak_rss_mb']:.0f}"])
    widths = [max(len(row[i]) for row in lines) for i in range(len(head))]
    for row in lines:
        print("  ".join(cell.rjust(w) for cell, w in zip(row, widths)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=DIR", help="serving directory (repeatable)")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--threads", type=int, default=None, help="intra-op threads (default: physical cores)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--latency-n", type=int, default=500)
    ap.add_argument("--batch-sizes", default="100,2000,10000")
    ap.add_argument("--warmup-n", type=int, default=2000)
    ap.add_argument("--pool-n", type=int, default=10000, help="synthetic queries to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gazetteer", action="store_true", help="skip the basemap gazetteer (tests only)")
    ap.add_argument("--no-sizes", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="keep the runs already in --out for sets/backends that are not measured now")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "experiments", "onnx_bench", "bench.json"))
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", help=argparse.SUPPRESS)
    ap.add_argument("--artifacts-dir", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.batch_sizes = [int(b) for b in str(args.batch_sizes).split(",") if b.strip()]
    if args.threads is None:
        args.threads = physical_cores()
    if args.repeats < 1 or args.latency_n < 1:
        ap.error("--repeats and --latency-n must be >= 1")

    if args.worker:
        print("RESULT " + json.dumps(worker(args)))
        return 0

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            print(f"error: unknown backend {b!r}; valid values: {', '.join(BACKENDS)}", file=sys.stderr)
            return 2
    sets = {}
    for item in args.set:
        name, _, path = item.partition("=")
        if not name or not path or not os.path.isdir(path):
            print(f"error: --set expects NAME=DIR with an existing directory, got {item!r}", file=sys.stderr)
            return 2
        sets[name] = os.path.abspath(path)
    if not sets:
        print("error: give at least one --set NAME=DIR", file=sys.stderr)
        return 2

    runs: dict = {name: {} for name in sets}
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(args.threads)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    for name, path in sets.items():
        for backend in backends:
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--backend", backend, "--artifacts-dir", path,
                   "--threads", str(args.threads), "--repeats", str(args.repeats), "--latency-n", str(args.latency_n),
                   "--batch-sizes", ",".join(map(str, args.batch_sizes)), "--warmup-n", str(args.warmup_n),
                   "--pool-n", str(args.pool_n), "--seed", str(args.seed)]
            if args.no_gazetteer:
                cmd.append("--no-gazetteer")
            print(f"[bench] {name} / {backend} ...", file=sys.stderr, flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
            if proc.returncode != 0 or not lines:
                print(f"error: worker {name}/{backend} failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
                return 1
            runs[name][backend] = json.loads(lines[-1][len("RESULT "):])

    result = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "physical_cores": physical_cores(), "logical_cores": os.cpu_count(), "python": sys.version.split()[0]},
        "config": {"threads": args.threads, "repeats": args.repeats, "latency_n": args.latency_n,
                   "batch_sizes": args.batch_sizes, "warmup_n": args.warmup_n, "pool_n": args.pool_n, "seed": args.seed,
                   "sets": sets, "queries": "public doc strings corrupted with augment.corrupt (mild/strong alternating)"},
        "runs": runs,
    }
    if not args.no_sizes:
        result["installed_sizes"] = installed_sizes()
        result["image_estimate"] = image_estimate(result["installed_sizes"])
    if args.merge and os.path.isfile(args.out):
        with open(args.out, "r", encoding="utf-8") as fh:
            previous = json.load(fh)
        for name, per_backend in previous.get("runs", {}).items():
            merged = dict(per_backend)
            merged.update(runs.get(name, {}))
            runs[name] = merged
        result["config"]["sets"] = {**previous.get("config", {}).get("sets", {}), **sets}
        for key in ("installed_sizes", "image_estimate"):
            if key not in result and key in previous:
                result[key] = previous[key]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    _print_table(runs, args.batch_sizes)
    if "installed_sizes" in result:
        s = result["installed_sizes"]
        print(f"installed: torch {s['torch']['bytes'] / 1e6:.0f} MB, onnxruntime {s['onnxruntime']['bytes'] / 1e6:.0f} MB, "
              f"onnx (export only) {s['onnx_export_only']['bytes'] / 1e6:.0f} MB; "
              f"estimated torch-free image {result['image_estimate']['estimated_torch_free_image_gb']} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
