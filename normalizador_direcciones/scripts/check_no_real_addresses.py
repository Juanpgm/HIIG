"""Privacy guard: fail if real citizen addresses or identifiers leak into tracked files.

Loads the private evaluation/survey data that may exist on a developer machine
(read-only, in memory only), builds a set of normalized real raw addresses plus
real person names, record ids, document/phone numbers and GPS points, then
scans every tracked source/test/doc/script/notebook file for them.

It never writes or prints a real value: findings are reported as
``path:line kind [sources]`` only.

Address matching compares two normalized keys:

* ``full``: the whole address, casefolded, accent-stripped, punctuation
  collapsed (near-verbatim match);
* ``core``: via type + via number + cross number + plate (letters/bis kept,
  quadrants and fillers dropped), so "Calle 9 # 9-99" and "CL 9 9 99 apto 3"
  share the core ``cl|9|9|99``.

The public cadastre is not citizen data. A real address whose core also exists
in the cadastre and whose private sources carry no person or record context (no
names, ids or survey fields: "low-context" sources) is treated as a public cadastre
string and is not reported. A cadastre string that also appears as a raw survey or
registry input with citizen context is reported.

Private sources are discovered, not hard-coded. Under the data root (default: the
parent of this repository; override with ``CALI_PRIVATE_DATA_ROOT``) every project
directory may hold ``context/*.xlsx`` workbooks, ``data/raw/*.parquet`` surveys,
``artifacts/splits/*.parquet`` evaluation splits and so on (see
``PRIVATE_SOURCE_GLOBS``; override the workbook pattern with
``CALI_PRIVATE_CONTEXT_GLOB``). Workbooks are analysed by column names, not by file
name, and are labelled ``source_a``, ``source_b``, ... in the report.

Besides sources, tests, docs and scripts, the generated ``artifacts/`` and ``docs/``
folders are scanned too (``.csv`` and ``.json`` included).

Exit codes: 0 clean, 1 hits found, 2 no private data available (nothing checked).

Usage::

    python scripts/check_no_real_addresses.py            # this repository
    python scripts/check_no_real_addresses.py --all      # every project under the data root
    python scripts/check_no_real_addresses.py --root PATH [--root PATH ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.environ.get("CALI_PRIVATE_DATA_ROOT") or REPO_ROOT.parent)


def sibling_projects(models_root: Path = MODELS_ROOT) -> list[Path]:
    """Project directories under the data root (they carry a pyproject or a requirements file)."""
    if not models_root.exists():
        return []
    return sorted(
        d for d in models_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and ((d / "pyproject.toml").exists() or (d / "requirements.txt").exists())
    )


# Directories that hold raw citizen data: never scanned (they are the source of truth
# for "real", not a leak). ``artifacts/`` and ``docs/`` ARE scanned: they are published.
EXCLUDED_DIRS = {
    ".venv", "venv", ".git", "data", "outputs", "context",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__", ".atl",
    "node_modules", "build", "dist",
}
BINARY_SUFFIXES = {
    ".parquet", ".xlsx", ".xls", ".jsonl", ".joblib", ".pkl", ".pt",
    ".npz", ".npy", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz",
    ".onnx", ".bin", ".lock", ".whl", ".ico", ".svg", ".db", ".sqlite",
}

# Private sources whose rows carry no person/record context are tracked in
# ``RealData.low_context``: an address that only appears there AND in the public
# cadastre is a public cadastre string.

# Public place names that some free-text "name" fields contain verbatim.
PUBLIC_PHRASES = {
    "santiago de cali", "valle del cauca", "cali valle del cauca",
    # Generic Spanish connective phrases that stray free-text cells may contain.
    "de la cadena", "de la fuente",
}

VIA_ALIASES = {
    "cl": ("calle", "cl", "cll", "call", "clle", "cal"),
    "kr": ("carrera", "cra", "cr", "kr", "kra", "carr", "crr", "krr"),
    "av": ("avenida", "av", "avda", "ave"),
    "dg": ("diagonal", "dg", "diag", "dig"),
    "tv": ("transversal", "tv", "tr", "trans", "transv", "tranv", "trv", "trasv"),
    "cq": ("circular", "cq", "circ"),
    "pj": ("pasaje", "psje", "pj"),
    "ac": ("ac",),
    "ak": ("ak",),
}
VIA = {alias: canon for canon, aliases in VIA_ALIASES.items() for alias in aliases}
FILLERS = {"no", "n", "nro", "num", "numero", "nu", "o"}
QUADRANTS = {"sur", "norte", "oeste", "este", "oe", "occidente", "oriente"}
NUM_RE = re.compile(r"^\d{1,4}[a-z]{0,2}$")
FLOAT_RE = re.compile(r"-?\d{1,3}\.\d{5,}")
OBJECTID_RE = re.compile(r"(?:object_?id|id_edan|ede_object_id)\W{0,4}(\d{3,7})\b", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9@._\-]+")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)  # "cra24" -> "cra 24"
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def cores_from_tokens(toks: list[str]) -> list[str]:
    """Core keys (via|number|cross|plate) of every address-like run of tokens."""
    out = []
    consumed = -1  # index of a via token already absorbed by "avenida calle/carrera"
    for i, tok in enumerate(toks):
        canon = VIA.get(tok)
        if canon is None or i == consumed:
            continue
        j = i + 1
        if canon == "av" and j < len(toks) and VIA.get(toks[j]) in ("cl", "kr"):
            canon = "ac" if VIA[toks[j]] == "cl" else "ak"
            consumed = j
            j += 1
        nums: list[str] = []
        while j < len(toks) and len(nums) < 3 and j <= i + 12:
            t = toks[j]
            if NUM_RE.match(t):
                nums.append(t)
            elif nums and (t == "bis" or (len(t) == 1 and t.isalpha() and t not in FILLERS)):
                nums[-1] += t
            elif t not in FILLERS and t not in QUADRANTS:
                break
            j += 1
        if len(nums) == 3:
            out.append(canon + "|" + "|".join(nums))
    return out


def address_keys(text: str) -> tuple[str, list[str]]:
    n = norm(text)
    return n, cores_from_tokens(n.split())


def _is_lat(v: float) -> bool:
    return 2.5 < v < 4.5


def _is_lon(v: float) -> bool:
    return -77.5 < v < -75.5


# --------------------------------------------------------------------------- data


@dataclass
class RealData:
    full: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    core: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    names: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    object_ids: set[str] = field(default_factory=set)
    points: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))
    cadastre_core: set[str] = field(default_factory=set)
    sources_loaded: list[str] = field(default_factory=list)
    low_context: set[str] = field(default_factory=set)

    def empty(self) -> bool:
        return not self.full

    def is_public(self, cores: list[str], sources: set[str]) -> bool:
        return bool(cores) and sources <= self.low_context and all(
            c in self.cadastre_core for c in cores
        )


def _values(df, col):
    if col not in df.columns:
        return []
    return [v for v in df[col].tolist() if v is not None and v == v and str(v).strip()]


def _cols(df, *cols):
    return [v for c in cols for v in _values(df, c)]


def _add_addresses(rd: RealData, values, source: str) -> None:
    for v in values:
        full, cores = address_keys(str(v))
        toks = full.split()
        # Numeric addresses need 3+ numbers; rural/descriptive ones (few or no
        # numbers) qualify when long enough to be specific.
        if sum(any(ch.isdigit() for ch in t) for t in toks) >= 3 or (
            len(toks) >= 5 and len(full) >= 25 and any(ch.isdigit() for ch in full)
        ):
            rd.full[full].add(source)
        for c in cores:
            rd.core[c].add(source)


def _add_person_names(rd: RealData, values, source: str) -> None:
    """Full person names (3+ tokens); 2-token names collide with barrio names."""
    for v in values:
        if "@" in str(v):  # e-mail/user handle: matched as an exact id
            rd.ids[str(v).strip().casefold()].add(source)
            continue
        n = norm(str(v))
        toks = n.split()
        if n in PUBLIC_PHRASES:
            continue
        if len(toks) >= 3 and len(n) >= 12 and not any(ch.isdigit() for ch in n):
            rd.names[n].add(source)


def _add_ids(rd: RealData, values, source: str) -> None:
    for v in values:
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        s = str(v).strip().casefold()
        if len(s) >= 7 and any(ch.isdigit() for ch in s) and len(set(s)) > 2:
            rd.ids[s].add(source)


def _add_points(rd: RealData, lats, lons, source: str) -> None:
    for la, lo in zip(lats, lons):
        try:
            la, lo = float(la), float(lo)
        except (TypeError, ValueError):
            continue
        if _is_lat(lo) and _is_lon(la):
            la, lo = lo, la
        if _is_lat(la) and _is_lon(lo):
            rd.points[(f"{la:.5f}", f"{lo:.5f}")].add(source)


def _pair(df, a, b):
    if a not in df.columns or b not in df.columns:
        return [], []
    return df[a].tolist(), df[b].tolist()


def _promote_header(raw):
    """Use the first row (within the top 15) with 3+ distinct text cells as the header."""
    for i in range(min(15, len(raw))):
        row = [str(v).strip() if v == v and v is not None else "" for v in raw.iloc[i].tolist()]
        texts = {r for r in row if r and not re.fullmatch(r"[\d.,\s-]+", r)}
        if len(texts) >= 3 and len(texts) >= 0.5 * sum(1 for r in row if r):
            df = raw.iloc[i + 1 :].copy()
            cols, seen = [], defaultdict(int)
            for r in row:
                c = norm(r).replace(" ", "_") or "col"
                seen[c] += 1
                cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
            df.columns = cols
            return df
    return None


def _load_survey_like(rd: RealData, df, source: str) -> None:
    _add_addresses(rd, _cols(df, "direccion", "direccion_norm"), source)
    _add_person_names(rd, _cols(df, "nombre_evaluador", "Creator", "Editor", "_updated_by"), source)
    _add_ids(rd, _cols(df, "id_edan", "id", "globalid", "GlobalID", "_source_hash", "dup_grupo_id"), source)
    for c in ("ObjectID", "objectid"):
        rd.object_ids.update(str(v).split(".")[0] for v in _values(df, c))
    for la, lo in (("y", "x"), ("y_form", "x_form"), ("geocode_lat", "geocode_lon")):
        _add_points(rd, *_pair(df, la, lo), source)


CONTEXT_GLOB = os.environ.get("CALI_PRIVATE_CONTEXT_GLOB", "*/context/*.xlsx")
SPLIT_NAMES = ("dev", "train_pairs", "unlabeled_addresses")

#: Relative to the data root. ``context/*.xlsx`` is configurable (see the module docstring).
PRIVATE_SOURCE_GLOBS = (
    "*/data/raw/*.parquet",
    "*/data/interim/*labeled*.parquet",
    *(f"*/artifacts/splits/{n}.parquet" for n in SPLIT_NAMES),
    CONTEXT_GLOB,
    "*/data/*private*/**/*examples.json",
    "*/artifacts/predictions.csv",
)

_NAME_COL = re.compile(
    r"^(primer_|segundo_)?(nombre|apellido)s?$|nombre_completo|nombre_evaluador|inspector_nombre"
    r"|^profesional$|^(creator|editor|_?updated_by)$"
)
_ID_COL = re.compile(r"(^id$|^id_|_id$|documento|telefono|identificacion|formulario|globalid|codigo_edif|hash)")
_OBJECTID_COL = re.compile(r"object_?id")


def private_sources_present(models_root: Path = MODELS_ROOT) -> bool:
    """Cheap check (no file is opened) used to skip the guard on clean machines."""
    return any(next(iter(models_root.glob(pattern)), None) is not None for pattern in PRIVATE_SOURCE_GLOBS)


def _load_workbook(rd: RealData, path: Path, source: str) -> bool:
    """Generic loader: columns are recognised by name (address / person / id / coordinates)."""
    import pandas as pd

    loaded = False
    for raw in pd.read_excel(path, sheet_name=None, header=None, dtype=object).values():
        df = _promote_header(raw)
        if df is None or df.empty:
            continue
        before = (len(rd.names), len(rd.ids), len(rd.object_ids))
        cols = list(df.columns)
        addr = [c for c in cols if c.startswith(("direccion", "address"))]
        _add_addresses(rd, [v for c in addr for v in _values(df, c)], source)
        name_cols = [c for c in cols if _NAME_COL.search(c)]
        for c in name_cols:
            _add_person_names(rd, _values(df, c), source)
        if 1 < len(name_cols) <= 6:
            full = [" ".join(str(x) for x in row if x == x and x) for row in df[name_cols].itertuples(index=False)]
            _add_person_names(rd, full, source)
        _add_ids(rd, [v for c in cols if _ID_COL.search(c) for v in _values(df, c)], source)
        for c in cols:
            if _OBJECTID_COL.search(c):
                rd.object_ids.update(str(v).split(".")[0] for v in _values(df, c))
        lat = next((c for c in cols if c.startswith(("lat", "y_"))), None)
        lon = next((c for c in cols if c.startswith(("lon", "lng", "x_"))), None)
        if lat and lon:
            _add_points(rd, df[lat].tolist(), df[lon].tolist(), source)
        for c in cols:
            if c.startswith("coord"):
                pieces = df[c].astype(str).str.split(",", n=1, expand=True)
                if pieces.shape[1] == 2:
                    _add_points(rd, pieces[0].tolist(), pieces[1].tolist(), source)
        if before == (len(rd.names), len(rd.ids), len(rd.object_ids)):
            rd.low_context.add(source)
        loaded = True
    return loaded


def load_real_data(models_root: Path = MODELS_ROOT) -> RealData:
    rd = RealData()
    try:
        import pandas as pd
    except ImportError:
        return rd

    for pattern in ("*/data/raw/*.parquet", "*/data/interim/*labeled*.parquet"):
        for p in sorted(models_root.glob(pattern)):
            _load_survey_like(rd, pd.read_parquet(p), "survey")
            rd.sources_loaded.append(p.name)

    # Evaluation splits. The frozen test split is deliberately never opened.
    for name in SPLIT_NAMES:
        for p in sorted(models_root.glob(f"*/artifacts/splits/{name}.parquet")):
            df = pd.read_parquet(p)
            rd.sources_loaded.append(p.name)
            for ds, sub in df.groupby("dataset"):
                _add_addresses(rd, _values(sub, "raw_address"), str(ds))
                _add_points(rd, *_pair(sub, "gt_lat", "gt_lon"), str(ds))

    for k, p in enumerate(sorted(models_root.glob(CONTEXT_GLOB))):
        label = f"source_{chr(ord('a') + k % 26)}"
        try:
            if _load_workbook(rd, p, label):
                rd.sources_loaded.append(label)
        except Exception:  # noqa: BLE001 - a broken export must not crash the guard
            continue

    rw_dirs = [d for d in models_root.glob("*/data/*private*") if d.is_dir()]
    if rw_dirs:
        found = False

        def walk(o):
            nonlocal found
            if isinstance(o, dict):
                if isinstance(o.get("raw"), str):
                    _add_addresses(rd, [o["raw"]], str(o.get("dataset") or "realworld"))
                    found = True
                for k, v in o.items():
                    if k != "raw":
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        for rw in rw_dirs:
            for p in rw.rglob("*examples.json"):
                walk(json.loads(p.read_text(encoding="utf-8")))
        if found:
            rd.sources_loaded.append("private_examples")

    for pred in sorted(models_root.glob("*/artifacts/predictions.csv")):
        df = pd.read_csv(pred, dtype=str)
        rd.sources_loaded.append(pred.name)
        _add_addresses(rd, _values(df, "direccion"), "survey")
        _add_ids(rd, _values(df, "id"), "survey")

    for cad in sorted(models_root.glob("*/artifacts/catastro.parquet")):
        df = pd.read_parquet(cad, columns=["direccion", "direccion_py"])
        for v in _cols(df, "direccion", "direccion_py"):
            rd.cadastre_core.update(address_keys(str(v))[1])
    return rd


# --------------------------------------------------------------------------- scan


def tracked_files(root: Path) -> list[Path]:
    """Files git would commit (tracked + untracked-not-ignored); a plain walk otherwise."""
    root = root.resolve()
    if (root / ".git").exists():
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True, check=True,
        ).stdout.decode("utf-8", "replace")
        files = [root / f for f in out.split("\0") if f]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.endswith(".egg-info")]
            files.extend(Path(dirpath) / f for f in filenames)
    keep = []
    for f in files:
        rel = f.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if f.suffix.casefold() in BINARY_SUFFIXES or not f.is_file():
            continue
        keep.append(f)
    return sorted(keep)


def _tag(sources: set[str]) -> str:
    return ",".join(sorted(sources))


def scan_line(line: str, rd: RealData) -> list[tuple[str, str]]:
    """Return [(kind, sources)] for one line; never returns the matched value."""
    hits: list[tuple[str, str]] = []
    toks = norm(line).split()
    for i in range(len(toks)):
        for j in range(i + 3, min(len(toks), i + 16) + 1):
            key = " ".join(toks[i:j])
            src = rd.full.get(key)
            if src and not rd.is_public(cores_from_tokens(toks[i:j]), src):
                hits.append(("addr_full", _tag(src)))
    for core in cores_from_tokens(toks):
        src = rd.core.get(core)
        if src and not rd.is_public([core], src):
            hits.append(("addr_core", _tag(src)))
    alpha = [t for t in toks if not t.isdigit()]
    for i in range(len(alpha)):
        for j in range(i + 3, min(len(alpha), i + 6) + 1):
            src = rd.names.get(" ".join(alpha[i:j]))
            if src:
                hits.append(("person_name", _tag(src)))
    for w in WORD_RE.findall(line):
        src = rd.ids.get(w.strip("._-").casefold())
        if src:
            hits.append(("record_id", _tag(src)))
    for num in OBJECTID_RE.findall(line):
        if num in rd.object_ids:
            hits.append(("object_id", "survey"))
    return hits


def scan_points(lines: list[str], rd: RealData) -> list[tuple[int, str, str]]:
    """GPS points: a real (lat, lon) pair within 3 lines of each other."""
    lats: list[tuple[int, str]] = []
    lons: dict[int, set[str]] = defaultdict(set)
    for n, line in enumerate(lines, 1):
        if len(line) > 20000:
            continue
        for s in FLOAT_RE.findall(line):
            v = float(s)
            if _is_lat(v):
                lats.append((n, f"{v:.5f}"))
            elif _is_lon(v):
                lons[n].add(f"{v:.5f}")
    out = []
    for n, la in lats:
        for m in range(n - 3, n + 4):
            for lo in lons.get(m, ()):
                src = rd.points.get((la, lo))
                if src:
                    out.append((n, "gps_point", _tag(src)))
    return out


def scan(roots: list[Path], rd: RealData) -> list[tuple[str, int, str, str]]:
    findings = []
    for root in roots:
        root = root.resolve()
        for f in tracked_files(root):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = f.relative_to(root.parent).as_posix()
            lines = text.splitlines()
            for lineno, line in enumerate(lines, 1):
                if len(line) > 20000:  # embedded blobs (base64 images in notebooks)
                    continue
                for kind, src in scan_line(line, rd):
                    findings.append((rel, lineno, kind, src))
            for lineno, kind, src in scan_points(lines, rd):
                findings.append((rel, lineno, kind, src))
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail if real citizen addresses/ids appear in tracked files.")
    ap.add_argument("--root", action="append", type=Path, help="project root(s) to scan")
    ap.add_argument("--all", action="store_true", help="scan every project under the data root")
    ap.add_argument("--data-root", type=Path, default=MODELS_ROOT,
                    help="directory whose subfolders hold the private data (env CALI_PRIVATE_DATA_ROOT)")
    args = ap.parse_args(argv)
    if args.all:
        roots = sibling_projects(args.data_root)
    else:
        roots = args.root or [REPO_ROOT]
    rd = load_real_data(args.data_root) if private_sources_present(args.data_root) else RealData()
    if rd.empty():
        print("check_no_real_addresses: no private data found locally; nothing checked.")
        return 2
    findings = scan(roots, rd)
    for path, lineno, kind, src in findings:
        print(f"{path}:{lineno} {kind} [{src}]")
    print(
        f"check_no_real_addresses: {len(findings)} hit(s) in {len(roots)} project(s); "
        f"{len(rd.sources_loaded)} private sources loaded."
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
