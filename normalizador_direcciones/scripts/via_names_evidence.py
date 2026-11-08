"""Cadastral evidence for the street-name table (``src/cali_address/data/via_names_review.csv``).

The cadastre only carries numbered streets, so a street NAME (``AV ROOSEVELT # 26-19``) is verified indirectly: for every
distinct door ``cross - plate`` found in the real raw addresses that use the name, the cadastre is asked which streets own
that exact door. A name is ``verified`` only when ONE street explains the doors without ambiguity:

* at least ``MIN_DOORS`` distinct doors written with the name,
* at least ``MIN_HIT_SHARE`` of them exist in the cadastre,
* the top street owns at least ``TOP_SHARE`` of the doors that exist, and the runner-up at most ``RUNNER_RATIO`` of the top.

Everything else is ``ambiguous`` (evidence points to several streets or is too thin) or ``rejected`` (not a street name).
Read-only: it reads the cadastre parquet and a CSV of raw addresses and writes the review CSV; it touches no database.

    python scripts/via_names_evidence.py --raws raws.csv --catastro artifacts/catastro.parquet --out src/cali_address/data/via_names_review.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cali_address.parser import parse_address, strip_accents  # noqa: E402

MIN_DOORS = 10
MIN_HIT_SHARE = 0.5
TOP_SHARE = 0.7
RUNNER_RATIO = 0.5
_GENERIC = r"(?:(?:AV|AVENIDA|AVDA|AVD|AUTOPISTA|AUT|VIA|CALLE|CL|CLL|CARRERA|CRA|KR)\.?\s+)?(?:(?:DE\s+)?(?:LAS|LOS|LA|EL)\s+)?"

#: name -> (folded variants, forced status or None, reason for a forced status).
SEEDS: dict[str, tuple[tuple[str, ...], str | None, str]] = {
    "ROOSEVELT": (("ROOSEVELT", "ROOSVELT", "ROUSVELT", "ROSVELT", "ROSEVELT", "ROUSEVELT", "ROOSEVEL", "ROOVELTH", "ROOSVELTH",
                   "ROSELVET", "ROOSERET", "ROOSEVELTH"), None, ""),
    "AUTOPISTA SUR": (("AUTOPISTA SUR",), None, ""),
    "AUTOPISTA SUR ORIENTAL": (("AUTOPISTA SUR ORIENTAL",), None, ""),
    "AMERICAS": (("AMERICAS", "AMERICA"), None, ""),
    "ESTACION": (("ESTACION",), None, ""),
    "GUADALUPE": (("GUADALUPE",), None, ""),
    "COLOMBIA": (("COLOMBIA",), None, ""),
    "CIRCUNVALAR": (("CIRCUNVALAR",), None, ""),
    "SIMON BOLIVAR": (("SIMON BOLIVAR",), None, ""),
    "PASOANCHO": (("PASOANCHO", "PASO ANCHO"), None, ""),
    "LOS CERROS": (("CERROS",), None, ""),
    "NORTE": (("NORTE",), "rejected", "cardinal word, not a street name"),
    "OESTE": (("OESTE",), "rejected", "cardinal word, not a street name"),
}


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(str(text)).upper()).strip()


def door_key(p) -> tuple:
    return (p.cross_number or "", p.cross_letters or "", p.cross_suffix or "", p.cross_suffix_letters or "",
            bool(p.cross_bis), p.cross_quadrant or "", p.plate or "")


def via_key(p) -> tuple:
    return (p.via_type, p.via_number, p.via_letters or "", p.via_suffix or "", p.via_suffix_letters or "",
            bool(p.via_bis), p.via_quadrant or "")


def render(v: tuple) -> str:
    t, n, letters, suffix, sletters, bis, quad = v
    parts = [t, n, letters, suffix, sletters, "BIS" if bis else "", quad]
    return " ".join(x for x in parts if x)


def cadastre_index(parquet: str) -> dict:
    import pandas as pd

    idx: dict = collections.defaultdict(collections.Counter)
    for addr in pd.read_parquet(parquet, columns=["direccion"])["direccion"].dropna().drop_duplicates():
        p = parse_address(addr)
        if p.parse_ok and p.plate is not None:
            idx[door_key(p)][via_key(p)] += 1
    return idx


def doors_for(variants, raws) -> set:
    doors = set()
    for variant in sorted(variants, key=len, reverse=True):
        pat = re.compile(rf"(?<![A-Z0-9]){_GENERIC}{re.escape(variant)}(?![A-Z])")
        for raw in raws:
            m = pat.search(raw)
            if not m:
                continue
            rest = raw[m.end():].lstrip(" .")
            p = parse_address("CL 1 " + (rest if rest.startswith("#") else "# " + rest))
            if p.parse_ok and p.plate is not None and p.cross_number:
                doors.add(door_key(p))
    return doors


def judge(doors: set, idx: dict) -> dict:
    hits: collections.Counter = collections.Counter()
    with_hit = 0
    for d in doors:
        owners = idx.get(d)
        if owners:
            with_hit += 1
            hits.update(owners.keys())
    top = hits.most_common(2)
    out = {"doors": len(doors), "doors_with_hit": with_hit, "top_via": "", "top_hits": 0, "runner_up_via": "", "runner_up_hits": 0}
    if top:
        out.update(top_via=render(top[0][0]), top_hits=top[0][1])
    if len(top) > 1:
        out.update(runner_up_via=render(top[1][0]), runner_up_hits=top[1][1])
    if len(doors) < MIN_DOORS:
        return {**out, "status": "ambiguous", "reason": f"insufficient evidence: {len(doors)} distinct doors (< {MIN_DOORS})"}
    if with_hit < MIN_HIT_SHARE * len(doors):
        return {**out, "status": "ambiguous", "reason": f"only {with_hit}/{len(doors)} doors exist in the cadastre"}
    if out["top_hits"] < TOP_SHARE * with_hit:
        return {**out, "status": "ambiguous",
                "reason": f"no single street explains the doors: top {out['top_via']} owns {out['top_hits']}/{with_hit}"}
    if out["runner_up_hits"] > RUNNER_RATIO * out["top_hits"]:
        return {**out, "status": "ambiguous",
                "reason": f"runner-up {out['runner_up_via']} too close ({out['runner_up_hits']} vs {out['top_hits']})"}
    return {**out, "status": "verified", "reason": f"{out['top_hits']}/{with_hit} doors on {out['top_via']}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raws", required=True, help="CSV with a `raw` column (distinct raw addresses)")
    ap.add_argument("--catastro", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raws = sorted({fold(r["raw"]) for r in csv.DictReader(open(a.raws, encoding="utf-8")) if r.get("raw")})
    idx = cadastre_index(a.catastro)
    rows = []
    for name, (variants, forced, why) in SEEDS.items():
        r = judge(doors_for(variants, raws), idx)
        if forced:
            r["status"], r["reason"] = forced, why
        rows.append({"name": name, "variants": "|".join(variants),
                     "canonical_via": r["top_via"] if r["status"] == "verified" else "", **r})
    fields = ["name", "variants", "canonical_via", "doors", "doors_with_hit", "top_via", "top_hits",
              "runner_up_via", "runner_up_hits", "status", "reason"]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["status"].ljust(9), r["name"].ljust(24), r["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
