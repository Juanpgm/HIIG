"""Input loading (CSV / JSONL) and output writing."""
from __future__ import annotations

import csv
import io as _io
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from .metrics import CAVEAT
from .questions import MATCHER, NORMALIZER

DEFAULT_MAX_ROWS = 200
JSONL_SUFFIXES = (".jsonl", ".ndjson", ".json")
REQUIRED_COLUMNS = {NORMALIZER: ("entrada", "normalizada"), MATCHER: ("a", "b", "is_match")}
TRUE_TOKENS = frozenset({"1", "true", "si", "sí"})
FALSE_TOKENS = frozenset({"0", "false", "no"})


class InputError(ValueError):
    """The input file is unusable as a whole (missing, empty, wrong columns)."""


@dataclass
class Item:
    id: str
    a: Any
    b: Any
    decision: bool | None
    score: float | None = None
    score_name: str | None = None
    estado: str | None = None
    flags: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    detail: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class LoadResult:
    items: list[Item]
    total_rows: int
    capped: bool
    max_rows: int
    duplicates_skipped: int = 0


def parse_bool(token: Any) -> bool:
    if isinstance(token, bool):
        return token
    if isinstance(token, int):
        if token in (0, 1):
            return bool(token)
    elif isinstance(token, str):
        t = token.strip().lower()
        if t in TRUE_TOKENS:
            return True
        if t in FALSE_TOKENS:
            return False
    raise ValueError(f"Valor booleano no reconocido: {token!r}")


def _read_records(path: str, kind: str) -> list[dict | None]:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise InputError(f"No se pudo leer '{path}' ({type(exc).__name__}).") from None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError(f"'{path}' no es UTF-8 valido.") from None
    if not text.strip():
        raise InputError(f"'{path}' esta vacio.")
    if path.lower().endswith(JSONL_SUFFIXES):
        return _read_jsonl(text)
    return _read_csv(text, kind)


def _read_jsonl(text: str) -> list[dict | None]:
    records: list[dict | None] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            records.append(None)
            continue
        records.append({str(k).strip().lower(): v for k, v in obj.items()} if isinstance(obj, dict) else None)
    return records


def _read_csv(text: str, kind: str) -> list[dict | None]:
    first = text.splitlines()[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.reader(_io.StringIO(text), delimiter=delimiter)
    header = [h.strip().lower() for h in next(reader)]
    missing = [c for c in REQUIRED_COLUMNS[kind] if c not in header]
    if missing:
        raise InputError(f"Faltan columnas requeridas: {', '.join(missing)}.")
    records: list[dict | None] = []
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        records.append({h: (row[i] if i < len(row) else None) for i, h in enumerate(header)})
    return records


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_score(rec: dict, names: tuple[str, ...], flags: list[str]):
    for name in names:
        value = rec.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
        except (TypeError, ValueError):
            flags.append("bad_score")
            return None, name
        if not math.isfinite(number):
            flags.append("bad_score")
            return None, name
        return number, name
    return None, None


def _to_item(kind: str, rec: dict | None, index: int) -> Item:
    if rec is None:
        return Item(str(index), None, None, None, skip_reason="invalid_input", detail="fila no es un objeto valido")
    flags: list[str] = []
    ident = rec.get("id")
    item_id = str(ident).strip() if ident is not None and str(ident).strip() else str(index)
    if kind == NORMALIZER:
        a, b = rec.get("entrada"), rec.get("normalizada")
        decision: bool | None = True  # the normalizer implicitly claims equivalence
        score, score_name = _parse_score(rec, ("confiabilidad", "confianza"), flags)
        estado = _text(rec.get("estado"))
        bad = None
    else:
        a, b = rec.get("a"), rec.get("b")
        score, score_name = _parse_score(rec, ("score",), flags)
        estado = None
        try:
            decision, bad = parse_bool(rec.get("is_match")), None
        except ValueError as exc:
            decision, bad = None, str(exc)
    item = Item(item_id, a, b, decision, score, score_name, estado, flags, raw=dict(rec))
    if _text(a) is None or _text(b) is None:
        item.skip_reason, item.detail = "invalid_input", "direccion vacia o no textual"
    elif bad:
        item.skip_reason, item.detail = "invalid_input", bad
    return item


def load_items(path: str, kind: str, max_rows: int = DEFAULT_MAX_ROWS) -> LoadResult:
    if max_rows < 1:
        raise InputError("--max-rows debe ser un entero positivo.")
    records = _read_records(path, kind)
    total = len(records)
    items: list[Item] = []
    seen_ids: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    duplicates = 0
    for index, rec in enumerate(records[:max_rows], start=1):
        item = _to_item(kind, rec, index)
        count = seen_ids.get(item.id, 0) + 1
        seen_ids[item.id] = count
        if count > 1:
            item.id = f"{item.id}#{count}"
            item.flags.append("duplicate_id")
        if item.skip_reason is None:
            pair = (item.a.strip(), item.b.strip())
            if pair in seen_pairs:
                item.skip_reason, item.detail = "duplicate_pair", "par ya enviado en otra fila"
                duplicates += 1
            seen_pairs.add(pair)
        items.append(item)
    return LoadResult(items, total, total > max_rows, max_rows, duplicates)


# -------------------------------------------------------------------------- outputs


def _pct(x: float | None) -> str:
    return "n/d" if x is None else f"{x * 100:.1f} %"


def _num(x: float | None, digits: int = 3) -> str:
    return "n/d" if x is None else f"{x:.{digits}f}"


def render_summary_md(summary: dict) -> str:
    counts = summary.get("counts", {})
    agr = summary.get("agreement") or {}
    cm = summary.get("confusion") or {}
    prf = summary.get("precision_recall") or {}
    ci95 = agr.get("ci95")
    ci = f"{_pct(ci95[0])} a {_pct(ci95[1])}" if ci95 else "n/d"
    lines = [
        f"# Resumen de validacion JEV ({summary.get('kind', '?')})",
        "",
        f"> {summary.get('caveat', CAVEAT)}",
        "",
    ]
    if not summary.get("has_valid_evidence"):
        lines += [
            "**Sin evidencia valida**: no hay elementos con veredicto JEV definido (si/no). "
            "No se debe sacar ninguna conclusion de esta corrida.",
            "",
        ]
    inp = summary.get("input") or {}
    if inp.get("capped"):
        lines += [
            f"Aviso: el archivo tenia {inp.get('total_rows')} filas; se procesaron solo las primeras "
            f"{inp.get('max_rows')} (tope --max-rows).",
            "",
        ]
    lines += [
        "## Conteos",
        "",
        f"- Total de filas procesadas: {counts.get('total', 0)}",
        f"- Evaluadas correctamente: {counts.get('ok', 0)}",
        f"- Fallidas (error de API o respuesta invalida, excluidas): {counts.get('error', 0)}",
        f"- Omitidas (entrada invalida o duplicada): {counts.get('skipped', 0)}",
        f"- Indeterminadas (JEV no es concluyente, excluidas de la concordancia): {counts.get('indeterminate', 0)}",
        f"- Con veredicto si/no: {counts.get('labeled', 0)}",
        f"- Tasa de fallos de la API: {_pct(summary.get('api_failure_rate'))}",
        "",
        "## Concordancia con la decision del modelo",
        "",
        f"- Concordancia: {_pct(agr.get('rate'))} ({agr.get('agree', 0)} de {agr.get('n', 0)})",
        f"- IC 95 % (Wilson): {ci}",
        f"- Kappa de Cohen: {_num(summary.get('kappa'))}",
        f"- Matriz de confusion (modelo vs JEV): VP={cm.get('tp', 0)}, FP={cm.get('fp', 0)}, "
        f"FN={cm.get('fn', 0)}, VN={cm.get('tn', 0)}",
        f"- Precision / exhaustividad / F1 (JEV como referencia): {_num(prf.get('precision'))} / "
        f"{_num(prf.get('recall'))} / {_num(prf.get('f1'))}",
        f"- AUC del puntaje del modelo vs veredicto JEV: {_num(summary.get('auc'))}",
        f"- Brier vs veredicto JEV: {_num(summary.get('brier'))}",
        "",
    ]
    bins = (summary.get("reliability") or {}).get("bins") or []
    if bins:
        lines += [
            "## Tabla de fiabilidad (puntaje del modelo vs P(si) de JEV)",
            "",
            "| Bin | n | Puntaje min | Puntaje max | P(si) JEV media |",
            "|---|---|---|---|---|",
        ]
        for i, b in enumerate(bins, 1):
            lines.append(f"| {i} | {b['n']} | {_num(b['score_min'])} | {_num(b['score_max'])} | {_num(b['mean_p_yes'])} |")
        lines.append("")
    dis = summary.get("disagreements") or []
    if dis:
        lines += ["## Discrepancias para revision manual (mayor certeza JEV primero)", ""]
        for d in dis[:50]:
            lines.append(
                f"- `{d['id']}` (certeza JEV {_num(d.get('certainty'))}, modelo={'si' if d.get('decision') else 'no'}): "
                f"{d.get('a')!r} vs {d.get('b')!r}"
            )
        if len(dis) > 50:
            lines.append(f"- ... y {len(dis) - 50} mas en results.jsonl / summary.json")
        lines.append("")
    usage = summary.get("usage") or {}
    lines += [
        "## Uso de tokens",
        "",
        f"- Entrada: {usage.get('input_tokens', 0)}; salida: {usage.get('output_tokens', 0)}",
        "",
        "## Modelo reportado por la API",
        "",
        f"- {', '.join(summary.get('models_seen') or []) or 'no informado'}",
        "",
    ]
    return "\n".join(lines)


def write_outputs(out_dir: str, results: list[dict], summary: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_summary_md(summary))
