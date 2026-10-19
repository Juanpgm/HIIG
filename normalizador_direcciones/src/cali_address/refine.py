"""Opt-in, format-level refinement of a raw address string (never a rewrite).

``refine_address`` removes parasitic *format* noise (odd whitespace, invisible characters,
mojibake of the ordinal sign, stray edge punctuation, repeated commas, a comma that splits the
plate, number-sign spellings, a trailing city/department suffix) and nothing else. It is OFF
everywhere by default: callers pass ``refine_input=True`` to opt in.

Hard invariants (checked after every rule AND once for the whole pipeline; a violation drops the
rule, or the whole refinement, and the unrefined text is used):

* the ordered sequence of digit characters is identical;
* every token that contains a digit (``66C``, ``26G6``, ``1A9``...) is byte-identical and in order;
* the ordered quadrant tokens (N, NTE, NORTE, S, SUR, O, OESTE, E, ESTE) are identical (the ``N``
  of an ``N°`` number sign is not a quadrant);
* the ordered via-type words (CALLE, CARRERA, CL, KR, CRA, AV, DG, TV...) are identical.

Deliberately NOT done (two readings or too lossy): building/barrio text after the plate,
splitting multi-address cells (they are only flagged), glued tokens, via-type expansion, guessing
numbers, folding dashes or the middle dot, coordinate/phone fragments, NFKC (it turns ``²`` into a
digit and ``º`` into a letter).
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

from .parser import VIA_TYPE_CANON

__all__ = ["RefineResult", "refine_address"]


class RefineResult(NamedTuple):
    text: str
    changes: tuple[str, ...]
    is_placeholder: bool


# Mirrors ``service.PLACEHOLDERS`` (kept here so this module stays dependency-light; a test
# asserts both sets are equal).
_PLACEHOLDERS = {
    "", "-", "--", ".", "0", "00", "N/A", "NA", "N.A", "N.A.", "NAN", "NONE", "NULL",
    "NO APLICA", "NO REGISTRA", "NO REPORTA", "NO INFORMA", "NINGUNA", "NINGUNO",
    "SIN ESPECIFICAR", "SIN DATO", "SIN DATOS", "SIN DIRECCION", "SIN INFORMACION",
    "S/D", "S.D", "SD", "DESCONOCIDA", "DESCONOCIDO", "NO TIENE", "NO HAY", "X", "XX", "XXX",
}

_QUADRANTS = frozenset({"N", "NTE", "NORTE", "S", "SUR", "O", "OESTE", "E", "ESTE"})
_VIA_WORDS = frozenset(a for a in VIA_TYPE_CANON if len(a) >= 2 and a.isalpha())
_UNIT_WORDS = frozenset({
    "PISO", "AP", "APTO", "APT", "APARTAMENTO", "CASA", "CS", "TORRE", "TO", "BLOQUE", "BL", "LOCAL",
    "LC", "OF", "OFICINA", "INT", "INTERIOR", "MZ", "MANZANA", "LT", "LOTE", "ETAPA", "ET",
    "CONJUNTO", "CONJ", "UNIDAD", "UN", "CA", "AGRUPACION", "AG",
})
#: Words after which a trailing CALI/VALLE... is a proper name, not a city suffix.
_NAME_INTRODUCERS = frozenset({
    "CON", "Y", "DE", "DEL", "ENTRE", "EL", "LA", "LOS", "LAS", "BARRIO", "BRR", "URB", "URBANIZACION",
    "CIUDAD", "CDLA", "CIUDADELA", "SECTOR", "VEREDA", "CORREGIMIENTO", "COMUNA", "MUNICIPIO",
    "EDIFICIO", "ED", "CENTRO", "CC", "CLUB", "COLEGIO", "HOTEL", "PLAZA", "PARQUE",
})
_SIGN_PREV_WORDS = frozenset({"NTE", "NORTE", "SUR", "OESTE", "ESTE", "BIS", "BS"})

_WORD_RE = re.compile(r"[^\W\d_]+")
_ALNUM_RE = re.compile(r"[^\W_]+")
_DIGIT_TOKEN_RE = re.compile(r"[^\W_]*\d[^\W_]*")
_SIGN_MASK_RE = re.compile("\\bN\\s*[°º░]")


def _words(text: str, vocab) -> list[str]:
    return [t for t in _WORD_RE.findall(text.upper()) if t in vocab]


def _invariants_ok(before: str, after: str) -> bool:
    if [c for c in before if c.isdigit()] != [c for c in after if c.isdigit()]:
        return False
    if _DIGIT_TOKEN_RE.findall(before) != _DIGIT_TOKEN_RE.findall(after):
        return False
    q_before = _words(_SIGN_MASK_RE.sub(" ", before), _QUADRANTS)
    q_after = _words(_SIGN_MASK_RE.sub(" ", after), _QUADRANTS)
    if q_before != q_after:
        return False
    return _words(before, _VIA_WORDS) == _words(after, _VIA_WORDS)


def _clean_key(text: str) -> str:
    if text.lower() == "nan":
        return ""
    key = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", key).strip().upper()


def _is_placeholder(text: str) -> bool:
    key = _clean_key(text)
    if key in _PLACEHOLDERS:
        return True
    return not re.search(r"[A-Z]", key) and len(re.sub(r"\D", "", key)) < 3


def _last_word(text: str) -> str:
    found = _ALNUM_RE.findall(text)
    return found[-1].upper() if found else ""


# ---------------------------------------------------------------------------- rules
# Every rule maps text -> (new_text, [rule ids applied]).


def _rule_space(text: str):
    # Line breaks are boundaries for the parser (name, newline, address): keep one per break.
    lines = (" ".join(chunk.split()) for chunk in re.split(r"\r\n|\r|\n", text))
    new = "\n".join(line for line in lines if line)
    return new, (["space_normalize"] if new != text else [])


def _rule_invisible(text: str):
    def fix(m: re.Match) -> str:
        chunk = m.group(0)
        if any(c.isdigit() for c in chunk):
            return chunk
        return "".join(c for c in chunk if unicodedata.category(c) not in ("Cf", "Cc"))

    new = re.sub(r"\S+", fix, text)
    return new, (["strip_invisible"] if new != text else [])


def _rule_mojibake(text: str):
    def fix(m: re.Match) -> str:
        chunk = m.group(0)
        if ("Â" not in chunk and "Ã" not in chunk) or any(c.isdigit() for c in chunk):
            return chunk
        try:
            fixed = chunk.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return chunk
        return chunk if any(c.isdigit() for c in fixed) else fixed

    new = re.sub(r"\S+", fix, text)
    return new, (["mojibake_ordinal"] if new != text else [])


_LEAD_JUNK = re.compile("^[;:,.|<>*•·\"'`‘’“”_~\\s]+")
_LEAD_DASH = re.compile(r"^-+(?=\s|[^\W\d_])")
_TRAIL_JUNK = "•·\"'`‘’“”,;:|<>*_~ "
_DOT_KEEP = frozenset({"N", "NO", "NRO", "NUM"})
_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _rule_edges(text: str):
    new = _LEAD_JUNK.sub("", text)
    new = _LEAD_DASH.sub("", new).lstrip()
    while True:
        stripped = new.rstrip(_TRAIL_JUNK)
        if stripped.endswith(".") and _last_word(stripped[:-1]) not in _DOT_KEEP:
            stripped = stripped[:-1]
        if stripped == new:
            break
        new = stripped
    if new and new[0] in _PAIRS and new[-1] == _PAIRS[new[0]] and new.count(new[0]) == 1 and new.count(new[-1]) == 1:
        new = new[1:-1].strip()
    return new, (["strip_edges"] if new != text else [])


def _rule_punct(text: str):
    new = re.sub(r",(?:\s*,)+", ",", text)
    new = re.sub(r";(?:\s*;)+", ";", new)
    new = re.sub(r"(?<=\D)\.{2,}(?=\D|$)", ".", new)
    return new, (["collapse_punct"] if new != text else [])


_SIGN_RE = re.compile(
    r"(?<![^\W_])(?:(?:NO|NRO|NUM|NUMERO)(?:\s*[.,:]\s*|\s+)|N\s*[°º░]\s*[.,:]?\s*)(?=\d)",
    re.IGNORECASE,
)


def _rule_numsign(text: str):
    if "#" in text:
        return text, []
    m = _SIGN_RE.search(text)
    if m is None:
        return text, []
    prefix = text[: m.start()]
    if "-" in prefix or not any(c.isdigit() for c in prefix):
        return text, []
    prev = _last_word(prefix)
    ok = (
        any(c.isdigit() for c in prev)
        or prev in _SIGN_PREV_WORDS
        or (prev.isalpha() and len(prev) <= 2)
    )
    if not ok:
        return text, []
    return prefix.rstrip() + " # " + text[m.end():], ["numsign"]


def _digit_tokens(text: str) -> int:
    return len(_DIGIT_TOKEN_RE.findall(text))


def _comma_is_separator(left_raw: str, right_raw: str) -> bool:
    left, right = left_raw.rstrip(), right_raw.lstrip()
    if not left or not right:
        return False
    tight = left_raw == left and right_raw == right
    if right[0] == "#":
        if "#" in left:
            return False
    elif right[0].isdigit():
        if tight:
            return False
        rest = _ALNUM_RE.findall(right)
        if len(rest[0]) > 6 or (len(rest) > 1 and rest[1].upper() in _UNIT_WORDS):
            return False
    else:
        return False
    if "-" in left or not _words(left, _VIA_WORDS):
        return False
    last = _last_word(left)
    if not (any(c.isdigit() for c in last) or last in _VIA_WORDS or left.endswith("#")):
        return False
    if "#" in left:
        return _digit_tokens(left.rsplit("#", 1)[1]) <= 1
    return _digit_tokens(left) <= 2


def _rule_comma(text: str):
    out, pos, changed = text, 0, False
    while True:
        i = out.find(",", pos)
        if i < 0:
            break
        if _comma_is_separator(out[:i], out[i + 1:]):
            left = out[:i].rstrip()
            out = left + " " + out[i + 1:].lstrip()
            pos = len(left) + 1
            changed = True
        else:
            pos = i + 1
    return out, (["comma_to_space"] if changed else [])


_CITY_RE = re.compile(
    r"(?:^|[\s,;])(SANTIAGO DE CALI|CALI VALLE DEL CAUCA|CALI VALLE|VALLE DEL CAUCA|V/CAUCA|CALI|COLOMBIA"
    r"|(?<=,)\s*VALLE)\s*\.?\s*$",
    re.IGNORECASE,
)


def _rule_city(text: str):
    new = text
    for _ in range(6):
        m = _CITY_RE.search(new)
        if m is None:
            break
        rest = new[: m.start()].rstrip(" ,;.")
        if not rest or rest.endswith("#") or _last_word(rest) in (_NAME_INTRODUCERS | _VIA_WORDS):
            break
        new = rest
    return new, (["strip_city_suffix"] if new != text else [])


def _rule_flag_multi(text: str):
    if ";" in text and sum(1 for seg in text.split(";") if any(c.isdigit() for c in seg)) >= 2:
        return text, ["flag_multi_address"]
    return text, []


_RULES = [
    ("space_normalize", _rule_space),
    ("strip_invisible", _rule_invisible),
    ("mojibake_ordinal", _rule_mojibake),
    ("strip_edges", _rule_edges),
    ("collapse_punct", _rule_punct),
    ("numsign", _rule_numsign),
    ("comma_to_space", _rule_comma),
    ("strip_city_suffix", _rule_city),
    ("flag_multi_address", _rule_flag_multi),
]
_MAX_PASSES = 4


def _pipeline(text: str) -> tuple[str, list[str]]:
    ids: list[str] = []
    for _ in range(_MAX_PASSES):
        before_pass = text
        for _name, rule in _RULES:
            try:
                new, applied = rule(text)
            except Exception:  # a rule must never break the caller
                continue
            if not applied or not _invariants_ok(text, new):
                continue
            text = new
            ids.extend(i for i in applied if i not in ids)
        if text == before_pass:
            return text, ids
    return text, ids if text == before_pass else None  # type: ignore[return-value]


def refine_address(raw) -> RefineResult:
    """Return the refined text, the rule ids applied and whether the cell is a placeholder."""
    if raw is None:
        return RefineResult("", (), True)
    try:
        original = raw if isinstance(raw, str) else str(raw)
    except Exception:
        return RefineResult("", (), True)
    collapsed = " ".join(original.split())
    if _is_placeholder(original):
        return RefineResult(collapsed, (), True)
    text, ids = _pipeline(original)
    if ids is None or not _invariants_ok(original, text):
        return RefineResult(original, (), False)
    return RefineResult(text, tuple(ids), _is_placeholder(text))
