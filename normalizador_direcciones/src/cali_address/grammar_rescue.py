"""Grammar rescue for addresses the rule parser rejects (``NO_PARSEABLE``).

Only used when :func:`cali_address.parser.parse_address` ALREADY failed on the text, so an address that parses today is
never touched by construction. It proposes rewritten texts, in order, and the caller keeps the first one that parses:

1. street-name substitution (``AV ROOSEVELT # 39-25`` -> ``CL 5 # 39-25``) with the VERIFIED table of
   :mod:`cali_address.via_names` (only names that map to one street without ambiguity);
2. variants/abbreviations (``CAR``, ``AVD``, ``RA`` -> KR, ordinal words, ``AV, 3H``, ``Av.``);
3. free text before the first street-type word (building / hotel / tower names): dropped when the remainder parses;
   a tower / block (``TORRE D``) found in that prefix is never dropped, it moves after the address as a complement.

Nothing here guesses digits: a prefix containing digits (other than a tower / block code) is never stripped.
"""

from __future__ import annotations

import re

from .parser import VIA_TYPE_CANON, parse_address, strip_accents
from .via_names import VIA_NAMES

__all__ = ["rescue_parse", "rescue_candidates"]

# Street-type words that may open an address (single letters are excluded: too ambiguous in free text).
_VIA_WORDS = sorted(
    {w for w in VIA_TYPE_CANON if len(w) >= 2 and w.isalpha() and " " not in w} | {"CAR", "AVD", "AVDA", "AVENIDA", "AUTOPISTA"},
    key=len, reverse=True,
)
_VIA_ALT = "|".join(_VIA_WORDS)
_VIA_START_RE = re.compile(rf"(?<![A-Z0-9])(?:{_VIA_ALT})(?![A-Z0-9])")

_ORDINALS = {
    "PRIMERA": "1", "PRIMERO": "1", "SEGUNDA": "2", "SEGUNDO": "2", "TERCERA": "3", "TERCERO": "3",
    "CUARTA": "4", "CUARTO": "4", "QUINTA": "5", "QUINTO": "5", "SEXTA": "6", "SEXTO": "6",
    "SEPTIMA": "7", "SEPTIMO": "7", "OCTAVA": "8", "OCTAVO": "8", "NOVENA": "9", "NOVENO": "9",
    "DECIMA": "10", "DECIMO": "10",
}
_ORD_ALT = "|".join(sorted(_ORDINALS, key=len, reverse=True))
_ORD_RE = re.compile(rf"(?<![A-Z0-9])({_VIA_ALT})\s+({_ORD_ALT})(?![A-Z0-9])")
_DOT_RE = re.compile(r"(?<=[A-Z])\.(?=\s|,|$|\d)")
_VIA_COMMA_RE = re.compile(rf"(?<![A-Z0-9])({_VIA_ALT})\s*,\s*(?=[\dA-Z#])")
_CAR_RE = re.compile(rf"(?<![A-Z0-9])CAR(?![A-Z0-9])(?=\s*(?:[#\d]|(?:{_ORD_ALT})(?![A-Z0-9])))")
_AVD_RE = re.compile(r"(?<![A-Z0-9])AVD(?![A-Z0-9])")
_RA_RE = re.compile(r"(?<![0-9])(?<![0-9] )(?<![A-Z0-9])RA(?![A-Z0-9])(?=\s*\d)")
_OE_RE = re.compile(r"(?<![A-Z0-9])OE(?![A-Z0-9])")
_TOWER_RE = re.compile(r"(?<![A-Z0-9])(TORRE|BLOQUE|BLQ|TO|BL)\s+([A-Z]|\d{1,3}|[A-Z]\d{1,2})(?![A-Z0-9])")
_NOT_TOWER_VALUE = frozenset({"DE", "DEL", "LA", "EL", "LOS", "LAS", "Y"})
_KIND_OF = {"TORRE": "TORRE", "TO": "TORRE", "BLOQUE": "BLOQUE", "BLQ": "BLOQUE", "BL": "BLOQUE"}
_MAX_LEN = 400
# A name written right after one of these words is a place (BARRIO ROOSEVELT), not the street.
_PLACE_WORDS = frozenset({"BARRIO", "BRR", "URB", "URBANIZACION", "CONJUNTO", "CONJ", "EDIFICIO", "ED", "EDF", "CONDOMINIO",
                          "CC", "CENTRO", "CIUDADELA", "CDLA", "SECTOR", "VEREDA", "HOTEL", "COLEGIO", "HOSPITAL", "CLINICA"})
# Generic street-type word (and article) written right before a street name: it is part of the name span.
_GENERIC_BEFORE = re.compile(
    r"(?:(?:AV|AVENIDA|AVDA|AVD|AUTOPISTA|AUT|VIA|CALLE|CLL|CL|CARRERA|CRA|KR)\.?\s+)?(?:(?:DE\s+)?(?:LAS|LOS|LA|EL)\s+)?$"
)
# Name must end the street: the next token may only be a plate / number sign / separator.
_AFTER_NAME_OK = re.compile(r"^(?:$|[#,\-]|\d|NO(?![A-Z])|N(?![A-Z])|KR\b|CL\b)")


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(str(text)).upper()).strip()


def _rewrite(u: str) -> str:
    """Variants and abbreviations only (no text is removed)."""
    u = _DOT_RE.sub(" ", u)
    u = _VIA_COMMA_RE.sub(r"\1 ", u)
    u = _AVD_RE.sub("AV", u)
    u = _CAR_RE.sub("KR", u)
    if len(_VIA_START_RE.findall(u)) == 0:  # `RA 42 CALLE 5B ...` has two streets: never guess which one RA is
        u = _RA_RE.sub("KR", u)
    u = _OE_RE.sub("OESTE", u)
    u = _ORD_RE.sub(lambda m: f"{m.group(1)} {_ORDINALS[m.group(2)]}", u)
    return re.sub(r"\s+", " ", u).strip()


def _substitute_names(u: str) -> str | None:
    """Replace a verified street name (with its generic street-type word, if any) by its canonical street."""
    best = None
    for variant, canonical in VIA_NAMES:
        pos = u.find(variant)
        while pos != -1:
            end = pos + len(variant)
            before_ok = pos == 0 or not u[pos - 1].isalnum()
            rest = u[end:].lstrip(" .")
            words = u[:pos].split()
            if (before_ok and _AFTER_NAME_OK.match(rest) and re.search(r"\d", rest)
                    and not (words and words[-1].strip(".") in _PLACE_WORDS)):
                if best is None or len(variant) > best[0]:
                    best = (len(variant), pos, end, canonical)
                break
            pos = u.find(variant, pos + 1)
    if best is None:
        return None
    _, pos, end, canonical = best
    pos = _GENERIC_BEFORE.search(u, 0, pos).start()
    return f"{u[:pos]}{canonical} {u[end:].lstrip(' .')}".strip()


def _strip_prefixes(u: str) -> list[str]:
    """Candidates that drop the free text before a street-type word (tower / block codes move to the end)."""
    out: list[str] = []
    for m in _VIA_START_RE.finditer(u):
        if m.start() == 0:
            continue
        prefix, rest = u[: m.start()], u[m.start():]
        tail = ""
        t = _TOWER_RE.search(prefix)
        if t and t.group(2) not in _NOT_TOWER_VALUE:
            tail = f" {_KIND_OF[t.group(1)]} {t.group(2)}"
            prefix = prefix[: t.start()] + " " + prefix[t.end():]
        if re.search(r"\d", prefix):
            continue
        out.append(f"{rest}{tail}".strip())
    return out


def rescue_candidates(text: str) -> list[str]:
    """Rewritten texts to try, in order, never equal to the folded input and without duplicates."""
    if not isinstance(text, str) or not text.strip() or len(text) > _MAX_LEN:
        return []
    u = _fold(text)
    bases = [u]
    named = _substitute_names(u)
    if named is not None:
        bases.insert(0, named)
    seen = {u}
    out: list[str] = []

    def add(c: str) -> None:
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    for b in bases:
        r = _rewrite(b)
        add(r)
        for variant in dict.fromkeys([b, r]):
            for c in _strip_prefixes(variant):
                add(_rewrite(c))
    return out


def rescue_parse(*texts: str):
    """First ``(text, parsed)`` among the candidates of ``texts`` that parses, else ``None``."""
    for text in texts:
        for cand in rescue_candidates(text):
            parsed = parse_address(cand)
            if parsed.parse_ok:
                return cand, parsed
    return None
