"""Rule-based canonicalizer/parser for Colombian (Cali) street addresses.

The target canonical form is the one used by the Cali cadastral layer
(`urbano_terreno.direccion`) and, up to letter spacing, by the IDESC geocoder
(`dir_ajusta`)::

    KR 26 H 3 # 73 - 10
    CL 12 # 48 BIS - 31
    AV 15 OESTE # 9 OESTE - 137
    CL 25 NORTE # AV 6 - 30

Grammar (informal)::

    address    := via "#" cross "-" plate [complement]
    via        := via_type number [letters] [number] [letters] [BIS] [quadrant]
    cross      := [via_type] number [letters] [number] [letters] [BIS] [quadrant]
    plate      := digits                      # zero padded to 2 in canonical form
    complement := free text (AP 501, LC 1, TO 3, ED PAMPLONA, GASS 144, ...)

Design notes
------------
* The `#` and the plate `-` are located **before** any token gluing/splitting so
  the complement text can be preserved verbatim (``14C`` stays ``14C``).
* Legacy one-letter via types used by the cadastre (``K``=KR, ``C``=CL,
  ``A``=AV, ``D``=DG, ``T``=TV) are normalized.
* The parser never raises; unparseable input yields ``parse_ok=False``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterable

from . import complement_lexicon as _lexicon

__all__ = [
    "ParsedAddress",
    "parse_address",
    "parse_many",
    "TRUNCATION_NOTES",
    "is_truncated",
    "canonical",
    "canonicalize",
    "filter_complement",
    "merge_complement",
    "extract_complement_units",
    "is_tower_name_word",
    "ComplementMerge",
    "match_key",
    "strip_accents",
    "VIA_TYPE_CANON",
    "COMPLEMENT_CANON",
    "QUADRANT_CANON",
]

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

#: Every spelling we have observed -> canonical cadastral abbreviation.
VIA_TYPE_CANON: dict[str, str] = {}


def _register(canon: str, *aliases: str) -> None:
    VIA_TYPE_CANON[canon] = canon
    for alias in aliases:
        VIA_TYPE_CANON[alias] = canon


_register("CL", "CALLE", "CALLES", "CLL", "CLLE", "CLE", "CALL", "CAL", "C", "CLA")
_register("KR", "CARRERA", "CARRERAS", "CRA", "CR", "CRR", "CRRA", "KRA", "KRR",
          "KAR", "CARR", "CARERA", "CARREA", "K")
_register("AV", "AVENIDA", "AVDA", "AVE", "AVN", "AVEN", "A")
_register("AC", "AVENIDA CALLE", "AVCL", "AV CALLE", "AVC", "AV CL")
_register("AK", "AVENIDA CARRERA", "AVKR", "AV CARRERA", "AVK", "AV KR", "AV CRA")
_register("DG", "DIAGONAL", "DIAG", "DIGONAL", "D")
_register("TV", "TRANSVERSAL", "TRANSV", "TRANV", "TRV", "TRAV", "TRANSVESAL", "T")
_register("PJ", "PASAJE", "PSJ", "PJE")
_register("PS", "PASEO")
_register("CV", "CIRCUNVALAR", "CIRCUNV", "CIRC")
_register("AU", "AUTOPISTA", "AUTOP")
_register("CT", "CALLEJON")

#: Quadrant / cardinal suffixes. Only spellings of length >= 3 are accepted as a
#: quadrant when they sit right after a street number, because single letters in
#: that position are street letters in cadastral practice (``KR 26 N``).
QUADRANT_CANON: dict[str, str] = {
    "NORTE": "NORTE", "NTE": "NORTE", "NOR": "NORTE", "NORT": "NORTE",
    "SUR": "SUR",
    "ESTE": "ESTE", "EST": "ESTE",
    "OESTE": "OESTE", "OES": "OESTE", "OESTE.": "OESTE", "WEST": "OESTE",
}
#: Two-letter quadrant abbreviations. Single letters (``N``, ``S``, ``E``, ``O``,
#: ``W``) are deliberately NOT treated as quadrants: the cadastre writes
#: quadrants out in full (NORTE 30k / OESTE 21k occurrences, ESTE 0) and uses
#: single letters as street letters (``KR 41 E``, ``C 60 N``, ``K 73 B # 2 W``).
QUADRANT_SHORT: dict[str, str] = {"OE": "OESTE", "NTE": "NORTE"}
_QUADRANT_WORDS = ("NORTE", "SUR", "ESTE", "OESTE")

#: Complement (unit) kinds. Canonical spellings follow the cadastral layer, so
#: e.g. ``BLQ`` (not ``BL``) and ``PQ``/``GA`` are kept as they appear there.
COMPLEMENT_CANON: dict[str, str] = {}


def _register_comp(canon: str, *aliases: str) -> None:
    COMPLEMENT_CANON[canon] = canon
    for alias in aliases:
        COMPLEMENT_CANON[alias] = canon


_register_comp("AP", "APTO", "APARTAMENTO", "APT", "APA", "APARTAMENT", "APTO.")
_register_comp("TO", "TORRE", "TOR", "TORR")
_register_comp("LC", "LOCAL", "LOC", "LCL")
_register_comp("CA", "CASA", "CS")
_register_comp("ET", "ETAPA", "ETP")
_register_comp("BLQ", "BLOQUE", "BL", "BLK", "BQ")
_register_comp("MZ", "MANZANA", "MZN", "MZA")
_register_comp("LT", "LOTE", "LOT")
_register_comp("PH", "PENTHOUSE")
_register_comp("OF", "OFICINA", "OFC", "OFI")
_register_comp("ED", "EDIFICIO", "EDIF", "EDF")
_register_comp("CONJ", "CONJUNTO", "CJTO", "CONJT")
_register_comp("UR", "URBANIZACION", "URB")
_register_comp("INT", "INTERIOR", "IN")
_register_comp("PQ", "PARQUEADERO", "PARQ")
_register_comp("GA", "GARAJE", "GAR")
_register_comp("BG", "BODEGA", "BOD")
_register_comp("BR", "BARRIO", "BRR")
_register_comp("CGTO", "CORREGIMIENTO", "CORR")
_register_comp("VDA", "VEREDA")

#: Kind table the retriever was trained and tuned with. ``ParsedAddress.complement``
#: (encoder query text, quadrant-in-complement rule) is canonicalized with this
#: frozen snapshot only, so kinds added below for the complement stage cannot move
#: ``confianza`` or flip accept/abstain. Do not register new kinds before this line.
_SCORING_COMPLEMENT_CANON: dict[str, str] = dict(COMPLEMENT_CANON)

# Complement-stage-only kinds (merge_complement / render_complement).
_register_comp("PISO", "PIS")
_register_comp("AGR", "AGRUPACION", "AGRUP", "AGP")
_register_comp("SEC", "SECTOR", "SECT")

#: Single-letter spellings accepted only when glued to their value with a hyphen
#: (``T-F`` -> ``TO F``). A bare ``T`` is not a registered alias: it is too
#: ambiguous to trust outside that shape.
_GLUED_ONLY_ALIASES: dict[str, str] = {"T": "TO"}
_GLUED_KIND_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{1,12})\s*-\s*(?=[A-Z0-9])")

#: Trailing administrative noise that carries no address information.
CITY_NOISE = (
    "CALI", "SANTIAGO DE CALI", "VALLE DEL CAUCA", "VALLE", "COLOMBIA",
    "V/CAUCA", "VALLE DEL CAUCA COLOMBIA", "CAUCA", "VALLE CAUCA",
)

_BIS_WORDS = ("BIS", "BS", "BISS")

_LETTER_RE = re.compile(r"^[A-Z]{1,4}$")
_NUM_RE = re.compile(r"^\d{1,4}$")
_HAS_LETTER_RE = re.compile(r"[A-Z]")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ParsedAddress:
    """Structured representation of a Colombian address."""

    raw: str = ""
    parse_ok: bool = False
    via_type: str | None = None
    via_number: str | None = None
    via_letters: str | None = None
    via_suffix: str | None = None
    via_suffix_letters: str | None = None
    via_bis: bool = False
    via_quadrant: str | None = None
    cross_type: str | None = None
    cross_number: str | None = None
    cross_letters: str | None = None
    cross_suffix: str | None = None
    cross_suffix_letters: str | None = None
    cross_bis: bool = False
    cross_quadrant: str | None = None
    plate: str | None = None
    #: Scoring complement (encoder query text, gating rules). Frozen spelling: see
    #: ``_SCORING_COMPLEMENT_CANON``.
    complement: str | None = None
    #: The complement as typed (upper-case), input of the complement stage
    #: (:func:`merge_complement`, :func:`render_complement`). Never scored.
    complement_text: str | None = None
    had_hash: bool = False
    had_plate_sep: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical(self, style: str = "spaced", with_complement: bool = True) -> str:
        return canonical(self, style=style, with_complement=with_complement)

    @property
    def has_cross(self) -> bool:
        return self.cross_number is not None

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return self.canonical()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−_"), "-")


def strip_accents(text: str) -> str:
    """ASCII-fold a string (``Ñ`` -> ``N``), tolerating any unicode input."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


_NUMBER_SIGN_PUNCT_RE = re.compile(r"\b(?:NO|N\s*[°º])\s*[,:]\s*(?=\d)")


def _first_number_sign(text: str, pattern: re.Pattern) -> str:
    """Rewrite the first ``pattern`` match as ``#``, only before any `#` or `-` and not after a unit kind."""
    m = pattern.search(text)
    if m is None or "#" in text[: m.start()] or "-" in text[: m.start()]:
        return text
    before = text[: m.start()].split()
    if before and before[-1] in COMPLEMENT_CANON:
        return text         # `CASA NO18`: a house number inside the complement
    return text[: m.start()] + " # " + text[m.end():]


def _number_sign_with_punctuation(text: str) -> str:
    """`NO, 12` / `NO: 12` / `N°, 12` -> `# 12`, only as the number sign of the main address.

    Without it the token NO is read as N + OESTE. Only the first occurrence counts, and
    only before any `#` or `-`: inside the complement (`CASA NO,4`) it is not a number
    sign. A lone `N, 12` stays a cardinal letter.
    """
    return _first_number_sign(text, _NUMBER_SIGN_PUNCT_RE)


#: ``NO18`` / ``NO18-61``: the number sign glued to its digits (a quadrant is never glued to a number).
_NUMBER_SIGN_GLUED_RE = re.compile(r"\bNO(?=\d)")
#: ``48AN°90``: a degree sign glued to ``N`` is the number sign, the letter before it stays.
_GLUED_N_SIGN_RE = re.compile(r"(?<=\d)([A-Z]?)N\s*°\s*\.?\s*(?=\d)")


#: Coordinates glued to an address are not part of it: a decimal number with 4+ decimals (``-70.98765``). Long
#: digit runs are NOT dropped here (id-like garbage stays unparseable); the fidelity check ignores them instead.
_COORD_RE = re.compile(r"(?<![\w.,])-?\d{1,3}[.,]\d{4,}(?!\d)")


def _symbols_to_text(raw: str) -> str:
    """Ordinal indicators and degree signs before the ASCII fold (NFKD turns ``º`` / ``ª`` into the letters O / A).

    ``N°`` double-encoded as ``NÂ°`` is repaired (the same table as :func:`cali_address.fidelity._fold`), ``Nº`` is
    ``N°``, and any other ordinal indicator (``2ª``, ``6º``) is an ordinal marker, not a street letter.
    """
    text = raw.replace("Â°", "°").replace("Âº", "º")
    text = re.sub(r"(?i)(N)\s*[ºª]", r"\1°", text)
    return text.replace("º", " ").replace("ª", " ")


def _pre_normalize(raw: str) -> str:
    """Uppercase, ASCII-fold, unify separators and `number` words."""
    text = strip_accents(_symbols_to_text(raw)).upper()
    text = _COORD_RE.sub(" ", text.translate(_DASHES))
    # `N° 12` / `Nº 12` must be handled before the degree signs are stripped.
    text = re.sub(r"\bN\s*[°º]\s*\.?\s*(?=\d)", " # ", text)
    text = _GLUED_N_SIGN_RE.sub(lambda m: f" {m.group(1)} # ", text)
    text = _number_sign_with_punctuation(text)
    text = _first_number_sign(text, _NUMBER_SIGN_GLUED_RE)
    text = text.replace("º", " ").replace("°", " ").replace("ª", " ")
    # `No.12`, `No 12`, `NRO 12`, `NUMERO 12` -> `# 12`
    text = re.sub(r"\bNRO?S?\.?\s*(?=[\d#])", " # ", text)
    text = re.sub(r"\bNUMERO?S?\.?\s*(?=[\d#])", " # ", text)
    text = re.sub(r"\bNUM\.?\s*(?=[\d#])", " # ", text)
    text = re.sub(r"\bNO?\s*\.\s*(?=\d)", " # ", text)     # "No." / "N."
    text = re.sub(r"\bNO\s+(?=\d)", " # ", text)            # "No 12"
    # Ordinal noise: 3RA, 5TA, 1RO, 2DO, 4TO, 7MA, 9NA ...
    text = re.sub(r"(?<=\d)(RA|RO|TA|TO|DA|DO|MA|MO|NA|ER|ERA|VA|VO)\b", " ", text)
    text = re.sub(r"#+", " # ", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.replace("/", " ")
    text = re.sub(r"[^\w#\-;,. ]+", " ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _drop_city_noise(text: str) -> tuple[str, list[str]]:
    """Remove trailing `, Cali, Valle del Cauca`-style administrative tails."""
    notes: list[str] = []
    parts = [p.strip() for p in text.split(",")]
    kept: list[str] = []
    for i, part in enumerate(parts):
        norm = re.sub(r"\s+", " ", part).strip(" .")
        if i > 0 and norm in CITY_NOISE:
            notes.append("dropped_city_noise")
            continue
        kept.append(part)
    text = ", ".join(p for p in kept if p)
    for noise in sorted(CITY_NOISE, key=len, reverse=True):
        new = re.sub(r"(?:[\s,.-]+)" + re.escape(noise) + r"\s*$", "", text)
        if new != text:
            notes.append("dropped_city_noise")
            text = new
    # A trailing '-' is meaningful (the cadastre writes `KR 27 #  -` for records
    # with no plate), so it is preserved.
    return text.lstrip(" ,.-").rstrip(" ,."), notes


def _split_glued_marked(text: str) -> tuple[list[str], list[bool]]:
    """:func:`_split_glued` plus, per token, whether it was glued to the previous one (a letter<->digit boundary)."""
    text = re.sub(r"(?<=[A-Z])(?=\d)", "\x00", text)
    text = re.sub(r"(?<=\d)(?=[A-Z])", "\x00", text)
    for ch in "-.,#;:":
        text = text.replace(ch, " ")
    tokens: list[str] = []
    glued: list[bool] = []
    for piece in text.split():
        for k, part in enumerate(piece.split("\x00")):
            if part:
                tokens.append(part)
                glued.append(k > 0)
    return tokens, glued


def _split_glued(text: str) -> list[str]:
    """Tokenize a via/cross segment, splitting letter<->digit boundaries.

    ``CL12A`` -> ``['CL', '12', 'A']``; ``3E-1`` -> ``['3', 'E', '1']``.
    """
    return _split_glued_marked(text)[0]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_VIA_ALIASES_LONG = None


def _fuzzy_via_type(token: str) -> str | None:
    """Repair via-type typos (``CATRERA`` -> ``KR``, ``CLLLE`` -> ``CL``)."""
    global _VIA_ALIASES_LONG
    if len(token) < 4 or not token.isalpha():
        return None
    if _VIA_ALIASES_LONG is None:
        _VIA_ALIASES_LONG = [a for a in VIA_TYPE_CANON if len(a) >= 4 and a.isalpha()]
    best, best_d = None, 99
    for alias in _VIA_ALIASES_LONG:
        d = _levenshtein(token, alias)
        if d < best_d:
            best, best_d = alias, d
    if best is not None and best_d <= max(1, len(best) // 4):
        return VIA_TYPE_CANON[best]
    return None


def _fuzzy_quadrant(token: str) -> str | None:
    """Repair quadrant typos such as ``OEATE`` -> ``OESTE`` by edit distance."""
    if len(token) < 4:
        return None
    best, best_d = None, 99
    for word in _QUADRANT_WORDS:
        d = _levenshtein(token, word)
        if d < best_d:
            best, best_d = word, d
    if best is not None and best_d <= max(1, len(best) // 4):
        return best
    return None


# ---------------------------------------------------------------------------
# Segment parsing
# ---------------------------------------------------------------------------


def _parse_segment(tokens: list[str]) -> dict:
    """Parse ``KR 26 H 1 B BIS OESTE`` style token runs."""
    out = {
        "type": None, "number": None, "letters": None, "suffix": None,
        "suffix_letters": None, "bis": False, "quadrant": None, "leftover": [],
    }
    i, n = 0, len(tokens)
    if i + 1 < n and f"{tokens[i]} {tokens[i + 1]}" in VIA_TYPE_CANON:
        out["type"] = VIA_TYPE_CANON[f"{tokens[i]} {tokens[i + 1]}"]
        i += 2
    elif i < n and tokens[i] in VIA_TYPE_CANON:
        out["type"] = VIA_TYPE_CANON[tokens[i]]
        i += 1
    elif i < n and _fuzzy_via_type(tokens[i]) is not None:
        out["type"] = _fuzzy_via_type(tokens[i])
        i += 1
    # Leading BIS / spelled-out quadrant before the number is rare but legal.
    while i < n and not _NUM_RE.match(tokens[i]):
        tok = tokens[i]
        if tok in _BIS_WORDS:
            out["bis"] = True
            i += 1
            continue
        q = QUADRANT_CANON.get(tok)
        if q is not None:
            out["quadrant"] = q
            i += 1
            continue
        break
    if i < n and _NUM_RE.match(tokens[i]):
        out["number"] = str(int(tokens[i]))
        i += 1
    else:
        out["leftover"] = tokens[i:]
        return out
    # Letter run right after the number (street letter, e.g. `26 H`). The
    # cadastre occasionally stacks two of them (`A 2 A N`, `K 24 D W`).
    letters: list[str] = []
    while i < n and len(letters) < 2 and _LETTER_RE.match(tokens[i])             and tokens[i] not in _BIS_WORDS:
        tok = tokens[i]
        q = QUADRANT_CANON.get(tok) or QUADRANT_SHORT.get(tok)
        if q is not None:
            out["quadrant"] = q
            i += 1
            break
        if not letters and len(tok) == 4 and tok[1:] in QUADRANT_CANON and tok[0] != tok[1]:
            break       # `3 BSUR`: letter + spelled quadrant, read by the tail loop below
        if (
            len(tok) == 2
            and tok[1] in "NOS"
            and tok not in VIA_TYPE_CANON
            and tok not in QUADRANT_SHORT
            and not letters
        ):
            # `AV 5 AN` / `# 23 DN` -> street letter + glued cardinal suffix.
            letters.append(tok[0])
            out["quadrant"] = {"N": "NORTE", "O": "OESTE", "S": "SUR"}[tok[1]]
            i += 1
            break
        letters.append(tok)
        i += 1
    if letters:
        out["letters"] = " ".join(letters)
    # Optional secondary number (`KR 26 H 1`) and its letter (`KR 1 A 5 B`).
    if i < n and _NUM_RE.match(tokens[i]):
        out["suffix"] = str(int(tokens[i]))
        i += 1
        if i < n and _LETTER_RE.match(tokens[i]) and tokens[i] not in _BIS_WORDS \
                and tokens[i] not in QUADRANT_CANON:
            out["suffix_letters"] = tokens[i]
            i += 1
    # BIS / quadrant tail in any order.
    while i < n:
        tok = tokens[i]
        if tok in _BIS_WORDS:
            out["bis"] = True
            i += 1
            continue
        q = QUADRANT_CANON.get(tok) or QUADRANT_SHORT.get(tok)
        if q is not None and out["quadrant"] is None:
            out["quadrant"] = q
            i += 1
            continue
        if (
            out["letters"] is None and out["quadrant"] is None and len(tok[1:]) >= 3 and _is_letter_quadrant(tok)
        ):
            # `3BNORTE` / `3AOESTE`: a street letter glued to a spelled quadrant. The fuzzy repair below would
            # swallow the letter as a typo of the quadrant. A doubled first letter (`OOESTE`) is such a typo.
            out["letters"], out["quadrant"] = tok[0], QUADRANT_CANON[tok[1:]]
            i += 1
            continue
        fq = _fuzzy_quadrant(tok)
        if fq is not None and out["quadrant"] is None:
            out["quadrant"] = fq
            i += 1
            continue
        break
    out["leftover"] = tokens[i:]
    return out


def _glue_kinds(text: str) -> str:
    """Drop the hyphen that glues a unit kind to its value (``T-F`` -> ``TO F``).

    Works on upper-case text. Only a registered kind (or a glued-only alias such
    as ``T``) directly before the hyphen is rewritten; hyphens between values
    (``AP 101 - A``) are left alone.
    """
    def repl(m: re.Match) -> str:
        tok = m.group(1)
        canon = COMPLEMENT_CANON.get(tok) or _GLUED_ONLY_ALIASES.get(tok)
        return f"{canon} " if canon else m.group(0)

    return _GLUED_KIND_RE.sub(repl, text)


#: Digits glued to a run of two or more letters (``501TORRED``, ``5TO``, ``2PISOS``).
_GLUED_DIGITS_WORD_RE = re.compile(r"\b(\d+)([A-Z]{2,})\b")
_MAX_KIND_LEN = max(len(k) for k in COMPLEMENT_CANON)


def _split_glued_values(text: str) -> str:
    """Split digits glued to a word (upper-case, accent-free text).

    * Glued to a registered kind word plus at most one letter: split at the kind
      (``501TORRED`` -> ``501 TORRE D``, ``5TO`` -> ``5 TO``). A trailing ``S`` is a
      plural, not a unit letter (``2PISOS`` stays one word).
    * Glued to any other word of three or more letters: split the digits off, so
      the word can never be part of a unit code (``501UMBRAL`` -> ``501 UMBRAL``).
    * Digits plus one or two letters (``501A``, ``2AB``) are left alone: legit codes.
    """
    def repl(m: re.Match) -> str:
        digits, letters = m.groups()
        for n in range(min(len(letters), _MAX_KIND_LEN), 1, -1):
            if letters[:n] in COMPLEMENT_CANON:
                rest = letters[n:]
                if len(rest) <= 1 and rest != "S":
                    return f"{digits} {letters[:n]} {rest}".rstrip()
                break  # the longest kind prefix is followed by a word: not a unit
        return f"{digits} {letters}" if len(letters) >= 3 else m.group(0)

    return _GLUED_DIGITS_WORD_RE.sub(repl, text)


_TOWER_CODE_RE = re.compile(r"[A-Z]|\d{1,3}")
_GLUED_TOWER_RE = re.compile(r"T(\d{1,3})")


def _bare_tower_alias(text: str) -> str:
    """``T B`` / ``T 2`` / ``T2`` -> ``TO B`` / ``TO 2`` (upper-case text).

    A bare ``T`` is too ambiguous to be a registered kind, so it only reads as a tower
    when a single-letter or short-number code (up to 3 digits) follows it, or is glued to
    it (``T2``), and the word before it is not something the ``T`` could belong to: a unit
    kind (``BLQ T 3``), a marker word (``NO T 2``), a place word (``ESQUINA T 2``), a
    connector, or a number (``AP 101 T B``: a letter after a number is a unit letter).
    """
    tokens = text.split(" ")
    for i, tok in enumerate(tokens):
        glued = _GLUED_TOWER_RE.fullmatch(tok)
        if glued is None and (tok != "T" or i + 1 >= len(tokens) or not _TOWER_CODE_RE.fullmatch(tokens[i + 1])):
            continue
        before = tokens[i - 1] if i else ""
        if (
            before in COMPLEMENT_CANON or before in _MARKER_WORDS or before in DESCRIPTIVE_SEED
            or before in _VALUE_STOPWORDS or before[-1:].isdigit()
        ):
            continue
        tokens[i] = "TO" if glued is None else f"TO {glued.group(1)}"
    return " ".join(tokens)


def _units_text(text: str) -> str:
    """Upper-case, accent-free complement text with glued kinds and values split."""
    return _bare_tower_alias(_split_glued_values(_glue_kinds(strip_accents(str(text)).upper())))


def _complement_text(text: str) -> str | None:
    """Complement as typed: upper-case, commas dropped, whitespace collapsed."""
    text = re.sub(r"\s+", " ", text.upper().replace(",", " ")).strip(" .-")
    return text or None


def _canon_complement(text: str) -> str | None:
    """Scoring complement: map known kind words through the frozen
    :data:`_SCORING_COMPLEMENT_CANON` and keep every other token verbatim.

    This is the text the retriever sees, so it must not change when the complement
    stage learns new kinds or spellings; that stage reads ``complement_text``.
    """
    text = _complement_text(text)
    if text is None:
        return None
    return " ".join(_SCORING_COMPLEMENT_CANON.get(tok, tok) for tok in text.split()) or None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_JUNK_RE = re.compile(r"^[\d\s.,;:#/-]*$")


#: Parse notes that mean "the parse lost part of the address": such a parse can never be verified
#: against the cadastre (see :func:`is_truncated`). ``dropped_trailing_segment`` alone is NOT one of
#: them: a neighbourhood after a complete address (``CL 5 # 10 - 20, NAVARRO``) is not lost data.
TRUNCATION_NOTES = ("truncated_parse",)


def is_truncated(parsed) -> bool:
    """True when ``parsed`` carries a truncation note (never raises; ``None`` is not truncated)."""
    notes = getattr(parsed, "notes", None)
    return isinstance(notes, list) and any(note in notes for note in TRUNCATION_NOTES)


def _is_bare_via(head: str) -> bool:
    """True when ``head`` is only a via: type + number + optional letters / suffix number / BIS / quadrant."""
    seg = _parse_segment(_split_glued(head))
    if seg["type"] is None or seg["number"] is None or seg["leftover"]:
        return False
    # a numeric suffix (``AV 2 B 6``) only follows a one-letter street letter: ``KR 987 AP 3`` is a unit, not a via
    return seg["suffix"] is None or len(seg["letters"] or "") == 1


def _is_cross_plate_tail(tail: str) -> bool:
    """True when ``tail`` reads as ``[#] <cross> - <plate...>``: a cross segment (a via type such as ``AV`` is fine)
    with no leftover words (letters, a numeric suffix and a quadrant are fine) and a number after the dash."""
    text = tail.strip()
    if text.startswith("#"):
        text = text[1:].strip()
    left, sep, right = text.partition(" - ")
    if not sep or not right[:1].isdigit():
        return False
    seg = _parse_segment(_split_glued(left))
    return seg["number"] is not None and not seg["leftover"]


def parse_address(raw: str | None) -> ParsedAddress:
    """Parse a free-form Cali address into structured fields."""
    res = ParsedAddress(raw="" if raw is None else str(raw))
    if raw is None:
        res.notes.append("null_input")
        return res
    text0 = str(raw)
    if not text0.strip():
        res.notes.append("empty_input")
        return res
    tail_parts: list[str] = []
    dropped_digits = False
    semicolon_tail = ""
    if ";" in text0:
        res.notes.append("multiple_addresses_first_taken")
        text0, _, semicolon_tail = text0.partition(";")

    text = _pre_normalize(text0)
    text, noise_notes = _drop_city_noise(text)
    res.notes.extend(noise_notes)
    if not text:
        res.notes.append("empty_after_cleanup")
        return res
    if _JUNK_RE.match(text):
        res.notes.append("no_alphabetic_content")
        return res

    # Drop a trailing neighbourhood segment after a comma when the head already
    # looks like an address (`Cl. 55b # 47-57, Navarro` -> `Cl. 55b # 47-57`).
    while "," in text:
        head, _, tail = text.partition(",")
        head = head.strip()
        if (
            head and "#" not in head and "-" not in head and _is_bare_via(head)
            and _is_cross_plate_tail(tail)
        ):
            # `Carrera 987, 654B-20`: the head is ONLY a via (no cross, plate, units or words)
            # and the tail reads as cross + plate, so the comma just separates the two
            # instead of opening a trailing neighbourhood segment.
            # An explicit ``#``: without it a numeric via suffix (``AV 2 B 6``) would be read as the cross.
            text = f"{head} # {tail.strip().lstrip('#').strip()}"
            continue
        if head and ("#" in head or any(t in VIA_TYPE_CANON for t in _split_glued(head))):
            if tail.strip():
                res.notes.append("dropped_trailing_segment")
                tail_parts.append(tail)
                dropped_digits = dropped_digits or _tail_loses_digits(tail)
            text = head
        break
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # `Carrera 987 entre calle 654 y calle 321` / `CL 987 con KR 654`: the numbers are STREETS, never a plate.
    intersection = None
    if "#" not in text and " - " not in text:
        text, intersection = _split_intersection(text)
        if intersection is not None:
            res.notes.append("intersection_reference")
            dropped_digits = dropped_digits or intersection[1]

    # --- locate the '#' -------------------------------------------------
    if "#" in text:
        via_text, _, rest = text.partition("#")
        res.had_hash = True
        res.notes.append("explicit_hash")
    elif " - " in text:
        # No `#` but an explicit dash. Two shapes share this branch:
        # `Catrera 34 5b3-65` (a cross candidate `5B3` already sits between the
        # via and the dash, so the right side of the dash is the plate) and
        # `Calle 5 - 38` (nothing sits between the via and the dash, so there is
        # no cross candidate on the left: the number on the right IS the cross,
        # not a plate - a bare dash without `#` never introduces a plate on its
        # own).
        left, _, right = text.partition(" - ")
        via_text, cross_text_pre = _segment_without_hash(left, res, allow_plate=False)
        rest = f"{cross_text_pre} - {right}" if cross_text_pre else right
    else:
        via_text, rest = _segment_without_hash(text, res)

    # --- locate the plate separator ------------------------------------
    rest = rest.strip()
    if intersection is not None:
        cross_text, plate_text = intersection[0], ""
    elif " - " in rest:
        cross_text, _, plate_text = rest.partition(" - ")
        res.had_plate_sep = True
    elif rest.endswith("-"):
        cross_text, plate_text = rest[:-1], ""
        res.had_plate_sep = True
    elif rest.startswith("- "):
        cross_text, plate_text = "", rest[2:]
        res.had_plate_sep = True
    else:
        cross_text, plate_text = _split_cross_plate_no_dash(rest, res)

    via_tokens = _split_glued(via_text)
    if not via_tokens:
        res.notes.append("empty_via_segment")
        return res
    via = _parse_segment(via_tokens)
    if via["type"] is None or via["number"] is None:
        res.notes.append("via_unparsed")
        return res
    res.via_type = via["type"]
    res.via_number = via["number"]
    res.via_letters = via["letters"]
    res.via_suffix = via["suffix"]
    res.via_suffix_letters = via["suffix_letters"]
    res.via_bis = via["bis"]
    res.via_quadrant = via["quadrant"]

    cross = _parse_segment(_split_glued(cross_text))
    res.cross_type = cross["type"]
    res.cross_number = cross["number"]
    res.cross_letters = cross["letters"]
    res.cross_suffix = cross["suffix"]
    res.cross_suffix_letters = cross["suffix_letters"]
    res.cross_bis = cross["bis"]
    res.cross_quadrant = cross["quadrant"]

    plate_text = plate_text.strip()
    plate_rest = ""
    # Letters glued to the plate (`08CA`, `12A`) stay in the complement; hiding the plate
    # behind them made the plate rules skip the row (`- 08CA` vs `- 12` was accepted).
    m = re.match(r"^(\d{1,6})(?!\d)\s*(.*)$", plate_text)
    if m:
        res.plate = m.group(1)
        plate_rest = m.group(2)
    else:
        plate_rest = plate_text

    comp = " ".join(
        part for part in (
            " ".join(via["leftover"]), " ".join(cross["leftover"]), plate_rest,
        ) if part.strip()
    )
    res.complement = _canon_complement(comp)
    tail_units = _tail_unit_segments([*tail_parts, semicolon_tail])
    if tail_units:
        res.notes.append("complement_from_trailing_segment")
    res.complement_text = _complement_text(" ".join([comp, *tail_units]))
    if dropped_digits and (res.cross_number is None or res.plate is None):
        # The text after the comma had digits and the part kept has no cross or no plate: the cross / plate
        # may have been in the dropped part, so the kept base must never be verified as the whole address.
        res.notes.append("truncated_parse")
    res.parse_ok = True
    return res


_INTERSECTION_WORDS = ("ENTRE", "CON")


def _is_plain_via(head: str) -> bool:
    """A bare via with no secondary number and at most one street letter (what the `#`-less reading keeps whole)."""
    seg = _parse_segment(_split_glued(head))
    return (
        seg["type"] is not None and seg["number"] is not None and not seg["leftover"]
        and seg["suffix"] is None and len((seg["letters"] or "").split()) <= 1
    )


def _first_cross(tokens: list[str]) -> tuple[str, list[str]] | None:
    """The first cross street of an intersection (type + number + one letter / BIS / quadrant) and what is left."""
    toks = _split_glued(" ".join(tokens))
    i, n = 0, len(toks)
    if i + 1 < n and f"{toks[i]} {toks[i + 1]}" in VIA_TYPE_CANON:
        i += 2
    elif i < n and toks[i] in VIA_TYPE_CANON:
        i += 1
    if i >= n or not _NUM_RE.match(toks[i]):
        return None
    i += 1
    if i < n and _LETTER_RE.match(toks[i]) and toks[i] not in _BIS_WORDS and toks[i] != "Y":
        i += 1
    while i < n and (toks[i] in _BIS_WORDS or toks[i] in QUADRANT_CANON):
        i += 1
    return " ".join(toks[:i]), toks[i:]


def _split_intersection(text: str) -> tuple[str, tuple[str, bool] | None]:
    """``(via, (first cross, lost digits))`` for ``<via> ENTRE|CON <cross> ...`` (no ``#`` / ``-``), else ``(text, None)``.

    The head must be a bare via. The text after the first cross street (``Y CALLE 3``) is not kept; ``lost digits``
    says whether it held numbers (the parse then carries ``truncated_parse``: it can never be verified).
    """
    tokens = text.split()
    for k in range(2, len(tokens) - 1):
        if tokens[k] in _INTERSECTION_WORDS:
            head = " ".join(tokens[:k])
            after = tokens[k + 1:]
            # Only when the numbers after the marker would otherwise become a plate (two or more of them); a lone
            # cross (``CL 987 con KR 654``) keeps its historical reading.
            if sum(1 for tok in _split_glued(" ".join(after)) if _NUM_RE.match(tok)) < 2 or not _is_plain_via(head):
                return text, None
            cross = _first_cross(after)
            if cross is None:
                return text, None
            return head, (cross[0], any(ch.isdigit() for tok in cross[1] for ch in tok))
    return text, None


def _tail_unit_segments(parts: list[str]) -> list[str]:
    """Segments of the text after a comma / semicolon that are building units.

    ``Calle 5 # 38-10, Torre B`` keeps ``TORRE B``; ``, San Fernando``, ``, Cali``, a
    barrio / vereda, free text and a second address (``; KR 6 # 4-3``) are ignored. A
    segment qualifies when it starts with a registered unit kind and yields at least one
    unit (:func:`extract_complement_units`). The result only feeds ``complement_text``:
    the frozen scoring ``complement`` (the encoder query) never sees it.
    """
    kept: list[str] = []
    for part in parts:
        for segment in re.split(r"[;,]", _pre_normalize(part)):
            segment = segment.strip(" .-")
            tokens = _units_text(segment).split() if segment else []
            if tokens and tokens[0] in COMPLEMENT_CANON and extract_complement_units(segment):
                kept.append(segment)
    return kept


def _tail_loses_digits(tail: str) -> bool:
    """True when a dropped tail has a segment with digits that is not a building unit (``TORRE 3``)."""
    for segment in re.split(r"[;,]", _pre_normalize(tail)):
        if any(ch.isdigit() for ch in segment) and not _tail_unit_segments([segment]):
            return True
    return False


def _segment_without_hash(text: str, res: ParsedAddress, allow_plate: bool = True) -> tuple[str, str]:
    """Recover the via/cross boundary when the `#` is missing.

    ``CL 9 C 47 141``    -> via ``CL 9 C``,  rest ``47 - 141``
    ``KR 13 61 34 38``   -> via ``KR 13``,   rest ``61 - 34 38``
    """
    res.notes.append("implicit_hash")
    tokens, glued = _split_glued_marked(text)
    i, n = 0, len(tokens)
    via: list[str] = []
    if i + 1 < n and f"{tokens[i]} {tokens[i+1]}" in VIA_TYPE_CANON:
        via += tokens[i:i + 2]
        i += 2
    elif i < n and (tokens[i] in VIA_TYPE_CANON or _fuzzy_via_type(tokens[i])):
        via.append(tokens[i])
        i += 1
    if i < n and _NUM_RE.match(tokens[i]):
        via.append(tokens[i])
        i += 1
    # One optional letter run and one optional secondary number stay with the via.
    if i < n and _LETTER_RE.match(tokens[i]):
        via.append(tokens[i])
        i += 1
        if _glued_via_designator(tokens, glued, i, allow_plate):
            # `KR 987G6 654S1 39`, `KR 987 D2 N 654C 321`: the digit glued to the street letter is the via's
            # secondary number (the reading an explicit `#` already had), optionally followed by its letter.
            via.append(tokens[i])
            i += 1
            if i + 1 < n and _LETTER_RE.match(tokens[i]) and tokens[i] not in _BIS_WORDS and tokens[i] != "Y" \
                    and _NUM_RE.match(tokens[i + 1]):
                via.append(tokens[i])
                i += 1
    if (
        len(via) >= 3 and i + 1 < n and tokens[i] in ("N", "S", "E", "O", "W")
        and _NUM_RE.match(tokens[i + 1]) and not glued[i + 1]
    ):
        via.append(tokens[i])       # `KR 7A N 72 B 04`: a cardinal letter after the street letter; the cross starts with a digit
        i += 1
    while i < n and (
        tokens[i] in _BIS_WORDS or tokens[i] in QUADRANT_CANON or tokens[i] in QUADRANT_SHORT
        or (len(via) < 4 and _is_letter_quadrant(tokens[i]))
    ):
        via.append(tokens[i])
        i += 1
    rest = tokens[i:]
    if not allow_plate:
        return " ".join(via), " ".join(rest)
    if rest and _NUM_RE.match(rest[0]):
        stop = _stage_cut(rest)
        head, tail = rest[:stop], rest[stop:]
        cut = _plate_cut(head, glued[i:i + stop])
        return " ".join(via), " ".join(head[:cut]) + " - " + " ".join(head[cut:] + tail)
    return " ".join(via), " ".join(rest)


def _is_letter_quadrant(tok: str) -> bool:
    """``BNORTE`` / ``AOESTE``: a street letter glued to a spelled quadrant (not a doubled-letter typo such as ``OOESTE``)."""
    return len(tok) >= 4 and tok[1:] in QUADRANT_CANON and tok[0] != tok[1]


def _glued_via_designator(tokens: list[str], glued: list[bool], i: int, allow_plate: bool) -> bool:
    """True when ``tokens[i]`` is a digit glued to the SINGLE street letter before it and enough numbers follow.

    Reading ``987A9`` as street letter + secondary number needs a cross AND a plate after it; one number left of a
    dash is enough (the dash promises the plate). Without a dash the run of numbers before the first free word
    (``APARTAMENTO 211`` is not part of the address) must hold three numbers, or two with a letter glued to the cross
    number (``654C 321``): two bare numbers are genuinely ambiguous (``CALLE987C654 321``: cross 654, plate 321), so
    the historical reading stays. ``KR 987A9-250`` keeps it too (cross 9).
    """
    n = len(tokens)
    if i >= n or len(tokens[i - 1]) != 1 or not _NUM_RE.match(tokens[i]) or not glued[i]:
        return False
    j = i + 1
    while j < n and (tokens[j] in _BIS_WORDS or tokens[j] in QUADRANT_CANON):
        j += 1
    if j < n and len(tokens[j]) == 1 and tokens[j].isalpha():
        # an optional letter of the secondary number (`D2 N 654C`): it must be a separate token followed by a separate
        # number. A letter glued to the digit (`B5B2`) or to the next number (`B1 A9`) is a chain of designators:
        # genuinely ambiguous, so the historical reading stays.
        if glued[j] or j + 1 >= n or not _NUM_RE.match(tokens[j + 1]) or glued[j + 1]:
            return False
        j += 1
    if j >= n or not _NUM_RE.match(tokens[j]):
        return False        # the cross must start right here
    run: list[int] = []
    for k in range(j, n):
        if len(tokens[k]) >= 2 and tokens[k].isalpha():
            break
        run.append(k)
    numbers = sum(1 for k in run if _NUM_RE.match(tokens[k]))
    if not allow_plate:
        return numbers >= 1
    glued_letter = any(glued[k] and _LETTER_RE.match(tokens[k]) for k in run)
    return numbers >= 3 or (numbers >= 2 and glued_letter)


def _stage_cut(tokens: list[str]) -> int:
    """Index of the first stage marker followed by a number (``ET 2``, ``ETAPA 2``): units, never the plate."""
    for k in range(len(tokens) - 1):
        if COMPLEMENT_CANON.get(tokens[k]) == "ET" and _NUM_RE.match(tokens[k + 1]):
            return k
    return len(tokens)


def _plate_cut(tokens: list[str], glued: list[bool] | None = None) -> int:
    """Index at which the plate starts inside a `cross + plate` token run.

    ``['5','B','2','09']``          -> 3   (cross `5 B 2`, plate `09`)
    ``['49','141']``                -> 1   (cross `49`,    plate `141`)
    ``['60','34','38','LC','1']``   -> 1   (cross `60`,    plate `34`, rest complement)
    """
    numbers = [k for k, t in enumerate(tokens) if _NUM_RE.match(t)]
    if len(numbers) < 2:
        return len(tokens)
    cut = numbers[1]
    # `5 B 2 09`: when the cross carries a street letter, a secondary number can
    # follow it and the plate is then the very last token, so shift the cut one
    # number to the right. Bare number runs (`60 34 38`) are left alone.
    cross_has_letter = any(
        _LETTER_RE.match(t) and t not in _BIS_WORDS and t not in QUADRANT_CANON
        for t in tokens[:numbers[1]]
    )
    if (
        cross_has_letter
        and len(numbers) >= 3
        and numbers[2] == numbers[1] + 1
        and numbers[2] == len(tokens) - 1
    ):
        cut = numbers[2]
    if (
        glued is not None and cross_has_letter and len(numbers) >= 3
        and numbers[2] == numbers[1] + 1 and glued[numbers[1]]
    ):
        cut = numbers[2]        # `5B1 27 APT 201`: the digit glued to the letter belongs to the cross
    return cut


def _split_cross_plate_no_dash(rest: str, res: ParsedAddress) -> tuple[str, str]:
    """Split post-`#` text into cross and plate when no `-` separator is present."""
    tokens, glued = _split_glued_marked(rest)
    if not tokens:
        return "", ""
    stop = _stage_cut(tokens)
    head, tail = tokens[:stop], tokens[stop:]
    if sum(1 for t in head if _NUM_RE.match(t)) >= 2:
        res.notes.append("implicit_plate_separator")
        cut = _plate_cut(head, glued[:stop])
        return " ".join(head[:cut]), " ".join(head[cut:] + tail)
    return " ".join(head), " ".join(tail)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _plate_quadrant(p: ParsedAddress) -> str | None:
    """NORTE / SUR / OESTE when the first word after the plate spells it (``None`` without a plate or truncated)."""
    if p.plate is None or is_truncated(p):
        return None
    tokens = (p.complement_text or "").split()
    word = tokens[0].strip(".") if tokens else ""
    quadrant = QUADRANT_CANON.get(word) or QUADRANT_SHORT.get(word)
    if quadrant not in ("NORTE", "SUR", "OESTE"):
        return None
    # the word must sit right after the plate digits in the text, not come from a via / cross leftover
    pattern = rf"(?<!\d){re.escape(p.plate)}\s*{re.escape(tokens[0])}(?![A-Z0-9])"
    return quadrant if re.search(pattern, _pre_normalize(p.raw)) else None


def rule_canonical(p: ParsedAddress | None) -> str | None:
    """The rule-canonical text of a parse: base + recognised units (``None`` when nothing parsed).

    This is the text a non-OK row may emit (:func:`cali_address.catalog.keeps_rule_text`); never raises. A spelled
    quadrant that is the first word after the plate (``# 654 - 321 OESTE SAN FERNANDO``) is kept right after it, as the
    cadastre writes it; ``ESTE`` is not (no cadastral occurrence: it starts a place name).
    """
    if not isinstance(p, ParsedAddress):
        return None
    base = canonical(p, style="spaced", with_complement=False)
    quadrant = _plate_quadrant(p) if base else None
    if quadrant:
        base = f"{base} {quadrant}"
    comp = render_complement(p.complement_text) if base else None
    return (f"{base} {comp}" if comp else base) or None


def _render_segment(
    vtype: str | None, number: str | None, letters: str | None, suffix: str | None,
    suffix_letters: str | None, bis: bool, quadrant: str | None, style: str,
) -> str:
    if number is None:
        return ""
    parts: list[str] = []
    if vtype:
        parts.append(vtype)
    if style == "glued" and letters:
        parts.append(f"{number}{letters.replace(' ', '')}")
    else:
        parts.append(number)
        if letters:
            parts.append(letters)
    if suffix:
        if style == "glued" and suffix_letters:
            parts.append(f"{suffix}{suffix_letters}")
        else:
            parts.append(suffix)
            if suffix_letters:
                parts.append(suffix_letters)
    if bis:
        parts.append("BIS")
    if quadrant:
        parts.append(quadrant)
    return " ".join(parts)


def canonical(p: ParsedAddress, style: str = "spaced", with_complement: bool = True) -> str:
    """Render a parsed address in the cadastral canonical form.

    ``style='spaced'`` emits ``KR 96 F # 96 - 66`` (the cadastral layer style);
    ``style='glued'`` emits ``KR 96F # 96 - 66`` (the IDESC ``dir_ajusta`` style).
    The plate is zero-padded to two digits, as the cadastre does.
    """
    if not p.parse_ok:
        return ""
    out = _render_segment(
        p.via_type, p.via_number, p.via_letters, p.via_suffix, p.via_suffix_letters,
        p.via_bis, p.via_quadrant, style,
    )
    cross = _render_segment(
        p.cross_type, p.cross_number, p.cross_letters, p.cross_suffix,
        p.cross_suffix_letters, p.cross_bis, p.cross_quadrant, style,
    )
    if p.had_hash or cross or p.plate is not None:
        out += " #"
        if cross:
            out += f" {cross}"
        plate = p.plate
        if plate is not None:
            if plate.isdigit():
                plate = f"{int(plate):02d}"
            out += f" - {plate}"
        elif p.had_plate_sep:
            out += " -"
    if with_complement and p.complement:
        out += f" {p.complement}"
    return re.sub(r"\s+", " ", out).strip()


_COMPLEMENT_KINDS = frozenset(COMPLEMENT_CANON.values())


def _kind_key(token: str) -> str:
    """Comparable key of a complement token: accent-free, upper, dot-free, alias-resolved."""
    tok = strip_accents(token).upper().strip(".")
    return COMPLEMENT_CANON.get(tok, tok)


def _value_key(token: str) -> str:
    """Comparable key of a complement value token: upper, punctuation-free, no leading zeros.

    Punctuation splits sub-tokens (``101-A`` -> ``101`` + ``A``); each digit-only
    sub-token loses its leading zeros. Sub-tokens are then concatenated, so
    ``101-A``, ``101 A`` and ``101A`` all compare equal.
    """
    parts = re.findall(r"[A-Z0-9]+", strip_accents(token).upper())
    return "".join((p.lstrip("0") or "0") if p.isdigit() else p for p in parts)


def _complement_chunks(text: str) -> list[tuple[list[str], str, str]]:
    """Split complement text into ``(original tokens, kind key, value key)`` chunks.

    A chunk starts at every registered kind token; a leading chunk before any
    registered kind (``BO 002003``) is keyed by its first token.
    """
    chunks: list[list[str]] = []
    for tok in text.split():
        if not chunks or _kind_key(tok) in _COMPLEMENT_KINDS:
            chunks.append([tok])
        else:
            chunks[-1].append(tok)
    return [
        (c, _kind_key(c[0]), "".join(_value_key(t) for t in c[1:]))
        for c in chunks
    ]


def filter_complement(cadastral_address: str, input_complement: str | None) -> str:
    """Drop cadastral complement chunks (AP 101, TO 2, ...) the input did not mention.

    The cadastral complement is split into chunks that start at each registered kind
    (``BLQ G CA 7`` -> ``BLQ G`` + ``CA 7``); a leading chunk before any registered
    kind (``BO 002003`` in ``BO 002003 LT 0031``) is keyed by its first token. A chunk
    is kept iff the input has a chunk of the same kind (aliases resolved through
    ``COMPLEMENT_CANON``) with the same non-empty value: values are compared
    as one key after uppercasing, dropping accents/punctuation and leading zeros
    (``0031`` == ``31``). A kind mentioned without a value never matches. The kept
    chunk is spelled as the cadastre has it. Never raises; an unparseable cadastral
    address or one without a complement is returned unchanged.
    """
    if cadastral_address is None:
        return ""
    p = parse_address(cadastral_address)
    if not p.parse_ok or not p.complement:
        return cadastral_address
    base = canonical(p, with_complement=False)
    if cadastral_address.startswith(base):
        comp_text = cadastral_address[len(base):]
    else:
        comp_text = p.complement

    mentioned = {
        (kind, values)
        for _, kind, values in _complement_chunks(input_complement or "")
        if values
    }
    kept = [
        " ".join(tokens)
        for tokens, kind, values in _complement_chunks(comp_text)
        if values and (kind, values) in mentioned
    ]
    return re.sub(r"\s+", " ", " ".join([base, *kept])).strip()


#: Kinds whose value is a free-text name (``ED LAS PALMAS``); every other kind
#: takes a short code (``TO F``, ``AP 101A``).
_NAME_KINDS = frozenset({"ED", "CONJ", "UR", "BR", "AGR", "SEC", "VDA", "CGTO"})
#: Locality kinds (barrio, vereda, corregimiento): recognised, so their names never
#: leak into another value (``TORRE HUDSON BARRIO LA FLORA``), but never emitted. They
#: are not building-unique, and emitting them would give one building a different
#: key depending on whether the barrio was typed.
_LOCALITY_KINDS = frozenset({"BR", "VDA", "CGTO"})
_MAX_NAME_TOKENS = 4
#: A value token longer than this is noise, not a unit code or name word.
_MAX_VALUE_TOKEN_LEN = 12
#: Code tokens (joined by a hyphen or a space) one unit value may have (``3-02-A``).
_MAX_VALUE_PARTS = 4

#: Canonical emission order of merged units: broad place, then building, then unit.
_KIND_ORDER = (
    "UR", "CONJ", "AGR", "SEC", "ET", "MZ", "ED", "BLQ", "TO",
    "INT", "CA", "LT", "PH", "PISO", "AP", "OF", "LC", "PQ", "GA", "BG",
)
_KIND_RANK = {kind: i + 1 for i, kind in enumerate(_KIND_ORDER)}


#: Words that are never a unit code nor a name, however short: connectors,
#: prepositions, adverbs of place and abbreviations (``CASA DE DOS PISOS``,
#: ``TORRE SOBRE LA AVENIDA``, ``BLOQUE DETRAS``, ``TORRE KM 5``).
_VALUE_STOPWORDS = frozenset({
    "Y", "O", "DE", "DEL", "LA", "LAS", "EL", "LOS", "LO", "LE", "UN", "UNA",
    "EN", "AL", "CON", "SIN", "POR", "PARA", "SU", "NO", "SI", "SE", "ES", "QUE",
    "SOBRE", "DESDE", "HASTA", "HACIA", "TRAS", "ATRAS", "DETRAS", "DELANTE", "ADELANTE",
    "LEJOS", "ENSEGUIDA", "ANTES", "DESPUES", "MAS", "MUY", "KM",
})
#: Articles that, right after ``<digits> A``, mark the ``A`` as a preposition
#: (``CASA 5 A LA ESQUINA``), not a unit letter.
_ARTICLES = frozenset({"LA", "LAS", "EL", "LOS", "UN", "UNA"})
#: Marker words written before a value (``TORRE N 3``, ``APTO NO 101``, ``TORRE LETRA A``).
#: Skipped when a value follows; never a name.
_MARKER_WORDS = frozenset({"N", "NO", "NRO", "NUM", "NUMERO", "NUMERAL", "LETRA"})
#: Plural count nouns: a bare number right before one counts something, it is not part
#: of a value (``CASA 17 2 PISOS`` is ``CA 17``).
_COUNT_NOUNS = frozenset({
    "PISOS", "NIVELES", "PLANTAS", "CASAS", "APTOS", "APARTAMENTOS", "LOCALES",
    "HABITACIONES", "ALCOBAS", "CUARTOS", "CUADRAS", "METROS", "MTS", "MINUTOS",
})


def _word_forms(*stems: str) -> set[str]:
    """Gender and number forms of Spanish adjectives/nouns ending in ``O``/``A``/other."""
    out: set[str] = set()
    for stem in stems:
        out.add(stem)
        if stem.endswith(("O", "A")):
            base = stem[:-1]
            out.update({base + "O", base + "A", base + "OS", base + "AS"})
        elif stem.endswith(("L", "R", "N", "D", "Z")):
            out.add(stem + "ES")
        else:
            out.add(stem + "S")
    return out


def _plurals(*nouns: str) -> set[str]:
    """Each noun and its plural (``PARQUE`` -> ``PARQUES``, ``TALLER`` -> ``TALLERES``)."""
    return {form for noun in nouns for form in (noun, noun + ("S" if noun[-1] in "AEIOU" else "ES"))}


#: Adjectives that qualify a building instead of naming it (position, role, size, age,
#: use, ordinals). Skipped before a building name and ending one: ``CONJUNTO
#: RESIDENCIAL LOS ALAMOS`` -> ``CONJ LOS ALAMOS``, ``ED LAS PALMAS ESQUINERO`` ->
#: ``ED LAS PALMAS``, ``CONJUNTO CERRADO`` -> no unit.
_QUALIFIER_WORDS: frozenset[str] = frozenset(_word_forms(
    # position / orientation
    "ESQUINERO", "FRONTAL", "LATERAL", "POSTERIOR", "TRASERO", "DELANTERO", "INTERNO",
    "EXTERNO", "CENTRAL", "CONTIGUO", "VECINO", "ORIENTAL", "OCCIDENTAL", "SEPTENTRIONAL",
    "MERIDIONAL", "IZQUIERDO", "DERECHO",
    # role / size / age
    "PRINCIPAL", "SECUNDARIO", "NUEVO", "VIEJO", "ANTIGUO", "GRANDE", "PEQUENO",
    "UNICO", "MAYOR", "MENOR", "ULTIMO", "MISMO", "OTRO",
    # use
    "COMERCIAL", "RESIDENCIAL", "MULTIFAMILIAR", "UNIFAMILIAR", "BIFAMILIAR",
    "FAMILIAR", "ADMINISTRATIVO", "MEDICO", "EMPRESARIAL", "INDUSTRIAL", "HABITACIONAL",
    # ordinals
    "PRIMERO", "PRIMER", "SEGUNDO", "TERCERO", "TERCER", "CUARTO", "QUINTO", "SEXTO",
    # access
    "CERRADO", "PRIVADO",
))

#: Closed-class address-descriptive vocabulary: never a tower name (``TORRE BLANCA``,
#: ``BLOQUE DETRAS DEL TALLER``), and not a building name on its own. It is the seed of
#: ``complement_lexicon.DESCRIPTIVE_WORDS`` (``scripts/build_complement_lexicons.py``)
#: and wins over a place-name word (``LA ESQUINA`` does not make ESQUINA a name).
DESCRIPTIVE_SEED: frozenset[str] = frozenset(
    _QUALIFIER_WORDS
    | _word_forms(
        # position nouns and adverbs
        "ESQUINA", "FRENTE", "ALTO", "BAJO", "ARRIBA", "ABAJO", "JUNTO", "DIAGONAL",
        "ENTRADA", "SALIDA", "COSTADO", "LADO",
        # use nouns
        "VIVIENDA", "LOCALES", "COMERCIO", "PORTERIA", "RECEPCION", "SOTANO", "SEMISOTANO",
        # glue words of free text
        "NUMERO", "DIRECCION", "SEGUN", "ENTRE", "CERCA", "DONDE", "AQUI", "FRONT",
        # colours, materials and free-text participles
        "BLANCO", "NEGRO", "ROJO", "AMARILLO", "ROSADO", "MORADO", "METALICO", "PEGADO",
    )
    | {
        "AZUL", "AZULES", "VERDE", "VERDES", "GRIS", "GRISES", "NARANJA", "NARANJAS", "CAFE",
        "BEIGE", "CREMA", "VIOLETA", "LILA", "CELESTE",
        "SIGUIENTE", "SIGUIENTES", "QUEDA", "QUEDAN",
        # abbreviations of free text (principal, esquina, police post, city bus system)
        "PPAL", "ESQ", "CAI", "MIO",
    }
    | _plurals(
        # landmarks and institutions people give as directions
        "PARQUE", "IGLESIA", "COLEGIO", "ESCUELA", "HOSPITAL", "CLINICA", "UNIVERSIDAD",
        "ESTACION", "TIENDA", "SUPERMERCADO", "PANADERIA", "DROGUERIA", "FARMACIA", "CANCHA",
        "CAPILLA", "PARROQUIA", "ALCALDIA", "MERCADO", "GALERIA", "TALLER", "NEGOCIO",
        "PORTON", "REJA", "PUERTA", "ANDEN", "SEMAFORO",
        # the property itself
        "PREDIO", "INMUEBLE",
    )
)
#: Spanish gerunds (``ENTRANDO``, ``SUBIENDO``, ``YENDO``): free-text verbs, never a name
#: unless attested as a place-name word (``FERNANDO``).
_GERUND_RE = re.compile(r"^[A-Z]{2,}(?:ANDO|IENDO|YENDO)$|^YENDO$")


_LONG_LETTER_RUN_RE = re.compile(r"[A-Z]{3,}")
#: Roman numerals up to 39 (``II``, ``VIII``, ``XII``): cadastral block/tower codes.
_ROMAN_RE = re.compile(r"^(?=[IVX])X{0,3}(?:IX|IV|V?I{0,3})$")

#: Kinds that identify one building of a complex (tower, block, stage). Their value
#: is a code or a *tower name* (``TO HUDSON``); see :func:`is_tower_name_word`.
_TOWER_KINDS = frozenset({"TO", "BLQ", "ET"})
#: Words a tower name may have (``TO SAN MARTIN``), not counting a leading article.
_MAX_TOWER_NAME_WORDS = 2
#: Number words read as digits after a tower kind only (``TORRE DOS`` == ``TORRE 2``).
_NUMBER_WORDS: dict[str, str] = {
    "UNO": "1", "DOS": "2", "TRES": "3", "CUATRO": "4", "CINCO": "5", "SEIS": "6",
    "SIETE": "7", "OCHO": "8", "NUEVE": "9", "DIEZ": "10", "ONCE": "11", "DOCE": "12",
}
#: Address grammar words that are never a tower name: via types, quadrants, ``BIS``
#: and single-word city noise (``TORRE NORTE``, ``TORRE CALLE``, ``TORRE CALI``).
_GRAMMAR_WORDS = frozenset(
    w for w in (*VIA_TYPE_CANON, *QUADRANT_CANON, *_QUADRANT_WORDS, *_BIS_WORDS, *CITY_NOISE)
    if w.isalpha() and len(w) >= 3
)
#: Kind spellings a near-miss is compared against (``TORE``, ``EDIFICO``, ``COJ``, ``PISOS``).
_KIND_TYPO_TARGETS = tuple(k for k in COMPLEMENT_CANON if k.isalpha() and len(k) >= 4)


def _is_code(token: str) -> bool:
    """Whether ``token`` can be the value of a code-valued unit (``F``, ``101``, ``2B``, ``AB``).

    Accepted: a token with a digit whose letter runs are at most two long (``501A``,
    ``2AB``, ``A1``), a single letter, a two-letter code, or a roman numeral (``III``,
    ``VIII``). Rejected: registered kind words, connectors (``DE``, ``Y``...),
    descriptive words (three or more letters, no digit: ``ESQUINERA``, ``COMERCIAL``,
    ``FRONT``) and digits glued to a word (``501TORRED``, ``5B1TORRE``).
    """
    if token in COMPLEMENT_CANON or len(token) > _MAX_VALUE_TOKEN_LEN:
        return False
    if any(ch.isdigit() for ch in token):
        return _LONG_LETTER_RUN_RE.search(token) is None
    if token in _VALUE_STOPWORDS:
        return False
    return len(token) <= 2 or _ROMAN_RE.match(token) is not None


def _is_name_shaped(word: str) -> bool:
    """Hard, lexicon-free part of the tower-name test (upper-case, accent-free ``word``).

    A name word is 3 to ``_MAX_VALUE_TOKEN_LEN`` ASCII letters and is not a registered
    kind, a connector, a number word, a roman numeral, an address grammar word, or a
    one-edit misspelling of a kind word (``TORE``, ``TORRRE``, ``EDIFICO``, ``COJUNTO``,
    ``COJ``, ``PISOS``, ``CASAS``: observed in the address corpora as mistyped kinds,
    and a mistyped kind read as a name would invent a tower). Shared with
    ``scripts/build_complement_lexicons.py`` so the lexicons never list such words.
    """
    if not (3 <= len(word) <= _MAX_VALUE_TOKEN_LEN) or not (word.isascii() and word.isalpha()):
        return False
    if (
        word in COMPLEMENT_CANON or word in _VALUE_STOPWORDS or word in _NUMBER_WORDS
        or word in _MARKER_WORDS or word in _GRAMMAR_WORDS or _ROMAN_RE.match(word)
    ):
        return False
    return not any(
        abs(len(word) - len(kind)) <= 1 and _levenshtein(word, kind) <= 1 for kind in _KIND_TYPO_TARGETS
    )


@lru_cache(maxsize=8192)
def is_tower_name_word(token: str) -> bool:
    """Whether ``token`` can be (part of) the name of a tower or block (``HUDSON``).

    Decision, after the lexicon-free shape test (:func:`_is_name_shaped`):

    1. in :data:`DESCRIPTIVE_SEED` (closed-class descriptive vocabulary: ``ESQUINERA``,
       ``PRINCIPAL``, ``BLANCA``, ``PARQUE``, ``PPAL``...) -> not a name;
    2. attested in ``complement_lexicon.TOWER_NAME_WORDS`` (place names of the IDESC
       basemaps) -> name;
    3. listed in ``complement_lexicon.DESCRIPTIVE_WORDS`` (ordinary words of the
       cadastral address text plus the seed) -> not a name;
    4. an unseen gerund (``ENTRANDO``, ``SUBIENDO``) -> not a name;
    5. otherwise (unseen word) -> name.

    Unseen words are accepted on purpose. Mined complements show that a word after
    ``TORRE``/``BLOQUE`` is a proper name far more often than a description (the
    cadastre has no descriptive value after TO/BLQ at all; the survey's two word
    values are the tower names HUDSON and TAMESIS), and the two errors are not
    symmetric downstream: a rejected name merges two towers of one complex into one
    building key (labels silently lost), while an accepted description only splits
    one building into two keys. The lexicons are generated (no hand-picked names) by
    ``scripts/build_complement_lexicons.py`` and are disjoint (the seed is removed from
    the place-name set), so the rules never disagree. Case and accents are ignored.
    Never raises.
    """
    word = strip_accents(str(token)).upper()
    if not _is_name_shaped(word) or word in DESCRIPTIVE_SEED:
        return False
    if word in _lexicon.TOWER_NAME_WORDS:
        return True
    return word not in _lexicon.DESCRIPTIVE_WORDS and _GERUND_RE.match(word) is None


def _is_proper_word(word: str) -> bool:
    """Whether a word of a building name carries the name (``PALMAS``, ``PACIFICO``),
    as opposed to a kind word, connector, marker or descriptive word."""
    return not (
        word in COMPLEMENT_CANON or word in _VALUE_STOPWORDS or word in _MARKER_WORDS
        or word in DESCRIPTIVE_SEED or word in _COUNT_NOUNS or word in _NUMBER_WORDS
        or (_GERUND_RE.match(word) is not None and word not in _lexicon.TOWER_NAME_WORDS)
    )


#: Connectors a building name may contain (``TORRES DE COMFANDI``, ``VILLA DEL LAGO``);
#: any other stopword (``SOBRE``, ``KM``...) ends the name.
_NAME_CONNECTORS = frozenset({"DE", "DEL", "LA", "LAS", "EL", "LOS", "Y"})


def _name_value(words: list[str]) -> str | None:
    """Value of a name-kind chunk (``ED``, ``CONJ``, ``SEC``...), or ``None`` without a name.

    Leading qualifiers are skipped (``CONJUNTO RESIDENCIAL LOS ALAMOS`` -> ``LOS
    ALAMOS``); a leading connector means there is no name (``EDIFICIO DE TRES PISOS``,
    ``SECTOR DE LA IGLESIA``), and so does a leading unit kind word (``EDIFICIO PISO
    HUDSON``). The name has at most ``_MAX_NAME_TOKENS`` words, stops at the first
    descriptive word after its first word (``ED LAS PALMAS ESQUINERO`` -> ``LAS
    PALMAS``), drops trailing connectors and kind words, and must have at least one proper word
    (``CONJUNTO CERRADO``, ``EDIFICIO LA ESQUINA`` -> no unit). Kind words inside a name
    are kept (``EDIFICIO TORRE PACIFICO`` -> ``TORRE PACIFICO``).
    """
    i = 0
    while i < len(words) and words[i] in _QUALIFIER_WORDS:
        i += 1
    start = i
    if i < len(words) and words[i] in _ARTICLES:
        i += 1
    if i >= len(words) or words[i] in _VALUE_STOPWORDS:
        return None
    if words[i] in COMPLEMENT_CANON and COMPLEMENT_CANON[words[i]] not in _TOWER_KINDS:
        return None  # ``EDIFICIO PISO HUDSON``: a unit kind cannot start a name
    picked = words[start:i + 1]
    for word in words[i + 1:]:
        if len(picked) >= _MAX_NAME_TOKENS or not (_is_proper_word(word) or word in _NAME_CONNECTORS
                                                   or word in COMPLEMENT_CANON):
            break
        picked.append(word)
    while picked and (picked[-1] in _VALUE_STOPWORDS or picked[-1] in COMPLEMENT_CANON):
        picked.pop()
    if not any(_is_proper_word(w) for w in picked) or any(len(w) > _MAX_VALUE_TOKEN_LEN for w in picked):
        return None
    return " ".join(picked)


def _tower_value(tokens: list[str]) -> str | None:
    """Value of a tower/block/stage chunk that does not start with a code.

    A number word becomes its digits (``DOS`` -> ``2``). Otherwise an optional leading
    article plus one or two name words (``HUDSON``, ``SAN MARTIN``, ``LAS PALMAS``); the
    name stops at the first word that fails :func:`is_tower_name_word`, so trailing
    free text (``TORRE HUDSON FRENTE AL PARQUE``) is not absorbed. ``None`` when the
    chunk holds no name (``TORRE ESQUINERA``, ``TO DE LA ESQUINA``).
    """
    if not tokens:
        return None
    if tokens[0] in _NUMBER_WORDS:
        return _NUMBER_WORDS[tokens[0]]
    start = 1 if tokens[0] in _ARTICLES and len(tokens) > 1 else 0
    words: list[str] = []
    for tok in tokens[start:start + _MAX_TOWER_NAME_WORDS]:
        if not is_tower_name_word(tok):
            break
        words.append(tok)
    if not words:
        return None
    return " ".join([*tokens[:start], *words])


def _render_value(parts: list[str]) -> str:
    """Canonical spelling of a multi-part code value (see :func:`extract_complement_units`).

    The parts are split into digit and letter runs; runs of different class are glued
    (``101`` + ``A`` -> ``101A``, ``F`` + ``2`` -> ``F2``) and runs of the same class are
    joined with a hyphen (``3`` + ``02`` -> ``3-02``, ``A`` + ``B`` -> ``A-B``). The first
    digit run loses its leading zeros (``0101`` -> ``101``); later runs keep them, so
    ``3-02`` and ``3-2`` stay apart. Distinct run sequences give distinct strings, and
    the result re-extracts to itself.
    """
    runs = [run for part in parts for run in re.findall(r"\d+|[A-Z]+", part)]
    out: list[str] = []
    for k, run in enumerate(runs):
        if k == 0 and run.isdigit():
            run = run.lstrip("0") or "0"
        elif k and run[0].isdigit() == runs[k - 1][0].isdigit():
            out.append("-")
        out.append(run)
    return "".join(out)


def _continues_value(prev: str, tok: str, glued: bool, after: str | None) -> bool:
    """Whether ``tok`` extends a code value that so far ends with the part ``prev``.

    ``glued`` means a hyphen joined it to ``prev`` (``3-02``, ``F-2``, ``101-A``): any code
    extends. A space-separated part extends only when it is a bare number after a number
    or a single letter (``3 02``, ``F 2``), unless the word ``after`` it shows it counts
    something (``CASA 17 2 PISOS``); or a single unit letter after a number (``101 A``),
    unless it is a connector or the preposition of ``5 A LA ESQUINA``.
    """
    if glued:
        return _is_code(tok)
    if tok.isdigit():
        return (prev[-1].isdigit() or (len(prev) == 1 and prev.isalpha())) and after not in _COUNT_NOUNS
    if len(tok) == 1 and tok.isalpha():
        return (
            prev[-1].isdigit() and tok not in _VALUE_STOPWORDS
            and not (tok == "A" and after in _ARTICLES)
        )
    return False


def _unit_value(tokens: list[tuple[str, bool]], kind: str) -> str | None:
    """Value of one unit chunk, by kind. ``tokens`` are ``(word, glued_by_hyphen)``.

    * name kinds (``ED``, ``CONJ``, ``SEC``...): a building name (:func:`_name_value`);
    * tower kinds (``TO``, ``BLQ``, ``ET``): a code, else a number word or a tower
      name (:func:`_tower_value`);
    * every other kind (``AP``, ``CA``, ``LC``, ``OF``, ``PISO``, ``INT``, ``LT``,
      ``MZ``...): a code only.

    A code value is one code token plus the parts that continue it
    (:func:`_continues_value`), at most ``_MAX_VALUE_PARTS``, spelled by
    :func:`_render_value`. A leading marker word is skipped when a value follows it
    (``TORRE N 3`` -> ``3``, ``TORRE LETRA A`` -> ``A``), and after a tower kind a number
    word is read as its digits and keeps a following letter (``TORRE UNO A`` -> ``1A``).
    """
    if not tokens:
        return None
    if kind in _NAME_KINDS:
        return _name_value([t for t, _ in tokens])
    tokens = list(tokens)
    while (
        len(tokens) > 1 and tokens[0][0] in _MARKER_WORDS
        and (_is_code(tokens[1][0]) or tokens[1][0] in _NUMBER_WORDS)
    ):
        tokens = tokens[1:]
    if kind in _TOWER_KINDS and tokens[0][0] in _NUMBER_WORDS:
        tokens[0] = (_NUMBER_WORDS[tokens[0][0]], tokens[0][1])
    first = tokens[0][0]
    # ``TORRE O 2``: ``O`` is a connector elsewhere, but right after a tower kind it is a tower
    # letter, and it takes the parts that continue a value like any other (``TORRE B 2`` -> ``B2``).
    if not _is_code(first) and not (first == "O" and kind in _TOWER_KINDS):
        if kind in _TOWER_KINDS:
            return _tower_value([t for t, _ in tokens])
        return None  # descriptive text (``CASA ESQUINERA``), not a unit
    picked = [first]
    for j in range(1, min(len(tokens), _MAX_VALUE_PARTS)):
        tok, glued = tokens[j]
        after = tokens[j + 1][0] if j + 1 < len(tokens) else None
        if not _continues_value(picked[-1], tok, glued, after):
            break
        picked.append(tok)
    if any(len(t) > _MAX_VALUE_TOKEN_LEN for t in picked):
        return None
    return _render_value(picked)


def _opens_unit(tokens: list[str], i: int, kind: str, name_so_far: list) -> bool:
    """Whether the kind word ``tokens[i]``, met inside a name chunk, starts a new unit.

    A locality kind always does (``CONJ DIVENTO BARRIO LA FLORA``), and so does another
    name kind once the name has words. Otherwise it does when a code follows (``ED
    TORRE F``). A tower kind followed by a tower name starts a unit only once the name
    already has words (``ED LAS PALMAS TORRE HUDSON``); right after the name kind it is
    part of the name (``EDIFICIO TORRE PACIFICO``).
    """
    if kind in _LOCALITY_KINDS or (kind in _NAME_KINDS and name_so_far):
        return True
    if i + 1 >= len(tokens):
        return False
    if _is_code(tokens[i + 1]):
        return True
    return (
        kind in _TOWER_KINDS and bool(name_so_far)
        and _tower_value(tokens[i + 1:i + 2 + _MAX_TOWER_NAME_WORDS]) is not None
    )


#: A hyphen between two alphanumerics, with optional spaces (``3 - 02``).
_VALUE_HYPHEN_RE = re.compile(r"(?<=[A-Z0-9])\s*-\s*(?=[A-Z0-9])")


def _unit_tokens(text: str) -> list[tuple[str, bool]]:
    """``(word, glued_by_hyphen)`` tokens of a complement text (accent-free, upper-case)."""
    s = _VALUE_HYPHEN_RE.sub("-", _units_text(text))
    return [
        (part, j > 0)
        for group in re.findall(r"[A-Z0-9]+(?:-[A-Z0-9]+)*", s)
        for j, part in enumerate(group.split("-"))
    ]


def extract_complement_units(text: str | None) -> list[tuple[str, str]]:
    """Building-distinguishing units of a complement text as ``[(kind, value)]``.

    Kinds are canonical (``TORRE``/``TOR``/``T-`` -> ``TO``); values are upper-case and
    accent-free. Which values a kind takes is kind-aware (see :func:`_unit_value`):
    ``TORRE HUDSON`` is ``("TO", "HUDSON")`` but ``CASA HUDSON`` and ``TORRE ESQUINERA``
    are not units. A code value keeps all its parts in one canonical spelling
    (:func:`_render_value`): ``APTO 3-02``, ``APTO 3 02`` and ``AP 3 - 02`` are all
    ``3-02`` (never ``3``, which would merge it with ``3-03``); ``101A``, ``101 A`` and
    ``101-A`` are all ``101A``; ``TORRE F-2`` is ``F2``. A kind without a value
    (``TORRE``) is not a unit; words that are not a registered kind are never turned
    into units; locality kinds (barrio, vereda, corregimiento) are recognised but not
    returned; exact duplicates collapse. The same kind with different values is
    reported as is (the caller decides). Deterministic; never raises.
    """
    if not text:
        return []
    tokens = _unit_tokens(text)
    words = [t for t, _ in tokens]
    chunks: list[tuple[str, list[tuple[str, bool]]]] = []
    for i, (tok, glued) in enumerate(tokens):
        kind = COMPLEMENT_CANON.get(tok)
        if kind is None:
            if chunks:
                chunks[-1][1].append((tok, glued))
        elif chunks and chunks[-1][0] in _NAME_KINDS and not _opens_unit(words, i, kind, chunks[-1][1]):
            chunks[-1][1].append((tok, glued))  # a kind word inside a name (``ED TORRE DE CALI``)
        else:
            chunks.append((kind, []))
    units: list[tuple[str, str]] = []
    for kind, values in chunks:
        if kind in _LOCALITY_KINDS:
            continue
        value = _unit_value(values, kind)
        if value is not None and (kind, value) not in units:
            units.append((kind, value))
    return units


def _render_units(units: list[tuple[str, str]]) -> str | None:
    """``KIND VALUE`` pieces, first value per kind, in the fixed kind order."""
    pieces: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for kind, value in units:
        if kind in seen:
            continue
        seen.add(kind)
        pieces.append((_KIND_RANK.get(kind, 0), len(pieces), f"{kind} {value}"))
    return " ".join(p[2] for p in sorted(pieces)) or None


def render_complement(text: str | None) -> str | None:
    """Canonical unit text of a complement (``TO F AP 101``) or ``None`` when it has no unit.

    The units of :func:`extract_complement_units` with the first value per kind and the
    fixed kind order. It is exactly the complement :func:`merge_complement` emits for
    the same input, so OK and non-OK rows read alike. Free text around the units is
    dropped. Idempotent; never raises.
    """
    return _render_units(extract_complement_units(text))


@dataclass(frozen=True)
class ComplementMerge:
    """Result of :func:`merge_complement`."""

    address: str
    complement: str | None = None
    #: Kinds where the input disagrees with the cadastre (or with itself).
    conflicts: tuple[str, ...] = ()


def merge_complement(cadastral_address: str | None, input_complement: str | None) -> ComplementMerge:
    """Merge the input's building details into the cadastral address.

    The base (``<via> # <cross> - <plate>``) always comes from the cadastre. The
    complement is the *input's* canonical unit text (:func:`render_complement`): units
    the input states are emitted even if the cadastre lacks them (TORRE F vs TORRE G),
    cadastral-only units are never invented, and a unit both have is spelled the same
    canonical way whatever the cadastre wrote (``AP 0101`` -> ``AP 101``), so a matched
    row reads exactly like an unmatched one.

    Conflicts compare extracted unit values, never raw text: the input disagrees with
    the cadastre on a kind when the cadastre has that kind with other values only
    (``TORRE F FRENTE AL PARQUE`` agrees with cadastral ``TO F``); the same kind twice
    in the input with different values keeps the first and is reported too. An
    unparseable cadastral address is returned unchanged. Never raises.
    """
    if cadastral_address is None:
        return ComplementMerge("")
    p = parse_address(cadastral_address)
    if not p.parse_ok:
        return ComplementMerge(cadastral_address)

    cadastral_values: dict[str, set[str]] = {}
    if p.complement:
        base = canonical(p, with_complement=False)
        comp_text = (
            cadastral_address[len(base):] if cadastral_address.startswith(base)
            else (p.complement_text or p.complement)  # as written, not the scoring spelling
        )
        for kind, value in extract_complement_units(comp_text):
            cadastral_values.setdefault(kind, set()).add(value)
    else:
        base = cadastral_address.strip()

    units = extract_complement_units(input_complement)
    conflicts: list[str] = []
    first_value: dict[str, str] = {}
    for kind, value in units:
        if kind in first_value:
            conflict = value != first_value[kind]  # same kind, another value: first one wins
        else:
            first_value[kind] = value
            conflict = kind in cadastral_values and value not in cadastral_values[kind]
        if conflict and kind not in conflicts:
            conflicts.append(kind)

    complement = _render_units(units)
    address = re.sub(r"\s+", " ", f"{base} {complement}" if complement else base).strip()
    return ComplementMerge(address, complement, tuple(conflicts))


def match_key(text: str) -> str:
    """Whitespace/punctuation-insensitive key used for exact-match comparisons."""
    return re.sub(r"[^A-Z0-9]", "", strip_accents(text).upper())


def canonicalize(raw: str | None, style: str = "spaced", with_complement: bool = True) -> str:
    """Convenience wrapper: free-form text -> canonical string (``''`` on failure)."""
    return canonical(parse_address(raw), style=style, with_complement=with_complement)


def parse_many(values: Iterable[str | None]) -> list[ParsedAddress]:
    return [parse_address(v) for v in values]
