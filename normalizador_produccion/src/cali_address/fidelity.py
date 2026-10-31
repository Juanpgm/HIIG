"""Fidelity of a normalized address with respect to its raw input (semantic drift / corruption risk).

``assess_fidelity(raw, output)`` answers one question: did the pipeline hand back a DIFFERENT address
than the one it was given? It does not judge the characters (abbreviations, spacing, ``#``/``-``
insertion, case, accents, ``No.``/``N°`` -> ``#``, ``N`` -> ``NORTE`` in the same segment, city suffix
removal and leading zeros are all HARMLESS) but the door-identifying data: via type, via / cross /
plate numbers, the letters glued to them and the quadrant (position significant).

It is deliberately independent of the production parser (that is the point: it must be able to catch
parser bugs), of the cadastre and of any coordinate. It has its own small regex tokenizer and works on
two strings only.

Method. Both strings are tokenized into atoms (digit groups, letters, quadrants, ``BIS``, via types,
unit keywords, free words). Letters glued to digits are split into atoms (``66C`` == ``66 C``;
``15BNORTE`` == ``15 B NORTE``). The *door* (first via type up to the first unit keyword / free word /
``;``) of the raw string is compared with the door of the output item by item. Everything outside the
door is the *tail* (units, place text) and is judged with lower severity.

Severity (constants below, tunable):

* ``high``   - any change / loss / reorder / invention of door numbers, door letters, quadrant, via type;
  a glued token split by a ``#``/``-`` the input did not have; output with no input behind it.
* ``medium`` - unit digits (APTO 301) dropped, digits invented in the tail, a large share (>= 30 %) of the
  raw significant tokens missing, second address of a multi-address input dropped, ``#`` moved.
* ``low``    - unit relabelled, annotation digits (comuna 3, etapa 2) or a few tokens dropped, zero padding.
* ``none``   - nothing changed beyond harmless normalization, or no output at all (flag ``no_address``).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Iterable, NamedTuple

#: Severity of every flag. ``no_address`` is informational (nothing was generated, so nothing is corrupted).
SEVERITY: dict[str, str] = {
    **dict.fromkeys(
        (
            "via_type_changed", "via_type_dropped",
            "door_digits_dropped", "door_digits_added", "door_digits_changed", "door_digits_reordered",
            "via_number_changed", "cross_number_changed", "plate_changed", "scrambled_order",
            "letters_changed", "letters_dropped", "letters_added", "letters_moved",
            "quadrant_added", "quadrant_dropped", "quadrant_changed", "quadrant_moved",
            "glued_split", "output_from_nothing",
        ),
        "high",
    ),
    **dict.fromkeys(
        (
            "unit_digits_dropped", "tail_digits_added", "text_dropped_major", "multi_address_truncated",
            "separator_moved", "prefix_digits_dropped", "via_type_guessed", "trailing_number_dropped",
        ),
        "medium",
    ),
    **dict.fromkeys(("unit_relabelled", "annotation_digits_dropped", "text_dropped", "zero_padding", "cross_type_added", "cross_type_dropped",
     "duplicate_dropped"), "low"),
    "no_address": "none",
}

#: Share of the raw significant tokens missing from the output that turns ``text_dropped`` into medium.
TEXT_DROP_MEDIUM_SHARE = 0.30
#: A door holds at most this many digit groups (via, cross and plate, plus glued ``1A9``-style pieces).
MAX_DOOR_NUMBERS = 5
#: Score of a report whose worst flag has this severity (minus ``_SCORE_STEP`` per extra flag).
_SCORE_BASE = {"none": 1.0, "low": 0.95, "medium": 0.6, "high": 0.3}
_SCORE_STEP = {"none": 0.0, "low": 0.01, "medium": 0.04, "high": 0.04}
_SCORE_FLOOR = {"none": 1.0, "low": 0.9, "medium": 0.4, "high": 0.1}
_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_MAX_CHARS = 20000


class FidelityReport(NamedTuple):
    """``risk`` in none/low/medium/high, ``flags`` (worst first), ``score`` in [0, 1] (1 = faithful)."""

    risk: str
    flags: tuple
    score: float
    details: dict


# ---------------------------------------------------------------------------
# vocabulary (own tables: nothing is shared with the production parser)
# ---------------------------------------------------------------------------
def _table(kind: str, groups: dict) -> dict:
    return {word: (kind, canon) for canon, words in groups.items() for word in words.split()}


_VIA = _table("V", {
    "CL": "CALLE CLL CL CLLE CALL CAL CLE CALLA",
    "KR": "CARRERA CRA KRA KR CR CARR CRR KRR CRRA KARRERA KRRA CARRA CARRRA CARERA CRRERA",
    "AV": "AVENIDA AV AVDA AVE AVN AVEN AVNDA",
    "AC": "AC", "AK": "AK",
    "DG": "DIAGONAL DG DIAG DGL DIAGO",
    "TV": "TRANSVERSAL TV TRANS TRANSV TRV TRAV TRANSVERSA TRNSV",
    "CIR": "CIRCULAR CIR CIRC", "CQ": "CIRCUNVALAR CQ CIRCUN",
    "PJ": "PASAJE PJ PSJ", "VIA": "VIA", "AU": "AUTOPISTA AU AUT",
})
#: Single letters that are a via type only as the very first token of the string (``K 29A 17-84``).
_VIA_FIRST_LETTER = {"C": "CL", "K": "KR", "D": "DG", "T": "TV"}
_QUADRANT = _table("Q", {
    "N": "N NTE NORTE NORT NOR", "S": "S SUR", "O": "O OE OES OSTE OESTE OCCIDENTE W", "E": "E ESTE ORIENTE",
})
#: Units whose digits identify a door (dropping them is medium; floor / parking / storage are weak) ...
_UNIT_STRONG = _table("U", {
    "AP": "APTO APT APARTAMENTO APARTA AP APTOS", "TO": "TORRE TO TORR TORRES TR",
    "LC": "LOCAL LC LOC", "CS": "CASA CS CA", "BL": "BLOQUE BL BLQ BQ",
    "IN": "INTERIOR INT", "OF": "OFICINA OF OFC OFI",
    "MZ": "MANZANA MZ MZA MANZ", "LT": "LOTE LT",
})
#: ... and units that only describe the place (dropping their digits is low).
_UNIT_WEAK = _table("U", {
    "ED": "EDIFICIO ED EDF EDIF", "CJ": "CONJUNTO CJ CONJ COJ", "UN": "UNIDAD UN UND",
    "ET": "ETAPA ET", "SC": "SECTOR SEC SECT", "AG": "AGRUPACION AGR AGRUP",
    "UR": "URBANIZACION URB", "BR": "BARRIO BR", "CD": "CIUDADELA CDLA",
    "PI": "PISO PI", "BG": "BODEGA BG BOD", "PQ": "PARQUEADERO PQ PARQ", "DP": "DEPOSITO DP",
})
_STRONG_CANON = {canon for _kind, canon in _UNIT_STRONG.values()}
_KEYWORDS = {**_VIA, **_QUADRANT, **_UNIT_STRONG, **_UNIT_WEAK, "BIS": ("B", "BIS")}
#: Words that carry no address information (city suffix, placeholders, connectors).
_NOISE = frozenset(
    "CALI VALLE DEL CAUCA COLOMBIA DE LA LAS LOS EL Y EN CON SIN ESPECIFICAR DIRECCION DIR CIUDAD MUNICIPIO "
    "DEPARTAMENTO NA NINGUNA NINGUNO NO APLICA ESQUINA ESQ FRENTE COMUNA CORREGIMIENTO URBANO RURAL SANTIAGO "
    "COL CO".split()
)
#: Spelled-out numbers the output renders as digits in the unit tail (``TORRE UNO A`` -> ``TO 1A``).
_SPELLED = {
    "UNO": "1", "PRIMER": "1", "PRIMERO": "1", "PRIMERA": "1", "DOS": "2", "SEGUNDO": "2", "SEGUNDA": "2",
    "TRES": "3", "TERCER": "3", "TERCERO": "3", "TERCERA": "3", "CUATRO": "4", "CUARTO": "4", "CINCO": "5",
    "QUINTO": "5", "SEIS": "6", "SEXTO": "6", "SIETE": "7", "OCHO": "8", "NUEVE": "9", "DIEZ": "10",
}
_ORDINALS = frozenset("ERA ERO RA DA ER NDA NDO TH ST ND RD TA MA VA AVA".split())
_FUZZY_WORDS = [w for w in _KEYWORDS if len(w) >= 5 and w not in ("BIS",)]
_FUZZY_QUADRANT = [w for w in _QUADRANT if len(w) >= 5]
#: letter + optional BIS + optional quadrant, glued after a digit (``5A``, ``5AN``, ``15BNORTE``, ``2HN``).
_SUFFIX_RE = re.compile(r"([A-Z])?(BIS)?(NORTE|NTE|SUR|OESTE|ESTE|N|S|O|E)?")
_ATOM_RE = re.compile(r"\d+|[A-Z]+|[#;\-]")
_NUMBER_SIGN_RE = re.compile(r"(?<![A-Z0-9])(?:N°|NO|NRO|NUM|NUMERO)\s*[.,:°]*\s*(?=[A-Z]?\d)")
_DOTTED_N_RE = re.compile(r"(?<![A-Z0-9])N\.\s*(?=\d)")
#: Coordinates / long numbers are not door digits: a ``//`` tail that starts with a decimal number, any decimal number
#: with 3+ decimals (``1,2345678``, ``-70.98765``) and any run of 7+ digits (phones, ids).
_GLUED_N_SIGN_RE = re.compile(r"(?<=\d)([A-Z]?)N\s*°\s*\.?\s*(?=\d)")
_COORD_TAIL_RE = re.compile(r"//\s*-?\d+[.,]\d+.*$", re.S)
_COORD_NUMBER_RE = re.compile(r"(?<![\w.,])-?\d+[.,]\d{3,}(?!\d)|\d{7,}")


class _Atom(NamedTuple):
    kind: str        # D digits, L letter, Q quadrant, B bis, V via, U unit, W word, '#', '-', ';'
    value: str       # canonical value (digits without leading zeros, canonical via/unit/quadrant)
    text: str        # as written (after case/accent folding)
    glued: bool      # starts exactly where the previous atom ended


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def _as_text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    return value if isinstance(value, str) else ""


def _number_sign(text: str, match: re.Match) -> str:
    """Replacement for ``N.``: a number sign only before the first ``#`` (afterwards it is a quadrant)."""
    return " N " if "#" in text[:match.start()] else " # "


def _fold(text: str) -> str:
    text = "".join(c for c in text[:_MAX_CHARS] if unicodedata.category(c) != "Cf")
    text = text.replace("Â°", "°").replace("Âº", "º")     # ``N°`` double-encoded as ``NÂ°``
    text = re.sub(r"(?i)(N)\s*[ºª]", r"\1°", text).replace("º", " ").replace("ª", " ")     # ordinal marks are not letters
    text = _COORD_TAIL_RE.sub(" ", text)
    text = _COORD_NUMBER_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).replace("Ñ", "NH").replace("ñ", "nh")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)).upper()
    text = re.sub(r"(?<![A-Z0-9])N�\s*(?=[A-Z]?\d)", " # ", text)   # mojibake of ``Nº``
    text = text.replace("�", " ").replace("_", "-")
    text = re.sub(r"(?<![A-Z0-9])Y/O(?![A-Z0-9])", ";", text)
    text = _GLUED_N_SIGN_RE.sub(lambda m: f" {m.group(1)} # ", text)     # ``48AN°90``: letter + number sign
    text = _NUMBER_SIGN_RE.sub(" # ", text)
    text = _DOTTED_N_RE.sub(lambda m: _number_sign(text, m), text)
    return re.sub(r"[^A-Z0-9#;\-]", " ", text)


def _word_atoms(run: str, glued: bool, after_digit: bool, first: bool, nxt_digit: bool, spaced_ok: bool) -> list[tuple]:
    """Atoms (kind, value) of one letter run."""
    if first and run in _VIA_FIRST_LETTER and nxt_digit:
        return [("V", _VIA_FIRST_LETTER[run])]
    if run in _KEYWORDS and not (run == "CA" and after_digit and glued):    # ``08CA`` is letters C, A
        kind, canon = _KEYWORDS[run]
        return [(kind, canon)]
    if after_digit and glued and run in _ORDINALS:
        return []
    if after_digit and (glued or run not in _NOISE) and len(run) <= 6 and _SUFFIX_RE.fullmatch(run):
        m = _SUFFIX_RE.fullmatch(run)
        out = []
        if m.group(1):
            out.append(("Q", _QUADRANT[m.group(1)][1]) if m.group(1) in _QUADRANT else ("L", m.group(1)))
        if m.group(2):
            out.append(("B", "BIS"))
        if m.group(3):
            out.append(("Q", _QUADRANT[m.group(3)][1]))
        return out
    if len(run) == 1:
        return [("L", run)]
    if after_digit and (glued or spaced_ok) and run not in _NOISE and len(run) == 2:       # two letters after a number (``08CA``, ``44 AM``)
        return [_QUADRANT.get(c, ("L", c)) for c in run]
    if len(run) >= 5:
        close = difflib.get_close_matches(run, _FUZZY_QUADRANT, n=1, cutoff=0.8)
        if close and close[0][0] == run[0]:
            return [_QUADRANT[close[0]]]
        close = difflib.get_close_matches(run, _FUZZY_WORDS, n=1, cutoff=0.84)
        if close:
            return [_KEYWORDS[close[0]]]
    return [("W", run)]


def _stray_semicolon(atoms: list, matches: list, idx: int) -> bool:
    """A ``;`` glued to a number sign / dash (``#;58-65``, ``- ;5``) or repeated is typing noise, not a separator."""
    nxt = matches[idx + 1].group() if idx + 1 < len(matches) else ""
    return bool(atoms) and atoms[-1].kind in ("#", "-", ";") or nxt in ("#", "-")


def _atomize(text: str) -> list[_Atom]:
    atoms: list[_Atom] = []
    matches = list(_ATOM_RE.finditer(_fold(text)))
    prev_end = -1
    for idx, m in enumerate(matches):
        tok = m.group()
        glued = m.start() == prev_end
        prev_end = m.end()
        if tok.isdigit():
            atoms.append(_Atom("D", str(int(tok)), tok, glued))
        elif tok in "#;-":
            if tok == ";" and _stray_semicolon(atoms, matches, idx):
                continue
            atoms.append(_Atom(tok, tok, tok, glued))
        else:
            after_digit = bool(atoms) and atoms[-1].kind == "D"
            nxt = matches[idx + 1] if idx + 1 < len(matches) else None
            nxt_digit = nxt is not None and nxt.group().isdigit()
            starts_address = not atoms or atoms[-1].kind == ";" or (atoms[-1].kind in "WL" and atoms[-1].value == "Y")
            for n, (kind, value) in enumerate(_word_atoms(tok, glued, after_digit, starts_address, nxt_digit, nxt is None or not nxt.group().isalpha())):
                atoms.append(_Atom(kind, value, tok, glued or n > 0))
    return atoms


# ---------------------------------------------------------------------------
# zones: doors (via + numbers + letters + quadrant), prefix, tail
# ---------------------------------------------------------------------------
class _Parsed(NamedTuple):
    atoms: list
    doors: list      # list of lists of atoms (separators included)
    prefix: list
    tail: list
    spans: list = []     # (start, end) atom indices of every door


def _number_ahead(atoms: list, i: int) -> bool:
    """A number within the next three atoms and before another via type (``Gran Via, Cra 903 #969-959``)."""
    for a in atoms[i + 1:i + 4]:
        if a.kind == "V":
            return False
        if a.kind == "D":
            return True
    return False


def _scan_door(atoms: list, start: int, strict: int = 2) -> int:
    """End index (exclusive) of the door that begins at ``start``.

    A door holds three digit *clusters* (via, cross, plate); a group that follows a letter or quadrant
    (``26G6``, ``5b 2``) continues its cluster. After an explicit ``#`` only two clusters (cross and plate)
    may follow; a 4th cluster without ``#`` is accepted only when no free word follows it. A number right
    after a free word (``Cra 31, Comuna 10``) or after a unit marker (``... 37 T 1``) is
    never door. ``strict`` is 2 (all of that, for the raw input where such tails exist), 1 (only the
    free-word / unit-marker rules) or 0 (the machine-rendered output: only the overall cap applies).
    """
    n, nums, clusters, after_hash, hash_d = len(atoms), 0, 0, -1, 0
    j = start + 1 if atoms[start].kind == "V" else start
    while j < n:
        a = atoms[j]
        if a.kind in (";", "U"):
            break
        if a.kind == "V" and nums >= 3:
            break
        if a.kind == "#":
            if after_hash >= 0 and strict >= 1 and hash_d >= 2:
                break           # ``# 10-3 numero 10``: a repeated number sign restates the address
            after_hash, hash_d = 0, 0
        plate_seen = (hash_d >= 2) if after_hash >= 0 else clusters >= 3
        if (strict >= 1 and a.kind == "L" and plate_seen and (not a.glued or a.value == "T")
                and j + 1 < n and atoms[j + 1].kind == "D"):
            break               # a unit marker after the plate
        if a.value == "Y" and a.kind in "WL" and j + 1 < n and atoms[j + 1].kind == "V":
            break               # ``... 925 Y KR 907 # 910 - 920``: a second address starts
        if a.kind == "W" and nums >= 2:
            break
        if a.kind == "D":
            if nums >= MAX_DOOR_NUMBERS:
                break
            prev = atoms[j - 1]
            if strict >= 1 and j > start and prev.kind == "W" and prev.value != "CON":
                break
            continues = j > start and prev.kind in ("L", "Q", "B") and (a.glued or not prev.glued)
            if strict >= 2 and after_hash >= 0 and (hash_d >= 3 or (hash_d >= 2 and j + 1 < n and atoms[j + 1].kind == "W")):
                break           # a 3rd number after ``#`` followed by place text, or a 4th one
            hash_d += after_hash >= 0
            if not continues:
                if after_hash >= 0:
                    if strict >= 2 and after_hash >= 2:
                        break
                    after_hash += 1
                elif strict >= 2 and clusters >= 3 and (prev.kind in ("L", "Q", "B") or (j + 1 < n and atoms[j + 1].kind == "W")):
                    break
                clusters += 1
            nums += 1
        j += 1
    return j


def _trim_trailing_letters(door: list) -> list:
    """Letters after the last number (``... 34 X``, ``... 34 D AP 401``) are labels, not door letters."""
    digits = [k for k, a in enumerate(door) if a.kind == "D"]
    if len(digits) < 3:
        return door
    return door[:digits[-1] + 1] + [a for a in door[digits[-1] + 1:] if a.kind != "L"]


def _zones(text: str, strict: int = 2) -> _Parsed:
    atoms = _atomize(text)
    doors: list[list] = []
    spans: list[tuple] = []
    i = 0
    while i < len(atoms):
        if atoms[i].kind == "V" and _number_ahead(atoms, i):
            end = _scan_door(atoms, i, strict)
            spans.append((i, end))
            i = end
        else:
            i += 1
    if not spans:
        first = next((k for k, a in enumerate(atoms) if a.kind == "D"), None)
        if first is not None:
            spans.append((first, _scan_door(atoms, first, strict)))
    for s, e in spans:
        door = [a for a in atoms[s:e] if a.kind != "W"]
        doors.append(_trim_trailing_letters(door))
    inside = {k for s, e in spans for k in range(s, e)}
    first_start = spans[0][0] if spans else len(atoms)
    prefix = atoms[:first_start]
    tail = [a for k, a in enumerate(atoms) if k >= first_start and k not in inside]
    return _Parsed(atoms, doors, prefix, tail, spans)


def _items(door: list) -> list:
    return [(a.kind, a.value) for a in door if a.kind in "VDLQB"]


def _door_str(door: list) -> str:
    return " ".join(v for _k, v in _items(door))[:80]


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _segments(items: list) -> list:
    """(kind, value, segment) with segment = number of digit groups before the item."""
    seg, out = 0, []
    for kind, value in items:
        if kind == "D":
            seg += 1
        out.append((kind, value, seg))
    return out


def _diff_flags(raw: list, out: list, names: tuple, flags: set) -> None:
    dropped, added, changed = names
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, raw, out, autojunk=False).get_opcodes():
        if tag == "delete":
            flags.add(dropped)
        elif tag == "insert":
            flags.add(added)
        elif tag == "replace":
            flags.add(changed)


def _compare_door(rd: list, od: list, flags: set, details: dict) -> None:
    ri, oi = _items(rd), _items(od)
    rv, ov = [v for k, v in ri if k == "V"], [v for k, v in oi if k == "V"]
    if rv[:1] != ov[:1]:
        flags.add("via_type_guessed" if not rv else "via_type_dropped" if not ov else "via_type_changed")
    elif rv[1:] != ov[1:]:      # cross street named by type (``CL 46 KRA 40``)
        if not rv[1:]:
            flags.add("cross_type_added")     # the cadastre names the cross street's type: ``# AV 6``
        elif not ov[1:]:
            flags.add("cross_type_dropped")
        else:
            flags.add("via_type_changed")
    dr = [v for k, v in ri if k == "D"]
    do = [v for k, v in oi if k == "D"]
    if dr != do and len(dr) >= 4:           # with 3 numbers ``26-26`` is a real cross / plate pair
        collapsed = _collapse_repeats(ri)
        if [v for k, v in collapsed if k == "D"] == do:
            ri, dr = collapsed, do          # ``205-205`` / ``... con calle 13 #13-22``: a number typed twice
            flags.add("duplicate_dropped")
    if dr != do:
        if sorted(dr) == sorted(do):
            flags.add("door_digits_reordered")
            if len(dr) >= 3:
                flags.add("scrambled_order")
        elif not do:
            flags.add("door_digits_dropped")
        elif len(dr) == len(do):
            diff = [i for i, (a, b) in enumerate(zip(dr, do)) if a != b]
            if 0 in diff:
                flags.add("via_number_changed")
            if len(dr) - 1 in diff:
                flags.add("plate_changed")
                if dr[-1].isdigit() and do[-1].isdigit():
                    details["plate_delta"] = int(do[-1]) - int(dr[-1])
            if any(0 < i < len(dr) - 1 for i in diff):
                flags.add("cross_number_changed")
        else:
            _diff_flags(dr, do, ("door_digits_dropped", "door_digits_added", "door_digits_changed"), flags)
    else:
        raw_strs = [a.text for a in rd if a.kind == "D"]
        out_strs = [a.text for a in od if a.kind == "D"]
        if raw_strs != out_strs:
            flags.add("zero_padding")
    same_digits = dr == do
    rs, os_ = _segments(ri), _segments(oi)
    for kinds, label, moved in ((("L", "B"), "letters", "letters_moved"), (("Q",), "quadrant", "quadrant_moved")):
        rl = [(v, s) for k, v, s in rs if k in kinds]
        ol = [(v, s) for k, v, s in os_ if k in kinds]
        if rl == ol:
            continue
        rvals, ovals = [v for v, _s in rl], [v for v, _s in ol]
        if rvals == ovals:
            if same_digits:
                flags.add(moved)
        elif sorted(rvals) == sorted(ovals):
            flags.add(moved)
        else:
            _diff_flags(rvals, ovals, (f"{label}_dropped", f"{label}_added", f"{label}_changed"), flags)
    if same_digits and ri == oi:
        _compare_separators(rd, od, flags)


def _collapse_repeats(items: list) -> list:
    """The items without a digit group that repeats the digit group right before it."""
    out: list = []
    for item in items:
        if item[0] == "D" and out and out[-1] == item:
            continue
        out.append(item)
    return out


def _separator_map(door: list) -> tuple:
    """For consecutive items i, i+1: (separators between them, glued inside a ``D L D`` cluster)."""
    seps, glue, pending = [], [], set()
    items = [a for a in door if a.kind in "VDLQB" or a.kind in "#-"]
    clusters: list = []          # cluster id of every item (V is its own cluster)
    cid, prev_item = 0, None
    order = [a for a in items if a.kind in "VDLQB"]
    for a in items:
        if a.kind in "#-":
            pending.add(a.kind)
            continue
        if prev_item is not None:
            seps.append(pending)
            joined = a.glued and not pending and a.kind != "V" and prev_item.kind != "V"
            glue.append(joined)
            cid += 0 if joined else 1
        clusters.append(cid)
        pending, prev_item = set(), a
    digits = {}
    for c, a in zip(clusters, order):
        digits[c] = digits.get(c, 0) + (a.kind == "D")
    # a boundary is a split of a number only when its cluster holds 2+ digit groups
    glue = [g and digits[clusters[i]] >= 2 for i, g in enumerate(glue)]
    return seps, glue


def _digit_clusters(door: list) -> int:
    """Number of digit-bearing glued runs (``26G6`` is one) between spaces / separators / via types."""
    clusters, has_digit, open_run = 0, False, False
    for a in door:
        if a.kind in "DLQB":
            if not (open_run and a.glued):
                clusters, has_digit = clusters + has_digit, False
            open_run = True
            has_digit = has_digit or a.kind == "D"
        else:
            clusters, has_digit, open_run = clusters + has_digit, False, False
    return clusters + has_digit


def _compare_separators(rd: list, od: list, flags: set) -> None:
    rseps, rglue = _separator_map(rd)
    oseps, _og = _separator_map(od)
    # Splitting a glued token (``69a43`` -> ``69 A - 43``) is legitimate only when the input has too few
    # digit clusters to hold via, cross and plate: ``KR 926G6 972S1 939`` already has three clusters, so a
    # ``#`` inside ``26G6`` re-reads the address.
    split_is_forced = _digit_clusters(rd) < 3
    for i, glued in enumerate(rglue):
        if glued and not split_is_forced and i < len(oseps) and oseps[i]:
            flags.add("glued_split")
    r_hash = [i for i, s in enumerate(rseps) if "#" in s]
    o_hash = [i for i, s in enumerate(oseps) if "#" in s]
    if len(r_hash) == 1 and len(o_hash) == 1 and r_hash != o_hash:
        flags.add("separator_moved")


def _unit_entries(atoms: list) -> list:
    """(canonical unit keyword, digit value) for every unit keyword followed by a number."""
    out = []
    for i, a in enumerate(atoms):
        if a.kind != "U":
            continue
        for b in atoms[i + 1:i + 4]:
            if b.kind == "D":
                out.append((a.value, b.value))
                break
            if b.kind not in ("L", "#", "-"):
                break
    return out


def _compare_tail(raw: _Parsed, out: _Parsed, kept: _Parsed | None, flags: set) -> None:
    raw_door_complete = any(sum(a.kind == "D" for a in d) >= 2 for d in raw.doors)
    raw_zone = (raw.prefix if not raw_door_complete else []) + raw.tail
    out_zone = out.prefix + out.tail
    kept_values = None if kept is None else {(a.kind, a.value) for a in kept.atoms}
    removed_by_place = (lambda a: kept_values is not None and (a.kind, a.value) not in kept_values)
    out_units = _unit_entries(out_zone)
    out_digits = [a.value for a in out_zone if a.kind == "D"]
    out_door_digits = {a.value for d in out.doors for a in d if a.kind == "D"}
    raw_all_digits = {a.value for a in raw.atoms if a.kind == "D"} | {
        _SPELLED[a.text] for a in raw.atoms if a.text in _SPELLED}
    raw_units = _unit_entries(raw_zone)
    unit_digit_values = {d for _kw, d in raw_units}
    for kw, digit in raw_units:
        if (kw, digit) in out_units or digit in out_digits or digit in out_door_digits:
            if (kw, digit) not in out_units:
                flags.add("unit_relabelled")
            continue
        probe = next((a for a in raw_zone if a.kind == "D" and a.value == digit), None)
        if probe is not None and removed_by_place(probe):
            continue
        flags.add("unit_digits_dropped" if kw in _STRONG_CANON else "annotation_digits_dropped")
    for k, a in enumerate(raw_zone):
        if a.kind != "D" or a.value in unit_digit_values:
            continue
        if a.value in out_digits or a.value in out_door_digits or removed_by_place(a):
            continue
        flags.add("prefix_digits_dropped" if (not raw_door_complete and a in raw.prefix) else "annotation_digits_dropped")
    if any(d not in raw_all_digits for d in out_digits):
        flags.add("tail_digits_added")


#: Words that join the two ends of a range (``4-02 al 4-10``).
_RANGE_WORDS = frozenset(("AL", "HASTA"))


def _trailing_run(atoms: list, end: int) -> list:
    """Number groups that continue a door (``- 906``, ``/910``, ``# 904-910``, ``al 4-10``, ``-4D 42``).

    The run stops at a unit marker, a via type, a free word or ``;``: whatever follows is a unit / place
    annotation that the tail rules judge (unit digits stay low by policy). A bare number right before a
    unit or place word (``3er piso``, ``3 de Julio``) belongs to that text and is not part of the run.
    """
    run: list = []
    stop = None
    for k, a in enumerate(atoms[end:], start=end):
        lone_marker = a.kind == "L" and not a.glued and k + 1 < len(atoms) and atoms[k + 1].kind == "D"    # ``P 1``, ``T 9``
        if lone_marker and a.value == "A" and k + 2 < len(atoms) and atoms[k + 2].kind == "-":    # ``4-02 a 4-10``: a range word
            lone_marker = False
        if not lone_marker and (a.kind in ("#", "-", "D", "L", "Q", "B") or (a.kind == "W" and a.value in _RANGE_WORDS)):
            run.append(a)
        else:
            stop = a
            break
    if stop is not None and stop.kind in ("U", "W", "L"):
        for k in range(len(run) - 1, -1, -1):
            if run[k].kind == "D":
                before = run[k - 1] if k else atoms[end - 1] if end else None
                if before is None or before.kind not in ("#", "-"):
                    del run[k]
                break
    return run


def _compare_trailing(raw: _Parsed, out: _Parsed, flags: set) -> None:
    """Digits that trail a raw door and have no trace in the output (``... 35 - 10 - 106`` -> ``... 35 - 10``).

    The output's digits are matched as a multiset after removing the ones the raw doors explain, so a
    number the door already used cannot vouch for a dropped one. A door letter glued to the last door number
    (``40A 41``) that the output lost is a door change (high).
    """
    pool: dict = {}
    for a in out.atoms:
        if a.kind == "D":
            pool[a.value] = pool.get(a.value, 0) + 1
    for door in raw.doors:
        for a in door:
            if a.kind == "D" and pool.get(a.value, 0) > 0:
                pool[a.value] -= 1
    out_letters = {a.value for a in out.atoms if a.kind == "L"}
    for start, end in raw.spans:
        run = _trailing_run(raw.atoms, end)
        door_values = {a.value for a in raw.atoms[start:end] if a.kind == "D"}
        digits = [a for a in run if a.kind == "D" and a.value not in door_values]    # a restated number is not a loss
        missing = []
        for a in digits:
            if pool.get(a.value, 0) > 0:
                pool[a.value] -= 1
            else:
                missing.append(a)
        if not missing:
            continue
        range_like = any(a.kind == "#" or a.kind == "W" for a in run)
        flags.add("multi_address_truncated" if range_like else "trailing_number_dropped")
        last = raw.atoms[end - 1] if end - 1 > start else None
        if (last is not None and last.kind == "L" and last.glued and raw.atoms[end - 2].kind == "D"
                and run[0].kind == "D" and last.value not in out_letters):
            flags.add("letters_dropped")


def _share_dropped(raw: _Parsed, out: _Parsed, kept: _Parsed | None, flags: set, details: dict) -> None:
    """Share of the raw door tokens and strong-unit tokens that the output no longer holds.

    Duplicated doors (``X; X``), junk before the via type, place text and weak units / annotation digits
    are not counted here: the tail rules judge those with their own (lower) flags.
    """
    seen: list = []
    counted: list = []
    for door in raw.doors:
        items = _items(door)
        if items not in seen:
            seen.append(items)
            # a repeated number counts once and a named cross street's type is judged by its own (low) flag
            counted.extend(i for n, i in enumerate(_collapse_repeats(items)) if i[0] != "V" or n == 0)
    zone = (raw.prefix if not any(sum(a.kind == "D" for a in d) >= 2 for d in raw.doors) else []) + raw.tail
    kept_values = None if kept is None else {(a.kind, a.value) for a in kept.atoms}
    seen_units: list = []
    for i, a in enumerate(zone):
        if a.kind != "U" or a.value not in _STRONG_CANON:
            continue
        if kept_values is not None and (a.kind, a.value) not in kept_values:
            continue
        entry = [(a.kind, a.value)]
        for b in zone[i + 1:i + 4]:
            if b.kind == "D":
                entry.append(("D", b.value))
                break
            if b.kind not in ("L", "#", "-"):
                break
        if entry not in seen_units:        # a unit repeated in the raw text counts once
            seen_units.append(entry)
            counted.extend(entry)
    pool: dict = {}
    for a in out.atoms:
        if a.kind in "DLQBVU":
            pool[(a.kind, a.value)] = pool.get((a.kind, a.value), 0) + 1
    dropped = 0
    for key in counted:
        if pool.get(key, 0) > 0:
            pool[key] -= 1
        else:
            dropped += 1
    share = dropped / len(counted) if counted else 0.0
    details["dropped_share"] = round(share, 3)
    if dropped:
        flags.add("text_dropped_major" if share >= TEXT_DROP_MEDIUM_SHARE else "text_dropped")


def _is_address(door: list) -> bool:
    """A door that stands for an address of its own (not an ``ENTRE AV 3F Y 3H`` description)."""
    nums = sum(a.kind == "D" for a in door)
    return nums >= 3 or (nums >= 2 and any(a.kind == "#" for a in door))


def _best_door(raw_doors: list, out_door: list) -> int:
    target = _items(out_door)
    scores = [difflib.SequenceMatcher(None, _items(d), target, autojunk=False).ratio() for d in raw_doors]
    return max(range(len(scores)), key=scores.__getitem__)


def _report(flags: set, details: dict) -> FidelityReport:
    worst = max((SEVERITY[f] for f in flags), key=_RANK.__getitem__, default="none")
    ordered = tuple(sorted(flags, key=lambda f: (-_RANK[SEVERITY[f]], f)))
    extra = max(len(ordered) - 1, 0) if worst != "none" else 0
    score = max(_SCORE_BASE[worst] - _SCORE_STEP[worst] * extra, _SCORE_FLOOR[worst])
    return FidelityReport(worst, ordered, round(score, 4), details)


def _assess(raw_parsed: _Parsed, out_parsed: _Parsed, kept: _Parsed | None) -> FidelityReport:
    flags: set = set()
    details: dict = {}
    if not out_parsed.doors and not raw_parsed.doors:
        _compare_tail(raw_parsed, out_parsed, kept, flags)
    elif not raw_parsed.doors:
        flags.add("door_digits_added")
    elif not out_parsed.doors:
        flags.add("door_digits_dropped")
    else:
        best = _best_door(raw_parsed.doors, out_parsed.doors[0])
        _compare_door(raw_parsed.doors[best], out_parsed.doors[0], flags, details)
        out_sets = [_items(d) for d in out_parsed.doors]
        best_items = _items(raw_parsed.doors[best])
        for k, door in enumerate(raw_parsed.doors):
            items = _items(door)
            if k != best and items != best_items and _is_address(door) and items not in out_sets:
                flags.add("multi_address_truncated")
        details["raw_door"], details["out_door"] = _door_str(raw_parsed.doors[best]), _door_str(out_parsed.doors[0])
    if raw_parsed.doors and out_parsed.doors:
        _compare_tail(raw_parsed, out_parsed, kept, flags)
        _compare_trailing(raw_parsed, out_parsed, flags)
    _share_dropped(raw_parsed, out_parsed, kept, flags, details)
    details["n_raw_doors"] = len(raw_parsed.doors)
    return _report(flags, details)


def assess_fidelity(raw, output, cleaned: str | None = None) -> FidelityReport:
    """Compare a raw address with its final normalized form.

    ``cleaned`` (optional) is the raw text left after the gazetteer removed place names: tail tokens
    (never door tokens) that disappeared in that step are known, legitimate removals and are not counted.
    Non-string / empty values are treated as empty text.

    The raw door is read twice, with the full heuristics that keep place-text digits out of the door and
    with only the free-word / unit-marker ones; the reading that explains the output best (lowest risk)
    wins, so an ambiguous input is never blamed for a layout choice the output shares.
    """
    raw_t, out_t = _as_text(raw), _as_text(output)
    out_parsed = _zones(out_t, strict=0)
    if not any(a.kind in "DLQBVUW" for a in out_parsed.atoms):
        return _report({"no_address"}, {})
    first = _zones(raw_t, strict=2)
    if not first.atoms:
        return _report({"output_from_nothing"}, {"out_door": _door_str(out_parsed.doors[0]) if out_parsed.doors else ""})
    kept = None if cleaned is None else _zones(_as_text(cleaned))
    best = _assess(first, out_parsed, kept)
    if best.risk != "none":
        relaxed = _assess(_zones(raw_t, strict=1), out_parsed, kept)
        if (_RANK[relaxed.risk], len(relaxed.flags)) < (_RANK[best.risk], len(best.flags)):
            best = relaxed
    return best


def assess_fidelity_batch(raws: Iterable, outputs: Iterable, cleaned: Iterable | None = None) -> list[FidelityReport]:
    """``assess_fidelity`` over parallel sequences (``cleaned`` optional, same length)."""
    raws, outputs = list(raws), list(outputs)
    if len(raws) != len(outputs):
        raise ValueError(f"raws ({len(raws)}) and outputs ({len(outputs)}) must have the same length")
    cleaned_list = [None] * len(raws) if cleaned is None else list(cleaned)
    if len(cleaned_list) != len(raws):
        raise ValueError("cleaned must have the same length as raws")
    return [assess_fidelity(r, o, c) for r, o, c in zip(raws, outputs, cleaned_list)]
