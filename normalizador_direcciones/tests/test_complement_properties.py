"""Property tests of the complement stage over synthetic inputs (fixed seed).

The inputs are generated here from templates (kind spellings, value shapes,
separators, free-text noise); no survey, evaluation or cadastral data is read.
Properties:

* no exception, for any generated text;
* ``render_complement`` is idempotent and ``merge_complement`` is stable on its
  own output;
* distinct unit values of one kind never render identically;
* text with no real unit renders to no unit.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import extract_complement_units, merge_complement, render_complement  # noqa: E402

SEED = 20261009
BASE = "CL 99 # 9 - 99"

_KIND_SPELLINGS = {
    "AP": ["APTO", "AP", "APARTAMENTO", "apto", "Apto."],
    "TO": ["TORRE", "TO", "T-", "torre", "TOR"],
    "BLQ": ["BLOQUE", "BL", "BLQ", "bloque"],
    "CA": ["CASA", "CA", "casa"],
    "INT": ["INTERIOR", "INT"],
    "LC": ["LOCAL", "LC"],
    "OF": ["OFICINA", "OF"],
    "PISO": ["PISO", "piso"],
    "MZ": ["MANZANA", "MZ"],
    "LT": ["LOTE", "LT"],
}
_NOISE = [
    "", "FRENTE AL PARQUE", "ESQUINERA", "SEGUNDO PISO", "DE DOS PISOS", "SOBRE LA AVENIDA",
    "AL LADO DE LA IGLESIA", "PEGADO AL COLEGIO", "ENTRANDO POR LA PORTERIA", "CERCA DEL CAI",
    "BARRIO EL GUABAL", "VEREDA LA BUITRERA", "CASA BLANCA", "TORRE ROJA",
]
_NO_UNIT = [
    "FRENTE AL PARQUE", "CASA ESQUINERA", "LOCAL COMERCIAL", "TORRE PRINCIPAL", "EDIFICIO DE TRES PISOS",
    "SECTOR DE LA IGLESIA", "CONJUNTO CERRADO", "EDIFICIO ESQUINERO", "TORRE SOBRE LA AVENIDA",
    "BLOQUE DETRAS", "TORRE BLANCA", "BLOQUE AZUL", "TORRE KM 5", "TORRE LETRA", "BARRIO LA FLORA",
    "VEREDA LA BUITRERA", "CORREGIMIENTO PANCE", "TORRE", "APTO", "CASA DE DOS PISOS", "AP Y",
    "TORRE ENTRANDO", "TORRE DE LA ESQUINA", "PISO HUDSON", "APTO ZAFIRO", "SEGUNDO PISO",
    "CERCA AL MIO", "TORRE PPAL", "TORRE PREDIO", "OFICINA PRINCIPAL", "INTERIOR",
]


def _value(rng: random.Random) -> str:
    """A random code value as a user might type it."""
    shape = rng.randrange(8)
    if shape == 0:
        return str(rng.randrange(1, 2000))
    if shape == 1:
        return f"{rng.randrange(1, 30)}{rng.choice(['-', ' - ', ' '])}{rng.randrange(0, 20):02d}"
    if shape == 2:
        return rng.choice("ABCDEFGH")
    if shape == 3:
        return f"{rng.randrange(1, 999)}{rng.choice(['', ' ', '-'])}{rng.choice('ABCD')}"
    if shape == 4:
        return f"{rng.choice('ABCDEFGH')}{rng.choice(['-', ''])}{rng.randrange(1, 9)}"
    if shape == 5:
        return f"0{rng.randrange(1, 99)}"
    if shape == 6:
        return f"{rng.choice(['N', 'NO', 'NRO', 'NUM'])} {rng.randrange(1, 300)}"
    return rng.choice(["II", "III", "IV", "VIII", "XII"])


def _generated_texts(n: int) -> list[str]:
    rng = random.Random(SEED)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randrange(0, 4)):
            kind = rng.choice(list(_KIND_SPELLINGS))
            spelling = rng.choice(_KIND_SPELLINGS[kind])
            sep = "" if spelling.endswith("-") else " "
            parts.append(f"{spelling}{sep}{_value(rng)}")
            if rng.random() < 0.3:
                parts.append(rng.choice(_NOISE))
        if rng.random() < 0.2:
            parts.insert(0, rng.choice(_NOISE))
        out.append(" ".join(p for p in parts if p))
    return out


TEXTS = _generated_texts(3000)


def test_generator_is_deterministic():
    assert _generated_texts(50) == TEXTS[:50]


def test_no_exception_and_render_is_idempotent_on_generated_texts():
    for text in TEXTS:
        units = extract_complement_units(text)
        rendered = render_complement(text)
        assert (rendered is None) == (not units), text
        if rendered is not None:
            assert render_complement(rendered) == rendered, text
            assert extract_complement_units(rendered) == _first_per_kind(units), text


def test_merge_is_stable_on_its_own_output_for_generated_texts():
    for text in TEXTS[:1000]:
        first = merge_complement(BASE, text)
        assert first.complement == render_complement(text), text
        again = merge_complement(first.address, text)
        assert again.address == first.address and again.conflicts == first.conflicts, text


def _first_per_kind(units: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for kind, value in units:
        seen.setdefault(kind, value)
    from cali_address.parser import _KIND_RANK  # local: ordering is an implementation detail

    return sorted(seen.items(), key=lambda kv: _KIND_RANK.get(kv[0], 0))


def _family() -> list[str]:
    """Distinct apartment/tower values: one building unit each."""
    values = set()
    for floor in range(1, 13):
        for apt in range(0, 12):
            values.add(f"{floor}-{apt:02d}")
            values.add(f"{floor}{apt:02d}")
    for n in range(1, 40):
        for letter in "ABCDF":
            values.add(f"{n}{letter}")
            values.add(f"{letter}-{n}")
    values.update("ABCDEFGHJK")
    return sorted(values)


@pytest.mark.parametrize("kind", ["APTO", "TORRE", "BLOQUE", "CASA", "INTERIOR"])
def test_distinct_unit_values_never_render_identically(kind):
    rendered: dict[str, str] = {}
    for value in _family():
        out = render_complement(f"{kind} {value}")
        assert out is not None, value
        assert out not in rendered, (value, rendered.get(out), out)
        rendered[out] = value


@pytest.mark.parametrize("sep", [" ", "-", " - "])
def test_separator_spelling_never_changes_the_unit(sep):
    for floor in range(1, 13):
        for apt in range(0, 12):
            assert render_complement(f"APTO {floor}{sep}{apt:02d}") == f"AP {floor}-{apt:02d}"


@pytest.mark.parametrize("text", _NO_UNIT)
def test_text_without_a_real_unit_renders_no_unit(text):
    assert extract_complement_units(text) == []
    assert render_complement(text) is None
    assert merge_complement(BASE, text).address == BASE


def test_generated_noise_alone_renders_no_unit():
    rng = random.Random(SEED + 1)
    for _ in range(2000):
        text = " ".join(rng.choice(_NO_UNIT + _NOISE) for _ in range(rng.randrange(1, 4)))
        assert render_complement(text) is None, text
