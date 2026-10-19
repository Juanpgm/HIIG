"""Ordinary words are not tower or building names; locality kinds are not emitted.

* After ``TORRE``/``BLOQUE``/``ETAPA``: prepositions, adverbs, gerunds, colours,
  landmarks and common abbreviations are never a tower name, while an unseen
  plausible proper name still is (HUDSON, TAMESIS, ZAFIRO, MURANO).
* After a name kind (``EDIFICIO``, ``CONJUNTO``, ``SECTOR``...): a leading connector
  means there is no name, leading qualifiers (``RESIDENCIAL``, ``CERRADO``) are not
  part of it, and the name stops at the first qualifier.
* ``BARRIO``/``VEREDA``/``CORREGIMIENTO`` are recognised (so they never leak into
  another value) but not emitted: they are not building-unique.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address import complement_lexicon as lex  # noqa: E402
from cali_address.parser import (  # noqa: E402
    extract_complement_units,
    is_tower_name_word,
    merge_complement,
    render_complement,
)

BASE = "CL 99 # 9 - 99"

_ORDINARY_WORDS = [
    "SOBRE", "DESDE", "HASTA", "HACIA", "TRAS", "ATRAS", "DETRAS", "LEJOS", "PEGADO", "PEGADA",
    "ENSEGUIDA", "SIGUIENTE", "QUEDA", "ENTRANDO", "SUBIENDO", "BAJANDO", "SALIENDO",
    "BLANCA", "BLANCO", "ROJA", "AMARILLA", "AZUL", "VERDE", "NEGRA", "GRIS", "ROSADA", "MORADA",
    "NARANJA", "METALICA", "PARQUE", "IGLESIA", "COLEGIO", "ESCUELA", "HOSPITAL", "CAI", "MIO",
    "PREDIO", "INMUEBLE", "PPAL", "TALLER", "TIENDA",
]


# ---------------------------------------------------------------------------
# Tower kinds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("word", _ORDINARY_WORDS)
@pytest.mark.parametrize("kind", ["TORRE", "BLOQUE", "ETAPA"])
def test_ordinary_word_after_a_tower_kind_is_not_a_unit(kind, word):
    text = f"{kind} {word}"
    assert extract_complement_units(text) == [], text
    assert render_complement(text) is None
    assert merge_complement(BASE, text).address == BASE


@pytest.mark.parametrize("word", _ORDINARY_WORDS)
def test_ordinary_words_are_not_name_words(word):
    assert not is_tower_name_word(word)


@pytest.mark.parametrize("text", ["TORRE KM 5", "TORRE KM", "BLOQUE KM 3"])
def test_km_is_not_a_tower_code(text):
    assert extract_complement_units(text) == []


@pytest.mark.parametrize("colour", ["VERDE", "AZUL", "BLANCO", "ROJO"])
def test_colours_are_treated_alike_after_bloque(colour):
    assert extract_complement_units(f"BLOQUE {colour}") == []


@pytest.mark.parametrize("word", ["HUDSON", "TAMESIS", "ZAFIRO", "MURANO"])
def test_unseen_proper_names_are_still_accepted(word):
    assert is_tower_name_word(word)
    assert extract_complement_units(f"TORRE {word}") == [("TO", word)]


def test_hudson_and_tamesis_stay_distinct():
    assert render_complement("TORRE HUDSON") != render_complement("TORRE TAMESIS")


def test_name_followed_by_an_ordinary_word_keeps_only_the_name():
    assert extract_complement_units("TORRE HUDSON SOBRE LA AVENIDA") == [("TO", "HUDSON")]
    assert extract_complement_units("TORRE ZAFIRO ENTRANDO") == [("TO", "ZAFIRO")]


def test_new_seed_words_are_in_the_generated_descriptive_lexicon():
    for word in ("PARQUE", "IGLESIA", "COLEGIO", "BLANCA", "AZUL", "PPAL", "INMUEBLE", "TALLER"):
        assert word in lex.DESCRIPTIVE_WORDS
        assert word not in lex.TOWER_NAME_WORDS


# ---------------------------------------------------------------------------
# Name kinds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "EDIFICIO DE TRES PISOS", "SECTOR DE LA IGLESIA", "CONJUNTO CERRADO", "EDIFICIO ESQUINERO",
        "EDIFICIO LA ESQUINA", "CONJUNTO RESIDENCIAL", "EDIFICIO SOBRE LA AVENIDA", "SECTOR Y",
        "EDIFICIO", "ED DEL", "CONJUNTO CERRADO RESIDENCIAL", "EDIFICIO ESQUINERO PISO HUDSON",
        "EDIFICIO PISO HUDSON", "EDIFICIO ESQUINERO PISO HUDSON TORRE ENTRANDO",
    ],
)
def test_name_kind_without_a_real_name_is_not_a_unit(text):
    assert extract_complement_units(text) == []
    assert render_complement(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ED LAS PALMAS TORRE G", [("ED", "LAS PALMAS"), ("TO", "G")]),
        ("EDIFICIO TORRE PACIFICO", [("ED", "TORRE PACIFICO")]),
        ("CONJ TORRES DE COMFANDI", [("CONJ", "TORRES DE COMFANDI")]),
        ("CONJUNTO RESIDENCIAL LOS ALAMOS", [("CONJ", "LOS ALAMOS")]),
        ("CONJUNTO CERRADO LOS ALAMOS", [("CONJ", "LOS ALAMOS")]),
        ("CONJUNTO LOS ALAMOS", [("CONJ", "LOS ALAMOS")]),
        ("CONJUNTO RESIDENCIAL TORRE TAMESIS", [("TO", "TAMESIS")]),
        ("ED LAS PALMAS ESQUINERO", [("ED", "LAS PALMAS")]),
        ("ED LAS PALMAS DE", [("ED", "LAS PALMAS")]),
        ("ED LAS PALMAS TORRE", [("ED", "LAS PALMAS")]),
        ("ED. TORRE DE CALI", [("ED", "TORRE DE CALI")]),
        ("SECTOR LOS BALSOS", [("SEC", "LOS BALSOS")]),
    ],
)
def test_legit_names_still_work(text, expected):
    assert extract_complement_units(text) == expected


def test_residencial_qualifier_does_not_split_one_complex():
    assert render_complement("CONJUNTO RESIDENCIAL LOS ALAMOS") == render_complement("CONJUNTO LOS ALAMOS")


# ---------------------------------------------------------------------------
# Locality kinds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    ["BARRIO EL GUABAL", "BR LA FLORA", "VEREDA LA BUITRERA", "CORREGIMIENTO PANCE", "CGTO PANCE", "VDA 1"],
)
def test_locality_kinds_are_not_emitted(text):
    assert extract_complement_units(text) == []
    assert render_complement(text) is None
    assert merge_complement(BASE, text).address == BASE


@pytest.mark.parametrize(
    "text, expected",
    [
        ("TORRE HUDSON BARRIO LA FLORA", "TO HUDSON"),
        ("BARRIO LA FLORA TORRE HUDSON", "TO HUDSON"),
        ("BARRIO EL GUABAL TORRE 2 APTO 101", "TO 2 AP 101"),
        ("CONJ DIVENTO BARRIO LA FLORA", "CONJ DIVENTO"),
        ("APTO 101 VEREDA LA BUITRERA", "AP 101"),
    ],
)
def test_locality_text_never_leaks_into_other_values(text, expected):
    assert render_complement(text) == expected


def test_typing_the_barrio_does_not_change_the_building_key():
    with_barrio = merge_complement(BASE, "TORRE F BARRIO CHIMINANGOS").address
    without = merge_complement(BASE, "TORRE F").address
    assert with_barrio == without == f"{BASE} TO F"


def test_cadastral_locality_chunk_is_neither_emitted_nor_a_conflict():
    m = merge_complement(f"{BASE} BR EL GUABAL", "BARRIO LA FLORA TORRE F")
    assert m.address == f"{BASE} TO F" and m.conflicts == ()
