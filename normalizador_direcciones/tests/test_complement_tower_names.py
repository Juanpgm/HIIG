"""Kind-aware complement value grammar: named towers/blocks vs descriptive words.

Failure this covers (synthetic address): ``Cra 99 #9Z-99 torre Hudson`` and ``... torre Tamesis``
are two different towers of one complex, but both lost the name (a letter-only
value of three or more letters was rejected as "descriptive") and merged into
``KR 99 # 9 Z - 99``. The grammar per kind:

* unit kinds (AP, CA, LC, OF, PISO, INT, LT, MZ, ...): codes only;
* tower kinds (TO, BLQ, ET): a code, a roman numeral, a number word, or a
  tower name (an optional article plus one or two name words) that passes the
  data-derived name test (``complement_lexicon``);
* name kinds (ED, CONJ, UR, ...): free-text names, as before.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import complement_lexicon as lex  # noqa: E402
from cali_address.parser import (  # noqa: E402
    COMPLEMENT_CANON,
    extract_complement_units,
    is_tower_name_word,
    merge_complement,
    parse_address,
    render_complement,
)

BASE = "KR 99 # 9 Z - 99"


# ---------------------------------------------------------------------------
# The real failure: Hudson vs Tamesis
# ---------------------------------------------------------------------------
def test_hudson_and_tamesis_keep_their_names():
    assert extract_complement_units("torre Hudson") == [("TO", "HUDSON")]
    assert extract_complement_units("torre Tamesis") == [("TO", "TAMESIS")]


def test_hudson_and_tamesis_on_the_same_base_address_differ():
    hudson = merge_complement(BASE, parse_address("Cra 99 #9Z-99 torre Hudson").complement_text)
    tamesis = merge_complement(BASE, parse_address("Cra 99 #9Z-99 torre Tamesis").complement_text)
    assert hudson.address == f"{BASE} TO HUDSON"
    assert tamesis.address == f"{BASE} TO TAMESIS"
    assert hudson.address != tamesis.address


def test_rule_canonical_of_named_towers_differ():
    a = render_complement(parse_address("Cra 99 #9Z-99 torre Hudson").complement_text)
    b = render_complement(parse_address("Cra 99 #9Z-99 torre Tamesis").complement_text)
    assert (a, b) == ("TO HUDSON", "TO TAMESIS")


@pytest.mark.parametrize(
    "text",
    ["TORRE HÚDSON", "torre hudson", "Torre  HuDsOn", "TORRE HUDSON.", "T-HUDSON", "torre: Hudson", "TOR HUDSON"],
)
def test_accents_case_and_punctuation_do_not_change_the_name(text):
    assert extract_complement_units(text) == [("TO", "HUDSON")]


def test_unseen_name_is_accepted():
    assert extract_complement_units("torre Zafiro") == [("TO", "ZAFIRO")]
    assert extract_complement_units("bloque Zafiro") == [("BLQ", "ZAFIRO")]


def test_named_tower_with_an_apartment():
    assert extract_complement_units("TORRE HUDSON APTO 501") == [("TO", "HUDSON"), ("AP", "501")]
    assert render_complement("APTO 501 TORRE HUDSON") == "TO HUDSON AP 501"


# ---------------------------------------------------------------------------
# Descriptive words after a tower kind are still rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "torre esquinera", "TORRE PRINCIPAL", "torre principal", "TORRE ESQUINERO",
        "TORRE FRENTE AL PARQUE", "TORRE LATERAL", "TORRE COMERCIAL", "TORRE RESIDENCIAL",
        "TORRE NUEVA", "TORRE SEGUNDA", "TORRE NORTE", "TORRE SUR", "TORRE CALI",
        "TORRE CALLE", "TORRE CARRERA", "TORRE DE", "TO DE LA ESQUINA", "TORRE LA ESQUINA",
        "TORRE EL", "TORRE Y", "BLOQUE ESQUINERO", "BLOQUE PRINCIPAL", "ETAPA PRINCIPAL",
    ],
)
def test_descriptive_word_after_a_tower_kind_is_not_a_unit(text):
    assert extract_complement_units(text) == []
    assert render_complement(text) is None
    assert merge_complement(BASE, text).address == BASE


@pytest.mark.parametrize(
    "text",
    ["TORRE TORRE", "TORRE CASA", "TORRE APARTAMENTO", "TORRE TORE", "TORRE TORRRE",
     "TORRE EDIFICO", "TORRE COJUNTO", "TORRE COJ", "TORRE PISOS", "TORRE CASAS"],
)
def test_kind_words_and_their_typos_are_not_tower_names(text):
    assert extract_complement_units(text) == []


# ---------------------------------------------------------------------------
# Unit kinds stay code-only (the previous negative cases)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "CASA ESQUINERA", "LOCAL COMERCIAL", "OFICINA PRINCIPAL", "LOTE ESQUINERO",
        "IN FRONT", "CASA DE DOS PISOS", "TO DE", "CASA HUDSON", "APTO ZAFIRO",
        "LOCAL TAMESIS", "PISO HUDSON", "INTERIOR HUDSON", "MANZANA HUDSON", "LOTE HUDSON",
        "OFICINA HUDSON",
    ],
)
def test_unit_kinds_never_take_a_name(text):
    assert extract_complement_units(text) == []


def test_conjunction_between_apartments_still_cut():
    assert extract_complement_units("AP 101 Y 102") == [("AP", "101")]


# ---------------------------------------------------------------------------
# Name kinds and kind words inside names keep working
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("ED. TORRE DE CALI", [("ED", "TORRE DE CALI")]),
        ("CONJ TORRES DE COMFANDI", [("CONJ", "TORRES DE COMFANDI")]),
        ("ED LAS PALMAS TORRE G", [("ED", "LAS PALMAS"), ("TO", "G")]),
        ("ED LAS PALMAS TORRE HUDSON", [("ED", "LAS PALMAS"), ("TO", "HUDSON")]),
        ("CONJUNTO RESIDENCIAL TORRE TAMESIS", [("TO", "TAMESIS")]),
        ("EDIFICIO TORRE PACIFICO", [("ED", "TORRE PACIFICO")]),
        ("ED TORRE F", [("TO", "F")]),
        ("ED. Torre de Cristal", [("ED", "TORRE DE CRISTAL")]),
    ],
)
def test_name_kinds_and_tower_names_together(text, expected):
    assert extract_complement_units(text) == expected


# ---------------------------------------------------------------------------
# Codes, numerals and number words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("TORRE F", [("TO", "F")]),
        ("T-F", [("TO", "F")]),
        ("TORRE 2", [("TO", "2")]),
        ("TORRE 02", [("TO", "2")]),
        ("TORRE II", [("TO", "II")]),
        ("TORRE III", [("TO", "III")]),
        ("TORRE IV", [("TO", "IV")]),
        ("BLOQUE VIII", [("BLQ", "VIII")]),
        ("TORRE XII", [("TO", "XII")]),
        ("TORRE DOS", [("TO", "2")]),
        ("torre tres", [("TO", "3")]),
        ("ETAPA UNO", [("ET", "1")]),
        ("BLOQUE DOCE", [("BLQ", "12")]),
        ("TORRE 3C", [("TO", "3C")]),
    ],
)
def test_tower_codes_numerals_and_number_words(text, expected):
    assert extract_complement_units(text) == expected


def test_number_word_tower_equals_digit_tower():
    assert merge_complement(BASE, "TORRE DOS").address == merge_complement(BASE, "TORRE 2").address


def test_number_words_are_not_units_for_unit_kinds():
    # Unchanged behaviour: only tower kinds read number words.
    assert extract_complement_units("PISO UNO") == []


def test_tower_f_and_g_conflict_is_still_reported():
    assert extract_complement_units("TO F TO G") == [("TO", "F"), ("TO", "G")]
    m = merge_complement(BASE, "TO F TO G")
    assert m.address == f"{BASE} TO F" and m.conflicts == ("TO",)


def test_named_tower_against_cadastral_code_tower_is_a_conflict_input_wins():
    m = merge_complement(f"{BASE} TO 2", "TORRE HUDSON")
    assert m.address == f"{BASE} TO HUDSON"
    assert m.conflicts == ("TO",)


# ---------------------------------------------------------------------------
# Two-word names and articles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("TORRE SAN MARTIN", [("TO", "SAN MARTIN")]),
        ("TORRE LAS PALMAS", [("TO", "LAS PALMAS")]),
        ("torre las palmas apto 302", [("TO", "LAS PALMAS"), ("AP", "302")]),
        ("TORRE HUDSON ESQUINERA", [("TO", "HUDSON")]),
        ("TORRE HUDSON FRENTE AL PARQUE", [("TO", "HUDSON")]),
        ("TORRE HUDSON DEL CONJUNTO", [("TO", "HUDSON")]),
        ("TORRE SAN MARTIN PLAZA REAL", [("TO", "SAN MARTIN")]),
    ],
)
def test_two_word_names_are_capped_and_stop_at_descriptive_words(text, expected):
    assert extract_complement_units(text) == expected


def test_san_martin_and_san_diego_do_not_collapse():
    a = merge_complement(BASE, "TORRE SAN MARTIN").address
    b = merge_complement(BASE, "TORRE SAN DIEGO").address
    assert a != b


# ---------------------------------------------------------------------------
# Empty, huge and repeated inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("empty", [None, "", "   ", "\t\n", " . - "])
def test_empty_inputs(empty):
    assert extract_complement_units(empty) == []
    assert render_complement(empty) is None
    assert merge_complement(BASE, empty).address == BASE


def test_very_long_name_token_is_rejected():
    assert extract_complement_units("TORRE " + "H" * 5000) == []


def test_5000_char_input_is_fast_and_bounded():
    text = " ".join(["TORRE HUDSON"] * 385)
    assert 4990 <= len(text) <= 5010
    start = time.perf_counter()
    out = extract_complement_units(text)
    assert time.perf_counter() - start < 2.0
    assert out == [("TO", "HUDSON")]
    assert merge_complement(BASE, text).address == f"{BASE} TO HUDSON"


def test_long_descriptive_tower_text_is_fast_and_empty():
    text = "TORRE " + "ESQUINERA " * 500
    start = time.perf_counter()
    assert extract_complement_units(text) == []
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("text", ["TORRE HUDSON", "TORRE SAN MARTIN APTO 101", "TORRE II", "TORRE DOS"])
def test_merge_and_render_are_idempotent(text):
    first = merge_complement(BASE, text)
    assert merge_complement(first.address, text).address == first.address
    rendered = render_complement(text)
    assert render_complement(rendered) == rendered


# ---------------------------------------------------------------------------
# The scoring complement is untouched (legacy spelling)
# ---------------------------------------------------------------------------
def test_scoring_complement_keeps_the_legacy_spelling():
    assert parse_address("Cra 99 #9Z-99 torre Hudson").complement == "TO HUDSON"
    assert parse_address("Cra 99 #9Z-99 TORRE DOS").complement == "TO DOS"


# ---------------------------------------------------------------------------
# Name test and lexicons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("word", ["HUDSON", "TAMESIS", "ZAFIRO", "PALMAS", "MARTIN"])
def test_name_words(word):
    assert is_tower_name_word(word)


@pytest.mark.parametrize(
    "word",
    ["ESQUINERA", "PRINCIPAL", "COMERCIAL", "DE", "LA", "Y", "NORTE", "CALI", "TORRE",
     "TORE", "CASAS", "", "AB", "H" * 13, "HUD5ON", "DOS"],
)
def test_not_name_words(word):
    assert not is_tower_name_word(word)


def test_name_test_is_case_and_accent_insensitive():
    assert is_tower_name_word("Hudson") and is_tower_name_word("TÁMESIS")
    assert not is_tower_name_word("Esquinera") and not is_tower_name_word("príncipal")


def test_lexicons_are_disjoint_frozensets_of_clean_words():
    assert isinstance(lex.TOWER_NAME_WORDS, frozenset)
    assert isinstance(lex.DESCRIPTIVE_WORDS, frozenset)
    assert not (lex.TOWER_NAME_WORDS & lex.DESCRIPTIVE_WORDS)
    for word in lex.TOWER_NAME_WORDS | lex.DESCRIPTIVE_WORDS:
        assert word.isalpha() and word.isupper() and word.isascii() and len(word) >= 3
    assert not (lex.TOWER_NAME_WORDS & set(COMPLEMENT_CANON))


def test_lexicons_contain_the_expected_anchors():
    for word in ("ESQUINERA", "ESQUINERO", "PRINCIPAL", "COMERCIAL", "FRENTE", "LATERAL"):
        assert word in lex.DESCRIPTIVE_WORDS
    # Attested place names from the IDESC basemaps.
    for word in ("CHIMINANGOS", "GUAYACANES"):
        assert word in lex.TOWER_NAME_WORDS
    for word in ("HUDSON", "TAMESIS", "ZAFIRO"):
        assert word not in lex.DESCRIPTIVE_WORDS


# ---------------------------------------------------------------------------
# Generator: pure builder + sync with the committed module
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "build_complement_lexicons", os.path.join(ROOT, "scripts", "build_complement_lexicons.py")
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_builder_on_tiny_inputs():
    cadastre = [
        "KR 1 # 2 - 3 ZONA VERDE", "KR 1 # 2 - 4 ZONA VERDE", "KR 1 # 2 - 5 ZONA VERDE",
        "KR 1 # 2 - 6 TO 2 AP 101", "KR 1 # 2 - 7 SKY", "CL 5 # 38 - 10",
        "KR 1 # 2 - 8 LAURELES", "KR 1 # 2 - 9 LAURELES", "KR 1 # 2 - 10 LAURELES",
        None, "", "???",
    ]
    places = ["Los Laureles", "Ciudad Jardín", "El Guabal", None, "", "La Esquina"]
    names, descriptive = gen.build_lexicons(cadastre, places, min_cadastre_count=3)
    assert {"LAURELES", "CIUDAD", "JARDIN", "GUABAL"} <= names
    assert {"ZONA", "VERDE"} <= descriptive
    assert "SKY" not in descriptive  # below the frequency floor
    assert "LAURELES" not in descriptive  # attested place name wins over cadastre frequency
    assert "ESQUINA" in descriptive and "ESQUINA" not in names  # seed wins over a place token
    assert not names & descriptive
    assert not {"LOS", "EL", "LA"} & (names | descriptive)


def test_builder_on_empty_inputs_keeps_only_the_seed():
    names, descriptive = gen.build_lexicons([], [], min_cadastre_count=3)
    assert names == frozenset()
    assert gen.DESCRIPTIVE_SEED <= descriptive


def test_rendered_module_round_trips():
    names, descriptive = frozenset({"HUDSONIA"}), frozenset({"ESQUINERA"})
    source = gen.render_module(names, descriptive, sources={"cadastre": "x", "places": "y"})
    namespace: dict = {}
    exec(compile(source, "complement_lexicon", "exec"), namespace)
    assert namespace["TOWER_NAME_WORDS"] == names
    assert namespace["DESCRIPTIVE_WORDS"] == descriptive


@pytest.mark.skipif(
    not (os.path.exists(gen.DEFAULT_CADASTRE) and os.path.exists(gen.DEFAULT_PLACES)),
    reason="cadastre artifact or basemaps are not present",
)
def test_committed_lexicon_is_in_sync_with_the_generator():
    names, descriptive = gen.build_lexicons(
        gen.read_cadastre_addresses(gen.DEFAULT_CADASTRE), gen.read_place_names(gen.DEFAULT_PLACES),
    )
    assert names == lex.TOWER_NAME_WORDS
    assert descriptive == lex.DESCRIPTIVE_WORDS
