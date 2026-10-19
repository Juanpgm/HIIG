"""Tests for the complement stage: the input's own building details survive.

The retriever resolves the base address (``<via> # <cross> - <plate>``); the
complement (TORRE F, BLQ 3, PISO 2, ...) is taken from the *input* and merged with
the cadastral complement by ``merge_complement``. Rules under test:

* input values win over the cadastre (TORRE F vs TORRE G must stay different);
* a cadastral-only chunk is never invented;
* a kind mentioned without a value is not a unit;
* the complement never changes accept/abstain gating or the precision level.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import (  # noqa: E402
    COMPLEMENT_CANON,
    canonical,
    extract_complement_units,
    merge_complement,
    parse_address,
)
from cali_address.service import OUTPUT_COLUMNS, normalize_strict  # noqa: E402

BASE = "KR 29 A # 36 - 87"


# ---------------------------------------------------------------------------
# COMPLEMENT_CANON extensions + canonical spelling of the rule complement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alias, canon",
    [
        ("PISO", "PISO"), ("PIS", "PISO"),
        ("AGRUPACION", "AGR"), ("AGRUP", "AGR"), ("AGR", "AGR"),
        ("SECTOR", "SEC"), ("SECT", "SEC"),
    ],
)
def test_new_complement_kinds_are_registered(alias, canon):
    assert COMPLEMENT_CANON[alias] == canon


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CL 5 # 38-10 T-F", "TO F"),
        ("CL 5 # 38-10 t-f", "TO F"),
        ("CL 5 # 38-10 TORRE F", "TO F"),
        ("CL 5 # 38-10 t-f ap-101", "TO F AP 101"),
        ("CL 5 # 38-10 Torre F Piso 3", "TO F PISO 3"),
        ("CL 5 # 38-10 AGRUPACION X", "AGR X"),
        ("CL 5 # 38-10 BL 3", "BLQ 3"),
    ],
)
def test_rule_complement_is_canonical(raw, expected):
    # The complement stage renders ``complement_text``; ``complement`` itself keeps the
    # frozen scoring spelling (see test_complement_scoring_invariance.py).
    from cali_address.parser import render_complement

    assert render_complement(parse_address(raw).complement_text) == expected


def test_rule_complement_hyphen_between_values_is_not_swallowed():
    # Only a hyphen that glues a kind to its value is removed.
    assert parse_address("CL 5 # 38-10 AP 101 - A").complement == "AP 101 - A"


# ---------------------------------------------------------------------------
# extract_complement_units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("TORRE F", [("TO", "F")]),
        ("TO F", [("TO", "F")]),
        ("T - F", [("TO", "F")]),
        ("t-f", [("TO", "F")]),
        ("torre f", [("TO", "F")]),
        ("ToRrE   F", [("TO", "F")]),
        ("TO 2 AP 101", [("TO", "2"), ("AP", "101")]),
        ("AP 101 TO 2", [("AP", "101"), ("TO", "2")]),
        ("BL 3", [("BLQ", "3")]),
        ("BLOQUE A", [("BLQ", "A")]),
        ("PISO 3", [("PISO", "3")]),
        ("SECTOR 2", [("SEC", "2")]),
        ("AGRUPACION X", [("AGR", "X")]),
        ("ED LAS PALMAS TO G", [("ED", "LAS PALMAS"), ("TO", "G")]),
        ("AP 101 A", [("AP", "101A")]),
        ("AP 101-A", [("AP", "101A")]),
    ],
)
def test_extract_units(text, expected):
    assert extract_complement_units(text) == expected


@pytest.mark.parametrize("empty", [None, "", "   ", "\t\n", " . - "])
def test_extract_units_empty_inputs(empty):
    assert extract_complement_units(empty) == []


@pytest.mark.parametrize("text", ["TORRE", "TO", "TO  ", "T-", "T -", "-F", "AP", "TORRE   "])
def test_extract_units_kind_without_value_is_not_a_unit(text):
    assert extract_complement_units(text) == []


def test_extract_units_valueless_kind_between_units():
    assert extract_complement_units("TO AP 101") == [("AP", "101")]


def test_extract_units_unicode_and_accents():
    assert extract_complement_units("TÓRRE Á") == [("TO", "A")]
    assert extract_complement_units("edificio ñandú") == [("ED", "NANDU")]


def test_extract_units_leading_zeros_normalized_but_zero_kept():
    assert extract_complement_units("TO 0002") == [("TO", "2")]
    assert extract_complement_units("LT 00") == [("LT", "0")]


def test_extract_units_free_text_after_a_unit_is_not_absorbed():
    assert extract_complement_units("TO F FRENTE A LA TIENDA DE DON PEPE") == [("TO", "F")]


def test_extract_units_unregistered_words_are_not_units():
    assert extract_complement_units("FRENTE AL PARQUE") == []
    assert extract_complement_units("SIN NOMBRE 3") == []


def test_extract_units_exact_duplicates_collapse():
    assert extract_complement_units("TO F TORRE F T-F") == [("TO", "F")]


def test_extract_units_same_kind_different_values_are_both_reported():
    assert extract_complement_units("TO F TO G") == [("TO", "F"), ("TO", "G")]


def test_extract_units_absurdly_long_value_token_is_rejected():
    assert extract_complement_units("TO " + "A" * 5000) == []


def test_extract_units_very_long_input_is_fast_and_bounded():
    text = " ".join(["AP 1 TO 2 BLQ A"] * 5000)
    start = time.perf_counter()
    out = extract_complement_units(text)
    assert time.perf_counter() - start < 2.0
    assert out == [("AP", "1"), ("TO", "2"), ("BLQ", "A")]


def test_extract_units_very_long_name_value_is_capped():
    out = extract_complement_units("ED " + " ".join(["PALMAS"] * 400))
    assert len(out) == 1 and out[0][0] == "ED"
    assert len(out[0][1].split()) <= 4


# ---------------------------------------------------------------------------
# Descriptive free text must never become a unit
# ---------------------------------------------------------------------------
NOISE = [
    "CASA ESQUINERA",
    "CASA DE DOS PISOS",
    "LOCAL COMERCIAL",
    "OFICINA PRINCIPAL",
    "LOTE ESQUINERO",
    "IN FRONT",
    "casa esquinera",
    "Casa Esquinera",
    "LOCAL cómercial",
    "Oficína Principal",
    "TORRE DE",
    "TO DE LA ESQUINA",
    "CASA DEL FRENTE",
    "APTO LA ESQUINA",
    "PISO UNO",
    "LC EL CENTRO",
]


def test_very_long_descriptive_complement_is_ignored_and_fast():
    text = "CASA " + "ESQUINERA " * 5000
    start = time.perf_counter()
    assert extract_complement_units(text) == []
    assert merge_complement(BASE, text).address == BASE
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("text", NOISE)
def test_extract_units_descriptive_words_after_a_kind_are_not_units(text):
    assert extract_complement_units(text) == []


@pytest.mark.parametrize("text", NOISE)
def test_merge_with_descriptive_complement_equals_no_complement(text):
    m = merge_complement(BASE, text)
    assert m.address == BASE
    assert m.complement is None
    assert m.conflicts == ()
    assert m.address == merge_complement(BASE, None).address


@pytest.mark.parametrize("text", NOISE[:6])
def test_descriptive_complement_does_not_override_the_cadastral_chunk(text):
    m = merge_complement(f"{BASE} CA 5", text)
    assert m.address == BASE


@pytest.mark.parametrize(
    "text, expected",
    [
        ("TO F", [("TO", "F")]),
        ("TO AB", [("TO", "AB")]),
        ("TO a", [("TO", "A")]),
        ("LC 2B", [("LC", "2B")]),
        ("OF 301", [("OF", "301")]),
        ("CASA 5", [("CA", "5")]),
        ("LOTE 0031", [("LT", "31")]),
        ("AP 101 A", [("AP", "101A")]),
        ("CASA 5 DE DOS PISOS", [("CA", "5")]),
        ("LOCAL 2 COMERCIAL", [("LC", "2")]),
        ("TORRE F LOCAL COMERCIAL", [("TO", "F")]),
    ],
)
def test_extract_units_short_codes_still_work(text, expected):
    assert extract_complement_units(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("AP 101 Y 102", [("AP", "101")]),
        ("PISO 1 Y 2", [("PISO", "1")]),
        ("AP 101 O 102", [("AP", "101")]),
        ("apto 101 y 102", [("AP", "101")]),
        ("TO Y", []),
        # audit 2026-10: ``O`` right after a tower kind is the tower letter O (``TORRE O``);
        # everywhere else (``AP 101 O 102``) it stays a connector
        ("TO O", [("TO", "O")]),
        ("AP O", []),
        ("CASA 5 A LA ESQUINA", [("CA", "5")]),
        ("AP 101 A LAS 5", [("AP", "101")]),
    ],
)
def test_extract_units_conjunctions_and_prepositions_are_not_unit_letters(text, expected):
    assert extract_complement_units(text) == expected


@pytest.mark.parametrize("text", ["AP 101 Y 102", "PISO 1 Y 2"])
def test_merge_with_conjunction_emits_no_trailing_y(text):
    assert " Y" not in merge_complement(BASE, text).address


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ED. TORRE DE CALI", [("ED", "TORRE DE CALI")]),
        ("EDIFICIO TORRE DE CALI", [("ED", "TORRE DE CALI")]),
        ("edificio torre de cali", [("ED", "TORRE DE CALI")]),
        ("CONJ TORRES DE COMFANDI", [("CONJ", "TORRES DE COMFANDI")]),
        ("ED LAS PALMAS TORRE G", [("ED", "LAS PALMAS"), ("TO", "G")]),
        ("ED TORRE DE CALI TORRE G", [("ED", "TORRE DE CALI"), ("TO", "G")]),
        ("EDIFICIO TORRE DE CALI APTO 301", [("ED", "TORRE DE CALI"), ("AP", "301")]),
        ("ED TORRE F", [("TO", "F")]),
        ("UR LOS ALAMOS CASA 4", [("UR", "LOS ALAMOS"), ("CA", "4")]),
    ],
)
def test_kind_word_inside_a_building_name_stays_in_the_name(text, expected):
    assert extract_complement_units(text) == expected


def test_merge_keeps_a_building_name_that_contains_a_kind_word():
    assert merge_complement(BASE, "ED. TORRE DE CALI").address == f"{BASE} ED TORRE DE CALI"


def test_render_complement_keeps_only_units_in_canonical_order():
    from cali_address.parser import render_complement

    assert render_complement("TO F FRENTE A LA TIENDA") == "TO F"
    assert render_complement("AP 101 TO 2") == "TO 2 AP 101"
    assert render_complement("TO F TO G") == "TO F"
    assert render_complement("ED. TORRE DE CALI") == "ED TORRE DE CALI"
    assert render_complement("CASA ESQUINERA") is None


@pytest.mark.parametrize("empty", [None, "", "   ", "\t\n", " . - "])
def test_render_complement_empty_inputs(empty):
    from cali_address.parser import render_complement

    assert render_complement(empty) is None


# ---------------------------------------------------------------------------
# merge_complement
# ---------------------------------------------------------------------------
def test_input_unit_absent_from_cadastre_is_kept():
    m = merge_complement(BASE, "TORRE F")
    assert m.address == f"{BASE} TO F"
    assert m.complement == "TO F"
    assert m.conflicts == ()


def test_torre_f_and_torre_g_on_the_same_base_differ():
    f = merge_complement(BASE, "TORRE F").address
    g = merge_complement(BASE, "TORRE G").address
    assert f != g
    assert f.endswith("TO F") and g.endswith("TO G")


def test_hyphenated_torre_matches_spelled_out_torre():
    assert merge_complement(BASE, "T - F").address == merge_complement(BASE, "TORRE F").address


def test_cadastre_only_complement_is_not_invented():
    m = merge_complement(f"{BASE} TO 2 AP 101", None)
    assert m.address == BASE
    assert m.complement is None


def test_cadastre_only_chunk_dropped_when_input_mentions_other_kind():
    m = merge_complement(f"{BASE} TO 2 AP 101", "TORRE 2")
    assert m.address == f"{BASE} TO 2"


def test_matching_cadastral_chunk_is_spelled_canonically():
    # Same unit, canonical spelling (no leading zeros), so OK rows read like non-OK rows.
    m = merge_complement("KR 72 # - LT 0031", "LOTE 31")
    assert m.address == "KR 72 # - LT 31"
    assert m.conflicts == ()


def test_same_kind_conflict_input_wins_and_is_flagged():
    m = merge_complement(f"{BASE} AP 101", "APTO 505")
    assert m.address == f"{BASE} AP 505"
    assert m.conflicts == ("AP",)


def test_conflict_only_flags_the_conflicting_kind():
    m = merge_complement(f"{BASE} TO 2 AP 101", "TORRE 2 APTO 505")
    assert m.address == f"{BASE} TO 2 AP 505"
    assert m.conflicts == ("AP",)


def test_input_extra_unit_is_added_next_to_cadastral_match():
    m = merge_complement(f"{BASE} TO 2", "TORRE 2 AP 101")
    assert m.address == f"{BASE} TO 2 AP 101"
    assert m.conflicts == ()


def test_same_kind_twice_in_input_keeps_first_and_flags():
    m = merge_complement(BASE, "TO F TO G")
    assert m.address == f"{BASE} TO F"
    assert m.conflicts == ("TO",)


def test_first_input_value_wins_even_when_the_cadastre_confirms_the_second():
    # Same rule as a non-OK row (first value per kind), so both rows agree; flagged.
    m = merge_complement(f"{BASE} AP 202", "AP 101 AP 202")
    assert m.address == f"{BASE} AP 101"
    assert m.conflicts == ("AP",)


def test_multiple_units_are_emitted_in_canonical_kind_order():
    a = merge_complement(BASE, "AP 101 TO 2").address
    b = merge_complement(BASE, "TO 2 AP 101").address
    assert a == b == f"{BASE} TO 2 AP 101"


def test_name_unit_with_tower():
    m = merge_complement(BASE, "EDIFICIO LAS PALMAS TORRE G")
    assert m.address == f"{BASE} ED LAS PALMAS TO G"


def test_unregistered_cadastral_leading_chunk_is_free_text_never_emitted():
    # ``BO`` is not a unit kind: it is dropped like any free text, on OK and non-OK rows.
    cadastral = "CL 6 H OESTE # KR 50 C - BO 002003 LT 0031"
    assert merge_complement(cadastral, "BO 2003").address == "CL 6 H OESTE # KR 50 C -"
    assert merge_complement(cadastral, None).address == "CL 6 H OESTE # KR 50 C -"
    assert merge_complement(cadastral, "BO 2003 LOTE 31").address == "CL 6 H OESTE # KR 50 C - LT 31"


def test_empty_plate_cadastre_takes_input_unit():
    assert merge_complement("KR 72 # -", "LOTE 6").address == "KR 72 # - LT 6"


@pytest.mark.parametrize("empty", [None, "", "   ", "\t\n"])
def test_empty_input_complement_yields_base_only(empty):
    m = merge_complement(f"{BASE} AP 101", empty)
    assert m.address == BASE and m.complement is None and m.conflicts == ()


@pytest.mark.parametrize("junk", ["TORRE", "TO  ", "T-", "FRENTE AL PARQUE"])
def test_malformed_input_complement_adds_nothing(junk):
    m = merge_complement(f"{BASE} TO F", junk)
    assert m.address == BASE
    assert merge_complement(BASE, junk).address == BASE


@pytest.mark.parametrize("bad", ["", "   ", "???", "SIN DIRECCION", "12345"])
def test_unparseable_cadastral_address_is_returned_unchanged(bad):
    m = merge_complement(bad, "TO F")
    assert m.address == bad and m.conflicts == ()


def test_none_cadastral_address_does_not_raise():
    assert merge_complement(None, "TO F").address == ""


def test_merge_is_idempotent():
    first = merge_complement(f"{BASE} TO 2", "TORRE 2 APTO 101")
    again = merge_complement(first.address, "TORRE 2 APTO 101")
    assert again.address == first.address


def test_merge_with_very_long_input_stays_bounded():
    start = time.perf_counter()
    m = merge_complement(BASE, " ".join(["TORRE F"] * 5000) + " " + "X" * 20000)
    assert time.perf_counter() - start < 2.0
    assert m.address == f"{BASE} TO F"


def test_merged_address_keeps_the_contract_shape():
    # `<via> # <cross> - <plate>[ <complement>]`
    m = merge_complement(BASE, "torre f apto 101")
    assert m.address == f"{BASE} TO F AP 101"
    assert m.address.startswith(BASE + " ")


# ---------------------------------------------------------------------------
# normalize_strict integration (stub normalizer)
# ---------------------------------------------------------------------------
def _run(stub_normalizer, query, cadastral=BASE, scores=None, **doc_extra):
    stub = stub_normalizer([{"direccion": cadastral, **doc_extra}], scores=scores)
    return normalize_strict(stub, [query], gazetteer=None).iloc[0]


def test_output_columns_add_complemento_normalizado_without_moving_the_rest():
    assert OUTPUT_COLUMNS == [
        "direccion_entrada", "estado", "motivo", "direccion_normalizada",
        "fuente_normalizacion", "numero_predial_nacional", "manzana", "lat", "lon",
        "confianza", "confiabilidad", "confiabilidad_manzana", "margen",
        "nivel_precision", "barrio_vereda", "comuna_corregimiento",
        "complemento_normalizado",
        # position tiers, appended last
        "position_tier", "position_radius_m", "position_source", "selected_doc_id", "override_rejected",
        "direccion_catastral",
        # fidelity of the output vs the raw input, then the catalogue status, appended last
        "fidelity_risk", "fidelity_flags", "catalog_status", "address_verified",
    ]


def test_ok_row_keeps_the_input_tower_even_if_the_cadastre_lacks_it(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87 torre F")
    assert row["estado"] == "OK"
    assert row["direccion_normalizada"] == f"{BASE} TO F"
    assert row["complemento_normalizado"] == "TO F"


def test_ok_rows_torre_f_and_g_differ(stub_normalizer):
    f = _run(stub_normalizer, "Cra 29A # 36-87 torre F")
    g = _run(stub_normalizer, "Cra 29A # 36-87 T-G")
    assert f["direccion_normalizada"] != g["direccion_normalizada"]
    assert f["numero_predial_nacional"] == g["numero_predial_nacional"]


def test_ok_row_cadastre_only_complement_is_not_invented(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87", cadastral=f"{BASE} TO 2 AP 101")
    assert row["direccion_normalizada"] == BASE
    assert row["complemento_normalizado"] is None


def test_ok_row_conflicting_unit_input_wins_and_is_noted(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87 apto 505", cadastral=f"{BASE} AP 101")
    assert row["estado"] == "OK"
    assert row["direccion_normalizada"] == f"{BASE} AP 505"
    assert row["complemento_normalizado"] == "AP 505"
    assert "complemento" in row["motivo"] and "AP" in row["motivo"]
    assert row["nivel_precision"] == "predio"


def test_complement_stage_keeps_row_columns_and_feeds_the_legacy_query_text(stub_normalizer):
    """The stub's scores are fixed, so equal ``confianza``/``margen`` alone prove nothing
    about scoring: what the real normalizer scores is the encoder text built from the
    query parse (``canonical(..., with_complement=True)``, the frozen scoring spelling).
    That text is recorded here and pinned to the pre-complement-stage spelling."""
    seen: list[str] = []

    class RecordingStub(type(stub_normalizer([{"direccion": BASE}]))):
        def score_components(self, queries, k=20):
            seen.extend(canonical(parse_address(q), style="spaced", with_complement=True) for q in queries)
            return super().score_components(queries, k=k)

    def run(query):
        return normalize_strict(RecordingStub([{"direccion": BASE}]), [query], gazetteer=None).iloc[0]

    plain = run("Cra 29A # 36-87")
    with_unit = run("Cra 29A # 36-87 torre F apto 101 piso 3")
    numbered = run("Cra 29A # 36-87 torre uno a apto 3-02 barrio el guabal")
    assert seen == [
        BASE,
        f"{BASE} TO F AP 101 PISO 3",
        f"{BASE} TO UNO A AP 3 - 02 BR EL GUABAL",
    ]
    for col in ("estado", "numero_predial_nacional", "manzana", "confianza",
                "margen", "nivel_precision", "lat", "lon"):
        a, b, c = plain[col], with_unit[col], numbered[col]
        assert (pd.isna(a) and pd.isna(b) and pd.isna(c)) or a == b == c, col
    assert plain["motivo"] == with_unit["motivo"] == numbered["motivo"]


def test_shared_address_keeps_complement_and_shared_note(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87 torre F", n_predios=3, manzana_nunique=1)
    assert row["estado"] == "OK"
    assert row["direccion_normalizada"] == f"{BASE} TO F"
    assert row["motivo"] == "direccion compartida por 3 predios"
    assert row["nivel_precision"] == "manzana"


def test_shared_address_across_manzanas_still_capped_at_direccion(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87 torre F", n_predios=3, manzana_nunique=2)
    assert row["nivel_precision"] == "direccion"
    assert row["direccion_normalizada"].endswith("TO F")


def test_sin_match_row_keeps_rule_canonical_with_complement(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87 t-f", scores=[[0.1]])
    assert row["estado"] == "SIN_MATCH"
    assert row["fuente_normalizacion"] == "reglas"
    assert row["direccion_normalizada"] == f"{BASE} TO F"
    assert row["complemento_normalizado"] == "TO F"
    assert pd.isna(row["numero_predial_nacional"])


def test_sin_match_rows_torre_f_and_g_differ(stub_normalizer):
    f = _run(stub_normalizer, "Cra 29A # 36-87 torre F", scores=[[0.1]])
    g = _run(stub_normalizer, "Cra 29A # 36-87 torre G", scores=[[0.1]])
    assert f["direccion_normalizada"] != g["direccion_normalizada"]


def test_sin_match_row_without_complement_has_no_complemento(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87", scores=[[0.1]])
    assert row["estado"] == "SIN_MATCH"
    assert row["complemento_normalizado"] is None


@pytest.mark.parametrize("query", ["asdf qwer", "???", "", None, "   "])
def test_unparseable_or_empty_rows_have_no_complemento(stub_normalizer, query):
    row = _run(stub_normalizer, query)
    assert row["estado"] != "OK"
    assert pd.isna(row["complemento_normalizado"])


def test_dedup_contract_columns_are_untouched(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87 torre F apto 101")
    # dedup.py parses `<via> # <cross> - <plate>[ <complement>]`.
    import re
    shape = re.compile(r"^(?P<via>.+?)\s#\s(?P<cross>.*?)\s-\s(?P<plate>\S+)(?P<rest>.*)$")
    m = shape.match(row["direccion_normalizada"])
    assert m and m.group("plate") == "87" and m.group("rest").strip() == "TO F AP 101"


# ---------------------------------------------------------------------------
# Non-OK rows render the same filtered units as OK rows
# ---------------------------------------------------------------------------
def test_non_ok_row_drops_free_text_like_ok_rows(stub_normalizer):
    query = "Cra 29A # 36-87 TO F FRENTE A LA TIENDA"
    ok = _run(stub_normalizer, query)
    non_ok = _run(stub_normalizer, query, scores=[[0.1]])
    assert ok["estado"] == "OK" and non_ok["estado"] == "SIN_MATCH"
    for row in (ok, non_ok):
        assert row["direccion_normalizada"] == f"{BASE} TO F"
        assert row["complemento_normalizado"] == "TO F"


@pytest.mark.parametrize("scores", [None, [[0.1]]])
@pytest.mark.parametrize("noise", ["casa esquinera", "LOCAL COMERCIAL", "oficina principal"])
def test_noisy_complement_yields_no_unit_on_ok_and_non_ok_rows(stub_normalizer, noise, scores):
    row = _run(stub_normalizer, f"Cra 29A # 36-87 {noise}", scores=scores)
    assert row["direccion_normalizada"] == BASE
    assert row["complemento_normalizado"] is None


def test_non_ok_row_orders_units_like_ok_rows(stub_normalizer):
    row = _run(stub_normalizer, "Cra 29A # 36-87 apto 101 torre 2", scores=[[0.1]])
    assert row["direccion_normalizada"] == f"{BASE} TO 2 AP 101"


def test_build_row_with_multi_document_stub_and_noisy_inputs(stub_normalizer):
    stub = stub_normalizer(
        [{"direccion": BASE, "predial": "P1"}, {"direccion": "CL 5 # 38 - 10", "predial": "P2"}],
        scores=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
    )
    queries = [
        "Cra 29A # 36-87 casa esquinera",
        "Cl 5 # 38-10 ED. Torre de Cristal",
        "Cl 5 # 38-10 LOCAL COMERCIAL torre G",
        "Cra 29A # 36-87 AP 101 Y 102",
        "Cra 29A # 36-87 casa de dos pisos",
    ]
    out = normalize_strict(stub, queries, gazetteer=None)
    assert list(out["estado"]) == ["OK"] * 5
    assert list(out["numero_predial_nacional"]) == ["P1", "P2", "P2", "P1", "P1"]
    assert list(out["direccion_normalizada"]) == [
        BASE,
        "CL 5 # 38 - 10 ED TORRE DE CRISTAL",
        "CL 5 # 38 - 10 TO G",
        f"{BASE} AP 101",
        BASE,
    ]
    comp = out["complemento_normalizado"]
    assert [None if pd.isna(c) else c for c in comp] == [None, "ED TORRE DE CRISTAL", "TO G", "AP 101", None]
