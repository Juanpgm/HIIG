"""Grammar rescue (``cali_address.grammar_rescue``) for addresses the rule parser rejects.

Edge cases pinned: every new pattern parses; the same text without the prefix parses identically; a prefix that must NOT
be stripped keeps the address NO_PARSEABLE; place names without digits stay NO_PARSEABLE; towers / blocks are never
dropped; the verified street-name table (variants, accents, case, ambiguous names, names inside other names); degenerate
input; idempotence; determinism; and the invariant that an address which parses today is never touched.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address import grammar_rescue  # noqa: E402
from cali_address.grammar_rescue import rescue_candidates, rescue_parse  # noqa: E402
from cali_address.parser import canonical, parse_address  # noqa: E402
from cali_address.service import normalize_strict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rescued_canonical(raw: str) -> str | None:
    hit = rescue_parse(raw)
    return None if hit is None else canonical(hit[1])


# --------------------------------------------------------------------------
# 1. every new pattern is positive
# --------------------------------------------------------------------------
POSITIVE = [
    ("EDIFICIO PANOMARICO KR 987 # 654-21, Eucarístico, Cali", "KR 987 # 654 - 21"),
    ("Palacio Nacional de Cali CL 876 # 543-32", "CL 876 # 543 - 32"),
    ("Velodromo Alcides Nieto Patiño cra 765 #432-51", "KR 765 # 432 - 51"),
    ("ZUMOS Avenida 5ta norte No 321 28", "AV 5 NORTE # 321 - 28"),
    ("TORRE D CALLE 2a # 678b-30", "CL 2 A # 678 B - 30 TO D"),
    ("CAR 731 # 612C-48", "KR 731 # 612 C - 48"),
    ("Avd 3 OE # 108-153", "AV 3 OESTE # 108 - 153"),
    ("AVD 3 OESTE # 108-153", "AV 3 OESTE # 108 - 153"),
    ("Carrera primera B1 #613-09 bloque 21", "KR 1 B 1 # 613 - 09 BLQ 21"),
    ("Calle segunda # 711-20", "CL 2 # 711 - 20"),
    ("Avenida sexta oeste # 722-23", "AV 6 OESTE # 722 - 23"),
    ("AV, 3H # 733-96", "AV 3 H # 733 - 96"),
    ("Ra 741 # 614B-16", "KR 741 # 614 B - 16"),
]


@pytest.mark.parametrize("raw,expected", POSITIVE)
def test_new_patterns_parse(raw, expected):
    assert not parse_address(raw).parse_ok, "the pattern must be one the grammar rejects today"
    assert rescued_canonical(raw) == expected


def test_same_text_without_the_prefix_parses_identically():
    prefixed = rescued_canonical("EDIFICIO PANOMARICO KR 987 # 654-21")
    bare = canonical(parse_address("KR 987 # 654-21"))
    assert prefixed == bare == "KR 987 # 654 - 21"


@pytest.mark.parametrize("raw", ["KR 987 # 654-21", "CL 876 # 543-32", "AV 3 OESTE # 108-153", "Calle 5 # 752"])
def test_text_that_parses_is_not_a_rescue_input_of_the_service(raw):
    """The service only rescues rows the grammar rejected: a parseable address never goes through the rescue."""
    assert parse_address(raw).parse_ok


# --------------------------------------------------------------------------
# 2. prefixes that must NOT be stripped / place names stay unparseable
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "Km 99 vereda Ficticia alta",
        "Tienda el Crucero",
        "Finca Ecoparque Sinai",
        "CASA 12 CALLE 5 # 761-4",          # a prefix with digits is never guessed away
        "Ale 5B 2 # 762 - 35 2o piso",
        "HOTEL CALLE",                    # the remainder does not parse
        "Ra 743 calle 5b # 744-16",         # two streets: never guess which one RA is
        "EDIFICIO LOS CERROS",
    ],
)
def test_nothing_to_rescue_stays_unparseable(raw):
    assert rescue_parse(raw) is None


# --------------------------------------------------------------------------
# 3. towers and blocks are never dropped (S26)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,unit",
    [
        ("TORRE D CALLE 2a # 678b-30", "TO D"),
        ("Edificio Mirador TORRE 3 CALLE 5 # 751-10", "TO 3"),
        ("BLOQUE B KR 987 # 654-21", "BLQ B"),
        ("hotel Sol TO 2 CL 876 # 543-32", "TO 2"),
    ],
)
def test_tower_or_block_in_the_prefix_moves_to_the_complement(raw, unit):
    hit = rescue_parse(raw)
    assert hit is not None
    assert canonical(hit[1]).endswith(unit)
    assert hit[1].complement and unit.split()[1] in hit[1].complement


def test_tower_value_that_is_a_connector_is_not_a_tower():
    # "TORRE DE ..." is a building name, not tower D
    hit = rescue_parse("TORRE DE CALI CL 876 # 543-32")
    assert hit is not None and canonical(hit[1]) == "CL 876 # 543 - 32"


# --------------------------------------------------------------------------
# 4. degenerate input, idempotence, determinism
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, "", "   ", "\t\n", 12345, "@@@", "ÁÉÍÓÚ ñ ü"])
def test_degenerate_inputs_never_raise_and_give_nothing(value):
    assert rescue_candidates(value) == [] or rescue_parse(value) is None


def test_very_long_input_is_bounded():
    assert rescue_candidates("EDIFICIO " * 500 + "KR 987 # 654-21") == []
    assert rescue_parse("X" * 100000) is None


def test_non_ascii_and_accents():
    assert rescued_canonical("Edificio Eucarístico Ñandú CALLE 876 # 543-32") == "CL 876 # 543 - 32"
    assert rescued_canonical("EDIFICIO ÁLAMOS Avd 3 OE # 108-153") == "AV 3 OESTE # 108 - 153"


@pytest.mark.parametrize("raw,_", POSITIVE)
def test_idempotent(raw, _):
    hit = rescue_parse(raw)
    again = rescue_parse(canonical(hit[1]))
    # the canonical output already parses, so it needs no rescue and normalizes to itself
    assert again is None
    assert canonical(parse_address(canonical(hit[1]))) == canonical(hit[1])


def test_deterministic():
    runs = [[rescue_candidates(r) for r, _ in POSITIVE] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_candidates_are_unique_and_never_the_input():
    for raw, _ in POSITIVE:
        c = rescue_candidates(raw)
        assert len(c) == len(set(c))
        assert grammar_rescue._fold(raw) not in c


# --------------------------------------------------------------------------
# 5. street-name table
# --------------------------------------------------------------------------
@pytest.fixture
def names(monkeypatch):
    table = (
        ("AUTOPISTA SUR ORIENTAL", "CL 99"),
        ("ROOSEVELT", "CL 5"),
        ("ROOSVELT", "CL 5"),
        ("AUTOPISTA SUR", "KR 8"),
        ("GUADALUPE", "KR 56"),
    )
    monkeypatch.setattr(grammar_rescue, "VIA_NAMES", table)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Av. Roosevelt # 763-25", "CL 5 # 763 - 25"),
        ("AV ROOSEVELT #763-25, 3 de Julio, Cali", "CL 5 # 763 - 25"),
        ("avenida roosevelt # 763-25", "CL 5 # 763 - 25"),
        ("Avenida Roosvelt 763-25", "CL 5 # 763 - 25"),          # spelling variant listed in the table
        ("AV. ROOSEVELT # 763 - 25", "CL 5 # 763 - 25"),
        ("Autopista sur # 771-01", "KR 8 # 771 - 01"),
        ("AUTOPISTA SUR ORIENTAL # 771 - 01", "CL 99 # 771 - 01"),   # the longer name wins
        ("Avenida Guadalupe 781-157", "KR 56 # 781 - 157"),
    ],
)
def test_verified_names_resolve(names, raw, expected):
    assert not parse_address(raw).parse_ok
    assert rescued_canonical(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Av. Roosevelts # 763-25",            # not the listed spelling
        "AV ROSEVELT # 763-25",               # misspelling absent from the table: unresolved, never fuzzy-guessed
        "AUTOPISTA SUR NUEVA # 771-01",       # the name continues into another street name
        "BARRIO ROOSEVELT",                  # no door
        "Condominio Guadalupe Real",         # a place that merely contains the name
        "Roosevelt",
    ],
)
def test_names_outside_the_table_or_inside_other_names_stay_unresolved(names, raw):
    assert rescue_parse(raw) is None


def test_name_substitution_does_not_alter_text_with_other_content(names):
    # the substituted street must be followed by the door the user wrote
    assert rescued_canonical("Av Roosevelt # 791-62 LOCAL 1") == "CL 5 # 791 - 62 LC 1"


def test_shipped_table_contains_only_reviewed_verified_names():
    from cali_address.via_names import VIA_NAMES

    path = ROOT / "src" / "cali_address" / "data" / "via_names_review.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert {r["status"] for r in rows} <= {"verified", "ambiguous", "rejected"}
    verified = {v: r["canonical_via"] for r in rows if r["status"] == "verified" for v in r["variants"].split("|")}
    assert dict(VIA_NAMES) == verified
    for r in rows:
        if r["status"] == "verified":
            assert int(r["doors"]) >= 10 and r["canonical_via"]
        else:
            assert r["reason"], "ambiguous / rejected names must say why"


def test_generated_module_is_in_sync_with_the_review_csv():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_via_names

    assert build_via_names.render(build_via_names.entries()) == (ROOT / "src" / "cali_address" / "via_names.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 6. service integration (stub normalizer, no model)
# --------------------------------------------------------------------------
def test_service_rescues_a_rejected_row_and_leaves_parseable_rows_alone(stub_normalizer):
    stub = stub_normalizer([{"direccion": "KR 987 # 654 - 21", "predial": "P1"}])
    stats: dict = {}
    out = normalize_strict(stub, ["EDIFICIO PANOMARICO KR 987 # 654-21", "KR 987 # 654-21", "Tienda el Crucero"], gazetteer=None, stats=stats)
    assert list(out["estado"]) != []
    assert out.iloc[0]["estado"] != "NO_PARSEABLE"
    assert out.iloc[0]["direccion_normalizada"]
    assert out.iloc[1]["estado"] != "NO_PARSEABLE"
    assert out.iloc[2]["estado"] == "NO_PARSEABLE"
    assert stats["rescatadas_gramatica"] == 1


def test_service_result_of_a_parseable_row_does_not_depend_on_the_rescue(stub_normalizer, monkeypatch):
    stub = stub_normalizer([{"direccion": "KR 987 # 654 - 21", "predial": "P1"}])
    with_rescue = normalize_strict(stub, ["KR 987 # 654-21"], gazetteer=None)
    monkeypatch.setattr("cali_address.service._rescue_parse", lambda *a, **k: None)
    without = normalize_strict(stub_normalizer([{"direccion": "KR 987 # 654 - 21", "predial": "P1"}]), ["KR 987 # 654-21"], gazetteer=None)
    assert with_rescue.drop(columns=[]).equals(without)


def test_service_unrescuable_row_keeps_the_original_motive(stub_normalizer):
    stub = stub_normalizer([{"direccion": "KR 987 # 654 - 21"}])
    out = normalize_strict(stub, ["Km 99 vereda Ficticia alta"], gazetteer=None)
    assert out.iloc[0]["estado"] == "NO_PARSEABLE"
    assert out.iloc[0]["motivo"]
