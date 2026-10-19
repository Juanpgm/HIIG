"""The complement of a matched (OK) row is compared and rendered like any other row.

* Conflicts are decided on extracted unit values, so free text after a value the
  cadastre also has (``TORRE F FRENTE AL PARQUE`` vs cadastral ``TO F``) is not a
  conflict, and a cadastral string never conflicts with its own complement.
* ``direccion_normalizada`` and ``complemento_normalizado`` carry the same canonical
  unit text on OK and non-OK rows for the same input (``TORRE 2`` is ``TO 2`` on
  both, never ``TO 02`` on one of them).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.parser import merge_complement, parse_address, render_complement  # noqa: E402
from cali_address.service import normalize_strict  # noqa: E402

BASE = "KR 29 A # 36 - 87"


# ---------------------------------------------------------------------------
# Finding: false "complemento distinto al catastro"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cadastral_complement, typed",
    [
        ("TO F", "TORRE F FRENTE AL PARQUE"),
        ("AP 101", "APTO 101 SEGUNDO PISO"),
        ("CA 17", "CASA 17 ESQUINERA"),
        ("AP 0101", "APTO 101"),
        ("TO 02", "TORRE DOS"),
        ("AP 101A", "APTO 101 - A"),
        ("TO B SO 1 GA 1", "TORRE B GARAJE 1"),
        # the cadastre abbreviates AGRUPACION as AGP
        ("SEC 4 AGP 1 TO A", "SECTOR 4 AGRUPACION 1 TORRE A"),
    ],
)
def test_input_repeating_the_cadastral_value_is_not_a_conflict(cadastral_complement, typed):
    m = merge_complement(f"{BASE} {cadastral_complement}", typed)
    assert m.conflicts == ()
    assert m.complement == render_complement(typed)


@pytest.mark.parametrize(
    "cadastral",
    [
        "CL 99 # 9 - 99 TO B SO 1 GA 1",
        "CL 99 # 9 - 99 AP 101 PQ 3",
        "CL 99 # 9 - 99 ED LAS PALMAS AP 0201",
        "CL 99 # 9 - 99 BLQ G CA 7",
        "CL 99 # 9 - 99 BO 002003 LT 0031",
        "CL 99 # 9 - 99 AP 301 302",
        "CL 99 # 9 - 99 MZ A LT 5",
        "KR 1 # 2 - 03 BR EL GUABAL",
        # no ``#``: the base is not a prefix, so the complement is read from the parse
        "AVENIDA 99 NORTE 9 N 99 EDIF.HOTEL CASA DEL OLMO HABITACION 999",
    ],
)
def test_a_cadastral_string_never_conflicts_with_its_own_complement(cadastral):
    m = merge_complement(cadastral, parse_address(cadastral).complement_text)
    assert m.conflicts == ()
    assert merge_complement(m.address, parse_address(cadastral).complement_text).address == m.address


@pytest.mark.parametrize(
    "cadastral_complement, typed, kinds",
    [
        ("TO F", "TORRE G", ("TO",)),
        ("AP 101", "APTO 102 SEGUNDO PISO", ("AP",)),
        ("AP 3-02", "APTO 3-03", ("AP",)),
        ("TO F AP 101", "TORRE G APTO 102", ("TO", "AP")),
    ],
)
def test_real_disagreements_are_still_conflicts_and_input_wins(cadastral_complement, typed, kinds):
    m = merge_complement(f"{BASE} {cadastral_complement}", typed)
    assert m.conflicts == kinds
    assert m.complement == render_complement(typed)


def test_same_kind_twice_in_the_input_is_still_a_conflict():
    m = merge_complement(BASE, "TO F TO G")
    assert m.complement == "TO F" and m.conflicts == ("TO",)


def test_cadastral_kind_absent_from_the_input_is_not_a_conflict_nor_emitted():
    m = merge_complement(f"{BASE} TO 2 AP 101", "PISO 3")
    assert m.complement == "PISO 3" and m.conflicts == ()


@pytest.mark.parametrize("bad", [None, "", "   ", "FRENTE AL PARQUE", "TORRE"])
def test_empty_or_unitless_input_against_a_cadastral_complement(bad):
    m = merge_complement(f"{BASE} TO F", bad)
    assert m.address == BASE and m.complement is None and m.conflicts == ()


_CADASTRE = os.path.join(ROOT, "artifacts", "catastro_docs.parquet")


@pytest.mark.skipif(not os.path.exists(_CADASTRE), reason="cadastre artifact is not present")
def test_cadastral_strings_with_a_complement_never_flag_themselves():
    """Every cadastral address that has a complement, merged with that complement."""
    addresses = pd.read_parquet(_CADASTRE, columns=["direccion"])["direccion"].dropna().astype(str)
    flagged = []
    for address in addresses.drop_duplicates():
        text = parse_address(address).complement_text
        if text and merge_complement(address, text).conflicts:
            flagged.append(address)
    assert not flagged, f"{len(flagged)} cadastral strings flag themselves"


# ---------------------------------------------------------------------------
# Finding: OK rows vs non-OK rows render the same unit differently
# ---------------------------------------------------------------------------
def _row(stub_normalizer, query, cadastral, scores=None):
    stub = stub_normalizer([{"direccion": cadastral}], scores=scores)
    return normalize_strict(stub, [query], gazetteer=None).iloc[0]


@pytest.mark.parametrize(
    "typed, cadastral_complement, expected",
    [
        ("TORRE 2", "TO 02", "TO 2"),
        ("APTO 101", "AP 0101", "AP 101"),
        ("APTO 101A", "AP 101 A", "AP 101A"),
        ("APTO 101 A", "AP 101A", "AP 101A"),
        ("TO B SO 1 GA 1", "TO B SO 1 GA 1", "TO B GA 1"),
        ("TORRE DOS APTO 3-02", "TO 2 AP 3-02", "TO 2 AP 3-02"),
        ("BARRIO EL GUABAL TORRE F", "TO F", "TO F"),
    ],
)
def test_ok_and_non_ok_rows_give_the_same_unit_text(stub_normalizer, typed, cadastral_complement, expected):
    query = f"Cra 29A # 36-87 {typed}"
    ok = _row(stub_normalizer, query, f"{BASE} {cadastral_complement}")
    non_ok = _row(stub_normalizer, query, f"{BASE} {cadastral_complement}", scores=[[0.1]])
    assert ok["estado"] == "OK" and non_ok["estado"] != "OK"
    for row in (ok, non_ok):
        assert row["direccion_normalizada"] == f"{BASE} {expected}", row["estado"]
        assert row["complemento_normalizado"] == expected, row["estado"]
    assert "complemento distinto" not in ok["motivo"]
