"""Round-3: ``n_predios`` / ``n_lots_cert`` / ``n_lots_pcer`` / ``lot_conflict`` count the lots of ALL kept records
that resolve to a document (winning tier, merged lower tiers and unit rows), not only the winning tier.

All addresses are invented (numbers 900+).
"""

from __future__ import annotations

import pandas as pd

from nom_fixtures import nom_frame, npn, row

from cali_address.index_union import EXTRA_COLUMNS, build_nomenclatura, union_docs

ADDRESS = "KR 932B # 946 - 931"


def _doc(*rows):
    docs = build_nomenclatura(nom_frame(*rows)).docs
    assert len(docs) == 1
    return docs.iloc[0]


def test_new_columns_are_the_last_extra_columns_in_order():
    assert EXTRA_COLUMNS[-4:] == ["spread_m", "n_lots_cert", "n_lots_pcer", "lot_conflict"]
    docs = union_docs(build_nomenclatura(nom_frame(row(1, ADDRESS))).docs)
    assert list(docs.columns[-3:]) == ["n_lots_cert", "n_lots_pcer", "lot_conflict"]
    assert docs["lot_conflict"].dtype == bool
    assert docs["n_lots_cert"].dtype == "int64" and docs["n_lots_pcer"].dtype == "int64"


def test_single_lot_single_record():
    d = _doc(row(1, ADDRESS, npn_value=npn(lot="0917")))
    assert (d["n_predios"], d["n_lots_cert"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (1, 1, 0, False)


def test_cert_and_pcer_in_the_same_lot_do_not_conflict():
    d = _doc(row(1, ADDRESS, estado="cert", npn_value=npn(lot="0917", unit="000000001")),
             row(2, ADDRESS, estado="pcer", npn_value=npn(lot="0917", unit="000000002")))
    assert (d["n_predios"], d["n_lots_cert"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (1, 1, 1, False)


def test_cert_and_pcer_in_different_lots_flag_a_conflict_and_keep_the_cert_npn():
    d = _doc(row(1, ADDRESS, estado="cert", npn_value=npn(lot="0917")),
             row(2, ADDRESS, estado="pcer", npn_value=npn(lot="0918")))
    assert (d["n_predios"], d["n_lots_cert"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (2, 1, 1, True)
    assert d["numero_predial_nacional"] == npn(lot="0917")
    assert d["estado_nomenclatura"] == "cert"


def test_unit_rows_in_another_lot_still_count():
    d = _doc(row(1, ADDRESS, npn_value=npn(lot="0917")),
             row(2, ADDRESS + " AP 101", interior="AP 101", npn_value=npn(lot="0918", unit="000000001")))
    assert d["n_predios"] == 2 and d["n_lots_cert"] == 2 and not d["lot_conflict"]


def test_pcer_unit_row_in_another_lot_counts_for_pcer_and_conflicts():
    d = _doc(row(1, ADDRESS, estado="cert", npn_value=npn(lot="0917")),
             row(2, ADDRESS + " AP 101", interior="AP 101", estado="pcer", npn_value=npn(lot="0918", unit="000000001")))
    assert (d["n_predios"], d["n_lots_cert"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (2, 1, 1, True)


def test_invalid_npns_never_count():
    d = _doc(row(1, ADDRESS, estado="cert", npn_value=npn(lot="0917")),
             row(2, ADDRESS, estado="pcer", npn_value=npn(lot="0918")[:29]),
             row(3, ADDRESS, estado="pcer", npn_value=""))
    assert (d["n_predios"], d["n_lots_cert"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (1, 1, 0, False)


def test_only_invalid_npns_give_n_predios_one_and_no_lots():
    d = _doc(row(1, ADDRESS, npn_value=npn()[:28]))
    assert (d["n_predios"], d["n_lots_cert"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (1, 0, 0, False)


def test_pcer_only_doc_counts_its_lots_without_conflict():
    d = _doc(row(1, ADDRESS, estado="pcer", npn_value=npn(lot="0917")),
             row(2, ADDRESS, estado="pcer", npn_value=npn(lot="0918")))
    assert (d["n_predios"], d["n_lots_cert"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (2, 0, 2, False)


def test_several_cert_lots_and_a_pcer_subset_do_not_add_lots():
    d = _doc(row(1, ADDRESS, estado="cert", npn_value=npn(lot="0917")),
             row(2, ADDRESS, estado="cert", npn_value=npn(lot="0918")),
             row(3, ADDRESS, estado="pcer", npn_value=npn(lot="0917", unit="000000002")))
    assert d["n_predios"] == 2 and d["n_lots_cert"] == 2 and d["n_lots_pcer"] == 1


def test_cert_only_tiers_ignore_pcer_rows_entirely():
    docs = build_nomenclatura(nom_frame(row(1, ADDRESS, estado="cert", npn_value=npn(lot="0917")),
                                        row(2, ADDRESS, estado="pcer", npn_value=npn(lot="0918"))),
                              tiers=("cert",)).docs
    d = docs.iloc[0]
    assert (d["n_predios"], d["n_lots_pcer"], bool(d["lot_conflict"])) == (1, 0, False)


def test_empty_group_gives_typed_empty_frame_with_new_columns():
    docs = union_docs(build_nomenclatura(nom_frame()).docs)
    assert docs.empty
    assert list(docs.columns[-3:]) == ["n_lots_cert", "n_lots_pcer", "lot_conflict"]
    assert docs["lot_conflict"].dtype == bool


def test_counts_are_independent_of_input_order():
    rows = [row(1, ADDRESS, estado="cert", npn_value=npn(lot="0917")),
            row(2, ADDRESS, estado="pcer", npn_value=npn(lot="0918")),
            row(3, ADDRESS + " AP 1", interior="AP 1", estado="pcer", npn_value=npn(lot="0919", unit="000000001"))]
    a = build_nomenclatura(nom_frame(*rows)).docs
    b = build_nomenclatura(nom_frame(*reversed(rows))).docs
    pd.testing.assert_frame_equal(a, b)
    assert a.iloc[0]["n_predios"] == 3
