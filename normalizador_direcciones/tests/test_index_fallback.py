"""Cadastre fallback tier: selection rule, document frame, own centroids (invented fixtures, numbers 900+).

The rule is deterministic and has no tunable parameter: a cadastre doc is appended iff its valid lot (NPN[:21]) is absent
from every nomenclature source of the base index AND its ``doc_key`` collides with no existing doc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nom_fixtures import catastro_docs_frame, npn

from cali_address import index_fallback as F
from cali_address.index_union import COMPAT_COLUMNS, EXTRA_COLUMNS, doc_key


def lot_npn(lot: str, unit: str = "000000001") -> str:
    return npn(lot=lot, unit=unit)


def cat_frame(entries, start_oid: int = 70001) -> pd.DataFrame:
    return catastro_docs_frame(entries, start_oid=start_oid)


def keyed(entries, **kw) -> pd.DataFrame:
    from cali_address.index_union import catastro_key_table

    return catastro_key_table(cat_frame(entries, **kw))


# ---------------------------------------------------------------------------------------------------- select_fallback

def test_valid_lot_accepts_only_well_formed_cadastre_codes():
    good = lot_npn("0917")
    assert F.valid_lot(good) == good[:21]
    assert F.valid_lot(good[:21]) == good[:21]                       # a bare 21-character lot is still a lot
    for bad in (None, float("nan"), "", "   ", "7600101", "X" * 30, "99999" + good[5:], good[:20] + "A" + good[21:], 12345):
        assert F.valid_lot(bad) == "", repr(bad)


def test_select_adds_only_docs_whose_lot_is_absent():
    cat = keyed([("CL 901 # 931 - 912", lot_npn("0917")), ("CL 902 # 931 - 912", lot_npn("0918")),
                 ("CL 903 # 931 - 912", lot_npn("0919"))])
    present = {F.valid_lot(lot_npn("0917"))}
    mask, stats = F.select_fallback(cat, present, existing_keys=set())
    assert mask.tolist() == [False, True, True]
    assert stats["candidates"] == 3 and stats["selected"] == 2
    assert stats["skipped_lot_present"] == 1 and stats["skipped_invalid_lot"] == 0 and stats["collisions_skipped"] == 0


def test_select_skips_a_doc_key_that_already_exists_and_counts_the_collision():
    cat = keyed([("CL 901 # 931 - 912", lot_npn("0917")), ("CL 902 # 931 - 912", lot_npn("0918"))])
    mask, stats = F.select_fallback(cat, set(), existing_keys={doc_key("CL 901 # 931 - 912")})
    assert mask.tolist() == [False, True]
    assert stats["collisions_skipped"] == 1 and stats["selected"] == 1


def test_select_collision_inside_the_cadastre_keeps_the_first():
    cat = keyed([("CL 901 # 931 - 912", lot_npn("0917")), ("CL 901 # 931-912", lot_npn("0918"))])   # same doc_key
    assert cat["doc_key"].nunique() == 1
    mask, stats = F.select_fallback(cat, set(), existing_keys=set())
    assert mask.tolist() == [True, False] and stats["collisions_skipped"] == 1


def test_select_ignores_docs_without_a_valid_lot_and_counts_them():
    cat = keyed([("CL 901 # 931 - 912", ""), ("CL 902 # 931 - 912", "123"), ("CL 903 # 931 - 912", lot_npn("0919"))])
    mask, stats = F.select_fallback(cat, set(), existing_keys=set())
    assert mask.tolist() == [False, False, True]
    assert stats["skipped_invalid_lot"] == 2 and stats["skipped_lot_present"] == 0


def test_select_lot_rule_is_against_the_base_only_not_incremental():
    """Several cadastre addresses of ONE absent lot are all added: the lot set never grows during selection."""
    cat = keyed([(f"KR 9{i} # 931 - 912", lot_npn("0920", unit=f"{i:09d}")) for i in range(1, 4)])
    mask, stats = F.select_fallback(cat, set(), existing_keys=set())
    assert mask.tolist() == [True, True, True] and stats["selected"] == 3


def test_select_empty_inputs_add_nothing():
    empty = keyed([("CL 901 # 931 - 912", lot_npn("0917"))]).iloc[0:0]
    mask, stats = F.select_fallback(empty, set(), existing_keys=set())
    assert mask.shape == (0,) and stats["selected"] == 0 and stats["candidates"] == 0
    mask, stats = F.select_fallback(keyed([("CL 901 # 931 - 912", lot_npn("0917"))]), {F.valid_lot(lot_npn("0917"))}, set())
    assert mask.tolist() == [False] and stats["selected"] == 0


def test_select_is_deterministic():
    cat = keyed([(f"CL 9{i:02d} # 931 - 912", lot_npn(f"09{i:02d}")) for i in range(10, 30)])
    first = F.select_fallback(cat, {F.valid_lot(lot_npn("0915"))}, set())
    second = F.select_fallback(cat.copy(), {F.valid_lot(lot_npn("0915"))}, set())
    assert first[0].tolist() == second[0].tolist() and first[1] == second[1]


def test_present_lots_unions_docs_units_and_extra():
    docs = pd.DataFrame({"numero_predial_nacional": [lot_npn("0917"), None, "short"]})
    units = pd.DataFrame({"numero_predial_nacional": [lot_npn("0918", unit="000000002"), lot_npn("0917")]})
    got = F.present_lots(docs, units, extra={F.valid_lot(lot_npn("0919")), "junk"})
    assert got == {F.valid_lot(lot_npn(x)) for x in ("0917", "0918", "0919")} | {"junk"}
    assert F.present_lots(docs, None) == {F.valid_lot(lot_npn("0917"))}
    assert F.present_lots(docs.iloc[0:0], units.iloc[0:0]) == set()


# ------------------------------------------------------------------------------------------------ fallback_docs frame

def _base_docs(n: int = 3) -> pd.DataFrame:
    """A base docs frame in the nomenclature layout (production columns first, extras after)."""
    from cali_address.index_union import catastro_key_table

    cat = catastro_key_table(cat_frame([(f"CL 9{i:02d} # 931 - 912", lot_npn("0917", unit=f"{i:09d}")) for i in range(n)]))
    base = cat[COMPAT_COLUMNS].copy()
    base["objectid"] = base["objectid"] + 10_000_000
    base["doc_id"] = np.arange(n, dtype="int64")
    for col in EXTRA_COLUMNS:
        base[col] = pd.Series([None] * n, dtype="str")
    base["fuente"] = "nomenclatura_cert"
    base["estado_nomenclatura"] = "cert"
    for col, value, dtype in (("n_units", 1, "int64"), ("source_objectid", 1, "int64"), ("parse_ok", True, "bool"),
                              ("spread_m", 0.0, "float64"), ("n_lots_cert", 1, "int64"), ("n_lots_pcer", 0, "int64"),
                              ("lot_conflict", False, "bool")):
        base[col] = pd.Series([value] * n, dtype=dtype)
    base["doc_key"] = cat["doc_key"].astype("str")
    base["address_key"] = cat["address_key"].astype("str")
    return base[COMPAT_COLUMNS + EXTRA_COLUMNS]


def test_fallback_docs_have_the_base_schema_and_neutral_extras():
    base = _base_docs(3)
    cat = keyed([("CL 905 # 931 - 912", lot_npn("0918")), ("KR 906 # 931 - 912", lot_npn("0919"))], start_oid=70001)
    out = F.fallback_docs(cat, base, first_doc_id=len(base))
    assert list(out.columns) == list(base.columns)
    assert out.dtypes.astype(str).to_dict() == base.dtypes.astype(str).to_dict()
    assert out["doc_id"].tolist() == [3, 4]
    assert out["fuente"].tolist() == ["catastro", "catastro"]
    assert out["estado_nomenclatura"].isna().all()
    assert out["objectid"].tolist() == [70001 + F.CATASTRO_OBJECTID_OFFSET, 70002 + F.CATASTRO_OBJECTID_OFFSET]
    assert out["source_objectid"].tolist() == [70001, 70002]
    for col in ("direccion", "numero_predial_nacional", "manzana", "comuna", "barrio", "barrio_code", "centroid_lon",
                "centroid_lat", "n_predios", "manzana_nunique", "x_m", "y_m"):
        assert out[col].tolist() == cat[col].tolist(), col
    assert out["n_units"].tolist() == [0, 0] and out["spread_m"].tolist() == [0.0, 0.0]
    assert out["n_lots_cert"].tolist() == [0, 0] and out["n_lots_pcer"].tolist() == [0, 0]
    assert out["lot_conflict"].tolist() == [False, False]
    assert out["npn_choice"].tolist() == ["catastro", "catastro"]
    assert out["doc_key"].tolist() == cat["doc_key"].tolist() and out["address_key"].tolist() == cat["address_key"].tolist()
    assert out["parse_ok"].tolist() == cat["parse_ok"].tolist()


def test_fallback_objectids_never_collide_with_the_base():
    base = _base_docs(3)
    cat = keyed([("CL 905 # 931 - 912", lot_npn("0918"))], start_oid=10_000_001)   # same number as a base objectid
    out = F.fallback_docs(cat, base, first_doc_id=3)
    assert not set(out["objectid"]) & set(base["objectid"])


def test_fallback_docs_with_no_rows_keeps_columns_and_dtypes():
    base = _base_docs(2)
    out = F.fallback_docs(keyed([("CL 905 # 931 - 912", lot_npn("0918"))]).iloc[0:0], base, first_doc_id=2)
    assert len(out) == 0 and list(out.columns) == list(base.columns)
    assert out.dtypes.astype(str).to_dict() == base.dtypes.astype(str).to_dict()


def test_fallback_docs_refuse_a_base_without_the_nomenclature_extras():
    with pytest.raises(ValueError, match="columns"):
        F.fallback_docs(keyed([("CL 905 # 931 - 912", lot_npn("0918"))]), _base_docs(2)[COMPAT_COLUMNS], first_doc_id=2)


# --------------------------------------------------------------------------------------------------- own centroids

def test_fallback_own_centroids_match_by_objectid_and_npn_and_offset_the_id():
    cat = keyed([("CL 905 # 931 - 912", lot_npn("0918")), ("KR 906 # 931 - 912", lot_npn("0919")),
                 ("KR 907 # 931 - 912", lot_npn("0920"))], start_oid=70001)
    own = pd.DataFrame({
        "objectid": cat["objectid"].to_numpy(), "numero_predial_nacional": cat["numero_predial_nacional"].to_numpy(),
        "own_lon": [-76.1, -76.2, -76.3], "own_lat": [3.1, 3.2, 3.3]})
    own.loc[1, "numero_predial_nacional"] = "different"                  # NPN disagrees: never a wrong point
    own = own.iloc[[2, 1, 0]].reset_index(drop=True)                      # order is irrelevant: it is a lookup
    out = F.fallback_own_centroids(cat, own)
    assert list(out.columns) == ["objectid", "numero_predial_nacional", "own_lon", "own_lat"]
    assert out["objectid"].tolist() == (cat["objectid"] + F.CATASTRO_OBJECTID_OFFSET).tolist()
    assert out["numero_predial_nacional"].tolist() == cat["numero_predial_nacional"].tolist()
    assert out["own_lon"].iloc[0] == -76.1 and out["own_lat"].iloc[2] == 3.3
    assert np.isnan(out["own_lon"].iloc[1]) and np.isnan(out["own_lat"].iloc[1])


def test_fallback_own_centroids_missing_table_rows_are_nan_not_errors():
    cat = keyed([("CL 905 # 931 - 912", lot_npn("0918"))])
    out = F.fallback_own_centroids(cat, pd.DataFrame(columns=["objectid", "numero_predial_nacional", "own_lon", "own_lat"]))
    assert len(out) == 1 and np.isnan(out["own_lon"].iloc[0])
    assert len(F.fallback_own_centroids(cat.iloc[0:0], pd.DataFrame(
        columns=["objectid", "numero_predial_nacional", "own_lon", "own_lat"]))) == 0


# ------------------------------------------------------------------------------------------------------ extra pairs

def _pairs(cat, predials, ids=None):
    return pd.DataFrame({
        "raw_address": [f"raw {i}" for i in range(len(predials))], "address_key": [f"K{i}" for i in range(len(predials))],
        "dataset": "fasecolda", "gt_doc_id": list(range(len(predials))) if ids is None else ids, "gt_manzana": "m",
        "gt_predial": predials, "gt_comuna": "08", "source_row_id": [f"r{i}" for i in range(len(predials))]})


def _world_for_pairs():
    """Base docs (lot 0917 only) + a cadastre of three docs: one collides with the base (key), two are appended."""
    base = _base_docs(2)                                           # CL 900 / CL 901 # 931 - 912, lot 0917
    cat = keyed([("CL 900 # 931 - 912", lot_npn("0917", unit="000000000")),     # = base doc 0 (present lot)
                 ("KR 905 # 931 - 912", lot_npn("0918", unit="000000005")),     # lot absent -> appended
                 ("KR 906 # 931 - 912", lot_npn("0918", unit="000000006"))])    # same absent lot -> appended
    mask, _ = F.select_fallback(cat, F.present_lots(base), set(base["doc_key"]))
    added = F.fallback_docs(cat[mask].reset_index(drop=True), base, len(base))
    return base, cat, added


def test_extra_pairs_maps_dropped_pairs_to_the_appended_doc_with_the_exact_npn():
    base, cat, added = _world_for_pairs()
    pairs = _pairs(cat, [cat["numero_predial_nacional"][0], cat["numero_predial_nacional"][1], cat["numero_predial_nacional"][2]])
    new, report = F.extra_pairs(pairs, cat, base, added)
    assert new["gt_doc_id"].tolist() == [2, 3] and new["source_row_id"].tolist() == ["r1", "r2"]
    assert report == {"candidates": 2, "added": 2, "dropped_no_lot": 0, "dropped_ambiguous": 0}
    assert "__pos" not in new.columns and list(new.columns) == list(pairs.columns)


def test_extra_pairs_never_moves_a_pair_the_base_already_keeps_even_if_an_appended_doc_has_its_key():
    """A cadastre doc with a complement (own key, own lot) is appended; a pair that points at it but carries the lot of a
    base doc was remapped to that base doc by predial and must stay there."""
    base = _base_docs(1)
    cat = keyed([("CL 900 # 931 - 912 BLQ 1", lot_npn("0918", unit="000000007"))])        # old doc of the pair
    mask, _ = F.select_fallback(cat, F.present_lots(base), set(base["doc_key"]))
    added = F.fallback_docs(cat[mask].reset_index(drop=True), base, 1)
    assert len(added) == 1
    pairs = _pairs(cat, [base["numero_predial_nacional"][0]])                               # truth: the base doc's NPN
    new, report = F.extra_pairs(pairs, cat, base, added)
    assert len(new) == 0 and report["candidates"] == 0


def test_extra_pairs_resolves_several_docs_of_one_lot_by_exact_npn_then_by_key_and_else_drops_it():
    base = _base_docs(1)
    cat = keyed([("KR 905 # 931 - 912", lot_npn("0918", unit="000000005")), ("KR 906 # 931 - 912", lot_npn("0918", unit="000000006"))])
    mask, _ = F.select_fallback(cat, F.present_lots(base), set(base["doc_key"]))
    added = F.fallback_docs(cat[mask].reset_index(drop=True), base, 1)
    same_lot_other_npn = lot_npn("0918", unit="000000099")
    pairs = _pairs(cat, [lot_npn("0918", unit="000000006"), same_lot_other_npn, same_lot_other_npn], ids=[1, 1, 99])
    new, report = F.extra_pairs(pairs, cat, base, added)
    assert new["gt_doc_id"].tolist() == [2, 2]            # exact NPN -> doc 2; unknown NPN: the old doc's key (KR 906) -> doc 2
    assert report == {"candidates": 3, "added": 2, "dropped_no_lot": 0, "dropped_ambiguous": 1}   # old id 99 is unknown


def test_extra_pairs_counts_pairs_without_a_usable_lot():
    base, cat, added = _world_for_pairs()
    pairs = _pairs(cat, [None, "", "123", lot_npn("0999"), float("nan")], ids=[1, 1, 1, 1, 1])
    new, report = F.extra_pairs(pairs, cat, base, added)
    assert len(new) == 0 and report == {"candidates": 5, "added": 0, "dropped_no_lot": 5, "dropped_ambiguous": 0}


def test_extra_pairs_edge_inputs():
    base, cat, added = _world_for_pairs()
    empty, report = F.extra_pairs(_pairs(cat, []), cat, base, added)
    assert len(empty) == 0 and report["candidates"] == 0
    none_added, report = F.extra_pairs(_pairs(cat, [cat["numero_predial_nacional"][1]], ids=[1]), cat, base, added.iloc[0:0])
    assert len(none_added) == 0 and report["dropped_no_lot"] == 1
