"""Opt-in input refinement (``cali_address.refine``): format-level cleanup, never a rewrite.

Only synthetic addresses are used here.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from conftest import make_stub  # noqa: E402

from cali_address import refine as refine_module  # noqa: E402
from cali_address.refine import RefineResult, refine_address  # noqa: E402
from cali_address.service import OUTPUT_COLUMNS, normalize_strict  # noqa: E402

QUADRANTS = {"N", "NTE", "NORTE", "S", "SUR", "O", "OESTE", "E", "ESTE"}
VIA_WORDS = {"CALLE", "CARRERA", "CL", "KR", "CRA", "AV", "DG", "TV", "CR", "CLL", "AVENIDA",
             "DIAGONAL", "TRANSVERSAL"}
GLUED = ["66C", "26G6", "26G6 72S1", "5b1", "94C1", "Av5an", "08CA", "1A9-250", "15BNORTE", "22NORTE"]


def r(text):
    return refine_address(text).text


def digits(s):
    return [c for c in s if c.isdigit()]


def digit_tokens(s):
    return re.findall(r"[^\W_]*\d[^\W_]*", s)


def word_tokens(s, vocab):
    return [t for t in re.findall(r"[^\W\d_]+", s.upper()) if t in vocab]


def quadrants(s):
    masked = re.sub(r"\bN\s*[°º░]", " ", s)
    return word_tokens(masked, QUADRANTS)


def assert_invariants(before, after):
    assert digits(before) == digits(after), (before, after)
    assert digit_tokens(before) == digit_tokens(after), (before, after)
    assert quadrants(before) == quadrants(after), (before, after)
    assert word_tokens(before, VIA_WORDS) == word_tokens(after, VIA_WORDS), (before, after)


# --------------------------------------------------------------------------- boundaries
def test_result_shape():
    res = refine_address("CL 5 # 10 - 20")
    assert isinstance(res, RefineResult)
    assert res.text == "CL 5 # 10 - 20" and res.changes == () and res.is_placeholder is False


@pytest.mark.parametrize("raw", ["", "   ", "\t\n", ".", "-", ";", "; ;", "0", "N/A", "NA", "S/D",
                                 "SIN ESPECIFICAR", "NO APLICA", " sin especificar "])
def test_placeholders_flagged(raw):
    res = refine_address(raw)
    assert res.is_placeholder is True
    assert res.text in ("", raw.strip(), " ".join(raw.split()))


def test_one_char_and_only_punctuation_and_only_digits():
    assert r("A") == "A"
    assert refine_address(",,,").is_placeholder is True
    assert refine_address("12345").is_placeholder is False and r("12345") == "12345"


def test_very_long_input_does_not_raise_and_keeps_digits():
    raw = "CL 5 # 10 - 20 " * 30
    out = r(raw)
    assert len(raw) > 300 and digits(out) == digits(raw)


def test_only_city_suffix_stays():
    assert r("CALI") == "CALI"
    assert r("VALLE DEL CAUCA") == "VALLE DEL CAUCA"


def test_city_inside_street_name_or_after_via_word_stays():
    assert r("AV CALI # 5 - 10") == "AV CALI # 5 - 10"
    assert r("CL 5 CON AV CALI") == "CL 5 CON AV CALI"
    assert r("CL 5 # 10 - 20 VALLE") == "CL 5 # 10 - 20 VALLE"  # bare VALLE is open text


@pytest.mark.parametrize("raw", [None, 12, 3.5, b"CL 5", object(), ["CL 5"], float("nan")])
def test_non_string_never_raises(raw):
    res = refine_address(raw)
    assert isinstance(res, RefineResult) and isinstance(res.text, str)


# --------------------------------------------------------------------------- unicode
def test_nbsp_and_whitespace_collapse():
    assert r("CL\xa05 # 10 -\t20\n") == "CL 5 # 10 - 20"
    assert r("  CL   5 #  10 - 20  ") == "CL 5 # 10 - 20"
    assert "space_normalize" in refine_address("CL  5").changes


def test_invisible_chars_stripped_only_in_digit_free_chunks():
    assert r("MUNICI\xadPIO CL 5") == "MUNICIPIO CL 5"
    assert r("CL ​5 # 10") == "CL ​5 # 10"
    assert r("CL 5​B # 10") == "CL 5​B # 10"


def test_mojibake_ordinal_repaired_and_sign_normalized():
    assert r("CL 5 NÂ° 10 - 20") == "CL 5 # 10 - 20"
    assert r("CL 5 NÂº 10 - 20") == "CL 5 # 10 - 20"
    assert r("5Âº") == "5Âº"  # digit-bearing chunk is never touched


def test_shade_char_as_degree_sign_only_after_lone_n_before_digit():
    assert r("CL 5 N░ 10 - 20") == "CL 5 # 10 - 20"
    assert r("CL 5 N░ APTO") == "CL 5 N░ APTO"


def test_fullwidth_and_nfd_are_left_alone():
    assert r("CL ５ # １０") == "CL ５ # １０"
    nfd = "CALÍ CL 5"
    assert r(nfd) == nfd


def test_accents_and_enye_untouched():
    assert r("CL 5 # 10 - 20 ÑANDÚ") == "CL 5 # 10 - 20 ÑANDÚ"


# --------------------------------------------------------------------------- glued tokens
@pytest.mark.parametrize("glued", GLUED)
def test_glued_tokens_byte_identical(glued):
    out = r(f"CL {glued} # 10 - 20, CALI")
    assert glued in out
    assert_invariants(f"CL {glued} # 10 - 20, CALI", out)


# --------------------------------------------------------------------------- quadrants
@pytest.mark.parametrize("quad", ["N", "NTE", "NORTE", "O", "OESTE", "S", "SUR", "E", "ESTE"])
def test_quadrant_variants_untouched(quad):
    for raw in (f"CL 5 {quad} # 10 - 20", f"CL 5 # 10 {quad} - 20", f"CL 5 # 10 - 20 {quad}",
                f"CL 5 {quad}, # 10 - 20"):
        assert_invariants(raw, r(raw))
    assert r(f"CL 5 # 10 - 20 {quad}") == f"CL 5 # 10 - 20 {quad}"


def test_quadrant_position_distinguishes_strings():
    a, b = r("CL 5 N # 10 - 20"), r("CL 5 # 10 N - 20")
    assert a != b and a == "CL 5 N # 10 - 20" and b == "CL 5 # 10 N - 20"


# --------------------------------------------------------------------------- number sign
@pytest.mark.parametrize("raw,expected", [
    ("CL 5 NO, 10-20", "CL 5 # 10-20"),
    ("CL 5 NO. 10-20", "CL 5 # 10-20"),
    ("CL 5 N° 10 - 20", "CL 5 # 10 - 20"),
    ("CL 5 Nº 10 - 20", "CL 5 # 10 - 20"),
    ("CL 5 NRO 10-20", "CL 5 # 10-20"),
    ("CL 5 NORTE NO, 10-20", "CL 5 NORTE # 10-20"),
    ("CL 5 B NO 10-20", "CL 5 B # 10-20"),
])
def test_number_sign_variants_main_position(raw, expected):
    res = refine_address(raw)
    assert res.text == expected and "numsign" in res.changes


@pytest.mark.parametrize("raw", [
    "CASA NO,4", "CL 5 CASA NO,4", "CL 5 # 10 - 20 CASA NO,4", "CL 5 - 10 NO 4", "CL 5 N, 10-20",
    "CL 5 NO APLICA", "CL 5 # 10 NO. 20",
])
def test_number_sign_not_applied_inside_complement(raw):
    assert r(raw) == raw


# --------------------------------------------------------------------------- commas / edges
@pytest.mark.parametrize("raw,expected", [
    ("CL 5, # 10 - 20", "CL 5 # 10 - 20"),
    ("KR 47, 9C-21", "KR 47 9C-21"),
    ("CL, 5 # 10-20", "CL 5 # 10-20"),
    ("CL 5 # 10, 20", "CL 5 # 10 20"),
    ("CL 5,, # 10", "CL 5 # 10"),
    ("CL 5 # 10 - 20, CALI", "CL 5 # 10 - 20"),
    ("CL 5 # 10-20, CALI, VALLE DEL CAUCA, COLOMBIA", "CL 5 # 10-20"),
    ("CL 5 # 10 - 20 SUR CALI", "CL 5 # 10 - 20 SUR"),
    ("CL 5 # 10 - 20 CALI VALLE", "CL 5 # 10 - 20"),
    ("CL 5 # 10 - 20 SANTIAGO DE CALI.", "CL 5 # 10 - 20"),
    ("; CL 5 # 10 - 20", "CL 5 # 10 - 20"),
    ('"CL 5 # 10 - 20"', "CL 5 # 10 - 20"),
    ("(CL 5 # 10 - 20)", "CL 5 # 10 - 20"),
    ("CL 5 # 10 - 20,", "CL 5 # 10 - 20"),
])
def test_cleanup_cases(raw, expected):
    assert r(raw) == expected


@pytest.mark.parametrize("raw", [
    "CL 5 # 10 - 20, APTO 301", "CL 5 # 10 - 20, 301", "CL 5 # 10, 3 PISO", "CL 3,4512/-76,5123",
    "CL 5 # 10 - 20 (CASA 5)", "CL 5 # 10 - 20 BARRIO LOS ANDES", "CL 5 # 10 - 20 EDIF CENTRAL",
    "CL 5 # 10 - 20 -", "CL 5 #  -", "CL 5 # 10 - 20, BARRIO CALI",
])
def test_open_text_and_ambiguous_commas_unchanged(raw):
    assert r(raw) == " ".join(raw.split())


def test_multi_address_text_unchanged_but_flagged():
    raw = "CL 5 # 10 - 20; CL 6 # 11 - 21"
    res = refine_address(raw)
    assert res.text == raw and "flag_multi_address" in res.changes
    assert r("CL 5 # 10 - 20 Y CL 6 # 11 - 21") == "CL 5 # 10 - 20 Y CL 6 # 11 - 21"


def test_change_ids_reported():
    assert refine_address("CL 5, # 10 - 20, CALI").changes == ("comma_to_space", "strip_city_suffix")


# --------------------------------------------------------------------------- properties
def _grid():
    rng = random.Random(7)
    vias = ["CL", "KR", "CRA", "AV", "CALLE", "DG", "TV", "CARRERA"]
    nums = ["5", "26", "5A", "28D", "66C", "26G6", "08CA", "1A9", "94C1", "5b1", "15BNORTE", "22NORTE"]
    quads = ["", " N", " NORTE", " SUR", " OESTE", " O", " E", " S", " NTE", " ESTE"]
    seps = [" # ", ", # ", " , ", " NO, ", " NO. ", " N° ", " Nº ", " NRO ", " ", ", ", ",, ", " N░ ", " NO "]
    plates = ["10 - 20", "10-20", "10, 20", "8B-20", "10", "1A9-250", "10 - ", "3,4512"]
    tails = ["", ",", ", CALI", " CALI", ", CALI, VALLE DEL CAUCA", ", APTO 301", " CASA NO,4", " (CASA 5)",
             "; CL 6 # 1 - 2", " BARRIO CALI", " COLOMBIA.", " SANTIAGO DE CALI", ", 301", " PISO 3"]
    heads = ["", "; ", '"', "  ", "\xa0", "(", "DIRECCION "]
    rows = []
    for _ in range(25000):
        v, n, q, s, p, t, h = (rng.choice(x) for x in (vias, nums, quads, seps, plates, tails, heads))
        rows.append(f"{h}{v} {n}{q}{s}{p}{t}")
    return rows


GRID = _grid()


def test_grid_hard_invariants_idempotence_determinism():
    refined_count = 0
    for raw in GRID:
        a = refine_address(raw)
        assert_invariants(raw, a.text)
        assert refine_address(a.text).text == a.text, raw
        assert refine_address(raw) == a
        refined_count += a.text != raw
    assert refined_count > len(GRID) // 10  # the grid really exercises the rules


@pytest.mark.parametrize("glued", GLUED)
def test_grid_glued_survive_everywhere(glued):
    for sep in (" # ", ", # ", " NO, ", " N° "):
        raw = f"; CL 5{sep}{glued} - 20, CALI"
        assert glued in r(raw)


def test_runtime_guard_falls_back_to_unrefined_when_a_rule_breaks_invariants(monkeypatch):
    def evil(text):
        return text.replace("10", "11"), ["evil"]

    monkeypatch.setattr(refine_module, "_RULES", [("evil", evil)])
    res = refine_address("CL 5 # 10 - 20")
    assert res.text == "CL 5 # 10 - 20" and res.changes == ()


def test_guard_rejects_quadrant_drop(monkeypatch):
    monkeypatch.setattr(refine_module, "_RULES", [("evil", lambda t: (t.replace(" N ", " "), ["evil"]))])
    assert r("CL 5 N # 10 - 20") == "CL 5 N # 10 - 20"


# --------------------------------------------------------------------------- opt-in wiring
_SET = ["CL 5 # 10 - 20", "CL 5, # 10 - 20", "KR 47, 9C-21", "CL 5 # 10 - 20, CALI", "CL 5 NO, 10-20",
        "; CL 5 # 10 - 20", "SIN ESPECIFICAR", "", "CL 5 N 10 - 20  ", "CL 5 NÂ° 10 - 20", "CL 9 # 1 - 1",
        "texto libre"]
# Re-pinned for the position tiers: five columns are appended and the two one-sided via-letter
# rows (nivel esquina -> manzana) changed on purpose. Every other legacy cell is unchanged.
# Re-pinned in round 5: the geometric radii (predio 15, ...) and the parser reading ``CL 5, # 10 - 20`` as one address
# (that row went from SIN_MATCH "CL 5" to OK); the diff against the previous golden is exactly those cells.
# Re-pinned in round 7: ``CL 5 NÂ° 10 - 20`` (double-encoded ``N°``) no longer reads as the letters NA; it is the number sign.
_GOLDEN = "9633a9b6b3071b2f5c1ce5d7a97494580bc9fa4a4ab6733eccbb343cec960cfa"


def _stub():
    return make_stub([{"direccion": "CL 5 # 10 - 20"}, {"direccion": "KR 47 # 9C - 21"}], threshold=0.5)


def test_default_off_is_byte_identical_to_pre_change_output():
    df = normalize_strict(_stub(), _SET)
    assert list(df.columns) == OUTPUT_COLUMNS
    # the fidelity and catalogue columns are new (appended last): the historical columns must hash to the golden value
    historical = df.drop(columns=["fidelity_risk", "fidelity_flags", "catalog_status", "address_verified"])
    assert hashlib.sha256(historical.to_csv(index=False).encode()).hexdigest() == _GOLDEN
    explicit = normalize_strict(_stub(), _SET, refine_input=False)
    assert explicit.to_csv(index=False) == df.to_csv(index=False)


def test_refine_on_keeps_original_entrada_and_appends_column_last():
    df = normalize_strict(_stub(), _SET, refine_input=True)
    assert list(df.columns) == OUTPUT_COLUMNS + ["refine_changes"]
    assert df["direccion_entrada"].tolist() == _SET
    assert df.loc[1, "refine_changes"] == "comma_to_space"
    assert df.loc[1, "estado"] == "OK" and df.loc[1, "direccion_normalizada"].startswith("CL 5 # 10 - 20")


def test_refine_on_rescues_row_the_parser_truncates():
    # ``CL 5, # 10 - 20`` is read by the parser itself since round 5; a plate after the comma still truncates
    off = normalize_strict(_stub(), ["CL 5 # 10, 20"]).iloc[0]
    on = normalize_strict(_stub(), ["CL 5 # 10, 20"], refine_input=True).iloc[0]
    assert off["direccion_normalizada"] != on["direccion_normalizada"]


def test_refine_on_non_string_rows_pass_through():
    df = normalize_strict(_stub(), [None, float("nan"), 12], refine_input=True)
    assert len(df) == 3 and df["refine_changes"].tolist() == ["", "", ""]


@pytest.mark.requires_artifacts
def test_default_off_batch_and_strict_unchanged_on_real_artifacts():
    from cali_address.inference import AddressNormalizer
    from cali_address.paths import DEFAULT_ARTIFACTS_DIR
    import pandas as pd

    samples_path = os.path.join(DEFAULT_ARTIFACTS_DIR, "eval_samples.parquet")
    if not os.path.exists(samples_path):  # private evaluation rows (real addresses): never published
        pytest.skip("needs the private eval_samples.parquet")
    samples = pd.read_parquet(samples_path)
    raws = samples["raw_address"].astype(str).tolist()[:150]
    norm = AddressNormalizer(DEFAULT_ARTIFACTS_DIR)
    a = norm.normalize_batch(raws, use_model=False)
    b = norm.normalize_batch(raws, use_model=False, refine_input=False)
    assert a.to_csv(index=False) == b.to_csv(index=False)
    assert "refine_changes" not in a.columns
    c = norm.normalize_batch(raws, use_model=False, refine_input=True)
    assert list(c.columns)[-1] == "refine_changes" and c["raw_address"].tolist() == raws


def test_placeholder_set_mirrors_service():
    from cali_address.service import PLACEHOLDERS

    assert refine_module._PLACEHOLDERS == PLACEHOLDERS


def test_line_breaks_are_kept_because_the_parser_treats_them_as_boundaries():
    # A/B finding: collapsing "\n" into a space made `<name>\n<address>` unparseable.
    assert r("Torre uno\nCra 88 N2 D 150") == "Torre uno\nCra 88 N2 D 150"
    assert r("Torre uno \r\n  Cra 88 N2 D 150\r\n") == "Torre uno\nCra 88 N2 D 150"
    assert r("CL 5\n\n# 10 - 20") == "CL 5\n# 10 - 20"
    assert_invariants("A\nCL 5 # 10", r("A\nCL 5 # 10"))


def test_fixpoint_loop_makes_refine_idempotent_whatever_the_rule_order(monkeypatch):
    monkeypatch.setattr(refine_module, "_RULES", list(reversed(refine_module._RULES)))
    for raw in GRID[:4000]:
        a = refine_address(raw)
        assert_invariants(raw, a.text)
        assert refine_address(a.text).text == a.text, raw
