"""Fidelity hardening (round 6): stray punctuation, ranges, trailing number groups, glued vias, suffixes.

Every address here is synthetic (shifted 9xx-style numbers). The rules under test:

* a stray ``;`` / ``,`` / ``.`` between the ``#`` and the plate must not hide the plate from the door
  (``#;958-965`` lost its plate and was only "low");
* a number group that trails the door (``- 906``, ``/910``, ``al 904-910``, ``# 904-910``, ``-4D 942``,
  ``40A 941``) and has no trace in the output is at least ``medium``;
* the same group introduced by a unit marker (AP, TO, INT, LC, CS, BL, CASA, ...) is NOT flagged by the new
  rules (unit policy stays low / unchanged).
"""

from __future__ import annotations

import itertools

import pytest

from cali_address.fidelity import assess_fidelity

NEW_FLAGS = {"trailing_number_dropped", "multi_address_truncated"}
MED_UP = ("medium", "high")


def rep(raw, out):
    return assess_fidelity(raw, out)


def risk(raw, out):
    return rep(raw, out).risk


def flags(raw, out):
    return set(rep(raw, out).flags)


# ---------------------------------------------------------------------------
# stray punctuation: the plate hidden behind a ';'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sep", ["#;", "# ;", "#,;", "#;;", "# ; ", "#.;", "#；", "#;-", "# ;-"])
def test_stray_punctuation_before_the_plate_does_not_hide_a_dropped_plate(sep):
    assert risk(f"Carrera 901C {sep}958-965", "KR 901 C #") == "high"


def test_stray_semicolon_dropped_plate_is_door_digits_dropped():
    assert "door_digits_dropped" in flags("Carrera 901C #;958-965", "KR 901 C #")


@pytest.mark.parametrize("sep", ["#;", "# ;", "#,;", "#；"])
def test_stray_punctuation_with_a_faithful_output_is_harmless(sep):
    assert risk(f"Carrera 901C {sep}958-965", "KR 901 C # 958 - 965") in ("none", "low")


def test_stray_punctuation_changed_plate_is_still_high():
    assert risk("Carrera 901C #;958-965", "KR 901 C # 958 - 956") == "high"


def test_stray_double_dot_is_noise():
    assert risk("Carrera 901C #..958-965", "KR 901 C #") == "high"
    assert risk("Carrera 901C #..958-965", "KR 901 C # 958 - 965") in ("none", "low")


def test_semicolon_between_two_addresses_still_separates_them():
    raw = "CL 905 # 906-907; KR 908 # 909-910"
    assert risk(raw, "CL 905 # 906 - 907") in MED_UP
    assert "multi_address_truncated" in flags(raw, "CL 905 # 906 - 907")
    assert risk(raw, "CL 905 # 906 - 907; KR 908 # 909 - 910") in ("none", "low")


# ---------------------------------------------------------------------------
# ranges / multi-address collapsed
# ---------------------------------------------------------------------------
def test_second_number_sign_range_collapsed_is_multi_address_truncated():
    raw = "CRA 912 # 904-902 - # 904-910 APTO 401 Y 501"
    f = flags(raw, "KR 912 # 904 - 902 AP 401")
    assert "multi_address_truncated" in f
    assert risk(raw, "KR 912 # 904 - 902 AP 401") in MED_UP


@pytest.mark.parametrize(
    "raw",
    [
        "CRA 912 # 904-902 - # 904-910",
        "CRA 912 # 904-902/910",
        "CRA 912 # 904-902 al 904-910",
        "CRA 912 # 904-902 a 904-910",
        "CRA 912 # 904-902 hasta 904-910",
    ],
)
def test_range_forms_collapsed_to_one_plate_are_medium_or_high(raw):
    assert risk(raw, "KR 912 # 904 - 902") in MED_UP


def test_range_kept_in_the_output_is_harmless():
    assert risk("CRA 912 # 904-902 - # 904-910", "KR 912 # 904 - 902; KR 912 # 904 - 910") in ("none", "low")
    assert risk("CRA 912 # 904-902/910", "KR 912 # 904 - 902 / 910") in ("none", "low")


def test_range_collapse_with_changed_plate_is_still_high():
    assert risk("CRA 912 # 904-902 - # 904-910", "KR 912 # 904 - 920") == "high"


def test_restated_address_is_not_a_range():
    # the same cross/plate typed twice is a restatement, not a second address
    assert risk("CL 905 # 906-907 # 906-907", "CL 905 # 906 - 907") in ("none", "low")


# ---------------------------------------------------------------------------
# trailing number groups, with and without unit markers
# ---------------------------------------------------------------------------
def test_trailing_dash_group_dropped_is_medium():
    r = rep("CARRERA 935 # 935 - 910 - 906", "KR 935 # 935 - 910")
    assert "trailing_number_dropped" in r.flags
    assert r.risk in MED_UP


@pytest.mark.parametrize("raw", ["CRA 954 # 919-951- 902", "CRA 954 # 919-951 -902", "CRA 954 # 919-951-902"])
def test_trailing_group_spacing_variants(raw):
    assert risk(raw, "KR 954 # 919 - 951") in MED_UP


def test_trailing_group_kept_is_harmless():
    assert risk("CARRERA 935 # 935 - 910 - 906", "KR 935 # 935 - 910 - 906") in ("none", "low")


@pytest.mark.parametrize(
    "unit,canon",
    [("AP", "AP"), ("APTO", "AP"), ("TORRE", "TO"), ("TO", "TO"), ("INT", "IN"), ("INTERIOR", "IN"), ("LC", "LC"),
     ("LOCAL", "LC"), ("CS", "CS"), ("CASA", "CS"), ("BL", "BL"), ("BLOQUE", "BL"), ("OFICINA", "OF")],
)
@pytest.mark.parametrize("glue", [" ", " - ", "-"])
def test_unit_marker_groups_never_raise_the_trailing_flags(unit, canon, glue):
    raw = f"CARRERA 935 # 935 - 910{glue}{unit} 906"
    assert not (flags(raw, "KR 935 # 935 - 910") & NEW_FLAGS)
    assert not (flags(raw, f"KR 935 # 935 - 910 {canon} 906") & NEW_FLAGS)


@pytest.mark.parametrize("unit", ["PISO", "ETAPA", "EDIFICIO"])
def test_weak_unit_groups_stay_low(unit):
    assert risk(f"CARRERA 935 # 935 - 910 {unit} 906", "KR 935 # 935 - 910") in ("none", "low")


def test_bare_ordinal_number_before_a_unit_word_is_not_a_trailing_group():
    assert risk("CARRERA 935 # 935 - 910 3er piso", "KR 935 # 935 - 910") in ("none", "low")


def test_unit_after_a_dropped_group_still_flags_the_group_only():
    f = flags("CARRERA 935 # 935 - 910 - 906 AP 301", "KR 935 # 935 - 910 AP 301")
    assert "trailing_number_dropped" in f
    assert "unit_digits_dropped" not in f


def test_place_text_digits_are_not_trailing_groups():
    assert risk("CARRERA 935 # 935 - 910 comuna 3", "KR 935 # 935 - 910") in ("none", "low")
    assert risk("CARRERA 935 # 935 - 910 barrio 3", "KR 935 # 935 - 910") in ("none", "low")


# ---------------------------------------------------------------------------
# glued vias
# ---------------------------------------------------------------------------
def test_glued_vias_with_dropped_digits_are_flagged():
    assert risk("CL 931 932 KRA 940A 941", "CL 931 # 932 KRA - 940") in MED_UP


def test_glued_vias_dropped_door_letter_is_high():
    assert risk("CL 931 932 KRA 940A 941", "CL 931 # 932 KRA - 940") == "high"


@pytest.mark.parametrize("via", ["KRA", "CLL", "CRA", "KR", "CARRERA"])
def test_glued_via_alias_spellings_are_the_same_rule(via):
    out = "CL 931 # 932 KRA - 940"
    assert risk(f"CL 931 932 {via} 940A 941", out) in MED_UP


def test_glued_vias_faithful_output_is_harmless():
    assert risk("CL 931 932 KRA 940", "CL 931 # 932 KRA - 940") in ("none", "low")


# ---------------------------------------------------------------------------
# suffixes
# ---------------------------------------------------------------------------
def test_suffix_group_dropped_is_medium():
    r = rep("CLL 962B #944-965-4D 942", "CL 962 B # 944 - 965")
    assert r.risk in MED_UP
    assert "trailing_number_dropped" in r.flags


@pytest.mark.parametrize("raw", ["CLL 962B #944-965-4D", "CLL 962B #944-965-4D 942", "CLL 962B #944-965 -4D"])
def test_suffix_forms(raw):
    assert risk(raw, "CL 962 B # 944 - 965") in MED_UP


def test_suffix_group_kept_is_harmless():
    assert risk("CLL 962B #944-965-4D 942", "CL 962 B # 944 - 965 - 4 D 942") in ("none", "low")


def test_door_letter_suffix_not_a_trailing_group():
    assert risk("CLL 962B #944-965D", "CL 962 B # 944 - 965 D") in ("none", "low")


# ---------------------------------------------------------------------------
# quadrant positions
# ---------------------------------------------------------------------------
def test_quadrant_dropped_remains_high():
    assert risk("Carrera 952 #904-947 oeste", "KR 952 # 904 - 947") == "high"


def test_trailing_group_before_a_quadrant_is_flagged():
    assert risk("KR 952 # 904 - 947 - 906 OESTE", "KR 952 # 904 - 947 OESTE") in MED_UP


def test_quadrant_and_trailing_group_kept_is_harmless():
    assert risk("KR 952 # 904 - 947 OESTE", "KR 952 # 904 - 947 OESTE") == "none"


# ---------------------------------------------------------------------------
# boundaries and robustness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [None, "", "   ", 123, 4.5, b"", [], ";", "#;", ";;;", "- - -", "#"])
def test_degenerate_raw_does_not_crash(raw):
    assert rep(raw, "KR 935 # 935 - 910").risk in ("none", "low", "medium", "high")
    assert rep(raw, None).risk == "none"


@pytest.mark.parametrize("out", [None, "", 7, b"KR 935", ";", "#;"])
def test_degenerate_output_does_not_crash(out):
    assert rep("CARRERA 935 # 935 - 910 - 906", out).risk in ("none", "low", "medium", "high")


def test_very_long_input_with_stray_punctuation_is_fast_and_safe():
    raw = "Carrera 901C #;958-965 " + "; # - 906 " * 5000
    assert rep(raw, "KR 901 C #").risk == "high"


def test_trailing_rule_is_deterministic():
    raw, out = "CLL 962B #944-965-4D 942", "CL 962 B # 944 - 965"
    assert rep(raw, out) == rep(raw, out)


@pytest.mark.parametrize(
    "text",
    ["KR 935 # 935 - 910", "CL 962 B # 944 - 965", "KR 952 # 904 - 947 OESTE", "CL 931 # 932 KRA - 940",
     "KR 912 # 904 - 902 AP 401", "KR 901 C # 958 - 965"],
)
def test_idempotence_on_faithful_outputs(text):
    r = rep(text, text)
    assert r.risk == "none" and not r.flags


# ---------------------------------------------------------------------------
# property grid: changing / dropping any door digit still raises to high
# ---------------------------------------------------------------------------
GRID = list(itertools.product(("CL", "KR"), ("901", "925"), ("958", "912"), ("965", "907")))
RAW_FORMS = ("{t} {a} # {b}-{c}", "{t} {a} #;{b}-{c}", "{t} {a} # ;{b}-{c}", "{t} {a} #,{b} - {c}")


@pytest.mark.parametrize("form", RAW_FORMS)
def test_property_dropping_any_door_digit_group_is_high(form):
    for t, a, b, c in GRID:
        raw = form.format(t=t, a=a, b=b, c=c)
        for out in (f"{t} {b} # {c}", f"{t} {a} # {c}", f"{t} {a} # {b}", f"{t} {a}", f"{t} {a} #"):
            assert risk(raw, out) == "high", (raw, out)


@pytest.mark.parametrize("form", RAW_FORMS)
def test_property_changing_any_door_digit_is_high(form):
    for t, a, b, c in GRID:
        raw = form.format(t=t, a=a, b=b, c=c)
        for out in (f"{t} {a}1 # {b} - {c}", f"{t} {a} # {b}1 - {c}", f"{t} {a} # {b} - {c}1",
                    f"{t} {a} # {b} - {int(c) + 1}"):
            assert risk(raw, out) == "high", (raw, out)


@pytest.mark.parametrize("form", RAW_FORMS)
def test_property_faithful_output_is_never_raised(form):
    for t, a, b, c in GRID:
        raw = form.format(t=t, a=a, b=b, c=c)
        assert risk(raw, f"{t} {a} # {b} - {c}") in ("none", "low"), raw


def test_trailing_digit_in_a_different_place_of_the_output_counts_as_kept():
    # an output that still carries the digit somewhere (unit tail) is not a silent loss
    assert risk("CARRERA 935 # 935 - 910 - 906", "KR 935 # 935 - 910 AP 906") in ("none", "low", "medium")
    assert "trailing_number_dropped" not in flags("CARRERA 935 # 935 - 910 - 906", "KR 935 # 935 - 910 AP 906")


@pytest.mark.parametrize(
    "raw,out",
    [
        ("Cl. 905 #924-991, 903 de Julio, Cali, Valle del Cauca", "CL 905 # 924 - 991"),
        ("Calle 906 #927-918 901 de julio", "CL 906 # 927 - 918"),
        ("Cll 901c oeste #900bis -988, T 980 altos de Santa Helena", "CL 901 C OESTE # 900 BIS - 988"),
        ("Cl. 905 #924A-991, 903 de Julio", "CL 905 # 924 A - 991"),
    ],
)
def test_number_that_starts_a_place_name_is_not_a_trailing_group(raw, out):
    assert not (flags(raw, out) & NEW_FLAGS)
    assert risk(raw, out) in ("none", "low")


@pytest.mark.parametrize("tail", ["P 1", "T 9", "T2", "t 9", "T 9 P 1"])
def test_lone_letter_unit_markers_after_the_plate_are_not_trailing_groups(tail):
    raw = f"CL 987 # 654 - 12 {tail}"
    assert not (flags(raw, "CL 987 # 654 - 12") & NEW_FLAGS)
    assert risk(raw, "CL 987 # 654 - 12") in ("none", "low")


def test_lone_letter_marker_followed_by_a_unit_keeps_the_unit_policy():
    raw = "CL 987 # 654 - 12 P 1 AP 301"
    assert not (flags(raw, "CL 987 # 654 - 12 AP 301") & NEW_FLAGS)
    assert risk(raw, "CL 987 # 654 - 12 AP 301") in ("none", "low")

