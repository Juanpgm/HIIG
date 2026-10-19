"""Fidelity measure (raw input vs final normalized output): semantic drift / corruption risk.

Synthetic addresses only. The module under test must not depend on the production parser: the
cases below pin what counts as HARMLESS normalization (must stay none/low) and what is CORRUPTION
(must be high / medium).
"""

from __future__ import annotations

import itertools
import random
import math
import re

import pytest

from cali_address import fidelity
from cali_address.fidelity import FidelityReport, assess_fidelity, assess_fidelity_batch

OK_RISKS = ("none", "low")


def risk(raw, out, **kw):
    return assess_fidelity(raw, out, **kw).risk


def flags(raw, out, **kw):
    return set(assess_fidelity(raw, out, **kw).flags)


# ---------------------------------------------------------------------------
# harmless normalization: must NOT be flagged above low
# ---------------------------------------------------------------------------
FAITHFUL = [
    ("Calle 905 #938-925", "CL 905 # 938 - 925"),
    ("cl. 905 # 938 - 925", "CL 905 # 938 - 925"),
    ("CALLE 905 No. 938-925", "CL 905 # 938 - 925"),
    ("CALLE 905 NO, 938-925", "CL 905 # 938 - 925"),
    ("Calle 905 N° 938-925", "CL 905 # 938 - 925"),
    ("Carrera 929A #917-984", "KR 929 A # 917 - 984"),
    ("CRA 929a 917 984", "KR 929 A # 917 - 984"),
    ("K 929A 917-984", "KR 929 A # 917 - 984"),
    ("Cra966 #913b_929", "KR 966 # 913 B - 929"),
    ("Av 906 Norte #922-914", "AV 906 N # 922 - 914"),
    ("AV 906 N # 922 - 914", "AV 906 NORTE # 922 - 914"),
    ("Avenida 905oeste # 925-985", "AV 905 OESTE # 925 - 985"),
    ("Av905an # 910-920", "AV 905 A NORTE # 910 - 920"),
    ("Calle 923anorte 905n 944o-950", "CL 923 A NORTE # 905 N - 944 O 950"),
    ("Calle 962 B #901A909-1150", "CL 962 B # 901 A 909 - 1150"),
    ("Diagonal 926P # 1003-0906", "DG 926 P # 1003 - 906"),
    ("DG 926P No. 1003-0906", "DG 926 P # 1003 - 0906"),
    ("Transversal 929 # D927A-00", "TV 929 # D 927 A - 0"),
    ("Cra 907D bis 969a943", "KR 907 D BIS # 969 A - 943"),
    ("CL 905 # 0908 - 920", "CL 905 # 908 - 920"),
    ("CL 905 # 908 - 920", "CL 905 # 0908 - 920"),
    ("Calle 912 oeste# 954-1030", "CL 912 OESTE # 954 - 1030"),
    ("Cl. 906a # 914-973, COMUNA 903, Cali, Valle del Cauca", "CL 906 A # 914 - 973"),
    ("Calle 905 # 924a-991 San Fernando, Cali, Colombia", "CL 905 # 924 A - 991"),
    ("CL 947 # 996-955AP 1402 TE", "CL 947 # 996 - 955 AP 1402 TE"),
    ("CALLE 905 # 947-932 LOCAL 921", "CL 905 # 947 - 932 LC 921"),
    ("CALLE 905 # 947-932 APARTAMENTO 921", "CL 905 # 947 - 932 AP 921"),
    ("CL 902 B # 994 D 0907 APT 1101", "CL 902 B # 994 D - 907 AP 1101"),
    ("Calle 905 Nº 938-925", "CL 905 # 938 - 925"),
    ("calle 905 # 938 925", "CL 905 # 938 - 925"),
    ("CALLE 905 № 938-925", "CL 905 # 938 - 925"),
    ("CRA 932A # 909B 972; CRA 932A # 909B 972", "KR 932 A # 909 B - 972"),
    ("CR 966 B  #913 A  -949 CARRERA 966 B # 913 A -949", "KR 966 B # 913 A - 949"),
    ("999 0 0  #0 0  -0 Piso0 BBVA 1260 - CALLE 948 # 908A-918 EL TRONCAL SAS", "CL 948 # 908 A - 918"),
    ("Av 906 Oeate #922-914", "AV 906 OESTE # 922 - 914"),
    ("Carrra 901era c904#964-965", "KR 901 C 904 # 964 - 965"),
]


@pytest.mark.parametrize("raw, out", FAITHFUL)
def test_harmless_normalization_is_not_flagged(raw, out):
    rep = assess_fidelity(raw, out)
    assert rep.risk in OK_RISKS, (raw, out, rep)
    assert rep.score >= 0.9


def test_identical_strings_are_none_with_full_score():
    rep = assess_fidelity("CL 905 # 938 - 925", "CL 905 # 938 - 925")
    assert rep == FidelityReport(risk="none", flags=(), score=1.0, details=rep.details)


def test_padding_is_low_not_high():
    rep = assess_fidelity("CL 905 # 0908 - 920", "CL 905 # 908 - 920")
    assert rep.risk in OK_RISKS
    assert rep.score <= 1.0


# ---------------------------------------------------------------------------
# corruption: door-identifying data
# ---------------------------------------------------------------------------
KNOWN_CORRUPTIONS = [
    ("Carrera 946, 908B-920", "KR 946"),
    ("KR 926G906 972S901 939", "KR 926 G # 906 - 972 S 901 939"),
    ("CALLE 962 NO, 908 - 1000", "CL 962 N OESTE"),
    ("Carrera 929A #917-984", "KR 929 A # 917 - 948"),
    ("Calle 905#938-925", "CL 906 # 938 - 925"),
]


@pytest.mark.parametrize("raw, out", KNOWN_CORRUPTIONS)
def test_known_corruptions_are_high(raw, out):
    assert risk(raw, out) == "high"


CORRUPT = {
    "plate_digit_changed": ("CL 905 # 938 - 925", "CL 905 # 938 - 926"),
    "plate_neighbour_plus910": ("CL 905 # 938 - 925", "CL 905 # 938 - 935"),
    "cross_changed": ("CL 905 # 938 - 925", "CL 905 # 939 - 925"),
    "via_number_changed": ("CL 905 # 938 - 925", "CL 906 # 938 - 925"),
    "plate_dropped": ("CL 905 # 938 - 925", "CL 905 # 938"),
    "cross_dropped": ("CL 905 # 938 - 925", "CL 905 - 925"),
    "digit_added": ("CL 905 # 938 - 925", "CL 905 # 938 - 1025"),
    "digits_reordered": ("CL 905 # 938 - 925", "CL 905 # 925 - 938"),
    "letter_changed": ("CL 905 A # 938 - 925", "CL 905 B # 938 - 925"),
    "letter_dropped": ("CL 905 A # 938 - 925", "CL 905 # 938 - 925"),
    "letter_added": ("CL 905 # 938 - 925", "CL 905 A # 938 - 925"),
    "cross_letter_dropped": ("CL 905 # 938 B - 925", "CL 905 # 938 - 925"),
    "quadrant_dropped": ("CL 905 NORTE # 938 - 925", "CL 905 # 938 - 925"),
    "quadrant_added": ("CL 905 # 938 - 925", "CL 905 NORTE # 938 - 925"),
    "quadrant_changed": ("CL 905 NORTE # 938 - 925", "CL 905 SUR # 938 - 925"),
    "quadrant_moved_via_to_cross": ("CL 905 N # 910 - 920", "CL 905 # 910 N - 920"),
    "quadrant_moved_to_trailing": ("CL 905 N # 910 - 920", "CL 905 # 910 - 920 N"),
    "via_type_changed": ("CL 905 # 938 - 925", "KR 905 # 938 - 925"),
    "via_type_changed_av": ("AV 905 # 938 - 925", "DG 905 # 938 - 925"),
    "scrambled": ("KR 966 # 956 - 912", "# - 956 NAO 966"),
    "multi_address_truncated": ("CL 905 # 938 - 925; KR 907 # 910 - 920", "CL 905 # 938 - 925"),
}


@pytest.mark.parametrize("name", sorted(CORRUPT))
def test_corruptions_are_medium_or_high(name):
    raw, out = CORRUPT[name]
    rep = assess_fidelity(raw, out)
    assert rep.risk in ("medium", "high"), (name, rep)
    assert rep.score < 0.8


@pytest.mark.parametrize("name", [n for n in sorted(CORRUPT) if n != "multi_address_truncated"])
def test_door_corruptions_are_high(name):
    raw, out = CORRUPT[name]
    assert risk(raw, out) == "high", name


def test_quadrant_position_is_significant():
    assert risk("CL 905 N # 910 - 920", "CL 905 # 910 N - 920") == "high"
    assert "quadrant_moved" in flags("CL 905 N # 910 - 920", "CL 905 # 910 N - 920")
    assert risk("CL 905 N # 910 - 920", "CL 905 NORTE # 910 - 920") in OK_RISKS


def test_quadrant_synonyms_only_harmless_in_same_segment():
    assert risk("CL 905 NTE # 910 - 920", "CL 905 NORTE # 910 - 920") in OK_RISKS
    assert risk("CL 905 # 910 NTE - 920", "CL 905 NORTE # 910 - 920") == "high"


def test_glued_split_by_separator_flags():
    rep = assess_fidelity("KR 926G906 972S901 939", "KR 926 G # 906 - 972 S 901 939")
    assert "glued_split" in rep.flags and rep.risk == "high"


def test_glued_token_spaced_is_harmless():
    for raw, out in [("CL 966C # 910 - 920", "CL 966 C # 910 - 920"), ("CL 905A # 910 - 920", "CL 905 A # 910 - 920"),
                     ("KR 901C901 # 968 - 912", "KR 901 C 901 # 968 - 912")]:
        assert risk(raw, out) in OK_RISKS, (raw, out)


GLUED = {"966C": "966 C", "926G906": "926 G 906", "905b901": "905 B 901", "994C901": "994 C 901", "0908CA": "0908 C A", "901A909": "901 A 909",
         "915BNORTE": "915 B NORTE", "922NORTE": "922 NORTE"}


@pytest.mark.parametrize("token, spaced", sorted(GLUED.items()))
def test_glued_alphanumerics_kept_are_harmless_and_dropped_are_flagged(token, spaced):
    raw = f"KR {token} # 910 - 920"
    assert risk(raw, f"KR {spaced} # 910 - 920") in OK_RISKS
    assert risk(raw, "KR # 910 - 920") == "high"


def test_unit_digits_dropped_is_medium_not_high():
    rep = assess_fidelity("CL 905 # 938 - 925 APTO 1201", "CL 905 # 938 - 925")
    assert rep.risk == "medium" and "unit_digits_dropped" in rep.flags


def test_unit_relabelled_is_low():
    rep = assess_fidelity("CL 905 # 938 - 925 APTO 1201", "CL 905 # 938 - 925 CS 1201")
    assert rep.risk == "low" and "unit_relabelled" in rep.flags


def test_unit_digits_changed_is_not_silent():
    assert risk("CL 905 # 938 - 925 APTO 1201", "CL 905 # 938 - 925 AP 1202") in ("medium", "high")


def test_annotation_digits_dropped_is_low():
    assert risk("CL 905 # 938 - 925 etapa 902", "CL 905 # 938 - 925") in OK_RISKS


def test_multi_address_duplicates_deduplicated_is_harmless():
    assert risk("CRA 932A # 909B 972; CRA 932A # 909B 972", "KR 932 A # 909 B - 972") in OK_RISKS
    assert "multi_address_truncated" in flags("CL 905 # 938 - 925 Y KR 907 # 910 - 920", "CL 905 # 938 - 925")
    assert "multi_address_truncated" in flags("CL 905 # 938 - 925 Y/O KR 907 # 910 - 920", "CL 905 # 938 - 925")


def test_multi_address_both_kept_is_harmless():
    assert risk("CL 905 # 938 - 925; KR 907 # 910 - 920", "CL 905 # 938 - 925; KR 907 # 910 - 920") in OK_RISKS


def test_output_swapped_to_the_other_segment_is_a_door_change_or_truncation():
    assert risk("CL 905 # 938 - 925; KR 907 # 910 - 920", "KR 907 # 910 - 920") in ("medium", "high")


def test_place_text_removed_via_ignore_is_harmless():
    raw = "CL 912 # 923 - 911 San Fernando Viejo Junin"
    assert risk(raw, "CL 912 # 923 - 911") in OK_RISKS
    assert risk(raw, "CL 912 # 923 - 911", cleaned="CL 912 # 923 - 911") in OK_RISKS


def test_cadastral_substitution_neighbour_plate_is_high():
    for delta in (1, 2, 10, -1, -10):
        out = f"KR 928 # 906 - {915 + delta}"
        assert risk("CRA928 #906-915", out) == "high"
        assert "plate_changed" in flags("CRA928 #906-915", out)


# ---------------------------------------------------------------------------
# edge cases: empty / None / non-string / long / unicode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["", None, "   ", float("nan"), 5, [], "SIN ESPECIFICAR", "N/A", "SIN DIRECCION"])
@pytest.mark.parametrize("out", ["", None, float("nan"), "  "])
def test_empty_or_placeholder_with_empty_output_is_none_no_address(raw, out):
    rep = assess_fidelity(raw, out)
    assert rep.risk == "none" and "no_address" in rep.flags and rep.score == 1.0


@pytest.mark.parametrize("raw", ["", None, "SIN ESPECIFICAR"])
def test_output_invented_from_nothing_is_high(raw):
    assert risk(raw, "CL 905 # 938 - 925") == "high"


def test_non_string_output_does_not_crash():
    for out in (None, 5, 3.2, [], {}, b"CL 905"):
        assess_fidelity("CL 905 # 938 - 925", out)


def test_raw_with_address_and_empty_output_is_none_no_address():
    rep = assess_fidelity("Carrera 946, 908B-920", None)
    assert rep.risk == "none" and rep.flags == ("no_address",)


def test_very_long_input_does_not_crash_and_stays_fast():
    raw = "CL 905 # 938 - 925 " + "barrio los alamos " * 60
    assert len(raw) > 300
    assert risk(raw, "CL 905 # 938 - 925") in OK_RISKS
    assert risk("X" * 5000 + " CL 905 # 938 - 925", "CL 905 # 938 - 925") in OK_RISKS


def test_unicode_accents_nfd_ntilde_and_fullwidth_digits():
    import unicodedata

    assert risk("Calle 905 #938-925 Cañaverales", "CL 905 # 938 - 925") in OK_RISKS
    nfd = unicodedata.normalize("NFD", "Cárrera 929Á #917-984")
    assert risk(nfd, "KR 929 A # 917 - 984") in OK_RISKS
    assert risk("Calle 905 #９３８-９２５", "CL 905 # 938 - 925") in OK_RISKS
    assert risk("Calle 905 #938-925", "CL 905 # ９３８ - ９２５") in OK_RISKS
    assert risk("Calle 905 #９３８-９２５", "CL 905 # 938 - 926") == "high"


def test_unicode_zero_width_and_nbsp_are_invisible():
    assert risk("Calle 905​ #938‍-925", "CL 905 # 938 - 925") in OK_RISKS
    assert risk("Calle 905 #938-925", "CL 905 # 938 - 925") in OK_RISKS
    assert risk("Calle 905 #938�-925 Cali", "CL 905 # 938 - 925") in OK_RISKS


def test_mojibake_replacement_char_in_text_is_noise():
    assert risk("Av 906N #943N-921 La Campi�a", "AV 906 N # 943 N - 921") in OK_RISKS


# ---------------------------------------------------------------------------
# structural properties
# ---------------------------------------------------------------------------
def test_report_is_well_formed():
    rep = assess_fidelity("CL 905 # 938 - 925", "CL 905 # 938 - 926")
    assert isinstance(rep, FidelityReport)
    assert rep.risk in ("none", "low", "medium", "high")
    assert isinstance(rep.flags, tuple) and all(isinstance(f, str) for f in rep.flags)
    assert 0.0 <= rep.score <= 1.0 and not math.isnan(rep.score)
    assert isinstance(rep.details, dict)


def test_score_is_monotone_in_risk():
    scores = {r: assess_fidelity(*p).score for r, p in {
        "none": ("CL 905 # 938 - 925", "CL 905 # 938 - 925"),
        "low": ("CL 905 # 0908 - 925", "CL 905 # 908 - 925"),
        "medium": ("CL 905 # 938 - 925 APTO 1201", "CL 905 # 938 - 925"),
        "high": ("CL 905 # 938 - 925", "CL 905 # 938 - 926"),
    }.items()}
    assert scores["none"] >= scores["low"] > scores["medium"] > scores["high"] >= 0.0


def test_deterministic_and_idempotent_on_faithful_outputs():
    for raw, out in FAITHFUL:
        assert assess_fidelity(raw, out) == assess_fidelity(raw, out)
        assert risk(out, out) == "none", out


def test_batch_helper_matches_scalar_and_validates_lengths():
    pairs = [("CL 905 # 938 - 925", "CL 905 # 938 - 925"), ("CL 905 # 938 - 925", "CL 905 # 938 - 926"), (None, None)]
    reps = assess_fidelity_batch([p[0] for p in pairs], [p[1] for p in pairs])
    assert [r.risk for r in reps] == ["none", "high", "none"]
    assert reps == [assess_fidelity(*p) for p in pairs]
    assert assess_fidelity_batch([], []) == []
    with pytest.raises(ValueError):
        assess_fidelity_batch(["a"], [])


def test_batch_ignore_is_per_row():
    reps = assess_fidelity_batch(["CL 905 # 938 - 925 Junin"], ["CL 905 # 938 - 925"], cleaned=["CL 905 # 938 - 925"])
    assert reps[0].risk in OK_RISKS


def test_does_not_import_the_production_parser():
    src = open(fidelity.__file__, encoding="utf-8").read()
    assert not re.search(r"^\s*(from|import)\s+.*\bparser\b", src, re.M)
    assert "cali_address" not in re.sub(r'""".*?"""', "", src, flags=re.S).replace("cali_address.parser", "")


# property grid: any single digit mutation of a door number in a faithful output is high
DOORS = [
    ("CL 905 # 938 - 925", "CL 905 # 938 - 925"),
    ("Cra 929a #917-984", "KR 929 A # 917 - 984"),
    ("AV 906 Norte #922-914", "AV 906 N # 922 - 914"),
    ("CALLE 962B #901A909-1150", "CL 962 B # 901 A 909 - 1150"),
    ("Diagonal 926P # 1003-0906", "DG 926 P # 1003 - 906"),
    ("KR 1000 # 901 D OESTE - 978", "KR 1000 # 901 D OESTE - 978"),
]


def _digit_spans(text):
    return [m.span() for m in re.finditer(r"\d+", text)]


@pytest.mark.parametrize("raw, out", DOORS)
def test_property_any_door_digit_mutation_is_high(raw, out):
    spans = _digit_spans(out)
    assert spans
    for start, end in spans:
        num = out[start:end]
        mutants = {num[:i] + str((int(num[i]) + d) % 10) + num[i + 1:] for i in range(len(num)) for d in (1, 5)}
        mutants.add(num + "7")          # digit added
        if len(num) > 1:
            mutants.add(num[:-1])       # digit removed
        for mutant in mutants:
            if mutant.lstrip("0") == num.lstrip("0"):
                continue                # padding is harmless
            bad = out[:start] + mutant + out[end:]
            assert risk(raw, bad) == "high", (raw, bad)
    # removing a whole door number
    for start, end in spans:
        bad = re.sub(r"\s+", " ", out[:start] + out[end:]).strip()
        assert risk(raw, bad) == "high", (raw, bad)


@pytest.mark.parametrize("raw, out", DOORS)
def test_property_swapping_two_door_numbers_is_high(raw, out):
    spans = _digit_spans(out)
    for (a0, a1), (b0, b1) in itertools.combinations(spans, 2):
        if out[a0:a1].lstrip("0") == out[b0:b1].lstrip("0"):
            continue
        bad = out[:a0] + out[b0:b1] + out[a1:b0] + out[a0:a1] + out[b1:]
        assert risk(raw, bad) == "high", (raw, bad)


# ---------------------------------------------------------------------------
# ignore / tail specifics
# ---------------------------------------------------------------------------
def test_cleaned_never_hides_door_digits():
    assert risk("CL 905 # 938 - 925", "CL 905 # 938 - 926", cleaned="CL 905 # 938") == "high"


def test_junk_prefix_before_the_via_is_not_a_door_change():
    assert risk("TORRE 906 APT1905 CL 905 # 938 - 925", "CL 905 # 938 - 925") in OK_RISKS


def test_details_are_small():
    rep = assess_fidelity("CL 905 # 938 - 925 " * 50, "CL 905 # 938 - 926")
    assert len(rep.details) <= 12
    assert all(len(str(v)) < 400 for v in rep.details.values())


# ---------------------------------------------------------------------------
# rules added after auditing real outputs (each pins one harmless pattern or one real corruption)
# ---------------------------------------------------------------------------
HARMLESS_MORE = [
    ("CRA 927 N. 909-940", "KR 927 # 909 - 940"),                       # dotted N. is a number sign before the first '#'
    ("Av 903 N° 950 N. 935", "AV 903 # 950 NORTE - 935"),                # ... and a quadrant after it
    ("Cl. 974 No. 905 N. 928", "CL 974 # 905 N - 928"),
    ("CL 987 # 654 - 12 P 1", "CL 987 # 654 - 12"),                      # a floor / parking marker after the plate is a unit
    ("CL 987 # 654 - 12 T 9", "CL 987 # 654 - 12 TO 9"),
    ("CALLE 987 # 654 - 12 T2 AP 301", "CL 987 # 654 - 12 TO 2 AP 301"),
    ("Cra. 931, Comuna 10, Cali", "KR 931"),                             # digits of place text are not door digits
    ("Cra. 926 #905c6, 3 de Julio, Cali", "KR 926 # 905 C - 06"),
    ("Dollarcity Gran Via, Cra 903 #969-959, Cali", "KR 903 # 969 - 959"),  # an earlier via word must not become the anchor
    ("CR 995  OES #901   -961 CARRERA 995 OESTE # 901-961", "KR 995 OESTE # 901 - 961"),
    ("CRA 949 N 901 931 CASA 31", "KR 949 N # 901 - 931 CA 31"),               # CA is the abbreviation of CASA
    ("Calle 962B# 901AB-185", "CL 962 B # 901 AB - 185"),                   # two glued letters, spaced by the output
    ("Av 906DN #944AM-950 La campina", "AV 906 D NORTE # 944 AM - 950"),
    ("CALLE 934 TORRE UNO A", "CL 934 TO 1A"),                           # spelled number rendered as a digit in the tail
    ("Calle 910 # 29 6 - 78", "CL 910 # 29 6 - 78"),
    ("CLL 962B1A9-205-205 APT 2D-13", "CL 962 B 1 A 9 - 205 AP 2D13"),    # a number typed twice (4+ numbers) is deduplicated
    ("Carrera 916 con calle 913 #913- 22 Guayaquil", "KR 916 CON # 913 - 22"),
    ("Carrera 901 bb #963", "KR 901 BB # 963"),
    ("CL 987 C OESTE # 654 - 12", "CL 987 C OESTE # AV 654 - 12"),  # cross type named by the cadastre is low, not high
    ("CL 987 NORTE # 906 - 12", "CL 987 NORTE # AV 906 - 12"),
]


@pytest.mark.parametrize("raw, out", HARMLESS_MORE)
def test_harmless_patterns_found_in_real_data_stay_low(raw, out):
    assert risk(raw, out) in OK_RISKS, (raw, out, assess_fidelity(raw, out))


def test_pairs_of_equal_cross_and_plate_are_real_numbers_not_duplicates():
    assert risk("CRA 942 # 926-926", "KR 942 # 926 - 926") in OK_RISKS
    assert risk("CRA 942 # 926-926", "KR 942 # 926") == "high"
    assert risk("CRA 942 # 926-926", "KR 942 - 926") == "high"


def test_trailing_bis_dropped_and_second_address_dropped_are_flagged():
    assert risk("CL 987 # 654 - 12 BIS", "CL 987 # 654 - 12") == "high"
    assert "multi_address_truncated" in flags("CLL 922 C # 928E - 927 Y K 934#927 105", "CL 922 C # 928 E - 927")


def test_cross_type_changed_is_high_added_is_low_dropped_is_low():
    assert risk("CL 940 CRA 941", "CL 940 # DG 941") == "high"
    assert flags("CL 987 # 654 - 12", "CL 987 # AV 654 - 12") == {"cross_type_added"}
    assert risk("CL 940 CRA 941 - 12", "CL 940 # 941 - 12") == "low"


def test_invented_plate_from_a_unit_or_a_street_name_is_high():
    assert risk("CALLE 962B SECTOR 1", "CL 962 B # 901 - 01") == "high"
    assert risk("KR 901 A 5 ENTRE CALLES 947 Y 949", "KR 901 A # 5 - 947") == "high"
    assert risk("CARRERA 937 # 903BIS APARTAMENTO 204 B", "KR 937 # 903 BIS - 204 AP B") == "high"


def test_known_no_bug_nort_oeste_is_flagged():
    assert "quadrant_added" in flags("CARRERA 989 NO18 61 CASA 4", "KR 989 N OESTE # 18 - 61 CA 4")
    assert risk("DIAGONAL 910 NO20-65", "DG 910 N OESTE # 20 - 65") == "high"


def test_invariant_to_case_accents_whitespace_and_nbsp():
    rng = random.Random(11)
    base = [("Calle 905 # 938 - 925", "CL 905 # 938 - 925"), ("Carrera 929A #917-984 apto 301", "KR 929 A # 917 - 984 AP 301"),
            ("Av 906 Norte #922-914", "AV 906 N # 922 - 914"), ("Cra 988 N° 904 b - 950", "KR 988 # 904 B - 950")]
    for raw, out in base:
        want = assess_fidelity(raw, out).risk
        for _ in range(60):
            v = "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in raw)
            v = re.sub(" ", lambda _m: rng.choice([" ", "  ", " ", " ​"]), v)
            assert assess_fidelity(v, out).risk == want, (v, out)


def test_cleaned_text_only_excuses_tail_tokens_never_the_door():
    raw = "CL 987 # 654 - 12 COMUNA 3 SECTOR 2"
    assert risk(raw, "CL 987 # 654 - 12") in OK_RISKS
    assert risk(raw, "CL 987 # 654 - 12", cleaned="CL 987 # 654 - 12") in OK_RISKS
    assert risk(raw, "CL 987 # 654", cleaned="CL 987 # 654") == "high"       # a door digit is never excused
    assert risk("CL 987 # 654 - 12 APTO 301", "CL 987 # 654 - 12", cleaned="CL 987 # 654 - 12 APTO 301") == "medium"


def test_assess_never_raises_on_random_garbage():
    rng = random.Random(5)
    alphabet = "AB0123456789 #-;/,.NOSEoeCLKRAVTD ​Ññ�_°º"
    for _ in range(400):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        out = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        rep = assess_fidelity(raw, out)
        assert rep.risk in ("none", "low", "medium", "high") and 0.0 <= rep.score <= 1.0


def test_text_dropped_share_threshold_separates_low_from_medium():
    major = assess_fidelity("CL 987 # 654 - 12 TORRE B BLOQUE C", "CL 987 # 654 - 12")      # 2 of 6 tokens gone
    assert major.risk == "medium" and "text_dropped_major" in major.flags
    minor = assess_fidelity("CL 987 # 654 - 12 TORRE B BLOQUE C CASA D", "CL 987 # 654 - 12 TO B BLQ C")
    assert minor.risk == "low" and "text_dropped" in minor.flags and "text_dropped_major" not in minor.flags
    assert fidelity.TEXT_DROP_MEDIUM_SHARE == 0.30


def test_every_flag_has_a_documented_severity_and_scores_are_ordered():
    assert set(fidelity.SEVERITY.values()) == {"high", "medium", "low", "none"}
    worst = {r: assess_fidelity(*p) for r, p in {
        "high": ("CL 987 # 654 - 12", "CL 987 # 654 - 13"),
        "medium": ("CL 987 # 654 - 12 TORRE B BLOQUE C", "CL 987 # 654 - 12"),
        "low": ("CL 987 # 054 - 12", "CL 987 # 54 - 12"),
    }.items()}
    assert all(rep.risk == r for r, rep in worst.items())
    assert worst["low"].score > worst["medium"].score > worst["high"].score


def test_double_encoded_number_sign_is_a_number_sign():
    # ``N°`` read as cp1252 and re-saved as UTF-8 arrives as ``NÂ°``; the output keeps the garbage word
    assert risk("CL 905 NÂ° 910 - 920", "CL 905 NA # 910 - 920") in OK_RISKS
    assert risk("CL 905 NÂ° 910 - 920", "CL 905 NA # 910 - 921") == "high"
