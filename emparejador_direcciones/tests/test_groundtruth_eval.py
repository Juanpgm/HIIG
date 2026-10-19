"""Tests for tools/eval_groundtruth.py (label merge + precision/recall report)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emparejador import match  # noqa: E402
from tools import eval_groundtruth as ev  # noqa: E402

SAME = ("KR 5 # 10 - 20", "KR 5 # 10 - 20")
DIFF = ("KR 5 # 10 - 20", "KR 9 # 77 - 31")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_pairs_dir(tmp_path: Path, candidates=(), synthetic=(), sanity=()) -> Path:
    d = tmp_path / "pairs"
    d.mkdir(exist_ok=True)
    _write_jsonl(d / "candidates_to_label.jsonl", list(candidates))
    _write_jsonl(d / "synthetic_positives.jsonl", list(synthetic))
    _write_jsonl(d / "random_negatives.jsonl", list(sanity))
    return d


def cand(pid: str, pair, stratum="plate_letter") -> dict:
    return {"pair_id": pid, "addr_a": pair[0], "addr_b": pair[1], "stratum": stratum,
            "npn_a": "1", "npn_b": "2", "same_manzana": False}


def syn(pid: str, pair, tag="lowercase") -> dict:
    return {"pair_id": pid, "addr_a": pair[0], "addr_b": pair[1], "same_door": True, "tier": "synthetic",
            "transform": tag, "source_direccion": pair[0]}


def neg(pid: str, pair) -> dict:
    return {"pair_id": pid, "addr_a": pair[0], "addr_b": pair[1], "same_door": False, "tier": "sanity"}


def write_csv(path: Path, text: str, bom=False, crlf=False) -> Path:
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_precision_recall_f1(self):
        m = ev.compute_metrics(tp=8, fp=2, fn=2, tn=10)
        assert m["precision"] == pytest.approx(0.8)
        assert m["recall"] == pytest.approx(0.8)
        assert m["f1"] == pytest.approx(0.8)

    def test_f1_is_harmonic_mean_when_precision_and_recall_differ(self):
        m = ev.compute_metrics(tp=6, fp=2, fn=6, tn=0)
        assert m["precision"] == pytest.approx(0.75)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(0.6)  # the arithmetic mean would be 0.625

    def test_all_negative_labels_recall_undefined_not_zero_division(self):
        m = ev.compute_metrics(tp=0, fp=0, fn=0, tn=5)
        assert m["precision"] is None and m["recall"] is None and m["f1"] is None

    def test_all_positive_labels_precision_defined_recall_defined(self):
        m = ev.compute_metrics(tp=4, fp=0, fn=1, tn=0)
        assert m["precision"] == 1.0 and m["recall"] == pytest.approx(0.8)

    def test_no_predicted_positives_precision_undefined(self):
        m = ev.compute_metrics(tp=0, fp=0, fn=3, tn=2)
        assert m["precision"] is None and m["recall"] == 0.0 and m["f1"] is None

    def test_empty(self):
        m = ev.compute_metrics(0, 0, 0, 0)
        assert m["n"] == 0 and m["precision"] is None

    def test_format_ratio_na(self):
        assert ev.format_ratio(None) == "n/a"
        assert ev.format_ratio(0.5) == "50.00%"


# ---------------------------------------------------------------------------
# label reading
# ---------------------------------------------------------------------------


class TestReadLabels:
    def test_csv_mixed_case_whitespace_and_blank(self, tmp_path):
        p = write_csv(tmp_path / "l.csv",
                      "pair_id,stratum,addr_a,addr_b,same_door,same_unit,evidence_source,notes\n"
                      "a,s,x,y,YES,,google_maps,\n"
                      "b,s,x,y,  No ,,,\n"
                      "c,s,x,y,Unsure,,,hard\n"
                      "d,s,x,y,,,,\n"
                      "e,s,x,y,maybe,,,\n")
        labels, info = ev.read_labels(p)
        assert labels == {"a": "yes", "b": "no", "c": "unsure", "d": None, "e": None}
        assert info["invalid"] == {"e": "maybe"}

    def test_bom_and_crlf(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,same_door\na,yes\nb,no\n", bom=True, crlf=True)
        labels, _ = ev.read_labels(p)
        assert labels == {"a": "yes", "b": "no"}

    def test_header_case_and_spacing_tolerated(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", " Pair_ID , SAME_DOOR \na,yes\n")
        labels, _ = ev.read_labels(p)
        assert labels == {"a": "yes"}

    def test_jsonl_labels_and_blank_lines(self, tmp_path):
        p = tmp_path / "l.jsonl"
        p.write_text('{"pair_id":"a","same_door":"Yes"}\n\n{"pair_id":"b","same_door":null}\r\n{"pair_id":"c","same_door":"NO"}\n',
                     encoding="utf-8-sig")
        labels, _ = ev.read_labels(p)
        assert labels == {"a": "yes", "b": None, "c": "no"}

    def test_jsonl_booleans_count_as_yes_no(self, tmp_path):
        p = tmp_path / "l.jsonl"
        p.write_text('{"pair_id":"a","same_door":true,"same_unit":false}\n'
                     '{"pair_id":"b","same_door":false,"same_unit":null}\n'
                     '{"pair_id":"c","same_door":"TRUE"}\n', encoding="utf-8")
        labels, info = ev.read_labels(p)
        assert labels == {"a": "yes", "b": "no", "c": "yes"}
        assert info["units"] == {"a": "no", "b": None, "c": None}

    def test_same_unit_values_and_na_spellings(self, tmp_path):
        p = write_csv(tmp_path / "l.csv",
                      "pair_id,same_door,same_unit\n"
                      "a,yes,YES\nb,yes, No \nc,yes,unsure\nd,yes,n/a\ne,yes,N/A\nf,yes,\ng,yes,maybe\nh,no,\n")
        labels, info = ev.read_labels(p)
        assert info["units"] == {"a": "yes", "b": "no", "c": "unsure", "d": "n/a", "e": "n/a", "f": None,
                                 "g": None, "h": None}
        assert info["invalid_units"] == {"g": "maybe"}
        assert labels["h"] == "no"

    def test_n_a_is_not_a_valid_same_door(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,same_door\na,n/a\n")
        labels, info = ev.read_labels(p)
        assert labels == {"a": None} and info["invalid"] == {"a": "n/a"}

    def test_door_and_unit_columns_are_read_independently(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,same_unit,same_door\na,no,yes\nb,yes,no\n")
        labels, info = ev.read_labels(p)
        assert labels == {"a": "yes", "b": "no"}
        assert info["units"] == {"a": "no", "b": "yes"}

    def test_legacy_label_column_is_rejected_with_a_clear_message(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,label\na,yes\n")
        with pytest.raises(ev.GroundTruthError, match="same_door"):
            ev.read_labels(p)
        pj = tmp_path / "l.jsonl"
        pj.write_text('{"pair_id":"a","label":"yes"}\n', encoding="utf-8")
        with pytest.raises(ev.GroundTruthError, match="same_door"):
            ev.read_labels(pj)

    def test_missing_unit_column_is_fine(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,same_door\na,yes\n")
        _, info = ev.read_labels(p)
        assert info["units"] == {"a": None}

    def test_empty_file_gives_no_labels(self, tmp_path):
        p = tmp_path / "l.csv"
        p.write_bytes(b"")
        labels, _ = ev.read_labels(p)
        assert labels == {}
        p2 = write_csv(tmp_path / "l2.csv", "pair_id,same_door\n")
        assert ev.read_labels(p2)[0] == {}

    def test_duplicate_pair_ids_error_clearly(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,same_door\na,yes\na,no\n")
        with pytest.raises(ev.GroundTruthError, match="duplicate pair_id.*a"):
            ev.read_labels(p)

    def test_missing_pair_id_column_is_an_error(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "id,same_door\na,yes\n")
        with pytest.raises(ev.GroundTruthError, match="pair_id"):
            ev.read_labels(p)

    @pytest.mark.parametrize("header", ["pair_id,same-door", "pair_id,samedoor", "pair_id,door", "pair_id,same_unit"])
    def test_missing_same_door_column_is_an_error(self, tmp_path, header):
        p = write_csv(tmp_path / "l.csv", header + "\na,yes\n")
        with pytest.raises(ev.GroundTruthError, match="same_door"):
            ev.read_labels(p)

    def test_header_only_file_without_same_door_is_still_an_error(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,same-door\n")
        with pytest.raises(ev.GroundTruthError, match="same_door"):
            ev.read_labels(p)

    def test_jsonl_row_without_same_door_key_is_an_error(self, tmp_path):
        p = tmp_path / "l.jsonl"
        p.write_text('{"pair_id":"a","same-door":"yes"}\n', encoding="utf-8")
        with pytest.raises(ev.GroundTruthError, match="same_door"):
            ev.read_labels(p)

    def test_jsonl_null_same_door_is_blank_not_missing(self, tmp_path):
        p = tmp_path / "l.jsonl"
        p.write_text('{"pair_id":"a","same_door":null}\n', encoding="utf-8")
        assert ev.read_labels(p)[0] == {"a": None}

    def test_main_missing_same_door_exits_2_with_clear_error(self, tmp_path, capsys):
        d = make_pairs_dir(tmp_path, [cand("c1", SAME)])
        labels = write_csv(tmp_path / "labels.csv", "pair_id,same-door\nc1,yes\n")
        assert ev.main(["--labels", str(labels), "--pairs-dir", str(d)]) == 2
        assert "same_door" in capsys.readouterr().err

    def test_row_without_pair_id_is_skipped(self, tmp_path):
        p = write_csv(tmp_path / "l.csv", "pair_id,same_door\n,yes\na,no\n")
        labels, _ = ev.read_labels(p)
        assert labels == {"a": "no"}

    def test_malformed_jsonl_line_errors(self, tmp_path):
        p = tmp_path / "l.jsonl"
        p.write_text('{"pair_id":"a","same_door":"yes"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ev.GroundTruthError, match="line 2"):
            ev.read_labels(p)

    def test_missing_file_errors(self, tmp_path):
        with pytest.raises(ev.GroundTruthError):
            ev.read_labels(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# pairs loading
# ---------------------------------------------------------------------------


class TestLoadPairs:
    def test_loads_all_three_files(self, tmp_path):
        d = make_pairs_dir(tmp_path, [cand("c1", SAME)], [syn("s1", SAME)], [neg("n1", DIFF)])
        pairs = ev.load_pairs(d)
        assert {p["pair_id"]: p["group"] for p in pairs} == {"c1": "plate_letter", "s1": "synthetic", "n1": "sanity"}

    def test_duplicate_pair_ids_across_files_error(self, tmp_path):
        d = make_pairs_dir(tmp_path, [cand("x", SAME)], [syn("x", SAME)])
        with pytest.raises(ev.GroundTruthError, match="duplicate pair_id"):
            ev.load_pairs(d)

    def test_missing_dir_or_no_files_errors(self, tmp_path):
        with pytest.raises(ev.GroundTruthError):
            ev.load_pairs(tmp_path / "nope")
        (tmp_path / "empty").mkdir()
        with pytest.raises(ev.GroundTruthError):
            ev.load_pairs(tmp_path / "empty")

    def test_missing_optional_files_tolerated(self, tmp_path):
        d = tmp_path / "p"
        d.mkdir()
        _write_jsonl(d / "synthetic_positives.jsonl", [syn("s1", SAME)])
        assert len(ev.load_pairs(d)) == 1


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def run(tmp_path, cands=(), syns=(), negs=(), labels=None, **kw):
    d = make_pairs_dir(tmp_path, cands, syns, negs)
    pairs = ev.load_pairs(d)
    return ev.evaluate(pairs, labels or {}, **kw)


class TestEvaluate:
    def test_synthetic_and_sanity_only_no_human_labels(self, tmp_path):
        r = run(tmp_path, syns=[syn("s1", SAME), syn("s2", DIFF)], negs=[neg("n1", DIFF), neg("n2", SAME)])
        assert r["overall"]["tp"] == 1 and r["overall"]["fn"] == 1
        assert r["overall"]["tn"] == 1 and r["overall"]["fp"] == 1
        assert r["coverage"]["human_pairs"] == 0
        assert r["coverage"]["coverage_pct"] is None
        assert r["groups"]["synthetic"]["n"] == 2 and r["groups"]["sanity"]["n"] == 2

    def test_human_labels_per_stratum_and_coverage(self, tmp_path):
        cands = [cand("c1", SAME, "plate_letter"), cand("c2", DIFF, "plate_letter"),
                 cand("c3", SAME, "opaque_or_degenerate"), cand("c4", SAME, "opaque_or_degenerate")]
        labels = {"c1": "yes", "c2": "no", "c3": "no", "c4": None}
        r = run(tmp_path, cands=cands, labels=labels)
        assert r["groups"]["plate_letter"]["tp"] == 1 and r["groups"]["plate_letter"]["tn"] == 1
        assert r["groups"]["opaque_or_degenerate"]["fp"] == 1
        assert r["coverage"]["human_pairs"] == 4
        assert r["coverage"]["labeled"] == 3
        assert r["coverage"]["blank"] == 1
        assert r["coverage"]["coverage_pct"] == pytest.approx(75.0)
        assert r["human"]["n"] == 3

    def test_unsure_excluded_and_counted(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", SAME), cand("c2", SAME)], labels={"c1": "unsure", "c2": "yes"})
        assert r["coverage"]["unsure"] == 1
        assert r["overall"]["n"] == 1 and r["overall"]["tp"] == 1
        assert r["groups"]["plate_letter"]["excluded_unsure"] == 1

    def test_unknown_label_ids_warn_not_fail(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", SAME)], labels={"c1": "yes", "ghost": "no"})
        assert r["unknown_label_ids"] == ["ghost"]
        assert any("ghost" in w or "unknown" in w for w in r["warnings"])

    def test_builtin_labels_win_over_label_file(self, tmp_path):
        r = run(tmp_path, syns=[syn("s1", SAME)], labels={"s1": "no"})
        assert r["overall"]["tp"] == 1 and r["overall"]["fp"] == 0
        assert r["warnings"]

    def test_empty_labels_and_empty_pairs(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", SAME)], labels={})
        assert r["overall"]["n"] == 0
        assert r["overall"]["precision"] is None
        assert r["coverage"]["coverage_pct"] == 0.0

    def test_all_one_class_labels_report_na(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", DIFF), cand("c2", DIFF)], labels={"c1": "no", "c2": "no"})
        assert r["overall"]["recall"] is None and r["overall"]["precision"] is None
        md = ev.render_markdown(r)
        assert "n/a" in md

    def test_threshold_boundary_exact_score_is_a_match(self, tmp_path):
        a, b = "KR 5 # 10 - 20", "KR 5 # 10 - 20 AP 101"
        score = match(a, b).score
        assert 0.0 < score < 1.0
        r = run(tmp_path, cands=[cand("c1", (a, b))], labels={"c1": "yes"}, threshold=score)
        assert r["overall"]["tp"] == 1  # score == threshold => MATCH, consistent with the matcher
        assert match(a, b, threshold=score).is_match is True
        r2 = run(tmp_path, cands=[cand("c1", (a, b))], labels={"c1": "yes"},
                 threshold=min(1.0, score + 1e-9))
        assert r2["overall"]["fn"] == 1

    def test_stub_scorer_exact_090_counts_as_match(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", DIFF)], labels={"c1": "yes"}, scorer=lambda a, b: 0.90, threshold=0.90)
        assert r["overall"]["tp"] == 1
        r = run(tmp_path, cands=[cand("c1", DIFF)], labels={"c1": "yes"}, scorer=lambda a, b: 0.8999999, threshold=0.90)
        assert r["overall"]["fn"] == 1
        r = run(tmp_path, cands=[cand("c1", DIFF)], labels={"c1": "yes"}, scorer=lambda a, b: 0.90, threshold=0.90)
        assert r["overall"]["tp"] == 1

    def test_false_positive_and_negative_lists_sorted_and_limited(self, tmp_path):
        scores = {"c1": 0.99, "c2": 0.93, "c3": 0.95}
        cands = [cand(k, (k + "a", k + "b")) for k in scores]
        r = run(tmp_path, cands=cands, labels={k: "no" for k in scores},
                scorer=lambda a, b: scores[a[:-1]], top=2)
        fps = r["false_positives"]
        assert [f["pair_id"] for f in fps] == ["c1", "c3"]
        assert fps[0]["score"] == 0.99 and fps[0]["addr_a"] == "c1a"

    def test_false_negatives_sorted_ascending(self, tmp_path):
        scores = {"c1": 0.1, "c2": 0.5, "c3": 0.3}
        cands = [cand(k, (k + "a", k + "b")) for k in scores]
        r = run(tmp_path, cands=cands, labels={k: "yes" for k in scores}, scorer=lambda a, b: scores[a[:-1]], top=10)
        assert [f["pair_id"] for f in r["false_negatives"]] == ["c1", "c3", "c2"]

    def test_unparseable_and_weird_addresses_do_not_crash(self, tmp_path):
        weird = [cand("c1", ("", "")), cand("c2", ("Ñ" * 50_000, "KR 5 # 10 - 20")), cand("c3", ("CL 5 # 1 - 2", "\x00"))]
        r = run(tmp_path, cands=weird, labels={"c1": "yes", "c2": "no", "c3": "yes"})
        assert r["overall"]["n"] == 3

    def test_none_and_non_string_addresses_in_pairs(self, tmp_path):
        d = make_pairs_dir(tmp_path, [{"pair_id": "c1", "addr_a": None, "addr_b": 5, "stratum": "s",
                                        "npn_a": "1", "npn_b": "2", "same_manzana": False}])
        pairs = ev.load_pairs(d)
        r = ev.evaluate(pairs, {"c1": "yes"})
        assert r["overall"]["fn"] == 1

    def test_invalid_threshold_raises(self, tmp_path):
        with pytest.raises(Exception):
            run(tmp_path, cands=[cand("c1", SAME)], labels={"c1": "yes"}, threshold=2.0)

    def test_invalid_label_values_counted(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", SAME)], labels={"c1": None}, invalid={"c1": "maybe"})
        assert r["coverage"]["invalid"] == 1


class TestSyntheticByTransform:
    def test_recall_reported_per_transform_tag(self, tmp_path):
        r = run(tmp_path, syns=[syn("s1", SAME, "lowercase"), syn("s2", DIFF, "lowercase"), syn("s3", SAME, "glued_tokens")])
        by = r["synthetic_by_transform"]
        assert by["lowercase"]["tp"] == 1 and by["lowercase"]["fn"] == 1
        assert by["glued_tokens"]["recall"] == 1.0
        assert "lowercase" in ev.render_markdown(r)

    def test_missing_transform_field_is_tolerated(self, tmp_path):
        row = syn("s1", SAME)
        del row["transform"]
        r = run(tmp_path, syns=[row])
        assert r["synthetic_by_transform"]["unknown"]["tp"] == 1


class TestRenderAndCli:
    def test_markdown_contains_sections(self, tmp_path):
        r = run(tmp_path, syns=[syn("s1", SAME)], negs=[neg("n1", SAME)])
        md = ev.render_markdown(r)
        for token in ("Threshold", "Coverage", "Overall", "synthetic", "sanity", "False positives", "n/a"):
            assert token in md

    def test_main_end_to_end_with_json_out(self, tmp_path, capsys):
        d = make_pairs_dir(tmp_path, [cand("c1", SAME), cand("c2", DIFF)], [syn("s1", SAME)], [neg("n1", DIFF)])
        labels = write_csv(tmp_path / "labels.csv", "pair_id,same_door\nc1,YES\nc2,\n", bom=True, crlf=True)
        out = tmp_path / "res.json"
        rc = ev.main(["--labels", str(labels), "--pairs-dir", str(d), "--json-out", str(out)])
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["threshold"] == 0.90 and data["coverage"]["labeled"] == 1
        assert "Overall" in capsys.readouterr().out

    def test_main_prints_non_ascii_addresses_on_narrow_stdout(self, tmp_path, monkeypatch):
        import io

        d = make_pairs_dir(tmp_path, [], [syn("s1", ("KR 5 # 10 - 20", "KR 5 № 10 - 20 Ñ"))])
        raw = io.BytesIO()
        narrow = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        monkeypatch.setattr(sys, "stdout", narrow)
        assert ev.main(["--pairs-dir", str(d)]) == 0
        narrow.flush()
        assert "№".encode("utf-8") in raw.getvalue()

    def test_main_without_labels_uses_builtin_only(self, tmp_path, capsys):
        d = make_pairs_dir(tmp_path, [cand("c1", SAME)], [syn("s1", SAME)], [neg("n1", DIFF)])
        assert ev.main(["--pairs-dir", str(d)]) == 0
        assert "synthetic" in capsys.readouterr().out

    def test_main_duplicate_labels_returns_error_code(self, tmp_path, capsys):
        d = make_pairs_dir(tmp_path, [cand("c1", SAME)])
        labels = write_csv(tmp_path / "labels.csv", "pair_id,same_door\nc1,yes\nc1,no\n")
        rc = ev.main(["--labels", str(labels), "--pairs-dir", str(d)])
        assert rc == 2
        assert "duplicate" in capsys.readouterr().err

    def test_source_has_no_reference_to_other_project(self):
        text = (_ROOT / "tools" / "eval_groundtruth.py").read_text(encoding="utf-8").lower()
        forbidden = ("normali" + "zador", "cali_" + "address")
        assert not any(word in text for word in forbidden)


# ---------------------------------------------------------------------------
# door (headline) vs unit (secondary), tiers, novel split, frozen scores
# ---------------------------------------------------------------------------

HI = lambda a, b: 0.95  # noqa: E731  (always a MATCH)
LO = lambda a, b: 0.10  # noqa: E731  (never a MATCH)


class TestHeadlineBucket:
    def test_human_bucket_excludes_synthetic_and_sanity_pairs(self, tmp_path):
        cands = [cand("c1", SAME), cand("c2", DIFF)]
        syns = [syn(f"s{i}", SAME) for i in range(5)]
        negs = [neg(f"n{i}", DIFF) for i in range(5)]
        scorer = lambda a, b: 0.95 if (a, b) == SAME else 0.10  # noqa: E731
        r = run(tmp_path, cands=cands, syns=syns, negs=negs, labels={"c1": "yes", "c2": "yes"}, scorer=scorer)
        assert r["human"]["n"] == 2
        assert r["human"]["tp"] == 1 and r["human"]["fn"] == 1
        assert r["human"]["recall"] == pytest.approx(0.5)
        assert r["overall"]["n"] == 12
        assert r["overall"]["recall"] == pytest.approx(6 / 7)
        assert r["groups"]["synthetic"]["n"] == 5 and r["groups"]["sanity"]["n"] == 5

    def test_sanity_false_positives_never_reach_the_human_bucket(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", SAME)], negs=[neg("n1", DIFF)], labels={"c1": "no"}, scorer=HI)
        assert r["human"]["fp"] == 1 and r["human"]["n"] == 1
        assert r["overall"]["fp"] == 2


class TestSameDoorVsSameUnit:
    def _r(self, tmp_path, cands, labels, units, scorer=HI, invalid_units=None):
        return run(tmp_path, cands=cands, labels=labels, scorer=scorer, units=units, invalid_units=invalid_units)

    def test_same_unit_never_affects_headline_metrics(self, tmp_path):
        cands = [cand("c1", SAME), cand("c2", SAME), cand("c3", DIFF), cand("c4", SAME)]
        labels = {"c1": "yes", "c2": "yes", "c3": "no", "c4": "unsure"}
        base = self._r(tmp_path, cands, labels, units={})
        flipped = self._r(tmp_path, cands, labels, units={"c1": "no", "c2": "yes", "c3": "yes", "c4": "no"})
        for key in ("overall", "human", "groups", "coverage", "false_positives", "false_negatives"):
            assert flipped[key] == base[key], key

    def test_door_and_unit_are_scored_independently(self, tmp_path):
        cands = [cand("a", SAME), cand("b", SAME), cand("c", SAME)]
        labels = {"a": "yes", "b": "yes", "c": "yes"}
        units = {"a": "yes", "b": "no", "c": "n/a"}
        r = self._r(tmp_path, cands, labels, units)
        assert r["human"]["tp"] == 3 and r["human"]["fp"] == 0  # door: all three are the same building
        u = r["same_unit"]["overall"]
        assert (u["tp"], u["fp"], u["fn"], u["tn"]) == (1, 1, 0, 0)  # unit: the matcher merged two different units
        assert u["precision"] == pytest.approx(0.5)
        assert r["same_unit"]["coverage"]["n/a"] == 1

    def test_unit_uses_the_threshold_boundary(self, tmp_path):
        cands = [cand("a", SAME)]
        r = self._r(tmp_path, cands, {"a": "yes"}, {"a": "yes"}, scorer=lambda a, b: 0.90)
        assert r["same_unit"]["overall"]["tp"] == 1
        r = self._r(tmp_path, cands, {"a": "yes"}, {"a": "yes"}, scorer=lambda a, b: 0.8999999)
        assert r["same_unit"]["overall"]["fn"] == 1

    def test_blank_unit_counts_as_na(self, tmp_path):
        r = self._r(tmp_path, [cand("a", SAME)], {"a": "yes"}, {"a": None})
        assert r["same_unit"]["coverage"]["n/a"] == 1 and r["same_unit"]["overall"]["n"] == 0

    def test_unit_yes_no_with_door_no_warns_and_is_ignored(self, tmp_path):
        cands = [cand("bad", SAME), cand("ok", SAME), cand("fine", SAME)]
        labels = {"bad": "no", "ok": "no", "fine": "yes"}
        r = self._r(tmp_path, cands, labels, {"bad": "yes", "ok": "n/a", "fine": "yes"})
        unit_warnings = [w for w in r["warnings"] if "same_unit" in w]
        assert len(unit_warnings) == 1 and "bad" in unit_warnings[0] and "1 pair(s)" in unit_warnings[0]
        assert r["human"]["fp"] == 2  # both door=no pairs are still counted for the headline
        assert r["same_unit"]["overall"]["n"] == 1  # only the consistent pair
        assert r["same_unit"]["coverage"]["ignored_door_no"] == 1
        assert "WARNING" in ev.render_markdown(r)

    def test_unit_unsure_is_excluded_and_counted(self, tmp_path):
        r = self._r(tmp_path, [cand("a", SAME), cand("b", SAME)], {"a": "yes", "b": "yes"}, {"a": "unsure", "b": "yes"})
        assert r["same_unit"]["coverage"]["unsure"] == 1
        assert r["same_unit"]["overall"]["n"] == 1 and r["same_unit"]["overall"]["excluded_unsure"] == 1

    def test_unit_ignored_when_door_is_unsure_or_blank(self, tmp_path):
        r = self._r(tmp_path, [cand("a", SAME), cand("b", SAME)], {"a": "unsure", "b": None}, {"a": "yes", "b": "no"})
        assert r["same_unit"]["overall"]["n"] == 0
        assert r["coverage"]["unsure"] == 1 and r["coverage"]["blank"] == 1
        assert not r["warnings"]

    def test_invalid_unit_value_is_counted_not_scored(self, tmp_path):
        r = self._r(tmp_path, [cand("a", SAME)], {"a": "yes"}, {"a": None}, invalid_units={"a": "maybe"})
        assert r["same_unit"]["coverage"]["invalid"] == 1 and r["same_unit"]["overall"]["n"] == 0

    def test_invalid_unit_with_door_no_is_counted_and_warned_not_dropped(self, tmp_path):
        r = self._r(tmp_path, [cand("a", DIFF)], {"a": "no"}, {"a": None}, invalid_units={"a": "maybe"})
        assert r["same_unit"]["coverage"]["invalid"] == 1
        assert any("invalid same_unit" in w and "a" in w for w in r["warnings"])
        assert r["overall"]["n"] == 1  # the door label still counts

    def test_invalid_unit_with_door_yes_also_warns(self, tmp_path):
        r = self._r(tmp_path, [cand("a", SAME)], {"a": "yes"}, {"a": None}, invalid_units={"a": "maybe"})
        assert any("invalid same_unit" in w for w in r["warnings"])

    def test_main_invalid_unit_with_door_no_warns_in_report(self, tmp_path, capsys):
        d = make_pairs_dir(tmp_path, [cand("c1", DIFF)])
        labels = write_csv(tmp_path / "labels.csv", "pair_id,same_door,same_unit\nc1,no,maybe\n")
        assert ev.main(["--labels", str(labels), "--pairs-dir", str(d)]) == 0
        out = capsys.readouterr().out
        assert "invalid same_unit" in out and "invalid: 1" in out

    def test_all_one_class_unit_labels_report_na_not_zero_division(self, tmp_path):
        r = self._r(tmp_path, [cand("a", SAME)], {"a": "yes"}, {"a": "yes"}, scorer=LO)
        assert r["same_unit"]["overall"]["precision"] is None and r["same_unit"]["overall"]["recall"] == 0.0
        assert "n/a" in ev.render_markdown(r)

    def test_unit_section_rendered_separately(self, tmp_path):
        md = ev.render_markdown(self._r(tmp_path, [cand("a", SAME)], {"a": "yes"}, {"a": "yes"}))
        assert "Same-unit (secondary" in md
        assert md.index("Same-unit") > md.index("Per stratum")

    def test_main_reads_both_columns_from_a_csv(self, tmp_path):
        d = make_pairs_dir(tmp_path, [cand("c1", SAME), cand("c2", SAME)])
        labels = write_csv(tmp_path / "labels.csv", "pair_id,same_door,same_unit\nc1,yes,no\nc2,no,\n")
        out = tmp_path / "res.json"
        assert ev.main(["--labels", str(labels), "--pairs-dir", str(d), "--json-out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["human"]["tp"] == 1 and data["human"]["fp"] == 1
        assert data["same_unit"]["overall"]["fp"] == 1


class TestNovelSplit:
    def test_known_and_novel_recall_are_reported_separately(self, tmp_path):
        syns = [syn("k1", SAME, "lowercase"), syn("k2", SAME, "combo"),
                syn("n1", SAME, "novel_type_alias"), syn("n2", DIFF, "novel_complement_alias"),
                syn("n3", DIFF, "novel_number_word")]
        r = run(tmp_path, syns=syns)
        split = r["synthetic_recall_split"]
        assert split["known_vocab"]["recall"] == 1.0 and split["known_vocab"]["n"] == 2
        assert split["novel"]["n"] == 3 and split["novel"]["tp"] == 1 and split["novel"]["fn"] == 2
        assert split["novel"]["recall"] == pytest.approx(1 / 3)
        md = ev.render_markdown(r)
        assert "novel_* tags" in md and "NOT independent evidence" in md

    def test_no_novel_rows_gives_na_not_error(self, tmp_path):
        r = run(tmp_path, syns=[syn("k1", SAME, "lowercase")])
        assert r["synthetic_recall_split"]["novel"]["recall"] is None
        assert "n/a" in ev.render_markdown(r)

    def test_is_novel_tag(self):
        assert ev.is_novel_tag("novel_x") and not ev.is_novel_tag("lowercase") and not ev.is_novel_tag(None)


class TestHonestHeaders:
    def test_report_says_it_is_not_population_precision(self, tmp_path):
        md = ev.render_markdown(run(tmp_path, cands=[cand("a", SAME)], labels={"a": "yes"}))
        assert "not population precision" in md
        assert "Human-labeled only" not in md

    def test_builtin_same_door_false_is_a_negative(self, tmp_path):
        r = run(tmp_path, negs=[neg("n1", SAME)])
        assert r["overall"]["fp"] == 1


class TestFrozenScores:
    def test_load_scores_from_dir_and_file(self, tmp_path):
        _write_jsonl(tmp_path / "matcher_scores_sidecar.jsonl",
                     [{"pair_id": "a", "score": 0.5, "decision": "NO_MATCH"}, {"pair_id": "b", "score": 1, "decision": "MATCH"}])
        assert ev.load_scores(tmp_path) == {"a": 0.5, "b": 1.0}
        assert ev.load_scores(tmp_path / "matcher_scores_sidecar.jsonl") == {"a": 0.5, "b": 1.0}

    def test_missing_or_malformed_scores_error(self, tmp_path):
        with pytest.raises(ev.GroundTruthError):
            ev.load_scores(tmp_path)
        _write_jsonl(tmp_path / "matcher_scores_sidecar.jsonl", [{"pair_id": "a", "score": "high"}])
        with pytest.raises(ev.GroundTruthError, match="numeric"):
            ev.load_scores(tmp_path)

    def test_frozen_scores_override_live_scoring_with_a_warning(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", SAME)], labels={"c1": "yes"}, scorer=LO, scores={"c1": 0.90})
        assert r["human"]["tp"] == 1
        assert any("frozen" in w for w in r["warnings"])

    def test_pairs_missing_from_the_sidecar_are_scored_live(self, tmp_path):
        r = run(tmp_path, cands=[cand("c1", SAME)], labels={"c1": "yes"}, scorer=HI, scores={"other": 0.0})
        assert r["human"]["tp"] == 1

    def test_main_scores_dir_option(self, tmp_path, capsys):
        d = make_pairs_dir(tmp_path, [], [syn("s1", DIFF)])
        priv = tmp_path / "priv"
        priv.mkdir()
        _write_jsonl(priv / "matcher_scores_sidecar.jsonl", [{"pair_id": "s1", "score": 0.97, "decision": "MATCH"}])
        assert ev.main(["--pairs-dir", str(d), "--scores-dir", str(priv)]) == 0
        assert "frozen" in capsys.readouterr().out
        assert ev.main(["--pairs-dir", str(d), "--scores-dir", str(tmp_path / "nope")]) == 2
