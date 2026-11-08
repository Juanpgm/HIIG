"""JEV audit adapter (``cali_address.jev_check``) built on the sibling ``validacion_jev`` project.

Everything here runs against a fake client / fake transport and a sentinel key: nothing touches the network and the
real credential is never read. Addresses are invented (improbable numbers, 900+).
"""

from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd
import pytest

from cali_address import jev_check as jc
from cali_address.jev_check import AuditRow, PrivacyError

SENTINEL_KEY = "SENTINEL-NOT-A-REAL-KEY-0000"
QUESTION = "misma_ubicacion"


def _jev():
    return jc.load_validacion_jev()


def noul_result(p):
    mod = _jev()
    result = mod.client.QueryResult()
    result.answers[QUESTION] = {"type": "noul", "noul": p}
    return result


class FakeClient:
    """Records every state it is asked about; answers from ``p_by_input`` (default 0.95 = a confident yes)."""

    def __init__(self, p_by_input=None, default=0.95, error_inputs=()):
        self.p_by_input = p_by_input or {}
        self.default = default
        self.error_inputs = set(error_inputs)
        self.states = []

    def query(self, state, questions):
        self.states.append(json.loads(json.dumps(state)))
        assert set(questions) <= {"misma_ubicacion", "estado_adecuado"}
        text = state["direccion_original"]
        if text in self.error_inputs:
            result = _jev().client.QueryResult()
            result.errors[QUESTION] = "HTTP 500: boom"
            return result
        return noul_result(self.p_by_input.get(text, self.default))


def row(i, origin="nomenclatura_public", a=None, b=None, **tags):
    return AuditRow(id=str(i), origin=origin, input_text=a if a is not None else f"CL {900 + i} # 950 - {i}",
                    output_text=b if b is not None else f"CALLE {900 + i} 950 {i}", tags=tags)


# --------------------------------------------------------------------------- loader
def test_loader_finds_the_sibling_project():
    mod = jc.load_validacion_jev()
    assert hasattr(mod, "client") and hasattr(mod, "validators") and hasattr(mod, "metrics") and hasattr(mod, "questions")


def test_loader_reports_a_clear_error_when_the_package_cannot_be_found(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "validacion_jev", None)  # make `import validacion_jev` fail
    monkeypatch.setenv("VALIDACION_JEV_SRC", str(tmp_path / "nowhere"))
    monkeypatch.setattr(jc, "SIBLING_SRC", str(tmp_path / "also_nowhere"))
    with pytest.raises(jc.JevUnavailableError, match="VALIDACION_JEV_SRC"):
        jc.load_validacion_jev()


# --------------------------------------------------------------------------- privacy guard
@pytest.mark.parametrize("origin", ["dev", "frozen_test", "train_pairs", "unlabeled", "", None, "mystery", "NOMENCLATURA_PUBLIC"])
def test_private_or_unknown_origins_are_refused_and_nothing_leaves(origin):
    client = FakeClient()
    with pytest.raises(PrivacyError):
        jc.audit_rows([row(1, origin=origin)], client)
    assert client.states == []


def test_one_private_row_poisons_the_whole_batch():
    client = FakeClient()
    rows = [row(1), row(2), row(3, origin="dev")]
    with pytest.raises(PrivacyError):
        jc.audit_rows(rows, client)
    assert client.states == []


def test_dry_run_payloads_apply_the_same_guard():
    with pytest.raises(PrivacyError):
        jc.dry_run_payloads([row(1, origin="frozen_test")])


def test_privacy_error_never_echoes_the_addresses():
    with pytest.raises(PrivacyError) as info:
        jc.audit_rows([row(7, origin="dev", a="CL 901 # 777 - 1")], FakeClient())
    assert "777" not in str(info.value) and "dev" in str(info.value)


@pytest.mark.parametrize("origin", ["nomenclatura_public", "synthetic"])
def test_public_origins_are_sent(origin):
    client = FakeClient()
    result = jc.audit_rows([row(1, origin=origin)], client)
    assert len(client.states) == 1 and result.summary["counts"]["ok"] == 1


def test_allow_private_flag_is_the_only_way_to_send_private_rows():
    client = FakeClient()
    jc.audit_rows([row(1, origin="dev")], client, allow_private=True)
    assert len(client.states) == 1


def test_the_pipeline_never_passes_allow_private():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for folder in ("src", "scripts"):
        for dirpath, _dirs, files in os.walk(os.path.join(root, folder)):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                text = open(path, encoding="utf-8").read()
                if re.search(r"allow[_-]private\s*=\s*True|--allow-private[\"']\s*[,\]]", text) and name not in (
                    "jev_check.py", "audit_with_jev.py"
                ):
                    offenders.append(path)
    assert offenders == []


def test_default_cli_arguments_never_enable_allow_private():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("audit_with_jev_t", os.path.join(root, "scripts", "audit_with_jev.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_parser().parse_args(["--source", "nomenclatura"]).allow_private is False


# --------------------------------------------------------------------------- payload shape
def test_payload_contains_only_address_strings():
    rows = [row(1, ndnpn="760010000000000000000000000905", lat=3.4, estado_nomenclatura="cert")]
    client = FakeClient()
    jc.audit_rows(rows, client)
    assert set(client.states[0]) == {"direccion_original", "direccion_normalizada"}


def test_dry_run_payloads_are_the_exact_bodies_and_need_no_client():
    payloads = jc.dry_run_payloads([row(1), row(2)], model="jev-latest")
    assert [p["id"] for p in payloads] == ["1", "2"]
    body = payloads[0]["body"]
    assert set(body) == {"state", "model", "questions"} and set(body["state"]) == {"direccion_original", "direccion_normalizada"}
    assert body["model"] == "jev-latest" and QUESTION in body["questions"]


def test_dry_run_respects_the_row_cap_and_skips_unsendable_rows():
    rows = [row(i) for i in range(10)] + [row(99, a="   ")]
    assert len(jc.dry_run_payloads(rows, max_rows=4)) == 4
    assert [p["id"] for p in jc.dry_run_payloads([row(1, a="  "), row(2)])] == ["2"]


def test_long_strings_are_truncated_and_flagged_not_dropped():
    client = FakeClient()
    result = jc.audit_rows([row(1, a="CL " + "9" * 400)], client)
    assert len(client.states[0]["direccion_original"]) == 300
    assert "truncated" in result.results[0]["flags"]


# --------------------------------------------------------------------------- gate boundaries
@pytest.mark.parametrize("p,label", [
    (0.85, "yes"), (0.8499, "indeterminate"), (0.9, "yes"), (1.0, "yes"),
    (0.15, "no"), (0.1501, "indeterminate"), (0.0, "no"),
    (0.60, "indeterminate"), (0.5, "indeterminate"), (0.40, "indeterminate"),
])
def test_label_at_the_exact_gate_boundaries(p, label):
    result = jc.audit_rows([row(1)], FakeClient(default=p))
    assert result.results[0]["label"] == label


def test_gate_names_at_the_documented_boundaries():
    mod = _jev()
    th = mod.validators.Thresholds()
    assert mod.validators.gate(0.85, th) == "auto"
    assert mod.validators.gate(0.60, th) == "review"
    assert mod.validators.gate(0.5999, th) == "escalate"
    result = jc.audit_rows([row(1)], FakeClient(default=0.15))
    assert result.results[0]["gates"][QUESTION] == "auto"


def test_indeterminate_rows_are_counted_but_excluded_from_agreement():
    client = FakeClient(p_by_input={"CL 901 # 950 - 1": 0.5, "CL 902 # 950 - 2": 0.95, "CL 903 # 950 - 3": 0.05})
    result = jc.audit_rows([row(1), row(2), row(3)], client)
    counts, agreement = result.summary["counts"], result.summary["agreement"]
    assert counts["indeterminate"] == 1 and counts["labeled"] == 2
    assert agreement["n"] == 2 and agreement["agree"] == 1 and agreement["rate"] == 0.5
    assert len(agreement["ci95"]) == 2


# --------------------------------------------------------------------------- failures never count as agreement
def test_a_row_the_api_failed_on_is_marked_and_excluded():
    client = FakeClient(error_inputs={"CL 902 # 950 - 2"})
    result = jc.audit_rows([row(1), row(2)], client)
    failed = [r for r in result.results if r["id"] == "2"][0]
    assert failed["status"] == "error" and "error" in failed["flags"] and failed["label"] is None
    assert result.summary["counts"]["error"] == 1
    assert result.summary["agreement"]["n"] == 1
    assert result.summary["api_failure_rate"] == 0.5


def test_every_row_failing_leaves_no_valid_evidence():
    client = FakeClient(error_inputs={"CL 901 # 950 - 1"})
    result = jc.audit_rows([row(1)], client)
    assert result.summary["has_valid_evidence"] is False
    assert result.summary["agreement"]["rate"] is None and result.summary["agreement"]["ci95"] is None


def _transport(status=200, body=b"", exc=None):
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(json.loads(payload.decode("utf-8")))
        if exc is not None:
            raise exc
        return status, {}, body

    transport.calls = calls
    return transport


def _real_client(transport):
    mod = _jev()
    config = mod.client.JevConfig(api_key=SENTINEL_KEY)
    return mod.client.JevClient(config, transport=transport, sleep=lambda s: None)


@pytest.mark.parametrize("exc", [TimeoutError("timed out"), ConnectionError("reset")])
def test_timeouts_and_connection_errors_mark_the_row_instead_of_raising(exc):
    transport = _transport(exc=exc)
    result = jc.audit_rows([row(1)], _real_client(transport))
    assert result.results[0]["status"] == "error"
    assert result.summary["counts"]["error"] == 1 and result.summary["agreement"]["n"] == 0
    assert SENTINEL_KEY not in json.dumps(result.results, default=str)


@pytest.mark.parametrize("body", [
    b"not json", b"{}", b'{"answers": []}', b'{"answers": {"misma_ubicacion": {"type": "noul", "noul": 7}}}',
    b'{"answers": {"misma_ubicacion": {"type": "noul", "noul": "yes"}}}', b'{"answers": {"misma_ubicacion": {"type": "choice"}}}',
])
def test_malformed_api_responses_become_error_rows(body):
    result = jc.audit_rows([row(1)], _real_client(_transport(body=body)))
    assert result.results[0]["status"] == "error"
    assert result.summary["counts"]["ok"] == 0 and result.summary["agreement"]["n"] == 0


def test_a_server_error_status_is_an_error_row_after_retries():
    transport = _transport(status=500, body=b'{"error": "down"}')
    result = jc.audit_rows([row(1)], _real_client(transport))
    assert result.results[0]["status"] == "error" and len(transport.calls) == 3


def test_a_rejected_key_aborts_the_run_with_an_authentication_error():
    transport = _transport(status=401, body=b'{"error": "bad key"}')
    with pytest.raises(_jev().client.AuthenticationError):
        jc.audit_rows([row(1), row(2)], _real_client(transport))
    assert len(transport.calls) == 1


def test_a_valid_transport_response_flows_through_end_to_end():
    body = json.dumps({"answers": {QUESTION: {"type": "noul", "noul": 0.97}}, "usage": {"input_tokens": 5, "output_tokens": 1}})
    result = jc.audit_rows([row(1)], _real_client(_transport(body=body.encode())))
    assert result.results[0]["label"] == "yes" and result.summary["agreement"]["rate"] == 1.0


# --------------------------------------------------------------------------- duplicates / caps / empties
def test_duplicate_pairs_are_not_sent_twice():
    client = FakeClient()
    result = jc.audit_rows([row(1), row(1), row(2)], client)
    assert len(client.states) == 2
    statuses = [r["status"] for r in result.results]
    assert statuses.count("skipped") == 1 and result.summary["counts"]["skipped_by_reason"] == {"duplicate_pair": 1}


def test_duplicate_row_ids_are_kept_apart():
    result = jc.audit_rows([row(1), row(1, a="CL 904 # 950 - 4", b="CALLE 904 950 4")], FakeClient())
    assert len({r["id"] for r in result.results}) == 2


def test_default_cap_is_300_rows_and_is_reported():
    assert jc.DEFAULT_MAX_ROWS == 300
    client = FakeClient()
    rows = [row(i) for i in range(305)]
    result = jc.audit_rows(rows, client)
    assert len(client.states) == 300 and result.capped and result.total_rows == 305


def test_explicit_cap_below_the_row_count_limits_the_calls():
    client = FakeClient()
    result = jc.audit_rows([row(i) for i in range(10)], client, max_rows=3)
    assert len(client.states) == 3 and result.capped


def test_cap_must_be_positive():
    with pytest.raises(ValueError):
        jc.audit_rows([row(1)], FakeClient(), max_rows=0)


def test_empty_input_is_a_clean_empty_result():
    client = FakeClient()
    result = jc.audit_rows([], client)
    assert client.states == [] and result.results == [] and result.summary["has_valid_evidence"] is False
    assert result.summary["counts"]["total"] == 0


@pytest.mark.parametrize("a,b", [(None, "CALLE 900 1 2"), ("CL 900 # 1 - 2", None), ("", "x"), ("   ", "y"), (float("nan"), "z")])
def test_rows_with_a_missing_side_are_skipped_not_sent(a, b):
    client = FakeClient()
    r = AuditRow(id="1", origin="synthetic", input_text=a, output_text=b)
    result = jc.audit_rows([r], client)
    assert client.states == [] and result.results[0]["status"] == "skipped"
    assert result.summary["counts"]["skipped_by_reason"] == {"invalid_input": 1}


# --------------------------------------------------------------------------- subgroups and disagreement types
def test_agreement_is_split_by_subgroup_and_never_mixes_groups():
    rows = [
        row(1, estado_nomenclatura="cert", zona="URB"),
        row(2, estado_nomenclatura="cert", zona="URB"),
        row(3, estado_nomenclatura="pcer", zona="EXP"),
        row(4, estado_nomenclatura="pcer", zona="URB"),
    ]
    client = FakeClient(p_by_input={"CL 903 # 950 - 3": 0.02, "CL 904 # 950 - 4": 0.02})
    result = jc.audit_rows(rows, client)
    by_estado = result.groups["estado_nomenclatura"]
    assert by_estado["cert"]["agreement"]["rate"] == 1.0 and by_estado["cert"]["agreement"]["n"] == 2
    assert by_estado["pcer"]["agreement"]["rate"] == 0.0 and by_estado["pcer"]["agreement"]["n"] == 2
    assert result.groups["zona"]["EXP"]["agreement"]["agree"] == 0
    assert result.groups["zona"]["URB"]["agreement"]["n"] == 3


def test_boolean_cleaning_flags_become_true_false_groups():
    rows = [row(1, marker_removed=True), row(2, marker_removed=False), row(3, marker_removed=True)]
    result = jc.audit_rows(rows, FakeClient())
    assert set(result.groups["marker_removed"]) == {"true", "false"}
    assert result.groups["marker_removed"]["true"]["counts"]["total"] == 2


def test_group_with_only_failures_reports_no_rate():
    client = FakeClient(error_inputs={"CL 901 # 950 - 1"})
    result = jc.audit_rows([row(1, zona="EXP"), row(2, zona="URB")], client)
    assert result.groups["zona"]["EXP"]["agreement"]["rate"] is None
    assert result.groups["zona"]["EXP"]["counts"]["error"] == 1


@pytest.mark.parametrize("a,b,kind", [
    ("CL 900 # 12 - 34", "CALLE 900 12 35", "numbers_differ"),
    ("CL 900 # 12 - 34", "CALLE 900 12 34", "same_tokens_after_abbreviation"),
    ("CL 900 # 12 - 34 AP 5", "CALLE 900 12 34", "tokens_dropped"),
    ("CL 900 # 12 - 34", "CALLE 900 12 34 AP 5", "tokens_added"),
    ("CL 900 A # 12 - 34", "CALLE 900 B 12 34", "letters_differ"),
])
def test_disagreement_types_are_generic_patterns(a, b, kind):
    assert jc.disagreement_type(a, b) == kind


def test_disagreement_type_never_raises_on_odd_input():
    for a, b in [(None, "x"), ("", ""), (float("nan"), None), ("!!!", "???")]:
        assert isinstance(jc.disagreement_type(a, b), str)


def test_disagreements_are_counted_by_type_without_address_text():
    rows = [row(1, a="CL 900 # 12 - 34", b="CALLE 900 12 35"), row(2)]
    client = FakeClient(p_by_input={"CL 900 # 12 - 34": 0.02})
    result = jc.audit_rows(rows, client)
    assert result.disagreement_types == {"numbers_differ": 1}
    dumped = json.dumps(result.summary_for_json())
    assert "CL 900" not in dumped and "CALLE 900" not in dumped


# --------------------------------------------------------------------------- scores / reliability / AUC
def test_reliability_table_and_auc_use_the_model_score_when_present():
    rows = []
    for i in range(10):
        r = row(i)
        r.score = i / 10
        rows.append(r)
    client = FakeClient(p_by_input={r.input_text: (0.95 if r.score >= 0.5 else 0.05) for r in rows})
    result = jc.audit_rows(rows, client)
    assert result.summary["auc"] == 1.0
    assert sum(b["n"] for b in result.summary["reliability"]["bins"]) == 10


def test_auc_and_kappa_are_undefined_when_a_class_is_missing():
    result = jc.audit_rows([row(1), row(2)], FakeClient())
    assert result.summary["auc"] is None and result.summary["kappa"] is None


# --------------------------------------------------------------------------- sampling
def test_sampling_is_deterministic_by_seed_and_sorted():
    a = jc.sample_positions(1000, 50, seed=42)
    assert a == jc.sample_positions(1000, 50, seed=42) and a == sorted(a) and len(set(a)) == 50
    assert a != jc.sample_positions(1000, 50, seed=43)


@pytest.mark.parametrize("total,n,expected", [(10, 50, 10), (0, 5, 0), (5, 0, 0), (7, 7, 7)])
def test_sample_larger_than_the_data_returns_everything(total, n, expected):
    assert len(jc.sample_positions(total, n, seed=1)) == expected


def test_sample_rejects_negative_sizes():
    with pytest.raises(ValueError):
        jc.sample_positions(10, -1, seed=1)


def _nomenclatura_frame(n=30):
    return pd.DataFrame({
        "nddirecc_original": [f"CL {900 + i} # 950 - {i} (Direccion Catastral)" for i in range(n)],
        "nddirecc": [f"CL {900 + i} # 950 - {i}" for i in range(n)],
        "ndidestado": ["cert" if i % 2 else "pcer" for i in range(n)],
        "ndidtipzon": ["URB" if i % 3 else "EXP" for i in range(n)],
        "marker_removed": [i % 2 == 0 for i in range(n)],
        "rebuilt_from_parts": [False] * n,
        "ndnpn": ["760010000000000000000000000905"] * n,  # must never travel
    })


def test_nomenclatura_rows_are_public_origin_raw_input_vs_canonical_output():
    rows = jc.rows_from_nomenclatura(_nomenclatura_frame(), n=10, seed=42)
    assert len(rows) == 10 and {r.origin for r in rows} == {"nomenclatura_public"}
    assert all("Direccion Catastral" in r.input_text and "Direccion Catastral" not in r.output_text for r in rows)
    assert set(rows[0].tags) == {"estado_nomenclatura", "zona", "marker_removed", "rebuilt_from_parts"}
    assert "ndnpn" not in rows[0].tags and "ndnpn" not in rows[0].__dict__


def test_nomenclatura_rows_drop_blank_addresses_and_are_seeded():
    frame = _nomenclatura_frame(8)
    frame.loc[0, "nddirecc"] = None
    frame.loc[1, "nddirecc_original"] = "  "
    a = jc.rows_from_nomenclatura(frame, n=100, seed=7)
    assert len(a) == 6 and [r.id for r in a] == [r.id for r in jc.rows_from_nomenclatura(frame, n=100, seed=7)]


def test_nomenclatura_rows_need_the_expected_columns():
    with pytest.raises(KeyError, match="nddirecc_original"):
        jc.rows_from_nomenclatura(pd.DataFrame({"nddirecc": ["x"]}), n=1, seed=1)


def test_synthetic_rows_go_through_the_injected_normalizer_and_are_synthetic_origin():
    addresses = [f"CL {900 + i} # 950 - {i}" for i in range(20)]
    seen = {}

    def normalize(queries):
        seen["queries"] = list(queries)
        return pd.DataFrame({"direccion_normalizada": [q.upper() for q in queries],
                             "estado": ["OK"] * len(queries), "confiabilidad": [0.9] * len(queries),
                             "fuente_normalizacion": ["catastro"] * len(queries)})

    rows = jc.rows_from_synthetic(addresses, n=8, seed=42, normalize=normalize)
    assert len(rows) == 8 and {r.origin for r in rows} == {"synthetic"} and len(seen["queries"]) == 8
    assert all(r.input_text == q for r, q in zip(rows, seen["queries"]))
    assert rows[0].score == 0.9 and rows[0].tags["estado"] == "OK"
    again = jc.rows_from_synthetic(addresses, n=8, seed=42, normalize=normalize)
    assert [r.input_text for r in again] == [r.input_text for r in rows]


def test_synthetic_rows_with_no_output_text_are_kept_but_are_skipped_when_audited():
    def normalize(queries):
        return pd.DataFrame({"direccion_normalizada": [None] * len(queries), "estado": ["SIN_MATCH"] * len(queries)})

    rows = jc.rows_from_synthetic(["CL 901 # 950 - 1"], n=1, seed=1, normalize=normalize)
    client = FakeClient()
    result = jc.audit_rows(rows, client)
    assert client.states == [] and result.summary["counts"]["skipped"] == 1


def test_synthetic_with_no_addresses_is_empty():
    assert jc.rows_from_synthetic([], n=5, seed=1, normalize=lambda q: pd.DataFrame()) == []


def test_synthetic_rows_carry_band_and_source_tags_and_numeric_extras_when_the_normalizer_returns_them():
    addresses = [f"CL {900 + i} # 950 - {i}" for i in range(20)]

    def normalize(queries):
        n = len(queries)
        return pd.DataFrame({"direccion_normalizada": [q.upper() for q in queries], "estado": ["OK"] * n,
                             "confianza": [0.8] * n, "banda_confiabilidad": ["alta", "media"] * (n // 2),
                             "fuente": ["nomenclatura_cert"] * n, "p_predio_cal": [0.91] * (n - 1) + [None],
                             "p_manzana_cal": [0.97] * n})

    rows = jc.rows_from_synthetic(addresses, n=6, seed=3, normalize=normalize)
    assert rows[0].tags["banda_confiabilidad"] == "alta" and rows[1].tags["banda_confiabilidad"] == "media"
    assert rows[0].tags["fuente"] == "nomenclatura_cert"
    assert rows[0].extra["p_predio_cal"] == pytest.approx(0.91) and rows[0].extra["p_manzana_cal"] == pytest.approx(0.97)
    assert rows[-1].extra["p_predio_cal"] is None                       # a missing probability stays missing, not 0
    assert [r.extra["noise"] for r in rows] == ["light", "heavy"] * 3   # corrupt_many: variant 0 mild, variant 1 heavy
    assert "p_predio_cal" not in rows[0].tags and "noise" not in rows[0].tags   # extras are never grouped


def test_synthetic_rows_without_calibrated_columns_still_get_the_noise_extra_only():
    rows = jc.rows_from_synthetic(["CL 901 # 950 - 1", "CL 902 # 950 - 2"], n=2, seed=1,
                                  normalize=lambda q: pd.DataFrame({"direccion_normalizada": list(q), "estado": ["OK"] * len(q)}))
    assert set(rows[0].extra) == {"noise"} and "banda_confiabilidad" not in rows[0].tags


def test_extras_are_written_as_extra_columns_and_never_become_groups(tmp_path):
    r = row(1, estado="OK")
    r.extra = {"p_predio_cal": 0.9, "noise": "heavy"}
    result = jc.audit_rows([r, row(2, estado="OK")], FakeClient(default=0.95), source="synthetic")
    paths = jc.write_audit(result, str(tmp_path))
    frame = pd.read_parquet(paths["rows"])
    assert frame.loc[frame["id"] == "1", "extra_p_predio_cal"].iloc[0] == pytest.approx(0.9)
    assert frame.loc[frame["id"] == "1", "extra_noise"].iloc[0] == "heavy"
    assert pd.isna(frame.loc[frame["id"] == "2", "extra_p_predio_cal"].iloc[0])
    assert set(json.loads(open(paths["json"], encoding="utf-8").read())["groups"]) == {"estado"}


# --------------------------------------------------------------------------- outputs
def test_write_audit_creates_json_and_parquet_without_the_key(tmp_path):
    rows = [row(1, estado_nomenclatura="cert", zona="URB"), row(2, estado_nomenclatura="pcer", zona="EXP")]
    result = jc.audit_rows(rows, FakeClient(default=0.95), source="nomenclatura")
    paths = jc.write_audit(result, str(tmp_path))
    assert os.path.basename(paths["json"]) == "jev_audit.json" and os.path.basename(paths["rows"]) == "jev_audit_rows.parquet"
    payload = json.loads(open(paths["json"], encoding="utf-8").read())
    assert payload["source"] == "nomenclatura" and payload["agreement"]["n"] == 2
    assert set(payload["groups"]) >= {"estado_nomenclatura", "zona"}
    frame = pd.read_parquet(paths["rows"])
    assert len(frame) == 2 and {"id", "origin", "input_text", "output_text", "label", "jev_p", "status"} <= set(frame.columns)
    assert SENTINEL_KEY not in open(paths["json"], encoding="utf-8").read()


def test_write_audit_json_keeps_no_address_text(tmp_path):
    rows = [row(1, a="CL 900 # 12 - 34", b="CALLE 900 12 35")]
    result = jc.audit_rows(rows, FakeClient(default=0.02))
    paths = jc.write_audit(result, str(tmp_path))
    text = open(paths["json"], encoding="utf-8").read()
    assert "CL 900" not in text and "CALLE 900" not in text
