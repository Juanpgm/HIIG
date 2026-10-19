"""Tests for scripts/check_publishable.py (stdlib only, synthetic fixtures in temp dirs).

Run with ``python -m unittest discover -s tests`` from the repository root.
Every secret-looking value below is synthetic. Lines that must not trip the
guard on the real repository carry the ``publish-guard: allow`` marker.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_publishable.py"
_spec = importlib.util.spec_from_file_location("check_publishable", SCRIPT)
cp = importlib.util.module_from_spec(_spec)
sys.modules["check_publishable"] = cp
_spec.loader.exec_module(cp)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


class RepoCase(unittest.TestCase):
    """Builds a throwaway git repository per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        _git(self.root, "init", "-q")

    def put(self, rel: str, content: bytes | str = b"x\n", *, stage: bool = True, size: int | None = None) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if size is not None:
            with p.open("wb") as fh:  # sparse-ish file of the requested size
                fh.seek(size - 1)
                fh.write(b"\0")
        else:
            p.write_bytes(content.encode() if isinstance(content, str) else content)
        if stage:
            _git(self.root, "add", "-f", "--", rel)
        return p

    def rules(self) -> set[str]:
        out: set[str] = set()
        for line in cp.run(self.root):
            out.add(line.rsplit(": ", 1)[1])
        return out

    def report(self) -> list[str]:
        return cp.run(self.root)


class CleanTree(RepoCase):
    def test_clean_tree_passes(self) -> None:
        self.put("pkg/src/mod.py", "VALUE = 1\n")
        self.put("README.md", "# hello\n")
        self.assertEqual(self.report(), [])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cp.main(["--root", str(self.root)]), 0)

    def test_empty_repository_passes(self) -> None:
        self.assertEqual(self.report(), [])

    def test_allowlisted_artifacts_pass(self) -> None:
        for rel in (
            "normalizador_direcciones/artifacts/model.pt",
            "normalizador_direcciones/artifacts/tuning.json",
            "normalizador_direcciones/artifacts/reliability.json",
            "normalizador_direcciones/artifacts/reliability_v2.json",
            "normalizador_direcciones/artifacts/catastro_meta.json",
            "normalizador_direcciones/artifacts/catastro_docs.parquet",
            "normalizador_direcciones/artifacts/catastro_own_centroids.parquet",
            "normalizador_direcciones/basemaps/barrios.geojson",
            "emparejador_direcciones/data/MANIFEST.json",
        ):
            self.put(rel, b"{}\n" if rel.endswith((".json", ".geojson")) else b"\x00\x01")
        self.assertEqual(self.report(), [])


class BlockedPaths(RepoCase):
    def test_blocked_directories(self) -> None:
        for d in ("context", "outputs", "runs", "notebooks", "experiments", "splits",
                  "candidatos", "salida_jev_2026", "x.egg-info", ".venv", "repo_audit"):
            with self.subTest(d=d):
                self.assertIn("BLOCKED_DIR", cp.check_path(f"sub/{d}/file.txt"))

    def test_blocked_directory_is_case_insensitive(self) -> None:
        self.assertIn("BLOCKED_DIR", cp.check_path("sub/Outputs/a.txt"))

    def test_similar_names_are_not_blocked(self) -> None:
        for rel in ("docs/experiments.md", "docs/splits.md", "scripts/run_experiment.py",
                    "src/pkg/outputs_helper.py", "tests/test_runs_helper.py"):
            with self.subTest(rel=rel):
                self.assertEqual(cp.check_path(rel), [])

    def test_blocked_names(self) -> None:
        for rel in (
            "a/artifacts/strict_eval_paso8_summary.csv",
            "a/artifacts/eval_calibration.csv",
            "a/experiments_leaderboard.csv",
            "a/idesc_cache.json",
            "a/idesc_results.parquet",
            "a/my_predictions.parquet",
            "a/train_data.npz",
            "a/spatial_negatives.npy",
            "a/catastro_emb.pt",
            "a/eval_samples.parquet",
            "a/x_motivos.csv",
            "a/x_niveles.csv",
            "a/x_summary.csv",
            "a/summary.json",
            "a/results.jsonl",
            "a/registros_2026.csv",
            "a/maestra_predios.xlsx",
            "a/tabla_indices_x.csv",
            "a/interp_index_v1.pkl",
            "a/interp_index_v1.json",
            "a/catalog_keys_v1.pkl",
            "a/direcciones.ipynb",
            "a/train.log",
            "a/.coverage",
            "a/random_sample.csv",
        ):
            with self.subTest(rel=rel):
                self.assertTrue(
                    any(r.startswith("BLOCKED_NAME") for r in cp.check_path(rel)), rel
                )

    def test_source_files_named_like_results_are_allowed(self) -> None:
        for rel in ("scripts/eval_strict.py", "src/pkg/eval_data.py", "src/pkg/idesc.py",
                    "tests/test_predictions.py", "docs/sample_usage.md", "src/pkg/maestro.py"):
            with self.subTest(rel=rel):
                self.assertEqual(cp.check_path(rel), [])

    def test_blocked_file_is_reported_with_its_path(self) -> None:
        self.put("sub/outputs/run1.txt", "ok\n")
        self.assertEqual(self.report(), ["sub/outputs/run1.txt: BLOCKED_DIR"])


class SecretFiles(RepoCase):
    def test_env_files(self) -> None:
        for rel in (".env", ".env.local", "sub/.env.production", "sub/prod.env"):
            with self.subTest(rel=rel):
                self.assertIn("ENV_FILE", cp.check_path(rel))

    def test_env_example_is_allowed(self) -> None:
        self.assertEqual(cp.check_path(".env.example"), [])

    def test_key_files(self) -> None:
        for rel in ("a/server.pem", "a/private.key", "a/id_rsa", "a/credentials.json",
                    "a/service-account-prod.json", "a/bundle.p12"):
            with self.subTest(rel=rel):
                self.assertIn("KEY_FILE", cp.check_path(rel))


class DataAllowlist(RepoCase):
    def test_data_outside_allowlist_fails(self) -> None:
        for rel in ("a/data.parquet", "a/table.csv", "a/cfg.json", "a/book.xlsx",
                    "a/weights.pt", "a/rows.jsonl", "a/x.geojson", "a/artifacts/other.parquet",
                    "a/artifacts/experiments/baseline/tuning.json"):
            with self.subTest(rel=rel):
                self.assertIn("DATA_NOT_ALLOWED", cp.check_path(rel))

    def test_allowlist_is_tied_to_the_subproject_path(self) -> None:
        for rel in ("other_project/artifacts/model.pt", "artifacts/model.pt",
                    "normalizador_direcciones/deploy/artifacts/model.pt",
                    "normalizador_direcciones/artifacts/sub/model.pt"):
            with self.subTest(rel=rel):
                self.assertTrue({"DATA_NOT_ALLOWED"} <= set(cp.check_path(rel)), rel)

    def test_curated_serving_copy_is_blocked(self) -> None:
        for rel in ("normalizador_direcciones/deploy/artifacts/model.pt",
                    "deploy/artifacts/catastro_docs.parquet", "x/Deploy/Artifacts/tuning.json"):
            with self.subTest(rel=rel):
                self.assertIn("BLOCKED_DIR", cp.check_path(rel))
        self.assertEqual(cp.check_path("normalizador_direcciones/deploy/prepare_artifacts.py"), [])

    def test_allowlist_is_not_a_prefix_match(self) -> None:
        self.assertIn("DATA_NOT_ALLOWED", cp.check_path("normalizador_direcciones/artifacts/model.pt.bak.pt"))
        self.assertIn("DATA_NOT_ALLOWED", cp.check_path("normalizador_direcciones/artifacts/tuning.json.csv"))

    def test_text_sources_are_not_data(self) -> None:
        for rel in ("a/mod.py", "a/README.md", "a/pyproject.toml", "a/Dockerfile", "a/uv.lock"):
            self.assertEqual(cp.check_path(rel), [])


class Sizes(RepoCase):
    def test_boundaries(self) -> None:
        self.put("a/ok.bin", size=cp.SIZE_LIMIT)  # exactly 50 MB passes
        self.assertEqual(self.report(), [])
        self.put("a/big.bin", size=cp.SIZE_LIMIT + 1)
        self.assertEqual(self.rules(), {"SIZE_OVER_50MB"})

    def test_hard_limit(self) -> None:
        self.put("a/huge.bin", size=cp.SIZE_HARD_LIMIT + 1)
        self.assertEqual(self.rules(), {"SIZE_OVER_90MB"})


class Secrets(RepoCase):
    def test_secret_assignment_with_real_looking_value(self) -> None:
        self.put("a/cfg.py", 'API_KEY = "' + "Zk3" + "q9Lw8Xv2" + "Rt7Pm4Nd6Bc" + '"\n')
        self.assertEqual(self.rules(), {"SECRET:secret-assignment"})

    def test_placeholders_are_ignored(self) -> None:
        self.put("a/cfg.py", 'token = "your-token-goes-here-123"\npassword = "<password-from-vault>"\n'
                             'api_key = "${API_KEY_FROM_ENV_VAR}"\nsecret = "changeme-changeme-0"\n')
        self.assertEqual(self.report(), [])

    def test_short_values_are_not_secrets(self) -> None:
        self.put("a/t.py", 'SECRET = "s3cretpw"\n')
        self.assertEqual(self.report(), [])

    def test_env_var_names_are_not_secrets(self) -> None:
        self.put("a/c.py", 'TOKEN_ENV = "SERVICE_API_TOKEN"\nkey = os.environ["API_KEY"]\n')
        self.assertEqual(self.report(), [])

    def test_private_key_header(self) -> None:
        self.put("a/k.txt", "-----BEGIN RSA " + "PRIVATE KEY-----\nabc\n")
        self.assertEqual(self.rules(), {"SECRET:private-key-header"})

    def test_bearer_token(self) -> None:
        self.put("a/h.txt", "Authorization: Bearer " + "abcDEF0123456789" + "abcDEF0123456789\n")
        self.assertEqual(self.rules(), {"SECRET:bearer-token"})

    def test_bearer_placeholder(self) -> None:
        self.put("a/h.txt", "Authorization: Bearer <your-token-here-please>\n")
        self.assertEqual(self.report(), [])

    def test_connection_string_with_credentials(self) -> None:
        self.put("a/db.txt", "postgresql://svc_app:" + "Hq8vN2" + "zT5mR@db.internal.corp/prod\n")
        self.assertEqual(self.rules(), {"SECRET:connection-string-credentials"})

    def test_connection_string_placeholders(self) -> None:
        self.put("a/db.txt", "postgresql://u:p@host/db\npostgresql://user:pw@host/db\n"
                             "postgresql://bob:Hq8vN2zT5@example.invalid/x\n")
        self.assertEqual(self.report(), [])

    def test_connection_string_test_fixtures_are_not_secrets(self) -> None:
        self.put("a/t.py", 'url = f"postgresql://cali:{pw}@127.0.0.1:1/cali"\n'
                           'u2 = "postgresql://user:s3cretpw@db.internal.corp/db"\n'
                           'k = {"with-secret": "foo://user:s3cretpw@host/db"}\n')
        self.assertEqual(self.report(), [])

    def test_loopback_host_is_not_a_secret_even_with_long_password(self) -> None:
        self.put("a/t.py", "postgresql://svc_app:" + "Hq8vN2" + "zT5mRq@127.0.0.1/db\n")
        self.assertEqual(self.report(), [])

    def test_cloud_tokens(self) -> None:
        self.put("a/c.txt", "AKIA" + "ABCDEFGHIJKLMNOP" + "\n" + "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8" + "\n")
        self.assertEqual(self.rules(), {"SECRET:aws-access-key", "SECRET:github-token"})

    def test_allow_marker_skips_the_line(self) -> None:
        self.put("a/t.py", 'API_KEY = "Zk3q9Lw8Xv2Rt7Pm4Nd6Bc"  # publish-guard: allow\n')
        self.assertEqual(self.report(), [])

    def test_report_has_line_number_and_pattern_name_only(self) -> None:
        value = "Zk3q9Lw8Xv2Rt7Pm4Nd6Bc"
        self.put("a/cfg.py", f'x = 1\nAPI_KEY = "{value}"\n')
        report = self.report()
        self.assertEqual(report, ["a/cfg.py:2: SECRET:secret-assignment"])
        self.assertNotIn(value, "\n".join(report))

    def test_binary_files_are_not_text_scanned(self) -> None:
        self.put("normalizador_direcciones/artifacts/model.pt", b"\x00\x01-----BEGIN " + b"PRIVATE KEY-----\n")
        self.assertEqual(self.report(), [])

    def test_malformed_utf8_does_not_crash(self) -> None:
        self.put("a/odd.txt", b"caf\xe9 \xff\xfe token ok\n")
        self.assertEqual(self.report(), [])

    def test_very_long_line_does_not_crash(self) -> None:
        self.put("a/long.txt", "a" * 500_000 + "\n")
        self.assertEqual(self.report(), [])


class GitStates(RepoCase):
    def test_staged_only_file_is_checked(self) -> None:
        self.put("a/.env", "X=1\n")  # staged, never committed
        self.assertEqual(self.rules(), {"ENV_FILE"})

    def test_committed_file_is_checked(self) -> None:
        self.put("a/outputs/x.txt", "x\n")
        _git(self.root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "x")
        self.assertEqual(self.rules(), {"BLOCKED_DIR"})

    def test_untracked_ignored_file_is_not_checked(self) -> None:
        self.put("a/.env", "X=1\n", stage=False)
        self.assertEqual(self.report(), [])

    def test_file_deleted_from_disk_does_not_crash(self) -> None:
        p = self.put("a/mod.py", "x = 1\n")
        _git(self.root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "x")
        p.unlink()
        self.assertEqual(self.report(), [])

    def test_not_a_git_repository_returns_error_code(self) -> None:
        with tempfile.TemporaryDirectory() as d, redirect_stderr(io.StringIO()):
            self.assertEqual(cp.main(["--root", d]), 2)

    def test_main_returns_one_and_caps_output(self) -> None:
        for i in range(5):
            self.put(f"a/outputs/f{i}.txt", "x\n")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cp.main(["--root", str(self.root), "--max-lines", "2"])
        self.assertEqual(code, 1)
        self.assertIn("3 more", out.getvalue())
        self.assertIn("FAIL (5 violation(s))", err.getvalue())

    def test_file_with_spaces_and_unicode_name(self) -> None:
        self.put("a/outputs/dirección con espacios.txt", "x\n")
        self.assertEqual(self.report(), ["a/outputs/dirección con espacios.txt: BLOCKED_DIR"])


class ResultReportNames(RepoCase):
    NAMES = (
        "eval_catastro_after.md", "eval_results.md", "run_results.md", "model_results_2026.txt",
        "resultados.md", "resultados_finales.md", "08-metricas-y-confiabilidad.md",
        "metricas.md", "leaderboard.md", "LEADERBOARD.md", "benchmark.md", "speed_benchmark_v2.txt",
        "Results.md", "results.txt",
    )

    def test_result_report_names_are_blocked(self) -> None:
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertIn("RESULT_REPORT", cp.check_path(f"docs/{name}"))

    def test_other_extensions_are_not_judged_by_this_rule(self) -> None:
        for rel in ("scripts/eval_strict.py", "src/pkg/eval_data.py", "tools/eval_catastro.py",
                    "src/pkg/metrics.py", "tests/test_benchmark_helpers.py"):
            with self.subTest(rel=rel):
                self.assertNotIn("RESULT_REPORT", cp.check_path(rel))

    def test_ordinary_documents_pass(self) -> None:
        for rel in ("README.md", "docs/groundtruth.md", "docs/splits.md", "docs/experiments.md",
                    "documentacion/04-evaluacion-y-calibracion.md", "docs/model-artifacts.md",
                    "CHANGELOG.md", "docs/evaluation_protocol.md", "docs/medical_notes.md"):
            with self.subTest(rel=rel):
                self.assertNotIn("RESULT_REPORT", cp.check_path(rel))

    def test_match_is_on_the_file_name_not_on_directories(self) -> None:
        self.assertNotIn("RESULT_REPORT", cp.check_path("benchmark/README.md"))
        self.assertIn("RESULT_REPORT", cp.check_path("benchmark/benchmark_notes.md"))

    def test_case_insensitive_and_unicode_names(self) -> None:
        self.assertIn("RESULT_REPORT", cp.check_path("docs/Resultados_Finales.MD"))
        self.assertIn("RESULT_REPORT", cp.check_path("docs/métricas-resultados/eval_ñandú.md"))

    def test_allowlist_is_exact_path_and_documented(self) -> None:
        allowed = cp.RESULT_REPORT_ALLOWLIST
        self.assertTrue(all(a.endswith("08-metricas-y-confiabilidad.md") for a in allowed))
        for rel in allowed:
            with self.subTest(rel=rel):
                self.assertNotIn("RESULT_REPORT", cp.check_path(rel))
        # same file name under another path is still blocked
        self.assertIn("RESULT_REPORT", cp.check_path("other/documentacion/08-metricas-y-confiabilidad.md"))

    def test_allowlist_can_be_extended_explicitly(self) -> None:
        rel = "docs/eval_protocol.md"
        self.assertIn("RESULT_REPORT", cp.check_path(rel))
        original = cp.RESULT_REPORT_ALLOWLIST
        cp.RESULT_REPORT_ALLOWLIST = original + (rel,)
        try:
            self.assertNotIn("RESULT_REPORT", cp.check_path(rel))
        finally:
            cp.RESULT_REPORT_ALLOWLIST = original

    def test_tracked_report_is_reported_with_its_path(self) -> None:
        self.put("docs/eval_after.md", "# anything\n")
        self.assertEqual(self.report(), ["docs/eval_after.md: RESULT_REPORT"])

    def test_staged_report_is_blocked_and_clean_doc_passes(self) -> None:
        self.put("docs/guide.md", "# guide\n")
        self.put("docs/benchmark_run.txt", "x\n")
        self.assertEqual(self.rules(), {"RESULT_REPORT"})

    def test_empty_file_and_binary_content_do_not_crash(self) -> None:
        self.put("docs/resultados.md", b"")
        self.put("docs/benchmark.txt", b"\x00\x01\x02")
        self.assertEqual(self.rules(), {"RESULT_REPORT"})


if __name__ == "__main__":
    unittest.main()
