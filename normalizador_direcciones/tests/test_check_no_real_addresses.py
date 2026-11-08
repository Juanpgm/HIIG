"""Privacy guard (scripts/check_no_real_addresses.py).

Two layers:

* unit tests on SYNTHETIC data only (always run): normalization, core keys,
  the public-cadastre exemption, every detector kind, GPS pairing, file
  selection, exit codes, and that findings never echo the matched value;
* the real guard: loads the private survey/evaluation data that may exist on a
  developer machine (in memory only) and fails if any real address, name, id or
  GPS point appears in a tracked file. It SKIPS when that data is absent.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_no_real_addresses", ROOT / "scripts" / "check_no_real_addresses.py"
)
guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = guard  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(guard)


def _real(addresses=(), names=(), ids=(), object_ids=(), points=(), cadastre=(), source="stickers"):
    rd = guard.RealData()
    guard._add_addresses(rd, list(addresses), source)
    guard._add_person_names(rd, list(names), source)
    guard._add_ids(rd, list(ids), source)
    rd.object_ids.update(object_ids)
    if points:
        lats, lons = zip(*points)
        guard._add_points(rd, lats, lons, source)
    for c in cadastre:
        rd.cadastre_core.update(guard.address_keys(c)[1])
    return rd


def _kinds(line, rd):
    return sorted({kind for kind, _ in guard.scan_line(line, rd)})


# ---------------------------------------------------------------------------
# Normalization and core keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Cra. 77 # 7-77", "cra 77 7 77"),
        ("CALLE  7ª  #  7–77", "calle 7a 7 77"),
        ("Cra77#7-77", "cra 77 7 77"),
        ("Avenida Óeste", "avenida oeste"),
        ("", ""),
        ("   ", ""),
        ("###---", ""),
    ],
)
def test_norm(text, expected):
    assert guard.norm(text) == expected


@pytest.mark.parametrize(
    "text, cores",
    [
        ("Calle 7 # 7-77", ["cl|7|7|77"]),
        ("CL 7 7 77 apto 301", ["cl|7|7|77"]),  # complement ignored
        ("Carrera 77 A bis # 7 B - 77", ["kr|77abis|7b|77"]),
        ("Cra77A#7-77", ["kr|77a|7|77"]),  # glued
        ("CL 7 Norte No. 7 - 77", ["cl|7|7|77"]),  # quadrant and filler dropped
        ("Avenida Calle 7 # 7 - 77", ["ac|7|7|77"]),
        ("Avenida Carrera 7 # 7 - 77", ["ak|7|7|77"]),
        ("TV 7 # 7 - 77; DG 8 # 8 - 88", ["tv|7|7|77", "dg|8|8|88"]),  # two addresses
        ("CL 7 # 7", []),  # no plate: not specific enough
        ("calle sin numero", []),
        ("", []),
    ],
)
def test_core_keys(text, cores):
    assert guard.address_keys(text)[1] == cores


def test_core_keys_ignore_runs_broken_by_unrelated_words():
    assert guard.address_keys("calle 7 casa 7 77")[1] == []


# ---------------------------------------------------------------------------
# Detection kinds (synthetic "real" data)
# ---------------------------------------------------------------------------


def test_full_and_core_address_hits():
    rd = _real(addresses=["Cra 77 # 7-77 apto 7"])
    assert _kinds('x = "CRA 77 # 7 - 77 APTO 7"', rd) == ["addr_core", "addr_full"]
    assert _kinds('x = "Carrera 77 No 7 77"', rd) == ["addr_core"]  # same core, other spelling
    assert _kinds('x = "Carrera 77 # 7 - 78"', rd) == []  # different plate
    assert _kinds('x = "Carrera 77 # 7"', rd) == []


def test_rural_descriptive_address_matches_verbatim_without_three_numbers():
    rd = _real(addresses=["Corregimiento X, vereda Y Z, casa 7"])
    assert _kinds("corregimiento x vereda y z casa 7", rd) == ["addr_full"]
    assert _kinds("corregimiento x vereda y z casa 8", rd) == []


def test_short_or_numberless_addresses_are_not_loaded():
    rd = _real(addresses=["calle 7", "sin direccion conocida aqui mismo", None, float("nan")])
    assert rd.full == {} and rd.core == {}


def test_public_cadastre_exemption_only_for_low_context_sources():
    insured = _real(addresses=["CL 7 # 7 - 77"], cadastre=["CL 7 # 7 - 77"], source="fasecolda")
    assert _kinds("CL 7 # 7 - 77", insured) == []  # public cadastre string
    survey = _real(addresses=["CL 7 # 7 - 77"], cadastre=["CL 7 # 7 - 77"], source="stickers")
    assert _kinds("CL 7 # 7 - 77", survey) == ["addr_core", "addr_full"]  # also a survey input
    not_cadastre = _real(addresses=["CL 7 # 7 - 77"], source="fasecolda")
    assert _kinds("CL 7 # 7 - 77", not_cadastre) == ["addr_core", "addr_full"]  # private only


def test_public_exemption_requires_every_source_to_be_low_context():
    rd = _real(addresses=["CL 7 # 7 - 77"], cadastre=["CL 7 # 7 - 77"], source="fasecolda")
    guard._add_addresses(rd, ["CL 7 # 7 - 77"], "rud")
    assert _kinds("CL 7 # 7 - 77", rd) == ["addr_core", "addr_full"]


def test_person_names_need_three_tokens_and_skip_public_place_names():
    rd = _real(names=["Ana Maria Perez Gomez", "Juan Perez", "Santiago de Cali", "Ana 2 Perez"])
    assert _kinds("# reviewed by ana maria perez gomez", rd) == ["person_name"]
    assert _kinds("barrio juan perez", rd) == []  # two tokens: collides with place names
    assert _kinds("municipio de santiago de cali", rd) == []
    assert "ana 2 perez" not in rd.names


def test_record_ids_and_emails():
    rd = _real(ids=["ABC12345", "7654321", "1111111", "12345"], names=["someone@example.org"])
    assert _kinds('id = "abc12345"', rd) == ["record_id"]
    assert _kinds("doc 7654321", rd) == ["record_id"]
    assert _kinds("doc 1111111", rd) == []  # degenerate repeated digits are ignored
    assert _kinds("doc 12345", rd) == []  # too short to be identifying
    assert _kinds("mail someone@example.org", rd) == ["record_id"]


def test_object_ids_only_next_to_an_id_field():
    rd = _real(object_ids={"4321"})
    assert _kinds('{"ObjectID": 4321}', rd) == ["object_id"]
    assert _kinds("id_edan=4321", rd) == ["object_id"]
    assert _kinds("count 4321", rd) == []


def test_gps_points_pair_within_three_lines_only():
    rd = _real(points=[(3.123456, -76.654321)])
    near = ["lat = 3.123456", "", "lon = -76.654321"]
    assert guard.scan_points(near, rd) == [(1, "gps_point", "stickers")]
    far = ["lat = 3.123456", "", "", "", "", "lon = -76.654321"]
    assert guard.scan_points(far, rd) == []
    assert guard.scan_points(["3.123456 -76.654391"], rd) == []  # other point
    assert guard.scan_points(["lat 3.12345"], rd) == []  # too few decimals to be a GPS fix


def test_points_with_swapped_columns_are_normalized():
    rd = _real(points=[(-76.654321, 3.123456)])  # lon/lat stored in the wrong order
    assert guard.scan_points(["(3.123456, -76.654321)"], rd)


def test_very_long_lines_are_skipped(tmp_path):
    rd = _real(addresses=["CL 7 # 7 - 77"])
    root = tmp_path / "proj"
    root.mkdir()
    (root / "blob.md").write_text("CL 7 # 7 - 77 " + "x" * 30000 + "\n", encoding="utf-8")
    assert guard.scan([root], rd) == []


# ---------------------------------------------------------------------------
# File selection, reporting and exit codes
# ---------------------------------------------------------------------------


def test_walk_mode_excludes_data_dirs_and_binaries(tmp_path):
    root = tmp_path / "proj"
    for d in ("src", "data", "artifacts", "outputs", "context", ".venv", "pkg.egg-info"):
        (root / d).mkdir(parents=True)
        (root / d / "f.py").write_text("x\n", encoding="utf-8")
    (root / "src" / "table.parquet").write_bytes(b"\x00")
    files = [p.relative_to(root).as_posix() for p in guard.tracked_files(root)]
    assert files == ["src/f.py"]


def test_findings_report_location_but_never_the_value(tmp_path, capsys, monkeypatch):
    secret = "TV 77 # 7 B - 77 casa 7"
    rd = _real(addresses=[secret])
    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_x.py").write_text(f'ADDR = "{secret}"\n', encoding="utf-8")
    monkeypatch.setattr(guard, "load_real_data", lambda *_: rd)
    monkeypatch.setattr(guard, "private_sources_present", lambda *_: True)
    assert guard.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "proj/tests/test_x.py:1 addr_" in out
    assert secret.casefold() not in out.casefold()
    assert "77" not in out.replace("proj", "")


def test_clean_tree_exits_zero(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text('ADDR = "CL 8 # 8 - 88"\n', encoding="utf-8")
    monkeypatch.setattr(guard, "load_real_data", lambda *_: _real(addresses=["CL 7 # 7 - 77"]))
    monkeypatch.setattr(guard, "private_sources_present", lambda *_: True)
    assert guard.main(["--root", str(root)]) == 0


def test_no_private_data_exits_two_without_scanning(tmp_path, capsys):
    assert guard.private_sources_present(tmp_path) is False
    assert guard.main(["--root", str(tmp_path), "--data-root", str(tmp_path)]) == 2
    assert "nothing checked" in capsys.readouterr().out


def test_loader_on_an_empty_data_root_returns_empty(tmp_path):
    rd = guard.load_real_data(tmp_path)
    assert rd.empty() and rd.sources_loaded == []


# ---------------------------------------------------------------------------
# The real guard: skipped unless the private data is present locally
# ---------------------------------------------------------------------------

_PROJECTS = [p for p in guard.SIBLING_PROJECTS if (guard.MODELS_ROOT / p).is_dir()]


@pytest.fixture(scope="module")
def real_data():
    if os.environ.get("SKIP_PRIVACY_GUARD") or not guard.private_sources_present():
        pytest.skip("private survey/evaluation data not present on this machine")
    rd = guard.load_real_data()
    if rd.empty():
        pytest.skip("private data present but unreadable (missing pandas/openpyxl?)")
    return rd


@pytest.mark.parametrize("project", _PROJECTS)
def test_no_real_citizen_data_in_tracked_files(real_data, project):
    findings = guard.scan([guard.MODELS_ROOT / project], real_data)
    report = "\n".join(f"{path}:{line} {kind} [{src}]" for path, line, kind, src in findings)
    assert not findings, (
        "real citizen data found in tracked files (values intentionally not shown):\n" + report
    )
