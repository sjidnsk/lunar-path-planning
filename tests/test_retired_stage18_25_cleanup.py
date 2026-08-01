"""Regression contract for the retired Xunce Stage18--25 chain."""

import json
import re
import subprocess
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETIREMENT_FIXTURE = REPO_ROOT / "tests/fixtures/route_retirement_candidates_v1.json"
SOURCE_MANIFEST_SHA256 = "da9f23c3af83c3506143338a41e1c1a8f8f09dc4f59d4a4bac5adea160f6371c"
STAGE18_25_KEY = re.compile(r"xunce-stage(?:18i|18|19|2[0-5])(?:-|$)")


def _retired_stage18_25_paths() -> tuple[str, ...]:
    fixture = json.loads(RETIREMENT_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    assert {
        name: group["count"] for name, group in fixture["groups"].items()
    } == {
        "path_feedback": 29,
        "stage26": 148,
        "stage18_25": 250,
    }

    group = fixture["groups"]["stage18_25"]
    paths = tuple(group["paths"])
    assert group["count"] == len(paths) == 250
    assert Counter(Path(path).parts[0] for path in paths) == {
        "configs": 64,
        "scripts": 59,
        "tests": 59,
        "docs": 68,
    }
    return paths


def test_manifest_listed_stage18_25_paths_are_not_tracked_or_present() -> None:
    retired_paths = _retired_stage18_25_paths()
    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT)
        .decode("utf-8")
        .split("\0")
    )
    deleted = set(
        subprocess.check_output(["git", "ls-files", "--deleted", "-z"], cwd=REPO_ROOT)
        .decode("utf-8")
        .split("\0")
    )

    assert [path for path in retired_paths if path in tracked - deleted] == []
    assert [path for path in retired_paths if (REPO_ROOT / path).exists()] == []


def test_stage_registry_has_no_stage18_25_entries_or_missing_scripts() -> None:
    registry = json.loads(
        (REPO_ROOT / "configs/stage_registry.json").read_text(encoding="utf-8")
    )
    stages = registry["stages"]

    assert [key for key in stages if STAGE18_25_KEY.search(key)] == []
    assert "xunce-stage18-research-evidence-pipeline" not in stages
    assert [
        entry["script"]
        for entry in stages.values()
        if not (REPO_ROOT / entry["script"]).is_file()
    ] == []


def test_retained_platform_entrypoints_do_not_reference_retired_stage_runners() -> None:
    retained_entrypoints = (
        "scripts/run_platform_smoke.py",
        "scripts/run_platform_validation_matrix.py",
        "scripts/bootstrap_env.py",
        "scripts/bootstrap_ubuntu_conda.sh",
        "scripts/bootstrap_windows_conda.ps1",
        ".github/workflows/platform-compatibility.yml",
        "tests/test_platform_stage_runner.py",
    )
    retired_markers = (
        "run_xunce_stage18",
        "run_xunce_stage19",
        "run_xunce_stage20",
        "run_xunce_stage21",
        "run_xunce_stage22",
        "run_xunce_stage23",
        "run_xunce_stage24",
        "run_xunce_stage25",
        "xunce-stage18",
        "xunce-stage19",
        "xunce-stage20",
        "xunce-stage21",
        "xunce-stage22",
        "xunce-stage23",
        "xunce-stage24",
        "xunce-stage25",
    )

    offenders = {
        path: marker
        for path in retained_entrypoints
        for marker in retired_markers
        if marker in (REPO_ROOT / path).read_text(encoding="utf-8")
    }
    assert offenders == {}
