"""Regression contract for the retired Stage26 synthetic terrain route."""

import json
import subprocess
from collections import Counter
from pathlib import Path


FROZEN_MANIFEST = Path("D:/xunce/out/route_retire_preflight/candidate-manifest.json")


def _retired_stage26_paths() -> tuple[str, ...]:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    paths = tuple(
        record["path"]
        for record in manifest["records"]
        if record["classification"] == "retire_candidate"
        and "stage26" in record["path"].lower()
    )
    assert len(paths) == 148
    assert Counter(Path(path).parts[0] for path in paths) == {
        "configs": 39,
        "scripts": 38,
        "tests": 37,
        "docs": 34,
    }
    return paths


def test_manifest_listed_stage26_paths_are_not_tracked_or_present() -> None:
    retired_paths = _retired_stage26_paths()
    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")
    )
    deleted = set(
        subprocess.check_output(["git", "ls-files", "--deleted", "-z"])
        .decode("utf-8")
        .split("\0")
    )

    assert [path for path in retired_paths if path in tracked - deleted] == []
    assert [path for path in retired_paths if Path(path).exists()] == []


def test_stage_registry_has_no_stage26_entry_or_missing_script() -> None:
    registry = json.loads(Path("configs/stage_registry.json").read_text(encoding="utf-8"))
    serialized = json.dumps(registry["stages"]).lower()

    assert "stage26" not in serialized
    assert [
        entry["script"]
        for entry in registry["stages"].values()
        if not Path(entry["script"]).is_file()
    ] == []


def test_retained_platform_entrypoints_do_not_reference_stage26_runners() -> None:
    retained_entrypoints = (
        "scripts/run_platform_smoke.py",
        "scripts/run_platform_validation_matrix.py",
        "scripts/bootstrap_env.py",
        "scripts/bootstrap_ubuntu_conda.sh",
        "scripts/bootstrap_windows_conda.ps1",
        ".github/workflows/platform-compatibility.yml",
    )

    offenders = {
        path: Path(path).read_text(encoding="utf-8")
        for path in retained_entrypoints
        if "run_xunce_stage26" in Path(path).read_text(encoding="utf-8")
        or "xunce_stage26" in Path(path).read_text(encoding="utf-8")
    }
    assert offenders == {}
