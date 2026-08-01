"""Regression contract for the retired path-feedback route cleanup."""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETIREMENT_FIXTURE = REPO_ROOT / "tests/fixtures/route_retirement_candidates_v1.json"
SOURCE_MANIFEST_SHA256 = "da9f23c3af83c3506143338a41e1c1a8f8f09dc4f59d4a4bac5adea160f6371c"


def _retired_path_feedback_paths() -> tuple[str, ...]:
    fixture = json.loads(RETIREMENT_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    group = fixture["groups"]["path_feedback"]
    paths = tuple(group["paths"])
    assert group["count"] == len(paths) == 29
    return paths

RETIRED_RUNTIME_MODULES = (
    "run_batch_path_feedback_validation",
    "run_path_feedback_stability_analysis",
    "run_path_feedback_validation",
    "run_quasi_real_map_path_feedback_bridge",
)

def test_manifest_listed_path_feedback_paths_are_absent() -> None:
    paths = _retired_path_feedback_paths()
    assert [path for path in paths if Path(path).exists()] == []


def test_stage_registry_and_platform_surface_do_not_expose_path_feedback() -> None:
    registry = json.loads(Path("configs/stage_registry.json").read_text(encoding="utf-8"))
    assert not any("path-feedback" in stage for stage in registry["stages"])
    assert "run_path_feedback_" not in Path("scripts/run_platform_validation_matrix.py").read_text(
        encoding="utf-8"
    )
    assert "path_feedback" not in Path(".github/workflows/platform-compatibility.yml").read_text(
        encoding="utf-8"
    )


def test_registry_scripts_exist_after_retired_entries_are_removed() -> None:
    registry = json.loads(Path("configs/stage_registry.json").read_text(encoding="utf-8"))
    missing = [
        entry["script"]
        for entry in registry["stages"].values()
        if not Path(entry["script"]).is_file()
    ]
    assert missing == []


def test_retained_executables_do_not_import_or_call_retired_runtime_modules() -> None:
    registry = json.loads(Path("configs/stage_registry.json").read_text(encoding="utf-8"))
    retained_executables = tuple(entry["script"] for entry in registry["stages"].values())
    offenders = {
        path: [module for module in RETIRED_RUNTIME_MODULES if module in Path(path).read_text(encoding="utf-8")]
        for path in retained_executables
    }
    assert offenders == {path: [] for path in retained_executables}
