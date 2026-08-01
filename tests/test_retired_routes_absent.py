"""Unified regression guard for all retired parent-repository routes."""

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests/fixtures/route_retirement_candidates_v1.json"
SOURCE_MANIFEST_SHA256 = "da9f23c3af83c3506143338a41e1c1a8f8f09dc4f59d4a4bac5adea160f6371c"


def _active_tracked_paths() -> set[str]:
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT).decode().split("\0"))
    deleted = set(subprocess.check_output(["git", "ls-files", "--deleted", "-z"], cwd=REPO_ROOT).decode().split("\0"))
    return tracked - deleted - {""}


def test_all_manifest_retirement_candidates_are_absent() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    groups = fixture["groups"]
    assert {name: group["count"] for name, group in groups.items()} == {"path_feedback": 29, "stage26": 148, "stage18_25": 250}
    retired = [path for group in groups.values() for path in group["paths"]]
    assert len(retired) == 427
    active = _active_tracked_paths()
    assert [path for path in retired if path in active or (REPO_ROOT / path).exists()] == []


def test_retired_gitlinks_are_not_in_the_parent_index() -> None:
    active = _active_tracked_paths()
    assert {"path-planner", "dev-platform-constraints"} <= active
    assert "model-explorer" not in active
    assert "visual-workbench" not in active
