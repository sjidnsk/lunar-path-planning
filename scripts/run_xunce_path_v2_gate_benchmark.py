from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import path_v2_gate_artifacts as gate_artifacts
import run_xunce_path_v2_g0_baseline_and_isolation as gate0
import xunce_artifact_io as artifact_io


SCHEMA_VERSION = "xunce-path-v2-gate1-contract/v1"
STAGE_ID = "xunce-path-v2-gate1-contract"
FORMAL_PYTHON = Path("D:/conda_envs/lunar-explorer/python.exe")
FORMAL_TEMP_ROOT = Path("D:/xunce/tmp/path_v2_g1")
EXPECTED_BRANCH = "codex/multiplatform-path-planner-v2"
ORIGINAL_BASE_COMMIT = "b635740ee021258ef31811ec87c60add839fc5f9"
GATE_INPUT_COMMIT = "b2a36d31f3802eb5a37fcfcf594f74a499aa719b"
EXPECTED_PYTHON_VERSION = "3.12.13"
LEGACY_EXPECTED = {
    "passed": 156,
    "skipped": 17,
    "failures": 0,
    "errors": 0,
}
ALLOWED_SKIP_DEPENDENCY = "pydrake"
PATH_PLANNER_WORKING_DIRECTORY = "path-planner"
PATH_PLANNER_PYTHONPATH = ["path-planner/src"]
PASS_ROUTE = "implement_path_v2_wheel_provider"
FOCUSED_TARGETS = (
    "tests/test_v2_contracts.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_terrain.py",
    "tests/test_v2_fine_safety_anchor.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_api.py",
)
CANONICAL_ARTIFACT_NAMES = frozenset(
    {
        "config.json",
        "summary.json",
        "routing.json",
        "results.jsonl",
        "phase-state.jsonl",
        "review.json",
        "report.md",
        "manifest.json",
    }
)

GATE2_SCHEMA_VERSION = "xunce-path-v2-gate2-wheel/v1"
GATE2_STAGE_ID = "xunce-path-v2-gate2-wheel"
GATE2_INPUT_COMMIT = "0cc3eb9728a77473dd436dc2b05e2df009a296c7"
GATE2_FORMAL_TEMP_ROOT = Path("D:/xunce/tmp/path_v2_g2")
GATE2_PASS_ROUTE = "implement_path_v2_lazy_validation_and_cache"
GATE2_EXECUTE_ROUTE = "execute_gate2_wheel_evidence"
GATE2_FOCUSED_TARGETS = (
    "tests/test_v2_benchmark.py",
    "tests/test_v2_wheel_provider.py",
    "tests/test_v2_route_validation.py",
    "tests/test_v2_geometry.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_wheel_contracts.py",
    "tests/test_v2_contracts.py",
    "tests/test_v2_api.py",
    "tests/test_v2_runtime.py",
    "tests/test_hybrid_astar.py",
    "tests/test_astar.py",
)
GATE2_INPUTS = {
    "primitive_audit": "D:/xunce/inputs/path_v2/g2/independent_wheel_oracle_labels.jsonl",
    "exact_map_quality": "D:/xunce/inputs/path_v2/g2/independent_wheel_exact_map_optima.jsonl",
    "standard_episodes": "D:/xunce/inputs/path_v2/g2/standard_wheel_schedule.jsonl",
}
GATE2_BASELINE_EVIDENCE = {
    "schema_version": "xunce-path-v2-gate0-path-planner-junit/v1",
    "path": "D:/xunce/out/path_v2/g0/path_planner_baseline.junit.xml",
    "sha256": "90a02eb6af78805bfdf2fcce4828283cb3f4d77f30aae55519f533896a06e676",
    "test_count": 173,
}
GATE2_THRESHOLDS = {
    "min_primitive_independent_samples": 10000,
    "max_primitive_false_positives": 0,
    "min_primitive_recall": 0.98,
    "min_primitive_complete_l2_ratio": 1.0,
    "min_exact_map_independent_cases": 1,
    "min_exact_map_success_ratio": 1.0,
    "min_exact_map_resource_cost_ratio": 1.0,
    "max_exact_map_resource_cost_ratio": 1.10,
    "min_exact_map_complete_l2_ratio": 1.0,
    "min_standard_independent_episodes": 100,
    "min_standard_reachable_success_ratio": 0.99,
    "min_standard_complete_l2_ratio": 1.0,
    "max_standard_p95_runtime_ms": 250.0,
    "hard_timeout_ms": 2000.0,
    "max_hard_timeout_violations": 0,
}
GATE2_BLOCKERS = (
    ("primitive_audit", "provide_independent_wheel_oracle_labels"),
    ("exact_map_quality", "provide_independent_wheel_exact_map_optima"),
    ("standard_episodes", "provide_standard_wheel_schedule"),
)
GATE2_ROW_API = {
    "primitive_audit": ("PrimitiveAuditRowV2", "aggregate_primitive_audit_v2"),
    "exact_map_quality": ("ExactMapQualityRowV2", "aggregate_exact_map_quality_v2"),
    "standard_episodes": ("StandardEpisodeRowV2", "aggregate_standard_episodes_v2"),
}
class _Gate2LoaderContractError(RuntimeError):
    pass


BYTE_PROBE_CODE = r"""
import hashlib
import json
import path_planner
import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.search import AStarPlanner
from path_planner.v2.api import plan_v2
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.profiles import PlatformProfileRegistryV2
from path_planner.v2.serialization import canonical_json_bytes

request = PlanningRequestV2(
    request_id="gate1-byte-probe",
    platform_profile_id="unknown-profile/v1",
    start_state=PoseStateV2(0.25, 0.25, 0.0),
    goal_state=PoseStateV2(0.75, 0.75, 0.0),
    terrain_snapshot=object(),
    objective_profile=ObjectiveProfileV2(),
    resource_budget=ResourceBudgetV2(),
    timeout_s=1.0,
    accelerator_policy=AcceleratorPolicyV2.DISABLED,
    determinism_seed=1,
)
outcome = plan_v2(
    request,
    registry=PlatformProfileRegistryV2(()),
    providers={},
)
canonical = canonical_json_bytes(outcome)
spec = GridSpec(width=2, height=2, resolution=0.5)
grid = CostGrid(
    spec=spec,
    cost=np.ones((2, 2)),
    passable_mask=np.ones((2, 2), dtype=bool),
)
legacy_schema = AStarPlanner().plan(
    grid,
    PlanRequest(start=Cell(0, 0), goal=Cell(1, 1)),
).to_route_dict(spec)["schema_version"]
print(json.dumps({
    "canonical_hex": canonical.hex(),
    "digest": hashlib.sha256(canonical).hexdigest(),
    "root_has_plan_v2": "plan_v2" in path_planner.__dict__,
    "v1_route_schema": legacy_schema,
}, sort_keys=True, separators=(",", ":")))
"""


def _case_outcome(testcase) -> str:
    if testcase.find("failure") is not None:
        return "failures"
    if testcase.find("error") is not None:
        return "errors"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def _empty_counts() -> dict[str, int]:
    return {"passed": 0, "skipped": 0, "failures": 0, "errors": 0}


def _is_v2_nodeid(nodeid: str) -> bool:
    module_path = nodeid.split("::", 1)[0].replace("\\", "/")
    return any(part.startswith("test_v2_") for part in module_path.split("/"))


def audit_full_junit(
    path: Path,
    *,
    expected_legacy: dict[str, Any],
    allowed_skip_dependency: str,
) -> dict[str, Any]:
    total = _empty_counts()
    legacy = _empty_counts()
    v2 = _empty_counts()
    legacy_skip_messages: list[str] = []
    for testcase in gate0._junit_testcases(Path(path)):
        nodeid = gate0._testcase_nodeid(testcase)
        outcome = _case_outcome(testcase)
        total[outcome] += 1
        bucket = v2 if _is_v2_nodeid(nodeid) else legacy
        bucket[outcome] += 1
        if bucket is legacy and outcome == "skipped":
            skipped = testcase.find("skipped")
            legacy_skip_messages.append(
                " ".join(
                    filter(
                        None,
                        (skipped.attrib.get("message", ""), skipped.text or ""),
                    )
                )
            )

    expected = {
        key: int(expected_legacy[key])
        for key in ("passed", "skipped", "failures", "errors")
    }
    dependency = str(allowed_skip_dependency).lower()
    skip_contract = (
        len(legacy_skip_messages) == legacy["skipped"]
        and all(dependency in message.lower() for message in legacy_skip_messages)
    )
    v2_green = (
        v2["failures"] == 0
        and v2["errors"] == 0
        and v2["skipped"] == 0
    )
    status = "passed" if legacy == expected and skip_contract and v2_green else "failed"
    return {
        "schema_version": "xunce-path-v2-gate1-full-junit-audit/v1",
        "status": status,
        "total": total,
        "legacy": legacy,
        "v2": v2,
        "legacy_skip_messages": legacy_skip_messages,
        "legacy_skip_contract": skip_contract,
    }


def _audit_focused_junit(path: Path) -> dict[str, Any]:
    summary = gate0.parse_junit(Path(path))
    status = (
        "passed"
        if summary.tests > 0
        and summary.failures == 0
        and summary.errors == 0
        and summary.skipped == 0
        else "failed"
    )
    return {
        "schema_version": "xunce-path-v2-gate1-focused-junit-audit/v1",
        "status": status,
        "tests": summary.tests,
        "passed": summary.passed,
        "skipped": summary.skipped,
        "failures": summary.failures,
        "errors": summary.errors,
    }


def audit_repeat_digests(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized = list(rows)
    digests = [row.get("digest") for row in normalized]
    seeds = [str(row.get("python_hash_seed")) for row in normalized]
    repeats = [row.get("repeat") for row in normalized]
    valid_digests = all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in digests
    )
    passed = (
        len(normalized) >= 3
        and repeats == list(range(1, len(normalized) + 1))
        and len(seeds) == len(set(seeds))
        and valid_digests
        and len(set(digests)) == 1
    )
    return {
        "schema_version": "xunce-path-v2-gate1-byte-repeat-audit/v1",
        "status": "passed" if passed else "failed",
        "repeat_count": len(normalized),
        "python_hash_seeds": seeds,
        "digests": digests,
        "stable_digest": digests[0] if passed else None,
    }


def boundaries_match(configured: Any) -> bool:
    return (
        isinstance(configured, dict)
        and set(configured) == set(gate_artifacts.BOUNDARY_FIELDS)
        and all(configured[field] is False for field in gate_artifacts.BOUNDARY_FIELDS)
    )


def validate_output_root(repo_root: Path, output_root: Path) -> Path:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == repo_root or output_root.is_relative_to(repo_root):
        raise ValueError("output_root must be outside repo")
    return output_root


def _assert_no_stale_artifacts(output_root: Path) -> None:
    root = Path(output_root).resolve()
    safe_root = artifact_io.windows_safe_path(root)
    if not os.path.exists(safe_root):
        return
    if not os.path.isdir(safe_root):
        raise RuntimeError("output_root exists but is not a directory")
    stale = sorted(
        entry.name
        for entry in os.scandir(safe_root)
        if entry.name not in CANONICAL_ARTIFACT_NAMES
        or not entry.is_file(follow_symlinks=False)
    )
    if stale:
        raise RuntimeError(f"stale noncanonical Gate 1 artifacts: {stale}")


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _audit_nested_git(repo_root: Path, expected_branch: str) -> dict[str, Any]:
    nested_root = repo_root / "path-planner"
    try:
        gitlink = _git(repo_root, "rev-parse", "HEAD:path-planner")
        head = _git(nested_root, "rev-parse", "HEAD")
        branch = _git(nested_root, "rev-parse", "--abbrev-ref", "HEAD")
        porcelain = _git(nested_root, "status", "--porcelain", "--untracked-files=all")
    except subprocess.CalledProcessError as exc:
        return {
            "schema_version": "xunce-path-v2-gate1-nested-git-audit/v1",
            "status": "failed",
            "error": (exc.stderr or type(exc).__name__).strip(),
        }
    checks = {
        "head_matches_gitlink": head == gitlink,
        "branch_matches": branch == expected_branch,
        "clean_tree": not porcelain,
    }
    return {
        "schema_version": "xunce-path-v2-gate1-nested-git-audit/v1",
        "status": "passed" if all(checks.values()) else "failed",
        "gitlink": gitlink,
        "head": head,
        "branch": branch,
        "expected_branch": expected_branch,
        "dirty_paths": porcelain.splitlines(),
        **checks,
    }


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_frozen(configured: Any, expected: Any, field: str) -> None:
    if not _strict_equal(configured, expected):
        raise ValueError(f"frozen {field} must equal {expected!r}")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if config.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID!r}")
    expected_git = config.get("expected_git", {})
    _require_frozen(expected_git.get("branch"), EXPECTED_BRANCH, "expected_git.branch")
    _require_frozen(
        expected_git.get("nested_branch"),
        EXPECTED_BRANCH,
        "expected_git.nested_branch",
    )
    _require_frozen(
        expected_git.get("base_commit"),
        ORIGINAL_BASE_COMMIT,
        "expected_git.base_commit",
    )
    _require_frozen(
        expected_git.get("gate_input_commit"),
        GATE_INPUT_COMMIT,
        "expected_git.gate_input_commit",
    )
    _require_frozen(config.get("python"), FORMAL_PYTHON.as_posix(), "python")
    _require_frozen(
        config.get("expected_python_version"),
        EXPECTED_PYTHON_VERSION,
        "expected_python_version",
    )
    _require_frozen(config.get("temp_root"), FORMAL_TEMP_ROOT.as_posix(), "temp_root")
    focused = config.get("focused", {})
    full = config.get("full", {})
    _require_frozen(
        focused.get("working_directory"),
        PATH_PLANNER_WORKING_DIRECTORY,
        "focused.working_directory",
    )
    _require_frozen(
        focused.get("pythonpath"),
        PATH_PLANNER_PYTHONPATH,
        "focused.pythonpath",
    )
    _require_frozen(
        focused.get("pytest_targets"),
        list(FOCUSED_TARGETS),
        "focused.pytest_targets",
    )
    _require_frozen(
        full.get("working_directory"),
        PATH_PLANNER_WORKING_DIRECTORY,
        "full.working_directory",
    )
    _require_frozen(
        full.get("pythonpath"),
        PATH_PLANNER_PYTHONPATH,
        "full.pythonpath",
    )
    _require_frozen(full.get("pytest_targets"), ["tests"], "full.pytest_targets")
    _require_frozen(full.get("legacy_expected"), LEGACY_EXPECTED, "full.legacy_expected")
    _require_frozen(
        full.get("allowed_skip_dependency"),
        ALLOWED_SKIP_DEPENDENCY,
        "full.allowed_skip_dependency",
    )
    byte_repeat = config.get("byte_repeat", {})
    repeat_count = byte_repeat.get("repeat_count")
    seeds = byte_repeat.get("python_hash_seeds")
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 3
        or not isinstance(seeds, list)
        or len(seeds) != repeat_count
        or len({str(seed) for seed in seeds}) != repeat_count
    ):
        raise ValueError("byte_repeat requires at least three distinct hash seeds")
    if config.get("pass_route") != PASS_ROUTE:
        raise ValueError(f"pass_route must be {PASS_ROUTE!r}")


def _validate_gate2_config(config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "stage_id",
        "python",
        "expected_python_version",
        "expected_git",
        "temp_root",
        "focused",
        "full",
        "inputs",
        "baseline_evidence",
        "thresholds",
        "boundaries",
        "pass_route",
    }
    _require_frozen(set(config), expected_keys, "top-level keys")
    _require_frozen(config.get("schema_version"), GATE2_SCHEMA_VERSION, "schema_version")
    _require_frozen(config.get("stage_id"), GATE2_STAGE_ID, "stage_id")
    _require_frozen(config.get("python"), FORMAL_PYTHON.as_posix(), "python")
    _require_frozen(
        config.get("expected_python_version"),
        EXPECTED_PYTHON_VERSION,
        "expected_python_version",
    )
    _require_frozen(
        config.get("expected_git"),
        {
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE2_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
        },
        "expected_git",
    )
    _require_frozen(
        config.get("temp_root"),
        GATE2_FORMAL_TEMP_ROOT.as_posix(),
        "temp_root",
    )
    _require_frozen(
        config.get("focused"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": list(GATE2_FOCUSED_TARGETS),
        },
        "focused",
    )
    _require_frozen(
        config.get("full"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": ["tests"],
            "legacy_expected": LEGACY_EXPECTED,
            "allowed_skip_dependency": ALLOWED_SKIP_DEPENDENCY,
        },
        "full",
    )
    _require_frozen(config.get("inputs"), GATE2_INPUTS, "inputs")
    _require_frozen(
        config.get("baseline_evidence"),
        GATE2_BASELINE_EVIDENCE,
        "baseline_evidence",
    )
    _require_frozen(config.get("thresholds"), GATE2_THRESHOLDS, "thresholds")
    _require_frozen(
        config.get("boundaries"),
        gate_artifacts.BOUNDARY_FIELDS,
        "boundaries",
    )
    _require_frozen(config.get("pass_route"), GATE2_PASS_ROUTE, "pass_route")


def _junit_records(
    content: bytes,
) -> tuple[list[ET.Element], dict[str, dict[str, str]], list[str]]:
    testcases = list(ET.fromstring(content).iter("testcase"))
    records: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for testcase in testcases:
        nodeid = gate0._testcase_nodeid(testcase)
        if nodeid in records:
            duplicates.append(nodeid)
            continue
        skipped = testcase.find("skipped")
        skip_message = ""
        if skipped is not None:
            skip_message = " ".join(
                filter(
                    None,
                    (skipped.attrib.get("message", ""), skipped.text or ""),
                )
            )
        records[nodeid] = {
            "outcome": _case_outcome(testcase),
            "skip_message": skip_message,
        }
    return testcases, records, sorted(set(duplicates))


def _counts_for_records(records: Sequence[dict[str, str]]) -> dict[str, int]:
    counts = _empty_counts()
    for record in records:
        counts[record["outcome"]] += 1
    return counts


def _audit_gate2_full_junit(
    path: Path,
    *,
    expected_legacy: dict[str, Any],
    allowed_skip_dependency: str,
    baseline_evidence: dict[str, Any],
) -> dict[str, Any]:
    baseline = {
        key: int(expected_legacy[key])
        for key in ("passed", "skipped", "failures", "errors")
    }
    baseline_path = Path(str(baseline_evidence["path"]))
    expected_hash = str(baseline_evidence["sha256"])
    expected_test_count = int(baseline_evidence["test_count"])
    evidence_review = {
        "schema_version": baseline_evidence.get("schema_version"),
        "path": baseline_path.as_posix(),
        "expected_sha256": expected_hash,
        "actual_sha256": None,
        "expected_test_count": expected_test_count,
        "actual_test_count": None,
        "status": "failed",
    }
    try:
        baseline_bytes = artifact_io.read_bytes(baseline_path)
        evidence_review["actual_sha256"] = hashlib.sha256(baseline_bytes).hexdigest()
        baseline_cases, baseline_records, baseline_duplicates = _junit_records(
            baseline_bytes
        )
        evidence_review["actual_test_count"] = len(baseline_cases)
        current_bytes = artifact_io.read_bytes(Path(path))
        current_cases, current_records, current_duplicates = _junit_records(
            current_bytes
        )
    except (ET.ParseError, KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "schema_version": "xunce-path-v2-gate2-full-junit-audit/v1",
            "status": "failed",
            "total": _empty_counts(),
            "baseline": baseline,
            "legacy": _empty_counts(),
            "v2": _empty_counts(),
            "baseline_evidence": evidence_review,
            "baseline_not_reduced": False,
            "skip_contract": False,
            "v2_green": False,
            "missing_baseline_nodeids": [],
            "drifted_baseline_nodeids": [],
            "duplicate_baseline_nodeids": [],
            "duplicate_current_nodeids": [],
            "error_type": type(exc).__name__,
        }

    baseline_source_counts = _counts_for_records(tuple(baseline_records.values()))
    baseline_evidence_valid = (
        evidence_review["actual_sha256"] == expected_hash
        and len(baseline_cases) == expected_test_count
        and not baseline_duplicates
        and baseline_source_counts == baseline
    )
    evidence_review["status"] = "passed" if baseline_evidence_valid else "failed"
    baseline_nodeids = set(baseline_records)
    current_nodeids = set(current_records)
    missing = sorted(baseline_nodeids - current_nodeids)
    drifted = sorted(
        nodeid
        for nodeid in baseline_nodeids & current_nodeids
        if baseline_records[nodeid]["outcome"] != current_records[nodeid]["outcome"]
    )
    legacy_records = [
        current_records[nodeid]
        for nodeid in sorted(baseline_nodeids & current_nodeids)
    ]
    added_records = [
        current_records[nodeid]
        for nodeid in sorted(current_nodeids - baseline_nodeids)
    ]
    legacy = _counts_for_records(legacy_records)
    v2 = _counts_for_records(added_records)
    total = _counts_for_records(
        tuple({"outcome": _case_outcome(testcase)} for testcase in current_cases)
    )
    skip_messages = [
        current_records[nodeid]["skip_message"]
        for nodeid in sorted(baseline_nodeids & current_nodeids)
        if current_records[nodeid]["outcome"] == "skipped"
    ]
    dependency = str(allowed_skip_dependency).lower()
    skip_contract = (
        len(skip_messages) == baseline["skipped"]
        and all(dependency in message.lower() for message in skip_messages)
    )
    baseline_not_reduced = (
        baseline_evidence_valid
        and not missing
        and not drifted
        and not current_duplicates
        and legacy == baseline
    )
    v2_green = (
        not current_duplicates
        and v2["failures"] == 0
        and v2["errors"] == 0
        and v2["skipped"] == 0
    )
    passed = baseline_not_reduced and skip_contract and v2_green
    return {
        "schema_version": "xunce-path-v2-gate2-full-junit-audit/v1",
        "status": "passed" if passed else "failed",
        "total": total,
        "baseline": baseline,
        "legacy": legacy,
        "v2": v2,
        "baseline_evidence": evidence_review,
        "baseline_not_reduced": baseline_not_reduced,
        "skip_contract": skip_contract,
        "v2_green": v2_green,
        "skip_messages": skip_messages,
        "missing_baseline_nodeids": missing,
        "drifted_baseline_nodeids": drifted,
        "duplicate_baseline_nodeids": baseline_duplicates,
        "duplicate_current_nodeids": current_duplicates,
    }


def _gate2_runtime_audit(repo_root: Path) -> dict[str, Any]:
    superproject = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        ORIGINAL_BASE_COMMIT,
    )
    gate_input = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        GATE2_INPUT_COMMIT,
    )
    nested = _audit_nested_git(repo_root, EXPECTED_BRANCH)
    imports = gate0.audit_import_origins(FORMAL_PYTHON, repo_root)
    python_version_matches = imports.get("python_version") == EXPECTED_PYTHON_VERSION
    original_base_is_ancestor = (
        superproject.get("status") == "passed"
        and superproject.get("base_is_ancestor") is True
    )
    gate_input_is_ancestor = (
        gate_input.get("status") == "passed"
        and gate_input.get("base_is_ancestor") is True
    )
    passed = (
        original_base_is_ancestor
        and gate_input_is_ancestor
        and nested.get("status") == "passed"
        and imports.get("status") == "passed"
        and python_version_matches
    )
    return {
        "schema_version": "xunce-path-v2-gate2-runtime-audit/v1",
        "status": "passed" if passed else "failed",
        "superproject_git": superproject,
        "gate_input_git": gate_input,
        "nested_git": nested,
        "import_origins": imports,
        "python_version_matches": python_version_matches,
        "original_base_is_ancestor": original_base_is_ancestor,
        "gate_input_is_ancestor": gate_input_is_ancestor,
    }


def _gate2_preflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate2_runtime_audit(repo_root)


def _gate2_postflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate2_runtime_audit(repo_root)


def _runtime_audit(repo_root: Path) -> dict[str, Any]:
    superproject = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        ORIGINAL_BASE_COMMIT,
    )
    gate_input = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        GATE_INPUT_COMMIT,
    )
    nested = _audit_nested_git(repo_root, EXPECTED_BRANCH)
    imports = gate0.audit_import_origins(FORMAL_PYTHON, repo_root)
    python_version_matches = imports.get("python_version") == EXPECTED_PYTHON_VERSION
    original_base_is_ancestor = (
        superproject.get("status") == "passed"
        and superproject.get("base_is_ancestor") is True
    )
    gate_input_is_ancestor = (
        gate_input.get("status") == "passed"
        and gate_input.get("base_is_ancestor") is True
    )
    passed = (
        original_base_is_ancestor
        and gate_input_is_ancestor
        and nested.get("status") == "passed"
        and imports.get("status") == "passed"
        and python_version_matches
    )
    return {
        "schema_version": "xunce-path-v2-gate1-preflight/v1",
        "status": "passed" if passed else "failed",
        "superproject_git": superproject,
        "gate_input_git": gate_input,
        "nested_git": nested,
        "import_origins": imports,
        "python_version_matches": python_version_matches,
        "original_base_is_ancestor": original_base_is_ancestor,
        "gate_input_is_ancestor": gate_input_is_ancestor,
    }


def _preflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _runtime_audit(repo_root)


def _postflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _runtime_audit(repo_root)


def _selected_identity(audit: dict[str, Any]) -> dict[str, Any]:
    superproject = audit.get("superproject_git", {})
    gate_input = audit.get("gate_input_git", {})
    nested = audit.get("nested_git", {})
    imports = audit.get("import_origins", {})
    return {
        "superproject": {
            key: superproject.get(key)
            for key in ("head", "branch", "git_dir", "git_common_dir")
        },
        "gate_input": {
            key: gate_input.get(key)
            for key in ("head", "branch", "git_dir", "git_common_dir")
        },
        "nested": {
            key: nested.get(key)
            for key in ("head", "branch", "gitlink")
        },
        "imports": {
            "python": imports.get("python"),
            "python_version": imports.get("python_version"),
            "python_no_user_site": imports.get("python_no_user_site"),
            "pythonpath": imports.get("pythonpath"),
            "path_planner_origin": imports.get("path_planner", {}).get("origin"),
            "ppo_origin": imports.get("lunar_exploration_ppo", {}).get("origin"),
            "path_planner_from_worktree": imports.get("path_planner_from_worktree"),
            "ppo_from_worktree": imports.get("ppo_from_worktree"),
        },
    }


def _postflight_matches(
    preflight: dict[str, Any],
    postflight: dict[str, Any],
) -> bool:
    superproject = postflight.get("superproject_git", {})
    gate_input = postflight.get("gate_input_git", {})
    nested = postflight.get("nested_git", {})
    imports = postflight.get("import_origins", {})
    return (
        postflight.get("status") == "passed"
        and superproject.get("status") == "passed"
        and superproject.get("clean_tree") is True
        and gate_input.get("status") == "passed"
        and gate_input.get("clean_tree") is True
        and nested.get("status") == "passed"
        and nested.get("clean_tree") is True
        and nested.get("head_matches_gitlink") is True
        and imports.get("status") == "passed"
        and imports.get("python_no_user_site") is True
        and postflight.get("python_version_matches") is True
        and postflight.get("original_base_is_ancestor") is True
        and postflight.get("gate_input_is_ancestor") is True
        and _selected_identity(preflight) == _selected_identity(postflight)
    )


def _common_env(repo_root: Path, attempt_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str((repo_root / "path-planner" / "src").resolve()),
            "TEMP": str(attempt_root),
            "TMP": str(attempt_root),
            "MPLCONFIGDIR": str(attempt_root / "mpl"),
        }
    )
    return env


def _run_pytest(
    *,
    python: Path,
    repo_root: Path,
    targets: Sequence[str],
    junit_path: Path,
    basetemp: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    artifact_io.write_text(
        junit_path,
        '<?xml version="1.0" encoding="utf-8"?><testsuite tests="0" />',
    )
    command = [str(python), "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    command.extend(str(target) for target in targets)
    command.extend(("--basetemp", str(basetemp), "--junitxml", str(junit_path)))
    completed = subprocess.run(
        command,
        cwd=repo_root / "path-planner",
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def _run_byte_repeats(
    *,
    python: Path,
    repo_root: Path,
    seeds: Sequence[Any],
    common_env: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat, raw_seed in enumerate(seeds, start=1):
        seed = str(raw_seed)
        env = common_env.copy()
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [str(python), "-c", BYTE_PROBE_CODE],
            cwd=repo_root / "path-planner",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        payload: dict[str, Any] = {}
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError, TypeError, ValueError):
                payload = {}
        rows.append(
            {
                "suite": "byte-repeat",
                "check": "canonical_unknown_profile",
                "repeat": repeat,
                "python_hash_seed": seed,
                "returncode": int(completed.returncode),
                "digest": payload.get("digest"),
                "canonical_hex": payload.get("canonical_hex"),
                "root_has_plan_v2": payload.get("root_has_plan_v2"),
                "v1_route_schema": payload.get("v1_route_schema"),
                "stderr_tail": completed.stderr[-2000:],
            }
        )
    return rows


def _report(summary: dict[str, Any]) -> str:
    full = summary["full"]
    legacy = full.get("legacy", _empty_counts())
    return (
        "# Path Planner v2 Gate 1 合同报告\n\n"
        f"- Gate 状态：`{summary['status']}`\n"
        f"- focused：`{summary['focused'].get('status', 'not_run')}`\n"
        f"- legacy 分账：{legacy.get('passed', 0)} passed / "
        f"{legacy.get('skipped', 0)} skipped / {legacy.get('failures', 0)} failed / "
        f"{legacy.get('errors', 0)} errors\n"
        f"- byte repeat：`{summary['byte_repeat'].get('status', 'not_run')}`\n"
        f"- 下一步：`{summary['next_required_change']}`\n\n"
        "本 Gate 只冻结 profile、provider、API 与安全边界合同；不发布 checkpoint、"
        "不替换默认策略、不连接真实 executor、不启动 online canary。\n"
    )


def _dry_run_payloads(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    boundary_ok = boundaries_match(config.get("boundaries"))
    status = "dry_run" if boundary_ok else "failed"
    next_change = (
        "execute_gate1_contract" if boundary_ok else "restore_gate1_safety_boundaries"
    )
    preflight = {"status": "not_run"}
    postflight = {"status": "not_run"}
    focused = {"status": "not_run", "tests": 0, "passed": 0, "skipped": 0, "failures": 0, "errors": 0}
    full = {
        "status": "not_run",
        "total": _empty_counts(),
        "legacy": _empty_counts(),
        "v2": _empty_counts(),
    }
    byte_repeat = {"status": "not_run", "repeat_count": 0, "digests": []}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": next_change,
        "preflight": preflight,
        "postflight": postflight,
        "checks": {
            "preflight": False,
            "postflight": False,
            "focused": False,
            "full": False,
            "byte_repeat": False,
            "boundaries_strict_false": boundary_ok,
        },
        "focused": focused,
        "full": full,
        "byte_repeat": byte_repeat,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    rows = [
        {"suite": "boundary-review", "check": field, "status": "passed" if boundary_ok else "failed", "value": False}
        for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
    ]
    phases = [
        {"phase": phase, "status": "not_run" if phase != "boundary-review" else ("completed" if boundary_ok else "failed")}
        for phase in ("preflight", "focused", "full", "byte-repeat", "boundary-review")
    ]
    review = {
        "schema_version": "xunce-path-v2-gate1-review/v1",
        "status": "dry_run" if boundary_ok else "failed",
        "checks": {"boundaries_strict_false": boundary_ok},
        "preflight": preflight,
        "postflight": postflight,
    }
    routing = {
        "schema_version": "xunce-path-v2-gate1-routing/v1",
        "stage_id": STAGE_ID,
        "status": status,
        "route": next_change,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    return summary, routing, rows, phases, review


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _load_gate2_dataset(
    *,
    dataset_name: str,
    path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    input_path = Path(path)
    base = {
        "dataset_name": dataset_name,
        "input_path": input_path.as_posix(),
        "rows": [],
        "summary": None,
        "input_sha256": None,
        "rows_sha256": None,
    }
    if not artifact_io.path_is_file(input_path):
        return {**base, "status": "missing"}

    try:
        nested_src = str((Path(repo_root) / "path-planner" / "src").resolve())
        if not sys.path or sys.path[0] != nested_src:
            sys.path.insert(0, nested_src)
        benchmark = importlib.import_module("path_planner.v2.benchmark")
        expected_origin = (
            Path(repo_root) / "path-planner" / "src" / "path_planner" / "v2" / "benchmark.py"
        ).resolve()
        actual_origin = Path(str(getattr(benchmark, "__file__"))).resolve()
        if actual_origin != expected_origin:
            raise ImportError("Gate 2 benchmark module origin is outside the worktree")
        hard_timeout_ms = getattr(benchmark, "HARD_TIMEOUT_MS_V2")
        if (
            isinstance(hard_timeout_ms, bool)
            or not isinstance(hard_timeout_ms, (int, float))
            or float(hard_timeout_ms) != GATE2_THRESHOLDS["hard_timeout_ms"]
        ):
            raise _Gate2LoaderContractError(
                "Gate 2 benchmark hard timeout contract drifted"
            )
        row_type_name, aggregate_name = GATE2_ROW_API[dataset_name]
        row_type = getattr(benchmark, row_type_name)
        aggregate = getattr(benchmark, aggregate_name)
        input_bytes = artifact_io.read_bytes(input_path)
        base["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
        payloads: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            input_bytes.decode("utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(
                    f"Gate 2 JSONL row {line_number} must not be blank"
                )
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Gate 2 JSONL row {line_number} must be an object"
                )
            payloads.append(payload)
        parsed = tuple(row_type.from_dict(payload) for payload in payloads)
        aggregate_summary = aggregate(parsed)
        rows = [row.to_dict() for row in aggregate_summary.rows]
        summary = aggregate_summary.to_dict()
        canonical_rows = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **base,
            "status": "loaded",
            "rows": rows,
            "summary": summary,
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        }
    except (AttributeError, ImportError, KeyError, _Gate2LoaderContractError) as exc:
        return {
            **base,
            "status": "internal_error",
            "error_type": type(exc).__name__,
        }
    except (
        TypeError,
        ValueError,
        OverflowError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        return {
            **base,
            "status": "invalid",
            "error_type": type(exc).__name__,
        }


def _gate2_formal_count(dataset_name: str, dataset: dict[str, Any]) -> int:
    summary = dataset.get("summary")
    if not isinstance(summary, dict):
        return 0
    field = (
        "formal_episode_count"
        if dataset_name == "standard_episodes"
        else "formal_row_count"
    )
    value = summary.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _gate2_number(summary: Any, field: str) -> float | int | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _gate2_exact_cost_ratios(summary: Any) -> tuple[float, ...] | None:
    if not isinstance(summary, dict):
        return None
    raw_ratios = summary.get("resource_cost_ratios")
    if not isinstance(raw_ratios, list):
        return None
    ratios: list[float] = []
    for item in raw_ratios:
        if not isinstance(item, dict) or set(item) != {"row_id", "ratio"}:
            return None
        row_id = item.get("row_id")
        ratio = item.get("ratio")
        if not isinstance(row_id, str) or not row_id:
            return None
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            return None
        try:
            normalized_ratio = float(ratio)
        except OverflowError:
            return None
        if not math.isfinite(normalized_ratio):
            return None
        ratios.append(normalized_ratio)
    return tuple(ratios)


def _evaluate_gate2(
    *,
    preflight: dict[str, Any],
    postflight_ok: bool,
    focused: dict[str, Any],
    full: dict[str, Any],
    boundary_ok: bool,
    datasets: dict[str, dict[str, Any]],
) -> tuple[str, str, list[str], dict[str, bool]]:
    primitive = datasets["primitive_audit"]
    exact = datasets["exact_map_quality"]
    standard = datasets["standard_episodes"]
    primitive_summary = primitive.get("summary")
    exact_summary = exact.get("summary")
    standard_summary = standard.get("summary")
    exact_cost_ratios = _gate2_exact_cost_ratios(exact_summary)

    primitive_ready = (
        primitive.get("status") == "loaded"
        and _gate2_formal_count("primitive_audit", primitive)
        >= GATE2_THRESHOLDS["min_primitive_independent_samples"]
    )
    exact_ready = (
        exact.get("status") == "loaded"
        and _gate2_formal_count("exact_map_quality", exact)
        >= GATE2_THRESHOLDS["min_exact_map_independent_cases"]
    )
    standard_ready = (
        standard.get("status") == "loaded"
        and _gate2_formal_count("standard_episodes", standard)
        >= GATE2_THRESHOLDS["min_standard_independent_episodes"]
    )

    checks = {
        "preflight": preflight.get("status") == "passed",
        "postflight": postflight_ok,
        "focused": focused.get("status") == "passed",
        "full": full.get("status") == "passed",
        "benchmark_loader": all(
            dataset.get("status") != "internal_error"
            for dataset in datasets.values()
        ),
        "boundaries_strict_false": boundary_ok,
        "primitive_formal_denominator": primitive_ready,
        "exact_map_formal_denominator": exact_ready,
        "standard_formal_denominator": standard_ready,
        "primitive_false_positives": (
            primitive_ready
            and _gate2_number(primitive_summary, "false_positive_count") is not None
            and _gate2_number(primitive_summary, "false_positive_count")
            <= GATE2_THRESHOLDS["max_primitive_false_positives"]
        ),
        "primitive_recall": (
            primitive_ready
            and _gate2_number(primitive_summary, "primitive_recall") is not None
            and _gate2_number(primitive_summary, "primitive_recall")
            >= GATE2_THRESHOLDS["min_primitive_recall"]
        ),
        "primitive_complete_l2": (
            primitive_ready
            and _gate2_number(primitive_summary, "complete_l2_ratio")
            == GATE2_THRESHOLDS["min_primitive_complete_l2_ratio"]
        ),
        "exact_map_success": (
            exact_ready
            and _gate2_number(exact_summary, "provider_success_ratio") is not None
            and _gate2_number(exact_summary, "provider_success_ratio")
            >= GATE2_THRESHOLDS["min_exact_map_success_ratio"]
        ),
        "exact_map_complete_l2": (
            exact_ready
            and _gate2_number(exact_summary, "complete_l2_ratio")
            == GATE2_THRESHOLDS["min_exact_map_complete_l2_ratio"]
        ),
        "exact_map_resource_cost": (
            exact_ready
            and exact_cost_ratios is not None
            and len(exact_cost_ratios)
            == _gate2_number(exact_summary, "provider_success_count")
            and bool(exact_cost_ratios)
            and all(
                GATE2_THRESHOLDS["min_exact_map_resource_cost_ratio"]
                <= ratio
                <= GATE2_THRESHOLDS["max_exact_map_resource_cost_ratio"]
                for ratio in exact_cost_ratios
            )
            and _gate2_number(exact_summary, "max_resource_cost_ratio") is not None
            and max(exact_cost_ratios)
            == _gate2_number(exact_summary, "max_resource_cost_ratio")
            and _gate2_number(exact_summary, "max_resource_cost_ratio")
            <= GATE2_THRESHOLDS["max_exact_map_resource_cost_ratio"]
        ),
        "standard_reachable_success": (
            standard_ready
            and _gate2_number(standard_summary, "reachable_query_success_ratio")
            is not None
            and _gate2_number(standard_summary, "reachable_query_success_ratio")
            >= GATE2_THRESHOLDS["min_standard_reachable_success_ratio"]
        ),
        "standard_complete_l2": (
            standard_ready
            and _gate2_number(standard_summary, "complete_l2_ratio")
            == GATE2_THRESHOLDS["min_standard_complete_l2_ratio"]
        ),
        "standard_p95_runtime": (
            standard_ready
            and _gate2_number(standard_summary, "runtime_p95_ms") is not None
            and _gate2_number(standard_summary, "runtime_p95_ms")
            <= GATE2_THRESHOLDS["max_standard_p95_runtime_ms"]
        ),
        "hard_timeout_bound": all(
            ready
            and _gate2_number(dataset.get("summary"), "hard_timeout_violation_count")
            is not None
            and _gate2_number(dataset.get("summary"), "hard_timeout_violation_count")
            <= GATE2_THRESHOLDS["max_hard_timeout_violations"]
            for dataset, ready in (
                (primitive, primitive_ready),
                (exact, exact_ready),
                (standard, standard_ready),
            )
        ),
    }

    code_check_order = (
        ("preflight", "restore_gate2_runtime_isolation"),
        ("postflight", "restore_gate2_runtime_isolation"),
        ("boundaries_strict_false", "restore_gate2_safety_boundaries"),
        ("focused", "restore_gate2_focused_contracts"),
        ("full", "restore_path_planner_v1_regression"),
        ("benchmark_loader", "repair_gate2_benchmark_loader"),
    )
    for check, route in code_check_order:
        if not checks[check]:
            return "failed", route, [], checks

    blockers = [
        blocker
        for (dataset_name, blocker), ready in zip(
            GATE2_BLOCKERS,
            (primitive_ready, exact_ready, standard_ready),
            strict=True,
        )
        if not ready
    ]
    if blockers:
        return "blocked", blockers[0], blockers, checks

    metric_check_order = (
        ("primitive_false_positives", "repair_wheel_oracle_false_positives"),
        ("primitive_complete_l2", "repair_wheel_complete_l2_validation"),
        ("primitive_recall", "repair_wheel_primitive_recall"),
        ("hard_timeout_bound", "repair_wheel_hard_timeout_bound"),
        ("exact_map_success", "repair_wheel_exact_map_reachability"),
        ("exact_map_complete_l2", "repair_wheel_exact_map_l2_validation"),
        ("exact_map_resource_cost", "repair_wheel_exact_map_resource_cost"),
        ("standard_reachable_success", "repair_wheel_standard_reachability"),
        ("standard_complete_l2", "repair_wheel_standard_l2_validation"),
        ("standard_p95_runtime", "repair_wheel_standard_runtime"),
    )
    for check, route in metric_check_order:
        if not checks[check]:
            return "failed", route, [], checks
    return "passed", GATE2_PASS_ROUTE, [], checks


def _gate2_report(summary: dict[str, Any]) -> str:
    primitive = summary["datasets"]["primitive_audit"].get("summary") or {}
    exact = summary["datasets"]["exact_map_quality"].get("summary") or {}
    standard = summary["datasets"]["standard_episodes"].get("summary") or {}
    blockers = summary.get("blocking_reasons", [])
    blocker_text = "、".join(f"`{item}`" for item in blockers) if blockers else "无"
    return (
        "# Path Planner v2 Gate 2 轮式证据报告\n\n"
        f"- Gate 状态：`{summary['status']}`\n"
        f"- 下一步：`{summary['next_required_change']}`\n"
        f"- 正式阻塞项：{blocker_text}\n"
        f"- 独立 primitive 分母：{primitive.get('formal_row_count', 0)}\n"
        f"- oracle false positive：{primitive.get('false_positive_count', 'N/A')}\n"
        f"- primitive recall：{primitive.get('primitive_recall', 'N/A')}\n"
        f"- exact-map 最大成本比：{exact.get('max_resource_cost_ratio', 'N/A')}\n"
        f"- Standard 独立 episodes：{standard.get('formal_episode_count', 0)}\n"
        f"- Standard p95：{standard.get('runtime_p95_ms', 'N/A')} ms\n\n"
        "自标注行不进入正式分母；缺失的独立 oracle、exact optimum 或 Standard schedule "
        "只会产生可审计 blocker，不会伪造通过。v1 仍为默认；本 Gate 不发布 checkpoint、"
        "不替换 default policy、不连接 executor、不启动 canary。\n"
    )


def _gate2_dry_run_payloads(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    boundary_ok = boundaries_match(config.get("boundaries"))
    status = "dry_run" if boundary_ok else "failed"
    route = GATE2_EXECUTE_ROUTE if boundary_ok else "restore_gate2_safety_boundaries"
    datasets = {
        name: {
            "dataset_name": name,
            "input_path": config["inputs"][name],
            "status": "not_run",
            "rows": [],
            "summary": None,
        }
        for name in GATE2_INPUTS
    }
    checks = {
        "preflight": False,
        "postflight": False,
        "focused": False,
        "full": False,
        "benchmark_loader": False,
        "boundaries_strict_false": boundary_ok,
        "primitive_formal_denominator": False,
        "exact_map_formal_denominator": False,
        "standard_formal_denominator": False,
    }
    summary = {
        "schema_version": GATE2_SCHEMA_VERSION,
        "stage_id": GATE2_STAGE_ID,
        "status": status,
        "next_required_change": route,
        "blocking_reasons": [],
        "preflight": {"status": "not_run"},
        "postflight": {"status": "not_run"},
        "checks": checks,
        "focused": {"status": "not_run"},
        "full": {"status": "not_run"},
        "datasets": datasets,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    routing = {
        "schema_version": "xunce-path-v2-gate2-routing/v1",
        "stage_id": GATE2_STAGE_ID,
        "status": status,
        "route": route,
        "blocking_reasons": [],
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    rows = [
        {
            "suite": "boundary-review",
            "check": field,
            "status": "passed" if boundary_ok else "failed",
            "value": False,
        }
        for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
    ]
    phases = [
        {
            "phase": phase,
            "status": "completed" if phase == "boundary-review" and boundary_ok else "not_run",
        }
        for phase in (
            "preflight",
            "focused",
            "full",
            "primitive-audit",
            "exact-map-quality",
            "standard-episodes",
            "boundary-review",
        )
    ]
    review = {
        "schema_version": "xunce-path-v2-gate2-review/v1",
        "status": status,
        "checks": checks,
        "preflight": summary["preflight"],
        "postflight": summary["postflight"],
        "datasets": datasets,
    }
    return summary, routing, rows, phases, review


def _run_gate2_benchmark(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    execute_tests: bool,
) -> dict[str, Any]:
    _validate_gate2_config(config)
    _assert_no_stale_artifacts(output_root)
    if not execute_tests:
        summary, routing, rows, phases, review = _gate2_dry_run_payloads(config)
    else:
        preflight = _gate2_preflight(config, repo_root)
        attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        attempt_root = Path(str(config["temp_root"])) / attempt_id
        artifact_io.make_dirs(attempt_root / "mpl")
        env = _common_env(repo_root, attempt_root)
        python = Path(str(config["python"])).resolve()
        focused_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        full_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        focused: dict[str, Any] = {"status": "not_run"}
        full: dict[str, Any] = {"status": "not_run"}
        datasets: dict[str, dict[str, Any]] = {
            name: {
                "dataset_name": name,
                "input_path": config["inputs"][name],
                "status": "not_run",
                "rows": [],
                "summary": None,
            }
            for name in GATE2_INPUTS
        }

        if preflight.get("status") == "passed":
            focused_junit = attempt_root / "focused.junit.xml"
            full_junit = attempt_root / "full.junit.xml"
            focused_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=GATE2_FOCUSED_TARGETS,
                junit_path=focused_junit,
                basetemp=attempt_root / "focused-basetemp",
                env=env,
            )
            focused = _audit_focused_junit(focused_junit)
            focused["returncode"] = focused_run["returncode"]
            focused["status"] = (
                "passed"
                if focused.get("status") == "passed" and focused_run["returncode"] == 0
                else "failed"
            )
            full_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=("tests",),
                junit_path=full_junit,
                basetemp=attempt_root / "full-basetemp",
                env=env,
            )
            full = _audit_gate2_full_junit(
                full_junit,
                expected_legacy=LEGACY_EXPECTED,
                allowed_skip_dependency=ALLOWED_SKIP_DEPENDENCY,
                baseline_evidence=config["baseline_evidence"],
            )
            full["returncode"] = full_run["returncode"]
            full["status"] = (
                "passed"
                if full.get("status") == "passed" and full_run["returncode"] == 0
                else "failed"
            )
            if focused["status"] == "passed" and full["status"] == "passed":
                datasets = {
                    name: _load_gate2_dataset(
                        dataset_name=name,
                        path=Path(config["inputs"][name]),
                        repo_root=repo_root,
                    )
                    for name in GATE2_INPUTS
                }

        postflight = _gate2_postflight(config, repo_root)
        postflight_ok = _postflight_matches(preflight, postflight)
        boundary_ok = boundaries_match(config.get("boundaries"))
        status, route, blockers, checks = _evaluate_gate2(
            preflight=preflight,
            postflight_ok=postflight_ok,
            focused=focused,
            full=full,
            boundary_ok=boundary_ok,
            datasets=datasets,
        )
        public_datasets = {
            name: {key: value for key, value in dataset.items() if key != "rows"}
            for name, dataset in datasets.items()
        }
        summary = {
            "schema_version": GATE2_SCHEMA_VERSION,
            "stage_id": GATE2_STAGE_ID,
            "status": status,
            "next_required_change": route,
            "blocking_reasons": blockers,
            "preflight": preflight,
            "postflight": postflight,
            "checks": checks,
            "focused": focused,
            "full": full,
            "datasets": public_datasets,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        routing = {
            "schema_version": "xunce-path-v2-gate2-routing/v1",
            "stage_id": GATE2_STAGE_ID,
            "status": status,
            "route": route,
            "blocking_reasons": blockers,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        rows = [
            {
                "suite": "preflight",
                "check": "git_import_identity",
                "status": preflight.get("status", "failed"),
            },
            {
                "suite": "focused",
                "check": "pytest",
                "status": focused.get("status", "not_run"),
                "passed": focused.get("passed", 0),
                "skipped": focused.get("skipped", 0),
            },
            {
                "suite": "full",
                "check": "pytest",
                "status": full.get("status", "not_run"),
                **full.get("total", _empty_counts()),
            },
            {
                "suite": "boundary-review",
                "check": "runtime_postflight",
                "status": "passed" if postflight_ok else "failed",
            },
        ]
        for name, blocker in GATE2_BLOCKERS:
            dataset = datasets[name]
            dataset_suite = name.replace("_", "-")
            if dataset.get("rows"):
                rows.extend(
                    {"suite": dataset_suite, **row}
                    for row in dataset["rows"]
                )
            else:
                rows.append(
                    {
                        "suite": dataset_suite,
                        "check": "formal_input",
                        "status": dataset.get("status", "not_run"),
                        "reason": (
                            "repair_gate2_benchmark_loader"
                            if dataset.get("status") == "internal_error"
                            else blocker
                        ),
                    }
                )
        rows.extend(
            {
                "suite": "boundary-review",
                "check": field,
                "status": "passed" if boundary_ok else "failed",
                "value": False,
            }
            for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
        )
        rows.sort(
            key=lambda row: (
                str(row.get("suite", "")),
                int(row.get("seed", -1)),
                str(
                    row.get(
                        "row_id",
                        row.get("episode_id", row.get("check", "")),
                    )
                ),
            )
        )
        phases = [
            {
                "phase": "preflight",
                "status": "completed" if preflight.get("status") == "passed" else "failed",
            },
            {
                "phase": "focused",
                "status": "completed" if focused.get("status") == "passed" else focused.get("status", "not_run"),
            },
            {
                "phase": "full",
                "status": "completed" if full.get("status") == "passed" else full.get("status", "not_run"),
            },
            *[
                {
                    "phase": name.replace("_", "-"),
                    "status": (
                        "failed"
                        if datasets[name].get("status") == "internal_error"
                        else (
                            "not_run"
                            if datasets[name].get("status") == "not_run"
                            else (
                                "completed"
                                if datasets[name].get("status") == "loaded"
                                and checks[
                                    {
                                        "primitive_audit": "primitive_formal_denominator",
                                        "exact_map_quality": "exact_map_formal_denominator",
                                        "standard_episodes": "standard_formal_denominator",
                                    }[name]
                                ]
                                else "blocked"
                            )
                        )
                    ),
                }
                for name in GATE2_INPUTS
            ],
            {
                "phase": "boundary-review",
                "status": "completed" if boundary_ok and postflight_ok else "failed",
            },
        ]
        isolation_env = {
            key: env[key]
            for key in (
                "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONPATH",
                "TEMP",
                "TMP",
                "MPLCONFIGDIR",
            )
        }
        review = {
            "schema_version": "xunce-path-v2-gate2-review/v1",
            "status": status,
            "checks": checks,
            "preflight": preflight,
            "postflight": postflight,
            "datasets": public_datasets,
            "execution": {
                "attempt_root": str(attempt_root),
                "isolation_env": isolation_env,
                "focused_command_result": focused_run,
                "full_command_result": full_run,
            },
        }

    gate_artifacts.write_gate_artifacts(
        output_root=output_root,
        config=config,
        summary=summary,
        routing=routing,
        rows=rows,
        phases=phases,
        review=review,
        report=_gate2_report(summary),
    )
    return summary


def run_gate_benchmark(
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    execute_tests: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    repo_root = Path(repo_root).resolve()
    output_root = validate_output_root(repo_root, output_root)
    config = artifact_io.read_json(config_path)
    if config.get("schema_version") == GATE2_SCHEMA_VERSION:
        return _run_gate2_benchmark(
            config=config,
            output_root=output_root,
            repo_root=repo_root,
            execute_tests=execute_tests,
        )
    _validate_config(config)

    if execute_tests:
        preflight = _preflight(config, repo_root)
        if preflight.get("status") != "passed":
            raise RuntimeError("Gate 1 git/import preflight failed before output creation")
    else:
        preflight = {"status": "not_run"}

    _assert_no_stale_artifacts(output_root)

    if not execute_tests:
        summary, routing, rows, phases, review = _dry_run_payloads(config)
    else:
        attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        attempt_root = Path(str(config["temp_root"])) / attempt_id
        artifact_io.make_dirs(attempt_root / "mpl")
        env = _common_env(repo_root, attempt_root)
        python = Path(str(config["python"])).resolve()
        focused_junit = attempt_root / "focused.junit.xml"
        full_junit = attempt_root / "full.junit.xml"
        focused_run = _run_pytest(
            python=python,
            repo_root=repo_root,
            targets=FOCUSED_TARGETS,
            junit_path=focused_junit,
            basetemp=attempt_root / "focused-basetemp",
            env=env,
        )
        focused = _audit_focused_junit(focused_junit)
        focused["returncode"] = focused_run["returncode"]
        focused["status"] = (
            "passed"
            if focused["status"] == "passed" and focused_run["returncode"] == 0
            else "failed"
        )

        full_run = _run_pytest(
            python=python,
            repo_root=repo_root,
            targets=("tests",),
            junit_path=full_junit,
            basetemp=attempt_root / "full-basetemp",
            env=env,
        )
        full = audit_full_junit(
            full_junit,
            expected_legacy=LEGACY_EXPECTED,
            allowed_skip_dependency=ALLOWED_SKIP_DEPENDENCY,
        )
        full["returncode"] = full_run["returncode"]
        full["status"] = (
            "passed"
            if full["status"] == "passed" and full_run["returncode"] == 0
            else "failed"
        )

        repeat_rows = _run_byte_repeats(
            python=python,
            repo_root=repo_root,
            seeds=config["byte_repeat"]["python_hash_seeds"],
            common_env=env,
        )
        byte_repeat = audit_repeat_digests(repeat_rows)
        probe_boundary_ok = all(
            row["returncode"] == 0
            and row["root_has_plan_v2"] is False
            and row["v1_route_schema"] == "path-planner-route/v1"
            for row in repeat_rows
        )
        postflight = _postflight(config, repo_root)
        postflight_ok = _postflight_matches(preflight, postflight)
        boundary_ok = boundaries_match(config.get("boundaries")) and probe_boundary_ok
        checks = {
            "preflight": preflight["status"] == "passed",
            "postflight": postflight_ok,
            "focused": focused["status"] == "passed",
            "full": full["status"] == "passed",
            "byte_repeat": byte_repeat["status"] == "passed",
            "boundaries_strict_false": boundary_ok,
        }
        passed = all(checks.values())
        status = "passed" if passed else "failed"
        if not postflight_ok:
            next_change = "restore_gate1_runtime_isolation"
        elif not boundaries_match(config.get("boundaries")):
            next_change = "restore_gate1_safety_boundaries"
        elif not focused["status"] == "passed":
            next_change = "restore_gate1_focused_contracts"
        elif not full["status"] == "passed":
            next_change = "restore_path_planner_v1_regression"
        elif byte_repeat["status"] != "passed" or not probe_boundary_ok:
            next_change = "restore_gate1_byte_stability_and_v1_isolation"
        else:
            next_change = PASS_ROUTE

        summary = {
            "schema_version": SCHEMA_VERSION,
            "stage_id": STAGE_ID,
            "status": status,
            "next_required_change": next_change,
            "preflight": preflight,
            "postflight": postflight,
            "checks": checks,
            "focused": focused,
            "full": full,
            "byte_repeat": byte_repeat,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        routing = {
            "schema_version": "xunce-path-v2-gate1-routing/v1",
            "stage_id": STAGE_ID,
            "status": status,
            "route": next_change,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        rows = [
            {"suite": "preflight", "check": "git_import_identity", "status": preflight["status"]},
            {"suite": "boundary-review", "check": "runtime_postflight", "status": "passed" if postflight_ok else "failed"},
            {"suite": "focused", "check": "pytest", "status": focused["status"], "passed": focused["passed"], "skipped": focused["skipped"]},
            {"suite": "full", "check": "pytest", "status": full["status"], **full["total"]},
            {"suite": "full", "check": "legacy_counts", "status": "passed" if full["legacy"] == LEGACY_EXPECTED else "failed", **full["legacy"]},
            {"suite": "full", "check": "legacy_skips", "status": "passed" if full["legacy_skip_contract"] else "failed", "skipped": full["legacy"]["skipped"]},
            *repeat_rows,
            *[
                {"suite": "boundary-review", "check": field, "status": "passed" if config["boundaries"].get(field) is False else "failed", "value": False}
                for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
            ],
            {"suite": "boundary-review", "check": "root_api_isolation", "status": "passed" if probe_boundary_ok else "failed"},
        ]
        rows.sort(key=lambda row: (str(row.get("suite", "")), str(row.get("check", "")), int(row.get("repeat", 0))))
        phases = [
            {"phase": "preflight", "status": "completed"},
            {"phase": "focused", "status": "completed" if focused["status"] == "passed" else "failed"},
            {"phase": "full", "status": "completed" if full["status"] == "passed" else "failed"},
            {"phase": "byte-repeat", "status": "completed" if byte_repeat["status"] == "passed" else "failed"},
            {"phase": "boundary-review", "status": "completed" if boundary_ok and postflight_ok else "failed"},
        ]
        isolation_env = {
            key: env[key]
            for key in (
                "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONPATH",
                "TEMP",
                "TMP",
                "MPLCONFIGDIR",
            )
        }
        review = {
            "schema_version": "xunce-path-v2-gate1-review/v1",
            "status": status,
            "checks": checks,
            "preflight": preflight,
            "postflight": postflight,
            "execution": {
                "attempt_root": str(attempt_root),
                "isolation_env": isolation_env,
                "focused_command_result": focused_run,
                "full_command_result": full_run,
            },
            "focused_junit": focused,
            "full_junit": full,
            "repeat_rows": repeat_rows,
            "byte_repeat": byte_repeat,
            "probe_boundary_ok": probe_boundary_ok,
        }

    gate_artifacts.write_gate_artifacts(
        output_root=output_root,
        config=config,
        summary=summary,
        routing=routing,
        rows=rows,
        phases=phases,
        review=review,
        report=_report(summary),
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Path Planner v2 gate evidence")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/xunce_path_v2_gate1_contract_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("D:/xunce/out/path_v2/g1"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser


def status_exit_code(status: str) -> int:
    return 0 if status in {"passed", "blocked", "dry_run"} else 1


def main() -> int:
    args = _parser().parse_args()
    summary = run_gate_benchmark(
        config_path=args.config,
        output_root=args.output_root,
        repo_root=args.repo_root,
        execute_tests=not args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return status_exit_code(summary["status"])


if __name__ == "__main__":
    raise SystemExit(main())
