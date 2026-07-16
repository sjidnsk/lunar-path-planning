from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
    parser = argparse.ArgumentParser(description="Run Path Planner v2 Gate 1 contracts")
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


def main() -> int:
    args = _parser().parse_args()
    summary = run_gate_benchmark(
        config_path=args.config,
        output_root=args.output_root,
        repo_root=args.repo_root,
        execute_tests=not args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] in {"passed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
