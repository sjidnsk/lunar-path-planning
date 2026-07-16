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


def _python_matches(left: Path, right: Path) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if config.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID!r}")
    if not _python_matches(Path(str(config.get("python"))), FORMAL_PYTHON):
        raise ValueError(f"python must be {FORMAL_PYTHON.as_posix()}")
    if not _python_matches(Path(str(config.get("temp_root"))), FORMAL_TEMP_ROOT):
        raise ValueError(f"temp_root must be {FORMAL_TEMP_ROOT.as_posix()}")
    if tuple(config.get("focused", {}).get("pytest_targets", ())) != FOCUSED_TARGETS:
        raise ValueError("focused pytest_targets must be the exact six Gate 1 files")
    if config.get("full", {}).get("pytest_targets") != ["tests"]:
        raise ValueError("full pytest_targets must be exactly ['tests']")
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


def _preflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    expected = config["expected_git"]
    python = Path(str(config["python"]))
    superproject = gate0.audit_git_identity(
        repo_root,
        str(expected["branch"]),
        str(expected["base_commit"]),
    )
    nested = _audit_nested_git(repo_root, str(expected["nested_branch"]))
    imports = gate0.audit_import_origins(python, repo_root)
    python_version_matches = (
        imports.get("python_version") == config.get("expected_python_version")
    )
    passed = (
        superproject.get("status") == "passed"
        and nested.get("status") == "passed"
        and imports.get("status") == "passed"
        and python_version_matches
    )
    return {
        "schema_version": "xunce-path-v2-gate1-preflight/v1",
        "status": "passed" if passed else "failed",
        "superproject_git": superproject,
        "nested_git": nested,
        "import_origins": imports,
        "python_version_matches": python_version_matches,
    }


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
            targets=config["focused"]["pytest_targets"],
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
            targets=config["full"]["pytest_targets"],
            junit_path=full_junit,
            basetemp=attempt_root / "full-basetemp",
            env=env,
        )
        full = audit_full_junit(
            full_junit,
            expected_legacy=config["full"]["legacy_expected"],
            allowed_skip_dependency=str(config["full"]["allowed_skip_dependency"]),
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
        boundary_ok = boundaries_match(config.get("boundaries")) and probe_boundary_ok
        checks = {
            "preflight": preflight["status"] == "passed",
            "focused": focused["status"] == "passed",
            "full": full["status"] == "passed",
            "byte_repeat": byte_repeat["status"] == "passed",
            "boundaries_strict_false": boundary_ok,
        }
        passed = all(checks.values())
        status = "passed" if passed else "failed"
        if not boundaries_match(config.get("boundaries")):
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
            {"suite": "focused", "check": "pytest", "status": focused["status"], "passed": focused["passed"], "skipped": focused["skipped"]},
            {"suite": "full", "check": "pytest", "status": full["status"], **full["total"]},
            {"suite": "full", "check": "legacy_counts", "status": "passed" if full["legacy"] == {key: int(config["full"]["legacy_expected"][key]) for key in ("passed", "skipped", "failures", "errors")} else "failed", **full["legacy"]},
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
            {"phase": "boundary-review", "status": "completed" if boundary_ok else "failed"},
        ]
        review = {
            "schema_version": "xunce-path-v2-gate1-review/v1",
            "status": status,
            "checks": checks,
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
