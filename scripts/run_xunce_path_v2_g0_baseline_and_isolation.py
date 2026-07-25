from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import path_v2_gate_artifacts as gate_artifacts
import xunce_artifact_io as artifact_io


SCHEMA_VERSION = "xunce-path-v2-g0-baseline-and-isolation/v1"
PUBLIC_FUNCTIONS = (
    "parse_junit(path: Path) -> JUnitSummary",
    "audit_import_origins(python: Path, repo_root: Path) -> dict[str, Any]",
    "audit_git_identity(repo_root: Path, expected_branch: str, expected_base_commit: str) -> dict[str, Any]",
    "run_gate0(config_path: Path, output_root: Path, repo_root: Path, execute_tests: bool = True) -> dict[str, Any]",
)
FROZEN_FOUNDATION_IDENTITY_NODEIDS = frozenset(
    {
        "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
        "test_ten_episode_workflow_writes_finite_append_only_machine_artifacts",
        "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
        "test_stage1_gate_binds_required_sources_without_writing_gate_artifact",
        "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
        "test_stage1_gate_fails_closed_on_missing_review_jump_state_and_artifact_drift",
        "tests/ppo_highres_frontier/test_stage1_smoke_env.py::"
        "test_stage1_gate_rejects_dirty_tree_and_has_no_force_path",
    }
)


@dataclass(frozen=True)
class JUnitSummary:
    tests: int
    passed: int
    failures: int
    errors: int
    skipped: int
    failed_nodeids: Sequence[str]


def _testcase_nodeid(testcase: ET.Element) -> str:
    classname = testcase.attrib.get("classname", "").strip()
    name = testcase.attrib.get("name", "").strip()
    if not classname:
        return name
    module_path = classname.replace(".", "/")
    if not module_path.endswith(".py"):
        module_path += ".py"
    return f"{module_path}::{name}"


def _junit_testcases(path: Path) -> list[ET.Element]:
    root = ET.fromstring(artifact_io.read_bytes(path))
    return list(root.iter("testcase"))


def parse_junit(path: Path) -> JUnitSummary:
    testcases = _junit_testcases(path)
    failures = [case for case in testcases if case.find("failure") is not None]
    errors = [case for case in testcases if case.find("error") is not None]
    skipped = [case for case in testcases if case.find("skipped") is not None]
    passed = len(testcases) - len(failures) - len(errors) - len(skipped)
    return JUnitSummary(
        tests=len(testcases),
        passed=passed,
        failures=len(failures),
        errors=len(errors),
        skipped=len(skipped),
        failed_nodeids=tuple(_testcase_nodeid(case) for case in failures),
    )


def _skipped_messages(path: Path) -> tuple[str, ...]:
    messages: list[str] = []
    for testcase in _junit_testcases(path):
        skipped = testcase.find("skipped")
        if skipped is None:
            continue
        messages.append(" ".join(filter(None, (skipped.attrib.get("message", ""), skipped.text or ""))))
    return tuple(messages)


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


def _resolve_git_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def audit_git_identity(
    repo_root: Path,
    expected_branch: str,
    expected_base_commit: str,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    try:
        branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        head = _git(repo_root, "rev-parse", "HEAD")
        merge_base = _git(repo_root, "merge-base", expected_base_commit, "HEAD")
        git_dir = _resolve_git_path(repo_root, _git(repo_root, "rev-parse", "--git-dir"))
        common_dir = _resolve_git_path(repo_root, _git(repo_root, "rev-parse", "--git-common-dir"))
        porcelain = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    except subprocess.CalledProcessError as exc:
        return {
            "schema_version": "xunce-path-v2-git-identity-audit/v1",
            "status": "failed",
            "error": (exc.stderr or str(exc)).strip(),
        }

    is_linked_worktree = git_dir != common_dir and os.path.isfile(repo_root / ".git")
    clean_tree = not porcelain
    checks = {
        "branch_matches": branch == expected_branch,
        "is_linked_worktree": is_linked_worktree,
        "base_is_ancestor": merge_base == expected_base_commit,
        "clean_tree": clean_tree,
    }
    return {
        "schema_version": "xunce-path-v2-git-identity-audit/v1",
        "status": "passed" if all(checks.values()) else "failed",
        "repo_root": str(repo_root),
        "branch": branch,
        "expected_branch": expected_branch,
        "head": head,
        "expected_base_commit": expected_base_commit,
        "merge_base": merge_base,
        "git_dir": str(git_dir),
        "git_common_dir": str(common_dir),
        "dirty_paths": porcelain.splitlines(),
        **checks,
    }


def audit_import_origins(python: Path, repo_root: Path) -> dict[str, Any]:
    python = Path(python).resolve()
    repo_root = Path(repo_root).resolve()
    path_planner_src = (repo_root / "path-planner" / "src").resolve()
    ppo_src = (repo_root / "src").resolve()
    audit_temp = Path("D:/xunce/tmp/path_v2_g0/import-origin-audit")
    artifact_io.make_dirs(audit_temp)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": os.pathsep.join((str(ppo_src), str(path_planner_src))),
            "TEMP": str(audit_temp),
            "TMP": str(audit_temp),
            "MPLCONFIGDIR": str(audit_temp / "mpl"),
        }
    )
    artifact_io.make_dirs(audit_temp / "mpl")
    code = (
        "import json, pathlib, sys; "
        "import path_planner, lunar_exploration_ppo; "
        "print(json.dumps({"
        "'python_version': '.'.join(map(str, sys.version_info[:3])), "
        "'python_no_user_site': bool(sys.flags.no_user_site), "
        "'path_planner': {'origin': str(pathlib.Path(path_planner.__file__).resolve())}, "
        "'lunar_exploration_ppo': {'origin': str(pathlib.Path(lunar_exploration_ppo.__file__).resolve())}"
        "}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return {
            "schema_version": "xunce-path-v2-import-origin-audit/v1",
            "status": "failed",
            "python": str(python),
            "stderr": completed.stderr.strip(),
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        path_planner_origin = Path(payload["path_planner"]["origin"]).resolve()
        ppo_origin = Path(payload["lunar_exploration_ppo"]["origin"]).resolve()
    except (IndexError, KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "schema_version": "xunce-path-v2-import-origin-audit/v1",
            "status": "failed",
            "python": str(python),
            "error": f"invalid import audit output: {exc}",
            "stdout": completed.stdout.strip(),
        }
    checks = {
        "python_no_user_site": payload.get("python_no_user_site") is True,
        "path_planner_from_worktree": path_planner_origin.is_relative_to(path_planner_src),
        "ppo_from_worktree": ppo_origin.is_relative_to(ppo_src),
    }
    return {
        "schema_version": "xunce-path-v2-import-origin-audit/v1",
        "status": "passed" if all(checks.values()) else "failed",
        "python": str(python),
        "python_version": payload.get("python_version"),
        "pythonpath": env["PYTHONPATH"],
        "path_planner": {"origin": str(path_planner_origin)},
        "lunar_exploration_ppo": {"origin": str(ppo_origin)},
        **checks,
    }


def _resolve_junit(output_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = output_root / path
    return path.resolve()


def _empty_junit() -> JUnitSummary:
    return JUnitSummary(tests=0, passed=0, failures=0, errors=0, skipped=0, failed_nodeids=())


def _run_pytest(
    *,
    python: Path,
    repo_root: Path,
    suite: dict[str, Any],
    junit_path: Path,
    basetemp: Path,
    common_env: dict[str, str],
) -> dict[str, Any]:
    working_directory = (repo_root / str(suite["working_directory"])).resolve()
    pythonpath = [str((repo_root / str(item)).resolve()) for item in suite["pythonpath"]]
    env = common_env.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    command = [str(python), "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    command.extend(str(item) for item in suite.get("pytest_targets", []))
    command.extend(("--basetemp", str(basetemp), "--junitxml", str(junit_path)))
    artifact_io.write_text(
        junit_path,
        '<?xml version="1.0" encoding="utf-8"?><testsuite tests="0" failures="0" errors="0" skipped="0" />',
    )
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "working_directory": str(working_directory),
        "pythonpath": env["PYTHONPATH"],
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def _execute_baselines(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    python = Path(str(config["python"])).resolve()
    attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    attempt_root = Path(str(config["temp_root"])) / attempt_id
    artifact_io.make_dirs(attempt_root / "mpl")
    common_env = os.environ.copy()
    common_env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TEMP": str(attempt_root),
            "TMP": str(attempt_root),
            "MPLCONFIGDIR": str(attempt_root / "mpl"),
        }
    )
    path_config = config["path_planner"]
    ppo_config = config["ppo_stage1"]
    path_junit = _resolve_junit(output_root, str(path_config["junit_xml"]))
    ppo_junit = _resolve_junit(output_root, str(ppo_config["junit_xml"]))
    path_result = _run_pytest(
        python=python,
        repo_root=repo_root,
        suite=path_config,
        junit_path=path_junit,
        basetemp=attempt_root / "path_planner",
        common_env=common_env,
    )
    ppo_result = _run_pytest(
        python=python,
        repo_root=repo_root,
        suite=ppo_config,
        junit_path=ppo_junit,
        basetemp=attempt_root / "ppo_stage1",
        common_env=common_env,
    )
    return {
        "executed": True,
        "attempt_root": str(attempt_root),
        "isolation_env": {key: common_env[key] for key in (
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "TEMP",
            "TMP",
            "MPLCONFIGDIR",
        )},
        "path_planner": path_result,
        "ppo_stage1": ppo_result,
    }


def _counts_match(observed: JUnitSummary, expected: dict[str, Any]) -> bool:
    return all(
        (
            observed.tests == int(expected["tests"]),
            observed.passed == int(expected["passed"]),
            observed.skipped == int(expected["skipped"]),
            observed.failures == int(expected["failures"]),
            observed.errors == int(expected["errors"]),
        )
    )


def _boundaries_match(configured: Any) -> bool:
    return (
        isinstance(configured, dict)
        and set(configured) == set(gate_artifacts.BOUNDARY_FIELDS)
        and all(configured[field] is False for field in gate_artifacts.BOUNDARY_FIELDS)
    )


def _known_failure_categories_match(rows: Any) -> bool:
    if not isinstance(rows, list) or len(rows) != 13:
        return False
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("nodeid"), str):
            return False
        expected_category = (
            "frozen_foundation_identity_mismatch"
            if row["nodeid"] in FROZEN_FOUNDATION_IDENTITY_NODEIDS
            else "inherited_external_baseline_symptom"
        )
        if row.get("category") != expected_category:
            return False
    return True


def _report(summary: dict[str, Any]) -> str:
    path = summary["path_planner"]
    ppo = summary["ppo_stage1"]
    return (
        "# Path Planner v2 Gate 0 基线与隔离报告\n\n"
        f"- Gate 状态：`{summary['status']}`\n"
        f"- path-planner：{path['passed']} passed / {path['skipped']} skipped / "
        f"{path['failed']} failed / {path['errors']} errors\n"
        f"- PPO Stage1：{ppo['passed']} passed / {ppo['known_failure_count']} 个精确已知外部失败 / "
        f"{ppo['errors']} errors / {ppo['skipped']} skipped\n"
        f"- 下一步：`{summary['next_required_change']}`\n\n"
        "本 Gate 只冻结基线与隔离证据，不发布 checkpoint、不替换默认策略、不连接真实 executor、"
        "不启动 online canary。\n"
    )


def run_gate0(
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    execute_tests: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    repo_root = Path(repo_root).resolve()
    artifact_io.make_dirs(output_root)
    config = artifact_io.read_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    expected_git = config["expected_git"]
    python = Path(str(config["python"])).resolve()
    git_before: dict[str, Any] = {"status": "not_run"}
    git_after: dict[str, Any] = {"status": "not_run"}
    import_audit: dict[str, Any] = {"status": "not_run"}
    execution: dict[str, Any] = {"executed": False}
    preflight_passed = True
    if execute_tests:
        git_before = audit_git_identity(
            repo_root,
            str(expected_git["branch"]),
            str(expected_git["base_commit"]),
        )
        import_audit = audit_import_origins(python, repo_root)
        preflight_passed = (
            git_before.get("status") == "passed"
            and import_audit.get("status") == "passed"
            and import_audit.get("python_version") == config.get("expected_python_version")
        )
        if preflight_passed:
            execution = _execute_baselines(
                config=config,
                output_root=output_root,
                repo_root=repo_root,
            )
            git_after = audit_git_identity(
                repo_root,
                str(expected_git["branch"]),
                str(expected_git["base_commit"]),
            )

    path_config = config["path_planner"]
    ppo_config = config["ppo_stage1"]
    path_junit_path = _resolve_junit(output_root, str(path_config["junit_xml"]))
    ppo_junit_path = _resolve_junit(output_root, str(ppo_config["junit_xml"]))
    junit_error: str | None = None
    try:
        path_junit = parse_junit(path_junit_path)
        ppo_junit = parse_junit(ppo_junit_path)
        path_skip_messages = _skipped_messages(path_junit_path)
    except (FileNotFoundError, ET.ParseError, OSError, ValueError) as exc:
        path_junit = _empty_junit()
        ppo_junit = _empty_junit()
        path_skip_messages = ()
        junit_error = str(exc)

    if execute_tests:
        path_returncode = int(execution.get("path_planner", {}).get("returncode", -1))
        ppo_returncode = int(execution.get("ppo_stage1", {}).get("returncode", -1))
    else:
        path_returncode = 0
        ppo_returncode = 1

    allowed_skip_dependency = str(path_config["allowed_skip_dependency"]).lower()
    path_skips_allowed = (
        len(path_skip_messages) == path_junit.skipped
        and all(allowed_skip_dependency in message.lower() for message in path_skip_messages)
    )
    path_passed = (
        junit_error is None
        and path_returncode == 0
        and _counts_match(path_junit, path_config["expected"])
        and path_skips_allowed
    )

    known_failure_rows = ppo_config["known_failures"]
    allowlist = tuple(str(row["nodeid"]) for row in known_failure_rows)
    allowlist_set = set(allowlist)
    observed_failure_set = set(ppo_junit.failed_nodeids)
    unexpected_failure_nodeids = sorted(observed_failure_set - allowlist_set)
    missing_failure_nodeids = sorted(allowlist_set - observed_failure_set)
    known_failure_nodeids = [nodeid for nodeid in allowlist if nodeid in observed_failure_set]
    ppo_passed = (
        junit_error is None
        and ppo_returncode == 1
        and ppo_junit.errors == 0
        and ppo_junit.skipped == 0
        and _counts_match(ppo_junit, ppo_config["expected"])
        and not unexpected_failure_nodeids
        and not missing_failure_nodeids
        and len(allowlist) == len(allowlist_set) == 13
        and _known_failure_categories_match(known_failure_rows)
    )

    configured_boundaries = config.get("boundaries")
    boundaries_passed = _boundaries_match(configured_boundaries)
    runtime_passed = (
        not execute_tests
        or (
            preflight_passed
            and execution.get("executed") is True
            and git_after.get("status") == "passed"
        )
    )
    if not boundaries_passed:
        status = "failed"
        next_required_change = "restore_gate0_safety_boundaries"
    elif not ppo_passed:
        status = "failed"
        next_required_change = "restore_exact_ppo_stage1_external_baseline"
    elif not path_passed:
        status = "failed"
        next_required_change = "restore_path_planner_v1_green_baseline"
    elif not runtime_passed:
        status = "failed"
        next_required_change = "restore_gate0_runtime_isolation"
    else:
        status = "passed"
        next_required_change = "begin_gate1_multiplatform_contract"

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": config["stage_id"],
        "status": status,
        "next_required_change": next_required_change,
        "path_planner": {
            "passed": path_junit.passed,
            "skipped": path_junit.skipped,
            "failed": path_junit.failures,
            "errors": path_junit.errors,
        },
        "ppo_stage1": {
            "passed": ppo_junit.passed,
            "skipped": ppo_junit.skipped,
            "failed": ppo_junit.failures,
            "errors": ppo_junit.errors,
            "known_failure_count": len(known_failure_nodeids),
            "known_failure_nodeids": known_failure_nodeids,
            "unexpected_failure_nodeids": unexpected_failure_nodeids,
            "missing_failure_nodeids": missing_failure_nodeids,
        },
        "junit_error": junit_error,
        "tests_executed": execution.get("executed") is True,
        "git_identity": git_after if execute_tests else {"status": "not_run"},
        "import_origins": import_audit,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    routing = {
        "schema_version": "xunce-path-v2-g0-routing/v1",
        "status": status,
        "route": next_required_change,
        "blocking_reason": None if status == "passed" else next_required_change,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    rows = [
        {
            "suite": "path_planner",
            "status": "passed" if path_passed else "failed",
            "tests": path_junit.tests,
            "passed": path_junit.passed,
            "skipped": path_junit.skipped,
            "failures": path_junit.failures,
            "errors": path_junit.errors,
            "pytest_returncode": path_returncode,
        },
        {
            "suite": "ppo_stage1_external_baseline",
            "status": "passed" if ppo_passed else "failed",
            "tests": ppo_junit.tests,
            "passed": ppo_junit.passed,
            "skipped": ppo_junit.skipped,
            "failures": ppo_junit.failures,
            "errors": ppo_junit.errors,
            "pytest_returncode": ppo_returncode,
            "known_failure_count": len(known_failure_nodeids),
            "unexpected_failure_nodeids": unexpected_failure_nodeids,
            "missing_failure_nodeids": missing_failure_nodeids,
        },
    ]
    phases = [
        {
            "phase": "runtime_identity_and_isolation",
            "status": "completed" if runtime_passed else "failed",
            "execute_tests": execute_tests,
        },
        {"phase": "baseline_junit_audit", "status": status},
        {"phase": "boundary_freeze", "status": "completed" if boundaries_passed else "failed"},
    ]
    review = {
        "schema_version": "xunce-path-v2-g0-review/v1",
        "status": status,
        "config_path": str(config_path),
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "git_before": git_before,
        "git_after": git_after,
        "import_origins": import_audit,
        "execution": execution,
        "path_planner_optional_skip_messages": list(path_skip_messages),
        "ppo_known_failure_categories": known_failure_rows,
        "configured_boundaries": configured_boundaries,
        "effective_boundaries": gate_artifacts.BOUNDARY_FIELDS,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze Path Planner v2 Gate 0 baseline and isolation evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--no-execute-tests", action="store_true")
    args = parser.parse_args(argv)
    summary = run_gate0(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        execute_tests=not args.no_execute_tests,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
