from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from platform_command import display_command


PROFILES = {"windows-non-drake", "ubuntu-non-drake", "ubuntu-drake"}
SUMMARY_SCHEMA_VERSION = "platform-validation-matrix-summary/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cross-platform validation matrix profiles.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--real-bootstrap", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    output_root = _resolve_output_root(args.output_root, args.profile, repo_root)
    commands = _commands_for_profile(
        profile=args.profile,
        repo_root=repo_root,
        skip_bootstrap=args.skip_bootstrap,
        real_bootstrap=args.real_bootstrap,
    )

    if args.dry_run:
        for command in commands:
            print(f"[DRY RUN] (cd {command['cwd']} && {display_command(command['argv'])})")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failure_reason = ""
    final_status = "passed"
    for command in commands:
        if command.get("requires_node") and _npm_executable() is None:
            final_status = "failed"
            failure_reason = "node_runtime_missing"
            results.append(_skipped_command_result(command, reason=failure_reason))
            break
        print(f"==> (cd {command['cwd']} && {display_command(command['argv'])})")
        completed = subprocess.run(command["argv"], cwd=command["cwd"], text=True)
        result = {
            "label": command["label"],
            "cwd": _display_path(Path(command["cwd"]), repo_root),
            "argv": list(command["argv"]),
            "returncode": int(completed.returncode),
        }
        results.append(result)
        if completed.returncode != 0:
            final_status = "failed"
            failure_reason = str(command.get("failure_reason") or f"{command['label']}_failed")
            if args.profile == "ubuntu-drake" and command["label"] == "drake_runtime_probe":
                _write_summary(
                    output_root=output_root,
                    repo_root=repo_root,
                    profile=args.profile,
                    bootstrap_mode=_bootstrap_mode(args.skip_bootstrap, args.real_bootstrap),
                    command_results=results,
                    final_status=final_status,
                    failure_reason="drake_runtime_missing",
                )
                return 2
            break

    _write_summary(
        output_root=output_root,
        repo_root=repo_root,
        profile=args.profile,
        bootstrap_mode=_bootstrap_mode(args.skip_bootstrap, args.real_bootstrap),
        command_results=results,
        final_status=final_status,
        failure_reason=failure_reason,
    )
    if final_status != "passed":
        return 1
    return 0


def _commands_for_profile(
    *,
    profile: str,
    repo_root: Path,
    skip_bootstrap: bool,
    real_bootstrap: bool,
) -> list[dict[str, Any]]:
    if profile == "ubuntu-drake":
        return [
            {
                "label": "drake_runtime_probe",
                "cwd": repo_root,
                "argv": [sys.executable, "-c", "import pydrake; print('pydrake ok')"],
                "failure_reason": "drake_runtime_missing",
            },
            {
                "label": "path_planner_drake_tests",
                "cwd": repo_root / "path-planner",
                "argv": [sys.executable, "-m", "pytest", "-m", "drake", "-q"],
            },
        ]

    platform_name = "windows" if profile == "windows-non-drake" else "ubuntu"
    commands: list[dict[str, Any]] = []
    if not skip_bootstrap:
        bootstrap = [
            sys.executable,
            str(repo_root / "scripts" / "bootstrap_env.py"),
            "--platform",
            platform_name,
            "--install-editable",
            "--with-training",
            "--run-validation",
        ]
        if not real_bootstrap:
            bootstrap.append("--dry-run")
        commands.append({"label": "bootstrap_env", "cwd": repo_root, "argv": bootstrap})

    commands.extend(
        [
            {
                "label": "platform_compatibility_tests",
                "cwd": repo_root,
                "argv": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_bootstrap_env.py",
                    "tests/test_bootstrap_ubuntu_conda.py",
                    "tests/test_platform_smoke.py",
                    "tests/test_no_new_python_bash_dependencies.py",
                    "tests/test_platform_validation_matrix.py",
                    "tests/test_mainline_repository_surface.py",
                    "tests/test_retired_routes_absent.py",
                    "-q",
                ],
            },
            {
                "label": "path_planner_non_drake_tests",
                "cwd": repo_root / "path-planner",
                "argv": [sys.executable, "-m", "pytest", "-m", "not drake", "-q"],
            },
            {
                "label": "dev_platform_constraints_tests",
                "cwd": repo_root / "dev-platform-constraints",
                "argv": [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            },
        ]
    )
    return commands


def _write_summary(
    *,
    output_root: Path,
    repo_root: Path,
    profile: str,
    bootstrap_mode: str,
    command_results: list[dict[str, Any]],
    final_status: str,
    failure_reason: str,
) -> None:
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "profile": profile,
        "platform": sys.platform,
        "python_executable": sys.executable,
        "bootstrap_mode": bootstrap_mode,
        "drake_validation_executed": profile == "ubuntu-drake",
        "command_results": command_results,
        "final_status": final_status,
        "failure_reason": failure_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "platform-validation-matrix-summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": final_status, "summary": str(output_root / "platform-validation-matrix-summary.json")}))


def _skipped_command_result(command: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "label": command["label"],
        "cwd": str(command["cwd"]),
        "argv": list(command["argv"]),
        "returncode": 2,
        "skipped": True,
        "reason": reason,
    }


def _bootstrap_mode(skip_bootstrap: bool, real_bootstrap: bool) -> str:
    if skip_bootstrap:
        return "skipped"
    return "real" if real_bootstrap else "dry_run"


def _resolve_output_root(value: str | None, profile: str, repo_root: Path) -> Path:
    path = Path(value) if value else Path("outputs") / "platform_validation" / profile
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
