from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from platform_command import display_command


PROFILES = {"windows-non-drake", "ubuntu-non-drake", "ubuntu-drake"}
PARENT_NON_DRAKE_SMOKE_TESTS = (
    "tests/ppo_highres_frontier/test_stage6_standard_config.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run platform smoke validation profiles.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if args.profile == "ubuntu-drake":
        probe = [sys.executable, "-c", "import pydrake; print('pydrake ok')"]
        if _run(probe, cwd=repo_root, dry_run=args.dry_run) != 0:
            return 2
        return _run(
            [sys.executable, "-m", "pytest", "-m", "drake", "-q"],
            cwd=repo_root / "path-planner",
            dry_run=args.dry_run,
        )

    commands = [
        [sys.executable, "-m", "pytest", *PARENT_NON_DRAKE_SMOKE_TESTS, "-q"],
        [sys.executable, "-m", "pytest", "-m", "not drake", "-q"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
    ]
    cwd_by_index = [
        repo_root,
        repo_root / "path-planner",
        repo_root / "dev-platform-constraints",
    ]
    for command, cwd in zip(commands, cwd_by_index, strict=True):
        code = _run(command, cwd=cwd, dry_run=args.dry_run)
        if code != 0:
            return code
    return 0


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> int:
    if dry_run:
        print(f"[DRY RUN] (cd {cwd} && {display_command(command)})")
        return 0
    print(f"==> (cd {cwd} && {display_command(command)})")
    completed = subprocess.run(command, cwd=cwd, text=True)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
