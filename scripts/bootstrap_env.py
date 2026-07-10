from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from platform_command import display_command


MODULES = ("path-planner", "model-explorer", "dev-platform-constraints", "visual-workbench")
PYTHON_SPEC = "python=3.12"
WINDOWS_DEFAULT_ENV_PREFIX = r"D:\conda_envs\lunar-explorer"
WINDOWS_DEFAULT_DOWNLOAD_ROOT = r"D:\CodexDownloads\lunar-path-planning"
UBUNTU_DEFAULT_ENV_NAME = "lunar-explorer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or update the lunar-path-planning Conda runtime.")
    parser.add_argument("--platform", choices=("auto", "windows", "ubuntu"), default="auto")
    parser.add_argument("--conda", default="conda")
    parser.add_argument("--env-name")
    parser.add_argument("--env-prefix")
    parser.add_argument("--download-root")
    parser.add_argument("--install-editable", action="store_true")
    parser.add_argument("--with-training", action="store_true")
    parser.add_argument("--with-visual-workbench", action="store_true")
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument("--skip-submodules", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    platform_name = _resolve_platform(args.platform)
    env_target = _env_target(platform_name, args.env_name, args.env_prefix)
    environment_file = repo_root / ("environment.windows.yml" if platform_name == "windows" else "environment.yml")
    download_root = args.download_root or (
        WINDOWS_DEFAULT_DOWNLOAD_ROOT if platform_name == "windows" else str(Path.home() / ".cache" / "lunar-path-planning")
    )

    print(f"Repository: {repo_root}")
    print(f"Platform: {platform_name}")
    print(f"Conda executable: {args.conda}")
    print(f"Conda environment target: {' '.join(env_target)}")
    print(f"Environment file: {environment_file}")
    print(f"Download root: {download_root}")
    print(f"Python spec: {PYTHON_SPEC}")

    env = os.environ.copy()
    env.setdefault("LUNAR_DOWNLOAD_ROOT", download_root)
    env.setdefault("LUNAR_DATA_ROOT", download_root)

    if not args.skip_submodules:
        _run(
            [*["git", "-C", str(repo_root), "submodule", "update", "--init", "--recursive"], *MODULES],
            dry_run=args.dry_run,
            env=env,
        )

    exists_cmd = [args.conda, "run", *env_target, "python", "--version"]
    env_exists = False if args.dry_run else subprocess.run(exists_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    env_cmd = (
        [args.conda, "env", "update", *env_target, "-f", str(environment_file), "--prune"]
        if env_exists
        else [args.conda, "env", "create", *env_target, "-f", str(environment_file)]
    )
    _run(env_cmd, dry_run=args.dry_run, env=env)
    _run([args.conda, "install", *env_target, "-c", "conda-forge", PYTHON_SPEC, "--yes"], dry_run=args.dry_run, env=env)
    _run(
        [
            args.conda,
            "run",
            *env_target,
            "python",
            "-c",
            "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)",
        ],
        dry_run=args.dry_run,
        env=env,
    )

    if args.install_editable:
        editable_specs = [
            str(repo_root),
            str(repo_root / "path-planner"),
            str(repo_root / "dev-platform-constraints"),
            str(repo_root / "model-explorer") + ("[training]" if args.with_training else ""),
        ]
        if args.with_visual_workbench:
            editable_specs.append(str(repo_root / "visual-workbench"))
        editable_args = [item for spec in editable_specs for item in ("-e", spec)]
        _run([args.conda, "run", *env_target, "python", "-m", "pip", "install", *editable_args], dry_run=args.dry_run, env=env)
    elif args.with_training:
        _run([args.conda, "run", *env_target, "python", "-m", "pip", "install", "torch>=2.0"], dry_run=args.dry_run, env=env)
    if args.with_visual_workbench and not args.install_editable:
        _run([args.conda, "run", *env_target, "python", "-m", "pip", "install", "-e", str(repo_root / "visual-workbench")], dry_run=args.dry_run, env=env)

    _run_import_smoke(
        args.conda,
        env_target,
        repo_root,
        include_root=args.install_editable,
        dry_run=args.dry_run,
        env=env,
    )
    if args.run_validation:
        _run_validation(args.conda, env_target, repo_root, platform_name=platform_name, dry_run=args.dry_run, env=env)

    return 0


def _resolve_platform(value: str) -> str:
    if value != "auto":
        return value
    return "windows" if sys.platform.startswith("win") else "ubuntu"


def _env_target(platform_name: str, env_name: str | None, env_prefix: str | None) -> list[str]:
    if env_prefix:
        return ["-p", env_prefix]
    if env_name:
        return ["-n", env_name]
    if platform_name == "windows":
        return ["-p", WINDOWS_DEFAULT_ENV_PREFIX]
    return ["-n", UBUNTU_DEFAULT_ENV_NAME]


def _run(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    if dry_run:
        print(f"[DRY RUN] {display_command(command)}")
        return
    print(f"==> {display_command(command)}")
    subprocess.run(command, check=True, env=env)


def _run_import_smoke(
    conda: str,
    env_target: list[str],
    repo_root: Path,
    *,
    include_root: bool,
    dry_run: bool,
    env: dict[str, str],
) -> None:
    smokes = [
        ("path-planner", "path_planner"),
        ("model-explorer", "model_explorer"),
        ("dev-platform-constraints", "dev_platform_constraints"),
    ]
    if include_root:
        _run(
            [conda, "run", *env_target, "python", "-c", "import lunar_exploration_ppo; print('lunar_exploration_ppo import ok')"],
            dry_run=dry_run,
            env=env,
        )
    for module_dir, import_name in smokes:
        source_path = repo_root / module_dir / "src"
        smoke_env = dict(env)
        smoke_env["PYTHONPATH"] = str(source_path)
        _run(
            [conda, "run", *env_target, "python", "-c", f"import {import_name}; print('{import_name} import ok')"],
            dry_run=dry_run,
            env=smoke_env,
        )


def _run_validation(
    conda: str,
    env_target: list[str],
    repo_root: Path,
    *,
    platform_name: str,
    dry_run: bool,
    env: dict[str, str],
) -> None:
    profiles = {
        "windows": "windows-non-drake",
        "ubuntu": "ubuntu-non-drake",
    }
    print(f"Validation profile: {profiles[platform_name]} pytest non-drake smoke")
    _run(
        [
            conda,
            "run",
            *env_target,
            "python",
            str(repo_root / "scripts" / "run_platform_smoke.py"),
            "--profile",
            profiles[platform_name],
        ],
        dry_run=dry_run,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
