from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from platform_command import display_command, python_script_command


REGISTRY_SCHEMA_VERSION = "lunar-stage-registry/v1"
DEFAULT_REGISTRY = "configs/stage_registry.json"


class StageRegistryError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run registered lunar-path-planning stages cross-platform.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="Stage registry JSON path.")
    parser.add_argument("--list", action="store_true", help="List available stages.")
    parser.add_argument("--stage", help="Stage id to run.")
    parser.add_argument("--config", help="Override stage config path.")
    parser.add_argument("--output-root", help="Override stage output root.")
    parser.add_argument("--repo-root", help="Repository root override. Defaults to parent of scripts/.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument appended to the stage argv.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    registry_path = _resolve_path(args.registry, repo_root)
    try:
        registry = _load_registry(registry_path)
    except StageRegistryError as exc:
        print(f"stage registry error: {exc}", file=sys.stderr)
        return 2

    stages = registry["stages"]
    if args.list:
        for stage_id in sorted(stages):
            print(stage_id)
        return 0

    if not args.stage:
        print("--stage is required unless --list is used", file=sys.stderr)
        return 2
    if args.stage not in stages:
        available = ", ".join(sorted(stages))
        print(f"unknown stage: {args.stage}. Available stages: {available}", file=sys.stderr)
        return 2

    try:
        command = _build_command(
            stages[args.stage],
            repo_root=repo_root,
            config_override=args.config,
            output_root_override=args.output_root,
            extra_args=args.extra_arg,
        )
    except StageRegistryError as exc:
        print(f"stage registry error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[DRY RUN] (cd {repo_root} && {display_command(command)})")
        return 0

    completed = subprocess.run(command, cwd=repo_root, text=True)
    return int(completed.returncode)


def _load_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StageRegistryError(f"registry file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StageRegistryError(f"registry JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageRegistryError("registry root must be an object")
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise StageRegistryError(f"schema_version must be {REGISTRY_SCHEMA_VERSION!r}")
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        raise StageRegistryError("stages must be an object")
    for stage_id, stage in stages.items():
        if not isinstance(stage_id, str) or not stage_id:
            raise StageRegistryError("stage ids must be non-empty strings")
        if not isinstance(stage, dict):
            raise StageRegistryError(f"{stage_id}: stage must be an object")
    return payload


def _build_command(
    stage: dict[str, Any],
    *,
    repo_root: Path,
    config_override: str | None,
    output_root_override: str | None,
    extra_args: list[str],
) -> list[str]:
    script_value = _require_string(stage, "script")
    script_path = _resolve_path(script_value, repo_root)
    if not script_path.is_file():
        raise StageRegistryError(f"script does not exist: {script_path}")

    raw_args = stage.get("args", [])
    if not isinstance(raw_args, list) or any(not isinstance(item, str) for item in raw_args):
        raise StageRegistryError(f"{script_value}: args must be an array of strings")
    replacement_keys = set()
    for item in raw_args:
        if "{config}" in item:
            replacement_keys.add("config")
        if "{output_root}" in item:
            replacement_keys.add("output_root")
        if "{repo_root}" in item:
            replacement_keys.add("repo_root")

    replacements = {"repo_root": str(repo_root)}
    if "config" in replacement_keys:
        config = config_override if config_override is not None else _require_string(stage, "default_config")
        replacements["config"] = str(_resolve_path(config, repo_root))
    if "output_root" in replacement_keys:
        output_root = (
            output_root_override
            if output_root_override is not None
            else _require_string(stage, "default_output_root")
        )
        replacements["output_root"] = str(_resolve_path(output_root, repo_root))
    rendered_args = [item.format(**replacements) for item in raw_args]
    return python_script_command(script_path, *rendered_args, *extra_args)


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise StageRegistryError(f"{field} must be a non-empty string")
    return value


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
