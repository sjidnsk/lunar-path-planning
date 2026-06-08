from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_controlled_hybrid_policy_training_candidate import (
    CONFIG_SCHEMA_VERSION as CONTROLLED_CONFIG_SCHEMA_VERSION,
    run_controlled_hybrid_policy_training_candidate,
)


CONFIG_SCHEMA_VERSION = "raw-policy-decision-alignment-candidate-config/v1"
SUMMARY_SCHEMA_VERSION = "raw-policy-decision-alignment-candidate-summary/v1"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an experimental raw-policy decision alignment candidate."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--raw-mining-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    raw_mining_root = _resolve_path(args.raw_mining_root, repo_root)
    output_root = _resolve_path(args.output_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    controlled_config = _controlled_config(config, raw_mining_root=raw_mining_root)
    summary_path = output_root / config["output_files"]["summary"]
    summary = run_controlled_hybrid_policy_training_candidate(
        source_root=source_root,
        output_root=output_root,
        config=controlled_config,
        repo_root=repo_root,
        summary_path=summary_path,
        validate_only=args.validate_only,
    )
    summary = _raw_summary(
        summary,
        source_root=source_root,
        raw_mining_root=raw_mining_root,
        output_root=output_root,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "raw_mining_root": _display_path(raw_mining_root, repo_root),
                "output_root": _display_path(output_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "raw_policy_regression_preference_pair_count": summary[
                    "raw_policy_regression_preference_pair_count"
                ],
                "hybrid_train_signal_count": summary["hybrid_train_signal_count"],
                "candidate_training_status": summary["candidate_training_status"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def _controlled_config(config: dict[str, Any], *, raw_mining_root: Path) -> dict[str, Any]:
    inputs = dict(config["input_files"])
    inputs["raw_policy_regression_mining_summary"] = str(
        raw_mining_root / inputs["raw_policy_regression_mining_summary"]
    )
    inputs["raw_policy_regression_preference_samples"] = str(
        raw_mining_root / inputs["raw_policy_regression_preference_samples"]
    )
    return {
        "schema_version": CONTROLLED_CONFIG_SCHEMA_VERSION,
        "description": config.get("description"),
        "input_files": inputs,
        "output_files": dict(config["output_files"]),
        "source_preconditions": dict(config["source_preconditions"]),
        "validation": dict(config["validation"]),
        "training": dict(config["training"]),
        "checkpoint": dict(config["checkpoint"]),
        "non_goals": list(config.get("non_goals", [])),
    }


def _raw_summary(
    summary: dict[str, Any],
    *,
    source_root: Path,
    raw_mining_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    payload = dict(summary)
    payload["schema_version"] = SUMMARY_SCHEMA_VERSION
    payload["raw_policy_decision_alignment_training_status"] = summary.get("candidate_training_status")
    payload["source_root"] = _display_path(source_root, repo_root)
    payload["raw_mining_root"] = _display_path(raw_mining_root, repo_root)
    payload["output_root"] = _display_path(output_root, repo_root)
    payload["experimental_checkpoint"] = True
    payload["publishes_checkpoint"] = False
    payload["replaces_default_policy"] = False
    payload["performance_claimed"] = False
    payload["formal_training_ready_claimed"] = False
    return payload


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "source_preconditions", "validation", "training", "checkpoint"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    for key in ("raw_policy_regression_mining_summary", "raw_policy_regression_preference_samples"):
        if key not in payload["input_files"]:
            raise ConfigError(f"input_files.{key} is required")
    return payload


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


if __name__ == "__main__":
    raise SystemExit(main())
