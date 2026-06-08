from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "limited-policy-training-dry-run-config/v1"
SUMMARY_SCHEMA_VERSION = "limited-policy-training-dry-run-summary/v1"
MATERIALIZATION_SCHEMA_VERSION = "planner-validated-training-input-materialization-summary/v1"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a strict limited policy training dry-run on materialized planner-validated samples."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollout-episodes")
    parser.add_argument("--materialization-summary")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    input_files = config["input_files"]
    rollout_path = (
        _resolve_path(args.rollout_episodes, repo_root)
        if args.rollout_episodes
        else batch_root / input_files["rollout_episodes"]
    )
    materialization_path = (
        _resolve_path(args.materialization_summary, repo_root)
        if args.materialization_summary
        else batch_root / input_files["materialization_summary"]
    )
    summary_path = batch_root / config["output_files"]["summary"]
    summary = run_limited_policy_training_dry_run(
        batch_root=batch_root,
        rollout_path=rollout_path,
        materialization_path=materialization_path,
        summary_path=summary_path,
        config=config,
        repo_root=repo_root,
        validate_only=args.validate_only,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "input_positive_count": summary["input_positive_count"],
                "train_policy_sample_count": summary["train_policy_sample_count"],
                "dry_run_status": summary["dry_run_status"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def run_limited_policy_training_dry_run(
    *,
    batch_root: Path,
    rollout_path: Path,
    materialization_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    repo_root: Path,
    validate_only: bool,
) -> dict[str, Any]:
    from model_explorer.policy.dataset import validate_rollout_dataset
    from model_explorer.policy.rollout_io import read_rollout_episodes
    from model_explorer.policy.training import train_policy_on_episodes

    reason_codes: list[str] = []
    materialization = _load_materialization_summary(materialization_path, reason_codes=reason_codes)
    if materialization.get("status") != "passed":
        _append_reason(reason_codes, "materialization_summary_failed")
    episodes = ()
    dataset_summary: dict[str, Any] | None = None
    training_result: dict[str, Any] | None = None
    training_error: str | None = None
    try:
        episodes = read_rollout_episodes(rollout_path)
    except Exception as exc:  # noqa: BLE001 - report script input failure in summary.
        _append_reason(reason_codes, "rollout_episodes_unreadable")
        training_error = str(exc)
    if episodes:
        try:
            dataset_summary = validate_rollout_dataset(
                episodes,
                gates={
                    "max_invalid_action_mask_count": config["validation"].get(
                        "max_invalid_action_mask_count"
                    ),
                    "max_empty_action_mask_count": config["validation"].get(
                        "max_empty_action_mask_count"
                    ),
                    "min_trainable_transition_count": config["validation"].get(
                        "expected_input_positive_count"
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "rollout_dataset_validation_failed")
            training_error = str(exc)
    input_positive_count = _int_value(materialization.get("input_positive_count"))
    expected_count = config["validation"].get("expected_input_positive_count")
    if expected_count is not None and input_positive_count != _int_value(expected_count):
        _append_reason(reason_codes, "input_positive_count_mismatch")
    invalid_action_mask_count = _int_value(materialization.get("invalid_action_mask_count"))
    empty_action_mask_count = _int_value(materialization.get("empty_action_mask_count"))
    if invalid_action_mask_count > _int_value(config["validation"].get("max_invalid_action_mask_count")):
        _append_reason(reason_codes, "invalid_action_mask")
    if empty_action_mask_count > _int_value(config["validation"].get("max_empty_action_mask_count")):
        _append_reason(reason_codes, "empty_action_mask")
    train_policy_sample_count = 0
    if not reason_codes and not validate_only:
        try:
            checkpoint_path = _checkpoint_path(config, batch_root)
            training_result = train_policy_on_episodes(
                episodes,
                checkpoint_path=checkpoint_path,
                seed=_int_value(config["training"].get("seed")),
                hidden_size=_int_value(config["training"].get("hidden_size")),
                learning_rate=float(config["training"].get("learning_rate", 1.0e-3)),
                epochs=_int_value(config["training"].get("epochs")),
            )
            train_policy_sample_count = _int_value(training_result.get("sample_count"))
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "training_dry_run_failed")
            training_error = str(exc)
    elif dataset_summary is not None:
        train_policy_sample_count = _int_value(dataset_summary.get("trainable_transition_count"))

    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dry_run_status": "passed" if status == "passed" else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": {
            "materialization_summary": {
                "path": _display_path(materialization_path, repo_root),
                "exists": materialization_path.is_file(),
                "schema_version": materialization.get("schema_version"),
                "status": materialization.get("status"),
            },
            "rollout_episodes": {
                "path": _display_path(rollout_path, repo_root),
                "exists": rollout_path.is_file(),
            },
        },
        "input_positive_count": input_positive_count,
        "default_contract_positive_count": _int_value(
            materialization.get("default_contract_positive_count")
        ),
        "planner_validated_exception_positive_count": _int_value(
            materialization.get("planner_validated_exception_positive_count")
        ),
        "excluded_nontrainable_count": _int_value(materialization.get("excluded_nontrainable_count")),
        "invalid_action_mask_count": invalid_action_mask_count,
        "empty_action_mask_count": empty_action_mask_count,
        "train_policy_sample_count": train_policy_sample_count,
        "dataset_summary": dataset_summary,
        "training_result": training_result,
        "training_error": training_error,
        "rollout_episodes": _display_path(rollout_path, repo_root),
        "summary": _display_path(summary_path, repo_root),
        "runs_large_scale_training": False,
        "dry_run_only": True,
        "publishes_checkpoint": _checkpoint_path(config, batch_root) is not None,
        "performance_claimed": False,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _load_materialization_summary(path: Path, *, reason_codes: list[str]) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, "materialization_summary_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, "materialization_summary_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, "materialization_summary_invalid_json_root")
        return {}
    if payload.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        _append_reason(reason_codes, "materialization_summary_schema_version_mismatch")
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
    for section in ("input_files", "output_files", "validation", "training"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    if not payload["input_files"].get("rollout_episodes"):
        raise ConfigError("input_files.rollout_episodes is required")
    if not payload["input_files"].get("materialization_summary"):
        raise ConfigError("input_files.materialization_summary is required")
    if not payload["output_files"].get("summary"):
        raise ConfigError("output_files.summary is required")
    return payload


def _checkpoint_path(config: dict[str, Any], batch_root: Path) -> Path | None:
    value = config["training"].get("checkpoint_path")
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else batch_root / path


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
