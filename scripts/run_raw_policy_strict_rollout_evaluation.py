from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_scenario_disjoint_policy_rollout_evaluation import (
    CONFIG_SCHEMA_VERSION as SCENARIO_CONFIG_SCHEMA_VERSION,
    run_scenario_disjoint_policy_rollout_evaluation,
)


CONFIG_SCHEMA_VERSION = "raw-policy-strict-rollout-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "raw-policy-strict-rollout-evaluation-summary/v1"
NEXT_REQUIRED_CHANGE = "policy_objective_or_feature_refinement_required"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run strict raw-policy rollout evaluation for an alignment candidate."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    candidate_root = _resolve_path(args.candidate_root, repo_root)
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    output_paths = _output_paths(batch_root, config)
    scenario_config = _scenario_config(config)
    summary, decisions, regression_report = run_scenario_disjoint_policy_rollout_evaluation(
        source_root=source_root,
        candidate_root=candidate_root,
        batch_root=batch_root,
        config=scenario_config,
        repo_root=repo_root,
        output_paths=output_paths,
    )
    summary, regression_report = _strict_summary(summary, regression_report, config=config)
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "candidate_root": _display_path(candidate_root, repo_root),
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "raw_policy_regression_count": summary["raw_policy_regression_count"],
                "regression_count": summary["regression_count"],
                "summary": _display_path(output_paths["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        batch_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_paths["summary"], summary)
        _write_json(output_paths["regression_report"], regression_report)
        output_paths["decisions"].write_text(
            "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions),
            encoding="utf-8",
        )
    return 1 if summary["status"] == "failed" else 0


def _scenario_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCENARIO_CONFIG_SCHEMA_VERSION,
        "description": config.get("description"),
        "input_files": dict(config["input_files"]),
        "output_files": dict(config["output_files"]),
        "evaluation": dict(config["evaluation"]),
        "validation": dict(config["validation"]),
        "non_goals": list(config.get("non_goals", [])),
    }


def _strict_summary(
    summary: dict[str, Any],
    regression_report: dict[str, Any],
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reason_codes = list(summary.get("reason_codes") or [])
    validation = config["validation"]
    baseline_raw = _int_value(validation.get("baseline_raw_policy_regression_count"))
    raw_count = _int_value(summary.get("raw_policy_regression_count"))
    max_raw = _int_value(validation.get("max_raw_policy_regression_count"))
    if raw_count > max_raw and "raw_policy_regression_remains" not in reason_codes:
        reason_codes.append("raw_policy_regression_remains")
    delta = raw_count - baseline_raw
    status = "failed" if reason_codes else "passed"
    payload = dict(summary)
    payload.update(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": status,
            "reason_codes": reason_codes,
            "baseline_raw_policy_regression_count": baseline_raw,
            "raw_policy_regression_count_delta": delta,
            "raw_policy_alignment_improved": raw_count < baseline_raw,
            "next_required_change": NEXT_REQUIRED_CHANGE if reason_codes else None,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
    )
    report = dict(regression_report)
    report.update(
        {
            "schema_version": "raw-policy-strict-rollout-regression-report/v1",
            "status": status,
            "reason_codes": reason_codes,
            "baseline_raw_policy_regression_count": baseline_raw,
            "raw_policy_regression_count_delta": delta,
            "next_required_change": NEXT_REQUIRED_CHANGE if reason_codes else None,
        }
    )
    return payload, report


def _output_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "decisions": batch_root / outputs["decisions"],
        "regression_report": batch_root / outputs["regression_report"],
        "summary": batch_root / outputs["summary"],
    }


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
    for section in ("input_files", "output_files", "evaluation", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
