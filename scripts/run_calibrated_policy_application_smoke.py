from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "calibrated-policy-application-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "calibrated-policy-application-smoke-summary/v1"
CALIBRATION_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
COVERAGE_SCHEMA_VERSION = "channel-aware-contrast-coverage-summary/v1"
READINESS_SCHEMA_VERSION = "channel-aware-training-readiness-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
READY_COVERAGE_ACTION = "ready_for_calibrated_policy_application_smoke"
READY_READINESS_STATUS = "ready_for_calibrated_policy_application_smoke"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Calibrated Policy Application Smoke v1 to channel-aware calibration evidence."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing channel-aware summaries.")
    parser.add_argument(
        "--selection-contrast-calibration-summary",
        help="channel-aware-selection-contrast-calibration-summary/v1 JSON. Defaults to <batch-root>/channel-aware-selection-contrast-calibration-summary.json.",
    )
    parser.add_argument(
        "--contrast-coverage-summary",
        help="channel-aware-contrast-coverage-summary/v1 JSON. Defaults to <batch-root>/channel-aware-contrast-coverage-summary.json.",
    )
    parser.add_argument(
        "--readiness-summary",
        help="channel-aware-training-readiness-summary/v1 JSON. Defaults to <batch-root>/channel-aware-training-readiness-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Calibrated policy application smoke config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    calibration_path = (
        _resolve_path(args.selection_contrast_calibration_summary, repo_root)
        if args.selection_contrast_calibration_summary
        else batch_root / "channel-aware-selection-contrast-calibration-summary.json"
    )
    coverage_path = (
        _resolve_path(args.contrast_coverage_summary, repo_root)
        if args.contrast_coverage_summary
        else batch_root / "channel-aware-contrast-coverage-summary.json"
    )
    readiness_path = (
        _resolve_path(args.readiness_summary, repo_root)
        if args.readiness_summary
        else batch_root / "channel-aware-training-readiness-summary.json"
    )
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_calibrated_application_smoke(
        batch_root=batch_root,
        calibration_path=calibration_path,
        coverage_path=coverage_path,
        readiness_path=readiness_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "selection_contrast_calibration_summary": _display_path(calibration_path, repo_root),
                "contrast_coverage_summary": _display_path(coverage_path, repo_root),
                "readiness_summary": _display_path(readiness_path, repo_root),
                "config": _display_path(config_path, repo_root),
                "reason_codes": summary["reason_codes"],
                "source_selected_candidate_changed_rate": summary["source_selected_candidate_changed_rate"],
                "calibrated_selected_candidate_changed_rate": summary[
                    "calibrated_selected_candidate_changed_rate"
                ],
                "applied_calibrated_candidate_count": summary["applied_calibrated_candidate_count"],
                "recommended_next_action": summary["recommended_next_action"],
                "calibrated_policy_application_smoke_summary": _display_path(output_file, repo_root),
            },
            ensure_ascii=False,
        )
    )

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "calibrated_policy_application_smoke_summary": _display_path(output_file, repo_root),
                        },
                        "recommended_next_action": summary["recommended_next_action"],
                    },
                    ensure_ascii=False,
                )
            )
        return 1 if summary["status"] == "failed" else 0

    _write_json(output_file, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "calibrated_policy_application_smoke_summary": _display_path(output_file, repo_root),
                "recommended_next_action": summary["recommended_next_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_calibrated_application_smoke(
    *,
    batch_root: Path,
    calibration_path: Path,
    coverage_path: Path,
    readiness_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    calibration = _load_source(
        calibration_path,
        label="channel_aware_selection_contrast_calibration_summary",
        expected_schema=CALIBRATION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    coverage = _load_source(
        coverage_path,
        label="channel_aware_contrast_coverage_summary",
        expected_schema=COVERAGE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    readiness = _load_source(
        readiness_path,
        label="channel_aware_training_readiness_summary",
        expected_schema=READINESS_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    if _fail_on_input_failure(config):
        for label, payload in (
            ("channel_aware_selection_contrast_calibration_summary", calibration),
            ("channel_aware_contrast_coverage_summary", coverage),
            ("channel_aware_training_readiness_summary", readiness),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = [
        _inspect_git(
            calibration,
            label="channel_aware_selection_contrast_calibration_summary",
            current_git=current_git,
            config=config,
            reason_codes=reason_codes,
        ),
        _inspect_git(
            coverage,
            label="channel_aware_contrast_coverage_summary",
            current_git=current_git,
            config=config,
            reason_codes=reason_codes,
        ),
        _inspect_git(
            readiness,
            label="channel_aware_training_readiness_summary",
            current_git=current_git,
            config=config,
            reason_codes=reason_codes,
        ),
    ]

    application = _application_metrics(
        calibration=calibration,
        coverage=coverage,
        readiness=readiness,
        config=config,
    )
    status = "failed" if reason_codes else "passed"
    failure_reason_counts = Counter(reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "application_scope": "calibrated_policy_application_smoke_audit_only",
        "quality_signal_use": "calibrated_policy_target_application_audit_only",
        "audit_only": True,
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "selection_contrast_calibration": _public_git(calibration),
            "contrast_coverage": _public_git(coverage),
            "training_readiness": _public_git(readiness),
            "current_matches_sources": all(source_git_matches),
        },
        **application,
        "runs_training": False,
        "no_ppo_training": True,
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "channel_aware_backend_opt_in": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_path_planner_route_contract": True,
        "does_not_modify_model_explorer_contract": True,
        "does_not_modify_path_planner_sidecar_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _application_metrics(
    *,
    calibration: dict[str, Any],
    coverage: dict[str, Any],
    readiness: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    records = [
        record
        for record in calibration.get("calibrated_selection_records", [])
        if isinstance(record, dict)
    ]
    applied_records = [record for record in records if _is_applied_calibrated_record(record)]
    source_rate = _source_rate(calibration, coverage)
    calibrated_rate = _calibrated_rate(calibration, coverage)
    safety_regression_count = max(
        _int_value_or_default(calibration.get("safety_regression_count"), 0),
        _int_value_or_default(coverage.get("safety_regression_count"), 0),
        _int_value_or_default(readiness.get("calibration_safety_regression_count"), 0),
    )
    changed_scenario_ids = _unique(
        _string_list(coverage.get("changed_scenario_ids"))
        or _string_list(calibration.get("changed_scenario_ids"))
    )
    gate_reason_codes = _application_gate_reason_codes(
        source_rate=source_rate,
        calibrated_rate=calibrated_rate,
        applied_count=len(applied_records),
        safety_regression_count=safety_regression_count,
        coverage=coverage,
        readiness=readiness,
        config=config,
    )
    recommended = (
        "ready_for_policy_training_readiness_review"
        if not gate_reason_codes
        else "needs_application_gate_refinement"
    )
    return {
        "source_selected_candidate_changed_rate": source_rate,
        "calibrated_selected_candidate_changed_rate": calibrated_rate,
        "calibrated_selection_rate_delta": calibrated_rate - source_rate,
        "applied_calibrated_candidate_count": len(applied_records),
        "applied_calibrated_records": [_public_applied_record(record) for record in applied_records],
        "changed_scenario_ids": changed_scenario_ids,
        "rejected_goal_blocked_count": _int_value_or_default(calibration.get("goal_blocked_count"), 0),
        "platform_goal_contract_mismatch_count": _int_value_or_default(
            calibration.get("platform_goal_contract_mismatch_count"),
            0,
        ),
        "platform_goal_anchor_available_count": _int_value_or_default(
            calibration.get("platform_goal_anchor_available_count"),
            0,
        ),
        "platform_goal_unresolved_count": _int_value_or_default(
            calibration.get("platform_goal_unresolved_count"),
            0,
        ),
        "platform_goal_feasibility_class_counts": _dict_value(
            calibration.get("platform_goal_feasibility_class_counts")
        ),
        "safety_regression_count": safety_regression_count,
        "application_gate_reason_codes": gate_reason_codes,
        "recommended_next_action": recommended,
        "source_policy_target_selection_improvement_claimed": bool(
            calibration.get("source_supports_policy_target_selection_improvement_claim", False)
        ),
        "coverage_recommended_next_action": str(coverage.get("recommended_next_action", "")),
        "readiness_status": str(readiness.get("readiness_status", "")),
        "calibrated_readiness_status": str(readiness.get("calibrated_readiness_status", "")),
    }


def _is_applied_calibrated_record(record: dict[str, Any]) -> bool:
    if record.get("selection_reason") != "channel_quality_contrast_selected":
        return False
    if record.get("recommendation") != "keep":
        return False
    if bool(record.get("safety_regression", False)):
        return False
    if not bool(record.get("selected_candidate_changed", False)):
        return False
    source_cell = _list_value(record.get("astar_selected_cell"))
    calibrated_cell = _list_value(record.get("calibrated_channel_aware_selected_cell"))
    return bool(source_cell and calibrated_cell and source_cell != calibrated_cell)


def _application_gate_reason_codes(
    *,
    source_rate: float,
    calibrated_rate: float,
    applied_count: int,
    safety_regression_count: int,
    coverage: dict[str, Any],
    readiness: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    thresholds = config["application_thresholds"]
    reason_codes: list[str] = []
    if coverage.get("recommended_next_action") != READY_COVERAGE_ACTION:
        _append_reason(reason_codes, "contrast_coverage_not_ready_for_application_smoke")
    if readiness.get("calibrated_readiness_status") != READY_READINESS_STATUS:
        _append_reason(reason_codes, "readiness_not_ready_for_calibrated_application_smoke")
    if applied_count < thresholds["min_applied_calibrated_candidate_count"]:
        _append_reason(reason_codes, "applied_calibrated_candidate_count_below_threshold")
    if calibrated_rate - source_rate < thresholds["min_calibrated_selection_rate_delta"]:
        _append_reason(reason_codes, "calibrated_selection_rate_delta_below_threshold")
    if safety_regression_count > thresholds["max_safety_regression_count"]:
        _append_reason(reason_codes, "safety_regression_blocks_policy_training")
    return reason_codes


def _public_applied_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_key": record.get("pair_key"),
        "scenario_id": record.get("scenario_id"),
        "source_selected_cell": _list_value(record.get("astar_selected_cell")),
        "calibrated_selected_cell": _list_value(record.get("calibrated_channel_aware_selected_cell")),
        "selected_action_index": record.get("selected_action_index"),
        "selected_candidate_score": record.get("selected_candidate_score"),
        "path_cost_tradeoff": bool(record.get("path_cost_tradeoff", False)),
    }


def _source_rate(calibration: dict[str, Any], coverage: dict[str, Any]) -> float:
    if coverage.get("source_selected_candidate_changed_rate") is not None:
        return _float_value_or_default(coverage.get("source_selected_candidate_changed_rate"), 0.0)
    return _float_value_or_default(calibration.get("source_selected_candidate_changed_rate"), 0.0)


def _calibrated_rate(calibration: dict[str, Any], coverage: dict[str, Any]) -> float:
    if coverage.get("calibrated_selected_candidate_changed_rate") is not None:
        return _float_value_or_default(coverage.get("calibrated_selected_candidate_changed_rate"), 0.0)
    return _float_value_or_default(calibration.get("selected_candidate_changed_rate"), 0.0)


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
    output_files = payload.get("output_files")
    if not isinstance(output_files, dict) or not _nonempty_string(
        output_files.get("calibrated_policy_application_smoke_summary")
    ):
        raise ConfigError("output_files.calibrated_policy_application_smoke_summary must be a non-empty string")
    thresholds = payload.get("application_thresholds")
    if not isinstance(thresholds, dict):
        raise ConfigError("application_thresholds must be an object")
    normalized_thresholds = {
        "min_applied_calibrated_candidate_count": _int_value(
            thresholds.get("min_applied_calibrated_candidate_count", 1),
            "application_thresholds.min_applied_calibrated_candidate_count",
        ),
        "min_calibrated_selection_rate_delta": _float_value(
            thresholds.get("min_calibrated_selection_rate_delta", 0.01),
            "application_thresholds.min_calibrated_selection_rate_delta",
        ),
        "max_safety_regression_count": _int_value(
            thresholds.get("max_safety_regression_count", 0),
            "application_thresholds.max_safety_regression_count",
        ),
    }
    if normalized_thresholds["min_applied_calibrated_candidate_count"] < 0:
        raise ConfigError("application_thresholds.min_applied_calibrated_candidate_count must be >= 0")
    if normalized_thresholds["min_calibrated_selection_rate_delta"] < 0.0:
        raise ConfigError("application_thresholds.min_calibrated_selection_rate_delta must be >= 0")
    if normalized_thresholds["max_safety_regression_count"] < 0:
        raise ConfigError("application_thresholds.max_safety_regression_count must be >= 0")
    config = dict(payload)
    config["application_thresholds"] = normalized_thresholds
    return config


def _load_source(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _display_path(path, repo_root), "exists": path.is_file()}
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        source_summaries[label] = record
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        source_summaries[label] = record
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_not_object")
        source_summaries[label] = record
        return {}
    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "status": payload.get("status"),
            "reason_codes": _string_list(payload.get("reason_codes")),
        }
    )
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    source_summaries[label] = record
    return payload


def _inspect_git(
    payload: dict[str, Any],
    *,
    label: str,
    current_git: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> bool:
    if not _require_current_git_match(config):
        return True
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
    if source_current and not _git_snapshots_match(source_current, current_git):
        _append_reason(reason_codes, "current_git_provenance_mismatch")
        _append_reason(reason_codes, f"{label}_current_git_provenance_mismatch")
        return False
    return bool(source_current)


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["calibrated_policy_application_smoke_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "application_thresholds": dict(config.get("application_thresholds", {})),
        "output_files": dict(config.get("output_files", {})),
    }


def _public_git(payload: dict[str, Any]) -> dict[str, Any]:
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    return dict(git)


def _require_current_git_match(config: dict[str, Any]) -> bool:
    validation = config.get("validation")
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("require_current_git_match", True))


def _fail_on_input_failure(config: dict[str, Any]) -> bool:
    validation = config.get("validation")
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("fail_on_input_failure", True))


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    def git(path: Path, *args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    return {
        "parent": {
            "path": ".",
            "sha": git(repo_root, "rev-parse", "HEAD") or "unknown",
            "branch": git(repo_root, "branch", "--show-current"),
        },
        "submodules": {
            name: {
                "path": name,
                "sha": git(repo_root / name, "rev-parse", "HEAD") or "unknown",
                "branch": git(repo_root / name, "branch", "--show-current"),
            }
            for name in SUBMODULES
        },
    }


def _git_snapshots_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left or not right:
        return False
    left_parent = left.get("parent") if isinstance(left.get("parent"), dict) else {}
    right_parent = right.get("parent") if isinstance(right.get("parent"), dict) else {}
    if left_parent.get("sha") != right_parent.get("sha"):
        return False
    left_modules = left.get("submodules") if isinstance(left.get("submodules"), dict) else {}
    right_modules = right.get("submodules") if isinstance(right.get("submodules"), dict) else {}
    for name in SUBMODULES:
        left_module = left_modules.get(name) if isinstance(left_modules.get(name), dict) else {}
        right_module = right_modules.get(name) if isinstance(right_modules.get(name), dict) else {}
        if left_module.get("sha") != right_module.get("sha"):
            return False
    return True


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict_value(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        result[str(key)] = _int_value_or_default(count, 0)
    return result


def _unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer") from exc


def _float_value(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number") from exc


def _int_value_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
