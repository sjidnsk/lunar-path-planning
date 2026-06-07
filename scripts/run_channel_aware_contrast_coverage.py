from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _shared_git_snapshot
from git_provenance import git_snapshots_match as _shared_git_snapshots_match
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _shared_public_git


CONFIG_SCHEMA_VERSION = "channel-aware-contrast-coverage-config/v1"
SUMMARY_SCHEMA_VERSION = "channel-aware-contrast-coverage-summary/v1"
CALIBRATION_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize channel-aware contrast scenario coverage from calibration evidence."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing calibration summary.")
    parser.add_argument(
        "--selection-contrast-calibration-summary",
        help="channel-aware-selection-contrast-calibration-summary/v1 JSON. Defaults to <batch-root>/channel-aware-selection-contrast-calibration-summary.json.",
    )
    parser.add_argument("--config", required=True, help="Channel-aware contrast coverage config JSON.")
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
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_contrast_coverage(
        batch_root=batch_root,
        calibration_path=calibration_path,
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
                "config": _display_path(config_path, repo_root),
                "reason_codes": summary["reason_codes"],
                "scenario_count": summary["scenario_count"],
                "calibrated_selected_candidate_changed_rate": summary[
                    "calibrated_selected_candidate_changed_rate"
                ],
                "changed_scenario_ids": summary["changed_scenario_ids"],
                "recommended_next_action": summary["recommended_next_action"],
                "contrast_coverage_summary": _display_path(output_file, repo_root),
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
                            "contrast_coverage_summary": _display_path(output_file, repo_root),
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
                "contrast_coverage_summary": _display_path(output_file, repo_root),
                "recommended_next_action": summary["recommended_next_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_contrast_coverage(
    *,
    batch_root: Path,
    calibration_path: Path,
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
    if _fail_on_input_failure(config) and calibration.get("status") == "failed":
        _append_reason(reason_codes, "channel_aware_selection_contrast_calibration_summary_failed")

    current_git = _git_snapshot(repo_root)
    _inspect_git(calibration, current_git=current_git, config=config, reason_codes=reason_codes)
    coverage = _coverage_metrics(calibration, config=config)
    status = "failed" if reason_codes else "passed"
    failure_reason_counts = Counter(reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "selection_contrast_calibration_summary_path": _display_path(calibration_path, repo_root),
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "selection_contrast_calibration": _public_git(calibration),
        },
        **coverage,
        "runs_training": False,
        "channel_aware_backend_opt_in": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_path_planner_route_contract": True,
        "does_not_modify_model_explorer_contract": True,
        "does_not_modify_path_planner_sidecar_contract": True,
        "no_gcs_control_point_candidate_as_default_execution_trajectory": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "policy_target_selection_improvement_claimed": False,
        "non_goals": list(config.get("non_goals", [])),
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
    output_files = payload.get("output_files")
    if not isinstance(output_files, dict) or not _nonempty_string(output_files.get("contrast_coverage_summary")):
        raise ConfigError("output_files.contrast_coverage_summary must be a non-empty string")
    thresholds = payload.get("coverage_thresholds")
    if not isinstance(thresholds, dict):
        raise ConfigError("coverage_thresholds must be an object")
    normalized_thresholds = {
        "min_changed_scenario_count": _int_value(
            thresholds.get("min_changed_scenario_count", 4),
            "coverage_thresholds.min_changed_scenario_count",
        ),
        "min_calibrated_selected_candidate_changed_rate": _float_value(
            thresholds.get("min_calibrated_selected_candidate_changed_rate", 0.3),
            "coverage_thresholds.min_calibrated_selected_candidate_changed_rate",
        ),
        "max_blocked_candidate_rate": _float_value(
            thresholds.get("max_blocked_candidate_rate", 0.65),
            "coverage_thresholds.max_blocked_candidate_rate",
        ),
    }
    if normalized_thresholds["min_changed_scenario_count"] < 0:
        raise ConfigError("coverage_thresholds.min_changed_scenario_count must be >= 0")
    for key in ("min_calibrated_selected_candidate_changed_rate", "max_blocked_candidate_rate"):
        if normalized_thresholds[key] < 0.0 or normalized_thresholds[key] > 1.0:
            raise ConfigError(f"coverage_thresholds.{key} must be between 0 and 1")
    config = dict(payload)
    config["coverage_thresholds"] = normalized_thresholds
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
            "reason_codes": _string_list(payload.get("reason_codes", [])),
        }
    )
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    source_summaries[label] = record
    return payload


def _inspect_git(
    payload: dict[str, Any],
    *,
    current_git: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    _inspect_source_git_provenance(
        payload,
        label="channel_aware_selection_contrast_calibration_summary",
        current_git=current_git,
        require_current_git_match=_require_current_git_match(config),
        reason_codes=reason_codes,
        submodules=SUBMODULES,
    )


def _coverage_metrics(payload: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    records = [
        record
        for record in payload.get("calibrated_selection_records", [])
        if isinstance(record, dict)
    ]
    scenario_ids = _unique(str(record.get("scenario_id", "")) for record in records if record.get("scenario_id"))
    changed_scenario_ids = _unique(_string_list(payload.get("changed_scenario_ids", [])))
    eligible_records = [
        record
        for record in records
        if record.get("selection_reason") == "channel_quality_contrast_selected"
        or record.get("selected_candidate_score") is not None
    ]
    safety_regression_count = _int_value_or_default(payload.get("safety_regression_count"), 0)
    blocked_candidate_rate = _float_value_or_default(payload.get("blocked_candidate_rate"), 0.0)
    calibrated_rate = _float_value_or_default(payload.get("selected_candidate_changed_rate"), 0.0)
    thresholds = config["coverage_thresholds"]
    reason_codes: list[str] = []
    if len(changed_scenario_ids) < thresholds["min_changed_scenario_count"]:
        _append_reason(reason_codes, "changed_scenario_count_below_threshold")
    if calibrated_rate < thresholds["min_calibrated_selected_candidate_changed_rate"]:
        _append_reason(reason_codes, "calibrated_selected_candidate_changed_rate_below_threshold")
    if blocked_candidate_rate > thresholds["max_blocked_candidate_rate"]:
        _append_reason(reason_codes, "blocked_candidate_rate_high")
    if safety_regression_count > 0:
        _append_reason(reason_codes, "safety_regression_blocks_policy_smoke")
    recommended = (
        "ready_for_calibrated_policy_application_smoke"
        if not reason_codes
        else "needs_more_contrast_scenarios"
    )
    return {
        "coverage_reason_codes": reason_codes,
        "scenario_count": len(scenario_ids),
        "selection_context_count": len(records),
        "contrast_eligible_context_count": len(eligible_records),
        "source_selected_candidate_changed_rate": _float_value_or_default(
            payload.get("source_selected_candidate_changed_rate"),
            0.0,
        ),
        "calibrated_selected_candidate_changed_rate": calibrated_rate,
        "calibrated_selected_candidate_changed_count": _int_value_or_default(
            payload.get("selected_candidate_changed_count"),
            0,
        ),
        "changed_scenario_ids": changed_scenario_ids,
        "blocked_candidate_rate": blocked_candidate_rate,
        "no_eligible_contrast_count": max(0, len(records) - len(eligible_records)),
        "channel_cost_delta_stats": _dict_value(payload.get("channel_cost_delta_stats")),
        "high_cost_exposure_delta_stats": _dict_value(payload.get("high_cost_exposure_delta_stats")),
        "safety_regression_count": safety_regression_count,
        "recommended_next_action": recommended,
    }


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["contrast_coverage_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "coverage_thresholds": dict(config.get("coverage_thresholds", {})),
        "output_files": dict(config.get("output_files", {})),
    }


def _public_git(payload: dict[str, Any]) -> dict[str, Any]:
    return _shared_public_git(payload)


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
    return _shared_git_snapshot(repo_root, submodules=SUBMODULES)


def _git_snapshots_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _shared_git_snapshots_match(left, right, submodules=SUBMODULES)


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


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {"count": 0, "min": None, "max": None, "mean": None}


def _unique(values) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _int_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer") from exc


def _float_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number") from exc


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
