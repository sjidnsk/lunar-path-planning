from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "planner-validated-training-input-materialization-config/v1"
SUMMARY_SCHEMA_VERSION = "planner-validated-training-input-materialization-summary/v1"
EXCLUSION_SCHEMA_VERSION = "planner-validated-training-exclusion-report/v1"
MINING_SCHEMA_VERSION = "planner-validated-trainable-target-mining-summary/v1"
CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"
POSITIVE_DECISIONS = {
    "selected_default_contract_trainable",
    "selected_planner_validated_distance_exception",
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize planner-validated trainable targets as RolloutEpisode JSONL."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--planner-validated-trainable-target-mining-summary",
        help="Defaults to <batch-root>/planner-validated-trainable-target-mining-summary.json.",
    )
    parser.add_argument(
        "--anchor-projection-candidate-generation-summary",
        help="Defaults to <batch-root>/anchor-projection-candidate-generation-summary.json.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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

    input_files = config.get("input_files", {})
    mining_path = (
        _resolve_path(args.planner_validated_trainable_target_mining_summary, repo_root)
        if args.planner_validated_trainable_target_mining_summary
        else batch_root
        / str(input_files.get("planner_validated_trainable_target_mining_summary", "planner-validated-trainable-target-mining-summary.json"))
    )
    candidate_path = (
        _resolve_path(args.anchor_projection_candidate_generation_summary, repo_root)
        if args.anchor_projection_candidate_generation_summary
        else batch_root
        / str(input_files.get("anchor_projection_candidate_generation_summary", "anchor-projection-candidate-generation-summary.json"))
    )
    result = materialize_planner_validated_training_input(
        batch_root=batch_root,
        mining_path=mining_path,
        candidate_path=candidate_path,
        config=config,
        repo_root=repo_root,
    )
    message = {
        "status": "config validated" if result["summary"]["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "reason_codes": result["summary"]["reason_codes"],
        "input_positive_count": result["summary"]["input_positive_count"],
        "excluded_nontrainable_count": result["summary"]["excluded_nontrainable_count"],
        "rollout_episodes": _display_path(result["rollout_path"], repo_root),
        "summary": _display_path(result["summary_path"], repo_root),
        "exclusion_report": _display_path(result["exclusion_path"], repo_root),
    }
    print(json.dumps(message, ensure_ascii=False))
    if args.validate_only or args.dry_run:
        return 1 if result["summary"]["status"] == "failed" else 0
    _write_outputs(result)
    return 1 if result["summary"]["status"] == "failed" else 0


def materialize_planner_validated_training_input(
    *,
    batch_root: Path,
    mining_path: Path,
    candidate_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    from model_explorer.policy.dataset import summarize_rollout_dataset
    from model_explorer.policy.rollout_io import write_rollout_episodes_jsonl

    reason_codes: list[str] = []
    mining = _load_source(
        mining_path,
        expected_schema=MINING_SCHEMA_VERSION,
        label="planner_validated_mining_summary",
        reason_codes=reason_codes,
    )
    candidate = _load_source(
        candidate_path,
        expected_schema=CANDIDATE_SCHEMA_VERSION,
        label="anchor_projection_candidate_generation_summary",
        reason_codes=reason_codes,
    )
    validation = config["validation"]
    if validation.get("fail_on_input_failure", True):
        if mining.get("status") != "passed":
            _append_reason(reason_codes, "planner_validated_mining_summary_failed")
        if candidate.get("status") not in (None, "passed"):
            _append_reason(reason_codes, "anchor_projection_candidate_generation_summary_failed")
    if validation.get("fail_on_provenance_mismatch", True):
        if _int_value(mining.get("current_git_provenance_mismatch_count")) > 0 or _int_value(
            mining.get("git_provenance_mismatch_count")
        ) > 0:
            _append_reason(reason_codes, "current_git_provenance_mismatch")
    if validation.get("fail_on_fallback_or_open_grid", True):
        if _int_value(mining.get("fallback_or_open_grid_count")) > 0:
            _append_reason(reason_codes, "fallback_or_open_grid_blocks_materialization")
    if validation.get("fail_on_safety_regression", True):
        if _int_value(mining.get("safety_regression_count")) > 0:
            _append_reason(reason_codes, "safety_regression_blocks_materialization")
    if _int_value(candidate.get("candidate_contract_alignment_gap_count")) > 0:
        _append_reason(reason_codes, "candidate_contract_alignment_gap_count_nonzero")

    records = _merged_decision_records(mining, candidate)
    positives = [record for record in records if _decision(record) in POSITIVE_DECISIONS]
    exclusions = [record for record in records if _decision(record) not in POSITIVE_DECISIONS]
    default_count = sum(1 for record in positives if _decision(record) == "selected_default_contract_trainable")
    exception_count = sum(
        1 for record in positives if _decision(record) == "selected_planner_validated_distance_exception"
    )
    invalid_action_mask_records: list[dict[str, Any]] = []
    empty_action_mask_count = 0
    episodes = []
    for sample_index, record in enumerate(positives):
        action_index = _int_value(record.get("source_action_index"))
        action_mask = _action_mask(record, action_index=action_index, config=config)
        if not any(action_mask):
            empty_action_mask_count += 1
        if action_index < 0 or action_index >= len(action_mask) or not action_mask[action_index]:
            invalid_action_mask_records.append(_sample_identity(record, reason="invalid_action_mask"))
            continue
        episodes.append(_rollout_episode(record, action_mask=action_mask, sample_index=sample_index, config=config))
    if invalid_action_mask_records:
        _append_reason(reason_codes, "invalid_action_mask")
    if empty_action_mask_count:
        _append_reason(reason_codes, "empty_action_mask")

    expected = config.get("expected_counts", {})
    _check_expected_count(
        reason_codes,
        expected,
        "input_positive_count",
        len(positives),
    )
    _check_expected_count(
        reason_codes,
        expected,
        "default_contract_positive_count",
        default_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "planner_validated_exception_positive_count",
        exception_count,
    )
    _check_expected_count(
        reason_codes,
        expected,
        "excluded_nontrainable_count",
        len(exclusions),
    )
    dataset_summary: dict[str, Any] | None = None
    if episodes and not invalid_action_mask_records:
        dataset_summary = summarize_rollout_dataset(tuple(episodes))
    elif not episodes and positives:
        _append_reason(reason_codes, "no_valid_positive_samples_materialized")

    status = "failed" if reason_codes else "passed"
    output_files = config["output_files"]
    rollout_path = batch_root / output_files["rollout_episodes"]
    summary_path = batch_root / output_files["summary"]
    exclusion_path = batch_root / output_files["exclusion_report"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": {
            "planner_validated_trainable_target_mining_summary": _source_descriptor(mining_path, mining, repo_root),
            "anchor_projection_candidate_generation_summary": _source_descriptor(candidate_path, candidate, repo_root),
        },
        "input_positive_count": len(positives),
        "default_contract_positive_count": default_count,
        "planner_validated_exception_positive_count": exception_count,
        "excluded_nontrainable_count": len(exclusions),
        "invalid_action_mask_count": len(invalid_action_mask_records),
        "empty_action_mask_count": empty_action_mask_count,
        "materialized_episode_count": len(episodes),
        "materialized_transition_count": len(episodes),
        "rollout_episodes": _display_path(rollout_path, repo_root),
        "exclusion_report": _display_path(exclusion_path, repo_root),
        "dataset_summary": dataset_summary,
        "runs_training": False,
        "dry_run_only": False,
        "performance_claimed": False,
        "publishes_checkpoint": False,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "status": status,
        "excluded_nontrainable_count": len(exclusions),
        "excluded_decision_counts": dict(sorted(Counter(_decision(record) for record in exclusions).items())),
        "invalid_action_mask_records": invalid_action_mask_records,
        "excluded_records": [_exclusion_record(record) for record in exclusions],
    }
    return {
        "summary": summary,
        "exclusion_report": exclusion_report,
        "episodes": tuple(episodes),
        "rollout_path": rollout_path,
        "summary_path": summary_path,
        "exclusion_path": exclusion_path,
        "write_rollout": write_rollout_episodes_jsonl,
    }


def _write_outputs(result: dict[str, Any]) -> None:
    result["summary_path"].parent.mkdir(parents=True, exist_ok=True)
    result["write_rollout"](result["rollout_path"], result["episodes"])
    result["summary_path"].write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["exclusion_path"].write_text(
        json.dumps(result["exclusion_report"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _rollout_episode(
    record: dict[str, Any],
    *,
    action_mask: tuple[bool, ...],
    sample_index: int,
    config: dict[str, Any],
):
    from model_explorer.policy.features import (
        CANDIDATE_FEATURE_NAMES,
        GLOBAL_FEATURE_NAMES,
        MISSING_INDICATOR_NAMES,
        PolicyObservation,
    )
    from model_explorer.policy.rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition

    action_index = _int_value(record.get("source_action_index"))
    candidate_cells = _candidate_cells(record, action_mask=action_mask, action_index=action_index)
    observation = PolicyObservation(
        candidate_feature_names=CANDIDATE_FEATURE_NAMES,
        candidate_features=tuple(_candidate_features(record, cell=cell) for cell in candidate_cells),
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=tuple(0.0 for _ in GLOBAL_FEATURE_NAMES),
        action_mask=action_mask,
        candidate_cells=candidate_cells,
        candidate_missing_feature_names=tuple(() for _ in candidate_cells),
        candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
        candidate_missing_indicators=tuple(
            tuple(0.0 for _ in MISSING_INDICATOR_NAMES) for _ in candidate_cells
        ),
    )
    selected_cell = _cell(record.get("policy_target_cell"))
    path_cost = _float_value(
        record.get("selected_candidate_path_cost", record.get("projected_candidate_path_cost"))
    )
    risk = _float_value(record.get("selected_candidate_risk", record.get("projected_candidate_risk")))
    reward = float(config["materialization"].get("reward", 1.0))
    info = RolloutInfo(
        selected_cell=selected_cell,
        path_cost=0.0 if path_cost is None else path_cost,
        risk=0.0 if risk is None else risk,
        total_cost=0.0 if path_cost is None else path_cost,
        extra={
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "sample_index": sample_index,
            "run_id": record.get("run_id"),
            "scenario_id": record.get("scenario_id"),
            "source_path": record.get("source_path"),
            "source_action_index": action_index,
            "teacher_action_index": action_index,
            "final_training_decision": _decision(record),
            "policy_target_cell": _list_cell(record.get("policy_target_cell")),
            "execution_goal_cell": _list_cell(record.get("execution_goal_cell")),
            "target_binding_mode": record.get("target_binding_mode"),
            "planner_validated_distance_exception": bool(
                record.get("planner_validated_distance_exception")
            ),
            "contract_safe": bool(record.get("contract_safe")),
            "ppo_consumable_action": bool(record.get("ppo_consumable_action")),
            "projection_distance_cells": record.get("projection_distance_cells"),
            "projection_distance_m": record.get("projection_distance_m"),
            "sample_weight": 1.0,
            "selection_strategy": "planner_validated_trainable_target_mining",
            "requested_selection_strategy": "planner_validated_trainable_target_mining",
            "training_use": "planner_validated_trainable_anchor_projection_contrast",
            "dry_run_only": True,
        },
    )
    transition = RolloutTransition(
        observation=observation,
        action_index=action_index,
        log_prob=None,
        value=None,
        reward=reward,
        next_observation=None,
        done=True,
        info=info,
    )
    metrics = EpisodeMetrics(
        final_coverage_rate=0.0,
        cumulative_coverage_rate_delta=0.0,
        total_path_cost=0.0 if path_cost is None else path_cost,
        average_risk=0.0 if risk is None else risk,
    )
    return RolloutEpisode(transitions=(transition,), metrics=metrics)


def _candidate_features(record: dict[str, Any], *, cell: tuple[int, int] | None) -> tuple[float, ...]:
    from model_explorer.policy.features import CANDIDATE_FEATURE_NAMES

    if cell is None:
        return tuple(0.0 for _ in CANDIDATE_FEATURE_NAMES)
    x, y = cell
    utility = _float_value(record.get("selected_candidate_utility", record.get("projected_candidate_utility")))
    path_cost = _float_value(record.get("selected_candidate_path_cost", record.get("projected_candidate_path_cost")))
    risk = _float_value(record.get("selected_candidate_risk", record.get("projected_candidate_risk")))
    values = {
        "cell_x": _clip01(x / 100.0),
        "cell_y": _clip01(y / 100.0),
        "relative_dx": _clip_signed(x / 100.0),
        "relative_dy": _clip_signed(y / 100.0),
        "relative_distance": _clip01(((x * x + y * y) ** 0.5) / 141.5),
        "utility": _clip01(0.0 if utility is None else utility),
        "reachable": 1.0,
        "expected_coverage_rate_delta": 0.0,
        "expected_new_coverage_area": 0.0,
        "information_gain": 0.0,
        "confidence_gain": 0.0,
        "value": _clip01(0.0 if utility is None else utility),
        "risk": _clip01(0.0 if risk is None else risk),
        "path_cost": _clip01(0.0 if path_cost is None else path_cost / 100.0),
        "energy_cost": _clip01(0.0 if path_cost is None else path_cost / 100.0),
    }
    return tuple(float(values[name]) for name in CANDIDATE_FEATURE_NAMES)


def _merged_decision_records(mining: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {
        _record_key(record): record
        for record in candidate.get("context_records", [])
        if isinstance(record, dict)
    }
    records = []
    for record in mining.get("final_decision_records", []):
        if not isinstance(record, dict):
            continue
        merged = dict(candidates.get(_record_key(record), {}))
        merged.update(record)
        records.append(merged)
    return records


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("run_id"),
        record.get("scenario_id"),
        _int_value(record.get("source_action_index")),
        tuple(_list_cell(record.get("policy_target_cell")) or []),
        tuple(_list_cell(record.get("execution_goal_cell")) or []),
    )


def _action_mask(record: dict[str, Any], *, action_index: int, config: dict[str, Any]) -> tuple[bool, ...]:
    raw = record.get("action_mask")
    if isinstance(raw, list):
        return tuple(bool(value) for value in raw)
    action_count = max(int(config["materialization"].get("max_action_count", 1)), action_index + 1)
    return tuple(True for _ in range(action_count))


def _candidate_cells(
    record: dict[str, Any],
    *,
    action_mask: tuple[bool, ...],
    action_index: int,
) -> tuple[tuple[int, int] | None, ...]:
    raw = record.get("candidate_cells")
    if isinstance(raw, list) and len(raw) == len(action_mask):
        return tuple(_cell(cell) for cell in raw)
    cells: list[tuple[int, int] | None] = [None for _ in action_mask]
    if 0 <= action_index < len(cells):
        cells[action_index] = _cell(record.get("policy_target_cell")) or _cell(record.get("execution_goal_cell"))
    return tuple(cells)


def _exclusion_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = _sample_identity(record, reason=_decision(record))
    payload.update(
        {
            "source_selection_status": record.get("source_selection_status"),
            "projection_distance_cells": record.get("projection_distance_cells"),
            "projection_distance_m": record.get("projection_distance_m"),
            "ppo_consumable_action": record.get("ppo_consumable_action"),
            "contract_safe": record.get("contract_safe"),
            "planner_validated_distance_exception": record.get("planner_validated_distance_exception"),
        }
    )
    return payload


def _sample_identity(record: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "run_id": record.get("run_id"),
        "scenario_id": record.get("scenario_id"),
        "source_action_index": record.get("source_action_index"),
        "policy_target_cell": _list_cell(record.get("policy_target_cell")),
        "execution_goal_cell": _list_cell(record.get("execution_goal_cell")),
        "final_training_decision": _decision(record),
    }


def _check_expected_count(
    reason_codes: list[str],
    expected: dict[str, Any],
    field: str,
    actual: int,
) -> None:
    if field not in expected:
        return
    if actual != _int_value(expected.get(field)):
        _append_reason(reason_codes, f"{field}_mismatch")


def _source_descriptor(path: Path, payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, repo_root),
        "exists": path.is_file(),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
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
    for section in ("output_files", "validation", "expected_counts", "materialization"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    for field in ("rollout_episodes", "summary", "exclusion_report"):
        if not payload["output_files"].get(field):
            raise ConfigError(f"output_files.{field} is required")
    return payload


def _load_source(
    path: Path,
    *,
    expected_schema: str,
    label: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_invalid_json_root")
        return {}
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_version_mismatch")
    return payload


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


def _decision(record: dict[str, Any]) -> str:
    return str(record.get("final_training_decision") or "")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _list_cell(value: Any) -> list[int] | None:
    cell = _cell(value)
    return None if cell is None else [int(cell[0]), int(cell[1])]


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _clip_signed(value: float) -> float:
    return min(1.0, max(-1.0, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
