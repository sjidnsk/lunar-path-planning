from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import git_snapshots_match as _git_snapshots_match


CONFIG_SCHEMA_VERSION = "return-aligned-guarded-multistep-collector-config/v1"
SUMMARY_SCHEMA_VERSION = "return-aligned-guarded-multistep-collector-summary/v1"
EPISODE_SCHEMA_VERSION = "return-aligned-guarded-multistep-episode/v1"
TRANSITION_SCHEMA_VERSION = "return-aligned-guarded-multistep-transition/v1"
REWARD_AUDIT_SCHEMA_VERSION = "return-aligned-guarded-multistep-reward-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "return-aligned-guarded-multistep-rejection-report/v1"

CONTROLLED_REGRESSION_REASONS = {
    "safety_regression",
    "contract_violation",
    "contract_regression",
    "path_cost_regression",
    "risk_regression",
    "source_selection_regression",
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand guarded PPO collector evidence into return-aligned multi-step rollout audit."
    )
    parser.add_argument("--guarded-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1")
    parser.add_argument(
        "--evidence-freeze-summary",
        default="outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/guarded-ppo-evidence-freeze-summary.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/return_aligned_guarded_multi_step_ppo_collector_expansion_v1.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": _display_path(config_path, repo_root)}))
        return 0

    summary = run_return_aligned_guarded_multi_step_ppo_collector_expansion(
        guarded_root=_resolve_path(args.guarded_root, repo_root),
        evidence_freeze_summary_path=_resolve_path(args.evidence_freeze_summary, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "horizon": summary["horizon"],
                "trainable_episode_count": summary["trainable_episode_count"],
                "trainable_transition_count": summary["trainable_transition_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_return_aligned_guarded_multi_step_ppo_collector_expansion(
    *,
    guarded_root: Path,
    evidence_freeze_summary_path: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, config)
    input_files = config.get("input_files", {}) if isinstance(config.get("input_files"), dict) else {}
    guarded_summary_path = guarded_root / str(input_files.get("guarded_summary", "guarded-ppo-rollout-pilot-summary.json"))
    collector_summary_path = guarded_root / str(input_files.get("collector_summary", "pilot/collector/ppo-rollout-collector-summary.json"))
    collector_transitions_path = guarded_root / str(input_files.get("collector_transitions", "pilot/collector/ppo-rollout-transitions.jsonl"))

    reason_codes: list[str] = []
    guarded_summary = _load_json_required(guarded_summary_path, reason_codes, "guarded_pilot_summary")
    freeze_summary = _load_json_required(evidence_freeze_summary_path, reason_codes, "guarded_ppo_evidence_freeze_summary")
    collector_summary = _load_json_required(collector_summary_path, reason_codes, "guarded_collector_summary")
    transition_records = _read_jsonl_required(collector_transitions_path, reason_codes, "guarded_collector_transitions")
    current_git = _git_snapshot(repo_root)

    _validate_source_summary(
        guarded_summary,
        label="guarded_pilot_summary",
        reason_codes=reason_codes,
        current_git=current_git,
        expected_status="passed",
    )
    _validate_source_summary(
        freeze_summary,
        label="guarded_ppo_evidence_freeze_summary",
        reason_codes=reason_codes,
        current_git=current_git,
        expected_status="passed",
    )
    if freeze_summary and freeze_summary.get("training_readiness_status") != "guarded_ppo_rollout_pilot_evaluated":
        _append_reason(reason_codes, "evidence_freeze_readiness_not_guarded_pilot")
    if freeze_summary and freeze_summary.get("training_blockers"):
        _append_reason(reason_codes, "evidence_freeze_training_blockers_non_empty")
    _validate_source_summary(
        collector_summary,
        label="guarded_collector_summary",
        reason_codes=reason_codes,
        current_git=current_git,
        expected_status="passed",
    )

    horizon = max(1, _int_value(config.get("horizon"), 3))
    discount = _float_value(config.get("discount_factor"), 0.99)
    grouped = _group_by_episode(transition_records)
    transition_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for episode_id, records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda item: _int_value(item.get("step_index"), 0))
        episode_row = _episode_row(
            episode_id=episode_id,
            records=ordered,
            horizon=horizon,
            discount=discount,
            config=config,
        )
        episode_rows.append(episode_row)
        if not episode_row["ppo_trainable_episode"]:
            rejection_rows.append(_episode_rejection_row(episode_row))
        for transition in ordered:
            transition_row = _transition_row(
                transition,
                episode_row=episode_row,
                config=config,
            )
            transition_rows.append(transition_row)
            for key in _transition_counter_deltas(transition_row):
                counters[key] += 1

    for episode in episode_rows:
        for key in _episode_counter_deltas(episode):
            counters[key] += 1

    reward_audit = _reward_audit(episode_rows, transition_rows, config)
    _write_jsonl(paths["episodes"], episode_rows)
    _write_jsonl(paths["transitions"], transition_rows)
    _write_json(paths["reward_audit"], reward_audit)
    _write_json(paths["rejection_report"], _rejection_report(rejection_rows))

    _apply_validation(reason_codes, counters, config)
    input_invalid_action_mask_count = _int_value(collector_summary.get("invalid_action_mask_count"), 0)
    input_empty_action_mask_count = _int_value(collector_summary.get("empty_action_mask_count"), 0)
    input_missing_log_prob_count = _int_value(collector_summary.get("missing_log_prob_count"), 0)
    input_missing_value_count = _int_value(collector_summary.get("missing_value_count"), 0)
    if input_invalid_action_mask_count:
        _append_reason(reason_codes, "input_collector_invalid_action_mask_detected")
    if input_empty_action_mask_count:
        _append_reason(reason_codes, "input_collector_empty_action_mask_detected")
    if input_missing_log_prob_count:
        _append_reason(reason_codes, "input_collector_missing_log_prob_detected")
    if input_missing_value_count:
        _append_reason(reason_codes, "input_collector_missing_value_detected")
    if not episode_rows and not reason_codes:
        _append_reason(reason_codes, "return_aligned_episode_coverage_missing")
    status = "passed" if not reason_codes else "failed"
    return_values = [row["discounted_episode_return"] for row in episode_rows if _is_finite(row["discounted_episode_return"])]

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "guarded_root": _display_path(guarded_root, repo_root),
        "evidence_freeze_summary": _display_path(evidence_freeze_summary_path, repo_root),
        "guarded_pilot_summary": _display_path(guarded_summary_path, repo_root),
        "collector_summary": _display_path(collector_summary_path, repo_root),
        "collector_transitions": _display_path(collector_transitions_path, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "horizon": horizon,
        "discount_factor": discount,
        "episode_count": len(episode_rows),
        "step_count": len(transition_rows),
        "horizon_complete_episode_count": counters["horizon_complete_episode_count"],
        "trainable_episode_count": counters["trainable_episode_count"],
        "diagnostic_episode_count": len(episode_rows) - counters["trainable_episode_count"],
        "trainable_transition_count": counters["trainable_transition_count"],
        "ppo_trainable_transition_count": counters["trainable_transition_count"],
        "diagnostic_transition_count": len(transition_rows) - counters["trainable_transition_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "invalid_action_mask_count": input_invalid_action_mask_count,
        "empty_action_mask_count": input_empty_action_mask_count,
        "missing_log_prob_count": input_missing_log_prob_count,
        "missing_value_count": input_missing_value_count,
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "teacher_equivalent_episode_count": counters["teacher_equivalent_episode_count"],
        "safe_better_episode_count": counters["safe_better_episode_count"],
        "return_min": min(return_values) if return_values else None,
        "return_max": max(return_values) if return_values else None,
        "return_mean": sum(return_values) / len(return_values) if return_values else None,
        "uses_multistep_discounted_return": True,
        "not_single_step_best_action": True,
        "episodes": _display_path(paths["episodes"], repo_root),
        "transitions": _display_path(paths["transitions"], repo_root),
        "reward_audit": _display_path(paths["reward_audit"], repo_root),
        "rejection_report": _display_path(paths["rejection_report"], repo_root),
        "summary": _display_path(paths["summary"], repo_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_ppo_update": False,
        "runs_formal_ppo_rollout": False,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_default_astar": True,
        "does_not_relax_gates": True,
        "git_provenance": {
            "current": current_git,
            "current_matches_sources": _sources_match_current_head(
                [guarded_summary, freeze_summary, collector_summary],
                current_git=current_git,
            ),
        },
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(paths["summary"], summary)
    return summary


def _episode_row(
    *,
    episode_id: str,
    records: list[dict[str, Any]],
    horizon: int,
    discount: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    horizon_records = records[:horizon]
    rewards = [_float_or_nan(record.get("reward")) for record in horizon_records]
    finite_rewards = all(_is_finite(value) for value in rewards)
    discounted_return = _discounted_sum(rewards, discount) if finite_rewards else float("nan")
    teacher_return = _discounted_sum(
        [
            reward if str(record.get("controlled_choice_source") or "") == "source" else 0.0
            for record, reward in zip(horizon_records, rewards)
        ],
        discount,
    ) if finite_rewards else float("nan")
    regression_reasons = _controlled_regression_reasons(horizon_records)
    controlled_penalty = -float(config.get("reward", {}).get("controlled_regression_penalty", 1.0)) * len(
        regression_reasons
    )
    advantage_reference = float(config.get("reward", {}).get("advantage_reference_value", 0.0))
    advantage = discounted_return - advantage_reference if _is_finite(discounted_return) else float("nan")
    tolerance = float(config.get("reward", {}).get("teacher_return_tolerance", 1e-9))
    has_source_fallback = any(str(record.get("controlled_choice_source") or "") == "source_fallback" for record in horizon_records)
    splits = {str(record.get("split") or "") for record in horizon_records}
    split_allowed = splits <= {"train"}
    complete = len(horizon_records) >= horizon
    has_gate_reasons = any(_gate_reasons(record) for record in horizon_records)
    teacher_equivalent = bool(
        complete
        and finite_rewards
        and not regression_reasons
        and discounted_return + tolerance >= teacher_return
    )
    safe_better = bool(teacher_equivalent and discounted_return + tolerance >= teacher_return)
    ppo_trainable_episode = bool(
        complete
        and split_allowed
        and finite_rewards
        and not has_source_fallback
        and not has_gate_reasons
        and not regression_reasons
    )
    rejection_reasons: list[str] = []
    if not complete:
        rejection_reasons.append("incomplete_horizon")
    if not split_allowed:
        rejection_reasons.append("non_train_split_episode")
    if has_source_fallback:
        rejection_reasons.append("source_fallback_diagnostic_episode")
    if has_gate_reasons:
        rejection_reasons.append("gate_reason_diagnostic_episode")
    if not finite_rewards:
        rejection_reasons.append("non_finite_return")
    if regression_reasons:
        rejection_reasons.append("controlled_regression_episode")
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "horizon": horizon,
        "observed_step_count": len(records),
        "horizon_step_count": len(horizon_records),
        "horizon_complete": complete,
        "splits": sorted(splits),
        "ppo_trainable_episode": ppo_trainable_episode,
        "diagnostic_only": not ppo_trainable_episode,
        "trainable_transition_count": sum(1 for record in horizon_records if _strict_step_trainable(record, config)),
        "diagnostic_transition_count": sum(1 for record in horizon_records if not _strict_step_trainable(record, config)),
        "source_fallback_step_count": sum(1 for record in horizon_records if str(record.get("controlled_choice_source") or "") == "source_fallback"),
        "discounted_episode_return": discounted_return,
        "teacher_following_return": teacher_return,
        "teacher_equivalent_return": discounted_return if teacher_equivalent else 0.0,
        "safe_better_return": max(0.0, discounted_return - teacher_return) if _is_finite(discounted_return) and _is_finite(teacher_return) else float("nan"),
        "controlled_regression_penalty": controlled_penalty,
        "advantage_reference_value": advantage_reference,
        "advantage": advantage,
        "teacher_equivalent_episode": teacher_equivalent,
        "safe_better_episode": safe_better,
        "controlled_regression_reason_codes": regression_reasons,
        "rejection_reason_codes": _dedup(rejection_reasons),
        "uses_multistep_discounted_return": True,
        "not_single_step_best_action": True,
        "step_ids": [
            {
                "step_index": record.get("step_index"),
                "context_id": record.get("context_id"),
                "controlled_choice_source": record.get("controlled_choice_source"),
                "ppo_trainable": _strict_step_trainable(record, config),
            }
            for record in horizon_records
        ],
    }


def _transition_row(record: dict[str, Any], *, episode_row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    ppo_trainable = _strict_step_trainable(record, config)
    reasons = list(_gate_reasons(record))
    if not ppo_trainable:
        source = str(record.get("controlled_choice_source") or "")
        split = str(record.get("split") or "")
        if split in {"validation", "test"}:
            reasons.append(f"{split}_diagnostic_only")
        if source in {"source_fallback", "teacher_fallback", "none", "not_scored"}:
            reasons.append(f"{source}_diagnostic_only")
        if source == "source":
            reasons.append("teacher_source_diagnostic_only")
        if record.get("ppo_trainable") is not True:
            reasons.append("input_transition_not_ppo_trainable")
    return {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "episode_id": record.get("episode_id"),
        "step_index": record.get("step_index"),
        "scenario_id": record.get("scenario_id"),
        "scenario_family": record.get("scenario_family"),
        "context_id": record.get("context_id"),
        "split": record.get("split"),
        "controlled_choice_source": record.get("controlled_choice_source"),
        "controlled_action_index": record.get("controlled_action_index"),
        "input_ppo_trainable": bool(record.get("ppo_trainable")),
        "ppo_trainable": ppo_trainable,
        "diagnostic_only": not ppo_trainable,
        "reward": _float_or_nan(record.get("reward")),
        "discounted_episode_return": episode_row["discounted_episode_return"],
        "advantage_reference_value": episode_row["advantage_reference_value"],
        "advantage": episode_row["advantage"],
        "gate_reason_codes": _gate_reasons(record),
        "rejection_reason_codes": _dedup(reasons),
        "reward_components": record.get("reward_components") if isinstance(record.get("reward_components"), dict) else {},
    }


def _strict_step_trainable(record: dict[str, Any], config: dict[str, Any]) -> bool:
    trainable_filter = config.get("trainable_filter", {}) if isinstance(config.get("trainable_filter"), dict) else {}
    allowed_splits = {str(value) for value in trainable_filter.get("splits", ["train"])}
    allowed_sources = {str(value) for value in trainable_filter.get("controlled_choice_sources", ["policy"])}
    if str(record.get("split") or "") not in allowed_splits:
        return False
    if str(record.get("controlled_choice_source") or "") not in allowed_sources:
        return False
    if bool(trainable_filter.get("require_input_ppo_trainable", True)) and record.get("ppo_trainable") is not True:
        return False
    if bool(trainable_filter.get("require_empty_gate_reason_codes", True)) and _gate_reasons(record):
        return False
    reward = _float_or_nan(record.get("reward"))
    if not _is_finite(reward):
        return False
    return True


def _transition_counter_deltas(row: dict[str, Any]) -> list[str]:
    deltas = []
    split = str(row.get("split") or "")
    source = str(row.get("controlled_choice_source") or "")
    reward = _float_or_nan(row.get("reward"))
    if row["ppo_trainable"]:
        deltas.append("trainable_transition_count")
        if split == "validation":
            deltas.append("validation_trainable_count")
        if split == "test":
            deltas.append("test_trainable_count")
        if source == "source_fallback":
            deltas.append("source_fallback_trainable_count")
    if not _is_finite(reward):
        deltas.append("non_finite_reward_count")
    return deltas


def _episode_counter_deltas(row: dict[str, Any]) -> list[str]:
    deltas = []
    if row["horizon_complete"]:
        deltas.append("horizon_complete_episode_count")
    if row["ppo_trainable_episode"]:
        deltas.append("trainable_episode_count")
    if row["ppo_trainable_episode"] and row["teacher_equivalent_episode"]:
        deltas.append("teacher_equivalent_episode_count")
    if row["ppo_trainable_episode"] and row["safe_better_episode"]:
        deltas.append("safe_better_episode_count")
    if not _is_finite(row["discounted_episode_return"]):
        deltas.append("non_finite_return_count")
    if not _is_finite(row["advantage"]):
        deltas.append("non_finite_advantage_count")
    regression_reasons = row.get("controlled_regression_reason_codes") or []
    if regression_reasons:
        deltas.append("controlled_regression_count")
    if "safety_regression" in regression_reasons:
        deltas.append("controlled_safety_regression_count")
    if "contract_violation" in regression_reasons or "contract_regression" in regression_reasons:
        deltas.append("controlled_contract_regression_count")
    if "path_cost_regression" in regression_reasons or "risk_regression" in regression_reasons:
        deltas.append("controlled_path_risk_regression_count")
    if "source_selection_regression" in regression_reasons:
        deltas.append("controlled_source_selection_regression_count")
    return deltas


def _apply_validation(reason_codes: list[str], counters: Counter[str], config: dict[str, Any]) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    if counters["trainable_episode_count"] < _int_value(validation.get("min_trainable_episode_count"), 24):
        _append_reason(reason_codes, "trainable_episode_count_below_threshold")
    if counters["trainable_transition_count"] < _int_value(validation.get("min_trainable_transition_count"), 24):
        _append_reason(reason_codes, "trainable_transition_count_below_threshold")
    if counters["horizon_complete_episode_count"] < _int_value(validation.get("min_trainable_episode_count"), 24):
        _append_reason(reason_codes, "horizon_complete_episode_count_below_threshold")
    if counters["validation_trainable_count"] > _int_value(validation.get("max_validation_trainable_count"), 0):
        _append_reason(reason_codes, "validation_trainable_leakage")
    if counters["test_trainable_count"] > _int_value(validation.get("max_test_trainable_count"), 0):
        _append_reason(reason_codes, "test_trainable_leakage")
    if counters["source_fallback_trainable_count"] > _int_value(validation.get("max_source_fallback_trainable_count"), 0):
        _append_reason(reason_codes, "source_fallback_trainable_detected")
    if counters["non_finite_reward_count"] > _int_value(validation.get("max_non_finite_reward_count"), 0):
        _append_reason(reason_codes, "non_finite_reward_detected")
    if counters["non_finite_return_count"] > _int_value(validation.get("max_non_finite_return_count"), 0):
        _append_reason(reason_codes, "non_finite_return_detected")
    if counters["non_finite_advantage_count"] > _int_value(validation.get("max_non_finite_advantage_count"), 0):
        _append_reason(reason_codes, "non_finite_advantage_detected")
    if counters["controlled_regression_count"] > _int_value(validation.get("max_controlled_regression_count"), 0):
        _append_reason(reason_codes, "controlled_regression_detected")


def _reward_audit(
    episode_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    finite_returns = [row["discounted_episode_return"] for row in episode_rows if _is_finite(row["discounted_episode_return"])]
    return {
        "schema_version": REWARD_AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "episode_count": len(episode_rows),
        "transition_count": len(transition_rows),
        "horizon": _int_value(config.get("horizon"), 3),
        "discount_factor": _float_value(config.get("discount_factor"), 0.99),
        "uses_multistep_discounted_return": True,
        "not_single_step_best_action": True,
        "component_names": [
            "teacher_following_return",
            "teacher_equivalent_return",
            "safe_better_return",
            "controlled_regression_penalty",
            "discounted_episode_return",
            "advantage_reference_value",
        ],
        "return_min": min(finite_returns) if finite_returns else None,
        "return_max": max(finite_returns) if finite_returns else None,
        "return_mean": sum(finite_returns) / len(finite_returns) if finite_returns else None,
        "non_finite_return_count": sum(1 for row in episode_rows if not _is_finite(row["discounted_episode_return"])),
        "non_finite_advantage_count": sum(1 for row in episode_rows if not _is_finite(row["advantage"])),
        "teacher_equivalent_episode_count": sum(1 for row in episode_rows if row["teacher_equivalent_episode"]),
        "safe_better_episode_count": sum(1 for row in episode_rows if row["safe_better_episode"]),
    }


def _rejection_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(reason for row in rows for reason in row.get("rejection_reason_codes", []))
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "failed_episode_count": len(rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "failed_episodes": rows,
    }


def _episode_rejection_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "horizon_complete": row["horizon_complete"],
        "rejection_reason_codes": row["rejection_reason_codes"],
        "discounted_episode_return": row["discounted_episode_return"],
        "source_fallback_step_count": row["source_fallback_step_count"],
    }


def _controlled_regression_reasons(records: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for record in records:
        controlled = record.get("controlled_regression_reason_codes")
        if not isinstance(controlled, list):
            controlled = []
        for reason in [str(item) for item in controlled]:
            if reason in CONTROLLED_REGRESSION_REASONS:
                reasons.append(reason)
    return _dedup(reasons)


def _gate_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in (
        "gate_reason_codes",
        "rejection_reason_codes",
        "controlled_regression_reason_codes",
    ):
        value = record.get(key)
        if isinstance(value, list):
            reasons.extend(str(item) for item in value)
    return _dedup(reasons)


def _validate_source_summary(
    payload: dict[str, Any],
    *,
    label: str,
    reason_codes: list[str],
    current_git: dict[str, Any],
    expected_status: str,
) -> None:
    if not payload:
        return
    if payload.get("status") != expected_status or payload.get("reason_codes"):
        _append_reason(reason_codes, f"{label}_not_passed")
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    if git.get("current_matches_sources") is False:
        _append_reason(reason_codes, f"{label}_git_current_mismatch")
    source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
    if source_current and not _git_snapshots_match(
        _head_only_snapshot(source_current),
        _head_only_snapshot(current_git),
    ):
        _append_reason(reason_codes, f"{label}_git_current_mismatch")


def _sources_match_current_head(sources: list[dict[str, Any]], *, current_git: dict[str, Any]) -> bool:
    for payload in sources:
        git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
        if git.get("current_matches_sources") is False:
            return False
        source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
        if source_current and not _git_snapshots_match(
            _head_only_snapshot(source_current),
            _head_only_snapshot(current_git),
        ):
            return False
    return True


def _head_only_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    def clean_repo(repo: Any) -> dict[str, Any]:
        repo = repo if isinstance(repo, dict) else {}
        return {"sha": repo.get("sha"), "dirty": False}

    modules = snapshot.get("submodules") if isinstance(snapshot.get("submodules"), dict) else {}
    return {
        "parent": clean_repo(snapshot.get("parent")),
        "submodules": {name: clean_repo(value) for name, value in modules.items()},
    }


def _output_paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    output_files = config.get("output_files", {}) if isinstance(config.get("output_files"), dict) else {}
    return {
        "episodes": output_root / str(output_files.get("episodes", "return-aligned-ppo-episodes.jsonl")),
        "transitions": output_root / str(output_files.get("transitions", "return-aligned-ppo-transitions.jsonl")),
        "reward_audit": output_root / str(output_files.get("reward_audit", "return-aligned-reward-audit.json")),
        "rejection_report": output_root / str(output_files.get("rejection_report", "return-aligned-rejection-report.json")),
        "summary": output_root / str(output_files.get("summary", "return-aligned-collector-summary.json")),
    }


def _group_by_episode(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        episode_id = str(record.get("episode_id") or f"episode-{index:04d}")
        grouped[episode_id].append(record)
    return grouped


def _discounted_sum(values: list[float], discount: float) -> float:
    total = 0.0
    for index, value in enumerate(values):
        if not _is_finite(value):
            return float("nan")
        total += (discount**index) * value
    return total


def _read_jsonl_required(path: Path, reason_codes: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        _append_reason(reason_codes, f"{label}_empty")
    return rows


def _load_json_required(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"invalid schema_version: {payload.get('schema_version')}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _float_value(value: Any, default: float) -> float:
    result = _float_or_nan(value)
    return result if _is_finite(result) else default


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _dedup(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _next_required_change(reason_codes: list[str]) -> str | None:
    return reason_codes[0] if reason_codes else None


if __name__ == "__main__":
    raise SystemExit(main())
