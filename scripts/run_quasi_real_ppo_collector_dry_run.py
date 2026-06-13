from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scripts.run_ppo_rollout_collector_dry_run import (
        REJECTION_SCHEMA_VERSION,
        REWARD_AUDIT_SCHEMA_VERSION,
        SUMMARY_SCHEMA_VERSION,
        TRANSITION_RECORD_SCHEMA_VERSION,
        _PolicyEvaluator,
        _append_reason,
        _apply_validation_gates,
        _cell_tuple,
        _dedup,
        _episode_from_transitions,
        _finite_or_zero,
        _float_or_none,
        _int_value,
        _observation_from_step_or_scenario,
        _output_paths,
        _string_list,
    )
    from scripts.run_scenario_disjoint_policy_rollout_evaluation import (
        _candidate_record,
        _display_path,
    )
except ModuleNotFoundError:
    from run_ppo_rollout_collector_dry_run import (
        REJECTION_SCHEMA_VERSION,
        REWARD_AUDIT_SCHEMA_VERSION,
        SUMMARY_SCHEMA_VERSION,
        TRANSITION_RECORD_SCHEMA_VERSION,
        _PolicyEvaluator,
        _append_reason,
        _apply_validation_gates,
        _cell_tuple,
        _dedup,
        _episode_from_transitions,
        _finite_or_zero,
        _float_or_none,
        _int_value,
        _observation_from_step_or_scenario,
        _output_paths,
        _string_list,
    )
    from run_scenario_disjoint_policy_rollout_evaluation import (
        _candidate_record,
        _display_path,
    )


CONFIG_SCHEMA_VERSION = "quasi-real-ppo-collector-dry-run-config/v1"
DEFAULT_GUARDED_ROOT = "outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize quasi-real guarded teacher-following decisions as PPO dry-run rollout input."
    )
    parser.add_argument("--guarded-teacher-following-root", default=DEFAULT_GUARDED_ROOT)
    parser.add_argument("--guarded-teacher-following-summary", default=None)
    parser.add_argument("--candidate-root", default=None)
    parser.add_argument("--quasi-real-root", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    summary = run_quasi_real_ppo_collector_dry_run(
        guarded_teacher_following_root=_resolve_path(args.guarded_teacher_following_root, repo_root),
        guarded_teacher_following_summary_path=(
            None
            if args.guarded_teacher_following_summary is None
            else _resolve_path(args.guarded_teacher_following_summary, repo_root)
        ),
        candidate_root=None if args.candidate_root is None else _resolve_path(args.candidate_root, repo_root),
        quasi_real_root=None if args.quasi_real_root is None else _resolve_path(args.quasi_real_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=_load_json(_resolve_path(args.config, repo_root)),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "episode_count": summary["episode_count"],
                "step_count": summary["step_count"],
                "ppo_trainable_transition_count": summary["ppo_trainable_transition_count"],
                "diagnostic_transition_count": summary["diagnostic_transition_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_quasi_real_ppo_collector_dry_run(
    *,
    guarded_teacher_following_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
    guarded_teacher_following_summary_path: str | Path | None = None,
    candidate_root: str | Path | None = None,
    quasi_real_root: str | Path | None = None,
) -> dict[str, Any]:
    _install_model_explorer_path(Path(repo_root))
    from model_explorer.policy.dataset import validate_rollout_dataset
    from model_explorer.policy.rollout_io import write_rollout_episodes_jsonl

    repo = Path(repo_root).resolve()
    guarded_root = Path(guarded_teacher_following_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _validate_config(config)
    paths = _output_paths(output, config)

    summary_path = (
        Path(guarded_teacher_following_summary_path).resolve()
        if guarded_teacher_following_summary_path is not None
        else guarded_root / config["input_files"].get(
            "guarded_summary",
            "quasi-real-guarded-teacher-following-pilot-summary.json",
        )
    )
    guarded_summary = _load_json_if_exists(summary_path)
    candidate = (
        Path(candidate_root).resolve()
        if candidate_root is not None
        else _resolve_summary_path(guarded_summary.get("candidate_root"), repo)
    )
    quasi_real = (
        Path(quasi_real_root).resolve()
        if quasi_real_root is not None
        else _resolve_summary_path(guarded_summary.get("quasi_real_root"), repo)
    )

    decisions_path = guarded_root / config["input_files"].get(
        "decisions",
        "quasi-real-guarded-teacher-following-decisions.jsonl",
    )
    if not decisions_path.is_file() and guarded_summary.get("decisions_path"):
        decisions_path = _resolve_summary_path(guarded_summary.get("decisions_path"), repo)
    decisions = _read_jsonl(decisions_path)
    scenario_groups = _quasi_real_scenario_groups_by_id(quasi_real, config=config, repo_root=repo)
    slice_by_scenario = _slice_by_scenario_id(quasi_real, config=config)
    policy_evaluator = _PolicyEvaluator.from_candidate_root(candidate, config=config, repo_root=repo)

    reason_codes: list[str] = []
    transition_records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    episodes_by_id: dict[str, list[Any]] = defaultdict(list)
    counters: Counter[str] = Counter()
    gate_reason_counts: Counter[str] = Counter()

    for index, decision in enumerate(decisions):
        split = _decision_split(decision, slice_by_scenario)
        record, transition = _transition_from_decision(
            decision,
            split=split,
            scenario_group=scenario_groups.get(str(decision.get("scenario_id", ""))),
            policy_evaluator=policy_evaluator,
            config=config,
            step_ordinal=index,
        )
        transition_records.append(record)
        reward_records.append(record["reward_audit"])
        for key, value in record["counter_deltas"].items():
            counters[key] += int(value)
        for reason in _string_list(decision.get("gate_reason_codes")):
            gate_reason_counts[reason] += 1
        if record["rejection_reason_codes"]:
            rejection_records.append(record)
        if transition is not None:
            episodes_by_id[str(record["episode_id"])].append(transition)

    episodes = [_episode_from_transitions(transitions) for _, transitions in sorted(episodes_by_id.items()) if transitions]
    dataset_summary = None
    dataset_error = None
    if episodes:
        try:
            dataset_summary = validate_rollout_dataset(episodes)
        except Exception as exc:  # noqa: BLE001
            dataset_error = str(exc)
            _append_reason(reason_codes, "rollout_dataset_validation_failed")
    elif int(config.get("validation", {}).get("min_ppo_trainable_transition_count", 1)) > 0:
        _append_reason(reason_codes, "ppo_trainable_transition_count_insufficient")

    _apply_validation_gates(reason_codes, counters, config)
    if guarded_summary.get("status") not in {None, "passed"}:
        _append_reason(reason_codes, "quasi_real_guarded_teacher_following_pilot_not_passed")

    write_rollout_episodes_jsonl(paths["episodes"], tuple(episodes))
    _write_jsonl(paths["transitions"], transition_records)
    _write_json(paths["rejection_report"], _rejection_report(rejection_records))
    _write_json(paths["reward_audit"], _reward_audit(reward_records, counters))

    status = "failed" if reason_codes else "passed"
    current_git = _git_snapshot(repo)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "batch_root": _display_path(output, repo),
        "guarded_teacher_following_root": _display_path(guarded_root, repo),
        "guarded_teacher_following_summary": _display_path(summary_path, repo),
        "candidate_root": _display_path(candidate, repo),
        "quasi_real_root": _display_path(quasi_real, repo),
        "episode_count": _int_value(guarded_summary.get("quasi_real_context_count"), len(decisions)),
        "step_count": _int_value(guarded_summary.get("policy_decision_count"), len(decisions)),
        "materialized_episode_count": len(episodes),
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "diagnostic_transition_count": counters["diagnostic_transition_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "invalid_action_mask_count": counters["invalid_action_mask_count"],
        "empty_action_mask_count": counters["empty_action_mask_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "state_continuity_violation_count": 0,
        "fallback_or_open_grid_count": _int_value(guarded_summary.get("fallback_or_open_grid_count"), 0),
        "safety_regression_count": _int_value(guarded_summary.get("safety_regression_count"), 0),
        "contract_violation_count": _int_value(guarded_summary.get("contract_violation_count"), 0),
        "path_cost_regression_count": _int_value(guarded_summary.get("path_cost_regression_count"), 0),
        "risk_regression_count": _int_value(guarded_summary.get("risk_regression_count"), 0),
        "source_selection_regression_count": _int_value(guarded_summary.get("source_selection_regression_count"), 0),
        "gate_reason_counts": dict(sorted(gate_reason_counts.items())),
        "dataset_summary": dataset_summary,
        "dataset_error": dataset_error,
        "episodes": _display_path(paths["episodes"], repo),
        "transitions": _display_path(paths["transitions"], repo),
        "rejection_report": _display_path(paths["rejection_report"], repo),
        "reward_audit": _display_path(paths["reward_audit"], repo),
        "summary": _display_path(paths["summary"], repo),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "experimental_collector_dry_run": True,
        "quasi_real_guarded_teacher_following_collector": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(paths["summary"], summary)
    return summary


def _transition_from_decision(
    decision: dict[str, Any],
    *,
    split: str,
    scenario_group: dict[str, Any] | None,
    policy_evaluator: _PolicyEvaluator,
    config: dict[str, Any],
    step_ordinal: int,
) -> tuple[dict[str, Any], Any | None]:
    from model_explorer.policy.rollout import RolloutInfo, RolloutTransition

    counter_deltas: Counter[str] = Counter()
    rejection_reasons: list[str] = []
    controlled_source = str(decision.get("controlled_choice_source") or "")
    controlled_action_index = _optional_int(decision.get("controlled_action_index"))
    if controlled_action_index is None:
        controlled_action_index = _optional_int(decision.get("teacher_action_index"))
    gate_reasons = _string_list(decision.get("gate_reason_codes"))
    trainable_splits = set(config.get("splits", {}).get("trainable", ["train"]))
    trainable_sources = set(
        config.get(
            "trainable_controlled_choice_sources",
            ["policy_teacher_aligned", "policy_safe_disagreement"],
        )
    )
    ppo_candidate = (
        split in trainable_splits
        and controlled_source in trainable_sources
        and not gate_reasons
        and bool(decision.get("policy_takes_control", True))
    )
    if split not in trainable_splits:
        rejection_reasons.append("non_train_split")
    if controlled_source not in trainable_sources:
        rejection_reasons.append("controlled_choice_source_not_trainable")
    if gate_reasons:
        rejection_reasons.append("gate_reason_codes_present")
    if not ppo_candidate:
        counter_deltas["diagnostic_transition_count"] += 1
    if ppo_candidate and controlled_source == "teacher_fallback":
        counter_deltas["source_fallback_trainable_count"] += 1

    observation_step = {
        "observation": decision.get("observation"),
        "context_id": decision.get("context_id"),
        "policy_selected_context_id": decision.get("context_id"),
        "raw_policy_selected_context_id": decision.get("context_id"),
    }
    observation = _observation_from_step_or_scenario(
        observation_step,
        scenario_group,
        controlled_action_index,
    )
    if observation is None and ppo_candidate:
        rejection_reasons.append("ppo_observation_missing")
        counter_deltas["invalid_action_mask_count"] += 1
    if controlled_action_index is None and ppo_candidate:
        rejection_reasons.append("controlled_action_index_missing")

    log_prob = _float_or_none(decision.get("policy_action_log_prob"))
    value = _float_or_none(decision.get("policy_value"))
    if observation is not None and controlled_action_index is not None and (log_prob is None or value is None):
        evaluated = policy_evaluator.evaluate(observation, controlled_action_index)
        log_prob = evaluated.get("log_prob", log_prob)
        value = evaluated.get("value", value)
    if ppo_candidate and log_prob is None:
        counter_deltas["missing_log_prob_count"] += 1
        rejection_reasons.append("ppo_log_prob_missing")
    if ppo_candidate and value is None:
        counter_deltas["missing_value_count"] += 1
        rejection_reasons.append("ppo_value_missing")

    reward, reward_components, reward_reasons = _compute_reward(decision, ppo_candidate=ppo_candidate, config=config)
    if reward_reasons:
        rejection_reasons.extend(reward_reasons)
        counter_deltas["non_finite_reward_count"] += reward_reasons.count("non_finite_reward")

    invalid_mask = False
    empty_mask = False
    if observation is not None:
        empty_mask = not any(observation.action_mask)
        invalid_mask = (
            controlled_action_index is None
            or controlled_action_index < 0
            or controlled_action_index >= len(observation.action_mask)
            or not observation.action_mask[controlled_action_index]
        )
    if empty_mask:
        counter_deltas["empty_action_mask_count"] += 1
        rejection_reasons.append("empty_action_mask")
    if invalid_mask:
        counter_deltas["invalid_action_mask_count"] += 1
        rejection_reasons.append("invalid_action_mask")

    transition = None
    if ppo_candidate and not rejection_reasons and observation is not None and controlled_action_index is not None:
        counter_deltas["ppo_trainable_transition_count"] += 1
        transition = RolloutTransition(
            observation=observation,
            action_index=controlled_action_index,
            log_prob=log_prob,
            value=value,
            reward=reward,
            next_observation=None,
            done=True,
            info=RolloutInfo(
                selected_cell=_selected_cell(decision, observation, controlled_action_index),
                coverage_rate_delta=0.0,
                path_cost=abs(_finite_or_zero(decision.get("path_cost_delta"))),
                risk=abs(_finite_or_zero(decision.get("risk_delta"))),
                failure_reason=None,
                final_coverage_rate=None,
                total_cost=0.0,
                failure_count=0,
                replan_count=0,
                extra={
                    "ppo_trainable": True,
                    "episode_id": _episode_id(decision, step_ordinal),
                    "step_index": 0,
                    "context_id": decision.get("context_id"),
                    "scenario_id": decision.get("scenario_id"),
                    "scenario_family": decision.get("roi_group") or decision.get("scenario_group"),
                    "split": split,
                    "raw_policy_action_index": decision.get("raw_policy_action_index"),
                    "source_action_index": decision.get("source_action_index"),
                    "teacher_action_index": decision.get("teacher_action_index"),
                    "controlled_action_index": controlled_action_index,
                    "controlled_choice_source": controlled_source,
                    "path_cost_delta": decision.get("path_cost_delta"),
                    "risk_delta": decision.get("risk_delta"),
                    "reward_components": reward_components,
                    "gate_reason_codes": gate_reasons,
                },
            ),
        )

    record = {
        "schema_version": TRANSITION_RECORD_SCHEMA_VERSION,
        "episode_id": _episode_id(decision, step_ordinal),
        "step_index": 0,
        "decision_index": step_ordinal,
        "scenario_id": decision.get("scenario_id"),
        "scenario_family": decision.get("roi_group") or decision.get("scenario_group"),
        "context_id": decision.get("context_id"),
        "split": split,
        "controlled_choice_source": controlled_source,
        "controlled_action_index": controlled_action_index,
        "ppo_trainable": bool(transition is not None),
        "diagnostic_only": transition is None,
        "rejection_reason_codes": _dedup(rejection_reasons),
        "reward": reward,
        "reward_components": reward_components,
        "reward_audit": {
            "episode_id": _episode_id(decision, step_ordinal),
            "step_index": 0,
            "decision_index": step_ordinal,
            "split": split,
            "ppo_trainable_candidate": ppo_candidate,
            "reward": reward,
            "components": reward_components,
            "reason_codes": reward_reasons,
        },
        "counter_deltas": dict(counter_deltas),
    }
    return record, transition


def _compute_reward(
    decision: dict[str, Any],
    *,
    ppo_candidate: bool,
    config: dict[str, Any],
) -> tuple[float, dict[str, float], list[str]]:
    reward_config = config.get("reward", {})
    path_delta = _float_or_none(decision.get("path_cost_delta"))
    risk_delta = _float_or_none(decision.get("risk_delta"))
    values = [value for value in (path_delta, risk_delta) if value is not None]
    if any(not math.isfinite(value) for value in values):
        return float("nan"), {}, ["non_finite_reward"]
    controlled_source = str(decision.get("controlled_choice_source") or "")
    gate_reasons = _string_list(decision.get("gate_reason_codes"))
    components = {
        "teacher_following_bonus": (
            float(reward_config.get("teacher_following_bonus", 1.0))
            if ppo_candidate and controlled_source == "policy_teacher_aligned"
            else 0.0
        ),
        "safe_disagreement_bonus": (
            float(reward_config.get("safe_disagreement_bonus", 1.0))
            if ppo_candidate and controlled_source == "policy_safe_disagreement"
            else 0.0
        ),
        "gate_penalty": -float(reward_config.get("gate_regression_penalty", 1.0)) if gate_reasons else 0.0,
    }
    reward = sum(components.values())
    if not math.isfinite(reward):
        return reward, components, ["non_finite_reward"]
    return reward, components, []


def _quasi_real_scenario_groups_by_id(
    quasi_real_root: Path,
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    summary_path = quasi_real_root / config["input_files"].get(
        "quasi_real_path_feedback_summary",
        "quasi-real-map-path-feedback-summary.json",
    )
    payload = _load_json_if_exists(summary_path)
    groups: dict[str, dict[str, Any]] = {}
    for scenario in payload.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
        candidates = path_feedback.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        group = {
            "run_id": summary_path.parent.name,
            "source_path": _display_path(summary_path, repo_root),
            "scenario_id": str(scenario.get("scenario_id", "")),
            "scenario_group": str(scenario.get("scenario_group") or "unknown"),
            "scenario_seed": scenario.get("scenario_seed"),
            "scenario_variant_id": scenario.get("scenario_variant_id"),
            "diagnostic_profile": payload.get("diagnostic_profile"),
            "planning_backend": "quasi_real_path_feedback",
            "best_by_path_cost": path_feedback.get("best_by_path_cost") if isinstance(path_feedback.get("best_by_path_cost"), dict) else None,
            "candidates": [
                _candidate_record(candidate, scenario=scenario, summary=payload, group_path=summary_path, repo_root=repo_root)
                for candidate in candidates
                if isinstance(candidate, dict)
            ],
        }
        groups[group["scenario_id"]] = group
    return groups


def _slice_by_scenario_id(quasi_real_root: Path, *, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = quasi_real_root / config["input_files"].get("quasi_real_slices", "quasi-real-map-slices.jsonl")
    return {str(record.get("scenario_id")): record for record in _read_jsonl(path) if record.get("scenario_id")}


def _decision_split(decision: dict[str, Any], slice_by_scenario: dict[str, dict[str, Any]]) -> str:
    if decision.get("split"):
        return str(decision.get("split"))
    scenario = slice_by_scenario.get(str(decision.get("scenario_id", "")), {})
    return str(scenario.get("split") or "unknown")


def _selected_cell(decision: dict[str, Any], observation: Any, action_index: int) -> tuple[int, int] | None:
    for key in ("controlled_execution_goal_cell", "execution_goal_cell", "policy_target_cell"):
        cell = _cell_tuple(decision.get(key))
        if cell is not None:
            return cell
    if 0 <= action_index < len(observation.candidate_cells):
        return observation.candidate_cells[action_index]
    return None


def _episode_id(decision: dict[str, Any], step_ordinal: int) -> str:
    return str(decision.get("scenario_id") or decision.get("context_id") or f"quasi-real-step-{step_ordinal:04d}")


def _rejection_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(reason for record in records for reason in record.get("rejection_reason_codes", []))
    return {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "rejected_transition_count": len(records),
        "reason_code_counts": dict(sorted(counts.items())),
        "records": records,
    }


def _reward_audit(records: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    finite_records = [
        record
        for record in records
        if _float_or_none(record.get("reward")) is not None and math.isfinite(float(record.get("reward")))
    ]
    rewards = [float(record.get("reward", 0.0)) for record in finite_records]
    trainable_rewards = [
        float(record.get("reward", 0.0))
        for record in finite_records
        if record.get("ppo_trainable_candidate")
    ]
    return {
        "schema_version": REWARD_AUDIT_SCHEMA_VERSION,
        "record_count": len(records),
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_all_zero": bool(rewards and all(value == 0.0 for value in rewards)),
        "trainable_reward_all_zero": bool(trainable_rewards and all(value == 0.0 for value in trainable_rewards)),
        "records": records,
    }


def _next_required_change(reason_codes: list[str]) -> str:
    if "ppo_logprob_value_missing" in reason_codes:
        return "ppo_logprob_value_missing"
    if "ppo_reward_contract_invalid" in reason_codes:
        return "ppo_reward_contract_invalid"
    if "ppo_trainable_transition_count_insufficient" in reason_codes:
        return "ppo_trainable_transition_count_insufficient"
    if "quasi_real_guarded_teacher_following_pilot_not_passed" in reason_codes:
        return "quasi_real_guarded_teacher_following_pilot_required"
    return "ppo_rollout_collector_contract_invalid"


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "validation", "reward"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be an object")


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _resolve_summary_path(value: Any, repo_root: Path) -> Path:
    if value is None:
        return repo_root
    return _resolve_path(str(value), repo_root).resolve()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
