from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from git_provenance import git_snapshot as _git_snapshot

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scripts.run_quasi_real_ppo_collector_dry_run import (
        _PolicyEvaluator,
        _decision_split,
        _observation_from_step_or_scenario,
        _quasi_real_scenario_groups_by_id,
        _slice_by_scenario_id,
        _string_list,
        run_quasi_real_ppo_collector_dry_run,
    )
    from scripts.run_scenario_disjoint_policy_rollout_evaluation import _display_path
except ModuleNotFoundError:
    from run_quasi_real_ppo_collector_dry_run import (
        _PolicyEvaluator,
        _decision_split,
        _observation_from_step_or_scenario,
        _quasi_real_scenario_groups_by_id,
        _slice_by_scenario_id,
        _string_list,
        run_quasi_real_ppo_collector_dry_run,
    )
    from run_scenario_disjoint_policy_rollout_evaluation import _display_path


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-ppo-rollout-pilot-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-rollout-pilot-summary/v1"
STEP_SCHEMA_VERSION = "quasi-real-guarded-ppo-rollout-step/v1"
EPISODE_SCHEMA_VERSION = "quasi-real-guarded-ppo-rollout-episode/v1"
REJECTION_SCHEMA_VERSION = "quasi-real-guarded-ppo-rollout-rejection-report/v1"
REWARD_AUDIT_SCHEMA_VERSION = "quasi-real-guarded-ppo-rollout-reward-audit/v1"

NEXT_INPUT_INVALID = "quasi_real_guarded_ppo_rollout_input_invalid"
NEXT_CONTRACT_INVALID = "quasi_real_guarded_ppo_rollout_contract_invalid"
NEXT_CONTROLLED_REGRESSION = "quasi_real_guarded_ppo_rollout_controlled_regression"
NEXT_NON_FINITE_RETURN = "quasi_real_guarded_ppo_rollout_non_finite_return"

CONTROLLED_REGRESSION_REASONS = {
    "safety_regression",
    "contract_violation",
    "contract_regression",
    "path_cost_regression",
    "risk_regression",
    "source_selection_regression",
}

CollectorReplayRunner = Callable[..., dict[str, Any]]


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quasi-real guarded PPO rollout pilot.")
    parser.add_argument(
        "--update-smoke-root",
        default="outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1",
    )
    parser.add_argument("--candidate-root", default=None)
    parser.add_argument(
        "--quasi-real-root",
        default="outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1",
    )
    parser.add_argument("--config", required=True)
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

    update_root = _resolve_path(args.update_smoke_root, repo_root)
    summary = run_quasi_real_guarded_ppo_rollout_pilot(
        update_smoke_root=update_root,
        candidate_root=(
            _resolve_path(args.candidate_root, repo_root)
            if args.candidate_root
            else update_root
        ),
        quasi_real_root=_resolve_path(args.quasi_real_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "episode_count": summary["episode_count"],
                "step_count": summary["step_count"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "teacher_agreement_rate": summary["teacher_agreement_rate"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_quasi_real_guarded_ppo_rollout_pilot(
    *,
    update_smoke_root: str | Path,
    candidate_root: str | Path,
    quasi_real_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
    collector_replay_runner: CollectorReplayRunner | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    _install_model_explorer_path(repo)
    update_root = Path(update_smoke_root).resolve()
    candidate = Path(candidate_root).resolve()
    quasi_real = Path(quasi_real_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _validate_config(config)
    paths = _output_paths(output, config)
    inputs = config.get("input_files", {})

    reason_codes: list[str] = []
    update_summary_path = update_root / inputs.get(
        "update_smoke_summary",
        "return-aligned-guarded-ppo-update-smoke-summary.json",
    )
    update_summary = _load_json_required(update_summary_path, reason_codes, NEXT_INPUT_INVALID)
    teacher_summary_path = _resolve_input_path(
        update_root,
        inputs.get(
            "teacher_following_summary",
            "post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json",
        ),
        repo,
    )
    teacher_summary = _load_json_required(teacher_summary_path, reason_codes, NEXT_INPUT_INVALID)
    decisions_path = _resolve_input_path(
        update_root,
        teacher_summary.get(
            "decisions_path",
            inputs.get(
                "teacher_following_decisions",
                "post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-decisions.jsonl",
            ),
        ),
        repo,
    )
    decisions = _read_jsonl(decisions_path)
    long_horizon_path = _resolve_input_path(
        update_root,
        update_summary.get(
            "post_update_long_horizon_summary",
            inputs.get("long_horizon_summary", "post_update_long_horizon/long-horizon-teacher-skill-contract-summary.json"),
        ),
        repo,
    )
    long_horizon = _load_json_required(long_horizon_path, reason_codes, NEXT_INPUT_INVALID)
    return_aligned_replay_path = _resolve_input_path(
        update_root,
        update_summary.get(
            "post_update_return_aligned_replay_summary",
            inputs.get("return_aligned_replay_summary", "post_update_return_aligned_replay/return-aligned-collector-summary.json"),
        ),
        repo,
    )
    return_aligned_replay = _load_json_required(return_aligned_replay_path, reason_codes, NEXT_INPUT_INVALID)

    _validate_input_summaries(
        update_summary=update_summary,
        teacher_summary=teacher_summary,
        long_horizon=long_horizon,
        return_aligned_replay=return_aligned_replay,
        reason_codes=reason_codes,
    )

    scenario_groups = _quasi_real_scenario_groups_by_id(quasi_real, config=config, repo_root=repo)
    slice_by_scenario = _slice_by_scenario_id(quasi_real, config=config)
    policy_evaluator = _PolicyEvaluator.from_candidate_root(candidate, config=config, repo_root=repo)
    steps = _step_records(
        decisions,
        scenario_groups=scenario_groups,
        slice_by_scenario=slice_by_scenario,
        policy_evaluator=policy_evaluator,
        config=config,
    )
    episodes = _episode_records(steps, config=config)
    _apply_episode_returns(episodes, config=config)

    counters = _counters(steps, episodes)
    rejection_records = [step for step in steps if step.get("rejection_reason_codes")]
    reward_records = [
        {
            "episode_id": step["episode_id"],
            "step_index": step["step_index"],
            "reward": step["reward"],
            "discounted_return": step["discounted_return"],
            "advantage": step["advantage"],
            "components": step["reward_components"],
            "reason_codes": step["rejection_reason_codes"],
        }
        for step in steps
    ]

    collector_replay = _run_collector_replay(
        runner=collector_replay_runner,
        update_root=update_root,
        output_root=output,
        config=config,
        repo_root=repo,
        candidate_root=candidate,
        quasi_real_root=quasi_real,
    )
    _apply_validation(
        reason_codes,
        counters,
        config=config,
        teacher_summary=teacher_summary,
        long_horizon=long_horizon,
        collector_replay=collector_replay,
    )

    status = "failed" if reason_codes else "passed"
    _write_jsonl(paths["steps"], steps)
    _write_jsonl(paths["episodes"], episodes)
    _write_json(paths["rejection_report"], _rejection_report(rejection_records))
    _write_json(paths["reward_audit"], _reward_audit(reward_records, counters))
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": status,
        "reason_codes": _dedup(reason_codes),
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "update_smoke_root": _display_path(update_root, repo),
        "candidate_root": _display_path(candidate, repo),
        "quasi_real_root": _display_path(quasi_real, repo),
        "output_root": _display_path(output, repo),
        "update_smoke_summary": _display_path(update_summary_path, repo),
        "teacher_following_summary": _display_path(teacher_summary_path, repo),
        "teacher_following_decisions": _display_path(decisions_path, repo),
        "long_horizon_summary": _display_path(long_horizon_path, repo),
        "return_aligned_replay_summary": _display_path(return_aligned_replay_path, repo),
        "episodes": _display_path(paths["episodes"], repo),
        "steps": _display_path(paths["steps"], repo),
        "rejection_report": _display_path(paths["rejection_report"], repo),
        "reward_audit": _display_path(paths["reward_audit"], repo),
        "summary": _display_path(paths["summary"], repo),
        "horizon": _int_value(config.get("horizon"), 3),
        "discount_factor": _float_value(config.get("discount_factor"), 0.99),
        "episode_count": len(episodes),
        "step_count": len(steps),
        "trainable_transition_count": counters["trainable_transition_count"],
        "ppo_trainable_transition_count": counters["trainable_transition_count"],
        "diagnostic_transition_count": counters["diagnostic_transition_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "teacher_agreement_rate": _float_value(teacher_summary.get("teacher_agreement_rate"), 0.0),
        "quasi_real_collector_replay_status": collector_replay.get("status"),
        "quasi_real_collector_replay_reason_codes": _string_list(collector_replay.get("reason_codes")),
        "quasi_real_collector_replay_trainable_transition_count": _int_value(
            collector_replay.get("ppo_trainable_transition_count")
        ),
        "post_pilot_long_horizon_status": long_horizon.get("status"),
        "post_pilot_long_horizon_verdict": long_horizon.get("verdict"),
        "uses_multistep_discounted_return": True,
        "not_single_step_best_action": True,
        "guarded_rollout_pilot_passed": status == "passed",
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "experimental_checkpoint": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(paths["summary"], summary)
    return summary


def _step_records(
    decisions: list[dict[str, Any]],
    *,
    scenario_groups: dict[str, dict[str, Any]],
    slice_by_scenario: dict[str, dict[str, Any]],
    policy_evaluator: _PolicyEvaluator,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    horizon = max(1, _int_value(config.get("horizon"), 3))
    steps: list[dict[str, Any]] = []
    for ordinal, decision in enumerate(decisions):
        episode_index = ordinal // horizon
        step_index = ordinal % horizon
        split = _decision_split(decision, slice_by_scenario)
        original_source = str(decision.get("controlled_choice_source") or "")
        policy_controlled = bool(decision.get("policy_takes_control")) and original_source in {
            "policy_teacher_aligned",
            "policy_safe_disagreement",
            "policy",
        }
        controlled_source = "policy" if policy_controlled else original_source or "teacher_fallback"
        action_index = _optional_int(decision.get("controlled_action_index"))
        if action_index is None:
            action_index = _optional_int(decision.get("teacher_action_index"))
        observation = _observation_from_step_or_scenario(
            {
                "observation": decision.get("observation"),
                "context_id": decision.get("context_id"),
                "policy_selected_context_id": decision.get("context_id"),
                "raw_policy_selected_context_id": decision.get("context_id"),
            },
            scenario_groups.get(str(decision.get("scenario_id") or "")),
            action_index,
        )
        log_prob = _float_or_none(decision.get("policy_action_log_prob"))
        value = _float_or_none(decision.get("policy_value"))
        if observation is not None and action_index is not None and (log_prob is None or value is None):
            evaluated = policy_evaluator.evaluate(observation, action_index)
            log_prob = evaluated.get("log_prob", log_prob)
            value = evaluated.get("value", value)
        reward, reward_components, reward_reasons = _reward(decision, policy_controlled=policy_controlled, config=config)
        rejection_reasons = _rejection_reasons(
            decision,
            split=split,
            controlled_source=controlled_source,
            observation=observation,
            action_index=action_index,
            log_prob=log_prob,
            value=value,
            reward=reward,
            config=config,
        )
        rejection_reasons.extend(reward_reasons)
        gate_reasons = _string_list(decision.get("gate_reason_codes"))
        steps.append(
            {
                "schema_version": STEP_SCHEMA_VERSION,
                "episode_id": f"quasi-real-episode-{episode_index:04d}",
                "step_index": step_index,
                "decision_index": ordinal,
                "context_id": decision.get("context_id"),
                "scenario_id": decision.get("scenario_id"),
                "scenario_family": decision.get("roi_group") or decision.get("scenario_group"),
                "split": split,
                "raw_policy_action_index": decision.get("raw_policy_action_index"),
                "teacher_action_index": decision.get("teacher_action_index"),
                "controlled_action_index": action_index,
                "controlled_choice_source": controlled_source,
                "controlled_choice_detail": original_source,
                "policy_takes_control": policy_controlled,
                "gate_reason_codes": gate_reasons,
                "controlled_regression_reason_codes": _controlled_regression_reasons(gate_reasons),
                "ppo_trainable_candidate": _ppo_trainable_candidate(
                    split=split,
                    controlled_source=controlled_source,
                    gate_reasons=gate_reasons,
                    config=config,
                ),
                "ppo_trainable": False,
                "diagnostic_only": True,
                "rejection_reason_codes": _dedup(rejection_reasons),
                "observation": _observation_payload(observation),
                "missing_observation": observation is None,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "discounted_return": None,
                "advantage": None,
                "done": False,
                "reward_components": reward_components,
                "path_cost_delta": decision.get("path_cost_delta"),
                "risk_delta": decision.get("risk_delta"),
            }
        )
    return steps


def _episode_records(steps: list[dict[str, Any]], *, config: dict[str, Any]) -> list[dict[str, Any]]:
    horizon = max(1, _int_value(config.get("horizon"), 3))
    episodes: list[dict[str, Any]] = []
    for index in range(0, len(steps), horizon):
        episode_steps = steps[index : index + horizon]
        if not episode_steps:
            continue
        episode_steps[-1]["done"] = True
        episodes.append(
            {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "episode_id": episode_steps[0]["episode_id"],
                "horizon": horizon,
                "step_count": len(episode_steps),
                "splits": sorted({str(step.get("split")) for step in episode_steps}),
                "steps": episode_steps,
            }
        )
    return episodes


def _apply_episode_returns(episodes: list[dict[str, Any]], *, config: dict[str, Any]) -> None:
    discount = _float_value(config.get("discount_factor"), 0.99)
    for episode in episodes:
        steps = episode["steps"]
        running = 0.0
        running_finite = True
        for step in reversed(steps):
            reward = _float_or_none(step.get("reward"))
            if reward is None or not math.isfinite(reward) or not running_finite:
                running_finite = False
                step["discounted_return"] = float("nan")
            else:
                running = reward + discount * running
                step["discounted_return"] = running
            value = _float_or_none(step.get("value"))
            if value is None or not _is_finite(step.get("discounted_return")):
                step["advantage"] = float("nan")
            else:
                step["advantage"] = float(step["discounted_return"]) - value
            if not _is_finite(step.get("discounted_return")):
                step["rejection_reason_codes"] = _dedup(step["rejection_reason_codes"] + ["non_finite_return"])
            if not _is_finite(step.get("advantage")):
                step["rejection_reason_codes"] = _dedup(step["rejection_reason_codes"] + ["non_finite_advantage"])
            step["ppo_trainable"] = bool(step["ppo_trainable_candidate"] and not step["rejection_reason_codes"])
            step["diagnostic_only"] = not step["ppo_trainable"]
        finite_returns = [float(step["discounted_return"]) for step in steps if _is_finite(step.get("discounted_return"))]
        episode["discounted_episode_return"] = finite_returns[0] if finite_returns else float("nan")
        episode["ppo_trainable_transition_count"] = sum(1 for step in steps if step["ppo_trainable"])
        episode["diagnostic_transition_count"] = sum(1 for step in steps if step["diagnostic_only"])


def _reward(
    decision: dict[str, Any],
    *,
    policy_controlled: bool,
    config: dict[str, Any],
) -> tuple[float, dict[str, float], list[str]]:
    reward_config = config.get("reward", {}) if isinstance(config.get("reward"), dict) else {}
    values = [_float_or_none(decision.get("path_cost_delta")), _float_or_none(decision.get("risk_delta"))]
    if any(value is not None and not math.isfinite(value) for value in values):
        return float("nan"), {}, ["non_finite_reward"]
    detail = str(decision.get("controlled_choice_source") or "")
    gate_reasons = _string_list(decision.get("gate_reason_codes"))
    components = {
        "teacher_following_bonus": (
            float(reward_config.get("teacher_following_bonus", 1.0))
            if policy_controlled and detail in {"policy_teacher_aligned", "policy"}
            else 0.0
        ),
        "safe_disagreement_bonus": (
            float(reward_config.get("safe_disagreement_bonus", 1.0))
            if policy_controlled and detail == "policy_safe_disagreement"
            else 0.0
        ),
        "gate_penalty": -float(reward_config.get("gate_regression_penalty", 1.0)) if gate_reasons else 0.0,
    }
    reward = sum(components.values())
    if not math.isfinite(reward):
        return reward, components, ["non_finite_reward"]
    return reward, components, []


def _rejection_reasons(
    decision: dict[str, Any],
    *,
    split: str,
    controlled_source: str,
    observation: Any,
    action_index: int | None,
    log_prob: float | None,
    value: float | None,
    reward: float,
    config: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    gate_reasons = _string_list(decision.get("gate_reason_codes"))
    filter_config = config.get("trainable_filter", {}) if isinstance(config.get("trainable_filter"), dict) else {}
    allowed_splits = {str(item) for item in filter_config.get("splits", ["train"])}
    allowed_sources = {str(item) for item in filter_config.get("controlled_choice_sources", ["policy"])}
    if split not in allowed_splits:
        reasons.append("non_train_split")
    if controlled_source not in allowed_sources:
        reasons.append("controlled_choice_source_not_trainable")
    if bool(filter_config.get("require_empty_gate_reason_codes", True)) and gate_reasons:
        reasons.append("gate_reason_codes_present")
    if observation is None:
        reasons.append("missing_observation")
    if action_index is None:
        reasons.append("controlled_action_index_missing")
    if log_prob is None:
        reasons.append("missing_log_prob")
    if value is None:
        reasons.append("missing_value")
    if not math.isfinite(float(reward)):
        reasons.append("non_finite_reward")
    if observation is not None:
        mask = tuple(bool(item) for item in getattr(observation, "action_mask", ()))
        if not any(mask):
            reasons.append("empty_action_mask")
        if action_index is None or action_index < 0 or action_index >= len(mask) or not mask[action_index]:
            reasons.append("invalid_action_mask")
    return _dedup(reasons)


def _ppo_trainable_candidate(
    *,
    split: str,
    controlled_source: str,
    gate_reasons: list[str],
    config: dict[str, Any],
) -> bool:
    filter_config = config.get("trainable_filter", {}) if isinstance(config.get("trainable_filter"), dict) else {}
    return (
        split in {str(item) for item in filter_config.get("splits", ["train"])}
        and controlled_source in {str(item) for item in filter_config.get("controlled_choice_sources", ["policy"])}
        and (not bool(filter_config.get("require_empty_gate_reason_codes", True)) or not gate_reasons)
    )


def _counters(steps: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> Counter[str]:
    counters: Counter[str] = Counter()
    for step in steps:
        if step["ppo_trainable"]:
            counters["trainable_transition_count"] += 1
            if step["split"] == "validation":
                counters["validation_trainable_count"] += 1
            if step["split"] == "test":
                counters["test_trainable_count"] += 1
            if step["controlled_choice_source"] in {"source_fallback", "teacher_fallback"}:
                counters["source_fallback_trainable_count"] += 1
            if step["controlled_choice_source"] == "teacher_fallback":
                counters["teacher_fallback_trainable_count"] += 1
            if step.get("gate_reason_codes"):
                counters["non_empty_gate_reason_trainable_count"] += 1
        else:
            counters["diagnostic_transition_count"] += 1
        if step.get("missing_observation"):
            counters["missing_observation_count"] += 1
        if step.get("log_prob") is None:
            counters["missing_log_prob_count"] += 1
        if step.get("value") is None:
            counters["missing_value_count"] += 1
        if not _is_finite(step.get("reward")):
            counters["non_finite_reward_count"] += 1
        if not _is_finite(step.get("discounted_return")):
            counters["non_finite_return_count"] += 1
        if not _is_finite(step.get("advantage")):
            counters["non_finite_advantage_count"] += 1
        reasons = set(step.get("controlled_regression_reason_codes") or [])
        if reasons:
            counters["controlled_regression_count"] += 1
        if "safety_regression" in reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_violation" in reasons or "contract_regression" in reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in reasons or "risk_regression" in reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in reasons:
            counters["controlled_source_selection_regression_count"] += 1
    counters["episode_count"] = len(episodes)
    counters["step_count"] = len(steps)
    return counters


def _apply_validation(
    reason_codes: list[str],
    counters: Counter[str],
    *,
    config: dict[str, Any],
    teacher_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    collector_replay: dict[str, Any],
) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    if counters["episode_count"] < _int_value(validation.get("min_episode_count"), 36):
        _append_reason(reason_codes, "quasi_real_guarded_ppo_rollout_episode_count_below_threshold")
    if counters["step_count"] < _int_value(validation.get("min_step_count"), 108):
        _append_reason(reason_codes, "quasi_real_guarded_ppo_rollout_step_count_below_threshold")
    if counters["trainable_transition_count"] < _int_value(validation.get("min_trainable_transition_count"), 24):
        _append_reason(reason_codes, "quasi_real_guarded_ppo_rollout_trainable_transition_count_below_threshold")
    if _float_value(teacher_summary.get("teacher_agreement_rate"), 0.0) < float(validation.get("min_teacher_agreement_rate", 0.9)):
        _append_reason(reason_codes, "quasi_real_guarded_ppo_rollout_teacher_agreement_below_threshold")
    if collector_replay.get("status") != "passed" or _string_list(collector_replay.get("reason_codes")):
        _append_reason(reason_codes, "quasi_real_guarded_ppo_rollout_collector_replay_not_passed")
    if _int_value(collector_replay.get("ppo_trainable_transition_count")) < _int_value(
        validation.get("min_quasi_real_collector_replay_trainable_transition_count"),
        24,
    ):
        _append_reason(reason_codes, "quasi_real_guarded_ppo_rollout_collector_replay_trainable_below_threshold")
    if long_horizon.get("status") != "passed" or long_horizon.get("verdict") != "long_horizon_teacher_skill_contract_aligned":
        _append_reason(reason_codes, "quasi_real_guarded_ppo_rollout_long_horizon_not_aligned")
    for field in (
        "validation_trainable_count",
        "test_trainable_count",
        "source_fallback_trainable_count",
        "teacher_fallback_trainable_count",
        "non_empty_gate_reason_trainable_count",
        "missing_observation_count",
        "missing_log_prob_count",
        "missing_value_count",
    ):
        if counters[field]:
            _append_reason(reason_codes, NEXT_CONTRACT_INVALID)
    if counters["non_finite_reward_count"]:
        _append_reason(reason_codes, NEXT_CONTRACT_INVALID)
    if counters["non_finite_return_count"] or counters["non_finite_advantage_count"]:
        _append_reason(reason_codes, NEXT_NON_FINITE_RETURN)
    if counters["controlled_regression_count"]:
        _append_reason(reason_codes, NEXT_CONTROLLED_REGRESSION)


def _validate_input_summaries(
    *,
    update_summary: dict[str, Any],
    teacher_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    return_aligned_replay: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if update_summary.get("status") != "passed" or _string_list(update_summary.get("reason_codes")):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if update_summary.get("post_update_gates_evaluated") is not True:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if teacher_summary.get("status") != "passed" or _string_list(teacher_summary.get("reason_codes")):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if long_horizon.get("status") != "passed" or long_horizon.get("verdict") != "long_horizon_teacher_skill_contract_aligned":
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if return_aligned_replay.get("status") != "passed" or _string_list(return_aligned_replay.get("reason_codes")):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)


def _run_collector_replay(
    *,
    runner: CollectorReplayRunner | None,
    update_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    candidate_root: Path,
    quasi_real_root: Path,
) -> dict[str, Any]:
    replay_config = config.get("collector_replay", {}) if isinstance(config.get("collector_replay"), dict) else {}
    if runner is not None:
        return runner()
    collector_config_path = _resolve_path(
        replay_config.get("config", "configs/quasi_real_ppo_collector_dry_run_v1.json"),
        repo_root,
    )
    collector_config = _load_json(collector_config_path)
    replay_root = output_root / replay_config.get("output_root", "quasi_real_collector_replay")
    return run_quasi_real_ppo_collector_dry_run(
        guarded_teacher_following_root=update_root / "post_update_quasi_real_teacher_following",
        candidate_root=candidate_root,
        quasi_real_root=quasi_real_root,
        output_root=replay_root,
        config=collector_config,
        repo_root=repo_root,
    )


def _controlled_regression_reasons(gate_reasons: list[str]) -> list[str]:
    return [reason for reason in gate_reasons if reason in CONTROLLED_REGRESSION_REASONS]


def _observation_payload(observation: Any) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "candidate_feature_names": list(observation.candidate_feature_names),
        "candidate_features": [list(row) for row in observation.candidate_features],
        "global_feature_names": list(observation.global_feature_names),
        "global_features": list(observation.global_features),
        "action_mask": list(observation.action_mask),
        "candidate_cells": [list(cell) if cell is not None else None for cell in observation.candidate_cells],
        "candidate_missing_feature_names": [list(row) for row in observation.candidate_missing_feature_names],
        "candidate_missing_indicator_names": list(observation.candidate_missing_indicator_names),
        "candidate_missing_indicators": [list(row) for row in observation.candidate_missing_indicators],
    }


def _rejection_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(reason for record in records for reason in record.get("rejection_reason_codes", []))
    return {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "rejected_step_count": len(records),
        "reason_code_counts": dict(sorted(counts.items())),
        "records": records,
    }


def _reward_audit(records: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    finite_rewards = [float(record["reward"]) for record in records if _is_finite(record.get("reward"))]
    finite_returns = [float(record["discounted_return"]) for record in records if _is_finite(record.get("discounted_return"))]
    finite_advantages = [float(record["advantage"]) for record in records if _is_finite(record.get("advantage"))]
    return {
        "schema_version": REWARD_AUDIT_SCHEMA_VERSION,
        "record_count": len(records),
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "reward_min": min(finite_rewards) if finite_rewards else 0.0,
        "reward_max": max(finite_rewards) if finite_rewards else 0.0,
        "return_min": min(finite_returns) if finite_returns else 0.0,
        "return_max": max(finite_returns) if finite_returns else 0.0,
        "advantage_min": min(finite_advantages) if finite_advantages else 0.0,
        "advantage_max": max(finite_advantages) if finite_advantages else 0.0,
        "records": records,
    }


def _output_paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("output_files", {}) if isinstance(config.get("output_files"), dict) else {}
    return {
        "summary": output_root / outputs.get("summary", "quasi-real-guarded-ppo-rollout-pilot-summary.json"),
        "episodes": output_root / outputs.get("episodes", "quasi-real-guarded-ppo-rollout-episodes.jsonl"),
        "steps": output_root / outputs.get("steps", "quasi-real-guarded-ppo-rollout-steps.jsonl"),
        "rejection_report": output_root / outputs.get("rejection_report", "quasi-real-guarded-ppo-rollout-rejection-report.json"),
        "reward_audit": output_root / outputs.get("reward_audit", "quasi-real-guarded-ppo-rollout-reward-audit.json"),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "validation", "trainable_filter", "reward"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"{section} must be an object")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    return _load_json(path)


def _load_json_required(path: Path, reason_codes: list[str], reason: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, reason)
        return {}
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(_json_safe(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _resolve_input_path(root: Path, value: Any, repo_root: Path) -> Path:
    if not value:
        return root
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = root / path
    return candidate if candidate.exists() or not (repo_root / path).exists() else repo_root / path


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _dedup(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _next_required_change(reason_codes: list[str]) -> str:
    for reason in (
        NEXT_INPUT_INVALID,
        NEXT_CONTRACT_INVALID,
        NEXT_NON_FINITE_RETURN,
        NEXT_CONTROLLED_REGRESSION,
    ):
        if reason in reason_codes:
            return reason
    return reason_codes[0] if reason_codes else NEXT_CONTRACT_INVALID


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finite(value: Any) -> bool:
    parsed = _float_or_none(value)
    return parsed is not None and math.isfinite(parsed)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
