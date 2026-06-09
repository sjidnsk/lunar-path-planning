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
    from scripts.run_fresh_holdout_policy_candidate_evaluation import (
        _candidate_features,
        _global_features,
        _missing_indicators,
    )
    from scripts.run_scenario_disjoint_policy_rollout_evaluation import (
        _candidate_action_mask_valid,
        _collect_holdout_scenarios,
        _display_path,
    )
except ModuleNotFoundError:
    from run_fresh_holdout_policy_candidate_evaluation import (
        _candidate_features,
        _global_features,
        _missing_indicators,
    )
    from run_scenario_disjoint_policy_rollout_evaluation import (
        _candidate_action_mask_valid,
        _collect_holdout_scenarios,
        _display_path,
    )


CONFIG_SCHEMA_VERSION = "ppo-rollout-collector-dry-run-config/v1"
SUMMARY_SCHEMA_VERSION = "ppo-rollout-collector-summary/v1"
REJECTION_SCHEMA_VERSION = "ppo-rollout-rejection-report/v1"
REWARD_AUDIT_SCHEMA_VERSION = "ppo-rollout-reward-audit/v1"
TRANSITION_RECORD_SCHEMA_VERSION = "ppo-rollout-transition-record/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize policy-gated sequential canary steps as PPO dry-run rollout input.")
    parser.add_argument("--sequential-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-root", default=None)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    summary = run_ppo_rollout_collector_dry_run(
        sequential_root=_resolve_path(args.sequential_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        candidate_root=None if args.candidate_root is None else _resolve_path(args.candidate_root, repo_root),
        config=_load_json(_resolve_path(args.config, repo_root)),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "ppo_trainable_transition_count": summary["ppo_trainable_transition_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_ppo_rollout_collector_dry_run(
    *,
    sequential_root: Path,
    output_root: Path,
    candidate_root: Path | None,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    _install_model_explorer_path(repo_root)
    from model_explorer.policy.dataset import validate_rollout_dataset
    from model_explorer.policy.rollout_io import write_rollout_episodes_jsonl

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, config)
    steps = _read_jsonl(sequential_root / config.get("input_files", {}).get("steps", "policy-gated-sequential-canary-steps.jsonl"))
    sequential_summary = _load_json_if_exists(
        sequential_root / config.get("input_files", {}).get("sequential_summary", "policy-gated-sequential-canary-rollout-summary.json")
    )
    scenario_groups = _scenario_groups_by_id(sequential_root, repo_root)
    policy_evaluator = _PolicyEvaluator.from_candidate_root(candidate_root, config=config, repo_root=repo_root)

    reason_codes: list[str] = []
    transition_records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    episodes_by_id: dict[str, list[Any]] = defaultdict(list)
    counters: Counter[str] = Counter()

    for index, step in enumerate(steps):
        record, transition = _transition_from_step(
            step,
            scenario_group=scenario_groups.get(str(step.get("scenario_id", ""))),
            policy_evaluator=policy_evaluator,
            config=config,
            step_ordinal=index,
        )
        transition_records.append(record)
        reward_records.append(record["reward_audit"])
        for key, value in record["counter_deltas"].items():
            counters[key] += int(value)
        if record["rejection_reason_codes"]:
            rejection_records.append(record)
        if transition is not None:
            episodes_by_id[str(step.get("episode_id", f"episode-{index:04d}"))].append(transition)

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
    if sequential_summary.get("status") not in {None, "passed"}:
        _append_reason(reason_codes, "sequential_rollout_not_passed")

    write_rollout_episodes_jsonl(paths["episodes"], tuple(episodes))
    _write_jsonl(paths["transitions"], transition_records)
    _write_json(paths["rejection_report"], _rejection_report(rejection_records))
    _write_json(paths["reward_audit"], _reward_audit(reward_records, counters))

    status = "failed" if reason_codes else "passed"
    current_git = _git_snapshot(repo_root)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "batch_root": _display_path(output_root, repo_root),
        "sequential_root": _display_path(sequential_root, repo_root),
        "episode_count": _int_value(sequential_summary.get("episode_count"), len({step.get("episode_id") for step in steps})),
        "step_count": _int_value(sequential_summary.get("step_count"), len(steps)),
        "materialized_episode_count": len(episodes),
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "diagnostic_transition_count": counters["diagnostic_transition_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "invalid_action_mask_count": counters["invalid_action_mask_count"],
        "empty_action_mask_count": counters["empty_action_mask_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "state_continuity_violation_count": _int_value(sequential_summary.get("state_continuity_violation_count"), counters["state_continuity_violation_count"]),
        "fallback_or_open_grid_count": _int_value(sequential_summary.get("fallback_or_open_grid_count"), 0),
        "safety_regression_count": _int_value(sequential_summary.get("cumulative_safety_regression_count"), 0),
        "contract_violation_count": _int_value(sequential_summary.get("cumulative_contract_violation_count"), 0),
        "path_cost_regression_count": _int_value(sequential_summary.get("cumulative_path_cost_regression_count"), 0),
        "risk_regression_count": _int_value(sequential_summary.get("cumulative_risk_regression_count"), 0),
        "source_selection_regression_count": _int_value(sequential_summary.get("cumulative_source_selection_regression_count"), 0),
        "dataset_summary": dataset_summary,
        "dataset_error": dataset_error,
        "episodes": _display_path(paths["episodes"], repo_root),
        "transitions": _display_path(paths["transitions"], repo_root),
        "rejection_report": _display_path(paths["rejection_report"], repo_root),
        "reward_audit": _display_path(paths["reward_audit"], repo_root),
        "summary": _display_path(paths["summary"], repo_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "experimental_collector_dry_run": True,
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


def _transition_from_step(
    step: dict[str, Any],
    *,
    scenario_group: dict[str, Any] | None,
    policy_evaluator: "_PolicyEvaluator",
    config: dict[str, Any],
    step_ordinal: int,
) -> tuple[dict[str, Any], Any | None]:
    from model_explorer.policy.rollout import RolloutInfo, RolloutTransition

    counter_deltas: Counter[str] = Counter()
    rejection_reasons: list[str] = []
    controlled_source = str(step.get("controlled_choice_source") or "")
    ppo_trainable = (
        controlled_source == "policy"
        and bool(step.get("canary_gate_passed", step.get("decision_class") == "canary_accepted_policy_choice"))
        and not _string_list(step.get("canary_rejection_reason_codes"))
        and not _string_list(step.get("controlled_regression_reason_codes"))
        and not _string_list(step.get("raw_policy_regression_reason_codes"))
    )
    if not ppo_trainable:
        counter_deltas["diagnostic_transition_count"] += 1
    controlled_action_index = _controlled_action_index(step)
    observation = _observation_from_step_or_scenario(step, scenario_group, controlled_action_index)
    if observation is None:
        rejection_reasons.append("ppo_observation_missing")
        counter_deltas["invalid_action_mask_count"] += 1
    if controlled_action_index is None:
        rejection_reasons.append("controlled_action_index_missing")

    log_prob = _float_or_none(step.get("policy_action_log_prob"))
    value = _float_or_none(step.get("policy_value"))
    if observation is not None and controlled_action_index is not None and (log_prob is None or value is None):
        evaluated = policy_evaluator.evaluate(observation, controlled_action_index)
        log_prob = evaluated.get("log_prob", log_prob)
        value = evaluated.get("value", value)
    if ppo_trainable and log_prob is None:
        counter_deltas["missing_log_prob_count"] += 1
        rejection_reasons.append("ppo_log_prob_missing")
    if ppo_trainable and value is None:
        counter_deltas["missing_value_count"] += 1
        rejection_reasons.append("ppo_value_missing")

    reward, reward_components, reward_reasons = _compute_reward(step, config=config)
    if reward_reasons:
        rejection_reasons.extend(reward_reasons)
        counter_deltas["non_finite_reward_count"] += reward_reasons.count("non_finite_reward")

    invalid_mask = True
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

    if ppo_trainable and controlled_source == "source_fallback":
        counter_deltas["source_fallback_trainable_count"] += 1
    transition = None
    if ppo_trainable and not rejection_reasons and observation is not None and controlled_action_index is not None:
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
                selected_cell=_cell_tuple(step.get("controlled_execution_goal_cell")),
                coverage_rate_delta=0.0,
                path_cost=abs(_finite_or_zero(step.get("policy_selected_path_cost_delta"))),
                risk=abs(_finite_or_zero(step.get("policy_selected_risk_delta"))),
                failure_reason=None,
                final_coverage_rate=None,
                total_cost=0.0,
                failure_count=0,
                replan_count=0,
                extra={
                    "ppo_trainable": True,
                    "episode_id": step.get("episode_id"),
                    "step_index": step.get("step_index"),
                    "context_id": step.get("context_id"),
                    "scenario_family": step.get("scenario_group"),
                    "raw_policy_action_index": step.get("raw_policy_selected_action_index"),
                    "source_action_index": step.get("source_selected_action_index"),
                    "controlled_action_index": controlled_action_index,
                    "controlled_choice_source": controlled_source,
                    "execution_goal_cell": step.get("controlled_execution_goal_cell"),
                    "path_cost_delta": step.get("policy_selected_path_cost_delta"),
                    "risk_delta": step.get("policy_selected_risk_delta"),
                    "reward_components": reward_components,
                    "gate_reason_codes": [],
                },
            ),
        )

    record = {
        "schema_version": TRANSITION_RECORD_SCHEMA_VERSION,
        "episode_id": step.get("episode_id"),
        "step_index": step.get("step_index", step_ordinal),
        "scenario_id": step.get("scenario_id"),
        "scenario_family": step.get("scenario_group"),
        "context_id": step.get("context_id"),
        "controlled_choice_source": controlled_source,
        "controlled_action_index": controlled_action_index,
        "ppo_trainable": bool(transition is not None),
        "diagnostic_only": transition is None,
        "rejection_reason_codes": _dedup(rejection_reasons),
        "reward": reward,
        "reward_components": reward_components,
        "reward_audit": {
            "episode_id": step.get("episode_id"),
            "step_index": step.get("step_index", step_ordinal),
            "reward": reward,
            "components": reward_components,
            "reason_codes": reward_reasons,
        },
        "counter_deltas": dict(counter_deltas),
    }
    return record, transition


def _observation_from_step_or_scenario(step: dict[str, Any], scenario_group: dict[str, Any] | None, action_index: int | None):
    from model_explorer.policy.features import (
        CANDIDATE_FEATURE_NAMES,
        GLOBAL_FEATURE_NAMES,
        MISSING_INDICATOR_NAMES,
        PolicyObservation,
    )

    payload = step.get("observation")
    if isinstance(payload, dict):
        return _observation_from_payload(payload)
    if not scenario_group or action_index is None:
        return None
    candidates = scenario_group.get("candidates") if isinstance(scenario_group.get("candidates"), list) else []
    if not candidates:
        return None
    by_action: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_action = _candidate_action(candidate)
        if candidate_action is None:
            continue
        current = by_action.get(candidate_action)
        if current is None or _candidate_preferred(candidate, current, step=step, selected_action=action_index):
            by_action[candidate_action] = candidate
    if action_index not in by_action:
        return None
    action_count = max(max(by_action), action_index) + 1
    candidate_features = []
    candidate_cells = []
    masks = []
    missing = []
    for slot in range(action_count):
        candidate = by_action.get(slot)
        if candidate is None:
            candidate_features.append(tuple(0.0 for _ in CANDIDATE_FEATURE_NAMES))
            candidate_cells.append(None)
            masks.append(False)
            missing.append(tuple(1.0 for _ in MISSING_INDICATOR_NAMES))
            continue
        candidate_features.append(tuple(_candidate_features(candidate)))
        candidate_cells.append(_cell_tuple(candidate.get("execution_goal_cell") or candidate.get("policy_target_cell")))
        masks.append(_candidate_action_mask_valid(candidate))
        missing.append(tuple(_missing_indicators(candidate)))
    return PolicyObservation(
        candidate_feature_names=CANDIDATE_FEATURE_NAMES,
        candidate_features=tuple(candidate_features),
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=tuple(_global_features(candidates)),
        action_mask=tuple(masks),
        candidate_cells=tuple(candidate_cells),
        candidate_missing_feature_names=tuple(() for _ in range(action_count)),
        candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
        candidate_missing_indicators=tuple(missing),
    )


def _observation_from_payload(payload: dict[str, Any]):
    from model_explorer.policy.features import PolicyObservation

    return PolicyObservation(
        candidate_feature_names=tuple(str(value) for value in payload["candidate_feature_names"]),
        candidate_features=tuple(tuple(float(value) for value in row) for row in payload["candidate_features"]),
        global_feature_names=tuple(str(value) for value in payload["global_feature_names"]),
        global_features=tuple(float(value) for value in payload["global_features"]),
        action_mask=tuple(bool(value) for value in payload["action_mask"]),
        candidate_cells=tuple(_cell_tuple(value) for value in payload["candidate_cells"]),
        candidate_missing_feature_names=tuple(tuple(str(value) for value in row) for row in payload.get("candidate_missing_feature_names", []))
        or tuple(() for _ in payload["candidate_features"]),
        candidate_missing_indicator_names=tuple(str(value) for value in payload.get("candidate_missing_indicator_names", [])),
        candidate_missing_indicators=tuple(tuple(float(value) for value in row) for row in payload.get("candidate_missing_indicators", []))
        or tuple(() for _ in payload["candidate_features"]),
    )


def _compute_reward(step: dict[str, Any], *, config: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    reward_config = config.get("reward", {})
    reasons: list[str] = []
    path_delta = _float_or_none(step.get("policy_selected_path_cost_delta"))
    risk_delta = _float_or_none(step.get("policy_selected_risk_delta"))
    utility_delta = _float_or_none(step.get("policy_selected_utility_delta"))
    values = [value for value in (path_delta, risk_delta, utility_delta) if value is not None]
    if any(not math.isfinite(value) for value in values):
        return float("nan"), {}, ["non_finite_reward"]
    components = {
        "better_choice_bonus": float(reward_config.get("better_choice_bonus", 1.0))
        if step.get("accepted_choice_value_class") == "accepted_better"
        else 0.0,
        "path_improvement": max(0.0, -(path_delta or 0.0)) * float(reward_config.get("path_improvement_weight", 1.0)),
        "risk_improvement": max(0.0, -(risk_delta or 0.0)) * float(reward_config.get("risk_improvement_weight", 10.0)),
        "utility_improvement": max(0.0, utility_delta or 0.0) * float(reward_config.get("utility_improvement_weight", 1.0)),
        "gate_penalty": -float(reward_config.get("gate_regression_penalty", 1.0))
        if _string_list(step.get("canary_rejection_reason_codes"))
        or _string_list(step.get("controlled_regression_reason_codes"))
        or _string_list(step.get("raw_policy_regression_reason_codes"))
        else 0.0,
    }
    reward = sum(components.values())
    if not math.isfinite(reward):
        reasons.append("non_finite_reward")
    return reward, components, reasons


class _PolicyEvaluator:
    def __init__(self, network=None) -> None:
        self.network = network

    @classmethod
    def from_candidate_root(cls, candidate_root: Path | None, *, config: dict[str, Any], repo_root: Path) -> "_PolicyEvaluator":
        if candidate_root is None:
            return cls()
        checkpoint_name = config.get("input_files", {}).get("checkpoint", "experimental-hybrid-policy-candidate.pt")
        checkpoint_path = candidate_root / checkpoint_name
        if not checkpoint_path.is_file():
            return cls()
        _install_model_explorer_path(repo_root)
        import torch
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import (
            CANDIDATE_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            MISSING_INDICATOR_NAMES,
            PolicyObservation,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        training = checkpoint.get("training", {}) if isinstance(checkpoint, dict) else {}
        observation = PolicyObservation(
            candidate_feature_names=CANDIDATE_FEATURE_NAMES,
            candidate_features=(tuple([0.0] * len(CANDIDATE_FEATURE_NAMES)),),
            global_feature_names=GLOBAL_FEATURE_NAMES,
            global_features=tuple([0.0] * len(GLOBAL_FEATURE_NAMES)),
            action_mask=(True,),
            candidate_cells=((0, 0),),
            candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
            candidate_missing_indicators=(tuple([0.0] * len(MISSING_INDICATOR_NAMES)),),
        )
        network = build_policy_network(
            checkpoint.get("architecture"),
            observation=observation,
            hidden_size=_int_value(training.get("hidden_size"), config.get("evaluation", {}).get("hidden_size", 16)),
        )
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state is None:
            return cls()
        network.load_state_dict(state)
        network.eval()
        return cls(network)

    def evaluate(self, observation, action_index: int) -> dict[str, float | None]:
        if self.network is None:
            return {"log_prob": None, "value": None}
        import torch
        from model_explorer.policy.torch_policy import observation_to_tensors

        with torch.no_grad():
            output = self.network(**observation_to_tensors(observation))
            distribution = torch.distributions.Categorical(logits=output.masked_logits)
            action = torch.tensor([int(action_index)], dtype=torch.long)
            return {
                "log_prob": float(distribution.log_prob(action)[0].detach()),
                "value": float(output.value[0].detach()),
            }


def _episode_from_transitions(transitions: list[Any]):
    from model_explorer.policy.rollout import EpisodeMetrics, RolloutEpisode

    total_path_cost = sum(float(transition.info.path_cost) for transition in transitions)
    average_risk = (
        sum(float(transition.info.risk) for transition in transitions) / len(transitions)
        if transitions
        else 0.0
    )
    return RolloutEpisode(
        transitions=tuple(transitions),
        metrics=EpisodeMetrics(
            final_coverage_rate=0.0,
            cumulative_coverage_rate_delta=sum(float(t.info.coverage_rate_delta) for t in transitions),
            total_path_cost=total_path_cost,
            average_risk=average_risk,
            failure_count=0,
            replan_count=0,
            value_coverage=0.0,
        ),
    )


def _apply_validation_gates(reason_codes: list[str], counters: Counter[str], config: dict[str, Any]) -> None:
    validation = config.get("validation", {})
    if counters["ppo_trainable_transition_count"] < int(validation.get("min_ppo_trainable_transition_count", 1)):
        _append_reason(reason_codes, "ppo_trainable_transition_count_insufficient")
    for field, reason in (
        ("source_fallback_trainable_count", "ppo_rollout_collector_contract_invalid"),
        ("invalid_action_mask_count", "ppo_rollout_collector_contract_invalid"),
        ("empty_action_mask_count", "ppo_rollout_collector_contract_invalid"),
        ("missing_log_prob_count", "ppo_logprob_value_missing"),
        ("missing_value_count", "ppo_logprob_value_missing"),
        ("non_finite_reward_count", "ppo_reward_contract_invalid"),
    ):
        max_value = int(validation.get(f"max_{field}", 0))
        if counters[field] > max_value:
            _append_reason(reason_codes, reason)


def _output_paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("output_files", {})
    return {
        "episodes": output_root / outputs.get("episodes", "ppo-rollout-episodes.jsonl"),
        "transitions": output_root / outputs.get("transitions", "ppo-rollout-transitions.jsonl"),
        "summary": output_root / outputs.get("summary", "ppo-rollout-collector-summary.json"),
        "rejection_report": output_root / outputs.get("rejection_report", "ppo-rollout-rejection-report.json"),
        "reward_audit": output_root / outputs.get("reward_audit", "ppo-rollout-reward-audit.json"),
    }


def _scenario_groups_by_id(sequential_root: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    return {str(group.get("scenario_id")): group for group in _collect_holdout_scenarios(sequential_root, repo_root)}


def _controlled_action_index(step: dict[str, Any]) -> int | None:
    for key in ("controlled_action_index", "controlled_selected_action_index"):
        value = _optional_int(step.get(key))
        if value is not None:
            return value
    if step.get("controlled_choice_source") == "policy":
        return _optional_int(step.get("policy_selected_action_index", step.get("raw_policy_selected_action_index")))
    return _optional_int(step.get("source_selected_action_index"))


def _candidate_action(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("source_action_index")
    if value is None:
        value = candidate.get("action_index")
    return _optional_int(value)


def _candidate_preferred(candidate: dict[str, Any], current: dict[str, Any], *, step: dict[str, Any], selected_action: int) -> bool:
    candidate_context = candidate.get("context_id")
    selected_context = step.get("policy_selected_context_id") or step.get("raw_policy_selected_context_id") or step.get("context_id")
    if _candidate_action(candidate) == selected_action and candidate_context and candidate_context == selected_context:
        return True
    if _candidate_action(current) == selected_action and current.get("context_id") == selected_context:
        return False
    return _candidate_action_mask_valid(candidate) and not _candidate_action_mask_valid(current)


def _rejection_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(reason for record in records for reason in record.get("rejection_reason_codes", []))
    return {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "rejected_transition_count": len(records),
        "reason_code_counts": dict(sorted(counts.items())),
        "records": records,
    }


def _reward_audit(records: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    rewards = [float(record.get("reward", 0.0)) for record in records if math.isfinite(float(record.get("reward", 0.0)))]
    return {
        "schema_version": REWARD_AUDIT_SCHEMA_VERSION,
        "record_count": len(records),
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_all_zero": bool(rewards and all(value == 0.0 for value in rewards)),
        "records": records,
    }


def _next_required_change(reason_codes: list[str]) -> str:
    if "ppo_logprob_value_missing" in reason_codes:
        return "ppo_logprob_value_missing"
    if "ppo_reward_contract_invalid" in reason_codes:
        return "ppo_reward_contract_invalid"
    if "ppo_trainable_transition_count_insufficient" in reason_codes:
        return "ppo_trainable_transition_count_insufficient"
    return "ppo_rollout_collector_contract_invalid"


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


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _dedup(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_or_zero(value: Any) -> float:
    numeric = _float_or_none(value)
    return numeric if numeric is not None and math.isfinite(numeric) else 0.0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
