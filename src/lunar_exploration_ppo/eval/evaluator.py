"""Fair Stage 5 evaluator with one shared environment and episode schema."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Final, Sequence

import numpy as np
import torch
from torch import nn

from lunar_exploration_ppo.env.env import LunarExplorationEnv
from lunar_exploration_ppo.env.reachability import reachable_component
from lunar_exploration_ppo.eval.baselines import (
    ALL_METHODS,
    BASELINE_METHODS,
    reconstruct_recommended_theta,
    select_baseline_action,
    select_ppo_action,
    validate_selected_index,
)
from lunar_exploration_ppo.eval.metrics import (
    ZERO_DISTANCE_POLICY,
    EpisodeResult,
    build_episode_result,
    summarize_episodes,
)
from lunar_exploration_ppo.policy.cross_attention import batch_policy_observations
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


PPO_EVAL_POLICY_MODE: Final = "deterministic_argmax_frontier_mean_theta/v1"
METHOD_ACTION_RULES: Final = {
    "random_valid_frontier": {
        "candidate_index_rule": "uniform_valid_candidate_fixed_episode_rng/v1",
        "theta_rule": "selected_candidate_recommended_theta/v1",
    },
    "nearest_frontier": {
        "candidate_index_rule": (
            "min_distance_from_robot_norm_lowest_index_tie/v1"
        ),
        "theta_rule": "selected_candidate_recommended_theta/v1",
    },
    "max_potential_gain_frontier": {
        "candidate_index_rule": (
            "max_potential_coverage_gain_norm_lowest_index_tie/v1"
        ),
        "theta_rule": "selected_candidate_recommended_theta/v1",
    },
    "gain_over_cost_frontier": {
        "candidate_index_rule": (
            "max_gain_over_one_plus_reachable_cost_lowest_index_tie/v1"
        ),
        "theta_rule": "selected_candidate_recommended_theta/v1",
    },
    "ppo_policy": {
        "candidate_index_rule": "masked_logit_argmax_lowest_index_tie/v1",
        "theta_rule": "selected_candidate_policy_theta_mu/v1",
    },
}
_DECISION_INPUT_FIELDS: Final = (
    "current_observed_highres_state",
    "low_resolution_prior",
    "global_coverage_summary",
    "frontier_cells_features_valid_mask",
    "current_pose",
    "sensor_contract",
    "planner_validation_result",
)
_FORBIDDEN_DECISION_INPUTS: Final = (
    "hidden_high_resolution_truth",
    "dense_coverable_mask_spatial_input",
    "future_observation_results",
    "unknown_cell_true_obstacle_height_traversability",
)


class EvaluationError(RuntimeError):
    """The evaluator cannot preserve the Stage 5 fairness contract."""


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    scenario_key: str
    scenario_seed: int
    terrain_seed: int
    start_pose_seed: int
    evaluation_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_key, str) or not self.scenario_key:
            raise EvaluationError("scenario key must be nonempty")
        for name in (
            "scenario_seed",
            "terrain_seed",
            "start_pose_seed",
            "evaluation_seed",
        ):
            if type(getattr(self, name)) is not int:
                raise EvaluationError(f"{name} must be an integer")

    def record(self) -> dict[str, object]:
        return {
            "scenario_key": self.scenario_key,
            "scenario_seed": self.scenario_seed,
            "terrain_seed": self.terrain_seed,
            "start_pose_seed": self.start_pose_seed,
            "evaluation_seed": self.evaluation_seed,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    method: str
    scale_profile: str
    episodes: tuple[EpisodeResult, ...]
    metrics: dict[str, object]
    bootstrap_audit: dict[str, object]
    fairness_audit: dict[str, object]


EnvironmentFactory = Callable[[EvaluationScenario], LunarExplorationEnv]


class Evaluator:
    """Run one shared environment contract with method-specific action rules."""

    def __init__(
        self,
        *,
        env_factory: EnvironmentFactory,
        scale_profile: str,
        max_steps: int,
        success_threshold: float,
        zero_distance_policy: str,
        bootstrap_resamples: int,
        bootstrap_seed: int,
        policy: nn.Module | None = None,
        policy_device: str | torch.device = "cpu",
    ) -> None:
        if not callable(env_factory):
            raise EvaluationError("env_factory must be callable")
        if not isinstance(scale_profile, str) or not scale_profile:
            raise EvaluationError("scale_profile must be nonempty")
        if type(max_steps) is not int or max_steps <= 0:
            raise EvaluationError("max_steps must be positive")
        if not math.isfinite(float(success_threshold)) or success_threshold != 0.99:
            raise EvaluationError("Stage 5 success threshold is frozen at 0.99")
        if zero_distance_policy != ZERO_DISTANCE_POLICY:
            raise EvaluationError("Stage 5 zero-distance policy drift")
        if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
            raise EvaluationError("bootstrap resamples must be positive")
        if type(bootstrap_seed) is not int:
            raise EvaluationError("bootstrap seed must be an integer")
        if policy is not None and not isinstance(policy, nn.Module):
            raise EvaluationError("policy must be a torch module")
        device = torch.device(policy_device)
        if policy is not None:
            if device.type == "cuda" and device.index is None:
                if not torch.cuda.is_available():
                    raise EvaluationError("CUDA policy_device is unavailable")
                device = torch.device("cuda", torch.cuda.current_device())
            parameter_devices = {parameter.device for parameter in policy.parameters()}
            buffer_devices = {buffer.device for buffer in policy.buffers()}
            if any(item != device for item in parameter_devices | buffer_devices):
                raise EvaluationError("policy tensors do not match policy_device")
        self._env_factory = env_factory
        self.scale_profile = scale_profile
        self.max_steps = max_steps
        self.success_threshold = float(success_threshold)
        self.zero_distance_policy = zero_distance_policy
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_seed = bootstrap_seed
        self.policy = policy
        self.policy_device = device

    def evaluate(
        self,
        method: str,
        scenario_set: Sequence[EvaluationScenario],
    ) -> EvaluationSummary:
        if method not in ALL_METHODS:
            raise EvaluationError(f"unknown Stage 5 method: {method!r}")
        scenarios = tuple(scenario_set)
        if not scenarios or not all(
            isinstance(scenario, EvaluationScenario) for scenario in scenarios
        ):
            raise EvaluationError("scenario_set must contain EvaluationScenario rows")
        scenario_records = tuple(
            tuple(scenario.record().items()) for scenario in scenarios
        )
        if len(set(scenario_records)) != len(scenarios):
            raise EvaluationError("scenario schedule entries must be unique")
        if method == "ppo_policy" and self.policy is None:
            raise EvaluationError("ppo_policy evaluation requires a loaded policy")

        original_training = self.policy.training if self.policy is not None else None
        if method == "ppo_policy":
            assert self.policy is not None
            self.policy.eval()
        episodes: list[EpisodeResult] = []
        episode_audits: list[dict[str, object]] = []
        contracts: list[dict[str, object]] = []
        try:
            for scenario in scenarios:
                episode, audit, contract = self._run_episode(method, scenario)
                episodes.append(episode)
                episode_audits.append(audit)
                contracts.append(contract)
        finally:
            if method == "ppo_policy":
                assert self.policy is not None and original_training is not None
                self.policy.train(original_training)

        metrics, bootstrap = summarize_episodes(
            tuple(episodes),
            bootstrap_resamples=self.bootstrap_resamples,
            bootstrap_seed=self.bootstrap_seed,
        )
        shared_contract = {
            "schema_version": "stage5_shared_environment_contract/v1",
            "scale_profile": self.scale_profile,
            "max_steps": self.max_steps,
            "success_threshold": self.success_threshold,
            "scenario_schedule": [scenario.record() for scenario in scenarios],
            "episodes": contracts,
        }
        shared_contract_sha256 = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(shared_contract)
        ).hexdigest()
        selected_actions = [
            action
            for episode_audit in episode_audits
            for action in episode_audit["selected_actions"]
        ]
        all_safe = all(
            action["candidate_mask_valid"]
            and action["observed_safe"]
            and action["reachable_from_current_pose"]
            for action in selected_actions
        )
        all_baseline_thetas_recommended = (
            all(
                action["target_theta"] == action["recommended_theta"]
                for action in selected_actions
            )
            if method in BASELINE_METHODS
            else None
        )
        fairness = {
            "schema_version": "stage5_method_fairness_audit/v2",
            "method": method,
            "shared_environment_contract_identical": True,
            "method_specific_action_rule_only_difference": True,
            **METHOD_ACTION_RULES[method],
            "decision_input_fields": list(_DECISION_INPUT_FIELDS),
            "forbidden_decision_inputs": list(_FORBIDDEN_DECISION_INPUTS),
            "coverage_denominator_access": "environment_metric_layer_only/v1",
            "ppo_eval_policy_mode": PPO_EVAL_POLICY_MODE,
            "shared_contract": shared_contract,
            "shared_contract_sha256": shared_contract_sha256,
            "selected_action_count": len(selected_actions),
            "invalid_selected_action_count": 0,
            "all_selected_actions_reachable_observed_safe": all_safe,
            "all_baseline_thetas_recommended": all_baseline_thetas_recommended,
            "episode_audits": episode_audits,
        }
        return EvaluationSummary(
            method=method,
            scale_profile=self.scale_profile,
            episodes=tuple(episodes),
            metrics=metrics,
            bootstrap_audit=bootstrap,
            fairness_audit=fairness,
        )

    def _run_episode(
        self,
        method: str,
        scenario: EvaluationScenario,
    ) -> tuple[EpisodeResult, dict[str, object], dict[str, object]]:
        env = self._env_factory(scenario)
        if not isinstance(env, LunarExplorationEnv):
            raise EvaluationError("env_factory must return LunarExplorationEnv")
        if (
            env.config.scale_profile != self.scale_profile
            or env.config.max_steps != self.max_steps
            or env.config.success_coverage_rate != self.success_threshold
            or env.config.scenario_key != scenario.scenario_key
        ):
            raise EvaluationError("environment differs from the evaluator contract")
        contract = self._environment_contract(env, scenario)
        observation = env.reset()
        reset = env.last_reset_diagnostics
        if reset is None:
            raise EvaluationError("environment reset diagnostics are missing")
        coverage_curve = [float(reset.coverage_rate)]
        path_curve = [0.0]
        total_path_length = 0.0
        invalid_action_count = 0
        planner_failure_count = 0
        safety_violation_count = 0
        selected_actions: list[dict[str, object]] = []
        rng = np.random.Generator(np.random.PCG64(scenario.evaluation_seed))

        while not env.is_done and env.step_count < self.max_steps:
            if not env.needs_policy or not bool(observation.candidate_mask.any()):
                raise EvaluationError("nonterminal environment has no policy candidate")
            action = self._select_action(method, observation, rng)
            selected_record = self._validate_selected_action(env, observation, action)
            step_result = env.step(action)
            selected_record["planner_failure_reason"] = step_result.diagnostics.planner.get(
                "failure_reason",
                "none",
            )
            selected_record["planner_failure_classification"] = (
                step_result.diagnostics.planner.get("failure_classification", "none")
            )
            selected_record["environment_invalid_action"] = (
                step_result.diagnostics.invalid_action
            )
            selected_record["environment_safety_violation"] = (
                step_result.diagnostics.safety_violation
            )
            selected_actions.append(selected_record)
            invalid_action_count += int(step_result.diagnostics.invalid_action)
            failure_reason = step_result.diagnostics.planner.get("failure_reason")
            planner_failure_count += int(
                failure_reason not in (None, "none")
            )
            safety_violation_count += int(step_result.diagnostics.safety_violation)
            if step_result.diagnostics.execution is not None:
                total_path_length += float(step_result.diagnostics.path_length_m)
            coverage_curve.append(float(step_result.coverage_rate))
            path_curve.append(total_path_length)
            observation = step_result.observation

        if not env.is_done:
            raise EvaluationError("environment did not terminate at the fixed step budget")
        steps_executed = env.step_count
        if len(coverage_curve) != steps_executed + 1:
            raise EvaluationError("episode curve length does not match executed steps")
        coverage_curve.extend(
            [coverage_curve[-1]] * (self.max_steps + 1 - len(coverage_curve))
        )
        path_curve.extend([path_curve[-1]] * (self.max_steps + 1 - len(path_curve)))
        episode = build_episode_result(
            method=method,
            scale_profile=self.scale_profile,
            scenario_key=scenario.scenario_key,
            scenario_seed=scenario.scenario_seed,
            terrain_seed=scenario.terrain_seed,
            start_pose_seed=scenario.start_pose_seed,
            evaluation_seed=scenario.evaluation_seed,
            coverage_curve=coverage_curve,
            cumulative_path_length_curve=path_curve,
            steps_executed=steps_executed,
            invalid_action_count=invalid_action_count,
            planner_failure_count=planner_failure_count,
            safety_violation_count=safety_violation_count,
            termination_reason=env.terminal_reason,
            max_steps=self.max_steps,
            success_threshold=self.success_threshold,
            zero_distance_policy=self.zero_distance_policy,
        )
        audit = {
            "scenario": scenario.record(),
            "scenario_id": env.scenario_id,
            "scenario_hash": env.scenario_hash,
            "initial_coverage": float(reset.coverage_rate),
            "steps_executed": steps_executed,
            "selected_actions": selected_actions,
            "termination_reason": env.terminal_reason,
        }
        return episode, audit, contract

    def _select_action(
        self,
        method: str,
        observation: PolicyObservation,
        rng: np.random.Generator,
    ):
        if method in BASELINE_METHODS:
            return select_baseline_action(method, observation, rng)
        if method != "ppo_policy" or self.policy is None:
            raise EvaluationError("selected method has no evaluation policy")
        batch = batch_policy_observations(
            (observation,),
            device=self.policy_device,
        )
        with torch.inference_mode():
            output = self.policy(batch)
        return select_ppo_action(output, batch.candidate_mask)

    def _validate_selected_action(
        self,
        env: LunarExplorationEnv,
        observation: PolicyObservation,
        action,
    ) -> dict[str, object]:
        index = validate_selected_index(
            action.candidate_index,
            observation.candidate_mask,
        )
        if index >= env.current_action_set.candidate_count:
            raise EvaluationError("selected candidate index has no shared candidate cell")
        cell = env.current_action_set.cells[index]
        safe = bool(env.observed_state.observed_safe_mask[cell.y, cell.x])
        reachable = bool(
            reachable_component(
                env.observed_state.observed_safe_mask,
                env.pose.cell,
            )[cell.y, cell.x]
        )
        if not safe or not reachable:
            raise EvaluationError("selected action is not reachable observed-safe")
        if not math.isfinite(float(action.target_theta)):
            raise EvaluationError("selected action theta must be finite")
        if env.current_action_set.frontier_features is not observation.frontier_features:
            if not np.array_equal(
                env.current_action_set.frontier_features,
                observation.frontier_features,
            ):
                raise EvaluationError("observation candidate ordering drifted")
        return {
            "step": env.step_count,
            "candidate_index": index,
            "candidate_cell_xy": [cell.x, cell.y],
            "candidate_mask_valid": True,
            "observed_safe": safe,
            "reachable_from_current_pose": reachable,
            "same_connected_component_feature": float(
                observation.frontier_features[index, 21]
            ),
            "target_theta": float(action.target_theta),
            "recommended_theta": reconstruct_recommended_theta(
                observation.frontier_features[index]
            ),
        }

    def _environment_contract(
        self,
        env: LunarExplorationEnv,
        scenario: EvaluationScenario,
    ) -> dict[str, object]:
        coverage = env.coverage_metadata
        if (
            coverage.get("exact") is not True
            or not isinstance(coverage.get("sha256"), str)
            or len(coverage["sha256"]) != 64
            or not isinstance(coverage.get("algorithm_id"), str)
            or not isinstance(coverage.get("precompute_scope"), str)
            or type(coverage.get("coverable_cell_count")) is not int
            or coverage["coverable_cell_count"] <= 0
        ):
            raise EvaluationError("exact coverable denominator metadata is invalid")
        return {
            "scenario": scenario.record(),
            "scenario_id": env.scenario_id,
            "scenario_hash": env.scenario_hash,
            "environment_class": (
                "lunar_exploration_ppo.env.env.LunarExplorationEnv"
            ),
            "environment_config_sha256": hashlib.sha256(
                ArtifactStore.canonical_json_bytes(
                    env.config.model_dump(mode="json")
                )
            ).hexdigest(),
            "sensor_model_id": env.config.sensor_model_id,
            "initial_scan": "free_reset_scan_20m_90deg/v1",
            "frontier_generator_class": (
                f"{type(env.frontier_generator).__module__}."
                f"{type(env.frontier_generator).__qualname__}"
            ),
            "candidate_feature_schema": "frontier_features_22/v1",
            "reachability_prefilter": "observed_safe_connected_component/v1",
            "planner_class": (
                f"{type(env.planner).__module__}.{type(env.planner).__qualname__}"
            ),
            "planner_validation_source": "stage1_path_planner_adapter/v1",
            "path_execution_sensor_updates": "shared_environment_step/v1",
            "coverage_denominator_source": "coverable_mask_exact",
            "coverable_mask_exact": coverage["exact"],
            "coverable_mask_hash": coverage["sha256"],
            "coverable_mask_algorithm_id": coverage["algorithm_id"],
            "coverable_mask_precompute_scope": coverage["precompute_scope"],
            "coverable_cell_count": coverage["coverable_cell_count"],
            "max_steps": env.config.max_steps,
            "stagnation_no_gain_steps": env.config.stagnation_no_gain_steps,
            "success_threshold": env.config.success_coverage_rate,
            "safety_constants": {
                "vehicle_radius_m": env.config.vehicle_radius_m,
                "safety_margin_m": env.config.safety_margin_m,
                "min_clearance_m": env.config.min_clearance_m,
                "traversability_threshold": env.config.traversability_threshold,
                "max_traversable_slope_deg": env.config.max_traversable_slope_deg,
            },
        }


__all__ = [
    "METHOD_ACTION_RULES",
    "PPO_EVAL_POLICY_MODE",
    "EvaluationError",
    "EvaluationScenario",
    "EvaluationSummary",
    "Evaluator",
]
