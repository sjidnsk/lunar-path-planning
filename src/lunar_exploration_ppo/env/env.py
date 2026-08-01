"""Smoke v1 environment closed loop without policy-network or PPO concerns."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final, Literal, Mapping, Protocol

import numpy as np

from lunar_exploration_ppo.configs.stage1 import Stage1Config
from lunar_exploration_ppo.env.action_execution import ActionExecutionDiagnostics, build_sensor_poses
from lunar_exploration_ppo.env.coverage import CoverageMasks, compute_coverage_masks
from lunar_exploration_ppo.env.frontier import FrontierActionSet, FrontierGenerator
from lunar_exploration_ppo.env.frontier_oracle import audit_frontier_opportunities
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import LowResolutionPrior, ScenarioBundle, ScenarioSource
from lunar_exploration_ppo.env.sensor_model import ObservationDelta, SensorDiagnostics, SensorPose, SensorUpdater
from lunar_exploration_ppo.env.terrain_proxy import TerrainProxySettings
from lunar_exploration_ppo.integrations.path_planner_adapter import PathPlannerAdapter
from lunar_exploration_ppo.policy.observation import ObservationBuilder, PolicyObservation
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import CellXY, PoseXYTheta, normalize_theta


DoneReason = Literal[
    "success_done",
    "failure_done",
    "stagnation_done",
    "no_candidate_done",
    "safety_done",
    "none",
]
ALLOWED_DONE_REASONS = frozenset(
    {"success_done", "failure_done", "stagnation_done", "no_candidate_done", "safety_done", "none"}
)
EPISODE_STATE_SCHEMA_VERSION: Final = "lunar_exploration_env_episode_state/v1"
_EPISODE_ARRAY_DTYPES: Final = {
    "observed_mask": np.dtype("|b1"),
    "confidence": np.dtype("<f4"),
    "height": np.dtype("<f8"),
    "obstacle": np.dtype("|b1"),
    "slope_deg": np.dtype("<f8"),
    "traversability": np.dtype("<f8"),
    "observed_safe_mask": np.dtype("|b1"),
}


class FrontierExtractor(Protocol):
    def extract(
        self,
        observed_state: ObservedMapState,
        prior: LowResolutionPrior,
        pose: PoseXYTheta,
    ) -> FrontierActionSet: ...


class ExecutionSafetyChecker(Protocol):
    def check(self, safe_mask: np.ndarray, path_cells: tuple[CellXY, ...]) -> str | None: ...


class FinalMaskExecutionSafetyChecker:
    """Detect an unrecoverable path-safety violation after successful planning."""

    def check(self, safe_mask: np.ndarray, path_cells: tuple[CellXY, ...]) -> str | None:
        height, width = safe_mask.shape
        for cell in path_cells:
            if not (0 <= cell.x < width and 0 <= cell.y < height):
                return "execution_path_out_of_bounds"
            if not bool(safe_mask[cell.y, cell.x]):
                return "execution_path_entered_unsafe_cell"
        return None


@dataclass(frozen=True, slots=True)
class EnvAction:
    candidate_index: int
    target_theta: float


@dataclass(frozen=True, slots=True)
class _LegacyRulePolicyInput:
    frontier_features: np.ndarray
    pose_features: np.ndarray
    candidate_mask: np.ndarray

    def __post_init__(self) -> None:
        self.frontier_features.setflags(write=False)
        self.pose_features.setflags(write=False)
        self.candidate_mask.setflags(write=False)


def select_conservative_frontier_candidate_index(
    frontier_features: np.ndarray,
    candidate_mask: np.ndarray,
) -> int:
    """Select the lowest observed frontier risk with deterministic tie-breaks."""

    features = np.asarray(frontier_features)
    mask = np.asarray(candidate_mask)
    if features.ndim != 2 or features.shape[1] <= 8:
        raise ValueError("rule selector requires frontier feature columns 4, 7, and 8")
    if mask.ndim != 1 or mask.shape[0] != features.shape[0]:
        raise ValueError("rule selector candidate mask shape mismatch")
    valid_rows = [index for index, valid in enumerate(mask) if bool(valid)]
    if not valid_rows:
        raise RuntimeError("rule policy requires at least one observed candidate")
    return min(
        valid_rows,
        key=lambda row: (
            float(features[row, 8]),
            float(features[row, 7]),
            -float(features[row, 4]),
            row,
        ),
    )


def select_conservative_rule_action(observation: PolicyObservation) -> EnvAction:
    """Build the deterministic observed-only Smoke rule action."""

    index = select_conservative_frontier_candidate_index(
        observation.frontier_features,
        observation.candidate_mask,
    )
    selected = observation.frontier_features[index]
    target_theta = _finite_direction_theta_or_none(
        float(selected[5]),
        float(selected[6]),
    )
    if target_theta is None:
        target_theta = _finite_direction_theta_or_none(
            float(selected[3]),
            float(selected[2]),
        )
    if target_theta is None:
        target_theta = _finite_direction_theta_or_none(
            float(observation.pose_features[2]),
            float(observation.pose_features[3]),
        )
    if target_theta is None:
        target_theta = 0.0
    return EnvAction(candidate_index=index, target_theta=target_theta)


def select_stage2_diagnostic_rule_action(observation: PolicyObservation) -> EnvAction:
    """Opt-in Stage 2 demo selector; never used by the default rule policy."""

    features = np.asarray(observation.frontier_features)
    mask = np.asarray(observation.candidate_mask)
    if features.ndim != 2 or features.shape[1] < 22:
        raise ValueError("Stage 2 diagnostic selector requires the 22-field schema")
    if mask.ndim != 1 or mask.shape[0] != features.shape[0]:
        raise ValueError("Stage 2 diagnostic selector candidate mask shape mismatch")
    valid_rows = [index for index, valid in enumerate(mask) if bool(valid)]
    if not valid_rows:
        raise RuntimeError("Stage 2 diagnostic selector requires a candidate")
    index = min(
        valid_rows,
        key=lambda row: (
            -(
                0.5 * float(features[row, 5])
                + 0.3 * float(features[row, 7])
                - 0.1 * float(features[row, 2])
                - 0.1 * float(features[row, 18])
            ),
            row,
        ),
    )
    selected = features[index]
    target_theta = _finite_direction_theta_or_none(float(selected[14]), float(selected[15]))
    if target_theta is None:
        target_theta = _finite_direction_theta_or_none(float(selected[3]), float(selected[4]))
    if target_theta is None:
        target_theta = _finite_direction_theta_or_none(
            float(observation.pose_features[2]),
            float(observation.pose_features[3]),
        )
    if target_theta is None:
        target_theta = 0.0
    return EnvAction(candidate_index=index, target_theta=target_theta)


def _finite_direction_theta_or_none(
    sine_component: float,
    cosine_component: float,
) -> float | None:
    if not math.isfinite(sine_component) or not math.isfinite(cosine_component):
        return None
    if math.hypot(sine_component, cosine_component) <= 1.0e-12:
        return None
    return math.atan2(sine_component, cosine_component)


@dataclass(frozen=True, slots=True)
class TerminalDecision:
    done: bool
    reason: DoneReason
    terminal: bool
    bootstrap_value: float


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    coverage_progress: float
    step_progress: float
    stagnation_progress: float


@dataclass(frozen=True, slots=True)
class FrontierDiagnostics:
    candidate_count: int
    component_size: int
    unknown_count: int
    remaining_unobserved_coverable_count: int
    oracle_opportunity_count: int


@dataclass(frozen=True, slots=True)
class ResetDiagnostics:
    reward: float
    trainable: bool
    fake_logprob_created: bool
    coverage_rate: float
    progress: ProgressSnapshot
    reason: DoneReason
    frontier: FrontierDiagnostics
    sensor: SensorDiagnostics | None = None
    local_safety_sensor: SensorDiagnostics | None = None
    exploration_sensor: SensorDiagnostics | None = None
    scan_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        exploration = self.exploration_sensor or self.sensor
        if exploration is None:
            raise ValueError("reset exploration sensor diagnostics are required")
        local = self.local_safety_sensor or exploration
        object.__setattr__(self, "sensor", exploration)
        object.__setattr__(self, "exploration_sensor", exploration)
        object.__setattr__(self, "local_safety_sensor", local)
        if not self.scan_order:
            object.__setattr__(self, "scan_order", ("reset_exploration",))


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    planner: dict[str, object]
    sensor: SensorDiagnostics | None
    execution: ActionExecutionDiagnostics | None
    invalid_action: bool
    safety_violation: bool
    progress: ProgressSnapshot
    frontier: FrontierDiagnostics
    planned_path_cells: tuple[CellXY, ...] = ()
    path_length_m: float = 0.0
    path_observation_step_m: float = 0.0
    newly_observed_cell_count: int = 0


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: PolicyObservation
    reward: float
    done: bool
    reason: DoneReason
    coverage_gain_cells: int
    coverage_rate: float
    trainable: bool
    terminal: bool
    bootstrap_value: float
    diagnostics: StepDiagnostics


_COVERAGE_CACHE: dict[tuple[object, ...], CoverageMasks] = {}


def _validate_precomputed_coverage_masks(masks: CoverageMasks, scenario: ScenarioBundle) -> None:
    expected_shape = scenario.truth.geometry.shape
    arrays = (
        masks.safe_free_mask,
        masks.reachable_safe_mask,
        masks.coverable_mask,
    )
    if any(array.dtype != np.dtype(bool) or array.shape != expected_shape for array in arrays):
        raise ValueError("precomputed coverage mask geometry or dtype drifted")
    coverable_cell_count = int(np.count_nonzero(masks.coverable_mask))
    if coverable_cell_count <= 0:
        raise ValueError("precomputed coverage coverable denominator must be positive")
    coverable_hash = hashlib.sha256(
        np.ascontiguousarray(masks.coverable_mask).tobytes()
    ).hexdigest()
    expected_metadata = {
        "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
        "sha256": coverable_hash,
        "exact": True,
        "precompute_scope": "scenario_reset/v1",
        "coverable_cell_count": coverable_cell_count,
    }
    if not _strict_json_equal(dict(masks.metadata), expected_metadata):
        raise ValueError("precomputed coverage mask metadata drifted")


def _strict_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


class LunarExplorationEnv:
    def __init__(
        self,
        config: Stage1Config,
        *,
        scenario_source: ScenarioSource | None = None,
        frontier_generator: FrontierExtractor | None = None,
        execution_safety_checker: ExecutionSafetyChecker | None = None,
        precomputed_coverage_masks: CoverageMasks | None = None,
    ) -> None:
        self.config = config
        self._enforce_fixed_smoke_structural_gates = (
            scenario_source is None and frontier_generator is None
        )
        if scenario_source is None:
            scenario_source = ScenarioSource(
                settings=TerrainProxySettings(
                    generator_version=config.proxy_generator_version,
                    density_profile=config.proxy_density_profile,
                    base_seed=config.proxy_base_seed,
                    start_protection_m=config.proxy_start_protection_m,
                    max_scene_attempts=config.proxy_max_scene_attempts,
                    max_object_attempts=config.proxy_max_object_attempts,
                )
            )
        self._scenario: ScenarioBundle = scenario_source.load(config.scenario_key)
        if precomputed_coverage_masks is None:
            cache_key = (
                self._scenario.scenario_hash,
                config.sensor_range_m,
                config.min_clearance_m,
                config.max_traversable_slope_deg,
                config.traversability_threshold,
            )
            if cache_key not in _COVERAGE_CACHE:
                _COVERAGE_CACHE[cache_key] = compute_coverage_masks(
                    self._scenario.truth,
                    self._scenario.start_pose.cell,
                    sensor_range_m=config.sensor_range_m,
                    min_clearance_m=config.min_clearance_m,
                    max_slope_deg=config.max_traversable_slope_deg,
                    traversability_threshold=config.traversability_threshold,
                )
            cached_masks = _COVERAGE_CACHE[cache_key]
        else:
            _validate_precomputed_coverage_masks(precomputed_coverage_masks, self._scenario)
            cached_masks = precomputed_coverage_masks
        self._coverage_masks = CoverageMasks(
            safe_free_mask=cached_masks.safe_free_mask.copy(),
            reachable_safe_mask=cached_masks.reachable_safe_mask.copy(),
            coverable_mask=cached_masks.coverable_mask.copy(),
            metadata=dict(cached_masks.metadata),
        )
        self.frontier_generator = frontier_generator or FrontierGenerator(top_m=config.frontier_top_m)
        self._enforce_frontier_oracle = isinstance(self.frontier_generator, FrontierGenerator)
        self.observation_builder = ObservationBuilder()
        self.sensor_updater = SensorUpdater(
            range_m=config.sensor_range_m,
            fov_deg=config.sensor_fov_deg,
            ray_angle_step_deg=config.sensor_ray_angle_step_deg,
            min_clearance_m=config.min_clearance_m,
            max_slope_deg=config.max_traversable_slope_deg,
            traversability_threshold=config.traversability_threshold,
        )
        self.local_safety_sensor_updater = SensorUpdater(
            range_m=config.reset_local_safety_scan_range_m,
            fov_deg=config.reset_local_safety_scan_fov_deg,
            ray_angle_step_deg=config.reset_local_safety_scan_ray_angle_step_deg,
            min_clearance_m=config.min_clearance_m,
            max_slope_deg=config.max_traversable_slope_deg,
            traversability_threshold=config.traversability_threshold,
        )
        self.planner = PathPlannerAdapter(self._scenario.truth.geometry)
        self.execution_safety_checker = execution_safety_checker or FinalMaskExecutionSafetyChecker()
        self.observed_state = ObservedMapState.empty(self._scenario.truth.geometry)
        self.observed_state.remaining_step_budget_norm = 1.0
        self.pose = self._scenario.start_pose
        self.step_count = 0
        self.consecutive_no_gain_steps = 0
        self.current_action_set = _empty_action_set(config.frontier_top_m)
        self.current_observation: PolicyObservation | None = None
        self.last_reset_diagnostics: ResetDiagnostics | None = None
        self.is_done = False
        self.terminal_reason: DoneReason = "none"
        self.needs_policy = False

    def reset(self) -> PolicyObservation:
        self.observed_state = ObservedMapState.empty(self._scenario.truth.geometry)
        self.pose = self._scenario.start_pose
        self.step_count = 0
        self.consecutive_no_gain_steps = 0
        self.is_done = False
        self.terminal_reason = "none"
        reset_world = self._scenario.truth.geometry.cell_to_world_center(
            self.pose.cell
        )
        local_safety_pose = SensorPose(
            reset_world,
            self.pose.theta,
            "reset_local_safety",
        )
        exploration_pose = SensorPose(
            reset_world,
            self.pose.theta,
            "reset",
        )
        local_delta = self.local_safety_sensor_updater.reveal(
            self._scenario.truth,
            self.observed_state,
            (local_safety_pose,),
        )
        exploration_delta = self.sensor_updater.reveal(
            self._scenario.truth,
            self.observed_state,
            (exploration_pose,),
        )
        self.current_action_set = self.frontier_generator.extract(
            self.observed_state, self._scenario.prior, self.pose
        )
        self.current_observation = self.observation_builder.build(
            self._scenario.prior, self.observed_state, self.pose, self.current_action_set
        )
        coverage_rate = self._coverage_rate()
        if self._enforce_fixed_smoke_structural_gates:
            self._validate_fixed_smoke_structural_gates(coverage_rate)
        frontier_diagnostics = self._frontier_diagnostics()
        self.needs_policy = self.current_action_set.candidate_count > 0
        if not self.needs_policy:
            self.is_done = True
            self.terminal_reason = "no_candidate_done"
        self.last_reset_diagnostics = ResetDiagnostics(
            reward=0.0,
            trainable=False,
            fake_logprob_created=False,
            local_safety_sensor=local_delta.diagnostics,
            exploration_sensor=exploration_delta.diagnostics,
            scan_order=("reset_local_safety", "reset_exploration"),
            coverage_rate=coverage_rate,
            progress=self._progress(coverage_rate),
            reason=self.terminal_reason,
            frontier=frontier_diagnostics,
        )
        return self.current_observation

    def select_rule_action(self, observation: PolicyObservation) -> EnvAction:
        if observation is not self.current_observation:
            raise ValueError("rule action requires the current observation")
        if not self.needs_policy or self.current_action_set.candidate_count == 0:
            raise RuntimeError("policy must be skipped when no candidate exists")
        rule_input = _legacy_rule_policy_input(
            self.observed_state,
            self._scenario.prior,
            self.pose,
            self.current_action_set,
        )
        return select_conservative_rule_action(rule_input)

    def step(self, action: EnvAction) -> StepResult:
        if self.current_observation is None:
            raise RuntimeError("reset must be called before step")
        if self.is_done:
            raise RuntimeError("cannot step a terminal environment")
        invalid_action = not self._valid_sampled_action(action)
        severe_safety = False
        planner_diagnostics: dict[str, object] = {}
        sensor_delta: ObservationDelta | None = None
        execution: ActionExecutionDiagnostics | None = None
        planned_path_cells: tuple[CellXY, ...] = ()
        planned_path_length_m = 0.0
        before_covered = self._observed_coverable_count()

        if not invalid_action:
            target = self.current_action_set.cells[action.candidate_index]
            plan = self.planner.validate(
                self.observed_state.observed_safe_mask,
                self.observed_state.planning_safe_mask,
                self.pose.cell,
                target,
                action.target_theta,
            )
            planned_path_cells = tuple(plan.path_cells)
            planned_path_length_m = float(plan.path_length_m)
            planner_diagnostics = dict(plan.diagnostics)
            planner_diagnostics["failure_reason"] = plan.failure_reason
            planner_diagnostics["failure_classification"] = plan.failure_classification
            if not plan.valid:
                invalid_action = True
            else:
                execution_safety_failure = self.execution_safety_checker.check(
                    self.observed_state.observed_safe_mask,
                    plan.path_cells,
                )
                planner_diagnostics["execution_safety_failure_reason"] = (
                    execution_safety_failure or "none"
                )
                if execution_safety_failure is not None:
                    if not isinstance(execution_safety_failure, str) or not execution_safety_failure:
                        raise RuntimeError("execution safety failure reason must be a nonempty string")
                    severe_safety = True
                else:
                    assert plan.target_theta is not None
                    sensor_poses, execution = build_sensor_poses(
                        plan.path_cells,
                        self._scenario.truth.geometry,
                        target_theta=plan.target_theta,
                        step_m=self.config.path_observation_step_m,
                    )
                    sensor_delta = self.sensor_updater.reveal(
                        self._scenario.truth, self.observed_state, sensor_poses
                    )
                    self.pose = PoseXYTheta(target, plan.target_theta)
                    post_observation_safety_failure = _post_observation_safety_failure(
                        self.observed_state.observed_safe_mask,
                        self.observed_state.planning_safe_mask,
                        plan.path_cells,
                        endpoint=target,
                    )
                    planner_diagnostics["post_observation_safety_failure_reason"] = (
                        post_observation_safety_failure or "none"
                    )
                    if post_observation_safety_failure is not None:
                        severe_safety = True

        after_covered = self._observed_coverable_count()
        coverage_gain_cells = after_covered - before_covered
        if coverage_gain_cells < 0:
            raise RuntimeError("observed coverable count regressed")
        self.step_count += 1
        self.observed_state.remaining_step_budget_norm = max(
            0.0,
            (self.config.max_steps - self.step_count) / self.config.max_steps,
        )
        if coverage_gain_cells == 0:
            self.consecutive_no_gain_steps += 1
        else:
            self.consecutive_no_gain_steps = 0
        coverage_rate = after_covered / self._coverage_masks.coverable_cell_count
        self.current_action_set = self.frontier_generator.extract(
            self.observed_state, self._scenario.prior, self.pose
        )
        self.current_observation = self.observation_builder.build(
            self._scenario.prior, self.observed_state, self.pose, self.current_action_set
        )
        decision = resolve_terminal(
            severe_safety,
            coverage_rate,
            self.step_count,
            self.consecutive_no_gain_steps,
            self.current_action_set.candidate_count > 0,
            success_coverage_rate=self.config.success_coverage_rate,
            max_steps=self.config.max_steps,
            stagnation_no_gain_steps=self.config.stagnation_no_gain_steps,
        )
        frontier_diagnostics = self._frontier_diagnostics(
            enforce_oracle=(
                not decision.done or decision.reason == "no_candidate_done"
            )
        )
        reward = compute_step_reward(
            self.config,
            coverage_gain_cells,
            self._coverage_masks.coverable_cell_count,
            coverage_rate,
            invalid_action,
            severe_safety,
        )
        self.is_done = decision.done
        self.terminal_reason = decision.reason
        self.needs_policy = not decision.done and self.current_action_set.candidate_count > 0
        progress = self._progress(coverage_rate)
        return StepResult(
            observation=self.current_observation,
            reward=float(reward),
            done=decision.done,
            reason=decision.reason,
            coverage_gain_cells=coverage_gain_cells,
            coverage_rate=coverage_rate,
            trainable=True,
            terminal=decision.terminal,
            bootstrap_value=decision.bootstrap_value,
            diagnostics=StepDiagnostics(
                planner=planner_diagnostics,
                sensor=sensor_delta.diagnostics if sensor_delta is not None else None,
                execution=execution,
                invalid_action=invalid_action,
                safety_violation=severe_safety,
                progress=progress,
                frontier=frontier_diagnostics,
                planned_path_cells=planned_path_cells,
                path_length_m=planned_path_length_m,
                path_observation_step_m=self.config.path_observation_step_m,
                newly_observed_cell_count=(
                    sensor_delta.newly_observed_count if sensor_delta is not None else 0
                ),
            ),
        )

    def _valid_sampled_action(self, action: EnvAction) -> bool:
        if not isinstance(action, EnvAction) or type(action.candidate_index) is not int:
            return False
        index = action.candidate_index
        if not 0 <= index < self.current_action_set.candidate_mask.size:
            return False
        if not self.current_action_set.candidate_mask[index] or index >= self.current_action_set.candidate_count:
            return False
        try:
            normalize_theta(action.target_theta)
        except ValueError:
            return False
        return True

    def _observed_coverable_count(self) -> int:
        return int(np.count_nonzero(self.observed_state.observed_mask & self._coverage_masks.coverable_mask))

    def _coverage_rate(self) -> float:
        return self._observed_coverable_count() / self._coverage_masks.coverable_cell_count

    @property
    def scenario_id(self) -> str:
        return self._scenario.scenario_id

    @property
    def scenario_hash(self) -> str:
        return self._scenario.scenario_hash

    @property
    def coverage_metadata(self) -> dict[str, object]:
        return dict(self._coverage_masks.metadata)

    @property
    def proxy_morphology_metadata(self) -> dict[str, object]:
        catalog = self._scenario.proxy_catalog
        return {
            "proxy_generator_version": catalog.generator_version,
            "density_profile": catalog.density_profile,
            "generation_attempt": catalog.generation_attempt,
            "rock_count": len(catalog.rocks),
            "crater_count": len(catalog.craters),
            "object_catalog_sha256": catalog.sha256,
            "layer_hashes": dict(self._scenario.proxy_layer_hashes),
        }

    @property
    def coverable_cell_count(self) -> int:
        return self._coverage_masks.coverable_cell_count

    def export_episode_state(self) -> dict[str, object]:
        """导出仅含 observed episode 状态的 canonical JSON DTO。"""

        initialized = self.current_observation is not None
        body: dict[str, object] = {
            "schema_version": EPISODE_STATE_SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "initialized": initialized,
            "step_count": self.step_count,
            "consecutive_no_gain_steps": self.consecutive_no_gain_steps,
            "pose_cell_xy": [self.pose.cell.x, self.pose.cell.y],
            "pose_theta": self.pose.theta,
            "is_done": self.is_done,
            "terminal_reason": self.terminal_reason,
            "needs_policy": self.needs_policy,
            "remaining_step_budget_norm": (
                self.observed_state.remaining_step_budget_norm
            ),
            "observed_arrays": {
                name: _encode_episode_array(getattr(self.observed_state, name))
                for name in _EPISODE_ARRAY_DTYPES
            },
        }
        return {
            **body,
            "state_sha256": hashlib.sha256(
                ArtifactStore.canonical_json_bytes(body)
            ).hexdigest(),
        }

    def import_episode_state(self, value: Mapping[str, object]) -> None:
        """校验完整 DTO 后原子恢复 episode；不读取或导出 hidden truth。"""

        state = _copy_episode_state(value)
        expected_fields = {
            "schema_version",
            "scenario_id",
            "scenario_hash",
            "initialized",
            "step_count",
            "consecutive_no_gain_steps",
            "pose_cell_xy",
            "pose_theta",
            "is_done",
            "terminal_reason",
            "needs_policy",
            "remaining_step_budget_norm",
            "observed_arrays",
            "state_sha256",
        }
        if set(state) != expected_fields:
            raise ValueError("episode state field set drift")
        supplied_hash = state.pop("state_sha256")
        actual_hash = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(state)
        ).hexdigest()
        if supplied_hash != actual_hash:
            raise ValueError("episode state hash mismatch")
        if state["schema_version"] != EPISODE_STATE_SCHEMA_VERSION:
            raise ValueError("episode state schema mismatch")
        if (
            state["scenario_id"] != self.scenario_id
            or state["scenario_hash"] != self.scenario_hash
        ):
            raise ValueError("episode state scenario mismatch")

        initialized = state["initialized"]
        step_count = state["step_count"]
        no_gain = state["consecutive_no_gain_steps"]
        is_done = state["is_done"]
        needs_policy = state["needs_policy"]
        reason = state["terminal_reason"]
        if (
            type(initialized) is not bool
            or type(is_done) is not bool
            or type(needs_policy) is not bool
            or type(step_count) is not int
            or not 0 <= step_count <= self.config.max_steps
            or type(no_gain) is not int
            or not 0 <= no_gain <= step_count
            or reason not in ALLOWED_DONE_REASONS
            or is_done != (reason != "none")
            or needs_policy and is_done
        ):
            raise ValueError("episode state lifecycle fields are invalid")
        pose_value = state["pose_cell_xy"]
        theta_value = state["pose_theta"]
        if (
            not isinstance(pose_value, list)
            or len(pose_value) != 2
            or any(type(component) is not int for component in pose_value)
            or not isinstance(theta_value, (int, float))
        ):
            raise ValueError("episode state pose is invalid")
        pose_cell = CellXY(pose_value[0], pose_value[1])
        if not self.observed_state.geometry.in_bounds(pose_cell):
            raise ValueError("episode state pose is out of bounds")
        restored_theta = normalize_theta(float(theta_value))
        if restored_theta != float(theta_value):
            raise ValueError("episode state theta is not normalized")

        remaining = state["remaining_step_budget_norm"]
        expected_remaining = max(
            0.0,
            (self.config.max_steps - step_count) / self.config.max_steps,
        )
        if (
            not isinstance(remaining, (int, float))
            or not math.isfinite(float(remaining))
            or not math.isclose(
                float(remaining),
                expected_remaining,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("episode state remaining step budget is invalid")
        array_records = state["observed_arrays"]
        if not isinstance(array_records, dict) or set(array_records) != set(
            _EPISODE_ARRAY_DTYPES
        ):
            raise ValueError("episode state observed array set drift")
        restored_arrays = {
            name: _decode_episode_array(
                array_records[name],
                expected_dtype=dtype,
                expected_shape=self.observed_state.geometry.shape,
            )
            for name, dtype in _EPISODE_ARRAY_DTYPES.items()
        }
        restored_observed = ObservedMapState(
            geometry=self.observed_state.geometry,
            **restored_arrays,
            remaining_step_budget_norm=float(remaining),
        )
        safety_check = restored_observed.observed_safe_mask.copy()
        restored_observed.recompute_observed_safe_mask(
            min_clearance_m=self.config.min_clearance_m,
            max_slope_deg=self.config.max_traversable_slope_deg,
            traversability_threshold=self.config.traversability_threshold,
        )
        if not np.array_equal(restored_observed.observed_safe_mask, safety_check):
            raise ValueError("episode state observed safety mask is inconsistent")

        restored_pose = PoseXYTheta(pose_cell, restored_theta)
        if initialized:
            restored_action_set = self.frontier_generator.extract(
                restored_observed,
                self._scenario.prior,
                restored_pose,
            )
            restored_observation = self.observation_builder.build(
                self._scenario.prior,
                restored_observed,
                restored_pose,
                restored_action_set,
            )
            expected_needs_policy = (
                not is_done and restored_action_set.candidate_count > 0
            )
            if needs_policy != expected_needs_policy:
                raise ValueError("episode state policy lifecycle is inconsistent")
        else:
            if (
                step_count != 0
                or no_gain != 0
                or is_done
                or needs_policy
                or reason != "none"
            ):
                raise ValueError("uninitialized episode state is inconsistent")
            restored_action_set = _empty_action_set(self.config.frontier_top_m)
            restored_observation = None

        self.observed_state = restored_observed
        self.pose = restored_pose
        self.step_count = step_count
        self.consecutive_no_gain_steps = no_gain
        self.is_done = is_done
        self.terminal_reason = reason
        self.needs_policy = needs_policy
        self.current_action_set = restored_action_set
        self.current_observation = restored_observation
        self.last_reset_diagnostics = None

    def _progress(self, coverage_rate: float) -> ProgressSnapshot:
        return ProgressSnapshot(
            coverage_progress=coverage_rate / self.config.success_coverage_rate,
            step_progress=self.step_count / self.config.max_steps,
            stagnation_progress=self.consecutive_no_gain_steps / self.config.stagnation_no_gain_steps,
        )

    def _validate_fixed_smoke_structural_gates(self, initial_coverage_rate: float) -> None:
        safe_free_count = int(np.count_nonzero(self._coverage_masks.safe_free_mask))
        reachable_safe_count = int(np.count_nonzero(self._coverage_masks.reachable_safe_mask))
        start = self._scenario.start_pose.cell
        if safe_free_count <= 0 or not self._coverage_masks.safe_free_mask[start.y, start.x]:
            raise RuntimeError("fixed Smoke structural gate failed: safe start")
        if reachable_safe_count / safe_free_count < 0.70:
            raise RuntimeError("fixed Smoke structural gate failed: reachable safe ratio")
        if self._coverage_masks.coverable_cell_count <= 0:
            raise RuntimeError("fixed Smoke structural gate failed: coverable denominator")
        if initial_coverage_rate >= self.config.success_coverage_rate:
            raise RuntimeError("fixed Smoke structural gate failed: initial coverage")
        if self.current_action_set.candidate_count <= 0:
            raise RuntimeError("fixed Smoke structural gate failed: default candidate")

    def _frontier_diagnostics(self, *, enforce_oracle: bool = True) -> FrontierDiagnostics:
        if type(enforce_oracle) is not bool:
            raise TypeError("enforce_oracle must be boolean")
        candidate_count = self.current_action_set.candidate_count
        unknown_count = int(np.count_nonzero(~self.observed_state.observed_mask))
        remaining_coverable = int(
            np.count_nonzero(~self.observed_state.observed_mask & self._coverage_masks.coverable_mask)
        )
        component_size = 0
        oracle_count = 0
        if candidate_count == 0:
            audit = audit_frontier_opportunities(
                self.observed_state,
                self.pose,
                planning_safe_mask=self.observed_state.planning_safe_mask,
                sensor_range_m=self.config.sensor_range_m,
                max_slope_deg=self.config.max_traversable_slope_deg,
            )
            component_size = audit.component_size
            oracle_count = audit.opportunity_count
            if enforce_oracle and self._enforce_frontier_oracle and oracle_count > 0:
                raise RuntimeError("production frontier is empty while observed opportunities remain")
        return FrontierDiagnostics(
            candidate_count=candidate_count,
            component_size=component_size,
            unknown_count=unknown_count,
            remaining_unobserved_coverable_count=remaining_coverable,
            oracle_opportunity_count=oracle_count,
        )


def compute_step_reward(
    config: Stage1Config,
    coverage_gain_cells: int,
    coverable_cell_count: int,
    coverage_rate: float,
    invalid_action: bool,
    severe_safety: bool,
) -> float:
    if coverable_cell_count <= 0:
        raise ValueError("coverable_cell_count must be positive")
    reward = config.coverage_gain_reward_scale * (
        coverage_gain_cells / coverable_cell_count
    )
    if coverage_rate >= config.success_coverage_rate:
        reward += config.success_bonus
    if invalid_action:
        reward -= config.invalid_action_penalty
    if severe_safety:
        reward -= config.safety_violation_penalty
    return float(reward)


def resolve_terminal(
    severe_safety: bool,
    coverage_rate: float,
    step_count: int,
    consecutive_no_gain_steps: int,
    has_candidate: bool,
    *,
    success_coverage_rate: float = 0.99,
    max_steps: int = 64,
    stagnation_no_gain_steps: int = 8,
) -> TerminalDecision:
    reason: DoneReason = "none"
    if severe_safety:
        reason = "safety_done"
    elif coverage_rate >= success_coverage_rate:
        reason = "success_done"
    elif step_count >= max_steps:
        reason = "failure_done"
    elif consecutive_no_gain_steps >= stagnation_no_gain_steps:
        reason = "stagnation_done"
    elif not has_candidate:
        reason = "no_candidate_done"
    done = reason != "none"
    return TerminalDecision(
        done=done,
        reason=reason,
        terminal=done,
        bootstrap_value=0.0 if done else 1.0,
    )


def _post_observation_safety_failure(
    observed_safe_mask: np.ndarray,
    planning_safe_mask: np.ndarray,
    path_cells: tuple[CellXY, ...],
    *,
    endpoint: CellXY,
) -> str | None:
    height, width = observed_safe_mask.shape
    if not (0 <= endpoint.x < width and 0 <= endpoint.y < height):
        return "post_observation_endpoint_out_of_bounds"
    if not bool(observed_safe_mask[endpoint.y, endpoint.x]):
        return "post_observation_endpoint_unsafe"
    if not bool(planning_safe_mask[endpoint.y, endpoint.x]):
        return "post_observation_endpoint_unknown_buffer_unsafe"
    for cell in path_cells:
        if not (0 <= cell.x < width and 0 <= cell.y < height):
            return "post_observation_path_out_of_bounds"
        if not bool(observed_safe_mask[cell.y, cell.x]):
            return "post_observation_path_unsafe"
        if not bool(planning_safe_mask[cell.y, cell.x]):
            return "post_observation_path_unknown_buffer_unsafe"
    return None


def _legacy_rule_policy_input(
    state: ObservedMapState,
    prior: LowResolutionPrior,
    pose: PoseXYTheta,
    action_set: FrontierActionSet,
) -> _LegacyRulePolicyInput:
    """Adapt current candidates to the frozen Stage 1 rule-feature semantics."""

    height, width = state.geometry.shape
    features = np.zeros((action_set.candidate_mask.size, 22), dtype=np.float32)
    diagonal = max(math.hypot(width - 1, height - 1), 1.0)
    for index, cell in enumerate(action_set.cells):
        dx = cell.x - pose.cell.x
        dy = cell.y - pose.cell.y
        unknown_count, unknown_dx, unknown_dy = _legacy_unknown_neighbor_statistics(
            state.observed_mask,
            cell,
        )
        observed_blockers = _legacy_observed_blocker_neighbor_count(state, cell)
        prior_x = min(prior.channels.shape[2] - 1, int(cell.x * prior.channels.shape[2] / width))
        prior_y = min(prior.channels.shape[1] - 1, int(cell.y * prior.channels.shape[1] / height))
        if unknown_dx != 0 or unknown_dy != 0:
            recommended_theta = math.atan2(unknown_dy, unknown_dx)
        elif math.hypot(dx, dy) > 0.0:
            recommended_theta = math.atan2(dy, dx)
        else:
            recommended_theta = pose.theta
        features[index, 0:9] = (
            cell.x / max(width - 1, 1),
            cell.y / max(height - 1, 1),
            dx / max(width - 1, 1),
            dy / max(height - 1, 1),
            math.hypot(dx, dy) / diagonal,
            math.sin(recommended_theta),
            math.cos(recommended_theta),
            unknown_count / 8.0,
            observed_blockers / 8.0,
        )
        features[index, 9:14] = (
            state.confidence[cell.y, cell.x],
            state.height[cell.y, cell.x],
            state.slope_deg[cell.y, cell.x] / 30.0,
            state.traversability[cell.y, cell.x],
            1.0,
        )
        features[index, 14:21] = prior.channels[:, prior_y, prior_x]
        features[index, 21] = index / max(action_set.candidate_mask.size - 1, 1)
    pose_features = np.asarray(
        (
            pose.cell.x / max(width - 1, 1),
            pose.cell.y / max(height - 1, 1),
            math.sin(pose.theta),
            math.cos(pose.theta),
            float(np.mean(state.observed_mask, dtype=np.float64)),
            float(state.remaining_step_budget_norm),
        ),
        dtype=np.float32,
    )
    return _LegacyRulePolicyInput(
        frontier_features=features,
        pose_features=pose_features,
        candidate_mask=np.asarray(action_set.candidate_mask, dtype=bool).copy(),
    )


def _legacy_unknown_neighbor_statistics(
    observed_mask: np.ndarray,
    cell: CellXY,
) -> tuple[int, int, int]:
    height, width = observed_mask.shape
    count = 0
    sum_dx = 0
    sum_dy = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x = cell.x + dx
            y = cell.y + dy
            if 0 <= x < width and 0 <= y < height and not observed_mask[y, x]:
                count += 1
                sum_dx += dx
                sum_dy += dy
    return count, sum_dx, sum_dy


def _legacy_observed_blocker_neighbor_count(state: ObservedMapState, cell: CellXY) -> int:
    height, width = state.observed_mask.shape
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x = cell.x + dx
            y = cell.y + dy
            if (
                0 <= x < width
                and 0 <= y < height
                and state.observed_mask[y, x]
                and not state.observed_safe_mask[y, x]
            ):
                count += 1
    return count


def _empty_action_set(top_m: int) -> FrontierActionSet:
    return FrontierActionSet(
        cells=(),
        frontier_features=np.zeros((top_m, 22), dtype=np.float32),
        candidate_mask=np.zeros((top_m,), dtype=bool),
    )


def _copy_episode_state(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("episode state must be a mapping")
    try:
        copied = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("episode state must contain finite JSON values") from exc
    if not isinstance(copied, dict):
        raise ValueError("episode state must be a JSON object")
    return copied


def _encode_episode_array(value: np.ndarray) -> dict[str, object]:
    canonical = np.ascontiguousarray(value)
    payload = canonical.tobytes(order="C")
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "base64": base64.b64encode(payload).decode("ascii"),
    }


def _decode_episode_array(
    value: object,
    *,
    expected_dtype: np.dtype,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {
        "dtype",
        "shape",
        "size_bytes",
        "sha256",
        "base64",
    }:
        raise ValueError("episode state array record is invalid")
    if (
        value["dtype"] != expected_dtype.str
        or value["shape"] != list(expected_shape)
        or type(value["size_bytes"]) is not int
        or not isinstance(value["sha256"], str)
        or not isinstance(value["base64"], str)
    ):
        raise ValueError("episode state array metadata mismatch")
    try:
        payload = base64.b64decode(value["base64"], validate=True)
    except ValueError as exc:
        raise ValueError("episode state array base64 is invalid") from exc
    expected_size = int(np.prod(expected_shape, dtype=np.int64)) * (
        expected_dtype.itemsize
    )
    if (
        len(payload) != expected_size
        or value["size_bytes"] != expected_size
        or hashlib.sha256(payload).hexdigest() != value["sha256"]
    ):
        raise ValueError("episode state array hash or size mismatch")
    return np.frombuffer(payload, dtype=expected_dtype).reshape(expected_shape).copy()
