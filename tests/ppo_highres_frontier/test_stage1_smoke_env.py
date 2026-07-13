from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import hashlib
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = REPO_ROOT / "configs" / "ppo_highres_frontier_smoke_v1.json"


def test_stage1_geometry_api_is_available() -> None:
    assert importlib.util.find_spec("lunar_exploration_ppo.utils.geometry") is not None


def test_geometry_uses_cell_xy_array_yx_centers_and_raster_axis_boundary() -> None:
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, WorldXY

    geometry = GridGeometry(width=128, height=128, resolution_m=0.5)

    assert geometry.array_indices(CellXY(7, 11)) == (11, 7)
    assert geometry.cell_to_world_center(CellXY(0, 0)) == WorldXY(0.25, 0.25)
    assert geometry.cell_to_world_center(CellXY(127, 127)) == WorldXY(63.75, 63.75)
    assert geometry.raster_row_to_cell_y(0) == 127
    assert geometry.cell_y_to_raster_row(127) == 0


def test_geometry_normalizes_theta_and_fails_closed_on_nonfinite_values() -> None:
    from lunar_exploration_ppo.utils.geometry import normalize_theta

    assert normalize_theta(math.pi) == pytest.approx(-math.pi)
    assert normalize_theta(3.0 * math.pi) == pytest.approx(-math.pi)
    assert normalize_theta(-math.pi) == pytest.approx(-math.pi)
    with pytest.raises(ValueError, match="finite"):
        normalize_theta(math.nan)


def test_stage1_config_freezes_smoke_v1_constants_and_dimensions() -> None:
    from lunar_exploration_ppo.configs.stage1 import Stage1Config, load_stage1_config

    config = load_stage1_config(STAGE1_CONFIG)

    assert isinstance(config, Stage1Config)
    assert config.scale_profile == "Smoke v1"
    assert (config.roi_width_m, config.roi_height_m) == (64.0, 64.0)
    assert (config.highres_width, config.highres_height, config.highres_resolution_m) == (128, 128, 0.5)
    assert (config.lowres_width, config.lowres_height, config.lowres_resolution_m) == (32, 32, 2.0)
    assert config.local_crop_size == 64
    assert config.frontier_top_m == 512
    assert config.max_steps == 64
    assert config.stagnation_no_gain_steps == 8
    assert config.min_clearance_m == pytest.approx(config.vehicle_radius_m + config.safety_margin_m)
    assert config.sensor_range_cells == 40
    assert config.hard_path_budget_enabled is False
    assert config.proxy_generator_version == "procedural_lunar_rock_crater_proxy/v1"
    assert config.proxy_density_profile == "medium"
    assert config.proxy_base_seed == 20260710
    assert config.proxy_start_protection_m == 6.0
    assert config.proxy_max_scene_attempts == 64
    assert config.proxy_max_object_attempts == 256


def test_stage1_config_rejects_extra_fields_and_fixed_contract_drift() -> None:
    from lunar_exploration_ppo.configs.stage1 import Stage1Config, load_stage1_config

    payload = load_stage1_config(STAGE1_CONFIG).model_dump(mode="json")
    with pytest.raises(ValidationError):
        Stage1Config.model_validate(payload | {"future_override": True})
    with pytest.raises(ValidationError):
        Stage1Config.model_validate(payload | {"max_steps": 65})
    with pytest.raises(ValidationError):
        Stage1Config.model_validate(payload | {"min_clearance_m": 0.6})


def test_smoke_scenario_is_deterministic_proxy_with_expected_shapes() -> None:
    from lunar_exploration_ppo.env.scenario import ScenarioSource

    source = ScenarioSource()
    first = source.load("smoke-v1")
    second = source.load("smoke-v1")

    assert first.scenario_id == "smoke-v1/procedural-rock-crater/v1"
    assert first.scenario_hash == second.scenario_hash
    assert first.truth.height.shape == (128, 128)
    assert first.truth.hard_obstacle.shape == (128, 128)
    assert first.truth.slope_deg.shape == (128, 128)
    assert first.truth.traversability.shape == (128, 128)
    assert np.array_equal(first.truth.hard_obstacle, second.truth.hard_obstacle)
    assert first.prior.channels.shape == (7, 32, 32)
    assert np.all(np.isfinite(first.prior.channels))
    assert first.prior.value_prior_source == "constant_neutral/v1"
    assert first.truth.provenance["synthetic_source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert first.truth.provenance["physical_obstacle_cells_written"] is False
    assert first.truth.provenance["proxy_generator_version"] == "procedural_lunar_rock_crater_proxy/v1"
    assert first.proxy_catalog.sha256 == second.proxy_catalog.sha256
    assert dict(first.proxy_layer_hashes) == dict(second.proxy_layer_hashes)


def _small_truth(*, blocker: tuple[int, int] | None = None):
    from lunar_exploration_ppo.env.scenario import TruthMap
    from lunar_exploration_ppo.utils.geometry import GridGeometry

    geometry = GridGeometry(width=16, height=16, resolution_m=0.5)
    obstacle = np.zeros(geometry.shape, dtype=bool)
    if blocker is not None:
        obstacle[blocker[1], blocker[0]] = True
    return TruthMap(
        geometry=geometry,
        height=np.zeros(geometry.shape, dtype=float),
        hard_obstacle=obstacle,
        slope_deg=np.zeros(geometry.shape, dtype=float),
        traversability=np.ones(geometry.shape, dtype=float),
        provenance={
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
    )


def test_sensor_reveals_visible_blocker_but_not_cell_behind_and_deduplicates_gain() -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater
    from lunar_exploration_ppo.utils.geometry import CellXY

    truth = _small_truth(blocker=(6, 8))
    state = ObservedMapState.empty(truth.geometry)
    updater = SensorUpdater(range_m=20.0, fov_deg=90.0, ray_angle_step_deg=1.0)
    pose = SensorPose(
        world_xy=truth.geometry.cell_to_world_center(CellXY(2, 8)),
        heading=0.0,
        source="endpoint_theta",
    )

    first = updater.reveal(truth, state, (pose,))
    second = updater.reveal(truth, state, (pose,))

    assert state.observed_mask[8, 6]
    assert state.obstacle[8, 6]
    assert not state.observed_mask[8, 7]
    assert first.newly_observed_count > 0
    assert second.newly_observed_count == 0
    assert first.diagnostics.sample_count == 1
    assert first.diagnostics.ray_count == 91
    assert first.diagnostics.cell_visit_count > first.diagnostics.unique_visible_cell_count
    assert first.diagnostics.sample_sources == ("endpoint_theta",)


def test_observed_safe_mask_keeps_unknown_impassable_and_inflates_known_blocker_once() -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.utils.geometry import CellXY

    truth = _small_truth(blocker=(6, 8))
    state = ObservedMapState.empty(truth.geometry)
    revealed = tuple(CellXY(x, y) for y in range(5, 12) for x in range(3, 10))
    state.reveal_cells(
        truth,
        revealed,
        min_clearance_m=0.5215874761,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )

    assert not state.observed_safe_mask[0, 0]
    assert not state.observed_safe_mask[8, 6]
    assert not state.observed_safe_mask[8, 5]
    assert state.observed_safe_mask[7, 5]
    before = state.observed_safe_mask.copy()
    state.recompute_observed_safe_mask(
        min_clearance_m=0.5215874761,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )
    assert np.array_equal(state.observed_safe_mask, before)


def test_reachability_is_eight_connected_without_diagonal_corner_cutting() -> None:
    from lunar_exploration_ppo.env.reachability import reachable_component
    from lunar_exploration_ppo.utils.geometry import CellXY

    mask = np.array([[True, False], [False, True]], dtype=bool)
    reachable = reachable_component(mask, CellXY(0, 0))

    assert reachable.tolist() == [[True, False], [False, False]]


def test_official_coverage_masks_are_exact_hashed_and_nonempty() -> None:
    from lunar_exploration_ppo.env.coverage import compute_coverage_masks
    from lunar_exploration_ppo.env.scenario import ScenarioSource

    scenario = ScenarioSource().load("smoke-v1")
    masks = compute_coverage_masks(
        scenario.truth,
        scenario.start_pose.cell,
        sensor_range_m=20.0,
        min_clearance_m=0.5215874761,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )

    assert masks.safe_free_mask.shape == (128, 128)
    assert masks.reachable_safe_mask[scenario.start_pose.cell.y, scenario.start_pose.cell.x]
    assert masks.coverable_cell_count > 0
    assert np.all(~masks.coverable_mask | (~scenario.truth.hard_obstacle))
    assert masks.metadata["algorithm_id"] == "exact_reachable_safe_pose_range_los/v1"
    assert masks.metadata["exact"] is True
    assert masks.metadata["precompute_scope"] == "scenario_reset/v1"
    assert len(masks.metadata["sha256"]) == 64


def _initial_observed_scenario():
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.scenario import ScenarioSource
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater

    scenario = ScenarioSource().load("smoke-v1")
    state = ObservedMapState.empty(scenario.truth.geometry)
    updater = SensorUpdater()
    updater.reveal(
        scenario.truth,
        state,
        (
            SensorPose(
                scenario.truth.geometry.cell_to_world_center(scenario.start_pose.cell),
                scenario.start_pose.theta,
                "reset",
            ),
        ),
    )
    return scenario, state, updater


def _frontier_recommendation_row(
    unknown_offsets: tuple[tuple[int, int], ...],
) -> np.ndarray:
    from lunar_exploration_ppo.env.frontier import FrontierGenerator
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.scenario import LowResolutionPrior
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta

    geometry = GridGeometry(width=7, height=7, resolution_m=0.5)
    state = ObservedMapState.empty(geometry)
    state.observed_mask[...] = True
    state.confidence[...] = 1.0
    state.traversability[...] = 1.0
    state.observed_safe_mask[1:4, 3] = True
    target = CellXY(3, 3)
    for dx, dy in unknown_offsets:
        state.observed_mask[target.y + dy, target.x + dx] = False
    prior = LowResolutionPrior(
        channels=np.zeros((7, 2, 2), dtype=np.float32),
        resolution_m=2.0,
        value_prior_source="constant_neutral/v1",
    )
    pose = PoseXYTheta(CellXY(3, 1), -0.75)

    action_set = FrontierGenerator(top_m=49).extract(state, prior, pose)

    row = action_set.frontier_features[action_set.cells.index(target)]
    return row


@pytest.mark.parametrize(
    ("unknown_offsets", "expected_theta"),
    (
        (((1, 0),), 0.0),
        (((1, 1),), math.pi / 4.0),
        (((1, 0), (1, 1)), math.atan2(1.0, 2.0)),
        (((-1, 0), (1, 0)), math.pi / 2.0),
    ),
    ids=("single-side", "diagonal", "multi-neighbour-vector", "symmetric-travel-fallback"),
)
def test_frontier_recommended_theta_points_to_observed_unknown_side(
    unknown_offsets: tuple[tuple[int, int], ...],
    expected_theta: float,
) -> None:
    row = _frontier_recommendation_row(unknown_offsets)

    assert math.atan2(float(row[3]), float(row[2])) == pytest.approx(math.pi / 2.0)
    assert math.hypot(float(row[5]), float(row[6])) == pytest.approx(1.0)
    assert math.atan2(float(row[5]), float(row[6])) == pytest.approx(expected_theta)


def test_frontier_and_observation_are_observed_only_stable_and_finite() -> None:
    from lunar_exploration_ppo.env.frontier import FrontierGenerator
    from lunar_exploration_ppo.policy.observation import ObservationBuilder

    scenario, state, _ = _initial_observed_scenario()
    generator = FrontierGenerator(top_m=512)
    action_set = generator.extract(state, scenario.prior, scenario.start_pose)
    repeated = generator.extract(state, scenario.prior, scenario.start_pose)
    observation = ObservationBuilder().build(scenario.prior, state, scenario.start_pose, action_set)

    assert action_set.candidate_count > 0
    assert action_set.candidate_mask.shape == (512,)
    assert action_set.frontier_features.shape == (512, 22)
    assert action_set.cells == repeated.cells
    assert np.array_equal(action_set.frontier_features, repeated.frontier_features)
    assert all(state.observed_safe_mask[cell.y, cell.x] for cell in action_set.cells)
    assert observation.prior_channels.shape == (7, 32, 32)
    assert observation.coverage_summary.shape == (8, 32, 32)
    assert observation.local_crop.shape == (8, 64, 64)
    assert observation.frontier_features.shape == (512, 22)
    assert observation.pose_features.shape == (6,)
    assert observation.candidate_mask.shape == (512,)
    for value in observation.array_fields():
        assert np.all(np.isfinite(value))


def test_observation_builder_interface_has_no_truth_or_coverable_dependency() -> None:
    from lunar_exploration_ppo.policy.observation import ObservationBuilder

    signature = inspect.signature(ObservationBuilder.build)
    annotations = repr(ObservationBuilder.build.__annotations__)
    fields = vars(ObservationBuilder())

    assert "truth" not in signature.parameters
    assert "coverable_mask" not in signature.parameters
    assert "Truth" not in annotations
    assert "coverable" not in annotations.lower()
    assert not any("truth" in name or "coverable" in name for name in fields)


def test_hidden_truth_mutation_is_invariant_until_sensor_reveal() -> None:
    from lunar_exploration_ppo.env.frontier import FrontierGenerator
    from lunar_exploration_ppo.env.scenario import TruthMap
    from lunar_exploration_ppo.env.sensor_model import SensorPose
    from lunar_exploration_ppo.policy.observation import ObservationBuilder
    from lunar_exploration_ppo.utils.geometry import CellXY

    scenario, state, updater = _initial_observed_scenario()
    generator = FrontierGenerator(top_m=512)
    builder = ObservationBuilder()
    before_actions = generator.extract(state, scenario.prior, scenario.start_pose)
    before = builder.build(scenario.prior, state, scenario.start_pose, before_actions)
    hidden = CellXY(110, 64)
    assert not state.observed_mask[hidden.y, hidden.x]

    changed_obstacle = scenario.truth.hard_obstacle.copy()
    changed_obstacle[hidden.y, hidden.x] = True
    changed_truth = TruthMap(
        geometry=scenario.truth.geometry,
        height=scenario.truth.height.copy(),
        hard_obstacle=changed_obstacle,
        slope_deg=scenario.truth.slope_deg.copy(),
        traversability=scenario.truth.traversability.copy(),
        provenance=scenario.truth.provenance,
    )
    unchanged_actions = generator.extract(state, scenario.prior, scenario.start_pose)
    unchanged = builder.build(scenario.prior, state, scenario.start_pose, unchanged_actions)

    assert before_actions.cells == unchanged_actions.cells
    assert all(np.array_equal(left, right) for left, right in zip(before.array_fields(), unchanged.array_fields(), strict=True))

    updater.reveal(
        changed_truth,
        state,
        (
            SensorPose(
                changed_truth.geometry.cell_to_world_center(CellXY(106, 64)),
                0.0,
                "endpoint_theta",
            ),
        ),
    )
    assert state.observed_mask[hidden.y, hidden.x]
    assert state.obstacle[hidden.y, hidden.x]


def test_planner_adapter_uses_final_mask_corner_safe_path_cells_and_center_world_points() -> None:
    from lunar_exploration_ppo.integrations.path_planner_adapter import PathPlannerAdapter
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, WorldXY

    geometry = GridGeometry(width=4, height=4, resolution_m=0.5)
    adapter = PathPlannerAdapter(geometry)
    safe = np.ones(geometry.shape, dtype=bool)
    valid = adapter.validate(safe, CellXY(0, 0), CellXY(2, 2), math.pi / 2.0)

    assert valid.valid
    assert valid.failure_reason == "none"
    assert valid.path_cells == (CellXY(0, 0), CellXY(1, 1), CellXY(2, 2))
    assert valid.path_world == (WorldXY(0.25, 0.25), WorldXY(0.75, 0.75), WorldXY(1.25, 1.25))
    assert valid.target_theta == pytest.approx(math.pi / 2.0)
    assert valid.diagnostics["inflation_applied"] is False
    assert valid.diagnostics["planner_footprint_radius_m"] is None
    assert valid.diagnostics["original_blocked_count"] == valid.diagnostics["inflated_blocked_count"]

    corner_mask = np.array([[True, False], [False, True]], dtype=bool)
    corner = PathPlannerAdapter(GridGeometry(2, 2, 0.5)).validate(
        corner_mask, CellXY(0, 0), CellXY(1, 1), 0.0
    )
    assert not corner.valid
    assert corner.failure_reason == "target_unreachable"


@pytest.mark.parametrize(
    ("start", "target", "theta", "reason"),
    [
        ((-1, 0), (1, 1), 0.0, "start_out_of_bounds"),
        ((0, 0), (4, 1), 0.0, "target_out_of_bounds"),
        ((0, 0), (1, 1), math.nan, "invalid_theta"),
    ],
)
def test_planner_adapter_fails_stably_without_crash(start, target, theta, reason) -> None:
    from lunar_exploration_ppo.integrations.path_planner_adapter import PathPlannerAdapter
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    adapter = PathPlannerAdapter(GridGeometry(4, 4, 0.5))
    result = adapter.validate(np.ones((4, 4), dtype=bool), CellXY(*start), CellXY(*target), theta)

    assert not result.valid
    assert result.failure_reason == reason
    assert result.path_cells == ()


def test_planner_adapter_unknown_is_blocked_and_never_reads_astar_path_world() -> None:
    from lunar_exploration_ppo.integrations import path_planner_adapter as adapter_module
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    source = inspect.getsource(adapter_module)
    assert ".path_world" not in source
    safe = np.zeros((4, 4), dtype=bool)
    safe[1, 1] = True
    safe[1, 2] = True
    adapter = adapter_module.PathPlannerAdapter(GridGeometry(4, 4, 0.5))

    blocked = adapter.validate(safe, CellXY(1, 1), CellXY(3, 1), 0.0)

    assert not blocked.valid
    assert blocked.failure_reason == "target_unsafe"


def test_action_sensor_samples_use_segment_tangent_then_endpoint_theta() -> None:
    from lunar_exploration_ppo.env.action_execution import build_sensor_poses
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    geometry = GridGeometry(8, 8, 0.5)
    path = tuple(CellXY(x, 2) for x in range(5))
    poses, diagnostics = build_sensor_poses(path, geometry, target_theta=math.pi / 2.0, step_m=1.0)

    assert tuple(pose.source for pose in poses) == ("path_tangent", "path_tangent", "endpoint_theta")
    assert tuple(pose.heading for pose in poses[:-1]) == pytest.approx((0.0, 0.0))
    assert poses[-1].heading == pytest.approx(math.pi / 2.0)
    assert poses[-1].world_xy == geometry.cell_to_world_center(CellXY(4, 2))
    assert diagnostics.path_sample_count == 2
    assert diagnostics.endpoint_sample_count == 1
    assert diagnostics.path_length_m == pytest.approx(2.0)

    rotation, _ = build_sensor_poses((CellXY(4, 2),), geometry, target_theta=-math.pi / 2.0, step_m=1.0)
    assert len(rotation) == 1
    assert rotation[0].source == "endpoint_theta"
    assert rotation[0].heading == pytest.approx(-math.pi / 2.0)


def _stage1_env(**kwargs):
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    return LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG), **kwargs)


def test_rule_action_selection_is_conservative_risk_first_with_stable_row_tie_break() -> None:
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    features = np.zeros((512, 22), dtype=np.float32)
    mask = np.zeros((512,), dtype=bool)
    mask[:5] = True
    # Freeze the lexicographic order: fewer observed-blocker neighbours,
    # fewer unknown neighbours, farther distance, then the lowest stable row.
    features[0, 4], features[0, 7], features[0, 8] = 0.9, 0.0, 0.25
    features[1, 4], features[1, 7], features[1, 8] = 0.95, 0.75, 0.125
    features[2, 4], features[2, 7], features[2, 8] = 0.2, 0.25, 0.125
    features[3, 4], features[3, 7], features[3, 8] = 0.8, 0.25, 0.125
    features[4, 4], features[4, 7], features[4, 8] = 0.8, 0.25, 0.125
    features[3, 2:4] = (1.0, 0.0)
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 64, 64), dtype=np.float32),
        frontier_features=features,
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=mask,
    )

    action = select_rule_action(observation)

    assert action.candidate_index == 3
    assert action.target_theta == pytest.approx(0.0)


def test_rule_action_theta_prefers_selected_recommendation_over_conflicting_travel_direction() -> None:
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    features = np.zeros((512, 22), dtype=np.float32)
    mask = np.zeros((512,), dtype=bool)
    mask[3] = True
    mask[7] = True
    features[3, 2:4] = (1.0, 0.0)
    features[3, 5:7] = (1.0, 0.0)
    features[3, 7] = 1.0
    features[7, 5:7] = (math.nan, math.inf)
    features[7, 8] = 1.0
    pose_features = np.zeros((6,), dtype=np.float32)
    pose_features[2:4] = (-1.0, 0.0)
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 64, 64), dtype=np.float32),
        frontier_features=features,
        pose_features=pose_features,
        candidate_mask=mask,
    )

    action = select_rule_action(observation)

    assert action.candidate_index == 3
    assert action.target_theta == pytest.approx(math.pi / 2.0)


@pytest.mark.parametrize(
    "recommendation",
    (
        (0.0, 0.0),
        (1.0e-20, -1.0e-20),
        (math.nan, 1.0),
        (1.0, math.inf),
    ),
    ids=("zero", "near-zero", "nan", "infinite"),
)
def test_rule_action_theta_falls_back_from_invalid_recommendation_to_selected_travel_direction(
    recommendation: tuple[float, float],
) -> None:
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    features = np.zeros((512, 22), dtype=np.float32)
    mask = np.zeros((512,), dtype=bool)
    mask[4] = True
    features[4, 2:4] = (0.0, -1.0)
    features[4, 5:7] = recommendation
    pose_features = np.zeros((6,), dtype=np.float32)
    pose_features[2:4] = (1.0, 0.0)
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 64, 64), dtype=np.float32),
        frontier_features=features,
        pose_features=pose_features,
        candidate_mask=mask,
    )

    action = select_rule_action(observation)

    assert action.candidate_index == 4
    assert action.target_theta == pytest.approx(-math.pi / 2.0)


@pytest.mark.parametrize(
    "travel_direction",
    (
        (0.0, 0.0),
        (1.0e-20, -1.0e-20),
        (math.nan, 1.0),
        (math.inf, -math.inf),
    ),
    ids=("zero", "near-zero", "nan", "infinite"),
)
def test_rule_action_theta_falls_back_to_pose_heading_when_recommendation_and_travel_are_invalid(
    travel_direction: tuple[float, float],
) -> None:
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    features = np.zeros((512, 22), dtype=np.float32)
    mask = np.zeros((512,), dtype=bool)
    mask[4] = True
    features[4, 2:4] = travel_direction
    features[4, 5:7] = (0.0, 0.0)
    pose_features = np.zeros((6,), dtype=np.float32)
    pose_features[2:4] = (1.0, 0.0)
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 64, 64), dtype=np.float32),
        frontier_features=features,
        pose_features=pose_features,
        candidate_mask=mask,
    )

    action = select_rule_action(observation)

    assert action.candidate_index == 4
    assert action.target_theta == pytest.approx(math.pi / 2.0)


def test_env_and_workflow_rule_policy_share_one_conservative_selector() -> None:
    from lunar_exploration_ppo.env import env as env_module
    from lunar_exploration_ppo.env.frontier import FrontierActionSet
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.utils.geometry import CellXY
    from lunar_exploration_ppo.workflows import stage1 as workflow_module

    assert (
        workflow_module.select_conservative_frontier_candidate_index
        is env_module.select_conservative_frontier_candidate_index
    )
    assert workflow_module.select_conservative_rule_action is env_module.select_conservative_rule_action
    for module in (env_module, workflow_module):
        assert not hasattr(module, "select_gain_first_candidate_index")
        assert not hasattr(module, "select_gain_first_rule_action")
    features = np.zeros((512, 22), dtype=np.float32)
    mask = np.zeros((512,), dtype=bool)
    mask[:5] = True
    features[:5, 4] = (0.9, 0.95, 0.2, 0.8, 0.8)
    features[:5, 7] = (0.0, 0.75, 0.25, 0.25, 0.25)
    features[:5, 8] = (0.25, 0.125, 0.125, 0.125, 0.125)
    features[3, 2:4] = (1.0, 0.0)
    cells = (
        CellXY(15, 64),
        CellXY(16, 63),
        CellXY(17, 64),
        CellXY(16, 65),
        CellXY(17, 65),
    )
    action_set = FrontierActionSet(cells=cells, frontier_features=features, candidate_mask=mask)
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 64, 64), dtype=np.float32),
        frontier_features=features.copy(),
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=mask.copy(),
    )
    env = _stage1_env()
    env.current_action_set = action_set
    env.current_observation = observation
    env.needs_policy = True

    env_action = env.select_rule_action(observation)
    workflow_action = workflow_module.select_rule_action(observation)

    assert env_action == workflow_action
    assert env_action.candidate_index == 3


def test_environment_reset_runs_initial_scan_without_action_step_or_reward() -> None:
    env = _stage1_env()

    observation = env.reset()

    assert env.step_count == 0
    assert env.last_reset_diagnostics.reward == 0.0
    assert env.last_reset_diagnostics.trainable is False
    assert env.last_reset_diagnostics.sensor.sample_sources == ("reset",)
    assert np.count_nonzero(env.observed_state.observed_mask) > 0
    assert observation.candidate_mask.shape == (512,)
    assert np.all(np.isfinite(observation.pose_features))
    assert env.coverable_cell_count > 0


def test_valid_rule_action_executes_path_observes_and_uses_exact_reward_formula() -> None:
    env = _stage1_env()
    before_observation = env.reset()
    before_count = int(np.count_nonzero(env.observed_state.observed_mask))
    action = env.select_rule_action(before_observation)

    result = env.step(action)

    assert action.candidate_index < env.config.frontier_top_m
    assert result.trainable
    assert result.coverage_gain_cells >= 0
    assert np.count_nonzero(env.observed_state.observed_mask) >= before_count
    assert result.diagnostics.planner["inflation_applied"] is False
    assert "path_tangent" in result.diagnostics.sensor.sample_sources
    assert result.diagnostics.sensor.sample_sources[-1] == "endpoint_theta"
    expected = 100.0 * result.coverage_gain_cells / env.coverable_cell_count
    if result.coverage_rate >= 0.99:
        expected += 100.0
    assert result.reward == pytest.approx(expected)


def test_invalid_sampled_action_is_trainable_penalized_and_does_not_move_or_observe() -> None:
    from lunar_exploration_ppo.env.env import EnvAction

    env = _stage1_env()
    env.reset()
    pose_before = env.pose
    observed_before = env.observed_state.observed_mask.copy()

    result = env.step(EnvAction(candidate_index=512, target_theta=0.0))

    assert result.reward == -2.0
    assert result.coverage_gain_cells == 0
    assert result.trainable
    assert not result.done
    assert result.reason == "none"
    assert env.pose == pose_before
    assert np.array_equal(env.observed_state.observed_mask, observed_before)


def test_pre_execution_planner_failure_is_invalid_non_safety_transition() -> None:
    env = _stage1_env()
    observation = env.reset()
    action = env.select_rule_action(observation)
    target = env.current_action_set.cells[action.candidate_index]
    pose_before = env.pose
    observed_before = env.observed_state.observed_mask.copy()
    env.observed_state.observed_safe_mask[target.y, target.x] = False

    result = env.step(action)

    assert result.reward == -2.0
    assert not result.done
    assert not result.terminal
    assert result.reason == "none"
    assert result.bootstrap_value == 1.0
    assert result.trainable
    assert result.diagnostics.invalid_action is True
    assert result.diagnostics.safety_violation is False
    assert result.diagnostics.sensor is None
    assert result.diagnostics.planner["failure_reason"] == "target_unsafe"
    assert result.diagnostics.planner["failure_classification"] == "pre_execution_invalid_action"
    assert env.pose == pose_before
    assert np.array_equal(env.observed_state.observed_mask, observed_before)


class _InjectedSevereExecutionSafetyChecker:
    def check(self, safe_mask, path_cells):
        assert safe_mask.ndim == 2
        assert path_cells
        return "collision_detected"


def test_explicit_execution_safety_failure_is_terminal_and_subtracts_safety_penalty() -> None:
    env = _stage1_env(execution_safety_checker=_InjectedSevereExecutionSafetyChecker())
    observation = env.reset()
    action = env.select_rule_action(observation)
    pose_before = env.pose
    observed_before = env.observed_state.observed_mask.copy()

    result = env.step(action)

    assert result.reward == -20.0
    assert result.done and result.terminal
    assert result.reason == "safety_done"
    assert result.bootstrap_value == 0.0
    assert result.trainable
    assert result.diagnostics.invalid_action is False
    assert result.diagnostics.safety_violation is True
    assert result.diagnostics.sensor is None
    assert result.diagnostics.planner["failure_classification"] == "none"
    assert result.diagnostics.planner["execution_safety_failure_reason"] == "collision_detected"
    assert env.pose == pose_before
    assert np.array_equal(env.observed_state.observed_mask, observed_before)


def _post_reveal_clearance_env(*, blocker_kind: str, blocker_offset: int = 1):
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.scenario import ScenarioBundle, ScenarioSource, TruthMap

    base = ScenarioSource().load("smoke-v1")
    geometry = base.truth.geometry
    target_x, target_y = 56, 64
    blocker_x = target_x + blocker_offset
    height = np.zeros(geometry.shape, dtype=np.float64)
    hard_obstacle = np.zeros(geometry.shape, dtype=bool)
    slope_deg = np.zeros(geometry.shape, dtype=np.float64)
    traversability = np.ones(geometry.shape, dtype=np.float64)
    if blocker_kind == "hidden_rock":
        hard_obstacle[target_y, blocker_x] = True
    elif blocker_kind == "hidden_steep_terrain":
        slope_deg[target_y, blocker_x] = 31.0
        traversability[target_y, blocker_x] = 0.0
    else:
        raise ValueError("unsupported blocker kind")
    truth = TruthMap(
        geometry=geometry,
        height=height,
        hard_obstacle=hard_obstacle,
        slope_deg=slope_deg,
        traversability=traversability,
        provenance=base.truth.provenance,
    )
    scenario = ScenarioBundle(
        scenario_id=f"post-reveal-{blocker_kind}-{blocker_offset}",
        scenario_hash=hashlib.sha256(
            f"post-reveal-{blocker_kind}-{blocker_offset}".encode("utf-8")
        ).hexdigest(),
        truth=truth,
        prior=base.prior,
        start_pose=base.start_pose,
        proxy_catalog=base.proxy_catalog,
        proxy_layer_hashes=base.proxy_layer_hashes,
    )

    class _Source:
        def load(self, key: str) -> ScenarioBundle:
            assert key == "smoke-v1"
            return scenario

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG), scenario_source=_Source())
    observation = env.reset()
    from lunar_exploration_ppo.utils.geometry import CellXY

    target = CellXY(target_x, target_y)
    assert target in env.current_action_set.cells
    assert env.observed_state.observed_safe_mask[target_y, target_x]
    assert not env.observed_state.observed_mask[target_y, blocker_x]
    return env, observation, target, blocker_x


@pytest.mark.parametrize("blocker_kind", ["hidden_rock", "hidden_steep_terrain"])
def test_post_reveal_endpoint_clearance_failure_is_severe_safety(blocker_kind: str) -> None:
    from lunar_exploration_ppo.env.env import EnvAction

    env, _, target, blocker_x = _post_reveal_clearance_env(blocker_kind=blocker_kind)
    candidate_index = env.current_action_set.cells.index(target)

    result = env.step(EnvAction(candidate_index=candidate_index, target_theta=0.0))

    expected_without_safety = 100.0 * result.coverage_gain_cells / env.coverable_cell_count
    if result.coverage_rate >= 0.99:
        expected_without_safety += 100.0
    assert env.observed_state.observed_mask[target.y, blocker_x]
    assert not env.observed_state.observed_safe_mask[target.y, target.x]
    assert result.done and result.terminal
    assert result.reason == "safety_done"
    assert result.bootstrap_value == 0.0
    assert result.trainable
    assert result.diagnostics.invalid_action is False
    assert result.diagnostics.safety_violation is True
    assert result.diagnostics.planner["post_observation_safety_failure_reason"] == (
        "post_observation_endpoint_unsafe"
    )
    assert result.reward == pytest.approx(expected_without_safety - 20.0)


def test_post_reveal_distant_blocker_keeps_endpoint_safe() -> None:
    from lunar_exploration_ppo.env.env import EnvAction

    env, _, target, blocker_x = _post_reveal_clearance_env(
        blocker_kind="hidden_rock",
        blocker_offset=2,
    )
    candidate_index = env.current_action_set.cells.index(target)

    result = env.step(EnvAction(candidate_index=candidate_index, target_theta=0.0))

    assert env.observed_state.observed_mask[target.y, blocker_x]
    assert env.observed_state.observed_safe_mask[target.y, target.x]
    assert result.diagnostics.invalid_action is False
    assert result.diagnostics.safety_violation is False
    assert result.diagnostics.planner["post_observation_safety_failure_reason"] == "none"
    assert result.reason != "safety_done"


def test_step_reward_adds_success_bonus_at_exact_099_but_not_immediately_below() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import compute_step_reward

    config = load_stage1_config(STAGE1_CONFIG)
    at_boundary = compute_step_reward(
        config,
        coverage_gain_cells=0,
        coverable_cell_count=100,
        coverage_rate=0.99,
        invalid_action=False,
        severe_safety=False,
    )
    below_boundary = compute_step_reward(
        config,
        coverage_gain_cells=0,
        coverable_cell_count=100,
        coverage_rate=math.nextafter(0.99, 0.0),
        invalid_action=False,
        severe_safety=False,
    )

    assert at_boundary == 100.0
    assert below_boundary == 0.0


def test_step_reward_and_terminal_priority_combine_at_step64_and_safety_boundary() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import compute_step_reward, resolve_terminal

    config = load_stage1_config(STAGE1_CONFIG)
    step64_success = resolve_terminal(False, 0.99, 64, 0, True)
    step64_failure = resolve_terminal(False, math.nextafter(0.99, 0.0), 64, 0, True)
    safety = resolve_terminal(True, 0.99, 64, 0, True)

    assert step64_success.reason == "success_done"
    assert step64_failure.reason == "failure_done"
    assert safety.reason == "safety_done"
    assert compute_step_reward(config, 0, 100, 0.99, False, False) == 100.0
    assert compute_step_reward(config, 0, 100, math.nextafter(0.99, 0.0), False, False) == 0.0
    assert compute_step_reward(config, 0, 100, 0.99, False, True) == 80.0


def test_done_priority_success_boundary_budget_stagnation_and_no_candidate() -> None:
    from lunar_exploration_ppo.env.env import resolve_terminal

    success = resolve_terminal(
        severe_safety=True,
        coverage_rate=0.99,
        step_count=64,
        consecutive_no_gain_steps=8,
        has_candidate=False,
    )
    assert success.reason == "safety_done"
    success = resolve_terminal(
        severe_safety=False,
        coverage_rate=0.99,
        step_count=64,
        consecutive_no_gain_steps=8,
        has_candidate=False,
    )
    assert success.reason == "success_done"
    budget = resolve_terminal(False, math.nextafter(0.99, 0.0), 64, 8, False)
    stagnation = resolve_terminal(False, 0.5, 7, 8, False)
    no_candidate = resolve_terminal(False, 0.5, 7, 7, False)
    assert budget.reason == "failure_done"
    assert stagnation.reason == "stagnation_done"
    assert no_candidate.reason == "no_candidate_done"
    assert all(decision.terminal and decision.bootstrap_value == 0.0 for decision in (success, budget, stagnation, no_candidate))


class _EmptyFrontierGenerator:
    def extract(self, observed_state, prior, pose):
        from lunar_exploration_ppo.env.frontier import FrontierActionSet

        return FrontierActionSet(
            cells=(),
            frontier_features=np.zeros((512, 22), dtype=np.float32),
            candidate_mask=np.zeros((512,), dtype=bool),
        )


class _FirstThenEmptyFrontierGenerator:
    def __init__(self):
        from lunar_exploration_ppo.env.frontier import FrontierGenerator

        self.delegate = FrontierGenerator(top_m=512)
        self.calls = 0

    def extract(self, observed_state, prior, pose):
        self.calls += 1
        if self.calls == 1:
            return self.delegate.extract(observed_state, prior, pose)
        return _EmptyFrontierGenerator().extract(observed_state, prior, pose)


def test_reset_empty_candidate_is_nontrainable_and_skips_policy_state() -> None:
    env = _stage1_env(frontier_generator=_EmptyFrontierGenerator())

    observation = env.reset()

    assert not np.any(observation.candidate_mask)
    assert env.is_done
    assert not env.needs_policy
    assert env.terminal_reason == "no_candidate_done"
    assert env.last_reset_diagnostics.trainable is False
    assert env.last_reset_diagnostics.fake_logprob_created is False


def test_action_after_which_next_candidate_is_empty_closes_sampled_transition() -> None:
    env = _stage1_env(frontier_generator=_FirstThenEmptyFrontierGenerator())
    observation = env.reset()
    action = env.select_rule_action(observation)

    result = env.step(action)

    assert result.trainable
    assert result.done and result.terminal
    assert result.reason == "no_candidate_done"
    assert result.bootstrap_value == 0.0
    assert not np.any(result.observation.candidate_mask)


def test_eight_zero_gain_invalid_actions_end_in_true_stagnation_terminal() -> None:
    from lunar_exploration_ppo.env.env import EnvAction

    env = _stage1_env()
    env.reset()
    results = [env.step(EnvAction(candidate_index=512, target_theta=0.0)) for _ in range(8)]

    assert all(not result.done for result in results[:-1])
    assert results[-1].reason == "stagnation_done"
    assert results[-1].terminal
    assert results[-1].bootstrap_value == 0.0
    assert env.consecutive_no_gain_steps == 8


def _all_numeric_values_are_finite(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numeric_values_are_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numeric_values_are_finite(item) for item in value)
    return True


def test_run_episode_skips_policy_for_reset_empty_without_fake_transition() -> None:
    from lunar_exploration_ppo.workflows.stage1 import run_episode

    env = _stage1_env(frontier_generator=_EmptyFrontierGenerator())
    calls = 0

    def forbidden_policy(observation, environment):
        nonlocal calls
        calls += 1
        raise AssertionError("policy must not run for reset-empty episode")

    episode = run_episode(env, episode_index=0, policy=forbidden_policy)

    assert calls == 0
    assert episode.transition_count == 0
    assert episode.done_reason == "no_candidate_done"
    assert episode.reset_empty is True
    assert episode.trainable_transition_count == 0


def test_complete_smoke_episode_runs_from_reset_to_terminal() -> None:
    from lunar_exploration_ppo.workflows.stage1 import run_episode, select_rule_action

    env = _stage1_env()
    episode = run_episode(
        env,
        episode_index=0,
        policy=select_rule_action,
    )

    assert 1 <= episode.transition_count <= 64
    assert episode.trainable_transition_count == episode.transition_count
    assert episode.done_reason in {"success_done", "failure_done", "stagnation_done", "no_candidate_done", "safety_done"}
    assert math.isfinite(episode.total_reward)
    assert math.isfinite(episode.final_coverage_rate)


def test_conservative_smoke_contract_is_stable_across_python_hash_seeds() -> None:
    source = """
import json
from pathlib import Path
from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env.env import LunarExplorationEnv
from lunar_exploration_ppo.workflows.stage1 import select_rule_action

root = Path.cwd()
env = LunarExplorationEnv(load_stage1_config(root / 'configs/ppo_highres_frontier_smoke_v1.json'))
observation = env.reset()
steps = 0
final = None
while not env.is_done:
    final = env.step(select_rule_action(observation))
    observation = final.observation
    steps += 1
print(json.dumps({'steps': steps, 'reason': env.terminal_reason, 'coverage': final.coverage_rate}))
"""
    results: list[dict[str, object]] = []
    for python_hash_seed in ("0", "1", "2", "42"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = python_hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    assert all(result == results[0] for result in results[1:])
    assert all(result["reason"] == "success_done" for result in results)
    assert all(float(result["coverage"]) >= 0.99 for result in results)


def test_ten_episode_workflow_writes_finite_append_only_machine_artifacts(tmp_path: Path) -> None:
    from lunar_exploration_ppo.workflows.stage1 import run_stage1_smoke_workflow

    result = run_stage1_smoke_workflow(
        config_path=STAGE1_CONFIG,
        run_id="pytest-stage1-smoke",
        base_output_root=tmp_path,
    )
    stage = result.stage_root
    expected = {
        "config.json",
        "summary.json",
        "routing.json",
        "manifest.json",
        "report.md",
        "metrics.jsonl",
        "phase-state.jsonl",
    }

    assert {path.name for path in stage.iterdir()} == expected
    assert not (stage / "approval.json").exists()
    assert not (stage / "gate.json").exists()
    assert not (stage / "review.json").exists()
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    routing = json.loads((stage / "routing.json").read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in (stage / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    phases = [json.loads(line) for line in (stage / "phase-state.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["state"] == "machine_passed"
    assert summary["episodes_completed"] == 10
    assert summary["episodes_requested"] == 10
    assert summary["termination_counts"] == {"success_done": 10}
    assert summary["mean_final_coverage_rate"] >= 0.99
    assert summary["scenario_id"] == "smoke-v1/procedural-rock-crater/v1"
    assert len(summary["scenario_hash"]) == 64
    assert summary["coverage_mask"]["exact"] is True
    assert summary["proxy_morphology"] == {
        "proxy_generator_version": "procedural_lunar_rock_crater_proxy/v1",
        "density_profile": "medium",
        "generation_attempt": summary["proxy_morphology"]["generation_attempt"],
        "rock_count": summary["proxy_morphology"]["rock_count"],
        "crater_count": summary["proxy_morphology"]["crater_count"],
        "object_catalog_sha256": summary["proxy_morphology"]["object_catalog_sha256"],
        "layer_hashes": summary["proxy_morphology"]["layer_hashes"],
    }
    assert 0 <= summary["proxy_morphology"]["generation_attempt"] < 64
    assert 24 <= summary["proxy_morphology"]["rock_count"] <= 36
    assert 4 <= summary["proxy_morphology"]["crater_count"] <= 6
    assert len(summary["proxy_morphology"]["object_catalog_sha256"]) == 64
    assert set(summary["proxy_morphology"]["layer_hashes"]) == {
        "height",
        "hard_obstacle",
        "slope",
        "traversability",
    }
    assert routing == {
        "schema_version": "ppo_highres_frontier_stage1_routing/v1",
        "run_id": "pytest-stage1-smoke",
        "route": "awaiting_independent_review",
        "human_approval_required": True,
        "next_stage_entered": False,
    }
    assert phases == [
        {
            "route": "awaiting_independent_review",
            "run_id": "pytest-stage1-smoke",
            "state": "machine_passed",
        }
    ]
    assert len(metrics) > 10
    assert sum(metric["progress_type"] == "reset" for metric in metrics) == 10
    assert all(metric["scale_profile"] == "Smoke v1" for metric in metrics)
    assert all(metric["reason"] in {"success_done", "failure_done", "stagnation_done", "no_candidate_done", "safety_done", "none"} for metric in metrics)
    assert all(_all_numeric_values_are_finite(metric) for metric in metrics)
    assert all("timestamp" in metric and "path_length_m" in metric and "sensor_ray_count" in metric for metric in metrics)
    terminal_actions = [
        metric for metric in metrics if metric["progress_type"] == "action" and metric["done"]
    ]
    assert len(terminal_actions) == 10
    assert all(metric["reason"] == "success_done" for metric in terminal_actions)
    assert all(metric["coverage_rate"] >= 0.99 for metric in terminal_actions)
    episode_signatures = {
        (
            metric["step"],
            metric["reason"],
            metric["coverage_rate"],
        )
        for metric in terminal_actions
    }
    assert len(episode_signatures) == 1

    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    assert {entry["path"] for entry in manifest["artifacts"]} == expected - {"manifest.json"}
    for entry in manifest["artifacts"]:
        payload = (stage / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()


def test_stage1_import_isolation_and_only_adapter_imports_path_planner() -> None:
    source_root = REPO_ROOT / "src" / "lunar_exploration_ppo"
    forbidden = ("model_explorer", "scripts.xunce_", "sys.path")
    direct_planner_importers: list[Path] = []
    for source_file in source_root.rglob("*.py"):
        source = source_file.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), source_file
        tree = ast.parse(source, filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "path_planner" or alias.name.startswith("path_planner.")
                for alias in node.names
            ):
                direct_planner_importers.append(source_file)
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "path_planner" or node.module.startswith("path_planner.")
            ):
                direct_planner_importers.append(source_file)
    assert {
        path.relative_to(source_root).as_posix() for path in direct_planner_importers
    } == {"integrations/path_planner_adapter.py"}


def _foundation_gate_fixture(tmp_path: Path):
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.workflows.gates import NO_CHECKPOINT_SENTINEL

    environment_root = tmp_path / "environment-lock"
    environment_root.mkdir(parents=True)
    payload = environment_root / "python.txt"
    payload.write_text("Python 3.12\n", encoding="utf-8")
    environment_manifest = environment_root / "environment-lock-manifest.json"
    environment_manifest.write_bytes(
        ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "ppo_foundation_environment_lock_manifest/v1",
                "files": [
                    {
                        "file": "python.txt",
                        "size_bytes": len(payload.read_bytes()),
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    resolved = environment_manifest.resolve()
    environment_gate_hash = hashlib.sha256()
    environment_gate_hash.update(b"gate-environment-file/v1\0")
    environment_gate_hash.update(resolved.as_posix().encode("utf-8"))
    environment_gate_hash.update(b"\0")
    environment_gate_hash.update(resolved.read_bytes())
    foundation_gate = tmp_path / "foundation-gate.json"
    foundation_gate.write_bytes(
        ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "foundation_gate_record/v1",
                "state": "next_stage",
                "history": [],
                "bindings": {
                    "authorized_next_stage": "ppo_highres_frontier_stage1_smoke_environment/v1",
                    "checkpoint_hash": hashlib.sha256(NO_CHECKPOINT_SENTINEL.encode("utf-8")).hexdigest(),
                    "environment_hash": environment_gate_hash.hexdigest(),
                },
                "context": {"environment_manifest": str(environment_manifest)},
            }
        )
    )
    return foundation_gate, environment_manifest


def _write_stage1_clean_review_report(
    path: Path,
    *,
    review_package: Path,
    reviewed_tree: str,
) -> None:
    payload = review_package.read_bytes()
    fields = {
        "schema_version": "ppo_highres_frontier_stage1_review_report/v2",
        "spec_verdict": "APPROVED",
        "quality_verdict": "APPROVED",
        "critical_count": 0,
        "important_count": 0,
        "minor_count": 0,
        "final_gate_conclusion": "READY_FOR_HUMAN_APPROVAL",
        "foundation_base_commit": "7378737a0d18ce6e4779e30605b557bd1ab25e6e",
        "reviewed_prospective_git_tree": reviewed_tree,
        "review_package_sha256": hashlib.sha256(payload).hexdigest(),
        "review_package_bytes": len(payload),
        "review_package_lf_count": payload.count(b"\n"),
        "review_package_logical_line_count": len(payload.splitlines()),
    }
    path.write_text(
        "\n".join(
            [
                "---",
                *(f"{key}: {value}" for key, value in fields.items()),
                "---",
                "",
                "# Independent Stage 1 review",
                "",
                "Specification compliance verdict: APPROVED",
                "Code quality verdict: APPROVED",
                "Final gate conclusion: READY_FOR_HUMAN_APPROVAL",
                "",
                "All required checks completed cleanly.",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_stage1_git_native_review_package(path: Path) -> None:
    from lunar_exploration_ppo.workflows.stage1_source import STAGE1_REVIEWED_PATHS

    temporary_root = Path("D:/xunce/tmp/ppo_frontier/git-index")
    temporary_root.mkdir(parents=True, exist_ok=True)
    index_path = temporary_root / f"stage1-smoke-test-{uuid.uuid4().hex}.index"
    lock_path = Path(str(index_path) + ".lock")
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    paths = sorted(STAGE1_REVIEWED_PATHS)
    try:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "read-tree", "7378737a0d18ce6e4779e30605b557bd1ab25e6e"],
            check=True,
            capture_output=True,
            env=environment,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "update-index", "--add", "--remove", "--", *paths],
            check=True,
            capture_output=True,
            env=environment,
        )
        payload = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-renames",
                "7378737a0d18ce6e4779e30605b557bd1ab25e6e",
                "--",
                *paths,
            ],
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
        assert payload.startswith(b"diff --git ")
        path.write_bytes(payload)
    finally:
        if lock_path.is_file():
            lock_path.unlink()
        if index_path.is_file():
            index_path.unlink()


def _install_gate_committed_source_snapshot_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from lunar_exploration_ppo.workflows.stage1_source import Stage1SourceSnapshot

    original_capture = Stage1SourceSnapshot.capture.__func__
    review_snapshot = original_capture(
        Stage1SourceSnapshot,
        REPO_ROOT,
        base_commit="7378737a0d18ce6e4779e30605b557bd1ab25e6e",
    )
    committed_snapshot = replace(review_snapshot, mode="gate_committed")

    def capture_committed_view(
        cls,
        repo_root,
        *,
        base_commit,
        package_import_path=None,
        workflow_import_path=None,
        mode="review_dirty",
    ):
        if Path(repo_root).expanduser().resolve() == REPO_ROOT.resolve():
            assert mode == "gate_committed"
            assert base_commit == "7378737a0d18ce6e4779e30605b557bd1ab25e6e"
            assert package_import_path is None
            assert workflow_import_path is None
            return committed_snapshot
        return original_capture(
            cls,
            repo_root,
            base_commit=base_commit,
            package_import_path=package_import_path,
            workflow_import_path=workflow_import_path,
            mode=mode,
        )

    monkeypatch.setattr(
        Stage1SourceSnapshot,
        "capture",
        classmethod(capture_committed_view),
    )


def _reviewed_stage1_gate_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lunar_exploration_ppo.workflows.stage1 import (
        record_stage1_independent_review,
        run_stage1_smoke_workflow,
    )
    from lunar_exploration_ppo.workflows.stage1_gate import Stage1GateContext

    workflow = run_stage1_smoke_workflow(
        config_path=STAGE1_CONFIG,
        run_id="pytest-stage1-gate",
        base_output_root=tmp_path / "runs",
    )
    foundation_gate, environment_manifest = _foundation_gate_fixture(tmp_path)
    review_report = tmp_path / "review-report.md"
    review_package = tmp_path / "review-package.diff"
    _write_stage1_git_native_review_package(review_package)
    from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_prospective_tree

    reviewed_tree = compute_stage1_prospective_tree(
        REPO_ROOT,
        base_commit="7378737a0d18ce6e4779e30605b557bd1ab25e6e",
    )
    _write_stage1_clean_review_report(
        review_report,
        review_package=review_package,
        reviewed_tree=reviewed_tree,
    )
    record_stage1_independent_review(
        stage_root=workflow.stage_root,
        repo_root=REPO_ROOT,
        foundation_gate=foundation_gate,
        foundation_environment_manifest=environment_manifest,
        review_report=review_report,
        review_package=review_package,
    )
    _install_gate_committed_source_snapshot_fixture(monkeypatch)
    context = Stage1GateContext(
        repo_root=REPO_ROOT,
        stage_root=workflow.stage_root,
        foundation_gate=foundation_gate,
        foundation_environment_manifest=environment_manifest,
        run_id="pytest-stage1-gate",
    )
    return workflow.stage_root, context


def test_stage1_gate_binds_required_sources_without_writing_gate_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunar_exploration_ppo.workflows.stage1_gate import (
        STAGE1_AUTHORIZED_NEXT_STAGE,
        Stage1GateBindingVerifier,
    )

    stage, context = _reviewed_stage1_gate_context(tmp_path, monkeypatch)
    bindings = Stage1GateBindingVerifier.compute(context)

    assert STAGE1_AUTHORIZED_NEXT_STAGE == "ppo_highres_frontier_stage2_observation_frontier/v1"
    assert bindings.authorized_next_stage == STAGE1_AUTHORIZED_NEXT_STAGE
    assert len(bindings.goal_hash) == 64
    assert len(bindings.stage_hash) == 64
    assert len(bindings.scenario_data_hash) == 64
    assert len(bindings.foundation_gate_hash) == 64
    assert len(bindings.foundation_environment_hash) == 64
    assert len(bindings.checkpoint_hash) == 64
    assert not (stage / "gate.json").exists()
    assert not (stage / "approval.json").exists()


def test_stage1_gate_fails_closed_on_missing_review_jump_state_and_artifact_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunar_exploration_ppo.workflows.stage1 import run_stage1_smoke_workflow
    from lunar_exploration_ppo.workflows.stage1_gate import (
        Stage1GateBindingVerifier,
        Stage1GateContext,
        Stage1GateError,
    )

    machine = run_stage1_smoke_workflow(
        config_path=STAGE1_CONFIG,
        run_id="pytest-stage1-missing-review",
        base_output_root=tmp_path / "missing-runs",
    )
    foundation_gate, environment_manifest = _foundation_gate_fixture(tmp_path / "missing-foundation")
    missing_context = Stage1GateContext(
        repo_root=REPO_ROOT,
        stage_root=machine.stage_root,
        foundation_gate=foundation_gate,
        foundation_environment_manifest=environment_manifest,
        run_id="pytest-stage1-missing-review",
    )
    with pytest.raises(Stage1GateError, match="review|state"):
        Stage1GateBindingVerifier.compute(missing_context)

    stage, reviewed_context = _reviewed_stage1_gate_context(tmp_path / "reviewed", monkeypatch)
    phase_path = stage / "phase-state.jsonl"
    valid_phases = phase_path.read_text(encoding="utf-8")
    phase_path.write_text(
        json.dumps({"state": "machine_passed", "run_id": reviewed_context.run_id})
        + "\n"
        + json.dumps({"state": "awaiting_human_approval", "run_id": reviewed_context.run_id})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage1GateError, match="phase-state"):
        Stage1GateBindingVerifier.compute(reviewed_context)
    phase_path.write_text(valid_phases, encoding="utf-8")
    with (stage / "config.json").open("ab") as stream:
        stream.write(b" \n")
    with pytest.raises(Stage1GateError, match="manifest|config"):
        Stage1GateBindingVerifier.compute(reviewed_context)


def test_stage1_gate_rejects_dirty_tree_and_has_no_force_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from lunar_exploration_ppo.workflows import stage1_gate as gate_module

    def fake_run_git(cls, repo_root, arguments, label):
        values = {
            "top-level": str(REPO_ROOT.resolve()) + "\n",
            "branch": gate_module.FOUNDATION_BRANCH + "\n",
            "git-dir": gate_module.FOUNDATION_GIT_DIR + "\n",
            "git-common-dir": gate_module.FOUNDATION_GIT_COMMON_DIR + "\n",
            "status": " M dirty.py\n",
            "HEAD tree": "1" * 40 + "\n",
        }
        return values[label]

    monkeypatch.setattr(
        gate_module.Stage1GateBindingVerifier,
        "_run_git",
        classmethod(fake_run_git),
    )
    with pytest.raises(gate_module.Stage1GateError, match="clean"):
        gate_module.Stage1GateBindingVerifier._verify_repository_identity(REPO_ROOT)
    assert "--force" not in inspect.getsource(gate_module)
    assert "--force" not in (REPO_ROOT / "scripts" / "run_ppo_stage1_smoke_env.py").read_text(encoding="utf-8")
