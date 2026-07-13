from __future__ import annotations

import math
from dataclasses import fields

import numpy as np

from lunar_exploration_ppo.env.frontier import FrontierActionSet
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.policy import observation as observation_module
from lunar_exploration_ppo.policy.observation import ObservationBuilder, PolicyObservation
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta


def _state(size: int = 8) -> ObservedMapState:
    geometry = GridGeometry(size, size, 0.5)
    state = ObservedMapState.empty(geometry)
    state.observed_mask[2:6, 2:6] = True
    state.confidence[2:6, 2:6] = 1.0
    state.height[2:6, 2:6] = 12.0
    state.traversability[2:6, 2:6] = 0.8
    state.observed_safe_mask[2:6, 2:6] = True
    state.remaining_step_budget_norm = 0.25
    return state


def _prior(size: int = 2, resolution_m: float = 2.0) -> LowResolutionPrior:
    channels = np.zeros((7, size, size), dtype=np.float32)
    channels[0] = np.arange(size * size, dtype=np.float32).reshape(size, size)
    channels[1] = np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32) if size == 2 else 0.5
    channels[2] = 0.1
    channels[3] = 0.9
    return LowResolutionPrior(channels, resolution_m, "test_prior/v1")


def _actions(top_m: int = 4) -> FrontierActionSet:
    cells = (CellXY(2, 2), CellXY(5, 5))
    features = np.zeros((top_m, 22), dtype=np.float32)
    features[:2, 0] = (0.25, 0.75)
    mask = np.zeros(top_m, dtype=bool)
    mask[:2] = True
    frontier_mask = np.zeros((8, 8), dtype=bool)
    frontier_mask[2, 2] = True
    frontier_mask[5, 5] = True
    return FrontierActionSet(
        cells=cells,
        frontier_features=features,
        candidate_mask=mask,
        frontier_mask=frontier_mask,
        observed_safe_frontier_mask=frontier_mask.copy(),
        reachable_frontier_mask=frontier_mask.copy(),
    )


def test_stage2_schema_names_and_profile_shapes_are_exact() -> None:
    assert observation_module.OBSERVATION_SCHEMA_VERSION == "policy_observation/v1"
    assert observation_module.GLOBAL_PRIOR_CHANNELS == (
        "height_prior", "value_prior", "obstacle_prior", "traversability_prior",
        "current_position_marker_lowres", "heading_sin_marker_lowres", "heading_cos_marker_lowres",
    )
    assert observation_module.COVERAGE_SUMMARY_CHANNELS == (
        "observed_ratio", "unknown_ratio", "observed_free_ratio", "observed_blocked_ratio",
        "frontier_count", "observed_safe_frontier_count", "reachable_frontier_count",
        "mean_uncovered_value_prior",
    )
    assert observation_module.LOCAL_CROP_CHANNELS == (
        "observed_height", "coverage_mask", "obstacle", "traversability",
        "local_frontier_channel", "current_position_marker", "heading_sin_marker", "heading_cos_marker",
    )
    assert observation_module.FRONTIER_FEATURE_FIELDS == (
        "x_norm", "y_norm", "distance_from_robot_norm", "bearing_sin", "bearing_cos",
        "potential_coverage_gain_norm", "visible_unknown_count_norm", "value_gain_norm",
        "frontier_segment_id_norm", "segment_length_norm", "normal_sin", "normal_cos",
        "normal_confidence", "candidate_generation_mode", "recommended_theta_sin",
        "recommended_theta_cos", "traversability", "clearance_norm",
        "reachable_prefilter_cost_norm", "region_coverage_ratio", "region_unknown_ratio",
        "same_connected_component",
    )
    assert observation_module.POSE_FEATURE_FIELDS == (
        "x_norm", "y_norm", "sin(theta)", "cos(theta)", "observed_roi_ratio",
        "remaining_step_budget_norm",
    )
    observation = ObservationBuilder(local_crop_size=4).build(
        _prior(), _state(), PoseXYTheta(CellXY(3, 3), math.pi / 2), _actions()
    )
    assert tuple(field.name for field in fields(PolicyObservation)) == (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
    )
    assert not hasattr(observation, "frontier_cells")
    assert observation.prior_channels.shape == (7, 2, 2)
    assert observation.coverage_summary.shape == (8, 2, 2)
    assert observation.local_crop.shape == (8, 4, 4)
    assert observation.frontier_features.shape == (4, 22)
    assert observation.candidate_valid_mask.shape == (4,)
    assert observation.pose_features.shape == (6,)
    assert tuple(array.dtype for array in observation.array_fields()) == (
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("bool"),
    )


def test_summary_uses_all_tile_cells_and_observed_only_state() -> None:
    state = _state()
    state.obstacle[2, 2] = True
    state.observed_safe_mask[2, 2] = False
    observation = ObservationBuilder(local_crop_size=4).build(
        _prior(), state, PoseXYTheta(CellXY(3, 3), 0.0), _actions()
    )
    summary = observation.coverage_summary
    assert summary[0, 0, 0] == 0.25
    assert summary[1, 0, 0] == 0.75
    assert summary[2, 0, 0] == 3 / 16
    assert summary[3, 0, 0] == 1 / 16
    assert summary[4, 0, 0] == 1 / 16
    assert summary[5, 0, 0] == 1 / 16
    assert summary[6, 0, 0] == 1 / 16
    assert summary[7, 0, 0] == 0.2
    state.height[~state.observed_mask] = 1.0e30
    repeated = ObservationBuilder(local_crop_size=4).build(
        _prior(), state, PoseXYTheta(CellXY(3, 3), 0.0), _actions()
    )
    assert np.array_equal(summary, repeated.coverage_summary)


def test_local_crop_padding_is_semantically_unknown_at_all_edges_and_corners() -> None:
    state = _state()
    builder = ObservationBuilder(local_crop_size=6)
    positions = (
        CellXY(0, 0), CellXY(7, 0), CellXY(0, 7), CellXY(7, 7),
        CellXY(0, 4), CellXY(7, 4), CellXY(4, 0), CellXY(4, 7),
    )
    for cell in positions:
        crop = builder.build(_prior(), state, PoseXYTheta(cell, 0.25), _actions()).local_crop
        assert np.isfinite(crop).all()
        padding = crop[:, 0, 0] if cell.x == 0 or cell.y == 0 else crop[:, -1, -1]
        assert np.array_equal(padding, np.zeros(8, dtype=np.float32))


def test_markers_and_pose_fields_follow_frozen_order() -> None:
    pose = PoseXYTheta(CellXY(3, 3), math.pi / 2)
    observation = ObservationBuilder(local_crop_size=4).build(_prior(), _state(), pose, _actions())
    assert observation.prior_channels[4, 0, 0] == 1.0
    assert observation.prior_channels[5, 0, 0] == 1.0
    assert abs(float(observation.prior_channels[6, 0, 0])) < 1e-6
    assert np.allclose(observation.pose_features, (3 / 7, 3 / 7, 1.0, 0.0, 0.25, 0.25), atol=1e-6)


def test_npz_round_trip_preserves_arrays_dtypes_and_schema(tmp_path) -> None:
    observation = ObservationBuilder(local_crop_size=4).build(
        _prior(), _state(), PoseXYTheta(CellXY(3, 3), 0.0), _actions()
    )
    path = tmp_path / "sample.npz"
    observation.save_npz(path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_json",
            "prior_channels",
            "coverage_summary",
            "local_crop",
            "frontier_features",
            "pose_features",
            "candidate_mask",
        }
    restored = PolicyObservation.load_npz(path)
    assert restored.schema_metadata() == observation.schema_metadata()
    for expected, actual in zip(observation.array_fields(), restored.array_fields(), strict=True):
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape
        assert np.array_equal(expected, actual)


def test_environment_updates_remaining_step_budget_feature() -> None:
    from pathlib import Path

    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv, select_conservative_rule_action

    root = Path(__file__).resolve().parents[2]
    config = load_stage1_config(root / "configs/ppo_highres_frontier_smoke_v1.json")
    env = LunarExplorationEnv(config)
    observation = env.reset()
    assert observation.pose_features[5] == 1.0
    result = env.step(select_conservative_rule_action(observation))
    assert result.observation.pose_features[5] == (config.max_steps - 1) / config.max_steps
