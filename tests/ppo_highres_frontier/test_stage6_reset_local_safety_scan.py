"""Stage 6 reset 的 0.75m/360° 局部安全扫描合同。"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env.env import LunarExplorationEnv
from lunar_exploration_ppo.env.coverage import CoverageMasks
from lunar_exploration_ppo.env.frontier import FrontierActionSet
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import TruthMap
from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry


ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = ROOT / "configs/ppo_highres_frontier_smoke_v1.json"


def test_stage1_config_freezes_local_scan_and_planning_buffer_without_physical_drift() -> None:
    config = load_stage1_config(STAGE1_CONFIG)

    assert config.planning_unknown_buffer_m == 0.75
    assert config.reset_local_safety_scan_range_m == 0.75
    assert config.reset_local_safety_scan_fov_deg == 360.0
    assert config.reset_local_safety_scan_ray_angle_step_deg == 1.0
    assert config.sensor_range_m == 20.0
    assert config.sensor_fov_deg == 90.0
    assert config.sensor_ray_angle_step_deg == 1.0
    assert config.min_clearance_m == 0.5215874761


def test_reset_runs_local_then_exploration_scan_with_independent_diagnostics() -> None:
    config = load_stage1_config(STAGE1_CONFIG)
    env = LunarExplorationEnv(config)

    observation = env.reset()
    diagnostics = env.last_reset_diagnostics

    assert diagnostics is not None
    assert diagnostics.local_safety_sensor.sample_sources == (
        "reset_local_safety",
    )
    assert diagnostics.local_safety_sensor.sample_count == 1
    assert diagnostics.local_safety_sensor.ray_count == 361
    assert diagnostics.exploration_sensor.sample_sources == ("reset",)
    assert diagnostics.exploration_sensor.sample_count == 1
    assert diagnostics.exploration_sensor.ray_count == 91
    assert diagnostics.scan_order == (
        "reset_local_safety",
        "reset_exploration",
    )
    assert diagnostics.sensor == diagnostics.exploration_sensor
    assert env.step_count == 0
    assert env.consecutive_no_gain_steps == 0
    assert diagnostics.reward == 0.0
    assert diagnostics.trainable is False
    assert observation.frontier_features.shape[1] == 22


def test_local_scan_uses_los_and_does_not_reveal_outside_exact_range() -> None:
    geometry = GridGeometry(9, 9, 0.25)
    obstacle = np.zeros(geometry.shape, dtype=bool)
    obstacle[4, 6] = True
    truth = TruthMap(
        geometry=geometry,
        height=np.zeros(geometry.shape, dtype=np.float64),
        hard_obstacle=obstacle,
        slope_deg=np.zeros(geometry.shape, dtype=np.float64),
        traversability=np.ones(geometry.shape, dtype=np.float64),
        provenance={},
    )
    state = ObservedMapState.empty(geometry)
    updater = SensorUpdater(
        range_m=0.75,
        fov_deg=360.0,
        ray_angle_step_deg=1.0,
    )
    start = CellXY(4, 4)

    delta = updater.reveal(
        truth,
        state,
        (
            SensorPose(
                geometry.cell_to_world_center(start),
                math.pi / 3.0,
                "reset_local_safety",
            ),
        ),
    )

    assert state.observed_mask[4, 6]  # blocker is visible
    assert not state.observed_mask[4, 7]  # LOS behind blocker is hidden
    assert not state.observed_mask[4, 8]  # outside 0.75m center range
    assert delta.diagnostics.sample_sources == ("reset_local_safety",)


class _NoCandidateFrontier:
    def __init__(self, top_m: int) -> None:
        self.top_m = top_m

    def extract(self, observed_state, prior, pose) -> FrontierActionSet:
        del observed_state, prior, pose
        return FrontierActionSet(
            cells=(),
            frontier_features=np.zeros((self.top_m, 22), dtype=np.float32),
            candidate_mask=np.zeros((self.top_m,), dtype=bool),
        )


def test_no_candidate_after_dual_scan_remains_terminal_without_fallback_action() -> None:
    config = load_stage1_config(STAGE1_CONFIG)
    env = LunarExplorationEnv(
        config,
        frontier_generator=_NoCandidateFrontier(config.frontier_top_m),
    )

    observation = env.reset()

    assert env.is_done is True
    assert env.needs_policy is False
    assert env.terminal_reason == "no_candidate_done"
    assert not np.any(observation.candidate_mask)
    assert not hasattr(env, "select_fallback_action")


def test_frozen_first_sixteen_validation_resets_are_planning_safe_with_egress() -> None:
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        load_stage6_config,
    )
    from lunar_exploration_ppo.env.standard_training import (
        StandardEnvSettings,
        StandardScenarioFactory,
        build_standard_catalog,
    )

    config_path = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
    config = load_stage6_config(config_path)
    catalog = build_standard_catalog(verify_hashes=False)
    factory = StandardScenarioFactory(catalog)
    records = tuple(
        record for record in catalog.records if record.split == "validation"
    )[:16]
    assert len(records) == 16

    class Source:
        def __init__(self, key, bundle) -> None:
            self.key = key
            self.bundle = bundle

        def load(self, key):
            assert key == self.key
            return self.bundle

    for record in records:
        bundle = factory.build(record)
        coverable = np.ones(bundle.truth.geometry.shape, dtype=bool)
        coverable_sha = hashlib.sha256(
            np.ascontiguousarray(coverable).tobytes()
        ).hexdigest()
        masks = CoverageMasks(
            safe_free_mask=coverable.copy(),
            reachable_safe_mask=coverable.copy(),
            coverable_mask=coverable,
            metadata={
                "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
                "sha256": coverable_sha,
                "exact": True,
                "precompute_scope": "scenario_reset/v1",
                "coverable_cell_count": int(np.count_nonzero(coverable)),
            },
        )
        env = LunarExplorationEnv(
            StandardEnvSettings(
                scenario_key=record.scenario_id,
                safety_contract=SafetyContract.from_stage6_config(config),
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            ),
            scenario_source=Source(record.scenario_id, bundle),
            frontier_generator=_NoCandidateFrontier(1024),
            precomputed_coverage_masks=masks,
        )

        env.reset()
        start = bundle.start_pose.cell
        planning = env.observed_state.planning_safe_mask
        assert planning[start.y, start.x], record.scenario_id
        egress = tuple(
            CellXY(start.x + dx, start.y + dy)
            for dx, dy in (
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            )
            if bundle.truth.geometry.in_bounds(
                CellXY(start.x + dx, start.y + dy)
            )
        )
        assert any(planning[cell.y, cell.x] for cell in egress), record.scenario_id


def _build_standard_env_for_scenario(scenario_id: str) -> LunarExplorationEnv:
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        load_stage6_config,
    )
    from lunar_exploration_ppo.env.standard_training import (
        StandardEnvSettings,
        StandardScenarioFactory,
        build_standard_catalog,
    )

    config_path = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
    config = load_stage6_config(config_path)
    catalog = build_standard_catalog(verify_hashes=False)
    record = next(
        record for record in catalog.records if record.scenario_id == scenario_id
    )
    bundle = StandardScenarioFactory(catalog).build(record)
    coverable = np.ones(bundle.truth.geometry.shape, dtype=bool)
    coverable_sha = hashlib.sha256(
        np.ascontiguousarray(coverable).tobytes()
    ).hexdigest()
    masks = CoverageMasks(
        safe_free_mask=coverable.copy(),
        reachable_safe_mask=coverable.copy(),
        coverable_mask=coverable,
        metadata={
            "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
            "sha256": coverable_sha,
            "exact": True,
            "precompute_scope": "scenario_reset/v1",
            "coverable_cell_count": int(np.count_nonzero(coverable)),
        },
    )

    class Source:
        def load(self, key):
            assert key == scenario_id
            return bundle

    return LunarExplorationEnv(
        StandardEnvSettings(
            scenario_key=scenario_id,
            safety_contract=SafetyContract.from_stage6_config(config),
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        ),
        scenario_source=Source(),
        precomputed_coverage_masks=masks,
    )


@pytest.mark.parametrize(
    "scenario_id",
    ("train/scenario-0398", "train/scenario-0286"),
)
def test_isolated_planning_start_reset_emits_safe_start_fallback(
    scenario_id: str,
) -> None:
    env = _build_standard_env_for_scenario(scenario_id)

    observation = env.reset()
    diagnostics = env.last_reset_diagnostics

    assert diagnostics is not None
    assert diagnostics.scan_order == (
        "reset_local_safety",
        "reset_exploration",
    )
    assert diagnostics.local_safety_sensor.sample_sources == (
        "reset_local_safety",
    )
    assert diagnostics.local_safety_sensor.ray_count == 361
    assert diagnostics.exploration_sensor.sample_sources == ("reset",)
    assert diagnostics.exploration_sensor.ray_count == 91
    assert diagnostics.frontier.component_size == 0
    assert diagnostics.frontier.oracle_opportunity_count == 0
    assert env.is_done is False
    assert env.needs_policy is True
    assert env.terminal_reason == "none"
    assert np.flatnonzero(observation.candidate_mask).tolist() == [0]
    assert env.current_action_set.diagnostics[
        "start_fallback_activated"
    ] is True
    assert env.current_action_set.cells == (env.pose.cell,)
    assert not hasattr(env, "select_fallback_action")


@pytest.mark.parametrize(
    "scenario_id",
    ("train/scenario-0234", "train/scenario-0345"),
)
def test_collector_retry_known_next_scenario_reset_is_trainable(
    scenario_id: str,
) -> None:
    env = _build_standard_env_for_scenario(scenario_id)

    observation = env.reset()
    diagnostics = env.last_reset_diagnostics

    assert diagnostics is not None
    assert diagnostics.scan_order == (
        "reset_local_safety",
        "reset_exploration",
    )
    assert diagnostics.frontier.candidate_count > 0
    assert env.current_action_set.candidate_count > 0
    assert np.any(observation.candidate_mask)
    assert env.is_done is False
    assert env.needs_policy is True
    assert env.terminal_reason == "none"
