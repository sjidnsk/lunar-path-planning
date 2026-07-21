from __future__ import annotations

import numpy as np


def test_v2_offline_adapter_does_not_reveal_or_change_the_v1_default(monkeypatch) -> None:
    from lunar_exploration_ppo.env.sensor_model import SensorUpdater
    from lunar_exploration_ppo.integrations.path_planner_adapter import PathPlannerAdapter
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry
    from path_planner.v2.adapters import PpoTargetV2, build_ppo_request_v2
    from path_planner.v2.contracts import AcceleratorPolicyV2, ObjectiveProfileV2, PoseStateV2, ResourceBudgetV2
    from path_planner.v2.observation import ObservedTerrainInputV2
    from path_planner.v2.terrain import FineGridGeometryV2, TerrainProvenanceV2, TerrainSnapshotV2

    def forbidden_reveal(*_args, **_kwargs):
        raise AssertionError("offline v2 adapter must not call SensorUpdater.reveal")

    monkeypatch.setattr(SensorUpdater, "reveal", forbidden_reveal)

    legacy_geometry = GridGeometry(3, 3, 0.5)
    legacy = PathPlannerAdapter(legacy_geometry).validate(
        np.ones(legacy_geometry.shape, dtype=bool),
        CellXY(0, 0),
        CellXY(2, 2),
        0.5,
    )
    assert legacy.valid
    assert legacy.path_cells == (CellXY(0, 0), CellXY(1, 1), CellXY(2, 2))

    geometry = FineGridGeometryV2(3, 3)
    observed = np.ones(geometry.shape, dtype=bool)
    offline_input = ObservedTerrainInputV2(
        request_id="root-offline-v2",
        start_state=PoseStateV2(0.25, 0.25, 0.0),
        terrain_snapshot=TerrainSnapshotV2(
            geometry=geometry,
            elevation_m=np.zeros(geometry.shape, dtype=np.float64),
            slope_deg=np.zeros(geometry.shape, dtype=np.float64),
            traversable_mask=observed,
            hard_obstacle_mask=np.zeros(geometry.shape, dtype=bool),
            observed_mask=observed,
            confidence=np.ones(geometry.shape, dtype=np.float64),
            provenance=TerrainProvenanceV2(
                source_kind="root-observed-fixture/v1",
                source_id="root-observed-fixture",
                source_hash="root-observed-fixture-hash",
                physical_obstacle_cells_written=False,
            ),
        ),
    )
    request = build_ppo_request_v2(
        PpoTargetV2(1.25, 1.25, 2.5),
        offline_input,
        "wheel-safe/v1",
        ObjectiveProfileV2(),
        ResourceBudgetV2(),
        1.0,
        AcceleratorPolicyV2.OPTIONAL,
        7,
    )

    assert request.goal_state == PoseStateV2(1.25, 1.25, 2.5)
