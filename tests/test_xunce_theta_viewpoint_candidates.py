from __future__ import annotations

from scripts.xunce_theta_viewpoint_candidates import (
    expand_theta_aware_candidates,
    row_has_theta_viewpoint_contract,
    theta_metadata,
    theta_viewpoint_candidate_set_hash,
)


def test_two_point_candidates_expand_to_sixteen_viewpoints() -> None:
    expansion = expand_theta_aware_candidates(
        [
            {"cell": [2, 0], "action_index": 0, "path_cost": 2.0, "risk": 1.0},
            {"cell": [0, 2], "action_index": 1, "path_cost": 3.0, "risk": 1.5},
        ],
        current_cell=[0, 0],
        covered_cells=set(),
        config={
            "theta_aware_candidate_viewpoints_enabled": True,
            "theta_bin_count": 8,
            "theta_step_deg": 45,
            "sensor_range_cells": 2,
            "sensor_fov_deg": 90,
            "coverage_denominator_cells": 100,
        },
        base_candidate_set_hash="base-hash",
    )

    candidates = expansion["candidates"]
    assert len(candidates) == 16
    assert {row["candidate_theta_deg"] for row in candidates[:8]} == {0, 45, 90, 135, 180, 225, 270, 315}
    assert candidates[0]["cell"] == [2, 0]
    assert candidates[0]["candidate_viewpoint"] == [2, 0, 0]
    assert candidates[8]["base_candidate_index"] == 1
    assert all(row["theta_feature_available"] is True for row in candidates)
    metadata = theta_metadata(candidates)
    assert metadata["theta_new_visible_cell_counts"][0] == candidates[0]["theta_new_visible_cell_count"]
    assert metadata["theta_coverage_hashes"][0] == candidates[0]["theta_coverage_hash"]
    assert metadata["theta_coverage_gain_per_path_costs"][0] == candidates[0]["theta_coverage_gain_per_path_cost"]
    assert row_has_theta_viewpoint_contract({"info": {"candidate_cells": [[2, 0, 0]]}}) is True


def test_candidate_set_hash_includes_theta_and_sensor_model() -> None:
    first = [
        {
            "candidate_viewpoint": [2, 0, 0],
            "sensor_model_id": "theta-fov-90-range-radius/v1",
            "sensor_range_cells": 2,
            "sensor_fov_deg": 90,
        }
    ]
    second = [
        {
            "candidate_viewpoint": [2, 0, 45],
            "sensor_model_id": "theta-fov-90-range-radius/v1",
            "sensor_range_cells": 2,
            "sensor_fov_deg": 90,
        }
    ]

    assert theta_viewpoint_candidate_set_hash(first) != theta_viewpoint_candidate_set_hash(second)
