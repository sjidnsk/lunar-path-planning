from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "path-planner" / "src"):
    value = str(_path)
    if value not in sys.path:
        sys.path.insert(0, value)


def test_synthetic_terrain_generation_is_seed_deterministic_and_semantic() -> None:
    from scripts.xunce_synthetic_terrain_features import generate_synthetic_terrain_augmentation

    config = _config(seed=260001)
    first = generate_synthetic_terrain_augmentation(
        width=24,
        height=24,
        resolution_m=2.0,
        start_cell=[0, 0],
        existing_hard_obstacle_cells=[],
        config=config,
    )
    second = generate_synthetic_terrain_augmentation(
        width=24,
        height=24,
        resolution_m=2.0,
        start_cell=[0, 0],
        existing_hard_obstacle_cells=[],
        config=config,
    )

    assert first["source"]["synthetic_terrain_hash"] == second["source"]["synthetic_terrain_hash"]
    assert first["catalog"] == second["catalog"]
    assert first["source"]["source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert first["source"]["physical_obstacle_cells_written"] is False
    assert "physical_obstacle_cells" not in first["source"]
    assert first["audit"]["blocked_fraction_passed"] is True
    assert first["audit"]["start_clearance_passed"] is True
    assert first["audit"]["connectivity_passed"] is True
    assert first["source"]["synthetic_hard_obstacle_cells"]
    assert first["source"]["synthetic_los_blocker_cells"]


def test_synthetic_terrain_generation_changes_with_seed() -> None:
    from scripts.xunce_synthetic_terrain_features import generate_synthetic_terrain_augmentation

    first = generate_synthetic_terrain_augmentation(
        width=24,
        height=24,
        resolution_m=2.0,
        start_cell=[0, 0],
        existing_hard_obstacle_cells=[],
        config=_config(seed=260001),
    )
    second = generate_synthetic_terrain_augmentation(
        width=24,
        height=24,
        resolution_m=2.0,
        start_cell=[0, 0],
        existing_hard_obstacle_cells=[],
        config=_config(seed=260002),
    )

    assert first["source"]["synthetic_terrain_hash"] != second["source"]["synthetic_terrain_hash"]


def test_augmented_sidecar_keeps_synthetic_proxy_fields_separate() -> None:
    from scripts.xunce_synthetic_terrain_features import (
        augmented_sidecar_with_synthetic_source,
        generate_synthetic_terrain_augmentation,
    )

    sidecar = {"cost": [[1.0] * 8 for _ in range(8)], "passable_mask": [[True] * 8 for _ in range(8)]}
    augmentation = generate_synthetic_terrain_augmentation(
        width=8,
        height=8,
        resolution_m=1.0,
        start_cell=[0, 0],
        existing_hard_obstacle_cells=[],
        config=_config(seed=260010, rock_count_range=[3, 3], pit_count_range=[1, 1], max_blocked_fraction=0.5),
    )

    augmented = augmented_sidecar_with_synthetic_source(sidecar, augmentation["source"])

    assert augmented["synthetic_terrain_augmentation_enabled"] is True
    assert augmented["synthetic_source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert augmented["synthetic_hard_obstacle_cells"]
    assert "physical_obstacle_cells" not in augmented
    assert augmented["physical_obstacle_cells_written_by_synthetic"] is False


def test_hybrid_grid_treats_synthetic_hard_obstacle_as_unpassable() -> None:
    from path_planner.core import Cell
    from scripts.xunce_hybrid_astar_candidate_path_cost import build_cost_grid_from_config

    grid = build_cost_grid_from_config(
        {
            "width": 5,
            "height": 5,
            "resolution_m": 1.0,
            "synthetic_hard_obstacle_cells": [[2, 2]],
        }
    )

    assert grid.is_passable(Cell(2, 2)) is False
    assert grid.is_passable(Cell(1, 1)) is True


def _config(seed: int, **overrides):
    payload = {
        "synthetic_terrain_seed": seed,
        "rock_generation_enabled": True,
        "rock_count_range": [8, 8],
        "rock_radius_m_range": [0.75, 2.0],
        "rock_height_m_range": [1.0, 1.5],
        "rock_los_blocker_height_threshold_m": 0.5,
        "pit_generation_enabled": True,
        "pit_count_range": [3, 3],
        "pit_radius_m_range": [1.5, 3.0],
        "pit_depth_m_range": [1.0, 1.5],
        "pit_hard_depth_threshold_m": 0.8,
        "min_start_clearance_m": 2.0,
        "max_blocked_fraction": 0.4,
        "connectivity_check_enabled": True,
    }
    payload.update(overrides)
    return payload
