from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "path-planner" / "src"):
    value = str(_path)
    if value not in sys.path:
        sys.path.insert(0, value)


def test_stage26_0_runner_passes_and_writes_required_artifacts(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract as s26

    config = _write_config(tmp_path)

    summary = s26.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_1_synthetic_terrain_collector_smoke"
    assert summary["synthetic_terrain_hash"]
    assert summary["synthetic_rock_count"] > 0
    assert summary["synthetic_pit_count"] > 0
    assert summary["synthetic_hard_obstacle_cell_count"] > 0
    assert summary["synthetic_los_blocker_cell_count"] > 0
    assert summary["physical_obstacle_cells_written"] is False
    assert summary["effective_blocked_fraction_passed"] is True
    assert summary["effective_connectivity_passed"] is True
    assert summary["effective_start_clearance_passed"] is True
    assert summary["los_material_viewpoint_count"] > 0
    assert summary["hybrid_path_cost_material_candidate_count"] > 0
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False

    for name in (
        "xunce-stage26-0-summary.json",
        "xunce-stage26-0-synthetic-feature-catalog.jsonl",
        "xunce-stage26-0-synthetic-obstacle-source.json",
        "xunce-stage26-0-map-augmentation-audit.json",
        "xunce-stage26-0-los-impact-audit.json",
        "xunce-stage26-0-hybrid-path-impact-audit.json",
        "xunce-stage26-0-next-stage-routing.json",
        "xunce-stage26-0-report.md",
        "xunce-stage26-0-manifest.json",
    ):
        assert (tmp_path / "out" / name).is_file(), name

    source = json.loads((tmp_path / "out" / "xunce-stage26-0-synthetic-obstacle-source.json").read_text(encoding="utf-8"))
    assert source["source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert source["physical_obstacle_cells_written"] is False
    assert "physical_obstacle_cells" not in source


def test_stage26_0_runner_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract as s26

    config = _write_config(tmp_path)

    first = s26.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=config,
        output_root=tmp_path / "out-a",
        repo_root=REPO_ROOT,
    )
    second = s26.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=config,
        output_root=tmp_path / "out-b",
        repo_root=REPO_ROOT,
    )

    assert first["synthetic_terrain_hash"] == second["synthetic_terrain_hash"]
    assert (tmp_path / "out-a" / "xunce-stage26-0-synthetic-feature-catalog.jsonl").read_text(
        encoding="utf-8"
    ) == (tmp_path / "out-b" / "xunce-stage26-0-synthetic-feature-catalog.jsonl").read_text(encoding="utf-8")


def test_stage26_0_routes_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract as s26

    config = _write_config(tmp_path, overrides={"publishes_checkpoint": True})

    summary = s26.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_0_boundary_rejections"
    assert "config_publishes_checkpoint_enabled" in summary["blocking_reason_codes"]


def test_stage26_0_routes_missing_sidecar(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract as s26

    config = _write_config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["source_sidecar_paths"] = [str(tmp_path / "missing.path-planner-sidecar.json")]
    payload["source_roi_expansion_root"] = ""
    config.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    summary = s26.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_0_required_map_inputs"
    assert "source_sidecar_missing" in summary["blocking_reason_codes"]


def test_stage26_0_rejects_unusable_effective_map(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract as s26

    blocked = [[x, 0] for x in range(20)] + [[0, y] for y in range(20)]
    config = _write_config(tmp_path, sidecar_overrides={"slope_blocked_cells": blocked})

    summary = s26.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_0_synthetic_connectivity_constraints"
    assert summary["effective_connectivity_passed"] is False
    assert "effective_connectivity_failed" in summary["blocking_reason_codes"]


def test_stage26_0_effective_gate_includes_passable_mask_false(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract as s26

    passable_mask = [[True for _ in range(20)] for _ in range(20)]
    passable_mask[0][0] = False
    config = _write_config(tmp_path, sidecar_overrides={"passable_mask": passable_mask})

    summary = s26.run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_0_synthetic_connectivity_constraints"
    assert "effective_connectivity_failed" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, *, overrides: dict | None = None, sidecar_overrides: dict | None = None) -> Path:
    sidecar = tmp_path / "fixture.path-planner-sidecar.json"
    sidecar_payload = _sidecar_payload()
    if sidecar_overrides:
        sidecar_payload.update(sidecar_overrides)
    sidecar.write_text(json.dumps(sidecar_payload, indent=2, sort_keys=True), encoding="utf-8")
    payload = {
        "schema_version": "xunce-stage26-0-synthetic-rock-pit-terrain-augmentation-contract-config/v1",
        "source_sidecar_paths": [str(sidecar)],
        "source_roi_expansion_root": "",
        "max_sidecar_count": 1,
        "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
        "synthetic_terrain_seed": 260001,
        "rock_generation_enabled": True,
        "rock_count_range": [16, 16],
        "rock_radius_m_range": [1.0, 2.0],
        "rock_height_m_range": [1.0, 1.5],
        "rock_los_blocker_height_threshold_m": 0.5,
        "pit_generation_enabled": True,
        "pit_count_range": [4, 4],
        "pit_radius_m_range": [2.0, 3.0],
        "pit_depth_m_range": [1.0, 1.5],
        "pit_hard_depth_threshold_m": 0.8,
        "min_start_clearance_m": 2.0,
        "max_blocked_fraction": 0.35,
        "connectivity_check_enabled": True,
        "sensor_range_cells": 5,
        "sensor_fov_deg": 90.0,
        "los_sample_limit": 32,
        "hybrid_sample_limit": 4,
        "hybrid_astar_max_iterations": 5000,
        "max_traversable_slope_deg": 30.0,
        "platform_contract_hash": "platform-hash",
        "stage26_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    if overrides:
        payload.update(overrides)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return config


def _sidecar_payload() -> dict:
    width = 20
    height = 20
    return {
        "schema_version": "xunce-test-sidecar/v1",
        "cost": [[1.0 for _ in range(width)] for _ in range(height)],
        "passable_mask": [[True for _ in range(width)] for _ in range(height)],
        "resolution_m": 1.0,
        "start_cell": [0, 0],
        "slope_blocked_cells": [],
        "blocked_cells": [],
        "platform_contract_hash": "platform-hash",
        "platform_contract_id": "agilex_scout_mini_piper",
        "max_traversable_slope_deg": 30.0,
        "metadata": {
            "start_cell": [0, 0],
            "map_source": {"resolution_m": 1.0},
        },
    }
