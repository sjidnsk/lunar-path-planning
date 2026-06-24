from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "model-explorer" / "src"):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_platform_contract_hash_and_slope_defaults_are_stable() -> None:
    from scripts.xunce_platform_contract import load_platform_contract, platform_lineage, stable_contract_hash

    contract = load_platform_contract({"platform_contract": "configs/platforms/agilex_scout_mini_piper_v1.json"}, repo_root=REPO_ROOT)
    lineage = platform_lineage(contract)

    assert lineage["platform_contract_id"] == "agilex_scout_mini_piper"
    assert lineage["platform_max_climb_deg"] == 30.0
    assert lineage["max_traversable_slope_deg"] == 30.0
    assert lineage["slope_sensitivity_thresholds_deg"] == [20.0, 30.0]
    assert lineage["platform_contract_hash"] == stable_contract_hash(json.loads(json.dumps(contract, sort_keys=True)))
    assert "downloaded_at" not in json.dumps(contract)


def test_stage23_2b_runner_uses_30_as_default_and_20_as_sensitivity(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment import (
        SUMMARY_FILE,
        run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment,
    )

    default_root = tmp_path / "default30"
    sensitivity_root = tmp_path / "sensitivity20"
    _write_json(
        default_root / "xunce-stage23-2a-summary.json",
        {"status": "passed", "next_required_change": "implement_stage23_2_slope_obstacle_aware_theta_reward_contract", "slope_blocked_cell_count_total": 3},
    )
    _write_json(
        default_root / "xunce-stage23-2a-rerun-stage23-1-summary.json",
        {"status": "passed", "next_required_change": "implement_stage23_2_slope_obstacle_aware_theta_reward_contract", "slope_los_audit_row_count": 5, "slope_material_occlusion_viewpoint_count": 2},
    )
    _write_json(
        sensitivity_root / "xunce-stage23-2a-summary.json",
        {"status": "passed", "next_required_change": "implement_stage23_2_slope_obstacle_aware_theta_reward_contract", "slope_blocked_cell_count_total": 9},
    )
    _write_json(
        sensitivity_root / "xunce-stage23-2a-rerun-stage23-1-summary.json",
        {"status": "passed", "next_required_change": "implement_stage23_2_slope_obstacle_aware_theta_reward_contract", "slope_los_audit_row_count": 7, "slope_material_occlusion_viewpoint_count": 4},
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage23-2b-platform-geometry-sensor-contract-alignment-config/v1",
            "platform_contract": "configs/platforms/agilex_scout_mini_piper_v1.json",
            "run_stage23_2a_smoke": False,
            "run_sensitivity_smoke": False,
            "stage23_2a_default_root": str(default_root),
            "stage23_2a_sensitivity_root": str(sensitivity_root),
            "stage23_2b_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )

    summary = run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage23_2_slope_obstacle_aware_theta_reward_contract"
    assert summary["platform_contract_id"] == "agilex_scout_mini_piper"
    assert summary["platform_max_climb_deg"] == 30.0
    assert summary["max_traversable_slope_deg"] == 30.0
    assert summary["sensitivity_threshold_deg"] == 20.0
    assert summary["default_slope_blocked_cell_count_total"] == 3
    assert summary["sensitivity_slope_blocked_cell_count_total"] == 9
    assert summary["publishes_checkpoint"] is False
    assert (tmp_path / "out" / SUMMARY_FILE).is_file()


def test_stage23_2b_rejects_boundary_flags(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment import (
        run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment,
    )

    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage23-2b-platform-geometry-sensor-contract-alignment-config/v1",
            "platform_contract": "configs/platforms/agilex_scout_mini_piper_v1.json",
            "publishes_checkpoint": True,
            "runs_new_ppo_update": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )

    summary = run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage23_2b_boundary_rejections"
    assert "publishes_checkpoint_must_be_false" in summary["blocking_reason_codes"]
