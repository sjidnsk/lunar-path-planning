from __future__ import annotations

import json
from pathlib import Path

import scripts.run_xunce_stage22_0_theta_aware_sensor_action_space_contract as s22


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _stage21_fixture(tmp_path: Path) -> tuple[Path, Path]:
    stage21_18 = tmp_path / "stage21_18"
    stage21_6 = tmp_path / "stage21_6"
    seed_root = stage21_6 / "seed_2101"
    stage21_1 = seed_root / "stage21_1"
    stage21_3 = seed_root / "stage21_3"
    _write_jsonl(
        stage21_18 / "xunce-stage21-18-round-results.jsonl",
        [{"round_index": 1, "stage21_6_roots": [str(stage21_6)], "checkpoint_lineage_passed": True}],
    )
    _write_jsonl(
        stage21_6 / "xunce-stage21-6-seed-results.jsonl",
        [
            {
                "seed": 2101,
                "stage21_1_root": str(stage21_1),
                "stage21_3_root": str(stage21_3),
                "hard_risk_violation_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "canary_traffic_fraction": 0.0,
            }
        ],
    )
    transition = {
        "scenario_id": "s0",
        "step_index": 0,
        "action_index": 0,
        "observation": {
            "candidate_cells": [[2, 0], [0, 2]],
            "candidate_feature_names": ["expected_coverage_rate_delta", "path_cost"],
            "candidate_features": [[0.1, 2.0], [0.2, 2.0]],
            "action_mask": [True, True],
        },
        "info": {
            "current_cell_before": [0, 0],
            "selected_cell": [2, 0],
            "candidate_cells": [[2, 0], [0, 2]],
            "candidate_set_hash": "old-point-only-candidates",
            "covered_cells_hash": "covered-0",
            "path_cost": 2.0,
        },
        "xunce_batch": {"action_mask": {"values": [[True, True]]}},
    }
    _write_jsonl(stage21_1 / "xunce-stage21-1-ppo-trainable-batch.jsonl", [transition])
    _write_json(stage21_1 / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json", {"status": "passed"})
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [transition])
    _write_json(stage21_3 / "xunce-stage21-3-ppo-batch-validation-summary.json", {"status": "passed"})
    return stage21_18, stage21_6


def _stage21_19_fixture(tmp_path: Path) -> tuple[Path, Path]:
    stage21_18, _ = _stage21_fixture(tmp_path)
    stage21_19 = tmp_path / "stage21_19"
    _write_json(
        stage21_19 / "xunce-stage21-19-summary.json",
        {
            "schema_version": "xunce-stage21-19-summary/v1",
            "status": "failed",
            "next_required_change": "repair_stage21_return_advantage_credit_assignment",
            "stage21_18_root": str(stage21_18),
        },
    )
    return stage21_19, stage21_18


def _write_config(path: Path, stage21_18: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": s22.CONFIG_SCHEMA_VERSION,
        "stage_id": "xunce-stage22-0-theta-aware-sensor-action-space-contract",
        "stage21_18_root": str(stage21_18),
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90,
        "sensor_range_cells": 2,
        "material_new_cell_delta_threshold": 1,
        "max_sensor_footprint_audit_rows": 100,
        "stage22_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    _write_json(path, payload)
    return path


def test_stage22_0_detects_point_only_batch_and_routes_to_stage22_1(tmp_path: Path) -> None:
    stage21_18, _ = _stage21_fixture(tmp_path)

    summary = s22.run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=_write_config(tmp_path / "config.json", stage21_18),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "passed"
    assert summary["point_only_action_space_detected"] is True
    assert summary["theta_material_to_coverage"] is True
    assert summary["theta_changes_action_preference"] is True
    assert summary["stage21_ppo_batch_theta_incompatible"] is True
    assert summary["next_required_change"] == s22.ROUTE_STAGE22_1
    assert (tmp_path / "out" / "xunce-stage22-0-sensor-footprint-audit.jsonl").exists()


def test_stage22_0_can_discover_stage21_18_from_stage21_19_root(tmp_path: Path) -> None:
    stage21_19, _ = _stage21_19_fixture(tmp_path)

    summary = s22.run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=_write_config(
            tmp_path / "config.json",
            tmp_path / "missing-direct-stage21-18",
            stage21_19_root=str(stage21_19),
        ),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "passed"
    assert summary["stage21_19_root"] == str(stage21_19)
    assert summary["stage21_19_next_required_change"] == "repair_stage21_return_advantage_credit_assignment"


def test_stage22_0_preserves_original_candidate_index_when_mask_filters_rows(tmp_path: Path) -> None:
    stage21_18, stage21_6 = _stage21_fixture(tmp_path)
    stage21_3 = stage21_6 / "seed_2101" / "stage21_3"
    row = json.loads((stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row["info"]["sampling_mask"] = [False, True]
    row["observation"]["action_mask"] = [False, True]
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [row])

    summary = s22.run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=_write_config(tmp_path / "config.json", stage21_18),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )
    footprint_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "xunce-stage22-0-sensor-footprint-audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["status"] == "passed"
    assert {row["candidate_index"] for row in footprint_rows} == {1}


def test_stage22_0_accepts_future_theta_aware_candidate_cells(tmp_path: Path) -> None:
    stage21_18, stage21_6 = _stage21_fixture(tmp_path)
    stage21_3 = stage21_6 / "seed_2101" / "stage21_3"
    row = json.loads((stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row["observation"]["candidate_cells"] = [[2, 0, 0], [0, 2, 90]]
    row["info"]["candidate_cells"] = [[2, 0], [0, 2]]
    row["info"]["candidate_set_hash"] = "theta-aware-candidates"
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [row])

    summary = s22.run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=_write_config(tmp_path / "config.json", stage21_18),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "passed"
    assert summary["point_only_action_space_detected"] is False
    assert summary["stage21_ppo_batch_theta_incompatible"] is False
    assert summary["next_required_change"] == s22.ROUTE_STAGE22_2


def test_stage22_0_fails_when_any_discovered_stage21_3_batch_is_missing(tmp_path: Path) -> None:
    stage21_18, stage21_6 = _stage21_fixture(tmp_path)
    missing_stage21_3 = tmp_path / "missing_stage21_3"
    _write_jsonl(
        stage21_6 / "xunce-stage21-6-seed-results.jsonl",
        [
            {
                "seed": 2101,
                "stage21_3_root": str(stage21_6 / "seed_2101" / "stage21_3"),
                "stage21_1_root": str(stage21_6 / "seed_2101" / "stage21_1"),
            },
            {
                "seed": 2102,
                "stage21_3_root": str(missing_stage21_3),
                "stage21_1_root": str(stage21_6 / "seed_2101" / "stage21_1"),
            },
        ],
    )

    summary = s22.run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=_write_config(tmp_path / "config.json", stage21_18),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s22.ROUTE_INPUTS
    assert "stage21_3_trainable_batch_missing" in summary["input_reason_codes"]


def test_stage22_0_boundary_flags_hard_fail(tmp_path: Path) -> None:
    stage21_18, _ = _stage21_fixture(tmp_path)

    summary = s22.run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=_write_config(tmp_path / "config.json", stage21_18, publishes_checkpoint=True),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s22.ROUTE_BOUNDARY


def test_stage22_0_missing_inputs_routes_to_required_inputs(tmp_path: Path) -> None:
    summary = s22.run_xunce_stage22_0_theta_aware_sensor_action_space_contract(
        config_path=_write_config(tmp_path / "config.json", tmp_path / "missing"),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s22.ROUTE_INPUTS
