import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_1_runs_collector_reward_batch_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as s26

    _patch_stage21_runs(monkeypatch, s26)
    stage26_0 = _write_stage26_0_root(tmp_path)
    config = _write_config(tmp_path, stage26_0)

    summary = s26.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    audit = json.loads((tmp_path / "out" / "xunce-stage26-1-synthetic-transition-contract-audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_2_synthetic_terrain_ppo_update_smoke"
    assert summary["synthetic_transition_contract_missing_count"] == 0
    assert summary["synthetic_reward_provenance_missing_count"] == 0
    assert summary["synthetic_batch_contract_missing_count"] == 0
    assert summary["synthetic_physical_obstacle_pollution_count"] == 0
    assert audit["stage21_3_rejects_missing_synthetic_contract"] is True
    stage21_1_config = json.loads((tmp_path / "out" / "xunce-stage26-1-stage21-1-config.json").read_text(encoding="utf-8"))
    stage21_2_config = json.loads((tmp_path / "out" / "xunce-stage26-1-stage21-2-config.json").read_text(encoding="utf-8"))
    stage21_3_config = json.loads((tmp_path / "out" / "xunce-stage26-1-stage21-3-config.json").read_text(encoding="utf-8"))
    assert stage21_1_config["synthetic_terrain_contract_enabled"] is True
    assert stage21_2_config["require_synthetic_terrain_contract"] is True
    assert stage21_3_config["require_synthetic_terrain_contract"] is True
    assert summary["runs_new_ppo_update"] is False


def test_stage26_1_rejects_unready_stage26_0_without_running_substages(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as s26

    _patch_stage21_runs(monkeypatch, s26)
    stage26_0 = _write_stage26_0_root(tmp_path, next_required_change="repair_stage26_0_synthetic_source_semantics")
    config = _write_config(tmp_path, stage26_0)

    summary = s26.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_1_required_inputs"
    assert "stage26_0_route_not_stage26_1" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_1").exists()


def test_stage26_1_routes_reward_repair_when_synthetic_reward_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as s26

    _patch_stage21_runs(monkeypatch, s26, reward_mode="missing_synthetic")
    stage26_0 = _write_stage26_0_root(tmp_path)
    config = _write_config(tmp_path, stage26_0)

    summary = s26.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_1_synthetic_reward_provenance"
    assert "synthetic_reward_provenance_missing" in summary["blocking_reason_codes"]


def test_stage21_3_synthetic_gate_rejects_missing_contract() -> None:
    import scripts.run_xunce_stage21_3_ppo_batch_validation as s21_3

    assert s21_3._synthetic_terrain_contract_missing_count([{"reward": 0.0}]) == 1
    assert s21_3._synthetic_terrain_contract_missing_count([_batch_row(mode="valid")]) == 0


def _patch_stage21_runs(monkeypatch, s26, *, reward_mode: str = "valid", batch_mode: str = "valid") -> None:
    def fake_stage21_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        rows = [_transition()]
        _write_jsonl(output_root / s26.stage21_1.TRAINABLE_BATCH_FILE, rows)
        _write_jsonl(output_root / s26.stage21_1.TRANSITIONS_FILE, rows)
        _write_jsonl(output_root / s26.stage21_1.REJECTION_FILE, [])
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        summary = {"status": "passed", "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract", "trainable_transition_count": 1}
        (output_root / s26.stage21_1.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        manifest = {"model_audit": {"source_roi_expansion_root": cfg.get("source_roi_expansion_root")}}
        (output_root / s26.stage21_1.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
        return summary

    def fake_stage21_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        rows = [_reward(mode=reward_mode)]
        _write_jsonl(output_root / s26.stage21_2.EVALUATION_FILE, rows)
        summary = {
            "status": "passed",
            "next_required_change": "implement_stage21_3_ppo_batch_validation",
            "synthetic_terrain_contract_missing_count": 0 if reward_mode == "valid" else 1,
        }
        (output_root / s26.stage21_2.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fake_stage21_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_root / s26.stage21_3.BATCH_FILE, [_batch_row(mode=batch_mode)])
        summary = {"status": "passed", "next_required_change": "implement_stage21_4_tiny_ppo_update_smoke", "trainable_transition_count": 1}
        (output_root / s26.stage21_3.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage21_1, "run_xunce_stage21_1_on_policy_ppo_rollout_collector", fake_stage21_1)
    monkeypatch.setattr(s26.stage21_2, "run_xunce_stage21_2_coverage_first_ppo_reward_contract", fake_stage21_2)
    monkeypatch.setattr(s26.stage21_3, "run_xunce_stage21_3_ppo_batch_validation", fake_stage21_3)


def _transition() -> dict:
    return {
        "transition_id": "stage26:step-0:sample-1",
        "scenario_id": "stage26",
        "step_index": 0,
        "action_index": 1,
        "trainable": True,
        "info": _info(),
    }


def _reward(*, mode: str) -> dict:
    row = _batch_row(mode="valid")
    if mode == "missing_synthetic":
        row["synthetic_terrain_reward_provenance"] = False
        row.pop("synthetic_terrain_hash", None)
    return row


def _batch_row(*, mode: str) -> dict:
    row = {
        "transition_id": "stage26:step-0:sample-1",
        "action_index": 1,
        "theta_aware_reward_contract": True,
        "slope_obstacle_aware_theta_reward_contract": True,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "candidate_viewpoint": [1, 2, 45],
        "candidate_theta_deg": 45,
        "obstacle_aware_new_visible_cell_count": 4,
        "obstacle_aware_theta_coverage_hash": "obstacle-hash-45",
        "obstacle_aware_theta_coverage_gain_per_path_cost": 2.0,
        "obstacle_aware_theta_coverage_denominator_cells": 100.0,
        "slope_obstacle_source_hash": "synthetic-obstacle-source-hash",
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "point_only_reward_fallback_used": False,
        "unobstructed_theta_reward_fallback_used": False,
        "hybrid_astar_path_cost_reward_contract": True,
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "hybrid_astar_path_cost": 5.5,
        "hybrid_astar_pose_path_hash": "pose-path-hash",
        "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
        "legacy_grid_astar_path_cost": 4.0,
        "hybrid_vs_grid_path_cost_delta": 1.5,
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "point_grid_path_cost_fallback_used": False,
        "reward_metrics": {"path_cost_m": 5.5},
        "synthetic_terrain_reward_provenance": True,
        "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
        "synthetic_terrain_hash": "stage26-aggregate-hash",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "synthetic_hard_obstacle_cells_used": True,
        "synthetic_los_blocker_cells_used": True,
        "synthetic_high_risk_cells_available": True,
        "physical_obstacle_cells_written": False,
        "effective_hard_obstacle_source": ["slope_blocked_cells", "synthetic_hard_obstacle_cells"],
        "effective_los_blocker_source": ["slope_blocked_cells", "synthetic_los_blocker_cells"],
        "info": _info(),
    }
    if mode == "missing_synthetic":
        row["synthetic_terrain_reward_provenance"] = False
        row["synthetic_terrain_hash"] = None
    return row


def _info() -> dict:
    return {
        "candidate_viewpoints": [[1, 2, 0], [1, 2, 45]],
        "candidate_theta_deg": [0, 45],
        "selected_viewpoint": [1, 2, 45],
        "selected_theta_deg": 45,
        "action_mask": [True, True],
        "sampling_mask": [True, True],
        "hard_risk_clean_mask": [True, True],
        "obstacle_aware_new_visible_cell_counts": [2, 4],
        "obstacle_aware_theta_coverage_hashes": ["obstacle-hash-0", "obstacle-hash-45"],
        "obstacle_aware_theta_coverage_gain_per_path_costs": [1.0, 2.0],
        "slope_obstacle_source_hash": "synthetic-obstacle-source-hash",
        "slope_obstacle_source_hashes": ["synthetic-obstacle-source-hash", "synthetic-obstacle-source-hash"],
        "platform_contract_hash": "platform-hash",
        "platform_contract_hashes": ["platform-hash", "platform-hash"],
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "strict_obstacle_aware_new_visible_cell_count": True,
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "path_cost_sources": ["hybrid_astar_pose_path/v1", "hybrid_astar_pose_path/v1"],
        "hybrid_astar_path_costs": [4.5, 5.5],
        "hybrid_astar_pose_path_hashes": ["pose-path-hash-0", "pose-path-hash"],
        "hybrid_astar_trajectory_kinds": ["hybrid_astar_pose_path", "hybrid_astar_pose_path"],
        "legacy_grid_astar_path_costs": [3.0, 4.0],
        "hybrid_vs_grid_path_cost_deltas": [1.5, 1.5],
        "default_astar_replaced": False,
        "default_astar_replaced_flags": [False, False],
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_ackermann_feasible_claimed_flags": [False, False],
        "path_cost": 4.0,
        "hard_risk_violation": False,
        "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
        "synthetic_terrain_hash": "stage26-aggregate-hash",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "synthetic_hard_obstacle_cells_used": True,
        "synthetic_los_blocker_cells_used": True,
        "synthetic_high_risk_cells_available": True,
        "physical_obstacle_cells_written": False,
        "effective_hard_obstacle_source": ["slope_blocked_cells", "synthetic_hard_obstacle_cells"],
        "effective_los_blocker_source": ["slope_blocked_cells", "synthetic_los_blocker_cells"],
    }


def _write_stage26_0_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage26_1_synthetic_terrain_collector_smoke",
) -> Path:
    root = tmp_path / "stage26_0"
    sidecars = root / "augmented_sidecars"
    sidecars.mkdir(parents=True)
    sidecar = sidecars / "stage26_0_000.synthetic-sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "path-planner-sidecar/v1",
                "cost": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                "passable_mask": [[True, True, True], [True, True, True], [True, True, True]],
                "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
                "synthetic_terrain_hash": "stage26-sidecar-hash",
                "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
                "synthetic_hard_obstacle_cells": [[2, 2]],
                "synthetic_los_blocker_cells": [[1, 1]],
                "synthetic_high_risk_cells": [[1, 2]],
                "physical_obstacle_cells_written_by_synthetic": False,
                "max_traversable_slope_deg": 30.0,
            }
        ),
        encoding="utf-8",
    )
    sidecar.with_suffix(".contract.json").write_text(
        json.dumps(
            {
                "schema_version": "model-explorer-contract/v1",
                "goals": [{"cell": [1, 1], "reachable": True}],
                "start": {"cell": [0, 0]},
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "xunce-stage26-0-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
        "synthetic_terrain_hash": "stage26-aggregate-hash",
        "source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "max_traversable_slope_deg": 30.0,
        "augmented_sidecar_paths": [str(sidecar)],
    }
    (root / "xunce-stage26-0-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "xunce-stage26-0-map-augmentation-audit.json").write_text(
        json.dumps({"sidecar_audits": [{"source_sidecar": str(sidecar)}]}),
        encoding="utf-8",
    )
    return root


def _write_config(tmp_path: Path, stage26_0_root: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-1-synthetic-terrain-collector-smoke-config/v1",
        "stage26_0_root": str(stage26_0_root),
        "stage21_1_base_config": "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json",
        "stage21_2_base_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "stage21_3_base_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_trainable_transition_count": 1,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 3,
        "theta_coverage_denominator_cells": 100.0,
        "max_traversable_slope_deg": 30.0,
        "stage26_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_1_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
