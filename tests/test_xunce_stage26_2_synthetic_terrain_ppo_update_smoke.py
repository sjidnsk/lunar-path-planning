import json
import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_2_runs_synthetic_terrain_ppo_update_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path)
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage21_4_config = json.loads((tmp_path / "out" / "xunce-stage26-2-stage21-4-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke"
    assert summary["batch_row_count"] == 1
    assert summary["synthetic_terrain_contract_missing_count"] == 0
    assert summary["synthetic_terrain_reward_provenance_missing_count"] == 0
    assert summary["synthetic_info_provenance_missing_count"] == 0
    assert summary["synthetic_physical_obstacle_pollution_count"] == 0
    assert summary["effective_hard_obstacle_source_missing_synthetic_count"] == 0
    assert summary["effective_los_blocker_source_missing_synthetic_count"] == 0
    assert summary["coverage_source_mismatch_count"] == 0
    assert summary["path_cost_source_mismatch_count"] == 0
    assert summary["hybrid_path_batch_contract_missing_count"] == 0
    assert summary["action_viewpoint_path_cost_binding_mismatch_count"] == 0
    assert summary["point_grid_path_cost_fallback_used_count"] == 0
    assert summary["loss_finite"] is True
    assert summary["gradient_finite"] is True
    assert summary["component_grad_norms_readable"] is True
    assert summary["checkpoint_reload_passed"] is True
    assert summary["experimental_only"] is True
    assert summary["parameter_delta_l2"] == 0.01
    assert summary["runs_new_ppo_update"] is True
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False
    assert stage21_4_config["stage21_3_ppo_batch_validation_root"] == str(stage26_1 / "s21_3")
    assert stage21_4_config["stage26_2_authorized"] is False
    assert stage21_4_config["learning_rate"] == 2.0e-6


def test_stage26_2_rejects_unready_stage26_1_without_running_stage21_4(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path, next_required_change="repair_stage26_1_batch_gate")
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_2_required_inputs"
    assert "stage26_1_route_not_stage26_2" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage26_2_routes_batch_repair_when_synthetic_contract_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path, batch_mode="missing_synthetic")
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_2_synthetic_batch_update_contract"
    assert summary["synthetic_terrain_contract_missing_count"] == 1
    assert summary["synthetic_terrain_reward_provenance_missing_count"] == 1
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage26_2_routes_batch_repair_when_synthetic_writes_physical_payload(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path, batch_mode="physical_payload")
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_2_synthetic_batch_update_contract"
    assert summary["synthetic_physical_obstacle_payload_count"] == 1
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage26_2_rejects_source_checkpoint_mismatch_with_stage26_1_collector(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path, collector_checkpoint_bytes=b"collector-checkpoint")
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_2_required_inputs"
    assert "stage26_2_source_checkpoint_mismatch_stage26_1_collector" in summary["blocking_reason_codes"]
    assert summary["collector_source_checkpoint_match"] is False
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage26_2_rejects_upstream_physical_payload_before_stage21_4(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path, upstream_mode="physical_payload")
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_2_required_inputs"
    assert "stage26_1_upstream_physical_obstacle_payload_present" in summary["blocking_reason_codes"]
    assert summary["stage26_1_upstream_physical_obstacle_payload_count"] == 1
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage26_2_rejects_missing_upstream_artifacts_before_stage21_4(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path, write_upstream_artifacts=False)
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_2_required_inputs"
    assert any(reason.startswith("missing_stage26_1_upstream_") for reason in summary["blocking_reason_codes"])
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage26_2_routes_update_stability_repair_when_loss_is_non_finite(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path)
    _patch_stage21_4(monkeypatch, s26, stage21_4_mode="non_finite_loss")
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_2_synthetic_ppo_update_stability"
    assert summary["loss_finite"] is False


def test_stage26_2_routes_checkpoint_repair_when_checkpoint_boundary_fails(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path)
    _patch_stage21_4(monkeypatch, s26, stage21_4_mode="bad_checkpoint_boundary")
    config = _write_config(tmp_path, stage26_1)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_2_checkpoint_reload_boundary"
    assert summary["checkpoint_boundary_passed"] is False


def test_stage26_2_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as s26

    stage26_1 = _write_stage26_1_root(tmp_path)
    _patch_stage21_4(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_1, publishes_checkpoint=True)

    summary = s26.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_2_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_4").exists()


def _patch_stage21_4(monkeypatch, s26, *, stage21_4_mode: str = "valid") -> None:
    def fake_stage21_4(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = output_root / s26.stage21_4.CHECKPOINT_FILE
        checkpoint.write_bytes(b"checkpoint")
        _write_jsonl(output_root / s26.stage21_4.LOSS_AUDIT_FILE, [_loss_row(non_finite=stage21_4_mode == "non_finite_loss")])
        (output_root / s26.stage21_4.GRADIENT_AUDIT_FILE).write_text(
            json.dumps(_gradient_audit(parameter_delta_l2=0.0 if stage21_4_mode == "zero_parameter_delta" else 0.01)),
            encoding="utf-8",
        )
        (output_root / s26.stage21_4.CHECKPOINT_AUDIT_FILE).write_text(
            json.dumps(
                _checkpoint_audit(
                    output_root,
                    stage21_3_root=cfg["stage21_3_ppo_batch_validation_root"],
                    bad_boundary=stage21_4_mode == "bad_checkpoint_boundary",
                )
            ),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_5_single_seed_ppo_pilot",
            "train_transition_count": 1,
            "final_total_loss": None if stage21_4_mode == "non_finite_loss" else 0.25,
            "final_post_update_approx_kl": 0.001,
            "parameter_delta_l2": 0.0 if stage21_4_mode == "zero_parameter_delta" else 0.01,
            "source_xunce_checkpoint_sha256": "source-sha",
            "checkpoint_reload_passed": stage21_4_mode != "bad_checkpoint_boundary",
            "experimental_checkpoint": True,
            "experimental_checkpoint_path": str(checkpoint),
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        }
        (output_root / s26.stage21_4.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage21_4, "run_xunce_stage21_4_tiny_ppo_update_smoke", fake_stage21_4)


def _write_stage26_1_root(
    tmp_path: Path,
    *,
    next_required_change: str = "run_stage26_2_synthetic_terrain_ppo_update_smoke",
    batch_mode: str = "valid",
    upstream_mode: str = "valid",
    collector_checkpoint_bytes: bytes = b"checkpoint",
    write_upstream_artifacts: bool = True,
) -> Path:
    root = tmp_path / "stage26_1"
    s21_3 = root / "s21_3"
    s21_1 = root / "s21_1"
    s21_3.mkdir(parents=True)
    s21_1.mkdir(parents=True)
    summary = {
        "schema_version": "xunce-stage26-1-summary/v1",
        "status": "passed",
        "next_required_change": next_required_change,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
        "synthetic_terrain_hash": "stage26-aggregate-hash",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "synthetic_transition_contract_missing_count": 0,
        "synthetic_reward_provenance_missing_count": 0,
        "synthetic_batch_contract_missing_count": 0,
        "synthetic_physical_obstacle_pollution_count": 0,
        "point_grid_path_cost_fallback_used_count": 0,
        "point_only_reward_fallback_used_count": 0,
        "unobstructed_theta_reward_fallback_used_count": 0,
        "hard_risk_violation_count": 0,
        "mask_violation_count": 0,
        "path_planning_failure_count": 0,
        "open_grid_fallback_count": 0,
        "trainable_transition_count": 1,
        "reward_evaluation_row_count": 1,
        "batch_row_count": 1,
    }
    (root / "xunce-stage26-1-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    batch_summary = {
        "schema_version": "xunce-stage21-3-ppo-batch-validation-summary/v1",
        "status": "passed",
        "next_required_change": "implement_stage21_4_tiny_ppo_update_smoke",
        "trainable_transition_count": 1,
        "synthetic_terrain_contract_required": True,
        "synthetic_terrain_contract_missing_count": 0 if batch_mode == "valid" else 1,
    }
    (s21_3 / "xunce-stage21-3-ppo-batch-validation-summary.json").write_text(json.dumps(batch_summary), encoding="utf-8")
    (s21_3 / "xunce-stage21-3-lineage-audit.json").write_text(json.dumps({"reward_profile_hash_count": 1}), encoding="utf-8")
    _write_jsonl(s21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [_batch_row(mode=batch_mode)])
    if write_upstream_artifacts:
        _write_jsonl(s21_1 / "xunce-stage21-1-ppo-trainable-batch.jsonl", [_upstream_row(mode=upstream_mode)])
        _write_jsonl(s21_1 / "xunce-stage21-1-ppo-rollout-transitions.jsonl", [])
        _write_jsonl(s21_1 / "xunce-stage21-1-reward-audit.jsonl", [])
        s21_2 = root / "s21_2"
        s21_2.mkdir(parents=True, exist_ok=True)
        _write_jsonl(s21_2 / "xunce-stage21-2-reward-contract-evaluation.jsonl", [])
    checkpoint_path = tmp_path / "source-xunce.pt"
    manifest = {
        "schema_version": "xunce-stage21-1-manifest/v1",
        "model_audit": {
            "xunce_candidate_checkpoint": str(checkpoint_path),
            "xunce_checkpoint_audit": {
                "checkpoint_loaded": True,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(collector_checkpoint_bytes).hexdigest(),
            },
        },
    }
    (s21_1 / "xunce-stage21-1-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _upstream_row(*, mode: str = "valid") -> dict:
    row = {"transition_id": "stage26:step-0:sample-1", "info": {"physical_obstacle_cells_written": False}}
    if mode == "physical_payload":
        row["info"]["physical_obstacle_cells"] = [[9, 9]]
    return row


def _batch_row(*, mode: str = "valid") -> dict:
    candidate_count = 3
    info = {
        "candidate_viewpoints": [[0, 0, 0], [1, 2, 45], [2, 3, 90]],
        "candidate_theta_deg": [0, 45, 90],
        "selected_viewpoint": [1, 2, 45],
        "selected_theta_deg": 45,
        "action_mask": [True, True, True],
        "sampling_mask": [True, True, True],
        "hard_risk_clean_mask": [True, True, True],
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "path_cost_sources": ["hybrid_astar_pose_path/v1"] * candidate_count,
        "hybrid_astar_path_costs": [4.5, 5.5, 6.5],
        "hybrid_astar_pose_path_hashes": ["pose-path-0", "pose-path-hash", "pose-path-2"],
        "hybrid_astar_trajectory_kinds": ["hybrid_astar_pose_path"] * candidate_count,
        "legacy_grid_astar_path_costs": [3.0, 4.0, 5.0],
        "hybrid_vs_grid_path_cost_deltas": [1.5, 1.5, 1.5],
        "default_astar_replaced": False,
        "default_astar_replaced_flags": [False] * candidate_count,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_ackermann_feasible_claimed_flags": [False] * candidate_count,
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
    row = {
        "schema_version": "xunce-stage21-3-ppo-trainable-batch/v1",
        "transition_id": "stage26:step-0:sample-1",
        "scenario_id": "stage26",
        "step_index": 0,
        "stage21_3_split": "train",
        "transition_trainable": True,
        "reward_trainable": True,
        "action_index": 1,
        "old_log_prob": -0.5,
        "return": 1.0,
        "advantage": 0.2,
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
        "info": info,
        "xunce_batch": _xunce_batch(candidate_count),
    }
    if mode == "missing_synthetic":
        row["synthetic_terrain_reward_provenance"] = False
        row.pop("synthetic_terrain_hash", None)
        info.pop("synthetic_terrain_hash", None)
    if mode == "physical_payload":
        row["physical_obstacle_cells"] = [[1, 1]]
        info["physical_obstacle_cells"] = [[1, 1]]
    return row


def _xunce_batch(candidate_count: int) -> dict:
    return {
        "candidate_features": {"dtype": "float32", "shape": [1, candidate_count, 8], "values": [[[0.0] * 8 for _ in range(candidate_count)]]},
        "action_mask": {"dtype": "bool", "shape": [1, candidate_count], "values": [[True] * candidate_count]},
        "candidate_missing_indicators": {"dtype": "float32", "shape": [1, candidate_count, 3], "values": [[[0.0] * 3 for _ in range(candidate_count)]]},
        "context_features": {"dtype": "float32", "shape": [1, candidate_count, 7], "values": [[[0.0] * 7 for _ in range(candidate_count)]]},
        "edge_features": {"dtype": "float32", "shape": [1, 5], "values": [[0.0] * 5]},
        "edge_index": {"dtype": "int64", "shape": [1, 2], "values": [[0, 0]]},
        "memory_features": {"dtype": "float32", "shape": [1, 6], "values": [[0.0] * 6]},
    }


def _loss_row(*, non_finite: bool = False) -> dict:
    return {
        "schema_version": "xunce-stage21-4-ppo-loss-audit/v1",
        "epoch_index": 0,
        "transition_count": 1,
        "total_loss": None if non_finite else 0.25,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "post_update_approx_kl": 0.001,
        "post_update_clip_fraction": 0.0,
        "effective_advantage_std": 1.0,
    }


def _gradient_audit(*, parameter_delta_l2: float = 0.01) -> dict:
    return {
        "schema_version": "xunce-stage21-4-gradient-audit/v1",
        "pre_clip_grad_norm": 0.8,
        "post_clip_grad_norm": 0.7,
        "max_grad_norm": 1.0,
        "parameter_delta_l2": parameter_delta_l2,
        "parameter_with_grad_count": 4,
        "component_grad_norms": {
            "total_loss_grad_norm": 0.8,
            "policy_loss_grad_norm": 0.6,
            "value_loss_grad_norm": 0.1,
            "entropy_loss_grad_norm": 0.01,
        },
    }


def _checkpoint_audit(output_root: Path, *, stage21_3_root: str, bad_boundary: bool = False) -> dict:
    metadata = {
        "experimental_only": not bad_boundary,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "stage21_3_ppo_batch_validation_root": stage21_3_root,
        "source_checkpoint_sha256": "source-sha",
    }
    return {
        "schema_version": "xunce-stage21-4-checkpoint-audit/v1",
        "experimental_checkpoint_exists": True,
        "checkpoint_reload_passed": not bad_boundary,
        "experimental_checkpoint_path": str(output_root / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt"),
        "experimental_checkpoint_sha256": "experimental-sha",
        "metadata": metadata,
    }


def _write_config(tmp_path: Path, stage26_1: Path, **overrides) -> Path:
    source_checkpoint = tmp_path / "source-xunce.pt"
    source_checkpoint.write_bytes(b"checkpoint")
    high_fidelity = tmp_path / "hf.json"
    high_fidelity.write_text(json.dumps({"schema_version": "hf-test"}), encoding="utf-8")
    payload = {
        "schema_version": "xunce-stage26-2-synthetic-terrain-ppo-update-smoke-config/v1",
        "stage26_1_root": str(stage26_1),
        "stage21_4_base_config": "configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json",
        "xunce_candidate_checkpoint": str(source_checkpoint),
        "high_fidelity_config": str(high_fidelity),
        "stage26_2_authorized": False,
        "runs_new_ppo_update": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_2_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
