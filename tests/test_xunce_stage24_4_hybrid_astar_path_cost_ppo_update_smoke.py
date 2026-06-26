import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def test_stage24_4_runs_hybrid_path_cost_ppo_update_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path)
    _patch_stage21_4(monkeypatch, s24)
    config = _write_config(tmp_path, stage24_3)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage21_4_config = json.loads((tmp_path / "out" / "xunce-stage24-4-stage21-4-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke"
    assert summary["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert summary["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert summary["hybrid_path_batch_contract_missing_count"] == 0
    assert summary["xunce_batch_viewpoint_shape_mismatch_count"] == 0
    assert summary["action_viewpoint_path_cost_binding_mismatch_count"] == 0
    assert summary["selected_action_mask_violation_count"] == 0
    assert summary["point_grid_path_cost_fallback_used_count"] == 0
    assert summary["loss_finite"] is True
    assert summary["gradient_finite"] is True
    assert summary["component_grad_norms_readable"] is True
    assert summary["checkpoint_reload_passed"] is True
    assert summary["experimental_only"] is True
    assert summary["parameter_delta_l2"] == 0.01
    assert summary["default_astar_replaced"] is False
    assert summary["ackermann_feasible_claimed"] is False
    assert summary["runs_new_ppo_update"] is True
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False
    assert stage21_4_config["stage21_3_ppo_batch_validation_root"] == str(stage24_3 / "s21_3")
    assert stage21_4_config["learning_rate"] == 2.0e-6
    assert stage21_4_config["loss_scale"] == 0.25
    assert stage21_4_config["normalize_minibatch_advantages"] is True


def test_stage24_4_rejects_unready_stage24_3_without_running_stage21_4(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path, next_required_change="repair_stage24_3_batch_gate")
    _patch_stage21_4(monkeypatch, s24)
    config = _write_config(tmp_path, stage24_3)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_4_required_inputs"
    assert "stage24_3_route_not_stage24_4" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage24_4_routes_batch_repair_when_batch_is_grid_only(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path, batch_mode="grid_only")
    _patch_stage21_4(monkeypatch, s24)
    config = _write_config(tmp_path, stage24_3)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_4_hybrid_path_batch_update_contract"
    assert summary["hybrid_path_batch_contract_missing_count"] == 1
    assert summary["point_grid_path_cost_fallback_used_count"] == 1
    assert not (tmp_path / "out" / "s21_4").exists()


def test_stage24_4_routes_batch_repair_when_action_path_cost_binding_mismatches(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path, batch_mode="path_binding_mismatch")
    _patch_stage21_4(monkeypatch, s24)
    config = _write_config(tmp_path, stage24_3)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_4_hybrid_path_batch_update_contract"
    assert summary["action_viewpoint_path_cost_binding_mismatch_count"] == 1


def test_stage24_4_routes_update_stability_repair_when_loss_is_non_finite(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path)
    _patch_stage21_4(monkeypatch, s24, stage21_4_mode="non_finite_loss")
    config = _write_config(tmp_path, stage24_3)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_4_hybrid_path_ppo_update_stability"
    assert summary["loss_finite"] is False


def test_stage24_4_routes_update_stability_repair_when_parameter_delta_is_zero(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path)
    _patch_stage21_4(monkeypatch, s24, stage21_4_mode="zero_parameter_delta")
    config = _write_config(tmp_path, stage24_3)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_4_hybrid_path_ppo_update_stability"
    assert summary["gradient_finite"] is False


def test_stage24_4_routes_checkpoint_repair_when_checkpoint_boundary_fails(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path)
    _patch_stage21_4(monkeypatch, s24, stage21_4_mode="bad_checkpoint_boundary")
    config = _write_config(tmp_path, stage24_3)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_4_checkpoint_reload_boundary"
    assert summary["checkpoint_boundary_passed"] is False


def test_stage24_4_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as s24

    stage24_3 = _write_stage24_3_root(tmp_path)
    _patch_stage21_4(monkeypatch, s24)
    config = _write_config(tmp_path, stage24_3, publishes_checkpoint=True)

    summary = s24.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_4_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_4").exists()


def _patch_stage21_4(monkeypatch, s24, *, stage21_4_mode: str = "valid") -> None:
    def fake_stage21_4(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = output_root / s24.stage21_4.CHECKPOINT_FILE
        checkpoint.write_bytes(b"checkpoint")
        _write_jsonl(output_root / s24.stage21_4.LOSS_AUDIT_FILE, [_loss_row(non_finite=stage21_4_mode == "non_finite_loss")])
        (output_root / s24.stage21_4.GRADIENT_AUDIT_FILE).write_text(
            json.dumps(_gradient_audit(parameter_delta_l2=0.0 if stage21_4_mode == "zero_parameter_delta" else 0.01)),
            encoding="utf-8",
        )
        (output_root / s24.stage21_4.CHECKPOINT_AUDIT_FILE).write_text(
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
        (output_root / s24.stage21_4.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s24.stage21_4, "run_xunce_stage21_4_tiny_ppo_update_smoke", fake_stage21_4)


def _write_stage24_3_root(
    tmp_path: Path,
    *,
    next_required_change: str = "run_stage24_4_hybrid_astar_path_cost_ppo_update_smoke",
    batch_mode: str = "valid",
) -> Path:
    root = tmp_path / "stage24_3"
    s21_3 = root / "s21_3"
    s21_3.mkdir(parents=True)
    summary = {
        "schema_version": "xunce-stage24-3-summary/v1",
        "status": "passed",
        "next_required_change": next_required_change,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "max_traversable_slope_deg": 30.0,
        "platform_contract_hash": "platform-hash",
        "trainable_transition_count": 1,
        "reward_evaluation_row_count": 1,
        "batch_row_count": 1,
        "transition_id_count": 1,
        "reward_transition_id_count": 1,
        "batch_transition_id_count": 1,
        "stage21_1_source_roi_expansion_root_match": True,
        "stage21_3_rejects_old_reward_batch": True,
        "stage21_3_rejects_grid_only_path_cost_batch": True,
        "hybrid_path_transition_contract_missing_count": 0,
        "hybrid_path_reward_contract_missing_count": 0,
        "hybrid_path_batch_contract_missing_count": 0,
        "point_grid_path_cost_fallback_used_count": 0,
        "sampling_mask_contract_mismatch_count": 0,
        "action_viewpoint_path_cost_binding_mismatch_count": 0,
        "hard_risk_violation_count": 0,
        "mask_violation_count": 0,
        "path_planning_failure_count": 0,
        "open_grid_fallback_count": 0,
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    (root / "xunce-stage24-3-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    batch_summary = {
        "schema_version": "xunce-stage21-3-ppo-batch-validation-summary/v1",
        "status": "passed",
        "next_required_change": "implement_stage21_4_tiny_ppo_update_smoke",
        "trainable_transition_count": 1,
        "stage21_1_collector_root": str(root / "s21_1"),
    }
    (s21_3 / "xunce-stage21-3-ppo-batch-validation-summary.json").write_text(json.dumps(batch_summary), encoding="utf-8")
    (s21_3 / "xunce-stage21-3-lineage-audit.json").write_text(json.dumps({"reward_profile_hash_count": 1}), encoding="utf-8")
    _write_jsonl(s21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [_batch_row(mode=batch_mode)])
    return root


def _batch_row(*, mode: str = "valid") -> dict:
    candidate_count = 80
    xunce_candidate_count = 10 if mode == "point_only_xunce_batch" else candidate_count
    info = {
        "candidate_viewpoints": [[index, index + 1, (index % 8) * 45] for index in range(candidate_count)],
        "candidate_theta_deg": [(index % 8) * 45 for index in range(candidate_count)],
        "selected_viewpoint": [47, 48, 315],
        "selected_theta_deg": 315,
        "action_mask": [True] * candidate_count,
        "sampling_mask": [True] * candidate_count,
        "hard_risk_clean_mask": [True] * candidate_count,
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "path_cost_sources": ["hybrid_astar_pose_path/v1"] * candidate_count,
        "hybrid_astar_path_costs": [5.5] * candidate_count,
        "hybrid_astar_pose_path_hashes": ["pose-path-hash"] * candidate_count,
        "hybrid_astar_trajectory_kinds": ["hybrid_astar_pose_path"] * candidate_count,
        "legacy_grid_astar_path_costs": [4.0] * candidate_count,
        "hybrid_vs_grid_path_cost_deltas": [1.5] * candidate_count,
        "default_astar_replaced": False,
        "default_astar_replaced_flags": [False] * candidate_count,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_ackermann_feasible_claimed_flags": [False] * candidate_count,
    }
    row = {
        "schema_version": "xunce-stage21-3-ppo-trainable-batch/v1",
        "transition_id": "s1:step-0:sample-1",
        "scenario_id": "s1",
        "step_index": 0,
        "stage21_3_split": "train",
        "transition_trainable": True,
        "reward_trainable": True,
        "action_index": 47,
        "old_log_prob": -0.5,
        "return": 1.0,
        "advantage": 0.2,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "candidate_viewpoint": [47, 48, 315],
        "candidate_theta_deg": 315,
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
        "info": info,
        "xunce_batch": _xunce_batch(xunce_candidate_count),
    }
    if mode == "grid_only":
        row["hybrid_astar_path_cost_reward_contract"] = False
        row["path_cost_source"] = "legacy_grid_astar_path/v1"
        row["point_grid_path_cost_fallback_used"] = True
        row["reward_metrics"] = {"path_cost_m": 4.0}
    if mode == "path_binding_mismatch":
        info["hybrid_astar_pose_path_hashes"][47] = "wrong-hash"
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


def _write_config(tmp_path: Path, stage24_3: Path, **overrides) -> Path:
    source_checkpoint = tmp_path / "source-xunce.pt"
    source_checkpoint.write_bytes(b"checkpoint")
    high_fidelity = tmp_path / "hf.json"
    high_fidelity.write_text(json.dumps({"schema_version": "hf-test"}), encoding="utf-8")
    payload = {
        "schema_version": "xunce-stage24-4-hybrid-astar-path-cost-ppo-update-smoke-config/v1",
        "stage24_3_root": str(stage24_3),
        "stage21_4_base_config": "configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json",
        "xunce_candidate_checkpoint": str(source_checkpoint),
        "high_fidelity_config": str(high_fidelity),
        "stage24_4_authorized": False,
        "runs_new_ppo_update": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage24_4_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
