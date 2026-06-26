import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage24_5_runs_hybrid_path_eval_and_routes_stage24_6(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="changed_and_improved")
    planner_overrides = _hybrid_planner_overrides()
    config = _write_config(tmp_path, stage24_4, **planner_overrides)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage21_5_config = json.loads((tmp_path / "out" / "xunce-stage24-5-stage21-5-config.json").read_text(encoding="utf-8"))
    high_fidelity_config = json.loads((tmp_path / "out" / "xunce-stage24-5-high-fidelity-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage24_6_hybrid_astar_path_cost_multi_seed_ppo_pilot"
    assert summary["strong_state_join_available_count"] == 2
    assert summary["selected_viewpoint_changed_count"] == 1
    assert summary["selected_theta_changed_count"] == 1
    assert summary["selected_cell_changed_count"] == 0
    assert summary["mean_abs_probability_delta"] > 0
    assert summary["selected_action_probability_delta"] > 0
    assert summary["final_coverage_delta"] > 0
    assert summary["coverage_auc_delta"] > 0
    assert summary["path_cost_delta"] == 1.0
    assert summary["coverage_per_100m_delta"] == 2.0
    assert summary["grid_fallback_count"] == 0
    assert summary["default_astar_replaced"] is False
    assert summary["ackermann_feasible_claimed"] is False
    assert summary["publishes_checkpoint"] is False
    assert stage21_5_config["stage21_4_tiny_ppo_update_smoke_root"] == str(stage24_4 / "s21_4")
    assert stage21_5_config["coverage_source"] == s24.SLOPE_COVERAGE_SOURCE
    assert stage21_5_config["path_cost_source"] == s24.HYBRID_ASTAR_PATH_COST_SOURCE
    assert stage21_5_config["hybrid_astar_pose_path_cost_enabled"] is True
    assert stage21_5_config["runs_new_ppo_update"] is False
    assert high_fidelity_config["hybrid_astar_pose_path_cost_enabled"] is True
    assert high_fidelity_config["path_cost_source"] == s24.HYBRID_ASTAR_PATH_COST_SOURCE
    assert high_fidelity_config["coverage_source"] == s24.SLOPE_COVERAGE_SOURCE
    assert high_fidelity_config["default_astar_replaced"] is False
    assert high_fidelity_config["hybrid_astar_ackermann_feasible_claimed"] is False
    assert high_fidelity_config["required_scenario_count"] == 2
    assert high_fidelity_config["rollout_steps"] == 4
    assert high_fidelity_config["dynamic_max_candidates_per_step"] == 36
    assert high_fidelity_config["dynamic_proposal_pool_limit_per_step"] == 288
    for key, value in planner_overrides.items():
        assert stage21_5_config[key] == value
        assert high_fidelity_config[key] == value


def test_stage24_5_rejects_unready_stage24_4_without_running_stage21_5(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path, next_required_change="repair_stage24_4_hybrid_path_ppo_update_stability")
    _patch_stage21_5(monkeypatch, s24)
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_5_required_inputs"
    assert "stage24_4_route_not_stage24_5" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def test_stage24_5_routes_binding_repair_when_hybrid_path_fields_are_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="missing_hybrid_path_field")
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_5_hybrid_path_inference_binding"
    assert summary["hybrid_path_inference_required_field_missing_count"] > 0


def test_stage24_5_counts_path_source_mismatch_even_when_hybrid_fields_are_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="missing_hybrid_with_grid_source")
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_5_hybrid_path_inference_binding"
    assert summary["hybrid_path_inference_required_field_missing_count"] > 0
    assert summary["path_cost_source_mismatch_count"] > 0
    assert summary["grid_fallback_count"] > 0


def test_stage24_5_routes_binding_repair_when_path_cost_source_is_grid(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="grid_path_source")
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_5_hybrid_path_inference_binding"
    assert summary["path_cost_source_mismatch_count"] > 0
    assert summary["grid_fallback_count"] > 0


def test_stage24_5_routes_discrete_margin_when_probabilities_move_but_action_does_not(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="prob_changed_action_same")
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "calibrate_stage24_hybrid_path_discrete_margin_crossing"
    assert summary["selected_viewpoint_changed_count"] == 0
    assert summary["mean_abs_probability_delta"] > 0.00001


def test_stage24_5_routes_signal_repair_when_action_and_probabilities_do_not_change(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="unchanged")
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_hybrid_path_policy_update_signal_strength"
    assert summary["selected_viewpoint_changed_count"] == 0
    assert summary["mean_abs_probability_delta"] == 0.0


def test_stage24_5_routes_credit_repair_when_viewpoint_changes_without_coverage_lift(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="changed_no_improvement")
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_hybrid_path_credit_assignment"
    assert summary["selected_viewpoint_changed_count"] == 1
    assert summary["final_coverage_delta"] == 0.0


def test_stage24_5_routes_safety_repair_on_hard_risk_regression(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24, mode="hard_risk_regression")
    config = _write_config(tmp_path, stage24_4)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_5_hybrid_path_eval_safety_regression"
    assert summary["hard_risk_violation_count"] == 1


def test_stage24_5_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke as s24

    stage24_4 = _write_stage24_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s24)
    config = _write_config(tmp_path, stage24_4, publishes_checkpoint=True)

    summary = s24.run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_5_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def _patch_stage21_5(monkeypatch, s24, *, mode: str = "changed_and_improved") -> None:
    def fake_stage21_5(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["coverage_source"] == s24.SLOPE_COVERAGE_SOURCE
        assert cfg["path_cost_source"] == s24.HYBRID_ASTAR_PATH_COST_SOURCE
        assert cfg["hybrid_astar_pose_path_cost_enabled"] is True
        output_root.mkdir(parents=True, exist_ok=True)
        pre_root = output_root / "pre_ppo_xunce"
        post_root = output_root / "post_ppo_xunce"
        pre_root.mkdir(parents=True, exist_ok=True)
        post_root.mkdir(parents=True, exist_ok=True)
        pre_rows = [_inference_row(0, 0), _inference_row(1, 45)]
        if mode == "missing_hybrid_path_field":
            post_rows = [_inference_row(0, 45, omit_hybrid=True), _inference_row(1, 45)]
        elif mode == "missing_hybrid_with_grid_source":
            post_rows = [
                _inference_row(
                    0,
                    45,
                    omit_hybrid=True,
                    path_cost_source="legacy_grid_astar_path/v1",
                    grid_fallback=True,
                ),
                _inference_row(1, 45),
            ]
        elif mode == "grid_path_source":
            post_rows = [_inference_row(0, 45, path_cost_source="legacy_grid_astar_path/v1", grid_fallback=True), _inference_row(1, 45)]
        elif mode == "unchanged":
            post_rows = [_inference_row(0, 0), _inference_row(1, 45)]
        elif mode == "prob_changed_action_same":
            post_rows = [_inference_row(0, 0, probs=[0.51, 0.49]), _inference_row(1, 45, probs=[0.45, 0.55])]
        else:
            post_rows = [_inference_row(0, 45, probs=[0.3, 0.7]), _inference_row(1, 45, probs=[0.4, 0.6])]
        _write_jsonl(pre_root / s24.MODEL_INFERENCE_FILE, pre_rows)
        _write_jsonl(post_root / s24.MODEL_INFERENCE_FILE, post_rows)
        improved = mode == "changed_and_improved"
        delta = {
            "final_coverage_rate_mean_delta": 0.02 if improved else 0.0,
            "coverage_curve_auc_mean_delta": 0.03 if improved else 0.0,
            "path_cost_total_m_mean_delta": 1.0,
            "coverage_per_100m_mean_delta": 2.0,
        }
        (output_root / s24.stage21_5.DELTA_FILE).write_text(json.dumps(delta), encoding="utf-8")
        summary = {
            "schema_version": "xunce-stage21-5-post-update-evaluation-summary/v1",
            "status": "failed" if mode == "hard_risk_regression" else "passed",
            "next_required_change": (
                "repair_stage21_5_hard_risk_boundary"
                if mode == "hard_risk_regression"
                else "implement_stage21_6_multi_seed_ppo_pilot"
            ),
            "pre_evaluation_root": str(pre_root),
            "post_evaluation_root": str(post_root),
            "final_coverage_rate_delta": delta["final_coverage_rate_mean_delta"],
            "coverage_curve_auc_delta": delta["coverage_curve_auc_mean_delta"],
            "path_cost_total_m_delta": delta["path_cost_total_m_mean_delta"],
            "coverage_per_100m_delta": delta["coverage_per_100m_mean_delta"],
            "post_hard_risk_violation_count": 1 if mode == "hard_risk_regression" else 0,
            "post_mask_violation_count": 0,
            "post_path_planning_failure_count": 0,
            "post_open_grid_fallback_count": 0,
            "scenario_regression_count": 0,
            "reason_codes": [],
            "sample_count_too_low_for_performance_claim": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        (output_root / s24.stage21_5.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s24.stage21_5, "run_xunce_stage21_5_post_update_offline_trajectory_evaluation", fake_stage21_5)


def _inference_row(
    index: int,
    selected_theta: int,
    *,
    omit_hybrid: bool = False,
    path_cost_source: str = "hybrid_astar_pose_path/v1",
    grid_fallback: bool = False,
    probs: list[float] | None = None,
) -> dict:
    row = {
        "schema_version": "xunce-exploration-coverage-model-inference/v1",
        "scenario_id": f"scenario-{index}",
        "step_index": index,
        "current_cell": [index, 0],
        "covered_cells_hash": f"covered-{index}",
        "candidate_set_hash": f"candidate-set-{index}",
        "candidate_viewpoints": [[index, 1, 0], [index, 1, 45]],
        "candidate_theta_deg": selected_theta,
        "selected_action_index": 0 if selected_theta == 0 else 1,
        "candidate_viewpoint": [index, 1, selected_theta],
        "selected_viewpoint": [index, 1, selected_theta],
        "selected_theta_deg": selected_theta,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": path_cost_source,
        "hybrid_astar_path_cost": 5.5,
        "hybrid_astar_pose_path_hash": f"pose-path-{index}-{selected_theta}",
        "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
        "legacy_grid_astar_path_cost": 4.0,
        "hybrid_vs_grid_path_cost_delta": 1.5,
        "point_grid_path_cost_fallback_used": grid_fallback,
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "max_traversable_slope_deg": 30.0,
        "obstacle_occlusion_enabled": True,
        "obstacle_source_kind": "slope_blocked_as_obstacle_proxy",
        "obstacle_aware_theta_coverage_hash": f"obstacle-theta-hash-{index}-{selected_theta}",
        "slope_obstacle_source_hash": "slope-hash",
        "platform_contract_hash": "platform-hash",
        "action_probs": probs or ([0.6, 0.4] if selected_theta == 0 else [0.4, 0.6]),
        "logits": [0.1, 0.2],
        "masked_logits": [0.1, 0.2],
    }
    if omit_hybrid:
        row.pop("hybrid_astar_path_cost")
        row.pop("hybrid_astar_pose_path_hash")
    return row


def _write_stage24_4_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke",
) -> Path:
    root = tmp_path / "stage24_4"
    stage24_3 = tmp_path / "stage24_3"
    s21_4 = root / "s21_4"
    s21_4.mkdir(parents=True, exist_ok=True)
    stage24_3.mkdir(parents=True, exist_ok=True)
    (s21_4 / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt").write_bytes(b"checkpoint")
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source")
    stage24_3_summary = {
        "schema_version": "xunce-stage24-3-summary/v1",
        "status": "passed",
        "stage23_2b_high_res_root": str(tmp_path / "high_res_roi_expansion"),
        "stage21_1_actual_source_roi_expansion_root": str(tmp_path / "high_res_roi_expansion"),
    }
    summary = {
        "schema_version": "xunce-stage24-4-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "stage21_4_root": str(s21_4),
        "stage24_3_root": str(stage24_3),
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "max_traversable_slope_deg": 30.0,
        "checkpoint_exists": True,
        "checkpoint_reload_passed": True,
        "checkpoint_boundary_passed": True,
        "stage21_3_lineage_passed": True,
        "experimental_only": True,
        "source_checkpoint_sha_consistent": True,
        "hybrid_path_batch_contract_missing_count": 0,
        "xunce_batch_viewpoint_shape_mismatch_count": 0,
        "action_viewpoint_path_cost_binding_mismatch_count": 0,
        "selected_info_viewpoint_mismatch_count": 0,
        "hybrid_path_provenance_array_length_mismatch_count": 0,
        "selected_action_mask_violation_count": 0,
        "point_grid_path_cost_fallback_used_count": 0,
        "default_astar_replaced": False,
        "default_astar_replaced_count": 0,
        "ackermann_feasible_claimed": False,
        "ackermann_feasible_claimed_count": 0,
        "pre_clip_grad_norm": 37.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    checkpoint_audit = {
        "source_checkpoint_path": str(source_checkpoint),
        "checkpoint_path": str(s21_4 / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt"),
        "checkpoint_reload_passed": True,
        "experimental_only": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    (stage24_3 / "xunce-stage24-3-summary.json").write_text(json.dumps(stage24_3_summary), encoding="utf-8")
    (root / "xunce-stage24-4-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "xunce-stage24-4-checkpoint-boundary-audit.json").write_text(json.dumps(checkpoint_audit), encoding="utf-8")
    return root


def _write_config(tmp_path: Path, stage24_4_root: Path, **overrides) -> Path:
    base_config = tmp_path / "stage21_5_base.json"
    high_fidelity = tmp_path / "high_fidelity.json"
    base_config.write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-5-post-update-offline-trajectory-evaluation-config/v1",
                "stage_id": "xunce-stage21-5-post-update-offline-trajectory-evaluation",
                "stage21_4_tiny_ppo_update_smoke_root": "placeholder",
                "high_fidelity_config": str(high_fidelity),
                "execute_high_fidelity_evaluations": True,
                "required_scenario_count": 2,
                "rollout_steps": 4,
                "dynamic_max_candidates_per_step": 36,
                "dynamic_proposal_pool_limit_per_step": 288,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "canary_traffic_fraction": 0.0,
            }
        ),
        encoding="utf-8",
    )
    high_fidelity.write_text(json.dumps({"schema_version": "xunce-high-fidelity-exploration-coverage-comparison-config/v1"}), encoding="utf-8")
    payload = {
        "schema_version": "xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke-config/v1",
        "stage_id": "xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke",
        "stage24_4_root": str(stage24_4_root),
        "stage21_5_base_config": str(base_config),
        "high_fidelity_config": str(high_fidelity),
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 2,
        "min_mean_abs_probability_delta_for_signal": 0.00001,
        "stage24_5_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage24_5_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _hybrid_planner_overrides() -> dict:
    return {
        "hybrid_astar_theta_bin_count": 36,
        "hybrid_astar_goal_position_tolerance_m": 1.75,
        "hybrid_astar_goal_theta_tolerance_deg": 7.5,
        "hybrid_astar_max_iterations": 54321,
        "hybrid_astar_primitive_duration_s": 1.25,
        "hybrid_astar_integration_dt_s": 0.5,
        "hybrid_astar_max_speed_mps": 2.5,
        "hybrid_astar_max_angular_speed_degps": 60.0,
        "hybrid_astar_rotation_cost_weight": 0.33,
        "hybrid_astar_reverse_penalty_weight": 0.77,
        "hybrid_astar_turn_penalty_weight": 0.12,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
