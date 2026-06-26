import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage23_5_runs_slope_theta_eval_and_routes_stage23_6(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="changed_and_improved")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage21_5_config = json.loads((tmp_path / "out" / "xunce-stage23-5-stage21-5-config.json").read_text(encoding="utf-8"))
    high_fidelity_config = json.loads((tmp_path / "out" / "xunce-stage23-5-high-fidelity-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage23_6_slope_obstacle_aware_theta_multi_seed_ppo_pilot"
    assert summary["strong_state_join_available_count"] == 2
    assert summary["selected_viewpoint_changed_count"] == 1
    assert summary["selected_theta_changed_count"] == 1
    assert summary["selected_cell_changed_count"] == 0
    assert summary["mean_abs_probability_delta"] > 0
    assert summary["final_coverage_delta"] > 0
    assert summary["coverage_auc_delta"] > 0
    assert summary["publishes_checkpoint"] is False
    assert stage21_5_config["stage21_4_tiny_ppo_update_smoke_root"] == str(stage23_4 / "s21_4")
    assert stage21_5_config["pre_ppo_evaluation_root"].endswith("out\\pre") or stage21_5_config["pre_ppo_evaluation_root"].endswith("out/pre")
    assert stage21_5_config["post_ppo_evaluation_root"].endswith("out\\post") or stage21_5_config["post_ppo_evaluation_root"].endswith("out/post")
    assert stage21_5_config["coverage_source"] == s23.SLOPE_COVERAGE_SOURCE
    assert stage21_5_config["obstacle_occlusion_enabled"] is True
    assert stage21_5_config["sensor_model_id"] == "theta-fov-90-range-radius/v1"
    assert stage21_5_config["sensor_range_cells"] == 2
    assert high_fidelity_config["coverage_source"] == s23.SLOPE_COVERAGE_SOURCE
    assert high_fidelity_config["obstacle_occlusion_enabled"] is True
    assert high_fidelity_config["sensor_model_id"] == "theta-fov-90-range-radius/v1"
    assert high_fidelity_config["sensor_range_cells"] == 2
    assert high_fidelity_config["required_scenario_count"] == 2
    assert high_fidelity_config["rollout_steps"] == 4
    assert high_fidelity_config["dynamic_max_candidates_per_step"] == 36
    assert high_fidelity_config["dynamic_proposal_pool_limit_per_step"] == 288
    assert high_fidelity_config["dynamic_validation_work_root"].endswith("_xunce_dynamic_validation_work")
    assert high_fidelity_config["source_roi_expansion_root"].endswith("high_res_roi_expansion")


def test_stage23_5_rejects_unready_stage23_4_without_running_stage21_5(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path, next_required_change="repair_stage23_4_slope_theta_ppo_update_stability")
    _patch_stage21_5(monkeypatch, s23)
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage23_5_required_inputs"
    assert "stage23_4_route_not_stage23_5" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def test_stage23_5_routes_binding_repair_when_slope_inference_fields_are_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="missing_slope_field")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_5_slope_theta_inference_binding"
    assert summary["slope_theta_inference_required_field_missing_count"] > 0


def test_stage23_5_routes_binding_repair_when_coverage_source_is_old_theta(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="old_theta_source")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_5_slope_theta_inference_binding"
    assert summary["coverage_source_mismatch_count"] > 0


def test_stage23_5_routes_binding_repair_when_obstacle_contract_is_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="missing_obstacle_contract")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_5_slope_theta_inference_binding"
    assert summary["obstacle_contract_mismatch_count"] > 0


def test_stage23_5_routes_binding_repair_when_pre_strong_keys_are_duplicate(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="duplicate_pre_key")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_5_slope_theta_inference_binding"
    assert summary["pre_duplicate_strong_key_count"] > 0
    assert "slope_theta_pre_duplicate_strong_keys" in summary["blocking_reason_codes"]


def test_stage23_5_rejects_stage23_4_slope_batch_regression(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path, slope_provenance_array_length_mismatch_count=1)
    _patch_stage21_5(monkeypatch, s23)
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage23_5_required_inputs"
    assert "stage23_4_slope_provenance_array_length_mismatch_count_nonzero" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def test_stage23_5_routes_safety_repair_on_hard_risk_regression(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="hard_risk_regression")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_5_slope_theta_eval_safety_regression"
    assert summary["hard_risk_violation_count"] == 1


def test_stage23_5_routes_signal_repair_when_viewpoint_and_probabilities_do_not_change(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="unchanged")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_slope_theta_policy_update_signal_strength"
    assert summary["selected_viewpoint_changed_count"] == 0
    assert summary["mean_abs_probability_delta"] == 0.0


def test_stage23_5_routes_inputs_when_stage21_5_execution_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="execution_short")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage23_5_required_inputs"
    assert summary["stage21_5_execution_incomplete"] is True
    assert summary["action_probability_audit_is_partial_diagnostic"] is True
    assert "stage21_5_pre_scenario_count_short" in summary["blocking_reason_codes"]


def test_stage23_5_routes_credit_repair_when_viewpoint_changes_without_coverage_lift(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23, mode="changed_no_improvement")
    config = _write_config(tmp_path, stage23_4)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_slope_theta_credit_assignment"
    assert summary["selected_viewpoint_changed_count"] == 1
    assert summary["final_coverage_delta"] == 0.0


def test_stage23_5_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke as s23

    stage23_4 = _write_stage23_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s23)
    config = _write_config(tmp_path, stage23_4, publishes_checkpoint=True)

    summary = s23.run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage23_5_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def _patch_stage21_5(monkeypatch, s23, *, mode: str = "changed_and_improved") -> None:
    def fake_stage21_5(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["coverage_source"] == s23.SLOPE_COVERAGE_SOURCE
        output_root.mkdir(parents=True, exist_ok=True)
        pre_root = output_root / "pre_ppo_xunce"
        post_root = output_root / "post_ppo_xunce"
        pre_root.mkdir(parents=True, exist_ok=True)
        post_root.mkdir(parents=True, exist_ok=True)
        if mode == "duplicate_pre_key":
            pre_rows = [_inference_row(0, 0), _inference_row(0, 0)]
        else:
            pre_rows = [_inference_row(0, 0), _inference_row(1, 45)]
        _write_jsonl(pre_root / s23.MODEL_INFERENCE_FILE, pre_rows)
        if mode == "missing_slope_field":
            post_rows = [_inference_row(0, 45, omit_slope=True), _inference_row(1, 45, omit_slope=True)]
        elif mode == "old_theta_source":
            post_rows = [_inference_row(0, 45, coverage_source="theta_aware_sensor_footprint/v1"), _inference_row(1, 45)]
        elif mode == "missing_obstacle_contract":
            post_rows = [_inference_row(0, 45, omit_obstacle_contract=True), _inference_row(1, 45)]
        elif mode == "unchanged":
            post_rows = [_inference_row(0, 0), _inference_row(1, 45)]
        else:
            post_rows = [_inference_row(0, 45, probs=[0.3, 0.7]), _inference_row(1, 45, probs=[0.4, 0.6])]
        _write_jsonl(post_root / s23.MODEL_INFERENCE_FILE, post_rows)
        improved = mode == "changed_and_improved"
        delta = {
            "final_coverage_rate_mean_delta": 0.02 if improved else 0.0,
            "coverage_curve_auc_mean_delta": 0.03 if improved else 0.0,
            "path_cost_total_m_mean_delta": 1.0,
        }
        (output_root / s23.stage21_5.DELTA_FILE).write_text(json.dumps(delta), encoding="utf-8")
        summary = {
            "schema_version": "xunce-stage21-5-post-update-evaluation-summary/v1",
            "status": "failed" if mode in {"hard_risk_regression", "execution_short"} else "passed",
            "next_required_change": (
                "repair_stage21_5_hard_risk_boundary"
                if mode == "hard_risk_regression"
                else s23.stage21_5.ROUTE_EXECUTION
                if mode == "execution_short"
                else "implement_stage21_6_multi_seed_ppo_pilot"
            ),
            "pre_evaluation_root": str(pre_root),
            "post_evaluation_root": str(post_root),
            "final_coverage_rate_delta": delta["final_coverage_rate_mean_delta"],
            "coverage_curve_auc_delta": delta["coverage_curve_auc_mean_delta"],
            "path_cost_total_m_delta": delta["path_cost_total_m_mean_delta"],
            "post_hard_risk_violation_count": 1 if mode == "hard_risk_regression" else 0,
            "post_mask_violation_count": 0,
            "post_path_planning_failure_count": 0,
            "post_open_grid_fallback_count": 0,
            "scenario_regression_count": 1 if mode == "scenario_regression" else 0,
            "reason_codes": (
                [
                    "pre_scenario_count_short",
                    "post_scenario_count_short",
                    "pre_xunce_episode_count_short",
                    "post_xunce_episode_count_short",
                ]
                if mode == "execution_short"
                else []
            ),
            "sample_count_too_low_for_performance_claim": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        (output_root / s23.stage21_5.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s23.stage21_5, "run_xunce_stage21_5_post_update_offline_trajectory_evaluation", fake_stage21_5)


def _inference_row(
    index: int,
    selected_theta: int,
    *,
    omit_slope: bool = False,
    omit_obstacle_contract: bool = False,
    coverage_source: str = "endpoint_theta_slope_obstacle_los/v1",
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
        "coverage_source": coverage_source,
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
    if omit_slope:
        row.pop("coverage_source")
        row.pop("max_traversable_slope_deg")
    if omit_obstacle_contract:
        row["obstacle_occlusion_enabled"] = False
        row["obstacle_source_kind"] = "unobstructed_theta"
        row["obstacle_aware_theta_coverage_hash"] = ""
        row["slope_obstacle_source_hash"] = ""
    return row


def _write_stage23_4_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke",
    selected_info_viewpoint_mismatch_count: int = 0,
    slope_provenance_array_length_mismatch_count: int = 0,
) -> Path:
    root = tmp_path / "stage23_4"
    stage23_3 = tmp_path / "stage23_3"
    s21_4 = root / "s21_4"
    s21_4.mkdir(parents=True, exist_ok=True)
    stage23_3.mkdir(parents=True, exist_ok=True)
    (s21_4 / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt").write_bytes(b"checkpoint")
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source")
    stage23_3_summary = {
        "schema_version": "xunce-stage23-3-summary/v1",
        "status": "passed",
        "stage23_2b_high_res_root": str(tmp_path / "high_res_roi_expansion"),
        "stage21_1_actual_source_roi_expansion_root": str(tmp_path / "high_res_roi_expansion"),
    }
    summary = {
        "schema_version": "xunce-stage23-4-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "stage21_4_root": str(s21_4),
        "stage23_3_root": str(stage23_3),
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "max_traversable_slope_deg": 30.0,
        "checkpoint_exists": True,
        "checkpoint_reload_passed": True,
        "checkpoint_boundary_passed": True,
        "stage21_3_lineage_passed": True,
        "experimental_only": True,
        "source_checkpoint_sha_consistent": True,
        "slope_theta_batch_contract_missing_count": 0,
        "xunce_batch_viewpoint_shape_mismatch_count": 0,
        "action_viewpoint_binding_mismatch_count": 0,
        "slope_provenance_binding_mismatch_count": 0,
        "selected_info_viewpoint_mismatch_count": selected_info_viewpoint_mismatch_count,
        "slope_provenance_array_length_mismatch_count": slope_provenance_array_length_mismatch_count,
        "selected_action_mask_violation_count": 0,
        "reward_fallback_used_count": 0,
        "pre_clip_grad_norm": 21.0,
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
    (stage23_3 / "xunce-stage23-3-summary.json").write_text(json.dumps(stage23_3_summary), encoding="utf-8")
    (root / "xunce-stage23-4-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "xunce-stage23-4-checkpoint-boundary-audit.json").write_text(json.dumps(checkpoint_audit), encoding="utf-8")
    return root


def _write_config(tmp_path: Path, stage23_4_root: Path, **overrides) -> Path:
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
        "schema_version": "xunce-stage23-5-slope-obstacle-aware-theta-post-update-trajectory-eval-smoke-config/v1",
        "stage_id": "xunce-stage23-5-slope-obstacle-aware-theta-post-update-trajectory-eval-smoke",
        "stage23_4_root": str(stage23_4_root),
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
        "obstacle_occlusion_enabled": True,
        "stage23_5_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage23_5_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
