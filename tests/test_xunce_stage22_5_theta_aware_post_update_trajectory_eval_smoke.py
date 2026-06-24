import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage22_5_runs_theta_post_update_eval_and_routes_stage22_6(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke as s22

    stage22_4 = _write_stage22_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s22, mode="changed_and_improved")
    config = _write_config(tmp_path, stage22_4)

    summary = s22.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage21_5_config = json.loads((tmp_path / "out" / "xunce-stage22-5-stage21-5-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage22_6_theta_aware_multi_seed_ppo_pilot"
    assert summary["strong_state_join_available_count"] == 2
    assert summary["selected_viewpoint_changed_count"] == 1
    assert summary["selected_theta_changed_count"] == 1
    assert summary["mean_abs_probability_delta"] > 0
    assert summary["final_coverage_delta"] > 0
    assert summary["coverage_auc_delta"] > 0
    assert summary["publishes_checkpoint"] is False
    assert stage21_5_config["stage21_4_tiny_ppo_update_smoke_root"] == str(stage22_4 / "s21_4")
    assert stage21_5_config["required_scenario_count"] == 2
    assert stage21_5_config["rollout_steps"] == 4
    assert stage21_5_config["theta_aware_candidate_viewpoints_enabled"] is True


def test_stage22_5_rejects_unready_stage22_4_without_running_stage21_5(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke as s22

    stage22_4 = _write_stage22_4_root(tmp_path, next_required_change="repair_stage22_4_theta_ppo_update_stability")
    _patch_stage21_5(monkeypatch, s22)
    config = _write_config(tmp_path, stage22_4)

    summary = s22.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage22_5_required_inputs"
    assert "stage22_4_route_not_stage22_5" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def test_stage22_5_routes_binding_repair_when_theta_inference_fields_are_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke as s22

    stage22_4 = _write_stage22_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s22, mode="missing_theta_field")
    config = _write_config(tmp_path, stage22_4)

    summary = s22.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_5_theta_inference_binding"
    assert summary["theta_inference_required_field_missing_count"] > 0


def test_stage22_5_routes_safety_repair_on_hard_risk_regression(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke as s22

    stage22_4 = _write_stage22_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s22, mode="hard_risk_regression")
    config = _write_config(tmp_path, stage22_4)

    summary = s22.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_5_theta_eval_safety_regression"
    assert summary["hard_risk_violation_count"] == 1


def test_stage22_5_does_not_pass_when_any_scenario_regresses(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke as s22

    stage22_4 = _write_stage22_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s22, mode="scenario_regression")
    config = _write_config(tmp_path, stage22_4)

    summary = s22.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_theta_credit_assignment"
    assert summary["scenario_regression_count"] == 1
    assert "theta_post_update_scenario_regression_detected" in summary["blocking_reason_codes"]


def test_stage22_5_routes_signal_repair_when_viewpoint_and_probabilities_do_not_change(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke as s22

    stage22_4 = _write_stage22_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s22, mode="unchanged")
    config = _write_config(tmp_path, stage22_4)

    summary = s22.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_theta_policy_update_signal_strength"
    assert summary["selected_viewpoint_changed_count"] == 0
    assert summary["mean_abs_probability_delta"] == 0.0


def test_stage22_5_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke as s22

    stage22_4 = _write_stage22_4_root(tmp_path)
    _patch_stage21_5(monkeypatch, s22)
    config = _write_config(tmp_path, stage22_4, publishes_checkpoint=True)

    summary = s22.run_xunce_stage22_5_theta_aware_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage22_5_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def _patch_stage21_5(monkeypatch, s22, *, mode: str = "changed_and_improved") -> None:
    def fake_stage21_5(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        pre_root = output_root / "pre_ppo_xunce"
        post_root = output_root / "post_ppo_xunce"
        pre_root.mkdir(parents=True, exist_ok=True)
        post_root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(pre_root / s22.MODEL_INFERENCE_FILE, [_inference_row(0, 0), _inference_row(1, 45)])
        if mode == "missing_theta_field":
            post_rows = [_inference_row(0, 45, omit_theta=True), _inference_row(1, 45, omit_theta=True)]
        elif mode == "unchanged":
            post_rows = [_inference_row(0, 0), _inference_row(1, 45)]
        else:
            post_rows = [_inference_row(0, 45, probs=[0.3, 0.7]), _inference_row(1, 45, probs=[0.4, 0.6])]
        _write_jsonl(post_root / s22.MODEL_INFERENCE_FILE, post_rows)
        delta = {
            "final_coverage_rate_delta": 0.0 if mode != "changed_and_improved" else 0.02,
            "coverage_curve_auc_delta": 0.0 if mode != "changed_and_improved" else 0.03,
            "path_cost_total_m_delta": 1.0,
        }
        (output_root / s22.stage21_5.DELTA_FILE).write_text(json.dumps(delta), encoding="utf-8")
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
            "final_coverage_rate_delta": delta["final_coverage_rate_delta"],
            "coverage_curve_auc_delta": delta["coverage_curve_auc_delta"],
            "path_cost_total_m_delta": delta["path_cost_total_m_delta"],
            "post_hard_risk_violation_count": 1 if mode == "hard_risk_regression" else 0,
            "post_mask_violation_count": 0,
            "post_path_planning_failure_count": 0,
            "post_open_grid_fallback_count": 0,
            "scenario_regression_count": 1 if mode == "scenario_regression" else 0,
            "sample_count_too_low_for_performance_claim": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        (output_root / s22.stage21_5.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s22.stage21_5, "run_xunce_stage21_5_post_update_offline_trajectory_evaluation", fake_stage21_5)


def _inference_row(index: int, selected_theta: int, *, omit_theta: bool = False, probs: list[float] | None = None) -> dict:
    row = {
        "schema_version": "xunce-exploration-coverage-model-inference/v1",
        "scenario_id": f"scenario-{index}",
        "step_index": index,
        "current_cell": [index, 0],
        "covered_cells_hash": f"covered-{index}",
        "candidate_set_hash": f"candidate-set-{index}",
        "candidate_viewpoints": [[index, 1, 0], [index, 1, 45]],
        "candidate_theta_deg": [0, 45],
        "selected_action_index": 0 if selected_theta == 0 else 1,
        "selected_viewpoint": [index, 1, selected_theta],
        "selected_theta_deg": selected_theta,
        "action_probs": probs or ([0.6, 0.4] if selected_theta == 0 else [0.4, 0.6]),
        "logits": [0.1, 0.2],
        "masked_logits": [0.1, 0.2],
    }
    if omit_theta:
        row.pop("selected_viewpoint")
        row.pop("selected_theta_deg")
    return row


def _write_stage22_4_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage22_5_theta_aware_post_update_trajectory_eval_smoke",
) -> Path:
    root = tmp_path / "stage22_4"
    s21_4 = root / "s21_4"
    s21_4.mkdir(parents=True, exist_ok=True)
    checkpoint = s21_4 / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source")
    summary = {
        "schema_version": "xunce-stage22-4-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "stage21_4_root": str(s21_4),
        "checkpoint_reload_passed": True,
        "checkpoint_boundary_passed": True,
        "stage21_3_lineage_passed": True,
        "experimental_only": True,
        "theta_batch_contract_missing_count": 0,
        "selected_action_mask_violation_count": 0,
        "pre_clip_grad_norm": 245.6,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    checkpoint_audit = {
        "source_checkpoint_path": str(source_checkpoint),
        "checkpoint_path": str(checkpoint),
        "checkpoint_reload_passed": True,
        "experimental_only": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    (root / "xunce-stage22-4-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "xunce-stage22-4-checkpoint-boundary-audit.json").write_text(json.dumps(checkpoint_audit), encoding="utf-8")
    return root


def _write_config(tmp_path: Path, stage22_4_root: Path, **overrides) -> Path:
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
    high_fidelity.write_text(
        json.dumps(
            {
                "schema_version": "xunce-high-fidelity-exploration-coverage-comparison-config/v1",
                "theta_aware_candidate_viewpoints_enabled": True,
                "theta_bin_count": 8,
                "sensor_fov_deg": 90.0,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "xunce-stage22-5-theta-aware-post-update-trajectory-eval-smoke-config/v1",
        "stage_id": "xunce-stage22-5-theta-aware-post-update-trajectory-eval-smoke",
        "stage22_4_root": str(stage22_4_root),
        "stage21_5_base_config": str(base_config),
        "high_fidelity_config": str(high_fidelity),
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "theta_aware_candidate_viewpoints_enabled": True,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_fov_deg": 90.0,
        "stage22_5_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage22_5_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
