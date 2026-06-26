from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "model-explorer" / "src"):
    value = str(_path)
    if value not in sys.path:
        sys.path.insert(0, value)


def test_stage23_6_routes_collector_sample_expansion_when_transition_count_short(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair as s23

    config = _write_config(tmp_path)
    calls = _patch_stage_chain(monkeypatch, s23, trainable_count=4)

    summary = s23.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_COLLECTOR
    assert summary["trainable_transition_count"] == 4
    assert "stage23_3_trainable_transition_count_below_minimum" in summary["blocking_reason_codes"]
    assert calls == ["stage23_3"]


def test_stage23_6_routes_value_balance_when_value_loss_still_dominates(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair as s23

    config = _write_config(tmp_path)
    calls = _patch_stage_chain(monkeypatch, s23, value_to_policy_ratio=100.0, probability_delta=0.0)

    summary = s23.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_VALUE_BALANCE
    assert summary["stable_combo_count"] == 3
    assert summary["min_value_to_policy_grad_norm_ratio"] == 100.0
    assert "value_loss_grad_norm_ratio_above_threshold" in summary["blocking_reason_codes"]
    assert calls.count("stage23_4") == 3
    assert calls.count("stage23_5") == 3


def test_stage23_6_routes_strength_when_probability_delta_still_tiny(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair as s23

    config = _write_config(tmp_path)
    _patch_stage_chain(monkeypatch, s23, value_to_policy_ratio=0.2, probability_delta=1.0e-7)

    summary = s23.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_STRENGTH
    assert summary["max_mean_abs_probability_delta"] == 1.0e-7
    assert "slope_theta_probability_delta_below_threshold" in summary["blocking_reason_codes"]


def test_stage23_6_routes_margin_when_probability_moves_but_viewpoint_does_not(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair as s23

    config = _write_config(tmp_path)
    _patch_stage_chain(monkeypatch, s23, value_to_policy_ratio=0.2, probability_delta=2.0e-5)

    summary = s23.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_MARGIN
    assert summary["max_mean_abs_probability_delta"] == 2.0e-5
    assert summary["max_selected_viewpoint_changed_count"] == 0


def test_stage23_6_runs_independent_combo_roots_and_routes_stage23_7_when_coverage_improves(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair as s23

    config = _write_config(tmp_path)
    _patch_stage_chain(
        monkeypatch,
        s23,
        value_to_policy_ratio=0.2,
        probability_delta=3.0e-5,
        viewpoint_changed=1,
        coverage_delta=0.02,
        auc_delta=0.03,
    )

    summary = s23.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == s23.ROUTE_STAGE23_7
    assert summary["best_combo_id"] == "baseline_current"
    assert (tmp_path / "out" / "c" / "c1" / "s4").is_dir()
    assert (tmp_path / "out" / "c" / "c2" / "s4").is_dir()
    assert (tmp_path / "out" / "c" / "c3" / "s4").is_dir()
    recommended_4 = json.loads((tmp_path / "out" / s23.RECOMMENDED_23_4_CONFIG).read_text(encoding="utf-8"))
    assert recommended_4["stage23_3_root"] == str(tmp_path / "out" / "s23_3")
    assert recommended_4["publishes_checkpoint"] is False
    assert recommended_4["replaces_default_policy"] is False


def test_stage23_6_uses_short_combo_roots_for_stage21_4_checkpoint_paths() -> None:
    import scripts.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair as s23

    default_output_root = Path(
        "D:/CodexDownloads/lunar-path-planning/stage23_slope_obstacle_aware_theta_reward/outputs/"
        "path_feedback_batch_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair_v1"
    )
    combo = {"combo_id": "baseline_current"}
    combo_root = default_output_root / "c" / s23._combo_work_id(combo, 1)
    checkpoint_path = combo_root / "s4" / "s21_4" / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt"

    assert s23._combo_work_id(combo, 1) == "c1"
    assert len(str(checkpoint_path)) < 260


def test_stage23_6_rejects_stage23_5a_that_did_not_clear_required_inputs(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair as s23

    config = _write_config(tmp_path, stage23_5a_route="rerun_stage23_5_required_inputs")
    calls = _patch_stage_chain(monkeypatch, s23)

    summary = s23.run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_INPUTS
    assert "stage23_5a_route_not_slope_theta_signal_repair" in summary["blocking_reason_codes"]
    assert calls == []


def _write_config(tmp_path: Path, *, stage23_5a_route: str = "repair_stage23_slope_theta_policy_update_signal_strength") -> Path:
    stage23_5a_root = tmp_path / "stage23_5a"
    stage23_5a_root.mkdir()
    _write_json(
        stage23_5a_root / "xunce-stage23-5a-summary.json",
        {
            "schema_version": "xunce-stage23-5a-summary/v1",
            "status": "failed",
            "next_required_change": stage23_5a_route,
            "prior_stage23_5_required_input_short_blocker": True,
            "repaired_high_res_roi_expansion_slice_count": 2,
            "repaired_high_res_roi_expansion_scenario_count": 2,
            "stage23_5_action_probability_audit_is_partial_diagnostic": False,
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "max_traversable_slope_deg": 30.0,
            "platform_contract_hash": "platform-hash",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        },
    )
    r2 = stage23_5a_root / "r2"
    r2.mkdir()
    _write_json(
        r2 / "xunce-stage23-2-summary.json",
        {
            "schema_version": "xunce-stage23-2-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke",
        },
    )
    base_23_3 = tmp_path / "base_23_3.json"
    base_23_4 = tmp_path / "base_23_4.json"
    base_23_5 = tmp_path / "base_23_5.json"
    _write_json(base_23_3, {"schema_version": "xunce-stage23-3-slope-obstacle-aware-theta-ppo-collector-smoke-config/v1"})
    _write_json(base_23_4, {"schema_version": "xunce-stage23-4-slope-obstacle-aware-theta-ppo-update-smoke-config/v1"})
    _write_json(base_23_5, {"schema_version": "xunce-stage23-5-slope-obstacle-aware-theta-post-update-trajectory-eval-smoke-config/v1"})
    config = tmp_path / "stage23_6_config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage23-6-slope-theta-policy-update-signal-strength-repair-config/v1",
            "stage23_5a_root": str(stage23_5a_root),
            "stage23_3_base_config": str(base_23_3),
            "stage23_4_base_config": str(base_23_4),
            "stage23_5_base_config": str(base_23_5),
            "min_trainable_transition_count": 16,
            "required_scenario_count": 2,
            "rollout_steps": 8,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return config


def _patch_stage_chain(
    monkeypatch,
    s23,
    *,
    trainable_count: int = 16,
    value_to_policy_ratio: float = 0.2,
    probability_delta: float = 2.0e-5,
    viewpoint_changed: int = 0,
    coverage_delta: float = 0.0,
    auc_delta: float = 0.0,
) -> list[str]:
    calls: list[str] = []

    def fake_stage23_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_3")
        output_root.mkdir(parents=True, exist_ok=True)
        return {
            "schema_version": "xunce-stage23-3-summary/v1",
            "status": "passed",
            "next_required_change": s23.stage23_3.ROUTE_STAGE23_4,
            "trainable_transition_count": trainable_count,
            "platform_contract_hash": "platform-hash",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "max_traversable_slope_deg": 30.0,
        }

    def fake_stage23_4(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_4")
        output_root.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        policy_grad = 1.0
        value_grad = policy_grad * value_to_policy_ratio
        _write_json(
            output_root / s23.stage23_4.LOSS_GRADIENT_AUDIT_FILE,
            {
                "schema_version": "xunce-stage23-4-loss-gradient-audit/v1",
                "loss_finite": True,
                "gradient_finite": True,
                "final_post_update_approx_kl": 0.001,
                "pre_clip_grad_norm": policy_grad + value_grad,
                "post_clip_grad_norm": 1.0,
                "parameter_delta_l2": 0.1,
                "component_grad_norms": {
                    "policy_loss_grad_norm": policy_grad,
                    "value_loss_grad_norm": value_grad,
                    "entropy_loss_grad_norm": 0.01,
                    "total_loss_grad_norm": policy_grad + value_grad,
                },
            },
        )
        return {
            "schema_version": "xunce-stage23-4-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke",
            "loss_finite": True,
            "gradient_finite": True,
            "checkpoint_reload_passed": True,
            "checkpoint_boundary_passed": True,
            "final_post_update_approx_kl": 0.001,
            "pre_clip_grad_norm": policy_grad + value_grad,
            "post_clip_grad_norm": 1.0,
            "parameter_delta_l2": 0.1,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "stage23_3_root": cfg["stage23_3_root"],
        }

    def fake_stage23_5(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_5")
        output_root.mkdir(parents=True, exist_ok=True)
        return {
            "schema_version": "xunce-stage23-5-summary/v1",
            "status": "failed" if coverage_delta <= 0.0 or auc_delta <= 0.0 else "passed",
            "next_required_change": "run_stage23_7_slope_obstacle_aware_theta_multi_seed_ppo_pilot"
            if coverage_delta > 0.0 and auc_delta > 0.0
            else "repair_stage23_slope_theta_policy_update_signal_strength",
            "strong_state_join_available_count": 16,
            "mean_abs_probability_delta": probability_delta,
            "selected_action_probability_delta_mean": probability_delta / 2.0,
            "selected_viewpoint_changed_count": viewpoint_changed,
            "selected_theta_changed_count": viewpoint_changed,
            "selected_action_changed_count": viewpoint_changed,
            "final_coverage_delta": coverage_delta,
            "coverage_auc_delta": auc_delta,
            "path_cost_delta": 0.0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }

    monkeypatch.setattr(s23.stage23_3, "run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke", fake_stage23_3)
    monkeypatch.setattr(s23.stage23_4, "run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke", fake_stage23_4)
    monkeypatch.setattr(s23.stage23_5, "run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke", fake_stage23_5)
    return calls


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
