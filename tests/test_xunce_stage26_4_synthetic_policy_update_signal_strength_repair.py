from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "model-explorer" / "src"):
    value = str(_path)
    if value not in sys.path:
        sys.path.insert(0, value)


SYNTHETIC_HASH = "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85"


def test_stage26_4_routes_collector_sample_expansion_when_transition_count_short(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    config = _write_config(tmp_path)
    calls = _patch_stage_chain(monkeypatch, s26, trainable_count=8)

    summary = s26.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_COLLECTOR
    assert summary["trainable_transition_count"] == 8
    assert "stage26_1_trainable_transition_count_below_minimum" in summary["blocking_reason_codes"]
    assert calls == ["stage26_1"]


def test_stage26_4_routes_value_balance_when_value_loss_still_dominates(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    config = _write_config(tmp_path)
    calls = _patch_stage_chain(monkeypatch, s26, value_to_policy_ratio=100.0, probability_delta=0.0)

    summary = s26.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_VALUE_BALANCE
    assert summary["stable_combo_count"] == 3
    assert summary["min_value_to_policy_grad_norm_ratio"] == 100.0
    assert "value_loss_grad_norm_ratio_above_threshold" in summary["blocking_reason_codes"]
    assert calls.count("stage26_2") == 3
    assert calls.count("stage26_3") == 3


def test_stage26_4_routes_strength_when_probability_delta_still_tiny(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    config = _write_config(tmp_path)
    _patch_stage_chain(monkeypatch, s26, value_to_policy_ratio=0.2, probability_delta=1.0e-7)

    summary = s26.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_STRENGTH
    assert summary["max_mean_abs_probability_delta"] == 1.0e-7
    assert "synthetic_probability_delta_below_threshold" in summary["blocking_reason_codes"]


def test_stage26_4_routes_margin_when_probability_moves_but_viewpoint_does_not(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    config = _write_config(tmp_path)
    _patch_stage_chain(monkeypatch, s26, value_to_policy_ratio=0.2, probability_delta=2.0e-5)

    summary = s26.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_MARGIN
    assert summary["max_mean_abs_probability_delta"] == 2.0e-5
    assert summary["max_selected_viewpoint_changed_count"] == 0


def test_stage26_4_runs_independent_combo_roots_and_routes_stage26_5_when_metrics_improve(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    config = _write_config(tmp_path)
    _patch_stage_chain(
        monkeypatch,
        s26,
        value_to_policy_ratio=0.2,
        probability_delta=3.0e-5,
        viewpoint_changed=1,
        coverage_delta=0.02,
        auc_delta=0.03,
        path_cost_delta=-1.0,
    )

    summary = s26.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == s26.ROUTE_STAGE26_5
    assert summary["best_combo_id"] == "baseline_current"
    assert (tmp_path / "out" / "c" / "c1" / "s2").is_dir()
    assert (tmp_path / "out" / "c" / "c2" / "s2").is_dir()
    assert (tmp_path / "out" / "c" / "c3" / "s2").is_dir()
    recommended_2 = json.loads((tmp_path / "out" / s26.RECOMMENDED_26_2_CONFIG).read_text(encoding="utf-8"))
    recommended_3 = json.loads((tmp_path / "out" / s26.RECOMMENDED_26_3_CONFIG).read_text(encoding="utf-8"))
    action_audit = json.loads((tmp_path / "out" / s26.ACTION_AUDIT_FILE).read_text(encoding="utf-8"))
    assert recommended_2["stage26_1_root"] == str(tmp_path / "out" / "s26_1")
    assert recommended_2["publishes_checkpoint"] is False
    assert recommended_3["stage26_2_root"] == str(tmp_path / "out" / "c" / "c1" / "s2")
    assert recommended_3["xunce_only_evaluation"] is True
    assert action_audit["min_selected_vs_best_probability_margin"] is not None
    assert action_audit["max_selected_vs_best_rank_gap"] is not None


def test_stage26_4_does_not_route_success_by_mixing_metrics_across_combos() -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    rows = [
        _stable_sweep_row(
            combo_id="coverage_only",
            final_coverage_delta=0.02,
            coverage_auc_delta=0.03,
            hybrid_astar_path_cost_delta=1.0,
            selected_viewpoint_changed_count=1,
        ),
        _stable_sweep_row(
            combo_id="cost_only",
            final_coverage_delta=0.0,
            coverage_auc_delta=0.0,
            hybrid_astar_path_cost_delta=-2.0,
            selected_viewpoint_changed_count=1,
        ),
    ]
    gradient = s26._policy_value_gradient_audit(rows)
    action = s26._action_signal_audit(rows)
    credit = {"advantage_gap": 1.0}

    status, route, _ = s26._route(
        boundary_reasons=[],
        input_reasons=[],
        collector_summary={
            "status": "passed",
            "next_required_change": s26.stage26_1.ROUTE_STAGE26_2,
            "trainable_transition_count": 16,
        },
        sweep_rows=rows,
        gradient_audit=gradient,
        action_audit=action,
        credit_audit=credit,
        min_transitions=16,
        min_probability_delta=1.0e-5,
        value_to_policy_threshold=10.0,
    )

    assert status == "failed"
    assert route == s26.ROUTE_CREDIT


def test_stage26_4_treats_stage26_3_binding_failures_as_lineage_blocker() -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    rows = [_stable_sweep_row(physical_obstacle_payload_count=1)]

    assert s26._synthetic_lineage_failed(rows) is True
    assert s26._stable_rows(rows) == []


def test_stage26_4_rejects_stage26_3_with_unexpected_route(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    config = _write_config(tmp_path, stage26_3_route="repair_stage26_synthetic_credit_assignment")
    calls = _patch_stage_chain(monkeypatch, s26)

    summary = s26.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_INPUTS
    assert "stage26_3_route_not_synthetic_signal_repair" in summary["blocking_reason_codes"]
    assert calls == []


def test_stage26_4_uses_short_combo_roots_for_stage21_4_checkpoint_paths() -> None:
    import scripts.run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair as s26

    default_output_root = Path(
        "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/"
        "path_feedback_batch_xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1"
    )
    combo = {"combo_id": "baseline_current"}
    combo_root = default_output_root / "c" / s26._combo_work_id(combo, 1)
    checkpoint_path = combo_root / "s2" / "s21_4" / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt"

    assert s26._combo_work_id(combo, 1) == "c1"
    assert len(str(checkpoint_path)) < 260


def _write_config(tmp_path: Path, *, stage26_3_route: str = "repair_stage26_synthetic_policy_update_signal_strength") -> Path:
    stage26_3_root = tmp_path / "stage26_3"
    stage26_1_root = tmp_path / "stage26_1_prior"
    stage26_1_root.mkdir()
    stage26_3_root.mkdir()
    _write_json(
        stage26_1_root / "xunce-stage26-1-summary.json",
        {
            "schema_version": "xunce-stage26-1-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
            "stage26_0_root": str(tmp_path / "stage26_0"),
            "synthetic_terrain_hash": SYNTHETIC_HASH,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        },
    )
    _write_json(
        stage26_3_root / "xunce-stage26-3-summary.json",
        {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "failed",
            "next_required_change": stage26_3_route,
            "action_probability_audit_is_partial_diagnostic": False,
            "strong_state_join_available_count": 8,
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_terrain_hash": SYNTHETIC_HASH,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "max_traversable_slope_deg": 30.0,
            "synthetic_contract_mismatch_count": 0,
            "physical_obstacle_payload_count": 0,
            "grid_fallback_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "stage26_1_root": str(stage26_1_root),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        },
    )
    base_26_1 = tmp_path / "base_26_1.json"
    base_26_2 = tmp_path / "base_26_2.json"
    base_26_3 = tmp_path / "base_26_3.json"
    _write_json(base_26_1, {"schema_version": "xunce-stage26-1-synthetic-terrain-collector-smoke-config/v1"})
    _write_json(base_26_2, {"schema_version": "xunce-stage26-2-synthetic-terrain-ppo-update-smoke-config/v1"})
    _write_json(base_26_3, {"schema_version": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke-config/v1"})
    config = tmp_path / "stage26_4_config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage26-4-synthetic-policy-update-signal-strength-repair-config/v1",
            "stage26_3_root": str(stage26_3_root),
            "stage26_1_base_config": str(base_26_1),
            "stage26_2_base_config": str(base_26_2),
            "stage26_3_base_config": str(base_26_3),
            "min_trainable_transition_count": 16,
            "required_scenario_count": 2,
            "rollout_steps": 8,
            "eval_rollout_steps": 4,
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
    s26,
    *,
    trainable_count: int = 16,
    value_to_policy_ratio: float = 0.2,
    probability_delta: float = 2.0e-5,
    viewpoint_changed: int = 0,
    coverage_delta: float = 0.0,
    auc_delta: float = 0.0,
    path_cost_delta: float = 0.0,
) -> list[str]:
    calls: list[str] = []

    def fake_stage26_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage26_1")
        output_root.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["rollout_steps"] == 8
        assert cfg["min_trainable_transition_count"] == 16
        _write_jsonl(output_root / "s21_3" / s26.stage26_2.stage21_3.BATCH_FILE, [{"advantage": -1.0}, {"advantage": 2.0}])
        _write_jsonl(
            output_root / "s21_2" / s26.stage26_1.stage21_2.EVALUATION_FILE,
            [{"reward": 1.0, "coverage_gain_per_path_cost": 0.1}, {"reward": 3.0, "coverage_gain_per_path_cost": 0.5}],
        )
        return {
            "schema_version": "xunce-stage26-1-summary/v1",
            "status": "passed",
            "next_required_change": s26.stage26_1.ROUTE_STAGE26_2,
            "transition_count": trainable_count,
            "batch_row_count": trainable_count,
            "synthetic_terrain_hash": SYNTHETIC_HASH,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
        }

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage26_2")
        output_root.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        policy_grad = 1.0
        value_grad = policy_grad * value_to_policy_ratio
        _write_json(
            output_root / s26.stage26_2.LOSS_GRADIENT_AUDIT_FILE,
            {
                "schema_version": "xunce-stage26-2-loss-gradient-audit/v1",
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
            "schema_version": "xunce-stage26-2-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke",
            "batch_row_count": trainable_count,
            "loss_finite": True,
            "gradient_finite": True,
            "checkpoint_reload_passed": True,
            "checkpoint_boundary_passed": True,
            "final_post_update_approx_kl": 0.001,
            "pre_clip_grad_norm": policy_grad + value_grad,
            "post_clip_grad_norm": 1.0,
            "parameter_delta_l2": 0.1,
            "synthetic_terrain_hash": SYNTHETIC_HASH,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "stage26_1_root": cfg["stage26_1_root"],
        }

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage26_3")
        output_root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            output_root / "post" / s26.stage26_3.MODEL_INFERENCE_FILE,
            [
                {
                    "policy": "xunce",
                    "selected_action_index": 0,
                    "action_probs": [0.6, 0.3, 0.1],
                    "action_mask": [True, True, True],
                    "theta_coverage_gain_per_path_costs": [1.0, 3.0, 2.0],
                }
            ],
        )
        return {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "failed" if coverage_delta <= 0.0 or auc_delta <= 0.0 or path_cost_delta >= 0.0 else "passed",
            "next_required_change": "run_stage26_5_synthetic_terrain_multi_seed_ppo_pilot"
            if coverage_delta > 0.0 and auc_delta > 0.0 and path_cost_delta < 0.0
            else "repair_stage26_synthetic_policy_update_signal_strength",
            "strong_state_join_available_count": 16,
            "mean_abs_probability_delta": probability_delta,
            "selected_action_probability_delta_mean": probability_delta / 2.0,
            "selected_action_probability_delta": probability_delta / 2.0,
            "selected_viewpoint_changed_count": viewpoint_changed,
            "selected_theta_changed_count": viewpoint_changed,
            "selected_action_changed_count": viewpoint_changed,
            "final_coverage_delta": coverage_delta,
            "coverage_auc_delta": auc_delta,
            "hybrid_astar_path_cost_delta": path_cost_delta,
            "coverage_per_100m_delta": 0.0,
            "synthetic_contract_mismatch_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "synthetic_terrain_hash": SYNTHETIC_HASH,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }

    monkeypatch.setattr(s26.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fake_stage26_1)
    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)
    return calls


def _stable_sweep_row(**overrides) -> dict:
    row = {
        "combo_id": "combo",
        "stage26_2_status": "passed",
        "stage26_3_status": "failed",
        "stage26_3_next_required_change": "calibrate_stage26_synthetic_discrete_margin_crossing",
        "batch_row_count": 16,
        "synthetic_contract_mismatch_count": 0,
        "synthetic_inference_required_field_missing_count": 0,
        "physical_obstacle_payload_count": 0,
        "grid_fallback_count": 0,
        "default_astar_replaced_count": 0,
        "ackermann_feasible_claimed_count": 0,
        "action_probability_audit_is_partial_diagnostic": False,
        "loss_finite": True,
        "gradient_finite": True,
        "checkpoint_reload_passed": True,
        "checkpoint_boundary_passed": True,
        "release_boundary_clean": True,
        "final_post_update_approx_kl": 0.001,
        "pre_clip_grad_norm": 1.0,
        "post_clip_grad_norm": 1.0,
        "parameter_delta_l2": 0.1,
        "policy_loss_grad_norm": 1.0,
        "value_loss_grad_norm": 0.2,
        "entropy_loss_grad_norm": 0.01,
        "value_to_policy_grad_norm_ratio": 0.2,
        "strong_state_join_available_count": 8,
        "mean_abs_probability_delta": 2.0e-5,
        "selected_action_probability_delta_mean": 1.0e-5,
        "selected_viewpoint_changed_count": 0,
        "selected_theta_changed_count": 0,
        "selected_action_changed_count": 0,
        "final_coverage_delta": 0.0,
        "coverage_auc_delta": 0.0,
        "hybrid_astar_path_cost_delta": 0.0,
        "hard_risk_violation_count": 0,
        "mask_violation_count": 0,
        "path_planning_failure_count": 0,
        "open_grid_fallback_count": 0,
        "synthetic_terrain_hash": SYNTHETIC_HASH,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }
    row.update(overrides)
    return row


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
