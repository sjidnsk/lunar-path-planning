from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_7h_rejects_untrusted_stage26_7g_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path, route="some_other_route")
    config = _write_config(tmp_path, stage26_7g)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7h_required_inputs"
    assert "stage26_7g_route_mismatch" in summary["input_rejections"]


def test_stage26_7h_reruns_best_combo_stage26_3_only(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path)
    config = _write_config(tmp_path, stage26_7g)
    called: dict[str, Path] = {}

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        called["config_path"] = config_path
        called["output_root"] = output_root
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["stage26_2_root"] == str(stage26_7g / "u0" / "s26_2")
        assert cfg["coverage_denominator_source"] == "main_coverable_cells/v1"
        assert cfg["post_update_success_metric"] == "main_coverable_coverage_efficiency/v1"
        assert cfg["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
        assert cfg["path_cost_source"] == "hybrid_astar_pose_path/v1"
        assert cfg["synthetic_source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
        assert cfg["hybrid_astar_candidate_eval_workers"] == 4
        assert cfg["default_astar_replaced"] is False
        assert cfg["hybrid_astar_ackermann_feasible_claimed"] is False
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "failed",
            "next_required_change": "repair_stage26_3_synthetic_inference_binding",
            "synthetic_inference_required_field_missing_count": 3,
            "hybrid_path_contract_mismatch_count": 0,
            "pre_unreachable_selected_count": 0,
            "post_unreachable_selected_count": 0,
            "strong_state_join_available_count": 2,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "xunce-stage26-3-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert called["output_root"] == tmp_path / "out" / "s26_3_repaired"
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "continue_stage26_7h_eval_binding_repair"
    assert summary["best_stable_combo_id"] == "current_repro"
    assert summary["hard_risk_violation_count"] == 0
    assert summary["mask_violation_count"] == 0
    assert summary["path_planning_failure_count"] == 0
    assert summary["open_grid_fallback_count"] == 0


def test_stage26_7h_routes_pre_unreachable_baseline(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path)
    config = _write_config(tmp_path, stage26_7g)

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "failed",
            "next_required_change": "rerun_stage26_3_required_inputs",
            "synthetic_inference_required_field_missing_count": 0,
            "hybrid_path_contract_mismatch_count": 0,
            "pre_unreachable_selected_count": 2,
            "post_unreachable_selected_count": 0,
            "coverage_per_100m_delta": 0.0,
            "strong_state_join_available_count": 2,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "xunce-stage26-3-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "repair_stage26_7h_pre_policy_unreachable_baseline"
    assert summary["pre_unreachable_selected_count"] == 2


def test_stage26_7h_routes_post_unreachable_from_stage21_5_route(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path)
    config = _write_config(tmp_path, stage26_7g)

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "failed",
            "next_required_change": "rerun_stage26_3_required_inputs",
            "stage21_5_execution_incomplete": True,
            "stage21_5_next_required_change": "repair_stage21_5_post_policy_unreachable_regression",
            "synthetic_inference_required_field_missing_count": 0,
            "hybrid_path_contract_mismatch_count": 0,
            "pre_unreachable_selected_count": 0,
            "post_unreachable_selected_count": 1,
            "coverage_per_100m_delta": 0.0,
            "strong_state_join_available_count": 2,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "xunce-stage26-3-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "repair_stage26_7h_post_policy_unreachable_regression"


def test_stage26_7h_routes_multi_seed_when_efficiency_valid(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path)
    config = _write_config(tmp_path, stage26_7g)

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "synthetic_inference_required_field_missing_count": 0,
            "hybrid_path_contract_mismatch_count": 0,
            "pre_unreachable_selected_count": 0,
            "post_unreachable_selected_count": 0,
            "selected_action_changed_count": 0,
            "selected_viewpoint_changed_count": 2,
            "selected_theta_changed_count": 2,
            "stage21_5_next_required_change": "repair_stage21_5_pre_policy_unreachable_baseline",
            "main_final_coverage_delta": 0.01,
            "main_coverage_auc_delta": -0.02,
            "main_coverage_per_100m_delta": 0.0,
            "scenario_regression_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "strong_state_join_available_count": 2,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "xunce-stage26-3-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"


def test_stage26_7h_rejects_stale_stage26_3_success_route(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path)
    config = _write_config(tmp_path, stage26_7g)

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage26_4_synthetic_terrain_multi_seed_ppo_pilot",
            "post_update_success_metric": "coverage_auc_and_final/v1",
            "synthetic_inference_required_field_missing_count": 0,
            "hybrid_path_contract_mismatch_count": 0,
            "pre_unreachable_selected_count": 0,
            "post_unreachable_selected_count": 0,
            "selected_action_changed_count": 2,
            "main_final_coverage_delta": 0.01,
            "main_coverage_per_100m_delta": 1.0,
            "scenario_regression_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "strong_state_join_available_count": 2,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "xunce-stage26-3-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_7h_stage21_5_execution_contract"
    assert "stage26_3_route_mismatch" in summary["stage26_3_handoff_rejections"]
    assert "stage26_3_post_update_success_metric_mismatch" in summary["stage26_3_handoff_rejections"]


def test_stage26_7h_backfills_platform_hash_from_stage26_1_sidecar(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path)
    stage26_1_root = tmp_path / "stage26_1"
    sidecar_root = stage26_1_root / "src" / "sc"
    sidecar_root.mkdir(parents=True)
    _write_json(sidecar_root / "s26_1_000.sidecar.json", {"platform_contract_hash": "platform-from-sidecar"})
    stage26_2_summary_path = stage26_7g / "u0" / "s26_2" / "xunce-stage26-2-summary.json"
    stage26_2_summary = json.loads(stage26_2_summary_path.read_text(encoding="utf-8"))
    stage26_2_summary["stage26_1_root"] = str(stage26_1_root)
    _write_json(stage26_2_summary_path, stage26_2_summary)
    config = _write_config(tmp_path, stage26_7g)

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["platform_contract_hash"] == "platform-from-sidecar"
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "synthetic_inference_required_field_missing_count": 0,
            "hybrid_path_contract_mismatch_count": 0,
            "pre_unreachable_selected_count": 0,
            "post_unreachable_selected_count": 0,
            "selected_action_changed_count": 2,
            "main_final_coverage_delta": 0.01,
            "main_coverage_auc_delta": 0.02,
            "main_coverage_per_100m_delta": 1.0,
            "scenario_regression_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "strong_state_join_available_count": 2,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "xunce-stage26-3-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["platform_contract_hash"] == "platform-from-sidecar"


def test_stage26_7h_rejects_ambiguous_sidecar_platform_hash(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7h_repair_credit_post_update_eval_binding as s26

    stage26_7g = _make_stage26_7g_root(tmp_path)
    stage26_1_root = tmp_path / "stage26_1"
    sidecar_root = stage26_1_root / "src" / "sc"
    sidecar_root.mkdir(parents=True)
    _write_json(sidecar_root / "s26_1_000.sidecar.json", {"platform_contract_hash": "platform-a"})
    _write_json(sidecar_root / "s26_1_001.sidecar.json", {"platform_contract_hash": "platform-b"})
    stage26_2_summary_path = stage26_7g / "u0" / "s26_2" / "xunce-stage26-2-summary.json"
    stage26_2_summary = json.loads(stage26_2_summary_path.read_text(encoding="utf-8"))
    stage26_2_summary["stage26_1_root"] = str(stage26_1_root)
    _write_json(stage26_2_summary_path, stage26_2_summary)
    config = _write_config(tmp_path, stage26_7g)

    summary = s26.run_xunce_stage26_7h_repair_credit_post_update_eval_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7h_required_inputs"
    assert "stage26_1_sidecar_platform_contract_hash_ambiguous" in summary["input_rejections"]


def _make_stage26_7g_root(
    tmp_path: Path,
    *,
    route: str = "repair_stage26_7_credit_post_update_eval_binding",
) -> Path:
    root = tmp_path / "stage26_7g"
    u0 = root / "u0"
    s26_2 = u0 / "s26_2"
    s26_2.mkdir(parents=True)
    _write_json(
        s26_2 / "xunce-stage26-2-summary.json",
        {
            "schema_version": "xunce-stage26-2-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "synthetic_terrain_hash": "hash",
            "max_traversable_slope_deg": 30.0,
        },
    )
    _write_json(
        root / "xunce-stage26-7g-summary.json",
        {
            "schema_version": "xunce-stage26-7g-summary/v1",
            "stage_id": "xunce-stage26-7g-repair-behavior-policy-kl-baseline",
            "status": "failed",
            "next_required_change": route,
            "best_stable_combo_id": "current_repro",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "max_traversable_slope_deg": 30.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    _write_jsonl(
        root / "xunce-stage26-7g-update-sweep-results.jsonl",
        [
            {
                "combo_id": "current_repro",
                "work_dir": "u0",
                "stage26_2_status": "passed",
                "stage26_3_status": "failed",
                "stage26_3_next_required_change": "rerun_stage26_3_required_inputs",
                "checkpoint_reload_passed": True,
                "experimental_only": True,
                "final_post_update_policy_approx_kl": 0.1,
                "parameter_delta_l2": 0.1,
            }
        ],
    )
    return root


def _write_config(tmp_path: Path, stage26_7g_root: Path) -> Path:
    stage26_3_base = tmp_path / "stage26_3_base.json"
    stage21_5_base = tmp_path / "stage21_5_base.json"
    high_fidelity = tmp_path / "high_fidelity.json"
    _write_json(
        stage26_3_base,
        {
            "schema_version": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke-config/v1",
            "stage_id": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke",
            "stage26_2_root": "placeholder",
            "stage21_5_base_config": str(stage21_5_base),
            "high_fidelity_config": str(high_fidelity),
        },
    )
    _write_json(stage21_5_base, {"schema_version": "xunce-stage21-5-post-update-offline-trajectory-evaluation-config/v1"})
    _write_json(high_fidelity, {"schema_version": "xunce-high-fidelity-exploration-coverage-comparison-config/v1"})
    config = tmp_path / "stage26_7h.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage26-7h-repair-credit-post-update-eval-binding-config/v1",
            "stage_id": "xunce-stage26-7h-repair-credit-post-update-eval-binding",
            "stage26_7g_root": str(stage26_7g_root),
            "stage26_3_base_config": str(stage26_3_base),
            "stage21_5_base_config": str(stage21_5_base),
            "high_fidelity_config": str(high_fidelity),
            "required_scenario_count": 2,
            "eval_rollout_steps": 4,
            "hybrid_astar_candidate_eval_workers": 4,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
