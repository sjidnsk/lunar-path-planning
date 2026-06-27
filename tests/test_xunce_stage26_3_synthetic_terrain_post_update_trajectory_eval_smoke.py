import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


SYNTHETIC_HASH = "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85"


def test_stage26_3_runs_synthetic_post_update_eval_and_routes_stage26_4(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path)
    _patch_stage21_5(monkeypatch, s26, mode="changed_and_improved")
    config = _write_config(tmp_path, stage26_2)

    summary = s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage21_5_config = json.loads((tmp_path / "out" / "xunce-stage26-3-stage21-5-config.json").read_text(encoding="utf-8"))
    high_fidelity_config = json.loads((tmp_path / "out" / "xunce-stage26-3-high-fidelity-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_4_synthetic_terrain_multi_seed_ppo_pilot"
    assert summary["strong_state_join_available_count"] == 2
    assert summary["selected_viewpoint_changed_count"] == 1
    assert summary["selected_theta_changed_count"] == 1
    assert summary["mean_abs_probability_delta"] > 0
    assert summary["final_coverage_delta"] > 0
    assert summary["coverage_auc_delta"] > 0
    assert summary["hybrid_astar_path_cost_delta"] < 0
    assert summary["synthetic_contract_mismatch_count"] == 0
    assert summary["physical_obstacle_payload_count"] == 0
    assert summary["publishes_checkpoint"] is False
    assert stage21_5_config["stage21_4_tiny_ppo_update_smoke_root"] == str(stage26_2 / "s21_4")
    assert stage21_5_config["synthetic_terrain_contract_enabled"] is True
    assert stage21_5_config["synthetic_terrain_hash"] == SYNTHETIC_HASH
    assert stage21_5_config["synthetic_source_kind"] == s26.SYNTHETIC_SOURCE_KIND
    assert stage21_5_config["coverage_source"] == s26.COVERAGE_SOURCE
    assert stage21_5_config["path_cost_source"] == s26.PATH_COST_SOURCE
    assert stage21_5_config["include_oracle_baselines"] is False
    assert stage21_5_config["include_canonical_reward_rerank_oracle"] is False
    assert stage21_5_config["xunce_only_evaluation"] is True
    assert stage21_5_config["hybrid_astar_candidate_eval_workers"] == 1
    assert high_fidelity_config["synthetic_terrain_contract_enabled"] is True
    assert high_fidelity_config["synthetic_terrain_hash"] == SYNTHETIC_HASH
    assert high_fidelity_config["source_roi_expansion_root"] == str(stage26_2 / "stage26_1" / "src")
    assert high_fidelity_config["include_oracle_baselines"] is False
    assert high_fidelity_config["include_canonical_reward_rerank_oracle"] is False
    assert high_fidelity_config["xunce_only_evaluation"] is True
    assert high_fidelity_config["hybrid_astar_candidate_eval_workers"] == 1


def test_stage26_3_rejects_unready_stage26_2_without_running_stage21_5(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path, next_required_change="repair_stage26_2_checkpoint_reload_boundary")
    _patch_stage21_5(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_2)

    summary = s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_3_required_inputs"
    assert "stage26_2_route_not_stage26_3" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def test_stage26_3_routes_binding_repair_when_synthetic_fields_are_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path)
    _patch_stage21_5(monkeypatch, s26, mode="missing_synthetic_fields")
    config = _write_config(tmp_path, stage26_2)

    summary = s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_3_synthetic_inference_binding"
    assert summary["synthetic_inference_required_field_missing_count"] > 0


def test_stage26_3_routes_lineage_repair_on_synthetic_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path)
    _patch_stage21_5(monkeypatch, s26, mode="synthetic_hash_mismatch")
    config = _write_config(tmp_path, stage26_2)

    summary = s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_3_synthetic_lineage_binding"
    assert summary["synthetic_contract_mismatch_count"] > 0


def test_stage26_3_routes_semantics_repair_on_physical_payload(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path)
    _patch_stage21_5(monkeypatch, s26, mode="physical_payload")
    config = _write_config(tmp_path, stage26_2)

    summary = s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_3_synthetic_source_semantics"
    assert summary["physical_obstacle_payload_count"] > 0


def test_stage26_3_routes_signal_repair_when_action_and_probabilities_do_not_change(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path)
    _patch_stage21_5(monkeypatch, s26, mode="unchanged")
    config = _write_config(tmp_path, stage26_2)

    summary = s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"
    assert summary["selected_viewpoint_changed_count"] == 0
    assert summary["mean_abs_probability_delta"] == 0.0


def test_stage26_3_forwards_continuous_theta_and_synthetic_feature_exposure(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path)
    _patch_stage21_5(monkeypatch, s26, mode="unchanged")
    config = _write_config(
        tmp_path,
        stage26_2,
        action_space_type="hybrid_discrete_xy_continuous_theta/v1",
        continuous_theta_action_space_enabled=True,
        synthetic_credit_feature_exposure_enabled=True,
    )

    s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage21_5_config = json.loads((tmp_path / "out" / "xunce-stage26-3-stage21-5-config.json").read_text(encoding="utf-8"))
    high_fidelity_config = json.loads((tmp_path / "out" / "xunce-stage26-3-high-fidelity-config.json").read_text(encoding="utf-8"))
    for generated in (stage21_5_config, high_fidelity_config):
        assert generated["action_space_type"] == "hybrid_discrete_xy_continuous_theta/v1"
        assert generated["continuous_theta_action_space_enabled"] is True
        assert generated["synthetic_credit_feature_exposure_enabled"] is True


def test_stage26_3_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as s26

    stage26_2 = _write_stage26_2_root(tmp_path)
    _patch_stage21_5(monkeypatch, s26)
    config = _write_config(tmp_path, stage26_2, publishes_checkpoint=True)

    summary = s26.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_3_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_5").exists()


def _patch_stage21_5(monkeypatch, s26, *, mode: str = "changed_and_improved") -> None:
    def fake_stage21_5(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["coverage_source"] == s26.COVERAGE_SOURCE
        assert cfg["path_cost_source"] == s26.PATH_COST_SOURCE
        assert cfg["synthetic_terrain_hash"] == SYNTHETIC_HASH
        assert cfg["synthetic_source_kind"] == s26.SYNTHETIC_SOURCE_KIND
        output_root.mkdir(parents=True, exist_ok=True)
        pre_root = output_root / "pre_ppo_xunce"
        post_root = output_root / "post_ppo_xunce"
        pre_root.mkdir(parents=True, exist_ok=True)
        post_root.mkdir(parents=True, exist_ok=True)
        pre_rows = [_inference_row(0, 0), _inference_row(1, 45)]
        if mode == "missing_synthetic_fields":
            post_rows = [_inference_row(0, 45, omit_synthetic=True), _inference_row(1, 45)]
        elif mode == "synthetic_hash_mismatch":
            post_rows = [_inference_row(0, 45, synthetic_hash="bad-hash"), _inference_row(1, 45)]
        elif mode == "physical_payload":
            post_rows = [_inference_row(0, 45, physical_payload=True), _inference_row(1, 45)]
        elif mode == "unchanged":
            post_rows = [_inference_row(0, 0), _inference_row(1, 45)]
        else:
            post_rows = [_inference_row(0, 45, probs=[0.3, 0.7]), _inference_row(1, 45, probs=[0.4, 0.6])]
        _write_jsonl(pre_root / s26.MODEL_INFERENCE_FILE, pre_rows)
        _write_jsonl(post_root / s26.MODEL_INFERENCE_FILE, post_rows)
        improved = mode == "changed_and_improved"
        delta = {
            "final_coverage_rate_mean_delta": 0.02 if improved else 0.0,
            "coverage_curve_auc_mean_delta": 0.03 if improved else 0.0,
            "path_cost_total_m_mean_delta": -1.0 if improved else 0.0,
            "coverage_per_100m_mean_delta": 2.0 if improved else 0.0,
        }
        (output_root / s26.stage21_5.DELTA_FILE).write_text(json.dumps(delta), encoding="utf-8")
        summary = {
            "schema_version": "xunce-stage21-5-post-update-evaluation-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_6_multi_seed_ppo_pilot",
            "pre_evaluation_root": str(pre_root),
            "post_evaluation_root": str(post_root),
            "final_coverage_rate_delta": delta["final_coverage_rate_mean_delta"],
            "coverage_curve_auc_delta": delta["coverage_curve_auc_mean_delta"],
            "path_cost_total_m_delta": delta["path_cost_total_m_mean_delta"],
            "coverage_per_100m_delta": delta["coverage_per_100m_mean_delta"],
            "post_hard_risk_violation_count": 0,
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
        (output_root / s26.stage21_5.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage21_5, "run_xunce_stage21_5_post_update_offline_trajectory_evaluation", fake_stage21_5)


def _inference_row(
    index: int,
    selected_theta: int,
    *,
    omit_synthetic: bool = False,
    synthetic_hash: str = SYNTHETIC_HASH,
    physical_payload: bool = False,
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
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "hybrid_astar_path_cost": 5.5,
        "hybrid_astar_pose_path_hash": f"pose-path-{index}-{selected_theta}",
        "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
        "legacy_grid_astar_path_cost": 4.0,
        "hybrid_vs_grid_path_cost_delta": 1.5,
        "point_grid_path_cost_fallback_used": False,
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "synthetic_terrain_hash": synthetic_hash,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "synthetic_los_blocker_cells_used": True,
        "synthetic_hard_obstacle_cells_used": True,
        "physical_obstacle_cells_written": False,
        "action_probs": probs or ([0.6, 0.4] if selected_theta == 0 else [0.4, 0.6]),
        "logits": [0.1, 0.2],
    }
    if omit_synthetic:
        for key in (
            "synthetic_terrain_hash",
            "synthetic_source_kind",
            "synthetic_los_blocker_cells_used",
            "synthetic_hard_obstacle_cells_used",
        ):
            row.pop(key)
    if physical_payload:
        row["physical_obstacle_cells_written"] = True
        row["physical_obstacle_cells"] = [[7, 7]]
    return row


def _write_stage26_2_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke",
) -> Path:
    root = tmp_path / "stage26_2"
    stage26_1 = root / "stage26_1"
    s21_4 = root / "s21_4"
    s21_1 = stage26_1 / "s21_1"
    src = stage26_1 / "src"
    s21_4.mkdir(parents=True, exist_ok=True)
    s21_1.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    (s21_4 / "experimental-xunce-stage21-4-tiny-ppo-candidate.pt").write_bytes(b"checkpoint")
    stage26_1_summary = {
        "schema_version": "xunce-stage26-1-summary/v1",
        "status": "passed",
        "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
        "synthetic_source_root": str(src),
        "synthetic_terrain_hash": SYNTHETIC_HASH,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }
    s21_1_manifest = {
        "schema_version": "xunce-stage21-1-manifest/v1",
        "model_audit": {"source_roi_expansion_root": str(src)},
    }
    summary = {
        "schema_version": "xunce-stage26-2-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "stage26_1_root": str(stage26_1),
        "stage21_4_root": str(s21_4),
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
        "synthetic_terrain_hash": SYNTHETIC_HASH,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "max_traversable_slope_deg": 30.0,
        "checkpoint_exists": True,
        "checkpoint_reload_passed": True,
        "checkpoint_boundary_passed": True,
        "stage21_3_lineage_passed": True,
        "experimental_only": True,
        "source_checkpoint_sha_consistent": True,
        "collector_source_checkpoint_match": True,
        "synthetic_physical_obstacle_payload_count": 0,
        "stage26_1_upstream_physical_obstacle_payload_count": 0,
        "synthetic_physical_obstacle_pollution_count": 0,
        "point_grid_path_cost_fallback_used_count": 0,
        "default_astar_replaced_count": 0,
        "ackermann_feasible_claimed_count": 0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    (stage26_1 / "xunce-stage26-1-summary.json").write_text(json.dumps(stage26_1_summary), encoding="utf-8")
    (s21_1 / "xunce-stage21-1-manifest.json").write_text(json.dumps(s21_1_manifest), encoding="utf-8")
    (root / "xunce-stage26-2-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def _write_config(tmp_path: Path, stage26_2_root: Path, **overrides) -> Path:
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
        "schema_version": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke-config/v1",
        "stage_id": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke",
        "stage26_2_root": str(stage26_2_root),
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
        "stage26_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_3_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
