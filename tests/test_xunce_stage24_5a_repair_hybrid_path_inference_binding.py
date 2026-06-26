import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage24_5a_repairs_binding_and_preserves_repaired_route(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5a_repair_hybrid_path_inference_binding as s24a

    stage24_5_root = _write_stage24_5_root(tmp_path)
    base_config = _write_stage24_5_base_config(tmp_path)
    planner_overrides = _hybrid_planner_overrides()
    config = _write_config(tmp_path, stage24_5_root, base_config, **planner_overrides)
    _patch_stage24_5(monkeypatch, s24a, mode="signal")

    summary = s24a.run_xunce_stage24_5a_repair_hybrid_path_inference_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    repaired_config = json.loads((tmp_path / "out" / "xunce-stage24-5a-repaired-stage24-5-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "repair_stage24_hybrid_path_policy_update_signal_strength"
    assert summary["strong_state_join_available_count"] == 8
    assert summary["hybrid_path_inference_required_field_missing_count"] == 0
    assert summary["path_cost_source_mismatch_count"] == 0
    assert summary["grid_fallback_count"] == 0
    assert summary["default_astar_replaced_count"] == 0
    assert summary["ackermann_feasible_claimed_count"] == 0
    assert repaired_config["hybrid_astar_pose_path_cost_enabled"] is True
    assert repaired_config["coverage_source"] == s24a.SLOPE_COVERAGE_SOURCE
    assert repaired_config["path_cost_source"] == s24a.HYBRID_ASTAR_PATH_COST_SOURCE
    assert repaired_config["max_traversable_slope_deg"] == 30.0
    assert repaired_config["default_astar_replaced"] is False
    assert repaired_config["hybrid_astar_ackermann_feasible_claimed"] is False
    for key, value in planner_overrides.items():
        assert repaired_config[key] == value
    assert repaired_config["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False


def test_stage24_5a_rejects_non_binding_stage24_5_route(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5a_repair_hybrid_path_inference_binding as s24a

    stage24_5_root = _write_stage24_5_root(
        tmp_path,
        next_required_change="repair_stage24_hybrid_path_policy_update_signal_strength",
    )
    base_config = _write_stage24_5_base_config(tmp_path)
    config = _write_config(tmp_path, stage24_5_root, base_config)
    _patch_stage24_5(monkeypatch, s24a, mode="signal")

    summary = s24a.run_xunce_stage24_5a_repair_hybrid_path_inference_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_5a_required_inputs"
    assert "stage24_5_route_not_hybrid_path_inference_binding" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s24_5_repaired").exists()


def test_stage24_5a_continues_binding_repair_when_fields_still_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5a_repair_hybrid_path_inference_binding as s24a

    stage24_5_root = _write_stage24_5_root(tmp_path)
    base_config = _write_stage24_5_base_config(tmp_path)
    config = _write_config(tmp_path, stage24_5_root, base_config)
    _patch_stage24_5(monkeypatch, s24a, mode="still_missing")

    summary = s24a.run_xunce_stage24_5a_repair_hybrid_path_inference_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "continue_stage24_5_hybrid_path_inference_binding_repair"
    assert summary["hybrid_path_inference_required_field_missing_count"] > 0
    assert "hybrid_path_inference_binding_still_missing" in summary["blocking_reason_codes"]


def test_stage24_5a_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_5a_repair_hybrid_path_inference_binding as s24a

    stage24_5_root = _write_stage24_5_root(tmp_path)
    base_config = _write_stage24_5_base_config(tmp_path)
    config = _write_config(tmp_path, stage24_5_root, base_config, publishes_checkpoint=True)
    _patch_stage24_5(monkeypatch, s24a, mode="signal")

    summary = s24a.run_xunce_stage24_5a_repair_hybrid_path_inference_binding(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage24_5a_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s24_5_repaired").exists()


def _patch_stage24_5(monkeypatch, s24a, *, mode: str) -> None:
    def fake_stage24_5(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["hybrid_astar_pose_path_cost_enabled"] is True
        assert cfg["coverage_source"] == s24a.SLOPE_COVERAGE_SOURCE
        assert cfg["path_cost_source"] == s24a.HYBRID_ASTAR_PATH_COST_SOURCE
        assert cfg["runs_new_ppo_update"] is False
        output_root.mkdir(parents=True, exist_ok=True)
        if mode == "still_missing":
            summary = _stage24_5_summary(
                next_required_change="repair_stage24_5_hybrid_path_inference_binding",
                missing=12,
                source_mismatch=2,
            )
        else:
            summary = _stage24_5_summary(
                status="failed",
                next_required_change="repair_stage24_hybrid_path_policy_update_signal_strength",
                missing=0,
                source_mismatch=0,
            )
        (output_root / "xunce-stage24-5-hybrid-path-action-change-audit.json").write_text(
            json.dumps(
                {
                    "strong_state_join_available_count": summary["strong_state_join_available_count"],
                    "hybrid_path_inference_required_field_missing_count": summary[
                        "hybrid_path_inference_required_field_missing_count"
                    ],
                    "path_cost_source_mismatch_count": summary["path_cost_source_mismatch_count"],
                    "grid_fallback_count": summary["grid_fallback_count"],
                    "default_astar_replaced_count": summary["default_astar_replaced_count"],
                    "ackermann_feasible_claimed_count": summary["ackermann_feasible_claimed_count"],
                }
            ),
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(
        s24a.stage24_5,
        "run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke",
        fake_stage24_5,
    )


def _write_stage24_5_root(
    tmp_path: Path,
    *,
    next_required_change: str = "repair_stage24_5_hybrid_path_inference_binding",
) -> Path:
    root = tmp_path / "stage24_5"
    root.mkdir()
    (root / "xunce-stage24-5-summary.json").write_text(
        json.dumps(_stage24_5_summary(next_required_change=next_required_change, missing=80, source_mismatch=16)),
        encoding="utf-8",
    )
    return root


def _stage24_5_summary(
    *,
    status: str = "failed",
    next_required_change: str,
    missing: int,
    source_mismatch: int,
) -> dict:
    return {
        "status": status,
        "next_required_change": next_required_change,
        "strong_state_join_available_count": 8,
        "hybrid_path_inference_required_field_missing_count": missing,
        "path_cost_source_mismatch_count": source_mismatch,
        "grid_fallback_count": 0,
        "default_astar_replaced_count": 0,
        "ackermann_feasible_claimed_count": 0,
    }


def _write_stage24_5_base_config(tmp_path: Path) -> Path:
    config = tmp_path / "stage24_5_base.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke-config/v1",
                "stage_id": "xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke",
                "stage24_4_root": "stage24_4",
                "stage21_5_base_config": "configs/xunce_stage21_5_post_update_offline_trajectory_evaluation_v1.json",
                "high_fidelity_config": "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
                "required_scenario_count": 2,
                "rollout_steps": 4,
                "dynamic_max_candidates_per_step": 36,
                "dynamic_proposal_pool_limit_per_step": 288,
                "theta_bin_count": 8,
                "theta_step_deg": 45,
                "sensor_model_id": "theta-fov-90-range-radius/v1",
                "sensor_fov_deg": 90.0,
                "sensor_range_cells": 2,
                "min_mean_abs_probability_delta_for_signal": 1e-5,
                "stage24_5_authorized": False,
                "release_or_training_authorized": False,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "canary_traffic_fraction": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return config


def _write_config(tmp_path: Path, stage24_5_root: Path, base_config: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage24-5a-repair-hybrid-path-inference-binding-config/v1",
        "stage_id": "xunce-stage24-5a-repair-hybrid-path-inference-binding",
        "stage24_5_root": str(stage24_5_root),
        "stage24_5_base_config": str(base_config),
        "stage24_5a_authorized": False,
        "release_or_training_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    config = tmp_path / "stage24_5a.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config


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
