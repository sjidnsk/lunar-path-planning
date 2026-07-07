import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8r_selects_minimal_passing_combo() -> None:
    import scripts.run_xunce_stage26_8r_pose_gate_repair as s26

    recommended = s26._select_recommended_combo(
        [
            _result("theta5_iter100", cap=5, iterations=100, trainable=130, no_pose=0),
            _result("theta3_iter100", cap=3, iterations=100, trainable=120, no_pose=0),
            _result("theta3_iter50", cap=3, iterations=50, trainable=95, no_pose=0),
        ],
        min_trainable_transition_count=100,
    )

    assert recommended["combo_id"] == "theta3_iter100"


def test_stage26_8r_runs_collector_matrix_and_writes_recommendation(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8r_pose_gate_repair as s26

    aggressive = _write_aggressive_8m_config(tmp_path)
    calls: list[dict] = []

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        calls.append({"payload": payload, "kwargs": kwargs})
        cfg = s26.stage26_8m._load_config(config_path, repo_root)
        job = s26.stage26_8m._expand_jobs(cfg, output_root, repo_root)[0]
        assert kwargs["run_mode_override"] == "run_phase"
        assert kwargs["job_id_override"] == job["job_id"]
        assert kwargs["phase_override"] == "collector"
        assert kwargs["max_jobs_override"] == 1
        cap = int(payload["candidate_reachability_max_theta_proposals_per_candidate"])
        iterations = int(payload["hybrid_astar_max_iterations"])
        if cap == 1:
            trainable, no_pose = 52, 4
        elif cap == 3 and iterations == 50:
            trainable, no_pose = 80, 2
        else:
            trainable, no_pose = 120, 0
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.stage26_1.SUMMARY_FILE,
            _collector_summary(trainable=trainable, no_pose=no_pose),
        )
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.COLLECTOR_REUSE_MARKER_FILE,
            s26.stage26_8m._collector_reuse_marker(job),
        )
        return {
            "status": "failed",
            "next_required_change": "continue_stage26_8m_jobs",
            "phase_executions": [{"job_id": job["job_id"], "phase": "collector"}],
        }

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8r_pose_gate_repair(
        config_path=_write_config(tmp_path, aggressive),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    recommendation = json.loads((tmp_path / "out" / s26.RECOMMENDED_FILE).read_text(encoding="utf-8"))

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "use_stage26_8r_recommended_pose_gate_config"
    assert summary["recommended_combo_id"] == "theta3_iter100"
    assert recommendation["schema_version"] == s26.RECOMMENDED_SCHEMA_VERSION
    assert recommendation["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert recommendation["candidate_reachability_max_theta_proposals_per_candidate"] == 3
    assert recommendation["hybrid_astar_max_iterations"] == 100
    assert recommendation["trainable_transition_count"] == 120
    assert all(call["payload"]["run_mode"] == "run_next" for call in calls)
    assert len(calls) == 3


def test_stage26_8r_routes_repair_when_all_combos_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8r_pose_gate_repair as s26

    aggressive = _write_aggressive_8m_config(tmp_path)

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **_kwargs):
        cfg = s26.stage26_8m._load_config(config_path, repo_root)
        job = s26.stage26_8m._expand_jobs(cfg, output_root, repo_root)[0]
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.stage26_1.SUMMARY_FILE,
            _collector_summary(trainable=80, no_pose=1),
        )
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.COLLECTOR_REUSE_MARKER_FILE,
            s26.stage26_8m._collector_reuse_marker(job),
        )
        return {
            "status": "failed",
            "next_required_change": "continue_stage26_8m_jobs",
            "phase_executions": [{"job_id": job["job_id"], "phase": "collector"}],
        }

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8r_pose_gate_repair(
        config_path=_write_config(tmp_path, aggressive),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_candidate_pose_generation_or_pose_gate_budget"
    assert not (tmp_path / "out" / s26.RECOMMENDED_FILE).exists()


def test_stage26_8r_rejects_stale_collector_summary_without_reuse_marker(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8r_pose_gate_repair as s26

    aggressive = _write_aggressive_8m_config(tmp_path)
    combo = {
        "combo_id": "baseline_readonly",
        "candidate_reachability_theta_proposal_policy": "candidate_viewpoint_current_step/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 1,
        "hybrid_astar_max_iterations": 50,
    }
    base = json.loads(aggressive.read_text(encoding="utf-8"))
    generated = s26._combo_stage26_8m_config(base, combo)
    combo_root = tmp_path / "out" / "baseline_readonly"
    config_path = combo_root / "xunce-stage26-8m-config.json"
    _write_json(config_path, generated)
    cfg = s26.stage26_8m._load_config(config_path, REPO_ROOT)
    job = s26.stage26_8m._expand_jobs(cfg, combo_root / "m", REPO_ROOT)[0]
    _write_json(
        Path(job["collector_root"]) / s26.stage26_8m.stage26_1.SUMMARY_FILE,
        _collector_summary(trainable=120, no_pose=0),
    )

    def fail_if_run(*_args, **_kwargs):
        raise AssertionError("stale collector summary must not be executed or recommended")

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fail_if_run)

    summary = s26.run_xunce_stage26_8r_pose_gate_repair(
        config_path=_write_config(tmp_path, aggressive, repair_combos=[combo]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_candidate_pose_generation_or_pose_gate_budget"
    assert summary["combo_results"][0]["collector_validation_status"] == "failed"
    assert summary["combo_results"][0]["collector_validation_blocking_reason"] == "collector_reuse_key_missing"
    assert not (tmp_path / "out" / s26.RECOMMENDED_FILE).exists()


def test_stage26_8r_boundary_violation_routes_to_repair(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8r_pose_gate_repair as s26

    aggressive = _write_aggressive_8m_config(tmp_path)

    summary = s26.run_xunce_stage26_8r_pose_gate_repair(
        config_path=_write_config(tmp_path, aggressive, publishes_checkpoint=True),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_candidate_pose_generation_or_pose_gate_budget"
    assert summary["boundary_rejections"] == ["publishes_checkpoint"]
    assert not (tmp_path / "out" / s26.RECOMMENDED_FILE).exists()


def _result(combo_id: str, *, cap: int, iterations: int, trainable: int, no_pose: int) -> dict:
    return {
        "combo_id": combo_id,
        "status": "passed",
        "candidate_reachability_max_theta_proposals_per_candidate": cap,
        "hybrid_astar_max_iterations": iterations,
        "trainable_transition_count": trainable,
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": no_pose,
        "selected_pose_unreachable_terminal_count": 0,
    }


def _collector_summary(*, trainable: int, no_pose: int) -> dict:
    return {
        "schema_version": "xunce-stage26-1-summary/v1",
        "status": "passed",
        "batch_row_count": trainable,
        "transition_count": trainable,
        "stage21_1_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": no_pose,
        "stage21_1_selected_pose_unreachable_terminal_count": 0,
        "stage21_1_scenario_early_terminal_step_histogram": {},
    }


def _write_config(tmp_path: Path, aggressive_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8r-pose-gate-repair-config/v1",
        "stage_id": "xunce-stage26-8r-pose-gate-repair",
        "stage26_8n_root": str(tmp_path / "s26_8n"),
        "stage26_8n_aggressive_config_path": str(aggressive_path),
        "min_trainable_transition_count": 100,
        "repair_combos": [
            {
                "combo_id": "baseline_readonly",
                "candidate_reachability_theta_proposal_policy": "candidate_viewpoint_current_step/v1",
                "candidate_reachability_max_theta_proposals_per_candidate": 1,
                "hybrid_astar_max_iterations": 50,
            },
            {
                "combo_id": "theta3_iter50",
                "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
                "candidate_reachability_max_theta_proposals_per_candidate": 3,
                "hybrid_astar_max_iterations": 50,
            },
            {
                "combo_id": "theta3_iter100",
                "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
                "candidate_reachability_max_theta_proposals_per_candidate": 3,
                "hybrid_astar_max_iterations": 100,
            },
        ],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    path = tmp_path / "config.json"
    _write_json(path, payload)
    return path


def _write_aggressive_8m_config(tmp_path: Path) -> Path:
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir(parents=True)
    payload = {
        "schema_version": "xunce-stage26-8m-generalized-resumable-training-pipeline-config/v1",
        "stage_id": "xunce-stage26-8m-generalized-resumable-training-pipeline",
        "run_mode": "run_next",
        "max_jobs_per_invocation": 1,
        "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "base_stage26_2_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
        "base_stage26_3_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
        "source_scenario_fixture_root": str(fixture_root),
        "horizons": [16],
        "seeds": [260801],
        "scenario_counts": [6],
        "collector_rollout_steps": [20],
        "eval_rollout_steps": [20],
        "update_combos": [
            {
                "combo_id": "policy_amp_repro",
                "epochs": 8,
                "learning_rate": 2.0e-5,
                "policy_loss_coefficient": 2.0,
                "value_loss_coefficient": 0.01,
                "entropy_coefficient": 0.005,
                "loss_scale": 0.5,
            }
        ],
        "max_abs_approx_kl": 1.5,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 1,
        "candidate_reachability_theta_proposal_policy": "candidate_viewpoint_current_step/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "hybrid_astar_max_iterations": 50,
        "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
        "planner_grid_resolution_m": 1.0,
        "hybrid_astar_closed_key_xy_resolution_m": 1.0,
        "hybrid_astar_primitive_duration_s": 1.0,
        "hybrid_astar_goal_position_tolerance_m": 1.0,
        "hybrid_astar_goal_theta_tolerance_deg": 45.0,
        "hybrid_astar_integration_dt_s": 0.25,
        "hybrid_astar_max_speed_mps": 0.5,
        "hybrid_astar_max_angular_speed_degps": 45.0,
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "aggressive.json"
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
