import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8s_recommends_first_scenario_count_meeting_sample_floor_with_terminal_diagnostic(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_8s_terminal_aware_sample_expansion as s26

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
        assert payload["scenario_counts"] == [payload["stage26_8s_scenario_count"]]
        assert payload["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
        assert payload["candidate_reachability_max_theta_proposals_per_candidate"] == 5
        assert payload["hybrid_astar_max_iterations"] == 200
        scenario_count = int(payload["scenario_counts"][0])
        trainable, terminal_count = (80, 4) if scenario_count == 8 else (120, 3)
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.stage26_1.SUMMARY_FILE,
            _collector_summary(trainable=trainable, no_hybrid=terminal_count),
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

    summary = s26.run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=_write_config(tmp_path, aggressive),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    recommendation = json.loads((tmp_path / "out" / s26.RECOMMENDED_FILE).read_text(encoding="utf-8"))

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "use_stage26_8s_recommended_terminal_aware_config"
    assert summary["recommended_scenario_count"] == 10
    assert recommendation["schema_version"] == s26.RECOMMENDED_SCHEMA_VERSION
    assert recommendation["recommended_scenario_count"] == 10
    assert recommendation["planner_proxy_fields"]["hybrid_astar_planning_grid_source"] == (
        "derived_high_res_planning_proxy/v1"
    )
    assert recommendation["planner_proxy_fields"]["planner_grid_resolution_m"] == 1.0
    assert recommendation["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert recommendation["planner_grid_resolution_m"] == 1.0
    assert recommendation["no_hybrid_astar_pose_reachable_candidate_for_action_mask_count"] == 3
    assert recommendation["selected_pose_unreachable_terminal_count"] == 0
    assert recommendation["collector_phase_config_hash"]
    assert recommendation["collector_input_hash"]
    assert len(calls) == 2


@pytest.mark.parametrize(
    "summary_overrides",
    [
        {"stage21_1_selected_pose_unreachable_terminal_count": 1},
        {"hard_risk_violation_count": 1},
        {"mask_violation_count": 1},
        {"path_planning_failure_count": 1},
        {"open_grid_fallback_count": 1},
    ],
)
def test_stage26_8s_routes_repair_for_terminal_or_safety_violations(
    tmp_path: Path, monkeypatch, summary_overrides: dict
) -> None:
    import scripts.run_xunce_stage26_8s_terminal_aware_sample_expansion as s26

    aggressive = _write_aggressive_8m_config(tmp_path)

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **_kwargs):
        cfg = s26.stage26_8m._load_config(config_path, repo_root)
        job = s26.stage26_8m._expand_jobs(cfg, output_root, repo_root)[0]
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.stage26_1.SUMMARY_FILE,
            _collector_summary(trainable=120, no_hybrid=0, **summary_overrides),
        )
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.COLLECTOR_REUSE_MARKER_FILE,
            s26.stage26_8m._collector_reuse_marker(job),
        )
        return {"phase_executions": [{"job_id": job["job_id"], "phase": "collector"}]}

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=_write_config(tmp_path, aggressive, scenario_counts=[8]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8s_terminal_or_reachability_evidence"
    assert summary["scenario_results"][0]["safety_or_evidence_rejections"]
    assert not (tmp_path / "out" / s26.RECOMMENDED_FILE).exists()


def test_stage26_8s_routes_expand_when_all_scenario_counts_below_sample_floor(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8s_terminal_aware_sample_expansion as s26

    aggressive = _write_aggressive_8m_config(tmp_path)

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **_kwargs):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = s26.stage26_8m._load_config(config_path, repo_root)
        job = s26.stage26_8m._expand_jobs(cfg, output_root, repo_root)[0]
        trainable_by_scenario = {8: 60, 10: 80, 12: 99}
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.stage26_1.SUMMARY_FILE,
            _collector_summary(trainable=trainable_by_scenario[int(payload["scenario_counts"][0])], no_hybrid=2),
        )
        _write_json(
            Path(job["collector_root"]) / s26.stage26_8m.COLLECTOR_REUSE_MARKER_FILE,
            s26.stage26_8m._collector_reuse_marker(job),
        )
        return {"phase_executions": [{"job_id": job["job_id"], "phase": "collector"}]}

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=_write_config(tmp_path, aggressive),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "expand_stage26_8s_terminal_aware_scenario_budget"
    assert [row["scenario_count"] for row in summary["scenario_results"]] == [8, 10, 12]
    assert not (tmp_path / "out" / s26.RECOMMENDED_FILE).exists()


def test_stage26_8s_rejects_stale_collector_summary_without_reuse_marker(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8s_terminal_aware_sample_expansion as s26

    aggressive = _write_aggressive_8m_config(tmp_path)
    base = json.loads(aggressive.read_text(encoding="utf-8"))
    scenario = s26._scenario_probe(8)
    generated = s26._scenario_stage26_8m_config(base, scenario)
    scenario_root = tmp_path / "out" / "scenario_count_8"
    config_path = scenario_root / "xunce-stage26-8m-config.json"
    _write_json(config_path, generated)
    cfg = s26.stage26_8m._load_config(config_path, REPO_ROOT)
    job = s26.stage26_8m._expand_jobs(cfg, scenario_root / "m", REPO_ROOT)[0]
    _write_json(
        Path(job["collector_root"]) / s26.stage26_8m.stage26_1.SUMMARY_FILE,
        _collector_summary(trainable=120, no_hybrid=0),
    )

    def fail_if_run(*_args, **_kwargs):
        raise AssertionError("stale collector summary must not be executed or recommended")

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fail_if_run)

    summary = s26.run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=_write_config(tmp_path, aggressive, scenario_counts=[8]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8s_terminal_or_reachability_evidence"
    assert summary["scenario_results"][0]["collector_validation_status"] == "failed"
    assert summary["scenario_results"][0]["collector_validation_blocking_reason"] == "collector_reuse_key_missing"
    assert not (tmp_path / "out" / s26.RECOMMENDED_FILE).exists()


def test_stage26_8s_routes_required_inputs_for_unreadable_aggressive_config(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8s_terminal_aware_sample_expansion as s26

    bad_aggressive = tmp_path / "bad-aggressive.json"
    bad_aggressive.write_text("{", encoding="utf-8")

    output_root = tmp_path / "out"
    summary = s26.run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=_write_config(tmp_path, bad_aggressive),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    persisted = json.loads((output_root / s26.SUMMARY_FILE).read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8s_required_inputs"
    assert summary["input_rejections"] == ["stage26_8n_aggressive_config_unreadable"]
    assert persisted["input_rejections"] == ["stage26_8n_aggressive_config_unreadable"]


def test_stage26_8s_routes_required_inputs_for_stage26_8m_input_rejections(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8s_terminal_aware_sample_expansion as s26

    aggressive = _write_aggressive_8m_config(tmp_path)

    def fake_8m(*_args, **_kwargs):
        return {
            "status": "failed",
            "next_required_change": "rerun_stage26_8m_required_inputs",
            "input_rejections": ["base_stage26_1_config_missing"],
            "phase_executions": [],
        }

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=_write_config(tmp_path, aggressive, scenario_counts=[8]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8s_required_inputs"
    assert summary["scenario_results"][0]["stage26_8m_input_rejections"] == ["base_stage26_1_config_missing"]
    assert summary["scenario_results"][0]["collector_validation_blocking_reason"] == "stage26_8m_input_rejections"


def test_stage26_8s_boundary_true_routes_repair_and_records_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8s_terminal_aware_sample_expansion as s26

    aggressive = _write_aggressive_8m_config(tmp_path)
    summary = s26.run_xunce_stage26_8s_terminal_aware_sample_expansion(
        config_path=_write_config(tmp_path, aggressive, publishes_checkpoint=True),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8s_terminal_or_reachability_evidence"
    assert summary["boundary_rejections"] == ["publishes_checkpoint"]
    assert not (tmp_path / "out" / s26.RECOMMENDED_FILE).exists()


def test_stage26_8s_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    stage = registry["stages"]["xunce-stage26-8s-terminal-aware-sample-expansion"]

    assert stage["script"] == "scripts/run_xunce_stage26_8s_terminal_aware_sample_expansion.py"
    assert stage["default_config"] == "configs/xunce_stage26_8s_terminal_aware_sample_expansion_v1.json"
    assert stage["default_output_root"] == "D:/xunce/out/s26_8s"


def _collector_summary(
    *,
    trainable: int,
    no_hybrid: int,
    stage21_1_selected_pose_unreachable_terminal_count: int = 0,
    hard_risk_violation_count: int = 0,
    mask_violation_count: int = 0,
    path_planning_failure_count: int = 0,
    open_grid_fallback_count: int = 0,
) -> dict:
    return {
        "schema_version": "xunce-stage26-1-summary/v1",
        "status": "passed",
        "batch_row_count": trainable,
        "transition_count": trainable,
        "stage21_1_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": no_hybrid,
        "stage21_1_selected_pose_unreachable_terminal_count": stage21_1_selected_pose_unreachable_terminal_count,
        "stage21_1_scenario_early_terminal_step_histogram": {},
        "stage21_1_trainable_transition_count_by_scenario": {},
        "hard_risk_violation_count": hard_risk_violation_count,
        "mask_violation_count": mask_violation_count,
        "path_planning_failure_count": path_planning_failure_count,
        "open_grid_fallback_count": open_grid_fallback_count,
    }


def _write_config(tmp_path: Path, aggressive_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8s-terminal-aware-sample-expansion-config/v1",
        "stage_id": "xunce-stage26-8s-terminal-aware-sample-expansion",
        "stage26_8n_root": str(tmp_path / "s26_8n"),
        "stage26_8n_aggressive_config_path": str(aggressive_path),
        "min_trainable_transition_count": 100,
        "scenario_counts": [8, 10, 12],
        "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 5,
        "hybrid_astar_max_iterations": 200,
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
