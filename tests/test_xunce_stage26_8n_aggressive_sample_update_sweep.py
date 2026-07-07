import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8n_rejects_non_update_strength_stage26_8i_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path, next_required_change="run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot")
    output_root = tmp_path / "out"

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(tmp_path, root),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    routing = json.loads((output_root / "xunce-stage26-8n-next-stage-routing.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert "stage26_8i_route_mismatch" in summary["input_rejections"]
    assert routing["release_or_training_authorized"] is False
    assert routing["publishes_checkpoint"] is False
    assert routing["replaces_default_policy"] is False
    assert routing["connects_real_executor"] is False
    assert routing["starts_online_canary"] is False
    assert routing["canary_traffic_fraction"] == 0.0


def test_stage26_8n_writes_aggressive_8m_config_with_collector_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    captured: dict = {}

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        captured.update(json.loads(config_path.read_text(encoding="utf-8")))
        return _write_8m_pending_summary(output_root)

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_8n_aggressive_jobs"
    assert captured["collector_reuse_policy"] == "by_horizon_seed_scenario_rollout/v1"
    assert captured["scenario_counts"] == [6]
    assert captured["collector_rollout_steps"] == [20]
    assert captured["eval_rollout_steps"] == [20]
    assert captured["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert captured["hybrid_astar_max_iterations"] == 100
    assert captured["candidate_reachability_max_theta_proposals_per_candidate"] == 3
    assert captured["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert [combo["combo_id"] for combo in captured["update_combos"]] == [
        "policy_amp_repro",
        "depth16_lr3e5_policy3",
        "depth24_lr3e5_policy3",
        "lr5e5_policy3",
        "value_off_policy4_probe",
    ]
    assert captured["max_abs_approx_kl"] == 1.5


def test_stage26_8n_accepts_stage26_8s_terminal_aware_recommendation(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_8s_path = _write_8s_recommended_config(
        tmp_path,
        recommended_scenario_count=10,
        candidate_reachability_max_theta_proposals_per_candidate=5,
        candidate_reachability_theta_proposal_policy="candidate_current_bearing_sweep/v1",
        hybrid_astar_max_iterations=200,
        trainable_transition_count=130,
        no_hybrid_astar_pose_reachable_candidate_for_action_mask_count=7,
    )
    missing_8r_path = tmp_path / "missing-8r.json"
    captured: dict = {}

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        captured.update(json.loads(config_path.read_text(encoding="utf-8")))
        return _write_8m_pending_summary(output_root)

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    output_root = tmp_path / "out"
    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            stage26_8s_recommended_config_path=str(recommended_8s_path),
            stage26_8r_recommended_config_path=str(missing_8r_path),
        ),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    manifest = json.loads((output_root / "xunce-stage26-8n-manifest.json").read_text(encoding="utf-8"))
    report = (output_root / "xunce-stage26-8n-report.md").read_text(encoding="utf-8")

    assert summary["next_required_change"] == "continue_stage26_8n_aggressive_jobs"
    assert summary["input_rejections"] == []
    assert captured["scenario_counts"] == [10]
    assert captured["hybrid_astar_max_iterations"] == 200
    assert captured["candidate_reachability_max_theta_proposals_per_candidate"] == 5
    assert captured["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert summary["stage26_8s_recommended_config_path"] == str(recommended_8s_path)
    assert summary["stage26_8s_recommended_scenario_count"] == 10
    assert summary["stage26_8s_trainable_transition_count"] == 130
    assert summary["stage26_8s_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count"] == 7
    assert summary["stage26_8s_selected_pose_unreachable_terminal_count"] == 0
    assert summary["stage26_8r_recommended_combo_id"] is None
    assert manifest["stage26_8s_recommended_config_path"] == str(recommended_8s_path)
    assert manifest["stage26_8s_recommended_scenario_count"] == 10
    assert manifest["stage26_8s_trainable_transition_count"] == 130
    assert manifest["stage26_8s_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count"] == 7
    assert manifest["stage26_8s_selected_pose_unreachable_terminal_count"] == 0
    assert str(recommended_8s_path) in report
    assert "stage26_8s_recommended_scenario_count" in report
    assert "stage26_8s_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count" in report


def test_stage26_8n_prefers_stage26_8s_over_stage26_8r_and_stage26_8q_fields(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_8q_path = _write_recommended_config(
        tmp_path,
        name="stage26_8q_recommended_with_iterations.json",
        hybrid_astar_max_iterations=50,
    )
    recommended_8r_path = _write_8r_recommended_config(
        tmp_path,
        name="stage26_8r_recommended_theta3_iter100.json",
        candidate_reachability_max_theta_proposals_per_candidate=3,
        hybrid_astar_max_iterations=100,
    )
    recommended_8s_path = _write_8s_recommended_config(
        tmp_path,
        recommended_scenario_count=10,
        candidate_reachability_max_theta_proposals_per_candidate=5,
        candidate_reachability_theta_proposal_policy="candidate_viewpoint_current_step/v1",
        hybrid_astar_max_iterations=200,
        trainable_transition_count=140,
    )
    captured: dict = {}

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        captured.update(json.loads(config_path.read_text(encoding="utf-8")))
        return _write_8m_pending_summary(output_root)

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            stage26_8q_recommended_config_path=str(recommended_8q_path),
            stage26_8r_recommended_config_path=str(recommended_8r_path),
            stage26_8s_recommended_config_path=str(recommended_8s_path),
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_8n_aggressive_jobs"
    assert captured["scenario_counts"] == [10]
    assert captured["candidate_reachability_max_theta_proposals_per_candidate"] == 5
    assert captured["candidate_reachability_theta_proposal_policy"] == "candidate_viewpoint_current_step/v1"
    assert captured["hybrid_astar_max_iterations"] == 200
    assert summary["stage26_8s_recommended_scenario_count"] == 10
    assert summary["stage26_8r_recommended_combo_id"] is None


def test_stage26_8n_consumes_stage26_8q_recommended_planner_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_path = _write_recommended_config(tmp_path)
    captured: dict = {}

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        captured.update(json.loads(config_path.read_text(encoding="utf-8")))
        return _write_8m_pending_summary(output_root)

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    output_root = tmp_path / "out"
    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(tmp_path, root, stage26_8q_recommended_config_path=str(recommended_path)),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    manifest = json.loads((output_root / "xunce-stage26-8n-manifest.json").read_text(encoding="utf-8"))
    report = (output_root / "xunce-stage26-8n-report.md").read_text(encoding="utf-8")

    assert summary["next_required_change"] == "continue_stage26_8n_aggressive_jobs"
    assert captured["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert captured["planner_grid_resolution_m"] == 1.0
    assert captured["hybrid_astar_closed_key_xy_resolution_m"] == 1.0
    assert captured["hybrid_astar_primitive_duration_s"] == 1.0
    assert captured["hybrid_astar_goal_position_tolerance_m"] == 1.0
    assert captured["hybrid_astar_goal_theta_tolerance_deg"] == 45.0
    assert captured["hybrid_astar_max_iterations"] == 100
    assert captured["candidate_reachability_max_theta_proposals_per_candidate"] == 3
    assert captured["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert captured["hybrid_astar_integration_dt_s"] == 0.25
    assert captured["hybrid_astar_max_speed_mps"] == 0.5
    assert captured["hybrid_astar_max_angular_speed_degps"] == 45.0
    assert captured["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert summary["stage26_8q_recommended_config_path"] == str(recommended_path)
    assert summary["stage26_8q_recommended_combo_id"] == "aligned_1m_proxy"
    assert summary["stage26_8q_aligned_trainable_transition_count"] == 120
    assert summary["stage26_8q_planner_overrides"]["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert summary["stage26_8q_planner_overrides"]["planner_grid_resolution_m"] == 1.0
    assert summary["stage26_8q_planner_overrides"]["hybrid_astar_closed_key_xy_resolution_m"] == 1.0
    assert summary["stage26_8r_recommended_combo_id"] == "theta3_iter100"
    assert summary["stage26_8r_trainable_transition_count"] == 120
    assert summary["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert summary["hybrid_astar_max_iterations"] == 100
    assert summary["candidate_reachability_max_theta_proposals_per_candidate"] == 3
    assert summary["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert manifest["stage26_8q_recommended_config_path"] == str(recommended_path)
    assert manifest["stage26_8q_recommended_combo_id"] == "aligned_1m_proxy"
    assert manifest["stage26_8q_aligned_trainable_transition_count"] == 120
    assert manifest["stage26_8q_planner_overrides"]["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert manifest["stage26_8r_recommended_combo_id"] == "theta3_iter100"
    assert manifest["stage26_8r_trainable_transition_count"] == 120
    assert manifest["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert manifest["hybrid_astar_max_iterations"] == 100
    assert manifest["candidate_reachability_max_theta_proposals_per_candidate"] == 3
    assert manifest["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert str(recommended_path) in report
    assert "aligned_1m_proxy" in report
    assert "derived_high_res_planning_proxy/v1" in report
    assert "theta3_iter100" in report
    assert "hybrid_astar_pose_reachability/v1" in report
    assert "hybrid_astar_max_iterations" in report
    assert "candidate_reachability_max_theta_proposals_per_candidate" in report
    assert "candidate_reachability_theta_proposal_policy" in report


def test_stage26_8n_prefers_stage26_8r_pose_gate_iterations_over_stage26_8q_override(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_8q_path = _write_recommended_config(
        tmp_path,
        name="stage26_8q_recommended_with_iterations.json",
        hybrid_astar_max_iterations=50,
    )
    recommended_8r_path = _write_8r_recommended_config(
        tmp_path,
        name="stage26_8r_recommended_theta5_iter200.json",
        recommended_combo_id="theta5_iter200",
        candidate_reachability_max_theta_proposals_per_candidate=5,
        hybrid_astar_max_iterations=200,
        trainable_transition_count=140,
    )
    captured: dict = {}

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        captured.update(json.loads(config_path.read_text(encoding="utf-8")))
        return _write_8m_pending_summary(output_root)

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    output_root = tmp_path / "out"
    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            stage26_8q_recommended_config_path=str(recommended_8q_path),
            stage26_8r_recommended_config_path=str(recommended_8r_path),
        ),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    manifest = json.loads((output_root / "xunce-stage26-8n-manifest.json").read_text(encoding="utf-8"))

    assert summary["next_required_change"] == "continue_stage26_8n_aggressive_jobs"
    assert captured["hybrid_astar_max_iterations"] == 200
    assert captured["candidate_reachability_max_theta_proposals_per_candidate"] == 5
    assert captured["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert summary["stage26_8r_recommended_combo_id"] == "theta5_iter200"
    assert summary["hybrid_astar_max_iterations"] == 200
    assert summary["candidate_reachability_max_theta_proposals_per_candidate"] == 5
    assert manifest["hybrid_astar_max_iterations"] == 200
    assert manifest["candidate_reachability_max_theta_proposals_per_candidate"] == 5


@pytest.mark.parametrize(
    ("case_name", "overrides", "expected_reason"),
    [
        ("schema", {"schema_version": "wrong/v1"}, "stage26_8s_recommended_schema_version_mismatch"),
        ("boundary", {"publishes_checkpoint": True}, "stage26_8s_recommended_publishes_checkpoint_not_false"),
        (
            "count",
            {"trainable_transition_count": 99},
            "stage26_8s_recommended_trainable_transition_count_below_min",
        ),
        (
            "selected",
            {"selected_pose_unreachable_terminal_count": 1},
            "stage26_8s_recommended_selected_pose_unreachable_terminal_count_nonzero",
        ),
    ],
)
def test_stage26_8n_rejects_invalid_stage26_8s_recommended_config(
    tmp_path: Path,
    case_name: str,
    overrides: dict,
    expected_reason: str,
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_path = _write_8s_recommended_config(tmp_path, name=f"8s-{case_name}.json", **overrides)

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            run_stage26_8m=False,
            stage26_8s_recommended_config_path=str(recommended_path),
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert expected_reason in summary["input_rejections"]


@pytest.mark.parametrize(
    ("case_name", "overrides", "expected_reason"),
    [
        ("schema", {"schema_version": "wrong/v1"}, "stage26_8q_recommended_schema_version_mismatch"),
        ("combo", {"recommended_combo_id": "baseline"}, "stage26_8q_recommended_combo_id_mismatch"),
        ("count", {"aligned_trainable_transition_count": 99}, "stage26_8q_aligned_trainable_transition_count_below_min"),
    ],
)
def test_stage26_8n_rejects_invalid_stage26_8q_recommended_config(
    tmp_path: Path,
    case_name: str,
    overrides: dict,
    expected_reason: str,
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_path = _write_recommended_config(tmp_path, name=f"{case_name}.json", **overrides)

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            run_stage26_8m=False,
            stage26_8q_recommended_config_path=str(recommended_path),
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert expected_reason in summary["input_rejections"]


def test_stage26_8n_rejects_missing_stage26_8q_recommended_config(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    missing_path = tmp_path / "missing-recommended.json"

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            run_stage26_8m=False,
            stage26_8q_recommended_config_path=str(missing_path),
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert "stage26_8q_recommended_config_missing" in summary["input_rejections"]


def test_stage26_8n_rejects_unreadable_stage26_8q_recommended_config(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    bad_path = tmp_path / "bad-recommended.json"
    bad_path.write_text("[1]\n", encoding="utf-8")

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            run_stage26_8m=False,
            stage26_8q_recommended_config_path=str(bad_path),
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert "stage26_8q_recommended_config_unreadable" in summary["input_rejections"]


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_reason"),
    [
        ("release_or_training_authorized", True, "stage26_8q_recommended_release_or_training_authorized_not_false"),
        ("publishes_checkpoint", True, "stage26_8q_recommended_publishes_checkpoint_not_false"),
        ("replaces_default_policy", True, "stage26_8q_recommended_replaces_default_policy_not_false"),
        ("connects_real_executor", True, "stage26_8q_recommended_connects_real_executor_not_false"),
        ("starts_online_canary", True, "stage26_8q_recommended_starts_online_canary_not_false"),
        ("canary_traffic_fraction", 0.1, "stage26_8q_recommended_canary_traffic_fraction_not_zero"),
    ],
)
def test_stage26_8n_rejects_stage26_8q_recommended_boundary_violations(
    tmp_path: Path,
    field: str,
    bad_value,
    expected_reason: str,
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_path = _write_recommended_config(tmp_path, name=f"{field}.json", **{field: bad_value})

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            run_stage26_8m=False,
            stage26_8q_recommended_config_path=str(recommended_path),
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert expected_reason in summary["input_rejections"]


@pytest.mark.parametrize(
    ("case_name", "overrides", "expected_reason"),
    [
        ("schema", {"schema_version": "wrong/v1"}, "stage26_8r_recommended_schema_version_mismatch"),
        (
            "source",
            {"candidate_reachability_gate_source": "legacy_action_mask_validation/v1"},
            "stage26_8r_recommended_gate_source_mismatch",
        ),
        (
            "count",
            {"trainable_transition_count": 99},
            "stage26_8r_recommended_trainable_transition_count_below_min",
        ),
        (
            "pose",
            {"no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": 1},
            "stage26_8r_recommended_pose_gate_empty_count_nonzero",
        ),
        (
            "boundary",
            {"publishes_checkpoint": True},
            "stage26_8r_recommended_publishes_checkpoint_not_false",
        ),
    ],
)
def test_stage26_8n_rejects_invalid_stage26_8r_recommended_config(
    tmp_path: Path,
    case_name: str,
    overrides: dict,
    expected_reason: str,
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    recommended_path = _write_8r_recommended_config(tmp_path, name=f"8r-{case_name}.json", **overrides)

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(
            tmp_path,
            root,
            run_stage26_8m=False,
            stage26_8r_recommended_config_path=str(recommended_path),
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert expected_reason in summary["input_rejections"]


def test_stage26_8n_routes_to_expand_sample_budget_when_shared_collector_under_target(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    output_root = tmp_path / "out"
    collector_root = output_root / "m" / "cabc123"
    _write_json(
        collector_root / "xunce-stage26-1-summary.json",
        {"schema_version": "xunce-stage26-1-summary/v1", "status": "passed", "trainable_transition_count": 80},
    )
    _write_8m_state(output_root / "m", collector_root=collector_root, collector_status="complete")

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(tmp_path, root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "expand_stage26_8n_aggressive_sample_budget"
    assert summary["trainable_transition_count"] == 80


def test_stage26_8n_routes_to_expand_sample_budget_when_terminal_count_nonzero() -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        collector_audit={
            "collector_complete": True,
            "trainable_transition_count": 52,
            "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": 4,
        },
        update_rows=[],
        selected_eval_job_ids=[],
        job_rows=[],
        stage26_8m_summary={},
        min_trainable_transition_count=100,
        max_abs_approx_kl=1.5,
    )

    assert route == "expand_stage26_8n_aggressive_sample_budget"


def test_stage26_8n_routes_collector_failure_before_expand_sample_budget(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    rows = [
        {
            "job_id": "shared",
            "phase": "collector",
            "status": "failed",
            "blocking_reason": "input_hash_mismatch",
            "binding_or_safety_failure": False,
        }
    ]

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        collector_audit={"collector_complete": True, "trainable_transition_count": 3},
        update_rows=[],
        selected_eval_job_ids=[],
        job_rows=rows,
        stage26_8m_summary={},
        min_trainable_transition_count=100,
        max_abs_approx_kl=1.5,
    )

    assert route == "repair_stage26_8n_collector_binding_or_safety"


def test_stage26_8n_selects_only_two_stable_combos_for_eval(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    rows = [
        _update_row("a", stable=True, parameter_delta_l2=0.2, policy_kl=1.4),
        _update_row("b", stable=True, parameter_delta_l2=0.5, policy_kl=1.2),
        _update_row("c", stable=True, parameter_delta_l2=0.3, policy_kl=1.0),
        _update_row("d", stable=False, parameter_delta_l2=0.9, policy_kl=2.0),
    ]

    selected = s26._selected_eval_job_ids(rows, max_abs_approx_kl=1.5, max_eval_count=2)

    assert selected == ["b", "c"]


def test_stage26_8n_efficiency_audit_exposes_stage26_9_required_zero_fields(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    aggregate_summary = {
        "schema_version": "xunce-stage26-3-summary/v1",
        "selected_action_changed_count": 2,
        "coverage_per_100m_delta": 0.5,
        "synthetic_inference_required_field_missing_count": 0,
        "hybrid_path_missing_provenance_count": 0,
        "hybrid_path_contract_mismatch_count": 0,
        "explicit_unreachable_selected_provenance_count": 0,
        "pre_unreachable_selected_count": 0,
        "post_unreachable_selected_count": 0,
        "pre_selected_reachability_provenance_invalid_count": 0,
        "post_selected_reachability_provenance_invalid_count": 0,
    }
    aggregate_path = tmp_path / "aggregate" / "xunce-stage26-3-summary.json"
    _write_json(aggregate_path, aggregate_summary)

    audit = s26._efficiency_audit(
        [
            {
                "job_id": "selected_job",
                "phase": "aggregate",
                "status": "complete",
                "summary_path": str(aggregate_path),
            }
        ]
    )

    for field in s26.EFFICIENCY_ZERO_COUNT_FIELDS:
        assert field in audit
        assert audit[field] == 0


def test_stage26_8n_ignores_nonselected_eval_pending_when_selected_eval_complete(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    selected = ["selected_job"]
    rows = [
        {"job_id": "selected_job", "phase": "aggregate", "status": "complete", "selected_action_changed_count": 1, "main_coverage_per_100m_delta": 0.0},
        {"job_id": "skipped_job", "phase": "eval_pre", "status": "pending"},
        {"job_id": "skipped_job", "phase": "eval_post", "status": "pending"},
        {"job_id": "skipped_job", "phase": "aggregate", "status": "pending"},
    ]

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        collector_audit={"collector_complete": True, "trainable_transition_count": 120},
        update_rows=[_update_row("selected_job", stable=True, parameter_delta_l2=0.5, policy_kl=0.3)],
        selected_eval_job_ids=selected,
        job_rows=rows,
        stage26_8m_summary={},
        min_trainable_transition_count=100,
        max_abs_approx_kl=1.5,
    )

    assert route == "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"


def test_stage26_8n_routes_eval_repair_for_lineage_mismatched_positive_aggregate(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    rows = [
        {
            "job_id": "selected_job",
            "phase": "aggregate",
            "status": "complete",
            "selected_action_changed_count": 1,
            "main_coverage_per_100m_delta": 1.0,
            "lineage_mismatch": True,
        }
    ]

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        collector_audit={"collector_complete": True, "trainable_transition_count": 120},
        update_rows=[_update_row("selected_job", stable=True, parameter_delta_l2=0.5, policy_kl=0.3)],
        selected_eval_job_ids=["selected_job"],
        job_rows=rows,
        stage26_8m_summary={},
        min_trainable_transition_count=100,
        max_abs_approx_kl=1.5,
    )

    assert route == "repair_stage26_8n_eval_binding_or_safety"


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("min_trainable_transition_count", "min_trainable_transition_count must be positive"),
        ("max_stable_eval_count", "max_stable_eval_count must be positive"),
    ],
)
def test_stage26_8n_rejects_nonpositive_count_config(tmp_path: Path, field: str, message: str) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)

    with pytest.raises(ValueError, match=message):
        s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
            config_path=_write_config(tmp_path, root, **{field: 0}),
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )


def test_stage26_8n_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    assert "xunce-stage26-8n-aggressive-sample-update-sweep" in registry["stages"]


def _write_config(tmp_path: Path, stage26_8i_root: Path, **overrides) -> Path:
    (tmp_path / "stage26_8g").mkdir(parents=True, exist_ok=True)
    recommended_path = overrides.pop("stage26_8q_recommended_config_path", None)
    if recommended_path is None:
        recommended_path = str(_write_recommended_config(tmp_path))
    stage26_8r_recommended_path = overrides.pop("stage26_8r_recommended_config_path", None)
    if stage26_8r_recommended_path is None:
        stage26_8r_recommended_path = str(_write_8r_recommended_config(tmp_path))
    stage26_8s_recommended_path = overrides.pop("stage26_8s_recommended_config_path", None)
    if stage26_8s_recommended_path is None:
        stage26_8s_recommended_path = str(tmp_path / "missing-stage26-8s-recommended.json")
    payload = {
        "schema_version": "xunce-stage26-8n-aggressive-sample-update-sweep-config/v1",
        "stage_id": "xunce-stage26-8n-aggressive-sample-update-sweep",
        "stage26_8i_root": str(stage26_8i_root),
        "stage26_8q_recommended_config_path": str(recommended_path),
        "stage26_8r_recommended_config_path": str(stage26_8r_recommended_path),
        "stage26_8s_recommended_config_path": str(stage26_8s_recommended_path),
        "source_scenario_fixture_root": str(tmp_path / "stage26_8g"),
        "run_stage26_8m": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    path = tmp_path / "config.json"
    _write_json(path, payload)
    return path


def _write_8s_recommended_config(tmp_path: Path, *, name: str = "stage26_8s_recommended.json", **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8s-recommended-stage26-8n-config/v1",
        "stage_id": "xunce-stage26-8s-terminal-aware-sample-expansion",
        "recommended_scenario_count": 8,
        "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
        "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 4,
        "hybrid_astar_max_iterations": 150,
        "trainable_transition_count": 120,
        "min_trainable_transition_count": 100,
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": 0,
        "selected_pose_unreachable_terminal_count": 0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    path = tmp_path / name
    _write_json(path, payload)
    return path


def _write_8r_recommended_config(tmp_path: Path, *, name: str = "stage26_8r_recommended.json", **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8r-recommended-stage26-8n-config/v1",
        "stage_id": "xunce-stage26-8r-pose-gate-repair",
        "recommended_combo_id": "theta3_iter100",
        "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
        "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 3,
        "hybrid_astar_max_iterations": 100,
        "trainable_transition_count": 120,
        "min_trainable_transition_count": 100,
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": 0,
        "selected_pose_unreachable_terminal_count": 0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    path = tmp_path / name
    _write_json(path, payload)
    return path


def _write_recommended_config(tmp_path: Path, *, name: str = "stage26_8q_recommended.json", **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8q-recommended-stage26-8n-config/v1",
        "recommended_combo_id": "aligned_1m_proxy",
        "aligned_trainable_transition_count": 120,
        "min_trainable_transition_count": 100,
        "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
        "planner_grid_resolution_m": 1.0,
        "hybrid_astar_closed_key_xy_resolution_m": 1.0,
        "hybrid_astar_primitive_duration_s": 1.0,
        "hybrid_astar_goal_position_tolerance_m": 1.0,
        "hybrid_astar_goal_theta_tolerance_deg": 45.0,
        "hybrid_astar_integration_dt_s": 0.25,
        "hybrid_astar_max_speed_mps": 0.5,
        "hybrid_astar_max_angular_speed_degps": 45.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    path = tmp_path / name
    _write_json(path, payload)
    return path


def _write_stage26_8i_root(
    tmp_path: Path,
    *,
    next_required_change: str = "increase_stage26_synthetic_update_strength_or_sample_count",
) -> Path:
    root = tmp_path / "stage26_8i"
    _write_json(
        root / "xunce-stage26-8i-summary.json",
        {
            "schema_version": "xunce-stage26-8i-summary/v1",
            "stage_id": "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair",
            "status": "failed",
            "next_required_change": next_required_change,
            "stable_combo_count": 5,
            "selected_action_changed_count": 0,
            "strong_state_join_available_count": 48,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return root


def _write_8m_pending_summary(root: Path) -> dict:
    summary = {
        "schema_version": "xunce-stage26-8m-summary/v1",
        "stage_id": "xunce-stage26-8m-generalized-resumable-training-pipeline",
        "status": "passed",
        "next_required_change": "continue_stage26_8m_jobs",
        "pending_job_count": 5,
        "completed_job_count": 0,
        "failed_job_count": 0,
    }
    _write_json(root / "xunce-stage26-8m-summary.json", summary)
    return summary


def _write_8m_state(root: Path, *, collector_root: Path, collector_status: str) -> None:
    rows = []
    for combo in ("policy_amp_repro", "depth16_lr3e5_policy3"):
        rows.append(
            {
                "schema_version": "xunce-stage26-8m-job-state/v1",
                "job_id": f"h16_s260801_sc6_cr20_er20_u_{combo}",
                "phase": "collector",
                "status": collector_status,
                "output_root": str(collector_root),
                "summary_path": str(collector_root / "xunce-stage26-1-summary.json"),
                "blocking_reason": "",
            }
        )
        rows.append(
            {
                "schema_version": "xunce-stage26-8m-job-state/v1",
                "job_id": f"h16_s260801_sc6_cr20_er20_u_{combo}",
                "phase": "update",
                "status": "pending",
                "output_root": str(root / "jobs" / f"h16_s260801_sc6_cr20_er20_u_{combo}" / "update"),
                "summary_path": str(root / "jobs" / f"h16_s260801_sc6_cr20_er20_u_{combo}" / "update" / "xunce-stage26-2-summary.json"),
                "blocking_reason": "summary_missing",
            }
        )
    _write_jsonl(root / "xunce-stage26-8m-job-state.jsonl", rows)


def _update_row(combo_id: str, *, stable: bool, parameter_delta_l2: float, policy_kl: float) -> dict:
    return {
        "job_id": combo_id,
        "combo_id": combo_id,
        "phase": "update",
        "status": "complete" if stable else "failed",
        "stage21_4_status": "passed" if stable else "failed",
        "loss_finite": stable,
        "gradient_finite": stable,
        "final_post_update_policy_approx_kl": policy_kl,
        "parameter_delta_l2": parameter_delta_l2,
        "checkpoint_exists": stable,
        "checkpoint_reload_passed": stable,
        "experimental_only": stable,
        "checkpoint_boundary_passed": stable,
        "stage21_3_lineage_passed": stable,
        "source_checkpoint_sha_consistent": stable,
        "collector_source_checkpoint_match": stable,
        "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
        "kl_gate_source": "policy_old_logprob_when_available/v1",
        "behavior_policy_kl_diagnostic_only": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
