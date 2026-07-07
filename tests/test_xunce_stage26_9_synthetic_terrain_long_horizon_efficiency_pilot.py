import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_9_rejects_unready_stage26_8n_summary(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path, status="failed")
    output_root = tmp_path / "out"

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    routing = json.loads((output_root / "xunce-stage26-9-routing.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_required_inputs"
    assert "stage26_8n_status_not_passed" in summary["input_rejections"]
    assert routing["release_or_training_authorized"] is False
    assert routing["publishes_checkpoint"] is False
    assert routing["replaces_default_policy"] is False
    assert routing["connects_real_executor"] is False
    assert routing["starts_online_canary"] is False
    assert routing["canary_traffic_fraction"] == 0.0


def test_stage26_9_generates_single_tuple_8m_configs_without_cross_product_and_runs_one_subjob(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    calls: list[dict] = []

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        calls.append({"config": payload, "output_root": output_root, "kwargs": kwargs})
        return _write_8m_summary(output_root, next_required_change="continue_stage26_8m_jobs")

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    output_root = tmp_path / "out"
    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    matrix = _read_jsonl(output_root / "xunce-stage26-9-long-horizon-matrix.jsonl")
    manifest = json.loads((output_root / "xunce-stage26-9-manifest.json").read_text(encoding="utf-8"))
    report = (output_root / "xunce-stage26-9-report.md").read_text(encoding="utf-8")

    assert summary["next_required_change"] == "continue_stage26_9_long_horizon_jobs"
    assert len(calls) == 1
    assert calls[0]["kwargs"]["run_mode_override"] == "run_next"
    assert calls[0]["kwargs"]["max_jobs_override"] == 1
    assert len(matrix) == 9
    for field in (
        "release_or_training_authorized",
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "starts_online_canary",
    ):
        assert manifest[field] is False
        assert all(row[field] is False for row in matrix)
        assert field in report
    assert manifest["canary_traffic_fraction"] == 0.0
    assert all(row["canary_traffic_fraction"] == 0.0 for row in matrix)
    assert "canary_traffic_fraction" in report
    assert manifest["source_scenario_fixture_root_is_legacy_readonly_input"] is True
    assert {(row["horizon"], row["rollout_steps"]) for row in matrix} == {(20, 20), (24, 24), (32, 32)}
    assert {row["seed"] for row in matrix} == {260801, 260802, 260803}
    assert all(Path(row["sub_config_path"]).is_file() for row in matrix)

    for row in matrix:
        sub_config = json.loads(Path(row["sub_config_path"]).read_text(encoding="utf-8"))
        assert sub_config["horizons"] == [row["horizon"]]
        assert sub_config["collector_rollout_steps"] == [row["rollout_steps"]]
        assert sub_config["eval_rollout_steps"] == [row["rollout_steps"]]
        assert sub_config["seeds"] == [row["seed"]]
        assert sub_config["scenario_counts"] == [11]
        assert [combo["combo_id"] for combo in sub_config["update_combos"]] == [
            "depth16_lr3e5_policy3",
            "lr5e5_policy3",
        ]


def test_stage26_9_transmits_stage26_8s_gate_and_stage26_8q_proxy_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(
        tmp_path,
        stage26_8s_recommended_scenario_count=13,
        candidate_reachability_max_theta_proposals_per_candidate=5,
        candidate_reachability_theta_proposal_policy="candidate_current_bearing_sweep/v1",
        hybrid_astar_max_iterations=250,
    )
    captured: dict = {}

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        captured.update(json.loads(config_path.read_text(encoding="utf-8")))
        return _write_8m_summary(output_root, next_required_change="continue_stage26_8m_jobs")

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_9_long_horizon_jobs"
    assert captured["scenario_counts"] == [13]
    assert captured["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert captured["candidate_reachability_max_theta_proposals_per_candidate"] == 5
    assert captured["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert captured["hybrid_astar_max_iterations"] == 250
    assert captured["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert captured["planner_grid_resolution_m"] == 1.0
    assert captured["hybrid_astar_closed_key_xy_resolution_m"] == 1.0
    assert captured["hybrid_astar_primitive_duration_s"] == 1.0
    assert captured["hybrid_astar_goal_position_tolerance_m"] == 1.0
    assert captured["hybrid_astar_goal_theta_tolerance_deg"] == 45.0
    assert captured["collector_reuse_policy"] == "by_horizon_seed_scenario_rollout/v1"
    assert captured["max_abs_approx_kl"] == 1.5


def test_stage26_9_routes_continue_passed_expand_credit_and_eval_safety() -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    assert s26._route([], [], [_matrix_row(20, 260801, status="pending")]) == "continue_stage26_9_long_horizon_jobs"

    passed_rows = [
        _matrix_row(20, 260801, delta=1.0),
        _matrix_row(20, 260802, delta=0.0),
        _matrix_row(20, 260803, delta=-0.5),
        _matrix_row(24, 260801, delta=0.2),
        _matrix_row(24, 260802, delta=0.1),
        _matrix_row(24, 260803, delta=-0.1),
        _matrix_row(32, 260801, delta=-0.2),
        _matrix_row(32, 260802, delta=-0.3),
        _matrix_row(32, 260803, delta=-0.4),
    ]
    assert s26._route([], [], passed_rows) == "review_stage26_9_long_horizon_efficiency_readiness"

    mixed_rows = [
        _matrix_row(20, 260801, delta=1.0),
        _matrix_row(20, 260802, delta=-1.0),
        _matrix_row(24, 260801, delta=0.0),
    ]
    assert s26._route([], [], mixed_rows) == "expand_stage26_9_seed_or_horizon_budget"

    credit_rows = [
        _matrix_row(20, 260801, delta=-1.0, selected_action_changed_count=1),
        _matrix_row(20, 260802, delta=-0.5, selected_action_changed_count=1),
        _matrix_row(24, 260801, delta=-0.2, selected_action_changed_count=1),
    ]
    assert s26._route([], [], credit_rows) == "repair_stage26_synthetic_credit_assignment"

    unsafe_rows = [_matrix_row(20, 260801, delta=1.0, mask_violation_count=1)]
    assert s26._route([], [], unsafe_rows) == "repair_stage26_9_eval_binding_or_safety"


def test_stage26_9_passes_through_no_valid_action_terminal_counts_without_eval_safety_route(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    for horizon, rollout_steps in ((20, 20), (24, 24), (32, 32)):
        for seed in (260801, 260802, 260803):
            _write_8m_summary(
                output_root / f"h{horizon}_s{seed}_r{rollout_steps}",
                next_required_change="run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot",
                completed_job_count=1,
                mean_main_coverage_per_100m_delta=1.0,
                pre_no_valid_action_terminal_count=1,
                post_no_valid_action_terminal_count=1,
                no_valid_action_terminal_count=2,
                pre_model_inference_skipped_terminal_count=1,
                post_model_inference_skipped_terminal_count=1,
                model_inference_skipped_terminal_count=2,
            )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    matrix = _read_jsonl(output_root / "xunce-stage26-9-long-horizon-matrix.jsonl")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "review_stage26_9_long_horizon_efficiency_readiness"
    assert summary["pre_no_valid_action_terminal_count"] == 9
    assert summary["post_no_valid_action_terminal_count"] == 9
    assert summary["no_valid_action_terminal_count"] == 18
    assert summary["model_inference_skipped_terminal_count"] == 18
    assert all(row["no_valid_action_terminal_count"] == 2 for row in matrix)
    assert all(row["model_inference_skipped_terminal_count"] == 2 for row in matrix)


def test_stage26_9_routes_required_inputs_when_selected_combo_ids_cannot_be_resolved(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path, selected_eval_job_ids=["h16_s260801_sc11_cr20_er20_u_missing_combo"])
    output_root = tmp_path / "out"

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    matrix = _read_jsonl(output_root / "xunce-stage26-9-long-horizon-matrix.jsonl")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_required_inputs"
    assert "selected_update_combo_ids_unresolved" in summary["input_rejections"]
    assert matrix
    assert not any(Path(row["sub_config_path"]).is_file() for row in matrix)


def test_stage26_9_rejects_non_fixed_matrix_config(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)

    with pytest.raises(ValueError, match="fixed matrix"):
        s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
            config_path=_write_config(
                tmp_path,
                stage26_8n_root,
                locked_tuples=[
                    {"horizon": 20, "rollout_steps": 20},
                    {"horizon": 24, "rollout_steps": 20},
                    {"horizon": 32, "rollout_steps": 32},
                ],
            ),
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )

    with pytest.raises(ValueError, match="fixed seeds"):
        s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
            config_path=_write_config(tmp_path, stage26_8n_root, seeds=[260801, 260802]),
            output_root=tmp_path / "out2",
            repo_root=REPO_ROOT,
        )


def test_stage26_9_rejects_non_boolean_run_stage26_8m(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)

    with pytest.raises(ValueError, match="run_stage26_8m must be a JSON boolean"):
        s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
            config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m="false"),
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )


def test_stage26_9_rejects_max_selected_update_combo_count_above_two(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)

    with pytest.raises(ValueError, match="max_selected_update_combo_count must be 2"):
        s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
            config_path=_write_config(tmp_path, stage26_8n_root, max_selected_update_combo_count=3),
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )


def test_stage26_9_rejects_stage26_8n_theta_and_planner_contract_violations(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(
        tmp_path,
        candidate_reachability_theta_proposal_policy="candidate_viewpoint_current_step/v1",
        candidate_reachability_max_theta_proposals_per_candidate=4,
        hybrid_astar_max_iterations=199,
        stage26_8q_planner_overrides={
            "hybrid_astar_planning_grid_source": "legacy/v1",
            "planner_grid_resolution_m": 2.0,
            "hybrid_astar_closed_key_xy_resolution_m": 1.0,
            "hybrid_astar_primitive_duration_s": 1.0,
            "hybrid_astar_goal_position_tolerance_m": 1.0,
            "hybrid_astar_goal_theta_tolerance_deg": 45.0,
        },
    )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_required_inputs"
    assert "stage26_8n_candidate_reachability_theta_proposal_policy_mismatch" in summary["input_rejections"]
    assert "stage26_8n_candidate_reachability_max_theta_proposals_per_candidate_below_min" in summary["input_rejections"]
    assert "stage26_8n_hybrid_astar_max_iterations_below_min" in summary["input_rejections"]
    assert "stage26_8n_stage26_8q_planner_overrides_hybrid_astar_planning_grid_source_mismatch" in summary["input_rejections"]
    assert "stage26_8n_stage26_8q_planner_overrides_planner_grid_resolution_m_mismatch" in summary["input_rejections"]


def test_stage26_9_rejects_missing_stage26_8n_required_zero_and_boundary_fields(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    summary_path = stage26_8n_root / "xunce-stage26-8n-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for field in (
        "synthetic_inference_required_field_missing_count",
        "hybrid_path_missing_provenance_count",
        "hybrid_path_contract_mismatch_count",
        "explicit_unreachable_selected_provenance_count",
        "pre_selected_reachability_provenance_invalid_count",
        "post_selected_reachability_provenance_invalid_count",
        "pre_unreachable_selected_count",
        "post_unreachable_selected_count",
        "release_or_training_authorized",
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "starts_online_canary",
        "canary_traffic_fraction",
    ):
        payload.pop(field)
    _write_json(summary_path, payload)

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_required_inputs"
    assert "stage26_8n_synthetic_inference_required_field_missing_count_missing" in summary["input_rejections"]
    assert "stage26_8n_hybrid_path_missing_provenance_count_missing" in summary["input_rejections"]
    assert "stage26_8n_hybrid_path_contract_mismatch_count_missing" in summary["input_rejections"]
    assert "stage26_8n_explicit_unreachable_selected_provenance_count_missing" in summary["input_rejections"]
    assert "stage26_8n_pre_selected_reachability_provenance_invalid_count_missing" in summary["input_rejections"]
    assert "stage26_8n_post_selected_reachability_provenance_invalid_count_missing" in summary["input_rejections"]
    assert "stage26_8n_release_or_training_authorized_missing" in summary["input_rejections"]
    assert "stage26_8n_canary_traffic_fraction_missing" in summary["input_rejections"]


def test_stage26_9_rejects_non_false_stage26_8n_boundary_fields(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(
        tmp_path,
        publishes_checkpoint=True,
        canary_traffic_fraction=0.1,
    )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_required_inputs"
    assert "stage26_8n_publishes_checkpoint_not_false" in summary["input_rejections"]
    assert "stage26_8n_canary_traffic_fraction_not_zero" in summary["input_rejections"]


def test_stage26_9_routes_failed_8m_subjob_to_eval_safety(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    _write_8m_summary(
        output_root / "h20_s260801_r20",
        status="failed",
        next_required_change="expand_stage26_9_seed_or_horizon_budget",
        failed_job_count=1,
    )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_eval_binding_or_safety"


def test_stage26_9_routes_wrong_8m_summary_schema_to_eval_safety(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    for horizon, rollout_steps in ((20, 20), (24, 24), (32, 32)):
        for seed in (260801, 260802, 260803):
            _write_8m_summary(
                output_root / f"h{horizon}_s{seed}_r{rollout_steps}",
                schema_version="wrong/v1",
                stage_id="wrong-stage",
                next_required_change="review_stage26_9_long_horizon_efficiency_readiness",
                completed_job_count=1,
                mean_main_coverage_per_100m_delta=1.0,
            )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_eval_binding_or_safety"


def test_stage26_9_handles_bad_integer_fields_as_eval_safety_risk(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    _write_8m_summary(
        output_root / "h20_s260801_r20",
        next_required_change="expand_stage26_9_seed_or_horizon_budget",
        pending_job_count="bad",
        completed_job_count="bad",
        failed_job_count="bad",
        selected_action_changed_count="bad",
        mask_violation_count="bad",
    )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_eval_binding_or_safety"


def test_stage26_9_routes_unknown_8m_summary_status_or_route_to_eval_safety(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    _write_8m_summary(
        output_root / "h20_s260801_r20",
        status="mystery",
        next_required_change="unknown_next_step",
        pending_job_count=0,
        completed_job_count=1,
        failed_job_count=0,
        mean_main_coverage_per_100m_delta=1.0,
    )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_eval_binding_or_safety"


def test_stage26_9_routes_corrupt_8m_summary_to_eval_safety(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    corrupt_summary = output_root / "h20_s260801_r20" / "xunce-stage26-8m-summary.json"
    corrupt_summary.parent.mkdir(parents=True, exist_ok=True)
    corrupt_summary.write_text("{", encoding="utf-8")

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    matrix = _read_jsonl(output_root / "xunce-stage26-9-long-horizon-matrix.jsonl")
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_eval_binding_or_safety"
    assert matrix[0]["summary_contract_failure"] == "stage26_8m_summary_unreadable"


def test_stage26_9_routes_corrupt_8m_job_state_to_eval_safety(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    sub_root = output_root / "h20_s260801_r20"
    _write_8m_summary(
        sub_root,
        next_required_change="run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot",
        pending_job_count=0,
        completed_job_count=1,
        failed_job_count=0,
        mean_main_coverage_per_100m_delta=1.0,
    )
    (sub_root / "xunce-stage26-8m-job-state.jsonl").write_text("{bad jsonl}\n", encoding="utf-8")

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    matrix = _read_jsonl(output_root / "xunce-stage26-9-long-horizon-matrix.jsonl")
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_eval_binding_or_safety"
    assert matrix[0]["summary_contract_failure"] == "stage26_8m_job_state_unreadable"


def test_stage26_9_routes_clean_8m_credit_subjobs_to_credit_assignment(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    output_root = tmp_path / "out"
    for horizon in (20, 24, 32):
        for seed in (260801, 260802, 260803):
            sub_root = output_root / f"h{horizon}_s{seed}_r{horizon}"
            _write_8m_summary(
                sub_root,
                status="failed",
                next_required_change="repair_stage26_synthetic_credit_assignment",
                pending_job_count=0,
                completed_job_count=1,
                failed_job_count=0,
                mean_main_coverage_per_100m_delta=1.0,
            )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_synthetic_credit_assignment"


def test_stage26_9_routes_corrupt_stage26_8n_summary_to_required_inputs(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(tmp_path)
    (stage26_8n_root / "xunce-stage26-8n-summary.json").write_text("{", encoding="utf-8")
    output_root = tmp_path / "out"

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_required_inputs"
    assert "stage26_8n_summary_unreadable" in summary["input_rejections"]
    assert (output_root / "xunce-stage26-9-summary.json").is_file()


def test_stage26_9_rejects_non_hybrid_astar_pose_gate_from_stage26_8n(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as s26

    stage26_8n_root = _write_stage26_8n_root(
        tmp_path,
        candidate_reachability_gate_source="legacy_action_mask_validation/v1",
    )

    summary = s26.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
        config_path=_write_config(tmp_path, stage26_8n_root, run_stage26_8m=False),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_9_required_inputs"
    assert "stage26_8n_candidate_reachability_gate_source_mismatch" in summary["input_rejections"]


def test_stage26_9_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    assert "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot" in registry["stages"]


def _write_config(tmp_path: Path, stage26_8n_root: Path, **overrides) -> Path:
    fixture_root = tmp_path / "scenario-fixtures"
    fixture_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot-config/v1",
        "stage_id": "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot",
        "stage26_8n_root": str(stage26_8n_root),
        "source_scenario_fixture_root": str(fixture_root),
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


def _write_stage26_8n_root(tmp_path: Path, **overrides) -> Path:
    root = tmp_path / "stage26_8n"
    payload = {
        "schema_version": "xunce-stage26-8n-summary/v1",
        "stage_id": "xunce-stage26-8n-aggressive-sample-update-sweep",
        "status": "passed",
        "next_required_change": "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot",
        "trainable_transition_count": 140,
        "min_trainable_transition_count": 100,
        "selected_eval_job_ids": [
            "h16_s260801_sc11_cr20_er20_u_depth16_lr3e5_policy3",
            "h16_s260801_sc11_cr20_er20_u_lr5e5_policy3",
        ],
        "update_combos": _update_combos(),
        "stage26_8s_recommended_scenario_count": 11,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 5,
        "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
        "hybrid_astar_max_iterations": 200,
        "stage26_8q_planner_overrides": {
            "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
            "planner_grid_resolution_m": 1.0,
            "hybrid_astar_closed_key_xy_resolution_m": 1.0,
            "hybrid_astar_primitive_duration_s": 1.0,
            "hybrid_astar_goal_position_tolerance_m": 1.0,
            "hybrid_astar_goal_theta_tolerance_deg": 45.0,
        },
        "max_abs_approx_kl": 1.5,
        "synthetic_inference_required_field_missing_count": 0,
        "hybrid_path_missing_provenance_count": 0,
        "hybrid_path_contract_mismatch_count": 0,
        "explicit_unreachable_selected_provenance_count": 0,
        "pre_selected_reachability_provenance_invalid_count": 0,
        "post_selected_reachability_provenance_invalid_count": 0,
        "pre_unreachable_selected_count": 0,
        "post_unreachable_selected_count": 0,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    _write_json(root / "xunce-stage26-8n-summary.json", payload)
    return root


def _write_8m_summary(root: Path, *, next_required_change: str, **overrides) -> dict:
    payload = {
        "schema_version": "xunce-stage26-8m-summary/v1",
        "stage_id": "xunce-stage26-8m-generalized-resumable-training-pipeline",
        "status": "passed",
        "next_required_change": next_required_change,
        "pending_job_count": 1 if next_required_change == "continue_stage26_8m_jobs" else 0,
        "completed_job_count": 0,
        "failed_job_count": 0,
        "mean_main_coverage_per_100m_delta": 0.0,
    } | overrides
    _write_json(root / "xunce-stage26-8m-summary.json", payload)
    return payload


def _matrix_row(horizon: int, seed: int, *, status: str = "complete", delta: float = 0.0, **overrides) -> dict:
    return {
        "horizon": horizon,
        "seed": seed,
        "status": status,
        "stage26_8m_status": "passed",
        "stage26_8m_next_required_change": "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot",
        "main_coverage_per_100m_delta": delta,
        "selected_action_changed_count": 0,
        "lineage_mismatch": False,
        "binding_or_safety_failure": False,
    } | overrides


def _update_combos() -> list[dict]:
    return [
        {"combo_id": "depth16_lr3e5_policy3", "epochs": 16, "learning_rate": 3.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.005, "loss_scale": 0.5},
        {"combo_id": "lr5e5_policy3", "epochs": 16, "learning_rate": 5.0e-5, "policy_loss_coefficient": 3.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.003, "loss_scale": 0.5},
        {"combo_id": "unused", "epochs": 8, "learning_rate": 2.0e-5, "policy_loss_coefficient": 2.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.005, "loss_scale": 0.5},
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
