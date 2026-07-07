from __future__ import annotations

import json
from pathlib import Path

from scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation import (
    ROUTE_BOUNDARY,
    ROUTE_EXECUTION,
    ROUTE_HARD_RISK,
    ROUTE_POST_UNREACHABLE,
    ROUTE_PRE_UNREACHABLE,
    ROUTE_REPAIR,
    ROUTE_STAGE21_6,
    run_xunce_stage21_5_post_update_offline_trajectory_evaluation,
)


def test_stage21_5_routes_to_stage21_6_when_post_update_does_not_regress(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4", route="implement_stage21_5_single_seed_ppo_pilot")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE21_6
    assert summary["sample_count_too_low_for_performance_claim"] is True
    assert summary["training_or_release_authorized"] is False
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False
    assert summary["connects_real_executor"] is False
    assert summary["starts_online_canary"] is False


def test_stage21_5_accepts_explicit_offline_evaluation_route_alias(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4", route="implement_stage21_5_post_update_offline_trajectory_evaluation")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.40, auc=1.10)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE21_6


def test_stage21_5_xunce_only_uses_episode_count_when_pair_scenario_count_is_zero(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, scenario_count=0, incumbent_checkpoint_loaded=False)
    _write_eval_root(root / "post", final=0.41, auc=1.11, scenario_count=0, incumbent_checkpoint_loaded=False)
    config = _write_config(root, xunce_only_evaluation=True, include_oracle_baselines=False)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE21_6
    assert "pre_scenario_count_short" not in summary["reason_codes"]
    assert "post_scenario_count_short" not in summary["reason_codes"]
    assert "pre_incumbent_checkpoint_not_loaded" not in summary["reason_codes"]
    assert "post_incumbent_checkpoint_not_loaded" not in summary["reason_codes"]


def test_stage21_5_rejects_final_coverage_regression_even_when_auc_is_flat(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.39, auc=1.10)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REPAIR
    assert "post_ppo_coverage_or_auc_regressed" in summary["reason_codes"]


def test_stage21_5_rejects_auc_regression_even_when_final_coverage_is_flat(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.40, auc=1.09)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REPAIR
    assert "post_ppo_coverage_or_auc_regressed" in summary["reason_codes"]


def test_stage21_5_efficiency_metric_ignores_auc_regression_when_efficiency_improves(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, path_cost_total_m=100.0, coverage_per_100m=40.0)
    _write_eval_root(root / "post", final=0.41, auc=1.09, path_cost_total_m=90.0, coverage_per_100m=45.0)
    config = _write_config(root, post_update_success_metric="main_coverable_coverage_efficiency/v1")

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE21_6
    assert summary["post_update_success_metric"] == "main_coverable_coverage_efficiency/v1"
    assert summary["coverage_curve_auc_delta"] < 0
    assert summary["coverage_per_100m_delta"] > 0
    assert "post_ppo_coverage_or_auc_regressed" not in summary["reason_codes"]


def test_stage21_5_rejects_single_scenario_regression_even_when_mean_is_flat(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final_by_scenario={"scenario-a": 0.40, "scenario-b": 0.40}, auc_by_scenario={"scenario-a": 1.10, "scenario-b": 1.10})
    _write_eval_root(root / "post", final_by_scenario={"scenario-a": 0.39, "scenario-b": 0.41}, auc_by_scenario={"scenario-a": 1.10, "scenario-b": 1.10})
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REPAIR
    assert summary["scenario_regression_count"] == 1


def test_stage21_5_efficiency_metric_scenario_regression_uses_per_100m(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(
        root / "pre",
        final_by_scenario={"scenario-a": 0.40, "scenario-b": 0.40},
        auc_by_scenario={"scenario-a": 1.10, "scenario-b": 1.10},
        coverage_per_100m_by_scenario={"scenario-a": 20.0, "scenario-b": 20.0},
    )
    _write_eval_root(
        root / "post",
        final_by_scenario={"scenario-a": 0.39, "scenario-b": 0.43},
        auc_by_scenario={"scenario-a": 1.08, "scenario-b": 1.11},
        coverage_per_100m_by_scenario={"scenario-a": 21.0, "scenario-b": 23.0},
    )
    config = _write_config(root, post_update_success_metric="main_coverable_coverage_efficiency/v1")

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["scenario_regression_count"] == 0
    assert summary["coverage_per_100m_delta"] > 0


def test_stage21_5_inherits_continuous_theta_head_seed_from_collector_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation as s21

    root = _fixture_root(tmp_path)
    stage21_4_root = root / "stage21_4"
    stage21_3_root = root / "stage21_3"
    stage21_1_root = root / "stage21_1"
    _write_stage21_4(stage21_4_root)
    stage21_3_root.mkdir()
    stage21_1_root.mkdir()
    stage21_4_summary_path = stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json"
    stage21_4_summary = json.loads(stage21_4_summary_path.read_text(encoding="utf-8"))
    stage21_4_summary["stage21_3_ppo_batch_validation_root"] = str(stage21_3_root)
    stage21_4_summary_path.write_text(json.dumps(stage21_4_summary), encoding="utf-8")
    (stage21_3_root / "xunce-stage21-3-ppo-batch-validation-summary.json").write_text(
        json.dumps({"stage21_1_collector_root": str(stage21_1_root)}),
        encoding="utf-8",
    )
    (stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json").write_text(
        json.dumps({"sampling_seed": 2101}),
        encoding="utf-8",
    )
    captured: list[dict[str, object]] = []

    def fake_high_fidelity(*, config_path: Path, output_root: Path, repo_root: Path, config_overrides: dict) -> dict:
        captured.append(dict(config_overrides))
        _write_eval_root(output_root, final=0.40 + 0.01 * len(captured), auc=1.10 + 0.01 * len(captured))
        return {"status": "passed"}

    monkeypatch.setattr(s21, "run_xunce_high_fidelity_exploration_coverage_comparison", fake_high_fidelity)
    config = _write_config(
        root,
        execute_high_fidelity_evaluations=True,
        pre_ppo_evaluation_root="",
        post_ppo_evaluation_root="",
    )

    summary = s21.run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["continuous_theta_head_init_seed"] == 2101
    assert [call["continuous_theta_head_init_seed"] for call in captured] == [2101, 2101]


def test_stage21_5_rejects_post_update_hard_risk_violation(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.41, auc=1.11, hard_risk_violation_count=1)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_HARD_RISK
    assert "post_ppo_hard_risk_or_safety_boundary_regression_detected" in summary["reason_codes"]


def test_stage21_5_rejects_mismatched_pre_post_scenarios(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, scenario_ids=("scenario-a", "scenario-b"))
    _write_eval_root(root / "post", final=0.41, auc=1.11, scenario_ids=("scenario-a", "scenario-c"))
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_EXECUTION
    assert "pre_post_xunce_scenario_id_mismatch" in summary["reason_codes"]


def test_stage21_5_routes_pre_unreachable_selected_separately(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, unreachable_selected_count=1)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_PRE_UNREACHABLE
    assert summary["pre_unreachable_selected_count"] == 2
    assert summary["post_unreachable_selected_count"] == 0
    assert "pre_unreachable_selected_count_nonzero" in summary["reason_codes"]


def test_stage21_5_routes_post_unreachable_selected_as_regression(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12, unreachable_selected_count=1)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POST_UNREACHABLE
    assert summary["pre_unreachable_selected_count"] == 0
    assert summary["post_unreachable_selected_count"] == 2
    assert "post_unreachable_selected_count_nonzero" in summary["reason_codes"]


def test_stage21_5_rejects_missing_selected_reachability_provenance_when_hard_gate_enabled(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, model_inference_rows=_model_inference_rows())
    _write_eval_root(
        root / "post",
        final=0.42,
        auc=1.12,
        model_inference_rows=_model_inference_rows(omit_provenance=True),
    )
    config = _write_config(
        root,
        candidate_reachability_gate_source="hybrid_astar_pose_reachability/v1",
    )

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POST_UNREACHABLE
    assert summary["pre_selected_reachability_provenance_invalid_count"] == 0
    assert summary["post_selected_reachability_provenance_invalid_count"] == 2
    assert "post_selected_reachability_provenance_invalid_count_nonzero" in summary["reason_codes"]


def test_stage21_5_excludes_no_valid_action_terminal_from_selected_reachability_audit(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, model_inference_rows=_model_inference_rows())
    rows = _model_inference_rows()
    rows.append(
        {
            "policy": "xunce",
            "scenario_id": "scenario-terminal",
            "step_index": 7,
            "true_model_inference_executed": False,
            "inference_skipped_reason": "no_valid_action",
            "terminal_reason": "no_valid_action",
            "episode_termination_reason": "no_valid_action",
            "reason_codes": ["no_valid_action"],
        }
    )
    _write_eval_root(root / "post", final=0.42, auc=1.12, model_inference_rows=rows)
    config = _write_config(
        root,
        candidate_reachability_gate_source="hybrid_astar_pose_reachability/v1",
    )

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["post_selected_reachability_audited_count"] == 2
    assert summary["post_selected_reachability_provenance_pass_count"] == 2
    assert summary["post_selected_reachability_provenance_invalid_count"] == 0
    assert "post_selected_reachability_provenance_invalid_count_nonzero" not in summary["reason_codes"]


def test_stage21_5_does_not_exclude_executed_selected_row_with_no_valid_action_marker(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, model_inference_rows=_model_inference_rows())
    rows = _model_inference_rows()
    rows[0].pop("selected_candidate_reachability_provenance")
    rows[0]["true_model_inference_executed"] = True
    rows[0]["selected_action_index"] = 0
    rows[0]["inference_skipped_reason"] = None
    rows[0]["reason_codes"] = ["no_valid_action"]
    _write_eval_root(root / "post", final=0.42, auc=1.12, model_inference_rows=rows)
    config = _write_config(
        root,
        candidate_reachability_gate_source="hybrid_astar_pose_reachability/v1",
    )

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["post_selected_reachability_audited_count"] == 2
    assert summary["post_selected_reachability_provenance_invalid_count"] == 1
    assert "post_selected_reachability_provenance_invalid_count_nonzero" in summary["reason_codes"]


def test_stage21_5_candidate_index_mismatch_invalid_sample_includes_binding_diagnostics(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, model_inference_rows=_model_inference_rows())
    rows = _model_inference_rows()
    rows[0]["selected_action_index"] = 2
    rows[0]["selected_base_candidate_index"] = 2
    rows[0]["selected_candidate_reachability_provenance"] = _selected_reachability_provenance(
        candidate_index=0,
        candidate_set_hash="set-0",
    )
    _write_eval_root(root / "post", final=0.42, auc=1.12, model_inference_rows=rows)
    config = _write_config(
        root,
        candidate_reachability_gate_source="hybrid_astar_pose_reachability/v1",
    )

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POST_UNREACHABLE
    assert summary["post_selected_reachability_candidate_index_mismatch_count"] == 1
    audit = json.loads((root / "out" / "xunce-stage21-5-selected-pose-evidence-audit.json").read_text(encoding="utf-8"))
    post_sample = audit["post"]["invalid_samples"][0]
    assert "selected_reachability_candidate_index_mismatch" in post_sample["reason_codes"]
    assert post_sample["row_selected_action_index"] == 2
    assert post_sample["row_selected_base_candidate_index"] == 2
    assert post_sample["provenance_candidate_index"] == 0
    assert post_sample["candidate_set_hash_match"] is True
    assert post_sample["planner_hash_match"] is True


def test_stage21_5_allows_selected_base_candidate_set_hash_to_match_provenance_hash(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, model_inference_rows=_model_inference_rows())
    rows = _model_inference_rows()
    rows[0]["candidate_set_hash"] = "expanded-set-0"
    rows[0]["selected_base_candidate_set_hash"] = "set-0"
    rows[0]["base_candidate_set_hash"] = "unused-base-set-0"
    _write_eval_root(root / "post", final=0.42, auc=1.12, model_inference_rows=rows)
    config = _write_config(
        root,
        candidate_reachability_gate_source="hybrid_astar_pose_reachability/v1",
    )

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["post_selected_reachability_provenance_invalid_count"] == 0
    audit = json.loads((root / "out" / "xunce-stage21-5-selected-pose-evidence-audit.json").read_text(encoding="utf-8"))
    assert audit["post"]["selected_reachability_candidate_set_hash_mismatch_count"] == 0
    assert audit["post"]["selected_reachability_candidate_set_hash_conflict_count"] == 1
    assert (root / "out" / "selected_pose_audit.json").is_file()


def test_stage21_5_rejects_selected_base_candidate_set_hash_mismatch_even_when_fallback_matches(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, model_inference_rows=_model_inference_rows())
    rows = _model_inference_rows()
    rows[0]["selected_base_candidate_set_hash"] = "wrong-selected-base-set"
    rows[0]["candidate_set_hash"] = "set-0"
    _write_eval_root(root / "post", final=0.42, auc=1.12, model_inference_rows=rows)
    config = _write_config(
        root,
        candidate_reachability_gate_source="hybrid_astar_pose_reachability/v1",
    )

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["post_selected_reachability_provenance_invalid_count"] == 1
    audit = json.loads((root / "out" / "xunce-stage21-5-selected-pose-evidence-audit.json").read_text(encoding="utf-8"))
    post_sample = audit["post"]["invalid_samples"][0]
    assert "selected_reachability_candidate_set_hash_mismatch" in post_sample["reason_codes"]
    assert post_sample["candidate_set_hash_match"] is False
    assert post_sample["candidate_set_hash_preferred_field"] == "selected_base_candidate_set_hash"
    assert post_sample["candidate_set_hash_preferred_value"] == "wrong-selected-base-set"
    assert post_sample["candidate_set_hash_conflict"] is True


def test_stage21_5_int_or_none_rejects_non_integral_or_infinite_values() -> None:
    import scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5

    assert stage21_5._int_or_none(2.0) == 2
    assert stage21_5._int_or_none(1.7) is None
    assert stage21_5._int_or_none(float("inf")) is None


def test_stage21_5_rejects_missing_selected_reachability_binding_fields(tmp_path: Path) -> None:
    cases = (
        ("candidate_set_hash", "selected_reachability_candidate_set_hash_missing"),
        ("candidate_index", "selected_reachability_candidate_index_missing"),
        ("candidate_theta_deg", "selected_reachability_theta_missing"),
    )
    for field, expected_reason in cases:
        case_root = tmp_path / field
        case_root.mkdir()
        root = _fixture_root(case_root)
        _write_stage21_4(root / "stage21_4")
        _write_eval_root(root / "pre", final=0.40, auc=1.10, model_inference_rows=_model_inference_rows())
        rows = _model_inference_rows()
        rows[0]["selected_candidate_reachability_provenance"].pop(field)
        _write_eval_root(root / "post", final=0.42, auc=1.12, model_inference_rows=rows)
        config = _write_config(
            root,
            candidate_reachability_gate_source="hybrid_astar_pose_reachability/v1",
        )

        summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
            config_path=config,
            output_root=root / "out",
            repo_root=root,
        )

        assert summary["status"] == "failed"
        assert summary["post_selected_reachability_provenance_invalid_count"] == 1
        audit = json.loads((root / "out" / "xunce-stage21-5-selected-pose-evidence-audit.json").read_text(encoding="utf-8"))
        assert expected_reason in audit["post"]["reason_codes"]
        assert expected_reason in audit["post"]["invalid_samples"][0]["reason_codes"]


def test_stage21_5_prioritizes_post_unreachable_when_pre_and_post_unreachable(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10, unreachable_selected_count=1)
    _write_eval_root(root / "post", final=0.42, auc=1.12, unreachable_selected_count=1)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["next_required_change"] == ROUTE_POST_UNREACHABLE
    assert "pre_unreachable_selected_count_nonzero" in summary["reason_codes"]
    assert "post_unreachable_selected_count_nonzero" in summary["reason_codes"]


def test_stage21_5_rejects_stage21_4_routing_summary_mismatch(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4", routing_route="wrong_route")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage21_5_required_inputs"
    assert "stage21_4_routing_next_required_change_mismatch" in summary["reason_codes"]


def test_stage21_5_rejects_stage21_4_routing_status_mismatch(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4", routing_status="failed")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage21_5_required_inputs"
    assert "stage21_4_routing_status_mismatch" in summary["reason_codes"]


def test_stage21_5_rejects_stage21_4_checkpoint_audit_path_mismatch(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4", audit_checkpoint_path=str(root / "stage21_4" / "other.pt"))
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage21_5_required_inputs"
    assert "stage21_4_checkpoint_audit_path_mismatch" in summary["reason_codes"]


def test_stage21_5_rejects_stage21_4_checkpoint_audit_hash_mismatch(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4", audit_checkpoint_sha256="different")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage21_5_required_inputs"
    assert "stage21_4_checkpoint_audit_sha256_mismatch" in summary["reason_codes"]


def test_stage21_5_rejects_stage21_4_checkpoint_audit_hash_missing(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4", omit_audit_checkpoint_sha256=True)
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage21_5_required_inputs"
    assert "stage21_4_checkpoint_audit_sha256_missing" in summary["reason_codes"]


def test_stage21_5_boundary_flag_hard_fails(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_4(root / "stage21_4")
    _write_eval_root(root / "pre", final=0.40, auc=1.10)
    _write_eval_root(root / "post", final=0.42, auc=1.12)
    config = _write_config(root, publishes_checkpoint=True)

    summary = run_xunce_stage21_5_post_update_offline_trajectory_evaluation(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "publishes_checkpoint" in summary["reason_codes"]


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    (root / "configs" / "hf.json").write_text("{}", encoding="utf-8")
    (root / "configs" / "profile.json").write_text("{}", encoding="utf-8")
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-5-post-update-offline-trajectory-evaluation-config/v1",
        "stage21_4_tiny_ppo_update_smoke_root": str(root / "stage21_4"),
        "high_fidelity_config": str(root / "configs" / "hf.json"),
        "execute_high_fidelity_evaluations": False,
        "pre_ppo_evaluation_root": str(root / "pre"),
        "post_ppo_evaluation_root": str(root / "post"),
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "include_canonical_reward_rerank_oracle": True,
        "canonical_reward_rerank_profile": str(root / "configs" / "profile.json"),
        "emit_candidate_metric_audit": True,
        "min_post_update_coverage_delta": 0.0,
        "stage21_5_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_stage21_4(
    root: Path,
    *,
    route: str = "implement_stage21_5_single_seed_ppo_pilot",
    routing_status: str = "passed",
    routing_route: str | None = None,
    audit_checkpoint_path: str | None = None,
    audit_checkpoint_sha256: str = "abc123",
    omit_audit_checkpoint_sha256: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    experimental_checkpoint_path = str(root / "experimental.pt")
    summary = {
        "status": "passed",
        "next_required_change": route,
        "source_xunce_candidate_checkpoint": str(root / "source.pt"),
        "experimental_checkpoint_path": experimental_checkpoint_path,
        "experimental_checkpoint_sha256": "abc123",
        "sample_count_too_low_for_performance_claim": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    checkpoint_audit = {
        "checkpoint_reload_passed": True,
        "experimental_checkpoint_path": audit_checkpoint_path or experimental_checkpoint_path,
        "metadata": {
            "experimental_only": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        },
        "reload_audit": {
            "checkpoint_path": audit_checkpoint_path or experimental_checkpoint_path,
        },
    }
    if not omit_audit_checkpoint_sha256:
        checkpoint_audit["experimental_checkpoint_sha256"] = audit_checkpoint_sha256
        checkpoint_audit["reload_audit"]["checkpoint_sha256"] = audit_checkpoint_sha256
    routing = {
        "status": routing_status,
        "next_required_change": routing_route or route,
    }
    (root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "xunce-stage21-4-checkpoint-audit.json").write_text(json.dumps(checkpoint_audit), encoding="utf-8")
    (root / "xunce-stage21-4-next-stage-routing.json").write_text(json.dumps(routing), encoding="utf-8")


def _write_eval_root(
    root: Path,
    *,
    final: float = 0.40,
    auc: float = 1.10,
    path_cost_total_m: float = 100.0,
    coverage_per_100m: float | None = None,
    final_by_scenario: dict[str, float] | None = None,
    auc_by_scenario: dict[str, float] | None = None,
    coverage_per_100m_by_scenario: dict[str, float] | None = None,
    scenario_ids: tuple[str, str] = ("scenario-a", "scenario-b"),
    hard_risk_violation_count: int = 0,
    unreachable_selected_count: int = 0,
    scenario_count: int = 2,
    incumbent_checkpoint_loaded: bool = True,
    model_inference_rows: list[dict] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "failed",
        "xunce_checkpoint_loaded": True,
        "incumbent_checkpoint_loaded": incumbent_checkpoint_loaded,
        "true_model_inference_executed": True,
        "proxy_selection_used": False,
        "scenario_count": scenario_count,
        "rollout_steps": 4,
        "model_inference_failure_count": 0,
        "model_inference_mask_violation_count": 0,
        "open_grid_fallback_count": 0,
        "unreachable_selected_count": unreachable_selected_count,
        "path_planning_failure_count": 0,
    }
    rows = []
    for scenario_id in scenario_ids:
        scenario_final = (final_by_scenario or {}).get(scenario_id, final)
        scenario_auc = (auc_by_scenario or {}).get(scenario_id, auc)
        scenario_coverage_per_100m = (coverage_per_100m_by_scenario or {}).get(
            scenario_id,
            coverage_per_100m if coverage_per_100m is not None else scenario_final,
        )
        rows.append(
            {
                "policy": "xunce",
                "scenario_id": scenario_id,
                "final_coverage_rate": scenario_final,
                "final_coverage_rate_capped": scenario_final,
                "coverage_curve_auc": scenario_auc,
                "coverage_curve_auc_capped": scenario_auc,
                "coverage_return": scenario_auc,
                "new_covered_cell_count": int(scenario_final * 100),
                "path_cost_total_m": path_cost_total_m,
                "coverage_per_100m": scenario_coverage_per_100m,
                "soft_risk_exposure_total": 0.0,
                "hard_risk_violation_count": hard_risk_violation_count,
                "model_inference_failure_count": 0,
                "mask_violation_count": 0,
                "unreachable_selected_count": unreachable_selected_count,
                "path_planning_failure_count": 0,
                "open_grid_fallback_count": 0,
                "candidate_generation_exhausted_count": 0,
            }
        )
    (root / "xunce-exploration-coverage-comparison-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "xunce-exploration-coverage-episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    if model_inference_rows is not None:
        (root / "xunce-exploration-coverage-model-inference.jsonl").write_text(
            "\n".join(json.dumps(row) for row in model_inference_rows) + "\n",
            encoding="utf-8",
        )


def _model_inference_rows(*, omit_provenance: bool = False) -> list[dict]:
    rows: list[dict] = []
    for index, scenario_id in enumerate(("scenario-a", "scenario-b")):
        provenance = _selected_reachability_provenance(candidate_index=0, candidate_set_hash=f"set-{index}")
        row = {
            "policy": "xunce",
            "scenario_id": scenario_id,
            "step_index": 0,
            "true_model_inference_executed": True,
            "selected_action_index": 0,
            "candidate_set_hash": f"set-{index}",
            "candidate_theta_deg": 0.0,
            "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
            "candidate_reachability_planner_config_hash": "planner-hash",
            "selected_reachability_planner_config_hash": "planner-hash",
            "selected_reachability_provenance_valid": not omit_provenance,
            "reason_codes": [],
        }
        if not omit_provenance:
            row["selected_candidate_reachability_provenance"] = provenance
        rows.append(row)
    return rows


def _selected_reachability_provenance(*, candidate_index: int, candidate_set_hash: str) -> dict:
    return {
        "schema_version": "xunce-candidate-reachability-provenance/v1",
        "source": "hybrid_astar_pose_reachability/v1",
        "backend": "hybrid_astar_pose_path/v1",
        "candidate_index": candidate_index,
        "candidate_set_hash": candidate_set_hash,
        "candidate_theta_deg": 0.0,
        "planner_config_hash": "planner-hash",
        "reachable": True,
        "path_cost": 3.0,
        "pose_path_hash": "pose-hash",
    }
