from __future__ import annotations

import json
from pathlib import Path

from scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation import (
    ROUTE_BOUNDARY,
    ROUTE_EXECUTION,
    ROUTE_HARD_RISK,
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
    root.mkdir(parents=True)
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
    final_by_scenario: dict[str, float] | None = None,
    auc_by_scenario: dict[str, float] | None = None,
    scenario_ids: tuple[str, str] = ("scenario-a", "scenario-b"),
    hard_risk_violation_count: int = 0,
) -> None:
    root.mkdir(parents=True)
    summary = {
        "status": "failed",
        "xunce_checkpoint_loaded": True,
        "incumbent_checkpoint_loaded": True,
        "true_model_inference_executed": True,
        "proxy_selection_used": False,
        "scenario_count": 2,
        "rollout_steps": 4,
        "model_inference_failure_count": 0,
        "model_inference_mask_violation_count": 0,
        "open_grid_fallback_count": 0,
        "unreachable_selected_count": 0,
        "path_planning_failure_count": 0,
    }
    rows = []
    for scenario_id in scenario_ids:
        scenario_final = (final_by_scenario or {}).get(scenario_id, final)
        scenario_auc = (auc_by_scenario or {}).get(scenario_id, auc)
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
                "path_cost_total_m": 100.0,
                "coverage_per_100m": scenario_final,
                "soft_risk_exposure_total": 0.0,
                "hard_risk_violation_count": hard_risk_violation_count,
                "model_inference_failure_count": 0,
                "mask_violation_count": 0,
                "unreachable_selected_count": 0,
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
