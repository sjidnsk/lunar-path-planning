import json
import sys
from pathlib import Path

import pytest


def _import_runner(repo_root: Path):
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from scripts.run_xunce_stage18_5_evidence_attribution_review import (  # noqa: PLC0415
        run_xunce_stage18_5_evidence_attribution_review,
    )

    return run_xunce_stage18_5_evidence_attribution_review


def test_cost_and_risk_within_budget_do_not_fail_guard_but_remain_attributed(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(tmp_path, coverage_delta=12.0, path_cost_delta=4.0, risk_delta=0.2)
    config_path = _write_config(tmp_path, coverage_root)

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=repo_root,
    )

    assert summary["status"] == "passed"
    assert summary["guard_evaluation"]["passed"] is True
    assert summary["guard_evaluation"]["checks"]["coverage_gain_per_path_cost_delta_audit_only"] is True
    assert summary["next_required_change"] == "prepare_stage19_evaluator_critic_preflight"
    assert summary["attribution_class_counts"]["path_cost"] >= 1
    assert summary["attribution_class_counts"]["risk_proxy"] >= 1
    assert summary["regression_count_summary"]["source_efficiency_regression_count"] == 1
    assert summary["regression_count_summary"]["source_safety_regression_count"] == 1
    assert summary["regression_count_summary"]["efficiency_regression_attributed_count"] == 1
    assert summary["regression_count_summary"]["safety_regression_attributed_count"] == 1
    assert summary["stage19_readiness"]["readiness"] == "ready_for_stage19_preflight_human_review_only"
    assert summary["stage19_readiness"]["authorized"] is False
    assert summary["profile_hash"]

    assert (tmp_path / "out" / "xunce-stage18-5-evidence-attribution-summary.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-5-regression-attribution.jsonl").is_file()
    assert (tmp_path / "out" / "xunce-stage18-5-paired-decision-summary.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-5-guard-evaluation.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-5-next-stage-routing.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-5-evidence-attribution-report.md").is_file()
    assert (tmp_path / "out" / "xunce-stage18-5-manifest.json").is_file()


def test_cost_or_risk_over_budget_fails_guard_and_routes_to_refinement(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=12.0,
        path_cost_delta=24.0,
        risk_delta=0.6,
        risk_cost_weighted_delta=26.0,
    )
    config_path = _write_config(tmp_path, coverage_root)

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out-over-budget",
        repo_root=repo_root,
    )

    assert summary["status"] == "passed"
    assert summary["guard_evaluation"]["passed"] is False
    assert "path_cost_regression" in summary["guard_evaluation"]["failed_guards"]
    assert "risk_regression" in summary["guard_evaluation"]["failed_guards"]
    assert summary["next_required_change"] == "refine_coverage_reward_and_cost_guard"
    assert summary["next_stage_routing"]["primary_route"] == "refine_coverage_reward_and_cost_guard"
    assert summary["stage19_readiness"]["authorized"] is False


def test_v3_risk_delta_is_audit_only_and_soft_risk_exposure_is_guarded(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=12.0,
        path_cost_delta=4.0,
        risk_delta=99.0,
        risk_cost_weighted_delta=4.0,
        useful_disagreement_count=1,
    )
    config_path = _write_config(
        tmp_path,
        coverage_root,
        canonical_reward_profile=repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3.json",
    )

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out-v3",
        repo_root=repo_root,
    )

    assert summary["profile_version"] == "v3"
    assert summary["guard_evaluation"]["passed"] is True
    assert summary["guard_evaluation"]["checks"]["risk_delta_audit_only"] is True
    assert "risk_regression" not in summary["guard_evaluation"]["failed_guards"]
    assert summary["guard_config"]["risk_delta_hard_gate_enabled"] is False
    assert summary["guard_config"]["candidate_level_risk_delta_guard_is_diagnostic_only"] is True


def test_v3_missing_risk_delta_does_not_create_missing_guard_metric(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=12.0,
        path_cost_delta=4.0,
        risk_delta=0.0,
        risk_cost_weighted_delta=4.0,
        useful_disagreement_count=1,
    )
    summary_path = coverage_root / "xunce-exploration-coverage-comparison-summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload.pop("xunce_risk_delta_vs_incumbent", None)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_path = coverage_root / "xunce-exploration-coverage-comparison-aggregate.json"
    aggregate_payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_payload.pop("risk_delta_mean", None)
    aggregate_path.write_text(json.dumps(aggregate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs_path = coverage_root / "xunce-exploration-coverage-comparison-pairs.jsonl"
    pair_payload = json.loads(pairs_path.read_text(encoding="utf-8").splitlines()[0])
    pair_payload.pop("risk_delta", None)
    pairs_path.write_text(json.dumps(pair_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        coverage_root,
        canonical_reward_profile=repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3.json",
    )

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out-v3-missing-risk-delta",
        repo_root=repo_root,
    )

    assert summary["profile_version"] == "v3"
    assert summary["guard_evaluation"]["passed"] is True
    assert "missing_guard_metric" not in summary["guard_evaluation"]["failed_guards"]
    assert summary["guard_evaluation"]["observed"]["risk_delta"] is None


def test_paired_decision_audit_without_same_candidate_advantage_blocks_stage19(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=5.0,
        path_cost_delta=0.0,
        risk_delta=0.0,
        useful_disagreement_count=0,
    )
    config_path = _write_config(tmp_path, coverage_root)

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=repo_root,
    )

    assert summary["guard_evaluation"]["passed"] is True
    assert summary["stage19_readiness"]["authorized"] is False
    assert summary["stage19_readiness"]["readiness"] == "not_authorized"
    assert "same_candidate_set_policy_selection_advantage_not_established" in summary["diagnostic_reason_codes"]
    paired = json.loads((tmp_path / "out" / "xunce-stage18-5-paired-decision-summary.json").read_text(encoding="utf-8"))
    assert paired["same_candidate_set_advantage_established"] is False
    assert paired["paired_decision_audit_row_count"] == 2


def test_min_coverage_delta_guard_uses_worst_scenario_not_mean(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=12.0,
        path_cost_delta=0.0,
        risk_delta=0.0,
        coverage_per_100m_delta=1.0,
        useful_disagreement_count=1,
    )
    aggregate_path = coverage_root / "xunce-exploration-coverage-comparison-aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["coverage_delta_cells_min"] = 0.0
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path = _write_config(tmp_path, coverage_root)

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=repo_root,
    )

    assert summary["guard_evaluation"]["passed"] is False
    assert "coverage_gain_insufficient" in summary["guard_evaluation"]["failed_guards"]
    assert summary["next_required_change"] == "refine_coverage_reward_and_cost_guard"


def test_candidate_exhaustion_is_diagnostic_not_evidence_authenticity_blocker(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=5.0,
        path_cost_delta=0.0,
        risk_delta=0.0,
        candidate_exhaustion=True,
        useful_disagreement_count=1,
    )
    config_path = _write_config(tmp_path, coverage_root)

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=repo_root,
    )

    assert summary["evidence_authenticity_gate_passed"] is True
    assert "candidate_generation_exhausted" not in summary["blocking_reason_codes"]
    assert "candidate_generation_exhausted" in summary["diagnostic_reason_codes"]
    assert summary["attribution_class_counts"]["candidate_exhaustion"] >= 1
    assert summary["candidate_generation_diagnostics"]["candidate_generation_exhausted_count"] >= 1


def test_missing_required_artifacts_fail_with_explicit_next_change(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    config_path = _write_config(tmp_path, coverage_root)

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=repo_root,
    )

    assert summary["status"] == "failed"
    assert summary["evidence_authenticity_gate_passed"] is False
    assert "missing_coverage_comparison_summary" in summary["blocking_reason_codes"]
    assert "missing_coverage_comparison_pairs" in summary["blocking_reason_codes"]
    assert "missing_paired_decision_audit" in summary["blocking_reason_codes"]
    assert summary["next_required_change"] == "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts"


def test_boundary_violation_or_canary_nonzero_hard_fails(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=5.0,
        path_cost_delta=0.0,
        risk_delta=0.0,
        boundary_updates={"publishes_checkpoint": True},
    )
    config_path = _write_config(tmp_path, coverage_root)

    summary = run_review(
        config_path=config_path,
        output_root=tmp_path / "boundary-out",
        repo_root=repo_root,
    )

    assert summary["status"] == "failed"
    assert summary["evidence_authenticity_gate_passed"] is False
    assert "boundary_violation" in summary["blocking_reason_codes"]
    assert summary["next_required_change"] == "resolve_stage18_5_evidence_review_boundary_rejections"

    canary_root = _write_stage18_5_fixture(
        tmp_path / "canary-case",
        coverage_delta=5.0,
        path_cost_delta=0.0,
        risk_delta=0.0,
        boundary_updates={"canary_traffic_fraction": 0.1},
    )
    canary_config = _write_config(tmp_path / "canary-case", canary_root, canary_traffic_fraction=0.1)

    canary_summary = run_review(
        config_path=canary_config,
        output_root=tmp_path / "canary-out",
        repo_root=repo_root,
    )

    assert canary_summary["status"] == "failed"
    assert canary_summary["evidence_authenticity_gate_passed"] is False
    assert "canary_traffic_fraction_nonzero" in canary_summary["blocking_reason_codes"]
    assert canary_summary["canary_traffic_fraction"] == 0.0


def test_invalid_guard_threshold_config_hard_fails_instead_of_defaulting_to_zero(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=5.0,
        path_cost_delta=0.0,
        risk_delta=0.0,
    )
    config_path = _write_config(tmp_path, coverage_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_risk_delta"] = "0.5"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(Exception, match="max_risk_delta must be a finite number"):
        run_review(
            config_path=config_path,
            output_root=tmp_path / "invalid-threshold-out",
            repo_root=repo_root,
        )


def test_coverage_gain_per_path_cost_delta_mode_must_stay_audit_only(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_review = _import_runner(repo_root)
    coverage_root = _write_stage18_5_fixture(
        tmp_path,
        coverage_delta=5.0,
        path_cost_delta=0.0,
        risk_delta=0.0,
    )
    config_path = _write_config(tmp_path, coverage_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["coverage_gain_per_path_cost_delta_mode"] = "hard_gate"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(Exception, match="coverage_gain_per_path_cost_delta_mode must be audit_only"):
        run_review(
            config_path=config_path,
            output_root=tmp_path / "invalid-mode-out",
            repo_root=repo_root,
        )


def _write_config(
    tmp_path: Path,
    coverage_root: Path,
    *,
    canary_traffic_fraction: float = 0.0,
    canonical_reward_profile: str | Path | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "stage18_5_config.json"
    canonical_reward_profile = canonical_reward_profile or (
        Path(__file__).resolve().parents[1] / "configs" / "xunce_canonical_reward_guard_profile_v2.json"
    )
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage18-5-evidence-attribution-review-config/v1",
                "coverage_comparison_root": str(coverage_root),
                "canonical_reward_profile": str(canonical_reward_profile),
                "max_path_cost_delta_m": 20.0,
                "max_risk_delta": 0.5,
                "max_risk_cost_weighted_delta": 25.0,
                "min_coverage_delta_cells": 1.0,
                "min_coverage_per_100m_delta": 0.0,
                "coverage_gain_per_path_cost_delta_mode": "audit_only",
                "canary_traffic_fraction": canary_traffic_fraction,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_stage18_5_fixture(
    tmp_path: Path,
    *,
    coverage_delta: float,
    path_cost_delta: float,
    risk_delta: float,
    risk_cost_weighted_delta: float | None = None,
    coverage_per_100m_delta: float = 0.0,
    useful_disagreement_count: int = 1,
    candidate_exhaustion: bool = False,
    boundary_updates: dict | None = None,
) -> Path:
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir(parents=True, exist_ok=True)
    risk_cost_weighted_delta = risk_delta if risk_cost_weighted_delta is None else risk_cost_weighted_delta
    boundary_updates = boundary_updates or {}
    summary = {
        "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
        "status": "passed",
        "xunce_coverage_advantage_established": coverage_delta > 0,
        "xunce_new_covered_cell_delta_vs_incumbent": coverage_delta,
        "xunce_path_cost_delta_vs_incumbent": path_cost_delta,
        "xunce_risk_delta_vs_incumbent": risk_delta,
        "risk_cost_weighted_delta_vs_incumbent": risk_cost_weighted_delta,
        "coverage_per_100m_delta_vs_incumbent": coverage_per_100m_delta,
        "coverage_gain_per_path_cost_delta_vs_incumbent": -0.2 if path_cost_delta > 0 else 0.1,
        "xunce_efficiency_regression_count": 1 if path_cost_delta > 0 or coverage_per_100m_delta < 0 else 0,
        "xunce_safety_regression_count": 1 if risk_delta > 0 else 0,
        "candidate_refresh_mode": "dynamic_frontier_nbv_in_process",
        "dynamic_candidate_generation_executed": True,
        "dynamic_proposal_count": 4,
        "dynamic_validation_attempt_count": 4,
        "dynamic_validation_success_count": 3,
        "candidate_generation_exhausted_count": 1 if candidate_exhaustion else 0,
        "coverage_frontier_candidate_count": 0,
        "undercovered_component_candidate_count": 0,
        "low_cost_bridge_candidate_count": 3,
        "conservative_local_candidate_count": 1,
        "validated_pareto_frontier_count": 1,
        "validated_low_cost_candidate_count": 0,
        "validated_efficiency_candidate_count": 0,
        "risk_source_counts": {"route_derived_proxy": 2},
        "formal_risk_source_counts": {},
        "selected_risk_source_counts": {"route_derived_proxy": 1},
        "route_derived_risk_count": 2,
        "same_candidate_set_policy_selection_summary": {
            "paired_decision_audit_row_count": 2,
            "policy_disagreement_count": 2,
            "useful_disagreement_count": useful_disagreement_count,
        },
        "same_candidate_set_policy_selection_advantage_established": useful_disagreement_count > 0,
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
    }
    summary.update(boundary_updates)
    _write_json(coverage_root / "xunce-exploration-coverage-comparison-summary.json", summary)
    _write_json(
        coverage_root / "xunce-exploration-coverage-comparison-aggregate.json",
        {
            "schema_version": "xunce-exploration-coverage-comparison-aggregate/v1",
            "scenario_count": 2,
            "coverage_delta_cells_mean": coverage_delta,
            "path_cost_delta_m_mean": path_cost_delta,
            "risk_delta_mean": risk_delta,
            "coverage_per_100m_delta_mean": coverage_per_100m_delta,
        },
    )
    _write_jsonl(
        coverage_root / "xunce-exploration-coverage-comparison-pairs.jsonl",
        [
            {
                "schema_version": "xunce-exploration-coverage-comparison-pair/v1",
                "scenario_id": "scenario-001",
                "coverage_delta_cells": coverage_delta,
                "path_cost_delta_m": path_cost_delta,
                "risk_delta": risk_delta,
                "risk_cost_weighted_delta": risk_cost_weighted_delta,
                "coverage_per_100m_delta": coverage_per_100m_delta,
                "roi_weighted_coverage_delta": coverage_delta,
                "undefined_metric_reason_codes": [],
            }
        ],
    )
    _write_jsonl(
        coverage_root / "xunce-exploration-coverage-episodes.jsonl",
        [
            {
                "schema_version": "xunce-exploration-coverage-episode/v1",
                "scenario_id": "scenario-001",
                "policy": "xunce",
                "candidate_generation_exhausted": candidate_exhaustion,
                "episode_termination_reason": "candidate_generation_exhausted" if candidate_exhaustion else "rollout_complete",
                "candidate_generation_exhausted_count": 1 if candidate_exhaustion else 0,
                "risk_source": "route_derived_proxy",
            }
        ],
    )
    _write_jsonl(
        coverage_root / "xunce-exploration-coverage-paired-decision-audit.jsonl",
        [
            {
                "schema_version": "xunce-exploration-coverage-paired-decision-audit/v1",
                "scenario_id": "scenario-001",
                "same_state_same_candidate_set": True,
                "policy_disagreement": True,
                "xunce_selected_action_index": 1,
                "incumbent_selected_action_index": 0,
                "xunce_expected_coverage_rate_delta": 0.4,
                "incumbent_expected_coverage_rate_delta": 0.2,
                "xunce_coverage_gain_per_path_cost": 0.6,
                "incumbent_coverage_gain_per_path_cost": 0.4,
                "xunce_path_cost": 12.0,
                "incumbent_path_cost": 5.0,
                "xunce_risk": 0.3,
                "incumbent_risk": 0.1,
            },
            {
                "schema_version": "xunce-exploration-coverage-paired-decision-audit/v1",
                "scenario_id": "scenario-002",
                "same_state_same_candidate_set": True,
                "policy_disagreement": useful_disagreement_count > 0,
                "xunce_selected_action_index": 0,
                "incumbent_selected_action_index": 0,
                "xunce_expected_coverage_rate_delta": 0.2,
                "incumbent_expected_coverage_rate_delta": 0.2,
                "xunce_coverage_gain_per_path_cost": 0.4,
                "incumbent_coverage_gain_per_path_cost": 0.4,
                "xunce_path_cost": 5.0,
                "incumbent_path_cost": 5.0,
                "xunce_risk": 0.1,
                "incumbent_risk": 0.1,
            },
        ],
    )
    return coverage_root


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
