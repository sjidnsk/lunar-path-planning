import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def _load_profile():
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    return load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json")


def _run_review(config_path: Path, output_root: Path):
    from scripts.run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement import (
        run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement,
    )

    return run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_current_guard_failure_routes_to_candidate_metric_audit(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        coverage_delta=150.0,
        path_cost_delta=232.7,
        risk_delta=4.8,
        risk_cost_weighted_delta=347.1,
        coverage_per_100m_delta=-54.7,
        useful_disagreement_count=0,
        include_candidate_metric_audit=False,
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["guard_refinement_passed"] is False
    assert summary["counterfactual_reselection_claimed"] is False
    assert summary["candidate_metric_readiness"]["full_candidate_metric_replay_available"] is False
    assert summary["next_required_change"] == "rerun_stage18_4e_with_candidate_metric_audit"
    assert summary["stage19_readiness"]["authorized"] is False
    assert "cost_efficiency_regression" in summary["diagnostic_reason_codes"]
    assert (tmp_path / "out" / "xunce-stage18-6-guard-refinement-summary.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-6-selected-action-guard-replay.jsonl").is_file()
    assert (tmp_path / "out" / "xunce-stage18-6-scenario-guard-replay.jsonl").is_file()
    assert (tmp_path / "out" / "xunce-stage18-6-candidate-metric-readiness.json").is_file()


def test_risk_inside_budget_can_pass_but_over_budget_fails(tmp_path: Path) -> None:
    clean_config = _write_stage18_6_fixture(
        tmp_path / "clean",
        coverage_delta=25.0,
        path_cost_delta=5.0,
        risk_delta=0.5,
        risk_cost_weighted_delta=25.0,
        coverage_per_100m_delta=0.1,
        useful_disagreement_count=1,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(),
    )
    clean = _run_review(clean_config, tmp_path / "clean-out")

    assert clean["scenario_guard_replay_summary"]["guard_clean_coverage_win_count"] == 1
    assert clean["guard_refinement_passed"] is True
    assert clean["next_required_change"] == "prepare_stage19_evaluator_critic_preflight"
    assert clean["stage19_readiness"]["authorized"] is False

    over_config = _write_stage18_6_fixture(
        tmp_path / "over",
        coverage_delta=25.0,
        path_cost_delta=5.0,
        risk_delta=0.5001,
        risk_cost_weighted_delta=25.1,
        coverage_per_100m_delta=0.1,
        useful_disagreement_count=1,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(),
    )
    over = _run_review(over_config, tmp_path / "over-out")

    assert over["guard_refinement_passed"] is False
    assert "risk_regression" in over["diagnostic_reason_codes"]
    assert over["next_required_change"] == "refine_coverage_reward_and_cost_guard"


def test_risk_cost_weighted_budget_fails_independently(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        coverage_delta=25.0,
        path_cost_delta=5.0,
        risk_delta=0.4,
        risk_cost_weighted_delta=25.1,
        coverage_per_100m_delta=0.1,
        useful_disagreement_count=1,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(),
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["guard_refinement_passed"] is False
    assert "risk_regression" in summary["diagnostic_reason_codes"]
    assert "risk_cost_weighted_budget_exceeded" in summary["reason_codes"]
    assert "risk_budget_exceeded" not in summary["reason_codes"]


def test_selected_action_per_cost_delta_is_audit_only(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        coverage_delta=25.0,
        path_cost_delta=5.0,
        risk_delta=0.1,
        risk_cost_weighted_delta=2.0,
        coverage_per_100m_delta=0.1,
        useful_disagreement_count=1,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(),
        selected_action_overrides={
            "xunce_selected_expected_new_coverage_cell_count": 11.0,
            "incumbent_selected_expected_new_coverage_cell_count": 10.0,
            "xunce_selected_path_cost": 20.0,
            "incumbent_selected_path_cost": 10.0,
        },
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["selected_action_guard_replay_summary"]["guard_clean_coverage_win_count"] == 1
    assert "coverage_gain_per_path_cost_regression" not in summary["reason_codes"]
    assert summary["guard_refinement_passed"] is True


def test_utility_profile_win_cannot_bypass_canonical_guard(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        coverage_delta=30.0,
        path_cost_delta=40.0,
        risk_delta=0.1,
        risk_cost_weighted_delta=4.0,
        coverage_per_100m_delta=1.0,
        useful_disagreement_count=1,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(),
        utility_profile_outcomes={"coverage_first": {"winner": "xunce"}},
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["guard_refinement_passed"] is False
    assert summary["stage19_readiness"]["authorized"] is False
    assert summary["utility_profile_bypass_blocked"] is True
    assert "utility_profile_win_blocked_by_canonical_guard" in summary["diagnostic_reason_codes"]


def test_profile_hash_missing_or_mismatch_hard_fails(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(profile_hash="bad-hash"),
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert "profile_hash_mismatch" in summary["blocking_reason_codes"]
    assert summary["next_required_change"] == "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts"
    assert summary["stage19_readiness"]["authorized"] is False

    missing_config = _write_stage18_6_fixture(tmp_path / "missing", omit_profile_hash=True)
    missing = _run_review(missing_config, tmp_path / "missing-out")

    assert missing["status"] == "failed"
    assert "profile_hash_missing" in missing["blocking_reason_codes"]
    assert missing["next_required_change"] == "rerun_xunce_stage18_5_evidence_attribution_review"


def test_profile_id_and_version_are_required_for_lineage(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(),
        omit_profile_id=True,
    )

    missing = _run_review(config, tmp_path / "missing-out")

    assert missing["status"] == "failed"
    assert "profile_hash_missing" in missing["blocking_reason_codes"]

    mismatch_config = _write_stage18_6_fixture(
        tmp_path / "mismatch",
        include_candidate_metric_audit=True,
        candidate_metric_rows=[{**row, "profile_version": "v1"} for row in _safe_candidate_rows()],
        profile_version_override="v1",
    )

    mismatch = _run_review(mismatch_config, tmp_path / "mismatch-out")

    assert mismatch["status"] == "failed"
    assert "profile_hash_mismatch" in mismatch["blocking_reason_codes"]


def test_candidate_metric_audit_schema_and_hash_are_required(tmp_path: Path) -> None:
    rows = _safe_candidate_rows()
    rows[0].pop("covered_cells_hash")
    config = _write_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=rows,
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["candidate_metric_readiness"]["full_candidate_metric_replay_available"] is False
    assert "missing_candidate_metric_field:covered_cells_hash" in summary["diagnostic_reason_codes"]
    assert summary["counterfactual_reselection_claimed"] is False
    assert summary["next_required_change"] == "rerun_stage18_4e_with_candidate_metric_audit"


def test_candidate_metric_audit_must_cover_all_paired_decision_keys(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=_safe_candidate_rows(),
        extra_paired_decision_rows=[
            {
                "scenario_id": "s1",
                "split": "test",
                "roi_group": "roi_1",
                "executing_policy": "xunce",
                "step_index": 1,
                "current_cell": [1, 1],
                "covered_cells_hash": "covered-1",
                "candidate_set_id": "set-1",
                "candidate_set_hash": "hash-1",
                "same_state_same_candidate_set": True,
                "policy_disagreement": True,
                "xunce_selected_action_index": 0,
                "incumbent_selected_action_index": 1,
                "xunce_selected_expected_new_coverage_cell_count": 5.0,
                "incumbent_selected_expected_new_coverage_cell_count": 4.0,
                "xunce_selected_roi_weighted_coverage_delta": 5.0,
                "incumbent_selected_roi_weighted_coverage_delta": 4.0,
                "xunce_selected_path_cost": 2.0,
                "incumbent_selected_path_cost": 2.0,
                "xunce_selected_risk": 0.1,
                "incumbent_selected_risk": 0.1,
                "reason_codes": [],
            }
        ],
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["candidate_metric_readiness"]["full_candidate_metric_replay_available"] is False
    assert "candidate_metric_missing_paired_decision_key" in summary["diagnostic_reason_codes"]
    assert summary["next_required_change"] == "rerun_stage18_4e_with_candidate_metric_audit"


def test_extra_candidate_metric_keys_do_not_block_full_replay(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=[
            _candidate_row(candidate_index=1, coverage=12.0, roi=12.0, path_cost=11.0, risk=1.2),
            _candidate_row(
                candidate_index=0,
                coverage=12.0,
                roi=12.0,
                path_cost=11.0,
                risk=1.2,
                covered_cells_hash="extra-covered",
            ),
        ],
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["candidate_metric_readiness"]["full_candidate_metric_replay_available"] is True
    assert summary["candidate_metric_readiness"]["extra_candidate_set_key_count"] == 1
    assert "candidate_metric_extra_candidate_set_key" in summary["diagnostic_reason_codes"]
    assert "candidate_metric_candidate_set_mismatch" not in summary["reason_codes"]


def test_candidate_guard_uses_relative_risk_delta_not_absolute_risk(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=[
            _candidate_row(candidate_index=0, coverage=10.0, roi=10.0, path_cost=10.0, risk=1.0),
            _candidate_row(candidate_index=1, coverage=12.0, roi=12.0, path_cost=11.0, risk=1.2),
        ],
    )

    summary = _run_review(config, tmp_path / "out")

    guarded = summary["guarded_reselection_summary"]
    assert guarded["candidate_guard_mode"] == "paired_baseline_delta_vs_incumbent_selected"
    assert guarded["absolute_risk_proxy_is_audit_only"] is True
    assert guarded["safe_candidate_available_count"] == 1
    assert guarded["unsafe_high_coverage_candidate_rejected_count"] == 0


def test_guarded_reselection_rejects_high_coverage_unsafe_candidate(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        coverage_delta=20.0,
        path_cost_delta=2.0,
        risk_delta=0.1,
        risk_cost_weighted_delta=2.0,
        coverage_per_100m_delta=0.1,
        useful_disagreement_count=0,
        include_candidate_metric_audit=True,
        candidate_metric_rows=[
            _candidate_row(candidate_index=0, coverage=8.0, roi=8.0, path_cost=2.0, risk=0.1),
            _candidate_row(candidate_index=1, coverage=30.0, roi=30.0, path_cost=80.0, risk=2.0),
            _candidate_row(candidate_index=2, coverage=12.0, roi=12.0, path_cost=11.0, risk=1.2),
        ],
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["candidate_metric_readiness"]["full_candidate_metric_replay_available"] is True
    assert summary["guarded_reselection_summary"]["unsafe_high_coverage_candidate_rejected_count"] == 1
    assert summary["guarded_reselection_summary"]["safe_candidate_available_count"] == 1
    assert summary["next_required_change"] == "refine_coverage_reward_and_cost_guard"


def test_boundary_flags_hard_fail_and_block_stage19(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        config_overrides={"runs_new_ppo_update": True, "canary_traffic_fraction": 0.1},
    )

    summary = _run_review(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert "boundary_violation" in summary["blocking_reason_codes"]
    assert "canary_traffic_fraction_nonzero" in summary["blocking_reason_codes"]
    assert summary["next_required_change"] == "resolve_stage18_6_guard_refinement_boundary_rejections"
    assert summary["stage19_readiness"]["authorized"] is False


def test_legacy_v1_reward_components_are_diagnostic_blockers(tmp_path: Path) -> None:
    config = _write_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=[
            {
                **_candidate_row(candidate_index=0, coverage=8.0, roi=8.0, path_cost=2.0, risk=0.1),
                "risk_cost_exposure_penalty": 1.0,
            }
        ],
    )

    summary = _run_review(config, tmp_path / "out")

    assert "legacy_v1_reward_component_detected" in summary["diagnostic_reason_codes"]
    assert summary["next_required_change"] == "refine_coverage_reward_and_cost_guard"
    assert summary["stage19_readiness"]["authorized"] is False


def _write_stage18_6_fixture(
    base: Path,
    *,
    coverage_delta: float = 12.0,
    path_cost_delta: float = 4.0,
    risk_delta: float = 0.2,
    risk_cost_weighted_delta: float = 3.0,
    coverage_per_100m_delta: float = 0.1,
    useful_disagreement_count: int = 1,
    include_candidate_metric_audit: bool = False,
    candidate_metric_rows: list[dict] | None = None,
    utility_profile_outcomes: dict | None = None,
    config_overrides: dict | None = None,
    omit_profile_hash: bool = False,
    omit_profile_id: bool = False,
    profile_version_override: str | None = None,
    selected_action_overrides: dict | None = None,
    extra_paired_decision_rows: list[dict] | None = None,
) -> Path:
    profile = _load_profile()
    stage18_5 = base / "stage18_5"
    coverage = base / "coverage"
    stage18_5.mkdir(parents=True, exist_ok=True)
    coverage.mkdir(parents=True, exist_ok=True)
    profile_hash = None if omit_profile_hash else profile.profile_hash
    profile_id = None if omit_profile_id else profile.profile_id
    profile_version = profile_version_override or profile.profile_version
    guard = {
        "schema_version": "xunce-stage18-5-guard-evaluation/v1",
        "passed": (
            path_cost_delta <= profile.guards["max_acceptable_path_cost_delta_m"]
            and risk_delta <= profile.guards["max_acceptable_risk_delta"]
            and risk_cost_weighted_delta <= profile.guards["max_acceptable_risk_cost_weighted_delta"]
            and coverage_per_100m_delta >= profile.guards["min_coverage_per_100m_delta"]
        ),
        "failed_guards": [] if path_cost_delta <= 20 and risk_delta <= 0.5 and risk_cost_weighted_delta <= 25 and coverage_per_100m_delta >= 0 else ["path_cost_regression"],
        "thresholds": {
            "profile_id": profile_id,
            "profile_version": profile_version,
            "profile_hash": profile_hash,
        },
    }
    routing = {
        "schema_version": "xunce-stage18-5-next-stage-routing/v1",
        "primary_route": "refine_coverage_reward_and_cost_guard",
        "stage19_authorized": False,
    }
    stage18_5_summary = {
        "schema_version": "xunce-stage18-5-evidence-attribution-summary/v1",
        "status": "passed",
        "coverage_comparison_root": str(coverage.resolve()),
        "profile_id": profile_id,
        "profile_version": profile_version,
        "profile_hash": profile_hash,
        "evidence_authenticity_gate_passed": True,
        "candidate_validity_gate_passed": True,
        "guard_evaluation": guard,
        "next_stage_routing": routing,
        "stage19_readiness": {"readiness": "not_authorized", "authorized": False},
        "paired_decision_summary": {
            "schema_version": "xunce-stage18-5-paired-decision-summary/v1",
            "same_candidate_set_advantage_established": useful_disagreement_count > 0,
            "useful_disagreement_count": useful_disagreement_count,
        },
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
    }
    _write_json(stage18_5 / "xunce-stage18-5-evidence-attribution-summary.json", stage18_5_summary)
    _write_json(stage18_5 / "xunce-stage18-5-guard-evaluation.json", guard)
    _write_json(stage18_5 / "xunce-stage18-5-next-stage-routing.json", routing)
    _write_json(
        stage18_5 / "xunce-stage18-5-paired-decision-summary.json",
        stage18_5_summary["paired_decision_summary"],
    )
    _write_jsonl(stage18_5 / "xunce-stage18-5-regression-attribution.jsonl", [])

    coverage_summary = {
        "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
        "status": "passed",
        "coverage_comparison_root": str(coverage.resolve()),
        "profile_id": profile_id,
        "profile_version": profile_version,
        "profile_hash": profile_hash,
        "canonical_guard_thresholds": {
            "min_coverage_delta_cells": profile.guards["min_coverage_delta_cells"],
            "max_path_cost_delta_m": profile.guards["max_acceptable_path_cost_delta_m"],
            "max_risk_delta": profile.guards["max_acceptable_risk_delta"],
            "max_risk_cost_weighted_delta": profile.guards["max_acceptable_risk_cost_weighted_delta"],
            "min_coverage_per_100m_delta": profile.guards["min_coverage_per_100m_delta"],
            "coverage_gain_per_path_cost_delta_mode": profile.guards["coverage_gain_per_path_cost_delta_mode"],
        },
        "same_candidate_set_policy_selection_advantage_established": useful_disagreement_count > 0,
        "same_candidate_set_policy_selection_summary": {"useful_disagreement_count": useful_disagreement_count},
        "candidate_metric_audit": str(coverage / "xunce-exploration-coverage-candidate-metric-audit.jsonl")
        if include_candidate_metric_audit
        else None,
        "candidate_metric_audit_row_count": len(candidate_metric_rows or []) if include_candidate_metric_audit else 0,
    }
    pair = {
        "schema_version": "xunce-exploration-coverage-comparison-pair/v1",
        "scenario_id": "s0",
        "coverage_delta_cells": coverage_delta,
        "path_cost_delta_m": path_cost_delta,
        "risk_delta": risk_delta,
        "risk_cost_weighted_delta": risk_cost_weighted_delta,
        "coverage_per_100m_delta": coverage_per_100m_delta,
        "coverage_gain_per_path_cost_delta": 0.1 if coverage_per_100m_delta >= 0 else -0.1,
        "utility_profile_outcomes": utility_profile_outcomes or {},
        "canonical_guard_profile_id": profile_id,
        "canonical_guard_profile_version": profile_version,
        "canonical_guard_profile_hash": profile_hash,
    }
    paired = {
        "schema_version": "xunce-exploration-coverage-paired-decision-audit/v1",
        "scenario_id": "s0",
        "split": "test",
        "roi_group": "roi_0",
        "executing_policy": "xunce",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": "covered-0",
        "candidate_set_id": "set-0",
        "candidate_set_hash": "hash-0",
        "same_state_same_candidate_set": True,
        "policy_disagreement": useful_disagreement_count > 0,
        "xunce_selected_action_index": 1,
        "incumbent_selected_action_index": 0,
        "xunce_selected_expected_new_coverage_cell_count": 20.0,
        "incumbent_selected_expected_new_coverage_cell_count": 10.0,
        "xunce_selected_roi_weighted_coverage_delta": 20.0,
        "incumbent_selected_roi_weighted_coverage_delta": 10.0,
        "xunce_selected_path_cost": 10.0 + path_cost_delta,
        "incumbent_selected_path_cost": 10.0,
        "xunce_selected_risk": 1.0 + risk_delta,
        "incumbent_selected_risk": 1.0,
        "reason_codes": [],
    }
    if selected_action_overrides:
        paired.update(selected_action_overrides)
    _write_json(coverage / "xunce-exploration-coverage-comparison-summary.json", coverage_summary)
    _write_json(
        coverage / "xunce-exploration-coverage-comparison-aggregate.json",
        {"schema_version": "xunce-exploration-coverage-comparison-aggregate/v1", "scenario_count": 1},
    )
    _write_jsonl(coverage / "xunce-exploration-coverage-comparison-pairs.jsonl", [pair])
    _write_jsonl(coverage / "xunce-exploration-coverage-episodes.jsonl", [{"schema_version": "episode/v1", "scenario_id": "s0"}])
    _write_jsonl(coverage / "xunce-exploration-coverage-paired-decision-audit.jsonl", [paired, *(extra_paired_decision_rows or [])])
    if include_candidate_metric_audit:
        _write_jsonl(
            coverage / "xunce-exploration-coverage-candidate-metric-audit.jsonl",
            candidate_metric_rows or _safe_candidate_rows(),
        )

    config_payload = {
        "schema_version": "xunce-stage18-6-coverage-reward-cost-risk-guard-refinement-config/v1",
        "stage18_5_attribution_root": str(stage18_5),
        "coverage_comparison_root": str(coverage),
        "canonical_reward_profile": str(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json"),
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
    }
    if config_overrides:
        config_payload.update(config_overrides)
    config_path = base / "stage18_6_config.json"
    _write_json(config_path, config_payload)
    return config_path


def _safe_candidate_rows(profile_hash: str | None = None) -> list[dict]:
    return [
        _candidate_row(candidate_index=0, coverage=8.0, roi=8.0, path_cost=2.0, risk=0.1, profile_hash=profile_hash),
        _candidate_row(candidate_index=1, coverage=12.0, roi=12.0, path_cost=3.0, risk=0.2, profile_hash=profile_hash),
    ]


def _candidate_row(
    *,
    candidate_index: int,
    coverage: float,
    roi: float,
    path_cost: float,
    risk: float,
    profile_hash: str | None = None,
    covered_cells_hash: str = "covered-0",
) -> dict:
    profile = _load_profile()
    return {
        "schema_version": "xunce-exploration-coverage-candidate-metric-audit-row/v1",
        "scenario_id": "s0",
        "split": "test",
        "roi_group": "roi_0",
        "policy": "xunce",
        "executing_policy": "xunce",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": covered_cells_hash,
        "candidate_set_id": "set-0",
        "candidate_set_hash": "hash-0",
        "candidate_index": candidate_index,
        "candidate_cell": [candidate_index + 1, candidate_index + 2],
        "action_mask_valid": True,
        "expected_new_coverage_cell_count": coverage,
        "roi_weighted_coverage_delta": roi,
        "path_cost": path_cost,
        "risk": risk,
        "risk_cost_weighted": path_cost * risk,
        "risk_source": "offline_proxy",
        "coverage_gain_per_path_cost": coverage / path_cost,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile_hash or profile.profile_hash,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
