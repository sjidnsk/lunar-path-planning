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


SWEEPS = ((6, 48), (12, 96), (24, 192), (36, 288))


def _load_profile():
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    return load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json")


def _run_audit(config_path: Path, output_root: Path):
    from scripts.run_xunce_stage18_7_candidate_count_scaling_audit import run_xunce_stage18_7_candidate_count_scaling_audit

    return run_xunce_stage18_7_candidate_count_scaling_audit(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_missing_sweeps_route_to_metric_audit_command_plan(tmp_path: Path) -> None:
    config = _write_config(tmp_path, [])

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_missing_candidate_count_sweeps_with_metric_audit"
    assert summary["stage19_authorized"] is False
    assert summary["runs_new_ppo_update"] is False
    assert len(summary["candidate_count_results"]) == 4
    assert all(row["candidate_metric_audit_available"] is False for row in summary["candidate_count_results"])
    command_plan = json.loads((tmp_path / "out" / "xunce-stage18-7-candidate-count-scaling-command-plan.json").read_text(encoding="utf-8"))
    stage18_4e_commands = [row["display"] for row in command_plan["commands"] if row["stage"] == "18.4E"]
    assert len(stage18_4e_commands) == 4
    assert all("--emit-candidate-metric-audit" in command for command in stage18_4e_commands)
    assert any("--dynamic-max-candidates-per-step" in command and " 36" in command for command in stage18_4e_commands)


def test_strict_v3_command_plan_uses_strict_v3_configs(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        [],
        {
            "canonical_reward_profile": str(
                REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json"
            )
        },
    )

    summary = _run_audit(config, tmp_path / "out")

    assert summary["profile_version"] == "v3"
    command_plan = json.loads((tmp_path / "out" / "xunce-stage18-7-candidate-count-scaling-command-plan.json").read_text(encoding="utf-8"))
    displays = [row["display"] for row in command_plan["commands"]]
    assert any("xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json" in row for row in displays)
    assert any("xunce_stage18_5_evidence_attribution_review_strict_v3.json" in row for row in displays)
    assert any("xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_strict_v3.json" in row for row in displays)
    assert not any("xunce_stage18_5_evidence_attribution_review_v1.json" in row for row in displays)


def test_complete_sweeps_with_clean_stage18_6_route_to_preflight_but_not_authorized(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = []
    for count, pool in SWEEPS:
        sweeps.append(
            _write_sweep(
                tmp_path,
                count=count,
                proposal_pool=pool,
                profile=profile,
                safe_candidates=True,
                xunce_selects_safe=True,
                incumbent_selects_safe=False,
                stage18_6_route="prepare_stage19_evaluator_critic_preflight",
                guard_refinement_passed=True,
                same_candidate_set_guard_clean_advantage_established=True,
            )
        )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "prepare_stage19_evaluator_critic_preflight"
    assert summary["stage19_authorized"] is False
    assert summary["stage18_6_guard_refinement_passed_count"] == 4
    assert summary["same_candidate_set_guard_clean_advantage_established_count"] == 4
    assert all(row["boundary_flags_all_false"] is True for row in summary["candidate_count_results"])
    result_lines = (tmp_path / "out" / "xunce-stage18-7-candidate-count-scaling-results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 4


def test_stage18_6_incomplete_replay_cannot_be_used_for_preflight(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = []
    for count, pool in SWEEPS:
        sweeps.append(
            _write_sweep(
                tmp_path,
                count=count,
                proposal_pool=pool,
                profile=profile,
                safe_candidates=True,
                xunce_selects_safe=True,
                incumbent_selects_safe=False,
                stage18_6_route="prepare_stage19_evaluator_critic_preflight",
                guard_refinement_passed=True,
                same_candidate_set_guard_clean_advantage_established=True,
                stage18_6_candidate_metric_replay_available=False,
            )
        )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage18_7_lineage_or_config_drift"
    assert "invalid_stage18_6_preflight_semantics" in summary["blocking_reason_codes"]
    assert summary["stage19_authorized"] is False


def test_non_count_config_drift_hard_fails(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    _write_sweep(
        tmp_path,
        count=24,
        proposal_pool=192,
        profile=profile,
        safe_candidates=True,
        normalized_config_overrides={"dynamic_candidate_generation_mode": "changed_generation_mode"},
    )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage18_7_lineage_or_config_drift"
    assert "non_count_config_drift" in summary["blocking_reason_codes"]


def test_extra_normalized_config_drift_hard_fails(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    _write_sweep(
        tmp_path,
        count=12,
        proposal_pool=96,
        profile=profile,
        safe_candidates=True,
        normalized_config_overrides={"dynamic_frontier_radius_cells": 99},
    )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert "non_count_config_drift" in summary["blocking_reason_codes"]


def test_proposal_pool_matrix_is_fixed(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    config = _write_config(tmp_path, sweeps, {"proposal_pool_overrides": {12: 120}})

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage18_7_lineage_or_config_drift"
    assert "candidate_count_matrix_mismatch" in summary["blocking_reason_codes"]


def test_profile_hash_mismatch_hard_fails_without_defaulting(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    _write_sweep(
        tmp_path,
        count=12,
        proposal_pool=96,
        profile=profile,
        safe_candidates=True,
        coverage_profile_hash="different-profile-hash",
    )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage18_7_lineage_or_config_drift"
    assert "profile_hash_mismatch" in summary["blocking_reason_codes"]


def test_malformed_candidate_metric_jsonl_hard_fails(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    audit_path = tmp_path / "count_006" / "stage18_4e" / "xunce-exploration-coverage-candidate-metric-audit.jsonl"
    audit_path.write_text(
        audit_path.read_text(encoding="utf-8") + "{bad json}\n",
        encoding="utf-8",
    )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert "malformed_candidate_metric_audit_jsonl" in summary["blocking_reason_codes"]


def test_non_numeric_candidate_step_index_does_not_crash_readiness(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    audit_path = Path(sweeps[0]["coverage_comparison_root"]) / "xunce-exploration-coverage-candidate-metric-audit.jsonl"
    rows = _read_jsonl(audit_path)
    rows[0]["step_index"] = "bad"
    _write_jsonl(audit_path, rows)
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert "candidate_metric_extra_candidate_set_key" in summary["diagnostic_reason_codes"]


def test_missing_candidate_risk_cost_weighted_uses_guard_fallback(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    for sweep in sweeps:
        audit_path = Path(sweep["coverage_comparison_root"]) / "xunce-exploration-coverage-candidate-metric-audit.jsonl"
        rows = _read_jsonl(audit_path)
        for row in rows:
            row.pop("risk_cost_weighted", None)
        _write_jsonl(audit_path, rows)
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["best_guard_clean_advantage_candidate_available_rate"] > 0
    assert not any(
        "missing_candidate_metric_field:risk_cost_weighted" in row["blocking_reason_codes"]
        for row in summary["candidate_count_results"]
    )


def test_stage18_6_lineage_must_match_same_count_stage18_5_root(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(tmp_path, count=count, proposal_pool=pool, profile=profile, safe_candidates=True)
        for count, pool in SWEEPS
    ]
    stage18_6_summary = tmp_path / "count_036" / "stage18_6" / "xunce-stage18-6-guard-refinement-summary.json"
    payload = json.loads(stage18_6_summary.read_text(encoding="utf-8"))
    payload["stage18_5_attribution_root"] = str((tmp_path / "other_count" / "stage18_5").resolve())
    stage18_6_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage18_7_lineage_or_config_drift"
    assert "stale_stage18_6_guard_refinement_root" in summary["blocking_reason_codes"]


def test_guard_clean_candidates_appear_but_xunce_does_not_select_routes_to_reward_guard(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = []
    for count, pool in SWEEPS:
        sweeps.append(
            _write_sweep(
                tmp_path,
                count=count,
                proposal_pool=pool,
                profile=profile,
                safe_candidates=True,
                xunce_selects_safe=False,
                incumbent_selects_safe=False,
                stage18_6_route="refine_coverage_reward_and_cost_guard",
                guard_refinement_passed=False,
            )
        )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "refine_coverage_reward_and_cost_guard"
    assert summary["best_guard_clean_candidate_available_rate"] > 0
    assert summary["best_guard_clean_advantage_candidate_available_rate"] > 0
    assert summary["best_xunce_selected_guard_clean_advantage_rate"] == 0


def test_high_count_sweeps_without_guard_clean_candidates_route_to_candidate_generation(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(
            tmp_path,
            count=count,
            proposal_pool=pool,
            profile=profile,
            safe_candidates=False,
            xunce_selects_safe=False,
            incumbent_selects_safe=False,
            stage18_6_route="expand_candidate_generation_roi_complexity",
            guard_refinement_passed=False,
        )
        for count, pool in SWEEPS
    ]
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "expand_candidate_generation_roi_complexity"
    assert summary["best_guard_clean_candidate_available_rate"] == 0


def test_risk_cost_weighted_delta_mean_falls_back_to_pair_rows(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(
            tmp_path,
            count=count,
            proposal_pool=pool,
            profile=profile,
            safe_candidates=True,
        )
        for count, pool in SWEEPS
    ]
    for sweep in sweeps:
        aggregate_path = Path(sweep["coverage_comparison_root"]) / "xunce-exploration-coverage-comparison-aggregate.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate.pop("risk_cost_weighted_delta_mean")
        _write_json(aggregate_path, aggregate)
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert all(row["risk_cost_weighted_delta_mean"] == 3.0 for row in summary["candidate_count_results"])


def test_stage18_6_failed_utility_win_cannot_enter_preflight(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(
            tmp_path,
            count=count,
            proposal_pool=pool,
            profile=profile,
            safe_candidates=True,
            utility_profile_winner="xunce",
            stage18_6_route="refine_coverage_reward_and_cost_guard",
            guard_refinement_passed=False,
            same_candidate_set_guard_clean_advantage_established=False,
        )
        for count, pool in SWEEPS
    ]
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["stage18_6_guard_refinement_passed_count"] == 0
    assert summary["next_required_change"] == "refine_coverage_reward_and_cost_guard"
    assert summary["stage19_authorized"] is False


def test_boundary_flags_hard_fail(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = [
        _write_sweep(
            tmp_path,
            count=count,
            proposal_pool=pool,
            profile=profile,
            safe_candidates=True,
            stage18_6_boundary_overrides={"runs_new_ppo_update": True},
        )
        for count, pool in SWEEPS
    ]
    config = _write_config(tmp_path, sweeps, {"canary_traffic_fraction": 0.1})

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage18_7_candidate_count_scaling_boundary_rejections"
    assert "boundary_violation" in summary["blocking_reason_codes"]
    assert "canary_traffic_fraction_nonzero" in summary["blocking_reason_codes"]
    assert summary["stage19_authorized"] is False


def test_runtime_budget_exceeded_for_36_continues_bounded_sweep(tmp_path: Path) -> None:
    profile = _load_profile()
    sweeps = []
    for count, pool in SWEEPS:
        sweeps.append(
            _write_sweep(
                tmp_path,
                count=count,
                proposal_pool=pool,
                profile=profile,
                safe_candidates=True,
                runtime_budget_exceeded=(count == 36),
            )
        )
    config = _write_config(tmp_path, sweeps)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "continue_candidate_count_scaling_with_bounded_budget"
    assert "candidate_count_36_runtime_budget_exceeded" in summary["diagnostic_reason_codes"]


def _write_config(tmp_path: Path, sweeps: list[dict], overrides: dict | None = None) -> Path:
    sweep_by_count = {row["candidate_count"]: row for row in sweeps}
    proposal_pool_overrides = (overrides or {}).pop("proposal_pool_overrides", {}) if overrides else {}
    payload = {
        "schema_version": "xunce-stage18-7-candidate-count-scaling-audit-config/v1",
        "artifact_workspace_root": str(tmp_path / "stage18_7_workspace"),
        "canonical_reward_profile": str(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json"),
        "sweeps": [
            {
                "candidate_count": count,
                "proposal_pool_limit": proposal_pool_overrides.get(count, pool),
                "coverage_comparison_root": sweep_by_count.get(count, {}).get(
                    "coverage_comparison_root",
                    str(tmp_path / f"missing_count_{count:03d}" / "stage18_4e"),
                ),
                "stage18_5_attribution_root": sweep_by_count.get(count, {}).get(
                    "stage18_5_attribution_root",
                    str(tmp_path / f"missing_count_{count:03d}" / "stage18_5"),
                ),
                "stage18_6_guard_refinement_root": sweep_by_count.get(count, {}).get(
                    "stage18_6_guard_refinement_root",
                    str(tmp_path / f"missing_count_{count:03d}" / "stage18_6"),
                ),
            }
            for count, pool in SWEEPS
        ],
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
    payload.update(overrides or {})
    path = tmp_path / "stage18_7_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_sweep(
    tmp_path: Path,
    *,
    count: int,
    proposal_pool: int,
    profile,
    safe_candidates: bool,
    xunce_selects_safe: bool = True,
    incumbent_selects_safe: bool = False,
    stage18_6_route: str = "refine_coverage_reward_and_cost_guard",
    guard_refinement_passed: bool = False,
    same_candidate_set_guard_clean_advantage_established: bool = False,
    coverage_profile_hash: str | None = None,
    normalized_config_overrides: dict | None = None,
    utility_profile_winner: str | None = None,
    stage18_6_boundary_overrides: dict | None = None,
    runtime_budget_exceeded: bool = False,
    stage18_6_candidate_metric_replay_available: bool = True,
) -> dict:
    root = tmp_path / f"count_{count:03d}"
    coverage = root / "stage18_4e"
    stage18_5 = root / "stage18_5"
    stage18_6 = root / "stage18_6"
    coverage.mkdir(parents=True, exist_ok=True)
    stage18_5.mkdir(parents=True, exist_ok=True)
    stage18_6.mkdir(parents=True, exist_ok=True)
    profile_hash = coverage_profile_hash or profile.profile_hash
    normalized_config = {
        "candidate_refresh_mode": "dynamic_frontier_nbv_in_process",
        "dynamic_candidate_generation_mode": "map_aware_coverage_frontier_nbv",
        "dynamic_candidate_selection_mode": "validated_pareto_diverse",
        "dynamic_candidate_validation_mode": "in_process_path_planner_astar_batch",
        "coverage_metric_mode": "path_line_plus_endpoint",
        "include_oracle_baselines": True,
        "include_roi_weighted_coverage": True,
        "rollout_steps": 40,
        "required_scenario_count": 24,
        "dynamic_max_candidates_per_step": count,
        "dynamic_proposal_pool_limit_per_step": proposal_pool,
        "emit_candidate_metric_audit": True,
    }
    normalized_config.update(normalized_config_overrides or {})
    candidate_rows = _candidate_rows(profile, profile_hash, count, safe_candidates=safe_candidates)
    paired = {
        "schema_version": "xunce-exploration-coverage-paired-decision-audit/v1",
        "scenario_id": "s0",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": "covered-0",
        "candidate_set_id": "set-0",
        "candidate_set_hash": "hash-0",
        "same_state_same_candidate_set": True,
        "policy_disagreement": True,
        "xunce_selected_action_index": 0 if xunce_selects_safe else 1,
        "incumbent_selected_action_index": 0 if incumbent_selects_safe else 1,
    }
    coverage_summary = {
        "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
        "status": "passed",
        "coverage_comparison_root": str(coverage.resolve()),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile_hash,
        "normalized_config": normalized_config,
        **normalized_config,
        "candidate_metric_audit": str((coverage / "xunce-exploration-coverage-candidate-metric-audit.jsonl").resolve()),
        "candidate_metric_audit_row_count": len(candidate_rows),
        "candidate_generation_exhausted_count": 0,
        "dynamic_validation_attempt_count": count,
        "dynamic_validation_success_count": count,
        "candidate_set_hash_mismatch_count": 0,
        "model_inference_failure_count": 0,
        "mask_violation_count": 0,
        "runtime_budget_exceeded": runtime_budget_exceeded,
        "utility_profile_summary": {"coverage_first": {"winner": utility_profile_winner}} if utility_profile_winner else {},
    }
    aggregate = {
        "schema_version": "xunce-exploration-coverage-comparison-aggregate/v1",
        "coverage_delta_cells_mean": 12.0,
        "path_cost_delta_m_mean": 4.0,
        "risk_delta_mean": 0.2,
        "risk_cost_weighted_delta_mean": 3.0,
        "coverage_per_100m_delta_mean": 0.1,
    }
    pair = {
        "schema_version": "xunce-exploration-coverage-comparison-pair/v1",
        "scenario_id": "s0",
        "coverage_delta_cells": 12.0,
        "path_cost_delta_m": 4.0,
        "risk_delta": 0.2,
        "risk_cost_weighted_delta": 3.0,
        "coverage_per_100m_delta": 0.1,
        "utility_profile_outcomes": {"coverage_first": {"winner": utility_profile_winner}} if utility_profile_winner else {},
    }
    _write_json(coverage / "xunce-exploration-coverage-comparison-summary.json", coverage_summary)
    _write_json(coverage / "xunce-exploration-coverage-comparison-aggregate.json", aggregate)
    _write_jsonl(coverage / "xunce-exploration-coverage-comparison-pairs.jsonl", [pair])
    _write_jsonl(coverage / "xunce-exploration-coverage-episodes.jsonl", [{"scenario_id": "s0"}])
    _write_jsonl(coverage / "xunce-exploration-coverage-paired-decision-audit.jsonl", [paired])
    _write_jsonl(coverage / "xunce-exploration-coverage-candidate-metric-audit.jsonl", candidate_rows)

    stage18_5_summary = {
        "schema_version": "xunce-stage18-5-evidence-attribution-summary/v1",
        "status": "passed",
        "coverage_comparison_root": str(coverage.resolve()),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "next_stage_routing": {
            "schema_version": "xunce-stage18-5-next-stage-routing/v1",
            "primary_route": "refine_coverage_reward_and_cost_guard",
            "stage19_authorized": False,
        },
        "stage19_readiness": {"readiness": "not_authorized", "authorized": False},
    }
    _write_json(stage18_5 / "xunce-stage18-5-evidence-attribution-summary.json", stage18_5_summary)
    stage18_6_summary = {
        "schema_version": "xunce-stage18-6-guard-refinement-summary/v1",
        "status": "passed",
        "coverage_comparison_root": str(coverage.resolve()),
        "stage18_5_attribution_root": str(stage18_5.resolve()),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "guard_refinement_passed": guard_refinement_passed,
        "counterfactual_reselection_claimed": False,
        "candidate_metric_readiness": {
            "schema_version": "xunce-stage18-6-candidate-metric-readiness/v1",
            "full_candidate_metric_replay_available": stage18_6_candidate_metric_replay_available,
            "counterfactual_reselection_claim_allowed": stage18_6_candidate_metric_replay_available,
            "reason_codes": [] if stage18_6_candidate_metric_replay_available else ["missing_candidate_metric_audit"],
        },
        "paired_decision_summary": {
            "same_candidate_set_guard_clean_advantage_established": same_candidate_set_guard_clean_advantage_established,
        },
        "next_stage_routing": {
            "schema_version": "xunce-stage18-6-next-stage-routing/v1",
            "primary_route": stage18_6_route,
            "stage19_authorized": False,
        },
        "stage19_readiness": {"readiness": "not_authorized", "authorized": False},
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
    stage18_6_summary.update(stage18_6_boundary_overrides or {})
    _write_json(stage18_6 / "xunce-stage18-6-guard-refinement-summary.json", stage18_6_summary)
    return {
        "candidate_count": count,
        "proposal_pool_limit": proposal_pool,
        "coverage_comparison_root": str(coverage),
        "stage18_5_attribution_root": str(stage18_5),
        "stage18_6_guard_refinement_root": str(stage18_6),
    }


def _candidate_rows(profile, profile_hash: str, count: int, *, safe_candidates: bool) -> list[dict]:
    rows = []
    for index in range(max(2, count)):
        safe = safe_candidates and index == 0
        coverage = 8.0 if safe else 6.0
        path_cost = 5.0 if safe else 80.0
        risk = 1.2 if safe else 1.0
        rows.append(
            {
                "schema_version": "xunce-exploration-coverage-candidate-metric-audit-row/v1",
                "scenario_id": "s0",
                "step_index": 0,
                "current_cell": [0, 0],
                "covered_cells_hash": "covered-0",
                "candidate_set_id": "set-0",
                "candidate_set_hash": "hash-0",
                "candidate_index": index,
                "candidate_cell": [index, index],
                "action_mask_valid": True,
                "expected_new_coverage_cell_count": coverage,
                "roi_weighted_coverage_delta": coverage,
                "path_cost": path_cost,
                "risk": risk,
                "risk_proxy": risk,
                "risk_cost_weighted": path_cost * risk,
                "risk_source": "proxy",
                "risk_proxy_source": "proxy",
                "coverage_gain_per_path_cost": coverage / path_cost,
                "profile_id": profile.profile_id,
                "profile_version": profile.profile_version,
                "profile_hash": profile_hash,
            }
        )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
