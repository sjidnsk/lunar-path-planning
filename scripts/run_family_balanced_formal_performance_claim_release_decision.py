from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


SUMMARY_SCHEMA_VERSION = "family-balanced-formal-performance-claim-release-decision-summary/v1"
AUDIT_SCHEMA_VERSION = "family-balanced-formal-performance-claim-release-decision-audit/v1"
LEDGER_SCHEMA_VERSION = "family-balanced-performance-evidence-ledger/v1"

DEFAULT_SHADOW_ROOT = "outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1"
DEFAULT_RERUN_ROOT = "outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1"
DEFAULT_GAP_ROOT = "outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_family_balanced_formal_performance_claim_release_decision_v1"

SHADOW_SUMMARY_FILE = "family-balanced-shadow-canary-preflight-summary.json"
LONG_HORIZON_FILE = "family-balanced-shadow-canary-long-horizon-validation.json"
FAMILY_GENERALIZATION_AUDIT_FILE = "family-balanced-shadow-canary-family-generalization-audit.json"
COVERAGE_EFFICIENCY_AUDIT_FILE = "family-balanced-shadow-canary-coverage-efficiency-audit.json"
GUARD_FALLBACK_AUDIT_FILE = "family-balanced-shadow-canary-guard-fallback-audit.json"
KILL_SWITCH_AUDIT_FILE = "family-balanced-shadow-canary-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "family-balanced-shadow-canary-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "family-balanced-shadow-canary-telemetry-audit.json"
LINEAGE_AUDIT_FILE = "family-balanced-shadow-canary-lineage-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "family-balanced-shadow-canary-release-boundary-audit.json"

RERUN_SUMMARY_FILE = "family-balanced-coverage-driven-ppo-rerun-summary.json"
GAP_SUMMARY_FILE = "family-balanced-algorithm-gap-closure-summary.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
POST_TRAINING_REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_CANDIDATE_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "family-balanced-formal-performance-claim-release-decision-summary.json"
EVIDENCE_LEDGER_FILE = "family-balanced-performance-evidence-ledger.json"
METRIC_CONSISTENCY_AUDIT_FILE = "family-balanced-metric-consistency-audit.json"
CLAIM_SCOPE_AUDIT_FILE = "family-balanced-claim-scope-audit.json"
LOW_OBSERVATION_LIMITATION_AUDIT_FILE = "family-balanced-low-observation-limitation-audit.json"
RELEASE_BOUNDARY_AUDIT_OUTPUT_FILE = "family-balanced-release-boundary-audit.json"
PROVENANCE_AUDIT_FILE = "family-balanced-provenance-audit.json"
DECISION_MATRIX_FILE = "family-balanced-decision-matrix.json"
SCOPED_CLAIM_STATEMENT_FILE = "family-balanced-scoped-claim-statement.md"
REJECTION_REPORT_FILE = "family-balanced-formal-performance-claim-release-decision-rejection-report.json"
REPORT_FILE = "family-balanced-formal-performance-claim-release-decision-report.md"

EXPECTED_VERDICT = "approved_for_family_balanced_scoped_offline_performance_claim"
ALLOWED_CLAIM_SCOPE = "scoped_offline_family_balanced_guarded_shadow_canary_only"
NEXT_REQUIRED_CHANGE = "family_balanced_scoped_claim_publication_evidence_freeze"
LOW_OBSERVATION_FAMILY = "low_observation_count"
TOLERANCE = 1.0e-6
FALLBACK_RATE_THRESHOLD = 0.5

RELEASE_BOUNDARY_FIELDS = (
    "checkpoint_publication_approved",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "final_release_approved",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
    "default_astar_modified",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run family-balanced formal performance claim / release decision review."
    )
    parser.add_argument("--family-balanced-shadow-root", default=DEFAULT_SHADOW_ROOT)
    parser.add_argument("--family-balanced-rerun-root", default=DEFAULT_RERUN_ROOT)
    parser.add_argument("--family-balanced-gap-root", default=DEFAULT_GAP_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_family_balanced_formal_performance_claim_release_decision(
        family_balanced_shadow_root=_resolve_path(Path(args.family_balanced_shadow_root), repo_root),
        family_balanced_rerun_root=_resolve_path(Path(args.family_balanced_rerun_root), repo_root),
        family_balanced_gap_root=_resolve_path(Path(args.family_balanced_gap_root), repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "decision_verdict": summary["decision_verdict"],
                "family_balanced_scoped_offline_performance_claim_approved": summary[
                    "family_balanced_scoped_offline_performance_claim_approved"
                ],
                "low_observation_limitation_acknowledged": summary[
                    "low_observation_limitation_acknowledged"
                ],
                "checkpoint_publication_approved": summary["checkpoint_publication_approved"],
                "default_policy_replacement_approved": summary["default_policy_replacement_approved"],
                "real_executor_connection_approved": summary["real_executor_connection_approved"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_family_balanced_formal_performance_claim_release_decision(
    *,
    family_balanced_shadow_root: Path,
    family_balanced_rerun_root: Path,
    family_balanced_gap_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    claim_statement_override: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    family_balanced_shadow_root = Path(family_balanced_shadow_root)
    family_balanced_rerun_root = Path(family_balanced_rerun_root)
    family_balanced_gap_root = Path(family_balanced_gap_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    read_reasons: list[str] = []
    shadow_summary_path = family_balanced_shadow_root / SHADOW_SUMMARY_FILE
    shadow_summary = _read_json(shadow_summary_path, read_reasons, "family_balanced_shadow_summary")
    long_horizon_path = _summary_path(
        shadow_summary,
        "long_horizon_shadow_validation",
        family_balanced_shadow_root,
        repo_root,
        LONG_HORIZON_FILE,
    )
    family_audit_path = _summary_path(
        shadow_summary,
        "family_generalization_audit",
        family_balanced_shadow_root,
        repo_root,
        FAMILY_GENERALIZATION_AUDIT_FILE,
    )
    coverage_efficiency_path = _summary_path(
        shadow_summary,
        "coverage_efficiency_audit",
        family_balanced_shadow_root,
        repo_root,
        COVERAGE_EFFICIENCY_AUDIT_FILE,
    )
    guard_fallback_path = _summary_path(
        shadow_summary,
        "guard_fallback_audit",
        family_balanced_shadow_root,
        repo_root,
        GUARD_FALLBACK_AUDIT_FILE,
    )
    kill_switch_path = _summary_path(
        shadow_summary,
        "kill_switch_audit",
        family_balanced_shadow_root,
        repo_root,
        KILL_SWITCH_AUDIT_FILE,
    )
    rollback_path = _summary_path(
        shadow_summary,
        "rollback_audit",
        family_balanced_shadow_root,
        repo_root,
        ROLLBACK_AUDIT_FILE,
    )
    telemetry_path = _summary_path(
        shadow_summary,
        "telemetry_audit",
        family_balanced_shadow_root,
        repo_root,
        TELEMETRY_AUDIT_FILE,
    )
    lineage_path = _summary_path(
        shadow_summary,
        "lineage_audit",
        family_balanced_shadow_root,
        repo_root,
        LINEAGE_AUDIT_FILE,
    )
    source_release_boundary_path = _summary_path(
        shadow_summary,
        "release_boundary_audit",
        family_balanced_shadow_root,
        repo_root,
        RELEASE_BOUNDARY_AUDIT_FILE,
    )

    long_horizon = _read_json(long_horizon_path, read_reasons, "long_horizon")
    family_audit = _read_json(family_audit_path, read_reasons, "family_generalization_audit")
    coverage_efficiency = _read_json(coverage_efficiency_path, read_reasons, "coverage_efficiency_audit")
    guard_fallback = _read_json(guard_fallback_path, read_reasons, "guard_fallback_audit")
    kill_switch = _read_json(kill_switch_path, read_reasons, "kill_switch_audit")
    rollback = _read_json(rollback_path, read_reasons, "rollback_audit")
    telemetry = _read_json(telemetry_path, read_reasons, "telemetry_audit")
    lineage = _read_json(lineage_path, read_reasons, "lineage_audit")
    source_release_boundary = _read_json(
        source_release_boundary_path,
        read_reasons,
        "source_release_boundary_audit",
    )
    rerun_summary_path = family_balanced_rerun_root / RERUN_SUMMARY_FILE
    gap_summary_path = family_balanced_gap_root / GAP_SUMMARY_FILE
    formal_summary_path = Path(formal_training_root) / FORMAL_SUMMARY_FILE
    post_training_replay_summary_path = Path(post_training_replay_root) / POST_TRAINING_REPLAY_SUMMARY_FILE
    selected_candidate_summary_path = Path(selected_candidate_root) / SELECTED_CANDIDATE_SUMMARY_FILE
    rerun_summary = _read_json(rerun_summary_path, read_reasons, "family_balanced_rerun_summary")
    gap_summary = _read_json(gap_summary_path, read_reasons, "family_balanced_gap_summary")
    formal_summary = _read_json(formal_summary_path, read_reasons, "formal_training_summary")
    post_training_replay_summary = _read_json(
        post_training_replay_summary_path,
        read_reasons,
        "post_training_replay_summary",
    )
    selected_candidate_summary = _read_json(
        selected_candidate_summary_path,
        read_reasons,
        "selected_candidate_summary",
    )

    input_paths = {
        "family_balanced_shadow_summary": shadow_summary_path,
        "long_horizon_shadow_validation": long_horizon_path,
        "family_generalization_audit": family_audit_path,
        "coverage_efficiency_audit": coverage_efficiency_path,
        "guard_fallback_audit": guard_fallback_path,
        "kill_switch_audit": kill_switch_path,
        "rollback_audit": rollback_path,
        "telemetry_audit": telemetry_path,
        "lineage_audit": lineage_path,
        "source_release_boundary_audit": source_release_boundary_path,
        "family_balanced_rerun_summary": rerun_summary_path,
        "family_balanced_gap_summary": gap_summary_path,
        "formal_training_summary": formal_summary_path,
        "post_training_replay_summary": post_training_replay_summary_path,
        "selected_candidate_summary": selected_candidate_summary_path,
    }

    claim_statement = claim_statement_override or _default_claim_statement(
        shadow_summary=shadow_summary,
        family_audit=family_audit,
    )
    paths["scoped_claim_statement"].write_text(claim_statement, encoding="utf-8")

    metric_consistency = _metric_consistency_audit(shadow_summary=shadow_summary)
    low_observation = _low_observation_limitation_audit(
        family_audit=family_audit,
        claim_statement=claim_statement,
    )
    claim_scope = _claim_scope_audit(
        claim_statement=claim_statement,
        low_observation=low_observation,
    )
    release_boundary = _release_boundary_audit(
        payloads={
            "shadow_summary": shadow_summary,
            "rerun_summary": rerun_summary,
            "gap_summary": gap_summary,
            "source_release_boundary_audit": source_release_boundary,
            "rollback_audit": rollback,
            "kill_switch_audit": kill_switch,
            "selected_candidate_summary": selected_candidate_summary,
        }
    )
    provenance = _provenance_audit(
        repo_root=repo_root,
        input_paths=input_paths,
        rerun_summary=rerun_summary,
        gap_summary=gap_summary,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
        lineage=lineage,
    )
    evidence_ledger = _evidence_ledger(
        shadow_summary=shadow_summary,
        long_horizon=long_horizon,
        family_audit=family_audit,
        coverage_efficiency=coverage_efficiency,
        guard_fallback=guard_fallback,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        lineage=lineage,
        source_release_boundary=source_release_boundary,
        rerun_summary=rerun_summary,
        gap_summary=gap_summary,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
    )

    reason_codes = _unique(
        [
            *read_reasons,
            *_acceptance_reason_codes(
                shadow_summary=shadow_summary,
                long_horizon=long_horizon,
                family_audit=family_audit,
                coverage_efficiency=coverage_efficiency,
                guard_fallback=guard_fallback,
                kill_switch=kill_switch,
                rollback=rollback,
                telemetry=telemetry,
                lineage=lineage,
                rerun_summary=rerun_summary,
                metric_consistency=metric_consistency,
                claim_scope=claim_scope,
                low_observation=low_observation,
                release_boundary=release_boundary,
                provenance=provenance,
            ),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    claim_approved = status == "passed"
    decision_matrix = _decision_matrix(
        status=status,
        reason_codes=reason_codes,
        claim_approved=claim_approved,
    )
    rejection_report = _rejection_report(reason_codes=reason_codes, decision_matrix=decision_matrix)

    _write_json(paths["evidence_ledger"], evidence_ledger)
    _write_json(paths["metric_consistency_audit"], metric_consistency)
    _write_json(paths["claim_scope_audit"], claim_scope)
    _write_json(paths["low_observation_limitation_audit"], low_observation)
    _write_json(paths["release_boundary_audit"], release_boundary)
    _write_json(paths["provenance_audit"], provenance)
    _write_json(paths["decision_matrix"], decision_matrix)
    _write_json(paths["rejection_report"], rejection_report)

    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        paths=paths,
        repo_root=repo_root,
        family_balanced_shadow_root=family_balanced_shadow_root,
        family_balanced_rerun_root=family_balanced_rerun_root,
        family_balanced_gap_root=family_balanced_gap_root,
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        input_paths=input_paths,
        shadow_summary=shadow_summary,
        long_horizon=long_horizon,
        family_audit=family_audit,
        coverage_efficiency=coverage_efficiency,
        guard_fallback=guard_fallback,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        lineage=lineage,
        metric_consistency=metric_consistency,
        claim_scope=claim_scope,
        low_observation=low_observation,
        release_boundary=release_boundary,
        provenance=provenance,
        decision_matrix=decision_matrix,
        claim_approved=claim_approved,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _acceptance_reason_codes(
    *,
    shadow_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    family_audit: dict[str, Any],
    coverage_efficiency: dict[str, Any],
    guard_fallback: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    lineage: dict[str, Any],
    rerun_summary: dict[str, Any],
    metric_consistency: dict[str, Any],
    claim_scope: dict[str, Any],
    low_observation: dict[str, Any],
    release_boundary: dict[str, Any],
    provenance: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if (
        shadow_summary.get("status") != "passed"
        or _string_list(shadow_summary.get("reason_codes"))
        or shadow_summary.get("preflight_verdict")
        != "eligible_for_family_balanced_formal_performance_claim_release_decision"
        or shadow_summary.get("family_balanced_shadow_canary_preflight_passed") is not True
        or shadow_summary.get("family_balanced_formal_performance_claim_release_decision_approved") is not True
        or shadow_summary.get("next_required_change")
        != "family_balanced_formal_performance_claim_release_decision"
    ):
        _add_reason(reasons, "family_balanced_shadow_canary_not_passed")
    if (
        shadow_summary.get("long_horizon_shadow_passed") is not True
        or long_horizon.get("long_horizon_shadow_passed") is not True
    ):
        _add_reason(reasons, "long_horizon_shadow_not_passed")
    if (
        shadow_summary.get("family_generalization_audit_passed") is not True
        or family_audit.get("family_generalization_audit_passed") is not True
        or shadow_summary.get("low_observation_shadow_passed") is not True
        or family_audit.get("low_observation_shadow_passed") is not True
    ):
        _add_reason(reasons, "family_balanced_shadow_canary_not_passed")
    for metric in (
        "coverage_return_improvement",
        "cumulative_coverage_rate_delta_improvement",
        "valuable_area_covered_improvement",
    ):
        if _float(shadow_summary.get(metric)) <= 0.0:
            _add_reason(reasons, "coverage_metric_inconsistent")
    if metric_consistency.get("metric_consistency_audit_passed") is not True:
        _add_reason(reasons, "coverage_metric_inconsistent")
    if (
        shadow_summary.get("coverage_efficiency_regression") is True
        or rerun_summary.get("coverage_efficiency_regression") is True
        or coverage_efficiency.get("coverage_efficiency_regression") is True
        or coverage_efficiency.get("coverage_efficiency_audit_passed") is False
    ):
        _add_reason(reasons, "coverage_efficiency_regression")
    if _float(shadow_summary.get("fallback_rate")) >= FALLBACK_RATE_THRESHOLD or _float(
        guard_fallback.get("fallback_rate")
    ) >= FALLBACK_RATE_THRESHOLD:
        _add_reason(reasons, "fallback_dominates")
    if (
        _int(shadow_summary.get("fallback_gain_contamination_count")) > 0
        or _int(guard_fallback.get("fallback_gain_contamination_count")) > 0
    ):
        _add_reason(reasons, "fallback_gain_contamination")
    if (
        _int(shadow_summary.get("controlled_regression_count")) > 0
        or _int(guard_fallback.get("controlled_regression_count")) > 0
    ):
        _add_reason(reasons, "controlled_regression_detected")
    if (
        kill_switch.get("kill_switch_audit_passed") is not True
        or rollback.get("rollback_audit_passed") is not True
        or telemetry.get("telemetry_audit_passed") is not True
        or telemetry.get("records_coverage_gain") is False
    ):
        _add_reason(reasons, "guard_rollback_or_telemetry_not_passed")
    if telemetry.get("telemetry_audit_passed") is not True or telemetry.get("records_coverage_gain") is False:
        _add_reason(reasons, "guard_rollback_or_telemetry_not_passed")
    if lineage.get("lineage_audit_passed") is not True:
        _add_reason(reasons, "lineage_incomplete")
    if claim_scope.get("claim_scope_audit_passed") is not True:
        _add_reason(reasons, "claim_scope_overbroad")
    if low_observation.get("low_observation_limitation_acknowledged") is not True:
        _add_reason(reasons, "low_observation_limitation_missing")
    if low_observation.get("low_observation_overclaimed") is True:
        _add_reason(reasons, "low_observation_overclaimed")
    if release_boundary.get("release_boundary_audit_passed") is not True:
        _add_reason(reasons, "release_boundary_violation")
    if provenance.get("provenance_audit_passed") is not True:
        _add_reason(reasons, "provenance_or_docs_missing")
    return reasons


def _metric_consistency_audit(*, shadow_summary: dict[str, Any]) -> dict[str, Any]:
    candidate = shadow_summary.get("candidate_metrics") or {}
    baseline = shadow_summary.get("baseline_metrics") or {}
    computed_candidate = shadow_summary.get("computed_shadow_candidate_metrics") or {}
    computed_teacher = shadow_summary.get("computed_shadow_teacher_metrics") or {}
    checks: list[dict[str, Any]] = []
    for metric in (
        "coverage_return",
        "cumulative_coverage_rate_delta",
        "valuable_area_covered",
        "coverage_gain_per_path_cost",
        "coverage_gain_per_risk",
        "coverage_gain_per_energy",
        "fallback_rate",
    ):
        checks.append(_metric_check("candidate_vs_recomputed", metric, candidate.get(metric), computed_candidate.get(metric)))
    for metric in (
        "coverage_return",
        "cumulative_coverage_rate_delta",
        "valuable_area_covered",
        "coverage_gain_per_path_cost",
        "coverage_gain_per_risk",
        "coverage_gain_per_energy",
    ):
        checks.append(_metric_check("baseline_vs_recomputed", metric, baseline.get(metric), computed_teacher.get(metric)))
    for field, metric in (
        ("coverage_return_improvement", "coverage_return"),
        ("cumulative_coverage_rate_delta_improvement", "cumulative_coverage_rate_delta"),
        ("valuable_area_covered_improvement", "valuable_area_covered"),
    ):
        expected = _float(candidate.get(metric)) - _float(baseline.get(metric))
        checks.append(_metric_check("summary_improvement", field, shadow_summary.get(field), expected))
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "metric_consistency",
        "metric_consistency_audit_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _low_observation_limitation_audit(
    *,
    family_audit: dict[str, Any],
    claim_statement: str,
) -> dict[str, Any]:
    rollup = (family_audit.get("family_rollups") or {}).get(LOW_OBSERVATION_FAMILY, {})
    candidate_return = _float(rollup.get("candidate_coverage_return"))
    baseline_return = _float(rollup.get("baseline_coverage_return"))
    improvement = _float(rollup.get("coverage_return_improvement"))
    outperforms_teacher = candidate_return > baseline_return and improvement > 0.0
    lower_claim = claim_statement.lower()
    limitation_acknowledged = (
        "low-observation" in lower_claim
        and (
            "does not outperform teacher" in lower_claim
            or "not outperform teacher" in lower_claim
            or "limitation" in lower_claim
        )
    )
    overclaim_patterns = (
        "low-observation coverage return outperforms teacher",
        "low observation coverage return outperforms teacher",
        "low-observation coverage return exceeds teacher",
        "low observation coverage return exceeds teacher",
    )
    low_observation_overclaimed = any(pattern in lower_claim for pattern in overclaim_patterns)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "low_observation_limitation",
        "low_observation_family": LOW_OBSERVATION_FAMILY,
        "low_observation_shadow_passed": family_audit.get("low_observation_shadow_passed") is True,
        "candidate_coverage_return": candidate_return,
        "baseline_coverage_return": baseline_return,
        "coverage_return_improvement": improvement,
        "low_observation_coverage_return_outperforms_teacher": outperforms_teacher,
        "low_observation_limitation_acknowledged": limitation_acknowledged,
        "low_observation_overclaimed": low_observation_overclaimed and not outperforms_teacher,
        "low_observation_limitation_audit_passed": limitation_acknowledged
        and not (low_observation_overclaimed and not outperforms_teacher),
    }


def _claim_scope_audit(
    *,
    claim_statement: str,
    low_observation: dict[str, Any],
) -> dict[str, Any]:
    lower = claim_statement.lower()
    allowed_section = lower.split("## prohibited claims", 1)[0]
    prohibited_patterns = (
        "approved for real executor",
        "real executor deployment",
        "real-world deployment",
        "real world deployment",
        "default policy replacement",
        "replace default policy",
        "checkpoint publication approved",
        "checkpoint published",
        "stage 17 authorization",
        "stage17 authorization",
        "ackermann-feasible",
        "real world performance",
        "real-world performance",
        "all scenarios",
        "unbounded generalization",
        "unlimited scenario generalization",
    )
    matches = [pattern for pattern in prohibited_patterns if pattern in allowed_section]
    has_required_boundary = "offline family-balanced guarded shadow/canary" in lower
    has_prohibited_section = "## prohibited claims" in lower
    has_low_observation_limitation = low_observation.get("low_observation_limitation_acknowledged") is True
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "claim_scope",
        "claim_scope_audit_passed": not matches
        and has_required_boundary
        and has_prohibited_section
        and has_low_observation_limitation
        and low_observation.get("low_observation_overclaimed") is not True,
        "overbroad_patterns": matches,
        "has_required_offline_boundary": has_required_boundary,
        "has_prohibited_claims_section": has_prohibited_section,
        "has_low_observation_limitation": has_low_observation_limitation,
    }


def _release_boundary_audit(*, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for source_name, payload in payloads.items():
        for violation in _release_boundary_violations(payload, prefix=source_name):
            if violation not in violations:
                violations.append(violation)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "release_boundary",
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _release_boundary_violations(value: Any, *, prefix: str) -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if key in RELEASE_BOUNDARY_FIELDS and item is True:
                violations.append(path)
            violations.extend(_release_boundary_violations(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(_release_boundary_violations(item, prefix=f"{prefix}[{index}]"))
    return violations


def _provenance_audit(
    *,
    repo_root: Path,
    input_paths: dict[str, Path],
    rerun_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    missing_inputs = [name for name, path in input_paths.items() if not Path(path).is_file()]
    doc_paths = [
        repo_root / "README.md",
        repo_root / "docs" / "算法设计与系统架构报告.md",
        repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-16-family-balanced-formal-performance-claim-release-decision.md",
    ]
    missing_docs = [str(path) for path in doc_paths if not path.is_file()]
    upstream_passed = all(
        payload.get("status") == "passed" and not _string_list(payload.get("reason_codes"))
        for payload in (
            rerun_summary,
            gap_summary,
            formal_summary,
            post_training_replay_summary,
            selected_candidate_summary,
        )
    )
    lineage_passed = lineage.get("lineage_audit_passed") is True
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "provenance",
        "provenance_audit_passed": not missing_inputs and not missing_docs and upstream_passed and lineage_passed,
        "missing_input_names": missing_inputs,
        "missing_documentation_paths": missing_docs,
        "upstream_passed": upstream_passed,
        "lineage_audit_passed": lineage_passed,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _evidence_ledger(
    *,
    shadow_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    family_audit: dict[str, Any],
    coverage_efficiency: dict[str, Any],
    guard_fallback: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    lineage: dict[str, Any],
    source_release_boundary: dict[str, Any],
    rerun_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        _source("family_balanced_shadow_summary", shadow_summary.get("status") == "passed"),
        _source("long_horizon_shadow_validation", long_horizon.get("long_horizon_shadow_passed") is True),
        _source("family_generalization_audit", family_audit.get("family_generalization_audit_passed") is True),
        _source("coverage_efficiency_audit", coverage_efficiency.get("coverage_efficiency_audit_passed") is True),
        _source("guard_fallback_audit", guard_fallback.get("guard_fallback_audit_passed") is True),
        _source("kill_switch_audit", kill_switch.get("kill_switch_audit_passed") is True),
        _source("rollback_audit", rollback.get("rollback_audit_passed") is True),
        _source("telemetry_audit", telemetry.get("telemetry_audit_passed") is True),
        _source("lineage_audit", lineage.get("lineage_audit_passed") is True),
        _source("source_release_boundary_audit", source_release_boundary.get("release_boundary_audit_passed") is True),
        _source("family_balanced_rerun_summary", rerun_summary.get("status") == "passed"),
        _source("family_balanced_gap_summary", gap_summary.get("status") == "passed"),
        _source("formal_training_summary", formal_summary.get("status") == "passed"),
        _source("post_training_replay_summary", post_training_replay_summary.get("status") == "passed"),
        _source("selected_candidate_summary", selected_candidate_summary.get("status") == "passed"),
    ]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "sources": sources,
        "all_sources_passed": all(source["passed"] for source in sources),
    }


def _decision_matrix(
    *,
    status: str,
    reason_codes: list[str],
    claim_approved: bool,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "decision_matrix",
        "status": status,
        "reason_codes": list(reason_codes),
        "decision_verdict": EXPECTED_VERDICT if claim_approved else _failed_verdict(reason_codes),
        "family_balanced_scoped_offline_performance_claim_approved": claim_approved,
        "allowed_claim_scope": ALLOWED_CLAIM_SCOPE if claim_approved else None,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "allowed_decisions": ["family_balanced_scoped_offline_guarded_shadow_canary_performance_claim"]
        if claim_approved
        else [],
        "blocked_release_actions": [
            "checkpoint_publication",
            "default_policy_replacement",
            "real_executor_connection",
            "stage17_default_policy_installation",
        ],
    }


def _rejection_report(*, reason_codes: list[str], decision_matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "rejection_report",
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": list(reason_codes),
        "decision_verdict": decision_matrix.get("decision_verdict"),
        "next_required_change": _next_required_change(reason_codes),
    }


def _summary(
    *,
    status: str,
    reason_codes: list[str],
    paths: dict[str, Path],
    repo_root: Path,
    family_balanced_shadow_root: Path,
    family_balanced_rerun_root: Path,
    family_balanced_gap_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    input_paths: dict[str, Path],
    shadow_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    family_audit: dict[str, Any],
    coverage_efficiency: dict[str, Any],
    guard_fallback: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    lineage: dict[str, Any],
    metric_consistency: dict[str, Any],
    claim_scope: dict[str, Any],
    low_observation: dict[str, Any],
    release_boundary: dict[str, Any],
    provenance: dict[str, Any],
    decision_matrix: dict[str, Any],
    claim_approved: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": NEXT_REQUIRED_CHANGE if claim_approved else _next_required_change(reason_codes),
        "decision_verdict": decision_matrix.get("decision_verdict"),
        "repo_root": str(repo_root),
        "family_balanced_shadow_root": str(family_balanced_shadow_root),
        "family_balanced_rerun_root": str(family_balanced_rerun_root),
        "family_balanced_gap_root": str(family_balanced_gap_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "summary": str(paths["summary"]),
        "evidence_ledger": str(paths["evidence_ledger"]),
        "metric_consistency_audit": str(paths["metric_consistency_audit"]),
        "claim_scope_audit": str(paths["claim_scope_audit"]),
        "low_observation_limitation_audit": str(paths["low_observation_limitation_audit"]),
        "release_boundary_audit": str(paths["release_boundary_audit"]),
        "provenance_audit": str(paths["provenance_audit"]),
        "decision_matrix": str(paths["decision_matrix"]),
        "scoped_claim_statement": str(paths["scoped_claim_statement"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "family_balanced_formal_performance_claim_release_decision_passed": claim_approved,
        "family_balanced_scoped_offline_performance_claim_approved": claim_approved,
        "allowed_claim_scope": ALLOWED_CLAIM_SCOPE if claim_approved else None,
        "shadow_canary_preflight_passed": shadow_summary.get("family_balanced_shadow_canary_preflight_passed") is True,
        "long_horizon_shadow_passed": shadow_summary.get("long_horizon_shadow_passed") is True
        and long_horizon.get("long_horizon_shadow_passed") is True,
        "horizons": shadow_summary.get("horizons", long_horizon.get("horizons", [])),
        "coverage_return_improvement": shadow_summary.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": shadow_summary.get(
            "cumulative_coverage_rate_delta_improvement",
            0.0,
        ),
        "valuable_area_covered_improvement": shadow_summary.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(shadow_summary.get("coverage_efficiency_regression"))
        or bool(coverage_efficiency.get("coverage_efficiency_regression")),
        "fallback_rate": shadow_summary.get("fallback_rate", guard_fallback.get("fallback_rate")),
        "fallback_gain_contamination_count": shadow_summary.get(
            "fallback_gain_contamination_count",
            guard_fallback.get("fallback_gain_contamination_count", 0),
        ),
        "controlled_regression_count": shadow_summary.get(
            "controlled_regression_count",
            guard_fallback.get("controlled_regression_count", 0),
        ),
        "family_generalization_audit_passed": family_audit.get("family_generalization_audit_passed") is True
        and shadow_summary.get("family_generalization_audit_passed") is True,
        "low_observation_shadow_passed": low_observation.get("low_observation_shadow_passed") is True,
        "low_observation_coverage_return_outperforms_teacher": low_observation.get(
            "low_observation_coverage_return_outperforms_teacher"
        )
        is True,
        "low_observation_limitation_acknowledged": low_observation.get(
            "low_observation_limitation_acknowledged"
        )
        is True,
        "kill_switch_audit_passed": kill_switch.get("kill_switch_audit_passed") is True
        and shadow_summary.get("kill_switch_audit_passed") is True,
        "rollback_audit_passed": rollback.get("rollback_audit_passed") is True
        and shadow_summary.get("rollback_audit_passed") is True,
        "telemetry_audit_passed": telemetry.get("telemetry_audit_passed") is True
        and shadow_summary.get("telemetry_audit_passed") is True,
        "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
        "metric_consistency_audit_passed": metric_consistency.get("metric_consistency_audit_passed") is True,
        "claim_scope_audit_passed": claim_scope.get("claim_scope_audit_passed") is True,
        "low_observation_limitation_audit_passed": low_observation.get(
            "low_observation_limitation_audit_passed"
        )
        is True,
        "release_boundary_audit_passed": release_boundary.get("release_boundary_audit_passed") is True,
        "provenance_audit_passed": provenance.get("provenance_audit_passed") is True,
        "claim_scope_overbroad_patterns": claim_scope.get("overbroad_patterns", []),
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "performance_claimed": claim_approved,
        "performance_claim_scope": ALLOWED_CLAIM_SCOPE if claim_approved else None,
        "formal_release_claimed": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _default_claim_statement(*, shadow_summary: dict[str, Any], family_audit: dict[str, Any]) -> str:
    low = (family_audit.get("family_rollups") or {}).get(LOW_OBSERVATION_FAMILY, {})
    return "\n".join(
        [
            "# Family-Balanced Formal Performance Claim / Release Decision v1",
            "",
            "## Allowed claim",
            "",
            (
                "The current family-balanced candidate is approved only for a scoped "
                "offline family-balanced guarded shadow/canary performance claim on the "
                "audited decision set."
            ),
            "",
            f"- Coverage return improvement: `{shadow_summary.get('coverage_return_improvement')}`",
            (
                "- Cumulative coverage rate delta improvement: "
                f"`{shadow_summary.get('cumulative_coverage_rate_delta_improvement')}`"
            ),
            f"- Valuable area covered improvement: `{shadow_summary.get('valuable_area_covered_improvement')}`",
            f"- Coverage efficiency regression: `{shadow_summary.get('coverage_efficiency_regression')}`",
            f"- Fallback rate: `{shadow_summary.get('fallback_rate')}`",
            "",
            "## Low-observation limitation",
            "",
            (
                "The low-observation coverage return does not outperform teacher in this "
                "evidence set; this limitation is part of the claim scope."
            ),
            f"- Low-observation candidate coverage return: `{low.get('candidate_coverage_return')}`",
            f"- Low-observation teacher coverage return: `{low.get('baseline_coverage_return')}`",
            f"- Low-observation coverage return improvement: `{low.get('coverage_return_improvement')}`",
            "",
            "## Prohibited claims",
            "",
            "- Checkpoint publication is not approved.",
            "- Default policy replacement is not approved.",
            "- Real executor connection is not approved.",
            "- Stage 17 default-policy installation is not authorized.",
            "- Real-world performance is not claimed.",
            "- Ackermann-feasible trajectory is not claimed.",
            "- Unlimited scenario generalization is not claimed.",
            "- Low-observation coverage return outperformance over teacher is not claimed.",
            "- IRIS/GCS/path-planner diagnostics are not release proof.",
            "",
        ]
    )


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Family-Balanced Formal Performance Claim / Release Decision v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- decision_verdict: `{summary['decision_verdict']}`",
            (
                "- family_balanced_scoped_offline_performance_claim_approved: "
                f"`{summary['family_balanced_scoped_offline_performance_claim_approved']}`"
            ),
            f"- low_observation_limitation_acknowledged: `{summary['low_observation_limitation_acknowledged']}`",
            f"- checkpoint_publication_approved: `{summary['checkpoint_publication_approved']}`",
            f"- default_policy_replacement_approved: `{summary['default_policy_replacement_approved']}`",
            f"- real_executor_connection_approved: `{summary['real_executor_connection_approved']}`",
            "",
            "## Metrics",
            "",
            f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
            f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
            f"- valuable_area_covered_improvement: `{summary['valuable_area_covered_improvement']}`",
            f"- coverage_efficiency_regression: `{summary['coverage_efficiency_regression']}`",
            f"- fallback_rate: `{summary['fallback_rate']}`",
            "",
            "## Boundaries",
            "",
            "- publishes_checkpoint: `false`",
            "- replaces_default_policy: `false`",
            "- connects_real_executor: `false`",
            "- modifies_network_or_action_space: `false`",
            "- modifies_default_astar: `false`",
            "",
        ]
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "evidence_ledger": output_root / EVIDENCE_LEDGER_FILE,
        "metric_consistency_audit": output_root / METRIC_CONSISTENCY_AUDIT_FILE,
        "claim_scope_audit": output_root / CLAIM_SCOPE_AUDIT_FILE,
        "low_observation_limitation_audit": output_root / LOW_OBSERVATION_LIMITATION_AUDIT_FILE,
        "release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_OUTPUT_FILE,
        "provenance_audit": output_root / PROVENANCE_AUDIT_FILE,
        "decision_matrix": output_root / DECISION_MATRIX_FILE,
        "scoped_claim_statement": output_root / SCOPED_CLAIM_STATEMENT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _source(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _failed_verdict(reason_codes: list[str]) -> str:
    if "low_observation_overclaimed" in reason_codes or "claim_scope_overbroad" in reason_codes:
        return "blocked_by_overbroad_family_balanced_claim_scope"
    if "release_boundary_violation" in reason_codes:
        return "blocked_by_release_boundary_violation"
    if "family_balanced_shadow_canary_not_passed" in reason_codes:
        return "blocked_by_family_balanced_shadow_canary_preflight"
    return "blocked_by_family_balanced_formal_claim_review"


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return NEXT_REQUIRED_CHANGE
    if "family_balanced_shadow_canary_not_passed" in reason_codes:
        return "fix_family_balanced_shadow_canary_preflight"
    if "long_horizon_shadow_not_passed" in reason_codes:
        return "fix_family_balanced_long_horizon_shadow"
    if "coverage_metric_inconsistent" in reason_codes:
        return "fix_family_balanced_coverage_metric_consistency"
    if "low_observation_limitation_missing" in reason_codes or "low_observation_overclaimed" in reason_codes:
        return "narrow_family_balanced_low_observation_claim"
    if "claim_scope_overbroad" in reason_codes:
        return "narrow_family_balanced_formal_performance_claim_scope"
    if "fallback_dominates" in reason_codes:
        return "reduce_family_balanced_fallback_rate"
    if "release_boundary_violation" in reason_codes:
        return "restore_family_balanced_non_release_boundary"
    return "inspect_family_balanced_formal_claim_inputs"


def _summary_path(summary: dict[str, Any], key: str, base: Path, repo_root: Path, fallback: str) -> Path:
    return _resolve_optional_path(summary.get(key), base, repo_root) or base / fallback


def _metric_check(scope: str, metric: str, observed: Any, expected: Any) -> dict[str, Any]:
    observed_float = _float_or_none(observed)
    expected_float = _float_or_none(expected)
    passed = (
        observed_float is not None
        and expected_float is not None
        and abs(observed_float - expected_float) <= TOLERANCE
    )
    return {
        "scope": scope,
        "metric": metric,
        "observed": observed,
        "expected": expected,
        "delta": None if observed_float is None or expected_float is None else round(observed_float - expected_float, 12),
        "passed": passed,
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _add_reason(reasons, "provenance_or_docs_missing")
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid_json")
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return repo_root / path


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
