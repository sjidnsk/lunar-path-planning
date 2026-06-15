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


SUMMARY_SCHEMA_VERSION = "formal-performance-claim-release-decision-summary/v1"
AUDIT_SCHEMA_VERSION = "formal-performance-claim-release-decision-audit/v1"
LEDGER_SCHEMA_VERSION = "formal-performance-evidence-ledger/v1"

DEFAULT_STAGE6_ROOT = "outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1"
DEFAULT_COST_EFFICIENCY_ROOT = "outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"

STAGE6_SUMMARY_FILE = "shadow-canary-release-performance-validation-preflight-summary.json"
STAGE6_LONG_HORIZON_FILE = "shadow-canary-long-horizon-validation.json"
STAGE6_COVERAGE_EFFICIENCY_AUDIT_FILE = "shadow-canary-coverage-efficiency-audit.json"
STAGE6_GUARD_FALLBACK_AUDIT_FILE = "shadow-canary-guard-fallback-audit.json"
STAGE6_KILL_SWITCH_AUDIT_FILE = "shadow-canary-kill-switch-audit.json"
STAGE6_ROLLBACK_AUDIT_FILE = "shadow-canary-rollback-audit.json"
STAGE6_TELEMETRY_AUDIT_FILE = "shadow-canary-telemetry-audit.json"
STAGE6_ELIGIBILITY_LEDGER_FILE = "shadow-canary-eligibility-ledger.json"

COST_SUMMARY_FILE = "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json"
COST_METRIC_TABLE_FILE = "refined-coverage-driven-ppo-performance-metric-table.jsonl"
COST_REPLAY_AUDIT_FILE = "coverage-driven-ppo-replay-audit.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
POST_TRAINING_REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_CANDIDATE_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"
EVIDENCE_LEDGER_FILE = "formal-performance-evidence-ledger.json"
METRIC_CONSISTENCY_AUDIT_FILE = "formal-performance-metric-consistency-audit.json"
CLAIM_SCOPE_AUDIT_FILE = "formal-performance-claim-scope-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "formal-performance-release-boundary-audit.json"
PROVENANCE_AUDIT_FILE = "formal-performance-provenance-audit.json"
DECISION_MATRIX_FILE = "formal-performance-decision-matrix.json"
SCOPED_CLAIM_STATEMENT_FILE = "formal-performance-scoped-claim-statement.md"
REJECTION_REPORT_FILE = "formal-performance-rejection-report.json"
REPORT_FILE = "formal-performance-claim-release-decision-report.md"

TOLERANCE = 1.0e-6
FALLBACK_RATE_THRESHOLD = 0.5
EXPECTED_VERDICT = "approved_for_scoped_offline_performance_claim"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run formal scoped offline performance claim / release decision review."
    )
    parser.add_argument("--stage6-root", default=DEFAULT_STAGE6_ROOT)
    parser.add_argument("--cost-efficiency-root", default=DEFAULT_COST_EFFICIENCY_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_formal_performance_claim_release_decision(
        stage6_root=_resolve_path(Path(args.stage6_root), repo_root),
        cost_efficiency_root=_resolve_path(Path(args.cost_efficiency_root), repo_root),
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
                "scoped_offline_performance_claim_approved": summary[
                    "scoped_offline_performance_claim_approved"
                ],
                "checkpoint_publication_approved": summary["checkpoint_publication_approved"],
                "default_policy_replacement_approved": summary["default_policy_replacement_approved"],
                "real_executor_connection_approved": summary["real_executor_connection_approved"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_formal_performance_claim_release_decision(
    *,
    stage6_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    claim_statement_override: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stage6_root = Path(stage6_root)
    cost_efficiency_root = Path(cost_efficiency_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    read_reasons: list[str] = []
    stage6_summary_path = stage6_root / STAGE6_SUMMARY_FILE
    stage6_summary = _read_json(stage6_summary_path, read_reasons, "stage6_summary")
    long_horizon_path = _summary_path(
        stage6_summary,
        "long_horizon_shadow_validation",
        stage6_root,
        repo_root,
        STAGE6_LONG_HORIZON_FILE,
    )
    coverage_efficiency_path = _summary_path(
        stage6_summary,
        "coverage_efficiency_audit",
        stage6_root,
        repo_root,
        STAGE6_COVERAGE_EFFICIENCY_AUDIT_FILE,
    )
    guard_fallback_path = _summary_path(
        stage6_summary,
        "guard_fallback_audit",
        stage6_root,
        repo_root,
        STAGE6_GUARD_FALLBACK_AUDIT_FILE,
    )
    kill_switch_path = _summary_path(
        stage6_summary,
        "kill_switch_audit",
        stage6_root,
        repo_root,
        STAGE6_KILL_SWITCH_AUDIT_FILE,
    )
    rollback_path = _summary_path(
        stage6_summary,
        "rollback_audit",
        stage6_root,
        repo_root,
        STAGE6_ROLLBACK_AUDIT_FILE,
    )
    telemetry_path = _summary_path(
        stage6_summary,
        "telemetry_audit",
        stage6_root,
        repo_root,
        STAGE6_TELEMETRY_AUDIT_FILE,
    )
    eligibility_path = _summary_path(
        stage6_summary,
        "eligibility_ledger",
        stage6_root,
        repo_root,
        STAGE6_ELIGIBILITY_LEDGER_FILE,
    )

    long_horizon = _read_json(long_horizon_path, read_reasons, "stage6_long_horizon")
    coverage_efficiency = _read_json(coverage_efficiency_path, read_reasons, "stage6_coverage_efficiency_audit")
    guard_fallback = _read_json(guard_fallback_path, read_reasons, "stage6_guard_fallback_audit")
    kill_switch = _read_json(kill_switch_path, read_reasons, "stage6_kill_switch_audit")
    rollback = _read_json(rollback_path, read_reasons, "stage6_rollback_audit")
    telemetry = _read_json(telemetry_path, read_reasons, "stage6_telemetry_audit")
    eligibility = _read_json(eligibility_path, read_reasons, "stage6_eligibility_ledger")

    cost_summary_path = cost_efficiency_root / COST_SUMMARY_FILE
    cost_summary = _read_json(cost_summary_path, read_reasons, "cost_efficiency_summary")
    cost_metric_rows = _read_jsonl(cost_efficiency_root / COST_METRIC_TABLE_FILE, read_reasons, "cost_metric_table")
    cost_replay = _read_json(cost_efficiency_root / COST_REPLAY_AUDIT_FILE, read_reasons, "cost_replay_audit")
    formal_summary_path = Path(formal_training_root) / FORMAL_SUMMARY_FILE
    post_training_replay_summary_path = Path(post_training_replay_root) / POST_TRAINING_REPLAY_SUMMARY_FILE
    selected_candidate_summary_path = Path(selected_candidate_root) / SELECTED_CANDIDATE_SUMMARY_FILE
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
        "stage6_summary": stage6_summary_path,
        "stage6_long_horizon": long_horizon_path,
        "stage6_coverage_efficiency_audit": coverage_efficiency_path,
        "stage6_guard_fallback_audit": guard_fallback_path,
        "stage6_kill_switch_audit": kill_switch_path,
        "stage6_rollback_audit": rollback_path,
        "stage6_telemetry_audit": telemetry_path,
        "stage6_eligibility_ledger": eligibility_path,
        "cost_efficiency_summary": cost_summary_path,
        "cost_metric_table": cost_efficiency_root / COST_METRIC_TABLE_FILE,
        "cost_replay_audit": cost_efficiency_root / COST_REPLAY_AUDIT_FILE,
        "formal_training_summary": formal_summary_path,
        "post_training_replay_summary": post_training_replay_summary_path,
        "selected_candidate_summary": selected_candidate_summary_path,
    }

    default_claim_statement = _default_claim_statement(stage6_summary)
    claim_statement = claim_statement_override or default_claim_statement
    paths["scoped_claim_statement"].write_text(claim_statement, encoding="utf-8")

    metric_consistency = _metric_consistency_audit(stage6_summary=stage6_summary)
    claim_scope = _claim_scope_audit(claim_statement=claim_statement)
    release_boundary = _release_boundary_audit(
        stage6_summary=stage6_summary,
        rollback=rollback,
        kill_switch=kill_switch,
        selected_candidate_summary=selected_candidate_summary,
    )
    provenance = _provenance_audit(
        repo_root=repo_root,
        input_paths=input_paths,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
    )
    evidence_ledger = _evidence_ledger(
        stage6_summary=stage6_summary,
        long_horizon=long_horizon,
        coverage_efficiency=coverage_efficiency,
        guard_fallback=guard_fallback,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        eligibility=eligibility,
        cost_summary=cost_summary,
        cost_metric_rows=cost_metric_rows,
        cost_replay=cost_replay,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
    )

    reason_codes = _unique(
        [
            *read_reasons,
            *_acceptance_reason_codes(
                stage6_summary=stage6_summary,
                long_horizon=long_horizon,
                coverage_efficiency=coverage_efficiency,
                guard_fallback=guard_fallback,
                kill_switch=kill_switch,
                rollback=rollback,
                telemetry=telemetry,
                eligibility=eligibility,
                cost_summary=cost_summary,
                metric_consistency=metric_consistency,
                claim_scope=claim_scope,
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
        scoped_offline_performance_claim_approved=claim_approved,
    )
    rejection_report = _rejection_report(reason_codes=reason_codes, decision_matrix=decision_matrix)

    _write_json(paths["evidence_ledger"], evidence_ledger)
    _write_json(paths["metric_consistency_audit"], metric_consistency)
    _write_json(paths["claim_scope_audit"], claim_scope)
    _write_json(paths["release_boundary_audit"], release_boundary)
    _write_json(paths["provenance_audit"], provenance)
    _write_json(paths["decision_matrix"], decision_matrix)
    _write_json(paths["rejection_report"], rejection_report)

    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        paths=paths,
        repo_root=repo_root,
        stage6_root=stage6_root,
        cost_efficiency_root=cost_efficiency_root,
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        input_paths=input_paths,
        stage6_summary=stage6_summary,
        long_horizon=long_horizon,
        coverage_efficiency=coverage_efficiency,
        guard_fallback=guard_fallback,
        kill_switch=kill_switch,
        rollback=rollback,
        telemetry=telemetry,
        eligibility=eligibility,
        metric_consistency=metric_consistency,
        claim_scope=claim_scope,
        release_boundary=release_boundary,
        provenance=provenance,
        decision_matrix=decision_matrix,
        scoped_offline_performance_claim_approved=claim_approved,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _acceptance_reason_codes(
    *,
    stage6_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    coverage_efficiency: dict[str, Any],
    guard_fallback: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    eligibility: dict[str, Any],
    cost_summary: dict[str, Any],
    metric_consistency: dict[str, Any],
    claim_scope: dict[str, Any],
    release_boundary: dict[str, Any],
    provenance: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if (
        stage6_summary.get("status") != "passed"
        or _string_list(stage6_summary.get("reason_codes"))
        or stage6_summary.get("next_required_change") != "formal_performance_claim_release_decision"
        or stage6_summary.get("eligible_for_formal_performance_claim_review") is not True
    ):
        _add_reason(reasons, "stage6_not_passed")
    if stage6_summary.get("long_horizon_shadow_passed") is not True or long_horizon.get("long_horizon_shadow_passed") is not True:
        _add_reason(reasons, "stage6_not_passed")
    for metric in (
        "coverage_return_improvement",
        "cumulative_coverage_rate_delta_improvement",
        "valuable_area_covered_improvement",
    ):
        if _float(stage6_summary.get(metric)) <= 0.0:
            _add_reason(reasons, "coverage_metric_inconsistent")
    if metric_consistency.get("metric_consistency_audit_passed") is not True:
        _add_reason(reasons, "coverage_metric_inconsistent")
    if (
        stage6_summary.get("coverage_efficiency_regression") is True
        or cost_summary.get("coverage_efficiency_regression") is True
        or coverage_efficiency.get("coverage_efficiency_audit_passed") is False
        or coverage_efficiency.get("coverage_efficiency_regression") is True
    ):
        _add_reason(reasons, "coverage_efficiency_regression")
    if _float(stage6_summary.get("fallback_rate")) >= FALLBACK_RATE_THRESHOLD or _float(guard_fallback.get("fallback_rate")) >= FALLBACK_RATE_THRESHOLD:
        _add_reason(reasons, "fallback_dominates")
    if (
        _int(stage6_summary.get("controlled_regression_count")) > 0
        or _int(stage6_summary.get("fallback_gain_contamination_count")) > 0
        or _int(guard_fallback.get("controlled_regression_count")) > 0
        or _int(guard_fallback.get("fallback_gain_contamination_count")) > 0
        or kill_switch.get("kill_switch_audit_passed") is not True
        or rollback.get("rollback_audit_passed") is not True
    ):
        _add_reason(reasons, "guard_or_rollback_not_passed")
    if telemetry.get("telemetry_audit_passed") is not True or telemetry.get("records_coverage_gain") is False:
        _add_reason(reasons, "telemetry_missing_coverage_gain")
    if claim_scope.get("claim_scope_audit_passed") is not True:
        _add_reason(reasons, "claim_scope_overbroad")
    if release_boundary.get("release_boundary_audit_passed") is not True:
        _add_reason(reasons, "release_boundary_violation")
    if provenance.get("provenance_audit_passed") is not True:
        _add_reason(reasons, "provenance_or_docs_missing")
    if eligibility and eligibility.get("eligible_for_formal_performance_claim_review") is False:
        _add_reason(reasons, "stage6_not_passed")
    return reasons


def _metric_consistency_audit(*, stage6_summary: dict[str, Any]) -> dict[str, Any]:
    candidate = stage6_summary.get("candidate_metrics") or {}
    baseline = stage6_summary.get("baseline_metrics") or {}
    computed_candidate = stage6_summary.get("computed_shadow_candidate_metrics") or {}
    computed_teacher = stage6_summary.get("computed_shadow_teacher_metrics") or {}
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
        checks.append(_metric_check("summary_improvement", field, stage6_summary.get(field), expected))
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "metric_consistency",
        "metric_consistency_audit_passed": all(row["passed"] for row in checks),
        "checks": checks,
    }


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


def _claim_scope_audit(*, claim_statement: str) -> dict[str, Any]:
    allowed_section = claim_statement.split("## Prohibited claims", 1)[0].lower()
    prohibited_patterns = (
        "approved for real executor",
        "real executor deployment",
        "default policy replacement",
        "replace default policy",
        "publishes checkpoint",
        "checkpoint publication approved",
        "ackermann-feasible",
        "real world",
        "all scenarios",
        "unbounded generalization",
    )
    matches = [pattern for pattern in prohibited_patterns if pattern in allowed_section]
    has_required_boundary = "offline guarded shadow/canary" in claim_statement.lower()
    has_prohibited_section = "## Prohibited claims" in claim_statement
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "claim_scope",
        "claim_scope_audit_passed": not matches and has_required_boundary and has_prohibited_section,
        "overbroad_patterns": matches,
        "has_required_offline_boundary": has_required_boundary,
        "has_prohibited_claims_section": has_prohibited_section,
    }


def _release_boundary_audit(
    *,
    stage6_summary: dict[str, Any],
    rollback: dict[str, Any],
    kill_switch: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    violations: list[str] = []
    for source_name, payload in (
        ("stage6_summary", stage6_summary),
        ("rollback_audit", rollback),
        ("selected_candidate_summary", selected_candidate_summary),
    ):
        for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor"):
            if payload.get(field) is True:
                violations.append(f"{source_name}.{field}")
    if stage6_summary.get("shadow_policy_takes_control") is True:
        violations.append("stage6_summary.shadow_policy_takes_control")
    if _int(stage6_summary.get("experimental_control_activation_count")) > 0:
        violations.append("stage6_summary.experimental_control_activation_count")
    if kill_switch.get("real_executor_connected") is True:
        violations.append("kill_switch_audit.real_executor_connected")
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


def _provenance_audit(
    *,
    repo_root: Path,
    input_paths: dict[str, Path],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    missing = [name for name, path in input_paths.items() if not Path(path).is_file()]
    doc_paths = [
        repo_root / "README.md",
        repo_root / "docs" / "算法设计与系统架构报告.md",
        repo_root / "docs" / "superpowers" / "specs" / "2026-06-15-formal-performance-claim-release-decision.md",
    ]
    missing_docs = [str(path) for path in doc_paths if not path.is_file()]
    upstream_passed = all(
        payload.get("status") == "passed" and not _string_list(payload.get("reason_codes"))
        for payload in (formal_summary, post_training_replay_summary, selected_candidate_summary)
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "provenance",
        "provenance_audit_passed": not missing and not missing_docs and upstream_passed,
        "missing_input_names": missing,
        "missing_documentation_paths": missing_docs,
        "upstream_stability_passed": upstream_passed,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _evidence_ledger(
    *,
    stage6_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    coverage_efficiency: dict[str, Any],
    guard_fallback: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    eligibility: dict[str, Any],
    cost_summary: dict[str, Any],
    cost_metric_rows: list[dict[str, Any]],
    cost_replay: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        _source("stage6_summary", stage6_summary.get("status") == "passed"),
        _source("stage6_long_horizon", long_horizon.get("long_horizon_shadow_passed") is True),
        _source("stage6_coverage_efficiency", coverage_efficiency.get("coverage_efficiency_audit_passed") is True),
        _source("stage6_guard_fallback", guard_fallback.get("guard_fallback_audit_passed") is True),
        _source("stage6_kill_switch", kill_switch.get("kill_switch_audit_passed") is True),
        _source("stage6_rollback", rollback.get("rollback_audit_passed") is True),
        _source("stage6_telemetry", telemetry.get("telemetry_audit_passed") is True),
        _source("stage6_eligibility", eligibility.get("eligible_for_formal_performance_claim_review") is True),
        _source("cost_efficiency_summary", cost_summary.get("status") == "passed"),
        _source("cost_metric_table", bool(cost_metric_rows)),
        _source("cost_replay_audit", not _string_list(cost_replay.get("reason_codes"))),
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
    scoped_offline_performance_claim_approved: bool,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "decision_matrix",
        "status": status,
        "reason_codes": list(reason_codes),
        "decision_verdict": EXPECTED_VERDICT if scoped_offline_performance_claim_approved else _failed_verdict(reason_codes),
        "scoped_offline_performance_claim_approved": scoped_offline_performance_claim_approved,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "allowed_decisions": [
            "scoped_offline_guarded_shadow_canary_performance_claim"
        ]
        if scoped_offline_performance_claim_approved
        else [],
        "blocked_release_actions": [
            "checkpoint_publication",
            "default_policy_replacement",
            "real_executor_connection",
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
    stage6_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    input_paths: dict[str, Path],
    stage6_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    coverage_efficiency: dict[str, Any],
    guard_fallback: dict[str, Any],
    kill_switch: dict[str, Any],
    rollback: dict[str, Any],
    telemetry: dict[str, Any],
    eligibility: dict[str, Any],
    metric_consistency: dict[str, Any],
    claim_scope: dict[str, Any],
    release_boundary: dict[str, Any],
    provenance: dict[str, Any],
    decision_matrix: dict[str, Any],
    scoped_offline_performance_claim_approved: bool,
) -> dict[str, Any]:
    verdict = decision_matrix.get("decision_verdict")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": _next_required_change(reason_codes),
        "decision_verdict": verdict,
        "repo_root": str(repo_root),
        "stage6_root": str(stage6_root),
        "cost_efficiency_root": str(cost_efficiency_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "summary": str(paths["summary"]),
        "evidence_ledger": str(paths["evidence_ledger"]),
        "metric_consistency_audit": str(paths["metric_consistency_audit"]),
        "claim_scope_audit": str(paths["claim_scope_audit"]),
        "release_boundary_audit": str(paths["release_boundary_audit"]),
        "provenance_audit": str(paths["provenance_audit"]),
        "decision_matrix": str(paths["decision_matrix"]),
        "scoped_claim_statement": str(paths["scoped_claim_statement"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "scoped_offline_performance_claim_approved": scoped_offline_performance_claim_approved,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "stage6_long_horizon_shadow_passed": stage6_summary.get("long_horizon_shadow_passed") is True
        and long_horizon.get("long_horizon_shadow_passed") is True,
        "coverage_return_improvement": stage6_summary.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": stage6_summary.get(
            "cumulative_coverage_rate_delta_improvement",
            0.0,
        ),
        "valuable_area_covered_improvement": stage6_summary.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(stage6_summary.get("coverage_efficiency_regression"))
        or bool(coverage_efficiency.get("coverage_efficiency_regression")),
        "fallback_rate": stage6_summary.get("fallback_rate", guard_fallback.get("fallback_rate")),
        "controlled_regression_count": stage6_summary.get(
            "controlled_regression_count",
            guard_fallback.get("controlled_regression_count", 0),
        ),
        "fallback_gain_contamination_count": stage6_summary.get(
            "fallback_gain_contamination_count",
            guard_fallback.get("fallback_gain_contamination_count", 0),
        ),
        "kill_switch_audit_passed": kill_switch.get("kill_switch_audit_passed") is True
        and stage6_summary.get("kill_switch_audit_passed") is True,
        "rollback_audit_passed": rollback.get("rollback_audit_passed") is True
        and stage6_summary.get("rollback_audit_passed") is True,
        "telemetry_audit_passed": telemetry.get("telemetry_audit_passed") is True
        and stage6_summary.get("telemetry_audit_passed") is True,
        "eligible_for_formal_performance_claim_review": eligibility.get(
            "eligible_for_formal_performance_claim_review"
        )
        is True
        or stage6_summary.get("eligible_for_formal_performance_claim_review") is True,
        "metric_consistency_audit_passed": metric_consistency.get("metric_consistency_audit_passed") is True,
        "claim_scope_audit_passed": claim_scope.get("claim_scope_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary.get("release_boundary_audit_passed") is True,
        "provenance_audit_passed": provenance.get("provenance_audit_passed") is True,
        "claim_scope_overbroad_patterns": claim_scope.get("overbroad_patterns", []),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "performance_claimed": scoped_offline_performance_claim_approved,
        "performance_claim_scope": "scoped_offline_guarded_shadow_canary_only"
        if scoped_offline_performance_claim_approved
        else None,
        "formal_release_claimed": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _default_claim_statement(stage6_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Formal Performance Claim / Release Decision v1",
            "",
            "## Allowed claim",
            "",
            (
                "The current 5B.7 candidate is approved only for a scoped offline guarded "
                "shadow/canary performance claim on the audited decision set."
            ),
            "",
            f"- Coverage return improvement: `{stage6_summary.get('coverage_return_improvement')}`",
            f"- Cumulative coverage rate delta improvement: `{stage6_summary.get('cumulative_coverage_rate_delta_improvement')}`",
            f"- Valuable area covered improvement: `{stage6_summary.get('valuable_area_covered_improvement')}`",
            f"- Fallback rate: `{stage6_summary.get('fallback_rate')}`",
            "",
            "## Prohibited claims",
            "",
            "- Checkpoint publication is not approved.",
            "- Default policy replacement is not approved.",
            "- Real executor connection is not approved.",
            "- Real world performance is not claimed.",
            "- Ackermann-feasible trajectory is not claimed.",
            "- Unlimited scenario generalization is not claimed.",
            "- IRIS/GCS/path-planner diagnostics are not release proof.",
            "",
        ]
    )


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Formal Performance Claim / Release Decision v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- decision_verdict: `{summary['decision_verdict']}`",
            f"- scoped_offline_performance_claim_approved: `{summary['scoped_offline_performance_claim_approved']}`",
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
        "release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_FILE,
        "provenance_audit": output_root / PROVENANCE_AUDIT_FILE,
        "decision_matrix": output_root / DECISION_MATRIX_FILE,
        "scoped_claim_statement": output_root / SCOPED_CLAIM_STATEMENT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _source(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _failed_verdict(reason_codes: list[str]) -> str:
    if "claim_scope_overbroad" in reason_codes:
        return "blocked_by_overbroad_claim_scope"
    if "release_boundary_violation" in reason_codes:
        return "blocked_by_release_boundary_violation"
    if "stage6_not_passed" in reason_codes:
        return "blocked_by_stage6_preflight"
    return "blocked_by_formal_claim_review"


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    if "stage6_not_passed" in reason_codes:
        return "fix_shadow_canary_release_performance_validation_preflight"
    if "claim_scope_overbroad" in reason_codes:
        return "narrow_formal_performance_claim_scope"
    if "coverage_metric_inconsistent" in reason_codes:
        return "fix_coverage_metric_consistency"
    if "fallback_dominates" in reason_codes:
        return "reduce_fallback_rate_before_formal_claim"
    if "telemetry_missing_coverage_gain" in reason_codes:
        return "fix_coverage_gain_telemetry"
    if "release_boundary_violation" in reason_codes:
        return "restore_non_release_boundary"
    return "inspect_formal_performance_claim_release_decision_inputs"


def _summary_path(summary: dict[str, Any], key: str, base: Path, repo_root: Path, fallback: str) -> Path:
    return _resolve_optional_path(summary.get(key), base, repo_root) or base / fallback


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _add_reason(reasons, "provenance_or_docs_missing")
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid_json")
    return {}


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except FileNotFoundError:
        _add_reason(reasons, "provenance_or_docs_missing")
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid_jsonl")
    return rows


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
