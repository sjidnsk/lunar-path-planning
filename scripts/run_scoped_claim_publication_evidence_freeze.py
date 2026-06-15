from __future__ import annotations

import argparse
import json
import re
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


SUMMARY_SCHEMA_VERSION = "scoped-claim-publication-evidence-freeze-summary/v1"
AUDIT_SCHEMA_VERSION = "scoped-claim-publication-evidence-freeze-audit/v1"
LEDGER_SCHEMA_VERSION = "scoped-claim-publication-evidence-freeze-ledger/v1"
MANIFEST_SCHEMA_VERSION = "scoped-claim-publication-bundle-manifest/v1"

DEFAULT_STAGE7_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"
DEFAULT_STAGE6_ROOT = "outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1"
DEFAULT_COST_EFFICIENCY_ROOT = "outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1"

STAGE7_SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"
STAGE7_EVIDENCE_LEDGER_FILE = "formal-performance-evidence-ledger.json"
STAGE7_METRIC_CONSISTENCY_AUDIT_FILE = "formal-performance-metric-consistency-audit.json"
STAGE7_CLAIM_SCOPE_AUDIT_FILE = "formal-performance-claim-scope-audit.json"
STAGE7_RELEASE_BOUNDARY_AUDIT_FILE = "formal-performance-release-boundary-audit.json"
STAGE7_PROVENANCE_AUDIT_FILE = "formal-performance-provenance-audit.json"
STAGE7_DECISION_MATRIX_FILE = "formal-performance-decision-matrix.json"
STAGE7_SCOPED_CLAIM_STATEMENT_FILE = "formal-performance-scoped-claim-statement.md"

STAGE6_SUMMARY_FILE = "shadow-canary-release-performance-validation-preflight-summary.json"
COST_SUMMARY_FILE = "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
POST_TRAINING_REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_CANDIDATE_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "scoped-claim-publication-evidence-freeze-summary.json"
BUNDLE_MANIFEST_FILE = "scoped-claim-publication-bundle-manifest.json"
CLAIM_TEXT_FILE = "scoped-claim-publication-claim-text.md"
SCOPE_AUDIT_FILE = "scoped-claim-publication-scope-audit.json"
EVIDENCE_FREEZE_LEDGER_FILE = "scoped-claim-publication-evidence-freeze-ledger.json"
DOC_CONSISTENCY_AUDIT_FILE = "scoped-claim-publication-doc-consistency-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "scoped-claim-publication-release-boundary-audit.json"
REJECTION_REPORT_FILE = "scoped-claim-publication-rejection-report.json"
REPORT_FILE = "scoped-claim-publication-evidence-freeze-report.md"

EXPECTED_SCOPE = "scoped_offline_guarded_shadow_canary_only"
EXPECTED_STAGE7_VERDICT = "approved_for_scoped_offline_performance_claim"
EXPECTED_PUBLICATION_VERDICT = "approved_for_scoped_claim_publication"
NEXT_REQUIRED_CHANGE = "checkpoint_publication_authorization_preflight"
TOLERANCE = 1.0e-6

METRIC_FIELDS = (
    "coverage_return_improvement",
    "cumulative_coverage_rate_delta_improvement",
    "valuable_area_covered_improvement",
    "fallback_rate",
)
RELEASE_BOUNDARY_FIELDS = (
    "checkpoint_publication_approved",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)
PROHIBITED_ALLOWED_SECTION_PATTERNS = (
    "real world performance",
    "real-world performance",
    "ackermann-feasible",
    "default policy replacement",
    "replace default policy",
    "checkpoint publication approved",
    "published checkpoint",
    "real executor connection",
    "real executor deployment",
    "unlimited scenario generalization",
    "unbounded generalization",
    "all scenarios",
    "stage 9 authorization",
    "stage 10 authorization",
    "stage 12 authorization",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and publish the scoped offline performance claim evidence bundle."
    )
    parser.add_argument("--stage7-root", default=DEFAULT_STAGE7_ROOT)
    parser.add_argument("--stage6-root", default=DEFAULT_STAGE6_ROOT)
    parser.add_argument("--cost-efficiency-root", default=DEFAULT_COST_EFFICIENCY_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_scoped_claim_publication_evidence_freeze(
        stage7_root=_resolve_path(Path(args.stage7_root), repo_root),
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
                "publication_verdict": summary["publication_verdict"],
                "scoped_claim_publication_approved": summary["scoped_claim_publication_approved"],
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


def run_scoped_claim_publication_evidence_freeze(
    *,
    stage7_root: Path,
    stage6_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    claim_text_override: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    read_reasons: list[str] = []
    stage7_summary_path = Path(stage7_root) / STAGE7_SUMMARY_FILE
    stage7_summary = _read_json(stage7_summary_path, read_reasons, "stage7_summary")
    stage7_artifact_paths = {
        "stage7_evidence_ledger": _summary_path(
            stage7_summary,
            "evidence_ledger",
            Path(stage7_root),
            repo_root,
            STAGE7_EVIDENCE_LEDGER_FILE,
        ),
        "stage7_metric_consistency_audit": _summary_path(
            stage7_summary,
            "metric_consistency_audit",
            Path(stage7_root),
            repo_root,
            STAGE7_METRIC_CONSISTENCY_AUDIT_FILE,
        ),
        "stage7_claim_scope_audit": _summary_path(
            stage7_summary,
            "claim_scope_audit",
            Path(stage7_root),
            repo_root,
            STAGE7_CLAIM_SCOPE_AUDIT_FILE,
        ),
        "stage7_release_boundary_audit": _summary_path(
            stage7_summary,
            "release_boundary_audit",
            Path(stage7_root),
            repo_root,
            STAGE7_RELEASE_BOUNDARY_AUDIT_FILE,
        ),
        "stage7_provenance_audit": _summary_path(
            stage7_summary,
            "provenance_audit",
            Path(stage7_root),
            repo_root,
            STAGE7_PROVENANCE_AUDIT_FILE,
        ),
        "stage7_decision_matrix": _summary_path(
            stage7_summary,
            "decision_matrix",
            Path(stage7_root),
            repo_root,
            STAGE7_DECISION_MATRIX_FILE,
        ),
        "stage7_scoped_claim_statement": _summary_path(
            stage7_summary,
            "scoped_claim_statement",
            Path(stage7_root),
            repo_root,
            STAGE7_SCOPED_CLAIM_STATEMENT_FILE,
        ),
    }
    stage7_evidence_ledger = _read_json(
        stage7_artifact_paths["stage7_evidence_ledger"],
        read_reasons,
        "stage7_evidence_ledger",
    )
    stage7_metric_consistency = _read_json(
        stage7_artifact_paths["stage7_metric_consistency_audit"],
        read_reasons,
        "stage7_metric_consistency_audit",
    )
    stage7_claim_scope = _read_json(
        stage7_artifact_paths["stage7_claim_scope_audit"],
        read_reasons,
        "stage7_claim_scope_audit",
    )
    stage7_release_boundary = _read_json(
        stage7_artifact_paths["stage7_release_boundary_audit"],
        read_reasons,
        "stage7_release_boundary_audit",
    )
    stage7_provenance = _read_json(
        stage7_artifact_paths["stage7_provenance_audit"],
        read_reasons,
        "stage7_provenance_audit",
    )
    stage7_decision_matrix = _read_json(
        stage7_artifact_paths["stage7_decision_matrix"],
        read_reasons,
        "stage7_decision_matrix",
    )
    stage7_claim_statement = _read_text(
        stage7_artifact_paths["stage7_scoped_claim_statement"],
        read_reasons,
        "stage7_scoped_claim_statement",
    )

    stage6_summary_path = Path(stage6_root) / STAGE6_SUMMARY_FILE
    cost_summary_path = Path(cost_efficiency_root) / COST_SUMMARY_FILE
    formal_summary_path = Path(formal_training_root) / FORMAL_SUMMARY_FILE
    post_training_replay_summary_path = Path(post_training_replay_root) / POST_TRAINING_REPLAY_SUMMARY_FILE
    selected_candidate_summary_path = Path(selected_candidate_root) / SELECTED_CANDIDATE_SUMMARY_FILE
    stage6_summary = _read_json(stage6_summary_path, read_reasons, "stage6_summary")
    cost_summary = _read_json(cost_summary_path, read_reasons, "cost_efficiency_summary")
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
        "stage7_summary": stage7_summary_path,
        **stage7_artifact_paths,
        "stage6_summary": stage6_summary_path,
        "cost_efficiency_summary": cost_summary_path,
        "formal_training_summary": formal_summary_path,
        "post_training_replay_summary": post_training_replay_summary_path,
        "selected_candidate_summary": selected_candidate_summary_path,
    }

    claim_text = claim_text_override or _default_claim_text(stage7_summary)
    paths["claim_text"].write_text(claim_text, encoding="utf-8")

    scope_audit = _scope_audit(
        stage7_summary=stage7_summary,
        stage7_claim_statement=stage7_claim_statement,
        claim_text=claim_text,
    )
    evidence_ledger = _evidence_freeze_ledger(
        input_paths=input_paths,
        stage7_summary=stage7_summary,
        stage7_evidence_ledger=stage7_evidence_ledger,
        stage7_metric_consistency=stage7_metric_consistency,
        stage7_claim_scope=stage7_claim_scope,
        stage7_release_boundary=stage7_release_boundary,
        stage7_provenance=stage7_provenance,
        stage7_decision_matrix=stage7_decision_matrix,
        stage6_summary=stage6_summary,
        cost_summary=cost_summary,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
        read_reasons=read_reasons,
    )
    doc_consistency = _doc_consistency_audit(repo_root=repo_root)
    release_boundary = _release_boundary_audit(
        stage7_summary=stage7_summary,
        stage7_release_boundary=stage7_release_boundary,
        stage7_decision_matrix=stage7_decision_matrix,
        stage6_summary=stage6_summary,
        cost_summary=cost_summary,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
    )
    metric_mismatch = _metric_mismatches(
        stage7_summary=stage7_summary,
        stage6_summary=stage6_summary,
        cost_summary=cost_summary,
        claim_text=claim_text,
    )

    reason_codes = _unique(
        [
            *_stage7_reason_codes(stage7_summary, stage7_decision_matrix),
            *([] if scope_audit["scope_audit_passed"] else ["claim_scope_overbroad"]),
            *([] if not scope_audit["prohibited_claim_patterns"] else ["prohibited_claim_present"]),
            *([] if not metric_mismatch else ["claim_metric_mismatch"]),
            *([] if doc_consistency["doc_consistency_audit_passed"] else ["docs_not_updated"]),
            *([] if release_boundary["release_boundary_audit_passed"] else ["release_boundary_violation"]),
            *([] if evidence_ledger["evidence_freeze_complete"] else ["evidence_freeze_incomplete"]),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    approved = status == "passed"

    bundle_manifest = _bundle_manifest(
        approved=approved,
        reason_codes=reason_codes,
        input_paths=input_paths,
        output_paths=paths,
        stage7_summary=stage7_summary,
        metric_mismatch=metric_mismatch,
    )
    rejection_report = _rejection_report(reason_codes)

    _write_json(paths["bundle_manifest"], bundle_manifest)
    _write_json(paths["scope_audit"], scope_audit)
    _write_json(paths["evidence_freeze_ledger"], evidence_ledger)
    _write_json(paths["doc_consistency_audit"], doc_consistency)
    _write_json(paths["release_boundary_audit"], release_boundary)
    _write_json(paths["rejection_report"], rejection_report)

    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        approved=approved,
        paths=paths,
        repo_root=repo_root,
        stage7_root=Path(stage7_root),
        stage6_root=Path(stage6_root),
        cost_efficiency_root=Path(cost_efficiency_root),
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        stage7_summary=stage7_summary,
        scope_audit=scope_audit,
        evidence_ledger=evidence_ledger,
        doc_consistency=doc_consistency,
        release_boundary=release_boundary,
        metric_mismatch=metric_mismatch,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage7_reason_codes(stage7_summary: dict[str, Any], decision_matrix: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if (
        stage7_summary.get("status") != "passed"
        or _string_list(stage7_summary.get("reason_codes"))
        or stage7_summary.get("decision_verdict") != EXPECTED_STAGE7_VERDICT
        or stage7_summary.get("scoped_offline_performance_claim_approved") is not True
        or decision_matrix.get("decision_verdict") != EXPECTED_STAGE7_VERDICT
        or decision_matrix.get("scoped_offline_performance_claim_approved") is not True
    ):
        reasons.append("stage7_not_passed")
    if stage7_summary.get("performance_claim_scope") != EXPECTED_SCOPE:
        reasons.append("claim_scope_overbroad")
    for field in (
        "metric_consistency_audit_passed",
        "claim_scope_audit_passed",
        "release_boundary_audit_passed",
        "provenance_audit_passed",
        "stage6_long_horizon_shadow_passed",
        "kill_switch_audit_passed",
        "rollback_audit_passed",
        "telemetry_audit_passed",
    ):
        if stage7_summary.get(field) is not True:
            reasons.append("stage7_not_passed")
    for metric in (
        "coverage_return_improvement",
        "cumulative_coverage_rate_delta_improvement",
        "valuable_area_covered_improvement",
    ):
        if _float(stage7_summary.get(metric)) <= 0.0:
            reasons.append("claim_metric_mismatch")
    return reasons


def _scope_audit(
    *,
    stage7_summary: dict[str, Any],
    stage7_claim_statement: str,
    claim_text: str,
) -> dict[str, Any]:
    allowed_section = _allowed_section(claim_text).lower()
    prohibited_matches = [
        pattern for pattern in PROHIBITED_ALLOWED_SECTION_PATTERNS if pattern in allowed_section
    ]
    required_phrases = [
        "current offline guarded shadow/canary",
        "same-decision-set baseline",
        "coverage return improvement",
        "cumulative coverage rate delta improvement",
        "valuable area covered improvement",
    ]
    missing_required = [phrase for phrase in required_phrases if phrase not in claim_text.lower()]
    stage7_statement_ok = "offline guarded shadow/canary" in stage7_claim_statement.lower()
    scope_ok = stage7_summary.get("performance_claim_scope") == EXPECTED_SCOPE
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "scoped_claim_publication_scope",
        "scope_audit_passed": scope_ok and stage7_statement_ok and not missing_required,
        "prohibited_claim_patterns": prohibited_matches,
        "missing_required_phrases": missing_required,
        "stage7_claim_statement_has_offline_boundary": stage7_statement_ok,
        "performance_claim_scope": stage7_summary.get("performance_claim_scope"),
    }


def _evidence_freeze_ledger(
    *,
    input_paths: dict[str, Path],
    stage7_summary: dict[str, Any],
    stage7_evidence_ledger: dict[str, Any],
    stage7_metric_consistency: dict[str, Any],
    stage7_claim_scope: dict[str, Any],
    stage7_release_boundary: dict[str, Any],
    stage7_provenance: dict[str, Any],
    stage7_decision_matrix: dict[str, Any],
    stage6_summary: dict[str, Any],
    cost_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
    read_reasons: list[str],
) -> dict[str, Any]:
    sources = [
        _source("stage7_summary", stage7_summary.get("status") == "passed"),
        _source("stage7_evidence_ledger", stage7_evidence_ledger.get("all_sources_passed") is True),
        _source(
            "stage7_metric_consistency_audit",
            stage7_metric_consistency.get("metric_consistency_audit_passed") is True,
        ),
        _source("stage7_claim_scope_audit", stage7_claim_scope.get("claim_scope_audit_passed") is True),
        _source(
            "stage7_release_boundary_audit",
            stage7_release_boundary.get("release_boundary_audit_passed") is True,
        ),
        _source("stage7_provenance_audit", stage7_provenance.get("provenance_audit_passed") is True),
        _source(
            "stage7_decision_matrix",
            stage7_decision_matrix.get("decision_verdict") == EXPECTED_STAGE7_VERDICT,
        ),
        _source("stage6_summary", stage6_summary.get("status") == "passed"),
        _source("cost_efficiency_summary", cost_summary.get("status") == "passed"),
        _source("formal_training_summary", formal_summary.get("status") == "passed"),
        _source("post_training_replay_summary", post_training_replay_summary.get("status") == "passed"),
        _source("selected_candidate_summary", selected_candidate_summary.get("status") == "passed"),
    ]
    missing_paths = [name for name, path in input_paths.items() if not Path(path).is_file()]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "sources": sources,
        "missing_input_names": sorted(set(missing_paths + list(read_reasons))),
        "all_sources_passed": all(source["passed"] for source in sources),
        "evidence_freeze_complete": not missing_paths and not read_reasons and all(source["passed"] for source in sources),
    }


def _doc_consistency_audit(*, repo_root: Path) -> dict[str, Any]:
    required_docs = {
        "README.md": repo_root / "README.md",
        "docs/算法设计与系统架构报告.md": repo_root / "docs" / "算法设计与系统架构报告.md",
        "docs/superpowers/specs/2026-06-15-scoped-claim-publication-evidence-freeze.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-15-scoped-claim-publication-evidence-freeze.md",
    }
    required_phrases = {
        "README.md": [
            "Stage 8 `Scoped Claim Publication & Evidence Freeze v1`",
            "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1/",
            "checkpoint_publication_authorization_preflight",
        ],
        "docs/算法设计与系统架构报告.md": [
            "阶段 8 `Scoped Claim Publication & Evidence Freeze v1`",
            "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1/",
            "checkpoint_publication_authorization_preflight",
        ],
        "docs/superpowers/specs/2026-06-15-scoped-claim-publication-evidence-freeze.md": [
            "scoped-claim-publication-evidence-freeze-summary.json",
            "README.md",
            "docs/算法设计与系统架构报告.md",
            "不发布 checkpoint",
        ],
    }
    missing_docs: list[str] = []
    missing_phrases: dict[str, list[str]] = {}
    for label, path in required_docs.items():
        if not path.is_file():
            missing_docs.append(label)
            continue
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in required_phrases[label] if phrase not in text]
        if missing:
            missing_phrases[label] = missing
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "documentation_consistency",
        "doc_consistency_audit_passed": not missing_docs and not missing_phrases,
        "missing_docs": missing_docs,
        "missing_phrases": missing_phrases,
    }


def _release_boundary_audit(
    *,
    stage7_summary: dict[str, Any],
    stage7_release_boundary: dict[str, Any],
    stage7_decision_matrix: dict[str, Any],
    stage6_summary: dict[str, Any],
    cost_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    violations: list[str] = []
    for name, payload in (
        ("stage7_summary", stage7_summary),
        ("stage7_release_boundary_audit", stage7_release_boundary),
        ("stage7_decision_matrix", stage7_decision_matrix),
        ("stage6_summary", stage6_summary),
        ("cost_efficiency_summary", cost_summary),
        ("formal_training_summary", formal_summary),
        ("post_training_replay_summary", post_training_replay_summary),
        ("selected_candidate_summary", selected_candidate_summary),
    ):
        for field in RELEASE_BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append(f"{name}.{field}")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "scoped_claim_publication_release_boundary",
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _metric_mismatches(
    *,
    stage7_summary: dict[str, Any],
    stage6_summary: dict[str, Any],
    cost_summary: dict[str, Any],
    claim_text: str,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for field in METRIC_FIELDS:
        stage7_value = _float_or_none(stage7_summary.get(field))
        for source_name, payload in (("stage6_summary", stage6_summary), ("cost_efficiency_summary", cost_summary)):
            source_value = _float_or_none(payload.get(field))
            if stage7_value is None or source_value is None or abs(stage7_value - source_value) > TOLERANCE:
                mismatches.append(
                    {
                        "field": field,
                        "source": source_name,
                        "stage7_value": stage7_summary.get(field),
                        "source_value": payload.get(field),
                    }
                )
        if stage7_value is not None and _format_metric(stage7_value) not in claim_text:
            mismatches.append(
                {
                    "field": field,
                    "source": "claim_text",
                    "stage7_value": stage7_summary.get(field),
                    "source_value": None,
                }
            )
    if stage7_summary.get("coverage_efficiency_regression") is not False:
        mismatches.append(
            {
                "field": "coverage_efficiency_regression",
                "source": "stage7_summary",
                "stage7_value": stage7_summary.get("coverage_efficiency_regression"),
                "source_value": False,
            }
        )
    return mismatches


def _bundle_manifest(
    *,
    approved: bool,
    reason_codes: list[str],
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    stage7_summary: dict[str, Any],
    metric_mismatch: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "publication_verdict": EXPECTED_PUBLICATION_VERDICT if approved else _failed_verdict(reason_codes),
        "scoped_claim_publication_approved": approved,
        "performance_claim_scope": stage7_summary.get("performance_claim_scope"),
        "frozen_input_artifacts": {name: str(path) for name, path in input_paths.items()},
        "publication_artifacts": {name: str(path) for name, path in output_paths.items()},
        "metric_mismatches": metric_mismatch,
        "blocked_release_actions": [
            "checkpoint_publication",
            "default_policy_replacement",
            "real_executor_connection",
        ],
    }


def _rejection_report(reason_codes: list[str]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": list(reason_codes),
        "rejected": bool(reason_codes),
    }


def _summary(
    *,
    status: str,
    reason_codes: list[str],
    approved: bool,
    paths: dict[str, Path],
    repo_root: Path,
    stage7_root: Path,
    stage6_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    stage7_summary: dict[str, Any],
    scope_audit: dict[str, Any],
    evidence_ledger: dict[str, Any],
    doc_consistency: dict[str, Any],
    release_boundary: dict[str, Any],
    metric_mismatch: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "publication_verdict": EXPECTED_PUBLICATION_VERDICT if approved else _failed_verdict(reason_codes),
        "scoped_claim_publication_approved": approved,
        "performance_claim_scope": stage7_summary.get("performance_claim_scope"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_scoped_claim_publication_rejections",
        "coverage_return_improvement": stage7_summary.get("coverage_return_improvement"),
        "cumulative_coverage_rate_delta_improvement": stage7_summary.get(
            "cumulative_coverage_rate_delta_improvement"
        ),
        "valuable_area_covered_improvement": stage7_summary.get("valuable_area_covered_improvement"),
        "fallback_rate": stage7_summary.get("fallback_rate"),
        "coverage_efficiency_regression": stage7_summary.get("coverage_efficiency_regression"),
        "scope_audit_passed": scope_audit.get("scope_audit_passed") is True,
        "evidence_freeze_complete": evidence_ledger.get("evidence_freeze_complete") is True,
        "doc_consistency_audit_passed": doc_consistency.get("doc_consistency_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary.get("release_boundary_audit_passed") is True,
        "metric_mismatch_count": len(metric_mismatch),
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "runs_new_ppo_update": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "repo_root": str(repo_root),
        "stage7_root": str(stage7_root),
        "stage6_root": str(stage6_root),
        "cost_efficiency_root": str(cost_efficiency_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "bundle_manifest": str(paths["bundle_manifest"]),
        "claim_text": str(paths["claim_text"]),
        "scope_audit": str(paths["scope_audit"]),
        "evidence_freeze_ledger": str(paths["evidence_freeze_ledger"]),
        "doc_consistency_audit": str(paths["doc_consistency_audit"]),
        "release_boundary_audit": str(paths["release_boundary_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    return summary


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Scoped Claim Publication & Evidence Freeze v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Publication verdict: `{summary['publication_verdict']}`\n"
        f"- Claim scope: `{summary.get('performance_claim_scope')}`\n"
        f"- Next required change: `{summary.get('next_required_change')}`\n\n"
        "## Release Boundaries\n\n"
        f"- Checkpoint publication approved: `{summary['checkpoint_publication_approved']}`\n"
        f"- Default policy replacement approved: `{summary['default_policy_replacement_approved']}`\n"
        f"- Real executor connection approved: `{summary['real_executor_connection_approved']}`\n"
    )


def _default_claim_text(stage7_summary: dict[str, Any]) -> str:
    return (
        "# Scoped Claim Publication & Evidence Freeze v1\n\n"
        "## Allowed scoped offline claim\n\n"
        "The current 5B.7 candidate may be described only as having positive evidence "
        "within the current offline guarded shadow/canary evaluation and the same-decision-set baseline. "
        "This publication freezes the Stage 7 scoped claim text and evidence references; it does not add "
        "model execution authority.\n\n"
        f"- Coverage return improvement: `{_format_metric(stage7_summary.get('coverage_return_improvement'))}`\n"
        "- Cumulative coverage rate delta improvement: "
        f"`{_format_metric(stage7_summary.get('cumulative_coverage_rate_delta_improvement'))}`\n"
        f"- Valuable area covered improvement: `{_format_metric(stage7_summary.get('valuable_area_covered_improvement'))}`\n"
        f"- Fallback rate: `{_format_metric(stage7_summary.get('fallback_rate'))}`\n"
        f"- Coverage efficiency regression: `{stage7_summary.get('coverage_efficiency_regression')}`\n\n"
        "## Prohibited claims and actions\n\n"
        "- checkpoint publication is not approved.\n"
        "- default policy replacement is not approved.\n"
        "- real executor connection is not approved.\n"
        "- Real world performance is not claimed.\n"
        "- Ackermann-feasible trajectory is not claimed.\n"
        "- Unlimited scenario generalization is not claimed.\n"
        "- IRIS/GCS/path-planner diagnostics are not release proof.\n"
        "- Stage 8 is not Stage 9, Stage 10, or Stage 12 authorization.\n"
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "bundle_manifest": output_root / BUNDLE_MANIFEST_FILE,
        "claim_text": output_root / CLAIM_TEXT_FILE,
        "scope_audit": output_root / SCOPE_AUDIT_FILE,
        "evidence_freeze_ledger": output_root / EVIDENCE_FREEZE_LEDGER_FILE,
        "doc_consistency_audit": output_root / DOC_CONSISTENCY_AUDIT_FILE,
        "release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        _add_reason(reasons, label)
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reasons, label)
        return {}


def _read_text(path: Path, reasons: list[str], label: str) -> str:
    if not Path(path).is_file():
        _add_reason(reasons, label)
        return ""
    return Path(path).read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_path(
    summary: dict[str, Any],
    field: str,
    root: Path,
    repo_root: Path,
    default_name: str,
) -> Path:
    raw = summary.get(field)
    if isinstance(raw, str) and raw:
        return _resolve_path(Path(raw), repo_root)
    return Path(root) / default_name


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _source(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _allowed_section(text: str) -> str:
    return re.split(r"^##\s+Prohibited", text, flags=re.IGNORECASE | re.MULTILINE)[0]


def _format_metric(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    return format(number, ".15g")


def _float(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _failed_verdict(reason_codes: list[str]) -> str:
    return reason_codes[0] if reason_codes else "failed"


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
