from __future__ import annotations

import argparse
import hashlib
import json
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


SUMMARY_SCHEMA_VERSION = "checkpoint-publication-authorization-preflight-summary/v1"
AUDIT_SCHEMA_VERSION = "checkpoint-publication-authorization-preflight-audit/v1"
MANIFEST_SCHEMA_VERSION = "checkpoint-publication-candidate-manifest/v1"

DEFAULT_STAGE8_ROOT = "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1"
DEFAULT_STAGE7_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"
DEFAULT_STAGE6_ROOT = "outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1"
DEFAULT_COST_EFFICIENCY_ROOT = "outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1"

STAGE8_SUMMARY_FILE = "scoped-claim-publication-evidence-freeze-summary.json"
STAGE8_BUNDLE_MANIFEST_FILE = "scoped-claim-publication-bundle-manifest.json"
STAGE8_CLAIM_TEXT_FILE = "scoped-claim-publication-claim-text.md"
STAGE8_SCOPE_AUDIT_FILE = "scoped-claim-publication-scope-audit.json"
STAGE8_EVIDENCE_FREEZE_LEDGER_FILE = "scoped-claim-publication-evidence-freeze-ledger.json"
STAGE8_DOC_CONSISTENCY_AUDIT_FILE = "scoped-claim-publication-doc-consistency-audit.json"
STAGE8_RELEASE_BOUNDARY_AUDIT_FILE = "scoped-claim-publication-release-boundary-audit.json"

STAGE7_SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"
STAGE6_SUMMARY_FILE = "shadow-canary-release-performance-validation-preflight-summary.json"
COST_SUMMARY_FILE = "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
POST_TRAINING_REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_CANDIDATE_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "checkpoint-publication-authorization-preflight-summary.json"
CANDIDATE_MANIFEST_FILE = "checkpoint-publication-candidate-manifest.json"
IDENTITY_AUDIT_FILE = "checkpoint-publication-identity-audit.json"
METADATA_AUDIT_FILE = "checkpoint-publication-metadata-audit.json"
LOAD_EVIDENCE_AUDIT_FILE = "checkpoint-publication-load-evidence-audit.json"
LINEAGE_AUDIT_FILE = "checkpoint-publication-lineage-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "checkpoint-publication-release-boundary-audit.json"
AUTHORIZATION_MATRIX_FILE = "checkpoint-publication-authorization-matrix.json"
REJECTION_REPORT_FILE = "checkpoint-publication-rejection-report.json"
REPORT_FILE = "checkpoint-publication-authorization-preflight-report.md"

EXPECTED_STAGE8_VERDICT = "approved_for_scoped_claim_publication"
EXPECTED_STAGE8_NEXT = "checkpoint_publication_authorization_preflight"
EXPECTED_SCOPE = "scoped_offline_guarded_shadow_canary_only"
EXPECTED_AUTHORIZATION_VERDICT = "eligible_for_checkpoint_publication_package_preparation"
NEXT_REQUIRED_CHANGE = "checkpoint_publication_package_preparation"

RELEASE_BOUNDARY_FIELDS = (
    "checkpoint_publication_approved",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "real_world_release_approved",
    "relaxes_guard",
    "modifies_network_or_action_space",
    "modifies_default_astar",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run checkpoint publication authorization preflight."
    )
    parser.add_argument("--stage8-root", default=DEFAULT_STAGE8_ROOT)
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
    summary = run_checkpoint_publication_authorization_preflight(
        stage8_root=_resolve_path(Path(args.stage8_root), repo_root),
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
                "authorization_verdict": summary["authorization_verdict"],
                "checkpoint_publication_authorization_preflight_passed": summary[
                    "checkpoint_publication_authorization_preflight_passed"
                ],
                "checkpoint_publication_package_preparation_approved": summary[
                    "checkpoint_publication_package_preparation_approved"
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


def run_checkpoint_publication_authorization_preflight(
    *,
    stage8_root: Path,
    stage7_root: Path,
    stage6_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    read_reasons: list[str] = []
    stage8_summary_path = Path(stage8_root) / STAGE8_SUMMARY_FILE
    stage8_summary = _read_json(stage8_summary_path, read_reasons, "stage8_summary")
    stage8_artifact_paths = {
        "stage8_bundle_manifest": _summary_path(
            stage8_summary, "bundle_manifest", Path(stage8_root), repo_root, STAGE8_BUNDLE_MANIFEST_FILE
        ),
        "stage8_claim_text": _summary_path(
            stage8_summary, "claim_text", Path(stage8_root), repo_root, STAGE8_CLAIM_TEXT_FILE
        ),
        "stage8_scope_audit": _summary_path(
            stage8_summary, "scope_audit", Path(stage8_root), repo_root, STAGE8_SCOPE_AUDIT_FILE
        ),
        "stage8_evidence_freeze_ledger": _summary_path(
            stage8_summary,
            "evidence_freeze_ledger",
            Path(stage8_root),
            repo_root,
            STAGE8_EVIDENCE_FREEZE_LEDGER_FILE,
        ),
        "stage8_doc_consistency_audit": _summary_path(
            stage8_summary,
            "doc_consistency_audit",
            Path(stage8_root),
            repo_root,
            STAGE8_DOC_CONSISTENCY_AUDIT_FILE,
        ),
        "stage8_release_boundary_audit": _summary_path(
            stage8_summary,
            "release_boundary_audit",
            Path(stage8_root),
            repo_root,
            STAGE8_RELEASE_BOUNDARY_AUDIT_FILE,
        ),
    }
    stage8_bundle_manifest = _read_json(
        stage8_artifact_paths["stage8_bundle_manifest"], read_reasons, "stage8_bundle_manifest"
    )
    stage8_claim_text = _read_text(
        stage8_artifact_paths["stage8_claim_text"], read_reasons, "stage8_claim_text"
    )
    stage8_scope_audit = _read_json(
        stage8_artifact_paths["stage8_scope_audit"], read_reasons, "stage8_scope_audit"
    )
    stage8_evidence_freeze_ledger = _read_json(
        stage8_artifact_paths["stage8_evidence_freeze_ledger"],
        read_reasons,
        "stage8_evidence_freeze_ledger",
    )
    stage8_doc_consistency = _read_json(
        stage8_artifact_paths["stage8_doc_consistency_audit"],
        read_reasons,
        "stage8_doc_consistency_audit",
    )
    stage8_release_boundary = _read_json(
        stage8_artifact_paths["stage8_release_boundary_audit"],
        read_reasons,
        "stage8_release_boundary_audit",
    )

    stage7_summary_path = Path(stage7_root) / STAGE7_SUMMARY_FILE
    stage6_summary_path = Path(stage6_root) / STAGE6_SUMMARY_FILE
    cost_summary_path = Path(cost_efficiency_root) / COST_SUMMARY_FILE
    formal_summary_path = Path(formal_training_root) / FORMAL_SUMMARY_FILE
    replay_summary_path = Path(post_training_replay_root) / POST_TRAINING_REPLAY_SUMMARY_FILE
    selected_summary_path = Path(selected_candidate_root) / SELECTED_CANDIDATE_SUMMARY_FILE
    stage7_summary = _read_json(stage7_summary_path, read_reasons, "stage7_summary")
    stage6_summary = _read_json(stage6_summary_path, read_reasons, "stage6_summary")
    cost_summary = _read_json(cost_summary_path, read_reasons, "cost_efficiency_summary")
    formal_summary = _read_json(formal_summary_path, read_reasons, "formal_training_summary")
    replay_summary = _read_json(replay_summary_path, read_reasons, "post_training_replay_summary")
    selected_summary = _read_json(selected_summary_path, read_reasons, "selected_candidate_summary")

    checkpoint_path = _resolve_optional_path(selected_summary.get("checkpoint_path"), Path(selected_candidate_root), repo_root)
    metadata_path = _resolve_optional_path(
        selected_summary.get("checkpoint_metadata_path"), Path(selected_candidate_root), repo_root
    )
    metadata = _read_json(metadata_path, read_reasons, "checkpoint_metadata") if metadata_path else {}
    inference_audit_path = _resolve_optional_path(
        selected_summary.get("inference_audit"), Path(selected_candidate_root), repo_root
    )
    inference_audit = _read_json(inference_audit_path, read_reasons, "checkpoint_load_inference_audit") if inference_audit_path else {}

    stage8_gate = _stage8_gate_audit(
        stage8_summary=stage8_summary,
        stage8_bundle_manifest=stage8_bundle_manifest,
        stage8_scope_audit=stage8_scope_audit,
        stage8_evidence_freeze_ledger=stage8_evidence_freeze_ledger,
        stage8_doc_consistency=stage8_doc_consistency,
        stage8_release_boundary=stage8_release_boundary,
        stage8_claim_text=stage8_claim_text,
    )
    identity_audit = _identity_audit(checkpoint_path=checkpoint_path, selected_summary=selected_summary)
    metadata_audit = _metadata_audit(
        metadata_path=metadata_path,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        selected_summary=selected_summary,
        repo_root=repo_root,
    )
    load_evidence_audit = _load_evidence_audit(
        selected_summary=selected_summary,
        inference_audit=inference_audit,
        inference_audit_path=inference_audit_path,
    )
    lineage_audit = _lineage_audit(
        read_reasons=read_reasons,
        stage8_summary=stage8_summary,
        stage7_summary=stage7_summary,
        stage6_summary=stage6_summary,
        cost_summary=cost_summary,
        formal_summary=formal_summary,
        replay_summary=replay_summary,
        selected_summary=selected_summary,
    )
    release_boundary_audit = _release_boundary_audit(
        payloads={
            "stage8_summary": stage8_summary,
            "stage8_release_boundary_audit": stage8_release_boundary,
            "stage7_summary": stage7_summary,
            "stage6_summary": stage6_summary,
            "cost_efficiency_summary": cost_summary,
            "formal_training_summary": formal_summary,
            "post_training_replay_summary": replay_summary,
            "selected_candidate_summary": selected_summary,
            "checkpoint_metadata": metadata,
        }
    )
    docs_audit = _docs_audit(repo_root=repo_root)

    reason_codes = _unique(
        [
            *([] if stage8_gate["stage8_gate_audit_passed"] else stage8_gate["reason_codes"]),
            *identity_audit["reason_codes"],
            *metadata_audit["reason_codes"],
            *load_evidence_audit["reason_codes"],
            *([] if lineage_audit["lineage_audit_passed"] else ["lineage_incomplete"]),
            *([] if release_boundary_audit["release_boundary_audit_passed"] else ["release_boundary_violation"]),
            *([] if docs_audit["docs_audit_passed"] else ["docs_not_updated"]),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    approved = status == "passed"
    authorization_matrix = _authorization_matrix(
        approved=approved,
        reason_codes=reason_codes,
        stage8_gate=stage8_gate,
        identity_audit=identity_audit,
        metadata_audit=metadata_audit,
        load_evidence_audit=load_evidence_audit,
        lineage_audit=lineage_audit,
        release_boundary_audit=release_boundary_audit,
        docs_audit=docs_audit,
    )
    candidate_manifest = _candidate_manifest(
        approved=approved,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        selected_summary=selected_summary,
        identity_audit=identity_audit,
        stage8_summary=stage8_summary,
    )
    rejection_report = _rejection_report(reason_codes)

    _write_json(paths["candidate_manifest"], candidate_manifest)
    _write_json(paths["identity_audit"], identity_audit)
    _write_json(paths["metadata_audit"], metadata_audit)
    _write_json(paths["load_evidence_audit"], load_evidence_audit)
    _write_json(paths["lineage_audit"], lineage_audit)
    _write_json(paths["release_boundary_audit"], release_boundary_audit)
    _write_json(paths["authorization_matrix"], authorization_matrix)
    _write_json(paths["rejection_report"], rejection_report)

    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        approved=approved,
        paths=paths,
        repo_root=repo_root,
        stage8_root=Path(stage8_root),
        stage7_root=Path(stage7_root),
        stage6_root=Path(stage6_root),
        cost_efficiency_root=Path(cost_efficiency_root),
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        selected_summary=selected_summary,
        identity_audit=identity_audit,
        metadata_audit=metadata_audit,
        load_evidence_audit=load_evidence_audit,
        lineage_audit=lineage_audit,
        release_boundary_audit=release_boundary_audit,
        docs_audit=docs_audit,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage8_gate_audit(
    *,
    stage8_summary: dict[str, Any],
    stage8_bundle_manifest: dict[str, Any],
    stage8_scope_audit: dict[str, Any],
    stage8_evidence_freeze_ledger: dict[str, Any],
    stage8_doc_consistency: dict[str, Any],
    stage8_release_boundary: dict[str, Any],
    stage8_claim_text: str,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if (
        stage8_summary.get("status") != "passed"
        or _string_list(stage8_summary.get("reason_codes"))
        or stage8_summary.get("publication_verdict") != EXPECTED_STAGE8_VERDICT
        or stage8_summary.get("scoped_claim_publication_approved") is not True
        or stage8_summary.get("next_required_change") != EXPECTED_STAGE8_NEXT
    ):
        _add_reason(reason_codes, "stage8_not_passed")
    if stage8_summary.get("performance_claim_scope") != EXPECTED_SCOPE:
        _add_reason(reason_codes, "claim_scope_overbroad")
    if stage8_scope_audit and stage8_scope_audit.get("scope_audit_passed") is not True:
        _add_reason(reason_codes, "claim_scope_overbroad")
    if stage8_bundle_manifest and stage8_bundle_manifest.get("publication_verdict") != EXPECTED_STAGE8_VERDICT:
        _add_reason(reason_codes, "stage8_not_passed")
    if stage8_evidence_freeze_ledger and stage8_evidence_freeze_ledger.get("evidence_freeze_complete") is not True:
        _add_reason(reason_codes, "stage8_not_passed")
    if stage8_doc_consistency and stage8_doc_consistency.get("doc_consistency_audit_passed") is not True:
        _add_reason(reason_codes, "stage8_not_passed")
    if stage8_release_boundary and stage8_release_boundary.get("release_boundary_audit_passed") is not True:
        _add_reason(reason_codes, "stage8_not_passed")
    if "current offline guarded shadow/canary" not in stage8_claim_text.lower():
        _add_reason(reason_codes, "claim_scope_overbroad")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "stage8_gate",
        "stage8_gate_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
    }


def _identity_audit(*, checkpoint_path: Path | None, selected_summary: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    expected_sha = selected_summary.get("checkpoint_sha256")
    expected_size = _int_or_none(selected_summary.get("checkpoint_size_bytes"))
    actual_sha = None
    actual_size = None
    exists = checkpoint_path is not None and checkpoint_path.is_file()
    if not exists:
        _add_reason(reason_codes, "checkpoint_missing")
    else:
        assert checkpoint_path is not None
        actual_sha, actual_size = _sha256_and_size(checkpoint_path)
        if expected_sha and actual_sha != expected_sha:
            _add_reason(reason_codes, "checkpoint_hash_mismatch")
        if expected_size is not None and actual_size != expected_size:
            _add_reason(reason_codes, "checkpoint_size_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "checkpoint_identity",
        "checkpoint_identity_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_exists": exists,
        "checkpoint_sha256": actual_sha,
        "expected_checkpoint_sha256": expected_sha,
        "checkpoint_size_bytes": actual_size,
        "expected_checkpoint_size_bytes": expected_size,
    }


def _metadata_audit(
    *,
    metadata_path: Path | None,
    metadata: dict[str, Any],
    checkpoint_path: Path | None,
    selected_summary: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    violations: list[str] = []
    if not metadata_path or not metadata_path.is_file() or not metadata:
        violations.append("metadata_missing")
    if metadata.get("experimental") is not True:
        violations.append("metadata_not_experimental")
    selected_seed = selected_summary.get("selected_seed")
    metadata_seed = metadata.get("selected_seed", metadata.get("seed"))
    if selected_seed is not None and metadata_seed is not None and int(metadata_seed) != int(selected_seed):
        violations.append("selected_seed_mismatch")
    selected_budget = selected_summary.get("selected_budget")
    metadata_budget = metadata.get("selected_budget", metadata.get("budget"))
    if selected_budget is not None and metadata_budget is not None and str(metadata_budget) != str(selected_budget):
        violations.append("selected_budget_mismatch")
    metadata_checkpoint = _resolve_optional_path(metadata.get("checkpoint_path"), metadata_path.parent if metadata_path else repo_root, repo_root)
    if checkpoint_path and metadata_checkpoint and metadata_checkpoint.resolve() != checkpoint_path.resolve():
        violations.append("checkpoint_path_mismatch")
    for field in (
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "performance_claimed",
        "formal_training_ready_claimed",
        "real_world_release_approved",
    ):
        if metadata.get(field) is True:
            violations.append(field)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "checkpoint_metadata",
        "checkpoint_metadata_audit_passed": not violations,
        "reason_codes": [] if not violations else ["checkpoint_metadata_invalid"],
        "metadata_path": str(metadata_path) if metadata_path else None,
        "violations": violations,
        "metadata_experimental": metadata.get("experimental"),
        "metadata_seed": metadata_seed,
        "metadata_budget": metadata_budget,
    }


def _load_evidence_audit(
    *,
    selected_summary: dict[str, Any],
    inference_audit: dict[str, Any],
    inference_audit_path: Path | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    selected_checks = _load_counts(selected_summary)
    inference_checks = _load_counts(inference_audit) if inference_audit else {}
    selected_passed = _load_passed(selected_summary, selected_checks)
    inference_passed = True if not inference_audit else _load_passed(inference_audit, inference_checks)
    if not selected_passed or not inference_passed:
        _add_reason(reason_codes, "checkpoint_load_evidence_not_passed")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "checkpoint_load_evidence",
        "checkpoint_load_evidence_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "inference_audit": str(inference_audit_path) if inference_audit_path else None,
        "selected_summary_counts": selected_checks,
        "inference_audit_counts": inference_checks,
    }


def _lineage_audit(
    *,
    read_reasons: list[str],
    stage8_summary: dict[str, Any],
    stage7_summary: dict[str, Any],
    stage6_summary: dict[str, Any],
    cost_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    selected_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        _source("stage8_summary", _passed(stage8_summary)),
        _source("stage7_summary", _passed(stage7_summary)),
        _source("stage6_summary", _passed(stage6_summary)),
        _source("cost_efficiency_summary", _passed(cost_summary)),
        _source("formal_training_summary", _passed(formal_summary)),
        _source("post_training_replay_summary", _passed(replay_summary)),
        _source("selected_candidate_summary", _passed(selected_summary)),
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "checkpoint_publication_lineage",
        "lineage_audit_passed": not read_reasons and all(source["passed"] for source in sources),
        "sources": sources,
        "missing_or_unreadable_inputs": list(read_reasons),
    }


def _release_boundary_audit(*, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for name, payload in payloads.items():
        for field in RELEASE_BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append(f"{name}.{field}")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "checkpoint_publication_release_boundary",
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _docs_audit(*, repo_root: Path) -> dict[str, Any]:
    required_docs = {
        "README.md": repo_root / "README.md",
        "docs/算法设计与系统架构报告.md": repo_root / "docs" / "算法设计与系统架构报告.md",
        "docs/superpowers/specs/2026-06-15-checkpoint-publication-authorization-preflight.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-15-checkpoint-publication-authorization-preflight.md",
    }
    required_phrases = {
        "README.md": [
            "Stage 9 `Checkpoint Publication Authorization Preflight v1`",
            "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1/",
            "checkpoint_publication_package_preparation",
        ],
        "docs/算法设计与系统架构报告.md": [
            "阶段 9 `Checkpoint Publication Authorization Preflight v1`",
            "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1/",
            "checkpoint_publication_package_preparation",
        ],
        "docs/superpowers/specs/2026-06-15-checkpoint-publication-authorization-preflight.md": [
            "checkpoint-publication-authorization-preflight-summary.json",
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
        "audit_name": "documentation",
        "docs_audit_passed": not missing_docs and not missing_phrases,
        "missing_docs": missing_docs,
        "missing_phrases": missing_phrases,
    }


def _authorization_matrix(
    *,
    approved: bool,
    reason_codes: list[str],
    stage8_gate: dict[str, Any],
    identity_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    load_evidence_audit: dict[str, Any],
    lineage_audit: dict[str, Any],
    release_boundary_audit: dict[str, Any],
    docs_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "authorization_matrix",
        "authorization_verdict": EXPECTED_AUTHORIZATION_VERDICT if approved else _failed_verdict(reason_codes),
        "checkpoint_publication_package_preparation_approved": approved,
        "reason_codes": list(reason_codes),
        "stage8_gate_audit_passed": stage8_gate.get("stage8_gate_audit_passed") is True,
        "checkpoint_identity_audit_passed": identity_audit.get("checkpoint_identity_audit_passed") is True,
        "checkpoint_metadata_audit_passed": metadata_audit.get("checkpoint_metadata_audit_passed") is True,
        "checkpoint_load_evidence_audit_passed": load_evidence_audit.get("checkpoint_load_evidence_audit_passed") is True,
        "lineage_audit_passed": lineage_audit.get("lineage_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary_audit.get("release_boundary_audit_passed") is True,
        "docs_audit_passed": docs_audit.get("docs_audit_passed") is True,
        "blocked_release_actions": [
            "checkpoint_publication",
            "default_policy_replacement",
            "real_executor_connection",
        ],
    }


def _candidate_manifest(
    *,
    approved: bool,
    checkpoint_path: Path | None,
    metadata_path: Path | None,
    selected_summary: dict[str, Any],
    identity_audit: dict[str, Any],
    stage8_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "authorization_verdict": EXPECTED_AUTHORIZATION_VERDICT if approved else "not_authorized",
        "selected_seed": selected_summary.get("selected_seed"),
        "selected_budget": selected_summary.get("selected_budget"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_metadata_path": str(metadata_path) if metadata_path else None,
        "checkpoint_sha256": identity_audit.get("checkpoint_sha256"),
        "checkpoint_size_bytes": identity_audit.get("checkpoint_size_bytes"),
        "stage8_summary": stage8_summary.get("summary"),
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
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
    stage8_root: Path,
    stage7_root: Path,
    stage6_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    selected_summary: dict[str, Any],
    identity_audit: dict[str, Any],
    metadata_audit: dict[str, Any],
    load_evidence_audit: dict[str, Any],
    lineage_audit: dict[str, Any],
    release_boundary_audit: dict[str, Any],
    docs_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "authorization_verdict": EXPECTED_AUTHORIZATION_VERDICT if approved else _failed_verdict(reason_codes),
        "checkpoint_publication_authorization_preflight_passed": approved,
        "checkpoint_publication_package_preparation_approved": approved,
        "checkpoint_identity_audit_passed": identity_audit.get("checkpoint_identity_audit_passed") is True,
        "checkpoint_metadata_audit_passed": metadata_audit.get("checkpoint_metadata_audit_passed") is True,
        "checkpoint_load_evidence_audit_passed": load_evidence_audit.get("checkpoint_load_evidence_audit_passed") is True,
        "lineage_audit_passed": lineage_audit.get("lineage_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary_audit.get("release_boundary_audit_passed") is True,
        "docs_audit_passed": docs_audit.get("docs_audit_passed") is True,
        "selected_seed": selected_summary.get("selected_seed"),
        "selected_budget": selected_summary.get("selected_budget"),
        "checkpoint_path": identity_audit.get("checkpoint_path"),
        "checkpoint_sha256": identity_audit.get("checkpoint_sha256"),
        "checkpoint_size_bytes": identity_audit.get("checkpoint_size_bytes"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_authorization_rejections",
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "runs_new_ppo_update": False,
        "copies_checkpoint_to_publication_path": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "repo_root": str(repo_root),
        "stage8_root": str(stage8_root),
        "stage7_root": str(stage7_root),
        "stage6_root": str(stage6_root),
        "cost_efficiency_root": str(cost_efficiency_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "candidate_manifest": str(paths["candidate_manifest"]),
        "identity_audit": str(paths["identity_audit"]),
        "metadata_audit": str(paths["metadata_audit"]),
        "load_evidence_audit": str(paths["load_evidence_audit"]),
        "lineage_audit": str(paths["lineage_audit"]),
        "release_boundary_audit": str(paths["release_boundary_audit"]),
        "authorization_matrix": str(paths["authorization_matrix"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Checkpoint Publication Authorization Preflight v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Authorization verdict: `{summary['authorization_verdict']}`\n"
        f"- Selected seed: `{summary.get('selected_seed')}`\n"
        f"- Selected budget: `{summary.get('selected_budget')}`\n"
        f"- Checkpoint SHA-256: `{summary.get('checkpoint_sha256')}`\n\n"
        "## Release Boundaries\n\n"
        f"- Checkpoint publication approved: `{summary['checkpoint_publication_approved']}`\n"
        f"- Default policy replacement approved: `{summary['default_policy_replacement_approved']}`\n"
        f"- Real executor connection approved: `{summary['real_executor_connection_approved']}`\n"
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "candidate_manifest": output_root / CANDIDATE_MANIFEST_FILE,
        "identity_audit": output_root / IDENTITY_AUDIT_FILE,
        "metadata_audit": output_root / METADATA_AUDIT_FILE,
        "load_evidence_audit": output_root / LOAD_EVIDENCE_AUDIT_FILE,
        "lineage_audit": output_root / LINEAGE_AUDIT_FILE,
        "release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_FILE,
        "authorization_matrix": output_root / AUTHORIZATION_MATRIX_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "invalid_action_mask_count": _int(payload.get("invalid_action_mask_count")),
        "missing_observation_count": _int(payload.get("missing_observation_count")),
        "non_finite_logits_count": _int(payload.get("non_finite_logits_count")),
        "non_finite_log_prob_count": _int(payload.get("non_finite_log_prob_count")),
        "non_finite_value_count": _int(payload.get("non_finite_value_count")),
    }


def _load_passed(payload: dict[str, Any], counts: dict[str, int]) -> bool:
    return payload.get("checkpoint_load_passed") is True and all(value == 0 for value in counts.values())


def _passed(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "passed" and not _string_list(payload.get("reason_codes"))


def _source(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _read_json(path: Path | None, reasons: list[str], label: str) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        _add_reason(reasons, label)
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reasons, label)
        return {}


def _read_text(path: Path | None, reasons: list[str], label: str) -> str:
    if path is None or not Path(path).is_file():
        _add_reason(reasons, label)
        return ""
    return Path(path).read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_path(summary: dict[str, Any], field: str, root: Path, repo_root: Path, default_name: str) -> Path:
    raw = summary.get(field)
    if isinstance(raw, str) and raw:
        return _resolve_path(Path(raw), repo_root)
    return root / default_name


def _resolve_optional_path(value: Any, root: Path, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return _resolve_path(Path(value), repo_root if not Path(value).is_absolute() else root)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _int(value: Any) -> int:
    parsed = _int_or_none(value)
    return 0 if parsed is None else parsed


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
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
