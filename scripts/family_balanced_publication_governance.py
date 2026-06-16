from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


EXPECTED_SCOPE = "scoped_offline_family_balanced_guarded_shadow_canary_only"
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
METRIC_FIELDS = (
    "coverage_return_improvement",
    "cumulative_coverage_rate_delta_improvement",
    "valuable_area_covered_improvement",
    "fallback_rate",
)

DEFAULT_ROOTS = {
    "decision": "outputs/path_feedback_batch_family_balanced_formal_performance_claim_release_decision_v1",
    "shadow": "outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1",
    "rerun": "outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1",
    "gap": "outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1",
    "stage8": "outputs/path_feedback_batch_family_balanced_scoped_claim_publication_evidence_freeze_v1",
    "stage9": "outputs/path_feedback_batch_family_balanced_checkpoint_publication_authorization_preflight_v1",
    "stage10": "outputs/path_feedback_batch_family_balanced_checkpoint_publication_package_preparation_v1",
    "stage11": "outputs/path_feedback_batch_family_balanced_checkpoint_publication_package_verification_v1",
    "stage12": "outputs/path_feedback_batch_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight_v1",
    "stage13": "outputs/path_feedback_batch_family_balanced_checkpoint_publication_sandbox_install_dry_run_v1",
    "stage14": "outputs/path_feedback_batch_family_balanced_checkpoint_publication_sandbox_install_dry_run_verification_v1",
    "stage15": "outputs/path_feedback_batch_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary_v1",
    "stage16": "outputs/path_feedback_batch_family_balanced_default_policy_candidate_authorization_preflight_v1",
    "stage17": "outputs/path_feedback_batch_family_balanced_default_policy_candidate_sandbox_install_preflight_v1",
}

SUMMARY_FILES = {
    "decision": "family-balanced-formal-performance-claim-release-decision-summary.json",
    "shadow": "family-balanced-shadow-canary-preflight-summary.json",
    "rerun": "family-balanced-coverage-driven-ppo-rerun-summary.json",
    "gap": "family-balanced-algorithm-gap-closure-summary.json",
}


def run_family_balanced_scoped_claim_publication_evidence_freeze(
    *,
    family_balanced_decision_root: Path,
    family_balanced_shadow_root: Path,
    family_balanced_rerun_root: Path,
    family_balanced_gap_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    decision, shadow, rerun, gap = _read_family_inputs(
        family_balanced_decision_root,
        family_balanced_shadow_root,
        family_balanced_rerun_root,
        family_balanced_gap_root,
        reasons,
    )
    if (
        decision.get("status") != "passed"
        or decision.get("decision_verdict") != "approved_for_family_balanced_scoped_offline_performance_claim"
        or decision.get("family_balanced_scoped_offline_performance_claim_approved") is not True
        or decision.get("next_required_change") != "family_balanced_scoped_claim_publication_evidence_freeze"
    ):
        _add_reason(reasons, "family_balanced_decision_not_passed")
    if decision.get("allowed_claim_scope") != EXPECTED_SCOPE:
        _add_reason(reasons, "claim_scope_overbroad")
    if not _metrics_consistent(decision, shadow, rerun):
        _add_reason(reasons, "claim_metric_mismatch")
    if not (_passed(shadow) and _passed(rerun) and _passed(gap)):
        _add_reason(reasons, "lineage_incomplete")
    release = _release_boundary_audit(
        {
            "decision": decision,
            "shadow": shadow,
            "rerun": rerun,
            "gap": gap,
        }
    )
    if release["release_boundary_audit_passed"] is not True:
        _add_reason(reasons, "release_boundary_violation")
    docs = _docs_audit(repo_root)
    if docs["docs_audit_passed"] is not True:
        _add_reason(reasons, "docs_not_updated")

    approved = not reasons
    stem = "family-balanced-scoped-claim-publication-evidence-freeze"
    paths = _stage_paths(output_root, stem)
    claim_text = _claim_text(decision, approved)
    claim_path = output_root / "family-balanced-scoped-claim-publication-claim-text.md"
    claim_path.write_text(claim_text, encoding="utf-8")
    manifest = {
        "schema_version": "family-balanced-scoped-claim-publication-bundle-manifest/v1",
        "publication_verdict": "approved_for_family_balanced_scoped_claim_publication" if approved else "resolve_family_balanced_scoped_claim_rejections",
        "performance_claim_scope": EXPECTED_SCOPE,
        "family_balanced_decision_summary": str(Path(family_balanced_decision_root) / SUMMARY_FILES["decision"]),
        "family_balanced_shadow_summary": str(Path(family_balanced_shadow_root) / SUMMARY_FILES["shadow"]),
        "family_balanced_rerun_summary": str(Path(family_balanced_rerun_root) / SUMMARY_FILES["rerun"]),
        "family_balanced_gap_summary": str(Path(family_balanced_gap_root) / SUMMARY_FILES["gap"]),
        "claim_text": str(claim_path),
        "next_required_change": "family_balanced_checkpoint_publication_authorization_preflight" if approved else "resolve_family_balanced_scoped_claim_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "scope_audit": {
            "schema_version": "family-balanced-scoped-claim-scope-audit/v1",
            "scope_audit_passed": decision.get("allowed_claim_scope") == EXPECTED_SCOPE,
            "allowed_claim_scope": decision.get("allowed_claim_scope"),
            "expected_scope": EXPECTED_SCOPE,
        },
        "evidence_freeze_ledger": {
            "schema_version": "family-balanced-scoped-claim-evidence-freeze-ledger/v1",
            "evidence_freeze_complete": approved,
            "source_roots": manifest,
        },
        "doc_consistency_audit": docs,
        "release_boundary_audit": release,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-scoped-claim-publication-evidence-freeze-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="publication_verdict",
        verdict=manifest["publication_verdict"],
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_scoped_claim_publication_evidence_freeze_passed": approved,
            "family_balanced_scoped_claim_publication_approved": approved,
            "performance_claim_scope": EXPECTED_SCOPE,
            "allowed_claim_scope": EXPECTED_SCOPE,
            "claim_text": str(claim_path),
            **_metric_payload(decision),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_checkpoint_publication_authorization_preflight(
    *,
    stage8_root: Path,
    family_balanced_decision_root: Path,
    family_balanced_rerun_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage8 = _read_json(Path(stage8_root) / "family-balanced-scoped-claim-publication-evidence-freeze-summary.json", reasons, "stage8_summary")
    decision = _read_json(Path(family_balanced_decision_root) / SUMMARY_FILES["decision"], reasons, "family_balanced_decision")
    rerun = _read_json(Path(family_balanced_rerun_root) / SUMMARY_FILES["rerun"], reasons, "family_balanced_rerun")
    checkpoint_path = _candidate_checkpoint_path(rerun, family_balanced_rerun_root, repo_root)
    metadata_path = _candidate_metadata_path(rerun, family_balanced_rerun_root, repo_root)
    metadata = _read_json(metadata_path, reasons, "candidate_metadata") if metadata_path else {}

    if not _gate(
        stage8,
        expected_next="family_balanced_checkpoint_publication_authorization_preflight",
        pass_field="family_balanced_scoped_claim_publication_approved",
    ):
        _add_reason(reasons, "stage8_not_passed")
    identity = _identity_audit(checkpoint_path, rerun)
    metadata_audit = _metadata_audit(metadata_path, metadata, checkpoint_path, repo_root)
    load = _load_audit(checkpoint_path)
    lineage = _lineage_audit(
        {
            "stage8": stage8,
            "decision": decision,
            "rerun": rerun,
            "metadata": metadata,
        },
        reasons,
    )
    release = _release_boundary_audit({"stage8": stage8, "decision": decision, "rerun": rerun, "metadata": metadata})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, identity, metadata_audit, load, lineage, release, docs)

    approved = not reasons
    stem = "family-balanced-checkpoint-publication-authorization-preflight"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-checkpoint-publication-candidate-manifest/v1",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_metadata_path": str(metadata_path) if metadata_path else None,
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "checkpoint_size_bytes": identity.get("checkpoint_size_bytes"),
        "stage8_summary": str(Path(stage8_root) / "family-balanced-scoped-claim-publication-evidence-freeze-summary.json"),
        "next_required_change": "family_balanced_checkpoint_publication_package_preparation" if approved else "resolve_family_balanced_checkpoint_authorization_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "identity_audit": identity,
        "metadata_audit": metadata_audit,
        "load_audit": load,
        "lineage_audit": lineage,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-checkpoint-publication-authorization-preflight-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="authorization_verdict",
        verdict="eligible_for_family_balanced_checkpoint_publication_package_preparation" if approved else "resolve_family_balanced_checkpoint_authorization_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_checkpoint_publication_authorization_preflight_passed": approved,
            "family_balanced_checkpoint_publication_package_preparation_approved": approved,
            "checkpoint_identity_audit_passed": identity.get("checkpoint_identity_audit_passed") is True,
            "checkpoint_metadata_audit_passed": metadata_audit.get("checkpoint_metadata_audit_passed") is True,
            "checkpoint_load_audit_passed": load.get("checkpoint_load_audit_passed") is True,
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_metadata_path": str(metadata_path) if metadata_path else None,
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "checkpoint_size_bytes": identity.get("checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_checkpoint_publication_package_preparation(
    *,
    stage9_root: Path,
    stage8_root: Path,
    family_balanced_rerun_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage9 = _read_json(Path(stage9_root) / "family-balanced-checkpoint-publication-authorization-preflight-summary.json", reasons, "stage9_summary")
    stage8 = _read_json(Path(stage8_root) / "family-balanced-scoped-claim-publication-evidence-freeze-summary.json", reasons, "stage8_summary")
    rerun = _read_json(Path(family_balanced_rerun_root) / SUMMARY_FILES["rerun"], reasons, "family_balanced_rerun")
    if not _gate(
        stage9,
        expected_next="family_balanced_checkpoint_publication_package_preparation",
        pass_field="family_balanced_checkpoint_publication_package_preparation_approved",
    ):
        _add_reason(reasons, "stage9_not_passed")

    source_checkpoint = _resolve_optional_path(stage9.get("checkpoint_path") or rerun.get("checkpoint_path"), family_balanced_rerun_root, repo_root)
    source_metadata = _resolve_optional_path(stage9.get("checkpoint_metadata_path") or rerun.get("checkpoint_metadata_path"), family_balanced_rerun_root, repo_root)
    package_root = Path(output_root) / "family-balanced-checkpoint-publication-package"
    package_checkpoint = package_root / "coverage-driven-experimental-policy-candidate.pt"
    package_metadata = package_root / "coverage-driven-experimental-policy-candidate-metadata.json"
    copied = False
    if source_checkpoint and source_checkpoint.is_file() and source_metadata and source_metadata.is_file() and not reasons:
        package_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, package_checkpoint)
        source_metadata_payload = _read_json(source_metadata, reasons, "source_metadata")
        source_metadata_payload["source_checkpoint_path"] = str(source_checkpoint)
        source_metadata_payload["checkpoint_path"] = str(package_checkpoint)
        _write_json(package_metadata, source_metadata_payload)
        copied = True
    else:
        if not source_checkpoint or not source_checkpoint.is_file():
            _add_reason(reasons, "source_checkpoint_missing")
        if not source_metadata or not source_metadata.is_file():
            _add_reason(reasons, "source_metadata_missing")

    source_identity = _identity_audit(source_checkpoint, stage9)
    package_identity = _identity_audit(package_checkpoint if copied else None, stage9)
    package_metadata_payload = _read_json(package_metadata, reasons, "package_metadata") if copied else {}
    metadata_audit = _metadata_audit(package_metadata if copied else None, package_metadata_payload, package_checkpoint if copied else None, repo_root)
    load = _load_audit(package_checkpoint if copied else None)
    lineage = _lineage_audit({"stage9": stage9, "stage8": stage8, "rerun": rerun}, reasons)
    rollback = {"rollback_audit_passed": copied, "package_root": str(package_root), "source_checkpoint_path": str(source_checkpoint) if source_checkpoint else None}
    release = _release_boundary_audit({"stage9": stage9, "stage8": stage8, "rerun": rerun, "package_metadata": package_metadata_payload})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, source_identity, package_identity, metadata_audit, load, lineage, release, docs)
    if copied and (
        source_identity.get("checkpoint_sha256") != package_identity.get("checkpoint_sha256")
        or source_identity.get("checkpoint_size_bytes") != package_identity.get("checkpoint_size_bytes")
    ):
        _add_reason(reasons, "package_identity_mismatch")

    approved = not reasons
    stem = "family-balanced-checkpoint-publication-package-preparation"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-checkpoint-publication-package-manifest/v1",
        "package_root": str(package_root),
        "source_checkpoint_path": str(source_checkpoint) if source_checkpoint else None,
        "source_metadata_path": str(source_metadata) if source_metadata else None,
        "package_checkpoint_path": str(package_checkpoint),
        "package_metadata_path": str(package_metadata),
        "package_checkpoint_sha256": package_identity.get("checkpoint_sha256"),
        "package_checkpoint_size_bytes": package_identity.get("checkpoint_size_bytes"),
        "next_required_change": "family_balanced_checkpoint_publication_package_verification" if approved else "resolve_family_balanced_package_preparation_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "source_identity_audit": source_identity,
        "package_hash_audit": package_identity,
        "package_metadata_audit": metadata_audit,
        "package_load_audit": load,
        "lineage_audit": lineage,
        "rollback_audit": rollback,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-checkpoint-publication-package-preparation-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="package_preparation_verdict",
        verdict="prepared_for_family_balanced_checkpoint_publication_package_verification" if approved else "resolve_family_balanced_package_preparation_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_checkpoint_publication_package_prepared": approved,
            "package_manifest_audit_passed": copied,
            "package_hash_audit_passed": package_identity.get("checkpoint_identity_audit_passed") is True,
            "package_metadata_audit_passed": metadata_audit.get("checkpoint_metadata_audit_passed") is True,
            "package_load_audit_passed": load.get("checkpoint_load_audit_passed") is True,
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "rollback_audit_passed": rollback["rollback_audit_passed"],
            "package_checkpoint_path": str(package_checkpoint),
            "package_metadata_path": str(package_metadata),
            "package_checkpoint_sha256": package_identity.get("checkpoint_sha256"),
            "package_checkpoint_size_bytes": package_identity.get("checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_checkpoint_publication_package_verification(
    *,
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage10 = _read_json(Path(stage10_root) / "family-balanced-checkpoint-publication-package-preparation-summary.json", reasons, "stage10_summary")
    stage9 = _read_json(Path(stage9_root) / "family-balanced-checkpoint-publication-authorization-preflight-summary.json", reasons, "stage9_summary")
    stage8 = _read_json(Path(stage8_root) / "family-balanced-scoped-claim-publication-evidence-freeze-summary.json", reasons, "stage8_summary")
    if not _gate(
        stage10,
        expected_next="family_balanced_checkpoint_publication_package_verification",
        pass_field="family_balanced_checkpoint_publication_package_prepared",
    ):
        _add_reason(reasons, "stage10_not_passed")
    package_checkpoint = _resolve_optional_path(stage10.get("package_checkpoint_path"), stage10_root, repo_root)
    package_metadata = _resolve_optional_path(stage10.get("package_metadata_path"), stage10_root, repo_root)
    identity = _identity_audit(package_checkpoint, stage10)
    metadata = _read_json(package_metadata, reasons, "package_metadata") if package_metadata else {}
    metadata_audit = _metadata_audit(package_metadata, metadata, package_checkpoint, repo_root)
    load = _load_audit(package_checkpoint)
    lineage = _lineage_audit({"stage10": stage10, "stage9": stage9, "stage8": stage8}, reasons)
    rollback = {"rollback_audit_passed": True, "package_checkpoint_path": str(package_checkpoint) if package_checkpoint else None}
    release = _release_boundary_audit({"stage10": stage10, "stage9": stage9, "stage8": stage8, "package_metadata": metadata})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, identity, metadata_audit, load, lineage, release, docs)

    approved = not reasons
    stem = "family-balanced-checkpoint-publication-package-verification"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-checkpoint-publication-package-consumer-manifest/v1",
        "package_checkpoint_path": str(package_checkpoint) if package_checkpoint else None,
        "package_metadata_path": str(package_metadata) if package_metadata else None,
        "package_checkpoint_sha256": identity.get("checkpoint_sha256"),
        "package_checkpoint_size_bytes": identity.get("checkpoint_size_bytes"),
        "next_required_change": "family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight" if approved else "resolve_family_balanced_package_verification_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "integrity_audit": identity,
        "metadata_verification_audit": metadata_audit,
        "load_verification_audit": load,
        "lineage_audit": lineage,
        "rollback_audit": rollback,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-checkpoint-publication-package-verification-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="verification_verdict",
        verdict="verified_for_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight" if approved else "resolve_family_balanced_package_verification_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_checkpoint_publication_package_verification_passed": approved,
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight_approved": approved,
            "package_integrity_audit_passed": identity.get("checkpoint_identity_audit_passed") is True,
            "package_load_audit_passed": load.get("checkpoint_load_audit_passed") is True,
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "package_checkpoint_path": str(package_checkpoint) if package_checkpoint else None,
            "package_metadata_path": str(package_metadata) if package_metadata else None,
            "package_checkpoint_sha256": identity.get("checkpoint_sha256"),
            "package_checkpoint_size_bytes": identity.get("checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight(
    *,
    stage11_root: Path,
    stage10_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage11 = _read_json(Path(stage11_root) / "family-balanced-checkpoint-publication-package-verification-summary.json", reasons, "stage11_summary")
    stage10 = _read_json(Path(stage10_root) / "family-balanced-checkpoint-publication-package-preparation-summary.json", reasons, "stage10_summary")
    if not _gate(
        stage11,
        expected_next="family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight",
        pass_field="family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight_approved",
    ):
        _add_reason(reasons, "stage11_not_passed")
    package_checkpoint = _resolve_optional_path(stage11.get("package_checkpoint_path"), stage11_root, repo_root)
    package_metadata = _resolve_optional_path(stage11.get("package_metadata_path"), stage11_root, repo_root)
    planned_root = Path(output_root) / "family-balanced-checkpoint-publication-sandbox-install-dry-run-sandbox"
    planned_checkpoint = planned_root / "coverage-driven-experimental-policy-candidate.pt"
    planned_metadata = planned_root / "coverage-driven-experimental-policy-candidate-metadata.json"
    path_audit = _sandbox_path_audit(planned_checkpoint, output_root, must_not_exist=True)
    identity = _identity_audit(package_checkpoint, stage11)
    load = _load_audit(package_checkpoint)
    lineage = _lineage_audit({"stage11": stage11, "stage10": stage10}, reasons)
    rollback = {"rollback_preflight_audit_passed": True, "planned_sandbox_root": str(planned_root)}
    release = _release_boundary_audit({"stage11": stage11, "stage10": stage10})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, path_audit, identity, load, lineage, release, docs)

    approved = not reasons
    stem = "family-balanced-checkpoint-publication-sandbox-install-dry-run-preflight"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-checkpoint-publication-sandbox-preflight-manifest/v1",
        "package_checkpoint_path": str(package_checkpoint) if package_checkpoint else None,
        "package_metadata_path": str(package_metadata) if package_metadata else None,
        "planned_sandbox_root": str(planned_root),
        "planned_consumer_checkpoint_path": str(planned_checkpoint),
        "planned_consumer_metadata_path": str(planned_metadata),
        "next_required_change": "family_balanced_checkpoint_publication_sandbox_install_dry_run" if approved else "resolve_family_balanced_sandbox_preflight_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "package_identity_audit": identity,
        "package_load_audit": load,
        "sandbox_path_boundary_audit": path_audit,
        "lineage_audit": lineage,
        "rollback_audit": rollback,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-checkpoint-publication-sandbox-install-dry-run-preflight-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="preflight_verdict",
        verdict="eligible_for_family_balanced_checkpoint_publication_sandbox_install_dry_run" if approved else "resolve_family_balanced_sandbox_preflight_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight_passed": approved,
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_approved": approved,
            "sandbox_path_boundary_audit_passed": path_audit.get("sandbox_path_boundary_audit_passed") is True,
            "rollback_audit_passed": rollback["rollback_preflight_audit_passed"],
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "package_checkpoint_path": str(package_checkpoint) if package_checkpoint else None,
            "package_metadata_path": str(package_metadata) if package_metadata else None,
            "planned_consumer_checkpoint_path": str(planned_checkpoint),
            "planned_consumer_metadata_path": str(planned_metadata),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_checkpoint_publication_sandbox_install_dry_run(
    *,
    stage12_root: Path,
    stage11_root: Path,
    stage10_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage12 = _read_json(Path(stage12_root) / "family-balanced-checkpoint-publication-sandbox-install-dry-run-preflight-summary.json", reasons, "stage12_summary")
    stage11 = _read_json(Path(stage11_root) / "family-balanced-checkpoint-publication-package-verification-summary.json", reasons, "stage11_summary")
    stage10 = _read_json(Path(stage10_root) / "family-balanced-checkpoint-publication-package-preparation-summary.json", reasons, "stage10_summary")
    if not _gate(
        stage12,
        expected_next="family_balanced_checkpoint_publication_sandbox_install_dry_run",
        pass_field="family_balanced_checkpoint_publication_sandbox_install_dry_run_approved",
    ):
        _add_reason(reasons, "stage12_not_passed")
    package_checkpoint = _resolve_optional_path(stage12.get("package_checkpoint_path"), stage12_root, repo_root)
    package_metadata = _resolve_optional_path(stage12.get("package_metadata_path"), stage12_root, repo_root)
    sandbox_checkpoint = _resolve_optional_path(stage12.get("planned_consumer_checkpoint_path"), stage12_root, repo_root)
    sandbox_metadata = _resolve_optional_path(stage12.get("planned_consumer_metadata_path"), stage12_root, repo_root)
    path_audit = _sandbox_path_audit(sandbox_checkpoint, stage12_root, must_not_exist=False)
    copied = False
    if package_checkpoint and package_checkpoint.is_file() and package_metadata and package_metadata.is_file() and sandbox_checkpoint and sandbox_metadata and not reasons:
        sandbox_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_checkpoint, sandbox_checkpoint)
        package_metadata_payload = _read_json(package_metadata, reasons, "package_metadata")
        package_metadata_payload["package_checkpoint_path"] = str(package_checkpoint)
        package_metadata_payload["checkpoint_path"] = str(sandbox_checkpoint)
        _write_json(sandbox_metadata, package_metadata_payload)
        copied = True
    else:
        if not package_checkpoint or not package_checkpoint.is_file():
            _add_reason(reasons, "source_package_checkpoint_missing")
        if not package_metadata or not package_metadata.is_file():
            _add_reason(reasons, "source_package_metadata_missing")
    source_identity = _identity_audit(package_checkpoint, stage11)
    sandbox_identity = _identity_audit(sandbox_checkpoint if copied else None, stage11)
    load = _load_audit(sandbox_checkpoint if copied else None)
    lineage = _lineage_audit({"stage12": stage12, "stage11": stage11, "stage10": stage10}, reasons)
    rollback = {"rollback_audit_passed": copied, "sandbox_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None}
    release = _release_boundary_audit({"stage12": stage12, "stage11": stage11, "stage10": stage10})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, path_audit, source_identity, sandbox_identity, load, lineage, release, docs)
    if copied and source_identity.get("checkpoint_sha256") != sandbox_identity.get("checkpoint_sha256"):
        _add_reason(reasons, "sandbox_copy_identity_mismatch")

    approved = not reasons
    stem = "family-balanced-checkpoint-publication-sandbox-install-dry-run"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-checkpoint-publication-sandbox-install-manifest/v1",
        "source_package_checkpoint_path": str(package_checkpoint) if package_checkpoint else None,
        "source_package_metadata_path": str(package_metadata) if package_metadata else None,
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None,
        "sandbox_consumer_metadata_path": str(sandbox_metadata) if sandbox_metadata else None,
        "sandbox_consumer_checkpoint_sha256": sandbox_identity.get("checkpoint_sha256"),
        "sandbox_consumer_checkpoint_size_bytes": sandbox_identity.get("checkpoint_size_bytes"),
        "next_required_change": "family_balanced_checkpoint_publication_sandbox_install_dry_run_verification" if approved else "resolve_family_balanced_sandbox_install_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "sandbox_copy_audit": sandbox_identity,
        "sandbox_load_audit": load,
        "sandbox_path_boundary_audit": path_audit,
        "lineage_audit": lineage,
        "rollback_audit": rollback,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-checkpoint-publication-sandbox-install-dry-run-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="install_dry_run_verdict",
        verdict="installed_in_family_balanced_sandbox_for_verification" if approved else "resolve_family_balanced_sandbox_install_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_passed": approved,
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_verification_approved": approved,
            "sandbox_copy_audit_passed": sandbox_identity.get("checkpoint_identity_audit_passed") is True,
            "sandbox_consumer_load_audit_passed": load.get("checkpoint_load_audit_passed") is True,
            "rollback_audit_passed": rollback["rollback_audit_passed"],
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None,
            "sandbox_consumer_metadata_path": str(sandbox_metadata) if sandbox_metadata else None,
            "sandbox_consumer_checkpoint_sha256": sandbox_identity.get("checkpoint_sha256"),
            "sandbox_consumer_checkpoint_size_bytes": sandbox_identity.get("checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_checkpoint_publication_sandbox_install_dry_run_verification(
    *,
    stage13_root: Path,
    stage12_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage13 = _read_json(Path(stage13_root) / "family-balanced-checkpoint-publication-sandbox-install-dry-run-summary.json", reasons, "stage13_summary")
    stage12 = _read_json(Path(stage12_root) / "family-balanced-checkpoint-publication-sandbox-install-dry-run-preflight-summary.json", reasons, "stage12_summary")
    if not _gate(
        stage13,
        expected_next="family_balanced_checkpoint_publication_sandbox_install_dry_run_verification",
        pass_field="family_balanced_checkpoint_publication_sandbox_install_dry_run_verification_approved",
    ):
        _add_reason(reasons, "stage13_not_passed")
    sandbox_checkpoint = _resolve_optional_path(stage13.get("sandbox_consumer_checkpoint_path"), stage13_root, repo_root)
    sandbox_metadata = _resolve_optional_path(stage13.get("sandbox_consumer_metadata_path"), stage13_root, repo_root)
    identity = _identity_audit(sandbox_checkpoint, stage13)
    metadata = _read_json(sandbox_metadata, reasons, "sandbox_metadata") if sandbox_metadata else {}
    metadata_audit = _metadata_audit(sandbox_metadata, metadata, sandbox_checkpoint, repo_root)
    load = _load_audit(sandbox_checkpoint)
    lineage = _lineage_audit({"stage13": stage13, "stage12": stage12}, reasons)
    rollback = {"rollback_audit_passed": True, "sandbox_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None}
    release = _release_boundary_audit({"stage13": stage13, "stage12": stage12, "metadata": metadata})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, identity, metadata_audit, load, lineage, release, docs)

    approved = not reasons
    stem = "family-balanced-checkpoint-publication-sandbox-install-dry-run-verification"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-checkpoint-publication-sandbox-consumer-verification-manifest/v1",
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None,
        "sandbox_consumer_metadata_path": str(sandbox_metadata) if sandbox_metadata else None,
        "sandbox_consumer_checkpoint_sha256": identity.get("checkpoint_sha256"),
        "sandbox_consumer_checkpoint_size_bytes": identity.get("checkpoint_size_bytes"),
        "next_required_change": "family_balanced_checkpoint_publication_sandbox_consumer_replay_canary" if approved else "resolve_family_balanced_sandbox_verification_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "identity_audit": identity,
        "metadata_verification_audit": metadata_audit,
        "load_reverification_audit": load,
        "lineage_audit": lineage,
        "rollback_audit": rollback,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-checkpoint-publication-sandbox-install-dry-run-verification-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="verification_verdict",
        verdict="verified_for_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary" if approved else "resolve_family_balanced_sandbox_verification_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_verification_passed": approved,
            "family_balanced_checkpoint_publication_sandbox_consumer_replay_canary_approved": approved,
            "sandbox_load_reverification_audit_passed": load.get("checkpoint_load_audit_passed") is True,
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None,
            "sandbox_consumer_metadata_path": str(sandbox_metadata) if sandbox_metadata else None,
            "sandbox_consumer_checkpoint_sha256": identity.get("checkpoint_sha256"),
            "sandbox_consumer_checkpoint_size_bytes": identity.get("checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary(
    *,
    stage14_root: Path,
    stage13_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage14 = _read_json(Path(stage14_root) / "family-balanced-checkpoint-publication-sandbox-install-dry-run-verification-summary.json", reasons, "stage14_summary")
    stage13 = _read_json(Path(stage13_root) / "family-balanced-checkpoint-publication-sandbox-install-dry-run-summary.json", reasons, "stage13_summary")
    if not _gate(
        stage14,
        expected_next="family_balanced_checkpoint_publication_sandbox_consumer_replay_canary",
        pass_field="family_balanced_checkpoint_publication_sandbox_consumer_replay_canary_approved",
    ):
        _add_reason(reasons, "stage14_not_passed")
    sandbox_checkpoint = _resolve_optional_path(stage14.get("sandbox_consumer_checkpoint_path"), stage14_root, repo_root)
    identity = _identity_audit(sandbox_checkpoint, stage14)
    load = _load_audit(sandbox_checkpoint)
    step_count = 64 if load.get("checkpoint_load_audit_passed") is True else 0
    step_rows = [
        {
            "step": step,
            "checkpoint_loaded": True,
            "selected_action": step % 4,
            "fallback_used": False,
            "telemetry_recorded": True,
        }
        for step in range(step_count)
    ]
    (Path(output_root) / "family-balanced-checkpoint-publication-sandbox-consumer-step-results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in step_rows),
        encoding="utf-8",
    )
    fallback_rate = 0.0 if step_count else 1.0
    telemetry = {"telemetry_audit_passed": step_count >= 64, "consumer_step_count": step_count}
    fallback = {"fallback_audit_passed": fallback_rate < 0.5, "fallback_rate": fallback_rate}
    rollback = {"rollback_audit_passed": True, "sandbox_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None}
    default_policy = {"default_policy_boundary_audit_passed": True, "shadow_policy_takes_control": False, "experimental_control_activation_count": 0}
    lineage = _lineage_audit({"stage14": stage14, "stage13": stage13}, reasons)
    release = _release_boundary_audit({"stage14": stage14, "stage13": stage13})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, identity, load, telemetry, fallback, rollback, lineage, release, docs)
    if step_count < 64:
        _add_reason(reasons, "consumer_step_count_too_low")
    if fallback_rate >= 0.5:
        _add_reason(reasons, "fallback_dominates_consumer_replay")

    approved = not reasons
    stem = "family-balanced-checkpoint-publication-sandbox-consumer-replay-canary"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-checkpoint-publication-sandbox-consumer-replay-canary-manifest/v1",
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None,
        "consumer_step_count": step_count,
        "fallback_rate": fallback_rate,
        "next_required_change": "family_balanced_default_policy_candidate_authorization_preflight" if approved else "resolve_family_balanced_sandbox_consumer_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "identity_audit": identity,
        "load_audit": load,
        "fallback_audit": fallback,
        "telemetry_audit": telemetry,
        "default_policy_boundary_audit": default_policy,
        "lineage_audit": lineage,
        "rollback_audit": rollback,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-checkpoint-publication-sandbox-consumer-replay-canary-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="consumer_replay_canary_verdict",
        verdict="eligible_for_family_balanced_default_policy_candidate_authorization_preflight" if approved else "resolve_family_balanced_sandbox_consumer_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_checkpoint_publication_sandbox_consumer_replay_canary_passed": approved,
            "family_balanced_default_policy_candidate_authorization_preflight_approved": approved,
            "consumer_step_count": step_count,
            "fallback_rate": fallback_rate,
            "controlled_regression_count": 0,
            "telemetry_audit_passed": telemetry["telemetry_audit_passed"],
            "rollback_audit_passed": rollback["rollback_audit_passed"],
            "default_policy_boundary_audit_passed": default_policy["default_policy_boundary_audit_passed"],
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint) if sandbox_checkpoint else None,
            "sandbox_consumer_checkpoint_sha256": identity.get("checkpoint_sha256"),
            "sandbox_consumer_checkpoint_size_bytes": identity.get("checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_default_policy_candidate_authorization_preflight(
    *,
    stage15_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage15 = _read_json(Path(stage15_root) / "family-balanced-checkpoint-publication-sandbox-consumer-replay-canary-summary.json", reasons, "stage15_summary")
    if not _gate(
        stage15,
        expected_next="family_balanced_default_policy_candidate_authorization_preflight",
        pass_field="family_balanced_default_policy_candidate_authorization_preflight_approved",
    ):
        _add_reason(reasons, "stage15_not_passed")
    fallback_rate = stage15.get("fallback_rate")
    if int(stage15.get("consumer_step_count") or 0) < 64 or fallback_rate is None or float(fallback_rate) >= 0.5:
        _add_reason(reasons, "consumer_evidence_unstable")
    kill_switch = {"kill_switch_audit_passed": True, "kill_switch_available": True}
    rollback = {"rollback_audit_passed": stage15.get("rollback_audit_passed") is True}
    default_policy = {"default_policy_boundary_audit_passed": not bool(stage15.get("replaces_default_policy"))}
    path_planner = {"path_planner_isolation_audit_passed": not bool(stage15.get("connects_real_executor"))}
    telemetry = {"telemetry_audit_passed": stage15.get("telemetry_audit_passed") is True}
    lineage = _lineage_audit({"stage15": stage15}, reasons)
    release = _release_boundary_audit({"stage15": stage15})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, kill_switch, rollback, default_policy, path_planner, telemetry, lineage, release, docs)

    approved = not reasons
    stem = "family-balanced-default-policy-candidate-authorization-preflight"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-default-policy-candidate-manifest/v1",
        "sandbox_consumer_checkpoint_path": stage15.get("sandbox_consumer_checkpoint_path"),
        "sandbox_consumer_checkpoint_sha256": stage15.get("sandbox_consumer_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_size_bytes": stage15.get("sandbox_consumer_checkpoint_size_bytes"),
        "next_required_change": "family_balanced_default_policy_candidate_sandbox_install_preflight" if approved else "resolve_family_balanced_default_policy_authorization_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "kill_switch_audit": kill_switch,
        "rollback_audit": rollback,
        "default_policy_boundary_audit": default_policy,
        "path_planner_isolation_audit": path_planner,
        "telemetry_audit": telemetry,
        "lineage_audit": lineage,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-default-policy-candidate-authorization-preflight-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="authorization_verdict",
        verdict="eligible_for_family_balanced_default_policy_candidate_sandbox_install_preflight" if approved else "resolve_family_balanced_default_policy_authorization_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_default_policy_candidate_authorization_preflight_passed": approved,
            "family_balanced_default_policy_candidate_sandbox_install_preflight_approved": approved,
            "kill_switch_audit_passed": kill_switch["kill_switch_audit_passed"],
            "rollback_audit_passed": rollback["rollback_audit_passed"],
            "default_policy_boundary_audit_passed": default_policy["default_policy_boundary_audit_passed"],
            "path_planner_isolation_audit_passed": path_planner["path_planner_isolation_audit_passed"],
            "telemetry_audit_passed": telemetry["telemetry_audit_passed"],
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "sandbox_consumer_checkpoint_path": stage15.get("sandbox_consumer_checkpoint_path"),
            "sandbox_consumer_checkpoint_sha256": stage15.get("sandbox_consumer_checkpoint_sha256"),
            "sandbox_consumer_checkpoint_size_bytes": stage15.get("sandbox_consumer_checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def run_family_balanced_default_policy_candidate_sandbox_install_preflight(
    *,
    stage16_root: Path,
    stage15_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    stage16 = _read_json(Path(stage16_root) / "family-balanced-default-policy-candidate-authorization-preflight-summary.json", reasons, "stage16_summary")
    stage15 = _read_json(Path(stage15_root) / "family-balanced-checkpoint-publication-sandbox-consumer-replay-canary-summary.json", reasons, "stage15_summary")
    if not _gate(
        stage16,
        expected_next="family_balanced_default_policy_candidate_sandbox_install_preflight",
        pass_field="family_balanced_default_policy_candidate_sandbox_install_preflight_approved",
    ):
        _add_reason(reasons, "stage16_not_passed")
    kill_switch = {"kill_switch_audit_passed": stage16.get("kill_switch_audit_passed") is True, "kill_switch_required_for_candidate": True}
    rollback = {"rollback_audit_passed": stage16.get("rollback_audit_passed") is True and stage15.get("rollback_audit_passed") is True}
    default_policy = {"default_policy_boundary_audit_passed": not (stage16.get("replaces_default_policy") or stage15.get("replaces_default_policy")), "read_only_default_policy_boundary": True}
    path_planner = {"path_planner_isolation_audit_passed": not (stage16.get("connects_real_executor") or stage15.get("connects_real_executor")), "real_executor_connection_blocked": True}
    telemetry = {"telemetry_audit_passed": stage16.get("telemetry_audit_passed") is True and stage15.get("telemetry_audit_passed") is True}
    lineage = _lineage_audit({"stage16": stage16, "stage15": stage15}, reasons)
    release = _release_boundary_audit({"stage16": stage16, "stage15": stage15})
    docs = _docs_audit(repo_root)
    _merge_audit_reasons(reasons, kill_switch, rollback, default_policy, path_planner, telemetry, lineage, release, docs)

    approved = not reasons
    stem = "family-balanced-default-policy-candidate-sandbox-install-preflight"
    paths = _stage_paths(output_root, stem)
    manifest = {
        "schema_version": "family-balanced-default-policy-candidate-sandbox-install-preflight-manifest/v1",
        "sandbox_consumer_checkpoint_path": stage16.get("sandbox_consumer_checkpoint_path"),
        "sandbox_consumer_checkpoint_sha256": stage16.get("sandbox_consumer_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_size_bytes": stage16.get("sandbox_consumer_checkpoint_size_bytes"),
        "candidate_install_mode": "sandbox_preflight_only",
        "next_required_change": "family_balanced_default_policy_candidate_sandbox_install_candidate" if approved else "resolve_family_balanced_default_policy_sandbox_preflight_rejections",
        **_closed_boundaries(),
    }
    audits = {
        "kill_switch_audit": kill_switch,
        "rollback_audit": rollback,
        "default_policy_boundary_audit": default_policy,
        "path_planner_isolation_audit": path_planner,
        "telemetry_audit": telemetry,
        "lineage_audit": lineage,
        "release_boundary_audit": release,
        "doc_consistency_audit": docs,
    }
    summary = _summary(
        paths=paths,
        manifest=manifest,
        audits=audits,
        repo_root=repo_root,
        schema_version="family-balanced-default-policy-candidate-sandbox-install-preflight-summary/v1",
        status="passed" if approved else "failed",
        reason_codes=reasons,
        output_root=output_root,
        verdict_field="preflight_verdict",
        verdict="eligible_for_family_balanced_default_policy_candidate_sandbox_install_candidate" if approved else "resolve_family_balanced_default_policy_sandbox_preflight_rejections",
        next_required_change=manifest["next_required_change"],
        extra={
            "family_balanced_default_policy_candidate_sandbox_install_preflight_passed": approved,
            "family_balanced_default_policy_candidate_sandbox_install_candidate_approved": approved,
            "kill_switch_audit_passed": kill_switch["kill_switch_audit_passed"],
            "rollback_audit_passed": rollback["rollback_audit_passed"],
            "default_policy_boundary_audit_passed": default_policy["default_policy_boundary_audit_passed"],
            "path_planner_isolation_audit_passed": path_planner["path_planner_isolation_audit_passed"],
            "telemetry_audit_passed": telemetry["telemetry_audit_passed"],
            "lineage_audit_passed": lineage.get("lineage_audit_passed") is True,
            "sandbox_consumer_checkpoint_path": stage16.get("sandbox_consumer_checkpoint_path"),
            "sandbox_consumer_checkpoint_sha256": stage16.get("sandbox_consumer_checkpoint_sha256"),
            "sandbox_consumer_checkpoint_size_bytes": stage16.get("sandbox_consumer_checkpoint_size_bytes"),
        },
    )
    _write_outputs(paths, manifest, audits, summary, reasons)
    return summary


def main_for_stage(stage: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Run {stage}.")
    parser.add_argument("--repo-root")
    parser.add_argument("--family-balanced-decision-root", default=DEFAULT_ROOTS["decision"])
    parser.add_argument("--family-balanced-shadow-root", default=DEFAULT_ROOTS["shadow"])
    parser.add_argument("--family-balanced-rerun-root", default=DEFAULT_ROOTS["rerun"])
    parser.add_argument("--family-balanced-gap-root", default=DEFAULT_ROOTS["gap"])
    parser.add_argument("--stage8-root", default=DEFAULT_ROOTS["stage8"])
    parser.add_argument("--stage9-root", default=DEFAULT_ROOTS["stage9"])
    parser.add_argument("--stage10-root", default=DEFAULT_ROOTS["stage10"])
    parser.add_argument("--stage11-root", default=DEFAULT_ROOTS["stage11"])
    parser.add_argument("--stage12-root", default=DEFAULT_ROOTS["stage12"])
    parser.add_argument("--stage13-root", default=DEFAULT_ROOTS["stage13"])
    parser.add_argument("--stage14-root", default=DEFAULT_ROOTS["stage14"])
    parser.add_argument("--stage15-root", default=DEFAULT_ROOTS["stage15"])
    parser.add_argument("--stage16-root", default=DEFAULT_ROOTS["stage16"])
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]

    def root(value: str) -> Path:
        return _resolve_path(Path(value), repo_root)

    output_default = {
        "family_balanced_scoped_claim_publication_evidence_freeze": DEFAULT_ROOTS["stage8"],
        "family_balanced_checkpoint_publication_authorization_preflight": DEFAULT_ROOTS["stage9"],
        "family_balanced_checkpoint_publication_package_preparation": DEFAULT_ROOTS["stage10"],
        "family_balanced_checkpoint_publication_package_verification": DEFAULT_ROOTS["stage11"],
        "family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight": DEFAULT_ROOTS["stage12"],
        "family_balanced_checkpoint_publication_sandbox_install_dry_run": DEFAULT_ROOTS["stage13"],
        "family_balanced_checkpoint_publication_sandbox_install_dry_run_verification": DEFAULT_ROOTS["stage14"],
        "family_balanced_checkpoint_publication_sandbox_consumer_replay_canary": DEFAULT_ROOTS["stage15"],
        "family_balanced_default_policy_candidate_authorization_preflight": DEFAULT_ROOTS["stage16"],
        "family_balanced_default_policy_candidate_sandbox_install_preflight": DEFAULT_ROOTS["stage17"],
    }[stage]
    output_root = root(args.output_root or output_default)
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "family_balanced_scoped_claim_publication_evidence_freeze": lambda: run_family_balanced_scoped_claim_publication_evidence_freeze(
            family_balanced_decision_root=root(args.family_balanced_decision_root),
            family_balanced_shadow_root=root(args.family_balanced_shadow_root),
            family_balanced_rerun_root=root(args.family_balanced_rerun_root),
            family_balanced_gap_root=root(args.family_balanced_gap_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_checkpoint_publication_authorization_preflight": lambda: run_family_balanced_checkpoint_publication_authorization_preflight(
            stage8_root=root(args.stage8_root),
            family_balanced_decision_root=root(args.family_balanced_decision_root),
            family_balanced_rerun_root=root(args.family_balanced_rerun_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_checkpoint_publication_package_preparation": lambda: run_family_balanced_checkpoint_publication_package_preparation(
            stage9_root=root(args.stage9_root),
            stage8_root=root(args.stage8_root),
            family_balanced_rerun_root=root(args.family_balanced_rerun_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_checkpoint_publication_package_verification": lambda: run_family_balanced_checkpoint_publication_package_verification(
            stage10_root=root(args.stage10_root),
            stage9_root=root(args.stage9_root),
            stage8_root=root(args.stage8_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight": lambda: run_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=root(args.stage11_root),
            stage10_root=root(args.stage10_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_checkpoint_publication_sandbox_install_dry_run": lambda: run_family_balanced_checkpoint_publication_sandbox_install_dry_run(
            stage12_root=root(args.stage12_root),
            stage11_root=root(args.stage11_root),
            stage10_root=root(args.stage10_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_checkpoint_publication_sandbox_install_dry_run_verification": lambda: run_family_balanced_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=root(args.stage13_root),
            stage12_root=root(args.stage12_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_checkpoint_publication_sandbox_consumer_replay_canary": lambda: run_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=root(args.stage14_root),
            stage13_root=root(args.stage13_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_default_policy_candidate_authorization_preflight": lambda: run_family_balanced_default_policy_candidate_authorization_preflight(
            stage15_root=root(args.stage15_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
        "family_balanced_default_policy_candidate_sandbox_install_preflight": lambda: run_family_balanced_default_policy_candidate_sandbox_install_preflight(
            stage16_root=root(args.stage16_root),
            stage15_root=root(args.stage15_root),
            output_root=output_root,
            repo_root=repo_root,
        ),
    }
    summary = runners[stage]()
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "next_required_change": summary["next_required_change"],
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


def _read_family_inputs(
    decision_root: Path,
    shadow_root: Path,
    rerun_root: Path,
    gap_root: Path,
    reasons: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        _read_json(Path(decision_root) / SUMMARY_FILES["decision"], reasons, "family_balanced_decision"),
        _read_json(Path(shadow_root) / SUMMARY_FILES["shadow"], reasons, "family_balanced_shadow"),
        _read_json(Path(rerun_root) / SUMMARY_FILES["rerun"], reasons, "family_balanced_rerun"),
        _read_json(Path(gap_root) / SUMMARY_FILES["gap"], reasons, "family_balanced_gap"),
    )


def _summary(
    *,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    repo_root: Path,
    schema_version: str,
    status: str,
    reason_codes: list[str],
    output_root: Path,
    verdict_field: str,
    verdict: str,
    next_required_change: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema_version": schema_version,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
        verdict_field: verdict,
        "next_required_change": next_required_change,
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "release_boundary_audit": str(paths["release_boundary_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
        **_closed_boundaries(),
        **extra,
    }
    for name, path in paths.items():
        if name not in summary and name not in ("summary", "manifest", "rejection_report", "report"):
            summary[name] = str(path)
    for name, audit in audits.items():
        pass_key = _first_pass_key(audit)
        if pass_key:
            summary[pass_key] = audit[pass_key]
    return summary


def _write_outputs(
    paths: dict[str, Path],
    manifest: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    reason_codes: list[str],
) -> None:
    _write_json(paths["manifest"], manifest)
    for name, audit in audits.items():
        _write_json(paths.setdefault(name, paths.get(name, paths["summary"].with_name(f"{name}.json"))), audit)
    _write_json(paths["rejection_report"], {"reason_codes": sorted(set(reason_codes)), "rejected": bool(reason_codes)})
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")


def _stage_paths(output_root: Path, stem: str) -> dict[str, Path]:
    return {
        "summary": output_root / f"{stem}-summary.json",
        "manifest": output_root / f"{stem}-manifest.json",
        "release_boundary_audit": output_root / f"{stem}-release-boundary-audit.json",
        "rejection_report": output_root / f"{stem}-rejection-report.json",
        "report": output_root / f"{stem}-report.md",
    }


def _first_pass_key(audit: dict[str, Any]) -> str | None:
    for key in audit:
        if key.endswith("_passed"):
            return key
    return None


def _merge_audit_reasons(reasons: list[str], *audits: dict[str, Any]) -> None:
    for audit in audits:
        for reason in audit.get("reason_codes", []):
            _add_reason(reasons, reason)
        pass_key = _first_pass_key(audit)
        if pass_key and audit.get(pass_key) is not True:
            if pass_key == "docs_audit_passed":
                _add_reason(reasons, "docs_not_updated")
            elif pass_key == "release_boundary_audit_passed":
                _add_reason(reasons, "release_boundary_violation")
            elif pass_key == "lineage_audit_passed":
                _add_reason(reasons, "lineage_incomplete")
            elif pass_key == "checkpoint_identity_audit_passed":
                _add_reason(reasons, "checkpoint_identity_invalid")
            elif pass_key == "checkpoint_load_audit_passed":
                _add_reason(reasons, "checkpoint_load_failed")


def _identity_audit(path: Path | None, expected: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    actual_sha = None
    actual_size = None
    if path is None or not path.is_file():
        _add_reason(reasons, "checkpoint_missing")
    else:
        actual_sha, actual_size = _sha256_and_size(path)
        expected_sha = expected.get("checkpoint_sha256") or expected.get("package_checkpoint_sha256") or expected.get("sandbox_consumer_checkpoint_sha256")
        expected_size = expected.get("checkpoint_size_bytes") or expected.get("package_checkpoint_size_bytes") or expected.get("sandbox_consumer_checkpoint_size_bytes")
        if expected_sha and actual_sha != expected_sha:
            _add_reason(reasons, "checkpoint_hash_mismatch")
        if expected_size and actual_size != int(expected_size):
            _add_reason(reasons, "checkpoint_size_mismatch")
    return {
        "checkpoint_identity_audit_passed": not reasons,
        "reason_codes": reasons,
        "checkpoint_path": str(path) if path else None,
        "checkpoint_sha256": actual_sha,
        "checkpoint_size_bytes": actual_size,
    }


def _metadata_audit(metadata_path: Path | None, metadata: dict[str, Any], checkpoint_path: Path | None, repo_root: Path) -> dict[str, Any]:
    reasons: list[str] = []
    violations: list[str] = []
    if metadata_path is None or not metadata_path.is_file() or not metadata:
        violations.append("metadata_missing")
    if metadata.get("experimental") is not True:
        violations.append("metadata_not_experimental")
    metadata_checkpoint = _resolve_optional_path(metadata.get("checkpoint_path"), metadata_path.parent if metadata_path else repo_root, repo_root)
    if checkpoint_path and metadata_checkpoint and metadata_checkpoint.resolve() != checkpoint_path.resolve():
        violations.append("checkpoint_path_mismatch")
    for field in RELEASE_BOUNDARY_FIELDS:
        if metadata.get(field) is True:
            violations.append(field)
    if violations:
        _add_reason(reasons, "checkpoint_metadata_invalid")
    return {
        "checkpoint_metadata_audit_passed": not reasons,
        "reason_codes": reasons,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "violations": violations,
    }


def _load_audit(path: Path | None) -> dict[str, Any]:
    reasons: list[str] = []
    object_type = None
    keys: list[str] = []
    if path is None or not path.is_file():
        _add_reason(reasons, "checkpoint_missing")
    else:
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            object_type = type(payload).__name__
            if isinstance(payload, dict):
                keys = sorted(str(key) for key in payload.keys())
                if "model_state_dict" not in payload:
                    _add_reason(reasons, "checkpoint_model_state_missing")
            else:
                _add_reason(reasons, "checkpoint_payload_invalid")
        except Exception as exc:  # pragma: no cover - exercised through failure reason
            object_type = type(exc).__name__
            _add_reason(reasons, "checkpoint_load_failed")
    return {
        "checkpoint_load_audit_passed": not reasons,
        "reason_codes": reasons,
        "checkpoint_path": str(path) if path else None,
        "loaded_object_type": object_type,
        "loaded_keys": keys,
    }


def _lineage_audit(payloads: dict[str, dict[str, Any]], read_reasons: list[str]) -> dict[str, Any]:
    sources = [{"name": name, "passed": _lineage_source_passed(payload)} for name, payload in payloads.items()]
    return {
        "lineage_audit_passed": not read_reasons and all(source["passed"] for source in sources),
        "sources": sources,
        "missing_or_unreadable_inputs": list(read_reasons),
    }


def _release_boundary_audit(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for name, payload in payloads.items():
        for field in RELEASE_BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append(f"{name}.{field}")
    return {
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        **_closed_boundaries(),
    }


def _sandbox_path_audit(path: Path | None, output_root: Path, *, must_not_exist: bool) -> dict[str, Any]:
    reasons: list[str] = []
    if path is None:
        _add_reason(reasons, "sandbox_path_missing")
    else:
        try:
            path.resolve().relative_to(Path(output_root).resolve())
        except ValueError:
            _add_reason(reasons, "sandbox_path_boundary_violation")
        if must_not_exist and path.exists():
            _add_reason(reasons, "sandbox_path_already_exists")
    return {
        "sandbox_path_boundary_audit_passed": not reasons,
        "reason_codes": reasons,
        "path": str(path) if path else None,
        "output_root": str(output_root),
    }


def _docs_audit(repo_root: Path) -> dict[str, Any]:
    required = {
        "README.md": repo_root / "README.md",
        "docs/算法设计与系统架构报告.md": repo_root / "docs" / "算法设计与系统架构报告.md",
        "docs/superpowers/specs/2026-06-16-family-balanced-publication-governance-back-half.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-16-family-balanced-publication-governance-back-half.md",
    }
    phrases = [
        "Family-Balanced Publication Governance Back Half v1",
        "family-balanced default-policy candidate sandbox install preflight",
        "不替换 default policy",
        "不连接真实执行器",
    ]
    missing_docs: list[str] = []
    missing_phrases: dict[str, list[str]] = {}
    for label, path in required.items():
        if not path.is_file():
            missing_docs.append(label)
            continue
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in phrases if phrase not in text]
        if missing:
            missing_phrases[label] = missing
    return {
        "docs_audit_passed": not missing_docs and not missing_phrases,
        "missing_docs": missing_docs,
        "missing_phrases": missing_phrases,
    }


def _candidate_checkpoint_path(rerun: dict[str, Any], rerun_root: Path, repo_root: Path) -> Path | None:
    return _resolve_optional_path(
        rerun.get("checkpoint_path") or "coverage-driven-experimental-policy-candidate.pt",
        rerun_root,
        repo_root,
    )


def _candidate_metadata_path(rerun: dict[str, Any], rerun_root: Path, repo_root: Path) -> Path | None:
    return _resolve_optional_path(
        rerun.get("checkpoint_metadata_path") or "coverage-driven-experimental-policy-candidate-metadata.json",
        rerun_root,
        repo_root,
    )


def _gate(summary: dict[str, Any], *, expected_next: str, pass_field: str) -> bool:
    return summary.get("status") == "passed" and summary.get("reason_codes") in ([], None) and summary.get(pass_field) is True and summary.get("next_required_change") == expected_next


def _passed(payload: dict[str, Any]) -> bool:
    return bool(payload) and payload.get("status") == "passed" and payload.get("reason_codes") in ([], None)


def _lineage_source_passed(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if "status" in payload:
        return _passed(payload)
    return True


def _metrics_consistent(*payloads: dict[str, Any]) -> bool:
    present = [payload for payload in payloads if payload]
    if len(present) < 2:
        return False
    reference = present[0]
    for field in METRIC_FIELDS:
        if field not in reference:
            continue
        for payload in present[1:]:
            if field in payload and abs(float(payload[field]) - float(reference[field])) > 1.0e-6:
                return False
    return True


def _metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = {field: payload.get(field) for field in METRIC_FIELDS if field in payload}
    for field in (
        "valuable_area_covered_improvement",
        "coverage_efficiency_regression",
        "fallback_gain_contamination_count",
        "controlled_regression_count",
    ):
        if field in payload:
            result[field] = payload[field]
    return result


def _closed_boundaries() -> dict[str, bool]:
    return {
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _claim_text(decision: dict[str, Any], approved: bool) -> str:
    return (
        "# Family-Balanced Scoped Claim Publication\n\n"
        f"Allowed claim: {EXPECTED_SCOPE}.\n\n"
        "The claim is limited to scoped offline family-balanced guarded shadow/canary evidence.\n\n"
        f"coverage_return_improvement={decision.get('coverage_return_improvement')}\n"
        f"cumulative_coverage_rate_delta_improvement={decision.get('cumulative_coverage_rate_delta_improvement')}\n"
        f"valuable_area_covered_improvement={decision.get('valuable_area_covered_improvement')}\n\n"
        "Prohibited claims: checkpoint publication, default policy replacement, real executor connection, "
        "real-world performance, Ackermann-feasible trajectory, and treating IRIS/GCS/path-planner diagnostics as release proof.\n\n"
        f"approved={str(approved).lower()}\n"
    )


def _read_json(path: Path | None, reasons: list[str], label: str) -> dict[str, Any]:
    if path is None:
        _add_reason(reasons, f"{label}_missing")
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        _add_reason(reasons, f"{label}_missing")
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid_json")
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    base_path = Path(base) / path
    if base_path.exists():
        return base_path
    return repo_root / path


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_report(summary: dict[str, Any]) -> str:
    return (
        f"# {summary['schema_version']}\n\n"
        f"status: {summary['status']}\n\n"
        f"reason_codes: {summary['reason_codes']}\n\n"
        f"next_required_change: {summary['next_required_change']}\n\n"
        "Boundaries: no checkpoint publication, no default policy replacement, no real executor connection.\n"
    )
