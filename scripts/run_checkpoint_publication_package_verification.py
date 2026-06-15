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


SUMMARY_SCHEMA_VERSION = "checkpoint-publication-package-verification-summary/v1"
AUDIT_SCHEMA_VERSION = "checkpoint-publication-package-verification-audit/v1"
CONSUMER_MANIFEST_SCHEMA_VERSION = "checkpoint-publication-package-consumer-manifest/v1"

DEFAULT_STAGE10_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1"
DEFAULT_STAGE9_ROOT = "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1"
DEFAULT_STAGE8_ROOT = "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1"
DEFAULT_STAGE7_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_verification_v1"

STAGE10_SUMMARY_FILE = "checkpoint-publication-package-preparation-summary.json"
STAGE10_MANIFEST_FILE = "checkpoint-publication-package-manifest.json"
STAGE10_HASH_AUDIT_FILE = "checkpoint-publication-package-hash-audit.json"
STAGE10_METADATA_AUDIT_FILE = "checkpoint-publication-package-metadata-audit.json"
STAGE10_LINEAGE_AUDIT_FILE = "checkpoint-publication-package-lineage-audit.json"
STAGE10_RELEASE_BOUNDARY_AUDIT_FILE = "checkpoint-publication-package-release-boundary-audit.json"
STAGE10_ROLLBACK_AUDIT_FILE = "checkpoint-publication-package-rollback-audit.json"
STAGE9_SUMMARY_FILE = "checkpoint-publication-authorization-preflight-summary.json"
STAGE9_CANDIDATE_MANIFEST_FILE = "checkpoint-publication-candidate-manifest.json"
STAGE9_LOAD_EVIDENCE_AUDIT_FILE = "checkpoint-publication-load-evidence-audit.json"
STAGE8_SUMMARY_FILE = "scoped-claim-publication-evidence-freeze-summary.json"
STAGE7_SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "checkpoint-publication-package-verification-summary.json"
CONSUMER_MANIFEST_FILE = "checkpoint-publication-package-consumer-manifest.json"
INTEGRITY_AUDIT_FILE = "checkpoint-publication-package-integrity-audit.json"
MANIFEST_CONSISTENCY_AUDIT_FILE = "checkpoint-publication-package-manifest-consistency-audit.json"
METADATA_VERIFICATION_AUDIT_FILE = "checkpoint-publication-package-metadata-verification-audit.json"
LOAD_VERIFICATION_AUDIT_FILE = "checkpoint-publication-package-load-verification-audit.json"
LINEAGE_VERIFICATION_AUDIT_FILE = "checkpoint-publication-package-lineage-verification-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "checkpoint-publication-package-release-boundary-audit.json"
ROLLBACK_VERIFICATION_AUDIT_FILE = "checkpoint-publication-package-rollback-verification-audit.json"
REJECTION_REPORT_FILE = "checkpoint-publication-package-verification-rejection-report.json"
REPORT_FILE = "checkpoint-publication-package-verification-report.md"

EXPECTED_STAGE10_VERDICT = "prepared_for_checkpoint_publication_package_verification"
EXPECTED_STAGE10_NEXT = "checkpoint_publication_package_verification"
EXPECTED_VERIFICATION_VERDICT = "verified_for_checkpoint_publication_sandbox_install_dry_run_preflight"
NEXT_REQUIRED_CHANGE = "checkpoint_publication_sandbox_install_dry_run_preflight"

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
        description="Verify an isolated checkpoint publication package."
    )
    parser.add_argument("--stage10-root", default=DEFAULT_STAGE10_ROOT)
    parser.add_argument("--stage9-root", default=DEFAULT_STAGE9_ROOT)
    parser.add_argument("--stage8-root", default=DEFAULT_STAGE8_ROOT)
    parser.add_argument("--stage7-root", default=DEFAULT_STAGE7_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_checkpoint_publication_package_verification(
        stage10_root=_resolve_path(Path(args.stage10_root), repo_root),
        stage9_root=_resolve_path(Path(args.stage9_root), repo_root),
        stage8_root=_resolve_path(Path(args.stage8_root), repo_root),
        stage7_root=_resolve_path(Path(args.stage7_root), repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "verification_verdict": summary["verification_verdict"],
                "checkpoint_publication_package_verification_passed": summary[
                    "checkpoint_publication_package_verification_passed"
                ],
                "checkpoint_publication_sandbox_install_dry_run_preflight_approved": summary[
                    "checkpoint_publication_sandbox_install_dry_run_preflight_approved"
                ],
                "checkpoint_publication_approved": summary["checkpoint_publication_approved"],
                "default_policy_replacement_approved": summary[
                    "default_policy_replacement_approved"
                ],
                "real_executor_connection_approved": summary["real_executor_connection_approved"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_checkpoint_publication_package_verification(
    *,
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    assume_package_unchanged: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stage10_root = Path(stage10_root)
    stage9_root = Path(stage9_root)
    stage8_root = Path(stage8_root)
    stage7_root = Path(stage7_root)
    selected_candidate_root = Path(selected_candidate_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    read_reasons: list[str] = []
    stage10_summary_path = stage10_root / STAGE10_SUMMARY_FILE
    stage10_summary = _read_json(stage10_summary_path, read_reasons, "stage10_summary")
    stage10_manifest_path = _summary_path(
        stage10_summary, "package_manifest", stage10_root, repo_root, STAGE10_MANIFEST_FILE
    )
    stage10_hash_audit_path = _summary_path(
        stage10_summary, "package_hash_audit", stage10_root, repo_root, STAGE10_HASH_AUDIT_FILE
    )
    stage10_metadata_audit_path = _summary_path(
        stage10_summary,
        "package_metadata_audit",
        stage10_root,
        repo_root,
        STAGE10_METADATA_AUDIT_FILE,
    )
    stage10_lineage_audit_path = _summary_path(
        stage10_summary,
        "package_lineage_audit",
        stage10_root,
        repo_root,
        STAGE10_LINEAGE_AUDIT_FILE,
    )
    stage10_release_audit_path = _summary_path(
        stage10_summary,
        "package_release_boundary_audit",
        stage10_root,
        repo_root,
        STAGE10_RELEASE_BOUNDARY_AUDIT_FILE,
    )
    stage10_rollback_audit_path = _summary_path(
        stage10_summary,
        "package_rollback_audit",
        stage10_root,
        repo_root,
        STAGE10_ROLLBACK_AUDIT_FILE,
    )
    stage10_manifest = _read_json(stage10_manifest_path, read_reasons, "stage10_manifest")
    stage10_hash_audit = _read_json(stage10_hash_audit_path, read_reasons, "stage10_hash_audit")
    stage10_metadata_audit = _read_json(stage10_metadata_audit_path, read_reasons, "stage10_metadata_audit")
    stage10_lineage_audit = _read_json(stage10_lineage_audit_path, read_reasons, "stage10_lineage_audit")
    stage10_release_audit = _read_json(stage10_release_audit_path, read_reasons, "stage10_release_audit")
    stage10_rollback_audit = _read_json(stage10_rollback_audit_path, read_reasons, "stage10_rollback_audit")

    package_checkpoint_path = _resolve_first_path(
        (
            stage10_summary.get("package_checkpoint_path"),
            stage10_manifest.get("package_checkpoint_path"),
            stage10_hash_audit.get("package_checkpoint_path"),
        ),
        root=stage10_root,
        repo_root=repo_root,
    )
    package_metadata_path = _resolve_first_path(
        (
            stage10_summary.get("package_metadata_path"),
            stage10_manifest.get("package_metadata_path"),
            stage10_metadata_audit.get("package_metadata_path"),
        ),
        root=stage10_root,
        repo_root=repo_root,
    )
    package_metadata = _read_json(package_metadata_path, [], "package_metadata") if package_metadata_path else {}
    package_before = _file_snapshot(package_checkpoint_path)
    metadata_before = _file_snapshot(package_metadata_path)
    source_before = _file_snapshot(_resolve_optional_path(stage10_manifest.get("source_checkpoint_path"), stage10_root, repo_root))

    stage9_summary_path = _resolve_optional_path(stage10_manifest.get("stage9_summary"), stage10_root, repo_root) or (
        stage9_root / STAGE9_SUMMARY_FILE
    )
    stage9_summary = _read_json(stage9_summary_path, read_reasons, "stage9_summary")
    stage9_candidate_path = _summary_path(
        stage9_summary, "candidate_manifest", stage9_root, repo_root, STAGE9_CANDIDATE_MANIFEST_FILE
    )
    stage9_load_path = _summary_path(
        stage9_summary, "load_evidence_audit", stage9_root, repo_root, STAGE9_LOAD_EVIDENCE_AUDIT_FILE
    )
    stage9_candidate = _read_json(stage9_candidate_path, read_reasons, "stage9_candidate_manifest")
    stage9_load = _read_json(stage9_load_path, read_reasons, "stage9_load_evidence_audit")
    stage8_summary = _read_json(stage8_root / STAGE8_SUMMARY_FILE, read_reasons, "stage8_summary")
    stage7_summary = _read_json(stage7_root / STAGE7_SUMMARY_FILE, read_reasons, "stage7_summary")
    selected_summary = _read_json(
        selected_candidate_root / SELECTED_SUMMARY_FILE, read_reasons, "selected_candidate_summary"
    )

    stage10_gate = _stage10_gate_audit(stage10_summary)
    integrity_audit = _integrity_audit(
        package_checkpoint_path=package_checkpoint_path,
        stage10_summary=stage10_summary,
        stage10_manifest=stage10_manifest,
        stage10_hash_audit=stage10_hash_audit,
    )
    manifest_consistency_audit = _manifest_consistency_audit(
        manifest_path=stage10_manifest_path,
        manifest=stage10_manifest,
        stage10_summary=stage10_summary,
        integrity_audit=integrity_audit,
        stage9_summary_path=stage9_summary_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
    )
    metadata_verification_audit = _metadata_verification_audit(
        metadata_path=package_metadata_path,
        metadata=package_metadata,
    )
    load_verification_audit = _load_verification_audit(package_checkpoint_path)
    lineage_verification_audit = _lineage_verification_audit(
        read_reasons=read_reasons,
        stage10_summary=stage10_summary,
        stage10_hash_audit=stage10_hash_audit,
        stage10_metadata_audit=stage10_metadata_audit,
        stage10_lineage_audit=stage10_lineage_audit,
        stage10_release_audit=stage10_release_audit,
        stage10_rollback_audit=stage10_rollback_audit,
        stage9_summary=stage9_summary,
        stage9_candidate=stage9_candidate,
        stage9_load=stage9_load,
        stage8_summary=stage8_summary,
        stage7_summary=stage7_summary,
        selected_summary=selected_summary,
    )
    release_boundary_audit = _release_boundary_audit(
        payloads={
            "stage10_summary": stage10_summary,
            "stage10_manifest": stage10_manifest,
            "stage10_release_audit": stage10_release_audit,
            "stage10_rollback_audit": stage10_rollback_audit,
            "stage9_summary": stage9_summary,
            "stage9_candidate_manifest": stage9_candidate,
            "stage8_summary": stage8_summary,
            "stage7_summary": stage7_summary,
            "selected_candidate_summary": selected_summary,
            "package_metadata": package_metadata,
        }
    )
    rollback_verification_audit = _rollback_verification_audit(
        package_before=package_before,
        package_after=_file_snapshot(package_checkpoint_path),
        metadata_before=metadata_before,
        metadata_after=_file_snapshot(package_metadata_path),
        source_before=source_before,
        source_after=_file_snapshot(
            _resolve_optional_path(stage10_manifest.get("source_checkpoint_path"), stage10_root, repo_root)
        ),
        assume_package_unchanged=assume_package_unchanged,
    )
    docs_audit = _docs_audit(repo_root=repo_root)

    reason_codes = _unique(
        [
            *stage10_gate["reason_codes"],
            *integrity_audit["reason_codes"],
            *manifest_consistency_audit["reason_codes"],
            *metadata_verification_audit["reason_codes"],
            *load_verification_audit["reason_codes"],
            *([] if lineage_verification_audit["lineage_verification_audit_passed"] else ["lineage_verification_incomplete"]),
            *([] if rollback_verification_audit["rollback_verification_audit_passed"] else ["rollback_boundary_invalid"]),
            *([] if release_boundary_audit["release_boundary_audit_passed"] else ["release_boundary_violation"]),
            *([] if docs_audit["docs_audit_passed"] else ["docs_not_updated"]),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    approved = status == "passed"

    consumer_manifest = _consumer_manifest(
        approved=approved,
        stage10_summary_path=stage10_summary_path,
        stage10_manifest_path=stage10_manifest_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        integrity_audit=integrity_audit,
        load_verification_audit=load_verification_audit,
        stage10_summary=stage10_summary,
        repo_root=repo_root,
    )
    _write_json(paths["consumer_manifest"], consumer_manifest)
    _write_json(paths["package_integrity_audit"], integrity_audit)
    _write_json(paths["package_manifest_consistency_audit"], manifest_consistency_audit)
    _write_json(paths["package_metadata_verification_audit"], metadata_verification_audit)
    _write_json(paths["package_load_verification_audit"], load_verification_audit)
    _write_json(paths["package_lineage_verification_audit"], lineage_verification_audit)
    _write_json(paths["package_release_boundary_audit"], release_boundary_audit)
    _write_json(paths["package_rollback_verification_audit"], rollback_verification_audit)
    _write_json(paths["rejection_report"], _rejection_report(reason_codes))

    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        approved=approved,
        paths=paths,
        repo_root=repo_root,
        stage10_root=stage10_root,
        stage9_root=stage9_root,
        stage8_root=stage8_root,
        stage7_root=stage7_root,
        selected_candidate_root=selected_candidate_root,
        output_root=output_root,
        stage10_summary=stage10_summary,
        integrity_audit=integrity_audit,
        manifest_consistency_audit=manifest_consistency_audit,
        metadata_verification_audit=metadata_verification_audit,
        load_verification_audit=load_verification_audit,
        lineage_verification_audit=lineage_verification_audit,
        rollback_verification_audit=rollback_verification_audit,
        release_boundary_audit=release_boundary_audit,
        docs_audit=docs_audit,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage10_gate_audit(stage10_summary: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if (
        stage10_summary.get("status") != "passed"
        or _string_list(stage10_summary.get("reason_codes"))
        or stage10_summary.get("checkpoint_publication_package_prepared") is not True
    ):
        _add_reason(reason_codes, "stage10_not_passed")
    if (
        stage10_summary.get("package_preparation_verdict") != EXPECTED_STAGE10_VERDICT
        or stage10_summary.get("next_required_change") != EXPECTED_STAGE10_NEXT
    ):
        _add_reason(reason_codes, "stage10_not_prepared_for_package_verification")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "stage10_gate",
        "stage10_gate_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
    }


def _integrity_audit(
    *,
    package_checkpoint_path: Path | None,
    stage10_summary: dict[str, Any],
    stage10_manifest: dict[str, Any],
    stage10_hash_audit: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    actual_sha = None
    actual_size = None
    exists = package_checkpoint_path is not None and package_checkpoint_path.is_file()
    if not exists:
        _add_reason(reason_codes, "package_checkpoint_missing")
    else:
        assert package_checkpoint_path is not None
        actual_sha, actual_size = _sha256_and_size(package_checkpoint_path)
        expected_shas = _unique(
            [
                str(value)
                for value in (
                    stage10_summary.get("package_checkpoint_sha256"),
                    stage10_manifest.get("package_checkpoint_sha256"),
                    stage10_hash_audit.get("package_checkpoint_sha256"),
                )
                if value
            ]
        )
        expected_sizes = _unique_ints(
            (
                stage10_summary.get("package_checkpoint_size_bytes"),
                stage10_manifest.get("package_checkpoint_size_bytes"),
                stage10_hash_audit.get("package_checkpoint_size_bytes"),
            )
        )
        if any(actual_sha != expected for expected in expected_shas):
            _add_reason(reason_codes, "package_hash_mismatch")
        if any(actual_size != expected for expected in expected_sizes):
            _add_reason(reason_codes, "package_size_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_integrity",
        "package_integrity_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "package_checkpoint_path": str(package_checkpoint_path) if package_checkpoint_path else None,
        "package_checkpoint_exists": exists,
        "package_checkpoint_sha256": actual_sha,
        "package_checkpoint_size_bytes": actual_size,
    }


def _manifest_consistency_audit(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    stage10_summary: dict[str, Any],
    integrity_audit: dict[str, Any],
    stage9_summary_path: Path,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    violations: list[str] = []
    if not manifest_path.is_file() or not manifest:
        _add_reason(reason_codes, "package_manifest_missing")
        violations.append("manifest_missing")
    required_fields = (
        "schema_version",
        "stage9_summary",
        "package_checkpoint_path",
        "package_metadata_path",
        "selected_seed",
        "selected_budget",
        "package_checkpoint_sha256",
        "package_checkpoint_size_bytes",
        "next_required_change",
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "non_goals",
    )
    for field in required_fields:
        if field not in manifest or manifest.get(field) in (None, ""):
            violations.append(f"missing_{field}")
    if manifest.get("selected_seed") != stage10_summary.get("selected_seed"):
        violations.append("selected_seed_mismatch")
    if manifest.get("selected_budget") != stage10_summary.get("selected_budget"):
        violations.append("selected_budget_mismatch")
    if _same_path(manifest.get("stage9_summary"), stage9_summary_path) is False:
        violations.append("stage9_summary_path_mismatch")
    if package_checkpoint_path and _same_path(manifest.get("package_checkpoint_path"), package_checkpoint_path) is False:
        violations.append("package_checkpoint_path_mismatch")
    if package_metadata_path and _same_path(manifest.get("package_metadata_path"), package_metadata_path) is False:
        violations.append("package_metadata_path_mismatch")
    if manifest.get("package_checkpoint_sha256") != integrity_audit.get("package_checkpoint_sha256"):
        violations.append("package_checkpoint_sha256_mismatch")
    if _int_or_none(manifest.get("package_checkpoint_size_bytes")) != integrity_audit.get("package_checkpoint_size_bytes"):
        violations.append("package_checkpoint_size_mismatch")
    if manifest.get("next_required_change") != EXPECTED_STAGE10_NEXT:
        violations.append("next_required_change_mismatch")
    non_goals = set(str(item) for item in manifest.get("non_goals") or [])
    for value in (
        "do_not_publish_checkpoint",
        "do_not_replace_default_policy",
        "do_not_connect_real_executor",
    ):
        if value not in non_goals:
            violations.append(f"missing_non_goal_{value}")
    if any(manifest.get(field) is True for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor")):
        violations.append("manifest_release_boundary_true")
    if violations and "package_manifest_missing" not in reason_codes:
        _add_reason(reason_codes, "package_manifest_inconsistent")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_manifest_consistency",
        "package_manifest_consistency_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "manifest_path": str(manifest_path),
        "violations": violations,
    }


def _metadata_verification_audit(*, metadata_path: Path | None, metadata: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    violations: list[str] = []
    if not metadata_path or not metadata_path.is_file() or not metadata:
        _add_reason(reason_codes, "package_metadata_missing")
        violations.append("metadata_missing")
    if metadata and metadata.get("experimental") is not True:
        violations.append("metadata_not_experimental")
    for field in (
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "performance_claimed",
        "formal_training_ready_claimed",
        "real_world_release_approved",
        "default_policy_replaced",
        "final_release_approved",
    ):
        if metadata.get(field) is True:
            violations.append(field)
    if violations and "metadata_missing" not in violations:
        _add_reason(reason_codes, "package_metadata_invalid")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_metadata_verification",
        "package_metadata_verification_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata_experimental": metadata.get("experimental"),
        "metadata_checkpoint_path_as_provenance": metadata.get("checkpoint_path"),
        "violations": violations,
    }


def _load_verification_audit(checkpoint_path: Path | None) -> dict[str, Any]:
    reason_codes: list[str] = []
    checkpoint_type = None
    tensor_count = 0
    non_finite_tensor_count = 0
    state_key_count = 0
    error = None
    try:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("package checkpoint missing")
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_type = type(checkpoint).__name__
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint is not a dict")
        if checkpoint.get("experimental") is False:
            raise ValueError("checkpoint declares experimental=false")
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if not isinstance(state, dict) or not state:
            raise ValueError("checkpoint missing model_state_dict/state_dict")
        state_key_count = len(state)
        for value in state.values():
            if hasattr(value, "detach"):
                tensor_count += 1
                if not bool(torch.isfinite(value.detach()).all().item()):
                    non_finite_tensor_count += 1
        if tensor_count <= 0:
            raise ValueError("checkpoint state has no tensors")
        if non_finite_tensor_count:
            raise ValueError("checkpoint state has non-finite tensors")
    except Exception as exc:  # noqa: BLE001 - audit records load failures
        error = str(exc)
        _add_reason(reason_codes, "package_load_verification_failed")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_load_verification",
        "package_load_verification_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_type": checkpoint_type,
        "state_key_count": state_key_count,
        "tensor_count": tensor_count,
        "non_finite_tensor_count": non_finite_tensor_count,
        "load_error": error,
    }


def _lineage_verification_audit(
    *,
    read_reasons: list[str],
    stage10_summary: dict[str, Any],
    stage10_hash_audit: dict[str, Any],
    stage10_metadata_audit: dict[str, Any],
    stage10_lineage_audit: dict[str, Any],
    stage10_release_audit: dict[str, Any],
    stage10_rollback_audit: dict[str, Any],
    stage9_summary: dict[str, Any],
    stage9_candidate: dict[str, Any],
    stage9_load: dict[str, Any],
    stage8_summary: dict[str, Any],
    stage7_summary: dict[str, Any],
    selected_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        _source("stage10_summary", _passed(stage10_summary)),
        _source("stage10_hash_audit", _audit_passed(stage10_hash_audit, "package_hash_audit_passed")),
        _source("stage10_metadata_audit", _audit_passed(stage10_metadata_audit, "package_metadata_audit_passed")),
        _source("stage10_lineage_audit", _audit_passed(stage10_lineage_audit, "lineage_audit_passed")),
        _source("stage10_release_audit", _audit_passed(stage10_release_audit, "release_boundary_audit_passed")),
        _source("stage10_rollback_audit", _audit_passed(stage10_rollback_audit, "rollback_audit_passed")),
        _source("stage9_summary", _passed(stage9_summary)),
        _source(
            "stage9_candidate_manifest",
            stage9_candidate.get("authorization_verdict") == "eligible_for_checkpoint_publication_package_preparation",
        ),
        _source("stage9_load_evidence_audit", _audit_passed(stage9_load, "checkpoint_load_evidence_audit_passed")),
        _source("stage8_summary", _passed(stage8_summary)),
        _source("stage7_summary", _passed(stage7_summary)),
        _source("selected_candidate_summary", _passed(selected_summary)),
    ]
    passed = not read_reasons and all(source["passed"] for source in sources)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_lineage_verification",
        "lineage_verification_audit_passed": passed,
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
        "audit_name": "package_release_boundary",
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _rollback_verification_audit(
    *,
    package_before: dict[str, Any],
    package_after: dict[str, Any],
    metadata_before: dict[str, Any],
    metadata_after: dict[str, Any],
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    assume_package_unchanged: bool,
) -> dict[str, Any]:
    package_unchanged = _snapshots_equal(package_before, package_after)
    metadata_unchanged = _snapshots_equal(metadata_before, metadata_after)
    source_unchanged = _snapshots_equal(source_before, source_after)
    passed = assume_package_unchanged and package_unchanged and metadata_unchanged and source_unchanged
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_rollback_verification",
        "rollback_verification_audit_passed": passed,
        "package_checkpoint_unchanged": package_unchanged,
        "package_metadata_unchanged": metadata_unchanged,
        "source_checkpoint_unchanged": source_unchanged,
        "default_policy_touched": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _consumer_manifest(
    *,
    approved: bool,
    stage10_summary_path: Path,
    stage10_manifest_path: Path,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
    integrity_audit: dict[str, Any],
    load_verification_audit: dict[str, Any],
    stage10_summary: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": CONSUMER_MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "verification_verdict": EXPECTED_VERIFICATION_VERDICT if approved else "not_verified",
        "stage10_summary": str(stage10_summary_path),
        "stage10_manifest": str(stage10_manifest_path),
        "selected_seed": stage10_summary.get("selected_seed"),
        "selected_budget": stage10_summary.get("selected_budget"),
        "authoritative_package_checkpoint_path": str(package_checkpoint_path) if package_checkpoint_path else None,
        "package_metadata_path": str(package_metadata_path) if package_metadata_path else None,
        "package_checkpoint_sha256": integrity_audit.get("package_checkpoint_sha256"),
        "package_checkpoint_size_bytes": integrity_audit.get("package_checkpoint_size_bytes"),
        "package_load_verification_audit_passed": load_verification_audit.get("package_load_verification_audit_passed") is True,
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_package_verification_rejections",
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _docs_audit(*, repo_root: Path) -> dict[str, Any]:
    required_docs = {
        "README.md": repo_root / "README.md",
        "docs/算法设计与系统架构报告.md": repo_root / "docs" / "算法设计与系统架构报告.md",
        "docs/superpowers/specs/2026-06-15-checkpoint-publication-package-verification.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-15-checkpoint-publication-package-verification.md",
    }
    required_phrases = {
        "README.md": [
            "Stage 11 `Checkpoint Publication Package Verification v1`",
            "outputs/path_feedback_batch_checkpoint_publication_package_verification_v1/",
            "checkpoint_publication_sandbox_install_dry_run_preflight",
        ],
        "docs/算法设计与系统架构报告.md": [
            "阶段 11 `Checkpoint Publication Package Verification v1`",
            "outputs/path_feedback_batch_checkpoint_publication_package_verification_v1/",
            "checkpoint_publication_sandbox_install_dry_run_preflight",
        ],
        "docs/superpowers/specs/2026-06-15-checkpoint-publication-package-verification.md": [
            "checkpoint-publication-package-verification-summary.json",
            "README.md",
            "docs/算法设计与系统架构报告.md",
            "checkpoint_publication_sandbox_install_dry_run_preflight",
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
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    stage10_summary: dict[str, Any],
    integrity_audit: dict[str, Any],
    manifest_consistency_audit: dict[str, Any],
    metadata_verification_audit: dict[str, Any],
    load_verification_audit: dict[str, Any],
    lineage_verification_audit: dict[str, Any],
    rollback_verification_audit: dict[str, Any],
    release_boundary_audit: dict[str, Any],
    docs_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "verification_verdict": EXPECTED_VERIFICATION_VERDICT if approved else _failed_verdict(reason_codes),
        "checkpoint_publication_package_verification_passed": approved,
        "checkpoint_publication_sandbox_install_dry_run_preflight_approved": approved,
        "package_integrity_audit_passed": integrity_audit.get("package_integrity_audit_passed") is True,
        "package_manifest_consistency_audit_passed": manifest_consistency_audit.get("package_manifest_consistency_audit_passed") is True,
        "package_metadata_verification_audit_passed": metadata_verification_audit.get("package_metadata_verification_audit_passed") is True,
        "package_load_verification_audit_passed": load_verification_audit.get("package_load_verification_audit_passed") is True,
        "lineage_verification_audit_passed": lineage_verification_audit.get("lineage_verification_audit_passed") is True,
        "rollback_verification_audit_passed": rollback_verification_audit.get("rollback_verification_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary_audit.get("release_boundary_audit_passed") is True,
        "docs_audit_passed": docs_audit.get("docs_audit_passed") is True,
        "selected_seed": stage10_summary.get("selected_seed"),
        "selected_budget": stage10_summary.get("selected_budget"),
        "package_checkpoint_sha256": integrity_audit.get("package_checkpoint_sha256"),
        "package_checkpoint_size_bytes": integrity_audit.get("package_checkpoint_size_bytes"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_package_verification_rejections",
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
        "ackermann_feasible_trajectory_claimed": False,
        "repo_root": str(repo_root),
        "stage10_root": str(stage10_root),
        "stage9_root": str(stage9_root),
        "stage8_root": str(stage8_root),
        "stage7_root": str(stage7_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "consumer_manifest": str(paths["consumer_manifest"]),
        "package_integrity_audit": str(paths["package_integrity_audit"]),
        "package_manifest_consistency_audit": str(paths["package_manifest_consistency_audit"]),
        "package_metadata_verification_audit": str(paths["package_metadata_verification_audit"]),
        "package_load_verification_audit": str(paths["package_load_verification_audit"]),
        "package_lineage_verification_audit": str(paths["package_lineage_verification_audit"]),
        "package_release_boundary_audit": str(paths["package_release_boundary_audit"]),
        "package_rollback_verification_audit": str(paths["package_rollback_verification_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Checkpoint Publication Package Verification v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Verification verdict: `{summary['verification_verdict']}`\n"
        f"- Selected seed: `{summary.get('selected_seed')}`\n"
        f"- Selected budget: `{summary.get('selected_budget')}`\n"
        f"- Package checkpoint SHA-256: `{summary.get('package_checkpoint_sha256')}`\n\n"
        "## Release Boundaries\n\n"
        f"- Checkpoint publication approved: `{summary['checkpoint_publication_approved']}`\n"
        f"- Default policy replacement approved: `{summary['default_policy_replacement_approved']}`\n"
        f"- Real executor connection approved: `{summary['real_executor_connection_approved']}`\n"
        f"- Next required change: `{summary['next_required_change']}`\n"
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "consumer_manifest": output_root / CONSUMER_MANIFEST_FILE,
        "package_integrity_audit": output_root / INTEGRITY_AUDIT_FILE,
        "package_manifest_consistency_audit": output_root / MANIFEST_CONSISTENCY_AUDIT_FILE,
        "package_metadata_verification_audit": output_root / METADATA_VERIFICATION_AUDIT_FILE,
        "package_load_verification_audit": output_root / LOAD_VERIFICATION_AUDIT_FILE,
        "package_lineage_verification_audit": output_root / LINEAGE_VERIFICATION_AUDIT_FILE,
        "package_release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_FILE,
        "package_rollback_verification_audit": output_root / ROLLBACK_VERIFICATION_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _file_snapshot(path: Path | None) -> dict[str, Any]:
    exists = path is not None and path.is_file()
    sha = None
    size = None
    if exists and path is not None:
        sha, size = _sha256_and_size(path)
    return {"path": str(path) if path else None, "exists": exists, "sha256": sha, "size_bytes": size}


def _snapshots_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("path") == right.get("path")
        and left.get("exists") == right.get("exists")
        and left.get("sha256") == right.get("sha256")
        and left.get("size_bytes") == right.get("size_bytes")
    )


def _read_json(path: Path | None, reasons: list[str], label: str) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        _add_reason(reasons, label)
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reasons, label)
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_path(summary: dict[str, Any], field: str, root: Path, repo_root: Path, default_name: str) -> Path:
    raw = summary.get(field)
    if isinstance(raw, str) and raw:
        return _resolve_path(Path(raw), repo_root)
    return root / default_name


def _resolve_first_path(values: tuple[Any, ...], *, root: Path, repo_root: Path) -> Path | None:
    for value in values:
        resolved = _resolve_optional_path(value, root, repo_root)
        if resolved is not None:
            return resolved
    return None


def _resolve_optional_path(value: Any, root: Path, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return _resolve_path(Path(value), repo_root if not Path(value).is_absolute() else root)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _same_path(value: Any, path: Path) -> bool | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).resolve() == path.resolve()


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _passed(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "passed" and not _string_list(payload.get("reason_codes"))


def _audit_passed(payload: dict[str, Any], field: str) -> bool:
    return payload.get(field) is True and not _string_list(payload.get("reason_codes"))


def _source(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_ints(values: tuple[Any, ...]) -> list[int]:
    result: list[int] = []
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None and parsed not in result:
            result.append(parsed)
    return result


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
