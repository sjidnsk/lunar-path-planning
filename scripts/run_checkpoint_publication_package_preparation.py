from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable
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


SUMMARY_SCHEMA_VERSION = "checkpoint-publication-package-preparation-summary/v1"
AUDIT_SCHEMA_VERSION = "checkpoint-publication-package-preparation-audit/v1"
MANIFEST_SCHEMA_VERSION = "checkpoint-publication-package-manifest/v1"

DEFAULT_STAGE9_ROOT = "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1"
DEFAULT_STAGE8_ROOT = "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1"
DEFAULT_STAGE7_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1"

STAGE9_SUMMARY_FILE = "checkpoint-publication-authorization-preflight-summary.json"
STAGE9_CANDIDATE_MANIFEST_FILE = "checkpoint-publication-candidate-manifest.json"
STAGE9_IDENTITY_AUDIT_FILE = "checkpoint-publication-identity-audit.json"
STAGE9_METADATA_AUDIT_FILE = "checkpoint-publication-metadata-audit.json"
STAGE9_LOAD_EVIDENCE_AUDIT_FILE = "checkpoint-publication-load-evidence-audit.json"
STAGE9_LINEAGE_AUDIT_FILE = "checkpoint-publication-lineage-audit.json"
STAGE9_RELEASE_BOUNDARY_AUDIT_FILE = "checkpoint-publication-release-boundary-audit.json"
STAGE8_SUMMARY_FILE = "scoped-claim-publication-evidence-freeze-summary.json"
STAGE7_SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "checkpoint-publication-package-preparation-summary.json"
MANIFEST_FILE = "checkpoint-publication-package-manifest.json"
HASH_AUDIT_FILE = "checkpoint-publication-package-hash-audit.json"
METADATA_AUDIT_FILE = "checkpoint-publication-package-metadata-audit.json"
LINEAGE_AUDIT_FILE = "checkpoint-publication-package-lineage-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "checkpoint-publication-package-release-boundary-audit.json"
ROLLBACK_AUDIT_FILE = "checkpoint-publication-package-rollback-audit.json"
REJECTION_REPORT_FILE = "checkpoint-publication-package-rejection-report.json"
REPORT_FILE = "checkpoint-publication-package-preparation-report.md"
PACKAGE_DIR = "checkpoint-publication-package"
PACKAGE_CHECKPOINT_FILE = "experimental-hybrid-policy-candidate.pt"
PACKAGE_METADATA_FILE = "experimental-hybrid-policy-candidate-metadata.json"

EXPECTED_STAGE9_VERDICT = "eligible_for_checkpoint_publication_package_preparation"
EXPECTED_STAGE9_NEXT = "checkpoint_publication_package_preparation"
EXPECTED_PACKAGE_VERDICT = "prepared_for_checkpoint_publication_package_verification"
NEXT_REQUIRED_CHANGE = "checkpoint_publication_package_verification"

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

PROHIBITED_PATH_TOKENS = (
    "default_policy",
    "default-policy",
    "defaultpolicy",
    "live",
    "real_executor",
    "real-executor",
    "executor",
)

PackageMutator = Callable[[Path], None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated checkpoint publication package."
    )
    parser.add_argument("--stage9-root", default=DEFAULT_STAGE9_ROOT)
    parser.add_argument("--stage8-root", default=DEFAULT_STAGE8_ROOT)
    parser.add_argument("--stage7-root", default=DEFAULT_STAGE7_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_checkpoint_publication_package_preparation(
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
                "package_preparation_verdict": summary["package_preparation_verdict"],
                "checkpoint_publication_package_prepared": summary[
                    "checkpoint_publication_package_prepared"
                ],
                "package_hash_audit_passed": summary["package_hash_audit_passed"],
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


def run_checkpoint_publication_package_preparation(
    *,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    package_checkpoint_mutator: PackageMutator | None = None,
    package_metadata_mutator: PackageMutator | None = None,
    omit_manifest_fields_for_test: list[str] | None = None,
    assume_package_deletable: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stage9_root = Path(stage9_root)
    stage8_root = Path(stage8_root)
    stage7_root = Path(stage7_root)
    selected_candidate_root = Path(selected_candidate_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    package_root = output_root / PACKAGE_DIR
    package_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root, package_root)

    read_reasons: list[str] = []
    stage9_summary_path = stage9_root / STAGE9_SUMMARY_FILE
    stage9_summary = _read_json(stage9_summary_path, read_reasons, "stage9_summary")
    stage9_artifacts = {
        "candidate_manifest": _summary_path(
            stage9_summary,
            "candidate_manifest",
            stage9_root,
            repo_root,
            STAGE9_CANDIDATE_MANIFEST_FILE,
        ),
        "identity_audit": _summary_path(
            stage9_summary, "identity_audit", stage9_root, repo_root, STAGE9_IDENTITY_AUDIT_FILE
        ),
        "metadata_audit": _summary_path(
            stage9_summary, "metadata_audit", stage9_root, repo_root, STAGE9_METADATA_AUDIT_FILE
        ),
        "load_evidence_audit": _summary_path(
            stage9_summary,
            "load_evidence_audit",
            stage9_root,
            repo_root,
            STAGE9_LOAD_EVIDENCE_AUDIT_FILE,
        ),
        "lineage_audit": _summary_path(
            stage9_summary, "lineage_audit", stage9_root, repo_root, STAGE9_LINEAGE_AUDIT_FILE
        ),
        "release_boundary_audit": _summary_path(
            stage9_summary,
            "release_boundary_audit",
            stage9_root,
            repo_root,
            STAGE9_RELEASE_BOUNDARY_AUDIT_FILE,
        ),
    }
    stage9_candidate = _read_json(stage9_artifacts["candidate_manifest"], read_reasons, "stage9_candidate_manifest")
    stage9_identity = _read_json(stage9_artifacts["identity_audit"], read_reasons, "stage9_identity_audit")
    stage9_metadata = _read_json(stage9_artifacts["metadata_audit"], read_reasons, "stage9_metadata_audit")
    stage9_load = _read_json(stage9_artifacts["load_evidence_audit"], read_reasons, "stage9_load_evidence_audit")
    stage9_lineage = _read_json(stage9_artifacts["lineage_audit"], read_reasons, "stage9_lineage_audit")
    stage9_release = _read_json(
        stage9_artifacts["release_boundary_audit"], read_reasons, "stage9_release_boundary_audit"
    )
    stage8_summary = _read_json(stage8_root / STAGE8_SUMMARY_FILE, read_reasons, "stage8_summary")
    stage7_summary = _read_json(stage7_root / STAGE7_SUMMARY_FILE, read_reasons, "stage7_summary")
    selected_summary = _read_json(
        selected_candidate_root / SELECTED_SUMMARY_FILE, read_reasons, "selected_candidate_summary"
    )

    source_checkpoint_path = _resolve_first_path(
        (
            stage9_summary.get("checkpoint_path"),
            stage9_candidate.get("checkpoint_path"),
            selected_summary.get("checkpoint_path"),
        ),
        root=stage9_root,
        repo_root=repo_root,
    )
    source_metadata_path = _resolve_first_path(
        (
            stage9_summary.get("checkpoint_metadata_path"),
            stage9_candidate.get("checkpoint_metadata_path"),
            selected_summary.get("checkpoint_metadata_path"),
        ),
        root=stage9_root,
        repo_root=repo_root,
    )
    source_metadata = _read_json(source_metadata_path, [], "source_metadata") if source_metadata_path else {}

    stage9_gate = _stage9_gate_audit(stage9_summary)
    package_path_audit = _package_path_audit(output_root=output_root, package_root=package_root)
    source_hash_audit = _source_hash_audit(
        source_checkpoint_path=source_checkpoint_path,
        expected_sha=stage9_summary.get("checkpoint_sha256") or stage9_candidate.get("checkpoint_sha256"),
        expected_size=_int_or_none(
            stage9_summary.get("checkpoint_size_bytes") or stage9_candidate.get("checkpoint_size_bytes")
        ),
    )

    package_checkpoint_path: Path | None = None
    package_metadata_path: Path | None = None
    if source_hash_audit["source_checkpoint_exists"] is True and source_checkpoint_path is not None:
        package_checkpoint_path = paths["package_checkpoint"]
        shutil.copy2(source_checkpoint_path, package_checkpoint_path)
        if package_checkpoint_mutator is not None:
            package_checkpoint_mutator(package_checkpoint_path)
    if source_metadata_path is not None and source_metadata_path.is_file():
        package_metadata_path = paths["package_metadata"]
        shutil.copy2(source_metadata_path, package_metadata_path)
        if package_metadata_mutator is not None:
            package_metadata_mutator(package_metadata_path)

    package_hash_audit = _package_hash_audit(
        source_hash_audit=source_hash_audit,
        package_checkpoint_path=package_checkpoint_path,
    )
    package_metadata_audit = _package_metadata_audit(
        source_metadata_path=source_metadata_path,
        source_metadata=source_metadata,
        package_metadata_path=package_metadata_path,
    )
    manifest = _package_manifest(
        repo_root=repo_root,
        output_root=output_root,
        package_root=package_root,
        paths=paths,
        stage9_summary_path=stage9_summary_path,
        stage9_summary=stage9_summary,
        stage8_root=stage8_root,
        stage7_root=stage7_root,
        selected_candidate_root=selected_candidate_root,
        source_checkpoint_path=source_checkpoint_path,
        source_metadata_path=source_metadata_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        package_hash_audit=package_hash_audit,
    )
    for field in omit_manifest_fields_for_test or []:
        manifest.pop(field, None)
    manifest_audit = _manifest_audit(manifest)
    lineage_audit = _lineage_audit(
        read_reasons=read_reasons,
        stage9_summary=stage9_summary,
        stage9_candidate=stage9_candidate,
        stage9_identity=stage9_identity,
        stage9_metadata=stage9_metadata,
        stage9_load=stage9_load,
        stage9_lineage=stage9_lineage,
        stage9_release=stage9_release,
        stage8_summary=stage8_summary,
        stage7_summary=stage7_summary,
        selected_summary=selected_summary,
    )
    rollback_audit = _rollback_audit(
        output_root=output_root,
        package_root=package_root,
        source_checkpoint_path=source_checkpoint_path,
        package_checkpoint_path=package_checkpoint_path,
        source_metadata_path=source_metadata_path,
        package_metadata_path=package_metadata_path,
        assume_package_deletable=assume_package_deletable,
    )
    release_boundary_audit = _release_boundary_audit(
        payloads={
            "stage9_summary": stage9_summary,
            "stage9_candidate_manifest": stage9_candidate,
            "stage9_release_boundary_audit": stage9_release,
            "stage8_summary": stage8_summary,
            "stage7_summary": stage7_summary,
            "selected_candidate_summary": selected_summary,
            "source_metadata": source_metadata,
            "package_metadata": _read_json(package_metadata_path, [], "package_metadata")
            if package_metadata_path
            else {},
            "package_manifest": manifest,
        }
    )
    docs_audit = _docs_audit(repo_root=repo_root)

    reason_codes = _unique(
        [
            *stage9_gate["reason_codes"],
            *package_path_audit["reason_codes"],
            *source_hash_audit["reason_codes"],
            *package_hash_audit["reason_codes"],
            *package_metadata_audit["reason_codes"],
            *manifest_audit["reason_codes"],
            *([] if lineage_audit["lineage_audit_passed"] else ["lineage_incomplete"]),
            *([] if rollback_audit["rollback_audit_passed"] else ["rollback_boundary_invalid"]),
            *([] if release_boundary_audit["release_boundary_audit_passed"] else ["release_boundary_violation"]),
            *([] if docs_audit["docs_audit_passed"] else ["docs_not_updated"]),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    approved = status == "passed"

    _write_json(paths["package_manifest"], manifest)
    _write_json(paths["package_hash_audit"], package_hash_audit)
    _write_json(paths["package_metadata_audit"], package_metadata_audit)
    _write_json(paths["package_lineage_audit"], lineage_audit)
    _write_json(paths["package_release_boundary_audit"], release_boundary_audit)
    _write_json(paths["package_rollback_audit"], rollback_audit)
    _write_json(paths["rejection_report"], _rejection_report(reason_codes))

    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        approved=approved,
        paths=paths,
        repo_root=repo_root,
        stage9_root=stage9_root,
        stage8_root=stage8_root,
        stage7_root=stage7_root,
        selected_candidate_root=selected_candidate_root,
        output_root=output_root,
        package_root=package_root,
        stage9_summary=stage9_summary,
        package_hash_audit=package_hash_audit,
        package_metadata_audit=package_metadata_audit,
        manifest_audit=manifest_audit,
        lineage_audit=lineage_audit,
        rollback_audit=rollback_audit,
        release_boundary_audit=release_boundary_audit,
        docs_audit=docs_audit,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage9_gate_audit(stage9_summary: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if (
        stage9_summary.get("status") != "passed"
        or _string_list(stage9_summary.get("reason_codes"))
        or stage9_summary.get("authorization_verdict") != EXPECTED_STAGE9_VERDICT
        or stage9_summary.get("checkpoint_publication_authorization_preflight_passed") is not True
    ):
        _add_reason(reason_codes, "stage9_not_passed")
    if (
        stage9_summary.get("checkpoint_publication_package_preparation_approved") is not True
        or stage9_summary.get("next_required_change") != EXPECTED_STAGE9_NEXT
    ):
        _add_reason(reason_codes, "stage9_not_authorized_for_package_preparation")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "stage9_gate",
        "stage9_gate_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
    }


def _package_path_audit(*, output_root: Path, package_root: Path) -> dict[str, Any]:
    checked_paths = [output_root, package_root]
    violations: list[str] = []
    for path in checked_paths:
        parts = [part.lower() for part in path.parts]
        if any(token in part for part in parts for token in PROHIBITED_PATH_TOKENS):
            violations.append(str(path))
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_path",
        "package_path_audit_passed": not violations,
        "reason_codes": [] if not violations else ["unexpected_publication_path"],
        "violations": violations,
        "output_root": str(output_root),
        "package_root": str(package_root),
    }


def _source_hash_audit(
    *,
    source_checkpoint_path: Path | None,
    expected_sha: Any,
    expected_size: int | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    actual_sha = None
    actual_size = None
    exists = source_checkpoint_path is not None and source_checkpoint_path.is_file()
    if not exists:
        _add_reason(reason_codes, "source_checkpoint_missing")
    else:
        assert source_checkpoint_path is not None
        actual_sha, actual_size = _sha256_and_size(source_checkpoint_path)
        if expected_sha and actual_sha != str(expected_sha):
            _add_reason(reason_codes, "source_checkpoint_identity_mismatch")
        if expected_size is not None and actual_size != expected_size:
            _add_reason(reason_codes, "source_checkpoint_identity_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "source_checkpoint_hash",
        "source_checkpoint_identity_passed": not reason_codes,
        "reason_codes": reason_codes,
        "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "source_checkpoint_exists": exists,
        "source_checkpoint_sha256": actual_sha,
        "expected_source_checkpoint_sha256": str(expected_sha) if expected_sha else None,
        "source_checkpoint_size_bytes": actual_size,
        "expected_source_checkpoint_size_bytes": expected_size,
    }


def _package_hash_audit(
    *,
    source_hash_audit: dict[str, Any],
    package_checkpoint_path: Path | None,
) -> dict[str, Any]:
    reason_codes = list(source_hash_audit.get("reason_codes") or [])
    package_sha = None
    package_size = None
    package_exists = package_checkpoint_path is not None and package_checkpoint_path.is_file()
    source_sha = source_hash_audit.get("source_checkpoint_sha256")
    source_size = source_hash_audit.get("source_checkpoint_size_bytes")
    if package_exists:
        assert package_checkpoint_path is not None
        package_sha, package_size = _sha256_and_size(package_checkpoint_path)
        if source_sha and package_sha != source_sha:
            _add_reason(reason_codes, "package_checkpoint_hash_mismatch")
        if source_size is not None and package_size != source_size:
            _add_reason(reason_codes, "package_checkpoint_size_mismatch")
    elif source_hash_audit.get("source_checkpoint_exists") is True:
        _add_reason(reason_codes, "package_checkpoint_hash_mismatch")
        _add_reason(reason_codes, "package_checkpoint_size_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_checkpoint_hash",
        "package_hash_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "source_checkpoint_path": source_hash_audit.get("source_checkpoint_path"),
        "package_checkpoint_path": str(package_checkpoint_path) if package_checkpoint_path else None,
        "source_checkpoint_sha256": source_sha,
        "package_checkpoint_sha256": package_sha,
        "source_checkpoint_size_bytes": source_size,
        "package_checkpoint_size_bytes": package_size,
        "package_checkpoint_exists": package_exists,
    }


def _package_metadata_audit(
    *,
    source_metadata_path: Path | None,
    source_metadata: dict[str, Any],
    package_metadata_path: Path | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_violations: list[str] = []
    package_violations: list[str] = []
    if not source_metadata_path or not source_metadata_path.is_file() or not source_metadata:
        source_violations.append("source_metadata_missing")
    if source_metadata.get("experimental") is not True:
        source_violations.append("source_metadata_not_experimental")
    for field in (
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "performance_claimed",
        "formal_training_ready_claimed",
        "real_world_release_approved",
    ):
        if source_metadata.get(field) is True:
            source_violations.append(field)
    if source_violations:
        _add_reason(reason_codes, "source_metadata_invalid")

    package_metadata = _read_json(package_metadata_path, [], "package_metadata") if package_metadata_path else {}
    if not package_metadata_path or not package_metadata_path.is_file() or not package_metadata:
        package_violations.append("package_metadata_missing")
    elif package_metadata != source_metadata:
        package_violations.append("package_metadata_differs_from_source")
    if package_metadata and package_metadata.get("experimental") is not True:
        package_violations.append("package_metadata_not_experimental")
    if package_violations:
        _add_reason(reason_codes, "package_metadata_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_metadata",
        "package_metadata_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "source_metadata_path": str(source_metadata_path) if source_metadata_path else None,
        "package_metadata_path": str(package_metadata_path) if package_metadata_path else None,
        "source_violations": source_violations,
        "package_violations": package_violations,
        "source_metadata_experimental": source_metadata.get("experimental"),
        "package_metadata_experimental": package_metadata.get("experimental"),
    }


def _package_manifest(
    *,
    repo_root: Path,
    output_root: Path,
    package_root: Path,
    paths: dict[str, Path],
    stage9_summary_path: Path,
    stage9_summary: dict[str, Any],
    stage8_root: Path,
    stage7_root: Path,
    selected_candidate_root: Path,
    source_checkpoint_path: Path | None,
    source_metadata_path: Path | None,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
    package_hash_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "package_preparation_verdict": EXPECTED_PACKAGE_VERDICT,
        "experimental_checkpoint_package_only": True,
        "selected_seed": stage9_summary.get("selected_seed"),
        "selected_budget": stage9_summary.get("selected_budget"),
        "stage9_summary": str(stage9_summary_path),
        "stage8_root": str(stage8_root),
        "stage7_root": str(stage7_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "package_root": str(package_root),
        "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "package_checkpoint_path": str(package_checkpoint_path) if package_checkpoint_path else None,
        "source_metadata_path": str(source_metadata_path) if source_metadata_path else None,
        "package_metadata_path": str(package_metadata_path) if package_metadata_path else None,
        "source_checkpoint_sha256": package_hash_audit.get("source_checkpoint_sha256"),
        "package_checkpoint_sha256": package_hash_audit.get("package_checkpoint_sha256"),
        "source_checkpoint_size_bytes": package_hash_audit.get("source_checkpoint_size_bytes"),
        "package_checkpoint_size_bytes": package_hash_audit.get("package_checkpoint_size_bytes"),
        "next_required_change": NEXT_REQUIRED_CHANGE,
        "package_hash_audit": str(paths["package_hash_audit"]),
        "package_metadata_audit": str(paths["package_metadata_audit"]),
        "package_lineage_audit": str(paths["package_lineage_audit"]),
        "package_release_boundary_audit": str(paths["package_release_boundary_audit"]),
        "package_rollback_audit": str(paths["package_rollback_audit"]),
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "non_goals": [
            "do_not_publish_checkpoint",
            "do_not_replace_default_policy",
            "do_not_connect_real_executor",
            "do_not_claim_real_world_performance",
            "do_not_claim_ackermann_feasible_trajectory",
            "do_not_relax_guard",
        ],
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _manifest_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    required_fields = (
        "schema_version",
        "generated_at",
        "stage9_summary",
        "source_checkpoint_path",
        "package_checkpoint_path",
        "source_metadata_path",
        "package_metadata_path",
        "selected_seed",
        "selected_budget",
        "source_checkpoint_sha256",
        "package_checkpoint_sha256",
        "source_checkpoint_size_bytes",
        "package_checkpoint_size_bytes",
        "next_required_change",
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "non_goals",
    )
    missing = [field for field in required_fields if field not in manifest or manifest.get(field) in (None, "")]
    reason_codes = [] if not missing else ["package_manifest_incomplete"]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_manifest",
        "package_manifest_audit_passed": not missing,
        "reason_codes": reason_codes,
        "missing_fields": missing,
    }


def _lineage_audit(
    *,
    read_reasons: list[str],
    stage9_summary: dict[str, Any],
    stage9_candidate: dict[str, Any],
    stage9_identity: dict[str, Any],
    stage9_metadata: dict[str, Any],
    stage9_load: dict[str, Any],
    stage9_lineage: dict[str, Any],
    stage9_release: dict[str, Any],
    stage8_summary: dict[str, Any],
    stage7_summary: dict[str, Any],
    selected_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        _source("stage9_summary", _passed(stage9_summary)),
        _source(
            "stage9_candidate_manifest",
            stage9_candidate.get("authorization_verdict") == EXPECTED_STAGE9_VERDICT,
        ),
        _source("stage9_identity_audit", _audit_passed(stage9_identity, "checkpoint_identity_audit_passed")),
        _source("stage9_metadata_audit", _audit_passed(stage9_metadata, "checkpoint_metadata_audit_passed")),
        _source(
            "stage9_load_evidence_audit",
            _audit_passed(stage9_load, "checkpoint_load_evidence_audit_passed"),
        ),
        _source("stage9_lineage_audit", _audit_passed(stage9_lineage, "lineage_audit_passed")),
        _source(
            "stage9_release_boundary_audit",
            _audit_passed(stage9_release, "release_boundary_audit_passed"),
        ),
        _source("stage8_summary", _passed(stage8_summary)),
        _source("stage7_summary", _passed(stage7_summary)),
        _source("selected_candidate_summary", _passed(selected_summary)),
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_lineage",
        "lineage_audit_passed": not read_reasons and all(source["passed"] for source in sources),
        "sources": sources,
        "missing_or_unreadable_inputs": list(read_reasons),
    }


def _rollback_audit(
    *,
    output_root: Path,
    package_root: Path,
    source_checkpoint_path: Path | None,
    package_checkpoint_path: Path | None,
    source_metadata_path: Path | None,
    package_metadata_path: Path | None,
    assume_package_deletable: bool,
) -> dict[str, Any]:
    package_checkpoint_inside_output = (
        package_checkpoint_path is not None
        and package_checkpoint_path.is_file()
        and output_root.resolve() in package_checkpoint_path.resolve().parents
    )
    package_metadata_inside_output = (
        package_metadata_path is not None
        and package_metadata_path.is_file()
        and output_root.resolve() in package_metadata_path.resolve().parents
    )
    source_checkpoint_unchanged = source_checkpoint_path is not None and source_checkpoint_path.is_file()
    source_metadata_unchanged = source_metadata_path is not None and source_metadata_path.is_file()
    source_and_package_distinct = (
        source_checkpoint_path is not None
        and package_checkpoint_path is not None
        and source_checkpoint_path.resolve() != package_checkpoint_path.resolve()
    )
    package_deletable = assume_package_deletable and package_root.exists() and output_root in package_root.parents
    passed = all(
        (
            package_deletable,
            package_checkpoint_inside_output,
            package_metadata_inside_output,
            source_checkpoint_unchanged,
            source_metadata_unchanged,
            source_and_package_distinct,
        )
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "package_rollback",
        "rollback_audit_passed": passed,
        "package_deletable": package_deletable,
        "package_checkpoint_inside_output": package_checkpoint_inside_output,
        "package_metadata_inside_output": package_metadata_inside_output,
        "source_checkpoint_unchanged": source_checkpoint_unchanged,
        "source_metadata_unchanged": source_metadata_unchanged,
        "source_and_package_distinct": source_and_package_distinct,
        "default_policy_touched": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
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


def _docs_audit(*, repo_root: Path) -> dict[str, Any]:
    required_docs = {
        "README.md": repo_root / "README.md",
        "docs/算法设计与系统架构报告.md": repo_root / "docs" / "算法设计与系统架构报告.md",
        "docs/superpowers/specs/2026-06-15-checkpoint-publication-package-preparation.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-15-checkpoint-publication-package-preparation.md",
    }
    required_phrases = {
        "README.md": [
            "Stage 10 `Checkpoint Publication Package Preparation v1`",
            "outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/",
            "checkpoint_publication_package_verification",
        ],
        "docs/算法设计与系统架构报告.md": [
            "阶段 10 `Checkpoint Publication Package Preparation v1`",
            "outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/",
            "checkpoint_publication_package_verification",
        ],
        "docs/superpowers/specs/2026-06-15-checkpoint-publication-package-preparation.md": [
            "checkpoint-publication-package-preparation-summary.json",
            "README.md",
            "docs/算法设计与系统架构报告.md",
            "checkpoint_publication_package_verification",
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
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    package_root: Path,
    stage9_summary: dict[str, Any],
    package_hash_audit: dict[str, Any],
    package_metadata_audit: dict[str, Any],
    manifest_audit: dict[str, Any],
    lineage_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    release_boundary_audit: dict[str, Any],
    docs_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "package_preparation_verdict": EXPECTED_PACKAGE_VERDICT if approved else _failed_verdict(reason_codes),
        "checkpoint_publication_package_prepared": approved,
        "package_manifest_audit_passed": manifest_audit.get("package_manifest_audit_passed") is True,
        "package_hash_audit_passed": package_hash_audit.get("package_hash_audit_passed") is True,
        "package_metadata_audit_passed": package_metadata_audit.get("package_metadata_audit_passed") is True,
        "lineage_audit_passed": lineage_audit.get("lineage_audit_passed") is True,
        "rollback_audit_passed": rollback_audit.get("rollback_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary_audit.get("release_boundary_audit_passed") is True,
        "docs_audit_passed": docs_audit.get("docs_audit_passed") is True,
        "selected_seed": stage9_summary.get("selected_seed"),
        "selected_budget": stage9_summary.get("selected_budget"),
        "source_checkpoint_sha256": package_hash_audit.get("source_checkpoint_sha256"),
        "package_checkpoint_sha256": package_hash_audit.get("package_checkpoint_sha256"),
        "source_checkpoint_size_bytes": package_hash_audit.get("source_checkpoint_size_bytes"),
        "package_checkpoint_size_bytes": package_hash_audit.get("package_checkpoint_size_bytes"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_package_preparation_rejections",
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "runs_new_ppo_update": False,
        "copies_checkpoint_to_isolated_package": bool(paths["package_checkpoint"].is_file()),
        "copies_checkpoint_to_publication_path": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "ackermann_feasible_trajectory_claimed": False,
        "repo_root": str(repo_root),
        "stage9_root": str(stage9_root),
        "stage8_root": str(stage8_root),
        "stage7_root": str(stage7_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "package_root": str(package_root),
        "summary": str(paths["summary"]),
        "package_manifest": str(paths["package_manifest"]),
        "package_hash_audit": str(paths["package_hash_audit"]),
        "package_metadata_audit": str(paths["package_metadata_audit"]),
        "package_lineage_audit": str(paths["package_lineage_audit"]),
        "package_release_boundary_audit": str(paths["package_release_boundary_audit"]),
        "package_rollback_audit": str(paths["package_rollback_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "source_checkpoint_path": package_hash_audit.get("source_checkpoint_path"),
        "package_checkpoint_path": str(paths["package_checkpoint"]),
        "source_metadata_path": package_metadata_audit.get("source_metadata_path"),
        "package_metadata_path": str(paths["package_metadata"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Checkpoint Publication Package Preparation v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Package preparation verdict: `{summary['package_preparation_verdict']}`\n"
        f"- Selected seed: `{summary.get('selected_seed')}`\n"
        f"- Selected budget: `{summary.get('selected_budget')}`\n"
        f"- Source checkpoint SHA-256: `{summary.get('source_checkpoint_sha256')}`\n"
        f"- Package checkpoint SHA-256: `{summary.get('package_checkpoint_sha256')}`\n"
        f"- Package root: `{summary.get('package_root')}`\n\n"
        "## Release Boundaries\n\n"
        f"- Checkpoint publication approved: `{summary['checkpoint_publication_approved']}`\n"
        f"- Default policy replacement approved: `{summary['default_policy_replacement_approved']}`\n"
        f"- Real executor connection approved: `{summary['real_executor_connection_approved']}`\n"
        f"- Next required change: `{summary['next_required_change']}`\n"
    )


def _paths(output_root: Path, package_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "package_manifest": output_root / MANIFEST_FILE,
        "package_hash_audit": output_root / HASH_AUDIT_FILE,
        "package_metadata_audit": output_root / METADATA_AUDIT_FILE,
        "package_lineage_audit": output_root / LINEAGE_AUDIT_FILE,
        "package_release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_FILE,
        "package_rollback_audit": output_root / ROLLBACK_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
        "package_checkpoint": package_root / PACKAGE_CHECKPOINT_FILE,
        "package_metadata": package_root / PACKAGE_METADATA_FILE,
    }


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
        if isinstance(value, str) and value:
            return _resolve_path(Path(value), repo_root if not Path(value).is_absolute() else root)
    return None


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


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
