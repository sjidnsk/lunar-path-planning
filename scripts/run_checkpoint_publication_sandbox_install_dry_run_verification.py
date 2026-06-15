from __future__ import annotations

import argparse
import hashlib
import json
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


SUMMARY_SCHEMA_VERSION = "checkpoint-publication-sandbox-install-dry-run-verification-summary/v1"
AUDIT_SCHEMA_VERSION = "checkpoint-publication-sandbox-install-dry-run-verification-audit/v1"
CONSUMER_MANIFEST_SCHEMA_VERSION = "checkpoint-publication-sandbox-consumer-verification-manifest/v1"

DEFAULT_STAGE13_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_v1"
DEFAULT_STAGE12_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1"
DEFAULT_STAGE11_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_verification_v1"
DEFAULT_STAGE10_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1"
DEFAULT_STAGE9_ROOT = "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1"
DEFAULT_STAGE8_ROOT = "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1"
DEFAULT_STAGE7_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1"

STAGE13_SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-summary.json"
STAGE13_INSTALL_MANIFEST_FILE = "checkpoint-publication-sandbox-install-manifest.json"
STAGE13_COPY_AUDIT_FILE = "checkpoint-publication-sandbox-copy-audit.json"
STAGE13_LOAD_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-load-audit.json"
STAGE13_DEFAULT_POLICY_AUDIT_FILE = "checkpoint-publication-sandbox-default-policy-boundary-audit.json"
STAGE13_PATH_AUDIT_FILE = "checkpoint-publication-sandbox-path-boundary-audit.json"
STAGE13_LINEAGE_AUDIT_FILE = "checkpoint-publication-sandbox-lineage-audit.json"
STAGE13_RELEASE_AUDIT_FILE = "checkpoint-publication-sandbox-release-boundary-audit.json"
STAGE13_ROLLBACK_AUDIT_FILE = "checkpoint-publication-sandbox-rollback-audit.json"
STAGE12_SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-preflight-summary.json"
STAGE12_PREFLIGHT_MANIFEST_FILE = "checkpoint-publication-sandbox-preflight-manifest.json"
STAGE11_CONSUMER_MANIFEST_FILE = "checkpoint-publication-package-consumer-manifest.json"
STAGE11_INTEGRITY_AUDIT_FILE = "checkpoint-publication-package-integrity-audit.json"
STAGE11_LOAD_AUDIT_FILE = "checkpoint-publication-package-load-verification-audit.json"
STAGE11_RELEASE_AUDIT_FILE = "checkpoint-publication-package-release-boundary-audit.json"
STAGE11_ROLLBACK_AUDIT_FILE = "checkpoint-publication-package-rollback-verification-audit.json"
STAGE10_MANIFEST_FILE = "checkpoint-publication-package-manifest.json"
STAGE9_SUMMARY_FILE = "checkpoint-publication-authorization-preflight-summary.json"
STAGE8_SUMMARY_FILE = "scoped-claim-publication-evidence-freeze-summary.json"
STAGE7_SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"

SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-verification-summary.json"
CONSUMER_MANIFEST_FILE = "checkpoint-publication-sandbox-consumer-verification-manifest.json"
INSTALLED_FILE_IDENTITY_AUDIT_FILE = "checkpoint-publication-sandbox-installed-file-identity-audit.json"
INSTALL_MANIFEST_CONSISTENCY_AUDIT_FILE = "checkpoint-publication-sandbox-install-manifest-consistency-audit.json"
LOAD_REVERIFICATION_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-load-reverification-audit.json"
METADATA_VERIFICATION_AUDIT_FILE = "checkpoint-publication-sandbox-metadata-verification-audit.json"
LINEAGE_VERIFICATION_AUDIT_FILE = "checkpoint-publication-sandbox-lineage-verification-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "checkpoint-publication-sandbox-release-boundary-audit.json"
ROLLBACK_VERIFICATION_AUDIT_FILE = "checkpoint-publication-sandbox-rollback-verification-audit.json"
REJECTION_REPORT_FILE = "checkpoint-publication-sandbox-install-dry-run-verification-rejection-report.json"
REPORT_FILE = "checkpoint-publication-sandbox-install-dry-run-verification-report.md"

EXPECTED_STAGE13_VERDICT = "installed_in_sandbox_for_checkpoint_publication_sandbox_install_dry_run_verification"
EXPECTED_STAGE13_NEXT = "checkpoint_publication_sandbox_install_dry_run_verification"
EXPECTED_VERIFICATION_VERDICT = "verified_for_checkpoint_publication_sandbox_consumer_smoke_preflight"
NEXT_REQUIRED_CHANGE = "checkpoint_publication_sandbox_consumer_smoke_preflight"

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
    "ackermann_feasible_trajectory",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Stage 13 sandbox install dry-run without mutating installed files."
    )
    parser.add_argument("--stage13-root", default=DEFAULT_STAGE13_ROOT)
    parser.add_argument("--stage12-root", default=DEFAULT_STAGE12_ROOT)
    parser.add_argument("--stage11-root", default=DEFAULT_STAGE11_ROOT)
    parser.add_argument("--stage10-root", default=DEFAULT_STAGE10_ROOT)
    parser.add_argument("--stage9-root", default=DEFAULT_STAGE9_ROOT)
    parser.add_argument("--stage8-root", default=DEFAULT_STAGE8_ROOT)
    parser.add_argument("--stage7-root", default=DEFAULT_STAGE7_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--default-policy-path")
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    default_policy_path = (
        _resolve_path(Path(args.default_policy_path), repo_root) if args.default_policy_path else None
    )
    summary = run_checkpoint_publication_sandbox_install_dry_run_verification(
        stage13_root=_resolve_path(Path(args.stage13_root), repo_root),
        stage12_root=_resolve_path(Path(args.stage12_root), repo_root),
        stage11_root=_resolve_path(Path(args.stage11_root), repo_root),
        stage10_root=_resolve_path(Path(args.stage10_root), repo_root),
        stage9_root=_resolve_path(Path(args.stage9_root), repo_root),
        stage8_root=_resolve_path(Path(args.stage8_root), repo_root),
        stage7_root=_resolve_path(Path(args.stage7_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        default_policy_path=default_policy_path,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "verification_verdict": summary["verification_verdict"],
                "checkpoint_publication_sandbox_install_dry_run_verification_passed": summary[
                    "checkpoint_publication_sandbox_install_dry_run_verification_passed"
                ],
                "checkpoint_publication_sandbox_consumer_smoke_preflight_approved": summary[
                    "checkpoint_publication_sandbox_consumer_smoke_preflight_approved"
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


def run_checkpoint_publication_sandbox_install_dry_run_verification(
    *,
    stage13_root: Path,
    stage12_root: Path,
    stage11_root: Path,
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    output_root: Path,
    repo_root: Path,
    default_policy_path: Path | None = None,
    default_policy_mutator: Callable[[Path], None] | None = None,
    assume_rollback_valid: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stage13_root = Path(stage13_root)
    stage12_root = Path(stage12_root)
    stage11_root = Path(stage11_root)
    stage10_root = Path(stage10_root)
    stage9_root = Path(stage9_root)
    stage8_root = Path(stage8_root)
    stage7_root = Path(stage7_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    read_reasons: list[str] = []
    stage13_summary_path = stage13_root / STAGE13_SUMMARY_FILE
    stage13_summary = _read_json(stage13_summary_path, read_reasons, "stage13_summary")
    install_manifest_path = _summary_path(
        stage13_summary,
        "sandbox_install_manifest",
        stage13_root,
        repo_root,
        STAGE13_INSTALL_MANIFEST_FILE,
    )
    copy_audit_path = _summary_path(
        stage13_summary, "sandbox_copy_audit", stage13_root, repo_root, STAGE13_COPY_AUDIT_FILE
    )
    load_audit_path = _summary_path(
        stage13_summary,
        "sandbox_consumer_load_audit",
        stage13_root,
        repo_root,
        STAGE13_LOAD_AUDIT_FILE,
    )
    default_policy_audit_path = _summary_path(
        stage13_summary,
        "default_policy_boundary_audit",
        stage13_root,
        repo_root,
        STAGE13_DEFAULT_POLICY_AUDIT_FILE,
    )
    path_audit_path = _summary_path(
        stage13_summary,
        "sandbox_path_boundary_audit",
        stage13_root,
        repo_root,
        STAGE13_PATH_AUDIT_FILE,
    )
    lineage_audit_path = _summary_path(
        stage13_summary,
        "sandbox_lineage_audit",
        stage13_root,
        repo_root,
        STAGE13_LINEAGE_AUDIT_FILE,
    )
    release_audit_path = _summary_path(
        stage13_summary,
        "sandbox_release_boundary_audit",
        stage13_root,
        repo_root,
        STAGE13_RELEASE_AUDIT_FILE,
    )
    rollback_audit_path = _summary_path(
        stage13_summary, "rollback_audit", stage13_root, repo_root, STAGE13_ROLLBACK_AUDIT_FILE
    )
    install_manifest = _read_json(install_manifest_path, read_reasons, "stage13_install_manifest")
    copy_audit = _read_json(copy_audit_path, read_reasons, "stage13_copy_audit")
    load_audit = _read_json(load_audit_path, read_reasons, "stage13_load_audit")
    default_policy_audit = _read_json(default_policy_audit_path, read_reasons, "stage13_default_policy_audit")
    path_audit = _read_json(path_audit_path, read_reasons, "stage13_path_audit")
    stage13_lineage_audit = _read_json(lineage_audit_path, read_reasons, "stage13_lineage_audit")
    stage13_release_audit = _read_json(release_audit_path, read_reasons, "stage13_release_audit")
    stage13_rollback_audit = _read_json(rollback_audit_path, read_reasons, "stage13_rollback_audit")

    stage12_summary = _read_json(stage12_root / STAGE12_SUMMARY_FILE, read_reasons, "stage12_summary")
    stage12_manifest = _read_json(stage12_root / STAGE12_PREFLIGHT_MANIFEST_FILE, read_reasons, "stage12_manifest")
    stage11_consumer = _read_json(stage11_root / STAGE11_CONSUMER_MANIFEST_FILE, read_reasons, "stage11_consumer")
    stage11_integrity = _read_json(stage11_root / STAGE11_INTEGRITY_AUDIT_FILE, read_reasons, "stage11_integrity")
    stage11_load = _read_json(stage11_root / STAGE11_LOAD_AUDIT_FILE, read_reasons, "stage11_load")
    stage11_release = _read_json(stage11_root / STAGE11_RELEASE_AUDIT_FILE, read_reasons, "stage11_release")
    stage11_rollback = _read_json(stage11_root / STAGE11_ROLLBACK_AUDIT_FILE, read_reasons, "stage11_rollback")
    stage10_manifest = _read_json(stage10_root / STAGE10_MANIFEST_FILE, read_reasons, "stage10_manifest")
    stage9_summary = _read_json(stage9_root / STAGE9_SUMMARY_FILE, read_reasons, "stage9_summary")
    stage8_summary = _read_json(stage8_root / STAGE8_SUMMARY_FILE, read_reasons, "stage8_summary")
    stage7_summary = _read_json(stage7_root / STAGE7_SUMMARY_FILE, read_reasons, "stage7_summary")

    source_checkpoint_path = _resolve_first_path(
        (
            install_manifest.get("source_package_checkpoint_path"),
            stage12_manifest.get("source_package_checkpoint_path"),
            stage11_consumer.get("authoritative_package_checkpoint_path"),
            stage10_manifest.get("package_checkpoint_path"),
        ),
        root=stage13_root,
        repo_root=repo_root,
    )
    source_metadata_path = _resolve_first_path(
        (
            install_manifest.get("source_package_metadata_path"),
            stage12_manifest.get("source_package_metadata_path"),
            stage11_consumer.get("package_metadata_path"),
            stage10_manifest.get("package_metadata_path"),
        ),
        root=stage13_root,
        repo_root=repo_root,
    )
    sandbox_checkpoint_path = _resolve_first_path(
        (
            install_manifest.get("sandbox_consumer_checkpoint_path"),
            stage13_summary.get("sandbox_consumer_checkpoint_path"),
            stage12_manifest.get("planned_consumer_checkpoint_path"),
        ),
        root=stage13_root,
        repo_root=repo_root,
    )
    sandbox_metadata_path = _resolve_first_path(
        (
            install_manifest.get("sandbox_consumer_metadata_path"),
            stage12_manifest.get("planned_consumer_metadata_path"),
        ),
        root=stage13_root,
        repo_root=repo_root,
    )
    sandbox_root = _resolve_optional_path(stage12_manifest.get("sandbox_root"), stage12_root, repo_root)
    source_before = _file_snapshot(source_checkpoint_path)
    source_metadata_before = _file_snapshot(source_metadata_path)
    sandbox_before = _file_snapshot(sandbox_checkpoint_path)
    sandbox_metadata_before = _file_snapshot(sandbox_metadata_path)
    default_before = _file_snapshot(default_policy_path)

    stage13_gate = _stage13_gate_audit(stage13_summary)
    identity_audit = _installed_file_identity_audit(
        source_checkpoint_path=source_checkpoint_path,
        source_metadata_path=source_metadata_path,
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        stage13_summary=stage13_summary,
        install_manifest=install_manifest,
        copy_audit=copy_audit,
        load_audit=load_audit,
        stage12_manifest=stage12_manifest,
        stage11_consumer=stage11_consumer,
        stage10_manifest=stage10_manifest,
    )
    manifest_consistency_audit = _install_manifest_consistency_audit(
        install_manifest_path=install_manifest_path,
        install_manifest=install_manifest,
        stage13_summary=stage13_summary,
        identity_audit=identity_audit,
        source_checkpoint_path=source_checkpoint_path,
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        source_metadata_path=source_metadata_path,
        sandbox_metadata_path=sandbox_metadata_path,
    )
    load_reverification_audit = _load_reverification_audit(
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        load_audit=load_audit,
        identity_audit=identity_audit,
    )
    metadata_verification_audit = _metadata_verification_audit(
        sandbox_metadata_path=sandbox_metadata_path,
        source_metadata_path=source_metadata_path,
    )
    if default_policy_mutator is not None and default_policy_path is not None:
        default_policy_mutator(default_policy_path)
    default_policy_boundary_audit = _default_policy_boundary_audit(
        default_policy_path=default_policy_path,
        before=default_before,
        after=_file_snapshot(default_policy_path),
    )
    rollback_verification_audit = _rollback_verification_audit(
        source_before=source_before,
        source_after=_file_snapshot(source_checkpoint_path),
        source_metadata_before=source_metadata_before,
        source_metadata_after=_file_snapshot(source_metadata_path),
        sandbox_before=sandbox_before,
        sandbox_after=_file_snapshot(sandbox_checkpoint_path),
        sandbox_metadata_before=sandbox_metadata_before,
        sandbox_metadata_after=_file_snapshot(sandbox_metadata_path),
        default_policy_boundary_audit=default_policy_boundary_audit,
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        sandbox_root=sandbox_root,
        identity_audit=identity_audit,
        assume_rollback_valid=assume_rollback_valid,
    )
    lineage_verification_audit = _lineage_verification_audit(
        read_reasons=read_reasons,
        stage13_summary=stage13_summary,
        install_manifest=install_manifest,
        copy_audit=copy_audit,
        load_audit=load_audit,
        default_policy_audit=default_policy_audit,
        path_audit=path_audit,
        stage13_lineage_audit=stage13_lineage_audit,
        stage13_release_audit=stage13_release_audit,
        stage13_rollback_audit=stage13_rollback_audit,
        stage12_summary=stage12_summary,
        stage12_manifest=stage12_manifest,
        stage11_consumer=stage11_consumer,
        stage11_integrity=stage11_integrity,
        stage11_load=stage11_load,
        stage11_release=stage11_release,
        stage11_rollback=stage11_rollback,
        stage10_manifest=stage10_manifest,
        stage9_summary=stage9_summary,
        stage8_summary=stage8_summary,
        stage7_summary=stage7_summary,
    )
    release_boundary_audit = _release_boundary_audit(
        payloads={
            "stage13_summary": stage13_summary,
            "install_manifest": install_manifest,
            "copy_audit": copy_audit,
            "load_audit": load_audit,
            "stage13_default_policy_audit": default_policy_audit,
            "stage13_path_audit": path_audit,
            "stage13_lineage_audit": stage13_lineage_audit,
            "stage13_release_audit": stage13_release_audit,
            "stage13_rollback_audit": stage13_rollback_audit,
            "stage12_summary": stage12_summary,
            "stage12_manifest": stage12_manifest,
            "stage11_consumer": stage11_consumer,
            "stage11_integrity": stage11_integrity,
            "stage11_load": stage11_load,
            "stage11_release": stage11_release,
            "stage11_rollback": stage11_rollback,
            "stage10_manifest": stage10_manifest,
            "stage9_summary": stage9_summary,
            "stage8_summary": stage8_summary,
            "stage7_summary": stage7_summary,
            "identity_audit": identity_audit,
            "manifest_consistency_audit": manifest_consistency_audit,
            "load_reverification_audit": load_reverification_audit,
            "metadata_verification_audit": metadata_verification_audit,
            "default_policy_boundary_audit": default_policy_boundary_audit,
            "rollback_verification_audit": rollback_verification_audit,
        }
    )
    docs_audit = _docs_audit(repo_root=repo_root)
    consumer_manifest = _consumer_manifest(
        approved=False,
        stage13_summary_path=stage13_summary_path,
        install_manifest_path=install_manifest_path,
        source_checkpoint_path=source_checkpoint_path,
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        identity_audit=identity_audit,
        load_reverification_audit=load_reverification_audit,
        repo_root=repo_root,
    )

    reason_codes = _unique(
        [
            *stage13_gate["reason_codes"],
            *identity_audit["reason_codes"],
            *manifest_consistency_audit["reason_codes"],
            *load_reverification_audit["reason_codes"],
            *metadata_verification_audit["reason_codes"],
            *default_policy_boundary_audit["reason_codes"],
            *([] if lineage_verification_audit["lineage_verification_audit_passed"] else ["lineage_verification_incomplete"]),
            *([] if rollback_verification_audit["rollback_verification_audit_passed"] else ["rollback_boundary_invalid"]),
            *([] if release_boundary_audit["release_boundary_audit_passed"] else ["release_boundary_violation"]),
            *([] if docs_audit["docs_audit_passed"] else ["docs_not_updated"]),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    approved = status == "passed"
    consumer_manifest["verification_verdict"] = EXPECTED_VERIFICATION_VERDICT if approved else "not_verified"
    consumer_manifest["next_required_change"] = (
        NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_sandbox_install_dry_run_verification_rejections"
    )

    _write_json(paths["consumer_verification_manifest"], consumer_manifest)
    _write_json(paths["installed_file_identity_audit"], identity_audit)
    _write_json(paths["install_manifest_consistency_audit"], manifest_consistency_audit)
    _write_json(paths["consumer_load_reverification_audit"], load_reverification_audit)
    _write_json(paths["metadata_verification_audit"], metadata_verification_audit)
    _write_json(paths["lineage_verification_audit"], lineage_verification_audit)
    _write_json(paths["sandbox_release_boundary_audit"], release_boundary_audit)
    _write_json(paths["rollback_verification_audit"], rollback_verification_audit)
    _write_json(paths["rejection_report"], _rejection_report(reason_codes))
    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        approved=approved,
        paths=paths,
        repo_root=repo_root,
        stage13_root=stage13_root,
        stage12_root=stage12_root,
        stage11_root=stage11_root,
        stage10_root=stage10_root,
        stage9_root=stage9_root,
        stage8_root=stage8_root,
        stage7_root=stage7_root,
        output_root=output_root,
        stage13_summary=stage13_summary,
        identity_audit=identity_audit,
        manifest_consistency_audit=manifest_consistency_audit,
        load_reverification_audit=load_reverification_audit,
        metadata_verification_audit=metadata_verification_audit,
        lineage_verification_audit=lineage_verification_audit,
        rollback_verification_audit=rollback_verification_audit,
        release_boundary_audit=release_boundary_audit,
        docs_audit=docs_audit,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage13_gate_audit(stage13_summary: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if (
        stage13_summary.get("status") != "passed"
        or _string_list(stage13_summary.get("reason_codes"))
        or stage13_summary.get("checkpoint_publication_sandbox_install_dry_run_passed") is not True
    ):
        _add_reason(reason_codes, "stage13_not_passed")
    if (
        stage13_summary.get("install_dry_run_verdict") != EXPECTED_STAGE13_VERDICT
        or stage13_summary.get("checkpoint_publication_sandbox_install_dry_run_verification_approved") is not True
        or stage13_summary.get("next_required_change") != EXPECTED_STAGE13_NEXT
    ):
        _add_reason(reason_codes, "stage13_not_ready_for_verification")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "stage13_gate",
        "stage13_gate_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
    }


def _installed_file_identity_audit(
    *,
    source_checkpoint_path: Path | None,
    source_metadata_path: Path | None,
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    stage13_summary: dict[str, Any],
    install_manifest: dict[str, Any],
    copy_audit: dict[str, Any],
    load_audit: dict[str, Any],
    stage12_manifest: dict[str, Any],
    stage11_consumer: dict[str, Any],
    stage10_manifest: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_sha = source_size = sandbox_sha = sandbox_size = None
    source_metadata_sha = sandbox_metadata_sha = None
    source_metadata_size = sandbox_metadata_size = None
    if source_checkpoint_path is None or not source_checkpoint_path.is_file():
        _add_reason(reason_codes, "source_package_checkpoint_missing")
    else:
        source_sha, source_size = _sha256_and_size(source_checkpoint_path)
    if sandbox_checkpoint_path is None or not sandbox_checkpoint_path.is_file():
        _add_reason(reason_codes, "sandbox_checkpoint_missing")
    else:
        sandbox_sha, sandbox_size = _sha256_and_size(sandbox_checkpoint_path)
    if source_metadata_path is None or not source_metadata_path.is_file():
        _add_reason(reason_codes, "source_package_metadata_missing")
    else:
        source_metadata_sha, source_metadata_size = _sha256_and_size(source_metadata_path)
    if sandbox_metadata_path is None or not sandbox_metadata_path.is_file():
        _add_reason(reason_codes, "sandbox_metadata_missing")
    else:
        sandbox_metadata_sha, sandbox_metadata_size = _sha256_and_size(sandbox_metadata_path)

    source_expected_shas = _unique(
        [
            str(value)
            for value in (
                stage13_summary.get("source_package_checkpoint_sha256"),
                install_manifest.get("source_package_checkpoint_sha256"),
                copy_audit.get("source_package_checkpoint_sha256"),
                stage12_manifest.get("package_checkpoint_sha256"),
                stage11_consumer.get("package_checkpoint_sha256"),
                stage10_manifest.get("package_checkpoint_sha256"),
            )
            if value
        ]
    )
    sandbox_expected_shas = _unique(
        [
            str(value)
            for value in (
                stage13_summary.get("sandbox_consumer_checkpoint_sha256"),
                install_manifest.get("sandbox_consumer_checkpoint_sha256"),
                copy_audit.get("sandbox_consumer_checkpoint_sha256"),
                load_audit.get("sandbox_consumer_checkpoint_sha256"),
                stage12_manifest.get("package_checkpoint_sha256"),
                stage11_consumer.get("package_checkpoint_sha256"),
            )
            if value
        ]
    )
    source_expected_sizes = _unique_ints(
        (
            stage13_summary.get("source_package_checkpoint_size_bytes"),
            install_manifest.get("source_package_checkpoint_size_bytes"),
            copy_audit.get("source_package_checkpoint_size_bytes"),
            stage12_manifest.get("package_checkpoint_size_bytes"),
            stage11_consumer.get("package_checkpoint_size_bytes"),
            stage10_manifest.get("package_checkpoint_size_bytes"),
        )
    )
    sandbox_expected_sizes = _unique_ints(
        (
            stage13_summary.get("sandbox_consumer_checkpoint_size_bytes"),
            install_manifest.get("sandbox_consumer_checkpoint_size_bytes"),
            copy_audit.get("sandbox_consumer_checkpoint_size_bytes"),
            load_audit.get("sandbox_consumer_checkpoint_size_bytes"),
            stage12_manifest.get("package_checkpoint_size_bytes"),
            stage11_consumer.get("package_checkpoint_size_bytes"),
        )
    )
    if source_sha and (
        any(source_sha != expected for expected in source_expected_shas)
        or any(source_size != expected for expected in source_expected_sizes)
    ):
        _add_reason(reason_codes, "source_package_identity_mismatch")
    if sandbox_sha and (
        any(sandbox_sha != expected for expected in sandbox_expected_shas)
        or any(sandbox_size != expected for expected in sandbox_expected_sizes)
    ):
        _add_reason(reason_codes, "sandbox_identity_mismatch")
    if source_sha and sandbox_sha and (source_sha != sandbox_sha or source_size != sandbox_size):
        _add_reason(reason_codes, "sandbox_identity_mismatch")
    if source_metadata_sha and sandbox_metadata_sha and (
        source_metadata_sha != sandbox_metadata_sha or source_metadata_size != sandbox_metadata_size
    ):
        _add_reason(reason_codes, "sandbox_identity_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_installed_file_identity",
        "sandbox_installed_file_identity_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "source_package_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "source_package_metadata_path": str(source_metadata_path) if source_metadata_path else None,
        "sandbox_consumer_metadata_path": str(sandbox_metadata_path) if sandbox_metadata_path else None,
        "source_package_checkpoint_sha256": source_sha,
        "sandbox_consumer_checkpoint_sha256": sandbox_sha,
        "source_package_checkpoint_size_bytes": source_size,
        "sandbox_consumer_checkpoint_size_bytes": sandbox_size,
        "source_package_metadata_sha256": source_metadata_sha,
        "sandbox_consumer_metadata_sha256": sandbox_metadata_sha,
    }


def _install_manifest_consistency_audit(
    *,
    install_manifest_path: Path,
    install_manifest: dict[str, Any],
    stage13_summary: dict[str, Any],
    identity_audit: dict[str, Any],
    source_checkpoint_path: Path | None,
    sandbox_checkpoint_path: Path | None,
    source_metadata_path: Path | None,
    sandbox_metadata_path: Path | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    violations: list[str] = []
    if not install_manifest_path.is_file() or not install_manifest:
        _add_reason(reason_codes, "sandbox_install_manifest_missing")
        violations.append("manifest_missing")
    required_fields = (
        "install_dry_run_verdict",
        "sandbox_install_manifest_passed",
        "source_package_checkpoint_path",
        "sandbox_consumer_checkpoint_path",
        "source_package_checkpoint_sha256",
        "sandbox_consumer_checkpoint_sha256",
        "source_package_checkpoint_size_bytes",
        "sandbox_consumer_checkpoint_size_bytes",
        "next_required_change",
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
    )
    for field in required_fields:
        if field not in install_manifest:
            violations.append(f"missing_{field}")
    if install_manifest.get("install_dry_run_verdict") != EXPECTED_STAGE13_VERDICT:
        violations.append("install_dry_run_verdict_mismatch")
    if install_manifest.get("next_required_change") != EXPECTED_STAGE13_NEXT:
        violations.append("next_required_change_mismatch")
    if install_manifest.get("sandbox_install_manifest_passed") is not True:
        violations.append("sandbox_install_manifest_not_passed")
    if source_checkpoint_path and _same_path(install_manifest.get("source_package_checkpoint_path"), source_checkpoint_path) is False:
        violations.append("source_checkpoint_path_mismatch")
    if sandbox_checkpoint_path and _same_path(install_manifest.get("sandbox_consumer_checkpoint_path"), sandbox_checkpoint_path) is False:
        violations.append("sandbox_checkpoint_path_mismatch")
    if source_metadata_path and _same_path(install_manifest.get("source_package_metadata_path"), source_metadata_path) is False:
        violations.append("source_metadata_path_mismatch")
    if sandbox_metadata_path and _same_path(install_manifest.get("sandbox_consumer_metadata_path"), sandbox_metadata_path) is False:
        violations.append("sandbox_metadata_path_mismatch")
    if install_manifest.get("source_package_checkpoint_sha256") != identity_audit.get("source_package_checkpoint_sha256"):
        violations.append("source_sha_mismatch")
    if install_manifest.get("sandbox_consumer_checkpoint_sha256") != identity_audit.get("sandbox_consumer_checkpoint_sha256"):
        violations.append("sandbox_sha_mismatch")
    if _int_or_none(install_manifest.get("source_package_checkpoint_size_bytes")) != identity_audit.get("source_package_checkpoint_size_bytes"):
        violations.append("source_size_mismatch")
    if _int_or_none(install_manifest.get("sandbox_consumer_checkpoint_size_bytes")) != identity_audit.get("sandbox_consumer_checkpoint_size_bytes"):
        violations.append("sandbox_size_mismatch")
    if any(install_manifest.get(field) is True for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor")):
        violations.append("release_boundary_true")
    if violations and "sandbox_install_manifest_missing" not in reason_codes:
        _add_reason(reason_codes, "sandbox_install_manifest_inconsistent")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_install_manifest_consistency",
        "sandbox_install_manifest_consistency_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "install_manifest_path": str(install_manifest_path),
        "violations": violations,
    }


def _load_reverification_audit(
    *,
    sandbox_checkpoint_path: Path | None,
    load_audit: dict[str, Any],
    identity_audit: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    checkpoint_type = None
    tensor_count = 0
    state_key_count = 0
    non_finite_tensor_count = 0
    load_error = None
    try:
        if sandbox_checkpoint_path is None or not sandbox_checkpoint_path.is_file():
            raise FileNotFoundError("sandbox checkpoint missing")
        import torch

        checkpoint = torch.load(sandbox_checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_type = type(checkpoint).__name__
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint is not a dict")
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
        prior_tensor_count = _int_or_none(load_audit.get("tensor_count"))
        prior_non_finite = _int_or_none(load_audit.get("non_finite_tensor_count"))
        if prior_tensor_count is not None and prior_tensor_count != tensor_count:
            raise ValueError("tensor count mismatch with Stage 13 load audit")
        if prior_non_finite is not None and prior_non_finite != non_finite_tensor_count:
            raise ValueError("non-finite tensor count mismatch with Stage 13 load audit")
    except Exception as exc:  # noqa: BLE001 - audit records load failures
        load_error = str(exc)
        _add_reason(reason_codes, "sandbox_load_reverification_failed")
    if (
        not reason_codes
        and identity_audit.get("sandbox_consumer_checkpoint_sha256") != identity_audit.get("source_package_checkpoint_sha256")
    ):
        _add_reason(reason_codes, "sandbox_identity_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_consumer_load_reverification",
        "sandbox_consumer_load_reverification_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "checkpoint_type": checkpoint_type,
        "state_key_count": state_key_count,
        "tensor_count": tensor_count,
        "non_finite_tensor_count": non_finite_tensor_count,
        "load_error": load_error,
        "sandbox_consumer_checkpoint_sha256": identity_audit.get("sandbox_consumer_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_size_bytes": identity_audit.get("sandbox_consumer_checkpoint_size_bytes"),
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _metadata_verification_audit(*, sandbox_metadata_path: Path | None, source_metadata_path: Path | None) -> dict[str, Any]:
    reason_codes: list[str] = []
    violations: list[str] = []
    metadata = _read_json(sandbox_metadata_path, [], "sandbox_metadata") if sandbox_metadata_path else {}
    if sandbox_metadata_path is None or not sandbox_metadata_path.is_file() or not metadata:
        _add_reason(reason_codes, "sandbox_metadata_missing")
        violations.append("sandbox_metadata_missing")
    if source_metadata_path is None or not source_metadata_path.is_file():
        _add_reason(reason_codes, "source_package_metadata_missing")
        violations.append("source_metadata_missing")
    if metadata and metadata.get("experimental") is not True:
        violations.append("metadata_not_experimental")
    for field in (
        *RELEASE_BOUNDARY_FIELDS,
        "performance_claimed",
        "formal_training_ready_claimed",
        "default_policy_replaced",
    ):
        if metadata.get(field) is True:
            violations.append(field)
    if violations and not any(
        reason in reason_codes for reason in ("sandbox_metadata_missing", "source_package_metadata_missing")
    ):
        _add_reason(reason_codes, "sandbox_metadata_invalid")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_metadata_verification",
        "sandbox_metadata_verification_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "sandbox_metadata_path": str(sandbox_metadata_path) if sandbox_metadata_path else None,
        "source_metadata_path": str(source_metadata_path) if source_metadata_path else None,
        "metadata_experimental": metadata.get("experimental"),
        "violations": violations,
    }


def _default_policy_boundary_audit(
    *,
    default_policy_path: Path | None,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    configured = default_policy_path is not None
    violations: list[str] = []
    if configured and not before.get("exists"):
        violations.append("configured_default_policy_missing")
    if configured and not _snapshots_equal(before, after):
        violations.append("default_policy_changed")
    reason_codes = ["default_policy_boundary_violation"] if violations else []
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_default_policy_boundary",
        "default_policy_boundary_audit_passed": not violations,
        "reason_codes": reason_codes,
        "default_policy_configured": configured,
        "default_policy_path": str(default_policy_path) if default_policy_path else None,
        "before": before,
        "after": after,
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
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    source_metadata_before: dict[str, Any],
    source_metadata_after: dict[str, Any],
    sandbox_before: dict[str, Any],
    sandbox_after: dict[str, Any],
    sandbox_metadata_before: dict[str, Any],
    sandbox_metadata_after: dict[str, Any],
    default_policy_boundary_audit: dict[str, Any],
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    sandbox_root: Path | None,
    identity_audit: dict[str, Any],
    assume_rollback_valid: bool,
) -> dict[str, Any]:
    source_unchanged = _snapshots_equal(source_before, source_after)
    source_metadata_unchanged = _snapshots_equal(source_metadata_before, source_metadata_after)
    sandbox_unchanged = _snapshots_equal(sandbox_before, sandbox_after)
    sandbox_metadata_unchanged = _snapshots_equal(sandbox_metadata_before, sandbox_metadata_after)
    default_policy_unchanged = default_policy_boundary_audit.get("default_policy_boundary_audit_passed") is True
    sandbox_in_planned_root = bool(
        sandbox_root
        and sandbox_checkpoint_path
        and sandbox_metadata_path
        and _is_relative_to(sandbox_checkpoint_path, sandbox_root)
        and _is_relative_to(sandbox_metadata_path, sandbox_root)
    )
    sandbox_traceable = (
        identity_audit.get("source_package_checkpoint_sha256")
        == identity_audit.get("sandbox_consumer_checkpoint_sha256")
        and identity_audit.get("source_package_checkpoint_sha256") is not None
    )
    passed = (
        assume_rollback_valid
        and source_unchanged
        and source_metadata_unchanged
        and sandbox_unchanged
        and sandbox_metadata_unchanged
        and default_policy_unchanged
        and sandbox_in_planned_root
        and sandbox_traceable
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_rollback_verification",
        "rollback_verification_audit_passed": passed,
        "source_checkpoint_unchanged": source_unchanged,
        "source_metadata_unchanged": source_metadata_unchanged,
        "sandbox_checkpoint_unchanged": sandbox_unchanged,
        "sandbox_metadata_unchanged": sandbox_metadata_unchanged,
        "default_policy_unchanged": default_policy_unchanged,
        "sandbox_in_planned_root": sandbox_in_planned_root,
        "sandbox_checkpoint_traceable": sandbox_traceable,
        "assume_rollback_valid": assume_rollback_valid,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _lineage_verification_audit(
    *,
    read_reasons: list[str],
    stage13_summary: dict[str, Any],
    install_manifest: dict[str, Any],
    copy_audit: dict[str, Any],
    load_audit: dict[str, Any],
    default_policy_audit: dict[str, Any],
    path_audit: dict[str, Any],
    stage13_lineage_audit: dict[str, Any],
    stage13_release_audit: dict[str, Any],
    stage13_rollback_audit: dict[str, Any],
    stage12_summary: dict[str, Any],
    stage12_manifest: dict[str, Any],
    stage11_consumer: dict[str, Any],
    stage11_integrity: dict[str, Any],
    stage11_load: dict[str, Any],
    stage11_release: dict[str, Any],
    stage11_rollback: dict[str, Any],
    stage10_manifest: dict[str, Any],
    stage9_summary: dict[str, Any],
    stage8_summary: dict[str, Any],
    stage7_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        _source("stage13_summary", _passed(stage13_summary)),
        _source("stage13_install_manifest", install_manifest.get("sandbox_install_manifest_passed") is True),
        _source("stage13_copy_audit", _audit_passed(copy_audit, "sandbox_copy_audit_passed")),
        _source("stage13_load_audit", _audit_passed(load_audit, "sandbox_consumer_load_audit_passed")),
        _source("stage13_default_policy_audit", _audit_passed(default_policy_audit, "default_policy_boundary_audit_passed")),
        _source("stage13_path_audit", _audit_passed(path_audit, "sandbox_path_boundary_audit_passed")),
        _source("stage13_lineage_audit", _audit_passed(stage13_lineage_audit, "lineage_audit_passed")),
        _source("stage13_release_audit", _audit_passed(stage13_release_audit, "release_boundary_audit_passed")),
        _source("stage13_rollback_audit", _audit_passed(stage13_rollback_audit, "rollback_audit_passed")),
        _source("stage12_summary", _passed(stage12_summary)),
        _source("stage12_manifest", stage12_manifest.get("sandbox_preflight_manifest_passed") is True),
        _source("stage11_consumer", bool(stage11_consumer)),
        _source("stage11_integrity", _audit_passed(stage11_integrity, "package_integrity_audit_passed")),
        _source("stage11_load", _audit_passed(stage11_load, "package_load_verification_audit_passed")),
        _source("stage11_release", _audit_passed(stage11_release, "release_boundary_audit_passed")),
        _source("stage11_rollback", _audit_passed(stage11_rollback, "rollback_verification_audit_passed")),
        _source("stage10_manifest", bool(stage10_manifest)),
        _source("stage9_summary", _passed(stage9_summary)),
        _source("stage8_summary", _passed(stage8_summary)),
        _source("stage7_summary", _passed(stage7_summary)),
    ]
    missing = [reason for reason in read_reasons if reason != "stage13_install_manifest"]
    passed = not missing and all(source["passed"] for source in sources)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_lineage_verification",
        "lineage_verification_audit_passed": passed,
        "sources": sources,
        "missing_or_unreadable_inputs": list(missing),
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
        "audit_name": "sandbox_release_boundary",
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _consumer_manifest(
    *,
    approved: bool,
    stage13_summary_path: Path,
    install_manifest_path: Path,
    source_checkpoint_path: Path | None,
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    identity_audit: dict[str, Any],
    load_reverification_audit: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": CONSUMER_MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "verification_verdict": EXPECTED_VERIFICATION_VERDICT if approved else "pending",
        "stage13_summary": str(stage13_summary_path),
        "stage13_install_manifest": str(install_manifest_path),
        "source_package_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "sandbox_consumer_metadata_path": str(sandbox_metadata_path) if sandbox_metadata_path else None,
        "source_package_checkpoint_sha256": identity_audit.get("source_package_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_sha256": identity_audit.get("sandbox_consumer_checkpoint_sha256"),
        "source_package_checkpoint_size_bytes": identity_audit.get("source_package_checkpoint_size_bytes"),
        "sandbox_consumer_checkpoint_size_bytes": identity_audit.get("sandbox_consumer_checkpoint_size_bytes"),
        "sandbox_consumer_load_reverified": load_reverification_audit.get("sandbox_consumer_load_reverification_audit_passed") is True,
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "pending",
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
        "docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run-verification.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-16-checkpoint-publication-sandbox-install-dry-run-verification.md",
    }
    required_phrases = {
        "README.md": [
            "Stage 14 `Checkpoint Publication Sandbox Install Dry-Run Verification v1`",
            "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1/",
            "checkpoint_publication_sandbox_consumer_smoke_preflight",
        ],
        "docs/算法设计与系统架构报告.md": [
            "阶段 14 `Checkpoint Publication Sandbox Install Dry-Run Verification v1`",
            "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1/",
            "checkpoint_publication_sandbox_consumer_smoke_preflight",
        ],
        "docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run-verification.md": [
            "checkpoint-publication-sandbox-install-dry-run-verification-summary.json",
            "README.md",
            "docs/算法设计与系统架构报告.md",
            "checkpoint_publication_sandbox_consumer_smoke_preflight",
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
    stage13_root: Path,
    stage12_root: Path,
    stage11_root: Path,
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    output_root: Path,
    stage13_summary: dict[str, Any],
    identity_audit: dict[str, Any],
    manifest_consistency_audit: dict[str, Any],
    load_reverification_audit: dict[str, Any],
    metadata_verification_audit: dict[str, Any],
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
        "checkpoint_publication_sandbox_install_dry_run_verification_passed": approved,
        "checkpoint_publication_sandbox_consumer_smoke_preflight_approved": approved,
        "sandbox_installed_file_identity_audit_passed": identity_audit.get("sandbox_installed_file_identity_audit_passed") is True,
        "sandbox_install_manifest_consistency_audit_passed": manifest_consistency_audit.get("sandbox_install_manifest_consistency_audit_passed") is True,
        "sandbox_consumer_load_reverification_audit_passed": load_reverification_audit.get("sandbox_consumer_load_reverification_audit_passed") is True,
        "sandbox_metadata_verification_audit_passed": metadata_verification_audit.get("sandbox_metadata_verification_audit_passed") is True,
        "lineage_verification_audit_passed": lineage_verification_audit.get("lineage_verification_audit_passed") is True,
        "rollback_verification_audit_passed": rollback_verification_audit.get("rollback_verification_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary_audit.get("release_boundary_audit_passed") is True,
        "docs_audit_passed": docs_audit.get("docs_audit_passed") is True,
        "selected_seed": stage13_summary.get("selected_seed"),
        "selected_budget": stage13_summary.get("selected_budget"),
        "source_package_checkpoint_sha256": identity_audit.get("source_package_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_sha256": identity_audit.get("sandbox_consumer_checkpoint_sha256"),
        "source_package_checkpoint_size_bytes": identity_audit.get("source_package_checkpoint_size_bytes"),
        "sandbox_consumer_checkpoint_size_bytes": identity_audit.get("sandbox_consumer_checkpoint_size_bytes"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_sandbox_install_dry_run_verification_rejections",
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "runs_new_ppo_update": False,
        "copies_checkpoint_to_sandbox": False,
        "copies_checkpoint_to_publication_path": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "ackermann_feasible_trajectory_claimed": False,
        "repo_root": str(repo_root),
        "stage13_root": str(stage13_root),
        "stage12_root": str(stage12_root),
        "stage11_root": str(stage11_root),
        "stage10_root": str(stage10_root),
        "stage9_root": str(stage9_root),
        "stage8_root": str(stage8_root),
        "stage7_root": str(stage7_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "consumer_verification_manifest": str(paths["consumer_verification_manifest"]),
        "installed_file_identity_audit": str(paths["installed_file_identity_audit"]),
        "install_manifest_consistency_audit": str(paths["install_manifest_consistency_audit"]),
        "consumer_load_reverification_audit": str(paths["consumer_load_reverification_audit"]),
        "metadata_verification_audit": str(paths["metadata_verification_audit"]),
        "lineage_verification_audit": str(paths["lineage_verification_audit"]),
        "sandbox_release_boundary_audit": str(paths["sandbox_release_boundary_audit"]),
        "rollback_verification_audit": str(paths["rollback_verification_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Checkpoint Publication Sandbox Install Dry-Run Verification v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Verification verdict: `{summary['verification_verdict']}`\n"
        f"- Source checkpoint SHA-256: `{summary.get('source_package_checkpoint_sha256')}`\n"
        f"- Sandbox checkpoint SHA-256: `{summary.get('sandbox_consumer_checkpoint_sha256')}`\n\n"
        "## Release Boundaries\n\n"
        f"- Checkpoint publication approved: `{summary['checkpoint_publication_approved']}`\n"
        f"- Default policy replacement approved: `{summary['default_policy_replacement_approved']}`\n"
        f"- Real executor connection approved: `{summary['real_executor_connection_approved']}`\n"
        f"- Next required change: `{summary['next_required_change']}`\n"
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "consumer_verification_manifest": output_root / CONSUMER_MANIFEST_FILE,
        "installed_file_identity_audit": output_root / INSTALLED_FILE_IDENTITY_AUDIT_FILE,
        "install_manifest_consistency_audit": output_root / INSTALL_MANIFEST_CONSISTENCY_AUDIT_FILE,
        "consumer_load_reverification_audit": output_root / LOAD_REVERIFICATION_AUDIT_FILE,
        "metadata_verification_audit": output_root / METADATA_VERIFICATION_AUDIT_FILE,
        "lineage_verification_audit": output_root / LINEAGE_VERIFICATION_AUDIT_FILE,
        "sandbox_release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_FILE,
        "rollback_verification_audit": output_root / ROLLBACK_VERIFICATION_AUDIT_FILE,
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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


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
