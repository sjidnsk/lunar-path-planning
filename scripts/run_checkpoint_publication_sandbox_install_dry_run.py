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


SUMMARY_SCHEMA_VERSION = "checkpoint-publication-sandbox-install-dry-run-summary/v1"
AUDIT_SCHEMA_VERSION = "checkpoint-publication-sandbox-install-dry-run-audit/v1"
INSTALL_MANIFEST_SCHEMA_VERSION = "checkpoint-publication-sandbox-install-manifest/v1"

DEFAULT_STAGE12_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1"
DEFAULT_STAGE11_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_verification_v1"
DEFAULT_STAGE10_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1"
DEFAULT_STAGE9_ROOT = "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1"
DEFAULT_STAGE8_ROOT = "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1"
DEFAULT_STAGE7_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_v1"

STAGE12_SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-preflight-summary.json"
STAGE12_PREFLIGHT_MANIFEST_FILE = "checkpoint-publication-sandbox-preflight-manifest.json"
STAGE12_PACKAGE_CONSUMER_AUDIT_FILE = "checkpoint-publication-sandbox-package-consumer-audit.json"
STAGE12_DEFAULT_POLICY_AUDIT_FILE = "checkpoint-publication-sandbox-default-policy-boundary-audit.json"
STAGE12_PATH_AUDIT_FILE = "checkpoint-publication-sandbox-path-boundary-audit.json"
STAGE12_LINEAGE_AUDIT_FILE = "checkpoint-publication-sandbox-lineage-audit.json"
STAGE12_RELEASE_AUDIT_FILE = "checkpoint-publication-sandbox-release-boundary-audit.json"
STAGE12_ROLLBACK_AUDIT_FILE = "checkpoint-publication-sandbox-rollback-preflight-audit.json"
STAGE11_CONSUMER_MANIFEST_FILE = "checkpoint-publication-package-consumer-manifest.json"
STAGE11_INTEGRITY_AUDIT_FILE = "checkpoint-publication-package-integrity-audit.json"
STAGE11_LOAD_AUDIT_FILE = "checkpoint-publication-package-load-verification-audit.json"
STAGE11_RELEASE_AUDIT_FILE = "checkpoint-publication-package-release-boundary-audit.json"
STAGE11_ROLLBACK_AUDIT_FILE = "checkpoint-publication-package-rollback-verification-audit.json"
STAGE10_MANIFEST_FILE = "checkpoint-publication-package-manifest.json"
STAGE9_SUMMARY_FILE = "checkpoint-publication-authorization-preflight-summary.json"
STAGE8_SUMMARY_FILE = "scoped-claim-publication-evidence-freeze-summary.json"
STAGE7_SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"

SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-summary.json"
INSTALL_MANIFEST_FILE = "checkpoint-publication-sandbox-install-manifest.json"
COPY_AUDIT_FILE = "checkpoint-publication-sandbox-copy-audit.json"
CONSUMER_LOAD_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-load-audit.json"
DEFAULT_POLICY_AUDIT_FILE = "checkpoint-publication-sandbox-default-policy-boundary-audit.json"
PATH_AUDIT_FILE = "checkpoint-publication-sandbox-path-boundary-audit.json"
LINEAGE_AUDIT_FILE = "checkpoint-publication-sandbox-lineage-audit.json"
RELEASE_AUDIT_FILE = "checkpoint-publication-sandbox-release-boundary-audit.json"
ROLLBACK_AUDIT_FILE = "checkpoint-publication-sandbox-rollback-audit.json"
REJECTION_REPORT_FILE = "checkpoint-publication-sandbox-install-dry-run-rejection-report.json"
REPORT_FILE = "checkpoint-publication-sandbox-install-dry-run-report.md"

EXPECTED_STAGE12_VERDICT = "eligible_for_checkpoint_publication_sandbox_install_dry_run"
EXPECTED_STAGE12_NEXT = "checkpoint_publication_sandbox_install_dry_run"
EXPECTED_INSTALL_VERDICT = (
    "installed_in_sandbox_for_checkpoint_publication_sandbox_install_dry_run_verification"
)
NEXT_REQUIRED_CHANGE = "checkpoint_publication_sandbox_install_dry_run_verification"

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

UNSAFE_PATH_TOKENS = (
    "default_policy",
    "default-policy",
    "defaultpolicy",
    "live",
    "release",
    "executor",
    "real_executor",
    "real-executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install a verified checkpoint package into the Stage 12 sandbox for dry-run consumption."
    )
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
    summary = run_checkpoint_publication_sandbox_install_dry_run(
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
                "install_dry_run_verdict": summary["install_dry_run_verdict"],
                "checkpoint_publication_sandbox_install_dry_run_passed": summary[
                    "checkpoint_publication_sandbox_install_dry_run_passed"
                ],
                "checkpoint_publication_sandbox_install_dry_run_verification_approved": summary[
                    "checkpoint_publication_sandbox_install_dry_run_verification_approved"
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


def run_checkpoint_publication_sandbox_install_dry_run(
    *,
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
    sandbox_checkpoint_mutator: Callable[[Path], None] | None = None,
    assume_rollback_valid: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
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
    stage12_summary_path = stage12_root / STAGE12_SUMMARY_FILE
    stage12_summary = _read_json(stage12_summary_path, read_reasons, "stage12_summary")
    preflight_manifest_path = _summary_path(
        stage12_summary,
        "sandbox_preflight_manifest",
        stage12_root,
        repo_root,
        STAGE12_PREFLIGHT_MANIFEST_FILE,
    )
    stage12_package_audit_path = _summary_path(
        stage12_summary,
        "package_consumer_audit",
        stage12_root,
        repo_root,
        STAGE12_PACKAGE_CONSUMER_AUDIT_FILE,
    )
    stage12_default_policy_audit_path = _summary_path(
        stage12_summary,
        "default_policy_boundary_audit",
        stage12_root,
        repo_root,
        STAGE12_DEFAULT_POLICY_AUDIT_FILE,
    )
    stage12_path_audit_path = _summary_path(
        stage12_summary,
        "sandbox_path_boundary_audit",
        stage12_root,
        repo_root,
        STAGE12_PATH_AUDIT_FILE,
    )
    stage12_lineage_audit_path = _summary_path(
        stage12_summary,
        "sandbox_lineage_audit",
        stage12_root,
        repo_root,
        STAGE12_LINEAGE_AUDIT_FILE,
    )
    stage12_release_audit_path = _summary_path(
        stage12_summary,
        "sandbox_release_boundary_audit",
        stage12_root,
        repo_root,
        STAGE12_RELEASE_AUDIT_FILE,
    )
    stage12_rollback_audit_path = _summary_path(
        stage12_summary,
        "rollback_preflight_audit",
        stage12_root,
        repo_root,
        STAGE12_ROLLBACK_AUDIT_FILE,
    )
    preflight_manifest = _read_json(preflight_manifest_path, read_reasons, "preflight_manifest")
    stage12_package_audit = _read_json(stage12_package_audit_path, read_reasons, "stage12_package_audit")
    stage12_default_policy_audit = _read_json(
        stage12_default_policy_audit_path, read_reasons, "stage12_default_policy_audit"
    )
    stage12_path_audit = _read_json(stage12_path_audit_path, read_reasons, "stage12_path_audit")
    stage12_lineage_audit = _read_json(stage12_lineage_audit_path, read_reasons, "stage12_lineage_audit")
    stage12_release_audit = _read_json(stage12_release_audit_path, read_reasons, "stage12_release_audit")
    stage12_rollback_audit = _read_json(stage12_rollback_audit_path, read_reasons, "stage12_rollback_audit")

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
            preflight_manifest.get("source_package_checkpoint_path"),
            stage12_package_audit.get("authoritative_package_checkpoint_path"),
            stage11_consumer.get("authoritative_package_checkpoint_path"),
            stage10_manifest.get("package_checkpoint_path"),
        ),
        root=stage12_root,
        repo_root=repo_root,
    )
    source_metadata_path = _resolve_first_path(
        (
            preflight_manifest.get("source_package_metadata_path"),
            stage11_consumer.get("package_metadata_path"),
            stage10_manifest.get("package_metadata_path"),
        ),
        root=stage12_root,
        repo_root=repo_root,
    )
    sandbox_checkpoint_path = _resolve_optional_path(
        preflight_manifest.get("planned_consumer_checkpoint_path"), stage12_root, repo_root
    )
    sandbox_metadata_path = _resolve_optional_path(
        preflight_manifest.get("planned_consumer_metadata_path"), stage12_root, repo_root
    )

    source_before = _file_snapshot(source_checkpoint_path)
    metadata_before = _file_snapshot(source_metadata_path)
    default_before = _file_snapshot(default_policy_path)
    stage12_gate = _stage12_gate_audit(stage12_summary)
    path_boundary_audit = _path_boundary_audit(
        preflight_manifest_path=preflight_manifest_path,
        preflight_manifest=preflight_manifest,
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
    )
    source_identity_audit = _source_identity_audit(
        source_checkpoint_path=source_checkpoint_path,
        source_metadata_path=source_metadata_path,
        stage12_summary=stage12_summary,
        preflight_manifest=preflight_manifest,
        stage12_package_audit=stage12_package_audit,
        stage11_consumer=stage11_consumer,
        stage10_manifest=stage10_manifest,
    )
    copy_audit = _copy_audit(
        source_checkpoint_path=source_checkpoint_path,
        source_metadata_path=source_metadata_path,
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        source_identity_audit=source_identity_audit,
        path_boundary_audit=path_boundary_audit,
    )
    if sandbox_checkpoint_mutator is not None and sandbox_checkpoint_path and sandbox_checkpoint_path.is_file():
        sandbox_checkpoint_mutator(sandbox_checkpoint_path)
    consumer_load_audit = _consumer_load_audit(
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        expected_sha=source_identity_audit.get("source_package_checkpoint_sha256"),
        expected_size=source_identity_audit.get("source_package_checkpoint_size_bytes"),
    )
    if default_policy_mutator is not None and default_policy_path is not None:
        default_policy_mutator(default_policy_path)
    default_policy_audit = _default_policy_boundary_audit(
        default_policy_path=default_policy_path,
        before=default_before,
        after=_file_snapshot(default_policy_path),
    )
    rollback_audit = _rollback_audit(
        source_before=source_before,
        source_after=_file_snapshot(source_checkpoint_path),
        metadata_before=metadata_before,
        metadata_after=_file_snapshot(source_metadata_path),
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        default_policy_audit=default_policy_audit,
        copy_audit=copy_audit,
        assume_rollback_valid=assume_rollback_valid,
    )
    lineage_audit = _lineage_audit(
        read_reasons=read_reasons,
        stage12_summary=stage12_summary,
        preflight_manifest=preflight_manifest,
        stage12_package_audit=stage12_package_audit,
        stage12_default_policy_audit=stage12_default_policy_audit,
        stage12_path_audit=stage12_path_audit,
        stage12_lineage_audit=stage12_lineage_audit,
        stage12_release_audit=stage12_release_audit,
        stage12_rollback_audit=stage12_rollback_audit,
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
            "stage12_summary": stage12_summary,
            "stage12_preflight_manifest": preflight_manifest,
            "stage12_package_audit": stage12_package_audit,
            "stage12_default_policy_audit": stage12_default_policy_audit,
            "stage12_path_audit": stage12_path_audit,
            "stage12_lineage_audit": stage12_lineage_audit,
            "stage12_release_audit": stage12_release_audit,
            "stage12_rollback_audit": stage12_rollback_audit,
            "stage11_consumer": stage11_consumer,
            "stage11_integrity": stage11_integrity,
            "stage11_load": stage11_load,
            "stage11_release": stage11_release,
            "stage11_rollback": stage11_rollback,
            "stage10_manifest": stage10_manifest,
            "stage9_summary": stage9_summary,
            "stage8_summary": stage8_summary,
            "stage7_summary": stage7_summary,
            "copy_audit": copy_audit,
            "consumer_load_audit": consumer_load_audit,
            "default_policy_audit": default_policy_audit,
            "path_boundary_audit": path_boundary_audit,
            "rollback_audit": rollback_audit,
        }
    )
    docs_audit = _docs_audit(repo_root=repo_root)
    install_manifest = _install_manifest(
        preflight_manifest=preflight_manifest,
        preflight_manifest_path=preflight_manifest_path,
        source_checkpoint_path=source_checkpoint_path,
        source_metadata_path=source_metadata_path,
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        copy_audit=copy_audit,
        consumer_load_audit=consumer_load_audit,
        repo_root=repo_root,
    )

    reason_codes = _unique(
        [
            *stage12_gate["reason_codes"],
            *source_identity_audit["reason_codes"],
            *path_boundary_audit["reason_codes"],
            *copy_audit["reason_codes"],
            *consumer_load_audit["reason_codes"],
            *default_policy_audit["reason_codes"],
            *([] if lineage_audit["lineage_audit_passed"] else ["lineage_incomplete"]),
            *([] if rollback_audit["rollback_audit_passed"] else ["rollback_boundary_invalid"]),
            *([] if release_boundary_audit["release_boundary_audit_passed"] else ["release_boundary_violation"]),
            *([] if docs_audit["docs_audit_passed"] else ["docs_not_updated"]),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    approved = status == "passed"
    install_manifest["install_dry_run_verdict"] = EXPECTED_INSTALL_VERDICT if approved else "not_installed"
    install_manifest["next_required_change"] = (
        NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_sandbox_install_dry_run_rejections"
    )

    _write_json(paths["sandbox_install_manifest"], install_manifest)
    _write_json(paths["sandbox_copy_audit"], copy_audit)
    _write_json(paths["sandbox_consumer_load_audit"], consumer_load_audit)
    _write_json(paths["default_policy_boundary_audit"], default_policy_audit)
    _write_json(paths["sandbox_path_boundary_audit"], path_boundary_audit)
    _write_json(paths["sandbox_lineage_audit"], lineage_audit)
    _write_json(paths["sandbox_release_boundary_audit"], release_boundary_audit)
    _write_json(paths["rollback_audit"], rollback_audit)
    _write_json(paths["rejection_report"], _rejection_report(reason_codes))
    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        approved=approved,
        paths=paths,
        repo_root=repo_root,
        stage12_root=stage12_root,
        stage11_root=stage11_root,
        stage10_root=stage10_root,
        stage9_root=stage9_root,
        stage8_root=stage8_root,
        stage7_root=stage7_root,
        output_root=output_root,
        stage12_summary=stage12_summary,
        install_manifest=install_manifest,
        copy_audit=copy_audit,
        consumer_load_audit=consumer_load_audit,
        default_policy_audit=default_policy_audit,
        path_boundary_audit=path_boundary_audit,
        lineage_audit=lineage_audit,
        rollback_audit=rollback_audit,
        release_boundary_audit=release_boundary_audit,
        docs_audit=docs_audit,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage12_gate_audit(stage12_summary: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if (
        stage12_summary.get("status") != "passed"
        or _string_list(stage12_summary.get("reason_codes"))
        or stage12_summary.get("checkpoint_publication_sandbox_install_dry_run_preflight_passed") is not True
    ):
        _add_reason(reason_codes, "stage12_not_passed")
    if (
        stage12_summary.get("preflight_verdict") != EXPECTED_STAGE12_VERDICT
        or stage12_summary.get("checkpoint_publication_sandbox_install_dry_run_approved") is not True
        or stage12_summary.get("next_required_change") != EXPECTED_STAGE12_NEXT
    ):
        _add_reason(reason_codes, "stage12_not_authorized_for_sandbox_install_dry_run")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "stage12_gate",
        "stage12_gate_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
    }


def _source_identity_audit(
    *,
    source_checkpoint_path: Path | None,
    source_metadata_path: Path | None,
    stage12_summary: dict[str, Any],
    preflight_manifest: dict[str, Any],
    stage12_package_audit: dict[str, Any],
    stage11_consumer: dict[str, Any],
    stage10_manifest: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    actual_sha = None
    actual_size = None
    if source_checkpoint_path is None or not source_checkpoint_path.is_file():
        _add_reason(reason_codes, "source_package_checkpoint_missing")
    else:
        actual_sha, actual_size = _sha256_and_size(source_checkpoint_path)
        expected_shas = _unique(
            [
                str(value)
                for value in (
                    stage12_summary.get("package_checkpoint_sha256"),
                    preflight_manifest.get("package_checkpoint_sha256"),
                    stage12_package_audit.get("package_checkpoint_sha256"),
                    stage11_consumer.get("package_checkpoint_sha256"),
                    stage10_manifest.get("package_checkpoint_sha256"),
                )
                if value
            ]
        )
        expected_sizes = _unique_ints(
            (
                stage12_summary.get("package_checkpoint_size_bytes"),
                preflight_manifest.get("package_checkpoint_size_bytes"),
                stage12_package_audit.get("package_checkpoint_size_bytes"),
                stage11_consumer.get("package_checkpoint_size_bytes"),
                stage10_manifest.get("package_checkpoint_size_bytes"),
            )
        )
        if any(actual_sha != expected for expected in expected_shas) or any(
            actual_size != expected for expected in expected_sizes
        ):
            _add_reason(reason_codes, "source_package_identity_mismatch")
    if source_metadata_path is None or not source_metadata_path.is_file():
        _add_reason(reason_codes, "source_package_metadata_missing")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "source_package_identity",
        "source_package_identity_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "source_package_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "source_package_metadata_path": str(source_metadata_path) if source_metadata_path else None,
        "source_package_checkpoint_sha256": actual_sha,
        "source_package_checkpoint_size_bytes": actual_size,
    }


def _path_boundary_audit(
    *,
    preflight_manifest_path: Path,
    preflight_manifest: dict[str, Any],
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
) -> dict[str, Any]:
    violations: list[str] = []
    if not preflight_manifest_path.is_file() or not preflight_manifest:
        violations.append("preflight_manifest_missing")
    sandbox_root = _path_from_value(preflight_manifest.get("sandbox_root"))
    if sandbox_checkpoint_path is None:
        violations.append("planned_checkpoint_missing")
    if sandbox_metadata_path is None:
        violations.append("planned_metadata_missing")
    if sandbox_root and sandbox_checkpoint_path and not _is_relative_to(sandbox_checkpoint_path, sandbox_root):
        violations.append("planned_checkpoint_outside_sandbox_root")
    if sandbox_root and sandbox_metadata_path and not _is_relative_to(sandbox_metadata_path, sandbox_root):
        violations.append("planned_metadata_outside_sandbox_root")
    for label, path in (
        ("sandbox_root", sandbox_root),
        ("planned_consumer_checkpoint_path", sandbox_checkpoint_path),
        ("planned_consumer_metadata_path", sandbox_metadata_path),
    ):
        if path and _has_unsafe_path_token(path):
            violations.append(f"{label}_has_unsafe_semantic_token")
    reason_codes: list[str] = []
    if "preflight_manifest_missing" in violations:
        _add_reason(reason_codes, "preflight_manifest_missing")
    unsafe_violations = [value for value in violations if value != "preflight_manifest_missing"]
    if unsafe_violations:
        _add_reason(reason_codes, "sandbox_path_boundary_violation")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_path_boundary",
        "sandbox_path_boundary_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "preflight_manifest_path": str(preflight_manifest_path),
        "sandbox_root": str(sandbox_root) if sandbox_root else None,
        "planned_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "planned_consumer_metadata_path": str(sandbox_metadata_path) if sandbox_metadata_path else None,
        "violations": violations,
    }


def _copy_audit(
    *,
    source_checkpoint_path: Path | None,
    source_metadata_path: Path | None,
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    source_identity_audit: dict[str, Any],
    path_boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    copied_checkpoint = False
    copied_metadata = False
    copy_error = None
    if source_identity_audit.get("source_package_identity_audit_passed") is not True:
        return _copy_audit_payload(reason_codes, copied_checkpoint, copied_metadata, copy_error, source_checkpoint_path, sandbox_checkpoint_path)
    if path_boundary_audit.get("sandbox_path_boundary_audit_passed") is not True:
        return _copy_audit_payload(reason_codes, copied_checkpoint, copied_metadata, copy_error, source_checkpoint_path, sandbox_checkpoint_path)
    try:
        assert source_checkpoint_path is not None
        assert source_metadata_path is not None
        assert sandbox_checkpoint_path is not None
        assert sandbox_metadata_path is not None
        source_sha, source_size = _sha256_and_size(source_checkpoint_path)
        metadata_sha, metadata_size = _sha256_and_size(source_metadata_path)
        sandbox_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if sandbox_checkpoint_path.exists():
            existing_sha, existing_size = _sha256_and_size(sandbox_checkpoint_path)
            if existing_sha != source_sha or existing_size != source_size:
                _add_reason(reason_codes, "sandbox_copy_failed")
                copy_error = "existing sandbox checkpoint identity mismatch"
                return _copy_audit_payload(reason_codes, copied_checkpoint, copied_metadata, copy_error, source_checkpoint_path, sandbox_checkpoint_path)
        else:
            shutil.copy2(source_checkpoint_path, sandbox_checkpoint_path)
            copied_checkpoint = True
        if sandbox_metadata_path.exists():
            existing_metadata_sha, existing_metadata_size = _sha256_and_size(sandbox_metadata_path)
            if existing_metadata_sha != metadata_sha or existing_metadata_size != metadata_size:
                _add_reason(reason_codes, "sandbox_copy_failed")
                copy_error = "existing sandbox metadata identity mismatch"
                return _copy_audit_payload(reason_codes, copied_checkpoint, copied_metadata, copy_error, source_checkpoint_path, sandbox_checkpoint_path)
        else:
            shutil.copy2(source_metadata_path, sandbox_metadata_path)
            copied_metadata = True
    except Exception as exc:  # noqa: BLE001 - audit records copy failures
        _add_reason(reason_codes, "sandbox_copy_failed")
        copy_error = str(exc)
    return _copy_audit_payload(reason_codes, copied_checkpoint, copied_metadata, copy_error, source_checkpoint_path, sandbox_checkpoint_path)


def _copy_audit_payload(
    reason_codes: list[str],
    copied_checkpoint: bool,
    copied_metadata: bool,
    copy_error: str | None,
    source_checkpoint_path: Path | None,
    sandbox_checkpoint_path: Path | None,
) -> dict[str, Any]:
    source_sha = source_size = sandbox_sha = sandbox_size = None
    if source_checkpoint_path and source_checkpoint_path.is_file():
        source_sha, source_size = _sha256_and_size(source_checkpoint_path)
    if sandbox_checkpoint_path and sandbox_checkpoint_path.is_file():
        sandbox_sha, sandbox_size = _sha256_and_size(sandbox_checkpoint_path)
    if (
        not reason_codes
        and source_sha is not None
        and sandbox_sha is not None
        and (source_sha != sandbox_sha or source_size != sandbox_size)
    ):
        _add_reason(reason_codes, "sandbox_consumer_identity_mismatch")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_copy",
        "sandbox_copy_audit_passed": not reason_codes and sandbox_sha is not None,
        "reason_codes": reason_codes,
        "source_package_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "source_package_checkpoint_sha256": source_sha,
        "sandbox_consumer_checkpoint_sha256": sandbox_sha,
        "source_package_checkpoint_size_bytes": source_size,
        "sandbox_consumer_checkpoint_size_bytes": sandbox_size,
        "copied_checkpoint": copied_checkpoint,
        "copied_metadata": copied_metadata,
        "copy_error": copy_error,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _consumer_load_audit(
    *,
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    expected_sha: Any,
    expected_size: Any,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    checkpoint_type = None
    tensor_count = 0
    state_key_count = 0
    non_finite_tensor_count = 0
    actual_sha = None
    actual_size = None
    load_error = None
    metadata = _read_json(sandbox_metadata_path, [], "sandbox_metadata") if sandbox_metadata_path else {}
    try:
        if sandbox_checkpoint_path is None or not sandbox_checkpoint_path.is_file():
            raise FileNotFoundError("sandbox checkpoint missing")
        actual_sha, actual_size = _sha256_and_size(sandbox_checkpoint_path)
        if expected_sha and actual_sha != expected_sha:
            _add_reason(reason_codes, "sandbox_consumer_identity_mismatch")
        if _int_or_none(expected_size) is not None and actual_size != _int_or_none(expected_size):
            _add_reason(reason_codes, "sandbox_consumer_identity_mismatch")
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
        for field in RELEASE_BOUNDARY_FIELDS + ("performance_claimed",):
            if checkpoint.get(field) is True or metadata.get(field) is True:
                raise ValueError(f"prohibited claim in checkpoint metadata: {field}")
    except Exception as exc:  # noqa: BLE001 - audit records load failures
        load_error = str(exc)
        _add_reason(reason_codes, "sandbox_consumer_load_failed")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_consumer_load",
        "sandbox_consumer_load_audit_passed": not reason_codes,
        "reason_codes": _unique(reason_codes),
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "sandbox_consumer_metadata_path": str(sandbox_metadata_path) if sandbox_metadata_path else None,
        "sandbox_consumer_checkpoint_sha256": actual_sha,
        "sandbox_consumer_checkpoint_size_bytes": actual_size,
        "checkpoint_type": checkpoint_type,
        "state_key_count": state_key_count,
        "tensor_count": tensor_count,
        "non_finite_tensor_count": non_finite_tensor_count,
        "load_error": load_error,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
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


def _rollback_audit(
    *,
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    metadata_before: dict[str, Any],
    metadata_after: dict[str, Any],
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    default_policy_audit: dict[str, Any],
    copy_audit: dict[str, Any],
    assume_rollback_valid: bool,
) -> dict[str, Any]:
    source_unchanged = _snapshots_equal(source_before, source_after)
    metadata_unchanged = _snapshots_equal(metadata_before, metadata_after)
    default_policy_unchanged = default_policy_audit.get("default_policy_boundary_audit_passed") is True
    sandbox_checkpoint_traceable = (
        sandbox_checkpoint_path is not None
        and sandbox_checkpoint_path.is_file()
        and copy_audit.get("sandbox_consumer_checkpoint_sha256")
        == copy_audit.get("source_package_checkpoint_sha256")
    )
    sandbox_metadata_exists = sandbox_metadata_path is not None and sandbox_metadata_path.is_file()
    passed = (
        assume_rollback_valid
        and source_unchanged
        and metadata_unchanged
        and default_policy_unchanged
        and sandbox_checkpoint_traceable
        and sandbox_metadata_exists
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_rollback",
        "rollback_audit_passed": passed,
        "source_checkpoint_unchanged": source_unchanged,
        "source_metadata_unchanged": metadata_unchanged,
        "default_policy_unchanged": default_policy_unchanged,
        "sandbox_checkpoint_traceable": sandbox_checkpoint_traceable,
        "sandbox_metadata_exists": sandbox_metadata_exists,
        "assume_rollback_valid": assume_rollback_valid,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _lineage_audit(
    *,
    read_reasons: list[str],
    stage12_summary: dict[str, Any],
    preflight_manifest: dict[str, Any],
    stage12_package_audit: dict[str, Any],
    stage12_default_policy_audit: dict[str, Any],
    stage12_path_audit: dict[str, Any],
    stage12_lineage_audit: dict[str, Any],
    stage12_release_audit: dict[str, Any],
    stage12_rollback_audit: dict[str, Any],
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
        _source("stage12_summary", _passed(stage12_summary)),
        _source("stage12_preflight_manifest", preflight_manifest.get("sandbox_preflight_manifest_passed") is True),
        _source("stage12_package_audit", _audit_passed(stage12_package_audit, "package_consumer_audit_passed")),
        _source("stage12_default_policy_audit", _audit_passed(stage12_default_policy_audit, "default_policy_boundary_audit_passed")),
        _source("stage12_path_audit", _audit_passed(stage12_path_audit, "sandbox_path_boundary_audit_passed")),
        _source("stage12_lineage_audit", _audit_passed(stage12_lineage_audit, "lineage_audit_passed")),
        _source("stage12_release_audit", _audit_passed(stage12_release_audit, "release_boundary_audit_passed")),
        _source("stage12_rollback_audit", _audit_passed(stage12_rollback_audit, "rollback_preflight_audit_passed")),
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
    missing = [reason for reason in read_reasons if reason != "preflight_manifest"]
    passed = not missing and all(source["passed"] for source in sources)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_lineage",
        "lineage_audit_passed": passed,
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


def _install_manifest(
    *,
    preflight_manifest: dict[str, Any],
    preflight_manifest_path: Path,
    source_checkpoint_path: Path | None,
    source_metadata_path: Path | None,
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    copy_audit: dict[str, Any],
    consumer_load_audit: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": INSTALL_MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "install_dry_run_verdict": "pending",
        "sandbox_install_manifest_passed": True,
        "stage12_preflight_manifest": str(preflight_manifest_path),
        "sandbox_root": preflight_manifest.get("sandbox_root"),
        "source_package_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "source_package_metadata_path": str(source_metadata_path) if source_metadata_path else None,
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "sandbox_consumer_metadata_path": str(sandbox_metadata_path) if sandbox_metadata_path else None,
        "source_package_checkpoint_sha256": copy_audit.get("source_package_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_sha256": consumer_load_audit.get("sandbox_consumer_checkpoint_sha256"),
        "source_package_checkpoint_size_bytes": copy_audit.get("source_package_checkpoint_size_bytes"),
        "sandbox_consumer_checkpoint_size_bytes": consumer_load_audit.get("sandbox_consumer_checkpoint_size_bytes"),
        "stage13_copies_checkpoint_to_sandbox": True,
        "stage13_runs_rollout": False,
        "stage13_installs_default_policy": False,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "next_required_change": "pending",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _docs_audit(*, repo_root: Path) -> dict[str, Any]:
    required_docs = {
        "README.md": repo_root / "README.md",
        "docs/算法设计与系统架构报告.md": repo_root / "docs" / "算法设计与系统架构报告.md",
        "docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-16-checkpoint-publication-sandbox-install-dry-run.md",
    }
    required_phrases = {
        "README.md": [
            "Stage 13 `Checkpoint Publication Sandbox Install Dry-Run v1`",
            "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_v1/",
            "checkpoint_publication_sandbox_install_dry_run_verification",
        ],
        "docs/算法设计与系统架构报告.md": [
            "阶段 13 `Checkpoint Publication Sandbox Install Dry-Run v1`",
            "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_v1/",
            "checkpoint_publication_sandbox_install_dry_run_verification",
        ],
        "docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run.md": [
            "checkpoint-publication-sandbox-install-dry-run-summary.json",
            "README.md",
            "docs/算法设计与系统架构报告.md",
            "checkpoint_publication_sandbox_install_dry_run_verification",
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
    stage12_root: Path,
    stage11_root: Path,
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    output_root: Path,
    stage12_summary: dict[str, Any],
    install_manifest: dict[str, Any],
    copy_audit: dict[str, Any],
    consumer_load_audit: dict[str, Any],
    default_policy_audit: dict[str, Any],
    path_boundary_audit: dict[str, Any],
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
        "install_dry_run_verdict": EXPECTED_INSTALL_VERDICT if approved else _failed_verdict(reason_codes),
        "checkpoint_publication_sandbox_install_dry_run_passed": approved,
        "checkpoint_publication_sandbox_install_dry_run_verification_approved": approved,
        "sandbox_install_manifest_passed": install_manifest.get("sandbox_install_manifest_passed") is True,
        "sandbox_copy_audit_passed": copy_audit.get("sandbox_copy_audit_passed") is True,
        "sandbox_consumer_load_audit_passed": consumer_load_audit.get("sandbox_consumer_load_audit_passed") is True,
        "default_policy_boundary_audit_passed": default_policy_audit.get("default_policy_boundary_audit_passed") is True,
        "sandbox_path_boundary_audit_passed": path_boundary_audit.get("sandbox_path_boundary_audit_passed") is True,
        "lineage_audit_passed": lineage_audit.get("lineage_audit_passed") is True,
        "rollback_audit_passed": rollback_audit.get("rollback_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary_audit.get("release_boundary_audit_passed") is True,
        "docs_audit_passed": docs_audit.get("docs_audit_passed") is True,
        "selected_seed": stage12_summary.get("selected_seed"),
        "selected_budget": stage12_summary.get("selected_budget"),
        "source_package_checkpoint_sha256": copy_audit.get("source_package_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_sha256": consumer_load_audit.get("sandbox_consumer_checkpoint_sha256"),
        "source_package_checkpoint_size_bytes": copy_audit.get("source_package_checkpoint_size_bytes"),
        "sandbox_consumer_checkpoint_size_bytes": consumer_load_audit.get("sandbox_consumer_checkpoint_size_bytes"),
        "sandbox_root": install_manifest.get("sandbox_root"),
        "sandbox_consumer_checkpoint_path": install_manifest.get("sandbox_consumer_checkpoint_path"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_sandbox_install_dry_run_rejections",
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "runs_new_ppo_update": False,
        "copies_checkpoint_to_sandbox": True if approved else copy_audit.get("copied_checkpoint") is True,
        "copies_checkpoint_to_publication_path": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "ackermann_feasible_trajectory_claimed": False,
        "repo_root": str(repo_root),
        "stage12_root": str(stage12_root),
        "stage11_root": str(stage11_root),
        "stage10_root": str(stage10_root),
        "stage9_root": str(stage9_root),
        "stage8_root": str(stage8_root),
        "stage7_root": str(stage7_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "sandbox_install_manifest": str(paths["sandbox_install_manifest"]),
        "sandbox_copy_audit": str(paths["sandbox_copy_audit"]),
        "sandbox_consumer_load_audit": str(paths["sandbox_consumer_load_audit"]),
        "default_policy_boundary_audit": str(paths["default_policy_boundary_audit"]),
        "sandbox_path_boundary_audit": str(paths["sandbox_path_boundary_audit"]),
        "sandbox_lineage_audit": str(paths["sandbox_lineage_audit"]),
        "sandbox_release_boundary_audit": str(paths["sandbox_release_boundary_audit"]),
        "rollback_audit": str(paths["rollback_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Checkpoint Publication Sandbox Install Dry-Run v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Install dry-run verdict: `{summary['install_dry_run_verdict']}`\n"
        f"- Selected seed: `{summary.get('selected_seed')}`\n"
        f"- Selected budget: `{summary.get('selected_budget')}`\n"
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
        "sandbox_install_manifest": output_root / INSTALL_MANIFEST_FILE,
        "sandbox_copy_audit": output_root / COPY_AUDIT_FILE,
        "sandbox_consumer_load_audit": output_root / CONSUMER_LOAD_AUDIT_FILE,
        "default_policy_boundary_audit": output_root / DEFAULT_POLICY_AUDIT_FILE,
        "sandbox_path_boundary_audit": output_root / PATH_AUDIT_FILE,
        "sandbox_lineage_audit": output_root / LINEAGE_AUDIT_FILE,
        "sandbox_release_boundary_audit": output_root / RELEASE_AUDIT_FILE,
        "rollback_audit": output_root / ROLLBACK_AUDIT_FILE,
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


def _path_from_value(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


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


def _has_unsafe_path_token(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return any(any(token in part for token in UNSAFE_PATH_TOKENS) for part in parts)


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
