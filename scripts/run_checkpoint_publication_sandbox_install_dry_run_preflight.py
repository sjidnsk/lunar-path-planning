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


SUMMARY_SCHEMA_VERSION = "checkpoint-publication-sandbox-install-dry-run-preflight-summary/v1"
AUDIT_SCHEMA_VERSION = "checkpoint-publication-sandbox-install-dry-run-preflight-audit/v1"
MANIFEST_SCHEMA_VERSION = "checkpoint-publication-sandbox-preflight-manifest/v1"

DEFAULT_STAGE11_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_verification_v1"
DEFAULT_STAGE10_ROOT = "outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1"
DEFAULT_STAGE9_ROOT = "outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1"
DEFAULT_STAGE8_ROOT = "outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1"
DEFAULT_STAGE7_ROOT = "outputs/path_feedback_batch_formal_performance_claim_release_decision_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1"

STAGE11_SUMMARY_FILE = "checkpoint-publication-package-verification-summary.json"
STAGE11_CONSUMER_MANIFEST_FILE = "checkpoint-publication-package-consumer-manifest.json"
STAGE11_INTEGRITY_AUDIT_FILE = "checkpoint-publication-package-integrity-audit.json"
STAGE11_LOAD_AUDIT_FILE = "checkpoint-publication-package-load-verification-audit.json"
STAGE11_LINEAGE_AUDIT_FILE = "checkpoint-publication-package-lineage-verification-audit.json"
STAGE11_RELEASE_AUDIT_FILE = "checkpoint-publication-package-release-boundary-audit.json"
STAGE11_ROLLBACK_AUDIT_FILE = "checkpoint-publication-package-rollback-verification-audit.json"
STAGE10_MANIFEST_FILE = "checkpoint-publication-package-manifest.json"
STAGE9_SUMMARY_FILE = "checkpoint-publication-authorization-preflight-summary.json"
STAGE8_SUMMARY_FILE = "scoped-claim-publication-evidence-freeze-summary.json"
STAGE7_SUMMARY_FILE = "formal-performance-claim-release-decision-summary.json"

SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-preflight-summary.json"
SANDBOX_MANIFEST_FILE = "checkpoint-publication-sandbox-preflight-manifest.json"
PACKAGE_CONSUMER_AUDIT_FILE = "checkpoint-publication-sandbox-package-consumer-audit.json"
DEFAULT_POLICY_BOUNDARY_AUDIT_FILE = "checkpoint-publication-sandbox-default-policy-boundary-audit.json"
PATH_BOUNDARY_AUDIT_FILE = "checkpoint-publication-sandbox-path-boundary-audit.json"
LINEAGE_AUDIT_FILE = "checkpoint-publication-sandbox-lineage-audit.json"
RELEASE_BOUNDARY_AUDIT_FILE = "checkpoint-publication-sandbox-release-boundary-audit.json"
ROLLBACK_PREFLIGHT_AUDIT_FILE = "checkpoint-publication-sandbox-rollback-preflight-audit.json"
REJECTION_REPORT_FILE = "checkpoint-publication-sandbox-install-dry-run-preflight-rejection-report.json"
REPORT_FILE = "checkpoint-publication-sandbox-install-dry-run-preflight-report.md"

EXPECTED_STAGE11_VERDICT = "verified_for_checkpoint_publication_sandbox_install_dry_run_preflight"
EXPECTED_STAGE11_NEXT = "checkpoint_publication_sandbox_install_dry_run_preflight"
EXPECTED_PREFLIGHT_VERDICT = "eligible_for_checkpoint_publication_sandbox_install_dry_run"
NEXT_REQUIRED_CHANGE = "checkpoint_publication_sandbox_install_dry_run"

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
        description="Preflight a sandbox install dry-run for a verified checkpoint package."
    )
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
    summary = run_checkpoint_publication_sandbox_install_dry_run_preflight(
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
                "preflight_verdict": summary["preflight_verdict"],
                "checkpoint_publication_sandbox_install_dry_run_preflight_passed": summary[
                    "checkpoint_publication_sandbox_install_dry_run_preflight_passed"
                ],
                "checkpoint_publication_sandbox_install_dry_run_approved": summary[
                    "checkpoint_publication_sandbox_install_dry_run_approved"
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


def run_checkpoint_publication_sandbox_install_dry_run_preflight(
    *,
    stage11_root: Path,
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    output_root: Path,
    repo_root: Path,
    default_policy_path: Path | None = None,
    default_policy_mutator: Callable[[Path], None] | None = None,
    assume_rollback_preflight_valid: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stage11_root = Path(stage11_root)
    stage10_root = Path(stage10_root)
    stage9_root = Path(stage9_root)
    stage8_root = Path(stage8_root)
    stage7_root = Path(stage7_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    read_reasons: list[str] = []
    stage11_summary_path = stage11_root / STAGE11_SUMMARY_FILE
    stage11_summary = _read_json(stage11_summary_path, read_reasons, "stage11_summary")
    consumer_manifest_path = _summary_path(
        stage11_summary,
        "consumer_manifest",
        stage11_root,
        repo_root,
        STAGE11_CONSUMER_MANIFEST_FILE,
    )
    integrity_audit_path = _summary_path(
        stage11_summary,
        "package_integrity_audit",
        stage11_root,
        repo_root,
        STAGE11_INTEGRITY_AUDIT_FILE,
    )
    load_audit_path = _summary_path(
        stage11_summary,
        "package_load_verification_audit",
        stage11_root,
        repo_root,
        STAGE11_LOAD_AUDIT_FILE,
    )
    lineage_audit_path = _summary_path(
        stage11_summary,
        "package_lineage_verification_audit",
        stage11_root,
        repo_root,
        STAGE11_LINEAGE_AUDIT_FILE,
    )
    release_audit_path = _summary_path(
        stage11_summary,
        "package_release_boundary_audit",
        stage11_root,
        repo_root,
        STAGE11_RELEASE_AUDIT_FILE,
    )
    rollback_audit_path = _summary_path(
        stage11_summary,
        "package_rollback_verification_audit",
        stage11_root,
        repo_root,
        STAGE11_ROLLBACK_AUDIT_FILE,
    )

    consumer_manifest = _read_json(consumer_manifest_path, read_reasons, "consumer_manifest")
    integrity_audit = _read_json(integrity_audit_path, read_reasons, "stage11_integrity_audit")
    load_audit = _read_json(load_audit_path, read_reasons, "stage11_load_audit")
    stage11_lineage_audit = _read_json(lineage_audit_path, read_reasons, "stage11_lineage_audit")
    stage11_release_audit = _read_json(release_audit_path, read_reasons, "stage11_release_audit")
    stage11_rollback_audit = _read_json(rollback_audit_path, read_reasons, "stage11_rollback_audit")

    stage10_manifest_path = _resolve_optional_path(
        consumer_manifest.get("stage10_manifest"), stage11_root, repo_root
    ) or (stage10_root / STAGE10_MANIFEST_FILE)
    stage10_manifest = _read_json(stage10_manifest_path, read_reasons, "stage10_manifest")
    package_checkpoint_path = _resolve_optional_path(
        consumer_manifest.get("authoritative_package_checkpoint_path"), stage11_root, repo_root
    )
    package_metadata_path = _resolve_optional_path(
        consumer_manifest.get("package_metadata_path"), stage11_root, repo_root
    ) or _resolve_optional_path(stage10_manifest.get("package_metadata_path"), stage10_root, repo_root)
    package_metadata = _read_json(package_metadata_path, [], "package_metadata") if package_metadata_path else {}
    package_before = _file_snapshot(package_checkpoint_path)
    metadata_before = _file_snapshot(package_metadata_path)
    default_before = _file_snapshot(default_policy_path)

    stage9_summary = _read_json(stage9_root / STAGE9_SUMMARY_FILE, read_reasons, "stage9_summary")
    stage8_summary = _read_json(stage8_root / STAGE8_SUMMARY_FILE, read_reasons, "stage8_summary")
    stage7_summary = _read_json(stage7_root / STAGE7_SUMMARY_FILE, read_reasons, "stage7_summary")

    stage11_gate = _stage11_gate_audit(stage11_summary)
    package_consumer_audit = _package_consumer_audit(
        consumer_manifest_path=consumer_manifest_path,
        consumer_manifest=consumer_manifest,
        integrity_audit=integrity_audit,
        load_audit=load_audit,
        stage11_summary=stage11_summary,
        package_checkpoint_path=package_checkpoint_path,
    )
    sandbox_manifest = _sandbox_preflight_manifest(
        output_root=output_root,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        package_consumer_audit=package_consumer_audit,
        stage11_summary_path=stage11_summary_path,
        consumer_manifest_path=consumer_manifest_path,
        repo_root=repo_root,
    )
    path_boundary_audit = _path_boundary_audit(sandbox_manifest=sandbox_manifest, output_root=output_root)
    if default_policy_mutator is not None and default_policy_path is not None:
        default_policy_mutator(default_policy_path)
    default_policy_boundary_audit = _default_policy_boundary_audit(
        default_policy_path=default_policy_path,
        before=default_before,
        after=_file_snapshot(default_policy_path),
    )
    rollback_preflight_audit = _rollback_preflight_audit(
        package_before=package_before,
        package_after=_file_snapshot(package_checkpoint_path),
        metadata_before=metadata_before,
        metadata_after=_file_snapshot(package_metadata_path),
        default_policy_boundary_audit=default_policy_boundary_audit,
        sandbox_manifest=sandbox_manifest,
        assume_rollback_preflight_valid=assume_rollback_preflight_valid,
    )
    lineage_audit = _lineage_audit(
        read_reasons=read_reasons,
        stage11_summary=stage11_summary,
        consumer_manifest=consumer_manifest,
        integrity_audit=integrity_audit,
        load_audit=load_audit,
        stage11_lineage_audit=stage11_lineage_audit,
        stage11_release_audit=stage11_release_audit,
        stage11_rollback_audit=stage11_rollback_audit,
        stage10_manifest=stage10_manifest,
        stage9_summary=stage9_summary,
        stage8_summary=stage8_summary,
        stage7_summary=stage7_summary,
    )
    release_boundary_audit = _release_boundary_audit(
        payloads={
            "stage11_summary": stage11_summary,
            "stage11_consumer_manifest": consumer_manifest,
            "stage11_integrity_audit": integrity_audit,
            "stage11_load_audit": load_audit,
            "stage11_lineage_audit": stage11_lineage_audit,
            "stage11_release_audit": stage11_release_audit,
            "stage11_rollback_audit": stage11_rollback_audit,
            "stage10_manifest": stage10_manifest,
            "stage9_summary": stage9_summary,
            "stage8_summary": stage8_summary,
            "stage7_summary": stage7_summary,
            "package_metadata": package_metadata,
            "sandbox_manifest": sandbox_manifest,
            "package_consumer_audit": package_consumer_audit,
            "default_policy_boundary_audit": default_policy_boundary_audit,
            "path_boundary_audit": path_boundary_audit,
            "rollback_preflight_audit": rollback_preflight_audit,
        }
    )
    docs_audit = _docs_audit(repo_root=repo_root)

    reason_codes = _unique(
        [
            *stage11_gate["reason_codes"],
            *package_consumer_audit["reason_codes"],
            *path_boundary_audit["reason_codes"],
            *default_policy_boundary_audit["reason_codes"],
            *([] if lineage_audit["lineage_audit_passed"] else ["lineage_incomplete"]),
            *([] if rollback_preflight_audit["rollback_preflight_audit_passed"] else ["rollback_preflight_invalid"]),
            *([] if release_boundary_audit["release_boundary_audit_passed"] else ["release_boundary_violation"]),
            *([] if docs_audit["docs_audit_passed"] else ["docs_not_updated"]),
        ]
    )
    status = "passed" if not reason_codes else "failed"
    approved = status == "passed"
    sandbox_manifest["preflight_verdict"] = EXPECTED_PREFLIGHT_VERDICT if approved else "not_eligible"
    sandbox_manifest["next_required_change"] = (
        NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_sandbox_install_dry_run_preflight_rejections"
    )

    _write_json(paths["sandbox_preflight_manifest"], sandbox_manifest)
    _write_json(paths["package_consumer_audit"], package_consumer_audit)
    _write_json(paths["default_policy_boundary_audit"], default_policy_boundary_audit)
    _write_json(paths["sandbox_path_boundary_audit"], path_boundary_audit)
    _write_json(paths["sandbox_lineage_audit"], lineage_audit)
    _write_json(paths["sandbox_release_boundary_audit"], release_boundary_audit)
    _write_json(paths["rollback_preflight_audit"], rollback_preflight_audit)
    _write_json(paths["rejection_report"], _rejection_report(reason_codes))

    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        approved=approved,
        paths=paths,
        repo_root=repo_root,
        stage11_root=stage11_root,
        stage10_root=stage10_root,
        stage9_root=stage9_root,
        stage8_root=stage8_root,
        stage7_root=stage7_root,
        output_root=output_root,
        stage11_summary=stage11_summary,
        package_consumer_audit=package_consumer_audit,
        sandbox_manifest=sandbox_manifest,
        path_boundary_audit=path_boundary_audit,
        default_policy_boundary_audit=default_policy_boundary_audit,
        lineage_audit=lineage_audit,
        rollback_preflight_audit=rollback_preflight_audit,
        release_boundary_audit=release_boundary_audit,
        docs_audit=docs_audit,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _stage11_gate_audit(stage11_summary: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if (
        stage11_summary.get("status") != "passed"
        or _string_list(stage11_summary.get("reason_codes"))
        or stage11_summary.get("checkpoint_publication_package_verification_passed") is not True
    ):
        _add_reason(reason_codes, "stage11_not_passed")
    if (
        stage11_summary.get("verification_verdict") != EXPECTED_STAGE11_VERDICT
        or stage11_summary.get("checkpoint_publication_sandbox_install_dry_run_preflight_approved") is not True
        or stage11_summary.get("next_required_change") != EXPECTED_STAGE11_NEXT
    ):
        _add_reason(reason_codes, "stage11_not_authorized_for_sandbox_preflight")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "stage11_gate",
        "stage11_gate_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
    }


def _package_consumer_audit(
    *,
    consumer_manifest_path: Path,
    consumer_manifest: dict[str, Any],
    integrity_audit: dict[str, Any],
    load_audit: dict[str, Any],
    stage11_summary: dict[str, Any],
    package_checkpoint_path: Path | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    exists = package_checkpoint_path is not None and package_checkpoint_path.is_file()
    actual_sha = None
    actual_size = None
    consumer_manifest_exists = consumer_manifest_path.is_file() and bool(consumer_manifest)
    if not consumer_manifest_exists:
        _add_reason(reason_codes, "consumer_manifest_missing")
    if not exists:
        _add_reason(reason_codes, "package_checkpoint_missing")
    else:
        assert package_checkpoint_path is not None
        actual_sha, actual_size = _sha256_and_size(package_checkpoint_path)
        expected_shas = _unique(
            [
                str(value)
                for value in (
                    stage11_summary.get("package_checkpoint_sha256"),
                    consumer_manifest.get("package_checkpoint_sha256"),
                    integrity_audit.get("package_checkpoint_sha256"),
                )
                if value
            ]
        )
        expected_sizes = _unique_ints(
            (
                stage11_summary.get("package_checkpoint_size_bytes"),
                consumer_manifest.get("package_checkpoint_size_bytes"),
                integrity_audit.get("package_checkpoint_size_bytes"),
            )
        )
        if any(actual_sha != expected for expected in expected_shas) or any(
            actual_size != expected for expected in expected_sizes
        ):
            _add_reason(reason_codes, "package_identity_mismatch")
    load_verified = (
        consumer_manifest.get("package_load_verification_audit_passed") is True
        and load_audit.get("package_load_verification_audit_passed") is True
        and not _string_list(load_audit.get("reason_codes"))
    )
    if not load_verified:
        _add_reason(reason_codes, "package_load_not_verified")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_package_consumer",
        "package_consumer_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "consumer_manifest_path": str(consumer_manifest_path),
        "consumer_manifest_exists": consumer_manifest_exists,
        "authoritative_package_checkpoint_path": str(package_checkpoint_path) if package_checkpoint_path else None,
        "package_checkpoint_exists": exists,
        "package_checkpoint_sha256": actual_sha,
        "package_checkpoint_size_bytes": actual_size,
        "package_load_verified": load_verified,
    }


def _sandbox_preflight_manifest(
    *,
    output_root: Path,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
    package_consumer_audit: dict[str, Any],
    stage11_summary_path: Path,
    consumer_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    sandbox_root = output_root / "checkpoint-publication-sandbox-install-dry-run-sandbox"
    checkpoint_name = package_checkpoint_path.name if package_checkpoint_path else "experimental-hybrid-policy-candidate.pt"
    metadata_name = (
        package_metadata_path.name if package_metadata_path else "experimental-hybrid-policy-candidate-metadata.json"
    )
    planned_checkpoint = sandbox_root / checkpoint_name
    planned_metadata = sandbox_root / metadata_name
    manifest_passed = (
        _is_relative_to(planned_checkpoint, output_root)
        and _is_relative_to(planned_metadata, output_root)
        and not planned_checkpoint.exists()
        and not planned_metadata.exists()
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "preflight_verdict": "pending",
        "sandbox_preflight_manifest_passed": manifest_passed,
        "stage11_summary": str(stage11_summary_path),
        "stage11_consumer_manifest": str(consumer_manifest_path),
        "sandbox_root": str(sandbox_root),
        "planned_consumer_checkpoint_path": str(planned_checkpoint),
        "planned_consumer_metadata_path": str(planned_metadata),
        "source_package_checkpoint_path": str(package_checkpoint_path) if package_checkpoint_path else None,
        "source_package_metadata_path": str(package_metadata_path) if package_metadata_path else None,
        "package_checkpoint_sha256": package_consumer_audit.get("package_checkpoint_sha256"),
        "package_checkpoint_size_bytes": package_consumer_audit.get("package_checkpoint_size_bytes"),
        "stage12_copies_checkpoint_to_sandbox": False,
        "stage12_runs_rollout": False,
        "stage12_installs_policy": False,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "non_goals": [
            "do_not_copy_checkpoint_to_sandbox_in_preflight",
            "do_not_publish_checkpoint",
            "do_not_replace_default_policy",
            "do_not_connect_real_executor",
        ],
        "next_required_change": "pending",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _path_boundary_audit(*, sandbox_manifest: dict[str, Any], output_root: Path) -> dict[str, Any]:
    checked_paths = {
        "output_root": str(output_root),
        "sandbox_root": sandbox_manifest.get("sandbox_root"),
        "planned_consumer_checkpoint_path": sandbox_manifest.get("planned_consumer_checkpoint_path"),
        "planned_consumer_metadata_path": sandbox_manifest.get("planned_consumer_metadata_path"),
    }
    violations: list[str] = []
    sandbox_root = Path(str(sandbox_manifest.get("sandbox_root")))
    planned_checkpoint = Path(str(sandbox_manifest.get("planned_consumer_checkpoint_path")))
    planned_metadata = Path(str(sandbox_manifest.get("planned_consumer_metadata_path")))
    if not _is_relative_to(sandbox_root, output_root):
        violations.append("sandbox_root_outside_output_root")
    if not _is_relative_to(planned_checkpoint, output_root):
        violations.append("planned_checkpoint_outside_output_root")
    if not _is_relative_to(planned_metadata, output_root):
        violations.append("planned_metadata_outside_output_root")
    for label, raw_path in checked_paths.items():
        if raw_path and _has_unsafe_path_token(Path(raw_path)):
            violations.append(f"{label}_has_unsafe_semantic_token")
    if planned_checkpoint.exists():
        violations.append("planned_checkpoint_already_exists")
    if planned_metadata.exists():
        violations.append("planned_metadata_already_exists")
    reason_codes = ["sandbox_path_boundary_violation"] if violations else []
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_path_boundary",
        "sandbox_path_boundary_audit_passed": not violations,
        "reason_codes": reason_codes,
        "checked_paths": checked_paths,
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


def _rollback_preflight_audit(
    *,
    package_before: dict[str, Any],
    package_after: dict[str, Any],
    metadata_before: dict[str, Any],
    metadata_after: dict[str, Any],
    default_policy_boundary_audit: dict[str, Any],
    sandbox_manifest: dict[str, Any],
    assume_rollback_preflight_valid: bool,
) -> dict[str, Any]:
    planned_checkpoint = Path(str(sandbox_manifest.get("planned_consumer_checkpoint_path")))
    planned_metadata = Path(str(sandbox_manifest.get("planned_consumer_metadata_path")))
    package_unchanged = _snapshots_equal(package_before, package_after)
    metadata_unchanged = _snapshots_equal(metadata_before, metadata_after)
    default_policy_unchanged = default_policy_boundary_audit.get("default_policy_boundary_audit_passed") is True
    planned_checkpoint_absent = not planned_checkpoint.exists()
    planned_metadata_absent = not planned_metadata.exists()
    passed = (
        assume_rollback_preflight_valid
        and package_unchanged
        and metadata_unchanged
        and default_policy_unchanged
        and planned_checkpoint_absent
        and planned_metadata_absent
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_rollback_preflight",
        "rollback_preflight_audit_passed": passed,
        "package_checkpoint_unchanged": package_unchanged,
        "package_metadata_unchanged": metadata_unchanged,
        "default_policy_unchanged": default_policy_unchanged,
        "planned_checkpoint_absent": planned_checkpoint_absent,
        "planned_metadata_absent": planned_metadata_absent,
        "assume_rollback_preflight_valid": assume_rollback_preflight_valid,
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
    stage11_summary: dict[str, Any],
    consumer_manifest: dict[str, Any],
    integrity_audit: dict[str, Any],
    load_audit: dict[str, Any],
    stage11_lineage_audit: dict[str, Any],
    stage11_release_audit: dict[str, Any],
    stage11_rollback_audit: dict[str, Any],
    stage10_manifest: dict[str, Any],
    stage9_summary: dict[str, Any],
    stage8_summary: dict[str, Any],
    stage7_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        _source("stage11_summary", _passed(stage11_summary)),
        _source("stage11_consumer_manifest", consumer_manifest.get("next_required_change") == EXPECTED_STAGE11_NEXT),
        _source("stage11_integrity_audit", _audit_passed(integrity_audit, "package_integrity_audit_passed")),
        _source("stage11_load_audit", _audit_passed(load_audit, "package_load_verification_audit_passed")),
        _source("stage11_lineage_audit", _audit_passed(stage11_lineage_audit, "lineage_verification_audit_passed")),
        _source("stage11_release_audit", _audit_passed(stage11_release_audit, "release_boundary_audit_passed")),
        _source("stage11_rollback_audit", _audit_passed(stage11_rollback_audit, "rollback_verification_audit_passed")),
        _source("stage10_manifest", bool(stage10_manifest)),
        _source("stage9_summary", _passed(stage9_summary)),
        _source("stage8_summary", _passed(stage8_summary)),
        _source("stage7_summary", _passed(stage7_summary)),
    ]
    passed = not read_reasons and all(source["passed"] for source in sources)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "sandbox_lineage",
        "lineage_audit_passed": passed,
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


def _docs_audit(*, repo_root: Path) -> dict[str, Any]:
    required_docs = {
        "README.md": repo_root / "README.md",
        "docs/算法设计与系统架构报告.md": repo_root / "docs" / "算法设计与系统架构报告.md",
        "docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run-preflight.md": repo_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-16-checkpoint-publication-sandbox-install-dry-run-preflight.md",
    }
    required_phrases = {
        "README.md": [
            "Stage 12 `Checkpoint Publication Sandbox Install Dry-Run Preflight v1`",
            "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1/",
            "checkpoint_publication_sandbox_install_dry_run",
        ],
        "docs/算法设计与系统架构报告.md": [
            "阶段 12 `Checkpoint Publication Sandbox Install Dry-Run Preflight v1`",
            "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1/",
            "checkpoint_publication_sandbox_install_dry_run",
        ],
        "docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run-preflight.md": [
            "checkpoint-publication-sandbox-install-dry-run-preflight-summary.json",
            "README.md",
            "docs/算法设计与系统架构报告.md",
            "checkpoint_publication_sandbox_install_dry_run",
            "不复制到发布/default/live/executor 路径",
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
    stage11_root: Path,
    stage10_root: Path,
    stage9_root: Path,
    stage8_root: Path,
    stage7_root: Path,
    output_root: Path,
    stage11_summary: dict[str, Any],
    package_consumer_audit: dict[str, Any],
    sandbox_manifest: dict[str, Any],
    path_boundary_audit: dict[str, Any],
    default_policy_boundary_audit: dict[str, Any],
    lineage_audit: dict[str, Any],
    rollback_preflight_audit: dict[str, Any],
    release_boundary_audit: dict[str, Any],
    docs_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "preflight_verdict": EXPECTED_PREFLIGHT_VERDICT if approved else _failed_verdict(reason_codes),
        "checkpoint_publication_sandbox_install_dry_run_preflight_passed": approved,
        "checkpoint_publication_sandbox_install_dry_run_approved": approved,
        "sandbox_preflight_manifest_passed": sandbox_manifest.get("sandbox_preflight_manifest_passed") is True,
        "package_consumer_audit_passed": package_consumer_audit.get("package_consumer_audit_passed") is True,
        "default_policy_boundary_audit_passed": default_policy_boundary_audit.get("default_policy_boundary_audit_passed") is True,
        "sandbox_path_boundary_audit_passed": path_boundary_audit.get("sandbox_path_boundary_audit_passed") is True,
        "lineage_audit_passed": lineage_audit.get("lineage_audit_passed") is True,
        "rollback_preflight_audit_passed": rollback_preflight_audit.get("rollback_preflight_audit_passed") is True,
        "release_boundary_audit_passed": release_boundary_audit.get("release_boundary_audit_passed") is True,
        "docs_audit_passed": docs_audit.get("docs_audit_passed") is True,
        "selected_seed": stage11_summary.get("selected_seed"),
        "selected_budget": stage11_summary.get("selected_budget"),
        "package_checkpoint_sha256": package_consumer_audit.get("package_checkpoint_sha256"),
        "package_checkpoint_size_bytes": package_consumer_audit.get("package_checkpoint_size_bytes"),
        "sandbox_root": sandbox_manifest.get("sandbox_root"),
        "planned_consumer_checkpoint_path": sandbox_manifest.get("planned_consumer_checkpoint_path"),
        "default_policy_configured": default_policy_boundary_audit.get("default_policy_configured") is True,
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_checkpoint_publication_sandbox_install_dry_run_preflight_rejections",
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
        "stage11_root": str(stage11_root),
        "stage10_root": str(stage10_root),
        "stage9_root": str(stage9_root),
        "stage8_root": str(stage8_root),
        "stage7_root": str(stage7_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "sandbox_preflight_manifest": str(paths["sandbox_preflight_manifest"]),
        "package_consumer_audit": str(paths["package_consumer_audit"]),
        "default_policy_boundary_audit": str(paths["default_policy_boundary_audit"]),
        "sandbox_path_boundary_audit": str(paths["sandbox_path_boundary_audit"]),
        "sandbox_lineage_audit": str(paths["sandbox_lineage_audit"]),
        "sandbox_release_boundary_audit": str(paths["sandbox_release_boundary_audit"]),
        "rollback_preflight_audit": str(paths["rollback_preflight_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Checkpoint Publication Sandbox Install Dry-Run Preflight v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Preflight verdict: `{summary['preflight_verdict']}`\n"
        f"- Selected seed: `{summary.get('selected_seed')}`\n"
        f"- Selected budget: `{summary.get('selected_budget')}`\n"
        f"- Package checkpoint SHA-256: `{summary.get('package_checkpoint_sha256')}`\n"
        f"- Sandbox root: `{summary.get('sandbox_root')}`\n\n"
        "## Release Boundaries\n\n"
        f"- Checkpoint publication approved: `{summary['checkpoint_publication_approved']}`\n"
        f"- Default policy replacement approved: `{summary['default_policy_replacement_approved']}`\n"
        f"- Real executor connection approved: `{summary['real_executor_connection_approved']}`\n"
        f"- Next required change: `{summary['next_required_change']}`\n"
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "sandbox_preflight_manifest": output_root / SANDBOX_MANIFEST_FILE,
        "package_consumer_audit": output_root / PACKAGE_CONSUMER_AUDIT_FILE,
        "default_policy_boundary_audit": output_root / DEFAULT_POLICY_BOUNDARY_AUDIT_FILE,
        "sandbox_path_boundary_audit": output_root / PATH_BOUNDARY_AUDIT_FILE,
        "sandbox_lineage_audit": output_root / LINEAGE_AUDIT_FILE,
        "sandbox_release_boundary_audit": output_root / RELEASE_BOUNDARY_AUDIT_FILE,
        "rollback_preflight_audit": output_root / ROLLBACK_PREFLIGHT_AUDIT_FILE,
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


def _resolve_optional_path(value: Any, root: Path, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return _resolve_path(Path(value), repo_root if not Path(value).is_absolute() else root)


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
