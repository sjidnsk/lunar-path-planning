from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
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


CONFIG_SCHEMA_VERSION = "guarded-experimental-policy-release-candidate-packaging-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-experimental-policy-release-candidate-packaging-summary/v1"
MANIFEST_SCHEMA_VERSION = (
    "guarded-experimental-policy-release-candidate-package-manifest/v1"
)
DECISION_SCHEMA_VERSION = (
    "selected-formal-ppo-candidate-promotion-decision-review-summary/v1"
)
CHECKPOINT_METADATA_SCHEMA_VERSION = "controlled-hybrid-policy-candidate-checkpoint-metadata/v1"
EXPECTED_DECISION_VERDICT = "eligible_for_guarded_release_candidate_packaging"
EXPECTED_PACKAGE_VERDICT = "eligible_for_guarded_install_dry_run"
EXPECTED_READINESS_STATUS = (
    "guarded_experimental_policy_release_candidate_packaging_evaluated"
)

SUMMARY_FILE = "guarded-experimental-policy-release-candidate-packaging-summary.json"
MANIFEST_FILE = "release-candidate-package-manifest.json"
CHECKPOINT_HASH_AUDIT_FILE = "checkpoint-hash-audit.json"
CHECKPOINT_LOAD_AUDIT_FILE = "checkpoint-load-audit.json"
ROLLBACK_AUDIT_FILE = "rollback-audit.json"
READINESS_FILE = "packaging-readiness-validate-only.json"
REPORT_FILE = "release-candidate-packaging-report.md"
PACKAGE_DIR = "release-candidate-package"

ReadinessRunner = Callable[..., dict[str, Any]]
LoadAuditRunner = Callable[..., dict[str, Any]]


def run_guarded_experimental_policy_release_candidate_packaging(
    *,
    decision_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
    load_audit_runner: LoadAuditRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    decision_root = Path(decision_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)
    package_root = output_root / PACKAGE_DIR
    package_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    manifest_path = output_root / files["package_manifest"]
    checkpoint_hash_audit_path = output_root / files["checkpoint_hash_audit"]
    checkpoint_load_audit_path = output_root / files["checkpoint_load_audit"]
    rollback_audit_path = output_root / files["rollback_audit"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    inputs = _input_files(config)
    decision_summary_path = decision_root / inputs["decision_review_summary"]
    decision_summary = _read_json_if_exists(decision_summary_path)
    checkpoint_identity_source = _read_json_if_exists(
        _resolve_optional_path(
            decision_summary.get("checkpoint_identity_audit"),
            decision_root,
            repo_root,
        )
    )
    release_boundary_source = _read_json_if_exists(
        _resolve_optional_path(
            decision_summary.get("release_boundary_audit"),
            decision_root,
            repo_root,
        )
    )
    lineage_source = _read_json_if_exists(
        _resolve_optional_path(decision_summary.get("lineage_report"), decision_root, repo_root)
    )
    preflight_summary_path = _resolve_optional_path(
        decision_summary.get("preflight_summary"),
        decision_root,
        repo_root,
    )
    preflight_summary = _read_json_if_exists(preflight_summary_path)

    reason_codes: list[str] = []
    _validate_decision_review(
        decision_summary=decision_summary,
        checkpoint_identity_source=checkpoint_identity_source,
        release_boundary_source=release_boundary_source,
        lineage_source=lineage_source,
        reason_codes=reason_codes,
    )

    original_checkpoint_path = _resolve_optional_path(
        decision_summary.get("checkpoint_path"),
        decision_root,
        repo_root,
    )
    original_metadata_path = _resolve_optional_path(
        decision_summary.get("checkpoint_metadata_path"),
        decision_root,
        repo_root,
    )
    checkpoint_metadata = _read_json_if_exists(original_metadata_path)
    checkpoint_hash_audit = _checkpoint_hash_audit(
        original_checkpoint_path=original_checkpoint_path,
        expected_sha256=decision_summary.get("checkpoint_sha256"),
        expected_size=_int(decision_summary.get("checkpoint_size_bytes")),
        package_root=package_root,
    )
    for reason in checkpoint_hash_audit["reason_codes"]:
        _add_reason(reason_codes, reason)

    package_checkpoint_path: Path | None = None
    package_metadata_path: Path | None = None
    if checkpoint_hash_audit["checkpoint_identity_audit_passed"]:
        assert original_checkpoint_path is not None
        package_checkpoint_path = package_root / original_checkpoint_path.name
        shutil.copy2(original_checkpoint_path, package_checkpoint_path)
        if original_metadata_path and original_metadata_path.is_file():
            package_metadata_path = package_root / original_metadata_path.name
            shutil.copy2(original_metadata_path, package_metadata_path)
        package_sha256, package_size = _sha256_and_size(package_checkpoint_path)
        checkpoint_hash_audit["package_checkpoint_path"] = str(package_checkpoint_path)
        checkpoint_hash_audit["package_checkpoint_exists"] = True
        checkpoint_hash_audit["package_checkpoint_sha256"] = package_sha256
        checkpoint_hash_audit["package_checkpoint_size_bytes"] = package_size
        if package_sha256 != checkpoint_hash_audit.get("checkpoint_sha256"):
            _add_reason(reason_codes, "release_candidate_package_hash_mismatch")
            checkpoint_hash_audit["checkpoint_identity_audit_passed"] = False
        if package_size != checkpoint_hash_audit.get("checkpoint_size_bytes"):
            _add_reason(reason_codes, "release_candidate_package_size_mismatch")
            checkpoint_hash_audit["checkpoint_identity_audit_passed"] = False
    _write_json(checkpoint_hash_audit_path, checkpoint_hash_audit)

    manifest = _package_manifest(
        repo_root=repo_root,
        output_root=output_root,
        package_root=package_root,
        decision_root=decision_root,
        decision_summary_path=decision_summary_path,
        preflight_summary_path=preflight_summary_path,
        manifest_path=manifest_path,
        checkpoint_hash_audit_path=checkpoint_hash_audit_path,
        checkpoint_load_audit_path=checkpoint_load_audit_path,
        rollback_audit_path=rollback_audit_path,
        decision_summary=decision_summary,
        checkpoint_hash_audit=checkpoint_hash_audit,
        original_checkpoint_path=original_checkpoint_path,
        original_metadata_path=original_metadata_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        lineage_source=lineage_source,
    )
    _write_json(manifest_path, manifest)

    if package_checkpoint_path is not None and checkpoint_hash_audit[
        "checkpoint_identity_audit_passed"
    ]:
        runner = load_audit_runner or _checkpoint_load_audit
        checkpoint_load_audit = runner(
            checkpoint_path=package_checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            preflight_summary=preflight_summary,
            config=config,
            repo_root=repo_root,
        )
    else:
        checkpoint_load_audit = {
            "checkpoint_load_passed": False,
            "checkpoint_load_sample_count": 0,
            "invalid_action_mask_count": 0,
            "missing_observation_count": 0,
            "non_finite_logits_count": 0,
            "non_finite_log_prob_count": 0,
            "non_finite_value_count": 0,
            "sampled_rows": [],
            "checkpoint_error": "checkpoint identity failed before load audit",
        }
    _write_json(checkpoint_load_audit_path, checkpoint_load_audit)
    _validate_load_audit(checkpoint_load_audit, config, reason_codes)

    rollback_audit = _rollback_audit(
        decision_summary=decision_summary,
        checkpoint_metadata=checkpoint_metadata,
        manifest=manifest,
        output_root=output_root,
        package_checkpoint_path=package_checkpoint_path,
    )
    _write_json(rollback_audit_path, rollback_audit)
    if not rollback_audit["rollback_audit_passed"]:
        _add_reason(reason_codes, "release_candidate_rollback_boundary_invalid")

    package_verdict = _package_verdict(reason_codes)
    pre_readiness_summary = _summary_payload(
        status="passed" if not reason_codes else "failed",
        reason_codes=reason_codes,
        package_verdict=package_verdict,
        repo_root=repo_root,
        decision_root=decision_root,
        output_root=output_root,
        package_root=package_root,
        batch_root=batch_root,
        decision_summary_path=decision_summary_path,
        preflight_summary_path=preflight_summary_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        checkpoint_hash_audit_path=checkpoint_hash_audit_path,
        checkpoint_load_audit_path=checkpoint_load_audit_path,
        rollback_audit_path=rollback_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        decision_summary=decision_summary,
        checkpoint_hash_audit=checkpoint_hash_audit,
        checkpoint_load_audit=checkpoint_load_audit,
        rollback_audit=rollback_audit,
        readiness={},
        original_checkpoint_path=original_checkpoint_path,
        package_checkpoint_path=package_checkpoint_path,
        original_metadata_path=original_metadata_path,
        package_metadata_path=package_metadata_path,
    )
    _write_json(summary_path, pre_readiness_summary)

    if not reason_codes:
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            packaging_summary_path=summary_path,
            config_path=Path(
                config.get("readiness", {}).get(
                    "config",
                    "configs/policy_training_readiness_review_v1.json",
                )
            ),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_guarded_experimental_policy_release_candidate_packaging",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    package_verdict = _package_verdict(reason_codes)
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        package_verdict=package_verdict,
        repo_root=repo_root,
        decision_root=decision_root,
        output_root=output_root,
        package_root=package_root,
        batch_root=batch_root,
        decision_summary_path=decision_summary_path,
        preflight_summary_path=preflight_summary_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        checkpoint_hash_audit_path=checkpoint_hash_audit_path,
        checkpoint_load_audit_path=checkpoint_load_audit_path,
        rollback_audit_path=rollback_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        decision_summary=decision_summary,
        checkpoint_hash_audit=checkpoint_hash_audit,
        checkpoint_load_audit=checkpoint_load_audit,
        rollback_audit=rollback_audit,
        readiness=readiness,
        original_checkpoint_path=original_checkpoint_path,
        package_checkpoint_path=package_checkpoint_path,
        original_metadata_path=original_metadata_path,
        package_metadata_path=package_metadata_path,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package a reviewed experimental policy as a guarded release candidate."
    )
    parser.add_argument(
        "--decision-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/guarded_experimental_policy_release_candidate_packaging_v1.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(Path(args.config), repo_root)
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"invalid config schema: {config.get('schema_version')}")
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": str(args.config)}, sort_keys=True))
        return 0

    summary = run_guarded_experimental_policy_release_candidate_packaging(
        decision_root=_resolve_path(Path(args.decision_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        batch_root=_resolve_path(Path(args.batch_root), repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "package_verdict": summary["package_verdict"],
                "readiness_status": summary.get("readiness_status"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _validate_decision_review(
    *,
    decision_summary: dict[str, Any],
    checkpoint_identity_source: dict[str, Any],
    release_boundary_source: dict[str, Any],
    lineage_source: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if decision_summary.get("schema_version") != DECISION_SCHEMA_VERSION:
        _add_reason(reason_codes, "release_candidate_decision_review_schema_invalid")
    if decision_summary.get("status") != "passed" or _string_list(
        decision_summary.get("reason_codes")
    ):
        _add_reason(reason_codes, "release_candidate_decision_review_not_passed")
    if decision_summary.get("decision_verdict") != EXPECTED_DECISION_VERDICT:
        _add_reason(reason_codes, "release_candidate_decision_review_not_eligible")
    if decision_summary.get("readiness_status") != "selected_formal_ppo_candidate_promotion_decision_review_evaluated":
        _add_reason(reason_codes, "release_candidate_decision_review_readiness_invalid")
    if decision_summary.get("lineage_audit_passed") is not True:
        _add_reason(reason_codes, "release_candidate_lineage_failed")
    if _int(decision_summary.get("source_lineage_count")) < 4:
        _add_reason(reason_codes, "release_candidate_lineage_incomplete")
    if decision_summary.get("checkpoint_identity_audit_passed") is not True:
        _add_reason(reason_codes, "release_candidate_checkpoint_identity_source_failed")
    if decision_summary.get("release_boundary_audit_passed") is not True:
        _add_reason(reason_codes, "release_candidate_release_boundary_source_failed")
    if _git_current_matches_sources(decision_summary) is False:
        _add_reason(reason_codes, "release_candidate_decision_review_git_provenance_mismatch")
    if checkpoint_identity_source and checkpoint_identity_source.get("checkpoint_identity_audit_passed") is not True:
        _add_reason(reason_codes, "release_candidate_checkpoint_identity_source_failed")
    if release_boundary_source and release_boundary_source.get("release_boundary_audit_passed") is not True:
        _add_reason(reason_codes, "release_candidate_release_boundary_source_failed")
    if lineage_source:
        sources = lineage_source.get("sources") or []
        if len(sources) < 4 or not all(source.get("passed") for source in sources):
            _add_reason(reason_codes, "release_candidate_lineage_failed")
    for field, reason in (
        ("runs_new_ppo_update", "release_candidate_unexpected_ppo_update"),
        ("publishes_checkpoint", "release_candidate_checkpoint_publication_claimed"),
        ("replaces_default_policy", "release_candidate_default_policy_replacement_claimed"),
        ("performance_claimed", "release_candidate_policy_performance_claimed"),
        ("formal_training_ready_claimed", "release_candidate_formal_ready_claimed"),
    ):
        if decision_summary.get(field) is True:
            _add_reason(reason_codes, reason)


def _checkpoint_hash_audit(
    *,
    original_checkpoint_path: Path | None,
    expected_sha256: str | None,
    expected_size: int,
    package_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if original_checkpoint_path is None or not original_checkpoint_path.is_file():
        _add_reason(reason_codes, "release_candidate_checkpoint_missing")
        checkpoint_sha256 = None
        checkpoint_size = 0
    else:
        checkpoint_sha256, checkpoint_size = _sha256_and_size(original_checkpoint_path)
    if checkpoint_sha256 and expected_sha256 and checkpoint_sha256 != expected_sha256:
        _add_reason(reason_codes, "release_candidate_checkpoint_hash_mismatch")
    if checkpoint_size and expected_size and checkpoint_size != expected_size:
        _add_reason(reason_codes, "release_candidate_checkpoint_size_mismatch")
    if not expected_sha256:
        _add_reason(reason_codes, "release_candidate_checkpoint_hash_missing")
    if expected_size <= 0:
        _add_reason(reason_codes, "release_candidate_checkpoint_size_missing")
    return {
        "schema_version": "guarded-experimental-policy-release-candidate-checkpoint-hash-audit/v1",
        "checkpoint_identity_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "original_checkpoint_path": None
        if original_checkpoint_path is None
        else str(original_checkpoint_path),
        "checkpoint_exists": bool(original_checkpoint_path and original_checkpoint_path.is_file()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size,
        "expected_checkpoint_sha256": expected_sha256,
        "expected_checkpoint_size_bytes": expected_size,
        "package_root": str(package_root),
        "package_checkpoint_path": None,
        "package_checkpoint_exists": False,
        "package_checkpoint_sha256": None,
        "package_checkpoint_size_bytes": 0,
    }


def _checkpoint_load_audit(
    *,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    preflight_summary: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    try:
        from run_selected_formal_ppo_candidate_promotion_preflight import (
            _checkpoint_inference_audit,
            _install_model_explorer_path,
            _read_jsonl,
        )

        _install_model_explorer_path(repo_root)
        steps_path = _resolve_optional_path(
            preflight_summary.get("multihorizon_steps"),
            repo_root,
            repo_root,
        )
        if steps_path is None:
            return {
                "checkpoint_load_passed": False,
                "checkpoint_load_sample_count": 0,
                "invalid_action_mask_count": 0,
                "missing_observation_count": 0,
                "non_finite_logits_count": 0,
                "non_finite_log_prob_count": 0,
                "non_finite_value_count": 0,
                "sampled_rows": [],
                "checkpoint_error": "multihorizon steps missing from preflight summary",
            }
        steps = _read_jsonl(steps_path)
        audit = _checkpoint_inference_audit(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            steps=steps,
            config={"validation": {"min_inference_audit_count": _min_load_sample_count(config)}},
        )
        return {
            **audit,
            "schema_version": "guarded-experimental-policy-release-candidate-checkpoint-load-audit/v1",
            "checkpoint_load_sample_count": _int(audit.get("inference_audit_count")),
        }
    except Exception as exc:  # noqa: BLE001 - surfaced in summary.
        return {
            "schema_version": "guarded-experimental-policy-release-candidate-checkpoint-load-audit/v1",
            "checkpoint_load_passed": False,
            "checkpoint_load_sample_count": 0,
            "invalid_action_mask_count": 0,
            "missing_observation_count": 0,
            "non_finite_logits_count": 0,
            "non_finite_log_prob_count": 0,
            "non_finite_value_count": 0,
            "sampled_rows": [],
            "checkpoint_error": str(exc),
        }


def _validate_load_audit(
    checkpoint_load_audit: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if checkpoint_load_audit.get("checkpoint_load_passed") is not True:
        _add_reason(reason_codes, "release_candidate_checkpoint_load_failed")
    if _int(checkpoint_load_audit.get("checkpoint_load_sample_count")) < _min_load_sample_count(config):
        _add_reason(reason_codes, "release_candidate_checkpoint_load_sample_below_threshold")
    for field, reason in (
        ("invalid_action_mask_count", "release_candidate_invalid_action_mask"),
        ("missing_observation_count", "release_candidate_missing_observation"),
        ("non_finite_logits_count", "release_candidate_non_finite_inference"),
        ("non_finite_log_prob_count", "release_candidate_non_finite_inference"),
        ("non_finite_value_count", "release_candidate_non_finite_inference"),
    ):
        if _int(checkpoint_load_audit.get(field)) > 0:
            _add_reason(reason_codes, reason)


def _rollback_audit(
    *,
    decision_summary: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    manifest: dict[str, Any],
    output_root: Path,
    package_checkpoint_path: Path | None,
) -> dict[str, Any]:
    sources = {
        "decision_summary": decision_summary,
        "checkpoint_metadata": checkpoint_metadata,
        "package_manifest": manifest,
    }
    publication_flags = {
        name: {
            "runs_new_ppo_update": payload.get("runs_new_ppo_update") is True,
            "executes_install_or_canary": payload.get("executes_install_or_canary") is True,
            "publishes_checkpoint": payload.get("publishes_checkpoint") is True,
            "replaces_default_policy": payload.get("replaces_default_policy") is True
            or payload.get("default_policy_replaced") is True,
            "performance_claimed": payload.get("performance_claimed") is True,
            "formal_training_ready_claimed": payload.get("formal_training_ready_claimed") is True,
        }
        for name, payload in sources.items()
    }
    package_deletable = package_checkpoint_path is not None and output_root in package_checkpoint_path.parents
    rollback_source_traceable = bool(
        decision_summary.get("selected_candidate_root") and decision_summary.get("checkpoint_path")
    )
    rollback_audit_passed = (
        package_deletable
        and rollback_source_traceable
        and not any(any(flags.values()) for flags in publication_flags.values())
    )
    return {
        "schema_version": "guarded-experimental-policy-release-candidate-rollback-audit/v1",
        "rollback_audit_passed": rollback_audit_passed,
        "package_deletable": package_deletable,
        "rollback_source_traceable": rollback_source_traceable,
        "recommended_rollback_source": decision_summary.get("selected_candidate_root"),
        "default_policy_replaced": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "publication_flags": publication_flags,
    }


def _package_manifest(
    *,
    repo_root: Path,
    output_root: Path,
    package_root: Path,
    decision_root: Path,
    decision_summary_path: Path,
    preflight_summary_path: Path | None,
    manifest_path: Path,
    checkpoint_hash_audit_path: Path,
    checkpoint_load_audit_path: Path,
    rollback_audit_path: Path,
    decision_summary: dict[str, Any],
    checkpoint_hash_audit: dict[str, Any],
    original_checkpoint_path: Path | None,
    original_metadata_path: Path | None,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
    lineage_source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "experimental_candidate_only": True,
        "selected_seed": decision_summary.get("selected_seed"),
        "selected_budget": decision_summary.get("selected_budget"),
        "selected_candidate_root": decision_summary.get("selected_candidate_root"),
        "decision_root": str(decision_root),
        "decision_review_summary": str(decision_summary_path),
        "preflight_summary": None if preflight_summary_path is None else str(preflight_summary_path),
        "output_root": str(output_root),
        "package_root": str(package_root),
        "manifest": str(manifest_path),
        "checkpoint_hash_audit": str(checkpoint_hash_audit_path),
        "checkpoint_load_audit": str(checkpoint_load_audit_path),
        "rollback_audit": str(rollback_audit_path),
        "original_checkpoint_path": None if original_checkpoint_path is None else str(original_checkpoint_path),
        "original_checkpoint_sha256": checkpoint_hash_audit.get("checkpoint_sha256"),
        "original_checkpoint_size_bytes": _int(
            checkpoint_hash_audit.get("checkpoint_size_bytes")
        ),
        "original_checkpoint_metadata_path": None
        if original_metadata_path is None
        else str(original_metadata_path),
        "package_checkpoint_path": None if package_checkpoint_path is None else str(package_checkpoint_path),
        "package_checkpoint_sha256": checkpoint_hash_audit.get("package_checkpoint_sha256"),
        "package_checkpoint_size_bytes": _int(
            checkpoint_hash_audit.get("package_checkpoint_size_bytes")
        ),
        "package_checkpoint_metadata_path": None
        if package_metadata_path is None
        else str(package_metadata_path),
        "source_lineage": _manifest_source_lineage(lineage_source),
        "source_lineage_count": len(lineage_source.get("sources") or []),
        "rollback_source_traceable": bool(
            decision_summary.get("selected_candidate_root")
            and decision_summary.get("checkpoint_path")
        ),
        "runs_new_ppo_update": False,
        "executes_install_or_canary": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    package_verdict: str,
    repo_root: Path,
    decision_root: Path,
    output_root: Path,
    package_root: Path,
    batch_root: Path,
    decision_summary_path: Path,
    preflight_summary_path: Path | None,
    summary_path: Path,
    manifest_path: Path,
    checkpoint_hash_audit_path: Path,
    checkpoint_load_audit_path: Path,
    rollback_audit_path: Path,
    readiness_path: Path,
    report_path: Path,
    decision_summary: dict[str, Any],
    checkpoint_hash_audit: dict[str, Any],
    checkpoint_load_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    readiness: dict[str, Any],
    original_checkpoint_path: Path | None,
    package_checkpoint_path: Path | None,
    original_metadata_path: Path | None,
    package_metadata_path: Path | None,
) -> dict[str, Any]:
    package_sha256 = checkpoint_hash_audit.get("package_checkpoint_sha256")
    package_size = _int(checkpoint_hash_audit.get("package_checkpoint_size_bytes"))
    checkpoint_sha256 = checkpoint_hash_audit.get("checkpoint_sha256")
    checkpoint_size = _int(checkpoint_hash_audit.get("checkpoint_size_bytes"))
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_guarded_experimental_policy_release_candidate_packaging",
        "package_verdict": package_verdict,
        "decision_root": str(decision_root),
        "decision_review_summary": str(decision_summary_path),
        "preflight_summary": None if preflight_summary_path is None else str(preflight_summary_path),
        "output_root": str(output_root),
        "package_root": str(package_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "package_manifest": str(manifest_path),
        "checkpoint_hash_audit": str(checkpoint_hash_audit_path),
        "checkpoint_load_audit": str(checkpoint_load_audit_path),
        "rollback_audit": str(rollback_audit_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "selected_seed": decision_summary.get("selected_seed"),
        "selected_budget": decision_summary.get("selected_budget"),
        "selected_candidate_root": decision_summary.get("selected_candidate_root"),
        "original_checkpoint_path": None if original_checkpoint_path is None else str(original_checkpoint_path),
        "original_checkpoint_metadata_path": None
        if original_metadata_path is None
        else str(original_metadata_path),
        "package_checkpoint_path": None if package_checkpoint_path is None else str(package_checkpoint_path),
        "package_checkpoint_metadata_path": None
        if package_metadata_path is None
        else str(package_metadata_path),
        "checkpoint_sha256": checkpoint_sha256,
        "package_checkpoint_sha256": package_sha256,
        "checkpoint_size_bytes": checkpoint_size,
        "package_checkpoint_size_bytes": package_size,
        "checkpoint_identity_audit_passed": bool(
            checkpoint_hash_audit.get("checkpoint_identity_audit_passed")
        )
        and checkpoint_sha256 == package_sha256
        and checkpoint_size == package_size
        and checkpoint_size > 0,
        "checkpoint_load_passed": checkpoint_load_audit.get("checkpoint_load_passed") is True,
        "checkpoint_load_sample_count": _int(
            checkpoint_load_audit.get("checkpoint_load_sample_count")
        ),
        "invalid_action_mask_count": _int(checkpoint_load_audit.get("invalid_action_mask_count")),
        "missing_observation_count": _int(checkpoint_load_audit.get("missing_observation_count")),
        "non_finite_logits_count": _int(checkpoint_load_audit.get("non_finite_logits_count")),
        "non_finite_log_prob_count": _int(
            checkpoint_load_audit.get("non_finite_log_prob_count")
        ),
        "non_finite_value_count": _int(checkpoint_load_audit.get("non_finite_value_count")),
        "rollback_audit_passed": rollback_audit.get("rollback_audit_passed") is True,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_release_candidate_packaging": True,
        "runs_new_ppo_update": False,
        "executes_install_or_canary": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "recommended_next_action": "guarded_install_canary_dry_run"
        if status == "passed"
        else "fix_guarded_experimental_policy_release_candidate_packaging",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _manifest_source_lineage(lineage_source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in lineage_source.get("sources") or []:
        if not isinstance(source, dict):
            continue
        rows.append(
            {
                "name": source.get("name"),
                "path": source.get("path"),
                "schema_version": source.get("schema_version"),
                "expected_schema_version": source.get("expected_schema_version"),
                "status": source.get("status"),
                "reason_codes": _string_list(source.get("reason_codes")),
                "git_current_matches_sources": source.get("git_current_matches_sources"),
                "passed": source.get("passed") is True,
            }
        )
    return rows


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    packaging_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-experimental-policy-release-candidate-packaging-summary",
        str(packaging_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    try:
        result = json.loads(first_line)
    except json.JSONDecodeError:
        return {
            "training_readiness_status": "readiness_validate_only_unparseable",
            "reason_codes": ["readiness_validate_only_stdout_unparseable"],
            "training_blockers": [completed.stderr.strip() or completed.stdout[:1000]],
            "command": command,
            "returncode": completed.returncode,
        }
    result["command"] = command
    result["returncode"] = completed.returncode
    return result


def _validate_readiness(
    readiness: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = config.get("readiness", {}).get("expected_status", EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "release_candidate_packaging_readiness_status_mismatch")
    if _string_list(readiness.get("training_blockers")):
        _add_reason(reason_codes, "release_candidate_packaging_readiness_blocked")
    if _string_list(readiness.get("reason_codes")):
        _add_reason(reason_codes, "release_candidate_packaging_readiness_reason_codes")
    if _int(readiness.get("returncode")) not in (0,):
        _add_reason(reason_codes, "release_candidate_packaging_readiness_command_failed")


def _package_verdict(reason_codes: list[str]) -> str:
    if not reason_codes:
        return EXPECTED_PACKAGE_VERDICT
    if any(
        reason in reason_codes
        for reason in (
            "release_candidate_checkpoint_missing",
            "release_candidate_checkpoint_hash_mismatch",
            "release_candidate_checkpoint_size_mismatch",
            "release_candidate_package_hash_mismatch",
            "release_candidate_package_size_mismatch",
        )
    ):
        return "blocked_by_checkpoint_identity"
    if any(
        reason in reason_codes
        for reason in (
            "release_candidate_release_boundary_source_failed",
            "release_candidate_rollback_boundary_invalid",
            "release_candidate_checkpoint_publication_claimed",
            "release_candidate_default_policy_replacement_claimed",
        )
    ):
        return "blocked_by_release_boundary"
    return "blocked_by_decision_review_or_load_audit"


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Experimental Policy Release Candidate Packaging v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- package_verdict: `{summary['package_verdict']}`",
            f"- selected_seed: `{summary.get('selected_seed')}`",
            f"- selected_budget: `{summary.get('selected_budget')}`",
            f"- checkpoint_sha256: `{summary.get('checkpoint_sha256')}`",
            f"- package_checkpoint_sha256: `{summary.get('package_checkpoint_sha256')}`",
            f"- checkpoint_load_sample_count: `{summary.get('checkpoint_load_sample_count')}`",
            f"- readiness_status: `{summary.get('readiness_status')}`",
            "",
            "This stage packages an experimental candidate for a later guarded install dry-run. "
            "It does not install, publish, or replace any default policy.",
            "",
        ]
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    output_files = config.get("output_files", {})
    defaults = {
        "summary": SUMMARY_FILE,
        "package_manifest": MANIFEST_FILE,
        "checkpoint_hash_audit": CHECKPOINT_HASH_AUDIT_FILE,
        "checkpoint_load_audit": CHECKPOINT_LOAD_AUDIT_FILE,
        "rollback_audit": ROLLBACK_AUDIT_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    return {key: str(output_files.get(key, value)) for key, value in defaults.items()}


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    input_files = config.get("input_files", {})
    return {
        "decision_review_summary": str(
            input_files.get(
                "decision_review_summary",
                "selected-formal-ppo-candidate-promotion-decision-review-summary.json",
            )
        )
    }


def _min_load_sample_count(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("min_load_sample_count"), 64)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return _read_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _resolve_optional_path(
    value: Any,
    base: Path,
    repo_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return repo_root / path


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _git_current_matches_sources(payload: dict[str, Any]) -> bool | None:
    provenance = payload.get("git_provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("current_matches_sources")
    return value if isinstance(value, bool) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
