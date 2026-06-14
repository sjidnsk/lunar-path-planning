from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
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


CONFIG_SCHEMA_VERSION = "selected-formal-ppo-candidate-promotion-decision-review-config/v1"
SUMMARY_SCHEMA_VERSION = "selected-formal-ppo-candidate-promotion-decision-review-summary/v1"
PREFLIGHT_SCHEMA_VERSION = "selected-formal-ppo-candidate-promotion-preflight-summary/v1"
PROMOTION_MANIFEST_SCHEMA_VERSION = "selected-formal-ppo-candidate-promotion-manifest/v1"
MULTIHORIZON_SCHEMA_VERSION = (
    "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1"
)
SELECTION_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1"
)
STABILITY_SCHEMA_VERSION = "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1"
CHECKPOINT_METADATA_SCHEMA_VERSION = "controlled-hybrid-policy-candidate-checkpoint-metadata/v1"
EXPECTED_READINESS_STATUS = "selected_formal_ppo_candidate_promotion_decision_review_evaluated"

SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-decision-review-summary.json"
LINEAGE_FILE = "evidence-lineage-report.json"
CHECKPOINT_IDENTITY_FILE = "checkpoint-identity-audit.json"
RELEASE_BOUNDARY_FILE = "release-boundary-audit.json"
READINESS_FILE = "promotion-decision-readiness-validate-only.json"
REPORT_FILE = "promotion-decision-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_selected_formal_ppo_candidate_promotion_decision_review(
    *,
    preflight_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    preflight_root = Path(preflight_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    lineage_path = output_root / files["lineage_report"]
    checkpoint_identity_path = output_root / files["checkpoint_identity_audit"]
    release_boundary_path = output_root / files["release_boundary_audit"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    inputs = _input_files(config)
    preflight_summary_path = preflight_root / inputs["preflight_summary"]
    preflight_summary = _read_json_if_exists(preflight_summary_path)

    reason_codes: list[str] = []
    if preflight_summary.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        _add_reason(reason_codes, "promotion_decision_preflight_schema_invalid")
    if preflight_summary.get("status") != "passed" or _string_list(
        preflight_summary.get("reason_codes")
    ):
        _add_reason(reason_codes, "promotion_decision_preflight_not_passed")
    if preflight_summary.get("readiness_status") != "selected_formal_ppo_candidate_promotion_preflight_evaluated":
        _add_reason(reason_codes, "promotion_decision_preflight_readiness_invalid")
    if _git_current_matches_sources(preflight_summary) is False:
        _add_reason(reason_codes, "promotion_decision_preflight_git_provenance_mismatch")

    manifest_path = _resolve_optional_path(
        preflight_summary.get("promotion_manifest"),
        preflight_root,
        repo_root,
    )
    manifest = _read_json_if_exists(manifest_path)
    hash_audit_path = _resolve_optional_path(
        preflight_summary.get("checkpoint_hash_audit"),
        preflight_root,
        repo_root,
    )
    hash_audit = _read_json_if_exists(hash_audit_path)
    rollback_audit_path = _resolve_optional_path(
        preflight_summary.get("rollback_audit"),
        preflight_root,
        repo_root,
    )
    rollback_audit = _read_json_if_exists(rollback_audit_path)

    lineage = _lineage_report(
        repo_root=repo_root,
        preflight_summary_path=preflight_summary_path,
        preflight_summary=preflight_summary,
        manifest=manifest,
    )
    _write_json(lineage_path, lineage)
    _validate_lineage(lineage, reason_codes)

    checkpoint_identity = _checkpoint_identity_audit(
        preflight_summary=preflight_summary,
        manifest=manifest,
        hash_audit=hash_audit,
        repo_root=repo_root,
    )
    _write_json(checkpoint_identity_path, checkpoint_identity)
    if not checkpoint_identity["checkpoint_identity_audit_passed"]:
        for reason in checkpoint_identity["reason_codes"]:
            _add_reason(reason_codes, reason)

    release_boundary = _release_boundary_audit(
        preflight_summary=preflight_summary,
        manifest=manifest,
        rollback_audit=rollback_audit,
        lineage=lineage,
    )
    _write_json(release_boundary_path, release_boundary)
    if not release_boundary["release_boundary_audit_passed"]:
        _add_reason(reason_codes, "promotion_decision_release_boundary_invalid")

    if not reason_codes:
        decision_verdict = "eligible_for_guarded_release_candidate_packaging"
    elif any(
        reason in reason_codes
        for reason in (
            "promotion_decision_checkpoint_missing",
            "promotion_decision_checkpoint_hash_mismatch",
            "promotion_decision_checkpoint_metadata_invalid",
            "promotion_decision_candidate_identity_mismatch",
        )
    ):
        decision_verdict = "hold_for_more_evidence"
    else:
        decision_verdict = "blocked_by_preflight_or_provenance"

    pre_readiness_summary = _summary_payload(
        status="passed" if not reason_codes else "failed",
        reason_codes=reason_codes,
        decision_verdict=decision_verdict,
        repo_root=repo_root,
        preflight_root=preflight_root,
        output_root=output_root,
        batch_root=batch_root,
        preflight_summary_path=preflight_summary_path,
        summary_path=summary_path,
        lineage_path=lineage_path,
        checkpoint_identity_path=checkpoint_identity_path,
        release_boundary_path=release_boundary_path,
        readiness_path=readiness_path,
        report_path=report_path,
        preflight_summary=preflight_summary,
        lineage=lineage,
        checkpoint_identity=checkpoint_identity,
        release_boundary=release_boundary,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if not reason_codes:
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            decision_summary_path=summary_path,
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
            "recommended_next_action": "fix_selected_formal_ppo_candidate_promotion_decision_review",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        decision_verdict=decision_verdict if not reason_codes else _failed_verdict(reason_codes),
        repo_root=repo_root,
        preflight_root=preflight_root,
        output_root=output_root,
        batch_root=batch_root,
        preflight_summary_path=preflight_summary_path,
        summary_path=summary_path,
        lineage_path=lineage_path,
        checkpoint_identity_path=checkpoint_identity_path,
        release_boundary_path=release_boundary_path,
        readiness_path=readiness_path,
        report_path=report_path,
        preflight_summary=preflight_summary,
        lineage=lineage,
        checkpoint_identity=checkpoint_identity,
        release_boundary=release_boundary,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run selected formal PPO candidate promotion decision review."
    )
    parser.add_argument(
        "--preflight-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/selected_formal_ppo_candidate_promotion_decision_review_v1.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(args.config, repo_root)
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"invalid config schema: {config.get('schema_version')}")
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": str(args.config)}, sort_keys=True))
        return 0

    summary = run_selected_formal_ppo_candidate_promotion_decision_review(
        preflight_root=_resolve_path(args.preflight_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        batch_root=_resolve_path(args.batch_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "decision_verdict": summary["decision_verdict"],
                "readiness_status": summary.get("readiness_status"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _lineage_report(
    *,
    repo_root: Path,
    preflight_summary_path: Path,
    preflight_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    multihorizon_path = _resolve_optional_path(
        preflight_summary.get("multihorizon_summary") or manifest.get("multihorizon_summary"),
        preflight_summary_path.parent,
        repo_root,
    )
    multihorizon = _read_json_if_exists(multihorizon_path)
    selection_path = _resolve_optional_path(
        multihorizon.get("candidate_selection_summary"),
        multihorizon_path.parent if multihorizon_path else preflight_summary_path.parent,
        repo_root,
    )
    selection = _read_json_if_exists(selection_path)
    candidate_manifest_path = _resolve_optional_path(
        multihorizon.get("candidate_manifest") or selection.get("candidate_manifest"),
        multihorizon_path.parent if multihorizon_path else preflight_summary_path.parent,
        repo_root,
    )
    candidate_manifest = _read_json_if_exists(candidate_manifest_path)
    stability_path = _resolve_optional_path(
        selection.get("stability_summary") or candidate_manifest.get("stability_summary"),
        selection_path.parent if selection_path else preflight_summary_path.parent,
        repo_root,
    )
    stability = _read_json_if_exists(stability_path)

    sources = [
        _lineage_source(
            "promotion_preflight",
            preflight_summary_path,
            preflight_summary,
            PREFLIGHT_SCHEMA_VERSION,
        ),
        _lineage_source(
            "multihorizon_shadow_rollout",
            multihorizon_path,
            multihorizon,
            MULTIHORIZON_SCHEMA_VERSION,
        ),
        _lineage_source(
            "candidate_selection_long_horizon_holdout",
            selection_path,
            selection,
            SELECTION_SCHEMA_VERSION,
        ),
        _lineage_source(
            "formal_stability_holdout",
            stability_path,
            stability,
            STABILITY_SCHEMA_VERSION,
        ),
    ]
    return {
        "schema_version": "selected-formal-ppo-candidate-promotion-decision-lineage/v1",
        "generated_at": _utc_now(),
        "sources": sources,
        "candidate_manifest": None if candidate_manifest_path is None else str(candidate_manifest_path),
        "candidate_manifest_present": bool(candidate_manifest),
        "selected_seed_values": _field_values(
            [preflight_summary, manifest, multihorizon, selection, candidate_manifest],
            "selected_seed",
        ),
        "selected_budget_values": _field_values(
            [preflight_summary, manifest, multihorizon, selection, candidate_manifest],
            "selected_budget",
        ),
        "selected_candidate_root_values": _field_values(
            [preflight_summary, manifest, multihorizon, selection, candidate_manifest],
            "selected_candidate_root",
        ),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _lineage_source(
    name: str,
    path: Path | None,
    payload: dict[str, Any],
    expected_schema: str,
) -> dict[str, Any]:
    exists = path is not None and Path(path).is_file()
    schema_valid = payload.get("schema_version") == expected_schema
    status_passed = payload.get("status") == "passed"
    reason_codes = _string_list(payload.get("reason_codes"))
    provenance_ok = _git_current_matches_sources(payload) is not False
    passed = exists and schema_valid and status_passed and not reason_codes and provenance_ok
    return {
        "name": name,
        "path": None if path is None else str(path),
        "exists": exists,
        "schema_version": payload.get("schema_version"),
        "expected_schema_version": expected_schema,
        "schema_valid": schema_valid,
        "status": payload.get("status"),
        "reason_codes": reason_codes,
        "git_current_matches_sources": _git_current_matches_sources(payload),
        "publication_flags": _publication_flags(payload),
        "passed": passed,
    }


def _validate_lineage(lineage: dict[str, Any], reason_codes: list[str]) -> None:
    for source in lineage.get("sources", []):
        if not source.get("exists"):
            _add_reason(reason_codes, "promotion_decision_lineage_source_missing")
        if source.get("exists") and not source.get("schema_valid"):
            _add_reason(reason_codes, "promotion_decision_lineage_schema_invalid")
        if source.get("exists") and (
            source.get("status") != "passed" or source.get("reason_codes")
        ):
            _add_reason(reason_codes, "promotion_decision_lineage_source_not_passed")
        if source.get("git_current_matches_sources") is False:
            _add_reason(reason_codes, "promotion_decision_lineage_git_provenance_mismatch")
    if len(set(lineage.get("selected_seed_values", []))) > 1:
        _add_reason(reason_codes, "promotion_decision_candidate_identity_mismatch")
    if len(set(lineage.get("selected_budget_values", []))) > 1:
        _add_reason(reason_codes, "promotion_decision_candidate_identity_mismatch")
    if len(set(lineage.get("selected_candidate_root_values", []))) > 1:
        _add_reason(reason_codes, "promotion_decision_candidate_identity_mismatch")


def _checkpoint_identity_audit(
    *,
    preflight_summary: dict[str, Any],
    manifest: dict[str, Any],
    hash_audit: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    checkpoint_path = _resolve_optional_path(
        preflight_summary.get("checkpoint_path") or manifest.get("checkpoint_path"),
        repo_root,
        repo_root,
    )
    metadata_path = _resolve_optional_path(
        preflight_summary.get("checkpoint_metadata_path") or manifest.get("checkpoint_metadata_path"),
        repo_root,
        repo_root,
    )
    metadata = _read_json_if_exists(metadata_path)
    exists = checkpoint_path is not None and checkpoint_path.is_file()
    if not exists:
        _add_reason(reason_codes, "promotion_decision_checkpoint_missing")
        digest = None
        size = 0
    else:
        digest, size = _sha256_and_size(checkpoint_path)

    expected_hashes = [
        value
        for value in (
            preflight_summary.get("checkpoint_sha256"),
            manifest.get("checkpoint_sha256"),
            hash_audit.get("checkpoint_sha256"),
        )
        if value
    ]
    expected_sizes = [
        _int(value)
        for value in (
            preflight_summary.get("checkpoint_size_bytes"),
            manifest.get("checkpoint_size_bytes"),
            hash_audit.get("checkpoint_size_bytes"),
        )
        if _int(value) > 0
    ]
    checkpoint_path_values = _field_values(
        [preflight_summary, manifest, hash_audit, metadata],
        "checkpoint_path",
    )
    if exists and (not expected_hashes or any(value != digest for value in expected_hashes)):
        _add_reason(reason_codes, "promotion_decision_checkpoint_hash_mismatch")
    if exists and (not expected_sizes or any(value != size for value in expected_sizes)):
        _add_reason(reason_codes, "promotion_decision_checkpoint_size_mismatch")
    if checkpoint_path is not None and any(
        _resolve_path(value, repo_root) != checkpoint_path for value in checkpoint_path_values
    ):
        _add_reason(reason_codes, "promotion_decision_checkpoint_path_mismatch")
    if metadata_path is None or not metadata_path.is_file():
        _add_reason(reason_codes, "promotion_decision_checkpoint_metadata_missing")
    elif metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION or metadata.get("experimental") is not True:
        _add_reason(reason_codes, "promotion_decision_checkpoint_metadata_invalid")
    if manifest and manifest.get("schema_version") != PROMOTION_MANIFEST_SCHEMA_VERSION:
        _add_reason(reason_codes, "promotion_decision_manifest_schema_invalid")
    for field in ("selected_seed", "selected_budget", "selected_candidate_root"):
        values = _field_values([preflight_summary, manifest, metadata], field)
        if len(set(values)) > 1:
            _add_reason(reason_codes, "promotion_decision_candidate_identity_mismatch")

    return {
        "schema_version": "selected-formal-ppo-candidate-promotion-checkpoint-identity-audit/v1",
        "checkpoint_identity_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_metadata_path": None if metadata_path is None else str(metadata_path),
        "checkpoint_exists": exists,
        "checkpoint_sha256": digest,
        "checkpoint_size_bytes": size,
        "checkpoint_path_values": checkpoint_path_values,
        "expected_checkpoint_sha256_values": expected_hashes,
        "expected_checkpoint_size_values": expected_sizes,
        "metadata_schema_version": metadata.get("schema_version"),
        "metadata_experimental": metadata.get("experimental"),
    }


def _release_boundary_audit(
    *,
    preflight_summary: dict[str, Any],
    manifest: dict[str, Any],
    rollback_audit: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    sources = {
        "promotion_preflight": preflight_summary,
        "promotion_manifest": manifest,
        "rollback_audit": rollback_audit,
    }
    flags: dict[str, dict[str, bool]] = {}
    for name, payload in sources.items():
        flags[name] = {
            "runs_new_ppo_update": payload.get("runs_new_ppo_update") is True,
            "publishes_checkpoint": payload.get("publishes_checkpoint") is True,
            "replaces_default_policy": payload.get("replaces_default_policy") is True
            or payload.get("default_policy_replaced") is True,
            "performance_claimed": payload.get("performance_claimed") is True,
            "formal_training_ready_claimed": payload.get("formal_training_ready_claimed") is True,
        }
    lineage_publication_flags = {
        source.get("name", "unknown"): source.get("publication_flags", {})
        for source in lineage.get("sources", [])
    }
    boundary_ok = (
        not any(any(values.values()) for values in flags.values())
        and not any(any(values.values()) for values in lineage_publication_flags.values())
        and rollback_audit.get("rollback_audit_passed") is not False
        and rollback_audit.get("default_policy_replaced") is not True
    )
    return {
        "schema_version": "selected-formal-ppo-candidate-promotion-release-boundary-audit/v1",
        "release_boundary_audit_passed": boundary_ok,
        "experimental_candidate_only": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "publication_flags": flags,
        "lineage_publication_flags": lineage_publication_flags,
    }


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    decision_verdict: str,
    repo_root: Path,
    preflight_root: Path,
    output_root: Path,
    batch_root: Path,
    preflight_summary_path: Path,
    summary_path: Path,
    lineage_path: Path,
    checkpoint_identity_path: Path,
    release_boundary_path: Path,
    readiness_path: Path,
    report_path: Path,
    preflight_summary: dict[str, Any],
    lineage: dict[str, Any],
    checkpoint_identity: dict[str, Any],
    release_boundary: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_selected_formal_ppo_candidate_promotion_decision_review",
        "decision_verdict": decision_verdict,
        "preflight_root": str(preflight_root),
        "preflight_summary": str(preflight_summary_path),
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "lineage_report": str(lineage_path),
        "checkpoint_identity_audit": str(checkpoint_identity_path),
        "release_boundary_audit": str(release_boundary_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "selected_seed": preflight_summary.get("selected_seed"),
        "selected_budget": preflight_summary.get("selected_budget"),
        "selected_candidate_root": preflight_summary.get("selected_candidate_root"),
        "checkpoint_path": checkpoint_identity.get("checkpoint_path")
        or preflight_summary.get("checkpoint_path"),
        "checkpoint_metadata_path": checkpoint_identity.get("checkpoint_metadata_path")
        or preflight_summary.get("checkpoint_metadata_path"),
        "checkpoint_sha256": checkpoint_identity.get("checkpoint_sha256"),
        "checkpoint_size_bytes": _int(checkpoint_identity.get("checkpoint_size_bytes")),
        "lineage_audit_passed": all(
            source.get("passed") for source in lineage.get("sources", [])
        ),
        "source_lineage_count": len(lineage.get("sources", [])),
        "checkpoint_identity_audit_passed": bool(
            checkpoint_identity.get("checkpoint_identity_audit_passed")
        ),
        "release_boundary_audit_passed": bool(
            release_boundary.get("release_boundary_audit_passed")
        ),
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_promotion_decision_review": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "recommended_next_action": "guarded_experimental_policy_release_candidate_packaging"
        if status == "passed"
        else "fix_selected_formal_ppo_candidate_promotion_decision_review",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    decision_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--selected-formal-ppo-candidate-promotion-decision-review-summary",
        str(decision_summary_path),
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


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_selected_formal_ppo_candidate_promotion_decision_review_evaluated")
    if readiness.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")
    if _int(readiness.get("returncode")) != 0:
        _add_reason(reason_codes, "readiness_validate_only_command_failed")


def _failed_verdict(reason_codes: list[str]) -> str:
    if any(reason.startswith("promotion_decision_checkpoint") for reason in reason_codes):
        return "hold_for_more_evidence"
    return "blocked_by_preflight_or_provenance"


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Selected Formal PPO Candidate Promotion Decision Review v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- decision_verdict: `{summary.get('decision_verdict')}`",
            f"- selected seed/budget: `{summary.get('selected_seed')}` / `{summary.get('selected_budget')}`",
            f"- checkpoint sha256: `{summary.get('checkpoint_sha256')}`",
            f"- lineage/checkpoint/release audits: `{summary.get('lineage_audit_passed')}` / `{summary.get('checkpoint_identity_audit_passed')}` / `{summary.get('release_boundary_audit_passed')}`",
            f"- readiness: `{summary.get('readiness_status')}`",
            "",
            "This stage is a decision review only. It does not run PPO, publish a checkpoint, "
            "replace the default policy, claim policy performance, or claim formal training readiness. "
            "A passed verdict only means the experimental candidate is eligible for guarded release-candidate packaging.",
            "",
        ]
    )


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "preflight_summary": "selected-formal-ppo-candidate-promotion-preflight-summary.json"
    }
    defaults.update(config.get("input_files") or {})
    return defaults


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "lineage_report": LINEAGE_FILE,
        "checkpoint_identity_audit": CHECKPOINT_IDENTITY_FILE,
        "release_boundary_audit": RELEASE_BOUNDARY_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    defaults.update(config.get("output_files") or {})
    return defaults


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _field_values(payloads: list[dict[str, Any]], field: str) -> list[str]:
    return [str(payload[field]) for payload in payloads if payload.get(field) is not None]


def _publication_flags(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        "runs_new_ppo_update": payload.get("runs_new_ppo_update") is True,
        "publishes_checkpoint": payload.get("publishes_checkpoint") is True,
        "replaces_default_policy": payload.get("replaces_default_policy") is True
        or payload.get("default_policy_replaced") is True,
        "performance_claimed": payload.get("performance_claimed") is True,
        "formal_training_ready_claimed": payload.get("formal_training_ready_claimed") is True,
    }


def _git_current_matches_sources(payload: dict[str, Any]) -> bool | None:
    provenance = payload.get("git_provenance")
    if isinstance(provenance, dict) and provenance.get("current_matches_sources") is False:
        return False
    return True if isinstance(provenance, dict) else None


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
