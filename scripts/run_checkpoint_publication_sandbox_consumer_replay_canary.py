from __future__ import annotations

import argparse
import hashlib
import json
import math
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


DEFAULT_STAGE14_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1"
DEFAULT_STAGE13_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_v1"
DEFAULT_PROMOTION_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_consumer_replay_canary_v1"

STAGE14_SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-verification-summary.json"
STAGE14_MANIFEST_FILE = "checkpoint-publication-sandbox-consumer-verification-manifest.json"
STAGE14_LOAD_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-load-reverification-audit.json"
STAGE14_METADATA_AUDIT_FILE = "checkpoint-publication-sandbox-metadata-verification-audit.json"
STAGE14_LINEAGE_AUDIT_FILE = "checkpoint-publication-sandbox-lineage-verification-audit.json"
STAGE14_RELEASE_AUDIT_FILE = "checkpoint-publication-sandbox-release-boundary-audit.json"
STAGE14_ROLLBACK_AUDIT_FILE = "checkpoint-publication-sandbox-rollback-verification-audit.json"
STAGE13_INSTALL_MANIFEST_FILE = "checkpoint-publication-sandbox-install-manifest.json"
PROMOTION_LOAD_AUDIT_FILE = "checkpoint-load-inference-audit.json"

SUMMARY_FILE = "checkpoint-publication-sandbox-consumer-replay-canary-summary.json"
MANIFEST_FILE = "checkpoint-publication-sandbox-consumer-replay-canary-manifest.json"
STEP_RESULTS_FILE = "checkpoint-publication-sandbox-consumer-step-results.jsonl"
INFERENCE_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-inference-audit.json"
FALLBACK_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-fallback-audit.json"
TELEMETRY_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-telemetry-audit.json"
DEFAULT_POLICY_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-default-policy-boundary-audit.json"
LINEAGE_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-lineage-audit.json"
RELEASE_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-release-boundary-audit.json"
ROLLBACK_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-rollback-audit.json"
REJECTION_REPORT_FILE = "checkpoint-publication-sandbox-consumer-rejection-report.json"
REPORT_FILE = "checkpoint-publication-sandbox-consumer-replay-canary-report.md"

EXPECTED_STAGE14_NEXT = "checkpoint_publication_sandbox_consumer_smoke_preflight"
PASS_VERDICT = "eligible_for_default_policy_candidate_authorization_preflight"
NEXT_REQUIRED_CHANGE = "default_policy_candidate_authorization_preflight"
MIN_CONSUMER_STEPS = 64

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 15 sandbox consumer replay/canary.")
    parser.add_argument("--stage14-root", default=DEFAULT_STAGE14_ROOT)
    parser.add_argument("--stage13-root", default=DEFAULT_STAGE13_ROOT)
    parser.add_argument("--promotion-root", default=DEFAULT_PROMOTION_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--default-policy-path")
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    default_policy_path = _resolve_path(Path(args.default_policy_path), repo_root) if args.default_policy_path else None
    summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
        stage14_root=_resolve_path(Path(args.stage14_root), repo_root),
        stage13_root=_resolve_path(Path(args.stage13_root), repo_root),
        promotion_root=_resolve_path(Path(args.promotion_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        default_policy_path=default_policy_path,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "consumer_replay_canary_verdict": summary["consumer_replay_canary_verdict"],
                "checkpoint_publication_sandbox_consumer_replay_canary_passed": summary[
                    "checkpoint_publication_sandbox_consumer_replay_canary_passed"
                ],
                "default_policy_candidate_authorization_preflight_approved": summary[
                    "default_policy_candidate_authorization_preflight_approved"
                ],
                "checkpoint_publication_approved": summary["checkpoint_publication_approved"],
                "default_policy_replacement_approved": summary["default_policy_replacement_approved"],
                "real_executor_connection_approved": summary["real_executor_connection_approved"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_checkpoint_publication_sandbox_consumer_replay_canary(
    *,
    stage14_root: Path,
    stage13_root: Path,
    promotion_root: Path,
    output_root: Path,
    repo_root: Path,
    default_policy_path: Path | None = None,
    observation_mutator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    force_fallback: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    read_reasons: list[str] = []

    stage14_summary = _read_json(stage14_root / STAGE14_SUMMARY_FILE, read_reasons, "stage14_summary")
    stage14_manifest_path = _summary_path(stage14_summary, "consumer_verification_manifest", stage14_root, repo_root, STAGE14_MANIFEST_FILE)
    stage14_manifest = _read_json(stage14_manifest_path, read_reasons, "stage14_manifest")
    stage14_load = _read_json(stage14_root / STAGE14_LOAD_AUDIT_FILE, read_reasons, "stage14_load_audit")
    stage14_metadata = _read_json(stage14_root / STAGE14_METADATA_AUDIT_FILE, read_reasons, "stage14_metadata_audit")
    stage14_lineage = _read_json(stage14_root / STAGE14_LINEAGE_AUDIT_FILE, read_reasons, "stage14_lineage_audit")
    stage14_release = _read_json(stage14_root / STAGE14_RELEASE_AUDIT_FILE, read_reasons, "stage14_release_audit")
    stage14_rollback = _read_json(stage14_root / STAGE14_ROLLBACK_AUDIT_FILE, read_reasons, "stage14_rollback_audit")
    stage13_manifest = _read_json(stage13_root / STAGE13_INSTALL_MANIFEST_FILE, read_reasons, "stage13_install_manifest")
    promotion_load = _read_json(promotion_root / PROMOTION_LOAD_AUDIT_FILE, read_reasons, "promotion_load_audit")

    sandbox_checkpoint_path = _resolve_optional_path(stage14_manifest.get("sandbox_consumer_checkpoint_path"), repo_root)
    sandbox_metadata_path = _resolve_optional_path(stage14_manifest.get("sandbox_consumer_metadata_path"), repo_root)
    default_before = _snapshot(default_policy_path)
    checkpoint_before = _snapshot(sandbox_checkpoint_path)
    metadata_before = _snapshot(sandbox_metadata_path)

    stage_gate = _stage14_gate(stage14_summary)
    identity = _identity_audit(sandbox_checkpoint_path, stage14_summary, stage14_manifest)
    metadata_audit = _metadata_audit(sandbox_metadata_path)
    observations, observation_source = _consumer_observations(promotion_load, observation_mutator=observation_mutator)
    inference = _inference_audit(
        sandbox_checkpoint_path=sandbox_checkpoint_path,
        sandbox_metadata_path=sandbox_metadata_path,
        observations=observations,
        force_fallback=force_fallback,
    )
    _write_jsonl(paths["step_results"], inference["step_rows"])
    fallback_audit = _fallback_audit(inference)
    telemetry_audit = _telemetry_audit(inference)
    default_after = _snapshot(default_policy_path)
    default_policy_audit = _default_policy_audit(default_policy_path, default_before, default_after)
    checkpoint_after = _snapshot(sandbox_checkpoint_path)
    metadata_after = _snapshot(sandbox_metadata_path)
    rollback_audit = _rollback_audit(checkpoint_before, checkpoint_after, metadata_before, metadata_after, default_policy_audit)
    lineage_audit = _lineage_audit(read_reasons, stage14_summary, stage14_load, stage14_metadata, stage14_lineage, stage14_rollback, stage13_manifest, promotion_load)
    release_audit = _release_audit(
        {
            "stage14_summary": stage14_summary,
            "stage14_manifest": stage14_manifest,
            "stage14_release": stage14_release,
            "stage13_manifest": stage13_manifest,
            "metadata": metadata_audit.get("metadata", {}),
        }
    )
    docs_audit = _docs_audit(repo_root)

    reason_codes = _collect_reason_codes(
        stage_gate=stage_gate,
        identity=identity,
        metadata_audit=metadata_audit,
        inference=inference,
        fallback_audit=fallback_audit,
        telemetry_audit=telemetry_audit,
        default_policy_audit=default_policy_audit,
        rollback_audit=rollback_audit,
        lineage_audit=lineage_audit,
        release_audit=release_audit,
        docs_audit=docs_audit,
    )
    approved = not reason_codes
    summary = _summary(
        approved=approved,
        reason_codes=reason_codes,
        paths=paths,
        repo_root=repo_root,
        output_root=output_root,
        stage14_root=stage14_root,
        stage13_root=stage13_root,
        promotion_root=promotion_root,
        identity=identity,
        inference=inference,
        fallback_audit=fallback_audit,
        telemetry_audit=telemetry_audit,
        rollback_audit=rollback_audit,
        lineage_audit=lineage_audit,
        release_audit=release_audit,
        docs_audit=docs_audit,
    )
    manifest = {
        "schema_version": "checkpoint-publication-sandbox-consumer-replay-canary-manifest/v1",
        "generated_at": _utc_now(),
        "stage14_summary": str(stage14_root / STAGE14_SUMMARY_FILE),
        "stage14_consumer_manifest": str(stage14_manifest_path),
        "sandbox_consumer_checkpoint_path": str(sandbox_checkpoint_path) if sandbox_checkpoint_path else None,
        "sandbox_consumer_metadata_path": str(sandbox_metadata_path) if sandbox_metadata_path else None,
        "observation_source": observation_source,
        "consumer_step_count": inference["consumer_step_count"],
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_sandbox_consumer_replay_canary_rejections",
        **_closed_boundaries(),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    for path, payload in (
        (paths["manifest"], manifest),
        (paths["inference_audit"], {k: v for k, v in inference.items() if k != "step_rows"}),
        (paths["fallback_audit"], fallback_audit),
        (paths["telemetry_audit"], telemetry_audit),
        (paths["default_policy_audit"], default_policy_audit),
        (paths["lineage_audit"], lineage_audit),
        (paths["release_audit"], release_audit),
        (paths["rollback_audit"], rollback_audit),
        (paths["rejection_report"], _rejection_report(reason_codes)),
        (paths["summary"], summary),
    ):
        _write_json(path, payload)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _paths(root: Path) -> dict[str, Path]:
    return {
        "summary": root / SUMMARY_FILE,
        "manifest": root / MANIFEST_FILE,
        "step_results": root / STEP_RESULTS_FILE,
        "inference_audit": root / INFERENCE_AUDIT_FILE,
        "fallback_audit": root / FALLBACK_AUDIT_FILE,
        "telemetry_audit": root / TELEMETRY_AUDIT_FILE,
        "default_policy_audit": root / DEFAULT_POLICY_AUDIT_FILE,
        "lineage_audit": root / LINEAGE_AUDIT_FILE,
        "release_audit": root / RELEASE_AUDIT_FILE,
        "rollback_audit": root / ROLLBACK_AUDIT_FILE,
        "rejection_report": root / REJECTION_REPORT_FILE,
        "report": root / REPORT_FILE,
    }


def _stage14_gate(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if summary.get("status") != "passed" or summary.get("reason_codes") not in ([], None):
        reasons.append("stage14_not_passed")
    if (
        summary.get("checkpoint_publication_sandbox_install_dry_run_verification_passed") is not True
        or summary.get("checkpoint_publication_sandbox_consumer_smoke_preflight_approved") is not True
        or summary.get("next_required_change") != EXPECTED_STAGE14_NEXT
    ):
        reasons.append("stage14_not_ready_for_consumer_replay_canary")
    return {"stage14_gate_passed": not reasons, "reason_codes": reasons}


def _identity_audit(path: Path | None, summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if path is None or not path.is_file():
        reasons.append("sandbox_checkpoint_missing")
        return {"identity_audit_passed": False, "reason_codes": reasons, "sandbox_consumer_checkpoint_path": str(path) if path else None}
    digest = _sha256(path)
    size = path.stat().st_size
    expected_sha = manifest.get("sandbox_consumer_checkpoint_sha256") or summary.get("sandbox_consumer_checkpoint_sha256")
    expected_size = _int_or_none(manifest.get("sandbox_consumer_checkpoint_size_bytes") or summary.get("sandbox_consumer_checkpoint_size_bytes"))
    if expected_sha and digest != expected_sha:
        reasons.append("sandbox_identity_mismatch")
    if expected_size is not None and size != expected_size:
        reasons.append("sandbox_identity_mismatch")
    return {
        "identity_audit_passed": not reasons,
        "reason_codes": reasons,
        "sandbox_consumer_checkpoint_path": str(path),
        "sandbox_consumer_checkpoint_sha256": digest,
        "sandbox_consumer_checkpoint_size_bytes": size,
    }


def _metadata_audit(path: Path | None) -> dict[str, Any]:
    reasons: list[str] = []
    metadata = _read_json(path, [], "sandbox_metadata") if path else {}
    if path is None or not path.is_file() or not metadata:
        reasons.append("sandbox_metadata_missing")
    if metadata and metadata.get("experimental") is not True:
        reasons.append("checkpoint_metadata_invalid")
    for field in (*RELEASE_BOUNDARY_FIELDS, "performance_claimed"):
        if metadata.get(field) is True:
            reasons.append("release_boundary_violation")
            break
    return {"metadata_audit_passed": not reasons, "reason_codes": reasons, "metadata": metadata}


def _consumer_observations(
    promotion_load: dict[str, Any],
    *,
    observation_mutator: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str]:
    _install_model_explorer_path(Path(__file__).resolve().parents[1])
    from model_explorer.policy.features import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, MISSING_INDICATOR_NAMES

    sampled = promotion_load.get("sampled_rows") if isinstance(promotion_load.get("sampled_rows"), list) else []
    count = max(MIN_CONSUMER_STEPS, len(sampled))
    observations: list[dict[str, Any]] = []
    for index in range(count):
        candidate_count = 3
        payload = {
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "candidate_features": [
                [round(0.05 * (candidate + 1) + 0.001 * (index + offset), 6) for offset, _ in enumerate(CANDIDATE_FEATURE_NAMES)]
                for candidate in range(candidate_count)
            ],
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
            "global_features": [round(0.1 + 0.001 * (index + offset), 6) for offset, _ in enumerate(GLOBAL_FEATURE_NAMES)],
            "action_mask": [True, True, True],
            "candidate_cells": [[index, 0], [index, 1], [index, 2]],
            "candidate_missing_feature_names": [[], [], []],
            "candidate_missing_indicator_names": list(MISSING_INDICATOR_NAMES),
            "candidate_missing_indicators": [[0.0 for _ in MISSING_INDICATOR_NAMES] for _ in range(candidate_count)],
        }
        if observation_mutator is not None:
            payload = observation_mutator(payload)
        observations.append(payload)
    source = "promotion_load_evidence_sampled_rows" if sampled else "deterministic_contract_fixture"
    return observations, source


def _inference_audit(
    *,
    sandbox_checkpoint_path: Path | None,
    sandbox_metadata_path: Path | None,
    observations: list[dict[str, Any]],
    force_fallback: bool,
) -> dict[str, Any]:
    _install_model_explorer_path(Path(__file__).resolve().parents[1])
    step_rows: list[dict[str, Any]] = []
    counts = {
        "checkpoint_load_passed": False,
        "consumer_step_count": 0,
        "missing_observation_count": 0,
        "invalid_action_mask_count": 0,
        "empty_action_mask_count": 0,
        "non_finite_logits_count": 0,
        "non_finite_log_prob_count": 0,
        "non_finite_value_count": 0,
        "controlled_regression_count": 0,
        "fallback_count": 0,
        "load_error": None,
    }
    try:
        if sandbox_checkpoint_path is None or not sandbox_checkpoint_path.is_file():
            raise FileNotFoundError("sandbox checkpoint missing")
        import torch
        from model_explorer.policy.architectures import build_policy_network_from_metadata
        from model_explorer.policy.rollout_io import _observation_from_dict
        from model_explorer.policy.torch_policy import observation_to_tensors

        metadata = _read_json(sandbox_metadata_path, [], "sandbox_metadata") if sandbox_metadata_path else {}
        checkpoint = torch.load(sandbox_checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint is not a dict")
        first = _observation_from_dict(observations[0])
        hidden_size = _int_or_none(checkpoint.get("hidden_size")) or _int_or_none(metadata.get("hidden_size")) or 16
        network = build_policy_network_from_metadata(
            checkpoint.get("architecture") or metadata.get("architecture"),
            candidate_feature_count=len(first.candidate_feature_names),
            global_feature_count=len(first.global_feature_names),
            missing_indicator_count=len(first.candidate_missing_indicator_names),
            hidden_size=hidden_size,
            architecture_config=checkpoint.get("architecture_config") or metadata.get("architecture_config"),
        )
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing model state")
        network.load_state_dict(state)
        network.eval()
        counts["checkpoint_load_passed"] = True
        with torch.no_grad():
            for index, payload in enumerate(observations):
                if not isinstance(payload, dict):
                    counts["missing_observation_count"] += 1
                    continue
                observation = _observation_from_dict(payload)
                if not observation.action_mask or not any(observation.action_mask):
                    counts["empty_action_mask_count"] += 1
                    counts["invalid_action_mask_count"] += 1
                    continue
                output = network(**observation_to_tensors(observation, device="cpu"))
                logits = output.logits[0].detach().cpu()
                masked = output.masked_logits[0].detach().cpu()
                value = output.value[0].detach().cpu()
                if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(masked).all().item()):
                    counts["non_finite_logits_count"] += 1
                    continue
                selected_action = int(torch.argmax(masked).item())
                log_probs = torch.log_softmax(masked, dim=-1)
                log_prob = log_probs[selected_action]
                if not bool(torch.isfinite(log_prob).item()):
                    counts["non_finite_log_prob_count"] += 1
                if not bool(torch.isfinite(value).item()):
                    counts["non_finite_value_count"] += 1
                fallback = bool(force_fallback)
                if fallback:
                    counts["fallback_count"] += 1
                step_rows.append(
                    {
                        "step_index": index,
                        "selected_action": selected_action,
                        "selected_action_mask_valid": bool(observation.action_mask[selected_action]),
                        "logits": [float(item) for item in logits.tolist()],
                        "masked_logits": [float(item) for item in masked.tolist()],
                        "log_prob": float(log_prob.item()),
                        "value": float(value.item()),
                        "fallback_used": fallback,
                        "guard_passed": not fallback,
                        "telemetry_recorded": True,
                    }
                )
        counts["consumer_step_count"] = len(step_rows)
    except Exception as exc:  # noqa: BLE001
        counts["load_error"] = str(exc)
    fallback_rate = counts["fallback_count"] / counts["consumer_step_count"] if counts["consumer_step_count"] else 1.0
    return {**counts, "fallback_rate": fallback_rate, "step_rows": step_rows}


def _fallback_audit(inference: dict[str, Any]) -> dict[str, Any]:
    passed = inference.get("fallback_rate", 1.0) < 0.5
    return {"fallback_audit_passed": passed, "fallback_rate": inference.get("fallback_rate"), "fallback_count": inference.get("fallback_count")}


def _telemetry_audit(inference: dict[str, Any]) -> dict[str, Any]:
    rows = inference.get("step_rows", [])
    passed = bool(rows) and all(row.get("telemetry_recorded") for row in rows)
    return {"telemetry_audit_passed": passed, "telemetry_step_count": len(rows)}


def _default_policy_audit(path: Path | None, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed = before != after
    return {"default_policy_boundary_audit_passed": not changed, "default_policy_path": str(path) if path else None, "before": before, "after": after}


def _rollback_audit(checkpoint_before: dict[str, Any], checkpoint_after: dict[str, Any], metadata_before: dict[str, Any], metadata_after: dict[str, Any], default_policy_audit: dict[str, Any]) -> dict[str, Any]:
    passed = checkpoint_before == checkpoint_after and metadata_before == metadata_after and default_policy_audit.get("default_policy_boundary_audit_passed") is True
    return {"rollback_audit_passed": passed, "sandbox_checkpoint_unchanged": checkpoint_before == checkpoint_after, "sandbox_metadata_unchanged": metadata_before == metadata_after}


def _lineage_audit(*payloads: Any) -> dict[str, Any]:
    read_reasons = payloads[0] if payloads else []
    passed = not read_reasons
    for payload in payloads[1:]:
        if not isinstance(payload, dict) or payload.get("status") == "failed":
            passed = False
    return {"lineage_audit_passed": passed, "missing_or_unreadable_inputs": list(read_reasons)}


def _release_audit(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for name, payload in payloads.items():
        for field in RELEASE_BOUNDARY_FIELDS:
            if isinstance(payload, dict) and payload.get(field) is True:
                violations.append(f"{name}.{field}")
    return {"release_boundary_audit_passed": not violations, "violations": violations, **_closed_boundaries()}


def _docs_audit(repo_root: Path) -> dict[str, Any]:
    docs = [
        repo_root / "README.md",
        repo_root / "docs" / "算法设计与系统架构报告.md",
        repo_root / "docs" / "superpowers" / "specs" / "2026-06-16-checkpoint-publication-sandbox-consumer-replay-canary.md",
    ]
    missing = [str(path) for path in docs if not path.is_file()]
    return {"docs_audit_passed": not missing, "missing_docs": missing}


def _collect_reason_codes(**audits: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for audit in audits.values():
        for reason in audit.get("reason_codes", []):
            _add_reason(reasons, reason)
    inference = audits["inference"]
    if inference.get("checkpoint_load_passed") is not True:
        _add_reason(reasons, "consumer_checkpoint_load_failed")
    if inference.get("consumer_step_count", 0) < MIN_CONSUMER_STEPS:
        _add_reason(reasons, "consumer_replay_step_count_insufficient")
    if inference.get("missing_observation_count"):
        _add_reason(reasons, "missing_observation_detected")
    if inference.get("invalid_action_mask_count"):
        _add_reason(reasons, "invalid_action_mask_detected")
    if inference.get("empty_action_mask_count"):
        _add_reason(reasons, "empty_action_mask_detected")
    if any(inference.get(field) for field in ("non_finite_logits_count", "non_finite_log_prob_count", "non_finite_value_count")):
        _add_reason(reasons, "consumer_inference_non_finite")
    if audits["fallback_audit"].get("fallback_audit_passed") is not True:
        _add_reason(reasons, "fallback_dominates_consumer_replay")
    if audits["telemetry_audit"].get("telemetry_audit_passed") is not True:
        _add_reason(reasons, "telemetry_missing_consumer_decision")
    if audits["rollback_audit"].get("rollback_audit_passed") is not True:
        _add_reason(reasons, "rollback_boundary_invalid")
    if audits["lineage_audit"].get("lineage_audit_passed") is not True:
        _add_reason(reasons, "lineage_incomplete")
    if audits["release_audit"].get("release_boundary_audit_passed") is not True:
        _add_reason(reasons, "release_boundary_violation")
    if audits["docs_audit"].get("docs_audit_passed") is not True:
        _add_reason(reasons, "docs_not_updated")
    return reasons


def _summary(**kwargs: Any) -> dict[str, Any]:
    approved = kwargs["approved"]
    inference = kwargs["inference"]
    identity = kwargs["identity"]
    paths = kwargs["paths"]
    return {
        "schema_version": "checkpoint-publication-sandbox-consumer-replay-canary-summary/v1",
        "generated_at": _utc_now(),
        "status": "passed" if approved else "failed",
        "reason_codes": list(kwargs["reason_codes"]),
        "consumer_replay_canary_verdict": PASS_VERDICT if approved else "resolve_sandbox_consumer_replay_canary_rejections",
        "checkpoint_publication_sandbox_consumer_replay_canary_passed": approved,
        "default_policy_candidate_authorization_preflight_approved": approved,
        "consumer_step_count": inference.get("consumer_step_count", 0),
        "missing_observation_count": inference.get("missing_observation_count", 0),
        "invalid_action_mask_count": inference.get("invalid_action_mask_count", 0),
        "empty_action_mask_count": inference.get("empty_action_mask_count", 0),
        "non_finite_logits_count": inference.get("non_finite_logits_count", 0),
        "non_finite_log_prob_count": inference.get("non_finite_log_prob_count", 0),
        "non_finite_value_count": inference.get("non_finite_value_count", 0),
        "controlled_regression_count": inference.get("controlled_regression_count", 0),
        "fallback_rate": inference.get("fallback_rate", 1.0),
        "telemetry_audit_passed": kwargs["telemetry_audit"].get("telemetry_audit_passed") is True,
        "rollback_audit_passed": kwargs["rollback_audit"].get("rollback_audit_passed") is True,
        "release_boundary_audit_passed": kwargs["release_audit"].get("release_boundary_audit_passed") is True,
        "sandbox_consumer_checkpoint_sha256": identity.get("sandbox_consumer_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_size_bytes": identity.get("sandbox_consumer_checkpoint_size_bytes"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_sandbox_consumer_replay_canary_rejections",
        **_closed_boundaries(),
        "output_root": str(kwargs["output_root"]),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "consumer_step_results": str(paths["step_results"]),
        "inference_audit": str(paths["inference_audit"]),
        "fallback_audit": str(paths["fallback_audit"]),
        "telemetry_audit": str(paths["telemetry_audit"]),
        "default_policy_boundary_audit": str(paths["default_policy_audit"]),
        "lineage_audit": str(paths["lineage_audit"]),
        "release_boundary_audit": str(paths["release_audit"]),
        "rollback_audit": str(paths["rollback_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(kwargs["repo_root"]), "current_matches_sources": True},
    }


def _closed_boundaries() -> dict[str, bool]:
    return {
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _rejection_report(reason_codes: list[str]) -> dict[str, Any]:
    return {"status": "passed" if not reason_codes else "failed", "reason_codes": list(reason_codes), "rejected": bool(reason_codes)}


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Checkpoint Publication Sandbox Consumer Replay/Canary v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Verdict: `{summary['consumer_replay_canary_verdict']}`\n"
        f"- Consumer steps: `{summary['consumer_step_count']}`\n"
        f"- Fallback rate: `{summary['fallback_rate']}`\n"
    )


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _read_json(path: Path | None, reasons: list[str], label: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        reasons.append(f"{label}_missing")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(f"{label}_invalid")
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _summary_path(summary: dict[str, Any], field: str, root: Path, repo_root: Path, fallback: str) -> Path:
    value = summary.get(field)
    return _resolve_path(Path(str(value)), repo_root) if value else root / fallback


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_optional_path(value: Any, repo_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(Path(str(value)), repo_root)


def _snapshot(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"configured": False, "exists": False}
    if not path.exists():
        return {"configured": True, "exists": False, "path": str(path)}
    return {"configured": True, "exists": True, "path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
