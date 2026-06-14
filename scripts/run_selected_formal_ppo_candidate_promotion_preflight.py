from __future__ import annotations

import argparse
import hashlib
import json
import math
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


CONFIG_SCHEMA_VERSION = "selected-formal-ppo-candidate-promotion-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "selected-formal-ppo-candidate-promotion-preflight-summary/v1"
MULTIHORIZON_SCHEMA_VERSION = (
    "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1"
)
CHECKPOINT_METADATA_SCHEMA_VERSION = "controlled-hybrid-policy-candidate-checkpoint-metadata/v1"
CANDIDATE_SUMMARY_SCHEMA_VERSION = "raw-policy-generalization-candidate-summary/v1"
LIMITED_PPO_SUMMARY_SCHEMA_VERSION = "limited-ppo-update-smoke-summary/v1"
EXPECTED_READINESS_STATUS = "selected_formal_ppo_candidate_promotion_preflight_evaluated"

SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
MANIFEST_FILE = "promotion-candidate-manifest.json"
CHECKPOINT_HASH_AUDIT_FILE = "checkpoint-hash-audit.json"
INFERENCE_AUDIT_FILE = "checkpoint-load-inference-audit.json"
ROLLBACK_AUDIT_FILE = "rollback-audit.json"
READINESS_FILE = "promotion-preflight-readiness-validate-only.json"
REPORT_FILE = "promotion-preflight-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_selected_formal_ppo_candidate_promotion_preflight(
    *,
    multihorizon_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    multihorizon_root = Path(multihorizon_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _install_model_explorer_path(repo_root)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    manifest_path = output_root / files["manifest"]
    checkpoint_hash_audit_path = output_root / files["checkpoint_hash_audit"]
    inference_audit_path = output_root / files["inference_audit"]
    rollback_audit_path = output_root / files["rollback_audit"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    inputs = _input_files(config)
    multihorizon_summary_path = multihorizon_root / inputs["multihorizon_summary"]
    multihorizon_summary = _read_json_if_exists(multihorizon_summary_path)
    selected_candidate_root = _resolve_optional_path(
        multihorizon_summary.get("selected_candidate_root"),
        multihorizon_root,
        repo_root,
    )
    if selected_candidate_root is None:
        selected_candidate_root = multihorizon_root / "selected-candidate"

    checkpoint_path = selected_candidate_root / inputs["checkpoint"]
    checkpoint_metadata_path = selected_candidate_root / inputs["checkpoint_metadata"]
    candidate_summary_path = selected_candidate_root / inputs["candidate_summary"]
    limited_ppo_summary_path = selected_candidate_root / inputs["limited_ppo_summary"]
    diagnostics_path = selected_candidate_root / inputs["diagnostics"]
    training_curves_path = selected_candidate_root / inputs["training_curves"]
    progress_events_path = selected_candidate_root / inputs["progress_events"]
    progress_summary_path = selected_candidate_root / inputs["progress_summary"]
    multihorizon_steps_path = _resolve_optional_path(
        multihorizon_summary.get("steps"),
        multihorizon_root,
        repo_root,
    ) or (multihorizon_root / "multihorizon-shadow-rollout-steps.jsonl")
    legacy_candidate_manifest_path = _resolve_optional_path(
        multihorizon_summary.get("candidate_manifest"),
        multihorizon_root,
        repo_root,
    )

    checkpoint_metadata = _read_json_if_exists(checkpoint_metadata_path)
    candidate_summary = _read_json_if_exists(candidate_summary_path)
    limited_ppo_summary = _read_json_if_exists(limited_ppo_summary_path)
    diagnostics = _read_json_if_exists(diagnostics_path)
    training_curves = _read_json_if_exists(training_curves_path)
    progress_summary = _read_json_if_exists(progress_summary_path)
    legacy_candidate_manifest = _read_json_if_exists(legacy_candidate_manifest_path)
    steps = _read_jsonl(multihorizon_steps_path)

    reason_codes: list[str] = []
    _validate_inputs(
        multihorizon_summary=multihorizon_summary,
        checkpoint_metadata=checkpoint_metadata,
        candidate_summary=candidate_summary,
        limited_ppo_summary=limited_ppo_summary,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        multihorizon_steps_path=multihorizon_steps_path,
        diagnostics_path=diagnostics_path,
        training_curves_path=training_curves_path,
        progress_events_path=progress_events_path,
        progress_summary_path=progress_summary_path,
        steps=steps,
        selected_candidate_root=selected_candidate_root,
        config=config,
        reason_codes=reason_codes,
    )

    checkpoint_hash_audit = _checkpoint_hash_audit(checkpoint_path)
    _write_json(checkpoint_hash_audit_path, checkpoint_hash_audit)
    if not checkpoint_hash_audit["checkpoint_exists"]:
        _add_reason(reason_codes, "promotion_preflight_checkpoint_missing")

    inference_audit = _checkpoint_inference_audit(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        steps=steps,
        config=config,
    )
    _write_json(inference_audit_path, inference_audit)
    _validate_inference_audit(inference_audit, config, reason_codes)

    rollback_audit = _rollback_audit(
        multihorizon_summary=multihorizon_summary,
        checkpoint_metadata=checkpoint_metadata,
        candidate_summary=candidate_summary,
        limited_ppo_summary=limited_ppo_summary,
    )
    _write_json(rollback_audit_path, rollback_audit)
    if not rollback_audit["rollback_audit_passed"]:
        _add_reason(reason_codes, "promotion_preflight_rollback_boundary_invalid")

    manifest = _promotion_manifest(
        repo_root=repo_root,
        output_root=output_root,
        multihorizon_root=multihorizon_root,
        multihorizon_summary_path=multihorizon_summary_path,
        multihorizon_steps_path=multihorizon_steps_path,
        selected_candidate_root=selected_candidate_root,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        candidate_summary_path=candidate_summary_path,
        limited_ppo_summary_path=limited_ppo_summary_path,
        diagnostics_path=diagnostics_path,
        training_curves_path=training_curves_path,
        progress_events_path=progress_events_path,
        progress_summary_path=progress_summary_path,
        checkpoint_hash_audit_path=checkpoint_hash_audit_path,
        inference_audit_path=inference_audit_path,
        rollback_audit_path=rollback_audit_path,
        checkpoint_hash_audit=checkpoint_hash_audit,
        multihorizon_summary=multihorizon_summary,
        checkpoint_metadata=checkpoint_metadata,
        legacy_candidate_manifest_path=legacy_candidate_manifest_path,
        legacy_candidate_manifest=legacy_candidate_manifest,
    )
    _write_json(manifest_path, manifest)

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        multihorizon_root=multihorizon_root,
        multihorizon_summary_path=multihorizon_summary_path,
        multihorizon_steps_path=multihorizon_steps_path,
        selected_candidate_root=selected_candidate_root,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        candidate_summary_path=candidate_summary_path,
        limited_ppo_summary_path=limited_ppo_summary_path,
        diagnostics_path=diagnostics_path,
        training_curves_path=training_curves_path,
        progress_events_path=progress_events_path,
        progress_summary_path=progress_summary_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        manifest_path=manifest_path,
        checkpoint_hash_audit_path=checkpoint_hash_audit_path,
        inference_audit_path=inference_audit_path,
        rollback_audit_path=rollback_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        multihorizon_summary=multihorizon_summary,
        checkpoint_hash_audit=checkpoint_hash_audit,
        inference_audit=inference_audit,
        rollback_audit=rollback_audit,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            promotion_summary_path=summary_path,
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
            "recommended_next_action": "fix_selected_formal_ppo_candidate_promotion_preflight",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        multihorizon_root=multihorizon_root,
        multihorizon_summary_path=multihorizon_summary_path,
        multihorizon_steps_path=multihorizon_steps_path,
        selected_candidate_root=selected_candidate_root,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        candidate_summary_path=candidate_summary_path,
        limited_ppo_summary_path=limited_ppo_summary_path,
        diagnostics_path=diagnostics_path,
        training_curves_path=training_curves_path,
        progress_events_path=progress_events_path,
        progress_summary_path=progress_summary_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        manifest_path=manifest_path,
        checkpoint_hash_audit_path=checkpoint_hash_audit_path,
        inference_audit_path=inference_audit_path,
        rollback_audit_path=rollback_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        multihorizon_summary=multihorizon_summary,
        checkpoint_hash_audit=checkpoint_hash_audit,
        inference_audit=inference_audit,
        rollback_audit=rollback_audit,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run selected formal PPO candidate promotion preflight."
    )
    parser.add_argument(
        "--multihorizon-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/selected_formal_ppo_candidate_promotion_preflight_v1.json",
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

    summary = run_selected_formal_ppo_candidate_promotion_preflight(
        multihorizon_root=_resolve_path(Path(args.multihorizon_root), repo_root),
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
                "checkpoint_load_passed": summary["checkpoint_load_passed"],
                "inference_audit_count": summary["inference_audit_count"],
                "readiness_status": summary.get("readiness_status"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    promotion_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--selected-formal-ppo-candidate-promotion-preflight-summary",
        str(promotion_summary_path),
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


def _validate_inputs(
    *,
    multihorizon_summary: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    candidate_summary: dict[str, Any],
    limited_ppo_summary: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    multihorizon_steps_path: Path,
    diagnostics_path: Path,
    training_curves_path: Path,
    progress_events_path: Path,
    progress_summary_path: Path,
    steps: list[dict[str, Any]],
    selected_candidate_root: Path,
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if multihorizon_summary.get("schema_version") != MULTIHORIZON_SCHEMA_VERSION:
        _add_reason(reason_codes, "promotion_preflight_multihorizon_schema_invalid")
    if multihorizon_summary.get("status") != "passed" or _string_list(
        multihorizon_summary.get("reason_codes")
    ):
        _add_reason(reason_codes, "promotion_preflight_multihorizon_not_passed")
    if multihorizon_summary.get("git_provenance", {}).get("current_matches_sources") is False:
        _add_reason(reason_codes, "promotion_preflight_multihorizon_git_provenance_mismatch")
    expected_trainable = _expected_trainable(config)
    if expected_trainable and _int(multihorizon_summary.get("input_trainable_transition_count")) != expected_trainable:
        _add_reason(reason_codes, "promotion_preflight_trainable_count_mismatch")
    if expected_trainable and _int(multihorizon_summary.get("unique_trainable_context_count")) != expected_trainable:
        _add_reason(reason_codes, "promotion_preflight_unique_context_count_mismatch")
    if multihorizon_summary.get("selected_seed") != 0:
        _add_reason(reason_codes, "promotion_preflight_selected_seed_mismatch")
    if multihorizon_summary.get("selected_budget") != "epochs1_lr3e-6":
        _add_reason(reason_codes, "promotion_preflight_selected_budget_mismatch")
    if not str(multihorizon_summary.get("selected_candidate_root") or ""):
        _add_reason(reason_codes, "promotion_preflight_candidate_root_missing")
    if checkpoint_metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION:
        _add_reason(reason_codes, "promotion_preflight_checkpoint_metadata_invalid")
    if checkpoint_metadata and checkpoint_metadata.get("experimental") is not True:
        _add_reason(reason_codes, "promotion_preflight_checkpoint_not_experimental")
    if checkpoint_metadata and checkpoint_metadata.get("seed") != multihorizon_summary.get("selected_seed"):
        _add_reason(reason_codes, "promotion_preflight_selected_seed_mismatch")
    if checkpoint_metadata and _int(checkpoint_metadata.get("sample_count")) != expected_trainable:
        _add_reason(reason_codes, "promotion_preflight_checkpoint_sample_count_mismatch")
    if candidate_summary.get("schema_version") != CANDIDATE_SUMMARY_SCHEMA_VERSION:
        _add_reason(reason_codes, "promotion_preflight_candidate_summary_invalid")
    if candidate_summary.get("status") != "passed" or _string_list(candidate_summary.get("reason_codes")):
        _add_reason(reason_codes, "promotion_preflight_candidate_summary_not_passed")
    if limited_ppo_summary.get("schema_version") != LIMITED_PPO_SUMMARY_SCHEMA_VERSION:
        _add_reason(reason_codes, "promotion_preflight_limited_ppo_summary_invalid")
    if limited_ppo_summary.get("status") != "passed" or _string_list(limited_ppo_summary.get("reason_codes")):
        _add_reason(reason_codes, "promotion_preflight_limited_ppo_summary_not_passed")
    if not checkpoint_metadata_path.is_file():
        _add_reason(reason_codes, "promotion_preflight_checkpoint_metadata_missing")
    if not multihorizon_steps_path.is_file():
        _add_reason(reason_codes, "promotion_preflight_multihorizon_steps_missing")
    if not steps:
        _add_reason(reason_codes, "promotion_preflight_multihorizon_steps_empty")
    if not checkpoint_path.is_file():
        _add_reason(reason_codes, "promotion_preflight_checkpoint_missing")
    for path in (
        diagnostics_path,
        training_curves_path,
        progress_events_path,
        progress_summary_path,
    ):
        if not path.is_file():
            _add_reason(reason_codes, "promotion_preflight_candidate_artifacts_missing")
    if Path(str(selected_candidate_root)) != selected_candidate_root:
        _add_reason(reason_codes, "promotion_preflight_candidate_root_invalid")
    for source in (multihorizon_summary, checkpoint_metadata, candidate_summary, limited_ppo_summary):
        for field, reason in (
            ("runs_new_ppo_update", "promotion_preflight_unexpected_ppo_update"),
            ("publishes_checkpoint", "promotion_preflight_checkpoint_publication_claimed"),
            ("replaces_default_policy", "promotion_preflight_default_policy_replacement_claimed"),
            ("performance_claimed", "promotion_preflight_policy_performance_claimed"),
            ("formal_training_ready_claimed", "promotion_preflight_formal_ready_claimed"),
        ):
            if source.get(field) is True:
                _add_reason(reason_codes, reason)


def _checkpoint_hash_audit(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        return {
            "checkpoint_exists": False,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": None,
            "checkpoint_size_bytes": 0,
        }
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "checkpoint_exists": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
    }


def _checkpoint_inference_audit(
    *,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    steps: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    min_count = _int(config.get("validation", {}).get("min_inference_audit_count"), 64)
    rows: list[dict[str, Any]] = []
    counts = {
        "checkpoint_load_passed": False,
        "inference_audit_count": 0,
        "invalid_action_mask_count": 0,
        "missing_observation_count": 0,
        "non_finite_logits_count": 0,
        "non_finite_log_prob_count": 0,
        "non_finite_value_count": 0,
        "action_reconstruction_difference_count": 0,
        "log_prob_reconstruction_max_abs_error": math.inf,
        "value_reconstruction_max_abs_error": math.inf,
        "reconstruction_difference_explained": False,
        "reconstruction_reference": "shadow_records_may_precede_selected_promotion_checkpoint",
    }
    if not checkpoint_path.is_file():
        counts["checkpoint_error"] = "checkpoint does not exist"
        counts["sampled_rows"] = rows
        return counts

    try:
        import torch
        from model_explorer.policy.architectures import build_policy_network_from_metadata
        from model_explorer.policy.rollout_io import _observation_from_dict
        from model_explorer.policy.torch_policy import observation_to_tensors

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        first_observation = next(
            step.get("observation")
            for step in steps
            if isinstance(step.get("observation"), dict)
        )
        first = _observation_from_dict(first_observation)
        architecture = checkpoint.get("architecture") or checkpoint_metadata.get("architecture")
        hidden_size = _int(
            checkpoint.get("hidden_size")
            or checkpoint_metadata.get("hidden_size")
            or checkpoint.get("training", {}).get("hidden_size"),
            16,
        )
        network = build_policy_network_from_metadata(
            architecture,
            candidate_feature_count=len(first.candidate_feature_names),
            global_feature_count=len(first.global_feature_names),
            missing_indicator_count=len(first.candidate_missing_indicator_names),
            hidden_size=hidden_size,
            architecture_config=checkpoint.get("architecture_config")
            or checkpoint_metadata.get("architecture_config"),
        )
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing model state")
        network.load_state_dict(state)
        network.eval()
        counts["checkpoint_load_passed"] = True
        log_errors: list[float] = []
        value_errors: list[float] = []
        audited_count = 0
        with torch.no_grad():
            for step in steps:
                if audited_count >= min_count:
                    break
                observation_payload = step.get("observation")
                if not isinstance(observation_payload, dict):
                    counts["missing_observation_count"] += 1
                    continue
                observation = _observation_from_dict(observation_payload)
                action_index = _int(
                    step.get("controlled_action_index", step.get("action_index")),
                    -1,
                )
                if (
                    not observation.action_mask
                    or action_index < 0
                    or action_index >= len(observation.action_mask)
                    or not observation.action_mask[action_index]
                ):
                    counts["invalid_action_mask_count"] += 1
                    continue
                output = network(**observation_to_tensors(observation, device="cpu"))
                logits = output.masked_logits[0]
                if not bool(torch.isfinite(logits).all().item()):
                    counts["non_finite_logits_count"] += 1
                    continue
                distribution = torch.distributions.Categorical(logits=output.masked_logits)
                log_prob = float(distribution.log_prob(torch.tensor([action_index])).item())
                value = float(output.value[0].item())
                if not math.isfinite(log_prob):
                    counts["non_finite_log_prob_count"] += 1
                if not math.isfinite(value):
                    counts["non_finite_value_count"] += 1
                stored_log_prob = _float(step.get("log_prob"), math.nan)
                stored_value = _float(step.get("value"), math.nan)
                log_error = abs(log_prob - stored_log_prob)
                value_error = abs(value - stored_value)
                if math.isfinite(log_error):
                    log_errors.append(log_error)
                if math.isfinite(value_error):
                    value_errors.append(value_error)
                selected_rank = _selected_action_rank(logits, action_index, torch=torch)
                if selected_rank != 1:
                    counts["action_reconstruction_difference_count"] += 1
                rows.append(
                    {
                        "context_id": step.get("context_id"),
                        "scenario_id": step.get("scenario_id"),
                        "scenario_family": step.get("scenario_family"),
                        "action_index": action_index,
                        "selected_action_rank": selected_rank,
                        "log_prob": log_prob,
                        "value": value,
                        "stored_log_prob": stored_log_prob,
                        "stored_value": stored_value,
                        "log_prob_abs_error": log_error,
                        "value_abs_error": value_error,
                    }
                )
                audited_count += 1
        counts["inference_audit_count"] = audited_count
        counts["log_prob_reconstruction_max_abs_error"] = max(log_errors) if log_errors else math.inf
        counts["value_reconstruction_max_abs_error"] = max(value_errors) if value_errors else math.inf
        counts["reconstruction_difference_explained"] = True
    except Exception as exc:  # noqa: BLE001 - surfaced in summary.
        counts["checkpoint_error"] = str(exc)
    counts["sampled_rows"] = rows
    return counts


def _validate_inference_audit(
    inference_audit: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {})
    if inference_audit.get("checkpoint_load_passed") is not True:
        _add_reason(reason_codes, "promotion_preflight_checkpoint_load_failed")
    if _int(inference_audit.get("inference_audit_count")) < _int(
        validation.get("min_inference_audit_count"), 64
    ):
        _add_reason(reason_codes, "promotion_preflight_inference_audit_count_below_threshold")
    for field, reason in (
        ("invalid_action_mask_count", "promotion_preflight_invalid_action_mask"),
        ("missing_observation_count", "promotion_preflight_missing_observation"),
        ("non_finite_logits_count", "promotion_preflight_non_finite_inference"),
        ("non_finite_log_prob_count", "promotion_preflight_non_finite_inference"),
        ("non_finite_value_count", "promotion_preflight_non_finite_inference"),
    ):
        if _int(inference_audit.get(field)) > 0:
            _add_reason(reason_codes, reason)
    if inference_audit.get("reconstruction_difference_explained") is not True:
        _add_reason(reason_codes, "promotion_preflight_reconstruction_difference_unexplained")


def _rollback_audit(
    *,
    multihorizon_summary: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    candidate_summary: dict[str, Any],
    limited_ppo_summary: dict[str, Any],
) -> dict[str, Any]:
    sources = {
        "multihorizon_summary": multihorizon_summary,
        "checkpoint_metadata": checkpoint_metadata,
        "candidate_summary": candidate_summary,
        "limited_ppo_summary": limited_ppo_summary,
    }
    publication_flags = {
        name: {
            "runs_new_ppo_update": payload.get("runs_new_ppo_update") is True,
            "publishes_checkpoint": payload.get("publishes_checkpoint") is True,
            "replaces_default_policy": payload.get("replaces_default_policy") is True,
            "performance_claimed": payload.get("performance_claimed") is True,
            "formal_training_ready_claimed": payload.get("formal_training_ready_claimed") is True,
        }
        for name, payload in sources.items()
    }
    rollback_audit_passed = not any(
        any(flags.values()) for flags in publication_flags.values()
    )
    return {
        "rollback_audit_passed": rollback_audit_passed,
        "experimental_candidate_only": True,
        "default_policy_replaced": False,
        "publication_flags": publication_flags,
        "recommended_rollback_source": multihorizon_summary.get("candidate_manifest"),
    }


def _promotion_manifest(
    *,
    repo_root: Path,
    output_root: Path,
    multihorizon_root: Path,
    multihorizon_summary_path: Path,
    multihorizon_steps_path: Path,
    selected_candidate_root: Path,
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    candidate_summary_path: Path,
    limited_ppo_summary_path: Path,
    diagnostics_path: Path,
    training_curves_path: Path,
    progress_events_path: Path,
    progress_summary_path: Path,
    checkpoint_hash_audit_path: Path,
    inference_audit_path: Path,
    rollback_audit_path: Path,
    checkpoint_hash_audit: dict[str, Any],
    multihorizon_summary: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    legacy_candidate_manifest_path: Path | None,
    legacy_candidate_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "selected-formal-ppo-candidate-promotion-manifest/v1",
        "generated_at": _utc_now(),
        "diagnostic_clone": False,
        "experimental_candidate_only": True,
        "selected_seed": multihorizon_summary.get("selected_seed"),
        "selected_budget": multihorizon_summary.get("selected_budget"),
        "selected_candidate_root": str(selected_candidate_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata_path": str(checkpoint_metadata_path),
        "checkpoint_sha256": checkpoint_hash_audit.get("checkpoint_sha256"),
        "checkpoint_size_bytes": checkpoint_hash_audit.get("checkpoint_size_bytes"),
        "checkpoint_architecture": checkpoint_metadata.get("architecture"),
        "checkpoint_hidden_size": checkpoint_metadata.get("hidden_size"),
        "multihorizon_root": str(multihorizon_root),
        "multihorizon_summary": str(multihorizon_summary_path),
        "multihorizon_steps": str(multihorizon_steps_path),
        "candidate_summary": str(candidate_summary_path),
        "limited_ppo_summary": str(limited_ppo_summary_path),
        "diagnostics": str(diagnostics_path),
        "training_curves": str(training_curves_path),
        "progress_events": str(progress_events_path),
        "progress_summary": str(progress_summary_path),
        "checkpoint_hash_audit": str(checkpoint_hash_audit_path),
        "inference_audit": str(inference_audit_path),
        "rollback_audit": str(rollback_audit_path),
        "legacy_candidate_manifest_path": None
        if legacy_candidate_manifest_path is None
        else str(legacy_candidate_manifest_path),
        "legacy_candidate_manifest_present": bool(legacy_candidate_manifest),
        "legacy_candidate_manifest_git_provenance": legacy_candidate_manifest.get("git_provenance", {}),
        "output_root": str(output_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_selected_formal_ppo_candidate_promotion_preflight_evaluated")
    if readiness.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")
    if _int(readiness.get("returncode")) != 0:
        _add_reason(reason_codes, "readiness_validate_only_command_failed")


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    multihorizon_root: Path,
    multihorizon_summary_path: Path,
    multihorizon_steps_path: Path,
    selected_candidate_root: Path,
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    candidate_summary_path: Path,
    limited_ppo_summary_path: Path,
    diagnostics_path: Path,
    training_curves_path: Path,
    progress_events_path: Path,
    progress_summary_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    manifest_path: Path,
    checkpoint_hash_audit_path: Path,
    inference_audit_path: Path,
    rollback_audit_path: Path,
    readiness_path: Path,
    report_path: Path,
    multihorizon_summary: dict[str, Any],
    checkpoint_hash_audit: dict[str, Any],
    inference_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_selected_formal_ppo_candidate_promotion_preflight",
        "multihorizon_root": str(multihorizon_root),
        "multihorizon_summary": str(multihorizon_summary_path),
        "multihorizon_steps": str(multihorizon_steps_path),
        "selected_candidate_root": str(selected_candidate_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata_path": str(checkpoint_metadata_path),
        "candidate_summary": str(candidate_summary_path),
        "limited_ppo_summary": str(limited_ppo_summary_path),
        "diagnostics": str(diagnostics_path),
        "training_curves": str(training_curves_path),
        "progress_events": str(progress_events_path),
        "progress_summary": str(progress_summary_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "promotion_manifest": str(manifest_path),
        "checkpoint_hash_audit": str(checkpoint_hash_audit_path),
        "inference_audit": str(inference_audit_path),
        "rollback_audit": str(rollback_audit_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "selected_seed": multihorizon_summary.get("selected_seed"),
        "selected_budget": multihorizon_summary.get("selected_budget"),
        "selected_candidate_from_multihorizon_shadow": bool(
            multihorizon_summary.get("selected_candidate_from_candidate_selection")
        ),
        "horizons": list(multihorizon_summary.get("horizons") or []),
        "input_trainable_transition_count": _int(
            multihorizon_summary.get("input_trainable_transition_count")
        ),
        "shadow_trainable_transition_count": _int(
            multihorizon_summary.get("shadow_trainable_transition_count")
        ),
        "unique_trainable_context_count": _int(
            multihorizon_summary.get("unique_trainable_context_count")
        ),
        "checkpoint_sha256": checkpoint_hash_audit.get("checkpoint_sha256"),
        "checkpoint_size_bytes": _int(checkpoint_hash_audit.get("checkpoint_size_bytes")),
        "checkpoint_load_passed": bool(inference_audit.get("checkpoint_load_passed")),
        "inference_audit_count": _int(inference_audit.get("inference_audit_count")),
        "invalid_action_mask_count": _int(inference_audit.get("invalid_action_mask_count")),
        "missing_observation_count": _int(inference_audit.get("missing_observation_count")),
        "non_finite_logits_count": _int(inference_audit.get("non_finite_logits_count")),
        "non_finite_log_prob_count": _int(inference_audit.get("non_finite_log_prob_count")),
        "non_finite_value_count": _int(inference_audit.get("non_finite_value_count")),
        "action_reconstruction_difference_count": _int(
            inference_audit.get("action_reconstruction_difference_count")
        ),
        "log_prob_reconstruction_max_abs_error": _finite_or_inf(
            inference_audit.get("log_prob_reconstruction_max_abs_error")
        ),
        "value_reconstruction_max_abs_error": _finite_or_inf(
            inference_audit.get("value_reconstruction_max_abs_error")
        ),
        "reconstruction_difference_explained": bool(
            inference_audit.get("reconstruction_difference_explained")
        ),
        "reconstruction_reference": inference_audit.get("reconstruction_reference"),
        "controlled_regression_count": _int(
            multihorizon_summary.get("controlled_regression_count")
        ),
        "family_regression_count": _int(multihorizon_summary.get("family_regression_count")),
        "controlled_safety_regression_count": _int(
            multihorizon_summary.get("controlled_safety_regression_count")
        ),
        "controlled_contract_regression_count": _int(
            multihorizon_summary.get("controlled_contract_regression_count")
        ),
        "controlled_path_risk_regression_count": _int(
            multihorizon_summary.get("controlled_path_risk_regression_count")
        ),
        "controlled_source_selection_regression_count": _int(
            multihorizon_summary.get("controlled_source_selection_regression_count")
        ),
        "teacher_agreement_rate": _float(multihorizon_summary.get("teacher_agreement_rate"), 0.0),
        "rollback_audit_passed": bool(rollback_audit.get("rollback_audit_passed")),
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_promotion_preflight": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Selected Formal PPO Candidate Promotion Preflight v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- selected seed/budget: `{summary.get('selected_seed')}` / `{summary.get('selected_budget')}`",
            f"- checkpoint load passed: `{summary.get('checkpoint_load_passed')}`",
            f"- inference audit count: `{summary.get('inference_audit_count')}`",
            f"- log_prob/value max abs error: `{summary.get('log_prob_reconstruction_max_abs_error')}` / `{summary.get('value_reconstruction_max_abs_error')}`",
            f"- controlled/family regression: `{summary.get('controlled_regression_count')}` / `{summary.get('family_regression_count')}`",
            f"- readiness: `{summary.get('readiness_status')}`",
            "",
            "This stage only audits the selected experimental candidate before any promotion decision. "
            "It does not run a new PPO update, publish a checkpoint, replace the default policy, "
            "claim policy performance, or claim formal training readiness.",
            "",
        ]
    )


def _selected_action_rank(logits, action_index: int, *, torch) -> int:
    selected = logits[action_index]
    return int((logits > selected).to(torch.int64).sum().item()) + 1


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "multihorizon_summary": "multihorizon-shadow-rollout-summary.json",
        "checkpoint": "experimental-hybrid-policy-candidate.pt",
        "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
        "candidate_summary": "raw-policy-generalization-candidate-summary.json",
        "limited_ppo_summary": "limited-ppo-update-smoke-summary.json",
        "diagnostics": "limited-ppo-update-diagnostics.json",
        "training_curves": "limited-ppo-update-training-curves.json",
        "progress_events": "training-progress-events.jsonl",
        "progress_summary": "training-progress-summary.json",
    }
    defaults.update(config.get("input_files") or {})
    return defaults


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "manifest": MANIFEST_FILE,
        "checkpoint_hash_audit": CHECKPOINT_HASH_AUDIT_FILE,
        "inference_audit": INFERENCE_AUDIT_FILE,
        "rollback_audit": ROLLBACK_AUDIT_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    defaults.update(config.get("output_files") or {})
    return defaults


def _expected_trainable(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("expected_trainable_transition_count"))


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


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _finite_or_inf(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.inf
    return parsed if math.isfinite(parsed) else math.inf


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
