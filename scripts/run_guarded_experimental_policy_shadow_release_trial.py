from __future__ import annotations

import argparse
import hashlib
import json
import math
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


CONFIG_SCHEMA_VERSION = "guarded-experimental-policy-shadow-release-trial-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-experimental-policy-shadow-release-trial-summary/v1"
RUNTIME_MANIFEST_SCHEMA_VERSION = (
    "guarded-experimental-policy-shadow-release-trial-runtime-manifest/v1"
)
INSTALL_CANARY_SCHEMA_VERSION = "guarded-experimental-policy-install-canary-dry-run-summary/v1"
MULTIHORIZON_SHADOW_SCHEMA_VERSION = (
    "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1"
)
EXPECTED_INSTALL_CANARY_VERDICT = "eligible_for_guarded_shadow_release_trial"
EXPECTED_SHADOW_TRIAL_VERDICT = "eligible_for_guarded_staged_release_preflight"
EXPECTED_READINESS_STATUS = "guarded_experimental_policy_shadow_release_trial_evaluated"

SUMMARY_FILE = "guarded-experimental-policy-shadow-release-trial-summary.json"
RUNTIME_MANIFEST_FILE = "shadow-release-runtime-manifest.json"
STEP_COMPARISON_FILE = "shadow-release-step-comparison.jsonl"
REJECTION_REPORT_FILE = "shadow-release-rejection-report.json"
RISK_REWARD_AUDIT_FILE = "shadow-release-risk-reward-audit.json"
READINESS_FILE = "shadow-release-readiness-validate-only.json"
REPORT_FILE = "shadow-release-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_experimental_policy_shadow_release_trial(
    *,
    install_canary_root: Path,
    multihorizon_shadow_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    install_canary_root = Path(install_canary_root)
    multihorizon_shadow_root = Path(multihorizon_shadow_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    runtime_manifest_path = output_root / files["runtime_manifest"]
    step_comparison_path = output_root / files["step_comparison"]
    rejection_report_path = output_root / files["rejection_report"]
    risk_reward_audit_path = output_root / files["risk_reward_audit"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    inputs = _input_files(config)
    install_canary_summary_path = install_canary_root / inputs["install_canary_summary"]
    multihorizon_summary_path = (
        multihorizon_shadow_root / inputs["multihorizon_shadow_summary"]
    )
    install_canary_summary = _read_json_if_exists(install_canary_summary_path)
    multihorizon_summary = _read_json_if_exists(multihorizon_summary_path)
    source_steps_path = (
        _resolve_optional_path(
            multihorizon_summary.get("steps"),
            multihorizon_shadow_root,
            repo_root,
        )
        or multihorizon_shadow_root / inputs["multihorizon_shadow_steps"]
    )
    source_steps = _read_jsonl(source_steps_path)

    reason_codes: list[str] = []
    _validate_install_canary_input(install_canary_summary, config, reason_codes)
    _validate_multihorizon_input(multihorizon_summary, source_steps, config, reason_codes)

    default_before = _default_snapshot(config, install_canary_summary, repo_root)
    runtime_manifest = _runtime_manifest(
        repo_root=repo_root,
        output_root=output_root,
        install_canary_root=install_canary_root,
        multihorizon_shadow_root=multihorizon_shadow_root,
        install_canary_summary_path=install_canary_summary_path,
        multihorizon_summary_path=multihorizon_summary_path,
        source_steps_path=source_steps_path,
        install_canary_summary=install_canary_summary,
        default_before=default_before,
        default_after=default_before,
    )
    _write_json(runtime_manifest_path, runtime_manifest)
    if not runtime_manifest["shadow_runtime_manifest_passed"]:
        _add_reason(reason_codes, "shadow_release_trial_runtime_manifest_invalid")

    step_rows = _build_step_comparison(source_steps)
    counters = _step_counters(step_rows)
    _write_jsonl(step_comparison_path, step_rows)
    _write_json(rejection_report_path, _rejection_report(step_rows, counters))
    _write_json(risk_reward_audit_path, _risk_reward_audit(step_rows, counters))
    _validate_shadow_trial(counters, config, reason_codes)

    default_after = _default_snapshot(config, install_canary_summary, repo_root)
    runtime_manifest = _runtime_manifest(
        repo_root=repo_root,
        output_root=output_root,
        install_canary_root=install_canary_root,
        multihorizon_shadow_root=multihorizon_shadow_root,
        install_canary_summary_path=install_canary_summary_path,
        multihorizon_summary_path=multihorizon_summary_path,
        source_steps_path=source_steps_path,
        install_canary_summary=install_canary_summary,
        default_before=default_before,
        default_after=default_after,
    )
    _write_json(runtime_manifest_path, runtime_manifest)
    if runtime_manifest.get("default_policy_unchanged") is False:
        _add_reason(reason_codes, "shadow_release_trial_default_policy_modified")

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        install_canary_root=install_canary_root,
        multihorizon_shadow_root=multihorizon_shadow_root,
        output_root=output_root,
        batch_root=batch_root,
        install_canary_summary_path=install_canary_summary_path,
        multihorizon_summary_path=multihorizon_summary_path,
        source_steps_path=source_steps_path,
        summary_path=summary_path,
        runtime_manifest_path=runtime_manifest_path,
        step_comparison_path=step_comparison_path,
        rejection_report_path=rejection_report_path,
        risk_reward_audit_path=risk_reward_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        install_canary_summary=install_canary_summary,
        multihorizon_summary=multihorizon_summary,
        runtime_manifest=runtime_manifest,
        counters=counters,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            shadow_release_trial_summary_path=summary_path,
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
            "recommended_next_action": "fix_guarded_experimental_policy_shadow_release_trial",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        install_canary_root=install_canary_root,
        multihorizon_shadow_root=multihorizon_shadow_root,
        output_root=output_root,
        batch_root=batch_root,
        install_canary_summary_path=install_canary_summary_path,
        multihorizon_summary_path=multihorizon_summary_path,
        source_steps_path=source_steps_path,
        summary_path=summary_path,
        runtime_manifest_path=runtime_manifest_path,
        step_comparison_path=step_comparison_path,
        rejection_report_path=rejection_report_path,
        risk_reward_audit_path=risk_reward_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        install_canary_summary=install_canary_summary,
        multihorizon_summary=multihorizon_summary,
        runtime_manifest=runtime_manifest,
        counters=counters,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a guarded experimental policy shadow release trial."
    )
    parser.add_argument(
        "--install-canary-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1",
    )
    parser.add_argument(
        "--multihorizon-shadow-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/guarded_experimental_policy_shadow_release_trial_v1.json",
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

    summary = run_guarded_experimental_policy_shadow_release_trial(
        install_canary_root=_resolve_path(Path(args.install_canary_root), repo_root),
        multihorizon_shadow_root=_resolve_path(Path(args.multihorizon_shadow_root), repo_root),
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
                "shadow_release_trial_verdict": summary["shadow_release_trial_verdict"],
                "readiness_status": summary.get("readiness_status"),
                "shadow_step_count": summary.get("shadow_step_count"),
                "unique_shadow_context_count": summary.get("unique_shadow_context_count"),
                "controlled_regression_count": summary.get("controlled_regression_count"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _validate_install_canary_input(
    summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if summary.get("schema_version") != INSTALL_CANARY_SCHEMA_VERSION:
        _add_reason(reason_codes, "shadow_release_trial_install_canary_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "shadow_release_trial_install_canary_not_passed")
    if summary.get("install_canary_verdict") != EXPECTED_INSTALL_CANARY_VERDICT:
        _add_reason(reason_codes, "shadow_release_trial_install_canary_not_eligible")
    if _int(summary.get("canary_step_count")) < _int(
        config.get("validation", {}).get("min_install_canary_step_count"),
        64,
    ):
        _add_reason(reason_codes, "shadow_release_trial_install_canary_step_count_low")
    for field, reason in (
        ("missing_observation_count", "shadow_release_trial_install_canary_contract_invalid"),
        ("invalid_action_mask_count", "shadow_release_trial_install_canary_contract_invalid"),
        ("non_finite_logits_count", "shadow_release_trial_install_canary_non_finite"),
        ("non_finite_log_prob_count", "shadow_release_trial_install_canary_non_finite"),
        ("non_finite_value_count", "shadow_release_trial_install_canary_non_finite"),
        ("non_finite_reward_count", "shadow_release_trial_install_canary_non_finite"),
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(reason_codes, reason)
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(reason_codes, "shadow_release_trial_install_canary_controlled_regression")
    if summary.get("rollback_default_audit_passed") is not True:
        _add_reason(reason_codes, "shadow_release_trial_install_canary_rollback_failed")
    if summary.get("default_policy_unchanged") is False:
        _add_reason(reason_codes, "shadow_release_trial_install_canary_default_changed")
    for field, reason in (
        ("runs_new_ppo_update", "shadow_release_trial_unexpected_ppo_update"),
        ("publishes_checkpoint", "shadow_release_trial_checkpoint_publication_claimed"),
        ("replaces_default_policy", "shadow_release_trial_default_policy_replacement_claimed"),
        ("performance_claimed", "shadow_release_trial_policy_performance_claimed"),
        ("formal_training_ready_claimed", "shadow_release_trial_formal_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)
    package_sha = summary.get("package_checkpoint_sha256")
    consumer_sha = summary.get("consumer_checkpoint_sha256")
    package_size = _int(summary.get("package_checkpoint_size_bytes"))
    consumer_size = _int(summary.get("consumer_checkpoint_size_bytes"))
    if not isinstance(package_sha, str) or len(package_sha) != 64 or package_sha != consumer_sha:
        _add_reason(reason_codes, "shadow_release_trial_checkpoint_hash_mismatch")
    if package_size <= 0 or package_size != consumer_size:
        _add_reason(reason_codes, "shadow_release_trial_checkpoint_size_mismatch")
    if _git_current_matches_sources(summary) is False:
        _add_reason(reason_codes, "shadow_release_trial_install_canary_git_provenance_mismatch")


def _validate_multihorizon_input(
    summary: dict[str, Any],
    source_steps: list[dict[str, Any]],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if summary.get("schema_version") != MULTIHORIZON_SHADOW_SCHEMA_VERSION:
        _add_reason(reason_codes, "shadow_release_trial_multihorizon_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "shadow_release_trial_multihorizon_not_passed")
    if summary.get("runs_multihorizon_shadow_rollout") is not True:
        _add_reason(reason_codes, "shadow_release_trial_multihorizon_not_run")
    if _int(summary.get("shadow_trainable_transition_count")) < _min_shadow_step_count(config):
        _add_reason(reason_codes, "shadow_release_trial_multihorizon_step_count_low")
    if _int(summary.get("unique_trainable_context_count")) < _min_unique_context_count(config):
        _add_reason(reason_codes, "shadow_release_trial_multihorizon_unique_context_low")
    if len(source_steps) < _min_shadow_step_count(config):
        _add_reason(reason_codes, "shadow_release_trial_source_step_count_low")
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(reason_codes, "shadow_release_trial_multihorizon_controlled_regression")
    for field, reason in (
        ("runs_new_ppo_update", "shadow_release_trial_unexpected_ppo_update"),
        ("publishes_checkpoint", "shadow_release_trial_checkpoint_publication_claimed"),
        ("replaces_default_policy", "shadow_release_trial_default_policy_replacement_claimed"),
        ("performance_claimed", "shadow_release_trial_policy_performance_claimed"),
        ("formal_training_ready_claimed", "shadow_release_trial_formal_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)
    if _git_current_matches_sources(summary) is False:
        _add_reason(reason_codes, "shadow_release_trial_multihorizon_git_provenance_mismatch")


def _build_step_comparison(source_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(source_steps):
        gate_reasons = _string_list(step.get("gate_reason_codes"))
        rejection_reasons = _string_list(step.get("rejection_reason_codes"))
        controlled_reasons = _string_list(step.get("controlled_regression_reason_codes"))
        source = str(step.get("controlled_choice_source") or "")
        rows.append(
            {
                "schema_version": "guarded-experimental-policy-shadow-release-trial-step/v1",
                "source_index": index,
                "context_id": step.get("context_id"),
                "scenario_id": step.get("scenario_id"),
                "scenario_family": step.get("scenario_family"),
                "split": step.get("split"),
                "default_policy_action_index": step.get("controlled_action_index"),
                "teacher_action_index": step.get("teacher_action_index"),
                "controlled_baseline_action_index": step.get("controlled_action_index"),
                "controlled_choice_source": source,
                "experimental_shadow_action_index": step.get(
                    "raw_policy_action_index",
                    step.get("controlled_action_index"),
                ),
                "shadow_policy_takes_control": False,
                "shadow_diagnostic_only": bool(gate_reasons or rejection_reasons or source != "policy"),
                "shadow_gate_reason_codes": gate_reasons + rejection_reasons,
                "controlled_regression_reason_codes": controlled_reasons,
                "missing_observation": step.get("missing_observation") is True
                or not isinstance(step.get("observation"), dict),
                "invalid_action_mask": _invalid_action_mask(step),
                "non_finite_logits": step.get("non_finite_logits") is True,
                "non_finite_log_prob": not _finite(step.get("log_prob")),
                "non_finite_value": not _finite(step.get("value")),
                "non_finite_reward": not _finite(step.get("reward")),
                "log_prob": step.get("log_prob"),
                "value": step.get("value"),
                "reward": step.get("reward"),
                "shadow_discounted_return": step.get(
                    "shadow_discounted_return",
                    step.get("discounted_return"),
                ),
                "shadow_advantage": step.get("shadow_advantage", step.get("advantage")),
                "path_cost_delta": _float(step.get("path_cost_delta")),
                "risk_delta": _float(step.get("risk_delta")),
            }
        )
    return rows


def _step_counters(rows: list[dict[str, Any]]) -> Counter[str]:
    counters: Counter[str] = Counter()
    counters["shadow_step_count"] = len(rows)
    counters["unique_shadow_context_count"] = len(
        {row.get("context_id") for row in rows if row.get("context_id")}
    )
    for row in rows:
        if row.get("shadow_diagnostic_only"):
            counters["shadow_diagnostic_count"] += 1
        if _string_list(row.get("shadow_gate_reason_codes")):
            counters["shadow_rejection_diagnostic_count"] += 1
        if row.get("controlled_choice_source") != "policy":
            counters["shadow_fallback_diagnostic_count"] += 1
        for key in (
            "missing_observation",
            "invalid_action_mask",
            "non_finite_logits",
            "non_finite_log_prob",
            "non_finite_value",
            "non_finite_reward",
        ):
            if row.get(key):
                counters[f"{key}_count"] += 1
        if row.get("missing_observation"):
            counters["missing_log_prob_count"] += 1
            counters["missing_value_count"] += 1
        reasons = set(_string_list(row.get("controlled_regression_reason_codes")))
        if reasons:
            counters["controlled_regression_count"] += 1
        if "safety_regression" in reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_regression" in reasons or "contract_violation" in reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in reasons or "risk_regression" in reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in reasons:
            counters["controlled_source_selection_regression_count"] += 1
    return counters


def _validate_shadow_trial(
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if counters["shadow_step_count"] < _min_shadow_step_count(config):
        _add_reason(reason_codes, "shadow_release_trial_step_count_below_threshold")
    if counters["unique_shadow_context_count"] < _min_unique_context_count(config):
        _add_reason(reason_codes, "shadow_release_trial_unique_context_below_threshold")
    for field, reason in (
        ("missing_observation_count", "shadow_release_trial_missing_observation"),
        ("missing_log_prob_count", "shadow_release_trial_missing_log_prob_or_value"),
        ("missing_value_count", "shadow_release_trial_missing_log_prob_or_value"),
        ("invalid_action_mask_count", "shadow_release_trial_invalid_action_mask"),
        ("non_finite_logits_count", "shadow_release_trial_non_finite"),
        ("non_finite_log_prob_count", "shadow_release_trial_non_finite"),
        ("non_finite_value_count", "shadow_release_trial_non_finite"),
        ("non_finite_reward_count", "shadow_release_trial_non_finite"),
    ):
        if counters[field] > 0:
            _add_reason(reason_codes, reason)
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if counters[field] > 0:
            _add_reason(reason_codes, "shadow_release_trial_controlled_regression")


def _runtime_manifest(
    *,
    repo_root: Path,
    output_root: Path,
    install_canary_root: Path,
    multihorizon_shadow_root: Path,
    install_canary_summary_path: Path,
    multihorizon_summary_path: Path,
    source_steps_path: Path,
    install_canary_summary: dict[str, Any],
    default_before: dict[str, Any],
    default_after: dict[str, Any],
) -> dict[str, Any]:
    default_unchanged = _snapshots_same(default_before, default_after)
    default_boundary_ok = default_unchanged and install_canary_summary.get(
        "default_policy_unchanged"
    ) is not False
    checkpoint_path = _resolve_optional_path(
        install_canary_summary.get("package_checkpoint_path"),
        install_canary_root,
        repo_root,
    )
    checkpoint_exists = bool(checkpoint_path and checkpoint_path.is_file())
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "shadow_runtime_manifest_passed": default_boundary_ok and checkpoint_exists,
        "output_root": str(output_root),
        "install_canary_root": str(install_canary_root),
        "multihorizon_shadow_root": str(multihorizon_shadow_root),
        "install_canary_summary": str(install_canary_summary_path),
        "multihorizon_shadow_summary": str(multihorizon_summary_path),
        "source_steps": str(source_steps_path),
        "package_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "package_checkpoint_exists": checkpoint_exists,
        "package_checkpoint_sha256": install_canary_summary.get("package_checkpoint_sha256"),
        "consumer_checkpoint_sha256": install_canary_summary.get("consumer_checkpoint_sha256"),
        "package_checkpoint_size_bytes": _int(
            install_canary_summary.get("package_checkpoint_size_bytes")
        ),
        "consumer_checkpoint_size_bytes": _int(
            install_canary_summary.get("consumer_checkpoint_size_bytes")
        ),
        "default_policy_path": default_before.get("path") or default_after.get("path"),
        "default_policy_exists_before": default_before.get("exists"),
        "default_policy_exists_after": default_after.get("exists"),
        "default_policy_sha256_before": default_before.get("sha256"),
        "default_policy_sha256_after": default_after.get("sha256"),
        "default_policy_unchanged": default_unchanged,
        "shadow_policy_takes_control": False,
        "writes_default_policy": False,
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
    repo_root: Path,
    install_canary_root: Path,
    multihorizon_shadow_root: Path,
    output_root: Path,
    batch_root: Path,
    install_canary_summary_path: Path,
    multihorizon_summary_path: Path,
    source_steps_path: Path,
    summary_path: Path,
    runtime_manifest_path: Path,
    step_comparison_path: Path,
    rejection_report_path: Path,
    risk_reward_audit_path: Path,
    readiness_path: Path,
    report_path: Path,
    install_canary_summary: dict[str, Any],
    multihorizon_summary: dict[str, Any],
    runtime_manifest: dict[str, Any],
    counters: Counter[str],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_guarded_experimental_policy_shadow_release_trial",
        "shadow_release_trial_verdict": _shadow_trial_verdict(reason_codes),
        "install_canary_root": str(install_canary_root),
        "install_canary_summary": str(install_canary_summary_path),
        "multihorizon_shadow_root": str(multihorizon_shadow_root),
        "multihorizon_shadow_summary": str(multihorizon_summary_path),
        "source_steps": str(source_steps_path),
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "runtime_manifest": str(runtime_manifest_path),
        "step_comparison": str(step_comparison_path),
        "rejection_report": str(rejection_report_path),
        "risk_reward_audit": str(risk_reward_audit_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "package_checkpoint_sha256": install_canary_summary.get("package_checkpoint_sha256"),
        "consumer_checkpoint_sha256": install_canary_summary.get("consumer_checkpoint_sha256"),
        "package_checkpoint_size_bytes": _int(
            install_canary_summary.get("package_checkpoint_size_bytes")
        ),
        "consumer_checkpoint_size_bytes": _int(
            install_canary_summary.get("consumer_checkpoint_size_bytes")
        ),
        "runtime_manifest_passed": runtime_manifest.get("shadow_runtime_manifest_passed") is True,
        "shadow_step_count": counters["shadow_step_count"],
        "unique_shadow_context_count": counters["unique_shadow_context_count"],
        "source_shadow_trainable_transition_count": _int(
            multihorizon_summary.get("shadow_trainable_transition_count")
        ),
        "source_unique_trainable_context_count": _int(
            multihorizon_summary.get("unique_trainable_context_count")
        ),
        "shadow_diagnostic_count": counters["shadow_diagnostic_count"],
        "shadow_rejection_diagnostic_count": counters["shadow_rejection_diagnostic_count"],
        "shadow_fallback_diagnostic_count": counters["shadow_fallback_diagnostic_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "invalid_action_mask_count": counters["invalid_action_mask_count"],
        "non_finite_logits_count": counters["non_finite_logits_count"],
        "non_finite_log_prob_count": counters["non_finite_log_prob_count"],
        "non_finite_value_count": counters["non_finite_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "rollback_default_audit_passed": install_canary_summary.get(
            "rollback_default_audit_passed"
        )
        is True,
        "default_policy_unchanged": runtime_manifest.get("default_policy_unchanged") is True
        and install_canary_summary.get("default_policy_unchanged") is not False,
        "shadow_policy_takes_control": False,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_shadow_release_trial": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "recommended_next_action": "guarded_staged_release_preflight"
        if status == "passed"
        else "fix_guarded_experimental_policy_shadow_release_trial",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    shadow_release_trial_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-experimental-policy-shadow-release-trial-summary",
        str(shadow_release_trial_summary_path),
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
        _add_reason(reason_codes, "shadow_release_trial_readiness_status_mismatch")
    if _string_list(readiness.get("training_blockers")):
        _add_reason(reason_codes, "shadow_release_trial_readiness_blocked")
    if _string_list(readiness.get("reason_codes")):
        _add_reason(reason_codes, "shadow_release_trial_readiness_reason_codes")
    if _int(readiness.get("returncode")) not in (0,):
        _add_reason(reason_codes, "shadow_release_trial_readiness_command_failed")


def _rejection_report(rows: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    rejected = []
    for row in rows:
        reasons = _string_list(row.get("shadow_gate_reason_codes"))
        if row.get("controlled_choice_source") != "policy":
            reasons.append("shadow_fallback_diagnostic_only")
        if reasons:
            rejected.append(
                {
                    "context_id": row.get("context_id"),
                    "scenario_id": row.get("scenario_id"),
                    "split": row.get("split"),
                    "controlled_choice_source": row.get("controlled_choice_source"),
                    "reason_origin": "shadow_policy_probe",
                    "reasons": reasons,
                }
            )
    return {
        "schema_version": "guarded-experimental-policy-shadow-release-trial-rejection-report/v1",
        "shadow_rejection_diagnostic_count": counters["shadow_rejection_diagnostic_count"],
        "shadow_fallback_diagnostic_count": counters["shadow_fallback_diagnostic_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "rows": rejected,
    }


def _risk_reward_audit(rows: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    rewards = [_float(row.get("reward"), math.nan) for row in rows]
    return {
        "schema_version": "guarded-experimental-policy-shadow-release-trial-risk-reward-audit/v1",
        "shadow_step_count": counters["shadow_step_count"],
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_path_risk_regression_count": counters[
            "controlled_path_risk_regression_count"
        ],
        "shadow_rejection_is_diagnostic_only": True,
        "shadow_policy_takes_control": False,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Experimental Policy Shadow Release Trial v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- shadow_release_trial_verdict: `{summary['shadow_release_trial_verdict']}`",
            f"- shadow_step_count: `{summary.get('shadow_step_count')}`",
            f"- unique_shadow_context_count: `{summary.get('unique_shadow_context_count')}`",
            f"- controlled_regression_count: `{summary.get('controlled_regression_count')}`",
            f"- readiness_status: `{summary.get('readiness_status')}`",
            "",
            "This stage keeps the experimental policy in shadow mode only. "
            "The default policy remains authoritative; rejected or fallback shadow decisions "
            "are diagnostic and do not become controlled rollout regressions.",
            "",
        ]
    )


def _shadow_trial_verdict(reason_codes: list[str]) -> str:
    if not reason_codes:
        return EXPECTED_SHADOW_TRIAL_VERDICT
    if any("install_canary" in reason or "checkpoint" in reason for reason in reason_codes):
        return "blocked_by_install_canary_or_package"
    return "blocked_by_shadow_release_trial"


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "runtime_manifest": RUNTIME_MANIFEST_FILE,
        "step_comparison": STEP_COMPARISON_FILE,
        "rejection_report": REJECTION_REPORT_FILE,
        "risk_reward_audit": RISK_REWARD_AUDIT_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    output_files = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(output_files.get(key) or default) for key, default in defaults.items()}


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "install_canary_summary": "guarded-experimental-policy-install-canary-dry-run-summary.json",
        "multihorizon_shadow_summary": "multihorizon-shadow-rollout-summary.json",
        "multihorizon_shadow_steps": "multihorizon-shadow-rollout-steps.jsonl",
    }
    input_files = config.get("input_files") if isinstance(config.get("input_files"), dict) else {}
    return {key: str(input_files.get(key) or default) for key, default in defaults.items()}


def _min_shadow_step_count(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("min_shadow_step_count"), 256)


def _min_unique_context_count(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("min_unique_shadow_context_count"), 256)


def _default_snapshot(
    config: dict[str, Any],
    install_canary_summary: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    value = config.get("default_policy", {}).get("path") or install_canary_summary.get(
        "default_policy_path"
    )
    if not value:
        return {"path": None, "exists": None, "sha256": None, "size_bytes": None}
    path = _resolve_path(Path(str(value)), repo_root)
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "size_bytes": 0}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _snapshots_same(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("path") == right.get("path")
        and left.get("exists") == right.get("exists")
        and left.get("sha256") == right.get("sha256")
        and _int(left.get("size_bytes")) == _int(right.get("size_bytes"))
    )


def _invalid_action_mask(step: dict[str, Any]) -> bool:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return False
    mask = observation.get("action_mask")
    if not isinstance(mask, list) or not mask or not any(bool(item) for item in mask):
        return True
    action = _int(step.get("raw_policy_action_index", step.get("controlled_action_index")), -1)
    return action < 0 or action >= len(mask) or not bool(mask[action])


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return _resolve_path(path, repo_root if path.is_absolute() else base)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    return _read_json(path) if path and path.is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _git_current_matches_sources(payload: dict[str, Any]) -> bool | None:
    provenance = payload.get("git_provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("current_matches_sources")
    return value if isinstance(value, bool) else None


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
