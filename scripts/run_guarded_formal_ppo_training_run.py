from __future__ import annotations

import argparse
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

try:
    from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _run_seed_smoke,
        _unique_trainable_steps,
    )
except ModuleNotFoundError:  # pragma: no cover
    from run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _run_seed_smoke,
        _unique_trainable_steps,
    )


CONFIG_SCHEMA_VERSION = "guarded-formal-ppo-training-run-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-formal-ppo-training-run-summary/v1"
SEED_SCHEMA_VERSION = "guarded-formal-ppo-training-run-seed-summary/v1"
EXPECTED_READINESS_STATUS = "guarded_formal_ppo_training_run_evaluated"
EXPECTED_AUTHORIZATION_VERDICT = "authorized_for_guarded_formal_ppo_training_run"

AUTHORIZATION_SUMMARY_FILE = "formal-ppo-training-authorization-summary.json"
SUMMARY_FILE = "formal-ppo-training-run-summary.json"
SEED_SUMMARIES_FILE = "formal-ppo-training-run-seed-summaries.jsonl"
PROGRESS_FILE = "formal-ppo-training-run-progress.jsonl"
TRAINING_CURVES_FILE = "formal-ppo-training-run-training-curves.json"
GATE_AUDIT_FILE = "formal-ppo-training-run-gate-audit.json"
ROLLBACK_MANIFEST_FILE = "formal-ppo-training-run-rollback-manifest.json"
READINESS_FILE = "formal-ppo-training-run-readiness-validate-only.json"
REPORT_FILE = "formal-ppo-training-run-report.md"

SeedTrainingRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_formal_ppo_training_run(
    *,
    authorization_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    seed_training_runner: SeedTrainingRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    authorization_root = Path(authorization_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    seed_summaries_path = output_root / files["seed_summaries"]
    progress_path = output_root / files["progress"]
    training_curves_path = output_root / files["training_curves"]
    gate_audit_path = output_root / files["gate_audit"]
    rollback_manifest_path = output_root / files["rollback_manifest"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    authorization_summary_path = authorization_root / AUTHORIZATION_SUMMARY_FILE
    authorization_summary = _read_json_if_exists(authorization_summary_path)
    stability_summary_path = _resolve_optional_path(
        authorization_summary.get("formal_stability_holdout_summary"),
        authorization_root,
        repo_root,
    )
    stability_summary = _read_json_if_exists(stability_summary_path) if stability_summary_path else {}
    steps_path = _resolve_steps_path(stability_summary, stability_summary_path or authorization_root, repo_root)
    steps = _read_jsonl(steps_path)
    trainable_steps = _unique_trainable_steps(steps)
    counters = _input_counters(steps, trainable_steps)

    reason_codes: list[str] = []
    _validate_authorization(
        authorization_summary=authorization_summary,
        stability_summary=stability_summary,
        counters=counters,
        config=config,
        reason_codes=reason_codes,
    )

    progress_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    if not reason_codes:
        runner = seed_training_runner or _run_seed_training
        for seed_index, seed in enumerate(_seeds(config), start=1):
            progress_rows.append(
                _progress_row(seed=seed, event="seed_started", current=seed_index - 1, total=len(_seeds(config)))
            )
            seed_summary = runner(
                seed=seed,
                trainable_steps=trainable_steps,
                output_root=output_root,
                config=config,
                repo_root=repo_root,
                batch_root=batch_root,
            )
            normalized = _normalize_seed_summary(seed_summary, seed=seed)
            seed_summaries.append(normalized)
            progress_rows.append(
                _progress_row(
                    seed=seed,
                    event="seed_completed",
                    current=seed_index,
                    total=len(_seeds(config)),
                    metrics={
                        "optimizer_train_transition_count": normalized.get("optimizer_train_transition_count"),
                        "approx_kl": normalized.get("approx_kl"),
                        "max_grad_norm_after_clip": normalized.get("max_grad_norm_after_clip"),
                    },
                )
            )
        _validate_seed_summaries(seed_summaries, counters, config, reason_codes)

    _write_jsonl(seed_summaries_path, seed_summaries)
    _write_jsonl(progress_path, progress_rows)
    _write_json(training_curves_path, _training_curves(seed_summaries))
    _write_json(gate_audit_path, _gate_audit(counters, seed_summaries))
    _write_json(
        rollback_manifest_path,
        _rollback_manifest(
            output_root=output_root,
            authorization_summary_path=authorization_summary_path,
            seed_summaries=seed_summaries,
        ),
    )

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        authorization_root=authorization_root,
        authorization_summary_path=authorization_summary_path,
        stability_summary_path=stability_summary_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        seed_summaries_path=seed_summaries_path,
        progress_path=progress_path,
        training_curves_path=training_curves_path,
        gate_audit_path=gate_audit_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        authorization_summary=authorization_summary,
        counters=counters,
        seed_summaries=seed_summaries,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            training_run_summary_path=summary_path,
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
            "recommended_next_action": "fix_guarded_formal_ppo_training_run",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        authorization_root=authorization_root,
        authorization_summary_path=authorization_summary_path,
        stability_summary_path=stability_summary_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        seed_summaries_path=seed_summaries_path,
        progress_path=progress_path,
        training_curves_path=training_curves_path,
        gate_audit_path=gate_audit_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        authorization_summary=authorization_summary,
        counters=counters,
        seed_summaries=seed_summaries,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the guarded formal PPO training pass.")
    parser.add_argument(
        "--authorization-root",
        default="outputs/path_feedback_batch_guarded_formal_ppo_training_authorization_v1",
    )
    parser.add_argument("--batch-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1")
    parser.add_argument("--output-root", default="outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1")
    parser.add_argument("--config", default="configs/guarded_formal_ppo_training_run_v1.json")
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

    summary = run_guarded_formal_ppo_training_run(
        authorization_root=_resolve_path(Path(args.authorization_root), repo_root),
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
                "optimizer_train_transition_count": summary["optimizer_train_transition_count"],
                "seed_count": summary["seed_count"],
                "passed_seed_count": summary["passed_seed_count"],
                "readiness_status": summary["readiness_status"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_seed_training(
    *,
    seed: int,
    trainable_steps: list[dict[str, Any]],
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    batch_root: Path,
) -> dict[str, Any]:
    summary = _run_seed_smoke(
        seed=seed,
        trainable_steps=trainable_steps,
        output_root=output_root,
        config=config,
        repo_root=repo_root,
        batch_root=batch_root,
    )
    training = dict(summary)
    training["schema_version"] = SEED_SCHEMA_VERSION
    training["runs_guarded_formal_ppo_training_run"] = True
    training["runs_new_ppo_update"] = True
    training.setdefault("post_training_holdout_status", "passed" if training.get("status") == "passed" else "failed")
    training.setdefault("post_training_canary_status", "passed" if training.get("status") == "passed" else "failed")
    training.setdefault("experimental_checkpoint", True)
    training["publishes_checkpoint"] = False
    training["replaces_default_policy"] = False
    training["performance_claimed"] = False
    training["formal_training_ready_claimed"] = False
    training["training_curve_records"] = _seed_training_curve_records(training)
    return training


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    training_run_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-formal-ppo-training-run-summary",
        str(training_run_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(command, cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def _validate_authorization(
    *,
    authorization_summary: dict[str, Any],
    stability_summary: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = _expected_trainable(config)
    if authorization_summary.get("schema_version") != "guarded-formal-ppo-training-authorization-summary/v1":
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorization_schema_invalid")
    if authorization_summary.get("status") != "passed" or _string_list(authorization_summary.get("reason_codes")):
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorization_not_passed")
    if authorization_summary.get("authorization_verdict") != EXPECTED_AUTHORIZATION_VERDICT:
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorization_not_eligible")
    if authorization_summary.get("readiness_status") != "guarded_formal_ppo_training_authorized":
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorization_readiness_invalid")
    if _git_current_matches(authorization_summary) is False:
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorization_git_current_mismatch")
    if _int(authorization_summary.get("authorized_trainable_transition_count")) != expected:
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorized_count_mismatch")
    if _int(authorization_summary.get("authorized_optimizer_train_transition_count")) != expected:
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorized_count_mismatch")
    if _int(authorization_summary.get("unique_authorized_trainable_context_count")) != expected:
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_authorized_unique_count_mismatch")
    if authorization_summary.get("runs_new_ppo_update") is True:
        _add_reason(reason_codes, "formal_ppo_update_unexpected_in_authorization")
    for field, reason in (
        ("validation_trainable_count", "guarded_formal_ppo_training_run_split_leakage"),
        ("test_trainable_count", "guarded_formal_ppo_training_run_split_leakage"),
        ("fallback_trainable_count", "guarded_formal_ppo_training_run_fallback_trainable"),
        ("source_fallback_trainable_count", "guarded_formal_ppo_training_run_fallback_trainable"),
        ("teacher_fallback_trainable_count", "guarded_formal_ppo_training_run_fallback_trainable"),
        ("diagnostic_trainable_count", "guarded_formal_ppo_training_run_diagnostic_trainable"),
        ("non_empty_gate_reason_trainable_count", "guarded_formal_ppo_training_run_gate_reason_trainable"),
        ("missing_observation_count", "guarded_formal_ppo_training_run_contract_invalid"),
        ("missing_log_prob_count", "guarded_formal_ppo_training_run_contract_invalid"),
        ("missing_value_count", "guarded_formal_ppo_training_run_contract_invalid"),
        ("invalid_action_mask_count", "guarded_formal_ppo_training_run_contract_invalid"),
        ("non_finite_reward_count", "guarded_formal_ppo_training_run_non_finite"),
        ("non_finite_return_count", "guarded_formal_ppo_training_run_non_finite"),
        ("non_finite_advantage_count", "guarded_formal_ppo_training_run_non_finite"),
        ("controlled_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_safety_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_contract_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_path_risk_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_source_selection_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
    ):
        if _int(authorization_summary.get(field)):
            _add_reason(reason_codes, reason)
    for field, reason in (
        ("publishes_checkpoint", "limited_ppo_update_checkpoint_publication_claimed"),
        ("replaces_default_policy", "limited_ppo_update_default_policy_replacement_claimed"),
        ("performance_claimed", "limited_ppo_update_policy_performance_claimed"),
        ("formal_training_ready_claimed", "limited_ppo_update_formal_training_ready_claimed"),
    ):
        if authorization_summary.get(field) is True:
            _add_reason(reason_codes, reason)
    if stability_summary.get("status") != "passed" or _string_list(stability_summary.get("reason_codes")):
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_stability_source_not_passed")
    if counters["input_trainable_transition_count"] != expected:
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_trainable_count_mismatch")
    if counters["unique_trainable_context_count"] != expected:
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_unique_context_count_mismatch")
    for field, reason in (
        ("validation_trainable_count", "guarded_formal_ppo_training_run_split_leakage"),
        ("test_trainable_count", "guarded_formal_ppo_training_run_split_leakage"),
        ("fallback_trainable_count", "guarded_formal_ppo_training_run_fallback_trainable"),
        ("source_fallback_trainable_count", "guarded_formal_ppo_training_run_fallback_trainable"),
        ("teacher_fallback_trainable_count", "guarded_formal_ppo_training_run_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "guarded_formal_ppo_training_run_gate_reason_trainable"),
        ("missing_observation_count", "guarded_formal_ppo_training_run_contract_invalid"),
        ("missing_log_prob_count", "guarded_formal_ppo_training_run_contract_invalid"),
        ("missing_value_count", "guarded_formal_ppo_training_run_contract_invalid"),
        ("non_finite_reward_count", "guarded_formal_ppo_training_run_non_finite"),
        ("non_finite_return_count", "guarded_formal_ppo_training_run_non_finite"),
        ("non_finite_advantage_count", "guarded_formal_ppo_training_run_non_finite"),
        ("controlled_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_safety_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_contract_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_path_risk_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ("controlled_source_selection_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)


def _validate_seed_summaries(
    seed_summaries: list[dict[str, Any]],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = counters["input_trainable_transition_count"]
    validation = config.get("validation", {})
    if len(seed_summaries) < _int(validation.get("min_seed_count"), len(_seeds(config))):
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_seed_not_all_passed")
    if len(seed_summaries) != len(_seeds(config)):
        _add_reason(reason_codes, "guarded_formal_ppo_training_run_seed_not_all_passed")
    for summary in seed_summaries:
        if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
            _add_reason(reason_codes, "guarded_formal_ppo_training_run_seed_not_all_passed")
        if _int(summary.get("optimizer_train_transition_count")) != expected:
            _add_reason(reason_codes, "guarded_formal_ppo_training_run_optimizer_train_count_mismatch")
        if _float(summary.get("old_log_prob_max_abs_error"), math.inf) > _float(
            validation.get("max_old_log_prob_abs_error"), 1.0e-4
        ):
            _add_reason(reason_codes, "ppo_update_not_on_collector_policy")
        if _float(summary.get("old_value_max_abs_error"), math.inf) > _float(
            validation.get("max_old_value_abs_error"), 1.0e-4
        ):
            _add_reason(reason_codes, "ppo_update_not_on_collector_policy")
        if abs(_float(summary.get("approx_kl"), math.inf)) > _float(validation.get("max_abs_approx_kl"), 0.25):
            _add_reason(reason_codes, "ppo_update_too_large")
        if _float(summary.get("max_grad_norm_after_clip"), math.inf) > _float(
            validation.get("max_grad_norm_after_clip"), 1.0
        ) + 1.0e-8:
            _add_reason(reason_codes, "ppo_update_too_large")
        if _float(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
            _add_reason(reason_codes, "limited_ppo_update_input_contract_invalid")
        if _float(summary.get("teacher_agreement_rate"), 0.0) < _float(
            validation.get("min_teacher_agreement_rate"), 0.95
        ):
            _add_reason(reason_codes, "guarded_formal_ppo_training_run_teacher_alignment_insufficient")
        if summary.get("post_training_holdout_status") != "passed" or summary.get("post_training_canary_status") != "passed":
            _add_reason(reason_codes, "guarded_formal_ppo_training_run_post_training_gate_failed")
        for field, reason in (
            ("loss_non_finite_count", "ppo_update_loss_non_finite"),
            ("non_finite_gradient_count", "ppo_update_loss_non_finite"),
            ("non_finite_reward_count", "ppo_reward_contract_invalid"),
            ("non_finite_return_count", "ppo_update_loss_non_finite"),
            ("non_finite_advantage_count", "ppo_update_loss_non_finite"),
            ("controlled_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
            ("controlled_safety_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
            ("controlled_contract_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
            ("controlled_path_risk_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
            ("controlled_source_selection_regression_count", "guarded_formal_ppo_training_run_controlled_regression"),
        ):
            if _int(summary.get(field)):
                _add_reason(reason_codes, reason)
        for field, reason in (
            ("publishes_checkpoint", "limited_ppo_update_checkpoint_publication_claimed"),
            ("replaces_default_policy", "limited_ppo_update_default_policy_replacement_claimed"),
            ("performance_claimed", "limited_ppo_update_policy_performance_claimed"),
            ("formal_training_ready_claimed", "limited_ppo_update_formal_training_ready_claimed"),
        ):
            if summary.get(field) is True:
                _add_reason(reason_codes, reason)


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_guarded_formal_ppo_training_run_evaluated")
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
    authorization_root: Path,
    authorization_summary_path: Path,
    stability_summary_path: Path | None,
    steps_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    seed_summaries_path: Path,
    progress_path: Path,
    training_curves_path: Path,
    gate_audit_path: Path,
    rollback_manifest_path: Path,
    readiness_path: Path,
    report_path: Path,
    authorization_summary: dict[str, Any],
    counters: Counter[str],
    seed_summaries: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    metrics = _seed_metrics(seed_summaries)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None if status == "passed" else "fix_guarded_formal_ppo_training_run",
        "authorization_root": str(authorization_root),
        "authorization_summary": str(authorization_summary_path),
        "formal_stability_holdout_summary": str(stability_summary_path) if stability_summary_path else None,
        "steps": str(steps_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "seed_summaries": str(seed_summaries_path),
        "progress": str(progress_path),
        "training_curves": str(training_curves_path),
        "gate_audit": str(gate_audit_path),
        "rollback_manifest": str(rollback_manifest_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "input_authorization_status": authorization_summary.get("status"),
        "authorization_verdict": authorization_summary.get("authorization_verdict"),
        "authorized_trainable_transition_count": _int(authorization_summary.get("authorized_trainable_transition_count")),
        "authorized_optimizer_train_transition_count": _int(
            authorization_summary.get("authorized_optimizer_train_transition_count")
        ),
        "input_trainable_transition_count": counters["input_trainable_transition_count"],
        "optimizer_train_transition_count": metrics["optimizer_train_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "step_count": counters["step_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "fallback_trainable_count": counters["fallback_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "diagnostic_trainable_count": counters["diagnostic_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "invalid_action_mask_count": counters["invalid_action_mask_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        **metrics,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_guarded_formal_ppo_training_run": True,
        "runs_new_ppo_update": True,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "seeds": _seeds_from_summaries(seed_summaries) or _int_list(authorization_summary.get("seeds")),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _seed_metrics(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_summaries:
        return {
            "seed_count": 0,
            "passed_seed_count": 0,
            "seed_failure_count": 0,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "min_parameter_l2_delta": 0.0,
            "max_abs_approx_kl": 0.0,
            "max_grad_norm_after_clip": 0.0,
            "teacher_agreement_rate": 0.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "post_training_holdout_status": None,
            "post_training_canary_status": None,
        }
    return {
        "seed_count": len(seed_summaries),
        "passed_seed_count": sum(1 for item in seed_summaries if item.get("status") == "passed" and not item.get("reason_codes")),
        "seed_failure_count": sum(1 for item in seed_summaries if item.get("status") != "passed" or item.get("reason_codes")),
        "optimizer_train_transition_count": min(_int(item.get("optimizer_train_transition_count")) for item in seed_summaries),
        "max_old_log_prob_abs_error": max(_float(item.get("old_log_prob_max_abs_error")) for item in seed_summaries),
        "max_old_value_abs_error": max(_float(item.get("old_value_max_abs_error")) for item in seed_summaries),
        "loss_non_finite_count": sum(_int(item.get("loss_non_finite_count")) for item in seed_summaries),
        "non_finite_gradient_count": sum(_int(item.get("non_finite_gradient_count")) for item in seed_summaries),
        "non_finite_reward_count": sum(_int(item.get("non_finite_reward_count")) for item in seed_summaries),
        "non_finite_return_count": sum(_int(item.get("non_finite_return_count")) for item in seed_summaries),
        "non_finite_advantage_count": sum(_int(item.get("non_finite_advantage_count")) for item in seed_summaries),
        "min_parameter_l2_delta": min(_float(item.get("parameter_l2_delta")) for item in seed_summaries),
        "max_abs_approx_kl": max(abs(_float(item.get("approx_kl"))) for item in seed_summaries),
        "max_grad_norm_after_clip": max(_float(item.get("max_grad_norm_after_clip")) for item in seed_summaries),
        "teacher_agreement_rate": min(_float(item.get("teacher_agreement_rate")) for item in seed_summaries),
        "controlled_regression_count": sum(_int(item.get("controlled_regression_count")) for item in seed_summaries),
        "controlled_safety_regression_count": sum(_int(item.get("controlled_safety_regression_count")) for item in seed_summaries),
        "controlled_contract_regression_count": sum(_int(item.get("controlled_contract_regression_count")) for item in seed_summaries),
        "controlled_path_risk_regression_count": sum(_int(item.get("controlled_path_risk_regression_count")) for item in seed_summaries),
        "controlled_source_selection_regression_count": sum(
            _int(item.get("controlled_source_selection_regression_count")) for item in seed_summaries
        ),
        "post_training_holdout_status": (
            "passed" if all(item.get("post_training_holdout_status") == "passed" for item in seed_summaries) else "failed"
        ),
        "post_training_canary_status": (
            "passed" if all(item.get("post_training_canary_status") == "passed" for item in seed_summaries) else "failed"
        ),
    }


def _input_counters(steps: list[dict[str, Any]], trainable_steps: list[dict[str, Any]]) -> Counter[str]:
    counters: Counter[str] = Counter()
    counters["step_count"] = len(steps)
    counters["input_trainable_transition_count"] = len(trainable_steps)
    counters["unique_trainable_context_count"] = len({step.get("context_id") for step in trainable_steps})
    for step in steps:
        if step.get("ppo_trainable") is True and step.get("split") == "validation":
            counters["validation_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("split") == "test":
            counters["test_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("diagnostic_only") is True:
            counters["diagnostic_trainable_count"] += 1
        if step.get("ppo_trainable") is True and str(step.get("controlled_choice_source")) in {
            "source_fallback",
            "teacher_fallback",
        }:
            counters["fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "source_fallback":
            counters["source_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "teacher_fallback":
            counters["teacher_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and _string_list(step.get("gate_reason_codes")):
            counters["non_empty_gate_reason_trainable_count"] += 1
        if step.get("ppo_trainable") is True and (step.get("observation") is None or step.get("missing_observation") is True):
            counters["missing_observation_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("log_prob")):
            counters["missing_log_prob_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("value")):
            counters["missing_value_count"] += 1
        if step.get("ppo_trainable") is True and step.get("invalid_action_mask") is True:
            counters["invalid_action_mask_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("reward")):
            counters["non_finite_reward_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("discounted_return")):
            counters["non_finite_return_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("advantage")):
            counters["non_finite_advantage_count"] += 1
        reasons = set(_string_list(step.get("controlled_regression_reason_codes")))
        if reasons:
            counters["controlled_regression_count"] += 1
        if "safety_regression" in reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_violation" in reasons or "contract_regression" in reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in reasons or "risk_regression" in reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in reasons:
            counters["controlled_source_selection_regression_count"] += 1
    return counters


def _training_curves(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for summary in seed_summaries:
        records.extend(summary.get("training_curve_records") or [])
    return {
        "schema_version": "guarded-formal-ppo-training-run-training-curves/v1",
        "record_count": len(records),
        "records": records,
    }


def _gate_audit(counters: Counter[str], seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _seed_metrics(seed_summaries)
    return {
        "schema_version": "guarded-formal-ppo-training-run-gate-audit/v1",
        "input_counters": dict(counters),
        "seed_count": metrics["seed_count"],
        "passed_seed_count": metrics["passed_seed_count"],
        "controlled_regression_count": metrics["controlled_regression_count"],
        "post_training_holdout_status": metrics["post_training_holdout_status"],
        "post_training_canary_status": metrics["post_training_canary_status"],
    }


def _rollback_manifest(
    *,
    output_root: Path,
    authorization_summary_path: Path,
    seed_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "guarded-formal-ppo-training-run-rollback-manifest/v1",
        "authorization_summary": str(authorization_summary_path),
        "output_root": str(output_root),
        "seed_candidate_roots": [
            str(item.get("updated_candidate_root") or item.get("limited_ppo_update_smoke_root"))
            for item in seed_summaries
            if item.get("updated_candidate_root") or item.get("limited_ppo_update_smoke_root")
        ],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _normalize_seed_summary(seed_summary: dict[str, Any], *, seed: int) -> dict[str, Any]:
    normalized = dict(seed_summary)
    normalized.setdefault("schema_version", SEED_SCHEMA_VERSION)
    normalized.setdefault("seed", int(seed))
    normalized.setdefault("reason_codes", [])
    normalized.setdefault("runs_guarded_formal_ppo_training_run", True)
    normalized.setdefault("runs_new_ppo_update", True)
    normalized.setdefault("post_training_holdout_status", "passed" if normalized.get("status") == "passed" else "failed")
    normalized.setdefault("post_training_canary_status", "passed" if normalized.get("status") == "passed" else "failed")
    normalized.setdefault("experimental_checkpoint", True)
    normalized.setdefault("publishes_checkpoint", False)
    normalized.setdefault("replaces_default_policy", False)
    normalized.setdefault("performance_claimed", False)
    normalized.setdefault("formal_training_ready_claimed", False)
    normalized["training_curve_records"] = _seed_training_curve_records(normalized)
    return normalized


def _seed_training_curve_records(seed_summary: dict[str, Any]) -> list[dict[str, Any]]:
    existing = seed_summary.get("training_curve_records")
    if isinstance(existing, list) and existing:
        return existing
    return [
        {
            "seed": _int(seed_summary.get("seed")),
            "epoch": 1,
            "optimizer_train_transition_count": _int(seed_summary.get("optimizer_train_transition_count")),
            "approx_kl": _float(seed_summary.get("approx_kl")),
            "max_grad_norm_after_clip": _float(seed_summary.get("max_grad_norm_after_clip")),
        }
    ]


def _progress_row(
    *,
    seed: int,
    event: str,
    current: int,
    total: int,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "guarded-formal-ppo-training-run-progress/v1",
        "timestamp": _utc_now(),
        "seed": int(seed),
        "event": event,
        "current": int(current),
        "total": int(total),
        "metrics": metrics or {},
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Guarded Formal PPO Training Run\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Trainable transitions: `{summary.get('optimizer_train_transition_count')}`\n"
        f"- Seeds passed: `{summary.get('passed_seed_count')}` / `{summary.get('seed_count')}`\n"
        f"- Teacher agreement: `{summary.get('teacher_agreement_rate')}`\n"
        f"- Controlled regression count: `{summary.get('controlled_regression_count')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This is the authorized guarded formal PPO training run over frozen "
        "train-split evidence. It produces experimental candidates only; it does "
        "not launch online canary traffic, connect a real executor, publish a "
        "checkpoint, replace the default policy, or claim release readiness.\n"
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "seed_summaries": SEED_SUMMARIES_FILE,
        "progress": PROGRESS_FILE,
        "training_curves": TRAINING_CURVES_FILE,
        "gate_audit": GATE_AUDIT_FILE,
        "rollback_manifest": ROLLBACK_MANIFEST_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _resolve_steps_path(summary: dict[str, Any], base: Path, repo_root: Path) -> Path:
    configured = summary.get("steps") or "formal-ppo-training-run-steps.jsonl"
    path = Path(str(configured))
    if path.is_absolute():
        return path
    candidate = base.parent / path if base.is_file() else base / path
    return candidate if candidate.is_file() else repo_root / path


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = base / path
    return candidate if candidate.exists() else repo_root / path


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _expected_trainable(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("expected_trainable_transition_count"), 684)


def _seeds(config: dict[str, Any]) -> list[int]:
    return [_int(seed) for seed in config.get("seeds", [0, 1, 2, 3, 4])]


def _seeds_from_summaries(seed_summaries: list[dict[str, Any]]) -> list[int]:
    return [_int(summary.get("seed")) for summary in seed_summaries if "seed" in summary]


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [_int(item) for item in value]


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
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _git_current_matches(summary: dict[str, Any]) -> bool | None:
    provenance = summary.get("git_provenance")
    if not isinstance(provenance, dict):
        return None
    if provenance.get("current_matches_sources") is False:
        return False
    current = provenance.get("current")
    if isinstance(current, dict) and current.get("dirty") is True:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


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
    return parsed if math.isfinite(parsed) else default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
