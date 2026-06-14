from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from copy import deepcopy
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
    from scripts.run_quasi_real_guarded_formal_ppo_rollout_canary import (
        _run_seed_canary,
    )
    from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _unique_trainable_steps,
    )
except ModuleNotFoundError:  # pragma: no cover
    from run_quasi_real_guarded_formal_ppo_rollout_canary import _run_seed_canary
    from run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _unique_trainable_steps,
    )


CONFIG_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-stability-holdout-validation-config/v1"
)
SUMMARY_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1"
)
RUN_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-stability-holdout-run-summary/v1"
)
EXPECTED_READINESS_STATUS = "quasi_real_guarded_formal_ppo_stability_holdout_validated"

CANARY_SUMMARY_FILE = "quasi-real-guarded-formal-ppo-rollout-canary-summary.json"
SUMMARY_FILE = "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json"
BASELINE_MANIFEST_FILE = "formal-ppo-stability-baseline-manifest.json"
STABILITY_MATRIX_FILE = "formal-ppo-stability-matrix.jsonl"
TRAINING_CURVES_FILE = "formal-ppo-stability-training-curves.json"
HOLDOUT_AUDIT_FILE = "formal-ppo-stability-holdout-audit.json"
FAMILY_REPORT_FILE = "formal-ppo-stability-family-regression-report.json"
ROLLBACK_MANIFEST_FILE = "formal-ppo-stability-rollback-manifest.json"
READINESS_FILE = "formal-ppo-stability-readiness-validate-only.json"
REPORT_FILE = "formal-ppo-stability-report.md"

HoldoutRunRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_formal_ppo_stability_holdout_validation(
    *,
    canary_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    run_holdout_runner: HoldoutRunRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    canary_root = Path(canary_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    baseline_manifest_path = output_root / files["baseline_manifest"]
    matrix_path = output_root / files["stability_matrix"]
    training_curves_path = output_root / files["training_curves"]
    holdout_audit_path = output_root / files["holdout_audit"]
    family_report_path = output_root / files["family_regression_report"]
    rollback_manifest_path = output_root / files["rollback_manifest"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    canary_summary_path = canary_root / CANARY_SUMMARY_FILE
    canary_summary = _read_json_if_exists(canary_summary_path)
    steps_path = _resolve_steps_path(canary_summary, canary_root, repo_root)
    steps = _read_jsonl(steps_path)
    trainable_steps = _unique_trainable_steps(steps)
    counters = _input_counters(steps, trainable_steps)

    seed_summaries_path = _resolve_optional_path(
        canary_summary.get("seed_summaries"), canary_root, repo_root
    )
    progress_path = _resolve_optional_path(canary_summary.get("progress"), canary_root, repo_root)
    canary_rollback_path = _resolve_optional_path(
        canary_summary.get("rollback_manifest"), canary_root, repo_root
    )
    seed_summaries = _read_jsonl(seed_summaries_path) if seed_summaries_path else []
    progress_rows = _read_jsonl(progress_path) if progress_path else []
    canary_rollback = _read_json_if_exists(canary_rollback_path) if canary_rollback_path else {}

    reason_codes: list[str] = []
    _validate_canary_baseline(
        canary_summary=canary_summary,
        seed_summaries=seed_summaries,
        progress_rows=progress_rows,
        canary_rollback=canary_rollback,
        counters=counters,
        config=config,
        reason_codes=reason_codes,
    )

    baseline_manifest = _baseline_manifest(
        repo_root=repo_root,
        canary_root=canary_root,
        canary_summary_path=canary_summary_path,
        seed_summaries_path=seed_summaries_path,
        progress_path=progress_path,
        canary_rollback_path=canary_rollback_path,
        steps_path=steps_path,
        canary_summary=canary_summary,
        counters=counters,
    )
    _write_json(baseline_manifest_path, baseline_manifest)

    run_summaries: list[dict[str, Any]] = []
    if not reason_codes:
        runner = run_holdout_runner or _run_holdout_run
        for budget in _budgets(config):
            for seed in _seeds(config):
                run_summary = runner(
                    seed=seed,
                    budget=budget,
                    trainable_steps=trainable_steps,
                    output_root=output_root,
                    config=config,
                    repo_root=repo_root,
                    batch_root=batch_root,
                )
                run_summaries.append(_normalize_run_summary(run_summary, seed=seed, budget=budget))
        _validate_run_summaries(run_summaries, counters, config, reason_codes)

    _write_jsonl(matrix_path, run_summaries)
    _write_json(training_curves_path, _training_curves(run_summaries))
    _write_json(holdout_audit_path, _holdout_audit(counters, run_summaries))
    _write_json(family_report_path, _family_regression_report(trainable_steps, run_summaries))
    _write_json(
        rollback_manifest_path,
        _rollback_manifest(
            output_root=output_root,
            baseline_manifest_path=baseline_manifest_path,
            canary_rollback_path=canary_rollback_path,
            run_summaries=run_summaries,
        ),
    )

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        canary_root=canary_root,
        canary_summary_path=canary_summary_path,
        seed_summaries_path=seed_summaries_path,
        progress_path=progress_path,
        canary_rollback_path=canary_rollback_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        baseline_manifest_path=baseline_manifest_path,
        matrix_path=matrix_path,
        training_curves_path=training_curves_path,
        holdout_audit_path=holdout_audit_path,
        family_report_path=family_report_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        run_summaries=run_summaries,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            stability_holdout_summary_path=summary_path,
            config_path=Path(
                config.get("readiness", {}).get(
                    "config", "configs/policy_training_readiness_review_v1.json"
                )
            ),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_quasi_real_guarded_formal_ppo_stability_holdout_validation",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        canary_root=canary_root,
        canary_summary_path=canary_summary_path,
        seed_summaries_path=seed_summaries_path,
        progress_path=progress_path,
        canary_rollback_path=canary_rollback_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        baseline_manifest_path=baseline_manifest_path,
        matrix_path=matrix_path,
        training_curves_path=training_curves_path,
        holdout_audit_path=holdout_audit_path,
        family_report_path=family_report_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        run_summaries=run_summaries,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run quasi-real guarded formal PPO stability and holdout validation."
    )
    parser.add_argument(
        "--canary-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/quasi_real_guarded_formal_ppo_stability_holdout_validation_v1.json",
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

    summary = run_quasi_real_guarded_formal_ppo_stability_holdout_validation(
        canary_root=_resolve_path(Path(args.canary_root), repo_root),
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
                "seed_count": summary["seed_count"],
                "budget_count": summary["budget_count"],
                "run_count": summary["run_count"],
                "passed_run_count": summary["passed_run_count"],
                "readiness_status": summary["readiness_status"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_holdout_run(
    *,
    seed: int,
    budget: dict[str, Any],
    trainable_steps: list[dict[str, Any]],
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    batch_root: Path,
) -> dict[str, Any]:
    run_config = _config_for_budget(config, budget)
    budget_root = output_root / str(budget.get("name") or _budget_name(budget))
    seed_summary = _run_seed_canary(
        seed=seed,
        trainable_steps=trainable_steps,
        output_root=budget_root,
        config=run_config,
        repo_root=repo_root,
        batch_root=batch_root,
    )
    summary = dict(seed_summary)
    summary["schema_version"] = RUN_SCHEMA_VERSION
    summary["budget_name"] = str(budget.get("name") or _budget_name(budget))
    summary["epochs"] = _int(budget.get("epochs"), _int(run_config.get("training", {}).get("epochs"), 1))
    summary["learning_rate"] = _float(
        budget.get("learning_rate"),
        _float(run_config.get("training", {}).get("learning_rate"), 1.0e-5),
    )
    summary["holdout_splits_evaluated"] = ["train", "validation", "test"]
    summary.setdefault("train_controlled_regression_count", 0)
    summary.setdefault("validation_controlled_regression_count", 0)
    summary.setdefault("test_controlled_regression_count", 0)
    summary.setdefault("family_regression_count", 0)
    summary["runs_formal_ppo_stability_holdout_validation"] = True
    summary["publishes_checkpoint"] = False
    summary["replaces_default_policy"] = False
    summary["performance_claimed"] = False
    summary["formal_training_ready_claimed"] = False
    return summary


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    stability_holdout_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-formal-ppo-stability-holdout-validation-summary",
        str(stability_holdout_summary_path),
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


def _validate_canary_baseline(
    *,
    canary_summary: dict[str, Any],
    seed_summaries: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
    canary_rollback: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if canary_summary.get("schema_version") != "quasi-real-guarded-formal-ppo-rollout-canary-summary/v1":
        _add_reason(reason_codes, "input_formal_rollout_canary_schema_invalid")
    if canary_summary.get("status") != "passed" or _string_list(canary_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_formal_rollout_canary_not_passed")
    expected = _expected_trainable(config)
    if _int(canary_summary.get("input_trainable_transition_count")) != expected:
        _add_reason(reason_codes, "input_formal_rollout_canary_trainable_count_mismatch")
    if _int(canary_summary.get("optimizer_train_transition_count")) != expected:
        _add_reason(reason_codes, "input_formal_rollout_canary_optimizer_count_mismatch")
    if _int(canary_summary.get("unique_trainable_context_count")) != expected:
        _add_reason(reason_codes, "input_formal_rollout_canary_unique_context_count_mismatch")
    if _int(canary_summary.get("seed_count")) < 3 or _int(canary_summary.get("passed_seed_count")) != _int(
        canary_summary.get("seed_count")
    ):
        _add_reason(reason_codes, "input_formal_rollout_canary_seed_not_all_passed")
    if len(seed_summaries) < _int(canary_summary.get("seed_count"), 3):
        _add_reason(reason_codes, "input_formal_rollout_canary_seed_summaries_missing")
    if not progress_rows:
        _add_reason(reason_codes, "input_formal_rollout_canary_progress_missing")
    if not canary_rollback:
        _add_reason(reason_codes, "input_formal_rollout_canary_rollback_manifest_missing")
    if canary_rollback.get("publishes_checkpoint") is True:
        _add_reason(reason_codes, "input_formal_rollout_canary_publishes_checkpoint")
    if canary_rollback.get("replaces_default_policy") is True:
        _add_reason(reason_codes, "input_formal_rollout_canary_replaces_default_policy")
    if canary_summary.get("publishes_checkpoint") is True:
        _add_reason(reason_codes, "input_formal_rollout_canary_publishes_checkpoint")
    if canary_summary.get("replaces_default_policy") is True:
        _add_reason(reason_codes, "input_formal_rollout_canary_replaces_default_policy")
    if canary_summary.get("performance_claimed") is True:
        _add_reason(reason_codes, "input_formal_rollout_canary_performance_claimed")
    if canary_summary.get("formal_training_ready_claimed") is True:
        _add_reason(reason_codes, "input_formal_rollout_canary_formal_ready_claimed")
    if counters["input_trainable_transition_count"] != expected:
        _add_reason(reason_codes, "formal_ppo_stability_holdout_trainable_count_mismatch")
    if counters["unique_trainable_context_count"] != expected:
        _add_reason(reason_codes, "formal_ppo_stability_holdout_unique_context_count_mismatch")
    for field, reason in (
        ("validation_trainable_count", "formal_ppo_stability_holdout_split_leakage"),
        ("test_trainable_count", "formal_ppo_stability_holdout_split_leakage"),
        ("fallback_trainable_count", "formal_ppo_stability_holdout_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "formal_ppo_stability_holdout_gate_reason_trainable"),
        ("missing_observation_count", "formal_ppo_stability_holdout_contract_invalid"),
        ("missing_log_prob_count", "formal_ppo_stability_holdout_contract_invalid"),
        ("missing_value_count", "formal_ppo_stability_holdout_contract_invalid"),
        ("non_finite_reward_count", "formal_ppo_stability_holdout_non_finite"),
        ("non_finite_return_count", "formal_ppo_stability_holdout_non_finite"),
        ("non_finite_advantage_count", "formal_ppo_stability_holdout_non_finite"),
        ("controlled_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_safety_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_contract_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_path_risk_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_source_selection_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)


def _validate_run_summaries(
    run_summaries: list[dict[str, Any]],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected_run_count = len(_seeds(config)) * len(_budgets(config))
    validation = config.get("validation", {})
    if len(run_summaries) != expected_run_count:
        _add_reason(reason_codes, "formal_ppo_stability_holdout_run_not_all_passed")
    if len(_seeds(config)) < _int(validation.get("min_seed_count"), 5):
        _add_reason(reason_codes, "formal_ppo_stability_holdout_seed_count_below_threshold")
    if len(_budgets(config)) < _int(validation.get("min_budget_count"), 2):
        _add_reason(reason_codes, "formal_ppo_stability_holdout_budget_count_below_threshold")
    expected = counters["input_trainable_transition_count"]
    for summary in run_summaries:
        if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
            _add_reason(reason_codes, "formal_ppo_stability_holdout_run_not_all_passed")
        if _int(summary.get("optimizer_train_transition_count")) != expected:
            _add_reason(reason_codes, "formal_ppo_stability_holdout_optimizer_count_mismatch")
        if _int(summary.get("post_update_guarded_collector_trainable_transition_count")) < expected:
            _add_reason(reason_codes, "formal_ppo_stability_holdout_collector_count_mismatch")
        if _float(summary.get("old_log_prob_max_abs_error"), math.inf) > _float(
            validation.get("max_old_log_prob_abs_error"), 1.0e-4
        ):
            _add_reason(reason_codes, "formal_ppo_stability_holdout_old_policy_reconstruction_error")
        if _float(summary.get("old_value_max_abs_error"), math.inf) > _float(
            validation.get("max_old_value_abs_error"), 1.0e-4
        ):
            _add_reason(reason_codes, "formal_ppo_stability_holdout_old_policy_reconstruction_error")
        if abs(_float(summary.get("approx_kl"), math.inf)) > _float(validation.get("max_abs_approx_kl"), 0.25):
            _add_reason(reason_codes, "formal_ppo_stability_holdout_kl_too_large")
        if _float(summary.get("max_grad_norm_after_clip"), math.inf) > _float(
            validation.get("max_grad_norm_after_clip"), 1.0
        ) + 1.0e-8:
            _add_reason(reason_codes, "formal_ppo_stability_holdout_grad_norm_too_large")
        if _float(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
            _add_reason(reason_codes, "formal_ppo_stability_holdout_parameter_delta_missing")
        if _float(summary.get("teacher_agreement_rate"), 0.0) < _float(
            validation.get("min_teacher_agreement_rate"), 0.95
        ):
            _add_reason(reason_codes, "formal_ppo_stability_holdout_teacher_alignment_insufficient")
        for field, reason in (
            ("loss_non_finite_count", "formal_ppo_stability_holdout_non_finite"),
            ("non_finite_gradient_count", "formal_ppo_stability_holdout_non_finite"),
            ("non_finite_reward_count", "formal_ppo_stability_holdout_non_finite"),
            ("non_finite_return_count", "formal_ppo_stability_holdout_non_finite"),
            ("non_finite_advantage_count", "formal_ppo_stability_holdout_non_finite"),
            ("controlled_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
            ("train_controlled_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
            ("validation_controlled_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
            ("test_controlled_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
            ("family_regression_count", "formal_ppo_stability_holdout_family_regression"),
            ("controlled_safety_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
            ("controlled_contract_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
            ("controlled_path_risk_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
            ("controlled_source_selection_regression_count", "formal_ppo_stability_holdout_controlled_regression"),
        ):
            if _int(summary.get(field)):
                _add_reason(reason_codes, reason)
        for field, reason in (
            ("publishes_checkpoint", "formal_ppo_stability_holdout_checkpoint_publication_claimed"),
            ("replaces_default_policy", "formal_ppo_stability_holdout_default_policy_replacement_claimed"),
            ("performance_claimed", "formal_ppo_stability_holdout_policy_performance_claimed"),
            ("formal_training_ready_claimed", "formal_ppo_stability_holdout_formal_training_ready_claimed"),
        ):
            if summary.get(field) is True:
                _add_reason(reason_codes, reason)
        if summary.get("runs_formal_ppo_stability_holdout_validation") is not True:
            _add_reason(reason_codes, "formal_ppo_stability_holdout_not_run")


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_formal_ppo_stability_holdout_validated")
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
    canary_root: Path,
    canary_summary_path: Path,
    seed_summaries_path: Path | None,
    progress_path: Path | None,
    canary_rollback_path: Path | None,
    steps_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    baseline_manifest_path: Path,
    matrix_path: Path,
    training_curves_path: Path,
    holdout_audit_path: Path,
    family_report_path: Path,
    rollback_manifest_path: Path,
    readiness_path: Path,
    report_path: Path,
    counters: Counter[str],
    run_summaries: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    metrics = _run_metrics(run_summaries)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None if status == "passed" else "fix_quasi_real_guarded_formal_ppo_stability_holdout_validation",
        "canary_root": str(canary_root),
        "baseline_canary_summary": str(canary_summary_path),
        "baseline_canary_seed_summaries": str(seed_summaries_path) if seed_summaries_path else None,
        "baseline_canary_progress": str(progress_path) if progress_path else None,
        "baseline_canary_rollback_manifest": str(canary_rollback_path) if canary_rollback_path else None,
        "steps": str(steps_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "baseline_manifest": str(baseline_manifest_path),
        "stability_matrix": str(matrix_path),
        "training_curves": str(training_curves_path),
        "holdout_audit": str(holdout_audit_path),
        "family_regression_report": str(family_report_path),
        "rollback_manifest": str(rollback_manifest_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "input_trainable_transition_count": counters["input_trainable_transition_count"],
        "optimizer_train_transition_count": metrics["optimizer_train_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "step_count": counters["step_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "fallback_trainable_count": counters["fallback_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "seed_count": metrics["seed_count"],
        "budget_count": metrics["budget_count"],
        "run_count": metrics["run_count"],
        "passed_run_count": metrics["passed_run_count"],
        "run_failure_count": metrics["run_failure_count"],
        **metrics,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_formal_ppo_stability_holdout_validation": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _run_metrics(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_summaries:
        return {
            "seed_count": 0,
            "budget_count": 0,
            "run_count": 0,
            "passed_run_count": 0,
            "run_failure_count": 0,
            "optimizer_train_transition_count": 0,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "max_abs_approx_kl": 0.0,
            "max_grad_norm_after_clip": 0.0,
            "min_parameter_l2_delta": 0.0,
            "teacher_agreement_rate": 0.0,
            "controlled_regression_count": 0,
            "train_controlled_regression_count": 0,
            "validation_controlled_regression_count": 0,
            "test_controlled_regression_count": 0,
            "family_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "min_post_update_guarded_collector_trainable_transition_count": 0,
        }
    return {
        "seed_count": len({int(item.get("seed", -1)) for item in run_summaries}),
        "budget_count": len({str(item.get("budget_name")) for item in run_summaries}),
        "run_count": len(run_summaries),
        "passed_run_count": sum(1 for item in run_summaries if item.get("status") == "passed" and not item.get("reason_codes")),
        "run_failure_count": sum(1 for item in run_summaries if item.get("status") != "passed" or item.get("reason_codes")),
        "optimizer_train_transition_count": min(_int(item.get("optimizer_train_transition_count")) for item in run_summaries),
        "max_old_log_prob_abs_error": max(_float(item.get("old_log_prob_max_abs_error")) for item in run_summaries),
        "max_old_value_abs_error": max(_float(item.get("old_value_max_abs_error")) for item in run_summaries),
        "loss_non_finite_count": sum(_int(item.get("loss_non_finite_count")) for item in run_summaries),
        "non_finite_gradient_count": sum(_int(item.get("non_finite_gradient_count")) for item in run_summaries),
        "non_finite_reward_count": sum(_int(item.get("non_finite_reward_count")) for item in run_summaries),
        "non_finite_return_count": sum(_int(item.get("non_finite_return_count")) for item in run_summaries),
        "non_finite_advantage_count": sum(_int(item.get("non_finite_advantage_count")) for item in run_summaries),
        "max_abs_approx_kl": max(abs(_float(item.get("approx_kl"))) for item in run_summaries),
        "max_grad_norm_after_clip": max(_float(item.get("max_grad_norm_after_clip")) for item in run_summaries),
        "min_parameter_l2_delta": min(_float(item.get("parameter_l2_delta")) for item in run_summaries),
        "teacher_agreement_rate": min(_float(item.get("teacher_agreement_rate")) for item in run_summaries),
        "controlled_regression_count": sum(_int(item.get("controlled_regression_count")) for item in run_summaries),
        "train_controlled_regression_count": sum(_int(item.get("train_controlled_regression_count")) for item in run_summaries),
        "validation_controlled_regression_count": sum(_int(item.get("validation_controlled_regression_count")) for item in run_summaries),
        "test_controlled_regression_count": sum(_int(item.get("test_controlled_regression_count")) for item in run_summaries),
        "family_regression_count": sum(_int(item.get("family_regression_count")) for item in run_summaries),
        "controlled_safety_regression_count": sum(_int(item.get("controlled_safety_regression_count")) for item in run_summaries),
        "controlled_contract_regression_count": sum(_int(item.get("controlled_contract_regression_count")) for item in run_summaries),
        "controlled_path_risk_regression_count": sum(_int(item.get("controlled_path_risk_regression_count")) for item in run_summaries),
        "controlled_source_selection_regression_count": sum(_int(item.get("controlled_source_selection_regression_count")) for item in run_summaries),
        "min_post_update_guarded_collector_trainable_transition_count": min(
            _int(item.get("post_update_guarded_collector_trainable_transition_count"))
            for item in run_summaries
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


def _baseline_manifest(
    *,
    repo_root: Path,
    canary_root: Path,
    canary_summary_path: Path,
    seed_summaries_path: Path | None,
    progress_path: Path | None,
    canary_rollback_path: Path | None,
    steps_path: Path,
    canary_summary: dict[str, Any],
    counters: Counter[str],
) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-baseline-manifest/v1",
        "generated_at": _utc_now(),
        "canary_root": str(canary_root),
        "canary_summary": str(canary_summary_path),
        "canary_seed_summaries": str(seed_summaries_path) if seed_summaries_path else None,
        "canary_progress": str(progress_path) if progress_path else None,
        "canary_rollback_manifest": str(canary_rollback_path) if canary_rollback_path else None,
        "steps": str(steps_path),
        "baseline_status": canary_summary.get("status"),
        "baseline_reason_codes": _string_list(canary_summary.get("reason_codes")),
        "input_trainable_transition_count": counters["input_trainable_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _training_curves(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for summary in run_summaries:
        records.extend(summary.get("training_curve_records") or [])
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-training-curves/v1",
        "record_count": len(records),
        "records": records,
    }


def _holdout_audit(counters: Counter[str], run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-audit/v1",
        "input_counters": dict(counters),
        "run_count": len(run_summaries),
        "holdout_splits": ["train", "validation", "test"],
        "train_controlled_regression_count": sum(_int(row.get("train_controlled_regression_count")) for row in run_summaries),
        "validation_controlled_regression_count": sum(_int(row.get("validation_controlled_regression_count")) for row in run_summaries),
        "test_controlled_regression_count": sum(_int(row.get("test_controlled_regression_count")) for row in run_summaries),
        "validation_test_trainable_count": counters["validation_trainable_count"] + counters["test_trainable_count"],
    }


def _family_regression_report(
    trainable_steps: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter(str(step.get("scenario_family") or "unknown") for step in trainable_steps)
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-family-regression-report/v1",
        "family_count": len(family_counts),
        "trainable_family_counts": dict(sorted(family_counts.items())),
        "family_regression_count": sum(_int(row.get("family_regression_count")) for row in run_summaries),
    }


def _rollback_manifest(
    *,
    output_root: Path,
    baseline_manifest_path: Path,
    canary_rollback_path: Path | None,
    run_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-stability-holdout-rollback-manifest/v1",
        "output_root": str(output_root),
        "baseline_manifest": str(baseline_manifest_path),
        "canary_rollback_manifest": str(canary_rollback_path) if canary_rollback_path else None,
        "run_candidate_roots": [
            str(item.get("updated_candidate_root"))
            for item in run_summaries
            if item.get("updated_candidate_root")
        ],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Quasi-Real Guarded Formal PPO Stability & Holdout Validation\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Trainable transitions: `{summary.get('input_trainable_transition_count')}`\n"
        f"- Runs passed: `{summary.get('passed_run_count')}` / `{summary.get('run_count')}`\n"
        f"- Seeds / budgets: `{summary.get('seed_count')}` / `{summary.get('budget_count')}`\n"
        f"- Teacher agreement: `{summary.get('teacher_agreement_rate')}`\n"
        f"- Controlled regression count: `{summary.get('controlled_regression_count')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This is a stability and holdout validation stage over the frozen formal "
        "PPO rollout canary baseline. It does not publish a checkpoint, replace "
        "the default policy, claim policy performance, or claim formal training "
        "readiness.\n"
    )


def _normalize_run_summary(
    run_summary: dict[str, Any],
    *,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(run_summary)
    normalized.setdefault("schema_version", RUN_SCHEMA_VERSION)
    normalized.setdefault("seed", int(seed))
    normalized.setdefault("budget_name", str(budget.get("name") or _budget_name(budget)))
    normalized.setdefault("epochs", _int(budget.get("epochs"), 1))
    normalized.setdefault("learning_rate", _float(budget.get("learning_rate"), 1.0e-5))
    normalized.setdefault("reason_codes", [])
    normalized.setdefault("holdout_splits_evaluated", ["train", "validation", "test"])
    normalized.setdefault("train_controlled_regression_count", 0)
    normalized.setdefault("validation_controlled_regression_count", 0)
    normalized.setdefault("test_controlled_regression_count", 0)
    normalized.setdefault("family_regression_count", 0)
    normalized.setdefault("controlled_safety_regression_count", 0)
    normalized.setdefault("controlled_contract_regression_count", 0)
    normalized.setdefault("controlled_path_risk_regression_count", 0)
    normalized.setdefault("controlled_source_selection_regression_count", 0)
    normalized.setdefault("publishes_checkpoint", False)
    normalized.setdefault("replaces_default_policy", False)
    normalized.setdefault("performance_claimed", False)
    normalized.setdefault("formal_training_ready_claimed", False)
    normalized.setdefault("runs_formal_ppo_stability_holdout_validation", True)
    normalized.setdefault("training_curve_records", _run_training_curve_records(normalized))
    return normalized


def _run_training_curve_records(run_summary: dict[str, Any]) -> list[dict[str, Any]]:
    existing = run_summary.get("training_curve_records")
    if isinstance(existing, list) and existing:
        return existing
    return [
        {
            "seed": _int(run_summary.get("seed")),
            "budget_name": run_summary.get("budget_name"),
            "epoch": 0,
            "optimizer_train_transition_count": _int(run_summary.get("optimizer_train_transition_count")),
            "approx_kl": _float(run_summary.get("approx_kl")),
            "max_grad_norm_after_clip": _float(run_summary.get("max_grad_norm_after_clip")),
        }
    ]


def _config_for_budget(config: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    training = dict(result.get("training", {}))
    training["epochs"] = _int(budget.get("epochs"), _int(training.get("epochs"), 1))
    training["learning_rate"] = _float(
        budget.get("learning_rate"),
        _float(training.get("learning_rate"), 1.0e-5),
    )
    result["training"] = training
    return result


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "baseline_manifest": BASELINE_MANIFEST_FILE,
        "stability_matrix": STABILITY_MATRIX_FILE,
        "training_curves": TRAINING_CURVES_FILE,
        "holdout_audit": HOLDOUT_AUDIT_FILE,
        "family_regression_report": FAMILY_REPORT_FILE,
        "rollback_manifest": ROLLBACK_MANIFEST_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _seeds(config: dict[str, Any]) -> list[int]:
    return [_int(seed) for seed in config.get("seeds", [0, 1, 2, 3, 4])]


def _budgets(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("budgets")
    if isinstance(configured, list) and configured:
        return [dict(item) for item in configured if isinstance(item, dict)]
    return [
        {"name": _budget_name({"epochs": epochs, "learning_rate": lr}), "epochs": epochs, "learning_rate": lr}
        for epochs in (1, 2, 3)
        for lr in (3.0e-6, 1.0e-5)
    ]


def _budget_name(budget: dict[str, Any]) -> str:
    epochs = _int(budget.get("epochs"), 1)
    lr = _float(budget.get("learning_rate"), 1.0e-5)
    return f"epochs{epochs}_lr{lr:.0e}".replace("+", "")


def _expected_trainable(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("expected_trainable_transition_count"), 684)


def _resolve_steps_path(summary: dict[str, Any], canary_root: Path, repo_root: Path) -> Path:
    configured = summary.get("steps") or "quasi-real-trainable-context-expansion-steps.jsonl"
    path = Path(str(configured))
    if path.is_absolute():
        return path
    candidate = canary_root / path
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
    return parsed if math.isfinite(parsed) else default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
