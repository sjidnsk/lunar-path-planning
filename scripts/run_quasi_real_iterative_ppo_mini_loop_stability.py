from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot
from training_progress import (
    add_progress_argument,
    collector_progress_metrics,
    make_progress_reporter,
    ppo_update_progress_metrics,
    progress_child_env,
)


CONFIG_SCHEMA_VERSION = "quasi-real-iterative-ppo-mini-loop-stability-config/v1"
SUMMARY_SCHEMA_VERSION = "iterative-ppo-mini-loop-stability-summary/v1"

NEXT_ON_POLICY_INVALID = "iterative_ppo_on_policy_contract_invalid"
NEXT_TRAINABLE_INSUFFICIENT = "iterative_ppo_trainable_transition_count_insufficient"
NEXT_NON_FINITE = "iterative_ppo_update_loss_non_finite"
NEXT_POLICY_DRIFT = "iterative_ppo_policy_drift_detected"
NEXT_POST_GATE_REGRESSION = "iterative_ppo_post_update_gate_regression"

EXPECTED_WRAPPER_GATE_REASON = "limited_quasi_real_ppo_update_post_update_gate_regression"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run quasi-real iterative PPO mini-loop stability evaluation."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--initial-candidate-root", required=True)
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    add_progress_argument(parser)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(
            json.dumps(
                {"status": "config validated", "config": _display_path(config_path, repo_root)},
                ensure_ascii=False,
            )
        )
        return 0

    summary = run_quasi_real_iterative_ppo_mini_loop_stability(
        source_root=_resolve_path(args.source_root, repo_root),
        initial_candidate_root=_resolve_path(args.initial_candidate_root, repo_root),
        quasi_real_root=_resolve_path(args.quasi_real_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
        progress_mode=args.progress,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "round_count": summary["round_count"],
                "failed_round_count": summary["failed_round_count"],
                "stability_passed": summary["stability_passed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def build_quasi_real_round_plan(
    *,
    output_root: Path,
    initial_candidate_root: Path,
    round_count: int,
) -> list[dict[str, Path | int]]:
    plan: list[dict[str, Path | int]] = []
    base_candidate_root = initial_candidate_root
    for round_index in range(round_count):
        round_root = output_root / f"round-{round_index:02d}"
        record = {
            "round_index": round_index,
            "round_root": round_root,
            "base_candidate_root": base_candidate_root,
            "teacher_following_root": round_root / "teacher-following",
            "collector_root": round_root / "collector",
            "update_root": round_root / "update",
            "compatibility_root": round_root / "compatibility",
            "accounting_root": round_root / "accounting",
            "long_horizon_root": round_root / "long-horizon",
        }
        plan.append(record)
        base_candidate_root = record["update_root"]  # type: ignore[assignment]
    return plan


def run_quasi_real_iterative_ppo_mini_loop_stability(
    *,
    source_root: Path,
    initial_candidate_root: Path,
    quasi_real_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    progress_mode: str | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    round_count = _int_value(config["validation"].get("round_count"), 3)
    plan = build_quasi_real_round_plan(
        output_root=output_root,
        initial_candidate_root=initial_candidate_root,
        round_count=round_count,
    )
    python_bin = sys.executable
    env = subprocess_env(python_bin=python_bin)
    progress = make_progress_reporter(output_root=output_root, mode=progress_mode, config=config)
    env = progress_child_env(
        env,
        output_root=progress.output_root,
        mode=progress.mode,
        run_id=progress.run_id,
    )
    round_records: list[dict[str, Any]] = []
    stages_per_round = 7

    for step in plan:
        round_index = int(step["round_index"])
        progress.emit(
            stage="iterative_round",
            status="start",
            current=round_index + 1,
            total=round_count,
            round_index=round_index,
            message=f"iterative round {round_index + 1}/{round_count}",
            output_root=step["round_root"],
        )
        try:
            _run_progress_stage(
                progress=progress,
                stage="quasi_real_teacher_following",
                current=1,
                total=stages_per_round,
                round_index=round_index,
                summary_path=Path(step["teacher_following_root"]) / "quasi-real-guarded-teacher-following-pilot-summary.json",
                repo_root=repo_root,
                metrics_loader=_teacher_following_progress_metrics,
                command=[
                    python_bin,
                    str(repo_root / "scripts" / "run_quasi_real_guarded_teacher_following_pilot.py"),
                    "--source-root",
                    str(source_root),
                    "--candidate-root",
                    str(step["base_candidate_root"]),
                    "--quasi-real-root",
                    str(quasi_real_root),
                    "--output-root",
                    str(step["teacher_following_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["teacher_following_config"]),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
            _run_progress_stage(
                progress=progress,
                stage="quasi_real_collector",
                current=2,
                total=stages_per_round,
                round_index=round_index,
                summary_path=Path(step["collector_root"]) / "ppo-rollout-collector-summary.json",
                repo_root=repo_root,
                metrics_loader=collector_progress_metrics,
                command=[
                    python_bin,
                    str(repo_root / "scripts" / "run_quasi_real_ppo_collector_dry_run.py"),
                    "--guarded-teacher-following-root",
                    str(step["teacher_following_root"]),
                    "--candidate-root",
                    str(step["base_candidate_root"]),
                    "--quasi-real-root",
                    str(quasi_real_root),
                    "--output-root",
                    str(step["collector_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["collector_config"]),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
            update_summary_path = (
                Path(step["update_root"]) / "limited-quasi-real-ppo-update-smoke-summary.json"
            )
            update_result = _run_progress_stage(
                progress=progress,
                stage="quasi_real_ppo_update",
                current=3,
                total=stages_per_round,
                round_index=round_index,
                summary_path=update_summary_path,
                repo_root=repo_root,
                metrics_loader=ppo_update_progress_metrics,
                command=[
                    python_bin,
                    str(repo_root / "scripts" / "run_limited_quasi_real_ppo_update_smoke.py"),
                    "--source-root",
                    str(source_root),
                    "--base-candidate-root",
                    str(step["base_candidate_root"]),
                    "--collector-root",
                    str(step["collector_root"]),
                    "--output-root",
                    str(step["update_root"]),
                    "--quasi-real-root",
                    str(quasi_real_root),
                    "--guarded-teacher-following-root",
                    str(step["teacher_following_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["update_config"]),
                ],
                cwd=repo_root,
                env=env,
                check=False,
            )
            if update_result.returncode and not update_summary_path.is_file():
                raise subprocess.CalledProcessError(update_result.returncode, update_result.args)
            _run_progress_stage(
                progress=progress,
                stage="generated_sequential_compatibility",
                current=4,
                total=stages_per_round,
                round_index=round_index,
                summary_path=(
                    Path(step["compatibility_root"])
                    / "quasi-real-generated-sequential-contract-compatibility-summary.json"
                ),
                repo_root=repo_root,
                metrics_loader=_compatibility_progress_metrics,
                command=[
                    python_bin,
                    str(repo_root / "scripts" / "run_quasi_real_generated_sequential_contract_compatibility_diagnosis.py"),
                    "--update-smoke-root",
                    str(step["update_root"]),
                    "--base-candidate-root",
                    str(step["base_candidate_root"]),
                    "--source-root",
                    str(source_root),
                    "--output-root",
                    str(step["compatibility_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["compatibility_config"]),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
            _run_progress_stage(
                progress=progress,
                stage="generated_sequential_accounting",
                current=5,
                total=stages_per_round,
                round_index=round_index,
                summary_path=(
                    Path(step["accounting_root"])
                    / "generated-sequential-gate-metric-accounting-audit-summary.json"
                ),
                repo_root=repo_root,
                metrics_loader=_accounting_progress_metrics,
                command=[
                    python_bin,
                    str(repo_root / "scripts" / "run_generated_sequential_gate_metric_accounting_audit.py"),
                    "--diagnosis-root",
                    str(step["compatibility_root"]),
                    "--output-root",
                    str(step["accounting_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["accounting_config"]),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
            _run_progress_stage(
                progress=progress,
                stage="long_horizon_alignment",
                current=6,
                total=stages_per_round,
                round_index=round_index,
                summary_path=(
                    Path(step["long_horizon_root"])
                    / "long-horizon-teacher-skill-contract-summary.json"
                ),
                repo_root=repo_root,
                metrics_loader=_long_horizon_progress_metrics,
                command=[
                    python_bin,
                    str(repo_root / "scripts" / "run_generated_sequential_long_horizon_teacher_skill_contract_alignment.py"),
                    "--diagnosis-root",
                    str(step["compatibility_root"]),
                    "--accounting-audit-root",
                    str(step["accounting_root"]),
                    "--output-root",
                    str(step["long_horizon_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["long_horizon_config"]),
                    "--quasi-real-teacher-following-summary",
                    str(
                        Path(step["update_root"])
                        / "post_update_quasi_real_teacher_following"
                        / "quasi-real-guarded-teacher-following-pilot-summary.json"
                    ),
                    "--quasi-real-collector-summary",
                    str(
                        Path(step["update_root"])
                        / "post_update_quasi_real_collector"
                        / "ppo-rollout-collector-summary.json"
                    ),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
            round_records.append(_load_round_record(step, repo_root=repo_root))
            progress.emit(
                stage="iterative_round",
                status="passed",
                current=round_index + 1,
                total=round_count,
                round_index=round_index,
                message=f"iterative round {round_index + 1}/{round_count} completed",
                output_root=step["round_root"],
            )
        except subprocess.CalledProcessError as exc:
            round_records.append(_failed_round_record(step, exc, repo_root=repo_root))
            progress.emit(
                stage="iterative_round",
                status="failed",
                current=round_index + 1,
                total=round_count,
                round_index=round_index,
                message=f"iterative round {round_index + 1}/{round_count} failed",
                output_root=step["round_root"],
            )
            break

    summary, drift_report, rejection_report = summarize_quasi_real_iterative_rounds(
        round_records=round_records,
        output_root=output_root,
        config=config,
        repo_root=repo_root,
    )
    _write_outputs(
        output_root=output_root,
        config=config,
        summary=summary,
        drift_report=drift_report,
        rejection_report=rejection_report,
        round_records=round_records,
        repo_root=repo_root,
    )
    if round_records:
        _copy_final_artifacts(round_records[-1], output_root=output_root, repo_root=repo_root)
    progress.emit(
        stage="iterative_summary",
        status=summary["status"],
        current=round_count,
        total=round_count,
        message=f"iterative mini-loop {summary['status']}",
        summary_path=summary["summary"],
        reason_codes=summary["reason_codes"],
        metrics={
            "round_count": summary["round_count"],
            "failed_round_count": summary["failed_round_count"],
            "stability_passed": summary["stability_passed"],
        },
    )
    progress.finalize(status=summary["status"])
    return summary


def summarize_quasi_real_iterative_rounds(
    *,
    round_records: list[dict[str, Any]],
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validation = config["validation"]
    expected_rounds = _int_value(validation.get("round_count"), 3)
    reason_codes: list[str] = []
    rejection_records: list[dict[str, Any]] = []
    drift_records: list[dict[str, Any]] = []
    cumulative_parameter_l2_delta = 0.0
    max_abs_approx_kl = 0.0
    long_horizon_controlled_regression_count = 0

    for record in round_records:
        round_reasons = _round_reason_codes(record, validation)
        update = record.get("update_summary") or {}
        long_horizon = record.get("long_horizon_summary") or {}
        approx_kl = _float_value(update.get("approx_kl"), math.inf)
        abs_kl = abs(approx_kl) if math.isfinite(approx_kl) else math.inf
        parameter_delta = _float_value(update.get("parameter_l2_delta"), 0.0)
        if math.isfinite(abs_kl):
            max_abs_approx_kl = max(max_abs_approx_kl, abs_kl)
        else:
            max_abs_approx_kl = math.inf
        cumulative_parameter_l2_delta += max(parameter_delta, 0.0)
        long_horizon_controlled_regression_count += _int_value(
            long_horizon.get("controlled_regression_episode_count")
        )
        drift_records.append(
            {
                "round_index": _int_value(record.get("round_index")),
                "approx_kl": _finite_or_none(approx_kl),
                "abs_approx_kl": _finite_or_none(abs_kl),
                "parameter_l2_delta": parameter_delta,
                "cumulative_parameter_l2_delta": cumulative_parameter_l2_delta,
                "controlled_regression_episode_count": _int_value(
                    long_horizon.get("controlled_regression_episode_count")
                ),
                "reason_codes": round_reasons,
            }
        )
        if round_reasons:
            rejection_records.append(
                {
                    "round_index": _int_value(record.get("round_index")),
                    "reason_codes": round_reasons,
                    "base_candidate_root": record.get("base_candidate_root"),
                    "update_root": record.get("update_root"),
                }
            )
            for reason in round_reasons:
                _append_reason(reason_codes, reason)

    if len(round_records) != expected_rounds:
        _append_reason(reason_codes, NEXT_POST_GATE_REGRESSION)
    if cumulative_parameter_l2_delta <= 0.0:
        _append_reason(reason_codes, NEXT_POLICY_DRIFT)
    if cumulative_parameter_l2_delta > float(validation.get("max_cumulative_parameter_l2_delta", 0.05)):
        _append_reason(reason_codes, NEXT_POLICY_DRIFT)
    if max_abs_approx_kl > float(validation.get("max_abs_approx_kl", 0.25)):
        _append_reason(reason_codes, NEXT_POLICY_DRIFT)

    status = "failed" if reason_codes else "passed"
    final_record = round_records[-1] if round_records else {}
    summary_path = output_root / config["output_files"]["summary"]
    rounds_path = output_root / config["output_files"]["rounds"]
    drift_path = output_root / config["output_files"]["drift_report"]
    rejection_path = output_root / config["output_files"]["rejection_report"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "quasi_real_iterative_ppo_mini_loop": True,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "next_required_change": _next_required_change(reason_codes),
        "round_count": len(round_records),
        "expected_round_count": expected_rounds,
        "completed_round_count": len([record for record in round_records if not record.get("command_failed")]),
        "failed_round_count": len(rejection_records) + max(expected_rounds - len(round_records), 0),
        "stability_passed": status == "passed",
        "max_abs_approx_kl": _finite_or_inf(max_abs_approx_kl),
        "cumulative_parameter_l2_delta": cumulative_parameter_l2_delta,
        "min_optimizer_train_transition_count": min(
            [_int_value((record.get("update_summary") or {}).get("optimizer_train_transition_count")) for record in round_records]
            or [0]
        ),
        "min_ppo_trainable_transition_count": min(
            [
                _int_value((record.get("post_update_quasi_real_collector_summary") or {}).get("ppo_trainable_transition_count"))
                for record in round_records
            ]
            or [0]
        ),
        "min_teacher_agreement_rate": min(
            [
                _float_value((record.get("teacher_following_summary") or {}).get("teacher_agreement_rate"), 0.0)
                for record in round_records
            ]
            or [0.0]
        ),
        "long_horizon_controlled_regression_count": long_horizon_controlled_regression_count,
        "raw_generated_rejected_choice_count": _int_value(
            (final_record.get("update_summary") or {}).get("post_update_sequential_rejected_choice_count")
        ),
        "raw_test_regression_count": _int_value(
            (final_record.get("update_summary") or {}).get("post_update_raw_test_regression_count")
        ),
        "sequential_rejected_count": 0,
        "collector_regression_count": _collector_regression_count(
            final_record.get("post_update_quasi_real_collector_summary") or {}
        ),
        "final_candidate_root": final_record.get("update_root"),
        "final_quasi_real_teacher_following_summary_path": "final/quasi-real-teacher-following/quasi-real-guarded-teacher-following-pilot-summary.json",
        "final_quasi_real_collector_summary_path": "final/quasi-real-collector/ppo-rollout-collector-summary.json",
        "final_long_horizon_summary_path": "final/long-horizon/long-horizon-teacher-skill-contract-summary.json",
        "summary": _display_path(summary_path, repo_root),
        "rounds": _display_path(rounds_path, repo_root),
        "drift_report": _display_path(drift_path, repo_root),
        "rejection_report": _display_path(rejection_path, repo_root),
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_formal_ppo_rollout": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    drift_report = {
        "schema_version": "quasi-real-iterative-ppo-mini-loop-drift-report/v1",
        "generated_at": summary["generated_at"],
        "max_abs_approx_kl": summary["max_abs_approx_kl"],
        "cumulative_parameter_l2_delta": cumulative_parameter_l2_delta,
        "records": drift_records,
    }
    rejection_report = {
        "schema_version": "quasi-real-iterative-ppo-mini-loop-rejection-report/v1",
        "generated_at": summary["generated_at"],
        "records": rejection_records,
    }
    return summary, drift_report, rejection_report


def subprocess_env(*, python_bin: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHON"] = python_bin
    return env


def _round_reason_codes(record: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if record.get("command_failed"):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
        return reasons

    teacher = record.get("teacher_following_summary") or {}
    collector = record.get("collector_summary") or {}
    update = record.get("update_summary") or {}
    post_collector = record.get("post_update_quasi_real_collector_summary") or {}
    compatibility = record.get("compatibility_summary") or {}
    accounting = record.get("accounting_summary") or {}
    long_horizon = record.get("long_horizon_summary") or {}

    if teacher.get("status") != "passed" or _string_list(teacher.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _float_value(teacher.get("teacher_agreement_rate"), 0.0) < float(
        validation.get("min_teacher_agreement_rate", 0.9)
    ):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _int_value(teacher.get("unsafe_disagreement_count")) or _int_value(
        teacher.get("policy_changed_gate_rejected_count")
    ):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)

    for summary in (collector, post_collector):
        if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
            _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
        if _int_value(summary.get("ppo_trainable_transition_count")) < _int_value(
            validation.get("min_ppo_trainable_transition_count"), 24
        ):
            _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
        for field in (
            "source_fallback_trainable_count",
            "invalid_action_mask_count",
            "empty_action_mask_count",
            "missing_log_prob_count",
            "missing_value_count",
            "non_finite_reward_count",
            "fallback_or_open_grid_count",
            "safety_regression_count",
            "contract_violation_count",
            "path_cost_regression_count",
            "risk_regression_count",
            "source_selection_regression_count",
        ):
            if _int_value(summary.get(field)):
                _append_reason(reasons, NEXT_POST_GATE_REGRESSION)

    update_reasons = set(_string_list(update.get("reason_codes")))
    unexpected_update_reasons = update_reasons - {EXPECTED_WRAPPER_GATE_REASON}
    if update.get("status") not in {"passed", "failed"} or unexpected_update_reasons:
        _append_reason(reasons, NEXT_NON_FINITE)
    if _int_value(update.get("optimizer_train_transition_count")) < _int_value(
        validation.get("min_optimizer_train_transition_count"), 24
    ):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    if _int_value(update.get("source_fallback_trainable_count")):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    for field in (
        "validation_test_optimizer_transition_count",
        "non_empty_gate_reason_optimizer_transition_count",
        "disallowed_source_optimizer_transition_count",
    ):
        if _int_value(update.get(field)):
            _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    if _float_value(update.get("old_log_prob_max_abs_error"), math.inf) > float(
        validation.get("max_old_log_prob_abs_error", 1.0e-4)
    ):
        _append_reason(reasons, NEXT_ON_POLICY_INVALID)
    if _float_value(update.get("old_value_max_abs_error"), math.inf) > float(
        validation.get("max_old_value_abs_error", 1.0e-4)
    ):
        _append_reason(reasons, NEXT_ON_POLICY_INVALID)
    for field in (
        "loss_non_finite_count",
        "non_finite_gradient_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
    ):
        if _int_value(update.get(field)):
            _append_reason(reasons, NEXT_NON_FINITE)
    if _float_value(update.get("parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(reasons, NEXT_POLICY_DRIFT)
    if abs(_float_value(update.get("approx_kl"), math.inf)) > float(validation.get("max_abs_approx_kl", 0.25)):
        _append_reason(reasons, NEXT_POLICY_DRIFT)
    if _float_value(update.get("max_grad_norm_after_clip"), math.inf) > float(
        validation.get("max_grad_norm_after_clip", 1.0)
    ) + 1.0e-8:
        _append_reason(reasons, NEXT_POLICY_DRIFT)
    if update.get("post_update_quasi_real_teacher_following_status") != "passed":
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _float_value(update.get("post_update_quasi_real_teacher_agreement_rate"), 0.0) < float(
        validation.get("min_teacher_agreement_rate", 0.9)
    ):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _int_value(update.get("post_update_quasi_real_unsafe_disagreement_count")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if update.get("post_update_quasi_real_collector_status") != "passed":
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _int_value(update.get("post_update_quasi_real_collector_trainable_transition_count")) < _int_value(
        validation.get("min_ppo_trainable_transition_count"), 24
    ):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    for flag in ("publishes_checkpoint", "replaces_default_policy", "performance_claimed", "formal_training_ready_claimed"):
        if update.get(flag) is True:
            _append_reason(reasons, NEXT_POST_GATE_REGRESSION)

    if compatibility.get("status") != "passed" or _string_list(compatibility.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if compatibility.get("diagnosis_verdict") != "pre_existing_generated_sequential_contract_mismatch":
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if accounting.get("status") != "passed" or _string_list(accounting.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if accounting.get("diagnosis_verdict_after_origin_split") != "pre_existing_generated_sequential_contract_mismatch":
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _int_value(accounting.get("controlled_path_cost_regression_count")) or _int_value(
        accounting.get("controlled_risk_regression_count")
    ):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)

    if long_horizon.get("status") != "passed" or _string_list(long_horizon.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if long_horizon.get("verdict") != "long_horizon_teacher_skill_contract_aligned":
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _int_value(long_horizon.get("controlled_regression_episode_count")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)

    return reasons


def _load_round_record(step: dict[str, Path | int], *, repo_root: Path) -> dict[str, Any]:
    update_root = Path(step["update_root"])
    return {
        "round_index": int(step["round_index"]),
        "base_candidate_root": _display_path(Path(step["base_candidate_root"]), repo_root),
        "teacher_following_root": _display_path(Path(step["teacher_following_root"]), repo_root),
        "collector_root": _display_path(Path(step["collector_root"]), repo_root),
        "update_root": _display_path(update_root, repo_root),
        "compatibility_root": _display_path(Path(step["compatibility_root"]), repo_root),
        "accounting_root": _display_path(Path(step["accounting_root"]), repo_root),
        "long_horizon_root": _display_path(Path(step["long_horizon_root"]), repo_root),
        "teacher_following_summary": _load_json(
            Path(step["teacher_following_root"]) / "quasi-real-guarded-teacher-following-pilot-summary.json"
        ),
        "collector_summary": _load_json(Path(step["collector_root"]) / "ppo-rollout-collector-summary.json"),
        "update_summary": _load_json(update_root / "limited-quasi-real-ppo-update-smoke-summary.json"),
        "post_update_quasi_real_collector_summary": _load_json(
            update_root / "post_update_quasi_real_collector" / "ppo-rollout-collector-summary.json"
        ),
        "compatibility_summary": _load_json(
            Path(step["compatibility_root"]) / "quasi-real-generated-sequential-contract-compatibility-summary.json"
        ),
        "accounting_summary": _load_json(
            Path(step["accounting_root"]) / "generated-sequential-gate-metric-accounting-audit-summary.json"
        ),
        "long_horizon_summary": _load_json(
            Path(step["long_horizon_root"]) / "long-horizon-teacher-skill-contract-summary.json"
        ),
    }


def _failed_round_record(
    step: dict[str, Path | int],
    exc: subprocess.CalledProcessError,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "round_index": int(step["round_index"]),
        "base_candidate_root": _display_path(Path(step["base_candidate_root"]), repo_root),
        "update_root": _display_path(Path(step["update_root"]), repo_root),
        "command_failed": True,
        "command": list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
        "returncode": exc.returncode,
        "teacher_following_summary": {},
        "collector_summary": {},
        "update_summary": {},
        "post_update_quasi_real_collector_summary": {},
        "long_horizon_summary": {},
    }


def _write_outputs(
    *,
    output_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    drift_report: dict[str, Any],
    rejection_report: dict[str, Any],
    round_records: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / config["output_files"]["summary"], summary)
    _write_json(output_root / config["output_files"]["drift_report"], drift_report)
    _write_json(output_root / config["output_files"]["rejection_report"], rejection_report)
    rounds_path = output_root / config["output_files"]["rounds"]
    with rounds_path.open("w", encoding="utf-8") as handle:
        for record in round_records:
            handle.write(json.dumps(_json_safe(record, repo_root), ensure_ascii=False) + "\n")


def _copy_final_artifacts(record: dict[str, Any], *, output_root: Path, repo_root: Path) -> None:
    final_root = output_root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    update_root = _resolve_path(record.get("update_root", ""), repo_root)
    if update_root.is_dir():
        candidate_dest = final_root / "candidate"
        if candidate_dest.exists():
            shutil.rmtree(candidate_dest)
        shutil.copytree(update_root, candidate_dest, ignore=shutil.ignore_patterns("post_update_*"))
    for key, dest_name in (
        ("teacher_following_root", "quasi-real-teacher-following"),
        ("collector_root", "quasi-real-collector"),
        ("long_horizon_root", "long-horizon"),
    ):
        source = _resolve_path(record.get(key, ""), repo_root)
        dest = final_root / dest_name
        if source.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)


def _run_progress_stage(
    *,
    progress,
    stage: str,
    current: int,
    total: int,
    round_index: int,
    summary_path: Path,
    repo_root: Path,
    metrics_loader,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    check: bool,
) -> subprocess.CompletedProcess[str]:
    progress.emit(
        stage=stage,
        status="start",
        current=current,
        total=total,
        round_index=round_index,
        message=f"round {round_index + 1}: {stage.replace('_', ' ')}",
        summary_path=_display_path(summary_path, repo_root),
    )
    try:
        result = _run_command(command, cwd=cwd, env=env, check=check)
    except subprocess.CalledProcessError:
        progress.emit(
            stage=stage,
            status="failed",
            current=current,
            total=total,
            round_index=round_index,
            message=f"round {round_index + 1}: {stage.replace('_', ' ')} failed",
            summary_path=_display_path(summary_path, repo_root),
        )
        progress.finalize(status="failed", recommended_debug_artifact=_display_path(summary_path, repo_root))
        raise
    summary = _load_json(summary_path)
    progress.emit(
        stage=stage,
        status="passed",
        current=current,
        total=total,
        round_index=round_index,
        message=f"round {round_index + 1}: {stage.replace('_', ' ')} completed",
        summary_path=_display_path(summary_path, repo_root),
        reason_codes=_string_list(summary.get("reason_codes")),
        metrics=metrics_loader(summary) if summary else {},
    )
    return result


def _teacher_following_progress_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "quasi_real_context_count": _int_value(summary.get("quasi_real_context_count")),
        "teacher_agreement_rate": _float_value(summary.get("teacher_agreement_rate"), 0.0),
        "unsafe_disagreement_count": _int_value(summary.get("unsafe_disagreement_count")),
    }


def _compatibility_progress_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagnosis_verdict": summary.get("diagnosis_verdict"),
        "failed_step_count": _int_value(summary.get("failed_step_count")),
        "base_generated_sequential_status": summary.get("base_generated_sequential_status"),
        "updated_generated_sequential_status": summary.get("updated_generated_sequential_status"),
    }


def _accounting_progress_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "legacy_mismatch_count": _int_value(summary.get("legacy_mismatch_count")),
        "diagnosis_verdict_after_origin_split": summary.get("diagnosis_verdict_after_origin_split"),
        "controlled_path_cost_regression_count": _int_value(summary.get("controlled_path_cost_regression_count")),
        "controlled_risk_regression_count": _int_value(summary.get("controlled_risk_regression_count")),
    }


def _long_horizon_progress_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": summary.get("verdict"),
        "teacher_equivalent_episode_count": _int_value(summary.get("teacher_equivalent_episode_count")),
        "beyond_teacher_episode_count": _int_value(summary.get("beyond_teacher_episode_count")),
        "controlled_regression_episode_count": _int_value(summary.get("controlled_regression_episode_count")),
    }


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True)


def _collector_regression_count(summary: dict[str, Any]) -> int:
    return sum(
        _int_value(summary.get(field))
        for field in (
            "invalid_action_mask_count",
            "empty_action_mask_count",
            "missing_log_prob_count",
            "missing_value_count",
            "non_finite_reward_count",
            "fallback_or_open_grid_count",
            "safety_regression_count",
            "contract_violation_count",
            "path_cost_regression_count",
            "risk_regression_count",
            "source_selection_regression_count",
        )
    )


def _load_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("config_paths", "output_files", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    if NEXT_ON_POLICY_INVALID in reason_codes:
        return "rebuild_on_policy_quasi_real_collector"
    if NEXT_TRAINABLE_INSUFFICIENT in reason_codes:
        return "quasi_real_trainable_transition_filter_refinement_required"
    if NEXT_NON_FINITE in reason_codes:
        return "quasi_real_ppo_update_numeric_guard_refinement_required"
    if NEXT_POLICY_DRIFT in reason_codes:
        return "quasi_real_iterative_update_drift_guard_refinement_required"
    return "quasi_real_iterative_post_update_contract_refinement_required"


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _finite_or_inf(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _json_safe(value: Any, repo_root: Path) -> Any:
    if isinstance(value, Path):
        return _display_path(value, repo_root)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, repo_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item, repo_root) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
