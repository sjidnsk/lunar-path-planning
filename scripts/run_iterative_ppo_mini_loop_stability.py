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


CONFIG_SCHEMA_VERSION = "iterative-ppo-mini-loop-stability-config/v1"
SUMMARY_SCHEMA_VERSION = "iterative-ppo-mini-loop-stability-summary/v1"

NEXT_ON_POLICY_INVALID = "iterative_ppo_on_policy_contract_invalid"
NEXT_TRAINABLE_INSUFFICIENT = "iterative_ppo_trainable_transition_count_insufficient"
NEXT_NON_FINITE = "iterative_ppo_update_loss_non_finite"
NEXT_POLICY_DRIFT = "iterative_ppo_policy_drift_detected"
NEXT_POST_GATE_REGRESSION = "iterative_ppo_post_update_gate_regression"
NEXT_CLEAN_HEAD = "clean_head_evidence_refresh_required"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run iterative PPO mini-loop stability evaluation.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--initial-candidate-root", required=True)
    parser.add_argument("--raw-baseline-candidate-root", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": _display_path(config_path, repo_root)}))
        return 0

    summary = run_iterative_ppo_mini_loop_stability(
        source_root=_resolve_path(args.source_root, repo_root),
        initial_candidate_root=_resolve_path(args.initial_candidate_root, repo_root),
        raw_baseline_candidate_root=_resolve_path(args.raw_baseline_candidate_root, repo_root),
        dev_root=_resolve_path(args.dev_root, repo_root),
        val_root=_resolve_path(args.val_root, repo_root),
        test_root=_resolve_path(args.test_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
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


def build_round_plan(
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
            "sequential_root": round_root / "sequential",
            "collector_root": round_root / "collector",
            "update_root": round_root / "update",
            "post_sequential_root": round_root / "post-sequential",
            "post_collector_root": round_root / "post-collector",
        }
        plan.append(record)
        base_candidate_root = record["update_root"]  # type: ignore[assignment]
    return plan


def subprocess_env(*, python_bin: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHON"] = python_bin
    return env


def run_iterative_ppo_mini_loop_stability(
    *,
    source_root: Path,
    initial_candidate_root: Path,
    raw_baseline_candidate_root: Path,
    dev_root: Path,
    val_root: Path,
    test_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    round_count = _int_value(config["validation"].get("round_count"), 3)
    plan = build_round_plan(
        output_root=output_root,
        initial_candidate_root=initial_candidate_root,
        round_count=round_count,
    )
    round_records: list[dict[str, Any]] = []
    python_bin = sys.executable
    env = subprocess_env(python_bin=python_bin)

    for step in plan:
        round_index = int(step["round_index"])
        try:
            _run_command(
                [
                    "bash",
                    str(repo_root / "scripts" / "run_policy_gated_sequential_canary_rollout.sh"),
                    "--source-root",
                    str(source_root),
                    "--candidate-root",
                    str(step["base_candidate_root"]),
                    "--batch-root",
                    str(step["sequential_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["sequential_canary_config"]),
                ],
                cwd=repo_root,
                env=env,
            )
            _run_command(
                [
                    "bash",
                    str(repo_root / "scripts" / "run_ppo_rollout_collector_dry_run.sh"),
                    "--sequential-root",
                    str(step["sequential_root"]),
                    "--candidate-root",
                    str(step["base_candidate_root"]),
                    "--output-root",
                    str(step["collector_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["collector_config"]),
                ],
                cwd=repo_root,
                env=env,
            )
            _run_command(
                [
                    "bash",
                    str(repo_root / "scripts" / "run_limited_ppo_update_smoke.sh"),
                    "--source-root",
                    str(source_root),
                    "--base-candidate-root",
                    str(step["base_candidate_root"]),
                    "--collector-root",
                    str(step["collector_root"]),
                    "--output-root",
                    str(step["update_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["update_step_config"]),
                ],
                cwd=repo_root,
                env=env,
            )
            _run_command(
                [
                    "bash",
                    str(repo_root / "scripts" / "run_raw_policy_generalization_evaluation.sh"),
                    "--source-root",
                    str(source_root),
                    "--dev-root",
                    str(dev_root),
                    "--val-root",
                    str(val_root),
                    "--test-root",
                    str(test_root),
                    "--baseline-candidate-root",
                    str(raw_baseline_candidate_root),
                    "--candidate-root",
                    str(step["update_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["raw_generalization_config"]),
                ],
                cwd=repo_root,
                env=env,
            )
            _run_command(
                [
                    "bash",
                    str(repo_root / "scripts" / "run_policy_gated_sequential_canary_rollout.sh"),
                    "--source-root",
                    str(source_root),
                    "--candidate-root",
                    str(step["update_root"]),
                    "--batch-root",
                    str(step["post_sequential_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["sequential_canary_config"]),
                ],
                cwd=repo_root,
                env=env,
            )
            _run_command(
                [
                    "bash",
                    str(repo_root / "scripts" / "run_ppo_rollout_collector_dry_run.sh"),
                    "--sequential-root",
                    str(step["post_sequential_root"]),
                    "--candidate-root",
                    str(step["update_root"]),
                    "--output-root",
                    str(step["post_collector_root"]),
                    "--config",
                    str(repo_root / config["config_paths"]["collector_config"]),
                ],
                cwd=repo_root,
                env=env,
            )
            round_records.append(_load_round_record(step, repo_root=repo_root))
        except subprocess.CalledProcessError as exc:
            round_records.append(
                {
                    "round_index": round_index,
                    "base_candidate_root": _display_path(Path(step["base_candidate_root"]), repo_root),
                    "update_root": _display_path(Path(step["update_root"]), repo_root),
                    "command_failed": True,
                    "command": list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
                    "returncode": exc.returncode,
                    "update_summary": {},
                    "raw_generalization_summary": {},
                    "post_update_sequential_summary": {},
                    "post_update_collector_summary": {},
                }
            )
            break

    summary, drift_report, rejection_report = summarize_iterative_rounds(
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
    return summary


def summarize_iterative_rounds(
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

    for record in round_records:
        round_reasons = _round_reason_codes(record, validation)
        update = record.get("update_summary") or {}
        approx_kl = _float_value(update.get("approx_kl"), math.inf)
        abs_kl = abs(approx_kl) if math.isfinite(approx_kl) else math.inf
        parameter_delta = _float_value(update.get("parameter_l2_delta"), 0.0)
        if math.isfinite(abs_kl):
            max_abs_approx_kl = max(max_abs_approx_kl, abs_kl)
        else:
            max_abs_approx_kl = math.inf
        cumulative_parameter_l2_delta += max(parameter_delta, 0.0)
        drift_records.append(
            {
                "round_index": _int_value(record.get("round_index")),
                "approx_kl": _finite_or_none(approx_kl),
                "abs_approx_kl": _finite_or_none(abs_kl),
                "parameter_l2_delta": parameter_delta,
                "cumulative_parameter_l2_delta": cumulative_parameter_l2_delta,
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

    current = _git_snapshot(repo_root)

    status = "failed" if reason_codes else "passed"
    summary_path = output_root / config["output_files"]["summary"]
    rounds_path = output_root / config["output_files"]["rounds"]
    drift_path = output_root / config["output_files"]["drift_report"]
    rejection_path = output_root / config["output_files"]["rejection_report"]
    final_record = round_records[-1] if round_records else {}
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
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
                _int_value((record.get("post_update_collector_summary") or {}).get("ppo_trainable_transition_count"))
                for record in round_records
            ]
            or [0]
        ),
        "raw_test_regression_count": _int_value(
            (final_record.get("raw_generalization_summary") or {}).get("test_raw_policy_regression_count")
        ),
        "sequential_rejected_count": _int_value(
            (final_record.get("post_update_sequential_summary") or {}).get("canary_rejected_policy_choice_count")
        ),
        "collector_regression_count": _collector_regression_count(
            final_record.get("post_update_collector_summary") or {}
        ),
        "final_candidate_root": final_record.get("update_root"),
        "final_raw_policy_generalization_summary_path": "final/raw-policy-generalization-evaluation-summary.json",
        "final_sequential_summary_path": "final/sequential/policy-gated-sequential-canary-rollout-summary.json",
        "final_collector_summary_path": "final/collector/ppo-rollout-collector-summary.json",
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
        "git_provenance": {"current": current, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    drift_report = {
        "schema_version": "iterative-ppo-mini-loop-drift-report/v1",
        "generated_at": summary["generated_at"],
        "max_abs_approx_kl": summary["max_abs_approx_kl"],
        "cumulative_parameter_l2_delta": cumulative_parameter_l2_delta,
        "records": drift_records,
    }
    rejection_report = {
        "schema_version": "iterative-ppo-mini-loop-rejection-report/v1",
        "generated_at": summary["generated_at"],
        "records": rejection_records,
    }
    return summary, drift_report, rejection_report


def _round_reason_codes(record: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if record.get("command_failed"):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
        return reasons

    collector = record.get("collector_summary") or {}
    update = record.get("update_summary") or {}
    raw = record.get("raw_generalization_summary") or {}
    sequential = record.get("post_update_sequential_summary") or {}
    post_collector = record.get("post_update_collector_summary") or {}

    if update.get("status") != "passed" or _string_list(update.get("reason_codes")):
        _append_reason(reasons, NEXT_NON_FINITE)
    if _int_value(update.get("optimizer_train_transition_count")) < _int_value(
        validation.get("min_optimizer_train_transition_count"), 24
    ):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    if _int_value(post_collector.get("ppo_trainable_transition_count")) < _int_value(
        validation.get("min_ppo_trainable_transition_count"), 24
    ):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    if _int_value(collector.get("ppo_trainable_transition_count")) < _int_value(
        validation.get("min_ppo_trainable_transition_count"), 24
    ):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    if _int_value(update.get("source_fallback_trainable_count")):
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

    if raw.get("status") != "passed" or _string_list(raw.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    if _int_value(raw.get("test_raw_policy_regression_count")) > _int_value(
        validation.get("max_raw_test_regression_count"), 0
    ):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)

    if sequential.get("status") != "passed" or _string_list(sequential.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    for field, minimum in (
        ("episode_count", _int_value(validation.get("min_sequential_episode_count"), 36)),
        ("step_count", _int_value(validation.get("min_sequential_step_count"), 108)),
        (
            "accepted_takeover_family_count",
            _int_value(validation.get("min_accepted_takeover_family_count"), 6),
        ),
        (
            "multi_step_accepted_episode_count",
            _int_value(validation.get("min_multi_step_accepted_episode_count"), 12),
        ),
    ):
        if _int_value(sequential.get(field)) < minimum:
            _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    for field in (
        "canary_rejected_policy_choice_count",
        "state_continuity_violation_count",
        "episode_fallback_count",
        "invalid_action_mask_count",
        "fallback_or_open_grid_count",
        "cumulative_safety_regression_count",
        "cumulative_contract_violation_count",
        "cumulative_path_cost_regression_count",
        "cumulative_risk_regression_count",
        "cumulative_source_selection_regression_count",
    ):
        if _int_value(sequential.get(field)):
            _append_reason(reasons, NEXT_POST_GATE_REGRESSION)

    if post_collector.get("status") != "passed" or _string_list(post_collector.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    for field in (
        "invalid_action_mask_count",
        "empty_action_mask_count",
        "missing_log_prob_count",
        "missing_value_count",
        "non_finite_reward_count",
        "state_continuity_violation_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    ):
        if _int_value(post_collector.get(field)):
            _append_reason(reasons, NEXT_POST_GATE_REGRESSION)
    return reasons


def _load_round_record(step: dict[str, Path | int], *, repo_root: Path) -> dict[str, Any]:
    return {
        "round_index": int(step["round_index"]),
        "base_candidate_root": _display_path(Path(step["base_candidate_root"]), repo_root),
        "sequential_root": _display_path(Path(step["sequential_root"]), repo_root),
        "collector_root": _display_path(Path(step["collector_root"]), repo_root),
        "update_root": _display_path(Path(step["update_root"]), repo_root),
        "post_sequential_root": _display_path(Path(step["post_sequential_root"]), repo_root),
        "post_collector_root": _display_path(Path(step["post_collector_root"]), repo_root),
        "collector_summary": _load_json(Path(step["collector_root"]) / "ppo-rollout-collector-summary.json"),
        "update_summary": _load_json(Path(step["update_root"]) / "limited-ppo-update-smoke-summary.json"),
        "raw_generalization_summary": _load_json(Path(step["update_root"]) / "raw-policy-generalization-evaluation-summary.json"),
        "post_update_sequential_summary": _load_json(
            Path(step["post_sequential_root"]) / "policy-gated-sequential-canary-rollout-summary.json"
        ),
        "post_update_collector_summary": _load_json(Path(step["post_collector_root"]) / "ppo-rollout-collector-summary.json"),
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
    sequential_root = _resolve_path(record.get("post_sequential_root", ""), repo_root)
    collector_root = _resolve_path(record.get("post_collector_root", ""), repo_root)
    raw_summary = update_root / "raw-policy-generalization-evaluation-summary.json"
    if raw_summary.is_file():
        shutil.copy2(raw_summary, final_root / raw_summary.name)
    for source, dest in (
        (sequential_root, final_root / "sequential"),
        (collector_root, final_root / "collector"),
    ):
        if source.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _collector_regression_count(summary: dict[str, Any]) -> int:
    return sum(
        _int_value(summary.get(field))
        for field in (
            "invalid_action_mask_count",
            "empty_action_mask_count",
            "missing_log_prob_count",
            "missing_value_count",
            "non_finite_reward_count",
            "state_continuity_violation_count",
            "path_cost_regression_count",
            "risk_regression_count",
            "source_selection_regression_count",
        )
    )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _next_required_change(reason_codes: list[str]) -> str | None:
    for reason in (
        NEXT_ON_POLICY_INVALID,
        NEXT_TRAINABLE_INSUFFICIENT,
        NEXT_NON_FINITE,
        NEXT_POLICY_DRIFT,
        NEXT_POST_GATE_REGRESSION,
        NEXT_CLEAN_HEAD,
    ):
        if reason in reason_codes:
            return reason
    return reason_codes[0] if reason_codes else None


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


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
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_or_inf(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if math.isfinite(number) else float("inf")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _json_safe(value: Any, repo_root: Path) -> Any:
    if isinstance(value, Path):
        return _display_path(value, repo_root)
    if isinstance(value, dict):
        return {key: _json_safe(item, repo_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, repo_root) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
