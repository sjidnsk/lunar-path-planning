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


CONFIG_SCHEMA_VERSION = "guarded-ppo-rollout-pilot-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-ppo-rollout-pilot-summary/v1"
REJECTION_SCHEMA_VERSION = "guarded-ppo-rollout-rejection-report/v1"

NEXT_CONTRACT_INVALID = "guarded_ppo_rollout_contract_invalid"
NEXT_ON_POLICY_INVALID = "guarded_ppo_on_policy_contract_invalid"
NEXT_GATE_REGRESSION = "guarded_ppo_rollout_gate_regression"
NEXT_TRAINABLE_INSUFFICIENT = "guarded_ppo_trainable_transition_count_insufficient"
NEXT_NON_FINITE = "guarded_ppo_update_loss_non_finite"
NEXT_POLICY_DRIFT = "guarded_ppo_policy_drift_detected"
NEXT_POST_UPDATE_REGRESSION = "guarded_ppo_post_update_regression"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a guarded PPO rollout pilot.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--base-candidate-root", required=True)
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

    summary = run_guarded_ppo_rollout_pilot(
        source_root=_resolve_path(args.source_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
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
                "ppo_trainable_transition_count": summary["ppo_trainable_transition_count"],
                "optimizer_train_transition_count": summary["optimizer_train_transition_count"],
                "guarded_rollout_pilot_passed": summary["guarded_rollout_pilot_passed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def build_guarded_pilot_plan(*, output_root: Path, base_candidate_root: Path) -> dict[str, Path]:
    return {
        "base_candidate_root": base_candidate_root,
        "sequential_root": output_root / "pilot" / "sequential",
        "collector_root": output_root / "pilot" / "collector",
        "update_root": output_root / "update",
        "post_sequential_root": output_root / "final" / "sequential",
        "post_collector_root": output_root / "final" / "collector",
    }


def run_guarded_ppo_rollout_pilot(
    *,
    source_root: Path,
    base_candidate_root: Path,
    raw_baseline_candidate_root: Path,
    dev_root: Path,
    val_root: Path,
    test_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    plan = build_guarded_pilot_plan(
        output_root=output_root,
        base_candidate_root=base_candidate_root,
    )
    python_bin = sys.executable
    env = dict(os.environ)
    env["PYTHON"] = python_bin
    paths = config["config_paths"]

    for path in (
        plan["sequential_root"],
        plan["collector_root"],
        plan["update_root"],
        plan["post_sequential_root"],
        plan["post_collector_root"],
    ):
        if path.exists():
            shutil.rmtree(path)

    _run_command(
        [
            "bash",
            str(repo_root / "scripts" / "run_policy_gated_sequential_canary_rollout.sh"),
            "--source-root",
            str(source_root),
            "--candidate-root",
            str(base_candidate_root),
            "--batch-root",
            str(plan["sequential_root"]),
            "--config",
            str(repo_root / paths["sequential_canary_config"]),
        ],
        cwd=repo_root,
        env=env,
    )
    _run_command(
        [
            "bash",
            str(repo_root / "scripts" / "run_ppo_rollout_collector_dry_run.sh"),
            "--sequential-root",
            str(plan["sequential_root"]),
            "--candidate-root",
            str(base_candidate_root),
            "--output-root",
            str(plan["collector_root"]),
            "--config",
            str(repo_root / paths["collector_config"]),
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
            str(base_candidate_root),
            "--collector-root",
            str(plan["collector_root"]),
            "--output-root",
            str(plan["update_root"]),
            "--config",
            str(repo_root / paths["update_config"]),
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
            str(plan["update_root"]),
            "--config",
            str(repo_root / paths["raw_generalization_config"]),
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
            str(plan["update_root"]),
            "--batch-root",
            str(plan["post_sequential_root"]),
            "--config",
            str(repo_root / paths["sequential_canary_config"]),
        ],
        cwd=repo_root,
        env=env,
    )
    _run_command(
        [
            "bash",
            str(repo_root / "scripts" / "run_ppo_rollout_collector_dry_run.sh"),
            "--sequential-root",
            str(plan["post_sequential_root"]),
            "--candidate-root",
            str(plan["update_root"]),
            "--output-root",
            str(plan["post_collector_root"]),
            "--config",
            str(repo_root / paths["collector_config"]),
        ],
        cwd=repo_root,
        env=env,
    )

    summary, rejection_report = summarize_guarded_ppo_rollout_pilot(
        output_root=output_root,
        config=config,
        repo_root=repo_root,
        base_candidate_root=base_candidate_root,
        update_root=plan["update_root"],
        pilot_sequential_summary=_load_json(
            plan["sequential_root"] / "policy-gated-sequential-canary-rollout-summary.json"
        ),
        pilot_collector_summary=_load_json(plan["collector_root"] / "ppo-rollout-collector-summary.json"),
        update_summary=_load_json(plan["update_root"] / "limited-ppo-update-smoke-summary.json"),
        raw_generalization_summary=_load_json(
            plan["update_root"] / "raw-policy-generalization-evaluation-summary.json"
        ),
        post_update_sequential_summary=_load_json(
            plan["post_sequential_root"] / "policy-gated-sequential-canary-rollout-summary.json"
        ),
        post_update_collector_summary=_load_json(
            plan["post_collector_root"] / "ppo-rollout-collector-summary.json"
        ),
    )
    _write_outputs(output_root=output_root, config=config, summary=summary, rejection_report=rejection_report)
    _copy_guarded_artifacts(plan, output_root=output_root, config=config)
    _copy_post_update_artifacts(plan, output_root=output_root)
    return summary


def summarize_guarded_ppo_rollout_pilot(
    *,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    base_candidate_root: Path,
    update_root: Path,
    pilot_sequential_summary: dict[str, Any],
    pilot_collector_summary: dict[str, Any],
    update_summary: dict[str, Any],
    raw_generalization_summary: dict[str, Any],
    post_update_sequential_summary: dict[str, Any],
    post_update_collector_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = config["validation"]
    reason_codes: list[str] = []
    rejection_records: list[dict[str, Any]] = []

    _extend_stage_reasons(
        reason_codes,
        rejection_records,
        stage="pilot_sequential",
        reasons=_sequential_reasons(pilot_sequential_summary, validation, post_update=False),
    )
    _extend_stage_reasons(
        reason_codes,
        rejection_records,
        stage="pilot_collector",
        reasons=_collector_reasons(pilot_collector_summary, validation),
    )
    _extend_stage_reasons(
        reason_codes,
        rejection_records,
        stage="ppo_update",
        reasons=_update_reasons(update_summary, validation),
    )
    _extend_stage_reasons(
        reason_codes,
        rejection_records,
        stage="raw_generalization",
        reasons=_raw_generalization_reasons(raw_generalization_summary, validation),
    )
    _extend_stage_reasons(
        reason_codes,
        rejection_records,
        stage="post_update_sequential",
        reasons=_sequential_reasons(post_update_sequential_summary, validation, post_update=True),
    )
    _extend_stage_reasons(
        reason_codes,
        rejection_records,
        stage="post_update_collector",
        reasons=_collector_reasons(post_update_collector_summary, validation),
    )

    summary_path = output_root / config["output_files"]["summary"]
    rejection_path = output_root / config["output_files"]["rejection_report"]
    current_git = _git_snapshot(repo_root)
    status = "failed" if reason_codes else "passed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "next_required_change": _next_required_change(reason_codes),
        "guarded_rollout_pilot_passed": status == "passed",
        "base_candidate_root": _display_path(base_candidate_root, repo_root),
        "update_root": _display_path(update_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "episode_count": _int_value(pilot_sequential_summary.get("episode_count")),
        "step_count": _int_value(pilot_sequential_summary.get("step_count")),
        "ppo_trainable_transition_count": _int_value(
            pilot_collector_summary.get("ppo_trainable_transition_count")
        ),
        "optimizer_train_transition_count": _int_value(update_summary.get("optimizer_train_transition_count")),
        "source_fallback_trainable_count": _int_value(
            pilot_collector_summary.get("source_fallback_trainable_count")
        )
        + _int_value(update_summary.get("source_fallback_trainable_count")),
        "state_continuity_violation_count": _int_value(
            pilot_collector_summary.get("state_continuity_violation_count")
        )
        + _int_value(pilot_sequential_summary.get("state_continuity_violation_count")),
        "invalid_action_mask_count": _int_value(pilot_collector_summary.get("invalid_action_mask_count")),
        "empty_action_mask_count": _int_value(pilot_collector_summary.get("empty_action_mask_count")),
        "missing_log_prob_count": _int_value(pilot_collector_summary.get("missing_log_prob_count")),
        "missing_value_count": _int_value(pilot_collector_summary.get("missing_value_count")),
        "non_finite_reward_count": _int_value(pilot_collector_summary.get("non_finite_reward_count"))
        + _int_value(update_summary.get("non_finite_reward_count")),
        "old_log_prob_max_abs_error": _finite_or_inf(update_summary.get("old_log_prob_max_abs_error")),
        "old_value_max_abs_error": _finite_or_inf(update_summary.get("old_value_max_abs_error")),
        "update_requested_device": update_summary.get("requested_device"),
        "update_resolved_device": update_summary.get("resolved_device"),
        "update_cuda_available": update_summary.get("cuda_available"),
        "update_cuda_device_name": update_summary.get("cuda_device_name"),
        "update_fallback_to_cpu": update_summary.get("fallback_to_cpu"),
        "parameter_l2_delta": _finite_or_inf(update_summary.get("parameter_l2_delta")),
        "approx_kl": _finite_or_inf(update_summary.get("approx_kl")),
        "max_grad_norm_after_clip": _finite_or_inf(update_summary.get("max_grad_norm_after_clip")),
        "post_update_raw_test_regression_count": _int_value(
            raw_generalization_summary.get("test_raw_policy_regression_count")
        ),
        "post_update_sequential_rejected_count": _int_value(
            post_update_sequential_summary.get("canary_rejected_policy_choice_count")
        ),
        "post_update_collector_regression_count": _collector_regression_count(
            post_update_collector_summary
        ),
        "pilot_sequential_summary_path": "pilot/sequential/policy-gated-sequential-canary-rollout-summary.json",
        "pilot_collector_summary_path": "pilot/collector/ppo-rollout-collector-summary.json",
        "update_summary_path": "update/limited-ppo-update-smoke-summary.json",
        "post_update_raw_policy_generalization_summary_path": "final/raw-policy-generalization-evaluation-summary.json",
        "post_update_sequential_summary_path": "final/sequential/policy-gated-sequential-canary-rollout-summary.json",
        "post_update_collector_summary_path": "final/collector/ppo-rollout-collector-summary.json",
        "summary": _display_path(summary_path, repo_root),
        "rejection_report": _display_path(rejection_path, repo_root),
        "episodes": _display_path(output_root / config["output_files"].get("episodes", ""), repo_root)
        if config["output_files"].get("episodes")
        else None,
        "transitions": _display_path(output_root / config["output_files"].get("transitions", ""), repo_root)
        if config["output_files"].get("transitions")
        else None,
        "reward_audit": _display_path(output_root / config["output_files"].get("reward_audit", ""), repo_root)
        if config["output_files"].get("reward_audit")
        else None,
        "update_summary": _display_path(output_root / config["output_files"].get("update_summary", ""), repo_root)
        if config["output_files"].get("update_summary")
        else None,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_formal_ppo_rollout": False,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    rejection_report = {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "rejected_stage_count": len(rejection_records),
        "reason_code_counts": dict(
            sorted(Counter(reason for record in rejection_records for reason in record["reason_codes"]).items())
        ),
        "records": rejection_records,
    }
    return summary, rejection_report


def _sequential_reasons(summary: dict[str, Any], validation: dict[str, Any], *, post_update: bool) -> list[str]:
    reasons: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(reasons, NEXT_GATE_REGRESSION)
    for field, minimum in (
        ("episode_count", _int_value(validation.get("min_episode_count"), 36)),
        ("step_count", _int_value(validation.get("min_step_count"), 108)),
        (
            "accepted_takeover_family_count",
            _int_value(validation.get("min_accepted_takeover_family_count"), 6),
        ),
        (
            "multi_step_accepted_episode_count",
            _int_value(validation.get("min_multi_step_accepted_episode_count"), 12),
        ),
    ):
        if _int_value(summary.get(field)) < minimum:
            _append_reason(reasons, NEXT_GATE_REGRESSION if post_update else NEXT_CONTRACT_INVALID)
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
        if _int_value(summary.get(field)):
            _append_reason(reasons, NEXT_GATE_REGRESSION if not post_update else NEXT_POST_UPDATE_REGRESSION)
    return reasons


def _collector_reasons(summary: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(reasons, NEXT_CONTRACT_INVALID)
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
        "state_continuity_violation_count",
        "fallback_or_open_grid_count",
        "safety_regression_count",
        "contract_violation_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    ):
        if _int_value(summary.get(field)):
            _append_reason(reasons, NEXT_CONTRACT_INVALID)
    return reasons


def _update_reasons(summary: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(reasons, NEXT_NON_FINITE)
    if _int_value(summary.get("optimizer_train_transition_count")) < _int_value(
        validation.get("min_optimizer_train_transition_count"), 24
    ):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    if _int_value(summary.get("source_fallback_trainable_count")):
        _append_reason(reasons, NEXT_TRAINABLE_INSUFFICIENT)
    if _float_value(summary.get("old_log_prob_max_abs_error"), math.inf) > float(
        validation.get("max_old_log_prob_abs_error", 1.0e-4)
    ):
        _append_reason(reasons, NEXT_ON_POLICY_INVALID)
    if _float_value(summary.get("old_value_max_abs_error"), math.inf) > float(
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
        if _int_value(summary.get(field)):
            _append_reason(reasons, NEXT_NON_FINITE)
    if _float_value(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(reasons, NEXT_POLICY_DRIFT)
    if abs(_float_value(summary.get("approx_kl"), math.inf)) > float(validation.get("max_abs_approx_kl", 0.25)):
        _append_reason(reasons, NEXT_POLICY_DRIFT)
    if _float_value(summary.get("max_grad_norm_after_clip"), math.inf) > float(
        validation.get("max_grad_norm_after_clip", 1.0)
    ) + 1.0e-8:
        _append_reason(reasons, NEXT_POLICY_DRIFT)
    return reasons


def _raw_generalization_reasons(summary: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(reasons, NEXT_POST_UPDATE_REGRESSION)
    if _int_value(summary.get("test_raw_policy_regression_count")) > _int_value(
        validation.get("max_raw_test_regression_count"), 0
    ):
        _append_reason(reasons, NEXT_POST_UPDATE_REGRESSION)
    return reasons


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


def _extend_stage_reasons(
    reason_codes: list[str],
    rejection_records: list[dict[str, Any]],
    *,
    stage: str,
    reasons: list[str],
) -> None:
    if not reasons:
        return
    for reason in reasons:
        _append_reason(reason_codes, reason)
    rejection_records.append({"stage": stage, "reason_codes": _dedup(reasons)})


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    if NEXT_ON_POLICY_INVALID in reason_codes:
        return NEXT_ON_POLICY_INVALID
    if NEXT_TRAINABLE_INSUFFICIENT in reason_codes:
        return NEXT_TRAINABLE_INSUFFICIENT
    if NEXT_NON_FINITE in reason_codes:
        return NEXT_NON_FINITE
    if NEXT_POLICY_DRIFT in reason_codes:
        return NEXT_POLICY_DRIFT
    if NEXT_POST_UPDATE_REGRESSION in reason_codes:
        return NEXT_POST_UPDATE_REGRESSION
    if NEXT_GATE_REGRESSION in reason_codes:
        return NEXT_GATE_REGRESSION
    return NEXT_CONTRACT_INVALID


def _write_outputs(
    *,
    output_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    rejection_report: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / config["output_files"]["summary"], summary)
    _write_json(output_root / config["output_files"]["rejection_report"], rejection_report)


def _copy_post_update_artifacts(plan: dict[str, Path], *, output_root: Path) -> None:
    final_root = output_root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    raw_summary = plan["update_root"] / "raw-policy-generalization-evaluation-summary.json"
    if raw_summary.is_file():
        shutil.copy2(raw_summary, final_root / raw_summary.name)


def _copy_guarded_artifacts(plan: dict[str, Path], *, output_root: Path, config: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = config["output_files"]
    for source, key in (
        (plan["collector_root"] / "ppo-rollout-episodes.jsonl", "episodes"),
        (plan["collector_root"] / "ppo-rollout-transitions.jsonl", "transitions"),
        (plan["collector_root"] / "ppo-rollout-reward-audit.json", "reward_audit"),
        (plan["update_root"] / "limited-ppo-update-smoke-summary.json", "update_summary"),
    ):
        target_name = outputs.get(key)
        if target_name and source.is_file():
            shutil.copy2(source, output_root / target_name)


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("config_paths", "output_files", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_reason(values: list[str], reason: str) -> None:
    if reason and reason not in values:
        values.append(reason)


def _dedup(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        _append_reason(result, value)
    return result


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_or_inf(value: Any) -> float | str:
    result = _float_value(value, math.inf)
    if math.isfinite(result):
        return result
    return "inf"


if __name__ == "__main__":
    raise SystemExit(main())
