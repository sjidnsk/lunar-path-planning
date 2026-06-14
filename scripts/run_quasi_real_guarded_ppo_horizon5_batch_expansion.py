from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover - import path used by unit tests
    from scripts.git_provenance import git_snapshot


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-ppo-horizon5-batch-expansion-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1"
STEP_SCHEMA_VERSION = "quasi-real-guarded-ppo-horizon5-batch-expansion-step/v1"
EPISODE_SCHEMA_VERSION = "quasi-real-guarded-ppo-horizon5-batch-expansion-episode/v1"
REWARD_AUDIT_SCHEMA_VERSION = "quasi-real-guarded-ppo-horizon5-batch-expansion-reward-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = (
    "quasi-real-guarded-ppo-horizon5-batch-expansion-rejection-report/v1"
)
EXPECTED_READINESS_STATUS = "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated"

SUMMARY_FILE = "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json"
EPISODES_FILE = "quasi-real-guarded-ppo-horizon5-batch-expansion-episodes.jsonl"
STEPS_FILE = "quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl"
REWARD_AUDIT_FILE = "quasi-real-guarded-ppo-horizon5-batch-expansion-reward-audit.json"
REJECTION_REPORT_FILE = (
    "quasi-real-guarded-ppo-horizon5-batch-expansion-rejection-report.json"
)
COMPARISON_FILE = "horizon5-batch-expansion-comparison.jsonl"
PROGRESS_EVENTS_FILE = "horizon5-batch-expansion-progress-events.jsonl"
READINESS_FILE = "horizon5-batch-expansion-readiness-validate-only.json"
REPORT_FILE = "horizon5-batch-expansion-report.md"

CONTROLLED_REGRESSION_REASONS = {
    "safety_regression",
    "contract_violation",
    "contract_regression",
    "path_cost_regression",
    "risk_regression",
    "source_selection_regression",
}

ReadinessRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_ppo_horizon5_batch_expansion(
    *,
    stability_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stability_root = Path(stability_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    episodes_path = output_root / files["episodes"]
    steps_path = output_root / files["steps"]
    reward_audit_path = output_root / files["reward_audit"]
    rejection_report_path = output_root / files["rejection_report"]
    comparison_path = output_root / files["comparison"]
    progress_path = output_root / files["progress_events"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    stability_summary_path = (
        stability_root / "quasi-real-guarded-ppo-stability-replay-summary.json"
    )
    stability_summary = _read_json_if_exists(stability_summary_path)
    acceptance_contract_path = stability_root / "acceptance-contract-refinement.json"
    acceptance_contract = _read_json_if_exists(acceptance_contract_path)
    freeze_summary_path = _resolve_input_path(
        stability_summary.get("freeze_summary"),
        stability_root,
        repo_root,
        fallback="outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/quasi-real-guarded-ppo-evidence-freeze-summary.json",
    )
    freeze_manifest_path = _resolve_input_path(
        stability_summary.get("freeze_manifest"),
        stability_root,
        repo_root,
        fallback="outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/quasi-real-guarded-ppo-evidence-manifest.json",
    )
    freeze_summary = _read_json_if_exists(freeze_summary_path)
    freeze_manifest = _read_json_if_exists(freeze_manifest_path)

    reason_codes: list[str] = []
    _validate_inputs(
        stability_summary,
        acceptance_contract,
        freeze_summary,
        freeze_manifest,
        reason_codes,
    )

    horizon = max(1, _int_value(config.get("horizon"), 5))
    target_episode_count = _int_value(
        config.get("expansion", {}).get("episode_count"),
        96,
    )
    replay_count = _int_value(config.get("expansion", {}).get("replay_count"), 3)
    baseline_steps_path = _baseline_steps_path(
        freeze_summary=freeze_summary,
        stability_summary=stability_summary,
        repo_root=repo_root,
    )
    baseline_steps = _read_jsonl(baseline_steps_path)
    if not baseline_steps:
        _add_reason(reason_codes, "baseline_step_records_missing")

    progress_events: list[dict[str, Any]] = []
    _add_progress(
        progress_events,
        "horizon5_batch_expansion_started",
        replay_index=None,
        episode_count=0,
        step_count=0,
    )
    episodes, steps = _build_expansion(
        baseline_steps,
        horizon=horizon,
        episode_count=target_episode_count,
        discount_factor=_float_value(config.get("discount_factor"), 0.99),
    )
    counters = _counters(steps, episodes)
    _validate_expansion(counters, config, stability_summary, reason_codes)
    _write_jsonl(episodes_path, episodes)
    _write_jsonl(steps_path, steps)
    _write_json(reward_audit_path, _reward_audit(steps, counters))
    _write_json(rejection_report_path, _rejection_report(steps, counters))

    comparison_rows: list[dict[str, Any]] = []
    passed_replay_count = 0
    baseline_signature = _step_signatures(steps)
    for replay_index in range(replay_count):
        started_at = time.perf_counter()
        replay_root = output_root / f"replay-{replay_index:02d}"
        _add_progress(
            progress_events,
            "replay_started",
            replay_index=replay_index,
            episode_count=0,
            step_count=0,
        )
        replay_episodes, replay_steps = _build_expansion(
            baseline_steps,
            horizon=horizon,
            episode_count=target_episode_count,
            discount_factor=_float_value(config.get("discount_factor"), 0.99),
        )
        replay_counters = _counters(replay_steps, replay_episodes)
        replay_summary = _replay_summary(
            replay_index=replay_index,
            horizon=horizon,
            counters=replay_counters,
        )
        replay_files = _replay_output_paths(replay_root)
        _write_jsonl(replay_files["episodes"], replay_episodes)
        _write_jsonl(replay_files["steps"], replay_steps)
        _write_json(replay_files["summary"], replay_summary)
        replay_signature = _step_signatures(replay_steps)
        comparison_row = _compare_replay(
            replay_index=replay_index,
            baseline_signature=baseline_signature,
            replay_signature=replay_signature,
            baseline_counters=counters,
            replay_counters=replay_counters,
        )
        comparison_rows.append(comparison_row)
        if comparison_row["status"] == "matched" and replay_summary["status"] == "passed":
            passed_replay_count += 1
        _add_progress(
            progress_events,
            "replay_finished",
            replay_index=replay_index,
            status=replay_summary["status"],
            reason_codes=replay_summary["reason_codes"],
            episode_count=replay_counters["episode_count"],
            step_count=replay_counters["step_count"],
            duration_seconds=round(time.perf_counter() - started_at, 6),
        )

    behavior_drift_count = sum(1 for row in comparison_rows if row["status"] != "matched")
    if behavior_drift_count:
        _add_reason(reason_codes, "baseline_replay_behavior_drift_detected")
    if passed_replay_count != replay_count:
        _add_reason(reason_codes, "horizon5_replay_not_all_passed")

    _write_jsonl(comparison_path, comparison_rows)
    _write_jsonl(progress_path, progress_events)

    status_without_readiness = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        stability_root=stability_root,
        stability_summary_path=stability_summary_path,
        acceptance_contract_path=acceptance_contract_path,
        freeze_summary_path=freeze_summary_path,
        freeze_manifest_path=freeze_manifest_path,
        baseline_steps_path=baseline_steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        episodes_path=episodes_path,
        steps_path=steps_path,
        reward_audit_path=reward_audit_path,
        rejection_report_path=rejection_report_path,
        comparison_path=comparison_path,
        progress_path=progress_path,
        readiness_path=readiness_path,
        report_path=report_path,
        config=config,
        stability_summary=stability_summary,
        counters=counters,
        replay_count=replay_count,
        passed_replay_count=passed_replay_count,
        behavior_drift_count=behavior_drift_count,
        progress_event_count=len(progress_events),
    )
    _write_json(summary_path, summary)

    readiness_runner = readiness_runner or _run_readiness_validate_only
    readiness = readiness_runner(
        repo_root=repo_root,
        batch_root=batch_root,
        horizon5_summary_path=summary_path,
        config_path=Path(config.get("readiness", {}).get("config", "configs/policy_training_readiness_review_v1.json")),
    )
    _write_json(readiness_path, readiness)
    _validate_readiness(readiness, config, reason_codes)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        stability_root=stability_root,
        stability_summary_path=stability_summary_path,
        acceptance_contract_path=acceptance_contract_path,
        freeze_summary_path=freeze_summary_path,
        freeze_manifest_path=freeze_manifest_path,
        baseline_steps_path=baseline_steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        episodes_path=episodes_path,
        steps_path=steps_path,
        reward_audit_path=reward_audit_path,
        rejection_report_path=rejection_report_path,
        comparison_path=comparison_path,
        progress_path=progress_path,
        readiness_path=readiness_path,
        report_path=report_path,
        config=config,
        stability_summary=stability_summary,
        counters=counters,
        replay_count=replay_count,
        passed_replay_count=passed_replay_count,
        behavior_drift_count=behavior_drift_count,
        progress_event_count=len(progress_events),
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(
        _render_report(summary, comparison_rows),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand quasi-real guarded PPO stability evidence to horizon=5."
    )
    parser.add_argument(
        "--stability-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/quasi_real_guarded_ppo_horizon5_batch_expansion_v1.json",
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

    summary = run_quasi_real_guarded_ppo_horizon5_batch_expansion(
        stability_root=_resolve_path(Path(args.stability_root), repo_root),
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
                "horizon": summary["horizon"],
                "episode_count": summary["episode_count"],
                "step_count": summary["step_count"],
                "ppo_trainable_transition_count": summary["ppo_trainable_transition_count"],
                "replay_count": summary["replay_count"],
                "passed_replay_count": summary["passed_replay_count"],
                "readiness_status": summary["readiness_status"],
                "controlled_regression_count": summary["controlled_regression_count"],
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
    horizon5_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-ppo-horizon5-batch-expansion-summary",
        str(horizon5_summary_path),
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
    stability_summary: dict[str, Any],
    acceptance_contract: dict[str, Any],
    freeze_summary: dict[str, Any],
    freeze_manifest: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if stability_summary.get("status") != "passed" or _string_list(stability_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_stability_replay_not_passed")
    if _int_value(stability_summary.get("replay_count")) < 3:
        _add_reason(reason_codes, "input_stability_replay_count_below_threshold")
    if _int_value(stability_summary.get("passed_replay_count")) != _int_value(stability_summary.get("replay_count")):
        _add_reason(reason_codes, "input_stability_replay_not_all_passed")
    for field, reason in (
        ("controlled_regression_count", "input_stability_controlled_regression"),
        ("baseline_replay_behavior_drift_count", "input_stability_behavior_drift"),
    ):
        if _int_value(stability_summary.get(field)):
            _add_reason(reason_codes, reason)
    if _float_value(stability_summary.get("teacher_agreement_rate")) < 0.95:
        _add_reason(reason_codes, "input_stability_teacher_alignment_insufficient")
    if stability_summary.get("long_horizon_verdict") != "long_horizon_teacher_skill_contract_aligned":
        _add_reason(reason_codes, "input_stability_long_horizon_not_aligned")
    if not acceptance_contract:
        _add_reason(reason_codes, "input_acceptance_contract_missing")
    if freeze_summary.get("status") != "passed" or _string_list(freeze_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_freeze_summary_not_passed")
    if _int_value(freeze_summary.get("required_artifact_missing_count")):
        _add_reason(reason_codes, "input_freeze_required_artifact_missing")
    if not freeze_manifest:
        _add_reason(reason_codes, "input_freeze_manifest_missing")
    elif _int_value(freeze_manifest.get("required_artifact_missing_count")):
        _add_reason(reason_codes, "input_freeze_manifest_required_artifact_missing")
    for field, reason in (
        ("runs_ppo_update", "limited_ppo_update_unexpected"),
        ("publishes_checkpoint", "limited_ppo_update_checkpoint_publication_claimed"),
        ("replaces_default_policy", "limited_ppo_update_default_policy_replacement_claimed"),
        ("performance_claimed", "limited_ppo_update_policy_performance_claimed"),
        ("formal_training_ready_claimed", "limited_ppo_update_formal_training_ready_claimed"),
    ):
        if stability_summary.get(field) is True:
            _add_reason(reason_codes, reason)


def _build_expansion(
    baseline_steps: list[dict[str, Any]],
    *,
    horizon: int,
    episode_count: int,
    discount_factor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not baseline_steps:
        return [], []
    total_steps = horizon * episode_count
    steps: list[dict[str, Any]] = []
    for ordinal in range(total_steps):
        source = copy.deepcopy(baseline_steps[ordinal % len(baseline_steps)])
        episode_index = ordinal // horizon
        step_index = ordinal % horizon
        step = {
            **source,
            "schema_version": STEP_SCHEMA_VERSION,
            "episode_id": f"quasi-real-horizon5-episode-{episode_index:04d}",
            "step_index": step_index,
            "decision_index": ordinal,
            "expansion_source_episode_id": source.get("episode_id"),
            "expansion_source_step_index": source.get("step_index"),
            "expansion_source_decision_index": source.get("decision_index"),
            "done": False,
        }
        step["gate_reason_codes"] = _string_list(step.get("gate_reason_codes"))
        step["controlled_regression_reason_codes"] = _controlled_regression_reasons(
            step.get("controlled_regression_reason_codes") or step.get("gate_reason_codes")
        )
        step["reward"] = _float_value(step.get("reward"), 0.0)
        step["rejection_reason_codes"] = _rejection_reasons(step)
        step["ppo_trainable_candidate"] = _ppo_trainable_candidate(step)
        step["ppo_trainable"] = False
        step["diagnostic_only"] = True
        steps.append(step)

    episodes: list[dict[str, Any]] = []
    for episode_index in range(episode_count):
        episode_steps = steps[episode_index * horizon : (episode_index + 1) * horizon]
        if not episode_steps:
            continue
        episode_steps[-1]["done"] = True
        episodes.append(
            {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "episode_id": episode_steps[0]["episode_id"],
                "horizon": horizon,
                "step_count": len(episode_steps),
                "splits": sorted({str(step.get("split")) for step in episode_steps}),
                "steps": episode_steps,
            }
        )
    _apply_episode_returns(episodes, discount_factor=discount_factor)
    return episodes, steps


def _apply_episode_returns(
    episodes: list[dict[str, Any]],
    *,
    discount_factor: float,
) -> None:
    for episode in episodes:
        running = 0.0
        running_finite = True
        for step in reversed(episode["steps"]):
            reward = _float_or_none(step.get("reward"))
            if reward is None or not math.isfinite(reward) or not running_finite:
                running_finite = False
                step["discounted_return"] = None
            else:
                running = reward + discount_factor * running
                step["discounted_return"] = running
            value = _float_or_none(step.get("value"))
            if value is None or not _is_finite(step.get("discounted_return")):
                step["advantage"] = None
            else:
                step["advantage"] = float(step["discounted_return"]) - value
            if not _is_finite(step.get("discounted_return")):
                step["rejection_reason_codes"] = _dedup(
                    _string_list(step.get("rejection_reason_codes")) + ["non_finite_return"]
                )
            if not _is_finite(step.get("advantage")):
                step["rejection_reason_codes"] = _dedup(
                    _string_list(step.get("rejection_reason_codes")) + ["non_finite_advantage"]
                )
            step["ppo_trainable"] = bool(
                step.get("ppo_trainable_candidate") and not step.get("rejection_reason_codes")
            )
            step["diagnostic_only"] = not step["ppo_trainable"]
        finite_returns = [
            float(step["discounted_return"])
            for step in episode["steps"]
            if _is_finite(step.get("discounted_return"))
        ]
        episode["discounted_episode_return"] = finite_returns[0] if finite_returns else None
        episode["ppo_trainable_transition_count"] = sum(
            1 for step in episode["steps"] if step["ppo_trainable"]
        )
        episode["diagnostic_transition_count"] = sum(
            1 for step in episode["steps"] if step["diagnostic_only"]
        )


def _rejection_reasons(step: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    split = str(step.get("split") or "")
    controlled_source = str(step.get("controlled_choice_source") or "")
    gate_reasons = _string_list(step.get("gate_reason_codes"))
    if split != "train":
        reasons.append("non_train_split")
    if controlled_source != "policy":
        reasons.append("controlled_choice_source_not_trainable")
    if gate_reasons:
        reasons.append("gate_reason_codes_present")
    if step.get("observation") is None or step.get("missing_observation") is True:
        reasons.append("missing_observation")
    if step.get("controlled_action_index") is None:
        reasons.append("controlled_action_index_missing")
    if _float_or_none(step.get("log_prob")) is None:
        reasons.append("missing_log_prob")
    if _float_or_none(step.get("value")) is None:
        reasons.append("missing_value")
    if not _is_finite(step.get("reward")):
        reasons.append("non_finite_reward")
    if _string_list(step.get("controlled_regression_reason_codes")):
        reasons.append("controlled_regression")
    return _dedup(reasons)


def _ppo_trainable_candidate(step: dict[str, Any]) -> bool:
    return (
        str(step.get("split") or "") == "train"
        and str(step.get("controlled_choice_source") or "") == "policy"
        and not _string_list(step.get("gate_reason_codes"))
        and not _string_list(step.get("controlled_regression_reason_codes"))
    )


def _counters(steps: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> Counter[str]:
    counters: Counter[str] = Counter()
    for step in steps:
        if step.get("ppo_trainable"):
            counters["ppo_trainable_transition_count"] += 1
            counters["trainable_transition_count"] += 1
            if step.get("split") == "validation":
                counters["validation_trainable_count"] += 1
            if step.get("split") == "test":
                counters["test_trainable_count"] += 1
            if step.get("controlled_choice_source") == "source_fallback":
                counters["source_fallback_trainable_count"] += 1
            if step.get("controlled_choice_source") == "teacher_fallback":
                counters["teacher_fallback_trainable_count"] += 1
            if step.get("gate_reason_codes"):
                counters["non_empty_gate_reason_trainable_count"] += 1
        else:
            counters["diagnostic_transition_count"] += 1
        if step.get("split") == "train":
            counters["train_split_transition_count"] += 1
        elif step.get("split") == "validation":
            counters["validation_split_transition_count"] += 1
        elif step.get("split") == "test":
            counters["test_split_transition_count"] += 1
        if step.get("observation") is None or step.get("missing_observation") is True:
            counters["missing_observation_count"] += 1
        if _float_or_none(step.get("log_prob")) is None:
            counters["missing_log_prob_count"] += 1
        if _float_or_none(step.get("value")) is None:
            counters["missing_value_count"] += 1
        if not _is_finite(step.get("reward")):
            counters["non_finite_reward_count"] += 1
        if not _is_finite(step.get("discounted_return")):
            counters["non_finite_return_count"] += 1
        if not _is_finite(step.get("advantage")):
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
    counters["episode_count"] = len(episodes)
    counters["step_count"] = len(steps)
    return counters


def _validate_expansion(
    counters: Counter[str],
    config: dict[str, Any],
    stability_summary: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    if _int_value(config.get("horizon")) != 5:
        _add_reason(reason_codes, "horizon_not_five")
    if counters["episode_count"] < _int_value(validation.get("min_episode_count"), 96):
        _add_reason(reason_codes, "horizon5_episode_count_below_threshold")
    if counters["step_count"] < _int_value(validation.get("min_step_count"), 480):
        _add_reason(reason_codes, "horizon5_step_count_below_threshold")
    if counters["ppo_trainable_transition_count"] < _int_value(
        validation.get("min_ppo_trainable_transition_count"),
        96,
    ):
        _add_reason(reason_codes, "horizon5_trainable_transition_count_below_threshold")
    for field, reason in (
        ("validation_trainable_count", "horizon5_split_leakage"),
        ("test_trainable_count", "horizon5_split_leakage"),
        ("source_fallback_trainable_count", "horizon5_fallback_trainable"),
        ("teacher_fallback_trainable_count", "horizon5_fallback_trainable"),
        ("missing_observation_count", "horizon5_contract_invalid"),
        ("missing_log_prob_count", "horizon5_contract_invalid"),
        ("missing_value_count", "horizon5_contract_invalid"),
        ("non_finite_reward_count", "horizon5_non_finite_value_detected"),
        ("non_finite_return_count", "horizon5_non_finite_value_detected"),
        ("non_finite_advantage_count", "horizon5_non_finite_value_detected"),
        ("controlled_regression_count", "horizon5_controlled_regression_detected"),
        ("controlled_safety_regression_count", "horizon5_controlled_regression_detected"),
        ("controlled_contract_regression_count", "horizon5_controlled_regression_detected"),
        ("controlled_path_risk_regression_count", "horizon5_controlled_regression_detected"),
        ("controlled_source_selection_regression_count", "horizon5_controlled_regression_detected"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)
    if _float_value(stability_summary.get("teacher_agreement_rate")) < float(
        validation.get("min_teacher_agreement_rate", 0.95)
    ):
        _add_reason(reason_codes, "horizon5_teacher_alignment_insufficient")


def _validate_readiness(
    readiness: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated")
    if readiness.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")
    if _int_value(readiness.get("returncode")) != 0:
        _add_reason(reason_codes, "readiness_validate_only_command_failed")


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    stability_root: Path,
    stability_summary_path: Path,
    acceptance_contract_path: Path,
    freeze_summary_path: Path,
    freeze_manifest_path: Path,
    baseline_steps_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    episodes_path: Path,
    steps_path: Path,
    reward_audit_path: Path,
    rejection_report_path: Path,
    comparison_path: Path,
    progress_path: Path,
    readiness_path: Path,
    report_path: Path,
    config: dict[str, Any],
    stability_summary: dict[str, Any],
    counters: Counter[str],
    replay_count: int,
    passed_replay_count: int,
    behavior_drift_count: int,
    progress_event_count: int,
    readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = readiness or {}
    trainable = counters["ppo_trainable_transition_count"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "reason_codes": list(reason_codes),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
        "stability_root": str(stability_root),
        "stability_summary": str(stability_summary_path),
        "acceptance_contract": str(acceptance_contract_path),
        "freeze_summary": str(freeze_summary_path),
        "freeze_manifest": str(freeze_manifest_path),
        "baseline_steps": str(baseline_steps_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "episodes": str(episodes_path),
        "steps": str(steps_path),
        "reward_audit": str(reward_audit_path),
        "rejection_report": str(rejection_report_path),
        "comparison": str(comparison_path),
        "progress_events": str(progress_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "horizon": _int_value(config.get("horizon"), 5),
        "discount_factor": _float_value(config.get("discount_factor"), 0.99),
        "episode_count": counters["episode_count"],
        "step_count": counters["step_count"],
        "train_split_transition_count": counters["train_split_transition_count"],
        "validation_split_transition_count": counters["validation_split_transition_count"],
        "test_split_transition_count": counters["test_split_transition_count"],
        "trainable_transition_count": trainable,
        "ppo_trainable_transition_count": trainable,
        "diagnostic_transition_count": counters["diagnostic_transition_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "teacher_agreement_rate": _float_value(stability_summary.get("teacher_agreement_rate"), 0.0),
        "quasi_real_collector_replay_status": "passed" if trainable else "failed",
        "quasi_real_collector_replay_trainable_transition_count": trainable,
        "long_horizon_verdict": stability_summary.get("long_horizon_verdict"),
        "uses_multistep_discounted_return": True,
        "not_single_step_best_action": True,
        "collector_replay_passed": bool(trainable),
        "long_horizon_teacher_skill_contract_aligned": (
            stability_summary.get("long_horizon_verdict")
            == "long_horizon_teacher_skill_contract_aligned"
        ),
        "replay_count": replay_count,
        "passed_replay_count": passed_replay_count,
        "baseline_replay_behavior_drift_count": behavior_drift_count,
        "progress_event_count": progress_event_count,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "readiness_returncode": readiness.get("returncode"),
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "experimental_checkpoint": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "formal_ppo_preflight_still_requires_at_least_512_trainable_and_multi_seed": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _reward_audit(steps: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    finite_rewards = [float(step["reward"]) for step in steps if _is_finite(step.get("reward"))]
    finite_returns = [
        float(step["discounted_return"])
        for step in steps
        if _is_finite(step.get("discounted_return"))
    ]
    finite_advantages = [
        float(step["advantage"])
        for step in steps
        if _is_finite(step.get("advantage"))
    ]
    return {
        "schema_version": REWARD_AUDIT_SCHEMA_VERSION,
        "reward_count": len(finite_rewards),
        "return_count": len(finite_returns),
        "advantage_count": len(finite_advantages),
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "min_reward": min(finite_rewards) if finite_rewards else None,
        "max_reward": max(finite_rewards) if finite_rewards else None,
        "min_discounted_return": min(finite_returns) if finite_returns else None,
        "max_discounted_return": max(finite_returns) if finite_returns else None,
        "min_advantage": min(finite_advantages) if finite_advantages else None,
        "max_advantage": max(finite_advantages) if finite_advantages else None,
        "uses_multistep_discounted_return": True,
        "not_single_step_best_action": True,
    }


def _rejection_report(steps: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    rows = [
        {
            "episode_id": step.get("episode_id"),
            "step_index": step.get("step_index"),
            "context_id": step.get("context_id"),
            "scenario_id": step.get("scenario_id"),
            "scenario_family": step.get("scenario_family"),
            "split": step.get("split"),
            "controlled_choice_source": step.get("controlled_choice_source"),
            "rejection_reason_codes": _string_list(step.get("rejection_reason_codes")),
        }
        for step in steps
        if _string_list(step.get("rejection_reason_codes"))
    ]
    reason_counts = Counter(
        reason
        for row in rows
        for reason in _string_list(row.get("rejection_reason_codes"))
    )
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "rejected_transition_count": len(rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "rows": rows,
    }


def _compare_replay(
    *,
    replay_index: int,
    baseline_signature: list[dict[str, Any]],
    replay_signature: list[dict[str, Any]],
    baseline_counters: Counter[str],
    replay_counters: Counter[str],
) -> dict[str, Any]:
    mismatches = []
    for field in (
        "episode_count",
        "step_count",
        "ppo_trainable_transition_count",
        "diagnostic_transition_count",
        "validation_trainable_count",
        "test_trainable_count",
        "source_fallback_trainable_count",
        "teacher_fallback_trainable_count",
        "missing_observation_count",
        "missing_log_prob_count",
        "missing_value_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
        "controlled_regression_count",
    ):
        if baseline_counters[field] != replay_counters[field]:
            mismatches.append(
                {
                    "field": field,
                    "baseline": baseline_counters[field],
                    "replay": replay_counters[field],
                }
            )
    step_signature_mismatch_count = 0
    if baseline_signature != replay_signature:
        step_signature_mismatch_count = _step_signature_mismatch_count(
            baseline_signature,
            replay_signature,
        )
        mismatches.append(
            {
                "field": "step_signatures",
                "mismatch_count": step_signature_mismatch_count,
            }
        )
    return {
        "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-comparison-row/v1",
        "replay_index": replay_index,
        "status": "matched" if not mismatches else "mismatched",
        "mismatch_count": len(mismatches),
        "step_signature_mismatch_count": step_signature_mismatch_count,
        "mismatches": mismatches,
    }


def _step_signatures(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": step.get("episode_id"),
            "step_index": step.get("step_index"),
            "context_id": step.get("context_id"),
            "scenario_id": step.get("scenario_id"),
            "split": step.get("split"),
            "controlled_choice_source": step.get("controlled_choice_source"),
            "ppo_trainable": step.get("ppo_trainable"),
            "controlled_action_index": step.get("controlled_action_index"),
            "raw_policy_action_index": step.get("raw_policy_action_index"),
            "teacher_action_index": step.get("teacher_action_index"),
            "gate_reason_codes": _string_list(step.get("gate_reason_codes")),
            "controlled_regression_reason_codes": _string_list(
                step.get("controlled_regression_reason_codes")
            ),
            "reward": _rounded_float(step.get("reward")),
            "discounted_return": _rounded_float(step.get("discounted_return")),
            "advantage": _rounded_float(step.get("advantage")),
        }
        for step in steps
    ]


def _step_signature_mismatch_count(
    baseline: list[dict[str, Any]],
    replay: list[dict[str, Any]],
) -> int:
    count = abs(len(baseline) - len(replay))
    for left, right in zip(baseline, replay):
        if left != right:
            count += 1
    return count


def _replay_summary(
    *,
    replay_index: int,
    horizon: int,
    counters: Counter[str],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if counters["controlled_regression_count"]:
        reason_codes.append("horizon5_controlled_regression_detected")
    if counters["non_finite_reward_count"] or counters["non_finite_return_count"] or counters["non_finite_advantage_count"]:
        reason_codes.append("horizon5_non_finite_value_detected")
    return {
        "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-replay-summary/v1",
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "replay_index": replay_index,
        "horizon": horizon,
        "episode_count": counters["episode_count"],
        "step_count": counters["step_count"],
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "diagnostic_transition_count": counters["diagnostic_transition_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
    }


def _render_report(summary: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> str:
    comparisons = "\n".join(
        f"- replay-{row['replay_index']:02d}: status={row['status']} mismatches={row['mismatch_count']}"
        for row in comparison_rows
    )
    return (
        "# Quasi-Real Guarded PPO Horizon-5 Batch Expansion\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n"
        f"- Horizon: `{summary.get('horizon')}`\n"
        f"- Episodes / steps: `{summary.get('episode_count')}` / `{summary.get('step_count')}`\n"
        f"- Trainable / diagnostic: `{summary.get('ppo_trainable_transition_count')}` / `{summary.get('diagnostic_transition_count')}`\n"
        f"- Replays passed: `{summary.get('passed_replay_count')}` / `{summary.get('replay_count')}`\n"
        f"- Controlled regression count: `{summary.get('controlled_regression_count')}`\n"
        f"- Teacher agreement rate: `{summary.get('teacher_agreement_rate')}`\n\n"
        "## Replay Comparison\n\n"
        f"{comparisons}\n\n"
        "## Scope\n\n"
        "This is a first-stage expansion audit, not formal PPO. It checks horizon=5 "
        "multi-step accounting, split isolation, replay determinism, and guarded "
        "contract stability before any larger PPO preflight.\n\n"
        "## Remaining Gate\n\n"
        "Formal PPO preflight still needs at least 512 trainable transitions and "
        "multi-seed stability evidence.\n"
    )


def _baseline_steps_path(
    *,
    freeze_summary: dict[str, Any],
    stability_summary: dict[str, Any],
    repo_root: Path,
) -> Path:
    candidates = []
    if stability_summary.get("baseline_steps"):
        candidates.append(_resolve_path(Path(str(stability_summary["baseline_steps"])), repo_root))
    pilot_root = freeze_summary.get("pilot_root")
    if pilot_root:
        candidates.append(
            _resolve_path(Path(str(pilot_root)), repo_root)
            / "quasi-real-guarded-ppo-rollout-steps.jsonl"
        )
    candidates.append(
        repo_root
        / "outputs"
        / "path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1"
        / "quasi-real-guarded-ppo-rollout-steps.jsonl"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else repo_root / "missing-baseline-steps.jsonl"


def _replay_output_paths(root: Path) -> dict[str, Path]:
    return {
        "summary": root / "quasi-real-guarded-ppo-horizon5-batch-expansion-replay-summary.json",
        "episodes": root / EPISODES_FILE,
        "steps": root / STEPS_FILE,
    }


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "episodes": EPISODES_FILE,
        "steps": STEPS_FILE,
        "reward_audit": REWARD_AUDIT_FILE,
        "rejection_report": REJECTION_REPORT_FILE,
        "comparison": COMPARISON_FILE,
        "progress_events": PROGRESS_EVENTS_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _resolve_input_path(
    value: Any,
    base_root: Path,
    repo_root: Path,
    *,
    fallback: str,
) -> Path:
    path = Path(str(value or fallback))
    if path.is_absolute():
        return path
    base_candidate = base_root / path
    if base_candidate.is_file():
        return base_candidate
    return repo_root / path


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _add_progress(
    events: list[dict[str, Any]],
    stage: str,
    *,
    replay_index: int | None,
    status: str | None = None,
    reason_codes: list[str] | None = None,
    episode_count: int | None = None,
    step_count: int | None = None,
    duration_seconds: float | None = None,
) -> None:
    events.append(
        {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-progress-event/v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "replay_index": replay_index,
            "status": status,
            "reason_codes": reason_codes or [],
            "episode_count": episode_count,
            "step_count": step_count,
            "duration_seconds": duration_seconds,
        }
    )


def _controlled_regression_reasons(value: Any) -> list[str]:
    return [reason for reason in _string_list(value) if reason in CONTROLLED_REGRESSION_REASONS]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _dedup(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rounded_float(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return round(parsed, 6) if parsed is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
