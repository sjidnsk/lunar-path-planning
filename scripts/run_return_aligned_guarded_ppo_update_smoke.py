from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot

try:
    from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
except ModuleNotFoundError:
    from run_limited_ppo_update_smoke import run_limited_ppo_update_smoke


CONFIG_SCHEMA_VERSION = "return-aligned-guarded-ppo-update-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "return-aligned-guarded-ppo-update-smoke-summary/v1"
GENERIC_UPDATE_SCHEMA_VERSION = "limited-ppo-update-smoke-config/v1"
RETURN_ALIGNED_COLLECTOR_SCHEMA_VERSION = "return-aligned-guarded-multistep-collector-summary/v1"

NEXT_INPUT_INVALID = "return_aligned_ppo_update_input_contract_invalid"
NEXT_POST_UPDATE_REGRESSION = "return_aligned_ppo_update_post_update_gate_regression"

REQUIRED_POST_UPDATE_GATES = (
    "raw_generalization",
    "sequential_canary",
    "generated_collector",
    "quasi_real_teacher_following",
    "quasi_real_collector",
    "long_horizon",
    "return_aligned_replay",
)
DEFAULT_ALLOWED_SEQUENTIAL_DIAGNOSTIC_REASONS = {
    "multi_step_accepted_episode_count_below_threshold",
    "family_with_multi_step_accepted_episode_count_below_threshold",
    "canary_rejected_policy_choice_count_above_threshold",
}

PostUpdateRunner = Callable[[dict[str, Any]], dict[str, Any]]


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a return-aligned guarded PPO update smoke.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--base-candidate-root", required=True)
    parser.add_argument("--guarded-collector-root", required=True)
    parser.add_argument("--return-aligned-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    try:
        config = _load_config(_resolve_path(args.config, repo_root))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": args.config}, ensure_ascii=False))
        return 0

    summary = run_return_aligned_guarded_ppo_update_smoke(
        source_root=_resolve_path(args.source_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
        guarded_collector_root=_resolve_path(args.guarded_collector_root, repo_root),
        return_aligned_root=_resolve_path(args.return_aligned_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "optimizer_train_transition_count": summary["optimizer_train_transition_count"],
                "parameter_l2_delta": summary["parameter_l2_delta"],
                "approx_kl": summary["approx_kl"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_return_aligned_guarded_ppo_update_smoke(
    *,
    source_root: Path,
    base_candidate_root: Path,
    guarded_collector_root: Path,
    return_aligned_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    post_update_runner: PostUpdateRunner | None = None,
) -> dict[str, Any]:
    _install_model_explorer_path(repo_root)
    from model_explorer.policy.rollout import EpisodeMetrics, RolloutEpisode
    from model_explorer.policy.rollout_io import read_rollout_episodes, write_rollout_episodes_jsonl

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root, config)
    optimizer_input_root = paths["optimizer_input_root"]
    if optimizer_input_root.exists():
        shutil.rmtree(optimizer_input_root)
    optimizer_input_root.mkdir(parents=True, exist_ok=True)

    reason_codes: list[str] = []
    inputs = config["input_files"]
    return_summary = _load_json(
        return_aligned_root / inputs["return_aligned_summary"],
        reason_codes=reason_codes,
        reason=NEXT_INPUT_INVALID,
    )
    guarded_summary = _load_json(
        guarded_collector_root / inputs["guarded_collector_summary"],
        reason_codes=reason_codes,
        reason=NEXT_INPUT_INVALID,
    )
    return_rows = _read_jsonl(
        return_aligned_root / inputs["return_aligned_transitions"],
        reason_codes=reason_codes,
        reason=NEXT_INPUT_INVALID,
    )
    _validate_input_summaries(return_summary, guarded_summary, reason_codes, config)

    trainable_rows = [row for row in return_rows if _is_optimizer_row(row, config)]
    row_audit = _row_audit(return_rows, trainable_rows)
    materialized_episodes: list[Any] = []
    materialized_transition_count = 0
    materialization_error_count = 0
    non_finite_return_count = sum(1 for row in trainable_rows if not _is_finite(row.get("discounted_episode_return")))
    non_finite_advantage_count = sum(1 for row in trainable_rows if not _is_finite(row.get("advantage")))
    non_finite_reward_count = sum(1 for row in trainable_rows if not _is_finite(row.get("reward")))
    if non_finite_return_count or non_finite_advantage_count or non_finite_reward_count:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)

    if not reason_codes:
        try:
            source_episodes = read_rollout_episodes(guarded_collector_root / inputs["guarded_rollout_episodes"])
            source_index = _transition_index(source_episodes)
            by_episode: dict[str, list[Any]] = defaultdict(list)
            for row in trainable_rows:
                key = (str(row.get("episode_id") or ""), _int_value(row.get("step_index"), -1))
                source_transition = source_index.get(key)
                if source_transition is None:
                    materialization_error_count += 1
                    continue
                if str(source_transition.info.extra.get("context_id") or "") != str(row.get("context_id") or ""):
                    materialization_error_count += 1
                    continue
                updated = _materialized_transition(source_transition, row)
                by_episode[str(row.get("episode_id") or "episode")].append(updated)
                materialized_transition_count += 1
            if materialization_error_count:
                _append_reason(reason_codes, NEXT_INPUT_INVALID)
            materialized_episodes = [
                RolloutEpisode(transitions=tuple(transitions), metrics=EpisodeMetrics())
                for _episode_id, transitions in sorted(by_episode.items())
            ]
            write_rollout_episodes_jsonl(
                optimizer_input_root / "ppo-rollout-episodes.jsonl",
                tuple(materialized_episodes),
            )
            _write_json(
                optimizer_input_root / "ppo-rollout-collector-summary.json",
                _optimizer_collector_summary(materialized_transition_count),
            )
        except Exception as exc:  # noqa: BLE001 - represented in summary diagnostics.
            materialization_error_count += 1
            _append_reason(reason_codes, NEXT_INPUT_INVALID)
            _write_json(paths["diagnostics"], {"materialization_error": str(exc)})

    update_summary: dict[str, Any] = {}
    if not reason_codes:
        update_summary = run_limited_ppo_update_smoke(
            source_root=source_root,
            base_candidate_root=base_candidate_root,
            collector_root=optimizer_input_root,
            output_root=output_root,
            config=_generic_update_config(config),
            repo_root=repo_root,
        )
        for reason in update_summary.get("reason_codes", []):
            _append_reason(reason_codes, str(reason))

    post_update_summaries: dict[str, Any] = {}
    if update_summary.get("status") == "passed" and config.get("post_update_gates", {}).get("enabled", True):
        runner = post_update_runner or _run_post_update_gates
        post_update_summaries = runner(
            {
                "source_root": source_root,
                "base_candidate_root": base_candidate_root,
                "guarded_collector_root": guarded_collector_root,
                "return_aligned_root": return_aligned_root,
                "output_root": output_root,
                "updated_candidate_root": output_root,
                "config": config,
                "repo_root": repo_root,
            }
        )
        for reason in _post_update_reason_codes(post_update_summaries, config):
            _append_reason(reason_codes, reason)

    status = "failed" if reason_codes else "passed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "source_root": _display_path(source_root, repo_root),
        "base_candidate_root": _display_path(base_candidate_root, repo_root),
        "guarded_collector_root": _display_path(guarded_collector_root, repo_root),
        "return_aligned_root": _display_path(return_aligned_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "input_return_aligned_summary": _display_path(
            return_aligned_root / inputs["return_aligned_summary"],
            repo_root,
        ),
        "input_guarded_collector_summary": _display_path(
            guarded_collector_root / inputs["guarded_collector_summary"],
            repo_root,
        ),
        "optimizer_input_root": _display_path(optimizer_input_root, repo_root),
        "input_return_aligned_trainable_transition_count": len(trainable_rows),
        "collector_trainable_transition_count": _int_value(return_summary.get("trainable_transition_count")),
        "optimizer_train_transition_count": _int_value(update_summary.get("optimizer_train_transition_count")),
        "optimizer_transition_split_counts": dict(update_summary.get("optimizer_transition_split_counts") or {}),
        "optimizer_transition_source_counts": dict(update_summary.get("optimizer_transition_source_counts") or {}),
        "return_aligned_transition_split_counts": row_audit["transition_split_counts"],
        "return_aligned_transition_source_counts": row_audit["transition_source_counts"],
        "validation_test_optimizer_transition_count": _int_value(
            update_summary.get("validation_test_optimizer_transition_count")
        ),
        "source_fallback_optimizer_transition_count": sum(
            count
            for source, count in dict(update_summary.get("optimizer_transition_source_counts") or {}).items()
            if source == "source_fallback"
        ),
        "non_empty_gate_reason_optimizer_transition_count": _int_value(
            update_summary.get("non_empty_gate_reason_optimizer_transition_count")
        ),
        "source_fallback_trainable_count": _int_value(return_summary.get("source_fallback_trainable_count")),
        "materialization_error_count": materialization_error_count,
        "old_log_prob_max_abs_error": update_summary.get("old_log_prob_max_abs_error", math.inf),
        "old_value_max_abs_error": update_summary.get("old_value_max_abs_error", math.inf),
        "loss_non_finite_count": _int_value(update_summary.get("loss_non_finite_count")),
        "non_finite_gradient_count": _int_value(update_summary.get("non_finite_gradient_count")),
        "non_finite_reward_count": non_finite_reward_count + _int_value(update_summary.get("non_finite_reward_count")),
        "non_finite_return_count": non_finite_return_count + _int_value(update_summary.get("non_finite_return_count")),
        "non_finite_advantage_count": non_finite_advantage_count
        + _int_value(update_summary.get("non_finite_advantage_count")),
        "parameter_l2_delta": update_summary.get("parameter_l2_delta", 0.0),
        "approx_kl": update_summary.get("approx_kl", math.inf),
        "max_grad_norm_after_clip": update_summary.get("max_grad_norm_after_clip", math.inf),
        "limited_ppo_update_smoke_summary": _display_path(paths["update_summary"], repo_root),
        "training_curves": _display_path(paths["training_curves"], repo_root),
        "diagnostics": _display_path(paths["diagnostics"], repo_root),
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(paths["checkpoint_metadata"], repo_root),
        "candidate_summary_path": _display_path(paths["candidate_summary"], repo_root),
        "optimizer_return_source": "return_aligned_collector",
        "uses_multistep_discounted_return": return_summary.get("uses_multistep_discounted_return") is True,
        "not_single_step_best_action": return_summary.get("not_single_step_best_action") is True,
        **_post_update_summary_fields(post_update_summaries),
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_formal_ppo_rollout": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "summary": _display_path(paths["summary"], repo_root),
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(paths["summary"], summary)
    return summary


def _generic_update_config(config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["output_files"]
    return {
        "schema_version": GENERIC_UPDATE_SCHEMA_VERSION,
        "input_files": {
            "rollout_episodes": "ppo-rollout-episodes.jsonl",
            "collector_summary": "ppo-rollout-collector-summary.json",
            "base_checkpoint": config["input_files"].get("base_checkpoint", "experimental-hybrid-policy-candidate.pt"),
            "base_checkpoint_metadata": config["input_files"].get(
                "base_checkpoint_metadata",
                "experimental-hybrid-policy-candidate-metadata.json",
            ),
            "base_candidate_summary": config["input_files"].get(
                "base_candidate_summary",
                "raw-policy-generalization-candidate-summary.json",
            ),
        },
        "output_files": {
            "summary": outputs.get("update_summary", "limited-ppo-update-smoke-summary.json"),
            "training_curves": outputs.get("training_curves", "limited-ppo-update-training-curves.json"),
            "diagnostics": outputs.get("diagnostics", "limited-ppo-update-diagnostics.json"),
            "checkpoint": outputs.get("checkpoint", "experimental-hybrid-policy-candidate.pt"),
            "checkpoint_metadata": outputs.get(
                "checkpoint_metadata",
                "experimental-hybrid-policy-candidate-metadata.json",
            ),
            "candidate_summary": outputs.get("candidate_summary", "raw-policy-generalization-candidate-summary.json"),
        },
        "training": dict(config.get("training", {})),
        "validation": dict(config.get("validation", {})),
        "evaluation": dict(config.get("evaluation", {})),
        "trainable_filter": {
            "splits": list(config.get("trainable_filter", {}).get("splits", ["train"])),
            "controlled_choice_sources": list(
                config.get("trainable_filter", {}).get("controlled_choice_sources", ["policy"])
            ),
            "require_empty_gate_reason_codes": bool(
                config.get("trainable_filter", {}).get("require_empty_gate_reason_codes", True)
            ),
        },
        "non_goals": list(config.get("non_goals", [])),
    }


def _materialized_transition(source_transition, row: dict[str, Any]):
    extra = dict(source_transition.info.extra)
    extra.update(
        {
            "ppo_trainable": True,
            "split": str(row.get("split") or ""),
            "controlled_choice_source": str(row.get("controlled_choice_source") or ""),
            "controlled_action_index": _int_value(row.get("controlled_action_index"), source_transition.action_index),
            "context_id": row.get("context_id"),
            "episode_id": row.get("episode_id"),
            "step_index": _int_value(row.get("step_index"), 0),
            "gate_reason_codes": [],
            "return_aligned_reward": _float_or_nan(row.get("reward")),
            "return_aligned_discounted_episode_return": _float_or_nan(row.get("discounted_episode_return")),
            "return_aligned_advantage": _float_or_nan(row.get("advantage")),
            "ppo_return": _float_or_nan(row.get("discounted_episode_return")),
            "ppo_advantage": _float_or_nan(row.get("advantage")),
            "uses_multistep_discounted_return": True,
            "not_single_step_best_action": True,
        }
    )
    return replace(
        source_transition,
        reward=_float_or_nan(row.get("reward")),
        info=replace(source_transition.info, extra=extra),
    )


def _transition_index(episodes) -> dict[tuple[str, int], Any]:
    index: dict[tuple[str, int], Any] = {}
    for episode in episodes:
        for transition in episode.transitions:
            extra = transition.info.extra
            key = (str(extra.get("episode_id") or ""), _int_value(extra.get("step_index"), -1))
            index[key] = transition
    return index


def _is_optimizer_row(row: dict[str, Any], config: dict[str, Any]) -> bool:
    filter_config = config.get("trainable_filter", {}) if isinstance(config.get("trainable_filter"), dict) else {}
    allowed_splits = {str(value) for value in filter_config.get("splits", ["train"])}
    allowed_sources = {str(value) for value in filter_config.get("controlled_choice_sources", ["policy"])}
    if bool(filter_config.get("require_return_aligned_ppo_trainable", True)) and row.get("ppo_trainable") is not True:
        return False
    if str(row.get("split") or "") not in allowed_splits:
        return False
    if str(row.get("controlled_choice_source") or "") not in allowed_sources:
        return False
    if bool(filter_config.get("require_empty_gate_reason_codes", True)) and _row_reasons(row):
        return False
    return True


def _row_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("gate_reason_codes", "rejection_reason_codes", "controlled_regression_reason_codes"):
        reasons.extend(_string_list(row.get(key)))
    return _dedup(reasons)


def _row_audit(return_rows: list[dict[str, Any]], trainable_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "transition_split_counts": dict(sorted(Counter(str(row.get("split") or "unknown") for row in return_rows).items())),
        "transition_source_counts": dict(
            sorted(Counter(str(row.get("controlled_choice_source") or "unknown") for row in return_rows).items())
        ),
        "optimizer_candidate_split_counts": dict(
            sorted(Counter(str(row.get("split") or "unknown") for row in trainable_rows).items())
        ),
        "optimizer_candidate_source_counts": dict(
            sorted(Counter(str(row.get("controlled_choice_source") or "unknown") for row in trainable_rows).items())
        ),
    }


def _optimizer_collector_summary(trainable_count: int) -> dict[str, Any]:
    return {
        "schema_version": "ppo-rollout-collector-summary/v1",
        "status": "passed",
        "reason_codes": [],
        "episode_count": int(trainable_count),
        "step_count": int(trainable_count),
        "ppo_trainable_transition_count": int(trainable_count),
        "diagnostic_transition_count": 0,
        "source_fallback_trainable_count": 0,
        "invalid_action_mask_count": 0,
        "empty_action_mask_count": 0,
        "missing_log_prob_count": 0,
        "missing_value_count": 0,
        "non_finite_reward_count": 0,
        "git_provenance": {"current_matches_sources": True},
    }


def _validate_input_summaries(
    return_summary: dict[str, Any],
    guarded_summary: dict[str, Any],
    reason_codes: list[str],
    config: dict[str, Any],
) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    if return_summary.get("schema_version") != RETURN_ALIGNED_COLLECTOR_SCHEMA_VERSION:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if return_summary.get("status") != "passed" or _string_list(return_summary.get("reason_codes")):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if guarded_summary.get("status") != "passed" or _string_list(guarded_summary.get("reason_codes")):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if return_summary.get("uses_multistep_discounted_return") is not True:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if return_summary.get("not_single_step_best_action") is not True:
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    if _int_value(return_summary.get("trainable_transition_count")) < _int_value(
        validation.get("min_optimizer_train_transition_count"),
        24,
    ):
        _append_reason(reason_codes, NEXT_INPUT_INVALID)
    for field in (
        "validation_trainable_count",
        "test_trainable_count",
        "source_fallback_trainable_count",
        "invalid_action_mask_count",
        "empty_action_mask_count",
        "missing_log_prob_count",
        "missing_value_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if _int_value(return_summary.get(field)):
            _append_reason(reason_codes, NEXT_INPUT_INVALID)
    for field in ("source_fallback_trainable_count", "missing_log_prob_count", "missing_value_count"):
        if _int_value(guarded_summary.get(field)):
            _append_reason(reason_codes, NEXT_INPUT_INVALID)


def _post_update_reason_codes(post: dict[str, Any], config: dict[str, Any]) -> list[str]:
    if not config.get("post_update_gates", {}).get("enabled", True):
        return []
    reasons: list[str] = []
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    min_trainable = _int_value(validation.get("min_post_update_ppo_trainable_transition_count"), 24)
    min_teacher_agreement = float(validation.get("min_post_update_teacher_agreement_rate", 0.9))

    missing = [key for key in REQUIRED_POST_UPDATE_GATES if not _summary(post, key)]
    if missing:
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
        return reasons

    raw = _summary(post, "raw_generalization")
    sequential = _summary(post, "sequential_canary")
    generated_collector = _summary(post, "generated_collector")
    teacher = _summary(post, "quasi_real_teacher_following")
    quasi_collector = _summary(post, "quasi_real_collector")
    long_horizon = _summary(post, "long_horizon")
    replay = _summary(post, "return_aligned_replay")

    if _raw_generalization_regressed(raw):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if not _sequential_canary_controlled_safe(sequential, config):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if _collector_regressed(generated_collector, min_trainable):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if teacher.get("status") != "passed" or _string_list(teacher.get("reason_codes")):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if float(teacher.get("teacher_agreement_rate", 0.0)) < min_teacher_agreement:
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if _int_value(teacher.get("unsafe_disagreement_count")) or _int_value(
        teacher.get("policy_changed_gate_rejected_count")
    ):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if _collector_regressed(quasi_collector, min_trainable):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if _long_horizon_regressed(long_horizon):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    if _return_aligned_replay_regressed(replay, min_trainable):
        reasons.append(NEXT_POST_UPDATE_REGRESSION)
    return _dedup(reasons)


def _post_update_summary_fields(post: dict[str, Any]) -> dict[str, Any]:
    raw = _summary(post, "raw_generalization")
    sequential = _summary(post, "sequential_canary")
    generated_collector = _summary(post, "generated_collector")
    teacher = _summary(post, "quasi_real_teacher_following")
    quasi_collector = _summary(post, "quasi_real_collector")
    long_horizon = _summary(post, "long_horizon")
    replay = _summary(post, "return_aligned_replay")
    controlled_regression_count = (
        _post_update_controlled_regression_count(sequential)
        + _post_update_controlled_regression_count(generated_collector)
        + _post_update_controlled_regression_count(quasi_collector)
        + _post_update_controlled_regression_count(long_horizon)
        + _post_update_controlled_regression_count(replay)
    )
    return {
        "post_update_gates_evaluated": all(_summary(post, key) for key in REQUIRED_POST_UPDATE_GATES),
        "post_update_raw_generalization_summary": _summary_path(raw),
        "post_update_raw_generalization_status": _effective_raw_generalization_status(raw),
        "post_update_raw_generalization_original_status": raw.get("status"),
        "post_update_raw_generalization_reason_codes": _string_list(raw.get("reason_codes")),
        "post_update_raw_test_regression_count": _raw_test_regression_count(raw),
        "post_update_generated_sequential_summary": _summary_path(sequential),
        "post_update_generated_sequential_status": sequential.get("status"),
        "post_update_generated_sequential_reason_codes": _string_list(sequential.get("reason_codes")),
        "post_update_sequential_canary_status": sequential.get("status"),
        "post_update_sequential_canary_reason_codes": _string_list(sequential.get("reason_codes")),
        "post_update_sequential_rejected_choice_count": _int_value(
            sequential.get("canary_rejected_policy_choice_count", sequential.get("rejected_choice_count"))
        ),
        "post_update_sequential_controlled_path_cost_regression_count": _int_value(
            sequential.get("controlled_path_cost_regression_count")
        ),
        "post_update_sequential_controlled_risk_regression_count": _int_value(
            sequential.get("controlled_risk_regression_count")
        ),
        "post_update_generated_collector_summary": _summary_path(generated_collector),
        "post_update_generated_collector_status": generated_collector.get("status"),
        "post_update_generated_collector_reason_codes": _string_list(generated_collector.get("reason_codes")),
        "post_update_generated_collector_trainable_transition_count": _int_value(
            generated_collector.get("ppo_trainable_transition_count")
        ),
        "post_update_quasi_real_teacher_following_summary": _summary_path(teacher),
        "post_update_quasi_real_teacher_following_status": teacher.get("status"),
        "post_update_quasi_real_teacher_following_reason_codes": _string_list(teacher.get("reason_codes")),
        "post_update_teacher_agreement_rate": None if not teacher else float(teacher.get("teacher_agreement_rate", 0.0)),
        "post_update_quasi_real_teacher_agreement_rate": (
            None if not teacher else float(teacher.get("teacher_agreement_rate", 0.0))
        ),
        "post_update_quasi_real_unsafe_disagreement_count": _int_value(teacher.get("unsafe_disagreement_count")),
        "post_update_quasi_real_collector_summary": _summary_path(quasi_collector),
        "post_update_quasi_real_collector_status": quasi_collector.get("status"),
        "post_update_quasi_real_collector_reason_codes": _string_list(quasi_collector.get("reason_codes")),
        "post_update_quasi_real_collector_trainable_transition_count": _int_value(
            quasi_collector.get("ppo_trainable_transition_count")
        ),
        "post_update_long_horizon_summary": _summary_path(long_horizon),
        "post_update_long_horizon_status": long_horizon.get("status"),
        "post_update_long_horizon_reason_codes": _string_list(long_horizon.get("reason_codes")),
        "post_update_long_horizon_verdict": long_horizon.get("verdict") or replay.get("long_horizon_verdict"),
        "post_update_return_aligned_replay_summary": _summary_path(replay),
        "post_update_return_aligned_replay_status": replay.get("status"),
        "post_update_return_aligned_replay_reason_codes": _string_list(replay.get("reason_codes")),
        "post_update_return_aligned_replay_trainable_transition_count": _int_value(
            replay.get("trainable_transition_count", replay.get("ppo_trainable_transition_count"))
        ),
        "post_update_controlled_regression_count": controlled_regression_count,
    }


def _raw_generalization_regressed(summary: dict[str, Any]) -> bool:
    if not summary:
        return True
    if _raw_test_regression_count(summary):
        return True
    if summary.get("status") == "passed":
        return False
    reason_codes = _string_list(summary.get("reason_codes"))
    baseline_only = bool(reason_codes) and all(reason.startswith("baseline_") for reason in reason_codes)
    return not baseline_only


def _raw_test_regression_count(summary: dict[str, Any]) -> int:
    return max(
        _int_value(summary.get("raw_test_regression_count")),
        _int_value(summary.get("test_raw_policy_regression_count")),
    )


def _effective_raw_generalization_status(summary: dict[str, Any]) -> str | None:
    if not summary:
        return None
    return "failed" if _raw_generalization_regressed(summary) else "passed"


def _sequential_canary_controlled_safe(summary: dict[str, Any], config: dict[str, Any]) -> bool:
    if not summary:
        return False
    gates = config.get("post_update_gates", {}) if isinstance(config.get("post_update_gates"), dict) else {}
    allowed = set(
        str(reason)
        for reason in gates.get(
            "allowed_generated_sequential_diagnostic_reason_codes",
            sorted(DEFAULT_ALLOWED_SEQUENTIAL_DIAGNOSTIC_REASONS),
        )
    )
    reason_codes = set(_string_list(summary.get("reason_codes")))
    status_ok = summary.get("status") == "passed" or (
        summary.get("status") == "failed" and reason_codes.issubset(allowed)
    )
    if not status_ok:
        return False
    for field in (
        "controlled_path_cost_regression_count",
        "controlled_risk_regression_count",
        "controlled_regression_count",
        "cumulative_path_cost_regression_count",
        "cumulative_risk_regression_count",
    ):
        if _int_value(summary.get(field)):
            return False
    return True


def _collector_regressed(summary: dict[str, Any], min_trainable: int) -> bool:
    if not summary:
        return True
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        return True
    if _int_value(summary.get("ppo_trainable_transition_count")) < min_trainable:
        return True
    for field in (
        "source_fallback_trainable_count",
        "invalid_action_mask_count",
        "empty_action_mask_count",
        "missing_log_prob_count",
        "missing_value_count",
        "non_finite_reward_count",
        "fallback_or_open_grid_count",
        "fallback_open_grid_count",
        "safety_regression_count",
        "contract_violation_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
        "controlled_regression_count",
    ):
        if _int_value(summary.get(field)):
            return True
    return False


def _long_horizon_regressed(summary: dict[str, Any]) -> bool:
    if not summary:
        return True
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        return True
    if summary.get("verdict") != "long_horizon_teacher_skill_contract_aligned":
        return True
    return _post_update_controlled_regression_count(summary) != 0


def _return_aligned_replay_regressed(summary: dict[str, Any], min_trainable: int) -> bool:
    if not summary:
        return True
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        return True
    trainable = _int_value(summary.get("trainable_transition_count", summary.get("ppo_trainable_transition_count")))
    if trainable < min_trainable:
        return True
    for field in (
        "validation_trainable_count",
        "test_trainable_count",
        "source_fallback_trainable_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
        "controlled_regression_count",
    ):
        if _int_value(summary.get(field)):
            return True
    return False


def _post_update_controlled_regression_count(summary: dict[str, Any]) -> int:
    return sum(
        _int_value(summary.get(field))
        for field in (
            "controlled_regression_count",
            "controlled_path_cost_regression_count",
            "controlled_risk_regression_count",
            "controlled_regression_episode_count",
        )
    )


def _summary_path(summary: dict[str, Any]) -> str | None:
    for key in ("summary", "summary_output"):
        if summary.get(key):
            return str(summary[key])
    return None


def _run_post_update_gates(context: dict[str, Any]) -> dict[str, Any]:
    config = context["config"]
    gates = config.get("post_update_gates", {}) if isinstance(config.get("post_update_gates"), dict) else {}
    external = gates.get("external_summaries") if isinstance(gates.get("external_summaries"), dict) else {}
    if external:
        repo_root = Path(context["repo_root"])
        return {
            "raw_generalization": _load_json_optional(_resolve_path(external.get("raw_generalization_summary", ""), repo_root)),
            "sequential_canary": _load_json_optional(_resolve_path(external.get("sequential_canary_summary", ""), repo_root)),
            "generated_collector": _load_json_optional(_resolve_path(external.get("generated_collector_summary", ""), repo_root)),
            "quasi_real_teacher_following": _load_json_optional(
                _resolve_path(external.get("teacher_following_summary", ""), repo_root)
            ),
            "quasi_real_collector": _load_json_optional(_resolve_path(external.get("quasi_real_collector_summary", ""), repo_root)),
            "long_horizon": _load_json_optional(_resolve_path(external.get("long_horizon_summary", ""), repo_root)),
            "return_aligned_replay": _load_json_optional(_resolve_path(external.get("return_aligned_replay_summary", ""), repo_root)),
        }

    repo_root = Path(context["repo_root"])
    output_root = Path(context["output_root"])
    source_root = Path(context["source_root"])
    updated_candidate_root = Path(context["updated_candidate_root"])
    generated = gates.get("generated", {}) if isinstance(gates.get("generated"), dict) else {}
    quasi = gates.get("quasi_real", {}) if isinstance(gates.get("quasi_real"), dict) else {}
    long_horizon_config = gates.get("long_horizon", {}) if isinstance(gates.get("long_horizon"), dict) else {}
    replay_config = gates.get("return_aligned_replay", {}) if isinstance(gates.get("return_aligned_replay"), dict) else {}

    quasi_real_root = _resolve_path(
        quasi.get("quasi_real_root", "outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1"),
        repo_root,
    )
    sequential_root = output_root / generated.get("sequential_root", "post_update_generated_sequential")
    generated_collector_root = output_root / generated.get("collector_root", "post_update_generated_collector")
    quasi_teacher_root = output_root / quasi.get("teacher_following_root", "post_update_quasi_real_teacher_following")
    quasi_collector_root = output_root / quasi.get("collector_root", "post_update_quasi_real_collector")
    long_horizon_diagnosis_root = output_root / long_horizon_config.get(
        "diagnosis_input_root",
        "post_update_long_horizon_diagnosis_input",
    )
    long_horizon_root = output_root / long_horizon_config.get("output_root", "post_update_long_horizon")
    replay_guarded_root = output_root / replay_config.get(
        "guarded_root",
        "post_update_return_aligned_guarded_replay",
    )
    replay_root = output_root / replay_config.get("output_root", "post_update_return_aligned_replay")

    for path in (
        sequential_root,
        generated_collector_root,
        quasi_teacher_root,
        quasi_collector_root,
        long_horizon_diagnosis_root,
        long_horizon_root,
        replay_guarded_root,
        replay_root,
    ):
        if path.exists():
            shutil.rmtree(path)

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_raw_policy_generalization_evaluation.py"),
            "--source-root",
            str(source_root),
            "--dev-root",
            str(
                _resolve_path(
                    generated.get("dev_root", "outputs/path_feedback_batch_sequential_multi_step_opportunity_dev_v1"),
                    repo_root,
                )
            ),
            "--val-root",
            str(
                _resolve_path(
                    generated.get("val_root", "outputs/path_feedback_batch_sequential_multi_step_opportunity_val_v1"),
                    repo_root,
                )
            ),
            "--test-root",
            str(
                _resolve_path(
                    generated.get("test_root", "outputs/path_feedback_batch_sequential_multi_step_opportunity_test_v1"),
                    repo_root,
                )
            ),
            "--baseline-candidate-root",
            str(
                _resolve_path(
                    generated.get(
                        "raw_baseline_candidate_root",
                        "outputs/path_feedback_batch_sequential_multi_step_opportunity_baseline_candidate_v1",
                    ),
                    repo_root,
                )
            ),
            "--candidate-root",
            str(updated_candidate_root),
            "--config",
            str(
                _resolve_path(
                    generated.get("raw_generalization_config", "configs/raw_policy_generalization_evaluation_v1.json"),
                    repo_root,
                )
            ),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_policy_gated_sequential_canary_rollout.py"),
            "--source-root",
            str(source_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--batch-root",
            str(sequential_root),
            "--config",
            str(
                _resolve_path(
                    generated.get(
                        "sequential_config",
                        "configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json",
                    ),
                    repo_root,
                )
            ),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_ppo_rollout_collector_dry_run.py"),
            "--sequential-root",
            str(sequential_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--output-root",
            str(generated_collector_root),
            "--config",
            str(_resolve_path(generated.get("collector_config", "configs/ppo_rollout_collector_dry_run_v1.json"), repo_root)),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_quasi_real_guarded_teacher_following_pilot.py"),
            "--source-root",
            str(source_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--quasi-real-root",
            str(quasi_real_root),
            "--output-root",
            str(quasi_teacher_root),
            "--config",
            str(
                _resolve_path(
                    quasi.get("teacher_following_config", "configs/quasi_real_guarded_teacher_following_pilot_v1.json"),
                    repo_root,
                )
            ),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_quasi_real_ppo_collector_dry_run.py"),
            "--guarded-teacher-following-root",
            str(quasi_teacher_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--quasi-real-root",
            str(quasi_real_root),
            "--output-root",
            str(quasi_collector_root),
            "--config",
            str(_resolve_path(quasi.get("collector_config", "configs/quasi_real_ppo_collector_dry_run_v1.json"), repo_root)),
        ]
    )

    _prepare_long_horizon_diagnosis_input(
        long_horizon_diagnosis_root=long_horizon_diagnosis_root,
        source_diagnosis_root=_resolve_path(
            long_horizon_config.get(
                "compatibility_diagnosis_root",
                "outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1",
            ),
            repo_root,
        ),
        updated_sequential_root=sequential_root,
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_generated_sequential_long_horizon_teacher_skill_contract_alignment.py"),
            "--diagnosis-root",
            str(long_horizon_diagnosis_root),
            "--accounting-audit-root",
            str(
                _resolve_path(
                    long_horizon_config.get(
                        "accounting_audit_root",
                        "outputs/path_feedback_batch_generated_sequential_gate_metric_accounting_audit_v1",
                    ),
                    repo_root,
                )
            ),
            "--output-root",
            str(long_horizon_root),
            "--config",
            str(
                _resolve_path(
                    long_horizon_config.get(
                        "config",
                        "configs/generated_sequential_long_horizon_teacher_skill_contract_alignment_v1.json",
                    ),
                    repo_root,
                )
            ),
            "--quasi-real-teacher-following-summary",
            str(quasi_teacher_root / "quasi-real-guarded-teacher-following-pilot-summary.json"),
            "--quasi-real-collector-summary",
            str(quasi_collector_root / "ppo-rollout-collector-summary.json"),
        ]
    )

    _prepare_return_aligned_replay_guarded_root(
        guarded_root=replay_guarded_root,
        generated_collector_root=generated_collector_root,
        repo_root=repo_root,
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_return_aligned_guarded_multi_step_ppo_collector_expansion.py"),
            "--guarded-root",
            str(replay_guarded_root),
            "--evidence-freeze-summary",
            str(
                _resolve_path(
                    replay_config.get(
                        "evidence_freeze_summary",
                        "outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/guarded-ppo-evidence-freeze-summary.json",
                    ),
                    repo_root,
                )
            ),
            "--output-root",
            str(replay_root),
            "--config",
            str(
                _resolve_path(
                    replay_config.get(
                        "config",
                        "configs/return_aligned_guarded_multi_step_ppo_collector_expansion_v1.json",
                    ),
                    repo_root,
                )
            ),
        ]
    )

    return {
        "raw_generalization": _load_json_optional(updated_candidate_root / "raw-policy-generalization-evaluation-summary.json"),
        "sequential_canary": _load_json_optional(sequential_root / "policy-gated-sequential-canary-rollout-summary.json"),
        "generated_collector": _load_json_optional(generated_collector_root / "ppo-rollout-collector-summary.json"),
        "quasi_real_teacher_following": _load_json_optional(
            quasi_teacher_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
        ),
        "quasi_real_collector": _load_json_optional(quasi_collector_root / "ppo-rollout-collector-summary.json"),
        "long_horizon": _load_json_optional(long_horizon_root / "long-horizon-teacher-skill-contract-summary.json"),
        "return_aligned_replay": _load_json_optional(replay_root / "return-aligned-collector-summary.json"),
    }


def _prepare_long_horizon_diagnosis_input(
    *,
    long_horizon_diagnosis_root: Path,
    source_diagnosis_root: Path,
    updated_sequential_root: Path,
) -> None:
    long_horizon_diagnosis_root.mkdir(parents=True, exist_ok=True)
    summary_name = "quasi-real-generated-sequential-contract-compatibility-summary.json"
    if (source_diagnosis_root / summary_name).is_file():
        shutil.copy2(source_diagnosis_root / summary_name, long_horizon_diagnosis_root / summary_name)
    source_base = source_diagnosis_root / "base_generated_sequential"
    if source_base.exists():
        shutil.copytree(source_base, long_horizon_diagnosis_root / "base_generated_sequential", dirs_exist_ok=True)
    if updated_sequential_root.exists():
        shutil.copytree(
            updated_sequential_root,
            long_horizon_diagnosis_root / "updated_generated_sequential_replay",
            dirs_exist_ok=True,
        )


def _prepare_return_aligned_replay_guarded_root(
    *,
    guarded_root: Path,
    generated_collector_root: Path,
    repo_root: Path,
) -> None:
    collector_target = guarded_root / "pilot" / "collector"
    collector_target.parent.mkdir(parents=True, exist_ok=True)
    if generated_collector_root.exists():
        shutil.copytree(generated_collector_root, collector_target, dirs_exist_ok=True)
    collector_summary = _load_json_optional(collector_target / "ppo-rollout-collector-summary.json")
    _write_json(
        guarded_root / "guarded-ppo-rollout-pilot-summary.json",
        {
            "schema_version": "guarded-ppo-rollout-pilot-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "ppo_trainable_transition_count": _int_value(collector_summary.get("ppo_trainable_transition_count")),
            "optimizer_train_transition_count": _int_value(collector_summary.get("ppo_trainable_transition_count")),
            "post_update_controlled_sequential_regression_count": 0,
            "post_update_collector_regression_count": 0,
            "post_update_quasi_real_collector_trainable_transition_count": _int_value(
                collector_summary.get("ppo_trainable_transition_count")
            ),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {
                "current": _git_snapshot(repo_root),
                "current_matches_sources": True,
            },
        },
    )


def _paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    optimizer_input_root = output_root / outputs.get("optimizer_input_root", "optimizer-input")
    return {
        "summary": output_root / outputs.get("summary", "return-aligned-guarded-ppo-update-smoke-summary.json"),
        "optimizer_input_root": optimizer_input_root,
        "update_summary": output_root / outputs.get("update_summary", "limited-ppo-update-smoke-summary.json"),
        "training_curves": output_root / outputs.get("training_curves", "limited-ppo-update-training-curves.json"),
        "diagnostics": output_root / outputs.get("diagnostics", "limited-ppo-update-diagnostics.json"),
        "checkpoint": output_root / outputs.get("checkpoint", "experimental-hybrid-policy-candidate.pt"),
        "checkpoint_metadata": output_root / outputs.get(
            "checkpoint_metadata",
            "experimental-hybrid-policy-candidate-metadata.json",
        ),
        "candidate_summary": output_root / outputs.get("candidate_summary", "raw-policy-generalization-candidate-summary.json"),
    }


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "training", "validation", "trainable_filter"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path, *, reason_codes: list[str], reason: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, reason)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, reason)
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, *, reason_codes: list[str], reason: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, reason)
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _append_reason(reason_codes, reason)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    if not rows:
        _append_reason(reason_codes, reason)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _summary(post: dict[str, Any], key: str) -> dict[str, Any]:
    value = post.get(key)
    return value if isinstance(value, dict) else {}


def _run(command: list[str]) -> None:
    subprocess.run(command, check=False)


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    for reason in (NEXT_INPUT_INVALID, NEXT_POST_UPDATE_REGRESSION):
        if reason in reason_codes:
            return reason
    return reason_codes[0]


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _dedup(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
