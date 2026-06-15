from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from run_coverage_driven_ppo_improvement_run import (
        _aggregate_post_metrics,
        _baseline_metric_rows,
        _comparison,
        _episodes_from_transitions,
        _install_model_explorer_path,
        _ppo_update_config,
        _read_json,
        _read_jsonl,
        _replay_updated_policy,
        _resolve_optional_path,
        _transition_record,
        _write_json,
        _write_jsonl,
    )
    from run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
    from run_policy_coverage_opportunity_margin_audit import (
        run_policy_coverage_opportunity_margin_audit,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_coverage_driven_ppo_improvement_run import (
        _aggregate_post_metrics,
        _baseline_metric_rows,
        _comparison,
        _episodes_from_transitions,
        _install_model_explorer_path,
        _ppo_update_config,
        _read_json,
        _read_jsonl,
        _replay_updated_policy,
        _resolve_optional_path,
        _transition_record,
        _write_json,
        _write_jsonl,
    )
    from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
    from scripts.run_policy_coverage_opportunity_margin_audit import (
        run_policy_coverage_opportunity_margin_audit,
    )


SUMMARY_SCHEMA_VERSION = "refined-coverage-reward-margin-tuning-summary/v1"
CONFIG_ROW_SCHEMA_VERSION = "refined-coverage-reward-margin-tuning-config-row/v1"

DEFAULT_REFINED_ROOT = "outputs/path_feedback_batch_refined_coverage_driven_ppo_improvement_run_v2"
DEFAULT_STAGE5A2_ROOT = "outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1"
DEFAULT_COVERAGE_DRIVEN_ROOT = "outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_PERFORMANCE_ROOT = "outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1"
DEFAULT_REWARD_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_refined_coverage_reward_margin_tuning_v1"

REFINED_SUMMARY_FILE = "refined-coverage-driven-ppo-improvement-run-summary.json"
REFINED_TRANSITIONS_FILE = "refined-coverage-ppo-batch/refined-trainable-transitions.jsonl"
REFINED_EPISODES_FILE = "coverage-aware-ppo-batch/ppo-rollout-episodes.jsonl"
STAGE5A2_SUMMARY_FILE = "policy-differentiating-counterfactual-coverage-rollouts-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
PERFORMANCE_METRIC_TABLE_FILE = "coverage-performance-metric-table.jsonl"
REWARD_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"
CANDIDATE_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"

SUMMARY_FILE = "refined-coverage-reward-margin-tuning-summary.json"
CONFIG_AUDIT_FILE = "tuning-config-audit.jsonl"
REJECTION_REPORT_FILE = "rejection-report.json"
REPORT_FILE = "refined-coverage-reward-margin-tuning-report.md"
CONFIG_ROOT_DIR = "tuning-configs"
BATCH_DIR = "coverage-aware-ppo-batch"
BATCH_EPISODES_FILE = "ppo-rollout-episodes.jsonl"
BATCH_TRANSITIONS_FILE = "ppo-rollout-transitions.jsonl"
BATCH_COLLECTOR_SUMMARY_FILE = "ppo-rollout-collector-summary.json"
COMPAT_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
UPDATE_SUMMARY_FILE = "coverage-driven-ppo-update-summary.json"
UPDATE_TRAINING_CURVES_FILE = "coverage-driven-ppo-training-curves.json"
UPDATE_DIAGNOSTICS_FILE = "coverage-driven-ppo-diagnostics.json"
UPDATE_CHECKPOINT_FILE = "coverage-driven-experimental-policy-candidate.pt"
UPDATE_CHECKPOINT_METADATA_FILE = "coverage-driven-experimental-policy-candidate-metadata.json"
UPDATE_CANDIDATE_SUMMARY_FILE = "coverage-driven-raw-policy-generalization-candidate-summary.json"
REPLAY_AUDIT_FILE = "coverage-driven-ppo-replay-audit.json"
METRIC_TABLE_FILE = "coverage-driven-ppo-performance-metric-table.jsonl"
STAGE5A_RERUN_ROOT = "stage5a-rerun"

TOLERANCE = 1.0e-9
TUNING_CONFIGS = (
    {"config_id": "advantage_x3", "advantage_scale": 3.0, "margin_scale": 0.0},
    {"config_id": "advantage_x5", "advantage_scale": 5.0, "margin_scale": 0.0},
    {"config_id": "advantage_x8", "advantage_scale": 8.0, "margin_scale": 0.0},
    {"config_id": "reward_margin_x5", "advantage_scale": 5.0, "margin_scale": 5.0},
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune refined coverage reward and margin strengths.")
    parser.add_argument("--refined-root", default=DEFAULT_REFINED_ROOT)
    parser.add_argument("--stage5a2-root", default=DEFAULT_STAGE5A2_ROOT)
    parser.add_argument("--coverage-driven-root", default=DEFAULT_COVERAGE_DRIVEN_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--coverage-signal-root", default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--coverage-performance-root", default=DEFAULT_PERFORMANCE_ROOT)
    parser.add_argument("--reward-refinement-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_refined_coverage_reward_margin_tuning(
        refined_root=_resolve_path(Path(args.refined_root), repo_root, repo_root),
        stage5a2_root=_resolve_path(Path(args.stage5a2_root), repo_root, repo_root),
        coverage_driven_root=_resolve_path(Path(args.coverage_driven_root), repo_root, repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root, repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root, repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root, repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root, repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root, repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root, repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "next_required_change": summary["next_required_change"],
                "best_config_id": summary["best_config_id"],
                "policy_argmax_changed_count": summary["policy_argmax_changed_count"],
                "coverage_return_improvement": summary["coverage_return_improvement"],
                "cumulative_coverage_rate_delta_improvement": summary[
                    "cumulative_coverage_rate_delta_improvement"
                ],
                "performance_claimed": summary["performance_claimed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_refined_coverage_reward_margin_tuning(
    *,
    refined_root: Path,
    stage5a2_root: Path,
    coverage_driven_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    refined_root = Path(refined_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root)
    _install_model_explorer_path(repo_root)
    paths = _paths(output_root)
    paths["config_root"].mkdir(parents=True, exist_ok=True)

    input_reasons: list[str] = []
    refined_summary = _read_json(refined_root / REFINED_SUMMARY_FILE, input_reasons, "refined_summary")
    stage5a2_summary = _read_json(Path(stage5a2_root) / STAGE5A2_SUMMARY_FILE, input_reasons, "stage5a2_summary")
    selected_summary = _read_json(
        Path(selected_candidate_root) / SELECTED_SUMMARY_FILE,
        input_reasons,
        "selected_candidate_summary",
    )
    formal_summary = _read_json(Path(formal_training_root) / FORMAL_SUMMARY_FILE, input_reasons, "formal_summary")
    replay_summary = _read_json(Path(post_training_replay_root) / REPLAY_SUMMARY_FILE, input_reasons, "replay_summary")
    signal_summary = _read_json(Path(coverage_signal_root) / SIGNAL_SUMMARY_FILE, input_reasons, "signal_summary")
    performance_summary = _read_json(
        Path(coverage_performance_root) / PERFORMANCE_SUMMARY_FILE,
        input_reasons,
        "performance_summary",
    )
    reward_summary = _read_json(Path(reward_refinement_root) / REWARD_SUMMARY_FILE, input_reasons, "reward_summary")

    refined_transitions_path = _resolve_optional_path(
        refined_summary.get("refined_trainable_transitions"),
        refined_root,
        repo_root,
    ) or refined_root / REFINED_TRANSITIONS_FILE
    refined_episodes_path = _resolve_optional_path(
        refined_summary.get("coverage_aware_batch_episodes"),
        refined_root,
        repo_root,
    ) or refined_root / REFINED_EPISODES_FILE
    overlay_path = _resolve_optional_path(
        stage5a2_summary.get("candidate_coverage_overlay"),
        Path(stage5a2_root),
        repo_root,
    ) or Path(stage5a2_root) / CANDIDATE_OVERLAY_FILE
    metric_table_path = _resolve_optional_path(
        performance_summary.get("metric_table"),
        Path(coverage_performance_root),
        repo_root,
    ) or Path(coverage_performance_root) / PERFORMANCE_METRIC_TABLE_FILE

    refined_rows = _read_jsonl(refined_transitions_path, input_reasons, "refined_trainable_transitions")
    refined_episodes = _read_jsonl(refined_episodes_path, input_reasons, "refined_episodes")
    previous_metric_rows = _read_jsonl(metric_table_path, input_reasons, "coverage_performance_metric_table")
    base_candidate_root = _base_candidate_root(refined_summary, selected_summary, repo_root)
    source_transitions = _iter_transitions(refined_episodes)
    reason_codes = list(input_reasons)
    _validate_inputs(
        refined_summary=refined_summary,
        stage5a2_summary=stage5a2_summary,
        formal_summary=formal_summary,
        replay_summary=replay_summary,
        signal_summary=signal_summary,
        reward_summary=reward_summary,
        source_transitions=source_transitions,
        refined_rows=refined_rows,
        base_candidate_root=base_candidate_root,
        reason_codes=reason_codes,
    )

    config_rows: list[dict[str, Any]] = []
    if not reason_codes:
        for config in TUNING_CONFIGS:
            config_rows.append(
                _run_tuning_config(
                    config=config,
                    source_transitions=source_transitions,
                    output_root=output_root,
                    selected_summary=selected_summary,
                    base_candidate_root=base_candidate_root,
                    previous_metric_rows=previous_metric_rows,
                    performance_summary=performance_summary,
                    reward_refinement_root=Path(reward_refinement_root),
                    coverage_signal_root=Path(coverage_signal_root),
                    coverage_performance_root=Path(coverage_performance_root),
                    overlay_path=overlay_path,
                    repo_root=repo_root,
                )
            )
    best_row = _best_config_row(config_rows)
    ppo_advantage_nonzero_count = sum(int(row.get("ppo_advantage_nonzero_count") or 0) for row in config_rows)
    acceptance_reasons = _acceptance_reasons(best_row, reason_codes)
    for reason in acceptance_reasons:
        _add_reason(reason_codes, reason)
    next_required_change = _next_required_change(reason_codes, ppo_advantage_nonzero_count, best_row)
    status = "passed" if not reason_codes else "failed"

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "refined_root": str(refined_root),
        "stage5a2_root": str(stage5a2_root),
        "coverage_driven_root": str(coverage_driven_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "tuning_config_audit": str(paths["config_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "tuning_config_count": len(TUNING_CONFIGS),
        "executed_tuning_config_count": len(config_rows),
        "best_config_id": best_row.get("config_id") if best_row else None,
        "safe_better_training_pair_count": _max_int(
            [row.get("safe_better_training_pair_count") for row in config_rows],
            default=_int(refined_summary.get("safe_better_training_pair_count")),
        ),
        "ppo_advantage_nonzero_count": ppo_advantage_nonzero_count,
        "policy_argmax_changed_count": _int((best_row or {}).get("policy_argmax_changed_count")),
        "coverage_return_improvement": _float_or_default(
            (best_row or {}).get("coverage_return_improvement"),
            0.0,
        ),
        "cumulative_coverage_rate_delta_improvement": _float_or_default(
            (best_row or {}).get("cumulative_coverage_rate_delta_improvement"),
            0.0,
        ),
        "valuable_area_covered_improvement": _float_or_default(
            (best_row or {}).get("valuable_area_covered_improvement"),
            0.0,
        ),
        "coverage_efficiency_regression": bool((best_row or {}).get("coverage_efficiency_regression")),
        "accepted_policy_activation_rate": (best_row or {}).get("accepted_policy_activation_rate"),
        "fallback_rate": (best_row or {}).get("fallback_rate"),
        "teacher_agreement_rate": (best_row or {}).get("teacher_agreement_rate"),
        "controlled_regression_count": _int((best_row or {}).get("controlled_regression_count")),
        "fallback_gain_contamination_count": _int(
            (best_row or {}).get("fallback_gain_contamination_count")
        ),
        "runs_new_ppo_update": any(row.get("ppo_update_status") == "passed" for row in config_rows),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "connects_real_executor": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    rejection_report = {
        "schema_version": "refined-coverage-reward-margin-tuning-rejection-report/v1",
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "best_config_id": summary["best_config_id"],
        "config_count": len(config_rows),
        "config_status_counts": dict(sorted(_counts(row.get("status") for row in config_rows).items())),
    }
    _write_jsonl(paths["config_audit"], config_rows)
    _write_json(paths["summary"], summary)
    _write_json(paths["rejection_report"], rejection_report)
    paths["report"].write_text(_render_report(summary, config_rows), encoding="utf-8")
    return summary


def _run_tuning_config(
    *,
    config: dict[str, Any],
    source_transitions: list[dict[str, Any]],
    output_root: Path,
    selected_summary: dict[str, Any],
    base_candidate_root: Path,
    previous_metric_rows: list[dict[str, Any]],
    performance_summary: dict[str, Any],
    reward_refinement_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    overlay_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config_id = str(config["config_id"])
    config_root = output_root / CONFIG_ROOT_DIR / config_id
    batch_root = config_root / BATCH_DIR
    batch_root.mkdir(parents=True, exist_ok=True)
    transitions = [_tuned_transition(transition, config) for transition in source_transitions]
    episodes = _episodes_from_transitions(transitions)
    _write_jsonl(batch_root / BATCH_EPISODES_FILE, episodes)
    _write_jsonl(batch_root / BATCH_TRANSITIONS_FILE, [_transition_record(i, row) for i, row in enumerate(transitions)])
    _write_json(batch_root / BATCH_COLLECTOR_SUMMARY_FILE, _collector_summary(transitions))

    update_config = _tuning_ppo_update_config(
        expected_count=len(transitions),
        selected_summary=selected_summary,
    )
    update_summary = run_limited_ppo_update_smoke(
        source_root=config_root,
        base_candidate_root=base_candidate_root,
        collector_root=batch_root,
        output_root=config_root,
        config=update_config,
        repo_root=repo_root,
    )
    replay_audit = _replay_updated_policy(
        checkpoint_path=config_root / UPDATE_CHECKPOINT_FILE,
        batch_transitions=transitions,
        update_summary=update_summary,
        repo_root=repo_root,
    )
    _write_json(config_root / REPLAY_AUDIT_FILE, replay_audit)
    baseline_rows = _baseline_metric_rows(previous_metric_rows, performance_summary)
    post_metrics = _aggregate_post_metrics(replay_audit.get("rows", []))
    metric_rows = baseline_rows + [post_metrics]
    comparison = _comparison({row["actor"]: row for row in metric_rows})
    _write_jsonl(config_root / METRIC_TABLE_FILE, metric_rows)
    compat_summary = _compat_summary(
        config_root=config_root,
        batch_root=batch_root,
        base_candidate_root=base_candidate_root,
        update_summary=update_summary,
        post_metrics=post_metrics,
        comparison=comparison,
        repo_root=repo_root,
    )
    _write_json(config_root / COMPAT_SUMMARY_FILE, compat_summary)
    stage5a_summary = run_policy_coverage_opportunity_margin_audit(
        coverage_driven_root=config_root,
        reward_refinement_root=reward_refinement_root,
        coverage_signal_root=coverage_signal_root,
        coverage_performance_root=coverage_performance_root,
        output_root=config_root / STAGE5A_RERUN_ROOT,
        repo_root=repo_root,
        candidate_coverage_overlay_path=overlay_path,
    )
    ppo_advantages = [_float_or_default(row.get("info", {}).get("ppo_advantage"), 0.0) for row in transitions]
    return {
        "schema_version": CONFIG_ROW_SCHEMA_VERSION,
        "config_id": config_id,
        "config_root": str(config_root),
        "advantage_scale": float(config["advantage_scale"]),
        "margin_scale": float(config["margin_scale"]),
        "status": _config_status(update_summary, comparison, post_metrics, stage5a_summary),
        "reason_codes": _config_reason_codes(update_summary, comparison, post_metrics, stage5a_summary),
        "safe_better_training_pair_count": len(transitions),
        "ppo_advantage_nonzero_count": sum(1 for value in ppo_advantages if abs(value) > TOLERANCE),
        "ppo_advantage_min": min(ppo_advantages) if ppo_advantages else None,
        "ppo_advantage_max": max(ppo_advantages) if ppo_advantages else None,
        "ppo_update_status": update_summary.get("status"),
        "optimizer_train_transition_count": update_summary.get("optimizer_train_transition_count", 0),
        "parameter_l2_delta": update_summary.get("parameter_l2_delta", 0.0),
        "approx_kl": update_summary.get("approx_kl"),
        "stage5a_status": stage5a_summary.get("status"),
        "stage5a_next_required_change": stage5a_summary.get("next_required_change"),
        "policy_argmax_changed_count": _int(stage5a_summary.get("policy_argmax_changed_count")),
        "coverage_return_improvement": comparison.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": comparison.get(
            "cumulative_coverage_rate_delta_improvement",
            0.0,
        ),
        "valuable_area_covered_improvement": comparison.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(comparison.get("coverage_efficiency_regression")),
        "accepted_policy_activation_rate": post_metrics.get("accepted_policy_activation_rate"),
        "fallback_rate": post_metrics.get("fallback_rate"),
        "teacher_agreement_rate": post_metrics.get("teacher_agreement_rate"),
        "controlled_regression_count": _int(post_metrics.get("controlled_regression_count")),
        "fallback_gain_contamination_count": 0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }


def _tuned_transition(transition: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(transition)
    info = result.get("info") if isinstance(result.get("info"), dict) else {}
    base_advantage = _float_or_default(info.get("counterfactual_coverage_advantage"), 0.0)
    rank_margin = _float_or_default(info.get("coverage_rank_margin"), base_advantage)
    ppo_advantage = base_advantage * float(config["advantage_scale"]) + rank_margin * float(config["margin_scale"])
    old_value = _float_or_default(result.get("value"), 0.0)
    info.update(
        {
            "old_value": old_value,
            "ppo_advantage": float(ppo_advantage),
            "ppo_return": float(old_value + ppo_advantage),
            "advantage_scale": float(config["advantage_scale"]),
            "margin_scale": float(config["margin_scale"]),
            "tuning_config_id": config["config_id"],
        }
    )
    result["info"] = info
    result["reward"] = float(ppo_advantage)
    reward_components = dict(result.get("reward_components") if isinstance(result.get("reward_components"), dict) else {})
    reward_components["tuned_ppo_advantage_signal"] = float(ppo_advantage)
    result["reward_components"] = reward_components
    return result


def _tuning_ppo_update_config(*, expected_count: int, selected_summary: dict[str, Any]) -> dict[str, Any]:
    config = _ppo_update_config(expected_count=expected_count, selected_summary=selected_summary)
    config["training"]["return_source"] = "transition_info"
    config["training"]["return_field"] = "ppo_return"
    config["training"]["advantage_field"] = "ppo_advantage"
    config["training"]["learning_rate"] = 2.0e-5
    config["training"]["epochs"] = 2
    config["non_goals"].append("does_not_claim_tuned_refined_coverage_performance_without_eval")
    return config


def _compat_summary(
    *,
    config_root: Path,
    batch_root: Path,
    base_candidate_root: Path,
    update_summary: dict[str, Any],
    post_metrics: dict[str, Any],
    comparison: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "coverage-driven-ppo-improvement-run-summary/v1",
        "generated_at": _utc_now(),
        "status": "failed",
        "reason_codes": ["tuning_config_diagnostic"],
        "next_required_change": "refined_coverage_reward_margin_tuning",
        "coverage_aware_batch_episodes": str(batch_root / BATCH_EPISODES_FILE),
        "coverage_aware_batch_transitions": str(batch_root / BATCH_TRANSITIONS_FILE),
        "coverage_aware_batch_collector_summary": str(batch_root / BATCH_COLLECTOR_SUMMARY_FILE),
        "ppo_update_summary": str(config_root / UPDATE_SUMMARY_FILE),
        "checkpoint_path": str(config_root / UPDATE_CHECKPOINT_FILE),
        "checkpoint_metadata_path": str(config_root / UPDATE_CHECKPOINT_METADATA_FILE),
        "candidate_summary_path": str(config_root / UPDATE_CANDIDATE_SUMMARY_FILE),
        "replay_audit": str(config_root / REPLAY_AUDIT_FILE),
        "performance_metric_table": str(config_root / METRIC_TABLE_FILE),
        "base_candidate_root": str(base_candidate_root),
        "optimizer_train_transition_count": update_summary.get("optimizer_train_transition_count", 0),
        "coverage_return_improvement": comparison.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": comparison.get(
            "cumulative_coverage_rate_delta_improvement",
            0.0,
        ),
        "valuable_area_covered_improvement": comparison.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(comparison.get("coverage_efficiency_regression")),
        "fallback_rate": post_metrics.get("fallback_rate"),
        "teacher_agreement_rate": post_metrics.get("teacher_agreement_rate"),
        "controlled_regression_count": post_metrics.get("controlled_regression_count", 0),
        "runs_new_ppo_update": bool(update_summary.get("status") == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _validate_inputs(
    *,
    refined_summary: dict[str, Any],
    stage5a2_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    signal_summary: dict[str, Any],
    reward_summary: dict[str, Any],
    source_transitions: list[dict[str, Any]],
    refined_rows: list[dict[str, Any]],
    base_candidate_root: Path,
    reason_codes: list[str],
) -> None:
    if refined_summary.get("next_required_change") not in {
        "tune_refined_reward_margin_or_expand_safe_better_pairs",
        "tune_refined_reward_margin",
        "expand_safe_better_pair_generation_across_families",
    }:
        _add_reason(reason_codes, "refined_v2_not_ready_for_tuning")
    if _int(refined_summary.get("safe_better_training_pair_count")) <= 0 or not source_transitions:
        _add_reason(reason_codes, "insufficient_tuning_advantage_materialization")
    if not any(_transition_advantage(row) > TOLERANCE for row in source_transitions):
        _add_reason(reason_codes, "insufficient_tuning_advantage_materialization")
    if refined_rows and not any(_float_or_default(row.get("counterfactual_coverage_advantage"), 0.0) > TOLERANCE for row in refined_rows):
        _add_reason(reason_codes, "insufficient_tuning_advantage_materialization")
    if stage5a2_summary.get("status") != "passed":
        _add_reason(reason_codes, "stage5a2_not_passed")
    for label, summary in (("formal_training", formal_summary), ("post_training_replay", replay_summary)):
        if summary and summary.get("status") != "passed":
            _add_reason(reason_codes, f"{label}_not_passed")
    if signal_summary and signal_summary.get("coverage_signal_status") != "passed":
        _add_reason(reason_codes, "coverage_signal_not_passed")
    if reward_summary and reward_summary.get("reward_refinement_status") != "passed":
        _add_reason(reason_codes, "reward_refinement_not_passed")
    if _int(refined_summary.get("controlled_regression_count")) > 0 or _int(stage5a2_summary.get("controlled_regression_count")) > 0:
        _add_reason(reason_codes, "controlled_regression_present")
    if _int(refined_summary.get("fallback_gain_contamination_count")) > 0 or _int(stage5a2_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reason_codes, "fallback_gain_contamination_present")
    if not (base_candidate_root / "experimental-hybrid-policy-candidate.pt").is_file():
        _add_reason(reason_codes, "base_candidate_checkpoint_missing")


def _acceptance_reasons(best_row: dict[str, Any] | None, existing_reasons: list[str]) -> list[str]:
    if existing_reasons:
        return []
    reasons: list[str] = []
    if best_row is None:
        return ["insufficient_tuning_advantage_materialization"]
    if _int(best_row.get("ppo_advantage_nonzero_count")) <= 0:
        _add_reason(reasons, "insufficient_tuning_advantage_materialization")
    if best_row.get("ppo_update_status") != "passed":
        _add_reason(reasons, "ppo_update_failed")
    if _int(best_row.get("policy_argmax_changed_count")) <= 0:
        _add_reason(reasons, "post_update_policy_teacher_equivalent")
    if _float_or_default(best_row.get("coverage_return_improvement"), 0.0) <= TOLERANCE:
        _add_reason(reasons, "no_coverage_return_improvement")
    if _float_or_default(best_row.get("cumulative_coverage_rate_delta_improvement"), 0.0) <= TOLERANCE:
        _add_reason(reasons, "no_cumulative_coverage_rate_delta_improvement")
    if _float_or_default(best_row.get("valuable_area_covered_improvement"), 0.0) < -TOLERANCE:
        _add_reason(reasons, "valuable_coverage_regressed")
    if best_row.get("coverage_efficiency_regression"):
        _add_reason(reasons, "coverage_efficiency_regression")
    if _float_or_default(best_row.get("fallback_rate"), 0.0) >= 0.5:
        _add_reason(reasons, "fallback_dominates")
    if _int(best_row.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_present")
    if _int(best_row.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    return reasons


def _next_required_change(
    reason_codes: list[str],
    ppo_advantage_nonzero_count: int,
    best_row: dict[str, Any] | None,
) -> str:
    if ppo_advantage_nonzero_count <= 0 or "insufficient_tuning_advantage_materialization" in reason_codes:
        return "fix_tuning_advantage_materialization"
    if not reason_codes:
        return "shadow_canary_release_performance_validation_preflight"
    if "post_update_policy_teacher_equivalent" in reason_codes or "no_coverage_return_improvement" in reason_codes:
        if best_row and _int(best_row.get("safe_better_training_pair_count")) < 128:
            return "expand_safe_better_pair_generation_across_families"
        return "tune_refined_reward_margin"
    return "tune_refined_reward_margin"


def _config_status(
    update_summary: dict[str, Any],
    comparison: dict[str, Any],
    post_metrics: dict[str, Any],
    stage5a_summary: dict[str, Any],
) -> str:
    reasons = _config_reason_codes(update_summary, comparison, post_metrics, stage5a_summary)
    return "passed" if not reasons else "failed"


def _config_reason_codes(
    update_summary: dict[str, Any],
    comparison: dict[str, Any],
    post_metrics: dict[str, Any],
    stage5a_summary: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if update_summary.get("status") != "passed":
        _add_reason(reasons, "ppo_update_failed")
    if _int(stage5a_summary.get("policy_argmax_changed_count")) <= 0:
        _add_reason(reasons, "post_update_policy_teacher_equivalent")
    if _float_or_default(comparison.get("coverage_return_improvement"), 0.0) <= TOLERANCE:
        _add_reason(reasons, "no_coverage_return_improvement")
    if _float_or_default(comparison.get("cumulative_coverage_rate_delta_improvement"), 0.0) <= TOLERANCE:
        _add_reason(reasons, "no_cumulative_coverage_rate_delta_improvement")
    if _float_or_default(post_metrics.get("fallback_rate"), 0.0) >= 0.5:
        _add_reason(reasons, "fallback_dominates")
    if _int(post_metrics.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_present")
    return reasons


def _best_config_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _float_or_default(row.get("coverage_return_improvement"), -math.inf),
            _float_or_default(row.get("cumulative_coverage_rate_delta_improvement"), -math.inf),
            _int(row.get("policy_argmax_changed_count")),
            -_float_or_default(row.get("fallback_rate"), 1.0),
        ),
    )


def _collector_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "coverage-aware-ppo-collector-summary/v1",
        "status": "passed" if transitions else "failed",
        "reason_codes": [] if transitions else ["insufficient_tuning_advantage_materialization"],
        "episode_count": len({transition.get("info", {}).get("episode_id") for transition in transitions}),
        "step_count": len(transitions),
        "ppo_trainable_transition_count": len(transitions),
        "diagnostic_transition_count": 0,
        "source_fallback_trainable_count": 0,
        "coverage_aware_reward_transition_count": len(transitions),
        "reward_component_names": sorted(
            {
                name
                for transition in transitions
                for name in (transition.get("reward_components") or {})
            }
        ),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "config_audit": output_root / CONFIG_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
        "config_root": output_root / CONFIG_ROOT_DIR,
    }


def _iter_transitions(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for episode in episodes:
        if isinstance(episode.get("transitions"), list):
            transitions.extend(row for row in episode["transitions"] if isinstance(row, dict))
    return transitions


def _base_candidate_root(refined_summary: dict[str, Any], selected_summary: dict[str, Any], repo_root: Path) -> Path:
    value = refined_summary.get("base_candidate_root") or selected_summary.get("selected_candidate_root")
    if value:
        path = Path(str(value))
        return path if path.is_absolute() else repo_root / path
    checkpoint = selected_summary.get("checkpoint_path")
    if checkpoint:
        path = Path(str(checkpoint))
        path = path if path.is_absolute() else repo_root / path
        return path.parent
    return repo_root


def _transition_advantage(transition: dict[str, Any]) -> float:
    info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
    return _float_or_default(info.get("counterfactual_coverage_advantage"), 0.0)


def _render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Refined Coverage Reward/Margin Tuning v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- best_config_id: `{summary['best_config_id']}`",
        f"- tuning_config_count: `{summary['tuning_config_count']}`",
        f"- ppo_advantage_nonzero_count: `{summary['ppo_advantage_nonzero_count']}`",
        f"- policy_argmax_changed_count: `{summary['policy_argmax_changed_count']}`",
        f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
        f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
        f"- fallback_rate: `{summary['fallback_rate']}`",
        "",
        "## Configs",
    ]
    for row in rows:
        lines.append(
            f"- `{row['config_id']}`: status=`{row['status']}`, "
            f"argmax_changed=`{row['policy_argmax_changed_count']}`, "
            f"coverage_return_improvement=`{row['coverage_return_improvement']}`"
        )
    return "\n".join(lines) + "\n"


def _resolve_path(path: Path, base: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    if (base / path).exists():
        return base / path
    return repo_root / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _max_int(values: list[Any], *, default: int = 0) -> int:
    parsed = [_int(value) for value in values if value is not None]
    return max(parsed) if parsed else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


if __name__ == "__main__":
    raise SystemExit(main())
