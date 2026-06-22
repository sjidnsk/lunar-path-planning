from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
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
        _policy_observation,
        _ppo_update_config,
        _read_json,
        _read_jsonl,
        _replay_updated_policy,
        _resolve_optional_path,
        _selected_base_candidate_root,
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
        _policy_observation,
        _ppo_update_config,
        _read_json,
        _read_jsonl,
        _replay_updated_policy,
        _resolve_optional_path,
        _selected_base_candidate_root,
        _transition_record,
        _write_json,
        _write_jsonl,
    )
    from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
    from scripts.run_policy_coverage_opportunity_margin_audit import (
        run_policy_coverage_opportunity_margin_audit,
    )


SUMMARY_SCHEMA_VERSION = "refined-coverage-driven-ppo-improvement-run-summary/v2"
ADVANTAGE_AUDIT_SCHEMA_VERSION = "refined-coverage-advantage-margin-audit-row/v1"

DEFAULT_STAGE5A2_ROOT = "outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1"
DEFAULT_COVERAGE_DRIVEN_ROOT = "outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_PERFORMANCE_ROOT = "outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1"
DEFAULT_REWARD_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_refined_coverage_driven_ppo_improvement_run_v2"

STAGE5A2_SUMMARY_FILE = "policy-differentiating-counterfactual-coverage-rollouts-summary.json"
COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
CANDIDATE_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
OLD_COVERAGE_DRIVEN_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
PERFORMANCE_METRIC_TABLE_FILE = "coverage-performance-metric-table.jsonl"
REWARD_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"

SUMMARY_FILE = "refined-coverage-driven-ppo-improvement-run-summary.json"
COMPAT_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
REFINED_BATCH_DIR = "refined-coverage-ppo-batch"
REFINED_TRANSITIONS_FILE = "refined-trainable-transitions.jsonl"
COMPAT_BATCH_DIR = "coverage-aware-ppo-batch"
COMPAT_EPISODES_FILE = "ppo-rollout-episodes.jsonl"
COMPAT_TRANSITIONS_FILE = "ppo-rollout-transitions.jsonl"
COMPAT_COLLECTOR_SUMMARY_FILE = "ppo-rollout-collector-summary.json"
ADVANTAGE_AUDIT_FILE = "advantage-margin-audit.jsonl"
UPDATE_SUMMARY_FILE = "coverage-driven-ppo-update-summary.json"
UPDATE_TRAINING_CURVES_FILE = "coverage-driven-ppo-training-curves.json"
UPDATE_DIAGNOSTICS_FILE = "coverage-driven-ppo-diagnostics.json"
UPDATE_CHECKPOINT_FILE = "coverage-driven-experimental-policy-candidate.pt"
UPDATE_CHECKPOINT_METADATA_FILE = "coverage-driven-experimental-policy-candidate-metadata.json"
UPDATE_CANDIDATE_SUMMARY_FILE = "coverage-driven-raw-policy-generalization-candidate-summary.json"
REPLAY_AUDIT_FILE = "coverage-driven-ppo-replay-audit.json"
METRIC_TABLE_FILE = "refined-coverage-driven-ppo-performance-metric-table.jsonl"
COMPAT_METRIC_TABLE_FILE = "coverage-driven-ppo-performance-metric-table.jsonl"
STAGE5A_RERUN_ROOT = "stage5a-rerun"
STAGE5A_RERUN_SUMMARY_FILE = "stage5a-rerun-summary.json"
REPORT_FILE = "refined-coverage-driven-ppo-improvement-run-report.md"

TOLERANCE = 1.0e-9
MAX_ALLOWED_RELATIVE_REGRESSION = 0.05
DISCOUNT_FACTOR = 0.99
COVERAGE_SIGNAL_FIELDS = ("expected_coverage_rate_delta", "information_gain", "value")
REFINED_PPO_ADVANTAGE_SCALE = 1.0
LEGACY_V1_REWARD_COMPONENT_BLOCKER = "legacy_v1_reward_components_blocked_by_canonical_v2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run refined coverage-driven PPO using Stage 5A.2 safe-better counterfactual signals."
    )
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
    summary = run_refined_coverage_driven_ppo_improvement_run(
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
                "safe_better_training_pair_count": summary["safe_better_training_pair_count"],
                "counterfactual_advantage_nonzero_count": summary[
                    "counterfactual_advantage_nonzero_count"
                ],
                "coverage_return_improvement": summary["coverage_return_improvement"],
                "policy_argmax_changed_count": summary["policy_argmax_changed_count"],
                "performance_claimed": summary["performance_claimed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_refined_coverage_driven_ppo_improvement_run(
    *,
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
    enforce_cost_risk_energy_filter: bool = True,
    ppo_advantage_scale: float = REFINED_PPO_ADVANTAGE_SCALE,
    use_transition_info_advantage: bool = False,
    ppo_learning_rate: float | None = None,
    ppo_epochs: int | None = None,
    ppo_max_approx_kl: float | None = None,
    ppo_advantage_field: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stage5a2_root = Path(stage5a2_root)
    coverage_driven_root = Path(coverage_driven_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    paths["compat_batch_root"].mkdir(parents=True, exist_ok=True)
    paths["refined_batch_root"].mkdir(parents=True, exist_ok=True)

    input_reasons: list[str] = []
    stage5a2_summary = _read_json(
        stage5a2_root / STAGE5A2_SUMMARY_FILE,
        input_reasons,
        "stage5a2_summary",
    )
    old_summary = _read_json(
        coverage_driven_root / OLD_COVERAGE_DRIVEN_SUMMARY_FILE,
        input_reasons,
        "coverage_driven_summary",
    )
    formal_summary = _read_json(
        Path(formal_training_root) / FORMAL_SUMMARY_FILE,
        input_reasons,
        "formal_training_summary",
    )
    replay_summary = _read_json(
        Path(post_training_replay_root) / REPLAY_SUMMARY_FILE,
        input_reasons,
        "post_training_replay_summary",
    )
    selected_summary = _read_json(
        Path(selected_candidate_root) / SELECTED_SUMMARY_FILE,
        input_reasons,
        "selected_candidate_summary",
    )
    signal_summary = _read_json(
        Path(coverage_signal_root) / SIGNAL_SUMMARY_FILE,
        input_reasons,
        "coverage_signal_summary",
    )
    performance_summary = _read_json(
        Path(coverage_performance_root) / PERFORMANCE_SUMMARY_FILE,
        input_reasons,
        "coverage_performance_summary",
    )
    reward_summary = _read_json(
        Path(reward_refinement_root) / REWARD_SUMMARY_FILE,
        input_reasons,
        "reward_refinement_summary",
    )

    overlay_path = _resolve_optional_path(
        stage5a2_summary.get("candidate_coverage_overlay"),
        stage5a2_root,
        repo_root,
    ) or stage5a2_root / CANDIDATE_OVERLAY_FILE
    counterfactual_path = _resolve_optional_path(
        stage5a2_summary.get("counterfactual_coverage_rollouts"),
        stage5a2_root,
        repo_root,
    ) or stage5a2_root / COUNTERFACTUAL_FILE
    old_episodes_path = _resolve_optional_path(
        old_summary.get("coverage_aware_batch_episodes"),
        coverage_driven_root,
        repo_root,
    ) or coverage_driven_root / COMPAT_BATCH_DIR / COMPAT_EPISODES_FILE
    metric_table_path = _resolve_optional_path(
        performance_summary.get("metric_table"),
        Path(coverage_performance_root),
        repo_root,
    ) or Path(coverage_performance_root) / PERFORMANCE_METRIC_TABLE_FILE

    overlay_rows = _read_jsonl(overlay_path, input_reasons, "candidate_coverage_overlay")
    counterfactual_rows = _read_jsonl(counterfactual_path, input_reasons, "counterfactual_coverage_rollouts")
    old_episodes = _read_jsonl(old_episodes_path, input_reasons, "old_coverage_aware_episodes")
    previous_metric_rows = _read_jsonl(metric_table_path, input_reasons, "coverage_performance_metric_table")

    reason_codes = list(input_reasons)
    _validate_inputs(
        stage5a2_summary=stage5a2_summary,
        old_summary=old_summary,
        formal_summary=formal_summary,
        replay_summary=replay_summary,
        selected_summary=selected_summary,
        signal_summary=signal_summary,
        reward_summary=reward_summary,
        reason_codes=reason_codes,
    )
    _add_reason(reason_codes, LEGACY_V1_REWARD_COMPONENT_BLOCKER)

    base_candidate_root = _base_candidate_root(
        old_summary=old_summary,
        selected_summary=selected_summary,
        selected_candidate_root=Path(selected_candidate_root),
        repo_root=repo_root,
    )
    old_transitions = _iter_old_transitions(old_episodes)
    old_transition_index = _transition_index(old_transitions)
    overlay_index = _rows_by_decision(overlay_rows)
    counterfactual_index = _counterfactual_index(counterfactual_rows)

    refined = _materialize_refined_batch(
        overlay_rows=overlay_rows,
        overlay_index=overlay_index,
        counterfactual_index=counterfactual_index,
        old_transition_index=old_transition_index,
        base_candidate_root=base_candidate_root,
        output_paths=paths,
        repo_root=repo_root,
        enforce_cost_risk_energy_filter=enforce_cost_risk_energy_filter,
        ppo_advantage_scale=ppo_advantage_scale,
        ppo_advantage_field=ppo_advantage_field,
    )
    for reason in refined["reason_codes"]:
        _add_reason(reason_codes, reason)

    update_summary: dict[str, Any] = {}
    if not reason_codes and refined["trainable_transition_count"] > 0:
        config = _refined_ppo_update_config(
            expected_count=refined["trainable_transition_count"],
            selected_summary=selected_summary,
            use_transition_info_advantage=use_transition_info_advantage,
            ppo_learning_rate=ppo_learning_rate,
            ppo_epochs=ppo_epochs,
            ppo_max_approx_kl=ppo_max_approx_kl,
        )
        update_summary = run_limited_ppo_update_smoke(
            source_root=coverage_driven_root,
            base_candidate_root=base_candidate_root,
            collector_root=paths["compat_batch_root"],
            output_root=output_root,
            config=config,
            repo_root=repo_root,
        )
        if update_summary.get("status") != "passed":
            _add_reason(reason_codes, "ppo_update_failed")
    elif refined["trainable_transition_count"] <= 0:
        _add_reason(reason_codes, "insufficient_refined_advantage_materialization")

    replay_audit = _replay_updated_policy(
        checkpoint_path=paths["checkpoint"],
        batch_transitions=refined["transitions"],
        update_summary=update_summary,
        repo_root=repo_root,
    )
    if update_summary and replay_audit.get("status") != "passed":
        _add_reason(reason_codes, "guard_replay_failed")

    baseline_rows = _baseline_metric_rows(previous_metric_rows, performance_summary)
    post_metrics = _aggregate_post_metrics(replay_audit.get("rows", []))
    metric_rows = baseline_rows + [post_metrics]
    metrics_by_actor = {row["actor"]: row for row in metric_rows}
    comparison = _comparison(metrics_by_actor)

    compatibility_summary = _compatibility_summary(
        paths=paths,
        reason_codes=reason_codes,
        output_root=output_root,
        coverage_driven_root=coverage_driven_root,
        base_candidate_root=base_candidate_root,
        refined=refined,
        update_summary=update_summary,
        replay_audit=replay_audit,
        post_metrics=post_metrics,
        comparison=comparison,
        repo_root=repo_root,
    )
    _write_json(paths["compat_summary"], compatibility_summary)
    _write_json(paths["replay_audit"], replay_audit)
    _write_jsonl(paths["metric_table"], metric_rows)
    _write_jsonl(paths["compat_metric_table"], metric_rows)

    stage5a_rerun_summary = _run_stage5a_rerun(
        paths=paths,
        output_root=output_root,
        reward_refinement_root=Path(reward_refinement_root),
        coverage_signal_root=Path(coverage_signal_root),
        coverage_performance_root=Path(coverage_performance_root),
        overlay_path=overlay_path,
        repo_root=repo_root,
        update_summary=update_summary,
    )
    policy_argmax_changed_count = _int(stage5a_rerun_summary.get("policy_argmax_changed_count"))
    fallback_gain_contamination_count = _fallback_gain_contamination_count(
        refined["audit_rows"],
        signal_summary,
        reward_summary,
    )
    controlled_regression_count = _controlled_regression_count(
        refined["audit_rows"],
        stage5a_rerun_summary,
        signal_summary,
        reward_summary,
    )
    acceptance_reasons = _acceptance_reason_codes(
        refined=refined,
        update_summary=update_summary,
        comparison=comparison,
        post_metrics=post_metrics,
        policy_argmax_changed_count=policy_argmax_changed_count,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    for reason in acceptance_reasons:
        _add_reason(reason_codes, reason)

    status = "passed" if not reason_codes else "failed"
    next_required_change = _next_required_change(
        reason_codes=reason_codes,
        refined=refined,
        comparison=comparison,
        policy_argmax_changed_count=policy_argmax_changed_count,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "refined_coverage_driven_ppo_improvement_status": status,
        "stage5a2_root": str(stage5a2_root),
        "coverage_driven_root": str(coverage_driven_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "reward_refinement_root": str(reward_refinement_root),
        "output_root": str(output_root),
        "legacy_v1_reward_components_read_only": True,
        "canonical_v2_training_consumer_blocked": True,
        "profile_id": reward_summary.get("profile_id") or old_summary.get("profile_id"),
        "profile_version": reward_summary.get("profile_version") or old_summary.get("profile_version"),
        "profile_hash": reward_summary.get("profile_hash") or old_summary.get("profile_hash"),
        "summary": str(paths["summary"]),
        "compatibility_summary": str(paths["compat_summary"]),
        "refined_batch_root": str(paths["refined_batch_root"]),
        "coverage_aware_batch_root": str(paths["compat_batch_root"]),
        "coverage_aware_batch_episodes": str(paths["compat_episodes"]),
        "coverage_aware_batch_transitions": str(paths["compat_transitions"]),
        "coverage_aware_batch_collector_summary": str(paths["compat_collector_summary"]),
        "refined_trainable_transitions": str(paths["refined_transitions"]),
        "advantage_margin_audit": str(paths["advantage_audit"]),
        "ppo_update_summary": str(paths["update_summary"]),
        "training_curves": str(paths["training_curves"]),
        "diagnostics": str(paths["diagnostics"]),
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_metadata_path": str(paths["checkpoint_metadata"]),
        "candidate_summary_path": str(paths["candidate_summary"]),
        "guard_replay_audit": str(paths["replay_audit"]),
        "performance_metric_table": str(paths["metric_table"]),
        "stage5a_rerun_root": str(paths["stage5a_rerun_root"]),
        "stage5a_rerun_summary": str(paths["stage5a_rerun_summary"]),
        "report": str(paths["report"]),
        "base_candidate_root": str(base_candidate_root),
        "old_coverage_aware_batch_episodes": str(old_episodes_path),
        "candidate_coverage_overlay": str(overlay_path),
        "counterfactual_coverage_rollouts": str(counterfactual_path),
        "refined_trainable_transition_count": refined["trainable_transition_count"],
        "safe_better_training_pair_count": refined["safe_better_training_pair_count"],
        "counterfactual_advantage_nonzero_count": refined["counterfactual_advantage_nonzero_count"],
        "policy_argmax_changed_count": policy_argmax_changed_count,
        "coverage_return_improvement": comparison.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": comparison.get(
            "cumulative_coverage_rate_delta_improvement",
            0.0,
        ),
        "valuable_area_covered_improvement": comparison.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(comparison.get("coverage_efficiency_regression")),
        "fallback_rate": post_metrics.get("fallback_rate"),
        "teacher_agreement_rate": post_metrics.get("teacher_agreement_rate"),
        "controlled_regression_count": controlled_regression_count,
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "accepted_policy_activation_rate": post_metrics.get("accepted_policy_activation_rate"),
        "post_improvement_metrics": post_metrics,
        "best_baseline_actor": comparison.get("best_baseline_actor"),
        "baseline_actors": comparison.get("baseline_actors", []),
        "ppo_update_status": update_summary.get("status"),
        "input_ppo_trainable_transition_count": update_summary.get(
            "input_ppo_trainable_transition_count",
            0,
        ),
        "optimizer_train_transition_count": update_summary.get("optimizer_train_transition_count", 0),
        "old_log_prob_max_abs_error": update_summary.get("old_log_prob_max_abs_error"),
        "old_value_max_abs_error": update_summary.get("old_value_max_abs_error"),
        "parameter_l2_delta": update_summary.get("parameter_l2_delta", 0.0),
        "approx_kl": update_summary.get("approx_kl"),
        "stage5a_rerun_status": stage5a_rerun_summary.get("status"),
        "stage5a_rerun_next_required_change": stage5a_rerun_summary.get("next_required_change"),
        "experimental_checkpoint": bool(update_summary.get("experimental_checkpoint")),
        "runs_new_ppo_update": bool(update_summary.get("status") == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "connects_real_executor": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "enforce_cost_risk_energy_filter": enforce_cost_risk_energy_filter,
        "ppo_advantage_scale": ppo_advantage_scale,
        "ppo_advantage_field": ppo_advantage_field,
        "use_transition_info_advantage": use_transition_info_advantage,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    _write_json(paths["summary"], summary)
    _write_json(paths["compat_summary"], {**compatibility_summary, **_compatibility_summary_updates(summary)})
    paths["report"].write_text(_render_report(summary, refined, stage5a_rerun_summary), encoding="utf-8")
    return summary


def _paths(output_root: Path) -> dict[str, Path]:
    refined_batch_root = output_root / REFINED_BATCH_DIR
    compat_batch_root = output_root / COMPAT_BATCH_DIR
    stage5a_rerun_root = output_root / STAGE5A_RERUN_ROOT
    return {
        "summary": output_root / SUMMARY_FILE,
        "compat_summary": output_root / COMPAT_SUMMARY_FILE,
        "refined_batch_root": refined_batch_root,
        "refined_transitions": refined_batch_root / REFINED_TRANSITIONS_FILE,
        "compat_batch_root": compat_batch_root,
        "compat_episodes": compat_batch_root / COMPAT_EPISODES_FILE,
        "compat_transitions": compat_batch_root / COMPAT_TRANSITIONS_FILE,
        "compat_collector_summary": compat_batch_root / COMPAT_COLLECTOR_SUMMARY_FILE,
        "advantage_audit": output_root / ADVANTAGE_AUDIT_FILE,
        "update_summary": output_root / UPDATE_SUMMARY_FILE,
        "training_curves": output_root / UPDATE_TRAINING_CURVES_FILE,
        "diagnostics": output_root / UPDATE_DIAGNOSTICS_FILE,
        "checkpoint": output_root / UPDATE_CHECKPOINT_FILE,
        "checkpoint_metadata": output_root / UPDATE_CHECKPOINT_METADATA_FILE,
        "candidate_summary": output_root / UPDATE_CANDIDATE_SUMMARY_FILE,
        "replay_audit": output_root / REPLAY_AUDIT_FILE,
        "metric_table": output_root / METRIC_TABLE_FILE,
        "compat_metric_table": output_root / COMPAT_METRIC_TABLE_FILE,
        "stage5a_rerun_root": stage5a_rerun_root,
        "stage5a_rerun_summary": output_root / STAGE5A_RERUN_SUMMARY_FILE,
        "report": output_root / REPORT_FILE,
    }


def _validate_inputs(
    *,
    stage5a2_summary: dict[str, Any],
    old_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    signal_summary: dict[str, Any],
    reward_summary: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if stage5a2_summary.get("next_required_change") not in {
        "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
        "rerun_coverage_driven_ppo_with_refined_reward",
        "rerun_coverage_driven_ppo_with_refined_advantage",
    }:
        _add_reason(reason_codes, "stage5a2_not_ready_for_refined_ppo")
    if _int(stage5a2_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reason_codes, "fallback_gain_contamination_present")
    if _int(stage5a2_summary.get("controlled_regression_count")) > 0:
        _add_reason(reason_codes, "controlled_regression_present")
    if not old_summary:
        _add_reason(reason_codes, "old_coverage_driven_summary_missing")
    for label, summary in (
        ("formal_training", formal_summary),
        ("post_training_replay", replay_summary),
        ("selected_candidate", selected_summary),
    ):
        if summary and summary.get("status") != "passed":
            _add_reason(reason_codes, f"{label}_not_passed")
        if _int(summary.get("controlled_regression_count")) > 0:
            _add_reason(reason_codes, "controlled_regression_present")
    if signal_summary and signal_summary.get("coverage_signal_status") != "passed":
        _add_reason(reason_codes, "coverage_signal_not_passed")
    if _int(signal_summary.get("fallback_coverage_gain_claimed_as_policy_gain_count")) > 0:
        _add_reason(reason_codes, "fallback_gain_contamination_present")
    if reward_summary and reward_summary.get("reward_refinement_status") != "passed":
        _add_reason(reason_codes, "reward_refinement_not_passed")
    if _int(reward_summary.get("source_field_missing_component_count")) > 0:
        _add_reason(reason_codes, "reward_refinement_source_fields_missing")


def _base_candidate_root(
    *,
    old_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    selected_candidate_root: Path,
    repo_root: Path,
) -> Path:
    root = _resolve_optional_path(old_summary.get("base_candidate_root"), repo_root, repo_root)
    if root is not None:
        return root
    return _selected_base_candidate_root(selected_summary, selected_candidate_root, repo_root)


def _materialize_refined_batch(
    *,
    overlay_rows: list[dict[str, Any]],
    overlay_index: dict[tuple[str, str, int], list[dict[str, Any]]],
    counterfactual_index: dict[tuple[str, str, int, int], dict[str, Any]],
    old_transition_index: dict[tuple[str, str, int], dict[str, Any]],
    base_candidate_root: Path,
    output_paths: dict[str, Path],
    repo_root: Path,
    enforce_cost_risk_energy_filter: bool = True,
    ppo_advantage_scale: float = REFINED_PPO_ADVANTAGE_SCALE,
    ppo_advantage_field: str | None = None,
) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    refined_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    candidate_rows = [
        row
        for row in overlay_rows
        if not bool(row.get("is_teacher_action")) and _first_int(row.get("action_index")) is not None
    ]

    for audit_index, candidate in enumerate(candidate_rows):
        decision_key = _decision_key(candidate)
        action_index = _first_int(candidate.get("action_index"))
        teacher_action_index = _first_int(candidate.get("teacher_action_index"))
        old_transition = old_transition_index.get(decision_key)
        teacher = _teacher_row(overlay_index.get(decision_key, []), teacher_action_index)
        counterfactual = counterfactual_index.get((*decision_key, action_index if action_index is not None else -1))
        row_reasons = _candidate_rejection_reasons(
            candidate=candidate,
            teacher=teacher,
            counterfactual=counterfactual,
            old_transition=old_transition,
            enforce_cost_risk_energy_filter=enforce_cost_risk_energy_filter,
        )
        if row_reasons:
            audit_rows.append(
                _audit_row(
                    audit_index=audit_index,
                    candidate=candidate,
                    teacher=teacher,
                    counterfactual=counterfactual,
                    row_reason_codes=row_reasons,
                )
            )
            continue

        assert old_transition is not None
        assert teacher is not None
        assert action_index is not None
        advantage = _coverage_decision_score(candidate) - _coverage_decision_score(teacher)
        ppo_advantage_signal = _ppo_advantage_signal(candidate, advantage, ppo_advantage_field)
        margin = advantage
        transition = _refined_transition(
            old_transition=old_transition,
            candidate=candidate,
            teacher=teacher,
            counterfactual=counterfactual or candidate,
            action_index=action_index,
            advantage=advantage,
            ppo_advantage_signal=ppo_advantage_signal,
            coverage_rank_margin=margin,
            base_candidate_root=base_candidate_root,
            repo_root=repo_root,
            ppo_advantage_scale=ppo_advantage_scale,
        )
        transitions.append(transition)
        refined_row = _refined_transition_record(len(refined_rows), transition, candidate, teacher, advantage, margin)
        refined_rows.append(refined_row)
        audit_rows.append(
            _audit_row(
                audit_index=audit_index,
                candidate=candidate,
                teacher=teacher,
                counterfactual=counterfactual,
                row_reason_codes=[],
                counterfactual_coverage_advantage=advantage,
                coverage_rank_margin=margin,
                teacher_margin_target=transition["info"]["teacher_margin_target"],
            )
        )

    if not transitions:
        _add_reason(reason_codes, "insufficient_refined_advantage_materialization")
    if transitions and not any(abs(row["counterfactual_coverage_advantage"]) > TOLERANCE for row in refined_rows):
        _add_reason(reason_codes, "insufficient_refined_advantage_materialization")

    episodes = _episodes_from_transitions(transitions)
    transition_records = [_transition_record(index, transition) for index, transition in enumerate(transitions)]
    collector_summary = _collector_summary(transitions=transitions, rejected_count=len(candidate_rows) - len(transitions))
    _write_jsonl(output_paths["compat_episodes"], episodes)
    _write_jsonl(output_paths["compat_transitions"], transition_records)
    _write_json(output_paths["compat_collector_summary"], collector_summary)
    _write_jsonl(output_paths["refined_transitions"], refined_rows)
    _write_jsonl(output_paths["advantage_audit"], audit_rows)
    return {
        "candidate_row_count": len(candidate_rows),
        "safe_better_training_pair_count": len(transitions),
        "trainable_transition_count": len(transitions),
        "counterfactual_advantage_nonzero_count": sum(
            1 for row in refined_rows if abs(row["counterfactual_coverage_advantage"]) > TOLERANCE
        ),
        "transitions": transitions,
        "refined_rows": refined_rows,
        "audit_rows": audit_rows,
        "collector_summary": collector_summary,
        "reason_codes": reason_codes,
    }


def _candidate_rejection_reasons(
    *,
    candidate: dict[str, Any],
    teacher: dict[str, Any] | None,
    counterfactual: dict[str, Any] | None,
    old_transition: dict[str, Any] | None,
    enforce_cost_risk_energy_filter: bool = True,
) -> list[str]:
    reasons: list[str] = []
    if not bool(candidate.get("safe_better_than_teacher_candidate")):
        reasons.append("not_safe_better_than_teacher")
    if str(candidate.get("split") or "") != "train":
        reasons.append("non_train_split")
    if not bool(candidate.get("coverage_source_available")):
        reasons.append("candidate_coverage_source_missing")
    if not bool(candidate.get("action_mask_valid", True)):
        reasons.append("action_mask_invalid")
    if bool(candidate.get("fallback_like")):
        reasons.append("fallback_like_candidate")
    if bool(candidate.get("guard_rejected")):
        reasons.append("guard_rejected_candidate")
    if _string_list(candidate.get("controlled_regression_reason_codes")):
        reasons.append("controlled_regression_present")
    if _string_list(candidate.get("missing_reason_codes")):
        reasons.append("counterfactual_evidence_missing")
    if teacher is None:
        reasons.append("teacher_reference_missing")
    if counterfactual is None:
        reasons.append("counterfactual_row_missing")
    elif not bool(counterfactual.get("coverage_source_available")):
        reasons.append("counterfactual_coverage_source_missing")
    if old_transition is None:
        reasons.append("old_transition_missing")
    if enforce_cost_risk_energy_filter and teacher is not None and not _safe_cost_risk_energy(candidate, teacher):
        reasons.append("material_cost_risk_energy_regression")
    if teacher is not None and _coverage_decision_score(candidate) <= _coverage_decision_score(teacher) + TOLERANCE:
        reasons.append("coverage_advantage_not_positive")
    return _unique(reasons)


def _refined_transition(
    *,
    old_transition: dict[str, Any],
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    counterfactual: dict[str, Any],
    action_index: int,
    advantage: float,
    coverage_rank_margin: float,
    base_candidate_root: Path,
    repo_root: Path,
    ppo_advantage_scale: float = REFINED_PPO_ADVANTAGE_SCALE,
    ppo_advantage_signal: float | None = None,
) -> dict[str, Any]:
    transition = copy.deepcopy(old_transition)
    observation = _observation_with_candidate_overlay(
        transition.get("observation") if isinstance(transition.get("observation"), dict) else {},
        [teacher, candidate],
    )
    log_prob, value = _policy_log_prob_and_value(
        observation_payload=observation,
        action_index=action_index,
        base_candidate_root=base_candidate_root,
        repo_root=repo_root,
    )
    reward_components = _reward_components(candidate, advantage, coverage_rank_margin)
    extra_reward_components = candidate.get("cost_efficiency_reward_components")
    if isinstance(extra_reward_components, dict):
        for name, component_value in extra_reward_components.items():
            reward_components[str(name)] = _float_or_default(component_value, 0.0)
    reward = round(sum(float(value) for value in reward_components.values()), 12)
    expanded_family_weight = _float_or_default(candidate.get("expanded_family_weight"), 1.0)
    advantage_for_ppo = float(ppo_advantage_signal if ppo_advantage_signal is not None else advantage)
    ppo_advantage = advantage_for_ppo * max(expanded_family_weight, 0.0) * float(ppo_advantage_scale)
    info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
    info = {
        **info,
        "selected_cell": candidate.get("candidate_cell"),
        "coverage_rate_delta": _float_or_default(
            counterfactual.get("expected_coverage_rate_delta"),
            _float_or_default(candidate.get("expected_coverage_rate_delta"), 0.0),
        ),
        "cumulative_coverage_rate_delta": _float_or_default(
            counterfactual.get("expected_coverage_rate_delta"),
            _float_or_default(candidate.get("expected_coverage_rate_delta"), 0.0),
        ),
        "final_coverage_rate": _float_or_default(
            counterfactual.get("expected_coverage_rate_delta"),
            _float_or_default(candidate.get("expected_coverage_rate_delta"), 0.0),
        ),
        "new_area_covered": _float_or_default(
            counterfactual.get("expected_new_coverage_area"),
            _float_or_default(candidate.get("expected_new_coverage_area"), 0.0),
        ),
        "valuable_area_covered": _first_float(
            counterfactual.get("valuable_coverage_proxy"),
            counterfactual.get("value"),
            candidate.get("valuable_coverage_proxy"),
            candidate.get("value"),
        )
        or 0.0,
        "information_gain": _float_or_default(
            counterfactual.get("information_gain"),
            _float_or_default(candidate.get("information_gain"), 0.0),
        ),
        "path_cost": _float_or_default(candidate.get("path_cost"), 0.0),
        "risk": _float_or_default(candidate.get("risk"), 0.0),
        "energy_cost": _float_or_default(candidate.get("energy_cost"), 0.0),
        "total_cost": _float_or_default(candidate.get("path_cost"), 0.0),
        "failure_reason": None,
        "failure_count": 0,
        "replan_count": 0,
        "ppo_trainable": False,
        "legacy_read_only": True,
        "canonical_v2_training_consumer_blocked": True,
        "controlled_choice_source": "policy",
        "controlled_choice_detail": "policy_safe_better_counterfactual_refined_reward",
        "controlled_action_index": action_index,
        "teacher_action_index": _first_int(candidate.get("teacher_action_index")),
        "context_id": str(candidate.get("context_id") or info.get("context_id") or ""),
        "episode_id": str(candidate.get("episode_id") or info.get("episode_id") or ""),
        "source_episode_id": str(info.get("source_episode_id") or info.get("episode_id") or candidate.get("episode_id") or ""),
        "step_index": _int(candidate.get("step_index")),
        "source_step_index": _int(info.get("source_step_index", candidate.get("step_index"))),
        "split": "train",
        "gate_reason_codes": [LEGACY_V1_REWARD_COMPONENT_BLOCKER],
        "scenario_id": candidate.get("scenario_id") or info.get("scenario_id"),
        "scenario_family": candidate.get("scenario_family") or info.get("scenario_family"),
        "fallback_like": False,
        "guard_rejected": False,
        "controlled_regression_reason_codes": [],
        "counterfactual_coverage_advantage": float(advantage),
        "coverage_rank_margin": float(coverage_rank_margin),
        "teacher_margin_target": max(float(coverage_rank_margin), 0.01),
        "ppo_advantage": float(ppo_advantage),
        "ppo_return": float(value + ppo_advantage),
        "ppo_advantage_signal": float(advantage_for_ppo),
        "expanded_family_weight": expanded_family_weight,
            "ppo_advantage_scale": float(ppo_advantage_scale),
            "legacy_read_only": True,
            "canonical_v2_training_consumer_blocked": True,
            "legacy_blocker_reason": LEGACY_V1_REWARD_COMPONENT_BLOCKER,
            "counterfactual_coverage_source_path": counterfactual.get("source_path"),
        "counterfactual_coverage_match_method": counterfactual.get("match_method") or candidate.get("match_method"),
        "counterfactual_coverage_source_confidence": _float_or_default(
            counterfactual.get("source_confidence"),
            _float_or_default(candidate.get("source_confidence"), 0.0),
        ),
        "actual_coverage_gain_source": counterfactual.get("match_method") or candidate.get("match_method"),
        "source_values": {
            "counterfactual_coverage_advantage": float(advantage),
            "coverage_rank_margin": float(coverage_rank_margin),
            "ppo_advantage_signal": float(advantage_for_ppo),
            "expanded_family_weight": expanded_family_weight,
            "ppo_advantage_scale": float(ppo_advantage_scale),
            "candidate_expected_coverage_rate_delta": candidate.get("expected_coverage_rate_delta"),
            "teacher_expected_coverage_rate_delta": teacher.get("expected_coverage_rate_delta"),
            "candidate_information_gain": candidate.get("information_gain"),
            "teacher_information_gain": teacher.get("information_gain"),
            "candidate_value": candidate.get("value"),
            "teacher_value": teacher.get("value"),
        },
    }
    transition.update(
        {
            "observation": observation,
            "action_index": action_index,
            "action_mask": list(observation.get("action_mask", [])),
            "log_prob": float(log_prob),
            "value": float(value),
            "ppo_advantage": float(ppo_advantage),
            "ppo_return": float(value + ppo_advantage),
            "reward": float(reward),
            "reward_components": reward_components,
            "ppo_trainable": False,
            "legacy_read_only": True,
            "canonical_v2_training_consumer_blocked": True,
            "done": bool(transition.get("done", True)),
            "next_observation": None,
            "info": info,
        }
    )
    return transition


def _observation_with_candidate_overlay(
    observation: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = copy.deepcopy(observation)
    names = [str(name) for name in result.get("candidate_feature_names", [])]
    features = [list(row) for row in result.get("candidate_features", []) if isinstance(row, list)]
    missing_names = [str(name) for name in result.get("candidate_missing_indicator_names", [])]
    missing_indicators = [
        list(row) for row in result.get("candidate_missing_indicators", []) if isinstance(row, list)
    ]
    while len(missing_indicators) < len(features):
        missing_indicators.append([0.0 for _ in missing_names])
    feature_aliases = {
        "expected_coverage_rate_delta": "expected_coverage_rate_delta",
        "expected_new_coverage_area": "expected_new_coverage_area",
        "information_gain": "information_gain",
        "value": "value",
        "risk": "risk",
        "path_cost": "path_cost",
        "energy_cost": "energy_cost",
    }
    for row in rows:
        action_index = _first_int(row.get("action_index"))
        if action_index is None or action_index < 0 or action_index >= len(features):
            continue
        for feature_name, source_field in feature_aliases.items():
            if feature_name not in names:
                continue
            value = _first_float(row.get(source_field))
            if value is None:
                continue
            features[action_index][names.index(feature_name)] = float(value)
            missing_name = f"{feature_name}_missing"
            if missing_name in missing_names and action_index < len(missing_indicators):
                missing_indicators[action_index][missing_names.index(missing_name)] = 0.0
    result["candidate_features"] = features
    result["candidate_missing_indicators"] = missing_indicators
    return result


def _policy_log_prob_and_value(
    *,
    observation_payload: dict[str, Any],
    action_index: int,
    base_candidate_root: Path,
    repo_root: Path,
) -> tuple[float, float]:
    _install_model_explorer_path(repo_root)
    import torch
    from model_explorer.policy.architectures import build_policy_network_from_metadata
    from model_explorer.policy.features import PolicyObservation
    from model_explorer.policy.torch_policy import observation_to_tensors

    checkpoint_path = base_candidate_root / "experimental-hybrid-policy-candidate.pt"
    metadata_path = base_candidate_root / "experimental-hybrid-policy-candidate-metadata.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observation = _policy_observation(observation_payload, PolicyObservation)
    network = build_policy_network_from_metadata(
        checkpoint.get("architecture") or metadata.get("architecture"),
        candidate_feature_count=len(observation.candidate_feature_names),
        global_feature_count=len(observation.global_feature_names),
        missing_indicator_count=len(observation.candidate_missing_indicator_names),
        hidden_size=_int(
            checkpoint.get("training", {}).get("hidden_size")
            or checkpoint.get("hidden_size")
            or metadata.get("hidden_size"),
            16,
        ),
        architecture_config=checkpoint.get("architecture_config") or metadata.get("architecture_config"),
    )
    network.load_state_dict(checkpoint.get("model_state_dict") or checkpoint.get("state_dict"))
    network.eval()
    with torch.no_grad():
        output = network(**observation_to_tensors(observation))
        distribution = torch.distributions.Categorical(logits=output.masked_logits)
        action = torch.tensor([action_index], dtype=torch.long)
        log_prob = float(distribution.log_prob(action).item())
        value = float(output.value[0].item())
    return log_prob, value


def _reward_components(
    candidate: dict[str, Any],
    advantage: float,
    coverage_rank_margin: float,
) -> dict[str, float]:
    coverage = _float_or_default(candidate.get("expected_coverage_rate_delta"), 0.0)
    valuable = _first_float(candidate.get("valuable_coverage_proxy"), candidate.get("value")) or 0.0
    information = _float_or_default(candidate.get("information_gain"), 0.0)
    path_cost = _float_or_default(candidate.get("path_cost"), 0.0)
    risk = _float_or_default(candidate.get("risk"), 0.0)
    energy = _float_or_default(candidate.get("energy_cost"), 0.0)
    expanded_family_weight = _float_or_default(candidate.get("expanded_family_weight"), 1.0)
    return {
        "coverage_gain_bonus": coverage,
        "valuable_area_bonus": valuable,
        "information_gain_bonus": information,
        "counterfactual_coverage_advantage_bonus": float(advantage),
        "coverage_rank_margin_bonus": float(coverage_rank_margin),
        "expanded_family_weight_bonus": float(coverage_rank_margin) * max(expanded_family_weight, 0.0),
        "teacher_skill_retention_bonus": 0.0,
        "path_cost_penalty": -0.01 * path_cost,
        "risk_penalty": -0.05 * risk,
        "energy_penalty": -0.001 * energy,
        "fallback_penalty": 0.0,
        "controlled_regression_penalty": 0.0,
    }


def _refined_transition_record(
    index: int,
    transition: dict[str, Any],
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    advantage: float,
    margin: float,
) -> dict[str, Any]:
    info = transition.get("info", {})
    return {
        "schema_version": "refined-coverage-trainable-transition/v1",
        "decision_index": index,
        "context_id": info.get("context_id"),
        "episode_id": info.get("episode_id"),
        "step_index": info.get("step_index"),
        "scenario_id": info.get("scenario_id"),
        "scenario_family": info.get("scenario_family"),
        "split": info.get("split"),
        "action_index": transition.get("action_index"),
        "controlled_action_index": info.get("controlled_action_index"),
        "teacher_action_index": info.get("teacher_action_index"),
        "candidate_expected_coverage_rate_delta": candidate.get("expected_coverage_rate_delta"),
        "teacher_expected_coverage_rate_delta": teacher.get("expected_coverage_rate_delta"),
        "candidate_information_gain": candidate.get("information_gain"),
        "teacher_information_gain": teacher.get("information_gain"),
        "candidate_value": candidate.get("value"),
        "teacher_value": teacher.get("value"),
        "counterfactual_coverage_advantage": float(advantage),
        "coverage_rank_margin": float(margin),
        "teacher_margin_target": info.get("teacher_margin_target"),
        "ppo_advantage": transition.get("ppo_advantage"),
        "ppo_return": transition.get("ppo_return"),
        "ppo_advantage_signal": info.get("ppo_advantage_signal"),
        "expanded_family_weight": info.get("expanded_family_weight"),
        "reward": transition.get("reward"),
        "reward_components": transition.get("reward_components", {}),
        "log_prob": transition.get("log_prob"),
        "value": transition.get("value"),
        "path_cost": candidate.get("path_cost"),
        "risk": candidate.get("risk"),
        "energy_cost": candidate.get("energy_cost"),
        "fallback_like": False,
        "controlled_regression_reason_codes": [],
        "ppo_trainable": False,
        "legacy_read_only": True,
        "canonical_v2_training_consumer_blocked": True,
        "legacy_blocker_reason": LEGACY_V1_REWARD_COMPONENT_BLOCKER,
    }


def _audit_row(
    *,
    audit_index: int,
    candidate: dict[str, Any],
    teacher: dict[str, Any] | None,
    counterfactual: dict[str, Any] | None,
    row_reason_codes: list[str],
    counterfactual_coverage_advantage: float | None = None,
    coverage_rank_margin: float | None = None,
    teacher_margin_target: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ADVANTAGE_AUDIT_SCHEMA_VERSION,
        "audit_index": audit_index,
        "context_id": candidate.get("context_id"),
        "episode_id": candidate.get("episode_id"),
        "step_index": candidate.get("step_index"),
        "scenario_id": candidate.get("scenario_id"),
        "scenario_family": candidate.get("scenario_family"),
        "split": candidate.get("split"),
        "action_index": candidate.get("action_index"),
        "teacher_action_index": candidate.get("teacher_action_index"),
        "safe_better_than_teacher_candidate": bool(candidate.get("safe_better_than_teacher_candidate")),
        "coverage_source_available": bool(candidate.get("coverage_source_available")),
        "counterfactual_row_connected": counterfactual is not None,
        "teacher_row_connected": teacher is not None,
        "candidate_decision_score": _coverage_decision_score(candidate),
        "teacher_decision_score": _coverage_decision_score(teacher or {}),
        "counterfactual_coverage_advantage": counterfactual_coverage_advantage,
        "coverage_rank_margin": coverage_rank_margin,
        "teacher_margin_target": teacher_margin_target,
        "candidate_path_cost": candidate.get("path_cost"),
        "teacher_path_cost": (teacher or {}).get("path_cost"),
        "candidate_risk": candidate.get("risk"),
        "teacher_risk": (teacher or {}).get("risk"),
        "candidate_energy_cost": candidate.get("energy_cost"),
        "teacher_energy_cost": (teacher or {}).get("energy_cost"),
        "row_reason_codes": row_reason_codes,
        "trainable": not row_reason_codes,
    }


def _collector_summary(*, transitions: list[dict[str, Any]], rejected_count: int) -> dict[str, Any]:
    return {
        "schema_version": "coverage-aware-ppo-collector-summary/v1",
        "status": "passed" if transitions else "failed",
        "reason_codes": [] if transitions else ["insufficient_refined_advantage_materialization"],
        "episode_count": len({transition["info"].get("episode_id") for transition in transitions}),
        "step_count": len(transitions),
        "ppo_trainable_transition_count": len(transitions),
        "diagnostic_transition_count": rejected_count,
        "source_fallback_trainable_count": 0,
        "fallback_or_open_grid_count": 0,
        "safety_regression_count": 0,
        "contract_violation_count": 0,
        "path_cost_regression_count": 0,
        "risk_regression_count": 0,
        "source_selection_regression_count": 0,
        "coverage_aware_reward_transition_count": len(transitions),
        "reward_component_names": sorted(
            {
                name
                for transition in transitions
                for name in transition.get("reward_components", {})
            }
        ),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _refined_ppo_update_config(
    *,
    expected_count: int,
    selected_summary: dict[str, Any],
    use_transition_info_advantage: bool = False,
    ppo_learning_rate: float | None = None,
    ppo_epochs: int | None = None,
    ppo_max_approx_kl: float | None = None,
) -> dict[str, Any]:
    config = _ppo_update_config(expected_count=expected_count, selected_summary=selected_summary)
    if use_transition_info_advantage:
        config["training"]["return_source"] = "transition_info"
        config["training"]["return_field"] = "ppo_return"
        config["training"]["advantage_field"] = "ppo_advantage"
    config["training"]["learning_rate"] = float(ppo_learning_rate) if ppo_learning_rate is not None else 2.0e-5
    config["training"]["epochs"] = int(ppo_epochs) if ppo_epochs is not None else 2
    if ppo_max_approx_kl is not None:
        config["validation"]["max_approx_kl"] = float(ppo_max_approx_kl)
    config["non_goals"].append("does_not_claim_refined_coverage_performance_without_eval")
    return config


def _run_stage5a_rerun(
    *,
    paths: dict[str, Path],
    output_root: Path,
    reward_refinement_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    overlay_path: Path,
    repo_root: Path,
    update_summary: dict[str, Any],
) -> dict[str, Any]:
    if update_summary.get("status") != "passed":
        summary = {
            "schema_version": "refined-stage5a-rerun-summary/v1",
            "status": "failed",
            "reason_codes": ["ppo_update_not_passed"],
            "next_required_change": "fix_refined_advantage_materialization",
            "policy_argmax_changed_count": 0,
            "runs_new_ppo_update": False,
            "performance_claimed": False,
        }
        _write_json(paths["stage5a_rerun_summary"], summary)
        return summary
    summary = run_policy_coverage_opportunity_margin_audit(
        coverage_driven_root=output_root,
        reward_refinement_root=reward_refinement_root,
        coverage_signal_root=coverage_signal_root,
        coverage_performance_root=coverage_performance_root,
        output_root=paths["stage5a_rerun_root"],
        repo_root=repo_root,
        candidate_coverage_overlay_path=overlay_path,
    )
    _write_json(paths["stage5a_rerun_summary"], summary)
    return summary


def _compatibility_summary(
    *,
    paths: dict[str, Path],
    reason_codes: list[str],
    output_root: Path,
    coverage_driven_root: Path,
    base_candidate_root: Path,
    refined: dict[str, Any],
    update_summary: dict[str, Any],
    replay_audit: dict[str, Any],
    post_metrics: dict[str, Any],
    comparison: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "coverage-driven-ppo-improvement-run-summary/v1",
        "generated_at": _utc_now(),
        "status": "failed" if reason_codes else "passed",
        "reason_codes": list(reason_codes),
        "next_required_change": "refined_coverage_driven_ppo_improvement_run_v2",
        "coverage_driven_ppo_improvement_status": "failed" if reason_codes else "passed",
        "source_coverage_driven_root": str(coverage_driven_root),
        "output_root": str(output_root),
        "summary": str(paths["compat_summary"]),
        "coverage_aware_batch_root": str(paths["compat_batch_root"]),
        "coverage_aware_batch_episodes": str(paths["compat_episodes"]),
        "coverage_aware_batch_transitions": str(paths["compat_transitions"]),
        "coverage_aware_batch_collector_summary": str(paths["compat_collector_summary"]),
        "ppo_update_summary": str(paths["update_summary"]),
        "training_curves": str(paths["training_curves"]),
        "diagnostics": str(paths["diagnostics"]),
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_metadata_path": str(paths["checkpoint_metadata"]),
        "candidate_summary_path": str(paths["candidate_summary"]),
        "performance_metric_table": str(paths["compat_metric_table"]),
        "replay_audit": str(paths["replay_audit"]),
        "base_candidate_root": str(base_candidate_root),
        "coverage_aware_reward_transition_count": refined["trainable_transition_count"],
        "input_ppo_trainable_transition_count": update_summary.get("input_ppo_trainable_transition_count", 0),
        "optimizer_train_transition_count": update_summary.get("optimizer_train_transition_count", 0),
        "post_improvement_metrics": post_metrics,
        "best_baseline_actor": comparison.get("best_baseline_actor"),
        "baseline_actors": comparison.get("baseline_actors", []),
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
        "controlled_regression_count": post_metrics.get("controlled_regression_count", 0),
        "experimental_checkpoint": bool(update_summary.get("experimental_checkpoint")),
        "runs_new_ppo_update": bool(update_summary.get("status") == "passed"),
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


def _compatibility_summary_updates(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": summary["status"],
        "reason_codes": summary["reason_codes"],
        "next_required_change": summary["next_required_change"],
        "coverage_return_improvement": summary["coverage_return_improvement"],
        "cumulative_coverage_rate_delta_improvement": summary[
            "cumulative_coverage_rate_delta_improvement"
        ],
        "valuable_area_covered_improvement": summary["valuable_area_covered_improvement"],
        "coverage_efficiency_regression": summary["coverage_efficiency_regression"],
        "fallback_rate": summary["fallback_rate"],
        "teacher_agreement_rate": summary["teacher_agreement_rate"],
        "controlled_regression_count": summary["controlled_regression_count"],
        "runs_new_ppo_update": summary["runs_new_ppo_update"],
        "performance_claimed": False,
    }


def _acceptance_reason_codes(
    *,
    refined: dict[str, Any],
    update_summary: dict[str, Any],
    comparison: dict[str, Any],
    post_metrics: dict[str, Any],
    policy_argmax_changed_count: int,
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> list[str]:
    reasons: list[str] = []
    if refined["safe_better_training_pair_count"] <= 0 or refined["counterfactual_advantage_nonzero_count"] <= 0:
        _add_reason(reasons, "insufficient_refined_advantage_materialization")
    if update_summary and update_summary.get("status") != "passed":
        _add_reason(reasons, "ppo_update_failed")
    if update_summary.get("status") == "passed":
        if policy_argmax_changed_count <= 0:
            _add_reason(reasons, "post_update_policy_teacher_equivalent")
        if _float_or_default(comparison.get("coverage_return_improvement"), 0.0) <= TOLERANCE:
            _add_reason(reasons, "no_coverage_return_improvement")
        if _float_or_default(comparison.get("cumulative_coverage_rate_delta_improvement"), 0.0) <= TOLERANCE:
            _add_reason(reasons, "no_cumulative_coverage_rate_delta_improvement")
        if _float_or_default(comparison.get("valuable_area_covered_improvement"), 0.0) < -TOLERANCE:
            _add_reason(reasons, "valuable_coverage_regressed")
        if comparison.get("coverage_efficiency_regression"):
            _add_reason(reasons, "coverage_efficiency_regression")
        if _float_or_default(post_metrics.get("fallback_rate"), 0.0) > 0.5:
            _add_reason(reasons, "fallback_dominates")
    if controlled_regression_count > 0:
        _add_reason(reasons, "controlled_regression_present")
    if fallback_gain_contamination_count > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    return reasons


def _next_required_change(
    *,
    reason_codes: list[str],
    refined: dict[str, Any],
    comparison: dict[str, Any],
    policy_argmax_changed_count: int,
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> str:
    if LEGACY_V1_REWARD_COMPONENT_BLOCKER in reason_codes:
        return "migrate_refined_ppo_to_canonical_reward_v2_or_use_stage18_5_guard"
    if (
        refined["safe_better_training_pair_count"] <= 0
        or refined["counterfactual_advantage_nonzero_count"] <= 0
        or "insufficient_refined_advantage_materialization" in reason_codes
    ):
        return "fix_refined_advantage_materialization"
    if controlled_regression_count > 0 or fallback_gain_contamination_count > 0:
        return "fix_guard_or_fallback_gain_contamination"
    coverage_improved = _float_or_default(comparison.get("coverage_return_improvement"), 0.0) > TOLERANCE
    cumulative_improved = (
        _float_or_default(comparison.get("cumulative_coverage_rate_delta_improvement"), 0.0) > TOLERANCE
    )
    valuable_not_down = _float_or_default(comparison.get("valuable_area_covered_improvement"), 0.0) >= -TOLERANCE
    if (
        coverage_improved
        and cumulative_improved
        and valuable_not_down
        and not comparison.get("coverage_efficiency_regression")
        and policy_argmax_changed_count > 0
        and not reason_codes
    ):
        return "shadow_canary_release_performance_validation_preflight"
    return "tune_refined_reward_margin_or_expand_safe_better_pairs"


def _iter_old_transitions(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for episode in episodes:
        for transition in episode.get("transitions", []) if isinstance(episode.get("transitions"), list) else []:
            if isinstance(transition, dict):
                transitions.append(transition)
    return transitions


def _transition_index(transitions: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for transition in transitions:
        info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
        for key in _decision_keys(info | transition):
            index.setdefault(key, transition)
    return index


def _rows_by_decision(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_decision_key(row)].append(row)
    return groups


def _counterfactual_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    index: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in rows:
        action_index = _first_int(row.get("action_index"))
        if action_index is None:
            continue
        index.setdefault((*_decision_key(row), action_index), row)
    return index


def _decision_key(row: dict[str, Any]) -> tuple[str, str, int]:
    context_id = str(row.get("context_id") or "")
    episode_id = str(row.get("episode_id") or row.get("shadow_episode_id") or row.get("source_episode_id") or "")
    step_index = _int(row.get("step_index", row.get("shadow_step_index", row.get("source_step_index"))))
    return context_id, episode_id, step_index


def _decision_keys(row: dict[str, Any]) -> list[tuple[str, str, int]]:
    context_id = str(row.get("context_id") or "")
    keys: list[tuple[str, str, int]] = []
    for episode_field, step_field in (
        ("episode_id", "step_index"),
        ("source_episode_id", "source_step_index"),
        ("source_episode_id", "step_index"),
        ("shadow_episode_id", "shadow_step_index"),
    ):
        if row.get(episode_field) is not None and row.get(step_field) is not None:
            keys.append((context_id, str(row.get(episode_field) or ""), _int(row.get(step_field))))
    if not keys:
        keys.append(_decision_key(row))
    return keys


def _teacher_row(rows: list[dict[str, Any]], teacher_action_index: int | None) -> dict[str, Any] | None:
    for row in rows:
        if bool(row.get("is_teacher_action")):
            return row
    if teacher_action_index is not None:
        for row in rows:
            if _first_int(row.get("action_index")) == teacher_action_index:
                return row
    return None


def _coverage_decision_score(row: dict[str, Any]) -> float:
    return sum(_float_or_default(row.get(field), 0.0) for field in COVERAGE_SIGNAL_FIELDS)


def _ppo_advantage_signal(
    candidate: dict[str, Any],
    fallback_advantage: float,
    ppo_advantage_field: str | None,
) -> float:
    if ppo_advantage_field:
        value = _first_float(candidate.get(ppo_advantage_field))
        if value is not None:
            return float(value)
    return float(fallback_advantage)


def _safe_cost_risk_energy(candidate: dict[str, Any], teacher: dict[str, Any]) -> bool:
    return (
        _not_materially_worse(_first_float(candidate.get("path_cost")), _first_float(teacher.get("path_cost")))
        and _not_materially_worse(_first_float(candidate.get("risk")), _first_float(teacher.get("risk")))
        and _not_materially_worse(_first_float(candidate.get("energy_cost")), _first_float(teacher.get("energy_cost")))
    )


def _not_materially_worse(value: float | None, baseline: float | None) -> bool:
    if value is None or baseline is None:
        return True
    return value <= baseline * (1.0 + MAX_ALLOWED_RELATIVE_REGRESSION) + TOLERANCE


def _fallback_gain_contamination_count(*items: Any) -> int:
    count = 0
    for item in items:
        if isinstance(item, list):
            count += sum(1 for row in item if bool(row.get("fallback_like")) and row.get("trainable"))
        elif isinstance(item, dict):
            count += _int(item.get("fallback_gain_contamination_count"))
            count += _int(item.get("fallback_coverage_gain_claimed_as_policy_gain_count"))
    return count


def _controlled_regression_count(*items: Any) -> int:
    count = 0
    for item in items:
        if isinstance(item, list):
            count += sum(1 for row in item if _string_list(row.get("controlled_regression_reason_codes")))
        elif isinstance(item, dict):
            count += _int(item.get("controlled_regression_count"))
    return count


def _render_report(
    summary: dict[str, Any],
    refined: dict[str, Any],
    stage5a_rerun_summary: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Refined Coverage-Driven PPO Improvement Run v2",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- safe_better_training_pair_count: `{summary['safe_better_training_pair_count']}`",
            f"- counterfactual_advantage_nonzero_count: `{summary['counterfactual_advantage_nonzero_count']}`",
            f"- policy_argmax_changed_count: `{summary['policy_argmax_changed_count']}`",
            f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
            f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
            f"- valuable_area_covered_improvement: `{summary['valuable_area_covered_improvement']}`",
            f"- coverage_efficiency_regression: `{summary['coverage_efficiency_regression']}`",
            f"- fallback_gain_contamination_count: `{summary['fallback_gain_contamination_count']}`",
            f"- controlled_regression_count: `{summary['controlled_regression_count']}`",
            "",
            "## Scope Guards",
            "",
            "- publishes_checkpoint: `false`",
            "- replaces_default_policy: `false`",
            "- performance_claimed: `false`",
            "- connects_real_executor: `false`",
            "- relaxes_guard: `false`",
            "",
            "## Stage 5A Rerun",
            "",
            f"- status: `{stage5a_rerun_summary.get('status')}`",
            f"- next_required_change: `{stage5a_rerun_summary.get('next_required_change')}`",
            "",
            "## Trainable Signal",
            "",
            f"- candidate_row_count: `{refined['candidate_row_count']}`",
            f"- trainable_transition_count: `{refined['trainable_transition_count']}`",
        ]
    )


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


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        _add_reason(result, value)
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is None:
                continue
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _float_or_default(value: Any, default: float = 0.0) -> float:
    number = _first_float(value)
    return default if number is None else number


if __name__ == "__main__":
    raise SystemExit(main())
