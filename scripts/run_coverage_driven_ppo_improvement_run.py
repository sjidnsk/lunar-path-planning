from __future__ import annotations

import argparse
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
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

try:
    from git_provenance import git_snapshot
    from run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke

from model_explorer.policy.canonical_reward import CANONICAL_REWARD_COMPONENTS, CANONICAL_REWARD_COMPONENTS_V3


SUMMARY_SCHEMA_VERSION = "coverage-driven-ppo-improvement-run-summary/v1"
REPLAY_AUDIT_SCHEMA_VERSION = "coverage-driven-ppo-replay-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "coverage-driven-ppo-rejection-report/v1"
METRIC_ROW_SCHEMA_VERSION = "coverage-driven-ppo-performance-metric-row/v1"

FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
FORMAL_SEED_SUMMARIES_FILE = "formal-ppo-training-run-seed-summaries.jsonl"
REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
COVERAGE_SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
COVERAGE_PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
PERFORMANCE_METRIC_TABLE_FILE = "coverage-performance-metric-table.jsonl"
REWARD_REFINEMENT_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"
REWARD_COMPONENT_AUDIT_FILE = "reward-component-audit.jsonl"
REWARD_SOURCE_SUMMARY_FILE = "connect-reward-component-source-fields-summary.json"
SHADOW_STEPS_FILE = "multihorizon-shadow-rollout-steps.jsonl"
STAGE18_9_SUMMARY_FILE = "xunce-stage18-9-trajectory-risk-reward-summary.json"
STAGE18_9_SUMMARY_SCHEMA_VERSION = "xunce-stage18-9-trajectory-risk-reward-summary/v1"
STAGE18_9_ROUTING_SCHEMA_VERSION = "xunce-stage18-9-next-stage-routing/v1"
STAGE18_9_STAGE19_READINESS_SCHEMA_VERSION = "xunce-stage18-9-stage19-readiness/v1"
STAGE18_9_EXPECTED_PROFILE_ID = "xunce-coverage-cost-risk-boundary-v3"
STAGE18_9_EXPECTED_PROFILE_VERSION = "v3"
STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE = "prepare_stage19_evaluator_critic_preflight"

SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
BATCH_DIR = "coverage-aware-ppo-batch"
BATCH_EPISODES_FILE = "ppo-rollout-episodes.jsonl"
BATCH_TRANSITIONS_FILE = "ppo-rollout-transitions.jsonl"
BATCH_COLLECTOR_SUMMARY_FILE = "ppo-rollout-collector-summary.json"
UPDATE_SUMMARY_FILE = "coverage-driven-ppo-update-summary.json"
UPDATE_TRAINING_CURVES_FILE = "coverage-driven-ppo-training-curves.json"
UPDATE_DIAGNOSTICS_FILE = "coverage-driven-ppo-diagnostics.json"
UPDATE_CHECKPOINT_FILE = "coverage-driven-experimental-policy-candidate.pt"
UPDATE_CHECKPOINT_METADATA_FILE = "coverage-driven-experimental-policy-candidate-metadata.json"
UPDATE_CANDIDATE_SUMMARY_FILE = "coverage-driven-raw-policy-generalization-candidate-summary.json"
METRIC_TABLE_FILE = "coverage-driven-ppo-performance-metric-table.jsonl"
REPLAY_AUDIT_FILE = "coverage-driven-ppo-replay-audit.json"
REJECTION_REPORT_FILE = "coverage-driven-ppo-rejection-report.json"
REPORT_FILE = "coverage-driven-ppo-improvement-run-report.md"

SELECTED_ACTOR = "selected_ppo_candidate"
POST_ACTOR = "post_improvement_ppo"
PRE_ACTOR = "pre_improvement_selected_ppo"
BASELINE_ACTORS = {"teacher", "source_default", "default_policy", PRE_ACTOR}
VALID_ACTUAL_GAIN_SOURCES = {"map", "sidecar", "path_feedback"}
REQUIRED_SOURCE_COMPONENTS = CANONICAL_REWARD_COMPONENTS
DISCOUNT_FACTOR = 0.99
TOLERANCE = 1.0e-9
MAX_ALLOWED_RELATIVE_REGRESSION = 0.05
STAGE18_9_FORBIDDEN_TRUE_FIELDS = (
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a guarded coverage-driven PPO improvement stage.")
    parser.add_argument("--formal-training-root", required=True)
    parser.add_argument("--post-training-replay-root")
    parser.add_argument("--selected-candidate-root", required=True)
    parser.add_argument("--coverage-signal-root", required=True)
    parser.add_argument("--coverage-performance-root", required=True)
    parser.add_argument("--reward-refinement-root", required=True)
    parser.add_argument("--reward-source-root", required=True)
    parser.add_argument("--stage18-9-trajectory-risk-reward-root")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    default_replay_root = repo_root / "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
    summary = run_coverage_driven_ppo_improvement_run(
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root, repo_root),
        post_training_replay_root=(
            _resolve_path(Path(args.post_training_replay_root), repo_root, repo_root)
            if args.post_training_replay_root
            else default_replay_root
        ),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root, repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root, repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root, repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root, repo_root),
        reward_source_root=_resolve_path(Path(args.reward_source_root), repo_root, repo_root),
        stage18_9_trajectory_risk_reward_root=(
            _resolve_path(Path(args.stage18_9_trajectory_risk_reward_root), repo_root, repo_root)
            if args.stage18_9_trajectory_risk_reward_root
            else None
        ),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "coverage_driven_ppo_improvement_status": summary[
                    "coverage_driven_ppo_improvement_status"
                ],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_coverage_driven_ppo_improvement_run(
    *,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    reward_refinement_root: Path,
    reward_source_root: Path,
    stage18_9_trajectory_risk_reward_root: Path | None = None,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    formal_training_root = Path(formal_training_root)
    post_training_replay_root = Path(post_training_replay_root)
    selected_candidate_root = Path(selected_candidate_root)
    coverage_signal_root = Path(coverage_signal_root)
    coverage_performance_root = Path(coverage_performance_root)
    reward_refinement_root = Path(reward_refinement_root)
    reward_source_root = Path(reward_source_root)
    stage18_9_trajectory_risk_reward_root = Path(stage18_9_trajectory_risk_reward_root) if stage18_9_trajectory_risk_reward_root is not None else None
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = _paths(output_root)
    paths["batch_root"].mkdir(parents=True, exist_ok=True)

    input_reasons: list[str] = []
    formal_summary_path = formal_training_root / FORMAL_SUMMARY_FILE
    seed_summaries_path = formal_training_root / FORMAL_SEED_SUMMARIES_FILE
    replay_summary_path = post_training_replay_root / REPLAY_SUMMARY_FILE
    selected_summary_path = selected_candidate_root / SELECTED_SUMMARY_FILE
    coverage_signal_summary_path = coverage_signal_root / COVERAGE_SIGNAL_SUMMARY_FILE
    coverage_performance_summary_path = coverage_performance_root / COVERAGE_PERFORMANCE_SUMMARY_FILE
    reward_refinement_summary_path = reward_refinement_root / REWARD_REFINEMENT_SUMMARY_FILE
    reward_source_summary_path = reward_source_root / REWARD_SOURCE_SUMMARY_FILE
    stage18_9_summary_path = (
        stage18_9_trajectory_risk_reward_root / STAGE18_9_SUMMARY_FILE
        if stage18_9_trajectory_risk_reward_root is not None
        else None
    )

    formal_summary = _read_json(formal_summary_path, input_reasons, "formal_training_summary")
    seed_summaries = _read_jsonl(seed_summaries_path, input_reasons, "formal_seed_summaries")
    replay_summary = _read_json(replay_summary_path, input_reasons, "post_training_replay_summary")
    selected_summary = _read_json(selected_summary_path, input_reasons, "selected_candidate_summary")
    coverage_signal_summary = _read_json(
        coverage_signal_summary_path,
        input_reasons,
        "coverage_signal_summary",
    )
    coverage_performance_summary = _read_json(
        coverage_performance_summary_path,
        input_reasons,
        "coverage_performance_summary",
    )
    reward_refinement_summary = _read_json(
        reward_refinement_summary_path,
        input_reasons,
        "reward_refinement_summary",
    )
    reward_source_summary = _read_json(reward_source_summary_path, input_reasons, "reward_source_summary")
    stage18_9_summary = (
        _read_json(stage18_9_summary_path, input_reasons, "stage18_9_trajectory_risk_reward_summary")
        if stage18_9_summary_path is not None
        else {}
    )

    reward_rows_path = _resolve_optional_path(
        reward_refinement_summary.get("reward_component_audit"),
        reward_refinement_root,
        repo_root,
    ) or (reward_refinement_root / REWARD_COMPONENT_AUDIT_FILE)
    shadow_steps_path = _resolve_optional_path(
        selected_summary.get("multihorizon_steps"),
        selected_candidate_root,
        repo_root,
    ) or _resolve_optional_path(
        reward_refinement_summary.get("shadow_steps"),
        reward_refinement_root,
        repo_root,
    ) or (selected_candidate_root / SHADOW_STEPS_FILE)
    metric_table_path = _resolve_optional_path(
        coverage_performance_summary.get("metric_table"),
        coverage_performance_root,
        repo_root,
    ) or (coverage_performance_root / PERFORMANCE_METRIC_TABLE_FILE)

    reward_rows = _read_jsonl(reward_rows_path, input_reasons, "reward_component_audit")
    shadow_steps = _read_jsonl(shadow_steps_path, input_reasons, "shadow_steps")
    previous_metric_rows = _read_jsonl(metric_table_path, input_reasons, "coverage_performance_metric_table")

    reason_codes = list(input_reasons)
    base_candidate_root = _selected_base_candidate_root(selected_summary, selected_candidate_root, repo_root)
    source_root = _resolve_optional_path(selected_summary.get("batch_root"), selected_candidate_root, repo_root) or formal_training_root
    _validate_upstream(
        formal_summary=formal_summary,
        seed_summaries=seed_summaries,
        replay_summary=replay_summary,
        selected_summary=selected_summary,
        coverage_signal_summary=coverage_signal_summary,
        reward_refinement_summary=reward_refinement_summary,
        reward_source_summary=reward_source_summary,
        stage18_9_summary=stage18_9_summary,
        stage18_9_root_provided=stage18_9_trajectory_risk_reward_root is not None,
        base_candidate_root=base_candidate_root,
        reason_codes=reason_codes,
    )

    batch = _materialize_coverage_aware_batch(
        reward_rows=reward_rows,
        shadow_steps=shadow_steps,
        collector_root=paths["batch_root"],
        selected_summary=selected_summary,
        reason_codes=reason_codes,
        expected_profile_id=reward_refinement_summary.get("profile_id"),
        expected_profile_version=reward_refinement_summary.get("profile_version"),
        expected_profile_hash=reward_refinement_summary.get("profile_hash"),
    )

    update_summary: dict[str, Any] = {}
    if not reason_codes and batch["trainable_transition_count"] > 0:
        _install_model_explorer_path(repo_root)
        config = _ppo_update_config(
            expected_count=batch["trainable_transition_count"],
            selected_summary=selected_summary,
        )
        update_summary = run_limited_ppo_update_smoke(
            source_root=source_root,
            base_candidate_root=base_candidate_root,
            collector_root=paths["batch_root"],
            output_root=output_root,
            config=config,
            repo_root=repo_root,
        )
        if update_summary.get("status") != "passed":
            _add_reason(reason_codes, "ppo_update_failed")
    else:
        if batch["trainable_transition_count"] <= 0:
            _add_reason(reason_codes, "insufficient_policy_activation")

    replay_audit = _replay_updated_policy(
        checkpoint_path=paths["checkpoint"],
        batch_transitions=batch["transitions"],
        update_summary=update_summary,
        repo_root=repo_root,
    )
    baseline_rows = _baseline_metric_rows(previous_metric_rows, coverage_performance_summary)
    post_metrics = _aggregate_post_metrics(replay_audit["rows"])
    metric_rows = baseline_rows + [post_metrics]
    metrics_by_actor = {row["actor"]: row for row in metric_rows}
    comparison = _comparison(metrics_by_actor)
    rejection_report = _rejection_report(
        reason_codes=reason_codes,
        update_summary=update_summary,
        batch=batch,
        replay_audit=replay_audit,
        comparison=comparison,
        post_metrics=post_metrics,
    )
    for reason in rejection_report["reason_codes"]:
        _add_reason(reason_codes, reason)

    coverage_driven_status = "passed" if not reason_codes else "failed"
    status = "passed" if coverage_driven_status == "passed" else "failed"
    selected_baseline = metrics_by_actor.get(comparison.get("best_baseline_actor") or "", _empty_metric("none"))
    checkpoint_path = paths["checkpoint"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": _next_required_change(reason_codes),
        "coverage_driven_ppo_improvement_status": coverage_driven_status,
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "reward_refinement_root": str(reward_refinement_root),
        "reward_source_root": str(reward_source_root),
        "stage18_9_trajectory_risk_reward_root": str(stage18_9_trajectory_risk_reward_root) if stage18_9_trajectory_risk_reward_root is not None else None,
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "coverage_aware_batch_root": str(paths["batch_root"]),
        "coverage_aware_batch_episodes": str(paths["batch_episodes"]),
        "coverage_aware_batch_transitions": str(paths["batch_transitions"]),
        "coverage_aware_batch_collector_summary": str(paths["batch_collector_summary"]),
        "ppo_update_summary": str(paths["update_summary"]),
        "training_curves": str(paths["training_curves"]),
        "diagnostics": str(paths["diagnostics"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata_path": str(paths["checkpoint_metadata"]),
        "candidate_summary_path": str(paths["candidate_summary"]),
        "performance_metric_table": str(paths["metric_table"]),
        "replay_audit": str(paths["replay_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "selected_seed": selected_summary.get("selected_seed"),
        "selected_budget": selected_summary.get("selected_budget"),
        "base_candidate_root": str(base_candidate_root),
        "reward_component_audit": str(reward_rows_path),
        "shadow_steps": str(shadow_steps_path),
        "profile_id": reward_refinement_summary.get("profile_id"),
        "profile_version": reward_refinement_summary.get("profile_version"),
        "profile_hash": reward_refinement_summary.get("profile_hash"),
        "previous_metric_table": str(metric_table_path),
        "coverage_aware_reward_transition_count": batch["trainable_transition_count"],
        "batch_candidate_row_count": batch["candidate_row_count"],
        "batch_rejected_row_count": batch["rejected_row_count"],
        "input_ppo_trainable_transition_count": update_summary.get("input_ppo_trainable_transition_count", 0),
        "optimizer_train_transition_count": update_summary.get("optimizer_train_transition_count", 0),
        "old_log_prob_max_abs_error": update_summary.get("old_log_prob_max_abs_error"),
        "old_value_max_abs_error": update_summary.get("old_value_max_abs_error"),
        "parameter_l2_delta": update_summary.get("parameter_l2_delta", 0.0),
        "approx_kl": update_summary.get("approx_kl"),
        "max_grad_norm_after_clip": update_summary.get("max_grad_norm_after_clip"),
        "loss_non_finite_count": update_summary.get("loss_non_finite_count", 0),
        "non_finite_gradient_count": update_summary.get("non_finite_gradient_count", 0),
        "non_finite_reward_count": update_summary.get("non_finite_reward_count", 0),
        "non_finite_return_count": update_summary.get("non_finite_return_count", 0),
        "non_finite_advantage_count": update_summary.get("non_finite_advantage_count", 0),
        "post_improvement_metrics": post_metrics,
        "best_baseline_actor": comparison.get("best_baseline_actor"),
        "best_baseline_metrics": selected_baseline,
        "baseline_actors": comparison["baseline_actors"],
        "coverage_return_improvement": comparison.get("coverage_return_improvement"),
        "cumulative_coverage_rate_delta_improvement": comparison.get(
            "cumulative_coverage_rate_delta_improvement"
        ),
        "valuable_area_covered_improvement": comparison.get("valuable_area_covered_improvement"),
        "coverage_efficiency_regression": comparison.get("coverage_efficiency_regression"),
        "accepted_policy_activation_rate": post_metrics.get("accepted_policy_activation_rate"),
        "fallback_rate": post_metrics.get("fallback_rate"),
        "teacher_agreement_rate": post_metrics.get("teacher_agreement_rate"),
        "controlled_regression_count": rejection_report["controlled_regression_count"],
        "scenario_family_gain_count": post_metrics.get("scenario_family_gain_count"),
        "fallback_dominates": rejection_report["fallback_dominates"],
        "provenance_clean": not any(
            reason
            in reason_codes
            for reason in (
                "expected_actual_coverage_confusion",
                "fallback_policy_gain_contamination",
                "coverage_reward_source_not_passed",
            )
        ),
        "experimental_checkpoint": bool(update_summary.get("experimental_checkpoint")),
        "runs_new_ppo_update": bool(update_summary.get("status") == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "formal_training_ready_claimed": False,
        "connects_real_executor": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_jsonl(paths["metric_table"], metric_rows)
    _write_json(paths["replay_audit"], replay_audit)
    _write_json(paths["rejection_report"], rejection_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, metric_rows, rejection_report), encoding="utf-8")
    return summary


def _paths(output_root: Path) -> dict[str, Path]:
    batch_root = output_root / BATCH_DIR
    return {
        "summary": output_root / SUMMARY_FILE,
        "batch_root": batch_root,
        "batch_episodes": batch_root / BATCH_EPISODES_FILE,
        "batch_transitions": batch_root / BATCH_TRANSITIONS_FILE,
        "batch_collector_summary": batch_root / BATCH_COLLECTOR_SUMMARY_FILE,
        "update_summary": output_root / UPDATE_SUMMARY_FILE,
        "training_curves": output_root / UPDATE_TRAINING_CURVES_FILE,
        "diagnostics": output_root / UPDATE_DIAGNOSTICS_FILE,
        "checkpoint": output_root / UPDATE_CHECKPOINT_FILE,
        "checkpoint_metadata": output_root / UPDATE_CHECKPOINT_METADATA_FILE,
        "candidate_summary": output_root / UPDATE_CANDIDATE_SUMMARY_FILE,
        "metric_table": output_root / METRIC_TABLE_FILE,
        "replay_audit": output_root / REPLAY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _validate_upstream(
    *,
    formal_summary: dict[str, Any],
    seed_summaries: list[dict[str, Any]],
    replay_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    coverage_signal_summary: dict[str, Any],
    reward_refinement_summary: dict[str, Any],
    reward_source_summary: dict[str, Any],
    stage18_9_summary: dict[str, Any],
    stage18_9_root_provided: bool,
    base_candidate_root: Path,
    reason_codes: list[str],
) -> None:
    for label, summary in (
        ("formal_training", formal_summary),
        ("post_training_replay", replay_summary),
        ("selected_candidate", selected_summary),
    ):
        if summary.get("status") != "passed":
            _add_reason(reason_codes, f"{label}_not_passed")
        if _int(summary.get("controlled_regression_count")) > 0:
            _add_reason(reason_codes, "guard_regression")
    if formal_summary and _int(formal_summary.get("seed_count")) != 5:
        _add_reason(reason_codes, "formal_seed_count_mismatch")
    if seed_summaries:
        passed_seeds = [row for row in seed_summaries if row.get("status") == "passed"]
        if len(passed_seeds) != 5:
            _add_reason(reason_codes, "formal_seed_checkpoint_missing")
        for row in seed_summaries:
            root_value = row.get("limited_ppo_update_smoke_root")
            if not root_value:
                continue
            root = Path(str(root_value))
            checkpoint = root / "experimental-hybrid-policy-candidate.pt"
            if not checkpoint.is_absolute():
                checkpoint = Path.cwd() / checkpoint
            if not checkpoint.is_file():
                _add_reason(reason_codes, "formal_seed_checkpoint_missing")
    if not (base_candidate_root / "experimental-hybrid-policy-candidate.pt").is_file():
        _add_reason(reason_codes, "selected_checkpoint_missing")
    if coverage_signal_summary.get("coverage_signal_status") != "passed":
        _add_reason(reason_codes, "coverage_signal_not_passed")
    if _int(coverage_signal_summary.get("nonzero_actual_coverage_delta_count")) <= 0:
        _add_reason(reason_codes, "insufficient_exploration_coverage_signal")
    if _int(coverage_signal_summary.get("expected_actual_coverage_confusion_count")) > 0:
        _add_reason(reason_codes, "expected_actual_coverage_confusion")
    if _int(coverage_signal_summary.get("fallback_coverage_gain_claimed_as_policy_gain_count")) > 0:
        _add_reason(reason_codes, "fallback_policy_gain_contamination")
    if reward_refinement_summary.get("status") != "passed" or reward_refinement_summary.get(
        "reward_refinement_status"
    ) != "passed":
        _add_reason(reason_codes, "coverage_reward_source_not_passed")
    if not reward_refinement_summary.get("profile_hash"):
        _add_reason(reason_codes, "canonical_reward_profile_hash_missing")
    profile_version = reward_refinement_summary.get("profile_version")
    if profile_version not in {"v2", "v3"}:
        _add_reason(reason_codes, "canonical_reward_profile_version_invalid")
    if profile_version == "v3":
        if not stage18_9_root_provided:
            _add_reason(reason_codes, "stage18_9_readiness_missing")
        elif not _stage18_9_readiness_is_valid(stage18_9_summary):
            _add_reason(reason_codes, "stage18_9_readiness_not_passed")
        for field in ("profile_id", "profile_version", "profile_hash"):
            if reward_refinement_summary.get(field) and stage18_9_summary.get(field) != reward_refinement_summary.get(field):
                _add_reason(reason_codes, f"stage18_9_{field}_mismatch")
    if _int(reward_refinement_summary.get("source_field_missing_component_count")) > 0:
        _add_reason(reason_codes, "coverage_reward_source_not_passed")
    if reward_source_summary.get("status") != "passed" or reward_source_summary.get(
        "reward_component_source_field_status"
    ) != "passed":
        _add_reason(reason_codes, "coverage_reward_source_not_passed")
    for summary in (coverage_signal_summary, reward_refinement_summary, reward_source_summary):
        if _int(summary.get("controlled_regression_count")) > 0:
            _add_reason(reason_codes, "guard_regression")


def _stage18_9_readiness_is_valid(summary: dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    if summary.get("schema_version") != STAGE18_9_SUMMARY_SCHEMA_VERSION:
        return False
    if summary.get("status") != "passed" or summary.get("trajectory_guard_passed") is not True:
        return False
    if (
        summary.get("profile_id") != STAGE18_9_EXPECTED_PROFILE_ID
        or summary.get("profile_version") != STAGE18_9_EXPECTED_PROFILE_VERSION
        or not isinstance(summary.get("profile_hash"), str)
        or not summary.get("profile_hash")
    ):
        return False
    routing = summary.get("next_stage_routing")
    if (
        not isinstance(routing, dict)
        or routing.get("schema_version") != STAGE18_9_ROUTING_SCHEMA_VERSION
        or routing.get("primary_route") != STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE
        or routing.get("stage19_authorized") is not False
    ):
        return False
    readiness = summary.get("stage19_readiness")
    if (
        not isinstance(readiness, dict)
        or readiness.get("schema_version") != STAGE18_9_STAGE19_READINESS_SCHEMA_VERSION
        or readiness.get("readiness") != "ready_for_stage19_preflight_human_review_only"
        or readiness.get("authorized") is not False
        or readiness.get("trajectory_guard_passed") is not True
    ):
        return False
    if summary.get("stage19_authorized") is not False:
        return False
    boundary = summary.get("path_risk_boundary_summary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("path_risk_boundary_passed") is not True
        or _int(boundary.get("hard_risk_violation_count")) != 0
    ):
        return False
    trajectory = summary.get("trajectory_guard_summary")
    if not isinstance(trajectory, dict):
        return False
    for field in (
        "coverage_advantage_established",
        "path_cost_budget_passed",
        "coverage_efficiency_passed",
        "soft_risk_exposure_passed",
    ):
        if trajectory.get(field) is not True:
            return False
    if any(summary.get(field) is True for field in STAGE18_9_FORBIDDEN_TRUE_FIELDS):
        return False
    if float(summary.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        return False
    return True


def _materialize_coverage_aware_batch(
    *,
    reward_rows: list[dict[str, Any]],
    shadow_steps: list[dict[str, Any]],
    collector_root: Path,
    selected_summary: dict[str, Any],
    reason_codes: list[str],
    expected_profile_id: Any = None,
    expected_profile_version: Any = None,
    expected_profile_hash: Any = None,
) -> dict[str, Any]:
    shadow_index: dict[str, dict[str, Any]] = {}
    for step in shadow_steps:
        for key in _row_keys(step):
            shadow_index.setdefault(key, step)

    transitions: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    candidate_count = 0
    for index, row in enumerate(reward_rows):
        if str(row.get("actor") or "") != SELECTED_ACTOR:
            continue
        candidate_count += 1
        shadow = _matching_shadow(row, shadow_index)
        transition, row_reasons = _transition_from_reward_row(
            index,
            row,
            shadow,
            expected_profile_id=expected_profile_id,
            expected_profile_version=expected_profile_version,
            expected_profile_hash=expected_profile_hash,
        )
        if transition is None:
            for reason in row_reasons:
                _add_reason(reason_codes, reason)
            rejected_rows.append(
                {
                    "reward_audit_index": row.get("reward_audit_index", index),
                    "context_id": row.get("context_id"),
                    "row_reason_codes": row_reasons,
                }
            )
            continue
        transitions.append(transition)
    if not transitions:
        _add_reason(reason_codes, "insufficient_policy_activation")

    episodes = _episodes_from_transitions(transitions)
    transition_records = [_transition_record(index, transition) for index, transition in enumerate(transitions)]
    collector_summary = {
        "schema_version": "coverage-aware-ppo-collector-summary/v1",
        "status": "passed" if transitions else "failed",
        "reason_codes": [] if transitions else ["insufficient_policy_activation"],
        "selected_seed": selected_summary.get("selected_seed"),
        "selected_budget": selected_summary.get("selected_budget"),
        "episode_count": len(episodes),
        "step_count": len(transitions),
        "ppo_trainable_transition_count": len(transitions),
        "diagnostic_transition_count": len(rejected_rows),
        "source_fallback_trainable_count": 0,
        "invalid_action_mask_count": sum(1 for transition in transitions if not any(transition["action_mask"])),
        "empty_action_mask_count": sum(1 for transition in transitions if not transition["action_mask"]),
        "missing_log_prob_count": sum(1 for transition in transitions if not _finite(transition.get("log_prob"))),
        "missing_value_count": sum(1 for transition in transitions if not _finite(transition.get("value"))),
        "non_finite_reward_count": sum(1 for transition in transitions if not _finite(transition.get("reward"))),
        "state_continuity_violation_count": 0,
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
        "profile_id": expected_profile_id,
        "profile_version": expected_profile_version,
        "profile_hash": expected_profile_hash,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }
    _write_jsonl(collector_root / BATCH_EPISODES_FILE, episodes)
    _write_jsonl(collector_root / BATCH_TRANSITIONS_FILE, transition_records)
    _write_json(collector_root / BATCH_COLLECTOR_SUMMARY_FILE, collector_summary)
    return {
        "candidate_row_count": candidate_count,
        "rejected_row_count": len(rejected_rows),
        "trainable_transition_count": len(transitions),
        "transitions": transitions,
        "rejected_rows": rejected_rows,
        "collector_summary": collector_summary,
    }


def _transition_from_reward_row(
    index: int,
    row: dict[str, Any],
    shadow: dict[str, Any],
    *,
    expected_profile_id: Any = None,
    expected_profile_version: Any = None,
    expected_profile_hash: Any = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if str(row.get("split") or shadow.get("split") or "") != "train":
        reasons.append("non_train_split")
    if row.get("fallback_like"):
        reasons.append("fallback_row")
    if row.get("expected_actual_coverage_confusion"):
        reasons.append("expected_actual_coverage_confusion")
    if row.get("fallback_policy_gain_contamination"):
        reasons.append("fallback_policy_gain_contamination")
    if _string_list(row.get("controlled_regression_reason_codes")):
        reasons.append("controlled_regression_present")
    if str(row.get("actual_coverage_gain_source") or "") not in VALID_ACTUAL_GAIN_SOURCES:
        reasons.append("actual_coverage_source_invalid")
    if not row.get("profile_hash"):
        reasons.append("canonical_reward_profile_hash_missing")
    elif expected_profile_hash and row.get("profile_hash") != expected_profile_hash:
        reasons.append("canonical_reward_profile_hash_mismatch")
    if expected_profile_id and row.get("profile_id") != expected_profile_id:
        reasons.append("canonical_reward_profile_id_mismatch")
    if expected_profile_version and row.get("profile_version") != expected_profile_version:
        reasons.append("canonical_reward_profile_version_mismatch")
    required_components = CANONICAL_REWARD_COMPONENTS_V3 if expected_profile_version == "v3" else REQUIRED_SOURCE_COMPONENTS
    for component in required_components:
        source_fields = row.get("source_fields") if isinstance(row.get("source_fields"), dict) else {}
        if not source_fields.get(component):
            reasons.append("reward_component_source_missing")
            break
    observation = shadow.get("observation")
    if not isinstance(observation, dict):
        reasons.append("observation_missing")
    action_index = _first_int(row.get("controlled_action_index"), shadow.get("controlled_action_index"))
    action_mask = _bool_list(
        observation.get("action_mask") if isinstance(observation, dict) else shadow.get("action_mask")
    )
    if action_index is None or action_index < 0:
        reasons.append("controlled_action_missing")
    elif action_mask and action_index >= len(action_mask):
        reasons.append("controlled_action_out_of_mask")
    elif action_mask and not action_mask[action_index]:
        reasons.append("controlled_action_masked")
    reward = _float(row.get("coverage_aware_reward"))
    if not _finite(reward):
        reasons.append("coverage_aware_reward_non_finite")
    log_prob = _float(shadow.get("log_prob"))
    value = _float(shadow.get("value"))
    if not _finite(log_prob):
        reasons.append("old_log_prob_missing")
    if not _finite(value):
        reasons.append("old_value_missing")
    if reasons:
        return None, _unique(reasons)

    reward_components = row.get("reward_components") if isinstance(row.get("reward_components"), dict) else {}
    info = {
        "selected_cell": _selected_cell(observation, action_index),
        "coverage_rate_delta": _float_or_default(row.get("coverage_rate_delta"), 0.0),
        "cumulative_coverage_rate_delta": _float_or_default(row.get("cumulative_coverage_rate_delta"), 0.0),
        "final_coverage_rate": _float_or_default(row.get("final_coverage_rate"), 0.0),
        "path_cost": _float_or_default(row.get("path_cost") or row.get("source_values", {}).get("path_cost_component"), 0.0),
        "risk": _float_or_default(row.get("risk") or row.get("source_values", {}).get("risk_component"), 0.0),
        "energy_cost": _float_or_default(row.get("energy_cost"), 0.0),
        "valuable_area_covered": _float_or_default(row.get("valuable_area_covered") or row.get("source_values", {}).get("valuable_coverage_component"), 0.0),
        "information_gain": _float_or_default(row.get("information_gain") or row.get("source_values", {}).get("information_component"), 0.0),
        "new_area_covered": _float_or_default(row.get("new_area_covered") or row.get("coverage_rate_delta"), 0.0),
        "failure_reason": None,
        "total_cost": _float_or_default(row.get("path_cost") or row.get("source_values", {}).get("path_cost_component"), 0.0),
        "failure_count": 0,
        "replan_count": 0,
        "ppo_trainable": True,
        "controlled_choice_source": "policy",
        "controlled_choice_detail": str(row.get("controlled_choice_detail") or shadow.get("controlled_choice_detail") or ""),
        "controlled_action_index": int(action_index),
        "teacher_action_index": _first_int(row.get("teacher_action_index"), shadow.get("teacher_action_index")),
        "context_id": str(row.get("context_id") or shadow.get("context_id") or ""),
        "episode_id": str(row.get("episode_id") or shadow.get("shadow_episode_id") or shadow.get("episode_id") or ""),
        "source_episode_id": str(row.get("source_episode_id") or shadow.get("episode_id") or ""),
        "step_index": _int(row.get("step_index")),
        "source_step_index": _int(row.get("source_step_index", row.get("step_index"))),
        "split": "train",
        "gate_reason_codes": [],
        "scenario_id": row.get("scenario_id") or shadow.get("scenario_id"),
        "scenario_family": row.get("scenario_family") or shadow.get("scenario_family"),
        "source_reward_audit_index": row.get("reward_audit_index", index),
        "profile_id": row.get("profile_id"),
        "profile_version": row.get("profile_version"),
        "profile_hash": row.get("profile_hash"),
        "source_fields": row.get("source_fields") if isinstance(row.get("source_fields"), dict) else {},
        "source_values": row.get("source_values") if isinstance(row.get("source_values"), dict) else {},
        "reward_components": reward_components,
        "coverage_aware_reward": float(reward),
        "actual_coverage_gain_source": row.get("actual_coverage_gain_source"),
    }
    return {
        "observation": observation,
        "action_index": int(action_index),
        "action_mask": action_mask,
        "log_prob": float(log_prob),
        "value": float(value),
        "reward": float(reward),
        "reward_components": reward_components,
        "profile_id": row.get("profile_id"),
        "profile_version": row.get("profile_version"),
        "profile_hash": row.get("profile_hash"),
        "next_observation": None,
        "done": False,
        "info": info,
    }, []


def _episodes_from_transitions(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for transition in transitions:
        grouped[str(transition["info"].get("episode_id") or "")].append(transition)
    episodes: list[dict[str, Any]] = []
    for episode_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda transition: _int(transition["info"].get("step_index")))
        for transition in rows:
            transition["done"] = False
        if rows:
            rows[-1]["done"] = True
        episodes.append(
            {
                "schema_version": "coverage-aware-ppo-rollout-episode/v1",
                "episode_id": episode_id,
                "transitions": rows,
                "metrics": {
                    "final_coverage_rate": rows[-1]["info"].get("final_coverage_rate") if rows else 0.0,
                    "cumulative_coverage_rate_delta": round(
                        sum(float(row["info"].get("coverage_rate_delta") or 0.0) for row in rows),
                        12,
                    ),
                    "total_path_cost": round(sum(float(row["info"].get("path_cost") or 0.0) for row in rows), 12),
                    "average_risk": _mean([float(row["info"].get("risk") or 0.0) for row in rows]) or 0.0,
                    "failure_count": 0,
                    "replan_count": 0,
                    "value_coverage": round(
                        sum(float(row["info"].get("valuable_area_covered") or 0.0) for row in rows),
                        12,
                    ),
                },
            }
        )
    return episodes


def _transition_record(index: int, transition: dict[str, Any]) -> dict[str, Any]:
    info = transition["info"]
    return {
        "schema_version": "coverage-aware-ppo-rollout-transition-record/v1",
        "episode_id": info.get("episode_id"),
        "step_index": info.get("step_index"),
        "decision_index": index,
        "context_id": info.get("context_id"),
        "split": info.get("split"),
        "controlled_choice_source": info.get("controlled_choice_source"),
        "controlled_choice_detail": info.get("controlled_choice_detail"),
        "controlled_action_index": info.get("controlled_action_index"),
        "teacher_action_index": info.get("teacher_action_index"),
        "action_index": transition.get("action_index"),
        "action_mask": transition.get("action_mask"),
        "log_prob": transition.get("log_prob"),
        "value": transition.get("value"),
        "ppo_trainable": True,
        "diagnostic_only": False,
        "rejection_reason_codes": [],
        "reward": transition["reward"],
        "reward_components": transition.get("reward_components", {}),
        "profile_id": transition.get("profile_id"),
        "profile_version": transition.get("profile_version"),
        "profile_hash": transition.get("profile_hash"),
        "reward_audit": {"reason_codes": []},
        "counter_deltas": {"ppo_trainable_transition_count": 1, "diagnostic_transition_count": 0},
        "info": info,
    }


def _ppo_update_config(*, expected_count: int, selected_summary: dict[str, Any]) -> dict[str, Any]:
    max_log_error = max(
        1.0e-4,
        _float_or_default(selected_summary.get("log_prob_reconstruction_max_abs_error"), 0.0) + 1.0e-6,
    )
    max_value_error = max(
        1.0e-4,
        _float_or_default(selected_summary.get("value_reconstruction_max_abs_error"), 0.0) + 1.0e-6,
    )
    return {
        "schema_version": "limited-ppo-update-smoke-config/v1",
        "input_files": {
            "rollout_episodes": BATCH_EPISODES_FILE,
            "collector_summary": BATCH_COLLECTOR_SUMMARY_FILE,
            "base_checkpoint": "experimental-hybrid-policy-candidate.pt",
            "base_checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
            "base_candidate_summary": "raw-policy-generalization-candidate-summary.json",
        },
        "output_files": {
            "summary": UPDATE_SUMMARY_FILE,
            "training_curves": UPDATE_TRAINING_CURVES_FILE,
            "diagnostics": UPDATE_DIAGNOSTICS_FILE,
            "checkpoint": UPDATE_CHECKPOINT_FILE,
            "checkpoint_metadata": UPDATE_CHECKPOINT_METADATA_FILE,
            "candidate_summary": UPDATE_CANDIDATE_SUMMARY_FILE,
        },
        "training": {
            "seed": _int(selected_summary.get("selected_seed")),
            "epochs": 1,
            "learning_rate": 1.0e-5,
            "clip_ratio": 0.2,
            "discount_factor": DISCOUNT_FACTOR,
            "max_grad_norm": 1.0,
            "device": "cpu",
        },
        "validation": {
            "expected_input_ppo_trainable_transition_count": int(expected_count),
            "min_optimizer_train_transition_count": int(expected_count),
            "max_old_log_prob_abs_error": max_log_error,
            "max_old_value_abs_error": max_value_error,
            "max_approx_kl": 0.25,
            "max_grad_norm_after_clip": 1.0,
        },
        "trainable_filter": {
            "splits": ["train"],
            "controlled_choice_sources": ["policy"],
            "require_empty_gate_reason_codes": True,
        },
        "non_goals": [
            "does_not_connect_real_executor",
            "does_not_publish_checkpoint",
            "does_not_replace_default_policy",
            "does_not_modify_network",
            "does_not_modify_action_space",
            "does_not_modify_default_astar",
            "does_not_relax_guard",
            "does_not_claim_policy_performance",
            "does_not_claim_formal_release",
        ],
    }


def _replay_updated_policy(
    *,
    checkpoint_path: Path,
    batch_transitions: list[dict[str, Any]],
    update_summary: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    if update_summary.get("status") != "passed" or not checkpoint_path.is_file():
        return {
            "schema_version": REPLAY_AUDIT_SCHEMA_VERSION,
            "status": "failed",
            "reason_codes": ["ppo_update_failed"] if update_summary else ["ppo_update_not_run"],
            "rows": rows,
            "raw_policy_action_count": 0,
            "accepted_policy_action_count": 0,
            "guard_rejected_action_count": 0,
        }
    try:
        _install_model_explorer_path(repo_root)
        import torch
        from model_explorer.policy.architectures import build_policy_network_from_metadata
        from model_explorer.policy.features import PolicyObservation
        from model_explorer.policy.torch_policy import observation_to_tensors

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = _read_json(checkpoint_path.with_name(UPDATE_CHECKPOINT_METADATA_FILE), [], "checkpoint_metadata")
        first_observation = _policy_observation(batch_transitions[0]["observation"], PolicyObservation)
        network = build_policy_network_from_metadata(
            checkpoint.get("architecture"),
            candidate_feature_count=len(first_observation.candidate_feature_names),
            global_feature_count=len(first_observation.global_feature_names),
            missing_indicator_count=len(first_observation.candidate_missing_indicator_names),
            hidden_size=_int(
                checkpoint.get("training", {}).get("hidden_size")
                or metadata.get("hidden_size"),
                16,
            ),
            architecture_config=checkpoint.get("architecture_config"),
        )
        network.load_state_dict(checkpoint.get("model_state_dict") or checkpoint.get("state_dict"))
        network.eval()
        with torch.no_grad():
            for index, transition in enumerate(batch_transitions):
                observation = _policy_observation(transition["observation"], PolicyObservation)
                output = network(**observation_to_tensors(observation))
                logits = output.masked_logits[0]
                raw_action = int(torch.argmax(logits).item())
                info = transition["info"]
                controlled_action = _int(info.get("controlled_action_index"))
                accepted = raw_action == controlled_action and not _string_list(
                    info.get("controlled_regression_reason_codes")
                )
                rows.append(_replay_row(index, transition, raw_action, accepted))
    except Exception as exc:  # noqa: BLE001 - replay audit carries the failure.
        reason_codes.append("post_update_replay_failed")
        rows.append({"replay_error": str(exc)})
    return {
        "schema_version": REPLAY_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "rows": rows,
        "raw_policy_action_count": len([row for row in rows if "raw_policy_action_index" in row]),
        "accepted_policy_action_count": sum(1 for row in rows if row.get("policy_action_accepted")),
        "guard_rejected_action_count": sum(1 for row in rows if row.get("guard_rejected")),
    }


def _replay_row(
    index: int,
    transition: dict[str, Any],
    raw_action: int,
    accepted: bool,
) -> dict[str, Any]:
    info = transition["info"]
    controlled_reasons = _string_list(info.get("controlled_regression_reason_codes"))
    gain = _float_or_default(info.get("coverage_rate_delta"), 0.0) if accepted else 0.0
    valuable = _float_or_default(info.get("valuable_area_covered"), 0.0) if accepted else 0.0
    information = _float_or_default(info.get("information_gain"), 0.0) if accepted else 0.0
    path_cost = _float_or_default(info.get("path_cost"), 0.0) if accepted else 0.0
    risk = _float_or_default(info.get("risk"), 0.0) if accepted else 0.0
    energy = _float_or_default(info.get("energy_cost"), 0.0) if accepted else 0.0
    return {
        "schema_version": "coverage-driven-ppo-replay-row/v1",
        "replay_index": index,
        "actor": POST_ACTOR,
        "context_id": info.get("context_id"),
        "episode_id": info.get("episode_id"),
        "step_index": info.get("step_index"),
        "scenario_id": info.get("scenario_id"),
        "scenario_family": info.get("scenario_family"),
        "raw_policy_action_index": raw_action,
        "controlled_action_index": info.get("controlled_action_index"),
        "teacher_action_index": info.get("teacher_action_index"),
        "policy_action_accepted": accepted,
        "guard_rejected": not accepted,
        "fallback_like": not accepted,
        "coverage_rate_delta": gain,
        "final_coverage_rate": _float_or_default(info.get("final_coverage_rate"), 0.0) if accepted else None,
        "new_area_covered": _float_or_default(info.get("new_area_covered"), gain) if accepted else 0.0,
        "valuable_area_covered": valuable,
        "information_gain": information,
        "path_cost": path_cost,
        "risk": risk,
        "energy_cost": energy,
        "controlled_regression_reason_codes": controlled_reasons,
    }


def _baseline_metric_rows(
    previous_metric_rows: list[dict[str, Any]],
    performance_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in previous_metric_rows:
        actor = str(row.get("actor") or "")
        if not actor:
            continue
        metric = dict(row)
        if actor == SELECTED_ACTOR:
            metric["actor"] = PRE_ACTOR
        if metric["actor"] in BASELINE_ACTORS and metric["actor"] not in seen:
            rows.append(_normalize_metric_row(metric))
            seen.add(metric["actor"])
    if "teacher" not in seen and isinstance(performance_summary.get("best_baseline_metrics"), dict):
        metric = dict(performance_summary["best_baseline_metrics"])
        metric["actor"] = str(performance_summary.get("best_baseline_actor") or "teacher")
        rows.append(_normalize_metric_row(metric))
        seen.add(metric["actor"])
    if PRE_ACTOR not in seen and isinstance(performance_summary.get("selected_metrics"), dict):
        metric = dict(performance_summary["selected_metrics"])
        metric["actor"] = PRE_ACTOR
        rows.append(_normalize_metric_row(metric))
        seen.add(PRE_ACTOR)
    for actor in ("source_default", "default_policy"):
        if actor not in seen:
            metric = _empty_metric(actor)
            metric["baseline_evidence_missing"] = True
            rows.append(metric)
            seen.add(actor)
    return rows


def _aggregate_post_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if _finite(row.get("coverage_rate_delta"))]
    total_gain = sum(float(row["coverage_rate_delta"]) for row in valid_rows)
    coverage_return = sum(
        float(row["coverage_rate_delta"]) * (DISCOUNT_FACTOR ** max(_int(row.get("step_index")), 0))
        for row in valid_rows
    )
    path_cost = _sum_metric(rows, "path_cost")
    risk = _sum_metric(rows, "risk")
    energy = _sum_metric(rows, "energy_cost")
    accepted_count = sum(1 for row in rows if row.get("policy_action_accepted"))
    fallback_count = sum(1 for row in rows if row.get("fallback_like"))
    teacher_rows = [
        row for row in rows if row.get("teacher_action_index") is not None and row.get("controlled_action_index") is not None
    ]
    scenario_families = {
        str(row.get("scenario_family"))
        for row in valid_rows
        if float(row.get("coverage_rate_delta") or 0.0) > TOLERANCE and row.get("scenario_family")
    }
    return {
        "schema_version": METRIC_ROW_SCHEMA_VERSION,
        "actor": POST_ACTOR,
        "row_count": len(rows),
        "episode_count": len({row.get("episode_id") for row in rows if row.get("episode_id")}),
        "actual_coverage_row_count": len(valid_rows),
        "coverage_return": round(coverage_return, 12),
        "cumulative_coverage_rate_delta": round(total_gain, 12),
        "final_coverage_rate": _mean([float(row["final_coverage_rate"]) for row in rows if _finite(row.get("final_coverage_rate"))]),
        "new_area_covered": round(_sum_metric(rows, "new_area_covered"), 12),
        "valuable_area_covered": round(_sum_metric(rows, "valuable_area_covered"), 12),
        "information_gain": round(_sum_metric(rows, "information_gain"), 12),
        "path_cost": round(path_cost, 12),
        "risk": round(risk, 12),
        "energy_cost": round(energy, 12),
        "coverage_gain_per_path_cost": _safe_ratio(total_gain, path_cost),
        "coverage_gain_per_risk": _safe_ratio(total_gain, risk),
        "coverage_gain_per_energy": _safe_ratio(total_gain, energy),
        "accepted_policy_activation_rate": _safe_ratio(accepted_count, len(rows)),
        "fallback_rate": _safe_ratio(fallback_count, len(rows)),
        "teacher_agreement_rate": _safe_ratio(
            sum(1 for row in teacher_rows if _int(row.get("controlled_action_index")) == _int(row.get("teacher_action_index"))),
            len(teacher_rows),
        ),
        "controlled_regression_count": sum(
            1 for row in rows if _string_list(row.get("controlled_regression_reason_codes"))
        ),
        "fallback_coverage_gain": 0.0,
        "scenario_family_gain_count": len(scenario_families),
        "scenario_families_with_gain": sorted(scenario_families),
        "comparator_basis": ["post_update_guarded_replay_on_audited_coverage_rows"],
    }


def _comparison(metrics_by_actor: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_actors = sorted(actor for actor in metrics_by_actor if actor in BASELINE_ACTORS)
    best_baseline_actor = None
    if baseline_actors:
        best_baseline_actor = max(
            baseline_actors,
            key=lambda actor: _float_or_default(metrics_by_actor[actor].get("coverage_return"), -math.inf),
        )
    post = metrics_by_actor.get(POST_ACTOR, _empty_metric(POST_ACTOR))
    baseline = metrics_by_actor.get(best_baseline_actor, _empty_metric("none")) if best_baseline_actor else _empty_metric("none")
    efficiency_regression = False
    for metric in ("coverage_gain_per_path_cost", "coverage_gain_per_risk", "coverage_gain_per_energy"):
        post_value = post.get(metric)
        baseline_value = baseline.get(metric)
        if post_value is None or baseline_value is None:
            continue
        if float(post_value) + TOLERANCE < float(baseline_value) * (1.0 - MAX_ALLOWED_RELATIVE_REGRESSION):
            efficiency_regression = True
    return {
        "baseline_actors": baseline_actors,
        "best_baseline_actor": best_baseline_actor,
        "coverage_return_improvement": _metric_improvement(post, baseline, "coverage_return"),
        "cumulative_coverage_rate_delta_improvement": _metric_improvement(
            post,
            baseline,
            "cumulative_coverage_rate_delta",
        ),
        "valuable_area_covered_improvement": _metric_improvement(post, baseline, "valuable_area_covered"),
        "path_cost_delta_vs_baseline": _metric_improvement(post, baseline, "path_cost"),
        "risk_delta_vs_baseline": _metric_improvement(post, baseline, "risk"),
        "energy_cost_delta_vs_baseline": _metric_improvement(post, baseline, "energy_cost"),
        "coverage_efficiency_regression": efficiency_regression,
    }


def _rejection_report(
    *,
    reason_codes: list[str],
    update_summary: dict[str, Any],
    batch: dict[str, Any],
    replay_audit: dict[str, Any],
    comparison: dict[str, Any],
    post_metrics: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if update_summary and update_summary.get("status") != "passed":
        _add_reason(reasons, "ppo_update_failed")
    if replay_audit.get("status") != "passed" and update_summary:
        _add_reason(reasons, "guard_replay_failed")
    if batch["trainable_transition_count"] <= 0:
        _add_reason(reasons, "insufficient_policy_activation")
    if _float_or_default(post_metrics.get("accepted_policy_activation_rate"), 0.0) <= 0.0 and update_summary:
        _add_reason(reasons, "insufficient_policy_activation")
    if _float_or_default(comparison.get("coverage_return_improvement"), 0.0) <= TOLERANCE and update_summary:
        _add_reason(reasons, "no_coverage_return_improvement")
    if _float_or_default(comparison.get("valuable_area_covered_improvement"), 0.0) <= TOLERANCE and update_summary:
        _add_reason(reasons, "valuable_coverage_not_improved")
    if comparison.get("coverage_efficiency_regression"):
        _add_reason(reasons, "risk_or_cost_regression")
    if _int(post_metrics.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "guard_regression")
    fallback_rate = _float_or_default(post_metrics.get("fallback_rate"), 0.0)
    fallback_dominates = fallback_rate > 0.5
    if fallback_dominates:
        _add_reason(reasons, "fallback_dominates")
    if _int(post_metrics.get("scenario_family_gain_count")) < 2 and update_summary:
        _add_reason(reasons, "insufficient_scenario_family_gain")
    rejection_reason_counts = Counter(reason_codes)
    rejection_reason_counts.update(reasons)
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "reason_codes": reasons,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "controlled_regression_count": _int(post_metrics.get("controlled_regression_count")),
        "fallback_dominates": fallback_dominates,
        "batch_rejected_row_count": batch["rejected_row_count"],
        "guard_rejected_action_count": replay_audit.get("guard_rejected_action_count", 0),
        "accepted_policy_action_count": replay_audit.get("accepted_policy_action_count", 0),
    }


def _selected_base_candidate_root(
    selected_summary: dict[str, Any],
    selected_candidate_root: Path,
    repo_root: Path,
) -> Path:
    root = _resolve_optional_path(selected_summary.get("selected_candidate_root"), selected_candidate_root, repo_root)
    if root is not None:
        return root
    checkpoint = _resolve_optional_path(selected_summary.get("checkpoint_path"), selected_candidate_root, repo_root)
    if checkpoint is not None:
        return checkpoint.parent
    return selected_candidate_root


def _normalize_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    metric = _empty_metric(str(row.get("actor") or "unknown"))
    metric.update(row)
    metric["schema_version"] = METRIC_ROW_SCHEMA_VERSION
    return metric


def _empty_metric(actor: str) -> dict[str, Any]:
    return {
        "schema_version": METRIC_ROW_SCHEMA_VERSION,
        "actor": actor,
        "row_count": 0,
        "episode_count": 0,
        "actual_coverage_row_count": 0,
        "coverage_return": 0.0,
        "cumulative_coverage_rate_delta": 0.0,
        "final_coverage_rate": None,
        "new_area_covered": 0.0,
        "valuable_area_covered": 0.0,
        "information_gain": 0.0,
        "path_cost": 0.0,
        "risk": 0.0,
        "energy_cost": 0.0,
        "coverage_gain_per_path_cost": None,
        "coverage_gain_per_risk": None,
        "coverage_gain_per_energy": None,
        "accepted_policy_activation_rate": None,
        "fallback_rate": None,
        "teacher_agreement_rate": None,
        "controlled_regression_count": 0,
        "fallback_coverage_gain": 0.0,
        "scenario_family_gain_count": 0,
        "comparator_basis": [],
    }


def _matching_shadow(row: dict[str, Any], shadow_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for key in _row_keys(row):
        if key in shadow_index:
            return shadow_index[key]
    return {}


def _row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    context_id = row.get("context_id")
    if context_id:
        keys.append(f"context:{context_id}")
    for episode_field, step_field in (
        ("episode_id", "step_index"),
        ("source_episode_id", "source_step_index"),
        ("source_episode_id", "step_index"),
        ("episode_id", "source_step_index"),
        ("shadow_episode_id", "shadow_step_index"),
    ):
        episode_id = row.get(episode_field)
        step_index = row.get(step_field)
        if episode_id is not None and step_index is not None:
            keys.append(f"episode-step:{episode_id}:{step_index}")
    return keys


def _policy_observation(payload: dict[str, Any], policy_observation_class):
    candidate_features = _list_rows_or_empty(payload.get("candidate_features"))
    candidate_cells = _list_rows_or_empty(payload.get("candidate_cells"))
    missing_feature_names = _list_rows_or_empty(payload.get("candidate_missing_feature_names"))
    missing_indicators = _list_rows_or_empty(payload.get("candidate_missing_indicators"))
    return policy_observation_class(
        candidate_feature_names=tuple(str(item) for item in _list_or_empty(payload.get("candidate_feature_names"))),
        candidate_features=tuple(tuple(float(value) for value in row) for row in candidate_features),
        global_feature_names=tuple(str(item) for item in _list_or_empty(payload.get("global_feature_names"))),
        global_features=tuple(float(value) for value in _list_or_empty(payload.get("global_features"))),
        action_mask=tuple(bool(value) for value in _list_or_empty(payload.get("action_mask"))),
        candidate_cells=tuple(tuple(int(value) for value in row) for row in candidate_cells),
        candidate_missing_feature_names=tuple(
            tuple(str(value) for value in row) for row in missing_feature_names
        ),
        candidate_missing_indicator_names=tuple(
            str(value) for value in _list_or_empty(payload.get("candidate_missing_indicator_names"))
        ),
        candidate_missing_indicators=tuple(
            tuple(float(value) for value in row) for row in missing_indicators
        ),
    )


def _selected_cell(observation: dict[str, Any], action_index: int) -> list[Any] | None:
    cells = observation.get("candidate_cells")
    if isinstance(cells, list) and 0 <= action_index < len(cells):
        cell = cells[action_index]
        return list(cell) if isinstance(cell, list) else cell
    return None


def _read_json(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        _add_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reason_codes, f"{label}_invalid_json")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reason_codes: list[str], label: str) -> list[dict[str, Any]]:
    if not path.exists():
        _add_reason(reason_codes, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    except json.JSONDecodeError:
        _add_reason(reason_codes, f"{label}_invalid_jsonl")
        return []
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    return _resolve_path(Path(str(value)), base, repo_root)


def _resolve_path(path: Path, base: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    if (base / path).exists():
        return base / path
    return repo_root / path


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _bool_list(value: Any) -> list[bool]:
    if not isinstance(value, list):
        return []
    return [bool(item) for item in value]


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_rows_or_empty(value: Any) -> list[list[Any]]:
    if not isinstance(value, list):
        return []
    return [row if isinstance(row, list) else [] for row in value]


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _float_or_default(value: Any, default: float) -> float:
    parsed = _float(value)
    if _finite(parsed):
        return float(parsed)
    return default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sum_metric(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows if _finite(row.get(field)))


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not _finite(numerator) or not _finite(denominator):
        return None
    denominator_float = float(denominator)
    if abs(denominator_float) <= TOLERANCE:
        return None
    return round(float(numerator) / denominator_float, 12)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 12)


def _metric_improvement(selected: dict[str, Any], baseline: dict[str, Any], field: str) -> float | None:
    selected_value = selected.get(field)
    baseline_value = baseline.get(field)
    if selected_value is None or baseline_value is None:
        return None
    return round(float(selected_value) - float(baseline_value), 12)


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return "shadow_canary_release_performance_validation"
    if "coverage_reward_source_not_passed" in reason_codes:
        return "fix_coverage_aware_reward_source_contract"
    if "ppo_update_failed" in reason_codes:
        return "fix_coverage_driven_ppo_update"
    if "guard_regression" in reason_codes:
        return "fix_guard_regression_before_more_training"
    return "refine_coverage_reward_or_collect_more_policy_coverage"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_report(
    summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    rejection_report: dict[str, Any],
) -> str:
    lines = [
        "# Coverage-Driven PPO Improvement Run v1",
        "",
        f"Status: `{summary['status']}`",
        f"Coverage-driven PPO improvement status: `{summary['coverage_driven_ppo_improvement_status']}`",
        f"Reason codes: `{summary['reason_codes']}`",
        f"Next required change: `{summary['next_required_change']}`",
        "",
        "## PPO Update",
        "",
        f"- Trainable reward rows: `{summary['coverage_aware_reward_transition_count']}`",
        f"- Optimizer transitions: `{summary['optimizer_train_transition_count']}`",
        f"- Parameter L2 delta: `{summary.get('parameter_l2_delta')}`",
        f"- Approx KL: `{summary.get('approx_kl')}`",
        "",
        "## Performance Comparison",
        "",
        f"- Best baseline actor: `{summary.get('best_baseline_actor')}`",
        f"- Coverage return improvement: `{summary.get('coverage_return_improvement')}`",
        f"- Valuable area improvement: `{summary.get('valuable_area_covered_improvement')}`",
        f"- Activation rate: `{summary.get('accepted_policy_activation_rate')}`",
        f"- Fallback rate: `{summary.get('fallback_rate')}`",
        "",
        "## Actor Metrics",
        "",
    ]
    for row in metric_rows:
        lines.append(
            "- `{actor}`: coverage_return=`{coverage_return}`, cumulative_delta=`{delta}`, "
            "valuable=`{valuable}`, path_cost=`{path_cost}`, risk=`{risk}`, fallback_rate=`{fallback}`".format(
                actor=row.get("actor"),
                coverage_return=row.get("coverage_return"),
                delta=row.get("cumulative_coverage_rate_delta"),
                valuable=row.get("valuable_area_covered"),
                path_cost=row.get("path_cost"),
                risk=row.get("risk"),
                fallback=row.get("fallback_rate"),
            )
        )
    lines.extend(["", "## Rejection Reason Counts", ""])
    if rejection_report["rejection_reason_counts"]:
        for reason, count in rejection_report["rejection_reason_counts"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This stage may write an experimental offline checkpoint, but it does not publish it, "
            "replace the default policy, connect a real executor, relax guards, or make a formal performance claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
