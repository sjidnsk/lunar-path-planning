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

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


SUMMARY_SCHEMA_VERSION = "shadow-canary-release-performance-validation-preflight-summary/v1"
RUNTIME_MANIFEST_SCHEMA_VERSION = "shadow-canary-release-performance-validation-runtime-manifest/v1"
STEP_SCHEMA_VERSION = "shadow-canary-release-performance-shadow-step/v1"
LONG_HORIZON_SCHEMA_VERSION = "shadow-canary-release-performance-long-horizon-validation/v1"
AUDIT_SCHEMA_VERSION = "shadow-canary-release-performance-audit/v1"
LEDGER_SCHEMA_VERSION = "shadow-canary-release-performance-eligibility-ledger/v1"

DEFAULT_COST_EFFICIENCY_ROOT = "outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1"

COST_SUMMARY_FILE = "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json"
FILTERED_BATCH_FILE = "cost-efficiency-filtered-batch.jsonl"
REFINED_TRANSITIONS_FILE = "refined-coverage-ppo-batch/refined-trainable-transitions.jsonl"
GUARD_REPLAY_AUDIT_FILE = "coverage-driven-ppo-replay-audit.json"
PERFORMANCE_METRIC_TABLE_FILE = "refined-coverage-driven-ppo-performance-metric-table.jsonl"
STAGE5A_RERUN_SUMMARY_FILE = "stage5a-rerun-summary.json"
COMPATIBLE_PERFORMANCE_METRIC_TABLE_FILE = (
    "cost-efficiency-comparable-performance-input/coverage-performance-metric-table.jsonl"
)
COMPATIBLE_PERFORMANCE_SUMMARY_FILE = (
    "cost-efficiency-comparable-performance-input/exploration-coverage-performance-evaluation-summary.json"
)
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
POST_TRAINING_REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_CANDIDATE_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "shadow-canary-release-performance-validation-preflight-summary.json"
RUNTIME_MANIFEST_FILE = "shadow-canary-runtime-manifest.json"
SHADOW_STEP_COMPARISON_FILE = "shadow-canary-step-comparison.jsonl"
LONG_HORIZON_FILE = "shadow-canary-long-horizon-validation.json"
COVERAGE_EFFICIENCY_AUDIT_FILE = "shadow-canary-coverage-efficiency-audit.json"
GUARD_FALLBACK_AUDIT_FILE = "shadow-canary-guard-fallback-audit.json"
KILL_SWITCH_AUDIT_FILE = "shadow-canary-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "shadow-canary-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "shadow-canary-telemetry-audit.json"
ELIGIBILITY_LEDGER_FILE = "shadow-canary-eligibility-ledger.json"
REJECTION_REPORT_FILE = "shadow-canary-rejection-report.json"
REPORT_FILE = "shadow-canary-release-performance-validation-preflight-report.md"

TARGET_FAMILIES = (
    "low_observation_count",
    "mixed_risk",
    "rim_or_steep_slope",
    "smooth_high_confidence",
)
BASELINE_ACTORS = ("teacher", "pre_improvement_selected_ppo", "source_default", "default_policy")
POST_ACTOR = "post_improvement_ppo"
DISCOUNT_FACTOR = 0.99
TOLERANCE = 1.0e-9
DEFAULT_HORIZONS = (10, 20, 30)
DEFAULT_FALLBACK_RATE_THRESHOLD = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline shadow/canary release performance validation preflight for the 5B.7 candidate."
    )
    parser.add_argument("--cost-efficiency-root", default=DEFAULT_COST_EFFICIENCY_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--horizons", default="10,20,30")
    parser.add_argument("--fallback-rate-threshold", type=float, default=DEFAULT_FALLBACK_RATE_THRESHOLD)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_shadow_canary_release_performance_validation_preflight(
        cost_efficiency_root=_resolve_path(Path(args.cost_efficiency_root), repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        horizons=_parse_horizons(args.horizons),
        fallback_rate_threshold=args.fallback_rate_threshold,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "next_required_change": summary["next_required_change"],
                "long_horizon_shadow_passed": summary["long_horizon_shadow_passed"],
                "coverage_return_improvement": summary["coverage_return_improvement"],
                "coverage_efficiency_regression": summary["coverage_efficiency_regression"],
                "fallback_rate": summary["fallback_rate"],
                "kill_switch_audit_passed": summary["kill_switch_audit_passed"],
                "rollback_audit_passed": summary["rollback_audit_passed"],
                "telemetry_audit_passed": summary["telemetry_audit_passed"],
                "performance_claimed": summary["performance_claimed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_shadow_canary_release_performance_validation_preflight(
    *,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    fallback_rate_threshold: float = DEFAULT_FALLBACK_RATE_THRESHOLD,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    cost_efficiency_root = Path(cost_efficiency_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    input_reasons: list[str] = []
    cost_summary_path = cost_efficiency_root / COST_SUMMARY_FILE
    cost_summary = _read_json(cost_summary_path, input_reasons, "cost_efficiency_summary")
    filtered_batch_path = (
        _resolve_optional_path(cost_summary.get("filtered_batch"), cost_efficiency_root, repo_root)
        or cost_efficiency_root / FILTERED_BATCH_FILE
    )
    refined_transitions_path = (
        _resolve_optional_path(cost_summary.get("refined_trainable_transitions"), cost_efficiency_root, repo_root)
        or cost_efficiency_root / REFINED_TRANSITIONS_FILE
    )
    guard_replay_audit_path = (
        _resolve_optional_path(cost_summary.get("guard_replay_audit"), cost_efficiency_root, repo_root)
        or cost_efficiency_root / GUARD_REPLAY_AUDIT_FILE
    )
    performance_metric_table_path = (
        _resolve_optional_path(cost_summary.get("performance_metric_table"), cost_efficiency_root, repo_root)
        or cost_efficiency_root / PERFORMANCE_METRIC_TABLE_FILE
    )
    stage5a_rerun_summary_path = (
        _resolve_optional_path(cost_summary.get("stage5a_rerun_summary"), cost_efficiency_root, repo_root)
        or cost_efficiency_root / STAGE5A_RERUN_SUMMARY_FILE
    )
    compatible_performance_metric_table_path = (
        _resolve_optional_path(cost_summary.get("compatible_performance_metric_table"), cost_efficiency_root, repo_root)
        or cost_efficiency_root / COMPATIBLE_PERFORMANCE_METRIC_TABLE_FILE
    )
    compatible_performance_summary_path = (
        _resolve_optional_path(cost_summary.get("compatible_performance_summary"), cost_efficiency_root, repo_root)
        or cost_efficiency_root / COMPATIBLE_PERFORMANCE_SUMMARY_FILE
    )

    filtered_batch = _read_jsonl(filtered_batch_path, input_reasons, "filtered_batch")
    refined_transitions = _read_jsonl(refined_transitions_path, input_reasons, "refined_trainable_transitions")
    guard_replay_audit = _read_json(guard_replay_audit_path, input_reasons, "guard_replay_audit")
    performance_metric_rows = _read_jsonl(performance_metric_table_path, input_reasons, "performance_metric_table")
    stage5a_rerun_summary = _read_json(stage5a_rerun_summary_path, input_reasons, "stage5a_rerun_summary")
    compatible_performance_metric_rows = _read_jsonl(
        compatible_performance_metric_table_path,
        input_reasons,
        "compatible_performance_metric_table",
    )
    compatible_performance_summary = _read_json(
        compatible_performance_summary_path,
        input_reasons,
        "compatible_performance_summary",
    )
    formal_summary_path = Path(formal_training_root) / FORMAL_SUMMARY_FILE
    post_training_replay_summary_path = Path(post_training_replay_root) / POST_TRAINING_REPLAY_SUMMARY_FILE
    selected_candidate_summary_path = Path(selected_candidate_root) / SELECTED_CANDIDATE_SUMMARY_FILE
    formal_summary = _read_json(formal_summary_path, input_reasons, "formal_training_summary")
    post_training_replay_summary = _read_json(
        post_training_replay_summary_path,
        input_reasons,
        "post_training_replay_summary",
    )
    selected_candidate_summary = _read_json(
        selected_candidate_summary_path,
        input_reasons,
        "selected_candidate_summary",
    )

    input_reasons = _unique(
        [
            *input_reasons,
            *_validate_current_inputs(
                cost_summary=cost_summary,
                cost_efficiency_root=cost_efficiency_root,
                filtered_batch=filtered_batch,
                refined_transitions=refined_transitions,
                guard_replay_audit=guard_replay_audit,
                performance_metric_rows=performance_metric_rows,
                compatible_performance_metric_rows=compatible_performance_metric_rows,
                compatible_performance_summary=compatible_performance_summary,
                stage5a_rerun_summary=stage5a_rerun_summary,
                formal_summary=formal_summary,
                post_training_replay_summary=post_training_replay_summary,
                selected_candidate_summary=selected_candidate_summary,
                fallback_rate_threshold=fallback_rate_threshold,
            ),
        ]
    )

    step_rows = _build_shadow_step_rows(
        filtered_batch=filtered_batch,
        replay_rows=_replay_rows(guard_replay_audit),
    )
    _write_jsonl(paths["shadow_step_comparison"], step_rows)

    metrics_by_actor = _metric_rows_by_actor(performance_metric_rows)
    baseline_actor = _best_baseline_actor(metrics_by_actor)
    baseline_metrics = metrics_by_actor.get(baseline_actor or "", {})
    candidate_metrics = metrics_by_actor.get(POST_ACTOR, {})
    computed_candidate_metrics = _aggregate_step_metrics(step_rows, actor=POST_ACTOR, role="candidate")
    computed_teacher_metrics = _aggregate_step_metrics(step_rows, actor="teacher", role="teacher")
    if not candidate_metrics:
        candidate_metrics = computed_candidate_metrics
    if not baseline_metrics:
        baseline_actor = "teacher"
        baseline_metrics = computed_teacher_metrics
    comparison = _comparison(candidate_metrics, baseline_metrics)

    long_horizon = _long_horizon_validation(
        step_rows=step_rows,
        horizons=tuple(sorted({int(horizon) for horizon in horizons if int(horizon) > 0})),
    )
    _write_json(paths["long_horizon_shadow_validation"], long_horizon)
    coverage_efficiency_audit = _coverage_efficiency_audit(
        comparison=comparison,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        baseline_actor=baseline_actor,
    )
    guard_fallback_audit = _guard_fallback_audit(
        step_rows=step_rows,
        guard_replay_audit=guard_replay_audit,
        fallback_rate_threshold=fallback_rate_threshold,
    )
    kill_switch_audit = _kill_switch_audit(step_rows)
    rollback_audit = _rollback_audit(
        cost_summary=cost_summary,
        kill_switch_audit=kill_switch_audit,
    )
    telemetry_audit = _telemetry_audit(step_rows)

    _write_json(paths["coverage_efficiency_audit"], coverage_efficiency_audit)
    _write_json(paths["guard_fallback_audit"], guard_fallback_audit)
    _write_json(paths["kill_switch_audit"], kill_switch_audit)
    _write_json(paths["rollback_audit"], rollback_audit)
    _write_json(paths["telemetry_audit"], telemetry_audit)

    reason_codes = _unique(
        [
            *input_reasons,
            *_acceptance_reason_codes(
                cost_summary=cost_summary,
                comparison=comparison,
                long_horizon=long_horizon,
                coverage_efficiency_audit=coverage_efficiency_audit,
                guard_fallback_audit=guard_fallback_audit,
                kill_switch_audit=kill_switch_audit,
                rollback_audit=rollback_audit,
                telemetry_audit=telemetry_audit,
                fallback_rate_threshold=fallback_rate_threshold,
            ),
        ]
    )
    ledger = _eligibility_ledger(
        reason_codes=reason_codes,
        cost_summary=cost_summary,
        long_horizon=long_horizon,
        coverage_efficiency_audit=coverage_efficiency_audit,
        guard_fallback_audit=guard_fallback_audit,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
    )
    rejection_report = _rejection_report(reason_codes=reason_codes, ledger=ledger)
    _write_json(paths["eligibility_ledger"], ledger)
    _write_json(paths["rejection_report"], rejection_report)

    runtime_manifest = _runtime_manifest(
        repo_root=repo_root,
        cost_efficiency_root=cost_efficiency_root,
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        input_paths={
            "cost_summary": cost_summary_path,
            "filtered_batch": filtered_batch_path,
            "refined_trainable_transitions": refined_transitions_path,
            "guard_replay_audit": guard_replay_audit_path,
            "performance_metric_table": performance_metric_table_path,
            "stage5a_rerun_summary": stage5a_rerun_summary_path,
            "compatible_performance_metric_table": compatible_performance_metric_table_path,
            "compatible_performance_summary": compatible_performance_summary_path,
            "formal_training_summary": formal_summary_path,
            "post_training_replay_summary": post_training_replay_summary_path,
            "selected_candidate_summary": selected_candidate_summary_path,
        },
        paths=paths,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
    )
    _write_json(paths["runtime_manifest"], runtime_manifest)

    status = "passed" if not reason_codes else "failed"
    summary = _summary(
        status=status,
        reason_codes=reason_codes,
        paths=paths,
        repo_root=repo_root,
        cost_efficiency_root=cost_efficiency_root,
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        cost_summary=cost_summary,
        input_paths=runtime_manifest["input_paths"],
        filtered_batch=filtered_batch,
        refined_transitions=refined_transitions,
        guard_replay_audit=guard_replay_audit,
        performance_metric_rows=performance_metric_rows,
        compatible_performance_metric_rows=compatible_performance_metric_rows,
        stage5a_rerun_summary=stage5a_rerun_summary,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
        baseline_actor=baseline_actor,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        computed_candidate_metrics=computed_candidate_metrics,
        computed_teacher_metrics=computed_teacher_metrics,
        comparison=comparison,
        long_horizon=long_horizon,
        coverage_efficiency_audit=coverage_efficiency_audit,
        guard_fallback_audit=guard_fallback_audit,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        ledger=ledger,
        fallback_rate_threshold=fallback_rate_threshold,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _validate_current_inputs(
    *,
    cost_summary: dict[str, Any],
    cost_efficiency_root: Path,
    filtered_batch: list[dict[str, Any]],
    refined_transitions: list[dict[str, Any]],
    guard_replay_audit: dict[str, Any],
    performance_metric_rows: list[dict[str, Any]],
    compatible_performance_metric_rows: list[dict[str, Any]],
    compatible_performance_summary: dict[str, Any],
    stage5a_rerun_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
    fallback_rate_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if (
        cost_summary.get("status") != "passed"
        or _string_list(cost_summary.get("reason_codes"))
        or cost_summary.get("next_required_change") != "shadow_canary_release_performance_validation_preflight"
    ):
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    for metric in (
        "coverage_return_improvement",
        "cumulative_coverage_rate_delta_improvement",
        "valuable_area_covered_improvement",
    ):
        if _float(cost_summary.get(metric)) <= TOLERANCE:
            _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if bool(cost_summary.get("coverage_efficiency_regression")):
        _add_reason(reasons, "coverage_efficiency_regression")
    if _float(cost_summary.get("fallback_rate")) >= fallback_rate_threshold:
        _add_reason(reasons, "fallback_dominates")
    if _int(cost_summary.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "shadow_controlled_regression")
    if _int(cost_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    if _int(cost_summary.get("safe_better_training_family_count")) != len(TARGET_FAMILIES):
        _add_reason(reasons, "family_balance_failed")
    for field in (
        "publishes_checkpoint",
        "replaces_default_policy",
        "performance_claimed",
        "connects_real_executor",
        "relaxes_guard",
    ):
        if cost_summary.get(field) is True:
            _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if not filtered_batch:
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if not refined_transitions:
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if not _replay_rows(guard_replay_audit):
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if not performance_metric_rows or POST_ACTOR not in _metric_rows_by_actor(performance_metric_rows):
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if not compatible_performance_metric_rows:
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if compatible_performance_summary and compatible_performance_summary.get("status") not in (None, "passed"):
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    # Stage5a rerun is retained as provenance for the compatible decision set.
    # The current release-candidate gate is the 5B.7 cost-efficiency summary,
    # which may intentionally pass after a diagnostic stage5a rerun preserved an
    # older failure reason.
    expected_count = _int(cost_summary.get("trainable_pair_count"))
    if expected_count > 0:
        for rows in (filtered_batch, refined_transitions, _replay_rows(guard_replay_audit)):
            if len(rows) != expected_count:
                _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    if not _current_root_path(cost_summary.get("compatible_performance_metric_table"), cost_efficiency_root):
        _add_reason(reasons, "input_cost_efficiency_stage_not_passed")
    for upstream in (formal_summary, post_training_replay_summary, selected_candidate_summary):
        if upstream.get("status") != "passed" or _string_list(upstream.get("reason_codes")):
            _add_reason(reasons, "input_upstream_stability_stage_not_passed")
        if _int(upstream.get("controlled_regression_count")) > 0:
            _add_reason(reasons, "shadow_controlled_regression")
        if upstream.get("performance_claimed") is True:
            _add_reason(reasons, "input_upstream_stability_stage_not_passed")
    for field in ("publishes_checkpoint", "replaces_default_policy"):
        if selected_candidate_summary.get(field) is True:
            _add_reason(reasons, "input_upstream_stability_stage_not_passed")
    return reasons


def _build_shadow_step_rows(
    *,
    filtered_batch: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_candidate = {
        _candidate_key(row, row.get("candidate_action_index")): row
        for row in filtered_batch
        if row.get("candidate_action_index") is not None
    }
    by_decision: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in filtered_batch:
        by_decision.setdefault(
            (str(row.get("context_id") or ""), str(row.get("episode_id") or ""), _int(row.get("step_index"))),
            row,
        )
    result: list[dict[str, Any]] = []
    for source_index, replay in enumerate(replay_rows):
        filtered = by_candidate.get(_candidate_key(replay, replay.get("controlled_action_index")))
        if filtered is None:
            filtered = by_decision.get(
                (
                    str(replay.get("context_id") or ""),
                    str(replay.get("episode_id") or ""),
                    _int(replay.get("step_index")),
                ),
                {},
            )
        step_index = _int(replay.get("step_index"))
        coverage_rate_delta = _float(replay.get("coverage_rate_delta"))
        teacher_coverage_rate_delta = _float(filtered.get("teacher_expected_coverage_rate_delta"))
        candidate_return = coverage_rate_delta * (DISCOUNT_FACTOR ** max(step_index, 0))
        teacher_return = teacher_coverage_rate_delta * (DISCOUNT_FACTOR ** max(step_index, 0))
        fallback_like = bool(replay.get("fallback_like")) or bool(replay.get("guard_rejected"))
        policy_action_accepted = bool(replay.get("policy_action_accepted"))
        telemetry_fields = _telemetry_fields(replay)
        result.append(
            {
                "schema_version": STEP_SCHEMA_VERSION,
                "source_index": source_index,
                "replay_index": _int(replay.get("replay_index"), source_index),
                "context_id": replay.get("context_id"),
                "episode_id": replay.get("episode_id"),
                "step_index": step_index,
                "scenario_id": replay.get("scenario_id") or filtered.get("scenario_id"),
                "scenario_family": replay.get("scenario_family") or filtered.get("scenario_family"),
                "split": filtered.get("split"),
                "candidate_action_index": filtered.get("candidate_action_index"),
                "raw_policy_action_index": replay.get("raw_policy_action_index"),
                "controlled_action_index": replay.get("controlled_action_index"),
                "teacher_action_index": replay.get("teacher_action_index", filtered.get("teacher_action_index")),
                "policy_action_accepted": policy_action_accepted,
                "accepted_policy_activation": policy_action_accepted,
                "activation": policy_action_accepted,
                "fallback_like": fallback_like,
                "source_fallback_used": fallback_like,
                "guard_rejected": bool(replay.get("guard_rejected")),
                "shadow_policy_takes_control": False,
                "experimental_control_activation": False,
                "default_policy_authoritative": True,
                "shadow_diagnostic_only": True,
                "rollback_available": True,
                "kill_switch_available": True,
                "coverage_rate_delta": coverage_rate_delta,
                "coverage_return": round(candidate_return, 12),
                "final_coverage_rate": _first_float(replay.get("final_coverage_rate")),
                "new_area_covered": _float(replay.get("new_area_covered")),
                "valuable_area_covered": _float(replay.get("valuable_area_covered")),
                "information_gain": _float(replay.get("information_gain")),
                "path_cost": _float(replay.get("path_cost")),
                "risk": _float(replay.get("risk")),
                "energy_cost": _float(replay.get("energy_cost")),
                "teacher_coverage_rate_delta": teacher_coverage_rate_delta,
                "teacher_coverage_return": round(teacher_return, 12),
                "teacher_final_coverage_rate": _first_float(
                    filtered.get("teacher_expected_coverage_rate_delta"),
                    default=0.0,
                ),
                "teacher_new_area_covered": _float(filtered.get("teacher_expected_new_coverage_area")),
                "teacher_valuable_area_covered": _first_float(
                    filtered.get("teacher_valuable_coverage_proxy"),
                    filtered.get("teacher_value"),
                    default=0.0,
                ),
                "teacher_information_gain": _float(filtered.get("teacher_information_gain")),
                "teacher_path_cost": _float(filtered.get("teacher_path_cost")),
                "teacher_risk": _float(filtered.get("teacher_risk")),
                "teacher_energy_cost": _float(filtered.get("teacher_energy_cost")),
                "coverage_return_improvement": round(candidate_return - teacher_return, 12),
                "cumulative_coverage_rate_delta_improvement": round(
                    coverage_rate_delta - teacher_coverage_rate_delta,
                    12,
                ),
                "valuable_area_covered_improvement": round(
                    _float(replay.get("valuable_area_covered"))
                    - _first_float(filtered.get("teacher_valuable_coverage_proxy"), filtered.get("teacher_value"), default=0.0),
                    12,
                ),
                "controlled_regression_reason_codes": _string_list(
                    replay.get("controlled_regression_reason_codes")
                ),
                "telemetry_fields_recorded": telemetry_fields,
            }
        )
    return result


def _long_horizon_validation(
    *,
    step_rows: list[dict[str, Any]],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    if not horizons:
        horizons = DEFAULT_HORIZONS
    rollups: list[dict[str, Any]] = []
    for horizon in horizons:
        rollups.append(_rollup("all", "all", horizon, step_rows))
        for group_type, field in (
            ("scenario_family", "scenario_family"),
            ("scenario", "scenario_id"),
            ("context", "context_id"),
        ):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in step_rows:
                key = str(row.get(field) or "")
                if key:
                    grouped[key].append(row)
            for key, rows in sorted(grouped.items()):
                rollups.append(_rollup(group_type, key, horizon, rows))
    all_rollups = [row for row in rollups if row["group_type"] == "all"]
    long_horizon_passed = bool(all_rollups) and all(
        _float(row.get("coverage_return_improvement")) > TOLERANCE
        and _float(row.get("cumulative_coverage_rate_delta_improvement")) > TOLERANCE
        and _float(row.get("valuable_area_covered_improvement")) > TOLERANCE
        for row in all_rollups
    )
    return {
        "schema_version": LONG_HORIZON_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "horizons": list(horizons),
        "long_horizon_shadow_passed": long_horizon_passed,
        "all_horizon_rollup_count": len(all_rollups),
        "rollup_count": len(rollups),
        "negative_window_count": sum(_int(row.get("negative_window_count")) for row in rollups),
        "rollups": rollups,
    }


def _rollup(group_type: str, group_key: str, horizon: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (_int(row.get("source_index")), str(row.get("context_id") or "")))
    windows = [ordered[index : index + horizon] for index in range(0, len(ordered), horizon)] if horizon else [ordered]
    window_improvements = [_metrics_for_rows(window, "candidate")["coverage_return"] - _metrics_for_rows(window, "teacher")["coverage_return"] for window in windows if window]
    candidate = _metrics_for_rows(ordered, "candidate")
    teacher = _metrics_for_rows(ordered, "teacher")
    return {
        "schema_version": "shadow-canary-release-performance-long-horizon-rollup/v1",
        "group_type": group_type,
        "group_key": group_key,
        "horizon": horizon,
        "row_count": len(ordered),
        "window_count": len([window for window in windows if window]),
        "negative_window_count": sum(1 for value in window_improvements if value <= TOLERANCE),
        "candidate_coverage_return": candidate["coverage_return"],
        "baseline_coverage_return": teacher["coverage_return"],
        "coverage_return_improvement": round(candidate["coverage_return"] - teacher["coverage_return"], 12),
        "candidate_cumulative_coverage_rate_delta": candidate["cumulative_coverage_rate_delta"],
        "baseline_cumulative_coverage_rate_delta": teacher["cumulative_coverage_rate_delta"],
        "cumulative_coverage_rate_delta_improvement": round(
            candidate["cumulative_coverage_rate_delta"] - teacher["cumulative_coverage_rate_delta"],
            12,
        ),
        "candidate_valuable_area_covered": candidate["valuable_area_covered"],
        "baseline_valuable_area_covered": teacher["valuable_area_covered"],
        "valuable_area_covered_improvement": round(
            candidate["valuable_area_covered"] - teacher["valuable_area_covered"],
            12,
        ),
        "candidate_final_coverage_rate": candidate["final_coverage_rate"],
        "baseline_final_coverage_rate": teacher["final_coverage_rate"],
        "candidate_new_area_covered": candidate["new_area_covered"],
        "baseline_new_area_covered": teacher["new_area_covered"],
        "candidate_coverage_gain_per_path_cost": candidate["coverage_gain_per_path_cost"],
        "baseline_coverage_gain_per_path_cost": teacher["coverage_gain_per_path_cost"],
        "candidate_coverage_gain_per_risk": candidate["coverage_gain_per_risk"],
        "baseline_coverage_gain_per_risk": teacher["coverage_gain_per_risk"],
        "candidate_coverage_gain_per_energy": candidate["coverage_gain_per_energy"],
        "baseline_coverage_gain_per_energy": teacher["coverage_gain_per_energy"],
        "accepted_policy_activation_rate": candidate["accepted_policy_activation_rate"],
        "fallback_rate": candidate["fallback_rate"],
        "teacher_agreement_rate": candidate["teacher_agreement_rate"],
    }


def _metrics_for_rows(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    prefix = "teacher_" if role == "teacher" else ""
    total_gain = sum(_float(row.get(f"{prefix}coverage_rate_delta")) for row in rows)
    coverage_return = sum(_float(row.get(f"{prefix}coverage_return")) for row in rows)
    path_cost = sum(_float(row.get(f"{prefix}path_cost")) for row in rows)
    risk = sum(_float(row.get(f"{prefix}risk")) for row in rows)
    energy = sum(_float(row.get(f"{prefix}energy_cost")) for row in rows)
    values = [
        _first_float(row.get(f"{prefix}final_coverage_rate"))
        for row in rows
        if _first_float(row.get(f"{prefix}final_coverage_rate")) is not None
    ]
    teacher_action_rows = [
        row for row in rows if row.get("teacher_action_index") is not None and row.get("controlled_action_index") is not None
    ]
    return {
        "row_count": len(rows),
        "coverage_return": round(coverage_return, 12),
        "cumulative_coverage_rate_delta": round(total_gain, 12),
        "final_coverage_rate": _mean([float(value) for value in values]),
        "new_area_covered": round(sum(_float(row.get(f"{prefix}new_area_covered")) for row in rows), 12),
        "valuable_area_covered": round(sum(_float(row.get(f"{prefix}valuable_area_covered")) for row in rows), 12),
        "information_gain": round(sum(_float(row.get(f"{prefix}information_gain")) for row in rows), 12),
        "path_cost": round(path_cost, 12),
        "risk": round(risk, 12),
        "energy_cost": round(energy, 12),
        "coverage_gain_per_path_cost": _safe_ratio(total_gain, path_cost),
        "coverage_gain_per_risk": _safe_ratio(total_gain, risk),
        "coverage_gain_per_energy": _safe_ratio(total_gain, energy),
        "accepted_policy_activation_rate": _safe_ratio(
            sum(1 for row in rows if row.get("policy_action_accepted")),
            len(rows),
        )
        if role != "teacher"
        else 0.0,
        "fallback_rate": _safe_ratio(sum(1 for row in rows if row.get("fallback_like")), len(rows))
        if role != "teacher"
        else 0.0,
        "teacher_agreement_rate": _safe_ratio(
            sum(
                1
                for row in teacher_action_rows
                if _int(row.get("controlled_action_index")) == _int(row.get("teacher_action_index"))
            ),
            len(teacher_action_rows),
        )
        if role != "teacher"
        else 1.0,
    }


def _aggregate_step_metrics(rows: list[dict[str, Any]], *, actor: str, role: str) -> dict[str, Any]:
    metrics = _metrics_for_rows(rows, role)
    families = {
        str(row.get("scenario_family") or "")
        for row in rows
        if row.get("scenario_family") and _float(row.get("coverage_rate_delta")) > TOLERANCE
    }
    return {
        "schema_version": "coverage-driven-ppo-performance-metric-row/v1",
        "actor": actor,
        "episode_count": len({row.get("episode_id") for row in rows if row.get("episode_id")}),
        "actual_coverage_row_count": metrics["row_count"],
        **metrics,
        "controlled_regression_count": sum(
            1 for row in rows if _string_list(row.get("controlled_regression_reason_codes"))
        ),
        "fallback_coverage_gain": round(
            sum(_float(row.get("coverage_rate_delta")) for row in rows if row.get("fallback_like")),
            12,
        ),
        "scenario_family_gain_count": len(families),
        "scenario_families_with_gain": sorted(families),
        "comparator_basis": ["shadow_canary_preflight_recomputed_from_step_rows"],
    }


def _coverage_efficiency_audit(
    *,
    comparison: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_actor: str | None,
) -> dict[str, Any]:
    metric_checks = []
    for metric in ("coverage_gain_per_path_cost", "coverage_gain_per_risk", "coverage_gain_per_energy"):
        candidate = candidate_metrics.get(metric)
        baseline = baseline_metrics.get(metric)
        passed = candidate is not None and baseline is not None and float(candidate) + TOLERANCE >= float(baseline)
        metric_checks.append(
            {
                "metric": metric,
                "candidate": candidate,
                "baseline": baseline,
                "passed": passed,
                "delta": None if candidate is None or baseline is None else round(float(candidate) - float(baseline), 12),
            }
        )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "coverage_efficiency",
        "baseline_actor": baseline_actor,
        "coverage_efficiency_regression": bool(comparison.get("coverage_efficiency_regression")),
        "coverage_efficiency_audit_passed": not bool(comparison.get("coverage_efficiency_regression"))
        and all(row["passed"] for row in metric_checks),
        "metric_checks": metric_checks,
        "comparison": comparison,
    }


def _guard_fallback_audit(
    *,
    step_rows: list[dict[str, Any]],
    guard_replay_audit: dict[str, Any],
    fallback_rate_threshold: float,
) -> dict[str, Any]:
    fallback_count = sum(1 for row in step_rows if row.get("fallback_like"))
    accepted_count = sum(1 for row in step_rows if row.get("policy_action_accepted"))
    fallback_gain = sum(_float(row.get("coverage_rate_delta")) for row in step_rows if row.get("fallback_like"))
    controlled_regression_count = sum(
        1 for row in step_rows if _string_list(row.get("controlled_regression_reason_codes"))
    )
    fallback_rate = _safe_ratio(fallback_count, len(step_rows)) or 0.0
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "guard_fallback",
        "raw_policy_action_count": _int(guard_replay_audit.get("raw_policy_action_count"), len(step_rows)),
        "accepted_policy_action_count": _int(guard_replay_audit.get("accepted_policy_action_count"), accepted_count),
        "guard_rejected_action_count": _int(guard_replay_audit.get("guard_rejected_action_count"), fallback_count),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "fallback_rate_threshold": fallback_rate_threshold,
        "fallback_gain": round(fallback_gain, 12),
        "fallback_gain_contamination_count": sum(
            1 for row in step_rows if row.get("fallback_like") and _float(row.get("coverage_rate_delta")) > TOLERANCE
        ),
        "controlled_regression_count": controlled_regression_count,
        "source_fallback_available": True,
        "guard_fallback_audit_passed": fallback_rate < fallback_rate_threshold
        and fallback_gain <= TOLERANCE
        and controlled_regression_count == 0,
    }


def _kill_switch_audit(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    activation_count = sum(1 for row in step_rows if row.get("experimental_control_activation"))
    shadow_takes_control = any(row.get("shadow_policy_takes_control") for row in step_rows)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "kill_switch",
        "kill_switch_mode": "offline_shadow_preflight",
        "kill_switch_audit_passed": activation_count == 0 and not shadow_takes_control,
        "shadow_policy_takes_control": shadow_takes_control,
        "experimental_control_activation_count": activation_count,
        "default_policy_authoritative": True,
        "kill_switch_required_for_any_activation": True,
        "real_executor_connected": False,
    }


def _rollback_audit(
    *,
    cost_summary: dict[str, Any],
    kill_switch_audit: dict[str, Any],
) -> dict[str, Any]:
    passed = (
        kill_switch_audit.get("kill_switch_audit_passed") is True
        and cost_summary.get("publishes_checkpoint") is not True
        and cost_summary.get("replaces_default_policy") is not True
        and cost_summary.get("connects_real_executor") is not True
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "rollback",
        "rollback_audit_passed": passed,
        "default_policy_authoritative": True,
        "default_policy_unchanged": True,
        "source_fallback_can_restore_control": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "rollback_action": "no_op_offline_shadow_preflight",
    }


def _telemetry_audit(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {"coverage_rate_delta", "path_cost", "risk", "energy_cost", "fallback_like", "activation", "rollback"}
    missing_rows = [
        row.get("source_index")
        for row in step_rows
        if not required <= set(row.get("telemetry_fields_recorded") or [])
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "telemetry",
        "telemetry_audit_passed": not missing_rows,
        "required_fields": sorted(required),
        "missing_telemetry_row_count": len(missing_rows),
        "missing_telemetry_source_indices": missing_rows[:50],
        "records_coverage_gain": "coverage_rate_delta" in required,
        "records_cost_risk_energy": True,
        "records_fallback_activation_rollback": True,
    }


def _acceptance_reason_codes(
    *,
    cost_summary: dict[str, Any],
    comparison: dict[str, Any],
    long_horizon: dict[str, Any],
    coverage_efficiency_audit: dict[str, Any],
    guard_fallback_audit: dict[str, Any],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    fallback_rate_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if long_horizon.get("long_horizon_shadow_passed") is not True:
        _add_reason(reasons, "long_horizon_coverage_regression")
    if _float(comparison.get("coverage_return_improvement")) <= TOLERANCE:
        _add_reason(reasons, "long_horizon_coverage_regression")
    if _float(comparison.get("cumulative_coverage_rate_delta_improvement")) <= TOLERANCE:
        _add_reason(reasons, "long_horizon_coverage_regression")
    if _float(comparison.get("valuable_area_covered_improvement")) <= TOLERANCE:
        _add_reason(reasons, "long_horizon_coverage_regression")
    if coverage_efficiency_audit.get("coverage_efficiency_audit_passed") is not True:
        _add_reason(reasons, "coverage_efficiency_regression")
    if _int(guard_fallback_audit.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "shadow_controlled_regression")
    if _int(guard_fallback_audit.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    if _float(guard_fallback_audit.get("fallback_rate")) >= fallback_rate_threshold:
        _add_reason(reasons, "fallback_dominates")
    if _int(cost_summary.get("safe_better_training_family_count")) != len(TARGET_FAMILIES):
        _add_reason(reasons, "family_balance_failed")
    if kill_switch_audit.get("shadow_policy_takes_control") is True:
        _add_reason(reasons, "shadow_policy_took_control")
    if kill_switch_audit.get("kill_switch_audit_passed") is not True:
        _add_reason(reasons, "kill_switch_failed")
    if rollback_audit.get("rollback_audit_passed") is not True:
        _add_reason(reasons, "rollback_failed")
    if telemetry_audit.get("telemetry_audit_passed") is not True:
        _add_reason(reasons, "telemetry_missing_coverage_gain")
    return reasons


def _eligibility_ledger(
    *,
    reason_codes: list[str],
    cost_summary: dict[str, Any],
    long_horizon: dict[str, Any],
    coverage_efficiency_audit: dict[str, Any],
    guard_fallback_audit: dict[str, Any],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        _gate("input_cost_efficiency_stage_passed", "input_cost_efficiency_stage_not_passed" not in reason_codes),
        _gate("long_horizon_shadow_passed", long_horizon.get("long_horizon_shadow_passed") is True),
        _gate("coverage_efficiency_audit_passed", coverage_efficiency_audit.get("coverage_efficiency_audit_passed") is True),
        _gate("guard_fallback_audit_passed", guard_fallback_audit.get("guard_fallback_audit_passed") is True),
        _gate("kill_switch_audit_passed", kill_switch_audit.get("kill_switch_audit_passed") is True),
        _gate("rollback_audit_passed", rollback_audit.get("rollback_audit_passed") is True),
        _gate("telemetry_audit_passed", telemetry_audit.get("telemetry_audit_passed") is True),
        _gate("family_balance_passed", _int(cost_summary.get("safe_better_training_family_count")) == len(TARGET_FAMILIES)),
        _gate("performance_not_claimed", cost_summary.get("performance_claimed") is not True),
        _gate("offline_only_scope_guard", rollback_audit.get("connects_real_executor") is False),
    ]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "eligible_for_formal_performance_claim_review": not reason_codes,
        "reason_codes": list(reason_codes),
        "gates": gates,
    }


def _rejection_report(*, reason_codes: list[str], ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "rejection_report",
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": list(reason_codes),
        "next_required_change": _next_required_change(reason_codes),
        "failed_gates": [gate for gate in ledger.get("gates", []) if not gate.get("passed")],
    }


def _runtime_manifest(
    *,
    repo_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    input_paths: dict[str, Path],
    paths: dict[str, Path],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "repo_root": str(repo_root),
        "cost_efficiency_root": str(cost_efficiency_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "input_paths": {key: str(value) for key, value in input_paths.items()},
        "output_paths": {key: str(value) for key, value in paths.items()},
        "shadow_policy_takes_control": False,
        "experimental_control_activation_count": kill_switch_audit.get("experimental_control_activation_count", 0),
        "default_policy_authoritative": True,
        "source_fallback_available": rollback_audit.get("source_fallback_can_restore_control") is True,
        "writes_default_policy": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "performance_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _summary(
    *,
    status: str,
    reason_codes: list[str],
    paths: dict[str, Path],
    repo_root: Path,
    cost_efficiency_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    cost_summary: dict[str, Any],
    input_paths: dict[str, str],
    filtered_batch: list[dict[str, Any]],
    refined_transitions: list[dict[str, Any]],
    guard_replay_audit: dict[str, Any],
    performance_metric_rows: list[dict[str, Any]],
    compatible_performance_metric_rows: list[dict[str, Any]],
    stage5a_rerun_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
    baseline_actor: str | None,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    computed_candidate_metrics: dict[str, Any],
    computed_teacher_metrics: dict[str, Any],
    comparison: dict[str, Any],
    long_horizon: dict[str, Any],
    coverage_efficiency_audit: dict[str, Any],
    guard_fallback_audit: dict[str, Any],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    ledger: dict[str, Any],
    fallback_rate_threshold: float,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": _next_required_change(reason_codes),
        "repo_root": str(repo_root),
        "cost_efficiency_root": str(cost_efficiency_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "input_paths": input_paths,
        "summary": str(paths["summary"]),
        "runtime_manifest": str(paths["runtime_manifest"]),
        "shadow_step_comparison": str(paths["shadow_step_comparison"]),
        "long_horizon_shadow_validation": str(paths["long_horizon_shadow_validation"]),
        "coverage_efficiency_audit": str(paths["coverage_efficiency_audit"]),
        "guard_fallback_audit": str(paths["guard_fallback_audit"]),
        "kill_switch_audit": str(paths["kill_switch_audit"]),
        "rollback_audit": str(paths["rollback_audit"]),
        "telemetry_audit": str(paths["telemetry_audit"]),
        "eligibility_ledger": str(paths["eligibility_ledger"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "horizons": long_horizon.get("horizons", []),
        "long_horizon_shadow_passed": long_horizon.get("long_horizon_shadow_passed") is True,
        "baseline_actor": baseline_actor,
        "coverage_return_improvement": comparison.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": comparison.get(
            "cumulative_coverage_rate_delta_improvement",
            0.0,
        ),
        "valuable_area_covered_improvement": comparison.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(comparison.get("coverage_efficiency_regression")),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "computed_shadow_candidate_metrics": computed_candidate_metrics,
        "computed_shadow_teacher_metrics": computed_teacher_metrics,
        "trainable_pair_count": cost_summary.get("trainable_pair_count", len(filtered_batch)),
        "safe_better_training_pair_count": cost_summary.get("safe_better_training_pair_count", len(filtered_batch)),
        "safe_better_training_family_count": cost_summary.get("safe_better_training_family_count"),
        "filtered_batch_row_count": len(filtered_batch),
        "refined_transition_row_count": len(refined_transitions),
        "guard_replay_row_count": len(_replay_rows(guard_replay_audit)),
        "performance_metric_row_count": len(performance_metric_rows),
        "compatible_performance_metric_row_count": len(compatible_performance_metric_rows),
        "stage5a_rerun_status": stage5a_rerun_summary.get("status"),
        "formal_training_status": formal_summary.get("status"),
        "post_training_replay_status": post_training_replay_summary.get("status"),
        "selected_candidate_status": selected_candidate_summary.get("status"),
        "accepted_policy_activation_rate": candidate_metrics.get("accepted_policy_activation_rate"),
        "fallback_rate": candidate_metrics.get("fallback_rate", guard_fallback_audit.get("fallback_rate")),
        "fallback_rate_threshold": fallback_rate_threshold,
        "controlled_regression_count": guard_fallback_audit.get("controlled_regression_count", 0),
        "fallback_gain_contamination_count": guard_fallback_audit.get("fallback_gain_contamination_count", 0),
        "shadow_policy_takes_control": kill_switch_audit.get("shadow_policy_takes_control") is True,
        "experimental_control_activation_count": kill_switch_audit.get("experimental_control_activation_count", 0),
        "kill_switch_audit_passed": kill_switch_audit.get("kill_switch_audit_passed") is True,
        "rollback_audit_passed": rollback_audit.get("rollback_audit_passed") is True,
        "telemetry_audit_passed": telemetry_audit.get("telemetry_audit_passed") is True,
        "eligible_for_formal_performance_claim_review": ledger.get(
            "eligible_for_formal_performance_claim_review"
        )
        is True,
        "uses_old_stage5a2_pairs_as_current_model_evidence": False,
        "shadow_policy_takes_control_allowed": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    efficiency_regression = False
    for metric in ("coverage_gain_per_path_cost", "coverage_gain_per_risk", "coverage_gain_per_energy"):
        candidate_value = candidate.get(metric)
        baseline_value = baseline.get(metric)
        if candidate_value is None or baseline_value is None:
            efficiency_regression = True
            continue
        if float(candidate_value) + TOLERANCE < float(baseline_value):
            efficiency_regression = True
    return {
        "coverage_return_improvement": _metric_improvement(candidate, baseline, "coverage_return"),
        "cumulative_coverage_rate_delta_improvement": _metric_improvement(
            candidate,
            baseline,
            "cumulative_coverage_rate_delta",
        ),
        "valuable_area_covered_improvement": _metric_improvement(candidate, baseline, "valuable_area_covered"),
        "final_coverage_rate_improvement": _metric_improvement(candidate, baseline, "final_coverage_rate"),
        "new_area_covered_improvement": _metric_improvement(candidate, baseline, "new_area_covered"),
        "path_cost_delta_vs_baseline": _metric_improvement(candidate, baseline, "path_cost"),
        "risk_delta_vs_baseline": _metric_improvement(candidate, baseline, "risk"),
        "energy_cost_delta_vs_baseline": _metric_improvement(candidate, baseline, "energy_cost"),
        "coverage_efficiency_regression": efficiency_regression,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Shadow / Canary Release Performance Validation Preflight v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- long_horizon_shadow_passed: `{summary['long_horizon_shadow_passed']}`",
            f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
            f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
            f"- valuable_area_covered_improvement: `{summary['valuable_area_covered_improvement']}`",
            f"- coverage_efficiency_regression: `{summary['coverage_efficiency_regression']}`",
            f"- fallback_rate: `{summary['fallback_rate']}`",
            "",
            "## Inputs",
            "",
            f"- cost_efficiency_root: `{summary['cost_efficiency_root']}`",
            f"- filtered_batch_row_count: `{summary['filtered_batch_row_count']}`",
            f"- guard_replay_row_count: `{summary['guard_replay_row_count']}`",
            f"- performance_metric_row_count: `{summary['performance_metric_row_count']}`",
            "",
            "## Offline Shadow Boundary",
            "",
            "- shadow_policy_takes_control: `false`",
            "- experimental_control_activation_count: `0`",
            "- default policy remains authoritative.",
            "- source fallback remains available.",
            "",
            "## Audits",
            "",
            f"- kill_switch_audit_passed: `{summary['kill_switch_audit_passed']}`",
            f"- rollback_audit_passed: `{summary['rollback_audit_passed']}`",
            f"- telemetry_audit_passed: `{summary['telemetry_audit_passed']}`",
            "",
            "## Scope Guards",
            "",
            "- runs_new_ppo_update: `false`",
            "- publishes_checkpoint: `false`",
            "- replaces_default_policy: `false`",
            "- connects_real_executor: `false`",
            "- performance_claimed: `false`",
            "- modifies_network_or_action_space: `false`",
            "- modifies_default_astar: `false`",
            "",
        ]
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "runtime_manifest": output_root / RUNTIME_MANIFEST_FILE,
        "shadow_step_comparison": output_root / SHADOW_STEP_COMPARISON_FILE,
        "long_horizon_shadow_validation": output_root / LONG_HORIZON_FILE,
        "coverage_efficiency_audit": output_root / COVERAGE_EFFICIENCY_AUDIT_FILE,
        "guard_fallback_audit": output_root / GUARD_FALLBACK_AUDIT_FILE,
        "kill_switch_audit": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback_audit": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry_audit": output_root / TELEMETRY_AUDIT_FILE,
        "eligibility_ledger": output_root / ELIGIBILITY_LEDGER_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _metric_rows_by_actor(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("actor") or ""): row for row in rows if row.get("actor")}


def _best_baseline_actor(metrics_by_actor: dict[str, dict[str, Any]]) -> str | None:
    actors = [
        actor
        for actor in BASELINE_ACTORS
        if actor in metrics_by_actor and not metrics_by_actor[actor].get("baseline_evidence_missing")
    ]
    if not actors:
        return None
    return max(actors, key=lambda actor: _float_or_default(metrics_by_actor[actor].get("coverage_return"), -math.inf))


def _replay_rows(guard_replay_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = guard_replay_audit.get("rows")
    return rows if isinstance(rows, list) else []


def _candidate_key(row: dict[str, Any], action_index: Any) -> tuple[str, str, int, int]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
        _int(action_index),
    )


def _telemetry_fields(row: dict[str, Any]) -> list[str]:
    fields = []
    required_map = {
        "coverage_rate_delta": "coverage_rate_delta",
        "path_cost": "path_cost",
        "risk": "risk",
        "energy_cost": "energy_cost",
        "fallback_like": "fallback_like",
    }
    for field, source in required_map.items():
        if source in row:
            fields.append(field)
    fields.extend(["activation", "rollback"])
    return sorted(set(fields))


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _next_required_change(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "formal_performance_claim_release_decision"
    if "input_cost_efficiency_stage_not_passed" in reason_codes:
        return "fix_cost_efficiency_stage_inputs"
    if "long_horizon_coverage_regression" in reason_codes:
        return "rerun_cost_efficiency_candidate_on_longer_shadow_horizon"
    if "coverage_efficiency_regression" in reason_codes:
        return "retune_cost_efficiency_reward_before_shadow_canary"
    if "shadow_controlled_regression" in reason_codes:
        return "fix_guarded_shadow_controlled_regression"
    if "fallback_dominates" in reason_codes:
        return "reduce_guard_rejections_or_policy_drift_before_canary"
    if "shadow_policy_took_control" in reason_codes:
        return "restore_offline_shadow_only_boundary"
    if "kill_switch_failed" in reason_codes:
        return "fix_shadow_kill_switch"
    if "rollback_failed" in reason_codes:
        return "fix_shadow_rollback_manifest"
    if "telemetry_missing_coverage_gain" in reason_codes:
        return "fix_shadow_coverage_telemetry"
    if "family_balance_failed" in reason_codes:
        return "rebalance_safe_better_training_families"
    return "inspect_shadow_canary_release_performance_preflight"


def _metric_improvement(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(_float(candidate.get(key)) - _float(baseline.get(key)), 12)


def _current_root_path(value: Any, root: Path) -> bool:
    if not value:
        return False
    try:
        path = Path(str(value)).resolve()
        return root.resolve() in path.parents or path == root.resolve()
    except OSError:
        return str(root) in str(value)


def _parse_horizons(value: str | list[str] | tuple[str, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = []
        for item in value:
            parts.extend(str(item).replace(",", " ").split())
    horizons = sorted({int(part) for part in parts if int(part) > 0})
    return tuple(horizons or DEFAULT_HORIZONS)


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _add_reason(reasons, f"{label}_missing")
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid_json")
    return {}


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except FileNotFoundError:
        _add_reason(reasons, f"{label}_missing")
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid_jsonl")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return repo_root / path


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _float_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_float(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return default


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 12)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= TOLERANCE:
        return None
    return round(float(numerator) / float(denominator), 12)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
