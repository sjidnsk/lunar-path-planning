from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    import run_shadow_canary_release_performance_validation_preflight as shadow
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts import run_shadow_canary_release_performance_validation_preflight as shadow


SUMMARY_SCHEMA_VERSION = "family-balanced-shadow-canary-preflight-summary/v1"
AUDIT_SCHEMA_VERSION = "family-balanced-shadow-canary-audit/v1"
RUNTIME_MANIFEST_SCHEMA_VERSION = "family-balanced-shadow-canary-runtime-manifest/v1"
STEP_SCHEMA_VERSION = "family-balanced-shadow-canary-step/v1"

DEFAULT_RERUN_ROOT = "outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1"
DEFAULT_GAP_ROOT = "outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1"

RERUN_SUMMARY_FILE = "family-balanced-coverage-driven-ppo-rerun-summary.json"
REFINED_SUMMARY_FILE = "refined-coverage-driven-ppo-improvement-run-summary.json"
GUARD_REPLAY_AUDIT_FILE = "coverage-driven-ppo-replay-audit.json"
PERFORMANCE_METRIC_TABLE_FILE = "refined-coverage-driven-ppo-performance-metric-table.jsonl"
COMPATIBLE_PERFORMANCE_METRIC_TABLE_FILE = (
    "family-balanced-compatible-performance-evaluation/coverage-performance-metric-table.jsonl"
)
COMPATIBLE_PERFORMANCE_SUMMARY_FILE = (
    "family-balanced-compatible-performance-evaluation/exploration-coverage-performance-evaluation-summary.json"
)
STAGE5A_RERUN_SUMMARY_FILE = "stage5a-rerun-summary.json"
REFINED_TRANSITIONS_FILE = "refined-coverage-ppo-batch/refined-trainable-transitions.jsonl"
CHECKPOINT_METADATA_FILE = "coverage-driven-experimental-policy-candidate-metadata.json"
GAP_SUMMARY_FILE = "family-balanced-algorithm-gap-closure-summary.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
POST_TRAINING_REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_CANDIDATE_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"

SUMMARY_FILE = "family-balanced-shadow-canary-preflight-summary.json"
RUNTIME_MANIFEST_FILE = "family-balanced-shadow-canary-runtime-manifest.json"
SHADOW_STEP_COMPARISON_FILE = "family-balanced-shadow-canary-step-comparison.jsonl"
LONG_HORIZON_FILE = "family-balanced-shadow-canary-long-horizon-validation.json"
FAMILY_GENERALIZATION_AUDIT_FILE = "family-balanced-shadow-canary-family-generalization-audit.json"
COVERAGE_EFFICIENCY_AUDIT_FILE = "family-balanced-shadow-canary-coverage-efficiency-audit.json"
GUARD_FALLBACK_AUDIT_FILE = "family-balanced-shadow-canary-guard-fallback-audit.json"
KILL_SWITCH_AUDIT_FILE = "family-balanced-shadow-canary-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "family-balanced-shadow-canary-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "family-balanced-shadow-canary-telemetry-audit.json"
LINEAGE_AUDIT_FILE = "family-balanced-shadow-canary-lineage-audit.json"
RELEASE_AUDIT_FILE = "family-balanced-shadow-canary-release-boundary-audit.json"
REJECTION_REPORT_FILE = "family-balanced-shadow-canary-rejection-report.json"
REPORT_FILE = "family-balanced-shadow-canary-preflight-report.md"

TARGET_FAMILIES = (
    "low_observation_count",
    "mixed_risk",
    "rim_or_steep_slope",
    "smooth_high_confidence",
)
LOW_OBSERVATION_FAMILY = "low_observation_count"
POST_ACTOR = "post_improvement_ppo"
DEFAULT_HORIZONS = (10, 20, 30)
DEFAULT_FALLBACK_RATE_THRESHOLD = 0.5
DISCOUNT_FACTOR = 0.99
TOLERANCE = 1.0e-9

RELEASE_BOUNDARY_FIELDS = (
    "checkpoint_publication_approved",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "final_release_approved",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
    "default_astar_modified",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run family-balanced offline shadow/canary preflight.")
    parser.add_argument("--family-balanced-rerun-root", default=DEFAULT_RERUN_ROOT)
    parser.add_argument("--family-balanced-gap-root", default=DEFAULT_GAP_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--horizons", default="10,20,30")
    parser.add_argument("--fallback-rate-threshold", type=float, default=DEFAULT_FALLBACK_RATE_THRESHOLD)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_family_balanced_shadow_canary_preflight(
        family_balanced_rerun_root=_resolve_path(Path(args.family_balanced_rerun_root), repo_root),
        family_balanced_gap_root=_resolve_path(Path(args.family_balanced_gap_root), repo_root),
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
                "preflight_verdict": summary["preflight_verdict"],
                "long_horizon_shadow_passed": summary["long_horizon_shadow_passed"],
                "coverage_return_improvement": summary["coverage_return_improvement"],
                "cumulative_coverage_rate_delta_improvement": summary[
                    "cumulative_coverage_rate_delta_improvement"
                ],
                "valuable_area_covered_improvement": summary["valuable_area_covered_improvement"],
                "coverage_efficiency_regression": summary["coverage_efficiency_regression"],
                "fallback_rate": summary["fallback_rate"],
                "next_required_change": summary["next_required_change"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_family_balanced_shadow_canary_preflight(
    *,
    family_balanced_rerun_root: Path,
    family_balanced_gap_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    fallback_rate_threshold: float = DEFAULT_FALLBACK_RATE_THRESHOLD,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    family_balanced_rerun_root = Path(family_balanced_rerun_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    read_reasons: list[str] = []

    rerun_summary_path = family_balanced_rerun_root / RERUN_SUMMARY_FILE
    rerun_summary = _read_json(rerun_summary_path, read_reasons, "family_balanced_rerun_summary")
    refined_summary_path = _summary_path(rerun_summary, "refined_summary", family_balanced_rerun_root, repo_root, REFINED_SUMMARY_FILE)
    refined_summary = _read_json(refined_summary_path, read_reasons, "refined_summary")
    pair_path = _summary_path(rerun_summary, "family_balanced_safe_better_pairs", family_balanced_rerun_root, repo_root, "family-balanced-safe-better-pairs.jsonl")
    pair_rows = _read_jsonl(pair_path, read_reasons, "family_balanced_safe_better_pairs")
    guard_replay_audit_path = _summary_path(rerun_summary, "guard_replay_audit", family_balanced_rerun_root, repo_root, GUARD_REPLAY_AUDIT_FILE)
    guard_replay_audit = _read_json(guard_replay_audit_path, read_reasons, "guard_replay_audit")
    performance_metric_table_path = _summary_path(refined_summary or rerun_summary, "performance_metric_table", family_balanced_rerun_root, repo_root, PERFORMANCE_METRIC_TABLE_FILE)
    performance_metric_rows = _read_jsonl(performance_metric_table_path, read_reasons, "performance_metric_table")
    compatible_metric_table_path = _summary_path(rerun_summary, "compatible_performance_metric_table", family_balanced_rerun_root, repo_root, COMPATIBLE_PERFORMANCE_METRIC_TABLE_FILE)
    compatible_performance_metric_rows = _read_jsonl(compatible_metric_table_path, read_reasons, "compatible_performance_metric_table")
    compatible_performance_summary_path = (
        compatible_metric_table_path.parent / COMPATIBLE_PERFORMANCE_SUMMARY_FILE.split("/")[-1]
    )
    compatible_performance_summary = _read_json(
        compatible_performance_summary_path,
        [],
        "compatible_performance_summary",
    )
    stage5a_rerun_summary_path = _summary_path(refined_summary or rerun_summary, "stage5a_rerun_summary", family_balanced_rerun_root, repo_root, STAGE5A_RERUN_SUMMARY_FILE)
    stage5a_rerun_summary = _read_json(stage5a_rerun_summary_path, [], "stage5a_rerun_summary")
    refined_transitions_path = _summary_path(refined_summary or rerun_summary, "refined_trainable_transitions", family_balanced_rerun_root, repo_root, REFINED_TRANSITIONS_FILE)
    refined_transitions = _read_jsonl(refined_transitions_path, read_reasons, "refined_trainable_transitions")
    checkpoint_metadata_path = _summary_path(rerun_summary, "checkpoint_metadata_path", family_balanced_rerun_root, repo_root, CHECKPOINT_METADATA_FILE)
    checkpoint_metadata = _read_json(checkpoint_metadata_path, read_reasons, "checkpoint_metadata")
    gap_summary_path = Path(family_balanced_gap_root) / GAP_SUMMARY_FILE
    gap_summary = _read_json(gap_summary_path, read_reasons, "family_balanced_gap_summary")
    formal_summary_path = Path(formal_training_root) / FORMAL_SUMMARY_FILE
    post_training_replay_summary_path = Path(post_training_replay_root) / POST_TRAINING_REPLAY_SUMMARY_FILE
    selected_candidate_summary_path = Path(selected_candidate_root) / SELECTED_CANDIDATE_SUMMARY_FILE
    formal_summary = _read_json(formal_summary_path, read_reasons, "formal_training_summary")
    post_training_replay_summary = _read_json(
        post_training_replay_summary_path,
        read_reasons,
        "post_training_replay_summary",
    )
    selected_candidate_summary = _read_json(
        selected_candidate_summary_path,
        read_reasons,
        "selected_candidate_summary",
    )

    input_reasons = _unique(
        [
            *read_reasons,
            *_validate_inputs(
                rerun_summary=rerun_summary,
                refined_summary=refined_summary,
                gap_summary=gap_summary,
                pair_rows=pair_rows,
                replay_rows=_replay_rows(guard_replay_audit),
                performance_metric_rows=performance_metric_rows,
                compatible_performance_metric_rows=compatible_performance_metric_rows,
                compatible_performance_summary=compatible_performance_summary,
                refined_transitions=refined_transitions,
                checkpoint_metadata=checkpoint_metadata,
                formal_summary=formal_summary,
                post_training_replay_summary=post_training_replay_summary,
                selected_candidate_summary=selected_candidate_summary,
                fallback_rate_threshold=fallback_rate_threshold,
            ),
        ]
    )

    step_rows = _build_family_balanced_shadow_step_rows(
        pair_rows=pair_rows,
        replay_rows=_replay_rows(guard_replay_audit),
    )
    _write_jsonl(paths["shadow_step_comparison"], step_rows)

    metrics_by_actor = _metric_rows_by_actor(performance_metric_rows)
    compatible_metrics_by_actor = _metric_rows_by_actor(compatible_performance_metric_rows)
    candidate_metrics = metrics_by_actor.get(POST_ACTOR) or _aggregate_step_metrics(step_rows, actor=POST_ACTOR, role="candidate")
    baseline_actor = "teacher"
    baseline_metrics = compatible_metrics_by_actor.get("teacher") or _aggregate_step_metrics(step_rows, actor="teacher", role="teacher")
    computed_candidate_metrics = _aggregate_step_metrics(step_rows, actor=POST_ACTOR, role="candidate")
    computed_teacher_metrics = _aggregate_step_metrics(step_rows, actor="teacher", role="teacher")
    comparison = _comparison(candidate_metrics, baseline_metrics)

    long_horizon = _long_horizon_validation(
        step_rows=step_rows,
        horizons=tuple(sorted({int(horizon) for horizon in horizons if int(horizon) > 0})),
    )
    family_generalization_audit = _family_generalization_audit(step_rows)
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
    rollback_audit = _rollback_audit(rerun_summary=rerun_summary, kill_switch_audit=kill_switch_audit)
    telemetry_audit = _telemetry_audit(step_rows)
    lineage_audit = _lineage_audit(
        read_reasons=read_reasons,
        rerun_summary=rerun_summary,
        refined_summary=refined_summary,
        gap_summary=gap_summary,
        formal_summary=formal_summary,
        post_training_replay_summary=post_training_replay_summary,
        selected_candidate_summary=selected_candidate_summary,
    )
    release_audit = _release_audit(
        {
            "rerun_summary": rerun_summary,
            "refined_summary": refined_summary,
            "gap_summary": gap_summary,
            "checkpoint_metadata": checkpoint_metadata,
            "formal_summary": formal_summary,
            "post_training_replay_summary": post_training_replay_summary,
            "selected_candidate_summary": selected_candidate_summary,
        }
    )
    docs_audit = _docs_audit(repo_root)

    _write_json(paths["long_horizon_shadow_validation"], long_horizon)
    _write_json(paths["family_generalization_audit"], family_generalization_audit)
    _write_json(paths["coverage_efficiency_audit"], coverage_efficiency_audit)
    _write_json(paths["guard_fallback_audit"], guard_fallback_audit)
    _write_json(paths["kill_switch_audit"], kill_switch_audit)
    _write_json(paths["rollback_audit"], rollback_audit)
    _write_json(paths["telemetry_audit"], telemetry_audit)
    _write_json(paths["lineage_audit"], lineage_audit)
    _write_json(paths["release_boundary_audit"], release_audit)

    reason_codes = _unique(
        [
            *input_reasons,
            *_acceptance_reason_codes(
                comparison=comparison,
                long_horizon=long_horizon,
                family_generalization_audit=family_generalization_audit,
                coverage_efficiency_audit=coverage_efficiency_audit,
                guard_fallback_audit=guard_fallback_audit,
                kill_switch_audit=kill_switch_audit,
                rollback_audit=rollback_audit,
                telemetry_audit=telemetry_audit,
                lineage_audit=lineage_audit,
                release_audit=release_audit,
                docs_audit=docs_audit,
                fallback_rate_threshold=fallback_rate_threshold,
            ),
        ]
    )
    runtime_manifest = _runtime_manifest(
        repo_root=repo_root,
        family_balanced_rerun_root=family_balanced_rerun_root,
        family_balanced_gap_root=Path(family_balanced_gap_root),
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        input_paths={
            "rerun_summary": rerun_summary_path,
            "refined_summary": refined_summary_path,
            "family_balanced_safe_better_pairs": pair_path,
            "guard_replay_audit": guard_replay_audit_path,
            "performance_metric_table": performance_metric_table_path,
            "compatible_performance_metric_table": compatible_metric_table_path,
            "compatible_performance_summary": compatible_performance_summary_path,
            "stage5a_rerun_summary": stage5a_rerun_summary_path,
            "refined_trainable_transitions": refined_transitions_path,
            "checkpoint_metadata": checkpoint_metadata_path,
            "gap_summary": gap_summary_path,
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
        runtime_manifest=runtime_manifest,
        family_balanced_rerun_root=family_balanced_rerun_root,
        family_balanced_gap_root=Path(family_balanced_gap_root),
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        output_root=output_root,
        rerun_summary=rerun_summary,
        refined_summary=refined_summary,
        gap_summary=gap_summary,
        pair_rows=pair_rows,
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
        family_generalization_audit=family_generalization_audit,
        coverage_efficiency_audit=coverage_efficiency_audit,
        guard_fallback_audit=guard_fallback_audit,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        lineage_audit=lineage_audit,
        release_audit=release_audit,
        docs_audit=docs_audit,
        fallback_rate_threshold=fallback_rate_threshold,
        repo_root=repo_root,
    )
    _write_json(paths["summary"], summary)
    _write_json(paths["rejection_report"], _rejection_report(reason_codes=reason_codes, summary=summary))
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _validate_inputs(
    *,
    rerun_summary: dict[str, Any],
    refined_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    performance_metric_rows: list[dict[str, Any]],
    compatible_performance_metric_rows: list[dict[str, Any]],
    compatible_performance_summary: dict[str, Any],
    refined_transitions: list[dict[str, Any]],
    checkpoint_metadata: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
    fallback_rate_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if rerun_summary.get("status") != "passed" or _string_list(rerun_summary.get("reason_codes")):
        _add_reason(reasons, "family_balanced_rerun_not_passed")
    if rerun_summary.get("family_balanced_coverage_driven_ppo_rerun_passed") is not True:
        _add_reason(reasons, "family_balanced_rerun_not_passed")
    if rerun_summary.get("next_required_change") != "family_balanced_shadow_canary_preflight":
        _add_reason(reasons, "family_balanced_rerun_not_routed_to_shadow_canary")
    for metric in (
        "coverage_return_improvement",
        "cumulative_coverage_rate_delta_improvement",
        "valuable_area_covered_improvement",
    ):
        if _float(rerun_summary.get(metric)) <= TOLERANCE:
            _add_reason(reasons, "family_balanced_rerun_not_passed")
    if bool(rerun_summary.get("coverage_efficiency_regression")):
        _add_reason(reasons, "coverage_efficiency_regression")
    if _float(rerun_summary.get("fallback_rate")) >= fallback_rate_threshold:
        _add_reason(reasons, "fallback_dominates")
    if _int(rerun_summary.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_detected")
    if _int(rerun_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination")
    if _int(rerun_summary.get("safe_better_training_family_count")) != len(TARGET_FAMILIES):
        _add_reason(reasons, "family_generalization_regression")
    if _int(rerun_summary.get("low_observation_trainable_transition_count")) <= 0:
        _add_reason(reasons, "low_observation_shadow_regression")
    if not pair_rows or not replay_rows or not refined_transitions:
        _add_reason(reasons, "family_balanced_rerun_not_passed")
    if performance_metric_rows and POST_ACTOR not in _metric_rows_by_actor(performance_metric_rows):
        _add_reason(reasons, "family_balanced_rerun_not_passed")
    if not compatible_performance_metric_rows:
        _add_reason(reasons, "family_balanced_rerun_not_passed")
    if compatible_performance_summary and compatible_performance_summary.get("status") not in (None, "passed"):
        _add_reason(reasons, "family_balanced_rerun_not_passed")
    if refined_summary and refined_summary.get("status") not in (None, "passed"):
        _add_reason(reasons, "family_balanced_rerun_not_passed")
    if gap_summary.get("status") != "passed" or _string_list(gap_summary.get("reason_codes")):
        _add_reason(reasons, "lineage_incomplete")
    if checkpoint_metadata and checkpoint_metadata.get("experimental") is not True:
        _add_reason(reasons, "lineage_incomplete")
    for upstream in (formal_summary, post_training_replay_summary, selected_candidate_summary):
        if upstream.get("status") != "passed" or _string_list(upstream.get("reason_codes")):
            _add_reason(reasons, "lineage_incomplete")
        if _int(upstream.get("controlled_regression_count")) > 0:
            _add_reason(reasons, "controlled_regression_detected")
    return reasons


def _build_family_balanced_shadow_step_rows(
    *,
    pair_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_candidate = {
        _candidate_key(row, row.get("candidate_action_index")): row
        for row in pair_rows
        if row.get("candidate_action_index") is not None
    }
    by_decision: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in pair_rows:
        by_decision.setdefault(
            (str(row.get("context_id") or ""), str(row.get("episode_id") or ""), _int(row.get("step_index"))),
            row,
        )
    result: list[dict[str, Any]] = []
    for source_index, replay in enumerate(replay_rows):
        pair = by_candidate.get(_candidate_key(replay, replay.get("controlled_action_index")))
        if pair is None:
            pair = by_decision.get(
                (str(replay.get("context_id") or ""), str(replay.get("episode_id") or ""), _int(replay.get("step_index"))),
                {},
            )
        step_index = _int(replay.get("step_index"))
        coverage_rate_delta = _float(replay.get("coverage_rate_delta"))
        teacher_coverage_rate_delta = _float(pair.get("teacher_expected_coverage_rate_delta"))
        candidate_return = coverage_rate_delta * (DISCOUNT_FACTOR ** max(step_index, 0))
        teacher_return = teacher_coverage_rate_delta * (DISCOUNT_FACTOR ** max(step_index, 0))
        fallback_like = bool(replay.get("fallback_like")) or bool(replay.get("guard_rejected"))
        policy_action_accepted = bool(replay.get("policy_action_accepted"))
        teacher_value_proxy = _first_float(pair.get("teacher_valuable_coverage_proxy"), pair.get("teacher_value"), default=0.0)
        telemetry_fields = _telemetry_fields(replay)
        result.append(
            {
                "schema_version": STEP_SCHEMA_VERSION,
                "source_index": source_index,
                "replay_index": _int(replay.get("replay_index"), source_index),
                "context_id": replay.get("context_id"),
                "episode_id": replay.get("episode_id"),
                "step_index": step_index,
                "scenario_id": replay.get("scenario_id") or pair.get("scenario_id"),
                "scenario_family": replay.get("scenario_family") or pair.get("scenario_family"),
                "split": pair.get("split"),
                "candidate_action_index": pair.get("candidate_action_index"),
                "raw_policy_action_index": replay.get("raw_policy_action_index"),
                "controlled_action_index": replay.get("controlled_action_index"),
                "teacher_action_index": replay.get("teacher_action_index", pair.get("teacher_action_index")),
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
                "teacher_final_coverage_rate": teacher_coverage_rate_delta,
                "teacher_new_area_covered": _float(pair.get("teacher_expected_new_coverage_area")),
                "teacher_valuable_area_covered": teacher_coverage_rate_delta * float(teacher_value_proxy or 0.0),
                "teacher_information_gain": _float(pair.get("teacher_information_gain")),
                "teacher_path_cost": _float(pair.get("teacher_path_cost")),
                "teacher_risk": _float(pair.get("teacher_risk")),
                "teacher_energy_cost": _float(pair.get("teacher_energy_cost")),
                "coverage_return_improvement": round(candidate_return - teacher_return, 12),
                "cumulative_coverage_rate_delta_improvement": round(coverage_rate_delta - teacher_coverage_rate_delta, 12),
                "valuable_area_covered_improvement": round(
                    _float(replay.get("valuable_area_covered")) - teacher_coverage_rate_delta * float(teacher_value_proxy or 0.0),
                    12,
                ),
                "controlled_regression_reason_codes": _string_list(replay.get("controlled_regression_reason_codes")),
                "telemetry_fields_recorded": telemetry_fields,
            }
        )
    return result


def _family_generalization_audit(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        family = str(row.get("scenario_family") or "")
        if family:
            grouped[family].append(row)
    rollups = {}
    for family in TARGET_FAMILIES:
        rows = grouped.get(family, [])
        candidate = _metrics_for_rows(rows, "candidate")
        teacher = _metrics_for_rows(rows, "teacher")
        fallback_gain = sum(_float(row.get("coverage_rate_delta")) for row in rows if row.get("fallback_like"))
        controlled = sum(1 for row in rows if _string_list(row.get("controlled_regression_reason_codes")))
        rollups[family] = {
            "row_count": len(rows),
            "candidate_coverage_return": candidate["coverage_return"],
            "baseline_coverage_return": teacher["coverage_return"],
            "coverage_return_improvement": round(candidate["coverage_return"] - teacher["coverage_return"], 12),
            "candidate_cumulative_coverage_rate_delta": candidate["cumulative_coverage_rate_delta"],
            "baseline_cumulative_coverage_rate_delta": teacher["cumulative_coverage_rate_delta"],
            "candidate_valuable_area_covered": candidate["valuable_area_covered"],
            "baseline_valuable_area_covered": teacher["valuable_area_covered"],
            "fallback_gain": round(fallback_gain, 12),
            "controlled_regression_count": controlled,
            "family_shadow_passed": bool(rows)
            and candidate["coverage_return"] > TOLERANCE
            and fallback_gain <= TOLERANCE
            and controlled == 0,
        }
    low = rollups.get(LOW_OBSERVATION_FAMILY, {})
    family_passed = all(rollups[family]["family_shadow_passed"] for family in TARGET_FAMILIES)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "family_generalization",
        "family_generalization_audit_passed": family_passed,
        "low_observation_shadow_passed": low.get("family_shadow_passed") is True,
        "family_rollups": rollups,
        "target_families": list(TARGET_FAMILIES),
    }


def _lineage_audit(
    *,
    read_reasons: list[str],
    rerun_summary: dict[str, Any],
    refined_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    formal_summary: dict[str, Any],
    post_training_replay_summary: dict[str, Any],
    selected_candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "read_inputs": not read_reasons,
        "rerun_passed": rerun_summary.get("status") == "passed" and not _string_list(rerun_summary.get("reason_codes")),
        "refined_passed": not refined_summary or refined_summary.get("status") == "passed",
        "gap_passed": gap_summary.get("status") == "passed" and not _string_list(gap_summary.get("reason_codes")),
        "formal_training_passed": formal_summary.get("status") == "passed",
        "post_training_replay_passed": post_training_replay_summary.get("status") == "passed",
        "selected_candidate_passed": selected_candidate_summary.get("status") == "passed",
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "lineage",
        "lineage_audit_passed": all(checks.values()),
        "checks": checks,
        "read_reason_codes": read_reasons,
    }


def _release_audit(payloads: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for name, payload in payloads.items():
        _collect_release_violations(payload, name, violations)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "release_boundary",
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _collect_release_violations(payload: Any, path: str, violations: list[dict[str, str]]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            current = f"{path}.{key}"
            if key in RELEASE_BOUNDARY_FIELDS and value is True:
                violations.append({"path": current, "reason": "release_boundary_true"})
            _collect_release_violations(value, current, violations)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _collect_release_violations(item, f"{path}[{index}]", violations)


def _docs_audit(repo_root: Path) -> dict[str, Any]:
    required = {
        "README.md": (
            "Family-Balanced Shadow/Canary Preflight v1",
            "outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1/",
            "family_balanced_formal_performance_claim_release_decision",
        ),
        "docs/算法设计与系统架构报告.md": (
            "Family-Balanced Shadow/Canary Preflight v1",
            "离线 shadow/canary 预检",
            "不发布 checkpoint",
        ),
        "docs/superpowers/specs/2026-06-16-family-balanced-shadow-canary-preflight.md": (
            "Family-Balanced Shadow/Canary Preflight v1",
            "family-balanced-shadow-canary-preflight-summary.json",
            "不连接真实执行器",
        ),
    }
    missing = []
    for relative, markers in required.items():
        path = repo_root / relative
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not path.exists() or any(marker not in text for marker in markers):
            missing.append(relative)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "docs",
        "docs_updated": not missing,
        "missing_or_stale_docs": missing,
    }


def _acceptance_reason_codes(
    *,
    comparison: dict[str, Any],
    long_horizon: dict[str, Any],
    family_generalization_audit: dict[str, Any],
    coverage_efficiency_audit: dict[str, Any],
    guard_fallback_audit: dict[str, Any],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    lineage_audit: dict[str, Any],
    release_audit: dict[str, Any],
    docs_audit: dict[str, Any],
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
    if family_generalization_audit.get("family_generalization_audit_passed") is not True:
        _add_reason(reasons, "family_generalization_regression")
    if family_generalization_audit.get("low_observation_shadow_passed") is not True:
        _add_reason(reasons, "low_observation_shadow_regression")
    if coverage_efficiency_audit.get("coverage_efficiency_audit_passed") is not True:
        _add_reason(reasons, "coverage_efficiency_regression")
    if _int(guard_fallback_audit.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_detected")
    if _int(guard_fallback_audit.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination")
    if _float(guard_fallback_audit.get("fallback_rate")) >= fallback_rate_threshold:
        _add_reason(reasons, "fallback_dominates")
    if kill_switch_audit.get("shadow_policy_takes_control") is True:
        _add_reason(reasons, "shadow_policy_took_control")
    if kill_switch_audit.get("kill_switch_audit_passed") is not True:
        _add_reason(reasons, "kill_switch_failed")
    if rollback_audit.get("rollback_audit_passed") is not True:
        _add_reason(reasons, "rollback_failed")
    if telemetry_audit.get("telemetry_audit_passed") is not True:
        _add_reason(reasons, "telemetry_missing_coverage_gain")
    if lineage_audit.get("lineage_audit_passed") is not True:
        _add_reason(reasons, "lineage_incomplete")
    if release_audit.get("release_boundary_audit_passed") is not True:
        _add_reason(reasons, "release_boundary_violation")
    if docs_audit.get("docs_updated") is not True:
        _add_reason(reasons, "docs_not_updated")
    return reasons


def _runtime_manifest(
    *,
    repo_root: Path,
    family_balanced_rerun_root: Path,
    family_balanced_gap_root: Path,
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
        "family_balanced_rerun_root": str(family_balanced_rerun_root),
        "family_balanced_gap_root": str(family_balanced_gap_root),
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
    runtime_manifest: dict[str, Any],
    family_balanced_rerun_root: Path,
    family_balanced_gap_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    output_root: Path,
    rerun_summary: dict[str, Any],
    refined_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    pair_rows: list[dict[str, Any]],
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
    family_generalization_audit: dict[str, Any],
    coverage_efficiency_audit: dict[str, Any],
    guard_fallback_audit: dict[str, Any],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    lineage_audit: dict[str, Any],
    release_audit: dict[str, Any],
    docs_audit: dict[str, Any],
    fallback_rate_threshold: float,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "preflight_verdict": "eligible_for_family_balanced_formal_performance_claim_release_decision"
        if status == "passed"
        else "not_eligible_for_family_balanced_formal_performance_claim_release_decision",
        "family_balanced_shadow_canary_preflight_passed": status == "passed",
        "family_balanced_formal_performance_claim_release_decision_approved": status == "passed",
        "next_required_change": _next_required_change(reason_codes),
        "repo_root": str(repo_root),
        "family_balanced_rerun_root": str(family_balanced_rerun_root),
        "family_balanced_gap_root": str(family_balanced_gap_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "output_root": str(output_root),
        "input_paths": runtime_manifest["input_paths"],
        "summary": str(paths["summary"]),
        "runtime_manifest": str(paths["runtime_manifest"]),
        "shadow_step_comparison": str(paths["shadow_step_comparison"]),
        "long_horizon_shadow_validation": str(paths["long_horizon_shadow_validation"]),
        "family_generalization_audit": str(paths["family_generalization_audit"]),
        "coverage_efficiency_audit": str(paths["coverage_efficiency_audit"]),
        "guard_fallback_audit": str(paths["guard_fallback_audit"]),
        "kill_switch_audit": str(paths["kill_switch_audit"]),
        "rollback_audit": str(paths["rollback_audit"]),
        "telemetry_audit": str(paths["telemetry_audit"]),
        "lineage_audit": str(paths["lineage_audit"]),
        "release_boundary_audit": str(paths["release_boundary_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "horizons": long_horizon.get("horizons", []),
        "long_horizon_shadow_passed": long_horizon.get("long_horizon_shadow_passed") is True,
        "baseline_actor": baseline_actor,
        "coverage_return_improvement": comparison.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": comparison.get("cumulative_coverage_rate_delta_improvement", 0.0),
        "valuable_area_covered_improvement": comparison.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(comparison.get("coverage_efficiency_regression")),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "computed_shadow_candidate_metrics": computed_candidate_metrics,
        "computed_shadow_teacher_metrics": computed_teacher_metrics,
        "trainable_pair_count": rerun_summary.get("safe_better_training_pair_count", len(pair_rows)),
        "safe_better_training_pair_count": rerun_summary.get("safe_better_training_pair_count", len(pair_rows)),
        "safe_better_training_family_count": rerun_summary.get("safe_better_training_family_count"),
        "low_observation_trainable_transition_count": rerun_summary.get("low_observation_trainable_transition_count"),
        "family_balanced_pair_row_count": len(pair_rows),
        "refined_transition_row_count": len(refined_transitions),
        "guard_replay_row_count": len(_replay_rows(guard_replay_audit)),
        "performance_metric_row_count": len(performance_metric_rows),
        "compatible_performance_metric_row_count": len(compatible_performance_metric_rows),
        "stage5a_rerun_status": stage5a_rerun_summary.get("status"),
        "formal_training_status": formal_summary.get("status"),
        "post_training_replay_status": post_training_replay_summary.get("status"),
        "selected_candidate_status": selected_candidate_summary.get("status"),
        "gap_closure_status": gap_summary.get("status"),
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
        "family_generalization_audit_passed": family_generalization_audit.get("family_generalization_audit_passed") is True,
        "low_observation_shadow_passed": family_generalization_audit.get("low_observation_shadow_passed") is True,
        "lineage_audit_passed": lineage_audit.get("lineage_audit_passed") is True,
        "release_boundary_audit_passed": release_audit.get("release_boundary_audit_passed") is True,
        "docs_updated": docs_audit.get("docs_updated") is True,
        "shadow_policy_takes_control_allowed": False,
        "runs_new_ppo_update": False,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "refined_summary_status": refined_summary.get("status"),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _rejection_report(*, reason_codes: list[str], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "audit_name": "rejection_report",
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": list(reason_codes),
        "next_required_change": summary["next_required_change"],
    }


def _rollback_audit(*, rerun_summary: dict[str, Any], kill_switch_audit: dict[str, Any]) -> dict[str, Any]:
    passed = (
        kill_switch_audit.get("kill_switch_audit_passed") is True
        and rerun_summary.get("publishes_checkpoint") is not True
        and rerun_summary.get("replaces_default_policy") is not True
        and rerun_summary.get("connects_real_executor") is not True
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
        "rollback_action": "no_op_family_balanced_offline_shadow_preflight",
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Family-Balanced Shadow/Canary Preflight v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- preflight_verdict: `{summary['preflight_verdict']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- long_horizon_shadow_passed: `{summary['long_horizon_shadow_passed']}`",
            f"- family_generalization_audit_passed: `{summary['family_generalization_audit_passed']}`",
            f"- low_observation_shadow_passed: `{summary['low_observation_shadow_passed']}`",
            f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
            f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
            f"- valuable_area_covered_improvement: `{summary['valuable_area_covered_improvement']}`",
            f"- coverage_efficiency_regression: `{summary['coverage_efficiency_regression']}`",
            f"- fallback_rate: `{summary['fallback_rate']}`",
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
        "family_generalization_audit": output_root / FAMILY_GENERALIZATION_AUDIT_FILE,
        "coverage_efficiency_audit": output_root / COVERAGE_EFFICIENCY_AUDIT_FILE,
        "guard_fallback_audit": output_root / GUARD_FALLBACK_AUDIT_FILE,
        "kill_switch_audit": output_root / KILL_SWITCH_AUDIT_FILE,
        "rollback_audit": output_root / ROLLBACK_AUDIT_FILE,
        "telemetry_audit": output_root / TELEMETRY_AUDIT_FILE,
        "lineage_audit": output_root / LINEAGE_AUDIT_FILE,
        "release_boundary_audit": output_root / RELEASE_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _next_required_change(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "family_balanced_formal_performance_claim_release_decision"
    if "family_balanced_rerun_not_passed" in reason_codes or "family_balanced_rerun_not_routed_to_shadow_canary" in reason_codes:
        return "fix_family_balanced_shadow_canary_inputs"
    if "long_horizon_coverage_regression" in reason_codes:
        return "rerun_family_balanced_ppo_or_shadow_horizon"
    if "family_generalization_regression" in reason_codes or "low_observation_shadow_regression" in reason_codes:
        return "fix_family_balanced_generalization_before_shadow_canary"
    if "coverage_efficiency_regression" in reason_codes:
        return "retune_family_balanced_cost_efficiency_before_shadow_canary"
    if "fallback_dominates" in reason_codes:
        return "reduce_family_balanced_guard_rejections_before_canary"
    if "release_boundary_violation" in reason_codes:
        return "restore_family_balanced_shadow_release_boundary"
    if "docs_not_updated" in reason_codes:
        return "update_family_balanced_shadow_canary_docs"
    return "inspect_family_balanced_shadow_canary_preflight"


def _summary_path(summary: dict[str, Any], key: str, root: Path, repo_root: Path, default_relative: str) -> Path:
    return _resolve_optional_path(summary.get(key), root, repo_root) or root / default_relative


_long_horizon_validation = shadow._long_horizon_validation
_metrics_for_rows = shadow._metrics_for_rows
_aggregate_step_metrics = shadow._aggregate_step_metrics
_coverage_efficiency_audit = shadow._coverage_efficiency_audit
_guard_fallback_audit = shadow._guard_fallback_audit
_kill_switch_audit = shadow._kill_switch_audit
_telemetry_audit = shadow._telemetry_audit
_comparison = shadow._comparison
_metric_rows_by_actor = shadow._metric_rows_by_actor
_replay_rows = shadow._replay_rows
_candidate_key = shadow._candidate_key
_telemetry_fields = shadow._telemetry_fields
_parse_horizons = shadow._parse_horizons
_resolve_optional_path = shadow._resolve_optional_path
_float = shadow._float
_int = shadow._int
_first_float = shadow._first_float
_string_list = shadow._string_list
_safe_ratio = shadow._safe_ratio


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    return shadow._read_json(path, reasons, label)


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    return shadow._read_jsonl(path, reasons, label)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    shadow._write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    shadow._write_jsonl(path, rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
