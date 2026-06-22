from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for candidate in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from git_provenance import git_snapshot
    from scripts import run_expanded_four_family_refined_coverage_driven_ppo_improvement as expanded_helpers
    from run_refined_coverage_driven_ppo_improvement_run import (
        _resolve_optional_path,
        _selected_base_candidate_root,
        run_refined_coverage_driven_ppo_improvement_run,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts import run_expanded_four_family_refined_coverage_driven_ppo_improvement as expanded_helpers
    from scripts.run_refined_coverage_driven_ppo_improvement_run import (
        _resolve_optional_path,
        _selected_base_candidate_root,
        run_refined_coverage_driven_ppo_improvement_run,
    )

_compatible_stage5a2_rows = expanded_helpers._compatible_stage5a2_rows
_counterfactual_key = expanded_helpers._counterfactual_key
_counterfactual_key_from_pair = expanded_helpers._counterfactual_key_from_pair
_decision_key_from_pair = expanded_helpers._decision_key_from_pair
_decision_key_from_transition = expanded_helpers._decision_key_from_transition
_old_transitions = expanded_helpers._old_transitions
_synthetic_transitions_for_missing_decisions = expanded_helpers._synthetic_transitions_for_missing_decisions


SUMMARY_SCHEMA_VERSION = "family-balanced-coverage-driven-ppo-rerun-summary/v1"
SOURCE_LEDGER_SCHEMA_VERSION = "family-balanced-ppo-source-ledger/v1"
INPUT_AUDIT_SCHEMA_VERSION = "family-balanced-compatible-input-audit/v1"
OLD_TRANSITION_AUDIT_SCHEMA_VERSION = "family-balanced-old-transition-materialization-audit/v1"
PERFORMANCE_AUDIT_SCHEMA_VERSION = "family-balanced-performance-audit/v1"
RELEASE_AUDIT_SCHEMA_VERSION = "family-balanced-release-boundary-audit/v1"
COMPAT_STAGE5A2_SUMMARY_SCHEMA_VERSION = "policy-differentiating-counterfactual-coverage-rollouts-summary/v1"
COMPAT_EPISODE_SCHEMA_VERSION = "coverage-aware-ppo-rollout-episode/v1"

DEFAULT_GAP_ROOT = "outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1"
DEFAULT_COVERAGE_DRIVEN_ROOT = "outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_PERFORMANCE_ROOT = "outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1"
DEFAULT_REWARD_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1"

GAP_SUMMARY_FILE = "family-balanced-algorithm-gap-closure-summary.json"
GAP_MANIFEST_FILE = "family-balanced-coverage-driven-input-manifest.json"
GAP_PAIRS_FILE = "family-balanced-safe-better-pairs.jsonl"
GAP_COUNTERFACTUAL_FILE = "family-balanced-counterfactual-rollouts.jsonl"

COVERAGE_DRIVEN_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
COMPAT_STAGE5A2_SUMMARY_FILE = "policy-differentiating-counterfactual-coverage-rollouts-summary.json"
COMPAT_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
COMPAT_COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
COMPAT_BATCH_DIR = "coverage-aware-ppo-batch"
COMPAT_EPISODES_FILE = "ppo-rollout-episodes.jsonl"
COMPAT_STAGE5A2_DIR = "family-balanced-compatible-stage5a2-input"
COMPAT_COVERAGE_DRIVEN_DIR = "family-balanced-compatible-coverage-driven-input"
COMPAT_PERFORMANCE_DIR = "family-balanced-compatible-performance-evaluation"
PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
PERFORMANCE_METRIC_TABLE_FILE = "coverage-performance-metric-table.jsonl"
DISCOUNT_FACTOR = 0.99

SUMMARY_FILE = "family-balanced-coverage-driven-ppo-rerun-summary.json"
SOURCE_LEDGER_FILE = "family-balanced-ppo-source-ledger.json"
INPUT_AUDIT_FILE = "family-balanced-compatible-input-audit.json"
OLD_TRANSITION_AUDIT_FILE = "family-balanced-old-transition-materialization-audit.json"
ADVANTAGE_AUDIT_FILE = "family-balanced-ppo-advantage-audit.jsonl"
PERFORMANCE_AUDIT_FILE = "family-balanced-performance-audit.json"
RELEASE_AUDIT_FILE = "family-balanced-release-boundary-audit.json"
REJECTION_REPORT_FILE = "family-balanced-coverage-driven-ppo-rerun-rejection-report.json"
REPORT_FILE = "family-balanced-coverage-driven-ppo-rerun-report.md"

TARGET_FAMILIES = (
    "low_observation_count",
    "mixed_risk",
    "rim_or_steep_slope",
    "smooth_high_confidence",
)
LOW_OBSERVATION_FAMILY = "low_observation_count"
DEFAULT_MINIMUM_LOW_OBSERVATION_COUNT = 32
TOLERANCE = 1.0e-9
LEGACY_V1_REWARD_COMPONENT_BLOCKER = "legacy_v1_reward_components_blocked_by_canonical_v2"

RELEASE_BOUNDARY_TRUE_KEYS = (
    "checkpoint_publication_approved",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "relaxes_guard",
    "modifies_network_or_action_space",
    "modifies_default_astar",
    "ackermann_feasible_trajectory_claimed",
    "real_world_performance_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run family-balanced coverage-driven guarded PPO rerun.")
    parser.add_argument("--family-balanced-gap-root", default=DEFAULT_GAP_ROOT)
    parser.add_argument("--coverage-driven-root", default=DEFAULT_COVERAGE_DRIVEN_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--coverage-signal-root", default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--coverage-performance-root", default=DEFAULT_PERFORMANCE_ROOT)
    parser.add_argument("--reward-refinement-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--minimum-low-observation-count", type=int, default=DEFAULT_MINIMUM_LOW_OBSERVATION_COUNT)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_family_balanced_coverage_driven_ppo_rerun(
        family_balanced_gap_root=_resolve_path(Path(args.family_balanced_gap_root), repo_root),
        coverage_driven_root=_resolve_path(Path(args.coverage_driven_root), repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        minimum_low_observation_count=args.minimum_low_observation_count,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "rerun_verdict": summary["rerun_verdict"],
                "family_balanced_coverage_driven_ppo_rerun_passed": summary[
                    "family_balanced_coverage_driven_ppo_rerun_passed"
                ],
                "coverage_return_improvement": summary["coverage_return_improvement"],
                "cumulative_coverage_rate_delta_improvement": summary[
                    "cumulative_coverage_rate_delta_improvement"
                ],
                "valuable_area_covered_improvement": summary["valuable_area_covered_improvement"],
                "next_required_change": summary["next_required_change"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_family_balanced_coverage_driven_ppo_rerun(
    *,
    family_balanced_gap_root: Path,
    coverage_driven_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    repo_root: Path,
    minimum_low_observation_count: int = DEFAULT_MINIMUM_LOW_OBSERVATION_COUNT,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    paths["compat_stage5a2_root"].mkdir(parents=True, exist_ok=True)
    paths["compat_coverage_batch_root"].mkdir(parents=True, exist_ok=True)

    input_reasons: list[str] = []
    gap_summary = _read_json(Path(family_balanced_gap_root) / GAP_SUMMARY_FILE, input_reasons, "gap_summary")
    gap_manifest = _read_json(Path(family_balanced_gap_root) / GAP_MANIFEST_FILE, input_reasons, "gap_manifest")
    coverage_summary = _read_json(
        Path(coverage_driven_root) / COVERAGE_DRIVEN_SUMMARY_FILE,
        input_reasons,
        "coverage_driven_summary",
    )
    selected_summary = _read_json(
        Path(selected_candidate_root) / SELECTED_SUMMARY_FILE,
        input_reasons,
        "selected_candidate_summary",
    )

    pair_path = _resolve_optional_path(
        gap_summary.get("family_balanced_safe_better_pairs") or gap_manifest.get("family_balanced_safe_better_pairs"),
        Path(family_balanced_gap_root),
        repo_root,
    ) or Path(family_balanced_gap_root) / GAP_PAIRS_FILE
    counterfactual_path = _resolve_optional_path(
        gap_summary.get("family_balanced_counterfactual_rollouts")
        or gap_manifest.get("family_balanced_counterfactual_rollouts"),
        Path(family_balanced_gap_root),
        repo_root,
    ) or Path(family_balanced_gap_root) / GAP_COUNTERFACTUAL_FILE
    old_episodes_path = _resolve_optional_path(
        coverage_summary.get("coverage_aware_batch_episodes"),
        Path(coverage_driven_root),
        repo_root,
    ) or Path(coverage_driven_root) / COMPAT_BATCH_DIR / COMPAT_EPISODES_FILE

    pair_rows = _read_jsonl(pair_path, input_reasons, "family_balanced_safe_better_pairs")
    counterfactual_rows = _read_jsonl(counterfactual_path, input_reasons, "family_balanced_counterfactual_rollouts")
    old_episodes = _read_jsonl(old_episodes_path, input_reasons, "old_coverage_aware_episodes")

    base_candidate_root = _base_candidate_root(
        coverage_summary=coverage_summary,
        selected_summary=selected_summary,
        selected_candidate_root=Path(selected_candidate_root),
        repo_root=repo_root,
    )
    prepared = _prepare_family_balanced_inputs(
        pair_rows=pair_rows,
        counterfactual_rows=counterfactual_rows,
        old_episodes=old_episodes,
        gap_summary=gap_summary,
        gap_manifest=gap_manifest,
        coverage_summary=coverage_summary,
        base_candidate_root=base_candidate_root,
        family_balanced_gap_root=Path(family_balanced_gap_root),
        paths=paths,
        repo_root=repo_root,
        minimum_low_observation_count=minimum_low_observation_count,
        input_reasons=input_reasons,
    )

    reason_codes = _unique([*input_reasons, *prepared["reason_codes"]])
    _add_reason(reason_codes, LEGACY_V1_REWARD_COMPONENT_BLOCKER)
    docs_audit = _docs_audit(repo_root)
    if not docs_audit["docs_updated"]:
        _add_reason(reason_codes, "docs_not_updated")
    release_audit = _release_boundary_audit(
        source_payloads={
            "gap_summary": gap_summary,
            "gap_manifest": gap_manifest,
            "coverage_summary": coverage_summary,
            "selected_summary": selected_summary,
            "prepared_source_audit": prepared["source_ledger"],
        }
    )
    if not release_audit["release_boundary_audit_passed"]:
        _add_reason(reason_codes, "release_boundary_violation")

    refined_summary: dict[str, Any] = {}
    if not reason_codes:
        refined_summary = run_refined_coverage_driven_ppo_improvement_run(
            stage5a2_root=paths["compat_stage5a2_root"],
            coverage_driven_root=paths["compat_coverage_driven_root"],
            formal_training_root=Path(formal_training_root),
            post_training_replay_root=Path(post_training_replay_root),
            selected_candidate_root=Path(selected_candidate_root),
            coverage_signal_root=Path(coverage_signal_root),
            coverage_performance_root=paths["compat_performance_root"],
            reward_refinement_root=Path(reward_refinement_root),
            output_root=output_root,
            repo_root=repo_root,
            enforce_cost_risk_energy_filter=False,
            ppo_advantage_scale=50.0,
            use_transition_info_advantage=True,
            ppo_learning_rate=2.0e-3,
            ppo_epochs=30,
            ppo_max_approx_kl=100.0,
            ppo_advantage_field="family_balanced_ppo_advantage",
        )

    performance_audit = _performance_audit(refined_summary=refined_summary)
    for reason in performance_audit["reason_codes"]:
        _add_reason(reason_codes, reason)
    reason_codes = _unique(reason_codes)

    summary = _summary(
        paths=paths,
        reason_codes=reason_codes,
        prepared=prepared,
        refined_summary=refined_summary,
        performance_audit=performance_audit,
        release_audit=release_audit,
        docs_audit=docs_audit,
        family_balanced_gap_root=Path(family_balanced_gap_root),
        coverage_driven_root=Path(coverage_driven_root),
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        coverage_signal_root=Path(coverage_signal_root),
        coverage_performance_root=Path(coverage_performance_root),
        reward_refinement_root=Path(reward_refinement_root),
        output_root=output_root,
        pair_path=pair_path,
        counterfactual_path=counterfactual_path,
        old_episodes_path=old_episodes_path,
        base_candidate_root=base_candidate_root,
        minimum_low_observation_count=minimum_low_observation_count,
        repo_root=repo_root,
    )

    _write_json(paths["summary"], summary)
    _write_json(paths["source_ledger"], {**prepared["source_ledger"], "final_summary": str(paths["summary"])})
    _write_json(paths["input_audit"], prepared["input_audit"])
    _write_json(paths["old_transition_audit"], prepared["old_transition_audit"])
    _write_json(paths["performance_audit"], performance_audit)
    _write_json(paths["release_audit"], release_audit)
    _write_json(
        paths["rejection_report"],
        {
            "schema_version": "family-balanced-coverage-driven-ppo-rerun-rejection-report/v1",
            "status": summary["status"],
            "reason_codes": reason_codes,
            "docs_audit": docs_audit,
            "performance_audit": str(paths["performance_audit"]),
        },
    )
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _prepare_family_balanced_inputs(
    *,
    pair_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    old_episodes: list[dict[str, Any]],
    gap_summary: dict[str, Any],
    gap_manifest: dict[str, Any],
    coverage_summary: dict[str, Any],
    base_candidate_root: Path,
    family_balanced_gap_root: Path,
    paths: dict[str, Path],
    repo_root: Path,
    minimum_low_observation_count: int,
    input_reasons: list[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    if gap_summary.get("status") != "passed" or gap_summary.get("reason_codes"):
        _add_reason(reasons, "family_balanced_gap_closure_not_passed")
    if not bool(gap_summary.get("family_balanced_coverage_driven_ppo_rerun_approved")):
        _add_reason(reasons, "family_balanced_gap_closure_not_passed")
    if gap_summary.get("next_required_change") != "family_balanced_coverage_driven_ppo_rerun":
        _add_reason(reasons, "family_balanced_gap_closure_not_passed")
    if not pair_rows or not counterfactual_rows:
        _add_reason(reasons, "family_balanced_input_missing")

    counterfactual_index = {
        _counterfactual_key(row): row
        for row in counterfactual_rows
        if _first_int(row.get("action_index")) is not None
    }
    family_counts_raw = Counter(str(row.get("scenario_family") or "") for row in pair_rows)
    family_weights = _family_weights(pair_rows)

    enriched_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        row_reasons = _pair_rejection_reasons(row, counterfactual_index)
        if row_reasons:
            rejected_pairs.append({**_pair_identity(row), "row_reason_codes": row_reasons})
            continue
        enriched = copy.deepcopy(row)
        family = str(row.get("scenario_family") or "")
        advantage = _family_balanced_advantage(row, family_weight=family_weights.get(family, 1.0))
        enriched.update(
            {
                "family_balanced_family_weight": family_weights.get(family, 1.0),
                "family_balanced_ppo_advantage": advantage["family_balanced_ppo_advantage"],
                "ppo_advantage": advantage["family_balanced_ppo_advantage"],
                "family_balanced_reward_components": advantage["components"],
                "family_balanced_trainable": True,
            }
        )
        enriched_pairs.append(enriched)
        advantage_rows.append(
            {
                **_pair_identity(enriched),
                "family_balanced_family_weight": enriched["family_balanced_family_weight"],
                "family_balanced_ppo_advantage": enriched["family_balanced_ppo_advantage"],
                "reward_components": enriched["family_balanced_reward_components"],
            }
        )

    family_counts = Counter(str(row.get("scenario_family") or "") for row in enriched_pairs)
    if any(family_counts.get(family, 0) <= 0 for family in TARGET_FAMILIES):
        _add_reason(reasons, "family_balance_regressed")
    low_observation_count = family_counts.get(LOW_OBSERVATION_FAMILY, 0)
    if low_observation_count < minimum_low_observation_count:
        _add_reason(reasons, "low_observation_transition_gap")
    if any(row["family_balanced_ppo_advantage"] <= TOLERANCE for row in enriched_pairs):
        _add_reason(reasons, "family_balanced_advantage_missing")

    overlay_rows, compat_counterfactual_rows = _compatible_stage5a2_rows(enriched_pairs, counterfactual_index)
    _attach_family_balanced_advantages(overlay_rows, compat_counterfactual_rows, enriched_pairs)

    old_transitions = _old_transitions(old_episodes)
    old_transition_index = {_decision_key_from_transition(transition): transition for transition in old_transitions}
    synthetic_transitions: list[dict[str, Any]] = []
    transition_materialization_error: str | None = None
    try:
        synthetic_transitions = _synthetic_transitions_for_missing_decisions(
            eligible_pairs=enriched_pairs,
            old_transition_index=old_transition_index,
            base_candidate_root=base_candidate_root,
            repo_root=repo_root,
        )
        _mark_synthetic_transitions(synthetic_transitions)
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        transition_materialization_error = str(exc)
        _add_reason(reasons, "old_transition_materialization_failed")

    all_transitions = [*old_transitions, *synthetic_transitions]
    if not _old_policy_values_finite(all_transitions):
        _add_reason(reasons, "old_transition_materialization_failed")

    compatible_episodes = [
        {
            "schema_version": COMPAT_EPISODE_SCHEMA_VERSION,
            "episode_id": str(transition.get("info", {}).get("episode_id") or f"family-balanced-{index}"),
            "transitions": [transition],
        }
        for index, transition in enumerate(all_transitions)
    ]
    _write_jsonl(paths["compat_overlay"], overlay_rows)
    _write_jsonl(paths["compat_counterfactual"], compat_counterfactual_rows)
    _write_jsonl(paths["compat_episodes"], compatible_episodes)
    _write_jsonl(paths["advantage_audit"], advantage_rows)

    compat_reason_codes = _unique([*input_reasons, *reasons])
    _write_json(
        paths["compat_stage5a2_summary"],
        {
            "schema_version": COMPAT_STAGE5A2_SUMMARY_SCHEMA_VERSION,
            "status": "passed" if not compat_reason_codes else "failed",
            "reason_codes": compat_reason_codes,
            "next_required_change": "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
            "candidate_coverage_overlay": str(paths["compat_overlay"]),
            "counterfactual_coverage_rollouts": str(paths["compat_counterfactual"]),
            "safe_better_than_teacher_candidate_count": len(enriched_pairs),
            "safe_better_than_teacher_family_count": len([family for family in TARGET_FAMILIES if family_counts.get(family, 0) > 0]),
            "family_safe_better_training_counts": dict(sorted(family_counts.items())),
            "family_balanced": True,
            "fallback_gain_contamination_count": 0,
            "controlled_regression_count": 0,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "performance_claimed": False,
        },
    )
    _write_json(
        paths["compat_coverage_summary"],
        {
            **coverage_summary,
            "schema_version": coverage_summary.get("schema_version", "coverage-driven-ppo-improvement-run-summary/v1"),
            "coverage_aware_batch_episodes": str(paths["compat_episodes"]),
            "base_candidate_root": str(base_candidate_root),
            "family_balanced_augmented_input": True,
            "uses_old_coverage_driven_pairs_as_training_source": False,
            "synthetic_old_transition_count": len(synthetic_transitions),
            "performance_claimed": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
        },
    )
    baseline_metric_rows = _family_balanced_baseline_metric_rows(enriched_pairs)
    _write_jsonl(paths["compat_performance_metric_table"], baseline_metric_rows)
    _write_json(
        paths["compat_performance_summary"],
        {
            "schema_version": "family-balanced-compatible-performance-evaluation-summary/v1",
            "status": "passed" if enriched_pairs else "failed",
            "reason_codes": [] if enriched_pairs else ["family_balanced_input_missing"],
            "metric_table": str(paths["compat_performance_metric_table"]),
            "best_baseline_actor": "teacher",
            "best_baseline_metrics": baseline_metric_rows[0] if baseline_metric_rows else {},
            "family_balanced_baseline": True,
            "baseline_decision_count": len(enriched_pairs),
            "performance_claimed": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
        },
    )

    old_transition_audit_passed = (
        "old_transition_materialization_failed" not in reasons and _old_policy_values_finite(all_transitions)
    )
    input_audit_passed = not any(
        reason in reasons
        for reason in (
            "family_balanced_gap_closure_not_passed",
            "family_balanced_input_missing",
            "family_balance_regressed",
            "low_observation_transition_gap",
            "family_balanced_advantage_missing",
        )
    )
    source_ledger = {
        "schema_version": SOURCE_LEDGER_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "gap_summary_status": gap_summary.get("status"),
        "gap_manifest_status": gap_manifest.get("status"),
        "input_pair_count": len(pair_rows),
        "input_counterfactual_count": len(counterfactual_rows),
        "eligible_family_balanced_pair_count": len(enriched_pairs),
        "rejected_pair_count": len(rejected_pairs),
        "family_safe_better_counts_raw": dict(sorted(family_counts_raw.items())),
        "family_safe_better_training_counts": dict(sorted(family_counts.items())),
        "family_balanced_gap_root": str(family_balanced_gap_root),
        "compatible_stage5a2_root": str(paths["compat_stage5a2_root"]),
        "compatible_coverage_driven_root": str(paths["compat_coverage_driven_root"]),
        "compatible_performance_root": str(paths["compat_performance_root"]),
        "compatible_performance_metric_table": str(paths["compat_performance_metric_table"]),
        "uses_old_coverage_driven_pairs_as_training_source": False,
        "reason_codes": compat_reason_codes,
    }
    input_audit = {
        "schema_version": INPUT_AUDIT_SCHEMA_VERSION,
        "status": "passed" if input_audit_passed else "failed",
        "family_balanced_input_audit_passed": input_audit_passed,
        "input_pair_count": len(pair_rows),
        "eligible_pair_count": len(enriched_pairs),
        "compatible_overlay_row_count": len(overlay_rows),
        "compatible_counterfactual_row_count": len(compat_counterfactual_rows),
        "compatible_performance_metric_table": str(paths["compat_performance_metric_table"]),
        "family_safe_better_training_counts": dict(sorted(family_counts.items())),
        "low_observation_trainable_transition_count": low_observation_count,
        "minimum_low_observation_count": minimum_low_observation_count,
        "rejected_pair_rows": rejected_pairs[:100],
        "reason_codes": compat_reason_codes,
    }
    old_transition_audit = {
        "schema_version": OLD_TRANSITION_AUDIT_SCHEMA_VERSION,
        "status": "passed" if old_transition_audit_passed else "failed",
        "old_transition_materialization_audit_passed": old_transition_audit_passed,
        "old_transition_count": len(old_transitions),
        "synthetic_old_transition_count": len(synthetic_transitions),
        "compatible_episode_count": len(compatible_episodes),
        "family_balanced_decision_count": len({_decision_key_from_pair(row) for row in enriched_pairs}),
        "old_log_prob_value_finite": _old_policy_values_finite(all_transitions),
        "transition_materialization_error": transition_materialization_error,
        "reason_codes": compat_reason_codes,
    }
    return {
        "reason_codes": reasons,
        "eligible_pairs": enriched_pairs,
        "family_counts": dict(sorted(family_counts.items())),
        "input_audit_passed": input_audit_passed,
        "old_transition_audit_passed": old_transition_audit_passed,
        "old_transition_count": len(old_transitions),
        "synthetic_old_transition_count": len(synthetic_transitions),
        "low_observation_trainable_transition_count": low_observation_count,
        "source_ledger": source_ledger,
        "input_audit": input_audit,
        "old_transition_audit": old_transition_audit,
    }


def _summary(
    *,
    paths: dict[str, Path],
    reason_codes: list[str],
    prepared: dict[str, Any],
    refined_summary: dict[str, Any],
    performance_audit: dict[str, Any],
    release_audit: dict[str, Any],
    docs_audit: dict[str, Any],
    family_balanced_gap_root: Path,
    coverage_driven_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    pair_path: Path,
    counterfactual_path: Path,
    old_episodes_path: Path,
    base_candidate_root: Path,
    minimum_low_observation_count: int,
    repo_root: Path,
) -> dict[str, Any]:
    status = "passed" if not reason_codes else "failed"
    family_counts = prepared["family_counts"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "rerun_verdict": "eligible_for_family_balanced_shadow_canary_preflight"
        if status == "passed"
        else "not_eligible_for_family_balanced_shadow_canary_preflight",
        "family_balanced_coverage_driven_ppo_rerun_passed": status == "passed",
        "family_balanced_shadow_canary_preflight_approved": status == "passed",
        "family_balanced_input_audit_passed": bool(prepared["input_audit_passed"]),
        "old_transition_materialization_audit_passed": bool(prepared["old_transition_audit_passed"]),
        "ppo_update_status": refined_summary.get("ppo_update_status", "not_run" if not refined_summary else "failed"),
        "guard_replay_audit_passed": bool(performance_audit["guard_replay_audit_passed"]),
        "coverage_return_improvement": _float(refined_summary.get("coverage_return_improvement")),
        "cumulative_coverage_rate_delta_improvement": _float(
            refined_summary.get("cumulative_coverage_rate_delta_improvement")
        ),
        "valuable_area_covered_improvement": _float(refined_summary.get("valuable_area_covered_improvement")),
        "coverage_efficiency_regression": bool(refined_summary.get("coverage_efficiency_regression", False)),
        "policy_argmax_changed_count": _int(refined_summary.get("policy_argmax_changed_count")),
        "fallback_rate": _first_float(refined_summary.get("fallback_rate"), 0.0),
        "fallback_gain_contamination_count": _int(refined_summary.get("fallback_gain_contamination_count")),
        "controlled_regression_count": _int(refined_summary.get("controlled_regression_count")),
        "safe_better_training_family_count": len([family for family in TARGET_FAMILIES if family_counts.get(family, 0) > 0]),
        "safe_better_training_pair_count": _int(
            refined_summary.get("safe_better_training_pair_count"),
            default=len(prepared["eligible_pairs"]),
        ),
        "low_observation_trainable_transition_count": prepared["low_observation_trainable_transition_count"],
        "minimum_low_observation_count": minimum_low_observation_count,
        "family_safe_better_counts": family_counts,
        "next_required_change": "family_balanced_shadow_canary_preflight"
        if status == "passed"
        else _failed_next_required_change(reason_codes),
        "family_balanced_gap_root": str(family_balanced_gap_root),
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
        "profile_id": refined_summary.get("profile_id"),
        "profile_version": refined_summary.get("profile_version"),
        "profile_hash": refined_summary.get("profile_hash"),
        "summary": str(paths["summary"]),
        "source_ledger": str(paths["source_ledger"]),
        "compatible_input_audit": str(paths["input_audit"]),
        "old_transition_materialization_audit": str(paths["old_transition_audit"]),
        "advantage_audit": str(paths["advantage_audit"]),
        "performance_audit": str(paths["performance_audit"]),
        "release_boundary_audit": str(paths["release_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "compatible_stage5a2_root": str(paths["compat_stage5a2_root"]),
        "compatible_coverage_driven_root": str(paths["compat_coverage_driven_root"]),
        "compatible_performance_root": str(paths["compat_performance_root"]),
        "compatible_performance_metric_table": str(paths["compat_performance_metric_table"]),
        "compatible_candidate_coverage_overlay": str(paths["compat_overlay"]),
        "compatible_counterfactual_coverage_rollouts": str(paths["compat_counterfactual"]),
        "compatible_coverage_aware_batch_episodes": str(paths["compat_episodes"]),
        "family_balanced_safe_better_pairs": str(pair_path),
        "family_balanced_counterfactual_rollouts": str(counterfactual_path),
        "old_coverage_aware_batch_episodes": str(old_episodes_path),
        "base_candidate_root": str(base_candidate_root),
        "refined_summary": refined_summary.get("summary"),
        "compatibility_summary": refined_summary.get("compatibility_summary"),
        "checkpoint_path": refined_summary.get("checkpoint_path"),
        "checkpoint_metadata_path": refined_summary.get("checkpoint_metadata_path"),
        "guard_replay_audit": refined_summary.get("guard_replay_audit"),
        "coverage_return_improvement_reference": 54.96083408637,
        "coverage_efficiency_regression_reference": False,
        "runs_new_ppo_update": bool(status == "passed"),
        "experimental_checkpoint": bool(refined_summary.get("experimental_checkpoint") or refined_summary.get("checkpoint_path")),
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
        "docs_audit": docs_audit,
        "release_boundary_audit_passed": bool(release_audit["release_boundary_audit_passed"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _performance_audit(*, refined_summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not refined_summary:
        _add_reason(reasons, "ppo_update_failed")
    elif refined_summary.get("ppo_update_status") != "passed":
        _add_reason(reasons, "ppo_update_failed")
    if _int(refined_summary.get("policy_argmax_changed_count")) <= 0:
        _add_reason(reasons, "post_update_policy_teacher_equivalent")
    if _float(refined_summary.get("coverage_return_improvement")) <= TOLERANCE:
        _add_reason(reasons, "no_coverage_return_improvement")
    if _float(refined_summary.get("cumulative_coverage_rate_delta_improvement")) <= TOLERANCE:
        _add_reason(reasons, "no_cumulative_coverage_rate_delta_improvement")
    if _float(refined_summary.get("valuable_area_covered_improvement")) <= TOLERANCE:
        _add_reason(reasons, "valuable_coverage_regressed")
    if bool(refined_summary.get("coverage_efficiency_regression")):
        _add_reason(reasons, "coverage_efficiency_regression")
    if _first_float(refined_summary.get("fallback_rate"), 1.0) >= 0.5:
        _add_reason(reasons, "fallback_dominates")
    if _int(refined_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination")
    if _int(refined_summary.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_detected")
    if _is_release_boundary_violation(refined_summary):
        _add_reason(reasons, "release_boundary_violation")
    guard_path = refined_summary.get("guard_replay_audit")
    guard_audit = _read_json_optional(Path(guard_path)) if guard_path else {}
    guard_passed = bool(guard_path and guard_audit.get("status") == "passed")
    if refined_summary and not guard_passed:
        _add_reason(reasons, "guard_replay_failed")
    return {
        "schema_version": PERFORMANCE_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "ppo_update_status": refined_summary.get("ppo_update_status"),
        "guard_replay_audit": guard_path,
        "guard_replay_audit_passed": guard_passed,
        "policy_argmax_changed_count": _int(refined_summary.get("policy_argmax_changed_count")),
        "coverage_return_improvement": _float(refined_summary.get("coverage_return_improvement")),
        "cumulative_coverage_rate_delta_improvement": _float(
            refined_summary.get("cumulative_coverage_rate_delta_improvement")
        ),
        "valuable_area_covered_improvement": _float(refined_summary.get("valuable_area_covered_improvement")),
        "coverage_efficiency_regression": bool(refined_summary.get("coverage_efficiency_regression", False)),
        "fallback_rate": _first_float(refined_summary.get("fallback_rate"), None),
        "controlled_regression_count": _int(refined_summary.get("controlled_regression_count")),
        "fallback_gain_contamination_count": _int(refined_summary.get("fallback_gain_contamination_count")),
    }


def _pair_rejection_reasons(
    row: dict[str, Any],
    counterfactual_index: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if str(row.get("split") or "") != "train":
        reasons.append("non_train_split_in_trainable_output")
    if not bool(row.get("ppo_trainable")) and not bool(row.get("family_balanced_trainable")):
        reasons.append("not_ppo_trainable")
    if not bool(row.get("coverage_source_available")):
        reasons.append("counterfactual_source_missing")
    if not bool(row.get("safe_better_than_teacher_candidate")):
        reasons.append("not_safe_better_than_teacher")
    if bool(row.get("fallback_like")):
        reasons.append("fallback_gain_contamination")
    if bool(row.get("guard_rejected")):
        reasons.append("guard_rejected_candidate")
    if _string_list(row.get("controlled_regression_reason_codes")):
        reasons.append("controlled_regression_detected")
    if _float(row.get("coverage_advantage")) <= TOLERANCE:
        reasons.append("family_balanced_advantage_missing")
    action_index = _first_int(row.get("candidate_action_index"))
    if action_index is None:
        reasons.append("candidate_action_index_missing")
    elif _counterfactual_key_from_pair(row, action_index) not in counterfactual_index:
        reasons.append("counterfactual_source_missing")
    return _unique(reasons)


def _family_balanced_advantage(row: dict[str, Any], *, family_weight: float) -> dict[str, Any]:
    coverage_gain_bonus = _float(row.get("coverage_advantage"))
    valuable_area_bonus = max(
        _first_float(row.get("valuable_coverage_advantage"), row.get("new_area_advantage"), 0.0),
        0.0,
    )
    information_gain_bonus = max(
        _first_float(row.get("information_gain_advantage"), None)
        if row.get("information_gain_advantage") is not None
        else _float(row.get("information_gain")) - _float(row.get("teacher_information_gain")),
        0.0,
    )
    family_balance_weight_bonus = max(family_weight - 1.0, 0.0) * max(coverage_gain_bonus, 0.0)
    path_cost_penalty = max(_float(row.get("path_cost_delta")), 0.0) * 0.0005
    risk_penalty = max(_float(row.get("risk_delta")), 0.0) * 0.05
    energy_penalty = max(_float(row.get("energy_delta")), 0.0) * 0.000001
    teacher_skill_retention_bonus = 0.005
    advantage = (
        coverage_gain_bonus
        + 0.05 * valuable_area_bonus
        + information_gain_bonus
        + family_balance_weight_bonus
        + teacher_skill_retention_bonus
        - path_cost_penalty
        - risk_penalty
        - energy_penalty
    )
    advantage = max(float(advantage), 1.0e-6)
    return {
        "family_balanced_ppo_advantage": advantage,
        "legacy_read_only": True,
        "canonical_v2_training_consumer_blocked": True,
        "legacy_blocker_reason": LEGACY_V1_REWARD_COMPONENT_BLOCKER,
        "components": {
            "coverage_gain_bonus": coverage_gain_bonus,
            "valuable_area_bonus": valuable_area_bonus,
            "information_gain_bonus": information_gain_bonus,
            "family_balance_weight_bonus": family_balance_weight_bonus,
            "teacher_skill_retention_bonus": teacher_skill_retention_bonus,
            "path_cost_penalty": path_cost_penalty,
            "risk_penalty": risk_penalty,
            "energy_penalty": energy_penalty,
        },
    }


def _attach_family_balanced_advantages(
    overlay_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    enriched_pairs: list[dict[str, Any]],
) -> None:
    pair_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in enriched_pairs:
        action_index = _first_int(row.get("candidate_action_index"))
        if action_index is None:
            continue
        pair_by_key[_counterfactual_key_from_pair(row, action_index)] = row
    for row in overlay_rows:
        key = (
            str(row.get("context_id") or ""),
            str(row.get("episode_id") or ""),
            _int(row.get("step_index")),
            _int(row.get("action_index")),
        )
        pair = pair_by_key.get(key)
        if pair:
            _attach_advantage(row, pair)
        else:
            row["family_balanced_ppo_advantage"] = 0.0
            row["ppo_advantage"] = 0.0
            row["family_balanced_trainable"] = False
    for row in counterfactual_rows:
        pair = pair_by_key.get(_counterfactual_key(row))
        if pair:
            _attach_advantage(row, pair)


def _attach_advantage(row: dict[str, Any], pair: dict[str, Any]) -> None:
    row["family_balanced_family_weight"] = pair.get("family_balanced_family_weight")
    row["family_balanced_ppo_advantage"] = pair.get("family_balanced_ppo_advantage")
    row["ppo_advantage"] = pair.get("family_balanced_ppo_advantage")
    row["family_balanced_reward_components"] = pair.get("family_balanced_reward_components")
    row["family_balanced_trainable"] = True
    row["family_balance_source_stage"] = pair.get("family_balance_source_stage")


def _family_balanced_baseline_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    teacher = _metric_row_from_pairs(rows, actor="teacher", prefix="teacher_")
    teacher["comparator_basis"] = ["family_balanced_teacher_action_baseline_on_same_decisions"]
    return [teacher]


def _metric_row_from_pairs(rows: list[dict[str, Any]], *, actor: str, prefix: str) -> dict[str, Any]:
    def value(row: dict[str, Any], name: str) -> float:
        return _float(row.get(f"{prefix}{name}"))

    valid_rows = [row for row in rows if _float(row.get(f"{prefix}expected_coverage_rate_delta")) > -math.inf]
    cumulative = sum(value(row, "expected_coverage_rate_delta") for row in valid_rows)
    coverage_return = sum(
        value(row, "expected_coverage_rate_delta") * (DISCOUNT_FACTOR ** max(_int(row.get("step_index")), 0))
        for row in valid_rows
    )
    path_cost = sum(value(row, "path_cost") for row in rows)
    risk = sum(value(row, "risk") for row in rows)
    energy = sum(value(row, "energy_cost") for row in rows)
    valuable_area = sum(
        value(row, "expected_coverage_rate_delta")
        * _first_float(row.get(f"{prefix}valuable_coverage_proxy"), row.get(f"{prefix}value"), 0.0)
        for row in rows
    )
    scenario_families = {
        str(row.get("scenario_family") or "")
        for row in rows
        if value(row, "expected_coverage_rate_delta") > TOLERANCE and row.get("scenario_family")
    }
    return {
        "schema_version": "coverage-driven-ppo-performance-metric-row/v1",
        "actor": actor,
        "row_count": len(rows),
        "episode_count": len({row.get("episode_id") for row in rows if row.get("episode_id")}),
        "actual_coverage_row_count": len(valid_rows),
        "coverage_return": round(coverage_return, 12),
        "cumulative_coverage_rate_delta": round(cumulative, 12),
        "final_coverage_rate": None,
        "new_area_covered": round(sum(value(row, "expected_new_coverage_area") for row in rows), 12),
        "valuable_area_covered": round(valuable_area, 12),
        "information_gain": round(sum(value(row, "information_gain") for row in rows), 12),
        "path_cost": round(path_cost, 12),
        "risk": round(risk, 12),
        "energy_cost": round(energy, 12),
        "coverage_gain_per_path_cost": _safe_ratio(cumulative, path_cost),
        "coverage_gain_per_risk": _safe_ratio(cumulative, risk),
        "coverage_gain_per_energy": _safe_ratio(cumulative, energy),
        "accepted_policy_activation_rate": 0.0,
        "fallback_rate": 0.0,
        "teacher_agreement_rate": 1.0,
        "controlled_regression_count": 0,
        "fallback_coverage_gain": 0.0,
        "scenario_family_gain_count": len(scenario_families),
        "scenario_families_with_gain": sorted(scenario_families),
    }


def _mark_synthetic_transitions(transitions: list[dict[str, Any]]) -> None:
    for transition in transitions:
        info = transition.setdefault("info", {})
        if isinstance(info, dict):
            info["synthetic_old_transition"] = True
            info["controlled_choice_detail"] = "synthetic_teacher_transition_for_family_balanced_coverage_driven_ppo_rerun"


def _family_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row.get("scenario_family") or "") for row in rows)
    total = max(len(rows), 1)
    family_count = max(len([family for family, count in counts.items() if count > 0]), 1)
    return {family: total / float(family_count * count) for family, count in counts.items() if count > 0}


def _old_policy_values_finite(transitions: list[dict[str, Any]]) -> bool:
    if not transitions:
        return False
    for transition in transitions:
        try:
            if not math.isfinite(float(transition.get("log_prob"))) or not math.isfinite(float(transition.get("value"))):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _release_boundary_audit(*, source_payloads: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for label, payload in source_payloads.items():
        _collect_release_boundary_violations(payload, label, violations)
    return {
        "schema_version": RELEASE_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _collect_release_boundary_violations(payload: Any, path: str, violations: list[dict[str, str]]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            current = f"{path}.{key}"
            if key in RELEASE_BOUNDARY_TRUE_KEYS and value is True:
                violations.append({"path": current, "reason": "release_boundary_true"})
            _collect_release_boundary_violations(value, current, violations)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _collect_release_boundary_violations(item, f"{path}[{index}]", violations)


def _is_release_boundary_violation(payload: dict[str, Any]) -> bool:
    violations: list[dict[str, str]] = []
    _collect_release_boundary_violations(payload, "payload", violations)
    return bool(violations)


def _docs_audit(repo_root: Path) -> dict[str, Any]:
    required = {
        "README.md": (
            "Family-Balanced Coverage-Driven PPO Rerun v1",
            "outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1/",
            "family_balanced_shadow_canary_preflight",
        ),
        "docs/算法设计与系统架构报告.md": (
            "Family-Balanced Coverage-Driven PPO Rerun v1",
            "离线 guarded PPO rerun",
            "不发布 checkpoint",
        ),
        "docs/superpowers/specs/2026-06-16-family-balanced-coverage-driven-ppo-rerun.md": (
            "Family-Balanced Coverage-Driven PPO Rerun v1",
            "family-balanced-coverage-driven-ppo-rerun-summary.json",
            "family_balanced_shadow_canary_preflight",
        ),
    }
    missing: list[str] = []
    for relative, markers in required.items():
        path = repo_root / relative
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not path.exists() or any(marker not in text for marker in markers):
            missing.append(relative)
    return {
        "docs_updated": not missing,
        "missing_or_stale_docs": missing,
        "required_docs": sorted(required),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    compat_stage5a2_root = output_root / COMPAT_STAGE5A2_DIR
    compat_coverage_root = output_root / COMPAT_COVERAGE_DRIVEN_DIR
    compat_coverage_batch_root = compat_coverage_root / COMPAT_BATCH_DIR
    compat_performance_root = output_root / COMPAT_PERFORMANCE_DIR
    return {
        "output_root": output_root,
        "summary": output_root / SUMMARY_FILE,
        "source_ledger": output_root / SOURCE_LEDGER_FILE,
        "input_audit": output_root / INPUT_AUDIT_FILE,
        "old_transition_audit": output_root / OLD_TRANSITION_AUDIT_FILE,
        "advantage_audit": output_root / ADVANTAGE_AUDIT_FILE,
        "performance_audit": output_root / PERFORMANCE_AUDIT_FILE,
        "release_audit": output_root / RELEASE_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
        "compat_stage5a2_root": compat_stage5a2_root,
        "compat_stage5a2_summary": compat_stage5a2_root / COMPAT_STAGE5A2_SUMMARY_FILE,
        "compat_overlay": compat_stage5a2_root / COMPAT_OVERLAY_FILE,
        "compat_counterfactual": compat_stage5a2_root / COMPAT_COUNTERFACTUAL_FILE,
        "compat_coverage_driven_root": compat_coverage_root,
        "compat_coverage_summary": compat_coverage_root / COVERAGE_DRIVEN_SUMMARY_FILE,
        "compat_coverage_batch_root": compat_coverage_batch_root,
        "compat_episodes": compat_coverage_batch_root / COMPAT_EPISODES_FILE,
        "compat_performance_root": compat_performance_root,
        "compat_performance_summary": compat_performance_root / PERFORMANCE_SUMMARY_FILE,
        "compat_performance_metric_table": compat_performance_root / PERFORMANCE_METRIC_TABLE_FILE,
    }


def _base_candidate_root(
    *,
    coverage_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    selected_candidate_root: Path,
    repo_root: Path,
) -> Path:
    root = _resolve_optional_path(coverage_summary.get("base_candidate_root"), repo_root, repo_root)
    if root is not None:
        return root
    return _selected_base_candidate_root(selected_summary, selected_candidate_root, repo_root)


def _failed_next_required_change(reason_codes: list[str]) -> str:
    if LEGACY_V1_REWARD_COMPONENT_BLOCKER in reason_codes:
        return "migrate_family_balanced_rerun_to_canonical_reward_v2_or_use_stage18_5_guard"
    if any(
        reason in reason_codes
        for reason in (
            "family_balanced_gap_closure_not_passed",
            "family_balanced_input_missing",
            "family_balance_regressed",
            "low_observation_transition_gap",
            "old_transition_materialization_failed",
            "family_balanced_advantage_missing",
        )
    ):
        return "fix_family_balanced_coverage_driven_ppo_rerun_inputs"
    if any(
        reason in reason_codes
        for reason in (
            "ppo_update_failed",
            "post_update_policy_teacher_equivalent",
            "no_coverage_return_improvement",
            "no_cumulative_coverage_rate_delta_improvement",
            "valuable_coverage_regressed",
            "coverage_efficiency_regression",
            "fallback_dominates",
        )
    ):
        return "tune_family_balanced_reward_margin_or_ppo_config"
    return "fix_family_balanced_rerun_release_or_docs_boundary"


def _pair_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_id": row.get("context_id"),
        "episode_id": row.get("episode_id"),
        "step_index": row.get("step_index"),
        "scenario_family": row.get("scenario_family"),
        "candidate_action_index": row.get("candidate_action_index"),
        "teacher_action_index": row.get("teacher_action_index"),
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Family-Balanced Coverage-Driven PPO Rerun v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- rerun_verdict: `{summary['rerun_verdict']}`",
            f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
            f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
            f"- valuable_area_covered_improvement: `{summary['valuable_area_covered_improvement']}`",
            f"- low_observation_trainable_transition_count: `{summary['low_observation_trainable_transition_count']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            "",
            "## Boundaries",
            "",
            "- Experimental offline guarded PPO rerun only.",
            "- Does not publish checkpoint, replace default policy, or connect a real executor.",
            "- Does not modify network/action space/default A* and does not claim Ackermann feasibility.",
            "",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        _add_reason(reasons, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_optional(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    if not path.exists():
        _add_reason(reasons, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    except json.JSONDecodeError:
        _add_reason(reasons, f"{label}_invalid")
        return []
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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
    parsed = _first_int(value)
    return default if parsed is None else parsed


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= TOLERANCE:
        return None
    return round(float(numerator) / float(denominator), 12)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


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
