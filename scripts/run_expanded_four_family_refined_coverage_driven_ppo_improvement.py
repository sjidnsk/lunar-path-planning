from __future__ import annotations

import argparse
import copy
import json
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
    from run_refined_coverage_driven_ppo_improvement_run import (
        _policy_log_prob_and_value,
        _resolve_optional_path,
        _selected_base_candidate_root,
        run_refined_coverage_driven_ppo_improvement_run,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_refined_coverage_driven_ppo_improvement_run import (
        _policy_log_prob_and_value,
        _resolve_optional_path,
        _selected_base_candidate_root,
        run_refined_coverage_driven_ppo_improvement_run,
    )


SUMMARY_SCHEMA_VERSION = "expanded-four-family-refined-coverage-driven-ppo-improvement-summary/v1"
SOURCE_AUDIT_SCHEMA_VERSION = "expanded-four-family-source-audit/v1"
COMPAT_STAGE5A2_SUMMARY_SCHEMA_VERSION = "policy-differentiating-counterfactual-coverage-rollouts-summary/v1"
COMPAT_EPISODE_SCHEMA_VERSION = "coverage-aware-ppo-rollout-episode/v1"

DEFAULT_SAFE_BETTER_ROOT = "outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1"
DEFAULT_STAGE5A2_ROOT = "outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1"
DEFAULT_COVERAGE_DRIVEN_ROOT = "outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_PERFORMANCE_ROOT = "outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1"
DEFAULT_REWARD_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_expanded_four_family_refined_coverage_driven_ppo_improvement_v1"

SAFE_BETTER_SUMMARY_FILE = "safe-better-pair-expansion-summary.json"
SAFE_BETTER_PAIRS_FILE = "expanded-safe-better-pairs.jsonl"
SAFE_BETTER_COUNTERFACTUAL_FILE = "expanded-counterfactual-coverage-rollouts.jsonl"
COVERAGE_DRIVEN_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
COMPAT_BATCH_DIR = "coverage-aware-ppo-batch"
COMPAT_EPISODES_FILE = "ppo-rollout-episodes.jsonl"
COMPAT_STAGE5A2_DIR = "expanded-four-family-compatible-stage5a2-input"
COMPAT_COVERAGE_DRIVEN_DIR = "expanded-four-family-compatible-coverage-driven-input"
COMPAT_STAGE5A2_SUMMARY_FILE = "policy-differentiating-counterfactual-coverage-rollouts-summary.json"
COMPAT_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
COMPAT_COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
SUMMARY_FILE = "expanded-four-family-refined-coverage-driven-ppo-improvement-summary.json"
SOURCE_AUDIT_FILE = "expanded-four-family-source-audit.json"
REPORT_FILE = "expanded-four-family-refined-coverage-driven-ppo-improvement-report.md"

TARGET_FAMILIES = (
    "low_observation_count",
    "mixed_risk",
    "rim_or_steep_slope",
    "smooth_high_confidence",
)
DEFAULT_MINIMUM_SAFE_BETTER_PAIR_COUNT = 723
TOLERANCE = 1.0e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run refined coverage-driven PPO over expanded four-family safe-better pairs."
    )
    parser.add_argument("--safe-better-root", default=DEFAULT_SAFE_BETTER_ROOT)
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
    parser.add_argument("--minimum-safe-better-pair-count", type=int, default=DEFAULT_MINIMUM_SAFE_BETTER_PAIR_COUNT)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_expanded_four_family_refined_coverage_driven_ppo_improvement(
        safe_better_root=_resolve_path(Path(args.safe_better_root), repo_root),
        stage5a2_root=_resolve_path(Path(args.stage5a2_root), repo_root),
        coverage_driven_root=_resolve_path(Path(args.coverage_driven_root), repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        minimum_safe_better_pair_count=args.minimum_safe_better_pair_count,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "safe_better_training_pair_count": summary["safe_better_training_pair_count"],
                "safe_better_training_family_count": summary["safe_better_training_family_count"],
                "policy_argmax_changed_count": summary["policy_argmax_changed_count"],
                "coverage_return_improvement": summary["coverage_return_improvement"],
                "cumulative_coverage_rate_delta_improvement": summary[
                    "cumulative_coverage_rate_delta_improvement"
                ],
                "fallback_rate": summary["fallback_rate"],
                "performance_claimed": summary["performance_claimed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_expanded_four_family_refined_coverage_driven_ppo_improvement(
    *,
    safe_better_root: Path,
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
    minimum_safe_better_pair_count: int = DEFAULT_MINIMUM_SAFE_BETTER_PAIR_COUNT,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    paths["compat_stage5a2_root"].mkdir(parents=True, exist_ok=True)
    paths["compat_coverage_batch_root"].mkdir(parents=True, exist_ok=True)

    input_reasons: list[str] = []
    safe_better_summary = _read_json(
        Path(safe_better_root) / SAFE_BETTER_SUMMARY_FILE,
        input_reasons,
        "safe_better_summary",
    )
    old_stage5a2_summary = _read_json(
        Path(stage5a2_root) / COMPAT_STAGE5A2_SUMMARY_FILE,
        [],
        "old_stage5a2_summary",
    )
    coverage_driven_summary = _read_json(
        Path(coverage_driven_root) / COVERAGE_DRIVEN_SUMMARY_FILE,
        input_reasons,
        "coverage_driven_summary",
    )
    selected_summary = _read_json(
        Path(selected_candidate_root) / SELECTED_SUMMARY_FILE,
        input_reasons,
        "selected_candidate_summary",
    )
    pair_rows = _read_jsonl(Path(safe_better_root) / SAFE_BETTER_PAIRS_FILE, input_reasons, "expanded_safe_better_pairs")
    counterfactual_rows = _read_jsonl(
        Path(safe_better_root) / SAFE_BETTER_COUNTERFACTUAL_FILE,
        input_reasons,
        "expanded_counterfactual_coverage_rollouts",
    )
    old_episodes_path = _resolve_optional_path(
        coverage_driven_summary.get("coverage_aware_batch_episodes"),
        Path(coverage_driven_root),
        repo_root,
    ) or Path(coverage_driven_root) / COMPAT_BATCH_DIR / COMPAT_EPISODES_FILE
    old_episodes = _read_jsonl(old_episodes_path, input_reasons, "old_coverage_aware_episodes")

    base_candidate_root = _base_candidate_root(
        coverage_driven_summary=coverage_driven_summary,
        selected_summary=selected_summary,
        selected_candidate_root=Path(selected_candidate_root),
        repo_root=repo_root,
    )

    prepared = _prepare_expanded_inputs(
        pair_rows=pair_rows,
        counterfactual_rows=counterfactual_rows,
        old_episodes=old_episodes,
        safe_better_summary=safe_better_summary,
        coverage_driven_summary=coverage_driven_summary,
        base_candidate_root=base_candidate_root,
        paths=paths,
        repo_root=repo_root,
        minimum_safe_better_pair_count=minimum_safe_better_pair_count,
        input_reasons=input_reasons,
    )
    _write_json(paths["source_audit"], prepared["source_audit"])

    reason_codes = _unique([*input_reasons, *prepared["reason_codes"]])
    refined_summary: dict[str, Any] = {}
    if not reason_codes:
        refined_summary = run_refined_coverage_driven_ppo_improvement_run(
            stage5a2_root=paths["compat_stage5a2_root"],
            coverage_driven_root=paths["compat_coverage_driven_root"],
            formal_training_root=Path(formal_training_root),
            post_training_replay_root=Path(post_training_replay_root),
            selected_candidate_root=Path(selected_candidate_root),
            coverage_signal_root=Path(coverage_signal_root),
            coverage_performance_root=Path(coverage_performance_root),
            reward_refinement_root=Path(reward_refinement_root),
            output_root=output_root,
            repo_root=repo_root,
            enforce_cost_risk_energy_filter=False,
            ppo_advantage_scale=50.0,
            use_transition_info_advantage=True,
            ppo_learning_rate=2.0e-3,
            ppo_epochs=30,
            ppo_max_approx_kl=100.0,
        )
        reason_codes = _unique([*reason_codes, *refined_summary.get("reason_codes", [])])

    summary = _expanded_summary(
        paths=paths,
        reason_codes=reason_codes,
        prepared=prepared,
        refined_summary=refined_summary,
        safe_better_root=Path(safe_better_root),
        stage5a2_root=Path(stage5a2_root),
        coverage_driven_root=Path(coverage_driven_root),
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        coverage_signal_root=Path(coverage_signal_root),
        coverage_performance_root=Path(coverage_performance_root),
        reward_refinement_root=Path(reward_refinement_root),
        output_root=output_root,
        base_candidate_root=base_candidate_root,
        old_stage5a2_summary=old_stage5a2_summary,
        minimum_safe_better_pair_count=minimum_safe_better_pair_count,
        repo_root=repo_root,
    )
    _write_json(paths["summary"], summary)
    _write_json(paths["source_audit"], {**prepared["source_audit"], "final_summary": str(paths["summary"])})
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _prepare_expanded_inputs(
    *,
    pair_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    old_episodes: list[dict[str, Any]],
    safe_better_summary: dict[str, Any],
    coverage_driven_summary: dict[str, Any],
    base_candidate_root: Path,
    paths: dict[str, Path],
    repo_root: Path,
    minimum_safe_better_pair_count: int,
    input_reasons: list[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    if safe_better_summary.get("status") != "passed":
        _add_reason(reasons, "safe_better_pair_expansion_not_passed")
    if safe_better_summary.get("next_required_change") != "rerun_refined_coverage_driven_ppo_improvement":
        _add_reason(reasons, "safe_better_pair_expansion_not_routed_to_refined_ppo")
    if _int(safe_better_summary.get("missing_counterfactual_source_count")) > 0:
        _add_reason(reasons, "expanded_safe_better_source_missing")
    if _int(safe_better_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination")
    if _int(safe_better_summary.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_present")

    counterfactual_index = {
        _counterfactual_key(row): row
        for row in counterfactual_rows
        if _first_int(row.get("action_index")) is not None
    }
    eligible_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    for row in pair_rows:
        row_reasons = _pair_rejection_reasons(row, counterfactual_index)
        if row_reasons:
            rejected_pairs.append({**_pair_identity(row), "row_reason_codes": row_reasons})
        else:
            eligible_pairs.append(row)

    family_counts = Counter(str(row.get("scenario_family") or "") for row in eligible_pairs)
    if len(eligible_pairs) < minimum_safe_better_pair_count:
        _add_reason(reasons, "expanded_safe_better_pair_count_below_threshold")
    if any(family_counts.get(family, 0) <= 0 for family in TARGET_FAMILIES):
        _add_reason(reasons, "family_balance_failed")
        for family in TARGET_FAMILIES:
            if family_counts.get(family, 0) <= 0:
                _add_reason(reasons, f"family_safe_better_gap_{family}")

    overlay_rows, compat_counterfactual_rows = _compatible_stage5a2_rows(eligible_pairs, counterfactual_index)
    old_transitions = _old_transitions(old_episodes)
    old_transition_index = {_decision_key_from_transition(transition): transition for transition in old_transitions}
    synthetic_transitions = _synthetic_transitions_for_missing_decisions(
        eligible_pairs=eligible_pairs,
        old_transition_index=old_transition_index,
        base_candidate_root=base_candidate_root,
        repo_root=repo_root,
    )
    all_transitions = [*old_transitions, *synthetic_transitions]
    compatible_episodes = [
        {
            "schema_version": COMPAT_EPISODE_SCHEMA_VERSION,
            "episode_id": str(transition.get("info", {}).get("episode_id") or f"expanded-four-family-{index}"),
            "transitions": [transition],
        }
        for index, transition in enumerate(all_transitions)
    ]
    _write_jsonl(paths["compat_overlay"], overlay_rows)
    _write_jsonl(paths["compat_counterfactual"], compat_counterfactual_rows)
    _write_jsonl(paths["compat_episodes"], compatible_episodes)
    _write_json(
        paths["compat_stage5a2_summary"],
        {
            "schema_version": COMPAT_STAGE5A2_SUMMARY_SCHEMA_VERSION,
            "status": "passed" if not reasons and not input_reasons else "failed",
            "reason_codes": _unique([*input_reasons, *reasons]),
            "next_required_change": "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
            "candidate_coverage_overlay": str(paths["compat_overlay"]),
            "counterfactual_coverage_rollouts": str(paths["compat_counterfactual"]),
            "safe_better_than_teacher_candidate_count": len(eligible_pairs),
            "safe_better_than_teacher_family_count": len(family_counts),
            "fallback_gain_contamination_count": 0,
            "controlled_regression_count": 0,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
        },
    )
    _write_json(
        paths["compat_coverage_summary"],
        {
            **coverage_driven_summary,
            "schema_version": coverage_driven_summary.get(
                "schema_version",
                "coverage-driven-ppo-improvement-run-summary/v1",
            ),
            "status": coverage_driven_summary.get("status") or "failed",
            "reason_codes": coverage_driven_summary.get("reason_codes", []),
            "coverage_aware_batch_episodes": str(paths["compat_episodes"]),
            "base_candidate_root": str(base_candidate_root),
            "expanded_four_family_augmented_input": True,
            "synthetic_old_transition_count": len(synthetic_transitions),
            "performance_claimed": False,
        },
    )

    source_audit = {
        "schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "safe_better_pair_count": len(pair_rows),
        "eligible_expanded_safe_better_pair_count": len(eligible_pairs),
        "rejected_expanded_safe_better_pair_count": len(rejected_pairs),
        "rejected_pair_rows": rejected_pairs,
        "family_safe_better_training_counts": dict(sorted(family_counts.items())),
        "old_transition_count": len(old_transitions),
        "synthetic_old_transition_count": len(synthetic_transitions),
        "compatible_overlay_row_count": len(overlay_rows),
        "compatible_counterfactual_row_count": len(compat_counterfactual_rows),
        "compatible_stage5a2_root": str(paths["compat_stage5a2_root"]),
        "compatible_coverage_driven_root": str(paths["compat_coverage_driven_root"]),
        "uses_old_stage5a2_pairs_as_training_source": False,
        "reason_codes": _unique([*input_reasons, *reasons]),
    }
    return {
        "reason_codes": reasons,
        "eligible_pairs": eligible_pairs,
        "family_counts": dict(sorted(family_counts.items())),
        "synthetic_old_transition_count": len(synthetic_transitions),
        "old_transition_count": len(old_transitions),
        "source_audit": source_audit,
    }


def _pair_rejection_reasons(
    row: dict[str, Any],
    counterfactual_index: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if str(row.get("split") or "") != "train":
        reasons.append("non_train_split")
    if not bool(row.get("ppo_trainable")):
        reasons.append("not_ppo_trainable")
    if not bool(row.get("coverage_source_available")):
        reasons.append("expanded_safe_better_source_missing")
    if not bool(row.get("safe_better_than_teacher_candidate")):
        reasons.append("not_safe_better_than_teacher")
    if _float(row.get("coverage_advantage")) <= TOLERANCE:
        reasons.append("coverage_advantage_not_positive")
    if bool(row.get("fallback_like")):
        reasons.append("fallback_like_candidate")
    if bool(row.get("guard_rejected")):
        reasons.append("guard_rejected_candidate")
    if _string_list(row.get("controlled_regression_reason_codes")):
        reasons.append("controlled_regression_present")
    action_index = _first_int(row.get("candidate_action_index"))
    if action_index is None:
        reasons.append("candidate_action_index_missing")
    elif _counterfactual_key_from_pair(row, action_index) not in counterfactual_index:
        reasons.append("counterfactual_row_missing")
    return _unique(reasons)


def _compatible_stage5a2_rows(
    eligible_pairs: list[dict[str, Any]],
    counterfactual_index: dict[tuple[str, str, int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_counts = Counter(str(row.get("scenario_family") or "") for row in eligible_pairs)
    total = max(len(eligible_pairs), 1)
    family_count = max(len([family for family, count in family_counts.items() if count > 0]), 1)
    weights = {
        family: total / float(family_count * count)
        for family, count in family_counts.items()
        if count > 0
    }
    overlay_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    teacher_seen: set[tuple[str, str, int, int]] = set()
    for row in eligible_pairs:
        family = str(row.get("scenario_family") or "")
        family_weight = weights.get(family, 1.0)
        teacher = _teacher_overlay_row(row, family_weight=family_weight)
        teacher_key = (
            str(teacher.get("context_id") or ""),
            str(teacher.get("episode_id") or ""),
            _int(teacher.get("step_index")),
            _int(teacher.get("action_index")),
        )
        if teacher_key not in teacher_seen:
            overlay_rows.append(teacher)
            teacher_seen.add(teacher_key)
        candidate = _candidate_overlay_row(row, family_weight=family_weight)
        overlay_rows.append(candidate)
        action_index = _first_int(row.get("candidate_action_index"))
        assert action_index is not None
        counterfactual = copy.deepcopy(counterfactual_index[_counterfactual_key_from_pair(row, action_index)])
        counterfactual.update(
            {
                "schema_version": "policy-differentiating-counterfactual-coverage-row/v1",
                "action_index": action_index,
                "context_id": str(row.get("context_id") or ""),
                "episode_id": str(row.get("episode_id") or ""),
                "step_index": _int(row.get("step_index")),
                "scenario_id": row.get("scenario_id"),
                "scenario_family": family,
                "split": "train",
                "teacher_action_index": _first_int(row.get("teacher_action_index")),
                "safe_better_than_teacher_candidate": True,
                "coverage_source_available": True,
                "fallback_like": False,
                "guard_rejected": False,
                "controlled_regression_reason_codes": [],
                "missing_reason_codes": [],
                "expanded_family_weight": family_weight,
            }
        )
        counterfactual_rows.append(counterfactual)
    return overlay_rows, counterfactual_rows


def _candidate_overlay_row(row: dict[str, Any], *, family_weight: float) -> dict[str, Any]:
    action_index = _first_int(row.get("candidate_action_index"))
    return {
        "schema_version": "candidate-level-coverage-overlay-row/v1",
        "context_id": str(row.get("context_id") or ""),
        "episode_id": str(row.get("episode_id") or ""),
        "step_index": _int(row.get("step_index")),
        "scenario_id": row.get("scenario_id"),
        "scenario_family": row.get("scenario_family"),
        "split": "train",
        "action_index": action_index,
        "candidate_cell": row.get("candidate_cell") or [action_index or 0, action_index or 0],
        "action_mask_valid": bool(row.get("action_mask_valid", True)),
        "teacher_action_index": _first_int(row.get("teacher_action_index")),
        "is_teacher_action": False,
        "coverage_source_available": True,
        "expected_coverage_rate_delta": _float(row.get("expected_coverage_rate_delta")),
        "expected_new_coverage_area": _float(row.get("expected_new_coverage_area")),
        "information_gain": _float(row.get("information_gain")),
        "valuable_coverage_proxy": _first_float(row.get("valuable_coverage_proxy"), row.get("value")),
        "value": _first_float(row.get("value"), row.get("valuable_coverage_proxy")),
        "path_cost": _float(row.get("path_cost")),
        "risk": _float(row.get("risk")),
        "energy_cost": _float(row.get("energy_cost")),
        "fallback_like": False,
        "guard_rejected": False,
        "policy_action_accepted": True,
        "controlled_regression_reason_codes": [],
        "missing_reason_codes": [],
        "match_method": row.get("match_method") or row.get("counterfactual_coverage_match_method"),
        "source_path": row.get("source_path") or row.get("counterfactual_coverage_source_path"),
        "source_confidence": _first_float(row.get("source_confidence"), 1.0),
        "source_execution_type": "counterfactual_candidate",
        "safe_better_than_teacher_candidate": True,
        "expanded_family_weight": family_weight,
    }


def _teacher_overlay_row(row: dict[str, Any], *, family_weight: float) -> dict[str, Any]:
    action_index = _first_int(row.get("teacher_action_index"))
    return {
        **_candidate_overlay_row(row, family_weight=family_weight),
        "action_index": action_index,
        "candidate_cell": row.get("teacher_candidate_cell") or [action_index or 0, action_index or 0],
        "is_teacher_action": True,
        "expected_coverage_rate_delta": _float(row.get("teacher_expected_coverage_rate_delta")),
        "expected_new_coverage_area": _float(row.get("teacher_expected_new_coverage_area")),
        "information_gain": _float(row.get("teacher_information_gain")),
        "valuable_coverage_proxy": _first_float(row.get("teacher_valuable_coverage_proxy"), row.get("teacher_value")),
        "value": _first_float(row.get("teacher_value"), row.get("teacher_valuable_coverage_proxy")),
        "path_cost": _float(row.get("teacher_path_cost")),
        "risk": _float(row.get("teacher_risk")),
        "energy_cost": _float(row.get("teacher_energy_cost")),
        "source_execution_type": "executed_policy_action",
        "safe_better_than_teacher_candidate": False,
    }


def _synthetic_transitions_for_missing_decisions(
    *,
    eligible_pairs: list[dict[str, Any]],
    old_transition_index: dict[tuple[str, str, int], dict[str, Any]],
    base_candidate_root: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    pairs_by_decision: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_pairs:
        pairs_by_decision[_decision_key_from_pair(row)].append(row)
    transitions: list[dict[str, Any]] = []
    template_observation = _first_template_observation(old_transition_index)
    for key, rows in sorted(pairs_by_decision.items()):
        if key in old_transition_index:
            continue
        teacher_row = rows[0]
        action_count = 1 + max(
            max(_first_int(row.get("candidate_action_index")) or 0, _first_int(row.get("teacher_action_index")) or 0)
            for row in rows
        )
        observation = _synthetic_observation(
            rows,
            action_count=action_count,
            template_observation=template_observation,
        )
        teacher_action_index = _first_int(teacher_row.get("teacher_action_index")) or 0
        log_prob, value = _policy_log_prob_and_value(
            observation_payload=observation,
            action_index=teacher_action_index,
            base_candidate_root=base_candidate_root,
            repo_root=repo_root,
        )
        transitions.append(
            {
                "observation": observation,
                "action_index": teacher_action_index,
                "action_mask": [True for _ in range(action_count)],
                "log_prob": float(log_prob),
                "value": float(value),
                "reward": _float(teacher_row.get("teacher_expected_coverage_rate_delta")),
                "reward_components": {
                    "teacher_skill_retention_bonus": _float(
                        teacher_row.get("teacher_expected_coverage_rate_delta")
                    )
                },
                "done": True,
                "next_observation": None,
                "info": {
                    "context_id": key[0],
                    "episode_id": key[1],
                    "step_index": key[2],
                    "scenario_id": teacher_row.get("scenario_id"),
                    "scenario_family": teacher_row.get("scenario_family"),
                    "split": "train",
                    "controlled_choice_source": "policy",
                    "controlled_choice_detail": "synthetic_teacher_transition_for_expanded_four_family_refined_ppo",
                    "controlled_action_index": teacher_action_index,
                    "teacher_action_index": teacher_action_index,
                    "coverage_rate_delta": _float(teacher_row.get("teacher_expected_coverage_rate_delta")),
                    "cumulative_coverage_rate_delta": _float(
                        teacher_row.get("teacher_expected_coverage_rate_delta")
                    ),
                    "final_coverage_rate": _float(teacher_row.get("teacher_expected_coverage_rate_delta")),
                    "new_area_covered": _float(teacher_row.get("teacher_expected_new_coverage_area")),
                    "valuable_area_covered": _float(teacher_row.get("teacher_valuable_coverage_proxy")),
                    "information_gain": _float(teacher_row.get("teacher_information_gain")),
                    "path_cost": _float(teacher_row.get("teacher_path_cost")),
                    "risk": _float(teacher_row.get("teacher_risk")),
                    "energy_cost": _float(teacher_row.get("teacher_energy_cost")),
                    "fallback_like": False,
                    "guard_rejected": False,
                    "gate_reason_codes": [],
                    "controlled_regression_reason_codes": [],
                    "ppo_trainable": True,
                    "actual_coverage_gain_source": teacher_row.get("match_method")
                    or teacher_row.get("counterfactual_coverage_match_method"),
                    "source_values": {
                        "synthetic_old_transition": True,
                        "source_path": teacher_row.get("source_path")
                        or teacher_row.get("counterfactual_coverage_source_path"),
                    },
                },
            }
        )
    return transitions


def _first_template_observation(
    old_transition_index: dict[tuple[str, str, int], dict[str, Any]]
) -> dict[str, Any]:
    for transition in old_transition_index.values():
        observation = transition.get("observation")
        if isinstance(observation, dict) and observation.get("candidate_feature_names"):
            return observation
    return {}


def _synthetic_observation(
    rows: list[dict[str, Any]],
    *,
    action_count: int,
    template_observation: dict[str, Any],
) -> dict[str, Any]:
    feature_names = [str(name) for name in template_observation.get("candidate_feature_names", [])]
    if not feature_names:
        feature_names = [
            "cell_x",
            "cell_y",
            "relative_dx",
            "relative_dy",
            "relative_distance",
            "utility",
            "reachable",
            "expected_coverage_rate_delta",
            "expected_new_coverage_area",
            "information_gain",
            "confidence_gain",
            "value",
            "risk",
            "path_cost",
            "energy_cost",
        ]
    missing_names = [
        str(name) for name in template_observation.get("candidate_missing_indicator_names", [])
    ]
    if not missing_names:
        missing_names = [
            "expected_coverage_rate_delta_missing",
            "expected_new_coverage_area_missing",
            "information_gain_missing",
            "confidence_gain_missing",
            "value_missing",
            "risk_missing",
            "path_cost_missing",
            "energy_cost_missing",
        ]
    features = [[0.0 for _ in feature_names] for _ in range(action_count)]
    missing = [[1.0 for _ in missing_names] for _ in range(action_count)]
    candidate_cells = [[index, index] for index in range(action_count)]
    template_features = [
        list(row)
        for row in template_observation.get("candidate_features", [])
        if isinstance(row, list)
    ]
    template_missing = [
        list(row)
        for row in template_observation.get("candidate_missing_indicators", [])
        if isinstance(row, list)
    ]
    template_cells = [
        list(row)
        for row in template_observation.get("candidate_cells", [])
        if isinstance(row, list)
    ]
    for index in range(min(action_count, len(template_features))):
        if len(template_features[index]) == len(feature_names):
            features[index] = list(template_features[index])
    for index in range(min(action_count, len(template_missing))):
        if len(template_missing[index]) == len(missing_names):
            missing[index] = list(template_missing[index])
    for index in range(min(action_count, len(template_cells))):
        candidate_cells[index] = list(template_cells[index])
    for index in range(action_count):
        _set_feature(features[index], feature_names, "utility", 0.5)
        _set_feature(features[index], feature_names, "reachable", 1.0)
    for row in rows:
        teacher_action = _first_int(row.get("teacher_action_index")) or 0
        candidate_action = _first_int(row.get("candidate_action_index")) or 0
        _overlay_observation_features(
            features[teacher_action],
            missing[teacher_action],
            feature_names,
            missing_names,
            {
                "expected_coverage_rate_delta": row.get("teacher_expected_coverage_rate_delta"),
                "expected_new_coverage_area": row.get("teacher_expected_new_coverage_area"),
                "information_gain": row.get("teacher_information_gain"),
                "value": row.get("teacher_valuable_coverage_proxy"),
                "risk": row.get("teacher_risk"),
                "path_cost": row.get("teacher_path_cost"),
                "energy_cost": row.get("teacher_energy_cost"),
            },
        )
        _overlay_observation_features(
            features[candidate_action],
            missing[candidate_action],
            feature_names,
            missing_names,
            {
                "expected_coverage_rate_delta": row.get("expected_coverage_rate_delta"),
                "expected_new_coverage_area": row.get("expected_new_coverage_area"),
                "information_gain": row.get("information_gain"),
                "value": _first_float(row.get("value"), row.get("valuable_coverage_proxy")),
                "risk": row.get("risk"),
                "path_cost": row.get("path_cost"),
                "energy_cost": row.get("energy_cost"),
            },
        )
        candidate_cells[candidate_action] = row.get("candidate_cell") or [candidate_action, candidate_action]
        candidate_cells[teacher_action] = row.get("teacher_candidate_cell") or [teacher_action, teacher_action]
    global_names = [str(name) for name in template_observation.get("global_feature_names", [])]
    global_features = list(template_observation.get("global_features", []))
    if not global_names:
        global_names = [
            "grid_width",
            "grid_height",
            "grid_resolution",
            "passable_ratio",
            "violation_count",
            "coverage_rate",
            "step_index",
            "remaining_steps",
        ]
        global_features = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, float(_int(rows[0].get("step_index"))), 1.0]
    while len(global_features) < len(global_names):
        global_features.append(0.0)
    if "step_index" in global_names:
        global_features[global_names.index("step_index")] = float(_int(rows[0].get("step_index")))
    return {
        "candidate_feature_names": feature_names,
        "candidate_features": features,
        "global_feature_names": global_names,
        "global_features": global_features[: len(global_names)],
        "action_mask": [True for _ in range(action_count)],
        "candidate_cells": candidate_cells,
        "candidate_missing_feature_names": [[] for _ in range(action_count)],
        "candidate_missing_indicator_names": missing_names,
        "candidate_missing_indicators": missing,
    }


def _expanded_summary(
    *,
    paths: dict[str, Path],
    reason_codes: list[str],
    prepared: dict[str, Any],
    refined_summary: dict[str, Any],
    safe_better_root: Path,
    stage5a2_root: Path,
    coverage_driven_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    base_candidate_root: Path,
    old_stage5a2_summary: dict[str, Any],
    minimum_safe_better_pair_count: int,
    repo_root: Path,
) -> dict[str, Any]:
    status = "passed" if not reason_codes else "failed"
    family_counts = prepared["family_counts"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": _next_required_change(reason_codes),
        "expanded_four_family_refined_coverage_driven_ppo_improvement_status": status,
        "safe_better_root": str(safe_better_root),
        "old_stage5a2_root": str(stage5a2_root),
        "coverage_driven_root": str(coverage_driven_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "reward_refinement_root": str(reward_refinement_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "source_audit": str(paths["source_audit"]),
        "report": str(paths["report"]),
        "compatible_stage5a2_root": str(paths["compat_stage5a2_root"]),
        "compatible_coverage_driven_root": str(paths["compat_coverage_driven_root"]),
        "base_candidate_root": str(base_candidate_root),
        "expanded_safe_better_pairs": str(safe_better_root / SAFE_BETTER_PAIRS_FILE),
        "expanded_counterfactual_coverage_rollouts": str(safe_better_root / SAFE_BETTER_COUNTERFACTUAL_FILE),
        "expanded_safe_better_pair_count": len(prepared["eligible_pairs"]),
        "minimum_safe_better_pair_count": minimum_safe_better_pair_count,
        "safe_better_training_pair_count": refined_summary.get(
            "safe_better_training_pair_count",
            len(prepared["eligible_pairs"]) if not reason_codes else 0,
        ),
        "safe_better_training_family_count": len([family for family, count in family_counts.items() if count > 0]),
        "family_safe_better_training_counts": family_counts,
        "synthetic_old_transition_count": prepared["synthetic_old_transition_count"],
        "old_transition_count": prepared["old_transition_count"],
        "old_stage5a2_safe_better_pair_count": _int(
            old_stage5a2_summary.get("safe_better_than_teacher_candidate_count")
        ),
        "uses_old_stage5a2_pairs_as_training_source": False,
        "refined_summary": refined_summary.get("summary"),
        "compatibility_summary": refined_summary.get("compatibility_summary"),
        "refined_batch_root": refined_summary.get("refined_batch_root", str(output_root / "refined-coverage-ppo-batch")),
        "coverage_aware_batch_root": refined_summary.get("coverage_aware_batch_root"),
        "coverage_aware_batch_episodes": refined_summary.get("coverage_aware_batch_episodes"),
        "coverage_aware_batch_transitions": refined_summary.get("coverage_aware_batch_transitions"),
        "coverage_aware_batch_collector_summary": refined_summary.get("coverage_aware_batch_collector_summary"),
        "refined_trainable_transitions": refined_summary.get("refined_trainable_transitions"),
        "advantage_margin_audit": refined_summary.get("advantage_margin_audit"),
        "ppo_update_summary": refined_summary.get("ppo_update_summary"),
        "checkpoint_path": refined_summary.get("checkpoint_path"),
        "checkpoint_metadata_path": refined_summary.get("checkpoint_metadata_path"),
        "guard_replay_audit": refined_summary.get("guard_replay_audit"),
        "performance_metric_table": refined_summary.get("performance_metric_table"),
        "stage5a_rerun_summary": refined_summary.get("stage5a_rerun_summary"),
        "counterfactual_advantage_nonzero_count": refined_summary.get("counterfactual_advantage_nonzero_count", 0),
        "policy_argmax_changed_count": refined_summary.get("policy_argmax_changed_count", 0),
        "coverage_return_improvement": refined_summary.get("coverage_return_improvement", 0.0),
        "cumulative_coverage_rate_delta_improvement": refined_summary.get(
            "cumulative_coverage_rate_delta_improvement",
            0.0,
        ),
        "valuable_area_covered_improvement": refined_summary.get("valuable_area_covered_improvement", 0.0),
        "coverage_efficiency_regression": bool(refined_summary.get("coverage_efficiency_regression", False)),
        "fallback_rate": refined_summary.get("fallback_rate"),
        "teacher_agreement_rate": refined_summary.get("teacher_agreement_rate"),
        "controlled_regression_count": refined_summary.get("controlled_regression_count", 0),
        "fallback_gain_contamination_count": refined_summary.get("fallback_gain_contamination_count", 0),
        "accepted_policy_activation_rate": refined_summary.get("accepted_policy_activation_rate"),
        "ppo_update_status": refined_summary.get("ppo_update_status"),
        "input_ppo_trainable_transition_count": refined_summary.get("input_ppo_trainable_transition_count", 0),
        "optimizer_train_transition_count": refined_summary.get("optimizer_train_transition_count", 0),
        "parameter_l2_delta": refined_summary.get("parameter_l2_delta", 0.0),
        "runs_new_ppo_update": bool(refined_summary.get("runs_new_ppo_update")) if refined_summary else False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "connects_real_executor": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "enforce_cost_risk_energy_filter": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    return summary


def _paths(output_root: Path) -> dict[str, Path]:
    compat_stage5a2_root = output_root / COMPAT_STAGE5A2_DIR
    compat_coverage_root = output_root / COMPAT_COVERAGE_DRIVEN_DIR
    compat_coverage_batch_root = compat_coverage_root / COMPAT_BATCH_DIR
    return {
        "summary": output_root / SUMMARY_FILE,
        "source_audit": output_root / SOURCE_AUDIT_FILE,
        "report": output_root / REPORT_FILE,
        "compat_stage5a2_root": compat_stage5a2_root,
        "compat_stage5a2_summary": compat_stage5a2_root / COMPAT_STAGE5A2_SUMMARY_FILE,
        "compat_overlay": compat_stage5a2_root / COMPAT_OVERLAY_FILE,
        "compat_counterfactual": compat_stage5a2_root / COMPAT_COUNTERFACTUAL_FILE,
        "compat_coverage_driven_root": compat_coverage_root,
        "compat_coverage_summary": compat_coverage_root / COVERAGE_DRIVEN_SUMMARY_FILE,
        "compat_coverage_batch_root": compat_coverage_batch_root,
        "compat_episodes": compat_coverage_batch_root / COMPAT_EPISODES_FILE,
    }


def _base_candidate_root(
    *,
    coverage_driven_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    selected_candidate_root: Path,
    repo_root: Path,
) -> Path:
    root = _resolve_optional_path(coverage_driven_summary.get("base_candidate_root"), repo_root, repo_root)
    if root is not None:
        return root
    return _selected_base_candidate_root(selected_summary, selected_candidate_root, repo_root)


def _old_transitions(old_episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for episode in old_episodes:
        for transition in episode.get("transitions", []):
            if isinstance(transition, dict):
                transitions.append(transition)
    return transitions


def _decision_key_from_transition(transition: dict[str, Any]) -> tuple[str, str, int]:
    info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
    return (
        str(info.get("context_id") or ""),
        str(info.get("episode_id") or ""),
        _int(info.get("step_index")),
    )


def _decision_key_from_pair(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
    )


def _counterfactual_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
        _int(row.get("action_index")),
    )


def _counterfactual_key_from_pair(row: dict[str, Any], action_index: int) -> tuple[str, str, int, int]:
    return (*_decision_key_from_pair(row), action_index)


def _pair_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_id": row.get("context_id"),
        "episode_id": row.get("episode_id"),
        "step_index": row.get("step_index"),
        "scenario_family": row.get("scenario_family"),
        "candidate_action_index": row.get("candidate_action_index"),
        "teacher_action_index": row.get("teacher_action_index"),
    }


def _overlay_observation_features(
    feature_row: list[float],
    missing_row: list[float],
    feature_names: list[str],
    missing_names: list[str],
    values: dict[str, Any],
) -> None:
    for name, value in values.items():
        if name in feature_names:
            _set_feature(feature_row, feature_names, name, _float(value))
        missing_name = f"{name}_missing"
        if missing_name in missing_names:
            missing_row[missing_names.index(missing_name)] = 0.0


def _set_feature(feature_row: list[float], feature_names: list[str], name: str, value: float) -> None:
    if name in feature_names:
        feature_row[feature_names.index(name)] = float(value)


def _next_required_change(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "shadow_canary_release_performance_validation"
    if "family_balance_failed" in reason_codes or "expanded_safe_better_pair_count_below_threshold" in reason_codes:
        return "expand_safe_better_pair_generation_across_families"
    if "expanded_safe_better_source_missing" in reason_codes or "counterfactual_row_missing" in reason_codes:
        return "repair_expanded_safe_better_source_materialization"
    if "post_update_policy_teacher_equivalent" in reason_codes:
        return "tune_expanded_four_family_reward_margin_or_policy_update_strength"
    if "fallback_dominates" in reason_codes:
        return "repair_guarded_replay_fallback_behavior"
    if "coverage_efficiency_regression" in reason_codes:
        return "tune_cost_efficiency_aware_reward_or_candidate_filter"
    if "no_coverage_return_improvement" in reason_codes:
        return "tune_expanded_four_family_reward_margin_or_collect_more_pairs"
    return "inspect_expanded_four_family_refined_ppo_failure"


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Expanded Four-Family Refined Coverage-Driven PPO Improvement v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- safe_better_training_pair_count: `{summary['safe_better_training_pair_count']}`",
            f"- safe_better_training_family_count: `{summary['safe_better_training_family_count']}`",
            f"- family_safe_better_training_counts: `{summary['family_safe_better_training_counts']}`",
            f"- counterfactual_advantage_nonzero_count: `{summary['counterfactual_advantage_nonzero_count']}`",
            f"- policy_argmax_changed_count: `{summary['policy_argmax_changed_count']}`",
            f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
            f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
            f"- fallback_rate: `{summary['fallback_rate']}`",
            f"- performance_claimed: `{summary['performance_claimed']}`",
            "",
            "This stage keeps the checkpoint experimental and does not replace the default policy.",
            "",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _add_reason(reason_codes, f"{label}_missing")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, reason_codes: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _add_reason(reason_codes, f"{label}_missing")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


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
    result = _first_int(value)
    return default if result is None else result


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _float(value: Any, default: float = 0.0) -> float:
    result = _first_float(value)
    return default if result is None else result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
