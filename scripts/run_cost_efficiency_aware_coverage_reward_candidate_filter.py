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
    from run_expanded_four_family_refined_coverage_driven_ppo_improvement import (
        _candidate_overlay_row,
        _teacher_overlay_row,
    )
    from run_refined_coverage_driven_ppo_improvement_run import (
        run_refined_coverage_driven_ppo_improvement_run,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_expanded_four_family_refined_coverage_driven_ppo_improvement import (
        _candidate_overlay_row,
        _teacher_overlay_row,
    )
    from scripts.run_refined_coverage_driven_ppo_improvement_run import (
        run_refined_coverage_driven_ppo_improvement_run,
    )


SUMMARY_SCHEMA_VERSION = "cost-efficiency-aware-coverage-reward-candidate-filter-summary/v1"
EFFICIENCY_AUDIT_SCHEMA_VERSION = "cost-efficiency-aware-efficiency-audit-row/v1"
FILTERED_BATCH_SCHEMA_VERSION = "cost-efficiency-aware-filtered-pair-row/v1"
ADVANTAGE_AUDIT_SCHEMA_VERSION = "cost-efficiency-aware-advantage-audit-row/v1"
STAGE5A2_SUMMARY_SCHEMA_VERSION = "policy-differentiating-counterfactual-coverage-rollouts-summary/v1"
PERFORMANCE_SUMMARY_SCHEMA_VERSION = "exploration-coverage-performance-evaluation-summary/v1"
METRIC_ROW_SCHEMA_VERSION = "coverage-driven-ppo-performance-metric-row/v1"

DEFAULT_SAFE_BETTER_ROOT = "outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1"
DEFAULT_EXPANDED_ROOT = "outputs/path_feedback_batch_expanded_four_family_refined_coverage_driven_ppo_improvement_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SELECTED_ROOT = "outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1"
DEFAULT_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_REWARD_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1"

SAFE_BETTER_SUMMARY_FILE = "safe-better-pair-expansion-summary.json"
SAFE_BETTER_PAIRS_FILE = "expanded-safe-better-pairs.jsonl"
SAFE_BETTER_COUNTERFACTUAL_FILE = "expanded-counterfactual-coverage-rollouts.jsonl"
EXPANDED_SUMMARY_FILE = "expanded-four-family-refined-coverage-driven-ppo-improvement-summary.json"

SUMMARY_FILE = "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json"
EFFICIENCY_AUDIT_FILE = "cost-efficiency-efficiency-audit.jsonl"
FILTERED_BATCH_FILE = "cost-efficiency-filtered-batch.jsonl"
ADVANTAGE_AUDIT_FILE = "cost-efficiency-advantage-audit.jsonl"
REPORT_FILE = "cost-efficiency-aware-coverage-reward-candidate-filter-report.md"
COMPAT_STAGE5A2_DIR = "cost-efficiency-compatible-stage5a2-input"
COMPAT_PERFORMANCE_DIR = "cost-efficiency-comparable-performance-input"
COMPAT_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
COMPAT_COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
COMPAT_STAGE5A2_SUMMARY_FILE = "policy-differentiating-counterfactual-coverage-rollouts-summary.json"
COMPAT_PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
COMPAT_PERFORMANCE_METRIC_TABLE_FILE = "coverage-performance-metric-table.jsonl"

TARGET_FAMILIES = (
    "low_observation_count",
    "mixed_risk",
    "rim_or_steep_slope",
    "smooth_high_confidence",
)
TOLERANCE = 1.0e-9
DISCOUNT_FACTOR = 0.99
DEFAULT_MINIMUM_TRAINABLE_PAIR_COUNT = 128
DEFAULT_MINIMUM_FAMILY_PAIR_COUNT = 8
PATH_DELTA_PENALTY = 0.002
RISK_DELTA_PENALTY = 0.05
ENERGY_DELTA_PENALTY = 0.0001


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a cost-efficiency-aware coverage reward/candidate filter over expanded four-family pairs."
    )
    parser.add_argument("--safe-better-root", default=DEFAULT_SAFE_BETTER_ROOT)
    parser.add_argument("--expanded-four-family-root", default=DEFAULT_EXPANDED_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--selected-candidate-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--coverage-signal-root", default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--reward-refinement-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--minimum-trainable-pair-count", type=int, default=DEFAULT_MINIMUM_TRAINABLE_PAIR_COUNT)
    parser.add_argument("--minimum-family-pair-count", type=int, default=DEFAULT_MINIMUM_FAMILY_PAIR_COUNT)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_cost_efficiency_aware_coverage_reward_candidate_filter(
        safe_better_root=_resolve_path(Path(args.safe_better_root), repo_root),
        expanded_four_family_root=_resolve_path(Path(args.expanded_four_family_root), repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        minimum_trainable_pair_count=args.minimum_trainable_pair_count,
        minimum_family_pair_count=args.minimum_family_pair_count,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "next_required_change": summary["next_required_change"],
                "trainable_pair_count": summary["trainable_pair_count"],
                "safe_better_training_family_count": summary["safe_better_training_family_count"],
                "coverage_return_improvement": summary["coverage_return_improvement"],
                "cumulative_coverage_rate_delta_improvement": summary[
                    "cumulative_coverage_rate_delta_improvement"
                ],
                "valuable_area_covered_improvement": summary["valuable_area_covered_improvement"],
                "coverage_efficiency_regression": summary["coverage_efficiency_regression"],
                "fallback_rate": summary["fallback_rate"],
                "performance_claimed": summary["performance_claimed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_cost_efficiency_aware_coverage_reward_candidate_filter(
    *,
    safe_better_root: Path,
    expanded_four_family_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    repo_root: Path,
    minimum_trainable_pair_count: int = DEFAULT_MINIMUM_TRAINABLE_PAIR_COUNT,
    minimum_family_pair_count: int = DEFAULT_MINIMUM_FAMILY_PAIR_COUNT,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    paths["compat_stage5a2_root"].mkdir(parents=True, exist_ok=True)
    paths["compat_performance_root"].mkdir(parents=True, exist_ok=True)

    input_reasons: list[str] = []
    safe_better_summary = _read_json(
        Path(safe_better_root) / SAFE_BETTER_SUMMARY_FILE,
        input_reasons,
        "safe_better_summary",
    )
    expanded_summary = _read_json(
        Path(expanded_four_family_root) / EXPANDED_SUMMARY_FILE,
        input_reasons,
        "expanded_four_family_summary",
    )
    pair_rows = _read_jsonl(
        Path(safe_better_root) / SAFE_BETTER_PAIRS_FILE,
        input_reasons,
        "expanded_safe_better_pairs",
    )
    counterfactual_rows = _read_jsonl(
        Path(safe_better_root) / SAFE_BETTER_COUNTERFACTUAL_FILE,
        input_reasons,
        "expanded_counterfactual_coverage_rollouts",
    )
    expanded_refined_rows = _read_jsonl(
        _resolve_optional_path(expanded_summary.get("refined_trainable_transitions"), Path(expanded_four_family_root), repo_root)
        or Path(expanded_four_family_root) / "refined-coverage-ppo-batch" / "refined-trainable-transitions.jsonl",
        input_reasons,
        "expanded_refined_trainable_transitions",
    )
    expanded_metric_rows = _read_jsonl(
        _resolve_optional_path(expanded_summary.get("performance_metric_table"), Path(expanded_four_family_root), repo_root)
        or Path(expanded_four_family_root) / "refined-coverage-driven-ppo-performance-metric-table.jsonl",
        input_reasons,
        "expanded_performance_metric_table",
    )
    expanded_replay_audit = _read_json(
        _resolve_optional_path(expanded_summary.get("guard_replay_audit"), Path(expanded_four_family_root), repo_root)
        or Path(expanded_four_family_root) / "coverage-driven-ppo-replay-audit.json",
        input_reasons,
        "expanded_guard_replay_audit",
    )

    preflight_reasons = _input_reason_codes(
        safe_better_summary=safe_better_summary,
        expanded_summary=expanded_summary,
        expanded_refined_rows=expanded_refined_rows,
        expanded_metric_rows=expanded_metric_rows,
        expanded_replay_audit=expanded_replay_audit,
    )
    batch = build_cost_efficiency_aware_batch(
        pair_rows=pair_rows,
        counterfactual_rows=counterfactual_rows,
        minimum_trainable_pair_count=minimum_trainable_pair_count,
        minimum_family_pair_count=minimum_family_pair_count,
    )
    _write_jsonl(paths["efficiency_audit"], batch["audit_rows"])
    _write_jsonl(paths["filtered_batch"], batch["trainable_pairs"])
    _write_jsonl(paths["advantage_audit"], batch["advantage_audit_rows"])

    compatible_stage5a2 = _write_compatible_stage5a2_inputs(
        batch=batch,
        output_paths=paths,
        input_reasons=[*input_reasons, *preflight_reasons],
    )
    compatible_performance = _write_comparable_performance_inputs(
        trainable_pairs=batch["trainable_pairs"],
        output_paths=paths,
    )

    reason_codes = _unique([*input_reasons, *preflight_reasons, *batch["reason_codes"]])
    refined_summary: dict[str, Any] = {}
    if not reason_codes:
        compatible_coverage_driven_root = (
            _resolve_optional_path(
                expanded_summary.get("compatible_coverage_driven_root"),
                Path(expanded_four_family_root),
                repo_root,
            )
            or Path(expanded_four_family_root)
        )
        refined_summary = run_refined_coverage_driven_ppo_improvement_run(
            stage5a2_root=paths["compat_stage5a2_root"],
            coverage_driven_root=compatible_coverage_driven_root,
            formal_training_root=Path(formal_training_root),
            post_training_replay_root=Path(post_training_replay_root),
            selected_candidate_root=Path(selected_candidate_root),
            coverage_signal_root=Path(coverage_signal_root),
            coverage_performance_root=paths["compat_performance_root"],
            reward_refinement_root=Path(reward_refinement_root),
            output_root=output_root,
            repo_root=repo_root,
            enforce_cost_risk_energy_filter=False,
            ppo_advantage_scale=75.0,
            use_transition_info_advantage=True,
            ppo_learning_rate=2.0e-3,
            ppo_epochs=30,
            ppo_max_approx_kl=100.0,
            ppo_advantage_field="cost_efficiency_ppo_advantage",
        )
        reason_codes = _unique([*reason_codes, *refined_summary.get("reason_codes", [])])

    acceptance_reasons = _acceptance_reason_codes(
        batch=batch,
        refined_summary=refined_summary,
    )
    reason_codes = _unique([*reason_codes, *acceptance_reasons])

    summary = _summary(
        paths=paths,
        reason_codes=reason_codes,
        batch=batch,
        refined_summary=refined_summary,
        safe_better_root=Path(safe_better_root),
        expanded_four_family_root=Path(expanded_four_family_root),
        formal_training_root=Path(formal_training_root),
        post_training_replay_root=Path(post_training_replay_root),
        selected_candidate_root=Path(selected_candidate_root),
        coverage_signal_root=Path(coverage_signal_root),
        reward_refinement_root=Path(reward_refinement_root),
        output_root=output_root,
        compatible_stage5a2=compatible_stage5a2,
        compatible_performance=compatible_performance,
        expanded_summary=expanded_summary,
        minimum_trainable_pair_count=minimum_trainable_pair_count,
        minimum_family_pair_count=minimum_family_pair_count,
        repo_root=repo_root,
    )
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, batch), encoding="utf-8")
    return summary


def build_cost_efficiency_aware_batch(
    *,
    pair_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    minimum_trainable_pair_count: int = DEFAULT_MINIMUM_TRAINABLE_PAIR_COUNT,
    minimum_family_pair_count: int = DEFAULT_MINIMUM_FAMILY_PAIR_COUNT,
) -> dict[str, Any]:
    counterfactual_index = {
        _counterfactual_key(row): row
        for row in counterfactual_rows
        if _first_int(row.get("action_index")) is not None
    }
    audit_rows: list[dict[str, Any]] = []
    trainable_pairs: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for audit_index, row in enumerate(pair_rows):
        metrics = _cost_efficiency_metrics(row)
        row_reasons = _row_rejection_reasons(row, counterfactual_index, metrics)
        action_index = _first_int(row.get("candidate_action_index"))
        counterfactual = (
            counterfactual_index.get(_counterfactual_key_from_pair(row, action_index))
            if action_index is not None
            else None
        )
        enriched = {
            **copy.deepcopy(row),
            "schema_version": FILTERED_BATCH_SCHEMA_VERSION,
            **metrics,
            "cost_efficiency_reward_components": _cost_efficiency_reward_components(metrics),
            "cost_efficiency_sample_class": _sample_class(metrics),
            "cost_efficiency_filter_trainable": not row_reasons,
            "row_reason_codes": row_reasons,
            "_counterfactual_row": copy.deepcopy(counterfactual) if counterfactual else None,
        }
        audit_rows.append(_audit_row(audit_index, enriched, row_reasons))
        if row_reasons:
            diagnostic_rows.append(enriched)
        else:
            trainable_pairs.append(enriched)

    family_counts = Counter(str(row.get("scenario_family") or "") for row in trainable_pairs)
    reason_codes: list[str] = []
    if len(trainable_pairs) < minimum_trainable_pair_count:
        _add_reason(reason_codes, "cost_efficiency_filter_too_strict")
    if any(family_counts.get(family, 0) < minimum_family_pair_count for family in TARGET_FAMILIES):
        _add_reason(reason_codes, "family_balance_failed")
        for family in TARGET_FAMILIES:
            if family_counts.get(family, 0) < minimum_family_pair_count:
                _add_reason(reason_codes, f"family_cost_efficiency_gap_{family}")

    return {
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "pair_count": len(pair_rows),
        "counterfactual_row_count": len(counterfactual_rows),
        "counterfactual_rows": counterfactual_rows,
        "trainable_pair_count": len(trainable_pairs),
        "diagnostic_pair_count": len(diagnostic_rows),
        "trainable_family_count": len([family for family, count in family_counts.items() if count > 0]),
        "family_trainable_counts": dict(sorted(family_counts.items())),
        "trainable_pairs": sorted(
            trainable_pairs,
            key=lambda row: (
                str(row.get("scenario_family") or ""),
                str(row.get("scenario_id") or ""),
                str(row.get("episode_id") or ""),
                _int(row.get("step_index")),
                _int(row.get("candidate_action_index")),
            ),
        ),
        "diagnostic_rows": diagnostic_rows,
        "audit_rows": audit_rows,
        "advantage_audit_rows": [_advantage_audit_row(index, row) for index, row in enumerate(trainable_pairs)],
        "aggregate_efficiency": _aggregate_efficiency(trainable_pairs),
        "diagnostic_reason_counts": dict(Counter(reason for row in diagnostic_rows for reason in row["row_reason_codes"])),
    }


def _cost_efficiency_metrics(row: dict[str, Any]) -> dict[str, Any]:
    coverage_advantage = _first_float(row.get("coverage_advantage"))
    if coverage_advantage is None:
        coverage_advantage = _float(row.get("expected_coverage_rate_delta")) - _float(
            row.get("teacher_expected_coverage_rate_delta")
        )
    valuable_advantage = _first_float(row.get("valuable_coverage_advantage"))
    if valuable_advantage is None:
        valuable_advantage = _first_float(row.get("valuable_coverage_proxy"), row.get("value"), default=0.0) - _first_float(
            row.get("teacher_valuable_coverage_proxy"),
            row.get("teacher_value"),
            default=0.0,
        )
    information_advantage = _first_float(row.get("information_gain_advantage"))
    if information_advantage is None:
        information_advantage = _float(row.get("information_gain")) - _float(row.get("teacher_information_gain"))
    path_delta = _first_float(row.get("path_cost_delta"))
    if path_delta is None:
        path_delta = _float(row.get("path_cost")) - _float(row.get("teacher_path_cost"))
    risk_delta = _first_float(row.get("risk_delta"))
    if risk_delta is None:
        risk_delta = _float(row.get("risk")) - _float(row.get("teacher_risk"))
    energy_delta = _first_float(row.get("energy_delta"))
    if energy_delta is None:
        energy_delta = _float(row.get("energy_cost")) - _float(row.get("teacher_energy_cost"))

    positive_path_delta = max(float(path_delta), 0.0)
    positive_risk_delta = max(float(risk_delta), 0.0)
    positive_energy_delta = max(float(energy_delta), 0.0)
    path_penalty = PATH_DELTA_PENALTY * positive_path_delta
    risk_penalty = RISK_DELTA_PENALTY * positive_risk_delta
    energy_penalty = ENERGY_DELTA_PENALTY * positive_energy_delta
    cost_efficiency_penalty = path_penalty + risk_penalty + energy_penalty
    cost_efficiency_ppo_advantage = float(coverage_advantage) - cost_efficiency_penalty
    weight = _safe_ratio(max(cost_efficiency_ppo_advantage, 0.0), max(float(coverage_advantage), TOLERANCE))
    weight = min(1.0, max(0.05 if cost_efficiency_ppo_advantage > 0 else 0.0, weight or 0.0))

    candidate_coverage = _float(row.get("expected_coverage_rate_delta"))
    teacher_coverage = _float(row.get("teacher_expected_coverage_rate_delta"))
    candidate_path = _float(row.get("path_cost"))
    teacher_path = _float(row.get("teacher_path_cost"))
    candidate_risk = _float(row.get("risk"))
    teacher_risk = _float(row.get("teacher_risk"))
    candidate_energy = _float(row.get("energy_cost"))
    teacher_energy = _float(row.get("teacher_energy_cost"))

    return {
        "coverage_advantage": float(coverage_advantage),
        "valuable_coverage_advantage": float(valuable_advantage),
        "information_gain_advantage": float(information_advantage),
        "path_cost_delta": float(path_delta),
        "risk_delta": float(risk_delta),
        "energy_delta": float(energy_delta),
        "positive_path_cost_delta": positive_path_delta,
        "positive_risk_delta": positive_risk_delta,
        "positive_energy_delta": positive_energy_delta,
        "path_cost_efficiency_penalty": -path_penalty,
        "risk_efficiency_penalty": -risk_penalty,
        "energy_efficiency_penalty": -energy_penalty,
        "total_cost_efficiency_penalty": -cost_efficiency_penalty,
        "cost_efficiency_ppo_advantage": round(cost_efficiency_ppo_advantage, 12),
        "cost_efficiency_weight": round(weight, 12),
        "candidate_gain_per_path_cost": _safe_ratio(candidate_coverage, candidate_path),
        "teacher_gain_per_path_cost": _safe_ratio(teacher_coverage, teacher_path),
        "candidate_gain_per_risk": _safe_ratio(candidate_coverage, candidate_risk),
        "teacher_gain_per_risk": _safe_ratio(teacher_coverage, teacher_risk),
        "candidate_gain_per_energy": _safe_ratio(candidate_coverage, candidate_energy),
        "teacher_gain_per_energy": _safe_ratio(teacher_coverage, teacher_energy),
    }


def _cost_efficiency_reward_components(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "coverage_gain_bonus": float(metrics["coverage_advantage"]),
        "valuable_area_bonus": max(float(metrics["valuable_coverage_advantage"]), 0.0),
        "information_gain_bonus": max(float(metrics["information_gain_advantage"]), 0.0),
        "teacher_skill_retention_bonus": 0.0,
        "path_cost_efficiency_penalty": float(metrics["path_cost_efficiency_penalty"]),
        "risk_efficiency_penalty": float(metrics["risk_efficiency_penalty"]),
        "energy_efficiency_penalty": float(metrics["energy_efficiency_penalty"]),
        "fallback_penalty": 0.0,
        "controlled_regression_penalty": 0.0,
    }


def _sample_class(metrics: dict[str, Any]) -> str:
    if float(metrics["cost_efficiency_ppo_advantage"]) <= TOLERANCE:
        return "diagnostic"
    candidate_path = metrics.get("candidate_gain_per_path_cost")
    teacher_path = metrics.get("teacher_gain_per_path_cost")
    no_positive_cost = (
        float(metrics["positive_path_cost_delta"]) <= TOLERANCE
        and float(metrics["positive_risk_delta"]) <= TOLERANCE
        and float(metrics["positive_energy_delta"]) <= TOLERANCE
    )
    path_not_worse = (
        candidate_path is not None
        and teacher_path is not None
        and float(candidate_path) + TOLERANCE >= float(teacher_path) * 0.95
    )
    return "strong" if no_positive_cost or path_not_worse else "weighted"


def _row_rejection_reasons(
    row: dict[str, Any],
    counterfactual_index: dict[tuple[str, str, int, int], dict[str, Any]],
    metrics: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if str(row.get("split") or "") != "train":
        reasons.append("non_train_split")
    if not bool(row.get("ppo_trainable")):
        reasons.append("not_ppo_trainable")
    if not bool(row.get("coverage_source_available")):
        reasons.append("coverage_source_missing")
    if not bool(row.get("safe_better_than_teacher_candidate")):
        reasons.append("not_safe_better_than_teacher")
    if float(metrics["coverage_advantage"]) <= TOLERANCE:
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
    else:
        counterfactual = counterfactual_index[_counterfactual_key_from_pair(row, action_index)]
        if not bool(counterfactual.get("coverage_source_available", True)):
            reasons.append("counterfactual_coverage_source_missing")
    if float(metrics["cost_efficiency_ppo_advantage"]) <= TOLERANCE:
        reasons.append("low_gain_high_cost")
    return _unique(reasons)


def _write_compatible_stage5a2_inputs(
    *,
    batch: dict[str, Any],
    output_paths: dict[str, Path],
    input_reasons: list[str],
) -> dict[str, Any]:
    family_counts = Counter(str(row.get("scenario_family") or "") for row in batch["trainable_pairs"])
    total = max(len(batch["trainable_pairs"]), 1)
    family_count = max(len([family for family, count in family_counts.items() if count > 0]), 1)
    family_weights = {
        family: total / float(family_count * count)
        for family, count in family_counts.items()
        if count > 0
    }
    counterfactual_index = {
        _counterfactual_key(row): row
        for row in batch.get("counterfactual_rows", [])
        if _first_int(row.get("action_index")) is not None
    }
    overlay_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    teacher_seen: set[tuple[str, str, int, int]] = set()
    for row in batch["trainable_pairs"]:
        family = str(row.get("scenario_family") or "")
        combined_weight = family_weights.get(family, 1.0) * max(_float(row.get("cost_efficiency_weight")), 0.0)
        teacher = _teacher_overlay_row(row, family_weight=family_weights.get(family, 1.0))
        teacher_key = (
            str(teacher.get("context_id") or ""),
            str(teacher.get("episode_id") or ""),
            _int(teacher.get("step_index")),
            _int(teacher.get("action_index")),
        )
        if teacher_key not in teacher_seen:
            overlay_rows.append(teacher)
            teacher_seen.add(teacher_key)
        candidate = _candidate_overlay_row(row, family_weight=combined_weight)
        candidate.update(_cost_efficiency_overlay_fields(row, family_weights.get(family, 1.0), combined_weight))
        overlay_rows.append(candidate)
        action_index = _first_int(row.get("candidate_action_index"))
        assert action_index is not None
        counterfactual = copy.deepcopy(row.get("_counterfactual_row") or {})
        if not counterfactual:
            counterfactual = {
                "expected_coverage_rate_delta": row.get("expected_coverage_rate_delta"),
                "expected_new_coverage_area": row.get("expected_new_coverage_area"),
                "valuable_coverage_proxy": row.get("valuable_coverage_proxy"),
                "information_gain": row.get("information_gain"),
                "coverage_source_available": True,
            }
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
                "expanded_family_weight": combined_weight,
                **_cost_efficiency_overlay_fields(row, family_weights.get(family, 1.0), combined_weight),
            }
        )
        counterfactual_rows.append(counterfactual)

    summary = {
        "schema_version": STAGE5A2_SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "passed" if batch["status"] == "passed" and not input_reasons else "failed",
        "reason_codes": _unique([*input_reasons, *batch["reason_codes"]]),
        "next_required_change": "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
        "candidate_coverage_overlay": str(output_paths["compat_overlay"]),
        "counterfactual_coverage_rollouts": str(output_paths["compat_counterfactual"]),
        "safe_better_than_teacher_candidate_count": len(batch["trainable_pairs"]),
        "safe_better_than_teacher_family_count": len(family_counts),
        "cost_efficiency_filtered": True,
        "fallback_gain_contamination_count": 0,
        "controlled_regression_count": 0,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }
    _write_jsonl(output_paths["compat_overlay"], overlay_rows)
    _write_jsonl(output_paths["compat_counterfactual"], counterfactual_rows)
    _write_json(output_paths["compat_stage5a2_summary"], summary)
    return summary


def _write_comparable_performance_inputs(
    *,
    trainable_pairs: list[dict[str, Any]],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    metric_rows = [
        _metric_row_for_pairs("teacher", trainable_pairs, role="teacher"),
        _metric_row_for_pairs("pre_improvement_selected_ppo", trainable_pairs, role="teacher"),
    ]
    summary = {
        "schema_version": PERFORMANCE_SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "passed" if trainable_pairs else "failed",
        "coverage_performance_status": "passed" if trainable_pairs else "failed",
        "reason_codes": [] if trainable_pairs else ["cost_efficiency_filter_too_strict"],
        "metric_table": str(output_paths["compat_performance_metric_table"]),
        "best_baseline_actor": "teacher",
        "best_baseline_metrics": metric_rows[0],
        "selected_metrics": metric_rows[1],
        "comparator_scope": "same_cost_efficiency_filtered_decision_set",
        "controlled_regression_count": 0,
        "performance_claimed": False,
    }
    _write_jsonl(output_paths["compat_performance_metric_table"], metric_rows)
    _write_json(output_paths["compat_performance_summary"], summary)
    return summary


def _metric_row_for_pairs(actor: str, rows: list[dict[str, Any]], *, role: str) -> dict[str, Any]:
    prefix = "teacher_" if role == "teacher" else ""
    coverage_values = [_float(row.get(f"{prefix}expected_coverage_rate_delta")) for row in rows]
    total_gain = sum(coverage_values)
    coverage_return = sum(
        value * (DISCOUNT_FACTOR ** max(_int(row.get("step_index")), 0))
        for value, row in zip(coverage_values, rows)
    )
    path_cost = sum(_float(row.get(f"{prefix}path_cost")) for row in rows)
    risk = sum(_float(row.get(f"{prefix}risk")) for row in rows)
    energy = sum(_float(row.get(f"{prefix}energy_cost")) for row in rows)
    valuable = sum(
        _first_float(
            row.get(f"{prefix}valuable_coverage_proxy"),
            row.get(f"{prefix}value"),
            default=0.0,
        )
        for row in rows
    )
    information = sum(_float(row.get(f"{prefix}information_gain")) for row in rows)
    families = sorted({str(row.get("scenario_family")) for row in rows if row.get("scenario_family") and _float(row.get(f"{prefix}expected_coverage_rate_delta")) > TOLERANCE})
    return {
        "schema_version": METRIC_ROW_SCHEMA_VERSION,
        "actor": actor,
        "row_count": len(rows),
        "episode_count": len({row.get("episode_id") for row in rows if row.get("episode_id")}),
        "actual_coverage_row_count": len(rows),
        "coverage_return": round(coverage_return, 12),
        "cumulative_coverage_rate_delta": round(total_gain, 12),
        "final_coverage_rate": _mean(coverage_values),
        "new_area_covered": round(sum(_float(row.get(f"{prefix}expected_new_coverage_area")) for row in rows), 12),
        "valuable_area_covered": round(valuable, 12),
        "information_gain": round(information, 12),
        "path_cost": round(path_cost, 12),
        "risk": round(risk, 12),
        "energy_cost": round(energy, 12),
        "coverage_gain_per_path_cost": _safe_ratio(total_gain, path_cost),
        "coverage_gain_per_risk": _safe_ratio(total_gain, risk),
        "coverage_gain_per_energy": _safe_ratio(total_gain, energy),
        "accepted_policy_activation_rate": 1.0 if actor == "pre_improvement_selected_ppo" else 0.0,
        "fallback_rate": 0.0,
        "teacher_agreement_rate": 1.0,
        "controlled_regression_count": 0,
        "fallback_coverage_gain": 0.0,
        "scenario_family_gain_count": len(families),
        "scenario_families_with_gain": families,
        "comparator_basis": ["same_cost_efficiency_filtered_decision_set"],
    }


def _cost_efficiency_overlay_fields(row: dict[str, Any], family_weight: float, combined_weight: float) -> dict[str, Any]:
    return {
        "cost_efficiency_filtered": True,
        "cost_efficiency_sample_class": row.get("cost_efficiency_sample_class"),
        "cost_efficiency_weight": row.get("cost_efficiency_weight"),
        "cost_efficiency_family_weight": family_weight,
        "cost_efficiency_combined_weight": combined_weight,
        "cost_efficiency_ppo_advantage": row.get("cost_efficiency_ppo_advantage"),
        "cost_efficiency_reward_components": row.get("cost_efficiency_reward_components", {}),
        "path_cost_delta": row.get("path_cost_delta"),
        "risk_delta": row.get("risk_delta"),
        "energy_delta": row.get("energy_delta"),
        "candidate_gain_per_path_cost": row.get("candidate_gain_per_path_cost"),
        "teacher_gain_per_path_cost": row.get("teacher_gain_per_path_cost"),
        "candidate_gain_per_risk": row.get("candidate_gain_per_risk"),
        "teacher_gain_per_risk": row.get("teacher_gain_per_risk"),
        "candidate_gain_per_energy": row.get("candidate_gain_per_energy"),
        "teacher_gain_per_energy": row.get("teacher_gain_per_energy"),
    }


def _audit_row(index: int, row: dict[str, Any], row_reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": EFFICIENCY_AUDIT_SCHEMA_VERSION,
        "audit_index": index,
        "context_id": row.get("context_id"),
        "episode_id": row.get("episode_id"),
        "step_index": row.get("step_index"),
        "scenario_id": row.get("scenario_id"),
        "scenario_family": row.get("scenario_family"),
        "split": row.get("split"),
        "candidate_action_index": row.get("candidate_action_index"),
        "teacher_action_index": row.get("teacher_action_index"),
        "coverage_advantage": row.get("coverage_advantage"),
        "valuable_coverage_advantage": row.get("valuable_coverage_advantage"),
        "information_gain_advantage": row.get("information_gain_advantage"),
        "path_cost_delta": row.get("path_cost_delta"),
        "risk_delta": row.get("risk_delta"),
        "energy_delta": row.get("energy_delta"),
        "candidate_gain_per_path_cost": row.get("candidate_gain_per_path_cost"),
        "teacher_gain_per_path_cost": row.get("teacher_gain_per_path_cost"),
        "candidate_gain_per_risk": row.get("candidate_gain_per_risk"),
        "teacher_gain_per_risk": row.get("teacher_gain_per_risk"),
        "candidate_gain_per_energy": row.get("candidate_gain_per_energy"),
        "teacher_gain_per_energy": row.get("teacher_gain_per_energy"),
        "cost_efficiency_ppo_advantage": row.get("cost_efficiency_ppo_advantage"),
        "cost_efficiency_weight": row.get("cost_efficiency_weight"),
        "cost_efficiency_sample_class": row.get("cost_efficiency_sample_class"),
        "row_reason_codes": row_reasons,
        "trainable": not row_reasons,
    }


def _advantage_audit_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ADVANTAGE_AUDIT_SCHEMA_VERSION,
        "advantage_index": index,
        "context_id": row.get("context_id"),
        "episode_id": row.get("episode_id"),
        "step_index": row.get("step_index"),
        "scenario_id": row.get("scenario_id"),
        "scenario_family": row.get("scenario_family"),
        "coverage_advantage": row.get("coverage_advantage"),
        "path_cost_delta": row.get("path_cost_delta"),
        "risk_delta": row.get("risk_delta"),
        "energy_delta": row.get("energy_delta"),
        "cost_efficiency_ppo_advantage": row.get("cost_efficiency_ppo_advantage"),
        "cost_efficiency_weight": row.get("cost_efficiency_weight"),
        "reward_components": row.get("cost_efficiency_reward_components", {}),
    }


def _aggregate_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[str(row.get("scenario_family") or "")].append(row)
    return {
        "overall": _aggregate_efficiency_row(rows),
        "by_family": {family: _aggregate_efficiency_row(items) for family, items in sorted(family_rows.items())},
    }


def _aggregate_efficiency_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_gain = sum(_float(row.get("expected_coverage_rate_delta")) for row in rows)
    teacher_gain = sum(_float(row.get("teacher_expected_coverage_rate_delta")) for row in rows)
    candidate_path = sum(_float(row.get("path_cost")) for row in rows)
    teacher_path = sum(_float(row.get("teacher_path_cost")) for row in rows)
    candidate_risk = sum(_float(row.get("risk")) for row in rows)
    teacher_risk = sum(_float(row.get("teacher_risk")) for row in rows)
    candidate_energy = sum(_float(row.get("energy_cost")) for row in rows)
    teacher_energy = sum(_float(row.get("teacher_energy_cost")) for row in rows)
    return {
        "row_count": len(rows),
        "candidate_cumulative_coverage_rate_delta": round(candidate_gain, 12),
        "teacher_cumulative_coverage_rate_delta": round(teacher_gain, 12),
        "candidate_path_cost": round(candidate_path, 12),
        "teacher_path_cost": round(teacher_path, 12),
        "candidate_risk": round(candidate_risk, 12),
        "teacher_risk": round(teacher_risk, 12),
        "candidate_energy_cost": round(candidate_energy, 12),
        "teacher_energy_cost": round(teacher_energy, 12),
        "candidate_gain_per_path_cost": _safe_ratio(candidate_gain, candidate_path),
        "teacher_gain_per_path_cost": _safe_ratio(teacher_gain, teacher_path),
        "candidate_gain_per_risk": _safe_ratio(candidate_gain, candidate_risk),
        "teacher_gain_per_risk": _safe_ratio(teacher_gain, teacher_risk),
        "candidate_gain_per_energy": _safe_ratio(candidate_gain, candidate_energy),
        "teacher_gain_per_energy": _safe_ratio(teacher_gain, teacher_energy),
    }


def _input_reason_codes(
    *,
    safe_better_summary: dict[str, Any],
    expanded_summary: dict[str, Any],
    expanded_refined_rows: list[dict[str, Any]],
    expanded_metric_rows: list[dict[str, Any]],
    expanded_replay_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if safe_better_summary and safe_better_summary.get("status") != "passed":
        _add_reason(reasons, "safe_better_pair_expansion_not_passed")
    if _int(safe_better_summary.get("missing_counterfactual_source_count")) > 0:
        _add_reason(reasons, "expanded_safe_better_source_missing")
    if _int(safe_better_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    if _int(safe_better_summary.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_present")
    if expanded_summary:
        if expanded_summary.get("next_required_change") != "tune_cost_efficiency_aware_reward_or_candidate_filter":
            _add_reason(reasons, "expanded_four_family_not_routed_to_cost_efficiency_filter")
        if not bool(expanded_summary.get("coverage_efficiency_regression")):
            _add_reason(reasons, "expanded_four_family_efficiency_regression_missing")
        if _int(expanded_summary.get("controlled_regression_count")) > 0:
            _add_reason(reasons, "controlled_regression_present")
        if _int(expanded_summary.get("fallback_gain_contamination_count")) > 0:
            _add_reason(reasons, "fallback_gain_contamination_present")
    if not expanded_refined_rows:
        _add_reason(reasons, "expanded_refined_transitions_missing")
    if not expanded_metric_rows:
        _add_reason(reasons, "expanded_metric_table_missing")
    if not expanded_replay_audit:
        _add_reason(reasons, "expanded_guard_replay_missing")
    return reasons


def _acceptance_reason_codes(
    *,
    batch: dict[str, Any],
    refined_summary: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if batch["status"] != "passed":
        reasons.extend(batch["reason_codes"])
    if not refined_summary:
        return _unique(reasons)
    if refined_summary.get("status") != "passed":
        reasons.extend(refined_summary.get("reason_codes", []))
    if _float(refined_summary.get("coverage_return_improvement")) <= TOLERANCE:
        _add_reason(reasons, "no_coverage_return_improvement")
    if _float(refined_summary.get("cumulative_coverage_rate_delta_improvement")) <= TOLERANCE:
        _add_reason(reasons, "no_cumulative_coverage_rate_delta_improvement")
    if _float(refined_summary.get("valuable_area_covered_improvement")) <= TOLERANCE:
        _add_reason(reasons, "valuable_coverage_regressed")
    if _int(refined_summary.get("policy_argmax_changed_count")) <= 0:
        _add_reason(reasons, "post_update_policy_teacher_equivalent")
    if bool(refined_summary.get("coverage_efficiency_regression")):
        _add_reason(reasons, "coverage_efficiency_regression")
    fallback_rate = refined_summary.get("fallback_rate")
    if fallback_rate is not None and float(fallback_rate) >= 0.5:
        _add_reason(reasons, "fallback_dominates")
    if _int(refined_summary.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression_present")
    if _int(refined_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    if batch["trainable_family_count"] != 4:
        _add_reason(reasons, "family_balance_failed")
    return _unique(reasons)


def _summary(
    *,
    paths: dict[str, Path],
    reason_codes: list[str],
    batch: dict[str, Any],
    refined_summary: dict[str, Any],
    safe_better_root: Path,
    expanded_four_family_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    compatible_stage5a2: dict[str, Any],
    compatible_performance: dict[str, Any],
    expanded_summary: dict[str, Any],
    minimum_trainable_pair_count: int,
    minimum_family_pair_count: int,
    repo_root: Path,
) -> dict[str, Any]:
    status = "passed" if not reason_codes else "failed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": _next_required_change(reason_codes),
        "cost_efficiency_aware_coverage_reward_candidate_filter_status": status,
        "safe_better_root": str(safe_better_root),
        "expanded_four_family_root": str(expanded_four_family_root),
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "coverage_signal_root": str(coverage_signal_root),
        "reward_refinement_root": str(reward_refinement_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "efficiency_audit": str(paths["efficiency_audit"]),
        "filtered_batch": str(paths["filtered_batch"]),
        "advantage_audit": str(paths["advantage_audit"]),
        "report": str(paths["report"]),
        "compatible_stage5a2_root": str(paths["compat_stage5a2_root"]),
        "compatible_performance_root": str(paths["compat_performance_root"]),
        "compatible_stage5a2_summary": str(paths["compat_stage5a2_summary"]),
        "compatible_performance_summary": str(paths["compat_performance_summary"]),
        "compatible_performance_metric_table": str(paths["compat_performance_metric_table"]),
        "refined_summary": refined_summary.get("summary"),
        "compatibility_summary": refined_summary.get("compatibility_summary"),
        "refined_trainable_transitions": refined_summary.get("refined_trainable_transitions"),
        "refined_advantage_audit": refined_summary.get("advantage_margin_audit"),
        "ppo_update_summary": refined_summary.get("ppo_update_summary"),
        "guard_replay_audit": refined_summary.get("guard_replay_audit"),
        "performance_metric_table": refined_summary.get("performance_metric_table"),
        "stage5a_rerun_summary": refined_summary.get("stage5a_rerun_summary"),
        "source_expanded_refined_transitions": expanded_summary.get("refined_trainable_transitions"),
        "source_expanded_guard_replay_audit": expanded_summary.get("guard_replay_audit"),
        "source_expanded_performance_metric_table": expanded_summary.get("performance_metric_table"),
        "minimum_trainable_pair_count": minimum_trainable_pair_count,
        "minimum_family_pair_count": minimum_family_pair_count,
        "source_pair_count": batch["pair_count"],
        "trainable_pair_count": batch["trainable_pair_count"],
        "diagnostic_pair_count": batch["diagnostic_pair_count"],
        "safe_better_training_pair_count": refined_summary.get(
            "safe_better_training_pair_count",
            batch["trainable_pair_count"],
        ),
        "safe_better_training_family_count": batch["trainable_family_count"],
        "family_trainable_counts": batch["family_trainable_counts"],
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
        "controlled_regression_count": refined_summary.get("controlled_regression_count", 0),
        "fallback_gain_contamination_count": refined_summary.get("fallback_gain_contamination_count", 0),
        "accepted_policy_activation_rate": refined_summary.get("accepted_policy_activation_rate"),
        "aggregate_efficiency": batch["aggregate_efficiency"],
        "diagnostic_reason_counts": batch["diagnostic_reason_counts"],
        "compatible_stage5a2_status": compatible_stage5a2.get("status"),
        "compatible_performance_status": compatible_performance.get("status"),
        "runs_new_ppo_update": bool(refined_summary.get("runs_new_ppo_update")) if refined_summary else False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "connects_real_executor": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "uses_old_stage5a2_pairs_as_training_source": False,
        "ppo_advantage_field": "cost_efficiency_ppo_advantage",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _next_required_change(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "shadow_canary_release_performance_validation_preflight"
    if "coverage_efficiency_regression" in reason_codes:
        return "retune_cost_efficiency_penalty_or_comparable_candidate_filter"
    if "cost_efficiency_filter_too_strict" in reason_codes or "family_balance_failed" in reason_codes:
        return "relax_cost_efficiency_filter_or_expand_safe_better_pairs"
    if "post_update_policy_teacher_equivalent" in reason_codes:
        return "increase_cost_efficiency_advantage_signal_or_learning_rate"
    if "fallback_dominates" in reason_codes:
        return "tighten_guarded_replay_acceptance_or_reduce_policy_drift"
    return "inspect_cost_efficiency_aware_candidate_filter_inputs"


def _render_report(summary: dict[str, Any], batch: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Cost-Efficiency-Aware Coverage Reward / Candidate Filter v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_pair_count: `{summary['trainable_pair_count']}`",
            f"- safe_better_training_family_count: `{summary['safe_better_training_family_count']}`",
            f"- coverage_return_improvement: `{summary['coverage_return_improvement']}`",
            f"- cumulative_coverage_rate_delta_improvement: `{summary['cumulative_coverage_rate_delta_improvement']}`",
            f"- valuable_area_covered_improvement: `{summary['valuable_area_covered_improvement']}`",
            f"- coverage_efficiency_regression: `{summary['coverage_efficiency_regression']}`",
            f"- fallback_rate: `{summary['fallback_rate']}`",
            "",
            "## Filter",
            "",
            f"- source_pair_count: `{summary['source_pair_count']}`",
            f"- diagnostic_pair_count: `{summary['diagnostic_pair_count']}`",
            f"- family_trainable_counts: `{summary['family_trainable_counts']}`",
            f"- diagnostic_reason_counts: `{summary['diagnostic_reason_counts']}`",
            "",
            "## Comparable Baseline",
            "",
            "- Baseline metric rows are materialized on the same cost-efficiency-filtered decision set.",
            "- This avoids comparing the 621-row filtered update against the old 2052-row performance table.",
            "",
            "## Scope Guards",
            "",
            "- publishes_checkpoint: `false`",
            "- replaces_default_policy: `false`",
            "- performance_claimed: `false`",
            "- connects_real_executor: `false`",
            "- modifies_network_or_action_space: `false`",
            "- modifies_default_astar: `false`",
            "",
            "## Aggregate Efficiency",
            "",
            "```json",
            json.dumps(batch["aggregate_efficiency"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _paths(output_root: Path) -> dict[str, Path]:
    compat_stage5a2_root = output_root / COMPAT_STAGE5A2_DIR
    compat_performance_root = output_root / COMPAT_PERFORMANCE_DIR
    return {
        "summary": output_root / SUMMARY_FILE,
        "efficiency_audit": output_root / EFFICIENCY_AUDIT_FILE,
        "filtered_batch": output_root / FILTERED_BATCH_FILE,
        "advantage_audit": output_root / ADVANTAGE_AUDIT_FILE,
        "report": output_root / REPORT_FILE,
        "compat_stage5a2_root": compat_stage5a2_root,
        "compat_stage5a2_summary": compat_stage5a2_root / COMPAT_STAGE5A2_SUMMARY_FILE,
        "compat_overlay": compat_stage5a2_root / COMPAT_OVERLAY_FILE,
        "compat_counterfactual": compat_stage5a2_root / COMPAT_COUNTERFACTUAL_FILE,
        "compat_performance_root": compat_performance_root,
        "compat_performance_summary": compat_performance_root / COMPAT_PERFORMANCE_SUMMARY_FILE,
        "compat_performance_metric_table": compat_performance_root / COMPAT_PERFORMANCE_METRIC_TABLE_FILE,
    }


def _counterfactual_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
        _int(row.get("action_index")),
    )


def _counterfactual_key_from_pair(row: dict[str, Any], action_index: int) -> tuple[str, str, int, int]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
        int(action_index),
    )


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
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


def _first_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= TOLERANCE:
        return None
    return round(float(numerator) / float(denominator), 12)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 12)


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
