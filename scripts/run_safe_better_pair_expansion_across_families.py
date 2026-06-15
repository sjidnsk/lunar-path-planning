from __future__ import annotations

import argparse
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
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


SUMMARY_SCHEMA_VERSION = "safe-better-pair-expansion-across-families-summary/v1"
PAIR_SCHEMA_VERSION = "safe-better-pair-expansion-row/v1"
GAP_SCHEMA_VERSION = "safe-better-pair-expansion-family-gap-report/v1"

DEFAULT_TUNING_ROOT = "outputs/path_feedback_batch_refined_coverage_reward_margin_tuning_v1"
DEFAULT_STAGE5A2_ROOT = "outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1"
DEFAULT_QUASI_REAL_ROOT = "outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1"
DEFAULT_FORMAL_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1"
DEFAULT_REPLAY_ROOT = "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
DEFAULT_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_REWARD_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1"

TUNING_SUMMARY_FILE = "refined-coverage-reward-margin-tuning-summary.json"
STAGE5A2_SUMMARY_FILE = "policy-differentiating-counterfactual-coverage-rollouts-summary.json"
STAGE5A2_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
STAGE5A2_COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
STAGE5A2_GAP_FILE = "family-action-gap-report.json"
QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE = "quasi-real-map-path-feedback-summary.json"
QUASI_REAL_SAFE_ALTERNATIVE_SUMMARY_FILE = "quasi-real-safe-alternative-opportunity-summary.json"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
REWARD_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"

SUMMARY_FILE = "safe-better-pair-expansion-summary.json"
PAIR_FILE = "expanded-safe-better-pairs.jsonl"
COUNTERFACTUAL_FILE = "expanded-counterfactual-coverage-rollouts.jsonl"
FAMILY_GAP_FILE = "family-safe-better-gap-report.json"
REPORT_FILE = "safe-better-pair-expansion-report.md"

TARGET_FAMILIES = (
    "low_observation_count",
    "mixed_risk",
    "rim_or_steep_slope",
    "smooth_high_confidence",
)
MIN_SAFE_BETTER_PAIR_COUNT = 128
MIN_SAFE_BETTER_FAMILY_COUNT = 4
MIN_SAFE_BETTER_PAIR_COUNT_PER_FAMILY = 8
TOLERANCE = 1.0e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand and audit safe better-than-teacher counterfactual coverage pairs across families."
    )
    parser.add_argument("--tuning-root", default=DEFAULT_TUNING_ROOT)
    parser.add_argument("--stage5a2-root", default=DEFAULT_STAGE5A2_ROOT)
    parser.add_argument("--quasi-real-root", default=DEFAULT_QUASI_REAL_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--coverage-signal-root", default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--reward-refinement-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--supplemental-overlay", action="append", default=[])
    parser.add_argument("--supplemental-counterfactual-rollouts", action="append", default=[])
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_safe_better_pair_expansion_across_families(
        tuning_root=_resolve_path(Path(args.tuning_root), repo_root),
        stage5a2_root=_resolve_path(Path(args.stage5a2_root), repo_root),
        quasi_real_root=_resolve_path(Path(args.quasi_real_root), repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        supplemental_overlay_paths=[
            _resolve_path(Path(path), repo_root) for path in args.supplemental_overlay
        ],
        supplemental_counterfactual_paths=[
            _resolve_path(Path(path), repo_root) for path in args.supplemental_counterfactual_rollouts
        ],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "next_required_change": summary["next_required_change"],
                "safe_better_than_teacher_candidate_count": summary[
                    "safe_better_than_teacher_candidate_count"
                ],
                "safe_better_than_teacher_family_count": summary[
                    "safe_better_than_teacher_family_count"
                ],
                "missing_counterfactual_source_count": summary["missing_counterfactual_source_count"],
                "performance_claimed": summary["performance_claimed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_safe_better_pair_expansion_across_families(
    *,
    tuning_root: Path,
    stage5a2_root: Path,
    quasi_real_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    coverage_signal_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    repo_root: Path,
    supplemental_overlay_paths: list[Path] | None = None,
    supplemental_counterfactual_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    input_reasons: list[str] = []
    supplemental_overlay_paths = [
        _resolve_path(Path(path), repo_root) for path in (supplemental_overlay_paths or [])
    ]
    supplemental_counterfactual_paths = [
        _resolve_path(Path(path), repo_root) for path in (supplemental_counterfactual_paths or [])
    ]

    tuning_summary = _read_json(tuning_root / TUNING_SUMMARY_FILE, input_reasons, "tuning_summary_missing")
    stage5a2_summary = _read_json(
        stage5a2_root / STAGE5A2_SUMMARY_FILE,
        input_reasons,
        "stage5a2_summary_missing",
    )
    quasi_real_summary = _read_json(
        quasi_real_root / QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE,
        input_reasons,
        "quasi_real_path_feedback_summary_missing",
    )
    quasi_real_safe_summary = _read_json(
        quasi_real_root / QUASI_REAL_SAFE_ALTERNATIVE_SUMMARY_FILE,
        input_reasons,
        "quasi_real_safe_alternative_summary_missing",
    )
    formal_summary = _read_json(
        formal_training_root / FORMAL_SUMMARY_FILE,
        input_reasons,
        "formal_training_summary_missing",
    )
    replay_summary = _read_json(
        post_training_replay_root / REPLAY_SUMMARY_FILE,
        input_reasons,
        "post_training_replay_summary_missing",
    )
    signal_summary = _read_json(
        coverage_signal_root / SIGNAL_SUMMARY_FILE,
        input_reasons,
        "coverage_signal_summary_missing",
    )
    reward_summary = _read_json(
        reward_refinement_root / REWARD_SUMMARY_FILE,
        input_reasons,
        "reward_refinement_summary_missing",
    )

    overlay_path = _resolve_optional_path(
        stage5a2_summary.get("candidate_coverage_overlay"),
        stage5a2_root / STAGE5A2_OVERLAY_FILE,
        repo_root,
    )
    counterfactual_path = _resolve_optional_path(
        stage5a2_summary.get("counterfactual_coverage_rollouts"),
        stage5a2_root / STAGE5A2_COUNTERFACTUAL_FILE,
        repo_root,
    )
    stage5a2_gap_path = _resolve_optional_path(
        stage5a2_summary.get("family_action_gap_report"),
        stage5a2_root / STAGE5A2_GAP_FILE,
        repo_root,
    )
    base_overlay_rows = _read_jsonl(overlay_path, input_reasons, "candidate_coverage_overlay_missing")
    base_counterfactual_rows = _read_jsonl(
        counterfactual_path,
        input_reasons,
        "counterfactual_coverage_rollouts_missing",
    )
    stage5a2_gap_report = _read_json(stage5a2_gap_path, input_reasons, "family_action_gap_report_missing")
    supplemental_overlay_rows = _read_jsonl_many(
        supplemental_overlay_paths,
        input_reasons,
        "supplemental_candidate_coverage_overlay_missing",
    )
    supplemental_counterfactual_rows = _read_jsonl_many(
        supplemental_counterfactual_paths,
        input_reasons,
        "supplemental_counterfactual_coverage_rollouts_missing",
    )
    overlay_rows, deduplicated_overlay_row_count = _dedupe_rows(
        [*base_overlay_rows, *supplemental_overlay_rows],
        key_fields=("context_id", "episode_id", "step_index", "scenario_family", "split", "action_index"),
    )
    counterfactual_rows, deduplicated_counterfactual_row_count = _dedupe_rows(
        [*base_counterfactual_rows, *supplemental_counterfactual_rows],
        key_fields=("context_id", "episode_id", "step_index", "scenario_family", "split", "action_index"),
    )

    expansion = _expanded_pairs(overlay_rows=overlay_rows, counterfactual_rows=counterfactual_rows)
    reason_codes = _reason_codes(input_reasons=input_reasons, expansion=expansion)
    status = "passed" if not reason_codes else "failed"
    next_required_change = (
        "rerun_refined_coverage_driven_ppo_improvement"
        if status == "passed"
        else "expand_safe_better_pair_generation_across_families"
    )

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "output_root": str(output_root),
        "input_artifact_paths": {
            "tuning_summary": str(tuning_root / TUNING_SUMMARY_FILE),
            "stage5a2_summary": str(stage5a2_root / STAGE5A2_SUMMARY_FILE),
            "candidate_coverage_overlay": str(overlay_path),
            "counterfactual_coverage_rollouts": str(counterfactual_path),
            "family_action_gap_report": str(stage5a2_gap_path),
            "quasi_real_path_feedback_summary": str(quasi_real_root / QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE),
            "quasi_real_safe_alternative_summary": str(
                quasi_real_root / QUASI_REAL_SAFE_ALTERNATIVE_SUMMARY_FILE
            ),
            "formal_training_summary": str(formal_training_root / FORMAL_SUMMARY_FILE),
            "post_training_replay_summary": str(post_training_replay_root / REPLAY_SUMMARY_FILE),
            "coverage_signal_summary": str(coverage_signal_root / SIGNAL_SUMMARY_FILE),
            "reward_refinement_summary": str(reward_refinement_root / REWARD_SUMMARY_FILE),
        },
        "supplemental_input_artifact_paths": {
            "candidate_coverage_overlays": [str(path) for path in supplemental_overlay_paths],
            "counterfactual_coverage_rollouts": [
                str(path) for path in supplemental_counterfactual_paths
            ],
        },
        "produced_artifacts": {
            "safe_better_pairs": str(paths["pairs"]),
            "expanded_counterfactual_coverage_rollouts": str(paths["counterfactual_rows"]),
            "family_safe_better_gap_report": str(paths["family_gap"]),
            "report": str(paths["report"]),
        },
        "tuning_status": tuning_summary.get("status"),
        "tuning_next_required_change": tuning_summary.get("next_required_change"),
        "tuning_safe_better_training_pair_count": _int(
            tuning_summary.get("safe_better_training_pair_count")
        ),
        "tuning_ppo_advantage_nonzero_count": _int(tuning_summary.get("ppo_advantage_nonzero_count")),
        "stage5a2_status": stage5a2_summary.get("status"),
        "stage5a2_safe_better_than_teacher_candidate_count": _int(
            stage5a2_summary.get("safe_better_than_teacher_candidate_count")
        ),
        "stage5a2_safe_better_than_teacher_family_count": _int(
            stage5a2_summary.get("safe_better_than_teacher_family_count")
        ),
        "stage5a2_family_action_gap_report": stage5a2_gap_report,
        "formal_training_status": formal_summary.get("status"),
        "post_training_replay_status": replay_summary.get("status"),
        "coverage_signal_status": signal_summary.get("status"),
        "reward_refinement_status": reward_summary.get("status"),
        "quasi_real_status": quasi_real_summary.get("status"),
        "quasi_real_safe_alternative_status": quasi_real_safe_summary.get("status"),
        "base_candidate_overlay_row_count": len(base_overlay_rows),
        "supplemental_overlay_row_count": len(supplemental_overlay_rows),
        "deduplicated_overlay_row_count": deduplicated_overlay_row_count,
        "candidate_overlay_row_count": len(overlay_rows),
        "base_counterfactual_coverage_row_count": len(base_counterfactual_rows),
        "supplemental_counterfactual_row_count": len(supplemental_counterfactual_rows),
        "deduplicated_counterfactual_row_count": deduplicated_counterfactual_row_count,
        "counterfactual_coverage_row_count": len(counterfactual_rows),
        "expanded_safe_better_pair_count": len(expansion["pairs"]),
        "safe_better_than_teacher_candidate_count": expansion["trainable_pair_count"],
        "safe_better_than_teacher_family_count": expansion["trainable_family_count"],
        "trainable_safe_better_pair_count": expansion["trainable_pair_count"],
        "diagnostic_safe_better_pair_count": expansion["diagnostic_pair_count"],
        "minimum_safe_better_pair_count": MIN_SAFE_BETTER_PAIR_COUNT,
        "minimum_safe_better_family_count": MIN_SAFE_BETTER_FAMILY_COUNT,
        "minimum_safe_better_pair_count_per_family": MIN_SAFE_BETTER_PAIR_COUNT_PER_FAMILY,
        "target_families": list(TARGET_FAMILIES),
        "family_safe_better_counts": expansion["family_trainable_counts"],
        "family_diagnostic_safe_better_counts": expansion["family_diagnostic_counts"],
        "missing_counterfactual_source_count": expansion["missing_counterfactual_source_count"],
        "fallback_gain_contamination_count": expansion["fallback_gain_contamination_count"],
        "controlled_regression_count": expansion["controlled_regression_count"],
        "guard_rejected_candidate_count": expansion["guard_rejected_candidate_count"],
        "non_positive_coverage_advantage_count": expansion["non_positive_coverage_advantage_count"],
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "git_provenance": git_snapshot(repo_root),
        "summary": _summary_sentence(status=status, expansion=expansion, reason_codes=reason_codes),
    }
    family_gap_report = _family_gap_report(expansion)

    _write_jsonl(paths["pairs"], expansion["pairs"])
    _write_jsonl(paths["counterfactual_rows"], expansion["expanded_counterfactual_rows"])
    _write_json(paths["family_gap"], family_gap_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, family_gap_report), encoding="utf-8")
    return summary


def _expanded_pairs(
    *,
    overlay_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in overlay_rows:
        decisions[_decision_key(row)].append(row)
    counterfactual_index = {
        (*_decision_key(row), row.get("action_index")): row
        for row in counterfactual_rows
        if row.get("action_index") is not None
    }

    pairs: list[dict[str, Any]] = []
    expanded_counterfactual_rows: list[dict[str, Any]] = []
    family_trainable_counts: Counter[str] = Counter()
    family_diagnostic_counts: Counter[str] = Counter()
    missing_counterfactual_source_count = 0
    fallback_gain_contamination_count = 0
    controlled_regression_count = 0
    guard_rejected_candidate_count = 0
    non_positive_coverage_advantage_count = 0

    for key, rows in sorted(decisions.items(), key=lambda item: tuple(str(part) for part in item[0])):
        teacher = next((row for row in rows if bool(row.get("is_teacher_action"))), None)
        if teacher is None:
            continue
        for candidate in rows:
            if bool(candidate.get("is_teacher_action")):
                continue
            coverage_advantage = _coverage_value(candidate) - _coverage_value(teacher)
            if coverage_advantage <= TOLERANCE:
                non_positive_coverage_advantage_count += 1
                continue
            if _is_fallback_like(candidate):
                fallback_gain_contamination_count += 1
                continue
            if bool(candidate.get("guard_rejected")) or not bool(candidate.get("action_mask_valid", True)):
                guard_rejected_candidate_count += 1
                continue
            if _controlled_regression_reasons(candidate):
                controlled_regression_count += 1
                continue

            counterfactual = counterfactual_index.get((*key, candidate.get("action_index")))
            source_available = bool(candidate.get("coverage_source_available")) or bool(
                (counterfactual or {}).get("coverage_source_available")
            )
            if not source_available:
                missing_counterfactual_source_count += 1
                continue

            pair = _pair_row(candidate=candidate, teacher=teacher, counterfactual=counterfactual)
            pairs.append(pair)
            expanded_counterfactual_rows.append(
                _expanded_counterfactual_row(
                    candidate=candidate,
                    teacher=teacher,
                    counterfactual=counterfactual,
                    pair=pair,
                )
            )
            family = str(pair["scenario_family"])
            if pair["ppo_trainable"]:
                family_trainable_counts[family] += 1
            else:
                family_diagnostic_counts[family] += 1

    return {
        "pairs": pairs,
        "expanded_counterfactual_rows": expanded_counterfactual_rows,
        "trainable_pair_count": sum(1 for row in pairs if bool(row.get("ppo_trainable"))),
        "diagnostic_pair_count": sum(1 for row in pairs if not bool(row.get("ppo_trainable"))),
        "trainable_family_count": sum(1 for value in family_trainable_counts.values() if value > 0),
        "family_trainable_counts": dict(sorted(family_trainable_counts.items())),
        "family_diagnostic_counts": dict(sorted(family_diagnostic_counts.items())),
        "missing_counterfactual_source_count": missing_counterfactual_source_count,
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "controlled_regression_count": controlled_regression_count,
        "guard_rejected_candidate_count": guard_rejected_candidate_count,
        "non_positive_coverage_advantage_count": non_positive_coverage_advantage_count,
    }


def _pair_row(
    *,
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    counterfactual: dict[str, Any] | None,
) -> dict[str, Any]:
    split = str(candidate.get("split") or teacher.get("split") or "unknown")
    coverage_advantage = _coverage_value(candidate) - _coverage_value(teacher)
    new_area_advantage = _float(candidate.get("expected_new_coverage_area")) - _float(
        teacher.get("expected_new_coverage_area")
    )
    information_gain_advantage = _float(candidate.get("information_gain")) - _float(
        teacher.get("information_gain")
    )
    valuable_advantage = _valuable_value(candidate) - _valuable_value(teacher)
    path_cost_delta = _float(candidate.get("path_cost")) - _float(teacher.get("path_cost"))
    risk_delta = _float(candidate.get("risk")) - _float(teacher.get("risk"))
    energy_delta = _float(candidate.get("energy_cost")) - _float(teacher.get("energy_cost"))
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "context_id": candidate.get("context_id"),
        "episode_id": candidate.get("episode_id"),
        "step_index": candidate.get("step_index"),
        "scenario_id": candidate.get("scenario_id") or teacher.get("scenario_id"),
        "scenario_family": candidate.get("scenario_family") or teacher.get("scenario_family"),
        "split": split,
        "candidate_action_index": candidate.get("action_index"),
        "teacher_action_index": teacher.get("action_index", candidate.get("teacher_action_index")),
        "expected_coverage_rate_delta": _coverage_value(candidate),
        "teacher_expected_coverage_rate_delta": _coverage_value(teacher),
        "coverage_advantage": float(coverage_advantage),
        "expected_new_coverage_area": _float(candidate.get("expected_new_coverage_area")),
        "teacher_expected_new_coverage_area": _float(teacher.get("expected_new_coverage_area")),
        "new_area_advantage": float(new_area_advantage),
        "information_gain": _float(candidate.get("information_gain")),
        "teacher_information_gain": _float(teacher.get("information_gain")),
        "information_gain_advantage": float(information_gain_advantage),
        "valuable_coverage_proxy": _valuable_value(candidate),
        "teacher_valuable_coverage_proxy": _valuable_value(teacher),
        "valuable_coverage_advantage": float(valuable_advantage),
        "path_cost": _float(candidate.get("path_cost")),
        "teacher_path_cost": _float(teacher.get("path_cost")),
        "path_cost_delta": float(path_cost_delta),
        "risk": _float(candidate.get("risk")),
        "teacher_risk": _float(teacher.get("risk")),
        "risk_delta": float(risk_delta),
        "energy_cost": _float(candidate.get("energy_cost")),
        "teacher_energy_cost": _float(teacher.get("energy_cost")),
        "energy_delta": float(energy_delta),
        "coverage_source_available": True,
        "source_path": candidate.get("source_path") or (counterfactual or {}).get("source_path"),
        "match_method": candidate.get("match_method") or (counterfactual or {}).get("match_method"),
        "source_confidence": _float(candidate.get("source_confidence"), default=1.0),
        "counterfactual_coverage_source_path": (counterfactual or {}).get("source_path")
        or candidate.get("source_path"),
        "counterfactual_coverage_match_method": (counterfactual or {}).get("match_method")
        or candidate.get("match_method"),
        "safe_better_than_teacher_candidate": True,
        "safe_better_basis": "counterfactual_coverage_advantage_positive",
        "ppo_trainable": split == "train",
        "diagnostic_only": split != "train",
    }


def _expanded_counterfactual_row(
    *,
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    counterfactual: dict[str, Any] | None,
    pair: dict[str, Any],
) -> dict[str, Any]:
    base = dict(counterfactual or candidate)
    base.update(
        {
            "schema_version": "safe-better-expanded-counterfactual-coverage-row/v1",
            "context_id": pair["context_id"],
            "episode_id": pair["episode_id"],
            "step_index": pair["step_index"],
            "scenario_family": pair["scenario_family"],
            "split": pair["split"],
            "action_index": pair["candidate_action_index"],
            "teacher_action_index": pair["teacher_action_index"],
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "coverage_advantage": pair["coverage_advantage"],
            "teacher_expected_coverage_rate_delta": pair["teacher_expected_coverage_rate_delta"],
            "teacher_source_path": teacher.get("source_path"),
            "ppo_trainable": pair["ppo_trainable"],
            "diagnostic_only": pair["diagnostic_only"],
        }
    )
    return base


def _reason_codes(*, input_reasons: list[str], expansion: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for reason in input_reasons:
        _add_reason(reasons, reason)
    if expansion["trainable_pair_count"] < MIN_SAFE_BETTER_PAIR_COUNT:
        _add_reason(reasons, "safe_better_pair_count_below_threshold")
    if expansion["trainable_family_count"] < MIN_SAFE_BETTER_FAMILY_COUNT:
        _add_reason(reasons, "safe_better_family_count_below_threshold")
    for family in TARGET_FAMILIES:
        if int(expansion["family_trainable_counts"].get(family, 0)) < MIN_SAFE_BETTER_PAIR_COUNT_PER_FAMILY:
            _add_reason(reasons, f"family_safe_better_gap_{family}")
    if expansion["missing_counterfactual_source_count"] > 0:
        _add_reason(reasons, "counterfactual_coverage_source_missing")
    if expansion["fallback_gain_contamination_count"] > 0:
        _add_reason(reasons, "fallback_gain_contamination")
    if expansion["controlled_regression_count"] > 0:
        _add_reason(reasons, "controlled_regression")
    return reasons


def _family_gap_report(expansion: dict[str, Any]) -> dict[str, Any]:
    rows = []
    all_families = set(TARGET_FAMILIES)
    all_families.update(expansion["family_trainable_counts"].keys())
    all_families.update(expansion["family_diagnostic_counts"].keys())
    for family in sorted(all_families):
        trainable_count = int(expansion["family_trainable_counts"].get(family, 0))
        diagnostic_count = int(expansion["family_diagnostic_counts"].get(family, 0))
        rows.append(
            {
                "scenario_family": family,
                "trainable_safe_better_pair_count": trainable_count,
                "diagnostic_safe_better_pair_count": diagnostic_count,
                "minimum_trainable_safe_better_pair_count": MIN_SAFE_BETTER_PAIR_COUNT_PER_FAMILY,
                "family_gate_passed": trainable_count >= MIN_SAFE_BETTER_PAIR_COUNT_PER_FAMILY,
                "reason_codes": []
                if trainable_count >= MIN_SAFE_BETTER_PAIR_COUNT_PER_FAMILY
                else [f"family_safe_better_gap_{family}"],
            }
        )
    return {
        "schema_version": GAP_SCHEMA_VERSION,
        "rows": rows,
        "target_families": list(TARGET_FAMILIES),
        "minimum_safe_better_pair_count_per_family": MIN_SAFE_BETTER_PAIR_COUNT_PER_FAMILY,
    }


def _render_report(summary: dict[str, Any], family_gap_report: dict[str, Any]) -> str:
    rows = family_gap_report["rows"]
    family_lines = [
        "| Family | Trainable safe-better | Diagnostic safe-better | Gate |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        family_lines.append(
            "| {family} | {trainable} | {diagnostic} | {gate} |".format(
                family=row["scenario_family"],
                trainable=row["trainable_safe_better_pair_count"],
                diagnostic=row["diagnostic_safe_better_pair_count"],
                gate="passed" if row["family_gate_passed"] else "failed",
            )
        )
    return "\n".join(
        [
            "# Safe-Better Pair Expansion Across Families v1",
            "",
            "## Summary",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_safe_better_pair_count: `{summary['trainable_safe_better_pair_count']}`",
            f"- safe_better_than_teacher_family_count: `{summary['safe_better_than_teacher_family_count']}`",
            f"- base_candidate_overlay_row_count: `{summary['base_candidate_overlay_row_count']}`",
            f"- supplemental_overlay_row_count: `{summary['supplemental_overlay_row_count']}`",
            f"- deduplicated_overlay_row_count: `{summary['deduplicated_overlay_row_count']}`",
            f"- missing_counterfactual_source_count: `{summary['missing_counterfactual_source_count']}`",
            f"- fallback_gain_contamination_count: `{summary['fallback_gain_contamination_count']}`",
            f"- performance_claimed: `{summary['performance_claimed']}`",
            "",
            "## Family Gap",
            "",
            *family_lines,
            "",
            "## Non-Goals",
            "",
            "- no PPO update",
            "- no checkpoint publication",
            "- no default policy replacement",
            "- no real executor connection",
            "- no network/action-space/default-A* change",
            "- no guard relaxation",
            "- no performance claim",
            "",
        ]
    )


def _summary_sentence(*, status: str, expansion: dict[str, Any], reason_codes: list[str]) -> str:
    if status == "passed":
        return (
            "Safe-better pair expansion passed with "
            f"{expansion['trainable_pair_count']} trainable pairs across "
            f"{expansion['trainable_family_count']} families."
        )
    return (
        "Safe-better pair expansion failed because "
        f"{reason_codes}; current trainable pairs="
        f"{expansion['trainable_pair_count']} across {expansion['trainable_family_count']} families."
    )


def _decision_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("context_id"), row.get("episode_id"), row.get("step_index"))


def _coverage_value(row: dict[str, Any]) -> float:
    return _float(row.get("expected_coverage_rate_delta"))


def _valuable_value(row: dict[str, Any]) -> float:
    value = row.get("valuable_coverage_proxy")
    if value is None:
        value = row.get("value")
    return _float(value)


def _is_fallback_like(row: dict[str, Any]) -> bool:
    return bool(row.get("fallback_like")) or str(row.get("source_execution_type") or "").startswith("fallback")


def _controlled_regression_reasons(row: dict[str, Any]) -> list[Any]:
    reasons = row.get("controlled_regression_reason_codes")
    return reasons if isinstance(reasons, list) else []


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "pairs": output_root / PAIR_FILE,
        "counterfactual_rows": output_root / COUNTERFACTUAL_FILE,
        "family_gap": output_root / FAMILY_GAP_FILE,
        "report": output_root / REPORT_FILE,
    }


def _read_json(path: Path, reasons: list[str], missing_reason: str) -> dict[str, Any]:
    if not path.is_file():
        _add_reason(reasons, missing_reason)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reasons, f"{missing_reason}_invalid_json")
        return {}


def _read_jsonl(path: Path, reasons: list[str], missing_reason: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _add_reason(reasons, missing_reason)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                _add_reason(reasons, f"{missing_reason}_invalid_jsonl")
                break
    return rows


def _read_jsonl_many(
    paths: list[Path],
    reasons: list[str],
    missing_reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path, reasons, missing_reason))
    return rows


def _dedupe_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], int]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(field) for field in key_fields)
        if all(value is None for value in key):
            key = ("__row_index__", index)
        indexed[key] = row
    return list(indexed.values()), len(rows) - len(indexed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_optional_path(value: Any, fallback: Path, repo_root: Path) -> Path:
    if isinstance(value, str) and value:
        return _resolve_path(Path(value), repo_root)
    return fallback


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, *, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
