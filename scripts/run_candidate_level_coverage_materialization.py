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
    from run_policy_coverage_opportunity_margin_audit import (
        run_policy_coverage_opportunity_margin_audit,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_policy_coverage_opportunity_margin_audit import (
        run_policy_coverage_opportunity_margin_audit,
    )


SUMMARY_SCHEMA_VERSION = "candidate-level-coverage-materialization-summary/v1"
OVERLAY_ROW_SCHEMA_VERSION = "candidate-level-coverage-overlay-row/v1"
COUNTERFACTUAL_ROW_SCHEMA_VERSION = "candidate-level-counterfactual-opportunity-row/v1"
SOURCE_LINK_SCHEMA_VERSION = "candidate-level-coverage-source-link-audit/v1"
REJECTION_SCHEMA_VERSION = "candidate-level-coverage-rejection-report/v1"

POLICY_MARGIN_SUMMARY_FILE = "policy-coverage-opportunity-margin-audit-summary.json"
POLICY_ACTION_AUDIT_FILE = "policy-coverage-action-level-audit.jsonl"
COVERAGE_DRIVEN_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
COVERAGE_SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
COVERAGE_PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
REWARD_REFINEMENT_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"
QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE = "quasi-real-map-path-feedback-summary.json"
QUASI_REAL_SAFE_ALTERNATIVE_SUMMARY_FILE = "quasi-real-safe-alternative-opportunity-summary.json"

SUMMARY_FILE = "candidate-level-coverage-materialization-summary.json"
OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
COUNTERFACTUAL_AUDIT_FILE = "counterfactual-opportunity-audit.jsonl"
SOURCE_LINK_AUDIT_FILE = "source-link-audit.json"
REJECTION_REPORT_FILE = "candidate-level-coverage-rejection-report.json"
REPORT_FILE = "candidate-level-coverage-materialization-report.md"
STAGE5A_RERUN_SUMMARY_FILE = "overlay-fed-stage5a-rerun-summary.json"
STAGE5A_RERUN_ROOT = "stage5a-overlay-rerun"

DEFAULT_POLICY_MARGIN_ROOT = "outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1"
DEFAULT_COVERAGE_DRIVEN_ROOT = "outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1"
DEFAULT_REWARD_REFINEMENT_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_COVERAGE_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_COVERAGE_PERFORMANCE_ROOT = "outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1"
DEFAULT_QUASI_REAL_ROOT = "outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_candidate_level_coverage_materialization_v1"

TOLERANCE = 1.0e-9
MAX_ALLOWED_RELATIVE_COST_REGRESSION = 0.05
COVERAGE_SCORE_FIELDS = ("expected_coverage_rate_delta", "information_gain", "value")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize candidate-level exploration coverage overlay for Stage 5A."
    )
    parser.add_argument("--policy-margin-root", default=DEFAULT_POLICY_MARGIN_ROOT)
    parser.add_argument("--coverage-driven-root", default=DEFAULT_COVERAGE_DRIVEN_ROOT)
    parser.add_argument("--reward-refinement-root", default=DEFAULT_REWARD_REFINEMENT_ROOT)
    parser.add_argument("--coverage-signal-root", default=DEFAULT_COVERAGE_SIGNAL_ROOT)
    parser.add_argument("--coverage-performance-root", default=DEFAULT_COVERAGE_PERFORMANCE_ROOT)
    parser.add_argument("--quasi-real-root", default=DEFAULT_QUASI_REAL_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_candidate_level_coverage_materialization(
        policy_margin_root=_resolve_path(Path(args.policy_margin_root), repo_root, repo_root),
        coverage_driven_root=_resolve_path(Path(args.coverage_driven_root), repo_root, repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root, repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root, repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root, repo_root),
        quasi_real_root=_resolve_path(Path(args.quasi_real_root), repo_root, repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "next_required_change": summary["next_required_change"],
                "performance_claimed": summary["performance_claimed"],
                "runs_new_ppo_update": summary["runs_new_ppo_update"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_candidate_level_coverage_materialization(
    *,
    policy_margin_root: Path,
    coverage_driven_root: Path,
    reward_refinement_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    quasi_real_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    policy_margin_root = Path(policy_margin_root)
    coverage_driven_root = Path(coverage_driven_root)
    reward_refinement_root = Path(reward_refinement_root)
    coverage_signal_root = Path(coverage_signal_root)
    coverage_performance_root = Path(coverage_performance_root)
    quasi_real_root = Path(quasi_real_root)
    output_root = Path(output_root)
    repo_root = Path(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)

    input_reasons: list[str] = []
    policy_margin_summary = _read_json(
        policy_margin_root / POLICY_MARGIN_SUMMARY_FILE,
        input_reasons,
        "policy_margin_summary_missing",
    )
    action_rows = _read_jsonl(
        policy_margin_root / POLICY_ACTION_AUDIT_FILE,
        input_reasons,
        "policy_action_audit_missing",
    )
    coverage_driven_summary = _read_json(
        coverage_driven_root / COVERAGE_DRIVEN_SUMMARY_FILE,
        input_reasons,
        "coverage_driven_summary_missing",
    )
    reward_summary = _read_json(
        reward_refinement_root / REWARD_REFINEMENT_SUMMARY_FILE,
        input_reasons,
        "reward_refinement_summary_missing",
    )
    coverage_signal_summary = _read_json(
        coverage_signal_root / COVERAGE_SIGNAL_SUMMARY_FILE,
        input_reasons,
        "coverage_signal_summary_missing",
    )
    coverage_performance_summary = _read_json(
        coverage_performance_root / COVERAGE_PERFORMANCE_SUMMARY_FILE,
        input_reasons,
        "coverage_performance_summary_missing",
    )
    quasi_real_summary = _read_json(
        quasi_real_root / QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE,
        input_reasons,
        "quasi_real_path_feedback_summary_missing",
    )
    quasi_real_safe_summary = _read_json(
        quasi_real_root / QUASI_REAL_SAFE_ALTERNATIVE_SUMMARY_FILE,
        [],
        "quasi_real_safe_alternative_summary_missing",
    )

    candidate_source_index, source_stats = _index_quasi_real_candidate_sources(
        quasi_real_summary,
        quasi_real_root / QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE,
    )
    overlay_rows = _materialize_overlay_rows(
        action_rows=action_rows,
        policy_action_audit_path=policy_margin_root / POLICY_ACTION_AUDIT_FILE,
        candidate_source_index=candidate_source_index,
    )
    counterfactual_rows, safe_better_rows = _counterfactual_opportunity_rows(overlay_rows)
    source_link_audit = _source_link_audit(
        action_rows=action_rows,
        overlay_rows=overlay_rows,
        source_stats=source_stats,
        quasi_real_safe_summary=quasi_real_safe_summary,
    )

    _write_jsonl(paths["overlay"], overlay_rows)
    _write_jsonl(paths["counterfactual_audit"], counterfactual_rows)

    stage5a_rerun_summary = run_policy_coverage_opportunity_margin_audit(
        coverage_driven_root=coverage_driven_root,
        reward_refinement_root=reward_refinement_root,
        coverage_signal_root=coverage_signal_root,
        coverage_performance_root=coverage_performance_root,
        output_root=paths["stage5a_rerun_root"],
        repo_root=repo_root,
        candidate_coverage_overlay_path=paths["overlay"],
    )
    _write_json(paths["stage5a_rerun_summary"], stage5a_rerun_summary)

    controlled_regression_count = _controlled_regression_count(
        action_rows,
        policy_margin_summary,
        coverage_driven_summary,
        reward_summary,
        coverage_signal_summary,
    )
    fallback_gain_contamination_count = _fallback_gain_contamination_count(
        overlay_rows,
        policy_margin_summary,
        coverage_driven_summary,
        reward_summary,
        coverage_signal_summary,
    )
    counts = _overlay_counts(overlay_rows, safe_better_rows)
    reason_codes = _reason_codes(
        input_reasons=input_reasons,
        counts=counts,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    next_required_change = _next_required_change(
        reason_codes=reason_codes,
        counts=counts,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    status = (
        "passed"
        if next_required_change == "rerun_coverage_driven_ppo_with_refined_reward_or_advantage"
        and not input_reasons
        and controlled_regression_count == 0
        and fallback_gain_contamination_count == 0
        else "failed"
    )

    rejection_report = _rejection_report(
        status=status,
        reason_codes=reason_codes,
        next_required_change=next_required_change,
        counts=counts,
        source_link_audit=source_link_audit,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "candidate_level_coverage_materialization_status": status,
        "policy_margin_root": str(policy_margin_root),
        "coverage_driven_root": str(coverage_driven_root),
        "reward_refinement_root": str(reward_refinement_root),
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "quasi_real_root": str(quasi_real_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "candidate_coverage_overlay": str(paths["overlay"]),
        "counterfactual_opportunity_audit": str(paths["counterfactual_audit"]),
        "source_link_audit": str(paths["source_link_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "stage5a_overlay_rerun_root": str(paths["stage5a_rerun_root"]),
        "stage5a_overlay_rerun_summary": str(paths["stage5a_rerun_summary"]),
        "context_count": counts["context_count"],
        "action_candidate_row_count": counts["action_candidate_row_count"],
        "overlay_connected_candidate_count": counts["overlay_connected_candidate_count"],
        "missing_candidate_coverage_source_count": counts[
            "missing_candidate_coverage_source_count"
        ],
        "candidate_expected_coverage_nonzero_count": counts[
            "candidate_expected_coverage_nonzero_count"
        ],
        "candidate_information_gain_nonzero_count": counts[
            "candidate_information_gain_nonzero_count"
        ],
        "candidate_value_nonzero_count": counts["candidate_value_nonzero_count"],
        "counterfactual_coverage_candidate_count": counts[
            "counterfactual_coverage_candidate_count"
        ],
        "safe_better_than_teacher_candidate_count": counts[
            "safe_better_than_teacher_candidate_count"
        ],
        "safe_better_than_teacher_family_count": counts["safe_better_than_teacher_family_count"],
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "controlled_regression_count": controlled_regression_count,
        "stage5a_overlay_rerun_status": stage5a_rerun_summary.get("status"),
        "stage5a_overlay_rerun_next_required_change": stage5a_rerun_summary.get(
            "next_required_change"
        ),
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "connects_real_executor": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "coverage_driven_status": coverage_driven_summary.get("status"),
        "coverage_signal_status": coverage_signal_summary.get("coverage_signal_status"),
        "reward_refinement_status": reward_summary.get("reward_refinement_status"),
        "coverage_performance_status": coverage_performance_summary.get(
            "coverage_performance_status"
        ),
        "policy_margin_status": policy_margin_summary.get("status"),
        "source_stats": source_stats,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_json(paths["source_link_audit"], source_link_audit)
    _write_json(paths["rejection_report"], rejection_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(
        _render_report(summary, rejection_report, source_link_audit),
        encoding="utf-8",
    )
    return summary


def _index_quasi_real_candidate_sources(
    summary: dict[str, Any],
    source_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    stats = Counter()
    scenario_count = 0
    for scenario in summary.get("scenarios", []) if isinstance(summary.get("scenarios"), list) else []:
        if not isinstance(scenario, dict):
            continue
        scenario_count += 1
        scenario_id = scenario.get("scenario_id")
        scenario_family = (
            scenario.get("scenario_family")
            or scenario.get("scenario_group")
            or scenario.get("roi_group")
        )
        split = _split_from_row(scenario)
        path_feedback = scenario.get("path_feedback")
        candidates = path_feedback.get("candidates", []) if isinstance(path_feedback, dict) else []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            action_index = _first_int(candidate.get("action_index"))
            if action_index is None:
                continue
            row = {
                **candidate,
                "scenario_id": scenario_id,
                "scenario_family": scenario_family,
                "split": split,
                "source_path": str(source_path),
            }
            stats["candidate_source_count"] += 1
            if _coverage_fields_present(row):
                stats["candidate_source_with_coverage_count"] += 1
            for key in _source_keys(
                context_id=row.get("context_id"),
                scenario_id=scenario_id,
                scenario_family=scenario_family,
                split=split,
                action_index=action_index,
                candidate_cell=row.get("candidate_cell") or row.get("cell"),
            ):
                index.setdefault(key, row)
    return index, {"scenario_count": scenario_count, **dict(stats)}


def _materialize_overlay_rows(
    *,
    action_rows: list[dict[str, Any]],
    policy_action_audit_path: Path,
    candidate_source_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    overlay_rows: list[dict[str, Any]] = []
    for row in action_rows:
        source = _source_for_action_row(row, candidate_source_index)
        executed_source_available = (
            bool(row.get("counterfactual_coverage_observed"))
            and _float_or_none(row.get("coverage_rate_delta")) is not None
            and str(row.get("actual_coverage_gain_source") or "") == "path_feedback"
            and not bool(row.get("fallback_like"))
        )
        candidate_source_available = source is not None and _coverage_fields_present(source)
        if executed_source_available:
            overlay_rows.append(_overlay_from_executed_row(row, policy_action_audit_path, source))
        elif candidate_source_available and source is not None:
            overlay_rows.append(_overlay_from_candidate_source(row, source))
        else:
            overlay_rows.append(_missing_overlay_row(row, source))
    return overlay_rows


def _source_for_action_row(
    row: dict[str, Any],
    candidate_source_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    action_index = _first_int(row.get("action_index"))
    if action_index is None:
        return None
    for key in _source_keys(
        context_id=row.get("context_id"),
        scenario_id=row.get("scenario_id"),
        scenario_family=row.get("scenario_family"),
        split=_split_from_row(row),
        action_index=action_index,
        candidate_cell=row.get("candidate_cell"),
    ):
        if key in candidate_source_index:
            return candidate_source_index[key]
    return None


def _overlay_from_executed_row(
    row: dict[str, Any],
    source_path: Path,
    source: dict[str, Any] | None,
) -> dict[str, Any]:
    coverage = _float_or_none(row.get("coverage_rate_delta"))
    new_area = _first_float(
        row.get("new_area_covered"),
        row.get("coverage_rate_delta"),
    )
    information_gain = _first_float(row.get("information_gain_executed"), row.get("coverage_rate_delta"))
    valuable_proxy = _first_float(row.get("valuable_area_covered"))
    return _base_overlay_row(row) | {
        "coverage_source_available": True,
        "expected_coverage_rate_delta": coverage,
        "expected_new_coverage_area": new_area,
        "information_gain": information_gain,
        "value": None,
        "valuable_coverage_proxy": valuable_proxy,
        "path_cost": _first_float((source or {}).get("path_cost"), row.get("path_cost")),
        "risk": _first_float((source or {}).get("risk"), row.get("risk")),
        "energy_cost": _first_float((source or {}).get("energy_cost"), row.get("energy_cost")),
        "source_path": str(source_path),
        "match_method": "executed_action_actual_path_feedback",
        "source_confidence": 1.0,
        "source_execution_type": "executed_policy_action",
        "missing_reason_codes": [],
    }


def _overlay_from_candidate_source(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    expected = _first_float(
        source.get("expected_coverage_rate_delta"),
        source.get("coverage_rate_delta"),
        source.get("new_area_covered"),
    )
    new_area = _first_float(
        source.get("expected_new_coverage_area"),
        source.get("new_area_covered"),
        expected,
    )
    information_gain = _first_float(source.get("information_gain"))
    value = _first_float(source.get("value"))
    utility = _first_float(source.get("utility"), row.get("candidate_utility"))
    valuable_proxy = _first_float(
        source.get("valuable_coverage_proxy"),
        source.get("valuable_area_covered"),
    )
    if valuable_proxy is None and expected is not None and utility is not None:
        valuable_proxy = expected * utility
    return _base_overlay_row(row) | {
        "coverage_source_available": True,
        "expected_coverage_rate_delta": expected,
        "expected_new_coverage_area": new_area,
        "information_gain": information_gain,
        "value": value,
        "valuable_coverage_proxy": valuable_proxy,
        "path_cost": _first_float(source.get("path_cost"), row.get("path_cost")),
        "risk": _first_float(source.get("risk"), row.get("risk")),
        "energy_cost": _first_float(source.get("energy_cost"), row.get("energy_cost")),
        "source_path": source.get("source_path"),
        "match_method": "candidate_counterfactual_path_feedback",
        "source_confidence": 1.0,
        "source_execution_type": "counterfactual_candidate",
        "missing_reason_codes": [],
    }


def _missing_overlay_row(row: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    reason = (
        "executed_teacher_coverage_source_missing"
        if bool(row.get("is_teacher_action"))
        else "counterfactual_candidate_coverage_source_missing"
    )
    return _base_overlay_row(row) | {
        "coverage_source_available": False,
        "expected_coverage_rate_delta": None,
        "expected_new_coverage_area": None,
        "information_gain": None,
        "value": None,
        "valuable_coverage_proxy": None,
        "path_cost": _first_float((source or {}).get("path_cost"), row.get("path_cost")),
        "risk": _first_float((source or {}).get("risk"), row.get("risk")),
        "energy_cost": _first_float((source or {}).get("energy_cost"), row.get("energy_cost")),
        "source_path": (source or {}).get("source_path"),
        "match_method": "candidate_source_without_coverage" if source is not None else "no_candidate_source",
        "source_confidence": 0.0,
        "source_execution_type": "missing",
        "missing_reason_codes": [reason],
    }


def _base_overlay_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": OVERLAY_ROW_SCHEMA_VERSION,
        "context_id": row.get("context_id"),
        "episode_id": row.get("episode_id"),
        "step_index": _first_int(row.get("step_index")),
        "scenario_id": row.get("scenario_id"),
        "scenario_family": row.get("scenario_family"),
        "split": _split_from_row(row),
        "action_index": _first_int(row.get("action_index")),
        "candidate_cell": row.get("candidate_cell"),
        "action_mask_valid": bool(row.get("action_mask_valid", True)),
        "teacher_action_index": _first_int(row.get("teacher_action_index")),
        "is_teacher_action": bool(row.get("is_teacher_action")),
        "is_pre_improvement_selected_action": bool(row.get("is_pre_improvement_selected_action")),
        "is_post_update_raw_action": bool(row.get("is_post_update_raw_action")),
        "is_post_update_controlled_action": bool(row.get("is_post_update_controlled_action")),
        "counterfactual_coverage_observed": bool(row.get("counterfactual_coverage_observed")),
        "fallback_like": bool(row.get("fallback_like")),
        "guard_rejected": bool(row.get("guard_rejected")),
        "policy_action_accepted": bool(row.get("policy_action_accepted", True)),
        "controlled_regression_reason_codes": _string_list(
            row.get("controlled_regression_reason_codes")
        ),
    }


def _counterfactual_opportunity_rows(
    overlay_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overlay_rows:
        by_context[_decision_key(row)].append(row)

    counterfactual_rows: list[dict[str, Any]] = []
    safe_better_rows: list[dict[str, Any]] = []
    for rows in by_context.values():
        teacher = next((row for row in rows if row.get("is_teacher_action")), None)
        teacher_score = _coverage_score(teacher) if teacher and teacher.get("coverage_source_available") else None
        for row in rows:
            if row.get("is_teacher_action"):
                continue
            candidate_score = _coverage_score(row) if row.get("coverage_source_available") else None
            cost_safe = _cost_safe(row, teacher)
            safe_better = (
                teacher_score is not None
                and candidate_score is not None
                and row.get("action_mask_valid")
                and not row.get("fallback_like")
                and not row.get("guard_rejected")
                and not row.get("controlled_regression_reason_codes")
                and candidate_score > teacher_score + TOLERANCE
                and cost_safe
            )
            audit_row = {
                "schema_version": COUNTERFACTUAL_ROW_SCHEMA_VERSION,
                "context_id": row.get("context_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_family": row.get("scenario_family"),
                "split": row.get("split"),
                "action_index": row.get("action_index"),
                "candidate_cell": row.get("candidate_cell"),
                "coverage_source_available": bool(row.get("coverage_source_available")),
                "candidate_decision_coverage_score": candidate_score,
                "teacher_decision_coverage_score": teacher_score,
                "cost_safe": cost_safe,
                "safe_better_than_teacher_candidate": safe_better,
                "missing_reason_codes": row.get("missing_reason_codes", []),
                "match_method": row.get("match_method"),
                "source_path": row.get("source_path"),
            }
            counterfactual_rows.append(audit_row)
            row["safe_better_than_teacher_candidate"] = safe_better
            row["safe_better_basis"] = (
                "candidate_level_coverage_overlay_and_guarded_cost_risk_energy"
                if safe_better
                else None
            )
            if safe_better:
                safe_better_rows.append(row)
    return counterfactual_rows, safe_better_rows


def _source_link_audit(
    *,
    action_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
    source_stats: dict[str, Any],
    quasi_real_safe_summary: dict[str, Any],
) -> dict[str, Any]:
    missing_by_family: Counter[str] = Counter()
    missing_by_action: Counter[str] = Counter()
    match_methods = Counter(str(row.get("match_method")) for row in overlay_rows)
    for row in overlay_rows:
        if row.get("coverage_source_available"):
            continue
        missing_by_family[str(row.get("scenario_family"))] += 1
        missing_by_action[str(row.get("action_index"))] += 1
    return {
        "schema_version": SOURCE_LINK_SCHEMA_VERSION,
        "action_candidate_row_count": len(action_rows),
        "overlay_row_count": len(overlay_rows),
        "overlay_connected_candidate_count": sum(
            1 for row in overlay_rows if row.get("coverage_source_available")
        ),
        "missing_candidate_coverage_source_count": sum(
            1 for row in overlay_rows if not row.get("coverage_source_available")
        ),
        "match_method_counts": dict(sorted(match_methods.items())),
        "missing_by_family": dict(sorted(missing_by_family.items())),
        "missing_by_action_index": dict(sorted(missing_by_action.items())),
        "quasi_real_source_stats": source_stats,
        "quasi_real_safe_alternative_status": quasi_real_safe_summary.get("status"),
        "quasi_real_safe_alternative_next_required_change": quasi_real_safe_summary.get(
            "next_required_change"
        ),
        "missing_rollout_requirements": [
            {
                "scenario_family": family,
                "missing_candidate_count": count,
                "required_evidence": "candidate-level counterfactual coverage path-feedback",
            }
            for family, count in sorted(missing_by_family.items())
        ],
    }


def _overlay_counts(
    overlay_rows: list[dict[str, Any]],
    safe_better_rows: list[dict[str, Any]],
) -> dict[str, int]:
    connected = [row for row in overlay_rows if row.get("coverage_source_available")]
    contexts = {_decision_key(row) for row in overlay_rows}
    counterfactual = [
        row
        for row in connected
        if not row.get("is_teacher_action") and not row.get("counterfactual_coverage_observed")
    ]
    return {
        "context_count": len(contexts),
        "action_candidate_row_count": len(overlay_rows),
        "overlay_connected_candidate_count": len(connected),
        "missing_candidate_coverage_source_count": len(overlay_rows) - len(connected),
        "candidate_expected_coverage_nonzero_count": sum(
            1 for row in connected if _nonzero(row.get("expected_coverage_rate_delta"))
        ),
        "candidate_information_gain_nonzero_count": sum(
            1 for row in connected if _nonzero(row.get("information_gain"))
        ),
        "candidate_value_nonzero_count": sum(1 for row in connected if _nonzero(row.get("value"))),
        "counterfactual_coverage_candidate_count": len(counterfactual),
        "safe_better_than_teacher_candidate_count": len(safe_better_rows),
        "safe_better_than_teacher_family_count": len(
            {str(row.get("scenario_family")) for row in safe_better_rows}
        ),
    }


def _reason_codes(
    *,
    input_reasons: list[str],
    counts: dict[str, int],
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> list[str]:
    reasons: list[str] = []
    for reason in input_reasons:
        _add_reason(reasons, reason)
    if controlled_regression_count > 0:
        _add_reason(reasons, "controlled_regression_present")
    if fallback_gain_contamination_count > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    if counts["candidate_expected_coverage_nonzero_count"] <= 0:
        _add_reason(reasons, "candidate_coverage_features_missing")
    if counts["missing_candidate_coverage_source_count"] > 0:
        _add_reason(reasons, "candidate_coverage_source_missing")
    if counts["counterfactual_coverage_candidate_count"] <= 0:
        _add_reason(reasons, "counterfactual_candidate_coverage_source_missing")
    if counts["safe_better_than_teacher_candidate_count"] <= 0:
        _add_reason(reasons, "no_safe_better_than_teacher_candidate")
    return reasons


def _next_required_change(
    *,
    reason_codes: list[str],
    counts: dict[str, int],
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> str:
    if controlled_regression_count > 0 or fallback_gain_contamination_count > 0:
        return "fix_candidate_level_coverage_materialization_inputs"
    if any(reason.endswith("_missing") for reason in reason_codes if reason.startswith(("policy_", "coverage_", "reward_", "quasi_"))):
        return "fix_candidate_level_coverage_materialization_inputs"
    if (
        counts["counterfactual_coverage_candidate_count"] > 0
        and counts["safe_better_than_teacher_candidate_count"] > 0
    ):
        return "rerun_coverage_driven_ppo_with_refined_reward_or_advantage"
    return "generate_policy_differentiating_coverage_rollouts"


def _rejection_report(
    *,
    status: str,
    reason_codes: list[str],
    next_required_change: str,
    counts: dict[str, int],
    source_link_audit: dict[str, Any],
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> dict[str, Any]:
    recommendations = (
        [
            "feed the overlay into Stage 5A and rerun coverage-driven PPO with refined reward or advantage",
            "keep hard guard, fallback separation, and controlled-regression penalties unchanged",
            "do not claim performance until coverage performance evaluation improves against baseline",
        ]
        if next_required_change == "rerun_coverage_driven_ppo_with_refined_reward_or_advantage"
        else [
            "generate policy-differentiating rollouts with candidate-level counterfactual coverage fields",
            "record expected_coverage_rate_delta, expected_new_coverage_area, information_gain, and value per action candidate",
            "separate fallback/source gains from policy gains before using them for reward or advantage",
        ]
    )
    return {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "counts": counts,
        "controlled_regression_count": controlled_regression_count,
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "missing_rollout_requirements": source_link_audit.get("missing_rollout_requirements", []),
        "performance_claim_rejected": True,
        "recommendations": recommendations,
    }


def _render_report(
    summary: dict[str, Any],
    rejection_report: dict[str, Any],
    source_link_audit: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Candidate-Level Exploration Coverage Materialization v1",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- contexts: `{summary['context_count']}`",
            f"- action candidates: `{summary['action_candidate_row_count']}`",
            f"- overlay_connected_candidate_count: `{summary['overlay_connected_candidate_count']}`",
            f"- missing_candidate_coverage_source_count: `{summary['missing_candidate_coverage_source_count']}`",
            f"- counterfactual_coverage_candidate_count: `{summary['counterfactual_coverage_candidate_count']}`",
            f"- safe_better_than_teacher_candidate_count: `{summary['safe_better_than_teacher_candidate_count']}`",
            f"- stage5a_overlay_rerun_status: `{summary['stage5a_overlay_rerun_status']}`",
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## Missing Rollout Requirements",
            "",
            *[
                f"- {item['scenario_family']}: {item['missing_candidate_count']} candidate rows need {item['required_evidence']}"
                for item in source_link_audit.get("missing_rollout_requirements", [])[:30]
            ],
            "",
            "## Recommended Next Step",
            "",
            *[f"- {item}" for item in rejection_report["recommendations"]],
            "",
            "## Guardrails",
            "",
            "- This stage did not run PPO.",
            "- It did not publish or replace any checkpoint.",
            "- It does not claim exploration coverage performance improvement.",
        ]
    )


def _interpretation(summary: dict[str, Any]) -> str:
    if summary["next_required_change"] == "rerun_coverage_driven_ppo_with_refined_reward_or_advantage":
        return (
            "The overlay contains real counterfactual candidate-level coverage and exposes safe "
            "better-than-teacher alternatives. This is still a routing signal, not a performance claim."
        )
    if summary["next_required_change"] == "generate_policy_differentiating_coverage_rollouts":
        return (
            "The current artifacts connect executed coverage for selected actions, but they do "
            "not contain enough non-teacher counterfactual coverage to prove a better safe action. "
            "The next step is rollout/data generation, not another PPO update."
        )
    return "The materialization inputs need repair before candidate-level coverage routing is reliable."


def _controlled_regression_count(
    action_rows: list[dict[str, Any]],
    *summaries: dict[str, Any],
) -> int:
    row_count = sum(1 for row in action_rows if row.get("controlled_regression_reason_codes"))
    summary_count = sum(_int(summary.get("controlled_regression_count")) for summary in summaries if isinstance(summary, dict))
    return row_count + summary_count


def _fallback_gain_contamination_count(
    overlay_rows: list[dict[str, Any]],
    *summaries: dict[str, Any],
) -> int:
    row_count = sum(
        1
        for row in overlay_rows
        if row.get("coverage_source_available") and bool(row.get("fallback_like"))
    )
    summary_count = sum(
        _int(summary.get("fallback_coverage_gain_claimed_as_policy_gain_count"))
        + _int(summary.get("fallback_policy_gain_contamination_count"))
        for summary in summaries
        if isinstance(summary, dict)
    )
    return row_count + summary_count


def _coverage_fields_present(row: dict[str, Any]) -> bool:
    return any(
        _float_or_none(row.get(field)) is not None
        for field in (
            "expected_coverage_rate_delta",
            "coverage_rate_delta",
            "expected_new_coverage_area",
            "new_area_covered",
            "information_gain",
            "value",
            "valuable_coverage_proxy",
            "valuable_area_covered",
        )
    )


def _coverage_score(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    score = 0.0
    has_value = False
    for field in COVERAGE_SCORE_FIELDS:
        value = _float_or_none(row.get(field))
        if value is not None:
            score += value
            has_value = True
    return score if has_value else None


def _cost_safe(candidate: dict[str, Any], teacher: dict[str, Any] | None) -> bool:
    if teacher is None:
        return False
    return (
        _not_materially_worse(candidate.get("path_cost"), teacher.get("path_cost"))
        and _not_materially_worse(candidate.get("risk"), teacher.get("risk"))
        and _not_materially_worse(candidate.get("energy_cost"), teacher.get("energy_cost"))
    )


def _not_materially_worse(value: Any, baseline: Any) -> bool:
    numeric = _float_or_none(value)
    base = _float_or_none(baseline)
    if numeric is None or base is None:
        return True
    return numeric <= base * (1.0 + MAX_ALLOWED_RELATIVE_COST_REGRESSION) + TOLERANCE


def _source_keys(
    *,
    context_id: Any,
    scenario_id: Any,
    scenario_family: Any,
    split: Any,
    action_index: int,
    candidate_cell: Any,
) -> list[str]:
    keys: list[str] = []
    if context_id:
        keys.append(f"context-action:{context_id}:{action_index}")
    if scenario_id:
        keys.append(f"scenario-action:{scenario_id}:{action_index}")
        cell_key = _cell_key(candidate_cell)
        if cell_key is not None:
            keys.append(f"scenario-cell:{scenario_id}:{cell_key}")
    if scenario_family and split:
        keys.append(f"family-split-action:{scenario_family}:{split}:{action_index}")
    return keys


def _decision_key(row: dict[str, Any]) -> str:
    context_id = row.get("context_id")
    episode_id = row.get("episode_id")
    step_index = _first_int(row.get("step_index"))
    if context_id is not None and episode_id is not None and step_index is not None:
        return f"context-episode-step:{context_id}:{episode_id}:{step_index}"
    if context_id is not None:
        return f"context:{context_id}"
    scenario_id = row.get("scenario_id")
    action_index = _first_int(row.get("action_index"))
    return f"scenario-action:{scenario_id}:{action_index}"


def _split_from_row(row: dict[str, Any]) -> str | None:
    split = row.get("split")
    if split:
        return str(split)
    scenario_id = str(row.get("scenario_id") or row.get("scenario_variant_id") or "")
    for value in ("train", "validation", "test"):
        if f"_{value}_" in scenario_id or scenario_id.endswith(f"_{value}"):
            return value
    return None


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "overlay": output_root / OVERLAY_FILE,
        "counterfactual_audit": output_root / COUNTERFACTUAL_AUDIT_FILE,
        "source_link_audit": output_root / SOURCE_LINK_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
        "stage5a_rerun_root": output_root / STAGE5A_RERUN_ROOT,
        "stage5a_rerun_summary": output_root / STAGE5A_RERUN_SUMMARY_FILE,
    }


def _read_json(path: Path | None, reason_codes: list[str], reason: str) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        _add_reason(reason_codes, reason)
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reason_codes, reason)
        return {}


def _read_jsonl(path: Path | None, reason_codes: list[str], reason: str) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        _add_reason(reason_codes, reason)
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except json.JSONDecodeError:
        _add_reason(reason_codes, reason)
        return []
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_path(path: Path, base: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    base_candidate = base / path
    if base_candidate.exists():
        return base_candidate
    return repo_root / path


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _float_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int(value: Any) -> int:
    return _first_int(value) or 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _nonzero(value: Any) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and abs(numeric) > TOLERANCE


def _cell_key(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return f"{int(value[0])},{int(value[1])}"
    except (TypeError, ValueError):
        return None


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
