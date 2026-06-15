from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
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
    from run_candidate_level_coverage_materialization import (
        run_candidate_level_coverage_materialization,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_candidate_level_coverage_materialization import (
        run_candidate_level_coverage_materialization,
    )


SUMMARY_SCHEMA_VERSION = "policy-differentiating-counterfactual-coverage-rollouts-summary/v1"
COUNTERFACTUAL_ROW_SCHEMA_VERSION = "policy-differentiating-counterfactual-coverage-row/v1"
SOURCE_LINK_SCHEMA_VERSION = "policy-differentiating-counterfactual-source-link-audit/v1"
GAP_REPORT_SCHEMA_VERSION = "policy-differentiating-counterfactual-family-action-gap-report/v1"

POLICY_MARGIN_SUMMARY_FILE = "policy-coverage-opportunity-margin-audit-summary.json"
POLICY_ACTION_AUDIT_FILE = "policy-coverage-action-level-audit.jsonl"
STAGE5A1_SUMMARY_FILE = "candidate-level-coverage-materialization-summary.json"
STAGE5A1_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
STAGE5A1_SOURCE_LINK_FILE = "source-link-audit.json"
COVERAGE_DRIVEN_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
COVERAGE_SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
COVERAGE_PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
REWARD_REFINEMENT_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"
QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE = "quasi-real-map-path-feedback-summary.json"
QUASI_REAL_SAFE_ALTERNATIVE_SUMMARY_FILE = "quasi-real-safe-alternative-opportunity-summary.json"

SUMMARY_FILE = "policy-differentiating-counterfactual-coverage-rollouts-summary.json"
COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
CANDIDATE_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
SOURCE_LINK_AUDIT_FILE = "source-link-audit.json"
FAMILY_ACTION_GAP_REPORT_FILE = "family-action-gap-report.json"
STAGE5A1_RERUN_SUMMARY_FILE = "stage5a1-rerun-summary.json"
REPORT_FILE = "policy-differentiating-counterfactual-coverage-rollouts-report.md"
STAGE5A1_RERUN_ROOT = "stage5a1-rerun"
STAGE5A1_AUGMENTED_INPUT_ROOT = "stage5a1-augmented-quasi-real-input"

DEFAULT_POLICY_MARGIN_ROOT = "outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1"
DEFAULT_CANDIDATE_MATERIALIZATION_ROOT = "outputs/path_feedback_batch_candidate_level_coverage_materialization_v1"
DEFAULT_COVERAGE_DRIVEN_ROOT = "outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1"
DEFAULT_REWARD_REFINEMENT_ROOT = "outputs/path_feedback_batch_coverage_aware_reward_refinement_v1"
DEFAULT_COVERAGE_SIGNAL_ROOT = "outputs/path_feedback_batch_exploration_coverage_signal_audit_v1"
DEFAULT_COVERAGE_PERFORMANCE_ROOT = "outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1"
DEFAULT_QUASI_REAL_ROOT = "outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1"

TOLERANCE = 1.0e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate policy-differentiating counterfactual coverage rollouts."
    )
    parser.add_argument("--policy-margin-root", default=DEFAULT_POLICY_MARGIN_ROOT)
    parser.add_argument("--candidate-materialization-root", default=DEFAULT_CANDIDATE_MATERIALIZATION_ROOT)
    parser.add_argument("--coverage-driven-root", default=DEFAULT_COVERAGE_DRIVEN_ROOT)
    parser.add_argument("--reward-refinement-root", default=DEFAULT_REWARD_REFINEMENT_ROOT)
    parser.add_argument("--coverage-signal-root", default=DEFAULT_COVERAGE_SIGNAL_ROOT)
    parser.add_argument("--coverage-performance-root", default=DEFAULT_COVERAGE_PERFORMANCE_ROOT)
    parser.add_argument("--quasi-real-root", default=DEFAULT_QUASI_REAL_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_policy_differentiating_counterfactual_coverage_rollouts(
        policy_margin_root=_resolve_path(Path(args.policy_margin_root), repo_root, repo_root),
        candidate_materialization_root=_resolve_path(
            Path(args.candidate_materialization_root),
            repo_root,
            repo_root,
        ),
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
                "counterfactual_coverage_candidate_count": summary[
                    "counterfactual_coverage_candidate_count"
                ],
                "safe_better_than_teacher_candidate_count": summary[
                    "safe_better_than_teacher_candidate_count"
                ],
                "performance_claimed": summary["performance_claimed"],
                "runs_new_ppo_update": summary["runs_new_ppo_update"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_policy_differentiating_counterfactual_coverage_rollouts(
    *,
    policy_margin_root: Path,
    candidate_materialization_root: Path,
    coverage_driven_root: Path,
    reward_refinement_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    quasi_real_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    policy_margin_root = Path(policy_margin_root)
    candidate_materialization_root = Path(candidate_materialization_root)
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
    stage5a1_summary = _read_json(
        candidate_materialization_root / STAGE5A1_SUMMARY_FILE,
        input_reasons,
        "candidate_materialization_summary_missing",
    )
    stage5a1_overlay_rows = _read_jsonl(
        candidate_materialization_root / STAGE5A1_OVERLAY_FILE,
        input_reasons,
        "candidate_materialization_overlay_missing",
    )
    stage5a1_source_link = _read_json(
        candidate_materialization_root / STAGE5A1_SOURCE_LINK_FILE,
        input_reasons,
        "candidate_materialization_source_link_missing",
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

    candidate_index, source_stats = _index_quasi_real_candidates(
        quasi_real_summary,
        quasi_real_root / QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE,
    )
    sidecar_cache: dict[str, dict[str, Any]] = {}
    previous_state = _previous_coverage_state_by_decision(
        action_rows=action_rows,
        candidate_index=candidate_index,
    )
    target_rows = _target_missing_counterfactual_rows(
        action_rows=action_rows,
        stage5a1_overlay_rows=stage5a1_overlay_rows,
    )
    counterfactual_rows = _counterfactual_rows(
        target_rows=target_rows,
        candidate_index=candidate_index,
        previous_state=previous_state,
        quasi_real_root=quasi_real_root,
        sidecar_cache=sidecar_cache,
    )

    augmented_root = paths["stage5a1_augmented_input_root"]
    _write_augmented_quasi_real_summary(
        quasi_real_summary=quasi_real_summary,
        quasi_real_safe_summary=quasi_real_safe_summary,
        counterfactual_rows=counterfactual_rows,
        augmented_root=augmented_root,
    )
    stage5a1_rerun_summary = run_candidate_level_coverage_materialization(
        policy_margin_root=policy_margin_root,
        coverage_driven_root=coverage_driven_root,
        reward_refinement_root=reward_refinement_root,
        coverage_signal_root=coverage_signal_root,
        coverage_performance_root=coverage_performance_root,
        quasi_real_root=augmented_root,
        output_root=paths["stage5a1_rerun_root"],
        repo_root=repo_root,
    )
    _write_json(paths["stage5a1_rerun_summary"], stage5a1_rerun_summary)
    rerun_overlay = Path(stage5a1_rerun_summary.get("candidate_coverage_overlay", ""))
    if rerun_overlay.is_file():
        shutil.copyfile(rerun_overlay, paths["candidate_overlay"])

    source_link_audit = _source_link_audit(
        target_rows=target_rows,
        counterfactual_rows=counterfactual_rows,
        source_stats=source_stats,
        stage5a1_source_link=stage5a1_source_link,
    )
    family_action_gap_report = _family_action_gap_report(counterfactual_rows)
    controlled_regression_count = _controlled_regression_count(
        action_rows,
        policy_margin_summary,
        coverage_driven_summary,
        reward_summary,
        coverage_signal_summary,
    )
    fallback_gain_contamination_count = _fallback_gain_contamination_count(
        counterfactual_rows,
        policy_margin_summary,
        coverage_driven_summary,
        reward_summary,
        coverage_signal_summary,
    )
    counts = _summary_counts(
        target_rows=target_rows,
        counterfactual_rows=counterfactual_rows,
        stage5a1_rerun_summary=stage5a1_rerun_summary,
    )
    reason_codes = _reason_codes(
        input_reasons=input_reasons,
        counts=counts,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
        coverage_signal_summary=coverage_signal_summary,
        reward_summary=reward_summary,
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

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "policy_differentiating_counterfactual_coverage_rollouts_status": status,
        "policy_margin_root": str(policy_margin_root),
        "candidate_materialization_root": str(candidate_materialization_root),
        "coverage_driven_root": str(coverage_driven_root),
        "reward_refinement_root": str(reward_refinement_root),
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "quasi_real_root": str(quasi_real_root),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "counterfactual_coverage_rollouts": str(paths["counterfactual_rows"]),
        "candidate_coverage_overlay": str(paths["candidate_overlay"]),
        "source_link_audit": str(paths["source_link_audit"]),
        "family_action_gap_report": str(paths["family_action_gap_report"]),
        "stage5a1_rerun_root": str(paths["stage5a1_rerun_root"]),
        "stage5a1_rerun_summary": str(paths["stage5a1_rerun_summary"]),
        "report": str(paths["report"]),
        "input_missing_candidate_count": counts["input_missing_candidate_count"],
        "counterfactual_coverage_candidate_count": counts[
            "counterfactual_coverage_candidate_count"
        ],
        "candidate_expected_coverage_nonzero_count": counts[
            "candidate_expected_coverage_nonzero_count"
        ],
        "candidate_information_gain_nonzero_count": counts[
            "candidate_information_gain_nonzero_count"
        ],
        "candidate_value_nonzero_count": counts["candidate_value_nonzero_count"],
        "safe_better_than_teacher_candidate_count": counts[
            "safe_better_than_teacher_candidate_count"
        ],
        "safe_better_than_teacher_family_count": counts[
            "safe_better_than_teacher_family_count"
        ],
        "missing_counterfactual_source_count": counts["missing_counterfactual_source_count"],
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "controlled_regression_count": controlled_regression_count,
        "stage5a1_rerun_status": stage5a1_rerun_summary.get("status"),
        "stage5a_overlay_rerun_status": stage5a1_rerun_summary.get("stage5a_overlay_rerun_status"),
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
        "stage5a1_input_status": stage5a1_summary.get("status"),
        "source_stats": source_stats,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_jsonl(paths["counterfactual_rows"], counterfactual_rows)
    _write_json(paths["source_link_audit"], source_link_audit)
    _write_json(paths["family_action_gap_report"], family_action_gap_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(
        _render_report(summary, source_link_audit, family_action_gap_report),
        encoding="utf-8",
    )
    return summary


def _target_missing_counterfactual_rows(
    *,
    action_rows: list[dict[str, Any]],
    stage5a1_overlay_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    action_index = {_candidate_row_key(row): row for row in action_rows}
    targets: list[dict[str, Any]] = []
    if stage5a1_overlay_rows:
        for overlay in stage5a1_overlay_rows:
            if overlay.get("coverage_source_available") or overlay.get("is_teacher_action"):
                continue
            action = action_index.get(_candidate_row_key(overlay), overlay)
            targets.append({**action, "_stage5a1_overlay": overlay})
        return targets
    for row in action_rows:
        if row.get("is_teacher_action") or row.get("coverage_rate_delta") is not None:
            continue
        targets.append(row)
    return targets


def _counterfactual_rows(
    *,
    target_rows: list[dict[str, Any]],
    candidate_index: dict[str, dict[str, Any]],
    previous_state: dict[str, set[tuple[int, int]]],
    quasi_real_root: Path,
    sidecar_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in target_rows:
        source = _source_for_action_row(row, candidate_index)
        sidecar = _sidecar_for_source(row, source, quasi_real_root, sidecar_cache)
        missing: list[str] = []
        if source is None:
            _add_reason(missing, "candidate_source_missing")
        route_cells = _route_cells_from_candidate(source or {})
        if not route_cells:
            _add_reason(missing, "candidate_route_cells_missing")
        passable_cells = _passable_cells(sidecar)
        if sidecar is None:
            _add_reason(missing, "sidecar_missing")
        elif not passable_cells:
            _add_reason(missing, "sidecar_passable_mask_missing")
        if bool(row.get("fallback_like")) or bool((source or {}).get("open_grid_fallback_used")):
            _add_reason(missing, "fallback_like_candidate_excluded")

        if missing:
            rows.append(_missing_counterfactual_row(row, source, missing))
            continue

        assert source is not None
        assert sidecar is not None
        previous_cells = previous_state.get(_decision_key(row), set())
        passable_route_cells = {cell for cell in route_cells if cell in passable_cells}
        new_cells = passable_route_cells - previous_cells
        universe = max(len(passable_cells), 1)
        expected_delta = len(new_cells) / universe
        information_gain = _information_gain(new_cells, sidecar) / universe
        utility = _first_float(source.get("utility"), row.get("candidate_utility")) or 0.0
        value = expected_delta * utility
        rows.append(
            _base_counterfactual_row(row, source)
            | {
                "coverage_source_available": True,
                "expected_coverage_rate_delta": expected_delta,
                "expected_new_coverage_area": expected_delta,
                "new_coverage_cell_count": len(new_cells),
                "route_cell_count": len(passable_route_cells),
                "coverage_universe_cell_count": len(passable_cells),
                "information_gain": information_gain,
                "value": value,
                "valuable_coverage_proxy": value,
                "path_cost": _first_float(source.get("path_cost"), row.get("path_cost")),
                "risk": _first_float(source.get("risk"), row.get("risk")),
                "energy_cost": _first_float(source.get("energy_cost"), row.get("energy_cost")),
                "source_path": source.get("source_path"),
                "source_scenario_id": source.get("scenario_id"),
                "source_candidate_cell": source.get("candidate_cell"),
                "sidecar_path": str(_sidecar_path_for_source(row, source, quasi_real_root)),
                "match_method": "counterfactual_candidate_expanded_cells_sidecar",
                "source_confidence": 0.75,
                "source_execution_type": "counterfactual_candidate",
                "missing_reason_codes": [],
            }
        )
    return rows


def _previous_coverage_state_by_decision(
    *,
    action_rows: list[dict[str, Any]],
    candidate_index: dict[str, dict[str, Any]],
) -> dict[str, set[tuple[int, int]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        groups[_decision_key(row)].append(row)
    grouped_rows = sorted(
        groups.values(),
        key=lambda rows: (
            str(rows[0].get("episode_id") or ""),
            _first_int(rows[0].get("step_index")) or 0,
            str(rows[0].get("context_id") or ""),
        ),
    )
    covered_by_episode: dict[str, set[tuple[int, int]]] = defaultdict(set)
    previous_by_decision: dict[str, set[tuple[int, int]]] = {}
    for rows in grouped_rows:
        episode_id = str(rows[0].get("episode_id") or "")
        decision_key = _decision_key(rows[0])
        previous_by_decision[decision_key] = set(covered_by_episode[episode_id])
        executed = next((row for row in rows if row.get("counterfactual_coverage_observed")), None)
        if executed is None:
            executed = next((row for row in rows if row.get("is_pre_improvement_selected_action")), None)
        if executed is None:
            executed = next((row for row in rows if row.get("is_teacher_action")), None)
        if executed is None:
            continue
        source = _source_for_action_row(executed, candidate_index)
        covered_by_episode[episode_id].update(_route_cells_from_candidate(source or {}))
    return previous_by_decision


def _write_augmented_quasi_real_summary(
    *,
    quasi_real_summary: dict[str, Any],
    quasi_real_safe_summary: dict[str, Any],
    counterfactual_rows: list[dict[str, Any]],
    augmented_root: Path,
) -> None:
    augmented_root.mkdir(parents=True, exist_ok=True)
    augmented_summary = copy.deepcopy(quasi_real_summary)
    overlay_by_candidate = {
        _candidate_source_key(
            {
                "scenario_id": row.get("source_scenario_id") or row.get("scenario_id"),
                "action_index": row.get("action_index"),
                "candidate_cell": row.get("source_candidate_cell") or row.get("candidate_cell"),
            }
        ): row
        for row in counterfactual_rows
        if row.get("coverage_source_available")
    }
    for scenario in augmented_summary.get("scenarios", []) if isinstance(augmented_summary.get("scenarios"), list) else []:
        path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
        candidates = path_feedback.get("candidates") if isinstance(path_feedback.get("candidates"), list) else []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            key = _candidate_source_key(
                {
                    "scenario_id": scenario.get("scenario_id"),
                    "action_index": candidate.get("action_index"),
                    "candidate_cell": candidate.get("cell") or candidate.get("candidate_cell"),
                }
            )
            overlay = overlay_by_candidate.get(key)
            if not overlay:
                continue
            for field in (
                "expected_coverage_rate_delta",
                "expected_new_coverage_area",
                "information_gain",
                "value",
                "valuable_coverage_proxy",
            ):
                candidate[field] = overlay.get(field)
            candidate["counterfactual_coverage_source"] = {
                "schema_version": COUNTERFACTUAL_ROW_SCHEMA_VERSION,
                "source_path": overlay.get("source_path"),
                "sidecar_path": overlay.get("sidecar_path"),
                "match_method": overlay.get("match_method"),
                "source_confidence": overlay.get("source_confidence"),
                "new_coverage_cell_count": overlay.get("new_coverage_cell_count"),
                "coverage_universe_cell_count": overlay.get("coverage_universe_cell_count"),
            }
    _write_json(augmented_root / QUASI_REAL_PATH_FEEDBACK_SUMMARY_FILE, augmented_summary)
    if quasi_real_safe_summary:
        _write_json(augmented_root / QUASI_REAL_SAFE_ALTERNATIVE_SUMMARY_FILE, quasi_real_safe_summary)


def _index_quasi_real_candidates(
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
        candidates = (
            scenario.get("path_feedback", {}).get("candidates", [])
            if isinstance(scenario.get("path_feedback"), dict)
            else []
        )
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
                "candidate_cell": candidate.get("candidate_cell") or candidate.get("cell"),
                "source_path": str(source_path),
            }
            stats["candidate_source_count"] += 1
            if _route_cells_from_candidate(row):
                stats["candidate_source_with_route_cells_count"] += 1
            for key in _source_keys(
                context_id=row.get("context_id"),
                scenario_id=scenario_id,
                scenario_family=scenario_family,
                split=split,
                action_index=action_index,
                candidate_cell=row.get("candidate_cell"),
            ):
                index.setdefault(key, row)
    return index, {"scenario_count": scenario_count, **dict(stats)}


def _source_for_action_row(
    row: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
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
        if key in candidate_index:
            return candidate_index[key]
    return None


def _sidecar_for_source(
    row: dict[str, Any],
    source: dict[str, Any] | None,
    quasi_real_root: Path,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    path = _sidecar_path_for_source(row, source, quasi_real_root)
    if not path:
        return None
    key = str(path)
    if key not in cache:
        cache[key] = _read_json(path, [], "sidecar_missing")
    return cache[key] or None


def _sidecar_path_for_source(
    row: dict[str, Any],
    source: dict[str, Any] | None,
    quasi_real_root: Path,
) -> Path:
    scenario_id = (source or {}).get("scenario_id") or row.get("scenario_id")
    return _sidecar_path_for_scenario(scenario_id, quasi_real_root)


def _sidecar_path_for_scenario(scenario_id: Any, quasi_real_root: Path) -> Path:
    return Path(quasi_real_root) / "path_planner_sidecars" / f"{scenario_id}.path-planner-sidecar.json"


def _passable_cells(sidecar: dict[str, Any] | None) -> set[tuple[int, int]]:
    if not sidecar:
        return set()
    mask = sidecar.get("passable_mask")
    if not isinstance(mask, list):
        return set()
    cells: set[tuple[int, int]] = set()
    for y, row in enumerate(mask):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            if bool(value):
                cells.add((x, y))
    return cells


def _route_cells_from_candidate(candidate: dict[str, Any]) -> set[tuple[int, int]]:
    diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
    candidates = [
        diagnostics.get("route_cells"),
        diagnostics.get("path_cells"),
        diagnostics.get("expanded_cells"),
        candidate.get("route_cells"),
        candidate.get("path_cells"),
    ]
    cells: set[tuple[int, int]] = set()
    for values in candidates:
        if not isinstance(values, list):
            continue
        for value in values:
            cell = _cell_tuple(value)
            if cell is not None:
                cells.add(cell)
        if cells:
            return cells
    return cells


def _information_gain(new_cells: set[tuple[int, int]], sidecar: dict[str, Any]) -> float:
    terrain_layers = sidecar.get("terrain_layers") if isinstance(sidecar.get("terrain_layers"), dict) else {}
    confidence_grid = terrain_layers.get("confidence")
    if not isinstance(confidence_grid, list):
        return float(len(new_cells))
    gain = 0.0
    for x, y in new_cells:
        confidence = _grid_value(confidence_grid, x, y)
        if confidence is None:
            gain += 1.0
        else:
            gain += max(0.0, 1.0 - confidence)
    return gain


def _source_link_audit(
    *,
    target_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    source_stats: dict[str, Any],
    stage5a1_source_link: dict[str, Any],
) -> dict[str, Any]:
    match_methods = Counter(str(row.get("match_method")) for row in counterfactual_rows)
    missing_by_family: Counter[str] = Counter()
    missing_by_action: Counter[str] = Counter()
    missing_by_reason: Counter[str] = Counter()
    for row in counterfactual_rows:
        if row.get("coverage_source_available"):
            continue
        missing_by_family[str(row.get("scenario_family"))] += 1
        missing_by_action[str(row.get("action_index"))] += 1
        for reason in row.get("missing_reason_codes", []):
            missing_by_reason[str(reason)] += 1
    return {
        "schema_version": SOURCE_LINK_SCHEMA_VERSION,
        "input_missing_candidate_count": len(target_rows),
        "counterfactual_row_count": len(counterfactual_rows),
        "counterfactual_coverage_candidate_count": sum(
            1 for row in counterfactual_rows if row.get("coverage_source_available")
        ),
        "missing_counterfactual_source_count": sum(
            1 for row in counterfactual_rows if not row.get("coverage_source_available")
        ),
        "match_method_counts": dict(sorted(match_methods.items())),
        "missing_by_family": dict(sorted(missing_by_family.items())),
        "missing_by_action_index": dict(sorted(missing_by_action.items())),
        "missing_by_reason": dict(sorted(missing_by_reason.items())),
        "quasi_real_source_stats": source_stats,
        "stage5a1_previous_missing_by_family": stage5a1_source_link.get("missing_by_family", {}),
        "stage5a1_previous_missing_by_action_index": stage5a1_source_link.get(
            "missing_by_action_index",
            {},
        ),
    }


def _family_action_gap_report(counterfactual_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family_action: dict[str, dict[str, Any]] = {}
    for row in counterfactual_rows:
        key = f"{row.get('scenario_family')}::{row.get('action_index')}"
        bucket = by_family_action.setdefault(
            key,
            {
                "scenario_family": row.get("scenario_family"),
                "action_index": row.get("action_index"),
                "candidate_count": 0,
                "coverage_source_available_count": 0,
                "missing_counterfactual_source_count": 0,
                "missing_reason_counts": Counter(),
            },
        )
        bucket["candidate_count"] += 1
        if row.get("coverage_source_available"):
            bucket["coverage_source_available_count"] += 1
        else:
            bucket["missing_counterfactual_source_count"] += 1
            for reason in row.get("missing_reason_codes", []):
                bucket["missing_reason_counts"][str(reason)] += 1
    rows = []
    for item in by_family_action.values():
        item = dict(item)
        item["missing_reason_counts"] = dict(sorted(item["missing_reason_counts"].items()))
        rows.append(item)
    rows.sort(key=lambda item: (str(item["scenario_family"]), int(item["action_index"] or 0)))
    return {"schema_version": GAP_REPORT_SCHEMA_VERSION, "rows": rows}


def _summary_counts(
    *,
    target_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    stage5a1_rerun_summary: dict[str, Any],
) -> dict[str, int]:
    available = [row for row in counterfactual_rows if row.get("coverage_source_available")]
    missing = [row for row in counterfactual_rows if not row.get("coverage_source_available")]
    return {
        "input_missing_candidate_count": len(target_rows),
        "counterfactual_coverage_candidate_count": len(available),
        "candidate_expected_coverage_nonzero_count": _int(
            stage5a1_rerun_summary.get("candidate_expected_coverage_nonzero_count")
        ),
        "candidate_information_gain_nonzero_count": _int(
            stage5a1_rerun_summary.get("candidate_information_gain_nonzero_count")
        ),
        "candidate_value_nonzero_count": _int(
            stage5a1_rerun_summary.get("candidate_value_nonzero_count")
        ),
        "safe_better_than_teacher_candidate_count": _int(
            stage5a1_rerun_summary.get("safe_better_than_teacher_candidate_count")
        ),
        "safe_better_than_teacher_family_count": _int(
            stage5a1_rerun_summary.get("safe_better_than_teacher_family_count")
        ),
        "missing_counterfactual_source_count": len(missing),
    }


def _reason_codes(
    *,
    input_reasons: list[str],
    counts: dict[str, int],
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
    coverage_signal_summary: dict[str, Any],
    reward_summary: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for reason in input_reasons:
        _add_reason(reasons, reason)
    if coverage_signal_summary.get("coverage_signal_status") != "passed":
        _add_reason(reasons, "coverage_signal_guardrail_not_passed")
    if reward_summary.get("reward_refinement_status") != "passed":
        _add_reason(reasons, "reward_refinement_guardrail_not_passed")
    if controlled_regression_count > 0:
        _add_reason(reasons, "controlled_regression_present")
    if fallback_gain_contamination_count > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    if counts["input_missing_candidate_count"] <= 0:
        _add_reason(reasons, "no_missing_counterfactual_candidates_to_materialize")
    if counts["missing_counterfactual_source_count"] > 0:
        _add_reason(reasons, "counterfactual_coverage_source_missing")
    if counts["counterfactual_coverage_candidate_count"] <= 0:
        _add_reason(reasons, "no_counterfactual_coverage_generated")
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
        return "fix_counterfactual_coverage_rollout_inputs"
    if any(reason.endswith("_missing") for reason in reason_codes if reason.startswith(("policy_", "coverage_", "reward_", "quasi_", "candidate_"))):
        return "fix_counterfactual_coverage_rollout_inputs"
    if counts["missing_counterfactual_source_count"] > 0 or counts["counterfactual_coverage_candidate_count"] <= 0:
        return "expand_counterfactual_coverage_source_generation"
    if counts["safe_better_than_teacher_candidate_count"] > 0:
        return "rerun_coverage_driven_ppo_with_refined_reward_or_advantage"
    return "collect_more_policy_differentiating_coverage_or_refine_candidate_set"


def _controlled_regression_count(
    action_rows: list[dict[str, Any]],
    *summaries: dict[str, Any],
) -> int:
    row_count = sum(1 for row in action_rows if row.get("controlled_regression_reason_codes"))
    summary_count = sum(_int(summary.get("controlled_regression_count")) for summary in summaries if isinstance(summary, dict))
    return row_count + summary_count


def _fallback_gain_contamination_count(
    counterfactual_rows: list[dict[str, Any]],
    *summaries: dict[str, Any],
) -> int:
    row_count = sum(
        1
        for row in counterfactual_rows
        if row.get("coverage_source_available") and bool(row.get("fallback_like"))
    )
    summary_count = sum(
        _int(summary.get("fallback_coverage_gain_claimed_as_policy_gain_count"))
        + _int(summary.get("fallback_policy_gain_contamination_count"))
        for summary in summaries
        if isinstance(summary, dict)
    )
    return row_count + summary_count


def _missing_counterfactual_row(
    row: dict[str, Any],
    source: dict[str, Any] | None,
    missing_reason_codes: list[str],
) -> dict[str, Any]:
    return _base_counterfactual_row(row, source) | {
        "coverage_source_available": False,
        "expected_coverage_rate_delta": None,
        "expected_new_coverage_area": None,
        "new_coverage_cell_count": None,
        "route_cell_count": None,
        "coverage_universe_cell_count": None,
        "information_gain": None,
        "value": None,
        "valuable_coverage_proxy": None,
        "path_cost": _first_float((source or {}).get("path_cost"), row.get("path_cost")),
        "risk": _first_float((source or {}).get("risk"), row.get("risk")),
        "energy_cost": _first_float((source or {}).get("energy_cost"), row.get("energy_cost")),
        "source_scenario_id": (source or {}).get("scenario_id"),
        "source_candidate_cell": (source or {}).get("candidate_cell"),
        "source_path": (source or {}).get("source_path"),
        "sidecar_path": None,
        "match_method": "counterfactual_source_missing",
        "source_confidence": 0.0,
        "source_execution_type": "missing",
        "missing_reason_codes": missing_reason_codes,
    }


def _base_counterfactual_row(
    row: dict[str, Any],
    source: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": COUNTERFACTUAL_ROW_SCHEMA_VERSION,
        "context_id": row.get("context_id"),
        "episode_id": row.get("episode_id"),
        "step_index": _first_int(row.get("step_index")),
        "scenario_id": row.get("scenario_id"),
        "scenario_family": row.get("scenario_family") or (source or {}).get("scenario_family"),
        "split": _split_from_row(row) or (source or {}).get("split"),
        "action_index": _first_int(row.get("action_index")),
        "candidate_cell": row.get("candidate_cell"),
        "teacher_action_index": _first_int(row.get("teacher_action_index")),
        "is_teacher_action": bool(row.get("is_teacher_action")),
        "action_mask_valid": bool(row.get("action_mask_valid", True)),
        "fallback_like": bool(row.get("fallback_like")) or bool((source or {}).get("open_grid_fallback_used")),
        "guard_rejected": bool(row.get("guard_rejected")),
        "policy_action_accepted": bool(row.get("policy_action_accepted", True)),
        "controlled_regression_reason_codes": _string_list(
            row.get("controlled_regression_reason_codes")
        ),
    }


def _render_report(
    summary: dict[str, Any],
    source_link_audit: dict[str, Any],
    family_action_gap_report: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Policy-Differentiating Counterfactual Coverage Rollouts v1",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- input_missing_candidate_count: `{summary['input_missing_candidate_count']}`",
            f"- counterfactual_coverage_candidate_count: `{summary['counterfactual_coverage_candidate_count']}`",
            f"- missing_counterfactual_source_count: `{summary['missing_counterfactual_source_count']}`",
            f"- safe_better_than_teacher_candidate_count: `{summary['safe_better_than_teacher_candidate_count']}`",
            f"- stage5a1_rerun_status: `{summary['stage5a1_rerun_status']}`",
            f"- stage5a_overlay_rerun_status: `{summary['stage5a_overlay_rerun_status']}`",
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## Source Link Audit",
            "",
            f"- match_method_counts: `{source_link_audit.get('match_method_counts', {})}`",
            f"- missing_by_family: `{source_link_audit.get('missing_by_family', {})}`",
            f"- missing_by_action_index: `{source_link_audit.get('missing_by_action_index', {})}`",
            "",
            "## Family / Action Gaps",
            "",
            *[
                f"- {row['scenario_family']} action {row['action_index']}: "
                f"{row['missing_counterfactual_source_count']} missing / {row['candidate_count']} candidates"
                for row in family_action_gap_report.get("rows", [])[:30]
            ],
            "",
            "## Guardrails",
            "",
            "- This stage did not run PPO.",
            "- It did not publish or replace any checkpoint.",
            "- It does not claim exploration coverage performance improvement.",
            "- It does not connect a real executor or relax the hard guard.",
        ]
    )


def _interpretation(summary: dict[str, Any]) -> str:
    if summary["next_required_change"] == "rerun_coverage_driven_ppo_with_refined_reward_or_advantage":
        return (
            "Counterfactual non-teacher coverage is available and exposes safe better-than-teacher "
            "candidates. This routes the pipeline back to coverage-driven PPO refinement; it is not "
            "a performance claim."
        )
    if summary["next_required_change"] == "expand_counterfactual_coverage_source_generation":
        return (
            "The stage still lacks enough candidate route or sidecar evidence to compute real "
            "counterfactual coverage. Missing values stayed missing instead of being converted to zero."
        )
    return "Counterfactual coverage inputs or candidate opportunities need more refinement before PPO."


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
        cell_key = _cell_key(candidate_cell)
        if cell_key is not None:
            keys.append(f"family-split-action-cell:{scenario_family}:{split}:{action_index}:{cell_key}")
        keys.append(f"family-split-action:{scenario_family}:{split}:{action_index}")
    return keys


def _candidate_row_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("context_id")),
            str(row.get("episode_id")),
            str(_first_int(row.get("step_index"))),
            str(row.get("scenario_id")),
            str(_first_int(row.get("action_index"))),
            str(_cell_key(row.get("candidate_cell"))),
        ]
    )


def _candidate_source_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("scenario_id")),
            str(_first_int(row.get("action_index"))),
            str(_cell_key(row.get("candidate_cell") or row.get("cell"))),
        ]
    )


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
        "counterfactual_rows": output_root / COUNTERFACTUAL_FILE,
        "candidate_overlay": output_root / CANDIDATE_OVERLAY_FILE,
        "source_link_audit": output_root / SOURCE_LINK_AUDIT_FILE,
        "family_action_gap_report": output_root / FAMILY_ACTION_GAP_REPORT_FILE,
        "stage5a1_rerun_root": output_root / STAGE5A1_RERUN_ROOT,
        "stage5a1_augmented_input_root": output_root / STAGE5A1_AUGMENTED_INPUT_ROOT,
        "stage5a1_rerun_summary": output_root / STAGE5A1_RERUN_SUMMARY_FILE,
        "report": output_root / REPORT_FILE,
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


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


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _cell_key(value: Any) -> str | None:
    cell = _cell_tuple(value)
    if cell is None:
        return None
    return f"{cell[0]},{cell[1]}"


def _grid_value(grid: Any, x: int, y: int) -> float | None:
    try:
        return _float_or_none(grid[y][x])
    except (IndexError, TypeError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
