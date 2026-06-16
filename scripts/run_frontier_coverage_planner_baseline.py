from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from frontier_coverage_planner_common import (
        coverage_rate,
        frontier_cells,
        score_frontier_candidate,
        select_frontier_candidate,
    )
    from global_99_coverage_contract import (
        ConfigError,
        TOLERANCE,
        coverage_footprint,
        load_global_99_config,
        resolve_path,
        scenario_geometry,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.frontier_coverage_planner_common import (
        coverage_rate,
        frontier_cells,
        score_frontier_candidate,
        select_frontier_candidate,
    )
    from scripts.global_99_coverage_contract import (
        ConfigError,
        TOLERANCE,
        coverage_footprint,
        load_global_99_config,
        resolve_path,
        scenario_geometry,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )


CONFIG_SCHEMA_VERSION = "frontier-coverage-planner-baseline-config/v1"
SUMMARY_SCHEMA_VERSION = "frontier-coverage-planner-baseline-summary/v1"
MANIFEST_SCHEMA_VERSION = "frontier-coverage-planner-baseline-manifest/v1"
PLAN_ROW_SCHEMA_VERSION = "frontier-coverage-plan-row/v1"
LEDGER_ROW_SCHEMA_VERSION = "frontier-coverage-ledger-row/v1"
BUDGET_AUDIT_SCHEMA_VERSION = "frontier-coverage-budget-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "frontier-coverage-rejection-report/v1"

DEFAULT_CONFIG = "configs/frontier_coverage_planner_baseline_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_frontier_coverage_planner_baseline_v1"

SUMMARY_FILE = "frontier-coverage-planner-baseline-summary.json"
MANIFEST_FILE = "frontier-coverage-planner-baseline-manifest.json"
PLAN_FILE = "frontier-coverage-plan.jsonl"
LEDGER_FILE = "frontier-coverage-ledger.jsonl"
BUDGET_AUDIT_FILE = "frontier-coverage-budget-audit.json"
REJECTION_REPORT_FILE = "frontier-coverage-rejection-report.json"
REPORT_FILE = "frontier-coverage-planner-baseline-report.md"

PASS_NEXT_REQUIRED_CHANGE = "coverage_memory_replanning_loop"
FAIL_NEXT_REQUIRED_CHANGE = "fix_frontier_coverage_planner_baseline"

Cell = tuple[int, int]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Frontier Coverage Planner Baseline v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_frontier_coverage_planner_baseline(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "achieved_coverage_rate": summary["achieved_coverage_rate"],
                "coverage_target_met": summary["coverage_target_met"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_frontier_coverage_planner_baseline(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path, repo_root)
    source_config_path = _resolve_config_reference(config["source_global_99_config"], config_path, repo_root)
    source_config = load_global_99_config(source_config_path)
    target = float(config["target_coverage_rate"])
    path_budget_m = float(config["path_budget_m"])
    scenario_results = [
        _run_frontier_scenario(
            scenario,
            target_coverage_rate=target,
            path_budget_m=path_budget_m,
            coverage_radius_cells=int(config["coverage_radius_cells"]),
            frontier_step_limit=int(config["frontier_step_limit"]),
            revisit_penalty_weight=float(config["revisit_penalty_weight"]),
            new_coverage_weight=float(config["new_coverage_weight"]),
        )
        for scenario in source_config["scenarios"]
    ]

    plan_rows = [row for result in scenario_results for row in result["plan_rows"]]
    ledger_rows = [row for result in scenario_results for row in result["ledger_rows"]]
    total_denominator = sum(result["reachable_safe_cell_count"] for result in scenario_results)
    total_covered = sum(result["covered_reachable_safe_cell_count"] for result in scenario_results)
    achieved = (total_covered / total_denominator) if total_denominator else 0.0
    coverage_target_met = (
        total_denominator > 0
        and achieved + TOLERANCE >= target
        and all(result["coverage_target_met"] for result in scenario_results)
    )
    reason_codes = _summary_reason_codes(scenario_results, coverage_target_met, total_denominator)
    status = "passed" if not reason_codes else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE if status == "passed" else FAIL_NEXT_REQUIRED_CHANGE
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "plan": output_root / PLAN_FILE,
        "ledger": output_root / LEDGER_FILE,
        "budget_audit": output_root / BUDGET_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }
    infeasible_reason_codes = unique_sorted(
        reason for result in scenario_results for reason in result["infeasible_reason_codes"]
    )
    planned_path_cost_m = sum(result["planned_path_cost_m"] for result in scenario_results)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "config": str(config_path),
        "source_global_99_config": str(source_config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "frontier_coverage_plan": str(paths["plan"]),
        "frontier_coverage_ledger": str(paths["ledger"]),
        "budget_audit": str(paths["budget_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "target_coverage_rate": target,
        "achieved_coverage_rate": achieved,
        "coverage_target_met": coverage_target_met,
        "reachable_safe_cell_count": total_denominator,
        "covered_reachable_safe_cell_count": total_covered,
        "generated_coverage_event_count": len(plan_rows),
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_m": path_budget_m,
        "path_budget_exhausted": any(result["path_budget_exhausted"] for result in scenario_results),
        "frontier_plan_complete": status == "passed",
        "scenario_count": len(scenario_results),
        "passed_scenario_count": sum(1 for result in scenario_results if result["status"] == "passed"),
        "failed_scenario_count": sum(1 for result in scenario_results if result["status"] == "failed"),
        "infeasible_reason_codes": infeasible_reason_codes,
        "coverage_radius_cells": config["coverage_radius_cells"],
        "frontier_step_limit": config["frontier_step_limit"],
        "revisit_penalty_weight": config["revisit_penalty_weight"],
        "new_coverage_weight": config["new_coverage_weight"],
        "next_required_change": next_required_change,
        "scenario_summaries": [_public_scenario_summary(result) for result in scenario_results],
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "checkpoint_publication_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "runs_new_ppo_update": False,
        "modifies_network": False,
        "modifies_action_space": False,
        "modifies_default_astar": False,
        "ackermann_feasible_trajectory_claimed": False,
        "performance_claimed": False,
        "real_world_performance_claimed": False,
        "uses_ppo_policy": False,
        "uses_default_policy": False,
        "uses_path_planner": False,
        "uses_npz_or_sidecar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    manifest = _manifest(config_path, source_config_path, output_root, paths, summary)
    budget_audit = _budget_audit(config, scenario_results, planned_path_cost_m)
    rejection_report = _rejection_report(reason_codes, scenario_results)

    write_jsonl(paths["plan"], plan_rows)
    write_jsonl(paths["ledger"], ledger_rows)
    write_json(paths["budget_audit"], budget_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _run_frontier_scenario(
    scenario: dict[str, Any],
    *,
    target_coverage_rate: float,
    path_budget_m: float,
    coverage_radius_cells: int,
    frontier_step_limit: int,
    revisit_penalty_weight: float,
    new_coverage_weight: float,
) -> dict[str, Any]:
    geometry = scenario_geometry(scenario)
    scenario_id = geometry["scenario_id"]
    width = geometry["width"]
    height = geometry["height"]
    resolution_m = geometry["resolution_m"]
    navigation_cells: set[Cell] = geometry["navigation_cells"]
    target_cells: set[Cell] = geometry["reachable_safe_cells"]
    start: Cell = geometry["start"]
    plan_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    path_budget_exhausted = False
    frontier_exhausted = False
    frontier_step_limit_exhausted = False
    frontier_unreachable = False
    planned_path_cost_m = 0.0
    current_cell = start

    coverage_map_cells = coverage_footprint(start, width, height, coverage_radius_cells) & navigation_cells
    covered_target_cells = coverage_map_cells & target_cells
    _append_plan_and_ledger_row(
        plan_rows=plan_rows,
        ledger_rows=ledger_rows,
        scenario_id=scenario_id,
        event_index=0,
        event_id=f"{scenario_id}-initial-coverage",
        selected_waypoint=start,
        path=[start] if start in navigation_cells else [],
        event_cells=coverage_map_cells,
        counted=True,
        attempted_path_cost_m=0.0,
        planned_path_cost_m=0.0,
        path_budget_m=path_budget_m,
        path_cost_m=0.0,
        score=0.0,
        revisited_path_cell_count=0,
        new_target_cell_count=len(covered_target_cells),
        target_cells=target_cells,
        covered_target_cells=covered_target_cells,
        frontier_cluster_count=0,
        choice_reason="initial_coverage",
    )

    step_index = 1
    while coverage_rate(covered_target_cells, target_cells) + TOLERANCE < target_coverage_rate:
        if step_index > frontier_step_limit:
            frontier_step_limit_exhausted = True
            break
        frontier = frontier_cells(coverage_map_cells, navigation_cells)
        if not frontier:
            frontier_exhausted = True
            break
        candidate = select_frontier_candidate(
            current_cell=current_cell,
            frontier=frontier,
            navigation_cells=navigation_cells,
            target_cells=target_cells,
            covered_target_cells=covered_target_cells,
            coverage_map_cells=coverage_map_cells,
            width=width,
            height=height,
            resolution_m=resolution_m,
            coverage_radius_cells=coverage_radius_cells,
            revisit_penalty_weight=revisit_penalty_weight,
            new_coverage_weight=new_coverage_weight,
        )
        if candidate is None:
            frontier_unreachable = True
            break
        attempted_path_cost_m = planned_path_cost_m + candidate["path_cost_m"]
        counted = attempted_path_cost_m <= path_budget_m + TOLERANCE
        if not counted:
            path_budget_exhausted = True
            _append_plan_and_ledger_row(
                plan_rows=plan_rows,
                ledger_rows=ledger_rows,
                scenario_id=scenario_id,
                event_index=step_index,
                event_id=f"{scenario_id}-frontier-{step_index:04d}",
                selected_waypoint=candidate["selected_waypoint"],
                path=candidate["path"],
                event_cells=candidate["event_cells"],
                counted=False,
                attempted_path_cost_m=attempted_path_cost_m,
                planned_path_cost_m=planned_path_cost_m,
                path_budget_m=path_budget_m,
                path_cost_m=candidate["path_cost_m"],
                score=candidate["score"],
                revisited_path_cell_count=candidate["revisited_path_cell_count"],
                new_target_cell_count=0,
                target_cells=target_cells,
                covered_target_cells=covered_target_cells,
                frontier_cluster_count=candidate["frontier_cluster_count"],
                choice_reason="path_budget_exhausted",
            )
            break

        before_target_count = len(covered_target_cells)
        planned_path_cost_m = attempted_path_cost_m
        current_cell = candidate["selected_waypoint"]
        coverage_map_cells |= candidate["event_cells"]
        covered_target_cells = coverage_map_cells & target_cells
        new_target_cell_count = len(covered_target_cells) - before_target_count
        _append_plan_and_ledger_row(
            plan_rows=plan_rows,
            ledger_rows=ledger_rows,
            scenario_id=scenario_id,
            event_index=step_index,
            event_id=f"{scenario_id}-frontier-{step_index:04d}",
            selected_waypoint=candidate["selected_waypoint"],
            path=candidate["path"],
            event_cells=candidate["event_cells"],
            counted=True,
            attempted_path_cost_m=attempted_path_cost_m,
            planned_path_cost_m=planned_path_cost_m,
            path_budget_m=path_budget_m,
            path_cost_m=candidate["path_cost_m"],
            score=candidate["score"],
            revisited_path_cell_count=candidate["revisited_path_cell_count"],
            new_target_cell_count=new_target_cell_count,
            target_cells=target_cells,
            covered_target_cells=covered_target_cells,
            frontier_cluster_count=candidate["frontier_cluster_count"],
            choice_reason="frontier_score",
        )
        step_index += 1

    target_met = bool(target_cells) and coverage_rate(covered_target_cells, target_cells) + TOLERANCE >= target_coverage_rate
    if not target_cells:
        reason_codes.append("coverage_denominator_invalid")
    if not target_met and target_cells:
        reason_codes.append("coverage_target_not_met")
    if path_budget_exhausted:
        reason_codes.append("insufficient_budget")
    if frontier_step_limit_exhausted:
        reason_codes.append("frontier_step_limit_exhausted")
    if frontier_exhausted:
        reason_codes.append("frontier_exhausted")
    if frontier_unreachable:
        reason_codes.append("frontier_unreachable")

    infeasible_reason_codes: list[str] = []
    if geometry["unsafe_roi_cells"]:
        infeasible_reason_codes.append("unsafe_roi_cells")
    if geometry["unreachable_roi_cells"]:
        infeasible_reason_codes.append("unreachable_roi_cells")
    if path_budget_exhausted:
        infeasible_reason_codes.append("insufficient_budget")
    if not target_cells:
        infeasible_reason_codes.append("coverage_denominator_invalid")
    if target_cells and not target_met:
        infeasible_reason_codes.append("coverage_target_not_met")

    achieved = coverage_rate(covered_target_cells, target_cells)
    return {
        "scenario_id": scenario_id,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": unique_sorted(reason_codes),
        "infeasible_reason_codes": unique_sorted(infeasible_reason_codes),
        "target_coverage_rate": target_coverage_rate,
        "achieved_coverage_rate": achieved,
        "coverage_target_met": target_met,
        "reachable_safe_cell_count": len(target_cells),
        "covered_reachable_safe_cell_count": len(covered_target_cells),
        "generated_coverage_event_count": len(plan_rows),
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_m": path_budget_m,
        "path_budget_exhausted": path_budget_exhausted,
        "frontier_plan_complete": target_met and not reason_codes,
        "frontier_step_limit_exhausted": frontier_step_limit_exhausted,
        "frontier_exhausted": frontier_exhausted,
        "frontier_unreachable": frontier_unreachable,
        "roi_cell_count": len(geometry["roi_cells"]),
        "blocked_roi_cell_count": len(geometry["roi_cells"] & geometry["blocked_cells"]),
        "unsafe_roi_cell_count": len(geometry["roi_cells"] & geometry["unsafe_cells"]),
        "unreachable_roi_cell_count": len(geometry["unreachable_roi_cells"]),
        "plan_rows": plan_rows,
        "ledger_rows": ledger_rows,
    }


def _append_plan_and_ledger_row(
    *,
    plan_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    scenario_id: str,
    event_index: int,
    event_id: str,
    selected_waypoint: Cell,
    path: list[Cell],
    event_cells: set[Cell],
    counted: bool,
    attempted_path_cost_m: float,
    planned_path_cost_m: float,
    path_budget_m: float,
    path_cost_m: float,
    score: float,
    revisited_path_cell_count: int,
    new_target_cell_count: int,
    target_cells: set[Cell],
    covered_target_cells: set[Cell],
    frontier_cluster_count: int,
    choice_reason: str,
) -> None:
    achieved = coverage_rate(covered_target_cells, target_cells)
    plan_rows.append(
        {
            "schema_version": PLAN_ROW_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "event_index": event_index,
            "event_id": event_id,
            "choice_source": "frontier_baseline",
            "choice_reason": choice_reason,
            "selected_waypoint": list(selected_waypoint),
            "path_cell_count": len(path),
            "path_cost_m": path_cost_m,
            "attempted_path_cost_m": attempted_path_cost_m,
            "planned_path_cost_m": planned_path_cost_m,
            "path_budget_m": path_budget_m,
            "counted": counted,
            "frontier_cluster_count": frontier_cluster_count,
            "revisited_path_cell_count": revisited_path_cell_count,
            "new_covered_reachable_safe_cell_count": new_target_cell_count,
            "candidate_score": score,
            "covered_reachable_safe_cell_count": len(covered_target_cells),
            "reachable_safe_cell_count": len(target_cells),
            "achieved_coverage_rate": achieved,
        }
    )
    ledger_rows.append(
        {
            "schema_version": LEDGER_ROW_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "event_index": event_index,
            "event_id": event_id,
            "selected_waypoint": list(selected_waypoint),
            "counted": counted,
            "event_cell_count": len(event_cells),
            "event_reachable_safe_cell_count": len(event_cells & target_cells),
            "new_covered_reachable_safe_cell_count": new_target_cell_count,
            "cumulative_covered_reachable_safe_cell_count": len(covered_target_cells),
            "reachable_safe_cell_count": len(target_cells),
            "achieved_coverage_rate": achieved,
            "path_cost_m": path_cost_m,
            "attempted_path_cost_m": attempted_path_cost_m,
            "planned_path_cost_m": planned_path_cost_m,
            "path_budget_m": path_budget_m,
        }
    )


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if not isinstance(payload.get("source_global_99_config"), str) or not payload["source_global_99_config"].strip():
        raise ConfigError("source_global_99_config must be a non-empty string")
    target = _float(payload.get("target_coverage_rate"), "target_coverage_rate")
    if target <= 0.0 or target > 1.0:
        raise ConfigError("target_coverage_rate must be > 0 and <= 1")
    path_budget = _float(payload.get("path_budget_m"), "path_budget_m")
    if path_budget < 0.0:
        raise ConfigError("path_budget_m must be >= 0")
    coverage_radius = _int(payload.get("coverage_radius_cells"), "coverage_radius_cells")
    if coverage_radius < 0:
        raise ConfigError("coverage_radius_cells must be >= 0")
    step_limit = _int(payload.get("frontier_step_limit"), "frontier_step_limit")
    if step_limit < 0:
        raise ConfigError("frontier_step_limit must be >= 0")
    revisit_penalty = _float(payload.get("revisit_penalty_weight"), "revisit_penalty_weight")
    if revisit_penalty < 0.0:
        raise ConfigError("revisit_penalty_weight must be >= 0")
    new_coverage_weight = _float(payload.get("new_coverage_weight"), "new_coverage_weight")
    if new_coverage_weight < 0.0:
        raise ConfigError("new_coverage_weight must be >= 0")
    normalized = dict(payload)
    normalized.update(
        {
            "target_coverage_rate": target,
            "path_budget_m": path_budget,
            "coverage_radius_cells": coverage_radius,
            "frontier_step_limit": step_limit,
            "revisit_penalty_weight": revisit_penalty,
            "new_coverage_weight": new_coverage_weight,
        }
    )
    _resolve_config_reference(normalized["source_global_99_config"], path, repo_root)
    return normalized


def _resolve_config_reference(reference: str, config_path: Path, repo_root: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        resolved = candidate
    else:
        repo_candidate = repo_root / candidate
        resolved = repo_candidate if repo_candidate.exists() else config_path.parent / candidate
    if not resolved.is_file():
        raise ConfigError(f"source_global_99_config does not exist: {resolved}")
    return resolved


def _summary_reason_codes(
    scenario_results: list[dict[str, Any]],
    coverage_target_met: bool,
    total_denominator: int,
) -> list[str]:
    reasons: list[str] = []
    if total_denominator <= 0:
        reasons.append("coverage_denominator_invalid")
    for result in scenario_results:
        for reason in result["reason_codes"]:
            reasons.append(reason)
    if total_denominator > 0 and not coverage_target_met:
        reasons.append("coverage_target_not_met")
    return unique_sorted(reasons)


def _public_scenario_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": result["scenario_id"],
        "status": result["status"],
        "reason_codes": result["reason_codes"],
        "infeasible_reason_codes": result["infeasible_reason_codes"],
        "target_coverage_rate": result["target_coverage_rate"],
        "achieved_coverage_rate": result["achieved_coverage_rate"],
        "coverage_target_met": result["coverage_target_met"],
        "reachable_safe_cell_count": result["reachable_safe_cell_count"],
        "covered_reachable_safe_cell_count": result["covered_reachable_safe_cell_count"],
        "generated_coverage_event_count": result["generated_coverage_event_count"],
        "planned_path_cost_m": result["planned_path_cost_m"],
        "path_budget_m": result["path_budget_m"],
        "path_budget_exhausted": result["path_budget_exhausted"],
        "frontier_plan_complete": result["frontier_plan_complete"],
    }


def _manifest(
    config_path: Path,
    source_config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "config": str(config_path),
        "source_global_99_config": str(source_config_path),
        "output_root": str(output_root),
        "artifacts": {name: str(path) for name, path in paths.items()},
        "target_coverage_rate": summary["target_coverage_rate"],
        "achieved_coverage_rate": summary["achieved_coverage_rate"],
        "next_required_change": summary["next_required_change"],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _budget_audit(config: dict[str, Any], scenario_results: list[dict[str, Any]], planned_path_cost_m: float) -> dict[str, Any]:
    return {
        "schema_version": BUDGET_AUDIT_SCHEMA_VERSION,
        "path_budget_m": config["path_budget_m"],
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_exhausted": any(result["path_budget_exhausted"] for result in scenario_results),
        "scenario_budgets": [
            {
                "scenario_id": result["scenario_id"],
                "planned_path_cost_m": result["planned_path_cost_m"],
                "path_budget_m": result["path_budget_m"],
                "path_budget_exhausted": result["path_budget_exhausted"],
                "generated_coverage_event_count": result["generated_coverage_event_count"],
                "frontier_plan_complete": result["frontier_plan_complete"],
            }
            for result in scenario_results
        ],
    }


def _rejection_report(reason_codes: list[str], scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "scenario_failures": [
            {
                "scenario_id": result["scenario_id"],
                "reason_codes": result["reason_codes"],
                "infeasible_reason_codes": result["infeasible_reason_codes"],
            }
            for result in scenario_results
            if result["reason_codes"] or result["infeasible_reason_codes"]
        ],
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Frontier Coverage Planner Baseline v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- achieved_coverage_rate: `{summary['achieved_coverage_rate']}`",
        f"- generated_coverage_event_count: `{summary['generated_coverage_event_count']}`",
        f"- planned_path_cost_m: `{summary['planned_path_cost_m']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Scenario Summaries",
        "",
    ]
    for scenario in summary["scenario_summaries"]:
        lines.extend(
            [
                f"### {scenario['scenario_id']}",
                "",
                f"- status: `{scenario['status']}`",
                f"- achieved_coverage_rate: `{scenario['achieved_coverage_rate']}`",
                f"- generated_coverage_event_count: `{scenario['generated_coverage_event_count']}`",
                f"- planned_path_cost_m: `{scenario['planned_path_cost_m']}`",
                f"- infeasible_reason_codes: `{scenario['infeasible_reason_codes']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Rejection Report",
            "",
            f"- failure_reason_code_counts: `{rejection_report['failure_reason_code_counts']}`",
            "",
            "This stage is a deterministic non-learning frontier baseline. It does not train PPO, call path-planner, use NPZ/sidecar maps, publish a checkpoint, replace default policy, connect a real executor, or modify network/action space/default A*.",
            "",
        ]
    )
    return "\n".join(lines)


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
