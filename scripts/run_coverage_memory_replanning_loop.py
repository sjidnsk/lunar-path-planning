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
    from frontier_coverage_planner_common import (
        coverage_rate,
        frontier_cells,
        score_frontier_candidate,
        select_frontier_candidate,
    )
    from git_provenance import git_snapshot
    from global_99_coverage_contract import (
        Cell,
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
    from scripts.frontier_coverage_planner_common import (
        coverage_rate,
        frontier_cells,
        score_frontier_candidate,
        select_frontier_candidate,
    )
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import (
        Cell,
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


CONFIG_SCHEMA_VERSION = "coverage-memory-replanning-loop-config/v1"
FRONTIER_CONFIG_SCHEMA_VERSION = "frontier-coverage-planner-baseline-config/v1"
SUMMARY_SCHEMA_VERSION = "coverage-memory-replanning-loop-summary/v1"
MANIFEST_SCHEMA_VERSION = "coverage-memory-replanning-loop-manifest/v1"
TRACE_ROW_SCHEMA_VERSION = "coverage-memory-replanning-trace-row/v1"
SNAPSHOT_ROW_SCHEMA_VERSION = "coverage-memory-snapshot-row/v1"
LEDGER_ROW_SCHEMA_VERSION = "coverage-memory-ledger-row/v1"
BUDGET_AUDIT_SCHEMA_VERSION = "coverage-memory-budget-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "coverage-memory-rejection-report/v1"

DEFAULT_CONFIG = "configs/coverage_memory_replanning_loop_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_coverage_memory_replanning_loop_v1"

SUMMARY_FILE = "coverage-memory-replanning-loop-summary.json"
MANIFEST_FILE = "coverage-memory-replanning-loop-manifest.json"
TRACE_FILE = "coverage-memory-replanning-trace.jsonl"
SNAPSHOT_FILE = "coverage-memory-snapshots.jsonl"
LEDGER_FILE = "coverage-memory-ledger.jsonl"
BUDGET_AUDIT_FILE = "coverage-memory-budget-audit.json"
REJECTION_REPORT_FILE = "coverage-memory-rejection-report.json"
REPORT_FILE = "coverage-memory-replanning-loop-report.md"

PASS_NEXT_REQUIRED_CHANGE = "policy_guided_global_coverage"
FAIL_NEXT_REQUIRED_CHANGE = "fix_coverage_memory_replanning_loop"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Coverage Memory + Replanning Loop v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_coverage_memory_replanning_loop(
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


def run_coverage_memory_replanning_loop(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path)
    frontier_config_path = _resolve_config_reference(config["source_frontier_baseline_config"], config_path, repo_root)
    source_config_path = _resolve_config_reference(config["source_global_99_config"], config_path, repo_root)
    frontier_config = _load_frontier_config(frontier_config_path)
    source_config = load_global_99_config(source_config_path)
    resume_snapshot = _load_resume_snapshot(config, config_path, repo_root)

    target = float(config["target_coverage_rate"])
    path_budget_m = float(config["path_budget_m"])
    scenario_results = [
        _run_memory_scenario(
            scenario,
            target_coverage_rate=target,
            path_budget_m=path_budget_m,
            coverage_radius_cells=int(config["coverage_radius_cells"]),
            replanning_cycle_limit=int(config["replanning_cycle_limit"]),
            segment_step_limit=int(config["segment_step_limit"]),
            memory_snapshot_interval=int(config["memory_snapshot_interval"]),
            revisit_penalty_weight=float(frontier_config["revisit_penalty_weight"]),
            new_coverage_weight=float(frontier_config["new_coverage_weight"]),
            resume_snapshot=resume_snapshot,
        )
        for scenario in source_config["scenarios"]
    ]

    trace_rows = [row for result in scenario_results for row in result["trace_rows"]]
    snapshot_rows = [row for result in scenario_results for row in result["snapshot_rows"]]
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
    planned_path_cost_m = sum(result["planned_path_cost_m"] for result in scenario_results)
    infeasible_reason_codes = unique_sorted(
        reason for result in scenario_results for reason in result["infeasible_reason_codes"]
    )
    memory_resume_verified = all(result["memory_resume_verified"] for result in scenario_results)
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "trace": output_root / TRACE_FILE,
        "snapshots": output_root / SNAPSHOT_FILE,
        "ledger": output_root / LEDGER_FILE,
        "budget_audit": output_root / BUDGET_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "config": str(config_path),
        "source_frontier_baseline_config": str(frontier_config_path),
        "source_global_99_config": str(source_config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "replanning_trace": str(paths["trace"]),
        "memory_snapshots": str(paths["snapshots"]),
        "coverage_memory_ledger": str(paths["ledger"]),
        "budget_audit": str(paths["budget_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "target_coverage_rate": target,
        "achieved_coverage_rate": achieved,
        "coverage_target_met": coverage_target_met,
        "reachable_safe_cell_count": total_denominator,
        "covered_reachable_safe_cell_count": total_covered,
        "replanning_cycle_count": sum(result["replanning_cycle_count"] for result in scenario_results),
        "memory_snapshot_count": len(snapshot_rows),
        "memory_resume_supported": True,
        "memory_resume_verified": memory_resume_verified,
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_m": path_budget_m,
        "path_budget_exhausted": any(result["path_budget_exhausted"] for result in scenario_results),
        "coverage_memory_complete": status == "passed",
        "scenario_count": len(scenario_results),
        "passed_scenario_count": sum(1 for result in scenario_results if result["status"] == "passed"),
        "failed_scenario_count": sum(1 for result in scenario_results if result["status"] == "failed"),
        "infeasible_reason_codes": infeasible_reason_codes,
        "coverage_radius_cells": config["coverage_radius_cells"],
        "replanning_cycle_limit": config["replanning_cycle_limit"],
        "segment_step_limit": config["segment_step_limit"],
        "memory_snapshot_interval": config["memory_snapshot_interval"],
        "revisit_penalty_weight": frontier_config["revisit_penalty_weight"],
        "new_coverage_weight": frontier_config["new_coverage_weight"],
        "next_required_change": next_required_change,
        "scenario_summaries": [_public_scenario_summary(result) for result in scenario_results],
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "checkpoint_publication_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
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
    manifest = _manifest(config_path, frontier_config_path, source_config_path, output_root, paths, summary)
    budget_audit = _budget_audit(config, frontier_config, scenario_results, planned_path_cost_m)
    rejection_report = _rejection_report(reason_codes, scenario_results)

    write_jsonl(paths["trace"], trace_rows)
    write_jsonl(paths["snapshots"], snapshot_rows)
    write_jsonl(paths["ledger"], ledger_rows)
    write_json(paths["budget_audit"], budget_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _run_memory_scenario(
    scenario: dict[str, Any],
    *,
    target_coverage_rate: float,
    path_budget_m: float,
    coverage_radius_cells: int,
    replanning_cycle_limit: int,
    segment_step_limit: int,
    memory_snapshot_interval: int,
    revisit_penalty_weight: float,
    new_coverage_weight: float,
    resume_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    geometry = scenario_geometry(scenario)
    scenario_id = geometry["scenario_id"]
    width = geometry["width"]
    height = geometry["height"]
    resolution_m = geometry["resolution_m"]
    navigation_cells: set[Cell] = geometry["navigation_cells"]
    target_cells: set[Cell] = geometry["reachable_safe_cells"]
    start: Cell = geometry["start"]
    trace_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    path_budget_exhausted = False
    replanning_cycle_limit_exhausted = False
    frontier_exhausted = False
    frontier_unreachable = False
    no_memory_progress = False
    loaded_resume = resume_snapshot if resume_snapshot and resume_snapshot.get("scenario_id") == scenario_id else None

    if loaded_resume:
        current_cell = _cell_from_snapshot(loaded_resume["current_cell"])
        planned_path_cost_m = float(loaded_resume["planned_path_cost_m"])
        coverage_memory_cells = _cells_from_snapshot(loaded_resume["coverage_memory_cells"])
        completed_cycles = int(loaded_resume["next_cycle_index"])
    else:
        current_cell = start
        planned_path_cost_m = 0.0
        coverage_memory_cells = coverage_footprint(start, width, height, coverage_radius_cells) & navigation_cells
        completed_cycles = 0
        covered_target_cells = coverage_memory_cells & target_cells
        _append_trace_and_ledger_row(
            trace_rows=trace_rows,
            ledger_rows=ledger_rows,
            scenario_id=scenario_id,
            cycle_index=0,
            event_id=f"{scenario_id}-initial-memory",
            selected_waypoint=start,
            path=[start] if start in navigation_cells else [],
            event_cells=coverage_memory_cells,
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
            choice_reason="initial_memory",
        )

    covered_target_cells = coverage_memory_cells & target_cells
    _append_snapshot_row(
        snapshot_rows=snapshot_rows,
        scenario_id=scenario_id,
        cycle_index=completed_cycles,
        current_cell=current_cell,
        coverage_memory_cells=coverage_memory_cells,
        target_cells=target_cells,
        planned_path_cost_m=planned_path_cost_m,
        path_budget_m=path_budget_m,
        resume_source="loaded_snapshot" if loaded_resume else "initial_state",
    )

    while coverage_rate(covered_target_cells, target_cells) + TOLERANCE < target_coverage_rate:
        if completed_cycles >= replanning_cycle_limit:
            replanning_cycle_limit_exhausted = True
            break
        frontier = frontier_cells(coverage_memory_cells, navigation_cells)
        if not frontier:
            frontier_exhausted = True
            break
        candidate = select_frontier_candidate(
            current_cell=current_cell,
            frontier=frontier,
            navigation_cells=navigation_cells,
            target_cells=target_cells,
            covered_target_cells=covered_target_cells,
            coverage_map_cells=coverage_memory_cells,
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

        segment_path = candidate["path"][: segment_step_limit + 1]
        if not segment_path:
            frontier_unreachable = True
            break
        selected_waypoint = segment_path[-1]
        path_cells = set(segment_path)
        footprint = coverage_footprint(selected_waypoint, width, height, coverage_radius_cells) & navigation_cells
        event_cells = footprint | path_cells
        new_target_cells = (event_cells & target_cells) - covered_target_cells
        revisited_path_cell_count = sum(1 for path_cell in segment_path if path_cell in coverage_memory_cells)
        path_cost_m = (len(segment_path) - 1) * resolution_m
        score = score_frontier_candidate(
            path_cost_m=path_cost_m,
            revisited_path_cell_count=revisited_path_cell_count,
            new_covered_cell_count=len(new_target_cells),
            revisit_penalty_weight=revisit_penalty_weight,
            new_coverage_weight=new_coverage_weight,
        )
        attempted_path_cost_m = planned_path_cost_m + path_cost_m
        cycle_index = completed_cycles + 1
        if attempted_path_cost_m > path_budget_m + TOLERANCE:
            path_budget_exhausted = True
            _append_trace_and_ledger_row(
                trace_rows=trace_rows,
                ledger_rows=ledger_rows,
                scenario_id=scenario_id,
                cycle_index=cycle_index,
                event_id=f"{scenario_id}-memory-{cycle_index:04d}",
                selected_waypoint=selected_waypoint,
                path=segment_path,
                event_cells=event_cells,
                counted=False,
                attempted_path_cost_m=attempted_path_cost_m,
                planned_path_cost_m=planned_path_cost_m,
                path_budget_m=path_budget_m,
                path_cost_m=path_cost_m,
                score=score,
                revisited_path_cell_count=revisited_path_cell_count,
                new_target_cell_count=0,
                target_cells=target_cells,
                covered_target_cells=covered_target_cells,
                frontier_cluster_count=candidate["frontier_cluster_count"],
                choice_reason="path_budget_exhausted",
            )
            break

        before_count = len(covered_target_cells)
        before_memory_count = len(coverage_memory_cells)
        planned_path_cost_m = attempted_path_cost_m
        current_cell = selected_waypoint
        coverage_memory_cells |= event_cells
        covered_target_cells = coverage_memory_cells & target_cells
        new_target_cell_count = len(covered_target_cells) - before_count
        if len(coverage_memory_cells) == before_memory_count:
            no_memory_progress = True
            break

        _append_trace_and_ledger_row(
            trace_rows=trace_rows,
            ledger_rows=ledger_rows,
            scenario_id=scenario_id,
            cycle_index=cycle_index,
            event_id=f"{scenario_id}-memory-{cycle_index:04d}",
            selected_waypoint=selected_waypoint,
            path=segment_path,
            event_cells=event_cells,
            counted=True,
            attempted_path_cost_m=attempted_path_cost_m,
            planned_path_cost_m=planned_path_cost_m,
            path_budget_m=path_budget_m,
            path_cost_m=path_cost_m,
            score=score,
            revisited_path_cell_count=revisited_path_cell_count,
            new_target_cell_count=new_target_cell_count,
            target_cells=target_cells,
            covered_target_cells=covered_target_cells,
            frontier_cluster_count=candidate["frontier_cluster_count"],
            choice_reason="memory_replanning",
        )
        completed_cycles = cycle_index
        if completed_cycles % memory_snapshot_interval == 0:
            _append_snapshot_row(
                snapshot_rows=snapshot_rows,
                scenario_id=scenario_id,
                cycle_index=completed_cycles,
                current_cell=current_cell,
                coverage_memory_cells=coverage_memory_cells,
                target_cells=target_cells,
                planned_path_cost_m=planned_path_cost_m,
                path_budget_m=path_budget_m,
                resume_source="replanning_cycle",
            )

    target_met = bool(target_cells) and coverage_rate(covered_target_cells, target_cells) + TOLERANCE >= target_coverage_rate
    if not target_cells:
        reason_codes.append("coverage_denominator_invalid")
    if target_cells and not target_met:
        reason_codes.append("coverage_target_not_met")
    if path_budget_exhausted:
        reason_codes.append("insufficient_budget")
    if replanning_cycle_limit_exhausted:
        reason_codes.append("replanning_cycle_limit_exhausted")
    if frontier_exhausted:
        reason_codes.append("frontier_exhausted")
    if frontier_unreachable:
        reason_codes.append("frontier_unreachable")
    if no_memory_progress:
        reason_codes.append("coverage_memory_no_progress")

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

    snapshot_counts = [row["covered_reachable_safe_cell_count"] for row in snapshot_rows]
    memory_resume_verified = bool(snapshot_rows) and snapshot_counts == sorted(snapshot_counts)
    if loaded_resume:
        memory_resume_verified = memory_resume_verified and target_met

    return {
        "scenario_id": scenario_id,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": unique_sorted(reason_codes),
        "infeasible_reason_codes": unique_sorted(infeasible_reason_codes),
        "target_coverage_rate": target_coverage_rate,
        "achieved_coverage_rate": coverage_rate(covered_target_cells, target_cells),
        "coverage_target_met": target_met,
        "reachable_safe_cell_count": len(target_cells),
        "covered_reachable_safe_cell_count": len(covered_target_cells),
        "replanning_cycle_count": completed_cycles,
        "memory_snapshot_count": len(snapshot_rows),
        "memory_resume_verified": memory_resume_verified,
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_m": path_budget_m,
        "path_budget_exhausted": path_budget_exhausted,
        "coverage_memory_complete": target_met and not reason_codes,
        "replanning_cycle_limit_exhausted": replanning_cycle_limit_exhausted,
        "frontier_exhausted": frontier_exhausted,
        "frontier_unreachable": frontier_unreachable,
        "roi_cell_count": len(geometry["roi_cells"]),
        "blocked_roi_cell_count": len(geometry["roi_cells"] & geometry["blocked_cells"]),
        "unsafe_roi_cell_count": len(geometry["roi_cells"] & geometry["unsafe_cells"]),
        "unreachable_roi_cell_count": len(geometry["unreachable_roi_cells"]),
        "trace_rows": trace_rows,
        "snapshot_rows": snapshot_rows,
        "ledger_rows": ledger_rows,
    }


def _append_trace_and_ledger_row(
    *,
    trace_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    scenario_id: str,
    cycle_index: int,
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
    trace_rows.append(
        {
            "schema_version": TRACE_ROW_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "cycle_index": cycle_index,
            "event_id": event_id,
            "choice_source": "coverage_memory_replanning",
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
            "cycle_index": cycle_index,
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


def _append_snapshot_row(
    *,
    snapshot_rows: list[dict[str, Any]],
    scenario_id: str,
    cycle_index: int,
    current_cell: Cell,
    coverage_memory_cells: set[Cell],
    target_cells: set[Cell],
    planned_path_cost_m: float,
    path_budget_m: float,
    resume_source: str,
) -> None:
    covered_target_cells = coverage_memory_cells & target_cells
    snapshot_rows.append(
        {
            "schema_version": SNAPSHOT_ROW_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "cycle_index": cycle_index,
            "next_cycle_index": cycle_index,
            "resume_source": resume_source,
            "current_cell": list(current_cell),
            "planned_path_cost_m": planned_path_cost_m,
            "path_budget_m": path_budget_m,
            "coverage_memory_cell_count": len(coverage_memory_cells),
            "covered_reachable_safe_cell_count": len(covered_target_cells),
            "reachable_safe_cell_count": len(target_cells),
            "achieved_coverage_rate": coverage_rate(covered_target_cells, target_cells),
            "coverage_memory_cells": _sorted_cells(coverage_memory_cells),
        }
    )


def _load_config(path: Path) -> dict[str, Any]:
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
    for key in ("source_frontier_baseline_config", "source_global_99_config"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    target = _float(payload.get("target_coverage_rate"), "target_coverage_rate")
    if target <= 0.0 or target > 1.0:
        raise ConfigError("target_coverage_rate must be > 0 and <= 1")
    path_budget = _float(payload.get("path_budget_m"), "path_budget_m")
    if path_budget < 0.0:
        raise ConfigError("path_budget_m must be >= 0")
    coverage_radius = _int(payload.get("coverage_radius_cells"), "coverage_radius_cells")
    if coverage_radius < 0:
        raise ConfigError("coverage_radius_cells must be >= 0")
    replanning_limit = _int(payload.get("replanning_cycle_limit"), "replanning_cycle_limit")
    if replanning_limit < 0:
        raise ConfigError("replanning_cycle_limit must be >= 0")
    segment_limit = _int(payload.get("segment_step_limit"), "segment_step_limit")
    if segment_limit <= 0:
        raise ConfigError("segment_step_limit must be > 0")
    snapshot_interval = _int(payload.get("memory_snapshot_interval"), "memory_snapshot_interval")
    if snapshot_interval <= 0:
        raise ConfigError("memory_snapshot_interval must be > 0")
    resume_value = payload.get("resume_from_memory_snapshot")
    if resume_value is not None and not isinstance(resume_value, str):
        raise ConfigError("resume_from_memory_snapshot must be null or a string")
    normalized = dict(payload)
    normalized["target_coverage_rate"] = target
    normalized["path_budget_m"] = path_budget
    normalized["coverage_radius_cells"] = coverage_radius
    normalized["replanning_cycle_limit"] = replanning_limit
    normalized["segment_step_limit"] = segment_limit
    normalized["memory_snapshot_interval"] = snapshot_interval
    return normalized


def _load_frontier_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"frontier config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"frontier config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("frontier config root must be an object")
    if payload.get("schema_version") != FRONTIER_CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"frontier schema_version must be {FRONTIER_CONFIG_SCHEMA_VERSION!r}")
    revisit_penalty = _float(payload.get("revisit_penalty_weight"), "revisit_penalty_weight")
    new_coverage = _float(payload.get("new_coverage_weight"), "new_coverage_weight")
    if revisit_penalty < 0.0 or new_coverage < 0.0:
        raise ConfigError("frontier weights must be >= 0")
    normalized = dict(payload)
    normalized["revisit_penalty_weight"] = revisit_penalty
    normalized["new_coverage_weight"] = new_coverage
    return normalized


def _load_resume_snapshot(config: dict[str, Any], config_path: Path, repo_root: Path) -> dict[str, Any] | None:
    snapshot_path = config.get("resume_from_memory_snapshot")
    if snapshot_path is None:
        return None
    path = _resolve_config_reference(snapshot_path, config_path, repo_root)
    if not path.is_file():
        raise ConfigError(f"resume snapshot does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"resume snapshot JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("resume snapshot root must be an object")
    if payload.get("schema_version") != SNAPSHOT_ROW_SCHEMA_VERSION:
        raise ConfigError(f"resume snapshot schema_version must be {SNAPSHOT_ROW_SCHEMA_VERSION!r}")
    return payload


def _resolve_config_reference(value: str, config_path: Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_relative = config_path.parent / path
    if config_relative.exists():
        return config_relative
    return resolve_path(path, repo_root)


def _summary_reason_codes(
    scenario_results: list[dict[str, Any]],
    coverage_target_met: bool,
    total_denominator: int,
) -> list[str]:
    reasons: list[str] = []
    for result in scenario_results:
        reasons.extend(result["reason_codes"])
    if total_denominator <= 0:
        reasons.append("coverage_denominator_invalid")
    if total_denominator > 0 and not coverage_target_met:
        reasons.append("coverage_target_not_met")
    return unique_sorted(reasons)


def _public_scenario_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "scenario_id",
            "status",
            "reason_codes",
            "infeasible_reason_codes",
            "target_coverage_rate",
            "achieved_coverage_rate",
            "coverage_target_met",
            "reachable_safe_cell_count",
            "covered_reachable_safe_cell_count",
            "replanning_cycle_count",
            "memory_snapshot_count",
            "memory_resume_verified",
            "planned_path_cost_m",
            "path_budget_m",
            "path_budget_exhausted",
            "coverage_memory_complete",
            "replanning_cycle_limit_exhausted",
            "frontier_exhausted",
            "frontier_unreachable",
            "roi_cell_count",
            "blocked_roi_cell_count",
            "unsafe_roi_cell_count",
            "unreachable_roi_cell_count",
        )
    }


def _manifest(
    config_path: Path,
    frontier_config_path: Path,
    source_config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "stage": "Coverage Memory + Replanning Loop v1",
        "config": str(config_path),
        "source_frontier_baseline_config": str(frontier_config_path),
        "source_global_99_config": str(source_config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "boundary": {
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "uses_ppo_policy": False,
            "uses_path_planner": False,
            "uses_npz_or_sidecar": False,
        },
    }


def _budget_audit(
    config: dict[str, Any],
    frontier_config: dict[str, Any],
    scenario_results: list[dict[str, Any]],
    planned_path_cost_m: float,
) -> dict[str, Any]:
    return {
        "schema_version": BUDGET_AUDIT_SCHEMA_VERSION,
        "target_coverage_rate": config["target_coverage_rate"],
        "path_budget_m": config["path_budget_m"],
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_exhausted": any(result["path_budget_exhausted"] for result in scenario_results),
        "coverage_radius_cells": config["coverage_radius_cells"],
        "replanning_cycle_limit": config["replanning_cycle_limit"],
        "segment_step_limit": config["segment_step_limit"],
        "revisit_penalty_weight": frontier_config["revisit_penalty_weight"],
        "new_coverage_weight": frontier_config["new_coverage_weight"],
        "scenario_budget": [
            {
                "scenario_id": result["scenario_id"],
                "planned_path_cost_m": result["planned_path_cost_m"],
                "path_budget_m": result["path_budget_m"],
                "path_budget_exhausted": result["path_budget_exhausted"],
                "replanning_cycle_count": result["replanning_cycle_count"],
            }
            for result in scenario_results
        ],
    }


def _rejection_report(reason_codes: list[str], scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_reason_counts = Counter(reason for result in scenario_results for reason in result["reason_codes"])
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(Counter(reason_codes)),
        "scenario_failure_reason_code_counts": dict(scenario_reason_counts),
        "scenario_rejections": [
            {
                "scenario_id": result["scenario_id"],
                "status": result["status"],
                "reason_codes": result["reason_codes"],
                "infeasible_reason_codes": result["infeasible_reason_codes"],
            }
            for result in scenario_results
        ],
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Coverage Memory + Replanning Loop v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- achieved_coverage_rate: `{summary['achieved_coverage_rate']}`",
        f"- coverage_target_met: `{summary['coverage_target_met']}`",
        f"- replanning_cycle_count: `{summary['replanning_cycle_count']}`",
        f"- memory_snapshot_count: `{summary['memory_snapshot_count']}`",
        f"- memory_resume_verified: `{summary['memory_resume_verified']}`",
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
                f"- replanning_cycle_count: `{scenario['replanning_cycle_count']}`",
                f"- memory_snapshot_count: `{scenario['memory_snapshot_count']}`",
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
            "This stage is a deterministic coverage-memory replanning loop. It does not train PPO, call path-planner, use NPZ/sidecar maps, publish a checkpoint, replace default policy, connect a real executor, or modify network/action space/default A*.",
            "",
        ]
    )
    return "\n".join(lines)


def _cell_from_snapshot(value: Any) -> Cell:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError("snapshot current_cell must be a two-item cell")
    return (_int(value[0], "snapshot.current_cell[0]"), _int(value[1], "snapshot.current_cell[1]"))


def _cells_from_snapshot(values: Any) -> set[Cell]:
    if not isinstance(values, list):
        raise ConfigError("snapshot coverage_memory_cells must be a list")
    return {_cell_from_snapshot(value) for value in values}


def _sorted_cells(cells: set[Cell]) -> list[list[int]]:
    return [[x, y] for x, y in sorted(cells, key=lambda cell: (cell[1], cell[0]))]


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
