from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import (
        ConfigError as ContractConfigError,
        evaluate_static_coverage_scenario,
        load_global_99_config,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import (
        ConfigError as ContractConfigError,
        evaluate_static_coverage_scenario,
        load_global_99_config,
    )


CONFIG_SCHEMA_VERSION = "global-99-coverage-benchmark-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-coverage-benchmark-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-coverage-benchmark-manifest/v1"
LEDGER_ROW_SCHEMA_VERSION = "global-99-coverage-ledger-row/v1"
DENOMINATOR_AUDIT_SCHEMA_VERSION = "global-99-coverage-denominator-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "global-99-coverage-rejection-report/v1"

DEFAULT_CONFIG = "configs/global_99_coverage_benchmark_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_coverage_benchmark_v1"

SUMMARY_FILE = "global-99-coverage-benchmark-summary.json"
MANIFEST_FILE = "global-99-coverage-benchmark-manifest.json"
LEDGER_FILE = "global-99-coverage-ledger.jsonl"
DENOMINATOR_AUDIT_FILE = "global-99-coverage-denominator-audit.json"
REJECTION_REPORT_FILE = "global-99-coverage-rejection-report.json"
REPORT_FILE = "global-99-coverage-benchmark-report.md"

PASS_NEXT_REQUIRED_CHANGE = "frontier_coverage_planner_baseline"
FAIL_NEXT_REQUIRED_CHANGE = "fix_global_99_coverage_benchmark_contract"
TOLERANCE = 1.0e-12

Cell = tuple[int, int]


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99% Coverage Benchmark v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_coverage_benchmark(
            config_path=_resolve_path(Path(args.config), repo_root),
            output_root=_resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ContractConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "target_coverage_rate": summary["target_coverage_rate"],
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


def run_global_99_coverage_benchmark(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_global_99_config(config_path)
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "ledger": output_root / LEDGER_FILE,
        "denominator_audit": output_root / DENOMINATOR_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }

    target = float(config["target_coverage_rate"])
    path_budget_m = float(config["path_budget_m"])
    scenario_results = [
        evaluate_static_coverage_scenario(scenario, target_coverage_rate=target, path_budget_m=path_budget_m)
        for scenario in config["scenarios"]
    ]
    ledger_rows = [row for result in scenario_results for row in result["ledger_rows"]]
    total_denominator = sum(result["reachable_safe_cell_count"] for result in scenario_results)
    total_covered = sum(result["covered_reachable_safe_cell_count"] for result in scenario_results)
    achieved = (total_covered / total_denominator) if total_denominator else 0.0
    coverage_denominator_valid = all(result["coverage_denominator_valid"] for result in scenario_results)
    coverage_ledger_complete = all(result["coverage_ledger_complete"] for result in scenario_results)
    path_budget_exhausted = any(result["path_budget_exhausted"] for result in scenario_results)
    coverage_target_met = (
        coverage_denominator_valid
        and coverage_ledger_complete
        and achieved + TOLERANCE >= target
        and all(result["coverage_target_met"] for result in scenario_results)
    )

    infeasible_reason_codes = _unique_sorted(
        reason for result in scenario_results for reason in result["infeasible_reason_codes"]
    )
    reason_codes = _reason_codes(
        scenario_results=scenario_results,
        coverage_target_met=coverage_target_met,
        coverage_denominator_valid=coverage_denominator_valid,
        coverage_ledger_complete=coverage_ledger_complete,
    )
    status = "passed" if not reason_codes else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE if status == "passed" else FAIL_NEXT_REQUIRED_CHANGE

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "coverage_ledger": str(paths["ledger"]),
        "denominator_audit": str(paths["denominator_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "target_coverage_rate": target,
        "path_budget_m": path_budget_m,
        "achieved_coverage_rate": achieved,
        "coverage_target_met": coverage_target_met,
        "reachable_safe_cell_count": total_denominator,
        "covered_reachable_safe_cell_count": total_covered,
        "scenario_count": len(scenario_results),
        "passed_scenario_count": sum(1 for result in scenario_results if result["status"] == "passed"),
        "failed_scenario_count": sum(1 for result in scenario_results if result["status"] == "failed"),
        "infeasible_reason_codes": infeasible_reason_codes,
        "path_budget_exhausted": path_budget_exhausted,
        "coverage_denominator_valid": coverage_denominator_valid,
        "coverage_ledger_complete": coverage_ledger_complete,
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
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    denominator_audit = _denominator_audit(config, scenario_results, total_denominator, total_covered)
    rejection_report = _rejection_report(reason_codes, scenario_results)
    manifest = _manifest(config_path, output_root, paths, summary, config)

    _write_jsonl(paths["ledger"], ledger_rows)
    _write_json(paths["denominator_audit"], denominator_audit)
    _write_json(paths["rejection_report"], rejection_report)
    _write_json(paths["manifest"], manifest)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _evaluate_scenario(
    scenario: dict[str, Any],
    *,
    target_coverage_rate: float,
    path_budget_m: float,
) -> dict[str, Any]:
    scenario_id = str(scenario["scenario_id"])
    grid = scenario["grid"]
    width = int(grid["width"])
    height = int(grid["height"])
    resolution_m = float(grid.get("resolution_m", 1.0))
    all_cells = {(x, y) for y in range(height) for x in range(width)}
    roi_cells = _roi_cells(scenario["roi"], width, height)
    blocked_cells = _rectangles_to_cells(scenario.get("blocked_rectangles", []), width, height)
    unsafe_cells = _rectangles_to_cells(scenario.get("unsafe_rectangles", []), width, height)
    excluded_cells = blocked_cells | unsafe_cells
    start = _cell(scenario["start_cell"], "start_cell")
    reachable_cells = _reachable_cells(start, width, height, excluded_cells)
    safe_roi_cells = roi_cells - excluded_cells
    reachable_safe_cells = safe_roi_cells & reachable_cells
    unsafe_roi_cells = roi_cells & excluded_cells
    unreachable_roi_cells = safe_roi_cells - reachable_cells

    covered: set[Cell] = set()
    ledger_rows: list[dict[str, Any]] = []
    cumulative_path_cost_m = 0.0
    path_budget_exhausted = False
    coverage_ledger_complete = True
    seen_event_ids: set[str] = set()
    events = scenario.get("coverage_events", [])
    for event_index, event in enumerate(events):
        event_id = str(event.get("event_id", f"event-{event_index:04d}"))
        if event_id in seen_event_ids:
            coverage_ledger_complete = False
        seen_event_ids.add(event_id)
        path_cost_m = float(event.get("path_cost_m", 0.0))
        attempted_cumulative = cumulative_path_cost_m + path_cost_m
        counted = not path_budget_exhausted and attempted_cumulative <= path_budget_m + TOLERANCE
        event_cells = _coverage_event_cells(event, width, height) & all_cells
        event_reachable_safe_cells = event_cells & reachable_safe_cells
        before_count = len(covered)
        if counted:
            cumulative_path_cost_m = attempted_cumulative
            covered |= event_reachable_safe_cells
        else:
            path_budget_exhausted = True
        new_count = len(covered) - before_count
        denominator_count = len(reachable_safe_cells)
        ledger_rows.append(
            {
                "schema_version": LEDGER_ROW_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "event_index": event_index,
                "event_id": event_id,
                "path_cost_m": path_cost_m,
                "attempted_cumulative_path_cost_m": attempted_cumulative,
                "cumulative_path_cost_m": cumulative_path_cost_m,
                "path_budget_m": path_budget_m,
                "counted": counted,
                "event_cell_count": len(event_cells),
                "event_reachable_safe_cell_count": len(event_reachable_safe_cells),
                "new_covered_reachable_safe_cell_count": new_count,
                "cumulative_covered_reachable_safe_cell_count": len(covered),
                "reachable_safe_cell_count": denominator_count,
                "achieved_coverage_rate": (len(covered) / denominator_count) if denominator_count else 0.0,
            }
        )

    reachable_safe_count = len(reachable_safe_cells)
    covered_count = len(covered)
    achieved = (covered_count / reachable_safe_count) if reachable_safe_count else 0.0
    denominator_valid = reachable_safe_count > 0
    target_met = denominator_valid and coverage_ledger_complete and achieved + TOLERANCE >= target_coverage_rate
    infeasible_reason_codes: list[str] = []
    if unsafe_roi_cells:
        infeasible_reason_codes.append("unsafe_roi_cells")
    if unreachable_roi_cells:
        infeasible_reason_codes.append("unreachable_roi_cells")
    if path_budget_exhausted:
        infeasible_reason_codes.append("insufficient_budget")
    if not denominator_valid:
        infeasible_reason_codes.append("coverage_denominator_invalid")
    if not coverage_ledger_complete:
        infeasible_reason_codes.append("coverage_ledger_incomplete")
    if denominator_valid and not target_met:
        infeasible_reason_codes.append("coverage_target_not_met")

    failure_reasons: list[str] = []
    if not denominator_valid:
        failure_reasons.append("coverage_denominator_invalid")
    if not coverage_ledger_complete:
        failure_reasons.append("coverage_ledger_incomplete")
    if denominator_valid and not target_met:
        failure_reasons.append("coverage_target_not_met")
        if path_budget_exhausted:
            failure_reasons.append("insufficient_budget")

    return {
        "scenario_id": scenario_id,
        "status": "passed" if not failure_reasons else "failed",
        "reason_codes": _unique_sorted(failure_reasons),
        "target_coverage_rate": target_coverage_rate,
        "achieved_coverage_rate": achieved,
        "coverage_target_met": target_met,
        "coverage_denominator_valid": denominator_valid,
        "coverage_ledger_complete": coverage_ledger_complete,
        "path_budget_exhausted": path_budget_exhausted,
        "grid": {
            "width": width,
            "height": height,
            "resolution_m": resolution_m,
            "cell_count": width * height,
        },
        "roi_cell_count": len(roi_cells),
        "blocked_roi_cell_count": len(roi_cells & blocked_cells),
        "unsafe_roi_cell_count": len(roi_cells & unsafe_cells),
        "excluded_roi_cell_count": len(unsafe_roi_cells),
        "unreachable_roi_cell_count": len(unreachable_roi_cells),
        "reachable_safe_cell_count": reachable_safe_count,
        "covered_reachable_safe_cell_count": covered_count,
        "infeasible_reason_codes": _unique_sorted(infeasible_reason_codes),
        "ledger_rows": ledger_rows,
    }


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
    target = _float(payload.get("target_coverage_rate"), "target_coverage_rate")
    if target <= 0.0 or target > 1.0:
        raise ConfigError("target_coverage_rate must be > 0 and <= 1")
    path_budget = _float(payload.get("path_budget_m"), "path_budget_m")
    if path_budget < 0.0:
        raise ConfigError("path_budget_m must be >= 0")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ConfigError("scenarios must be a non-empty list")
    normalized = dict(payload)
    normalized["target_coverage_rate"] = target
    normalized["path_budget_m"] = path_budget
    normalized["scenarios"] = [_validate_scenario_config(item, index) for index, item in enumerate(scenarios)]
    return normalized


def _validate_scenario_config(scenario: Any, index: int) -> dict[str, Any]:
    if not isinstance(scenario, dict):
        raise ConfigError(f"scenarios[{index}] must be an object")
    if not _nonempty_string(scenario.get("scenario_id")):
        raise ConfigError(f"scenarios[{index}].scenario_id must be a non-empty string")
    grid = scenario.get("grid")
    if not isinstance(grid, dict):
        raise ConfigError(f"scenarios[{index}].grid must be an object")
    width = _int(grid.get("width"), f"scenarios[{index}].grid.width")
    height = _int(grid.get("height"), f"scenarios[{index}].grid.height")
    if width <= 0 or height <= 0:
        raise ConfigError(f"scenarios[{index}].grid width/height must be positive")
    if "resolution_m" in grid and _float(grid["resolution_m"], f"scenarios[{index}].grid.resolution_m") <= 0:
        raise ConfigError(f"scenarios[{index}].grid.resolution_m must be positive")
    _cell(scenario.get("start_cell"), f"scenarios[{index}].start_cell")
    if not isinstance(scenario.get("roi"), dict):
        raise ConfigError(f"scenarios[{index}].roi must be an object")
    for key in ("blocked_rectangles", "unsafe_rectangles", "coverage_events"):
        if key in scenario and not isinstance(scenario[key], list):
            raise ConfigError(f"scenarios[{index}].{key} must be a list")
    return scenario


def _roi_cells(roi: dict[str, Any], width: int, height: int) -> set[Cell]:
    kind = roi.get("kind")
    if kind == "rect":
        return _rectangle_to_cells(roi, width, height)
    if kind == "cells":
        cells = roi.get("cells")
        if not isinstance(cells, list):
            raise ConfigError("roi.cells must be a list for cells ROI")
        return {_cell(cell, "roi.cells[]") for cell in cells if _cell_in_bounds(_cell(cell, "roi.cells[]"), width, height)}
    raise ConfigError("roi.kind must be 'rect' or 'cells'")


def _coverage_event_cells(event: dict[str, Any], width: int, height: int) -> set[Cell]:
    cells: set[Cell] = set()
    for rect in event.get("covered_rectangles", []) or []:
        cells |= _rectangle_to_cells(rect, width, height)
    for cell in event.get("covered_cells", []) or []:
        parsed = _cell(cell, "covered_cells[]")
        if _cell_in_bounds(parsed, width, height):
            cells.add(parsed)
    return cells


def _rectangles_to_cells(rectangles: list[Any], width: int, height: int) -> set[Cell]:
    cells: set[Cell] = set()
    for rect in rectangles:
        cells |= _rectangle_to_cells(rect, width, height)
    return cells


def _rectangle_to_cells(rect: Any, width: int, height: int) -> set[Cell]:
    if not isinstance(rect, dict):
        raise ConfigError("rectangle entries must be objects")
    x0 = _int(rect.get("x0"), "rectangle.x0")
    y0 = _int(rect.get("y0"), "rectangle.y0")
    x1 = _int(rect.get("x1"), "rectangle.x1")
    y1 = _int(rect.get("y1"), "rectangle.y1")
    if x1 < x0 or y1 < y0:
        raise ConfigError("rectangle x1/y1 must be >= x0/y0")
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return {(x, y) for y in range(y0, y1) for x in range(x0, x1)}


def _reachable_cells(start: Cell, width: int, height: int, excluded_cells: set[Cell]) -> set[Cell]:
    if not _cell_in_bounds(start, width, height) or start in excluded_cells:
        return set()
    visited = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in visited or neighbor in excluded_cells or not _cell_in_bounds(neighbor, width, height):
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def _reason_codes(
    *,
    scenario_results: list[dict[str, Any]],
    coverage_target_met: bool,
    coverage_denominator_valid: bool,
    coverage_ledger_complete: bool,
) -> list[str]:
    reasons: list[str] = []
    if not coverage_denominator_valid:
        reasons.append("coverage_denominator_invalid")
    if not coverage_ledger_complete:
        reasons.append("coverage_ledger_incomplete")
    for result in scenario_results:
        for reason in result["reason_codes"]:
            reasons.append(reason)
    if coverage_denominator_valid and coverage_ledger_complete and not coverage_target_met:
        reasons.append("coverage_target_not_met")
    return _unique_sorted(reasons)


def _denominator_audit(
    config: dict[str, Any],
    scenario_results: list[dict[str, Any]],
    total_denominator: int,
    total_covered: int,
) -> dict[str, Any]:
    return {
        "schema_version": DENOMINATOR_AUDIT_SCHEMA_VERSION,
        "target_coverage_rate": config["target_coverage_rate"],
        "path_budget_m": config["path_budget_m"],
        "reachable_safe_cell_count": total_denominator,
        "covered_reachable_safe_cell_count": total_covered,
        "coverage_denominator_valid": total_denominator > 0
        and all(result["coverage_denominator_valid"] for result in scenario_results),
        "scenario_denominators": [
            {
                "scenario_id": result["scenario_id"],
                "roi_cell_count": result["roi_cell_count"],
                "blocked_roi_cell_count": result["blocked_roi_cell_count"],
                "unsafe_roi_cell_count": result["unsafe_roi_cell_count"],
                "excluded_roi_cell_count": result["excluded_roi_cell_count"],
                "unreachable_roi_cell_count": result["unreachable_roi_cell_count"],
                "reachable_safe_cell_count": result["reachable_safe_cell_count"],
                "covered_reachable_safe_cell_count": result["covered_reachable_safe_cell_count"],
                "infeasible_reason_codes": result["infeasible_reason_codes"],
            }
            for result in scenario_results
        ],
    }


def _rejection_report(reason_codes: list[str], scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_failures = [
        {
            "scenario_id": result["scenario_id"],
            "reason_codes": result["reason_codes"],
            "infeasible_reason_codes": result["infeasible_reason_codes"],
        }
        for result in scenario_results
        if result["reason_codes"] or result["infeasible_reason_codes"]
    ]
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "scenario_failures": scenario_failures,
    }


def _manifest(
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "config": str(config_path),
        "output_root": str(output_root),
        "target_coverage_rate": config["target_coverage_rate"],
        "path_budget_m": config["path_budget_m"],
        "scenario_count": len(config["scenarios"]),
        "artifacts": {name: str(path) for name, path in paths.items()},
        "next_required_change": summary["next_required_change"],
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


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
        "path_budget_exhausted": result["path_budget_exhausted"],
        "coverage_denominator_valid": result["coverage_denominator_valid"],
        "coverage_ledger_complete": result["coverage_ledger_complete"],
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Global 99% Coverage Benchmark v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- target_coverage_rate: `{summary['target_coverage_rate']}`",
        f"- achieved_coverage_rate: `{summary['achieved_coverage_rate']}`",
        f"- reachable_safe_cell_count: `{summary['reachable_safe_cell_count']}`",
        f"- covered_reachable_safe_cell_count: `{summary['covered_reachable_safe_cell_count']}`",
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
                f"- reachable_safe_cell_count: `{scenario['reachable_safe_cell_count']}`",
                f"- covered_reachable_safe_cell_count: `{scenario['covered_reachable_safe_cell_count']}`",
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
            "This benchmark is a contract and deterministic fixture stage only. It does not train PPO, publish a checkpoint, replace default policy, connect a real executor, modify the policy network/action space/default A*, or claim Ackermann-feasible trajectories.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cell(value: Any, label: str) -> Cell:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError(f"{label} must be a two-item cell")
    return (_int(value[0], f"{label}[0]"), _int(value[1], f"{label}[1]"))


def _cell_in_bounds(cell: Cell, width: int, height: int) -> bool:
    x, y = cell
    return 0 <= x < width and 0 <= y < height


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    return float(value)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unique_sorted(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value})


if __name__ == "__main__":
    raise SystemExit(main())
