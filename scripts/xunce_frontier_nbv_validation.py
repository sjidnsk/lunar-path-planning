from __future__ import annotations

import math
import sys
from heapq import heappop, heappush
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


DETERMINISTIC_VALIDATION_SOURCE = "deterministic_exact_cell_match/stage18a_path_feedback"
IN_PROCESS_VALIDATION_MODE = "in_process_evaluate_candidate_paths"
BATCH_ASTAR_VALIDATION_MODE = "in_process_path_planner_astar_batch"
DETERMINISTIC_VALIDATION_MODE = "deterministic_exact_cell_match"
IN_PROCESS_VALIDATION_SOURCE = "in_process_evaluate_candidate_paths/v1"
MISSING_EXACT_CELL_MATCH_SOURCE = "missing_exact_cell_match"
SOURCE_CANDIDATE_UNVALIDATED = "source_candidate_not_path_feedback_validated"
SOURCE_CANDIDATE_METRIC_MISSING = "source_candidate_path_feedback_metric_missing"
VALIDATION_COMMAND_FAILED = "candidate_validation_command_failed"
VALIDATION_CONTRACT_MISSING = "candidate_validation_contract_missing"
VALIDATION_SIDECAR_MISSING = "candidate_validation_sidecar_missing"
VALIDATION_PATH_LENGTH_PREFLIGHT_FAILED = "path_length_preflight_failed"
ADAPTER_FAILURE_REASON = "path_planner_route_adapter_failed"
PATH_PLANNER_ROUTE_ADAPTER_BACKEND = "path_planner_route_adapter"
SIDECAR_GRID_ASTAR_SCREENING_BACKEND = "sidecar_grid_astar_screening"
SIDECAR_GRID_ASTAR_DIAGNOSTIC_BACKEND = "sidecar_grid_astar_diagnostic"
SIDECAR_FALLBACK_DIAGNOSTIC_ONLY = "diagnostic_only"
SIDECAR_FALLBACK_FORMAL_SCREENING = "formal_screening"
FULL_PATH_PLANNER_ADAPTER_EVIDENCE_KIND = "full_path_planner_adapter"
IN_PROCESS_ASTAR_EVIDENCE_KIND = "in_process_astar_screening"
DETERMINISTIC_EXACT_CELL_MATCH_EVIDENCE_KIND = "deterministic_exact_cell_match"
VALIDATION_FAILURE_EVIDENCE_KIND = "validation_failure"


def validate_candidate_cells(
    *,
    scenario: dict[str, Any],
    proposal_rows: Iterable[dict[str, Any]],
    contract_path: Path | str,
    sidecar_path: Path | str,
    current_cell: tuple[int, int],
    repo_root: Path | str,
    output_work_root: Path | str,
    validation_mode: str = IN_PROCESS_VALIDATION_MODE,
    top_k: int | None = None,
    allow_open_grid_fallback: bool = False,
    debug_validation_artifacts: bool = False,
    max_validation_path_length: int = 180,
    sidecar_fallback_mode: str = SIDECAR_FALLBACK_DIAGNOSTIC_ONLY,
    validation_batch_hash: str | None = None,
    route_cache: dict[str, dict[str, Any]] | None = None,
    planner_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate arbitrary proposal cells with the project's path planner boundary.

    The default mode constructs an in-memory ModelExplorerContract whose goals are
    the proposal cells, calls evaluate_candidate_paths(), and copies only
    per-candidate planner evidence back onto the proposal rows. The temporary
    reachable=True goal flag is an evaluation seed, not the final reachability
    claim; final reachable is copied from the planner result.
    """

    proposals = [dict(row) for row in proposal_rows]
    if validation_mode == DETERMINISTIC_VALIDATION_MODE:
        return validate_frontier_nbv_proposals(
            scenario=scenario,
            proposal_rows=proposals,
            repo_root=repo_root,
            output_work_root=output_work_root,
        )
    if validation_mode == BATCH_ASTAR_VALIDATION_MODE:
        return _validate_candidate_cells_in_process_batch_astar(
            proposal_rows=proposals,
            contract_path=Path(contract_path),
            sidecar_path=Path(sidecar_path),
            current_cell=current_cell,
            repo_root=Path(repo_root),
            output_work_root=Path(output_work_root),
            validation_batch_hash=validation_batch_hash,
            route_cache=route_cache,
            planner_options=planner_options,
        )
    if validation_mode != IN_PROCESS_VALIDATION_MODE:
        raise ValueError(f"unknown proposal validation mode: {validation_mode}")
    if sidecar_fallback_mode not in {SIDECAR_FALLBACK_DIAGNOSTIC_ONLY, SIDECAR_FALLBACK_FORMAL_SCREENING}:
        raise ValueError(f"unknown sidecar fallback mode: {sidecar_fallback_mode}")
    return _validate_candidate_cells_in_process(
        scenario=scenario,
        proposal_rows=proposals,
        contract_path=Path(contract_path),
        sidecar_path=Path(sidecar_path),
        current_cell=current_cell,
        repo_root=Path(repo_root),
        output_work_root=Path(output_work_root),
        top_k=top_k,
        allow_open_grid_fallback=allow_open_grid_fallback,
        debug_validation_artifacts=debug_validation_artifacts,
        max_validation_path_length=max_validation_path_length,
        sidecar_fallback_mode=sidecar_fallback_mode,
        validation_batch_hash=validation_batch_hash,
    )


def _validate_candidate_cells_in_process_batch_astar(
    *,
    proposal_rows: list[dict[str, Any]],
    contract_path: Path,
    sidecar_path: Path,
    current_cell: tuple[int, int],
    repo_root: Path,
    output_work_root: Path,
    validation_batch_hash: str | None,
    route_cache: dict[str, dict[str, Any]] | None,
    planner_options: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    prepared = [_prepared_proposal(row) for row in proposal_rows]
    if not prepared:
        return []
    if not contract_path.is_file():
        return [_validation_failure(row, VALIDATION_CONTRACT_MISSING) for row in prepared]
    if not sidecar_path.is_file():
        return [_validation_failure(row, VALIDATION_SIDECAR_MISSING) for row in prepared]
    try:
        validation_contract = _contract_with_proposal_goals(
            contract_path=contract_path,
            proposal_rows=prepared,
            repo_root=repo_root,
        )
        try:
            from xunce_path_planner_astar_batch import evaluate_candidate_paths_with_in_process_astar_batch
        except ModuleNotFoundError:  # pragma: no cover
            from scripts.xunce_path_planner_astar_batch import evaluate_candidate_paths_with_in_process_astar_batch

        rows = evaluate_candidate_paths_with_in_process_astar_batch(
            contract=validation_contract,
            sidecar_path=sidecar_path,
            current_cell=current_cell,
            proposal_rows=prepared,
            repo_root=repo_root,
            route_cache=route_cache,
            planner_options=planner_options,
        )
        return [
            {
                **row,
                "validation_work_root": str(output_work_root),
                "validation_work_root_path_length": len(str(output_work_root.resolve())),
                "validation_batch_hash": validation_batch_hash or "",
                "path_length_gate_passed": True,
            }
            for row in rows
        ]
    except Exception as exc:
        return [_validation_failure(row, VALIDATION_COMMAND_FAILED, message=str(exc)) for row in prepared]


def validate_frontier_nbv_proposals(
    *,
    scenario: dict[str, Any],
    proposal_rows: Iterable[dict[str, Any]],
    repo_root: Path | str | None = None,
    output_work_root: Path | str | None = None,
    validation_source_candidates: Iterable[dict[str, Any]] | None = None,
    validation_command: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate generated frontier/NBV proposals against deterministic path-feedback rows.

    The v1 adapter deliberately performs no path planning. A proposal only becomes
    formal when its cell exactly matches an existing path-feedback candidate with
    explicit reachable, path_cost/cost, and risk/path_risk_peak metrics.
    """

    del repo_root, output_work_root, validation_command
    source_candidates = list(validation_source_candidates or _candidate_rows(scenario))
    source_by_cell = _source_candidates_by_cell(source_candidates)
    validated_rows: list[dict[str, Any]] = []
    for proposal in proposal_rows:
        row = dict(proposal)
        row["proposal_only"] = True
        row["proposal_validated_by_path_feedback"] = False
        cell = _cell_tuple(row.get("cell"))
        source = source_by_cell.get(cell) if cell is not None else None
        if source is None:
            row["path_feedback_validation_source"] = MISSING_EXACT_CELL_MATCH_SOURCE
            validated_rows.append(row)
            continue
        if source.get("proposal_validated_by_path_feedback", True) is False:
            row["path_feedback_validation_source"] = SOURCE_CANDIDATE_UNVALIDATED
            row["source_action_index"] = source.get("action_index")
            validated_rows.append(row)
            continue
        path_cost = _metric(source, "path_cost", "cost")
        risk = _metric(source, "risk", "path_risk_peak")
        if path_cost is None or risk is None or "reachable" not in source:
            row["path_feedback_validation_source"] = SOURCE_CANDIDATE_METRIC_MISSING
            row["source_action_index"] = source.get("action_index")
            validated_rows.append(row)
            continue
        row.update(
            {
                "proposal_only": False,
                "proposal_validated_by_path_feedback": True,
                "path_feedback_validation_source": DETERMINISTIC_VALIDATION_SOURCE,
                "planner_validation_backend": DETERMINISTIC_VALIDATION_MODE,
                "validation_evidence_kind": DETERMINISTIC_EXACT_CELL_MATCH_EVIDENCE_KIND,
                "source_action_index": source.get("action_index"),
                "reachable": source.get("reachable") is True,
                "path_cost": path_cost,
                "risk": risk,
                "open_grid_fallback_used": bool(source.get("open_grid_fallback_used", False)),
            }
        )
        validated_rows.append(row)
    return validated_rows


def _validate_candidate_cells_in_process(
    *,
    scenario: dict[str, Any],
    proposal_rows: list[dict[str, Any]],
    contract_path: Path,
    sidecar_path: Path,
    current_cell: tuple[int, int],
    repo_root: Path,
    output_work_root: Path,
    top_k: int | None,
    allow_open_grid_fallback: bool,
    debug_validation_artifacts: bool,
    max_validation_path_length: int,
    sidecar_fallback_mode: str,
    validation_batch_hash: str | None,
) -> list[dict[str, Any]]:
    if not proposal_rows:
        return []
    prepared = [_prepared_proposal(row) for row in proposal_rows]
    if not contract_path.is_file():
        return [_validation_failure(row, VALIDATION_CONTRACT_MISSING) for row in prepared]
    if not sidecar_path.is_file():
        return [_validation_failure(row, VALIDATION_SIDECAR_MISSING) for row in prepared]
    try:
        validation_contract = _contract_with_proposal_goals(
            contract_path=contract_path,
            proposal_rows=prepared,
            repo_root=repo_root,
        )
        evaluations, planner_backend, validation_metadata = _evaluate_prepared_proposals(
            validation_contract,
            sidecar_path=sidecar_path,
            repo_root=repo_root,
            output_work_root=output_work_root,
            scenario_id=str(scenario.get("scenario_id") or "scenario"),
            current_cell=current_cell,
            top_k=len(prepared) if top_k is None else top_k,
            debug_validation_artifacts=debug_validation_artifacts,
            max_validation_path_length=max_validation_path_length,
            sidecar_fallback_mode=sidecar_fallback_mode,
            validation_batch_hash=validation_batch_hash,
        )
    except Exception as exc:
        return [_validation_failure(row, VALIDATION_COMMAND_FAILED, message=str(exc)) for row in prepared]
    if evaluations is None:
        return [
            _validation_failure(
                row,
                str(validation_metadata.get("failure_reason") or ADAPTER_FAILURE_REASON),
                metadata=validation_metadata,
            )
            for row in prepared
        ]
    by_action = {int(evaluation.action_index): evaluation.to_dict() for evaluation in evaluations}
    results: list[dict[str, Any]] = []
    for action_index, proposal in enumerate(prepared):
        candidate = by_action.get(action_index)
        if candidate is None:
            results.append(_validation_failure(proposal, "proposal_not_evaluated_by_planner"))
            continue
        results.append(
            _validated_from_evaluation(
                proposal,
                candidate,
                allow_open_grid_fallback=allow_open_grid_fallback,
                planner_validation_backend=planner_backend,
                validation_metadata=validation_metadata,
            )
        )
    return results


def _prepared_proposal(row: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(row)
    prepared["proposal_only"] = True
    prepared["proposal_validated_by_path_feedback"] = False
    prepared["validation_attempt_goal_reachable_seed"] = True
    prepared["coverage_validated_by_path_feedback"] = False
    prepared["coverage_validation_source"] = "offline_geometric_counterfactual_not_path_feedback"
    return prepared


def _validation_failure(
    row: dict[str, Any],
    reason: str,
    *,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed = dict(row)
    failed["proposal_only"] = True
    failed["proposal_validated_by_path_feedback"] = False
    failed["path_feedback_validation_source"] = reason
    failed["planner_validation_backend"] = failed.get("planner_validation_backend", "")
    failed["validation_evidence_kind"] = VALIDATION_FAILURE_EVIDENCE_KIND
    failed["planner_reachable"] = False
    failed["reachable"] = False
    failed["failure_reason"] = reason
    failed["replan_required"] = True
    failed["open_grid_fallback_used"] = False
    failed["validation_diagnostic_flags"] = [reason]
    failed["coverage_validated_by_path_feedback"] = False
    failed["coverage_validation_source"] = "offline_geometric_counterfactual_not_path_feedback"
    failed.setdefault("risk_source", "unavailable")
    if metadata:
        failed.update(metadata)
        failed["failure_reason"] = reason
        flags = set(str(item) for item in failed.get("validation_diagnostic_flags", []))
        flags.add(reason)
        failed["validation_diagnostic_flags"] = sorted(flags)
    if message:
        failed["path_feedback_validation_error"] = message[-2000:]
    return failed


def _contract_with_proposal_goals(
    *,
    contract_path: Path,
    proposal_rows: list[dict[str, Any]],
    repo_root: Path,
):
    _ensure_model_explorer_on_path(repo_root)
    from model_explorer.core.interfaces import GoalCandidate
    from model_explorer.io.scenario import load_scenario

    contract = load_scenario(contract_path).snapshots[0]
    goals = tuple(_proposal_to_goal_candidate(row, GoalCandidate) for row in proposal_rows)
    return replace(contract, top_goals=goals)


def _proposal_to_goal_candidate(row: dict[str, Any], goal_candidate_type):
    cell = _cell_tuple(row.get("cell"))
    if cell is None:
        cell = (0, 0)
    utility = _first_finite_float(
        row.get("roi_weighted_coverage_delta"),
        row.get("expected_coverage_rate_delta"),
        row.get("utility"),
        default=1.0,
    )
    experimental = {
        key: value
        for key, value in row.items()
        if key not in {"cell", "utility", "reachable"}
    }
    risk, risk_source = _proposal_risk_and_source(row)
    experimental["risk"] = risk
    experimental["risk_source"] = risk_source
    experimental["proposal_validation_source"] = IN_PROCESS_VALIDATION_SOURCE
    experimental["validation_attempt_goal_reachable_seed"] = True
    return goal_candidate_type(cell=cell, utility=utility, reachable=True, experimental=experimental)


def _proposal_risk_and_source(row: dict[str, Any]) -> tuple[float, str]:
    explicit = _finite_float(row.get("risk"))
    if explicit is not None:
        return explicit, "goal_experimental_fallback"
    for key in ("path_risk_peak", "candidate_point_risk"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value, "goal_experimental_fallback"
    return 0.0, "sidecar_cost_proxy_no_path_risk"


def _evaluate_prepared_proposals(
    validation_contract,
    *,
    sidecar_path: Path,
    repo_root: Path,
    output_work_root: Path,
    scenario_id: str,
    current_cell: tuple[int, int],
    top_k: int,
    debug_validation_artifacts: bool,
    max_validation_path_length: int,
    sidecar_fallback_mode: str,
    validation_batch_hash: str | None,
):
    output_work_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_work_root / "_proposal_validation_runtime" / _safe_path_name(scenario_id) if debug_validation_artifacts else output_work_root
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _validation_path_length_preflight(
        output_dir=output_dir,
        max_validation_path_length=max_validation_path_length,
        validation_batch_hash=validation_batch_hash,
    )
    if not preflight["path_length_gate_passed"]:
        return None, PATH_PLANNER_ROUTE_ADAPTER_BACKEND, {
            **preflight,
            "planner_validation_backend": PATH_PLANNER_ROUTE_ADAPTER_BACKEND,
            "proposal_validated_by_path_feedback": False,
            "proposal_only": True,
            "failure_reason": VALIDATION_PATH_LENGTH_PREFLIGHT_FAILED,
            "adapter_failure_reason": VALIDATION_PATH_LENGTH_PREFLIGHT_FAILED,
            "path_planner_adapter_audit_status": "not_attempted_path_length_preflight_failed",
        }

    planner = _planner_for_validation(
        sidecar_path=sidecar_path,
        repo_root=repo_root,
        output_dir=output_dir,
    )
    evaluations = _evaluate_candidate_paths(
        validation_contract,
        current_cell=current_cell,
        top_k=top_k,
        planner=planner,
        repo_root=repo_root,
    )
    if _all_path_planner_adapter_failed(evaluations):
        adapter_metadata = {
            **preflight,
            **_adapter_failure_metadata(evaluations),
            "planner_validation_backend": PATH_PLANNER_ROUTE_ADAPTER_BACKEND,
            "proposal_validated_by_path_feedback": False,
            "proposal_only": True,
            "failure_reason": ADAPTER_FAILURE_REASON,
            "path_planner_adapter_audit_status": "failed",
        }
        if sidecar_fallback_mode == SIDECAR_FALLBACK_FORMAL_SCREENING:
            sidecar_evaluations = _evaluate_candidate_paths(
                validation_contract,
                current_cell=current_cell,
                top_k=top_k,
                planner=_sidecar_grid_astar_planner(
                    sidecar_path=sidecar_path,
                    repo_root=repo_root,
                    backend_name=SIDECAR_GRID_ASTAR_SCREENING_BACKEND,
                ),
                repo_root=repo_root,
            )
            return sidecar_evaluations, SIDECAR_GRID_ASTAR_SCREENING_BACKEND, adapter_metadata
        return None, PATH_PLANNER_ROUTE_ADAPTER_BACKEND, adapter_metadata
    return evaluations, PATH_PLANNER_ROUTE_ADAPTER_BACKEND, {
        **preflight,
        "planner_validation_backend": PATH_PLANNER_ROUTE_ADAPTER_BACKEND,
        "path_planner_adapter_audit_status": "passed",
    }


def _planner_for_validation(
    *,
    sidecar_path: Path,
    repo_root: Path,
    output_dir: Path,
):
    _ensure_model_explorer_on_path(repo_root)
    from model_explorer.policy.planning import PathPlannerRouteAdapter, load_path_planner_sidecar

    return PathPlannerRouteAdapter(
        path_planner_root=repo_root / "path-planner",
        python_executable=sys.executable,
        output_dir=output_dir,
        sidecar=load_path_planner_sidecar(sidecar_path),
    )


def _sidecar_grid_astar_planner(*, sidecar_path: Path, repo_root: Path, backend_name: str = SIDECAR_GRID_ASTAR_DIAGNOSTIC_BACKEND):
    _ensure_model_explorer_on_path(repo_root)
    from model_explorer.policy.planning import load_path_planner_sidecar

    sidecar = load_path_planner_sidecar(sidecar_path)
    return _SidecarGridAStarPlanner(
        cost=sidecar["cost"],
        passable_mask=sidecar["passable_mask"],
        sidecar_path=sidecar_path,
        backend_name=backend_name,
    )


class _SidecarGridAStarPlanner:
    def __init__(self, *, cost: list[list[Any]], passable_mask: list[list[Any]], sidecar_path: Path, backend_name: str) -> None:
        self._cost = [[_finite_float(value) or 1.0 for value in row] for row in cost]
        self._passable = [[bool(value) for value in row] for row in passable_mask]
        self._sidecar_path = sidecar_path
        self._backend_name = backend_name

    def plan(self, request):
        from model_explorer.policy.planning import PathPlanResult

        start = request.current_cell
        goal = request.selected_goal.cell
        metadata = {
            "planner": self._backend_name,
            "request_payload": {
                "metadata": {
                    "cost_source": "sidecar_cost",
                    "passable_mask_source": "sidecar_passable_mask",
                    "sidecar": {"path": str(self._sidecar_path)},
                }
            },
        }
        if not self._is_passable(start) or not self._is_passable(goal):
            return PathPlanResult(
                feasible=False,
                risk=_goal_risk(request.selected_goal),
                failure_reason="path_blocked",
                replan_required=True,
                metadata=metadata,
            )
        path = self._find_path(start, goal)
        if path is None:
            return PathPlanResult(
                feasible=False,
                risk=_goal_risk(request.selected_goal),
                failure_reason="path_blocked",
                replan_required=True,
                metadata=metadata,
            )
        resolution = float(getattr(request.contract.grid, "resolution", 1.0) or 1.0)
        path_length = float(max(len(path) - 1, 0)) * resolution
        path_cost = sum(self._cost[y][x] for x, y in path[1:]) * resolution
        metadata["path_cells"] = [[x, y] for x, y in path]
        return PathPlanResult(
            feasible=True,
            path_cost=float(path_cost),
            path_length=path_length,
            risk=_goal_risk(request.selected_goal),
            metadata=metadata,
        )

    def _find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        frontier: list[tuple[float, int, tuple[int, int]]] = []
        heappush(frontier, (_manhattan(start, goal), 0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], int] = {start: 0}
        while frontier:
            _priority, current_cost, current = heappop(frontier)
            if current == goal:
                return _reconstruct_path(came_from, current)
            for neighbor in self._neighbors(current):
                new_cost = current_cost + 1
                if neighbor in cost_so_far and new_cost >= cost_so_far[neighbor]:
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heappush(frontier, (float(new_cost + _manhattan(neighbor, goal)), new_cost, neighbor))
        return None

    def _neighbors(self, cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        x, y = cell
        candidates = ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
        return tuple(candidate for candidate in candidates if self._is_passable(candidate))

    def _is_passable(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= y < len(self._passable) and 0 <= x < len(self._passable[y]) and self._passable[y][x]


def _all_path_planner_adapter_failed(evaluations: Any) -> bool:
    rows = list(evaluations or [])
    return bool(rows) and all(getattr(getattr(row, "result", None), "failure_reason", None) == "path_planner_adapter_failed" for row in rows)


def _validation_path_length_preflight(
    *,
    output_dir: Path,
    max_validation_path_length: int,
    validation_batch_hash: str | None,
) -> dict[str, Any]:
    expected_paths = {
        "path-planner-request.json": output_dir / "path-planner-request.json",
        "path-planner-route.json": output_dir / "path-planner-route.json",
        "diagnostics": output_dir / "diagnostics",
        "diagnostics/diagnostics.png": output_dir / "diagnostics" / "diagnostics.png",
        "diagnostics/diagnostics.html": output_dir / "diagnostics" / "diagnostics.html",
    }
    path_lengths = {key: len(str(path.resolve())) for key, path in expected_paths.items()}
    max_length = int(max_validation_path_length)
    max_observed = max(path_lengths.values()) if path_lengths else 0
    failing = {key: value for key, value in path_lengths.items() if max_length > 0 and value > max_length}
    return {
        "validation_work_root": str(output_dir),
        "validation_work_root_path_length": len(str(output_dir.resolve())),
        "validation_batch_hash": validation_batch_hash or "",
        "path_length_gate_passed": not failing,
        "max_validation_path_length": max_length,
        "validation_path_lengths": path_lengths,
        "validation_max_path_length_observed": max_observed,
        "path_length_preflight_failure_paths": sorted(failing),
    }


def _adapter_failure_metadata(evaluations: Any) -> dict[str, Any]:
    rows = list(evaluations or [])
    error_types: list[str] = []
    messages: list[str] = []
    failure_reasons: list[str] = []
    for row in rows:
        result = getattr(row, "result", None)
        failure_reason = getattr(result, "failure_reason", None)
        if failure_reason:
            failure_reasons.append(str(failure_reason))
        metadata = getattr(result, "metadata", {}) if result is not None else {}
        if isinstance(metadata, dict):
            if metadata.get("error_type"):
                error_types.append(str(metadata["error_type"]))
            if metadata.get("message"):
                messages.append(str(metadata["message"]))
    return {
        "adapter_error_type": error_types[0] if error_types else "unknown",
        "adapter_error_message_tail": _tail(messages[0]) if messages else "",
        "adapter_failure_reason": failure_reasons[0] if failure_reasons else ADAPTER_FAILURE_REASON,
        "adapter_error_type_counts": _count_values(error_types),
    }


def _reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int] | None], current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _goal_risk(goal: Any) -> float:
    experimental = getattr(goal, "experimental", {})
    if isinstance(experimental, dict):
        risk = _finite_float(experimental.get("risk"))
        if risk is not None:
            return risk
    return 0.0


def _evaluate_candidate_paths(contract, *, current_cell: tuple[int, int], top_k: int, planner, repo_root: Path):
    _ensure_model_explorer_on_path(repo_root)
    from model_explorer.policy.planning import evaluate_candidate_paths

    return evaluate_candidate_paths(
        contract,
        current_cell=current_cell,
        top_k=max(0, int(top_k)),
        planner=planner,
    )


def _validated_from_evaluation(
    proposal: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_open_grid_fallback: bool,
    planner_validation_backend: str,
    validation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(proposal)
    reachable = bool(candidate.get("reachable"))
    open_grid_fallback = bool(candidate.get("open_grid_fallback_used"))
    path_cost = _finite_float(candidate.get("path_cost"))
    path_length = _finite_float(candidate.get("path_length"))
    risk = _finite_float(candidate.get("risk"))
    diagnostics = candidate.get("diagnostic_interpretation")
    diagnostic_flags = []
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("diagnostic_flags"), list):
        diagnostic_flags = [str(item) for item in diagnostics["diagnostic_flags"]]
    if not reachable and "proposal_unreachable" not in diagnostic_flags:
        diagnostic_flags.append("proposal_unreachable")
    if open_grid_fallback and "proposal_open_grid_fallback" not in diagnostic_flags:
        diagnostic_flags.append("proposal_open_grid_fallback")
    if planner_validation_backend == SIDECAR_GRID_ASTAR_SCREENING_BACKEND and "sidecar_grid_astar_screening_used" not in diagnostic_flags:
        diagnostic_flags.append("sidecar_grid_astar_screening_used")
    is_formal = (
        reachable
        and path_cost is not None
        and path_length is not None
        and risk is not None
        and (allow_open_grid_fallback or not open_grid_fallback)
    )
    row.update(
        {
            "proposal_only": not is_formal,
            "proposal_validated_by_path_feedback": True,
            "path_feedback_validation_source": IN_PROCESS_VALIDATION_SOURCE,
            "planner_validation_backend": planner_validation_backend,
            "validation_evidence_kind": _validation_evidence_kind(planner_validation_backend, validation_metadata, is_formal),
            "planner_reachable": reachable,
            "reachable": reachable,
            "failure_reason": candidate.get("failure_reason"),
            "replan_required": bool(candidate.get("replan_required")),
            "open_grid_fallback_used": open_grid_fallback,
            "validation_diagnostic_flags": diagnostic_flags,
        }
    )
    if path_cost is not None:
        row["path_cost"] = path_cost
    if path_length is not None:
        row["path_length"] = path_length
    if risk is not None:
        row["risk"] = risk
    row["risk_source"] = _risk_source(candidate, proposal)
    if validation_metadata:
        for key in (
            "validation_work_root",
            "validation_work_root_path_length",
            "validation_batch_hash",
            "path_length_gate_passed",
            "max_validation_path_length",
            "validation_path_lengths",
            "validation_max_path_length_observed",
            "path_length_preflight_failure_paths",
            "path_planner_adapter_audit_status",
            "adapter_error_type",
            "adapter_error_message_tail",
            "adapter_failure_reason",
        ):
            if key in validation_metadata:
                row[key] = validation_metadata[key]
    return row


def _validation_evidence_kind(
    planner_validation_backend: str,
    validation_metadata: dict[str, Any] | None,
    is_formal: bool,
) -> str:
    if not is_formal:
        return VALIDATION_FAILURE_EVIDENCE_KIND
    if planner_validation_backend == PATH_PLANNER_ROUTE_ADAPTER_BACKEND and (validation_metadata or {}).get("path_planner_adapter_audit_status") == "passed":
        return FULL_PATH_PLANNER_ADAPTER_EVIDENCE_KIND
    if planner_validation_backend == SIDECAR_GRID_ASTAR_SCREENING_BACKEND:
        return IN_PROCESS_ASTAR_EVIDENCE_KIND
    if planner_validation_backend == DETERMINISTIC_VALIDATION_MODE:
        return DETERMINISTIC_EXACT_CELL_MATCH_EVIDENCE_KIND
    return VALIDATION_FAILURE_EVIDENCE_KIND


def _risk_source(candidate: dict[str, Any], proposal: dict[str, Any]) -> str:
    if _finite_float(candidate.get("risk")) is None:
        return "unavailable"
    if candidate.get("risk_source"):
        return str(candidate["risk_source"])
    generation = candidate.get("candidate_generation")
    if isinstance(generation, dict) and generation.get("risk_source"):
        return str(generation["risk_source"])
    if isinstance(generation, dict) and generation.get("risk_provenance_source"):
        return str(generation["risk_provenance_source"])
    if proposal.get("risk_source"):
        return str(proposal["risk_source"])
    if proposal.get("risk") is not None or proposal.get("path_risk_peak") is not None:
        return "goal_experimental_fallback"
    return "sidecar_cost_proxy_no_path_risk"


def _ensure_model_explorer_on_path(repo_root: Path) -> None:
    model_explorer_src = repo_root / "model-explorer" / "src"
    if str(model_explorer_src) not in sys.path:
        sys.path.insert(0, str(model_explorer_src))


def _safe_path_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value) or "scenario"


def _tail(value: str, limit: int = 500) -> str:
    return value[-limit:] if len(value) > limit else value


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = feedback.get("candidates") if isinstance(feedback.get("candidates"), list) else []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _source_candidates_by_cell(candidates: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        cell = _cell_tuple(candidate.get("cell"))
        if cell is not None and cell not in indexed:
            indexed[cell] = candidate
    return indexed


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None


def _metric(row: dict[str, Any], primary: str, secondary: str) -> float | None:
    for key in (primary, secondary):
        if key not in row:
            continue
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if value == value and value not in (float("inf"), float("-inf")):
            return value
    return None


def _first_finite_float(*values: Any, default: float) -> float:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return float(default)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
