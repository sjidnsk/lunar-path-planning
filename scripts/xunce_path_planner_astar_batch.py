from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np


BATCH_ASTAR_VALIDATION_MODE = "in_process_path_planner_astar_batch"
BATCH_ASTAR_VALIDATION_SOURCE = "in_process_path_planner_astar_batch/v1"
BATCH_ASTAR_BACKEND = "in_process_path_planner_astar_batch"
IN_PROCESS_ASTAR_EVIDENCE_KIND = "in_process_astar_screening"
VALIDATION_FAILURE_EVIDENCE_KIND = "validation_failure"


def evaluate_candidate_paths_with_in_process_astar_batch(
    *,
    contract: Any,
    sidecar_path: Path | str,
    current_cell: tuple[int, int],
    proposal_rows: list[dict[str, Any]],
    repo_root: Path | str,
    route_cache: MutableMapping[str, dict[str, Any]] | None = None,
    planner_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate proposal cells by calling path-planner's AStarPlanner in-process.

    This intentionally stops at A* route screening. It does not run the
    path-planner CLI postprocess stack and therefore must not be reported as
    full PathPlannerRouteAdapter evidence.
    """

    repo_root = Path(repo_root)
    sidecar_path = Path(sidecar_path)
    planner_options = dict(planner_options or {})
    route_cache = route_cache if route_cache is not None else {}
    _ensure_path_planner_on_path(repo_root)

    from path_planner.core import Cell, CostGrid, GridSpec, NeighborPolicy, PlanRequest
    from path_planner.search import AStarPlanner, build_planning_grid

    sidecar = _load_sidecar(sidecar_path)
    grid_spec = _grid_spec_from_contract(contract, GridSpec)
    cost_grid = CostGrid(
        spec=grid_spec,
        cost=np.asarray(sidecar["cost"], dtype=float),
        passable_mask=np.asarray(sidecar["passable_mask"], dtype=bool),
        metadata={"source": str(sidecar_path), "adapter": BATCH_ASTAR_BACKEND},
    )
    planning_grid = build_planning_grid(cost_grid, platform_profile=None)
    planner = AStarPlanner()
    neighbor_policy = _neighbor_policy(planner_options.get("neighbor_policy", "8-neighbor"), NeighborPolicy)
    prevent_corner_cutting = bool(planner_options.get("prevent_corner_cutting", True))
    max_iterations = _positive_int(planner_options.get("max_iterations"), default=100_000)
    sidecar_digest = _sidecar_digest(sidecar_path)
    rows: list[dict[str, Any]] = []

    for proposal in proposal_rows:
        row = dict(proposal)
        cell = _cell_tuple(row.get("cell"))
        if cell is None:
            rows.append(_failure_row(row, "invalid_candidate_cell", route_cache_hit=False))
            continue
        route_key = _route_cache_key(
            sidecar_digest=sidecar_digest,
            grid_spec=grid_spec,
            start=current_cell,
            goal=cell,
            neighbor_policy=neighbor_policy.value,
            prevent_corner_cutting=prevent_corner_cutting,
            max_iterations=max_iterations,
        )
        cache_hit = route_key in route_cache
        if cache_hit:
            route_payload = dict(route_cache[route_key])
        else:
            result = planner.plan(
                planning_grid,
                PlanRequest(
                    start=Cell(int(current_cell[0]), int(current_cell[1])),
                    goal=Cell(int(cell[0]), int(cell[1])),
                    neighbor_policy=neighbor_policy,
                    prevent_corner_cutting=prevent_corner_cutting,
                    max_iterations=max_iterations,
                ),
            )
            route_payload = _route_payload_from_result(result, risk=_proposal_risk(row), risk_source=_proposal_risk_source(row))
            route_cache[route_key] = dict(route_payload)
        rows.append(_row_from_route_payload(row, route_payload, route_cache_hit=cache_hit))
    return rows


def _row_from_route_payload(row: dict[str, Any], route: dict[str, Any], *, route_cache_hit: bool) -> dict[str, Any]:
    reachable = bool(route.get("reachable"))
    path_cost = _finite_float(route.get("path_cost"))
    path_length = _finite_float(route.get("path_length"))
    risk = _finite_float(route.get("risk"))
    open_grid_fallback = bool(route.get("open_grid_fallback_used", False))
    formal = reachable and path_cost is not None and path_length is not None and risk is not None and not open_grid_fallback
    flags = []
    if not reachable:
        flags.append("proposal_unreachable")
    if open_grid_fallback:
        flags.append("proposal_open_grid_fallback")
    failure_reason = route.get("failure_reason")
    if failure_reason and str(failure_reason) not in flags:
        flags.append(str(failure_reason))
    result = dict(row)
    result.update(
        {
            "proposal_only": not formal,
            "proposal_validated_by_path_feedback": True,
            "path_feedback_validation_source": BATCH_ASTAR_VALIDATION_SOURCE,
            "planner_validation_backend": BATCH_ASTAR_BACKEND,
            "validation_evidence_kind": IN_PROCESS_ASTAR_EVIDENCE_KIND if formal else VALIDATION_FAILURE_EVIDENCE_KIND,
            "path_planner_adapter_audit_status": "not_applicable_in_process_astar_screening",
            "planner_reachable": reachable,
            "reachable": reachable,
            "failure_reason": None if formal else failure_reason,
            "replan_required": not formal,
            "open_grid_fallback_used": open_grid_fallback,
            "validation_diagnostic_flags": flags,
            "coverage_validated_by_path_feedback": False,
            "coverage_validation_source": "offline_geometric_counterfactual_not_path_feedback",
            "route_cache_hit": bool(route_cache_hit),
        }
    )
    if path_cost is not None:
        result["path_cost"] = path_cost
    if path_length is not None:
        result["path_length"] = path_length
    if risk is not None:
        result["risk"] = risk
    result["risk_source"] = str(route.get("risk_source") or "sidecar_cost_proxy_no_path_risk")
    return result


def _route_payload_from_result(result: Any, *, risk: float, risk_source: str) -> dict[str, Any]:
    failure_reason = getattr(result, "failure_reason", None)
    diagnostics = getattr(result, "diagnostics", None)
    path_length = getattr(diagnostics, "path_length_m", None)
    success = bool(getattr(result, "success", False))
    payload = {
        "reachable": success,
        "path_cost": float(getattr(result, "total_cost", 0.0)) if success else None,
        "path_length": float(path_length) if path_length is not None and math.isfinite(float(path_length)) else None,
        "risk": risk,
        "risk_source": risk_source,
        "failure_reason": None if success else _failure_reason_value(failure_reason),
        "open_grid_fallback_used": False,
    }
    return payload


def _failure_row(row: dict[str, Any], reason: str, *, route_cache_hit: bool) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "proposal_only": True,
            "proposal_validated_by_path_feedback": False,
            "path_feedback_validation_source": BATCH_ASTAR_VALIDATION_SOURCE,
            "planner_validation_backend": BATCH_ASTAR_BACKEND,
            "validation_evidence_kind": VALIDATION_FAILURE_EVIDENCE_KIND,
            "planner_reachable": False,
            "reachable": False,
            "failure_reason": reason,
            "replan_required": True,
            "open_grid_fallback_used": False,
            "validation_diagnostic_flags": [reason],
            "coverage_validated_by_path_feedback": False,
            "coverage_validation_source": "offline_geometric_counterfactual_not_path_feedback",
            "risk_source": "unavailable",
            "route_cache_hit": bool(route_cache_hit),
        }
    )
    return result


def _load_sidecar(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "path-planner-sidecar/v1":
        raise ValueError("path planner sidecar schema_version must be path-planner-sidecar/v1")
    if "cost" not in payload or "passable_mask" not in payload:
        raise ValueError("path planner sidecar must contain cost and passable_mask")
    return payload


def _grid_spec_from_contract(contract: Any, grid_spec_type: Any):
    grid = getattr(contract, "grid")
    return grid_spec_type(
        width=int(grid.width),
        height=int(grid.height),
        resolution=float(grid.resolution),
        origin=(float(grid.origin[0]), float(grid.origin[1])),
        frame_id=str(grid.frame_id),
    )


def _neighbor_policy(value: Any, neighbor_policy_type: Any):
    text = str(value or "8-neighbor")
    return neighbor_policy_type.FOUR if text == "4-neighbor" else neighbor_policy_type.EIGHT


def _route_cache_key(
    *,
    sidecar_digest: str,
    grid_spec: Any,
    start: tuple[int, int],
    goal: tuple[int, int],
    neighbor_policy: str,
    prevent_corner_cutting: bool,
    max_iterations: int,
) -> str:
    payload = {
        "backend": BATCH_ASTAR_BACKEND,
        "sidecar_digest": sidecar_digest,
        "grid": {
            "width": grid_spec.width,
            "height": grid_spec.height,
            "resolution": grid_spec.resolution,
            "origin": list(grid_spec.origin),
            "frame_id": grid_spec.frame_id,
        },
        "start": list(start),
        "goal": list(goal),
        "neighbor_policy": neighbor_policy,
        "prevent_corner_cutting": bool(prevent_corner_cutting),
        "max_iterations": int(max_iterations),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _sidecar_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proposal_risk(row: dict[str, Any]) -> float:
    for key in ("risk", "path_risk_peak", "candidate_point_risk"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return 0.0


def _proposal_risk_source(row: dict[str, Any]) -> str:
    for key in ("risk", "path_risk_peak", "candidate_point_risk"):
        if _finite_float(row.get(key)) is not None:
            return "goal_experimental_fallback"
    return "sidecar_cost_proxy_no_path_risk"


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _failure_reason_value(value: Any) -> str:
    if value is None:
        return "unreachable"
    return str(getattr(value, "value", value))


def _ensure_path_planner_on_path(repo_root: Path) -> None:
    path_planner_src = repo_root / "path-planner" / "src"
    if str(path_planner_src) not in sys.path:
        sys.path.insert(0, str(path_planner_src))
