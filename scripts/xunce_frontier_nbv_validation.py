from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


DETERMINISTIC_VALIDATION_SOURCE = "deterministic_exact_cell_match/stage18a_path_feedback"
IN_PROCESS_VALIDATION_MODE = "in_process_evaluate_candidate_paths"
DETERMINISTIC_VALIDATION_MODE = "deterministic_exact_cell_match"
IN_PROCESS_VALIDATION_SOURCE = "in_process_evaluate_candidate_paths/v1"
MISSING_EXACT_CELL_MATCH_SOURCE = "missing_exact_cell_match"
SOURCE_CANDIDATE_UNVALIDATED = "source_candidate_not_path_feedback_validated"
SOURCE_CANDIDATE_METRIC_MISSING = "source_candidate_path_feedback_metric_missing"
VALIDATION_COMMAND_FAILED = "candidate_validation_command_failed"
VALIDATION_CONTRACT_MISSING = "candidate_validation_contract_missing"
VALIDATION_SIDECAR_MISSING = "candidate_validation_sidecar_missing"


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
    if validation_mode != IN_PROCESS_VALIDATION_MODE:
        raise ValueError(f"unknown proposal validation mode: {validation_mode}")
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
    )


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
        evaluations = _evaluate_prepared_proposals(
            validation_contract,
            sidecar_path=sidecar_path,
            repo_root=repo_root,
            output_work_root=output_work_root,
            scenario_id=str(scenario.get("scenario_id") or "scenario"),
            current_cell=current_cell,
            top_k=len(prepared) if top_k is None else top_k,
            debug_validation_artifacts=debug_validation_artifacts,
        )
    except Exception as exc:
        return [_validation_failure(row, VALIDATION_COMMAND_FAILED, message=str(exc)) for row in prepared]
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


def _validation_failure(row: dict[str, Any], reason: str, *, message: str | None = None) -> dict[str, Any]:
    failed = dict(row)
    failed["proposal_only"] = True
    failed["proposal_validated_by_path_feedback"] = False
    failed["path_feedback_validation_source"] = reason
    failed["planner_reachable"] = False
    failed["reachable"] = False
    failed["failure_reason"] = reason
    failed["replan_required"] = True
    failed["open_grid_fallback_used"] = False
    failed["validation_diagnostic_flags"] = [reason]
    failed["coverage_validated_by_path_feedback"] = False
    failed["coverage_validation_source"] = "offline_geometric_counterfactual_not_path_feedback"
    failed.setdefault("risk_source", "unavailable")
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
):
    output_work_root.mkdir(parents=True, exist_ok=True)
    if debug_validation_artifacts:
        output_dir = output_work_root / "_proposal_validation_runtime" / _safe_path_name(scenario_id)
        planner = _planner_for_validation(
            sidecar_path=sidecar_path,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        return _evaluate_candidate_paths(
            validation_contract,
            current_cell=current_cell,
            top_k=top_k,
            planner=planner,
            repo_root=repo_root,
        )
    with tempfile.TemporaryDirectory(prefix="xunce-proposal-validation-", dir=output_work_root) as output_dir_raw:
        planner = _planner_for_validation(
            sidecar_path=sidecar_path,
            repo_root=repo_root,
            output_dir=Path(output_dir_raw),
        )
        return _evaluate_candidate_paths(
            validation_contract,
            current_cell=current_cell,
            top_k=top_k,
            planner=planner,
            repo_root=repo_root,
        )


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
    return row


def _risk_source(candidate: dict[str, Any], proposal: dict[str, Any]) -> str:
    if _finite_float(candidate.get("risk")) is None:
        return "unavailable"
    generation = candidate.get("candidate_generation")
    if isinstance(generation, dict) and generation.get("risk_source"):
        return str(generation["risk_source"])
    if proposal.get("risk") is not None or proposal.get("path_risk_peak") is not None:
        return "goal_experimental_fallback"
    return "sidecar_cost_proxy_no_path_risk"


def _ensure_model_explorer_on_path(repo_root: Path) -> None:
    model_explorer_src = repo_root / "model-explorer" / "src"
    if str(model_explorer_src) not in sys.path:
        sys.path.insert(0, str(model_explorer_src))


def _safe_path_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value) or "scenario"


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
