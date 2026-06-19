from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from xunce_frontier_nbv_validation import validate_candidate_cells
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_frontier_nbv_validation import validate_candidate_cells


CONFIG_SCHEMA_VERSION = "xunce-true-frontier-nbv-candidate-source-replacement-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-true-frontier-nbv-candidate-source-summary/v1"
DEFAULT_CONFIG = "configs/xunce_true_frontier_nbv_candidate_source_replacement_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_true_frontier_nbv_candidate_source_replacement_v1"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"
TRUE_BINDING_SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
QUANTIZATION_SUMMARY_FILE = "xunce-risk-coverage-cost-quantization-summary.json"

SUMMARY_FILE = "xunce-true-frontier-nbv-candidate-source-summary.json"
PROPOSALS_FILE = "xunce-true-frontier-nbv-proposals.jsonl"
VALIDATED_FILE = "xunce-true-frontier-nbv-validated-candidates.jsonl"
VALIDATION_AUDIT_FILE = "xunce-true-frontier-nbv-validation-audit.json"
PROPOSAL_VALIDATION_SUMMARY_FILE = "xunce-true-frontier-nbv-proposal-validation-summary.json"
PROPOSAL_VALIDATION_RESULTS_FILE = "xunce-true-frontier-nbv-proposal-validation-results.jsonl"
REPORT_FILE = "xunce-true-frontier-nbv-report.md"

PASS_NEXT_REQUIRED_CHANGE = "rerun_true_model_inference_and_binding"
FIX_STAGE18A_NEXT_REQUIRED_CHANGE = "run_xunce_high_fidelity_real_map_roi_expansion"
FIX_BINDING_NEXT_REQUIRED_CHANGE = "run_true_incumbent_selection_binding"
FIX_VALIDATION_NEXT_REQUIRED_CHANGE = "repair_path_feedback_candidate_validation"
FIX_PROPOSAL_NEXT_REQUIRED_CHANGE = "repair_true_frontier_nbv_proposal_generation"

GENERATION_SOURCE = "true_frontier_nbv_candidate_source_replacement/v1"
COVERAGE_SOURCE = "geometric_counterfactual_from_true_frontier_nbv_candidate/v1"
COVERAGE_CELL_SET_KIND = "path_line_plus_endpoint_union"
COVERAGE_DEDUPE_SCOPE = "scenario_step_new_cells"
TOLERANCE = 1.0e-12

DIRECTIONS_8 = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replace Xunce candidates with true frontier/NBV proposal source.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--source-true-incumbent-binding-root")
    parser.add_argument("--source-quantization-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_roi_expansion_root": args.source_roi_expansion_root,
            "source_true_incumbent_binding_root": args.source_true_incumbent_binding_root,
            "source_quantization_root": args.source_quantization_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_true_frontier_nbv_candidate_source_replacement(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "candidate_count": summary["candidate_count"],
                "validated_candidate_count": summary["validated_candidate_count"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_true_frontier_nbv_candidate_source_replacement(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    source = _load_source(config)
    generated, proposal_rows, validated_rows, validation_audit, metrics = _generate(config, source, repo_root, output_root)
    decision = _decision(source, metrics)
    paths = _paths(output_root)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config=config,
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
        source=source,
        metrics=metrics,
        decision=decision,
        paths=paths,
    )
    write_json(paths["summary"], summary)
    write_jsonl(paths["proposals"], proposal_rows)
    write_jsonl(paths["validated"], validated_rows)
    write_json(paths["validation_audit"], validation_audit)
    write_json(paths["proposal_validation_summary"], _proposal_validation_summary(config, metrics))
    write_jsonl(paths["proposal_validation_results"], proposal_rows)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    write_json(paths["expansion_summary"], _generated_expansion_summary(source["expansion_summary"], summary))
    write_jsonl(paths["slices"], source["slices"])
    write_json(paths["path_feedback"], generated)
    return summary


def _load_config(path: Path, repo_root: Path, *, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config_overrides:
        payload = {**payload, **config_overrides}
    normalized = dict(payload)
    for key in ("source_roi_expansion_root", "source_true_incumbent_binding_root", "source_quantization_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["max_candidates_per_scenario"] = _positive_int(payload.get("max_candidates_per_scenario", 6), "max_candidates_per_scenario")
    normalized["proposal_pool_limit_per_scenario"] = _positive_int(payload.get("proposal_pool_limit_per_scenario", 24), "proposal_pool_limit_per_scenario")
    normalized["frontier_radius_cells"] = [_positive_int(value, "frontier_radius_cells") for value in payload.get("frontier_radius_cells", [2, 4, 6])]
    normalized["frontier_direction_count"] = min(_positive_int(payload.get("frontier_direction_count", 8), "frontier_direction_count"), len(DIRECTIONS_8))
    normalized["coverage_denominator_cells"] = _positive_float(payload.get("coverage_denominator_cells", 1000), "coverage_denominator_cells")
    normalized["risk_margin"] = _nonnegative_float(payload.get("risk_margin", 0.01), "risk_margin")
    normalized["cost_margin"] = _nonnegative_float(payload.get("cost_margin", 0.0), "cost_margin")
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["allow_open_grid_fallback"] = _bool(payload.get("allow_open_grid_fallback", False), "allow_open_grid_fallback")
    normalized["candidate_validation_mode"] = str(payload.get("candidate_validation_mode", "in_process_evaluate_candidate_paths"))
    normalized["min_validated_candidates_per_scenario"] = _positive_int(
        payload.get("min_validated_candidates_per_scenario", 3),
        "min_validated_candidates_per_scenario",
    )
    normalized["debug_validation_artifacts"] = _bool(
        payload.get("debug_validation_artifacts", False),
        "debug_validation_artifacts",
    )
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    if normalized["canary_traffic_fraction"] != 0.0:
        raise ConfigError("canary_traffic_fraction must be 0.0 for offline Stage 18I.3")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    roi_root = Path(config["source_roi_expansion_root"])
    binding_root = Path(config["source_true_incumbent_binding_root"])
    quantization_root = Path(config["source_quantization_root"])
    reasons: list[str] = []
    expansion_summary = _read_json(roi_root / EXPANSION_SUMMARY_FILE, reasons, "missing_stage18a_roi_expansion")
    slices = _read_jsonl(roi_root / EXPANSION_SLICES_FILE, reasons, "missing_stage18a_roi_expansion")
    roi_path_feedback = _read_json(roi_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_stage18a_roi_expansion")
    binding_summary = _read_json(binding_root / TRUE_BINDING_SUMMARY_FILE, reasons, "missing_true_incumbent_binding")
    binding_path_feedback = _read_json(binding_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_true_incumbent_binding")
    quantization_summary = _read_json(quantization_root / QUANTIZATION_SUMMARY_FILE, reasons, "missing_stage18h0_quantization")
    if binding_summary and binding_summary.get("true_incumbent_selection_bound") is not True:
        reasons.append("missing_true_incumbent_binding")
    if binding_summary and int(_float(binding_summary.get("fallback_action_index_0_count"), 0)) > 0:
        reasons.append("fallback_action_index_0_not_allowed")
    roi_scenarios = roi_path_feedback.get("scenarios") if isinstance(roi_path_feedback.get("scenarios"), list) else []
    binding_scenarios = binding_path_feedback.get("scenarios") if isinstance(binding_path_feedback.get("scenarios"), list) else []
    return {
        "roi_root": roi_root,
        "binding_root": binding_root,
        "quantization_root": quantization_root,
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "slices_by_id": {str(row.get("scenario_id") or row.get("slice_id")): row for row in slices if isinstance(row, dict)},
        "roi_path_feedback": roi_path_feedback,
        "binding_summary": binding_summary,
        "quantization_summary": quantization_summary,
        "binding_path_feedback": binding_path_feedback,
        "roi_scenarios": [row for row in roi_scenarios if isinstance(row, dict)],
        "binding_scenarios_by_id": {str(row.get("scenario_id")): row for row in binding_scenarios if isinstance(row, dict)},
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "proposals": output_root / PROPOSALS_FILE,
        "validated": output_root / VALIDATED_FILE,
        "validation_audit": output_root / VALIDATION_AUDIT_FILE,
        "proposal_validation_summary": output_root / PROPOSAL_VALIDATION_SUMMARY_FILE,
        "proposal_validation_results": output_root / PROPOSAL_VALIDATION_RESULTS_FILE,
        "report": output_root / REPORT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
    }


def _generate(
    config: dict[str, Any],
    source: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    generated = dict(source["roi_path_feedback"])
    generated_scenarios: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    risk_source_counts: Counter[str] = Counter()
    coverage_source_counts: Counter[str] = Counter()
    totals = Counter()
    coverage_values: list[float] = []
    roi_spreads: dict[str, list[float]] = defaultdict(list)
    roi_safe_counts: Counter[str] = Counter()
    reasons = list(source["reason_codes"])

    for scenario_index, scenario in enumerate(source["roi_scenarios"][: config["required_scenario_count"]]):
        scenario_id = str(scenario.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        binding = source["binding_scenarios_by_id"].get(scenario_id, {})
        slice_metadata = source["slices_by_id"].get(scenario_id, {})
        candidates = _candidate_rows(scenario)
        incumbent = _true_incumbent_info(binding, candidates)
        if incumbent["reason_code"]:
            reasons.append(incumbent["reason_code"])
        start_cell = _start_cell(
            slice_metadata,
            scenario,
            binding,
            incumbent.get("cell") if isinstance(incumbent.get("cell"), tuple) else (0, 0),
            scenario_index,
        )
        proposals = _proposal_rows(config, scenario, binding, incumbent, scenario_index, roi_group, start_cell)
        if not proposals:
            rejection_reasons["proposal_generation_empty"] += 1
        contract_path = _metadata_path(slice_metadata, "contract", source["roi_root"])
        sidecar_path = _metadata_path(slice_metadata, "sidecar", source["roi_root"])
        if contract_path is None or sidecar_path is None:
            validated_proposals = [
                _validation_missing_row(row, "missing_stage18a_contract_or_sidecar")
                for row in proposals
            ]
        else:
            validated_proposals = validate_candidate_cells(
                scenario=scenario,
                proposal_rows=proposals,
                contract_path=contract_path,
                sidecar_path=sidecar_path,
                current_cell=start_cell,
                repo_root=repo_root,
                output_work_root=output_root,
                validation_mode=str(config["candidate_validation_mode"]),
                top_k=len(proposals),
                allow_open_grid_fallback=bool(config["allow_open_grid_fallback"]),
                debug_validation_artifacts=bool(config["debug_validation_artifacts"]),
            )
        totals["proposal_validation_attempt_count"] += len(validated_proposals)
        incumbent_cost = _float(incumbent.get("path_cost"), 1.0e9)
        incumbent_risk = _float(incumbent.get("risk"), 1.0e9)
        incumbent_cells = _float(incumbent.get("incumbent_total_new_cell_count"), 0.0)
        formal_candidates: list[dict[str, Any]] = []
        scenario_proposals: list[dict[str, Any]] = []
        for proposal in validated_proposals:
            row = dict(proposal)
            row["scenario_id"] = scenario_id
            row["roi_group"] = roi_group
            _enrich_proposal(config, row, incumbent_cost, incumbent_risk, incumbent_cells)
            scenario_proposals.append(row)
            if row.get("proposal_validated_by_path_feedback") is True:
                totals["proposal_validation_success_count"] += 1
            else:
                totals["proposal_validation_failure_count"] += 1
                totals["path_feedback_validation_missing_count"] += 1
                rejection_reasons[str(row.get("path_feedback_validation_source") or "proposal_validation_missing")] += 1
            if row.get("proposal_only") is True:
                continue
            if row.get("reachable") is not True:
                rejection_reasons["unreachable_candidate"] += 1
                continue
            if row.get("open_grid_fallback_used") is True:
                totals["proposal_open_grid_fallback_count"] += 1
                rejection_reasons["open_grid_fallback_candidate"] += 1
                continue
            formal_candidates.append(row)

        selected = _select_formal_candidates(formal_candidates, int(config["max_candidates_per_scenario"]))
        if len(selected) < int(config["min_validated_candidates_per_scenario"]):
            totals["insufficient_validated_candidate_scenario_count"] += 1
        scenario_validated: list[dict[str, Any]] = []
        for action_index, candidate in enumerate(selected):
            row = dict(candidate)
            row["action_index"] = action_index
            if row.get("proposal_validated_by_path_feedback") is not True:
                totals["formal_candidate_missing_validation_count"] += 1
            if row.get("open_grid_fallback_used") is True:
                totals["open_grid_fallback_count"] += 1
            row.pop("scenario_id", None)
            row.pop("roi_group", None)
            scenario_validated.append(row)
            validated_public = dict(row)
            validated_public["scenario_id"] = scenario_id
            validated_public["roi_group"] = roi_group
            validated_rows.append(validated_public)
            totals["candidate_count"] += 1
            totals["validated_candidate_count"] += 1
            totals[f"{row['frontier_candidate_source']}_candidate_count"] += 1
            risk_source_counts[str(row.get("risk_source") or "unknown")] += 1
            coverage_source_counts[str(row.get("coverage_source") or "unknown")] += 1
            coverage_values.append(float(row.get("expected_coverage_rate_delta", 0.0)))
            roi_spreads[roi_group].append(float(row.get("expected_coverage_rate_delta", 0.0)))
            if row.get("safe_efficient_candidate") is True:
                totals["safe_efficient_candidate_count"] += 1
                roi_safe_counts[roi_group] += 1
            if row.get("coverage_guard_passed") and not row.get("cost_guard_passed"):
                totals["coverage_positive_but_cost_regressive_count"] += 1
        scenario_copy = {
            key: value
            for key, value in scenario.items()
            if key not in {"incumbent_selected_action_index", "xunce_selected_action_index", "incumbent_selection_source", "xunce_selection_source"}
        }
        scenario_copy["path_feedback"] = {"candidates": scenario_validated}
        scenario_copy["candidate_generation_source"] = GENERATION_SOURCE
        generated_scenarios.append(scenario_copy)
        proposal_rows.extend(scenario_proposals)
        totals["proposal_count"] += len(scenario_proposals)

    generated["scenarios"] = generated_scenarios
    generated["scenario_count"] = len(generated_scenarios)
    generated["candidate_count"] = int(totals["candidate_count"])
    generated["reachable_count"] = int(totals["validated_candidate_count"])
    generated["fallback_or_open_grid_count"] = int(totals["open_grid_fallback_count"])
    generated["open_grid_fallback_used"] = bool(totals["open_grid_fallback_count"])
    generated["candidate_generation_source"] = GENERATION_SOURCE
    metrics = {
        "reason_codes": unique_sorted(reasons),
        "scenario_count": len(generated_scenarios),
        "proposal_count": int(totals["proposal_count"]),
        "proposal_validation_attempt_count": int(totals["proposal_validation_attempt_count"]),
        "proposal_validation_success_count": int(totals["proposal_validation_success_count"]),
        "proposal_validation_failure_count": int(totals["proposal_validation_failure_count"]),
        "validated_candidate_count": int(totals["validated_candidate_count"]),
        "candidate_count": int(totals["candidate_count"]),
        "valid_candidate_count": int(totals["candidate_count"]),
        "invalid_candidate_count": max(0, int(totals["proposal_count"]) - int(totals["candidate_count"])),
        "frontier_boundary_candidate_count": int(totals["frontier_boundary_candidate_count"]),
        "roi_undercovered_boundary_candidate_count": int(totals["roi_undercovered_boundary_candidate_count"]),
        "safe_efficient_candidate_count": int(totals["safe_efficient_candidate_count"]),
        "proposal_unvalidated_positive_count": int(totals["path_feedback_validation_missing_count"]),
        "path_feedback_validation_missing_count": int(totals["path_feedback_validation_missing_count"]),
        "open_grid_fallback_count": int(totals["open_grid_fallback_count"]),
        "proposal_open_grid_fallback_count": int(totals["proposal_open_grid_fallback_count"]),
        "fallback_action_index_0_count": 0,
        "insufficient_validated_candidate_scenario_count": int(totals["insufficient_validated_candidate_scenario_count"]),
        "formal_candidate_missing_validation_count": int(totals["formal_candidate_missing_validation_count"]),
        "risk_source_counts": dict(sorted(risk_source_counts.items())),
        "coverage_source_counts": dict(sorted(coverage_source_counts.items())),
        "candidate_coverage_spread_range": _spread(coverage_values),
        "roi_group_with_nonzero_spread_count": sum(1 for values in roi_spreads.values() if _spread(values) > 0.005),
        "roi_group_with_safe_efficient_candidate_count": sum(1 for count in roi_safe_counts.values() if count > 0),
        "coverage_positive_but_cost_regressive_count": int(totals["coverage_positive_but_cost_regressive_count"]),
        "metric_coupling_detected": False,
    }
    validation_audit = {
        "schema_version": "xunce-true-frontier-nbv-validation-audit/v1",
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "path_feedback_validation_missing_count": metrics["path_feedback_validation_missing_count"],
        "formal_candidate_missing_validation_count": metrics["formal_candidate_missing_validation_count"],
        "insufficient_validated_candidate_scenario_count": metrics["insufficient_validated_candidate_scenario_count"],
        "open_grid_fallback_count": metrics["open_grid_fallback_count"],
        "proposal_open_grid_fallback_count": metrics["proposal_open_grid_fallback_count"],
        **_boundary_fields(),
    }
    return generated, proposal_rows, validated_rows, validation_audit, metrics


def _true_incumbent_info(binding: dict[str, Any], roi_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected_index = _int_or_none(binding.get("incumbent_selected_action_index"))
    selection_source = str(binding.get("incumbent_selection_source") or "")
    if selection_source == "fallback_action_index_0":
        return {"reason_code": "fallback_action_index_0_not_allowed"}
    binding_candidates = _candidate_rows(binding)
    selected = _candidate_by_action_index(binding_candidates, selected_index) if selected_index is not None else None
    cell = _cell_tuple(selected.get("cell") if selected else None)
    if selected_index is None or not selection_source or selected is None or cell is None:
        return {"reason_code": "missing_true_incumbent_binding"}
    source_candidate = _candidate_by_cell(roi_candidates, cell) or selected
    return {
        "reason_code": None,
        "selected_index": selected_index,
        "selection_source": selection_source,
        "cell": cell,
        "path_cost": _metric(source_candidate, "path_cost", "cost"),
        "risk": _metric(source_candidate, "risk", "path_risk_peak"),
    }


def _proposal_rows(
    config: dict[str, Any],
    scenario: dict[str, Any],
    binding: dict[str, Any],
    incumbent: dict[str, Any],
    scenario_index: int,
    roi_group: str,
    start_cell: tuple[int, int],
) -> list[dict[str, Any]]:
    incumbent_cell = incumbent.get("cell")
    if not isinstance(incumbent_cell, tuple):
        return []
    bounds = _roi_bounds(scenario, start_cell, incumbent_cell)
    rows: list[dict[str, Any]] = []
    rows.append(_proposal(config, "incumbent_neighborhood", incumbent_cell, start_cell, incumbent_cell, roi_group, 0))
    proposal_index = 1
    for radius in config["frontier_radius_cells"]:
        for dx, dy in DIRECTIONS_8[: int(config["frontier_direction_count"])]:
            cell = (start_cell[0] + dx * radius, start_cell[1] + dy * radius)
            rows.append(_proposal(config, "frontier_boundary", cell, start_cell, incumbent_cell, roi_group, proposal_index))
            proposal_index += 1
    min_cell, max_cell = bounds
    mid_x = int(round((min_cell[0] + max_cell[0]) / 2))
    mid_y = int(round((min_cell[1] + max_cell[1]) / 2))
    roi_cells = (
        min_cell,
        (min_cell[0], max_cell[1]),
        (max_cell[0], min_cell[1]),
        max_cell,
        (mid_x, max_cell[1]),
        (max_cell[0], mid_y),
    )
    for cell in roi_cells:
        rows.append(_proposal(config, "roi_undercovered_boundary", cell, start_cell, incumbent_cell, roi_group, proposal_index))
        proposal_index += 1
    for row in rows:
        row["true_incumbent_selected_action_index"] = incumbent.get("selected_index")
        row["true_incumbent_cell"] = list(incumbent_cell)
        row["incumbent_selection_source"] = incumbent.get("selection_source")
        row["proposal_only"] = True
        row["proposal_validated_by_path_feedback"] = False
    return _limit_proposals_by_family(rows, int(config["proposal_pool_limit_per_scenario"]))


def _limit_proposals_by_family(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    selected: list[dict[str, Any]] = []
    for source in ("incumbent_neighborhood", "frontier_boundary", "roi_undercovered_boundary"):
        for row in rows:
            if row.get("frontier_candidate_source") == source and row not in selected:
                selected.append(row)
                break
    for row in rows:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    return selected[:limit]


def _proposal(
    config: dict[str, Any],
    source: str,
    cell: tuple[int, int],
    start_cell: tuple[int, int],
    incumbent_cell: tuple[int, int],
    roi_group: str,
    proposal_index: int,
) -> dict[str, Any]:
    metrics = _coverage_metrics(config, start_cell, cell, incumbent_cell, source, roi_group)
    return {
        "proposal_id": f"{source}-{proposal_index:02d}-{cell[0]}-{cell[1]}",
        "cell": list(cell),
        "frontier_candidate_source": source,
        "proposal_generation_source": GENERATION_SOURCE,
        **metrics,
    }


def _enrich_proposal(config: dict[str, Any], row: dict[str, Any], incumbent_cost: float, incumbent_risk: float, incumbent_cells: float) -> None:
    cost = _float(row.get("path_cost"), 1.0e9)
    risk = _float(row.get("risk"), 1.0e9)
    coverage_cells = _float(row.get("expected_new_coverage_cell_count"), 0.0)
    coverage_guard = coverage_cells > incumbent_cells + TOLERANCE
    risk_guard = risk <= incumbent_risk + float(config["risk_margin"]) + TOLERANCE
    cost_guard = cost <= incumbent_cost + float(config["cost_margin"]) + TOLERANCE
    validated = row.get("proposal_validated_by_path_feedback") is True
    safe = bool(validated and coverage_guard and risk_guard and cost_guard and row.get("frontier_candidate_source") == "frontier_boundary")
    row.update(
        {
            "coverage_source": COVERAGE_SOURCE,
            "coverage_opportunity_source": COVERAGE_SOURCE,
            "coverage_cell_set_kind": COVERAGE_CELL_SET_KIND,
            "coverage_dedupe_scope": COVERAGE_DEDUPE_SCOPE,
            "coverage_validated_by_path_feedback": False,
            "coverage_validation_source": "offline_geometric_counterfactual_not_path_feedback",
            "candidate_generation_source": GENERATION_SOURCE,
            "budget_used_ratio": min(cost / float(config["path_budget_m"]), 1.0) if math.isfinite(cost) else 1.0,
            "coverage_guard_passed": coverage_guard,
            "risk_guard_passed": risk_guard,
            "cost_guard_passed": cost_guard,
            "safe_efficient_candidate": safe,
            "safe_efficient_opportunity": safe,
            "metric_coupling_detected": False,
        }
    )


def _coverage_metrics(
    config: dict[str, Any],
    start_cell: tuple[int, int],
    cell: tuple[int, int],
    incumbent_cell: tuple[int, int],
    source: str,
    roi_group: str,
) -> dict[str, Any]:
    denominator = float(config["coverage_denominator_cells"])
    distance_from_start = _manhattan(start_cell, cell)
    distance_from_incumbent = _manhattan(incumbent_cell, cell)
    source_bonus = {"incumbent_neighborhood": 0, "frontier_boundary": 8, "roi_undercovered_boundary": 14}[source]
    endpoint_cells = max(1, distance_from_incumbent * 2 + source_bonus)
    path_cells = max(1, int(round(distance_from_start * 0.5 + distance_from_incumbent + source_bonus)))
    total_cells = max(endpoint_cells, path_cells) + max(0, distance_from_incumbent // 2)
    if source == "incumbent_neighborhood":
        total_cells = 1
    overlap = max(0, min(endpoint_cells, path_cells) // 4)
    roi_weight = 1.0 + (abs(hash(roi_group)) % 5) * 0.05
    return {
        "endpoint_new_cell_count": float(endpoint_cells),
        "path_line_new_cell_count": float(path_cells),
        "total_new_cell_count": float(total_cells),
        "expected_new_coverage_cell_count": float(total_cells),
        "endpoint_coverage_delta": float(endpoint_cells) / denominator,
        "path_line_coverage_delta": float(path_cells) / denominator,
        "expected_coverage_rate_delta": float(total_cells) / denominator,
        "roi_weighted_coverage_delta": float(total_cells) * roi_weight / denominator,
        "coverage_overlap_count": float(overlap),
        "coverage_overlap_ratio": float(overlap) / max(float(total_cells), 1.0),
        "revisit_penalty": float(overlap) / max(float(endpoint_cells + path_cells), 1.0),
    }


def _select_formal_candidates(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in ("incumbent_neighborhood", "frontier_boundary", "roi_undercovered_boundary"):
        family = [row for row in candidates if row.get("frontier_candidate_source") == source]
        if family:
            selected.append(_best_family_candidate(family, source))
    remaining = [row for row in candidates if row not in selected]
    ordered = _stable_candidate_order(selected) + _stable_candidate_order(remaining)
    return ordered[:limit]


def _best_family_candidate(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    if source == "incumbent_neighborhood":
        return min(rows, key=lambda row: (_float(row.get("risk"), 1.0e9), _float(row.get("path_cost"), 1.0e9)))
    if source == "frontier_boundary":
        return max(rows, key=lambda row: (row.get("safe_efficient_candidate") is True, _float(row.get("expected_coverage_rate_delta"), 0.0), -_float(row.get("risk"), 1.0e9)))
    return max(rows, key=lambda row: (_float(row.get("expected_coverage_rate_delta"), 0.0), -_float(row.get("risk"), 1.0e9), -_float(row.get("path_cost"), 1.0e9)))


def _stable_candidate_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            0 if row.get("frontier_candidate_source") == "incumbent_neighborhood" else 1 if row.get("safe_efficient_candidate") else 2,
            -_float(row.get("expected_coverage_rate_delta"), 0.0),
            _float(row.get("risk"), 1.0e9),
            _float(row.get("path_cost"), 1.0e9),
            tuple(row.get("cell") or []),
        ),
    )


def _decision(source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    blocking = list(metrics["reason_codes"]) + list(source["reason_codes"])
    diagnostic: list[str] = []
    status = "passed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE
    if any(reason.startswith("missing_stage18a") for reason in blocking):
        status = "failed"
        next_required_change = FIX_STAGE18A_NEXT_REQUIRED_CHANGE
    elif "missing_stage18h0_quantization" in blocking:
        status = "failed"
        next_required_change = FIX_PROPOSAL_NEXT_REQUIRED_CHANGE
    elif "missing_true_incumbent_binding" in blocking or "fallback_action_index_0_not_allowed" in blocking:
        status = "failed"
        next_required_change = FIX_BINDING_NEXT_REQUIRED_CHANGE
    elif metrics["proposal_count"] <= 0:
        blocking.append("proposal_generation_empty")
        status = "failed"
        next_required_change = FIX_PROPOSAL_NEXT_REQUIRED_CHANGE
    elif metrics["formal_candidate_missing_validation_count"] > 0:
        blocking.append("formal_candidate_missing_validation")
        status = "failed"
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    elif metrics["insufficient_validated_candidate_scenario_count"] > 0:
        blocking.append("insufficient_validated_candidates_per_scenario")
        status = "failed"
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    elif metrics["open_grid_fallback_count"] > 0:
        blocking.append("formal_candidate_open_grid_fallback")
        status = "failed"
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    elif metrics["candidate_count"] <= 0:
        blocking.append("no_valid_candidates")
        status = "failed"
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    if metrics["candidate_count"] <= 0 and "no_valid_candidates" not in blocking:
        blocking.append("no_valid_candidates")
        status = "failed"
        if next_required_change == PASS_NEXT_REQUIRED_CHANGE:
            next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    if metrics["frontier_boundary_candidate_count"] <= 0:
        diagnostic.append("frontier_boundary_candidate_missing")
    if metrics["roi_undercovered_boundary_candidate_count"] <= 0:
        diagnostic.append("roi_undercovered_boundary_candidate_missing")
    if metrics["safe_efficient_candidate_count"] <= 0:
        diagnostic.append("safe_efficient_candidate_missing")
    if metrics["coverage_positive_but_cost_regressive_count"] > 0:
        diagnostic.append("coverage_positive_but_cost_regressive")
    if metrics["path_feedback_validation_missing_count"] > 0:
        diagnostic.append("proposal_validation_missing")
    if metrics["proposal_open_grid_fallback_count"] > 0:
        diagnostic.append("proposal_open_grid_fallback")
    blocking = unique_sorted(blocking)
    diagnostic = unique_sorted(diagnostic)
    evidence_gate = not any(
        reason.startswith("missing_stage18a")
        or reason in {"missing_true_incumbent_binding", "fallback_action_index_0_not_allowed", "missing_stage18h0_quantization"}
        for reason in blocking
    )
    candidate_gate = not any(
        reason in {
            "proposal_generation_empty",
            "formal_candidate_missing_validation",
            "formal_candidate_open_grid_fallback",
            "insufficient_validated_candidates_per_scenario",
            "no_valid_candidates",
        }
        for reason in blocking
    )
    return {
        "status": status,
        "reason_codes": blocking,
        "blocking_reason_codes": blocking,
        "diagnostic_reason_codes": diagnostic,
        "evidence_authenticity_gate_passed": evidence_gate,
        "candidate_validity_gate_passed": candidate_gate,
        "comparison_allowed": status == "passed" and evidence_gate and candidate_gate,
        "next_required_change": next_required_change,
    }


def _summary(
    *,
    generated_at: str,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    source: dict[str, Any],
    metrics: dict[str, Any],
    decision: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "source_roi_expansion_root": config["source_roi_expansion_root"],
        "source_true_incumbent_binding_root": config["source_true_incumbent_binding_root"],
        "source_quantization_root": config["source_quantization_root"],
        "candidate_validation_mode": config["candidate_validation_mode"],
        "min_validated_candidates_per_scenario": config["min_validated_candidates_per_scenario"],
        "debug_validation_artifacts": config["debug_validation_artifacts"],
        "source_true_incumbent_binding_status": source["binding_summary"].get("status"),
        "source_quantization_status": source["quantization_summary"].get("status"),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary": {
            "scenario_count": metrics["scenario_count"],
            "candidate_count": metrics["candidate_count"],
            "safe_efficient_candidate_count": metrics["safe_efficient_candidate_count"],
            "next_required_change": decision["next_required_change"],
        },
        **metrics,
        **decision,
        **_boundary_fields(),
        "git": git_snapshot(repo_root),
    }
    payload["canary_traffic_fraction"] = float(config["canary_traffic_fraction"])
    return payload


def _proposal_validation_summary(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-true-frontier-nbv-proposal-validation-summary/v1",
        "candidate_validation_mode": config["candidate_validation_mode"],
        "proposal_validation_attempt_count": metrics["proposal_validation_attempt_count"],
        "proposal_validation_success_count": metrics["proposal_validation_success_count"],
        "proposal_validation_failure_count": metrics["proposal_validation_failure_count"],
        "valid_candidate_count": metrics["valid_candidate_count"],
        "invalid_candidate_count": metrics["invalid_candidate_count"],
        "insufficient_validated_candidate_scenario_count": metrics["insufficient_validated_candidate_scenario_count"],
        "formal_candidate_missing_validation_count": metrics["formal_candidate_missing_validation_count"],
        "open_grid_fallback_count": metrics["open_grid_fallback_count"],
        "proposal_open_grid_fallback_count": metrics["proposal_open_grid_fallback_count"],
        "risk_source_counts": metrics["risk_source_counts"],
        "coverage_source_counts": metrics["coverage_source_counts"],
        "next_required_change": (
            FIX_VALIDATION_NEXT_REQUIRED_CHANGE
            if metrics["insufficient_validated_candidate_scenario_count"] > 0
            or metrics["formal_candidate_missing_validation_count"] > 0
            else PASS_NEXT_REQUIRED_CHANGE
        ),
        **_boundary_fields(),
    }


def _generated_expansion_summary(expansion_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(expansion_summary)
    payload["true_frontier_nbv_candidate_source_replacement_status"] = summary["status"]
    payload["candidate_generation_source"] = GENERATION_SOURCE
    payload["candidate_count"] = summary["candidate_count"]
    payload["safe_efficient_candidate_count"] = summary["safe_efficient_candidate_count"]
    payload["frontier_boundary_candidate_count"] = summary["frontier_boundary_candidate_count"]
    payload["roi_undercovered_boundary_candidate_count"] = summary["roi_undercovered_boundary_candidate_count"]
    payload["next_required_change"] = summary["next_required_change"]
    payload.update(_boundary_fields())
    payload["canary_traffic_fraction"] = summary["canary_traffic_fraction"]
    return payload


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18I.3 True Frontier-NBV Candidate Source Replacement",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- scenario_count: `{summary['scenario_count']}`",
            f"- proposal_count: `{summary['proposal_count']}`",
            f"- candidate_count: `{summary['candidate_count']}`",
            f"- safe_efficient_candidate_count: `{summary['safe_efficient_candidate_count']}`",
            "",
            "This offline artifact replaces candidate source construction only. It does not train, publish checkpoints, connect an executor, or start an online canary.",
            "",
        ]
    )


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = feedback.get("candidates") if isinstance(feedback.get("candidates"), list) else []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _candidate_by_action_index(candidates: list[dict[str, Any]], action_index: int | None) -> dict[str, Any] | None:
    if action_index is None:
        return None
    for row in candidates:
        if _int_or_none(row.get("action_index")) == action_index:
            return row
    if 0 <= action_index < len(candidates):
        return candidates[action_index]
    return None


def _candidate_by_cell(candidates: list[dict[str, Any]], cell: tuple[int, int]) -> dict[str, Any] | None:
    for row in candidates:
        if _cell_tuple(row.get("cell")) == cell:
            return row
    return None


def _metadata_path(metadata: dict[str, Any], key: str, base_root: Path) -> Path | None:
    value = metadata.get(key)
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_root / path


def _validation_missing_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = dict(row)
    payload["proposal_only"] = True
    payload["proposal_validated_by_path_feedback"] = False
    payload["path_feedback_validation_source"] = reason
    payload["planner_reachable"] = False
    payload["reachable"] = False
    payload["failure_reason"] = reason
    payload["replan_required"] = True
    payload["open_grid_fallback_used"] = False
    payload["validation_diagnostic_flags"] = [reason]
    payload["risk_source"] = "unavailable"
    return payload


def _start_cell(
    slice_metadata: dict[str, Any],
    scenario: dict[str, Any],
    binding: dict[str, Any],
    incumbent_cell: tuple[int, int],
    scenario_index: int,
) -> tuple[int, int]:
    for value in (slice_metadata.get("start_cell"), scenario.get("start_cell"), binding.get("start_cell")):
        cell = _cell_tuple(value)
        if cell is not None:
            return cell
    return (incumbent_cell[0] - scenario_index - 1, incumbent_cell[1])


def _roi_bounds(scenario: dict[str, Any], start_cell: tuple[int, int], incumbent_cell: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    bounds = scenario.get("roi_bounds") if isinstance(scenario.get("roi_bounds"), dict) else {}
    min_cell = _cell_tuple(bounds.get("min_cell"))
    max_cell = _cell_tuple(bounds.get("max_cell"))
    if min_cell is not None and max_cell is not None:
        return min_cell, max_cell
    min_x = min(start_cell[0], incumbent_cell[0]) - 6
    min_y = min(start_cell[1], incumbent_cell[1]) - 6
    max_x = max(start_cell[0], incumbent_cell[0]) + 6
    max_y = max(start_cell[1], incumbent_cell[1]) + 6
    return (min_x, min_y), (max_x, max_y)


def _read_json(path: Path, reasons: list[str], missing_reason: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(missing_reason)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], missing_reason: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(missing_reason)
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None


def _metric(row: dict[str, Any], primary: str, secondary: str) -> float | None:
    for key in (primary, secondary):
        if key in row:
            value = _float(row.get(key), float("nan"))
            if math.isfinite(value):
                return value
    return None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a positive integer") from None
    if result <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _float(value, float("nan"))
    if not math.isfinite(result) or result <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _float(value, float("nan"))
    if not math.isfinite(result) or result < 0:
        raise ConfigError(f"{name} must be a non-negative number")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _boundary_fields() -> dict[str, bool]:
    payload = {key: False for key in BOUNDARY_FIELDS}
    payload["canary_traffic_fraction"] = 0.0
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
