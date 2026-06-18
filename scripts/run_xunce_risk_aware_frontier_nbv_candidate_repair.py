from __future__ import annotations

import argparse
import json
import math
import statistics
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
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-risk-aware-frontier-nbv-candidate-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-risk-aware-frontier-nbv-candidate-repair-summary/v1"
DEFAULT_CONFIG = "configs/xunce_risk_aware_frontier_nbv_candidate_repair_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_risk_aware_frontier_nbv_candidate_repair_v1"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"
TRUE_BINDING_SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
QUANTIZATION_SUMMARY_FILE = "xunce-risk-coverage-cost-quantization-summary.json"
MATERIALIZATION_SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"

SUMMARY_FILE = "xunce-risk-aware-frontier-nbv-candidate-repair-summary.json"
PROPOSALS_FILE = "xunce-risk-aware-frontier-nbv-proposals.jsonl"
VALIDATED_FILE = "xunce-risk-aware-frontier-nbv-validated-candidates.jsonl"
REJECTION_AUDIT_FILE = "xunce-risk-aware-frontier-nbv-rejection-audit.json"
REPORT_FILE = "xunce-risk-aware-frontier-nbv-report.md"

PASS_NEXT_REQUIRED_CHANGE = "rerun_true_model_inference_and_binding"
COMPATIBLE_EXPANSION_NEXT_REQUIRED_CHANGE = "xunce_high_fidelity_real_map_policy_comparison"
FIX_STAGE18A_NEXT_REQUIRED_CHANGE = "run_xunce_high_fidelity_real_map_roi_expansion"
FIX_BINDING_NEXT_REQUIRED_CHANGE = "run_true_incumbent_selection_binding"
FIX_QUANTIZATION_NEXT_REQUIRED_CHANGE = "run_xunce_risk_coverage_cost_quantization_audit"
FIX_SAMPLING_NEXT_REQUIRED_CHANGE = "repair_frontier_nbv_sampling"
FIX_VALIDATION_NEXT_REQUIRED_CHANGE = "repair_path_feedback_candidate_validation"
FIX_RISK_AWARE_GENERATION_NEXT_REQUIRED_CHANGE = "repair_risk_aware_candidate_generation"
EXPAND_ROI_OR_MAP_NEXT_REQUIRED_CHANGE = "expand_roi_or_refine_frontier_nbv_candidate_generation"

GENERATION_SOURCE = "risk_aware_frontier_nbv_candidate_repair/v1"
COVERAGE_SOURCE = "geometric_counterfactual_from_risk_aware_frontier_nbv_repair/v1"
COVERAGE_CELL_SET_KIND = "path_line_plus_endpoint_union"
COVERAGE_DEDUPE_SCOPE = "scenario_step_new_cells"
SAFE_KNOWN_SOURCE = "path_feedback_and_true_incumbent_binding/v1"
TOLERANCE = 1.0e-12

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
    parser = argparse.ArgumentParser(description="Repair Xunce candidates with true-incumbent anchored frontier/NBV generation.")
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
        summary = run_xunce_risk_aware_frontier_nbv_candidate_repair(
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
                "safe_efficient_candidate_count": summary["safe_efficient_candidate_count"],
                "frontier_boundary_candidate_count": summary["frontier_boundary_candidate_count"],
                "roi_undercovered_boundary_candidate_count": summary["roi_undercovered_boundary_candidate_count"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_risk_aware_frontier_nbv_candidate_repair(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    source = _load_source(config)
    generated, proposal_rows, validated_rows, rejection_audit, metrics = _generate(config, source)
    decision = _decision(config, source, metrics)
    generated_at = utc_now()
    paths = _paths(output_root)
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
    write_json(paths["rejection_audit"], rejection_audit)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    write_json(paths["expansion_summary"], _generated_expansion_summary(source["expansion_summary"], summary))
    write_jsonl(paths["slices"], source["slices"])
    write_json(paths["path_feedback"], generated)
    write_json(paths["materialization_summary"], _materialization_summary(summary))
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
    normalized["coverage_denominator_cells"] = _positive_float(payload.get("coverage_denominator_cells", 1000), "coverage_denominator_cells")
    normalized["risk_margin"] = _nonnegative_float(payload.get("risk_margin", 0.01), "risk_margin")
    normalized["cost_margin"] = _nonnegative_float(payload.get("cost_margin", 0.0), "cost_margin")
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["allow_open_grid_fallback"] = _bool(payload.get("allow_open_grid_fallback", False), "allow_open_grid_fallback")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    roi_root = Path(config["source_roi_expansion_root"])
    binding_root = Path(config["source_true_incumbent_binding_root"])
    quant_root = Path(config["source_quantization_root"])
    reasons: list[str] = []
    expansion_summary = _read_json(roi_root / EXPANSION_SUMMARY_FILE, reasons, "missing_stage18a_roi_expansion")
    slices = _read_jsonl(roi_root / EXPANSION_SLICES_FILE, reasons, "missing_stage18a_roi_expansion")
    roi_path_feedback = _read_json(roi_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_stage18a_roi_expansion")
    binding_summary = _read_json(binding_root / TRUE_BINDING_SUMMARY_FILE, reasons, "missing_true_incumbent_binding")
    binding_path_feedback = _read_json(binding_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_true_incumbent_binding")
    quantization_summary = _read_json(quant_root / QUANTIZATION_SUMMARY_FILE, reasons, "missing_stage18h0_quantization")

    if binding_summary and binding_summary.get("true_incumbent_selection_bound") is not True:
        reasons.append("missing_true_incumbent_binding")
    if binding_summary and int(_float(binding_summary.get("fallback_action_index_0_count"), 0)) > 0:
        reasons.append("fallback_action_index_0_not_allowed")

    roi_scenarios = roi_path_feedback.get("scenarios") if isinstance(roi_path_feedback.get("scenarios"), list) else []
    binding_scenarios = binding_path_feedback.get("scenarios") if isinstance(binding_path_feedback.get("scenarios"), list) else []
    return {
        "roi_root": roi_root,
        "binding_root": binding_root,
        "quantization_root": quant_root,
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "roi_path_feedback": roi_path_feedback,
        "binding_summary": binding_summary,
        "binding_path_feedback": binding_path_feedback,
        "quantization_summary": quantization_summary,
        "roi_scenarios": [row for row in roi_scenarios if isinstance(row, dict)],
        "binding_scenarios_by_id": {str(row.get("scenario_id")): row for row in binding_scenarios if isinstance(row, dict)},
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "proposals": output_root / PROPOSALS_FILE,
        "validated": output_root / VALIDATED_FILE,
        "rejection_audit": output_root / REJECTION_AUDIT_FILE,
        "report": output_root / REPORT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "materialization_summary": output_root / MATERIALIZATION_SUMMARY_FILE,
    }


def _generate(
    config: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    generated = dict(source["roi_path_feedback"])
    generated_scenarios: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    coverage_values: list[float] = []
    roi_spreads: dict[str, list[float]] = defaultdict(list)
    roi_safe_counts: Counter[str] = Counter()
    totals = Counter()
    reasons = list(source["reason_codes"])

    for scenario_index, scenario in enumerate(source["roi_scenarios"][: config["required_scenario_count"]]):
        scenario_id = str(scenario.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        binding = source["binding_scenarios_by_id"].get(scenario_id, {})
        candidates = _candidate_rows(scenario)
        binding_candidates = _candidate_rows(binding)
        incumbent_info = _true_incumbent_info(binding, candidates, binding_candidates)
        if incumbent_info["reason_code"]:
            reasons.append(incumbent_info["reason_code"])
            if incumbent_info["reason_code"] == "candidate_cell_mismatch":
                rejection_reasons["candidate_cell_mismatch"] += 1
        proposals = _scenario_proposals(config, scenario, binding, candidates, incumbent_info, scenario_index, roi_group)
        formal_candidates: list[dict[str, Any]] = []
        scenario_proposal_rows: list[dict[str, Any]] = []

        for proposal in proposals:
            public = dict(proposal)
            public["scenario_id"] = scenario_id
            public["roi_group"] = roi_group
            scenario_proposal_rows.append(public)
            if proposal.get("reachable") is not True:
                rejection_reasons["unreachable_candidate"] += 1
                continue
            if bool(proposal.get("open_grid_fallback_used", False)):
                totals["open_grid_fallback_count"] += 1
                rejection_reasons["open_grid_fallback_candidate"] += 1
                continue
            if proposal.get("proposal_only") is True:
                if proposal.get("expected_new_coverage_cell_count", 0) > incumbent_info["incumbent_total_new_cell_count"]:
                    totals["proposal_unvalidated_positive_count"] += 1
                    rejection_reasons["path_feedback_validation_missing"] += 1
                continue
            formal_candidates.append(proposal)

        formal_candidates = _select_formal_candidates(formal_candidates, int(config["max_candidates_per_scenario"]))
        scenario_validated_rows: list[dict[str, Any]] = []
        for action_index, candidate in enumerate(formal_candidates):
            row = dict(candidate)
            row["action_index"] = action_index
            row["proposal_only"] = False
            row["proposal_validated_by_path_feedback"] = True
            row["scenario_id"] = scenario_id
            row["roi_group"] = roi_group
            scenario_validated_rows.append(row)
            validated_rows.append(row)
            totals["candidate_count"] += 1
            totals["validated_candidate_count"] += 1
            totals[f"{row['frontier_candidate_source']}_candidate_count"] += 1
            coverage_values.append(float(row["expected_coverage_rate_delta"]))
            roi_spreads[roi_group].append(float(row["expected_coverage_rate_delta"]))
            if row["coverage_guard_passed"]:
                totals["coverage_positive_candidate_count"] += 1
                if not row["risk_guard_passed"]:
                    totals["coverage_positive_but_risk_regressive_count"] += 1
                if not row["cost_guard_passed"]:
                    totals["coverage_positive_but_cost_regressive_count"] += 1
            if row["safe_efficient_candidate"]:
                totals["safe_efficient_candidate_count"] += 1
                roi_safe_counts[roi_group] += 1

        scenario_copy = {
            key: value
            for key, value in scenario.items()
            if key
            not in {
                "incumbent_selected_action_index",
                "xunce_selected_action_index",
                "incumbent_selection_source",
                "xunce_selection_source",
            }
        }
        scenario_copy["path_feedback"] = {"candidates": [{key: value for key, value in row.items() if key not in {"scenario_id", "roi_group"}} for row in scenario_validated_rows]}
        scenario_copy["frontier_nbv_candidate_repair_source"] = GENERATION_SOURCE
        scenario_copy["safe_known_source"] = SAFE_KNOWN_SOURCE
        generated_scenarios.append(scenario_copy)
        proposal_rows.extend(scenario_proposal_rows)
        totals["proposal_count"] += len(scenario_proposal_rows)

    generated["scenarios"] = generated_scenarios
    generated["scenario_count"] = len(generated_scenarios)
    generated["candidate_count"] = int(totals["candidate_count"])
    generated["reachable_count"] = int(totals["validated_candidate_count"])
    generated["fallback_or_open_grid_count"] = int(totals["open_grid_fallback_count"])
    generated["open_grid_fallback_used"] = bool(totals["open_grid_fallback_count"])
    generated["frontier_nbv_candidate_repair_source"] = GENERATION_SOURCE

    metrics = {
        "reason_codes": unique_sorted(reasons),
        "scenario_count": len(generated_scenarios),
        "proposal_count": int(totals["proposal_count"]),
        "validated_candidate_count": int(totals["validated_candidate_count"]),
        "candidate_count": int(totals["candidate_count"]),
        "frontier_boundary_candidate_count": int(totals["frontier_boundary_candidate_count"]),
        "roi_undercovered_boundary_candidate_count": int(totals["roi_undercovered_boundary_candidate_count"]),
        "incumbent_neighborhood_candidate_count": int(totals["incumbent_neighborhood_candidate_count"]),
        "candidate_coverage_spread_range": _spread(coverage_values),
        "roi_group_with_nonzero_spread_count": sum(1 for values in roi_spreads.values() if _spread(values) > 0.005),
        "safe_efficient_candidate_count": int(totals["safe_efficient_candidate_count"]),
        "roi_group_with_safe_efficient_candidate_count": sum(1 for count in roi_safe_counts.values() if count > 0),
        "coverage_positive_candidate_count": int(totals["coverage_positive_candidate_count"]),
        "coverage_positive_but_risk_regressive_count": int(totals["coverage_positive_but_risk_regressive_count"]),
        "coverage_positive_but_cost_regressive_count": int(totals["coverage_positive_but_cost_regressive_count"]),
        "path_feedback_validation_missing_count": int(totals["proposal_unvalidated_positive_count"]),
        "proposal_unvalidated_positive_count": int(totals["proposal_unvalidated_positive_count"]),
        "open_grid_fallback_count": int(totals["open_grid_fallback_count"]),
        "fallback_action_index_0_count": 0,
        "metric_coupling_detected": False,
        "candidate_pareto_frontier_count": _pareto_frontier_count(validated_rows),
        "valid_candidate_count": int(totals["validated_candidate_count"]),
        "invalid_candidate_count": max(int(totals["proposal_count"]) - int(totals["validated_candidate_count"]), 0),
        "valid_scenario_count": sum(1 for scenario in generated_scenarios if _candidate_rows(scenario)),
        "invalid_scenario_count": sum(1 for scenario in generated_scenarios if not _candidate_rows(scenario)),
        "invalid_candidate_reason_counts": dict(sorted(rejection_reasons.items())),
    }
    rejection_audit = {
        "schema_version": "xunce-risk-aware-frontier-nbv-rejection-audit/v1",
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "proposal_unvalidated_positive_count": metrics["proposal_unvalidated_positive_count"],
        "path_feedback_validation_missing_count": metrics["path_feedback_validation_missing_count"],
        **_boundary_fields(),
    }
    return generated, proposal_rows, validated_rows, rejection_audit, metrics


def _true_incumbent_info(
    binding: dict[str, Any],
    roi_candidates: list[dict[str, Any]],
    binding_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_index = _int_or_none(binding.get("incumbent_selected_action_index"))
    source = str(binding.get("incumbent_selection_source") or "")
    if selected_index is None or source != "true_checkpoint_inference":
        return {
            "reason_code": "missing_true_incumbent_binding",
            "selected_index": selected_index,
            "selection_source": source or "missing",
            "cell": None,
            "candidate": None,
            "incumbent_total_new_cell_count": 0.0,
        }
    selected = _candidate_by_action_index(binding_candidates, selected_index)
    selected_cell = _cell_tuple(selected.get("cell") if selected else None)
    if selected is None or selected_cell is None:
        return {
            "reason_code": "missing_true_incumbent_binding",
            "selected_index": selected_index,
            "selection_source": source,
            "cell": None,
            "candidate": None,
            "incumbent_total_new_cell_count": 0.0,
        }
    matched = _candidate_by_cell(roi_candidates, selected_cell)
    if matched is None:
        return {
            "reason_code": "candidate_cell_mismatch",
            "selected_index": selected_index,
            "selection_source": source,
            "cell": selected_cell,
            "candidate": selected,
            "incumbent_total_new_cell_count": 0.0,
        }
    return {
        "reason_code": None,
        "selected_index": selected_index,
        "selection_source": source,
        "cell": selected_cell,
        "candidate": matched,
        "incumbent_total_new_cell_count": 0.0,
    }


def _scenario_proposals(
    config: dict[str, Any],
    scenario: dict[str, Any],
    binding: dict[str, Any],
    candidates: list[dict[str, Any]],
    incumbent_info: dict[str, Any],
    scenario_index: int,
    roi_group: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    incumbent_cell = incumbent_info["cell"]
    incumbent_candidate = incumbent_info["candidate"] if isinstance(incumbent_info["candidate"], dict) else {}
    if incumbent_cell is None or not isinstance(incumbent_candidate, dict) or not incumbent_candidate:
        return rows
    start_cell = _start_cell(scenario, binding, candidates, incumbent_cell, scenario_index)
    incumbent_risk = _risk_value(incumbent_candidate)
    incumbent_cost = _path_cost(incumbent_candidate)
    incumbent_metrics = _geometric_coverage_metrics(config, start_cell, incumbent_cell, incumbent_cell, "incumbent_neighborhood", roi_group)
    incumbent_info["incumbent_total_new_cell_count"] = incumbent_metrics["total_new_cell_count"]

    for candidate in candidates:
        cell = _cell_tuple(candidate.get("cell"))
        if cell is None:
            continue
        distance_from_incumbent = _manhattan(cell, incumbent_cell)
        risk = _risk_value(candidate)
        cost = _path_cost(candidate)
        reachable = candidate.get("reachable") is True
        risk_missing = not _has_finite_number(candidate, "risk") and not _has_finite_number(candidate, "path_risk_peak")
        cost_missing = not _has_finite_number(candidate, "path_cost") and not _has_finite_number(candidate, "cost")
        open_grid = bool(candidate.get("open_grid_fallback_used", scenario.get("open_grid_fallback_used", False)))
        validated = (
            candidate.get("proposal_validated_by_path_feedback", True) is not False
            and reachable
            and (not open_grid or bool(config["allow_open_grid_fallback"]))
            and not risk_missing
            and not cost_missing
            and math.isfinite(risk)
            and math.isfinite(cost)
        )
        source = _candidate_source(
            distance_from_incumbent=distance_from_incumbent,
            cell=cell,
            incumbent_cell=incumbent_cell,
            risk=risk,
            incumbent_risk=incumbent_risk,
            cost=cost,
            incumbent_cost=incumbent_cost,
            risk_margin=float(config["risk_margin"]),
            cost_margin=float(config["cost_margin"]),
        )
        coverage = _geometric_coverage_metrics(config, start_cell, cell, incumbent_cell, source, roi_group)
        coverage_guard = coverage["total_new_cell_count"] > incumbent_metrics["total_new_cell_count"] + TOLERANCE
        risk_guard = risk <= incumbent_risk + float(config["risk_margin"]) + TOLERANCE
        cost_guard = cost <= incumbent_cost + float(config["cost_margin"]) + TOLERANCE
        safe = bool(validated and coverage_guard and risk_guard and cost_guard and source == "frontier_boundary")
        row = dict(candidate)
        row.update(
            {
                "source_action_index": candidate.get("action_index"),
                "true_incumbent_selected_action_index": incumbent_info["selected_index"],
                "true_incumbent_cell": list(incumbent_cell),
                "incumbent_selection_source": incumbent_info["selection_source"],
                "cell": list(cell),
                "reachable": reachable,
                "path_cost": cost,
                "risk": risk,
                "budget_used_ratio": min(cost / float(config["path_budget_m"]), 1.0),
                "endpoint_new_cell_count": coverage["endpoint_new_cell_count"],
                "path_line_new_cell_count": coverage["path_line_new_cell_count"],
                "total_new_cell_count": coverage["total_new_cell_count"],
                "endpoint_coverage_delta": coverage["endpoint_coverage_delta"],
                "path_line_coverage_delta": coverage["path_line_coverage_delta"],
                "expected_coverage_rate_delta": coverage["expected_coverage_rate_delta"],
                "expected_new_coverage_cell_count": coverage["total_new_cell_count"],
                "roi_weighted_coverage_delta": coverage["roi_weighted_coverage_delta"],
                "revisit_penalty": coverage["revisit_penalty"],
                "coverage_overlap_count": coverage["coverage_overlap_count"],
                "coverage_overlap_ratio": coverage["coverage_overlap_ratio"],
                "coverage_source": COVERAGE_SOURCE,
                "coverage_opportunity_source": COVERAGE_SOURCE,
                "coverage_cell_set_kind": COVERAGE_CELL_SET_KIND,
                "coverage_dedupe_scope": COVERAGE_DEDUPE_SCOPE,
                "candidate_generation_source": GENERATION_SOURCE,
                "frontier_candidate_source": source,
                "safe_known_source": SAFE_KNOWN_SOURCE,
                "proposal_only": not validated,
                "proposal_validated_by_path_feedback": bool(validated),
                "path_feedback_metric_missing": bool(risk_missing or cost_missing),
                "open_grid_fallback_used": open_grid,
                "coverage_guard_passed": coverage_guard,
                "risk_guard_passed": risk_guard,
                "cost_guard_passed": cost_guard,
                "safe_efficient_candidate": safe,
                "safe_efficient_opportunity": safe,
                "metric_coupling_detected": False,
            }
        )
        row["coverage_gain_per_path_cost"] = _safe_ratio(row["expected_coverage_rate_delta"], cost) or 0.0
        row["coverage_gain_per_risk"] = _safe_ratio(row["expected_coverage_rate_delta"], 1.0 + risk) or 0.0
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["expected_coverage_rate_delta"]), _risk_value(row), _path_cost(row), tuple(row.get("cell") or [])))
    for rank, row in enumerate(rows, start=1):
        row["coverage_opportunity_rank"] = rank
    return rows


def _candidate_source(
    *,
    distance_from_incumbent: int,
    cell: tuple[int, int],
    incumbent_cell: tuple[int, int],
    risk: float,
    incumbent_risk: float,
    cost: float,
    incumbent_cost: float,
    risk_margin: float,
    cost_margin: float,
) -> str:
    if cell == incumbent_cell:
        return "incumbent_neighborhood"
    if distance_from_incumbent < 3:
        return "incumbent_neighborhood"
    if risk <= incumbent_risk + risk_margin + TOLERANCE and cost <= incumbent_cost + cost_margin + TOLERANCE:
        return "frontier_boundary"
    if distance_from_incumbent >= 5:
        return "roi_undercovered_boundary"
    return "incumbent_neighborhood"


def _geometric_coverage_metrics(
    config: dict[str, Any],
    start_cell: tuple[int, int],
    cell: tuple[int, int],
    incumbent_cell: tuple[int, int],
    source: str,
    roi_group: str,
) -> dict[str, float]:
    denominator = float(config["coverage_denominator_cells"])
    distance_from_incumbent = _manhattan(cell, incumbent_cell)
    distance_from_start = _manhattan(start_cell, cell)
    source_bonus = 0
    if source == "frontier_boundary":
        source_bonus = 6
    elif source == "roi_undercovered_boundary":
        source_bonus = 10
    endpoint_count = max(1, distance_from_incumbent * 3 + source_bonus)
    path_line_count = max(1, int(round(distance_from_start * 0.5 + distance_from_incumbent * 2 + source_bonus)))
    overlap_count = max(0, min(endpoint_count, path_line_count) // 3)
    total_count = max(endpoint_count, path_line_count) + max(0, distance_from_incumbent - overlap_count)
    if source == "incumbent_neighborhood":
        total_count = min(total_count, max(1, distance_from_start + 2))
    roi_weight = _roi_weight(roi_group)
    return {
        "endpoint_new_cell_count": float(endpoint_count),
        "path_line_new_cell_count": float(path_line_count),
        "coverage_overlap_count": float(overlap_count),
        "total_new_cell_count": float(total_count),
        "endpoint_coverage_delta": float(endpoint_count) / denominator,
        "path_line_coverage_delta": float(path_line_count) / denominator,
        "expected_coverage_rate_delta": float(total_count) / denominator,
        "roi_weighted_coverage_delta": float(total_count) * roi_weight / denominator,
        "revisit_penalty": float(overlap_count) / max(float(endpoint_count + path_line_count), 1.0),
        "coverage_overlap_ratio": float(overlap_count) / max(float(total_count), 1.0),
    }


def _select_formal_candidates(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
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
        incumbent = [row for row in rows if _cell_tuple(row.get("cell")) == _cell_tuple(row.get("true_incumbent_cell"))]
        if incumbent:
            return incumbent[0]
        return min(rows, key=lambda row: (_risk_value(row), _path_cost(row), -float(row["expected_coverage_rate_delta"])))
    if source == "frontier_boundary":
        return max(rows, key=lambda row: (bool(row["safe_efficient_candidate"]), float(row["expected_coverage_rate_delta"]), -_risk_value(row), -_path_cost(row)))
    return max(rows, key=lambda row: (float(row["expected_coverage_rate_delta"]), -_risk_value(row), -_path_cost(row)))


def _stable_candidate_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            0
            if row.get("frontier_candidate_source") == "incumbent_neighborhood"
            else 1
            if row.get("safe_efficient_candidate")
            else 2
            if row.get("frontier_candidate_source") == "roi_undercovered_boundary"
            else 3,
            -float(row["expected_coverage_rate_delta"]),
            _risk_value(row),
            _path_cost(row),
            tuple(row.get("cell") or []),
        ),
    )


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    blocking_reasons = list(metrics["reason_codes"]) + list(source["reason_codes"])
    diagnostic_reasons: list[str] = []
    status = "passed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE
    required_validated = int(config["required_scenario_count"]) * 3
    if any(reason.startswith("missing_stage18a") for reason in source["reason_codes"]):
        status = "failed"
        next_required_change = FIX_STAGE18A_NEXT_REQUIRED_CHANGE
    elif any(reason in source["reason_codes"] for reason in ("missing_true_incumbent_binding", "fallback_action_index_0_not_allowed")):
        status = "failed"
        next_required_change = FIX_BINDING_NEXT_REQUIRED_CHANGE
    elif "missing_stage18h0_quantization" in source["reason_codes"]:
        status = "failed"
        next_required_change = FIX_QUANTIZATION_NEXT_REQUIRED_CHANGE
    elif "candidate_cell_mismatch" in blocking_reasons:
        status = "failed"
        next_required_change = FIX_BINDING_NEXT_REQUIRED_CHANGE
    elif metrics["path_feedback_validation_missing_count"] > 0:
        blocking_reasons.append("path_feedback_validation_missing")
        status = "failed"
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    elif metrics["open_grid_fallback_count"] > 0:
        blocking_reasons.append("open_grid_fallback_candidate")
        status = "failed"
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    elif metrics["validated_candidate_count"] < required_validated:
        blocking_reasons.append("path_feedback_validation_missing")
        status = "failed"
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    if metrics["frontier_boundary_candidate_count"] <= 0:
        diagnostic_reasons.append("frontier_boundary_candidate_missing")
    if metrics["roi_undercovered_boundary_candidate_count"] <= 0:
        diagnostic_reasons.append("roi_undercovered_boundary_candidate_missing")
    if metrics["candidate_coverage_spread_range"] <= 0.005 or metrics["roi_group_with_nonzero_spread_count"] < 3:
        diagnostic_reasons.append("proposal_coverage_spread_insufficient")
    if metrics["safe_efficient_candidate_count"] <= 0:
        diagnostic_reasons.append("safe_efficient_candidate_missing")
    if metrics["roi_group_with_safe_efficient_candidate_count"] < 3:
        diagnostic_reasons.append("safe_efficient_roi_spread_insufficient")
    if metrics.get("coverage_positive_but_risk_regressive_count", 0) > 0:
        diagnostic_reasons.append("coverage_positive_but_risk_regressive")
    if metrics.get("coverage_positive_but_cost_regressive_count", 0) > 0:
        diagnostic_reasons.append("coverage_positive_but_cost_regressive")
    blocking_reasons = unique_sorted(blocking_reasons)
    diagnostic_reasons = unique_sorted(diagnostic_reasons)
    evidence_gate = not any(
        reason
        in {
            "missing_true_incumbent_binding",
            "fallback_action_index_0_not_allowed",
            "candidate_cell_mismatch",
            "missing_stage18h0_quantization",
        }
        or reason.startswith("missing_stage18a")
        for reason in blocking_reasons + list(source["reason_codes"])
    )
    candidate_gate = not any(
        reason in {"path_feedback_validation_missing", "open_grid_fallback_candidate", "no_valid_candidates"}
        for reason in blocking_reasons
    )
    comparison_allowed = evidence_gate and candidate_gate and status == "passed"
    diagnostic_recommended_change = _diagnostic_recommended_change(diagnostic_reasons)
    return {
        "status": status,
        "reason_codes": blocking_reasons,
        "blocking_reason_codes": blocking_reasons,
        "diagnostic_reason_codes": diagnostic_reasons,
        "diagnostic_recommended_change": diagnostic_recommended_change,
        "evidence_authenticity_gate_passed": evidence_gate,
        "candidate_validity_gate_passed": candidate_gate,
        "comparison_allowed": comparison_allowed,
        "next_required_change": next_required_change,
    }


def _diagnostic_recommended_change(diagnostic_reasons: list[str]) -> str:
    reason_set = set(diagnostic_reasons)
    if {"frontier_boundary_candidate_missing", "roi_undercovered_boundary_candidate_missing", "proposal_coverage_spread_insufficient"} & reason_set:
        return FIX_SAMPLING_NEXT_REQUIRED_CHANGE
    if {"safe_efficient_candidate_missing", "coverage_positive_but_risk_regressive", "coverage_positive_but_cost_regressive"} & reason_set:
        return FIX_RISK_AWARE_GENERATION_NEXT_REQUIRED_CHANGE
    if "safe_efficient_roi_spread_insufficient" in reason_set:
        return EXPAND_ROI_OR_MAP_NEXT_REQUIRED_CHANGE
    return ""


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
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "diagnostic_reason_codes": decision["diagnostic_reason_codes"],
        "diagnostic_recommended_change": decision["diagnostic_recommended_change"],
        "evidence_authenticity_gate_passed": decision["evidence_authenticity_gate_passed"],
        "candidate_validity_gate_passed": decision["candidate_validity_gate_passed"],
        "comparison_allowed": decision["comparison_allowed"],
        "next_required_change": decision["next_required_change"],
        **_boundary_fields(),
        "git": git_snapshot(repo_root),
    }
    payload["canary_traffic_fraction"] = float(config["canary_traffic_fraction"])
    return payload


def _generated_expansion_summary(expansion_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(expansion_summary)
    payload["frontier_nbv_candidate_repair_status"] = summary["status"]
    payload["frontier_nbv_candidate_repair_source"] = GENERATION_SOURCE
    payload["candidate_count"] = summary["candidate_count"]
    payload["safe_efficient_candidate_count"] = summary["safe_efficient_candidate_count"]
    payload["frontier_boundary_candidate_count"] = summary["frontier_boundary_candidate_count"]
    payload["roi_undercovered_boundary_candidate_count"] = summary["roi_undercovered_boundary_candidate_count"]
    payload["next_required_change"] = COMPATIBLE_EXPANSION_NEXT_REQUIRED_CHANGE if summary["status"] == "passed" else summary["next_required_change"]
    payload.update(_boundary_fields())
    payload["canary_traffic_fraction"] = summary["canary_traffic_fraction"]
    return payload


def _materialization_summary(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "xunce-candidate-level-coverage-opportunity-summary/v1",
        "status": summary["status"],
        "reason_codes": summary["reason_codes"],
        "next_required_change": summary["next_required_change"],
        "candidate_coverage_spread_range": summary["candidate_coverage_spread_range"],
        "roi_group_with_nonzero_spread_count": summary["roi_group_with_nonzero_spread_count"],
        "safe_efficient_candidate_count": summary["safe_efficient_candidate_count"],
        "safe_efficient_opportunity_count": summary["safe_efficient_candidate_count"],
        "roi_group_with_safe_efficient_candidate_count": summary["roi_group_with_safe_efficient_candidate_count"],
        "coverage_opportunity_source": COVERAGE_SOURCE,
        **_boundary_fields(),
    }
    payload["canary_traffic_fraction"] = summary["canary_traffic_fraction"]
    return payload


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18I.2 Risk-Aware Frontier-NBV Candidate Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- scenario_count: `{summary['scenario_count']}`",
            f"- candidate_count: `{summary['candidate_count']}`",
            f"- frontier_boundary_candidate_count: `{summary['frontier_boundary_candidate_count']}`",
            f"- roi_undercovered_boundary_candidate_count: `{summary['roi_undercovered_boundary_candidate_count']}`",
            f"- safe_efficient_candidate_count: `{summary['safe_efficient_candidate_count']}`",
            "",
            "This is an offline unbound candidate root. It does not train models, publish checkpoints, replace the default policy, connect an executor, or start an online canary.",
            "",
        ]
    )


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = feedback.get("candidates") if isinstance(feedback.get("candidates"), list) else []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _candidate_by_action_index(candidates: list[dict[str, Any]], action_index: int) -> dict[str, Any] | None:
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


def _start_cell(
    scenario: dict[str, Any],
    binding: dict[str, Any],
    candidates: list[dict[str, Any]],
    incumbent_cell: tuple[int, int],
    scenario_index: int,
) -> tuple[int, int]:
    for value in (scenario.get("start_cell"), binding.get("start_cell")):
        cell = _cell_tuple(value)
        if cell is not None:
            return cell
    cells = [_cell_tuple(row.get("cell")) for row in candidates]
    cells = [row for row in cells if row is not None]
    if cells:
        min_x = min(row[0] for row in cells)
        min_y = min(row[1] for row in cells)
        return (min_x, min_y)
    return (incumbent_cell[0] - scenario_index - 1, incumbent_cell[1])


def _pareto_frontier_count(candidates: list[dict[str, Any]]) -> int:
    return sum(1 for row in candidates if not _is_dominated(row, candidates))


def _is_dominated(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    cov = float(candidate.get("expected_coverage_rate_delta", 0.0))
    risk = _risk_value(candidate)
    cost = _path_cost(candidate)
    for other in candidates:
        if other is candidate:
            continue
        other_cov = float(other.get("expected_coverage_rate_delta", 0.0))
        other_risk = _risk_value(other)
        other_cost = _path_cost(other)
        if (
            other_cov >= cov - TOLERANCE
            and other_risk <= risk + TOLERANCE
            and other_cost <= cost + TOLERANCE
            and (other_cov > cov + TOLERANCE or other_risk < risk - TOLERANCE or other_cost < cost - TOLERANCE)
        ):
            return True
    return False


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
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _roi_weight(roi_group: str) -> float:
    weights = {
        "smooth_high_confidence": 1.0,
        "mixed_risk": 1.2,
        "rim_or_steep_slope": 1.4,
        "shadowed_transition": 1.5,
        "low_observation_count": 1.6,
        "crater_rim_fragmented": 1.5,
        "low_sun_roughness": 1.4,
        "mixed_passability_edge": 1.5,
    }
    return float(weights.get(roi_group, 1.2))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _mean(values: list[float] | list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= TOLERANCE:
        return None
    return numerator / denominator


def _risk_value(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 1.0e9
    return _float(candidate.get("path_risk_peak", candidate.get("risk", 1.0e9)), 1.0e9)


def _path_cost(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 1.0e9
    return _float(candidate.get("path_cost", candidate.get("cost", 1.0e9)), 1.0e9)


def _has_finite_number(candidate: dict[str, Any], key: str) -> bool:
    if key not in candidate:
        return False
    try:
        value = float(candidate.get(key))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


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
