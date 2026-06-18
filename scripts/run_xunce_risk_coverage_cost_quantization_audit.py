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


CONFIG_SCHEMA_VERSION = "xunce-risk-coverage-cost-quantization-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-risk-coverage-cost-quantization-summary/v1"
DEFAULT_CONFIG = "configs/xunce_risk_coverage_cost_quantization_audit_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_risk_coverage_cost_quantization_audit_v1"

BINDING_SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
MATERIALIZATION_SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"
EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

SUMMARY_FILE = "xunce-risk-coverage-cost-quantization-summary.json"
CANDIDATES_FILE = "xunce-risk-coverage-cost-candidates.jsonl"
SCENARIOS_FILE = "xunce-risk-coverage-cost-scenarios.jsonl"
ROI_SUMMARY_FILE = "xunce-risk-coverage-cost-roi-summary.json"
PARETO_AUDIT_FILE = "xunce-risk-coverage-cost-pareto-audit.json"
DECISION_AUDIT_FILE = "xunce-risk-coverage-cost-decision-audit.json"
REPORT_FILE = "xunce-risk-coverage-cost-report.md"

PASS_NEXT_REQUIRED_CHANGE = "rerun_oracle_separability_with_quantized_root"
FIX_BINDING_NEXT_REQUIRED_CHANGE = "run_true_incumbent_selection_binding"
FIX_VALIDATION_NEXT_REQUIRED_CHANGE = "repair_path_feedback_candidate_validation"
FIX_COUPLING_NEXT_REQUIRED_CHANGE = "repair_metric_family_separation"
FIX_NORMALIZATION_NEXT_REQUIRED_CHANGE = "repair_metric_normalization"
CALIBRATE_RISK_NEXT_REQUIRED_CHANGE = "calibrate_risk_margin_or_risk_attribution"
REPAIR_RISK_GENERATION_NEXT_REQUIRED_CHANGE = "repair_risk_aware_candidate_generation"
EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE = "expand_roi_or_map_complexity"

TOLERANCE = 1.0e-12
COVERAGE_SOURCE = "geometric_counterfactual_from_stage18a_candidate/v1"
COVERAGE_CELL_SET_KIND = "path_line_plus_endpoint_union"
COVERAGE_DEDUPE_SCOPE = "scenario_step_new_cells"
RISK_SOURCE = "candidate_risk_scalar_reused_for_path_risk/v1"
COST_SOURCE = "path_feedback_candidate/v1"

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
    parser = argparse.ArgumentParser(description="Audit decoupled Xunce risk, coverage, and cost candidate metrics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-bound-coverage-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_bound_coverage_root": args.source_bound_coverage_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_risk_coverage_cost_quantization_audit(
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
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_risk_coverage_cost_quantization_audit(
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
    quantized, candidate_rows, scenario_rows, roi_summary, pareto_audit, metrics = _quantize(config, source)
    decision = _decision(config, source, metrics)
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
    write_jsonl(paths["candidates"], candidate_rows)
    write_jsonl(paths["scenarios"], scenario_rows)
    write_json(paths["roi_summary"], roi_summary)
    write_json(paths["pareto_audit"], pareto_audit)
    write_json(paths["decision_audit"], _decision_audit(source, metrics, decision))
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    write_json(paths["expansion_summary"], _quantized_expansion_summary(source["expansion_summary"], summary))
    write_jsonl(paths["slices"], source["slices"])
    write_json(paths["path_feedback"], quantized)
    write_json(paths["materialization_summary"], _quantized_materialization_summary(source["materialization_summary"], summary))
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
    value = payload.get("source_bound_coverage_root")
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("source_bound_coverage_root must be a non-empty string")
    normalized["source_bound_coverage_root"] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["coverage_denominator_cells"] = _positive_float(payload.get("coverage_denominator_cells", 1000), "coverage_denominator_cells")
    normalized["coverage_margin"] = _nonnegative_float(payload.get("coverage_margin", 0.0), "coverage_margin")
    normalized["risk_margin"] = _nonnegative_float(payload.get("risk_margin", 0.01), "risk_margin")
    normalized["cost_margin"] = _nonnegative_float(payload.get("cost_margin", 0.0), "cost_margin")
    normalized["budget_margin"] = _nonnegative_float(payload.get("budget_margin", 0.0), "budget_margin")
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["min_roi_group_with_safe_efficient_candidate"] = _positive_int(payload.get("min_roi_group_with_safe_efficient_candidate", 3), "min_roi_group_with_safe_efficient_candidate")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    bound_root = Path(config["source_bound_coverage_root"])
    reasons: list[str] = []
    binding_summary = _read_json(bound_root / BINDING_SUMMARY_FILE, reasons, "missing_true_incumbent_binding")
    materialization_summary = _read_json(bound_root / MATERIALIZATION_SUMMARY_FILE, [], "missing_stage18e_materialization")
    expansion_summary = _read_json(bound_root / EXPANSION_SUMMARY_FILE, reasons, "missing_true_incumbent_binding")
    slices = _read_jsonl(bound_root / EXPANSION_SLICES_FILE, reasons, "missing_true_incumbent_binding")
    path_feedback = _read_json(bound_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_true_incumbent_binding")
    if binding_summary.get("true_incumbent_selection_bound") is not True:
        reasons.append("missing_true_incumbent_binding")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    return {
        "bound_root": bound_root,
        "binding_summary": binding_summary,
        "materialization_summary": materialization_summary,
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "path_feedback": path_feedback,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "candidates": output_root / CANDIDATES_FILE,
        "scenarios": output_root / SCENARIOS_FILE,
        "roi_summary": output_root / ROI_SUMMARY_FILE,
        "pareto_audit": output_root / PARETO_AUDIT_FILE,
        "decision_audit": output_root / DECISION_AUDIT_FILE,
        "report": output_root / REPORT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "materialization_summary": output_root / MATERIALIZATION_SUMMARY_FILE,
    }


def _quantize(
    config: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    quantized = dict(source["path_feedback"])
    quantized_scenarios: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    roi_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    risk_axis_counts: Counter[str] = Counter()
    reason_codes = list(source["reason_codes"])

    totals = Counter()
    pareto_frontier_counts: list[int] = []
    normalization_instability_count = 0
    raw_pareto_mismatch_rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(source["scenarios"][: config["required_scenario_count"]]):
        scenario_copy = dict(scenario)
        scenario_id = str(scenario_copy.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario_copy.get("roi_group") or scenario_copy.get("scenario_group") or "unknown")
        candidates = _candidate_rows(scenario_copy)
        incumbent_index, incumbent_source = _incumbent_index(scenario_copy)
        if incumbent_source == "fallback_action_index_0":
            totals["fallback_action_index_0_count"] += 1
        incumbent = _candidate_at(candidates, incumbent_index)
        raw_vectors = [_raw_vector(candidate, config, incumbent) for candidate in candidates]
        normalized_vectors = _normalized_vectors(raw_vectors)
        raw_statuses = _pareto_statuses(raw_vectors)
        normalized_statuses = _pareto_statuses(normalized_vectors)
        scenario_quantized_candidates: list[dict[str, Any]] = []
        scenario_safe = 0
        scenario_coverage_positive = 0
        scenario_risk_regressive = 0
        scenario_cost_regressive = 0
        scenario_finite_issues = 0
        scenario_missing_atomic = 0
        scenario_normalization_instability = 0

        incumbent_vector = raw_vectors[incumbent_index] if incumbent_index is not None and 0 <= incumbent_index < len(raw_vectors) else _empty_vector(config)

        for action_index, candidate in enumerate(candidates):
            totals["candidate_count"] += 1
            raw = raw_vectors[action_index]
            normalized = normalized_vectors[action_index]
            finite_issue = _finite_issue_count(raw)
            missing_atomic = _missing_atomic_metric_count(raw)
            scenario_finite_issues += finite_issue
            scenario_missing_atomic += missing_atomic
            totals["finite_metric_issue_count"] += finite_issue
            totals["missing_atomic_metric_count"] += missing_atomic
            if candidate.get("open_grid_fallback_used") is True or scenario_copy.get("open_grid_fallback_used") is True:
                totals["open_grid_fallback_count"] += 1
            relative = _relative_to_incumbent(raw, incumbent_vector, config, candidate)
            if raw_statuses[action_index] != normalized_statuses[action_index]:
                normalization_instability_count += 1
                scenario_normalization_instability += 1
                raw_pareto_mismatch_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "action_index": action_index,
                        "raw_pareto_status": raw_statuses[action_index],
                        "normalized_pareto_status": normalized_statuses[action_index],
                    }
                )
            failure_primary_axis, failure_reasons = _failure(candidate, relative)
            if relative["coverage_guard_passed"]:
                totals["coverage_positive_candidate_count"] += 1
                scenario_coverage_positive += 1
                if not relative["risk_guard_passed"]:
                    totals["coverage_positive_but_risk_regressive_count"] += 1
                    scenario_risk_regressive += 1
                    risk_axis_counts[raw["risk_vector"]["risk_attribution_primary_axis"]] += 1
                if not relative["cost_guard_passed"]:
                    totals["coverage_positive_but_cost_regressive_count"] += 1
                    scenario_cost_regressive += 1
            if relative["unvalidated_positive_candidate"]:
                totals["path_feedback_validation_missing_count"] += 1
            if relative["safe_efficient_candidate"]:
                totals["safe_efficient_candidate_count"] += 1
                scenario_safe += 1
            if relative["hard_validity_passed"]:
                totals["valid_candidate_count"] += 1
            else:
                totals["invalid_candidate_count"] += 1
                for reason in failure_reasons:
                    totals[f"invalid_candidate_reason:{reason}"] += 1

            enriched = dict(candidate)
            enriched.update(
                {
                    "coverage_vector": raw["coverage_vector"],
                    "risk_vector": raw["risk_vector"],
                    "cost_vector": raw["cost_vector"],
                    "coverage_vector_norm": normalized["coverage_vector"],
                    "risk_vector_norm": normalized["risk_vector"],
                    "cost_vector_norm": normalized["cost_vector"],
                    "relative_to_incumbent": relative,
                    "coverage_delta_vs_incumbent": relative["coverage_delta_vs_incumbent"],
                    "risk_delta_vs_incumbent": relative["risk_delta_vs_incumbent"],
                    "cost_delta_vs_incumbent": relative["cost_delta_vs_incumbent"],
                    "budget_delta_vs_incumbent": relative["budget_delta_vs_incumbent"],
                    "coverage_guard_passed": relative["coverage_guard_passed"],
                    "risk_guard_passed": relative["risk_guard_passed"],
                    "cost_guard_passed": relative["cost_guard_passed"],
                    "hard_validity_passed": relative["hard_validity_passed"],
                    "safe_efficient_candidate": relative["safe_efficient_candidate"],
                    "safe_efficient_opportunity": relative["safe_efficient_opportunity"],
                    "pareto_status": raw_statuses[action_index],
                    "normalized_pareto_status": normalized_statuses[action_index],
                    "normalization_instability": raw_statuses[action_index] != normalized_statuses[action_index],
                    "failure_primary_axis": failure_primary_axis,
                    "failure_reason_codes": failure_reasons,
                    "risk_coverage_cost_quantization_source": "stage18h0_decoupled_atomic_vectors/v1",
                }
            )
            scenario_quantized_candidates.append(enriched)
            candidate_rows.append(
                {
                    "schema_version": "xunce-risk-coverage-cost-candidate/v1",
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "action_index": int(enriched.get("action_index", action_index)),
                    "coverage_vector": raw["coverage_vector"],
                    "risk_vector": raw["risk_vector"],
                    "cost_vector": raw["cost_vector"],
                    "coverage_vector_norm": normalized["coverage_vector"],
                    "risk_vector_norm": normalized["risk_vector"],
                    "cost_vector_norm": normalized["cost_vector"],
                    "relative_to_incumbent": relative,
                    "pareto_status": raw_statuses[action_index],
                    "normalized_pareto_status": normalized_statuses[action_index],
                    "safe_efficient_candidate": relative["safe_efficient_candidate"],
                    "safe_efficient_opportunity": relative["safe_efficient_opportunity"],
                    "failure_primary_axis": failure_primary_axis,
                    "failure_reason_codes": failure_reasons,
                }
            )

        frontier_count = sum(1 for status in raw_statuses if status == "pareto_frontier")
        pareto_frontier_counts.append(frontier_count)
        if scenario_safe > 0:
            totals["safe_efficient_candidate_scenario_count"] += 1
        scenario_row = {
            "schema_version": "xunce-risk-coverage-cost-scenario/v1",
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "candidate_count": len(candidates),
            "incumbent_selected_action_index": incumbent_index,
            "incumbent_selection_source": incumbent_source,
            "safe_efficient_candidate_count": scenario_safe,
            "coverage_positive_candidate_count": scenario_coverage_positive,
            "coverage_positive_but_risk_regressive_count": scenario_risk_regressive,
            "coverage_positive_but_cost_regressive_count": scenario_cost_regressive,
            "candidate_pareto_frontier_count": frontier_count,
            "finite_metric_issue_count": scenario_finite_issues,
            "missing_atomic_metric_count": scenario_missing_atomic,
            "normalization_instability_count": scenario_normalization_instability,
        }
        scenario_rows.append(scenario_row)
        roi_groups[roi_group].append(scenario_row)
        scenario_copy["path_feedback"] = dict(scenario_copy.get("path_feedback") or {})
        scenario_copy["path_feedback"]["candidates"] = scenario_quantized_candidates
        quantized_scenarios.append(scenario_copy)

    totals["scenario_count"] = len(quantized_scenarios)
    roi_rows = []
    for roi_group, rows in sorted(roi_groups.items()):
        safe = sum(row["safe_efficient_candidate_count"] for row in rows)
        roi_rows.append(
            {
                "roi_group": roi_group,
                "scenario_count": len(rows),
                "candidate_count": sum(row["candidate_count"] for row in rows),
                "safe_efficient_candidate_count": safe,
                "coverage_positive_candidate_count": sum(row["coverage_positive_candidate_count"] for row in rows),
                "has_safe_efficient_candidate": safe > 0,
            }
        )
    roi_summary = {
        "schema_version": "xunce-risk-coverage-cost-roi-summary/v1",
        "roi_group_count": len(roi_rows),
        "roi_group_with_safe_efficient_candidate_count": sum(1 for row in roi_rows if row["has_safe_efficient_candidate"]),
        "families": roi_rows,
    }
    quantized["scenarios"] = quantized_scenarios
    quantized["scenario_count"] = len(quantized_scenarios)
    quantized["candidate_count"] = totals["candidate_count"]
    quantized["risk_coverage_cost_quantization_source"] = "stage18h0_decoupled_atomic_vectors/v1"
    metrics = {
        "reason_codes": unique_sorted(reason_codes),
        "scenario_count": totals["scenario_count"],
        "candidate_count": totals["candidate_count"],
        "metric_coupling_detected": False,
        "finite_metric_issue_count": totals["finite_metric_issue_count"],
        "missing_atomic_metric_count": totals["missing_atomic_metric_count"],
        "fallback_action_index_0_count": totals["fallback_action_index_0_count"],
        "open_grid_fallback_count": totals["open_grid_fallback_count"],
        "safe_efficient_candidate_count": totals["safe_efficient_candidate_count"],
        "safe_efficient_candidate_scenario_count": totals["safe_efficient_candidate_scenario_count"],
        "roi_group_with_safe_efficient_candidate_count": roi_summary["roi_group_with_safe_efficient_candidate_count"],
        "coverage_positive_candidate_count": totals["coverage_positive_candidate_count"],
        "coverage_positive_but_risk_regressive_count": totals["coverage_positive_but_risk_regressive_count"],
        "coverage_positive_but_cost_regressive_count": totals["coverage_positive_but_cost_regressive_count"],
        "risk_regression_primary_axis_counts": dict(sorted(risk_axis_counts.items())),
        "candidate_pareto_frontier_count": _mean([float(value) for value in pareto_frontier_counts]),
        "normalization_instability_count": normalization_instability_count,
        "path_feedback_validation_missing_count": totals["path_feedback_validation_missing_count"],
        "valid_candidate_count": totals["valid_candidate_count"],
        "invalid_candidate_count": totals["invalid_candidate_count"],
        "valid_scenario_count": sum(1 for row in scenario_rows if row["candidate_count"] > 0),
        "invalid_scenario_count": sum(1 for row in scenario_rows if row["candidate_count"] <= 0),
        "invalid_candidate_reason_counts": {
            key.split(":", 1)[1]: value
            for key, value in sorted(totals.items())
            if key.startswith("invalid_candidate_reason:")
        },
    }
    pareto_audit = {
        "schema_version": "xunce-risk-coverage-cost-pareto-audit/v1",
        "candidate_pareto_frontier_count": metrics["candidate_pareto_frontier_count"],
        "normalization_instability_count": normalization_instability_count,
        "raw_normalized_mismatch_rows": raw_pareto_mismatch_rows,
    }
    return quantized, candidate_rows, scenario_rows, roi_summary, pareto_audit, metrics


def _raw_vector(candidate: dict[str, Any] | None, config: dict[str, Any], incumbent: dict[str, Any] | None) -> dict[str, Any]:
    denominator = float(config["coverage_denominator_cells"])
    endpoint_count = _count_from_candidate(candidate, "endpoint_new_cell_count", "endpoint_coverage_delta", denominator)
    path_line_count = _count_from_candidate(candidate, "path_line_new_cell_count", "path_line_coverage_delta", denominator)
    expected_new_count = _count_from_candidate(candidate, "expected_new_coverage_cell_count", "expected_coverage_rate_delta", denominator)
    total_new_count = _first_finite(
        candidate.get("total_new_cell_count") if candidate else None,
        candidate.get("expected_new_coverage_cell_count") if candidate else None,
        max(endpoint_count, path_line_count, expected_new_count),
    )
    overlap_count = _first_finite(candidate.get("coverage_overlap_count") if candidate else None, 0.0)
    overlap_ratio = _first_finite(candidate.get("coverage_overlap_ratio") if candidate else None, 0.0)
    revisit_count = _first_finite(
        candidate.get("revisited_cell_count") if candidate else None,
        _float(candidate.get("revisit_penalty")) * max(total_new_count + overlap_count, 1.0) if candidate else 0.0,
    )
    risk = _risk(candidate)
    incumbent_risk = _risk(incumbent)
    path_cost = _path_cost(candidate)
    path_risk_exposure = path_cost if risk > incumbent_risk + float(config["risk_margin"]) else 0.0
    return {
        "coverage_vector": {
            "endpoint_new_cell_count": endpoint_count,
            "path_line_new_cell_count": path_line_count,
            "roi_weighted_coverage_delta": _coverage_value(candidate),
            "expected_new_coverage_cell_count": expected_new_count,
            "total_new_cell_count": total_new_count,
            "revisited_cell_count": revisit_count,
            "coverage_overlap_count": overlap_count,
            "coverage_overlap_ratio": overlap_ratio,
            "coverage_source": COVERAGE_SOURCE,
            "coverage_cell_set_kind": COVERAGE_CELL_SET_KIND,
            "coverage_dedupe_scope": COVERAGE_DEDUPE_SCOPE,
        },
        "risk_vector": {
            "candidate_point_risk": risk,
            "path_risk_mean": risk,
            "path_risk_peak": risk,
            "path_risk_p95": risk,
            "path_risk_exposure_length": path_risk_exposure,
            "risk_source": RISK_SOURCE,
            "risk_attribution_primary_axis": "path_risk_peak" if risk > incumbent_risk + float(config["risk_margin"]) else "none",
        },
        "cost_vector": {
            "path_cost": path_cost,
            "budget_used_ratio": _safe_ratio(path_cost, float(config["path_budget_m"])),
            "planning_latency_ms": _first_finite(candidate.get("planning_latency_ms") if candidate else None, candidate.get("latency_ms") if candidate else None, 0.0),
            "cost_source": COST_SOURCE,
        },
    }


def _normalized_vectors(raw_vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage_values = [float(row["coverage_vector"]["total_new_cell_count"]) for row in raw_vectors]
    risk_values = [float(row["risk_vector"]["path_risk_peak"]) for row in raw_vectors]
    cost_values = [float(row["cost_vector"]["path_cost"]) for row in raw_vectors]
    normalized = []
    for row in raw_vectors:
        normalized.append(
            {
                "coverage_vector": {
                    "total_new_cell_count": _normalize(float(row["coverage_vector"]["total_new_cell_count"]), coverage_values),
                    "endpoint_new_cell_count": _normalize(float(row["coverage_vector"]["endpoint_new_cell_count"]), coverage_values),
                    "path_line_new_cell_count": _normalize(float(row["coverage_vector"]["path_line_new_cell_count"]), coverage_values),
                    "revisited_cell_count": _normalize(float(row["coverage_vector"]["revisited_cell_count"]), [float(item["coverage_vector"]["revisited_cell_count"]) for item in raw_vectors]),
                    "coverage_overlap_count": _normalize(float(row["coverage_vector"]["coverage_overlap_count"]), [float(item["coverage_vector"]["coverage_overlap_count"]) for item in raw_vectors]),
                },
                "risk_vector": {
                    "path_risk_peak": _normalize(float(row["risk_vector"]["path_risk_peak"]), risk_values),
                    "path_risk_mean": _normalize(float(row["risk_vector"]["path_risk_mean"]), risk_values),
                    "path_risk_p95": _normalize(float(row["risk_vector"]["path_risk_p95"]), risk_values),
                },
                "cost_vector": {
                    "path_cost": _normalize(float(row["cost_vector"]["path_cost"]), cost_values),
                    "budget_used_ratio": _normalize(float(row["cost_vector"]["budget_used_ratio"]), [float(item["cost_vector"]["budget_used_ratio"]) for item in raw_vectors]),
                    "planning_latency_ms": _normalize(float(row["cost_vector"]["planning_latency_ms"]), [float(item["cost_vector"]["planning_latency_ms"]) for item in raw_vectors]),
                },
            }
        )
    return normalized


def _relative_to_incumbent(raw: dict[str, Any], incumbent: dict[str, Any], config: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    coverage_delta = float(raw["coverage_vector"]["total_new_cell_count"]) - float(incumbent["coverage_vector"]["total_new_cell_count"])
    risk_delta = float(raw["risk_vector"]["path_risk_peak"]) - float(incumbent["risk_vector"]["path_risk_peak"])
    cost_delta = float(raw["cost_vector"]["path_cost"]) - float(incumbent["cost_vector"]["path_cost"])
    budget_delta = float(raw["cost_vector"]["budget_used_ratio"]) - float(incumbent["cost_vector"]["budget_used_ratio"])
    hard_validity = _hard_valid(candidate)
    coverage_guard = coverage_delta > float(config["coverage_margin"])
    risk_guard = float(raw["risk_vector"]["path_risk_peak"]) <= float(incumbent["risk_vector"]["path_risk_peak"]) + float(config["risk_margin"])
    cost_guard = (
        float(raw["cost_vector"]["path_cost"]) <= float(incumbent["cost_vector"]["path_cost"]) + float(config["cost_margin"])
        and float(raw["cost_vector"]["budget_used_ratio"]) <= float(incumbent["cost_vector"]["budget_used_ratio"]) + float(config["budget_margin"])
    )
    safe = bool(hard_validity and coverage_guard and risk_guard and cost_guard)
    return {
        "coverage_delta_vs_incumbent": coverage_delta,
        "risk_delta_vs_incumbent": risk_delta,
        "cost_delta_vs_incumbent": cost_delta,
        "budget_delta_vs_incumbent": budget_delta,
        "coverage_guard_passed": coverage_guard,
        "risk_guard_passed": risk_guard,
        "cost_guard_passed": cost_guard,
        "hard_validity_passed": hard_validity,
        "safe_efficient_candidate": safe,
        "safe_efficient_opportunity": safe,
        "unvalidated_positive_candidate": bool(coverage_guard and (candidate or {}).get("proposal_validated_by_path_feedback") is False),
    }


def _failure(candidate: dict[str, Any] | None, relative: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not relative["hard_validity_passed"]:
        reasons.append("hard_validity_failed")
    if not relative["coverage_guard_passed"]:
        reasons.append("coverage_guard_failed")
    if not relative["risk_guard_passed"]:
        reasons.append("risk_guard_failed")
    if not relative["cost_guard_passed"]:
        reasons.append("cost_guard_failed")
    if (candidate or {}).get("proposal_validated_by_path_feedback") is False:
        reasons.append("path_feedback_validation_missing")
    if (candidate or {}).get("proposal_only") is True:
        reasons.append("proposal_only_not_counted")
    if (candidate or {}).get("open_grid_fallback_used") is True:
        reasons.append("open_grid_fallback_candidate")
    if not reasons:
        return "none", []
    if "path_feedback_validation_missing" in reasons:
        return "validation", unique_sorted(reasons)
    if "risk_guard_failed" in reasons:
        return "risk", unique_sorted(reasons)
    if "cost_guard_failed" in reasons:
        return "cost", unique_sorted(reasons)
    if "coverage_guard_failed" in reasons:
        return "coverage", unique_sorted(reasons)
    return "validity", unique_sorted(reasons)


def _pareto_statuses(vectors: list[dict[str, Any]]) -> list[str]:
    statuses: list[str] = []
    for index, candidate in enumerate(vectors):
        dominated = False
        candidate_coverage = float(candidate["coverage_vector"]["total_new_cell_count"])
        candidate_risk = float(candidate["risk_vector"]["path_risk_peak"])
        candidate_cost = float(candidate["cost_vector"]["path_cost"])
        for other_index, other in enumerate(vectors):
            if other_index == index:
                continue
            other_coverage = float(other["coverage_vector"]["total_new_cell_count"])
            other_risk = float(other["risk_vector"]["path_risk_peak"])
            other_cost = float(other["cost_vector"]["path_cost"])
            weakly_better = other_coverage >= candidate_coverage - TOLERANCE and other_risk <= candidate_risk + TOLERANCE and other_cost <= candidate_cost + TOLERANCE
            strictly_better = other_coverage > candidate_coverage + TOLERANCE or other_risk < candidate_risk - TOLERANCE or other_cost < candidate_cost - TOLERANCE
            if weakly_better and strictly_better:
                dominated = True
                break
        statuses.append("dominated" if dominated else "pareto_frontier")
    return statuses


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    blocking_reasons = list(metrics["reason_codes"])
    diagnostic_reasons: list[str] = []
    if any(reason == "missing_true_incumbent_binding" for reason in blocking_reasons) or metrics["fallback_action_index_0_count"] > 0:
        blocking_reasons.append("missing_true_incumbent_binding")
        return _decision_payload(
            status="failed",
            blocking_reasons=blocking_reasons,
            diagnostic_reasons=diagnostic_reasons,
            next_required_change=FIX_BINDING_NEXT_REQUIRED_CHANGE,
            evidence_gate=False,
            candidate_gate=True,
        )
    if metrics["path_feedback_validation_missing_count"] > 0:
        blocking_reasons.append("path_feedback_validation_missing")
        return _decision_payload(
            status="failed",
            blocking_reasons=blocking_reasons,
            diagnostic_reasons=diagnostic_reasons,
            next_required_change=FIX_VALIDATION_NEXT_REQUIRED_CHANGE,
            evidence_gate=True,
            candidate_gate=False,
        )
    if metrics.get("valid_candidate_count", metrics.get("candidate_count", 0)) <= 0:
        blocking_reasons.append("no_valid_candidates")
        return _decision_payload(
            status="failed",
            blocking_reasons=blocking_reasons,
            diagnostic_reasons=diagnostic_reasons,
            next_required_change=FIX_VALIDATION_NEXT_REQUIRED_CHANGE,
            evidence_gate=True,
            candidate_gate=False,
        )
    if metrics["metric_coupling_detected"]:
        blocking_reasons.append("metric_coupling_detected")
        return _decision_payload(
            status="failed",
            blocking_reasons=blocking_reasons,
            diagnostic_reasons=diagnostic_reasons,
            next_required_change=FIX_COUPLING_NEXT_REQUIRED_CHANGE,
            evidence_gate=True,
            candidate_gate=True,
        )
    if metrics["normalization_instability_count"] > 0:
        blocking_reasons.append("normalization_instability")
        return _decision_payload(
            status="failed",
            blocking_reasons=blocking_reasons,
            diagnostic_reasons=diagnostic_reasons,
            next_required_change=FIX_NORMALIZATION_NEXT_REQUIRED_CHANGE,
            evidence_gate=True,
            candidate_gate=True,
        )
    if metrics["safe_efficient_candidate_count"] <= 0:
        diagnostic_reasons.append("safe_efficient_candidate_missing")
    if metrics["roi_group_with_safe_efficient_candidate_count"] < int(config["min_roi_group_with_safe_efficient_candidate"]):
        diagnostic_reasons.append("safe_efficient_roi_spread_insufficient")
    if metrics["coverage_positive_candidate_count"] == 0:
        diagnostic_reasons.append("roi_or_map_complexity_insufficient")
    if metrics["coverage_positive_but_risk_regressive_count"] >= metrics["coverage_positive_candidate_count"]:
        diagnostic_reasons.append("risk_guard_too_strict_or_miscalibrated")
    elif metrics["coverage_positive_but_risk_regressive_count"] > 0:
        diagnostic_reasons.append("candidate_generation_risk_biased")
    if metrics.get("coverage_positive_but_cost_regressive_count", 0) > 0:
        diagnostic_reasons.append("coverage_positive_but_cost_regressive")
    return _decision_payload(
        status="passed",
        blocking_reasons=blocking_reasons,
        diagnostic_reasons=diagnostic_reasons,
        next_required_change=PASS_NEXT_REQUIRED_CHANGE,
        evidence_gate=True,
        candidate_gate=True,
    )


def _decision_payload(
    *,
    status: str,
    blocking_reasons: list[str],
    diagnostic_reasons: list[str],
    next_required_change: str,
    evidence_gate: bool,
    candidate_gate: bool,
) -> dict[str, Any]:
    blocking_reason_codes = unique_sorted(blocking_reasons)
    diagnostic_reason_codes = unique_sorted(diagnostic_reasons)
    return {
        "status": status,
        "reason_codes": blocking_reason_codes,
        "blocking_reason_codes": blocking_reason_codes,
        "diagnostic_reason_codes": diagnostic_reason_codes,
        "diagnostic_recommended_change": _diagnostic_recommended_change(diagnostic_reason_codes),
        "evidence_authenticity_gate_passed": evidence_gate,
        "candidate_validity_gate_passed": candidate_gate,
        "comparison_allowed": status == "passed" and evidence_gate and candidate_gate,
        "next_required_change": next_required_change,
    }


def _diagnostic_recommended_change(diagnostic_reasons: list[str]) -> str:
    reason_set = set(diagnostic_reasons)
    if {"risk_guard_too_strict_or_miscalibrated"} & reason_set:
        return CALIBRATE_RISK_NEXT_REQUIRED_CHANGE
    if {"candidate_generation_risk_biased", "safe_efficient_candidate_missing", "coverage_positive_but_cost_regressive"} & reason_set:
        return REPAIR_RISK_GENERATION_NEXT_REQUIRED_CHANGE
    if {"roi_or_map_complexity_insufficient", "safe_efficient_roi_spread_insufficient"} & reason_set:
        return EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE
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
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "diagnostic_reason_codes": decision["diagnostic_reason_codes"],
        "diagnostic_recommended_change": decision["diagnostic_recommended_change"],
        "evidence_authenticity_gate_passed": decision["evidence_authenticity_gate_passed"],
        "candidate_validity_gate_passed": decision["candidate_validity_gate_passed"],
        "comparison_allowed": decision["comparison_allowed"],
        "next_required_change": decision["next_required_change"],
        **{key: value for key, value in metrics.items() if key != "reason_codes"},
        "source_bound_coverage_root": config["source_bound_coverage_root"],
        "source_true_incumbent_selection_bound": source["binding_summary"].get("true_incumbent_selection_bound"),
        "coverage_denominator_cells": config["coverage_denominator_cells"],
        "coverage_margin": config["coverage_margin"],
        "risk_margin": config["risk_margin"],
        "cost_margin": config["cost_margin"],
        "budget_margin": config["budget_margin"],
        "path_budget_m": config["path_budget_m"],
        "canary_traffic_fraction": config["canary_traffic_fraction"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "quantized_path_feedback": str(paths["path_feedback"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _decision_audit(source: dict[str, Any], metrics: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-risk-coverage-cost-decision-audit/v1",
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "diagnostic_reason_codes": decision["diagnostic_reason_codes"],
        "diagnostic_recommended_change": decision["diagnostic_recommended_change"],
        "evidence_authenticity_gate_passed": decision["evidence_authenticity_gate_passed"],
        "candidate_validity_gate_passed": decision["candidate_validity_gate_passed"],
        "comparison_allowed": decision["comparison_allowed"],
        "next_required_change": decision["next_required_change"],
        "source_true_incumbent_selection_bound": source["binding_summary"].get("true_incumbent_selection_bound"),
        "metric_coupling_detected": metrics["metric_coupling_detected"],
        "safe_efficient_candidate_count": metrics["safe_efficient_candidate_count"],
        "risk_regression_primary_axis_counts": metrics["risk_regression_primary_axis_counts"],
        **_boundary_fields(),
    }


def _quantized_expansion_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["risk_coverage_cost_quantization_audit_status"] = summary["status"]
    payload["safe_efficient_candidate_count"] = summary["safe_efficient_candidate_count"]
    payload["roi_group_with_safe_efficient_candidate_count"] = summary["roi_group_with_safe_efficient_candidate_count"]
    payload.update(_boundary_fields())
    return payload


def _quantized_materialization_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload.setdefault("schema_version", "xunce-candidate-level-coverage-opportunity-summary/v1")
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["risk_coverage_cost_quantization_audit_status"] = summary["status"]
    payload["safe_efficient_candidate_count"] = summary["safe_efficient_candidate_count"]
    payload["roi_group_with_safe_efficient_candidate_count"] = summary["roi_group_with_safe_efficient_candidate_count"]
    payload.update(_boundary_fields())
    return payload


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Risk-Coverage-Cost Quantization Audit",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- scenario_count: `{summary['scenario_count']}`",
            f"- candidate_count: `{summary['candidate_count']}`",
            f"- metric_coupling_detected: `{summary['metric_coupling_detected']}`",
            f"- safe_efficient_candidate_count: `{summary['safe_efficient_candidate_count']}`",
            f"- roi_group_with_safe_efficient_candidate_count: `{summary['roi_group_with_safe_efficient_candidate_count']}`",
            f"- coverage_positive_but_risk_regressive_count: `{summary['coverage_positive_but_risk_regressive_count']}`",
        ]
    )


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []


def _incumbent_index(scenario: dict[str, Any]) -> tuple[int | None, str]:
    if isinstance(scenario.get("incumbent_selected_action_index"), int):
        return int(scenario["incumbent_selected_action_index"]), str(scenario.get("incumbent_selection_source") or "scenario_incumbent_selected_action_index")
    feedback = scenario.get("path_feedback")
    if isinstance(feedback, dict) and isinstance(feedback.get("incumbent_selected_action_index"), int):
        return int(feedback["incumbent_selected_action_index"]), str(feedback.get("incumbent_selection_source") or "path_feedback_incumbent_selected_action_index")
    return 0, "fallback_action_index_0"


def _candidate_at(candidates: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _hard_valid(candidate: dict[str, Any] | None) -> bool:
    return bool(
        candidate
        and candidate.get("reachable") is True
        and candidate.get("open_grid_fallback_used") is not True
        and candidate.get("proposal_only") is not True
        and candidate.get("proposal_validated_by_path_feedback") is not False
        and math.isfinite(_path_cost(candidate))
        and math.isfinite(_risk(candidate))
    )


def _coverage_value(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    for field in ("roi_weighted_coverage_delta", "expected_coverage_rate_delta", "path_line_coverage_delta", "endpoint_coverage_delta"):
        if field in candidate:
            return _float(candidate.get(field))
    return 0.0


def _path_cost(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("path_cost")) if candidate else 0.0


def _risk(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("risk")) if candidate else 0.0


def _empty_vector(config: dict[str, Any]) -> dict[str, Any]:
    return _raw_vector(None, config, None)


def _count_from_candidate(candidate: dict[str, Any] | None, count_field: str, rate_field: str, denominator: float) -> float:
    if not candidate:
        return 0.0
    if count_field in candidate:
        return _float(candidate.get(count_field))
    return _float(candidate.get(rate_field)) * denominator


def _finite_issue_count(raw: dict[str, Any]) -> int:
    count = 0
    for family in ("coverage_vector", "risk_vector", "cost_vector"):
        for value in raw[family].values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                count += 1
    return count


def _missing_atomic_metric_count(raw: dict[str, Any]) -> int:
    required = {
        "coverage_vector": (
            "endpoint_new_cell_count",
            "path_line_new_cell_count",
            "roi_weighted_coverage_delta",
            "expected_new_coverage_cell_count",
            "total_new_cell_count",
            "revisited_cell_count",
            "coverage_overlap_count",
            "coverage_overlap_ratio",
        ),
        "risk_vector": ("candidate_point_risk", "path_risk_mean", "path_risk_peak", "path_risk_p95", "path_risk_exposure_length"),
        "cost_vector": ("path_cost", "budget_used_ratio", "planning_latency_ms"),
    }
    missing = 0
    for family, fields in required.items():
        for field in fields:
            if field not in raw[family]:
                missing += 1
    return missing


def _first_finite(*values: Any) -> float:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), TOLERANCE)


def _normalize(value: float, values: list[float]) -> float:
    finite_values = [float(item) for item in values if math.isfinite(float(item))]
    if not finite_values:
        return 0.0
    low = min(finite_values)
    high = max(finite_values)
    if abs(high - low) <= TOLERANCE:
        return 0.0
    return (float(value) - low) / (high - low)


def _read_json(path: Path, reasons: list[str], reason_code: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(reason_code)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(reason_code)
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], reason_code: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(reason_code)
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            reasons.append(reason_code)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _boundary_fields() -> dict[str, bool]:
    return {field: False for field in BOUNDARY_FIELDS}


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _positive_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ConfigError(f"{field} must be a positive number")
    return float(value)


def _nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConfigError(f"{field} must be a non-negative number")
    return float(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
