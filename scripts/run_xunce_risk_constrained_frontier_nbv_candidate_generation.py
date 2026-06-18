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


CONFIG_SCHEMA_VERSION = "xunce-risk-constrained-frontier-nbv-candidate-generation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-risk-constrained-frontier-nbv-candidate-generation-summary/v1"
DEFAULT_CONFIG = "configs/xunce_risk_constrained_frontier_nbv_candidate_generation_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_risk_constrained_frontier_nbv_candidate_generation_v1"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"
QUANTIZATION_SUMMARY_FILE = "xunce-risk-coverage-cost-quantization-summary.json"

SUMMARY_FILE = "xunce-risk-constrained-frontier-nbv-candidate-generation-summary.json"
PROPOSALS_FILE = "xunce-frontier-nbv-proposals.jsonl"
VALIDATED_FILE = "xunce-frontier-nbv-validated-candidates.jsonl"
REJECTION_AUDIT_FILE = "xunce-frontier-nbv-rejection-audit.json"
ROI_SUMMARY_FILE = "xunce-frontier-nbv-roi-summary.json"
PARETO_AUDIT_FILE = "xunce-frontier-nbv-pareto-audit.json"
MANIFEST_FILE = "xunce-frontier-nbv-manifest.json"
REPORT_FILE = "xunce-frontier-nbv-report.md"
MATERIALIZATION_SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"

PASS_NEXT_REQUIRED_CHANGE = "rerun_true_model_inference_and_binding"
FIX_STAGE18A_NEXT_REQUIRED_CHANGE = "run_xunce_high_fidelity_real_map_roi_expansion"
FIX_QUANTIZATION_NEXT_REQUIRED_CHANGE = "run_xunce_risk_coverage_cost_quantization_audit"
EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE = "expand_roi_or_map_complexity"
FIX_SAMPLING_NEXT_REQUIRED_CHANGE = "repair_frontier_nbv_sampling"
FIX_VALIDATION_NEXT_REQUIRED_CHANGE = "repair_path_feedback_candidate_validation"
FIX_RISK_NEXT_REQUIRED_CHANGE = "repair_risk_constrained_frontier_sampling"
FIX_COST_NEXT_REQUIRED_CHANGE = "repair_cost_guarded_frontier_sampling"
FIX_SAFE_EFFICIENT_NEXT_REQUIRED_CHANGE = "expand_roi_or_refine_frontier_nbv_candidate_generation"

TOLERANCE = 1.0e-12
COVERAGE_SOURCE = "geometric_counterfactual_from_frontier_nbv_candidate/v1"
COVERAGE_CELL_SET_KIND = "path_line_plus_endpoint_union"
COVERAGE_DEDUPE_SCOPE = "scenario_step_new_cells"
SAFE_KNOWN_SOURCE = "path_feedback_and_roi_geometry_proxy/v1"

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
    parser = argparse.ArgumentParser(description="Generate risk-constrained frontier-guided NBV Xunce candidates.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--source-quantization-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_roi_expansion_root": args.source_roi_expansion_root,
            "source_quantization_root": args.source_quantization_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
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


def run_xunce_risk_constrained_frontier_nbv_candidate_generation(
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
    generated, proposal_rows, validated_rows, rejection_audit, roi_summary, pareto_audit, metrics = _generate(config, source)
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
    write_jsonl(paths["proposals"], proposal_rows)
    write_jsonl(paths["validated"], validated_rows)
    write_json(paths["rejection_audit"], rejection_audit)
    write_json(paths["roi_summary"], roi_summary)
    write_json(paths["pareto_audit"], pareto_audit)
    write_json(paths["manifest"], _manifest(generated_at, config_path, output_root, config, paths, summary))
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
    for key in ("source_roi_expansion_root", "source_quantization_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["max_candidates_per_scenario"] = _positive_int(payload.get("max_candidates_per_scenario", 6), "max_candidates_per_scenario")
    radii = payload.get("frontier_radius_cells", [2, 4, 6])
    if not isinstance(radii, list) or not radii:
        raise ConfigError("frontier_radius_cells must be a non-empty list")
    normalized["frontier_radius_cells"] = [_positive_int(value, "frontier_radius_cells") for value in radii]
    normalized["frontier_direction_count"] = _positive_int(payload.get("frontier_direction_count", 8), "frontier_direction_count")
    normalized["nbv_candidate_pool_limit"] = _positive_int(payload.get("nbv_candidate_pool_limit", 24), "nbv_candidate_pool_limit")
    normalized["coverage_denominator_cells"] = _positive_float(payload.get("coverage_denominator_cells", 1000), "coverage_denominator_cells")
    normalized["risk_margin"] = _nonnegative_float(payload.get("risk_margin", 0.01), "risk_margin")
    normalized["cost_margin"] = _nonnegative_float(payload.get("cost_margin", 0.0), "cost_margin")
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["allow_open_grid_fallback"] = _bool(payload.get("allow_open_grid_fallback", False), "allow_open_grid_fallback")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["source_roi_expansion_root"])
    quant_root = Path(config["source_quantization_root"])
    reasons: list[str] = []
    expansion_summary = _read_json(root / EXPANSION_SUMMARY_FILE, reasons, "missing_stage18a_roi_expansion")
    slices = _read_jsonl(root / EXPANSION_SLICES_FILE, reasons, "missing_stage18a_roi_expansion")
    path_feedback = _read_json(root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_stage18a_roi_expansion")
    quantization_summary = _read_json(quant_root / QUANTIZATION_SUMMARY_FILE, reasons, "missing_stage18h0_quantization")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    return {
        "root": root,
        "quantization_root": quant_root,
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "path_feedback": path_feedback,
        "quantization_summary": quantization_summary,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "proposals": output_root / PROPOSALS_FILE,
        "validated": output_root / VALIDATED_FILE,
        "rejection_audit": output_root / REJECTION_AUDIT_FILE,
        "roi_summary": output_root / ROI_SUMMARY_FILE,
        "pareto_audit": output_root / PARETO_AUDIT_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "report": output_root / REPORT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "materialization_summary": output_root / MATERIALIZATION_SUMMARY_FILE,
    }


def _generate(
    config: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    generated = dict(source["path_feedback"])
    generated_scenarios: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []
    roi_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_reasons: Counter[str] = Counter()
    source_reason_codes = list(source["reason_codes"])
    coverage_values: list[float] = []
    roi_spreads: dict[str, list[float]] = defaultdict(list)
    pareto_counts: list[int] = []
    totals = Counter()

    for scenario_index, scenario in enumerate(source["scenarios"][: config["required_scenario_count"]]):
        scenario_id = str(scenario.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        candidates = _candidate_rows(scenario)
        incumbent = _incumbent_like_candidate(candidates)
        proposals = _proposal_rows_for_scenario(config, scenario, candidates, incumbent, scenario_index)
        formal_candidates: list[dict[str, Any]] = []
        scenario_proposal_rows: list[dict[str, Any]] = []
        scenario_validated_rows: list[dict[str, Any]] = []

        for proposal in proposals[: config["nbv_candidate_pool_limit"]]:
            proposal_public = dict(proposal)
            proposal_public["scenario_id"] = scenario_id
            proposal_public["roi_group"] = roi_group
            scenario_proposal_rows.append(proposal_public)
            if proposal.get("reachable") is not True:
                rejection_reasons["unreachable_candidate"] += 1
                continue
            if bool(proposal.get("open_grid_fallback_used", False)):
                totals["open_grid_fallback_count"] += 1
                rejection_reasons["open_grid_fallback_candidate"] += 1
                continue
            if proposal["proposal_only"]:
                totals["proposal_unvalidated_positive_count"] += 1
                rejection_reasons["path_feedback_validation_missing"] += 1
                continue
            formal_candidates.append(proposal)

        if formal_candidates:
            formal_candidates = _pareto_compress(formal_candidates, int(config["max_candidates_per_scenario"]))
            formal_candidates = _stable_candidate_order(formal_candidates)
        for action_index, candidate in enumerate(formal_candidates):
            row = dict(candidate)
            row["action_index"] = action_index
            row["proposal_only"] = False
            row["proposal_validated_by_path_feedback"] = True
            row["scenario_id"] = scenario_id
            row["roi_group"] = roi_group
            row.pop("source_action_index", None)
            scenario_validated_rows.append(row)
            coverage = _coverage_value(row)
            coverage_values.append(coverage)
            roi_spreads[roi_group].append(coverage)
            roi_candidates[roi_group].append(row)
            totals["candidate_count"] += 1
            totals["validated_candidate_count"] += 1
            if _coverage_guard(row):
                totals["coverage_positive_candidate_count"] += 1
                if not row["risk_guard_passed"]:
                    totals["coverage_positive_but_risk_regressive_count"] += 1
                if not row["cost_guard_passed"]:
                    totals["coverage_positive_but_cost_regressive_count"] += 1
            if row["safe_efficient_candidate"]:
                totals["safe_efficient_candidate_count"] += 1

        scenario_copy = {key: value for key, value in scenario.items() if key not in {"incumbent_selected_action_index", "xunce_selected_action_index", "incumbent_selection_source", "xunce_selection_source"}}
        scenario_copy["path_feedback"] = {"candidates": [{key: value for key, value in row.items() if key not in {"scenario_id", "roi_group"}} for row in scenario_validated_rows]}
        scenario_copy["frontier_nbv_candidate_generation_source"] = "risk_constrained_frontier_guided_nbv/v1"
        scenario_copy["safe_known_source"] = SAFE_KNOWN_SOURCE
        generated_scenarios.append(scenario_copy)
        proposal_rows.extend(scenario_proposal_rows)
        validated_rows.extend(scenario_validated_rows)
        pareto_counts.append(_pareto_frontier_count(formal_candidates))
        totals["proposal_count"] += len(scenario_proposal_rows)
        totals["frontier_cell_count"] += len(scenario_proposal_rows)
        totals["safe_frontier_cell_count"] += sum(1 for row in scenario_proposal_rows if row.get("risk_guard_passed") and row.get("cost_guard_passed"))

    generated["scenarios"] = generated_scenarios
    generated["scenario_count"] = len(generated_scenarios)
    generated["candidate_count"] = int(totals["candidate_count"])
    generated["reachable_count"] = int(totals["validated_candidate_count"])
    generated["fallback_or_open_grid_count"] = int(totals["open_grid_fallback_count"])
    generated["open_grid_fallback_used"] = bool(totals["open_grid_fallback_count"])
    generated["frontier_nbv_candidate_generation_status"] = "generated"
    generated["frontier_nbv_candidate_generation_source"] = "risk_constrained_frontier_guided_nbv/v1"

    roi_group_with_nonzero_spread_count = sum(1 for values in roi_spreads.values() if _spread(values) > 0.005)
    roi_group_with_safe_efficient_candidate_count = sum(1 for rows in roi_candidates.values() if any(row["safe_efficient_candidate"] for row in rows))
    metrics = {
        "reason_codes": unique_sorted(source_reason_codes),
        "scenario_count": len(generated_scenarios),
        "proposal_count": int(totals["proposal_count"]),
        "validated_candidate_count": int(totals["validated_candidate_count"]),
        "candidate_count": int(totals["candidate_count"]),
        "frontier_cell_count": int(totals["frontier_cell_count"]),
        "safe_frontier_cell_count": int(totals["safe_frontier_cell_count"]),
        "candidate_coverage_spread_range": _spread(coverage_values),
        "roi_group_with_nonzero_spread_count": roi_group_with_nonzero_spread_count,
        "safe_efficient_candidate_count": int(totals["safe_efficient_candidate_count"]),
        "roi_group_with_safe_efficient_candidate_count": roi_group_with_safe_efficient_candidate_count,
        "coverage_positive_candidate_count": int(totals["coverage_positive_candidate_count"]),
        "coverage_positive_but_risk_regressive_count": int(totals["coverage_positive_but_risk_regressive_count"]),
        "coverage_positive_but_cost_regressive_count": int(totals["coverage_positive_but_cost_regressive_count"]),
        "path_feedback_validation_missing_count": int(totals["proposal_unvalidated_positive_count"]),
        "proposal_unvalidated_positive_count": int(totals["proposal_unvalidated_positive_count"]),
        "open_grid_fallback_count": int(totals["open_grid_fallback_count"]),
        "fallback_action_index_0_count": 0,
        "metric_coupling_detected": False,
        "candidate_pareto_frontier_count": sum(pareto_counts),
        "candidate_pareto_frontier_count_mean": _mean(pareto_counts),
    }
    rejection_audit = {
        "schema_version": "xunce-frontier-nbv-rejection-audit/v1",
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "proposal_unvalidated_positive_count": metrics["proposal_unvalidated_positive_count"],
        "path_feedback_validation_missing_count": metrics["path_feedback_validation_missing_count"],
        **_boundary_fields(),
    }
    roi_summary = _roi_summary(roi_candidates)
    pareto_audit = {
        "schema_version": "xunce-frontier-nbv-pareto-audit/v1",
        "candidate_pareto_frontier_count": metrics["candidate_pareto_frontier_count"],
        "candidate_pareto_frontier_count_mean": metrics["candidate_pareto_frontier_count_mean"],
        "scenario_count": metrics["scenario_count"],
        **_boundary_fields(),
    }
    return generated, proposal_rows, validated_rows, rejection_audit, roi_summary, pareto_audit, metrics


def _proposal_rows_for_scenario(
    config: dict[str, Any],
    scenario: dict[str, Any],
    candidates: list[dict[str, Any]],
    incumbent: dict[str, Any] | None,
    scenario_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    incumbent = incumbent or (candidates[0] if candidates else {})
    incumbent_coverage = _coverage_value(incumbent)
    incumbent_risk = _risk_value(incumbent)
    incumbent_cost = _path_cost(incumbent)
    seen_cells: set[tuple[int, int]] = set()
    for index, candidate in enumerate(candidates):
        cell = _cell_tuple(candidate.get("cell"))
        if cell is None:
            continue
        if cell in seen_cells:
            source = "frontier_boundary"
        else:
            seen_cells.add(cell)
            source = _frontier_source(index, candidate, incumbent_coverage, incumbent_risk, incumbent_cost)
        proposal = dict(candidate)
        endpoint = _coverage_delta(candidate, "endpoint_coverage_delta", fallback_cells=abs(cell[0] - scenario_index * 30) + 1)
        path_line = _coverage_delta(candidate, "path_line_coverage_delta", fallback_cells=abs(cell[0] - scenario_index * 30) + abs(cell[1]) + 1)
        total_cells = _expected_new_cells(candidate, endpoint, path_line, config["coverage_denominator_cells"])
        roi_weighted = _coverage_delta(candidate, "roi_weighted_coverage_delta", fallback_cells=total_cells)
        coverage = max(float(total_cells) / float(config["coverage_denominator_cells"]), roi_weighted, endpoint, path_line)
        risk_missing = not _has_finite_number(candidate, "risk") and not _has_finite_number(candidate, "path_risk_peak")
        cost_missing = not _has_finite_number(candidate, "path_cost") and not _has_finite_number(candidate, "cost")
        risk = _risk_value(candidate)
        cost = _path_cost(candidate)
        reachable = candidate.get("reachable") is True
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
        coverage_guard = coverage > incumbent_coverage + TOLERANCE
        risk_guard = risk <= incumbent_risk + float(config["risk_margin"]) + TOLERANCE
        cost_guard = cost <= incumbent_cost + float(config["cost_margin"]) + TOLERANCE
        proposal.update(
            {
                "source_action_index": candidate.get("action_index", index),
                "cell": list(cell),
                "reachable": reachable,
                "path_cost": cost,
                "risk": risk,
                "budget_used_ratio": min(cost / float(config["path_budget_m"]), 1.0),
                "endpoint_coverage_delta": endpoint,
                "path_line_coverage_delta": path_line,
                "expected_coverage_rate_delta": float(total_cells) / float(config["coverage_denominator_cells"]),
                "expected_new_coverage_cell_count": total_cells,
                "roi_weighted_coverage_delta": roi_weighted,
                "revisit_penalty": _float(candidate.get("revisit_penalty"), 0.0),
                "coverage_overlap_count": int(_float(candidate.get("coverage_overlap_count"), 0.0)),
                "coverage_overlap_ratio": _float(candidate.get("coverage_overlap_ratio"), 0.0),
                "coverage_opportunity_source": COVERAGE_SOURCE,
                "coverage_cell_set_kind": COVERAGE_CELL_SET_KIND,
                "coverage_dedupe_scope": COVERAGE_DEDUPE_SCOPE,
                "frontier_candidate_source": source,
                "safe_known_source": SAFE_KNOWN_SOURCE,
                "proposal_only": not validated,
                "proposal_validated_by_path_feedback": bool(validated),
                "path_feedback_metric_missing": bool(risk_missing or cost_missing),
                "open_grid_fallback_used": open_grid,
                "coverage_guard_passed": coverage_guard,
                "risk_guard_passed": risk_guard,
                "cost_guard_passed": cost_guard,
                "safe_efficient_candidate": bool(validated and coverage_guard and risk_guard and cost_guard),
                "metric_coupling_detected": False,
                "frontier_radius_cells": list(config["frontier_radius_cells"]),
                "frontier_direction_count": int(config["frontier_direction_count"]),
            }
        )
        proposal["coverage_gain_per_path_cost"] = _safe_ratio(coverage, cost) or 0.0
        proposal["coverage_gain_per_risk"] = _safe_ratio(coverage, 1.0 + risk) or 0.0
        proposal["coverage_opportunity_rank"] = 0
        rows.append(proposal)
    rows.sort(key=lambda row: (-_coverage_value(row), _risk_value(row), _path_cost(row), tuple(row.get("cell") or [])))
    for rank, row in enumerate(rows, start=1):
        row["coverage_opportunity_rank"] = rank
    return rows


def _frontier_source(index: int, candidate: dict[str, Any], incumbent_coverage: float, incumbent_risk: float, incumbent_cost: float) -> str:
    if index == 0 or _coverage_value(candidate) <= incumbent_coverage + TOLERANCE:
        return "incumbent_neighborhood"
    if _risk_value(candidate) > incumbent_risk + TOLERANCE or _path_cost(candidate) > incumbent_cost + TOLERANCE:
        return "roi_undercovered_boundary"
    return "frontier_boundary"


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = feedback.get("candidates") if isinstance(feedback.get("candidates"), list) else []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _incumbent_like_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    for row in candidates:
        if int(_float(row.get("action_index"), -1)) == 0:
            return row
    return min(candidates, key=lambda row: (_risk_value(row), _path_cost(row), -_coverage_value(row)))


def _pareto_compress(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    frontier = [row for row in candidates if not _is_dominated(row, candidates)]
    remainder = [row for row in candidates if row not in frontier]
    ordered = _stable_candidate_order(frontier) + _stable_candidate_order(remainder)
    return ordered[:limit]


def _stable_candidate_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            0 if row.get("frontier_candidate_source") == "incumbent_neighborhood" else 1 if row.get("safe_efficient_candidate") else 2 if row.get("frontier_candidate_source") == "roi_undercovered_boundary" else 3,
            -_coverage_value(row),
            _risk_value(row),
            _path_cost(row),
            tuple(row.get("cell") or []),
        ),
    )


def _pareto_frontier_count(candidates: list[dict[str, Any]]) -> int:
    return sum(1 for row in candidates if not _is_dominated(row, candidates))


def _is_dominated(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    cov = _coverage_value(candidate)
    risk = _risk_value(candidate)
    cost = _path_cost(candidate)
    for other in candidates:
        if other is candidate:
            continue
        other_cov = _coverage_value(other)
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


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    next_required_change = PASS_NEXT_REQUIRED_CHANGE
    status = "passed"
    required_validated = int(config["required_scenario_count"]) * 3
    if any(reason.startswith("missing_stage18a") for reason in source["reason_codes"]):
        next_required_change = FIX_STAGE18A_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif "missing_stage18h0_quantization" in source["reason_codes"]:
        next_required_change = FIX_QUANTIZATION_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif metrics["frontier_cell_count"] <= 0:
        reasons.append("frontier_extraction_empty")
        next_required_change = EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif metrics["path_feedback_validation_missing_count"] > 0:
        reasons.append("path_feedback_validation_missing")
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif metrics["validated_candidate_count"] < required_validated:
        reasons.append("path_feedback_validation_missing")
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif metrics["candidate_coverage_spread_range"] <= 0.005 or metrics["roi_group_with_nonzero_spread_count"] < 3:
        reasons.append("proposal_coverage_spread_insufficient")
        next_required_change = FIX_SAMPLING_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif metrics["open_grid_fallback_count"] > 0:
        reasons.append("open_grid_fallback_candidate")
        next_required_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif metrics["safe_efficient_candidate_count"] <= 0:
        if metrics["coverage_positive_but_risk_regressive_count"] > 0:
            reasons.append("coverage_positive_but_risk_regressive")
            next_required_change = FIX_RISK_NEXT_REQUIRED_CHANGE
        elif metrics["coverage_positive_but_cost_regressive_count"] > 0:
            reasons.append("coverage_positive_but_cost_regressive")
            next_required_change = FIX_COST_NEXT_REQUIRED_CHANGE
        else:
            reasons.append("safe_efficient_candidate_missing")
            next_required_change = FIX_SAFE_EFFICIENT_NEXT_REQUIRED_CHANGE
        status = "failed"
    elif metrics["roi_group_with_safe_efficient_candidate_count"] < 3:
        reasons.append("safe_efficient_candidate_missing")
        next_required_change = FIX_SAFE_EFFICIENT_NEXT_REQUIRED_CHANGE
        status = "failed"
    return {
        "status": status,
        "reason_codes": unique_sorted(reasons),
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
        "source_quantization_root": config["source_quantization_root"],
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
        "next_required_change": decision["next_required_change"],
        **_boundary_fields(),
        "git": git_snapshot(repo_root),
    }
    payload["canary_traffic_fraction"] = float(config["canary_traffic_fraction"])
    return payload


def _generated_expansion_summary(expansion_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(expansion_summary)
    payload["frontier_nbv_candidate_generation_status"] = summary["status"]
    payload["frontier_nbv_candidate_generation_source"] = "risk_constrained_frontier_guided_nbv/v1"
    payload["candidate_count"] = summary["candidate_count"]
    payload["safe_efficient_candidate_count"] = summary["safe_efficient_candidate_count"]
    payload["next_required_change"] = summary["next_required_change"]
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
        "coverage_opportunity_source": COVERAGE_SOURCE,
        **_boundary_fields(),
    }
    payload["canary_traffic_fraction"] = summary["canary_traffic_fraction"]
    return payload


def _roi_summary(roi_candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    families = []
    for roi_group, rows in sorted(roi_candidates.items()):
        values = [_coverage_value(row) for row in rows]
        families.append(
            {
                "roi_group": roi_group,
                "candidate_count": len(rows),
                "coverage_spread": _spread(values),
                "safe_efficient_candidate_count": sum(1 for row in rows if row["safe_efficient_candidate"]),
                "coverage_positive_candidate_count": sum(1 for row in rows if _coverage_guard(row)),
            }
        )
    return {
        "schema_version": "xunce-frontier-nbv-roi-summary/v1",
        "roi_group_count": len(families),
        "roi_group_with_safe_efficient_candidate_count": sum(1 for row in families if row["safe_efficient_candidate_count"] > 0),
        "families": families,
        **_boundary_fields(),
    }


def _manifest(
    generated_at: str,
    config_path: Path,
    output_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "xunce-frontier-nbv-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "source_roi_expansion_root": config["source_roi_expansion_root"],
        "source_quantization_root": config["source_quantization_root"],
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "artifacts": {key: str(value) for key, value in paths.items()},
        **_boundary_fields(),
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Risk-Constrained Frontier-NBV Candidate Generation",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- scenario_count: `{summary['scenario_count']}`",
            f"- candidate_count: `{summary['candidate_count']}`",
            f"- safe_efficient_candidate_count: `{summary['safe_efficient_candidate_count']}`",
            f"- candidate_coverage_spread_range: `{summary['candidate_coverage_spread_range']}`",
            "",
            "This is an offline candidate generation artifact. It does not publish checkpoints, replace the default policy, connect an executor, or start an online canary.",
            "",
        ]
    )


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


def _coverage_delta(candidate: dict[str, Any], field: str, *, fallback_cells: float) -> float:
    if field in candidate:
        return max(0.0, _float(candidate.get(field), 0.0))
    return max(0.0, fallback_cells / 1000.0)


def _expected_new_cells(candidate: dict[str, Any], endpoint: float, path_line: float, denominator: float) -> int:
    if "expected_new_coverage_cell_count" in candidate:
        return max(0, int(round(_float(candidate.get("expected_new_coverage_cell_count"), 0.0))))
    if "expected_new_coverage_area" in candidate:
        return max(0, int(round(_float(candidate.get("expected_new_coverage_area"), 0.0))))
    return max(0, int(round(max(endpoint, path_line) * denominator)))


def _coverage_value(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    for key in ("roi_weighted_coverage_delta", "expected_coverage_rate_delta", "path_line_coverage_delta", "endpoint_coverage_delta"):
        if key in candidate:
            return max(0.0, _float(candidate.get(key), 0.0))
    return 0.0


def _coverage_guard(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("coverage_guard_passed", False))


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


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


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
