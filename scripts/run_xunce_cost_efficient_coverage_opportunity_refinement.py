from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
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


CONFIG_SCHEMA_VERSION = "xunce-cost-efficient-coverage-opportunity-refinement-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-cost-efficient-coverage-opportunity-summary/v1"
DEFAULT_CONFIG = "configs/xunce_cost_efficient_coverage_opportunity_refinement_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_cost_efficient_coverage_opportunity_refinement_v1"

MATERIALIZATION_SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"
ORACLE_SUMMARY_FILE = "xunce-oracle-separability-summary.json"
EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

SUMMARY_FILE = "xunce-cost-efficient-coverage-opportunity-summary.json"
OVERLAY_FILE = "xunce-cost-efficient-candidate-overlay.jsonl"
SPREAD_FILE = "xunce-cost-efficient-spread-by-scenario.jsonl"
ROI_SUMMARY_FILE = "xunce-cost-efficient-roi-summary.json"
DECISION_AUDIT_FILE = "xunce-cost-efficient-decision-audit.json"
MANIFEST_FILE = "xunce-cost-efficient-coverage-opportunity-manifest.json"
REPORT_FILE = "xunce-cost-efficient-coverage-opportunity-report.md"

PASS_NEXT_REQUIRED_CHANGE = "run_oracle_separability_benchmark"
FIX_MATERIALIZATION_NEXT_REQUIRED_CHANGE = "run_candidate_level_coverage_opportunity_materialization"
FIX_INPUT_NEXT_REQUIRED_CHANGE = "repair_candidate_materialization_inputs"
REFINE_NEXT_REQUIRED_CHANGE = "refine_cost_efficient_coverage_opportunity"
EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE = "expand_roi_or_map_complexity"

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
    parser = argparse.ArgumentParser(description="Refine Xunce coverage opportunities for cost/risk efficiency.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-materialized-coverage-root")
    parser.add_argument("--source-oracle-separability-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_materialized_coverage_root": args.source_materialized_coverage_root,
            "source_oracle_separability_root": args.source_oracle_separability_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_cost_efficient_coverage_opportunity_refinement(
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
                "cost_efficient_refinement_passed": summary["cost_efficient_refinement_passed"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_cost_efficient_coverage_opportunity_refinement(
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
    refined, overlay_rows, spread_rows, roi_summary, metrics = _refine(config, source)
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
    manifest = {
        "schema_version": "xunce-cost-efficient-coverage-opportunity-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "source_materialized_coverage_root": config["source_materialized_coverage_root"],
        "source_oracle_separability_root": config["source_oracle_separability_root"],
        "artifacts": {key: str(value) for key, value in paths.items()},
        **_boundary_fields(),
    }

    write_json(paths["summary"], summary)
    write_jsonl(paths["overlay"], overlay_rows)
    write_jsonl(paths["spread"], spread_rows)
    write_json(paths["roi_summary"], roi_summary)
    write_json(paths["decision_audit"], _decision_audit(source, metrics, decision))
    write_json(paths["manifest"], manifest)
    write_json(paths["expansion_summary"], _refined_expansion_summary(source["expansion_summary"], summary))
    write_jsonl(paths["slices"], source["slices"])
    write_json(paths["refined_path_feedback"], refined)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
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
    for key in ("source_materialized_coverage_root", "source_oracle_separability_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["min_cost_efficient_spread"] = _nonnegative_float(payload.get("min_cost_efficient_spread", 0.005), "min_cost_efficient_spread")
    normalized["min_safe_efficient_opportunity_count"] = _positive_int(payload.get("min_safe_efficient_opportunity_count", 1), "min_safe_efficient_opportunity_count")
    normalized["min_roi_group_with_safe_efficient_opportunity"] = _positive_int(payload.get("min_roi_group_with_safe_efficient_opportunity", 3), "min_roi_group_with_safe_efficient_opportunity")
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["epsilon"] = _positive_float(payload.get("epsilon", TOLERANCE), "epsilon")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    materialized_root = Path(config["source_materialized_coverage_root"])
    oracle_root = Path(config["source_oracle_separability_root"])
    reasons: list[str] = []
    materialization_summary = _read_json(materialized_root / MATERIALIZATION_SUMMARY_FILE, reasons, "missing_stage18e_materialization")
    expansion_summary = _read_json(materialized_root / EXPANSION_SUMMARY_FILE, reasons, "missing_stage18e_materialization")
    slices = _read_jsonl(materialized_root / EXPANSION_SLICES_FILE, reasons, "missing_stage18e_materialization_slices")
    path_feedback = _read_json(materialized_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_stage18e_materialization")
    oracle_summary = _read_json(oracle_root / ORACLE_SUMMARY_FILE, [], "missing_stage18f_oracle_separability")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    if materialization_summary.get("status") not in (None, "passed"):
        reasons.append("missing_stage18e_materialization")
    return {
        "materialized_root": materialized_root,
        "oracle_root": oracle_root,
        "materialization_summary": materialization_summary,
        "oracle_summary": oracle_summary,
        "oracle_summary_loaded": bool(oracle_summary),
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "path_feedback": path_feedback,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "overlay": output_root / OVERLAY_FILE,
        "spread": output_root / SPREAD_FILE,
        "roi_summary": output_root / ROI_SUMMARY_FILE,
        "decision_audit": output_root / DECISION_AUDIT_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "report": output_root / REPORT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "refined_path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
    }


def _refine(config: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    refined = dict(source["path_feedback"])
    refined_scenarios: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    roi_safe_counts: dict[str, int] = defaultdict(int)
    roi_spreads: dict[str, list[float]] = defaultdict(list)
    reason_codes = list(source["reason_codes"])
    safe_count = 0
    safe_scenario_count = 0
    inefficient_high_coverage_count = 0
    dominated_count = 0
    frontier_counts: list[int] = []
    spread_values: list[float] = []
    candidate_count = 0
    missing_input_count = 0
    incumbent_sources: list[str] = []

    for scenario_index, scenario in enumerate(source["scenarios"][: config["required_scenario_count"]]):
        scenario_copy = dict(scenario)
        scenario_id = str(scenario_copy.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario_copy.get("roi_group") or scenario_copy.get("scenario_group") or "unknown")
        candidates = _candidate_rows(scenario_copy)
        incumbent_index, incumbent_source = _incumbent_index(scenario_copy)
        incumbent_sources.append(incumbent_source)
        incumbent = _candidate_at(candidates, incumbent_index)
        incumbent_metrics = _candidate_metrics(incumbent, config)
        dominance = _dominance_statuses(candidates)
        enriched_candidates: list[dict[str, Any]] = []
        scenario_safe_count = 0
        scenario_scores: list[float] = []

        for action_index, candidate in enumerate(candidates):
            candidate_count += 1
            enriched = dict(candidate)
            if not _has_required_inputs(enriched):
                missing_input_count += 1
            metrics = _candidate_metrics(enriched, config)
            valid = _candidate_valid(enriched)
            coverage_beats_incumbent = metrics["coverage"] > incumbent_metrics["coverage"] + TOLERANCE
            cost_guard = valid and metrics["path_cost"] <= incumbent_metrics["path_cost"] + TOLERANCE
            risk_guard = valid and metrics["risk"] <= incumbent_metrics["risk"] + TOLERANCE
            efficiency_guard = valid and metrics["cost_efficiency"] > incumbent_metrics["cost_efficiency"] + TOLERANCE
            safe = bool(valid and coverage_beats_incumbent and cost_guard and risk_guard and efficiency_guard)
            high_coverage_inefficient = bool(coverage_beats_incumbent and valid and not safe)
            if safe:
                safe_count += 1
                scenario_safe_count += 1
                roi_safe_counts[roi_group] += 1
            if high_coverage_inefficient:
                inefficient_high_coverage_count += 1
            if dominance.get(action_index) == "dominated":
                dominated_count += 1
            if valid:
                scenario_scores.append(metrics["cost_efficiency"])
            enriched.update(
                {
                    "cost_efficiency_adjusted_coverage_delta": metrics["cost_efficiency"],
                    "risk_adjusted_coverage_delta": metrics["risk_adjusted"],
                    "budget_adjusted_coverage_delta": metrics["budget_adjusted"],
                    "coverage_cost_pareto_rank": 1 if dominance.get(action_index) == "pareto_frontier" else 2,
                    "dominance_status": dominance.get(action_index, "invalid"),
                    "safe_efficient_opportunity": safe,
                    "cost_efficiency_guard_passed": bool(cost_guard and efficiency_guard),
                    "risk_efficiency_guard_passed": bool(risk_guard),
                    "incumbent_roi_weighted_coverage_delta": incumbent_metrics["coverage"],
                    "incumbent_path_cost": incumbent_metrics["path_cost"],
                    "incumbent_risk": incumbent_metrics["risk"],
                    "incumbent_cost_efficiency_adjusted_coverage_delta": incumbent_metrics["cost_efficiency"],
                    "incumbent_selection_source": incumbent_source,
                    "efficiency_refinement_source": "cost_efficient_counterfactual_from_stage18e_candidate/v1",
                }
            )
            enriched_candidates.append(enriched)
            overlay_rows.append(
                {
                    "schema_version": "xunce-cost-efficient-candidate-overlay-row/v1",
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "action_index": enriched.get("action_index", action_index),
                    "reachable": enriched.get("reachable"),
                    "roi_weighted_coverage_delta": metrics["coverage"],
                    "path_cost": metrics["path_cost"],
                    "risk": metrics["risk"],
                    "cost_efficiency_adjusted_coverage_delta": metrics["cost_efficiency"],
                    "risk_adjusted_coverage_delta": metrics["risk_adjusted"],
                    "budget_adjusted_coverage_delta": metrics["budget_adjusted"],
                    "dominance_status": enriched["dominance_status"],
                    "safe_efficient_opportunity": safe,
                    "cost_efficiency_guard_passed": enriched["cost_efficiency_guard_passed"],
                    "risk_efficiency_guard_passed": enriched["risk_efficiency_guard_passed"],
                    "incumbent_selection_source": incumbent_source,
                }
            )

        frontier_count = sum(1 for status in dominance.values() if status == "pareto_frontier")
        frontier_counts.append(frontier_count)
        spread = _spread_range(scenario_scores)
        spread_values.append(spread)
        roi_spreads[roi_group].append(spread)
        if scenario_safe_count > 0:
            safe_scenario_count += 1
        spread_rows.append(
            {
                "schema_version": "xunce-cost-efficient-spread-by-scenario/v1",
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "candidate_count": len(candidates),
                "safe_efficient_opportunity_count": scenario_safe_count,
                "cost_efficient_coverage_spread_range": spread,
                "candidate_dominated_count": sum(1 for status in dominance.values() if status == "dominated"),
                "candidate_pareto_frontier_count": frontier_count,
                "incumbent_selected_action_index": incumbent_index,
                "incumbent_selection_source": incumbent_source,
            }
        )
        scenario_copy["incumbent_selected_action_index"] = incumbent_index
        scenario_copy["incumbent_selection_source"] = incumbent_source
        scenario_copy["path_feedback"] = dict(scenario_copy.get("path_feedback") or {})
        scenario_copy["path_feedback"]["candidates"] = enriched_candidates
        refined_scenarios.append(scenario_copy)

    if missing_input_count:
        reason_codes.append("candidate_efficiency_inputs_insufficient")
    roi_rows = [
        {
            "roi_group": roi_group,
            "scenario_count": len(roi_spreads[roi_group]),
            "safe_efficient_opportunity_count": roi_safe_counts[roi_group],
            "max_cost_efficient_coverage_spread": max(roi_spreads[roi_group]) if roi_spreads[roi_group] else 0.0,
            "mean_cost_efficient_coverage_spread": _mean(roi_spreads[roi_group]),
            "has_safe_efficient_opportunity": roi_safe_counts[roi_group] > 0,
        }
        for roi_group in sorted(set(roi_spreads) | set(roi_safe_counts))
    ]
    roi_summary = {
        "schema_version": "xunce-cost-efficient-roi-summary/v1",
        "roi_group_count": len(roi_rows),
        "roi_group_with_safe_efficient_opportunity_count": sum(1 for row in roi_rows if row["has_safe_efficient_opportunity"]),
        "families": roi_rows,
    }
    metrics = {
        "reason_codes": unique_sorted(reason_codes),
        "candidate_count": candidate_count,
        "scenario_count": len(refined_scenarios),
        "safe_efficient_opportunity_count": safe_count,
        "safe_efficient_opportunity_scenario_count": safe_scenario_count,
        "roi_group_with_safe_efficient_opportunity_count": roi_summary["roi_group_with_safe_efficient_opportunity_count"],
        "cost_efficient_coverage_spread_range": _mean(spread_values),
        "inefficient_high_coverage_candidate_count": inefficient_high_coverage_count,
        "candidate_dominated_count": dominated_count,
        "candidate_pareto_frontier_count": _mean(frontier_counts),
        "candidate_efficiency_inputs_missing_count": missing_input_count,
        "incumbent_selection_sources": unique_sorted(incumbent_sources),
    }
    refined["scenarios"] = refined_scenarios
    refined["scenario_count"] = len(refined_scenarios)
    refined["candidate_count"] = candidate_count
    refined["efficiency_refinement_source"] = "cost_efficient_counterfactual_from_stage18e_candidate/v1"
    return refined, overlay_rows, spread_rows, roi_summary, metrics


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    if any(reason.startswith("missing_stage18e") for reason in reasons):
        return {"status": "failed", "reason_codes": unique_sorted(reasons), "next_required_change": FIX_MATERIALIZATION_NEXT_REQUIRED_CHANGE}
    if metrics["candidate_efficiency_inputs_missing_count"] > 0:
        reasons.append("candidate_efficiency_inputs_insufficient")
        return {"status": "failed", "reason_codes": unique_sorted(reasons), "next_required_change": FIX_INPUT_NEXT_REQUIRED_CHANGE}
    if metrics["cost_efficient_coverage_spread_range"] <= float(config["min_cost_efficient_spread"]):
        reasons.append("cost_efficient_coverage_spread_insufficient")
    if metrics["safe_efficient_opportunity_count"] < int(config["min_safe_efficient_opportunity_count"]):
        reasons.append("safe_efficient_opportunity_insufficient")
    if metrics["roi_group_with_safe_efficient_opportunity_count"] < int(config["min_roi_group_with_safe_efficient_opportunity"]):
        reasons.append("safe_efficient_roi_spread_insufficient")
    reasons = unique_sorted(reasons)
    if not reasons:
        return {"status": "passed", "reason_codes": [], "next_required_change": PASS_NEXT_REQUIRED_CHANGE}
    if metrics["safe_efficient_opportunity_count"] == 0:
        next_change = EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE
    else:
        next_change = REFINE_NEXT_REQUIRED_CHANGE
    return {"status": "failed", "reason_codes": reasons, "next_required_change": next_change}


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
        "next_required_change": decision["next_required_change"],
        "cost_efficient_refinement_passed": decision["status"] == "passed",
        **{key: value for key, value in metrics.items() if key != "reason_codes"},
        "source_materialized_coverage_root": config["source_materialized_coverage_root"],
        "source_oracle_separability_root": config["source_oracle_separability_root"],
        "source_oracle_summary_loaded": source["oracle_summary_loaded"],
        "source_oracle_separable": source["oracle_summary"].get("oracle_separable"),
        "source_materialization_status": source["materialization_summary"].get("status"),
        "path_budget_m": config["path_budget_m"],
        "epsilon": config["epsilon"],
        "canary_traffic_fraction": config["canary_traffic_fraction"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "refined_path_feedback": str(paths["refined_path_feedback"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _decision_audit(source: dict[str, Any], metrics: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-cost-efficient-decision-audit/v1",
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "source_materialization_status": source["materialization_summary"].get("status"),
        "source_oracle_summary_loaded": source["oracle_summary_loaded"],
        "safe_efficient_opportunity_count": metrics["safe_efficient_opportunity_count"],
        "roi_group_with_safe_efficient_opportunity_count": metrics["roi_group_with_safe_efficient_opportunity_count"],
        "cost_efficient_coverage_spread_range": metrics["cost_efficient_coverage_spread_range"],
        **_boundary_fields(),
    }


def _refined_expansion_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["cost_efficient_refinement_passed"] = summary["cost_efficient_refinement_passed"]
    payload["safe_efficient_opportunity_count"] = summary["safe_efficient_opportunity_count"]
    payload["roi_group_with_safe_efficient_opportunity_count"] = summary["roi_group_with_safe_efficient_opportunity_count"]
    payload["cost_efficient_coverage_spread_range"] = summary["cost_efficient_coverage_spread_range"]
    payload.update(_boundary_fields())
    return payload


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []


def _incumbent_index(scenario: dict[str, Any]) -> tuple[int, str]:
    if isinstance(scenario.get("incumbent_selected_action_index"), int):
        return int(scenario["incumbent_selected_action_index"]), "scenario_incumbent_selected_action_index"
    feedback = scenario.get("path_feedback")
    if isinstance(feedback, dict) and isinstance(feedback.get("incumbent_selected_action_index"), int):
        return int(feedback["incumbent_selected_action_index"]), "path_feedback_incumbent_selected_action_index"
    return 0, "fallback_action_index_0"


def _candidate_metrics(candidate: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, float]:
    coverage = _coverage_value(candidate)
    path_cost = _path_cost(candidate)
    risk = _risk(candidate)
    epsilon = float(config["epsilon"])
    path_budget_m = float(config["path_budget_m"])
    return {
        "coverage": coverage,
        "path_cost": path_cost,
        "risk": risk,
        "cost_efficiency": coverage / max(path_cost * (1.0 + risk), epsilon),
        "risk_adjusted": coverage / max(1.0 + risk, epsilon),
        "budget_adjusted": coverage * max(0.0, 1.0 - path_cost / path_budget_m),
    }


def _dominance_statuses(candidates: list[dict[str, Any]]) -> dict[int, str]:
    statuses: dict[int, str] = {}
    for index, candidate in enumerate(candidates):
        if not _candidate_valid(candidate):
            statuses[index] = "invalid"
            continue
        coverage = _coverage_value(candidate)
        cost = _path_cost(candidate)
        risk = _risk(candidate)
        dominated = False
        for other_index, other in enumerate(candidates):
            if other_index == index or not _candidate_valid(other):
                continue
            other_coverage = _coverage_value(other)
            other_cost = _path_cost(other)
            other_risk = _risk(other)
            weakly_better = other_coverage >= coverage - TOLERANCE and other_cost <= cost + TOLERANCE and other_risk <= risk + TOLERANCE
            strictly_better = other_coverage > coverage + TOLERANCE or other_cost < cost - TOLERANCE or other_risk < risk - TOLERANCE
            if weakly_better and strictly_better:
                dominated = True
                break
        statuses[index] = "dominated" if dominated else "pareto_frontier"
    return statuses


def _has_required_inputs(candidate: dict[str, Any]) -> bool:
    return all(key in candidate for key in ("reachable", "path_cost", "risk")) and any(
        key in candidate for key in ("roi_weighted_coverage_delta", "expected_coverage_rate_delta", "path_line_coverage_delta")
    )


def _candidate_valid(candidate: dict[str, Any] | None) -> bool:
    return bool(
        candidate
        and candidate.get("reachable") is True
        and candidate.get("open_grid_fallback_used") is not True
        and math.isfinite(_path_cost(candidate))
        and math.isfinite(_risk(candidate))
    )


def _candidate_at(candidates: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _coverage_value(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    for field in ("roi_weighted_coverage_delta", "expected_coverage_rate_delta", "path_line_coverage_delta"):
        if field in candidate:
            return _float(candidate.get(field))
    return 0.0


def _path_cost(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("path_cost")) if candidate else 0.0


def _risk(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("risk")) if candidate else 0.0


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


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Cost-Efficient Coverage Opportunity Refinement",
            "",
            f"- status: `{summary['status']}`",
            f"- cost_efficient_refinement_passed: `{summary['cost_efficient_refinement_passed']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- safe_efficient_opportunity_count: `{summary['safe_efficient_opportunity_count']}`",
            f"- roi_group_with_safe_efficient_opportunity_count: `{summary['roi_group_with_safe_efficient_opportunity_count']}`",
            f"- cost_efficient_coverage_spread_range: `{summary['cost_efficient_coverage_spread_range']}`",
        ]
    )


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


def _spread_range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
