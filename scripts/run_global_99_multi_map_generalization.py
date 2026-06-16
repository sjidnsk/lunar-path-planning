from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
for import_path in (SCRIPT_DIR, MODEL_EXPLORER_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import merge_global_99_boundary_defaults
    from run_policy_guided_global_coverage import (
        _load_config as load_policy_guided_config,
        _load_frontier_config,
        _load_policy_bundle,
        _resolve_config_reference,
        _run_policy_guided_scenario,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import (
        ConfigError,
        resolve_path,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )
    from scripts.global_99_governance_common import merge_global_99_boundary_defaults
    from scripts.run_policy_guided_global_coverage import (
        _load_config as load_policy_guided_config,
        _load_frontier_config,
        _load_policy_bundle,
        _resolve_config_reference,
        _run_policy_guided_scenario,
    )


CONFIG_SCHEMA_VERSION = "global-99-multi-map-generalization-config/v1"
SUMMARY_SCHEMA_VERSION = "global-99-multi-map-generalization-summary/v1"
MANIFEST_SCHEMA_VERSION = "global-99-multi-map-generalization-manifest/v1"
SCENARIO_RESULT_SCHEMA_VERSION = "global-99-multi-map-scenario-result/v1"
FAMILY_SUMMARY_SCHEMA_VERSION = "global-99-multi-map-family-summary/v1"
POLICY_AUDIT_SCHEMA_VERSION = "global-99-multi-map-policy-vs-baseline-audit/v1"
BUDGET_AUDIT_SCHEMA_VERSION = "global-99-multi-map-budget-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "global-99-multi-map-rejection-report/v1"

DEFAULT_CONFIG = "configs/global_99_multi_map_generalization_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_global_99_multi_map_generalization_v1"

SUMMARY_FILE = "global-99-multi-map-generalization-summary.json"
MANIFEST_FILE = "global-99-multi-map-generalization-manifest.json"
SCENARIO_RESULTS_FILE = "global-99-multi-map-scenario-results.jsonl"
FAMILY_SUMMARY_FILE = "global-99-multi-map-family-summary.json"
POLICY_AUDIT_FILE = "global-99-multi-map-policy-vs-baseline-audit.json"
BUDGET_AUDIT_FILE = "global-99-multi-map-budget-audit.json"
REJECTION_REPORT_FILE = "global-99-multi-map-rejection-report.json"
REPORT_FILE = "global-99-multi-map-generalization-report.md"

PASS_NEXT_REQUIRED_CHANGE = "network_architecture_upgrade_readiness_review"
FAIL_NEXT_REQUIRED_CHANGE = "fix_global_99_multi_map_generalization"

DEFAULT_FAMILIES = (
    "open_field",
    "corridor",
    "rooms",
    "blocked_roi",
    "unsafe_patch",
    "narrow_passage",
    "cells_roi",
    "budget_limited",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Global 99 Multi-Map Generalization v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_global_99_multi_map_generalization(
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
                "scenario_count": summary["scenario_count"],
                "aggregate_achieved_coverage_rate": summary["aggregate_achieved_coverage_rate"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_global_99_multi_map_generalization(
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
    policy_config_path = _resolve_config_reference(config["source_policy_guided_config"], config_path, repo_root)
    policy_config = load_policy_guided_config(policy_config_path)
    frontier_config_path = _resolve_config_reference(
        policy_config["source_frontier_baseline_config"],
        policy_config_path,
        repo_root,
    )
    frontier_config = _load_frontier_config(frontier_config_path)
    policy_bundle, policy_load = _load_policy_bundle(policy_config, policy_config_path, repo_root)

    scenarios = _build_scenario_matrix(config)
    scenario_results: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_budget = float(scenario.get("path_budget_m", config["path_budget_m"]))
        result = _run_policy_guided_scenario(
            scenario,
            target_coverage_rate=float(config["target_coverage_rate"]),
            path_budget_m=scenario_budget,
            coverage_radius_cells=int(config["coverage_radius_cells"]),
            replanning_cycle_limit=int(policy_config["replanning_cycle_limit"]),
            segment_step_limit=int(policy_config["segment_step_limit"]),
            memory_snapshot_interval=int(policy_config["memory_snapshot_interval"]),
            max_policy_candidates=int(policy_config["max_policy_candidates"]),
            policy_logit_weight=float(policy_config["policy_logit_weight"]),
            revisit_penalty_weight=float(frontier_config["revisit_penalty_weight"]),
            new_coverage_weight=float(frontier_config["new_coverage_weight"]),
            policy_bundle=policy_bundle,
        )
        result["family_id"] = scenario["family_id"]
        result["generalization_required"] = bool(scenario.get("generalization_required", True))
        result["expected_infeasible"] = bool(scenario.get("expected_infeasible", False))
        result["scenario_path_budget_m"] = scenario_budget
        scenario_results.append(result)

    family_summary = _family_summary(scenario_results)
    policy_audit = _policy_vs_baseline_audit(policy_load, scenario_results)
    budget_audit = _budget_audit(config, scenario_results)
    reason_codes = _summary_reason_codes(config, policy_load, scenario_results, policy_audit)
    status = "passed" if not reason_codes else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE if status == "passed" else FAIL_NEXT_REQUIRED_CHANGE

    required_results = [result for result in scenario_results if result["generalization_required"]]
    required_denominator = sum(result["reachable_safe_cell_count"] for result in required_results)
    required_covered = sum(result["covered_reachable_safe_cell_count"] for result in required_results)
    aggregate_achieved = (required_covered / required_denominator) if required_denominator else 0.0
    min_required_coverage = min(
        (result["achieved_coverage_rate"] for result in required_results),
        default=0.0,
    )
    coverage_target_met_all_required = all(result["coverage_target_met"] for result in required_results)
    passed_required = sum(1 for result in required_results if result["status"] == "passed")
    failed_required = len(required_results) - passed_required
    infeasible_reason_codes = unique_sorted(
        reason for result in scenario_results for reason in result["infeasible_reason_codes"]
    )

    paths = {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "scenario_results": output_root / SCENARIO_RESULTS_FILE,
        "family_summary": output_root / FAMILY_SUMMARY_FILE,
        "policy_vs_baseline_audit": output_root / POLICY_AUDIT_FILE,
        "budget_audit": output_root / BUDGET_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }
    summary = merge_global_99_boundary_defaults({
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "config": str(config_path),
        "source_policy_guided_config": str(policy_config_path),
        "source_frontier_baseline_config": str(frontier_config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "scenario_results": str(paths["scenario_results"]),
        "family_summary": str(paths["family_summary"]),
        "policy_vs_baseline_audit": str(paths["policy_vs_baseline_audit"]),
        "budget_audit": str(paths["budget_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "target_coverage_rate": config["target_coverage_rate"],
        "scenario_count": len(scenario_results),
        "passed_scenario_count": sum(1 for result in scenario_results if result["status"] == "passed"),
        "failed_scenario_count": sum(1 for result in scenario_results if result["status"] == "failed"),
        "required_scenario_count": len(required_results),
        "passed_required_scenario_count": passed_required,
        "failed_required_scenario_count": failed_required,
        "family_count": len(family_summary["families"]),
        "passed_family_count": family_summary["passed_family_count"],
        "failed_family_count": family_summary["failed_family_count"],
        "aggregate_achieved_coverage_rate": aggregate_achieved,
        "min_scenario_achieved_coverage_rate": min_required_coverage,
        "coverage_target_met_all_required_scenarios": coverage_target_met_all_required,
        "policy_loaded": policy_load["policy_loaded"],
        "policy_guidance_applied": policy_audit["policy_guidance_applied"],
        "policy_scored_candidate_count": policy_audit["policy_scored_candidate_count"],
        "policy_guided_decision_count": policy_audit["policy_guided_decision_count"],
        "policy_guard_fallback_count": policy_audit["policy_guard_fallback_count"],
        "baseline_agreement_rate": policy_audit["baseline_agreement_rate"],
        "policy_better_than_baseline_count": policy_audit["policy_better_than_baseline_count"],
        "policy_worse_than_baseline_count": policy_audit["policy_worse_than_baseline_count"],
        "controlled_regression_count": policy_audit["controlled_regression_count"],
        "infeasible_reason_codes": infeasible_reason_codes,
        "next_required_change": next_required_change,
        "scenario_families": config["scenario_families"],
        "scenario_matrix_seed": config["scenario_matrix_seed"],
        "coverage_radius_cells": config["coverage_radius_cells"],
        "path_budget_m": config["path_budget_m"],
        "uses_policy_guidance": True,
        "uses_checkpoint_inference": True,
        "uses_ppo_policy": True,
        "policy_read_only": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "runs_new_ppo_update": False,
        "modifies_network": False,
        "modifies_action_space": False,
        "modifies_default_astar": False,
        "uses_path_planner": False,
        "uses_npz_or_sidecar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    })
    rejection_report = _rejection_report(reason_codes, scenario_results)
    manifest = _manifest(config_path, output_root, paths, summary)

    write_jsonl(paths["scenario_results"], [_scenario_row(result) for result in scenario_results])
    write_json(paths["family_summary"], family_summary)
    write_json(paths["policy_vs_baseline_audit"], policy_audit)
    write_json(paths["budget_audit"], budget_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, family_summary, rejection_report), encoding="utf-8")
    return summary


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
    if not isinstance(payload.get("source_policy_guided_config"), str) or not payload["source_policy_guided_config"]:
        raise ConfigError("source_policy_guided_config must be a non-empty string")
    normalized = dict(payload)
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["path_budget_m"] = _nonnegative_float(payload.get("path_budget_m"), "path_budget_m")
    normalized["coverage_radius_cells"] = _nonnegative_int(payload.get("coverage_radius_cells"), "coverage_radius_cells")
    normalized["scenario_matrix_seed"] = _nonnegative_int(
        payload.get("scenario_matrix_seed", 99),
        "scenario_matrix_seed",
    )
    families = payload.get("scenario_families", list(DEFAULT_FAMILIES))
    if not isinstance(families, list) or not families:
        raise ConfigError("scenario_families must be a non-empty list")
    unknown = [family for family in families if family not in DEFAULT_FAMILIES]
    if unknown:
        raise ConfigError(f"unsupported scenario_families: {unknown}")
    normalized["scenario_families"] = list(families)
    return normalized


def _build_scenario_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for family in config["scenario_families"]:
        for index in range(3):
            scenario = _scenario_for_family(family, index, config)
            scenario["family_id"] = family
            scenario["matrix_index"] = index
            scenario["coverage_events"] = []
            scenarios.append(scenario)
    return scenarios


def _scenario_for_family(family: str, index: int, config: dict[str, Any]) -> dict[str, Any]:
    scenario_id = f"{family}-{index + 1:02d}"
    base = {
        "scenario_id": scenario_id,
        "grid": {"width": 100, "height": 100, "resolution_m": 10.0, "origin": [0.0, 0.0]},
        "start_cell": [[0, 0], [50, 50], [99, 99]][index],
        "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 100, "y1": 100},
        "blocked_rectangles": [],
        "unsafe_rectangles": [],
        "generalization_required": True,
        "expected_infeasible": False,
    }
    if family == "open_field":
        return base
    if family == "corridor":
        base["start_cell"] = [[0, 50], [50, 50], [99, 50]][index]
        base["roi"] = {"kind": "rect", "x0": 0, "y0": 30 + index * 2, "x1": 100, "y1": 70 - index * 2}
        return base
    if family == "rooms":
        base["start_cell"] = [[5, 5], [50, 50], [95, 95]][index]
        base["roi"] = [
            {"kind": "rect", "x0": 0, "y0": 0, "x1": 60, "y1": 60},
            {"kind": "rect", "x0": 20, "y0": 20, "x1": 80, "y1": 80},
            {"kind": "rect", "x0": 40, "y0": 40, "x1": 100, "y1": 100},
        ][index]
        return base
    if family == "blocked_roi":
        base["start_cell"] = [[0, 0], [0, 80], [80, 20]][index]
        base["blocked_rectangles"] = [
            {"x0": 42, "y0": 42, "x1": 58, "y1": 58},
            {"x0": 18 + index * 4, "y0": 76, "x1": 28 + index * 4, "y1": 86},
        ]
        return base
    if family == "unsafe_patch":
        base["start_cell"] = [[0, 99], [50, 50], [99, 0]][index]
        base["unsafe_rectangles"] = [
            {"x0": 20, "y0": 20, "x1": 40, "y1": 40},
            {"x0": 60, "y0": 60, "x1": 80, "y1": 80},
        ]
        return base
    if family == "narrow_passage":
        base["start_cell"] = [[5, 50], [50, 50], [95, 50]][index]
        band_half_width = 12 + index * 2
        base["blocked_rectangles"] = [
            {"x0": 0, "y0": 0, "x1": 100, "y1": 50 - band_half_width},
            {"x0": 0, "y0": 50 + band_half_width, "x1": 100, "y1": 100},
        ]
        return base
    if family == "cells_roi":
        base["start_cell"] = [[0, 0], [50, 50], [99, 99]][index]
        cells = []
        offset = index * 3
        for y in range(10 + offset, 91, 10):
            for x in range(10, 91, 10):
                cells.append([x, y])
        base["roi"] = {"kind": "cells", "cells": cells}
        return base
    if family == "budget_limited":
        base["start_cell"] = [[0, 0], [0, 50], [50, 0]][index]
        base["path_budget_m"] = 10.0
        base["generalization_required"] = False
        base["expected_infeasible"] = True
        return base
    raise ConfigError(f"unsupported scenario family: {family}")


def _summary_reason_codes(
    config: dict[str, Any],
    policy_load: dict[str, Any],
    scenario_results: list[dict[str, Any]],
    policy_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    required_results = [result for result in scenario_results if result["generalization_required"]]
    required_denominator = sum(result["reachable_safe_cell_count"] for result in required_results)
    required_covered = sum(result["covered_reachable_safe_cell_count"] for result in required_results)
    aggregate = (required_covered / required_denominator) if required_denominator else 0.0
    min_required = min((result["achieved_coverage_rate"] for result in required_results), default=0.0)
    failed_required = [result for result in required_results if result["status"] == "failed"]
    if failed_required:
        reasons.append("required_scenario_failed")
        for result in failed_required:
            reasons.extend(result["reason_codes"])
    if required_denominator <= 0:
        reasons.append("coverage_denominator_invalid")
    if required_denominator > 0 and aggregate + 1.0e-12 < config["target_coverage_rate"]:
        reasons.append("aggregate_coverage_target_not_met")
    if required_results and min_required + 1.0e-12 < config["target_coverage_rate"]:
        reasons.append("min_required_scenario_coverage_target_not_met")
    reasons.extend(policy_load.get("reason_codes", []))
    if not policy_load.get("policy_loaded"):
        reasons.append("policy_guidance_unavailable")
    if not policy_audit["policy_guidance_applied"]:
        reasons.append("policy_guidance_unavailable")
    return unique_sorted(reasons)


def _family_summary(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in scenario_results:
        grouped[result["family_id"]].append(result)
    families: dict[str, dict[str, Any]] = {}
    passed_family_count = 0
    failed_family_count = 0
    for family_id, results in sorted(grouped.items()):
        required = [result for result in results if result["generalization_required"]]
        failed_required = [result for result in required if result["status"] == "failed"]
        required_coverages = [result["achieved_coverage_rate"] for result in required]
        family_passed = not failed_required
        if family_passed:
            passed_family_count += 1
        else:
            failed_family_count += 1
        reason_counts = Counter(reason for result in results for reason in result["reason_codes"])
        infeasible_counts = Counter(reason for result in results for reason in result["infeasible_reason_codes"])
        families[family_id] = {
            "scenario_count": len(results),
            "required_scenario_count": len(required),
            "passed_scenario_count": sum(1 for result in results if result["status"] == "passed"),
            "failed_scenario_count": sum(1 for result in results if result["status"] == "failed"),
            "failed_required_scenario_count": len(failed_required),
            "expected_infeasible_scenario_count": sum(1 for result in results if result["expected_infeasible"]),
            "min_required_coverage_rate": min(required_coverages) if required_coverages else None,
            "mean_required_coverage_rate": (
                sum(required_coverages) / len(required_coverages) if required_coverages else None
            ),
            "reason_code_counts": dict(reason_counts),
            "infeasible_reason_code_counts": dict(infeasible_counts),
            "family_passed": family_passed,
        }
    return {
        "schema_version": FAMILY_SUMMARY_SCHEMA_VERSION,
        "family_count": len(families),
        "passed_family_count": passed_family_count,
        "failed_family_count": failed_family_count,
        "families": families,
    }


def _policy_vs_baseline_audit(policy_load: dict[str, Any], scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = sum(result["policy_scored_candidate_count"] for result in scenario_results)
    guided = sum(result["policy_guided_decision_count"] for result in scenario_results)
    selected = sum(result["policy_selected_decision_count"] for result in scenario_results)
    fallbacks = sum(result["policy_guard_fallback_count"] for result in scenario_results)
    agreement_denominator = sum(result["baseline_agreement_denominator"] for result in scenario_results)
    agreement_count = sum(result["baseline_agreement_count"] for result in scenario_results)
    controlled_regression = sum(result["controlled_regression_count"] for result in scenario_results)
    non_agreement = max(guided - agreement_count, 0)
    policy_worse = controlled_regression
    policy_better = max(non_agreement - fallbacks - policy_worse, 0)
    return {
        "schema_version": POLICY_AUDIT_SCHEMA_VERSION,
        "policy_load": policy_load,
        "policy_loaded": policy_load["policy_loaded"],
        "policy_guidance_applied": policy_load["policy_loaded"] and scored > 0 and guided > 0,
        "policy_scored_candidate_count": scored,
        "policy_guided_decision_count": guided,
        "policy_selected_decision_count": selected,
        "policy_guard_fallback_count": fallbacks,
        "baseline_agreement_count": agreement_count,
        "baseline_agreement_denominator": agreement_denominator,
        "baseline_agreement_rate": agreement_count / agreement_denominator if agreement_denominator else 0.0,
        "policy_better_than_baseline_count": policy_better,
        "policy_worse_than_baseline_count": policy_worse,
        "controlled_regression_count": controlled_regression,
    }


def _budget_audit(config: dict[str, Any], scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": BUDGET_AUDIT_SCHEMA_VERSION,
        "path_budget_m": config["path_budget_m"],
        "scenario_budget": [
            {
                "scenario_id": result["scenario_id"],
                "family_id": result["family_id"],
                "generalization_required": result["generalization_required"],
                "expected_infeasible": result["expected_infeasible"],
                "planned_path_cost_m": result["planned_path_cost_m"],
                "path_budget_m": result["path_budget_m"],
                "path_budget_exhausted": result["path_budget_exhausted"],
                "coverage_target_met": result["coverage_target_met"],
                "reason_codes": result["reason_codes"],
            }
            for result in scenario_results
        ],
    }


def _rejection_report(reason_codes: list[str], scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(Counter(reason_codes)),
        "scenario_failure_reason_code_counts": dict(
            Counter(reason for result in scenario_results for reason in result["reason_codes"])
        ),
        "scenario_rejections": [
            {
                "scenario_id": result["scenario_id"],
                "family_id": result["family_id"],
                "generalization_required": result["generalization_required"],
                "expected_infeasible": result["expected_infeasible"],
                "status": result["status"],
                "reason_codes": result["reason_codes"],
                "infeasible_reason_codes": result["infeasible_reason_codes"],
            }
            for result in scenario_results
            if result["status"] == "failed" or result["infeasible_reason_codes"]
        ],
    }


def _scenario_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCENARIO_RESULT_SCHEMA_VERSION,
        "scenario_id": result["scenario_id"],
        "family_id": result["family_id"],
        "status": result["status"],
        "reason_codes": result["reason_codes"],
        "infeasible_reason_codes": result["infeasible_reason_codes"],
        "generalization_required": result["generalization_required"],
        "expected_infeasible": result["expected_infeasible"],
        "target_coverage_rate": result["target_coverage_rate"],
        "achieved_coverage_rate": result["achieved_coverage_rate"],
        "coverage_target_met": result["coverage_target_met"],
        "reachable_safe_cell_count": result["reachable_safe_cell_count"],
        "covered_reachable_safe_cell_count": result["covered_reachable_safe_cell_count"],
        "policy_scored_candidate_count": result["policy_scored_candidate_count"],
        "policy_guided_decision_count": result["policy_guided_decision_count"],
        "policy_guard_fallback_count": result["policy_guard_fallback_count"],
        "baseline_agreement_count": result["baseline_agreement_count"],
        "baseline_agreement_denominator": result["baseline_agreement_denominator"],
        "controlled_regression_count": result["controlled_regression_count"],
        "planned_path_cost_m": result["planned_path_cost_m"],
        "path_budget_m": result["path_budget_m"],
        "path_budget_exhausted": result["path_budget_exhausted"],
        "replanning_cycle_count": result["replanning_cycle_count"],
        "roi_cell_count": result["roi_cell_count"],
        "blocked_roi_cell_count": result["blocked_roi_cell_count"],
        "unsafe_roi_cell_count": result["unsafe_roi_cell_count"],
        "unreachable_roi_cell_count": result["unreachable_roi_cell_count"],
    }


def _manifest(
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "stage": "Global 99 Multi-Map Generalization v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "boundary": {
            "uses_policy_guidance": True,
            "uses_checkpoint_inference": True,
            "policy_read_only": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "uses_path_planner": False,
            "uses_npz_or_sidecar": False,
        },
    }


def _render_report(
    summary: dict[str, Any],
    family_summary: dict[str, Any],
    rejection_report: dict[str, Any],
) -> str:
    lines = [
        "# Global 99 Multi-Map Generalization v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- scenario_count: `{summary['scenario_count']}`",
        f"- required_scenario_count: `{summary['required_scenario_count']}`",
        f"- aggregate_achieved_coverage_rate: `{summary['aggregate_achieved_coverage_rate']}`",
        f"- min_scenario_achieved_coverage_rate: `{summary['min_scenario_achieved_coverage_rate']}`",
        f"- policy_guidance_applied: `{summary['policy_guidance_applied']}`",
        f"- policy_scored_candidate_count: `{summary['policy_scored_candidate_count']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Families",
        "",
    ]
    for family_id, family in family_summary["families"].items():
        lines.extend(
            [
                f"### {family_id}",
                "",
                f"- scenario_count: `{family['scenario_count']}`",
                f"- required_scenario_count: `{family['required_scenario_count']}`",
                f"- failed_required_scenario_count: `{family['failed_required_scenario_count']}`",
                f"- min_required_coverage_rate: `{family['min_required_coverage_rate']}`",
                f"- infeasible_reason_code_counts: `{family['infeasible_reason_code_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Rejection Report",
            "",
            f"- failure_reason_code_counts: `{rejection_report['failure_reason_code_counts']}`",
            "",
            "This stage verifies deterministic synthetic multi-map generalization. It does not train PPO, publish a checkpoint, replace default policy, connect a real executor, call path-planner, use NPZ/sidecar maps, or modify network/action space/default A*.",
            "",
        ]
    )
    return "\n".join(lines)


def _bounded_float(value: Any, label: str) -> float:
    numeric = _nonnegative_float(value, label)
    if numeric <= 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be > 0 and <= 1")
    return numeric


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric < 0.0:
        raise ConfigError(f"{label} must be >= 0")
    return numeric


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if value < 0:
        raise ConfigError(f"{label} must be >= 0")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
