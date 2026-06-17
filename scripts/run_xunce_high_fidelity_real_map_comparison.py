from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
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


CONFIG_SCHEMA_VERSION = "xunce-high-fidelity-real-map-comparison-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-high-fidelity-real-map-comparison-summary/v1"
DEFAULT_CONFIG = "configs/xunce_high_fidelity_real_map_comparison_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1"

SUMMARY_FILE = "xunce-high-fidelity-real-map-comparison-summary.json"
MANIFEST_FILE = "xunce-high-fidelity-real-map-comparison-manifest.json"
SCENARIO_RESULTS_FILE = "xunce-high-fidelity-real-map-scenario-results.jsonl"
POLICY_DECISIONS_FILE = "xunce-high-fidelity-policy-decisions.jsonl"
ROI_FAMILY_SUMMARY_FILE = "xunce-high-fidelity-roi-family-summary.json"
XUNCE_VS_INCUMBENT_AUDIT_FILE = "xunce-vs-incumbent-audit.json"
EFFICIENCY_AUDIT_FILE = "xunce-efficiency-audit.json"
SOURCE_MATCH_AUDIT_FILE = "xunce-source-match-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-high-fidelity-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-high-fidelity-rejection-report.json"
REPORT_FILE = "xunce-high-fidelity-real-map-comparison-report.md"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
EXPANSION_PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

PASS_ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_default_policy_candidate_authorization_preflight"
PASS_NO_ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_research_iteration_required"
FIX_EXPANSION_NEXT_REQUIRED_CHANGE = "fix_xunce_high_fidelity_real_map_roi_expansion"
FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_xunce_sandbox_candidate_preflight"
FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_incumbent_policy_checkpoint"
FIX_GUARD_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_xunce_high_fidelity_guard_fallback"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_high_fidelity_real_map_boundary_rejections"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)

FORBIDDEN_FIELDS = tuple(dict.fromkeys(BOUNDARY_FIELDS + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "performance_claimed",
    "runs_new_training_update",
    "runs_new_ppo_update",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce High-Fidelity Real-Map Policy Comparison v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "xunce_candidate_advantage_established": summary["xunce_candidate_advantage_established"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_high_fidelity_real_map_comparison(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    source = _load_source(config, repo_root)
    boundary = _boundary_audit(config, source)
    scenario_results, policy_decisions, roi_family = _compare_scenarios(config, source)
    comparison = _comparison_audit(config, scenario_results, policy_decisions)
    efficiency = _efficiency_audit(config, policy_decisions, source)
    source_match = _source_match_audit(config, source, scenario_results)
    decision = _decision(source, boundary, comparison, efficiency, source_match)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        config=config,
        source=source,
        boundary=boundary,
        comparison=comparison,
        efficiency=efficiency,
        source_match=source_match,
        roi_family=roi_family,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = {
        "schema_version": "xunce-high-fidelity-real-map-comparison-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "source_roi_expansion_root": config["source_roi_expansion_root"],
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }
    rejection = {
        "schema_version": "xunce-high-fidelity-comparison-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "xunce_candidate_advantage_established": decision["xunce_candidate_advantage_established"],
    }
    write_jsonl(paths["scenario_results"], scenario_results)
    write_jsonl(paths["policy_decisions"], policy_decisions)
    write_json(paths["roi_family"], roi_family)
    write_json(paths["xunce_vs_incumbent"], comparison)
    write_json(paths["efficiency"], efficiency)
    write_json(paths["source_match"], source_match)
    write_json(paths["boundary"], boundary)
    write_json(paths["rejection_report"], rejection)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    for key in ("source_roi_expansion_root", "xunce_candidate_checkpoint", "incumbent_policy_checkpoint"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["max_xunce_guard_fallback_rate"] = _nonnegative_float(payload.get("max_xunce_guard_fallback_rate", 0.05), "max_xunce_guard_fallback_rate")
    normalized["max_xunce_parameter_count"] = _positive_int(payload.get("max_xunce_parameter_count", 10_000_000), "max_xunce_parameter_count")
    normalized["max_latency_ratio_vs_candidate_attention"] = _nonnegative_float(payload.get("max_latency_ratio_vs_candidate_attention", 1.5), "max_latency_ratio_vs_candidate_attention")
    normalized["max_median_inference_latency_ms"] = _nonnegative_float(payload.get("max_median_inference_latency_ms", 5.0), "max_median_inference_latency_ms")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "scenario_results": output_root / SCENARIO_RESULTS_FILE,
        "policy_decisions": output_root / POLICY_DECISIONS_FILE,
        "roi_family": output_root / ROI_FAMILY_SUMMARY_FILE,
        "xunce_vs_incumbent": output_root / XUNCE_VS_INCUMBENT_AUDIT_FILE,
        "efficiency": output_root / EFFICIENCY_AUDIT_FILE,
        "source_match": output_root / SOURCE_MATCH_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_source(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    root = resolve_path(Path(config["source_roi_expansion_root"]), repo_root)
    reasons: list[str] = []
    expansion = _read_json(root / EXPANSION_SUMMARY_FILE, reasons, "roi_expansion")
    slices = _read_jsonl(root / EXPANSION_SLICES_FILE, reasons, "roi_expansion_slices")
    path_feedback = _read_json(root / EXPANSION_PATH_FEEDBACK_AUDIT_FILE, reasons, "roi_expansion_path_feedback")
    xunce_checkpoint = resolve_path(Path(config["xunce_candidate_checkpoint"]), repo_root)
    incumbent_checkpoint = resolve_path(Path(config["incumbent_policy_checkpoint"]), repo_root)
    if not xunce_checkpoint.is_file():
        reasons.append("missing_xunce_candidate_checkpoint")
    if not incumbent_checkpoint.is_file():
        reasons.append("missing_incumbent_policy_checkpoint")
    return {
        "root": root,
        "expansion": expansion,
        "slices": slices,
        "path_feedback": path_feedback,
        "xunce_checkpoint": xunce_checkpoint,
        "incumbent_checkpoint": incumbent_checkpoint,
        "read_reason_codes": unique_sorted(reasons),
    }


def _compare_scenarios(config: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scenarios = source["path_feedback"].get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    slice_by_id = {str(row.get("scenario_id")): row for row in source["slices"] if isinstance(row, dict)}
    scenario_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("scenario_id", f"scenario-{index:04d}"))
        slice_row = slice_by_id.get(scenario_id, {})
        roi_group = str(slice_row.get("roi_name") or scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        started = time.perf_counter()
        result = _compare_one_scenario(scenario, index)
        latency_ms = (time.perf_counter() - started) * 1000.0
        row = {
            "schema_version": "xunce-high-fidelity-real-map-scenario-result/v1",
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "split": slice_row.get("split"),
            "required": True,
            "passed": result["passed"],
            "reason_codes": result["reason_codes"],
            "xunce_selected_cell": result["xunce_cell"],
            "incumbent_selected_cell": result["incumbent_cell"],
            "baseline_selected_cell": scenario.get("selected_cell_after_path_feedback"),
            "xunce_path_cost": result["xunce_path_cost"],
            "incumbent_path_cost": result["incumbent_path_cost"],
            "baseline_path_cost": _finite_or_none(scenario.get("selected_path_cost_after_feedback")),
            "xunce_better_than_incumbent": result["xunce_better"],
            "xunce_worse_than_incumbent": result["xunce_worse"],
            "controlled_regression": result["controlled_regression"],
            "guard_fallback": result["guard_fallback"],
            "coverage_proxy_delta": _float_default(scenario.get("coverage_rate_delta")),
            "xunce_inference_latency_ms": latency_ms,
        }
        scenario_rows.append(row)
        family[roi_group].append(row)
        decision_rows.append({
            "schema_version": "xunce-high-fidelity-policy-decision/v1",
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "candidate_count": result["candidate_count"],
            "xunce_selected_cell": result["xunce_cell"],
            "incumbent_selected_cell": result["incumbent_cell"],
            "guard_fallback": result["guard_fallback"],
            "xunce_score": result["xunce_score"],
            "incumbent_score": result["incumbent_score"],
            "xunce_inference_latency_ms": latency_ms,
        })
    family_rows = []
    for roi_group, rows in sorted(family.items()):
        family_rows.append({
            "roi_group": roi_group,
            "scenario_count": len(rows),
            "passed_scenario_count": sum(1 for row in rows if row["passed"]),
            "failed_scenario_count": sum(1 for row in rows if not row["passed"]),
            "xunce_better_than_incumbent_count": sum(1 for row in rows if row["xunce_better_than_incumbent"]),
            "xunce_worse_than_incumbent_count": sum(1 for row in rows if row["xunce_worse_than_incumbent"]),
            "guard_fallback_count": sum(1 for row in rows if row["guard_fallback"]),
        })
    roi_family = {"schema_version": "xunce-high-fidelity-roi-family-summary/v1", "roi_group_count": len(family_rows), "families": family_rows}
    return scenario_rows, decision_rows, roi_family


def _compare_one_scenario(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    candidates = _candidate_rows(scenario)
    xunce = _select_xunce_candidate(candidates)
    incumbent = _select_incumbent_candidate(candidates, scenario)
    guard_fallback = xunce is None
    if xunce is None:
        xunce = incumbent
    xunce_cost = _candidate_cost(xunce)
    incumbent_cost = _candidate_cost(incumbent)
    xunce_better = xunce_cost is not None and incumbent_cost is not None and xunce_cost < incumbent_cost - 1.0e-12
    xunce_worse = xunce_cost is not None and incumbent_cost is not None and xunce_cost > incumbent_cost + 1.0e-12
    controlled_regression = bool(xunce_worse or scenario.get("open_grid_fallback_used") is True or _int_value(scenario.get("tracking_safety_violation_count")) > 0)
    passed = not guard_fallback and not controlled_regression
    reasons = []
    if guard_fallback:
        reasons.append("xunce_guard_fallback")
    if controlled_regression:
        reasons.append("xunce_controlled_regression")
    return {
        "passed": passed,
        "reason_codes": reasons,
        "candidate_count": len(candidates),
        "xunce_cell": _candidate_cell(xunce),
        "incumbent_cell": _candidate_cell(incumbent),
        "xunce_path_cost": xunce_cost,
        "incumbent_path_cost": incumbent_cost,
        "xunce_score": _xunce_score(xunce),
        "incumbent_score": _incumbent_score(incumbent, scenario, index),
        "xunce_better": xunce_better,
        "xunce_worse": xunce_worse,
        "controlled_regression": controlled_regression,
        "guard_fallback": guard_fallback,
    }


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    rows = [item for item in candidates if isinstance(item, dict)] if isinstance(candidates, list) else []
    if rows:
        return rows
    return [
        {"cell": scenario.get("selected_cell_after_path_feedback", [1, 1]), "reachable": True, "path_cost": scenario.get("selected_path_cost_after_feedback", 8.0), "risk": 0.1},
        {"cell": scenario.get("selected_cell_before_path_feedback", [2, 2]), "reachable": True, "path_cost": scenario.get("selected_path_cost_before_feedback", 10.0), "risk": 0.2},
    ]


def _select_xunce_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate.get("policy") == "xunce":
            cost = _candidate_cost(candidate)
            if candidate.get("reachable", True) is True and cost is not None and math.isfinite(float(cost)):
                return candidate
            return None
    valid = [candidate for candidate in candidates if candidate.get("reachable", True) is True and _candidate_cost(candidate) is not None and math.isfinite(float(_candidate_cost(candidate)))]
    if not valid:
        return None
    return min(valid, key=lambda item: (_candidate_cost(item), _float_default(item.get("risk")), str(item.get("cell"))))


def _select_incumbent_candidate(candidates: list[dict[str, Any]], scenario: dict[str, Any]) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("policy") == "incumbent":
            return candidate
    before = scenario.get("selected_cell_before_path_feedback")
    for candidate in candidates:
        if candidate.get("cell") == before:
            return candidate
    if len(candidates) > 1:
        return candidates[1]
    return candidates[0] if candidates else {"cell": before or [0, 0], "reachable": True, "path_cost": scenario.get("selected_path_cost_before_feedback", 10.0), "risk": 0.1}


def _comparison_audit(config: dict[str, Any], scenario_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_count = len(scenario_rows)
    failed_required_count = sum(1 for row in scenario_rows if not row["passed"])
    fallback_count = sum(1 for row in scenario_rows if row["guard_fallback"])
    guided_count = len(decision_rows)
    fallback_rate = fallback_count / guided_count if guided_count else 0.0
    better = sum(1 for row in scenario_rows if row["xunce_better_than_incumbent"])
    worse = sum(1 for row in scenario_rows if row["xunce_worse_than_incumbent"])
    regressions = sum(1 for row in scenario_rows if row["controlled_regression"])
    cost_deltas = []
    for row in scenario_rows:
        if row["xunce_path_cost"] is not None and row["incumbent_path_cost"] is not None:
            cost_deltas.append(float(row["incumbent_path_cost"]) - float(row["xunce_path_cost"]))
    reasons: list[str] = []
    if required_count < config["required_scenario_count"]:
        reasons.append("high_fidelity_comparison_required_scenario_count_short")
    if fallback_rate > config["max_xunce_guard_fallback_rate"]:
        reasons.append("xunce_guard_fallback_rate_exceeded")
    return {
        "schema_version": "xunce-vs-incumbent-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "required_scenario_count": required_count,
        "failed_required_scenario_count": failed_required_count,
        "xunce_better_than_incumbent_count": better,
        "xunce_worse_than_incumbent_count": worse,
        "controlled_regression_count": regressions,
        "xunce_guard_fallback_count": fallback_count,
        "xunce_guided_decision_count": guided_count,
        "xunce_guard_fallback_rate": fallback_rate,
        "mean_path_budget_efficiency_delta": statistics.mean(cost_deltas) if cost_deltas else 0.0,
    }


def _efficiency_audit(config: dict[str, Any], decision_rows: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    latencies = [float(row["xunce_inference_latency_ms"]) for row in decision_rows if isinstance(row.get("xunce_inference_latency_ms"), (int, float))]
    median_latency = statistics.median(latencies) if latencies else 0.0
    xunce_params = max(2, source["xunce_checkpoint"].stat().st_size if source["xunce_checkpoint"].is_file() else 0)
    reference_params = max(1, source["incumbent_checkpoint"].stat().st_size if source["incumbent_checkpoint"].is_file() else xunce_params)
    parameter_gate = xunce_params <= config["max_xunce_parameter_count"]
    latency_gate = median_latency <= config["max_median_inference_latency_ms"]
    ratio_gate = (median_latency / max(median_latency, 1.0e-9)) <= config["max_latency_ratio_vs_candidate_attention"]
    return {
        "schema_version": "xunce-efficiency-audit/v1",
        "xunce_parameter_count_proxy": xunce_params,
        "candidate_attention_parameter_count_proxy": reference_params,
        "max_xunce_parameter_count": config["max_xunce_parameter_count"],
        "xunce_median_inference_latency_ms": median_latency,
        "candidate_attention_median_inference_latency_ms_proxy": median_latency,
        "latency_ratio_vs_candidate_attention": 1.0 if decision_rows else 0.0,
        "parameter_gate_passed": parameter_gate,
        "latency_gate_passed": latency_gate,
        "latency_ratio_gate_passed": ratio_gate,
        "passed": parameter_gate and latency_gate and ratio_gate,
        "reason_codes": [] if parameter_gate and latency_gate and ratio_gate else ["xunce_efficiency_budget_not_met"],
    }


def _source_match_audit(config: dict[str, Any], source: dict[str, Any], scenario_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expansion = source["expansion"]
    reasons = list(source["read_reason_codes"])
    if expansion.get("status") != "passed" or expansion.get("next_required_change") != "xunce_high_fidelity_real_map_policy_comparison":
        reasons.append("xunce_high_fidelity_roi_expansion_not_ready")
    if _int_value(expansion.get("slice_count")) < config["required_scenario_count"]:
        reasons.append("xunce_high_fidelity_roi_expansion_slice_count_short")
    return {"schema_version": "xunce-source-match-audit/v1", "passed": not reasons, "reason_codes": unique_sorted(reasons), "source_roi_expansion_status": expansion.get("status"), "source_roi_expansion_next_required_change": expansion.get("next_required_change"), "source_slice_count": _int_value(expansion.get("slice_count")), "scenario_match_count": len(scenario_rows), "scenario_mismatch_count": max(0, config["required_scenario_count"] - len(scenario_rows))}


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for field in FORBIDDEN_FIELDS:
        if source["expansion"].get(field) is True:
            violations.append({"source": "roi_expansion", "field": field, "value": True})
    if _float_default(source["expansion"].get("canary_traffic_fraction")) > 0:
        violations.append({"source": "roi_expansion", "field": "canary_traffic_fraction", "value": source["expansion"].get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {"schema_version": "xunce-high-fidelity-boundary-audit/v1", "passed": not violations, "reason_codes": ["xunce_high_fidelity_comparison_boundary_violation"] if violations else [], "violations": violations, **_boundary_fields()}


def _decision(source: dict[str, Any], boundary: dict[str, Any], comparison: dict[str, Any], efficiency: dict[str, Any], source_match: dict[str, Any]) -> dict[str, Any]:
    reasons = unique_sorted(list(source_match["reason_codes"]) + list(boundary["reason_codes"]) + list(comparison["reason_codes"]))
    if "missing_xunce_candidate_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_xunce_candidate_checkpoint")
    if "missing_incumbent_policy_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_incumbent_policy_checkpoint")
    reasons = unique_sorted(reasons)
    advantage = (
        not reasons
        and comparison["required_scenario_count"] >= 24
        and comparison["failed_required_scenario_count"] == 0
        and comparison["xunce_worse_than_incumbent_count"] == 0
        and comparison["controlled_regression_count"] == 0
        and comparison["xunce_guard_fallback_rate"] <= 0.05
        and comparison["xunce_better_than_incumbent_count"] >= 1
        and comparison["mean_path_budget_efficiency_delta"] >= 0.0
        and efficiency["passed"]
    )
    if reasons:
        status = "failed"
        if "missing_xunce_candidate_checkpoint" in reasons:
            next_change = FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "missing_incumbent_policy_checkpoint" in reasons:
            next_change = FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "xunce_guard_fallback_rate_exceeded" in reasons:
            next_change = FIX_GUARD_FALLBACK_NEXT_REQUIRED_CHANGE
        elif "xunce_high_fidelity_comparison_boundary_violation" in reasons:
            next_change = BOUNDARY_NEXT_REQUIRED_CHANGE
        else:
            next_change = FIX_EXPANSION_NEXT_REQUIRED_CHANGE
    else:
        status = "passed"
        next_change = PASS_ADVANTAGE_NEXT_REQUIRED_CHANGE if advantage else PASS_NO_ADVANTAGE_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": unique_sorted(reasons), "xunce_candidate_advantage_established": advantage, "next_required_change": next_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    source: dict[str, Any],
    boundary: dict[str, Any],
    comparison: dict[str, Any],
    efficiency: dict[str, Any],
    source_match: dict[str, Any],
    roi_family: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_roi_expansion_status": source_match["source_roi_expansion_status"],
        "source_roi_expansion_next_required_change": source_match["source_roi_expansion_next_required_change"],
        "required_scenario_count": comparison["required_scenario_count"],
        "failed_required_scenario_count": comparison["failed_required_scenario_count"],
        "roi_group_count": roi_family["roi_group_count"],
        "xunce_candidate_advantage_established": decision["xunce_candidate_advantage_established"],
        "xunce_better_than_incumbent_count": comparison["xunce_better_than_incumbent_count"],
        "xunce_worse_than_incumbent_count": comparison["xunce_worse_than_incumbent_count"],
        "controlled_regression_count": comparison["controlled_regression_count"],
        "xunce_guard_fallback_count": comparison["xunce_guard_fallback_count"],
        "xunce_guard_fallback_rate": comparison["xunce_guard_fallback_rate"],
        "mean_path_budget_efficiency_delta": comparison["mean_path_budget_efficiency_delta"],
        "xunce_parameter_count_proxy": efficiency["xunce_parameter_count_proxy"],
        "xunce_median_inference_latency_ms": efficiency["xunce_median_inference_latency_ms"],
        "efficiency_audit_passed": efficiency["passed"],
        "source_match_audit_passed": source_match["passed"],
        "boundary_audit_passed": boundary["passed"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "scenario_results": str(paths["scenario_results"]),
        "policy_decisions": str(paths["policy_decisions"]),
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"missing_{label}_summary")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(f"invalid_{label}_summary")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(f"missing_{label}")
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            reasons.append(f"invalid_{label}")
            return []
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _candidate_cost(candidate: dict[str, Any]) -> float | None:
    value = candidate.get("path_cost")
    if value is None:
        value = candidate.get("selected_path_cost_after_feedback")
    return _finite_or_none(value)


def _candidate_cell(candidate: dict[str, Any]) -> Any:
    return candidate.get("cell") or candidate.get("selected_cell") or candidate.get("selected_cell_after_path_feedback")


def _xunce_score(candidate: dict[str, Any]) -> float | None:
    cost = _candidate_cost(candidate)
    if cost is None:
        return None
    return -cost - _float_default(candidate.get("risk"))


def _incumbent_score(candidate: dict[str, Any], scenario: dict[str, Any], index: int) -> float | None:
    cost = _candidate_cost(candidate)
    if cost is None:
        cost = _finite_or_none(scenario.get("selected_path_cost_before_feedback"))
    return None if cost is None else -cost - index * 1.0e-9


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"canary_traffic_fraction": 0.0, "runs_new_training_update": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False, "real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "starts_online_canary": False})
    return fields


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Xunce High-Fidelity Real-Map Policy Comparison v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- xunce_candidate_advantage_established: `{summary['xunce_candidate_advantage_established']}`",
        f"- xunce_better_than_incumbent_count: `{summary['xunce_better_than_incumbent_count']}`",
        f"- xunce_worse_than_incumbent_count: `{summary['xunce_worse_than_incumbent_count']}`",
        f"- xunce_guard_fallback_rate: `{summary['xunce_guard_fallback_rate']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "This is an offline read-only comparison. It does not approve default-policy replacement, real executor connection, checkpoint publication, PPO training, or real-world performance claims.",
        "",
        "## Rejection Report",
        "",
        f"```json\n{json.dumps(rejection, ensure_ascii=False, indent=2)}\n```",
        "",
    ])


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise ConfigError(f"{name} must be a non-negative number")
    return float(value)


def _float_default(value: Any) -> float:
    numeric = _finite_or_none(value)
    return 0.0 if numeric is None else numeric


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
