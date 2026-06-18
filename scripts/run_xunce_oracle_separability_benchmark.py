from __future__ import annotations

import argparse
import json
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


CONFIG_SCHEMA_VERSION = "xunce-oracle-separability-benchmark-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-oracle-separability-summary/v1"
DEFAULT_CONFIG = "configs/xunce_oracle_separability_benchmark_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_oracle_separability_benchmark_v1"

SUMMARY_FILE = "xunce-oracle-separability-summary.json"
SCENARIOS_FILE = "xunce-oracle-separability-scenarios.jsonl"
ROI_SUMMARY_FILE = "xunce-oracle-separability-roi-summary.json"
DECISION_AUDIT_FILE = "xunce-oracle-separability-decision-audit.json"
MANIFEST_FILE = "xunce-oracle-separability-manifest.json"
REPORT_FILE = "xunce-oracle-separability-report.md"
STAGE18C_V2_DIR = "stage18c_v2"

MATERIALIZATION_SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"
REFINEMENT_SUMMARY_FILE = "xunce-cost-efficient-coverage-opportunity-summary.json"
PATH_FEEDBACK_FILE = "xunce-high-fidelity-path-feedback-audit.json"

PASS_NEXT_REQUIRED_CHANGE = "xunce_default_policy_candidate_authorization_preflight"
FIX_MATERIALIZATION_NEXT_REQUIRED_CHANGE = "run_candidate_level_coverage_opportunity_materialization"
FIX_STAGE18F1_NEXT_REQUIRED_CHANGE = "run_cost_efficient_coverage_opportunity_refinement"
EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE = "expand_roi_or_map_complexity"
TRAINING_ITERATION_NEXT_REQUIRED_CHANGE = "xunce_training_or_adapter_iteration_required"
RUN_STAGE18C_V2_NEXT_REQUIRED_CHANGE = "run_stage18c_v2_with_refined_cost_efficient_root"
EFFICIENCY_NEXT_REQUIRED_CHANGE = "refine_coverage_reward_and_cost_guard"
REFINE_COST_EFFICIENT_NEXT_REQUIRED_CHANGE = "refine_cost_efficient_coverage_opportunity"

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
    parser = argparse.ArgumentParser(description="Benchmark Xunce oracle separability after Stage 18E.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-materialized-coverage-root")
    parser.add_argument("--xunce-candidate-checkpoint")
    parser.add_argument("--incumbent-policy-checkpoint")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_materialized_coverage_root": args.source_materialized_coverage_root,
            "xunce_candidate_checkpoint": args.xunce_candidate_checkpoint,
            "incumbent_policy_checkpoint": args.incumbent_policy_checkpoint,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_oracle_separability_benchmark(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "oracle_separable": summary["oracle_separable"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_oracle_separability_benchmark(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / STAGE18C_V2_DIR).mkdir(parents=True, exist_ok=True)
    source = _load_source(config)
    scenario_rows = _scenario_rows(config, source)
    roi_summary = _roi_summary(scenario_rows)
    metrics = _metrics(config, source, scenario_rows, roi_summary)
    decision = _decision(metrics)
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
    write_jsonl(paths["scenarios"], scenario_rows)
    write_json(paths["roi_summary"], roi_summary)
    write_json(paths["decision_audit"], {"schema_version": "xunce-oracle-separability-decision-audit/v1", **decision, **_boundary_fields()})
    write_json(paths["manifest"], _manifest(generated_at, config_path, output_root, config, paths))
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
    for key in ("source_materialized_coverage_root", "xunce_candidate_checkpoint", "incumbent_policy_checkpoint"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["scenario_count"] = _positive_int(payload.get("scenario_count", 24), "scenario_count")
    normalized["rollout_steps"] = _positive_int(payload.get("rollout_steps", 10), "rollout_steps")
    normalized["oracle_better_scenario_fraction_threshold"] = _nonnegative_float(payload.get("oracle_better_scenario_fraction_threshold", 0.60), "oracle_better_scenario_fraction_threshold")
    normalized["oracle_better_roi_group_count_threshold"] = _positive_int(payload.get("oracle_better_roi_group_count_threshold", 3), "oracle_better_roi_group_count_threshold")
    normalized["candidate_refresh_mode"] = _string(payload.get("candidate_refresh_mode", "dynamic_from_coverage_memory"), "candidate_refresh_mode")
    normalized["coverage_metric_mode"] = _string(payload.get("coverage_metric_mode", "path_line_plus_endpoint"), "coverage_metric_mode")
    normalized["include_oracle_baselines"] = _bool(payload.get("include_oracle_baselines", True), "include_oracle_baselines")
    normalized["include_roi_weighted_coverage"] = _bool(payload.get("include_roi_weighted_coverage", True), "include_roi_weighted_coverage")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["source_materialized_coverage_root"])
    reasons: list[str] = []
    materialization_summary = _read_json(root / MATERIALIZATION_SUMMARY_FILE, [], "missing_stage18e_materialization")
    refinement_summary = _read_json(root / REFINEMENT_SUMMARY_FILE, [], "missing_stage18f1_refined_root")
    path_feedback = _read_json(root / PATH_FEEDBACK_FILE, reasons, "missing_stage18e_materialization")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    if not materialization_summary and not refinement_summary:
        reasons.append("missing_stage18f1_refined_root" if "cost_efficient_coverage_opportunity_refinement" in str(root) else "missing_stage18e_materialization")
    if materialization_summary.get("status") not in (None, "passed") and refinement_summary.get("status") not in (None, "passed"):
        reasons.append("missing_stage18e_materialization")
    if not Path(config["xunce_candidate_checkpoint"]).is_file():
        reasons.append("missing_xunce_candidate_checkpoint")
    if not Path(config["incumbent_policy_checkpoint"]).is_file():
        reasons.append("missing_incumbent_policy_checkpoint")
    source_has_refinement = bool(refinement_summary) or _scenarios_have_safe_efficient_field([row for row in scenarios if isinstance(row, dict)])
    return {
        "root": root,
        "materialization_summary": materialization_summary,
        "refinement_summary": refinement_summary,
        "source_cost_efficient_refinement_detected": source_has_refinement,
        "path_feedback": path_feedback,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "scenarios": output_root / SCENARIOS_FILE,
        "roi_summary": output_root / ROI_SUMMARY_FILE,
        "decision_audit": output_root / DECISION_AUDIT_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "report": output_root / REPORT_FILE,
        "stage18c_v2": output_root / STAGE18C_V2_DIR,
    }


def _scenario_rows(config: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scenario in source["scenarios"][: config["scenario_count"]]:
        scenario_id = str(scenario.get("scenario_id") or "")
        roi_group = str(scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        candidates = _candidate_rows(scenario)
        incumbent_index = _selected_index_from_scenario(scenario, "incumbent_selected_action_index", default=0)
        xunce_index = _selected_index_from_scenario(scenario, "xunce_selected_action_index", default=incumbent_index)
        greedy_index = _oracle_index(candidates, mode="greedy")
        cost_aware_index = _oracle_index(candidates, mode="cost_aware")
        incumbent = _candidate_at(candidates, incumbent_index)
        xunce = _candidate_at(candidates, xunce_index)
        greedy = _candidate_at(candidates, greedy_index)
        cost_aware = _candidate_at(candidates, cost_aware_index)
        incumbent_return = _coverage_value(incumbent)
        xunce_return = _coverage_value(xunce)
        greedy_return = _coverage_value(greedy)
        cost_return = _coverage_value(cost_aware)
        rows.append(
            {
                "schema_version": "xunce-oracle-separability-scenario/v1",
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "incumbent_selected_action_index": incumbent_index,
                "xunce_selected_action_index": xunce_index,
                "greedy_oracle_selected_action_index": greedy_index,
                "cost_aware_oracle_selected_action_index": cost_aware_index,
                "incumbent_coverage_return": incumbent_return,
                "xunce_coverage_return": xunce_return,
                "greedy_oracle_coverage_return": greedy_return,
                "cost_aware_oracle_coverage_return": cost_return,
                "greedy_oracle_delta_vs_incumbent": greedy_return - incumbent_return,
                "cost_aware_oracle_delta_vs_incumbent": cost_return - incumbent_return,
                "xunce_delta_vs_incumbent": xunce_return - incumbent_return,
                "incumbent_path_cost": _path_cost(incumbent),
                "cost_aware_oracle_path_cost": _path_cost(cost_aware),
                "cost_aware_oracle_efficiency_regression": bool(cost_aware and incumbent and _path_cost(cost_aware) > _path_cost(incumbent) + TOLERANCE),
                "greedy_oracle_mask_violation": greedy_index is not None and not _candidate_valid(greedy),
                "cost_aware_oracle_mask_violation": cost_aware_index is not None and not _candidate_valid(cost_aware),
                "open_grid_fallback_used": scenario.get("open_grid_fallback_used") is True or any(candidate.get("open_grid_fallback_used") is True for candidate in candidates),
            }
        )
    return rows


def _metrics(config: dict[str, Any], source: dict[str, Any], rows: list[dict[str, Any]], roi_summary: dict[str, Any]) -> dict[str, Any]:
    greedy_deltas = [float(row["greedy_oracle_delta_vs_incumbent"]) for row in rows]
    cost_deltas = [float(row["cost_aware_oracle_delta_vs_incumbent"]) for row in rows]
    xunce_deltas = [float(row["xunce_delta_vs_incumbent"]) for row in rows]
    better_rows = [row for row in rows if row["greedy_oracle_delta_vs_incumbent"] > TOLERANCE and row["cost_aware_oracle_delta_vs_incumbent"] > TOLERANCE]
    cost_efficiency_regression = sum(1 for row in rows if row["cost_aware_oracle_efficiency_regression"])
    mask_violations = sum(1 for row in rows if row["greedy_oracle_mask_violation"] or row["cost_aware_oracle_mask_violation"])
    fallback_count = sum(1 for row in rows if row["open_grid_fallback_used"])
    safe_count = _safe_efficient_opportunity_count(source["scenarios"])
    safe_available = safe_count > 0
    oracle_separable = bool(
        not source["reason_codes"]
        and (not source["source_cost_efficient_refinement_detected"] or safe_available)
        and _mean(greedy_deltas) > TOLERANCE
        and _mean(cost_deltas) > TOLERANCE
        and _safe_ratio(len(better_rows), len(rows)) >= float(config["oracle_better_scenario_fraction_threshold"])
        and int(roi_summary["oracle_better_roi_group_count"]) >= int(config["oracle_better_roi_group_count_threshold"])
        and cost_efficiency_regression == 0
        and mask_violations == 0
        and fallback_count == 0
    )
    xunce_advantage = bool(oracle_separable and _mean(xunce_deltas) > TOLERANCE and _mean(xunce_deltas) >= _mean(cost_deltas) - TOLERANCE)
    xunce_efficiency_regression = sum(1 for row in rows if row["xunce_delta_vs_incumbent"] > TOLERANCE and row["xunce_selected_action_index"] != row["incumbent_selected_action_index"])
    return {
        "reason_codes": unique_sorted(source["reason_codes"]),
        "scenario_count": len(rows),
        "greedy_oracle_coverage_return_delta_vs_incumbent": _mean(greedy_deltas),
        "cost_aware_oracle_coverage_return_delta_vs_incumbent": _mean(cost_deltas),
        "xunce_coverage_return_delta_vs_incumbent": _mean(xunce_deltas),
        "oracle_better_scenario_fraction": _safe_ratio(len(better_rows), len(rows)),
        "oracle_better_roi_group_count": int(roi_summary["oracle_better_roi_group_count"]),
        "cost_aware_oracle_efficiency_regression_count": cost_efficiency_regression,
        "oracle_mask_violation_count": mask_violations,
        "open_grid_fallback_count": fallback_count,
        "oracle_separable": oracle_separable,
        "xunce_coverage_advantage_established": xunce_advantage,
        "xunce_efficiency_regression_count": xunce_efficiency_regression,
        "source_cost_efficient_refinement_detected": source["source_cost_efficient_refinement_detected"],
        "safe_efficient_opportunity_count": safe_count,
        "safe_efficient_opportunity_available": safe_available,
    }


def _decision(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    if reasons:
        next_change = FIX_STAGE18F1_NEXT_REQUIRED_CHANGE if "missing_stage18f1_refined_root" in reasons else FIX_MATERIALIZATION_NEXT_REQUIRED_CHANGE
        return {"status": "failed", "reason_codes": unique_sorted(reasons), "next_required_change": next_change}
    if metrics["source_cost_efficient_refinement_detected"] and not metrics["safe_efficient_opportunity_available"]:
        return {"status": "passed", "reason_codes": ["safe_efficient_opportunity_insufficient"], "next_required_change": REFINE_COST_EFFICIENT_NEXT_REQUIRED_CHANGE}
    if metrics["xunce_coverage_advantage_established"] and metrics["xunce_efficiency_regression_count"] > 0:
        return {"status": "passed", "reason_codes": [], "next_required_change": EFFICIENCY_NEXT_REQUIRED_CHANGE}
    if metrics["xunce_coverage_advantage_established"]:
        return {"status": "passed", "reason_codes": [], "next_required_change": PASS_NEXT_REQUIRED_CHANGE}
    if metrics["oracle_separable"]:
        next_change = RUN_STAGE18C_V2_NEXT_REQUIRED_CHANGE if metrics["source_cost_efficient_refinement_detected"] else TRAINING_ITERATION_NEXT_REQUIRED_CHANGE
        return {"status": "passed", "reason_codes": [], "next_required_change": next_change}
    return {"status": "passed", "reason_codes": [], "next_required_change": EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE}


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
        **{key: value for key, value in metrics.items() if key != "reason_codes"},
        "source_materialized_coverage_root": config["source_materialized_coverage_root"],
        "source_materialization_status": source["materialization_summary"].get("status"),
        "source_cost_efficient_refinement_detected": source["source_cost_efficient_refinement_detected"],
        "source_refinement_status": source["refinement_summary"].get("status"),
        "candidate_refresh_mode": config["candidate_refresh_mode"],
        "coverage_metric_mode": config["coverage_metric_mode"],
        "include_oracle_baselines": config["include_oracle_baselines"],
        "include_roi_weighted_coverage": config["include_roi_weighted_coverage"],
        "canary_traffic_fraction": config["canary_traffic_fraction"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _roi_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["roi_group"])].append(row)
    families = []
    for group, items in sorted(groups.items()):
        greedy_better = sum(1 for item in items if item["greedy_oracle_delta_vs_incumbent"] > TOLERANCE)
        cost_better = sum(1 for item in items if item["cost_aware_oracle_delta_vs_incumbent"] > TOLERANCE)
        families.append({"roi_group": group, "scenario_count": len(items), "greedy_better_count": greedy_better, "cost_aware_better_count": cost_better, "oracle_better": greedy_better > 0 and cost_better > 0})
    return {"schema_version": "xunce-oracle-separability-roi-summary/v1", "roi_group_count": len(families), "oracle_better_roi_group_count": sum(1 for row in families if row["oracle_better"]), "families": families}


def _manifest(generated_at: str, config_path: Path, output_root: Path, config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-oracle-separability-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "source_materialized_coverage_root": config["source_materialized_coverage_root"],
        "artifacts": {key: str(value) for key, value in paths.items()},
        **_boundary_fields(),
    }


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []


def _oracle_index(candidates: list[dict[str, Any]], *, mode: str) -> int | None:
    scored: list[tuple[float, float, int]] = []
    requires_safe_efficient = mode == "cost_aware" and any("safe_efficient_opportunity" in candidate for candidate in candidates)
    for index, candidate in enumerate(candidates):
        if not _candidate_valid(candidate):
            continue
        if requires_safe_efficient and candidate.get("safe_efficient_opportunity") is not True:
            continue
        coverage = _coverage_value(candidate)
        cost = _path_cost(candidate)
        risk = _risk(candidate)
        if mode == "cost_aware":
            score = _float(candidate.get("cost_efficiency_adjusted_coverage_delta")) if "cost_efficiency_adjusted_coverage_delta" in candidate else coverage / max(cost * (1.0 + risk), TOLERANCE)
        else:
            score = coverage
        scored.append((score, coverage, index))
    if not scored:
        return None
    return max(scored)[2]


def _candidate_at(candidates: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _candidate_valid(candidate: dict[str, Any] | None) -> bool:
    return bool(candidate and candidate.get("reachable") is True and candidate.get("open_grid_fallback_used") is not True)


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


def _selected_index_from_scenario(scenario: dict[str, Any], field: str, *, default: int) -> int:
    value = scenario.get(field, default)
    if isinstance(value, int):
        return value
    return default


def _scenarios_have_safe_efficient_field(scenarios: list[dict[str, Any]]) -> bool:
    return any("safe_efficient_opportunity" in candidate for scenario in scenarios for candidate in _candidate_rows(scenario))


def _safe_efficient_opportunity_count(scenarios: list[dict[str, Any]]) -> int:
    return sum(1 for scenario in scenarios for candidate in _candidate_rows(scenario) if candidate.get("safe_efficient_opportunity") is True)


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


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Oracle Separability Benchmark",
            "",
            f"- status: `{summary['status']}`",
            f"- oracle_separable: `{summary['oracle_separable']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- greedy_oracle_coverage_return_delta_vs_incumbent: `{summary['greedy_oracle_coverage_return_delta_vs_incumbent']}`",
            f"- cost_aware_oracle_coverage_return_delta_vs_incumbent: `{summary['cost_aware_oracle_coverage_return_delta_vs_incumbent']}`",
        ]
    )


def _boundary_fields() -> dict[str, bool]:
    return {field: False for field in BOUNDARY_FIELDS}


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConfigError(f"{field} must be a non-negative number")
    return float(value)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    den = _float(denominator)
    if abs(den) <= TOLERANCE:
        return 0.0
    return _float(numerator) / den


if __name__ == "__main__":
    raise SystemExit(main())
