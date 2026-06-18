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


CONFIG_SCHEMA_VERSION = "xunce-safe-efficient-opportunity-root-cause-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-safe-efficient-opportunity-root-cause-audit-summary/v1"
DEFAULT_CONFIG = "configs/xunce_safe_efficient_opportunity_root_cause_audit_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_safe_efficient_opportunity_root_cause_audit_v1"

BINDING_SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
COST_REFINEMENT_SUMMARY_FILE = "xunce-cost-efficient-coverage-opportunity-summary.json"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

SUMMARY_FILE = "xunce-safe-efficient-opportunity-root-cause-audit-summary.json"
SCENARIOS_FILE = "xunce-safe-efficient-opportunity-root-cause-scenarios.jsonl"
ROI_FILE = "xunce-safe-efficient-opportunity-root-cause-roi-summary.json"
AUDIT_FILE = "xunce-safe-efficient-opportunity-root-cause-audit.json"
REPORT_FILE = "xunce-safe-efficient-opportunity-root-cause-report.md"

PASS_NEXT_REQUIRED_CHANGE = "run_safe_efficient_candidate_repair"
FIX_BINDING_NEXT_REQUIRED_CHANGE = "run_true_incumbent_selection_binding"
EXPAND_FRONTIER_NEXT_REQUIRED_CHANGE = "expand_candidate_generation_near_frontier"
REFINE_COST_NEXT_REQUIRED_CHANGE = "refine_cost_guard_or_candidate_spacing"
REFINE_RISK_NEXT_REQUIRED_CHANGE = "refine_risk_guard_or_roi_weighting"
REPAIR_GENERATION_NEXT_REQUIRED_CHANGE = "repair_cost_efficient_candidate_generation"
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
    parser = argparse.ArgumentParser(description="Audit root causes for missing safe-efficient Xunce coverage opportunities.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-bound-coverage-root")
    parser.add_argument("--source-cost-efficient-refinement-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_bound_coverage_root": args.source_bound_coverage_root,
            "source_cost_efficient_refinement_root": args.source_cost_efficient_refinement_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_safe_efficient_opportunity_root_cause_audit(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "root_cause_route": summary["root_cause_route"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_safe_efficient_opportunity_root_cause_audit(
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
    scenario_rows, roi_summary, metrics = _audit_metrics(config, source)
    decision = _decision(config, source, metrics)
    paths = _paths(output_root)
    generated_at = utc_now()
    summary = _summary(generated_at, config, config_path, output_root, repo_root, source, metrics, decision, paths)
    write_json(paths["summary"], summary)
    write_jsonl(paths["scenarios"], scenario_rows)
    write_json(paths["roi"], roi_summary)
    write_json(paths["audit"], {"schema_version": "xunce-safe-efficient-opportunity-root-cause-audit/v1", **decision, **metrics, **_boundary_fields()})
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
    for key in ("source_bound_coverage_root", "source_cost_efficient_refinement_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["min_roi_group_count"] = _positive_int(payload.get("min_roi_group_count", 8), "min_roi_group_count")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    bound_root = Path(config["source_bound_coverage_root"])
    refinement_root = Path(config["source_cost_efficient_refinement_root"])
    reasons: list[str] = []
    binding_summary = _read_json(bound_root / BINDING_SUMMARY_FILE, reasons, "true_incumbent_binding_missing")
    path_feedback = _read_json(bound_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "true_incumbent_binding_missing")
    cost_refinement_summary = _read_json(refinement_root / COST_REFINEMENT_SUMMARY_FILE, [], "missing_cost_efficient_refinement")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    return {
        "bound_root": bound_root,
        "refinement_root": refinement_root,
        "binding_summary": binding_summary,
        "cost_refinement_summary": cost_refinement_summary,
        "path_feedback": path_feedback,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "scenarios": output_root / SCENARIOS_FILE,
        "roi": output_root / ROI_FILE,
        "audit": output_root / AUDIT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _audit_metrics(config: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scenario_rows: list[dict[str, Any]] = []
    roi_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals = defaultdict(int)
    cost_spreads: list[float] = []
    risk_spreads: list[float] = []
    coverage_cost_tradeoff_count = 0
    fallback_count = 0

    for scenario_index, scenario in enumerate(source["scenarios"][: config["required_scenario_count"]]):
        scenario_id = str(scenario.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        candidates = _candidate_rows(scenario)
        incumbent_index = _selected_index(scenario.get("incumbent_selected_action_index"))
        incumbent_source = str(scenario.get("incumbent_selection_source") or "")
        if incumbent_source == "fallback_action_index_0":
            fallback_count += 1
        incumbent = _candidate_at(candidates, incumbent_index)
        incumbent_coverage = _coverage(incumbent)
        incumbent_cost = _path_cost(incumbent)
        incumbent_risk = _risk(incumbent)
        incumbent_efficiency = _cost_efficiency(incumbent)
        incumbent_revisit = _revisit(incumbent)
        path_cost_values = [_path_cost(candidate) for candidate in candidates if _candidate_valid(candidate)]
        risk_values = [_risk(candidate) for candidate in candidates if _candidate_valid(candidate)]
        cost_spread = _spread(path_cost_values)
        risk_spread = _spread(risk_values)
        cost_spreads.append(cost_spread)
        risk_spreads.append(risk_spread)

        row = {
            "schema_version": "xunce-safe-efficient-opportunity-root-cause-scenario/v1",
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "incumbent_selected_action_index": incumbent_index,
            "incumbent_selection_source": incumbent_source,
            "coverage_positive_candidate_count": 0,
            "coverage_positive_but_cost_regressive_count": 0,
            "coverage_positive_but_risk_regressive_count": 0,
            "coverage_positive_but_efficiency_regressive_count": 0,
            "coverage_positive_but_revisit_regressive_count": 0,
            "incumbent_dominates_candidate_count": 0,
            "candidate_dominates_incumbent_count": 0,
            "candidate_path_cost_spread_range": cost_spread,
            "candidate_risk_spread_range": risk_spread,
            "safe_efficient_opportunity_count": 0,
        }
        for candidate_index, candidate in enumerate(candidates):
            if not _candidate_valid(candidate) or candidate_index == incumbent_index:
                continue
            coverage_positive = _coverage(candidate) > incumbent_coverage + TOLERANCE
            cost_regressive = _path_cost(candidate) > incumbent_cost + TOLERANCE
            risk_regressive = _risk(candidate) > incumbent_risk + TOLERANCE
            efficiency_regressive = _cost_efficiency(candidate) <= incumbent_efficiency + TOLERANCE
            revisit_regressive = _revisit(candidate) > incumbent_revisit + TOLERANCE
            if coverage_positive:
                row["coverage_positive_candidate_count"] += 1
                totals["coverage_positive_candidate_count"] += 1
                if cost_regressive:
                    row["coverage_positive_but_cost_regressive_count"] += 1
                    totals["coverage_positive_but_cost_regressive_count"] += 1
                if risk_regressive:
                    row["coverage_positive_but_risk_regressive_count"] += 1
                    totals["coverage_positive_but_risk_regressive_count"] += 1
                if efficiency_regressive:
                    row["coverage_positive_but_efficiency_regressive_count"] += 1
                    totals["coverage_positive_but_efficiency_regressive_count"] += 1
                if revisit_regressive:
                    row["coverage_positive_but_revisit_regressive_count"] += 1
                    totals["coverage_positive_but_revisit_regressive_count"] += 1
                if cost_regressive or risk_regressive:
                    coverage_cost_tradeoff_count += 1
            if _dominates(incumbent, candidate):
                row["incumbent_dominates_candidate_count"] += 1
                totals["incumbent_dominates_candidate_count"] += 1
            if _dominates(candidate, incumbent):
                row["candidate_dominates_incumbent_count"] += 1
                totals["candidate_dominates_incumbent_count"] += 1
            if _safe_efficient(candidate, incumbent):
                row["safe_efficient_opportunity_count"] += 1
                totals["safe_efficient_opportunity_count"] += 1
        scenario_rows.append(row)
        roi_groups[roi_group].append(row)

    roi_rows = []
    for roi_group, rows in sorted(roi_groups.items()):
        safe = sum(row["safe_efficient_opportunity_count"] for row in rows)
        roi_rows.append(
            {
                "roi_group": roi_group,
                "scenario_count": len(rows),
                "safe_efficient_opportunity_count": safe,
                "coverage_positive_candidate_count": sum(row["coverage_positive_candidate_count"] for row in rows),
                "has_safe_efficient_opportunity": safe > 0,
            }
        )
    roi_group_without_safe = sum(1 for row in roi_rows if not row["has_safe_efficient_opportunity"])
    coverage_positive = totals["coverage_positive_candidate_count"]
    metrics = {
        "reason_codes": unique_sorted(source["reason_codes"]),
        "scenario_count": len(scenario_rows),
        "roi_group_count": len(roi_rows),
        "coverage_positive_candidate_count": coverage_positive,
        "coverage_positive_but_cost_regressive_count": totals["coverage_positive_but_cost_regressive_count"],
        "coverage_positive_but_risk_regressive_count": totals["coverage_positive_but_risk_regressive_count"],
        "coverage_positive_but_efficiency_regressive_count": totals["coverage_positive_but_efficiency_regressive_count"],
        "coverage_positive_but_revisit_regressive_count": totals["coverage_positive_but_revisit_regressive_count"],
        "incumbent_dominates_candidate_count": totals["incumbent_dominates_candidate_count"],
        "candidate_dominates_incumbent_count": totals["candidate_dominates_incumbent_count"],
        "candidate_path_cost_spread_range": _mean(cost_spreads),
        "candidate_risk_spread_range": _mean(risk_spreads),
        "candidate_coverage_cost_tradeoff_count": coverage_cost_tradeoff_count,
        "roi_group_without_safe_efficient_opportunity_count": roi_group_without_safe,
        "candidate_generation_cost_bias_score": _ratio(totals["coverage_positive_but_cost_regressive_count"], coverage_positive),
        "candidate_generation_risk_bias_score": _ratio(totals["coverage_positive_but_risk_regressive_count"], coverage_positive),
        "safe_efficient_opportunity_count": totals["safe_efficient_opportunity_count"],
        "fallback_action_index_0_count": fallback_count,
    }
    roi_summary = {
        "schema_version": "xunce-safe-efficient-opportunity-root-cause-roi-summary/v1",
        "roi_group_count": len(roi_rows),
        "roi_group_without_safe_efficient_opportunity_count": roi_group_without_safe,
        "families": roi_rows,
    }
    return scenario_rows, roi_summary, metrics


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    if metrics["fallback_action_index_0_count"] > 0 or source["binding_summary"].get("true_incumbent_selection_bound") is not True:
        reasons.append("true_incumbent_binding_missing")
    if metrics["roi_group_count"] < int(config["min_roi_group_count"]):
        reasons.append("roi_or_map_complexity_insufficient")
    reasons = unique_sorted(reasons)
    if reasons:
        if "true_incumbent_binding_missing" in reasons:
            route = FIX_BINDING_NEXT_REQUIRED_CHANGE
        elif "roi_or_map_complexity_insufficient" in reasons:
            route = EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE
        else:
            route = FIX_BINDING_NEXT_REQUIRED_CHANGE
        return {"status": "failed", "reason_codes": reasons, "root_cause_route": reasons[0], "next_required_change": route}
    if metrics["safe_efficient_opportunity_count"] > 0:
        return {"status": "passed", "reason_codes": [], "root_cause_route": "ready_for_safe_efficient_candidate_repair", "next_required_change": PASS_NEXT_REQUIRED_CHANGE}
    coverage_positive = metrics["coverage_positive_candidate_count"]
    if coverage_positive == 0 or metrics["candidate_path_cost_spread_range"] <= TOLERANCE:
        root = "roi_or_map_complexity_insufficient"
        route = EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE
    elif metrics["incumbent_dominates_candidate_count"] >= max(1, coverage_positive):
        root = "incumbent_dominates_candidate_space"
        route = EXPAND_FRONTIER_NEXT_REQUIRED_CHANGE
    elif metrics["candidate_generation_cost_bias_score"] >= 0.5:
        root = "candidate_generation_cost_biased"
        route = REPAIR_GENERATION_NEXT_REQUIRED_CHANGE
    elif metrics["coverage_positive_but_cost_regressive_count"] >= metrics["coverage_positive_but_risk_regressive_count"]:
        root = "coverage_cost_tradeoff_too_strict"
        route = REFINE_COST_NEXT_REQUIRED_CHANGE
    elif metrics["candidate_generation_risk_bias_score"] >= 0.5:
        root = "risk_guard_too_strict"
        route = REFINE_RISK_NEXT_REQUIRED_CHANGE
    else:
        root = "ready_for_safe_efficient_candidate_repair"
        route = PASS_NEXT_REQUIRED_CHANGE
    return {"status": "passed", "reason_codes": [], "root_cause_route": root, "next_required_change": route}


def _summary(generated_at: str, config: dict[str, Any], config_path: Path, output_root: Path, repo_root: Path, source: dict[str, Any], metrics: dict[str, Any], decision: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "root_cause_route": decision["root_cause_route"],
        "next_required_change": decision["next_required_change"],
        **{key: value for key, value in metrics.items() if key != "reason_codes"},
        "source_bound_coverage_root": config["source_bound_coverage_root"],
        "source_cost_efficient_refinement_root": config["source_cost_efficient_refinement_root"],
        "canary_traffic_fraction": config["canary_traffic_fraction"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []


def _candidate_valid(candidate: dict[str, Any] | None) -> bool:
    return bool(candidate and candidate.get("reachable") is True and candidate.get("open_grid_fallback_used") is not True)


def _selected_index(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _candidate_at(candidates: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _coverage(candidate: dict[str, Any] | None) -> float:
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


def _revisit(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("revisit_penalty")) if candidate else 0.0


def _cost_efficiency(candidate: dict[str, Any] | None) -> float:
    return _coverage(candidate) / max(_path_cost(candidate) * (1.0 + _risk(candidate)), TOLERANCE)


def _dominates(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not _candidate_valid(left) or not _candidate_valid(right):
        return False
    weak = _coverage(left) >= _coverage(right) - TOLERANCE and _path_cost(left) <= _path_cost(right) + TOLERANCE and _risk(left) <= _risk(right) + TOLERANCE
    strict = _coverage(left) > _coverage(right) + TOLERANCE or _path_cost(left) < _path_cost(right) - TOLERANCE or _risk(left) < _risk(right) - TOLERANCE
    return bool(weak and strict)


def _safe_efficient(candidate: dict[str, Any] | None, incumbent: dict[str, Any] | None) -> bool:
    return bool(
        _candidate_valid(candidate)
        and _coverage(candidate) > _coverage(incumbent) + TOLERANCE
        and _path_cost(candidate) <= _path_cost(incumbent) + TOLERANCE
        and _risk(candidate) <= _risk(incumbent) + TOLERANCE
        and _cost_efficiency(candidate) > _cost_efficiency(incumbent) + TOLERANCE
    )


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
            "# Xunce Safe-Efficient Opportunity Root Cause Audit",
            "",
            f"- status: `{summary['status']}`",
            f"- root_cause_route: `{summary['root_cause_route']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- coverage_positive_candidate_count: `{summary['coverage_positive_candidate_count']}`",
            f"- safe_efficient_opportunity_count: `{summary['safe_efficient_opportunity_count']}`",
            f"- candidate_generation_cost_bias_score: `{summary['candidate_generation_cost_bias_score']}`",
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


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
