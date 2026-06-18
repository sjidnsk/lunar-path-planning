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


CONFIG_SCHEMA_VERSION = "xunce-coverage-discriminability-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-coverage-discriminability-audit-summary/v1"
DEFAULT_CONFIG = "configs/xunce_coverage_discriminability_audit_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_coverage_discriminability_audit_v1"

SUMMARY_FILE = "xunce-coverage-discriminability-audit-summary.json"
CANDIDATE_SPREAD_FILE = "xunce-coverage-candidate-spread.jsonl"
ORACLE_BASELINE_FILE = "xunce-coverage-oracle-baseline.jsonl"
ROOT_CAUSE_FILE = "xunce-coverage-root-cause-audit.json"
REPORT_FILE = "xunce-coverage-discriminability-audit-report.md"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
EXPANSION_PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"
COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_EPISODES_FILE = "xunce-exploration-coverage-episodes.jsonl"
COVERAGE_STEPS_FILE = "xunce-exploration-coverage-steps.jsonl"

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
TOLERANCE = 1.0e-12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Xunce coverage discriminability before Stage 18C-v2.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--source-coverage-comparison-root")
    parser.add_argument("--xunce-candidate-checkpoint")
    parser.add_argument("--incumbent-policy-checkpoint")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_roi_expansion_root": args.source_roi_expansion_root,
            "source_coverage_comparison_root": args.source_coverage_comparison_root,
            "xunce_candidate_checkpoint": args.xunce_candidate_checkpoint,
            "incumbent_policy_checkpoint": args.incumbent_policy_checkpoint,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_coverage_discriminability_audit(
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
                "root_cause_route": summary["root_cause_route"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_coverage_discriminability_audit(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    expansion = _load_expansion(config)
    coverage = _load_coverage(config)
    candidate_rows = _candidate_spread_rows(expansion["scenarios"])
    oracle_rows = _oracle_rows(candidate_rows, coverage["episodes"], config)
    metrics = _aggregate_metrics(config, candidate_rows, oracle_rows, coverage["steps"], coverage["summary"])
    root_cause = _root_cause(config, metrics)
    summary = _summary(
        config=config,
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
        expansion=expansion,
        coverage=coverage,
        metrics=metrics,
        root_cause=root_cause,
    )
    root_cause_audit = {
        "schema_version": "xunce-coverage-root-cause-audit/v1",
        "root_cause_route": root_cause["root_cause_route"],
        "next_required_change": root_cause["next_required_change"],
        "reason_codes": root_cause["reason_codes"],
        "interpretation": root_cause["interpretation"],
        **_boundary_fields(),
    }
    paths = _paths(output_root)
    write_jsonl(paths["candidate_spread"], candidate_rows)
    write_jsonl(paths["oracle_baseline"], oracle_rows)
    write_json(paths["root_cause"], root_cause_audit)
    write_json(paths["summary"], summary)
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
    for key in (
        "source_roi_expansion_root",
        "source_coverage_comparison_root",
        "xunce_candidate_checkpoint",
        "incumbent_policy_checkpoint",
    ):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["rollout_steps"] = _positive_int(payload.get("rollout_steps", 10), "rollout_steps")
    normalized["min_candidate_coverage_spread"] = _nonnegative_float(payload.get("min_candidate_coverage_spread", 0.005), "min_candidate_coverage_spread")
    normalized["max_static_candidate_reuse_rate"] = _nonnegative_float(payload.get("max_static_candidate_reuse_rate", 0.5), "max_static_candidate_reuse_rate")
    normalized["min_roi_group_with_nonzero_spread_count"] = _positive_int(payload.get("min_roi_group_with_nonzero_spread_count", 3), "min_roi_group_with_nonzero_spread_count")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_expansion(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["source_roi_expansion_root"])
    reasons: list[str] = []
    summary = _read_json(root / EXPANSION_SUMMARY_FILE, reasons, "missing_roi_expansion_summary")
    slices = _read_jsonl(root / EXPANSION_SLICES_FILE, reasons, "missing_roi_expansion_slices")
    path_feedback = _read_json(root / EXPANSION_PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_path_feedback_audit")
    scenarios = path_feedback.get("scenarios", []) if isinstance(path_feedback.get("scenarios"), list) else []
    return {
        "root": root,
        "summary": summary,
        "slices": slices,
        "path_feedback": path_feedback,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _load_coverage(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["source_coverage_comparison_root"])
    reasons: list[str] = []
    summary = _read_json(root / COVERAGE_SUMMARY_FILE, reasons, "missing_coverage_comparison_summary")
    episodes = _read_jsonl(root / COVERAGE_EPISODES_FILE, reasons, "missing_coverage_episodes")
    steps = _read_jsonl(root / COVERAGE_STEPS_FILE, reasons, "missing_coverage_steps")
    return {
        "root": root,
        "summary": summary,
        "episodes": episodes,
        "steps": steps,
        "reason_codes": unique_sorted(reasons),
    }


def _candidate_spread_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id") or "")
        roi_group = str(scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        candidates = _candidate_rows(scenario)
        coverage_values = [_coverage_value(candidate) for candidate in candidates if _candidate_valid(candidate)]
        path_line_values = [_path_line_coverage_value(candidate) for candidate in candidates if _candidate_valid(candidate)]
        cells = [_cell_key(candidate.get("cell") or candidate.get("candidate_cell")) for candidate in candidates]
        rows.append(
            {
                "schema_version": "xunce-coverage-candidate-spread-row/v1",
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "candidate_count": len(candidates),
                "reachable_candidate_count": sum(1 for candidate in candidates if _candidate_valid(candidate)),
                "nonzero_coverage_candidate_count": sum(1 for value in coverage_values if value > TOLERANCE),
                "candidate_coverage_spread_range": _spread_range(coverage_values),
                "candidate_coverage_spread_std": _std(coverage_values),
                "candidate_coverage_gini": _gini(coverage_values),
                "path_line_coverage_spread_range": _spread_range(path_line_values),
                "candidate_pareto_frontier_count": _pareto_frontier_count(candidates),
                "unique_candidate_cell_ratio": _unique_ratio(cells),
                "path_cost_coverage_tradeoff_count": _tradeoff_count(candidates, "path_cost"),
                "risk_coverage_tradeoff_count": _tradeoff_count(candidates, "risk"),
                "coverage_endpoint_tie_observed": bool(scenario.get("coverage_endpoint_tie_observed")),
                "max_candidate_coverage": max(coverage_values) if coverage_values else 0.0,
            }
        )
    return rows


def _oracle_rows(
    candidate_rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    episode_by_key = {(str(row.get("scenario_id")), str(row.get("policy"))): row for row in episodes}
    rows = []
    for row in candidate_rows[: config["required_scenario_count"]]:
        scenario_id = str(row["scenario_id"])
        xunce = episode_by_key.get((scenario_id, "xunce"), {})
        incumbent = episode_by_key.get((scenario_id, "incumbent"), {})
        oracle_return = float(row["max_candidate_coverage"]) * float(config["rollout_steps"])
        xunce_return = _float(xunce.get("coverage_return"))
        incumbent_return = _float(incumbent.get("coverage_return"))
        rows.append(
            {
                "schema_version": "xunce-coverage-oracle-baseline-row/v1",
                "scenario_id": scenario_id,
                "roi_group": row["roi_group"],
                "oracle_coverage_return": oracle_return,
                "xunce_coverage_return": xunce_return,
                "incumbent_coverage_return": incumbent_return,
                "oracle_vs_xunce_coverage_delta": oracle_return - xunce_return,
                "oracle_vs_incumbent_coverage_delta": oracle_return - incumbent_return,
                "xunce_oracle_regret": max(0.0, oracle_return - xunce_return),
                "incumbent_oracle_regret": max(0.0, oracle_return - incumbent_return),
            }
        )
    return rows


def _aggregate_metrics(
    config: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    coverage_summary: dict[str, Any],
) -> dict[str, Any]:
    static_reuse_rate = _static_candidate_reuse_rate(steps)
    roi_spread = defaultdict(float)
    for row in candidate_rows:
        roi_spread[str(row["roi_group"])] = max(roi_spread[str(row["roi_group"])], float(row["candidate_coverage_spread_range"]))
    metrics = {
        "candidate_coverage_spread_range": _mean([row["candidate_coverage_spread_range"] for row in candidate_rows]),
        "candidate_coverage_spread_std": _mean([row["candidate_coverage_spread_std"] for row in candidate_rows]),
        "candidate_coverage_gini": _mean([row["candidate_coverage_gini"] for row in candidate_rows]),
        "candidate_pareto_frontier_count": _mean([row["candidate_pareto_frontier_count"] for row in candidate_rows]),
        "reachable_candidate_count": sum(int(row["reachable_candidate_count"]) for row in candidate_rows),
        "unique_candidate_cell_ratio": _mean([row["unique_candidate_cell_ratio"] for row in candidate_rows]),
        "static_candidate_reuse_rate": static_reuse_rate,
        "nonzero_coverage_candidate_count": sum(int(row["nonzero_coverage_candidate_count"]) for row in candidate_rows),
        "oracle_coverage_return": _mean([row["oracle_coverage_return"] for row in oracle_rows]),
        "oracle_vs_incumbent_coverage_delta": _mean([row["oracle_vs_incumbent_coverage_delta"] for row in oracle_rows]),
        "oracle_vs_xunce_coverage_delta": _mean([row["oracle_vs_xunce_coverage_delta"] for row in oracle_rows]),
        "xunce_oracle_regret": _mean([row["xunce_oracle_regret"] for row in oracle_rows]),
        "incumbent_oracle_regret": _mean([row["incumbent_oracle_regret"] for row in oracle_rows]),
        "useful_disagreement_opportunity_count": _useful_disagreement_opportunity_count(candidate_rows),
        "path_cost_coverage_tradeoff_count": sum(int(row["path_cost_coverage_tradeoff_count"]) for row in candidate_rows),
        "risk_coverage_tradeoff_count": sum(int(row["risk_coverage_tradeoff_count"]) for row in candidate_rows),
        "roi_group_with_nonzero_spread_count": sum(1 for value in roi_spread.values() if value > TOLERANCE),
        "path_line_coverage_spread_range": _mean([row["path_line_coverage_spread_range"] for row in candidate_rows]),
        "endpoint_tie_observed_count": sum(1 for row in candidate_rows if row["coverage_endpoint_tie_observed"]),
        "xunce_coverage_return_delta_vs_incumbent": _float(coverage_summary.get("xunce_coverage_return_delta_vs_incumbent")),
        "policy_disagreement_count": _int(coverage_summary.get("policy_disagreement_count")),
    }
    metrics["evaluation_task_discriminative"] = bool(
        metrics["oracle_vs_incumbent_coverage_delta"] > TOLERANCE
        and metrics["useful_disagreement_opportunity_count"] > 0
        and metrics["roi_group_with_nonzero_spread_count"] >= int(config["min_roi_group_with_nonzero_spread_count"])
        and metrics["candidate_pareto_frontier_count"] > 1.0
        and metrics["static_candidate_reuse_rate"] < float(config["max_static_candidate_reuse_rate"])
    )
    return metrics


def _root_cause(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    route = "ready_for_stage18c_v2_dynamic_rollout"
    next_change = "ready_for_stage18c_v2_dynamic_rollout"
    interpretation = "The coverage comparison has enough spread to proceed to dynamic rollout."
    if metrics["candidate_coverage_spread_range"] < float(config["min_candidate_coverage_spread"]):
        route = "candidate_coverage_spread_insufficient"
        next_change = "repair_coverage_evaluation_discriminability"
        interpretation = "Candidate-level coverage spread is too small to distinguish exploration choices."
    elif metrics["static_candidate_reuse_rate"] >= float(config["max_static_candidate_reuse_rate"]):
        route = "coverage_task_not_discriminative_static_candidate_reuse"
        next_change = "repair_coverage_evaluation_discriminability"
        interpretation = "The same candidate sets are reused across rollout steps, so policy disagreement does not become new coverage."
    elif metrics["endpoint_tie_observed_count"] > 0 and metrics["path_line_coverage_spread_range"] > metrics["candidate_coverage_spread_range"] + TOLERANCE:
        route = "coverage_metric_too_coarse"
        next_change = "repair_coverage_evaluation_discriminability"
        interpretation = "Endpoint footprint coverage ties while path-line coverage has spread."
    elif metrics["oracle_vs_incumbent_coverage_delta"] <= TOLERANCE:
        route = "map_or_roi_complexity_insufficient"
        next_change = "expand_roi_or_map_complexity"
        interpretation = "Even a greedy coverage oracle cannot beat the incumbent on these candidates."
    elif abs(metrics["xunce_coverage_return_delta_vs_incumbent"]) <= TOLERANCE:
        route = "models_genuinely_no_coverage_advantage"
        next_change = "xunce_training_or_adapter_iteration_required"
        interpretation = "Oracle coverage exists, but the learned policies do not exploit it."
    elif not metrics["evaluation_task_discriminative"]:
        route = "candidate_materialization_insufficient"
        next_change = "repair_coverage_evaluation_discriminability"
        interpretation = "The task still lacks enough useful disagreement or ROI spread."
    reasons.append(route)
    return {
        "root_cause_route": route,
        "next_required_change": next_change,
        "reason_codes": unique_sorted(reasons),
        "interpretation": interpretation,
    }


def _summary(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    expansion: dict[str, Any],
    coverage: dict[str, Any],
    metrics: dict[str, Any],
    root_cause: dict[str, Any],
) -> dict[str, Any]:
    paths = _paths(output_root)
    reason_codes = unique_sorted(expansion["reason_codes"] + coverage["reason_codes"] + root_cause["reason_codes"])
    status = "failed" if expansion["reason_codes"] or coverage["reason_codes"] else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "root_cause_route": root_cause["root_cause_route"],
        "next_required_change": root_cause["next_required_change"],
        "source_roi_expansion_root": str(expansion["root"]),
        "source_coverage_comparison_root": str(coverage["root"]),
        "xunce_checkpoint_present": Path(config["xunce_candidate_checkpoint"]).is_file(),
        "incumbent_checkpoint_present": Path(config["incumbent_policy_checkpoint"]).is_file(),
        "summary": str(paths["summary"]),
        "candidate_spread": str(paths["candidate_spread"]),
        "oracle_baseline": str(paths["oracle_baseline"]),
        "root_cause_audit": str(paths["root_cause"]),
        "report": str(paths["report"]),
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **metrics,
        **_boundary_fields(),
    }


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return [candidate for candidate in candidates if isinstance(candidate, dict)] if isinstance(candidates, list) else []


def _candidate_valid(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("reachable", True) is True and _float_or_none(candidate.get("path_cost")) is not None)


def _coverage_value(candidate: dict[str, Any]) -> float:
    for field in ("expected_coverage_rate_delta", "coverage_rate_delta", "expected_new_coverage_area"):
        value = _float_or_none(candidate.get(field))
        if value is not None:
            return value if field != "expected_new_coverage_area" else value / 1000.0
    return 0.0


def _path_line_coverage_value(candidate: dict[str, Any]) -> float:
    value = _float_or_none(candidate.get("expected_path_line_coverage_delta"))
    return _coverage_value(candidate) if value is None else value


def _pareto_frontier_count(candidates: list[dict[str, Any]]) -> int:
    valid = [candidate for candidate in candidates if _candidate_valid(candidate)]
    frontier = 0
    for candidate in valid:
        coverage = _coverage_value(candidate)
        cost = _float(candidate.get("path_cost"))
        risk = _float(candidate.get("risk"))
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            other_coverage = _coverage_value(other)
            other_cost = _float(other.get("path_cost"))
            other_risk = _float(other.get("risk"))
            if (
                other_coverage >= coverage - TOLERANCE
                and other_cost <= cost + TOLERANCE
                and other_risk <= risk + TOLERANCE
                and (other_coverage > coverage + TOLERANCE or other_cost < cost - TOLERANCE or other_risk < risk - TOLERANCE)
            ):
                dominated = True
                break
        if not dominated:
            frontier += 1
    return frontier


def _tradeoff_count(candidates: list[dict[str, Any]], cost_field: str) -> int:
    valid = [candidate for candidate in candidates if _candidate_valid(candidate)]
    count = 0
    for left in valid:
        for right in valid:
            if left is right:
                continue
            if _coverage_value(left) > _coverage_value(right) + TOLERANCE and _float(left.get(cost_field)) > _float(right.get(cost_field)) + TOLERANCE:
                count += 1
                break
    return count


def _useful_disagreement_opportunity_count(candidate_rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in candidate_rows
        if row["candidate_pareto_frontier_count"] > 1
        and row["candidate_coverage_spread_range"] > TOLERANCE
        and row["nonzero_coverage_candidate_count"] > 1
    )


def _static_candidate_reuse_rate(steps: list[dict[str, Any]]) -> float:
    grouped: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for row in steps:
        candidate_cells = row.get("candidate_cells")
        if not isinstance(candidate_cells, list):
            continue
        grouped[(str(row.get("scenario_id")), str(row.get("policy")))].append(tuple(str(cell) for cell in candidate_cells))
    rates = []
    for values in grouped.values():
        if not values:
            continue
        rates.append(1.0 - (len(set(values)) / max(1, len(values))))
    return _mean(rates)


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "candidate_spread": output_root / CANDIDATE_SPREAD_FILE,
        "oracle_baseline": output_root / ORACLE_BASELINE_FILE,
        "root_cause": output_root / ROOT_CAUSE_FILE,
        "report": output_root / REPORT_FILE,
    }


def _read_json(path: Path, reasons: list[str], reason: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(reason)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(reason)
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], reason: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(reason)
        return []
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    except json.JSONDecodeError:
        reasons.append(reason)
        return []
    return rows


def _cell_key(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{value[0]},{value[1]}"
    return str(value)


def _unique_ratio(values: list[Any]) -> float:
    if not values:
        return 0.0
    return len(set(values)) / len(values)


def _spread_range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(max(0.0, value) for value in values)
    total = sum(sorted_values)
    if total <= TOLERANCE:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(sorted_values))
    return (2.0 * weighted) / (len(sorted_values) * total) - (len(sorted_values) + 1.0) / len(sorted_values)


def _mean(values: list[Any]) -> float:
    numeric = [float(value) for value in values if _float_or_none(value) is not None]
    return statistics.mean(numeric) if numeric else 0.0


def _float(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise ConfigError(f"{name} must be a non-negative number")
    return float(value)


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"canary_traffic_fraction": 0.0})
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Coverage Discriminability Audit v1",
            "",
            f"- status: `{summary['status']}`",
            f"- root_cause_route: `{summary['root_cause_route']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- candidate_coverage_spread_range: `{summary['candidate_coverage_spread_range']}`",
            f"- static_candidate_reuse_rate: `{summary['static_candidate_reuse_rate']}`",
            f"- oracle_vs_incumbent_coverage_delta: `{summary['oracle_vs_incumbent_coverage_delta']}`",
            "",
            "This is a read-only audit. It does not run PPO, publish checkpoints, replace the default policy, connect a real executor, or start online canary traffic.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
