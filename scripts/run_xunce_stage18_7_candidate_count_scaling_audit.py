from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODEL_EXPLORER_SRC = SCRIPT_DIR.parents[0] / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
from global_99_governance_common import global_99_boundary_defaults
from platform_command import display_command, python_script_command
from xunce_stage18_guard_thresholds import stage18_guard_thresholds
from xunce_candidate_guard_semantics import (
    CANDIDATE_GUARD_MODE,
    candidate_guard_clean,
    candidate_guard_observation,
    paired_decision_key,
    risk_proxy_value,
    selected_baseline_candidate,
)

from model_explorer.policy.canonical_reward import load_canonical_reward_profile


CONFIG_SCHEMA_VERSION = "xunce-stage18-7-candidate-count-scaling-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-7-candidate-count-scaling-summary/v1"
RESULT_ROW_SCHEMA_VERSION = "xunce-stage18-7-candidate-count-scaling-result/v1"
COMMAND_PLAN_SCHEMA_VERSION = "xunce-stage18-7-candidate-count-scaling-command-plan/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage18-7-manifest/v1"
STAGE18_6_SUMMARY_SCHEMA_VERSION = "xunce-stage18-6-guard-refinement-summary/v1"

DEFAULT_CONFIG = "configs/xunce_stage18_7_candidate_count_scaling_audit_v1.json"
DEFAULT_ARTIFACT_WORKSPACE = "D:/CodexDownloads/lunar-path-planning/stage18_7_candidate_count_scaling_audit"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage18_7_candidate_count_scaling_audit/"
    "outputs/path_feedback_batch_xunce_stage18_7_candidate_count_scaling_audit_v1"
)
DEFAULT_CANONICAL_PROFILE = "configs/xunce_canonical_reward_guard_profile_v2.json"
DEFAULT_SWEEPS = ((6, 48), (12, 96), (24, 192), (36, 288))

SUMMARY_FILE = "xunce-stage18-7-candidate-count-scaling-summary.json"
RESULTS_FILE = "xunce-stage18-7-candidate-count-scaling-results.jsonl"
COMMAND_PLAN_FILE = "xunce-stage18-7-candidate-count-scaling-command-plan.json"
REPORT_FILE = "xunce-stage18-7-candidate-count-scaling-report.md"
MANIFEST_FILE = "xunce-stage18-7-manifest.json"

COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_AGGREGATE_FILE = "xunce-exploration-coverage-comparison-aggregate.json"
COVERAGE_PAIRS_FILE = "xunce-exploration-coverage-comparison-pairs.jsonl"
PAIRED_DECISION_AUDIT_FILE = "xunce-exploration-coverage-paired-decision-audit.jsonl"
CANDIDATE_METRIC_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"
STAGE18_6_SUMMARY_FILE = "xunce-stage18-6-guard-refinement-summary.json"

ROUTE_MISSING_SWEEPS = "run_missing_candidate_count_sweeps_with_metric_audit"
ROUTE_REPAIR_LINEAGE = "repair_stage18_7_lineage_or_config_drift"
ROUTE_BOUNDED_BUDGET = "continue_candidate_count_scaling_with_bounded_budget"
ROUTE_EXPAND_CANDIDATES = "expand_candidate_generation_roi_complexity"
ROUTE_REFINE_REWARD_GUARD = "refine_coverage_reward_and_cost_guard"
ROUTE_STAGE19_PREFLIGHT = "prepare_stage19_evaluator_critic_preflight"
ROUTE_BOUNDARY_REPAIR = "resolve_stage18_7_candidate_count_scaling_boundary_rejections"

ALLOWED_STAGE18_6_ROUTES = {
    "rerun_stage18_4e_with_candidate_metric_audit",
    "rerun_xunce_stage18_5_evidence_attribution_review",
    "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts",
    "resolve_stage18_6_guard_refinement_boundary_rejections",
    ROUTE_EXPAND_CANDIDATES,
    ROUTE_REFINE_REWARD_GUARD,
    ROUTE_STAGE19_PREFLIGHT,
}

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "default_policy_replacement_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
)

ONLY_COUNT_FIELDS = {
    "dynamic_max_candidates_per_step",
    "dynamic_proposal_pool_limit_per_step",
}
OUTPUT_LOCAL_CONFIG_FIELDS = {
    "output_root",
    "dynamic_validation_work_root",
}

TOLERANCE = 1.0e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Xunce Stage 18.7 candidate-count scaling audit artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_root = resolve_path(Path(args.output_root), repo_root).resolve()
    try:
        summary = run_xunce_stage18_7_candidate_count_scaling_audit(
            config_path=resolve_path(Path(args.config), repo_root).resolve(),
            output_root=output_root,
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "stage19_authorized": summary["stage19_authorized"],
                "summary": str(output_root / SUMMARY_FILE),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage18_7_candidate_count_scaling_audit(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config = _load_config(config_path, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)

    rows = [_evaluate_sweep(sweep, config) for sweep in config["sweeps"]]
    rows = _apply_cross_sweep_config_drift(rows)
    command_plan = _command_plan(config, repo_root, output_root)
    blocking = _blocking_reasons(config, rows)
    diagnostic = _diagnostic_reasons(rows)
    route = _route(blocking=blocking, diagnostic=diagnostic, rows=rows)
    generated_at = utc_now()

    complete_rows = [row for row in rows if row["sweep_complete"]]
    guard_clean_rates = [row["guard_clean_candidate_available_rate"] for row in complete_rows]
    guard_clean_advantage_rates = [
        row["guard_clean_advantage_candidate_available_rate"] for row in complete_rows
    ]
    xunce_rates = [row["xunce_selected_guard_clean_rate"] for row in complete_rows]
    xunce_advantage_rates = [row["xunce_selected_guard_clean_advantage_rate"] for row in complete_rows]
    incumbent_rates = [row["incumbent_selected_guard_clean_rate"] for row in complete_rows]
    incumbent_advantage_rates = [row["incumbent_selected_guard_clean_advantage_rate"] for row in complete_rows]
    guard_refinement_passed_count = sum(1 for row in complete_rows if row["guard_refinement_passed"] is True)
    same_advantage_count = sum(
        1 for row in complete_rows if row["same_candidate_set_guard_clean_advantage_established"] is True
    )

    routing = {
        "schema_version": "xunce-stage18-7-next-stage-routing/v1",
        "primary_route": route,
        "stage19_authorized": False,
        "stage19_readiness": "ready_for_stage19_preflight_human_review_only"
        if route == ROUTE_STAGE19_PREFLIGHT
        else "not_authorized",
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "failed" if blocking else "passed",
        "reason_codes": unique_sorted([*blocking, *diagnostic]),
        "blocking_reason_codes": blocking,
        "diagnostic_reason_codes": diagnostic,
        "candidate_count_sweep": [
            {"candidate_count": count, "proposal_pool_limit": pool} for count, pool in DEFAULT_SWEEPS
        ],
        "candidate_count_results": rows,
        "sweep_complete_count": len(complete_rows),
        "stage18_6_guard_refinement_passed_count": guard_refinement_passed_count,
        "same_candidate_set_guard_clean_advantage_established_count": same_advantage_count,
        "best_guard_clean_candidate_available_rate": max(guard_clean_rates) if guard_clean_rates else 0.0,
        "best_guard_clean_advantage_candidate_available_rate": max(guard_clean_advantage_rates)
        if guard_clean_advantage_rates
        else 0.0,
        "best_xunce_selected_guard_clean_rate": max(xunce_rates) if xunce_rates else 0.0,
        "best_xunce_selected_guard_clean_advantage_rate": max(xunce_advantage_rates) if xunce_advantage_rates else 0.0,
        "best_incumbent_selected_guard_clean_rate": max(incumbent_rates) if incumbent_rates else 0.0,
        "best_incumbent_selected_guard_clean_advantage_rate": max(incumbent_advantage_rates)
        if incumbent_advantage_rates
        else 0.0,
        "next_stage_routing": routing,
        "next_required_change": route,
        "stage19_authorized": False,
        "stage19_readiness": {
            "readiness": routing["stage19_readiness"],
            "authorized": False,
            "stage18_6_guard_refinement_passed_count": guard_refinement_passed_count,
            "same_candidate_set_guard_clean_advantage_established_count": same_advantage_count,
        },
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
        "canonical_reward_profile": str(config["canonical_reward_profile"]),
        "artifact_workspace_root": str(config["artifact_workspace_root"]),
        "summary": str(paths["summary"]),
        "results": str(paths["results"]),
        "command_plan": str(paths["command_plan"]),
        "report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
        "config": str(config_path),
        "output_root": str(output_root),
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "modifies_network": False,
        "modifies_action_space": False,
        "modifies_default_astar": False,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary_status": summary["status"],
        "next_required_change": route,
        "stage19_authorized": False,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    write_jsonl(paths["results"], rows)
    write_json(paths["command_plan"], command_plan)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"expected schema_version {CONFIG_SCHEMA_VERSION}")

    profile_path = resolve_path(Path(str(payload.get("canonical_reward_profile", DEFAULT_CANONICAL_PROFILE))), repo_root).resolve()
    profile = load_canonical_reward_profile(profile_path)
    guard_thresholds = stage18_guard_thresholds(profile)
    workspace_root = resolve_path(Path(str(payload.get("artifact_workspace_root", DEFAULT_ARTIFACT_WORKSPACE))), repo_root).resolve()
    sweeps = _normalize_sweeps(payload.get("sweeps"), workspace_root, repo_root)
    config = {
        **payload,
        "config_path": path,
        "canonical_reward_profile": profile_path,
        "artifact_workspace_root": workspace_root,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "guard_thresholds": guard_thresholds.to_artifact_dict(),
        "sweeps": sweeps,
        "min_guard_clean_candidate_available_rate": float(payload.get("min_guard_clean_candidate_available_rate", 0.01)),
        "min_selected_guard_clean_rate_improvement": float(payload.get("min_selected_guard_clean_rate_improvement", 0.01)),
    }
    return config


def _normalize_sweeps(value: Any, workspace_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    raw_sweeps = value if isinstance(value, list) and value else []
    by_count = {int(row.get("candidate_count")): row for row in raw_sweeps if isinstance(row, dict) and row.get("candidate_count") is not None}
    sweeps = []
    for count, pool in DEFAULT_SWEEPS:
        raw = by_count.get(count, {})
        configured_pool = raw.get("proposal_pool_limit", pool) if isinstance(raw, dict) else pool
        matrix_mismatch = int(configured_pool) != pool
        count_root = workspace_root / f"count_{count:03d}"
        coverage_root = raw.get("coverage_comparison_root") if isinstance(raw, dict) else None
        stage18_5_root = raw.get("stage18_5_attribution_root") if isinstance(raw, dict) else None
        stage18_6_root = raw.get("stage18_6_guard_refinement_root") if isinstance(raw, dict) else None
        sweeps.append(
            {
                "candidate_count": count,
                "proposal_pool_limit": int(configured_pool),
                "expected_proposal_pool_limit": pool,
                "candidate_count_matrix_mismatch": matrix_mismatch,
                "coverage_comparison_root": resolve_path(Path(str(coverage_root or count_root / "stage18_4e")), repo_root).resolve(),
                "stage18_5_attribution_root": resolve_path(Path(str(stage18_5_root or count_root / "stage18_5")), repo_root).resolve(),
                "stage18_6_guard_refinement_root": resolve_path(Path(str(stage18_6_root or count_root / "stage18_6")), repo_root).resolve(),
            }
        )
    return sweeps


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "results": output_root / RESULTS_FILE,
        "command_plan": output_root / COMMAND_PLAN_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }


def _evaluate_sweep(sweep: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    coverage_root = Path(sweep["coverage_comparison_root"])
    stage18_5_root = Path(sweep["stage18_5_attribution_root"])
    stage18_6_root = Path(sweep["stage18_6_guard_refinement_root"])
    coverage_summary = _read_json(coverage_root / COVERAGE_SUMMARY_FILE)
    aggregate = _read_json(coverage_root / COVERAGE_AGGREGATE_FILE)
    pairs = _read_jsonl(coverage_root / COVERAGE_PAIRS_FILE)
    paired_rows = _read_jsonl(coverage_root / PAIRED_DECISION_AUDIT_FILE)
    candidate_path = coverage_root / CANDIDATE_METRIC_AUDIT_FILE
    candidate_rows = _read_jsonl(candidate_path)
    candidate_jsonl_errors = _jsonl_decode_errors(candidate_path)
    stage18_6 = _read_json(stage18_6_root / STAGE18_6_SUMMARY_FILE)
    reasons: list[str] = []
    blocking: list[str] = []
    if not coverage_summary:
        reasons.append("missing_coverage_comparison_summary")
    if not aggregate:
        reasons.append("missing_coverage_comparison_aggregate")
    if not pairs:
        reasons.append("missing_coverage_comparison_pairs")
    if not paired_rows:
        reasons.append("missing_paired_decision_audit")
    if not candidate_path.is_file() or not candidate_rows:
        reasons.append("missing_candidate_metric_audit")
    if not stage18_6:
        reasons.append("missing_stage18_6_guard_refinement_summary")
    if sweep.get("candidate_count_matrix_mismatch") is True:
        blocking.append("candidate_count_matrix_mismatch")
    if candidate_jsonl_errors:
        blocking.append("malformed_candidate_metric_audit_jsonl")
    if coverage_summary and coverage_summary.get("schema_version") != "xunce-exploration-coverage-comparison-summary/v1":
        blocking.append("invalid_coverage_comparison_summary_schema")
    if stage18_6 and stage18_6.get("schema_version") != STAGE18_6_SUMMARY_SCHEMA_VERSION:
        blocking.append("invalid_stage18_6_guard_refinement_summary_schema")
    if stage18_6 and not _same_path(stage18_6.get("coverage_comparison_root"), coverage_root):
        blocking.append("stale_stage18_6_guard_refinement_root")
    if stage18_6 and not _same_path(stage18_6.get("stage18_5_attribution_root"), stage18_5_root):
        blocking.append("stale_stage18_6_guard_refinement_root")
    profile_reasons = _profile_reasons(config, coverage_summary, stage18_6, candidate_rows)
    blocking.extend(profile_reasons)
    config_reasons = _normalized_config_reasons(sweep, coverage_summary)
    blocking.extend(config_reasons)
    boundary_reasons = _boundary_reasons(stage18_6, coverage_summary)
    blocking.extend(boundary_reasons)
    candidate_readiness = _candidate_readiness(candidate_rows, paired_rows, config)
    if candidate_readiness["reason_codes"]:
        blocking.extend(candidate_readiness["blocking_reason_codes"])
        reasons.extend(candidate_readiness["diagnostic_reason_codes"])
    metrics = _candidate_metrics(candidate_rows, paired_rows, config)
    stage18_6_routing = stage18_6.get("next_stage_routing", {}) if isinstance(stage18_6, dict) else {}
    stage18_6_route = stage18_6_routing.get("primary_route")
    if stage18_6_route and stage18_6_route not in ALLOWED_STAGE18_6_ROUTES:
        blocking.append("invalid_stage18_6_route")
    stage18_6_readiness = stage18_6.get("candidate_metric_readiness", {}) if isinstance(stage18_6, dict) else {}
    stage18_6_stage19 = stage18_6.get("stage19_readiness", {}) if isinstance(stage18_6, dict) else {}
    if stage18_6 and not _stage18_6_summary_semantics_are_valid(stage18_6, metrics):
        blocking.append("invalid_stage18_6_preflight_semantics")
    result = {
        "schema_version": RESULT_ROW_SCHEMA_VERSION,
        "candidate_count": int(sweep["candidate_count"]),
        "proposal_pool_limit": int(sweep["proposal_pool_limit"]),
        "expected_proposal_pool_limit": int(sweep["expected_proposal_pool_limit"]),
        "coverage_comparison_root": str(coverage_root),
        "stage18_6_guard_refinement_root": str(stage18_6_root),
        "profile_hash": coverage_summary.get("profile_hash") if coverage_summary else None,
        "candidate_metric_audit_available": candidate_path.is_file(),
        "candidate_metric_audit_row_count": len(candidate_rows),
        "candidate_metric_replay_available": bool(
            stage18_6_readiness.get("full_candidate_metric_replay_available") is True
            and candidate_readiness["full_candidate_metric_replay_available"] is True
        ),
        "guard_refinement_passed": stage18_6.get("guard_refinement_passed") is True if stage18_6 else False,
        "stage18_6_next_required_change": stage18_6_route,
        "stage19_authorized": False,
        "candidate_guard_mode": CANDIDATE_GUARD_MODE,
        "absolute_risk_proxy_is_audit_only": True,
        "candidate_level_risk_delta_guard_is_diagnostic_only": bool(
            config["guard_thresholds"].get("candidate_level_risk_delta_guard_is_diagnostic_only")
        ),
        "guard_clean_advantage_candidate_available_rate": metrics[
            "guard_clean_advantage_candidate_available_rate"
        ],
        "guard_clean_candidate_available_rate": metrics["guard_clean_candidate_available_rate"],
        "zero_guard_clean_advantage_candidate_step_rate": metrics[
            "zero_guard_clean_advantage_candidate_step_rate"
        ],
        "zero_guard_clean_candidate_step_rate": metrics["zero_guard_clean_candidate_step_rate"],
        "median_guard_clean_advantage_candidates_per_step": metrics[
            "median_guard_clean_advantage_candidates_per_step"
        ],
        "median_guard_clean_candidates_per_step": metrics["median_guard_clean_candidates_per_step"],
        "xunce_selected_guard_clean_advantage_rate": metrics["xunce_selected_guard_clean_advantage_rate"],
        "xunce_selected_guard_clean_rate": metrics["xunce_selected_guard_clean_rate"],
        "incumbent_selected_guard_clean_advantage_rate": metrics[
            "incumbent_selected_guard_clean_advantage_rate"
        ],
        "incumbent_selected_guard_clean_rate": metrics["incumbent_selected_guard_clean_rate"],
        "absolute_risk_proxy_min": metrics["absolute_risk_proxy_min"],
        "absolute_risk_proxy_median": metrics["absolute_risk_proxy_median"],
        "absolute_risk_proxy_p95": metrics["absolute_risk_proxy_p95"],
        "absolute_risk_proxy_max": metrics["absolute_risk_proxy_max"],
        "same_candidate_set_guard_clean_advantage_established": _same_candidate_advantage(stage18_6, metrics),
        "stage18_6_stage19_authorized": bool(stage18_6_stage19.get("authorized") is True)
        if isinstance(stage18_6_stage19, dict)
        else None,
        "coverage_delta_cells_mean": _aggregate_value(aggregate, "coverage_delta_cells_mean", "coverage_delta_cells"),
        "path_cost_delta_m_mean": _aggregate_value(aggregate, "path_cost_delta_m_mean", "path_cost_delta_m"),
        "risk_delta_mean": _aggregate_value(aggregate, "risk_delta_mean", "risk_delta"),
        "risk_cost_weighted_delta_mean": _aggregate_or_pair_mean(
            aggregate, pairs, "risk_cost_weighted_delta_mean", "risk_cost_weighted_delta"
        ),
        "coverage_per_100m_delta_mean": _aggregate_value(
            aggregate, "coverage_per_100m_delta_mean", "coverage_per_100m_delta"
        ),
        "candidate_generation_exhausted_count": int(coverage_summary.get("candidate_generation_exhausted_count", 0) or 0)
        if coverage_summary
        else 0,
        "dynamic_validation_attempt_count": int(coverage_summary.get("dynamic_validation_attempt_count", 0) or 0)
        if coverage_summary
        else 0,
        "dynamic_validation_success_count": int(coverage_summary.get("dynamic_validation_success_count", 0) or 0)
        if coverage_summary
        else 0,
        "candidate_set_hash_mismatch_count": int(coverage_summary.get("candidate_set_hash_mismatch_count", 0) or 0)
        if coverage_summary
        else 0,
        "model_inference_failure_count": int(coverage_summary.get("model_inference_failure_count", 0) or 0)
        if coverage_summary
        else 0,
        "mask_violation_count": int(coverage_summary.get("mask_violation_count", 0) or 0) if coverage_summary else 0,
        "boundary_flags_all_false": not bool(boundary_reasons),
        "runtime_budget_exceeded": bool(coverage_summary.get("runtime_budget_exceeded") is True) if coverage_summary else False,
        "normalized_config": _normalized_config_payload(coverage_summary),
        "sweep_complete": bool(coverage_summary and stage18_6 and candidate_path.is_file() and candidate_rows),
        "reason_codes": unique_sorted([*reasons, *blocking]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(reasons),
    }
    return result


def _apply_cross_sweep_config_drift(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = None
    for row in rows:
        normalized = row.get("normalized_config")
        if isinstance(normalized, dict) and normalized:
            baseline = _comparable_normalized_config(normalized)
            break
    if baseline is None:
        return rows
    updated = []
    for row in rows:
        normalized = row.get("normalized_config")
        if isinstance(normalized, dict) and normalized:
            comparable = _comparable_normalized_config(normalized)
            if comparable != baseline:
                row = dict(row)
                blocking = list(row.get("blocking_reason_codes", []))
                blocking.append("non_count_config_drift")
                row["blocking_reason_codes"] = unique_sorted(blocking)
                row["reason_codes"] = unique_sorted([*row.get("reason_codes", []), "non_count_config_drift"])
        updated.append(row)
    return updated


def _normalized_config_payload(coverage: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    normalized = coverage.get("normalized_config")
    if isinstance(normalized, dict):
        return dict(normalized)
    keys = set(ONLY_COUNT_FIELDS) | {
        "candidate_refresh_mode",
        "dynamic_candidate_generation_mode",
        "dynamic_candidate_selection_mode",
        "dynamic_candidate_validation_mode",
        "coverage_metric_mode",
        "include_oracle_baselines",
        "include_roi_weighted_coverage",
        "rollout_steps",
        "required_scenario_count",
        "emit_candidate_metric_audit",
        "dynamic_frontier_radius_cells",
        "dynamic_validation_max_path_length",
        "dynamic_validation_work_root",
    }
    return {key: coverage[key] for key in keys if key in coverage}


def _comparable_normalized_config(normalized: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(normalized.items())
        if key not in ONLY_COUNT_FIELDS and key not in OUTPUT_LOCAL_CONFIG_FIELDS
    }


def _profile_reasons(config: dict[str, Any], coverage: dict[str, Any], stage18_6: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> list[str]:
    if not coverage and not stage18_6:
        return []
    reasons: list[str] = []
    expected = {field: config[field] for field in ("profile_id", "profile_version", "profile_hash")}
    for label, payload in (("coverage", coverage), ("stage18_6", stage18_6)):
        if not payload:
            continue
        for field, expected_value in expected.items():
            actual = payload.get(field)
            if not actual:
                reasons.append("profile_hash_missing")
            elif actual != expected_value:
                reasons.append("profile_hash_mismatch")
    for row in candidate_rows:
        for field, expected_value in expected.items():
            actual = row.get(field)
            if not actual:
                reasons.append("profile_hash_missing")
            elif actual != expected_value:
                reasons.append("profile_hash_mismatch")
    return unique_sorted(reasons)


def _normalized_config_reasons(sweep: dict[str, Any], coverage: dict[str, Any]) -> list[str]:
    if not coverage:
        return []
    normalized = _normalized_config_payload(coverage)
    reasons = []
    expected_count = int(sweep["candidate_count"])
    expected_pool = int(sweep["proposal_pool_limit"])
    if int(normalized.get("dynamic_max_candidates_per_step", -1) or -1) != expected_count:
        reasons.append("candidate_count_config_mismatch")
    if int(normalized.get("dynamic_proposal_pool_limit_per_step", -1) or -1) != expected_pool:
        reasons.append("candidate_count_config_mismatch")
    expected_fixed = {
        "candidate_refresh_mode": "dynamic_frontier_nbv_in_process",
        "dynamic_candidate_generation_mode": "map_aware_coverage_frontier_nbv",
        "dynamic_candidate_selection_mode": "validated_pareto_diverse",
        "dynamic_candidate_validation_mode": "in_process_path_planner_astar_batch",
        "coverage_metric_mode": "path_line_plus_endpoint",
        "include_oracle_baselines": True,
        "include_roi_weighted_coverage": True,
        "rollout_steps": 40,
        "required_scenario_count": 24,
        "emit_candidate_metric_audit": True,
    }
    for field, expected in expected_fixed.items():
        if normalized.get(field) != expected:
            reasons.append("non_count_config_drift")
    return unique_sorted(reasons)


def _stage18_6_summary_semantics_are_valid(stage18_6: dict[str, Any], metrics: dict[str, float]) -> bool:
    if not isinstance(stage18_6, dict) or not stage18_6:
        return True
    routing = stage18_6.get("next_stage_routing", {})
    readiness = stage18_6.get("stage19_readiness", {})
    candidate_readiness = stage18_6.get("candidate_metric_readiness", {})
    paired = stage18_6.get("paired_decision_summary", {})
    route = routing.get("primary_route") if isinstance(routing, dict) else None
    if isinstance(routing, dict) and routing.get("stage19_authorized") is not False:
        return False
    if not isinstance(readiness, dict) or readiness.get("authorized") is not False:
        return False
    if stage18_6.get("guard_refinement_passed") is True:
        return (
            route == ROUTE_STAGE19_PREFLIGHT
            and isinstance(candidate_readiness, dict)
            and candidate_readiness.get("full_candidate_metric_replay_available") is True
            and isinstance(paired, dict)
            and paired.get("same_candidate_set_guard_clean_advantage_established") is True
            and metrics["guard_clean_candidate_available_rate"] > 0.0
            and metrics["xunce_selected_guard_clean_rate"] >= metrics["incumbent_selected_guard_clean_rate"]
        )
    if route == ROUTE_STAGE19_PREFLIGHT:
        return False
    return True


def _candidate_readiness(
    candidate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    blocking: list[str] = []
    diagnostic: list[str] = []
    required_fields = {
        "scenario_id",
        "step_index",
        "covered_cells_hash",
        "candidate_set_hash",
        "candidate_index",
        "action_mask_valid",
        "expected_new_coverage_cell_count",
        "path_cost",
        "risk",
        "profile_hash",
    }
    paired_keys = {paired_decision_key(row) for row in paired_rows}
    candidate_keys = set()
    for row in candidate_rows:
        missing = sorted(field for field in required_fields if field not in row)
        blocking.extend(f"missing_candidate_metric_field:{field}" for field in missing)
        candidate_keys.add(paired_decision_key(row))
        if row.get("profile_hash") and row.get("profile_hash") != config["profile_hash"]:
            blocking.append("profile_hash_mismatch")
    if paired_keys and candidate_keys and paired_keys - candidate_keys:
        blocking.append("candidate_metric_missing_paired_decision_key")
    if paired_keys and candidate_keys and candidate_keys - paired_keys:
        diagnostic.append("candidate_metric_extra_candidate_set_key")
    return {
        "full_candidate_metric_replay_available": bool(candidate_rows and not blocking),
        "reason_codes": unique_sorted([*blocking, *diagnostic]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
    }


def _candidate_metrics(
    candidate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, float]:
    rows_by_step: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        rows_by_step.setdefault(paired_decision_key(row), []).append(row)
    clean_counts = []
    xunce_clean = 0
    incumbent_clean = 0
    comparable = 0
    absolute_risks = [
        risk
        for risk in (risk_proxy_value(row) for row in candidate_rows)
        if risk is not None
    ]
    for paired in paired_rows:
        key = paired_decision_key(paired)
        candidates = rows_by_step.get(key, [])
        baseline = selected_baseline_candidate(paired, candidates)
        clean_by_index = {
            int(row.get("candidate_index", -1)): _candidate_guard_clean(row, paired, baseline, config)
            for row in candidates
        }
        clean_count = sum(1 for value in clean_by_index.values() if value)
        clean_counts.append(clean_count)
        if candidates:
            comparable += 1
            xunce_index = _int_or_none(paired.get("xunce_selected_action_index"))
            incumbent_index = _int_or_none(paired.get("incumbent_selected_action_index"))
            if xunce_index is not None and clean_by_index.get(xunce_index) is True:
                xunce_clean += 1
            if incumbent_index is not None and clean_by_index.get(incumbent_index) is True:
                incumbent_clean += 1
    step_count = len(clean_counts)
    clean_available_rate = sum(1 for count in clean_counts if count > 0) / step_count if step_count else 0.0
    zero_clean_rate = sum(1 for count in clean_counts if count == 0) / step_count if step_count else 1.0
    median_clean = float(median(clean_counts)) if clean_counts else 0.0
    xunce_rate = xunce_clean / comparable if comparable else 0.0
    incumbent_rate = incumbent_clean / comparable if comparable else 0.0
    return {
        "guard_clean_advantage_candidate_available_rate": clean_available_rate,
        "guard_clean_candidate_available_rate": clean_available_rate,
        "zero_guard_clean_advantage_candidate_step_rate": zero_clean_rate,
        "zero_guard_clean_candidate_step_rate": zero_clean_rate,
        "median_guard_clean_advantage_candidates_per_step": median_clean,
        "median_guard_clean_candidates_per_step": median_clean,
        "xunce_selected_guard_clean_advantage_rate": xunce_rate,
        "xunce_selected_guard_clean_rate": xunce_rate,
        "incumbent_selected_guard_clean_advantage_rate": incumbent_rate,
        "incumbent_selected_guard_clean_rate": incumbent_rate,
        "absolute_risk_proxy_min": min(absolute_risks) if absolute_risks else None,
        "absolute_risk_proxy_median": float(median(absolute_risks)) if absolute_risks else None,
        "absolute_risk_proxy_p95": _percentile(absolute_risks, 0.95),
        "absolute_risk_proxy_max": max(absolute_risks) if absolute_risks else None,
    }


def _candidate_guard_clean(
    row: dict[str, Any],
    paired: dict[str, Any],
    baseline: dict[str, Any] | None,
    config: dict[str, Any],
) -> bool:
    return candidate_guard_clean(
        candidate_guard_observation(row, paired, baseline_candidate_row=baseline),
        config["guard_thresholds"],
    )


def _same_candidate_advantage(stage18_6: dict[str, Any], metrics: dict[str, float]) -> bool:
    paired = stage18_6.get("paired_decision_summary", {}) if isinstance(stage18_6, dict) else {}
    return bool(
        stage18_6.get("guard_refinement_passed") is True
        and paired.get("same_candidate_set_guard_clean_advantage_established") is True
        and metrics["xunce_selected_guard_clean_rate"] >= metrics["incumbent_selected_guard_clean_rate"]
    )


def _blocking_reasons(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    reasons = _boundary_reasons(config)
    row_blocking = [reason for row in rows for reason in row.get("blocking_reason_codes", [])]
    if any(reason in row_blocking for reason in ("profile_hash_missing", "profile_hash_mismatch", "non_count_config_drift", "candidate_count_config_mismatch", "stale_stage18_6_guard_refinement_root")):
        reasons.extend(reason for reason in row_blocking if reason)
    elif row_blocking:
        reasons.extend(row_blocking)
    return unique_sorted(reasons)


def _diagnostic_reasons(rows: list[dict[str, Any]]) -> list[str]:
    reasons = []
    for row in rows:
        reasons.extend(row.get("diagnostic_reason_codes", []))
        if row.get("runtime_budget_exceeded") is True:
            reasons.append(f"candidate_count_{int(row['candidate_count']):d}_runtime_budget_exceeded")
        if not row["sweep_complete"]:
            reasons.append("missing_candidate_count_sweep")
    return unique_sorted(reasons)


def _route(*, blocking: list[str], diagnostic: list[str], rows: list[dict[str, Any]]) -> str:
    if {"boundary_violation", "canary_traffic_fraction_nonzero"} & set(blocking):
        return ROUTE_BOUNDARY_REPAIR
    if blocking:
        return ROUTE_REPAIR_LINEAGE
    complete_rows = [row for row in rows if row["sweep_complete"] and row.get("runtime_budget_exceeded") is not True]
    if any(reason.endswith("_runtime_budget_exceeded") for reason in diagnostic) and complete_rows:
        return ROUTE_BOUNDED_BUDGET
    if len(complete_rows) < len(DEFAULT_SWEEPS):
        return ROUTE_MISSING_SWEEPS
    if any(
        row["guard_refinement_passed"] is True and row["same_candidate_set_guard_clean_advantage_established"] is True
        for row in complete_rows
    ):
        return ROUTE_STAGE19_PREFLIGHT
    best_guard_clean = max(row["guard_clean_candidate_available_rate"] for row in complete_rows) if complete_rows else 0.0
    if best_guard_clean <= 0.0:
        return ROUTE_EXPAND_CANDIDATES
    return ROUTE_REFINE_REWARD_GUARD


def _boundary_reasons(*payloads: dict[str, Any]) -> list[str]:
    reasons = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        canary = _finite(payload.get("canary_traffic_fraction"))
        if canary not in (None, 0.0):
            reasons.append("canary_traffic_fraction_nonzero")
        for field in BOUNDARY_FIELDS:
            if payload.get(field) is True:
                reasons.append("boundary_violation")
    return unique_sorted(reasons)


def _command_plan(config: dict[str, Any], repo_root: Path, output_root: Path) -> dict[str, Any]:
    commands = []
    run_stage = repo_root / "scripts" / "run_stage.py"
    stage18_5_script = repo_root / "scripts" / "run_xunce_stage18_5_evidence_attribution_review.py"
    stage18_6_script = repo_root / "scripts" / "run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py"
    stage18_7_script = repo_root / "scripts" / "run_xunce_stage18_7_candidate_count_scaling_audit.py"
    strict_v3 = config.get("profile_version") == "v3"
    stage18_4e_config = (
        repo_root / "configs" / "xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json"
        if strict_v3
        else repo_root / "configs" / "xunce_high_fidelity_exploration_coverage_comparison_v1.json"
    )
    stage18_5_config = (
        repo_root / "configs" / "xunce_stage18_5_evidence_attribution_review_strict_v3.json"
        if strict_v3
        else repo_root / "configs" / "xunce_stage18_5_evidence_attribution_review_v1.json"
    )
    stage18_6_config = (
        repo_root / "configs" / "xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_strict_v3.json"
        if strict_v3
        else repo_root / "configs" / "xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json"
    )
    for sweep in config["sweeps"]:
        count = str(sweep["candidate_count"])
        pool = str(sweep["proposal_pool_limit"])
        coverage_root = str(sweep["coverage_comparison_root"])
        stage18_5_root = str(sweep["stage18_5_attribution_root"])
        stage18_6_root = str(sweep["stage18_6_guard_refinement_root"])
        argv18_4e = python_script_command(
            run_stage,
            "--stage",
            "xunce-high-fidelity-exploration-coverage-comparison",
            "--config",
            str(stage18_4e_config),
            "--output-root",
            coverage_root,
            "--extra-arg",
            "--rollout-steps",
            "--extra-arg",
            "40",
            "--extra-arg",
            "--candidate-refresh-mode",
            "--extra-arg",
            "dynamic_frontier_nbv_in_process",
            "--extra-arg",
            "--dynamic-candidate-validation-mode",
            "--extra-arg",
            "in_process_path_planner_astar_batch",
            "--extra-arg",
            "--coverage-metric-mode",
            "--extra-arg",
            "path_line_plus_endpoint",
            "--extra-arg",
            "--dynamic-max-candidates-per-step",
            "--extra-arg",
            count,
            "--extra-arg",
            "--dynamic-proposal-pool-limit-per-step",
            "--extra-arg",
            pool,
            "--extra-arg",
            "--include-oracle-baselines",
            "--extra-arg",
            "--include-roi-weighted-coverage",
            "--extra-arg",
            "--emit-candidate-metric-audit",
        )
        argv18_5 = python_script_command(
            stage18_5_script,
            "--config",
            str(stage18_5_config),
            "--coverage-comparison-root",
            coverage_root,
            "--output-root",
            stage18_5_root,
            "--repo-root",
            str(repo_root),
        )
        argv18_6 = python_script_command(
            stage18_6_script,
            "--config",
            str(stage18_6_config),
            "--coverage-comparison-root",
            coverage_root,
            "--stage18-5-attribution-root",
            stage18_5_root,
            "--output-root",
            stage18_6_root,
            "--repo-root",
            str(repo_root),
        )
        for stage, argv in (("18.4E", argv18_4e), ("18.5", argv18_5), ("18.6", argv18_6)):
            commands.append(
                {
                    "stage": stage,
                    "candidate_count": int(count),
                    "proposal_pool_limit": int(pool),
                    "argv": argv,
                    "display": display_command(argv),
                }
            )
    argv18_7 = python_script_command(
        stage18_7_script,
        "--config",
        str(config["config_path"]),
        "--output-root",
        str(output_root),
        "--repo-root",
        str(repo_root),
    )
    commands.append({"stage": "18.7", "argv": argv18_7, "display": display_command(argv18_7)})
    return {
        "schema_version": COMMAND_PLAN_SCHEMA_VERSION,
        "commands": commands,
        "read_only_runner": True,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18.7 Candidate Count Scaling Audit",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- stage19_authorized: `{summary['stage19_authorized']}`",
            f"- best_guard_clean_candidate_available_rate: `{summary['best_guard_clean_candidate_available_rate']}`",
            f"- best_guard_clean_advantage_candidate_available_rate: `{summary['best_guard_clean_advantage_candidate_available_rate']}`",
            f"- best_xunce_selected_guard_clean_rate: `{summary['best_xunce_selected_guard_clean_rate']}`",
            f"- best_xunce_selected_guard_clean_advantage_rate: `{summary['best_xunce_selected_guard_clean_advantage_rate']}`",
            f"- best_incumbent_selected_guard_clean_rate: `{summary['best_incumbent_selected_guard_clean_rate']}`",
            "",
            "Stage 18.7 only audits candidate-count scaling. It does not change candidate generation logic, train PPO, publish checkpoints, replace the default policy, connect a real executor, or start canary traffic.",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _jsonl_decode_errors(path: Path) -> list[str]:
    if not path.is_file():
        return []
    errors = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"{path.name}:{index}")
    return errors


def _same_path(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve() == expected.resolve()


def _aggregate_value(payload: dict[str, Any], direct_key: str, nested_key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    direct = _finite(payload.get(direct_key))
    if direct is not None:
        return direct
    distribution = payload.get(f"{nested_key}_distribution")
    if isinstance(distribution, dict):
        return _finite(distribution.get("mean"))
    return None


def _aggregate_or_pair_mean(
    payload: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    direct_key: str,
    nested_key: str,
) -> float | None:
    value = _aggregate_value(payload, direct_key, nested_key)
    if value is not None:
        return value
    numbers = [_finite(row.get(nested_key)) for row in pair_rows]
    finite_numbers = [float(number) for number in numbers if number is not None]
    if not finite_numbers:
        return None
    return sum(finite_numbers) / len(finite_numbers)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return float(ordered[index])


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
