from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now
from global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-stage19-evaluator-critic-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage19-evaluator-critic-preflight-summary/v1"
EVALUATOR_SCHEMA_VERSION = "xunce-stage19-diagnostic-rollout-evaluator/v1"
TARGET_SCHEMA_VERSION = "xunce-stage19-practical-target-selection/v1"
PREFERENCE_ROW_SCHEMA_VERSION = "xunce-stage19-preference-pair-audit-row/v1"
CRITIC_READINESS_SCHEMA_VERSION = "xunce-stage19-critic-target-readiness/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage19-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage19-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage19_evaluator_critic_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage19_evaluator_critic_preflight/"
    "outputs/path_feedback_batch_xunce_stage19_evaluator_critic_preflight_v1"
)

STAGE18_11_SUMMARY_FILE = "xunce-stage18-11-path-cost-weight-calibration-summary.json"
STAGE18_11_MANIFEST_FILE = "xunce-stage18-11-manifest.json"
COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_EPISODES_FILE = "xunce-exploration-coverage-episodes.jsonl"
COVERAGE_STEPS_FILE = "xunce-exploration-coverage-steps.jsonl"
PAIRED_DECISION_AUDIT_FILE = "xunce-exploration-coverage-paired-decision-audit.jsonl"
CANDIDATE_METRIC_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"

SUMMARY_FILE = "xunce-stage19-evaluator-critic-preflight-summary.json"
EVALUATOR_FILE = "xunce-stage19-diagnostic-rollout-evaluator.json"
TARGET_FILE = "xunce-stage19-practical-target-selection.json"
PREFERENCE_FILE = "xunce-stage19-preference-pair-audit.jsonl"
CRITIC_READINESS_FILE = "xunce-stage19-critic-target-readiness.json"
ROUTING_FILE = "xunce-stage19-next-stage-routing.json"
REPORT_FILE = "xunce-stage19-report.md"
MANIFEST_FILE = "xunce-stage19-manifest.json"

POLICY_ORACLE = "canonical_reward_rerank_oracle"
POLICY_XUNCE = "xunce"
POLICY_INCUMBENT = "incumbent"

ROUTE_BOUNDARY = "resolve_stage19_evaluator_critic_preflight_boundary_rejections"
ROUTE_INPUTS = "rerun_stage18_11_reward_rerank_diagnostic_rollouts"
ROUTE_RISK = "repair_path_risk_boundary_filtering"
ROUTE_STAGE18_12 = "stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage"
ROUTE_CONTINUE = "continue_path_cost_weight_calibration_at_99pct_coverage"
ROUTE_PREFERENCE = "collect_more_reward_rerank_preference_evidence"
ROUTE_STAGE20 = "stage20_reward_rerank_oracle_preference_dataset_preparation"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "stage20_authorized",
    "training_or_release_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "real_world_release_approved",
    "real_world_performance_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Stage 19 evaluator/critic preflight.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--stage18-11-path-cost-weight-calibration-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        "stage18_11_path_cost_weight_calibration_root": args.stage18_11_path_cost_weight_calibration_root,
    }
    try:
        summary = run_xunce_stage19_evaluator_critic_preflight(
            config_path=resolve_path(Path(args.config), repo_root).resolve(),
            output_root=resolve_path(Path(args.output_root), repo_root).resolve(),
            repo_root=repo_root,
            overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "oracle_target_feasible": summary["oracle_target_feasible"],
                "primary_target_selected": summary["primary_target_selected"],
                "xunce_checkpoint_advantage_established": summary["xunce_checkpoint_advantage_established"],
                "stage20_authorized": summary["stage20_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage19_evaluator_critic_preflight(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path=config_path, repo_root=repo_root, overrides=overrides or {})
    output_root.mkdir(parents=True, exist_ok=True)

    blocking: list[str] = []
    diagnostic: list[str] = []
    blocking.extend(_boundary_violations(config))

    stage18_11_root = config["stage18_11_path_cost_weight_calibration_root"]
    stage18_11, stage18_11_manifest, stage18_11_reasons = _load_stage18_11(stage18_11_root)
    blocking.extend(stage18_11_reasons)

    evaluator = _evaluate_diagnostic_rollouts(
        stage18_11=stage18_11,
        expected_profile=config["expected_profile"],
        target=config["target_final_coverage_rate_capped"],
    )
    blocking.extend(evaluator["blocking_reason_codes"])
    diagnostic.extend(evaluator["diagnostic_reason_codes"])

    target = _select_practical_target(config=config, evaluator=evaluator)
    preference_rows = _preference_pair_audit(target.get("selected_root"))
    critic = _critic_readiness(config=config, preference_rows=preference_rows)
    diagnostic.extend(critic["reason_codes"])

    route = _route(config=config, blocking=blocking, evaluator=evaluator, target=target, critic=critic)
    status = "passed" if route == ROUTE_STAGE20 else ("partial" if route in {ROUTE_CONTINUE, ROUTE_PREFERENCE} else "failed")

    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage20_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    oracle_target_feasible = bool(target.get("oracle_target_feasible") is True)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "stage18_11_path_cost_weight_calibration_root": str(stage18_11_root),
        "stage18_11_manifest": stage18_11_manifest,
        "profile_id": stage18_11.get("profile_id"),
        "profile_version": stage18_11.get("profile_version"),
        "profile_hash": stage18_11.get("profile_hash"),
        "target_final_coverage_rate_capped": config["target_final_coverage_rate_capped"],
        "primary_candidate_count": config["primary_candidate_count"],
        "primary_path_cost_weight": config["primary_path_cost_weight"],
        "upper_bound_candidate_count": config["upper_bound_candidate_count"],
        "upper_bound_path_cost_weight": config["upper_bound_path_cost_weight"],
        "oracle_target_feasible": oracle_target_feasible,
        "primary_target_selected": bool(target.get("primary_target_selected") is True),
        "selected_candidate_count": target.get("selected_candidate_count"),
        "selected_path_cost_weight": target.get("selected_path_cost_weight"),
        "selected_root": target.get("selected_root"),
        "xunce_checkpoint_advantage_established": evaluator["xunce_checkpoint_advantage_established"],
        "training_or_release_authorized": False,
        "stage20_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "reason_codes": unique_sorted([*blocking, *diagnostic]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
        "diagnostic_rollout_evaluator": evaluator,
        "practical_target_selection": target,
        "critic_target_readiness": critic,
        "preference_pair_count": critic["preference_pair_count"],
        "next_required_change": route,
        "next_stage_routing": routing,
        "stage20_readiness": {
            "schema_version": "xunce-stage19-stage20-readiness/v1",
            "readiness": "ready_for_stage20_preference_dataset_preparation_human_review_only"
            if route == ROUTE_STAGE20
            else "not_authorized",
            "authorized": False,
            "oracle_target_feasible": oracle_target_feasible,
            "xunce_checkpoint_advantage_established": evaluator["xunce_checkpoint_advantage_established"],
        },
        "summary": str(output_root / SUMMARY_FILE),
    }

    _write_json(output_root / EVALUATOR_FILE, evaluator)
    _write_json(output_root / TARGET_FILE, target)
    _write_json(output_root / CRITIC_READINESS_FILE, critic)
    _write_jsonl(output_root / PREFERENCE_FILE, preference_rows)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(
        output_root / MANIFEST_FILE,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": summary["generated_at"],
            "summary": str(output_root / SUMMARY_FILE),
            "artifacts": [
                SUMMARY_FILE,
                EVALUATOR_FILE,
                TARGET_FILE,
                PREFERENCE_FILE,
                CRITIC_READINESS_FILE,
                ROUTING_FILE,
                REPORT_FILE,
                MANIFEST_FILE,
            ],
            "stage20_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        },
    )
    return summary


def _load_config(*, config_path: Path, repo_root: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid config json: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config payload must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    merged = dict(payload)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    root = merged.get("stage18_11_path_cost_weight_calibration_root")
    if not isinstance(root, str) or not root:
        raise ConfigError("stage18_11_path_cost_weight_calibration_root must be a non-empty path string")
    config = {
        "stage18_11_path_cost_weight_calibration_root": resolve_path(Path(root), repo_root).resolve(),
        "target_final_coverage_rate_capped": _finite_required(merged, "target_final_coverage_rate_capped"),
        "primary_candidate_count": _int_required(merged, "primary_candidate_count"),
        "primary_path_cost_weight": _finite_required(merged, "primary_path_cost_weight"),
        "upper_bound_candidate_count": _int_required(merged, "upper_bound_candidate_count"),
        "upper_bound_path_cost_weight": _finite_required(merged, "upper_bound_path_cost_weight"),
        "max_primary_path_cost_total_m_mean": _finite_required(merged, "max_primary_path_cost_total_m_mean"),
        "max_primary_soft_risk_exposure_total_mean": _finite_required(merged, "max_primary_soft_risk_exposure_total_mean"),
        "max_hard_risk_violation_count": _finite_required(merged, "max_hard_risk_violation_count"),
        "min_preference_pair_count": _int_required(merged, "min_preference_pair_count"),
        "expected_profile": {
            "profile_id": _str_required(merged, "profile_id"),
            "profile_version": _str_required(merged, "profile_version"),
        },
    }
    if not (0.0 < config["target_final_coverage_rate_capped"] <= 1.0):
        raise ConfigError("target_final_coverage_rate_capped must be in (0, 1]")
    for field in BOUNDARY_FIELDS:
        value = merged.get(field, False)
        if field == "canary_traffic_fraction":
            config[field] = float(value or 0.0)
        elif not isinstance(value, bool):
            raise ConfigError(f"{field} must be a boolean")
        else:
            config[field] = value
    return config


def _load_stage18_11(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    reasons: list[str] = []
    summary = _read_json(root / STAGE18_11_SUMMARY_FILE, reasons, "missing_stage18_11_summary")
    manifest = _read_json(root / STAGE18_11_MANIFEST_FILE, reasons, "missing_stage18_11_manifest")
    if summary:
        if summary.get("schema_version") != "xunce-stage18-11-path-cost-weight-calibration-summary/v1":
            reasons.append("invalid_stage18_11_summary_schema")
        if summary.get("status") != "passed":
            reasons.append("stage18_11_not_passed")
        if summary.get("next_required_change") != "prepare_stage19_evaluator_critic_preflight":
            reasons.append("stage18_11_not_routed_to_stage19_preflight")
        if summary.get("stage19_authorized") is not False:
            reasons.append("stage18_11_stage19_authorized_not_false")
        if summary.get("profile_version") != "v3":
            reasons.append("stage18_11_profile_version_not_v3")
    return summary, manifest, unique_sorted(reasons)


def _evaluate_diagnostic_rollouts(
    *,
    stage18_11: dict[str, Any],
    expected_profile: dict[str, str],
    target: float,
) -> dict[str, Any]:
    rows = []
    blocking: list[str] = []
    diagnostic: list[str] = []
    rollout_rows = (
        stage18_11.get("diagnostic_rollout_summary", {}).get("rows", [])
        if isinstance(stage18_11.get("diagnostic_rollout_summary"), dict)
        else []
    )
    if not isinstance(rollout_rows, list) or not rollout_rows:
        blocking.append("missing_stage18_11_diagnostic_rollout_rows")
        rollout_rows = []
    xunce_advantage_established = False
    for row in rollout_rows:
        if not isinstance(row, dict):
            blocking.append("invalid_stage18_11_diagnostic_rollout_row")
            continue
        root = Path(str(row.get("root", "")))
        candidate_count = _safe_int(row.get("candidate_count"))
        path_cost_weight = _safe_float(row.get("path_cost_weight"))
        eval_row = _evaluate_one_rollout(
            root=root,
            candidate_count=candidate_count,
            path_cost_weight=path_cost_weight,
            expected_profile=expected_profile,
            target=target,
        )
        rows.append(eval_row)
        blocking.extend(eval_row["blocking_reason_codes"])
        diagnostic.extend(eval_row["diagnostic_reason_codes"])
        if eval_row.get("xunce_coverage_advantage_established") is True:
            xunce_advantage_established = True
    feasible_rows = [
        row
        for row in rows
        if row.get("oracle_target_feasible") is True
        and float(row.get("oracle_hard_risk_violation_count", 0.0) or 0.0) <= 0.0
    ]
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "target_final_coverage_rate_capped": target,
        "diagnostic_rollout_count": len(rows),
        "feasible_rollout_count": len(feasible_rows),
        "xunce_checkpoint_advantage_established": xunce_advantage_established,
        "rows": rows,
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
    }


def _evaluate_one_rollout(
    *,
    root: Path,
    candidate_count: int | None,
    path_cost_weight: float | None,
    expected_profile: dict[str, str],
    target: float,
) -> dict[str, Any]:
    blocking: list[str] = []
    diagnostic: list[str] = []
    summary = _read_json(root / COVERAGE_SUMMARY_FILE, blocking, "missing_diagnostic_rollout_summary")
    episodes = _read_jsonl(root / COVERAGE_EPISODES_FILE, blocking, "missing_diagnostic_rollout_episodes")
    if summary:
        if summary.get("profile_id") != expected_profile["profile_id"]:
            blocking.append("diagnostic_rollout_profile_id_mismatch")
        if summary.get("profile_version") != expected_profile["profile_version"]:
            blocking.append("diagnostic_rollout_profile_version_mismatch")
        if summary.get("include_canonical_reward_rerank_oracle") is not True:
            blocking.append("diagnostic_rollout_missing_reward_rerank_oracle")
    by_policy = {policy: [row for row in episodes if row.get("policy") == policy] for policy in (POLICY_ORACLE, POLICY_XUNCE, POLICY_INCUMBENT)}
    for policy, policy_rows in by_policy.items():
        if not policy_rows:
            blocking.append(f"missing_{policy}_episodes")
    oracle_stats = _policy_stats(by_policy[POLICY_ORACLE])
    xunce_stats = _policy_stats(by_policy[POLICY_XUNCE])
    incumbent_stats = _policy_stats(by_policy[POLICY_INCUMBENT])
    oracle_target_feasible = (
        oracle_stats["final_coverage_rate_capped_mean"] is not None
        and oracle_stats["final_coverage_rate_capped_min"] is not None
        and oracle_stats["final_coverage_rate_capped_mean"] >= target
        and oracle_stats["final_coverage_rate_capped_min"] >= target
        and oracle_stats["hard_risk_violation_count"] <= 0.0
    )
    return {
        "schema_version": "xunce-stage19-diagnostic-rollout-evaluator-row/v1",
        "root": str(root),
        "candidate_count": candidate_count,
        "path_cost_weight": path_cost_weight,
        "complete": bool(summary and episodes),
        "profile_id": summary.get("profile_id") if summary else None,
        "profile_version": summary.get("profile_version") if summary else None,
        "profile_hash": summary.get("profile_hash") if summary else None,
        "canonical_reward_rerank_profile_id": summary.get("canonical_reward_rerank_profile_id") if summary else None,
        "canonical_reward_rerank_profile_hash": summary.get("canonical_reward_rerank_profile_hash") if summary else None,
        "xunce_coverage_advantage_established": bool(summary.get("xunce_coverage_advantage_established") is True) if summary else False,
        "oracle_target_feasible": oracle_target_feasible,
        "oracle": oracle_stats,
        "xunce": xunce_stats,
        "incumbent": incumbent_stats,
        "oracle_final_coverage_rate_capped_mean": oracle_stats["final_coverage_rate_capped_mean"],
        "oracle_final_coverage_rate_capped_min": oracle_stats["final_coverage_rate_capped_min"],
        "oracle_final_coverage_rate_capped_max": oracle_stats["final_coverage_rate_capped_max"],
        "oracle_path_cost_total_m_mean": oracle_stats["path_cost_total_m_mean"],
        "oracle_path_cost_total_m_p95": oracle_stats["path_cost_total_m_p95"],
        "oracle_soft_risk_exposure_total_mean": oracle_stats["soft_risk_exposure_total_mean"],
        "oracle_soft_risk_exposure_total_p95": oracle_stats["soft_risk_exposure_total_p95"],
        "oracle_hard_risk_violation_count": oracle_stats["hard_risk_violation_count"],
        "xunce_vs_oracle_coverage_gap_mean": _gap(oracle_stats["final_coverage_rate_capped_mean"], xunce_stats["final_coverage_rate_capped_mean"]),
        "xunce_vs_oracle_path_cost_gap_mean": _gap(oracle_stats["path_cost_total_m_mean"], xunce_stats["path_cost_total_m_mean"]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
    }


def _select_practical_target(*, config: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    rows = evaluator.get("rows", [])
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("oracle_target_feasible") is True
        and float(row.get("oracle_hard_risk_violation_count", 0.0) or 0.0) <= config["max_hard_risk_violation_count"]
    ]
    selected = min(candidates, key=lambda row: float(row.get("oracle_path_cost_total_m_mean") or float("inf"))) if candidates else None
    primary = _find_rollout(rows, config["primary_candidate_count"], config["primary_path_cost_weight"])
    upper = _find_rollout(rows, config["upper_bound_candidate_count"], config["upper_bound_path_cost_weight"])
    primary_budget_passed = bool(
        primary
        and primary.get("oracle_target_feasible") is True
        and float(primary.get("oracle_path_cost_total_m_mean") or float("inf")) <= config["max_primary_path_cost_total_m_mean"]
        and float(primary.get("oracle_soft_risk_exposure_total_mean") or float("inf")) <= config["max_primary_soft_risk_exposure_total_mean"]
    )
    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "oracle_target_feasible": bool(candidates),
        "selected_candidate_count": selected.get("candidate_count") if selected else None,
        "selected_path_cost_weight": selected.get("path_cost_weight") if selected else None,
        "selected_root": selected.get("root") if selected else None,
        "selected_path_cost_total_m_mean": selected.get("oracle_path_cost_total_m_mean") if selected else None,
        "selected_soft_risk_exposure_total_mean": selected.get("oracle_soft_risk_exposure_total_mean") if selected else None,
        "primary_target_selected": bool(
            selected
            and selected.get("candidate_count") == config["primary_candidate_count"]
            and _float_equal(selected.get("path_cost_weight"), config["primary_path_cost_weight"])
        ),
        "primary_target_feasible": bool(primary and primary.get("oracle_target_feasible") is True),
        "primary_budget_passed": primary_budget_passed,
        "upper_bound_target_feasible": bool(upper and upper.get("oracle_target_feasible") is True),
        "primary": primary,
        "upper_bound": upper,
    }


def _preference_pair_audit(root_value: Any) -> list[dict[str, Any]]:
    if not root_value:
        return []
    root = Path(str(root_value))
    steps = _read_jsonl(root / COVERAGE_STEPS_FILE, [], "missing_steps")
    if not steps:
        return []
    by_policy: dict[str, dict[tuple[str, int], dict[str, Any]]] = {POLICY_ORACLE: {}, POLICY_XUNCE: {}, POLICY_INCUMBENT: {}}
    for row in steps:
        policy = row.get("policy")
        if policy not in by_policy:
            continue
        scenario_id = str(row.get("scenario_id"))
        step_index = _safe_int(row.get("step_index"))
        if step_index is None:
            continue
        by_policy[policy][(scenario_id, step_index)] = row
    rows: list[dict[str, Any]] = []
    for key, oracle in sorted(by_policy[POLICY_ORACLE].items()):
        for baseline_policy in (POLICY_XUNCE, POLICY_INCUMBENT):
            baseline = by_policy[baseline_policy].get(key)
            if not baseline:
                continue
            oracle_selected = oracle.get("selected_action_index")
            baseline_selected = baseline.get("selected_action_index")
            if oracle_selected == baseline_selected and oracle.get("candidate_set_hash") == baseline.get("candidate_set_hash"):
                continue
            oracle_coverage = _finite(oracle.get("new_covered_cell_count")) or _finite(oracle.get("raw_new_covered_cell_count")) or 0.0
            baseline_coverage = _finite(baseline.get("new_covered_cell_count")) or _finite(baseline.get("raw_new_covered_cell_count")) or 0.0
            oracle_cost = _finite(oracle.get("path_cost")) or 0.0
            baseline_cost = _finite(baseline.get("path_cost")) or 0.0
            oracle_hard_risk = _finite(oracle.get("hard_risk_violation_count")) or 0.0
            baseline_hard_risk = _finite(baseline.get("hard_risk_violation_count")) or 0.0
            rows.append(
                {
                    "schema_version": PREFERENCE_ROW_SCHEMA_VERSION,
                    "scenario_id": key[0],
                    "step_index": key[1],
                    "baseline_policy": baseline_policy,
                    "same_candidate_set": oracle.get("candidate_set_hash") == baseline.get("candidate_set_hash"),
                    "oracle_selected_action_index": oracle_selected,
                    "baseline_selected_action_index": baseline_selected,
                    "oracle_candidate_set_hash": oracle.get("candidate_set_hash"),
                    "baseline_candidate_set_hash": baseline.get("candidate_set_hash"),
                    "oracle_new_covered_cell_count": oracle_coverage,
                    "baseline_new_covered_cell_count": baseline_coverage,
                    "oracle_path_cost": oracle_cost,
                    "baseline_path_cost": baseline_cost,
                    "oracle_soft_risk_exposure": _finite(oracle.get("soft_risk_exposure")) or _finite(oracle.get("path_risk_exposure")) or 0.0,
                    "baseline_soft_risk_exposure": _finite(baseline.get("soft_risk_exposure")) or _finite(baseline.get("path_risk_exposure")) or 0.0,
                    "oracle_hard_risk_violation_count": oracle_hard_risk,
                    "baseline_hard_risk_violation_count": baseline_hard_risk,
                    "oracle_selected_higher_coverage": oracle_coverage > baseline_coverage,
                    "oracle_selected_lower_or_acceptable_cost": oracle_cost <= baseline_cost * 1.25 or oracle_coverage > baseline_coverage,
                    "hard_risk_clean_pair": oracle_hard_risk <= 0.0 and baseline_hard_risk <= 0.0,
                    "audit_only": True,
                    "not_training_data": True,
                }
            )
    return rows


def _critic_readiness(*, config: dict[str, Any], preference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    xunce_rows = [row for row in preference_rows if row.get("baseline_policy") == POLICY_XUNCE]
    hard_clean = [row for row in preference_rows if row.get("hard_risk_clean_pair") is True]
    same_candidate_hard_clean = [
        row
        for row in preference_rows
        if row.get("same_candidate_set") is True and row.get("hard_risk_clean_pair") is True
    ]
    same_candidate_xunce = [
        row
        for row in same_candidate_hard_clean
        if row.get("baseline_policy") == POLICY_XUNCE
    ]
    ready = (
        len(same_candidate_xunce) >= config["min_preference_pair_count"]
        and len(same_candidate_hard_clean) >= config["min_preference_pair_count"]
    )
    reasons = [] if ready else ["insufficient_reward_rerank_preference_pairs"]
    return {
        "schema_version": CRITIC_READINESS_SCHEMA_VERSION,
        "critic_target_ready": ready,
        "preference_pair_count": len(preference_rows),
        "oracle_xunce_disagreement_count": len(xunce_rows),
        "oracle_incumbent_disagreement_count": sum(1 for row in preference_rows if row.get("baseline_policy") == POLICY_INCUMBENT),
        "oracle_selected_higher_coverage_count": sum(1 for row in preference_rows if row.get("oracle_selected_higher_coverage") is True),
        "oracle_selected_lower_or_acceptable_cost_count": sum(
            1 for row in preference_rows if row.get("oracle_selected_lower_or_acceptable_cost") is True
        ),
        "hard_risk_clean_pair_count": len(hard_clean),
        "same_candidate_set_preference_pair_count": sum(1 for row in preference_rows if row.get("same_candidate_set") is True),
        "same_candidate_set_hard_risk_clean_pair_count": len(same_candidate_hard_clean),
        "same_candidate_set_xunce_pair_count": len(same_candidate_xunce),
        "min_preference_pair_count": config["min_preference_pair_count"],
        "reason_codes": reasons,
        "audit_only": True,
        "training_data_published": False,
    }


def _route(
    *,
    config: dict[str, Any],
    blocking: list[str],
    evaluator: dict[str, Any],
    target: dict[str, Any],
    critic: dict[str, Any],
) -> str:
    if blocking:
        if any("boundary" in reason or reason in BOUNDARY_FIELDS for reason in blocking):
            return ROUTE_BOUNDARY
        return ROUTE_INPUTS
    hard_risk = max(float(row.get("oracle_hard_risk_violation_count", 0.0) or 0.0) for row in evaluator.get("rows", []) or [{}])
    if hard_risk > config["max_hard_risk_violation_count"]:
        return ROUTE_RISK
    if not target.get("oracle_target_feasible"):
        return ROUTE_STAGE18_12
    if not target.get("primary_target_feasible") or not target.get("primary_budget_passed"):
        return ROUTE_CONTINUE
    if not critic.get("critic_target_ready"):
        return ROUTE_PREFERENCE
    return ROUTE_STAGE20


def _policy_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    capped = [_coverage_capped(row) for row in rows]
    raw = [_finite(row.get("final_coverage_rate")) for row in rows]
    costs = [_finite(row.get("path_cost_total_m")) or _finite(row.get("path_cost")) for row in rows]
    soft_risk = [_finite(row.get("soft_risk_exposure_total")) for row in rows]
    hard_risk = [_finite(row.get("hard_risk_violation_count")) or 0.0 for row in rows]
    saturation = [1.0 if row.get("coverage_saturation_exceeded") is True else 0.0 for row in rows]
    return {
        "episode_count": len(rows),
        "final_coverage_rate_capped_mean": _mean(capped),
        "final_coverage_rate_capped_min": _min(capped),
        "final_coverage_rate_capped_max": _max(capped),
        "final_coverage_rate_raw_mean": _mean(raw),
        "final_coverage_rate_raw_max": _max(raw),
        "path_cost_total_m_mean": _mean(costs),
        "path_cost_total_m_p95": _percentile(costs, 0.95),
        "soft_risk_exposure_total_mean": _mean(soft_risk),
        "soft_risk_exposure_total_p95": _percentile(soft_risk, 0.95),
        "hard_risk_violation_count": sum(hard_risk),
        "coverage_saturation_episode_count": int(sum(saturation)),
    }


def _coverage_capped(row: dict[str, Any]) -> float | None:
    for field in ("final_coverage_rate_capped", "coverage_rate_capped"):
        value = _finite(row.get(field))
        if value is not None:
            return min(1.0, max(0.0, value))
    value = _finite(row.get("final_coverage_rate"))
    if value is None:
        return None
    return min(1.0, max(0.0, value))


def _boundary_violations(config: dict[str, Any]) -> list[str]:
    reasons = []
    for field in BOUNDARY_FIELDS:
        value = config.get(field)
        if field == "canary_traffic_fraction":
            if float(value or 0.0) != 0.0:
                reasons.append(field)
        elif value is True:
            reasons.append(field)
    return reasons


def _find_rollout(rows: list[dict[str, Any]], count: int, weight: float) -> dict[str, Any] | None:
    for row in rows:
        if row.get("candidate_count") == count and _float_equal(row.get("path_cost_weight"), weight):
            return row
    return None


def _read_json(path: Path, reasons: list[str], reason_code: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(reason_code)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(reason_code.replace("missing", "invalid"))
        return {}
    if not isinstance(value, dict):
        reasons.append(reason_code.replace("missing", "invalid"))
        return {}
    return value


def _read_jsonl(path: Path, reasons: list[str], reason_code: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(reason_code)
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, json.JSONDecodeError):
        reasons.append(reason_code.replace("missing", "invalid"))
        return []
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _render_report(summary: dict[str, Any]) -> str:
    target = summary["practical_target_selection"]
    critic = summary["critic_target_readiness"]
    return "\n".join(
        [
            "# Xunce Stage 19 Evaluator / Critic Preflight",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- oracle_target_feasible: `{summary['oracle_target_feasible']}`",
            f"- primary_target_selected: `{summary['primary_target_selected']}`",
            f"- selected_candidate_count: `{target.get('selected_candidate_count')}`",
            f"- selected_path_cost_weight: `{target.get('selected_path_cost_weight')}`",
            f"- xunce_checkpoint_advantage_established: `{summary['xunce_checkpoint_advantage_established']}`",
            f"- preference_pair_count: `{critic['preference_pair_count']}`",
            f"- stage20_authorized: `{summary['stage20_authorized']}`",
            "",
            "Stage 19 is a human-review-only preflight. It validates oracle feasibility and critic target readiness, but does not train PPO, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _finite_required(payload: dict[str, Any], field: str) -> float:
    value = _finite(payload.get(field))
    if value is None:
        raise ConfigError(f"{field} must be a finite number")
    return value


def _int_required(payload: dict[str, Any], field: str) -> int:
    value = _safe_int(payload.get(field))
    if value is None:
        raise ConfigError(f"{field} must be an integer")
    return value


def _str_required(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    return _finite(value)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _mean(values: Iterable[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return mean(cleaned) if cleaned else None


def _min(values: Iterable[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return min(cleaned) if cleaned else None


def _max(values: Iterable[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return max(cleaned) if cleaned else None


def _percentile(values: Iterable[float | None], quantile: float) -> float | None:
    cleaned = sorted(value for value in values if value is not None)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    index = int(round((len(cleaned) - 1) * quantile))
    return cleaned[max(0, min(index, len(cleaned) - 1))]


def _gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _float_equal(left: Any, right: Any) -> bool:
    left_value = _finite(left)
    right_value = _finite(right)
    return left_value is not None and right_value is not None and abs(left_value - right_value) <= 1e-9


if __name__ == "__main__":
    raise SystemExit(main())
