from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_xunce_stage21_6_multi_seed_ppo_pilot import (  # noqa: E402
    run_xunce_stage21_6_multi_seed_ppo_pilot,
)


CONFIG_SCHEMA_VERSION = "xunce-stage21-7-reward-collector-advantage-horizon-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-7-diagnostic-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-7-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-7-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_7_reward_collector_advantage_horizon_repair_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_7_reward_collector_advantage_horizon_repair_v1"
)

ROUTE_INPUTS = "rerun_stage21_6_required_inputs"
ROUTE_REWARD = "repair_stage21_coverage_first_reward_signal"
ROUTE_ADVANTAGE = "repair_stage21_advantage_computation"
ROUTE_SCALE = "scale_stage21_collector_and_holdout_horizon"
ROUTE_UPDATE = "calibrate_stage21_ppo_update_strength"
ROUTE_RERUN_21_6 = "rerun_stage21_6_multi_seed_pilot_with_repaired_config"
ROUTE_CONTINUE = "continue_stage21_7_reward_collector_advantage_horizon_repair"

SUMMARY_FILE = "xunce-stage21-7-diagnostic-summary.json"
SEED_DIAGNOSTICS_FILE = "xunce-stage21-7-seed-diagnostics.jsonl"
REWARD_AUDIT_FILE = "xunce-stage21-7-reward-signal-audit.json"
ADVANTAGE_AUDIT_FILE = "xunce-stage21-7-advantage-signal-audit.json"
POLICY_AUDIT_FILE = "xunce-stage21-7-policy-shift-audit.json"
REPAIRED_CONFIG_FILE = "xunce-stage21-7-repaired-stage21-6-config.json"
REPAIRED_STAGE21_4_CONFIG_FILE = "xunce-stage21-7-repaired-stage21-4-base-config.json"
REPAIRED_RESULT_FILE = "xunce-stage21-7-repaired-stage21-6-result-summary.json"
ROUTING_FILE = "xunce-stage21-7-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-7-report.md"
MANIFEST_FILE = "xunce-stage21-7-manifest.json"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "stage21_6_authorized",
    "training_or_release_authorized",
)

CONFIG_BOUNDARY_FIELDS = (
    "stage21_7_authorized",
    "training_or_release_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.7 PPO no-uplift repair diagnostics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "primary_root_cause": summary["primary_root_cause"],
                "diagnostic_reason_codes": summary["diagnostic_reason_codes"],
                "repaired_pilot_status": summary.get("repaired_pilot_status"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config_path = _resolve_path(config_path, repo_root)
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage21_6_root = Path(config["stage21_6_root"])
    input_reasons, stage21_6 = _load_stage21_6(stage21_6_root)
    seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl") if not input_reasons else []
    seed_diagnostics = [_seed_diagnostic(row, config) for row in seed_rows]

    reward_audit = _reward_audit(seed_diagnostics, config)
    advantage_audit = _advantage_audit(seed_diagnostics, config)
    policy_audit = _policy_shift_audit(seed_diagnostics, config)
    diagnostic_reason_codes = _diagnostic_reasons(stage21_6, reward_audit, advantage_audit, policy_audit, config)
    boundary_reasons = _boundary_reasons(stage21_6) + _config_boundary_reasons(config)

    repaired_config_path, repaired_stage21_4_path = _write_repaired_configs(config, output_root, repo_root)
    repaired_result = _maybe_run_repaired_pilot(config, repaired_config_path, output_root, repo_root, input_reasons, boundary_reasons)
    status, route, primary_root_cause = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        diagnostic_reason_codes=diagnostic_reason_codes,
        repaired_result=repaired_result,
    )

    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        stage21_6=stage21_6,
        seed_diagnostics=seed_diagnostics,
        reward_audit=reward_audit,
        advantage_audit=advantage_audit,
        policy_audit=policy_audit,
        repaired_config_path=repaired_config_path,
        repaired_stage21_4_path=repaired_stage21_4_path,
        repaired_result=repaired_result,
        status=status,
        route=route,
        primary_root_cause=primary_root_cause,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        diagnostic_reason_codes=diagnostic_reason_codes,
    )


def _load_stage21_6(root: Path) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    summary = _read_json_or_empty(root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
    aggregate = _read_json_or_empty(root / "xunce-stage21-6-aggregate-metrics.json")
    lineage = _read_json_or_empty(root / "xunce-stage21-6-lineage-audit.json")
    routing = _read_json_or_empty(root / "xunce-stage21-6-next-stage-routing.json")
    if not summary:
        reasons.append("missing_stage21_6_summary")
    if not aggregate:
        reasons.append("missing_stage21_6_aggregate")
    if not lineage:
        reasons.append("missing_stage21_6_lineage")
    if not routing:
        reasons.append("missing_stage21_6_routing")
    if summary and summary.get("next_required_change") != "repair_stage21_6_reward_collector_advantage_or_horizon":
        reasons.append("stage21_6_route_not_repair_reward_collector_advantage_or_horizon")
    if lineage and lineage.get("passed") is not True:
        reasons.append("stage21_6_lineage_not_passed")
    return reasons, {"summary": summary, "aggregate": aggregate, "lineage": lineage, "routing": routing}


def _seed_diagnostic(seed_row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    seed_root = Path(str(seed_row["seed_root"]))
    stage21_3_root = seed_root / "stage21_3"
    stage21_4_root = seed_root / "stage21_4"
    stage21_5_root = seed_root / "stage21_5"
    batch_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    loss_rows = _read_jsonl(stage21_4_root / "xunce-stage21-4-ppo-loss-audit.jsonl")
    summary4 = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    summary5 = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
    trajectory_delta = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-trajectory-delta.json")
    pre_rows = _read_jsonl(Path(str(summary5.get("pre_evaluation_root", ""))) / "xunce-exploration-coverage-model-inference.jsonl")
    post_rows = _read_jsonl(Path(str(summary5.get("post_evaluation_root", ""))) / "xunce-exploration-coverage-model-inference.jsonl")

    reward_values = [_float(row.get("reward")) for row in batch_rows]
    reward_values = [value for value in reward_values if value is not None]
    coverage_values = [_coverage_value(row) for row in batch_rows]
    advantage_values = [_float(row.get("advantage")) for row in batch_rows]
    advantage_values = [value for value in advantage_values if value is not None]
    raw_advantage_values = [_float(row.get("raw_advantage")) for row in batch_rows]
    raw_advantage_values = [value for value in raw_advantage_values if value is not None]
    step_coverage_components = [
        _float((row.get("reward_components") or {}).get("step_coverage_gain_component"))
        for row in batch_rows
    ]
    step_coverage_components = [value for value in step_coverage_components if value is not None]
    policy_shift = _compare_policy_inference(pre_rows, post_rows)
    max_abs_kl = max((abs(_float(row.get("post_update_approx_kl")) or 0.0) for row in loss_rows), default=0.0)
    ratio_min = min((_float(row.get("ratio_min")) or 1.0 for row in loss_rows), default=1.0)
    ratio_max = max((_float(row.get("ratio_max")) or 1.0 for row in loss_rows), default=1.0)

    return {
        "schema_version": "xunce-stage21-7-seed-diagnostic/v1",
        "seed": seed_row.get("seed"),
        "sampling_seed": seed_row.get("sampling_seed"),
        "seed_root": str(seed_root),
        "trainable_transition_count": len(batch_rows),
        "reward_mean": _mean(reward_values),
        "reward_std": _std(reward_values),
        "reward_min": min(reward_values) if reward_values else None,
        "reward_max": max(reward_values) if reward_values else None,
        "reward_coverage_correlation": _correlation(reward_values, coverage_values),
        "step_coverage_component_nonzero_rate": _nonzero_rate(step_coverage_components),
        "advantage_mean": _mean(advantage_values),
        "advantage_std": _std(advantage_values),
        "advantage_positive_fraction": _fraction(advantage_values, lambda value: value > 0.0),
        "advantage_negative_fraction": _fraction(advantage_values, lambda value: value < 0.0),
        "advantage_coverage_correlation": _correlation(advantage_values, coverage_values),
        "raw_advantage_std": _std(raw_advantage_values),
        "raw_advantage_coverage_correlation": _correlation(raw_advantage_values, coverage_values),
        "max_abs_post_update_kl": max_abs_kl,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "parameter_delta_l2": _float(summary4.get("parameter_delta_l2")),
        "coverage_delta": _float(summary5.get("final_coverage_rate_delta")),
        "coverage_auc_delta": _float(summary5.get("coverage_curve_auc_delta")),
        "rollout_steps": _int(summary5.get("rollout_steps")),
        "required_scenario_count": _int(summary5.get("required_scenario_count")),
        "trajectory_delta": trajectory_delta,
        **policy_shift,
    }


def _reward_audit(seed_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    reward_std_values = [_float(row.get("reward_std")) for row in seed_rows]
    correlations = [_float(row.get("reward_coverage_correlation")) for row in seed_rows]
    nonzero_rates = [_float(row.get("step_coverage_component_nonzero_rate")) for row in seed_rows]
    mean_std = _mean([value for value in reward_std_values if value is not None])
    mean_corr = _mean([value for value in correlations if value is not None])
    mean_nonzero = _mean([value for value in nonzero_rates if value is not None])
    reason_codes: list[str] = []
    if mean_std is None or mean_std < float(config["min_reward_std"]):
        reason_codes.append("reward_std_too_low")
    if mean_nonzero is None or mean_nonzero < float(config["min_coverage_reward_nonzero_rate"]):
        reason_codes.append("coverage_reward_nonzero_rate_too_low")
    if mean_corr is None or mean_corr < float(config["min_reward_coverage_correlation"]):
        reason_codes.append("reward_coverage_correlation_too_low")
    return {
        "schema_version": "xunce-stage21-7-reward-signal-audit/v1",
        "reward_std_mean": mean_std,
        "reward_coverage_correlation_mean": mean_corr,
        "step_coverage_component_nonzero_rate_mean": mean_nonzero,
        "weak_or_misaligned_reward_signal": bool(reason_codes),
        "reason_codes": reason_codes,
    }


def _advantage_audit(seed_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    advantage_std_values = [_float(row.get("advantage_std")) for row in seed_rows]
    correlations = [_float(row.get("advantage_coverage_correlation")) for row in seed_rows]
    pos = [_float(row.get("advantage_positive_fraction")) for row in seed_rows]
    neg = [_float(row.get("advantage_negative_fraction")) for row in seed_rows]
    mean_std = _mean([value for value in advantage_std_values if value is not None])
    mean_corr = _mean([value for value in correlations if value is not None])
    mean_pos = _mean([value for value in pos if value is not None])
    mean_neg = _mean([value for value in neg if value is not None])
    reason_codes: list[str] = []
    if mean_std is None or mean_std < float(config["min_advantage_std"]):
        reason_codes.append("advantage_std_too_low")
    if mean_corr is None or mean_corr < float(config["min_advantage_coverage_correlation"]):
        reason_codes.append("advantage_coverage_correlation_too_low")
    if mean_pos in {0.0, 1.0} or mean_neg in {0.0, 1.0}:
        reason_codes.append("advantage_sign_distribution_extreme")
    return {
        "schema_version": "xunce-stage21-7-advantage-signal-audit/v1",
        "advantage_std_mean": mean_std,
        "advantage_coverage_correlation_mean": mean_corr,
        "advantage_positive_fraction_mean": mean_pos,
        "advantage_negative_fraction_mean": mean_neg,
        "flat_or_misaligned_advantage": bool(reason_codes),
        "reason_codes": reason_codes,
    }


def _policy_shift_audit(seed_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    kl_values = [_float(row.get("max_abs_post_update_kl")) for row in seed_rows]
    prob_deltas = [_float(row.get("selected_probability_abs_delta_mean")) for row in seed_rows]
    action_change_rates = [_float(row.get("selected_action_change_rate")) for row in seed_rows]
    parameter_deltas = [_float(row.get("parameter_delta_l2")) for row in seed_rows]
    mean_kl = _mean([value for value in kl_values if value is not None])
    mean_prob_delta = _mean([value for value in prob_deltas if value is not None])
    mean_action_change = _mean([value for value in action_change_rates if value is not None])
    mean_param_delta = _mean([value for value in parameter_deltas if value is not None])
    reason_codes: list[str] = []
    if (mean_kl is None or mean_kl < float(config["min_abs_kl_for_policy_shift"])) and (
        mean_prob_delta is None or mean_prob_delta < float(config["min_selected_probability_delta"])
    ) and (mean_action_change is None or mean_action_change <= 0.0):
        reason_codes.append("policy_shift_too_small")
    if mean_param_delta is None or mean_param_delta < float(config["min_parameter_delta_l2"]):
        reason_codes.append("parameter_delta_too_small")
    return {
        "schema_version": "xunce-stage21-7-policy-shift-audit/v1",
        "max_abs_post_update_kl_mean": mean_kl,
        "selected_probability_abs_delta_mean": mean_prob_delta,
        "selected_action_change_rate_mean": mean_action_change,
        "parameter_delta_l2_mean": mean_param_delta,
        "ppo_update_too_small": bool(reason_codes),
        "reason_codes": reason_codes,
    }


def _diagnostic_reasons(
    stage21_6: dict[str, Any],
    reward_audit: dict[str, Any],
    advantage_audit: dict[str, Any],
    policy_audit: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    aggregate = stage21_6.get("aggregate", {})
    if _int(aggregate.get("trainable_transition_count_total")) < int(config["min_transition_count_for_performance_claim"]):
        reasons.append("insufficient_trainable_transition_count")
    if reward_audit.get("weak_or_misaligned_reward_signal"):
        reasons.append("weak_or_misaligned_reward_signal")
    if advantage_audit.get("flat_or_misaligned_advantage"):
        reasons.append("flat_or_misaligned_advantage")
    if policy_audit.get("ppo_update_too_small"):
        reasons.append("ppo_update_too_small")
    if int(config["current_holdout_rollout_steps"]) < int(config["min_rollout_steps_for_horizon_confidence"]):
        reasons.append("holdout_horizon_too_short")
    return _unique(reasons)


def _boundary_reasons(stage21_6: dict[str, Any]) -> list[str]:
    summary = stage21_6.get("summary", {})
    aggregate = stage21_6.get("aggregate", {})
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if summary.get(field) is True:
            reasons.append(f"stage21_6_{field}_true")
    if float(summary.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_6_canary_traffic_fraction_nonzero")
    for field in (
        "hard_risk_violation_total",
        "safety_boundary_violation_total",
        "execution_boundary_violation_total",
        "model_inference_failure_total",
        "model_inference_mask_violation_total",
        "open_grid_fallback_total",
        "path_planning_failure_total",
        "unreachable_selected_total",
        "checkpoint_reload_failed_count",
        "checkpoint_not_experimental_only_count",
        "seed_release_boundary_violation_total",
    ):
        if _int(aggregate.get(field) or summary.get(field)) > 0:
            reasons.append(f"stage21_6_{field}_nonzero")
    return _unique(reasons)


def _config_boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in CONFIG_BOUNDARY_FIELDS:
        if config.get(field) is True:
            reasons.append(f"stage21_7_config_{field}_true")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_7_config_canary_traffic_fraction_nonzero")
    return _unique(reasons)


def _write_repaired_configs(config: dict[str, Any], output_root: Path, repo_root: Path) -> tuple[Path, Path]:
    stage21_4_base = _read_json(Path(config["stage21_4_base_config"]))
    stage21_4_base["epochs"] = int(config["repaired_stage21_4_epochs"])
    stage21_4_base["learning_rate"] = float(config["repaired_stage21_4_learning_rate"])
    stage21_4_path = output_root / REPAIRED_STAGE21_4_CONFIG_FILE
    _write_json(stage21_4_path, stage21_4_base)

    stage21_6_base = _read_json(Path(config["stage21_6_base_config"]))
    stage21_6_base.update(
        {
            "stage21_4_base_config": str(stage21_4_path),
            "seed_list": config["repaired_seed_list"],
            "required_scenario_count": int(config["repaired_required_scenario_count"]),
            "rollout_steps": int(config["repaired_rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["repaired_dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["repaired_dynamic_proposal_pool_limit_per_step"]),
            "execute_seed_pipeline": True,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    config_path = output_root / REPAIRED_CONFIG_FILE
    _write_json(config_path, stage21_6_base)
    return config_path, stage21_4_path


def _maybe_run_repaired_pilot(
    config: dict[str, Any],
    repaired_config_path: Path,
    output_root: Path,
    repo_root: Path,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    if input_reasons or boundary_reasons:
        return {"executed": False, "status": "blocked", "reason_codes": input_reasons + boundary_reasons}
    if not config["execute_repaired_pilot"]:
        return {"executed": False, "status": "not_executed", "reason_codes": ["execute_repaired_pilot_false"]}
    repaired_output_root = Path(config["repaired_stage21_6_output_root"])
    if config.get("reuse_completed_repaired_pilot_result"):
        existing = _repaired_pilot_result_from_summary(repaired_output_root)
        if existing:
            return existing
    try:
        summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
            config_path=repaired_config_path,
            output_root=repaired_output_root,
            repo_root=repo_root,
        )
    except Exception as exc:  # noqa: BLE001 - artifact records bounded execution blocker
        return {
            "executed": True,
            "status": "blocked",
            "reason_codes": ["stage21_7_repair_attempt_runtime_budget_blocked", type(exc).__name__],
            "error": str(exc),
            "output_root": str(repaired_output_root),
        }
    return {
        "executed": True,
        "status": summary.get("status"),
        "next_required_change": summary.get("next_required_change"),
        "mean_final_coverage_delta": summary.get("mean_final_coverage_delta"),
        "mean_coverage_auc_delta": summary.get("mean_coverage_auc_delta"),
        "sample_count_too_low_for_performance_claim": summary.get("sample_count_too_low_for_performance_claim"),
        "reason_codes": summary.get("reason_codes", []),
        "summary": summary.get("summary"),
        "output_root": str(repaired_output_root),
    }


def _repaired_pilot_result_from_summary(repaired_output_root: Path) -> dict[str, Any]:
    summary = _read_json_or_empty(repaired_output_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
    if not summary:
        return {}
    return {
        "executed": True,
        "reused_existing_result": True,
        "status": summary.get("status"),
        "next_required_change": summary.get("next_required_change"),
        "mean_final_coverage_delta": summary.get("mean_final_coverage_delta"),
        "mean_coverage_auc_delta": summary.get("mean_coverage_auc_delta"),
        "sample_count_too_low_for_performance_claim": summary.get("sample_count_too_low_for_performance_claim"),
        "reason_codes": summary.get("reason_codes", []),
        "summary": str(repaired_output_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json"),
        "output_root": str(repaired_output_root),
    }


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    diagnostic_reason_codes: list[str],
    repaired_result: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons or boundary_reasons:
        return "failed", ROUTE_INPUTS, "input_or_boundary"
    if repaired_result.get("executed") and repaired_result.get("status") == "passed":
        return "passed", ROUTE_RERUN_21_6, "repaired_pilot_improved"
    if repaired_result.get("executed") and repaired_result.get("status") == "failed":
        repaired_reasons = repaired_result.get("reason_codes", [])
        if repaired_result.get("next_required_change") == "repair_stage21_6_ppo_numerical_stability" or any(
            "grad_unstable" in str(reason) for reason in repaired_reasons
        ):
            return "failed", ROUTE_UPDATE, "repaired_pilot_ppo_numerical_stability"
        return "failed", ROUTE_CONTINUE, "repaired_pilot_still_no_uplift"
    if "weak_or_misaligned_reward_signal" in diagnostic_reason_codes:
        return "failed", ROUTE_REWARD, "reward"
    if "flat_or_misaligned_advantage" in diagnostic_reason_codes:
        return "failed", ROUTE_ADVANTAGE, "advantage"
    if "insufficient_trainable_transition_count" in diagnostic_reason_codes or "holdout_horizon_too_short" in diagnostic_reason_codes:
        return "partial", ROUTE_SCALE, "collector_or_horizon"
    if "ppo_update_too_small" in diagnostic_reason_codes:
        return "partial", ROUTE_UPDATE, "ppo_update_strength"
    return "partial", ROUTE_CONTINUE, "inconclusive"


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    stage21_6: dict[str, Any],
    seed_diagnostics: list[dict[str, Any]],
    reward_audit: dict[str, Any],
    advantage_audit: dict[str, Any],
    policy_audit: dict[str, Any],
    repaired_config_path: Path,
    repaired_stage21_4_path: Path,
    repaired_result: dict[str, Any],
    status: str,
    route: str,
    primary_root_cause: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
    diagnostic_reason_codes: list[str],
) -> dict[str, Any]:
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "seed_diagnostics": output_root / SEED_DIAGNOSTICS_FILE,
        "reward": output_root / REWARD_AUDIT_FILE,
        "advantage": output_root / ADVANTAGE_AUDIT_FILE,
        "policy": output_root / POLICY_AUDIT_FILE,
        "repaired_config": repaired_config_path,
        "repaired_stage21_4_config": repaired_stage21_4_path,
        "repaired_result": output_root / REPAIRED_RESULT_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }
    _write_jsonl(paths["seed_diagnostics"], seed_diagnostics)
    _write_json(paths["reward"], reward_audit)
    _write_json(paths["advantage"], advantage_audit)
    _write_json(paths["policy"], policy_audit)
    _write_json(paths["repaired_result"], repaired_result)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_root_cause": primary_root_cause,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "diagnostic_reason_codes": diagnostic_reason_codes,
        "stage21_7_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(paths["routing"], routing)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "next_required_change": route,
        "primary_root_cause": primary_root_cause,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "diagnostic_reason_codes": diagnostic_reason_codes,
        "stage21_6_status": stage21_6.get("summary", {}).get("status"),
        "stage21_6_next_required_change": stage21_6.get("summary", {}).get("next_required_change"),
        "stage21_6_trainable_transition_count_total": stage21_6.get("aggregate", {}).get("trainable_transition_count_total"),
        "stage21_6_mean_final_coverage_delta": stage21_6.get("summary", {}).get("mean_final_coverage_delta"),
        "stage21_6_mean_coverage_auc_delta": stage21_6.get("summary", {}).get("mean_coverage_auc_delta"),
        "repaired_stage21_6_config": str(repaired_config_path),
        "repaired_stage21_6_output_root": str(config["repaired_stage21_6_output_root"]),
        "repaired_pilot_executed": bool(repaired_result.get("executed")),
        "repaired_pilot_status": repaired_result.get("status"),
        "repaired_pilot_next_required_change": repaired_result.get("next_required_change"),
        "stage21_7_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
    }
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_report(summary, reward_audit, advantage_audit, policy_audit, repaired_result), encoding="utf-8")
    _write_json(
        paths["manifest"],
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": summary["generated_at"],
            "config": str(config_path),
            "artifacts": {key: str(path) for key, path in paths.items()},
            "summary_status": status,
            "next_required_change": route,
        },
    )
    return summary


def _report(
    summary: dict[str, Any],
    reward_audit: dict[str, Any],
    advantage_audit: dict[str, Any],
    policy_audit: dict[str, Any],
    repaired_result: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Stage 21.7 Reward / Collector / Advantage / Horizon Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- primary_root_cause: `{summary['primary_root_cause']}`",
            f"- diagnostic_reason_codes: `{summary['diagnostic_reason_codes']}`",
            f"- reward_weak_or_misaligned: `{reward_audit['weak_or_misaligned_reward_signal']}`",
            f"- advantage_flat_or_misaligned: `{advantage_audit['flat_or_misaligned_advantage']}`",
            f"- ppo_update_too_small: `{policy_audit['ppo_update_too_small']}`",
            f"- repaired_pilot_status: `{repaired_result.get('status')}`",
            "",
            "Stage 21.7 is diagnostic and repair-oriented. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _compare_policy_inference(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]]) -> dict[str, Any]:
    post_by_key = {
        (row.get("scenario_id"), row.get("step_index"), row.get("candidate_set_hash")): row
        for row in post_rows
    }
    action_changes = 0
    rank_changes = 0
    prob_deltas: list[float] = []
    logits_l2: list[float] = []
    matched = 0
    for pre in pre_rows:
        key = (pre.get("scenario_id"), pre.get("step_index"), pre.get("candidate_set_hash"))
        post = post_by_key.get(key)
        if not post:
            continue
        matched += 1
        pre_detail = pre.get("detail") if isinstance(pre.get("detail"), dict) else {}
        post_detail = post.get("detail") if isinstance(post.get("detail"), dict) else {}
        if pre_detail.get("selected_action_index") != post_detail.get("selected_action_index"):
            action_changes += 1
        if pre_detail.get("selected_rank") != post_detail.get("selected_rank"):
            rank_changes += 1
        pre_prob = _float(pre_detail.get("selected_probability"))
        post_prob = _float(post_detail.get("selected_probability"))
        if pre_prob is not None and post_prob is not None:
            prob_deltas.append(abs(post_prob - pre_prob))
        pre_logits = pre_detail.get("logits") if isinstance(pre_detail.get("logits"), list) else []
        post_logits = post_detail.get("logits") if isinstance(post_detail.get("logits"), list) else []
        if pre_logits and post_logits and len(pre_logits) == len(post_logits):
            sq = sum((float(a) - float(b)) ** 2 for a, b in zip(pre_logits, post_logits))
            logits_l2.append(math.sqrt(sq))
    return {
        "policy_shift_matched_step_count": matched,
        "selected_action_change_count": action_changes,
        "selected_action_change_rate": action_changes / matched if matched else None,
        "selected_rank_change_count": rank_changes,
        "selected_probability_abs_delta_mean": _mean(prob_deltas),
        "selected_probability_abs_delta_max": max(prob_deltas) if prob_deltas else None,
        "logits_l2_delta_mean": _mean(logits_l2),
        "logits_l2_delta_max": max(logits_l2) if logits_l2 else None,
    }


def _coverage_value(row: dict[str, Any]) -> float | None:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    for value in (
        info.get("coverage_rate_delta"),
        info.get("new_covered_cell_count"),
        (row.get("reward_components") or {}).get("step_coverage_gain_component") if isinstance(row.get("reward_components"), dict) else None,
    ):
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in (
        "stage21_6_root",
        "stage21_6_base_config",
        "stage21_4_base_config",
        "repaired_stage21_6_output_root",
    ):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["execute_repaired_pilot"] = bool(config.get("execute_repaired_pilot", True))
    config["reuse_completed_repaired_pilot_result"] = bool(config.get("reuse_completed_repaired_pilot_result", False))
    config["repaired_seed_list"] = [int(value) for value in config.get("repaired_seed_list", [2101, 2102, 2103])]
    for field in (
        "min_transition_count_for_performance_claim",
        "current_holdout_rollout_steps",
        "min_rollout_steps_for_horizon_confidence",
        "repaired_required_scenario_count",
        "repaired_rollout_steps",
        "repaired_dynamic_max_candidates_per_step",
        "repaired_dynamic_proposal_pool_limit_per_step",
        "repaired_stage21_4_epochs",
    ):
        config[field] = int(config[field])
    for field in (
        "min_reward_std",
        "min_reward_coverage_correlation",
        "min_coverage_reward_nonzero_rate",
        "min_advantage_std",
        "min_advantage_coverage_correlation",
        "min_abs_kl_for_policy_shift",
        "min_selected_probability_delta",
        "min_parameter_delta_l2",
        "repaired_stage21_4_learning_rate",
    ):
        config[field] = float(config[field])
    return config


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _nonzero_rate(values: list[float]) -> float | None:
    return sum(1 for value in values if abs(value) > 1.0e-12) / len(values) if values else None


def _fraction(values: list[float], predicate: Any) -> float | None:
    return sum(1 for value in values if predicate(value)) / len(values) if values else None


def _correlation(xs: list[float | None], ys: list[float | None]) -> float | None:
    paired = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(paired) < 2:
        return None
    x_values = [x for x, _ in paired]
    y_values = [y for _, y in paired]
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    x_var = sum((value - x_mean) ** 2 for value in x_values)
    y_var = sum((value - y_mean) ** 2 for value in y_values)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in paired)
    return cov / math.sqrt(x_var * y_var)


def _unique(values: list[str]) -> list[str]:
    return sorted(set(values))


if __name__ == "__main__":
    raise SystemExit(main())
