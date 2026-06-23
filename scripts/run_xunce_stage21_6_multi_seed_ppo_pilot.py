from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev, pvariance
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_xunce_stage21_1_on_policy_ppo_rollout_collector import (  # noqa: E402
    run_xunce_stage21_1_on_policy_ppo_rollout_collector,
)
from run_xunce_stage21_2_coverage_first_ppo_reward_contract import (  # noqa: E402
    run_xunce_stage21_2_coverage_first_ppo_reward_contract,
)
from run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation  # noqa: E402
from run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke  # noqa: E402
from run_xunce_stage21_5_post_update_offline_trajectory_evaluation import (  # noqa: E402
    run_xunce_stage21_5_post_update_offline_trajectory_evaluation,
)


CONFIG_SCHEMA_VERSION = "xunce-stage21-6-multi-seed-ppo-pilot-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-6-multi-seed-ppo-pilot-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-6-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-6-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_6_multi_seed_ppo_pilot_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_6_multi_seed_ppo_pilot_v1"
)

SUMMARY_FILE = "xunce-stage21-6-multi-seed-ppo-pilot-summary.json"
SEED_RESULTS_FILE = "xunce-stage21-6-seed-results.jsonl"
AGGREGATE_FILE = "xunce-stage21-6-aggregate-metrics.json"
LINEAGE_FILE = "xunce-stage21-6-lineage-audit.json"
ROUTING_FILE = "xunce-stage21-6-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-6-report.md"
MANIFEST_FILE = "xunce-stage21-6-manifest.json"

ROUTE_BOUNDARY = "resolve_stage21_6_multi_seed_boundary_rejections"
ROUTE_STAGE21_5 = "rerun_stage21_5_post_update_offline_trajectory_evaluation"
ROUTE_EXECUTION = "repair_stage21_6_seed_pipeline_execution"
ROUTE_HARD_RISK = "repair_stage21_6_hard_risk_or_execution_boundary_regression"
ROUTE_NUMERICS = "repair_stage21_6_ppo_numerical_stability"
ROUTE_LINEAGE = "repair_stage21_6_seed_lineage_or_batch_reuse"
ROUTE_REPAIR = "repair_stage21_6_reward_collector_advantage_or_horizon"
ROUTE_STAGE22 = "prepare_stage22_formal_pure_ppo_training_run"

BOUNDARY_FIELDS = (
    "stage21_6_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

EXECUTION_BOUNDARY_COUNT_FIELDS = (
    "model_inference_failure_count",
    "model_inference_mask_violation_count",
    "unreachable_selected_count",
    "path_planning_failure_count",
    "open_grid_fallback_count",
)

PER_SEED_RELEASE_BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "stage21_4_authorized",
    "training_or_release_authorized",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.6 multi-seed PPO pilot.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
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
                "seed_count": summary["seed_count"],
                "mean_final_coverage_delta": summary["mean_final_coverage_delta"],
                "worst_seed_id": summary["worst_seed_id"],
                "sample_count_too_low_for_performance_claim": summary["sample_count_too_low_for_performance_claim"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage21_6_multi_seed_ppo_pilot(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config_path = _resolve_path(config_path, repo_root)
    config = _load_config(config_path, repo_root=repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    boundary_reasons = _boundary_rejections(config)
    prerequisite_reasons = _stage21_5_prerequisite_rejections(Path(config["stage21_5_prerequisite_root"]))
    seed_results: list[dict[str, Any]] = []
    generated_configs: list[str] = []

    if not boundary_reasons and not prerequisite_reasons:
        for seed in config["seed_list"]:
            seed_root = output_root / f"seed_{int(seed)}"
            if config["execute_seed_pipeline"]:
                generated_configs.extend(_run_seed_pipeline(config, repo_root=repo_root, output_root=seed_root, seed=int(seed)))
            seed_results.append(_seed_result(seed_root, seed=int(seed)))

    aggregate = _aggregate_metrics(seed_results, config)
    lineage = _lineage_audit(seed_results)
    status, route, reason_codes = _route(config, boundary_reasons, prerequisite_reasons, seed_results, aggregate, lineage)

    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        seed_results=seed_results,
        aggregate=aggregate,
        lineage=lineage,
        generated_configs=generated_configs,
        status=status,
        route=route,
        reason_codes=reason_codes,
    )


def _run_seed_pipeline(config: dict[str, Any], *, repo_root: Path, output_root: Path, seed: int) -> list[str]:
    config_root = output_root / "generated_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    stage21_1_root = output_root / "stage21_1"
    stage21_2_root = output_root / "stage21_2"
    stage21_3_root = output_root / "stage21_3"
    stage21_4_root = output_root / "stage21_4"
    stage21_5_root = output_root / "stage21_5"

    config_paths: list[str] = []

    cfg1 = _read_json(Path(config["stage21_1_base_config"]))
    cfg1.update(
        {
            "sampling_seed": seed,
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "dynamic_validation_work_root": str(output_root / "_xunce_dynamic_validation_work_stage21_1"),
        }
    )
    cfg1_path = _write_seed_config(config_root / "stage21_1_config.json", cfg1)
    config_paths.append(str(cfg1_path))
    run_xunce_stage21_1_on_policy_ppo_rollout_collector(config_path=cfg1_path, output_root=stage21_1_root, repo_root=repo_root)

    cfg2 = _read_json(Path(config["stage21_2_base_config"]))
    cfg2["stage21_1_collector_root"] = str(stage21_1_root)
    cfg2_path = _write_seed_config(config_root / "stage21_2_config.json", cfg2)
    config_paths.append(str(cfg2_path))
    run_xunce_stage21_2_coverage_first_ppo_reward_contract(config_path=cfg2_path, output_root=stage21_2_root, repo_root=repo_root)

    cfg3 = _read_json(Path(config["stage21_3_base_config"]))
    cfg3["stage21_1_collector_root"] = str(stage21_1_root)
    cfg3["stage21_2_reward_contract_root"] = str(stage21_2_root)
    cfg3_path = _write_seed_config(config_root / "stage21_3_config.json", cfg3)
    config_paths.append(str(cfg3_path))
    run_xunce_stage21_3_ppo_batch_validation(config_path=cfg3_path, output_root=stage21_3_root, repo_root=repo_root)

    cfg4 = _read_json(Path(config["stage21_4_base_config"]))
    cfg4["stage21_3_ppo_batch_validation_root"] = str(stage21_3_root)
    cfg4["min_transition_count_for_performance_claim"] = int(config["min_transition_count_for_performance_claim"])
    cfg4_path = _write_seed_config(config_root / "stage21_4_config.json", cfg4)
    config_paths.append(str(cfg4_path))
    run_xunce_stage21_4_tiny_ppo_update_smoke(config_path=cfg4_path, output_root=stage21_4_root, repo_root=repo_root)

    cfg5 = _read_json(Path(config["stage21_5_base_config"]))
    cfg5.update(
        {
            "stage21_4_tiny_ppo_update_smoke_root": str(stage21_4_root),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
        }
    )
    cfg5_path = _write_seed_config(config_root / "stage21_5_config.json", cfg5)
    config_paths.append(str(cfg5_path))
    run_xunce_stage21_5_post_update_offline_trajectory_evaluation(config_path=cfg5_path, output_root=stage21_5_root, repo_root=repo_root)
    return config_paths


def _seed_result(root: Path, *, seed: int) -> dict[str, Any]:
    stage21_1_root = root / "stage21_1"
    stage21_3_root = root / "stage21_3"
    stage21_4_root = root / "stage21_4"
    stage21_5_root = root / "stage21_5"
    summary1 = _read_json_or_empty(stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json")
    summary3 = _read_json_or_empty(stage21_3_root / "xunce-stage21-3-ppo-batch-validation-summary.json")
    summary4 = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    summary5 = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
    post_eval = _read_json_or_empty(Path(str(summary5.get("post_evaluation_root", ""))) / "xunce-exploration-coverage-comparison-summary.json")
    gradient = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-gradient-audit.json")
    checkpoint = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-checkpoint-audit.json")
    loss_rows = _read_jsonl(stage21_4_root / "xunce-stage21-4-ppo-loss-audit.jsonl")
    batch_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    transition_ids = [str(row.get("transition_id", "")) for row in batch_rows]
    batch_fingerprint = _fingerprint_jsonl_rows(batch_rows)
    transition_fingerprint = _sha256_text("\n".join(sorted(transition_ids)))
    max_kl = max((_finite(row.get("post_update_approx_kl")) or 0.0 for row in loss_rows), default=None)
    min_entropy = min((_finite(row.get("entropy")) or 0.0 for row in loss_rows), default=None)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    return {
        "schema_version": "xunce-stage21-6-seed-result/v1",
        "seed": seed,
        "seed_root": str(root),
        "stage21_1_root": str(stage21_1_root),
        "stage21_3_root": str(stage21_3_root),
        "stage21_4_root": str(stage21_4_root),
        "stage21_5_root": str(stage21_5_root),
        "stage21_1_status": summary1.get("status"),
        "stage21_3_status": summary3.get("status"),
        "stage21_4_status": summary4.get("status"),
        "stage21_5_status": summary5.get("status"),
        "sampling_seed": summary1.get("sampling_seed", seed),
        "trainable_transition_count": _int_value(summary3.get("trainable_transition_count") or summary1.get("trainable_transition_count")),
        "stage21_3_batch_fingerprint": batch_fingerprint,
        "transition_id_fingerprint": transition_fingerprint,
        "transition_id_count": len([value for value in transition_ids if value]),
        "final_coverage_delta": _finite(summary5.get("final_coverage_rate_delta")),
        "coverage_auc_delta": _finite(summary5.get("coverage_curve_auc_delta")),
        "path_cost_delta_m": _finite(summary5.get("path_cost_total_m_delta")),
        "soft_risk_exposure_delta": _finite(summary5.get("soft_risk_exposure_total_delta")),
        "hard_risk_violation_count": _int_value(summary5.get("post_hard_risk_violation_count")),
        "safety_boundary_violation_count": _int_value(summary5.get("post_safety_boundary_violation_count")),
        "true_model_inference_executed": bool(post_eval.get("true_model_inference_executed")),
        "model_inference_failure_count": _int_value(post_eval.get("model_inference_failure_count")),
        "model_inference_mask_violation_count": _int_value(post_eval.get("model_inference_mask_violation_count")),
        "unreachable_selected_count": _int_value(post_eval.get("unreachable_selected_count")),
        "path_planning_failure_count": _int_value(post_eval.get("path_planning_failure_count")),
        "open_grid_fallback_count": _int_value(post_eval.get("open_grid_fallback_count")),
        "scenario_regression_count": _int_value(summary5.get("scenario_regression_count")),
        "scenario_safety_boundary_regression_count": _int_value(summary5.get("scenario_safety_boundary_regression_count")),
        "max_post_update_approx_kl": max_kl,
        "min_entropy": min_entropy,
        "grad_norm": _finite(gradient.get("grad_norm")),
        "pre_clip_grad_norm": _finite(gradient.get("pre_clip_grad_norm")),
        "post_clip_grad_norm": _finite(gradient.get("post_clip_grad_norm")),
        "grad_norm_finite": bool(gradient.get("grad_norm_finite")),
        "parameter_delta_l2": _finite(summary4.get("parameter_delta_l2")),
        "checkpoint_reload_passed": bool(checkpoint.get("checkpoint_reload_passed")),
        "experimental_only": metadata.get("experimental_only") is True,
        "experimental_checkpoint": summary4.get("experimental_checkpoint") is True,
        "stage21_4_authorized": summary4.get("stage21_4_authorized") is True,
        "training_or_release_authorized": summary4.get("training_or_release_authorized") is True or metadata.get("training_or_release_authorized") is True,
        "publishes_checkpoint": bool(summary4.get("publishes_checkpoint") or metadata.get("publishes_checkpoint")),
        "replaces_default_policy": bool(summary4.get("replaces_default_policy") or metadata.get("replaces_default_policy")),
        "connects_real_executor": bool(summary4.get("connects_real_executor") or metadata.get("connects_real_executor")),
        "starts_online_canary": bool(summary4.get("starts_online_canary") or metadata.get("starts_online_canary")),
        "canary_traffic_fraction": _finite(summary4.get("canary_traffic_fraction") or metadata.get("canary_traffic_fraction") or 0.0),
        "sample_count_too_low_for_performance_claim": bool(summary4.get("sample_count_too_low_for_performance_claim", True)),
    }


def _aggregate_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    coverage = [_finite(row.get("final_coverage_delta")) for row in rows]
    auc = [_finite(row.get("coverage_auc_delta")) for row in rows]
    path = [_finite(row.get("path_cost_delta_m")) for row in rows]
    soft_risk = [_finite(row.get("soft_risk_exposure_delta")) for row in rows]
    kl = [_finite(row.get("max_post_update_approx_kl")) for row in rows]
    entropy = [_finite(row.get("min_entropy")) for row in rows]
    grad = [_finite(row.get("pre_clip_grad_norm")) for row in rows]
    transitions = [int(row.get("trainable_transition_count", 0)) for row in rows]
    transition_total = sum(transitions)
    min_transition_count = int(config["min_transition_count_for_performance_claim"])
    valid_coverage = [float(value) for value in coverage if value is not None]
    worst_seed = None
    if rows:
        worst_seed = min(rows, key=lambda row: _finite(row.get("final_coverage_delta")) if _finite(row.get("final_coverage_delta")) is not None else -1.0e9).get("seed")
    return {
        "schema_version": "xunce-stage21-6-aggregate-metrics/v1",
        "seed_count": len(rows),
        "trainable_transition_count_total": transition_total,
        "trainable_transition_count_mean": _mean(transitions),
        "trainable_transition_count_min": min(transitions) if transitions else None,
        "trainable_transition_count_max": max(transitions) if transitions else None,
        **_distribution("final_coverage_delta", valid_coverage),
        **_distribution("coverage_auc_delta", [float(value) for value in auc if value is not None]),
        **_distribution("path_cost_delta_m", [float(value) for value in path if value is not None]),
        **_distribution("soft_risk_exposure_delta", [float(value) for value in soft_risk if value is not None]),
        **_distribution("max_post_update_approx_kl", [float(value) for value in kl if value is not None]),
        **_distribution("min_entropy", [float(value) for value in entropy if value is not None]),
        **_distribution("pre_clip_grad_norm", [float(value) for value in grad if value is not None]),
        "worst_seed_id": worst_seed,
        "hard_risk_violation_total": sum(_int_value(row.get("hard_risk_violation_count")) for row in rows),
        "safety_boundary_violation_total": sum(_int_value(row.get("safety_boundary_violation_count")) for row in rows),
        "model_inference_failure_total": sum(_int_value(row.get("model_inference_failure_count")) for row in rows),
        "model_inference_mask_violation_total": sum(_int_value(row.get("model_inference_mask_violation_count")) for row in rows),
        "unreachable_selected_total": sum(_int_value(row.get("unreachable_selected_count")) for row in rows),
        "path_planning_failure_total": sum(_int_value(row.get("path_planning_failure_count")) for row in rows),
        "open_grid_fallback_total": sum(_int_value(row.get("open_grid_fallback_count")) for row in rows),
        "execution_boundary_violation_total": sum(
            sum(_int_value(row.get(field)) for field in EXECUTION_BOUNDARY_COUNT_FIELDS)
            + (0 if row.get("true_model_inference_executed") is True else 1)
            for row in rows
        ),
        "scenario_regression_total": sum(_int_value(row.get("scenario_regression_count")) for row in rows),
        "scenario_safety_boundary_regression_total": sum(_int_value(row.get("scenario_safety_boundary_regression_count")) for row in rows),
        "checkpoint_reload_failed_count": sum(1 for row in rows if row.get("checkpoint_reload_passed") is not True),
        "checkpoint_experimental_only_count": sum(1 for row in rows if row.get("experimental_only") is True),
        "checkpoint_not_experimental_only_count": sum(1 for row in rows if row.get("experimental_only") is not True or row.get("experimental_checkpoint") is not True),
        "seed_release_boundary_violation_total": sum(
            sum(1 for field in PER_SEED_RELEASE_BOUNDARY_FIELDS if row.get(field) is True)
            + (1 if float(row.get("canary_traffic_fraction") or 0.0) > 0.0 else 0)
            for row in rows
        ),
        "seed_update_passed_count": sum(1 for row in rows if row.get("stage21_4_status") == "passed"),
        "all_seed_updates_passed": bool(rows) and all(row.get("stage21_4_status") == "passed" for row in rows),
        "min_transition_count_for_performance_claim": min_transition_count,
        "sample_count_too_low_for_performance_claim": (
            transition_total < min_transition_count
            or any(row.get("sample_count_too_low_for_performance_claim") is True for row in rows)
        ),
    }


def _lineage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [row.get("sampling_seed") for row in rows]
    stage21_1_roots = [row.get("stage21_1_root") for row in rows]
    batch_hashes = [row.get("stage21_3_batch_fingerprint") for row in rows]
    transition_hashes = [row.get("transition_id_fingerprint") for row in rows]
    duplicate_reasons: list[str] = []
    if _has_duplicates(seeds):
        duplicate_reasons.append("duplicate_sampling_seed")
    if _has_duplicates(stage21_1_roots):
        duplicate_reasons.append("duplicate_stage21_1_root")
    if _has_duplicates(batch_hashes):
        duplicate_reasons.append("duplicate_stage21_3_batch_fingerprint")
    if _has_duplicates(transition_hashes):
        duplicate_reasons.append("duplicate_transition_id_fingerprint")
    return {
        "schema_version": "xunce-stage21-6-lineage-audit/v1",
        "seed_count": len(rows),
        "sampling_seed_count": len(set(seeds)),
        "stage21_1_root_count": len(set(stage21_1_roots)),
        "stage21_3_batch_fingerprint_count": len(set(batch_hashes)),
        "transition_id_fingerprint_count": len(set(transition_hashes)),
        "passed": not duplicate_reasons,
        "reason_codes": duplicate_reasons,
    }


def _route(
    config: dict[str, Any],
    boundary_reasons: list[str],
    prerequisite_reasons: list[str],
    seed_results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    lineage: dict[str, Any],
) -> tuple[str, str, list[str]]:
    reasons = list(boundary_reasons + prerequisite_reasons)
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, reasons
    if prerequisite_reasons:
        return "failed", ROUTE_STAGE21_5, reasons
    if len(seed_results) != len(config["seed_list"]):
        reasons.append("seed_result_count_mismatch")
        return "failed", ROUTE_EXECUTION, reasons
    failed_stages = []
    for row in seed_results:
        for field in ("stage21_1_status", "stage21_3_status", "stage21_4_status", "stage21_5_status"):
            if row.get(field) != "passed":
                failed_stages.append(f"seed_{row.get('seed')}_{field}_not_passed")
    if failed_stages:
        reasons.extend(failed_stages)
        return "failed", ROUTE_EXECUTION, reasons
    if lineage.get("passed") is not True:
        reasons.extend(lineage.get("reason_codes", []))
        return "failed", ROUTE_LINEAGE, reasons
    seed_boundary_reasons = _seed_checkpoint_boundary_rejections(seed_results)
    if seed_boundary_reasons:
        reasons.extend(seed_boundary_reasons)
        return "failed", ROUTE_BOUNDARY, reasons
    boundary_count = (
        int(aggregate.get("hard_risk_violation_total", 0))
        + int(aggregate.get("safety_boundary_violation_total", 0))
        + int(aggregate.get("scenario_safety_boundary_regression_total", 0))
        + int(aggregate.get("execution_boundary_violation_total", 0))
    )
    if boundary_count > 0:
        reasons.append("hard_risk_or_execution_boundary_regression")
        return "failed", ROUTE_HARD_RISK, reasons
    numeric_reasons = _numeric_stability_rejections(seed_results, config)
    if numeric_reasons:
        reasons.extend(numeric_reasons)
        return "failed", ROUTE_NUMERICS, reasons
    if _coverage_or_cost_rejected(seed_results, aggregate, config):
        reasons.append("coverage_auc_worst_seed_or_cost_rejected")
        return "failed", ROUTE_REPAIR, reasons
    return "passed", ROUTE_STAGE22, reasons


def _seed_checkpoint_boundary_rejections(rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for row in rows:
        seed = row.get("seed")
        if row.get("checkpoint_reload_passed") is not True:
            reasons.append(f"seed_{seed}_checkpoint_reload_failed")
        if row.get("experimental_only") is not True or row.get("experimental_checkpoint") is not True:
            reasons.append(f"seed_{seed}_checkpoint_not_experimental_only")
        for field in PER_SEED_RELEASE_BOUNDARY_FIELDS:
            if row.get(field) is True:
                reasons.append(f"seed_{seed}_{field}_true")
        if float(row.get("canary_traffic_fraction") or 0.0) > 0.0:
            reasons.append(f"seed_{seed}_canary_traffic_fraction_nonzero")
    return reasons


def _numeric_stability_rejections(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for row in rows:
        seed = row.get("seed")
        kl = _finite(row.get("max_post_update_approx_kl"))
        entropy = _finite(row.get("min_entropy"))
        grad = _finite(row.get("pre_clip_grad_norm"))
        if kl is None or abs(float(kl)) > float(config["max_abs_approx_kl"]):
            reasons.append(f"seed_{seed}_kl_unstable")
        if entropy is None or float(entropy) < float(config["min_entropy"]):
            reasons.append(f"seed_{seed}_entropy_unstable")
        if grad is None or not bool(row.get("grad_norm_finite")) or float(grad) > float(config["max_grad_norm"]):
            reasons.append(f"seed_{seed}_grad_unstable")
    return reasons


def _coverage_or_cost_rejected(rows: list[dict[str, Any]], aggregate: dict[str, Any], config: dict[str, Any]) -> bool:
    mean_delta = _finite(aggregate.get("final_coverage_delta_mean"))
    mean_auc = _finite(aggregate.get("coverage_auc_delta_mean"))
    min_delta = _finite(aggregate.get("final_coverage_delta_min"))
    if mean_delta is None or mean_auc is None or min_delta is None:
        return True
    if float(mean_delta) <= float(config["min_mean_coverage_delta"]) or float(mean_auc) <= float(config["min_mean_coverage_delta"]):
        return True
    if float(min_delta) < float(config["min_worst_seed_coverage_delta"]):
        return True
    if int(aggregate.get("scenario_regression_total", 0)) > 0:
        return True
    for row in rows:
        path_delta = _finite(row.get("path_cost_delta_m"))
        soft_risk_delta = _finite(row.get("soft_risk_exposure_delta"))
        if path_delta is not None and float(path_delta) > float(config["max_path_cost_delta_m"]):
            return True
        if soft_risk_delta is not None and float(soft_risk_delta) > float(config["max_soft_risk_exposure_delta"]):
            return True
    return False


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    seed_results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    lineage: dict[str, Any],
    generated_configs: list[str],
    status: str,
    route: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "seed_results": output_root / SEED_RESULTS_FILE,
        "aggregate": output_root / AGGREGATE_FILE,
        "lineage": output_root / LINEAGE_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }
    _write_jsonl(paths["seed_results"], seed_results)
    paths["aggregate"].write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    paths["lineage"].write_text(json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    seed_update_passed_count = int(aggregate.get("seed_update_passed_count", 0) or 0)
    all_seed_updates_passed = bool(aggregate.get("all_seed_updates_passed"))
    offline_ppo_update_detected = bool(config["runs_new_ppo_update"] and seed_update_passed_count > 0)
    offline_ppo_update_executed = bool(config["runs_new_ppo_update"] and all_seed_updates_passed)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "reason_codes": _unique_sorted(reason_codes),
        "sample_count_too_low_for_performance_claim": bool(aggregate.get("sample_count_too_low_for_performance_claim", True)),
        "stage21_6_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": offline_ppo_update_executed,
        "offline_ppo_update_detected": offline_ppo_update_detected,
        "offline_ppo_update_executed": offline_ppo_update_executed,
        "seed_update_passed_count": seed_update_passed_count,
        "all_seed_updates_passed": all_seed_updates_passed,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    paths["routing"].write_text(json.dumps(routing, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "next_required_change": route,
        "reason_codes": _unique_sorted(reason_codes),
        "seed_count": len(seed_results),
        "seed_list": config["seed_list"],
        "mean_final_coverage_delta": aggregate.get("final_coverage_delta_mean"),
        "mean_coverage_auc_delta": aggregate.get("coverage_auc_delta_mean"),
        "worst_seed_id": aggregate.get("worst_seed_id"),
        "hard_risk_violation_total": aggregate.get("hard_risk_violation_total"),
        "safety_boundary_violation_total": aggregate.get("safety_boundary_violation_total"),
        "execution_boundary_violation_total": aggregate.get("execution_boundary_violation_total"),
        "checkpoint_reload_failed_count": aggregate.get("checkpoint_reload_failed_count"),
        "checkpoint_not_experimental_only_count": aggregate.get("checkpoint_not_experimental_only_count"),
        "seed_release_boundary_violation_total": aggregate.get("seed_release_boundary_violation_total"),
        "sample_count_too_low_for_performance_claim": bool(aggregate.get("sample_count_too_low_for_performance_claim", True)),
        "min_transition_count_for_performance_claim": aggregate.get("min_transition_count_for_performance_claim"),
        "trainable_transition_count_total": aggregate.get("trainable_transition_count_total"),
        "stage21_6_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": offline_ppo_update_executed,
        "offline_ppo_update_detected": offline_ppo_update_detected,
        "offline_ppo_update_executed": offline_ppo_update_executed,
        "seed_update_passed_count": seed_update_passed_count,
        "all_seed_updates_passed": all_seed_updates_passed,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(paths["summary"]),
        "seed_results": str(paths["seed_results"]),
        "aggregate_metrics": str(paths["aggregate"]),
        "lineage_audit": str(paths["lineage"]),
        "routing": str(paths["routing"]),
        "report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    paths["report"].write_text(_report(summary, aggregate), encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "config": str(config_path),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "generated_seed_configs": generated_configs,
        "summary_status": status,
        "next_required_change": route,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _report(summary: dict[str, Any], aggregate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.6 Multi-Seed PPO Pilot",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- seed_count: `{summary['seed_count']}`",
            f"- mean_final_coverage_delta: `{summary['mean_final_coverage_delta']}`",
            f"- mean_coverage_auc_delta: `{summary['mean_coverage_auc_delta']}`",
            f"- worst_seed_id: `{summary['worst_seed_id']}`",
            f"- hard_risk_violation_total: `{summary['hard_risk_violation_total']}`",
            f"- execution_boundary_violation_total: `{summary['execution_boundary_violation_total']}`",
            f"- checkpoint_reload_failed_count: `{summary['checkpoint_reload_failed_count']}`",
            f"- checkpoint_not_experimental_only_count: `{summary['checkpoint_not_experimental_only_count']}`",
            f"- sample_count_too_low_for_performance_claim: `{summary['sample_count_too_low_for_performance_claim']}`",
            f"- trainable_transition_count_total: `{summary['trainable_transition_count_total']}`",
            f"- offline_ppo_update_executed: `{summary['offline_ppo_update_executed']}`",
            "",
            "Stage 21.6 is an offline multi-seed PPO pilot. Checkpoints remain experimental-only and are not published, installed, connected to an executor, or used for canary traffic.",
            "",
        ]
    )


def _stage21_5_prerequisite_rejections(root: Path) -> list[str]:
    reasons: list[str] = []
    summary_path = root / "xunce-stage21-5-post-update-evaluation-summary.json"
    routing_path = root / "xunce-stage21-5-next-stage-routing.json"
    if not summary_path.is_file():
        return ["missing_stage21_5_summary"]
    if not routing_path.is_file():
        return ["missing_stage21_5_routing"]
    summary = _read_json(summary_path)
    routing = _read_json(routing_path)
    if summary.get("status") != "passed" or routing.get("status") != "passed":
        reasons.append("stage21_5_not_passed")
    if summary.get("next_required_change") != "implement_stage21_6_multi_seed_ppo_pilot":
        reasons.append("stage21_5_route_not_stage21_6")
    for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(field) is True or routing.get(field) is True:
            reasons.append(f"stage21_5_{field}_true")
    if float(summary.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_5_canary_traffic_fraction_nonzero")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if config.get(field) is True:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) > 0.0:
        reasons.append("canary_traffic_fraction")
    if config.get("runs_new_ppo_update") is not True:
        reasons.append("runs_new_ppo_update_not_true_for_offline_pilot")
    return reasons


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in (
        "stage21_5_prerequisite_root",
        "stage21_1_base_config",
        "stage21_2_base_config",
        "stage21_3_base_config",
        "stage21_4_base_config",
        "stage21_5_base_config",
    ):
        config[field] = str(_resolve_path(Path(config[field]), repo_root))
    config["execute_seed_pipeline"] = bool(config.get("execute_seed_pipeline", True))
    seeds = config.get("seed_list")
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("seed_list must be a non-empty list")
    config["seed_list"] = [_nonnegative_int(seed, "seed") for seed in seeds]
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 2), "required_scenario_count")
    config["rollout_steps"] = _positive_int(config.get("rollout_steps", 4), "rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(config.get("dynamic_max_candidates_per_step", 36), "dynamic_max_candidates_per_step")
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(config.get("dynamic_proposal_pool_limit_per_step", 288), "dynamic_proposal_pool_limit_per_step")
    config["min_transition_count_for_performance_claim"] = _positive_int(config.get("min_transition_count_for_performance_claim", 200), "min_transition_count_for_performance_claim")
    config["max_abs_approx_kl"] = _nonnegative_float(config.get("max_abs_approx_kl", 0.5), "max_abs_approx_kl")
    config["min_entropy"] = _nonnegative_float(config.get("min_entropy", 0.0), "min_entropy")
    config["max_grad_norm"] = _positive_float(config.get("max_grad_norm", 25.0), "max_grad_norm")
    config["max_path_cost_delta_m"] = _nonnegative_float(config.get("max_path_cost_delta_m", 20.0), "max_path_cost_delta_m")
    config["max_soft_risk_exposure_delta"] = _nonnegative_float(config.get("max_soft_risk_exposure_delta", 25.0), "max_soft_risk_exposure_delta")
    config["min_mean_coverage_delta"] = _nonnegative_float(config.get("min_mean_coverage_delta", 0.0), "min_mean_coverage_delta")
    config["min_worst_seed_coverage_delta"] = _nonnegative_float(config.get("min_worst_seed_coverage_delta", 0.0), "min_worst_seed_coverage_delta")
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return config


def _write_seed_config(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _distribution(prefix: str, values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_variance": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_std": pstdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_variance": pvariance(values) if len(values) > 1 else 0.0,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def _has_duplicates(values: list[Any]) -> bool:
    clean = [str(value) for value in values if value not in (None, "")]
    return len(clean) != len(set(clean))


def _fingerprint_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    return _sha256_text("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    return _read_json(path) if Path(path).is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if numeric <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return numeric


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a non-negative integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a non-negative integer") from exc
    if numeric < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return numeric


def _positive_float(value: Any, name: str) -> float:
    numeric = _finite(value)
    if numeric is None or numeric <= 0.0:
        raise ConfigError(f"{name} must be positive finite")
    return float(numeric)


def _nonnegative_float(value: Any, name: str) -> float:
    numeric = _finite(value)
    if numeric is None or numeric < 0.0:
        raise ConfigError(f"{name} must be non-negative finite")
    return float(numeric)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: list[Any]) -> float | None:
    numbers = [_finite(value) for value in values]
    finite = [float(value) for value in numbers if value is not None]
    return sum(finite) / len(finite) if finite else None


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


if __name__ == "__main__":
    raise SystemExit(main())
