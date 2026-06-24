from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)

from model_explorer.policy.coverage_first_reward import load_coverage_first_reward_profile  # noqa: E402
from run_xunce_stage21_6_multi_seed_ppo_pilot import run_xunce_stage21_6_multi_seed_ppo_pilot  # noqa: E402


CONFIG_SCHEMA_VERSION = "xunce-stage21-13-coverage-constrained-multi-seed-ppo-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-13-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-13-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-13-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke_v1"
)

SUMMARY_FILE = "xunce-stage21-13-summary.json"
STAGE21_6_CONFIG_FILE = "xunce-stage21-13-stage21-6-config.json"
STAGE21_6_RESULT_FILE = "xunce-stage21-13-stage21-6-result-summary.json"
ROUTING_FILE = "xunce-stage21-13-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-13-report.md"
MANIFEST_FILE = "xunce-stage21-13-manifest.json"

ROUTE_INPUTS = "rerun_stage21_13_required_inputs"
ROUTE_BOUNDARY = "resolve_stage21_13_boundary_rejections"
ROUTE_NUMERICS = "continue_stage21_9_gradient_normalization_loss_scaling_repair"
ROUTE_POLICY_SIGNAL = "calibrate_stage21_policy_update_signal_strength"
ROUTE_RETURN_ADVANTAGE = "repair_stage21_return_advantage_credit_assignment"
ROUTE_SCALE = "scale_stage21_ppo_pilot_scenarios_and_horizon"

V2_REWARD_PROFILE = "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json"

BOUNDARY_FIELDS = (
    "stage21_13_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

FIXED_STAGE21_6_FIELDS = {
    "seed_list": [2101, 2102, 2103],
    "required_scenario_count": 8,
    "rollout_steps": 10,
    "dynamic_max_candidates_per_step": 36,
    "dynamic_proposal_pool_limit_per_step": 288,
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.13 coverage-constrained multi-seed PPO smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
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
                "stage21_6_status": summary.get("stage21_6_status"),
                "mean_final_coverage_delta": summary.get("mean_final_coverage_delta"),
                "mean_coverage_auc_delta": summary.get("mean_coverage_auc_delta"),
                "mean_final_coverage_rate_capped_delta": summary.get("mean_final_coverage_rate_capped_delta"),
                "pre_clip_grad_norm_max": summary.get("pre_clip_grad_norm_max"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
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

    stage21_12_summary = _read_json_or_empty(Path(config["stage21_12_root"]) / "xunce-stage21-12-summary.json")
    recommended_config = _read_json_or_empty(Path(str(stage21_12_summary.get("recommended_stage21_6_config", ""))))
    input_reasons = _input_reasons(config, stage21_12_summary, recommended_config, repo_root)
    boundary_reasons = _boundary_reasons(config)
    stage21_6_config: Path | None = None
    run_reasons: list[str] = []
    stage21_6_summary: dict[str, Any] = {}
    stage21_6_aggregate: dict[str, Any] = {}
    stage21_6_lineage: dict[str, Any] = {}
    stage21_6_seed_rows: list[dict[str, Any]] = []
    capped_seed_rows: list[dict[str, Any]] = []

    if not input_reasons and not boundary_reasons:
        stage21_6_config = _write_stage21_6_config(recommended_config, output_root)
        if config["execute_stage21_6"]:
            run_reasons = _run_or_reuse_stage21_6(config, stage21_6_config, output_root, repo_root)
        stage21_6_root = _stage21_6_root(config, output_root)
        stage21_6_summary = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
        stage21_6_aggregate = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-aggregate-metrics.json")
        stage21_6_lineage = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-lineage-audit.json")
        stage21_6_seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl")
        capped_seed_rows = _capped_seed_rows(stage21_6_seed_rows)

    status, route, primary_reason = _route(
        config=config,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        run_reasons=run_reasons,
        stage21_6_summary=stage21_6_summary,
        stage21_6_aggregate=stage21_6_aggregate,
        stage21_6_lineage=stage21_6_lineage,
        stage21_6_seed_rows=stage21_6_seed_rows,
        capped_seed_rows=capped_seed_rows,
    )
    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        stage21_12_summary=stage21_12_summary,
        stage21_6_config=stage21_6_config,
        stage21_6_summary=stage21_6_summary,
        stage21_6_aggregate=stage21_6_aggregate,
        stage21_6_lineage=stage21_6_lineage,
        stage21_6_seed_rows=stage21_6_seed_rows,
        capped_seed_rows=capped_seed_rows,
        status=status,
        route=route,
        primary_reason=primary_reason,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        run_reasons=run_reasons,
    )


def _input_reasons(
    config: dict[str, Any],
    stage21_12_summary: dict[str, Any],
    recommended_config: dict[str, Any],
    repo_root: Path,
) -> list[str]:
    reasons: list[str] = []
    if not stage21_12_summary:
        return ["missing_stage21_12_summary"]
    if stage21_12_summary.get("status") != "passed":
        reasons.append("stage21_12_not_passed")
    if stage21_12_summary.get("next_required_change") != "run_stage21_13_coverage_constrained_multi_seed_ppo_smoke":
        reasons.append("stage21_12_route_not_stage21_13")
    recommended_path = Path(str(stage21_12_summary.get("recommended_stage21_6_config", "")))
    if not recommended_path.is_file():
        reasons.append("missing_stage21_12_recommended_stage21_6_config")
    if not recommended_config:
        reasons.append("unreadable_stage21_12_recommended_stage21_6_config")
        return sorted(set(reasons))
    if recommended_config.get("schema_version") != "xunce-stage21-6-multi-seed-ppo-pilot-config/v1":
        reasons.append("recommended_stage21_6_schema_invalid")
    for field, expected in FIXED_STAGE21_6_FIELDS.items():
        if recommended_config.get(field) != expected:
            reasons.append(f"recommended_stage21_6_{field}_not_expected")
    if recommended_config.get("stage21_2_base_config") != stage21_12_summary.get("recommended_stage21_2_config"):
        reasons.append("recommended_stage21_6_stage21_2_base_config_mismatch")
    reward_profile_path = _resolve_path(Path(str(recommended_config.get("reward_profile", ""))), repo_root)
    canonical_profile_path = _resolve_path(Path(V2_REWARD_PROFILE), repo_root)
    if reward_profile_path != canonical_profile_path:
        reasons.append("recommended_stage21_6_reward_profile_path_not_canonical_v2")
    if reward_profile_path.name != Path(V2_REWARD_PROFILE).name:
        reasons.append("recommended_stage21_6_reward_profile_not_v2")
    try:
        reward_profile = load_coverage_first_reward_profile(reward_profile_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        reasons.append(f"recommended_stage21_6_reward_profile_unreadable:{type(exc).__name__}")
        reward_profile = None
    if reward_profile is None or reward_profile.schema_version != "xunce-stage21-coverage-constrained-ppo-reward-profile/v2":
        reasons.append("recommended_stage21_6_reward_profile_not_v2")
    if reward_profile is not None:
        if reward_profile.profile_id != stage21_12_summary.get("profile_v2_id"):
            reasons.append("recommended_stage21_6_reward_profile_id_mismatch")
        if reward_profile.profile_version != stage21_12_summary.get("profile_v2_version"):
            reasons.append("recommended_stage21_6_reward_profile_version_mismatch")
        if reward_profile.profile_hash != stage21_12_summary.get("profile_v2_hash"):
            reasons.append("recommended_stage21_6_reward_profile_hash_mismatch")
    for field in ("stage21_6_authorized", "training_or_release_authorized", "publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if recommended_config.get(field) is True:
            reasons.append(f"recommended_stage21_6_{field}_true")
    if float(recommended_config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("recommended_stage21_6_canary_traffic_fraction_nonzero")
    if stage21_12_summary.get("v2_hard_risk_trainable_count") not in (0, 0.0):
        reasons.append("stage21_12_v2_hard_risk_trainable_count_nonzero")
    if not isinstance(config.get("stage21_6_output_subdir"), str):
        reasons.append("stage21_6_output_subdir_invalid")
    return sorted(set(reasons))


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_13_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_13_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _write_stage21_6_config(recommended_config: dict[str, Any], output_root: Path) -> Path:
    payload = dict(recommended_config)
    payload.update(
        {
            "execute_seed_pipeline": True,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "stage21_13_generated": True,
        }
    )
    path = output_root / STAGE21_6_CONFIG_FILE
    _write_json(path, payload)
    return path


def _run_or_reuse_stage21_6(config: dict[str, Any], stage21_6_config: Path, output_root: Path, repo_root: Path) -> list[str]:
    stage21_6_root = _stage21_6_root(config, output_root)
    existing = stage21_6_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json"
    if existing.is_file() and config["reuse_existing_stage21_6_result"]:
        return []
    try:
        run_xunce_stage21_6_multi_seed_ppo_pilot(config_path=stage21_6_config, output_root=stage21_6_root, repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001 - wrapper records execution blockers.
        _write_json(
            output_root / "xunce-stage21-13-stage21-6-runtime-blocker.json",
            {"status": "blocked", "reason_code": "stage21_13_stage21_6_execution_exception", "error": repr(exc)},
        )
        return ["stage21_13_stage21_6_execution_exception"]
    return []


def _route(
    *,
    config: dict[str, Any],
    input_reasons: list[str],
    boundary_reasons: list[str],
    run_reasons: list[str],
    stage21_6_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    stage21_6_lineage: dict[str, Any],
    stage21_6_seed_rows: list[dict[str, Any]],
    capped_seed_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    stage21_6_boundary = _stage21_6_boundary_reasons(stage21_6_summary, stage21_6_aggregate, stage21_6_seed_rows)
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage21_13_input_rejected"
    if boundary_reasons or stage21_6_boundary:
        return "failed", ROUTE_BOUNDARY, "stage21_13_boundary_rejected"
    if run_reasons:
        return "partial", ROUTE_INPUTS, "stage21_6_runtime_blocked"
    if not stage21_6_summary or not stage21_6_aggregate or not stage21_6_seed_rows:
        return "failed", ROUTE_INPUTS, "missing_stage21_6_outputs"
    if stage21_6_lineage.get("passed") is not True:
        return "failed", ROUTE_INPUTS, "stage21_6_lineage_not_passed"
    if not _all_seed_numerics_stable(stage21_6_seed_rows, config):
        return "failed", ROUTE_NUMERICS, "stage21_6_numerics_unstable"
    if not _policy_shift_observable(stage21_6_seed_rows, config):
        return "failed", ROUTE_POLICY_SIGNAL, "policy_shift_not_observable"
    capped = _capped_aggregate(capped_seed_rows)
    raw_final = _float(stage21_6_aggregate.get("final_coverage_delta_mean"))
    raw_auc = _float(stage21_6_aggregate.get("coverage_auc_delta_mean"))
    raw_final_min = _float(stage21_6_aggregate.get("final_coverage_delta_min"))
    raw_auc_min = _float(stage21_6_aggregate.get("coverage_auc_delta_min"))
    capped_final = _float(capped.get("final_coverage_rate_capped_delta_mean"))
    capped_auc = _float(capped.get("coverage_curve_auc_capped_delta_mean"))
    capped_final_min = _float(capped.get("final_coverage_rate_capped_delta_min"))
    capped_auc_min = _float(capped.get("coverage_curve_auc_capped_delta_min"))
    improves = all(
        value is not None and value > 0.0
        for value in (raw_final, raw_auc, raw_final_min, raw_auc_min, capped_final, capped_auc, capped_final_min, capped_auc_min)
    )
    if stage21_6_summary.get("status") == "passed" and improves:
        return "passed", ROUTE_SCALE, "coverage_constrained_smoke_improved"
    return "failed", ROUTE_RETURN_ADVANTAGE, "coverage_or_auc_not_improved"


def _stage21_6_boundary_reasons(summary: dict[str, Any], aggregate: dict[str, Any], seed_rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(field) is True:
            reasons.append(f"stage21_6_summary_{field}_true")
    if float(summary.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_6_summary_canary_traffic_fraction_nonzero")
    for field in (
        "hard_risk_violation_total",
        "model_inference_failure_total",
        "model_inference_mask_violation_total",
        "path_planning_failure_total",
        "open_grid_fallback_total",
        "safety_boundary_violation_total",
        "scenario_safety_boundary_regression_total",
        "execution_boundary_violation_total",
        "unreachable_selected_total",
        "seed_release_boundary_violation_total",
        "checkpoint_reload_failed_count",
        "checkpoint_not_experimental_only_count",
    ):
        if int(aggregate.get(field, 0) or 0) > 0:
            reasons.append(f"stage21_6_{field}_nonzero")
    for row in seed_rows:
        seed = row.get("seed")
        for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary", "stage21_4_authorized", "training_or_release_authorized"):
            if row.get(field) is True:
                reasons.append(f"seed_{seed}_{field}_true")
    return sorted(set(reasons))


def _all_seed_numerics_stable(seed_rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    if not seed_rows:
        return False
    for row in seed_rows:
        kl = _float(row.get("max_post_update_approx_kl"))
        entropy = _float(row.get("min_entropy"))
        grad = _float(row.get("pre_clip_grad_norm"))
        post_clip = _float(row.get("post_clip_grad_norm"))
        if row.get("grad_norm_finite") is not True:
            return False
        if kl is None or abs(kl) > float(config["max_abs_approx_kl"]):
            return False
        if entropy is None or entropy < float(config["min_entropy"]):
            return False
        if grad is None or grad > float(config["stage21_6_pre_clip_grad_norm_gate"]):
            return False
        if post_clip is None or post_clip > float(config["stage21_4_clip_grad_norm_limit"]) + 1.0e-5:
            return False
    return True


def _policy_shift_observable(seed_rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    threshold = float(config["min_parameter_delta_l2"])
    values = [_float(row.get("parameter_delta_l2")) for row in seed_rows]
    return bool(values) and all(value is not None and value > threshold for value in values)


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    stage21_12_summary: dict[str, Any],
    stage21_6_config: Path | None,
    stage21_6_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    stage21_6_lineage: dict[str, Any],
    stage21_6_seed_rows: list[dict[str, Any]],
    capped_seed_rows: list[dict[str, Any]],
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
    run_reasons: list[str],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stage21_6_boundary = _stage21_6_boundary_reasons(stage21_6_summary, stage21_6_aggregate, stage21_6_seed_rows)
    all_boundary_reasons = sorted(set(boundary_reasons + stage21_6_boundary))
    capped = _capped_aggregate(capped_seed_rows)
    parameter_deltas = [_float(row.get("parameter_delta_l2")) for row in stage21_6_seed_rows]
    stage21_6_result = {
        "stage21_6_summary": stage21_6_summary,
        "stage21_6_aggregate": stage21_6_aggregate,
        "stage21_6_lineage": stage21_6_lineage,
        "stage21_6_seed_count": len(stage21_6_seed_rows),
        "stage21_5_capped_coverage_aggregate": capped,
        "seed_result_keys": [
            {
                "seed": row.get("seed"),
                "pre_clip_grad_norm": row.get("pre_clip_grad_norm"),
                "post_clip_grad_norm": row.get("post_clip_grad_norm"),
                "parameter_delta_l2": row.get("parameter_delta_l2"),
                "final_coverage_delta": row.get("final_coverage_delta"),
                "coverage_auc_delta": row.get("coverage_auc_delta"),
                "stage21_4_status": row.get("stage21_4_status"),
                "stage21_5_status": row.get("stage21_5_status"),
            }
            for row in stage21_6_seed_rows
        ],
        "seed_capped_coverage_keys": capped_seed_rows,
    }
    _write_json(output_root / STAGE21_6_RESULT_FILE, stage21_6_result)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": all_boundary_reasons,
        "run_reason_codes": run_reasons,
        "stage21_6_lineage_passed": stage21_6_lineage.get("passed"),
        "stage21_13_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / ROUTING_FILE, routing)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_12_root": str(config["stage21_12_root"]),
        "stage21_12_status": stage21_12_summary.get("status"),
        "stage21_12_next_required_change": stage21_12_summary.get("next_required_change"),
        "stage21_6_config": str(stage21_6_config) if stage21_6_config else None,
        "stage21_6_root": str(_stage21_6_root(config, output_root)),
        "stage21_6_status": stage21_6_summary.get("status"),
        "stage21_6_next_required_change": stage21_6_summary.get("next_required_change"),
        "stage21_6_lineage_passed": stage21_6_lineage.get("passed"),
        "seed_count": stage21_6_aggregate.get("seed_count"),
        "trainable_transition_count_total": stage21_6_aggregate.get("trainable_transition_count_total"),
        "mean_final_coverage_delta": stage21_6_aggregate.get("final_coverage_delta_mean"),
        "mean_coverage_auc_delta": stage21_6_aggregate.get("coverage_auc_delta_mean"),
        "mean_final_coverage_rate_capped_delta": capped.get("final_coverage_rate_capped_delta_mean"),
        "mean_coverage_curve_auc_capped_delta": capped.get("coverage_curve_auc_capped_delta_mean"),
        "min_coverage_auc_delta": stage21_6_aggregate.get("coverage_auc_delta_min"),
        "min_final_coverage_rate_capped_delta": capped.get("final_coverage_rate_capped_delta_min"),
        "min_coverage_curve_auc_capped_delta": capped.get("coverage_curve_auc_capped_delta_min"),
        "worst_seed_id": stage21_6_aggregate.get("worst_seed_id"),
        "pre_clip_grad_norm_max": stage21_6_aggregate.get("pre_clip_grad_norm_max"),
        "pre_clip_grad_norm_mean": stage21_6_aggregate.get("pre_clip_grad_norm_mean"),
        "post_clip_grad_norm_max": _max([_float(row.get("post_clip_grad_norm")) for row in stage21_6_seed_rows]),
        "stage21_4_clip_grad_norm_limit": config["stage21_4_clip_grad_norm_limit"],
        "stage21_6_pre_clip_grad_norm_gate": config["stage21_6_pre_clip_grad_norm_gate"],
        "max_post_update_approx_kl_mean": stage21_6_aggregate.get("max_post_update_approx_kl_mean"),
        "min_entropy_mean": stage21_6_aggregate.get("min_entropy_mean"),
        "parameter_delta_l2_min": _min(parameter_deltas),
        "parameter_delta_l2_max": _max(parameter_deltas),
        "policy_shift_observable": _policy_shift_observable(stage21_6_seed_rows, config),
        "hard_risk_violation_total": stage21_6_aggregate.get("hard_risk_violation_total"),
        "model_inference_mask_violation_total": stage21_6_aggregate.get("model_inference_mask_violation_total"),
        "path_planning_failure_total": stage21_6_aggregate.get("path_planning_failure_total"),
        "open_grid_fallback_total": stage21_6_aggregate.get("open_grid_fallback_total"),
        "sample_count_too_low_for_performance_claim": stage21_6_aggregate.get("sample_count_too_low_for_performance_claim"),
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": all_boundary_reasons,
        "run_reason_codes": run_reasons,
        "stage21_13_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    (output_root / REPORT_FILE).write_text(_report(summary), encoding="utf-8")
    _write_json(
        output_root / MANIFEST_FILE,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": generated_at,
            "config": str(config_path),
            "artifacts": {
                "summary": str(output_root / SUMMARY_FILE),
                "stage21_6_config": str(stage21_6_config) if stage21_6_config else None,
                "stage21_6_root": str(_stage21_6_root(config, output_root)),
                "stage21_6_result": str(output_root / STAGE21_6_RESULT_FILE),
                "routing": str(output_root / ROUTING_FILE),
                "report": str(output_root / REPORT_FILE),
                "manifest": str(output_root / MANIFEST_FILE),
            },
            "summary_status": status,
            "next_required_change": route,
        },
    )
    return summary


def _capped_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in seed_rows:
        stage21_5_root = Path(str(row.get("stage21_5_root", "")))
        summary = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
        rows.append(
            {
                "seed": row.get("seed"),
                "stage21_5_root": str(stage21_5_root),
                "final_coverage_rate_delta": summary.get("final_coverage_rate_delta"),
                "coverage_curve_auc_delta": summary.get("coverage_curve_auc_delta"),
                "final_coverage_rate_capped_delta": summary.get("final_coverage_rate_capped_delta"),
                "coverage_curve_auc_capped_delta": summary.get("coverage_curve_auc_capped_delta"),
            }
        )
    return rows


def _capped_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_distribution("final_coverage_rate_delta", [_float(row.get("final_coverage_rate_delta")) for row in rows]),
        **_distribution("coverage_curve_auc_delta", [_float(row.get("coverage_curve_auc_delta")) for row in rows]),
        **_distribution("final_coverage_rate_capped_delta", [_float(row.get("final_coverage_rate_capped_delta")) for row in rows]),
        **_distribution("coverage_curve_auc_capped_delta", [_float(row.get("coverage_curve_auc_capped_delta")) for row in rows]),
    }


def _stage21_6_root(config: dict[str, Any], output_root: Path) -> Path:
    return output_root / str(config["stage21_6_output_subdir"])


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.13 Coverage-Constrained Multi-Seed PPO Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- primary_reason: `{summary['primary_reason']}`",
            f"- stage21_6_status: `{summary.get('stage21_6_status')}`",
            f"- trainable_transition_count_total: `{summary.get('trainable_transition_count_total')}`",
            f"- mean_final_coverage_delta: `{summary.get('mean_final_coverage_delta')}`",
            f"- mean_coverage_auc_delta: `{summary.get('mean_coverage_auc_delta')}`",
            f"- mean_final_coverage_rate_capped_delta: `{summary.get('mean_final_coverage_rate_capped_delta')}`",
            f"- mean_coverage_curve_auc_capped_delta: `{summary.get('mean_coverage_curve_auc_capped_delta')}`",
            f"- pre_clip_grad_norm_max: `{summary.get('pre_clip_grad_norm_max')}`",
            f"- post_clip_grad_norm_max: `{summary.get('post_clip_grad_norm_max')}`",
            f"- parameter_delta_l2_min: `{summary.get('parameter_delta_l2_min')}`",
            f"- policy_shift_observable: `{summary.get('policy_shift_observable')}`",
            f"- stage21_6_lineage_passed: `{summary.get('stage21_6_lineage_passed')}`",
            f"- hard_risk_violation_total: `{summary.get('hard_risk_violation_total')}`",
            "",
            "Stage21.13 is an offline PPO smoke using the Stage21.12 coverage-constrained reward v2 recommendation. "
            "It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    if not config.get("stage21_12_root"):
        raise ConfigError("missing required config field: stage21_12_root")
    config["stage21_12_root"] = str(_resolve_path(Path(str(config["stage21_12_root"])), repo_root))
    config["execute_stage21_6"] = bool(config.get("execute_stage21_6", True))
    config["reuse_existing_stage21_6_result"] = bool(config.get("reuse_existing_stage21_6_result", True))
    config["stage21_6_output_subdir"] = _safe_output_subdir(config.get("stage21_6_output_subdir", "s6"))
    config["stage21_4_clip_grad_norm_limit"] = _positive_float(config.get("stage21_4_clip_grad_norm_limit", 1.0), "stage21_4_clip_grad_norm_limit")
    config["stage21_6_pre_clip_grad_norm_gate"] = _positive_float(config.get("stage21_6_pre_clip_grad_norm_gate", 25.0), "stage21_6_pre_clip_grad_norm_gate")
    config["max_abs_approx_kl"] = _nonnegative_float(config.get("max_abs_approx_kl", 0.5), "max_abs_approx_kl")
    config["min_entropy"] = _nonnegative_float(config.get("min_entropy", 0.0), "min_entropy")
    config["min_parameter_delta_l2"] = _nonnegative_float(config.get("min_parameter_delta_l2", 1.0e-12), "min_parameter_delta_l2")
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _min(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def _max(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _distribution(prefix: str, values: list[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {f"{prefix}_mean": None, f"{prefix}_min": None, f"{prefix}_max": None}
    return {f"{prefix}_mean": mean(clean), f"{prefix}_min": min(clean), f"{prefix}_max": max(clean)}


def _positive_float(value: Any, field: str) -> float:
    parsed = _float(value)
    if parsed is None or parsed <= 0.0:
        raise ConfigError(f"{field} must be positive")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = _float(value)
    if parsed is None or parsed < 0.0:
        raise ConfigError(f"{field} must be nonnegative")
    return parsed


def _safe_output_subdir(value: Any) -> str:
    parsed = str(value or "").strip()
    if not parsed or parsed in {".", ".."} or "/" in parsed or "\\" in parsed:
        raise ConfigError("stage21_6_output_subdir must be a simple relative directory name")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
