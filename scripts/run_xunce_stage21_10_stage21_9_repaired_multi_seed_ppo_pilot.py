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

from run_xunce_stage21_6_multi_seed_ppo_pilot import run_xunce_stage21_6_multi_seed_ppo_pilot  # noqa: E402


CONFIG_SCHEMA_VERSION = "xunce-stage21-10-stage21-9-repaired-multi-seed-ppo-pilot-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-10-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-10-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-10-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot_v1"
)

SUMMARY_FILE = "xunce-stage21-10-summary.json"
STAGE21_4_CONFIG_FILE = "xunce-stage21-10-repaired-stage21-4-config.json"
STAGE21_6_CONFIG_FILE = "xunce-stage21-10-repaired-stage21-6-config.json"
STAGE21_6_RESULT_FILE = "xunce-stage21-10-stage21-6-result-summary.json"
ROUTING_FILE = "xunce-stage21-10-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-10-report.md"
MANIFEST_FILE = "xunce-stage21-10-manifest.json"

ROUTE_INPUTS = "rerun_stage21_10_required_inputs"
ROUTE_BOUNDARY = "resolve_stage21_10_boundary_rejections"
ROUTE_STAGE21_9 = "continue_stage21_9_gradient_normalization_loss_scaling_repair"
ROUTE_REWARD_ADVANTAGE = "repair_stage21_reward_signal_or_advantage_separation"
ROUTE_SCALE = "scale_stage21_ppo_pilot_scenarios_and_horizon"

BOUNDARY_FIELDS = (
    "stage21_10_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.10 Stage21.9 repaired multi-seed PPO pilot.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
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
                "pre_clip_grad_norm_max": summary.get("pre_clip_grad_norm_max"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
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

    stage21_9_summary = _read_json_or_empty(Path(config["stage21_9_root"]) / "xunce-stage21-9-diagnostic-summary.json")
    input_reasons = _input_reasons(config, stage21_9_summary)
    boundary_reasons = _boundary_reasons(config)
    repaired_stage21_4_config: Path | None = None
    repaired_stage21_6_config: Path | None = None
    stage21_6_summary: dict[str, Any] = {}
    stage21_6_aggregate: dict[str, Any] = {}
    stage21_6_lineage: dict[str, Any] = {}
    stage21_6_seed_rows: list[dict[str, Any]] = []
    capped_seed_rows: list[dict[str, Any]] = []
    run_reasons: list[str] = []

    if not input_reasons and not boundary_reasons:
        repaired_stage21_4_config = _write_stage21_4_config(config, output_root, stage21_9_summary)
        repaired_stage21_6_config = _write_stage21_6_config(config, output_root, repaired_stage21_4_config)
        if config["execute_stage21_6"]:
            run_reasons = _run_or_reuse_stage21_6(config, repaired_stage21_6_config, output_root, repo_root)
        stage21_6_root = _stage21_6_root(config, output_root)
        stage21_6_summary = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
        stage21_6_aggregate = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-aggregate-metrics.json")
        stage21_6_lineage = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-lineage-audit.json")
        stage21_6_seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl")
        capped_seed_rows = _capped_seed_rows(stage21_6_seed_rows)

    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        run_reasons=run_reasons,
        stage21_6_summary=stage21_6_summary,
        stage21_6_aggregate=stage21_6_aggregate,
        stage21_6_lineage=stage21_6_lineage,
        stage21_6_seed_rows=stage21_6_seed_rows,
        capped_seed_rows=capped_seed_rows,
        config=config,
    )

    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        stage21_9_summary=stage21_9_summary,
        repaired_stage21_4_config=repaired_stage21_4_config,
        repaired_stage21_6_config=repaired_stage21_6_config,
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


def _input_reasons(config: dict[str, Any], stage21_9_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage21_9_summary:
        reasons.append("missing_stage21_9_summary")
        return reasons
    if stage21_9_summary.get("status") != "passed":
        reasons.append("stage21_9_not_passed")
    if stage21_9_summary.get("next_required_change") != "rerun_stage21_6_multi_seed_pilot_with_stage21_9_repaired_config":
        reasons.append("stage21_9_route_not_stage21_10")
    pre_clip = _float(stage21_9_summary.get("repaired_pre_clip_grad_norm_max"))
    gate = _float(stage21_9_summary.get("repaired_stage21_6_pre_clip_grad_norm_gate"))
    if pre_clip is None or gate is None or pre_clip > gate:
        reasons.append("stage21_9_repaired_pre_clip_not_stable")
    if stage21_9_summary.get("repaired_numerically_stable") is not True:
        reasons.append("stage21_9_repaired_numerically_stable_not_true")
    if stage21_9_summary.get("repaired_policy_shift_observable") is not True:
        reasons.append("stage21_9_policy_shift_not_observable")
    if stage21_9_summary.get("repaired_coverage_auc_not_regressed") is not True:
        reasons.append("stage21_9_repaired_coverage_auc_regressed")
    if not Path(str(stage21_9_summary.get("repaired_stage21_4_config", ""))).is_file():
        reasons.append("missing_stage21_9_repaired_stage21_4_config")
    if not Path(config["stage21_7_repaired_stage21_6_config"]).is_file():
        reasons.append("missing_stage21_7_repaired_stage21_6_config")
    return reasons


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_10_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_10_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _write_stage21_4_config(config: dict[str, Any], output_root: Path, stage21_9_summary: dict[str, Any]) -> Path:
    payload = _read_json(Path(str(stage21_9_summary["repaired_stage21_4_config"])))
    payload.update(
        {
            "learning_rate": float(config["learning_rate"]),
            "epochs": int(config["epochs"]),
            "clip_ratio": float(config["clip_ratio"]),
            "max_grad_norm": float(config["stage21_4_max_grad_norm"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
            "loss_scale": float(config["loss_scale"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "stage21_4_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    path = output_root / STAGE21_4_CONFIG_FILE
    _write_json(path, payload)
    return path


def _write_stage21_6_config(config: dict[str, Any], output_root: Path, stage21_4_config: Path) -> Path:
    payload = _read_json(Path(config["stage21_7_repaired_stage21_6_config"]))
    payload.update(
        {
            "stage21_4_base_config": str(stage21_4_config),
            "seed_list": config["seed_list"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "max_grad_norm": float(config["stage21_6_pre_clip_grad_norm_gate"]),
            "execute_seed_pipeline": True,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
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
    except Exception as exc:  # noqa: BLE001 - stage wrapper records blocker for later inspection.
        _write_json(
            output_root / "xunce-stage21-10-stage21-6-runtime-blocker.json",
            {"status": "blocked", "reason_code": "stage21_10_stage21_6_execution_exception", "error": repr(exc)},
        )
        return ["stage21_10_stage21_6_execution_exception"]
    return []


def _stage21_6_root(config: dict[str, Any], output_root: Path) -> Path:
    return output_root / str(config["stage21_6_output_subdir"])


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    run_reasons: list[str],
    stage21_6_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    stage21_6_lineage: dict[str, Any],
    stage21_6_seed_rows: list[dict[str, Any]],
    capped_seed_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "input_rejections"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "boundary_rejections"
    if run_reasons or not stage21_6_summary or not stage21_6_aggregate:
        return "partial", ROUTE_INPUTS, "stage21_6_result_missing_or_blocked"
    if stage21_6_lineage.get("passed") is not True:
        return "failed", ROUTE_INPUTS, "stage21_6_lineage_not_passed"
    stage_boundary = _stage21_6_boundary_reasons(stage21_6_summary, stage21_6_aggregate, stage21_6_seed_rows)
    if stage_boundary:
        return "failed", ROUTE_BOUNDARY, "stage21_6_boundary_rejections"
    pre_clip = _float(stage21_6_aggregate.get("pre_clip_grad_norm_max"))
    if pre_clip is None or pre_clip > float(config["stage21_6_pre_clip_grad_norm_gate"]):
        return "failed", ROUTE_STAGE21_9, "pre_clip_grad_unstable"
    if not _all_seed_numerics_stable(stage21_6_seed_rows, config):
        return "failed", ROUTE_STAGE21_9, "kl_entropy_or_grad_unstable"
    mean_coverage = _float(stage21_6_aggregate.get("final_coverage_delta_mean"))
    mean_auc = _float(stage21_6_aggregate.get("coverage_auc_delta_mean"))
    min_coverage = _float(stage21_6_aggregate.get("final_coverage_delta_min"))
    min_auc = _float(stage21_6_aggregate.get("coverage_auc_delta_min"))
    capped = _capped_aggregate(capped_seed_rows)
    capped_final = _float(capped.get("final_coverage_rate_capped_delta_mean"))
    capped_auc = _float(capped.get("coverage_curve_auc_capped_delta_mean"))
    capped_min = _float(capped.get("final_coverage_rate_capped_delta_min"))
    capped_auc_min = _float(capped.get("coverage_curve_auc_capped_delta_min"))
    if None not in (mean_coverage, mean_auc, min_coverage, min_auc, capped_final, capped_auc, capped_min, capped_auc_min):
        if (
            stage21_6_summary.get("status") == "passed"
            and mean_coverage > 0.0
            and mean_auc > 0.0
            and min_coverage >= 0.0
            and min_auc >= 0.0
            and capped_final > 0.0
            and capped_auc > 0.0
            and capped_min >= 0.0
            and capped_auc_min >= 0.0
        ):
            return "passed", ROUTE_SCALE, "coverage_auc_improved_with_stable_update"
    return "failed", ROUTE_REWARD_ADVANTAGE, "stable_update_but_no_coverage_auc_uplift"


def _capped_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_row in seed_rows:
        stage21_5_root = Path(str(seed_row.get("stage21_5_root", "")))
        summary = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
        rows.append(
            {
                "seed": seed_row.get("seed"),
                "stage21_5_root": str(stage21_5_root),
                "final_coverage_rate_delta": _float(summary.get("final_coverage_rate_delta")),
                "coverage_curve_auc_delta": _float(summary.get("coverage_curve_auc_delta")),
                "final_coverage_rate_capped_delta": _float(summary.get("final_coverage_rate_capped_delta")),
                "coverage_curve_auc_capped_delta": _float(summary.get("coverage_curve_auc_capped_delta")),
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
    return reasons


def _all_seed_numerics_stable(seed_rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    if not seed_rows:
        return False
    for row in seed_rows:
        kl = _float(row.get("max_post_update_approx_kl"))
        entropy = _float(row.get("min_entropy"))
        grad = _float(row.get("pre_clip_grad_norm"))
        post_clip = _float(row.get("post_clip_grad_norm"))
        if kl is None or abs(kl) > float(config["max_abs_approx_kl"]):
            return False
        if entropy is None or entropy < float(config["min_entropy"]):
            return False
        if grad is None or grad > float(config["stage21_6_pre_clip_grad_norm_gate"]):
            return False
        if post_clip is None or post_clip > float(config["stage21_4_max_grad_norm"]) + 1.0e-5:
            return False
    return True


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    stage21_9_summary: dict[str, Any],
    repaired_stage21_4_config: Path | None,
    repaired_stage21_6_config: Path | None,
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
    all_boundary_reasons = sorted(
        set(boundary_reasons + _stage21_6_boundary_reasons(stage21_6_summary, stage21_6_aggregate, stage21_6_seed_rows))
    )
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "stage21_6_result": output_root / STAGE21_6_RESULT_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }
    stage21_6_result = {
        "stage21_6_summary": stage21_6_summary,
        "stage21_6_aggregate": stage21_6_aggregate,
        "stage21_6_lineage": stage21_6_lineage,
        "stage21_6_seed_count": len(stage21_6_seed_rows),
        "stage21_5_capped_coverage_aggregate": _capped_aggregate(capped_seed_rows),
        "seed_result_keys": [
            {
                "seed": row.get("seed"),
                "pre_clip_grad_norm": row.get("pre_clip_grad_norm"),
                "post_clip_grad_norm": row.get("post_clip_grad_norm"),
                "final_coverage_delta": row.get("final_coverage_delta"),
                "coverage_auc_delta": row.get("coverage_auc_delta"),
                "stage21_4_status": row.get("stage21_4_status"),
                "stage21_5_status": row.get("stage21_5_status"),
            }
            for row in stage21_6_seed_rows
        ],
        "seed_capped_coverage_keys": capped_seed_rows,
    }
    _write_json(paths["stage21_6_result"], stage21_6_result)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": all_boundary_reasons,
        "run_reason_codes": run_reasons,
        "stage21_6_lineage_passed": stage21_6_lineage.get("passed"),
        "stage21_10_authorized": False,
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
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_9_root": str(config["stage21_9_root"]),
        "stage21_9_status": stage21_9_summary.get("status"),
        "stage21_9_next_required_change": stage21_9_summary.get("next_required_change"),
        "repaired_stage21_4_config": str(repaired_stage21_4_config) if repaired_stage21_4_config else None,
        "repaired_stage21_6_config": str(repaired_stage21_6_config) if repaired_stage21_6_config else None,
        "stage21_6_root": str(_stage21_6_root(config, output_root)),
        "stage21_6_status": stage21_6_summary.get("status"),
        "stage21_6_next_required_change": stage21_6_summary.get("next_required_change"),
        "stage21_6_lineage_passed": stage21_6_lineage.get("passed"),
        "seed_count": stage21_6_aggregate.get("seed_count"),
        "trainable_transition_count_total": stage21_6_aggregate.get("trainable_transition_count_total"),
        "mean_final_coverage_delta": stage21_6_aggregate.get("final_coverage_delta_mean"),
        "mean_coverage_auc_delta": stage21_6_aggregate.get("coverage_auc_delta_mean"),
        "mean_final_coverage_rate_capped_delta": _capped_aggregate(capped_seed_rows).get("final_coverage_rate_capped_delta_mean"),
        "mean_coverage_curve_auc_capped_delta": _capped_aggregate(capped_seed_rows).get("coverage_curve_auc_capped_delta_mean"),
        "min_coverage_auc_delta": stage21_6_aggregate.get("coverage_auc_delta_min"),
        "min_final_coverage_rate_capped_delta": _capped_aggregate(capped_seed_rows).get("final_coverage_rate_capped_delta_min"),
        "min_coverage_curve_auc_capped_delta": _capped_aggregate(capped_seed_rows).get("coverage_curve_auc_capped_delta_min"),
        "worst_seed_id": stage21_6_aggregate.get("worst_seed_id"),
        "pre_clip_grad_norm_max": stage21_6_aggregate.get("pre_clip_grad_norm_max"),
        "pre_clip_grad_norm_mean": stage21_6_aggregate.get("pre_clip_grad_norm_mean"),
        "post_clip_grad_norm_max": _max([_float(row.get("post_clip_grad_norm")) for row in stage21_6_seed_rows]),
        "stage21_4_clip_grad_norm_limit": config["stage21_4_max_grad_norm"],
        "stage21_6_pre_clip_grad_norm_gate": config["stage21_6_pre_clip_grad_norm_gate"],
        "max_post_update_approx_kl_mean": stage21_6_aggregate.get("max_post_update_approx_kl_mean"),
        "min_entropy_mean": stage21_6_aggregate.get("min_entropy_mean"),
        "hard_risk_violation_total": stage21_6_aggregate.get("hard_risk_violation_total"),
        "model_inference_mask_violation_total": stage21_6_aggregate.get("model_inference_mask_violation_total"),
        "path_planning_failure_total": stage21_6_aggregate.get("path_planning_failure_total"),
        "open_grid_fallback_total": stage21_6_aggregate.get("open_grid_fallback_total"),
        "sample_count_too_low_for_performance_claim": stage21_6_aggregate.get("sample_count_too_low_for_performance_claim"),
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": all_boundary_reasons,
        "run_reason_codes": run_reasons,
        "stage21_10_authorized": False,
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
    paths["report"].write_text(_report(summary), encoding="utf-8")
    _write_json(
        paths["manifest"],
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": generated_at,
            "config": str(config_path),
            "artifacts": {
                "summary": str(paths["summary"]),
                "stage21_4_config": str(repaired_stage21_4_config) if repaired_stage21_4_config else None,
                "stage21_6_config": str(repaired_stage21_6_config) if repaired_stage21_6_config else None,
                "stage21_6_root": str(_stage21_6_root(config, output_root)),
                "stage21_6_result": str(paths["stage21_6_result"]),
                "routing": str(paths["routing"]),
                "report": str(paths["report"]),
                "manifest": str(paths["manifest"]),
            },
            "summary_status": status,
            "next_required_change": route,
        },
    )
    return summary


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.10 Stage21.9 Repaired Multi-Seed PPO Pilot",
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
            f"- stage21_4_clip_grad_norm_limit: `{summary.get('stage21_4_clip_grad_norm_limit')}`",
            f"- stage21_6_pre_clip_grad_norm_gate: `{summary.get('stage21_6_pre_clip_grad_norm_gate')}`",
            f"- stage21_6_lineage_passed: `{summary.get('stage21_6_lineage_passed')}`",
            f"- hard_risk_violation_total: `{summary.get('hard_risk_violation_total')}`",
            "",
            "Stage 21.10 is an offline repaired multi-seed pilot. It does not publish checkpoints, replace default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in ("stage21_9_root", "stage21_7_repaired_stage21_6_config"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["execute_stage21_6"] = bool(config.get("execute_stage21_6", True))
    config["reuse_existing_stage21_6_result"] = bool(config.get("reuse_existing_stage21_6_result", True))
    config["stage21_6_output_subdir"] = _safe_output_subdir(config.get("stage21_6_output_subdir", "stage21_6"))
    config["seed_list"] = [_nonnegative_int(seed, "seed") for seed in config.get("seed_list", [2101, 2102, 2103])]
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 8), "required_scenario_count")
    config["rollout_steps"] = _positive_int(config.get("rollout_steps", 10), "rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(config.get("dynamic_max_candidates_per_step", 36), "dynamic_max_candidates_per_step")
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(config.get("dynamic_proposal_pool_limit_per_step", 288), "dynamic_proposal_pool_limit_per_step")
    config["learning_rate"] = _positive_float(config.get("learning_rate", 2.0e-6), "learning_rate")
    config["epochs"] = _positive_int(config.get("epochs", 1), "epochs")
    config["clip_ratio"] = _positive_float(config.get("clip_ratio", 0.2), "clip_ratio")
    config["stage21_4_max_grad_norm"] = _positive_float(config.get("stage21_4_max_grad_norm", 1.0), "stage21_4_max_grad_norm")
    config["stage21_6_pre_clip_grad_norm_gate"] = _positive_float(config.get("stage21_6_pre_clip_grad_norm_gate", 25.0), "stage21_6_pre_clip_grad_norm_gate")
    config["advantage_clip_abs"] = _nonnegative_float(config.get("advantage_clip_abs", 5.0), "advantage_clip_abs")
    config["normalize_minibatch_advantages"] = bool(config.get("normalize_minibatch_advantages", True))
    config["loss_scale"] = _positive_float(config.get("loss_scale", 0.25), "loss_scale")
    config["value_loss_coefficient"] = _nonnegative_float(config.get("value_loss_coefficient", 0.1), "value_loss_coefficient")
    config["max_abs_approx_kl"] = _nonnegative_float(config.get("max_abs_approx_kl", 0.5), "max_abs_approx_kl")
    config["min_entropy"] = _nonnegative_float(config.get("min_entropy", 0.0), "min_entropy")
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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


def _max(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _distribution(prefix: str, values: list[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {f"{prefix}_mean": None, f"{prefix}_min": None, f"{prefix}_max": None}
    return {
        f"{prefix}_mean": mean(clean),
        f"{prefix}_min": min(clean),
        f"{prefix}_max": max(clean),
    }


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be positive") from exc
    if parsed <= 0:
        raise ConfigError(f"{field} must be positive")
    return parsed


def _nonnegative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be nonnegative") from exc
    if parsed < 0:
        raise ConfigError(f"{field} must be nonnegative")
    return parsed


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
