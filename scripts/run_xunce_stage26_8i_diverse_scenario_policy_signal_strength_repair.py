from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_high_fidelity_exploration_coverage_comparison as hf
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as stage26_7f
    import run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as stage26_7g
    import run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as stage26_8h
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as hf
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as stage26_7f
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as stage26_7g
    import scripts.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as stage26_8h


STAGE_ID = "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8i-summary/v1"
SWEEP_ROW_SCHEMA_VERSION = "xunce-stage26-8i-update-sweep-row/v1"
MARGIN_AUDIT_SCHEMA_VERSION = "xunce-stage26-8i-policy-margin-audit/v1"
EVAL_AUDIT_SCHEMA_VERSION = "xunce-stage26-8i-stable-combo-eval-audit/v1"
KL_AUDIT_SCHEMA_VERSION = "xunce-stage26-8i-kl-checkpoint-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8i-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8i-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair_v1.json"
DEFAULT_STAGE26_8H_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval_v1"
)
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair_v1"
)

SUMMARY_FILE = "xunce-stage26-8i-summary.json"
SWEEP_FILE = "xunce-stage26-8i-update-sweep-results.jsonl"
MARGIN_AUDIT_FILE = "xunce-stage26-8i-policy-margin-audit.json"
EVAL_AUDIT_FILE = "xunce-stage26-8i-stable-combo-eval-audit.json"
KL_AUDIT_FILE = "xunce-stage26-8i-kl-checkpoint-audit.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage26-8i-recommended-next-config.json"
ROUTING_FILE = "xunce-stage26-8i-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8i-report.md"
MANIFEST_FILE = "xunce-stage26-8i-manifest.json"

ROUTE_INPUTS = "rerun_stage26_8i_required_inputs"
ROUTE_STAGE26_8H_BINDING = "repair_stage26_8h_eval_binding_or_safety"
ROUTE_UPDATE_STABILITY = "repair_stage26_8i_update_stability"
ROUTE_MARGIN_BINDING = "repair_stage26_8i_policy_margin_audit_binding"
ROUTE_CONTINUE_POST_EVAL = "continue_stage26_8h_post_eval"
ROUTE_POST_EVAL_EXECUTION = "repair_stage26_8i_stable_combo_post_eval_execution"
ROUTE_INCREASE = "increase_stage26_synthetic_update_strength_or_sample_count"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_RESUME = "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
ROUTE_STAGE26_9 = "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
ROUTE_BOUNDARY = "resolve_stage26_8i_boundary_rejections"

ROUTE_FROM_STAGE26_8H = "repair_stage26_synthetic_policy_update_signal_strength"
MAX_ALLOWED_KL_LIMIT = 1.5
BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.8I diverse-scenario policy signal repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_8h_root = Path(config["stage26_8h_root"])
    stage26_8h_summary = _read_json_if_exists(stage26_8h_root / stage26_8h.SUMMARY_FILE)
    stage26_3_summary = _read_json_if_exists(Path(str(stage26_8h_summary.get("stage26_3_root") or stage26_8h_root / "s26_3_aggregate")) / "xunce-stage26-3-summary.json")
    stage26_8g_root = Path(str(stage26_8h_summary.get("stage26_8g_root") or config.get("stage26_8g_root") or ""))
    primary_stage26_1_root = stage26_8g_root / "h16_seed260801" / "s26_1"
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_8h_summary, stage26_3_summary, stage26_8g_root, primary_stage26_1_root)

    combo_details: list[dict[str, Any]] = []
    if not boundary_rejections and not input_rejections and bool(config["run_stage26_chain"]):
        for combo in config["update_sweep"]:
            detail = _run_combo(
                config,
                combo,
                primary_stage26_1_root,
                stage26_8g_root,
                output_root,
                repo_root,
                allow_eval=False,
            )
            combo_details.append(detail)
        for eval_detail in _best_update_details(combo_details, float(config["max_abs_approx_kl"]), int(config["max_stable_eval_count"])):
            _attach_eval_to_detail(config, eval_detail, stage26_8g_root, repo_root)

    sweep_rows = [detail["sweep_row"] for detail in combo_details]
    stable_rows = [row for row in sweep_rows if _combo_is_stable(row, float(config["max_abs_approx_kl"]))]
    best_detail = _best_stable_detail(combo_details, float(config["max_abs_approx_kl"]))
    margin_audit = _policy_margin_audit(best_detail)
    eval_audit = _stable_combo_eval_audit(best_detail)
    kl_audit = _kl_checkpoint_audit(sweep_rows, float(config["max_abs_approx_kl"]))
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        stage26_3_summary=stage26_3_summary,
        sweep_rows=sweep_rows,
        stable_rows=stable_rows,
        best_detail=best_detail,
        margin_audit=margin_audit,
        eval_audit=eval_audit,
        kl_audit=kl_audit,
    )
    status = "passed" if route in {ROUTE_RESUME, ROUTE_STAGE26_9} else "failed"
    best_row = best_detail.get("sweep_row", {})
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_8h_root": str(stage26_8h_root),
        "stage26_8h_status": stage26_8h_summary.get("status"),
        "stage26_8h_next_required_change": stage26_8h_summary.get("next_required_change"),
        "primary_stage26_1_root": str(primary_stage26_1_root),
        "combo_count": len(sweep_rows),
        "stable_combo_count": len(stable_rows),
        "best_stable_combo_id": best_row.get("combo_id"),
        "best_stable_final_post_update_policy_approx_kl": best_row.get("final_post_update_policy_approx_kl"),
        "best_stable_final_post_update_behavior_approx_kl": best_row.get("final_post_update_behavior_approx_kl"),
        "best_stable_parameter_delta_l2": best_row.get("parameter_delta_l2"),
        "strong_state_join_available_count": margin_audit.get("strong_state_join_available_count"),
        "margin_evaluable_join_count": margin_audit.get("margin_evaluable_join_count"),
        "selected_action_changed_count": eval_audit.get("selected_action_changed_count", 0),
        "selected_viewpoint_changed_count": eval_audit.get("selected_viewpoint_changed_count", 0),
        "selected_theta_changed_count": eval_audit.get("selected_theta_changed_count", 0),
        "mean_abs_probability_delta": margin_audit.get("mean_abs_probability_delta", 0.0),
        "mean_top1_probability_margin": margin_audit.get("mean_top1_probability_margin", 0.0),
        "mean_margin_closure_rate": margin_audit.get("mean_margin_closure_rate"),
        "estimated_updates_to_cross_margin_mean": margin_audit.get("estimated_updates_to_cross_margin_mean"),
        "policy_delta_too_small_for_margin": margin_audit.get("policy_delta_too_small_for_margin"),
        "main_final_coverage_delta": eval_audit.get("main_final_coverage_delta", 0.0),
        "main_coverage_auc_delta": eval_audit.get("main_coverage_auc_delta", 0.0),
        "main_coverage_per_100m_delta": eval_audit.get("main_coverage_per_100m_delta", 0.0),
        "hybrid_astar_path_cost_delta": eval_audit.get("hybrid_astar_path_cost_delta", 0.0),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "max_traversable_slope_deg": 30.0,
        "max_abs_approx_kl": config["max_abs_approx_kl"],
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary": str(output_root / SUMMARY_FILE),
        "update_sweep_results": str(output_root / SWEEP_FILE),
        "policy_margin_audit": str(output_root / MARGIN_AUDIT_FILE),
        "stable_combo_eval_audit": str(output_root / EVAL_AUDIT_FILE),
        "kl_checkpoint_audit": str(output_root / KL_AUDIT_FILE),
        "recommended_next_config": str(output_root / RECOMMENDED_CONFIG_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_jsonl(output_root / SWEEP_FILE, sweep_rows)
    _write_json(output_root / MARGIN_AUDIT_FILE, margin_audit)
    _write_json(output_root / EVAL_AUDIT_FILE, eval_audit)
    _write_json(output_root / KL_AUDIT_FILE, kl_audit)
    _write_json(output_root / RECOMMENDED_CONFIG_FILE, _recommended_config(config, best_row))
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, sweep_rows), encoding="utf-8")
    return summary


def _run_combo(
    config: dict[str, Any],
    combo: dict[str, Any],
    primary_stage26_1_root: Path,
    stage26_8g_root: Path,
    output_root: Path,
    repo_root: Path,
    *,
    allow_eval: bool,
) -> dict[str, Any]:
    combo_root = output_root / str(combo["work_dir"])
    combo_root.mkdir(parents=True, exist_ok=True)
    stage26_2_config = _build_stage26_2_config(config, combo, primary_stage26_1_root)
    stage26_2_config_path = combo_root / "xunce-stage26-8i-stage26-2-config.json"
    _write_json(stage26_2_config_path, stage26_2_config)
    stage26_2_summary_path = combo_root / "s26_2" / stage26_2.SUMMARY_FILE
    stage26_2_summary = _read_json_if_exists(stage26_2_summary_path)
    if stage26_2_summary.get("status") not in {"passed", "failed"}:
        stage26_2_summary = stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
            config_path=stage26_2_config_path,
            output_root=combo_root / "s26_2",
            repo_root=repo_root,
        )
        _write_json(stage26_2_summary_path, stage26_2_summary)
    stage26_8h_summary: dict[str, Any] = {}
    eval_source_root: Path | None = None
    if allow_eval and _stage26_2_is_stable(stage26_2_summary, float(config["max_abs_approx_kl"])):
        eval_source_root = _prepare_eval_source_root(config, combo_root, stage26_8g_root, combo_root / "s26_2", repo_root)
        stage26_8h_config = _build_stage26_8h_config(config, eval_source_root)
        stage26_8h_config_path = combo_root / "xunce-stage26-8i-stage26-8h-config.json"
        _write_json(stage26_8h_config_path, stage26_8h_config)
        stage26_8h_summary = _run_stage26_8h_to_completion(stage26_8h_config_path, combo_root / "s26_8h", repo_root)
    sweep_row = _sweep_row(combo, combo_root, stage26_2_summary, stage26_8h_summary)
    return {
        "combo": combo,
        "combo_root": combo_root,
        "eval_source_root": eval_source_root,
        "stage26_2_config": stage26_2_config,
        "stage26_2_summary": stage26_2_summary,
        "stage26_8h_summary": stage26_8h_summary,
        "sweep_row": sweep_row,
    }


def _attach_eval_to_detail(config: dict[str, Any], detail: dict[str, Any], stage26_8g_root: Path, repo_root: Path) -> None:
    combo_root = detail["combo_root"]
    eval_source_root = _prepare_eval_source_root(config, combo_root, stage26_8g_root, combo_root / "s26_2", repo_root)
    stage26_8h_config = _build_stage26_8h_config(config, eval_source_root)
    stage26_8h_config_path = combo_root / "xunce-stage26-8i-stage26-8h-config.json"
    _write_json(stage26_8h_config_path, stage26_8h_config)
    stage26_8h_root = combo_root / "s26_8h"
    existing_stage26_8h_summary = _read_json_if_exists(stage26_8h_root / stage26_8h.SUMMARY_FILE)
    if existing_stage26_8h_summary and _stage26_8h_post_eval_execution_blocked(stage26_8h_root):
        stage26_8h_summary = existing_stage26_8h_summary
    else:
        stage26_8h_summary = _run_stage26_8h_one_phase(stage26_8h_config_path, stage26_8h_root, repo_root)
    detail["eval_source_root"] = eval_source_root
    detail["stage26_8h_summary"] = stage26_8h_summary
    detail["sweep_row"] = _sweep_row(detail["combo"], combo_root, detail["stage26_2_summary"], stage26_8h_summary)


def _run_stage26_8h_one_phase(config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    return stage26_8h.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
    )


def _build_stage26_2_config(config: dict[str, Any], combo: dict[str, Any], stage26_1_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_2_base_config"]))
    cfg.update(
        {
            "stage26_1_root": str(stage26_1_root),
            "epochs": int(combo["epochs"]),
            "learning_rate": float(combo["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
            "value_loss_coefficient": float(combo["value_loss_coefficient"]),
            "entropy_coefficient": float(combo["entropy_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": True,
            "loss_scale": float(combo["loss_scale"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "stage26_2_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _prepare_eval_source_root(config: dict[str, Any], combo_root: Path, stage26_8g_root: Path, stage26_2_root: Path, repo_root: Path) -> Path:
    # Keep this path short enough for Windows environments without long-path support.
    source_root = combo_root / "e"
    job_root = source_root / "h16_seed260801"
    (job_root / "s26_1").mkdir(parents=True, exist_ok=True)
    (job_root / "s26_2").mkdir(parents=True, exist_ok=True)
    (job_root / "s26_3").mkdir(parents=True, exist_ok=True)

    _copy_required(stage26_8g_root / "h16_seed260801" / "s26_1" / "xunce-stage26-1-summary.json", job_root / "s26_1" / "xunce-stage26-1-summary.json")
    _copy_required(stage26_2_root / stage26_2.SUMMARY_FILE, job_root / "s26_2" / stage26_2.SUMMARY_FILE)
    _copy_stage26_8g_fixture(stage26_8g_root, source_root / stage26_8g_fixture_file_name())

    stage26_3_config = _read_json(stage26_8g_root / "h16_seed260801" / "xunce-stage26-8g-stage26-3-config.json")
    stage26_3_config.update(
        {
            "stage26_2_root": str(stage26_2_root),
            "stage21_5_timeout_seconds": 0.0,
            "coverage_denominator_source": "main_coverable_cells/v1",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(job_root / "xunce-stage26-8g-stage26-3-config.json", stage26_3_config)

    stage21_5_config = _read_json(stage26_8g_root / "h16_seed260801" / "s26_3" / "xunce-stage26-3-stage21-5-config.json")
    stage21_5_config.update(
        {
            "stage21_4_tiny_ppo_update_smoke_root": str(stage26_2_root / "s21_4"),
            "execute_high_fidelity_evaluations": True,
            "pre_ppo_evaluation_root": str(job_root / "s26_3" / "pre"),
            "post_ppo_evaluation_root": str(job_root / "s26_3" / "post"),
            "coverage_denominator_source": "main_coverable_cells/v1",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(job_root / "s26_3" / "xunce-stage26-3-stage21-5-config.json", stage21_5_config)
    high_fidelity_path = Path(str(stage21_5_config.get("high_fidelity_config") or ""))
    if not high_fidelity_path.is_absolute():
        high_fidelity_path = (repo_root / high_fidelity_path).resolve()
    if high_fidelity_path.is_file():
        _copy_required(high_fidelity_path, job_root / "s26_3" / "xunce-stage26-3-high-fidelity-config.json")
    else:
        _copy_required(stage26_8g_root / "h16_seed260801" / "s26_3" / "xunce-stage26-3-high-fidelity-config.json", job_root / "s26_3" / "xunce-stage26-3-high-fidelity-config.json")
    _copy_pre_eval_carryover(stage26_8g_root, job_root)
    return source_root


def stage26_8g_fixture_file_name() -> str:
    return "xunce-stage26-8g-scenario-fixture-catalog.jsonl"


def _build_stage26_8h_config(config: dict[str, Any], eval_source_root: Path) -> dict[str, Any]:
    return {
        "schema_version": stage26_8h.CONFIG_SCHEMA_VERSION,
        "stage_id": stage26_8h.STAGE_ID,
        "stage26_8g_root": str(eval_source_root),
        "run_mode": "run_next",
        "seed": 260801,
        "rollout_steps": int(config["eval_rollout_steps"]),
        "scenario_duplicate_ratio_threshold": float(config["scenario_duplicate_ratio_threshold"]),
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _sweep_row(combo: dict[str, Any], combo_root: Path, stage26_2_summary: dict[str, Any], stage26_8h_summary: dict[str, Any]) -> dict[str, Any]:
    stage21_4_summary = _stage21_4_summary(stage26_2_summary)
    row = {
        "schema_version": SWEEP_ROW_SCHEMA_VERSION,
        "combo_id": combo["combo_id"],
        "combo_work_dir": combo["work_dir"],
        "epochs": int(combo["epochs"]),
        "learning_rate": float(combo["learning_rate"]),
        "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
        "value_loss_coefficient": float(combo["value_loss_coefficient"]),
        "entropy_coefficient": float(combo["entropy_coefficient"]),
        "loss_scale": float(combo["loss_scale"]),
        "stage26_2_status": stage26_2_summary.get("status"),
        "stage26_2_next_required_change": stage26_2_summary.get("next_required_change"),
        "stage21_4_status": stage26_2_summary.get("stage21_4_status"),
        "stage21_4_next_required_change": stage26_2_summary.get("stage21_4_next_required_change"),
        "final_post_update_approx_kl": _finite(stage26_2_summary.get("final_post_update_approx_kl")),
        "final_post_update_policy_approx_kl": _first_finite(
            stage26_2_summary.get("final_post_update_policy_approx_kl"),
            stage21_4_summary.get("final_post_update_policy_approx_kl"),
            stage26_2_summary.get("final_post_update_approx_kl"),
        ),
        "final_post_update_behavior_approx_kl": _first_finite(
            stage26_2_summary.get("final_post_update_behavior_approx_kl"),
            stage21_4_summary.get("final_post_update_behavior_approx_kl"),
        ),
        "ppo_ratio_old_log_prob_source": stage21_4_summary.get("ppo_ratio_old_log_prob_source") or stage21_4_summary.get("old_log_prob_source"),
        "kl_gate_source": stage21_4_summary.get("kl_gate_source"),
        "behavior_policy_kl_diagnostic_only": stage21_4_summary.get("behavior_policy_kl_diagnostic_only"),
        "loss_finite": bool(stage26_2_summary.get("loss_finite", False)),
        "gradient_finite": bool(stage26_2_summary.get("gradient_finite", False)),
        "final_total_loss": _finite(stage26_2_summary.get("final_total_loss")),
        "final_policy_loss": _finite(stage26_2_summary.get("final_policy_loss")),
        "final_value_loss": _finite(stage26_2_summary.get("final_value_loss")),
        "final_entropy": _finite(stage26_2_summary.get("final_entropy")),
        "pre_clip_grad_norm": _finite(stage26_2_summary.get("pre_clip_grad_norm")),
        "post_clip_grad_norm": _finite(stage26_2_summary.get("post_clip_grad_norm")),
        "parameter_delta_l2": _finite(stage26_2_summary.get("parameter_delta_l2")),
        "checkpoint_exists": bool(stage26_2_summary.get("checkpoint_exists", False)),
        "checkpoint_reload_passed": bool(stage26_2_summary.get("checkpoint_reload_passed", False)),
        "experimental_only": bool(stage26_2_summary.get("experimental_only", False)),
        "checkpoint_boundary_passed": bool(stage26_2_summary.get("checkpoint_boundary_passed", False)),
        "stage21_3_lineage_passed": bool(stage26_2_summary.get("stage21_3_lineage_passed", False)),
        "source_checkpoint_sha_consistent": bool(stage26_2_summary.get("source_checkpoint_sha_consistent", False)),
        "collector_source_checkpoint_match": bool(stage26_2_summary.get("collector_source_checkpoint_match", False)),
        "publishes_checkpoint": bool(stage26_2_summary.get("publishes_checkpoint", False)),
        "replaces_default_policy": bool(stage26_2_summary.get("replaces_default_policy", False)),
        "connects_real_executor": bool(stage26_2_summary.get("connects_real_executor", False)),
        "starts_online_canary": bool(stage26_2_summary.get("starts_online_canary", False)),
        "canary_traffic_fraction": float(stage26_2_summary.get("canary_traffic_fraction") or 0.0),
        "stage26_8h_status": stage26_8h_summary.get("status"),
        "stage26_8h_next_required_change": stage26_8h_summary.get("next_required_change"),
        "stage26_3_status": stage26_8h_summary.get("stage26_3_status"),
        "stage26_3_next_required_change": stage26_8h_summary.get("stage26_3_next_required_change"),
        "combo_root": str(combo_root),
        "stage26_8h_root": stage26_8h_summary.get("summary") and str(Path(str(stage26_8h_summary["summary"])).parent),
    }
    row.update(_eval_metrics(stage26_8h_summary))
    return row


def _eval_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "strong_state_join_available_count": int(summary.get("strong_state_join_available_count") or 0),
        "selected_action_changed_count": int(summary.get("selected_action_changed_count") or 0),
        "selected_viewpoint_changed_count": int(summary.get("selected_viewpoint_changed_count") or 0),
        "selected_theta_changed_count": int(summary.get("selected_theta_changed_count") or 0),
        "mean_abs_probability_delta": float(summary.get("mean_abs_probability_delta") or 0.0),
        "main_final_coverage_delta": _first_finite(summary.get("main_final_coverage_delta"), 0.0),
        "main_coverage_auc_delta": _first_finite(summary.get("main_coverage_auc_delta"), 0.0),
        "main_coverage_per_100m_delta": _first_finite(summary.get("main_coverage_per_100m_delta"), summary.get("coverage_per_100m_delta"), 0.0),
        "hybrid_astar_path_cost_delta": _first_finite(summary.get("hybrid_astar_path_cost_delta"), 0.0),
        "hard_risk_violation_count": int(summary.get("hard_risk_violation_count") or 0),
        "mask_violation_count": int(summary.get("mask_violation_count") or 0),
        "path_planning_failure_count": int(summary.get("path_planning_failure_count") or 0),
        "open_grid_fallback_count": int(summary.get("open_grid_fallback_count") or 0),
    }


def _policy_margin_audit(best_detail: dict[str, Any]) -> dict[str, Any]:
    row = best_detail.get("sweep_row", {})
    eval_root = Path(str(row.get("stage26_8h_root") or ""))
    phase_rows = _read_jsonl_if_exists(eval_root / stage26_8h.PHASE_STATE_FILE)
    pre_root = _phase_source_root(phase_rows, "pre_eval")
    post_root = _phase_source_root(phase_rows, "post_eval")
    if not pre_root or not post_root:
        return _empty_margin_audit("pre_or_post_eval_root_missing")
    pre_rows = _read_jsonl_if_exists(pre_root / hf.MODEL_INFERENCE_FILE)
    post_rows = _read_jsonl_if_exists(post_root / hf.MODEL_INFERENCE_FILE)
    return _margin_from_rows(pre_rows, post_rows, combo_id=str(row.get("combo_id") or ""))


def _margin_from_rows(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]], *, combo_id: str) -> dict[str, Any]:
    joined: list[dict[str, Any]] = []
    missing = duplicate = unmatched_pre = unmatched_post = 0
    pre_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    post_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in pre_rows:
        key = _strong_key(row)
        if key is None:
            missing += 1
            continue
        duplicate += 1 if key in pre_index else 0
        pre_index[key] = row
    for row in post_rows:
        key = _strong_key(row)
        if key is None:
            missing += 1
            continue
        duplicate += 1 if key in post_index else 0
        post_index[key] = row
    unmatched_pre = len([key for key in pre_index if key not in post_index])
    unmatched_post = len([key for key in post_index if key not in pre_index])
    field_missing = 0
    for key in sorted(set(pre_index) & set(post_index), key=str):
        pre = pre_index[key]
        post = post_index[key]
        if _margin_fields_missing(pre) or _margin_fields_missing(post):
            field_missing += 1
            continue
        joined.append(_joined_margin_row(combo_id, pre, post))
    mean_abs_delta = _mean([row["mean_abs_probability_delta"] for row in joined])
    mean_margin = _mean([row["pre_top1_probability_margin"] for row in joined])
    mean_closure = _mean([row["probability_margin_closure_rate"] for row in joined if row["probability_margin_closure_rate"] is not None])
    return {
        "schema_version": MARGIN_AUDIT_SCHEMA_VERSION,
        "best_stable_combo_id": combo_id,
        "strong_state_join_available_count": len(set(pre_index) & set(post_index)),
        "margin_evaluable_join_count": len(joined),
        "pre_post_required_field_missing_count": missing,
        "margin_required_field_missing_count": field_missing,
        "duplicate_strong_key_count": duplicate,
        "unmatched_pre_row_count": unmatched_pre,
        "unmatched_post_row_count": unmatched_post,
        "selected_action_changed_count": sum(1 for row in joined if row["selected_action_changed"]),
        "selected_viewpoint_changed_count": sum(1 for row in joined if row["selected_viewpoint_changed"]),
        "mean_abs_probability_delta": mean_abs_delta,
        "mean_top1_probability_margin": mean_margin,
        "mean_margin_closure_rate": mean_closure,
        "policy_delta_too_small_for_margin": bool(joined) and mean_abs_delta < mean_margin * 0.25,
        "estimated_updates_to_cross_margin_mean": _mean([row["estimated_updates_to_cross_margin"] for row in joined if row["estimated_updates_to_cross_margin"] is not None]),
        "sample_rows": joined[:50],
    }


def _joined_margin_row(combo_id: str, pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    pre_probs = [_finite(value) or 0.0 for value in pre.get("action_probs") or []]
    post_probs = [_finite(value) or 0.0 for value in post.get("action_probs") or []]
    pre_logits = [_finite(value) or 0.0 for value in pre.get("logits") or []]
    post_logits = [_finite(value) or 0.0 for value in post.get("logits") or []]
    selected = int(pre.get("selected_action_index"))
    pre_top = _top_two(pre_probs)
    post_top = _top_two(post_probs)
    pre_logit_top = _top_two(pre_logits)
    post_logit_top = _top_two(post_logits)
    pre_margin = pre_top["top1_value"] - pre_top["top2_value"]
    post_margin = post_top["top1_value"] - post_top["top2_value"]
    closure = pre_margin - post_margin
    selected_delta = _indexed(post_probs, selected) - _indexed(pre_probs, selected)
    return {
        "combo_id": combo_id,
        "scenario_id": pre.get("scenario_id"),
        "step_index": pre.get("step_index"),
        "current_cell": pre.get("current_cell"),
        "candidate_set_hash": pre.get("candidate_set_hash"),
        "pre_selected_action_index": selected,
        "post_selected_action_index": int(post.get("selected_action_index")),
        "selected_action_changed": selected != int(post.get("selected_action_index")),
        "selected_viewpoint_changed": pre.get("selected_viewpoint") != post.get("selected_viewpoint"),
        "selected_action_probability_delta": selected_delta,
        "mean_abs_probability_delta": _mean([abs(after - before) for before, after in zip(pre_probs, post_probs)]),
        "pre_top1_index": pre_top["top1_index"],
        "pre_top2_index": pre_top["top2_index"],
        "post_top1_index": post_top["top1_index"],
        "post_top2_index": post_top["top2_index"],
        "pre_top1_probability_margin": pre_margin,
        "post_top1_probability_margin": post_margin,
        "pre_top1_logit_margin": pre_logit_top["top1_value"] - pre_logit_top["top2_value"],
        "post_top1_logit_margin": post_logit_top["top1_value"] - post_logit_top["top2_value"],
        "probability_margin_closure": closure,
        "probability_margin_closure_rate": closure / pre_margin if pre_margin > 0.0 else None,
        "estimated_updates_to_cross_margin": pre_margin / closure if closure > 1.0e-12 else None,
    }


def _stable_combo_eval_audit(best_detail: dict[str, Any]) -> dict[str, Any]:
    row = best_detail.get("sweep_row", {})
    return {
        "schema_version": EVAL_AUDIT_SCHEMA_VERSION,
        "best_stable_combo_id": row.get("combo_id"),
        "stage26_8h_status": row.get("stage26_8h_status"),
        "stage26_8h_next_required_change": row.get("stage26_8h_next_required_change"),
        "stage26_3_status": row.get("stage26_3_status"),
        "stage26_3_next_required_change": row.get("stage26_3_next_required_change"),
        "selected_action_changed_count": row.get("selected_action_changed_count", 0),
        "selected_viewpoint_changed_count": row.get("selected_viewpoint_changed_count", 0),
        "selected_theta_changed_count": row.get("selected_theta_changed_count", 0),
        "mean_abs_probability_delta": row.get("mean_abs_probability_delta", 0.0),
        "main_final_coverage_delta": row.get("main_final_coverage_delta", 0.0),
        "main_coverage_auc_delta": row.get("main_coverage_auc_delta", 0.0),
        "main_coverage_per_100m_delta": row.get("main_coverage_per_100m_delta", 0.0),
        "hybrid_astar_path_cost_delta": row.get("hybrid_astar_path_cost_delta", 0.0),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "safety_count_nonzero": any(int(row.get(key) or 0) != 0 for key in ("hard_risk_violation_count", "mask_violation_count", "path_planning_failure_count", "open_grid_fallback_count")),
    }


def _kl_checkpoint_audit(rows: list[dict[str, Any]], max_abs_approx_kl: float) -> dict[str, Any]:
    policy_values = [_finite(row.get("final_post_update_policy_approx_kl")) for row in rows]
    policy_clean = [value for value in policy_values if value is not None]
    return {
        "schema_version": KL_AUDIT_SCHEMA_VERSION,
        "combo_count": len(rows),
        "max_abs_approx_kl": max_abs_approx_kl,
        "stable_combo_count": sum(1 for row in rows if _combo_is_stable(row, max_abs_approx_kl)),
        "policy_kl_exceeded_count": sum(1 for value in policy_clean if abs(value) > max_abs_approx_kl),
        "loss_nonfinite_count": sum(1 for row in rows if row.get("loss_finite") is not True),
        "gradient_nonfinite_count": sum(1 for row in rows if row.get("gradient_finite") is not True),
        "checkpoint_reload_passed_count": sum(1 for row in rows if row.get("checkpoint_reload_passed") is True),
        "checkpoint_exists_count": sum(1 for row in rows if row.get("checkpoint_exists") is True),
        "experimental_only_count": sum(1 for row in rows if row.get("experimental_only") is True),
        "checkpoint_boundary_passed_count": sum(1 for row in rows if row.get("checkpoint_boundary_passed") is True),
        "stage21_3_lineage_passed_count": sum(1 for row in rows if row.get("stage21_3_lineage_passed") is True),
        "source_checkpoint_sha_consistent_count": sum(1 for row in rows if row.get("source_checkpoint_sha_consistent") is True),
        "collector_source_checkpoint_match_count": sum(1 for row in rows if row.get("collector_source_checkpoint_match") is True),
        "ppo_ratio_old_log_prob_source_mismatch_count": sum(1 for row in rows if row.get("ppo_ratio_old_log_prob_source") != "behavior_policy_when_present"),
        "kl_gate_source_mismatch_count": sum(1 for row in rows if row.get("kl_gate_source") != "policy_old_logprob_when_available/v1"),
        "behavior_policy_kl_hard_gate_count": sum(1 for row in rows if row.get("behavior_policy_kl_diagnostic_only") is not True),
        "boundary_enabled_count": sum(1 for row in rows if _boundary_enabled(row)),
        "behavior_policy_kl_diagnostic_only": True,
        "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
        "kl_gate_source": "policy_old_logprob_when_available/v1",
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    stage26_3_summary: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    stable_rows: list[dict[str, Any]],
    best_detail: dict[str, Any],
    margin_audit: dict[str, Any],
    eval_audit: dict[str, Any],
    kl_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS if not _stage26_8h_binding_dirty(stage26_3_summary) else ROUTE_STAGE26_8H_BINDING
    if int(kl_audit.get("loss_nonfinite_count") or 0) > 0 or int(kl_audit.get("gradient_nonfinite_count") or 0) > 0:
        return ROUTE_UPDATE_STABILITY
    if not stable_rows:
        return ROUTE_UPDATE_STABILITY
    if best_detail and _stable_combo_post_eval_execution_failed(best_detail.get("sweep_row", {})):
        return ROUTE_POST_EVAL_EXECUTION
    if best_detail and _stable_combo_eval_incomplete(best_detail.get("sweep_row", {})):
        return ROUTE_CONTINUE_POST_EVAL
    if best_detail and _eval_binding_failed(best_detail.get("sweep_row", {})):
        return ROUTE_MARGIN_BINDING
    if int(margin_audit.get("margin_evaluable_join_count") or 0) <= 0 or int(margin_audit.get("margin_required_field_missing_count") or 0) > 0:
        return ROUTE_MARGIN_BINDING
    if int(eval_audit.get("selected_action_changed_count") or 0) <= 0:
        return ROUTE_INCREASE if bool(margin_audit.get("policy_delta_too_small_for_margin")) else ROUTE_MARGIN
    if float(eval_audit.get("main_coverage_per_100m_delta") or 0.0) < 0.0:
        return ROUTE_CREDIT
    if _multiple_good_combos(sweep_rows):
        return ROUTE_STAGE26_9
    return ROUTE_RESUME


def _input_rejections(stage26_8h_summary: dict[str, Any], stage26_3_summary: dict[str, Any], stage26_8g_root: Path, stage26_1_root: Path) -> list[str]:
    reasons: list[str] = []
    if not stage26_8h_summary:
        return ["stage26_8h_summary_missing"]
    if stage26_8h_summary.get("stage_id") != stage26_8h.STAGE_ID:
        reasons.append("stage26_8h_wrong_stage_id")
    if stage26_8h_summary.get("status") != "passed":
        reasons.append("stage26_8h_status_not_passed_clean_policy_signal_repair")
    if stage26_8h_summary.get("next_required_change") != ROUTE_FROM_STAGE26_8H:
        reasons.append("stage26_8h_route_mismatch")
    if stage26_8h_summary.get("phase_execution_failed") is True:
        reasons.append("stage26_8h_phase_execution_failed")
    if stage26_8h_summary.get("pre_eval_complete") is not True or stage26_8h_summary.get("post_eval_complete") is not True or stage26_8h_summary.get("stage26_3_aggregate_complete") is not True:
        reasons.append("stage26_8h_not_fully_aggregated")
    if stage26_8h_summary.get("scenario_diversity_repaired") is not True or int(stage26_8h_summary.get("scenario_signature_duplicate_group_count") or 0) != 0:
        reasons.append("stage26_8h_scenario_diversity_not_clean")
    if _stage26_8h_binding_dirty(stage26_3_summary):
        reasons.append("stage26_8h_binding_or_safety_dirty")
    if not stage26_8g_root.exists():
        reasons.append("stage26_8g_root_missing")
    stage26_1_summary = _read_json_if_exists(stage26_1_root / "xunce-stage26-1-summary.json")
    if stage26_1_summary.get("status") != "passed":
        reasons.append("primary_stage26_1_not_passed")
    return reasons


def _stage26_8h_binding_dirty(stage26_3_summary: dict[str, Any]) -> bool:
    return any(
        int(stage26_3_summary.get(key) or 0) != 0
        for key in (
            "synthetic_inference_required_field_missing_count",
            "hybrid_path_contract_mismatch_count",
            "hard_risk_violation_count",
            "mask_violation_count",
            "path_planning_failure_count",
            "open_grid_fallback_count",
        )
    )


def _stage26_2_is_stable(summary: dict[str, Any], max_abs_approx_kl: float) -> bool:
    row = _sweep_row({"combo_id": "_probe", "work_dir": "_probe", "epochs": 1, "learning_rate": 1.0, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.0, "entropy_coefficient": 0.0, "loss_scale": 1.0}, Path("."), summary, {})
    return _combo_is_stable(row, max_abs_approx_kl)


def _combo_is_stable(row: dict[str, Any], max_abs_approx_kl: float) -> bool:
    kl = _finite(row.get("final_post_update_policy_approx_kl"))
    return (
        row.get("stage21_4_status") == "passed"
        and row.get("loss_finite") is True
        and row.get("gradient_finite") is True
        and kl is not None
        and abs(kl) <= max_abs_approx_kl
        and row.get("ppo_ratio_old_log_prob_source") == "behavior_policy_when_present"
        and row.get("kl_gate_source") == "policy_old_logprob_when_available/v1"
        and row.get("behavior_policy_kl_diagnostic_only") is True
        and row.get("checkpoint_exists") is True
        and row.get("checkpoint_reload_passed") is True
        and row.get("experimental_only") is True
        and row.get("checkpoint_boundary_passed") is True
        and row.get("stage21_3_lineage_passed") is True
        and row.get("source_checkpoint_sha_consistent") is True
        and row.get("collector_source_checkpoint_match") is True
        and not _boundary_enabled(row)
    )


def _best_update_detail(details: list[dict[str, Any]], max_abs_approx_kl: float) -> dict[str, Any]:
    selected = _best_update_details(details, max_abs_approx_kl, 1)
    return selected[0] if selected else {}


def _best_update_details(details: list[dict[str, Any]], max_abs_approx_kl: float, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    stable = [detail for detail in details if _combo_is_stable(detail["sweep_row"], max_abs_approx_kl)]
    return sorted(
        stable,
        key=lambda detail: (
            _kl_not_close_to_limit(detail["sweep_row"], max_abs_approx_kl),
            float(detail["sweep_row"].get("parameter_delta_l2") or 0.0),
            -abs(float(detail["sweep_row"].get("final_post_update_policy_approx_kl") or 0.0)),
        ),
        reverse=True,
    )[:limit]


def _kl_not_close_to_limit(row: dict[str, Any], max_abs_approx_kl: float) -> bool:
    kl = abs(float(row.get("final_post_update_policy_approx_kl") or 0.0))
    return kl <= 0.95 * max_abs_approx_kl


def _best_stable_detail(details: list[dict[str, Any]], max_abs_approx_kl: float) -> dict[str, Any]:
    stable = [detail for detail in details if _combo_is_stable(detail["sweep_row"], max_abs_approx_kl)]
    if not stable:
        return {}
    evaluated = [detail for detail in stable if detail["sweep_row"].get("stage26_8h_status") is not None]
    if evaluated:
        stable = evaluated
    return max(
        stable,
        key=lambda detail: (
            int(detail["sweep_row"].get("selected_action_changed_count") or 0),
            float(detail["sweep_row"].get("mean_abs_probability_delta") or 0.0),
            float(detail["sweep_row"].get("parameter_delta_l2") or 0.0),
            -abs(float(detail["sweep_row"].get("final_post_update_policy_approx_kl") or 0.0)),
        ),
    )


def _eval_binding_failed(row: dict[str, Any]) -> bool:
    return row.get("stage26_8h_status") is None or row.get("stage26_8h_next_required_change") in {
        "repair_stage26_8h_eval_binding_or_safety",
        "repair_stage26_8h_stage26_3_aggregate_contract",
        "rerun_stage26_8h_required_inputs",
    }


def _stable_combo_eval_incomplete(row: dict[str, Any]) -> bool:
    return row.get("stage26_8h_next_required_change") in {
        "continue_stage26_8h_pre_eval",
        "continue_stage26_8h_post_eval",
    }


def _stable_combo_post_eval_execution_failed(row: dict[str, Any]) -> bool:
    eval_root = Path(str(row.get("stage26_8h_root") or ""))
    return _stage26_8h_post_eval_execution_blocked(eval_root)


def _stage26_8h_post_eval_execution_blocked(eval_root: Path) -> bool:
    phase_rows = _read_jsonl_if_exists(eval_root / stage26_8h.PHASE_STATE_FILE)
    for phase in phase_rows:
        if phase.get("phase") != "post_eval":
            continue
        reason = str(phase.get("blocking_reason") or "")
        execution = phase.get("execution") if isinstance(phase.get("execution"), dict) else {}
        execution_reason = str(execution.get("blocking_reason") or "")
        return "BrokenProcessPool" in reason or "BrokenProcessPool" in execution_reason
    return False


def _multiple_good_combos(rows: list[dict[str, Any]]) -> bool:
    return sum(
        1
        for row in rows
        if _combo_eval_success_clean(row)
        and int(row.get("selected_action_changed_count") or 0) > 0
        and float(row.get("main_coverage_per_100m_delta") or 0.0) >= 0.0
        and not any(int(row.get(key) or 0) != 0 for key in ("hard_risk_violation_count", "mask_violation_count", "path_planning_failure_count", "open_grid_fallback_count"))
    ) >= 2


def _combo_eval_success_clean(row: dict[str, Any]) -> bool:
    return (
        row.get("stage26_8h_status") == "passed"
        and row.get("stage26_3_status") == "passed"
        and not _eval_binding_failed(row)
        and int(row.get("strong_state_join_available_count") or 0) > 0
    )


def _phase_source_root(rows: list[dict[str, Any]], phase: str) -> Path | None:
    for row in rows:
        if row.get("phase") == phase and row.get("status") == "complete":
            return Path(str(row.get("source_root")))
    return None


def _strong_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    required = ("scenario_id", "step_index", "current_cell", "covered_cells_hash", "candidate_set_hash", "synthetic_terrain_hash")
    if any(row.get(field) is None for field in required):
        return None
    return tuple(_json_key(row.get(field)) for field in required)


def _margin_fields_missing(row: dict[str, Any]) -> bool:
    selected = row.get("selected_action_index")
    return (
        not isinstance(selected, int)
        or not isinstance(row.get("action_probs"), list)
        or not row.get("action_probs")
        or not isinstance(row.get("logits"), list)
        or not row.get("logits")
        or selected < 0
        or selected >= len(row.get("action_probs"))
    )


def _empty_margin_audit(reason: str) -> dict[str, Any]:
    return {
        "schema_version": MARGIN_AUDIT_SCHEMA_VERSION,
        "strong_state_join_available_count": 0,
        "margin_evaluable_join_count": 0,
        "margin_required_field_missing_count": 0,
        "reason": reason,
        "mean_abs_probability_delta": 0.0,
        "mean_top1_probability_margin": 0.0,
        "mean_margin_closure_rate": None,
        "policy_delta_too_small_for_margin": False,
        "estimated_updates_to_cross_margin_mean": None,
        "sample_rows": [],
    }


def _stage21_4_summary(stage26_2_summary: dict[str, Any]) -> dict[str, Any]:
    root = stage26_2_summary.get("stage21_4_root")
    if not root:
        return {}
    return _read_json_if_exists(Path(str(root)) / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")


def _boundary_enabled(row: dict[str, Any]) -> bool:
    return (
        row.get("publishes_checkpoint") is True
        or row.get("replaces_default_policy") is True
        or row.get("connects_real_executor") is True
        or row.get("starts_online_canary") is True
        or float(row.get("canary_traffic_fraction") or 0.0) != 0.0
    )


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _recommended_config(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "combo_id": row.get("combo_id"),
        "epochs": row.get("epochs"),
        "learning_rate": row.get("learning_rate"),
        "policy_loss_coefficient": row.get("policy_loss_coefficient"),
        "value_loss_coefficient": row.get("value_loss_coefficient"),
        "entropy_coefficient": row.get("entropy_coefficient"),
        "loss_scale": row.get("loss_scale"),
        "max_abs_approx_kl": config["max_abs_approx_kl"],
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
    }


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    config["stage26_8h_root"] = str(_resolve_path(Path(config.get("stage26_8h_root") or DEFAULT_STAGE26_8H_ROOT), repo_root))
    config["stage26_2_base_config"] = str(_resolve_path(Path(config.get("stage26_2_base_config") or stage26_2.DEFAULT_CONFIG), repo_root))
    config["run_stage26_chain"] = bool(config.get("run_stage26_chain", True))
    config["eval_rollout_steps"] = int(config.get("eval_rollout_steps", 16))
    config["max_stable_eval_count"] = _positive_int(config.get("max_stable_eval_count", 1), "max_stable_eval_count")
    config["scenario_duplicate_ratio_threshold"] = float(config.get("scenario_duplicate_ratio_threshold", 0.01))
    config["clip_ratio"] = float(config.get("clip_ratio", 0.2))
    config["advantage_clip_abs"] = float(config.get("advantage_clip_abs", 5.0))
    config["max_grad_norm"] = float(config.get("max_grad_norm", 1.0))
    config["max_abs_approx_kl"] = float(config.get("max_abs_approx_kl", MAX_ALLOWED_KL_LIMIT))
    if not math.isfinite(config["max_abs_approx_kl"]) or config["max_abs_approx_kl"] > MAX_ALLOWED_KL_LIMIT:
        raise ValueError("max_abs_approx_kl must be finite and must not exceed 1.5 for Stage26.8I")
    config["update_sweep"] = _update_sweep(config.get("update_sweep"))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _update_sweep(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        value = [
            {"combo_id": "current_repro", "work_dir": "u0", "epochs": 4, "learning_rate": 1.0e-5, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "entropy_coefficient": 0.01, "loss_scale": 0.25},
            {"combo_id": "lr_x2", "work_dir": "u1", "epochs": 4, "learning_rate": 2.0e-5, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "entropy_coefficient": 0.01, "loss_scale": 0.25},
            {"combo_id": "depth_x2", "work_dir": "u2", "epochs": 8, "learning_rate": 1.0e-5, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.02, "entropy_coefficient": 0.01, "loss_scale": 0.25},
            {"combo_id": "policy_amp", "work_dir": "u3", "epochs": 8, "learning_rate": 2.0e-5, "policy_loss_coefficient": 2.0, "value_loss_coefficient": 0.01, "entropy_coefficient": 0.005, "loss_scale": 0.5},
            {"combo_id": "value_off_probe", "work_dir": "u4", "epochs": 4, "learning_rate": 1.0e-5, "policy_loss_coefficient": 1.0, "value_loss_coefficient": 0.0, "entropy_coefficient": 0.01, "loss_scale": 0.25},
        ]
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("update_sweep entries must be objects")
        result.append(
            {
                "combo_id": str(item.get("combo_id") or f"combo_{index:02d}"),
                "work_dir": str(item.get("work_dir") or f"u{index}"),
                "epochs": _positive_int(item.get("epochs", 1), "epochs"),
                "learning_rate": _positive_float(item.get("learning_rate", 1.0e-5), "learning_rate"),
                "policy_loss_coefficient": _nonnegative_float(item.get("policy_loss_coefficient", 1.0), "policy_loss_coefficient"),
                "value_loss_coefficient": _nonnegative_float(item.get("value_loss_coefficient", 0.02), "value_loss_coefficient"),
                "entropy_coefficient": _nonnegative_float(item.get("entropy_coefficient", 0.01), "entropy_coefficient"),
                "loss_scale": _positive_float(item.get("loss_scale", 0.25), "loss_scale"),
            }
        )
    return result


def _render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.8I Diverse Scenario Policy Signal Strength Repair",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- stable_combo_count: `{summary['stable_combo_count']}`",
        f"- best_stable_combo_id: `{summary.get('best_stable_combo_id')}`",
        f"- selected_action_changed_count: `{summary.get('selected_action_changed_count')}`",
        f"- mean_abs_probability_delta: `{summary.get('mean_abs_probability_delta')}`",
        f"- mean_top1_probability_margin: `{summary.get('mean_top1_probability_margin')}`",
        f"- main_coverage_per_100m_delta: `{summary.get('main_coverage_per_100m_delta')}`",
        "",
        "Combo results:",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row.get('combo_id')}: s26_2={row.get('stage26_2_status')}, "
            f"policy_kl={row.get('final_post_update_policy_approx_kl')}, "
            f"stable={_combo_is_stable(row, float(summary.get('max_abs_approx_kl') or MAX_ALLOWED_KL_LIMIT))}, "
            f"action_changed={row.get('selected_action_changed_count')}, "
            f"per100m={row.get('main_coverage_per_100m_delta')}"
        )
    lines.extend(
        [
            "",
            "This is a bounded offline diagnostic. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _copy_required(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise FileNotFoundError(f"required Stage26.8I source artifact is missing: {src}")
    try:
        shutil.copyfile(src, dst)
    except FileNotFoundError:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _copy_stage26_8g_fixture(stage26_8g_root: Path, dst: Path) -> None:
    for src in (
        stage26_8g_root / stage26_8g_fixture_file_name(),
        stage26_8g_root / "h16_seed260801" / "s26_1" / "src" / "xunce-stage26-scenario-fixtures.jsonl",
    ):
        if src.is_file() and _read_jsonl_if_exists(src):
            _copy_required(src, dst)
            return
    raise FileNotFoundError(f"required Stage26.8G scenario fixture is missing or empty under {stage26_8g_root}")


def _copy_pre_eval_carryover(stage26_8g_root: Path, job_root: Path) -> None:
    src = stage26_8g_root / "h16_seed260801" / "s26_3" / "pre"
    dst = job_root / "s26_3" / "pre"
    summary = "xunce-exploration-coverage-comparison-summary.json"
    required = (
        summary,
        "xunce-exploration-coverage-model-inference.jsonl",
        "xunce-exploration-coverage-episodes.jsonl",
        "xunce-exploration-coverage-steps.jsonl",
    )
    if all((dst / name).is_file() for name in required):
        return
    if not (src / summary).is_file():
        return
    for name in (
        summary,
        "xunce-exploration-coverage-comparison-v2-summary.json",
        "xunce-exploration-coverage-model-inference.jsonl",
        "xunce-exploration-coverage-episodes.jsonl",
        "xunce-exploration-coverage-steps.jsonl",
        "xunce-exploration-coverage-v2-episodes.jsonl",
        "xunce-exploration-coverage-v2-steps.jsonl",
        "xunce-exploration-coverage-comparison-aggregate.json",
        "xunce-exploration-coverage-comparison-manifest.json",
        "xunce-exploration-coverage-comparison-report.md",
    ):
        source = src / name
        if source.is_file():
            _copy_required(source, dst / name)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _top_two(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"top1_index": None, "top2_index": None, "top1_value": 0.0, "top2_value": 0.0}
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else top1
    return {"top1_index": top1[0], "top2_index": top2[0], "top1_value": top1[1], "top2_value": top2[1]}


def _indexed(values: list[float], index: int) -> float:
    return values[index] if 0 <= index < len(values) else 0.0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_finite(*values: Any) -> float:
    for value in values:
        number = _finite(value)
        if number is not None:
            return number
    return 0.0


def _mean(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else 0.0


def _positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive finite")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be nonnegative finite")
    return number


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
