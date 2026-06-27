from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3


STAGE_ID = "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair"
CONFIG_SCHEMA_VERSION = "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-7-summary/v1"
SWEEP_SCHEMA_VERSION = "xunce-stage26-7-score-sweep-result/v1"
TARGET_AUDIT_SCHEMA_VERSION = "xunce-stage26-7-credit-target-path-efficiency-audit/v1"
BEHAVIOR_AUDIT_SCHEMA_VERSION = "xunce-stage26-7-behavior-logprob-audit/v1"
EVAL_COMPARISON_SCHEMA_VERSION = "xunce-stage26-7-post-update-eval-comparison/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-7-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-7-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair_v1"
)
DEFAULT_STAGE26_6_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_6_synthetic_exploration_credit_assignment_v1"
)
DEFAULT_STAGE26_0_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1"
)

SUMMARY_FILE = "xunce-stage26-7-summary.json"
SWEEP_FILE = "xunce-stage26-7-score-sweep-results.jsonl"
TARGET_AUDIT_FILE = "xunce-stage26-7-credit-target-path-efficiency-audit.json"
BEHAVIOR_AUDIT_FILE = "xunce-stage26-7-behavior-logprob-audit.json"
EVAL_COMPARISON_FILE = "xunce-stage26-7-post-update-eval-comparison.json"
RECOMMENDED_STAGE26_1_CONFIG_FILE = "xunce-stage26-7-recommended-stage26-1-config.json"
RECOMMENDED_STAGE26_2_CONFIG_FILE = "xunce-stage26-7-recommended-stage26-2-config.json"
ROUTING_FILE = "xunce-stage26-7-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-7-report.md"
MANIFEST_FILE = "xunce-stage26-7-manifest.json"

SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
ACTION_SPACE_TYPE = "hybrid_discrete_xy_continuous_theta/v1"
ROUTE_FROM_STAGE26_6 = "repair_stage26_synthetic_credit_assignment"

ROUTE_INPUTS = "rerun_stage26_7_required_inputs"
ROUTE_SAMPLER = "repair_stage26_7_path_efficiency_credit_sampler"
ROUTE_LOGPROB = "repair_stage26_7_behavior_logprob_contract"
ROUTE_STABILITY = "repair_stage26_7_credit_ppo_update_stability"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_TARGET_SCORE = "repair_stage26_7_path_efficiency_target_score"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_MULTI_SEED = "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_7_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.7 synthetic credit path-efficiency repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_6_summary = _read_json_if_exists(Path(config["stage26_6_root"]) / "xunce-stage26-6-summary.json")
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_6_summary)
    sweep_rows: list[dict[str, Any]] = []
    combo_details: list[dict[str, Any]] = []
    if not boundary_rejections and not input_rejections and bool(config["run_stage26_chain"]):
        for combo in config["score_sweep"]:
            detail = _run_combo(config=config, combo=combo, output_root=output_root, repo_root=repo_root)
            combo_details.append(detail)
            sweep_rows.append(detail["sweep_row"])

    target_audit = _target_path_efficiency_audit(combo_details)
    behavior_audit = _behavior_logprob_audit(combo_details)
    eval_comparison = _post_update_eval_comparison(combo_details)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        target_audit=target_audit,
        behavior_audit=behavior_audit,
        eval_comparison=eval_comparison,
        sweep_rows=sweep_rows,
    )
    status = "passed" if route == ROUTE_MULTI_SEED else "failed"
    recommended = _recommended_combo(combo_details)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_6_root": config["stage26_6_root"],
        "stage26_6_status": stage26_6_summary.get("status"),
        "stage26_6_next_required_change": stage26_6_summary.get("next_required_change"),
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "action_space_type": ACTION_SPACE_TYPE,
        "max_traversable_slope_deg": 30.0,
        "completed_combo_count": len([row for row in sweep_rows if row.get("stage26_3_status")]),
        "best_combo_id": recommended.get("combo_id"),
        "best_combo_score_version": recommended.get("synthetic_credit_score_version"),
        "v2_target_selected_count": target_audit["v2_target_selected_count"],
        "behavior_logprob_recomputable": behavior_audit["behavior_logprob_recomputable"],
        "best_combo_selected_action_changed_count": recommended.get("selected_action_changed_count", 0),
        "best_combo_final_coverage_delta": recommended.get("final_coverage_delta", 0.0),
        "best_combo_coverage_auc_delta": recommended.get("coverage_auc_delta", 0.0),
        "best_combo_hybrid_astar_path_cost_delta": recommended.get("hybrid_astar_path_cost_delta", 0.0),
        "best_combo_coverage_per_100m_delta": recommended.get("coverage_per_100m_delta", 0.0),
        "best_combo_scenario_regression_count": recommended.get("scenario_regression_count", 0),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
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
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "score_sweep_results": str(output_root / SWEEP_FILE),
            "credit_target_path_efficiency_audit": str(output_root / TARGET_AUDIT_FILE),
            "behavior_logprob_audit": str(output_root / BEHAVIOR_AUDIT_FILE),
            "post_update_eval_comparison": str(output_root / EVAL_COMPARISON_FILE),
            "recommended_stage26_1_config": str(output_root / RECOMMENDED_STAGE26_1_CONFIG_FILE),
            "recommended_stage26_2_config": str(output_root / RECOMMENDED_STAGE26_2_CONFIG_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_jsonl(output_root / SWEEP_FILE, sweep_rows)
    _write_json(output_root / TARGET_AUDIT_FILE, target_audit)
    _write_json(output_root / BEHAVIOR_AUDIT_FILE, behavior_audit)
    _write_json(output_root / EVAL_COMPARISON_FILE, eval_comparison)
    _write_json(output_root / RECOMMENDED_STAGE26_1_CONFIG_FILE, recommended.get("stage26_1_config", {}))
    _write_json(output_root / RECOMMENDED_STAGE26_2_CONFIG_FILE, recommended.get("stage26_2_config", {}))
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, sweep_rows), encoding="utf-8")
    return summary


def _run_combo(*, config: dict[str, Any], combo: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    combo_id = str(combo["combo_id"])
    combo_work_dir = str(combo.get("work_dir") or combo_id)
    combo_root = output_root / combo_work_dir
    combo_root.mkdir(parents=True, exist_ok=True)
    stage26_1_config = _build_stage26_1_config(config, combo)
    stage26_1_config_path = combo_root / "xunce-stage26-7-stage26-1-config.json"
    _write_json(stage26_1_config_path, stage26_1_config)
    stage26_1_summary_path = combo_root / "s26_1" / "xunce-stage26-1-summary.json"
    reused_stage26_1 = stage26_1_summary_path.is_file()
    stage26_1_summary = (
        _read_json_if_exists(stage26_1_summary_path)
        if reused_stage26_1
        else stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
            config_path=stage26_1_config_path,
            output_root=combo_root / "s26_1",
            repo_root=repo_root,
        )
    )
    target_rows = _target_rows(combo_root / "s26_1")
    stage26_2_summary: dict[str, Any] = {}
    stage26_2_config: dict[str, Any] = {}
    reused_stage26_2 = False
    if stage26_1_summary.get("status") == "passed":
        stage26_2_config = _build_stage26_2_config(config, combo_root / "s26_1")
        stage26_2_config_path = combo_root / "xunce-stage26-7-stage26-2-config.json"
        _write_json(stage26_2_config_path, stage26_2_config)
        stage26_2_summary_path = combo_root / "s26_2" / "xunce-stage26-2-summary.json"
        reused_stage26_2 = stage26_2_summary_path.is_file()
        stage26_2_summary = (
            _read_json_if_exists(stage26_2_summary_path)
            if reused_stage26_2
            else stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
                config_path=stage26_2_config_path,
                output_root=combo_root / "s26_2",
                repo_root=repo_root,
            )
        )
    stage26_3_summary: dict[str, Any] = {}
    reused_stage26_3 = False
    if stage26_2_summary.get("status") == "passed":
        stage26_3_config = _build_stage26_3_config(config, combo_root / "s26_2")
        stage26_3_config_path = combo_root / "xunce-stage26-7-stage26-3-config.json"
        _write_json(stage26_3_config_path, stage26_3_config)
        stage26_3_summary_path = combo_root / "s26_3" / "xunce-stage26-3-summary.json"
        reused_stage26_3 = stage26_3_summary_path.is_file()
        stage26_3_summary = (
            _read_json_if_exists(stage26_3_summary_path)
            if reused_stage26_3
            else stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
                config_path=stage26_3_config_path,
                output_root=combo_root / "s26_3",
                repo_root=repo_root,
            )
        )
    target_audit = _combo_target_audit(combo, target_rows)
    behavior_audit = _combo_behavior_audit(combo_root / "s26_1", target_rows)
    sweep_row = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "combo_id": combo_id,
        "combo_work_dir": combo_work_dir,
        "synthetic_credit_score_version": combo["synthetic_credit_score_version"],
        "path_efficiency_max_cost_norm": combo.get("path_efficiency_max_cost_norm"),
        "stage26_1_status": stage26_1_summary.get("status"),
        "stage26_2_status": stage26_2_summary.get("status"),
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_1_reused_existing_output": reused_stage26_1,
        "stage26_2_reused_existing_output": reused_stage26_2,
        "stage26_3_reused_existing_output": reused_stage26_3,
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        **target_audit,
        **behavior_audit,
        **_eval_metrics(stage26_3_summary),
        "combo_root": str(combo_root),
    }
    return {
        "combo": combo,
        "combo_work_dir": combo_work_dir,
        "combo_root": combo_root,
        "stage26_1_summary": stage26_1_summary,
        "stage26_2_summary": stage26_2_summary,
        "stage26_3_summary": stage26_3_summary,
        "stage26_1_config": stage26_1_config,
        "stage26_2_config": stage26_2_config,
        "target_rows": target_rows,
        "target_audit": target_audit,
        "behavior_audit": behavior_audit,
        "sweep_row": sweep_row,
    }


def _build_stage26_1_config(config: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_1_base_config"]))
    cfg.update(
        {
            "stage26_0_root": config["stage26_0_root"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "min_trainable_transition_count": int(config["trainable_transition_target_count"]),
            "action_space_type": ACTION_SPACE_TYPE,
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "synthetic_exploration_credit_enabled": True,
            "synthetic_credit_mixture_probability": float(config["synthetic_credit_mixture_probability"]),
            "synthetic_credit_score_version": combo["synthetic_credit_score_version"],
            "path_efficiency_max_cost_norm": float(combo.get("path_efficiency_max_cost_norm", 0.70)),
            "allow_synthetic_credit_behavior_policy": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage26_2_config(config: dict[str, Any], stage26_1_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_2_base_config"]))
    cfg.update(
        {
            "stage26_1_root": str(stage26_1_root),
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(config["policy_loss_coefficient"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "entropy_coefficient": float(config["entropy_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": True,
            "loss_scale": float(config["loss_scale"]),
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


def _build_stage26_3_config(config: dict[str, Any], stage26_2_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_3_base_config"]))
    cfg.update(
        {
            "stage26_2_root": str(stage26_2_root),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["eval_rollout_steps"]),
            "action_space_type": ACTION_SPACE_TYPE,
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "stage26_3_authorized": False,
            "release_or_training_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _target_path_efficiency_audit(combo_details: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [detail["sweep_row"] for detail in combo_details]
    v2_rows = [row for row in rows if row.get("synthetic_credit_score_version") == "path_efficiency_v2"]
    return {
        "schema_version": TARGET_AUDIT_SCHEMA_VERSION,
        "combo_count": len(rows),
        "v2_combo_count": len(v2_rows),
        "target_selected_count": sum(int(row.get("synthetic_credit_target_selected_count") or 0) for row in rows),
        "v2_target_selected_count": sum(int(row.get("synthetic_credit_target_selected_count") or 0) for row in v2_rows),
        "path_efficiency_filter_relaxed_count": sum(int(row.get("path_efficiency_filter_relaxed_count") or 0) for row in rows),
        "mean_selected_target_hybrid_cost_norm": _mean(
            row.get("mean_selected_target_hybrid_cost_norm") for row in rows
        ),
        "mean_selected_target_gain_per_cost_norm": _mean(
            row.get("mean_selected_target_gain_per_cost_norm") for row in rows
        ),
    }


def _behavior_logprob_audit(combo_details: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [detail["sweep_row"] for detail in combo_details]
    mismatch = sum(int(row.get("behavior_logprob_mismatch_count") or 0) for row in rows)
    behavior_rows = sum(int(row.get("behavior_policy_row_count") or 0) for row in rows)
    missing = sum(int(row.get("stage21_3_behavior_logprob_missing_count") or 0) for row in rows)
    return {
        "schema_version": BEHAVIOR_AUDIT_SCHEMA_VERSION,
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        "behavior_policy_row_count": behavior_rows,
        "behavior_logprob_mismatch_count": mismatch,
        "stage21_3_behavior_logprob_missing_count": missing,
        "behavior_logprob_recomputable": behavior_rows > 0 and mismatch == 0 and missing == 0,
    }


def _post_update_eval_comparison(combo_details: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [detail["sweep_row"] for detail in combo_details]
    return {
        "schema_version": EVAL_COMPARISON_SCHEMA_VERSION,
        "combo_count": len(rows),
        "combos": rows,
        "best_combo": _recommended_combo(combo_details),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    target_audit: dict[str, Any],
    behavior_audit: dict[str, Any],
    eval_comparison: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if int(target_audit.get("v2_target_selected_count") or 0) <= 0:
        return ROUTE_SAMPLER
    if behavior_audit.get("behavior_logprob_recomputable") is not True:
        return ROUTE_LOGPROB
    if any(row.get("stage26_2_status") == "failed" for row in sweep_rows):
        return ROUTE_STABILITY
    best = eval_comparison.get("best_combo") if isinstance(eval_comparison.get("best_combo"), dict) else {}
    if _combo_success(best):
        return ROUTE_MULTI_SEED
    if int(best.get("selected_action_changed_count") or 0) <= 0:
        return ROUTE_MARGIN if float(best.get("mean_abs_probability_delta") or 0.0) > 0.0 else ROUTE_TARGET_SCORE
    if float(best.get("hybrid_astar_path_cost_delta") or 0.0) > 0.0 or float(best.get("coverage_per_100m_delta") or 0.0) < 0.0:
        return ROUTE_TARGET_SCORE
    if float(best.get("final_coverage_delta") or 0.0) <= 0.0 or float(best.get("coverage_auc_delta") or 0.0) <= 0.0:
        return ROUTE_CREDIT
    return ROUTE_CREDIT


def _combo_success(combo: dict[str, Any]) -> bool:
    return (
        combo.get("stage26_1_status") == "passed"
        and combo.get("stage26_2_status") == "passed"
        and combo.get("stage26_3_status") == "passed"
        and int(combo.get("selected_action_changed_count") or 0) > 0
        and float(combo.get("final_coverage_delta") or 0.0) > 0.0
        and float(combo.get("coverage_auc_delta") or 0.0) > 0.0
        and float(combo.get("hybrid_astar_path_cost_delta") or 0.0) <= 0.0
        and float(combo.get("coverage_per_100m_delta") or 0.0) >= 0.0
        and int(combo.get("scenario_regression_count") or 0) == 0
        and int(combo.get("hard_risk_violation_count") or 0) == 0
        and int(combo.get("mask_violation_count") or 0) == 0
        and int(combo.get("path_planning_failure_count") or 0) == 0
        and int(combo.get("open_grid_fallback_count") or 0) == 0
    )


def _recommended_combo(combo_details: list[dict[str, Any]]) -> dict[str, Any]:
    if not combo_details:
        return {}
    rows = [detail["sweep_row"] for detail in combo_details]
    v2_rows = [row for row in rows if row.get("synthetic_credit_score_version") == "path_efficiency_v2"]
    candidates = v2_rows or rows
    best = max(
        candidates,
        key=lambda row: (
            _combo_success(row),
            float(row.get("coverage_per_100m_delta") or 0.0),
            -max(0.0, float(row.get("hybrid_astar_path_cost_delta") or 0.0)),
            float(row.get("coverage_auc_delta") or 0.0),
        ),
    )
    detail = next((item for item in combo_details if item["sweep_row"].get("combo_id") == best.get("combo_id")), {})
    return dict(best, stage26_1_config=detail.get("stage26_1_config", {}), stage26_2_config=detail.get("stage26_2_config", {}))


def _combo_target_audit(combo: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("synthetic_credit_target_selected") is True]
    return {
        "synthetic_credit_target_selected_count": len(selected),
        "synthetic_credit_score_version_count": sum(
            1 for row in rows if row.get("synthetic_credit_score_version") == combo["synthetic_credit_score_version"]
        ),
        "path_efficiency_filter_relaxed_count": sum(1 for row in rows if row.get("path_efficiency_filter_relaxed") is True),
        "mean_selected_target_hybrid_cost_norm": _mean(
            row.get("selected_target_hybrid_cost_norm") for row in selected
        ),
        "mean_selected_target_gain_per_cost_norm": _mean(
            row.get("selected_target_gain_per_cost_norm") for row in selected
        ),
    }


def _combo_behavior_audit(stage26_1_root: Path, target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage21_3_summary = _read_json_if_exists(
        stage26_1_root / "s21_3" / "xunce-stage21-3-ppo-batch-validation-summary.json"
    )
    behavior_rows = [row for row in target_rows if row.get("behavior_policy_id") == "synthetic_credit_mixture_policy/v1"]
    mismatch = 0
    for row in behavior_rows:
        old_total = _finite(row.get("old_log_prob"))
        old_behavior = _finite(row.get("old_behavior_log_prob"))
        if old_total is None or old_behavior is None or abs(old_total - old_behavior) > 1.0e-5:
            mismatch += 1
    return {
        "behavior_policy_row_count": len(behavior_rows),
        "behavior_logprob_mismatch_count": mismatch,
        "stage21_3_behavior_logprob_missing_count": int(
            stage21_3_summary.get("synthetic_credit_behavior_logprob_missing_count", mismatch) or 0
        ),
    }


def _eval_metrics(stage26_3_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_action_changed_count": int(stage26_3_summary.get("selected_action_changed_count") or 0),
        "selected_viewpoint_changed_count": int(stage26_3_summary.get("selected_viewpoint_changed_count") or 0),
        "selected_theta_changed_count": int(stage26_3_summary.get("selected_theta_changed_count") or 0),
        "mean_abs_probability_delta": float(stage26_3_summary.get("mean_abs_probability_delta") or 0.0),
        "final_coverage_delta": float(stage26_3_summary.get("final_coverage_delta") or 0.0),
        "coverage_auc_delta": float(stage26_3_summary.get("coverage_auc_delta") or 0.0),
        "hybrid_astar_path_cost_delta": float(stage26_3_summary.get("hybrid_astar_path_cost_delta") or 0.0),
        "coverage_per_100m_delta": float(stage26_3_summary.get("coverage_per_100m_delta") or 0.0),
        "scenario_regression_count": int(stage26_3_summary.get("scenario_regression_count") or 0),
        "hard_risk_violation_count": int(stage26_3_summary.get("hard_risk_violation_count") or 0),
        "mask_violation_count": int(stage26_3_summary.get("mask_violation_count") or 0),
        "path_planning_failure_count": int(stage26_3_summary.get("path_planning_failure_count") or 0),
        "open_grid_fallback_count": int(stage26_3_summary.get("open_grid_fallback_count") or 0),
    }


def _target_rows(stage26_1_root: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl_if_exists(stage26_1_root / "s21_1" / "xunce-stage21-1-ppo-trainable-batch.jsonl")
    result: list[dict[str, Any]] = []
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        result.append(
            {
                "transition_id": row.get("transition_id"),
                "action_index": row.get("action_index"),
                "behavior_policy_id": row.get("behavior_policy_id", info.get("behavior_policy_id")),
                "old_log_prob": row.get("old_log_prob"),
                "old_behavior_log_prob": row.get("old_behavior_log_prob", info.get("old_behavior_log_prob")),
                "synthetic_credit_target_index": row.get(
                    "synthetic_credit_target_index",
                    info.get("synthetic_credit_target_index"),
                ),
                "synthetic_credit_target_selected": row.get(
                    "synthetic_credit_target_selected",
                    info.get("synthetic_credit_target_selected"),
                ),
                "synthetic_credit_score": row.get("synthetic_credit_score", info.get("synthetic_credit_score")),
                "synthetic_credit_score_version": row.get(
                    "synthetic_credit_score_version",
                    info.get("synthetic_credit_score_version"),
                ),
                "path_efficiency_filter_relaxed": row.get(
                    "path_efficiency_filter_relaxed",
                    info.get("path_efficiency_filter_relaxed"),
                ),
                "selected_target_hybrid_cost_norm": row.get(
                    "selected_target_hybrid_cost_norm",
                    info.get("selected_target_hybrid_cost_norm"),
                ),
                "selected_target_gain_per_cost_norm": row.get(
                    "selected_target_gain_per_cost_norm",
                    info.get("selected_target_gain_per_cost_norm"),
                ),
            }
        )
    return result


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config["stage26_6_root"] = str(_resolve_path(Path(config.get("stage26_6_root") or DEFAULT_STAGE26_6_ROOT), repo_root))
    config["stage26_0_root"] = str(_resolve_path(Path(config.get("stage26_0_root") or DEFAULT_STAGE26_0_ROOT), repo_root))
    config["stage26_1_base_config"] = str(_resolve_path(Path(config.get("stage26_1_base_config") or stage26_1.DEFAULT_CONFIG), repo_root))
    config["stage26_2_base_config"] = str(_resolve_path(Path(config.get("stage26_2_base_config") or stage26_2.DEFAULT_CONFIG), repo_root))
    config["stage26_3_base_config"] = str(_resolve_path(Path(config.get("stage26_3_base_config") or stage26_3.DEFAULT_CONFIG), repo_root))
    config["run_stage26_chain"] = bool(config.get("run_stage26_chain", True))
    config["required_scenario_count"] = int(config.get("required_scenario_count", 2))
    config["rollout_steps"] = int(config.get("rollout_steps", 6))
    config["eval_rollout_steps"] = int(config.get("eval_rollout_steps", 4))
    config["trainable_transition_target_count"] = int(config.get("trainable_transition_target_count", 12))
    config["synthetic_credit_mixture_probability"] = _fraction(
        config.get("synthetic_credit_mixture_probability", 1.0),
        "synthetic_credit_mixture_probability",
    )
    config["hybrid_astar_candidate_eval_workers"] = int(config.get("hybrid_astar_candidate_eval_workers", 4))
    config["epochs"] = int(config.get("epochs", 4))
    config["learning_rate"] = float(config.get("learning_rate", 1.0e-5))
    config["clip_ratio"] = float(config.get("clip_ratio", 0.2))
    config["policy_loss_coefficient"] = float(config.get("policy_loss_coefficient", 1.0))
    config["value_loss_coefficient"] = float(config.get("value_loss_coefficient", 0.02))
    config["entropy_coefficient"] = float(config.get("entropy_coefficient", 0.01))
    config["advantage_clip_abs"] = float(config.get("advantage_clip_abs", 5.0))
    config["loss_scale"] = float(config.get("loss_scale", 0.25))
    config["max_grad_norm"] = float(config.get("max_grad_norm", 1.0))
    config["max_abs_approx_kl"] = float(config.get("max_abs_approx_kl", 1.5))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    config["score_sweep"] = _score_sweep(config.get("score_sweep"))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _score_sweep(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        value = [
            {
                "combo_id": "v1_current_baseline",
                "work_dir": "c0",
                "synthetic_credit_score_version": "coverage_proxy_v1",
                "path_efficiency_max_cost_norm": 0.70,
            },
            {
                "combo_id": "path_efficiency_v2_default",
                "work_dir": "c1",
                "synthetic_credit_score_version": "path_efficiency_v2",
                "path_efficiency_max_cost_norm": 0.70,
            },
            {
                "combo_id": "path_efficiency_v2_strict_cost",
                "work_dir": "c2",
                "synthetic_credit_score_version": "path_efficiency_v2",
                "path_efficiency_max_cost_norm": 0.55,
            },
        ]
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("score_sweep entries must be objects")
        combo = {
            "combo_id": str(item.get("combo_id") or f"combo_{index:02d}"),
            "work_dir": str(item.get("work_dir") or f"c{index}"),
            "synthetic_credit_score_version": str(item.get("synthetic_credit_score_version") or "coverage_proxy_v1"),
            "path_efficiency_max_cost_norm": _fraction(item.get("path_efficiency_max_cost_norm", 0.70), "path_efficiency_max_cost_norm"),
        }
        result.append(combo)
    return result


def _input_rejections(stage26_6_summary: dict[str, Any]) -> list[str]:
    if not stage26_6_summary:
        return ["missing_stage26_6_summary"]
    reasons: list[str] = []
    if stage26_6_summary.get("status") != "failed":
        reasons.append("stage26_6_status_not_failed")
    if stage26_6_summary.get("next_required_change") != ROUTE_FROM_STAGE26_6:
        reasons.append("stage26_6_route_mismatch")
    if stage26_6_summary.get("feature_exposure_fixed") is not True:
        reasons.append("stage26_6_feature_exposure_not_fixed")
    if stage26_6_summary.get("behavior_logprob_contract_fixed") is not True:
        reasons.append("stage26_6_behavior_logprob_not_fixed")
    if int(stage26_6_summary.get("synthetic_credit_target_selected_count") or 0) <= 0:
        reasons.append("stage26_6_credit_target_not_selected")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _render_report(summary: dict[str, Any], sweep_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.7 Synthetic Credit Assignment Path-Efficiency Repair",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- best_combo_id: `{summary.get('best_combo_id')}`",
        f"- best_combo_hybrid_astar_path_cost_delta: `{summary.get('best_combo_hybrid_astar_path_cost_delta')}`",
        f"- best_combo_coverage_per_100m_delta: `{summary.get('best_combo_coverage_per_100m_delta')}`",
        "",
        "Combo results:",
    ]
    for row in sweep_rows:
        lines.append(
            "- "
            f"{row.get('combo_id')}: status={row.get('stage26_3_status')}, "
            f"route={row.get('stage26_3_next_required_change')}, "
            f"action_changed={row.get('selected_action_changed_count')}, "
            f"coverage_auc_delta={row.get('coverage_auc_delta')}, "
            f"path_cost_delta={row.get('hybrid_astar_path_cost_delta')}, "
            f"coverage_per_100m_delta={row.get('coverage_per_100m_delta')}"
        )
    lines.extend(
        [
            "",
            "This is a bounded offline smoke. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _fraction(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _mean(values: Any) -> float | None:
    finite_values = [_finite(value) for value in values]
    clean = [value for value in finite_values if value is not None]
    return None if not clean else sum(clean) / len(clean)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
