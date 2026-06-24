from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by script execution
    import run_xunce_stage21_1_on_policy_ppo_rollout_collector as stage21_1
    import run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import run_xunce_stage21_3_ppo_batch_validation as stage21_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as stage21_1
    import scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import scripts.run_xunce_stage21_3_ppo_batch_validation as stage21_3


CONFIG_SCHEMA_VERSION = "xunce-stage22-3-theta-aware-ppo-collector-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage22-3-summary/v1"
TRANSITION_AUDIT_SCHEMA_VERSION = "xunce-stage22-3-theta-transition-contract-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage22-3-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage22-3-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage22_3_theta_aware_ppo_collector_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
    "outputs/path_feedback_batch_xunce_stage22_3_theta_aware_ppo_collector_smoke_v1"
)

SUMMARY_FILE = "xunce-stage22-3-summary.json"
STAGE21_1_SUMMARY_FILE = "xunce-stage22-3-stage21-1-collector-summary.json"
STAGE21_2_SUMMARY_FILE = "xunce-stage22-3-stage21-2-reward-summary.json"
STAGE21_3_SUMMARY_FILE = "xunce-stage22-3-stage21-3-batch-summary.json"
TRANSITION_AUDIT_FILE = "xunce-stage22-3-theta-transition-contract-audit.json"
ROUTING_FILE = "xunce-stage22-3-next-stage-routing.json"
REPORT_FILE = "xunce-stage22-3-report.md"
MANIFEST_FILE = "xunce-stage22-3-manifest.json"

ROUTE_INPUTS = "rerun_stage22_3_required_inputs"
ROUTE_COLLECTOR = "repair_stage22_3_collector_theta_transition_contract"
ROUTE_BINDING = "repair_stage22_3_action_viewpoint_binding"
ROUTE_REWARD = "repair_stage22_3_theta_reward_provenance"
ROUTE_BATCH = "repair_stage22_3_stage21_batch_theta_contract"
ROUTE_STAGE22_4 = "run_stage22_4_theta_aware_ppo_update_smoke"
ROUTE_BOUNDARY = "resolve_stage22_3_boundary_rejections"
ROUTE_FROM_STAGE22_2 = "run_stage22_3_theta_aware_ppo_collector_smoke"

THETA_COVERAGE_SOURCE = "theta_aware_sensor_footprint/v1"
STAGE22_ID = "xunce-stage22-3-theta-aware-ppo-collector-smoke"

BOUNDARY_FIELDS = (
    "stage22_3_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage22.3 theta-aware PPO collector smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config)

    stage21_1_summary: dict[str, Any] = {}
    stage21_2_summary: dict[str, Any] = {}
    stage21_3_summary: dict[str, Any] = {}
    if not boundary_reasons and not input_reasons:
        stage21_1_summary = _run_stage21_1(config, output_root, repo_root)
        if stage21_1_summary.get("status") == "passed":
            stage21_2_summary = _run_stage21_2(config, output_root, repo_root)
        if stage21_2_summary.get("status") == "passed":
            stage21_3_summary = _run_stage21_3(config, output_root, repo_root)

    transition_rows = _read_jsonl_if_exists(output_root / "s21_1" / stage21_1.TRAINABLE_BATCH_FILE)
    reward_rows = _read_jsonl_if_exists(output_root / "s21_2" / stage21_2.EVALUATION_FILE)
    batch_rows = _read_jsonl_if_exists(output_root / "s21_3" / stage21_3.BATCH_FILE)
    rejections = _read_jsonl_if_exists(output_root / "s21_1" / stage21_1.REJECTION_FILE)

    transition_audit = _transition_contract_audit(transition_rows, reward_rows, batch_rows, rejections)
    status, route, route_reason = _route(
        boundary_reasons,
        input_reasons,
        stage21_1_summary,
        stage21_2_summary,
        stage21_3_summary,
        transition_audit,
    )
    blocking = _unique(boundary_reasons + input_reasons + _route_blocking_reasons(route, transition_audit, stage21_1_summary, stage21_2_summary, stage21_3_summary))

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE22_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage22_2_root": config["stage22_2_root"],
        "stage22_2_status": _read_stage22_2_summary(config).get("status") if not input_reasons else None,
        "stage21_1_root": str(output_root / "s21_1"),
        "stage21_1_status": stage21_1_summary.get("status"),
        "stage21_2_root": str(output_root / "s21_2"),
        "stage21_2_status": stage21_2_summary.get("status"),
        "stage21_3_root": str(output_root / "s21_3"),
        "stage21_3_status": stage21_3_summary.get("status"),
        "theta_aware_candidate_viewpoints_enabled": bool(config["theta_aware_candidate_viewpoints_enabled"]),
        "theta_bin_count": int(config["theta_bin_count"]),
        "theta_step_deg": int(config["theta_step_deg"]),
        "sensor_fov_deg": float(config["sensor_fov_deg"]),
        "required_scenario_count": int(config["required_scenario_count"]),
        "rollout_steps": int(config["rollout_steps"]),
        **_summary_counts(transition_audit),
        "release_or_training_authorized": False,
        "stage22_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "stage22_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "stage21_1_summary": str(output_root / STAGE21_1_SUMMARY_FILE),
        "stage21_2_summary": str(output_root / STAGE21_2_SUMMARY_FILE),
        "stage21_3_summary": str(output_root / STAGE21_3_SUMMARY_FILE),
        "theta_transition_contract_audit": str(output_root / TRANSITION_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE21_1_SUMMARY_FILE, stage21_1_summary)
    _write_json(output_root / STAGE21_2_SUMMARY_FILE, stage21_2_summary)
    _write_json(output_root / STAGE21_3_SUMMARY_FILE, stage21_3_summary)
    _write_json(output_root / TRANSITION_AUDIT_FILE, transition_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _run_stage21_1(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    stage_root = output_root / "s21_1"
    cfg = _load_json_template(config["stage21_1_base_config"], repo_root)
    cfg.update(
        {
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "dynamic_validation_work_root": str(output_root / "_dynamic_validation_work_stage22_3"),
            "theta_aware_candidate_viewpoints_enabled": True,
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": config["sensor_model_id"],
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "stage21_1_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    cfg_path = output_root / "xunce-stage22-3-stage21-1-config.json"
    _write_json(cfg_path, cfg)
    return stage21_1.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=cfg_path,
        output_root=stage_root,
        repo_root=repo_root,
    )


def _run_stage21_2(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    stage_root = output_root / "s21_2"
    cfg = _load_json_template(config["stage21_2_base_config"], repo_root)
    cfg.update(
        {
            "stage21_1_collector_root": str(output_root / "s21_1"),
            "coverage_first_reward_profile": config["coverage_first_reward_profile"],
            "require_theta_aware_reward_contract": True,
            "theta_coverage_denominator_cells": float(config["theta_coverage_denominator_cells"]),
            "stage21_2_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    cfg_path = output_root / "xunce-stage22-3-stage21-2-config.json"
    _write_json(cfg_path, cfg)
    return stage21_2.run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=cfg_path,
        output_root=stage_root,
        repo_root=repo_root,
    )


def _run_stage21_3(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    stage_root = output_root / "s21_3"
    cfg = _load_json_template(config["stage21_3_base_config"], repo_root)
    cfg.update(
        {
            "stage21_1_collector_root": str(output_root / "s21_1"),
            "stage21_2_reward_contract_root": str(output_root / "s21_2"),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "require_theta_aware_viewpoint_contract": True,
            "require_theta_aware_reward_contract": True,
            "stage21_3_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    cfg_path = output_root / "xunce-stage22-3-stage21-3-config.json"
    _write_json(cfg_path, cfg)
    return stage21_3.run_xunce_stage21_3_ppo_batch_validation(
        config_path=cfg_path,
        output_root=stage_root,
        repo_root=repo_root,
    )


def _transition_contract_audit(
    transitions: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
) -> dict[str, Any]:
    transition_missing = 0
    mask_mismatch = 0
    sampling_contract_mismatch = 0
    action_binding_mismatch = 0
    hard_risk = 0
    mask_violation = 0
    open_grid = 0
    for row in transitions:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        candidate_viewpoints = info.get("candidate_viewpoints")
        action_index = _int_or_none(row.get("action_index"))
        theta_arrays_ok = _theta_arrays_present(info)
        if not isinstance(candidate_viewpoints, list) or not candidate_viewpoints or not theta_arrays_ok:
            transition_missing += 1
            continue
        masks = [info.get("action_mask"), info.get("sampling_mask"), info.get("hard_risk_clean_mask")]
        if any(not isinstance(mask, list) or len(mask) != len(candidate_viewpoints) for mask in masks):
            mask_mismatch += 1
        else:
            action_mask, sampling_mask, hard_risk_clean_mask = masks
            for action_allowed, sampled_allowed, hard_risk_clean in zip(action_mask, sampling_mask, hard_risk_clean_mask):
                expected_sampling = action_allowed is True and hard_risk_clean is True
                if sampled_allowed is not expected_sampling:
                    sampling_contract_mismatch += 1
        if action_index is None or action_index < 0 or action_index >= len(candidate_viewpoints):
            action_binding_mismatch += 1
            continue
        selected_viewpoint = info.get("selected_viewpoint")
        selected_theta = info.get("selected_theta_deg")
        if list(candidate_viewpoints[action_index]) != list(selected_viewpoint or []):
            action_binding_mismatch += 1
        if selected_theta != _theta_from_viewpoint(selected_viewpoint):
            action_binding_mismatch += 1
        selected_cell = info.get("selected_cell")
        if isinstance(selected_viewpoint, list) and len(selected_viewpoint) >= 2 and selected_cell is not None:
            if list(selected_viewpoint[:2]) != list(selected_cell):
                action_binding_mismatch += 1
        if info.get("hard_risk_violation") is True:
            hard_risk += 1
        if _mask_allows(info, "action_mask", action_index) is not True:
            mask_violation += 1
        if _mask_allows(info, "sampling_mask", action_index) is not True:
            mask_violation += 1
        if _mask_allows(info, "hard_risk_clean_mask", action_index) is not True:
            mask_violation += 1
        if info.get("open_grid_fallback_used") is True:
            open_grid += 1
    reward_missing = sum(1 for row in rewards if not _row_has_theta_reward_contract(row))
    reward_fallback = sum(1 for row in rewards if row.get("point_only_reward_fallback_used") is True)
    transition_by_id = {
        str(row.get("transition_id")): row
        for row in transitions
        if row.get("transition_id") is not None
    }
    reward_viewpoint_mismatch = 0
    reward_footprint_mismatch = 0
    for row in rewards:
        transition = transition_by_id.get(str(row.get("transition_id")))
        if transition is None:
            continue
        info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
        if list(row.get("candidate_viewpoint") or []) != list(info.get("selected_viewpoint") or []):
            reward_viewpoint_mismatch += 1
        if _int_or_none(row.get("candidate_theta_deg")) != _int_or_none(info.get("selected_theta_deg")):
            reward_viewpoint_mismatch += 1
        action_index = _int_or_none(transition.get("action_index"))
        selected_footprint = _selected_theta_footprint(info, action_index)
        if selected_footprint is None:
            reward_footprint_mismatch += 1
            continue
        if _finite_float(row.get("theta_new_visible_cell_count")) != _finite_float(selected_footprint["theta_new_visible_cell_count"]):
            reward_footprint_mismatch += 1
        if row.get("theta_coverage_hash") != selected_footprint["theta_coverage_hash"]:
            reward_footprint_mismatch += 1
        row_gain = _finite_float(row.get("theta_coverage_gain_per_path_cost"))
        selected_gain = _finite_float(selected_footprint["theta_coverage_gain_per_path_cost"])
        if row_gain is None or selected_gain is None or abs(row_gain - selected_gain) > 1.0e-9:
            reward_footprint_mismatch += 1
    batch_viewpoint_missing = stage21_3._theta_viewpoint_contract_missing_count(batch_rows) if batch_rows else 0
    batch_reward_missing = stage21_3._theta_reward_contract_missing_count(batch_rows) if batch_rows else 0
    path_planning_failures = sum(1 for row in rejections if "path_planning" in str(row.get("reason") or row.get("reason_code") or ""))
    return {
        "schema_version": TRANSITION_AUDIT_SCHEMA_VERSION,
        "transition_count": len(transitions),
        "reward_row_count": len(rewards),
        "batch_row_count": len(batch_rows),
        "theta_transition_contract_missing_count": transition_missing,
        "mask_length_mismatch_count": mask_mismatch,
        "sampling_mask_contract_mismatch_count": sampling_contract_mismatch,
        "action_viewpoint_binding_mismatch_count": action_binding_mismatch,
        "hard_risk_violation_count": hard_risk,
        "mask_violation_count": mask_violation,
        "path_planning_failure_count": path_planning_failures,
        "open_grid_fallback_count": open_grid,
        "theta_reward_contract_missing_count": reward_missing,
        "point_only_reward_fallback_used_count": reward_fallback,
        "reward_viewpoint_binding_mismatch_count": reward_viewpoint_mismatch,
        "reward_footprint_binding_mismatch_count": reward_footprint_mismatch,
        "theta_batch_viewpoint_contract_missing_count": batch_viewpoint_missing,
        "theta_batch_reward_contract_missing_count": batch_reward_missing,
        "stage21_3_rejects_point_only_reward_batch": stage21_3._theta_reward_contract_missing_count([{"reward": 0.0}]) == 1,
    }


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    stage21_1_summary: dict[str, Any],
    stage21_2_summary: dict[str, Any],
    stage21_3_summary: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage22_3_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage22_3_inputs_missing_or_untrusted"
    if stage21_1_summary.get("status") != "passed" or audit["theta_transition_contract_missing_count"] > 0:
        return "failed", ROUTE_COLLECTOR, "stage21_1_theta_transition_contract_missing"
    if audit["transition_count"] <= 0:
        return "failed", ROUTE_COLLECTOR, "stage21_1_theta_transition_rows_missing"
    if (
        audit["mask_length_mismatch_count"]
        or audit["sampling_mask_contract_mismatch_count"]
        or audit["action_viewpoint_binding_mismatch_count"]
        or audit["mask_violation_count"]
        or audit["hard_risk_violation_count"]
        or audit["path_planning_failure_count"]
        or audit["open_grid_fallback_count"]
    ):
        return "failed", ROUTE_BINDING, "stage21_1_action_viewpoint_or_safety_binding_failed"
    if (
        stage21_2_summary.get("status") != "passed"
        or audit["reward_row_count"] <= 0
        or audit["theta_reward_contract_missing_count"]
        or audit["point_only_reward_fallback_used_count"]
        or audit["reward_viewpoint_binding_mismatch_count"]
        or audit["reward_footprint_binding_mismatch_count"]
    ):
        return "failed", ROUTE_REWARD, "stage21_2_theta_reward_provenance_missing"
    if (
        stage21_3_summary.get("status") != "passed"
        or audit["batch_row_count"] <= 0
        or audit["theta_batch_viewpoint_contract_missing_count"]
        or audit["theta_batch_reward_contract_missing_count"]
        or not audit["stage21_3_rejects_point_only_reward_batch"]
    ):
        return "failed", ROUTE_BATCH, "stage21_3_theta_batch_contract_missing"
    return "passed", ROUTE_STAGE22_4, "theta_aware_collector_reward_batch_contract_ready"


def _route_blocking_reasons(
    route: str,
    audit: dict[str, Any],
    stage21_1_summary: dict[str, Any],
    stage21_2_summary: dict[str, Any],
    stage21_3_summary: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_COLLECTOR:
        if stage21_1_summary.get("status") != "passed":
            reasons.append("stage21_1_not_passed")
        if audit["transition_count"] <= 0:
            reasons.append("stage21_1_transition_rows_missing")
        if audit["theta_transition_contract_missing_count"]:
            reasons.append("theta_transition_contract_missing")
    if route == ROUTE_BINDING:
        for key in (
            "mask_length_mismatch_count",
            "sampling_mask_contract_mismatch_count",
            "action_viewpoint_binding_mismatch_count",
            "mask_violation_count",
            "hard_risk_violation_count",
            "path_planning_failure_count",
            "open_grid_fallback_count",
        ):
            if audit[key]:
                reasons.append(key.removesuffix("_count"))
    if route == ROUTE_REWARD:
        if stage21_2_summary.get("status") != "passed":
            reasons.append("stage21_2_not_passed")
        if audit["reward_row_count"] <= 0:
            reasons.append("stage21_2_reward_rows_missing")
        if audit["theta_reward_contract_missing_count"]:
            reasons.append("theta_reward_contract_missing")
        if audit["point_only_reward_fallback_used_count"]:
            reasons.append("point_only_reward_fallback_used")
        if audit["reward_viewpoint_binding_mismatch_count"]:
            reasons.append("reward_viewpoint_binding_mismatch")
        if audit["reward_footprint_binding_mismatch_count"]:
            reasons.append("reward_footprint_binding_mismatch")
    if route == ROUTE_BATCH:
        if stage21_3_summary.get("status") != "passed":
            reasons.append("stage21_3_not_passed")
        if audit["batch_row_count"] <= 0:
            reasons.append("stage21_3_batch_rows_missing")
        if audit["theta_batch_viewpoint_contract_missing_count"]:
            reasons.append("theta_batch_viewpoint_contract_missing")
        if audit["theta_batch_reward_contract_missing_count"]:
            reasons.append("theta_batch_reward_contract_missing")
        if not audit["stage21_3_rejects_point_only_reward_batch"]:
            reasons.append("stage21_3_point_only_reward_gate_missing")
    return reasons


def _summary_counts(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "trainable_transition_count": audit["transition_count"],
        "reward_evaluation_row_count": audit["reward_row_count"],
        "batch_row_count": audit["batch_row_count"],
        "theta_transition_contract_missing_count": audit["theta_transition_contract_missing_count"],
        "mask_length_mismatch_count": audit["mask_length_mismatch_count"],
        "sampling_mask_contract_mismatch_count": audit["sampling_mask_contract_mismatch_count"],
        "action_viewpoint_binding_mismatch_count": audit["action_viewpoint_binding_mismatch_count"],
        "theta_reward_contract_missing_count": audit["theta_reward_contract_missing_count"],
        "point_only_reward_fallback_used_count": audit["point_only_reward_fallback_used_count"],
        "reward_viewpoint_binding_mismatch_count": audit["reward_viewpoint_binding_mismatch_count"],
        "reward_footprint_binding_mismatch_count": audit["reward_footprint_binding_mismatch_count"],
        "theta_batch_viewpoint_contract_missing_count": audit["theta_batch_viewpoint_contract_missing_count"],
        "theta_batch_reward_contract_missing_count": audit["theta_batch_reward_contract_missing_count"],
        "hard_risk_violation_count": audit["hard_risk_violation_count"],
        "mask_violation_count": audit["mask_violation_count"],
        "path_planning_failure_count": audit["path_planning_failure_count"],
        "open_grid_fallback_count": audit["open_grid_fallback_count"],
        "stage21_3_rejects_point_only_reward_batch": audit["stage21_3_rejects_point_only_reward_batch"],
    }


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    summary_path = Path(config["stage22_2_root"]) / "xunce-stage22-2-summary.json"
    if not summary_path.is_file():
        return ["missing_stage22_2_summary"]
    summary = _read_json(summary_path)
    if summary.get("status") != "passed":
        reasons.append("stage22_2_not_passed")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE22_2:
        reasons.append("stage22_2_route_not_stage22_3")
    for key in (
        "theta_reward_contract_missing_count",
        "point_only_reward_fallback_used_count",
        "coverage_diff_but_reward_equal_group_count",
    ):
        if int(summary.get(key, 0) or 0) != 0:
            reasons.append(f"stage22_2_{key}")
    return reasons


def _read_stage22_2_summary(config: dict[str, Any]) -> dict[str, Any]:
    return _read_json(Path(config["stage22_2_root"]) / "xunce-stage22-2-summary.json")


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(config_path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {payload.get('schema_version')}")
    defaults = {
        "stage22_2_root": (
            "D:/CodexDownloads/lunar-path-planning/stage22_theta_aware_sensor_action_space/"
            "outputs/path_feedback_batch_xunce_stage22_2_theta_aware_coverage_reward_contract_v1"
        ),
        "stage21_1_base_config": "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json",
        "stage21_2_base_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "stage21_3_base_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_trainable_transition_count": 1,
        "theta_aware_candidate_viewpoints_enabled": True,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 2,
        "theta_coverage_denominator_cells": 1.0,
        "stage22_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for key in (
        "stage22_2_root",
        "stage21_1_base_config",
        "stage21_2_base_config",
        "stage21_3_base_config",
        "coverage_first_reward_profile",
    ):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    config["required_scenario_count"] = _positive_int(config["required_scenario_count"], "required_scenario_count")
    config["rollout_steps"] = _positive_int(config["rollout_steps"], "rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(config["dynamic_max_candidates_per_step"], "dynamic_max_candidates_per_step")
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(
        config["dynamic_proposal_pool_limit_per_step"],
        "dynamic_proposal_pool_limit_per_step",
    )
    config["min_trainable_transition_count"] = _positive_int(config["min_trainable_transition_count"], "min_trainable_transition_count")
    config["theta_bin_count"] = _positive_int(config["theta_bin_count"], "theta_bin_count")
    config["theta_step_deg"] = _positive_int(config["theta_step_deg"], "theta_step_deg")
    config["sensor_fov_deg"] = _positive_float(config["sensor_fov_deg"], "sensor_fov_deg")
    config["sensor_range_cells"] = _positive_int(config["sensor_range_cells"], "sensor_range_cells")
    config["theta_coverage_denominator_cells"] = _positive_float(
        config["theta_coverage_denominator_cells"],
        "theta_coverage_denominator_cells",
    )
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _theta_arrays_present(info: dict[str, Any]) -> bool:
    candidate_viewpoints = info.get("candidate_viewpoints")
    if not isinstance(candidate_viewpoints, list) or not candidate_viewpoints:
        return False
    expected = len(candidate_viewpoints)
    for key in (
        "candidate_theta_deg",
        "theta_new_visible_cell_counts",
        "theta_coverage_hashes",
        "theta_coverage_gain_per_path_costs",
    ):
        value = info.get(key)
        if not isinstance(value, list) or len(value) != expected:
            return False
    return bool(info.get("selected_viewpoint")) and info.get("selected_theta_deg") is not None


def _row_has_theta_reward_contract(row: dict[str, Any]) -> bool:
    denominator = _finite_float(row.get("theta_coverage_denominator_cells"))
    theta_new = _finite_float(row.get("theta_new_visible_cell_count"))
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    coverage_rate_delta = _finite_float(metrics.get("coverage_rate_delta"))
    return (
        row.get("theta_aware_reward_contract") is True
        and row.get("coverage_source") == THETA_COVERAGE_SOURCE
        and isinstance(row.get("candidate_viewpoint"), list)
        and len(row.get("candidate_viewpoint")) >= 3
        and row.get("theta_coverage_hash")
        and theta_new is not None
        and row.get("theta_coverage_gain_per_path_cost") is not None
        and denominator is not None
        and denominator > 0.0
        and coverage_rate_delta is not None
        and abs(coverage_rate_delta - (theta_new / denominator)) <= 1.0e-9
        and row.get("point_only_reward_fallback_used") is False
    )


def _selected_theta_footprint(info: dict[str, Any], action_index: int | None) -> dict[str, Any] | None:
    if action_index is None or action_index < 0:
        return None
    arrays = {
        "theta_new_visible_cell_count": info.get("theta_new_visible_cell_counts"),
        "theta_coverage_hash": info.get("theta_coverage_hashes"),
        "theta_coverage_gain_per_path_cost": info.get("theta_coverage_gain_per_path_costs"),
    }
    for values in arrays.values():
        if not isinstance(values, list) or action_index >= len(values):
            return None
    return {key: values[action_index] for key, values in arrays.items()}


def _mask_allows(info: dict[str, Any], mask_name: str, action_index: int | None) -> bool | None:
    if action_index is None or action_index < 0:
        return None
    mask = info.get(mask_name)
    if not isinstance(mask, list) or action_index >= len(mask):
        return None
    return mask[action_index] is True


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _theta_from_viewpoint(value: Any) -> int | None:
    if isinstance(value, list) and len(value) >= 3:
        return _int_or_none(value[2])
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage22.3 Theta-Aware PPO Collector Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- theta_transition_contract_missing_count: `{summary['theta_transition_contract_missing_count']}`",
            f"- theta_reward_contract_missing_count: `{summary['theta_reward_contract_missing_count']}`",
            f"- theta_batch_reward_contract_missing_count: `{summary['theta_batch_reward_contract_missing_count']}`",
            f"- action_viewpoint_binding_mismatch_count: `{summary['action_viewpoint_binding_mismatch_count']}`",
            "",
            "Stage22.3 only validates the real collector/reward/batch data path. It does not run PPO updates, publish checkpoints, replace default policy, connect executor, or start canary traffic.",
        ]
    ) + "\n"


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
