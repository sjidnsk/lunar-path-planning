from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4
    import run_xunce_stage21_3_ppo_batch_validation as stage21_3
    import run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as stage24_4
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4
    import scripts.run_xunce_stage21_3_ppo_batch_validation as stage21_3
    import scripts.run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke as stage24_4


CONFIG_SCHEMA_VERSION = "xunce-stage26-2-synthetic-terrain-ppo-update-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-2-summary/v1"
BATCH_AUDIT_SCHEMA_VERSION = "xunce-stage26-2-synthetic-ppo-batch-update-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-2-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-2-manifest/v1"

STAGE_ID = "xunce-stage26-2-synthetic-terrain-ppo-update-smoke"
DEFAULT_CONFIG = "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1"
)
DEFAULT_STAGE26_1_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_1_synthetic_terrain_collector_smoke_v1"
)

SUMMARY_FILE = "xunce-stage26-2-summary.json"
STAGE21_4_CONFIG_FILE = "xunce-stage26-2-stage21-4-config.json"
STAGE21_4_SUMMARY_FILE = "xunce-stage26-2-stage21-4-summary.json"
BATCH_AUDIT_FILE = "xunce-stage26-2-synthetic-ppo-batch-update-audit.json"
LOSS_GRADIENT_AUDIT_FILE = "xunce-stage26-2-loss-gradient-audit.json"
CHECKPOINT_BOUNDARY_AUDIT_FILE = "xunce-stage26-2-checkpoint-boundary-audit.json"
ROUTING_FILE = "xunce-stage26-2-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-2-report.md"
MANIFEST_FILE = "xunce-stage26-2-manifest.json"

SYNTHETIC_MODEL_ID = "synthetic_rock_pit_terrain/v1"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"

ROUTE_INPUTS = "rerun_stage26_2_required_inputs"
ROUTE_BATCH = "repair_stage26_2_synthetic_batch_update_contract"
ROUTE_STABILITY = "repair_stage26_2_synthetic_ppo_update_stability"
ROUTE_CHECKPOINT = "repair_stage26_2_checkpoint_reload_boundary"
ROUTE_STAGE26_3 = "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke"
ROUTE_BOUNDARY = "resolve_stage26_2_boundary_rejections"
ROUTE_FROM_STAGE26_1 = "run_stage26_2_synthetic_terrain_ppo_update_smoke"
ROUTE_FROM_STAGE21_4 = "implement_stage21_5_single_seed_ppo_pilot"

BOUNDARY_FALSE_FIELDS = (
    "stage26_2_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.2 synthetic terrain PPO update smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
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
    stage26_1_summary = _read_json_if_exists(Path(config["stage26_1_root"]) / "xunce-stage26-1-summary.json")
    batch_rows = _read_jsonl_if_exists(Path(config["stage26_1_root"]) / "s21_3" / stage21_3.BATCH_FILE)
    batch_audit = _synthetic_batch_update_audit(batch_rows, expected_summary=stage26_1_summary)
    checkpoint_lineage = _stage26_1_collector_checkpoint_lineage(config)
    upstream_physical_payload_audit = _stage26_1_upstream_physical_payload_audit(config)

    stage21_4_summary: dict[str, Any] = {}
    stage21_4_config_path = output_root / STAGE21_4_CONFIG_FILE
    if not boundary_reasons and not input_reasons and _batch_contract_passed(batch_audit):
        stage21_4_summary = _run_stage21_4(config, output_root, repo_root, stage21_4_config_path)

    stage21_4_root = output_root / "s21_4"
    loss_rows = _read_jsonl_if_exists(stage21_4_root / stage21_4.LOSS_AUDIT_FILE)
    gradient_audit = _read_json_if_exists(stage21_4_root / stage21_4.GRADIENT_AUDIT_FILE)
    checkpoint_audit_raw = _read_json_if_exists(stage21_4_root / stage21_4.CHECKPOINT_AUDIT_FILE)
    loss_gradient_audit = stage24_4._loss_gradient_audit(loss_rows, gradient_audit, stage21_4_summary)
    checkpoint_boundary_audit = stage24_4._checkpoint_boundary_audit(
        checkpoint_audit_raw,
        expected_stage21_3_root=Path(config["stage26_1_root"]) / "s21_3",
        expected_stage21_4_root=stage21_4_root,
        stage21_4_summary=stage21_4_summary,
    )

    status, route, route_reason = _route(
        boundary_reasons,
        input_reasons,
        batch_audit,
        stage21_4_summary,
        loss_gradient_audit,
        checkpoint_boundary_audit,
    )
    blocking = _unique(
        boundary_reasons
        + input_reasons
        + _route_blocking_reasons(route, batch_audit, stage21_4_summary, loss_gradient_audit, checkpoint_boundary_audit)
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage26_1_root": config["stage26_1_root"],
        "stage26_1_status": stage26_1_summary.get("status"),
        "stage26_1_next_required_change": stage26_1_summary.get("next_required_change"),
        "stage26_1_collector_checkpoint_sha256": checkpoint_lineage.get("stage26_1_collector_checkpoint_sha256"),
        "stage26_2_source_checkpoint_sha256": checkpoint_lineage.get("stage26_2_source_checkpoint_sha256"),
        "collector_source_checkpoint_match": checkpoint_lineage.get("collector_source_checkpoint_match"),
        "stage26_1_upstream_physical_obstacle_payload_count": upstream_physical_payload_audit["physical_obstacle_payload_count"],
        "stage21_4_root": str(stage21_4_root),
        "stage21_4_status": stage21_4_summary.get("status"),
        "stage21_4_next_required_change": stage21_4_summary.get("next_required_change"),
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_terrain_model_id": stage26_1_summary.get("synthetic_terrain_model_id"),
        "synthetic_terrain_hash": stage26_1_summary.get("synthetic_terrain_hash"),
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "max_traversable_slope_deg": 30.0,
        **_summary_counts(batch_audit, loss_gradient_audit, checkpoint_boundary_audit),
        "stage26_2_authorized": False,
        "runs_new_ppo_update": bool(status == "passed"),
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
        "stage26_2_authorized": False,
        "runs_new_ppo_update": bool(status == "passed"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "stage21_4_config": str(stage21_4_config_path),
        "stage21_4_summary": str(output_root / STAGE21_4_SUMMARY_FILE),
        "synthetic_ppo_batch_update_audit": str(output_root / BATCH_AUDIT_FILE),
        "loss_gradient_audit": str(output_root / LOSS_GRADIENT_AUDIT_FILE),
        "checkpoint_boundary_audit": str(output_root / CHECKPOINT_BOUNDARY_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE21_4_SUMMARY_FILE, stage21_4_summary)
    _write_json(output_root / BATCH_AUDIT_FILE, batch_audit)
    _write_json(output_root / LOSS_GRADIENT_AUDIT_FILE, loss_gradient_audit)
    _write_json(output_root / CHECKPOINT_BOUNDARY_AUDIT_FILE, checkpoint_boundary_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_stage21_4(config: dict[str, Any], output_root: Path, repo_root: Path, config_path: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_4_base_config"], repo_root)
    cfg.update(
        {
            "stage21_3_ppo_batch_validation_root": str(Path(config["stage26_1_root"]) / "s21_3"),
            "xunce_candidate_checkpoint": config["xunce_candidate_checkpoint"],
            "high_fidelity_config": config["high_fidelity_config"],
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "clip_ratio": float(config["clip_ratio"]),
            "policy_loss_coefficient": float(config["policy_loss_coefficient"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "entropy_coefficient": float(config["entropy_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
            "loss_scale": float(config["loss_scale"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "offline_ppo_update_smoke_authorized": True,
            "stage21_4_authorized": False,
            "stage26_2_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    _write_json(config_path, cfg)
    return stage21_4.run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config_path,
        output_root=output_root / "s21_4",
        repo_root=repo_root,
    )


def _synthetic_batch_update_audit(rows: list[dict[str, Any]], *, expected_summary: dict[str, Any]) -> dict[str, Any]:
    hybrid_audit = stage24_4._hybrid_path_batch_update_audit(rows)
    synthetic_missing = stage21_3._synthetic_terrain_contract_missing_count(rows)
    synthetic_reward_missing = 0
    synthetic_info_missing = 0
    synthetic_physical_pollution = 0
    synthetic_physical_payload = 0
    effective_hard_missing = 0
    effective_los_missing = 0
    coverage_source_mismatch = 0
    path_cost_source_mismatch = 0
    expected_hash = expected_summary.get("synthetic_terrain_hash")
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        if not _row_has_synthetic_reward_provenance(row, expected_hash=expected_hash):
            synthetic_reward_missing += 1
        if not _info_has_synthetic_provenance(info, expected_hash=expected_hash):
            synthetic_info_missing += 1
        if row.get("physical_obstacle_cells_written") is True or info.get("physical_obstacle_cells_written") is True:
            synthetic_physical_pollution += 1
        if _physical_obstacle_payload_present(row) or _physical_obstacle_payload_present(info):
            synthetic_physical_payload += 1
        if "synthetic_hard_obstacle_cells" not in _as_list(row.get("effective_hard_obstacle_source") or info.get("effective_hard_obstacle_source")):
            effective_hard_missing += 1
        if "synthetic_los_blocker_cells" not in _as_list(row.get("effective_los_blocker_source") or info.get("effective_los_blocker_source")):
            effective_los_missing += 1
        if row.get("coverage_source") != COVERAGE_SOURCE:
            coverage_source_mismatch += 1
        if row.get("path_cost_source") != PATH_COST_SOURCE:
            path_cost_source_mismatch += 1
    return {
        "schema_version": BATCH_AUDIT_SCHEMA_VERSION,
        "batch_row_count": len(rows),
        "hybrid_path_audit": hybrid_audit,
        "synthetic_terrain_contract_missing_count": synthetic_missing,
        "synthetic_terrain_reward_provenance_missing_count": synthetic_reward_missing,
        "synthetic_info_provenance_missing_count": synthetic_info_missing,
        "synthetic_physical_obstacle_pollution_count": synthetic_physical_pollution,
        "synthetic_physical_obstacle_payload_count": synthetic_physical_payload,
        "effective_hard_obstacle_source_missing_synthetic_count": effective_hard_missing,
        "effective_los_blocker_source_missing_synthetic_count": effective_los_missing,
        "coverage_source_mismatch_count": coverage_source_mismatch,
        "path_cost_source_mismatch_count": path_cost_source_mismatch,
    }


def _row_has_synthetic_reward_provenance(row: dict[str, Any], *, expected_hash: Any) -> bool:
    return (
        row.get("synthetic_terrain_reward_provenance") is True
        and row.get("synthetic_terrain_model_id") == SYNTHETIC_MODEL_ID
        and row.get("synthetic_terrain_hash") == expected_hash
        and row.get("synthetic_source_kind") == SYNTHETIC_SOURCE_KIND
        and row.get("synthetic_hard_obstacle_cells_used") is True
        and row.get("synthetic_los_blocker_cells_used") is True
        and row.get("synthetic_high_risk_cells_available") is True
        and row.get("physical_obstacle_cells_written") is False
    )


def _info_has_synthetic_provenance(info: dict[str, Any], *, expected_hash: Any) -> bool:
    return (
        info.get("synthetic_terrain_model_id") == SYNTHETIC_MODEL_ID
        and info.get("synthetic_terrain_hash") == expected_hash
        and info.get("synthetic_source_kind") == SYNTHETIC_SOURCE_KIND
        and info.get("synthetic_hard_obstacle_cells_used") is True
        and info.get("synthetic_los_blocker_cells_used") is True
        and info.get("synthetic_high_risk_cells_available") is True
        and info.get("physical_obstacle_cells_written") is False
    )


def _physical_obstacle_payload_present(row: dict[str, Any]) -> bool:
    for key in ("physical_obstacle_cells", "physical_obstacle_rectangles", "physical_obstacle_polygons"):
        if _non_empty_payload(row.get(key)):
            return True
    return False


def _non_empty_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _batch_contract_passed(audit: dict[str, Any]) -> bool:
    hybrid = audit["hybrid_path_audit"]
    return (
        stage24_4._batch_contract_passed(hybrid)
        and audit["batch_row_count"] > 0
        and audit["synthetic_terrain_contract_missing_count"] == 0
        and audit["synthetic_terrain_reward_provenance_missing_count"] == 0
        and audit["synthetic_info_provenance_missing_count"] == 0
        and audit["synthetic_physical_obstacle_pollution_count"] == 0
        and audit["synthetic_physical_obstacle_payload_count"] == 0
        and audit["effective_hard_obstacle_source_missing_synthetic_count"] == 0
        and audit["effective_los_blocker_source_missing_synthetic_count"] == 0
        and audit["coverage_source_mismatch_count"] == 0
        and audit["path_cost_source_mismatch_count"] == 0
    )


def _route(
    boundary_reasons: list[str],
    input_reasons: list[str],
    batch_audit: dict[str, Any],
    stage21_4_summary: dict[str, Any],
    loss_gradient_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage26_2_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage26_2_inputs_missing_or_untrusted"
    if not _batch_contract_passed(batch_audit):
        return "failed", ROUTE_BATCH, "synthetic_batch_update_contract_failed"
    if (
        stage21_4_summary.get("status") != "passed"
        or stage21_4_summary.get("next_required_change") != ROUTE_FROM_STAGE21_4
        or not loss_gradient_audit["loss_finite"]
        or not loss_gradient_audit["gradient_finite"]
        or not loss_gradient_audit["component_grad_norms_readable"]
    ):
        return "failed", ROUTE_STABILITY, "synthetic_ppo_update_stability_failed"
    if not checkpoint_audit["checkpoint_boundary_passed"]:
        return "failed", ROUTE_CHECKPOINT, "synthetic_checkpoint_reload_boundary_failed"
    return "passed", ROUTE_STAGE26_3, "synthetic_terrain_ppo_update_smoke_ready"


def _route_blocking_reasons(
    route: str,
    batch_audit: dict[str, Any],
    stage21_4_summary: dict[str, Any],
    loss_gradient_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_BATCH:
        if batch_audit["batch_row_count"] <= 0:
            reasons.append("synthetic_ppo_batch_rows_missing")
        for key in (
            "synthetic_terrain_contract_missing_count",
            "synthetic_terrain_reward_provenance_missing_count",
            "synthetic_info_provenance_missing_count",
            "synthetic_physical_obstacle_pollution_count",
            "synthetic_physical_obstacle_payload_count",
            "effective_hard_obstacle_source_missing_synthetic_count",
            "effective_los_blocker_source_missing_synthetic_count",
            "coverage_source_mismatch_count",
            "path_cost_source_mismatch_count",
        ):
            if batch_audit[key]:
                reasons.append(key.removesuffix("_count"))
        for key, value in batch_audit["hybrid_path_audit"].items():
            if key.endswith("_count") and value:
                reasons.append(key.removesuffix("_count"))
    if route == ROUTE_STABILITY:
        if stage21_4_summary.get("status") != "passed":
            reasons.append("stage21_4_not_passed")
        if stage21_4_summary.get("next_required_change") != ROUTE_FROM_STAGE21_4:
            reasons.append("stage21_4_route_not_stage21_5")
        for key in ("loss_finite", "gradient_finite", "component_grad_norms_readable"):
            if not loss_gradient_audit[key]:
                reasons.append(key)
    if route == ROUTE_CHECKPOINT:
        for key in (
            "checkpoint_exists",
            "checkpoint_reload_passed",
            "experimental_only",
            "stage21_3_lineage_passed",
            "checkpoint_path_in_stage21_4_root",
            "source_checkpoint_sha_consistent",
            "checkpoint_boundary_passed",
        ):
            if not checkpoint_audit[key]:
                reasons.append(key)
    return reasons


def _summary_counts(batch_audit: dict[str, Any], loss_gradient_audit: dict[str, Any], checkpoint_audit: dict[str, Any]) -> dict[str, Any]:
    hybrid = batch_audit["hybrid_path_audit"]
    return {
        "batch_row_count": batch_audit["batch_row_count"],
        "synthetic_terrain_contract_missing_count": batch_audit["synthetic_terrain_contract_missing_count"],
        "synthetic_terrain_reward_provenance_missing_count": batch_audit["synthetic_terrain_reward_provenance_missing_count"],
        "synthetic_info_provenance_missing_count": batch_audit["synthetic_info_provenance_missing_count"],
        "synthetic_physical_obstacle_pollution_count": batch_audit["synthetic_physical_obstacle_pollution_count"],
        "synthetic_physical_obstacle_payload_count": batch_audit["synthetic_physical_obstacle_payload_count"],
        "effective_hard_obstacle_source_missing_synthetic_count": batch_audit["effective_hard_obstacle_source_missing_synthetic_count"],
        "effective_los_blocker_source_missing_synthetic_count": batch_audit["effective_los_blocker_source_missing_synthetic_count"],
        "coverage_source_mismatch_count": batch_audit["coverage_source_mismatch_count"],
        "path_cost_source_mismatch_count": batch_audit["path_cost_source_mismatch_count"],
        "hybrid_path_batch_contract_missing_count": hybrid["hybrid_path_batch_contract_missing_count"],
        "xunce_batch_viewpoint_shape_mismatch_count": hybrid["xunce_batch_viewpoint_shape_mismatch_count"],
        "action_viewpoint_path_cost_binding_mismatch_count": hybrid["action_viewpoint_path_cost_binding_mismatch_count"],
        "selected_action_mask_violation_count": hybrid["selected_action_mask_violation_count"],
        "point_grid_path_cost_fallback_used_count": hybrid["point_grid_path_cost_fallback_used_count"],
        "default_astar_replaced_count": hybrid["default_astar_replaced_count"],
        "ackermann_feasible_claimed_count": hybrid["ackermann_feasible_claimed_count"],
        "loss_row_count": loss_gradient_audit["loss_row_count"],
        "loss_finite": loss_gradient_audit["loss_finite"],
        "gradient_finite": loss_gradient_audit["gradient_finite"],
        "component_grad_norms_readable": loss_gradient_audit["component_grad_norms_readable"],
        "final_total_loss": loss_gradient_audit["final_total_loss"],
        "final_post_update_approx_kl": loss_gradient_audit["final_post_update_approx_kl"],
        "pre_clip_grad_norm": loss_gradient_audit["pre_clip_grad_norm"],
        "post_clip_grad_norm": loss_gradient_audit["post_clip_grad_norm"],
        "parameter_delta_l2": loss_gradient_audit["parameter_delta_l2"],
        "parameter_with_grad_count": loss_gradient_audit["parameter_with_grad_count"],
        "checkpoint_exists": checkpoint_audit["checkpoint_exists"],
        "checkpoint_reload_passed": checkpoint_audit["checkpoint_reload_passed"],
        "experimental_only": checkpoint_audit["experimental_only"],
        "experimental_checkpoint_sha256": checkpoint_audit["experimental_checkpoint_sha256"],
        "stage21_3_lineage_passed": checkpoint_audit["stage21_3_lineage_passed"],
        "checkpoint_path_in_stage21_4_root": checkpoint_audit["checkpoint_path_in_stage21_4_root"],
        "source_checkpoint_sha_consistent": checkpoint_audit["source_checkpoint_sha_consistent"],
        "checkpoint_boundary_passed": checkpoint_audit["checkpoint_boundary_passed"],
    }


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = Path(config["stage26_1_root"])
    summary_path = root / "xunce-stage26-1-summary.json"
    if not summary_path.is_file():
        return ["missing_stage26_1_summary"]
    summary = _read_json(summary_path)
    expected_zero = (
        "synthetic_transition_contract_missing_count",
        "synthetic_reward_provenance_missing_count",
        "synthetic_batch_contract_missing_count",
        "synthetic_physical_obstacle_pollution_count",
        "point_grid_path_cost_fallback_used_count",
        "point_only_reward_fallback_used_count",
        "unobstructed_theta_reward_fallback_used_count",
        "hard_risk_violation_count",
        "mask_violation_count",
        "path_planning_failure_count",
        "open_grid_fallback_count",
    )
    if summary.get("status") != "passed":
        reasons.append("stage26_1_not_passed")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE26_1:
        reasons.append("stage26_1_route_not_stage26_2")
    if summary.get("coverage_source") != COVERAGE_SOURCE:
        reasons.append("stage26_1_coverage_source_mismatch")
    if summary.get("path_cost_source") != PATH_COST_SOURCE:
        reasons.append("stage26_1_path_cost_source_mismatch")
    if summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_1_synthetic_source_kind_mismatch")
    if summary.get("synthetic_terrain_model_id") != SYNTHETIC_MODEL_ID:
        reasons.append("stage26_1_synthetic_model_mismatch")
    for key in expected_zero:
        if int(summary.get(key, 0) or 0) != 0:
            reasons.append(f"stage26_1_{key}_nonzero")
    s21_3_root = root / "s21_3"
    for name in (
        stage21_3.SUMMARY_FILE,
        "xunce-stage21-3-lineage-audit.json",
        stage21_3.BATCH_FILE,
    ):
        if not (s21_3_root / name).is_file():
            reasons.append(f"missing_stage26_1_s21_3_{name}")
    checkpoint_lineage = _stage26_1_collector_checkpoint_lineage(config)
    if not checkpoint_lineage["stage26_1_collector_manifest_exists"]:
        reasons.append("missing_stage26_1_s21_1_manifest")
    if not checkpoint_lineage["stage26_1_collector_checkpoint_sha256"]:
        reasons.append("missing_stage26_1_collector_checkpoint_sha256")
    if not checkpoint_lineage["stage26_2_source_checkpoint_sha256"]:
        reasons.append("missing_stage26_2_source_checkpoint_sha256")
    if checkpoint_lineage["stage26_1_collector_checkpoint_sha256"] and checkpoint_lineage["stage26_2_source_checkpoint_sha256"]:
        if not checkpoint_lineage["collector_source_checkpoint_match"]:
            reasons.append("stage26_2_source_checkpoint_mismatch_stage26_1_collector")
    upstream_physical_payload_audit = _stage26_1_upstream_physical_payload_audit(config)
    for missing_path in upstream_physical_payload_audit["missing_files"]:
        reasons.append(f"missing_stage26_1_upstream_{Path(missing_path).name}")
    if upstream_physical_payload_audit["physical_obstacle_payload_count"] > 0:
        reasons.append("stage26_1_upstream_physical_obstacle_payload_present")
    s21_3_summary = _read_json_if_exists(s21_3_root / stage21_3.SUMMARY_FILE)
    if s21_3_summary.get("status") != "passed":
        reasons.append("stage26_1_stage21_3_not_passed")
    if not Path(config["xunce_candidate_checkpoint"]).is_file():
        reasons.append("missing_xunce_candidate_checkpoint")
    if not Path(config["high_fidelity_config"]).is_file():
        reasons.append("missing_high_fidelity_config")
    return reasons


def _stage26_1_upstream_physical_payload_audit(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["stage26_1_root"])
    paths = (
        root / "s21_1" / "xunce-stage21-1-ppo-trainable-batch.jsonl",
        root / "s21_1" / "xunce-stage21-1-ppo-rollout-transitions.jsonl",
        root / "s21_1" / "xunce-stage21-1-reward-audit.jsonl",
        root / "s21_2" / "xunce-stage21-2-reward-contract-evaluation.jsonl",
    )
    payload_count = 0
    checked_rows = 0
    checked_files: list[str] = []
    missing_files: list[str] = []
    for path in paths:
        if not path.is_file():
            missing_files.append(str(path))
            continue
        rows = _read_jsonl_if_exists(path)
        checked_files.append(str(path))
        for row in rows:
            checked_rows += 1
            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            if _physical_obstacle_payload_present(row) or _physical_obstacle_payload_present(info):
                payload_count += 1
    return {
        "checked_files": checked_files,
        "missing_files": missing_files,
        "checked_row_count": checked_rows,
        "physical_obstacle_payload_count": payload_count,
    }


def _stage26_1_collector_checkpoint_lineage(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(config["stage26_1_root"]) / "s21_1" / "xunce-stage21-1-manifest.json"
    manifest = _read_json_if_exists(manifest_path)
    model_audit = manifest.get("model_audit") if isinstance(manifest.get("model_audit"), dict) else {}
    checkpoint_audit = model_audit.get("xunce_checkpoint_audit") if isinstance(model_audit.get("xunce_checkpoint_audit"), dict) else {}
    collector_sha = checkpoint_audit.get("checkpoint_sha256")
    source_sha = _sha256_file(Path(config["xunce_candidate_checkpoint"]))
    return {
        "stage26_1_collector_manifest_exists": manifest_path.is_file(),
        "stage26_1_collector_checkpoint_path": checkpoint_audit.get("checkpoint_path") or model_audit.get("xunce_candidate_checkpoint"),
        "stage26_1_collector_checkpoint_sha256": collector_sha,
        "stage26_2_source_checkpoint_path": config.get("xunce_candidate_checkpoint"),
        "stage26_2_source_checkpoint_sha256": source_sha,
        "collector_source_checkpoint_match": bool(collector_sha and source_sha and collector_sha == source_sha),
    }


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FALSE_FIELDS if config.get(field) is True]
    if config.get("runs_new_ppo_update") is not True:
        reasons.append("runs_new_ppo_update_not_enabled_for_stage26_2_smoke")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    defaults = {
        "stage26_1_root": DEFAULT_STAGE26_1_ROOT,
        "stage21_4_base_config": "configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json",
        "xunce_candidate_checkpoint": "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/sandbox_package/xunce-controlled-training-candidate.pt",
        "high_fidelity_config": "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
        "epochs": 1,
        "learning_rate": 2.0e-6,
        "clip_ratio": 0.2,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.1,
        "entropy_coefficient": 0.01,
        "advantage_clip_abs": 5.0,
        "normalize_minibatch_advantages": True,
        "loss_scale": 0.25,
        "max_grad_norm": 1.0,
        "max_abs_approx_kl": 0.5,
        "stage26_2_authorized": False,
        "runs_new_ppo_update": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for field in ("stage26_1_root", "stage21_4_base_config", "xunce_candidate_checkpoint", "high_fidelity_config"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["epochs"] = int(config["epochs"])
    for key in (
        "learning_rate",
        "clip_ratio",
        "policy_loss_coefficient",
        "value_loss_coefficient",
        "entropy_coefficient",
        "advantage_clip_abs",
        "loss_scale",
        "max_grad_norm",
        "max_abs_approx_kl",
        "canary_traffic_fraction",
    ):
        config[key] = float(config[key])
    config["normalize_minibatch_advantages"] = bool(config["normalize_minibatch_advantages"])
    return config


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


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
            rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.2 Synthetic Terrain PPO Update Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- stage21_4_status: `{summary.get('stage21_4_status')}`",
            f"- batch_row_count: `{summary.get('batch_row_count')}`",
            f"- synthetic_terrain_hash: `{summary.get('synthetic_terrain_hash')}`",
            f"- loss_finite: `{summary.get('loss_finite')}`",
            f"- gradient_finite: `{summary.get('gradient_finite')}`",
            f"- checkpoint_reload_passed: `{summary.get('checkpoint_reload_passed')}`",
            "",
            "This stage only proves that synthetic terrain PPO batches can be consumed by the offline tiny PPO update.",
            "It does not publish checkpoints, replace the default policy, connect an executor, start canary traffic, or claim performance.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
