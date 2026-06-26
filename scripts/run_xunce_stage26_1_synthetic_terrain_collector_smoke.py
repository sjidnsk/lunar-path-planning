from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage21_1_on_policy_ppo_rollout_collector as stage21_1
    import run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import run_xunce_stage21_3_ppo_batch_validation as stage21_3
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as stage21_1
    import scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import scripts.run_xunce_stage21_3_ppo_batch_validation as stage21_3


STAGE_ID = "xunce-stage26-1-synthetic-terrain-collector-smoke"
CONFIG_SCHEMA_VERSION = "xunce-stage26-1-synthetic-terrain-collector-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-1-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage26-1-synthetic-transition-contract-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-1-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-1-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_1_synthetic_terrain_collector_smoke_v1"
)
DEFAULT_STAGE26_0_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1"
)

SUMMARY_FILE = "xunce-stage26-1-summary.json"
STAGE21_1_SUMMARY_FILE = "xunce-stage26-1-stage21-1-collector-summary.json"
STAGE21_2_SUMMARY_FILE = "xunce-stage26-1-stage21-2-reward-summary.json"
STAGE21_3_SUMMARY_FILE = "xunce-stage26-1-stage21-3-batch-summary.json"
AUDIT_FILE = "xunce-stage26-1-synthetic-transition-contract-audit.json"
ROUTING_FILE = "xunce-stage26-1-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-1-report.md"
MANIFEST_FILE = "xunce-stage26-1-manifest.json"

COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
SYNTHETIC_MODEL_ID = "synthetic_rock_pit_terrain/v1"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
ROUTE_FROM_STAGE26_0 = "run_stage26_1_synthetic_terrain_collector_smoke"

ROUTE_INPUTS = "rerun_stage26_1_required_inputs"
ROUTE_COLLECTOR = "repair_stage26_1_collector_synthetic_map_binding"
ROUTE_TRANSITION = "repair_stage26_1_synthetic_transition_contract"
ROUTE_REWARD = "repair_stage26_1_synthetic_reward_provenance"
ROUTE_BATCH = "repair_stage26_1_batch_gate"
ROUTE_SAFETY = "repair_stage26_1_synthetic_safety_regression"
ROUTE_STAGE26_2 = "run_stage26_2_synthetic_terrain_ppo_update_smoke"
ROUTE_BOUNDARY = "resolve_stage26_1_boundary_rejections"

BOUNDARY_FIELDS = (
    "stage26_1_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
EXPANSION_PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.1 synthetic terrain collector smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summary = run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_1_synthetic_terrain_collector_smoke(
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
    input_reasons, stage26_0_summary = _input_rejections(config)
    synthetic_source_root: Path | None = None
    if not boundary_reasons and not input_reasons:
        synthetic_source_root = _prepare_synthetic_source_root(config, stage26_0_summary, output_root)

    stage21_1_summary: dict[str, Any] = {}
    stage21_2_summary: dict[str, Any] = {}
    stage21_3_summary: dict[str, Any] = {}
    if synthetic_source_root is not None:
        stage21_1_summary = _run_stage21_1(config, stage26_0_summary, synthetic_source_root, output_root, repo_root)
        if stage21_1_summary.get("status") == "passed":
            stage21_2_summary = _run_stage21_2(config, output_root, repo_root)
        if stage21_2_summary.get("status") == "passed":
            stage21_3_summary = _run_stage21_3(config, output_root, repo_root)

    transitions = _read_jsonl_if_exists(output_root / "s21_1" / stage21_1.TRAINABLE_BATCH_FILE)
    rewards = _read_jsonl_if_exists(output_root / "s21_2" / stage21_2.EVALUATION_FILE)
    batch_rows = _read_jsonl_if_exists(output_root / "s21_3" / stage21_3.BATCH_FILE)
    rejections = _read_jsonl_if_exists(output_root / "s21_1" / stage21_1.REJECTION_FILE)
    manifest = _read_json_if_exists(output_root / "s21_1" / stage21_1.MANIFEST_FILE)
    audit = _contract_audit(
        transitions,
        rewards,
        batch_rows,
        rejections,
        expected_source_root=str(synthetic_source_root) if synthetic_source_root else None,
        stage21_1_manifest=manifest,
        expected_synthetic_hash=str(stage26_0_summary.get("synthetic_terrain_hash") or ""),
    )
    status, route, route_reason = _route(
        boundary_reasons,
        input_reasons,
        stage21_1_summary,
        stage21_2_summary,
        stage21_3_summary,
        audit,
    )
    blocking = _unique(boundary_reasons + input_reasons + _route_blockers(route, audit, stage21_1_summary, stage21_2_summary, stage21_3_summary))

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage26_0_root": config["stage26_0_root"],
        "stage26_0_status": stage26_0_summary.get("status"),
        "stage26_0_next_required_change": stage26_0_summary.get("next_required_change"),
        "synthetic_source_root": str(synthetic_source_root) if synthetic_source_root else None,
        "stage21_1_root": str(output_root / "s21_1"),
        "stage21_1_status": stage21_1_summary.get("status"),
        "stage21_2_root": str(output_root / "s21_2"),
        "stage21_2_status": stage21_2_summary.get("status"),
        "stage21_3_root": str(output_root / "s21_3"),
        "stage21_3_status": stage21_3_summary.get("status"),
        "synthetic_terrain_model_id": SYNTHETIC_MODEL_ID,
        "synthetic_terrain_hash": stage26_0_summary.get("synthetic_terrain_hash"),
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "required_scenario_count": int(config["required_scenario_count"]),
        "rollout_steps": int(config["rollout_steps"]),
        **_summary_counts(audit),
        "stage26_1_authorized": False,
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
        "stage26_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "stage21_1_summary": str(output_root / STAGE21_1_SUMMARY_FILE),
        "stage21_2_summary": str(output_root / STAGE21_2_SUMMARY_FILE),
        "stage21_3_summary": str(output_root / STAGE21_3_SUMMARY_FILE),
        "synthetic_transition_contract_audit": str(output_root / AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE21_1_SUMMARY_FILE, stage21_1_summary)
    _write_json(output_root / STAGE21_2_SUMMARY_FILE, stage21_2_summary)
    _write_json(output_root / STAGE21_3_SUMMARY_FILE, stage21_3_summary)
    _write_json(output_root / AUDIT_FILE, audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest_payload)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_stage21_1(
    config: dict[str, Any],
    stage26_0_summary: dict[str, Any],
    synthetic_source_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_1_base_config"], repo_root)
    cfg.update(
        {
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "dynamic_validation_work_root": str(output_root / "_dynamic_validation_work_stage26_1"),
            "source_roi_expansion_root": str(synthetic_source_root),
            "theta_aware_candidate_viewpoints_enabled": True,
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": config["sensor_model_id"],
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "slope_obstacle_aware_theta_reward_enabled": True,
            "hybrid_astar_pose_path_cost_enabled": True,
            "hybrid_astar_goal_position_tolerance_m": float(config["hybrid_astar_goal_position_tolerance_m"]),
            "hybrid_astar_goal_theta_tolerance_deg": float(config["hybrid_astar_goal_theta_tolerance_deg"]),
            "hybrid_astar_primitive_duration_s": float(config["hybrid_astar_primitive_duration_s"]),
            "hybrid_astar_integration_dt_s": float(config["hybrid_astar_integration_dt_s"]),
            "hybrid_astar_max_speed_mps": float(config["hybrid_astar_max_speed_mps"]),
            "hybrid_astar_max_angular_speed_degps": float(config["hybrid_astar_max_angular_speed_degps"]),
            "hybrid_astar_max_iterations": int(config["hybrid_astar_max_iterations"]),
            "synthetic_terrain_contract_enabled": True,
            "synthetic_terrain_model_id": SYNTHETIC_MODEL_ID,
            "synthetic_terrain_hash": stage26_0_summary.get("synthetic_terrain_hash"),
            "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
            "obstacle_occlusion_enabled": True,
            "derive_slope_blocked_cells_from_sidecar_dem": True,
            "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
            "platform_contract_id": config.get("platform_contract_id"),
            "platform_contract_hash": config.get("platform_contract_hash"),
            "platform_max_climb_deg": float(config["max_traversable_slope_deg"]),
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
    cfg_path = output_root / "xunce-stage26-1-stage21-1-config.json"
    _write_json(cfg_path, cfg)
    return stage21_1.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=cfg_path,
        output_root=output_root / "s21_1",
        repo_root=repo_root,
    )


def _run_stage21_2(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_2_base_config"], repo_root)
    cfg.update(
        {
            "stage21_1_collector_root": str(output_root / "s21_1"),
            "coverage_first_reward_profile": config["coverage_first_reward_profile"],
            "require_theta_aware_reward_contract": True,
            "slope_obstacle_aware_theta_reward_enabled": True,
            "require_hybrid_astar_path_cost_contract": True,
            "require_synthetic_terrain_contract": True,
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
    cfg_path = output_root / "xunce-stage26-1-stage21-2-config.json"
    _write_json(cfg_path, cfg)
    return stage21_2.run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=cfg_path,
        output_root=output_root / "s21_2",
        repo_root=repo_root,
    )


def _run_stage21_3(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage21_3_base_config"], repo_root)
    cfg.update(
        {
            "stage21_1_collector_root": str(output_root / "s21_1"),
            "stage21_2_reward_contract_root": str(output_root / "s21_2"),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "require_theta_aware_viewpoint_contract": True,
            "require_theta_aware_reward_contract": True,
            "require_slope_obstacle_aware_theta_reward_contract": True,
            "require_hybrid_astar_path_cost_contract": True,
            "require_synthetic_terrain_contract": True,
            "stage21_3_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    cfg_path = output_root / "xunce-stage26-1-stage21-3-config.json"
    _write_json(cfg_path, cfg)
    return stage21_3.run_xunce_stage21_3_ppo_batch_validation(
        config_path=cfg_path,
        output_root=output_root / "s21_3",
        repo_root=repo_root,
    )


def _prepare_synthetic_source_root(config: dict[str, Any], summary: dict[str, Any], output_root: Path) -> Path:
    root = output_root / "src"
    sidecar_dir = root / "sc"
    root.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecars = [Path(path) for path in summary.get("augmented_sidecar_paths", []) if isinstance(path, str)]
    if not sidecars:
        raise ConfigError("Stage26.0 summary does not list augmented_sidecar_paths")
    map_audit = _read_json_if_exists(Path(config["stage26_0_root"]) / "xunce-stage26-0-map-augmentation-audit.json")
    source_audits = map_audit.get("sidecar_audits") if isinstance(map_audit.get("sidecar_audits"), list) else []
    first_audit = next((row for row in source_audits if isinstance(row, dict)), {})
    if not config.get("platform_contract_hash") and first_audit.get("platform_contract_hash"):
        config["platform_contract_hash"] = str(first_audit["platform_contract_hash"])
    if not config.get("platform_contract_id"):
        platform_id = first_audit.get("platform_contract_id") or summary.get("platform_contract_id") or "agilex_scout_mini_piper"
        config["platform_contract_id"] = str(platform_id)
    rows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    required = int(config["required_scenario_count"])
    for index in range(required):
        src = sidecars[index % len(sidecars)]
        if not src.is_file():
            raise ConfigError(f"augmented sidecar does not exist: {src}")
        sidecar = _read_json(src)
        sidecar["max_traversable_slope_deg"] = float(config["max_traversable_slope_deg"])
        sidecar["synthetic_terrain_hash"] = summary.get("synthetic_terrain_hash")
        sidecar["synthetic_source_kind"] = SYNTHETIC_SOURCE_KIND
        sidecar["synthetic_terrain_model_id"] = SYNTHETIC_MODEL_ID
        source_audit = source_audits[index] if index < len(source_audits) and isinstance(source_audits[index], dict) else first_audit
        platform_hash = source_audit.get("platform_contract_hash") or config.get("platform_contract_hash")
        platform_id = source_audit.get("platform_contract_id") or config.get("platform_contract_id")
        if platform_hash:
            sidecar["platform_contract_hash"] = str(platform_hash)
        if platform_id:
            sidecar["platform_contract_id"] = str(platform_id)
        dst_sidecar = sidecar_dir / f"s26_1_{index:03d}.sidecar.json"
        _write_json(dst_sidecar, sidecar)
        source_sidecar = None
        if source_audit:
            source_sidecar = source_audit.get("source_sidecar")
        if source_sidecar is None and source_audits and isinstance(source_audits[0], dict):
            source_sidecar = source_audits[0].get("source_sidecar")
        src_contract = _contract_path_for_source_sidecar(source_sidecar)
        dst_contract = sidecar_dir / f"s26_1_{index:03d}.contract.json"
        if src_contract is not None and src_contract.is_file():
            shutil.copy2(src_contract, dst_contract)
        else:
            raise ConfigError(f"source path-planner contract not found for synthetic sidecar: {source_sidecar}")
        scenario_id = f"stage26_synthetic_{index:03d}"
        start_cell = _safe_start_cell(sidecar)
        common = {
            "schema_version": "xunce-high-fidelity-real-map-slice/v1",
            "scenario_id": scenario_id,
            "slice_id": scenario_id,
            "scenario_seed": 260100 + index,
            "scenario_group": "stage26_synthetic_terrain_smoke",
            "scenario_variant_id": f"{scenario_id}-seed-{260100 + index}-start-{start_cell[0]}-{start_cell[1]}",
            "split": "train",
            "start_cell": start_cell,
            "current_cell": start_cell,
            "sidecar": str(dst_sidecar),
            "contract": str(dst_contract),
            "synthetic_terrain_model_id": SYNTHETIC_MODEL_ID,
            "synthetic_terrain_hash": summary.get("synthetic_terrain_hash"),
            "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
            "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
            "platform_contract_id": platform_id,
            "platform_contract_hash": platform_hash,
            "coverage_source": COVERAGE_SOURCE,
            "path_cost_source": PATH_COST_SOURCE,
        }
        rows.append(common)
        scenarios.append({k: v for k, v in common.items() if k not in {"schema_version", "slice_id", "split"}})
    _write_json(root / EXPANSION_SUMMARY_FILE, {
        "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
        "status": "passed",
        "scenario_count": len(scenarios),
        "source_kind": SYNTHETIC_SOURCE_KIND,
        "synthetic_terrain_hash": summary.get("synthetic_terrain_hash"),
        "source_stage26_0_root": config["stage26_0_root"],
    })
    _write_jsonl(root / EXPANSION_SLICES_FILE, rows)
    _write_jsonl(root / "xunce-high-res-terrain-slices.jsonl", rows)
    _write_json(root / EXPANSION_PATH_FEEDBACK_AUDIT_FILE, {
        "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    })
    return root


def _safe_start_cell(sidecar: dict[str, Any]) -> list[int]:
    cost = sidecar.get("cost")
    height = len(cost) if isinstance(cost, list) else 0
    width = len(cost[0]) if height and isinstance(cost[0], list) else 0
    passable = sidecar.get("passable_mask")
    blocked = _cells(sidecar.get("synthetic_hard_obstacle_cells")) | _cells(sidecar.get("slope_blocked_cells")) | _cells(sidecar.get("blocked_cells"))
    for y in range(height):
        row = passable[y] if isinstance(passable, list) and y < len(passable) and isinstance(passable[y], list) else []
        for x in range(width):
            if row and x < len(row) and not bool(row[x]):
                continue
            if (x, y) not in blocked:
                return [x, y]
    return [0, 0]


def _contract_path_for_source_sidecar(source_sidecar: Any) -> Path | None:
    if not isinstance(source_sidecar, str) or not source_sidecar.strip():
        return None
    path = Path(source_sidecar)
    name = path.name
    if name.endswith(".path-planner-sidecar.json"):
        return path.with_name(name[: -len(".path-planner-sidecar.json")] + ".contract.json")
    return path.with_suffix(".contract.json")


def _contract_audit(
    transitions: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    *,
    expected_source_root: str | None,
    stage21_1_manifest: dict[str, Any],
    expected_synthetic_hash: str,
) -> dict[str, Any]:
    transition_ids = {str(row.get("transition_id")) for row in transitions if str(row.get("transition_id") or "").strip()}
    reward_ids = {str(row.get("transition_id")) for row in rewards if str(row.get("transition_id") or "").strip()}
    batch_ids = {str(row.get("transition_id")) for row in batch_rows if str(row.get("transition_id") or "").strip()}
    actual_source_root = _manifest_source_roi_expansion_root(stage21_1_manifest)
    transition_missing = sum(1 for row in transitions if not _row_has_synthetic_info(row, expected_synthetic_hash))
    reward_missing = sum(1 for row in rewards if not _row_has_synthetic_reward(row, expected_synthetic_hash))
    batch_missing = stage21_3._synthetic_terrain_contract_missing_count(batch_rows) if batch_rows else 0
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "transition_count": len(transitions),
        "reward_row_count": len(rewards),
        "batch_row_count": len(batch_rows),
        "transition_id_count": len(transition_ids),
        "reward_transition_id_count": len(reward_ids),
        "batch_transition_id_count": len(batch_ids),
        "transition_missing_reward_count": len(transition_ids - reward_ids),
        "reward_transition_id_not_in_stage21_1_count": len(reward_ids - transition_ids),
        "batch_transition_id_not_in_stage21_1_count": len(batch_ids - transition_ids),
        "batch_transition_id_not_in_stage21_2_count": len(batch_ids - reward_ids),
        "transition_missing_batch_count": len(transition_ids - batch_ids),
        "reward_missing_batch_count": len(reward_ids - batch_ids),
        "stage21_1_expected_source_roi_expansion_root": expected_source_root,
        "stage21_1_actual_source_roi_expansion_root": actual_source_root,
        "stage21_1_source_roi_expansion_root_match": _same_path(expected_source_root, actual_source_root),
        "synthetic_transition_contract_missing_count": transition_missing,
        "synthetic_reward_provenance_missing_count": reward_missing,
        "synthetic_batch_contract_missing_count": batch_missing,
        "synthetic_physical_obstacle_pollution_count": sum(
            1 for row in [*transitions, *rewards, *batch_rows] if _row_info(row).get("physical_obstacle_cells_written") is True or row.get("physical_obstacle_cells_written") is True
        ),
        "point_only_reward_fallback_used_count": sum(1 for row in rewards if row.get("point_only_reward_fallback_used") is True),
        "unobstructed_theta_reward_fallback_used_count": sum(1 for row in rewards if row.get("unobstructed_theta_reward_fallback_used") is True),
        "point_grid_path_cost_fallback_used_count": sum(1 for row in rewards if row.get("point_grid_path_cost_fallback_used") is True),
        "hard_risk_violation_count": sum(1 for row in transitions if _row_info(row).get("hard_risk_violation") is True),
        "mask_violation_count": _mask_violation_count(transitions),
        "path_planning_failure_count": sum(1 for row in rejections if "path_planning" in str(row.get("reason") or row.get("reason_code") or "")),
        "open_grid_fallback_count": sum(1 for row in transitions if _row_info(row).get("open_grid_fallback_used") is True),
        "stage21_3_rejects_missing_synthetic_contract": stage21_3._synthetic_terrain_contract_missing_count([{"reward": 0.0}]) == 1,
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
        return "failed", ROUTE_BOUNDARY, "stage26_1_boundary_rejected"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage26_1_inputs_missing_or_untrusted"
    if stage21_1_summary.get("status") != "passed" or audit["stage21_1_source_roi_expansion_root_match"] is not True:
        return "failed", ROUTE_COLLECTOR, "stage21_1_did_not_load_synthetic_augmented_sidecar"
    if audit["transition_count"] <= 0 or audit["synthetic_transition_contract_missing_count"] > 0:
        return "failed", ROUTE_TRANSITION, "stage21_1_synthetic_transition_contract_missing"
    if any(audit[key] > 0 for key in ("hard_risk_violation_count", "mask_violation_count", "path_planning_failure_count", "open_grid_fallback_count")):
        return "failed", ROUTE_SAFETY, "synthetic_terrain_collector_safety_regression"
    if (
        stage21_2_summary.get("status") != "passed"
        or audit["reward_row_count"] <= 0
        or audit["transition_missing_reward_count"] > 0
        or audit["reward_transition_id_not_in_stage21_1_count"] > 0
        or audit["synthetic_reward_provenance_missing_count"] > 0
        or audit["point_only_reward_fallback_used_count"] > 0
        or audit["unobstructed_theta_reward_fallback_used_count"] > 0
        or audit["point_grid_path_cost_fallback_used_count"] > 0
    ):
        return "failed", ROUTE_REWARD, "stage21_2_synthetic_reward_provenance_missing"
    if (
        stage21_3_summary.get("status") != "passed"
        or audit["batch_row_count"] <= 0
        or audit["batch_transition_id_not_in_stage21_1_count"] > 0
        or audit["batch_transition_id_not_in_stage21_2_count"] > 0
        or audit["transition_missing_batch_count"] > 0
        or audit["reward_missing_batch_count"] > 0
        or audit["synthetic_batch_contract_missing_count"] > 0
        or not audit["stage21_3_rejects_missing_synthetic_contract"]
    ):
        return "failed", ROUTE_BATCH, "stage21_3_synthetic_batch_gate_missing"
    if audit["synthetic_physical_obstacle_pollution_count"] > 0:
        return "failed", ROUTE_BATCH, "synthetic_terrain_was_written_as_physical_obstacle"
    return "passed", ROUTE_STAGE26_2, "synthetic_terrain_collector_reward_batch_ready"


def _route_blockers(
    route: str,
    audit: dict[str, Any],
    stage21_1_summary: dict[str, Any],
    stage21_2_summary: dict[str, Any],
    stage21_3_summary: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if route == ROUTE_COLLECTOR and stage21_1_summary.get("status") != "passed":
        reasons.append("stage21_1_not_passed")
    for key, name in (
        ("stage21_1_source_roi_expansion_root_match", "stage21_1_source_roi_expansion_root_mismatch"),
        ("stage21_3_rejects_missing_synthetic_contract", "stage21_3_missing_synthetic_gate"),
    ):
        if key in audit and audit[key] is not True:
            reasons.append(name)
    for key in (
        "synthetic_transition_contract_missing_count",
        "synthetic_reward_provenance_missing_count",
        "synthetic_batch_contract_missing_count",
        "synthetic_physical_obstacle_pollution_count",
        "point_only_reward_fallback_used_count",
        "unobstructed_theta_reward_fallback_used_count",
        "point_grid_path_cost_fallback_used_count",
        "hard_risk_violation_count",
        "mask_violation_count",
        "path_planning_failure_count",
        "open_grid_fallback_count",
    ):
        if audit.get(key, 0):
            reasons.append(key.removesuffix("_count"))
    if route == ROUTE_REWARD and stage21_2_summary.get("status") != "passed":
        reasons.append("stage21_2_not_passed")
    if route == ROUTE_BATCH and stage21_3_summary.get("status") != "passed":
        reasons.append("stage21_3_not_passed")
    return _unique(reasons)


def _summary_counts(audit: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "transition_count",
        "reward_row_count",
        "batch_row_count",
        "transition_missing_reward_count",
        "reward_transition_id_not_in_stage21_1_count",
        "batch_transition_id_not_in_stage21_1_count",
        "batch_transition_id_not_in_stage21_2_count",
        "transition_missing_batch_count",
        "reward_missing_batch_count",
        "synthetic_transition_contract_missing_count",
        "synthetic_reward_provenance_missing_count",
        "synthetic_batch_contract_missing_count",
        "synthetic_physical_obstacle_pollution_count",
        "point_only_reward_fallback_used_count",
        "unobstructed_theta_reward_fallback_used_count",
        "point_grid_path_cost_fallback_used_count",
        "hard_risk_violation_count",
        "mask_violation_count",
        "path_planning_failure_count",
        "open_grid_fallback_count",
    )
    return {key: audit.get(key) for key in keys} | {
        "stage21_1_source_roi_expansion_root_match": audit.get("stage21_1_source_roi_expansion_root_match"),
        "stage21_3_rejects_missing_synthetic_contract": audit.get("stage21_3_rejects_missing_synthetic_contract"),
    }


def _row_has_synthetic_info(row: dict[str, Any], expected_hash: str) -> bool:
    info = _row_info(row)
    return (
        info.get("synthetic_terrain_model_id") == SYNTHETIC_MODEL_ID
        and info.get("synthetic_terrain_hash") == expected_hash
        and info.get("synthetic_source_kind") == SYNTHETIC_SOURCE_KIND
        and info.get("synthetic_hard_obstacle_cells_used") is True
        and info.get("synthetic_los_blocker_cells_used") is True
        and info.get("synthetic_high_risk_cells_available") is True
        and info.get("physical_obstacle_cells_written") is False
        and isinstance(info.get("effective_hard_obstacle_source"), list)
        and "synthetic_hard_obstacle_cells" in info.get("effective_hard_obstacle_source")
        and isinstance(info.get("effective_los_blocker_source"), list)
        and "synthetic_los_blocker_cells" in info.get("effective_los_blocker_source")
    )


def _row_has_synthetic_reward(row: dict[str, Any], expected_hash: str) -> bool:
    return (
        row.get("synthetic_terrain_reward_provenance") is True
        and row.get("synthetic_terrain_hash") == expected_hash
        and row.get("synthetic_source_kind") == SYNTHETIC_SOURCE_KIND
        and row.get("synthetic_los_blocker_cells_used") is True
        and row.get("point_only_reward_fallback_used") is False
        and row.get("unobstructed_theta_reward_fallback_used") is False
        and row.get("point_grid_path_cost_fallback_used") is False
    )


def _mask_violation_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        info = _row_info(row)
        action_index = _int_or_none(row.get("action_index"))
        for key in ("action_mask", "sampling_mask", "hard_risk_clean_mask"):
            mask = info.get(key)
            if action_index is None or not isinstance(mask, list) or action_index < 0 or action_index >= len(mask) or mask[action_index] is not True:
                count += 1
    return count


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config.setdefault("stage26_0_root", DEFAULT_STAGE26_0_ROOT)
    for key in ("stage26_0_root", "stage21_1_base_config", "stage21_2_base_config", "stage21_3_base_config", "coverage_first_reward_profile"):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ConfigError(f"{key} must be a non-empty string")
        config[key] = str(_resolve_path(Path(config[key]), repo_root))
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 2), "required_scenario_count")
    config["rollout_steps"] = _positive_int(config.get("rollout_steps", 4), "rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(config.get("dynamic_max_candidates_per_step", 36), "dynamic_max_candidates_per_step")
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(config.get("dynamic_proposal_pool_limit_per_step", 288), "dynamic_proposal_pool_limit_per_step")
    config["min_trainable_transition_count"] = _positive_int(config.get("min_trainable_transition_count", 1), "min_trainable_transition_count")
    config["theta_bin_count"] = _positive_int(config.get("theta_bin_count", 8), "theta_bin_count")
    config["theta_step_deg"] = _positive_int(config.get("theta_step_deg", 45), "theta_step_deg")
    config["sensor_model_id"] = str(config.get("sensor_model_id") or "theta-fov-90-range-radius/v1")
    config["sensor_fov_deg"] = _positive_float(config.get("sensor_fov_deg", 90.0), "sensor_fov_deg")
    config["sensor_range_cells"] = _positive_int(config.get("sensor_range_cells", 3), "sensor_range_cells")
    config["theta_coverage_denominator_cells"] = _positive_float(config.get("theta_coverage_denominator_cells", 100.0), "theta_coverage_denominator_cells")
    config["max_traversable_slope_deg"] = _positive_float(config.get("max_traversable_slope_deg", 30.0), "max_traversable_slope_deg")
    config["hybrid_astar_goal_position_tolerance_m"] = _positive_float(
        config.get("hybrid_astar_goal_position_tolerance_m", 4.0),
        "hybrid_astar_goal_position_tolerance_m",
    )
    config["hybrid_astar_goal_theta_tolerance_deg"] = _positive_float(
        config.get("hybrid_astar_goal_theta_tolerance_deg", 10.0),
        "hybrid_astar_goal_theta_tolerance_deg",
    )
    config["hybrid_astar_primitive_duration_s"] = _positive_float(
        config.get("hybrid_astar_primitive_duration_s", 4.0),
        "hybrid_astar_primitive_duration_s",
    )
    config["hybrid_astar_integration_dt_s"] = _positive_float(
        config.get("hybrid_astar_integration_dt_s", 0.25),
        "hybrid_astar_integration_dt_s",
    )
    config["hybrid_astar_max_speed_mps"] = _positive_float(
        config.get("hybrid_astar_max_speed_mps", 1.0),
        "hybrid_astar_max_speed_mps",
    )
    config["hybrid_astar_max_angular_speed_degps"] = _positive_float(
        config.get("hybrid_astar_max_angular_speed_degps", 45.0),
        "hybrid_astar_max_angular_speed_degps",
    )
    config["hybrid_astar_max_iterations"] = _positive_int(
        config.get("hybrid_astar_max_iterations", 200000),
        "hybrid_astar_max_iterations",
    )
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _input_rejections(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    root = Path(config["stage26_0_root"])
    summary_path = root / "xunce-stage26-0-summary.json"
    reasons: list[str] = []
    summary: dict[str, Any] = {}
    if not summary_path.is_file():
        reasons.append("missing_stage26_0_summary")
    else:
        summary = _read_json(summary_path)
        if summary.get("status") != "passed":
            reasons.append("stage26_0_not_passed")
        if summary.get("next_required_change") != ROUTE_FROM_STAGE26_0:
            reasons.append("stage26_0_route_not_stage26_1")
        if summary.get("synthetic_terrain_model_id") != SYNTHETIC_MODEL_ID:
            reasons.append("stage26_0_synthetic_model_id_mismatch")
        if summary.get("source_kind") != SYNTHETIC_SOURCE_KIND:
            reasons.append("stage26_0_synthetic_source_kind_mismatch")
        if not summary.get("synthetic_terrain_hash"):
            reasons.append("stage26_0_synthetic_hash_missing")
        if summary.get("physical_obstacle_cells_written") is not False:
            reasons.append("stage26_0_physical_obstacle_pollution")
        if summary.get("coverage_source") != COVERAGE_SOURCE:
            reasons.append("stage26_0_coverage_source_mismatch")
        if summary.get("path_cost_source") != PATH_COST_SOURCE:
            reasons.append("stage26_0_path_cost_source_mismatch")
    return reasons, summary


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _manifest_source_roi_expansion_root(manifest: dict[str, Any]) -> str | None:
    audit = manifest.get("model_audit") if isinstance(manifest.get("model_audit"), dict) else {}
    value = audit.get("source_roi_expansion_root")
    return str(value) if isinstance(value, str) and value.strip() else None


def _same_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def _row_info(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("info") if isinstance(row.get("info"), dict) else {}


def _cells(value: Any) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    if not isinstance(value, list):
        return result
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                result.add((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
    return result


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, field: str) -> int:
    parsed = _int_or_none(value)
    if parsed is None or parsed <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a positive number") from exc
    if parsed <= 0.0:
        raise ConfigError(f"{field} must be a positive number")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a non-negative number") from exc
    if parsed < 0.0:
        raise ConfigError(f"{field} must be a non-negative number")
    return parsed


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _load_json_template(path: str, repo_root: Path) -> dict[str, Any]:
    return _read_json(_resolve_path(Path(path), repo_root))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.1 Synthetic Terrain Collector Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- transition_count: `{summary['transition_count']}`",
            f"- synthetic_transition_contract_missing_count: `{summary['synthetic_transition_contract_missing_count']}`",
            f"- synthetic_reward_provenance_missing_count: `{summary['synthetic_reward_provenance_missing_count']}`",
            f"- synthetic_batch_contract_missing_count: `{summary['synthetic_batch_contract_missing_count']}`",
            "",
            "Stage26.1 only verifies the collector, reward, and batch contract for synthetic terrain proxy data. It does not run PPO, publish checkpoints, replace policy, connect an executor, or start canary traffic.",
        ]
    ) + "\n"


class ConfigError(ValueError):
    pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
